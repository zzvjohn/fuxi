"""Read-only status or completed-only resume of a verified frozen campaign.

Incomplete campaigns cannot use this wrapper to start workers.  The native worker
launcher is not a cross-version frozen child bootstrap.  Each completed resume
must preserve all existing attempts, research facts, and rewards.
"""
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.abc
import importlib.util
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import struct
import sys
from types import MappingProxyType

WORKSPACE=Path(__file__).resolve().parents[1]
REF_PATTERN=re.compile(rb'\{"dtype":("(?:[^"\\]|\\.)*"),"path":("(?:[^"\\]|\\.)*"),"sha256":"([0-9a-f]{64})","shape":(\[[0-9, ]+\])\}')
# Frozen native artifacts use sorted canonical JSON.  A bounded metadata scan
# avoids materializing each 70 MiB evaluation/IR result just to bind its grids.
GRID_PATTERN=re.compile(rb'\{"end":[0-9]+,"instruments":\[[^\]]*\],"membership":\{[^{}]*\},"reference_hash":"[0-9a-f]{64}","schema_version":"fuxi\.score_grid_ref\.1","sessions":\[[^\]]*\],"start":[0-9]+,"values":\{[^{}]*\}\}')


def require(condition,message):
    if not condition:raise ValueError(message)


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def strict_json(path):
    def pairs(rows):
        result={}
        for key,value in rows:
            require(key not in result,'duplicate JSON key: '+key);result[key]=value
        return result
    def constant(value):raise ValueError('nonfinite JSON constant: '+value)
    with Path(path).open(encoding='utf8') as stream:
        return json.load(stream,object_pairs_hook=pairs,parse_constant=constant)


def file_hash(path,*,canonical_newline=False):
    path=Path(path);length=path.stat().st_size
    with path.open('rb') as stream:
        if canonical_newline:
            require(length>0,'empty canonical input')
            stream.seek(length-1);require(stream.read(1)==b'\n','frozen input must retain its canonical LF terminator')
            stream.seek(0);length-=1
        content=hashlib.sha256()
        while length:
            block=stream.read(min(length,4*1024*1024));require(block,'source shortened while hashing')
            content.update(block);length-=len(block)
    return content.hexdigest()


def no_redirect(path):
    path=Path(path)
    require(path.is_absolute() and path.is_relative_to(WORKSPACE),'frozen asset must remain in the workspace')
    for item in (path,*path.parents):
        if item==WORKSPACE.parent:break
        require(not item.is_symlink() and not item.is_junction(),'frozen asset redirection forbidden: '+str(item))
    require(path.resolve()==path,'noncanonical frozen path')
    return path


def inside(root,relative):
    root=no_redirect(Path(root));path=Path(relative)
    require(not path.is_absolute() and not path.drive,'artifact path must be relative')
    require(all(part not in ('.','..') for part in path.parts),'artifact path escaped its campaign')
    target=root/path;require(target.is_relative_to(root),'artifact path escaped its campaign')
    # Check the lexical path before resolve can hide an internal redirect.
    return no_redirect(target)


def verify_code(directory,expected):
    directory=no_redirect(directory)
    require(directory.is_dir() and expected,'frozen code directory/identity missing')
    entries=list(directory.iterdir())
    require(all(p.is_file() and p.suffix=='.py' and p.stem.isidentifier()
                and not p.is_symlink() and not p.is_junction() for p in entries),
            'unregistered frozen code asset; .pyc, directories and redirects are refused')
    actual={p.name:file_hash(p) for p in entries}
    require(actual==expected,'frozen module set or source content changed')
    return actual


@contextmanager
def readonly_database(directory):
    path=no_redirect(directory/'evidence.sqlite');require(path.is_file(),'campaign evidence missing')
    connection=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=30)
    connection.row_factory=sqlite3.Row
    try:
        connection.execute('PRAGMA query_only=ON')
        yield connection
    finally:connection.close()


def row_fingerprint(connection):
    # Hashes identify immutable event payloads.  This guard checks that resume adds
    # or revises no research ledger row; it is not an independent economic audit.
    facts=[tuple(r) for r in connection.execute('SELECT id,kind,hash FROM facts ORDER BY id')]
    attempts=[tuple(r) for r in connection.execute('SELECT id,campaign,experiment,state,reserved,actual,started,ended,result FROM attempts ORDER BY id')]
    rewards=[tuple(r) for r in connection.execute('SELECT * FROM rewards ORDER BY slot,revision')]
    return {'fact_count':len(facts),'attempt_count':len(attempts),'reward_count':len(rewards),
            'fingerprint':digest([facts,attempts,rewards])}


def array_references(path):
    """Scan the canonical four-key reference objects without loading large IR JSON."""
    carry=b''
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            block=carry+block
            for match in REF_PATTERN.finditer(block):
                yield json.loads(match.group().decode('utf8'))
            carry=block[-4096:]


def verify_array(reference,checked):
    require(set(reference)=={'path','sha256','dtype','shape'},'array descriptor keys differ')
    path=no_redirect(Path(reference['path']))
    require(path.is_file() and re.fullmatch(r'[|<>][bifu][0-9]+',reference['dtype']), 'unsafe array descriptor')
    require(isinstance(reference['shape'],list) and reference['shape']
            and all(type(n) is int and n>0 for n in reference['shape']),'invalid array shape')
    identity=(reference['sha256'],reference['dtype'],tuple(reference['shape']))
    if str(path) in checked:
        require(checked[str(path)]==identity,'conflicting array descriptors');return
    require(file_hash(path)==reference['sha256'],'array source or sidecar changed: '+str(path))
    with path.open('rb') as stream:
        require(stream.read(6)==b'\x93NUMPY','not a NumPy array')
        version=tuple(stream.read(2));require(version in {(1,0),(2,0),(3,0)},'unsupported NumPy header')
        size=struct.unpack('<H' if version==(1,0) else '<I',stream.read(2 if version==(1,0) else 4))[0]
        require(size<=65536,'array header exceeds contract')
        header=ast.literal_eval(stream.read(size).decode('utf8' if version==(3,0) else 'latin1'))
        require(header['descr']==reference['dtype'] and tuple(header['shape'])==tuple(reference['shape']),
                'array header/descriptor mismatch')
    checked[str(path)]=identity


def native_score_references(path):
    carry=b'';seen=set()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            block=carry+block
            for match in GRID_PATTERN.finditer(block):
                row=json.loads(match.group().decode('utf8'));identity=row['reference_hash']
                if identity in seen:continue
                require(set(row)=={'schema_version','sessions','instruments','start','end','values','membership','reference_hash'},
                        'native score descriptor keys differ')
                require(digest({k:v for k,v in row.items() if k!='reference_hash'})==identity,'native score binding changed')
                seen.add(identity);yield row
            # Full-A 5,670-name metadata is under 100 KiB.  Refuse an unbound
            # portable grid if a larger future descriptor exceeds this scope.
            carry=block[-512*1024:]


def verify_portable_scores(path,native_paths,checked):
    path=no_redirect(Path(path));packed=strict_json(path)
    require(set(packed)=={'schema_version','sessions','instruments','start','end','values','membership',
                         'reference_hash','native_transport_hash'},'portable score descriptor keys differ')
    require(packed['schema_version']=='fuxi.score_grid_bundle.1','portable score schema mismatch')
    require(digest({k:v for k,v in packed.items() if k!='reference_hash'})==packed['reference_hash'],
            'portable score binding changed')
    native_hash=packed['native_transport_hash']
    require(isinstance(native_hash,str) and re.fullmatch('[0-9a-f]{64}',native_hash),'invalid native transport hash')
    native=None
    for source in native_paths:
        for reference in native_score_references(source):
            if reference['reference_hash']==native_hash:
                require(native is None or native==reference,'conflicting native score bindings');native=reference
    require(native is not None,'portable score native transport binding missing from its worker')
    expected=dict(native,schema_version='fuxi.score_grid_bundle.1',native_transport_hash=native_hash)
    for field in ('values','membership'):
        # Paths are local to the published bundle, while every other descriptor
        # field and the full section axes must match the exact native transport.
        require(isinstance(packed[field],dict) and set(packed[field])=={'path','sha256','dtype','shape'},
                'portable array descriptor keys differ')
        target=inside(path.parent,packed[field]['path'])
        expected[field]=dict(native[field],path=packed[field]['path'])
        verify_array(native[field],checked)
        verify_array(dict(packed[field],path=str(target)),checked)
    expected['reference_hash']=digest({k:v for k,v in expected.items() if k!='reference_hash'})
    require(expected==packed,'portable score differs from its native transport')


def verify_bundle(root):
    manifest=strict_json(root/'manifest.json')
    require(manifest.get('schema_version')=='fuxi.run.1' and manifest.get('complete') is True
            and manifest['run_id']==root.name,'published bundle identity mismatch')
    require(digest({k:v for k,v in manifest.items() if k!='manifest_hash'})==manifest['manifest_hash'],'bundle manifest changed')
    found={str(p.relative_to(root)).replace('\\','/') for p in root.rglob('*') if p.is_file()}
    require(found==set(manifest['files'])|{'manifest.json'},'bundle file set changed')
    for name,expected in manifest['files'].items():
        p=PurePosixPath(name)
        require(not p.is_absolute() and all(v not in ('','.','..') and ':' not in v for v in p.parts),'unsafe published artifact name')
        path=inside(root,name)
        require(file_hash(path)==expected,'published artifact changed: '+name)


def inspect_campaign(directory):
    directory=no_redirect(Path(directory).absolute())
    require(directory.is_dir() and directory.name.startswith('campaign_'),'campaign directory required')
    config=strict_json(directory/'config.json')
    require(config.get('schema_version')=='fuxi.campaign.1' and config['campaign_id']==directory.name,
            'campaign/config identity mismatch')
    with readonly_database(directory) as connection:
        registered=list(connection.execute('SELECT id,config_hash,config,status FROM campaigns'))
        require(len(registered)==1 and registered[0]['id']==config['campaign_id'],'cross-campaign ledger refused')
        row=registered[0]
        require(json.loads(row['config'])==config and row['config_hash']==digest(config),'config file/ledger binding changed')
        anchor=connection.execute('SELECT payload,hash FROM facts WHERE id=? AND kind=?',
                                  (config['campaign_id']+':config','campaign_config')).fetchone()
        require(anchor is not None and json.loads(anchor['payload'])==config and anchor['hash']==digest(config),'campaign config anchor changed')
        rounds=[]
        for event in connection.execute('SELECT id,payload,hash FROM facts WHERE kind=? ORDER BY rowid',('round_completed',)):
            payload=json.loads(event['payload']);index=payload['round']
            require(event['hash']==digest(payload) and event['id']==config['campaign_id']+':round:'+str(index), 'round identity or hash changed')
            require(type(index) is int and 0<=index<config['rounds'],'round outside frozen budget')
            rounds.append(index)
        require(len(rounds)==len(set(rounds)),'duplicate completed rounds')
        attempts=[dict(r) for r in connection.execute('SELECT id,state,result FROM attempts WHERE campaign=? ORDER BY rowid',(config['campaign_id'],))]
        candidate_rows=[json.loads(r['payload']) for r in connection.execute('SELECT payload FROM facts WHERE kind=?',('candidate_result',))]
        fingerprint=row_fingerprint(connection)
    modules=verify_code(directory/'code',config['code_hashes'])
    input_path=no_redirect(directory/'input.json')
    require(file_hash(input_path,canonical_newline=True)==config['input_hash'],'frozen input bytes changed')
    checked={};scan=[input_path];bundles=set();model_artifacts=0;portable_bundles=0
    for attempt in attempts:
        if attempt['state']=='running':continue  # active pending writes are not committed evidence
        attempt_root=inside(directory,'attempts/'+attempt['id'])
        native_paths=[]
        for name in ('request.json','result.json'):
            if (attempt_root/name).is_file():
                scan.append(attempt_root/name);native_paths.append(attempt_root/name)
        summary=json.loads(attempt['result']) if attempt['result'] else {}
        if summary.get('bundle_path'):
            bundle=inside(directory.parent,summary['bundle_path']);require(bundle.is_relative_to(directory),'bundle belongs to another campaign')
            if bundle not in bundles:
                verify_bundle(bundle);bundles.add(bundle)
                if (bundle/'scores.json').is_file():
                    scores_path=bundle/'scores.json'
                    scores=strict_json(scores_path)
                    if isinstance(scores,dict) and scores.get('schema_version')=='fuxi.score_grid_bundle.1':
                        verify_portable_scores(scores_path,native_paths,checked);portable_bundles+=1
                    else:scan.append(scores_path)
        if summary.get('artifact_relative_path'):
            artifact=inside(directory,summary['artifact_relative_path'])
            require(file_hash(artifact)==summary['artifact_sha256'],'frozen model artifact changed')
            scan.append(artifact);model_artifacts+=1
    for path in scan:
        for reference in array_references(path):verify_array(reference,checked)
    states=Counter(a['state'] for a in attempts)
    completed=set(rounds)==set(range(config['rounds'])) and not states['running']
    status={'schema_version':'fuxi.frozen_campaign_status.1','campaign_id':config['campaign_id'],
        'campaign_directory':str(directory),'config_hash':digest(config),'input_hash':config['input_hash'],
        'code_module_count':len(modules),'code_hashes':modules,'rounds_completed':len(rounds),'rounds_planned':config['rounds'],
        'completed_round_indices':sorted(rounds),'state':'completed' if completed else 'incomplete',
        'ledger_status':row['status'],'attempt_states':dict(states),
        'candidate_result_events':len(candidate_rows),'unique_candidate_count':len({c['candidate_id'] for c in candidate_rows}),
        'engineering_error_events':sum(c.get('eval_status')=='error' for c in candidate_rows),
        'jq_candidate_events':sum(c.get('jq_candidate_allowed') is True for c in candidate_rows),
        'verified_array_source_and_sidecar_files':len(checked),'verified_published_bundles':len(bundles),
        'verified_model_artifacts':model_artifacts,'verified_portable_score_bundles':portable_bundles,'ledger_rows':fingerprint,
        'status_is_read_only':True,'resume_scope':'completed_only; no new workers; incomplete cross-version resume refused'}
    return config,status


class VerifiedSourceLoader(importlib.abc.Loader):
    def __init__(self,path,expected):self.path,self.expected=path,expected
    def create_module(self,spec):return None
    def get_code(self,fullname):
        no_redirect(self.path);raw=self.path.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==self.expected[self.path.name],'frozen source drift at consumption')
        return compile(raw,str(self.path),'exec',dont_inherit=True)
    def get_filename(self,fullname):return str(self.path)
    def is_package(self,fullname):return self.path.name=='__init__.py'
    def exec_module(self,module):exec(self.get_code(module.__name__),module.__dict__)


class FrozenFinder(importlib.abc.MetaPathFinder):
    def __init__(self,source,expected):self.source,self.expected=source,MappingProxyType(dict(expected))
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='fuxi_v1':name='__init__.py'
        elif fullname.startswith('fuxi_v1.') and fullname.count('.')==1:name=fullname.split('.')[-1]+'.py'
        elif fullname.startswith('fuxi_v1.'):raise ImportError('unregistered frozen submodule: '+fullname)
        else:return None
        if name not in self.expected:raise ImportError('unregistered frozen module: '+fullname)
        return importlib.util.spec_from_file_location(fullname,self.source/name,
            loader=VerifiedSourceLoader(self.source/name,self.expected),
            submodule_search_locations=[str(self.source)] if name=='__init__.py' else None)


def install_package(directory,config):
    require(not any(n=='fuxi_v1' or n.startswith('fuxi_v1.') for n in sys.modules),'current core already loaded; use a fresh interpreter')
    verify_code(directory/'code',config['code_hashes'])
    finder=FrozenFinder(directory/'code',config['code_hashes']);sys.meta_path.insert(0,finder)
    sys.dont_write_bytecode=True
    spec=finder.find_spec('fuxi_v1');module=importlib.util.module_from_spec(spec)
    sys.modules['fuxi_v1']=module;spec.loader.exec_module(module)
    return finder


def resume_call_arguments(directory,config):
    return (directory/'input.json',directory.parent),{
        'rounds':config['rounds'],'candidates_per_round':config['candidates_per_round'],
        'mode':config['mode'],'seed':config['seed'],'llm_enabled':config['llm_enabled'],
        'model_algorithms':tuple(config['model_algorithms']),'campaign_id':config['campaign_id'],
        'authority_path':Path(config['authority_path']),'study_protocol':config.get('study_protocol'),
        'data_receipt_id':config.get('data_receipt_id'),'scheduler_policy':config.get('scheduler_policy','legacy')}


def resume_completed(directory,config,before):
    require(before['state']=='completed','Resume is completed-only: unfinished rounds or running attempts require the verified current native entry; this wrapper cannot start workers')
    install_package(directory,config)
    from fuxi_v1.campaign import environment, run_campaign
    require(environment()==config['environment'],'research runtime/package versions changed across frozen resume')
    from fuxi_v1.runtime import controlled_writer
    positional,keywords=resume_call_arguments(directory,config)
    # Validate the actual verified native callable before acquiring a writer.
    # Package imports, config kwargs and environment belong to this same freeze.
    inspect.signature(run_campaign).bind(*positional,**keywords)
    flags={'FUXI_V1_PARENT_POLICY':config.get('parent_policy','pareto'),
           'FUXI_V1_MOTIF_MEMORY':config.get('motif_memory','off'),
           'FUXI_V1_FITNESS_TRIAGE':config.get('fitness_triage','off')}
    previous={key:os.environ.get(key) for key in flags}
    try:
        os.environ.update(flags)
        with controlled_writer(directory.parent):
            current_config,current=inspect_campaign(directory)
            require(current_config==config and current['ledger_rows']==before['ledger_rows'],'campaign changed before locked resume')
            path,report=run_campaign(*positional,**keywords)
            path=no_redirect(Path(path))
            require(path.parent==directory and path.name=='report_'+digest(report)[:16]+'.json',
                    'native completed report escaped its campaign or identity')
            require(report['campaign_id']==config['campaign_id'] and report['code_hashes']==config['code_hashes']
                    and report['input_hash']==config['input_hash'],'native completed report changed frozen bindings')
            require(file_hash(path,canonical_newline=True)==digest(report),'native completed report bytes changed')
            with readonly_database(directory) as connection:after=row_fingerprint(connection)
            require(after==before['ledger_rows'],'completed resume changed attempts, research facts or rewards')
            require(report['status']=='completed' and len(report['rounds'])==config['rounds'],'completed resume changed completion state')
            return {'campaign_id':config['campaign_id'],'report_path':str(path),'rounds_completed':len(report['rounds']),
                'frozen_code_module_count':len(config['code_hashes']),'ledger_rows_unchanged':True,'new_worker_attempts':0,
                'research_facts_and_rewards_unchanged':True,'scope':'verified original code; completed-only native resume'}
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('status','resume'),default='status')
    parser.add_argument('--campaign',required=True,help='exact existing campaign directory')
    args=parser.parse_args();directory=Path(args.campaign).absolute()
    config,status=inspect_campaign(directory)
    result=status if args.action=='status' else resume_completed(directory,config,status)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'state':'refused','error_type':type(exc).__name__,'reason':str(exc)},ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
