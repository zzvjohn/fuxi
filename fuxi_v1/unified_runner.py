"""One bounded manual/automatic workflow, with source-keyed idempotency.

This controller may write research evidence. It cannot admit candidates, submit JQ,
change the judge, or publish a strategy. Legacy producers require explicit manual jobs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import uuid
from .budgets import budget_units as _budget_units
from .contracts import digest, load_json, require, canonical
from .governance import governed_writer, child_environment, find_workspace
from .runtime import authority_location
from .artifacts import code_manifest
from .effective_config import collect_effective_config

SCHEMA='fuxi.unified_run.1'
DEFAULT_LIMITS={'max_rounds':40,'max_candidates':80,'max_worker_seconds':600,
                'max_total_seconds':50000,'max_memory_mb':2048,
                'max_external_requests':30,'max_llm_calls':12,'max_output_tokens':24000}

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with tmp.open('xb') as stream:
        stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)

def source_identity(workspace):
    from .legacy_writers import ENTRIES, STATE_BOUNDARIES
    root=Path(workspace)
    return {'native':code_manifest(), 'legacy_boundaries':{
        name:file_sha(root/name) for name in sorted(set(ENTRIES)|set(STATE_BOUNDARIES))}}

def _within(root, path):
    path=Path(path).resolve()
    require(path.is_relative_to(root), 'unified asset must belong to the workspace')
    require(not path.is_relative_to(root/'data/raw'), 'raw data is not a controller output')
    return path

def normalize_request(root, request):
    require(isinstance(request,dict) and set(request)<= {
        'schema_version','input','input_sha256','rounds','candidates_per_round','seed',
        'mode','scheduler_policy','include_catalog','purpose','limits','memory_context',
        'llm_enabled','model_algorithms','external_hypotheses'},'unknown unified request fields')
    value={'schema_version':SCHEMA,'rounds':3,'candidates_per_round':2,'seed':17,
           'mode':'adaptive','scheduler_policy':'legacy','include_catalog':True,
           'llm_enabled':False,'model_algorithms':(),'external_hypotheses':(),
           'purpose':'bounded historical research',**request}
    require(value['schema_version']==SCHEMA,'unknown unified schema')
    source=Path(value['input']);source=source if source.is_absolute() else root/source
    source=source.resolve()
    require(source.is_file() and source.is_relative_to(root),'frozen native input must be in workspace')
    actual=file_sha(source)
    require(value.get('input_sha256')==actual,'input SHA256 must be explicitly frozen and match')
    payload=load_json(source)
    require(payload.get('schema_version')=='fuxi.campaign_input.1' and
            payload.get('snapshot',{}).get('schema_version')=='fuxi.snapshot.2' and
            payload['snapshot'].get('exposure')=='research_exposed',
            'first unified adapter requires a frozen revision-aware historical input')
    from .full_market_support import execution_options
    require(execution_options(payload) is None,'full-market execution needs a separately bounded adapter')
    require(file_sha(source)==actual,'input changed during validation')
    require(type(value['rounds']) is int and 1<=value['rounds']<=50
            and type(value['candidates_per_round']) is int and 1<=value['candidates_per_round']<=10,'invalid search bound')
    require(type(value['seed']) is int and value['seed']>=0,'invalid seed')
    require(value['mode'] in {'adaptive','fixed','no_memory'} and
            value['scheduler_policy'] in {'legacy','selection_v2'},'unknown research policy')
    require(type(value['include_catalog']) is bool,'catalog flag must be boolean')
    require(type(value['llm_enabled']) is bool,'llm flag must be boolean')
    require(isinstance(value['model_algorithms'],(list,tuple)) and
            set(value['model_algorithms'])<= {'ridge','lightgbm','xgboost'},'unknown model algorithm set')
    value['model_algorithms']=list(value['model_algorithms'])
    require(isinstance(value['external_hypotheses'],(list,tuple)),'external hypotheses must be a list')
    for h in value['external_hypotheses']:
        require(isinstance(h,dict) and set(h)<= {'mechanism','direction','source'} and
                isinstance(h.get('mechanism',''),str),'external hypothesis fields differ')
    value['external_hypotheses']=list(value['external_hypotheses'])
    value['input']=str(source)
    context=value.get('memory_context')
    if context is not None:
        from .learning import export_memory, compatibility
        from .data import Snapshot
        from .policy import calibration_request
        require(value['mode']!='no_memory' and isinstance(context,dict) and
                set(context)=={'compatibility_hash','sources'},'explicit memory context required')
        snapshot=Snapshot(payload['snapshot']);horizon=payload['labels']['label_spec']['horizon_sessions']
        index=max(snapshot.start+25,min(80,len(snapshot.sessions)//2))
        native=calibration_request(snapshot,calibration_end=snapshot.sessions[index],
                                   research_start=snapshot.sessions[index+horizon+2],horizon=horizon)
        require(context['compatibility_hash']==compatibility(payload,native),'shared memory input/window/cost mismatch')
        sources=context['sources']
        require(isinstance(sources,list) and 1<=len(sources)<=3,'shared memory sources must be bounded')
        normalized=[]
        for binding in sources:
            require(isinstance(binding,dict) and set(binding)=={'path','manifest_hash'},'memory binding fields differ')
            path=Path(binding['path']).resolve()
            require(path.is_relative_to(root) and path.is_file(),'memory source must be a workspace artifact')
            manifest=export_memory(path,context['compatibility_hash'])
            require(manifest['records'] and digest(manifest)==binding['manifest_hash'],'frozen shared memory changed or revoked')
            normalized.append({'path':str(path),'manifest_hash':binding['manifest_hash']})
        require(len({row['path'] for row in normalized})==len(normalized),'duplicate shared memory source')
        require(sum(len(export_memory(row['path'],context['compatibility_hash'])['records']) for row in normalized)<=100,
                'shared feedback import exceeds one hundred records')
        value['memory_context']={'compatibility_hash':context['compatibility_hash'],'sources':normalized}
    return value

def _limits(identity, request):
    limits=dict(identity.get('budgets') or DEFAULT_LIMITS)
    explicit=request.get('limits')
    if explicit:
        require(set(explicit)==set(DEFAULT_LIMITS) and all(type(v) is int and v>=0 for v in explicit.values()),
                'all budget axes must be explicit nonnegative integers')
        require(identity['trigger']=='manual' or all(explicit[k]<=limits[k] for k in limits),
                'automatic request cannot expand its budget')
        limits=dict(explicit)
    require(request['rounds']<=limits['max_rounds'] and
            request['rounds']*request['candidates_per_round']<=limits['max_candidates'],'candidate budget exceeded')
    require(limits['max_worker_seconds']>=120 and limits['max_memory_mb']>=2048,
            'current native worker contract requires 120s and 2048MiB')
    require(_budget_units(request['rounds'],request['candidates_per_round'])<=limits['max_total_seconds'],
            'native reservation exceeds workflow time budget')
    require(limits['max_external_requests']>=0 and limits['max_llm_calls']>=0 and limits['max_output_tokens']>=0,
            'external IO budget must be nonnegative')
    return limits

def _database(root):
    directory=root/'data/fuxi_unified';directory.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(directory/'runs.sqlite',timeout=30)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, identity TEXT UNIQUE NOT NULL, manifest TEXT NOT NULL, state TEXT NOT NULL, output TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, at TEXT NOT NULL)')
    db.commit();return db

def _event(db, run_id, kind, payload):
    db.execute('INSERT INTO events(run_id,kind,payload,at) VALUES(?,?,?,?)',
               (run_id,kind,canonical(payload).decode('utf8'),datetime.now(timezone.utc).isoformat()))
    db.commit()

def _verify_result(directory):
    result=load_json(directory/'COMPLETE.json')
    require(result['result_hash']==digest({k:v for k,v in result.items() if k!='result_hash'}),'unified completion changed')
    for path,expected in result['artifacts'].items():
        target=_within(directory, directory/path)
        require(target.is_file() and file_sha(target)==expected,'unified output content changed: '+path)
    return result

def bounded_process(command, directory, env, *, timeout, memory_mb):
    """Count the real process tree; propagate original failure and reclaim descendants."""
    from .controller_processes import ControllerTreeGuard as _WorkerTreeGuard
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();reason=None;original=None;cleanup_error=None;diagnostics_error=None
    process=None;guard=None
    controller_memory={'measurement':'outer_controller_rss_plus_child_tree_rss',
        'peak_controller_rss_bytes':0,'peak_combined_rss_bytes':0,'sample_count':0,'samples':[]}
    try:
        with (directory/'stdout.log').open('wb') as out, (directory/'stderr.log').open('wb') as err:
            process=subprocess.Popen(command,cwd=str(Path(__file__).resolve().parents[1]),env=env,stdout=out,stderr=err)
            guard=_WorkerTreeGuard(process)
            while process.poll() is None:
                if time.monotonic()-started>timeout:reason='workflow_timeout';break
                import psutil
                child_rss=guard.sample();outer_rss=psutil.Process(os.getpid()).memory_info().rss
                combined=child_rss+outer_rss
                controller_memory['peak_controller_rss_bytes']=max(controller_memory['peak_controller_rss_bytes'],outer_rss)
                controller_memory['peak_combined_rss_bytes']=max(controller_memory['peak_combined_rss_bytes'],combined)
                controller_memory['sample_count']+=1
                controller_memory['samples'].append({'controller_rss_bytes':outer_rss,'child_tree_rss_bytes':child_rss,'combined_rss_bytes':combined})
                controller_memory['samples']=controller_memory['samples'][-512:]
                if combined>memory_mb*1024*1024:reason='workflow_memory_exceeded';break
                time.sleep(0.1)
            if reason:guard.stop()
            process.wait(timeout=15)
    except BaseException as exc:
        original=exc
        reason=reason or 'process_or_monitor_error'
    finally:
        try:
            if guard is not None:guard.stop()
        except BaseException as exc:
            cleanup_error={'type':type(exc).__name__,'reason':str(exc)}
        # Popen owns this launcher handle. Always reap it even when tree cleanup
        # failed; do not infer descendant cleanup from launcher termination.
        if process is not None:
            try:
                if process.poll() is None:process.kill()
                process.wait(timeout=15)
            except BaseException as exc:
                fallback={'type':type(exc).__name__,'reason':str(exc)}
                if cleanup_error:cleanup_error['launcher_fallback_error']=fallback
                else:cleanup_error=fallback
    tree=None
    if guard is not None:
        try:tree=guard.diagnostics()
        except BaseException as exc:
            diagnostics_error={'type':type(exc).__name__,'reason':str(exc)}
            tree={'descendants_unverified':True,'cleanup':getattr(guard,'cleanup',None)}
    resource={'exit_code':process.returncode if process is not None else None,'reason':reason,'elapsed_seconds':time.monotonic()-started,
              'memory_limit_bytes':memory_mb*1024*1024,'timeout_seconds':timeout,
              'original_error':{'type':type(original).__name__,'reason':str(original)} if original else None,
              'cleanup_error':cleanup_error,'diagnostics_error':diagnostics_error,'tree':tree,
              'controller_memory':controller_memory}
    atomic(directory/'RESOURCE.json',resource)
    if original:raise original
    require(cleanup_error is None,'research cleanup error; inspect RESOURCE.json')
    require(diagnostics_error is None,'research diagnostics failed; inspect RESOURCE.json')
    require(reason is None and process.returncode==0,
            f'research child failed: reason={reason}, exit_code={process.returncode}; inspect {directory / "RESOURCE.json"}')
    require(guard.monitor_error is None,'research tree could not be verified')
    require(guard.cleanup and guard.cleanup['complete'],'research descendant cleanup incomplete')
    return resource

def _static_artifacts(directory):
    root=Path(directory)
    return {str(p.relative_to(root)).replace('\\','/'):file_sha(p) for p in root.rglob('*')
            if p.is_file() and p.name!='COMPLETE.json' and p.suffix!='.sqlite'
            and not any(s in p.name for s in ('sqlite-wal','sqlite-shm'))}

def _fixed_phase(root,owner,directory,action,specification,started,limits):
    require(action in ('_catalog','_feedback'),'unsupported fixed resource phase')
    path=Path(directory)/'REQUEST.json';atomic(path,specification)
    remaining=limits['max_total_seconds']-(time.monotonic()-started)
    require(remaining>0,'workflow time exhausted before '+action)
    command=[sys.executable,'-B','-X','utf8','-m','fuxi_v1.unified_runner',action,
             '--workspace',str(root),'--request',str(path)]
    return bounded_process(command,directory,child_environment(owner,'index_refresh' if action=='_catalog' else 'feedback'),
                           timeout=remaining,memory_mb=limits['max_memory_mb'])

def run(workspace, request, *, trigger='manual'):
    start=time.monotonic()
    root=Path(workspace).resolve()
    with governed_writer(root,'unified_run',trigger=trigger) as owner:
        try:
            req=normalize_request(root,request)
            for name,expected in {'FUXI_V1_PARENT_POLICY':'pareto','FUXI_V1_MOTIF_MEMORY':'off',
                                  'FUXI_V1_FITNESS_TRIAGE':'enforce'}.items():
                require(os.environ.get(name,expected)==expected,'unified native policy environment changed: '+name)
            limits=_limits(owner,req)
            codes=source_identity(root)
            policy=load_json(root/'codex_automations/fuxi_runtime_policy.json')
            if trigger=='automatic':
                require(policy['bounded_research'].get('approved_code_identity')==digest(codes),
                        'automatic code is not the approved release')
                approved=policy['bounded_research'].get('approved_request_hashes')
                require(isinstance(approved,list) and digest(req) in approved,'automatic research request is not frozen in the authorization')
        except Exception as exc:
            atomic(root/'data/fuxi_unified/rejections'/('rejection_'+uuid.uuid4().hex+'.json'),
                   {'trigger':trigger,'state':'refused','type':type(exc).__name__,'reason':str(exc),
                    'new_workers':0,'new_research_facts':0})
            raise
        # Trigger, date and output directory never manufacture new experiments.
        # P0.1 (v2.0): behaviour-affecting env knobs (e.g. FUXI_V1_GEN_EXPAND) are
        # resolved into effective_config and enter the run identity; two runs that
        # differ only in such a switch can no longer collide on one run_id.
        experiment={'request':req,'code':codes,'limits':limits,
                    'effective_config':collect_effective_config()}
        key=digest(experiment);run_id='unified_'+key[:24]
        directory=root/'data/fuxi_unified/runs'/run_id
        db=_database(root)
        try:
            row=db.execute('SELECT identity,manifest,state,output FROM runs WHERE id=?',(run_id,)).fetchone()
            if row:
                require(row[0]==key and json.loads(row[1])==experiment and Path(row[3])==directory,'run identity collision')
                if row[2]=='completed' or (directory/'COMPLETE.json').exists():
                    answer=_verify_result(directory)
                    require(answer['identity']==key and answer['run_id']==run_id and answer['request']==req,
                            'completion belongs to a different frozen run')
                    db.execute('UPDATE runs SET state=? WHERE id=?',('completed',run_id));db.commit()
                    require(time.monotonic()-start<=limits['max_total_seconds'],'workflow time exhausted during completion verification')
                    _event(db,run_id,'cached_trigger',{'trigger':trigger,'new_workers':0,'new_research_facts':0})
                    return dict(answer,cached=True)
            else:
                directory.mkdir(parents=True,exist_ok=False)
                atomic(directory/'RUN_PROTOCOL.json',{'identity':key,**experiment})
                db.execute('INSERT INTO runs VALUES(?,?,?,?,?)',(run_id,key,canonical(experiment).decode('utf8'),'frozen',str(directory)));db.commit()
            _event(db,run_id,'trigger',{'trigger':trigger,'owner_run_id':owner['run_id']})
            db.execute('UPDATE runs SET state=? WHERE id=?',('running',run_id));db.commit()
            phase_resources={}
            from .legacy_writers import inventory
            coverage=inventory(root)
            require(coverage['all_declared_boundaries_present'],'declared production writer guard missing')
            atomic(directory/'WRITER_INVENTORY.json',coverage)
            if req['include_catalog'] and not (directory/'catalog/SUMMARY.json').exists():
                phase_resources['catalog_before']=_fixed_phase(root,owner,directory/'process_catalog_before','_catalog',
                    {'output':str(directory/'catalog')},start,limits)
                _event(db,run_id,'phase_completed',{'phase':'factor_catalog'})
            invocation={'input':req['input'],'output':str(directory/'research'),
                        'rounds':req['rounds'],'candidates_per_round':req['candidates_per_round'],
                        'seed':req['seed'],'mode':req['mode'],'scheduler_policy':req['scheduler_policy'],
                        'llm_enabled':req.get('llm_enabled',False),
                        'model_algorithms':list(req.get('model_algorithms',())),
                        'external_hypotheses':list(req.get('external_hypotheses',())),
                        'campaign_id':'campaign_'+key[:24],'authority':str(authority_location(directory)),
                        'memory_context':req.get('memory_context')}
            atomic(directory/'CAMPAIGN_REQUEST.json',invocation)
            remaining=limits['max_total_seconds']-(time.monotonic()-start)
            require(remaining>0,'workflow time exhausted before research')
            command=[sys.executable,'-B','-X','utf8','-m','fuxi_v1.unified_runner','_campaign',
                     '--workspace',str(root),'--request',str(directory/'CAMPAIGN_REQUEST.json')]
            checkpoint=directory/'PHASE_RESEARCH.json'
            if checkpoint.exists():
                saved=load_json(checkpoint)
                require(saved['invocation_hash']==digest(invocation) and
                        saved['receipt_sha256']==file_sha(directory/'research/RECEIPT.json') and
                        saved['resource_sha256']==file_sha(directory/'process/RESOURCE.json') and
                        saved['research_artifacts']==_static_artifacts(directory/'research'),
                        'completed research phase changed before retry')
                resources=load_json(directory/'process/RESOURCE.json')
                require(resources['exit_code']==0 and resources['reason'] is None and
                        resources['tree']['cleanup']['complete'],'research checkpoint was not successful')
                _event(db,run_id,'phase_reused',{'phase':'research','new_workers':0})
            else:
                resources=bounded_process(command,directory/'process',child_environment(owner,'candidate_search'),
                                          timeout=remaining,memory_mb=limits['max_memory_mb'])
                atomic(checkpoint,{'invocation_hash':digest(invocation),
                    'receipt_sha256':file_sha(directory/'research/RECEIPT.json'),
                    'resource_sha256':file_sha(directory/'process/RESOURCE.json'),
                    'research_artifacts':_static_artifacts(directory/'research')})
            phase_resources['research']=resources
            research=load_json(directory/'research/RECEIPT.json')
            phase_resources['feedback']=_fixed_phase(root,owner,directory/'process_feedback','_feedback',
                {'report_path':research['report_path'],'output':str(directory/'feedback')},start,limits)
            feedback=load_json(directory/'feedback/RECEIPT.json')
            if req['include_catalog'] and not (directory/'catalog_after/SUMMARY.json').exists():
                phase_resources['catalog_after']=_fixed_phase(root,owner,directory/'process_catalog_after','_catalog',
                    {'output':str(directory/'catalog_after')},start,limits)
                _event(db,run_id,'phase_completed',{'phase':'factor_catalog_after_research'})
            require(source_identity(root)==codes and file_sha(req['input'])==req['input_sha256'],
                    'code or input changed during research; new batch required')
            artifacts=_static_artifacts(directory)
            workflow_elapsed=time.monotonic()-start
            require(workflow_elapsed<=limits['max_total_seconds'],'workflow time exhausted before completion')
            answer={'schema_version':SCHEMA,'run_id':run_id,'identity':key,'state':'completed','request':req,
                    'research':research,'feedback':feedback,'resources':resources,'artifacts':artifacts,
                    'workflow_elapsed_seconds':workflow_elapsed,'phase_resources':phase_resources,
                    'new_market_requests':0,'LLM_calls':0,'live_execution_allowed':False,
                    'qualification_changes':0,'production_default_changes':0}
            answer['result_hash']=digest(answer);atomic(directory/'COMPLETE.json',answer)
            db.execute('UPDATE runs SET state=? WHERE id=?',('completed',run_id));db.commit()
            _event(db,run_id,'completed',{'report':research['report_path'],'actual_candidates':research['attempted_candidates']})
            return dict(answer,cached=False)
        except BaseException as exc:
            if directory.exists():atomic(directory/'LAST_FAILURE.json',{'type':type(exc).__name__,'reason':str(exc)})
            db.execute('UPDATE runs SET state=? WHERE id=?',('failed',run_id));db.commit()
            _event(db,run_id,'failed',{'type':type(exc).__name__,'reason':str(exc)})
            raise
        finally:db.close()

def _campaign(workspace, request):
    with governed_writer(workspace,'candidate_search',trigger=_inherited_trigger(workspace)):
        from .campaign import run_campaign
        from .learning import export_memory
        context=request.get('memory_context');sources=[]
        if context:
            for binding in context['sources']:
                require(digest(export_memory(binding['path'],context['compatibility_hash']))==binding['manifest_hash'],
                        'memory changed between controller and child')
                sources.append(binding)
        path,report=run_campaign(request['input'],request['output'],rounds=request['rounds'],
            candidates_per_round=request['candidates_per_round'],mode=request['mode'],seed=request['seed'],
            llm_enabled=request.get('llm_enabled',False),
            model_algorithms=tuple(request.get('model_algorithms',())),
            external_hypotheses=tuple(request.get('external_hypotheses',())),
            campaign_id=request['campaign_id'],
            authority_path=request['authority'],scheduler_policy=request['scheduler_policy'],memory_sources=sources)
        config=load_json(Path(path).parent/'config.json')
        atomic(Path(request['output'])/'RECEIPT.json',{
            'report_path':str(path),'report_sha256':file_sha(path),
            'imported_memory_records':sum(len(m['records']) for m in config.get('memory_imports',[])),
            'memory_sources':[row['path'] for row in sources],
            **{k:report[k] for k in ('status','attempted_candidates','valid_descriptive_factors',
                 'engineering_errors','formal_account_valid','economic_reward_count','jq_candidate_count')}})

def _catalog(workspace,request):
    require(set(request)=={'output'},'catalog phase fields differ')
    with governed_writer(workspace,'index_refresh',trigger=_inherited_trigger(workspace)):
        from .shared_catalog import build_catalog
        build_catalog(workspace,request['output'])

def _feedback(workspace,request):
    require(set(request)=={'report_path','output'},'feedback phase fields differ')
    with governed_writer(workspace,'feedback',trigger=_inherited_trigger(workspace)):
        from .shared_feedback import import_feedback
        result=import_feedback(request['report_path'],request['output'],workspace)
        atomic(Path(request['output'])/'RECEIPT.json',result)

def _inherited_trigger(workspace):
    from .governance import inherited_identity
    identity=inherited_identity(workspace)
    require(identity is not None,'internal worker requires the real controller owner')
    return identity['trigger']

def study(workspace, specification):
    """Manual, serial three-arm acceptance through the same owner and tree limits."""
    root=Path(workspace).resolve()
    require(set(specification)=={'input','output','request'},'study envelope fields differ')
    source=_within(root,Path(specification['input']))
    output=_within(root,Path(specification['output']))
    with governed_writer(root,'unified_run',trigger='manual') as owner:
        from .fusion_study import _request, _checked, status as study_status
        request=_request(specification['request'])
        require(file_sha(source)==request.get('source_sha256'),'study source must be explicitly frozen')
        if (output/'SUMMARY.json').exists():
            _,protocol=_checked(output)
            require(protocol['request']==request and protocol['source']['path']==str(source),'cached study envelope changed')
            return {'state':'completed','cached':True,'summary':study_status(output)['summary'],'new_workers':0}
        output.mkdir(parents=True,exist_ok=True)
        bound={'max_candidates':30,'max_rounds_per_arm':5,'max_total_seconds':5400,
               'max_memory_mb':2048,'max_external_requests':0,'max_llm_calls':0,'max_output_tokens':0}
        envelope={'input':str(source),'output':str(output),'request':request,
                  'authority':str(authority_location(output)),'budget':bound,'code':source_identity(root)}
        path=output/'CONTROLLER_REQUEST.json'
        previous=0.;process_directory=output/'controller'
        if path.exists():
            frozen=load_json(path)
            require({k:v for k,v in frozen.items() if k!='code'}=={k:v for k,v in envelope.items() if k!='code'},
                    'study controller envelope changed')
            _,protocol=_checked(output)
            require(protocol['request']==request and protocol['source']['path']==str(source),'study research freeze changed')
            from .frozen_study_runtime import prepare
            runtime=prepare(output)
            records=list(output.glob('controller*/**/RESOURCE.json'))
            previous=sum(load_json(record)['elapsed_seconds'] for record in records)
            remaining=bound['max_total_seconds']-previous
            require(remaining>0,'study total active controller time exhausted')
            number=len(records)+1
            process_directory=output/'controller_attempts'/('attempt_'+str(number))
            require(not process_directory.exists(),'controller attempt identity already exists')
            atomic(output/'controller_attempts'/('protocol_'+str(number)+'.json'),{
                'original_controller_envelope_sha256':file_sha(path),'original_native_code':protocol['code_hashes'],
                'outer_controller_code':source_identity(root),'previous_active_seconds':previous,
                'remaining_active_seconds':remaining,'same_native_input_and_decisions':True,
                'recovery':'original verified native package; running attempt cost conservatively reserved'})
            command=[sys.executable,'-B','-X','utf8','-m','fuxi_v1.frozen_study_runtime',
                     '--workspace',str(root),'--request',str(path),'--runtime',str(runtime)]
        else:
            remaining=bound['max_total_seconds'];atomic(path,envelope)
            command=[sys.executable,'-B','-X','utf8','-m','fuxi_v1.unified_runner','_study',
                     '--workspace',str(root),'--request',str(path)]
        resources=bounded_process(command,process_directory,child_environment(owner,'evaluate'),
                                  timeout=remaining,memory_mb=bound['max_memory_mb'])
        return {'state':'completed','cached':False,'summary':study_status(output)['summary'],'resources':resources}

def _study(workspace, specification):
    with governed_writer(workspace,'evaluate',trigger=_inherited_trigger(workspace)):
        from .fusion_study import run_study
        from .shared_feedback import import_feedback
        require(source_identity(workspace)==specification['code'],'study controller code changed')
        summary=run_study(specification['input'],specification['output'],specification['request'],
                          authority_path=specification['authority'])
        for arm,record in summary['arms'].items():
            import_feedback(record['report_path'],Path(specification['output'])/'feedback'/arm,workspace)

def status(workspace, run_id=None):
    root=Path(workspace).resolve();path=root/'data/fuxi_unified/runs.sqlite'
    if not path.exists():return {'state':'not_started','runs':[]}
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    try:
        query='SELECT id,state,output FROM runs';params=()
        if run_id:query+=' WHERE id=?';params=(run_id,)
        rows=[dict(zip(('run_id','state','output'),r)) for r in db.execute(query,params)]
    finally:db.close()
    for row in rows:
        if row['state']=='completed':row['verified']=_verify_result(Path(row['output']))['state']=='completed'
    return {'state':'read_only','runs':rows}

def tick(workspace, inbox=None):
    """Consume at most one operator-frozen request; never invent another trial."""
    root=Path(workspace).resolve()
    queue=_within(root,Path(inbox) if inbox else root/'data/fuxi_unified/inbox')
    files=sorted(queue.glob('*.json')) if queue.exists() else []
    if not files:return {'state':'idle','reason':'no operator-frozen input request','new_workers':0}
    known=status(root)['runs'];completed={row['run_id'] for row in known if row['state']=='completed'}
    policy=load_json(root/'codex_automations/fuxi_runtime_policy.json')
    codes=source_identity(root)
    limits=policy.get('bounded_research',{}).get('budgets',{})
    for path in files:
        req=normalize_request(root,load_json(path))
        identity=digest({'request':req,'code':codes,'limits':req.get('limits') or limits})
        if 'unified_'+identity[:24] not in completed:
            return run(root,req,trigger='automatic')
    return {'state':'idle','reason':'same data/code/protocol already completed','new_workers':0}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','status','tick','study','_campaign','_study','_catalog','_feedback'))
    parser.add_argument('--workspace',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--request');parser.add_argument('--trigger',choices=('manual','automatic'),default='manual')
    parser.add_argument('--run-id');args=parser.parse_args()
    if args.action=='status':answer=status(args.workspace,args.run_id)
    elif args.action=='tick':answer=tick(args.workspace)
    else:
        require(args.request,'request is required')
        request=load_json(args.request)
        if args.action=='_campaign':_campaign(args.workspace,request);answer={'state':'worker_completed'}
        elif args.action=='_study':_study(args.workspace,request);answer={'state':'study_worker_completed'}
        elif args.action=='_catalog':_catalog(args.workspace,request);answer={'state':'catalog_worker_completed'}
        elif args.action=='_feedback':_feedback(args.workspace,request);answer={'state':'feedback_worker_completed'}
        elif args.action=='study':
            require(args.trigger=='manual','paired acceptance study requires the manual envelope')
            answer=study(args.workspace,request)
        else:answer=run(args.workspace,request,trigger=args.trigger)
    summary={k:answer[k] for k in ('state','run_id','cached','reason','new_workers','research','qualification_changes','live_execution_allowed','runs') if k in answer}
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'state':'refused','type':type(exc).__name__,'reason':str(exc)},ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
