"""Frozen latest-available versions for retrospective research, never PIT proof."""
from pathlib import Path
from .contracts import load_json,require,digest,timestamp
from .datasets import file_hash


def freeze_latest_style_policy(input_path,industry_paths,cutoff):
    """Select within the declared registry; economic dates remain unchanged."""
    require(industry_paths,'latest-version policy requires registered industry sources')
    clock=timestamp(cutoff);require(clock.utcoffset() is not None,'timezone required')
    versions=[]
    for value in sorted({str(Path(p).resolve()) for p in industry_paths}):
        source=load_json(value);captured=source.get('captured_at')
        require(isinstance(captured,str),'real source captured_at required; mtime is insufficient')
        when=timestamp(captured)
        require(when.utcoffset() is not None and when<=clock,'source capture after freeze or timezone missing')
        require(isinstance(source.get('queries'),list) and source['queries'],'empty registered industry source')
        versions.append({'path':value,'sha256':file_hash(value),'content_hash':digest(source),'captured_at':captured})
    latest=max(timestamp(v['captured_at']) for v in versions)
    tied=[v for v in versions if timestamp(v['captured_at'])==latest]
    require(len({v['sha256'] for v in tied})==1,'conflicting equally recent source versions')
    selected=min(tied,key=lambda v:v['path']);payload=load_json(input_path)
    provenance=payload['snapshot'].get('provenance') or {}
    policy={'schema_version':'fuxi.latest_vintage_research.1','mode':'latest_available_snapshot',
        'frozen_at':cutoff,'industry_candidates':versions,'industry_selected':selected,
        'selection_scope':'explicit registered versions only; no claim of global vendor completeness',
        'market_input_path':str(Path(input_path).resolve()),'market_input_sha256':file_hash(input_path),
        'market_payload_hash':digest(payload),'snapshot_hash':digest(payload['snapshot']),
        'market_source_identity':{k:provenance.get(k) for k in
            ('catalog_file_sha256','catalog_manifest_hash','catalog_metadata_hash','catalog_vintage')},
        'free_share_source_hash':(provenance.get('source_hashes') or {}).get('daily_basic.csv'),
        'free_share_selection':'values in the declared frozen market/catalog version; missing cells remain unknown',
        'alignment':'original historical observation dates and effective intervals; no current row backfill',
        'historical_revision_archive_required_for_this_diagnostic':False,
        'historical_PIT_certified':False,'investment_neutralization_certified':False,
        'formal_qualification_allowed':False,'live_execution_allowed':False,
        'limitations':['historical vendor revisions and original availability unproved',
                       'retrospective research-exposed data; no unknown-future claim']}
    policy['policy_hash']=digest(policy)
    return policy


def verify_latest_style_policy(policy,payload,source):
    require(policy.get('schema_version')=='fuxi.latest_vintage_research.1' and
            policy.get('mode')=='latest_available_snapshot','unsupported vintage policy')
    require(policy.get('policy_hash')==digest({k:v for k,v in policy.items() if k!='policy_hash'}),'vintage policy changed')
    require(policy['historical_PIT_certified'] is False and policy['formal_qualification_allowed'] is False,
            'latest retrieval cannot grant historical PIT or automatic qualification')
    for row in policy['industry_candidates']:
        require(file_hash(row['path'])==row['sha256'],'registered source version changed')
    require(file_hash(policy['market_input_path'])==policy['market_input_sha256'] and
            digest(payload)==policy['market_payload_hash'] and digest(payload['snapshot'])==policy['snapshot_hash'],
            'frozen market version changed')
    require(digest(source)==policy['industry_selected']['content_hash'],'selected industry version mismatch')
    return policy['policy_hash']
