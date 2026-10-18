"""Export frozen historical evidence into the existing operator admission route."""
import gzip
import json
from pathlib import Path
import re
import sqlite3
from .contracts import digest,require,load_json,load_fact
from .datasets import file_hash
from .artifacts import publish,code_manifest


def validate_context(frozen,plan,policy,outcome):
    require(digest(plan)==frozen['plan_hash']==policy['plan_hash'],'historical plan hash mismatch')
    require(outcome['freeze_hash']==digest(frozen),'historical freeze mismatch')
    require(outcome['policy_hash']==policy['policy_hash'] and outcome['snapshot_hash']==policy['snapshot_hash'],
            'historical outcome scope mismatch')
    require(digest({k:v for k,v in policy.items() if k!='policy_hash'})==policy['policy_hash'],'historical policy changed')
    require(policy['schema_version']=='fuxi.research_policy.2' and
            policy['request']['research_gate_version']==plan['protocol']['id']=='history_research_v2',
            'unsupported historical research protocol')
    request,p=policy['request'],plan['protocol']
    require(request['research_start']==plan['window']['research_start'] and request['research_end']==plan['window']['end'],
            'historical window mismatch')
    require(all(request[k]==p[k] for k in ('alpha','block_grid','bootstrap_replicates')) and
            request['screen_rules']=={k:v for k,v in p.items() if k.startswith('screen_')},'historical screen differs from plan')
    require(policy['minimum_effective_sections'] is None and policy['planning']['is_admission_gate'] is False and
            policy['planning']['calibration_n']>=30 and policy['planning']['planned_trials']==plan['reevaluate_formula_count'],
            'historical calibration scope invalid')
    require(re.fullmatch('[0-9a-f]{64}',policy['calibration_evaluation_hash']) is not None,
            'calibration provenance missing')
    require(policy['confirmatory_error_control_validated'] is False,'historical research cannot certify future error control')


def export_history(run_dir,formula_hash,output_root):
    """Read frozen results; no reevaluation, receipt fabrication, or JQ submission."""
    require(re.fullmatch('[0-9a-f]{64}',formula_hash) is not None,'full formula hash required')
    root=Path(run_dir).resolve();frozen=load_json(root/'RUN_FREEZE.json')
    plan=load_json(frozen['plan_path']);policy=load_json(root/'POLICY.json');outcome=load_json(root/'OUTCOME.json')
    validate_context(frozen,plan,policy,outcome)
    require(file_hash(frozen['input_path'])==frozen['input_sha256'],'historical input changed')
    require(all(file_hash(root/'code'/n)==h for n,h in frozen['code_hashes'].items()),'historical code archive changed')
    receipt=load_json(root/'results'/(formula_hash+'.receipt.json'))
    path=root/'results'/(formula_hash+'.json.gz')
    require(file_hash(path)==receipt['sha256'],'historical result corrupted')
    with gzip.open(path,'rt',encoding='utf-8') as stream:result=json.load(stream)
    require(result['snapshot_hash']==policy['snapshot_hash'] and result['policy_hash']==policy['policy_hash'],
            'result is outside frozen scope')
    card=receipt['card']
    db=sqlite3.connect((root/'evidence.sqlite').as_uri()+'?mode=ro',uri=True)
    try:
        row=db.execute('SELECT state FROM consumers WHERE id=?',(card['state_id'],)).fetchone()
        require(row is not None and row[0]=='valid','source research feedback revoked')
        fact=db.execute('SELECT payload,hash FROM facts WHERE id=?',(card['evaluation_id'],)).fetchone()
        require(fact is not None and digest(load_fact(fact[0]))==fact[1]==digest(result),'source evaluation evidence changed')
    finally:db.close()
    specs=load_json(frozen['candidate_path'])
    require(digest(specs)==frozen['candidate_hash'],'frozen candidate set changed')
    parents=load_json(root/'PARENT_COMPARATORS.json')
    require(digest(parents)==frozen['parent_comparators_hash'],'frozen ancestors changed')
    trials=load_json(root/'TRIALS.json')
    matches=[c for c in specs+parents+[r['candidate'] for r in trials] if c['candidate_id']==result['candidate_id']]
    require(matches and all(c['formula_hash']==formula_hash for c in matches),'result candidate identity mismatch')
    candidate=matches[0]
    files={'evaluation.json':result,'policy.json':policy,'candidate.json':candidate,'strategy_ir.json':result['strategy_ir'],
           'history_freeze.json':frozen,'history_plan.json':plan,'history_outcome.json':outcome,
           'source_result_receipt.json':receipt}
    destination=publish(output_root,files,{'source_run':str(root),'source_result_sha256':receipt['sha256'],
        'source_evaluation_id':card['evaluation_id'],'source_feedback_state':card['state_id'],
        'trial_code_hashes':frozen['code_hashes'],'exporter_code_hashes':code_manifest(),
        'data_contract_receipt_created':False,'JQ_submitted':False,'live_execution_allowed':False})
    return {'bundle_path':str(destination.resolve()),'formula_hash':formula_hash,'candidate_id':candidate['candidate_id'],
            'purpose':'operator_research_admission_review','eligibility_transferred':False}
