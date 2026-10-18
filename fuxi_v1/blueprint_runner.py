"""One explicit, resumable operator for the complete research and evidence lifecycle.

Engineering completion and genuine external acceptance are separate fields. No
scheduled writes, external LLM calls, automatic qualifications or live orders.
"""
import csv
from datetime import datetime
from pathlib import Path
from .contracts import load_json,digest,require,timestamp
from .datasets import file_hash
from .campaign import write_new
from .artifacts import code_manifest
from .data import snapshot_from_payload,Labels


def _checked(root):
    protocol=load_json(Path(root)/'RUN_PROTOCOL.json')
    require(protocol['protocol_hash']==digest({k:v for k,v in protocol.items() if k!='protocol_hash'}),'blueprint protocol changed')
    require(protocol['code_hashes']==code_manifest(),'blueprint code changed; freeze a new run')
    for row in protocol['inputs'].values():require(file_hash(row['path'])==row['sha256'],'blueprint input changed')
    return protocol


def freeze(spec,output):
    require(spec.get('schema_version')=='fuxi.blueprint_run_request.1','blueprint request schema')
    root=Path(output).resolve()
    if (root/'RUN_PROTOCOL.json').exists():
        protocol=_checked(root)
        require(protocol['request_hash']==digest(spec),'cannot revise a frozen blueprint request')
        _finish_freeze(root,protocol)
        return {'output':str(root),'protocol_hash':protocol['protocol_hash'],
                'cohort_size':protocol['cohort_size'],'state':'frozen','idempotent':True}
    inputs={}
    for key in ['input_path','industry_path','benchmark_path','calendar_path','lifecycle_path','draw_path']:
        require(spec.get(key),'blueprint '+key+' required')
        p=Path(spec[key]).resolve();inputs[key]={'path':str(p),'sha256':file_hash(p)}
    payload=load_json(spec['input_path']);snapshot=snapshot_from_payload(payload['snapshot']);Labels(payload['labels'],snapshot)
    require(snapshot.payload['exposure']=='research_exposed','blueprint history must disclose exposure')
    codes=load_json(spec['draw_path'])['codes']
    from .history import jq_code
    observed=set(snapshot.payload.get('instruments') or set().union(*map(set,snapshot.universe.values())))
    require(observed=={jq_code(c) for c in codes} and len(codes)==len(set(codes)),'blueprint frozen cohort mismatch')
    request={'research_start':spec['research_start'],'research_end':spec['research_end'],
      'alpha':.05,'block_length':12,'bootstrap_replicates':399,'max_trials':9,
      'minimum_net_increment_daily':0.,'cost_stress_multiplier':2.,'capacity_stress_multiplier':.5,
      'pool_top_n':len(codes),'style_data_policy':'latest_available_snapshot',
      'industry_candidates':[inputs['industry_path']['path']],'full_attribution':True,
      'neutralization_objective':'diagnostic_style_residual'}
    require(request['research_start']<request['research_end'] and all(d in snapshot.sessions for d in
            (request['research_start'],request['research_end'])),'blueprint window invalid')
    protocol={'schema_version':'fuxi.blueprint_run.1','frozen_at':datetime.now().astimezone().isoformat(),
      'inputs':inputs,'request':request,'request_hash':digest(spec),'snapshot_hash':snapshot.hash,'cohort_size':len(codes),
      'code_hashes':code_manifest(),'handoff_method':'screened','handoff_study':'selection',
      'budgets':{'single_feature_edges':1,'selection_candidates':2,'selection_methods':3,'stress_scenarios':2,
                 'conditional_rights_max_paths_per_account_per_scenario':8,
                 'new_external_LLM_calls':0},'evidence_references':spec.get('evidence_references',{}),
      'scope':'current-vintage research; no historical PIT certificate; genuine JQ and future receipts required',
      'scheduled_writes_enabled':False,'automatic_qualification_allowed':False,'live_execution_allowed':False}
    protocol['protocol_hash']=digest(protocol)
    write_new(root/'RUN_PROTOCOL.json',protocol)
    _finish_freeze(root,protocol)
    return {'output':str(root),'protocol_hash':protocol['protocol_hash'],'cohort_size':len(codes),'state':'frozen'}


def _finish_freeze(root,protocol):
    from .remaining_study import freeze as native_freeze
    inputs=protocol['inputs'];path=root/'research/REMEDIATION_PROTOCOL.json'
    if not path.exists():
        native_freeze(inputs['input_path']['path'],root/'research',protocol['request'],
                      industry=inputs['industry_path']['path'],benchmark=inputs['benchmark_path']['path'])
    native=load_json(path)
    require(native['protocol_hash']==digest({k:v for k,v in native.items() if k!='protocol_hash'})
            and native['code_hashes']==protocol['code_hashes'] and native['request']==protocol['request']
            and native['snapshot_hash']==protocol['snapshot_hash'],'nested research freeze changed')
    for base in (root/'code',root/'research/code'):
        for name,expected in protocol['code_hashes'].items():
            target=base/name
            if target.exists():require(file_hash(target)==expected,'frozen code copy changed')
            else:write_new(target,(Path(__file__).parent/name).read_bytes())


def _stage(root,protocol,name,callback):
    path=root/'stages'/(name+'.json')
    if path.exists():
        saved=load_json(path)
        require(saved['record_hash']==digest({k:v for k,v in saved.items() if k!='record_hash'})
                and saved['protocol_hash']==protocol['protocol_hash']
                and saved['result_hash']==digest(saved['result']),'blueprint cached stage changed')
        for rel,expected in saved['files'].items():
            require(file_hash(root/rel)==expected,'blueprint stage output changed: '+rel)
        return saved['result']
    print('BLUEPRINT_STAGE',name,flush=True)
    result=callback();_checked(root)
    files={p.relative_to(root).as_posix():file_hash(p) for p in root.rglob('*')
           if p.is_file() and 'stages' not in p.relative_to(root).parts}
    saved={'protocol_hash':protocol['protocol_hash'],'result_hash':digest(result),'result':result,'files':files}
    saved['record_hash']=digest(saved);write_new(path,saved)
    return result


def run(output):
    root=Path(output).resolve();protocol=_checked(root)
    _finish_freeze(root,protocol)
    stage=lambda name,callback:_stage(root,protocol,name,callback)
    from .remaining_study import run as native_run,proxy_styles
    research=stage('NATIVE_RESEARCH',lambda:native_run(root/'research'))
    native_scope=load_json(root/'research/REMEDIATION_PROTOCOL.json')
    payload=load_json(protocol['inputs']['input_path']['path']);snapshot=snapshot_from_payload(payload['snapshot'])
    market=payload['market'];request=protocol['request'];start,end=request['research_start'],request['research_end']
    styles,provenance=proxy_styles(payload,load_json(protocol['inputs']['industry_path']['path']),
                                  data_policy=native_scope['style_data_policy'])
    selection=load_json(root/'research/SELECTION_AB.json')['result']
    accounting=load_json(root/'research/ACCOUNT_REPAIR.json')['result']
    from .restricted_assets import freeze_restricted_policy,replay_restricted_pair
    from .pit_universe import read_lifecycle_file
    from .history import jq_code,iso
    life=read_lifecycle_file(protocol['inputs']['lifecycle_path']['path'])
    entries=[]
    for code in load_json(protocol['inputs']['draw_path']['path'])['codes']:
        terminal=iso(life[code].get('delist_date'))
        if terminal and start<=terminal<=end:
            entries.append({'instrument':jq_code(code),'restrict_from':terminal,
              'known_at':terminal+'T00:00:00+08:00',
              'known_at_basis':'latest-vintage effective lifecycle research restriction; original announcement clock/rights not certified',
              'source_paths':[protocol['inputs']['lifecycle_path']['path']]})
    policy=stage('RESTRICTED_POLICY',lambda:freeze_restricted_policy(market,entries,horizon_end=end) if entries else
                 {'schema_version':'fuxi.no_restricted_members.1','entries':[],'scope':'no declared terminal member inside this run'})
    from .account_attribution import full_account_attribution,residual_account_experiment
    benchmark=load_json(protocol['inputs']['benchmark_path']['path'])
    def scenarios():
        if not entries:return {'applicable':False,'reason':'no terminal members','qualification_granted':False}
        pair=replay_restricted_pair(accounting['pool_ir'],selection['outputs']['screened']['strategy_ir'],market,policy,start=start,end=end)
        for row in pair['rows']:
            row['full_attribution']=full_account_attribution(row['accounts']['candidate'],row['accounts']['baseline'],
                                          market,styles,benchmark,style_provenance=provenance)
        return {'applicable':True,'comparison':'screened candidate versus full frozen pool in each identical scenario',**pair}
    restricted=stage('RESTRICTED_ACCOUNTS',scenarios)
    def residuals():
        if not entries:return {'covered_by':'research/FULL_ATTRIBUTION.json','additional_scenario_replay':False}
        from .dsl import evaluate_ast
        base=evaluate_ast(native_scope['baseline']['canonical_ast'],snapshot)
        return residual_account_experiment(base,styles,snapshot,market,request,objective=request['neutralization_objective'],restricted_policy=policy)
    residual=stage('RESTRICTED_RESIDUAL_AB',residuals)
    def conditional_rights():
        from .conditional_rights import freeze_rules,replay
        rules=freeze_rules(market,protocol['budgets']['conditional_rights_max_paths_per_account_per_scenario'])
        if not entries or not rules['rules']:
            return {'applicable':False,'rules':rules,'reason':'no supported fractional/restricted overlap','formal_eligibility':False}
        return {'applicable':True,'rules':rules,'replay':replay(accounting['pool_ir'],
            selection['outputs']['screened']['strategy_ir'],market,policy,rules,start=start,end=end,
            styles=styles,benchmark=benchmark,style_provenance=provenance)}
    rights=stage('CONDITIONAL_RIGHTS',conditional_rights)
    def index_cards():
        from .factor_cards import card,query
        candidates=[native_scope['baseline'],native_scope['single_feature'],*native_scope['selection_candidates']]
        for candidate in candidates:
            value=card(candidate,{}, {'window':{'start':start,'end':end},'exposure':'research_exposed',
                     'system_version':'current_code_manifest'},request,source_path=protocol['inputs']['input_path']['path'])
            value['diagnostic_evidence']={'native_result_path':str(root/'research/RESULT.json'),
                'native_result_sha256':file_hash(root/'research/RESULT.json'),
                'selection_readings_path':str(root/'research/SELECTION_DIAGNOSTICS.json'),
                'selection_readings_sha256':file_hash(root/'research/SELECTION_DIAGNOSTICS.json'),
                'formal_S1_S5_stage_states_available':False,'qualification_granted':False}
            value['card_hash']=digest({k:v for k,v in value.items() if k!='card_hash'})
            path=root/'cards'/(value['card_hash']+'.json')
            if path.exists():require(load_json(path)==value,'blueprint factor card changed')
            else:write_new(path,value)
        rows=query(root);require(rows,'blueprint card index empty')
        return {'n_cards':len(rows),'card_hashes':[r['card_hash'] for r in rows],
                'jq_candidate_allowed_count':sum(r['jq_candidate_allowed'] for r in rows),'qualification_granted':False}
    cards=stage('CARDS',index_cards)
    from .lifecycle_status import build_requests,evidence_state
    from .forward_archive import freeze_archive,status as forward_status,build_observation
    def prospective():
        from .online import daily_model_decision
        from .search import node
        import math
        output=selection['outputs'][protocol['handoff_method']];baseline=selection['outputs']['base']
        valid=[f for f in output['folds'] if f.get('prediction_end')];require(valid and output['models'] and baseline['models'],'no frozen model available')
        fold=valid[-1];chosen=fold['selected_feature'];direction=fold['readings'][chosen]['direction']
        formulas={c['formula_hash']:c['canonical_ast'] for c in native_scope['selection_candidates']}
        require(direction in (-1,1),'last frozen feature direction unavailable')
        added=formulas[chosen] if direction==1 else node('neg',formulas[chosen])
        features={'baseline_return20':native_scope['baseline']['canonical_ast'],'incremental_feature':added}
        contracts={'candidate':{'strategy':output['strategy_ir']['strategy'],'model':output['models'][-1],'feature_asts':features},
                   'baseline':{'strategy':baseline['strategy_ir']['strategy'],'model':baseline['models'][-1],
                               'feature_asts':{'baseline_return20':native_scope['baseline']['canonical_ast']}}}
        support_asts=[native_scope['baseline']['canonical_ast'],*formulas.values()]
        for contract in contracts.values():contract['support_asts']=support_asts
        observed=datetime.now().astimezone().isoformat();day=end;checks={}
        for name,result in [('candidate',output),('baseline',baseline)]:
            contract=contracts[name]
            decision=daily_model_decision(contract['feature_asts'],contract['model'],snapshot,day,observed,
                                          contract['strategy'],support_asts=contract['support_asts'])
            known={s for s,v in result['scores'][day].items() if v is not None}
            require(known=={s for s,v in decision['scores'].items() if v is not None},'frozen online/model support mismatch')
            paired=[(decision['scores'][s],result['scores'][day][s]) for s in sorted(known)]
            require(paired and all(math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12) for a,b in paired),'frozen online/model prediction mismatch')
            checks[name]={'historical_recomputation_not_future':True,'n_pairs':len(paired),
                          'max_absolute_error':max(abs(a-b) for a,b in paired),'decision':decision}
        calendar_path=protocol['inputs']['calendar_path']['path']
        with Path(calendar_path).open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
        dates=sorted({iso(r['cal_date']) for r in rows if r.get('is_open')=='1' and r.get('exchange') in (None,'SSE')})
        future=[d for d in dates if d>observed[:10]]
        require(future,'no declared prospective calendar available')
        future=future[:140]
        forward_spec={'schema_version':'fuxi.forward_archive_request.1','mode':'research_observation',
          'candidate_id':output['strategy_ir']['strategy']['candidate_id'],'ir_hash':output['strategy_ir']['ir_hash'],
          'baseline_hash':baseline['strategy_ir']['ir_hash'],'cost_profile_hash':digest(market['execution']),
          **contracts,'calendar':{'sessions':future,'decision_times':{d:d+'T18:30:00+08:00' for d in future},
            'evidence_path':calendar_path,'evidence_sha256':protocol['inputs']['calendar_path']['sha256'],
            'source_limit':'existing vendor schedule; official/independent date assurance not granted'},
          'look_sessions':[60,120],'horizon_sessions':5,'minimum_sections':60}
        frozen=freeze_archive(forward_spec,root/'forward_archive')
        waiting=build_observation(root/'forward_archive',{'look_sessions':60})
        require(waiting.get('status') in ('not_due','pending_calendar','data_insufficient'),'unobserved future accepted')
        return {'request':forward_spec,'freeze':frozen,'model_recomputation':checks,
                'archive_status':forward_status(root/'forward_archive'),'first_observation':waiting,
                'formal_acceptance_granted':False}
    forward=stage('PROSPECTIVE_ARCHIVE',prospective)
    def handoff():
        selected=selection['outputs'][protocol['handoff_method']]['strategy_ir']
        spec={'schema_version':'fuxi.lifecycle_requests.1','candidate_id':selected['strategy']['candidate_id'],
          'ir_hash':selected['ir_hash'],'snapshot_hash':snapshot.hash,'policy_hash':None,
          'baseline_hash':selection['outputs']['base']['strategy_ir']['ir_hash'],
          'cost_profile_hash':digest(market['execution']),'bundle_path':None,
          'software_checks':{'native_64_input':True,'source_version_frozen':True,'native_research':True,
             'full_attribution':True,'restricted_scenario_accounts':True,'frozen_prediction':True,'forward_archive':True},
          'forward_archive_request':forward['request'],**protocol['evidence_references']}
        return build_requests(spec,root/'lifecycle_requests')
    lifecycle=stage('LIFECYCLE_REQUESTS',handoff)
    state=evidence_state(protocol['evidence_references'],software_checks={
         'source_bound_input':True,'native_study_completed':True,'same_support_attribution':True,
         'cash_share_reference':True,'restricted_policy_consumer':True,'manual_platform_lifecycle':True,
         'causal_frozen_prediction':True,'prospective_archive':True})
    result={'schema_version':'fuxi.blueprint_run_result.1','protocol_hash':protocol['protocol_hash'],
      'state':'completed_awaiting_genuine_evidence','engineering_execution_completed':True,
      'cohort_size':protocol['cohort_size'],'native_research':research,'cards':cards,
      'restricted_scenarios':len(restricted.get('rows',[])),'residual_scenarios':len(residual.get('scenarios',[])),
      'conditional_rights_scenarios':len(rights.get('replay',{}).get('scenarios',[])),
      'lifecycle':state,'lifecycle_requests':lifecycle,'forward_archive':forward['archive_status'],
      'real_JQ_returns':0,'real_forward_observations':0,'new_qualifications':0,
      'automatic_qualification_allowed':False,'scheduled_writes_enabled':False,'live_execution_allowed':False}
    result['result_hash']=digest(result)
    path=root/'COMPLETE.json'
    if path.exists():require(load_json(path)==result,'blueprint completed result changed')
    else:write_new(path,result)
    return result


def status(output):
    root=Path(output).resolve();protocol=_checked(root)
    complete=root/'COMPLETE.json'
    for path in (root/'stages').glob('*.json'):
        _stage(root,protocol,path.stem,lambda:None)
    result=load_json(complete) if complete.exists() else None
    if result is not None:
        require(result['result_hash']==digest({k:v for k,v in result.items() if k!='result_hash'})
                and result['protocol_hash']==protocol['protocol_hash'],'blueprint completion changed')
    return {'schema_version':'fuxi.blueprint_status.1','protocol_hash':protocol['protocol_hash'],
            'state':result['state'] if result is not None else 'frozen_or_partial_resume_available',
            'completed_stages':sorted(p.stem for p in (root/'stages').glob('*.json')),
            'complete':result,'live_execution_allowed':False}
