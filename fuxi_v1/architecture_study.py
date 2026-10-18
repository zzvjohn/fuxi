"""Manual, resumable execution of the revised architecture work packages.

All study artifacts are diagnostic. Native receipt-gated acceptance remains the
only consumer that can authorise formal JQ or forward acceptance.
"""
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import sqlite3
from .contracts import canonical, digest, require, load_json
from .artifacts import code_manifest
from .data import Snapshot, Labels
from .datasets import scope_of, validate_label_calendar
from .campaign import write_new
from .search import factor_spec, node, templates, generate
from .edits import refine_local, apply_patch, KINDS, llm_patches
from .dependency_audit import dependency_scope, pit_styles
from .factor_cards import card, query, legacy_projection
from .research_screening import diagnose, training_reading
from .incremental_study import compare_methods, account_for, common_support
from .account_reference import reconcile, account_diagnostics
from .calibration import random_population, baseline_summary
from .learning import paired_parent, ast_diff_motifs, motif_ledger, motif_vetoes, motif_gate, prompt_cards
from .research_evaluation import paired_scores
from .evaluation import factor_statistics, DEFAULT_POLICY


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def freeze(input_path, output, *, source_path=None, styles_path=None, llm=False):
    root = Path(output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    require(not (root/'RESEARCH_SCOPE.json').exists(), 'scope already frozen; use run to resume')
    payload = load_json(input_path)
    snapshot = Snapshot(payload['snapshot'])
    labels = Labels(payload['labels'],snapshot)
    validate_label_calendar(labels.payload,snapshot.sessions)
    require(snapshot.payload['exposure'] != 'forward_sealed' and len(snapshot.sessions) >= 130, 'research history required')
    require(payload['market']['market_hash'] == digest({k:v for k,v in payload['market'].items() if k != 'market_hash'}), 'market hash mismatch')
    start = snapshot.sessions[100]
    request = {'research_start':start, 'research_end':snapshot.sessions[-1], 'alpha':.05, 'max_trials':65+(3 if llm else 0),
               'block_length':12, 'bootstrap_replicates':399, 'minimum_net_increment_daily':0.,
               'cost_stress_multiplier':2.,'capacity_stress_multiplier':.5}
    base_ast = node('rank',node('return',{'field':'close'},window=20))
    source = templates(snapshot.payload['fields'])
    # Mechanisms are named before readings, never selected from whole-history IC.
    wanted = [0,1,2,3,4,5,6,10] if 'volume' in snapshot.payload['fields'] else list(range(8))
    fixed = [factor_spec(source[i][1],snapshot.payload['fields'],source[i][0],action='fixed_template') for i in wanted]
    candidates = {c['formula_hash']:c for c in fixed}
    base = factor_spec(base_ast,snapshot.payload['fields'],'frozen_return20_baseline')
    selection = [factor_spec(node('rank',node('return',{'field':'close'},window=5)),snapshot.payload['fields'],'price_persistence'),
                 factor_spec(node('rank',node('div',{'field':'volume'},node('mean',{'field':'volume'},window=5))),
                             snapshot.payload['fields'],'volume_surprise')] if 'volume' in snapshot.payload['fields'] else fixed[:2]
    registry = Path(__file__).resolve().parents[1]/'data/version_registry.json'
    system_version = load_json(registry)['versions'][-1]['version'] if registry.exists() else 'unregistered_workspace'
    scope = {'schema_version':'fuxi.architecture_study.1','frozen_at':datetime.now().astimezone().isoformat(),
             'system_version':system_version,
             'input_path':str(Path(input_path).resolve()),'input_sha256':file_hash(input_path),
             'input_hash':digest(payload),'snapshot_hash':snapshot.hash,'market_hash':payload['market']['market_hash'],
             'code_hashes':code_manifest(),'exposure':snapshot.payload['exposure'],'window':{'start':start,'end':snapshot.sessions[-1]},
             'scope':scope_of(snapshot),'request':request,'seed':1708,'baseline':base,'fixed_candidates':fixed,
             'selection_candidates':selection,'parameter_increment':factor_spec(node('rank',node('return',{'field':'close'},window=15)),
                             snapshot.payload['fields'],'one_window_increment',parent_ids=[base['candidate_id']],action='refine_local'),
             'budgets':{'fixed':8,'refine_local':8,'structural_edits':6,'random_templates':8,'random_ast':32,
                        'single_feature_edges':2,'selection_methods':3,'llm_calls':1 if llm else 0,'llm_tokens':1800},
             'source':{'path':str(Path(source_path).resolve()),'sha256':file_hash(source_path)} if source_path else None,
             'styles':{'path':str(Path(styles_path).resolve()),'sha256':file_hash(styles_path)} if styles_path else None,
             'model':{'algorithm':'ridge','alpha':1,'retrain_sessions':40,'lookback':120,'minimum_train_sessions':30,
                      'direction_and_selection':'mature_labels_in_each_outer_training_window'},
             'stop_conditions':['budget','all_arms_exhausted','changed_input_or_code','contract_error'],
             'cost_model':payload['market']['execution'],'legacy_S5_cost_bps':15,'cost_models_equivalent':False,
             'qualification_policy':'diagnostic_only; E0-E4 independent review + real JQ + unknown forward for formal release',
             'autonomous_writes_enabled':False,'live_execution_allowed':False}
    scope['protocol_hash'] = digest(scope)
    write_new(root/'RESEARCH_SCOPE.json',scope)
    for name,sha in scope['code_hashes'].items():
        source_code = (Path(__file__).parent/name).read_bytes()
        require(hashlib.sha256(source_code).hexdigest()==sha,'code changed during freeze')
        write_new(root/'code'/name,source_code)
    write_new(root/'SCREENING_PROTOCOL.json',{'scope_hash':scope['protocol_hash'],'direction':'first 100 sessions, mature labels only',
              'primary':'oriented_rank_ic','uncertainty':'399 circular moving blocks; descriptive only',
              'absolute_four_metric_FS':False,'random_AST_percentile_gate':False,'random_AST_budget':32,
              'reliable_tail_percentile':False,'nonvalid_rows':'retained with null readings','S5_denominator_changed':False})
    write_new(root/'PATCH_SCHEMA.json',{'kinds':list(KINDS),'keys':['parent_id','kind','path','value'],
              'bounds':'one node; native catalog; no source execution','consumer':'fuxi_v1.edits.apply_patch / campaign llm_patch'})
    for name in ('MODEL_BASELINE','SELECTION_AB_PROTOCOL'):
        write_new(root/(name+'.json'),{'protocol_hash':scope['protocol_hash'],'baseline':base,'candidate_pool':selection,
                  'model':scope['model'],'feature_budget':1,'method_budget':3,'same_complete_case_support':True,
                  'ordinary':'seeded identifier order, training-mature support eligibility only',
                  'screened':'rank by abs training RankIC; freeze sign from same train; never select using future folds',
                  'whole_method_axis':'separate selection study; not a single multi-feature edit','economic_claim_allowed':False})
    return scope


def run(output, *, authority_path=None):
    root = Path(output).resolve()
    scope = load_json(root/'RESEARCH_SCOPE.json')
    require(scope['protocol_hash'] == digest({k:v for k,v in scope.items() if k != 'protocol_hash'}), 'scope hash changed')
    require(scope['code_hashes'] == code_manifest(), 'code changed: new protocol/run required')
    require(file_hash(scope['input_path']) == scope['input_sha256'], 'input changed')
    for key in ('source','styles'):
        if scope[key]:
            require(file_hash(scope[key]['path']) == scope[key]['sha256'], key+' evidence changed')
    payload = load_json(scope['input_path'])
    snapshot, labels = Snapshot(payload['snapshot']), None
    labels = Labels(payload['labels'],snapshot)
    snapshot._score_cache = {}
    request, seed = scope['request'], scope['seed']
    start, end = request['research_start'], request['research_end']
    if authority_path:
        from .evidence import EvidenceStore
        store = EvidenceStore(authority_path)
        try:
            store.expose('architecture_'+scope['protocol_hash'],scope['scope'],'research')
            store.append('architecture_study_scope',{'protocol_hash':scope['protocol_hash'],'scope':scope['scope'],'formal_eligibility':False})
        finally:
            store.close()
    def stage(name, function):
        destination = root/(name+'.json')
        if destination.exists():
            saved = load_json(destination)
            require(saved['protocol_hash'] == scope['protocol_hash'] and saved['result_hash'] == digest(saved['result']), 'stage result changed')
            return saved['result']
        print(json.dumps({'work_package_stage':name,'status':'running'},ensure_ascii=False),flush=True)
        value = function()
        write_new(destination,{'protocol_hash':scope['protocol_hash'],'result_hash':digest(value),'result':value})
        return value
    base = scope['baseline']
    from .dsl import evaluate_ast
    base_scores = evaluate_ast(base['canonical_ast'],snapshot)
    candidates = scope['fixed_candidates']
    audit = stage('DATA_DEPENDENCIES',lambda:dependency_scope(snapshot,candidates+[base],start,end,payload['market']))
    local, local_receipt = refine_local(snapshot.payload['fields'],candidates,8)
    random = generate(snapshot.payload['fields'],arm='random',count=8,seed=seed)
    typed, typed_receipt = random_population(snapshot.payload['fields'],32,seed+1)
    all_candidates = candidates+local+random+typed
    # Six declared edits of one parent; no fabricated memory observations.
    edits = [{'parent_id':base['candidate_id'],'kind':kind,'path':path,'value':value} for kind,path,value in [
        ('window',[0],15),('field',[0,0],'open' if 'open' in snapshot.payload['fields'] else 'close'),
        ('operator',[0],'delta'),('wrap',[],'neg'),('unwrap',[],None),('binary',[],{'op':'sub','field':'volume' if 'volume' in snapshot.payload['fields'] else 'close'})]]
    edited, edit_rejections = [], []
    for edit in edits:
        try:
            edited.append(apply_patch(base,edit,snapshot.payload['fields']))
        except ValueError as exc:
            edit_rejections.append({'edit':edit,'reason':str(exc)})
    all_candidates += edited
    def screening():
        rows, scores = [], {}
        for i,c in enumerate(all_candidates):
            panel = evaluate_ast(c['canonical_ast'],snapshot)
            scores[c['candidate_id']] = panel
            train = training_reading(panel,snapshot,labels,snapshot.clocks[start],snapshot.sessions[snapshot.start:100])
            reading = diagnose(panel,snapshot,labels,start,end,payload['research_clock'],direction=train['direction'],seed=seed)
            row = {'candidate_id':c['candidate_id'],'formula_hash':c['formula_hash'],'action':c['action'],
                   'training':train,**reading}
            rows.append(row)
            destination = root/'cards'/(c['candidate_id']+'.json')
            if not destination.exists():
                write_new(destination,card(c,reading['metric'],scope,request,source_path='SCREENING_READINGS.json',training=train))
            if i % 10 == 0:
                print(json.dumps({'work_package_stage':'SCREENING_READINGS','evaluated':i+1,'budget':len(all_candidates)}),flush=True)
        pool = [r['oriented_rank_ic'] for r in rows if r['action']=='calibration_random_null']
        ranking = sorted(rows,key=lambda r:(r['training']['direction'] is None,
                         -abs(r['training']['mean_rank_ic']) if r['training']['mean_rank_ic'] is not None else 0.,r['candidate_id']))
        return {'rows':rows,'training_only_ranking':[{'rank':i+1,'candidate_id':r['candidate_id'],
                'training_mean_rank_ic':r['training']['mean_rank_ic'],'research_eval_status':r['metric']['eval_status']} for i,r in enumerate(ranking)],
                'generation':{'local':local_receipt,'typed_random':typed_receipt,'edit_rejections':edit_rejections},
                'actual_candidate_requests':len(rows),'unique_formula_count':len({r['formula_hash'] for r in rows}),
                'duplicate_requests_preserved_in_denominator':True,'random_AST_summary':baseline_summary(pool),
                'random_AST_not_statistical_null':True,'reliable_p95':False,'formal_hard_gate':None,
                'five_state_counts':{state:sum(r['metric']['eval_status']==state for r in rows) for state in
                    ('valid','data_insufficient','error','precheck_rejected','not_evaluated')}}
    readings = stage('SCREENING_READINGS',screening)
    def memory_ab():
        lookup = {c['candidate_id']:c for c in all_candidates}
        comparisons, observations = [], []
        for child in edited:
            a = evaluate_ast(child['canonical_ast'],snapshot)
            paired = paired_parent(a,base_scores,snapshot,labels,start,end,snapshot.clocks[end])
            train = paired_parent(a,base_scores,snapshot,labels,snapshot.sessions[snapshot.start],snapshot.sessions[99],snapshot.clocks[start])
            for motif in ast_diff_motifs(base['canonical_ast'],child['canonical_ast']):
                # One real observation per edit, never replicated to force a veto.
                if train['status'] == 'valid':
                    observations.append((motif,train['mean_ic_increment'] > 0))
            comparisons.append({'candidate_id':child['candidate_id'],'training':train,'later_research':paired})
        vetoes = motif_vetoes(motif_ledger(observations))
        gate = motif_gate(edited,lambda cid:base['canonical_ast'] if cid==base['candidate_id'] else lookup.get(cid,{}).get('canonical_ast'),vetoes,mode='enforce')
        rejected_ids = {r['candidate_id'] for r in gate['hits']}
        misreject = [r['candidate_id'] for r in comparisons if r['candidate_id'] in rejected_ids and r['later_research']['status']=='valid' and r['later_research']['mean_ic_increment']>0]
        return {'comparisons':comparisons,'motif_observations':len(observations),'vetoes':vetoes,'gate':gate,
                'matched_edit_count':len(edited),'no_memory_evaluations':len(edited),'memory_evaluations':len(gate['kept']),
                'false_rejected_later_positive':misreject,'power_to_establish_memory_value':False,
                'interpretation':'small paired process pilot; zero veto is a valid result; no artificial hit target'}
    memory = stage('MEMORY_AB',memory_ab)
    def source_trial():
        if not scope['source']:
            return {'eval_status':'not_evaluated','reason':'source_not_supplied','formal_use_allowed':False}
        from .announcement_feature import channel
        panel, manifest = channel(scope['source']['path'],snapshot)
        write_new(root/'SOURCE_EVIDENCE_INDEX.json',manifest) if not (root/'SOURCE_EVIDENCE_INDEX.json').exists() else None
        metric = factor_statistics({d:r for d,r in panel.items() if start<=d<=end},snapshot,labels,DEFAULT_POLICY,payload['research_clock'])
        try:
            comparison = paired_scores(panel,base_scores,start,end)
            common = paired_parent(panel,base_scores,snapshot,labels,start,end,snapshot.clocks[end])
        except ValueError as exc:
            comparison = common = {'status':'data_insufficient','reason':str(exc),'n_pairs':0}
        return {'manifest':{k:v for k,v in manifest.items() if k!='evidence'},'single_channel_evaluation':metric,
                'single_channel_vs_price_rank':comparison,'paired_common_support':common,
                'multi_channel_candidate_evaluated':False,'formal_use_allowed':False,
                'stop_reason':'historical_publication_clock_and_revision_vintage_unverified',
                'panel':panel}
    source_result = stage('SOURCE_TRIAL',source_trial)
    def attribution():
        if not scope['styles']:
            return {'eval_status':'data_insufficient','reason':'PIT_industry_and_float_market_cap_not_supplied',
                    'raw_residual_comparison':None,'investment_neutralization_certified':False}
        from .attribution import style_attribution
        panels, provenance = pit_styles(load_json(scope['styles']['path']),snapshot)
        return {'eval_status':'valid','provenance':provenance,'result':style_attribution(
            {d:r for d,r in base_scores.items() if start<=d<=end},panels,labels,snapshot,payload['research_clock'])}
    styles = stage('PIT_STYLE_ATTRIBUTION',attribution)
    parameter = scope['parameter_increment']
    parameter_scores = evaluate_ast(parameter['canonical_ast'],snapshot)
    single = stage('FEATURE_INCREMENT',lambda:compare_methods(payload,base_scores,{parameter['formula_hash']:parameter_scores},request,seed=seed))
    selection_panels = {c['formula_hash']:evaluate_ast(c['canonical_ast'],snapshot) for c in scope['selection_candidates']}
    selection = stage('SELECTION_AB',lambda:compare_methods(payload,base_scores,selection_panels,request,seed=seed))
    def accounts():
        ir, rank = account_for(base_scores,snapshot,payload['market'],request,identity='frozen_rank_baseline')
        equal = {d:{s:1. if v is not None else None for s,v in cross.items()} for d,cross in base_scores.items()}
        equal_ir, equal_account = account_for(equal,snapshot,payload['market'],request,identity='pool_equal_benchmark',
                                             top_n=max(map(len,snapshot.universe.values())))
        pressure = []
        from .simulation import run_account
        from .portfolio import compile_strategy, strategy_spec
        for cash_scale,cost,participation in ((.1,1.,1.),(1.,1.,1.),(10.,1.,1.),(1.,2.,1.),(1.,1.,.5)):
            market = deepcopy(payload['market'])
            from decimal import Decimal
            market['execution']['initial_cash'] = str(Decimal(market['execution']['initial_cash'])*Decimal(str(cash_scale)))
            market['market_hash'] = digest({k:v for k,v in market.items() if k!='market_hash'})
            pressure_ir = compile_strategy(strategy_spec('stress_baseline',top_n=6,max_weight=1/6,rebalance_sessions=5),
                                           base_scores,snapshot,digest(market['execution']))
            account = run_account(pressure_ir,market,start=start,end=end,cost_multiplier=cost,participation_multiplier=participation)
            pressure.append({'cash_scale':cash_scale,'cost_multiplier':cost,'participation_multiplier':participation,
                             'account_status':account['eval_status'],'formal_evidence':account['formal_evidence'],
                             'gaps':account['gaps'],'attribution':account_diagnostics(account,account)})
        return {'baseline_ir':ir,'rank_account':rank,'equal_ir':equal_ir,'pool_equal_account':equal_account,
                'reference_arithmetic':reconcile(rank,payload['market']),
                'relative_attribution':account_diagnostics(rank,equal_account),'stress':pressure}
    account_result = stage('ACCOUNTS_AND_STRESS',accounts)
    def platform():
        from .jq import compile_diagnostic
        # Provider adapter supports close only: do not compare volume cells as close.
        narrow = deepcopy(snapshot.payload)
        narrow['fields'] = {'close':narrow['fields']['close']}
        narrow['observations'] = [r for r in narrow['observations'] if r['field']=='close']
        narrow_snapshot = Snapshot(narrow)
        narrow_scores = evaluate_ast(base['canonical_ast'],narrow_snapshot)
        require(narrow_scores == base_scores,'narrow diagnostic field projection changed scores')
        text, compiler = compile_diagnostic(base,narrow_snapshot,narrow_scores)
        if not (root/'jq_diagnostic.py').exists():
            write_new(root/'jq_diagnostic.py',text)
        from .portfolio import target_weights, strategy_spec, validate_strategy
        decisions = [r for r in account_result['baseline_ir']['decisions'] if start<=r['signal_session']<=end][-3:]
        expected_rows = []
        for decision in decisions:
            weights = target_weights(decision['scores'],account_result['baseline_ir']['strategy']) or {}
            for code,value in decision['scores'].items():
                expected_rows.append({'signal_session':decision['signal_session'],'instrument':code,'factor_value':value,
                                      'target_weight':weights.get(code,0.) if weights else None,
                                      'decision_time':decision['decision_time'],'execution_session':decision['execution_session']})
        expected = {'protocol_hash':scope['protocol_hash'],'rows':expected_rows,
                    'weight_scope':'score-only intended weights before turnover, frozen holdings and actual account constraints',
                    'formal_account_parity':False}
        if not (root/'EXPECTED_JQ_DIAGNOSTIC.json').exists():
            write_new(root/'EXPECTED_JQ_DIAGNOSTIC.json',expected)
        import inspect
        export_text = text.decode('utf-8').replace('"session": session, "n_pairs": len(pairs),',
                     '"session": session, "factor_values": actual, "n_pairs": len(pairs),',1)
        export_text += '\n\n'+inspect.getsource(strategy_spec)+'\n'+inspect.getsource(validate_strategy)+'\n'+inspect.getsource(target_weights)
        export_text += '\nEXPORT_PROTOCOL = '+repr(scope['protocol_hash'])+'\nEXPORT_SESSIONS = '+repr([d['signal_session'] for d in decisions])+'\n'
        export_text += '''
def export_observed_values(get_price_fn):
    rows = []
    spec = strategy_spec("diagnostic", top_n=6, max_weight=1/6, rebalance_sessions=5)
    for day in EXPORT_SESSIONS:
        report = provider_diagnostic(get_price_fn, day)
        scores = report["factor_values"]
        weights = target_weights(scores, spec) or {}
        next_day = FROZEN["snapshot"]["sessions"][FROZEN["snapshot"]["sessions"].index(day)+1]
        for code, value in scores.items():
            rows.append({"signal_session":day,"instrument":code,"factor_value":value,
                         "target_weight":weights.get(code,0.) if weights else None,
                         "decision_time":FROZEN["snapshot"]["decision_times"][day],"execution_session":next_day})
    return {"schema_version":"fuxi.jq_observed_diagnostic.1","protocol_hash":EXPORT_PROTOCOL,
            "observed_on_jq":True,"rows":rows,"account_parity":False,
            "raw_export_sha256":digest(rows),"export_evidence":"operator must preserve the real JQ export and run identity"}

# Run in JQ research only: print(json.dumps(export_observed_values(get_price),ensure_ascii=False))
'''
        if not (root/'jq_export_adapter.py').exists():
            write_new(root/'jq_export_adapter.py',export_text.encode('utf-8'))
        return {'eval_status':'not_evaluated','compiler':compiler,'purpose':'diagnostic_only',
                'shortlist':[base['candidate_id']],'local_ir_hash':account_result['baseline_ir']['ir_hash'],
                'expected_factor_values':{d:r for d,r in base_scores.items() if start<=d<=end},
                'security_mapping':{s:s for s in audit['ever_eligible']},'expected_weights_contract':'native StrategyIR ties prorata',
                'expected_local_orders':account_result['rank_account']['orders'],
                'required_real_export_fields':['signal_session','decision_time','execution_session','instrument','factor_value','target_weight','nav','cash','holdings','receivables'],
                'null_policy':'retain null; unknown never zero','absolute_tolerance':1e-8,'n_pairs':0,
                'observed_on_jq':False,'jq_candidate_allowed':False,'platform_mode':'user_manual_jq_upload_return',
                'blockers':['independent_data_assurance','E0-E4_not_formally_accepted','real_JQ_export_missing']}
    parity = stage('DIAGNOSTIC_PARITY',platform)
    def llm_stage():
        if not scope['budgets']['llm_calls']:
            return {'eval_status':'not_evaluated','reason':'LLM_call_not_in_this_frozen_budget'}
        # No external message is sent; this is the explicitly bounded factor-generation adapter.
        proposals, receipt = llm_patches(snapshot.payload['fields'],[base],[],count=3,max_tokens=1800,timeout=40)
        return {'proposals':proposals,'receipt':receipt,'economic_claim_allowed':False}
    llm = stage('LLM_PATCH_TRIAL',llm_stage)
    status = {f'B{i:02d}':{'engineering':'implemented','scientific_acceptance':'diagnostic_only'} for i in range(1,13)}
    for key, reasons in {'B03':audit['blocking_gaps'],'B04':['PIT_industry/float_market_cap required'],
                         'B08':['source historic timing/revisions unverified'],'B11':parity['blockers'],
                         'B12':['formal_JQ_receipt_missing','future_unknown_samples_not_observed']}.items():
        status[key]['scientific_acceptance'] = 'pending_external_or_future_evidence'
        status[key]['blockers'] = reasons
    status['B03']['engineering'] = 'dependency_scope_and_reference_ready_certification_pending'
    status['B04']['engineering'] = 'PIT_adapter_and_attribution_ready_source_missing' if not scope['styles'] else 'diagnostic_attribution_ready'
    status['B08']['engineering'] = 'source_adapter_and_sensitivity_trial_ready_formal_use_stopped'
    status['B11']['engineering'] = 'diagnostic_export_package_ready_real_observation_pending'
    status['B12']['engineering'] = 'draft_prepared_existing_acceptance_and_order_guards_reused_not_activated'
    status['B06']['scientific_acceptance'] = 'small_pilot_inconclusive'
    status['B07']['scientific_acceptance'] = 'diagnostic_population_only_no_tail_claim'
    status['B10']['scientific_acceptance'] = 'historical_method_diagnostic_no_enforcement'
    forward = {'schema_version':'fuxi.forward_draft.1','status':'not_activated','candidate_id':None,
               'draft_created_at':datetime.now().astimezone().isoformat(),'protocol_hash':scope['protocol_hash'],
               'window':'future sessions after formal freeze; dates unresolved until real JQ acceptance',
               'registered_looks':[60,120],'family_alpha':.05,'alpha_per_look':.025,'candidate_budget':1,
               'future_feedback_may_revise_same_candidate':False,'cost_model':scope['cost_model'],
               'required_receipts':['research_acceptance','jq_parity_real','data_independent_assurance','forward_scope_sealed'],
               'monitors':['data_scope_hash','signal_clock','order_unknown','cash_holdings_rights_reconciliation'],
               'unknown_order_policy':'stop new submissions; reconcile receipt; no automatic resubmit',
               'live_execution_allowed':False,'autonomous_writes_enabled':False}
    if not (root/'FORWARD_DRAFT.json').exists():
        write_new(root/'FORWARD_DRAFT.json',forward)
    import csv
    legacy_path = Path(__file__).resolve().parents[1]/'data/unified_factor_pool.csv'
    if legacy_path.exists():
        with legacy_path.open(encoding='utf-8-sig',newline='') as stream:
            legacy_rows = list(csv.DictReader(stream))
        projection = legacy_projection([dict(r,factor_id=r['name']) for r in legacy_rows],file_hash(legacy_path))
        if not (root/'LEGACY_LIBRARY_INDEX.json').exists():
            write_new(root/'LEGACY_LIBRARY_INDEX.json',{'source_path':str(legacy_path),'entries':projection,
                'native_mapping':'no automatic name/formula equivalence; each mapping requires reviewed AST and source lineage'})
    report = {'protocol_hash':scope['protocol_hash'],'status':status,'scope':scope['scope'],
              'code_hashes':scope['code_hashes'],'candidate_requests':readings['actual_candidate_requests'],
              'reference_reconciliation':account_result['reference_arithmetic'],'cards':len(query(root)),
              'data_certification_granted':False,'new_JQ_qualification_count':0,'E5_observations':0,
              'live_execution_allowed':False,'autonomous_writes_enabled':False}
    require(code_manifest() == scope['code_hashes'], 'code changed during study')
    destination = root/'WORK_PACKAGE_STATUS.json'
    if not destination.exists():
        write_new(destination,report)
    return report
