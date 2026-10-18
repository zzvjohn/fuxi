"""Frozen follow-up diagnostics and evidence requests; never an eligibility engine."""
from pathlib import Path
from statistics import mean
from collections import Counter
from copy import deepcopy
import hashlib
from .contracts import digest, load_json, require, timestamp
from .campaign import write_new
from .artifacts import code_manifest
from .data import Snapshot, Labels
from .dsl import evaluate_ast, ranks
from .evaluation import correlation
from .research_screening import training_reading, orient
from .learning import paired_parent, ast_diff_motifs, motif_ledger, motif_vetoes, motif_gate
from .edits import apply_patch, refine_local


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(history, output, *, evidence_files=(), styles=None):
    """Freeze before reading diagnostic outcomes; old batches remain immutable."""
    root, old = Path(output).resolve(), Path(history).resolve()
    scope = load_json(old/'RESEARCH_SCOPE.json')
    payload = load_json(scope['input_path'])
    snapshot = Snapshot(payload['snapshot'])
    days = [d for d in snapshot.sessions if scope['request']['research_start'] <= d <= scope['request']['research_end']]
    require(len(days) >= 6, 'time-separated diagnostics need at least six sessions')
    boundary = len(days)//2
    files = {str(Path(p).resolve()):file_hash(p) for p in evidence_files}
    inputs = [old/'RESEARCH_SCOPE.json', Path(scope['input_path']), old/'SCREENING_READINGS.json',
              old/'ACCOUNTS_AND_STRESS.json', old/'FEATURE_INCREMENT.json', old/'SELECTION_AB.json', old/'SOURCE_TRIAL.json']
    if styles:
        inputs.append(Path(styles))
    body = {'schema_version':'fuxi.architecture_completion.1', 'history_root':str(old),
            'input_path':scope['input_path'], 'inputs':{str(p.resolve()):file_hash(p) for p in inputs},
            'evidence_files':files, 'styles_path':str(Path(styles).resolve()) if styles else None,
            'request':scope['request'], 'baseline':scope['baseline'],
            'memory_train':[days[0],days[boundary-1]], 'memory_test':[days[boundary],days[-1]],
            'ranking_windows':[[days[len(days)*i//3],days[len(days)*(i+1)//3-1]] for i in range(3)],
            'top_k':6, 'code_hashes':code_manifest(), 'online_LLM_budget':0,
            'known_source_budget':1, 'platform_mode':'user_manual_upload_return',
            'formal_claim_allowed':False, 'future_observation_count':0}
    body['protocol_hash'] = digest(body)
    write_new(root/'COMPLETION_PROTOCOL.json',body)
    for name in body['code_hashes']:
        write_new(root/'code'/name,(Path(__file__).parent/name).read_bytes())
    return body


def evidence_matrix(scope, payload, evidence_files):
    """Every dependency has a measured reading and a primary-evidence gap."""
    snapshot = Snapshot(payload['snapshot'])
    start, end = scope['request']['research_start'],scope['request']['research_end']
    old = load_json(Path(scope['history_root'])/'DATA_DEPENDENCIES.json')['result']
    bars = [r for r in payload['market']['bars'] if start <= r['session'] <= end
            and r['instrument'] in old['holdings_and_rights_closure']]
    require(bars, 'nonempty dependency bars required')
    require(all(file_hash(p)==h for p,h in evidence_files.items()),'primary evidence changed')
    index={(r['session'],r['instrument']):r for r in bars}
    keys=[(d,c) for d in snapshot.sessions if start<=d<=end for c in old['holdings_and_rights_closure']]
    require(len(keys)==old['potential_holding_cells'],'holding closure denominator changed')
    channels = {'prices':['close'], 'halts':['paused'], 'limits':['up_limit','down_limit'],
                'corporate_actions':['corporate_action_known'], 'lifecycle_and_delist':[],
                'security_identity':[], 'sellability':['paused','down_limit'],
                'cash_share_receivables':[], 'fees_execution':[], 'PIT_revision_vintage':[],
                'PIT_industry_float_cap':[]}
    rows=[]
    for name, fields in channels.items():
        required = len(keys)*len(fields)
        available = sum(index.get(k,{}).get(f) is not None and
                        (f != 'corporate_action_known' or index[k][f] is True) for k in keys for f in fields)
        rows.append({'channel':name,'required_cells':required if fields else None,
                     'available_cells':available if fields else None,
                     'unknown_cells':required-available if fields else None,
                     'snapshot_reading':'computed' if fields else 'requires_source_adjudication',
                     'evidence_file_index':evidence_files,
                     'primary_evidence_adjudicated':False,
                     'independent_review':'not_supplied', 'accepted':False,
                     'required_attestation':'primary timestamps/revisions and applicable account scope',
                     'blocking':True})
    return {'schema_version':'fuxi.architecture_evidence_matrix.1','snapshot_hash':snapshot.hash,
            'market_hash':payload['market']['market_hash'],'dependency_scope_hash':old['scope_hash'],
            'window':[start,end], 'holdings_closure':old['holdings_and_rights_closure'],
            'rows':rows,'review_request':{'generator_identity':'fuxi_architecture_completion',
            'reviewer':None,'reviewed_at':None,'account_scope':'ever_eligible_holdings_and_entitlements',
            'snapshot_hash':snapshot.hash,'market_digest':digest(payload['market']),
            'native_import_action':'operate import-data-assurance','automatic_acceptance':False},
            'investment_eligibility_certified':False}


def paired_edit_trial(scope, snapshot, labels):
    parent=scope['baseline']
    fields=snapshot.payload['fields']
    definitions=[('window',[0],15),('field',[0,0],'open' if 'open' in fields else 'close'),
                 ('operator',[0],'delta'),('wrap',[],'neg'),('unwrap',[],None),
                 ('binary',[],{'op':'sub','field':'volume' if 'volume' in fields else 'close'})]
    children, rejected=[],[]
    for kind,path,value in definitions:
        edit={'parent_id':parent['candidate_id'],'kind':kind,'path':path,'value':value}
        try: children.append(apply_patch(parent,edit,fields))
        except ValueError as exc: rejected.append({'edit':edit,'reason':str(exc)})
    local, attempts=refine_local(fields,[parent],2)
    requests=len(children)+len(local)
    children=list({c['formula_hash']:c for c in children+local}.values())
    start=scope['request']['research_start']
    training=[d for d in snapshot.sessions[snapshot.start:] if d < start]
    parent_scores=evaluate_ast(parent['canonical_ast'],snapshot)
    direction=training_reading(parent_scores,snapshot,labels,snapshot.clocks[start],training)
    if direction['direction'] is None:
        return {'eval_status':'data_insufficient','reason':'parent_training_direction_unidentified','n_pairs':0}
    parent_scores=orient(parent_scores,direction['direction'])
    rows,observations=[],[]
    train_end=scope['memory_train'][1]
    for child in children:
        scores=orient(evaluate_ast(child['canonical_ast'],snapshot),direction['direction'])
        before=paired_parent(scores,parent_scores,snapshot,labels,*scope['memory_train'],snapshot.clocks[train_end])
        after=paired_parent(scores,parent_scores,snapshot,labels,*scope['memory_test'],snapshot.clocks[scope['memory_test'][1]])
        motifs=ast_diff_motifs(parent['canonical_ast'],child['canonical_ast'])
        if before['status']=='valid':
            observations += [(m,before['mean_ic_increment']>0) for m in motifs]
        rows.append({'child':child,'axis':'one_declared_AST_edit','direction_policy':'same frozen parent sign',
                     'memory_training':before,'time_outside_memory':after,'motifs':motifs})
    ledger=motif_ledger(observations)
    vetoes=motif_vetoes(ledger)
    gated=motif_gate(children,lambda pid:parent['canonical_ast'] if pid==parent['candidate_id'] else None,vetoes,mode='enforce')
    kept={r['candidate_id'] for r in gated['kept']}
    wrong=[r['child']['candidate_id'] for r in rows if r['child']['candidate_id'] not in kept
           and r['time_outside_memory']['status']=='valid' and r['time_outside_memory']['mean_ic_increment']>0]
    require(rows and sum(r['memory_training']['n_pairs'] for r in rows)>0,'zero paired edit observations')
    return {'eval_status':'valid','training_direction':direction,'rows':rows,'rejections':rejected,
            'local_generation':attempts,'edit_requests':requests,'motif_ledger':ledger,'vetoes':vetoes,
            'no_memory_count':len(children),'with_memory_count':len(kept),
            'evaluation_savings_count':len(children)-len(kept),'false_rejection_ids':wrong,
            'n_pairs':sum(r['time_outside_memory']['n_pairs'] for r in rows),
            'compatibility':{'snapshot_hash':snapshot.hash,'scope_hash':scope['protocol_hash'],
                             'training_window':scope['memory_train'],'test_window':scope['memory_test']},
            'LLM_prompt_ab':'not_observed_online; offline AST edits cannot replace online prompt comparison',
            'LLM_requests':0,'new_LLM_cost':0,'economic_claim_allowed':False,
            'decision':'diagnostic_only; no production memory mode change'}


def ranking_trial(scope, snapshot, labels, candidates):
    """Common support across all formulas avoids ranking a coverage advantage."""
    panels={c['candidate_id']:evaluate_ast(c['canonical_ast'],snapshot) for c in candidates}
    start,end=scope['request']['research_start'],scope['request']['research_end']
    training=[d for d in snapshot.sessions[snapshot.start:] if d<start]
    train={key:training_reading(panel,snapshot,labels,snapshot.clocks[start],training) for key,panel in panels.items()}
    train_order=sorted(panels,key=lambda k:(train[k]['mean_rank_ic'] is None,
                      -abs(train[k]['mean_rank_ic']) if train[k]['mean_rank_ic'] is not None else 0.,k))
    windows=[]
    for a,b in scope['ranking_windows']:
        values={k:[] for k in panels};support=[]
        for d in snapshot.sessions:
            if not a<=d<=b: continue
            targets={s:labels.value(d,s,snapshot.clocks[b]) for s in snapshot.universe[d]}
            common=[s for s in targets if targets[s] is not None and all(panels[k][d][s] is not None for k in panels)]
            if len(common)<5: continue
            target_ranks=list(ranks({s:targets[s] for s in common}).values())
            support.append({'session':d,'instruments':common})
            for k,panel in panels.items():
                ic=correlation(list(ranks({s:panel[d][s] for s in common}).values()),target_ranks)
                if ic is not None and train[k]['direction'] is not None: values[k].append(train[k]['direction']*ic)
        readings={k:mean(v) if v else None for k,v in values.items()}
        order=sorted(readings,key=lambda k:(readings[k] is None,-readings[k] if readings[k] is not None else 0.,k))
        common_rank=[k for k in train_order if readings[k] is not None]
        stability=correlation([train_order.index(k) for k in common_rank],[order.index(k) for k in common_rank]) if len(common_rank)>=3 else None
        windows.append({'window':[a,b],'n_sections':len(support),'n_pairs':sum(len(r['instruments']) for r in support),
                        'support_hash':digest(support),'oriented_rank_ic':readings,'ranking':order,
                        'train_to_window_rank_correlation':stability,
                        'missed_top_k':[k for k in order[:scope['top_k']] if readings[k] is not None and k not in train_order[:scope['top_k']]]})
    # Pairwise redundancy reports its own support, independently of global rank support.
    redundancy=[];rank_cache={}
    def ranked(key,day,common):
        cache_key=(key,day,tuple(common))
        if cache_key not in rank_cache:
            rank_cache[cache_key]=list(ranks({s:panels[key][day][s] for s in common}).values())
        return rank_cache[cache_key]
    identifiers=sorted(panels)
    for i,left in enumerate(identifiers):
        for right in identifiers[i+1:]:
            corr=[];n=0
            for d in snapshot.sessions:
                if not start<=d<=end:continue
                common=[s for s in snapshot.universe[d] if panels[left][d][s] is not None and panels[right][d][s] is not None]
                if len(common)<5:continue
                r=correlation(ranked(left,d,common),ranked(right,d,common))
                if r is not None:corr.append(r);n+=len(common)
            redundancy.append({'left':left,'right':right,'n_sections':len(corr),'n_pairs':n,
                               'mean_rank_correlation':mean(corr) if corr else None})
    return {'eval_status':'valid' if any(w['n_pairs']>0 for w in windows) else 'data_insufficient',
            'n_candidates':len(panels),'candidate_requests_preserved':len(candidates),
            'training_order':train_order,'training_readings':train,'windows':windows,'redundancy':redundancy,
            'formal_selection_enabled':False,'reliable_random_tail_claim':False,
            'all_candidate_support_required':True,'B10_economic_decision':'native paired account guard retained'}


def source_ablation(scope, snapshot, labels, source):
    panel=source.get('panel')
    if not panel:return {'eval_status':'not_evaluated','reason':'no_source_panel','n_pairs':0}
    parent=evaluate_ast(scope['baseline']['canonical_ast'],snapshot)
    base,added,combined={}, {}, {}
    for d,cross in parent.items():
        common=[s for s in cross if cross[s] is not None and panel.get(d,{}).get(s) is not None]
        p=ranks({s:cross[s] for s in common});q=ranks({s:panel[d][s] for s in common})
        base[d]={s:p.get(s) for s in cross};added[d]={s:q.get(s) for s in cross}
        combined[d]={s:(p[s]+q[s])/2 if s in p and p[s] is not None and q[s] is not None else None for s in cross}
    a,b=scope['request']['research_start'],scope['request']['research_end']
    paired=paired_parent(combined,base,snapshot,labels,a,b,snapshot.clocks[b])
    source_only=paired_parent(added,base,snapshot,labels,a,b,snapshot.clocks[b])
    return {'eval_status':paired['status'],'n_pairs':paired['n_pairs'],
            'base_plus_source_vs_base_on_same_support':paired,'source_alone_vs_price_on_same_support':source_only,
            'combined_rule':'mean of within-common-support ranks, weights frozen at 1/2',
            'old_sparse_trial':{k:source.get(k) for k in ('manifest','stop_reason')},
            'formal_use_allowed':False,'decision':'stop_formal_use_historical_clocks_and_revision_unverified',
            'single_channel_stripping_executed':True,'no_zero_fill':True,'causal_independence_proven':False}


def pit_diagnostic(scope, snapshot, labels, payload):
    if not scope.get('styles_path'):
        return {'eval_status':'data_insufficient','reason':'historical_PIT_industry_free_float_cap_not_supplied',
                'native_consumer':'fuxi_v1.dependency_audit.pit_styles -> attribution.style_attribution',
                'current_industry_map_is_not_historical_archive':True,'n_pairs':0}
    from .dependency_audit import pit_styles
    from .attribution import style_attribution
    spec=load_json(scope['styles_path'])
    primary=spec.get('primary_evidence_files') or {}
    require(primary and all(scope['evidence_files'].get(str(Path(p).resolve()))==h for p,h in primary.items()),
            'PIT styles must bind actually reviewed primary files, not just an asserted hash')
    require(spec['primary_evidence_sha256']==digest(primary),'PIT primary manifest hash mismatch')
    panels,provenance=pit_styles(spec,snapshot)
    a,b=scope['request']['research_start'],scope['request']['research_end']
    scores=evaluate_ast(scope['baseline']['canonical_ast'],snapshot)
    result=style_attribution({d:r for d,r in scores.items() if a<=d<=b},panels,labels,snapshot,payload['research_clock'])
    return {'eval_status':'valid' if any(r['status']=='valid' for r in result['sections']) else 'data_insufficient',
            'result':result,'provenance':provenance,'n_pairs':result['n_pairs'],'residual_used_for_trading':False}


def run(output):
    root=Path(output).resolve();scope=load_json(root/'COMPLETION_PROTOCOL.json')
    require(scope['protocol_hash']==digest({k:v for k,v in scope.items() if k!='protocol_hash'}),'completion scope changed')
    require(scope['code_hashes']==code_manifest(),'new code requires new frozen batch')
    require(all(file_hash(p)==h for p,h in {**scope['inputs'],**scope['evidence_files']}.items()),'completion input changed')
    payload=load_json(scope['input_path']);snapshot=Snapshot(payload['snapshot']);labels=Labels(payload['labels'],snapshot)
    snapshot._score_cache={}
    old=Path(scope['history_root'])
    def stage(name,fn):
        path=root/(name+'.json')
        if path.exists():
            saved=load_json(path)
            require(saved['protocol_hash']==scope['protocol_hash'] and digest(saved['result'])==saved['result_hash'],'checkpoint changed')
            print(name,'cached',flush=True)
            return saved['result']
        print(name,'running',flush=True);value=fn()
        write_new(path,{'protocol_hash':scope['protocol_hash'],'result_hash':digest(value),'result':value})
        return value
    matrix=stage('EVIDENCE_MATRIX',lambda:evidence_matrix(scope,payload,scope['evidence_files']))
    edits=stage('PAIRED_EDIT_MEMORY',lambda:paired_edit_trial(scope,snapshot,labels))
    cards=sorted((old/'cards').glob('*.json'))
    # Request-level card list retains duplicates; panels key by candidate identity.
    candidates=[load_json(p) for p in cards]
    ranking=stage('RANKING_STABILITY',lambda:ranking_trial(scope,snapshot,labels,candidates))
    source=stage('SOURCE_ABLATION',lambda:source_ablation(scope,snapshot,labels,load_json(old/'SOURCE_TRIAL.json')['result']))
    styles=stage('PIT_ATTRIBUTION',lambda:pit_diagnostic(scope,snapshot,labels,payload))
    def accounts():
        from .account_reference import reconcile, account_diagnostics
        inputs=load_json(old/'ACCOUNTS_AND_STRESS.json')['result']
        increment=load_json(old/'FEATURE_INCREMENT.json')['result']
        selection=load_json(old/'SELECTION_AB.json')['result']
        rows=[]
        for study,value in [('feature',increment),('selection',selection)]:
            for method,fitted in value['outputs'].items():
                rows.append({'study':study,'method':method,'reference':reconcile(fitted['account'],payload['market']),
                             'relative_attribution':account_diagnostics(fitted['account'],value['outputs']['base']['account'])})
        return {'native_studies':rows,'feature_comparisons':increment['comparisons'],
                'selection_comparisons':selection['comparisons'],'stress':inputs['stress'],
                'positive_economic_pairs':sum(c['n_pairs'] for v in (increment,selection) for c in v['comparisons'].values()
                   if c['eval_status']=='valid'),'economic_claim_allowed':False,
                'decision':'data_insufficient means unmeasured; no economic reject fabricated'}
    account=stage('ACCOUNT_RECONCILIATION',accounts)
    return {'protocol_hash':scope['protocol_hash'],'engineering_diagnostics_executed':True,
            'paired_edit_count':len(edits.get('rows',[])),'ranking_candidates':ranking['n_candidates'],
            'source_pairs':source['n_pairs'],'PIT_status':styles['eval_status'],
            'account_economic_pairs':account['positive_economic_pairs'],
            'independent_data_review_present':False,'real_platform_status':'skipped_by_user_manual_return_pending',
            'future_status':'not_activated_requires_formal_receipts',
            'all_scientific_acceptance_complete':False,'live_execution_allowed':False}
