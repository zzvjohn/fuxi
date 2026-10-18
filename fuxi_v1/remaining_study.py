"""Source-bound remediation, matched model studies and honest acceptance readiness.

Effective-date vendor styles are explicitly a research proxy. They never grant
PIT certification or neutralize production signals. No external calls occur here.
"""
from pathlib import Path
from math import log
from collections import Counter
from statistics import mean
from datetime import datetime
from decimal import Decimal
from .contracts import load_json, digest, require, finite
from .datasets import file_hash
from .artifacts import code_manifest
from .campaign import write_new
from .data import snapshot_from_payload, Labels
from .search import node, factor_spec
from .dsl import evaluate_ast
from .incremental_study import compare_methods, account_for
from .research_evaluation import paired_accounts, paired_scores
from .account_reference import reconcile
from .simulation import run_account
from .attribution import style_attribution


def freeze(input_path, output, request, *, industry=None, benchmark=None):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    require(not (root/'REMEDIATION_PROTOCOL.json').exists(),'already frozen; resume run')
    payload=load_json(input_path);snapshot=snapshot_from_payload(payload['snapshot'])
    require(snapshot.payload['exposure']=='research_exposed','research-exposed history only')
    require(request['research_start'] in snapshot.sessions and request['research_end'] in snapshot.sessions,
            'research window outside calendar')
    require(request['research_start']<request['research_end'],'empty research window')
    if 'pool_top_n' in request:
        require(type(request['pool_top_n']) is int and request['pool_top_n']>0,'invalid declared pool size')
    if request.get('full_attribution'):
        require(industry is not None and benchmark is not None,'full attribution requires industry and benchmark sources')
        require(request.get('neutralization_objective')=='diagnostic_style_residual',
                'full attribution requires an explicit diagnostic residual objective')
    fields=snapshot.payload['fields'];f=lambda n:{'field':n}
    rank=lambda x:node('rank',x)
    base=factor_spec(rank(node('return',f('close'),window=20)),fields,'frozen_return20')
    single=factor_spec(rank(node('return',f('close'),window=15)),fields,'one_window_added_feature')
    candidates=[factor_spec(rank(node('return',f('close'),window=5)),fields,'price_persistence5'),
                factor_spec(rank(node('div',f('volume'),node('mean',f('volume'),window=5))),fields,'volume_surprise5')]
    frozen_at=datetime.now().astimezone().isoformat()
    data_policy=None
    if request.get('style_data_policy') is not None:
        require(request['style_data_policy']=='latest_available_snapshot','unsupported style data policy')
        require(industry is not None,'latest style policy requires industry source')
        from .vintage_policy import freeze_latest_style_policy
        paths=request.get('industry_candidates') or [industry]
        require(str(Path(industry).resolve()) in {str(Path(p).resolve()) for p in paths},
                'caller industry must belong to registered versions')
        data_policy=freeze_latest_style_policy(input_path,paths,frozen_at)
        industry=data_policy['industry_selected']['path']
    inputs={'market':{'path':str(Path(input_path).resolve()),'sha256':file_hash(input_path)}}
    for name,path in [('industry',industry),('benchmark',benchmark)]:
        if path:inputs[name]={'path':str(Path(path).resolve()),'sha256':file_hash(path)}
    scope={'schema_version':'fuxi.remediation_study.1','frozen_at':frozen_at,
           'inputs':inputs,'snapshot_hash':snapshot.hash,'code_hashes':code_manifest(),
           'request':request,'seed':1708,'baseline':base,'single_feature':single,'selection_candidates':candidates,
           'scope':'24-cohort or explicit caller cohort; research-exposed diagnostic; no automatic promotion',
           'style_objective':'raw/explained/residual signal attribution on shared support; no portfolio neutralization',
           'style_vintage':'current retrieval effective-date proxy; original publication/availability/revision unproved',
           'budgets':{'single_feature_edges':1,'selection_candidates':2,'selection_methods':3,'stress_scenarios':2},
           'benchmark_kind':'price_index; dividends/taxes/currency and scope difference are not style alpha',
           'formal_qualification_allowed':False,'live_execution_allowed':False}
    if data_policy is not None:scope['style_data_policy']=data_policy
    scope['protocol_hash']=digest(scope);write_new(root/'REMEDIATION_PROTOCOL.json',scope)
    for name,sha in scope['code_hashes'].items():
        raw=(Path(__file__).parent/name).read_bytes()
        import hashlib
        require(hashlib.sha256(raw).hexdigest()==sha,'code changed while freezing')
        write_new(root/'code'/name,raw)
    return scope


def proxy_styles(payload, source, *, data_policy=None):
    """Never invent historical ingestion/publication timestamps from in_date."""
    from .history import iso, jq_code
    if data_policy is not None:
        from .vintage_policy import verify_latest_style_policy
        verify_latest_style_policy(data_policy,payload,source)
    snapshot=snapshot_from_payload(payload['snapshot']);members={}
    for query in source['queries']:
        for row in query['rows']:
            require(row['ts_code']==query['code'],'industry query returned another security')
            if row.get('in_date') and row.get('l1_code'):
                members.setdefault(jq_code(row['ts_code']),{})[digest(row)]=row
    days=snapshot.sessions[snapshot.start:];raw={};conflict=missing=0
    for day in days:
        raw[day]={}
        for code in snapshot.universe[day]:
            rows=[r for r in members.get(code,{}).values() if iso(r['in_date'])<=day and
                  (not iso(r.get('out_date')) or day<iso(r['out_date']))]
            choices={r['l1_code'] for r in rows}
            raw[day][code]=next(iter(choices)) if len(choices)==1 else None
            conflict+=len(choices)>1;missing+=not choices
    categories=sorted({v for cross in raw.values() for v in cross.values() if v is not None})
    require(categories,'no effective-date industry support; current map forbidden')
    momentum=evaluate_ast(node('return',{'field':'close'},window=20),snapshot)
    volatility=evaluate_ast(node('div',node('std',{'field':'close'},window=20),
                           node('mean',{'field':'close'},window=20)),snapshot)
    cap={};missing_cap=0
    for day in days:
        cap[day]={};i=snapshot.sessions.index(day)
        for code in snapshot.universe[day]:
            close=snapshot.value(i,code,'close',snapshot.clocks[day])
            shares=snapshot.value(i,code,'free_shares',snapshot.clocks[day]) if 'free_shares' in snapshot.payload['fields'] else None
            v=log(close*shares) if close is not None and shares is not None and close>0 and shares>0 else None
            cap[day][code]=v;missing_cap+=v is None
    panels={'log_free_float_market_cap':cap,'momentum20':momentum,'price_cv20':volatility}
    for group in categories[1:]:
        panels['industry:'+group]={d:{s:None if v is None else float(v==group) for s,v in cross.items()} for d,cross in raw.items()}
    # A one-category surface still needs a support marker for missing industries.
    if len(categories)==1:panels['industry_known']={d:{s:None if v is None else 1. for s,v in x.items()} for d,x in raw.items()}
    provenance={'status':'effective_date_current_vendor_research_proxy','source_hash':digest(source),
                   'captured_at':source.get('captured_at'),'industry_categories':categories,
                   'conflicting_industry_cells':conflict,'missing_industry_cells':missing,'missing_cap_cells':missing_cap,
                   'cap_semantics':'log(raw close * free_shares); not float_shares or circ_mv',
                   'PIT_industry_certified':False,'PIT_market_cap_certified':False,
                   'historical_ingested_at_invented':False,'investment_neutralization_certified':False}
    if data_policy is not None:
        provenance.update(status='latest_available_snapshot_retrospective_research',
                          data_policy=data_policy,older_vintage_recovery_is_diagnostic_prerequisite=False)
    return panels,provenance


def ledger_attribution(account, market):
    """Cash/share journal bridge, retaining actions, trade effects and unknown cells."""
    bars={(b['session'],b['instrument']):b for b in market['bars']}
    rows=[];orders=account['orders'];daily=account['daily']
    for previous,current in zip(daily,daily[1:]):
        day=current['session'];nav0=previous['nav'];nav1=current['nav']
        carried={s:previous['holdings'].get(s,0)+previous.get('pending_shares',{}).get(s,0)
                 for s in set(previous['holdings'])|set(previous.get('pending_shares',{}))}
        p0={s:bars.get((previous['session'],s),{}).get('close') for s in carried}
        p1={s:bars.get((day,s),{}).get('close') for s in carried}
        known=nav0 is not None and nav1 is not None and all(p0[s] is not None and p1[s] is not None for s in carried)
        fees=sum((Decimal(o.get('fee','0')) for o in orders if o['session']==day),Decimal(0))
        price_pnl=sum((Decimal(str(p1[s]))-Decimal(str(p0[s])))*q for s,q in carried.items()) if known else None
        trade=slip=Decimal(0);trade_known=True
        for order in (o for o in orders if o['session']==day and o['filled']):
            bar=bars[(day,order['instrument'])];close=bar.get('close');opening=bar.get('open')
            if close is None or opening is None:trade_known=False;continue
            quantity=Decimal(order['filled'])*(1 if order['side']=='buy' else -1)
            trade+=quantity*(Decimal(str(close))-Decimal(str(opening)))
            slip+=quantity*(Decimal(order['price'])-Decimal(str(opening)))
        delta=Decimal(nav1)-Decimal(nav0) if known else None
        residual=delta-price_pnl-trade+slip+fees if known and trade_known else None
        unfilled={};unfilled_notional=Decimal(0);unfilled_marks_known=True
        for order in (o for o in orders if o['session']==day):
            quantity=order['requested']-order['filled']
            if quantity<=0:continue
            code=order['instrument'];unfilled[code]=unfilled.get(code,0)+quantity
            opening=bars.get((day,code),{}).get('open')
            if opening is None:unfilled_marks_known=False
            else:unfilled_notional+=quantity*Decimal(str(opening))
        # The residual includes action rights/new share marks and valuation conventions.
        # It is deliberately not relabelled as unexplained trading alpha.
        rows.append({'session':day,'nav_delta':str(delta) if delta is not None else None,
                     'carried_price_pnl':str(price_pnl) if price_pnl is not None else None,
                     'trade_open_to_close_pnl':str(trade) if trade_known else None,
                     'slippage_and_tick_cost':str(slip) if trade_known else None,'explicit_fees':str(fees),
                     'action_and_valuation_residual':str(residual) if residual is not None else None,
                     'return_denominator':nav0,'unfilled_share_quantities_by_instrument':unfilled,
                     'unfilled_notional_at_observed_open':str(unfilled_notional) if unfilled_marks_known else None,
                     'arithmetic_bridge_error':float(delta-price_pnl-trade+slip+fees-residual) if residual is not None else None})
    require(rows,'zero account attribution comparisons')
    return {'daily':rows,'n_positions':len(rows),'identified_positions':sum(r['nav_delta'] is not None for r in rows),
            'unfilled_orders_by_reason':dict(Counter(o['reason'] for o in orders if o.get('reason'))),
            'counterfactual_unfilled_PnL_identified':False,'PIT_style_return_split_identified':False,
            'currency':'CNY','unfilled_notional_is_opportunity_PnL':False,
            'initial_session_NAV':daily[0]['nav'],'initial_cash':market['execution']['initial_cash'],
            'first_session_change_reported_separately':True,'causal_claim_allowed':False}


def broad_comparison(candidate, pool, source):
    rows={r['trade_date'][:4]+'-'+r['trade_date'][4:6]+'-'+r['trade_date'][6:]:r for r in source['rows']}
    output=[]
    for i in range(1,len(candidate['daily'])):
        a0,a1=candidate['daily'][i-1:i+1];b0,b1=pool['daily'][i-1:i+1];day=a1['session'];previous=a0['session']
        require(day==b1['session'] and previous==b0['session'],'broad/account window mismatch')
        q0=rows.get(previous,{}).get('close');q1=rows.get(day,{}).get('close')
        available=all(v is not None for v in (a0['nav'],a1['nav'],b0['nav'],b1['nav'],q0,q1))
        if available:
            ra=float(Decimal(a1['nav'])/Decimal(a0['nav'])-1);rb=float(Decimal(b1['nav'])/Decimal(b0['nav'])-1)
            ri=float(q1)/float(q0)-1
            output.append({'session':day,'candidate_net':ra,'pool_net':rb,'broad_price_index':ri,
                           'pool_scope_net_difference':rb-ri,'within_pool_selection_execution_difference':ra-rb,
                           'bridge_residual':ra-ri-((rb-ri)+(ra-rb))})
        else:output.append({'session':day,'status':'data_insufficient'})
    require(output,'zero broad comparisons')
    return {'rows':output,'n_pairs':sum('candidate_net' in r for r in output),
            'benchmark':'000985.CSI price index; current vintage, not total return',
            'causal_style_alpha_identified':False,'formal_evidence':False}


def run(output):
    root=Path(output);scope=load_json(root/'REMEDIATION_PROTOCOL.json')
    require(scope['protocol_hash']==digest({k:v for k,v in scope.items() if k!='protocol_hash'}),'protocol changed')
    require(scope['code_hashes']==code_manifest(),'code changed; freeze a new study')
    for info in scope['inputs'].values():require(file_hash(info['path'])==info['sha256'],'input changed')
    payload=load_json(scope['inputs']['market']['path']);snapshot=snapshot_from_payload(payload['snapshot'])
    labels=Labels(payload['labels'],snapshot);request=scope['request'];start,end=request['research_start'],request['research_end']
    require(snapshot.hash==scope['snapshot_hash'],'snapshot changed')
    def stage(name,function):
        target=root/(name+'.json')
        if target.exists():
            saved=load_json(target);require(saved['protocol_hash']==scope['protocol_hash'] and
                saved['result_hash']==digest(saved['result']),'cached result changed');return saved['result']
        print('REMEDIATION_STAGE',name,flush=True);result=function()
        write_new(target,{'protocol_hash':scope['protocol_hash'],'result_hash':digest(result),'result':result});return result
    base=evaluate_ast(scope['baseline']['canonical_ast'],snapshot)
    parameter=evaluate_ast(scope['single_feature']['canonical_ast'],snapshot)
    candidates={c['formula_hash']:evaluate_ast(c['canonical_ast'],snapshot) for c in scope['selection_candidates']}
    def accounts():
        ir,account=account_for(base,snapshot,payload['market'],request,identity='source_bound_rank20')
        equal={d:{s:1. if value is not None else None for s,value in x.items()} for d,x in base.items()}
        pi,pool=account_for(equal,snapshot,payload['market'],request,identity='source_bound_pool',
                            top_n=request.get('pool_top_n',24))
        return {'strategy_ir':ir,'account':account,'pool_ir':pi,'pool_account':pool,
                'reference':reconcile(account,payload['market']),'pool_reference':reconcile(pool,payload['market']),
                'ledger_attribution':ledger_attribution(account,payload['market']),
                'pool_relative':paired_accounts(account,pool,request)}
    account_result=stage('ACCOUNT_REPAIR',accounts)
    single=stage('FEATURE_INCREMENT',lambda:compare_methods(payload,base,{scope['single_feature']['formula_hash']:parameter},request))
    selection=stage('SELECTION_AB',lambda:compare_methods(payload,base,candidates,request))
    def stress():
        rows=[]
        for kind,result in [('feature',single),('selection',selection)]:
            for name,cost,capacity in [('double_cost',request['cost_stress_multiplier'],1.),('half_capacity',1.,request['capacity_stress_multiplier'])]:
                accounts={method:run_account(result['outputs'][method]['strategy_ir'],payload['market'],start=start,end=end,
                          cost_multiplier=cost,participation_multiplier=capacity) for method in ('base','ordinary','screened')}
                comparisons={method:paired_accounts(accounts[method],accounts['base'],request) for method in ('ordinary','screened')}
                comparisons['selection_increment']=paired_accounts(accounts['screened'],accounts['ordinary'],request)
                rows.append({'study':kind,'scenario':name,'comparisons':comparisons,
                             'accounts':{m:{'eval_status':a['eval_status'],'formal_evidence':a['formal_evidence'],'gaps':a['gaps'],
                                            'reference':reconcile(a,payload['market'])} for m,a in accounts.items()}})
        return {'scenarios':rows,'same_frozen_models_reused':True,'no_fitting_using_stress_outcomes':True}
    stresses=stage('ACCOUNT_STRESS',stress)
    def styles():
        if 'industry' not in scope['inputs']:return {'eval_status':'data_insufficient','reason':'industry source missing','formal_PIT_certified':False}
        panels,provenance=proxy_styles(payload,load_json(scope['inputs']['industry']['path']),
                                       data_policy=scope.get('style_data_policy'))
        result=style_attribution({d:r for d,r in base.items() if start<=d<=end},panels,labels,snapshot,payload['research_clock'],provenance=provenance)
        return {'eval_status':'valid' if any(r['status']=='valid' for r in result['sections']) else 'data_insufficient',
                'attribution':result,'formal_PIT_certified':False,'production_neutralization_applied':False}
    attribution=stage('STYLE_ATTRIBUTION',styles)
    broad=stage('BROAD_BENCHMARK',lambda:broad_comparison(account_result['account'],account_result['pool_account'],
                load_json(scope['inputs']['benchmark']['path'])) if 'benchmark' in scope['inputs'] else {'n_pairs':0,'reason':'benchmark missing'})
    complete_attribution=None
    if request.get('full_attribution'):
        def complete_styles():
            from .account_attribution import full_account_attribution,residual_account_experiment
            panels,provenance=proxy_styles(payload,load_json(scope['inputs']['industry']['path']),
                                           data_policy=scope.get('style_data_policy'))
            return {'account_attribution':full_account_attribution(account_result['account'],account_result['pool_account'],
                     payload['market'],panels,load_json(scope['inputs']['benchmark']['path']),style_provenance=provenance),
                    'residual_experiment':residual_account_experiment(base,panels,snapshot,payload['market'],request,
                     objective=request['neutralization_objective']),
                    'formal_eligibility':False,'historical_PIT_certified':False}
        complete_attribution=stage('FULL_ATTRIBUTION',complete_styles)
    def selection_diagnostics():
        from .research_screening import training_reading,orient
        rows=[]
        for fold in selection['outputs']['screened']['folds']:
            if 'prediction_end' not in fold:continue
            days=[d for d in snapshot.sessions if fold['prediction_start']<=d<=fold['prediction_end']]
            for name,panel in candidates.items():
                sign=fold['readings'][name]['direction']
                reading=training_reading(orient(panel,sign),snapshot,labels,payload['research_clock'],days) if sign is not None else {'n_pairs':0,'mean_rank_ic':None}
                rows.append({'fold_start':fold['prediction_start'],'candidate':name,'selected':name==fold['selected_feature'],
                             'training_direction_frozen':sign,'following_fold_mean_rank_ic':reading['mean_rank_ic'],
                             'n_pairs':reading['n_pairs'],'used_to_choose_this_fold':False})
        return {'pairwise':paired_scores(*list(candidates.values()),start,end),
                'folds':selection['outputs']['screened']['folds'],
                'ordinary_folds':selection['outputs']['ordinary']['folds'],'missed_candidate_readings':rows,
                'missed_value_scope':'frozen following-fold IC is descriptive, not a hindsight optimum; all candidate trials retained',
                'formal_selection_enabled':False}
    redundancy=stage('SELECTION_DIAGNOSTICS',selection_diagnostics)
    summary={'schema_version':'fuxi.remediation_result.1','protocol_hash':scope['protocol_hash'],
             'account_reference':account_result['reference'],'pool_reference':account_result['pool_reference'],
             'feature_comparisons':single['comparisons'],'selection_comparisons':selection['comparisons'],
             'stress':stresses,'style_status':attribution['eval_status'],'broad_n_pairs':broad['n_pairs'],
             'measured_account_increment':all(c['n_pairs']>0 for c in [single['comparisons']['screened'],selection['comparisons']['selection_increment']]),
             'formal_qualification_count':0,'real_platform_pairs':0,'future_observations':0,
             'required_external_evidence':[
                 'independent acceptance of declared retrospective research scope and latest-vintage limitations'
                 if scope.get('style_data_policy') is not None else 'independent data review and historical PIT proof',
                 'real JQ exports after formal research qualification','unseen future maturity'],
             'autonomous_writes_enabled':False,'evidence_scope':'research_exposed current-vintage diagnostics'}
    if complete_attribution is not None:
        summary['full_attribution_n_pairs']=complete_attribution['account_attribution']['n_pairs']
        summary['residual_experiment_n_pairs']=[r['n_pairs'] for r in complete_attribution['residual_experiment']['scenarios']]
        summary['investment_neutralization_certified']=False
    if not (root/'RESULT.json').exists():write_new(root/'RESULT.json',summary)
    else:require(load_json(root/'RESULT.json')==summary,'resume summary changed')
    return summary
