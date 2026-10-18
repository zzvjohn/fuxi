"""Matched, additive account attribution for explicitly retrospective research.

Statistical style fits and unfilled-order marks are diagnostic projections, not
causal alpha or executable full-fill counterfactuals. Strict journals remain intact.
"""
from collections import Counter
from decimal import Decimal
from statistics import mean
from .contracts import require, digest, finite
from .account import money
from .account_reference import reconcile


def _nav(row, conditional):
    value=(row.get('restricted_research',{}).get('conditional_total_nav')
           if conditional and 'restricted_research' in row else row.get('nav'))
    return money(value) if value is not None else None


def _price(row, code, bars, conditional):
    restricted=row.get('restricted_research',{}).get('restricted_assets',{}) if conditional else {}
    if code in restricted:
        value=restricted[code]['unit_scenario_value']
        return money(value) if value is not None else None
    value=bars.get((row['session'],code),{}).get('close')
    return money(str(value)) if value is not None and finite(value) and value>0 else None


def _bridge(previous, current, orders, bars, conditional):
    """Identity of realized NAV changes, with separately labelled unfilled proxy."""
    n0,n1=_nav(previous,conditional),_nav(current,conditional)
    held={s:previous['holdings'].get(s,0)+previous.get('pending_shares',{}).get(s,0)
          for s in set(previous['holdings'])|set(previous.get('pending_shares',{}))}
    missing=[];carry={};restricted_carry=Decimal(0)
    restricted=set(previous.get('restricted_research',{}).get('restricted_assets',{}))|set(
                   current.get('restricted_research',{}).get('restricted_assets',{})) if conditional else set()
    for code,quantity in held.items():
        p0,p1=_price(previous,code,bars,conditional),_price(current,code,bars,conditional)
        if p0 is None or p1 is None:missing.append({'instrument':code,'reason':'carried_quote_unknown'});continue
        carry[code]=(p1-p0)*quantity
        if code in restricted:restricted_carry+=carry[code]
    trade=slip=fee=opportunity=Decimal(0);opportunity_missing=[];unfilled={}
    for order in orders:
        code=order['instrument'];bar=bars.get((current['session'],code),{})
        opening,closing=bar.get('open'),bar.get('close')
        observed=(opening is not None and closing is not None and finite(opening) and finite(closing)
                  and opening>0 and closing>0)
        sign=1 if order['side']=='buy' else -1
        fee+=money(order.get('fee','0'))
        if order['filled']:
            if not observed:missing.append({'instrument':code,'reason':'filled_open_close_quote_unknown'})
            else:
                quantity=Decimal(order['filled']*sign)
                trade+=quantity*(money(str(closing))-money(str(opening)))
                slip+=quantity*(money(order['price'])-money(str(opening)))
        remaining=order['requested']-order['filled']
        if remaining>0:
            unfilled[code]=unfilled.get(code,0)+remaining
            if observed:opportunity+=Decimal(remaining*sign)*(money(str(closing))-money(str(opening)))
            else:opportunity_missing.append({'instrument':code,'reason':order.get('reason'),'quantity':remaining})
    known=n0 is not None and n1 is not None and n0>0 and not missing
    delta=n1-n0 if known else None;price=sum(carry.values(),Decimal(0)) if not missing else None
    residual=delta-price-trade+slip+fee if known else None
    return {'nav0':n0,'nav1':n1,'known':known,'delta':delta,'carry':carry,'carry_total':price,
            'restricted_carry':restricted_carry,'trade':trade,'slip':slip,'fee':fee,
            'action_valuation_residual':residual,'opportunity':opportunity if not opportunity_missing else None,
            'unfilled_quantities':unfilled,'unfilled_proxy_missing':opportunity_missing,'missing':missing}


def _daily_style_fit(previous_day, day, panels, bars, exclude):
    import numpy as np
    names=sorted(panels);require(names,'explicit style columns required')
    codes=sorted(set.intersection(*(set(panels[n].get(previous_day,{})) for n in names)))
    sample=[]
    for code in codes:
        features=[panels[n][previous_day][code] for n in names]
        p0,p1=bars.get((previous_day,code),{}).get('close'),bars.get((day,code),{}).get('close')
        require(all(v is None or finite(v) for v in features),'nonfinite account style exposure')
        if code not in exclude and all(v is not None for v in features) and all(
                v is not None and finite(v) and v>0 for v in (p0,p1)):
            sample.append((code,float(p1)/float(p0)-1,features))
    row={'lagged_exposure_session':previous_day,'session':day,'n_pairs':len(sample),
         'support_instruments':[r[0] for r in sample],'support_hash':digest([r[0] for r in sample]),
         'excluded_restricted_instruments':sorted(exclude),'causal_claim_allowed':False}
    if len(sample)<=len(names)+2:
        row.update(status='data_insufficient',reason='insufficient_same_quote_style_support');return {},row
    y=np.asarray([r[1] for r in sample]);x=np.asarray([r[2] for r in sample])
    x=x-x.mean(axis=0);scale=x.std(axis=0)
    design=np.column_stack([np.ones(len(x)),x/np.where(scale>0,scale,1.)])
    beta,_,rank,_=np.linalg.lstsq(design,y,rcond=None);explained=design@beta
    values={r[0]:float(v) for r,v in zip(sample,explained)}
    row.update(status='valid',design_rank=int(rank),design_columns=int(design.shape[1]),
               rank_deficient=int(rank)<design.shape[1],
               coefficient_identification='minimum_norm_nonunique' if int(rank)<design.shape[1] else 'full_column_rank',
               statistical_explained_returns=values,
               coefficients={n:float(v) for n,v in zip(['intercept']+names,beta)})
    return values,row


def full_account_attribution(candidate, pool, market, style_panels, benchmark, *,
                             benchmark_id='000985.CSI', style_provenance=None):
    """Bridge all days; unknown components remain unknown and are counted explicitly."""
    require([r['session'] for r in candidate['daily']]==[r['session'] for r in pool['daily']],
            'full attribution account windows differ')
    require(candidate['market_hash']==pool['market_hash']==market['market_hash'] and
            candidate['assumptions']==pool['assumptions']==market['execution'],'full attribution market/costs differ')
    require(candidate['daily'][0]['nav']==pool['daily'][0]['nav']==market['execution']['initial_cash'],
            'full attribution original capital differs')
    conditional='restricted_policy_hash' in candidate or 'restricted_policy_hash' in pool
    if conditional:
        require(candidate.get('restricted_policy_hash')==pool.get('restricted_policy_hash') and
                candidate.get('restricted_scenario')==pool.get('restricted_scenario'),
                'full attribution requires identical restricted scenario')
    refs={n:reconcile(a,market) for n,a in [('candidate',candidate),('pool',pool)]}
    require(all(r['n_pairs']>0 and not r['mismatches'] for r in refs.values()),'account reference mismatch')
    bars={(r['session'],r['instrument']):r for r in market['bars']}
    require(len(bars)==len(market['bars']),'duplicate attribution market bar')
    index={}
    for row in benchmark['rows']:
        require(row.get('ts_code',benchmark_id)==benchmark_id,'benchmark identity mismatch')
        day=row['trade_date'];day=day if '-' in day else day[:4]+'-'+day[4:6]+'-'+day[6:]
        require(day not in index,'duplicate benchmark session');index[day]=row.get('close')
    order_maps={n:{} for n in ('candidate','pool')}
    for name,account in [('candidate',candidate),('pool',pool)]:
        for order in account['orders']:order_maps[name].setdefault(order['session'],[]).append(order)
    rows=[];style_rows=[];unknown=Counter();defined=Counter()
    for position in range(1,len(candidate['daily'])):
        previous,current=candidate['daily'][position-1:position+1]
        pp,pc=pool['daily'][position-1:position+1];day=current['session'];prior=previous['session']
        if prior.endswith(':initial'):
            available=[d for d in market['sessions'] if d<day];prior=available[-1] if available else None
        a=_bridge(previous,current,order_maps['candidate'].get(day,[]),bars,conditional)
        b=_bridge(pp,pc,order_maps['pool'].get(day,[]),bars,conditional)
        q0,q1=index.get(prior),index.get(day)
        benchmark_known=all(v is not None and finite(v) and v>0 for v in (q0,q1))
        restricted=set(previous.get('restricted_research',{}).get('restricted_assets',{}))|set(
                     current.get('restricted_research',{}).get('restricted_assets',{}))|set(
                     pp.get('restricted_research',{}).get('restricted_assets',{}))|set(
                     pc.get('restricted_research',{}).get('restricted_assets',{}))
        fit,style=_daily_style_fit(prior,day,style_panels,bars,restricted) if prior else ({},
             {'session':day,'status':'data_insufficient','reason':'no_prior_market_session','n_pairs':0})
        style_rows.append(style)
        row={'session':day,'lagged_exposure_session':prior,'conditional_restricted_scenario':candidate.get('restricted_scenario'),
             'candidate_strict_nav_known':current['nav'] is not None,'pool_strict_nav_known':pc['nav'] is not None,
             'unfilled_proxy_missing':{'candidate':a['unfilled_proxy_missing'],'pool':b['unfilled_proxy_missing']},
             'unknown_quotes':{'candidate':a['missing'],'pool':b['missing']},'style_fit_status':style['status'],
             'restricted_holding_count':len(restricted),
             'strict_ownership_unknown_count':len(set(current.get('unresolved_events',[]))|set(pc.get('unresolved_events',[])))}
        for name,bridge in [('candidate',a),('pool',b)]:
            row[name+'_cash_share_bridge']={k:str(bridge[k]) if bridge[k] is not None else None for k in
                ('nav0','nav1','delta','carry_total','restricted_carry','trade','slip','fee','action_valuation_residual','opportunity')}
            row[name+'_cash_share_bridge']['unfilled_share_quantities']=bridge['unfilled_quantities']
        if not (a['known'] and b['known'] and benchmark_known):
            row.update(status='data_insufficient',reason='account_or_benchmark_bridge_unknown')
            unknown[row['reason']]+=1;rows.append(row);continue
        ra,rb=float(a['delta']/a['nav0']),float(b['delta']/b['nav0']);ri=float(q1)/float(q0)-1
        def active(key):return float(a[key]/a['nav0']-b[key]/b['nav0'])
        def style_pnl(prev,bridge):
            result=Decimal(0);marked=[];unmatched=[]
            quantities={s:prev['holdings'].get(s,0)+prev.get('pending_shares',{}).get(s,0)
                        for s in set(prev['holdings'])|set(prev.get('pending_shares',{}))}
            for code,quantity in quantities.items():
                if code in restricted:continue
                if code not in fit:unmatched.append(code);continue
                price=_price(prev,code,bars,conditional)
                if price is None:unmatched.append(code);continue
                result+=price*quantity*money(str(fit[code]));marked.append(code)
            return result,marked,unmatched
        sa,matched_a,unmatched_a=style_pnl(previous,a);sb,matched_b,unmatched_b=style_pnl(pp,b)
        style_active=float(sa/a['nav0']-sb/b['nav0']) if style['status']=='valid' else None
        carry_active=active('carry_total');restricted_active=active('restricted_carry');filled_active=active('trade')
        proxy_known=a['opportunity'] is not None and b['opportunity'] is not None
        opportunity=active('opportunity') if proxy_known else None
        components={'pool_scope_net_difference':rb-ri,
          'statistical_style_carried_contribution':style_active,
          'within_pool_security_and_unmatched_style_residual':carry_active-restricted_active-(style_active or 0.),
          'restricted_carried_contribution':restricted_active,
          'filled_open_to_close_contribution':filled_active,
          'intended_open_to_close_mark_proxy':filled_active+opportunity if proxy_known else None,
          'unfilled_open_to_close_mark_proxy':-opportunity if proxy_known else None,
          'explicit_fee_and_slippage_contribution':-active('fee')-active('slip'),
          'action_valuation_and_denominator_interaction_residual':active('action_valuation_residual')}
        # Use the observed-filled contribution when the optional opportunity mark is unknown.
        executable_sum=sum(components[k] for k in ('pool_scope_net_difference','within_pool_security_and_unmatched_style_residual',
             'restricted_carried_contribution','filled_open_to_close_contribution','explicit_fee_and_slippage_contribution',
             'action_valuation_and_denominator_interaction_residual'))+(style_active or 0.)
        error=ra-ri-executable_sum
        require(abs(error)<1e-10,'full attribution arithmetic bridge error')
        row.update(status='valid',candidate_net_return=ra,pool_net_return=rb,broad_price_index_return=ri,
          candidate_minus_benchmark=ra-ri,components=components,arithmetic_bridge_error=error,
          style_supported_held={'candidate':matched_a,'pool':matched_b},
          style_unmatched_held={'candidate':unmatched_a,'pool':unmatched_b},
          candidate_cash=float(money(current['cash'])),pool_cash=float(money(pc['cash'])),
          opportunity_proxy_identified=proxy_known,
          complete_style_split_identified=style['status']=='valid' and not unmatched_a and not unmatched_b)
        for name,value in components.items():defined[name]+=value is not None
        rows.append(row)
    require(rows,'zero full attribution dates')
    valid=[r for r in rows if r['status']=='valid'];require(valid,'zero full attribution identified pairs')
    return {'schema_version':'fuxi.full_account_attribution.1','rows':rows,'n_positions':len(rows),'n_pairs':len(valid),
      'style_sections':style_rows,'style_return_pairs':sum(r['n_pairs'] for r in style_rows),
      'valid_style_sections':sum(r['status']=='valid' for r in style_rows),
      'component_identified_positions':dict(defined),'unknown_dates_by_reason':dict(unknown),
      'max_abs_bridge_error':max(abs(r['arithmetic_bridge_error']) for r in valid),
      'mean_candidate_minus_benchmark':mean(r['candidate_minus_benchmark'] for r in valid),
      'strict_references':refs,'benchmark_identity':benchmark_id,'benchmark_kind':'price_index_not_total_return',
      'initial_capital':market['execution']['initial_cash'],'first_session_benchmark_reference':'prior_calendar_close_when_present',
      'conditional_restricted_scenario':candidate.get('restricted_scenario'),'style_provenance':style_provenance or {},
      'style_method':'same_observed_raw_close_returns_on_previous_session_exposures_statistical_projection',
      'unfilled_proxy_method':'signed_requested_minus_filled_times_observed_same_session_close_minus_open',
      'unfilled_proxy_is_executable_counterfactual':False,'unfilled_proxy_is_realized_account_PnL':False,
      'additive_columns':'pool_scope + style_when_identified + within_pool_residual + restricted + filled + fees_slip + interactions',
      'alternative_proxy_columns':'replace filled by intended_proxy + unfilled_proxy only when both identified',
      'historical_PIT_certified':False,'investment_neutralization_certified':False,
      'formal_eligibility':False,'causal_claim_allowed':False}


def residual_account_experiment(scores, styles, snapshot, market, request, *, objective,
                                restricted_policy=None, research_context=None):
    """Two isolated IRs on identical label-independent decision support."""
    from .attribution import diagnostic_style_residual
    from .portfolio import compile_strategy,strategy_spec
    from .simulation import run_account
    fitted=diagnostic_style_residual(scores,styles,objective=objective)
    require(fitted['valid_sections']>0,'no identified diagnostic residual account sections')
    scenarios=restricted_policy['fractions'] if restricted_policy is not None else [None]
    irs={}
    for name,panel in [('raw',fitted['raw_common_support']),('residual',fitted['residual_common_support'])]:
        spec=strategy_spec('diagnostic_style_'+name,top_n=6,max_weight=1/6,rebalance_sessions=5)
        irs[name]=compile_strategy(spec,panel,snapshot,digest(market['execution']))
    require([(r['signal_session'],r['execution_session']) for r in irs['raw']['decisions']]==
            [(r['signal_session'],r['execution_session']) for r in irs['residual']['decisions']], 'residual IR date support differs')
    results=[]
    for scenario in scenarios:
        kwargs={'research_context':research_context}
        if scenario is not None:kwargs.update(restricted_policy=restricted_policy,restricted_scenario=scenario)
        accounts={n:run_account(ir,market,start=request['research_start'],end=request['research_end'],**kwargs)
                  for n,ir in irs.items()}
        refs={n:reconcile(a,market) for n,a in accounts.items()}
        require(all(r['n_pairs']>0 and not r['mismatches'] for r in refs.values()),'residual experiment reference mismatch')
        increments=[]
        for i in range(1,len(accounts['raw']['daily'])):
            p,q=accounts['raw']['daily'][i-1:i+1];r,s=accounts['residual']['daily'][i-1:i+1]
            require(p['session']==r['session'] and q['session']==s['session'],'residual account date mismatch')
            values=(_nav(p,scenario is not None),_nav(q,scenario is not None),_nav(r,scenario is not None),_nav(s,scenario is not None))
            if all(v is not None for v in values) and values[0]>0 and values[2]>0:
                increments.append(float(values[3]/values[2]-values[1]/values[0]))
        require(increments,'zero residual experiment account pairs')
        results.append({'scenario':scenario,'accounts':accounts,'references':refs,'n_pairs':len(increments),
                        'mean_residual_minus_raw_net_daily':mean(increments),'formal_eligibility':False})
    return {'schema_version':'fuxi.residual_account_ab.1','objective':objective,'fitted':fitted,
            'strategy_irs':irs,'scenarios':results,'same_support_hash':fitted['support_hash'],
            'future_labels_used_to_select_support':False,'production_signal_substitution':False,
            'historical_PIT_certified':False,'investment_neutralization_certified':False,
            'economic_claim_allowed':False,'formal_eligibility':False}
