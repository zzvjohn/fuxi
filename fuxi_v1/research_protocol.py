"""Versioned research screening; power planning and independent confirmation are distinct."""
import math
from statistics import mean, NormalDist
from .contracts import require, finite, digest


def block_inference_fast(values, *, block_length, alpha=.05, replicates=3999, seed=0):
    """Circular moving blocks, NumPy generator; missing dates remain slots, never compressed.

    Bootstrap ratio of valid sums/counts preserves the position of data gaps. CIs
    describe this selected historical research sample, not adaptive-search FWER.
    """
    import numpy as np
    require(values and all(x is None or finite(x) for x in values),'invalid bootstrap observations')
    require(type(block_length) is int and block_length>0 and 0<alpha<1 and replicates>=99,'invalid bootstrap specification')
    n=len(values);ok=np.array([x is not None for x in values]);valid=int(ok.sum())
    if valid<2*block_length:return {'status':'data_insufficient','n_pairs':valid,'mean':None,'lower':None,'upper':None,'p_one_sided':None}
    x=np.array([0. if v is None else v for v in values]);avg=float(x.sum()/valid)
    def sums(a,length):
        if not length:return np.zeros(n)
        extended=np.r_[a,a[:length-1]];cs=np.r_[0.,np.cumsum(extended)]
        return cs[length:]-cs[:-length]
    full,rem=divmod(n,block_length);bs=sums(x,block_length);bc=sums(ok,block_length)
    rs,rc=sums(x,rem),sums(ok,rem);rng=np.random.default_rng(seed);draws=[]
    for lo in range(0,replicates,128):
        size=min(128,replicates-lo);starts=rng.integers(0,n,size=(size,full+bool(rem)))
        numerator=bs[starts[:,:full]].sum(axis=1);denominator=bc[starts[:,:full]].sum(axis=1)
        if rem:numerator+=rs[starts[:,-1]];denominator+=rc[starts[:,-1]]
        require(bool((denominator>0).all()),'all-missing bootstrap draw; data support too sparse')
        draws.extend((numerator/denominator).tolist())
    array=np.array(draws);p=(1+int((array-avg>=avg).sum()))/(replicates+1)
    return {'status':'valid','n_pairs':valid,'calendar_slots':n,'mean':avg,
            'lower':float(np.quantile(array,alpha/2)),'upper':float(np.quantile(array,1-alpha/2)),
            'p_one_sided':p,'p_monte_carlo_se':math.sqrt(p*(1-p)/(replicates+1)),
            'block_length':block_length,'replicates':replicates,'tail_draws_expected':(replicates+1)*alpha/2,
            'uncertainty_scope':'calendar_preserving_stationary_block_research_diagnostic'}


def hac_planning(values, effect=.03, alpha=.05, power=.8, lags=(5,11,23), trials=1):
    """Bartlett HAC sensitivity on baseline calibration only; no hard sample-size gate."""
    import numpy as np
    x=np.array([np.nan if v is None else v for v in values]);n=int(np.isfinite(x).sum())
    require(n>=30 and effect>0 and trials>=1,'calibration support inadequate')
    center=float(np.nanmean(x));v=x-center;gamma0=float(np.nanmean(v*v));result=[]
    z=NormalDist().inv_cdf(1-alpha/trials)+NormalDist().inv_cdf(power)
    for lag in lags:
        lrv=gamma0;pair_counts=[]
        for k in range(1,lag+1):
            pairs=v[k:]*v[:-k];count=int(np.isfinite(pairs).sum());pair_counts.append(count)
            if count:lrv+=2*(1-k/(lag+1))*float(np.nansum(pairs))/len(v)
        lrv=max(0.,lrv);multiplier=lrv/gamma0 if gamma0 else None
        result.append({'lag':lag,'long_run_variance':lrv,'variance':gamma0,'dependence_ratio':multiplier,
                       'effective_n':min(n,n/multiplier) if multiplier and multiplier>0 else None,
                       'required_calendar_sections_approx':max(2,math.ceil(z*z*lrv/effect**2)),
                       'lag_pair_counts':pair_counts})
    return {'calibration_n':n,'minimum_detectable_ic':effect,'alpha':alpha,'planned_trials':trials,'power':power,
            'estimates':result,'is_admission_gate':False,'assumptions':'stationary mean approximation; bandwidth and calibration uncertainty remain'}


def screen(e1,request,yearly,exchanges):
    """Predeclared support, direction, block sensitivity and stability; no power gate."""
    sections=e1['diagnostics']['sections'];rules=request['screen_rules']
    values=[r['rank_ic'] if r['pair_count']>=rules['screen_min_pairs'] else None for r in sections]
    intervals=[block_inference_fast(values,block_length=b,alpha=request['alpha']/request.get('max_trials', 1),
                replicates=request['bootstrap_replicates'],seed=1729) for b in request['block_grid']]
    count=sum(v is not None for v in values)
    coverage=sum(r['pair_count'] for r in sections)/sum(r['universe_count'] for r in sections)
    data_ok=count>=rules['screen_min_sessions'] and coverage>=rules['screen_pair_coverage']
    eligible_years=[x for x in yearly.values() if x['n']>=rules['screen_min_sessions']]
    positive=sum(x['mean_rank_ic'] is not None and x['mean_rank_ic']>0 for x in eligible_years)
    year_fraction=positive/len(eligible_years) if eligible_years else 0
    exchange_positive=sum(x['n']>=rules['screen_min_sessions'] and x['mean_rank_ic'] is not None and x['mean_rank_ic']>0 for x in exchanges.values())
    passed=data_ok and all(x['status']=='valid' and x['lower']>0 for x in intervals)
    passed=passed and year_fraction>=rules['screen_positive_year_fraction'] and exchange_positive>=rules['screen_positive_exchange_count']
    e1['diagnostics'].update(time_uncertainty=intervals[0],block_sensitivity=intervals,
        power_required_sections=None,power_planning_is_gate=False,tail_resolution_adequate=all(x.get('tail_draws_expected',0)>=20 for x in intervals),
        research_pair_coverage=coverage,yearly=yearly,exchanges=exchanges,positive_year_fraction=year_fraction,
        multiple_testing_corrected=True,selection_scope='research_exposed; no confirmatory probability claim')
    if not data_ok:
        e1.update(eval_status='data_insufficient',decision='not_applicable',reject_class=None,metrics=None,reason_code='research_support_inadequate')
    else:
        e1.update(decision='pass' if passed else 'reject',reject_class=None if passed else 'statistical',
                  reason_code='research_v2_direction_and_stability_screen',evidence_scope='research')
    return e1


def inference(values,request,seed):
    if request.get('research_gate_version')=='history_research_v2':
        return block_inference_fast(values,block_length=request['block_length'],alpha=request['alpha'],
                                    replicates=request['bootstrap_replicates'],seed=seed)
    from .statistics import block_inference
    return block_inference(values,block_length=request['block_length'],alpha=request['alpha']/request['max_trials'],
                           replicates=request['bootstrap_replicates'],seed=seed)


def policy_from_plan(plan,snapshot,baseline_e1):
    p=plan['protocol'];values=[r['rank_ic'] for r in baseline_e1['diagnostics']['sections']]
    planning=hac_planning(values,p['minimum_detectable_ic'],p['alpha'],p['power'],trials=plan['reevaluate_formula_count'])
    request={'research_gate_version':p['id'],'research_start':plan['window']['research_start'],
             'research_end':plan['window']['end'],'alpha':p['alpha'],
             'max_trials':plan['reevaluate_formula_count']+plan.get('search',{}).get('max_candidate_requests',60),
             'block_length':12,'block_grid':p['block_grid'],'bootstrap_replicates':p['bootstrap_replicates'],
             'minimum_net_increment_daily':0.,'cost_stress_multiplier':2.,'capacity_stress_multiplier':.5,
             'screen_rules':{k:v for k,v in p.items() if k.startswith('screen_')}}
    result={'schema_version':'fuxi.research_policy.2','plan_hash':digest(plan),'snapshot_hash':snapshot.hash,
            'request':request,'minimum_effective_sections':None,'planning':planning,
            'confirmatory_error_control_validated':False,'calibration_evaluation_hash':digest(baseline_e1)}
    result['policy_hash']=digest(result);return result
