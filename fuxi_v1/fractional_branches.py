"""Pathwise bounded research accounts; never observed investor allocations."""
from decimal import ROUND_FLOOR,ROUND_CEILING
from .contracts import require,digest
from .simulation import run_account
from .statistics import performance


class ChoiceRequired(Exception):
    def __init__(self,term,held,exact):
        self.event={'evidence_id':term['evidence_id'],'instrument':term['instrument'],'ex_date':term['ex_date'],
                    'record_holdings':held,'exact_extra':str(exact),'floor':int(exact.to_integral_value(rounding=ROUND_FLOOR)),
                    'ceil':int(exact.to_integral_value(rounding=ROUND_CEILING))}
        super().__init__('source_supported_fractional_choice_required')


def enumerate_accounts(ir,market,rules,*,max_paths=128,research_context=None,**kwargs):
    require(type(max_paths) is int and max_paths>=1,'invalid branch budget')
    queue=[{}];leaves=[];splits=[];attempts=0;overflow=False
    while queue:
        choices=queue.pop();observed=[]
        def resolve(term,held,exact):
            rule=rules.get(term['evidence_id'])
            if rule is None:return None
            require(rule['term_hash']==digest(term) and rule['outcomes']=='floor_or_ceil'
                    and rule['source_scope']=='archived_primary_clause','fractional rule bound to other terms')
            key=term['evidence_id']
            if key not in choices:raise ChoiceRequired(term,held,exact)
            require(choices[key] in ('floor','ceil'),'invalid branch choice')
            quantity=int(exact.to_integral_value(rounding=ROUND_FLOOR if choices[key]=='floor' else ROUND_CEILING))
            observed.append({'event':key,'held':held,'exact':str(exact),'choice':choices[key],'quantity':quantity,'rule_hash':digest(rule)})
            return quantity
        attempts+=1
        try:
            account=run_account(ir,market,research_context=research_context,fractional_resolver=resolve,**kwargs)
        except ChoiceRequired as exc:
            splits.append(dict(exc.event,prefix=dict(choices)))
            # Frontier is the minimum number of eventual leaves, not just completed paths.
            if len(leaves)+len(queue)+2>max_paths:
                overflow=True;queue.append(choices);break
            for side in ('ceil','floor'):queue.append(dict(choices,**{exc.event['evidence_id']:side}))
            continue
        leaves.append({'path_id':digest(observed),'choices':choices,'observed_choices':observed,'account':account})
    complete=not overflow and not queue
    identifiable=complete and bool(leaves) and all(p['account']['eval_status']=='valid' for p in leaves)
    return {'schema_version':'fuxi.fractional_branches.1','enumeration_complete':complete,
            'all_paths_identifiable':identifiable,'leaf_count':len(leaves),'attempt_count':attempts,
            'max_paths':max_paths,'splits':splits,'unexplored_prefixes':queue,'paths':leaves,
            'actual_allocation_observed':False,'formal_assurance':False,
            'scope':'source-supported per-event hypothetical choices; no probabilities or actual settlement receipt'}


def compare_branches(left,right,request):
    if not left['all_paths_identifiable'] or not right['all_paths_identifiable']:
        return {'eval_status':'data_insufficient','decision':'not_applicable','lower':None,
                'reason':'branch_budget_or_held_path_evidence_incomplete','pairs':0,'formal_assurance':False}
    import numpy as np
    from .research_protocol import inference
    paths_a,paths_b=left['paths'],right['paths'];readings=[]
    for a in paths_a:
        for b in paths_b:
            aa,bb=a['account'],b['account']
            require(aa['market_hash']==bb['market_hash'] and aa['assumptions']==bb['assumptions'],'branch market/cost mismatch')
            require([d['session'] for d in aa['daily']]==[d['session'] for d in bb['daily']],'branch windows mismatch')
            ar,br=performance(aa['daily'])['returns'],performance(bb['daily'])['returns']
            require(len(ar)==len(br)>0,'zero/mismatched branch comparisons')
            delta=[x-y for x,y in zip(ar,br)]
            ci=inference(delta,request,731)
            readings.append({'candidate_path':a['path_id'],'baseline_path':b['path_id'],'mean':float(np.mean(delta)),
                             'lower':ci['lower'],'upper':ci['upper'],'n_pairs':len(delta)})
    require(readings,'zero branch path pairs')
    lower=min(r['lower'] for r in readings);upper=max(r['upper'] for r in readings)
    return {'eval_status':'valid','decision':'robust_positive_research' if lower>request['minimum_net_increment_daily'] else 'not_robust_positive',
            'lower':lower,'upper':upper,'pairs':len(readings),'path_pairs':readings,
            'mean_min':min(r['mean'] for r in readings),'mean_max':max(r['mean'] for r in readings),
            'multiple_testing_corrected':False,'formal_assurance':False,
            'scope':'Cartesian outer bound of complete conditional paths; possible global-allocation constraints may narrow it'}
