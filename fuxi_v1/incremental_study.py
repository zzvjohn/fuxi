"""Single-feature edges and within-fold selection, using the native trainer/account."""
from .contracts import digest, require
from .data import snapshot_from_payload, Labels
from .models import walk_forward, validate_axis
from .research_screening import training_reading, orient
from .portfolio import compile_strategy, strategy_spec
from .simulation import run_account
from .research_evaluation import paired_accounts, paired_scores


def common_support(base, additions):
    require(additions and all(set(a) == set(base) for a in additions.values()), 'common-support scope invalid')
    for cross in additions.values():
        require(all(set(cross[d]) == set(base[d]) for d in base), 'common-support securities mismatch')
    allowed = {d:{s:base[d][s] is not None and all(a[d][s] is not None for a in additions.values()) for s in base[d]} for d in base}
    def mask(panel):
        return {d:{s:v if allowed[d][s] else None for s,v in cross.items()} for d,cross in panel.items()}
    return mask(base), {name:mask(panel) for name,panel in additions.items()}, digest(allowed)


def fold_predictions(base, additions, snapshot, labels, request, *, method, seed=1708):
    require(method in {'base','ordinary','screened'}, 'selection method invalid')
    common, added, support = common_support(base, additions)
    names = sorted(added)
    days = snapshot.sessions[snapshot.start:]
    output = {d:{s:None for s in snapshot.universe[d]} for d in days}
    fold_rows, models = [], []
    first = days.index(request['research_start'])
    last = days.index(request['research_end'])+1
    require(last>first,'empty fold study window')
    for begin in range(first,last,40):
        day, cutoff = days[begin], snapshot.clocks[days[begin]]
        training = days[max(0,begin-120):max(0,begin-1)]
        readings = {name:training_reading(panel,snapshot,labels,cutoff,training) for name,panel in added.items()}
        eligible = [name for name in names if readings[name]['direction'] is not None]
        selected = None
        if method != 'base' and eligible:
            if method == 'ordinary':
                # Fixed identifier/seed order, no outcome-dependent fallback or extra trials.
                selected = sorted(eligible,key=lambda n:digest([seed,n]))[0]
            else:
                selected = max(eligible,key=lambda n:(abs(readings[n]['mean_rank_ic']),n))
        features = {'baseline_return20':common}
        if selected:
            features['incremental_feature'] = orient(added[selected],readings[selected]['direction'])
            validate_axis({'features':['baseline_return20'],'model':'ridge'},
                          {'features':['baseline_return20','incremental_feature'],'model':'ridge'},'features')
        elif method != 'base':
            fold_rows.append({'prediction_start':day,'eval_status':'data_insufficient','reason':'no_training_identified_candidate'})
            continue
        end = min(begin+40,last)
        fitted = walk_forward(features,snapshot,labels,algorithm='ridge',predict_start=day,predict_end=days[end-1],
                              retrain_every=len(days)+1,min_train_sessions=30,lookback_sessions=120,seed=seed)
        for d in days[begin:end]:
            output[d] = fitted['scores'][d]
        # Only the model actually used in this fold is kept; all later fit speculation is absent.
        models += fitted['models'][:1]
        fold_rows.append({'prediction_start':day,'prediction_end':days[end-1], 'eval_status':fitted['eval_status'],
                          'selected_feature':selected,'selection_cutoff':cutoff.isoformat(), 'readings':readings,
                          'provenance':[r for r in fitted['prediction_provenance'] if day <= r['session'] <= days[end-1]],
                          'changed_design_axis':'one_added_feature' if selected else 'base',
                          'candidate_count':len(additions),'added_feature_count':1 if selected else 0})
    return {'scores':output,'models':models,'folds':fold_rows,'support_hash':support,'selection_method':method,
            'study_axis':'within_fold_selection_method','economic_claim_allowed':False}


def account_for(scores, snapshot, market, request, *, identity, model_hash=None, top_n=6):
    spec = strategy_spec(identity,top_n=top_n,max_weight=1/top_n,rebalance_sessions=5)
    ir = compile_strategy(spec,scores,snapshot,digest(market['execution']),
                          score_origin='walk_forward' if model_hash else 'factor',model_hash=model_hash)
    account = run_account(ir,market,start=request['research_start'],end=request['research_end'])
    return ir, account


def compare_methods(payload, base, candidates, request, *, seed=1708):
    snapshot, labels = snapshot_from_payload(payload['snapshot']), None
    labels = Labels(payload['labels'],snapshot)
    outputs, accounts = {}, {}
    for method in ('base','ordinary','screened'):
        fitted = fold_predictions(base,candidates,snapshot,labels,request,method=method,seed=seed)
        ir, account = account_for(fitted['scores'],snapshot,payload['market'],request,
                                 identity='selection_'+method,model_hash=digest(fitted['models']))
        outputs[method] = {**fitted,'strategy_ir':ir,'account':account}
        accounts[method] = account
    require(len({o['support_hash'] for o in outputs.values()}) == 1, 'method support changed')
    comparisons = {name:paired_accounts(accounts[name],accounts['base'],request) for name in ('ordinary','screened')}
    comparisons['selection_increment'] = paired_accounts(accounts['screened'],accounts['ordinary'],request)
    return {'schema_version':'fuxi.selection_ab.1','outputs':outputs,'comparisons':comparisons,
            'candidate_budget':len(candidates),'feature_budget':1,'methods_frozen_before_outcomes':True,
            'scope':'research_exposed_history; component/selection study; no single-factor qualification',
            'formal_selection_enabled':False}
