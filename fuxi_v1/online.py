"""Causal daily recomputation with the same native DSL and portfolio contract."""
from .contracts import digest, require, finite, timestamp
from .dsl import evaluate_ast
from .portfolio import target_weights, validate_strategy, mix_scores
from .a_share import exclude_new_entries,exclusion_facts_at


class DecisionView:
    def __init__(self, snapshot, session, observed_at, exclusions=()):
        index = snapshot.sessions.index(session)
        require(index >= snapshot.start, "decision before evaluation start")
        require(timestamp(observed_at) >= snapshot.clocks[session], "data not yet available")
        self.sessions = snapshot.sessions[:index+1]
        self.start = index
        self.universe = {session: list(snapshot.universe[session])}
        self.universe[session] = exclude_new_entries(self.universe[session],exclusions,snapshot.payload['decision_times'][session])
        self.clocks = {session: snapshot.clocks[session]}
        self.payload = {"fields":snapshot.payload["fields"]}
        self._source, self._index = snapshot, index
        self._score_cache = {}  # Never reuse full-history cached output.
        # Bind only material that could be read at this decision, including barriers.
        first = max(0,index-252)
        material={"session":session,"universe":self.universe,"clock":str(self.clocks[session]),
                  "fields":self.payload["fields"],
                  "exclusions":exclusion_facts_at(exclusions,snapshot.payload['decision_times'][session])}
        if snapshot.payload.get('schema_version')=='fuxi.historical_snapshot.1':
            material.update(instruments=snapshot.payload['instruments'],
                columns={k:v[first:index+1] for k,v in snapshot.payload['columns'].items()},
                barrier=snapshot.payload['barriers'][index])
        else:
            material['observations']=[r for r in snapshot.payload['observations']
                if r['session']<=session and timestamp(r['available_at'])<=self.clocks[session]]
        self.hash=digest(material)

    def value(self,index,instrument,field,asof):
        require(asof == self.clocks[self.sessions[-1]] and index <= self._index, "future access forbidden")
        return self._source.value(index,instrument,field,asof)


def daily_decision(ast, snapshot, session, observed_at, spec, *, baseline_ast=None,
                   added_weight=.5, prior=None, industries=None, exclusions=(), support_asts=()):
    validate_strategy(spec)
    if prior is not None:
        require(all(finite(v) and v >= 0 for v in prior.values()) and sum(prior.values()) <= 1+1e-9,
                "actual marked prior weights invalid")
    view = DecisionView(snapshot,session,observed_at,exclusions)
    scores = evaluate_ast(ast,view)
    if baseline_ast is not None:
        scores = mix_scores(evaluate_ast(baseline_ast,view),scores,added_weight)
    row = scores[session]
    support=[]
    if support_asts:
        require(isinstance(support_asts,(list,tuple)), 'support ASTs must be a frozen sequence')
        support=[evaluate_ast(a,view)[session] for a in support_asts]
        row={code:value if all(p.get(code) is not None for p in support) else None for code,value in row.items()}
    frozen=[]
    if prior and exclusions:
        allowed=exclude_new_entries(list(prior),exclusions,snapshot.payload['decision_times'][session])
        frozen=sorted(set(prior)-set(allowed))
    weights = target_weights(row,spec,prior=prior,industries=industries,frozen_prior=frozen)
    values = [v for v in row.values() if v is not None]
    body = {"schema_version":"fuxi.daily_decision.1","session":session,
            "decision_time":snapshot.payload["decision_times"][session],
            "input_hash":view.hash,"formula_hash":digest(ast),"strategy_hash":digest(spec),
            "baseline_hash":digest(baseline_ast) if baseline_ast else None,
            "added_weight":added_weight if baseline_ast else None,
            "prior_hash":digest(prior),"scores":row,"target_weights":weights,
            "retained_excluded_holdings":frozen,
            "missing_signal_hold":weights is None,
            "ranking_degenerate":bool(values) and len(set(values)) == 1,
            "observed_at":observed_at,"live_execution_allowed":False}
    if support_asts:body['support_hash']=digest({'asts':support_asts,
        'eligible':{code:all(p.get(code) is not None for p in support) for code in row}})
    body["decision_id"] = digest({k:v for k,v in body.items() if k != "observed_at"})
    return body


def daily_model_decision(feature_asts, state, snapshot, session, observed_at, spec, *, prior=None, support_asts=()):
    """Recompute causal features and consume an immutable trained model state."""
    from .models import predict_frozen_model, validate_frozen_model
    validate_frozen_model(state)
    validate_strategy(spec)
    require(feature_asts and len(feature_asts)==len(state.get('center',[])),
            'frozen feature/model shape mismatch')
    if prior is not None:
        require(all(finite(v) and v>=0 for v in prior.values()) and sum(prior.values())<=1+1e-9,
                'actual marked prior weights invalid')
    view=DecisionView(snapshot,session,observed_at)
    names=sorted(feature_asts)
    if 'features' in state:
        require(names==state['features'] and timestamp(state['training_cutoff'])<=snapshot.clocks[session],
                'frozen model features/training cutoff mismatch')
    values={name:evaluate_ast(feature_asts[name],view)[session] for name in names}
    require(isinstance(support_asts,(list,tuple)), 'support ASTs must be a frozen sequence')
    support=[evaluate_ast(a,view)[session] for a in support_asts]
    codes=view.universe[session];eligible=[];matrix=[]
    for code in codes:
        row=[values[name].get(code) for name in names]
        if all(v is not None and finite(v) for v in row) and all(p.get(code) is not None for p in support):
            eligible.append(code);matrix.append(row)
    scores={code:None for code in codes}
    if matrix:scores.update(zip(eligible,predict_frozen_model(state,matrix)))
    body={'schema_version':'fuxi.daily_model_decision.1','session':session,
        'decision_time':snapshot.payload['decision_times'][session],'input_hash':view.hash,
        'model_hash':state['model_hash'],'feature_names':names,'feature_hash':digest(feature_asts),
        'strategy_hash':digest(spec),'prior_hash':digest(prior),'scores':scores,
        'target_weights':target_weights(scores,spec,prior=prior),'observed_at':observed_at,
        'live_execution_allowed':False}
    if support_asts:body['support_hash']=digest({'asts':support_asts,
        'eligible':{code:all(p.get(code) is not None for p in support) for code in codes}})
    body['decision_id']=digest({k:v for k,v in body.items() if k!='observed_at'})
    return body
