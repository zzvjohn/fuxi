"""Declarative long-only StrategyIR shared by native execution and JQ compilation."""

from .contracts import digest, keys, require, finite


def strategy_spec(candidate_id, *, top_n=6, rebalance_sessions=5, max_weight=.2,
                  turnover_budget=1., gross_exposure=1., industry_cap=1.):
    return {"schema_version": "fuxi.strategy.1", "candidate_id": candidate_id,
            "top_n": top_n, "rebalance_sessions": rebalance_sessions, "max_weight": max_weight,
            "turnover_budget": turnover_budget, "gross_exposure": gross_exposure, "industry_cap": industry_cap,
            "weight_rule": "equal_rank_ties_prorata", "signal_rule": "after_declared_availability",
            "execution_rule": "next_session_open", "exit_rule": "rebalance_to_target",
            "unfilled_order_policy": "retain_positions_cash_no_reallocation", "missing_signal": "hold"}


def validate_strategy(spec):
    keys(spec, strategy_spec("x").keys())
    require(spec["schema_version"] == "fuxi.strategy.1", "unknown strategy schema")
    for key in ("top_n", "rebalance_sessions"):
        require(type(spec[key]) is int and spec[key] > 0, "positive integer strategy parameter required")
    for key in ("max_weight", "gross_exposure", "industry_cap", "turnover_budget"):
        require(finite(spec[key]) and 0 < spec[key] <= 1, "invalid portfolio constraint")
    template = strategy_spec("x")
    for key in ("weight_rule", "signal_rule", "execution_rule", "exit_rule", "unfilled_order_policy", "missing_signal"):
        require(spec[key] == template[key], "unsupported strategy semantics")
    return digest(spec)


def target_weights(values, spec, prior=None, industries=None, frozen_prior=()):
    validate_strategy(spec)
    valid = sorted(((v, s) for s, v in values.items() if v is not None), key=lambda x: (-x[0], x[1]))
    if not valid:
        return None
    n = min(spec["top_n"], len(valid))
    cutoff = valid[n-1][0]
    above = [s for v, s in valid if v > cutoff]
    tied = [s for v, s in valid if v == cutoff]
    target = {s: min(spec["gross_exposure"]/n, spec["max_weight"]) for s in above}
    for s in tied:
        target[s] = min(spec["gross_exposure"]*(n-len(above))/n/len(tied), spec["max_weight"])
    if spec["industry_cap"] < 1:
        require(industries is not None and set(target) <= set(industries), "industry constraints need PIT classifications")
        for group in set(industries[s] for s in target):
            members = [s for s in target if industries[s] == group]
            gross = sum(target[s] for s in members)
            if gross > spec["industry_cap"]:
                for s in members:
                    target[s] *= spec["industry_cap"]/gross
    frozen_prior=set(frozen_prior)
    require(not frozen_prior or prior is not None and frozen_prior<=set(prior), 'frozen holdings need marked prior')
    if frozen_prior:
        locked=sum(prior[s] for s in frozen_prior)
        target={s:w for s,w in target.items() if s not in frozen_prior}
        free=sum(target.values());budget=max(0.,spec['gross_exposure']-locked)
        if free>budget:
            target={s:w*budget/free for s,w in target.items()}
        target.update({s:prior[s] for s in frozen_prior})
    # True turnover uses actual pre-trade marked portfolio, supplied by the executor.
    if prior is not None:
        names = set(target) | set(prior)
        distance = sum(abs(target.get(s, 0)-prior.get(s, 0)) for s in names)
        scale = min(1., spec["turnover_budget"]/distance) if distance else 1.
        target = {s: prior.get(s, 0)+scale*(target.get(s, 0)-prior.get(s, 0)) for s in names}
    return target


def compile_strategy(spec, scores, snapshot, execution_hash, *, score_origin="factor", model_hash=None):
    validate_strategy(spec)
    require(score_origin in {"factor", "walk_forward", "forward"}, "in-sample predictions cannot enter performance IR")
    decisions = []
    for i in range(snapshot.start, len(snapshot.sessions), spec["rebalance_sessions"]):
        if i+1 >= len(snapshot.sessions):
            continue
        day, next_day = snapshot.sessions[i], snapshot.sessions[i+1]
        values = scores.get(day, {})
        require(set(values) == set(snapshot.universe[day]), "strategy scores/universe mismatch")
        decisions.append({"decision_time": snapshot.payload["decision_times"][day], "signal_session": day,
                          "execution_session": next_day, "first_order_time": next_day+"T09:30:00+08:00",
                          "scores": values, "universe": snapshot.universe[day]})
    result = {"schema_version": "fuxi.strategy_ir.1", "strategy": spec, "snapshot_hash": snapshot.hash,
              "execution_hash": execution_hash, "model_hash": model_hash,
              "score_origin": score_origin, "decisions": decisions}
    result["ir_hash"] = digest(result)
    return result


def mix_scores(primary, added, added_weight=.5):
    from .columnar import ScoreGrid, mix_scores_columnar
    if isinstance(primary, ScoreGrid):
        return mix_scores_columnar(primary, added, added_weight)
    require(0 <= added_weight <= 1 and set(primary) == set(added), "mix scope mismatch")
    from .dsl import ranks
    result = {}
    for day in primary:
        require(set(primary[day]) == set(added[day]), "mix universe mismatch")
        a, b = ranks(primary[day]), ranks(added[day])
        result[day] = {s: (1-added_weight)*a[s]+added_weight*b[s] if a[s] is not None and b[s] is not None
                       else None for s in a}
    return result


def portfolio_experiment(payload, policy, scores, top_n):
    """Frozen features/predictions; only the rebalance interval changes, with actual account runs."""
    from .data import Snapshot
    from .simulation import run_account
    from .research_evaluation import paired_accounts
    snapshot, market = Snapshot(payload["snapshot"]), payload["market"]
    parent = strategy_spec("portfolio_baseline",top_n=top_n,max_weight=1/top_n,
                           rebalance_sessions=payload["labels"]["label_spec"]["horizon_sessions"])
    child = dict(parent,rebalance_sessions=parent["rebalance_sessions"]*2)
    require([k for k in parent if parent[k] != child[k]] == ["rebalance_sessions"],"portfolio experiment must change one axis")
    a = compile_strategy(parent,scores,snapshot,digest(market["execution"]))
    b = compile_strategy(child,scores,snapshot,digest(market["execution"]))
    request = policy["request"]
    aa = run_account(a,market,start=request["research_start"],end=request["research_end"])
    bb = run_account(b,market,start=request["research_start"],end=request["research_end"])
    result = paired_accounts(bb,aa,request)
    if snapshot.payload["exposure"] == "synthetic":
        result.update(economic_reward_eligible=False,evidence_scope="diagnostic",decision="not_applicable",reject_class=None)
    return {"axis":"portfolio","changed_keys":["rebalance_sessions"],"scores_hash":digest(scores),
            "parent_ir":a,"child_ir":b,"parent_account":aa,"child_account":bb,"comparison":result,
            "scope":"research_only","automatic_default_change":False}
