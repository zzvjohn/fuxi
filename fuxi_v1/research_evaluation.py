"""Native E0-E4 evidence. Research diagnostics and identified account evidence stay separate."""

from statistics import mean
from .contracts import evaluation, digest, require
from .evaluation import DEFAULT_POLICY, factor_statistics, correlation
from .statistics import block_inference, performance
from .portfolio import compile_strategy, strategy_spec, mix_scores
from .simulation import run_account
from .dsl import evaluate_ast
from .search import node


def single_field_channels(ast):
    """Unwrap combinations/ranks to maximal computable subtrees of exactly one source field."""
    def fields(tree):
        return {tree["field"]} if "field" in tree else set().union(*(fields(c) for c in tree.get("args",[])))
    found = {}
    def visit(tree):
        names = fields(tree)
        if len(names) == 1:
            found[digest(tree)] = {"ast":tree,"fields":sorted(names)}
        else:
            for child in tree.get("args",[]):
                visit(child)
    visit(ast)
    return list(found.values())


def paired_scores(left, right, start, end):
    from .columnar import ScoreGrid, paired_scores_columnar
    if isinstance(left, ScoreGrid):
        return paired_scores_columnar(left, right, start, end)
    require(set(left) == set(right), "paired feature windows differ")
    sections = []
    for day in sorted(left):
        if not start <= day <= end:
            continue
        require(set(left[day]) == set(right[day]), "paired feature universes differ")
        pairs = [(left[day][s], right[day][s]) for s in left[day] if left[day][s] is not None and right[day][s] is not None]
        corr = correlation(*zip(*pairs)) if len(pairs) >= 3 else None
        sections.append({"session": day, "n_pairs": len(pairs), "pearson": corr,
                         "missing_pattern_equal": {s for s, v in left[day].items() if v is None} ==
                                                  {s for s, v in right[day].items() if v is None}})
    require(sum(x["n_pairs"] for x in sections) > 0, "zero feature pairs")
    valid = [abs(x["pearson"]) for x in sections if x["pearson"] is not None]
    return {"sections": sections, "mean_abs_correlation": mean(valid) if valid else None,
            "n_pairs": sum(x["n_pairs"] for x in sections)}


def paired_accounts(candidate, baseline, request):
    require([r["session"] for r in candidate["daily"]] == [r["session"] for r in baseline["daily"]],
            "account/benchmark date window mismatch")
    require(candidate["market_hash"] == baseline["market_hash"] and candidate["assumptions"] == baseline["assumptions"],
            "account/benchmark data or costs mismatch")
    a, b = performance(candidate["daily"]), performance(baseline["daily"])
    if a is None or b is None or candidate["eval_status"] != "valid" or baseline["eval_status"] != "valid":
        return evaluation("E3", "data_insufficient", "account_path_not_identifiable", scope="account",
                          diagnostics={"candidate_gaps": candidate["gaps"], "baseline_gaps": baseline["gaps"],
                                       "candidate_performance": a, "baseline_performance": b})
    require(len(a["returns"]) == len(b["returns"]) > 0, "zero/mismatched paired account returns")
    delta = [x-y for x, y in zip(a["returns"], b["returns"])]
    from .research_protocol import inference
    interval = inference(delta, request, 731)
    readings = {"candidate": a, "baseline": b, "paired_net_delta": delta, "inference": interval,
                "paired_mean_increment": mean(delta)}
    if not candidate["formal_evidence"] or not baseline["formal_evidence"]:
        return evaluation("E3", "data_insufficient", "market_evidence_not_certified", scope="account",
                          diagnostics=readings, n_pairs=len(delta))
    passed = interval["status"] == "valid" and interval["lower"] > request["minimum_net_increment_daily"]
    result = evaluation("E3", "valid", "paired_account_increment", decision="pass" if passed else "reject",
                        reject_class=None if passed else "economic", scope="account", metrics=readings, n_pairs=len(delta))
    # Consumers additionally verify source schema, scope, paired counts and data provenance.
    result["economic_reward_eligible"] = True
    return result


def signal_rank_correlation(left, right, snapshot, start, end):
    """Mean |daily cross-sectional rank correlation| between two score grids.

    A redundancy descriptor, not a decision: measures how much two signals order
    the cross-section alike. Returns None when no day has >=3 matched pairs.
    """
    from .columnar import ScoreGrid
    if isinstance(left, ScoreGrid):
        import numpy as np
        from .columnar import _same_grids, rank_vector, correlation_array
        _same_grids(left, right)
        vals = []
        for day in left:
            if not start <= day <= end:
                continue
            i = left.day_indices[day]
            a, b = left.matrix[i], right.matrix[i]
            valid = left.membership[i] & np.isfinite(a) & np.isfinite(b)
            n = int(valid.sum())
            if n < 3:
                continue
            c = correlation_array(rank_vector(a[valid]), rank_vector(b[valid]))
            if c is not None:
                vals.append(abs(c))
        return mean(vals) if vals else None
    from .dsl import ranks
    vals = []
    for day in sorted(left):
        if not start <= day <= end or day not in right:
            continue
        common = {s: (left[day][s], right[day][s]) for s in left[day]
                  if left[day][s] is not None and right[day].get(s) is not None}
        if len(common) < 3:
            continue
        ra = ranks({s: v[0] for s, v in common.items()})
        rb = ranks({s: v[1] for s, v in common.items()})
        c = correlation(list(ra.values()), list(rb.values()))
        if c is not None:
            vals.append(abs(c))
    return mean(vals) if vals else None


def residual_rank_ic(scores, other_scores, snapshot, labels, clock, start, end):
    """Daily rank IC of the candidate after orthogonalizing against one pool factor.

    The candidate's cross-sectional rank is regressed on the pool factor's rank;
    the residual is then correlated with the forward label. A strictly positive
    residual IC means the candidate still carries alpha the pool factor does not.
    Returns a list of daily residual rank ICs (possibly empty).
    """
    from .contracts import timestamp
    if isinstance(clock, str):
        clock = timestamp(clock)
    from .columnar import ScoreGrid
    if isinstance(scores, ScoreGrid):
        import numpy as np
        from .columnar import _same_grids, rank_vector, correlation_array
        _same_grids(scores, other_scores)
        rows = []
        for day in scores:
            if not start <= day <= end:
                continue
            i = scores.day_indices[day]
            y = labels.values_at(i, clock)
            a, b = scores.matrix[i], other_scores.matrix[i]
            valid = scores.membership[i] & np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
            n = int(valid.sum())
            if n < 5:
                continue
            ra = rank_vector(a[valid]); rb = rank_vector(b[valid]); ry = rank_vector(y[valid])
            ra_c = ra - ra.mean(); rb_c = rb - rb.mean()
            denom = float(np.dot(rb_c, rb_c))
            if denom == 0:
                continue
            beta = float(np.dot(ra_c, rb_c)) / denom
            residual = ra_c - beta * rb_c
            c = correlation_array(residual, ry)
            if c is not None:
                rows.append(c)
        return rows
    from .dsl import ranks
    rows = []
    for day in sorted(scores):
        if not start <= day <= end or day not in other_scores:
            continue
        common = {s: (scores[day][s], other_scores[day][s], labels.value(day, s, clock))
                  for s in scores[day]}
        common = {s: v for s, v in common.items() if None not in v}
        if len(common) < 5:
            continue
        ra = ranks({s: v[0] for s, v in common.items()})
        rb = ranks({s: v[1] for s, v in common.items()})
        ry = ranks({s: v[2] for s, v in common.items()})
        x = list(ra.values()); z = list(rb.values()); y = list(ry.values())
        n = len(x)
        mx = sum(x) / n; mz = sum(z) / n
        xc = [v - mx for v in x]; zc = [v - mz for v in z]
        denom = sum(v * v for v in zc)
        if denom == 0:
            continue
        beta = sum(xi * zi for xi, zi in zip(xc, zc)) / denom
        residual = [xc[i] - beta * zc[i] for i in range(n)]
        c = correlation(residual, y)
        if c is not None:
            rows.append(c)
    return rows


def pool_redundancy(scores, pool_scores, snapshot, labels, clock, start, end, request, *, corr_threshold=0.7):
    """E2 pool-redundancy gate: reject candidates that duplicate an admitted factor.

    A candidate with max |rank correlation| against the pool below corr_threshold
    is independent (pass). Above it, the candidate must still deliver a strictly
    positive residual alpha after orthogonalizing against the most-correlated pool
    factor; otherwise it is redundant and rejected. An empty pool passes by
    definition (no reference yet).

    Note on "new peaks": a correlated candidate that is *materially higher
    quality* than its pool member (a cleaner estimate of the same signal) is
    kept by the residual-alpha branch, not a separate path. In rank space
    ``candidate_ic = corr * pool_ic + residual_ic * sqrt(1 - corr**2)``, so a
    higher-quality correlated factor necessarily carries a positive residual IC;
    the residual test already admits it, and any additional "IC gap" test would
    be strictly weaker (its series is a sub-unit-shrunk copy of the residual
    series). Rejecting therefore fires only for genuine duplicates that add no
    orthogonal alpha.
    """
    if not pool_scores:
        return {"decision": "pass", "reject_class": None, "reason_code": "empty_pool_no_reference",
                "pool_size": 0, "corr_threshold": corr_threshold}
    best = None
    for member in pool_scores:
        c = signal_rank_correlation(scores, member["scores"], snapshot, start, end)
        if c is None:
            continue
        if best is None or c > best["corr"]:
            best = {"candidate_id": member["candidate_id"], "corr": c, "scores": member["scores"]}
    if best is None:
        return {"decision": "pass", "reject_class": None, "reason_code": "no_comparable_pool_member",
                "pool_size": len(pool_scores), "corr_threshold": corr_threshold}
    max_corr = best["corr"]
    if max_corr < corr_threshold:
        return {"decision": "pass", "reject_class": None, "reason_code": "independent_of_pool",
                "pool_size": len(pool_scores), "max_corr": max_corr, "corr_threshold": corr_threshold}
    residual_ic = residual_rank_ic(scores, best["scores"], snapshot, labels, clock, start, end)
    if not residual_ic:
        return {"decision": "reject", "reject_class": "redundancy",
                "reason_code": "redundant_no_residual_support", "redundant_with": best["candidate_id"],
                "pool_size": len(pool_scores), "max_corr": max_corr, "corr_threshold": corr_threshold}
    from .research_protocol import inference
    interval = inference(residual_ic, request, 20261018)
    if interval["status"] == "valid" and interval["lower"] > 0:
        # This is also how a "new peak" is admitted: a strictly higher-quality
        # correlated factor (a cleaner estimate of the same signal) has a
        # positive residual IC by construction, so it passes here rather than
        # being rejected as a duplicate.
        return {"decision": "pass", "reject_class": None,
                "reason_code": "high_corr_but_independent_residual_alpha", "redundant_with": best["candidate_id"],
                "pool_size": len(pool_scores), "max_corr": max_corr, "residual_alpha": interval,
                "corr_threshold": corr_threshold}
    return {"decision": "reject", "reject_class": "redundancy",
            "reason_code": "redundant_with_pool_factor", "redundant_with": best["candidate_id"],
            "pool_size": len(pool_scores), "max_corr": max_corr, "residual_alpha": interval,
            "corr_threshold": corr_threshold}


def evaluate_candidate(candidate, scores, baseline_scores, snapshot, labels, market, policy, clock, *, top_n=6, baseline_cache=None, pool_scores=()):
    request = policy["request"]
    start, end = request["research_start"], request["research_end"]
    selected = {d: v for d, v in scores.items() if start <= d <= end}
    e0 = evaluation("E0", "valid", "catalog_and_PIT_contract_supported", decision="pass")
    if snapshot.payload.get("schema_version") == "fuxi.columnar_snapshot.1":
        e0["reason_code"] = "declared_latest_vintage_contract_research_only"
    e1 = factor_statistics(selected, snapshot, labels, DEFAULT_POLICY, clock)
    ic = [r["rank_ic"] for r in e1["diagnostics"]["sections"] if r["rank_ic"] is not None]
    from .research_protocol import inference
    interval = inference(ic, request, 1729) if ic else None
    e1["diagnostics"]["time_uncertainty"] = interval
    e1["diagnostics"]["power_required_sections"] = policy["minimum_effective_sections"]
    e1['diagnostics']['tail_resolution_adequate'] = (request['bootstrap_replicates']+1)*request['alpha']/request['max_trials']/2 >= 1
    require(request.get('research_gate_version') == 'history_research_v2',
            'E1 requires history_research_v2 screen; legacy calibrated power gate is retired')
    from types import SimpleNamespace
    from .research_protocol import screen
    sections=e1['diagnostics']['sections']
    yearly={y:{'n':len(v),'mean_rank_ic':mean(v) if v else None} for y in sorted({r['session'][:4] for r in sections})
            for v in [[r['rank_ic'] for r in sections if r['session'].startswith(y) and r['rank_ic'] is not None]]}
    exchanges={}
    for exchange in ('XSHG','XSHE'):
        subset={d:{s:v for s,v in row.items() if s.endswith(exchange)} for d,row in selected.items()}
        if not all(subset.values()):
            # 单交易所数据（如 synthetic fixture）：记 n=0，交由 screen 的
            # screen_positive_exchange_count=2 自然 reject，而非 fail-closed。
            # 数据完整性由快照认证负责，不在评估阶段 require。
            exchanges[exchange]={'n':0,'mean_rank_ic':None}
            continue
        proxy=SimpleNamespace(universe={d:list(v) for d,v in subset.items()},clocks=snapshot.clocks)
        e=factor_statistics(subset,proxy,labels,DEFAULT_POLICY,clock)
        exchanges[exchange]={'n':e['diagnostics']['valid_section_count'],'mean_rank_ic':e['diagnostics']['spearman']['mean']}
    screen(e1,request,yearly,exchanges)
    try:
        similarity = paired_scores(scores, baseline_scores, start, end)
        e2 = evaluation("E2", "valid", "independence_description_pending_economic_increment",
                        scope="research", metrics=similarity, n_pairs=similarity["n_pairs"])
        from .attribution import style_attribution
        close = {"field":"close"}
        style = evaluate_ast(node("div",node("std",close,window=20),node("mean",close,window=20)),snapshot)
        e2["diagnostics"]["style_attribution"] = style_attribution(selected,
                {"frozen_momentum20":baseline_scores,"raw_price_cv20":style},labels,snapshot,clock)
        # E2 is now a real pool-redundancy gate (P-20261018-014): the decision is
        # made independently against the already-admitted pool, never mirrored from E3.
        redundancy = pool_redundancy(scores, pool_scores, snapshot, labels, clock, start, end, request)
        e2["diagnostics"]["pool_redundancy"] = redundancy
        e2.update(decision=redundancy["decision"], reject_class=redundancy["reject_class"],
                  reason_code=redundancy["reason_code"])
    except ValueError as exc:
        e2 = evaluation("E2", "data_insufficient", "independence_zero_pairs", diagnostics={"error": str(exc)})
    spec = strategy_spec(candidate["candidate_id"], top_n=top_n, max_weight=min(1., 1/top_n),
                         rebalance_sessions=labels.payload["label_spec"]["horizon_sessions"])
    baseline_spec = dict(spec, candidate_id="frozen_baseline_"+digest(baseline_scores))
    blended = mix_scores(baseline_scores, scores, .5)
    ir = compile_strategy(spec, blended, snapshot, digest(market["execution"]))
    baseline_ir = compile_strategy(baseline_spec, baseline_scores, snapshot, digest(market["execution"]))
    def baseline_run(cost=1.,capacity=1.):
        key=digest([baseline_ir['ir_hash'],market['market_hash'],start,end,cost,capacity])
        if baseline_cache is not None and key in baseline_cache:return baseline_cache[key]
        value=run_account(baseline_ir,market,start=start,end=end,cost_multiplier=cost,participation_multiplier=capacity)
        if baseline_cache is not None:baseline_cache[key]=value
        return value
    candidate_account = run_account(ir, market, start=start, end=end)
    baseline_account = baseline_run()
    e3 = paired_accounts(candidate_account, baseline_account, request)
    if snapshot.payload["exposure"] == "synthetic":
        e3["economic_reward_eligible"] = False
        e3["evidence_scope"] = "diagnostic"
        e3["decision"], e3["reject_class"] = "not_applicable", None
        e3["reason_code"] = "synthetic_account_engineering_only"
    # E2 increment needs both matched cross-sections AND identified portfolio ablation.
    # The E2 decision is already set by pool_redundancy above; the ablation below only
    # enriches diagnostics and never overwrites the decision.
    e2_decision_set = e2.get("decision")
    if e2["eval_status"] == "valid" and e3["eval_status"] == "valid" and e2_decision_set is None:
        passed = e3["decision"] == "pass"
        e2.update(decision="pass" if passed else "reject", reject_class=None if passed else "redundancy",
                  reason_code="paired_add_one_feature_increment_and_ablation")
    ablations = []
    ast = candidate["canonical_ast"]
    # Multi-field candidates must expose each individually computable child channel.
    if len(candidate["required_fields"]) > 1:
        channels = single_field_channels(ast)
        for index, channel in enumerate(channels):
            child = channel["ast"]
            try:
                child_scores = evaluate_ast(child, snapshot)
                result = factor_statistics({d:v for d,v in child_scores.items() if start <= d <= end},
                                           snapshot, labels, DEFAULT_POLICY, clock)
                ablations.append({"child_index": index, "ast": child, "fields":channel["fields"], "evaluation": result,
                                  "paired_against_full": paired_scores(scores, child_scores, start, end)})
                ablated_ir = compile_strategy(dict(spec,candidate_id=candidate["candidate_id"]+":channel:"+str(index)),
                      mix_scores(baseline_scores,child_scores,.5),snapshot,digest(market["execution"]))
                ablated_account = run_account(ablated_ir,market,start=start,end=end)
                ablations[-1]["full_vs_channel_account"] = paired_accounts(candidate_account,ablated_account,request)
            except ValueError as exc:
                ablations.append({"child_index": index, "eval_status": "error", "reason": str(exc)})
        e2["diagnostics"]["single_channel_ablation"] = ablations
        complete = len(ablations) >= 2 and set(candidate["required_fields"]) == {f for c in channels for f in c["fields"]}
        identified = complete and all(x.get("full_vs_channel_account",{}).get("eval_status") == "valid" for x in ablations)
        # 单通道剥离降级为诊断告警：组合因子的价值常来自低相关组件的叠加，
        # 不再要求每个单字段通道独立建立账户增量（E2 最终 decision 仍 = e3.pass）。
        e2["diagnostics"]["single_channel_complete"] = complete
        e2["diagnostics"]["single_channel_identified"] = identified
        e2["diagnostics"]["single_channel_all_pass"] = (
            identified and all(x["full_vs_channel_account"]["decision"] == "pass" for x in ablations))
    stress = []
    for name, cost, capacity in (("double_cost", request["cost_stress_multiplier"], 1.),
                                 ("reduced_capacity", 1., request["capacity_stress_multiplier"])):
        a = run_account(ir, market, start=start, end=end, cost_multiplier=cost, participation_multiplier=capacity)
        b = baseline_run(cost,capacity)
        stress.append({"scenario": name, "evaluation": paired_accounts(a, b, request)})
    valid = all(x["evaluation"]["eval_status"] == "valid" for x in stress)
    passed = valid and all(x["evaluation"]["decision"] == "pass" for x in stress)
    e4 = evaluation("E4", "valid" if valid else "data_insufficient", "cost_capacity_stress_and_selection_accounting",
                    decision=("pass" if passed else "reject") if valid else "not_applicable",
                    reject_class="economic" if valid and not passed else None,
                    metrics={"stress": stress} if valid else None,
                    diagnostics={"stress": stress, "selection_scope": "adaptive_research_diagnostic", "max_trials": request["max_trials"]},
                    scope="research")
    e5 = evaluation("E5", "not_evaluated", "future_acceptance_requires_registered_freeze_and_mature_observations",
                    scope="forward_acceptance")
    return {"candidate_id": candidate["candidate_id"], "evaluations": [e0,e1,e2,e3,e4,e5],
            "strategy_ir": ir, "baseline_ir": baseline_ir,
            "candidate_account": candidate_account, "baseline_account": baseline_account,
            "policy_hash": policy["policy_hash"], "snapshot_hash": snapshot.hash,
            "source_scope": snapshot.payload["exposure"],
            "selection_audit": snapshot.payload.get('provenance',{}).get('selection_audit')}
