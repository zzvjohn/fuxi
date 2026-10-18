"""Capability-constrained hypotheses, local mutations, fixed baselines and a bounded LLM adapter."""

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import time
import urllib.request
from .contracts import digest, require, keys, ContractError
from .dsl import CATALOG, compile_ast


def node(op, *args, window=None):
    result = {"op": op, "args": list(args)}
    if window is not None:
        result["window"] = window
    return result


def templates(fields):
    close = {"field": "close"}
    found = []
    for w in (2, 5, 10, 20):
        for sign in (1, -1):
            formula = node("rank", node("return", close, window=w))
            found.append(("price_persistence" if sign > 0 else "short_reversal", formula if sign > 0 else node("neg", formula)))
        found.append(("price_range", node("rank", node("div", close, node("mean", close, window=w)))))
        if w >= 2:
            found.append(("volatility", node("neg", node("rank", node("div", node("std", close, window=w), node("mean", close, window=w))))))
        if "volume" in fields:
            volume = {"field": "volume"}
            found.append(("volume_surprise", node("rank", node("div", volume, node("mean", volume, window=w)))))
            found.append(("price_volume_disagreement", node("sub", node("rank", node("return", close, window=w)),
                          node("rank", node("div", volume, node("mean", volume, window=w))))))
    return found


def templates_expanded(fields):
    """Field-space expansion: legacy six mechanisms plus templates covering the other
    eight snapshot fields (amount/open/high/low/turnover_rate/float_shares/free_shares/
    total_shares). Every formula passes DSL unit checks (ratio/price/shares/currency)."""
    close = {"field": "close"}
    found = templates(fields)
    # Structural "calm-premium" leads (P-20261017): probe-verified on the frozen
    # full-A daily contract (probe_structural_factors.py). Lead-ordered so the
    # fixed arm evaluates them first -- close_cv5 primary, low_turnover fallback.
    # P-20261017 (narrow admission): the original narrow_range lead
    # neg(rank((high-low)/close)) is structurally blocked at E3 by the event-coverage
    # boundary (corporate_action_known is an adj_factor-unchanged proxy only, see
    # datasets.py). Amplitude-minimal stocks skew toward ex-dividend names whose bars
    # carry corp_known=False, chaining pretrade_NAV_unidentifiable (33 gaps, 157
    # unidentifiable sessions). close_cv5 (5-day close CV, same calm-premium family)
    # sidesteps the gap and is STRONGER: ICIR 0.341 vs 0.333, E3 pure-long daily_mean
    # +0.000453, fully identifiable account (0 gaps).
    leads = []
    if "close" in fields:
        leads.append(("close_cv5", node("neg", node("rank", node("div",
            node("std", {"field": "close"}, window=5), {"field": "close"})))))
    if "turnover_rate" in fields:
        leads.append(("low_turnover", node("neg", node("rank", {"field": "turnover_rate"}))))
    found = leads + found
    for w in (2, 5, 10, 20):
        if "amount" in fields:
            amount = {"field": "amount"}
            found.append(("amount_surprise", node("rank", node("div", amount, node("mean", amount, window=w)))))
            found.append(("price_amount_disagreement", node("sub", node("rank", node("return", close, window=w)),
                          node("rank", node("div", amount, node("mean", amount, window=w))))))
        if "turnover_rate" in fields:
            turnover = {"field": "turnover_rate"}
            found.append(("turnover_surprise", node("rank", node("div", turnover, node("mean", turnover, window=w)))))
            if w >= 2:
                found.append(("turnover_stability", node("neg", node("rank", node("div", node("std", turnover, window=w), node("mean", turnover, window=w))))))
        if "high" in fields and "low" in fields:
            high, low = {"field": "high"}, {"field": "low"}
            found.append(("intraday_range", node("rank", node("div", node("sub", high, low), close))))
            found.append(("close_location", node("rank", node("div", node("sub", close, low), node("sub", high, low)))))
        if "open" in fields:
            open_ = {"field": "open"}
            found.append(("open_surprise", node("rank", node("div", open_, node("mean", open_, window=w)))))
        if "float_shares" in fields:
            shares = {"field": "float_shares"}
            found.append(("size_proxy", node("neg", node("rank", node("mean", shares, window=w)))))
        if w >= 2:
            found.append(("price_zscore", node("rank", node("div", node("sub", close, node("mean", close, window=w)), node("std", close, window=w)))))
    # --- P-20261018 diversity expansion ----------------------------------
    # 冷启动单族坍缩修复：现有族补维度 + 补入此前缺失的族群
    # (technical / behavioral / crowding / multi_scale)。所有 AST 均过 DSL
    # 单元校验（time op 只接受 field leaf；add/sub 同 unit；div 跨 unit 合法）。
    # shuffle_source 下与 legacy 模板同池均匀采样，不设 lead 偏置。
    diverse = []
    if "close" in fields:
        # technical 趋势族（缺失）：均线方向 + 价格乖离
        diverse.append(("ma_trend", node("rank", node("div",
            node("mean", close, window=5), node("mean", close, window=20)))))
        diverse.append(("ma_deviation", node("zscore", node("sub", close, node("mean", close, window=20)))))
        # volatility 族补维度：波动率状态 + 短长波动率比
        diverse.append(("vol_zscore", node("zscore", node("std", close, window=5))))
        diverse.append(("vol_regime", node("rank", node("div",
            node("std", close, window=5), node("std", close, window=20)))))
        # momentum 族补维度：多尺度动量差（趋势加速）
        diverse.append(("momentum_accel", node("sub",
            node("rank", node("return", close, window=5)), node("rank", node("return", close, window=20)))))
        # behavioral 行为金融族（缺失）：距高/低点锚定（处置效应 / 抄底）
        diverse.append(("near_high", node("rank", node("div", node("max", close, window=20), close))))
        diverse.append(("near_low", node("rank", node("div", close, node("min", close, window=20)))))
    if "volume" in fields:
        vol = {"field": "volume"}
        # price_volume 族补维度：量能突变 + 量能极端
        diverse.append(("volume_delta", node("rank", node("delta", vol, window=5))))
        diverse.append(("volume_zscore", node("zscore", node("div", vol, node("mean", vol, window=20)))))
    if "amount" in fields:
        # price_volume 族补维度：成交额极端
        diverse.append(("amount_zscore", node("zscore", {"field": "amount"})))
    if "turnover_rate" in fields:
        # liquidity 补维度 + crowding 拥挤族（缺失）：换手变化 + 换手极端
        tr = {"field": "turnover_rate"}
        diverse.append(("turnover_delta", node("rank", node("delta", tr, window=5))))
        diverse.append(("turnover_zscore", node("zscore", tr)))
    if "high" in fields and "low" in fields:
        # geometry 族补维度：振幅极端
        diverse.append(("range_zscore", node("zscore", node("div",
            node("sub", {"field": "high"}, {"field": "low"}), close))))
    if "float_shares" in fields:
        # fundamental(size) 族补维度：规模极端
        diverse.append(("size_zscore", node("zscore", {"field": "float_shares"})))
    found = found + diverse
    return found


def regime_complement_pairs(parents, profiles, count, seen=()):
    """Encode the JQ cross-year complementarity mechanism (comb_skew_vot_ar_pvt doubled
    fd_skewness via odd/even-year regime alternation) as a parent-pairing primitive.

    A pair is complementary when per-year max(a_y, b_y) summed strictly exceeds the
    better single parent's year sum -- i.e. the two components alternate dominance
    across years. Pairs whose resulting combination formula is already seen are
    excluded (a novel pair must yield a novel formula). Deterministic ordering
    (no rng); tie-break by digest of the pair ids.
    """
    known = set(seen)
    scored = []
    for i in range(len(parents)):
        for j in range(i + 1, len(parents)):
            a, b = parents[i], parents[j]
            pa, pb = profiles.get(a["candidate_id"]), profiles.get(b["candidate_id"])
            if not pa or not pb:
                continue
            years = sorted(set(pa) & set(pb))
            if len(years) < 2:
                continue
            pair_sum = sum(max(pa[y], pb[y]) for y in years)
            better = max(sum(pa[y] for y in years), sum(pb[y] for y in years))
            if pair_sum <= better:
                continue
            combined = node("add", node("rank", a["canonical_ast"]), node("rank", b["canonical_ast"]))
            if digest(combined) in known:
                continue
            scored.append((pair_sum - better, a, b))
    scored.sort(key=lambda t: (-t[0], digest([t[1]["candidate_id"], t[2]["candidate_id"]])))
    return [(a, b) for _, a, b in scored[:count]]


def factor_spec(ast, fields, mechanism, *, parent_ids=(), action="template", source_ids=(), mutation=None):
    compiled = compile_ast(ast, fields)
    result = {"schema_version": "fuxi.factor.2", "canonical_ast": deepcopy(ast), **compiled,
              "hypothesis": {"mechanism": mechanism, "claim_type": "correlational_hypothesis",
                             "competing_explanations": ["market_or_industry_exposure", "liquidity_or_selection_artifact"],
                             "falsification": ["paired_baseline_increment", "single_channel_ablation", "time_block_uncertainty", "cost_stress"]},
              "frequency": "daily", "missing_policy": "propagate", "parent_ids": list(parent_ids),
              "source_evidence_ids": list(source_ids), "mutation_patch": mutation,
              "action": action, "role": "incremental_feature", "decision_time_rule": "snapshot_declared_availability"}
    result["candidate_id"] = "factor_"+digest(result)
    return result


def generate(fields, *, arm, count, seed, parents=(), seen=(), diagnostics=None, counters=None,
             regime_profiles=None, shuffle_source=False):
    require(arm in {"fixed", "random", "refine", "combine", "random_ast", "regime_combine"},
            "unknown local generator")
    gen_expand = os.environ.get("FUXI_V1_GEN_EXPAND", "legacy")
    require(gen_expand in {"legacy", "expanded"}, "unknown generator expansion mode")
    require(arm != "regime_combine" or gen_expand == "expanded",
            "regime_combine arm requires the expanded generator")
    rng, proposals = random.Random(seed), []
    if counters is not None:
        counters.update(generation_attempts=0, compiler_rejected=0, parameter_attempts=0)
    def propose(ast, mechanism, **metadata):
        if counters is not None:
            counters["generation_attempts"] += 1
        try:
            proposals.append(factor_spec(ast, fields, mechanism, **metadata))
        except ContractError as exc:
            if counters is not None:
                counters["compiler_rejected"] += 1
            # A generated formula can exceed the catalog; the catalog stays fixed.
            # Do not hide programming/IO errors or broaden the bounded proposal set.
            if diagnostics is not None:
                diagnostics.append({'status':'compiler_rejected','reason':str(exc),'ast_hash':digest(ast),
                    'action':metadata.get('action'),'parent_ids':list(metadata.get('parent_ids',()))})
    source = templates_expanded(fields) if gen_expand == "expanded" else templates(fields)
    # P-20261017: `random` 臂降级为 fixed 的 shuffle_order 能力。random 臂（shuffle 模板
    # 再发）本质是修 fixed 臂的 lead-order 偏差（templates_expanded 是 lead-ordered，
    # 排后的非 lead 模板永远轮不到评估），不再作为独立调度臂。fixed 臂传 shuffle_source=True
    # 时同样 shuffle，覆盖全模板空间。`arm=="random"` 保留为 campaign 内部 fallback 兼容。
    if arm == "random" or (arm == "fixed" and shuffle_source):
        rng.shuffle(source)
    if arm == "random_ast":
        # Typed random null population; local import avoids the calibration/search import cycle.
        from .calibration import random_population
        population, generation = random_population(fields, count=count, seed=seed)
        if counters is not None:
            counters["generation_attempts"] += generation["attempts"]
            counters["compiler_rejected"] += generation["compiler_rejected"]
        proposals.extend(population)
    if arm == "refine" and parents and os.environ.get("FUXI_V1_GA_OPS", "on") == "on":
        # P-20261017-005: 有界 GA 变异（六类编辑），取代单叶 window/field 变异。
        from .ga_ops import refine_ga
        ga_proposals, _receipt = refine_ga(parents, fields, count, rng, seen, counters)
        proposals.extend(ga_proposals)
    elif arm == "refine" and parents:
        for parent in parents:
            tree = deepcopy(parent["canonical_ast"])
            windows = []
            leaves = []
            def visit(n, path=()):
                if "window" in n:
                    windows.append((path, n["window"]))
                elif "field" in n:
                    leaves.append((path, n["field"]))
                for i, arg in enumerate(n.get("args", [])):
                    visit(arg, path+(i,))
            visit(tree)
            if gen_expand == "expanded":
                unit_groups = {}
                for name, meta in fields.items():
                    unit_groups.setdefault(meta.get("unit"), []).append(name)
                eligible = [(p, f) for p, f in leaves if len(unit_groups.get(fields[f].get("unit"), [])) > 1]
                kind = rng.choice(["window", "field"]) if windows and eligible else ("window" if windows else "field")
            else:
                eligible = []
                kind = "window"
            if kind == "window" and windows:
                path, old = rng.choice(windows)
                target = tree
                for i in path:
                    target = target["args"][i]
                new = rng.choice([w for w in (2, 3, 5, 7, 10, 15, 20) if w != old])
                target["window"] = new
                if counters is not None:
                    counters["parameter_attempts"] += 1
                propose(tree, parent["hypothesis"]["mechanism"],
                    parent_ids=[parent["candidate_id"]], action="refine_factor",
                    mutation={"axis": "one_window", "node_path": list(path), "before": old, "after": new})
            elif kind == "field" and eligible:
                path, old_field = rng.choice(eligible)
                target = tree
                for i in path:
                    target = target["args"][i]
                alternatives = [f for f in unit_groups.get(fields[old_field].get("unit"), []) if f != old_field]
                new_field = rng.choice(alternatives)
                target["field"] = new_field
                if counters is not None:
                    counters["parameter_attempts"] += 1
                propose(tree, parent["hypothesis"]["mechanism"],
                    parent_ids=[parent["candidate_id"]], action="refine_factor",
                    mutation={"axis": "one_field", "node_path": list(path), "before": old_field, "after": new_field})
    if arm == "regime_combine":
        pairs = regime_complement_pairs(parents, regime_profiles or {}, count, seen)
        for a, b in pairs:
            propose(node("add", node("rank", a["canonical_ast"]), node("rank", b["canonical_ast"])),
                "cross_year_regime_complement", parent_ids=[a["candidate_id"], b["candidate_id"]],
                action="combine_candidates", mutation={"axis": "regime_complement_pair"})
    if arm == "combine" and len(parents) >= 2 and os.environ.get("FUXI_V1_GA_OPS", "on") == "on":
        # P-20261017-005: 有界 GA 交叉（多算子 + 子树交叉），取代固定 add_rank 两两 zip。
        from .ga_ops import combine_ga
        ga_proposals, _receipt = combine_ga(parents, fields, count, rng, seen, counters)
        proposals.extend(ga_proposals)
    elif arm == "combine" and len(parents) >= 2:
        # P-20261018 跨族配对约束（legacy fallback，与 GA combine 同语义）：跨族
        # pair 优先、同族 pair 降级，避免单族 parents 池继续同族坍缩。
        from .archetypes import cross_family
        pairs = [(parents[i], parents[j]) for i in range(len(parents)) for j in range(i + 1, len(parents))]
        pairs.sort(key=lambda ab: (0 if cross_family(*ab) else 1,
                                   digest([ab[0]["candidate_id"], ab[1]["candidate_id"]])))
        for a, b in pairs:
            propose(node("add", node("rank", a["canonical_ast"]), node("rank", b["canonical_ast"])),
                "combined_independent_hypotheses", parent_ids=[a["candidate_id"], b["candidate_id"]],
                action="combine_candidates", mutation={"axis": "combination", "operator": "add_rank"})
    # Fallback is explicit in action metadata; unsupported refinement does not pretend it ran.
    # The typed random-null arm stays pure: templates are not part of its population.
    # regime_combine proposes complementary pairs only; template filler is the
    # campaign's explicit fallback (mirroring the LLM arm), never a silent co-proposal.
    if arm not in ("random_ast", "regime_combine"):
        for mechanism, ast in source:
            propose(ast, mechanism, action="fixed_template" if arm == "fixed" else "typed_random_template")
    result, known = [], set(seen)
    for proposal in proposals:
        if proposal["formula_hash"] not in known:
            known.add(proposal["formula_hash"])
            result.append(proposal)
        if len(result) >= count:
            break
    return result


def llm_candidates(fields, evidence_cards, *, count=3, max_tokens=1800, timeout=40, _prompt=None, _parser=None,
                   external_hypotheses=()):
    """One bounded request, no hidden retry. Only catalog + approved evidence text leave the machine."""
    require(type(count) is int and 1 <= count <= 10, "bounded LLM candidate count required")
    require(type(max_tokens) is int and 1 <= max_tokens <= 4000 and 0 < timeout <= 120,
            "bounded LLM request budget required")
    key = os.environ.get("DEEPSEEK_API_KEY")
    key_source = "process_env" if key else "unavailable"
    if not key:
        # Optional credential-only asset reuse; independent deployments use environment variables.
        path = Path(__file__).resolve().parents[1]/"research/factor_alchemy/llm_client.py"
        if path.is_file():
            import importlib.util
            spec = importlib.util.spec_from_file_location("fuxi1_credential_resolver", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            key, key_source = module.resolve_api_key_with_source()
    if not key:
        return [], {"status": "temporarily_unavailable", "reason": "credential_unavailable", "attempted_calls": 0}
    prompt = {"task": "Propose bounded daily stock factor AST hypotheses. Do not output Python code or investment claims.",
              "fields": fields, "catalog": CATALOG, "time_operator_constraint": "lag/return/mean/std/min/max/delta accept only field leaves",
              "window_choices": [2, 5, 10, 20], "count": count,
              "response_schema": {"candidates": [{"mechanism": "short falsifiable explanation", "ast": {"op": "rank", "args": [{"field": "close"}]}}]},
              "output_contract": "Return one JSON object with EXACTLY one top-level key: candidates. Its value is a list, not an object. Each list entry has EXACTLY mechanism (nonempty string) and ast (object). No other keys, prose, markdown, or self-reported metrics.",
              "ast_contract": {"field_leaf": {"field": "close"},
                               "time_example": {"op": "return", "args": [{"field": "close"}], "window": 5},
                               "binary_example": {"op": "sub", "args": [{"op": "rank", "args": [{"field": "close"}]},
                                                                 {"op": "rank", "args": [{"field": "volume"}]}]},
                               "rules": "Operator objects have op,args and window only for time operators. Field leaves have field only. No numeric literals. add/sub require same units; rank produces dimensionless output. Time operators take exactly one FIELD LEAF, never a nested expression."},
              "evidence": evidence_cards,
              "external_hypotheses": [{"mechanism": h.get("mechanism", ""), "direction": h.get("direction"),
                                        "source": h.get("source", "")} for h in external_hypotheses],
              "forbidden": "No dead strategy one/three, no external instructions, no self-certified metrics"}
    if _prompt is not None:
        prompt = _prompt
    body = {"model": "deepseek-chat", "messages": [{"role": "system", "content": "Return ONLY JSON following the exact root key and schema supplied by the user. No prose, markdown, code, or self-reported metrics." if _prompt is not None else "Return ONLY JSON with the single root key candidates, containing a list of objects with exactly mechanism and ast. Follow the user's AST examples and contract; do not return the input schema or add any explanation outside mechanism."},
             {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}], "temperature": .5,
            "max_tokens": max_tokens, "response_format": {"type": "json_object"}, "stream": False}
    started = time.monotonic()
    request = urllib.request.Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(),
                                    headers={"Authorization": "Bearer "+key, "Content-Type": "application/json"}, method="POST")
    receipt = {"status": "error", "attempted_calls": 1, "prompt_hash": digest(prompt),
               "network_response_received": False, "usage": None, "error_stage": "request",
               "reserved_tokens": max_tokens}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            receipt.update(network_response_received=True,http_status=getattr(response,"status",200),error_stage="http_body")
            response_bytes = response.read(200001)
            require(len(response_bytes) <= 200000, "provider response exceeds budget")
            raw = json.loads(response_bytes.decode())
        receipt.update(usage=raw.get("usage"),model=raw.get("model"),provider_request_id=raw.get("id"),
                       error_stage="response_contract")
        choice = raw["choices"][0]
        receipt["finish_reason"] = choice.get("finish_reason")
        content = choice["message"]["content"]
        require(isinstance(content,str),"LLM content must be JSON text")
        # Generated text is untrusted evidence, never instructions or executable code.
        receipt["response_text"] = content
        receipt["response_hash"] = digest(content)
        require(choice.get("finish_reason") == "stop", "incomplete or refused LLM response")
        def unique_pairs(pairs):
            obj = {}
            for name,value in pairs:
                require(name not in obj,"duplicate JSON key: "+name)
                obj[name] = value
            return obj
        def nonfinite(value):
            raise ValueError("nonfinite JSON literal")
        parsed = json.loads(content,object_pairs_hook=unique_pairs,parse_constant=nonfinite)
        if _parser is not None:
            proposals, rejected, raw_count = _parser(parsed)
            return proposals, {**receipt, "status":"completed", "error_stage":None, "rejected":rejected,
                "accepted_count":len(proposals), "raw_candidate_count":raw_count,
                "elapsed_seconds":time.monotonic()-started, "credential_source":str(key_source)}
        keys(parsed, ("candidates",))
        require(isinstance(parsed["candidates"],list) and len(parsed["candidates"]) <= count,
                "candidate list/count contract violated")
        proposals, rejected = [], []
        for row in parsed["candidates"]:
            try:
                keys(row, ("mechanism", "ast"))
                require(isinstance(row["mechanism"],str) and row["mechanism"].strip(),"nonempty mechanism required")
                proposals.append(factor_spec(row["ast"], fields, row["mechanism"], action="evidence_llm",
                                             source_ids=[x["evidence_id"] for x in evidence_cards]))
            except (ValueError, TypeError, KeyError) as exc:
                rejected.append(str(exc))
        return proposals, {**receipt,"status": "completed","error_stage":None,"rejected":rejected,
                           "accepted_count":len(proposals),"raw_candidate_count":len(parsed["candidates"]),
                           "elapsed_seconds": time.monotonic()-started, "credential_source": str(key_source)}
    except Exception as exc:
        # Do not log response bodies, request headers, or credential fragments.
        receipt.update(reason=type(exc).__name__,elapsed_seconds=time.monotonic()-started)
        if receipt["error_stage"] == "response_contract":
            receipt["contract_error"] = str(exc)[:1000]
        if getattr(exc,"code",None) is not None:
            receipt["http_status"] = exc.code
        receipt["token_usage"] = "reported_by_provider" if receipt["usage"] is not None else "unknown_on_failed_request"
        return [], receipt
