"""Bounded GA operators for the refine/combine arms (FUXI_V1_GA_OPS=on).

FUXI_V1_GA_OPS=on 时，``search.generate`` 的 refine/combine 分支改走本模块：

- ``refine_ga``: 有界多叶变异，复用 ``edits.apply_patch`` 的六类编辑
  (window / field / operator / wrap / unwrap / binary)，取代旧的「只改 1 个
  window 或 field 叶子」。
- ``combine_ga``: 多算子组合 (add/sub/mul/div 等权 rank) + 单点子树交叉，
  取代旧的「固定 add(rank(a), rank(b)) 两两 zip」。

硬约束不松：max_depth / max_nodes / unit 由 ``edits.apply_patch`` 与
``dsl.compile_ast`` 双层校验，非法 AST 记为 compiler_rejected 并跳过；
formula_hash 去重沿用 campaign 传入的 seen 集合。

机制标注：refine 继承父机制（局部微调语义不变）；combine 与子树交叉均标
``combined_independent_hypotheses``（组合独立假设），维持 archetype 归因不污染。

延迟 import 打破 search<->edits<->ga_ops 的模块级循环：本模块只做函数体内的
``from .search import ...`` / ``from .edits import ...``。
"""
import os

from .contracts import ContractError, digest


def ga_ops_enabled():
    return os.environ.get("FUXI_V1_GA_OPS", "on") == "on"


# ---------------------------------------------------------------- 预算常量
# P-20261017: 代数/种群预算从量化因子研究的统计约束反推，不拍脑袋。
# E1 用 alpha/max_trials 的 bootstrap 区间下界 > 0（research_protocol.screen），
# 故 max_trials 越大门槛越高：t_hurdle = z(1 - 0.05/(2*max_trials))，
# 最低可检测 ICIR = t_hurdle / sqrt(n_sections)，研究窗 1125 截面 → sqrt=33.54。
#
#   max_trials= 12 -> t=2.87, ICIR=0.085  （当前；门槛偏松，易把噪声当因子）
#   max_trials= 60 -> t=3.34, ICIR=0.100  （保守下限）
#   max_trials= 80 -> t=3.42, ICIR=0.102  （性价比拐点：门槛仅 +19%，预算 ×6.7）
#   max_trials=120 -> t=3.53, ICIR=0.105  （预算上限；A 股动量/反转常见 ICIR 0.1~0.3）
#   max_trials=240 -> t=3.71, ICIR=0.111  （不推荐：门槛过严，真实因子被误杀）
#   max_trials=400 -> t=3.84, ICIR=0.114  （明确不推荐）
#
# 采用 max_trials∈[60,120]（rounds×candidates）。campaign 每轮在
# fixed/random_ast/refine/combine 四臂间轮换、GA 占两臂，故实际 GA 代数 ≈ rounds*0.5。
# 推荐配置：rounds=20 / candidates_per_round=4 -> max_trials=80, GA≈10 代, 种群 4。
GA_POPULATION_PER_GENERATION = 4      # 每代新个体数（= candidates_per_round）
GA_MIN_GENERATIONS = 10               # GA 实际迭代代数下限（≈ rounds * 0.5）
GA_TRIALS_BUDGET = (60, 120)          # max_trials 目标区间（t 门槛 3.34~3.53）


# ---------------------------------------------------------------- refine
def _enumerate_refine_edits(parent, fields, rng, per_kind=2):
    """枚举一个 parent 的可行单叶编辑（六类，每类至多 per_kind 个）。

    P-20261018 跨族偏好：返回 (cross, same) 两桶。operator（改语义族）、binary
    （注入 interaction）、跨字段族 field（如 volume→float_shares）判为跨族；
    window（族内参数微调）、wrap/unwrap（neg/abs 翻方向）、同字段族 field 判为
    族内。refine_ga 跨族优先、族内补充，避免 refine 只在同族内打转。
    """
    from .edits import paths, WINDOWS
    from .dsl import CATALOG
    from .archetypes import field_family

    pid = parent["candidate_id"]
    unit_groups = {}
    for name, meta in fields.items():
        unit_groups.setdefault(meta.get("unit"), []).append(name)
    cross, same = [], []
    temporal_ops = {"lag", "return", "mean", "std", "min", "max", "delta"}
    for path, target in paths(parent["canonical_ast"]):
        if "window" in target:
            old = target["window"]
            neighbors = sorted((w for w in WINDOWS if w != old), key=lambda w: abs(w - old))[:per_kind]
            for w in neighbors:
                same.append({"parent_id": pid, "kind": "window", "path": list(path), "value": w})
        elif "field" in target:
            unit = fields[target["field"]].get("unit")
            alternatives = [f for f in unit_groups.get(unit, []) if f != target["field"]]
            for f in rng.sample(alternatives, min(per_kind, len(alternatives))):
                bucket = cross if field_family(f) != field_family(target["field"]) else same
                bucket.append({"parent_id": pid, "kind": "field", "path": list(path), "value": f})
        else:
            # operator edit: same arity, same temporal/non-temporal category
            arity = len(target.get("args", []))
            is_temporal = target["op"] in temporal_ops
            candidates = []
            for op, a in CATALOG["operators"].items():
                if a != arity or op == target["op"]:
                    continue
                if (op in temporal_ops) != is_temporal:
                    continue
                candidates.append(op)
            for op in rng.sample(candidates, min(per_kind, len(candidates))):
                cross.append({"parent_id": pid, "kind": "operator", "path": list(path), "value": op})
        # wrap / unwrap: unit-preserving only (neg/abs) to avoid rank/zscore unit churn
        for w in ("neg", "abs"):
            same.append({"parent_id": pid, "kind": "wrap", "path": list(path), "value": w})
        if target.get("op") in ("neg", "abs") and len(target.get("args", [])) == 1:
            same.append({"parent_id": pid, "kind": "unwrap", "path": list(path), "value": None})
    # binary edits: rank(target) OP rank(field) for each catalog field (interaction injection)
    fields_list = list(fields.keys())
    for op in ("add", "sub", "mul", "div"):
        for f in rng.sample(fields_list, min(per_kind, len(fields_list))):
            cross.append({"parent_id": pid, "kind": "binary", "path": [], "value": {"op": op, "field": f}})
    return cross, same


def refine_ga(parents, fields, count, rng, seen=(), counters=None):
    """有界多叶变异：shuffle 编辑候选后逐个 apply_patch，凑满 count 个合法新因子。

    P-20261018 跨族偏好：跨族编辑（operator/binary/跨字段族 field）优先于族内
    编辑（window/wrap/unwrap）入选，避免 refine 只在同族内微调。
    """
    from .edits import apply_patch

    proposals, known, attempts, rejected = [], set(seen), 0, 0
    cross_pool, same_pool = [], []
    for parent in parents:
        c, s = _enumerate_refine_edits(parent, fields, rng)
        cross_pool.extend(c)
        same_pool.extend(s)
    rng.shuffle(cross_pool)
    rng.shuffle(same_pool)
    edit_pool = cross_pool + same_pool
    lookup = {p["candidate_id"]: p for p in parents}
    for edit in edit_pool:
        parent = lookup.get(edit["parent_id"])
        if parent is None:
            continue
        attempts += 1
        try:
            child = apply_patch(parent, edit, fields)
            child["action"] = "refine_factor"
            child["candidate_id"] = "factor_" + digest({k: v for k, v in child.items() if k != "candidate_id"})
        except ContractError:
            rejected += 1
            continue
        if child["formula_hash"] in known:
            continue
        known.add(child["formula_hash"])
        proposals.append(child)
        if len(proposals) >= count:
            break
    if counters is not None:
        counters["generation_attempts"] += attempts
        counters["compiler_rejected"] += rejected
        counters["parameter_attempts"] += attempts
    return proposals, {"attempts": attempts, "compiler_rejected": rejected}


# ---------------------------------------------------------------- combine
def combine_ga(parents, fields, count, rng, seen=(), counters=None):
    """多算子组合 (add/sub/mul/div 等权 rank) + 单点子树交叉。

    P-20261018 跨族配对约束 + 轴多样性采样：
    - pair 按「是否跨 archetype 族」排序，跨族 pair 优先、同族 pair 降级补充；
    - combination 与 subtree_crossover 两轴轮转采样，避免纯随机让某轴被淹没；
    - seen 与 parent hash 一律去重（含替换子树后等于父代的退化 crossover）。
    候选内容（AST/mutation/parent_ids）不变，故 candidate_id 与可复现性不破坏。
    """
    from copy import deepcopy
    from .edits import paths
    from .search import node, factor_spec
    from .archetypes import cross_family

    attempts, rejected = 0, 0
    known = set(seen) | {p["formula_hash"] for p in parents}
    pairs = [(parents[i], parents[j]) for i in range(len(parents)) for j in range(i + 1, len(parents))]
    # 跨族 pair 优先（确定性：跨族先、同族后，tie-break 用 pair id digest）
    pairs.sort(key=lambda ab: (0 if cross_family(*ab) else 1,
                               digest([ab[0]["candidate_id"], ab[1]["candidate_id"]])))
    comb_cross, comb_same, xover_cross, xover_same = [], [], [], []

    def emit(spec, cross, axis):
        if spec["formula_hash"] in known:
            return
        known.add(spec["formula_hash"])
        if axis == "combination":
            (comb_cross if cross else comb_same).append(spec)
        else:
            (xover_cross if cross else xover_same).append(spec)

    for a, b in pairs:
        cross = cross_family(a, b)
        for op in ("add", "sub", "mul", "div"):
            attempts += 1
            ast = node(op, node("rank", a["canonical_ast"]), node("rank", b["canonical_ast"]))
            try:
                spec = factor_spec(ast, fields, "combined_independent_hypotheses",
                                   parent_ids=[a["candidate_id"], b["candidate_id"]],
                                   action="combine_candidates", mutation={"axis": "combination", "operator": op})
            except ContractError:
                rejected += 1
                continue
            emit(spec, cross, "combination")
        # single-point subtree crossover
        nodes_a = [(p, n) for p, n in paths(a["canonical_ast"]) if p]
        nodes_b = list(paths(b["canonical_ast"]))
        if not nodes_a or not nodes_b:
            continue
        for _ in range(2):
            attempts += 1
            path_a, _ = rng.choice(nodes_a)
            path_b, subtree_b = rng.choice(nodes_b)
            tree = deepcopy(a["canonical_ast"])
            target = tree
            for i in path_a[:-1]:
                target = target["args"][i]
            target["args"][path_a[-1]] = deepcopy(subtree_b)
            if digest(tree) in known:
                continue  # 退化 crossover：替换子树后等于父代/已见，跳过
            try:
                spec = factor_spec(tree, fields, "combined_independent_hypotheses",
                                   parent_ids=[a["candidate_id"], b["candidate_id"]],
                                   action="combine_candidates",
                                   mutation={"axis": "subtree_crossover",
                                             "path_a": list(path_a), "path_b": list(path_b)})
            except ContractError:
                rejected += 1
                continue
            emit(spec, cross, "subtree_crossover")

    # 轴多样性 + 跨族优先轮转采样：跨族组合 → 跨族交叉 → 同族组合 → 同族交叉。
    buckets = [comb_cross, xover_cross, comb_same, xover_same]
    for b in buckets:
        rng.shuffle(b)
    result = []
    while len(result) < count:
        progressed = False
        for b in buckets:
            if b:
                result.append(b.pop(0))
                progressed = True
                if len(result) >= count:
                    break
        if not progressed:
            break
    if counters is not None:
        counters["generation_attempts"] += attempts
        counters["compiler_rejected"] += rejected
    return result, {"attempts": attempts, "compiler_rejected": rejected,
                    "cross_family_pairs": sum(1 for p in pairs if cross_family(*p)),
                    "same_family_pairs": sum(1 for p in pairs if not cross_family(*p))}
