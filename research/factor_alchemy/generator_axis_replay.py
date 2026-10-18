"""M2 (2026-09-27, P-20260927-005): 生成器轴离线回放引擎（固定集合内预算分配检验）。

任务（方案 §M2）: 用**冻结**历史轮次数据，比较「按已成熟事件选择生成器」
(mature) vs「均匀分层调度」(uniform) 在**同一总预算**下的产出，回答
"选对引擎是否带来更多有效候选"。无证据前不消费（读端保持 shadow）。

回放协议（预注册，先于运行写定）:
  · 数据源: reports/fuxi_generator_full_batch_20260926/full_batch.json —
    5 种子 × 4 臂 (forge/quantum/same_grammar_random/same_decoder_eda),
    每臂每种子 12–13 条求值, unique_budget 以 Forge 实际求值配对（等预算）;
    protocol 注明 panel_research_exposed=true, economic_status=not_evaluated。
  · 槽位 = 一次候选求值（对某一臂的消费）。预算 B = round(frac × 总记录数)。
  · uniform: 按臂序轮转 (round-robin, 分层调度基线)。
  · mature: 前置 warmup（每臂 W 槽, 臂序轮转）后, 仅依据**已消费**观测的
    valid 率选臂（并列时 valid 计数高者 → 臂序靠前者）; 无 ε 探索（确定性,
    与生产 ε=0.2 的差异记录在报告）。
  · 指标: valid = 可裁决候选数 (fitness > -999 哨兵); clusters = 独立行为
    簇数（骨架 AST Jaccard 距离 ≥ 0.74 判独立, 与 §M4 同阈值; 仅计 valid
    且表达式非空的候选; 同骨架参数变体并入同簇）。
  · 因果性（"未选动作不猜收益"）: 策略只见已消费前缀; 未被消费的槽位永不
    推断/补值; 每臂消费上限 = 该臂记录条数（冻结边界, 超额由确定性回落处理）。
  · 判定: DECISION_RULE（无差异 → 保留均匀调度, 不回写任何消费端开关）。

边界:
  · 本模块纯离线, 不写任何生产状态; 不触碰 MAB / 候选字典 / 裁决字段;
  · 回放假设"臂产出序列与分配策略独立"(标准离线重放假设), 该假设写入
    LIMITATIONS, 报告不得越界宣称在线优越性。
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from paradigm_proposal import ast_distance

SCHEMA = "fuxi_generator_axis_replay/v1"

FITNESS_INVALID_SENTINEL = -999.0   # fitness <= -999 ⇒ 无效 (09-26 批次哨兵语义)
INDEP_MIN_DIST = 0.74               # 行为独立阈值 (§M4 同口径: 骨架距离 >= 0.74)
WARMUP_PER_ARM = 3                  # 与 FUXI_GENERATOR_AXIS_MIN_PULLS 默认一致
BUDGET_FRACS: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
PRIMARY_FRAC = 0.5
POS_SEED_MIN_FRAC = 0.8
DEFAULT_ARMS = ("forge", "quantum", "same_grammar_random", "same_decoder_eda")

DECISION_RULE = (
    "预注册判定 (M2, 2026-09-27): 在 primary_frac 预算下, 仅当 "
    "(median(delta_valid) > 0) 且 (delta_valid > 0 的种子数 >= ceil(0.8*n_seeds)) "
    "且 (median(delta_clusters) >= 0) 时判 mature 分配占优; 否则按停止规则"
    "保留均匀调度。回放为固定集合内预算分配检验, 未选动作不猜收益, "
    "不宣称在线优越性。"
)

LIMITATIONS = (
    "1) 回放仅在冻结等预算矩阵上做「固定集合内预算分配检验」; 结果不构成"
    "在线优越性证据 (方案 §M2 风险条: 历史/冻结日志含选择偏差)。",
    "2) 冻结序列对「臂被少跑」的反事实不可得: 回放假设臂产出序列与分配策略"
    "独立 (标准离线重放假设); 未选动作不猜收益, 不在未观测槽位上做任何推断。",
    "3) 面板为研究已暴露样本 (protocol: panel_research_exposed=true), "
    "fitness=legacy 口径, economic_status=not_evaluated ⇒ 不产生经济结论。",
    "4) 4 臂 = 09-26 全量工程挑战 {forge, quantum, same_grammar_random, "
    "same_decoder_eda}; 与生产 MAB 四引擎 {gp_breed, llm, forge, gp_evolve} "
    "不逐一对应 (仅 forge 同名); 结论限定在「分配机制」层面。",
    "5) clusters 口径 = 骨架 AST Jaccard ≥ 0.74 判独立 (忽略参数取值); "
    "与 JL 指纹/风格暴露族未做交叉验证。",
)

__all__ = [
    "SCHEMA", "INDEP_MIN_DIST", "WARMUP_PER_ARM", "BUDGET_FRACS",
    "PRIMARY_FRAC", "DEFAULT_ARMS", "DECISION_RULE", "LIMITATIONS",
    "load_full_batch", "validate_matrix", "make_uniform_policy",
    "make_mature_policy", "simulate", "evaluate_matrix",
]


# ══════════════════════════════════════════════════════════════════
# 1. 冻结矩阵加载与校验
# ══════════════════════════════════════════════════════════════════

def _obs_from_candidate(c: Dict) -> Dict:
    """单条候选 → 回放观测。valid 语义 = fitness > -999（哨兵，09-26 批次口径）。"""
    fv: Optional[float] = None
    valid = False
    positive = False
    try:
        fv = float(c.get("fitness"))
        if math.isfinite(fv) and fv > FITNESS_INVALID_SENTINEL:
            valid = True
            positive = fv > 0.0
    except (TypeError, ValueError):
        fv = None
    return {"fitness": fv, "valid": valid, "positive": positive,
            "expression": str(c.get("expression") or "")}


def load_full_batch(path: Any) -> Dict:
    """加载 full_batch.json → {"seeds": [{"seed", "arms": {arm: [obs]}}], "arms": [...]}"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    present = set()
    for r in rows:
        present.update(k for k in r.keys()
                       if k not in ("seed", "unique_budget"))
    arms = [a for a in DEFAULT_ARMS if a in present]
    arms += sorted(present - set(arms))
    seeds: List[Dict] = []
    for r in rows:
        arm_map: Dict[str, List[Dict]] = {}
        for a in arms:
            bucket = r.get(a) or {}
            cands = bucket.get("candidates") or []
            arm_map[a] = [_obs_from_candidate(c) for c in cands]
        seeds.append({"seed": r.get("seed"), "arms": arm_map,
                      "unique_budget": r.get("unique_budget")})
    return {"seeds": seeds, "arms": arms, "protocol_sha_field": data.get("sha256")}


def validate_matrix(matrix: Dict) -> Dict:
    """结构校验: 非空 / 每种子内各臂长度配对（等预算前提）。"""
    issues: List[Dict] = []
    seeds = matrix.get("seeds") or []
    arms = matrix.get("arms") or []
    if not seeds:
        issues.append({"kind": "empty_matrix"})
    for row in seeds:
        lens = {a: len((row.get("arms") or {}).get(a) or []) for a in arms}
        nonzero = {v for v in lens.values() if v > 0}
        if len(nonzero) > 1 or (nonzero and 0 in lens.values()):
            issues.append({"seed": row.get("seed"), "lens": lens,
                           "kind": "unpaired_arm_lengths"})
    return {"ok": not issues, "issues": issues}


# ══════════════════════════════════════════════════════════════════
# 2. 策略（确定性; state 仅含已消费前缀 —— 结构性保证无前视）
# ══════════════════════════════════════════════════════════════════

def make_uniform_policy(arm_order) -> Callable[[Dict], str]:
    """均匀分层调度: 按臂序轮转（当前步起始臂 = t % n, 跳过已耗尽臂）。"""
    order = tuple(arm_order)

    def policy(state: Dict) -> str:
        n = len(order)
        t = int(state.get("t", 0))
        for k in range(n):
            a = order[(t + k) % n]
            if state["remaining"].get(a, 0) > 0:
                return a
        return order[0] if order else ""

    policy.__name__ = "uniform"
    return policy


def make_mature_policy(arm_order, warmup: int = WARMUP_PER_ARM) -> Callable[[Dict], str]:
    """按已成熟事件选择: warmup 后按「已消费观测的 valid 率」贪心选臂。

    warmup: 每臂先获得 W 槽（臂序轮转）。
    exploit: 候选臂（remaining>0 且已有消费）按 (valid 率, valid 计数, 臂序)
             取最大; 全无合格臂时回落首个仍有配额的臂。
    """
    order = tuple(arm_order)
    W = max(1, int(warmup))

    def policy(state: Dict) -> str:
        remaining = state["remaining"]
        consumed = state["consumed"]
        # 阶段 1: warmup —— 每臂先获得 W 槽（消费最少者优先, 臂序破平 ⇒ 交织轮转）
        cands = [a for a in order
                 if remaining.get(a, 0) > 0 and len(consumed.get(a) or []) < W]
        if cands:
            return min(cands, key=lambda a: (len(consumed.get(a) or []),
                                             order.index(a)))
        best, best_key = None, None  # 阶段 2: exploit
        for idx, a in enumerate(order):
            if remaining.get(a, 0) <= 0:
                continue
            obs = consumed.get(a) or []
            if not obs:
                continue
            v = sum(1 for o in obs if o.get("valid"))
            key = (v / len(obs), v, -idx)
            if best_key is None or key > best_key:
                best, best_key = a, key
        if best is None:
            for a in order:
                if remaining.get(a, 0) > 0:
                    return a
        return best

    policy.__name__ = f"mature_w{W}"
    return policy


# ══════════════════════════════════════════════════════════════════
# 3. 回放执行
# ══════════════════════════════════════════════════════════════════

class _ClusterIndex:
    """贪心增量聚类: 与任何已存代表距离 < min_dist 并入, 否则新开簇。"""

    def __init__(self, min_dist: float = INDEP_MIN_DIST):
        self.min_dist = float(min_dist)
        self.reps: List[str] = []
        self.n_empty = 0

    def add(self, expr: str) -> bool:
        s = str(expr or "").strip()
        if not s:
            self.n_empty += 1
            return False
        for rep in self.reps:
            if ast_distance(s, rep, skeleton=True) < self.min_dist:
                return False
        self.reps.append(s)
        return True

    @property
    def count(self) -> int:
        return len(self.reps)


def simulate(arms_obs: Dict[str, List[Dict]], policy: Callable[[Dict], str],
             budget: int, *, cluster_min_dist: float = INDEP_MIN_DIST) -> Dict:
    """确定性顺序回放: 每步策略选臂 → 消费该臂下一条记录 → 更新读数。

    state 只包含 {"t", "remaining", "consumed"}; consumed 为已消费前缀的副本
    ⇒ 策略在结构上无法看到未消费数据（"未选动作不猜收益"）。
    """
    budget = int(budget)
    arms = list(arms_obs.keys())
    remaining = {a: len(arms_obs[a]) for a in arms}
    consumed: Dict[str, List[Dict]] = {a: [] for a in arms}
    ptr = {a: 0 for a in arms}
    ci = _ClusterIndex(cluster_min_dist)
    n_valid = 0
    n_pos = 0
    trace: List[Tuple[int, str]] = []
    t = 0
    while t < budget:
        state = {"t": t,
                 "remaining": dict(remaining),
                 "consumed": {a: list(consumed[a]) for a in arms}}
        arm = policy(state)
        if arm not in remaining or remaining[arm] <= 0:  # 确定性回落（防御）
            eligible = [a for a in arms if remaining[a] > 0]
            if not eligible:
                break
            arm = eligible[0]
        obs = arms_obs[arm][ptr[arm]]
        ptr[arm] += 1
        remaining[arm] -= 1
        consumed[arm].append(obs)
        if obs.get("valid"):
            n_valid += 1
            if obs.get("positive"):
                n_pos += 1
            ci.add(obs.get("expression") or "")
        trace.append((t, arm))
        t += 1
    return {"budget": budget, "consumed_total": t, "valid": n_valid,
            "positive": n_pos, "clusters": ci.count,
            "n_empty_expr": ci.n_empty,
            "per_arm_slots": {a: len(consumed[a]) for a in arms},
            "trace": [[int(x), y] for x, y in trace]}


def _summ(xs: List[float]) -> Dict:
    xs = [float(x) for x in xs]
    n = len(xs)
    pos = sum(1 for x in xs if x > 0)
    neg = sum(1 for x in xs if x < 0)
    return {"per_seed": xs,
            "median": (statistics.median(xs) if xs else None),
            "mean": ((sum(xs) / n) if n else None),
            "pos_seeds": pos, "neg_seeds": neg, "zero_seeds": n - pos - neg}


def evaluate_matrix(matrix: Dict, *, policies: Dict[str, Callable[[Dict], str]],
                    primary_pair: Tuple[str, str] = ("uniform", "mature"),
                    budget_fracs: Tuple[float, ...] = BUDGET_FRACS,
                    primary_frac: float = PRIMARY_FRAC,
                    cluster_min_dist: float = INDEP_MIN_DIST) -> Dict:
    """在冻结矩阵上跑全部 (种子 × 预算档 × 策略), 并按预注册规则出判定。"""
    per_seed: List[Dict] = []
    for row in matrix.get("seeds") or []:
        arms_obs = row["arms"]
        total = sum(len(v) for v in arms_obs.values())
        by_frac: Dict[str, Dict] = {}
        for frac in budget_fracs:
            budget = max(1, int(round(frac * total)))
            slot: Dict[str, Any] = {"frac": frac, "budget": budget,
                                    "total_recorded": total, "policies": {}}
            for name, policy in policies.items():
                slot["policies"][name] = simulate(
                    arms_obs, policy, budget, cluster_min_dist=cluster_min_dist)
            by_frac[str(frac)] = slot
        per_seed.append({"seed": row.get("seed"), "total_recorded": total,
                         "by_frac": by_frac})

    lo, hi = primary_pair
    aggregate: Dict[str, Dict] = {}
    for frac in budget_fracs:
        k = str(frac)
        dv = [ps["by_frac"][k]["policies"][hi]["valid"]
              - ps["by_frac"][k]["policies"][lo]["valid"] for ps in per_seed]
        dc = [ps["by_frac"][k]["policies"][hi]["clusters"]
              - ps["by_frac"][k]["policies"][lo]["clusters"] for ps in per_seed]
        aggregate[k] = {"delta_valid": _summ(dv), "delta_clusters": _summ(dc)}

    n_seeds = len(per_seed)
    agg_p = aggregate.get(str(primary_frac), {})
    need = int(math.ceil(POS_SEED_MIN_FRAC * max(1, n_seeds)))
    adv = bool(agg_p and agg_p["delta_valid"]["median"] is not None
               and agg_p["delta_valid"]["median"] > 0
               and agg_p["delta_valid"]["pos_seeds"] >= need
               and agg_p["delta_clusters"]["median"] is not None
               and agg_p["delta_clusters"]["median"] >= 0)
    verdict = {
        "primary_frac": primary_frac,
        "primary_pair": list(primary_pair),
        "n_seeds": n_seeds,
        "pos_seeds_required": need,
        "mature_advantage": adv,
        "action": ("mature_allocation" if adv else "keep_uniform"),
        "stop_rule_triggered": (not adv),
        "decision_rule": DECISION_RULE,
    }
    return {"schema": SCHEMA, "per_seed": per_seed, "aggregate": aggregate,
            "verdict": verdict, "limitations": list(LIMITATIONS),
            "constants": {"indep_min_dist": cluster_min_dist,
                          "budget_fracs": list(budget_fracs),
                          "primary_frac": primary_frac}}


def _cli(argv: Optional[List[str]] = None) -> int:
    """最小 CLI: python generator_axis_replay.py <full_batch.json> [out.json]"""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python generator_axis_replay.py <full_batch.json> [out.json]")
        return 2
    matrix = load_full_batch(argv[0])
    val = validate_matrix(matrix)
    arms = matrix["arms"]
    policies = {"uniform": make_uniform_policy(arms),
                "mature": make_mature_policy(arms)}
    res = evaluate_matrix(matrix, policies=policies)
    res["matrix_validation"] = val
    if len(argv) > 1:
        Path(argv[1]).write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"written: {argv[1]}")
    v = res["verdict"]
    print(f"verdict: action={v['action']} mature_advantage={v['mature_advantage']} "
          f"stop_rule_triggered={v['stop_rule_triggered']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
