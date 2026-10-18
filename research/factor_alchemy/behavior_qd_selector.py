"""M4 (2026-09-27, P1 第六项): 行为空间质量-多样性 (QD) 选择器 —— 隔离挑战引擎。

任务（方案 §M4）: 把 MAP-Elites 式「行为格保留最优」引入候选选择层，
在**同一冻结矩阵、同一预算**下比较「行为格覆盖优先」(QD) vs「按适应度取优」
(legacy)，回答"QD 选择是否用同等产出换来更多独立行为簇"。**先在隔离挑战
脚本内实现，不动生产默认**；同预算不优于 legacy ⇒ 停止 QD 投入。

预注册协议（先于运行写定）:
  · 数据: 复用 M2 冻结矩阵 reports/fuxi_generator_full_batch_20260926/full_batch.json
    （5 种子 × 4 臂 × 12–13 条；panel_research_exposed=true, economic_status=not_evaluated）。
  · 池: 每种子内全部臂候选池化（52 条量级），种子间独立（配对设计）。
  · 行为描述符（离线结构代理 —— 完整 JL 32 维指纹版须在线批, 见 LIMITATIONS）:
      dim1 = root_op 族（ts_* / unary / binary / other）
      dim2 = 复杂度档（<=1.0 low / <=3.5 mid / else high；阈值为 09-26 全矩阵 p25/p75）
    格 = (op_family, cx_bucket)。
  · legacy  : 按 (valid, fitness, expression) 降序取 Top-K。
  · qd_fill : 每格保留最优个体（格代表）→ 质量降序 → 不足 K 时用非代表填充
              （严格同预算 K, 主对比臂）。
  · qd_pure : 只取格代表（不填充; n_selected 可能 < K, 诊断臂 —— 检验
              "少而多样"的簇效率）。
  · 预算 K = round(frac × 池大小), frac ∈ {0.25, 0.5, 0.75, 1.0}。
  · 指标: clusters（骨架 AST Jaccard 距离 >= 0.74 判独立; 仅计 valid 且表达式
    非空的候选 — 与 M2 §闭环同口径）/ valid / positive / n_selected。
  · 判定: primary_frac=0.5, 主对比 (legacy, qd_fill); QD 占优 ⟺
    (median(Δclusters) > 0) 且 (Δclusters>0 种子数 >= ceil(0.8×n_seeds))
    且 (median(Δvalid) >= 0); 否则 stop_rule → 停止 QD 投入。

边界:
  · 纯离线后验选择重排：选在「评估完成后」发生（无前视; 不改变评估预算）。
  · 不模拟生成动力学（MAP-Elites 的真正形态是生成循环内从格代表出发的变异,
    本脚本不声称该形态的收益 —— 见 LIMITATIONS）。
  · 不写任何生产状态; 不宣称经济结论; 全部尝试与拒绝留档（pool 快照 + 选择集）。
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from generator_axis_replay import (                          # noqa: F401
    FITNESS_INVALID_SENTINEL, INDEP_MIN_DIST, _ClusterIndex, DEFAULT_ARMS,
)

SCHEMA = "fuxi_qd_selector_challenge/v1"
POOL_SCHEMA = "fuxi_qd_pool/v1"

BUDGET_FRACS: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
PRIMARY_FRAC = 0.5
POS_SEED_MIN_FRAC = 0.8
PRIMARY_PAIR: Tuple[str, str] = ("legacy", "qd_fill")

# 描述符常量（09-26 全矩阵实测: complexity p25=1.0 p75=3.5, 0 缺失）
CX_LOW_MAX = 1.0
CX_MID_MAX = 3.5
TS_PREFIX = "ts_"
UNARY_OPS = frozenset({"inv", "square", "sqrt", "abs", "neg", "scale", "zscore"})
BINARY_OPS = frozenset({"mul", "div", "add", "sub"})

QD_DECISION_RULE = (
    "预注册判定 (M4, 2026-09-27): 在 primary_frac 预算下, 主对比 (legacy, qd_fill) 上, "
    "仅当 (median(delta_clusters) > 0) 且 (delta_clusters > 0 的种子数 >= "
    "ceil(0.8*n_seeds)) 且 (median(delta_valid) >= 0) 时判 QD 选择占优; "
    "否则按停止规则停止 QD 投入 (不以'量子/新名称'续命)。对比为固定池内后验"
    "选择重排, 不模拟生成动力学, 不宣称在线优越性。"
)

LIMITATIONS = [
    "行为描述符为离线结构代理 (root_op 族 × 复杂度档), 非完整 JL 32 维行为指纹; "
    "在线指纹版的选择效果须后续批自证。",
    "本脚本是池内后验选择重排 (选择发生在评估完成后), 不模拟 MAP-Elites 的生成"
    "循环内形态 (从格代表出发的变异); 不得据此宣称生成动力学收益。",
    "冻结矩阵为面板研究已暴露批次 (panel_research_exposed=true), 结论限定在"
    "'选择层机制'层面, 不构成经济证据 (economic_status=not_evaluated)。",
    "同语法随机臂/同解码器 EDA 臂与 forge/quantum 共享产出序列, 池化后 QD 的"
    "收益部分来自跨臂去重, 属机制读数; 单臂版须另行复核。",
    "预算语义 = 下游槽位数 K (输出条数); 不改变生成/评估预算。",
]


# ══════════════════════════════════════════════════════════════════
# 1. 池加载与描述符
# ══════════════════════════════════════════════════════════════════

def op_family(root_op: Any) -> str:
    """root_op → 族 (ts / unary / binary / other / unknown)。"""
    op = str(root_op or "").strip().lower()
    if not op:
        return "unknown"
    if op.startswith(TS_PREFIX):
        return "ts"
    if op in UNARY_OPS:
        return "unary"
    if op in BINARY_OPS:
        return "binary"
    return "other"


def cx_bucket(complexity: Any) -> str:
    """复杂度 → 档 (low / mid / high / unknown); 阈值为 09-26 全矩阵 p25/p75。"""
    try:
        v = float(complexity)
    except (TypeError, ValueError):
        return "unknown"
    if v <= CX_LOW_MAX:
        return "low"
    if v <= CX_MID_MAX:
        return "mid"
    return "high"


def descriptor(c: Dict) -> Tuple[str, str]:
    return (op_family(c.get("root_op")), cx_bucket(c.get("complexity")))


def _cell_of(c: Dict) -> Dict:
    """候选 → 池条目（含 valid/positive 哨兵语义, 与 M2 同口径）。"""
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
    return {"arm": str(c.get("_arm") or ""),
            "expression": str(c.get("expression") or ""),
            "fitness": fv, "valid": valid, "positive": positive,
            "root_op": str(c.get("root_op") or ""),
            "complexity": c.get("complexity"),
            "cell": list(descriptor(c))}


def load_qd_pool(path: Any) -> Dict:
    """加载冻结矩阵 → 每种子池（全部臂候选合并, 种子间独立）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    seeds: List[Dict] = []
    for r in rows:
        pool: List[Dict] = []
        for arm in DEFAULT_ARMS:
            bucket = r.get(arm) or {}
            for c in bucket.get("candidates") or []:
                cc = dict(c)
                cc["_arm"] = arm
                pool.append(_cell_of(cc))
        seeds.append({"seed": r.get("seed"), "pool": pool,
                      "unique_budget": r.get("unique_budget")})
    return {"schema": POOL_SCHEMA, "seeds": seeds, "arms": list(DEFAULT_ARMS)}


# ══════════════════════════════════════════════════════════════════
# 2. 选择器（确定性; 全部只用评估后信息 —— 无前视）
# ══════════════════════════════════════════════════════════════════

def _sort_key(c: Dict) -> Tuple[int, float, str]:
    """确定性排序键（升序）: (valid, fitness, expression)。"""
    fv = c.get("fitness")
    f = float(fv) if isinstance(fv, (int, float)) and math.isfinite(float(fv)) else float("-inf")
    return (1 if c.get("valid") else 0, f, str(c.get("expression") or ""))


def select_legacy(pool: List[Dict], k: int) -> List[Dict]:
    """legacy: 按 (valid, fitness) 降序 Top-K。"""
    return sorted(pool, key=_sort_key, reverse=True)[:max(0, int(k))]


def build_grid(pool: List[Dict]) -> Dict[Tuple[str, str], int]:
    """MAP-Elites 网格: 每格保留最优个体的池内下标。"""
    best: Dict[Tuple[str, str], int] = {}
    for i, c in enumerate(pool):
        g = descriptor(c)
        if g not in best or _sort_key(c) > _sort_key(pool[best[g]]):
            best[g] = i
    return best


def select_qd_pure(pool: List[Dict], k: int) -> List[Dict]:
    """qd_pure: 只取格代表（质量降序截断到 K; n_selected 可 < K）。"""
    reps = [pool[i] for i in build_grid(pool).values()]
    reps.sort(key=_sort_key, reverse=True)
    return reps[:max(0, int(k))]


def select_qd_fill(pool: List[Dict], k: int) -> List[Dict]:
    """qd_fill: 格代表优先, 不足 K 时用非代表填充（严格同预算）。"""
    k = max(0, int(k))
    best = build_grid(pool)
    rep_idx = set(best.values())
    reps = [pool[i] for i in best.values()]
    reps.sort(key=_sort_key, reverse=True)
    if len(reps) >= k:
        return reps[:k]
    rest = [c for i, c in enumerate(pool) if i not in rep_idx]
    rest.sort(key=_sort_key, reverse=True)
    return reps + rest[: k - len(reps)]


SELECTORS: Dict[str, Callable[[List[Dict], int], List[Dict]]] = {
    "legacy": select_legacy,
    "qd_fill": select_qd_fill,
    "qd_pure": select_qd_pure,
}


# ══════════════════════════════════════════════════════════════════
# 3. 评估与判定
# ══════════════════════════════════════════════════════════════════

def _eval_selected(selected: List[Dict], cluster_min_dist: float) -> Dict:
    """选定集合读数。聚类前按表达式字典序做**规范排序** —— 簇数只依赖集合
    本身, 不依赖选择器输出顺序（否则「代表先到」的 qd_fill 会因贪心聚类
    顺序差异被污染比较; 测试 test_frac1_converges_same_set 捕获过该缺陷）。"""
    ci = _ClusterIndex(cluster_min_dist)
    n_valid = n_pos = 0
    for c in sorted(selected, key=lambda x: str(x.get("expression") or "")):
        if c.get("valid"):
            n_valid += 1
            if c.get("positive"):
                n_pos += 1
            ci.add(c.get("expression") or "")
    return {"n_selected": len(selected), "valid": n_valid,
            "positive": n_pos, "clusters": ci.count,
            "n_empty_expr": ci.n_empty}


def _summ(xs: List[float]) -> Dict:
    xs = [float(x) for x in xs]
    n = len(xs)
    pos = sum(1 for x in xs if x > 0)
    neg = sum(1 for x in xs if x < 0)
    return {"per_seed": xs,
            "median": (statistics.median(xs) if xs else None),
            "mean": ((sum(xs) / n) if n else None),
            "pos_seeds": pos, "neg_seeds": neg, "zero_seeds": n - pos - neg}


def evaluate_qd_matrix(pool_matrix: Dict, *,
                       budget_fracs: Tuple[float, ...] = BUDGET_FRACS,
                       primary_frac: float = PRIMARY_FRAC,
                       primary_pair: Tuple[str, str] = PRIMARY_PAIR,
                       cluster_min_dist: float = INDEP_MIN_DIST,
                       keep_selected: bool = True) -> Dict:
    """在冻结池上跑全部 (种子 × 预算档 × 选择器), 并按预注册规则出判定。"""
    per_seed: List[Dict] = []
    for row in pool_matrix.get("seeds") or []:
        pool = row.get("pool") or []
        by_frac: Dict[str, Dict] = {}
        for frac in budget_fracs:
            k = max(1, int(round(frac * len(pool)))) if pool else 0
            slot: Dict[str, Any] = {"frac": frac, "K": k, "pool_size": len(pool),
                                    "selectors": {}}
            for name, fn in SELECTORS.items():
                sel = fn(pool, k)
                r = _eval_selected(sel, cluster_min_dist)
                if keep_selected:
                    r["selected"] = [{"arm": c.get("arm"),
                                      "expression": c.get("expression"),
                                      "cell": c.get("cell")} for c in sel]
                slot["selectors"][name] = r
            by_frac[str(frac)] = slot
        per_seed.append({"seed": row.get("seed"), "pool_size": len(pool),
                         "by_frac": by_frac})

    lo, hi = primary_pair
    aggregate: Dict[str, Dict] = {}
    for frac in budget_fracs:
        kkey = str(frac)
        dv = [ps["by_frac"][kkey]["selectors"][hi]["valid"]
              - ps["by_frac"][kkey]["selectors"][lo]["valid"] for ps in per_seed]
        dc = [ps["by_frac"][kkey]["selectors"][hi]["clusters"]
              - ps["by_frac"][kkey]["selectors"][lo]["clusters"] for ps in per_seed]
        dn = [ps["by_frac"][kkey]["selectors"][hi]["n_selected"]
              - ps["by_frac"][kkey]["selectors"][lo]["n_selected"] for ps in per_seed]
        aggregate[kkey] = {"delta_valid": _summ(dv), "delta_clusters": _summ(dc),
                           "delta_n_selected": _summ(dn)}

    n_seeds = len(per_seed)
    agg_p = aggregate.get(str(primary_frac), {})
    need = int(math.ceil(POS_SEED_MIN_FRAC * max(1, n_seeds)))
    adv = bool(agg_p and agg_p["delta_clusters"]["median"] is not None
               and agg_p["delta_clusters"]["median"] > 0
               and agg_p["delta_clusters"]["pos_seeds"] >= need
               and agg_p["delta_valid"]["median"] is not None
               and agg_p["delta_valid"]["median"] >= 0)
    verdict = {
        "primary_frac": primary_frac,
        "primary_pair": list(primary_pair),
        "n_seeds": n_seeds,
        "pos_seeds_required": need,
        "qd_advantage": adv,
        "action": ("qd_selector_adopt_candidate" if adv else "stop_qd_investment"),
        "stop_rule_triggered": (not adv),
        "decision_rule": QD_DECISION_RULE,
    }
    return {"schema": SCHEMA, "per_seed": per_seed, "aggregate": aggregate,
            "verdict": verdict, "limitations": list(LIMITATIONS),
            "constants": {"indep_min_dist": cluster_min_dist,
                          "budget_fracs": list(budget_fracs),
                          "primary_frac": primary_frac,
                          "cx_low_max": CX_LOW_MAX, "cx_mid_max": CX_MID_MAX,
                          "descriptor": "root_op_family x complexity_bucket"}}


def run_challenge(matrix_path: Any) -> Dict:
    pool_matrix = load_qd_pool(matrix_path)
    grid_cover: List[Dict] = []
    for row in pool_matrix["seeds"]:
        pool = row["pool"]
        cells = {tuple(c["cell"]) for c in pool}
        grid_cover.append({"seed": row["seed"], "pool_size": len(pool),
                           "n_cells": len(cells),
                           "cells": sorted([list(c) for c in cells])})
    out = evaluate_qd_matrix(pool_matrix)
    out["grid_coverage"] = grid_cover
    return out


def _cli(argv: Optional[List[str]] = None) -> int:
    """python behavior_qd_selector.py <full_batch.json> [out.json]"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python behavior_qd_selector.py <full_batch.json> [out.json]")
        return 2
    rep = run_challenge(argv[0])
    if len(argv) > 1:
        Path(argv[1]).write_text(
            json.dumps(rep, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
        print("written:", argv[1])
    v = rep["verdict"]
    print("verdict:", v["action"], "| qd_advantage:", v["qd_advantage"],
          "| stop_rule:", v["stop_rule_triggered"])
    for frac, agg in rep["aggregate"].items():
        print("  frac=%s dvalid median=%s (pos %d/%d) dclusters median=%s (pos %d/%d)"
              % (frac, agg["delta_valid"]["median"], agg["delta_valid"]["pos_seeds"],
                 v["n_seeds"], agg["delta_clusters"]["median"],
                 agg["delta_clusters"]["pos_seeds"], v["n_seeds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
