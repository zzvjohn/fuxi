"""E1 单调性诊断（note-only）：分位组合收益 + Patton-Timmermann (2010) MR test。

定位与边界（用户拍板 2026-10-20）：
  * 单调性**不再作为 pass/reject 门槛**。A 股日频/周频单因子层的分组收益 Spearman
    真实量级 ≈0.05~0.1（文献 Rank IC 0.02~0.08），"是否单调"的二元判断信息量极低；
    Patton & Timmermann (2010, JFE) 自己报告单调性检验在低信噪比下功效仅 11-13%。
    硬卡（如 spearman≥0.8）只会误杀，方法论上不成立。
  * 单调性降级为 **note-only 诊断**，与 applicability 的 regime/market_cap/industry
    并列进入 factor_notes，供多因子合成阶段做"相对比较"（IC 相近时选单调性更强者），
    而非"绝对准入"。
  * 形状信息改用 Patton-Timmermann (2010) MR test 口径：相邻分组收益差
    Δ_i = G_{i+1}-G_i 的 block bootstrap 单边检验，**只报 p 值不设硬门槛**。

输出 fuxi.monotonicity.1：
  * spread 显著性：top-bottom spread 的 block bootstrap 下界；spread 不显著时
    status=data_insufficient（fail-open，不翻转 E1 已有裁决）；
  * mr_adjacent_deltas：PT MR test 相邻差 Δ_i 的单边 p 值（只报不裁）；
  * 分组收益序列 / 相邻单调比例 / 分组收益 Spearman：纯描述。

方向约定：按正向（高分→高收益）检验，与 E1 screen 的 rank_ic>0 方向一致，
故不做方向翻转。
"""
import math
from statistics import mean

from .contracts import require, finite, timestamp
from .evaluation import correlation
from .dsl import ranks

MONOTONICITY_DEFAULTS = {
    "n_groups": 5,             # 五分位（Q1..Q5）
    "min_sessions": 2,         # 有效 session 数下限（不足则 data_insufficient）
    "min_cross_section": 15,   # 单 session 有效样本下限（5 组 × ≥3 只）
    "block_length": 12,        # spread 分块 bootstrap 块长（默认；调用方用 request 覆盖）
    "alpha": 0.05,
    "replicates": 3999,
    "seed": 20261019,
}


def _spearman_group_order(group_returns):
    """分组序 [0..n-1] 与分组收益之间的 Spearman 秩相关。"""
    n = len(group_returns)
    order = [j for j in range(n)]
    rank_of_return = ranks({j: group_returns[j] for j in range(n)})
    return correlation(order, [rank_of_return[j] for j in range(n)])


def _group_stats(xs, ys, n_groups):
    """按 score 升序把截面等量切成 n_groups 组，算各组收益与单调性统计量。

    xs/ys 为同序的 score/label 有限样本对（已剔除 None/NaN）。组边界按
    score 排序等量切分（组大小差 ≤1）；组间平局跨界时切分顺序确定性（稳定排序）。
    """
    n = len(xs)
    require(n >= n_groups, "grouping requires at least one stock per group")
    order = sorted(range(n), key=lambda k: xs[k])  # 稳定排序
    sizes = [n // n_groups + (1 if j < n % n_groups else 0) for j in range(n_groups)]
    groups, pos = [], 0
    for size in sizes:
        seg = order[pos:pos + size]
        pos += size
        groups.append(mean(ys[k] for k in seg))
    spread = groups[-1] - groups[0]
    spearman = _spearman_group_order(groups)
    adjacent = sum(1 for j in range(n_groups - 1) if groups[j + 1] > groups[j]) / (n_groups - 1)
    return {"group_returns": groups, "spread": spread,
            "monotonic_spearman": spearman, "adjacent_monotonic_fraction": adjacent}


def _legacy_days(scores, snapshot, labels, clock, start, end, min_cross_section):
    days = []
    for day in sorted(scores):
        if not start <= day <= end:
            continue
        members = snapshot.universe[day]
        xs, ys = [], []
        for s in members:
            sv = scores[day].get(s)
            lv = labels.value(day, s, clock)
            if sv is not None and lv is not None and finite(sv) and finite(lv):
                xs.append(sv)
                ys.append(lv)
        if len(xs) >= min_cross_section:
            days.append((day, xs, ys, len(xs)))
    return days


def _columnar_days(scores, snapshot, labels, clock, start, end, min_cross_section):
    import numpy as np
    from .columnar import score_grid_view
    grid = score_grid_view(scores)
    days = []
    for day in grid:
        if not start <= day <= end:
            continue
        i = grid.day_indices[day]
        x = np.asarray(grid.matrix[i])
        y = labels.values_at(i, clock)
        valid = grid.membership[i] & np.isfinite(x) & np.isfinite(y)
        n = int(valid.sum())
        if n < min_cross_section:
            continue
        days.append((day, x[valid].tolist(), y[valid].tolist(), n))
    return days


def _mr_adjacent_deltas(sections, n_groups, block_length, alpha, replicates, seed):
    """Patton-Timmermann (2010) MR test：相邻分组收益差 Δ_i = G_{i+1}-G_i 的逐差
    block bootstrap 单边检验。

    只报每个 Δ_i 的均值/下界/单边 p 值，不设硬门槛（用户拍板"只报 p 值不裁"）。
    joint_p_bonferroni_upper_bound = min(p_i)·(n_groups-1) 是联合"弱单调递增"
    （所有 Δ_i>0）的保守 Bonferroni 上界，仅作参考。
    """
    from .research_protocol import block_inference_fast
    deltas = []
    for i in range(n_groups - 1):
        d = [s["group_returns"][i + 1] - s["group_returns"][i] for s in sections]
        inf = block_inference_fast(d, block_length=block_length, alpha=alpha,
                                   replicates=replicates, seed=seed + 1 + i)
        deltas.append({
            "delta_index": i, "from_group": i + 1, "to_group": i + 2,
            "mean": inf["mean"], "lower": inf["lower"],
            "p_one_sided": inf["p_one_sided"] if inf["status"] == "valid" else None,
            "inference_status": inf["status"],
        })
    ps = [x["p_one_sided"] for x in deltas if x["p_one_sided"] is not None]
    joint_bonferroni = min(1.0, min(ps) * (n_groups - 1)) if ps else None
    all_positive = all(x["mean"] is not None and x["mean"] > 0 for x in deltas)
    return {"method": "patton_timmermann_2010_mr_adjacent_portfolios",
            "n_adjacent_deltas": n_groups - 1, "deltas": deltas,
            "joint_p_bonferroni_upper_bound": joint_bonferroni,
            "all_adjacent_deltas_positive": all_positive}


def monotonicity_diagnostics(scores, snapshot, labels, clock, start, end, *,
                             n_groups=MONOTONICITY_DEFAULTS["n_groups"],
                             min_cross_section=MONOTONICITY_DEFAULTS["min_cross_section"],
                             block_length=MONOTONICITY_DEFAULTS["block_length"],
                             alpha=MONOTONICITY_DEFAULTS["alpha"],
                             replicates=MONOTONICITY_DEFAULTS["replicates"],
                             seed=MONOTONICITY_DEFAULTS["seed"]):
    """逐 session 分位组合收益 + 跨期聚合，产出单调性诊断（不做裁决）。"""
    require(n_groups >= 2 and min_cross_section >= n_groups, "invalid monotonicity bucketing")
    clock = timestamp(clock) if isinstance(clock, str) else clock
    if snapshot.payload.get("schema_version") == "fuxi.columnar_snapshot.1":
        days = _columnar_days(scores, snapshot, labels, clock, start, end, min_cross_section)
    else:
        days = _legacy_days(scores, snapshot, labels, clock, start, end, min_cross_section)

    sections, spreads, spears, adjacents = [], [], [], []
    group_acc = [0.0] * n_groups
    for day, xs, ys, n in days:
        st = _group_stats(xs, ys, n_groups)
        sections.append({"session": day, "n": n, **st})
        spreads.append(st["spread"])
        if st["monotonic_spearman"] is not None:
            spears.append(st["monotonic_spearman"])
        adjacents.append(st["adjacent_monotonic_fraction"])
        for j in range(n_groups):
            group_acc[j] += st["group_returns"][j]

    n_sessions = len(sections)
    time_avg = [group_acc[j] / n_sessions for j in range(n_groups)] if n_sessions else []
    spread_inference = None
    if spreads:
        from .research_protocol import block_inference_fast
        spread_inference = block_inference_fast(spreads, block_length=block_length,
                                                alpha=alpha, replicates=replicates, seed=seed)
    mr = _mr_adjacent_deltas(sections, n_groups, block_length, alpha, replicates, seed) \
        if sections else None
    return {
        "schema_version": "fuxi.monotonicity.1",
        "n_groups": n_groups,
        "n_sessions": n_sessions,
        "min_cross_section": min_cross_section,
        "sections": sections,
        "time_averaged_group_returns": time_avg,
        "spread_inference": spread_inference,
        "mr_adjacent_deltas": mr,
        "mean_monotonic_spearman": mean(spears) if spears else None,
        "mean_adjacent_monotonic_fraction": mean(adjacents) if adjacents else None,
    }


def monotonicity_note(diag, *, min_sessions=MONOTONICITY_DEFAULTS["min_sessions"]):
    """把单调性诊断包装为 note-only 结论（绝不参与 pass/reject）。

    status 语义（与 applicability 的 note 并列，非 gate）：
      * data_insufficient：样本不足 / spread 不显著 —— 形状信息不足，fail-open；
      * diagnostic：spread 显著 —— 产出 MR test 相邻差诊断（只报 p 值，不裁决）。
    """
    note = {"schema_version": "fuxi.monotonicity.1", "purpose": "note_only_not_gate",
            "min_sessions": min_sessions, "spread_significant": False}
    if diag["n_sessions"] < min_sessions:
        note.update(status="data_insufficient",
                    reason_code="monotonicity_insufficient_sessions")
    elif diag["spread_inference"] is None or diag["spread_inference"]["status"] != "valid":
        note.update(status="data_insufficient",
                    reason_code="monotonicity_spread_inference_inadequate")
    elif diag["spread_inference"]["lower"] <= 0:
        note.update(status="data_insufficient",
                    reason_code="monotonicity_spread_not_significant")
    else:
        note.update(status="diagnostic", spread_significant=True,
                    reason_code="monotonicity_shape_diagnostic_only")
        note["spread_inference"] = diag["spread_inference"]
        note["mr_adjacent_deltas"] = diag.get("mr_adjacent_deltas")
        note["time_averaged_group_returns"] = diag["time_averaged_group_returns"]
        note["mean_monotonic_spearman"] = diag["mean_monotonic_spearman"]
        note["mean_adjacent_monotonic_fraction"] = diag["mean_adjacent_monotonic_fraction"]
    note["diagnostics"] = diag
    return note
