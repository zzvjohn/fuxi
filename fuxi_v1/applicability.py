"""因子适用性备注（factor applicability note）：纯诊断，不做筛选门槛。

定位与边界（用户拍板）：
  * 稳健性检验（市场状态/行业/板块适用性）**不作为筛选门槛**；
  * 若存在相应适用性信号，作为**因子备注**沉淀到候选元数据；
  * 多因子合成阶段据此施加条件化限制（本模块只产出备注 schema，不实现合成）。

输出 `fuxi.factor_note.1`：
  · industry：按申万一级行业分组的分组 rank IC（weak_industries 列出 IC≤0 的行业）；
  · regime：按市场状态标签分组的分组 rank IC（需外部提供 {session: regime}，缺省跳过）；
  · 均带 `purpose: note_only_not_gate`，明确不参与 pass/reject。

行业分类 = 最新申万一级静态映射（非 PIT），与 exposure_monitor 同一口径；
本模块沿用该口径并如实披露，不假装有 PIT 行业历史。
"""
from .contracts import finite, require, timestamp
from .exposure_monitor import normalize_code, CAP_BUCKETS

__all__ = ["conditioned_rank_ic", "applicability_notes", "lookup_industry",
           "cap_bucket"]


def cap_bucket(cap):
    """流通市值(元) -> '大盘'/'中盘'/'小盘'（与 exposure_monitor 同口径，亿元阈值）。"""
    cap_yi = float(cap) / 1e8
    for name, lo in CAP_BUCKETS:
        if cap_yi >= lo:
            return name
    return None


def _lookup(code, mapping):
    candidates = [normalize_code(code)]
    if "." in candidates[0]:
        candidates.append(candidates[0].split(".")[0])
    for c in candidates:
        if c in mapping:
            return mapping[c]
    return None


def lookup_industry(code, industry_map):
    return _lookup(code, industry_map)


def load_shenwan_industry_map(path):
    """读取申万一级映射 CSV（code,industry）为 {normalized_code: industry}。

    文件缺失时返回 None（调用方据此刻画 ``industry_map_unavailable``），不做
    隐式失败——行业分类是静态元数据，不是 hash 绑定的研究输入。
    """
    import csv
    from pathlib import Path
    path = Path(path)
    if not path.is_file():
        return None
    mapping = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            code = (row.get("code") or "").strip()
            ind = (row.get("industry") or "").strip()
            if code and ind:
                mapping[normalize_code(code)] = ind
    return mapping or None


def default_shenwan_industry_map():
    """Best-effort 加载项目标准 sw_industry_map.csv（相对包目录），带进程级缓存。

    缺失返回 None（调用方据此标注 unavailable）。行业分类是静态参考，非 hash
    绑定的研究输入，故允许在评估阶段做一次 best-effort 读取而不破坏数据契约。
    """
    import functools
    from pathlib import Path

    @functools.lru_cache(maxsize=1)
    def _load():
        root = Path(__file__).resolve().parents[1]
        return load_shenwan_industry_map(root / "data" / "sw_industry_map.csv")

    return _load()


def _rank_ic(pairs):
    """对 (score, label) 样本对算截面 rank IC；<3 对返回 None。"""
    from .dsl import ranks
    from .evaluation import correlation
    n = len(pairs)
    if n < 3:
        return None
    rs = ranks({i: pairs[i][0] for i in range(n)})
    ry = ranks({i: pairs[i][1] for i in range(n)})
    return correlation([rs[i] for i in range(n)], [ry[i] for i in range(n)])


def conditioned_rank_ic(scores, snapshot, labels, clock, start, end, group_key):
    """按 group_key(day, code) 分组，逐 session 算组内截面 rank IC，跨期聚合。

    group_key 返回 None 表示该样本不参与分组。用于行业（按 code）与市场状态（按 day）。
    返回 {key: {"mean_rank_ic", "n_sections", "n_pairs"}}。
    """
    clock = timestamp(clock) if isinstance(clock, str) else clock
    acc = {}
    if snapshot.payload.get("schema_version") == "fuxi.columnar_snapshot.1":
        import numpy as np
        from .columnar import score_grid_view
        grid = score_grid_view(scores)
        for day in grid:
            if not (start <= day <= end):
                continue
            i = grid.day_indices[day]
            x = np.asarray(grid.matrix[i])
            y = labels.values_at(i, clock)
            valid = grid.membership[i] & np.isfinite(x) & np.isfinite(y)
            buckets = {}
            for j in np.flatnonzero(valid):
                key = group_key(day, grid.instruments[j])
                if key is None:
                    continue
                buckets.setdefault(key, []).append((float(x[j]), float(y[j])))
            for key, pairs in buckets.items():
                ic = _rank_ic(pairs)
                if ic is None:
                    continue
                a = acc.setdefault(key, {"ic_sum": 0.0, "n_sections": 0, "n_pairs": 0})
                a["ic_sum"] += ic
                a["n_sections"] += 1
                a["n_pairs"] += len(pairs)
    else:
        for day in sorted(scores):
            if not (start <= day <= end):
                continue
            buckets = {}
            for s in snapshot.universe[day]:
                sv = scores[day].get(s)
                lv = labels.value(day, s, clock)
                if sv is None or lv is None or not finite(sv) or not finite(lv):
                    continue
                key = group_key(day, s)
                if key is None:
                    continue
                buckets.setdefault(key, []).append((sv, lv))
            for key, pairs in buckets.items():
                ic = _rank_ic(pairs)
                if ic is None:
                    continue
                a = acc.setdefault(key, {"ic_sum": 0.0, "n_sections": 0, "n_pairs": 0})
                a["ic_sum"] += ic
                a["n_sections"] += 1
                a["n_pairs"] += len(pairs)
    return {key: {"mean_rank_ic": a["ic_sum"] / a["n_sections"],
                  "n_sections": a["n_sections"], "n_pairs": a["n_pairs"]}
            for key, a in acc.items()}


def applicability_notes(scores, snapshot, labels, clock, start, end, *,
                        industry_map=None, regime_by_session=None, cap_by_session=None,
                        min_sections=5):
    """产出因子适用性备注（note-only，绝不参与 pass/reject）。"""
    note = {"schema_version": "fuxi.factor_note.1", "purpose": "note_only_not_gate",
            "min_sections": min_sections}
    if industry_map:
        by_ind = conditioned_rank_ic(scores, snapshot, labels, clock, start, end,
                                     lambda day, code: _lookup(code, industry_map))
        weak = sorted(k for k, r in by_ind.items()
                      if r["n_sections"] >= min_sections and r["mean_rank_ic"] <= 0)
        note["industry"] = {"source": "latest_shenwan_l1_static_not_pit",
                            "by_industry": by_ind, "weak_industries": weak}
    else:
        note["industry"] = {"status": "industry_map_unavailable"}
    if regime_by_session:
        by_regime = conditioned_rank_ic(scores, snapshot, labels, clock, start, end,
                                        lambda day, code: regime_by_session.get(day))
        weak_regimes = sorted(k for k, r in by_regime.items()
                              if r["n_sections"] >= min_sections and r["mean_rank_ic"] <= 0)
        note["regime"] = {"by_regime": by_regime, "weak_regimes": weak_regimes}
    else:
        note["regime"] = {"status": "regime_labels_unavailable"}
    if cap_by_session:
        by_cap = conditioned_rank_ic(scores, snapshot, labels, clock, start, end,
                                     lambda day, code: _cap_lookup(cap_by_session, day, code))
        weak_cap = sorted(k for k, r in by_cap.items()
                          if r["n_sections"] >= min_sections and r["mean_rank_ic"] <= 0)
        note["market_cap"] = {"source": "tushare_daily_basic_circ_mv_contemporaneous",
                              "by_cap": by_cap, "weak_cap": weak_cap}
    else:
        note["market_cap"] = {"status": "cap_labels_unavailable"}
    return note


def _cap_lookup(cap_by_session, day, code):
    """{trade_date: {code: cap}} 里查 (day, code) 的市值桶；缺失返回 None。"""
    day_caps = cap_by_session.get(day)
    if not day_caps:
        return None
    cap = day_caps.get(code)
    if cap is None:
        cap = _lookup(code, day_caps)
    return cap_bucket(cap) if cap is not None else None
