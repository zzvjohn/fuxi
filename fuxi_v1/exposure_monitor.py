"""Portfolio beta-exposure monitor (research-side, read-only diagnostics).

Answers one question: where does the strategy's *money* sit, by industry and
(optionally) free-float market cap, and how has that drifted vs the previous
rebalance.

Scope / non-goals (deliberate):
  * Reads target weights only. Does NOT neutralize, orthogonalize, or alter any
    factor or DSL. Exposure is reported, never removed.
  * Industry classification = latest Shenwan L1 map (static snapshot, NOT
    point-in-time). Historical PIT industry is a known project gap; this module
    discloses that instead of pretending to have it.
  * Market-cap exposure is optional. The snapshot payload carries no cap field,
    so it needs an external {code: free_float_cap} map (e.g. tushare daily_basic
    circ_mv/total_mv). Without it, cap columns are emitted as None.

The same decomposition is reused verbatim on the JQ side (actual holdings) by
feeding actual weights instead of research-side target weights.
"""

from .contracts import digest, require, finite
from .portfolio import target_weights

__all__ = ["normalize_code", "industry_exposure", "market_cap_exposure",
           "exposure_snapshot", "period_exposures", "render_report"]

# Free-float market-cap buckets (CNY). Configurable; defaults are common A-share
# thresholds: small < 10bn, mid 10bn-30bn, large > 30bn.
CAP_BUCKETS = (("大盘", 300.0), ("中盘", 100.0), ("小盘", 0.0))  # 单位：亿元


def normalize_code(code):
    """Best-effort normalize to ``000019.SZ`` form (bare/suffixed/exchange-prefixed)."""
    s = str(code).strip().upper()
    if "." in s:
        return s
    if s.startswith(("SH", "SZ", "BJ")) and len(s) > 2:
        return s[2:] + "." + s[:2]
    return s


def _lookup(code, mapping):
    """Look a code up against a mapping keyed by several candidate spellings."""
    candidates = [normalize_code(code)]
    if "." in candidates[0]:
        candidates.append(candidates[0].split(".")[0])
    for c in candidates:
        if c in mapping:
            return mapping[c]
    return None


def industry_exposure(weights, industry_map):
    """Aggregate {code: weight} -> {industry: weight}. Unclassified bucketed."""
    out = {}
    unclassified = 0.0
    for code, w in weights.items():
        ind = _lookup(code, industry_map)
        if ind is None:
            unclassified += w
            continue
        out[ind] = out.get(ind, 0.0) + w
    if unclassified > 1e-12:
        out["未分类"] = unclassified
    return out


def market_cap_exposure(weights, cap_map):
    """Aggregate {code: weight} -> {bucket: weight} by free-float cap."""
    out = {name: 0.0 for name, _ in CAP_BUCKETS}
    unknown = 0.0
    for code, w in weights.items():
        cap = _lookup(code, cap_map) if cap_map else None
        if cap is None:
            unknown += w
            continue
        cap_yi = float(cap) / 1e8  # 元 -> 亿元
        for name, lo in CAP_BUCKETS:
            if cap_yi >= lo:
                out[name] += w
                break
    if unknown > 1e-12:
        out["未知市值"] = unknown
    return {k: v for k, v in out.items() if v > 1e-12}


def exposure_snapshot(weights, industry_map, cap_map=None):
    """One-period exposure decomposition for a set of target weights."""
    require(isinstance(weights, dict), "weights dict required")
    total = sum(weights.values())
    require(finite(total) and total > 0, "non-positive total weight")
    ind = industry_exposure(weights, industry_map)
    cap = market_cap_exposure(weights, cap_map) if cap_map else None
    return {
        "schema_version": "fuxi.exposure.1",
        "n_holdings": len(weights),
        "gross_weight": total,
        "industry": ind,
        "industry_hash": digest({k: round(v, 10) for k, v in ind.items()}),
        "market_cap": cap,
        "market_cap_available": cap_map is not None,
    }


def _drift(cur, prev):
    """Symmetric set difference of two name->weight maps."""
    if prev is None:
        return None
    keys = set(cur) | set(prev)
    return {k: round(cur.get(k, 0.0) - prev.get(k, 0.0), 10) for k in sorted(keys)}


def _resolve_cap_map(cap_map, session):
    """Return the {code: cap} map for one session (no look-ahead).

    ``cap_map`` may be single-layer {code: cap} (static, returned as-is) or
    two-layer {trade_date: {code: cap}} from ``cap_data.load_cap_snapshots``,
    matched against the session date (both 'YYYY-MM-DD' and 'YYYYMMDD' keys).
    """
    if not cap_map:
        return None
    top_vals = list(cap_map.values())[:3]
    if top_vals and all(isinstance(v, dict) for v in top_vals):
        session_key = session.replace("-", "")[:8]
        for key in (session, session_key):
            if key in cap_map:
                return cap_map[key]
        return None
    return cap_map


def period_exposures(ir, industry_map, cap_map=None):
    """Per-rebalance exposure + drift across a full strategy IR.

    Rebuilds target weights from each decision's scores via the shared
    ``target_weights`` so the exposure reflects the signal's intent (not the
    executor's turnover-limited prior). The same function accepts raw
    {session: weights} dicts for the JQ-side actual-holdings path.
    """
    require(ir.get("schema_version") == "fuxi.strategy_ir.1", "strategy IR required")
    spec = ir["strategy"]
    snapshots = []
    prev_ind = None
    prev_cap = None
    for decision in ir["decisions"]:
        weights = target_weights(decision["scores"], spec)
        if weights is None:
            snapshots.append({"signal_session": decision["signal_session"],
                              "skipped": "no_valid_scores"})
            continue
        session_cap = _resolve_cap_map(cap_map, decision["signal_session"])
        snap = exposure_snapshot(weights, industry_map, session_cap)
        snap["signal_session"] = decision["signal_session"]
        snap["industry_drift"] = _drift(snap["industry"], prev_ind)
        if session_cap is not None:
            snap["market_cap_drift"] = _drift(snap["market_cap"], prev_cap)
            prev_cap = snap["market_cap"]
        snapshots.append(snap)
        prev_ind = snap["industry"]
    return {
        "schema_version": "fuxi.exposure_report.1",
        "ir_hash": ir.get("ir_hash"),
        "strategy": spec,
        "industry_source": "latest_shenwan_l1_static_not_pit",
        "market_cap_available": cap_map is not None,
        "market_cap_source": "tushare_daily_basic_contemporaneous" if cap_map else None,
        "periods": snapshots,
    }


def render_report(report):
    """Markdown rendering of a period-exposure report."""
    lines = ["# 组合 beta 暴露监控", ""]
    s = report["strategy"]
    lines.append(f"- 候选: `{s.get('candidate_id', '?')}` · top_n={s['top_n']} · "
                 f"调仓每 {s['rebalance_sessions']} session · 行业口径=申万一级(静态快照,非PIT)")
    cap_note = "已提供(tushare 流通市值,当日)" if report['market_cap_available'] else "未提供(缺市值数据,已跳过)"
    lines.append(f"- 市值暴露: {cap_note}")
    lines.append("")
    has_cap = report.get("market_cap_available")
    if has_cap:
        lines.append("| 调仓 session | 持仓数 | 行业暴露 Top3 | 市值暴露 | 行业漂移 Top3 |")
        lines.append("|---|---|---|---|---|")
    else:
        lines.append("| 调仓 session | 持仓数 | 行业暴露 Top3 | 较上期漂移 Top3 |")
        lines.append("|---|---|---|---|")
    for p in report["periods"]:
        if "skipped" in p:
            lines.append(f"| {p['signal_session']} | — | (无有效分数) | — |" +
                         (" — |" if has_cap else ""))
            continue
        ind = p["industry"]
        top3 = "、".join(f"{k}{v:.0%}" for k, v in sorted(ind.items(), key=lambda x: -x[1])[:3])
        drift = p.get("industry_drift")
        d3 = "、".join(f"{k}{v:+.0%}" for k, v in sorted(drift.items(), key=lambda x: -abs(x[1]))[:3]) if drift else "—"
        if has_cap:
            cap = p.get("market_cap") or {}
            cap3 = "、".join(f"{k}{v:.0%}" for k, v in sorted(cap.items(), key=lambda x: -x[1])[:3]) or "—"
            lines.append(f"| {p['signal_session']} | {p['n_holdings']} | {top3} | {cap3} | {d3} |")
        else:
            lines.append(f"| {p['signal_session']} | {p['n_holdings']} | {top3} | {d3} |")
    lines.append("")
    lines.append("> 说明：暴露=按目标权重加权求和，只报告、不剥离。行业为最新申万一级静态映射，非历史 PIT；"
                 "市值口径=流通市值(circ_mv,当日快照)。")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys, json
    from .contracts import load_json
    if len(sys.argv) < 3:
        print("usage: python -m fuxi_v1.exposure_monitor <ir.json> <industry_map.csv> [cap_map.json]")
        sys.exit(2)
    ir = load_json(sys.argv[1])
    industry_map = {}
    import csv
    with open(sys.argv[2], encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for k in ("code", "ts_code"):
                if k in row and row[k]:
                    industry_map[row[k].strip()] = row.get("industry") or row.get("sw_l1") or ""
                    break
    cap_map = None
    if len(sys.argv) > 3:
        cap_map = load_json(sys.argv[3])
    report = period_exposures(ir, industry_map, cap_map)
    print(render_report(report))
