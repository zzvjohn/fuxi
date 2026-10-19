"""Tushare daily_basic market-cap snapshot fetcher (content-addressed cache).

Fetches one whole-market day of daily_basic, keeps the requested cap measure,
normalises to CNY *yuan*, and stores a content-addressed snapshot so the
exposure monitor can consume {code: cap} maps without re-hitting the API.

Cap measures (tushare daily_basic):
  * circ_mv   -- 流通市值 (万元)          -> 元 (circ_mv * 1e4)
  * total_mv  -- 总市值   (万元)          -> 元 (total_mv * 1e4)
  * free_float-- 自由流通市值 (元)        = free_share(万股)*1e4*close(元/股)

The snapshot is keyed by trade_date so the exposure monitor can look up each
rebalance session's *contemporaneous* cap (no look-ahead).
"""

import json
import time
from pathlib import Path

from .contracts import digest, require

CAP_FIELDS = {
    "circ_mv": {"name": "流通市值", "src_unit": "万元"},
    "total_mv": {"name": "总市值", "src_unit": "万元"},
    "free_float": {"name": "自由流通市值", "src_unit": "元(计算)"},
}


def _normalize_date(d):
    """tushare trade_date 有 'YYYYMMDD' 与 'YYYY-MM-DD' 两种，统一为 'YYYY-MM-DD'。"""
    s = str(d).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def to_tushare_code(code):
    """快照代码 '000001.XSHE'/'600000.XSHG' -> tushare '000001.SZ'/'600000.SH'。"""
    s = str(code).strip().upper()
    if s.endswith(".XSHE"):
        return s[:-5] + ".SZ"
    if s.endswith(".XSHG"):
        return s[:-5] + ".SH"
    return s


def from_tushare_code(code):
    """tushare '000001.SZ'/'600000.SH' -> 快照代码 '000001.XSHE'/'600000.XSHG'。"""
    s = str(code).strip().upper()
    if s.endswith(".SZ"):
        return s[:-3] + ".XSHE"
    if s.endswith(".SH"):
        return s[:-3] + ".XSHG"
    return s


def _extract_cap(r, cap_field):
    """从 daily_basic 单行提取市值（元）；失败返回 None。"""
    try:
        if cap_field == "total_mv":
            return float(r["total_mv"]) * 1e4
        if cap_field == "free_float":
            free_share, close = r.get("free_share"), r.get("close")
            if free_share is None or close is None or free_share != free_share or close != close:
                return None
            return float(free_share) * 1e4 * float(close)
        return float(r["circ_mv"]) * 1e4
    except (TypeError, ValueError, KeyError):
        return None


def fetch_cap_snapshot(pro, trade_date, cap_field="circ_mv"):
    """Fetch one whole-market day of daily_basic -> {ts_code: cap(元)}."""
    require(cap_field in CAP_FIELDS, "unknown cap field")
    df = pro.daily_basic(trade_date=trade_date,
                         fields="ts_code,trade_date,close,total_mv,circ_mv,free_share")
    if df is None or not len(df):
        return None
    out = {}
    for r in df.to_dict("records"):
        code = r.get("ts_code")
        if not code:
            continue
        cap = _extract_cap(r, cap_field)
        if cap is not None:
            out[code] = cap
    return out


def snapshot_document(trade_date, cap_field, caps, source="tushare_daily_basic"):
    """Wrap a {code: cap} map into a content-addressed, self-describing document."""
    return {
        "schema_version": "fuxi.cap_snapshot.1",
        "trade_date": trade_date,
        "cap_field": cap_field,
        "cap_field_name": CAP_FIELDS[cap_field]["name"],
        "cap_unit": "cny_yuan",
        "source": source,
        "n_codes": len(caps),
        "data_hash": digest({k: round(v, 4) for k, v in caps.items()}),
        "caps": caps,
    }


def save_snapshot(doc, out_dir, trade_date, cap_field):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cap_{trade_date}_{cap_field}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def fetch_cap_range(pro, dates, cap_field="circ_mv", out_dir="data/cap_snapshots",
                    sleep_s=0.6):
    """Fetch several trade dates with throttling + resume (skips existing files).

    Returns a list of saved Path objects.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for d in dates:
        target = out_dir / f"cap_{d}_{cap_field}.json"
        if target.exists():
            saved.append(target)
            continue
        caps = fetch_cap_snapshot(pro, d, cap_field)
        if caps is None:
            print(f"[skip] {d}: no data")
            continue
        doc = snapshot_document(d, cap_field, caps)
        saved.append(save_snapshot(doc, out_dir, d, cap_field))
        print(f"[ok] {d}: {len(caps)} codes -> {target.name}")
        time.sleep(sleep_s)
    return saved


def load_cap_snapshots(out_dir="data/cap_snapshots", cap_field="circ_mv"):
    """Load all cached snapshots for a field into {trade_date: {code: cap元}}."""
    out_dir = Path(out_dir)
    by_date = {}
    for p in sorted(out_dir.glob(f"cap_*_{cap_field}.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        require(doc["cap_field"] == cap_field, "cap field mismatch in cache")
        by_date[doc["trade_date"]] = doc["caps"]
    return by_date


# ---------------------------------------------------------------- 按代码批量拉序列
def fetch_cap_series_by_codes(pro, codes, start_date, end_date, cap_field="circ_mv",
                              sleep_s=0.15):
    """按代码批量拉 daily_basic 序列 -> {trade_date: {code: cap元}}。

    与 ``fetch_cap_range`` 的整市场逐日不同，这里按 ``ts_code`` 一次拉一整段序列，
    适合只关心研究池（如 cohort64）若干代码、逐日全市场太重的场景。
    start/end 用 YYYYMMDD；返回日期归一化为 YYYY-MM-DD。
    """
    from collections import defaultdict
    require(cap_field in CAP_FIELDS, "unknown cap field")
    out = defaultdict(dict)
    for code in codes:
        tscode = to_tushare_code(code)
        df = pro.daily_basic(ts_code=tscode, start_date=start_date, end_date=end_date,
                             fields="ts_code,trade_date,close,total_mv,circ_mv,free_share")
        if df is None or not len(df):
            continue
        for r in df.to_dict("records"):
            d = _normalize_date(r.get("trade_date"))
            if not d:
                continue
            cap = _extract_cap(r, cap_field)
            if cap is not None:
                out[d][code] = cap  # 以快照代码为键，供 applicability 直接按 universe code 查询
        time.sleep(sleep_s)
    return {d: dict(cs) for d, cs in sorted(out.items())}


def cap_series_document(cap_by_date, cap_field, codes, start_date, end_date):
    """包装 {trade_date: {code: cap}} 为 content-addressed 文档。"""
    flat = {f"{d}|{c}": round(v, 4) for d, cs in cap_by_date.items() for c, v in cs.items()}
    return {
        "schema_version": "fuxi.cap_series.1",
        "cap_field": cap_field,
        "cap_field_name": CAP_FIELDS[cap_field]["name"],
        "cap_unit": "cny_yuan",
        "source": "tushare_daily_basic",
        "codes": sorted(codes),
        "start_date": start_date,
        "end_date": end_date,
        "n_days": len(cap_by_date),
        "n_codes": len(codes),
        "data_hash": digest(flat),
        "cap_by_date": cap_by_date,
    }


def save_cap_series(doc, out_dir="data/cap_series"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cap_series_{doc['cap_field']}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def load_cap_series(out_dir="data/cap_series", cap_field="circ_mv"):
    """加载 {trade_date: {code: cap}} 单文件缓存；缺失返回 None（不触网）。"""
    path = Path(out_dir) / f"cap_series_{cap_field}.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    require(doc["schema_version"] == "fuxi.cap_series.1", "cap series schema mismatch")
    flat = {f"{d}|{c}": round(v, 4) for d, cs in doc["cap_by_date"].items() for c, v in cs.items()}
    require(doc["data_hash"] == digest(flat), "cap series hash changed")
    return doc["cap_by_date"]


def default_cap_by_session(cap_field="circ_mv", out_dir="data/cap_series"):
    """Best-effort 从缓存加载 {trade_date: {code: cap}}（不触网）；缺失返回 None。

    供评估阶段安全调用：缓存存在则返回逐日市值映射；缓存缺失则如实返回 None
    （applicability 据此标注 cap_labels_unavailable），不隐式失败。
    """
    import functools

    @functools.lru_cache(maxsize=1)
    def _load(field, od):
        return load_cap_series(od, field)

    return _load(cap_field, out_dir)


if __name__ == "__main__":
    import sys
    from credentials import get_tushare_api
    if len(sys.argv) < 2:
        print("usage: python -m fuxi_v1.cap_data <trade_date|start:end> "
              "[--field circ_mv|total_mv|free_float] [--out data/cap_snapshots]")
        sys.exit(2)
    arg = sys.argv[1]
    field = "circ_mv"
    out_dir = "data/cap_snapshots"
    if "--field" in sys.argv:
        field = sys.argv[sys.argv.index("--field") + 1]
    if "--out" in sys.argv:
        out_dir = sys.argv[sys.argv.index("--out") + 1]
    pro = get_tushare_api()
    if ":" in arg:
        start, end = arg.split(":")
        import pandas as pd
        cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")
        dates = sorted(cal["cal_date"].tolist())
    else:
        dates = [arg]
    fetch_cap_range(pro, dates, field, out_dir)
