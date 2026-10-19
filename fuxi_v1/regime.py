"""沪深300 市场状态（牛熊 regime）划分，产出 {session: regime}。

用途：为 applicability（factor_note）提供 regime 标签，按市场状态分组计算
因子 rank IC，识别因子在牛/熊市下的适用性差异（note-only，不参与 pass/reject）。

定义（可配置 ``ma_window``）：
  · 标的 = 沪深300 (000300.SH) 日收盘（tushare index_daily）；
  · 规则 = close > MA(ma_window) → 'bull'，否则 'bear'（长期趋势划分）；
  · warmup = 需 ma_window 个交易日前置历史，不足则该日不产出标签。

缓存：``data/regime/csi300_closes.json``（content-addressed，存原始收盘价，与
``ma_window`` 解耦——改 MA 窗口无需重新拉数）。加载函数不触网，可在评估阶段安全调用；
fetch 只在显式提供 ``pro``（tushare）且缓存缺失时发生。
"""
import json
from pathlib import Path

from .contracts import digest, require

__all__ = ["CSI300", "fetch_csi300_closes", "classify_regime", "csi300_regime",
           "load_csi300_closes", "default_csi300_regime", "DEFAULT_MA_WINDOW"]

CSI300 = "000300.SH"
DEFAULT_MA_WINDOW = 200
DEFAULT_CACHE_DIR = "data/regime"
CACHE_NAME = "csi300_closes.json"


def _normalize_date(d):
    """tushare trade_date 有 'YYYYMMDD' 与 'YYYY-MM-DD' 两种，统一为 'YYYY-MM-DD'。"""
    s = str(d).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def fetch_csi300_closes(pro, start_date, end_date):
    """tushare index_daily 拉沪深300日收盘 -> {trade_date: close}（升序）。

    兼容 DataFrame 与 list[dict] 两种返回（测试用 fake pro 返回 list）。
    """
    df = pro.index_daily(ts_code=CSI300, start_date=start_date, end_date=end_date)
    if df is None or not len(df):
        return None
    records = df.to_dict("records") if hasattr(df, "to_dict") else list(df)
    closes = {}
    for r in records:
        try:
            closes[_normalize_date(r["trade_date"])] = float(r["close"])
        except (TypeError, ValueError, KeyError):
            continue
    require(closes, "csi300 empty after parse")
    return dict(sorted(closes.items()))


def _closes_doc(closes, start_date, end_date):
    return {
        "schema_version": "fuxi.csi300_closes.1",
        "ts_code": CSI300,
        "start_date": start_date,
        "end_date": end_date,
        "n_days": len(closes),
        "source": "tushare_index_daily",
        "data_hash": digest({k: round(v, 4) for k, v in closes.items()}),
        "closes": closes,
    }


def _cache_path(cache_dir):
    return Path(cache_dir) / CACHE_NAME


def save_csi300_closes(closes, start_date, end_date, cache_dir=DEFAULT_CACHE_DIR):
    path = _cache_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _closes_doc(closes, start_date, end_date)
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def load_csi300_closes(cache_dir=DEFAULT_CACHE_DIR):
    """加载缓存原始收盘价；缺失返回 None（不触网）。"""
    path = _cache_path(cache_dir)
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    require(doc["schema_version"] == "fuxi.csi300_closes.1", "csi300 cache schema mismatch")
    require(doc["data_hash"] == digest({k: round(v, 4) for k, v in doc["closes"].items()}),
            "csi300 cache hash changed")
    return doc


def classify_regime(closes, ma_window=DEFAULT_MA_WINDOW):
    """按 MA(ma_window) 划分牛熊 -> {trade_date: 'bull'|'bear'}（仅满窗日期）。"""
    require(ma_window >= 2 and isinstance(ma_window, int), "invalid ma_window")
    result = {}
    window = []
    for d in sorted(closes):
        window.append(closes[d])
        if len(window) > ma_window:
            window.pop(0)
        if len(window) == ma_window:
            ma = sum(window) / ma_window
            result[d] = "bull" if closes[d] > ma else "bear"
    return result


def csi300_regime(sessions, pro=None, ma_window=DEFAULT_MA_WINDOW,
                  cache_dir=DEFAULT_CACHE_DIR, refresh=False):
    """产出 {session: regime}。缓存缺失且给了 pro 则先 fetch；否则只读缓存。

    session 日期若无对应 regime（warmup 不足/缺索引日）则跳过该 session。
    """
    doc = None if refresh else load_csi300_closes(cache_dir)
    if doc is None:
        if pro is None:
            return None
        sessions = sorted(sessions)
        require(sessions, "empty sessions")
        start = sessions[0]
        # 前置 ~2 年日历日以覆盖 ma_window 交易日前置历史（tushare 接受 YYYYMMDD）。
        from datetime import datetime, timedelta
        fetch_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=750)).strftime("%Y%m%d")
        fetch_end = sessions[-1].replace("-", "")
        closes = fetch_csi300_closes(pro, fetch_start, fetch_end)
        require(closes, "csi300 fetch failed")
        save_csi300_closes(closes, fetch_start, fetch_end, cache_dir)
        doc = _closes_doc(closes, fetch_start, fetch_end)
    regime_by_date = classify_regime(doc["closes"], ma_window)
    out = {}
    for d in sessions:
        r = regime_by_date.get(d)
        if r is not None:
            out[d] = r
    return out


def default_csi300_regime(ma_window=DEFAULT_MA_WINDOW, cache_dir=DEFAULT_CACHE_DIR):
    """Best-effort 从缓存加载 {session: regime}（不触网）；缺失返回 None。

    供评估阶段安全调用：缓存存在则按 session 全集产出标签；缓存缺失则如实
    返回 None（applicability 据此标注 regime_labels_unavailable），不隐式失败。
    """
    import functools

    @functools.lru_cache(maxsize=1)
    def _load(ma, cd):
        doc = load_csi300_closes(cd)
        return None if doc is None else classify_regime(doc["closes"], ma)

    return _load(ma_window, cache_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m fuxi_v1.regime <start_date:end_date> [--ma 200] [--cache data/regime]")
        sys.exit(2)
    arg = sys.argv[1]
    ma = DEFAULT_MA_WINDOW
    cache = DEFAULT_CACHE_DIR
    if "--ma" in sys.argv:
        ma = int(sys.argv[sys.argv.index("--ma") + 1])
    if "--cache" in sys.argv:
        cache = sys.argv[sys.argv.index("--cache") + 1]
    from credentials import get_tushare_api
    pro = get_tushare_api()
    start, end = arg.split(":")
    closes = fetch_csi300_closes(pro, start, end)
    save_csi300_closes(closes, start, end, cache)
    regime = classify_regime(closes, ma)
    bull = sum(1 for v in regime.values() if v == "bull")
    bear = len(regime) - bull
    print(f"csi300 closes={len(closes)} regime_days={len(regime)} bull={bull} bear={bear}")
    print(f"cache -> {_cache_path(cache)}")
