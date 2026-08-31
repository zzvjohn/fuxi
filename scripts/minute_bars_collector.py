# -*- coding: utf-8 -*-
"""
scripts/minute_bars_collector.py — P-20260822-004 落地一期: 分钟级行情采集基座
================================================================================
新浪财经 5 分钟 K 线全市场增量采集 (quotes.sina.cn CN_MarketDataService.getKLineData)。

数据源选型实录 (2026-08-22 实测):
- 东财 push2his (akshare stock_zh_a_hist_min_em): 调试期 15+ 次快速请求触发 WAF,
  当前 IP 被拉黑 (家庭网络同样 RemoteDisconnected) → 不适合每日 5000 次请求的采集。
- baostock: 首次 login 成功 (720 根 5min bar 验证通过), 此后所有 login 挂起 (>8 分钟),
  socket 超时无效, 疑似服务端连接节流 → 弃用。
- 新浪 KLineData: ✅ 稳定可用。datalen=1023 上限 (约 21 个交易日), 200 状态码,
  376ms/只 (Session 复用后更快), 无 WAF 迹象 → 选定为主源。

设计约束 (2026-08-22 提案审核裁决 + 数据源实测修正):
- 单次调用历史深度 = 1023 根 (~21 交易日), 不是提案原设的 3 个月。策略:
  * backfill: 一次拉满 1023 根建库;
  * incremental: 每日拉 48 根 (最近 1 个交易日) 追加, 本地滚动保留 keep_days (默认 20)。
  * 深历史 (3 个月+) 待后续数据源补充 (东财解封后 akshare 一次性回填, 或 baostock 恢复)。
- 静默失败防护: 返回非 JSON (参数越界/被限流) 重试 2 次, 仍失败记为 error (不污染下游)。
- 全市场 ~5000 只, 单只 ~200-400ms → 全量回填约 20-35 分钟 (后台跑), 增量约 10-15 分钟。
- 独立目录 data/raw/minute_bars/, 不影响现有 Stage0。
- 每 50 只落盘 meta 进度, 中断可续 (per-stock parquet 已 flush)。

用法:
    # 建库回填 (拉满 1023 根)
    C:/Python314/python.exe scripts/minute_bars_collector.py --mode backfill
    # 每日增量 (拉 48 根追加)
    C:/Python314/python.exe scripts/minute_bars_collector.py --mode incremental
    # 小样本验证
    C:/Python314/python.exe scripts/minute_bars_collector.py --limit 5
"""
import argparse
import json
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BARS_DIR = ROOT / "data" / "raw" / "minute_bars"
META_PATH = BARS_DIR / "_meta.json"
PRICE_PATH = ROOT / "data" / "raw" / "daily_prices.csv"

SINA_URL = "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_x=/CN_MarketDataService.getKLineData"
BACKFILL_LEN = 1023   # 单次调用历史上限 (~21 交易日)
INCREMENT_LEN = 48    # 一个交易日的 5 分钟 bar 数
MAX_RETRY = 2
SLEEP_JITTER = 0.08   # 限速抖动 (s)
SAVE_EVERY = 50
JSONP_RE = re.compile(r"var _x=\((.*)\);?$", re.S)


def _load_meta() -> dict:
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"created_at": None, "stocks": {}, "stats": {}}


def _save_meta(meta: dict):
    BARS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = META_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)
    tmp.replace(META_PATH)


def load_stock_codes() -> list[str]:
    """本地 daily_prices.csv 最新截面 SH/SZ 代码 (与研究工作区口径一致)。"""
    import pandas as pd
    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"无本地股票池: {PRICE_PATH}")
    df = pd.read_csv(PRICE_PATH, usecols=["ts_code"], low_memory=False)
    codes = df["ts_code"].dropna().astype(str).unique().tolist()
    codes = [c for c in codes if c.endswith((".SH", ".SZ"))]
    return sorted(codes)


def _to_sina_symbol(code: str) -> str:
    """'000001.SZ' → 'sz000001' ; '600000.SH' → 'sh600000'"""
    num, ex = code.split(".")
    return f"{ex.lower()}{num}"


def fetch_one(session, code: str, datalen: int):
    """单只拉取, 返回 (df, status) ; status ∈ ok/empty/error。含重试。"""
    symbol = _to_sina_symbol(code)
    for attempt in range(MAX_RETRY + 1):
        try:
            r = session.get(SINA_URL, params={
                "symbol": symbol, "scale": "5", "ma": "no", "datalen": str(datalen),
            }, timeout=15)
        except Exception as e:
            if attempt < MAX_RETRY:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None, f"error:{type(e).__name__}"
        if r.status_code != 200:
            if attempt < MAX_RETRY:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None, f"error:http{r.status_code}"
        m = JSONP_RE.search(r.text)
        if not m:
            # 静默失败防护: 非 JSON 响应 (参数越界/限流页) 不落盘
            if attempt < MAX_RETRY:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None, "error:bad_payload"
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            if attempt < MAX_RETRY:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None, "error:bad_json"
        if not data:
            return None, "empty"
        import pandas as pd
        df = pd.DataFrame(data)
        return normalize(df), "ok"
    return None, "error:unknown"


def normalize(df) -> "pd.DataFrame":
    """新浪 json → 标准列 (时间/OHLC/量额)。"""
    import pandas as pd
    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(df["day"], errors="coerce")
    out["open"] = pd.to_numeric(df["open"], errors="coerce")
    out["close"] = pd.to_numeric(df["close"], errors="coerce")
    out["high"] = pd.to_numeric(df["high"], errors="coerce")
    out["low"] = pd.to_numeric(df["low"], errors="coerce")
    out["volume"] = pd.to_numeric(df["volume"], errors="coerce")  # 股
    out["amount"] = pd.to_numeric(df["amount"], errors="coerce")  # 元
    out = out.dropna(subset=["datetime"])
    return out.sort_values("datetime").reset_index(drop=True)


def merge_and_prune(existing, new, keep_days: int):
    """合并去重 + 滚动裁剪。"""
    import pandas as pd
    if existing is None or len(existing) == 0:
        df = new
    else:
        df = pd.concat([existing, new], ignore_index=True)
        df = df.drop_duplicates(subset=["datetime"], keep="last")
        df = df.sort_values("datetime").reset_index(drop=True)
    if keep_days and len(df):
        cutoff = df["datetime"].max() - timedelta(days=keep_days)
        df = df[df["datetime"] >= cutoff]
    return df


def run(mode: str, keep_days: int, limit: int | None):
    import pandas as pd
    import requests

    codes = load_stock_codes()
    if limit:
        codes = codes[:limit]
    meta = _load_meta()
    if meta.get("created_at") is None:
        meta["created_at"] = datetime.now().isoformat()

    datalen = BACKFILL_LEN if mode == "backfill" else INCREMENT_LEN
    print(f"[collector] mode={mode} stocks={len(codes)} datalen={datalen} keep_days={keep_days}")
    BARS_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Referer": "https://finance.sina.com.cn/",
    })

    t0 = time.time()
    n_ok = n_empty = n_err = 0
    for i, code in enumerate(codes):
        st = meta["stocks"].get(code, {})
        df, status = fetch_one(session, code, datalen)
        if status == "ok":
            p = BARS_DIR / f"{code}.parquet"
            existing = pd.read_parquet(p) if p.exists() else None
            merged = merge_and_prune(existing, df, keep_days)
            merged.to_parquet(p, index=False)
            st = {
                "latest_bar": str(merged["datetime"].max()),
                "first_bar": str(merged["datetime"].min()),
                "n_bars": int(len(merged)),
                "last_fetch": datetime.now().isoformat(),
                "status": "ok",
            }
            n_ok += 1
        elif status == "empty":
            st = {**st, "last_fetch": datetime.now().isoformat(), "status": "empty"}
            n_empty += 1
        else:
            st = {**st, "last_fetch": datetime.now().isoformat(), "status": status}
            n_err += 1
        meta["stocks"][code] = st

        if (i + 1) % SAVE_EVERY == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(codes) - i - 1)
            meta["stats"] = {
                "last_run": datetime.now().isoformat(), "mode": mode,
                "done": i + 1, "total": len(codes),
                "ok": n_ok, "empty": n_empty, "error": n_err,
                "elapsed_min": round(elapsed / 60, 1),
                "eta_min": round(eta / 60, 1),
            }
            _save_meta(meta)
            print(f"  [{i+1}/{len(codes)}] ok={n_ok} empty={n_empty} err={n_err} "
                  f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
        time.sleep(random.random() * SLEEP_JITTER)

    meta["stats"] = {
        "last_run": datetime.now().isoformat(), "mode": mode,
        "done": len(codes), "total": len(codes),
        "ok": n_ok, "empty": n_empty, "error": n_err,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    _save_meta(meta)

    # ── 质量报告 ──
    print("\n" + "=" * 70)
    print(f"[collector] 完成: 成功 {n_ok} / 空 {n_empty} / 错误 {n_err} | 耗时 {(time.time()-t0)/60:.1f}m")
    ok_stocks = {c: s for c, s in meta["stocks"].items() if s.get("status") == "ok" and s.get("latest_bar")}
    if ok_stocks:
        latest = pd.Series({c: s["latest_bar"] for c, s in ok_stocks.items()})
        latest_dt = pd.to_datetime(latest)
        print(f"[质量] 成功股票 {len(ok_stocks)} 只")
        print(f"[质量] 最新 bar 分布: 最新={latest_dt.max()} | 中位={latest_dt.median()} | 最旧={latest_dt.min()}")
        bars = pd.Series({c: s.get("n_bars", 0) for c, s in ok_stocks.items()})
        print(f"[质量] 每只 bar 数: 中位={bars.median():.0f} | P10={bars.quantile(0.1):.0f} | 最大={bars.max()}")
        stale = (latest_dt < latest_dt.max() - timedelta(days=7)).sum()
        print(f"[质量] 落后 ≥7 天: {stale} 只 (停牌/数据源缺失)")
    print(f"[存储] {BARS_DIR} | meta: {META_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="分钟级行情采集 (新浪 5min)")
    ap.add_argument("--mode", choices=["backfill", "incremental"], default="incremental")
    ap.add_argument("--keep-days", type=int, default=20, help="本地滚动保留天数 (≤21, 受数据源深度限制)")
    ap.add_argument("--limit", type=int, default=None, help="只采前 N 只 (调试)")
    args = ap.parse_args()
    run(args.mode, args.keep_days, args.limit)
