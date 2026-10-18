# -*- coding: utf-8 -*-
"""
周频缓存刷新 / 新鲜度守卫 (P-20260915-008)

背景 (2026-09-15 实测):
- output/ap_batch/cache/weekly_prices.parquet 冻结于 2026-06-21 (mtime 07-01)
- output/ap_batch/cache/weekly_ohlc.parquet  冻结于 2026-06-21 (mtime 08-29)
- 而 data/raw/daily_prices.csv 已到 2026-09-14  ⇒ 落后 12 周 (约 2.8 个月)

根因: scripts/ap_batch_factor_test.py::load_price_weekly() 命中缓存即 return,
      无 TTL / 无日期校验 ⇒ 缓存一旦生成永不刷新 (死数据源)。

影响面 (grep 全仓确认的消费者):
- research/factor_alchemy/ralph_loop.py  → Forge 引擎 (~60% 生成量) 的进化地形
- research/factor_alchemy/weekly_lane.py → v0.7 周频裁决器 (WeeklyLaneJudge)
- research/factor_alchemy/lane_calibration.py → τ_w 校准集 (周频门槛标定)
- scripts/f_score_test.py / scripts/neutralize_c_factors.py

本脚本:
  --check    仅报告新鲜度 (退出码 0=新鲜 / 1=陈旧), 供流水线调用
  --rebuild  从 daily_prices.csv 全量重建两个 parquet (先备份旧文件)
  --verify   与现有 parquet 对拍 (重叠期一致性), 不写文件

口径 (严格复刻, 勿改):
  周标签 = 该周周日 (dayofweek=6), 与历史 parquet 一致
  close = 周内最后交易日 close; open = 周内首交易日 open
  high  = 周内 max(high); low = 周内 min(low)
  vol/amount = 周内 sum
  fwd_ret = 下一周 close / 本周 close - 1  (末周 NaN → drop)
  momentum_12_1 = close.pct_change(52)
  vol_60 = 周收益 rolling(12, min_periods=6).std()
  log_size = log(|close| * amount.clip(lower=1) + 1e-8)
  最后 dropna(subset=['fwd_ret','close'])

数据卫生: daily_prices.csv 行级重复严重 → 先 drop_duplicates (13.4M→7.0M)。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "daily_prices.csv"
CACHE = ROOT / "output" / "ap_batch" / "cache"
WK_PRICES = CACHE / "weekly_prices.parquet"
WK_OHLC = CACHE / "weekly_ohlc.parquet"

STALE_WARN_WEEKS = 2   # 落后 >= 2 周即视为陈旧


def _mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if p.exists() else "缺失"


def _parquet_max_week(p: Path) -> pd.Timestamp | None:
    if not p.exists():
        return None
    try:
        d = pd.read_parquet(p, columns=["trade_date"])
    except Exception:
        return None
    t = pd.to_datetime(d["trade_date"], errors="coerce")
    return t.max() if len(t) else None


def _daily_max_date() -> pd.Timestamp | None:
    if not SRC.exists():
        return None
    d = pd.read_csv(SRC, usecols=["trade_date"])
    t = pd.to_datetime(d["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    return t.max()


def _last_complete_week(max_day: pd.Timestamp) -> pd.Timestamp:
    """最后一个 fwd_ret 可算的周标签 (周日)。

    周标签 = 该周周日。本周的 fwd_ret 需要下一周 close,
    故最新可用周 = 上一周 (含) 之前。
    """
    # 该日所属周的周日
    dow = int(max_day.dayofweek)
    this_week = max_day + pd.Timedelta(days=6 - dow)
    # 本周末尾未收盘 ⇒ 退一周
    return this_week - pd.Timedelta(days=7)


def cmd_check() -> int:
    print("=" * 68)
    print("  周频缓存新鲜度检查")
    print("=" * 68)
    d_max = _daily_max_date()
    print(f"  daily_prices.csv 最新日期 : {d_max.date() if d_max is not None else '读取失败'}")
    stale = 0
    for p in (WK_PRICES, WK_OHLC):
        w = _parquet_max_week(p)
        if w is None:
            print(f"  {p.name:26s}: 缺失 → 需重建")
            stale += 1
            continue
        if d_max is None:
            print(f"  {p.name:26s}: 最新周 {w.date()} | mtime {_mtime(p)} | 无法判定(日频读取失败)")
            continue
        lag_w = int((_last_complete_week(d_max) - w).days // 7)
        flag = "✅ 新鲜" if lag_w < STALE_WARN_WEEKS else f"🔴 陈旧 (落后 {lag_w} 周)"
        print(f"  {p.name:26s}: 最新周 {w.date()} | mtime {_mtime(p)} | {flag}")
        if lag_w >= STALE_WARN_WEEKS:
            stale += 1
    print("-" * 68)
    if stale:
        print(f"  ⚠️ {stale} 个缓存陈旧 ⇒ 受影响消费者: Forge 引擎 / WeeklyLane 周频裁决器 /")
        print("     lane_calibration τ_w 校准集。请执行 --rebuild。")
    else:
        print("  ✅ 全部新鲜")
    return 1 if stale else 0


def _build_frames(verbose: bool = True):
    """从 daily_prices.csv 重建周频 prices + ohlc 两个 DataFrame (原口径)。"""
    t0 = time.time()
    if not SRC.exists():
        raise FileNotFoundError(f"缺少日频源: {SRC}")

    if verbose:
        print(f"  [1/5] 读取日频 {SRC.name} ...", flush=True)
    d = pd.read_csv(
        SRC,
        usecols=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"],
        low_memory=False,
    )
    _raw = len(d)
    if verbose:
        print(f"        原始 {_raw:,} 行 | {time.time() - t0:.0f}s", flush=True)
    d = d.drop_duplicates()
    if verbose:
        print(f"        去重后 {len(d):,} 行 (删 {_raw - len(d):,} 行级重复) | {time.time() - t0:.0f}s",
              flush=True)

    d["trade_date"] = pd.to_datetime(d["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    d = d.dropna(subset=["trade_date"])
    d = d.sort_values(["ts_code", "trade_date"], kind="mergesort")

    # 周标签 = 该周周日 (与历史 parquet 一致)
    dow = d["trade_date"].dt.dayofweek
    d["week_label"] = d["trade_date"] + pd.to_timedelta(6 - dow, unit="D")

    if verbose:
        print(f"  [2/5] 聚合周频 (close=last / vol,amount=sum / OHLC) ...", flush=True)
    g = d.groupby(["ts_code", "week_label"], sort=False)
    wk = g.agg(
        close=("close", "last"),
        vol=("vol", "sum"),
        amount=("amount", "sum"),
    ).reset_index()
    ohlc = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
    ).reset_index()
    wk = wk.rename(columns={"ts_code": "stock_code", "week_label": "trade_date"})
    ohlc = ohlc.rename(columns={"ts_code": "stock_code", "week_label": "trade_date"})
    if verbose:
        print(f"        周频 {len(wk):,} 行 / {wk['trade_date'].nunique()} 周 "
              f"| {time.time() - t0:.0f}s", flush=True)

    wk = wk.sort_values(["stock_code", "trade_date"], kind="mergesort")
    ohlc = ohlc.sort_values(["stock_code", "trade_date"], kind="mergesort")

    if verbose:
        print("  [3/5] 计算 fwd_ret / 风格代理 ...", flush=True)
    # 🔴 严格复刻原口径 (scripts/ap_batch_factor_test.py L72):
    #    groupby(...).pct_change()  →  再对整个 Series .shift(-1)
    #    不要改成组内 shift: 每只股票首行 pct_change=NaN 天然起到跨股隔离作用,
    #    两者结果在正常排序下等价, 但为免口径漂移一律照抄原写法。
    wk["fwd_ret"] = wk.groupby("stock_code", sort=False)["close"].pct_change().shift(-1)
    wk["momentum_12_1"] = wk.groupby("stock_code", sort=False)["close"].transform(
        lambda x: x.pct_change(52)
    )
    wk["_r"] = wk.groupby("stock_code", sort=False)["close"].transform(
        lambda x: x.pct_change()
    )
    wk["vol_60"] = wk.groupby("stock_code", sort=False)["_r"].transform(
        lambda x: x.rolling(12, min_periods=6).std()
    )
    wk = wk.drop(columns=["_r"])
    wk["log_size"] = np.log(wk["close"].abs() * wk["amount"].clip(lower=1) + 1e-8)
    wk["log_size"] = wk["log_size"].replace([np.inf, -np.inf], np.nan)

    if verbose:
        print("  [4/5] dropna(fwd_ret, close) ...", flush=True)
    before = len(wk)
    wk = wk.dropna(subset=["fwd_ret", "close"])
    if verbose:
        print(f"        {before:,} → {len(wk):,} 行 (末周无 fwd_ret 被剔除)", flush=True)

    wk = wk[["stock_code", "trade_date", "close", "vol", "amount",
             "fwd_ret", "momentum_12_1", "vol_60", "log_size"]]
    # OHLC 与 prices 键集合精确对齐 (复刻 _rebuild_weekly_ohlc_20260829.py 的 inner merge)
    ohlc = ohlc.merge(wk[["stock_code", "trade_date"]], on=["stock_code", "trade_date"], how="inner")
    ohlc = ohlc[["stock_code", "trade_date", "open", "high", "low"]]
    if verbose:
        print(f"  [5/5] 完成 | 周数 {wk['trade_date'].nunique()} "
              f"| 区间 {wk['trade_date'].min().date()} ~ {wk['trade_date'].max().date()} "
              f"| {time.time() - t0:.0f}s", flush=True)
    return wk, ohlc


def _verify_against(old: Path, new_df: pd.DataFrame, keys, cols) -> None:
    if not old.exists():
        print(f"  (无旧文件可对拍: {old.name})")
        return
    o = pd.read_parquet(old)
    o["trade_date"] = pd.to_datetime(o["trade_date"], errors="coerce")
    m = o.merge(new_df, on=keys, how="inner", suffixes=("_old", "_new"))
    print(f"  {old.name}: 重叠 {len(m):,} 行 ({len(o):,} 旧 / {len(new_df):,} 新)")
    for c in cols:
        co, cn = f"{c}_old", f"{c}_new"
        if co not in m.columns or cn not in m.columns:
            continue
        a, b = m[co], m[cn]
        both = a.notna() & b.notna()
        if not both.any():
            print(f"    {c:16s}: 无共同非空值")
            continue
        diff = (a[both] - b[both]).abs()
        rel = (diff / b[both].abs().clip(lower=1e-12)).median()
        print(f"    {c:16s}: max|Δ|={diff.max():.6g}  中位相对偏差={rel:.3%}  "
              f"非空重合={int(both.sum()):,}")


def cmd_verify() -> int:
    print("=" * 68)
    print("  对拍模式 (不写文件)")
    print("=" * 68)
    wk, ohlc = _build_frames(verbose=True)
    print("-" * 68)
    _verify_against(WK_PRICES, wk, ["stock_code", "trade_date"],
                    ["close", "vol", "amount", "fwd_ret", "momentum_12_1", "vol_60", "log_size"])
    print("-" * 68)
    _verify_against(WK_OHLC, ohlc, ["stock_code", "trade_date"], ["open", "high", "low"])
    return 0


def cmd_rebuild() -> int:
    print("=" * 68)
    print("  重建周频缓存")
    print("=" * 68)
    wk, ohlc = _build_frames(verbose=True)
    print("-" * 68)
    print("  对拍 (重叠期):")
    _verify_against(WK_PRICES, wk, ["stock_code", "trade_date"],
                    ["close", "vol", "amount", "fwd_ret", "momentum_12_1", "vol_60", "log_size"])
    _verify_against(WK_OHLC, ohlc, ["stock_code", "trade_date"], ["open", "high", "low"])
    print("-" * 68)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CACHE.mkdir(parents=True, exist_ok=True)
    for p in (WK_PRICES, WK_OHLC):
        if p.exists():
            bak = p.with_suffix(f".parquet.bak_{stamp}")
            shutil.copy2(p, bak)
            print(f"  已备份 {p.name} → {bak.name}")
    wk.to_parquet(WK_PRICES, index=False)
    ohlc.to_parquet(WK_OHLC, index=False)
    print(f"  ✅ 已写入 {WK_PRICES.name}: {len(wk):,} 行 / "
          f"{wk['trade_date'].nunique()} 周 / 至 {wk['trade_date'].max().date()}")
    print(f"  ✅ 已写入 {WK_OHLC.name}: {len(ohlc):,} 行 / 至 {ohlc['trade_date'].max().date()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="周频缓存刷新 / 新鲜度守卫")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="仅报告新鲜度 (陈旧退出码 1)")
    g.add_argument("--rebuild", action="store_true", help="全量重建 (先备份)")
    g.add_argument("--verify", action="store_true", help="与现有 parquet 对拍, 不写文件")
    a = ap.parse_args()
    if a.check:
        return cmd_check()
    if a.verify:
        return cmd_verify()
    return cmd_rebuild()


if __name__ == "__main__":
    sys.exit(main())
