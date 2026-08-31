# -*- coding: utf-8 -*-
"""
补采 tushare 日频 adj_factor -> data/raw/adj_factor_daily.csv
===============================================================
根因: 现有 adj_factor.csv 为季频快照(每季末2天, 46个唯一日期), 拆股/送转发生在季中时,
      dump_qlib_bin.py 的 searchsorted 命中上季末旧因子 -> 除权日~季末错误段价格 = 真实价/比例,
      季末快照更新日价格跳回 -> 组合模拟假暴涨 (2026-08-27 P2 对账发现的 12 只污染).

方案: 按 trade_date 逐日拉全市场日频 adj_factor (tushare adj_factor 接口支持单日全市场).
      2021-01-01 ~ 今天 约 1370 交易日 x ~5000 行/次, 断点续传 + 限频重试.

用法:
    python scripts/collect_adj_factor_daily.py            # 全量 (2021-01-01 起)
    python scripts/collect_adj_factor_daily.py --start 20260101   # 指定起点
    python scripts/collect_adj_factor_daily.py --dry-run
"""
import sys
import os
import time
from pathlib import Path
import argparse
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from credentials import get_tushare_api

RAW = Path(__file__).resolve().parent.parent / 'data' / 'raw'
OUT = RAW / "adj_factor_daily.csv"
STATE = RAW / "adj_factor_daily_state.txt"   # 断点: 每行一个已完成的 trade_date


def get_existing_dates():
    if not STATE.exists():
        return set()
    return set(l.strip() for l in STATE.read_text().splitlines() if l.strip())


def save_date(d8):
    with open(STATE, "a") as f:
        f.write(d8 + "\n")


def load_existing_out():
    if OUT.exists():
        df = pd.read_csv(OUT, dtype={"ts_code": str, "trade_date": str}, low_memory=False)
        return df
    return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])


def fetch_one_day(pro, d8, retries=5):
    """拉单日全市场 adj_factor, 限频退避重试"""
    for i in range(retries):
        try:
            df = pro.adj_factor(trade_date=d8, fields="ts_code,trade_date,adj_factor")
            if df is not None and len(df):
                df["trade_date"] = df["trade_date"].astype(str)
                return df
            return None
        except Exception as e:
            msg = str(e)
            if "每分钟" in msg or "limit" in msg.lower() or "频率" in msg:
                wait = min(60 + 30 * i, 300)
                print(f"    [WAIT] 限频 {d8} 等待 {wait}s ({msg[:60]})", flush=True)
                time.sleep(wait)
            else:
                print(f"    [ERR] {d8}: {msg[:120]}", flush=True)
                time.sleep(2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20210101", help="起点 YYYYMMDD")
    ap.add_argument("--end", default="", help="终点 YYYYMMDD, 默认今天")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.32, help="每次调用间隔秒 (5000积分 200次/分)")
    args = ap.parse_args()

    # 交易日历: 用 daily_prices.csv 的 trade_date 集合 (与 dump 口径一致)
    print("[1/3] 载入交易日历 ...", flush=True)
    cal = pd.read_csv(
        RAW / "daily_prices.csv", usecols=["trade_date"], dtype={"trade_date": str},
        low_memory=False)
    cal["trade_date"] = cal["trade_date"].astype(str).str.strip().str.replace("-", "", regex=False)
    days = sorted(d for d in cal["trade_date"].dropna().unique()
                  if d.isdigit() and len(d) == 8 and args.start <= d <= (args.end or "99991231"))
    print(f"    交易日: {len(days)} ({days[0]} ~ {days[-1]})", flush=True)

    done = get_existing_dates()
    todo = [d for d in days if d not in done]
    print(f"[2/3] 断点: 已完成 {len(done)}, 待拉 {len(todo)}", flush=True)
    if args.dry_run:
        print(f"    [DRY-RUN] 将拉取 {len(todo)} 天, 预估 {len(todo)*args.sleep:.0f}s+调用时间")
        return

    out = load_existing_out()
    pro = get_tushare_api()
    t0 = time.time()
    for i, d8 in enumerate(todo):
        df = fetch_one_day(pro, d8)
        if df is not None:
            out = pd.concat([out, df], ignore_index=True)
        save_date(d8)
        if (i + 1) % 50 == 0:
            out.to_csv(OUT, index=False, encoding="utf-8-sig")
            done_n = i + 1
            eta = (time.time() - t0) / done_n * (len(todo) - done_n)
            print(f"    {i+1}/{len(todo)} 天, 累计 {len(out):,} 行, ETA {eta/60:.1f} 分钟", flush=True)
        time.sleep(args.sleep)

    out = out.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    out = out.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"[3/3] DONE: {len(out):,} 行, {out['trade_date'].nunique()} 天, "
          f"{out['ts_code'].nunique()} 只, 耗时 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
