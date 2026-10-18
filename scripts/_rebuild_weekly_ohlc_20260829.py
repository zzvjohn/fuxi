# -*- coding: utf-8 -*-
"""
v0.7 P3c: 从日频 daily_prices.csv 重建周频真实 OHLC (open/high/low)

背景:
- weekly_prices.parquet 只有 close/vol/amount 等列, 无 OHLC
- ralph_loop._load_forge_data 的 open/high/low 全部用 close 代理 (假终端)
- GP 在假终端上演化出的 OHLC 因子, 翻译到 JQ (真实 OHLC) 后行为漂移
  → 第六个口径错配源

口径 (已对拍验证 000001.SZ 2021-01-10 周):
- 周标签 = 该周周日 (weekly_prices.trade_date 全部 dayofweek=6)
- close = 周内最后交易日 close; open = 周内首交易日 open
- high = 周内 max(high); low = 周内 min(low)

输出: output/ap_batch/cache/weekly_ohlc.parquet
  (stock_code, trade_date, open, high, low) — 与 weekly_prices 键集合精确对齐

数据卫生: daily_prices.csv 行级重复严重 (13.4M→6.98M), drop_duplicates;
vol 列行级错位污染不用, open/high/low/close 干净。
"""
import time
import pandas as pd

SRC = "data/raw/daily_prices.csv"
WK = "output/ap_batch/cache/weekly_prices.parquet"
DST = "output/ap_batch/cache/weekly_ohlc.parquet"

t0 = time.time()
print("=" * 60)
print("P3c: 重建周频真实 OHLC", flush=True)

# 1) 周频键集合 (对齐目标)
w = pd.read_parquet(WK, columns=["stock_code", "trade_date"])
w["trade_date"] = pd.to_datetime(w["trade_date"])
print(f"weekly keys: {len(w)} 行, {w['trade_date'].nunique()} 周, "
      f"{w['stock_code'].nunique()} 股", flush=True)

# 2) 读日频 (只取 6 列, 1GB CSV)
print("reading daily csv ...", flush=True)
d = pd.read_csv(SRC, usecols=["ts_code", "trade_date", "open", "high", "low", "close"],
                low_memory=False)
print(f"raw rows: {len(d)}, elapsed {time.time()-t0:.0f}s", flush=True)
d = d.drop_duplicates()
print(f"dedup rows: {len(d)}, elapsed {time.time()-t0:.0f}s", flush=True)
d["trade_date"] = pd.to_datetime(d["trade_date"], format="mixed")

# 3) 周标签 = 该周周日 (dayofweek 6)
dow = d["trade_date"].dt.dayofweek
d["week_label"] = d["trade_date"] + pd.to_timedelta(6 - dow, unit="D")

# 4) 组内按时间排序后聚合 (只输出 OHLC 三列, close 已在 weekly_prices 中,
#    避免 merge 时列名冲突 → close_x/close_y 事故)
d = d.sort_values(["ts_code", "trade_date"], kind="mergesort")
g = d.groupby(["ts_code", "week_label"], sort=False)
ohlc = g.agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min")).reset_index()
ohlc = ohlc.rename(columns={"ts_code": "stock_code", "week_label": "trade_date"})
print(f"weekly aggregated: {len(ohlc)} 行, elapsed {time.time()-t0:.0f}s", flush=True)

# 5) 与 weekly_prices 键集合精确对齐 (inner merge)
merged = w.merge(ohlc, on=["stock_code", "trade_date"], how="inner")
print(f"对齐后: {len(merged)} 行 / {len(w)} 目标 "
      f"(覆盖率 {len(merged)/len(w)*100:.2f}%)", flush=True)

# 6) 输出
merged.to_parquet(DST, index=False)
print(f"written: {DST}", flush=True)
print(f"总耗时 {time.time()-t0:.0f}s", flush=True)
