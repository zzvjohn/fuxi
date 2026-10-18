#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCC 网络中心度因子 — 计算 & 日频IC检验
==========================================
基于华西证券《股票网络与网络中心度因子研究》(2021)
SCC = 股票与全市场其他股票的平均绝对相关系数
方向: 低SCC(网络边缘) → 高收益

实现策略:
- 用 numpy corrcoef 计算相关系数矩阵 → mean abs corr 作为 SCC 代理
- 避免 networkx 全图最短路径 (O(N³) 对5000+股票不可行)
- 限制 2000 只股票（按流动性筛选）

阈值: |ICIR| >= 0.3 AND +IC% >= 55% → 注册到因子池
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys, io, warnings, time

warnings.filterwarnings("ignore")
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).parent.parent
PRICE_FILE = BASE / "data" / "raw" / "daily_prices.csv"
FACTOR_DIR = BASE / "research" / "factor_alchemy" / "output"

# 参数
LOOKBACK = 60        # 相关系数计算窗口 (交易日)
TEST_WINDOW = 200    # IC检验窗口 (交易日)
N_STOCKS = 2000      # 股票数量 (限制计算量)
CORR_THRESHOLD = 0.3  # 相关系数阈值 (用于网络边过滤, 但简化版不用)
ICIR_THRESHOLD = 0.3
IC_PCT_THRESHOLD = 0.55

# ============================================================
# Step 1: 加载数据
# ============================================================
print("=" * 60)
print("Step 1: 加载日频价格数据")
print("=" * 60)

prices = pd.read_csv(PRICE_FILE, encoding="utf-8")
prices["trade_date"] = pd.to_datetime(prices["trade_date"], format="mixed")
prices = prices.sort_values(["ts_code", "trade_date"])

# 筛选有足够交易量的股票
print(f"  原始: {prices['ts_code'].nunique()} 只股票, {prices['trade_date'].nunique()} 个交易日")

# 按最近3个月平均成交额选 Top N
recent = prices[prices["trade_date"] >= prices["trade_date"].max() - pd.Timedelta(days=90)]
avg_amount = recent.groupby("ts_code")["amount"].mean().sort_values(ascending=False)
top_stocks = avg_amount.head(N_STOCKS).index.tolist()
prices = prices[prices["ts_code"].isin(top_stocks)]
print(f"  筛选 Top {N_STOCKS} 按成交额: {prices['trade_date'].nunique()} 个交易日")

# 计算日收益率
prices["ret"] = prices.groupby("ts_code")["close"].pct_change()

# 构建收益率矩阵: (日期 × 股票)
ret_matrix = prices.pivot_table(
    values="ret", index="trade_date", columns="ts_code"
)
ret_matrix = ret_matrix.dropna(axis=1, thresh=int(len(ret_matrix) * 0.8))  # 至少80%非空
ret_matrix = ret_matrix.fillna(0)  # 停牌日收益率=0
print(f"  收益率矩阵: {ret_matrix.shape[0]} 天 × {ret_matrix.shape[1]} 只股票")

# ============================================================
# Step 2: 计算 SCC (平均绝对相关系数)
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 滚动计算 SCC 因子")
print("=" * 60)

dates = ret_matrix.index.tolist()
scc_values = {}  # {date: {stock: scc_value}}

# 从第 LOOKBACK 天开始
test_dates = dates[LOOKBACK:]
n_test = min(TEST_WINDOW, len(test_dates))
test_dates = test_dates[-n_test:]

print(f"  测试窗口: {test_dates[0].date()} ~ {test_dates[-1].date()} ({len(test_dates)} 天)")

t_start = time.time()
for i, test_date in enumerate(test_dates):
    idx = dates.index(test_date)
    window_returns = ret_matrix.iloc[idx - LOOKBACK:idx].values  # (60, N)

    # 计算相关系数矩阵
    # 用 numpy corrcoef — 比 pandas corr() 快
    corr = np.corrcoef(window_returns, rowvar=False)  # (N, N)

    # SCC = 平均绝对相关系数 (排除对角线)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0)
    mean_abs_corr = abs_corr.mean(axis=1)  # (N,) — 每只股票的平均绝对相关系数

    # 存储
    stocks = ret_matrix.columns.tolist()
    scc_values[test_date] = dict(zip(stocks, mean_abs_corr))

    if (i + 1) % 20 == 0:
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
        print(f"  [{i+1}/{len(test_dates)}] {test_date.date()} | {rate:.0f} 天/分 | "
              f"SCC range: [{mean_abs_corr.min():.3f}, {mean_abs_corr.max():.3f}]")

elapsed = time.time() - t_start
print(f"  完成! 耗时: {elapsed:.0f}s ({elapsed/len(test_dates):.1f}s/天)")

# ============================================================
# Step 3: 构建因子 DataFrame → long format
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 构建因子数据")
print("=" * 60)

factor_name = "scc_network_centrality"

records = []
for date, stock_dict in scc_values.items():
    for stock, val in stock_dict.items():
        records.append({"trade_date": date, "stock_code": stock, factor_name: val})

df_factor = pd.DataFrame(records)
print(f"  {factor_name}: {len(df_factor):,} 条")

# ============================================================
# Step 4: 日频 IC 检验
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 日频 IC 检验")
print("=" * 60)

# 计算下一日收益率
next_ret = ret_matrix.shift(-1)

# 匹配因子值和下一日收益
ic_records = []
for test_date in test_dates[:-1]:  # 最后一天没有下一日收益
    if test_date not in scc_values:
        continue

    factor_vals = scc_values[test_date]
    next_date_idx = dates.index(test_date) + 1
    if next_date_idx >= len(dates):
        break
    next_date = dates[next_date_idx]

    rets = next_ret.loc[next_date]

    # 匹配
    common = set(factor_vals.keys()) & set(rets.dropna().index)
    if len(common) < 100:
        continue

    f_vals = np.array([factor_vals[s] for s in common])
    r_vals = rets[list(common)].values

    # Rank IC
    from scipy.stats import spearmanr
    ic, p = spearmanr(f_vals, r_vals)
    ic_records.append({"date": test_date, "IC": ic, "p_value": p})

df_ic = pd.DataFrame(ic_records)
if len(df_ic) == 0:
    print("  ERROR: 无有效IC记录!")
    sys.exit(1)

ic_mean = df_ic["IC"].mean()
ic_std = df_ic["IC"].std()
icir = ic_mean / ic_std if ic_std > 0 else 0
ic_pct = (df_ic["IC"] > 0).sum() / len(df_ic)
t_stat = ic_mean / (ic_std / np.sqrt(len(df_ic))) if ic_std > 0 else 0

print(f"  IC样本: {len(df_ic)} 天")
print(f"  IC均值:  {ic_mean:+.4f}")
print(f"  IC标准差: {ic_std:.4f}")
print(f"  ICIR:     {icir:+.3f}")
print(f"  +IC%:     {ic_pct:.1%}")
print(f"  t统计量:  {t_stat:+.2f}")
print(f"  IC范围:   [{df_ic['IC'].min():+.4f}, {df_ic['IC'].max():+.4f}]")
print(f"  IC>0显著: {'是' if ic_pct > 0.5 else '否'} (p<0.05: {(df_ic['p_value'] < 0.05).sum()}/{len(df_ic)})")

# ============================================================
# Step 5: 判断 & 入库
# ============================================================
print("\n" + "=" * 60)
print("Step 5: 判断结果")
print("=" * 60)

pass_icir = abs(icir) >= ICIR_THRESHOLD
pass_pct = ic_pct >= IC_PCT_THRESHOLD

print(f"  |ICIR| >= {ICIR_THRESHOLD}: {'✅ PASS' if pass_icir else '❌ FAIL'} ({abs(icir):.3f})")
print(f"  +IC%  >= {IC_PCT_THRESHOLD:.0%}: {'✅ PASS' if pass_pct else '❌ FAIL'} ({ic_pct:.1%})")

if pass_icir and pass_pct:
    status = "candidate"
    print(f"\n  🟢 通过! 状态: {status}")
    print(f"  方向: {'做多' if ic_mean > 0 else '做空'}低SCC(网络边缘)股票")
else:
    if abs(icir) >= 0.15:
        status = "reserve"
        print(f"\n  🟡 未通过主阈值, 但ICIR可接受. 状态: {status}")
    else:
        status = "reject"
        print(f"\n  🔴 未通过. 状态: {status}")

# 保存因子CSV
output_path = FACTOR_DIR / f"factor_{factor_name}.csv"
out_df = df_factor.pivot_table(
    values=factor_name, index="trade_date", columns="stock_code"
)
out_df.to_csv(output_path, encoding="utf-8-sig")
print(f"\n  因子CSV已保存: {output_path}")

# 更新因子池
POOL_FILE = BASE / "data" / "passed_factor_pool.csv"
if POOL_FILE.exists():
    pool = pd.read_csv(POOL_FILE)
    # 检查是否已存在
    existing = pool[pool["factor_name"] == factor_name]
    if len(existing) > 0:
        # 更新
        idx = existing.index[0]
        pool.at[idx, "status"] = status
        pool.at[idx, "last_icir"] = round(icir, 4)
        pool.at[idx, "last_ic_pct"] = round(ic_pct, 4)
        pool.at[idx, "last_test_date"] = str(test_dates[-1].date())
        pool.at[idx, "frequency"] = "daily"
        pool.at[idx, "direction"] = "空头" if ic_mean > 0 else "多头"
        print(f"  因子池已更新: {factor_name} → {status}")
    else:
        # 新增
        new_row = {
            "factor_name": factor_name,
            "status": status,
            "category": "network",
            "frequency": "daily",
            "last_icir": round(icir, 4),
            "last_ic_pct": round(ic_pct, 4),
            "last_test_date": str(test_dates[-1].date()),
            "source": "华西证券2021_网络中心度",
            "direction": "空头" if ic_mean > 0 else "多头",
            "notes": f"SCC网络中心度, LOOKBACK={LOOKBACK}, N={N_STOCKS}"
        }
        pool = pd.concat([pool, pd.DataFrame([new_row])], ignore_index=True)
        print(f"  因子池已新增: {factor_name} → {status}")
    pool.to_csv(POOL_FILE, index=False, encoding="utf-8-sig")

print(f"\n✅ SCC因子测试完成!")
