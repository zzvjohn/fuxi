#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AlphaPurify 接入测试 — FA v7.3 因子 (turnover_cv_20d)
=======================================================
1. 读取因子CSV + 日频价格
2. 组装长表 → AlphaPurifier + FactorAnalyzer + PureExposures
3. 收益归因拆解
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys, io, warnings

warnings.filterwarnings("ignore")
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ============================================================
# 配置
# ============================================================
BASE = Path(__file__).parent.parent
FACTOR_CSV = BASE / "research" / "factor_alchemy" / "output" / "factor_turnover_cv_20d.csv"
PRICE_FILE = BASE / "data" / "raw" / "daily_prices.csv"
OUTPUT_DIR = BASE / "output" / "ap_test"
FACTOR_NAME = "turnover_cv_20d"

START_DATE = "2021-01-01"
END_DATE = "2026-06-25"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Step 1: 加载因子 (宽→长)
# ============================================================
print("=" * 60)
print("Step 1: 加载因子数据")
print("=" * 60)

df_fac = pd.read_csv(FACTOR_CSV)
# 修复列名
if df_fac.columns[0] in ("Unnamed: 0", "", None) or str(df_fac.columns[0]).startswith("Unnamed"):
    df_fac.rename(columns={df_fac.columns[0]: "trade_date"}, inplace=True)
df_fac.columns = [c.strip().lstrip("\ufeff") for c in df_fac.columns]

# wide → long
df_fac = df_fac.melt(id_vars="trade_date", var_name="stock_code", value_name=FACTOR_NAME)
df_fac["trade_date"] = pd.to_datetime(df_fac["trade_date"])
df_fac = df_fac.dropna(subset=[FACTOR_NAME])

# 过滤日期范围
df_fac = df_fac[df_fac["trade_date"].between(START_DATE, END_DATE)]

print(f"  {FACTOR_NAME}: {len(df_fac):,} 条, {df_fac['stock_code'].nunique()} 只股票, "
      f"{df_fac['trade_date'].nunique()} 周")

# ============================================================
# Step 2: 价格数据 → 周频前向收益 + 风格因子
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 价格数据 & 风格因子代理")
print("=" * 60)

prices = pd.read_csv(PRICE_FILE, encoding="utf-8")
# 日期可能是 20260525 或 2026-05-25 格式
prices["trade_date"] = pd.to_datetime(prices["trade_date"], format="mixed")
prices = prices[prices["trade_date"].between(START_DATE, END_DATE)]
prices = prices.sort_values(["ts_code", "trade_date"])
print(f"  日频价格: {len(prices):,} 条")

# 日收益
prices["daily_ret"] = prices.groupby("ts_code")["close"].pct_change()

# → 周频 (取每周五/最后一天)
prices["week"] = prices["trade_date"].dt.to_period("W").dt.end_time.dt.normalize()
weekly = prices.groupby(["ts_code", "week"], observed=True).agg(
    close=("close", "last"),
    vol=("vol", "sum"),
    amount=("amount", "sum"),
).reset_index()
weekly = weekly.rename(columns={"week": "trade_date", "ts_code": "stock_code"})

# 前向收益 (1周)
weekly = weekly.sort_values(["stock_code", "trade_date"])
weekly["fwd_ret"] = weekly.groupby("stock_code")["close"].pct_change().shift(-1)

# ===== 风格代理因子 =====
# momentum_12_1: 52周动量
print("  计算 momentum_12_1 ...")
weekly["momentum_12_1"] = weekly.groupby("stock_code")["close"].transform(
    lambda x: x.pct_change(52)
)

# vol_60: 12周波动率
print("  计算 vol_60 ...")
weekly["daily_ret_w"] = weekly.groupby("stock_code")["close"].pct_change()
weekly["vol_60"] = weekly.groupby("stock_code")["daily_ret_w"].transform(
    lambda x: x.rolling(12, min_periods=6).std()
)

# log_size: 市值代理
weekly["log_size"] = np.log(
    weekly["close"].abs() * weekly["amount"] / weekly["amount"].replace(0, np.nan)
    + 1e-8  # avoid log(0)
)
weekly["log_size"] = weekly["log_size"].replace([np.inf, -np.inf], np.nan)

weekly = weekly.drop(columns=["daily_ret_w"])
weekly = weekly.dropna(subset=["fwd_ret", "close"])
print(f"  周频: {len(weekly):,} 条, {weekly['stock_code'].nunique()} 只股票")

# ============================================================
# Step 3: 合并
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 合并因子 + 价格")
print("=" * 60)

merged = df_fac.merge(
    weekly[["stock_code", "trade_date", "close", "fwd_ret",
            "momentum_12_1", "vol_60", "log_size"]],
    on=["trade_date", "stock_code"], how="inner"
)
print(f"  合并: {len(merged):,} 条, {merged['stock_code'].nunique()} 只股票, "
      f"{merged['trade_date'].nunique()} 周")

# ============================================================
# Step 4: AlphaPurifier
# ============================================================
print("\n" + "=" * 60)
print("Step 4: AlphaPurifier 预处理")
print("=" * 60)

from alphapurify import AlphaPurifier, PureExposures

# A组: 仅MAD + Zscore (当前做法)
print("  [A组] MAD → Zscore (当前做法)")
purifier_a = AlphaPurifier(merged, factor_name=FACTOR_NAME,
                           trade_date_col="trade_date", symbol_col="stock_code")
df_a = purifier_a.winsorize("mad").standardize("zscore").to_result()

# B组: MAD + Ridge中性化(市值+波动) + Zscore
print("  [B组] MAD → Ridge(市值+波动) → Zscore")
purifier_b = AlphaPurifier(merged, factor_name=FACTOR_NAME,
                           trade_date_col="trade_date", symbol_col="stock_code")
df_b = purifier_b.winsorize("mad").neutralize(
    "ridge", ["log_size", "vol_60"]
).standardize("zscore").to_result()

style_cols = ["momentum_12_1", "vol_60", "log_size"]

print(f"  A组(无中性化): {len(df_a):,} 条")
print(f"  B组(Ridge中性化): {len(df_b):,} 条")

# ============================================================
# Step 5: PureExposures — 对比A/B组
# ============================================================
print("\n" + "=" * 60)
print("Step 5: PureExposures 收益归因对比")
print("=" * 60)

results = {}
for label, df_data in [("A组-无中性化", df_a), ("B组-Ridge中性化", df_b)]:
    print(f"\n  >>> {label}")

    pure = PureExposures(
        base_df=df_data,
        trade_date_col="trade_date",
        symbol_col="stock_code",
        price_col="close",
        factor_name=FACTOR_NAME,
        exposure_cols=style_cols,
    )
    pure.run()

    result = pure.result_df
    corr = pure.corr_df
    results[label] = (result, corr)

    if result is not None and len(result) > 0:
        # result is polars DataFrame
        result_pd = result.to_pandas() if hasattr(result, 'to_pandas') else result
        cum_cols = [c for c in result.columns if c.endswith("_cum_ret")]
        alpha_col = [c for c in result.columns if "alpha" in c.lower() or "Alpha" in c]
        port_col = f"{FACTOR_NAME}_cum_ret"

        print(f"    {'─' * 45}")
        for col in cum_cols:
            vals = result_pd[col].dropna()
            if len(vals) > 0:
                v = vals.iloc[-1]
                l = col.replace("_cum_ret", "").replace("_", " ")
                print(f"    {l:20s}: {v:+.2%}")
        for col in alpha_col:
            vals = result_pd[col].dropna()
            if len(vals) > 0:
                v = vals.iloc[-1]
                print(f"    {'Alpha (纯选股)':20s}: {v:+.2%}")
        if port_col in result_pd.columns:
            vals = result_pd[port_col].dropna()
            if len(vals) > 0:
                v = vals.iloc[-1]
                print(f"    {'─' * 45}")
                print(f"    {'组合总收益':20s}: {v:+.2%}")

    # 因子-风格相关性
    if corr is not None and len(corr) > 0:
        corr_pd = corr.to_pandas() if hasattr(corr, 'to_pandas') else corr
        print(f"\n    因子-风格相关性 (时序均值):")
        for c in style_cols:
            if c in corr_pd.columns:
                mc = corr_pd[c].mean()
                print(f"      corr({FACTOR_NAME}, {c:15s}): {mc:+.4f}")

# ============================================================
# Step 6: 归因占比对比
# ============================================================
print("\n" + "=" * 60)
print("Step 6: 归因占比汇总")
print("=" * 60)

for label, (res, _) in results.items():
    if res is None or len(res) == 0:
        continue
    res_pd = res.to_pandas() if hasattr(res, 'to_pandas') else res

    port_col = f"{FACTOR_NAME}_cum_ret"
    if port_col not in res_pd.columns:
        continue

    total = res_pd[port_col].dropna().iloc[-1]
    if abs(total) < 0.001:
        print(f"  {label}: 收益接近0, 无法计算占比")
        continue

    print(f"\n  【{label}】 收益来源占比 (总收益={total:+.2%}):")
    cum_cols = [c for c in res_pd.columns if c.endswith("_cum_ret") and c != port_col]
    alpha_col = [c for c in res_pd.columns if "alpha" in c.lower() or "Alpha" in c]

    # 风格贡献
    style_sum = 0
    for col in cum_cols:
        vals = res_pd[col].dropna()
        v = vals.iloc[-1] if len(vals) > 0 else 0
        l = col.replace("_cum_ret", "").replace("_", " ")
        pct = v / total * 100
        style_sum += v
        print(f"    {l:20s}: {v:+.2%}  ({pct:+.0f}%)")

    # Alpha
    alpha_sum = 0
    for col in alpha_col:
        vals = res_pd[col].dropna()
        v = vals.iloc[-1] if len(vals) > 0 else 0
        pct = v / total * 100
        alpha_sum += v
        print(f"    {'Alpha (纯选股能力)':20s}: {v:+.2%}  ({pct:+.0f}%)")

    # 残差 (总收益 - style - alpha)
    residual = total - style_sum - alpha_sum
    if abs(residual) > 0.0001:
        print(f"    {'其他/残差':20s}: {residual:+.2%}  ({residual/total*100:+.0f}%)")

    print(f"    {'─' * 45}")
    print(f"    {'合计':20s}: {style_sum + alpha_sum:+.2%}")

print(f"\n✅ 完成!")
