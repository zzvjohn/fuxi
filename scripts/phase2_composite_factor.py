# -*- coding: utf-8 -*-
"""
Phase 2B: IC加权组合因子 + 对比分析
基于18因子IC结果，构建加权复合因子
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy import stats

DATA_DIR = r"e:\quant\data\raw"
OUTPUT_DIR = r"e:\quant\reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 加载18因子数据
# ============================================================
print("[1/4] 加载数据...")
prices = pd.read_csv(os.path.join(DATA_DIR, "daily_prices.csv"),
                     encoding="utf-8-sig", dtype={"ts_code": str})
prices["trade_date"] = pd.to_datetime(prices["trade_date"], format="mixed")

basic = pd.read_csv(os.path.join(DATA_DIR, "daily_basic.csv"),
                    encoding="utf-8-sig", dtype={"ts_code": str})
basic["trade_date"] = pd.to_datetime(basic["trade_date"], format="mixed")

idx_data = pd.read_csv(os.path.join(DATA_DIR, "index_000300_daily.csv"),
                       encoding="utf-8-sig", dtype={"ts_code": str})
idx_data["trade_date"] = pd.to_datetime(idx_data["trade_date"], format="mixed")

fina = pd.read_csv(os.path.join(DATA_DIR, "fina_indicator_full.csv"),
                   encoding="utf-8-sig", dtype={"ts_code": str})
fina["ann_date"] = pd.to_datetime(fina["ann_date"], format="mixed")
fina["end_date"] = pd.to_datetime(fina["end_date"], format="mixed")

# ============================================================
# 重新计算所有因子（与phase2_factor_ic.py一致）
# ============================================================
print("[2/4] 计算因子...")

df = prices[["ts_code", "trade_date", "close", "pre_close", "vol"]].copy()
df = df.merge(basic[["ts_code", "trade_date", "pb", "pe_ttm", "ps_ttm",
                      "total_mv", "turnover_rate"]],
              on=["ts_code", "trade_date"], how="left")

# 规模
df["ln_cap"] = np.log(df["total_mv"]).replace([np.inf, -np.inf], np.nan)

# 估值
df["pb_inv"] = -df["pb"]
df["pe_ttm_inv"] = -df["pe_ttm"]
df["ps_ttm_inv"] = -df["ps_ttm"]

# 换手率
df["turnover_20d"] = df.groupby("ts_code")["turnover_rate"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# 收益率
df["ret_1d"] = df.groupby("ts_code")["close"].pct_change()

# 动量
for w, name in [(21, "mom_1m"), (63, "mom_3m"), (126, "mom_6m"), (252, "mom_12m")]:
    df[name] = df.groupby("ts_code")["close"].transform(lambda x: x.pct_change(w))

# 波动率
for w, name in [(21, "vol_1m"), (63, "vol_3m")]:
    df[name] = df.groupby("ts_code")["ret_1d"].transform(
        lambda x: x.rolling(w, min_periods=10).std()
    )
    df[name + "_inv"] = -df[name]

# Amihud
df["amihud_daily"] = np.abs(df["ret_1d"]) / (df["vol"] * df["close"] / 1e8).replace(0, np.nan)
df["amihud_20d"] = df.groupby("ts_code")["amihud_daily"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
df["illiq"] = -df["amihud_20d"]

# 市场收益
idx = idx_data[["trade_date", "close"]].copy()
idx["mkt_ret"] = idx["close"].pct_change()
idx = idx.dropna(subset=["mkt_ret"])
df = df.merge(idx[["trade_date", "mkt_ret"]], on="trade_date", how="left")

# 特质波动率 & 残差动量
print("  计算特质波动率 & 残差动量...")
df["idio_vol"] = np.nan
df["resid_mom"] = np.nan
codes = df["ts_code"].unique()
for i in range(0, len(codes), 500):
    batch_codes = codes[i:i+500]
    mask = df["ts_code"].isin(batch_codes)
    df_batch = df.loc[mask].copy()
    for code in df_batch["ts_code"].unique():
        idx_code = df_batch["ts_code"] == code
        df_code = df_batch.loc[idx_code].dropna(subset=["ret_1d", "mkt_ret"]).sort_values("trade_date").copy()
        if len(df_code) < 63:
            continue
        try:
            y = df_code["ret_1d"].values
            X = np.column_stack([np.ones(len(y)), df_code["mkt_ret"].values])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
            resid_series = pd.Series(resid, index=df_code.index)
            df.loc[df_code.index, "idio_vol"] = resid_series.rolling(63, min_periods=20).std()
            df.loc[df_code.index, "resid_mom"] = resid_series.rolling(63, min_periods=30).sum()
        except Exception:
            pass
    if i % 2500 == 0 and i > 0:
        print(f"    progress: {i}/{len(codes)}")

df["idio_vol_inv"] = -df["idio_vol"]

# 财务数据
fina_use = fina[["ts_code", "ann_date", "roe", "roa", "debt_to_assets", "assets_turn"]].dropna(subset=["ann_date"])
fina_use = fina_use.rename(columns={"ann_date": "trade_date"})
df = df.merge(fina_use, on=["ts_code", "trade_date"], how="left")
for col in ["roe", "roa", "debt_to_assets", "assets_turn"]:
    df[col] = df.groupby("ts_code")[col].ffill()
    df[col] = df.groupby("ts_code")[col].bfill()
df["debt_to_assets_inv"] = -df["debt_to_assets"]

# 前向收益
for h in [1, 5, 10, 21]:
    df[f"fwd_close_{h}d"] = df.groupby("ts_code")["close"].shift(-h)
    df[f"fwd_ret_{h}d"] = df[f"fwd_close_{h}d"] / df["close"] - 1

df = df[(df["close"] > 0) & (df["vol"] > 0)].copy()

# ============================================================
# 3. IC加权组合因子
# ============================================================
print("[3/4] 构建IC加权组合因子...")

# 从上一轮IC结果提取权重（|ICIR|加权，方向已由因子定义处理）
# 有效因子：直接使用原始值，因为因子方向已翻转（如 vol_1m_inv = -vol）
FACTOR_WEIGHTS = {
    "vol_1m_inv":     0.55,   # ICIR +0.55, 正向
    "pb_inv":         0.52,   # ICIR +0.52, 正向
    "idio_vol_inv":   0.49,   # ICIR +0.49, 正向
    "ps_ttm_inv":     0.41,   # ICIR +0.41, 正向
    "resid_mom":     -0.75,   # ICIR -0.75, 反转因子(高残差动量=低收益)
    "illiq":         -0.57,   # ICIR -0.57, 反转
    "mom_1m":        -0.55,   # ICIR -0.55, 反转
    "turnover_20d":  -0.51,   # ICIR -0.51, 反转
    "mom_3m":        -0.50,   # ICIR -0.50, 反转
}

# 构建组合：每个因子先rank归一化再加权
print("  计算因子rank & 加权组合...")
composite_columns = []
for fac, weight in FACTOR_WEIGHTS.items():
    if fac not in df.columns:
        print(f"    {fac}: 缺失，跳过")
        continue
    col_name = f"{fac}_rank"
    df[col_name] = df.groupby("trade_date")[fac].transform(
        lambda x: x.rank(pct=True)  # 0~1 的百分位rank
    )
    composite_columns.append(col_name)

# 加权加总（注意反转因子权重为负）
df["composite"] = 0.0
for fac, weight in FACTOR_WEIGHTS.items():
    col = f"{fac}_rank"
    if col in df.columns:
        df["composite"] += weight * df[col]

# 全部因子等权（无IC加权，用于对比）
df["composite_equal"] = 0.0
count = 0
for fac in FACTOR_WEIGHTS:
    col = f"{fac}_rank"
    if col in df.columns:
        df["composite_equal"] += np.sign(FACTOR_WEIGHTS[fac]) * df[col]
        count += 1
if count > 0:
    df["composite_equal"] /= count

# ============================================================
# 4. IC分析：组合因子 vs 单独因子
# ============================================================
print("[4/4] IC分析对比...")

# 选取6个最有效单因子 + 2个组合
FOCUS_FACTORS = {
    "vol_1m_inv":    ("波动", "低波(21日)"),
    "pb_inv":        ("估值", "低PB"),
    "idio_vol_inv":  ("波动", "特质低波"),
    "ps_ttm_inv":    ("估值", "低PS"),
    "resid_mom":     ("反转", "残差动量反转"),
    "mom_1m":        ("反转", "月动量反转"),
    "composite":     ("组合", "IC加权组合"),
    "composite_equal": ("组合", "等权组合"),
}

comparison = []

for fac, (ftype, fdesc) in FOCUS_FACTORS.items():
    if fac not in df.columns:
        continue

    df_ic = df[[fac, "fwd_ret_21d"]].dropna().copy()
    if len(df_ic) < 1000:
        continue

    ic_daily = df_ic.groupby(df["trade_date"]).apply(
        lambda g: stats.spearmanr(g[fac].rank(), g["fwd_ret_21d"].rank())[0]
        if len(g) >= 30 else np.nan
    ).dropna()

    if len(ic_daily) < 50:
        continue

    ic_mean = ic_daily.mean()
    icir = ic_mean / ic_daily.std()
    ic_pos = (ic_daily > 0).mean()

    # 分位数
    df_q = df_ic.copy()
    df_q["trade_date"] = df.loc[df_ic.index, "trade_date"]
    df_q["q"] = df_q.groupby("trade_date")[fac].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop")
    )

    q_rets = []
    for q in range(5):
        q_rets.append(df_q[df_q["q"] == q]["fwd_ret_21d"].mean() * 100)

    q5_q1 = q_rets[4] - q_rets[0] if len(q_rets) >= 5 else 0

    comparison.append({
        "factor": fac,
        "desc": fdesc,
        "type": ftype,
        "IC_mean": round(ic_mean, 6),
        "ICIR": round(icir, 4),
        "IC_pos": round(ic_pos, 4),
        "n_days": len(ic_daily),
        "Q1": round(q_rets[0], 4) if len(q_rets) >= 5 else np.nan,
        "Q5": round(q_rets[4], 4) if len(q_rets) >= 5 else np.nan,
        "Q5_Q1": round(q5_q1, 4),
    })

comp_df = pd.DataFrame(comparison).sort_values("ICIR", ascending=False)

# ============================================================
# 输出
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("\n" + "=" * 75)
print("组合因子 vs 单因子对比 (holding_period=21d)")
print("=" * 75)
print(f"{'因子':<20} {'类型':<8} {'IC_mean':>8} {'ICIR':>8} {'IC>0%':>8} {'Q5-Q1':>8}")
print("-" * 75)

for _, r in comp_df.iterrows():
    star = "NEW" if "composite" in r["factor"] else "   "
    print(f"{star} {r['factor']:<17} {r['type']:<8} {r['IC_mean']:>8.4f} {r['ICIR']:>8.4f} "
          f"{r['IC_pos']*100:>7.1f}% {r.get('Q5_Q1', 0):>8.4f}")

print("-" * 75)

# 关键对比
icir_best_single = comp_df[~comp_df["factor"].str.contains("composite")]["ICIR"].max()
icir_composite = comp_df[comp_df["factor"] == "composite"]["ICIR"].values[0] if "composite" in comp_df["factor"].values else 0

print(f"\n最佳单因子ICIR: {icir_best_single:.4f}")
print(f"IC加权组合ICIR:  {icir_composite:.4f}")
improvement = (icir_composite - icir_best_single) / abs(icir_best_single) * 100
print(f"提升: {improvement:+.0f}%")

# 保存
comp_csv = os.path.join(OUTPUT_DIR, f"composite_vs_single_{timestamp}.csv")
comp_df.to_csv(comp_csv, index=False, encoding="utf-8-sig")
print(f"\n对比结果: {comp_csv}")

print(f"\nPhase 2B 完成!")
