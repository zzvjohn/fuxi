# -*- coding: utf-8 -*-
"""
Phase 2: 单因子IC检验 + 分位数分组收益 + 行业中性分析
数据源: daily_prices + daily_basic
因子池: 估值、动量、波动、规模、换手率
"""
import pandas as pd
import numpy as np
import os, sys
from datetime import datetime
from scipy import stats

DATA_DIR = r"e:\quant\data\raw"
OUTPUT_DIR = r"e:\quant\reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("Phase 2: 单因子系统检验")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)


# ============================================================
# 1. 数据加载
# ============================================================
print("\n[1/5] 加载数据...")

print("  加载 daily_prices ...")
prices = pd.read_csv(os.path.join(DATA_DIR, "daily_prices.csv"),
                     encoding="utf-8-sig", dtype={"ts_code": str})
prices["trade_date"] = pd.to_datetime(prices["trade_date"], format="mixed")
prices = prices.sort_values(["ts_code", "trade_date"])
print(f"    {len(prices):,} 行 × {prices['ts_code'].nunique()} 只股票")

print("  加载 daily_basic ...")
basic = pd.read_csv(os.path.join(DATA_DIR, "daily_basic.csv"),
                    encoding="utf-8-sig", dtype={"ts_code": str})
basic["trade_date"] = pd.to_datetime(basic["trade_date"], format="mixed")
basic = basic.sort_values(["ts_code", "trade_date"])
print(f"    {len(basic):,} 行 × {basic['ts_code'].nunique()} 只股票")

print("  加载 stock_list ...")
stocks = pd.read_csv(os.path.join(DATA_DIR, "stock_list.csv"),
                     encoding="utf-8-sig", dtype=str)

# 2. 因子计算
print("\n[2/5] 计算因子...")

# 合并价格和估值数据
factor_df = prices[["ts_code", "trade_date", "close", "pre_close", "vol"]].copy()
factor_df = factor_df.merge(
    basic[["ts_code", "trade_date", "pb", "pe_ttm", "ps_ttm",
           "total_mv", "turnover_rate"]],
    on=["ts_code", "trade_date"], how="left"
)

# --- 规模因子 ---
factor_df["ln_cap"] = np.log(factor_df["total_mv"]).replace([np.inf, -np.inf], np.nan)

# --- 估值因子 ---
factor_df["pb_inv"] = -factor_df["pb"]  # PB取负（低PB=高估值因子值）
factor_df["pe_ttm_inv"] = -factor_df["pe_ttm"]
factor_df["ps_ttm_inv"] = -factor_df["ps_ttm"]

# --- 换手率因子 ---
factor_df["turnover_20d"] = factor_df.groupby("ts_code")["turnover_rate"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# --- 日收益率 ---
factor_df["ret_1d"] = factor_df.groupby("ts_code")["close"].pct_change()

# --- 动量因子 ---
for w, name in [(21, "mom_1m"), (63, "mom_3m"), (126, "mom_6m"), (252, "mom_12m")]:
    factor_df[name] = factor_df.groupby("ts_code")["close"].transform(
        lambda x: x.pct_change(w)
    )

# --- 波动率因子 ---
for w, name in [(21, "vol_1m"), (63, "vol_3m")]:
    factor_df[name] = factor_df.groupby("ts_code")["ret_1d"].transform(
        lambda x: x.rolling(w, min_periods=10).std()
    )
    factor_df[name + "_inv"] = -factor_df[name]  # 低波动高因子值

# --- Amihud 非流动性 ---
factor_df["amihud_daily"] = np.abs(factor_df["ret_1d"]) / (
    factor_df["vol"] * factor_df["close"] / 1e8
).replace(0, np.nan)
factor_df["amihud_20d"] = factor_df.groupby("ts_code")["amihud_daily"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
factor_df["illiq"] = -factor_df["amihud_20d"]  # 负非流动性=高流动性

# --- 特质波动率 & 残差动量 ---
print("  加载沪深300指数 (index_000300_daily.csv) ...")
idx_data = pd.read_csv(os.path.join(DATA_DIR, "index_000300_daily.csv"),
                       encoding="utf-8-sig", dtype={"ts_code": str})
idx_data["trade_date"] = pd.to_datetime(idx_data["trade_date"], format="mixed")
idx = idx_data[["trade_date", "close"]].copy()
idx["mkt_ret"] = idx["close"].pct_change()
idx = idx.rename(columns={"close": "idx_close"})
idx = idx.dropna(subset=["mkt_ret"])

factor_df = factor_df.merge(idx[["trade_date", "mkt_ret"]], on="trade_date", how="left")

print("  计算特质波动率 & 残差动量 (向量化) ...")
codes = factor_df["ts_code"].unique()
batch_size = 500

factor_df["idio_vol"] = np.nan
factor_df["resid_mom"] = np.nan

for i in range(0, len(codes), batch_size):
    batch_codes = codes[i:i + batch_size]
    mask = factor_df["ts_code"].isin(batch_codes)
    df_batch = factor_df.loc[mask].copy()

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

            # 特质波动率: 63日残差标准差
            resid_series = pd.Series(resid, index=df_code.index)
            idio_vol_series = resid_series.rolling(63, min_periods=20).std()
            factor_df.loc[df_code.index, "idio_vol"] = idio_vol_series

            # 残差动量: 63日残差累计
            resid_mom_series = resid_series.rolling(63, min_periods=30).sum()
            factor_df.loc[df_code.index, "resid_mom"] = resid_mom_series
        except Exception:
            pass

    if (i // batch_size) % 5 == 0 and i > 0:
        print(f"    progress: {i}/{len(codes)} stocks")

factor_df["idio_vol_inv"] = -factor_df["idio_vol"]

# --- 财务质量因子 (ROE, ROA, 资产负债率, 资产周转率) ---
print("  加载财务指标 (fina_indicator_full.csv) ...")
fina = pd.read_csv(os.path.join(DATA_DIR, "fina_indicator_full.csv"),
                   encoding="utf-8-sig", dtype={"ts_code": str})
fina["ann_date"] = pd.to_datetime(fina["ann_date"], format="mixed")
fina["end_date"] = pd.to_datetime(fina["end_date"], format="mixed")
fina = fina.sort_values(["ts_code", "ann_date"])

# 使用ann_date做点-in-time：公告日之后可用
# 按ts_code合并：将最近一个公告日的财务数据映射到每个交易日
print("  映射财务数据到交易日 (ann_date点-in-time) ...")
fina_use = fina[["ts_code", "ann_date", "roe", "roa", "debt_to_assets", "assets_turn"]].dropna(subset=["ann_date"])
fina_use = fina_use.rename(columns={"ann_date": "trade_date"})

# 简单的点-in-time：取最近公告日的数据
factor_df = factor_df.merge(fina_use, on=["ts_code", "trade_date"], how="left")

# forward fill: 公告日后一直使用直到下一个公告
factor_df["roe"] = factor_df.groupby("ts_code")["roe"].ffill()
factor_df["roa"] = factor_df.groupby("ts_code")["roa"].ffill()
factor_df["debt_to_assets"] = factor_df.groupby("ts_code")["debt_to_assets"].ffill()
factor_df["assets_turn"] = factor_df.groupby("ts_code")["assets_turn"].ffill()

# 回填最初的NaN（没有历史公告的）
factor_df["roe"] = factor_df.groupby("ts_code")["roe"].bfill()
factor_df["roa"] = factor_df.groupby("ts_code")["roa"].bfill()
factor_df["debt_to_assets"] = factor_df.groupby("ts_code")["debt_to_assets"].bfill()
factor_df["assets_turn"] = factor_df.groupby("ts_code")["assets_turn"].bfill()

# 低负债率（取负）
factor_df["debt_to_assets_inv"] = -factor_df["debt_to_assets"]

# --- 因子列表 ---
FACTORS = {
    "ln_cap":          ("规模",     "总市值对数"),
    "pb_inv":          ("估值",     "PB倒数"),
    "pe_ttm_inv":      ("估值",     "PE_TTM倒数"),
    "ps_ttm_inv":      ("估值",     "PS_TTM倒数"),
    "mom_1m":          ("动量",     "月动量(21日)"),
    "mom_3m":          ("动量",     "季动量(63日)"),
    "mom_6m":          ("动量",     "半年动量(126日)"),
    "mom_12m":         ("动量",     "年动量(252日)"),
    "resid_mom":       ("动量",     "残差动量(63日)"),
    "vol_1m_inv":      ("波动",     "低波(21日)"),
    "vol_3m_inv":      ("波动",     "低波(63日)"),
    "idio_vol_inv":    ("波动",     "特质低波"),
    "turnover_20d":    ("换手",     "20日均换手率"),
    "illiq":           ("流动性",   "Amihud流动性(负)"),
    "roe":             ("质量",     "ROE"),
    "roa":             ("质量",     "ROA"),
    "assets_turn":     ("质量",     "资产周转率"),
    "debt_to_assets_inv":  ("质量",     "低负债率"),
}

print(f"  因子总数: {len(FACTORS)}")

# ============================================================
# 3. 计算前向收益
# ============================================================
print("\n[3/5] 计算前向收益...")
HOLDING_PERIODS = [1, 5, 10, 21]  # 日度、周度、双周、月度

factor_df["fwd_close"] = factor_df.groupby("ts_code")["close"].shift(-1)
for h in HOLDING_PERIODS:
    # 未来h日收益
    factor_df[f"fwd_close_{h}d"] = factor_df.groupby("ts_code")["close"].shift(-h)
    factor_df[f"fwd_ret_{h}d"] = (
        factor_df[f"fwd_close_{h}d"] / factor_df["close"] - 1
    )

# 过滤：价格为零、停牌（成交量0）、前视无数据
factor_df = factor_df[
    (factor_df["close"] > 0) &
    (factor_df["vol"] > 0)
].copy()

# ============================================================
# 4. IC 分析
# ============================================================
print("\n[4/5] IC分析...")

ic_results = []
quantile_results = []

for factor_name, (factor_type, factor_desc) in FACTORS.items():
    if factor_name not in factor_df.columns:
        continue

    print(f"  分析: {factor_name} ({factor_desc})")

    for hp in HOLDING_PERIODS:
        fwd_ret_col = f"fwd_ret_{hp}d"

        # 准备数据：factor值 + forward return
        df_ic = factor_df[[factor_name, fwd_ret_col, "trade_date"]].dropna().copy()

        if len(df_ic) < 1000:
            continue

        # Cross-sectional rank IC by day
        ic_daily = df_ic.groupby("trade_date").apply(
            lambda g: stats.spearmanr(
                g[factor_name].rank(), g[fwd_ret_col].rank()
            )[0] if len(g) >= 30 else np.nan
        ).dropna()

        if len(ic_daily) < 50:
            continue

        ic_mean = ic_daily.mean()
        ic_std = ic_daily.std()
        icir = ic_mean / ic_std if ic_std > 0 else 0
        ic_pos_ratio = (ic_daily > 0).mean()
        ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_daily)))

        ic_results.append({
            "factor": factor_name,
            "factor_desc": factor_desc,
            "factor_type": factor_type,
            "holding_period": hp,
            "IC_mean": round(ic_mean, 6),
            "IC_std": round(ic_std, 4),
            "ICIR": round(icir, 4),
            "IC_pos_ratio": round(ic_pos_ratio, 4),
            "IC_tstat": round(ic_tstat, 2),
            "n_days": len(ic_daily),
            "n_obs": len(df_ic),
        })

        # 分位数分组收益 (仅monthly)
        if hp == 21:
            df_q = df_ic.dropna(subset=[factor_name, fwd_ret_col]).copy()
            df_q["q"] = df_q.groupby("trade_date")[factor_name].transform(
                lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop")
            )
            for q in range(5):
                q_ret = df_q[df_q["q"] == q][fwd_ret_col].mean()
                quantile_results.append({
                    "factor": factor_name,
                    "quantile": q + 1,
                    "avg_return": round(q_ret * 100, 4),
                })

ic_df = pd.DataFrame(ic_results)
q_df = pd.DataFrame(quantile_results)

print(f"\n  IC分析完成: {len(ic_df)} 条结果")

# ============================================================
# 5. 汇总 & 排名
# ============================================================
print("\n[5/5] 生成报告...")

# 按 21日 ICIR 排名
ic_monthly = ic_df[ic_df["holding_period"] == 21].sort_values("ICIR", ascending=False)

# 分位数单调性检验
mono_check = []
for fac in q_df["factor"].unique():
    qf = q_df[q_df["factor"] == fac].sort_values("quantile")
    if len(qf) == 5:
        q1_ret = qf.iloc[0]["avg_return"]
        q5_ret = qf.iloc[4]["avg_return"]
        spread = q5_ret - q1_ret
        # 单调性：检查是否 Q5>Q1
        mono_check.append({
            "factor": fac,
            "Q1_return": q1_ret,
            "Q5_return": q5_ret,
            "Q5_Q1_spread": round(spread, 4),
            "monotonic": q5_ret > q1_ret,
        })
mono_df = pd.DataFrame(mono_check)

# 整合排名
ranking = ic_monthly.merge(mono_df, on="factor", how="left")
ranking["star"] = ranking.apply(
    lambda r: "★★★" if abs(r["ICIR"]) > 1.5 else ("★★" if abs(r["ICIR"]) > 0.8 else "★"),
    axis=1
)
ranking = ranking.sort_values("ICIR", ascending=False)

# ============================================================
# 输出结果
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("\n" + "=" * 70)
print("因子ICIR排名 (holding_period=21d)")
print("=" * 70)
print(f"{'因子':<20} {'类型':<8} {'IC_mean':>8} {'ICIR':>8} {'Q1-Q5':>8} {'评级':>6}")
print("-" * 70)

for _, r in ranking.iterrows():
    sp = r.get("Q5_Q1_spread", 0)
    print(f"{r['factor']:<20} {r['factor_type']:<8} {r['IC_mean']:>8.4f} {r['ICIR']:>8.4f} {sp:>8.4f} {r['star']:>6}")

print("-" * 70)
print(f"\n有效因子 (|ICIR| > 0.5): {len(ranking[abs(ranking['ICIR']) > 0.5])}")
print(f"强有效因子 (|ICIR| > 1.0): {len(ranking[abs(ranking['ICIR']) > 1.0])}")

# 保存CSV
ic_csv = os.path.join(OUTPUT_DIR, f"factor_ic_report_{timestamp}.csv")
ic_df.to_csv(ic_csv, index=False, encoding="utf-8-sig")
print(f"\nIC详细数据: {ic_csv}")

rank_csv = os.path.join(OUTPUT_DIR, f"factor_ranking_{timestamp}.csv")
ranking.to_csv(rank_csv, index=False, encoding="utf-8-sig")
print(f"因子排名: {rank_csv}")

q_csv = os.path.join(OUTPUT_DIR, f"factor_quantile_{timestamp}.csv")
q_df.to_csv(q_csv, index=False, encoding="utf-8-sig")
print(f"分位数收益: {q_csv}")

# 生成文本报告
report_lines = [
    f"# Phase 2 单因子IC检验报告",
    f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"数据范围: {factor_df['trade_date'].min().date()} ~ {factor_df['trade_date'].max().date()}",
    f"股票总数: {factor_df['ts_code'].nunique()}",
    f"总观测数: {len(factor_df):,}",
    f"因子数量: {len(FACTORS)}",
    "",
    "## 因子排名 (holding_period=21d, 按ICIR降序)",
    "",
    ranking.to_markdown(index=False),
    "",
    "## 评级标准",
    "★★★ ICIR > 1.5 强有效因子",
    "★★  ICIR > 0.8 中等有效因子",
    "★   ICIR > 0   弱有效因子",
    "",
    "## 分位数分组收益 (Q1最便宜/最弱 → Q5最贵/最强)",
    "",
    "每组年化收益 = 月收益 × 12",
    "",
]

for fac in ranking["factor"].head(10):
    qf = q_df[q_df["factor"] == fac].sort_values("quantile")
    if len(qf) < 2:
        continue
    report_lines.append(f"### {FACTORS.get(fac, [fac, ''])[1]} ({fac})")
    report_lines.append("")
    for _, rq in qf.iterrows():
        report_lines.append(f"  Q{rq['quantile']:.0f}: {rq['avg_return']:+.4f}%")
    report_lines.append("")

report_path = os.path.join(OUTPUT_DIR, f"factor_ic_report_{timestamp}.md")
with open(report_path, "w", encoding="utf-8-sig") as f:
    f.write("\n".join(report_lines))
print(f"\n完整报告: {report_path}")

print(f"\nPhase 2 完成! 耗时: {datetime.now().strftime('%H:%M:%S')}")
