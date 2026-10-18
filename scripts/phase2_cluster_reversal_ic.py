# -*- coding: utf-8 -*-
"""
#15 聚类增强反转策略 — IC/ICIR 检验
方法：K-means月度聚类 → 聚类内反转排名 → 截面IC分析
对比：行业分类反转 vs 聚类反转 vs 标准全市场反转
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = r"e:\quant\data\raw"
OUTPUT_DIR = r"e:\quant\reports\factor_discovery"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print("=" * 70)
print("#15 聚类增强反转策略 — IC/ICIR 检验")
print(f"开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ============================================================
# 1. 数据加载
# ============================================================
print("\n[1/5] 加载数据...")

prices = pd.read_csv(os.path.join(DATA_DIR, "daily_prices.csv"),
                     encoding="utf-8-sig", dtype={"ts_code": str})
prices["trade_date"] = pd.to_datetime(prices["trade_date"], format="mixed")
prices = prices.sort_values(["ts_code", "trade_date"])
print(f"  日线: {len(prices):,} 行 × {prices['ts_code'].nunique()} 只股票")

# 加载行业分类
print("  加载行业分类...")
stocks = pd.read_csv(os.path.join(DATA_DIR, "stock_list.csv"),
                     encoding="utf-8-sig", dtype=str)
stocks = stocks[["ts_code", "industry"]].dropna(subset=["industry"])
# 行业代码映射到数字
industry_map = {ind: i for i, ind in enumerate(stocks["industry"].unique())}
stocks["industry_id"] = stocks["industry"].map(industry_map)
print(f"  行业数: {len(industry_map)}")

# ============================================================
# 2. 因子计算
# ============================================================
print("\n[2/5] 计算技术特征因子...")

df = prices[["ts_code", "trade_date", "close", "pre_close", "vol", "amount"]].copy()

# 日收益
df["ret_1d"] = df.groupby("ts_code")["close"].pct_change()

# 动量因子 (多窗口)
for w in [5, 10, 21, 63, 126]:
    df[f"mom_{w}d"] = df.groupby("ts_code")["close"].transform(
        lambda x: x.pct_change(w)
    )

# 波动率 (21日)
df["vol_21d"] = df.groupby("ts_code")["ret_1d"].transform(
    lambda x: x.rolling(21, min_periods=10).std()
)

# 换手率估算 (成交量/自由流通股 → 用 vol 作为代理)
# 20日均量变化
df["vol_ma20"] = df.groupby("ts_code")["vol"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
df["turnover_proxy"] = df["vol"] / df["vol_ma20"].replace(0, np.nan)

# Amihud 非流动性
df["amihud"] = np.abs(df["ret_1d"]) / (
    df["vol"] * df["close"] / 1e8
).replace(0, np.nan)
df["illiq_20d"] = df.groupby("ts_code")["amihud"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# 市值代理 (成交量×价格 / 1e8)
df["mv_proxy"] = df["vol"] * df["close"] / 1e8
df["ln_mv"] = np.log(df["mv_proxy"].replace(0, np.nan))

# 过滤无效数据
df = df[
    (df["close"] > 0) &
    (df["vol"] > 0)
].copy()

print(f"  有效观测: {len(df):,}")

# ============================================================
# 2.5 构建月度截面特征 → K-means聚类
# ============================================================
print("\n[2.5/5] K-means 月度聚类...")

# 聚类特征 (与论文一致：技术特征为主)
CLUSTER_FEATURES = [
    "mom_5d", "mom_10d", "mom_21d", "mom_63d",  # 多窗口动量
    "vol_21d",                                     # 波动率
    "ln_mv",                                       # 规模对数
    "illiq_20d",                                   # 非流动性
    "turnover_proxy",                              # 换手率代理
]

# 获取月末交易日
df["year_month"] = df["trade_date"].dt.to_period("M")
month_ends = df.groupby("year_month")["trade_date"].max().reset_index()
month_ends = sorted(month_ends["trade_date"].tolist())
print(f"  月末截面数: {len(month_ends)}")

# K-means 聚类数
K_VALUES = [10, 15, 20]

# 存储聚类标签
cluster_labels = {}  # {date: {k: pd.Series(ts_code -> label)}}

for date in month_ends:
    cross = df[df["trade_date"] == date].copy()
    if len(cross) < 500:
        continue
    
    # 提取特征
    feats = cross[CLUSTER_FEATURES].copy()
    feats = feats.fillna(feats.median())
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 标准化
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    
    cluster_labels[date] = {}
    for k in K_VALUES:
        if k > len(feats_scaled) // 5:  # 每聚类至少5只
            k_use = max(2, len(feats_scaled) // 5)
            if k_use == k:
                k_use = max(2, k // 2)
        else:
            k_use = k
        
        km = KMeans(n_clusters=k_use, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(feats_scaled)
        
        # 存为Series
        cluster_labels[date][k] = pd.Series(
            labels, index=cross["ts_code"].values
        )

print(f"  完成聚类: {len(cluster_labels)} 个月末截面")

# ============================================================
# 3. 构建三种反转因子 + 前向收益
# ============================================================
print("\n[3/5] 构建反转因子与前向收益...")

# 反转信号（论文核心：聚类内反转排名）
# 使用10日反转 (mom_10d 取负)
df["reversal_10d"] = -df["mom_10d"]  # 正的reversal = 跌得多的

# 合并行业ID
df = df.merge(stocks[["ts_code", "industry_id"]], on="ts_code", how="left")

# 因子函数：在某个分组内做 rank
def within_group_rank(df_sub, signal_col):
    """在分组内做rank，返回0~1之间"""
    valid = df_sub[signal_col].notna()
    if valid.sum() < 3:
        return pd.Series(np.nan, index=df_sub.index)
    ranked = df_sub.loc[valid, signal_col].rank(pct=True)
    result = pd.Series(np.nan, index=df_sub.index)
    result.loc[valid] = ranked
    return result

# 三种反转因子
# A: 全市场反转 (baseline)
# B: 行业分类反转 (当前策略)
# C: K-means聚类反转 (新方案)

factor_records = []  # 存储所有因子计算的日截面数据

# 先构建月度聚类标签的连续映射（月度内保持聚类不变）
# 对于每个交易日，使用当月（或最近月份）的聚类标签
print("  构建聚类标签映射（月度内保持聚类不变）...")
dates_sorted = sorted(df["trade_date"].unique())
month_end_dates = sorted(cluster_labels.keys())

# 为每个交易日分配所属的月末日
date_to_month_end = {}
for date in dates_sorted:
    # 找到不超过当前日期的最近月末
    candidates = [d for d in month_end_dates if d <= date]
    if candidates:
        date_to_month_end[date] = candidates[-1]
    else:
        date_to_month_end[date] = None
processed = 0
skipped = 0

for date in dates_sorted:
    day_data = df[df["trade_date"] == date].copy()
    n_stocks = len(day_data)
    
    if n_stocks < 100:
        skipped += 1
        continue
    
    # --- 因子A: 全市场反转排名 ---
    day_data["f_rev_market"] = day_data["reversal_10d"].rank(pct=True)
    
    # --- 因子B: 行业分类反转 ---
    day_data["f_rev_industry"] = day_data.groupby("industry_id")["reversal_10d"].transform(
        lambda x: x.rank(pct=True) if x.notna().sum() >= 3 else np.nan
    )
    
    # --- 因子C: K-means聚类反转 ---
    # 获取当月聚类标签
    month_end = date_to_month_end.get(date)
    
    for k in K_VALUES:
        col_name = f"f_rev_cluster_k{k}"
        day_data[col_name] = np.nan
        
        # 使用当月聚类标签
        if month_end is not None and k in cluster_labels.get(month_end, {}):
            labels = cluster_labels[month_end][k]
            # 将聚类标签合并到日数据
            day_data = day_data.join(
                labels.rename("_cluster_label"), on="ts_code", how="left"
            )
            # 在聚类内做rank
            day_data[col_name] = day_data.groupby("_cluster_label")["reversal_10d"].transform(
                lambda x: x.rank(pct=True) if x.notna().sum() >= 3 else np.nan
            )
            day_data = day_data.drop(columns=["_cluster_label"])
        
        # 如果没有聚类标签（新股或月初），回退到全市场排名
        day_data[col_name] = day_data[col_name].fillna(day_data["f_rev_market"])
    
    # --- 前向收益 (1d, 5d, 10d, 21d) ---
    # 对于每个持有期，从前向价格数据获取
    # 这里使用已经预先计算好的价格数据
    
    # 存储需要的列
    cols_to_keep = [
        "ts_code", "trade_date",
        "f_rev_market", "f_rev_industry",
    ] + [f"f_rev_cluster_k{k}" for k in K_VALUES]
    
    factor_records.append(day_data[cols_to_keep].copy())
    processed += 1

factor_snapshots = pd.concat(factor_records, ignore_index=True)
print(f"  处理截面: {processed} (跳过{skipped}个不足100只的截面)")

# 计算前向收益
print("  计算前向收益...")
df_price = prices[["ts_code", "trade_date", "close"]].copy()
df_price["trade_date"] = pd.to_datetime(df_price["trade_date"], format="mixed")
df_price = df_price.sort_values(["ts_code", "trade_date"])

# 为每个截面日计算不同持有期的前向收益
HOLDING_PERIODS = [1, 5, 10, 21]

for hp in HOLDING_PERIODS:
    col = f"fwd_ret_{hp}d"
    close_fwd_col = f"close_fwd_{hp}d"
    df_price[close_fwd_col] = df_price.groupby("ts_code")["close"].shift(-hp)
    df_price[col] = df_price[close_fwd_col] / df_price["close"] - 1

# 合并前向收益到因子快照
factor_snapshots = factor_snapshots.merge(
    df_price[["ts_code", "trade_date"] + [f"fwd_ret_{hp}d" for hp in HOLDING_PERIODS]],
    on=["ts_code", "trade_date"], how="left"
)

# 过滤无前向收益的
for hp in HOLDING_PERIODS:
    factor_snapshots = factor_snapshots[factor_snapshots[f"fwd_ret_{hp}d"].notna()]

print(f"  合并后有效观测: {len(factor_snapshots):,}")

# ============================================================
# 4. IC分析
# ============================================================
print("\n[4/5] IC 分析...")

FACTOR_NAMES = {
    "f_rev_market": "全市场反转",
    "f_rev_industry": "行业分类反转",
}
for k in K_VALUES:
    FACTOR_NAMES[f"f_rev_cluster_k{k}"] = f"K-means聚类反转(k={k})"

ic_results = []

for factor_col, factor_name in FACTOR_NAMES.items():
    if factor_col not in factor_snapshots.columns:
        continue
    
    for hp in HOLDING_PERIODS:
        fwd_col = f"fwd_ret_{hp}d"
        
        df_ic = factor_snapshots[[factor_col, fwd_col, "trade_date"]].dropna().copy()
        
        if len(df_ic) < 10000:
            continue
        
        # 日度截面 Rank IC
        ic_daily = df_ic.groupby("trade_date").apply(
            lambda g: stats.spearmanr(
                g[factor_col].rank(), g[fwd_col].rank()
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
            "factor_name": factor_name,
            "factor_col": factor_col,
            "holding_period": hp,
            "IC_mean": round(ic_mean, 6),
            "IC_std": round(ic_std, 4),
            "ICIR": round(icir, 4),
            "IC_pos_ratio": round(ic_pos_ratio, 4),
            "IC_tstat": round(ic_tstat, 2),
            "n_days": len(ic_daily),
            "n_obs": len(df_ic),
        })

ic_df = pd.DataFrame(ic_results)

print(f"  IC结果: {len(ic_df)} 条")

# ============================================================
# 5. 结果输出
# ============================================================
print("\n[5/5] 输出结果...")

print("\n" + "=" * 80)
print("反转因子 ICIR 对比 (按 holding_period=21d 排序)")
print("=" * 80)

for hp in HOLDING_PERIODS:
    ic_hp = ic_df[ic_df["holding_period"] == hp].sort_values("ICIR", ascending=False)
    print(f"\n--- Holding Period: {hp}d ---")
    print(f"{'因子':<28} {'IC_mean':>8} {'IC_std':>8} {'ICIR':>8} {'IC>0%':>8} {'t-stat':>8}")
    print("-" * 72)
    for _, r in ic_hp.iterrows():
        print(f"{r['factor_name']:<28} {r['IC_mean']:>8.4f} {r['IC_std']:>8.4f} {r['ICIR']:>8.4f} {r['IC_pos_ratio']:>8.2%} {r['IC_tstat']:>8.2f}")

# 排序对比表
print("\n" + "=" * 80)
print("月度(21d) ICIR 排名")
print("=" * 80)
ic_21d = ic_df[ic_df["holding_period"] == 21].sort_values("ICIR", ascending=False)
ic_21d = ic_21d.reset_index(drop=True)

# 计算与行业分类反转的ICIR差值
industry_icir = ic_21d[ic_21d["factor_col"] == "f_rev_industry"]["ICIR"].values
if len(industry_icir) > 0:
    industry_icir = industry_icir[0]
    ic_21d["vs_industry_ICIR"] = ic_21d["ICIR"] - industry_icir
    ic_21d["pct_improve"] = (ic_21d["vs_industry_ICIR"] / abs(industry_icir) * 100).round(1)
else:
    ic_21d["vs_industry_ICIR"] = np.nan
    ic_21d["pct_improve"] = np.nan

for _, r in ic_21d.iterrows():
    imp = f"{r['pct_improve']:+.1f}%" if not np.isnan(r['pct_improve']) else "---"
    print(f"{r['factor_name']:<28} ICIR={r['ICIR']:>7.4f}  vs行业: {imp}  t={r['IC_tstat']:>6.2f}")

# 保存CSV
ic_csv = os.path.join(OUTPUT_DIR, f"cluster_reversal_ic_{timestamp}.csv")
ic_df.to_csv(ic_csv, index=False, encoding="utf-8-sig")
print(f"\nIC详细数据: {ic_csv}")

# 生成报告
report_lines = [
    f"# #15 聚类增强反转策略 — IC/ICIR 检验报告",
    f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"数据范围: {factor_snapshots['trade_date'].min().date()} ~ {factor_snapshots['trade_date'].max().date()}",
    f"有效截面数: {processed}",
    f"总观测: {len(factor_snapshots):,}",
    "",
    "## 研究方法",
    "",
    "基于 Jiao & Zheng (2026) Pacific-Basin Finance Journal 的框架：",
    "- 每月末使用 K-means 对股票进行聚类（技术特征：多窗口动量+波动率+规模+流动性+换手率）",
    "- 在聚类群组内部计算10日反转信号的截面排名",
    "- 对比三种反转因子：全市场反转 / 行业分类反转 / K-means聚类反转",
    "",
    "## IC/ICIR 对比 (月度 21d)",
    "",
    ic_21d.to_markdown(index=False),
    "",
    "## 全持有期对比",
    "",
    ic_df.to_markdown(index=False),
    "",
    "## 结论",
    "",
]

# 判断最优
best = ic_21d.iloc[0]
report_lines.append(f"- **最优反转因子**: {best['factor_name']} (ICIR={best['ICIR']:.4f})")
report_lines.append(f"- **vs 行业分类反转**: ICIR差异 {best['vs_industry_ICIR']:+.4f} ({best['pct_improve']:+.1f}%)")

# 检查聚类是否优于行业
cluster_best = ic_21d[ic_21d["factor_col"].str.contains("cluster")].iloc[0] if any(ic_21d["factor_col"].str.contains("cluster")) else None
if cluster_best is not None:
    if cluster_best["vs_industry_ICIR"] > 0:
        report_lines.append(f"- **结论**: ✅ 聚类反转优于行业分类反转，建议推进Phase 3聚宽回测")
    else:
        report_lines.append(f"- **结论**: ❌ 聚类反转未优于行业分类反转（{cluster_best['vs_industry_ICIR']:+.4f}），不建议继续")
else:
    report_lines.append(f"- **结论**: 数据不足以判断，需扩大样本或调整聚类参数")

report_path = os.path.join(OUTPUT_DIR, f"cluster_reversal_ic_report_{timestamp}.md")
with open(report_path, "w", encoding="utf-8-sig") as f:
    f.write("\n".join(report_lines))
print(f"完整报告: {report_path}")

print(f"\n✅ #15 Phase 1 完成! 耗时: {datetime.now().strftime('%H:%M:%S')}")
