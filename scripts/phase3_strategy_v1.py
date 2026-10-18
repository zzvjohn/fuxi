# -*- coding: utf-8 -*-
"""
Phase 3: 基于组合因子的月度选股策略 v1
选股逻辑: 每月末计算复合因子打分 → 选Top N → 等权持有1个月
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy import stats

DATA_DIR = r"e:\quant\data\raw"
OUTPUT_DIR = r"e:\quant\reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Phase 3: 组合因子月度选股策略 v1")
print("=" * 60)

# ============================================================
# 1. 数据加载 & 因子计算
# ============================================================
print("\n[1/3] 加载数据 & 计算因子...")

prices = pd.read_csv(os.path.join(DATA_DIR, "daily_prices.csv"),
                     encoding="utf-8-sig", dtype={"ts_code": str})
prices["trade_date"] = pd.to_datetime(prices["trade_date"], format="mixed")
prices = prices.sort_values(["ts_code", "trade_date"])

basic = pd.read_csv(os.path.join(DATA_DIR, "daily_basic.csv"),
                    encoding="utf-8-sig", dtype={"ts_code": str})
basic["trade_date"] = pd.to_datetime(basic["trade_date"], format="mixed")
basic = basic.sort_values(["ts_code", "trade_date"])

idx_data = pd.read_csv(os.path.join(DATA_DIR, "index_000300_daily.csv"),
                       encoding="utf-8-sig", dtype={"ts_code": str})
idx_data["trade_date"] = pd.to_datetime(idx_data["trade_date"], format="mixed")

fina = pd.read_csv(os.path.join(DATA_DIR, "fina_indicator_full.csv"),
                   encoding="utf-8-sig", dtype={"ts_code": str})
fina["ann_date"] = pd.to_datetime(fina["ann_date"], format="mixed")

# 构建因子DataFrame
df = prices[["ts_code", "trade_date", "close", "vol"]].copy()
df = df.merge(basic[["ts_code", "trade_date", "pb", "ps_ttm",
                      "total_mv", "turnover_rate"]],
              on=["ts_code", "trade_date"], how="left")

df["ret_1d"] = df.groupby("ts_code")["close"].pct_change()

# 因子计算
df["pb_inv"] = -df["pb"]
df["ps_ttm_inv"] = -df["ps_ttm"]
df["turnover_20d"] = df.groupby("ts_code")["turnover_rate"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# 动量
for w, name in [(21, "mom_1m"), (63, "mom_3m")]:
    df[name] = df.groupby("ts_code")["close"].transform(lambda x: x.pct_change(w))

# 波动率
df["vol_1m"] = df.groupby("ts_code")["ret_1d"].transform(
    lambda x: x.rolling(21, min_periods=10).std()
)
df["vol_1m_inv"] = -df["vol_1m"]

# Amihud
df["amihud_daily"] = np.abs(df["ret_1d"]) / (df["vol"] * df["close"] / 1e8).replace(0, np.nan)
df["amihud_20d"] = df.groupby("ts_code")["amihud_daily"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
df["illiq"] = -df["amihud_20d"]

# 沪深300指数收益 → 特质波动+残差动量
idx = idx_data[["trade_date", "close"]].copy()
idx["mkt_ret"] = idx["close"].pct_change()
idx = idx.dropna(subset=["mkt_ret"])
df = df.merge(idx[["trade_date", "mkt_ret"]], on="trade_date", how="left")

print("  计算特质波动率 & 残差动量...")
df["idio_vol"] = np.nan
df["resid_mom"] = np.nan
codes = df["ts_code"].unique()
for i in range(0, len(codes), 500):
    batch = codes[i:i+500]
    mask = df["ts_code"].isin(batch)
    df_batch = df.loc[mask].copy()
    for code in df_batch["ts_code"].unique():
        idx_c = df_batch["ts_code"] == code
        df_c = df_batch.loc[idx_c].dropna(subset=["ret_1d", "mkt_ret"]).sort_values("trade_date")
        if len(df_c) < 63:
            continue
        try:
            y = df_c["ret_1d"].values
            X = np.column_stack([np.ones(len(y)), df_c["mkt_ret"].values])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
            r = pd.Series(resid, index=df_c.index)
            df.loc[df_c.index, "idio_vol"] = r.rolling(63, min_periods=20).std()
            df.loc[df_c.index, "resid_mom"] = r.rolling(63, min_periods=30).sum()
        except Exception:
            pass
    if i % 2500 == 0 and i > 0:
        print(f"    {i}/{len(codes)}")

df["idio_vol_inv"] = -df["idio_vol"]

# 过滤无效数据
df = df[(df["close"] > 0) & (df["vol"] > 0)].copy()

# ============================================================
# 2. 构建月频信号 & 回测
# ============================================================
print("\n[2/3] 构建月频信号...")

# 因子权重（基于 Phase 2 ICIR）
FACTOR_WEIGHTS = {
    "vol_1m_inv":    0.55,
    "pb_inv":        0.52,
    "idio_vol_inv":  0.49,
    "ps_ttm_inv":    0.41,
    "resid_mom":    -0.75,   # 反转
    "illiq":        -0.57,   # 反转
    "mom_1m":       -0.55,   # 反转
    "turnover_20d": -0.51,   # 反转
    "mom_3m":       -0.50,   # 反转
}

# 计算每日复合得分（忽略NaN因子）
df["composite"] = 0.0
df["composite_weight_sum"] = 0.0

for fac, w in FACTOR_WEIGHTS.items():
    if fac not in df.columns:
        continue
    col = f"{fac}_rank"
    df[col] = df.groupby("trade_date")[fac].transform(
        lambda x: x.rank(pct=True)
    )
    # 仅加非NaN因子
    valid = df[col].notna()
    df.loc[valid, "composite"] += w * df.loc[valid, col]
    df.loc[valid, "composite_weight_sum"] += abs(w)

# 按有效因子数量归一化
df["n_valid_factors"] = df[[f"{f}_rank" for f in FACTOR_WEIGHTS if f"{f}_rank" in df.columns]].notna().sum(axis=1)
mask_valid = df["composite_weight_sum"] > 0
df.loc[mask_valid, "composite"] /= df.loc[mask_valid, "composite_weight_sum"]

# 生成月频调仓日期：每月最后一个交易日
df["year_month"] = df["trade_date"].dt.to_period("M")
month_ends = df.groupby("year_month")["trade_date"].max().reset_index()
month_end_dates = sorted(month_ends["trade_date"].unique())
print(f"  月频调仓日数: {len(month_end_dates)} (从 {month_end_dates[0].date()} 到 {month_end_dates[-1].date()})")

# 回测
TOP_N = 30  # 持仓数量
MAX_WEIGHT = 0.08  # 单票上限

portfolio_returns = []
portfolio_details = []
benchmark_returns = []

# 基准：等权全市场
n_skipped = 0
for k in range(1, len(month_end_dates)):
    signal_date = month_end_dates[k-1]   # 信号日（月末）
    entry_date = signal_date             # 以月末收盘价买入
    exit_date = month_end_dates[k]       # 下月末收盘价卖出

    # 信号日数据：所有股票的因子得分（至少4个有效因子）
    signal_mask = df["trade_date"] == signal_date
    signal_df = df[signal_mask].dropna(subset=["composite"]).copy()
    signal_df = signal_df[signal_df["n_valid_factors"] >= 4]
    if k <= 3:
        print(f"  Debug k={k}: signal_date={signal_date.date()}, mask_sum={signal_mask.sum()}, "
              f"signal_df_len={len(signal_df)}, composite_notna={df.loc[signal_mask, 'composite'].notna().sum()}")
    if len(signal_df) < TOP_N:
        n_skipped += 1
        continue

    # 选Top N
    selected = signal_df.nlargest(TOP_N, "composite")

    # 计算持仓收益：exit_date收盘价 / signal_date收盘价 - 1
    exit_prices = df[df["trade_date"] == exit_date][["ts_code", "close"]].rename(
        columns={"close": "exit_close"}
    )
    selected = selected.merge(exit_prices, on="ts_code", how="left")
    selected["stock_return"] = selected["exit_close"] / selected["close"] - 1
    selected = selected.dropna(subset=["stock_return"])

    if len(selected) == 0:
        continue

    # 等权组合收益
    port_return = selected["stock_return"].mean()
    portfolio_returns.append({
        "entry_date": entry_date,
        "exit_date": exit_date,
        "return": port_return,
        "n_stocks": len(selected),
    })

    # 基准：等权全市场
    bench_df = df[df["trade_date"] == signal_date][["ts_code"]].copy()
    bench_df = bench_df.merge(exit_prices, on="ts_code", how="left")
    bench_df = bench_df.merge(
        df[df["trade_date"] == signal_date][["ts_code", "close"]],
        on="ts_code", how="left"
    )
    bench_df = bench_df.dropna()
    if len(bench_df) > 0:
        bench_df["bench_return"] = bench_df["exit_close"] / bench_df["close"] - 1
        benchmark_returns.append({
            "entry_date": entry_date,
            "return": bench_df["bench_return"].mean(),
        })

    # 记录持仓明细
    for _, s in selected.iterrows():
        portfolio_details.append({
            "entry_date": entry_date,
            "ts_code": s["ts_code"],
            "composite_score": s["composite"],
            "return": s["stock_return"],
        })

port_df = pd.DataFrame(portfolio_returns)
bench_df = pd.DataFrame(benchmark_returns)
detail_df = pd.DataFrame(portfolio_details)

if len(port_df) == 0:
    print("\nERROR: 没有生成任何持仓。请检查因子数据质量。")
    import sys; sys.exit(1)

print(f"  策略月份数: {len(port_df)} (跳过 {n_skipped} 个月)")
print(f"  平均持仓数: {port_df['n_stocks'].mean():.0f}")

# ============================================================
# 3. 绩效分析
# ============================================================
print("\n[3/3] 绩效分析...")

def calc_metrics(monthly_returns, name):
    """计算策略绩效指标"""
    if len(monthly_returns) == 0:
        return {}

    rets = monthly_returns.copy()

    # 累积收益
    cum_ret = (1 + rets).prod() - 1

    # 年化收益
    n_months = len(rets)
    annual_ret = (1 + cum_ret) ** (12 / n_months) - 1

    # 年化波动率
    annual_vol = rets.std() * np.sqrt(12)

    # Sharpe (假设无风险利率 2%)
    sharpe = (annual_ret - 0.02) / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    cum_series = np.array((1 + rets).cumprod())
    running_max = np.maximum.accumulate(cum_series)
    drawdown = (cum_series - running_max) / running_max
    max_dd = drawdown.min()

    # 胜率
    win_rate = (rets > 0).mean()

    # 月收益 stats
    avg_monthly = rets.mean()

    return {
        "name": name,
        "months": n_months,
        "cumulative_return": round(cum_ret * 100, 2),
        "annual_return": round(annual_ret * 100, 2),
        "annual_volatility": round(annual_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "avg_monthly_return": round(avg_monthly * 100, 2),
    }

strategy_metrics = calc_metrics(port_df["return"].values, "Composite Top30")
bench_metrics = calc_metrics(bench_df["return"].values, "全市场等权")

# 输出对比
print("\n" + "=" * 70)
print("策略绩效对比")
print("=" * 70)
print(f"{'指标':<20} {'组合策略':>15} {'全市场等权':>15}")
print("-" * 55)

for key, label in [
    ("cumulative_return", "累计收益(%)"),
    ("annual_return", "年化收益(%)"),
    ("annual_volatility", "年化波动率(%)"),
    ("sharpe_ratio", "Sharpe比率"),
    ("max_drawdown", "最大回撤(%)"),
    ("win_rate", "月胜率(%)"),
    ("avg_monthly_return", "月均收益(%)"),
    ("months", "回测月数"),
]:
    sv = strategy_metrics.get(key, "-")
    bv = bench_metrics.get(key, "-")
    if isinstance(sv, float):
        print(f"{label:<20} {sv:>15.2f} {bv:>15.2f}")
    else:
        print(f"{label:<20} {sv!s:>15} {bv!s:>15}")

print("-" * 55)

# 超额收益
excess = port_df["return"].values - bench_df["return"].values[:len(port_df)]
excess_cum = (1 + excess).prod() - 1
print(f"{'超额累计收益':<20} {excess_cum*100:>15.2f}%")
info_ratio = excess.mean() / excess.std() * np.sqrt(12) if excess.std() > 0 else 0
print(f"{'信息比率(年化)':<20} {info_ratio:>15.2f}")

# 逐年分析
print("\n" + "=" * 70)
print("逐年收益对比")
print("=" * 70)

port_df["year"] = port_df["entry_date"].dt.year
bench_df["year"] = bench_df["entry_date"].dt.year

for year in sorted(port_df["year"].unique()):
    py = port_df[port_df["year"] == year]
    by = bench_df[bench_df["year"] == year]
    p_ret = (1 + py["return"]).prod() - 1
    b_ret = (1 + by["return"].values[:len(py)]).prod() - 1 if len(by) >= len(py) else np.nan
    alpha = p_ret - b_ret
    print(f"  {year}: 策略 {p_ret*100:+.1f}%  |  基准 {b_ret*100:+.1f}%  |  超额 {alpha*100:+.1f}%")

# ============================================================
# 4. 保存
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

metrics_csv = os.path.join(OUTPUT_DIR, f"strategy_v1_metrics_{timestamp}.csv")
pd.DataFrame([strategy_metrics, bench_metrics]).to_csv(metrics_csv, index=False, encoding="utf-8-sig")

detail_csv = os.path.join(OUTPUT_DIR, f"strategy_v1_details_{timestamp}.csv")
detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

# 月度收益
returns_csv = os.path.join(OUTPUT_DIR, f"strategy_v1_returns_{timestamp}.csv")
port_df.assign(bench_return=bench_df["return"].values[:len(port_df)]).to_csv(
    returns_csv, index=False, encoding="utf-8-sig"
)

print(f"\n详细持仓: {detail_csv}")
print(f"月度收益: {returns_csv}")

print(f"\nPhase 3 完成!")
