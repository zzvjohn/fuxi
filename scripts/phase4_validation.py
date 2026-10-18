# -*- coding: utf-8 -*-
"""
Phase 4: 严格验证
1. 样本外检验: 2021-2023训练因子权重, 2024-2025纯样本外回测
2. 交易成本: 0.3%/月手续费
3. 涨停过滤: 买入日封板涨停的股票剔除
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
print("Phase 4: 严格验证")
print("=" * 60)

# ============================================================
# 1. 数据加载 & 因子计算
# ============================================================
print("\n[1/5] 加载数据...")

prices = pd.read_csv(os.path.join(DATA_DIR, "daily_prices.csv"),
                     encoding="utf-8-sig", dtype={"ts_code": str})
prices["trade_date"] = pd.to_datetime(prices["trade_date"], format="mixed")
prices = prices.sort_values(["ts_code", "trade_date"])

basic = pd.read_csv(os.path.join(DATA_DIR, "daily_basic.csv"),
                    encoding="utf-8-sig", dtype={"ts_code": str})
basic["trade_date"] = pd.to_datetime(basic["trade_date"], format="mixed")

idx_data = pd.read_csv(os.path.join(DATA_DIR, "index_000300_daily.csv"),
                       encoding="utf-8-sig", dtype={"ts_code": str})
idx_data["trade_date"] = pd.to_datetime(idx_data["trade_date"], format="mixed")

limits = pd.read_csv(os.path.join(DATA_DIR, "stk_limit.csv"),
                     encoding="utf-8-sig", dtype={"ts_code": str})
limits["trade_date"] = pd.to_datetime(limits["trade_date"], format="mixed")

# 合并
df = prices[["ts_code", "trade_date", "close", "vol"]].copy()
df = df.merge(basic[["ts_code", "trade_date", "pb", "ps_ttm", "total_mv", "turnover_rate"]],
              on=["ts_code", "trade_date"], how="left")
df = df.merge(limits[["ts_code", "trade_date", "up_limit", "down_limit"]],
              on=["ts_code", "trade_date"], how="left")

df["ret_1d"] = df.groupby("ts_code")["close"].pct_change()

print("[2/5] 计算因子...")

# 基础因子
df["pb_inv"] = -df["pb"]
df["ps_ttm_inv"] = -df["ps_ttm"]
df["turnover_20d"] = df.groupby("ts_code")["turnover_rate"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
for w, name in [(21, "mom_1m"), (63, "mom_3m")]:
    df[name] = df.groupby("ts_code")["close"].transform(lambda x: x.pct_change(w))
df["vol_1m"] = df.groupby("ts_code")["ret_1d"].transform(
    lambda x: x.rolling(21, min_periods=10).std()
)
df["vol_1m_inv"] = -df["vol_1m"]
df["amihud_daily"] = np.abs(df["ret_1d"]) / (df["vol"] * df["close"] / 1e8).replace(0, np.nan)
df["amihud_20d"] = df.groupby("ts_code")["amihud_daily"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
df["illiq"] = -df["amihud_20d"]

# 指数
idx = idx_data[["trade_date", "close"]].copy()
idx["mkt_ret"] = idx["close"].pct_change()
idx = idx.dropna(subset=["mkt_ret"])
df = df.merge(idx[["trade_date", "mkt_ret"]], on="trade_date", how="left")

# 特质波动+残差动量
print("  计算特质波动率 & 残差动量...")
df["idio_vol"] = np.nan
df["resid_mom"] = np.nan
codes = df["ts_code"].unique()
for i in range(0, len(codes), 500):
    batch = codes[i:i+500]
    mask = df["ts_code"].isin(batch)
    db = df.loc[mask].copy()
    for code in db["ts_code"].unique():
        ic = db["ts_code"] == code
        dc = db.loc[ic].dropna(subset=["ret_1d", "mkt_ret"]).sort_values("trade_date")
        if len(dc) < 63:
            continue
        try:
            y = dc["ret_1d"].values
            X = np.column_stack([np.ones(len(y)), dc["mkt_ret"].values])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
            r = pd.Series(resid, index=dc.index)
            df.loc[dc.index, "idio_vol"] = r.rolling(63, min_periods=20).std()
            df.loc[dc.index, "resid_mom"] = r.rolling(63, min_periods=30).sum()
        except Exception:
            pass
    if i % 2500 == 0 and i > 0:
        print(f"    {i}/{len(codes)}")

df["idio_vol_inv"] = -df["idio_vol"]
df = df[(df["close"] > 0) & (df["vol"] > 0)].copy()

# ============================================================
# 2. 训练期IC分析 (2021-2023)
# ============================================================
print("\n[3/5] 训练期IC分析 (2021-2023)...")

train_end = pd.Timestamp("2023-12-31")
train_df = df[df["trade_date"] <= train_end].copy()
test_df = df[df["trade_date"] >= "2024-01-01"].copy()

# 计算月频前向收益用于IC（而非日收益）
train_df["fwd_ret_21d"] = train_df.groupby("ts_code")["close"].transform(lambda x: x.shift(-21) / x - 1)

FACTOR_LIST = {
    "vol_1m_inv":    ("波动",  0.55),
    "pb_inv":        ("估值",  0.52),
    "idio_vol_inv":  ("波动",  0.49),
    "ps_ttm_inv":    ("估值",  0.41),
    "resid_mom":     ("反转", -0.75),
    "illiq":         ("反转", -0.57),
    "mom_1m":        ("反转", -0.55),
    "turnover_20d":  ("反转", -0.51),
    "mom_3m":        ("反转", -0.50),
}

# 训练期IC（月频前向收益）
train_ic_weights = {}
for fac, (ftype, _) in FACTOR_LIST.items():
    if fac not in train_df.columns:
        continue
    ic_data = train_df[[fac, "fwd_ret_21d", "trade_date"]].dropna().copy()
    ic_daily = ic_data.groupby("trade_date").apply(
        lambda g: stats.spearmanr(g[fac].rank(), g["fwd_ret_21d"].rank())[0]
        if len(g) >= 30 else np.nan
    ).dropna()

    if len(ic_daily) < 50:
        continue

    ic_mean = ic_daily.mean()
    icir = ic_mean / ic_daily.std()
    train_ic_weights[fac] = round(icir, 4)
    print(f"  训练期 {fac}: IC={ic_mean:.4f}, ICIR={icir:.4f}")

# 对比全样本权重
print(f"\n  训练期有效因子数: {sum(1 for v in train_ic_weights.values() if abs(v) > 0.5)}")

# ============================================================
# 3. 计算复合因子（训练期权重 + 全样本权重）
# ============================================================
print("\n[4/5] 计算复合因子 & 回测...")

def compute_composite(data, weights_dict):
    """计算加权复合因子"""
    data = data.copy()
    data["composite"] = 0.0
    data["weight_sum"] = 0.0
    for fac, w in weights_dict.items():
        if fac not in data.columns:
            continue
        col = f"{fac}_rank"
        data[col] = data.groupby("trade_date")[fac].transform(lambda x: x.rank(pct=True))
        valid = data[col].notna()
        data.loc[valid, "composite"] += w * data.loc[valid, col]
        data.loc[valid, "weight_sum"] += abs(w)
    mask = data["weight_sum"] > 0
    data.loc[mask, "composite"] /= data.loc[mask, "weight_sum"]
    data["n_valid"] = data[[f"{f}_rank" for f in weights_dict if f"{f}_rank" in data.columns]].notna().sum(axis=1)
    return data

# 全样本权重
full_weights = {fac: w for fac, (_, w) in FACTOR_LIST.items()}

# 先算全样本权重复合，再算训练权重复合
test_df_w_full = compute_composite(test_df.copy(), full_weights)
test_df["composite"] = test_df_w_full["composite"]
test_df["n_valid"] = test_df_w_full["n_valid"]

test_df_w_train = compute_composite(test_df.copy(), train_ic_weights)
test_df["composite_train"] = test_df_w_train["composite"]
test_df["train_n_valid"] = test_df_w_train["n_valid"]

# ============================================================
# 4. 回测（样本外 2024-2025）
# ============================================================
def backtest(df_data, composite_col, n_valid_col, top_n, cost_pct=0.0, filter_limit=True):
    """月度调仓回测"""
    df_data = df_data.copy()
    df_data["year_month"] = df_data["trade_date"].dt.to_period("M")
    month_ends = sorted(df_data.groupby("year_month")["trade_date"].max().unique())

    returns = []
    for k in range(1, len(month_ends)):
        sig_date = month_ends[k-1]
        exit_date = month_ends[k]

        sig = df_data[df_data["trade_date"] == sig_date].dropna(subset=[composite_col]).copy()
        sig = sig[sig[n_valid_col] >= 4]
        if len(sig) < top_n:
            continue

        # 涨停过滤
        if filter_limit:
            sig = sig[~(sig["close"] >= sig["up_limit"].fillna(np.inf))]

        selected = sig.nlargest(top_n, composite_col)
        exit_p = df_data[df_data["trade_date"] == exit_date][["ts_code", "close"]].rename(
            columns={"close": "exit_close"})
        selected = selected.merge(exit_p, on="ts_code", how="left")
        selected["ret"] = selected["exit_close"] / selected["close"] - 1
        selected = selected.dropna(subset=["ret"])

        if len(selected) == 0:
            continue

        port_ret = selected["ret"].mean()
        # 扣除手续费
        if cost_pct > 0:
            port_ret -= cost_pct / 100

        returns.append({"date": sig_date, "return": port_ret, "n": len(selected)})

    return pd.DataFrame(returns)

print("  回测: 全样本权重, 无手续费, 有涨停过滤")
bt_full = backtest(test_df, "composite", "n_valid", 30, cost_pct=0, filter_limit=True)

print("  回测: 训练期权重, 无手续费, 有涨停过滤")
bt_train = backtest(test_df, "composite_train", "train_n_valid", 30, cost_pct=0, filter_limit=True)

print("  回测: 全样本权重, 手续费0.3%, 有涨停过滤")
bt_full_cost = backtest(test_df, "composite", "n_valid", 30, cost_pct=0.3, filter_limit=True)

print("  回测: 全样本权重, 无手续费, 无涨停过滤(对比)")
bt_full_nolimit = backtest(test_df, "composite", "n_valid", 30, cost_pct=0, filter_limit=False)

# 基准
bench_df = test_df[test_df["trade_date"].isin(month_end_dates := sorted(
    test_df.groupby(test_df["trade_date"].dt.to_period("M"))["trade_date"].max().unique()
))].copy()
bench_returns = []
md = sorted(bench_df["trade_date"].unique())
for k in range(1, len(md)):
    sig = bench_df[bench_df["trade_date"] == md[k-1]]
    ext = bench_df[bench_df["trade_date"] == md[k]][["ts_code", "close"]].rename(columns={"close": "exit_close"})
    sig = sig.merge(ext, on="ts_code", how="inner")
    sig["ret"] = sig["exit_close"] / sig["close"] - 1
    sig = sig.dropna(subset=["ret"])
    if len(sig) > 0:
        bench_returns.append({"date": md[k-1], "return": sig["ret"].mean()})
bt_bench = pd.DataFrame(bench_returns)

# ============================================================
# 5. 绩效对比
# ============================================================
print("\n[5/5] 绩效对比...")

def metrics(rets):
    if len(rets) == 0:
        return {}
    r = np.array(rets)
    cum = (1 + r).prod() - 1
    n = len(r)
    ann = (1 + cum) ** (12 / n) - 1 if n > 0 else 0
    vol = r.std() * np.sqrt(12)
    sharpe = (ann - 0.02) / vol if vol > 0 else 0
    cum_s = np.cumprod(1 + r)
    dd = (cum_s - np.maximum.accumulate(cum_s)) / np.maximum.accumulate(cum_s)
    max_dd = dd.min()
    win = (r > 0).mean()
    return {
        "cum_ret": round(cum * 100, 2),
        "ann_ret": round(ann * 100, 2),
        "ann_vol": round(vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd * 100, 2),
        "win_rate": round(win * 100, 1),
        "months": n,
    }

scenarios = [
    ("全样本权重(无成本)", bt_full),
    ("训练期权重(OOS)", bt_train),
    ("全样本+手续费0.3%", bt_full_cost),
    ("全样本+无限价过滤", bt_full_nolimit),
    ("全市场等权(基准)", bt_bench),
]

print("\n" + "=" * 80)
print("Phase 4 严格验证结果 (2024-01 ~ 2026-05, 样本外)")
print("=" * 80)
print(f"{'场景':<25} {'累计%':>8} {'年化%':>8} {'波动%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'胜率%':>7}")
print("-" * 80)

for name, bt_df in scenarios:
    m = metrics(bt_df["return"].values)
    if m:
        print(f"{name:<25} {m['cum_ret']:>8.1f} {m['ann_ret']:>8.1f} {m['ann_vol']:>8.1f} "
              f"{m['sharpe']:>7.2f} {m['max_dd']:>8.1f} {m['win_rate']:>7.1f}")

print("-" * 80)

# 逐年OOS
print("\n样本外逐年收益:")
for year in sorted(bt_full["date"].dt.year.unique()):
    fy = bt_full[bt_full["date"].dt.year == year]
    ty = bt_train[bt_train["date"].dt.year == year]
    by = bt_bench[bt_bench["date"].dt.year == year]
    fr = (1 + fy["return"]).prod() - 1
    tr = (1 + ty["return"].values[:len(fy)]).prod() - 1
    br = (1 + by["return"].values[:len(fy)]).prod() - 1 if len(by) >= len(fy) else 0
    print(f"  {year}: 全样本OOS {fr*100:+.1f}% | 训练权重 {tr*100:+.1f}% | 基准 {br*100:+.1f}%")

# 保存
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results = []
for name, bt_df in scenarios:
    m = metrics(bt_df["return"].values)
    if m:
        m["scenario"] = name
        results.append(m)

pd.DataFrame(results).to_csv(
    os.path.join(OUTPUT_DIR, f"phase4_validation_{timestamp}.csv"),
    index=False, encoding="utf-8-sig"
)
print(f"\n结果已保存: reports/phase4_validation_{timestamp}.csv")
print(f"Phase 4 完成!")
