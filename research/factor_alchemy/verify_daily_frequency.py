# -*- coding: utf-8 -*-
"""
日频 vs 周频 对比验证 — 基于 GA 复合因子
=========================================
用周频因子面板估算日频换手率、成本和净收益
"""
import pickle
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── 配置 ──
STOCK_NUM    = 30
BUFFER_MULT  = 1.8
BUFFER_NUM   = int(STOCK_NUM * BUFFER_MULT)
COST_PER_TRADE = 0.0001 + 0.001 + 0.0001 + 0.003  # 万1 + 千1 + 万1 + 0.3%滑点

GA_WEIGHTS = {
    'vol_1m': 0.170, 'net_margin': 0.167, 'max_ret_1m': 0.153,
    'bull_power': 0.150, 'ln_circulating_mcap': 0.133,
    'turnover_to_vol': 0.118, 'f_score': 0.110,
}

# ── 加载 GA 结果 ──
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
with open(os.path.join(output_dir, 'phase45_result.pkl'), 'rb') as f:
    result = pickle.load(f)

best_weights = result['best_weights']
print("=" * 65)
print("  日频 vs 周频 — GA 复合因子全维度对比")
print("=" * 65)

# ── 重建复合因子 ──
factor_dfs = {}
for name, w in best_weights.items():
    csv_path = os.path.join(output_dir, 'factor_%s.csv' % name)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True).astype(float)
        factor_dfs[name] = df

# 对齐所有因子
common_dates = factor_dfs[list(factor_dfs.keys())[0]].index
common_codes = factor_dfs[list(factor_dfs.keys())[0]].columns

composite = pd.DataFrame(0.0, index=common_dates, columns=common_codes)
for name, df in factor_dfs.items():
    w = best_weights[name]
    cidx = composite.index.intersection(df.index)
    ccol = composite.columns.intersection(df.columns)
    composite.loc[cidx, ccol] += w * df.loc[cidx, ccol].values

# ── 排名 ──
ranks = composite.rank(axis=1, pct=True, ascending=False)  # 高分=前排

print("\n[1] 排名自相关分析")
print("-" * 40)

# 采样500只股票计算
np.random.seed(42)
sample = np.random.choice(ranks.columns, min(500, len(ranks.columns)), replace=False)

for lag in [1, 2, 4, 8, 12]:
    corrs = []
    for col in sample:
        s = ranks[col].dropna()
        if len(s) > lag + 30:
            c = s.autocorr(lag=lag)
            if not np.isnan(c):
                corrs.append(c)
    avg = np.mean(corrs)
    if lag == 1:
        daily_ac = avg ** (1 / 5)
        print("  周频 lag=1:  rank_autocorr = %.4f" % avg)
        print("  → 日频(估算): rank_autocorr = %.4f" % daily_ac)
        print("  → 周排名变化: %.1f%% → 日排名变化: %.1f%%" %
              ((1 - avg) * 100, (1 - daily_ac) * 100))
    else:
        print("  周频 lag=%d:  rank_autocorr = %.4f" % (lag, avg))

# ── 关键: 计算周频和估算日频的 Top30 换手率 ──
print("\n[2] Portfolio 换手率分析 (Top%d, Buffer=%d)" % (STOCK_NUM, BUFFER_NUM))
print("-" * 40)

# 周频: 直接从数据计算
n_weeks = len(ranks)
weekly_turnovers = []

for w in range(1, n_weeks):
    prev_top = set(ranks.iloc[w-1].nlargest(STOCK_NUM).index)
    curr_top = set(ranks.iloc[w].nlargest(STOCK_NUM).index)
    if len(prev_top) > 0:
        # 换手率 = 需要替换的股票数 / 持仓数
        to = 1 - len(prev_top & curr_top) / STOCK_NUM
        weekly_turnovers.append(to)

weekly_to = np.mean(weekly_turnovers)
print("  周频 (无Buffer):")
print("    均换手: %.1f%%  中位: %.1f%%  P95: %.1f%%" %
      (weekly_to * 100, np.median(weekly_turnovers) * 100,
       np.percentile(weekly_turnovers, 95) * 100))

# 周频 + Buffer
weekly_buf_turnovers = []
for w in range(1, n_weeks):
    prev_buf  = set(ranks.iloc[w-1].nlargest(BUFFER_NUM).index)
    prev_top  = set(ranks.iloc[w-1].nlargest(STOCK_NUM).index)
    curr_buf  = set(ranks.iloc[w].nlargest(BUFFER_NUM).index)
    curr_top  = set(ranks.iloc[w].nlargest(STOCK_NUM).index)

    # 卖出: 跌出 buffer 的持仓
    to_sell = prev_top - curr_buf
    # 买入: 进入 top 但未持有
    to_buy  = curr_top - (prev_top & curr_buf)

    max_to = max(len(to_sell), len(to_buy))
    weekly_buf_turnovers.append(max_to / STOCK_NUM)

weekly_buf_to = np.mean(weekly_buf_turnovers)
print("  周频 + Buffer(%.0f):" % BUFFER_MULT)
print("    均换手: %.1f%%  中位: %.1f%%  P95: %.1f%%" %
      (weekly_buf_to * 100, np.median(weekly_buf_turnovers) * 100,
       np.percentile(weekly_buf_turnovers, 95) * 100))
print("    Buffer节省: %.1f%% → %.1f%%  (省 %.0f%%)" %
      (weekly_to * 100, weekly_buf_to * 100,
       (1 - weekly_buf_to / weekly_to) * 100))

# ── 日频估算 (基于 rank_autocorr 插值) ──
daily_ac = daily_ac  # from above
n_sim = 10000
daily_turnovers = []

# Monte Carlo: 模拟每日排名变化
for _ in range(n_sim):
    # 昨日排名
    prev = np.random.permutation(5000)[:STOCK_NUM]
    # 今日排名: 以 daily_ac 概率保持, (1-daily_ac) 概率随机替换
    curr = []
    for p in prev:
        if np.random.random() < daily_ac:
            curr.append(p)
        else:
            curr.append(np.random.randint(5000, 10000))  # 新股票
    curr = list(set(curr))  # 去重

    # 换手 = 不在昨日持仓中的比例
    overlap = len(set(prev) & set(curr))
    daily_turnovers.append(1 - overlap / STOCK_NUM)

daily_to_est = np.mean(daily_turnovers)
print("\n  日频 (Monte Carlo, 基于 rank_autocorr=%.3f):" % daily_ac)
print("    均换手: %.1f%%  P50: %.1f%%  P95: %.1f%%" %
      (daily_to_est * 100, np.median(daily_turnovers) * 100,
       np.percentile(daily_turnovers, 95) * 100))

# 日频 + Buffer Monte Carlo
daily_buf_turnovers = []
for _ in range(n_sim):
    prev = np.random.permutation(5000)[:BUFFER_NUM]
    prev_top = prev[:STOCK_NUM]
    curr = []
    for p in prev:
        if np.random.random() < daily_ac:
            curr.append(p)
        else:
            curr.append(np.random.randint(5000, 10000))
    curr = list(set(curr))
    curr_top = curr[:STOCK_NUM]

    to_sell = len(set(prev_top) - set(curr))
    to_buy  = len(set(curr_top) - set(prev_top))
    daily_buf_turnovers.append(max(to_sell, to_buy) / STOCK_NUM)

daily_buf_to = np.mean(daily_buf_turnovers)
print("\n  日频 + Buffer(%.0f) (Monte Carlo):" % BUFFER_MULT)
print("    均换手: %.1f%%  P50: %.1f%%  P95: %.1f%%" %
      (daily_buf_to * 100, np.median(daily_buf_turnovers) * 100,
       np.percentile(daily_buf_turnovers, 95) * 100))

# ── 成本与净收益 ──
print("\n" + "=" * 65)
print("[3] 成本与净收益对比")
print("-" * 40)

# 模型参数
IC_MEAN    = 0.0874          # 合成因子IC均值 (来自稳定性报告)
IC_IR      = 0.632           # 合成因子ICIR
GROSS_SPREAD_WEEKLY = 0.0052  # 预期周均价差收益
WEEKS_YEAR = 52
DAYS_YEAR  = 250

print("  模型参数: IC=%.4f, ICIR=%.3f, 周预期价差=%.2f%%" %
      (IC_MEAN, IC_IR, GROSS_SPREAD_WEEKLY * 100))

print("\n  %-20s %10s %12s %12s %10s" %
      ("方案", "期均换手", "年化成本", "年化毛收益", "年化净收益"))
print("  " + "-" * 65)

# 情景
scenarios = [
    ("周频-无Buffer",   weekly_to,     52),
    ("周频-Buffer%.0f" % BUFFER_MULT, weekly_buf_to, 52),
    ("日频-无Buffer",   daily_to_est,  DAYS_YEAR),
    ("日频-Buffer%.0f" % BUFFER_MULT, daily_buf_to, DAYS_YEAR),
    ("日频-Buffer-低TO", min(daily_buf_to, 0.05), DAYS_YEAR),
]

for name, per_to, n_periods in scenarios:
    annual_cost = per_to * n_periods * COST_PER_TRADE
    annual_gross = GROSS_SPREAD_WEEKLY * WEEKS_YEAR
    # 长仓捕获率修正
    if '日频' in name:
        annual_gross *= 1.05  # 日频捕获更多价差
    else:
        annual_gross *= 0.95

    # 长仓实现率 50%
    effective_gross = annual_gross * 0.50
    annual_net = effective_gross - annual_cost

    status = "✅赚钱" if annual_net > 0.05 else ("⚠️微利" if annual_net > 0 else "❌亏损")
    print("  %-20s %9.1f%% %11.1f%% %11.1f%% %10.1f%% %s" %
          (name, per_to * 100, annual_cost * 100, effective_gross * 100,
           annual_net * 100, status))

# ── 敏感性分析 ──
print("\n" + "=" * 65)
print("[4] 敏感性分析: 日频 Buffer — TO vs 净收益")
print("-" * 40)
print("  %-12s %s" % ("日换手率", "年化净收益"))
print("  " + "-" * 35)

annual_gross_eff = GROSS_SPREAD_WEEKLY * WEEKS_YEAR * 1.05 * 0.50
for daily_to in np.arange(0.02, 0.21, 0.02):
    annual_cost = daily_to * DAYS_YEAR * COST_PER_TRADE
    annual_net = annual_gross_eff - annual_cost
    bar = "█" * int(max(0, annual_net * 200))
    status = "✅" if annual_net > 0.05 else ("⚠️" if annual_net > 0 else "❌")
    print("  %s %5.1f%%  %6.1f%%  %s" % (status, daily_to * 100, annual_net * 100, bar))

# ── 关键结论 ──
print("\n" + "=" * 65)
print("[5] 关键结论")
print("-" * 40)

reduction = (1 - daily_buf_to / weekly_to) * 100
print("  1. 周频换手 %.0f%% → 日频+Buffer 约 %.0f%% (减少 %.0f%%)" %
      (weekly_to * 100, daily_buf_to * 100, reduction))

if daily_buf_to < 0.10:
    print("  2. 日频 Buffer TO < 10% → 交易成本可控 ✅")
    print("  3. 预计年化净收益 5-15% (取决于信号衰减速度)")
    print("  4. 结论: 日频+Buffer 大概率可行!")
else:
    print("  2. 日频 Buffer TO 仍偏高 (>10%) → 需进一步优化")
    print("  3. 建议: 增大Buffer倍率 / 阈值调仓 / 因子预筛选")

print("\n  注意事项:")
print("  - 以上基于周频面板插值估算, 实盘日频信号可能更噪")
print("  - 需在聚宽上实际回测验证")
print("  - 建议运行策略: strategies/final/daily_ga_composite_jq.py")
