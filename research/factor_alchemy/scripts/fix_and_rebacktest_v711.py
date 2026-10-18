#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PortfolioSimulator 修复 + v7.11 全方位重回测
=============================================
修复:
  1. Bug: 换手过滤后 added/removed 未重算 → 成本失真
  2. Slippage: 改为 JQ FixedSlippage(0.003元/股) 模型
  3. 测试口径对齐: 同时测 regime-specific 和 all-weeks

输出:
  - 修复前后对比
  - Local vs JQ 已知结果校准
  - 更新后的 stats JSON
"""
import sys, os, time, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # factor_alchemy/
sys.path.insert(0, str(SCRIPT_DIR))

from data.loader import load_all_data
from factors.composite import standardize_factors

print("=" * 70)
print("  PortfolioSimulator 修复 v3 → v7.11 重回测")
print("=" * 70)

# =====================================================
# Part A: 修复版 PortfolioSimulator (v3)
# =====================================================

class PortfolioSimulatorV3:
    """
    v3 修复 (2026-07-20):
      - BugFix: turnover过滤后重算 added/removed, 成本对齐实际持仓
      - Slippage: 改为 per-share 模型 (JQ FixedSlippage(0.003)对齐)
      - 平均股价估算: 用于 per-share → 百分比成本转换
      - 诊断输出: 按需打印
    """

    def __init__(self, factor_df, price_df, top_n=30, weighting='equal',
                 max_turnover=0.60, commission=0.0001, stamp_tax=0.001,
                 slippage_per_share=0.003, min_commission=5, debug=False,
                 mcap_df=None, mcap_min=None):
        self.factor_df = factor_df
        self.price_df = price_df
        self.top_n = top_n
        self.weighting = weighting
        self.max_turnover = max_turnover
        self.commission = commission          # 佣金率 万1
        self.stamp_tax = stamp_tax            # 印花税 千1 (仅卖)
        self.slippage_per_share = slippage_per_share  # 元/股 (JQ FixedSlippage)
        self.min_commission = min_commission
        self.debug = debug
        self.mcap_df = mcap_df
        self.mcap_min = mcap_min

    def _slippage_pct(self, price):
        """per-share slippage → 百分比 (用于组合层面成本)"""
        return self.slippage_per_share / max(price, 0.01)

    def run(self):
        dates = sorted(self.factor_df.index)
        if len(dates) < 2:
            return {'stats': {'error': '日期不足'}}

        # 预过滤
        aligned_factor = self.factor_df.where(
            self.price_df.notna() & (self.price_df > 0))

        # 市值过滤
        if self.mcap_df is not None and self.mcap_min is not None:
            mcap_aligned = self.mcap_df.reindex(
                index=self.factor_df.index, columns=self.factor_df.columns)
            aligned_factor = aligned_factor.where(mcap_aligned >= self.mcap_min)

        nav = pd.Series(1.0, index=[dates[0]])
        weekly_returns = pd.Series(dtype=float)
        turnover_series = pd.Series(dtype=float)
        positions_history = {}
        current_holdings = set()

        for i in range(len(dates) - 1):
            rebalance_date = dates[i]
            forward_date = dates[i + 1]

            # 选股
            signal = aligned_factor.loc[rebalance_date].dropna()
            if len(signal) < self.top_n:
                continue

            top_stocks = signal.nlargest(self.top_n).index.tolist()

            # 二次过滤: 价格有效性
            if rebalance_date in self.price_df.index and forward_date in self.price_df.index:
                prices_reb = self.price_df.loc[rebalance_date]
                prices_fwd = self.price_df.loc[forward_date]
                valid_top = [s for s in top_stocks
                           if s in prices_reb.index and s in prices_fwd.index
                           and not np.isnan(prices_reb[s]) and not np.isnan(prices_fwd[s])
                           and prices_reb[s] > 0 and prices_fwd[s] > 0]
                if len(valid_top) < max(3, self.top_n // 3):
                    continue
                top_stocks = valid_top[:self.top_n]

            # ---- 换手计算 + 过滤 ----
            new_holdings_raw = set(top_stocks[:self.top_n])

            if len(current_holdings) > 0:
                added_raw = new_holdings_raw - current_holdings
                removed_raw = current_holdings - new_holdings_raw
                turnover_raw = (len(added_raw) + len(removed_raw)) / len(current_holdings)

                if turnover_raw > self.max_turnover:
                    # 换手限制: 保留信号最强的旧持仓
                    keep_from_current = sorted(
                        current_holdings & set(signal.index),
                        key=lambda s: signal.get(s, -np.inf), reverse=True)
                    max_new = int(self.top_n * self.max_turnover)
                    keep = set(keep_from_current[:self.top_n - max_new])
                    new_from_signal = [
                        s for s in top_stocks if s not in keep][:max_new]
                    new_holdings = keep | set(new_from_signal)
                else:
                    new_holdings = new_holdings_raw
            else:
                new_holdings = new_holdings_raw
                turnover_raw = 1.0

            # ★★★ BUGFIX: 重新计算 added/removed 基于 FILTERED new_holdings ★★★
            added = new_holdings - current_holdings
            removed = current_holdings - new_holdings
            turnover = (len(added) + len(removed)) / max(1, len(current_holdings)) \
                       if len(current_holdings) > 0 else 1.0

            # ---- 收益计算 ----
            if rebalance_date in self.price_df.index and forward_date in self.price_df.index:
                prices_reb = self.price_df.loc[rebalance_date]
                prices_fwd = self.price_df.loc[forward_date]

                valid_stocks = [s for s in new_holdings
                               if s in prices_reb.index and s in prices_fwd.index]
                if len(valid_stocks) == 0:
                    continue

                weight = 1.0 / self.top_n
                port_ret = 0.0

                # 记录平均股价用于 slippage 百分比换算
                avg_price = 0.0
                n_priced = 0
                for s in valid_stocks:
                    stock_ret = prices_fwd[s] / prices_reb[s] - 1
                    port_ret += weight * stock_ret
                    avg_price += prices_reb[s]
                    n_priced += 1
                avg_price = avg_price / n_priced if n_priced > 0 else 1.0

                # ★★★ v3: per-share slippage → 百分比 cost ★★★
                slip_pct = self._slippage_pct(avg_price)

                n_added = len(added & set(valid_stocks))
                n_removed = len(removed & set(valid_stocks))

                # 买入成本: commission + per-share slippage
                buy_cost = n_added * weight * (self.commission + slip_pct)
                # 卖出成本: commission + stamp_tax + per-share slippage
                sell_cost = n_removed * weight * (self.commission + self.stamp_tax + slip_pct)
                total_cost = buy_cost + sell_cost

                if self.debug and i < 5:
                    print(f"[{rebalance_date.date()}] hold={len(current_holdings)}→{len(new_holdings)} "
                          f"turnover={turnover:.0%} n_add={n_added} n_rem={n_removed} "
                          f"avg_px={avg_price:.1f} slip_pct={slip_pct:.4%} "
                          f"cost={total_cost:.4%} gross_ret={port_ret:.4%}")

                port_ret -= total_cost

                if not np.isnan(port_ret):
                    new_nav = nav.iloc[-1] * (1 + port_ret)
                    nav.loc[forward_date] = new_nav
                    weekly_returns.loc[forward_date] = port_ret
                    turnover_series.loc[forward_date] = turnover

                current_holdings = set(valid_stocks)
                positions_history[forward_date] = list(valid_stocks)

        stats = self._compute_stats(nav, weekly_returns, turnover_series)
        return {
            'nav': nav, 'returns': weekly_returns,
            'turnover': turnover_series, 'positions': positions_history,
            'stats': stats,
        }

    def _compute_stats(self, nav, returns, turnover):
        if len(returns) < 10:
            return {}
        valid_nav = nav.dropna()
        if len(valid_nav) < 2:
            return {k: np.nan for k in ['total_return','cagr','sharpe','max_drawdown','win_rate','avg_turnover']}

        total_return = valid_nav.iloc[-1] / valid_nav.iloc[0] - 1
        n_weeks = len(returns)
        years = n_weeks / 52
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(52)
        else:
            sharpe = 0

        cum = nav / nav.iloc[0]
        max_dd = (cum / cum.cummax() - 1).min()
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0
        win_rate = (returns > 0).mean()
        avg_turnover = turnover.mean() if len(turnover) > 0 else 0

        return {
            'total_return': total_return, 'cagr': cagr, 'sharpe': sharpe,
            'max_drawdown': max_dd, 'calmar': calmar, 'win_rate': win_rate,
            'avg_turnover': avg_turnover, 'n_weeks': n_weeks, 'years': years,
        }


# =====================================================
# Part B: 加载数据 + 重回测
# =====================================================

print("\n[1] 加载因子缓存")
t0 = time.time()
with open(SCRIPT_DIR / 'output' / 'factor_cache.pkl', 'rb') as f:
    factor_dfs = pickle.load(f)
factor_dfs_std = standardize_factors(factor_dfs)
factor_weeks = sorted(factor_dfs_std[list(factor_dfs_std.keys())[0]].index)
print(f"  {len(factor_dfs_std)} 因子 | {len(factor_weeks)}周 | {time.time()-t0:.1f}s")

print("\n[2] 加载价格+市值")
data = load_all_data('2021-01-01', '2026-06-28')
price_daily = data['price_data']['close']
price_weekly = price_daily.resample('W').last()
price_weekly.index = pd.to_datetime(price_weekly.index)

val = data['valuation_data']
mcap_daily = val.pivot_table(index='trade_date', columns='code', values='circ_mv')
mcap_daily.index = pd.to_datetime(mcap_daily.index)
mcap_weekly = mcap_daily.resample('W').last().ffill(limit=5) / 1e5  # 亿元
mcap_weekly.index = pd.to_datetime(mcap_weekly.index)

common_cols = factor_dfs_std[list(factor_dfs_std.keys())[0]].columns
price_weekly = price_weekly.reindex(columns=common_cols)
mcap_weekly = mcap_weekly.reindex(columns=common_cols)
print(f"  价格={price_weekly.shape} | 市值(亿)={mcap_weekly.shape} | {time.time()-t0:.1f}s")

# Regime 标签
print("\n[3] Regime 标签")
regime_df = pd.read_csv(SCRIPT_DIR / 'data' / 'regime_labels.csv', parse_dates=['date'])
regime_df['regime_merged'] = regime_df['regime'].replace('neutral', np.nan).ffill()
regime_df['regime_merged'] = regime_df['regime_merged'].fillna('small_bull')
regime_series = regime_df.set_index('date')['regime_merged']

fr_index = pd.DatetimeIndex(factor_weeks)
_aligned = regime_series.reindex(fr_index, method='nearest', tolerance=pd.Timedelta(days=10))
_aligned = _aligned.ffill().bfill()
regime_map = {d: _aligned.loc[d] for d in fr_index if pd.notna(_aligned.loc[d])}

from collections import Counter
for r, c in Counter(regime_map.values()).most_common():
    print(f"  {r}: {c}周")

# v711 权重
print("\n[4] v711 权重")
with open(SCRIPT_DIR / 'output' / 'regime_lasso_result_v711.json', encoding='utf-8') as f:
    res = json.load(f)

FLIP_THRESH = 0.10
def signed_weights(regime_name):
    w = res['regimes'][regime_name]['weights']
    icir = res['regimes'][regime_name].get('factor_icir', {})
    out = {}
    for fname, weight in w.items():
        sign = -1.0 if icir.get(fname, 0.0) < -FLIP_THRESH else 1.0
        out[fname] = sign * weight
    return out

signed = {}
for r in res['regimes']:
    if 'weights' in res['regimes'][r]:
        signed[r] = signed_weights(r)
        print(f"  {r}: " + ", ".join(f'{k}:{v:+.3f}' for k, v in sorted(
            signed[r].items(), key=lambda x: -abs(x[1]))))

# 构建合成因子
def build_composite(w_dict):
    comp = pd.DataFrame(0.0, index=factor_weeks, columns=common_cols)
    for fname, w in w_dict.items():
        if fname not in factor_dfs_std:
            continue
        f = factor_dfs_std[fname].reindex(index=factor_weeks, columns=common_cols).fillna(0)
        comp = comp.add(w * f)
    return comp

comp_large = build_composite(signed.get('large_dominant', {}))
comp_small = build_composite(signed.get('small_bull', {}))

regime_arr = pd.Series(regime_map)
comp_combined = comp_large.copy()
is_small = regime_arr == 'small_bull'
comp_combined[is_small.values] = comp_small[is_small.values]

# 检查 NaN 覆盖
large_mask = ~comp_large.isna().all(axis=1)
small_mask = ~comp_small.isna().all(axis=1)
print(f"  有效 large 周: {large_mask.sum()}/{len(large_mask)}")
print(f"  有效 small 周: {small_mask.sum()}/{len(small_mask)}")

# =====================================================
# Part C: V3 重回测 — 多个场景
# =====================================================
print("\n" + "=" * 70)
print("[5] V3 PortfolioSimulator 重回测")
print("=" * 70)

SIM_KW = dict(top_n=30, weighting='equal', max_turnover=0.60,
              commission=0.0001, stamp_tax=0.001, slippage_per_share=0.003,
              min_commission=5, mcap_df=mcap_weekly, mcap_min=5, debug=False)

V2_KW = dict(top_n=30, weighting='equal', max_turnover=0.60,
             commission=0.0001, stamp_tax=0.001, slippage=0.003,  # v2 percentage
             min_commission=5, mcap_df=mcap_weekly, mcap_min=5, debug=False)

results = {}
results_v2 = {}

def run_scenario(name, comp, price, weeks_desc, simulator_cls=PortfolioSimulatorV3, sim_kw=SIM_KW):
    sim = simulator_cls(factor_df=comp, price_df=price, **sim_kw)
    out = sim.run()
    s = out['stats']
    print(f"\n--- {name} ({weeks_desc}) ---")
    if 'error' in s:
        print(f"  ERROR: {s['error']}")
        return s
    print(f"  周数={s['n_weeks']} | 总收益={s['total_return']:+.1%} | 年化={s['cagr']:+.1%} | "
          f"Sharpe={s['sharpe']:.2f} | MaxDD={s['max_drawdown']:.1%} | 换手={s['avg_turnover']:.2%}")
    return s

# ---- V3 回测 ----
# 场景A: large_dominant 权重在 ALL weeks
results['large_dominant_all'] = run_scenario(
    'LD权重×全周期', comp_large, price_weekly,
    f'{len(factor_weeks)}周全样本')

# 场景B: small_bull 权重在 ALL weeks
results['small_bull_all'] = run_scenario(
    'SB权重×全周期', comp_small, price_weekly,
    f'{len(factor_weeks)}周全样本')

# 场景C: large_dominant 权重仅在 LD weeks
large_weeks = sorted([d for d, r in regime_map.items() if r == 'large_dominant'])
comp_large_only = comp_large.loc[large_weeks]
results['large_dominant_regime'] = run_scenario(
    'LD权重×LD周', comp_large_only, price_weekly,
    f'{len(large_weeks)}周LD regime')

# 场景D: small_bull 权重仅在 SB weeks
small_weeks = sorted([d for d, r in regime_map.items() if r == 'small_bull'])
comp_small_only = comp_small.loc[small_weeks]
results['small_bull_regime'] = run_scenario(
    'SB权重×SB周', comp_small_only, price_weekly,
    f'{len(small_weeks)}周SB regime')

# 场景E: regime-aware 合并
results['merged'] = run_scenario(
    'regime-aware合并', comp_combined, price_weekly,
    f'{len(factor_weeks)}周全样本(切换)')

# ---- V2 对照回测 (旧版, 用于对比) ----
print("\n--- V2对照 (旧版simulator, 相同数据) ---")
from portfolio.simulator import PortfolioSimulator as SimV2

for key in ['large_dominant_all', 'small_bull_all', 'merged']:
    comp_map = {'large_dominant_all': comp_large, 'small_bull_all': comp_small, 'merged': comp_combined}
    s = run_scenario(f'V2_{key}', comp_map[key], price_weekly, '', SimV2, V2_KW)
    results_v2[key] = s

# =====================================================
# Part D: 校准
# =====================================================
print("\n" + "=" * 70)
print("[6] 与已知 JQ 结果对比")
print("=" * 70)

# 已知 JQ 结果 (用户提供 / MEMORY记录)
jq_known = {
    'large_dominant_all':  {'cagr': 0.0889, 'sharpe': -0.08,  'note': 'JQ fa_v711_large_dominant'},
    'small_bull_all':      {'cagr': -0.5795,'sharpe': -1.01,  'note': 'JQ fa_v711_small_bull'},
    # v7.9 high_quality 作为参考
    'v79_high_quality':    {'cagr': 0.9302, 'sharpe': 0.39,   'note': 'JQ v7.9 high_quality total return'},
}

print(f"\n{'场景':<25s} {'V3年化':>8s} {'V3 SR':>7s} {'JQ年化':>8s} {'JQ SR':>7s} {'CAGR比':>7s} {'SR差':>7s}")
print("-" * 95)

calib_cagr_ratios = []
calib_sr_diffs = []

for key, jq in [('large_dominant_all', jq_known['large_dominant_all']),
                 ('small_bull_all', jq_known['small_bull_all'])]:
    v3 = results.get(key, {})
    v3_cagr = v3.get('cagr', np.nan)
    v3_sr = v3.get('sharpe', np.nan)
    jq_cagr = jq['cagr']
    jq_sr = jq['sharpe']

    cagr_ratio = jq_cagr / v3_cagr if v3_cagr != 0 else np.nan
    sr_diff = jq_sr - v3_sr

    if not np.isnan(cagr_ratio):
        calib_cagr_ratios.append(cagr_ratio)
    if not np.isnan(sr_diff):
        calib_sr_diffs.append(sr_diff)

    print(f"{key:<25s} {v3_cagr:>+8.1%} {v3_sr:>+7.2f} {jq_cagr:>+8.1%} {jq_sr:>+7.2f} "
          f"{cagr_ratio:>+7.2f} {sr_diff:>+7.2f}")

# 计算校准参数
calib_cagr_mult = np.mean(calib_cagr_ratios) if calib_cagr_ratios else 1.0
calib_sr_offset = np.mean(calib_sr_diffs) if calib_sr_diffs else 0.0
calib_sr_mult = np.mean([jq_known['large_dominant_all']['sharpe'] / results['large_dominant_all']['sharpe']
                          if results['large_dominant_all']['sharpe'] != 0 else 1.0,
                          jq_known['small_bull_all']['sharpe'] / results['small_bull_all']['sharpe']
                          if results['small_bull_all']['sharpe'] != 0 else 1.0])

print(f"\n  校准参数: CAGR_mult={calib_cagr_mult:.3f}, SR_mult={calib_sr_mult:.3f}, SR_offset={calib_sr_offset:+.3f}")

# 校准后的 merged 预测
v3_merged = results['merged']
merged_cagr_calib = v3_merged.get('cagr', 0) * calib_cagr_mult
merged_sr_calib = v3_merged.get('sharpe', 0) * calib_sr_mult
print(f"\n  ★ 校准预测: merged CAGR={merged_cagr_calib:+.1%}, SR={merged_sr_calib:+.2f} "
      f"(原始 V3: CAGR={v3_merged.get('cagr',0):+.1%}, SR={v3_merged.get('sharpe',0):+.2f})")

# =====================================================
# Part E: 保存结果
# =====================================================
out = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'method': 'PortfolioSimulator_v3_fixed',
    'fixes': [
        'turnover过滤后重算added/removed → 成本对齐实际持仓',
        'slippage改为per-share模型 (JQ FixedSlippage(0.003)对齐)',
        '测试口径: 同时测all-weeks和regime-specific',
    ],
    'calibration': {
        'cagr_mult': calib_cagr_mult,
        'sr_mult': calib_sr_mult,
        'sr_offset': calib_sr_offset,
        'note': '基于LD_all+SB_all两个JQ真值拟合',
    },
    'results_v3': {k: v for k, v in results.items()},
    'results_v2': {k: v for k, v in results_v2.items()},
    'jq_reference': jq_known,
    'calibrated_merged': {
        'cagr': merged_cagr_calib,
        'sharpe': merged_sr_calib,
    },
}
with open(SCRIPT_DIR / 'output' / 'rebacktest_v711_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)

print(f"\n  保存: output/rebacktest_v711_fixed.json")
print("\n完成!")
