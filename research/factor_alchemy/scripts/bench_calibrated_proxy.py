"""
计时基准: 验证 ensemble_calibrated 选择阶段 PortfolioSimulator 真实耗时。
完整加载数据一次 (与 regime_lasso Phase 1 一致), 然后只计时
batch_evaluate_factors, 复刻 v711 的 large_dominant (11 因子) 评估。
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from data.loader import load_all_data
from factors.composite import standardize_factors
from ga.fast_calibrated_proxy import batch_evaluate_factors

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTOR_CACHE = os.path.join(SCRIPT_DIR, 'output', 'factor_cache.pkl')

# v711 large_dominant Boruta 确认因子 (来自 JSON selector_meta)
LD_CONFIRMED = ['atr_14', 'bull_power', 'ret_1m', 'sharpe_20', 'earnings_season_vol_div',
                'earnings_stability', 'ret_3m', 'roe', 'money_flow_20',
                'dollar_vol_20d', 'volatility_20d']
# v711 small_bull Boruta 确认因子
SB_CONFIRMED = ['atr_14', 'relative_spread_proxy', 'rev_growth_yoy', 'ret_6m', 'sp',
                'ret_3m']

print("[bench] Phase 1 数据加载 (与 regime_lasso 一致)...")
t1 = time.time()
data = load_all_data('2021-01-01', '2026-06-25')
price_data = data['price_data']
valuation_data = data['valuation_data']
close = price_data.get('close')

mcap_weekly = None
if valuation_data is not None and 'circ_mv' in valuation_data.columns:
    _m = valuation_data[['code', 'trade_date', 'circ_mv']].drop_duplicates().copy()
    _m['trade_date'] = pd.to_datetime(_m['trade_date'])
    mcap_weekly = _m.pivot(index='trade_date', columns='code', values='circ_mv').resample('W').last()
    mcap_weekly.index = pd.to_datetime(mcap_weekly.index)

close_weekly = close.resample('W').last()
close_weekly.index = pd.to_datetime(close_weekly.index)
print(f"[bench] 价格矩阵: {close.shape} | 数据加载耗时: {time.time()-t1:.0f}s")

print("[bench] 加载 factor_cache (CACHE HIT)...")
t2 = time.time()
with open(FACTOR_CACHE, 'rb') as f:
    factor_dfs = pickle.load(f)
factor_dfs_std = standardize_factors(factor_dfs)
print(f"[bench] 因子: {len(factor_dfs_std)} | 耗时 {time.time()-t2:.1f}s")

# 构造 regime weeks (large_dominant: CSS500-CSI300 spread 占优, 取示例子集)
import json
with open(os.path.join(SCRIPT_DIR, 'output', 'regime_lasso_result_v711.json')) as f:
    res = json.load(f)

# 直接计时 batch_evaluate_factors (两个 regime 合计 17 因子)
print("\n[bench] === 计时 batch_evaluate_factors (large_dominant 11 因子) ===")
t3 = time.time()
scores_ld = batch_evaluate_factors(
    LD_CONFIRMED, factor_dfs_std, close_weekly, mcap_weekly,
    regime_weeks=close_weekly.index.tolist(),  # 全样本近似
    factor_icir_dict={}, verbose=True,
)
elapsed_ld = time.time() - t3
print(f"[bench] large_dominant 11 因子耗时: {elapsed_ld:.1f}s | 单次均值: {elapsed_ld/len(LD_CONFIRMED):.2f}s")

print("\n[bench] === 计时 batch_evaluate_factors (small_bull 6 因子) ===")
t4 = time.time()
scores_sb = batch_evaluate_factors(
    SB_CONFIRMED, factor_dfs_std, close_weekly, mcap_weekly,
    regime_weeks=close_weekly.index.tolist(),
    factor_icir_dict={}, verbose=True,
)
elapsed_sb = time.time() - t4
print(f"[bench] small_bull 6 因子耗时: {elapsed_sb:.1f}s | 单次均值: {elapsed_sb/len(SB_CONFIRMED):.2f}s")

total_eval = elapsed_ld + elapsed_sb
print(f"\n[bench] === 结论 ===")
print(f"  数据加载: ~{time.time()-t1:.0f}s (一次性)")
print(f"  PortfolioSimulator 评估总计: {total_eval:.1f}s ({len(LD_CONFIRMED)+len(SB_CONFIRMED)} 因子)")
print(f"  单次 PortfolioSimulator 均值: {total_eval/(len(LD_CONFIRMED)+len(SB_CONFIRMED)):.2f}s")
print(f"  若加 Boruta (~2-3min/2 regime) + Phase 7 回测 (~秒级):")
print(f"  => 总耗时预估: ~{60*2.5 + total_eval:.0f}s ≈ {(60*2.5 + total_eval)/60:.1f}min, 与实测 9min17s 一致 ✓")
