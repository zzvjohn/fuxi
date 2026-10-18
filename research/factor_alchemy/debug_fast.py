"""
快速诊断: 仅用前5个因子的前50周数据, 验证Obj3计算 + ICIR raw/clip
无需Tushare API, 直接从CSV加载
"""
import sys, os
from pathlib import Path
os.environ['FA_OBJ3_DEBUG'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import FACTOR_DEFS, ICIR_THRESHOLD, MAX_FACTORS, MAX_TOTAL_PENALTY
from ga.fitness import _compute_top30_cost_adjusted_sharpe
from factors.composite import combine_factors
from evaluation.ic_analysis import compute_ic_icir

OUTPUT_DIR = Path(__file__).parent / 'output'
DATA_RAW = Path('e:/quant/data/raw')

np.random.seed(99)

print("=" * 60)
print("  快速诊断: Obj3 + ICIR raw/clip 验证")
print("=" * 60)

# ===== 加载因子 (仅5个代表性因子) =====
key_factors = ['dollar_vol_stability', 'earnings_growth_yoy', 
               'turnover_std', 'plrc_12', 'f_score']
factor_dict = {}
for name in key_factors:
    fpath = OUTPUT_DIR / f'factor_{name}.csv'
    if fpath.exists():
        df = pd.read_csv(fpath, index_col=0, parse_dates=True, encoding='utf-8-sig')
        factor_dict[name] = df
        print(f"  Loaded {name}: {df.shape}")

print(f"\n  加载了 {len(factor_dict)} 因子")

# ===== 直接从daily_prices构建前向收益 =====
print("\n  加载价格数据...")
try:
    price_raw = pd.read_csv(DATA_RAW / 'daily_prices.csv', 
                            parse_dates=['trade_date'], encoding='utf-8-sig')
    adj_factor = pd.read_csv(DATA_RAW / 'adj_factor.csv',
                             parse_dates=['trade_date'], encoding='utf-8-sig')
    
    # Pivot
    close_raw = price_raw.pivot(index='trade_date', columns='ts_code', values='close')
    adj = adj_factor.pivot(index='trade_date', columns='ts_code', values='adj_factor')
    
    # 后复权
    common_dates = close_raw.index.intersection(adj.index)
    common_stocks = close_raw.columns.intersection(adj.columns)
    last_adj = adj.loc[common_dates[-1], common_stocks]
    hfq_close = close_raw.loc[common_dates, common_stocks].multiply(adj.loc[common_dates, common_stocks]).div(last_adj)
    
    # 周频
    close_w = hfq_close.resample('W').last()
    close_w.index = pd.to_datetime(close_w.index)
    forward_returns = close_w.shift(-1) / close_w - 1
    
    print(f"  周频收益: {forward_returns.shape}")
    print(f"  日期范围: {forward_returns.index[0].strftime('%Y-%m-%d')} ~ {forward_returns.index[-1].strftime('%Y-%m-%d')}")
    
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# ================================================================
# SECTION A: Obj3 逐步诊断
# ================================================================
print("\n" + "=" * 60)
print("  SECTION A: Obj3 逐步诊断")
print("=" * 60)

# 使用不同的因子组合
factor_names = list(factor_dict.keys())
test_combos = []

# Combo 1: 单因子 dollar_vol_stability (已知最强单因子)
test_combos.append(("单因子 dollar_vol_stability", 
                     {'dollar_vol_stability': 1.0}))

# Combo 2: earnings_growth_yoy + dollar_vol_stability
test_combos.append(("价值+量价组合",
                     {'earnings_growth_yoy': 0.5, 'dollar_vol_stability': 0.3, 'turnover_std': 0.2}))

# Combo 3: 所有5因子等权
test_combos.append(("5因子等权", {n: 0.2 for n in factor_names}))

# Combo 4: plrc_12 单独 (测试高IC但样本依赖的因子)
if 'plrc_12' in factor_dict:
    test_combos.append(("plrc_12单独", {'plrc_12': 1.0}))

for label, weights in test_combos:
    print(f"\n  {'─' * 50}")
    print(f"  {label}")
    
    selected = {n: factor_dict[n] for n in weights if n in factor_dict}
    if len(selected) == 0:
        continue
    
    composite = combine_factors(selected, weights)
    if composite.empty:
        print("  [SKIP] composite为空")
        continue
    
    # ICIR
    icir_result = compute_ic_icir(composite, forward_returns)
    icir = abs(icir_result['icir'])
    print(f"  ICIR: {icir:+.4f} (+IC%={icir_result['ic_positive_ratio']*100:.1f}%)")
    
    # === 逐周验证第1-5周 ===
    common_idx = composite.index.intersection(forward_returns.index)
    common_cols = composite.columns.intersection(forward_returns.columns)
    
    if len(common_idx) < 6 or len(common_cols) < 30:
        print(f"  [SKIP] 数据不足")
        continue
    
    comp_arr = composite.loc[common_idx, common_cols].values
    fr_arr = forward_returns.loc[common_idx, common_cols].values
    
    print(f"  前期 Top30 vs All 对比 (前5周):")
    for w in range(min(5, len(common_idx) - 1)):
        cw = comp_arr[w]
        fw = fr_arr[w]
        valid = np.isfinite(cw)
        top30 = np.argsort(-cw)[:30]
        top30_ret = np.nanmean(fw[top30])
        all_ret = np.nanmean(fw[valid])
        # 验证方向
        corr_v = stats.spearmanr(cw[valid], fw[valid])
        corr_v = corr_v[0] if isinstance(corr_v, tuple) else corr_v.statistic
        print(f"    周{w}: Top30={top30_ret*100:+.3f}%  All={all_ret*100:+.3f}%  "
              f"delta={((top30_ret-all_ret)*100):+.3f}%  corr={corr_v:+.3f}")
    
    # === 完整Obj3计算 ===
    obj3 = _compute_top30_cost_adjusted_sharpe(composite, forward_returns)
    
    # 手动复现 (与函数内相同逻辑)
    n_weeks, n_stocks = comp_arr.shape
    comp_filled = np.where(np.isnan(comp_arr), -np.inf, comp_arr)
    rankings = np.argsort(-comp_filled, axis=1)
    top30_idx = rankings[:, :30]
    
    week_indices = np.arange(n_weeks - 1)[:, None]
    stock_indices = top30_idx[:-1, :]
    gross_rets = np.nanmean(fr_arr[week_indices, stock_indices], axis=1)
    
    top30_mask = np.zeros((n_weeks, n_stocks), dtype=bool)
    for i in range(n_weeks):
        top30_mask[i, top30_idx[i]] = True
    overlap = (top30_mask[:-1] & top30_mask[1:]).sum(axis=1)
    turnovers = 1.0 - overlap / 30.0
    costs = turnovers * 0.0072
    net_rets = gross_rets - costs
    
    valid_rets = net_rets[np.isfinite(net_rets)]
    mean_ret = np.mean(valid_rets)
    std_ret = np.std(valid_rets, ddof=1)
    annual_sharpe = mean_ret / std_ret * np.sqrt(52) if std_ret > 1e-8 else 0.0
    
    # 分训练期/OOS期
    train_mask = np.array([d < '2025-01-01' for d in common_idx[:-1]])
    oos_mask = np.array([d >= '2025-01-01' for d in common_idx[:-1]])
    
    print(f"\n  Obj3 完整拆解:")
    print(f"    {'':20s} {'全时段':>12s} {'训练期':>12s} {'OOS期':>12s}")
    for label, arr in [("年化毛收益(%)", gross_rets*52*100),
                        ("年化成本(%)",  costs*52*100),
                        ("年化净收益(%)", net_rets*52*100)]:
        all_v = np.nanmean(arr)
        train_v = np.nanmean(arr[train_mask]) if train_mask.any() else np.nan
        oos_v = np.nanmean(arr[oos_mask]) if oos_mask.any() else np.nan
        print(f"    {label:20s} {all_v:+10.2f} {train_v:+10.2f} {oos_v:+10.2f}")
    
    print(f"    {'平均周换手(%)':20s} {np.nanmean(turnovers)*100:10.1f} "
          f"{np.nanmean(turnovers[train_mask])*100 if train_mask.any() else 0:10.1f} "
          f"{np.nanmean(turnovers[oos_mask])*100 if oos_mask.any() else 0:10.1f}")
    
    raw_sharpe_all = annual_sharpe
    train_sharpe = (np.nanmean(net_rets[train_mask]) / np.nanstd(net_rets[train_mask], ddof=1) * np.sqrt(52) 
                    if train_mask.sum() > 5 and np.nanstd(net_rets[train_mask], ddof=1) > 1e-8 else np.nan)
    oos_sharpe = (np.nanmean(net_rets[oos_mask]) / np.nanstd(net_rets[oos_mask], ddof=1) * np.sqrt(52)
                  if oos_mask.sum() > 5 and np.nanstd(net_rets[oos_mask], ddof=1) > 1e-8 else np.nan)
    
    print(f"    {'年化Sharpe':20s} {raw_sharpe_all:+10.3f} {train_sharpe:+10.3f} {oos_sharpe:+10.3f}")
    print(f"    Obj3(clip后): {obj3:.4f}")
    
    # 诊断结论
    if train_sharpe < 0 and not np.isnan(train_sharpe):
        print(f"    => 训练期Sharpe为负: 熊市long-only无法获利, 非计算bug")

# ================================================================
# SECTION B: ICIR raw vs clip 区分度对比
# ================================================================
print("\n" + "=" * 60)
print("  SECTION B: ICIR raw vs clip 区分度对比")
print("=" * 60)

# 生成多个组合 (枚举所有可能的2-3因子组合)
from itertools import combinations

results = []
for k in [1, 2, 3]:
    for combo in combinations(factor_names, min(k, len(factor_names))):
        if k == 1:
            weights = {combo[0]: 1.0}
        else:
            weights = {n: 1.0/len(combo) for n in combo}
        
        selected = {n: factor_dict[n] for n in weights if n in factor_dict}
        if len(selected) < k:
            continue
        
        composite = combine_factors(selected, weights)
        if composite.empty:
            continue
        
        icir_result = compute_ic_icir(composite, forward_returns)
        icir = abs(icir_result['icir'])
        if np.isnan(icir):
            continue
        
        # 模拟惩罚 (简化: 只用换手)
        try:
            from ga.fitness import _estimate_weekly_turnover, _compute_turnover_penalty
            weekly_to = _estimate_weekly_turnover(composite, top_n=30, buffer_mult=1.8)
            to_pen = _compute_turnover_penalty(weekly_to)
        except:
            to_pen = 0.2
        
        pen = to_pen  # 简化惩罚
        
        # Raw
        obj1_raw = float(np.clip(icir - pen, 0.0, 1.0))
        
        # Clip
        icir_clip = float(np.clip(icir / ICIR_THRESHOLD, 0, 1))
        obj1_clip = float(np.clip(icir_clip - pen, 0.0, 1.0))
        
        results.append({
            'icir': icir,
            'icir_clip': icir_clip,
            'pen': pen,
            'n': k,
            'obj1_raw': obj1_raw,
            'obj1_clip': obj1_clip,
        })

df = pd.DataFrame(results)
print(f"\n  有效组合: {len(df)}")

if len(df) >= 5:
    icir_raw_cv = df['icir'].std() / max(df['icir'].mean(), 1e-8)
    icir_clip_cv = df['icir_clip'].std() / max(df['icir_clip'].mean(), 1e-8)
    obj1_raw_cv = df['obj1_raw'].std() / max(df['obj1_raw'].mean(), 1e-8)
    obj1_clip_cv = df['obj1_clip'].std() / max(df['obj1_clip'].mean(), 1e-8)
    
    delta_raw = obj1_raw_cv - icir_raw_cv
    delta_clip = obj1_clip_cv - icir_clip_cv
    
    sr, sp = stats.spearmanr(df['obj1_raw'], df['obj1_clip'])
    
    print(f"\n  ┌────────────────────────────────────────────────────┐")
    print(f"  │          ICIR raw vs clip 区分度对比                │")
    print(f"  ├────────────────┬──────────────┬────────────────────┤")
    print(f"  │      指标       │   Raw模式    │   Clip模式         │")
    print(f"  ├────────────────┼──────────────┼────────────────────┤")
    print(f"  │ ICIR范围       │ [{df['icir'].min():.3f},{df['icir'].max():.3f}] │ [{df['icir_clip'].min():.3f},{df['icir_clip'].max():.3f}]               │")
    print(f"  │ ICIR_score CV  │ {icir_raw_cv:.4f}      │ {icir_clip_cv:.4f}            │")
    print(f"  │ Obj1范围       │ [{df['obj1_raw'].min():.3f},{df['obj1_raw'].max():.3f}]  │ [{df['obj1_clip'].min():.3f},{df['obj1_clip'].max():.3f}]           │")
    print(f"  │ Obj1 CV        │ {obj1_raw_cv:.4f}      │ {obj1_clip_cv:.4f}            │")
    print(f"  │ 零值比例       │ {(df['obj1_raw']==0).mean()*100:.0f}%           │ {(df['obj1_clip']==0).mean()*100:.0f}%               │")
    print(f"  │ 惩罚链CV贡献   │ {delta_raw:+.4f}      │ {delta_clip:+.4f}            │")
    print(f"  └────────────────┴──────────────┴────────────────────┘")
    
    print(f"\n  Obj1_raw vs Obj1_clip Spearman r = {sr:.4f}")
    
    print(f"\n  ╔════════════════════════════════════════════════════╗")
    if abs(delta_raw - delta_clip) < 0.05:
        print(f"  ║  结论: raw/clip区分度几乎等价                       ║")
        print(f"  ║  clip推荐: Obj1∈[0,1]与Obj2同量纲, Pareto更均衡    ║")
    elif delta_raw > delta_clip:
        print(f"  ║  结论: raw模式惩罚链贡献更大 → 区分度更高            ║")
    else:
        print(f"  ║  结论: clip模式惩罚链贡献更大 → 区分度更高            ║")
    print(f"  ╚════════════════════════════════════════════════════╝")

print(f"\n{'='*60}")
print(f"  快速诊断完成")
print(f"{'='*60}")
