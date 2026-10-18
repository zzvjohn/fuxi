"""
P0 分域ICIR检验 + 快速FA对比 (2026-07-12)
=========================================
基于 run_fa.py 精简:
  - Phase 1/2/3: 数据加载 + 86因子计算 + ICIR/相关性
  - Phase 4 前半: screen_factors 两阶段筛选 → 候选池(~33)
  - [P0新增]: 对候选池每只因子计算 市值三分域 / 波动率三分域 的 RankIC-IR
             P0 pass = 在两类域中都"多数子域同正"(cap_pos>=2 且 vol_pos>=2)
             打印并保存诊断表; 用 P0 保留池替换 GA 因子空间
  - Phase 4 GA: 降低预算(30pop×8gen)提速
  - Phase 4.5/5/5.5: 解码帕累托 → 最终因子组合 + 训练/OOS 成本后回测
  - 对比基线: v7.4 文档结果(tc_cv+f_score+dvol+bp, +86.69%)

用法:
  cd e:/quant/research/factor_alchemy
  python scripts/p0_run_fa.py
"""
import sys, time, os, pickle, traceback
from pathlib import Path
from datetime import datetime

FA_DIR = Path(r'e:/quant/research/factor_alchemy')
sys.path.insert(0, str(FA_DIR))
sys.path.insert(0, str(FA_DIR.parent))

# 编码修复
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd
from math import comb
import warnings
warnings.filterwarnings('ignore')

from config import FACTOR_DEFS, OOS_SPLIT_DATE
from factors import ALL_FACTORS
from data.loader import load_all_data

t_total = time.time()
print("=" * 70)
print("  P0 分域ICIR检验 + 快速FA对比 (v7.4管线精简版)")
print(f"  OOS: {OOS_SPLIT_DATE}")
print("=" * 70)

# =====================================================
# Phase 1: 数据加载
# =====================================================
print("\n[Phase 1] 加载数据 (2021-01-01 ~ 2026-06-25)")
t1 = time.time()
data = load_all_data('2021-01-01', '2026-06-25')
price_data = data['price_data']
financial_data = data['financial_data']
valuation_data = data['valuation_data']
industry_map = data.get('industry_map', {})

close = price_data.get('close')
close_weekly = close.resample('W').last()
close_weekly.index = pd.to_datetime(close_weekly.index)
forward_returns = close_weekly.shift(-1) / close_weekly - 1
print(f"  周频收益: {forward_returns.shape} ({forward_returns.index[0].strftime('%Y-%m-%d')} ~ {forward_returns.index[-1].strftime('%Y-%m-%d')})")

mcap_df = None
if valuation_data is not None and 'market_cap' in valuation_data.columns:
    mcap_raw = valuation_data[['code', 'trade_date', 'market_cap']].copy()
    mcap_raw['trade_date'] = pd.to_datetime(mcap_raw['trade_date'])
    mcap_df = mcap_raw.pivot(index='trade_date', columns='code', values='market_cap')
    mcap_df = mcap_df.resample('W').last()
    mcap_df = np.log(mcap_df.clip(lower=1))
    print(f"  市值(log): {mcap_df.shape}")
print(f"  耗时: {time.time()-t1:.0f}s")

# =====================================================
# Phase 2: 因子计算
# =====================================================
print("\n[Phase 2] 因子计算")
factor_dfs = {}
t2 = time.time()
for name in FACTOR_DEFS:
    if name not in ALL_FACTORS:
        continue
    factor_def = FACTOR_DEFS[name]
    factor_obj = ALL_FACTORS[name]()
    try:
        df = factor_obj.compute(
            price_data=price_data, financial_data=financial_data,
            valuation_data=valuation_data, industry_map=industry_map,
        )
        if df is not None and not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df_weekly = df.resample('W').last()
            is_financial = df_weekly.shape[0] <= 10 and factor_def['category'] in ('profitability', 'value', 'growth')
            if is_financial:
                df_weekly = df_weekly.reindex(close_weekly.index).ffill().bfill()
            else:
                common_idx = close_weekly.index.intersection(df_weekly.index)
                if len(common_idx) > 0:
                    df_weekly = df_weekly.reindex(common_idx)
            factor_dfs[name] = df_weekly
    except Exception as e:
        print(f"  {name} FAIL {e}")
print(f"  成功: {len(factor_dfs)}/{len(FACTOR_DEFS)} 因子 ({time.time()-t2:.0f}s)")

# =====================================================
# Phase 3: 单因子 ICIR + 相关性
# =====================================================
print("\n[Phase 3] 单因子 ICIR + 相关性")
from evaluation.ic_analysis import compute_ic_summary
ic_summary = compute_ic_summary(factor_dfs, forward_returns)
from evaluation.correlation import factor_correlation_matrix
corr_matrix = factor_correlation_matrix(factor_dfs)
valid_factor_names = [n for n in factor_dfs if n in ic_summary.index and not np.isnan(ic_summary.loc[n, 'ICIR'])]
print(f"  有效因子: {len(valid_factor_names)}")

# 预标准化 + 预计算IC序列 (GA用)
from evaluation.ic_analysis import compute_ic_icir
from factors.composite import standardize_factors
factor_dfs_std = standardize_factors(factor_dfs)
factor_ic_series = {}
for name in factor_dfs:
    if name not in ic_summary.index:
        continue
    try:
        icr = compute_ic_icir(factor_dfs[name], forward_returns)
        ic_s = icr.get('ic_series', pd.Series(dtype=float))
        if len(ic_s) > 50:
            factor_ic_series[name] = ic_s
    except Exception:
        pass

# =====================================================
# Phase 4 前半: 因子预筛选 (两阶段) → 候选池
# =====================================================
print("\n[Phase 4] 因子预筛选 (ICIR>=0.1 → 聚类 → WFA)")
from ga.factor_screening import screen_factors, print_screening_summary
_screened, _scr_stats = screen_factors(
    valid_factor_names, ic_summary, corr_matrix, factor_dfs, forward_returns,
    icir_threshold=0.1, corr_threshold=0.5, wfa_windows=8, wfa_min_pass=3, wfa_icir_threshold=0.1,
    verbose=True,
)
print_screening_summary(_scr_stats, _screened)
candidate_pool = _screened
print(f"  ★ 候选池({len(candidate_pool)}因子)")

# =====================================================
# [P0] 分域ICIR检验 — 市值/波动率三分域
# =====================================================
print("\n" + "=" * 70)
print("[P0] 分域ICIR检验 (市值三分域 + 波动率三分域, 全周期)")
print("=" * 70)
from ga.domain_split import rankic_per_domain

def build_domain_panel(control_panel, n_domains=3):
    """每周边按 control 分位切成 n_domains 域 (0=最低)."""
    dom_rows = []
    for t in control_panel.index:
        x = control_panel.loc[t]
        valid = ~x.isna()
        if valid.sum() < 50:
            dom_rows.append(pd.Series(np.nan, index=control_panel.columns))
            continue
        try:
            q = pd.qcut(x[valid], n_domains, labels=False, duplicates='drop')
            s = pd.Series(np.nan, index=control_panel.columns)
            s[valid] = q.values
            dom_rows.append(s)
        except Exception:
            dom_rows.append(pd.Series(np.nan, index=control_panel.columns))
    return pd.DataFrame(dom_rows, index=control_panel.index, columns=control_panel.columns)

cap_dom = build_domain_panel(mcap_df.reindex(forward_returns.index), 3).reindex(columns=forward_returns.columns)
vol_panel = close_weekly.pct_change().rolling(12).std()
vol_dom = build_domain_panel(vol_panel.reindex(forward_returns.index), 3).reindex(columns=forward_returns.columns)
print(f"  cap域有效覆盖: {cap_dom.notna().mean().mean():.1%} | vol域有效覆盖: {vol_dom.notna().mean().mean():.1%}")

def domain_icirs(name):
    f = factor_dfs[name].reindex(index=forward_returns.index, columns=forward_returns.columns)
    cap_r = rankic_per_domain(f, forward_returns, cap_dom, 3)
    vol_r = rankic_per_domain(f, forward_returns, vol_dom, 3)
    cap = [cap_r.loc[d, 'ICIR'] for d in range(3)]
    vol = [vol_r.loc[d, 'ICIR'] for d in range(3)]
    return cap, vol

p0_rows = []
for name in candidate_pool:
    cap, vol = domain_icirs(name)
    full = ic_summary.loc[name, 'ICIR'] if name in ic_summary.index else np.nan
    cap_pos = sum(1 for x in cap if pd.notna(x) and x > 0)
    vol_pos = sum(1 for x in vol if pd.notna(x) and x > 0)
    cap_valid = sum(1 for x in cap if pd.notna(x))
    vol_valid = sum(1 for x in vol if pd.notna(x))
    # P0 pass: 两类域都"多数子域同正" (>=2/3 正)
    p0_pass = (cap_valid >= 2 and cap_pos >= 2) and (vol_valid >= 2 and vol_pos >= 2)
    p0_rows.append({
        'factor': name, 'full_ICIR': round(full, 4) if pd.notna(full) else np.nan,
        'cap_IC_s': cap[0], 'cap_IC_m': cap[1], 'cap_IC_l': cap[2],
        'vol_IC_l': vol[0], 'vol_IC_m': vol[1], 'vol_IC_h': vol[2],
        'cap_pos': f"{cap_pos}/{cap_valid}", 'vol_pos': f"{vol_pos}/{vol_valid}",
        'p0_pass': p0_pass,
    })

p0_df = pd.DataFrame(p0_rows).sort_values('full_ICIR', key=lambda s: s.abs(), ascending=False)
p0_out = FA_DIR / 'output' / 'p0_domain_icir.csv'
p0_out.parent.mkdir(exist_ok=True)
p0_df.to_csv(p0_out, encoding='utf-8-sig', index=False)
print(f"\n  P0 诊断表已保存: {p0_out}")
print(f"  {'因子':22s} {'fullICIR':>9s} {'cap(小/中/大)':>24s} {'vol(低/中/高)':>24s} {'P0':>4s}")
print(f"  {'-'*85}")
for _, r in p0_df.iterrows():
    def fmt(x):
        return f"{x:+.3f}" if pd.notna(x) else "  n/a"
    print(f"  {r['factor']:22s} {r['full_ICIR']:>9.4f} "
          f"{fmt(r['cap_IC_s']):>8s}{fmt(r['cap_IC_m']):>8s}{fmt(r['cap_IC_l']):>8s} "
          f"{fmt(r['vol_IC_l']):>8s}{fmt(r['vol_IC_m']):>8s}{fmt(r['vol_IC_h']):>8s} "
          f"{'✅' if r['p0_pass'] else '❌':>4s}")

p0_keep = [r['factor'] for r in p0_rows if r['p0_pass']]
p0_drop = [r['factor'] for r in p0_rows if not r['p0_pass']]
print(f"\n  ★ P0 保留: {len(p0_keep)} | 剔除: {len(p0_drop)}")
print(f"  ★ 剔除因子: {p0_drop}")
# 与v7.4基线对比
v74_base = ['tc_cv', 'f_score', 'dvol', 'bp']
print(f"  v7.4基线因子存活检查: " + ", ".join(
    f"{f}{'✅' if f in p0_keep else '❌剔除'}" for f in v74_base))

# =====================================================
# Phase 4 GA: 用 P0 保留池
# =====================================================
print("\n" + "=" * 70)
print(f"[Phase 4 GA] NSGA-II 三目标 (P0保留池, 快速预算 30pop×8gen)")
print("=" * 70)

ga_factor_names = p0_keep
n_factors = len(ga_factor_names)
print(f"  GA 因子空间: {n_factors} 因子")

# 重建筛选后相关性子矩阵
_common = [n for n in ga_factor_names if n in corr_matrix.index]
corr_matrix_full = corr_matrix.loc[_common, _common] if len(_common) >= 2 else corr_matrix

# OOS 分割
fr_index = pd.DatetimeIndex(forward_returns.index)
train_mask = fr_index < pd.Timestamp(OOS_SPLIT_DATE)
oos_mask = fr_index >= pd.Timestamp(OOS_SPLIT_DATE)
train_fr = forward_returns.loc[train_mask]
oos_fr = forward_returns.loc[oos_mask]

# 训练集IC序列
oos_dt = pd.Timestamp(OOS_SPLIT_DATE)
train_ic = {}
for name, ic_series in factor_ic_series.items():
    if isinstance(ic_series, pd.Series) and len(ic_series) > 0:
        ic_train = ic_series[ic_series.index < oos_dt].dropna()
        if len(ic_train) > 10:
            train_ic[name] = ic_train

# ICIR偏置初始化
from ga.chromosome import icir_biased_chromosome
icir_list = [abs(ic_summary.loc[n, 'ICIR']) if n in ic_summary.index else 0.0 for n in ga_factor_names]
icir_arr = np.array(icir_list)
pop = [icir_biased_chromosome(n_factors, icir_arr, bias_strength=1.0) for _ in range(30)]

from ga.nsga2 import nsga2_run
from config import MAX_FACTORS
effective_max_factors = MAX_FACTORS
t4 = time.time()
print(f"  === NSGA-II 启动 (30pop × 8gen, {n_factors}因子) ===")
try:
    final_population, final_objectives, pareto_front = nsga2_run(
        pop, ga_factor_names, factor_dfs, train_fr, mcap_df,
        n_generations=8, pop_size=30,
        crossover_prob=0.7, mutation_prob=0.2, mutation_sigma=0.1,
        corr_matrix=corr_matrix_full, ic_summary=ic_summary,
        factor_ic_series=train_ic, max_factors=effective_max_factors,
        verbose=True, factor_dfs_std=factor_dfs_std, debug_obj3=True,
    )
    print(f"\n  NSGA-II 耗时: {time.time()-t4:.0f}s")
except Exception as e:
    print(f"\n  [FATAL] NSGA-II 崩溃: {e}")
    traceback.print_exc()
    pareto_front = {'objectives': np.empty((0, 3)), 'chromosomes': [], 'weights': [], 'details': []}
    final_population, final_objectives = [], np.empty((0, 3))

# 解码帕累托前沿
from factors.composite import weights_from_chromosome, combine_factors_vectorized
pareto_solutions = []
n_pareto = len(pareto_front['weights'])
for i, (obj, w) in enumerate(zip(pareto_front['objectives'], pareto_front['weights'])):
    try:
        sel = {n: factor_dfs_std[n] for n in w if n in factor_dfs_std}
        comp = combine_factors_vectorized(sel, w)
        if comp.empty or comp.shape[0] < 10:
            continue
        icir_train = abs(compute_ic_icir(comp, train_fr)['icir'])
        from ga.fitness import _estimate_weekly_turnover
        est_to = _estimate_weekly_turnover(comp, top_n=30, buffer_mult=1.8)
        pareto_solutions.append({
            'idx': i, 'quality_obj': float(obj[0]), 'stability_obj': float(obj[1]),
            'sharpe_obj': float(obj[2]), 'icir_train': icir_train, 'weights': w,
            'estimated_to': est_to, 'composite': comp,
        })
    except Exception as e:
        print(f"  [WARN] 前沿解#{i} 验证失败: {e}")

print(f"\n  === 帕累托前沿 ({len(pareto_solutions)} 个解) [训练集] ===")
for ps in sorted(pareto_solutions, key=lambda x: -x['quality_obj']):
    print(f"  #{ps['idx']:3d} Q={ps['quality_obj']:.3f} S={ps['stability_obj']:.3f} "
          f"C={ps['sharpe_obj']:.3f} ICIR_train={ps['icir_train']:.3f} TO={ps['estimated_to']:.1%} N={len(ps['weights'])}")

# =====================================================
# Phase 4.5: 选最优解 + 最终因子组合
# =====================================================
print("\n" + "=" * 70)
print("[Phase 4.5] 帕累托选点 + 最终因子组合")
print("=" * 70)
if pareto_solutions:
    high_quality = max(pareto_solutions, key=lambda x: x['quality_obj'])
    high_sharpe = max(pareto_solutions, key=lambda x: x['sharpe_obj'])
    high_stable = max(pareto_solutions, key=lambda x: x['stability_obj'])
    balanced = max(pareto_solutions, key=lambda x:
                   (max(x['quality_obj'], 0.001) ** 0.4) * (max(x['stability_obj'], 0.001) ** 0.3) *
                   (max(x['sharpe_obj'], 0.001) ** 0.3))
    strategies = [('high_quality', high_quality), ('high_sharpe', high_sharpe),
                  ('high_stable', high_stable), ('balanced', balanced)]
    best_weights_all = {}
    for strat_name, ps in strategies:
        w = ps['weights']
        composite = ps['composite']
        print(f"\n  [{strat_name}] 因子数={len(w)}")
        for name, wt in sorted(w.items(), key=lambda x: -x[1]):
            ic_val = ic_summary.loc[name].get('ICIR', 0) if name in ic_summary.index else 0
            print(f"    {name:25s} w={wt:7.3f}  ICIR={ic_val:+.3f}")
        best_weights_all[strat_name] = {'weights': w, 'composite': composite}
else:
    print("  [WARN] 无前沿解")

# =====================================================
# Phase 5: 虚拟组合 (QMT成本) + Phase 5.5 OOS
# =====================================================
print("\n" + "=" * 70)
print("[Phase 5 + 5.5] 成本后回测 (训练 + OOS, QMT成本)")
print("=" * 70)
QMT_COMMISSION = 0.0001
QMT_STAMP_TAX = 0.001
QMT_SLIPPAGE = 0.003
from portfolio.simulator import PortfolioSimulator

if 'best_weights_all' in dir():
    for strat_name in ['high_sharpe', 'high_quality', 'balanced', 'high_stable']:
        if strat_name not in best_weights_all:
            continue
        info = best_weights_all[strat_name]
        composite = info['composite']
        for label, fr, close_w in [('TRAIN', train_fr, close_weekly.loc[train_fr.index]),
                                    ('OOS', oos_fr, close_weekly.loc[oos_fr.index])]:
            try:
                comp_sub = composite.reindex(fr.index).dropna(how='all')
                if comp_sub.empty:
                    continue
                comp_sub = comp_sub.where(close_w.notna(), np.nan)
                sim = PortfolioSimulator(
                    factor_df=comp_sub, price_df=close_w, top_n=30, weighting='equal',
                    max_turnover=0.50, commission=QMT_COMMISSION, stamp_tax=QMT_STAMP_TAX,
                    slippage=QMT_SLIPPAGE,
                )
                res = sim.run()
                s = res.get('stats', {})
                print(f"  [{strat_name}/{label}] 总收益={s.get('total_return',0):.1%} "
                      f"年化={s.get('cagr',0):.1%} Sharpe={s.get('sharpe',0):.2f} "
                      f"MaxDD={s.get('max_drawdown',0):.1%} 换手={s.get('avg_turnover',0):.1%}")
            except Exception as e:
                print(f"  [{strat_name}/{label}] ERROR: {e}")

# 保存
if not p0_df.empty:
    p0_df.to_csv(p0_out, encoding='utf-8-sig', index=False)
    print(f"\n  P0表(含决策)已更新: {p0_out}")

total_time = time.time() - t_total
m, s = divmod(total_time, 60)
h, m = divmod(m, 60)
print(f"\n  总耗时: {int(h)}h{int(m)}m{int(s)}s")
print("  P0实验完成!")
