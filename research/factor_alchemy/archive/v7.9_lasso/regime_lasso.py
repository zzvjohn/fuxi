#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regime-aware Lasso 因子选择 + ICIR 加权
=========================================
替代 NSGA-II 的确定性方法。

流程:
  1. 加载数据 + 计算因子 (复用 factor_alchemy 模块)
  2. 标准化 (cross-sectional z-score)
  3. 加载 regime 标签, 对齐到因子周频日期
  4. 按 regime 切分:
     - ICIR >= 0.1 预筛选 (regime 内)
     - LassoCV 5-fold 特征选择 (因子 -> forward_return)
     - ICIR 加权 (|ICIR| 归一化为权重, 方向由 Lasso 系数符号决定)
  5. Walk-forward OOS 验证 (前70%训练, 后30%验证)
  6. 产出 JQ 策略 (仅对 JQ_FACTOR_REGISTRY 中有映射的因子)

用法:
  cd e:/quant && python research/factor_alchemy/regime_lasso.py

输出:
  - output/regime_lasso_result.json  (策略权重 + 回测指标)
  - strategies/drafts/fa_v79_lasso_*.py (JQ 策略文件)
"""
import sys, os, time, json, pickle, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ── 路径 ──
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data.loader import load_all_data
from factors import ALL_FACTORS, get_factor_instance
from config import FACTOR_DEFS
from factors.composite import standardize_factors
from evaluation.ic_analysis import compute_ic_icir

# ── JQ 生成 ──
try:
    from jq_generator import JQGenerator
    from jq_generator.registry import JQ_FACTOR_REGISTRY
    JQ_AVAILABLE = True
except Exception as e:
    print(f"[WARN] JQ 生成器不可用: {e}")
    JQ_AVAILABLE = False

# =====================================================
# 配置
# =====================================================
OUTPUT_DIR = SCRIPT_DIR / 'output'
STRATEGIES_DIR = SCRIPT_DIR.parent / 'strategies' / 'drafts'
REGIME_CSV = SCRIPT_DIR / 'data' / 'regime_labels.csv'

# 因子计算跳过列表 (已知崩溃)
SKIP_FACTORS = {'beta', 'idio_vol'}  # duplicate-index reindex error
SKIP_PREFIX = ()  # 不跳过 _neut (但在 Lasso 路径中会过滤)

# Lasso 参数
LASSO_CV = 5
LASSO_MAX_ITER = 5000
ICIR_THRESHOLD = 0.10  # regime 内 ICIR 绝对值 >= 0.10 才进 Lasso

# Walk-forward
WF_TRAIN_RATIO = 0.70  # 前 70% 训练, 后 30% OOS

# Top-N 组合回测
TOP_N = 30
QMT_COST_WEEKLY = 0.0072 / 5  # QMT 双边成本 0.72%, 周频 ~0.144%

print("=" * 70)
print("  Regime-aware Lasso + ICIR 因子选择 (确定性方法)")
print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# =====================================================
# Phase 1: 数据加载
# =====================================================
print("\n[Phase 1] 数据加载")
t1 = time.time()
data = load_all_data('2021-01-01', '2026-06-25')
price_data = data['price_data']
financial_data = data['financial_data']
valuation_data = data['valuation_data']
industry_map = data.get('industry_map', {})
close = price_data.get('close')
print(f"  价格矩阵: {close.shape} | 耗时: {time.time()-t1:.0f}s")

# 周频前向收益
close_weekly = close.resample('W').last()
close_weekly.index = pd.to_datetime(close_weekly.index)
forward_returns = close_weekly.shift(-1) / close_weekly - 1
print(f"  周频收益: {forward_returns.shape} ({forward_returns.index[0].strftime('%Y-%m-%d')} ~ {forward_returns.index[-1].strftime('%Y-%m-%d')})")

# =====================================================
# Phase 2: 因子计算
# =====================================================
# 因子缓存 (避免重跑 7 分钟因子计算)
FACTOR_CACHE = SCRIPT_DIR / 'output' / 'factor_cache.pkl'

if FACTOR_CACHE.exists():
    print(f"\n[Phase 2] 因子计算 [CACHE HIT]")
    t2 = time.time()
    with open(FACTOR_CACHE, 'rb') as f:
        factor_dfs = pickle.load(f)
    print(f"  从缓存加载: {len(factor_dfs)} 个因子 | 耗时: {time.time()-t2:.1f}s")
else:
    print("\n[Phase 2] 因子计算")
    t2 = time.time()
    factor_dfs = {}

    for name in FACTOR_DEFS:
        if name not in ALL_FACTORS:
            continue
        if name in SKIP_FACTORS:
            continue
        if name.endswith('_neut'):
            continue

        factor_def = FACTOR_DEFS[name]
        factor_obj = ALL_FACTORS[name]()

        try:
            df = factor_obj.compute(
                price_data=price_data,
                financial_data=financial_data,
                valuation_data=valuation_data,
                industry_map=industry_map,
            )
            if df is not None and not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df_weekly = df.resample('W').last()

                is_financial = df_weekly.shape[0] <= 10 and factor_def['category'] in \
                    ('profitability', 'value', 'growth')
                if is_financial:
                    df_weekly = df_weekly.reindex(close_weekly.index).ffill().bfill()
                else:
                    common_idx = close_weekly.index.intersection(df_weekly.index)
                    if len(common_idx) > 0:
                        df_weekly = df_weekly.reindex(common_idx)
                factor_dfs[name] = df_weekly
        except Exception as e:
            pass

    print(f"  成功: {len(factor_dfs)} 个因子 | 耗时: {time.time()-t2:.0f}s")
    # 缓存
    with open(FACTOR_CACHE, 'wb') as f:
        pickle.dump(factor_dfs, f)
    print(f"  缓存已保存: {FACTOR_CACHE}")

# =====================================================
# Phase 3: 标准化
# =====================================================
print("\n[Phase 3] 标准化 (cross-sectional z-score)")
factor_dfs_std = standardize_factors(factor_dfs)
print(f"  标准化完成: {len(factor_dfs_std)} 个因子")

# =====================================================
# Phase 4: Regime 标签加载 + 对齐
# =====================================================
print("\n[Phase 4] Regime 标签加载")
regime_df = pd.read_csv(REGIME_CSV, parse_dates=['date'])
regime_df = regime_df.sort_values('date').reset_index(drop=True)

# ffill 合并 neutral
regime_df['regime_merged'] = regime_df['regime'].replace('neutral', np.nan).ffill()
regime_df['regime_merged'] = regime_df['regime_merged'].fillna('small_bull')  # 头部兜底

# 对齐到 forward_returns 的周频日期 (W-SUN -> W-FRI, 容差 10 天)
fr_index = pd.DatetimeIndex(forward_returns.index)
regime_series = regime_df.set_index('date')['regime_merged']

# nearest match + ffill (与 run_fa.py Phase 2.6 一致)
_aligned = regime_series.reindex(fr_index, method='nearest', tolerance=pd.Timedelta(days=10))
_aligned = _aligned.ffill().bfill()
regime_map = {d: _aligned.loc[d] for d in fr_index if pd.notna(_aligned.loc[d])}

# 确保所有周在 forward_returns 和所有因子 DataFrame 中都有数据 (避免 KeyError)
_valid_weeks = forward_returns.index
for _fdf in factor_dfs_std.values():
    _valid_weeks = _valid_weeks.intersection(_fdf.index)
_n_before = len(regime_map)
regime_map = {d: r for d, r in regime_map.items() if d in _valid_weeks}
if _n_before != len(regime_map):
    print(f"  [INFO] 过滤到全因子共有周: {_n_before} -> {len(regime_map)}")

regime_set = sorted(set(regime_map.values()))
print(f"  对齐: {len(regime_map)}/{len(fr_index)} 周 (覆盖率 {100*len(regime_map)/len(fr_index):.0f}%)")
for r in regime_set:
    cnt = sum(1 for v in regime_map.values() if v == r)
    print(f"    {r}: {cnt} 周 ({100*cnt/len(regime_map):.0f}%)")

# =====================================================
# Phase 5: Per-regime Lasso + ICIR 因子选择
# =====================================================
print("\n" + "=" * 70)
print("[Phase 5] Per-regime Lasso + ICIR 因子选择")
print("=" * 70)

all_regime_strategies = {}

for regime_name in regime_set:
    print(f"\n--- {regime_name} ---")
    regime_weeks = sorted([d for d, r in regime_map.items() if r == regime_name])
    print(f"  训练周数: {len(regime_weeks)}")

    # --- 5.1 计算 regime 内每因子 ICIR ---
    print(f"  [5.1] 计算因子 ICIR...")
    factor_icir = {}
    factor_ic_series = {}

    for fname, fdf in factor_dfs_std.items():
        # 取 regime 周的数据
        regime_cols = [c for c in fdf.columns if c in forward_returns.columns]
        fdf_regime = fdf.loc[regime_weeks, regime_cols]
        fr_regime = forward_returns.loc[regime_weeks, regime_cols]

        try:
            ic_ir = compute_ic_icir(fdf_regime, fr_regime, min_samples=20)
            if ic_ir and 'icir' in ic_ir and pd.notna(ic_ir['icir']):
                factor_icir[fname] = ic_ir['icir']
                factor_ic_series[fname] = ic_ir.get('ic_series', pd.Series(dtype=float))
        except:
            pass

    print(f"  ICIR 计算: {len(factor_icir)} 因子")

    # ICIR 预筛选
    icir_pass = {k: v for k, v in factor_icir.items() if abs(v) >= ICIR_THRESHOLD}
    print(f"  ICIR >= {ICIR_THRESHOLD}: {len(icir_pass)} 因子")

    if len(icir_pass) < 3:
        print(f"  [WARN] ICIR 通过因子 < 3, 跳过此 regime")
        all_regime_strategies[regime_name] = {'error': 'insufficient_factors', 'n_pass': len(icir_pass)}
        continue

    # --- 5.2 构建回归矩阵 ---
    print(f"  [5.2] 构建回归矩阵 (Lasso 输入)...")
    selected_factors = sorted(icir_pass.keys())

    # 拉平: (N_stocks * N_weeks, N_factors) -> y = forward_return
    X_parts = []
    y_parts = []
    for week in regime_weeks:
        if week not in forward_returns.index:
            continue
        fr_week = forward_returns.loc[week]
        row_data = {}
        for fname in selected_factors:
            fdf = factor_dfs_std[fname]
            if week in fdf.index:
                row_data[fname] = fdf.loc[week]

        if len(row_data) < len(selected_factors) * 0.5:
            continue  # 缺太多因子

        # 构建 DataFrame
        week_df = pd.DataFrame(row_data)
        week_y = fr_week.reindex(week_df.index)

        # 丢弃 NaN
        mask = week_df.notna().all(axis=1) & week_y.notna()
        X_parts.append(week_df[mask])
        y_parts.append(week_y[mask])

    if not X_parts:
        print(f"  [WARN] 无法构建回归矩阵, 跳过此 regime")
        all_regime_strategies[regime_name] = {'error': 'no_valid_data'}
        continue

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    print(f"  回归矩阵: X={X.shape}, y={y.shape} | 因子: {selected_factors}")

    # --- 5.3 LassoCV 特征选择 ---
    print(f"  [5.3] LassoCV {LASSO_CV}-fold 特征选择...")
    t_lasso = time.time()

    lasso = LassoCV(
        cv=LASSO_CV,
        max_iter=LASSO_MAX_ITER,
        n_jobs=-1,
        random_state=42,
    )
    lasso.fit(X, y)

    print(f"  Lasso 完成: alpha={lasso.alpha_:.6f} | 耗时: {time.time()-t_lasso:.1f}s")

    # 非0系数因子
    coefs = pd.Series(lasso.coef_, index=selected_factors)
    nonzero = coefs[coefs != 0]
    print(f"  Lasso 选中: {len(nonzero)}/{len(selected_factors)} 因子")
    for fname, coef in nonzero.sort_values(key=abs, ascending=False).items():
        icir_val = factor_icir.get(fname, 0)
        direction = "+" if coef > 0 else "-"
        print(f"    {fname:35s} coef={coef:+.4f} ICIR={icir_val:+.3f} dir={direction}")

    if len(nonzero) == 0:
        print(f"  [WARN] Lasso 未选中任何因子, 回退到 ICIR top-4")
        nonzero = pd.Series(
            {k: v for k, v in sorted(icir_pass.items(), key=lambda x: abs(x[1]), reverse=True)[:4]}
        )

    # --- 5.4 ICIR 加权 ---
    print(f"  [5.4] ICIR 加权...")
    weights = {}
    for fname, coef in nonzero.items():
        icir_val = factor_icir.get(fname, 0)
        coef_sign = 1 if coef > 0 else -1

        # 方向一致性检查
        if coef_sign * icir_val > 0:
            # 方向一致: ICIR 正 + Lasso 正 -> 正向因子
            weights[fname] = abs(icir_val)
        elif abs(icir_val) > 0.3:
            # 方向矛盾但 ICIR 强: 信任 ICIR 方向
            weights[fname] = abs(icir_val)
            print(f"    [WARN] {fname}: Lasso={coef:+.4f} ICIR={icir_val:+.3f} 方向矛盾, 信任 ICIR")
        else:
            # 方向矛盾且 ICIR 弱: 丢弃
            print(f"    [SKIP] {fname}: Lasso={coef:+.4f} ICIR={icir_val:+.3f} 方向矛盾且弱, 丢弃")

    if len(weights) == 0:
        print(f"  [WARN] 所有因子方向矛盾, 回退到 ICIR top-4")
        for fname, icir_val in sorted(icir_pass.items(), key=lambda x: abs(x[1]), reverse=True)[:4]:
            weights[fname] = abs(icir_val)

    # --- Top-K 截断 (优先有 JQ 映射的因子) ---
    MAX_FACTORS = 5
    if len(weights) > MAX_FACTORS:
        jq_avail = {k: v for k, v in weights.items() if k in JQ_FACTOR_REGISTRY} if JQ_AVAILABLE else {}
        jq_missing = {k: v for k, v in weights.items() if k not in jq_avail}

        sorted_jq = sorted(jq_avail.items(), key=lambda x: -x[1])
        sorted_no = sorted(jq_missing.items(), key=lambda x: -x[1])

        selected = dict(sorted_jq[:MAX_FACTORS])
        for k, v in sorted_no:
            if len(selected) >= MAX_FACTORS:
                break
            selected[k] = v

        dropped = set(weights.keys()) - set(selected.keys())
        print(f"  [Top-{MAX_FACTORS}] 截断: {len(weights)} -> {len(selected)} 因子 (JQ映射: {len(jq_avail)}有/{len(jq_missing)}缺)")
        if dropped:
            print(f"           丢弃: {dropped}")
        weights = selected

    # 归一化
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    print(f"\n  ★ {regime_name} 最终策略:")
    for fname, w in sorted(weights.items(), key=lambda x: -x[1]):
        icir_val = factor_icir.get(fname, 0)
        print(f"    {fname:35s} weight={w:.3f} ICIR={icir_val:+.3f}")

    all_regime_strategies[regime_name] = {
        'weights': weights,
        'factor_icir': {k: v for k, v in factor_icir.items() if k in weights},
        'lasso_alpha': lasso.alpha_,
        'lasso_nonzero': len(nonzero),
        'n_train_weeks': len(regime_weeks),
        'n_factors_input': len(selected_factors),
    }

# =====================================================
# Phase 6: Walk-forward OOS 验证
# =====================================================
print("\n" + "=" * 70)
print("[Phase 6] Walk-forward OOS 验证")
print("=" * 70)

wf_results = {}

for regime_name, strat in all_regime_strategies.items():
    if 'error' in strat:
        wf_results[regime_name] = {'error': strat['error']}
        continue

    weights = strat['weights']
    regime_weeks = sorted([d for d, r in regime_map.items() if r == regime_name])

    n_train = int(len(regime_weeks) * WF_TRAIN_RATIO)
    train_weeks = regime_weeks[:n_train]
    oos_weeks = regime_weeks[n_train:]

    print(f"\n--- {regime_name} WF ---")
    print(f"  训练: {train_weeks[0].strftime('%Y-%m-%d')} ~ {train_weeks[-1].strftime('%Y-%m-%d')} ({len(train_weeks)}周)")
    print(f"  OOS:  {oos_weeks[0].strftime('%Y-%m-%d')} ~ {oos_weeks[-1].strftime('%Y-%m-%d')} ({len(oos_weeks)}周)")

    if len(oos_weeks) < 5:
        print(f"  [WARN] OOS 周 < 5, 跳过")
        wf_results[regime_name] = {'error': 'insufficient_oos'}
        continue

    # --- 6.1 训练集: 重做 Lasso + ICIR ---
    train_icir = {}
    for fname in weights:
        fdf = factor_dfs_std[fname]
        cols = [c for c in fdf.columns if c in forward_returns.columns]
        fdf_t = fdf.loc[train_weeks, cols]
        fr_t = forward_returns.loc[train_weeks, cols]
        try:
            ic_ir = compute_ic_icir(fdf_t, fr_t, min_samples=10)
            if ic_ir and 'icir' in ic_ir and pd.notna(ic_ir['icir']):
                train_icir[fname] = ic_ir['icir']
        except:
            pass

    # 训练集权重
    train_weights = {}
    for fname, w in weights.items():
        icir_val = train_icir.get(fname, 0)
        if abs(icir_val) > 0:
            train_weights[fname] = abs(icir_val)
        else:
            train_weights[fname] = w  # 回退到全样本权重

    total_tw = sum(train_weights.values())
    if total_tw > 0:
        train_weights = {k: v / total_tw for k, v in train_weights.items()}

    # --- 6.2 OOS 组合回测 ---
    # 合成因子 = sum(weight_i * z_score_i)
    portfolio_returns = []

    for week in oos_weeks:
        if week not in forward_returns.index:
            continue

        # 合成因子值
        composite = pd.Series(0.0, index=forward_returns.columns)
        for fname, w in train_weights.items():
            fdf = factor_dfs_std.get(fname)
            if fdf is not None and week in fdf.index:
                composite += w * fdf.loc[week].reindex(composite.index).fillna(0)

        # 排序选 Top-30
        composite = composite.dropna()
        if len(composite) < TOP_N:
            continue

        top_stocks = composite.nlargest(TOP_N).index
        fr_week = forward_returns.loc[week]

        # 等权组合收益
        valid = [s for s in top_stocks if s in fr_week.index and pd.notna(fr_week[s])]
        if len(valid) < TOP_N // 2:
            continue

        port_ret = fr_week[valid].mean()
        portfolio_returns.append({
            'week': week,
            'return': port_ret,
            'n_stocks': len(valid),
        })

    if not portfolio_returns:
        wf_results[regime_name] = {'error': 'no_oos_returns'}
        continue

    wf_df = pd.DataFrame(portfolio_returns).set_index('week')
    wf_df['cum_ret'] = (1 + wf_df['return']).cumprod() - 1

    # 指标
    weekly_mean = wf_df['return'].mean()
    weekly_std = wf_df['return'].std()
    annual_ret = weekly_mean * 52
    annual_std = weekly_std * np.sqrt(52)
    sharpe = annual_ret / annual_std if annual_std > 0 else 0
    max_dd = (wf_df['cum_ret'] - wf_df['cum_ret'].cummax()).min()

    # 换手率 (近似)
    stock_sets = [set(r['n_stocks'] for _ in [r]) for _, r in wf_df.iterrows()]

    # OOS ICIR
    oos_icir_vals = []
    for fname in train_weights:
        fdf = factor_dfs_std.get(fname)
        if fdf is None:
            continue
        cols = [c for c in fdf.columns if c in forward_returns.columns]
        fdf_oos = fdf.loc[oos_weeks, cols]
        fr_oos = forward_returns.loc[oos_weeks, cols]
        try:
            ic_ir = compute_ic_icir(fdf_oos, fr_oos, min_samples=5)
            if ic_ir and 'icir' in ic_ir and pd.notna(ic_ir['icir']):
                oos_icir_vals.append(ic_ir['icir'])
        except:
            pass

    oos_icir_mean = np.mean(oos_icir_vals) if oos_icir_vals else 0

    print(f"  OOS 年化收益: {annual_ret:+.1%}")
    print(f"  OOS Sharpe:   {sharpe:.2f}")
    print(f"  OOS MaxDD:    {max_dd:.1%}")
    print(f"  OOS ICIR:     {oos_icir_mean:+.3f} (mean of {len(oos_icir_vals)} factors)")

    wf_results[regime_name] = {
        'annual_return': float(annual_ret),
        'sharpe': float(sharpe),
        'max_dd': float(max_dd),
        'oos_icir': float(oos_icir_mean),
        'n_oos_weeks': len(oos_weeks),
        'train_weights': train_weights,
    }

# =====================================================
# Phase 7: 全样本回测 (所有 regime 合并)
# =====================================================
print("\n" + "=" * 70)
print("[Phase 7] 全样本回测 (regime-aware 合成)")
print("=" * 70)

# 合成: 每周根据 regime 标签选择对应的权重
all_weeks = sorted(regime_map.keys())
portfolio_returns_all = []

for week in all_weeks:
    if week not in forward_returns.index:
        continue

    regime = regime_map[week]
    strat = all_regime_strategies.get(regime, {})
    if 'error' in strat or 'weights' not in strat:
        continue

    weights = strat['weights']
    composite = pd.Series(0.0, index=forward_returns.columns)
    for fname, w in weights.items():
        fdf = factor_dfs_std.get(fname)
        if fdf is not None and week in fdf.index:
            composite += w * fdf.loc[week].reindex(composite.index).fillna(0)

    composite = composite.dropna()
    if len(composite) < TOP_N:
        continue

    top_stocks = composite.nlargest(TOP_N).index
    fr_week = forward_returns.loc[week]
    valid = [s for s in top_stocks if s in fr_week.index and pd.notna(fr_week[s])]
    if len(valid) < TOP_N // 2:
        continue

    port_ret = fr_week[valid].mean()
    # 扣除 QMT 成本
    port_ret_net = port_ret - QMT_COST_WEEKLY

    portfolio_returns_all.append({
        'week': week,
        'regime': regime,
        'return_gross': port_ret,
        'return_net': port_ret_net,
        'n_stocks': len(valid),
    })

if portfolio_returns_all:
    bt_df = pd.DataFrame(portfolio_returns_all).set_index('week')
    bt_df['cum_gross'] = (1 + bt_df['return_gross']).cumprod() - 1
    bt_df['cum_net'] = (1 + bt_df['return_net']).cumprod() - 1

    weekly_mean_net = bt_df['return_net'].mean()
    weekly_std_net = bt_df['return_net'].std()
    annual_ret_net = weekly_mean_net * 52
    annual_std_net = weekly_std_net * np.sqrt(52)
    sharpe_net = annual_ret_net / annual_std_net if annual_std_net > 0 else 0
    max_dd_net = (bt_df['cum_net'] - bt_df['cum_net'].cummax()).min()

    # 分 regime 统计
    regime_stats = bt_df.groupby('regime')['return_net'].agg(['mean', 'std', 'count'])
    regime_stats['annual'] = regime_stats['mean'] * 52
    regime_stats['sharpe'] = regime_stats['annual'] / (regime_stats['std'] * np.sqrt(52))

    print(f"\n  全样本 (regime-aware 合成, 扣 QMT 成本):")
    print(f"  周数: {len(bt_df)} | 年化收益: {annual_ret_net:+.1%} | Sharpe: {sharpe_net:.2f} | MaxDD: {max_dd_net:.1%}")
    print(f"\n  分 regime:")
    for r, row in regime_stats.iterrows():
        print(f"    {r:20s} {row['count']:>3.0f}周 | 年化: {row['annual']:+.1%} | Sharpe: {row['sharpe']:.2f}")

    # 2026 时效验证
    bt_2026 = bt_df[bt_df.index >= '2026-01-01']
    if len(bt_2026) > 0:
        w26 = bt_2026['return_net'].mean()
        s26 = bt_2026['return_net'].std()
        a26 = w26 * 52
        sh26 = a26 / (s26 * np.sqrt(52)) if s26 > 0 else 0
        print(f"\n  2026 时效验证 ({len(bt_2026)}周): 年化 {a26:+.1%} | Sharpe {sh26:.2f}")

    # 保存回测结果
    bt_result = {
        'annual_return_net': float(annual_ret_net),
        'sharpe_net': float(sharpe_net),
        'max_dd_net': float(max_dd_net),
        'n_weeks': len(bt_df),
    }
else:
    bt_result = {'error': 'no_portfolio_returns'}
    print("  [WARN] 无组合收益")

# =====================================================
# Phase 8: 保存结果 + JQ 策略生成
# =====================================================
print("\n" + "=" * 70)
print("[Phase 8] 保存结果 + JQ 策略生成")
print("=" * 70)

# 保存 JSON
result = {
    'timestamp': datetime.now().isoformat(),
    'method': 'Lasso_CV_ICIR_weighted',
    'regimes': {},
    'walk_forward': {},
    'backtest': bt_result,
}

for rname, strat in all_regime_strategies.items():
    if 'error' in strat:
        result['regimes'][rname] = {'error': strat['error']}
    else:
        result['regimes'][rname] = {
            'weights': strat['weights'],
            'factor_icir': strat['factor_icir'],
            'lasso_alpha': strat['lasso_alpha'],
            'n_train_weeks': strat['n_train_weeks'],
        }

for rname, wf in wf_results.items():
    result['walk_forward'][rname] = wf

result_path = OUTPUT_DIR / 'regime_lasso_result.json'
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
print(f"  结果保存: {result_path}")

# JQ 策略生成
if JQ_AVAILABLE:
    print(f"\n  JQ 策略生成:")
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

    for rname, strat in all_regime_strategies.items():
        if 'error' in strat:
            print(f"    {rname}: SKIP (error)")
            continue

        weights = strat['weights']
        # 只保留 JQ_FACTOR_REGISTRY 中有映射的因子
        jq_weights = {k: v for k, v in weights.items() if k in JQ_FACTOR_REGISTRY}
        missing = set(weights.keys()) - set(JQ_FACTOR_REGISTRY.keys())

        if len(jq_weights) < 2:
            print(f"    {rname}: SKIP (仅 {len(jq_weights)} 因子有 JQ 映射, 需 >= 2)")
            if missing:
                print(f"           缺映射: {missing}")
            continue

        # 重新归一化
        total = sum(jq_weights.values())
        jq_weights = {k: v / total for k, v in jq_weights.items()}

        # 生成
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        fname = f'fa_v79_lasso_{rname}_{ts}.py'
        fpath = STRATEGIES_DIR / fname

        try:
            gen = JQGenerator(
                weights=jq_weights,
                name=f'lasso_{rname}',
                mcap_min=5,
            )
            gen.generate(str(fpath))
            print(f"    {rname}: ✅ {fname}")
            if missing:
                print(f"           (缺 JQ 映射: {missing})")
        except Exception as e:
            print(f"    {rname}: ❌ {e}")

# 汇总
print("\n" + "=" * 70)
print("  完成!")
print("=" * 70)
print(f"  方法: Lasso CV={LASSO_CV} + ICIR 加权")
print(f"  Regimes: {list(all_regime_strategies.keys())}")
for rname, strat in all_regime_strategies.items():
    if 'weights' in strat:
        print(f"  {rname}: {len(strat['weights'])} 因子")
print(f"  结果文件: {result_path}")
