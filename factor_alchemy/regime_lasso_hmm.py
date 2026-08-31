#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HMM Regime-aware ElasticNetCV 因子选择
=========================================
基于 HMM 隐含市场状态 (4 states) 的 per-regime ElasticNetCV + rank-based 标准化。

与 regime_lasso.py 的区别:
  - 用 HMM 自动发现的 4 个状态替代规则型 (small_bull/large_dominant)
  - 固定使用 ElasticNetCV (l1_ratio 自动搜索)
  - 因子标准化: rank-based percentile→N(0,1) (v6, 与 composite.py 一致)

用法:
  cd <项目根目录> && python factor_alchemy/regime_lasso_hmm.py

输出:
  - output/regime_lasso_result_hmm.json
  - strategies/drafts/fa_hmm_elasticnet_{state}.py
"""
import sys, os, time, json, pickle, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ── 路径 ──
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from data.loader import load_all_data
from factors import ALL_FACTORS, get_factor_instance
from config import FACTOR_DEFS
from factors.composite import standardize_factors
from factors.preprocess import winsorize_price_data, winsorize_financial_data, winsorize_valuation_data
from evaluation.ic_analysis import compute_ic_icir
from sklearn.linear_model import ElasticNetCV

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
FACTOR_CACHE = OUTPUT_DIR / 'factor_cache.pkl'
HMM_LABELS_CSV = Path(os.environ.get("FUXI_DATA_DIR") or str(Path(__file__).resolve().parent.parent.parent / "data")) / 'regime_labels_hmm.csv'
RESULT_JSON = OUTPUT_DIR / 'regime_lasso_result_hmm.json'

ICIR_THRESHOLD = 0.10
MAX_FACTORS = 5
TOP_N = 30
QMT_COST_WEEKLY = 0.0072 / 52  # QMT 双边 0.72% 年化 → 周频

# ElasticNetCV 参数
EN_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
EN_CV = 5
EN_MAX_ITER = 5000

# Walk-forward
WF_TRAIN_RATIO = 0.70

print("=" * 70)
print("  HMM Regime-aware ElasticNetCV 因子选择")
print(f"  HMM标签: {HMM_LABELS_CSV}")
print(f"  标准化: rank-based percentile→N(0,1)")
print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# =====================================================
# Phase 0: HMM 状态命名 (从价格数据计算特征)
# =====================================================
print("\n[Phase 0] HMM 状态命名")

hmm_df = pd.read_csv(HMM_LABELS_CSV)
hmm_df['dt'] = pd.to_datetime(hmm_df['dt'])
print(f"  加载 HMM 标签: {len(hmm_df)} 周, states={sorted(hmm_df['hmm_state'].unique())}")

# 用指数数据计算各状态特征以命名
try:
    idx_dir = Path(os.environ.get("FUXI_DATA_DIR") or str(Path(__file__).resolve().parent.parent.parent / "data")) / 'jqdata'
    csi300 = pd.read_csv(idx_dir / 'index_000300_daily.csv')
    csi500 = pd.read_csv(idx_dir / 'index_000905_daily.csv')
    
    for df_i in [csi300, csi500]:
        df_i['dt'] = pd.to_datetime(df_i['trade_date'].astype(str), format='%Y%m%d')
        df_i.set_index('dt', inplace=True)
        df_i.sort_index(inplace=True)
    
    w300 = csi300['close'].resample('W').last().pct_change().dropna()
    w500 = csi500['close'].resample('W').last().pct_change().dropna()
    
    # 命名各状态
    state_names = {}
    for s in sorted(hmm_df['hmm_state'].unique()):
        weeks_s = hmm_df[hmm_df['hmm_state'] == s]['dt'].tolist()
        valid_ws = [w for w in weeks_s if w in w300.index and w in w500.index]
        if valid_ws:
            r300 = w300[valid_ws].mean()
            r500 = w500[valid_ws].mean()
            spread = r500 - r300
            
            if r300 > 0 and r500 > 0:
                name = "small_bull" if spread > 0.003 else "broad_bull"
            elif r300 > 0 and r500 <= 0:
                name = "large_dominant"
            elif r300 <= 0 and r500 <= 0:
                name = "small_resilient" if spread > 0 else "broad_bear"
            else:
                name = "small_bear"
        else:
            name = f"state_{s}"
        
        state_names[s] = name
        n = len(weeks_s)
        print(f"  State {s} → {name}: {n}周, CSI300={r300:+.4f}, CSI500={r500:+.4f}, spread={spread:+.4f}")
    
except Exception as e:
    print(f"  [WARN] 无法计算状态特征: {e}, 使用数字标签")
    state_names = {int(s): f"state_{int(s)}" for s in sorted(hmm_df['hmm_state'].unique())}

# 构建 regime_map: {date_str → state_name}
hmm_df['state_name'] = hmm_df['hmm_state'].map(state_names)
state_counts = hmm_df['state_name'].value_counts()
print(f"\n  状态分布:")
for name, cnt in state_counts.items():
    print(f"    {name}: {cnt} 周 ({100*cnt/len(hmm_df):.0f}%)")

# =====================================================
# Phase 1: 数据加载
# =====================================================
print("\n[Phase 1] 数据加载")
t1 = time.time()
data = load_all_data('2021-01-01', '2026-06-25')
price_data = data['price_data']
close = price_data.get('close')
print(f"  价格矩阵: {close.shape} | 耗时: {time.time()-t1:.0f}s")

# 前向收益用原始价格 (market truth), 不受 winsorize 影响
close_weekly = close.resample('W').last()
close_weekly.index = pd.to_datetime(close_weekly.index)
forward_returns = close_weekly.shift(-1) / close_weekly - 1
print(f"  周频收益: {forward_returns.shape}")

# [winsorize] 原始数据层截尾 1%/99%, 防止极端值污染滚动统计
# 只对因子 INPUT 做, 不对 target (forward_returns) 做
price_data = winsorize_price_data(price_data)
print(f"  winsorize 完成 (价格数据)")

# =====================================================
# Phase 2: 因子加载 (缓存优先)
# =====================================================
if FACTOR_CACHE.exists():
    print(f"\n[Phase 2] 因子加载 [CACHE HIT]")
    with open(FACTOR_CACHE, 'rb') as f:
        factor_dfs = pickle.load(f)
    print(f"  {len(factor_dfs)} 个因子")
else:
    print("\n[Phase 2] 因子计算 (无缓存, 需要 7min)...")
    # fallback: 重新计算
    financial_data = winsorize_financial_data(data.get('financial_data'))
    valuation_data = winsorize_valuation_data(data.get('valuation_data'))
    industry_map = data.get('industry_map', {})
    
    factor_dfs = {}
    for name in FACTOR_DEFS:
        if name not in ALL_FACTORS:
            continue
        if name.endswith('_neut'):
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
                is_financial = df_weekly.shape[0] <= 10
                if is_financial:
                    df_weekly = df_weekly.reindex(close_weekly.index).ffill().bfill()
                else:
                    common_idx = close_weekly.index.intersection(df_weekly.index)
                    if len(common_idx) > 0:
                        df_weekly = df_weekly.reindex(common_idx)
                factor_dfs[name] = df_weekly
        except:
            pass
    print(f"  {len(factor_dfs)} 个因子")

# =====================================================
# Phase 3: Rank-based 标准化
# =====================================================
print("\n[Phase 3] Rank-based 标准化 (percentile→N(0,1))")
factor_dfs_std = standardize_factors(factor_dfs)
print(f"  完成: {len(factor_dfs_std)} 个因子")

# =====================================================
# Phase 4: HMM 标签对齐
# =====================================================
print("\n[Phase 4] HMM 标签对齐到因子周频")

# 对齐: HMM dt → 最近的 forward_returns 周
fr_dates = pd.DatetimeIndex(forward_returns.index)
hmm_dates = pd.DatetimeIndex(hmm_df['dt'])
regime_map = {}

for fr_d in fr_dates:
    # 找 HMM 中最近的标签 (容差 7 天)
    diffs = abs((hmm_dates - fr_d).days)
    if diffs.min() <= 7:
        idx = diffs.argmin()
        regime_map[fr_d] = hmm_df['state_name'].iloc[idx]

# 过滤到有效周
valid_weeks = forward_returns.index
for fdf in factor_dfs_std.values():
    valid_weeks = valid_weeks.intersection(fdf.index)
regime_map = {d: r for d, r in regime_map.items() if d in valid_weeks}

regime_set = sorted(set(regime_map.values()))
print(f"  对齐: {len(regime_map)} 周")
for r in regime_set:
    cnt = sum(1 for v in regime_map.values() if v == r)
    print(f"    {r}: {cnt} 周")

# =====================================================
# Phase 5: Per-state ElasticNetCV
# =====================================================
print("\n" + "=" * 70)
print("[Phase 5] Per-state ElasticNetCV 因子选择")
print("=" * 70)

all_strategies = {}

for state_name in regime_set:
    state_weeks = sorted([d for d, r in regime_map.items() if r == state_name])
    print(f"\n--- {state_name} ({len(state_weeks)} 周) ---")
    
    if len(state_weeks) < 20:
        print(f"  [SKIP] 样本 < 20 周")
        all_strategies[state_name] = {'error': 'too_few_samples', 'n_weeks': len(state_weeks)}
        continue
    
    # 5.1 计算 state 内每因子 ICIR
    factor_icir = {}
    for fname, fdf in factor_dfs_std.items():
        regime_cols = [c for c in fdf.columns if c in forward_returns.columns]
        try:
            fdf_r = fdf.loc[state_weeks, regime_cols]
            fr_r = forward_returns.loc[state_weeks, regime_cols]
            ic_ir = compute_ic_icir(fdf_r, fr_r, min_samples=10)
            if ic_ir and 'icir' in ic_ir and pd.notna(ic_ir['icir']):
                factor_icir[fname] = ic_ir['icir']
        except:
            pass
    
    print(f"  ICIR 计算: {len(factor_icir)} 因子")
    icir_pass = {k: v for k, v in factor_icir.items() if abs(v) >= ICIR_THRESHOLD}
    print(f"  ICIR >= {ICIR_THRESHOLD}: {len(icir_pass)} 因子")
    
    if len(icir_pass) < 3:
        print(f"  [SKIP] ICIR 通过 < 3")
        all_strategies[state_name] = {'error': 'insufficient_factors', 'n_pass': len(icir_pass)}
        continue
    
    # 5.2 构建回归矩阵
    selected = sorted(icir_pass.keys())
    X_parts, y_parts = [], []
    for week in state_weeks:
        if week not in forward_returns.index:
            continue
        fr_w = forward_returns.loc[week]
        row_data = {}
        for fn in selected:
            fdf = factor_dfs_std[fn]
            if week in fdf.index:
                row_data[fn] = fdf.loc[week]
        if len(row_data) < len(selected) * 0.5:
            continue
        week_df = pd.DataFrame(row_data)
        week_y = fr_w.reindex(week_df.index)
        mask = week_df.notna().all(axis=1) & week_y.notna()
        if mask.sum() < 10:
            continue
        X_parts.append(week_df[mask])
        y_parts.append(week_y[mask])
    
    if not X_parts:
        all_strategies[state_name] = {'error': 'no_valid_data'}
        continue
    
    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    print(f"  回归矩阵: X={X.shape}, y={y.shape}")
    
    # 5.3 ElasticNetCV
    print(f"  ElasticNetCV (l1_ratio={EN_L1_RATIOS}, cv={EN_CV})...")
    t_en = time.time()
    en = ElasticNetCV(
        l1_ratio=EN_L1_RATIOS, cv=EN_CV, max_iter=EN_MAX_ITER,
        n_jobs=-1, random_state=42
    )
    en.fit(X, y)
    coefs = pd.Series(en.coef_, index=selected)
    nonzero = coefs[coefs != 0]
    print(f"  ElasticNet: alpha={en.alpha_:.6f}, l1_ratio={en.l1_ratio_:.3f}, "
          f"nonzero={len(nonzero)}/{len(selected)} | {time.time()-t_en:.1f}s")
    
    if len(nonzero) == 0:
        nonzero = pd.Series({k: v for k, v in
            sorted(icir_pass.items(), key=lambda x: abs(x[1]), reverse=True)[:4]})
    
    # 5.4 ICIR 方向加权
    weights = {}
    for fname, coef in nonzero.items():
        icir_val = factor_icir.get(fname, 0)
        coef_sign = 1 if coef > 0 else -1
        if coef_sign * icir_val > 0:
            weights[fname] = abs(icir_val)
        elif abs(icir_val) > 0.3:
            weights[fname] = abs(icir_val)
            print(f"    [WARN] {fname}: EN={coef:+.4f} ICIR={icir_val:+.3f} 方向矛盾, 信任ICIR")
        else:
            print(f"    [SKIP] {fname}: EN={coef:+.4f} ICIR={icir_val:+.3f} 方向矛盾, 丢弃")
    
    if len(weights) == 0:
        for fname, icir_val in sorted(icir_pass.items(), key=lambda x: abs(x[1]), reverse=True)[:4]:
            weights[fname] = abs(icir_val)
    
    # Top-K 截断
    if len(weights) > MAX_FACTORS:
        sorted_w = sorted(weights.items(), key=lambda x: -x[1])
        weights = dict(sorted_w[:MAX_FACTORS])
        print(f"  Top-{MAX_FACTORS}: {len(sorted_w)} → {MAX_FACTORS}")
    
    # 归一化
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    print(f"\n  ★ {state_name} 策略 ({len(weights)} 因子):")
    for fn, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {fn:35s} w={w:.3f} ICIR={factor_icir.get(fn,0):+.3f}")
    
    all_strategies[state_name] = {
        'weights': weights,
        'factor_icir': {k: factor_icir.get(k, 0) for k in weights},
        'elasticnet_alpha': float(en.alpha_),
        'elasticnet_l1_ratio': float(en.l1_ratio_),
        'n_nonzero_en': len(nonzero),
        'n_train_weeks': len(state_weeks),
        'n_input_factors': len(selected),
    }

# =====================================================
# Phase 6: Walk-forward OOS 验证
# =====================================================
print("\n" + "=" * 70)
print("[Phase 6] Walk-forward OOS 验证")
print("=" * 70)

wf_results = {}

for state_name, strat in all_strategies.items():
    if 'error' in strat:
        wf_results[state_name] = {'error': strat['error']}
        continue
    
    weights = strat['weights']
    state_weeks = sorted([d for d, r in regime_map.items() if r == state_name])
    
    n_train = int(len(state_weeks) * WF_TRAIN_RATIO)
    train_weeks = state_weeks[:n_train]
    oos_weeks = state_weeks[n_train:]
    
    print(f"\n--- {state_name} WF: train={len(train_weeks)}w, oos={len(oos_weeks)}w ---")
    
    if len(oos_weeks) < 5:
        wf_results[state_name] = {'error': 'insufficient_oos'}
        continue
    
    # 训练集 ICIR → WF 权重
    train_icir = {}
    for fname in weights:
        fdf = factor_dfs_std[fname]
        cols = [c for c in fdf.columns if c in forward_returns.columns]
        try:
            ic_ir = compute_ic_icir(fdf.loc[train_weeks, cols], 
                                     forward_returns.loc[train_weeks, cols], min_samples=10)
            if ic_ir and 'icir' in ic_ir and pd.notna(ic_ir['icir']):
                train_icir[fname] = ic_ir['icir']
        except:
            pass
    
    train_weights = {}
    for fname, w in weights.items():
        tv = train_icir.get(fname, 0)
        train_weights[fname] = abs(tv) if abs(tv) > 0 else w
    total_tw = sum(train_weights.values())
    if total_tw > 0:
        train_weights = {k: v / total_tw for k, v in train_weights.items()}
    
    # OOS 组合回测
    port_rets = []
    for week in oos_weeks:
        if week not in forward_returns.index:
            continue
        composite = pd.Series(0.0, index=forward_returns.columns)
        for fname, w in train_weights.items():
            fdf = factor_dfs_std.get(fname)
            if fdf is not None and week in fdf.index:
                composite += w * fdf.loc[week].reindex(composite.index).fillna(0)
        composite = composite.dropna()
        if len(composite) < TOP_N:
            continue
        top = composite.nlargest(TOP_N).index
        fr = forward_returns.loc[week]
        valid = [s for s in top if s in fr.index and pd.notna(fr[s])]
        if len(valid) < TOP_N // 2:
            continue
        port_rets.append({'week': week, 'return': fr[valid].mean(), 'n': len(valid)})
    
    if not port_rets:
        wf_results[state_name] = {'error': 'no_oos_returns'}
        continue
    
    wf_df = pd.DataFrame(port_rets).set_index('week')
    wf_df['cum'] = (1 + wf_df['return']).cumprod() - 1
    wm, ws = wf_df['return'].mean(), wf_df['return'].std()
    ann_ret = wm * 52
    sharpe = ann_ret / (ws * np.sqrt(52)) if ws > 0 else 0
    mdd = (wf_df['cum'] - wf_df['cum'].cummax()).min()
    
    # OOS ICIR
    oos_icirs = []
    for fname in train_weights:
        fdf = factor_dfs_std.get(fname)
        if fdf is None:
            continue
        cols = [c for c in fdf.columns if c in forward_returns.columns]
        try:
            ic_ir = compute_ic_icir(fdf.loc[oos_weeks, cols],
                                     forward_returns.loc[oos_weeks, cols], min_samples=5)
            if ic_ir and 'icir' in ic_ir and pd.notna(ic_ir['icir']):
                oos_icirs.append(ic_ir['icir'])
        except:
            pass
    
    oos_icir_mean = np.mean(oos_icirs) if oos_icirs else 0
    
    print(f"  OOS: 年化={ann_ret:+.1%} Sharpe={sharpe:.2f} MaxDD={mdd:.1%} ICIR={oos_icir_mean:+.3f}")
    
    wf_results[state_name] = {
        'annual_return': float(ann_ret), 'sharpe': float(sharpe),
        'max_dd': float(mdd), 'oos_icir': float(oos_icir_mean),
        'n_oos_weeks': len(oos_weeks), 'train_weights': train_weights,
    }

# =====================================================
# Phase 7: 全样本回测 (state-aware)
# =====================================================
print("\n" + "=" * 70)
print("[Phase 7] 全样本回测 (state-aware 合成)")
print("=" * 70)

all_weeks = sorted(regime_map.keys())
port_rets_all = []

for week in all_weeks:
    if week not in forward_returns.index:
        continue
    state = regime_map[week]
    strat = all_strategies.get(state, {})
    if 'error' in strat or 'weights' not in strat:
        continue
    
    weights = strat['weights']
    composite = pd.Series(0.0, index=forward_returns.columns)
    for fn, w in weights.items():
        fdf = factor_dfs_std.get(fn)
        if fdf is not None and week in fdf.index:
            composite += w * fdf.loc[week].reindex(composite.index).fillna(0)
    
    composite = composite.dropna()
    if len(composite) < TOP_N:
        continue
    top = composite.nlargest(TOP_N).index
    fr_w = forward_returns.loc[week]
    valid = [s for s in top if s in fr_w.index and pd.notna(fr_w[s])]
    if len(valid) < TOP_N // 2:
        continue
    
    port_ret = fr_w[valid].mean()
    port_rets_all.append({
        'week': week, 'state': state,
        'return_gross': port_ret, 'return_net': port_ret - QMT_COST_WEEKLY,
        'n_stocks': len(valid),
    })

if port_rets_all:
    bt = pd.DataFrame(port_rets_all).set_index('week')
    bt['cum_net'] = (1 + bt['return_net']).cumprod() - 1
    bt['cum_gross'] = (1 + bt['return_gross']).cumprod() - 1
    
    wm_n, ws_n = bt['return_net'].mean(), bt['return_net'].std()
    ann_net = wm_n * 52
    sh_net = ann_net / (ws_n * np.sqrt(52)) if ws_n > 0 else 0
    mdd_net = (bt['cum_net'] - bt['cum_net'].cummax()).min()
    
    print(f"\n  全样本 (扣 QMT 成本):")
    print(f"  周数: {len(bt)} | 年化: {ann_net:+.1%} | Sharpe: {sh_net:.2f} | MaxDD: {mdd_net:.1%}")
    
    # 分 state
    for st in regime_set:
        sub = bt[bt['state'] == st]
        if len(sub) > 0:
            sm, ss2 = sub['return_net'].mean(), sub['return_net'].std()
            sa = sm * 52
            sh = sa / (ss2 * np.sqrt(52)) if ss2 > 0 else 0
            print(f"    {st:20s} {len(sub):>3d}w | 年化={sa:+.1%} Sharpe={sh:.2f}")
    
    bt_result = {'annual_return_net': float(ann_net), 'sharpe_net': float(sh_net),
                 'max_dd_net': float(mdd_net), 'n_weeks': len(bt)}
else:
    bt_result = {'error': 'no_portfolio_returns'}

# =====================================================
# Phase 8: 保存 + JQ 生成
# =====================================================
print("\n" + "=" * 70)
print("[Phase 8] 保存结果 + JQ 策略")
print("=" * 70)

result = {
    'timestamp': datetime.now().isoformat(),
    'method': 'elasticnet_hmm',
    'standardization': 'rank_percentile_inverse_normal',
    'hmm_states': len(regime_set),
    'state_names': state_names,
    'regimes': {},
    'walk_forward': {},
    'backtest': bt_result,
}
for st, strat in all_strategies.items():
    result['regimes'][st] = strat
for st, wf in wf_results.items():
    result['walk_forward'][st] = wf

with open(RESULT_JSON, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
print(f"  结果: {RESULT_JSON}")

# JQ 生成
if JQ_AVAILABLE:
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    for st, strat in all_strategies.items():
        if 'error' in strat:
            continue
        weights = strat['weights']
        jq_w = {k: v for k, v in weights.items() if k in JQ_FACTOR_REGISTRY}
        missing = set(weights.keys()) - set(JQ_FACTOR_REGISTRY.keys())
        if len(jq_w) < 2:
            print(f"  {st}: SKIP (仅 {len(jq_w)} 因子有 JQ 映射)")
            continue
        total = sum(jq_w.values())
        jq_w = {k: v / total for k, v in jq_w.items()}
        
        fname = f'fa_hmm_elasticnet_{st}.py'
        fpath = STRATEGIES_DIR / fname
        try:
            gen = JQGenerator(weights=jq_w, name=f'hmm_en_{st}', mcap_min=5)
            gen.generate(str(fpath))
            # 验证生成文件可编译
            with open(fpath, 'r', encoding='utf-8') as fcheck:
                src = fcheck.read()
            compile(src, str(fpath), 'exec')
            print(f"  {st}: ✅ {fname} (compile OK)")
            if missing:
                print(f"        缺JQ映射: {missing}")
        except Exception as e:
            print(f"  {st}: ❌ {e}")

# 汇总
print("\n" + "=" * 70)
print("  完成!")
print("=" * 70)
for st, strat in all_strategies.items():
    if 'weights' in strat:
        top = sorted(strat['weights'].items(), key=lambda x: -x[1])[:3]
        print(f"  {st}: {' + '.join(f'{n}({w:.2f})' for n,w in top)}")
    else:
        print(f"  {st}: {strat.get('error', '?')}")
print(f"  结果: {RESULT_JSON}")
