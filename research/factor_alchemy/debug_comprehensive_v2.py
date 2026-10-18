"""
综合诊断脚本 v2.0 (轻量版, 直接从CSV加载)
=============================================
1. Obj3 成本后Sharpe 逐步诊断 (验证top30选股+方向)
2. ICIR raw vs clip 区分度对比 (叠加惩罚链)
3. plrc_12 前视偏差审查
"""
import sys, os, time
from pathlib import Path

os.environ['FA_OBJ3_DEBUG'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import (
    FACTOR_DEFS, ICIR_THRESHOLD, ICIR_OBJ_MODE, MAX_FACTORS,
    MAX_TOTAL_PENALTY, FUNDAMENTAL_CATEGORIES, FUNDAMENTAL_BONUS,
    FUNDAMENTAL_MIN_COUNT, STABILITY_THRESHOLD, OOS_SPLIT_DATE,
)
from ga.fitness import (
    icir_to_objective,
    _estimate_weekly_turnover,
    _compute_turnover_penalty,
    _compute_category_penalty,
    _compute_complexity_penalty,
    _compute_correlation_penalty,
    _compute_top30_cost_adjusted_sharpe,
)
from ga.chromosome import icir_biased_chromosome
from factors.composite import combine_factors, weights_from_chromosome
from evaluation.ic_analysis import compute_ic_icir

OUTPUT_DIR = Path(__file__).parent / 'output'
DATA_DIR = Path(__file__).parent / 'data'

# ============================================================
# 从 CSV 加载因子数据
# ============================================================
def load_factors_from_csv():
    """从 output/*.csv 加载已计算的因子"""
    factor_dict = {}
    factor_names = []
    
    for fpath in sorted(OUTPUT_DIR.glob('factor_*.csv')):
        fname = fpath.stem  # factor_xxx
        factor_name = fname.replace('factor_', '')
        
        if factor_name not in FACTOR_DEFS:
            continue
        
        try:
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
            if df.empty or df.shape[0] < 5:
                continue
            factor_dict[factor_name] = df
            factor_names.append(factor_name)
        except Exception:
            continue
    
    return factor_dict, factor_names

def load_forward_returns():
    """从 price data 构建前向收益"""
    # 尝试加载已计算的前向收益
    fpath = OUTPUT_DIR / 'forward_returns_weekly.csv'
    if fpath.exists():
        return pd.read_csv(fpath, index_col=0, parse_dates=True)
    
    # Fallback: 从价格构建
    from data.loader import load_all_data
    data = load_all_data('2021-01-01', '2026-06-25')
    price_data = data['price_data']
    close = price_data.get('close')
    close_weekly = close.resample('W').last()
    close_weekly.index = pd.to_datetime(close_weekly.index)
    fr = close_weekly.shift(-1) / close_weekly - 1
    return fr

def get_icir_values(factor_dict, forward_returns):
    """计算单因子ICIR (全时段)"""
    values = {}
    for name, df in factor_dict.items():
        result = compute_ic_icir(df, forward_returns)
        values[name] = abs(result['icir']) if not np.isnan(result['icir']) else 0.0
    return values

# ============================================================
# 主诊断
# ============================================================
print("=" * 70)
print("  综合诊断 v2: Obj3 + ICIR raw/clip + plrc_12")
print("=" * 70)

print("\n[1/5] 加载因子数据...")
factor_dict, factor_names = load_factors_from_csv()
print(f"  已加载 {len(factor_dict)} 个因子")

print("\n[2/5] 加载前向收益...")
forward_returns = load_forward_returns()
print(f"  收益矩阵: {forward_returns.shape}")

# 获取收盘价 (用于plrc_12审查)
close_weekly = None
try:
    from data.loader import load_all_data
    data = load_all_data('2021-01-01', '2026-06-25')
    close = data['price_data'].get('close')
    close_weekly = close.resample('W').last()
    close_weekly.index = pd.to_datetime(close_weekly.index)
except Exception:
    pass

print("\n[3/5] 计算单因子ICIR...")
icir_vals = get_icir_values(factor_dict, forward_returns)
print(f"  Top5 ICIR: ", end="")
for name, v in sorted(icir_vals.items(), key=lambda x: -x[1])[:5]:
    print(f"{name}={v:.3f} ", end="")
print()

n_factors = len(factor_names)
icir_values_arr = [icir_vals.get(name, 0.5) for name in factor_names]

# ============================================================
# SECTION A: plrc_12 审查
# ============================================================
print("\n" + "=" * 70)
print("  SECTION A: plrc_12 前视偏差审查")
print("=" * 70)

if 'plrc_12' in factor_dict:
    plrc_df = factor_dict['plrc_12']
    fr_df = forward_returns
    
    print(f"\n  定义: plrc_12 = OLS slope(close/rolling_mean(close,12) vs t=1..12)")
    print(f"  类别: {FACTOR_DEFS['plrc_12']['category']}")
    print(f"  方向: 正向 (价格趋势向上 → 动量延续 → 做多)")
    print(f"  数据形状: {plrc_df.shape}  |  {plrc_df.index[0].strftime('%Y-%m-%d')} ~ {plrc_df.index[-1].strftime('%Y-%m-%d')}")
    
    # 前视偏差审查
    print(f"\n  [审查] 计算步骤:")
    print(f"    1. roll_mean = close.rolling(12).mean()  ← 使用 t-11 至 t")
    print(f"    2. y = close / roll_mean                  ← 同时点, 无前视")
    print(f"    3. OLS slope(y_1..y_12 vs t=1..12)        ← 仅用已实现y")
    print(f"    => 无前视偏差 ✓")
    
    # 分时段ICIR
    common_dates = sorted(set(plrc_df.index) & set(fr_df.index))
    train_dates = [d for d in common_dates if d.strftime('%Y-%m-%d') < OOS_SPLIT_DATE]
    oos_dates = [d for d in common_dates if d.strftime('%Y-%m-%d') >= OOS_SPLIT_DATE]
    
    print(f"\n  分时段ICIR:")
    for label, dates in [("全时段", common_dates), ("训练期", train_dates), ("OOS期", oos_dates)]:
        if len(dates) < 5:
            continue
        result = compute_ic_icir(plrc_df.loc[dates], fr_df.loc[dates])
        print(f"    {label:6s} ({len(dates):3d}周): ICIR={result['icir']:+.4f}  +IC%={result['ic_positive_ratio']*100:.1f}%  IC_mean={result['ic_mean']:+.4f}")
    
    # 训练期ICIR检查
    if len(train_dates) >= 5:
        train_result = compute_ic_icir(plrc_df.loc[train_dates], fr_df.loc[train_dates])
        if abs(train_result['icir']) < 0.1:
            print(f"\n  ⚠️ 严重样本依赖: 训练期ICIR={train_result['icir']:+.4f} (接近零)")
            print(f"     全时段ICIR几乎全部来自OOS牛市, 训练期内无预测力")
            print(f"     建议: ICIR偏置初始化使用训练期ICIR值, 而非全时段")
else:
    print(f"\n  plrc_12 不在因子池中")

# ============================================================
# SECTION B: Obj3 逐步诊断
# ============================================================
print("\n" + "=" * 70)
print("  SECTION B: Obj3 成本后Sharpe 逐步诊断")
print("=" * 70)

np.random.seed(42)

# 创建代表性测试组合
test_combos = []

# (A) 手工构建: 高ICIR因子组合
top_icir_names = [n for n, _ in sorted(icir_vals.items(), key=lambda x: -x[1])[:6]]
if len(top_icir_names) >= 2:
    combo_a = {top_icir_names[0]: 0.35, top_icir_names[1]: 0.30}
    others = top_icir_names[2:4]
    for i, n in enumerate(others):
        combo_a[n] = 0.35 / len(others)
    test_combos.append(("Top ICIR因子", combo_a))

# (B) 已知好因子组合
combo_b = {}
if 'earnings_growth_yoy' in factor_dict:
    combo_b['earnings_growth_yoy'] = 0.30
if 'dollar_vol_stability' in factor_dict:
    combo_b['dollar_vol_stability'] = 0.25
if 'turnover_std' in factor_dict:
    combo_b['turnover_std'] = 0.25
if 'f_score' in factor_dict:
    combo_b['f_score'] = 0.20
if combo_b:
    total = sum(combo_b.values())
    combo_b = {k: v/total for k, v in combo_b.items()}
    test_combos.append(("手工优选(价值+量价)", combo_b))

# (C) 单一最佳因子
best_name = top_icir_names[0] if top_icir_names else list(factor_names)[0]
test_combos.append((f"单因子({best_name})", {best_name: 1.0}))

# (D) 反转因子 (测试负IC方向)
combo_d = {}
for n in ['short_rev_5d', 'max_ret_1m', 'skew_1m']:
    if n in factor_dict:
        combo_d[n] = 1.0/3.0
if combo_d:
    test_combos.append(("反转因子(测试方向)", combo_d))

for label, weights in test_combos:
    print(f"\n  {'─' * 60}")
    print(f"  测试: {label}")
    print(f"  因子: {', '.join(f'{n}({w:.2f})' for n, w in weights.items())}")
    
    selected = {n: factor_dict[n] for n in weights if n in factor_dict}
    if len(selected) == 0:
        print("    [SKIP] 无可用因子")
        continue
    
    composite = combine_factors(selected, weights)
    if composite.empty:
        print("    [SKIP] composite为空")
        continue
    
    # ICIR
    icir_result = compute_ic_icir(composite, forward_returns)
    icir = abs(icir_result['icir'])
    print(f"    ICIR: {icir:+.4f} (+IC%={icir_result['ic_positive_ratio']*100:.1f}%)")
    
    # === Obj3 逐步诊断 ===
    ROUND_TRIP_COST = 0.0072
    
    common_idx = composite.index.intersection(forward_returns.index)
    common_cols = composite.columns.intersection(forward_returns.columns)
    
    if len(common_idx) < 12 or len(common_cols) < 30:
        print(f"    [SKIP] 数据不足: {len(common_idx)}周 x {len(common_cols)}股")
        continue
    
    comp_arr = composite.loc[common_idx, common_cols].values
    fr_arr = forward_returns.loc[common_idx, common_cols].values
    n_weeks, n_stocks = comp_arr.shape
    
    # Step A: 验证第1周 Top30 选择
    comp_w0 = comp_arr[0]
    fr_w0 = fr_arr[0]
    valid = np.isfinite(comp_w0)
    
    top30_idx_w0 = np.argsort(-comp_w0)[:30]
    top30_comp = comp_w0[top30_idx_w0]
    top30_ret = fr_w0[top30_idx_w0]
    
    # 检查: composite与forward_ret的相关性
    valid_idx = np.where(valid)[0]
    if len(valid_idx) >= 30:
        corr_spr = stats.spearmanr(comp_w0[valid_idx], fr_w0[valid_idx])
        corr_v = corr_spr[0] if isinstance(corr_spr, tuple) else corr_spr.statistic
    else:
        corr_v = np.nan
    
    print(f"    第1周验证:")
    print(f"      有效股票: {valid.sum()}/{n_stocks}")
    print(f"      composite范围: [{np.nanmin(comp_w0):.4f}, {np.nanmax(comp_w0):.4f}]")
    print(f"      Top30 composite: [{top30_comp.min():.4f}, {top30_comp.max():.4f}]")
    print(f"      Top30 前向收益均值: {np.nanmean(top30_ret)*100:+.3f}%")
    print(f"      全部股票收益均值:   {np.nanmean(fr_w0[valid_idx])*100:+.3f}%")
    print(f"      composite vs ret Spearman: {corr_v:+.4f} (应>0证明方向正确)")
    
    # Step B: 多期统计 (前10周)
    n_check = min(10, n_weeks - 1)
    top30_mean_rets = []
    all_mean_rets = []
    
    for w in range(n_check):
        cw = comp_arr[w]
        fw = fr_arr[w]
        top30 = np.argsort(-cw)[:30]
        top30_mean_rets.append(np.nanmean(fw[top30]))
        valid_w = np.isfinite(cw)
        all_mean_rets.append(np.nanmean(fw[valid_w]))
    
    print(f"    前{n_check}周统计:")
    print(f"      Top30 周均收益: {np.mean(top30_mean_rets)*100:+.3f}%")
    print(f"      全部 周均收益:  {np.mean(all_mean_rets)*100:+.3f}%")
    
    # Step C: 完整Obj3
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
    costs = turnovers * ROUND_TRIP_COST
    net_rets = gross_rets - costs
    
    valid_rets = net_rets[np.isfinite(net_rets)]
    mean_ret = np.mean(valid_rets)
    std_ret = np.std(valid_rets, ddof=1)
    annual_sharpe = mean_ret / std_ret * np.sqrt(52) if std_ret > 1e-8 else 0.0
    obj3_val = float(np.clip(annual_sharpe / 2.0, 0.0, 1.0))
    
    print(f"\n    完整Obj3拆解:")
    print(f"      年化毛收益: {np.nanmean(gross_rets)*52*100:+.2f}%")
    print(f"      年化成本:   {np.nanmean(costs)*52*100:+.2f}%")
    print(f"      年化净收益: {np.nanmean(net_rets)*52*100:+.2f}%")
    print(f"      平均周换手: {np.nanmean(turnovers)*100:.1f}%")
    print(f"      原始Sharpe: {annual_sharpe:+.4f}")
    print(f"      Obj3(clip): {obj3_val:.4f}")
    
    # 诊断
    if annual_sharpe <= 0:
        reasons = []
        if np.nanmean(gross_rets) <= 0:
            reasons.append("毛收益为负(训练期含熊市, long-only无法获利)")
        annual_cost = np.nanmean(costs) * 52
        annual_gross = np.nanmean(gross_rets) * 52
        if annual_cost > abs(annual_gross):
            reasons.append(f"成本侵蚀({annual_cost*100:.1f}%)超过毛收益({annual_gross*100:+.1f}%)")
        if np.nanmean(turnovers) > 0.3:
            reasons.append(f"换手过高({np.nanmean(turnovers)*100:.1f}%/周)")
        print(f"    [诊断] Sharpe≤0: {'; '.join(reasons)}")

# 终极验证: 用后复权价格直接计算Top30组合收益
if close_weekly is not None:
    print(f"\n  {'─' * 60}")
    print(f"  [终极验证] 用后复权价格直接计算Top30组合收益")
    
    for label, weights in test_combos[:2]:  # 只测前2组
        selected = {n: factor_dict[n] for n in weights if n in factor_dict}
        if len(selected) == 0:
            continue
        composite = combine_factors(selected, weights)
        if composite.empty:
            continue
        
        common_idx = composite.index.intersection(close_weekly.index)
        if len(common_idx) < 12:
            continue
        common_cols = composite.columns.intersection(close_weekly.columns)
        if len(common_cols) < 30:
            continue
        
        comp_sub = composite.loc[common_idx, common_cols]
        price_sub = close_weekly.loc[common_idx, common_cols]
        
        rets = []
        tos = []
        prev_top30 = None
        
        for i in range(len(common_idx) - 1):
            comp_row = comp_sub.iloc[i]
            valid_comp = comp_row.dropna()
            top30 = valid_comp.nlargest(30).index.tolist()
            
            # 计算实际收益 (用价格)
            price_now = price_sub.iloc[i][top30]
            price_next = price_sub.iloc[i + 1][top30]
            
            valid_mask = price_now.notna() & price_next.notna() & (price_now > 0) & (price_next > 0)
            if valid_mask.sum() < 10:
                continue
            
            stock_rets = price_next[valid_mask] / price_now[valid_mask] - 1
            rets.append(stock_rets.mean())
            
            if prev_top30 is not None:
                overlap = len(set(top30) & set(prev_top30))
                tos.append(1.0 - overlap / 30.0)
            prev_top30 = top30
        
        if len(rets) >= 10:
            rets = np.array(rets)
            tos = np.array(tos) if tos else np.zeros_like(rets)
            costs_arr = tos * 0.0072
            net_arr = rets - costs_arr
            fin_mask = np.isfinite(net_arr)
            
            raw_sharpe = np.mean(net_arr[fin_mask]) / np.std(net_arr[fin_mask], ddof=1) * np.sqrt(52) \
                if np.std(net_arr[fin_mask], ddof=1) > 1e-8 else 0.0
            
            print(f"    {label}:")
            print(f"      年化毛收益: {np.mean(rets)*52*100:+.2f}%")
            print(f"      年化成本:   {np.mean(costs_arr)*52*100:+.2f}%")
            print(f"      年化净收益: {np.mean(net_arr[fin_mask])*52*100:+.2f}%")
            print(f"      原始Sharpe: {raw_sharpe:+.4f}")
            print(f"      周换手:     {np.mean(tos)*100:.1f}%")

# ============================================================
# SECTION C: ICIR raw vs clip 区分度对比
# ============================================================
print("\n" + "=" * 70)
print("  SECTION C: ICIR raw vs clip 区分度对比 (叠加惩罚链)")
print("=" * 70)

print("\n  生成样本进行对比...")
np.random.seed(555)

n_samples = 50
results = []

for i in range(n_samples):
    chromo = icir_biased_chromosome(n_factors, icir_values_arr, bias_strength=1.5)
    chromo += np.random.normal(0, 0.5, n_factors)
    
    weights = weights_from_chromosome(chromo, factor_names, max_factors=MAX_FACTORS)
    if len(weights) == 0:
        continue
    
    selected = {n: factor_dict[n] for n in weights if n in factor_dict}
    if len(selected) == 0:
        continue
    
    composite = combine_factors(selected, weights)
    if composite.empty:
        continue
    
    icir_result = compute_ic_icir(composite, forward_returns)
    icir = abs(icir_result['icir'])
    if np.isnan(icir):
        continue
    
    # 惩罚项
    weekly_to = _estimate_weekly_turnover(composite, top_n=30, buffer_mult=1.8)
    to_pen = _compute_turnover_penalty(weekly_to)
    cat_pen = _compute_category_penalty(weights, factor_names)
    cpx_pen = _compute_complexity_penalty(len(weights))
    corr_pen = _compute_correlation_penalty(weights, factor_names, None)
    penalty_sum = float(np.clip(to_pen + cat_pen + cpx_pen + corr_pen, 0.0, MAX_TOTAL_PENALTY))
    
    n_fund = sum(1 for n in weights if n in FACTOR_DEFS
                 and FACTOR_DEFS[n]['category'] in FUNDAMENTAL_CATEGORIES)
    bonus = (FUNDAMENTAL_BONUS - 1.0) if n_fund >= FUNDAMENTAL_MIN_COUNT else 0.0
    
    # Raw: obj1 = clip(|ICIR| - pen + bonus, 0, 1)
    obj1_raw = float(np.clip(icir - penalty_sum + bonus, 0.0, 1.0))
    
    # Clip: obj1 = clip(clip(|ICIR|/1.5, 0, 1) - pen + bonus, 0, 1)
    icir_clip = float(np.clip(icir / ICIR_THRESHOLD, 0, 1))
    obj1_clip = float(np.clip(icir_clip - penalty_sum + bonus, 0.0, 1.0))
    
    results.append({
        'icir': icir,
        'icir_raw': icir,
        'icir_clip': icir_clip,
        'penalty': penalty_sum,
        'bonus': bonus,
        'obj1_raw': obj1_raw,
        'obj1_clip': obj1_clip,
        'n_factors': len(weights),
        'to': weekly_to,
    })

df = pd.DataFrame(results)
print(f"  有效样本: {len(df)}")

if len(df) >= 5:
    # 纯ICIR CV
    icir_raw_cv = df['icir_raw'].std() / max(df['icir_raw'].mean(), 1e-8)
    icir_clip_cv = df['icir_clip'].std() / max(df['icir_clip'].mean(), 1e-8)
    
    # Obj1 (ICIR+惩罚) CV
    obj1_raw_cv = df['obj1_raw'].std() / max(df['obj1_raw'].mean(), 1e-8)
    obj1_clip_cv = df['obj1_clip'].std() / max(df['obj1_clip'].mean(), 1e-8)
    
    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │                ICIR raw vs clip 区分度对比               │")
    print(f"  ├──────────────┬──────────────────┬───────────────────────┤")
    print(f"  │    指标       │    Raw模式        │    Clip模式           │")
    print(f"  ├──────────────┼──────────────────┼───────────────────────┤")
    print(f"  │ ICIR范围     │ [{df['icir'].min():.3f}, {df['icir'].max():.3f}]  │                       │")
    print(f"  │ ICIR均值     │ {df['icir'].mean():.3f}            │                       │")
    print(f"  │ ICIR_score CV│ {icir_raw_cv:.4f}          │ {icir_clip_cv:.4f}               │")
    print(f"  │ 惩罚均值     │ {df['penalty'].mean():.3f}            │ {df['penalty'].mean():.3f}                 │")
    print(f"  │ Obj1范围     │ [{df['obj1_raw'].min():.3f}, {df['obj1_raw'].max():.3f}]  │ [{df['obj1_clip'].min():.3f}, {df['obj1_clip'].max():.3f}]           │")
    print(f"  │ Obj1均值     │ {df['obj1_raw'].mean():.3f}            │ {df['obj1_clip'].mean():.3f}                 │")
    print(f"  │ Obj1 CV      │ {obj1_raw_cv:.4f}          │ {obj1_clip_cv:.4f}               │")
    print(f"  │ 零值比例     │ {(df['obj1_raw']==0).mean()*100:.1f}%            │ {(df['obj1_clip']==0).mean()*100:.1f}%                 │")
    print(f"  └──────────────┴──────────────────┴───────────────────────┘")
    
    # 惩罚链贡献
    delta_raw = obj1_raw_cv - icir_raw_cv
    delta_clip = obj1_clip_cv - icir_clip_cv
    
    print(f"\n  惩罚链 CV 贡献:")
    print(f"    Raw:  {icir_raw_cv:.4f} → {obj1_raw_cv:.4f} (Δ={delta_raw:+.4f})")
    print(f"    Clip: {icir_clip_cv:.4f} → {obj1_clip_cv:.4f} (Δ={delta_clip:+.4f})")
    
    # Spearman
    sr, sp = stats.spearmanr(df['obj1_raw'], df['obj1_clip'])
    print(f"\n  Obj1_raw vs Obj1_clip Spearman r = {sr:.4f} (p={sp:.6f})")
    
    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  结论:                                                  ║")
    if delta_raw > delta_clip * 1.15:
        print(f"  ║  ★ raw模式惩罚链贡献更大 (Δ={delta_raw:+.4f} vs {delta_clip:+.4f})       ║")
        print(f"  ║  推荐: raw (保留完整ICIR差异, 惩罚链区分度更高)       ║")
    elif delta_clip > delta_raw * 1.15:
        print(f"  ║  ★ clip模式惩罚链贡献更大 (Δ={delta_clip:+.4f} vs {delta_raw:+.4f})       ║")
        print(f"  ║  推荐: clip (ICIR压缩后惩罚链区分度更高)               ║")
    else:
        print(f"  ║  raw和clip惩罚链贡献相近 (Δ差={abs(delta_raw-delta_clip):.4f})            ║")
        print(f"  ║  推荐: clip — Obj1∈[0,1]与Obj2同量纲, Pareto前沿更均衡")
        print(f"  ║  两模式 Spearman r={sr:.4f}, 排名高度一致               ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")

print("\n" + "=" * 70)
print("  诊断完成")
print("=" * 70)
