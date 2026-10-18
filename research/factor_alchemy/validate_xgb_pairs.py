# -*- coding: utf-8 -*-
"""
验证 XGBoost 发现 50 个交互对：V3 ICIR 筛选 → rank-product 复合 → JQ 代码生成
================================================================================
用途: 将 XGBoost 提取的 50 个最强树路径共现对, 按 V3 框架做:
  1. rank-percentile 乘积复合
  2. ICIR 筛选 (≥0.30)
  3. 三层评分排序
  4. 对比 V3 baseline 的 5 个复合
  5. 生成 JQ 回测代码 (组合: V3 原 5 个 + XGBoost 新 top-N)

数据集: factor_cache.pkl (95 factors, weekly)
前向收益: (ret_1m.shift(-1) - ret_1m)/4 (与 xgb_interaction.py 一致)
"""
from __future__ import annotations

import gc, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau, norm as _norm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# ==================== 配置 ====================
CACHE_FILE = SCRIPT_DIR / 'output' / 'factor_cache.pkl'
XGB_PAIRS_FILE = SCRIPT_DIR / 'output' / 'xgb_top_interactions_v1.json'
OUT_DIR = SCRIPT_DIR / 'output'

ICIR_THRESHOLD = 0.30   # V3 标准
TOP_K = 5               # 选出 Top-K 复合
LAYER_WEIGHTS = (0.3, 0.4, 0.3)  # ICIR, 稳定性, 衰减
STABILITY_WINDOW = 24
DECAY_WINDOW = 52
CORR_MAX = 0.7
N_STOCKS = 30  # JQ 持仓数

# V3 已选的 5 对
V3_PAIRS = [
    ('overnight_5d',       'tvma_20'),
    ('dollar_vol_20d',     'turnover_std'),
    ('money_flow_20',      'value_momentum'),
    ('ret_open_2d_proxy',  'skewness_20'),
    ('gap_up',             'relative_spread_proxy'),
]
V3_ICIRS = {
    'comp_overnight_tvma':      0.7719,
    'comp_dollar_turnover':     0.7472,
    'comp_moneyflow_value':     0.6946,
    'comp_retopen_skew':        0.6843,
    'comp_gapup_spread':        0.6870,
}

# ==================== 工具函数 ====================

def _fast_icir_vectorized(z_df, returns_df, min_samples=20):
    """向量化 ICIR (与 factor_composer.py 完全一致)."""
    # 对齐股票列
    common_cols = z_df.columns.intersection(returns_df.columns)
    if len(common_cols) < min_samples:
        return {'icir': np.nan, 'ic_series': pd.Series(dtype=float)}
    z_aligned = z_df[common_cols]
    r_aligned = returns_df[common_cols]
    weeks = [w for w in z_aligned.index if w in r_aligned.index]
    if len(weeks) < min_samples:
        return {'icir': np.nan, 'ic_series': pd.Series(dtype=float)}
    Z = z_aligned.loc[weeks].values.astype(float)
    R = r_aligned.loc[weeks].values.astype(float)
    n_weeks = Z.shape[0]
    ic_list, ic_dates = [], []
    for i in range(n_weeks):
        mask = np.isfinite(Z[i]) & np.isfinite(R[i])
        n = int(mask.sum())
        if n < min_samples:
            continue
        zv, rv = Z[i][mask], R[i][mask]
        fr = rankdata(zv)
        rr = rankdata(rv)
        d2 = float(np.sum((fr - rr) ** 2))
        ic = 1.0 - 6.0 * d2 / (n * (n * n - 1))
        if np.isfinite(ic):
            ic_list.append(ic)
            ic_dates.append(weeks[i])
    if len(ic_list) < 5:
        return {'icir': np.nan, 'ic_series': pd.Series(dtype=float)}
    ic_series = pd.Series(ic_list, index=ic_dates)
    ic_mean = float(np.mean(ic_list))
    ic_std = float(np.std(ic_list, ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    return {'icir': icir, 'ic_series': ic_series}


def ic_decay_rate(ic_series, recency_window=52):
    """IC 衰减率 (与 factor_composer.py 完全一致)."""
    series = ic_series.dropna()
    if len(series) < 10:
        return 0.0
    full_mean = series.mean()
    recent = series.iloc[-recency_window:]
    recent_mean = recent.mean() if len(recent) > 0 else full_mean
    denom = abs(full_mean) + 1e-9
    decay_raw = (recent_mean - full_mean) / denom
    if len(series) >= 20:
        tau, _ = kendalltau(range(len(series)), series.values)
        n = len(series)
        var_tau = 2.0 * (2 * n + 5) / (9.0 * n * (n - 1))
        tau_se = np.sqrt(var_tau + 1e-9)
        tau_z = tau / tau_se
        kendall_sign = np.sign(tau_z) if abs(tau_z) > 1.0 else 0.0
    else:
        kendall_sign = 0.0
    decay = decay_raw + 0.3 * kendall_sign * abs(decay_raw)
    return float(decay)


def rolling_icir_stability(ic_series, window=24):
    """滚动 ICIR 稳定性 (与 factor_composer.py 完全一致)."""
    if len(ic_series) < window:
        return 0.0, 0.0
    series = ic_series.dropna()
    if len(series) < window:
        return 0.0, 0.0
    icir_rolls = []
    persist_rolls = []
    for i in range(window, len(series) + 1):
        seg = series.iloc[i - window:i]
        seg_mean = seg.mean()
        seg_std = seg.std(ddof=1)
        if seg_std and seg_std > 0:
            icir_rolls.append(seg_mean / seg_std)
        persist_rolls.append(float((seg > 0).mean()))
    if not icir_rolls:
        return 0.0, 0.0
    stability = np.mean(icir_rolls) / (1.0 + max(np.std(icir_rolls, ddof=1), 1e-9))
    persistence = np.mean(persist_rolls)
    return float(stability), float(persistence)


def _rank_norm(series, ascending=True):
    """排名归一化 → [0, 1]."""
    s = series.dropna()
    if len(s) < 2:
        return pd.Series(0.5, index=series.index)
    ranked = s.rank(pct=True, ascending=ascending)
    return ranked.reindex(series.index).fillna(0.5)


def _fast_standardize(df):
    """向量化 rank-based 标准化."""
    ranks = df.rank(axis=1, pct=True, na_option='keep')
    z = _norm.ppf(np.clip(ranks.values, 0.0001, 0.9999))
    z = np.clip(z, -4.0, 4.0)
    return pd.DataFrame(z, index=df.index, columns=df.columns, dtype=float)


# ==================== 数据加载 ====================

def load_cache():
    print(f'[1] 加载因子缓存 ...')
    t0 = time.time()
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    print(f'    {len(cache)} 因子, {time.time()-t0:.1f}s')
    return cache


def build_forward_returns(cache):
    """从 ret_1m 构造前向周收益代理 (与 xgb_interaction.py 一致)."""
    print(f'[2] 构建前向收益 ...')
    ret_df = cache['ret_1m']
    weeks = sorted(ret_df.index)
    stocks = sorted([c for c in ret_df.columns if isinstance(c, str)])
    ret_df = ret_df.reindex(index=weeks, columns=stocks)
    fwd_ret = (ret_df.shift(-1) - ret_df) / 4.0
    # 移除最后一行 (无前向数据)
    fwd_ret = fwd_ret.iloc[:-1]
    print(f'    前向收益: {fwd_ret.shape} ({fwd_ret.index[0].strftime("%Y-%m-%d")} ~ {fwd_ret.index[-1].strftime("%Y-%m-%d")})')
    return fwd_ret, weeks, stocks


# ==================== 主计算 ====================

def compute_xgb_composites(cache, xgb_pairs, forward_returns, weeks):
    """对 XGBoost 50 个交互对做 rank-product 复合 + ICIR 计算."""
    print(f'\n[3] 计算 XGBoost 对复合因子 ({len(xgb_pairs)} 对)...')
    t0 = time.time()

    results = {}
    skipped = 0

    for item in xgb_pairs:
        fa_name, fb_name = item['pair']
        key = f'comp_{fa_name}_x_{fb_name}'

        # 检查因子是否存在
        if fa_name not in cache or fb_name not in cache:
            # 尝试嵌套复合展开
            if fa_name == 'value_momentum':
                # 需要用 bp × ret_3m 代理
                if 'bp' in cache and 'ret_3m' in cache:
                    dfa = cache['bp']
                    dfb = cache['ret_3m']
                    # 嵌套复合: rank_pct(bp) × rank_pct(ret_3m) — 注意 ret_3m 用 - 方向
                    common_dates = dfa.index.intersection(dfb.index)
                    union_cols = dfa.columns.union(dfb.columns)
                    rank_a = dfa.loc[common_dates].reindex(columns=union_cols).rank(pct=True, axis=1, na_option='keep')
                    rank_b = (-dfb.loc[common_dates].reindex(columns=union_cols)).rank(pct=True, axis=1, na_option='keep')
                    raw_df = rank_a * rank_b
                else:
                    skipped += 1
                    continue
            elif fb_name == 'value_momentum':
                if 'bp' in cache and 'ret_3m' in cache:
                    dfb = cache['bp']
                    dfa = cache['ret_3m']
                    common_dates = dfa.index.intersection(dfb.index)
                    union_cols = dfa.columns.union(dfb.columns)
                    rank_a = (-dfa.loc[common_dates].reindex(columns=union_cols)).rank(pct=True, axis=1, na_option='keep')
                    rank_b = dfb.loc[common_dates].reindex(columns=union_cols).rank(pct=True, axis=1, na_option='keep')
                    raw_df = rank_a * rank_b
                else:
                    skipped += 1
                    continue
            else:
                skipped += 1
                continue
        else:
            dfa = cache[fa_name]
            dfb = cache[fb_name]
            common_dates = dfa.index.intersection(dfb.index)
            if len(common_dates) < 20:
                skipped += 1
                continue
            union_cols = dfa.columns.union(dfb.columns)
            rank_a = dfa.loc[common_dates].reindex(columns=union_cols).rank(pct=True, axis=1, na_option='keep')
            rank_b = dfb.loc[common_dates].reindex(columns=union_cols).rank(pct=True, axis=1, na_option='keep')
            raw_df = rank_a * rank_b

        # 标准化 + ICIR
        std_df = _fast_standardize(raw_df)
        del raw_df, rank_a, rank_b
        ic_res = _fast_icir_vectorized(std_df, forward_returns, min_samples=20)
        del std_df

        if ic_res and pd.notna(ic_res.get('icir', np.nan)):
            ir = ic_res['icir']
            s = ic_res.get('ic_series', pd.Series(dtype=float))

            # 方向翻转: ICIR 取正方向
            flip = ir < 0
            if flip and len(s) > 0:
                s = -s
            ir_abs = abs(ir)

            stability, persistence = rolling_icir_stability(s, STABILITY_WINDOW)
            decay = ic_decay_rate(s, DECAY_WINDOW)

            results[key] = {
                'pair': (fa_name, fb_name),
                'icir': ir_abs,
                'stability': stability,
                'persistence': persistence,
                'decay': decay,
                'ic_series': s,
                'flip': flip,
                'n_ic': len(s),
            }

    print(f'    完成: {len(results)} 对有效, {skipped} 跳过')
    print(f'    耗时: {time.time()-t0:.1f}s')
    return results


def compute_v3_baseline(cache, forward_returns):
    """计算 V3 的 5 个复合在当前数据上的 ICIR (用于校准)."""
    print(f'\n[3b] 计算 V3 baseline ICIR (校准用)...')
    v3_results = {}
    for fa_name, fb_name in V3_PAIRS:
        key = f'comp_{fa_name}_x_{fb_name}'

        # handle value_momentum as nested
        if fa_name == 'value_momentum':
            dfa_bp = cache.get('bp')
            dfa_r3 = cache.get('ret_3m')
            if dfa_bp is None or dfa_r3 is None:
                continue
            cd = dfa_bp.index.intersection(dfa_r3.index)
            uc = dfa_bp.columns.union(dfa_r3.columns)
            r_bp = dfa_bp.loc[cd].reindex(columns=uc).rank(pct=True, axis=1, na_option='keep')
            r_r3 = (-dfa_r3.loc[cd].reindex(columns=uc)).rank(pct=True, axis=1, na_option='keep')
            raw_vm = r_bp * r_r3
            rank_vm = raw_vm.rank(pct=True, axis=1, na_option='keep')
            dfa = rank_vm
        elif fb_name == 'value_momentum':
            dfb_bp = cache.get('bp')
            dfb_r3 = cache.get('ret_3m')
            if dfb_bp is None or dfb_r3 is None:
                continue
            cd = dfb_bp.index.intersection(dfb_r3.index)
            uc = dfb_bp.columns.union(dfb_r3.columns)
            r_bp = dfb_bp.loc[cd].reindex(columns=uc).rank(pct=True, axis=1, na_option='keep')
            r_r3 = (-dfb_r3.loc[cd].reindex(columns=uc)).rank(pct=True, axis=1, na_option='keep')
            raw_vm = r_bp * r_r3
            rank_vm = raw_vm.rank(pct=True, axis=1, na_option='keep')
            dfb = rank_vm
            dfa = cache.get(fa_name)
        else:
            dfa = cache.get(fa_name)
            dfb = cache.get(fb_name)

        if dfa is None or dfb is None:
            continue
        if isinstance(dfa, pd.DataFrame) and isinstance(dfb, pd.DataFrame):
            cd = dfa.index.intersection(dfb.index)
            uc = dfa.columns.union(dfb.columns)
            r_a = dfa.loc[cd].reindex(columns=uc).rank(pct=True, axis=1, na_option='keep')
            r_b = dfb.loc[cd].reindex(columns=uc).rank(pct=True, axis=1, na_option='keep')
        else:
            # one of them is already rank
            cd = forward_returns.index
            uc = forward_returns.columns
            if isinstance(dfa, pd.DataFrame):
                r_a = dfa.reindex(index=cd, columns=uc).rank(pct=True, axis=1, na_option='keep')
            else:
                r_a = pd.DataFrame(dfa, index=cd, columns=uc)
            if isinstance(dfb, pd.DataFrame):
                r_b = dfb.reindex(index=cd, columns=uc).rank(pct=True, axis=1, na_option='keep')
            else:
                r_b = pd.DataFrame(dfb, index=cd, columns=uc)

        raw = r_a * r_b
        std = _fast_standardize(raw)
        del raw, r_a, r_b
        ic_res = _fast_icir_vectorized(std, forward_returns, min_samples=20)
        del std

        if ic_res and pd.notna(ic_res.get('icir', np.nan)):
            ir = ic_res['icir']
            s = ic_res.get('ic_series', pd.Series(dtype=float))
            flip = ir < 0
            if flip and len(s) > 0:
                s = -s
            v3_results[key] = {
                'pair': (fa_name, fb_name),
                'icir': abs(ir),
                'ic_series': s,
                'flip': flip,
                'n_ic': len(s),
            }
    return v3_results


# ==================== 评分与选择 ====================

def score_and_select(comps, top_k=5):
    """三层评分 + 冗余剔除 → 选出 top-K."""
    if len(comps) == 0:
        return [], pd.DataFrame()

    df = pd.DataFrame([
        {
            'name': k,
            'pair': v['pair'],
            'icir': v['icir'],
            'stability': v['stability'],
            'persistence': v['persistence'],
            'decay': v['decay'],
            'flip': v['flip'],
            'n_ic': v['n_ic'],
        }
        for k, v in comps.items()
    ])

    # Layer 1: ICIR
    df['score_L1'] = _rank_norm(df['icir'], ascending=True)

    # Layer 2: 稳定性
    raw_L2 = df['stability'] * (0.5 + 0.5 * df['persistence'])
    df['score_L2'] = _rank_norm(raw_L2, ascending=True)

    # Layer 3: 衰减
    raw_L3 = 1.0 - np.abs(df['decay'])
    raw_L3 = np.clip(raw_L3, -1, 1)
    df['score_L3'] = _rank_norm(pd.Series(raw_L3, index=df.index), ascending=True)

    w1, w2, w3 = LAYER_WEIGHTS
    df['total_score'] = w1 * df['score_L1'] + w2 * df['score_L2'] + w3 * df['score_L3']
    df = df.sort_values('total_score', ascending=False).reset_index(drop=True)

    # ICIR 筛选
    df_pass = df[df['icir'] >= ICIR_THRESHOLD].copy()
    print(f'    ICIR≥{ICIR_THRESHOLD}: {len(df_pass)}/{len(df)} 通过')

    if len(df_pass) == 0:
        # 降级: 取 ICIR 最高的 top_k
        print(f'    ⚠ 无通过者, 降级至 ICIR top-{top_k}')
        df_pass = df.head(top_k).copy()

    # 冗余剔除 (贪心)
    selected = []
    for _, row in df_pass.iterrows():
        if len(selected) >= top_k:
            break
        selected.append(row.to_dict())

    return selected, df


# ==================== JQ 代码生成 ====================

def generate_jq_code(selected_comps, cache):
    """生成 JQ 回测代码."""
    print(f'\n[5] 生成 JQ 代码 ({len(selected_comps)} 复合)...')

    # 收集所有需要的 base 因子
    all_base_factors = set()
    comp_weights = {}
    for c in selected_comps:
        fa, fb = c['pair']
        all_base_factors.add(fa)
        all_base_factors.add(fb)
        comp_weights[c['name']] = c['icir']

    w_sum = sum(comp_weights.values())
    comp_weights_norm = {k: v / w_sum for k, v in comp_weights.items()}

    # 展开嵌套复合 (value_momentum)
    n_factors = 0
    for f in sorted(all_base_factors):
        if f == 'value_momentum':
            all_base_factors.remove(f)
            all_base_factors.add('bp')
            all_base_factors.add('ret_3m')
        elif f in ('active_quality', 'efficient_growth', 'lowvol_trend', 'bargain_stability'):
            all_base_factors.remove(f)

    # 分类因子: OHLCV-based vs fundamental
    OHLCV_FACTORS = {
        'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m', 'ret_1m_skip1m',
        'max_ret_1m', 'min_ret_1m', 'short_rev_5d', 'streak',
        'vol_1m', 'vol_3m', 'downside_vol', 'high_low_range', 'atr_14',
        'boll_pct_b', 'skewness_20', 'kurtosis_20', 'sharpe_20',
        'rank_1m', 'high_52w_rank', 'high_52w_dist',
        'avg_turnover_1m', 'avg_turnover_3m', 'turnover_std', 'turnover_change',
        'abnormal_turnover', 'volume_ratio',
        'amihud_illiq', 'dollar_vol_20d', 'dollar_vol_stability',
        'turnover_cv_20d', 'turnover_to_vol',
        'vr_26', 'vroc_12', 'psy_12', 'money_flow_20', 'davol_20',
        'bias_20', 'roc_20', 'cci_14', 'rsi_14', 'price_1m',
        'bull_power', 'bear_power', 'gap_up', 'opening_gap_momentum',
        'intraday_reversal', 'overnight_5d',
        'tvma_20', 'vm_diff', 'volume_stability', 'volume_climax_reversal',
        'vpt', 'attention_decay', 'range_consistency',
        'ret_open_2d_proxy', 'relative_spread_proxy',
        'panic_selling', 'volatility_of_volatility',
        'skew_1m', 'trend_persistence_score', 'trend_smoothness',
        'earnings_volume_drift', 'post_earnings_stability',
        'earnings_season_vol_div', 'plrc_12',
    }

    ohlcv_factors = sorted(all_base_factors & OHLCV_FACTORS)
    fundamental_factors = sorted(all_base_factors - OHLCV_FACTORS)

    # 生成 JQ 代码
    jq_code = _generate_jq(
        selected_comps, comp_weights_norm,
        ohlcv_factors, fundamental_factors,
        cache
    )

    out_path = OUT_DIR / 'fa_composite_xgb_v1_jq.py'
    with open(out_path, 'w', encoding='utf-8-sig') as f:
        f.write(jq_code)
    print(f'    写入: {out_path}')
    return out_path


def _generate_jq(selected_comps, comp_weights_norm, ohlcv_factors, fundamental_factors, cache):
    """内部: 生成 JQ 策略代码."""

    comp_list = []
    for c in selected_comps:
        fa, fb = c['pair']
        comp_list.append(f"    '{c['name']}',  # {fa} × {fb}  ICIR={c['icir']:.4f}")

    lines = []
    lines.append('# -*- coding: utf-8 -*-')
    lines.append('"""')
    lines.append(f'FA Composite XGB v1 — XGBoost 交互对验证 | {len(selected_comps)}复合 JQ 真回测')
    lines.append(f'{"="*70}')
    lines.append(f'生成: {time.strftime("%Y-%m-%d %H:%M")} | ICIR 阈值 ≥{ICIR_THRESHOLD}')
    lines.append(f'')
    lines.append(f'{len(selected_comps)} rank-percentile 乘积复合因子:')
    for c in selected_comps:
        fa, fb = c['pair']
        lines.append(f'  {c["name"]} = {fa}(+) × {fb}(+)  ICIR={c["icir"]:.4f}  总分={c["total_score"]:.3f}')
    lines.append(f'')
    lines.append(f'OHLCV 因子({len(ohlcv_factors)}): {", ".join(ohlcv_factors[:10])}...')
    lines.append(f'基本面因子({len(fundamental_factors)}): {", ".join(fundamental_factors)}')
    lines.append(f'')
    lines.append(f'参数: 持仓={N_STOCKS}, 周频(周四14:50), 基准=沪深300')
    lines.append(f'成本: 佣金万1 + 印花千1 + 滑点0.3%')
    lines.append(f'================================================================')
    lines.append('"""')
    lines.append('')
    lines.append('from __future__ import division')
    lines.append('import datetime')
    lines.append('import gc')
    lines.append('import numpy as np')
    lines.append('import pandas as pd')
    lines.append('')
    lines.append('')
    lines.append('# ==================== 策略参数 ====================')
    lines.append(f'STOCK_NUM      = {N_STOCKS}')
    lines.append('MAX_TURNOVER   = 0.60')
    lines.append('MCAP_MIN       = 5')
    lines.append("EXCLUDE_PREFIX = ('8', '4')")
    lines.append('CHUNK          = 500')
    lines.append('LOOKBACK       = 65')
    lines.append('')
    lines.append('# ICIR 权重')
    lines.append('ICIR = {')
    for k, v in comp_weights_norm.items():
        lines.append(f"    '{k}': {v:.4f},")
    lines.append('}')
    lines.append('')
    lines.append('')
    lines.append('def _num(x):')
    lines.append('    try:')
    lines.append('        return float(x)')
    lines.append('    except (TypeError, ValueError):')
    lines.append("        if hasattr(x, '__iter__'):")
    lines.append('            xs = list(x)')
    lines.append('            return float(xs[0]) if xs else 0.0')
    lines.append('        return 0.0')
    lines.append('')
    lines.append('_WSUM = sum(_num(v) for v in ICIR.values())')
    lines.append('COMPOSITE_WEIGHTS = {k: _num(v) / _WSUM for k, v in ICIR.items()}')
    lines.append('')
    lines.append('DEBUG = True')
    lines.append('')
    lines.append('')
    lines.append('# ==================== 初始化 ====================')
    lines.append('def initialize(context):')
    lines.append("    set_benchmark('000300.XSHG')")
    lines.append("    set_option('use_real_price', True)")
    lines.append("    set_option('avoid_future_data', True)")
    lines.append('')
    lines.append('    set_order_cost(OrderCost(')
    lines.append('        open_tax=0, close_tax=0.001,')
    lines.append('        open_commission=0.0001, close_commission=0.0001,')
    lines.append('        close_today_commission=0, min_commission=0,')
    lines.append("    ), type='stock')")
    lines.append('    set_slippage(FixedSlippage(0.003))')
    lines.append('')
    lines.append("    run_weekly(main, weekday=4, time='14:50')")
    lines.append('')
    lines.append('    g.stock_num     = STOCK_NUM')
    lines.append('    g.max_turnover  = MAX_TURNOVER')
    lines.append('    g.mcap_min      = MCAP_MIN')
    lines.append('    g.exclude_prefix = EXCLUDE_PREFIX')
    lines.append('    g.debug         = DEBUG')
    lines.append('')
    lines.append('')
    lines.append('# ==================== 股票池 ====================')
    lines.append('def get_stock_pool(context):')
    lines.append('    date = context.previous_date')
    lines.append('    try:')
    lines.append("        all_stocks = get_index_stocks('000985.XSHG', date=date)")
    lines.append('    except Exception:')
    lines.append("        all_stocks = list(get_all_securities(types=['stock'], date=date).index)")
    lines.append('')
    lines.append("    stocks = [s for s in all_stocks if not s.startswith(g.exclude_prefix)]")
    lines.append('    if len(stocks) == 0:')
    lines.append('        return []')
    lines.append('')
    lines.append('    q = query(valuation.code, valuation.market_cap).filter(')
    lines.append('        valuation.code.in_(stocks), valuation.market_cap >= g.mcap_min)')
    lines.append('    df = get_fundamentals(q, date=context.previous_date)')
    lines.append("    stocks = list(df['code']) if not df.empty else []")
    lines.append('    if len(stocks) == 0:')
    lines.append('        return []')
    lines.append('')
    lines.append('    cd = get_current_data()')
    lines.append('    final = []')
    lines.append('    for s in stocks:')
    lines.append('        try:')
    lines.append('            if cd[s].paused:')
    lines.append('                continue')
    lines.append('            if cd[s].last_price >= cd[s].high_limit * 0.995:')
    lines.append('                continue')
    lines.append('        except Exception:')
    lines.append('            pass')
    lines.append('        final.append(s)')
    lines.append('    return final')
    lines.append('')
    lines.append('')
    lines.append('# ==================== 价格加载 ====================')
    lines.append('def _get_price_field(stocks, end_dt, count, field):')
    lines.append('    px = get_price(stocks, end_date=end_dt, count=count,')
    lines.append("                   frequency='daily', fields=field,")
    lines.append("                   skip_paused=False, fq='pre')")
    lines.append("    _Panel = getattr(pd, 'Panel', None)")
    lines.append('    if _Panel is not None and isinstance(px, _Panel):')
    lines.append("        px = px[field] if field in getattr(px, 'items', []) else px.minor_xs(field)")
    lines.append('    return px')
    lines.append('')
    lines.append('')

    # 基本面快照
    fund_needed = {f for f in fundamental_factors if f in cache}
    if fund_needed:
        lines.append('# ==================== 基本面快照 ====================')
        lines.append('def load_fundamental_map(context, stocks, factor_name):')
        lines.append('    """按因子名加载对应基本面数据."""')
        lines.append('    # 映射: factor → (table, field)')

        # Build mappings from config
        fundamental_map = {}
        for fn in sorted(fund_needed):
            if fn == 'bp':
                fundamental_map[fn] = ('valuation.pb_ratio', 'bp', lambda v: 1.0/v if v and v>0 else np.nan)
            elif fn == 'roe':
                fundamental_map[fn] = ('indicator.roe', 'roe', lambda v: v)
            elif fn == 'roa':
                fundamental_map[fn] = ('indicator.roa', 'roa', lambda v: v)
            elif fn == 'roic':
                fundamental_map[fn] = ('indicator.roic', 'roic', lambda v: v)
            elif fn == 'net_margin':
                fundamental_map[fn] = ('indicator.net_profit_margin', 'net_margin', lambda v: v)
            elif fn == 'gross_margin':
                fundamental_map[fn] = ('indicator.gross_profit_margin', 'gross_margin', lambda v: v)
            elif fn == 'accruals':
                fundamental_map[fn] = ('indicator.accruals', 'accruals', lambda v: v)
            elif fn == 'f_score':
                fundamental_map[fn] = ('indicator.f_score', 'f_score', lambda v: v)
            elif fn == 'earnings_stability':
                fundamental_map[fn] = ('indicator.earnings_stability', 'earnings_stability', lambda v: v)
            elif fn == 'earnings_quality_proxy':
                fundamental_map[fn] = ('indicator.earnings_quality', 'earnings_quality_proxy', lambda v: v)
            elif fn == 'rev_growth_yoy':
                fundamental_map[fn] = ('indicator.revenue_growth_rate', 'rev_growth_yoy', lambda v: v)
            elif fn == 'earnings_growth_yoy':
                fundamental_map[fn] = ('indicator.earnings_growth', 'earnings_growth_yoy', lambda v: v)
            elif fn == 'operational_efficiency_proxy':
                fundamental_map[fn] = ('indicator.operational_efficiency', 'operational_efficiency_proxy', lambda v: v)
            elif fn == 'ocf_quality':
                fundamental_map[fn] = ('indicator.ocf_quality', 'ocf_quality', lambda v: v)
            elif fn == 'debt_coverage':
                fundamental_map[fn] = ('indicator.debt_coverage', 'debt_coverage', lambda v: v)
            elif fn == 'asset_turnover':
                fundamental_map[fn] = ('indicator.asset_turnover', 'asset_turnover', lambda v: v)
            elif fn == 'earnings_consistency_proxy':
                fundamental_map[fn] = ('indicator.earnings_consistency', 'earnings_consistency_proxy', lambda v: v)
            elif fn == 'cashflow_matching_proxy':
                fundamental_map[fn] = ('indicator.cashflow_matching', 'cashflow_matching_proxy', lambda v: v)
            elif fn == 'ln_mcap' or fn == 'ln_circulating_mcap':
                fundamental_map[fn] = ('valuation.market_cap', fn, lambda v: np.log(v) if v and v>0 else np.nan)
            elif fn == 'sp':
                fundamental_map[fn] = ('valuation.ps_ratio', 'sp', lambda v: 1.0/v if v and v>0 else np.nan)
            elif fn == 'ep':
                fundamental_map[fn] = ('valuation.pe_ratio', 'ep', lambda v: 1.0/v if v and v>0 else np.nan)
            elif fn == 'cfp':
                fundamental_map[fn] = ('valuation.pcf_ratio', 'cfp', lambda v: 1.0/v if v and v>0 else np.nan)
            elif fn == 'dp':
                fundamental_map[fn] = ('valuation.dividend_yield', 'dp', lambda v: v)
            else:
                fundamental_map[fn] = ('valuation.pb_ratio', fn, lambda v: v)

        lines.append('    maps = {')
        for fn, (table_field, _, _) in fundamental_map.items():
            lines.append(f"        '{fn}': '{table_field}',")
        lines.append('    }')
        lines.append('    if factor_name not in maps:')
        lines.append('        return {}')
        lines.append("    table_col = maps[factor_name]")
        lines.append("    table_name, col_name = table_col.split('.')")
        lines.append('')
        lines.append('    if table_name == "valuation":')
        lines.append('        q = query(valuation.code,')
        lines.append(f'            getattr(valuation, col_name)).filter(')
        lines.append('            valuation.code.in_(stocks))')
        lines.append('        df = get_fundamentals(q, date=context.previous_date)')
        lines.append('    else:')
        lines.append('        try:')
        lines.append('            q = query(indicator.code,')
        lines.append(f'            getattr(indicator, col_name)).filter(')
        lines.append('            indicator.code.in_(stocks))')
        lines.append('            df = get_fundamentals(q, date=context.previous_date)')
        lines.append('        except Exception:')
        lines.append('            return {}')
        lines.append('')
        lines.append('    if df is None or df.empty:')
        lines.append('        return {}')
        lines.append('    result = {}')
        lines.append('    for _, row in df.iterrows():')
        lines.append('        code = row["code"]')
        lines.append(f'        val = row.get(col_name)')
        lines.append(f'        if val is not None and not (isinstance(val, float) and np.isnan(val)):')
        lines.append('            result[code] = float(val)')
        lines.append('    return result')
        lines.append('')

    # OHLCV 因子计算
    lines.append('')
    lines.append('# ==================== OHLCV 因子计算 (分块) ====================')
    lines.append('def compute_base_factors_chunk(open_df, high_df, low_df, close_df, volume_df,')
    lines.append('                                stocks, fundamental_data):')
    lines.append(f'    """计算 {len(ohlcv_factors)} 个 OHLCV 因子."""')
    lines.append('    result = {}')
    lines.append('')

    # Generate factor calculations
    for fn in ohlcv_factors:
        if fn == 'ret_1m':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = close_df / close_df.shift(20) - 1.0')
            lines.append('')
        elif fn == 'ret_3m':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = close_df / close_df.shift(60) - 1.0')
            lines.append('')
        elif fn == 'ret_6m':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = close_df / close_df.shift(120) - 1.0')
            lines.append('')
        elif fn == 'max_ret_1m':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = daily_ret.rolling(20, min_periods=10).max()')
            lines.append('')
        elif fn == 'min_ret_1m':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = daily_ret.rolling(20, min_periods=10).min()')
            lines.append('')
        elif fn == 'vol_1m':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = -(daily_ret.rolling(20, min_periods=10).std())')
            lines.append('')
        elif fn == 'vol_3m':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = -(daily_ret.rolling(60, min_periods=10).std())')
            lines.append('')
        elif fn == 'high_low_range':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = -(high_df - low_df) / close_df')
            lines.append('')
        elif fn == 'boll_pct_b':
            lines.append(f'    # {fn}')
            lines.append(f'    ma20 = close_df.rolling(20, min_periods=10).mean()')
            lines.append(f'    std20 = close_df.rolling(20, min_periods=10).std()')
            lines.append(f'    result[\'{fn}\'] = (close_df - (ma20 - 2*std20)) / (4*std20 + 1e-9)')
            lines.append('')
        elif fn == 'bias_20':
            lines.append(f'    # {fn}')
            lines.append(f'    ma20 = close_df.rolling(20, min_periods=10).mean()')
            lines.append(f'    result[\'{fn}\'] = (close_df - ma20) / (ma20 + 1e-9)')
            lines.append('')
        elif fn == 'roc_20':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = close_df / close_df.shift(20) - 1.0')
            lines.append('')
        elif fn == 'cci_14':
            lines.append(f'    # {fn}')
            lines.append(f'    tp = (high_df + low_df + close_df) / 3.0')
            lines.append(f'    ma14 = tp.rolling(14, min_periods=7).mean()')
            lines.append(f'    md14 = tp.rolling(14, min_periods=7).apply(lambda x: np.abs(x - x.mean()).mean())')
            lines.append(f'    result[\'{fn}\'] = (tp - ma14) / (0.015 * md14 + 1e-9)')
            lines.append('')
        elif fn == 'rank_1m':
            lines.append(f'    # {fn}: rank of ret_1m')
            lines.append(f'    ret_1m_val = close_df / close_df.shift(20) - 1.0')
            lines.append(f'    result[\'{fn}\'] = ret_1m_val')
            lines.append('')
        elif fn == 'skewness_20':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = -daily_ret.rolling(20, min_periods=10).skew()')
            lines.append('')
        elif fn == 'price_1m':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = close_df / close_df.rolling(20, min_periods=10).max() - 1.0')
            lines.append('')
        elif fn == 'amihud_illiq':
            lines.append(f'    # {fn}: -|ret|/volume')
            lines.append(f'    daily_ret = close_df.pct_change().abs()')
            lines.append(f'    result[\'{fn}\'] = -(daily_ret / (volume_df * close_df + 1e-9)).rolling(20, min_periods=10).mean()')
            lines.append('')
        elif fn == 'dollar_vol_20d':
            lines.append(f'    # {fn}')
            lines.append(f'    dollar_vol = close_df * volume_df')
            lines.append(f'    result[\'{fn}\'] = -np.log(dollar_vol.rolling(20, min_periods=10).mean().replace(0, np.nan))')
            lines.append('')
        elif fn == 'turnover_change':
            lines.append(f'    # {fn}: volume MA5/MA20 ratio')
            lines.append(f'    ma5 = volume_df.rolling(5, min_periods=3).mean()')
            lines.append(f'    ma20 = volume_df.rolling(20, min_periods=10).mean()')
            lines.append(f'    result[\'{fn}\'] = ma5 / (ma20 + 1e-9)')
            lines.append('')
        elif fn == 'turnover_std':
            lines.append(f'    # {fn}: volume CV proxy')
            lines.append(f'    vs = volume_df.rolling(20, min_periods=10).std()')
            lines.append(f'    vm = volume_df.rolling(20, min_periods=10).mean()')
            lines.append(f'    result[\'{fn}\'] = -(vs / (vm + 1e-9))')
            lines.append('')
        elif fn == 'dollar_vol_stability':
            lines.append(f'    # {fn}')
            lines.append(f'    dv = close_df * volume_df')
            lines.append(f'    result[\'{fn}\'] = -(dv.rolling(20, min_periods=10).std() / (dv.rolling(20, min_periods=10).mean() + 1e-9))')
            lines.append('')
        elif fn == 'turnover_cv_20d':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = -(volume_df.rolling(20, min_periods=10).std() / (volume_df.rolling(20, min_periods=10).mean() + 1e-9))')
            lines.append('')
        elif fn == 'turnover_to_vol':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    vret_std = daily_ret.rolling(20, min_periods=10).std()')
            lines.append(f'    result[\'{fn}\'] = -(volume_df.rolling(20, min_periods=10).mean() / (vret_std * close_df + 1e-9))')
            lines.append('')
        elif fn == 'vr_26':
            lines.append(f'    # {fn}: volume ratio 26')
            lines.append(f'    result[\'{fn}\'] = volume_df.rolling(5, min_periods=3).mean() / (volume_df.rolling(26, min_periods=10).mean() + 1e-9)')
            lines.append('')
        elif fn == 'vroc_12':
            lines.append(f'    # {fn}: volume ROC 12')
            lines.append(f'    result[\'{fn}\'] = volume_df / volume_df.shift(12) - 1.0')
            lines.append('')
        elif fn == 'psy_12':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = -(daily_ret.rolling(12, min_periods=6).apply(lambda x: (x>0).mean()))')
            lines.append('')
        elif fn == 'money_flow_20':
            lines.append(f'    # {fn}')
            lines.append(f'    tp = (high_df + low_df + close_df) / 3.0')
            lines.append(f'    mf = tp * volume_df')
            lines.append(f'    mf_ma = mf.rolling(20, min_periods=10).mean()')
            lines.append(f'    result[\'{fn}\'] = -(mf / (mf_ma + 1e-9) - 1.0)')
            lines.append('')
        elif fn == 'davol_20':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    pos_ret = daily_ret.clip(lower=0)')
            lines.append(f'    result[\'{fn}\'] = (pos_ret * volume_df).rolling(20, min_periods=10).sum() / (volume_df.rolling(20, min_periods=10).sum() + 1e-9)')
            lines.append('')
        elif fn == 'bull_power':
            lines.append(f'    # {fn}')
            lines.append(f'    ema13 = close_df.ewm(span=13, min_periods=7).mean()')
            lines.append(f'    result[\'{fn}\'] = (high_df - ema13) / (close_df + 1e-9)')
            lines.append('')
        elif fn == 'bear_power':
            lines.append(f'    # {fn}')
            lines.append(f'    ema13 = close_df.ewm(span=13, min_periods=7).mean()')
            lines.append(f'    result[\'{fn}\'] = (low_df - ema13) / (close_df + 1e-9)')
            lines.append('')
        elif fn == 'gap_up':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = open_df / close_df.shift(1) - 1.0')
            lines.append('')
        elif fn == 'opening_gap_momentum':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = (open_df / close_df.shift(1) - 1.0).rolling(5, min_periods=3).mean()')
            lines.append('')
        elif fn == 'intraday_reversal':
            lines.append(f'    # {fn}: -(close-open)/(high-low+eps)')
            lines.append(f'    result[\'{fn}\'] = -(close_df - open_df) / (high_df - low_df + 0.001)')
            lines.append('')
        elif fn == 'ret_open_2d_proxy':
            lines.append(f'    # {fn}')
            lines.append(f'    ro = open_df / close_df.shift(1) - 1.0')
            lines.append(f'    result[\'{fn}\'] = ro.iloc[-2:].mean()')
            lines.append('')
        elif fn == 'relative_spread_proxy':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = -(high_df - low_df) / (close_df + 0.001)')
            lines.append('')
        elif fn == 'tvma_20':
            lines.append(f'    # {fn}')
            lines.append(f'    money = close_df * volume_df')
            lines.append(f'    ma20 = money.rolling(20, min_periods=10).mean()')
            lines.append(f'    result[\'{fn}\'] = (-(money / (ma20 + 1e-9))).clip(0, 10)')
            lines.append('')
        elif fn == 'atr_14':
            lines.append(f'    # {fn}')
            lines.append(f'    h_l = high_df - low_df')
            lines.append(f'    h_pc = (high_df - close_df.shift(1)).abs()')
            lines.append(f'    l_pc = (low_df - close_df.shift(1)).abs()')
            lines.append(f'    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1) if len(high_df.shape)==2 else np.maximum(np.maximum(h_l, h_pc), l_pc)')
            lines.append(f'    result[\'{fn}\'] = -tr.rolling(14, min_periods=7).mean() / (close_df + 1e-9)')
            lines.append('')
        elif fn == 'abnormal_turnover':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = volume_df / volume_df.rolling(60, min_periods=20).mean()')
            lines.append('')
        elif fn == 'volume_ratio':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = volume_df / volume_df.shift(1)')
            lines.append('')
        elif fn == 'volume_stability':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = -(volume_df.rolling(20, min_periods=10).std() / (volume_df.rolling(20, min_periods=10).mean() + 1e-9))')
            lines.append('')
        elif fn == 'attention_decay':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = -(volume_df / volume_df.shift(5) - 1.0)')
            lines.append('')
        elif fn == 'post_earnings_stability':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = -(daily_ret.rolling(10, min_periods=5).std())')
            lines.append('')
        elif fn == 'earnings_volume_drift':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = daily_ret.rolling(5, min_periods=3).mean() * volume_df.rolling(5, min_periods=3).mean()')
            lines.append('')
        elif fn == 'earnings_season_vol_div':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = -(daily_ret.rolling(20, min_periods=10).std() / (daily_ret.rolling(60, min_periods=30).std() + 1e-9))')
            lines.append('')
        elif fn == 'rsi_14':
            lines.append(f'    # {fn}')
            lines.append(f'    delta = close_df.diff()')
            lines.append(f'    up = delta.clip(lower=0).rolling(14, min_periods=7).mean()')
            lines.append(f'    down = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()')
            lines.append(f'    result[\'{fn}\'] = up / (up + down + 1e-9)')
            lines.append('')
        elif fn == 'short_rev_5d':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = -(close_df / close_df.shift(5) - 1.0)')
            lines.append('')
        elif fn == 'streak':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    signs = np.sign(daily_ret)')
            lines.append(f'    streak = signs.copy()')
            lines.append(f'    for i in range(1, 5):')
            lines.append(f'        streak = streak + signs.shift(i).fillna(0)')
            lines.append(f'    result[\'{fn}\'] = streak')
            lines.append('')
        elif fn == 'high_52w_rank':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = close_df / close_df.rolling(250, min_periods=50).max()')
            lines.append('')
        elif fn == 'high_52w_dist':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = -(close_df.rolling(250, min_periods=50).max() / close_df - 1.0)')
            lines.append('')
        elif fn == 'kurtosis_20':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = -daily_ret.rolling(20, min_periods=10).kurt()')
            lines.append('')
        elif fn == 'sharpe_20':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = daily_ret.rolling(20, min_periods=10).mean() / (daily_ret.rolling(20, min_periods=10).std() + 1e-9)')
            lines.append('')
        elif fn == 'downside_vol':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    neg_ret = daily_ret.clip(upper=0)')
            lines.append(f'    result[\'{fn}\'] = -(neg_ret.rolling(20, min_periods=10).std())')
            lines.append('')
        elif fn in ('ret_1m_skip1m', 'ret_12m'):
            lines.append(f'    # {fn}: 占位')
            lines.append(f'    result[\'{fn}\'] = close_df / close_df.shift(60) - 1.0')
            lines.append('')
        elif fn == 'vpt':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = ((close_df / close_df.shift(1) - 1) * volume_df).rolling(20, min_periods=10).sum()')
            lines.append('')
        elif fn == 'volume_climax_reversal':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    vol_z = (volume_df - volume_df.rolling(20, min_periods=10).mean()) / (volume_df.rolling(20, min_periods=10).std() + 1e-9)')
            lines.append(f'    result[\'{fn}\'] = -daily_ret * vol_z')
            lines.append('')
        elif fn == 'panic_selling':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = -(daily_ret.clip(upper=0) * volume_df).rolling(10, min_periods=5).sum()')
            lines.append('')
        elif fn == 'skew_1m':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = -daily_ret.rolling(20, min_periods=10).skew()')
            lines.append('')
        elif fn == 'trend_persistence_score':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = daily_ret.rolling(20, min_periods=10).apply(lambda x: (x>0).mean())')
            lines.append('')
        elif fn == 'trend_smoothness':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = -(close_df.rolling(20, min_periods=10).std() / (close_df.rolling(20, min_periods=10).mean() + 1e-9))')
            lines.append('')
        elif fn == 'range_consistency':
            lines.append(f'    # {fn}')
            lines.append(f'    hl = high_df - low_df')
            lines.append(f'    result[\'{fn}\'] = -(hl.rolling(20, min_periods=10).std() / (hl.rolling(20, min_periods=10).mean() + 1e-9))')
            lines.append('')
        elif fn == 'volatility_of_volatility':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    vol = daily_ret.rolling(10, min_periods=5).std()')
            lines.append(f'    result[\'{fn}\'] = vol.rolling(20, min_periods=10).std()')
            lines.append('')
        elif fn == 'plrc_12':
            lines.append(f'    # {fn}')
            lines.append(f'    daily_ret = close_df.pct_change()')
            lines.append(f'    result[\'{fn}\'] = daily_ret.rolling(12, min_periods=6).apply(lambda x: ((x>0).sum() - (x<0).sum()) / 12.0)')
            lines.append('')
        elif fn == 'vm_diff':
            lines.append(f'    # {fn}')
            lines.append(f'    result[\'{fn}\'] = (volume_df - volume_df.shift(1)) / (volume_df.shift(1) + 1e-9)')
            lines.append('')
        elif fn == 'overnight_5d':
            lines.append(f'    # {fn}')
            lines.append(f'    ovn = open_df / close_df.shift(1) - 1.0')
            lines.append(f'    result[\'{fn}\'] = ovn.rolling(5, min_periods=3).sum()')
            lines.append('')
        else:
            # Generic fallback
            lines.append(f'    # {fn}: 通用动量/波动代理')
            lines.append(f'    result[\'{fn}\'] = close_df.pct_change(20)')
            lines.append('')

    lines.append('    return result')
    lines.append('')

    # 主逻辑
    lines.append('')
    lines.append('# ==================== 截面 rank % ====================')
    lines.append('def _rank_pct(series):')
    lines.append('    s = series.dropna()')
    lines.append('    if len(s) == 0:')
    lines.append('        return series')
    lines.append('    return s.rank(pct=True)')
    lines.append('')
    lines.append('')
    lines.append('# ==================== 复合因子构建 ====================')
    lines.append('def build_composites(base_df):')
    lines.append(f'    """{len(selected_comps)} 复合 (rank_pct 乘积)."""')
    lines.append('')

    # Rank each needed base factor
    all_ranked = set()
    for c in selected_comps:
        fa, fb = c['pair']
        if fa == 'value_momentum':
            # Nested: rank(rank(bp) × rank(-ret_3m))
            fa = 'bp'
            fb = 'ret_3m'
        elif fb == 'value_momentum':
            fb = 'bp'
            fa = 'ret_3m'
        all_ranked.add(fa)
        all_ranked.add(fb)

    lines.append('    # Rank all base factors')
    for f in sorted(all_ranked):
        if f == 'rank_1m':
            # rank_1m is rank of ret_1m
            lines.append(f"    r_rank_1m = _rank_pct(base_df['ret_1m'])")
        elif f == 'ret_3m':
            lines.append(f"    r_ret_3m = _rank_pct(-base_df['ret_3m'])  # 反转")
        else:
            lines.append(f"    r_{f} = _rank_pct(base_df['{f}'])")

    lines.append('')
    lines.append('    return {')
    for c in selected_comps:
        fa, fb = c['pair']
        name = c['name']
        if fa == 'value_momentum':
            lines.append(f"        '{name}': _rank_pct(r_bp * r_ret_3m) * r_{fb},")
        elif fb == 'value_momentum':
            lines.append(f"        '{name}': r_{fa} * _rank_pct(r_bp * r_ret_3m),")
        else:
            if fa == 'rank_1m':
                rfa = 'r_rank_1m'
            else:
                rfa = f'r_{fa}'
            if fb == 'rank_1m':
                rfb = 'r_rank_1m'
            else:
                rfb = f'r_{fb}'
            lines.append(f"        '{name}': {rfa} * {rfb},")
    lines.append('    }')
    lines.append('')

    # Main function
    lines.append('')
    lines.append('# ==================== 主逻辑 ====================')
    lines.append('def main(context):')
    lines.append("    date_str = context.current_dt.strftime('%Y-%m-%d')")
    lines.append('    stocks = get_stock_pool(context)')
    lines.append('    if g.debug:')
    lines.append("        log.info('[%s] pool=%d' % (date_str, len(stocks)))")
    lines.append('    if len(stocks) < g.stock_num:')
    lines.append('        return')
    lines.append('')
    lines.append('    end_dt = context.previous_date')
    lines.append('')

    # Base factor list
    base_list = sorted(all_ranked - {'value_momentum'})
    lines.append(f'    base_factors = {base_list}')
    lines.append('    base_data = {f: {} for f in base_factors}')
    lines.append('')

    # Load fundamental data if needed
    if fund_needed:
        lines.append('    # 加载基本面数据')
        for fn in sorted(fund_needed):
            lines.append(f"    base_data['{fn}'] = load_fundamental_map(context, stocks, '{fn}')")

    lines.append('')
    lines.append('    # 分块加载 OHLCV')
    lines.append('    for i in range(0, len(stocks), CHUNK):')
    lines.append('        chunk = stocks[i:i + CHUNK]')
    lines.append("        open_df   = _get_price_field(chunk, end_dt, LOOKBACK, 'open')")
    lines.append("        high_df   = _get_price_field(chunk, end_dt, LOOKBACK, 'high')")
    lines.append("        low_df    = _get_price_field(chunk, end_dt, LOOKBACK, 'low')")
    lines.append("        close_df  = _get_price_field(chunk, end_dt, LOOKBACK, 'close')")
    lines.append("        volume_df = _get_price_field(chunk, end_dt, LOOKBACK, 'volume')")
    lines.append('')
    lines.append('        res = compute_base_factors_chunk(open_df, high_df, low_df, close_df,')
    lines.append('                                          volume_df, chunk, base_data)')
    lines.append('        for f in base_factors:')
    lines.append('            if f in res:')
    lines.append('                base_data[f].update(res[f].to_dict())')
    lines.append('')
    lines.append('        del open_df, high_df, low_df, close_df, volume_df, res')
    lines.append('        gc.collect()')
    lines.append('')
    lines.append('    base_df = pd.DataFrame(base_data)')
    lines.append('    comp = build_composites(base_df)')
    lines.append('')

    # Composite combination
    lines.append('    # ICIR 加权合成')
    terms = []
    for c in selected_comps:
        terms.append(f"\n        COMPOSITE_WEIGHTS['{c['name']}'] * comp['{c['name']}']")
    lines.append('    total = (' + ' +'.join(terms))
    lines.append('    )')
    lines.append("    total = total.dropna().sort_values(ascending=False)")
    lines.append('    if len(total) == 0:')
    lines.append('        return')
    lines.append('')
    lines.append('    target = total.head(g.stock_num).index.tolist()')
    lines.append('    if g.debug:')
    lines.append("        log.info('[%s] top3 scores: %s' % (date_str,")
    lines.append("                 ', '.join('%.4f' % v for v in total.head(3).values)))")
    lines.append('')
    lines.append('    rebalance(context, target)')
    lines.append('')

    # Rebalance function
    lines.append('')
    lines.append('# ==================== 调仓 ====================')
    lines.append('def rebalance(context, target):')
    lines.append('    target_set = set(target)')
    lines.append('    positions = context.portfolio.positions')
    lines.append('    current = set(positions.keys())')
    lines.append('')
    lines.append('    total_value = context.portfolio.total_value')
    lines.append('    unit = total_value / g.stock_num')
    lines.append('    turnover = 0.0')
    lines.append('    for s in current - target_set:')
    lines.append('        turnover += abs(positions[s].value)')
    lines.append('    for s in target:')
    lines.append('        cur = positions[s].value if s in positions else 0.0')
    lines.append('        turnover += abs(unit - cur)')
    lines.append('    turnover = turnover / total_value if total_value > 0 else 0.0')
    lines.append('')
    lines.append('    if turnover > g.max_turnover and g.debug:')
    lines.append("        log.info('[rebalance] turnover=%.2f > cap=%.2f' % (turnover, g.max_turnover))")
    lines.append('')
    lines.append('    for s in current - target_set:')
    lines.append('        order_target_value(s, 0)')
    lines.append('')
    lines.append('    cd = get_current_data()')
    lines.append('    for s in target:')
    lines.append('        try:')
    lines.append('            if cd[s].paused:')
    lines.append('                continue')
    lines.append('            if cd[s].last_price >= cd[s].high_limit * 0.995:')
    lines.append('                continue')
    lines.append('        except Exception:')
    lines.append('            pass')
    lines.append('        order_target_value(s, unit)')

    return '\n'.join(lines)


# ==================== 主流程 ====================

def main():
    start = time.time()

    # 1. 加载数据
    cache = load_cache()
    forward_returns, weeks, stocks = build_forward_returns(cache)

    # 2. 加载 XGBoost 交互对
    with open(XGB_PAIRS_FILE, 'r', encoding='utf-8') as f:
        xgb_data = json.load(f)
    xgb_pairs = xgb_data['top_interactions']
    print(f'    加载 {len(xgb_pairs)} 个 XGBoost 交互对')

    # 3. 计算 XGBoost 复合 ICIR
    xgb_results = compute_xgb_composites(cache, xgb_pairs, forward_returns, weeks)

    # 4. 计算 V3 baseline
    v3_results = compute_v3_baseline(cache, forward_returns)

    # 5. 评分与选择
    print(f'\n[4] 三层评分与筛选...')
    selected, scored_df = score_and_select(xgb_results, TOP_K)

    # 6. 输出结果
    print(f'\n{"="*70}')
    print(f'=== XGBoost 对 ICIR 验证结果 ===')
    print(f'{":-"*35}')

    # V3 baseline
    print(f'\n▶ V3 baseline (5 对):')
    for k, v in v3_results.items():
        print(f'  {k:40s} ICIR={v["icir"]:+.4f}  nIC={v["n_ic"]}')

    # XGBoost top N
    print(f'\n▶ XGBoost 选中 Top-{len(selected)} (ICIR≥{ICIR_THRESHOLD}):')
    for i, s in enumerate(selected):
        fa, fb = s['pair']
        print(f'  {i+1}. comp_{fa}_x_{fb}')
        print(f'      ICIR={s["icir"]:+.4f}  stability={s["stability"]:+.3f}  '
              f'persistence={s["persistence"]:.2f}  decay={s["decay"]:+.3f}')
        print(f'      L1={s["score_L1"]:.3f} L2={s["score_L2"]:.3f} '
              f'L3={s["score_L3"]:.3f} TOTAL={s["total_score"]:.3f}')

    # Full ranking of top 30
    print(f'\n▶ XGBoost 对 ICIR 排名 (Top 30):')
    df_sorted = scored_df.sort_values('icir', ascending=False)
    for i, (_, row) in enumerate(df_sorted.head(30).iterrows()):
        fa, fb = row['pair']
        marker = '✓' if row['icir'] >= ICIR_THRESHOLD else ' '
        print(f'  {i+1:2d}. [{marker}] comp_{fa}_x_{fb:25s}  '
              f'ICIR={row["icir"]:+.4f}  total={row["total_score"]:.3f}')

    # V3 vs XGBoost calendar comparison
    if v3_results and selected:
        print(f'\n▶ 校准(V3 ICIR 比值):')
        for k, v in v3_results.items():
            ratio = selected[0]['icir'] / v['icir'] if v['icir'] > 0 else 0
            print(f'  {k}: ICIR_local={v["icir"]:+.4f}  (XGB top ICIR={selected[0]["icir"]:+.4f}, ratio={ratio:.2f}x)')

    # Save results
    result_json = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M'),
        'method': 'xgb_v3_validation',
        'icir_threshold': ICIR_THRESHOLD,
        'n_xgb_pairs': len(xgb_pairs),
        'n_valid_composites': len(xgb_results),
        'n_selected': len(selected),
        'v3_baseline': {k: {'icir': v['icir'], 'n_ic': v['n_ic']} for k, v in v3_results.items()},
        'selected': [{
            'name': s['name'],
            'pair': list(s['pair']),
            'icir': s['icir'],
            'total_score': s['total_score'],
            'stability': s['stability'],
            'persistence': s['persistence'],
            'decay': s['decay'],
        } for s in selected],
        'top_30_icir': [{
            'pair': list(row['pair']),
            'icir': row['icir'],
            'total_score': row['total_score'],
        } for _, row in df_sorted.head(30).iterrows()],
    }
    out_json = OUT_DIR / 'xgb_validation_v1.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n    结果保存: {out_json}')

    # 7. 生成 JQ 代码
    jq_path = generate_jq_code(selected, cache)

    print(f'\n{"="*70}')
    print(f'总耗时: {time.time()-start:.1f}s')
    print(f'JQ 代码: {jq_path}')

    return selected, scored_df, jq_path


if __name__ == '__main__':
    main()
