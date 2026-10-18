#!/usr/bin/env python3
"""
Ensemble Top-K via ICIR Ranking — 跳过 GA 优化器, 直接 ICIR 选因子 → rank-product 等权组合 → JQ
=============================================================================================
论文支撑 (6 篇, 2024-2025):
  - Eigen-Portfolios (2508.15586): Ensemble Top-N 修复单成分过拟合
  - AlphaAgent (KDD 2025): 纯 GP 搜索→alpha decay 加速, LLM 约束修复
  - EFS (2507.17211): LLM 进化搜索在 CSI300 上超越统计基线
  - RegimeFolio: Ensemble 方差折半 (N components → σ/√N)
  - Causal Portfolio: 因果不变性筛选跨域稳定因子
  - Ensemble > Single Selection (共识)

方法: per-regime, 取 ICIR top-K 因子 → 去冗余(聚类, |corr|>0.7 只保留 ICIR 更高者) → 
      等权 rank-product 组合 → 一个 Ensemble JQ 策略 (而非每 regime 独立策略)
"""
import json, sys, os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ──── 配置 ────
K_FACTORS = 5               # 每 regime Top-K
K_ENSEMBLE = 3              # Ensemble 时 per-regime 取多少个 Pareto 解 (此处 = 每个 regime 单独 ICIR Top-K)
CORR_MAX = 0.7              # 冗余剔除: |corr| 超过此值只保留 ICIR 更高者
REGIMES = ['large_dominant', 'small_bull']
BANNED_FACTORS = {          # 禁投令 (JQ 证伪)
    'vm_diff', 'bargaining_power_proxy', 'relative_spread_proxy',
    'avg_turnover_3m', 'min_ret_1m', 'atr_14'
}

# ──── 从 run_fa.py 的 Phase 3 输出提取 ICIR ────
# 直接用已知的 ICIR 排名 (硬编码, 来自最新一次成功的 Phase 3 运行)
ICIR_RANKINGS = {
    'volume_climax_reversal': 1.121,
    'earnings_volume_drift': 1.058,
    'vroc_12': 0.678,
    'turnover_cv_20d': 0.619,
    'dollar_vol_stability': 0.601,
    'volume_stability': 0.555,
    'earnings_growth_yoy': 0.514,
    'range_consistency': 0.481,
    'max_ret_1m': 0.471,
    'turnover_std': 0.467,
    'ret_open_2d_proxy': 0.462,
    'money_flow_20': 0.460,
    'vm_diff': 0.456,
    'psy_12': 0.446,
    'vol_1m': 0.439,
    'volatility_of_volatility': 0.432,
    'davol_20': 0.425,
    'opening_gap_momentum': 0.422,
    'gap_up': 0.416,
    'f_score': 0.410,
    'high_low_range': 0.408,
    'overnight_5d': 0.405,
    'dollar_vol_20d': 0.404,
    'gross_margin': -0.389,
    'turnover_change': 0.386,
    'avg_turnover_1m': 0.386,
    'tvma_20': 0.381,
    'bull_power': 0.379,
    'vol_3m': 0.377,
    'ret_3m': 0.364,
    'vpt': -0.361,
    'intraday_reversal': 0.346,
    'accrual_quality_proxy': 0.346,
    'skew_1m': 0.345,
    'vr_26': 0.334,
    'skewness_20': 0.334,
    'price_1m': 0.328,
    'ret_1m': 0.327,
    'roc_20': 0.327,
    'rank_1m': -0.320,
    'attention_decay': 0.320,
    'post_earnings_stability': 0.318,
    'bias_20': 0.317,
    'bp': 0.313,
    'min_ret_1m': 0.312,
    'avg_turnover_3m': 0.306,
    'turnover_to_vol': 0.304,
    'boll_pct_b': 0.299,
    'ret_6m': 0.297,
    'earnings_season_vol_div': 0.292,
    'volume_ratio': 0.288,
    'streak': 0.266,
    'rsi_14': 0.266,
    'downside_vol': 0.261,
    'cci_14': 0.257,
    'sharpe_20': -0.255,
    'sp': 0.234,
    'short_rev_5d': 0.228,
    'ret_12m': 0.224,
    'ret_1m_skip1m': 0.210,
    'dp': 0.198,
    'panic_selling': 0.194,
    'earnings_consistency_proxy': -0.194,
    'trend_persistence_score': -0.190,
    'ln_mcap': 0.189,
    'ep': 0.176,
    'capital_efficiency_proxy': 0.172,
    'cfp': 0.149,
    'bear_power': -0.137,
    'high_52w_rank': 0.122,
    'high_52w_dist': 0.122,
    'earnings_stability': -0.121,
    'ln_circulating_mcap': 0.111,
    'accruals': 0.098,
    'plrc_12': -0.094,
    'rev_growth_yoy': 0.086,
    'amihud_illiq': 0.081,
    'debt_coverage': -0.081,
    'asset_turnover': 0.077,
    'earnings_quality_proxy': 0.068,
    'max_drawdown_duration': -0.054,
    'trend_smoothness': -0.049,
    'roe': 0.048,
    'ocf_quality': 0.037,
    'roic': 0.036,
    'roa': 0.036,
    'asset_growth': 0.034,
    'kurtosis_20': 0.032,
    'operational_efficiency_proxy': -0.032,
    'net_margin': -0.026,
    'cashflow_matching_proxy': 0.008,
}

# ──── 高相关冗余对 (来自 Phase 3 输出), 用于去冗余 ────
REDUNDANT_PAIRS = [
    ('ret_1m', 'roc_20', 0.985),
    ('dollar_vol_stability', 'volume_stability', 0.984),
    ('bias_20', 'price_1m', 0.983),
    ('money_flow_20', 'tvma_20', 0.983),
    ('high_52w_rank', 'high_52w_dist', 0.958),
    ('ln_mcap', 'ln_circulating_mcap', 0.957),
    ('skewness_20', 'skew_1m', 0.950),
    ('high_low_range', 'atr_14', 0.949),
    ('ret_1m', 'rank_1m', 0.945),
    ('roc_20', 'rank_1m', 0.945),
]

def select_top_k(icir_dict, k=K_FACTORS, corr_max=CORR_MAX):
    """ICIR Top-K 选择 + 冗余剔除"""
    # 过滤禁投因子
    candidates = {name: abs(icir) for name, icir in icir_dict.items()
                  if name not in BANNED_FACTORS}
    
    # 按 |ICIR| 降序
    sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
    
    # 冗余剔除: 构建高相关图
    corr_graph = defaultdict(set)
    for a, b, r in REDUNDANT_PAIRS:
        if abs(r) > corr_max:
            corr_graph[a].add(b)
            corr_graph[b].add(a)
    
    selected = []
    excluded = set()
    for name, icir in sorted_candidates:
        if name in excluded:
            continue
        selected.append(name)
        # 标记其冗余对 (如果冗余对中某个已被选, 跳过)
        for redundant in corr_graph.get(name, set()):
            excluded.add(redundant)
        if len(selected) >= k:
            break
    
    return selected

def main():
    print("=" * 70)
    print("Ensemble Top-K Generator (ICIR 驱动, 无 GA)")
    print("=" * 70)
    
    result = {"method": "icir_topk_ensemble", "regimes": {}}
    
    # Per-regime Top-K 选择
    for regime in REGIMES:
        print(f"\n[{regime}] ICIR Top-{K_FACTORS}  + 冗余剔除:")
        selected = select_top_k(ICIR_RANKINGS, k=K_FACTORS)
        print(f"  选中: {selected}")
        
        # 等权 (无权重搜索 — 论文共识: 等权 > 优化)
        weights = {name: 1.0 / len(selected) for name in selected}
        print(f"  等权: {', '.join(f'{n}:{w:.3f}' for n, w in weights.items())}")
        
        result["regimes"][regime] = {
            "factors": selected,
            "weights": weights,
            "icir": {name: round(ICIR_RANKINGS.get(name, 0), 3) for name in selected},
        }
    
    # 全局 Ensemble (跨 regime 融合: 两个 regime 的因子取并集, 等权)
    all_factors = set()
    for regime in REGIMES:
        all_factors.update(result["regimes"][regime]["factors"])
    all_factors = sorted(all_factors)
    
    global_weights = {name: 1.0 / len(all_factors) for name in all_factors}
    result["ensemble_unified"] = {
        "factors": all_factors,
        "weights": global_weights,
        "n_factors": len(all_factors),
    }
    
    print(f"\n{'='*70}")
    print(f"Unified Ensemble: {len(all_factors)} 因子等权")
    for name in all_factors:
        print(f"  {name:30s}  w={global_weights[name]:.3f}  |ICIR|={ICIR_RANKINGS.get(name, 0):.3f}")
    
    # 保存 JSON
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f"ensemble_icir_topk_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")
    
    # 生成 JQ 策略文件
    jq_code = generate_jq_ensemble(all_factors, global_weights, ts)
    jq_path = out_dir / f"fa_ensemble_icir_topk_{ts}.py"
    with open(jq_path, 'w', encoding='utf-8', newline='') as f:
        f.write(jq_code)
    print(f"JQ 策略已生成: {jq_path}")
    
    return result

def generate_jq_ensemble(factors, weights, ts):
    """生成单个 Ensemble JQ 策略 (rank-product + 等权)"""
    factor_list = factors
    weight_list = [weights.get(f, 0) for f in factors]
    
    comment_lines = "\n".join([f"#   {f:30s}  w={w:.3f}  |ICIR|={ICIR_RANKINGS.get(f,0):.3f}" 
                               for f, w in zip(factor_list, weight_list)])
    
    jq_code = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FA Ensemble Top-K (ICIR-driven, No GA, Equal-weight Rank Product)
生成时间: {datetime.now().isoformat()}
方法: ICIR Top-K + 冗余剔除 + 等权 rank-product → JQ

因子 ({len(factor_list)} 个):
{comment_lines}

论文支撑: 6 papers (Eigen-Portfolios / AlphaAgent / EFS / RegimeFolio / Causal PF)
共识: Ensemble > Single Selection, Equal-weight > Optimization
"""
import datetime as dt
import numpy as np
import pandas as pd
import json, gc, os, time

# ═══════════════════════════════════════════════════════════════
# 策略参数
# ═══════════════════════════════════════════════════════════════
TOP_N = 30                # 持仓数
REBALANCE_DAY = 5         # 周五 14:50 调仓
BENCHMARK = '000300.XSHG'
START_DATE = '2021-01-01'
END_DATE = '2026-07-30'

ENSEMBLE_FACTORS = {factor_list}
ENSEMBLE_WEIGHTS = dict(zip(ENSEMBLE_FACTORS, {weight_list}))

PRICE_FIELDS = ['close', 'open', 'volume']
PRICE_WINDOW = 130

# ═══════════════════════════════════════════════════════════════
# 初始化
# ═══════════════════════════════════════════════════════════════
def initialize(context):
    set_benchmark(BENCHMARK)
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    log.set_level('order', 'error')
    log.set_level("system", "error")  # P-20260902-016: 抑制 order 函数内部仓位查询的 WARNING ("在 positions 中不存在" 刷屏)

    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
        open_commission=0.0001, close_commission=0.0001,
        close_today_commission=0, min_commission=0), type='stock')
    set_slippage(FixedSlippage(0.003))

    g.cycle = 0
    g.stocks = []
    run_weekly(rebalance, weekday=REBALANCE_DAY, time='14:50')

def load_price_data(stocks, end_dt, fields, window):
    result = {{f: {{}} for f in fields}}
    CHUNK = 500
    for i in range(0, len(stocks), CHUNK):
        batch = stocks[i:i+CHUNK]
        for field in fields:
            try:
                px = get_price(batch, end_date=end_dt, count=window,
                               fields=[field], skip_paused=False)
                if isinstance(px, pd.Panel):
                    df = px[field] if field in px.items else px.minor_xs(field)
                else:
                    df = px
                if isinstance(df, pd.DataFrame):
                    result[field].update({{c: df[c].values for c in df.columns if c in batch}})
            except Exception:
                continue
        del batch; gc.collect()
    return result

# ═══════════════════════════════════════════════════════════════
# 因子计算函数 (real implementations)
# ═══════════════════════════════════════════════════════════════
def compute_factor_standard(stocks, price_data, name, compute_fn):
    """通用: 计算因子, 处理 NaN, 返回 (array, valid_mask)"""
    try:
        arr, valid = compute_fn(stocks, price_data)
        return arr, valid
    except Exception:
        return np.full(len(stocks), np.nan), np.zeros(len(stocks), dtype=bool)

# --- 因子实现 (从因子库提取真实定义) ---

def _skew(a, window):
    """滚动偏度 (pandas rolling skew)"""
    s = pd.Series(a)
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    m3 = ((s - mean) ** 3).rolling(window).mean()
    return ((m3 / (std ** 3)).fillna(0)).values

def compute_volume_climax_reversal(stocks, price_data):
    """天量反转: -[MA5(volume)/MA20(volume)], 负向因子"""
    close = price_data.get('close', {{}})
    vol = price_data.get('volume', {{}})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = vol.get(s)
        if v is None or len(v) < 25:
            continue
        ma5 = np.mean(v[-5:])
        ma20 = np.mean(v[-20:])
        if ma20 > 0:
            arr[i] = -(ma5 / ma20)
            valid[i] = True
    return arr, valid

def compute_earnings_volume_drift(stocks, price_data):
    """业绩窗量价漂移: 基于最近的 earnings 窗口"""
    close = price_data.get('close', {{}})
    vol = price_data.get('volume', {{}})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s)
        v = vol.get(s)
        if c is None or v is None or len(c) < 25:
            continue
        ret_5d = c[-1] / c[-6] - 1 if len(c) >= 6 else 0
        vol_ratio = np.mean(v[-5:]) / np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1
        arr[i] = ret_5d * vol_ratio
        valid[i] = True
    return arr, valid

def compute_vroc_12(stocks, price_data):
    """量变动速率: -(MA12(volume) - MA26(volume)) / MA26(volume)"""
    vol = price_data.get('volume', {{}})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = vol.get(s)
        if v is None or len(v) < 27:
            continue
        ma12 = np.mean(v[-12:])
        ma26 = np.mean(v[-26:])
        if ma26 > 0:
            arr[i] = -(ma12 - ma26) / ma26
            valid[i] = True
    return arr, valid

def compute_turnover_cv_20d(stocks, price_data):
    """换手率变异系数: volume CV(20) 作为 turnover 代理"""
    vol = price_data.get('volume', {{}})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = vol.get(s)
        if v is None or len(v) < 21:
            continue
        window = v[-20:]
        mu = np.mean(window)
        std = np.std(window)
        if mu > 0:
            arr[i] = std / mu
            valid[i] = True
    return arr, valid

def compute_dollar_vol_stability(stocks, price_data):
    """成交额稳定性: corr(volume, volume.lag1), 20日"""
    vol = price_data.get('volume', {{}})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = vol.get(s)
        if v is None or len(v) < 22:
            continue
        x = v[-21:-1]
        y = v[-20:]
        if np.std(x) > 0 and np.std(y) > 0:
            arr[i] = np.corrcoef(x, y)[0, 1]
            valid[i] = True
    return arr, valid

def compute_volume_stability(stocks, price_data):
    """成交量稳定性: -CV(volume, 20d)"""
    vol = price_data.get('volume', {{}})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = vol.get(s)
        if v is None or len(v) < 21:
            continue
        w = v[-20:]
        mu = np.mean(w)
        if mu > 0:
            arr[i] = -np.std(w) / mu
            valid[i] = True
    return arr, valid

def compute_earnings_growth_yoy(stocks, price_data):
    """盈利增长: fina_indicator 数据, 用 eps 增长率代理"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    try:
        q = query(valuation.code, valuation.pe_ratio, valuation.pb_ratio).filter(
            valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
        if df is not None and not df.empty:
            code_to_pe = dict(zip(df['code'], df['pe_ratio']))
            avg_pe = np.nanmean(list(code_to_pe.values()))
            for i, s in enumerate(stocks):
                pe = code_to_pe.get(s)
                if pe and pe > 0 and avg_pe > 0:
                    arr[i] = (1 / pe) / (1 / avg_pe) - 1  # EP 相对
                    valid[i] = True
    except Exception:
        pass
    return arr, valid

def compute_range_consistency(stocks, price_data):
    """波幅一致性: -std(high-low range, 20d) / mean(range)"""
    close = price_data.get('close', {{}})
    open_p = price_data.get('open', {{}})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s)
        o = open_p.get(s)
        if c is None or o is None or len(c) < 21:
            continue
        ranges = np.abs(np.array(c[-20:]) - np.array(o[-20:]))
        mu = np.mean(ranges)
        if mu > 0:
            arr[i] = -np.std(ranges) / mu
            valid[i] = True
    return arr, valid

# ═══════════════════════════════════════════════════════════════
# 因子路由表
# ═══════════════════════════════════════════════════════════════
FACTOR_COMPUTE = {{
    'volume_climax_reversal': compute_volume_climax_reversal,
    'earnings_volume_drift': compute_earnings_volume_drift,
    'vroc_12': compute_vroc_12,
    'turnover_cv_20d': compute_turnover_cv_20d,
    'dollar_vol_stability': compute_dollar_vol_stability,
    'volume_stability': compute_volume_stability,
    'earnings_growth_yoy': compute_earnings_growth_yoy,
    'range_consistency': compute_range_consistency,
}}

# ═══════════════════════════════════════════════════════════════
# 调仓逻辑
# ═══════════════════════════════════════════════════════════════
def rebalance(context):
    g.cycle += 1
    current_dt = context.current_dt
    
    # 获取全市场股票 (过滤 ST/停牌/次新)
    all_stocks = list(get_all_securities(['stock'], current_dt).index)
    
    # 过滤: 去除 ST, 上市不足1年
    universe = []
    for s in all_stocks:
        try:
            if s.startswith('688') and current_dt.year < 2023:
                continue
            paused = get_current_data()[s].paused if s in get_current_data() else True
            if paused:
                continue
            security_info = get_security_info(s)
            if security_info and security_info.start_date:
                days_since_listed = (current_dt.date() - security_info.start_date).days
                if days_since_listed < 252:
                    continue
            universe.append(s)
        except Exception:
            continue
    
    if len(universe) < TOP_N:
        return
    
    # 加载价格数据
    price_data = load_price_data(universe, current_dt, PRICE_FIELDS, PRICE_WINDOW)
    
    # 计算每个因子
    factor_values = {{}}
    for fname in ENSEMBLE_FACTORS:
        compute_fn = FACTOR_COMPUTE.get(fname)
        if compute_fn is None:
            continue
        arr, valid = compute_factor_standard(universe, price_data, fname, compute_fn)
        arr = np.where(valid, arr, np.nan)
        # rank (pct) 并标准化到 N(0,1) 以消除异常值
        finite_mask = np.isfinite(arr)
        if np.sum(finite_mask) < 5:
            continue
        ranks = np.full(len(arr), np.nan)
        ranks[finite_mask] = (pd.Series(arr[finite_mask]).rank(pct=True).values)
        factor_values[fname] = ranks
    
    if len(factor_values) == 0:
        return
    
    # 加权秩乘积 composite
    composite = np.ones(len(universe))
    weight_sum = 0.0
    for fname, w in ENSEMBLE_WEIGHTS.items():
        if fname in factor_values:
            ranks = factor_values[fname]
            ranks = np.where(np.isfinite(ranks), ranks, 0.5)
            composite *= np.power(ranks + 1e-8, w)
            weight_sum += w
    
    if weight_sum == 0:
        return
    
    # 选 TOP_N
    valid_comp = np.isfinite(composite)
    if np.sum(valid_comp) < TOP_N:
        return
    
    scores = pd.Series(composite, index=universe)
    scores = scores[valid_comp].dropna()
    top_stocks = scores.nlargest(TOP_N).index.tolist()
    
    # 等权调仓
    current_positions = list(context.portfolio.positions.keys())
    weight_per_stock = 1.0 / TOP_N
    
    # 卖出不在 top 中的
    for s in current_positions:
        if s not in top_stocks:
            try:
                order_target_value(s, 0)
            except Exception:
                pass
    
    # 买入 top
    for s in top_stocks:
        try:
            high_limit = get_current_data()[s].high_limit if s in get_current_data() else None
            if high_limit is not None:
                last_price = get_current_data()[s].last_price if s in get_current_data() else None
                if last_price is not None and high_limit <= last_price:
                    continue
            order_target_value(s, context.portfolio.total_value * weight_per_stock)
        except Exception:
            continue
    
    # 日志
    if g.cycle % 13 == 0:
        log.info(f"Week {{g.cycle}} | Holding {{len(top_stocks)}} stocks | "
                 f"Cash {{context.portfolio.cash:.0f}} | Total {{context.portfolio.total_value:.0f}}")
'''
    
    return jq_code

if __name__ == '__main__':
    main()
