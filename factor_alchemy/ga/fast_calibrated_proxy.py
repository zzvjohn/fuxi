"""
快速校准代理 (Fast Calibrated Proxy)
=====================================
对每个单因子运行 PortfolioSimulator（JQ 对齐口径），
返回校准后的 portfolio 绩效指标，替代 IC/ICIR 做因子选择排序。

使用方式:
    from ga.fast_calibrated_proxy import batch_evaluate_factors
    
    scores = batch_evaluate_factors(
        factor_names, factor_dfs_std, close_weekly, mcap_weekly,
        regime_weeks, factor_icir,
    )
    # scores[factor_name] = {'sharpe': ..., 'annual_return': ..., ...}

口径对齐:
  - 市值过滤: circ_mv >= 5e4 万元 (≈5亿流通市值, 对齐 JQGenerator.mcap_min=5)
  - 成本: QMT 双边 (commission 万1 + stamp_tax 千1 + slippage 0.3%)
  - 换手限制: max_turnover=0.60
  - Top30 等权 (aligns with JQGenerator defaults)
  - 自动方向检测: ICIR<0 的因子自动翻转 (做空变做多)
"""

import numpy as np
import pandas as pd
import sys, time
from pathlib import Path

FA_DIR = Path(__file__).resolve().parent.parent
if str(FA_DIR) not in sys.path:
    sys.path.insert(0, str(FA_DIR))

from portfolio.simulator import PortfolioSimulator

# ── 口径参数 (对齐 JQGenerator + v3 PortfolioSimulator) ──
TOP_N = 30
MAX_TURNOVER = 0.60
MCAP_MIN = 5e4          # 5亿 万元 (流通市值, 对齐 JQGenerator.mcap_min=5)
COMMISSION = 0.0001     # 万1
STAMP_TAX = 0.001       # 千1 (仅卖)
SLIPPAGE = 0.003        # 元/股 (v3 per-share, JQ FixedSlippage(0.003))
SLIPPAGE_MODE = 'per_share'  # v3: per-share model
AUTO_FLIP_ICIR_THRESHOLD = 0.10  # |ICIR|>0.10 且为负则翻转因子方向


def evaluate_single_factor(factor_df, close_weekly, mcap_weekly, 
                           factor_icir=None, top_n=TOP_N, max_turnover=MAX_TURNOVER,
                           mcap_min=MCAP_MIN):
    """
    对单个因子运行 PortfolioSimulator，返回校准后绩效指标。
    
    Parameters
    ----------
    factor_df : pd.DataFrame
        单因子值 (index=weeks, columns=stocks), 已标准化。
    close_weekly : pd.DataFrame
        周频收盘价 (index=weeks, columns=stocks), hfq。
    mcap_weekly : pd.DataFrame or None
        周频流通市值 (index=weeks, columns=stocks), 万元。
    factor_icir : float or None
        因子 ICIR (用于方向检测)。若 < -threshold 则自动翻转因子。
    
    Returns
    -------
    dict : {'sharpe', 'annual_return', 'max_dd', 'avg_turnover', 
            'total_return', 'n_weeks', 'flipped', 'error'}
    """
    if factor_icir is not None and factor_icir < -AUTO_FLIP_ICIR_THRESHOLD:
        factor_df = -factor_df
        flipped = True
    else:
        flipped = False
    
    # 对齐索引
    common_idx = factor_df.index.intersection(close_weekly.index)
    if len(common_idx) < 12:
        return {'error': f'too few weeks ({len(common_idx)}<12)', 'sharpe': 0, 'annual_return': 0}
    
    fdf = factor_df.loc[common_idx]
    cl = close_weekly.loc[common_idx]
    
    mc = None
    if mcap_weekly is not None:
        mc = mcap_weekly.reindex(index=common_idx, columns=fdf.columns)
    
    try:
        sim = PortfolioSimulator(
            factor_df=fdf, price_df=cl, top_n=top_n,
            max_turnover=max_turnover, 
            commission=COMMISSION, stamp_tax=STAMP_TAX, slippage=SLIPPAGE,
            slippage_mode=SLIPPAGE_MODE,
            mcap_df=mc, mcap_min=mcap_min,
        )
        result = sim.run()
        stats = result.get('stats', {})
        
        if not stats or 'sharpe' not in stats:
            return {'error': 'sim returned empty stats', 'sharpe': 0, 'annual_return': 0}
        
        return {
            'sharpe': float(stats.get('sharpe', 0)),
            'annual_return': float(stats.get('cagr', 0)),
            'max_dd': float(stats.get('max_drawdown', 0)),
            'avg_turnover': float(stats.get('avg_turnover', 0)),
            'total_return': float(stats.get('total_return', 0)),
            'n_weeks': int(stats.get('n_weeks', 0)),
            'flipped': flipped,
            'error': None,
        }
    except Exception as e:
        return {'error': str(e), 'sharpe': 0, 'annual_return': 0}


def batch_evaluate_factors(factor_names, factor_dfs_std, close_weekly, mcap_weekly,
                            regime_weeks, factor_icir_dict, top_n=TOP_N,
                            max_turnover=MAX_TURNOVER, mcap_min=MCAP_MIN,
                            verbose=True):
    """
    批量评估多个因子，返回校准后绩效指标排序。
    
    Parameters
    ----------
    factor_names : list[str]
        待评估的因子名列表。
    factor_dfs_std : dict[str, pd.DataFrame]
        标准化后的因子 DataFrame。
    close_weekly : pd.DataFrame
        周频收盘价。
    mcap_weekly : pd.DataFrame or None
        周频流通市值。
    regime_weeks : list[Timestamp]
        该 regime 的周日期列表。
    factor_icir_dict : dict[str, float]
        因子 ICIR 字典 (用于方向检测)。
    
    Returns
    -------
    dict[str, dict]
        {factor_name: {'sharpe': ..., 'annual_return': ..., ...}}
    """
    if verbose:
        print(f"  [calibrated proxy] 评估 {len(factor_names)} 因子 (mcap≥{mcap_min/1e4:.0f}亿)...")
    
    t_start = time.time()
    results = {}
    
    # 子集化到 regime weeks
    regime_idx = pd.DatetimeIndex(regime_weeks)
    
    for fname in factor_names:
        fdf = factor_dfs_std.get(fname)
        if fdf is None:
            results[fname] = {'error': 'factor not found', 'sharpe': 0, 'annual_return': 0}
            continue
        
        # 子集化
        fdf_r = fdf.reindex(regime_idx)
        if fdf_r.dropna(how='all').empty:
            results[fname] = {'error': 'no data in regime', 'sharpe': 0, 'annual_return': 0}
            continue
        
        icir_val = factor_icir_dict.get(fname, 0)
        
        result = evaluate_single_factor(
            fdf_r, close_weekly, mcap_weekly,
            factor_icir=icir_val, top_n=top_n,
            max_turnover=max_turnover, mcap_min=mcap_min,
        )
        results[fname] = result
    
    if verbose:
        elapsed = time.time() - t_start
        n_success = sum(1 for v in results.values() if v.get('error') is None)
        n_flipped = sum(1 for v in results.values() if v.get('flipped'))
        print(f"  [calibrated proxy] 完成: {n_success}/{len(factor_names)} 有结果"
              f"{f' ({n_flipped} 翻转)' if n_flipped else ''} | 耗时: {elapsed:.1f}s")
    
    return results


def rank_by_calibrated_score(scores, metric='sharpe', top_k=5):
    """
    按校准后指标排序，返回 top-K 因子 + 分数。
    
    Parameters
    ----------
    scores : dict[str, dict]
        batch_evaluate_factors 的输出。
    metric : str
        排序指标 ('sharpe', 'annual_return', 或 'combined')。
    top_k : int
        保留因子数。
    
    Returns
    -------
    list[(str, float)]
        [(factor_name, score), ...] 按 score 降序。
    """
    valid = [(name, info[metric]) for name, info in scores.items()
             if info.get('error') is None and pd.notna(info.get(metric, np.nan))]
    
    if metric == 'combined':
        # combined = 0.5 * normalized_sharpe + 0.5 * normalized_annual_return
        sharpe_vals = [info.get('sharpe', 0) for _, info in valid_scores(scores)]
        ret_vals = [info.get('annual_return', 0) for _, info in valid_scores(scores)]
        # ... simplified: just use sharpe
        valid.sort(key=lambda x: -x[1])
    else:
        valid.sort(key=lambda x: -x[1])
    
    # 取 top-K，但排除 score <= 0 的因子
    positive = [(n, s) for n, s in valid if s > 0]
    if len(positive) >= 3:
        return positive[:top_k]
    
    # 不够则用全部非负（含零）
    non_neg = [(n, s) for n, s in valid if s >= 0]
    if len(non_neg) >= 3:
        return non_neg[:top_k]
    
    # 太少了，取前 top_k 个（含负的）
    return valid[:top_k]


def valid_scores(scores):
    """过滤出有效的评分条目."""
    return [(name, info) for name, info in scores.items()
            if info.get('error') is None]


# ── 便捷入口: 集成到 regime_lasso 的 ensemble selector ──

def calibrated_ensemble_select(factor_names, factor_dfs_std, close_weekly, mcap_weekly,
                                regime_weeks, factor_icir_dict, X, y, top_k=5,
                                boruta_n_estimators=200, boruta_max_depth=7, verbose=True):
    """
    完整校准集成选择流程:
      1) Boruta 确认信号因子 (Signal vs Noise 门控)
      2) 校准 PortfolioSimulator 跑每个确认因子
      3) 按校准 Sharpe 排序 → Top-K
    
    Parameters
    ----------
    factor_names : list[str]
        输入的候选因子名 (已过 ICIR 阈值)。
    factor_dfs_std : dict
    close_weekly : pd.DataFrame
    mcap_weekly : pd.DataFrame or None
    regime_weeks : list[Timestamp]
    factor_icir_dict : dict[str, float]
    X : pd.DataFrame
        拉平的回归矩阵 (用于 Boruta)。
    y : pd.Series
        拉平的前向收益 (用于 Boruta)。
    top_k : int
        保留因子数。
    
    Returns
    -------
    dict: {factor_name: calibrated_score, ...}
    meta: dict (含 boruta_confirmed, calibrated_rank 等)
    """
    from sklearn.ensemble import RandomForestRegressor
    from boruta import BorutaPy
    
    meta = {'method': 'ensemble_calibrated', 'n_input': len(factor_names)}
    n = len(factor_names)
    
    if n < 3:
        return {}, {'error': 'too few factors', 'n_input': n}
    
    # ── 1) Boruta 确认 ──
    if verbose:
        print(f"  [calibrated ensemble] Step 1/2: Boruta 确认 ({n} 因子)...")
    
    # 子采样 (防内存爆炸)
    subsample_cap = 40000
    if len(X) > subsample_cap:
        Xs = X.sample(subsample_cap, random_state=42)
        ys = y.loc[Xs.index]
    else:
        Xs, ys = X, y
    
    rf = RandomForestRegressor(n_estimators=boruta_n_estimators, max_depth=boruta_max_depth,
                               n_jobs=-1, random_state=42)
    boruta = BorutaPy(rf, n_estimators=50, random_state=42, verbose=0)
    boruta.fit(Xs.values, ys.values.ravel())
    
    confirmed = [factor_names[i] for i in range(n) if boruta.support_[i]]
    if len(confirmed) < 3:
        confirmed = [factor_names[i] for i in range(n) if boruta.support_weak_[i]]
    if len(confirmed) < 3:
        confirmed = factor_names[:max(3, min(8, n))]
    
    meta['boruta_confirmed'] = len(confirmed)
    if verbose:
        print(f"  [calibrated ensemble] Boruta 确认: {len(confirmed)}/{n} 因子 → {confirmed}")
    
    # ── 2) 校准 PortfolioSimulator ──
    if verbose:
        print(f"  [calibrated ensemble] Step 2/2: 校准 PortfolioSimulator 评估 {len(confirmed)} 因子...")
    
    scores = batch_evaluate_factors(
        confirmed, factor_dfs_std, close_weekly, mcap_weekly,
        regime_weeks, factor_icir_dict, verbose=verbose,
    )
    
    # ── 3) 排名 ──
    ranked = rank_by_calibrated_score(scores, metric='sharpe', top_k=top_k)
    
    if verbose:
        print(f"  [calibrated ensemble] 校准后排名 (Top-{top_k} by Sharpe):")
        for name, score in ranked:
            info = scores.get(name, {})
            print(f"    {name:30s} Sharpe={score:+.3f} 年化={info.get('annual_return',0):+.1%}"
                  f"  MaxDD={info.get('max_dd',0):+.1%} 换手={info.get('avg_turnover',0):.0%}"
                  f"{' [翻转]' if info.get('flipped') else ''}")
    
    meta['calibrated_rank'] = {name: round(float(score), 4) for name, score in ranked}
    meta['calibrated_details'] = {name: {
        'sharpe': round(scores[name].get('sharpe', 0), 4),
        'annual_return': round(scores[name].get('annual_return', 0), 4),
        'max_dd': round(scores[name].get('max_dd', 0), 4),
        'avg_turnover': round(scores[name].get('avg_turnover', 0), 4),
    } for name, _ in ranked}
    
    # 返回校准分数作为"coefficient"(方向由 ICIR 决定, 与 select_factors 接口一致)
    result = {}
    for name, score in ranked:
        icir_val = factor_icir_dict.get(name, 0)
        sign = 1 if icir_val >= 0 else -1
        result[name] = sign * score  # signed calibrated score
    
    return result, meta
