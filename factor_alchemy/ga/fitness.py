"""
GA 适应度函数
==============
评估染色体(因子权重组合)的五维度得分
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from factors.composite import (combine_factors, combine_factors_vectorized,
                                combine_factors_rank_product, prepare_ranked_factors,
                                weights_from_chromosome)
from evaluation.ic_analysis import compute_ic_icir_fast  # v7.8: 用numpy版替代scipy.spearmanr (释GIL)
from evaluation.decile_test import decile_portfolio_test, test_monotonicity
from evaluation.double_sort import independent_double_sort, size_single_sort
from evaluation.scoring import score_factor
from evaluation.stability import compute_stability_comprehensive

# === 换手惩罚配置 ===
MAX_ACCEPTABLE_WEEKLY_TO = 0.30   # 可接受的最大周换手率
MAX_TURNOVER_PENALTY = 0.25       # v7.2fix: 换手惩罚上限减半, clip模式可存活 (原0.50)


def icir_to_objective(icir, mode='clip', tanh_scale=0.5, clip_threshold=1.5):
    """
    将 ICIR 转换为 NSGA-II Obj1 目标值

    三种模式:
      'clip' — clip(|icir|/threshold, 0, 1), ratio 基准打分
      'raw'  — 直接用 |icir|, 无变换
      'tanh' — tanh(|icir|/scale), [0,1] 有界, 中间段梯度敏感

    Returns
    -------
    float
    """
    val = abs(icir)
    if mode == 'raw':
        return val
    elif mode == 'tanh':
        return float(np.tanh(val / tanh_scale))
    else:  # 'clip' (default)
        return float(np.clip(val / clip_threshold, 0, 1))


def _estimate_weekly_turnover(composite, top_n=30, buffer_mult=1.8):
    """
    基于合成因子的排名自相关估算组合周换手率

    模型: TO ≈ (1 - rank_autocorr_lag1) * (top_n / buffer_n)
    其中 buffer_n = top_n * buffer_mult

    Parameters
    ----------
    composite : pd.DataFrame
        合成因子 (index=日期, columns=股票代码)
    top_n : int
        持仓数
    buffer_mult : float
        Buffer倍数 (持有 top_n*buffer_mult, 仅交易 top_n)

    Returns
    -------
    float
        估计周换手率 [0, 1]
    """
    if composite.empty or composite.shape[1] < 10 or composite.shape[0] < 4:
        return 1.0  # 数据不足, 假设最坏

    try:
        # 计算截面百分位排名
        ranks_df = composite.rank(axis=1, pct=True)
        ranks = ranks_df.values  # (n_weeks, n_stocks)

        # ★ 向量化 rank autocorr lag=1 (替代逐股循环~5700次)
        # corr(ranks[:-1], ranks[1:]) => cov/(std_t * std_t1)
        ranks_t = ranks[:-1, :]   # lag 0: (n_weeks-1, n_stocks)
        ranks_t1 = ranks[1:, :]   # lag 1: (n_weeks-1, n_stocks)

        ranks_t_c = ranks_t - ranks_t.mean(axis=0, keepdims=True)
        ranks_t1_c = ranks_t1 - ranks_t1.mean(axis=0, keepdims=True)

        cov = (ranks_t_c * ranks_t1_c).sum(axis=0)  # (n_stocks,)
        var_t = (ranks_t_c ** 2).sum(axis=0)
        var_t1 = (ranks_t1_c ** 2).sum(axis=0)
        std_prod = np.sqrt(var_t * var_t1)

        valid = (std_prod > 1e-10) & (~np.isnan(ranks_t).any(axis=0)) & (~np.isnan(ranks_t1).any(axis=0))

        if valid.sum() < 10:
            return 1.0

        corrs = np.zeros(ranks.shape[1])
        corrs[valid] = cov[valid] / std_prod[valid]
        corrs = corrs[np.isfinite(corrs)]
        avg_rank_ac = float(np.mean(corrs))
        avg_rank_ac = max(min(avg_rank_ac, 0.999), 0.0)

        # 估计: 每周排名变化比例
        rank_change_rate = 1.0 - avg_rank_ac

        # 在Top N边界处, 放大因子
        buffer_n = int(top_n * buffer_mult)
        boundary_amplification = top_n / buffer_n

        turnover = rank_change_rate * boundary_amplification
        return float(np.clip(turnover, 0.0, 1.0))

    except Exception:
        return 1.0


def _compute_turnover_penalty(weekly_turnover):
    """
    换手惩罚函数 (v5.2 线性)

    penalty = min(weekly_TO / MAX_ACCEPTABLE_WEEKLY_TO, MAX_TURNOVER_PENALTY)

    例:
      TO=10% → penalty=0.100/0.30=0.333 → 稳定性打0.667折
      TO=30% → penalty=0.300/0.30=1.000 → 稳定性打0折(实际最低0.5折)
      TO=50% → penalty=0.500/0.30 但上限0.5 → 稳定性打0.5折
    """
    penalty = weekly_turnover / MAX_ACCEPTABLE_WEEKLY_TO
    return float(np.clip(penalty, 0.0, MAX_TURNOVER_PENALTY))


def _compute_category_penalty(weights, factor_names):
    """
    品类集中度惩罚 (v5 新增)

    遍历 CONCENTRATION_GROUPS 中每个品类族, 若族内因子权重之和 > 40%,
    超额部分 * 0.5 累加惩罚, 总惩罚上限 40%.

    Parameters
    ----------
    weights : dict {factor_name: weight}
    factor_names : list

    Returns
    -------
    float : penalty in [0, 0.4]
    """
    try:
        from config import CONCENTRATION_GROUPS, MAX_CATEGORY_WEIGHT, \
            CATEGORY_CONCENTRATION_PENALTY_MULT, FACTOR_DEFS

        total_penalty = 0.0

        for group_name, group_categories in CONCENTRATION_GROUPS.items():
            group_weight = 0.0
            for name, w in weights.items():
                if name in FACTOR_DEFS and FACTOR_DEFS[name]['category'] in group_categories:
                    group_weight += abs(w)

            if group_weight > MAX_CATEGORY_WEIGHT:
                excess = group_weight - MAX_CATEGORY_WEIGHT
                total_penalty += excess * CATEGORY_CONCENTRATION_PENALTY_MULT

        return float(np.clip(total_penalty, 0.0, 0.4))
    except Exception:
        return 0.0


def _compute_complexity_penalty(n_active, mode='linear'):
    """
    因子数复杂度惩罚 (v6 回退: MAX_FACTORS硬约束 + linear)

    超过惩罚起点后按所选模式增长:
      linear:      每个超额因子扣固定rate (v6默认)
      quadratic:   三角形惩罚 (v5.3, 备用)
      exponential: 指数级惩罚 (v5.4, 备用)

    Parameters
    ----------
    n_active : int  活跃因子数
    mode : str  'linear' | 'quadratic' | 'exponential'

    Returns
    -------
    float : penalty in [0, COMPLEXITY_PENALTY_CAP]
    """
    from config import MAX_FACTORS, COMPLEXITY_PENALTY_CAP

    # v6: 硬约束 — n_active 不应超过 MAX_FACTORS, 惩罚起点=MAX_FACTORS
    penalty_start = MAX_FACTORS

    if n_active <= penalty_start:
        return 0.0

    excess = n_active - penalty_start

    if mode == 'exponential':
        penalty = 0.30 * (2 ** (excess - 1))
        penalty = min(penalty, COMPLEXITY_PENALTY_CAP)
    elif mode == 'quadratic':
        penalty = 0.05 * excess * (excess + 1) / 2.0
    else:
        # 线性 (v6默认): 每超额1个扣 cap/start
        penalty = excess * (COMPLEXITY_PENALTY_CAP / max(penalty_start, 1))

    return float(np.clip(penalty, 0.0, COMPLEXITY_PENALTY_CAP))


def _compute_correlation_penalty(weights, factor_names, corr_matrix):
    """
    因子间相关性惩罚 (v5.2 新增)

    选中因子的平均 pairwise |corr| 越高 → 惩罚越大,
    鼓励各因子独立贡献组合收益.

    penalty = min(avg_pairwise_corr / MAX_ACCEPTABLE_CORR * MULT, 1.0)

    Parameters
    ----------
    weights : dict {factor_name: weight}
    factor_names : list (unused, kept for consistency)
    corr_matrix : pd.DataFrame or None

    Returns
    -------
    float : penalty in [0, 1]
    """
    from config import MAX_ACCEPTABLE_CORR, CORRELATION_PENALTY_MULT, CORRELATION_PENALTY_MAX

    if corr_matrix is None or len(weights) <= 1:
        return 0.0

    selected = [name for name in weights if name in corr_matrix.index]
    if len(selected) <= 1:
        return 0.0

    # 提取选中因子的 pairwise 相关性 (上三角)
    sub_corr = corr_matrix.loc[selected, selected]
    n = len(selected)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            v = sub_corr.iloc[i, j]
            if np.isfinite(v):
                vals.append(abs(v))

    if not vals:
        return 0.0

    avg_corr = np.mean(vals)
    penalty = (avg_corr / MAX_ACCEPTABLE_CORR) * CORRELATION_PENALTY_MULT
    return float(np.clip(penalty, 0.0, CORRELATION_PENALTY_MAX))


def _compute_top30_cost_adjusted_sharpe(composite, forward_returns, 
                                         force_direction=None, debug=False):
    """
    Obj3: Top30等权组合 成本后年化Sharpe (v7.2 方向修正 + 诊断增强)

    模拟真实交易: 每周末按合成因子排名选top30, 等权持有,
    扣除 QMT 标准双边成本后计算净周收益 → 年化Sharpe.

    v7.2 修复:
      - ★ 方向修正: 若composite ICIR<0 (做空因子), 自动翻转排名(选bottom30)
        force_direction=1强制做多, =-1强制做空, None=自动检测
      - ★ 诊断增强: debug=True 时强制输出成本拆解, 方便排查Obj3全负问题

    成本明细 (QMT实盘):
      买入: 万1佣金 + 0.3%滑点 = 0.0031
      卖出: 万1佣金 + 千1印花税 + 0.3%滑点 = 0.0041
      双边: 0.0072 per 换手单位

    Parameters
    ----------
    composite : pd.DataFrame
        合成因子 (index=周日期, columns=股票代码)
    forward_returns : pd.DataFrame
        前向周收益 (index=周日期, columns=股票代码)
    force_direction : int or None
        1=强制Top30(做多), -1=强制Bottom30(做空), None=自动ICIR检测
    debug : bool
        强制输出成本拆解诊断

    Returns
    -------
    dict : {'sharpe_score': float, 'annual_sharpe': float, 'gross_annual': float,
            'cost_annual': float, 'net_annual': float, 'avg_turnover': float,
            'direction': int, 'n_weeks': int}
    """
    ROUND_TRIP_COST = 0.0072  # 双边全部摩擦 (买+卖)

    # Align indices and columns
    common_idx = composite.index.intersection(forward_returns.index)
    if len(common_idx) < 12:
        return {'sharpe_score': 0.0, 'annual_sharpe': 0.0, 'gross_annual': 0.0,
                'cost_annual': 0.0, 'net_annual': 0.0, 'avg_turnover': 0.0,
                'direction': 0, 'n_weeks': 0}
    common_cols = composite.columns.intersection(forward_returns.columns)
    if len(common_cols) < 30:
        return {'sharpe_score': 0.0, 'annual_sharpe': 0.0, 'gross_annual': 0.0,
                'cost_annual': 0.0, 'net_annual': 0.0, 'avg_turnover': 0.0,
                'direction': 0, 'n_weeks': 0}

    # === v7.3: Obj3 用最近1年数据确保时效性 ===
    # 优先训练窗口最后一年(2024), 不足则回退到最近52周, 再不足用全部
    OBJ3_PREFERRED_START = '2024-01-01'  # 2026-07-13: 全样本训练(2021~2024), Obj3用最后一年2024
    OBJ3_RECENT_WEEKS = 52
    try:
        full_idx = common_idx.sort_values()
        # 优先: 仅用2026年后数据
        recent_mask = full_idx >= pd.Timestamp(OBJ3_PREFERRED_START)
        recent_idx = full_idx[recent_mask]
        if len(recent_idx) < 12:
            # 次选: 最近52周
            recent_idx = full_idx[-min(OBJ3_RECENT_WEEKS, len(full_idx)):]
        if len(recent_idx) >= 12:
            comp_arr = composite.loc[recent_idx, common_cols].values
            fr_arr = forward_returns.loc[recent_idx, common_cols].values
        else:
            # 回退用全部
            comp_arr = composite.loc[common_idx, common_cols].values
            fr_arr = forward_returns.loc[common_idx, common_cols].values
    except Exception:
        comp_arr = composite.loc[common_idx, common_cols].values
        fr_arr = forward_returns.loc[common_idx, common_cols].values

    n_weeks, n_stocks = comp_arr.shape

    # === ★ v7.2: 方向自动检测 ===
    # ★ v7.8: 用 numpy rankdata 替代 scipy.spearmanr (释GIL, 线程池友好)
    if force_direction is not None:
        direction = force_direction
    else:
        try:
            from scipy.stats import rankdata
            ic_vals = []
            for w in range(min(n_weeks, 52)):
                valid_mask = ~np.isnan(comp_arr[w]) & ~np.isnan(fr_arr[w])
                if valid_mask.sum() < 30:
                    continue
                x = comp_arr[w, valid_mask]
                y = fr_arr[w, valid_mask]
                n = len(x)
                rk_x = rankdata(x)
                rk_y = rankdata(y)
                d2 = np.sum((rk_x - rk_y) ** 2)
                rho = 1.0 - 6.0 * d2 / (n * (n * n - 1))
                if np.isfinite(rho):
                    ic_vals.append(rho)
            avg_ic = np.mean(ic_vals) if ic_vals else 0.0
            direction = 1 if avg_ic >= 0 else -1
        except Exception:
            direction = 1  # 默认做多

    # === 向量化 Top30 选择 (方向感知) ===
    comp_filled = np.where(np.isnan(comp_arr), -np.inf, comp_arr)
    if direction >= 0:
        # 做多: 选因子值最高的30只
        rankings = np.argsort(-comp_filled, axis=1)
    else:
        # 做空: 选因子值最低的30只 (反转排名)
        rankings = np.argsort(comp_filled, axis=1)
    top30_idx = rankings[:, :30]  # (n_weeks, 30)

    # === 向量化收益计算 ===
    week_indices = np.arange(n_weeks - 1)[:, None]  # (n_weeks-1, 1)
    stock_indices = top30_idx[:-1, :]  # (n_weeks-1, 30)
    gross_rets = np.nanmean(fr_arr[week_indices, stock_indices], axis=1)

    # === 向量化换手计算 ===
    top30_mask = np.zeros((n_weeks, n_stocks), dtype=bool)
    for i in range(n_weeks):
        top30_mask[i, top30_idx[i]] = True
    overlap = (top30_mask[:-1] & top30_mask[1:]).sum(axis=1)
    turnovers = 1.0 - overlap / 30.0

    # === 净收益 ===
    costs = turnovers * ROUND_TRIP_COST
    net_rets = gross_rets - costs

    # === Sharpe 计算 ===
    valid_rets = net_rets[np.isfinite(net_rets)]
    if len(valid_rets) < 10:
        return {'sharpe_score': 0.0, 'annual_sharpe': 0.0, 'gross_annual': 0.0,
                'cost_annual': 0.0, 'net_annual': 0.0, 'avg_turnover': 0.0,
                'direction': direction, 'n_weeks': len(valid_rets)}

    mean_ret = np.mean(valid_rets)
    std_ret = np.std(valid_rets, ddof=1)
    if std_ret < 1e-8:
        return {'sharpe_score': 0.0, 'annual_sharpe': 0.0, 'gross_annual': 0.0,
                'cost_annual': 0.0, 'net_annual': 0.0, 'avg_turnover': 0.0,
                'direction': direction, 'n_weeks': len(valid_rets)}

    annual_sharpe = mean_ret / std_ret * np.sqrt(52)

    # === 成本拆解 ===
    mean_gross = np.nanmean(gross_rets) * 52
    mean_cost = np.nanmean(costs) * 52
    mean_net = np.nanmean(net_rets) * 52
    mean_to = np.nanmean(turnovers)

    # ★ v7.2: debug模式或环境变量 → 强制输出诊断
    import os
    if debug or os.environ.get('FA_OBJ3_DEBUG', '0') == '1':
        dir_label = 'LONG(Top30)' if direction >= 0 else 'SHORT(Bot30)'
        print(f"  [Obj3 DIAG] {dir_label} | gross_ann={mean_gross:+.4f} "
              f"cost_ann={mean_cost:.4f} net_ann={mean_net:+.4f} "
              f"avg_wk_to={mean_to:.3f} sharpe={annual_sharpe:+.3f} "
              f"n_wks={len(valid_rets)}")

    result = {
        'sharpe_score': float(np.clip(annual_sharpe / 2.0, 0.0, 1.0)),
        'annual_sharpe': float(annual_sharpe),
        'gross_annual': float(mean_gross),
        'cost_annual': float(mean_cost),
        'net_annual': float(mean_net),
        'avg_turnover': float(mean_to),
        'direction': direction,
        'n_weeks': len(valid_rets),
    }
    return result


def _compute_max_corr_composite(composite, factor_dict, min_pairs=30):
    """
    高效计算 composite 与各因子间的最大 Spearman 相关系数
    
    直接拉平时间序列计算, 避免每次重建完整相关矩阵
    """
    if composite.empty or not factor_dict:
        return 0.0
    
    # 拉平 composite
    flat_comp = composite.stack().dropna().rename('comp')
    
    max_corr = 0.0
    for name, df in factor_dict.items():
        try:
            flat_factor = df.stack().dropna().rename(name)
            # 取交集
            aligned = pd.concat([flat_comp, flat_factor], axis=1).dropna()
            if len(aligned) < min_pairs:
                continue
            corr = stats.spearmanr(aligned['comp'], aligned[name])
            corr_val = abs(corr[0] if isinstance(corr, tuple) else corr.statistic)
            if not np.isnan(corr_val) and corr_val > max_corr:
                max_corr = corr_val
        except:
            continue
    
    return float(max_corr)


def compute_fitness(chromosome, factor_names, factor_dict, forward_returns, 
                    mcap_df, corr_matrix=None, is_size_map=None,
                    ic_summary=None, return_details=False):
    """
    计算单个染色体的适应度 (五维度)
    
    Parameters
    ----------
    chromosome : np.ndarray
        权重向量
    factor_names : list
    factor_dict : dict {name: DataFrame}
    forward_returns : pd.DataFrame
    mcap_df : pd.DataFrame
    corr_matrix : pd.DataFrame or None
        预计算的因子相关性矩阵 (若提供, 避免重复计算)
    is_size_map : dict or None
        {factor_name: bool} 规模因子标记 (默认 {})
    ic_summary : pd.DataFrame or None
        预计算的 IC summary (未使用, 占位)
    return_details : bool
        是否返回详细分解 (默认 False)
    
    Returns
    -------
    float or (float, dict)
        适应度值 [0, 1], 若 return_details=True 则返回 (fitness, details)
    """
    try:
        # 1. 提取权重
        weights = weights_from_chromosome(chromosome, factor_names)
        
        if len(weights) == 0:
            return (0.0, {}) if return_details else 0.0
        
        # 2. 合成因子
        selected_factors = {name: factor_dict[name] for name in weights if name in factor_dict}
        if len(selected_factors) == 0:
            return (0.0, {}) if return_details else 0.0
        
        composite = combine_factors(selected_factors, weights)
        if composite.empty:
            return (0.0, {}) if return_details else 0.0
        
        # 确定是否为规模因子
        is_size_map_local = is_size_map or {}
        is_size = any(is_size_map_local.get(name, False) for name in weights)
        
        # 3. ICIR
        icir_result = compute_ic_icir_fast(composite, forward_returns)
        icir = abs(icir_result['icir'])
        if np.isnan(icir):
            icir = 0.0
        
        # 4. 单调性
        decile_result = decile_portfolio_test(composite, forward_returns)
        mono_result = test_monotonicity(decile_result)
        mono_pval = mono_result.get('p_value', 1.0)
        if np.isnan(mono_pval):
            mono_pval = 1.0
        
        # 5. 低相关性: 计算 composite 与各入选因子的最大相关
        max_corr = _compute_max_corr_composite(composite, selected_factors)
        if np.isnan(max_corr):
            max_corr = 0
        
        # 6. 双重排序
        if is_size:
            ds_result = size_single_sort(mcap_df, forward_returns)
        else:
            ds_result = independent_double_sort(composite, forward_returns, mcap_df)
        
        ds_pval = ds_result.get('p_value', 1.0)
        if np.isnan(ds_pval):
            ds_pval = 1.0
        
        # 7. 时序稳定性
        try:
            ic_series = icir_result.get('ic_series', None)
            stab_result = compute_stability_comprehensive(
                composite, forward_returns=forward_returns,
                ic_series=ic_series, top_n=30,
            )
            stab_score = stab_result.get('total_stability_score', 0.5)
            if np.isnan(stab_score):
                stab_score = 0.5
        except:
            stab_score = 0.5
            stab_result = {}
        
        # 8. 综合评分 (五维度)
        from evaluation.scoring import ICIR_THRESHOLD, MONOTONICITY_P_THRESHOLD
        from evaluation.scoring import CORRELATION_THRESHOLD, DOUBLE_SORT_P_THRESHOLD
        from evaluation.scoring import STABILITY_THRESHOLD
        from evaluation.scoring import GA_WEIGHT_ICIR, GA_WEIGHT_MONOTONICITY
        from evaluation.scoring import GA_WEIGHT_CORRELATION, GA_WEIGHT_DOUBLE_SORT
        from evaluation.scoring import GA_WEIGHT_STABILITY
        
        # 各维度得分
        icir_score = np.clip(icir / ICIR_THRESHOLD, 0, 1)
        mono_score = 1.0 - np.clip(mono_pval / MONOTONICITY_P_THRESHOLD, 0, 1)
        if np.isnan(mono_score): mono_score = 0
        corr_score = np.clip(1.0 - max_corr / CORRELATION_THRESHOLD, 0, 1)
        ds_score = 1.0 - np.clip(ds_pval / DOUBLE_SORT_P_THRESHOLD, 0, 1)
        if np.isnan(ds_score): ds_score = 0
        stab_norm = np.clip(stab_score / STABILITY_THRESHOLD, 0, 1)
        
        # 加权总分
        total = (
            GA_WEIGHT_ICIR * icir_score +
            GA_WEIGHT_MONOTONICITY * mono_score +
            GA_WEIGHT_CORRELATION * corr_score +
            GA_WEIGHT_DOUBLE_SORT * ds_score +
            GA_WEIGHT_STABILITY * stab_norm
        )
        
        fitness = float(np.clip(total, 0, 1))
        
        if return_details:
            details = {
                'icir': icir,
                'icir_score': icir_score,
                'mono_pval': mono_pval,
                'mono_score': mono_score,
                'max_corr': max_corr,
                'corr_score': corr_score,
                'ds_pval': ds_pval,
                'ds_score': ds_score,
                'stab_score': stab_score,
                'stab_norm': stab_norm,
                'total': fitness,
                'weights': weights,
                'stab_result': stab_result,
            }
            return fitness, details
        
        return fitness

    except Exception:
        return (0.0, {}) if return_details else 0.0


def evaluate_population(population, factor_names, factor_dict, forward_returns,
                         mcap_df, corr_matrix=None, is_size_map=None):
    """
    评估整个种群的适应度
    
    Returns
    -------
    np.ndarray
        适应度数组
    """
    fitness = np.zeros(len(population))
    for i, chromo in enumerate(population):
        fitness[i] = compute_fitness(
            chromo, factor_names, factor_dict, forward_returns,
            mcap_df, corr_matrix=corr_matrix, is_size_map=is_size_map
        )
    return fitness


def compute_multi_objective(chromosome, factor_names, factor_dict,
                             forward_returns, mcap_df, corr_matrix=None,
                             is_size_map=None, ic_summary=None,
                             factor_ic_series=None,
                             factor_dfs_std=None, factor_dfs_rp=None, debug_obj3=False):
    """
    三目标适应度 (v7.2: 真·加法链 + 纯稳定 + Top30成本后夏普)
    =====================================================
    Obj1 = icir_score - Σpenalties + bonus  净预测力质量 (真加法链, ∈[0,1])
    Obj2 = stab_norm                          纯时序稳定性 (∈[0,1])
    Obj3 = top30等权成本后年化Sharpe           真实可执行收益 (∈[0,1])

    惩罚项 (Obj1 真加法链): TO + 品类 + 复杂度 + 相关性, 总上限80%
    基本面加分: >=2个fundamental因子 → +0.08

    v7.2 关键改进:
      - 方向感知: 复合因子ICIR < 0 → 自动反转选股方向 (做空→选Bottom30)
      - 标准化: factor_dfs_std不为None时使用预标准化因子加速组合
      - 诊断: debug_obj3=True强制输出Obj3成本拆解
    """
    # v8.0: rank-product + OOS + regime鲁棒
    try:
        from config import (FACTOR_DEFS, FUNDAMENTAL_CATEGORIES,
                           FUNDAMENTAL_BONUS, FUNDAMENTAL_MIN_COUNT, MAX_FACTORS,
                           N_OBJECTIVES, ICIR_OBJ_MODE, ICIR_TANH_SCALE, ICIR_THRESHOLD,
                           MAX_TOTAL_PENALTY)
        
        # 1. 提取权重 (硬约束 MAX_FACTORS=4)
        weights = weights_from_chromosome(chromosome, factor_names, max_factors=MAX_FACTORS)
        if len(weights) == 0:
            return (0.0,) * N_OBJECTIVES
        
        # 2. Rank-product 组合 (对齐 V2/V3 JQ)
        if factor_dfs_rp is not None:
            selected_rp = {name: factor_dfs_rp[name]
                          for name in weights if name in factor_dfs_rp}
            if len(selected_rp) == 0:
                return (0.0,) * N_OBJECTIVES
            composite = combine_factors_rank_product(selected_rp, weights)
        else:
            selected_factors = {name: factor_dict[name]
                              for name in weights if name in factor_dict}
            if len(selected_factors) == 0:
                return (0.0,) * N_OBJECTIVES
            ranked = prepare_ranked_factors(selected_factors)
            composite = combine_factors_rank_product(ranked, weights)
        
        if composite.empty or composite.shape[0] < 20:
            return (0.0,) * N_OBJECTIVES
        
        # 3. OOS 70:30 time split
        n_dates = len(composite.index)
        split_idx = int(n_dates * 0.70)
        if split_idx < 10 or (n_dates - split_idx) < 10:
            is_composite = oos_composite = composite
            is_fr = oos_fr = forward_returns
            n_oos = n_dates
        else:
            is_composite = composite.iloc[:split_idx]
            oos_composite = composite.iloc[split_idx:]
            is_fr = forward_returns.iloc[:split_idx]
            oos_fr = forward_returns.iloc[split_idx:]
            n_oos = n_dates - split_idx
        
        # 4. Obj1: OOS ICIR + penalties
        oos_icir_result = compute_ic_icir_fast(oos_composite, oos_fr)
        raw_icir = oos_icir_result['icir']
        oos_icir = abs(raw_icir)
        if np.isnan(oos_icir):
            oos_icir = 0.0
        icir_score = icir_to_objective(oos_icir, mode=ICIR_OBJ_MODE,
                                       tanh_scale=ICIR_TANH_SCALE,
                                       clip_threshold=ICIR_THRESHOLD)
        weekly_to = _estimate_weekly_turnover(is_composite, top_n=30, buffer_mult=1.8)
        to_pen = _compute_turnover_penalty(weekly_to)
        cat_pen = _compute_category_penalty(weights, factor_names)
        cpx_pen = _compute_complexity_penalty(len(weights))
        corr_pen = _compute_correlation_penalty(weights, factor_names, corr_matrix)
        penalty_sum = float(np.clip(to_pen + cat_pen + cpx_pen + corr_pen,
                                    0.0, MAX_TOTAL_PENALTY))
        n_fund = sum(1 for name in weights if name in FACTOR_DEFS
                     and FACTOR_DEFS[name]['category'] in FUNDAMENTAL_CATEGORIES)
        bonus = (FUNDAMENTAL_BONUS - 1.0) if n_fund >= FUNDAMENTAL_MIN_COUNT else 0.0
        obj1 = float(np.clip(icir_score - penalty_sum + bonus, 0.0, 1.0))
        
        # 5. Obj2: 跨窗口鲁棒性 (OOS切4窗口取最差ICIR)
        if n_oos >= 40:
            k = 4
            window_size = max(n_oos // k, 10)
            window_icirs = []
            for w in range(k):
                start = w * window_size
                end = min(start + window_size, n_oos)
                if end - start < 5:
                    continue
                w_comp = oos_composite.iloc[start:end]
                w_fr = oos_fr.iloc[start:end]
                try:
                    w_ic = compute_ic_icir_fast(w_comp, w_fr)['icir']
                    window_icirs.append(abs(w_ic))
                except Exception:
                    window_icirs.append(0.0)
            if window_icirs:
                min_w_icir = min(window_icirs)
                obj2 = float(np.clip(min_w_icir / 1.5, 0, 1))
            else:
                obj2 = 0.3
        else:
            if oos_icir > 0 and 'ic_series' in oos_icir_result:
                ic_s = oos_icir_result['ic_series']
                if ic_s is not None and len(ic_s) > 5:
                    obj2 = float(np.clip(
                        1.0 - np.std(ic_s) / (abs(np.mean(ic_s)) + 0.01), 0, 1))
                else:
                    obj2 = 0.3
            else:
                obj2 = 0.3
        
        # 6. Obj3: OOS 成本后夏普
        force_dir = 1 if raw_icir >= 0 else -1
        obj3_result = _compute_top30_cost_adjusted_sharpe(
            oos_composite, oos_fr,
            force_direction=force_dir,
            debug=debug_obj3
        )
        obj3 = obj3_result['sharpe_score']
        
        return obj1, obj2, obj3

    except Exception as e:
        import traceback
        chromo_preview = chromo[:16].tolist() if hasattr(chromo, 'tolist') else str(chromo)[:80]
        print(f"  [FATAL] compute_multi_objective 崩溃: chromo[:16]={chromo_preview} | {type(e).__name__}: {e}")
        traceback.print_exc()
        raise  # 交由 nsga2.evaluate_multi_objective 记录 gen/idx 并置零(不连坐整代)
