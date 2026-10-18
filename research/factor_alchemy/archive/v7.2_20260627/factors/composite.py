"""
因子组合引擎
============
- 线性加权合成
- 等权 / 自定义权重
- GA 生成的最优权重向量
- v4: 向量化加速 (2026-06-24)
"""
import numpy as np
import pandas as pd
from typing import Dict


def standardize_factors(factor_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    预计算因子截面z-score标准化
    
    对每个因子TimeFrame做截面标准化 (axis=1):
      z_score = (x - cross_sectional_mean) / cross_sectional_std
    
    v4: 预计算后 combine_factors_vectorized 可直接加权求和,
        避免逐日 .loc 循环的巨大开销。
    
    Parameters
    ----------
    factor_dict : {name: DataFrame(date x stocks)}
    
    Returns
    -------
    {name: DataFrame(date x stocks)} 标准化后的因子, 保留NaN
    """
    std_dict = {}
    for name, df in factor_dict.items():
        # 截面z-score: (x - mean) / std
        row_mean = df.mean(axis=1, skipna=True)
        row_std = df.std(axis=1, skipna=True).replace(0, np.nan)
        z_scored = df.subtract(row_mean, axis=0).div(row_std, axis=0)
        std_dict[name] = z_scored
    return std_dict


def combine_factors_vectorized(
    std_factor_dict: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    min_common_dates: int = 5,
) -> pd.DataFrame:
    """
    向量化线性加权合成因子 (从预标准化数据)
    
    v4: 替代原 combine_factors 的逐日循环, 直接对 full DataFrame
        做加权求和, 性能提升 50-200x (取决于因子数和日期数)。
    
    Parameters
    ----------
    std_factor_dict : {name: DataFrame(date x stocks)} 预标准化后的因子
    weights : {name: weight}
    min_common_dates : 返回的最低有效日期数
    
    Returns
    -------
    pd.DataFrame 综合因子得分 (date x stocks)
    """
    composite = None
    total_weight = 0.0
    
    for name, z_df in std_factor_dict.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        
        if composite is None:
            composite = z_df.multiply(w)
        else:
            # 对齐后加权加总
            aligned = composite.align(z_df.multiply(w), join='outer')
            composite = aligned[0].add(aligned[1], fill_value=0)
        
        total_weight += w
    
    if composite is None or total_weight == 0.0:
        return pd.DataFrame()
    
    composite = composite.div(total_weight)
    
    # 只保留有足够数据的日期
    valid_mask = composite.notna().sum(axis=1) >= 10
    composite = composite[valid_mask]
    
    if len(composite) < min_common_dates:
        return pd.DataFrame()
    
    return composite


def combine_factors(factor_dict, weights, method='linear'):
    """
    线性加权合成因子 (逐日循环, 向后兼容)
    
    注意: 对于 Phase 3.5 Beam Search 等高频调用场景,
    请使用 standardize_factors() + combine_factors_vectorized() 组合。
    
    Parameters
    ----------
    factor_dict : dict
        {factor_name: pd.DataFrame(index=date, columns=stocks)}
    weights : dict
        {factor_name: weight}
    method : str
        'linear' 或 'equal'
    
    Returns
    -------
    pd.DataFrame
        综合因子得分
    """
    if method == 'equal':
        w = 1.0 / len(factor_dict)
        weights = {k: w for k in factor_dict}
    
    # 对齐所有因子的时间和股票
    all_dates = set()
    all_stocks = set()
    for df in factor_dict.values():
        all_dates.update(df.index)
        all_stocks.update(df.columns)
    
    all_dates = sorted(all_dates)
    all_stocks = sorted(all_stocks)
    
    # 初始化综合得分矩阵
    composite = pd.DataFrame(
        np.nan, index=all_dates, columns=all_stocks, dtype=float
    )
    
    for date in all_dates:
        scores = []
        total_weight = 0.0
        
        for name, df in factor_dict.items():
            w = weights.get(name, 0.0)
            if w == 0 or date not in df.index:
                continue
            
            row = df.loc[date].reindex(all_stocks)
            if row.notna().sum() < 10:
                continue
            
            # 截面标准化
            valid = row.dropna()
            if len(valid) < 10:
                continue
            z = (valid - valid.mean()) / valid.std()
            
            scores.append(z * w)
            total_weight += w
        
        if total_weight > 0 and scores:
            combined = sum(scores) / total_weight
            composite.loc[date, combined.index] = combined
    
    return composite.dropna(how='all')


def weights_from_chromosome(chromosome, factor_names, threshold=0.05, max_factors=None):
    """
    从 GA 染色体提取权重（动态阈值 + Top-K fallback）
    
    Parameters
    ----------
    chromosome : np.ndarray or list
        权重向量 [w1, w2, ..., wn]
    factor_names : list
        因子名列表
    threshold : float
        权重阈值基准, 实际使用 max(threshold, 1/(2*N))
    max_factors : int or None
        最多保留的因子数, None=不限制
    
    Returns
    -------
    dict
        {factor_name: normalized_weight}
    """
    chromo = np.array(chromosome)
    n_factors = len(factor_names)
    
    # 动态阈值: 至少为 1/(2*N), 确保大规模因子池时不会全军覆没
    effective_threshold = max(threshold, 1.0 / (2 * n_factors))
    
    # Softmax 归一化
    exp_w = np.exp(np.clip(chromo, -10, 10))
    w = exp_w / exp_w.sum()
    
    # 确定入选的因子索引
    if max_factors is not None:
        # Top-K 模式: 按权重排序取前K
        top_k = min(max_factors, n_factors)
        indices = np.argsort(w)[-top_k:]
    else:
        # 阈值模式
        indices = [i for i in range(n_factors) if w[i] >= effective_threshold]
    
    result = {factor_names[i]: w[i] for i in indices}
    
    # Fallback: 如果没选到任何因子, 取 Top-10
    if not result:
        top_k = min(10, n_factors)
        indices = np.argsort(w)[-top_k:]
        result = {factor_names[i]: w[i] for i in indices}
    
    # 重归一化
    total = sum(result.values())
    result = {k: v / total for k, v in result.items()}
    
    return result


def compute_combined_factor_score(factor_dict, chromosome, factor_names):
    """
    GA 染色体 → 综合因子得分
    
    Parameters
    ----------
    factor_dict : dict
    chromosome : list/array
    factor_names : list
    
    Returns
    -------
    pd.DataFrame
    """
    weights = weights_from_chromosome(chromosome, factor_names)
    return combine_factors(factor_dict, weights)
