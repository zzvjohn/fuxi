"""
因子组合引擎
============
- 线性加权合成
- 等权 / 自定义权重
- GA 生成的最优权重向量
- v4: 向量化加速 (2026-06-24)
- v6 (2026-07-21): rank-based标准化 — 截面percentile→inverse normal N(0,1)
  免疫数据源单调变换差异(复权/价格缩放), 行业标准做法
"""
import numpy as np
import pandas as pd
from typing import Dict
from scipy.stats import norm as _norm


def standardize_factors(factor_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    预计算因子截面 rank-based 标准化 (v6)

    对每个因子逐周做截面标准化:
      1. percentile rank (0~1, 均匀分布)
      2. inverse normal → N(0,1)

    优势:
      - 对因子值单调变换完全免疫 (复权常数倍/价格缩放无影响)
      - 自动抑制极端值 (ranks bounded → 不需要 clip)
      - 跨数据源一致 (TS/JQ 即使原始值不同, 只要排名一致则标准分一致)
      - 多因子量化行业标准 (Barra/Axioma 均用 rank-based)

    v4: 预计算后 combine_factors_vectorized 可直接加权求和.
    v5 (2026-07-20): median/std z-score + clip[-3,3].
    v6 (2026-07-21): rank-based percentile → inverse normal.

    Parameters
    ----------
    factor_dict : {name: DataFrame(date x stocks)}

    Returns
    -------
    {name: DataFrame(date x stocks)} 标准化后的因子, 保留NaN
    """
    std_dict = {}
    for name, df in factor_dict.items():
        # 截面 rank-based: percentile rank → N(0,1)
        # rank(pct=True) 得到 [0,1] 均匀分布
        # norm.ppf 映射到标准正态 (尾部自然截断, 无需 clip)
        ranks = df.rank(axis=1, pct=True, na_option='keep')
        # 将 [0, 1] 映射到标准正态, clip 到 [-4, 4] 防止极端尾部
        z_scored = ranks.apply(
            lambda row: _norm.ppf(np.clip(row.values, 0.0001, 0.9999)),
            axis=1, result_type='broadcast'
        )
        # result_type='broadcast' 在某些pandas版本返回np.array, 手动转回
        if isinstance(z_scored, np.ndarray):
            z_scored = pd.DataFrame(z_scored, index=df.index, columns=df.columns)
        elif not isinstance(z_scored, pd.DataFrame):
            z_scored = pd.DataFrame(z_scored, index=df.index, columns=df.columns)
        z_scored = z_scored.clip(-4, 4)
        std_dict[name] = z_scored.astype(float)
    return std_dict


def standardize_factors_zscore(factor_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    (保留) v5 z-score 标准化, 用于对比测试

    z = (x - median) / std, clip[-3, 3]
    """
    std_dict = {}
    for name, df in factor_dict.items():
        row_median = df.median(axis=1, skipna=True)
        row_std = df.std(axis=1, skipna=True).replace(0, np.nan)
        z_scored = df.subtract(row_median, axis=0).div(row_std, axis=0)
        z_scored = z_scored.clip(-3, 3)
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

    v6 (2026-07-21): 内部标准化改为 rank-based percentile→N(0,1),
    与 standardize_factors 保持一致.
    """
    from scipy.stats import norm as _n

    if method == 'equal':
        w = 1.0 / len(factor_dict)
        weights = {k: w for k in factor_dict}

    all_dates = set()
    all_stocks = set()
    for df in factor_dict.values():
        all_dates.update(df.index)
        all_stocks.update(df.columns)

    all_dates = sorted(all_dates)
    all_stocks = sorted(all_stocks)

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
            valid = row.dropna()
            if len(valid) < 10:
                continue

            # rank-based: percentile → N(0,1), clip[-4,4]
            ranks = valid.rank(pct=True)
            z = pd.Series(
                _n.ppf(np.clip(ranks.values, 0.0001, 0.9999)),
                index=valid.index
            )
            z = z.clip(-4, 4)

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


def prepare_ranked_factors(factor_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    预计算因子截面 rank percentile → [0,1] 均匀分布
    用于 rank-product 组合, 与 V2/V3 JQ 策略方法对齐。
    
    Parameters
    ----------
    factor_dict : {name: DataFrame(date x stocks)} 原始因子值
    
    Returns
    -------
    {name: DataFrame(date x stocks)} rank(pct=True) 后的因子, 保留NaN
    """
    rp_dict = {}
    for name, df in factor_dict.items():
        rp_dict[name] = df.rank(axis=1, pct=True, na_option='keep').astype(float)
    return rp_dict


def combine_factors_rank_product(
    ranked_factor_dict: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    min_common_dates: int = 5,
) -> pd.DataFrame:
    """
    Rank-product 加权复合 (v8.0)
    ==========================
    与 V2/V3 JQ 策略方法完全一致: rank(pct=True).prod() 范式。
    
    对每个因子 i:
        r_i = rank_pct(factor_i)  ∈ [0,1]
        w_i = 归一化权重
    
    composite = Π (r_i ** w_i)  → 加权秩乘积
    然后对 composite 再做一次 rank(pct=True) → 最终截面排名
    
    关键性质:
      - 乘法交互: 因子间非补偿性 — 单因子近零 → 整体近零
      - 权重指数量纲: w_i→0 时 r_i^w_i→1 (无影响), w_i→1 时满乘影响力
      - 最终 rank 化: 输出截面排名, 免疫原始因子量纲
    
    Parameters
    ----------
    ranked_factor_dict : {name: DataFrame(date x stocks)} 已作 rank(pct=True) 的因子
    weights : {name: weight} 非归一化权重 (内部自动归一化)
    min_common_dates : 返回的最低有效日期数
    
    Returns
    -------
    pd.DataFrame 综合因子得分 (date x stocks), [0,1] rank percentile
    """
    composite = None
    total_weight = 0.0
    
    for name, r_df in ranked_factor_dict.items():
        w = abs(weights.get(name, 0.0))
        if w < 1e-10:
            continue
        
        # Apply weight as exponent
        # w→0 ⇒ r^w→1 ⇒ no multiplicative impact
        # w→1 ⇒ full multiplicative impact
        weighted = r_df.pow(w)
        
        if composite is None:
            composite = weighted
        else:
            aligned = composite.align(weighted, join='outer')
            composite = aligned[0].multiply(aligned[1], fill_value=1.0)
        
        total_weight += w
    
    if composite is None or total_weight == 0.0:
        return pd.DataFrame()
    
    # Rank-normalize: clip small values before final rank
    composite = composite.clip(lower=1e-10)
    composite = composite.rank(axis=1, pct=True, na_option='keep')
    
    valid_mask = composite.notna().sum(axis=1) >= 10
    composite = composite[valid_mask]
    
    if len(composite) < min_common_dates:
        return pd.DataFrame()
    
    return composite


def compute_combined_factor_score(factor_dict, chromosome, factor_names):
    """
    GA 染色体 → 综合因子得分 (兼容接口, 内部调用 rank-product)

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
    ranked = prepare_ranked_factors(factor_dict)
    selected = {n: ranked[n] for n in weights if n in ranked}
    return combine_factors_rank_product(selected, weights)
