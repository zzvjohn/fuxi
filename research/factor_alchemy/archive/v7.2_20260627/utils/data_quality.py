"""
数据质量过滤
"""
import numpy as np
import pandas as pd


def filter_negative_equity(df, equity_col='equity_parent'):
    """
    剔除净资产为负的股票
    返回布尔 mask
    """
    if equity_col not in df.columns:
        return pd.Series(True, index=df.index)
    return df[equity_col] > 0


def handle_suspension(price_df, suspend_df, momentum_cols=None, volatility_cols=None):
    """
    停牌处理:
    - 动量类指标: ffill (用上一个非停牌日价格填充)
    - 波动率类指标: 停牌日设为 NaN
    
    Parameters
    ----------
    price_df : pd.DataFrame
        价格数据, index=date, columns=stocks
    suspend_df : pd.DataFrame
        停牌数据 (需有 stock_code, suspend_date 列)
    momentum_cols : list
        后续会用于动量计算的列
    volatility_cols : list
        后续会用于波动率计算的列
    
    Returns
    -------
    pd.DataFrame
        处理后的价格
    """
    # TODO: 实现停牌标记和填充逻辑
    # 此函数在数据加载阶段使用, 暂时返回原数据
    return price_df


def check_min_samples(data, min_ratio=2/3):
    """
    检查有效样本比例
    
    Parameters
    ----------
    data : pd.Series or np.ndarray
    min_ratio : float
        最低有效比例
    
    Returns
    -------
    bool
        True 如果有效样本 >= min_ratio
    """
    n_total = len(data)
    n_valid = (~np.isnan(data)).sum()
    return n_valid / max(n_total, 1) >= min_ratio


def filter_st_stocks(stock_list, st_status_map):
    """过滤ST股票"""
    if st_status_map is None:
        return stock_list
    return [s for s in stock_list if not st_status_map.get(s, False)]


def filter_ke_chuang(stock_list):
    """过滤科创板 (688/689)"""
    return [s for s in stock_list if not (str(s).startswith('688') or str(s).startswith('689'))]


def filter_chi_next(stock_list):
    """过滤创业板 (300/301)"""
    return [s for s in stock_list if not (str(s).startswith('300') or str(s).startswith('301'))]
