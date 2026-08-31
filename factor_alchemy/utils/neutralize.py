"""
行业 + 市值中性化
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def neutralize_industry_mcap(factor_series, industry_map, mcap_series, 
                              factor_name='factor'):
    """
    行业 + 市值正交化
    
    对因子做: factor ~ industry_dummies + ln(mcap) 回归, 取残差
    
    Parameters
    ----------
    factor_series : pd.Series
        因子值 (index=stock_code), 已缩尾
    industry_map : pd.Series or dict
        股票代码 → 行业代码 (如申万一级)
    mcap_series : pd.Series
        股票代码 → 总市值
    factor_name : str
        因子名 (日志用)
    
    Returns
    -------
    pd.Series
        中性化后的因子值 (残差)
    """
    # 对齐
    common = factor_series.index & set(industry_map.keys()) & mcap_series.index
    if len(common) < 30:
        return factor_series
    
    y = factor_series.loc[common].dropna()
    ln_mcap = np.log(mcap_series.loc[y.index])
    industry = pd.Series({c: industry_map[c] for c in y.index})
    
    if y.empty or len(y) < 30:
        return factor_series
    
    # 构建设计矩阵
    ind_dummies = pd.get_dummies(industry, drop_first=True)
    X = pd.concat([ind_dummies, ln_mcap.rename('ln_mcap')], axis=1)
    
    # 对齐
    aligned = y.index & X.index
    X = X.loc[aligned]
    y = y.loc[aligned]
    
    if len(y) < 30:
        return factor_series
    
    # OLS 回归取残差
    try:
        model = sm.OLS(y.values, sm.add_constant(X.values.astype(float)))
        results = model.fit()
        residuals = y - results.fittedvalues
        return pd.Series(residuals, index=y.index)
    except Exception:
        return factor_series


def neutralize_mcap_only(factor_series, mcap_series):
    """仅市值中性化"""
    common = factor_series.index & mcap_series.index
    if len(common) < 30:
        return factor_series
    
    y = factor_series.loc[common].dropna()
    ln_mcap = np.log(mcap_series.loc[y.index])
    
    aligned = y.index & ln_mcap.index
    y = y.loc[aligned]
    ln_mcap = ln_mcap.loc[aligned]
    
    if len(y) < 30:
        return factor_series
    
    try:
        model = sm.OLS(y.values, sm.add_constant(ln_mcap.values.astype(float)))
        results = model.fit()
        residuals = y - results.fittedvalues
        return pd.Series(residuals, index=y.index)
    except Exception:
        return factor_series
