# -*- coding: utf-8 -*-
"""
因子计算前的原始数据预处理
===========================
- winsorize_price_data: 对日频价格数据逐日截面截尾(1%/99%)
- winsorize_financial_data: 对财报数据逐期截面截尾(1%/99%)
- winsorize_valuation_data: 对日频估值数据逐日截面截尾(1%/99%)

为什么在原始数据层做:
  极端单日值会通过 rolling.std()/rolling.mean() 污染整个窗口
  (如 turnover_cv_20d, 20天统计被一个异常日打坏)。
  winsorize 后滚动统计在「干净」数据上算, 与 Barra/Axioma 一致。

为什么不用 z-score clip[-3,3]:
  金融数据厚尾, 很多合法波动 >3σ → clip 会截掉真实信号。
  percentile winsorize(1%/99%) 只截真正异常值。
"""
import numpy as np
import pandas as pd
from typing import Dict


def _winsorize_df(df, lower_pct: float = 0.01, upper_pct: float = 0.99):
    """
    对 DataFrame (逐列) 或 Series 做 winsorize。
    
    DataFrame: 每列 (每只股票的时间序列) 独立 winsorize
    Series: 整个序列 winsorize
    
    返回: 同类型 (不修改原数据)
    """
    if df is None:
        return df
    result = df.copy()
    
    if isinstance(result, pd.Series):
        s = result.dropna()
        if len(s) < 20:
            return result
        lo = s.quantile(lower_pct)
        hi = s.quantile(upper_pct)
        if lo < hi:
            result = result.clip(lo, hi)
        return result
    
    # DataFrame
    if result.empty:
        return result
    for col in result.columns:
        s = result[col].dropna()
        if len(s) < 20:
            continue
        lo = s.quantile(lower_pct)
        hi = s.quantile(upper_pct)
        if lo < hi:
            result[col] = result[col].clip(lo, hi)
    return result


def winsorize_price_data(price_data: Dict[str, pd.DataFrame],
                          lower_pct: float = 0.01,
                          upper_pct: float = 0.99) -> Dict[str, pd.DataFrame]:
    """
    预处理价格数据字典。

    对 close/open/high/low: 按股票时间序列 winsorize
    对 volume/turnover/amount: 按股票时间序列 winsorize

    price_data keys: 'close', 'open', 'high', 'low', 'volume', 'turnover', 'amount' 等
    每个 key 对应 DataFrame(index=date, columns=stocks)

    返回: 新 dict, 每个 DataFrame 已 winsorize
    """
    if price_data is None:
        return price_data
    result = {}
    for key, df in price_data.items():
        if df is None or df.empty:
            result[key] = df
            continue
        # 对量价数据用时间序列 winsorize (每只股独立处理)
        result[key] = _winsorize_df(df, lower_pct, upper_pct)
    return result


def winsorize_financial_data(financial_data: pd.DataFrame,
                              lower_pct: float = 0.01,
                              upper_pct: float = 0.99) -> pd.DataFrame:
    """
    预处理财报数据 (长表格式)。

    financial_data: 长表, 含 ts_code / ann_date / end_date + 多个数值指标列。
    按报告期 end_date 分组, 对每只股票同期的截面做 winsorize,
    截掉财报异常值 (如 roe/roa 极端值)。

    返回: 新 DataFrame (索引/列结构不变, 仅数值列被 clip)
    """
    if financial_data is None or financial_data.empty:
        return financial_data

    result = financial_data.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return result

    # 报告期列: 优先 end_date
    period_col = 'end_date' if 'end_date' in result.columns else None

    if period_col is None:
        # 无报告期 → 全局截面 winsorize
        for col in numeric_cols:
            s = result[col].dropna()
            if len(s) < 10:
                continue
            lo, hi = s.quantile(lower_pct), s.quantile(upper_pct)
            if lo < hi:
                result[col] = result[col].clip(lo, hi)
        return result

    # 按报告期分组, 组内截面 winsorize
    for _, idx in result.groupby(period_col).groups.items():
        for col in numeric_cols:
            s = result.loc[idx, col].dropna()
            if len(s) < 10:
                continue
            lo, hi = s.quantile(lower_pct), s.quantile(upper_pct)
            if lo < hi:
                result.loc[idx, col] = result.loc[idx, col].clip(lo, hi)
    return result


def winsorize_valuation_data(valuation_data: pd.DataFrame,
                              lower_pct: float = 0.01,
                              upper_pct: float = 0.99) -> pd.DataFrame:
    """
    预处理日频估值数据 (长表格式)。

    valuation_data: columns=['code', 'trade_date', turnover, pe_ttm, pb, ...]
    对每日截面做 winsorize (groupby.transform 向量化, 避免逐日 mask 循环)

    返回: 新 DataFrame (索引/列结构不变, 仅数值列被 clip)
    """
    if valuation_data is None or valuation_data.empty:
        return valuation_data

    result = valuation_data.copy()
    value_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    if 'trade_date' not in result.columns:
        # 无日期列 → 全局 winsorize
        for col in value_cols:
            s = result[col].dropna()
            if len(s) < 10:
                continue
            lo = s.quantile(lower_pct)
            hi = s.quantile(upper_pct)
            if lo < hi:
                result[col] = result[col].clip(lo, hi)
        return result

    # 向量化: 每日截面分位数 → clip
    g = result.groupby('trade_date')
    for col in value_cols:
        lo = g[col].transform('quantile', lower_pct)
        hi = g[col].transform('quantile', upper_pct)
        result[col] = result[col].clip(lo, hi)

    return result
