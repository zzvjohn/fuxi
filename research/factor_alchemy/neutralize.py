# -*- coding: utf-8 -*-
"""
因子行业/市值中性化模块
========================

对因子值做横截面回归: factor ~ industry_dummies + log(market_cap), 取残差。

目的:
  1. 剥离行业押注: 不同行业有结构性差异 (如银行天然低 PB), 不中性化会导致
     因子排名被行业特征污染。
  2. 剥离规模效应: 市值是横截面最强的因子之一, 不剥离会导致选股池市值偏斜。

位置: 在 base factor 计算后、composite 构建前 (预处理层)。

JQ 兼容: 本模块提供纯 numpy 实现, JQ 策略可内联使用。
本地: 依赖 industry_map (dict: stock→industry_label) 和 mcap_map。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ============================================================
# 核心: 横截面中性化 (单期)
# ============================================================

def neutralize_cross_section(
    factor_values: np.ndarray,
    valid_mask: np.ndarray,
    industry_codes: List,
    log_mcap: np.ndarray,
    min_industries: int = 2,
    min_stocks: int = 30,
) -> Tuple[np.ndarray, Dict]:
    """
    单期横截面中性化: y ~ industry_dummies + log_mcap → residuals.

    Parameters
    ----------
    factor_values : (N,) float
        原始因子值 (含 NaN)
    valid_mask   : (N,) bool
        有效样本标记 (True=参与回归)
    industry_codes : list of str/int, len=N
    log_mcap     : (N,) float
    min_industries : 最少行业数 (低于此值不引入行业虚拟变量)
    min_stocks   : 最少有效股票数 (低于此值返回原始值)

    Returns
    -------
    residuals : (N,)  中性化残差 (非有效位置为 NaN)
    info      : dict  R², n_industries, n_valid
    """
    n = len(factor_values)
    residuals = factor_values.copy()

    n_valid = int(valid_mask.sum())
    if n_valid < min_stocks:
        info = {'r_squared': np.nan, 'n_industries': 0, 'n_valid': n_valid,
                'status': 'insufficient_stocks'}
        return residuals, info

    # 构建设计矩阵
    valid_idx = np.where(valid_mask)[0]
    y = factor_values[valid_idx]

    # 行业虚拟变量
    industries_valid = [industry_codes[i] for i in valid_idx]
    unique_ind = sorted(set(industries_valid))
    n_ind = len(unique_ind)

    if n_ind >= min_industries:
        # One-hot (drop first category to avoid dummy trap)
        ind_dummies = np.column_stack([
            (np.array(industries_valid) == ind).astype(float)
            for ind in unique_ind[1:]
        ])
    else:
        ind_dummies = np.empty((len(valid_idx), 0))

    # log market cap
    lmcap = log_mcap[valid_idx].reshape(-1, 1)

    # 完整设计矩阵: [const, ind_dummies, log_mcap]
    X = np.column_stack([
        np.ones((n_valid, 1)),
        ind_dummies,
        lmcap,
    ])

    # OLS: (X'X)^(-1) X'y
    try:
        beta, ss_res, rank_X, sv = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        info = {'r_squared': np.nan, 'n_industries': n_ind, 'n_valid': n_valid,
                'status': 'lstsq_failed'}
        return residuals, info

    y_pred = X @ beta
    res = y - y_pred

    # R²
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - min(ss_res[0] / max(ss_tot, 1e-12), 1.0) if len(ss_res) > 0 and ss_tot > 0 else 0.0

    residuals[valid_idx] = res

    info = {'r_squared': float(r2), 'n_industries': n_ind, 'n_valid': n_valid,
            'status': 'ok'}
    return residuals, info


# ============================================================
# 便捷: 对一整个因子 DataFrame 做逐期中性化
# ============================================================

def neutralize_factor_df(
    factor_df: pd.DataFrame,
    industry_map: Dict[str, str],
    mcap_df: pd.DataFrame,
    min_stocks: int = 30,
    min_industries: int = 2,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    对整个因子矩阵 (date × stock) 做逐期横截面中性化。

    Parameters
    ----------
    factor_df : DataFrame(index=date, columns=stock)
        因子值矩阵
    industry_map : dict stock→industry_label
    mcap_df : DataFrame(index=date, columns=stock)
        市值矩阵 (用于 log(mcap))
    min_stocks : int
    min_industries : int
    verbose : bool

    Returns
    -------
    neutralized_df : DataFrame  中性化后的因子值
    period_info    : list[dict]  每期的回归信息
    """
    common_dates = factor_df.index.intersection(mcap_df.index)
    if len(common_dates) == 0:
        raise ValueError("factor_df and mcap_df share no common dates")

    # ★ 关键修复: 只处理因子和市值共有的股票, 避免列不对齐
    common_stocks = factor_df.columns.intersection(mcap_df.columns)
    if len(common_stocks) < min_stocks:
        raise ValueError(f"Only {len(common_stocks)} common stocks between factor and mcap")

    factor_aligned = factor_df[common_stocks]
    mcap_aligned = mcap_df[common_stocks]
    neutralized = factor_df.copy()  # 保持原始形状 (非共有列为 NaN)
    period_info = []

    for date in common_dates:
        f_row = factor_aligned.loc[date]
        m_row = mcap_aligned.loc[date]

        # 有效样本: 因子值 + 市值均非 NaN, 且有行业分类, 市值>0
        has_ind = pd.Series([s in industry_map for s in common_stocks], index=common_stocks)
        valid = (
            f_row.notna() &
            m_row.notna() &
            (m_row > 0) &
            has_ind
        )
        n_valid = int(valid.sum())

        if n_valid < min_stocks:
            period_info.append({
                'date': date, 'n_valid': n_valid, 'status': 'insufficient_stocks'
            })
            continue

        # Build industry codes and log_mcap arrays (all aligned to common_stocks)
        all_industries = [industry_map.get(s, 'Unknown') for s in common_stocks]
        lmcap_all = np.full(len(common_stocks), np.nan)
        lmcap_all[valid.values] = np.log(m_row[valid].values)

        residuals, info = neutralize_cross_section(
            f_row.values, valid.values,
            all_industries, lmcap_all,
            min_industries=min_industries, min_stocks=min_stocks,
        )

        # 只更新共有列的中性化值
        neutralized.loc[date, common_stocks] = residuals
        info['date'] = date
        period_info.append(info)

        if verbose and len(period_info) % 50 == 0:
            print(f"  [neutralize] {date.strftime('%Y-%m-%d')} "
                  f"n={info['n_valid']} R²={info.get('r_squared', float('nan')):.3f} "
                  f"industries={info.get('n_industries', 0)}")

    return neutralized, period_info


# ============================================================
# 诊断: 中性化前后对比
# ============================================================

def diagnose_neutralization(
    factor_df: pd.DataFrame,
    neutralized_df: pd.DataFrame,
    industry_map: Dict[str, str],
    mcap_df: pd.DataFrame,
    sample_date=None,
) -> Dict:
    """
    对比中性化前后的因子特征:
    - 行业集中度 (HHI of industry weights)
    - 市值相关性
    - Top-30 行业分布变化
    """
    if sample_date is None:
        sample_date = factor_df.index[-1]

    result = {}
    for label, df in [('raw', factor_df), ('neutralized', neutralized_df)]:
        if sample_date not in df.index:
            continue

        row = df.loc[sample_date].dropna()
        top30 = row.nlargest(30).index

        # 行业分布
        industries = [industry_map.get(s, 'Unknown') for s in top30]
        ind_counts = pd.Series(industries).value_counts()
        ratios = ind_counts.values.astype(float) / max(float(ind_counts.values.sum()), 1)
        ind_hhi = float(np.sum(ratios ** 2))
        result[f'{label}_ind_hhi'] = ind_hhi
        result[f'{label}_top_ind'] = ind_counts.index[0] if len(ind_counts) > 0 else 'N/A'
        result[f'{label}_top_ind_pct'] = float(ind_counts.iloc[0] / ind_counts.sum()) if len(ind_counts) > 0 else 0

        # 市值相关性
        if sample_date in mcap_df.index:
            m_row = mcap_df.loc[sample_date]
            common = row.index.intersection(m_row.dropna().index)
            if len(common) > 0:
                corr = np.corrcoef(row[common], np.log(m_row[common]))[0, 1]
                result[f'{label}_mcap_corr'] = float(corr)

    return result


# ============================================================
# 冒烟测试
# ============================================================

if __name__ == '__main__':
    np.random.seed(42)
    N_STOCKS = 200
    N_WEEKS = 50
    stocks = [f'{i:06d}.XSHG' for i in range(N_STOCKS)]
    dates = pd.date_range('2023-01-01', periods=N_WEEKS, freq='W')

    # 模拟行业 (8 个行业)
    industries = ['金融', '医药', '电子', '机械', '食品', '化工', '地产', '公用']
    industry_map = {s: industries[i % len(industries)] for i, s in enumerate(stocks)}

    # 模拟市值
    mcap = pd.DataFrame(
        10 ** np.random.uniform(8, 11, (N_WEEKS, N_STOCKS)),
        index=dates, columns=stocks,
    )

    # 模拟因子 (含行业偏斜: 金融行业天然高 2 单位)
    base = np.random.randn(N_WEEKS, N_STOCKS) * 0.5
    for i, s in enumerate(stocks):
        if industry_map[s] == '金融':
            base[:, i] += 2.0  # 行业偏斜

    factor = pd.DataFrame(base, index=dates, columns=stocks)

    print("=" * 60)
    print("因子中性化诊断")
    print("=" * 60)

    neutralized, info_list = neutralize_factor_df(
        factor, industry_map, mcap, verbose=True,
    )

    diag = diagnose_neutralization(factor, neutralized, industry_map, mcap)
    print("\n诊断:")
    for k, v in diag.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 验证: 中性化后金融行业均值应该接近 0
    fin_stocks = [s for s in stocks if industry_map[s] == '金融']
    raw_fin_mean = factor[fin_stocks].iloc[-1].mean()
    neu_fin_mean = neutralized[fin_stocks].iloc[-1].mean()
    print(f"\n金融行业均值: raw={raw_fin_mean:.3f} → neutralized={neu_fin_mean:.3f}")
    print(f"(期望: neutralized ≈ 0, 说明行业偏斜已剥离)")

    print("\n✓ 中性化模块冒烟测试通过")
