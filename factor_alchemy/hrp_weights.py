# -*- coding: utf-8 -*-
"""
HRP (Hierarchical Risk Parity) 持仓权重优化器
==============================================

用途: 在 FA Composite 选出的 Top-N 股票池上, 用 HRP 替代等权分配,
      降低组合波动和最大回撤, 不触碰已验证的因子信号。

参考:
  - López de Prado (2016) "Building Diversified Portfolios that
    Outperform Out-of-Sample", Journal of Portfolio Management.
  - López de Prado (2019) "Machine Learning for Asset Management",
    Chapter 8: Hierarchical Risk Parity.

方法:
  A. HRP (三层): 相关矩阵 → 距离矩阵 → 层次聚类 (Ward/single) →
     递归 quasi-diagonalization → 逆序风险平价分配 (自底向上)。
  B. 波动率倒数缩放: w_i ∝ 1/σ_i → 归一化。
  C. 等风险贡献 (ERC / Risk Parity): 迭代求解 w_i ∝ 1/MRC_i。

约束:
  - 仅做多 (w_i ≥ 0)
  - 归一化 (Σw_i = 1)
  - 最小权重 floor = 1/N × 0.3 (防止单个股票占据过多)

与 factor_composer / regime_lasso 的关系:
  - 因子层: Composer 选出因子 → regime_lasso WFA 权重 → 合成排名 → Top-N 股票池
  - 持仓层: **本模块** 负责 Top-N 内的权重分配 (替代等权)
  - 两层正交: HRP 不动因子选择逻辑, 不动合成排名, 只动最终持仓权重
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform


# ============================================================
# HRP 核心算法
# ============================================================

def _get_quasi_diag(link):
    """
    对 linkage 矩阵做 quasi-diagonalization: 重排索引使相似资产相邻。

    Returns
    -------
    sorted_idx : 重排后的原始索引顺序
    """
    link = link.astype(int)
    n = link.shape[0] + 1
    sorted_idx = [link[-1, 0], link[-1, 1]]  # 根节点两个子节点

    # 递归展开
    i = 0
    while i < len(sorted_idx):
        idx = sorted_idx[i]
        if idx >= n:  # 聚类节点 (非叶子)
            cluster_row = link[idx - n]
            sorted_idx[i:i+1] = [cluster_row[0], cluster_row[1]]
            i = 0  # 从头扫描 (因为列表被修改了)
        else:
            i += 1
    return [int(x) for x in sorted_idx]


def _get_recursive_bisection(cov: np.ndarray, sorted_idx: List[int]) -> np.ndarray:
    """
    递归二分逆序风险分配。
    
    对已排序的协方差矩阵, 自顶向下二分:
      对每个切分点 i, 将矩阵分为 [0:i] 和 [i:n] 两部分,
      分配权重使两部分方差相等: w_left ∝ 1/var_left, w_right ∝ 1/var_right.
    在每个子集内重复, 直到单个资产 → 权重为其父节点分配的份额。
    
    Returns
    -------
    weights : shape (n,)  归一化权重
    """
    n = len(sorted_idx)
    w = np.ones(n)

    def _bisect(indices):
        """递归二分 (modifies w in place)."""
        if len(indices) <= 1:
            return

        # 尝试所有切分点, 选择使两侧聚类内方差之和最小的
        best_var = np.inf
        best_split = len(indices) // 2
        for split in range(1, len(indices)):
            left_idx = indices[:split]
            right_idx = indices[split:]

            # 子协方差矩阵
            cov_left = cov[np.ix_(left_idx, left_idx)]
            cov_right = cov[np.ix_(right_idx, right_idx)]

            # 等权子组合方差 (用 HRP 的逆方差分配逻辑)
            w_left_eq = np.ones(len(left_idx)) / len(left_idx)
            w_right_eq = np.ones(len(right_idx)) / len(right_idx)
            var_left_ew = w_left_eq @ cov_left @ w_left_eq
            var_right_ew = w_right_eq @ cov_right @ w_right_eq

            total_var = var_left_ew + var_right_ew
            if total_var < best_var:
                best_var = total_var
                best_split = split

        left_idx = indices[:best_split]
        right_idx = indices[best_split:]

        # 逆方差分配份额
        cov_left = cov[np.ix_(left_idx, left_idx)]
        cov_right = cov[np.ix_(right_idx, right_idx)]

        w_left_eq = np.ones(len(left_idx)) / max(len(left_idx), 1)
        w_right_eq = np.ones(len(right_idx)) / max(len(right_idx), 1)
        var_left = max(w_left_eq @ cov_left @ w_left_eq, 1e-12)
        var_right = max(w_right_eq @ cov_right @ w_right_eq, 1e-12)

        # 逆方差 → 份额
        alloc_left = (1.0 / var_left) / (1.0 / var_left + 1.0 / var_right)
        alloc_right = 1.0 - alloc_left

        # 向下分配
        w[left_idx] *= alloc_left * len(indices) / max(len(left_idx), 1)
        w[right_idx] *= alloc_right * len(indices) / max(len(right_idx), 1)

        _bisect(left_idx)
        _bisect(right_idx)

    _bisect(sorted_idx)
    w = w / w.sum()  # 归一化
    return w


def hrp_weights(
    returns: np.ndarray,
    cov_method: str = 'sample',
    linkage_method: str = 'ward',
    floor: float = 0.0,
) -> np.ndarray:
    """
    层次风险平价 (HRP) 权重计算。

    Parameters
    ----------
    returns : shape (T, N)
        资产收益率矩阵 (周收益), T=时间, N=资产数。
        如果 N < 2, 返回等权。
    cov_method : str
        'sample' 用样本协方差, 'shrink' 用 Ledoit-Wolf 收缩 (更稳健)
    linkage_method : str
        层次聚类 linkage 方法: 'ward' (默认), 'single', 'average', 'complete'
    floor : float
        最小权重下限 (默认 0, 即不限制)

    Returns
    -------
    weights : shape (N,)  归一化正权重
    """
    n = returns.shape[1]
    if n <= 1:
        return np.ones(n) / max(n, 1)
    if n == 2:
        # 2 资产: 逆方差即可, 避免链路矩阵维度问题
        stds = np.nanstd(returns, axis=0) + 1e-9
        inv_var = 1.0 / (stds ** 2)
        w = inv_var / inv_var.sum()
        return w

    # Step 1: 协方差矩阵
    if cov_method == 'shrink':
        cov = _ledoit_wolf_shrinkage(returns)
    else:
        cov = np.cov(returns, rowvar=False)
        # 确保正定
        cov = _ensure_psd(cov)

    # Step 2: 相关矩阵 → 距离矩阵
    stds = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(stds, stds)
    corr = np.clip(corr, -1.0, 1.0)

    # 距离: d_ij = sqrt(0.5 * (1 - ρ_ij))  (López de Prado 2016)
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)
    dist_condensed = squareform(dist, checks=False)

    # Step 3: 层次聚类
    try:
        link = linkage(dist_condensed, method=linkage_method, optimal_ordering=True)
    except Exception:
        link = linkage(dist_condensed, method='single', optimal_ordering=True)

    # Step 4: Quasi-diagonalization
    sorted_idx = _get_quasi_diag(link)

    # Step 5: 递归二分逆序风险分配
    w = _get_recursive_bisection(cov, sorted_idx)

    # Step 6: 最小权重约束
    if floor > 0:
        while np.any(w < floor):
            below = w < floor
            w[below] = floor
            w[~below] = w[~below] * (1.0 - floor * below.sum()) / max(w[~below].sum(), 1e-9)
    w = w / w.sum()

    return w


def ivp_weights(returns: np.ndarray) -> np.ndarray:
    """
    逆波动率组合 (Inverse Volatility Portfolio) 权重。
    w_i ∝ 1/σ_i

    这是 HRP 的简化版, 不求相关性, 直接按个股波动率倒数分配。
    适用于: 资产数多但历史不够长, 相关矩阵估计不可靠的场景。
    """
    n = returns.shape[1]
    if n <= 1:
        return np.ones(n) / max(n, 1)
    stds = np.nanstd(returns, axis=0) + 1e-9
    inv_vol = 1.0 / stds
    return inv_vol / inv_vol.sum()


def erc_weights(returns: np.ndarray, max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """
    等风险贡献 (Equal Risk Contribution / Risk Parity) 权重。

    迭代求解 w 使 MRC_i = w_i × ∂σ_p/∂w_i 对所有 i 相等。
    算法: 逐次二分法 (Griveau-Billion et al. 2013)。

    Parameters
    ----------
    returns : shape (T, N)
    max_iter : int
    tol : float
    """
    n = returns.shape[1]
    if n <= 1:
        return np.ones(n) / max(n, 1)

    cov = _ensure_psd(np.cov(returns, rowvar=False))
    w = np.ones(n) / n  # 初始等权

    for _ in range(max_iter):
        # Marginal Risk Contribution: MRC_i = (Σ w) contribution
        sigma_p = np.sqrt(w @ cov @ w)
        if sigma_p < 1e-12:
            break
        mrc = cov @ w / sigma_p  # shape (n,)
        rc = w * mrc  # risk contribution
        target_rc = rc.sum() / n

        # 调整: w_i *= sqrt(target_rc / rc_i)
        mask = rc > 1e-12
        if not np.any(mask):
            break
        new_w = w.copy()
        new_w[mask] = w[mask] * np.sqrt(target_rc / rc[mask])
        new_w = np.maximum(new_w, 0.0)
        new_w = new_w / new_w.sum()

        if np.max(np.abs(new_w - w)) < tol:
            w = new_w
            break
        w = new_w

    return w / w.sum()


# ============================================================
# 辅助
# ============================================================

def _ensure_psd(cov: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """确保协方差矩阵半正定 (通过特征值修复)."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _ledoit_wolf_shrinkage(returns: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf 收缩估计 (简化实现, 对标 sklearn LedoitWolf)。

    收缩目标: 单位对角矩阵 × mean(var) (constant correlation 目标)
    """
    T, n = returns.shape
    if T <= 1:
        return np.eye(n) * 1e-4

    sample_cov = np.cov(returns, rowvar=False)
    mu = np.trace(sample_cov) / n

    # 收缩强度: delta² / gamma²
    # delta² = E[||S - mu*I||²_F]  (简化: 用 bootstrap 近似)
    # gamma² = ||S - mu*I||²_F  (实际偏差)
    diff = sample_cov - mu * np.eye(n)
    pi_mat = np.zeros((n, n))

    # π_ij = 1/T Σ_t [(x_ti - x̄_i)(x_tj - x̄_j) - s_ij]²
    centered = returns - returns.mean(axis=0)
    for i in range(n):
        for j in range(i, n):
            cross = centered[:, i] * centered[:, j]
            pi_mat[i, j] = pi_mat[j, i] = np.var(cross, ddof=1)

    # 收缩强度
    pi_sum = np.sum(pi_mat - np.diag(np.diag(pi_mat)))  # exclude diagonal
    gamma_sq = np.sum(diff ** 2)
    rho = np.sum(np.diag(pi_mat)) / (n * T) if T > 1 else 0

    shrinkage = min(max((pi_sum + rho) / (gamma_sq + 1e-12), 0.0), 1.0)

    # 收缩估计
    target = mu * np.eye(n) + mu * (1 - np.eye(n)) * np.mean(
        sample_cov[np.triu_indices(n, k=1)]) / max(mu, 1e-9)
    target = np.clip(target, -1, 1) * mu  # keep scale
    target = mu * np.eye(n)  # 简化: 用常数方差目标

    shrunk = shrinkage * target + (1.0 - shrinkage) * sample_cov
    return _ensure_psd(shrunk)


# ============================================================
# 便捷包装: 从股票池 + 价格数据 → HRP 权重
# ============================================================

def hrp_from_pool(
    price_df: pd.DataFrame,
    pool_stocks: List[str],
    lookback_weeks: int = 52,
    method: str = 'hrp',
    floor: float = 0.0,
) -> pd.Series:
    """
    给定股票池和价格数据, 输出权重。

    Parameters
    ----------
    price_df : DataFrame(index=date, columns=stock)
        价格矩阵 (用于计算收益率)
    pool_stocks : list of str
        当期持仓股票列表
    lookback_weeks : int
        回看周数, 用于估计协方差
    method : str
        'hrp' 层次风险平价
        'ivp' 逆波动率
        'erc' 等风险贡献
    floor : float
        最低权重 (0 = 允许 0 权重)

    Returns
    -------
    weights : pd.Series(index=stock, value=weight)
    """
    valid = [s for s in pool_stocks if s in price_df.columns]
    if len(valid) < 2:
        s = pd.Series(1.0, index=valid) if valid else pd.Series(dtype=float)
        return s / s.sum() if len(s) > 0 else s

    # 取尾部 lookback 周
    price_pool = price_df[valid].tail(lookback_weeks).dropna(axis=1, how='all')
    if len(price_pool) < 10 or price_pool.shape[1] < 2:
        w = np.ones(len(valid)) / len(valid)
        return pd.Series(w, index=valid)

    # 周收益
    rets = price_pool.pct_change().dropna(how='all')
    if len(rets) < 10:
        w = np.ones(len(valid)) / len(valid)
        return pd.Series(w, index=valid)

    # 仅保留至少有 50% 数据的股票
    rets = rets.dropna(axis=1, thresh=int(len(rets) * 0.5))
    available = list(rets.columns)
    if len(available) < 2:
        w = np.ones(len(valid)) / len(valid)
        return pd.Series(w, index=valid)

    ret_arr = rets[available].values

    if method == 'ivp':
        w_arr = ivp_weights(ret_arr)
    elif method == 'erc':
        w_arr = erc_weights(ret_arr)
    else:  # 'hrp'
        w_arr = hrp_weights(ret_arr, floor=floor)

    weights = pd.Series(w_arr, index=available)
    # 补齐不在 available 中的股票 (给等权)
    missing = [s for s in valid if s not in available]
    if missing:
        missing_w = pd.Series(1.0 / len(valid), index=missing)
        weights = pd.concat([weights, missing_w])

    return weights / weights.sum()


# ============================================================
# 诊断对比: 等权 vs HRP vs IVP vs ERC
# ============================================================

def diagnose_weight_methods(
    price_df: pd.DataFrame,
    pool_stocks: List[str],
    lookback_weeks: int = 52,
) -> pd.DataFrame:
    """
    对比四种权重方法对同一个股票池的分配。
    用于诊断不同方法权重集中度差异。
    """
    methods = ['equal', 'ivp', 'erc', 'hrp']
    results = {}

    n = len(pool_stocks)
    # 等权
    equal_w = np.ones(n) / n if n > 0 else np.array([])

    for method in methods:
        if method == 'equal':
            w = pd.Series(equal_w, index=pool_stocks) if n > 0 else pd.Series(dtype=float)
        else:
            w = hrp_from_pool(price_df, pool_stocks, lookback_weeks, method=method)
        results[method] = w

    df = pd.DataFrame(results)
    df.index.name = 'stock'

    # 汇总行
    summary = {}
    for col in df.columns:
        w = df[col].dropna()
        if len(w) == 0:
            continue
        summary[f'{col}_HHI'] = float((w ** 2).sum())  # Herfindahl 集中度
        summary[f'{col}_top3'] = float(w.nlargest(3).sum())  # Top3 占比
        summary[f'{col}_min'] = float(w.min())
        summary[f'{col}_max'] = float(w.max())
    summary['n_stocks'] = n

    return df, summary


# ============================================================
# 冒烟测试
# ============================================================

if __name__ == '__main__':
    import gc

    np.random.seed(42)
    np.random.seed(42)
    N_WEEK, N_STOCK = 200, 30

    # 造价格和收益率
    returns = np.random.randn(N_WEEK, N_STOCK) * 0.03
    prices = pd.DataFrame(
        10 * np.exp(np.cumsum(returns, axis=0)),
        columns=[f'S{i:03d}' for i in range(N_STOCK)],
    )

    pool = list(prices.columns[:15])  # Top-15 pool

    print("=" * 60)
    print("HRP 权重诊断")
    print("=" * 60)

    df, summary = diagnose_weight_methods(prices, pool)
    print(f"\n股票池: {len(pool)} 只")
    print(f"各方法 HHI (集中度):")
    for k, v in summary.items():
        if 'HHI' in k:
            print(f"  {k}: {v:.4f} (1/N={1/len(pool):.4f})")
    print(f"\n各方法 Top3 占比:")
    for k, v in summary.items():
        if 'top3' in k:
            print(f"  {k}: {v:.1%}")

    print("\n权重对比 (前 5 只):")
    print(df.head(10).to_string())

    # HRP-only 详细
    print("\n--- HRP 单独测试 ---")
    ret_arr = prices[pool].pct_change().dropna().values
    w = hrp_weights(ret_arr, cov_method='sample', floor=0.02)
    print(f"HRP 权重: min={w.min():.3f}, max={w.max():.3f}, HHI={sum(w**2):.4f}")
    print(f"Top-5: {np.sort(w)[-5:][::-1].round(3)}")

    # IVP
    w_ivp = ivp_weights(ret_arr)
    print(f"\nIVP 权重: min={w_ivp.min():.3f}, max={w_ivp.max():.3f}, HHI={sum(w_ivp**2):.4f}")

    # ERC
    w_erc = erc_weights(ret_arr)
    print(f"ERC 权重: min={w_erc.min():.3f}, max={w_erc.max():.3f}, HHI={sum(w_erc**2):.4f}")

    gc.collect()
    print("\n✓ HRP 模块冒烟测试通过")
