"""
Ψ 正交化算子 — FactorMiner 核心模块

对新因子做横截面正交化：在每个时间截面上，将新因子对已有因子池做
Ridge 回归，取残差为"正交化因子"。

原理 (FactorMiner Eq.3):
    ψ(f_new) = f_new - X(X'X + λI)^(-1)X'f_new

其中 X 为已有因子矩阵 (stocks × k), λ 为 Ridge 正则化参数。

这确保新因子添加的是"已有因子无法解释的增量信息",
直接解决 Phantom 共线性问题。

用法:
    from research.factor_alchemy.psi_orthogonalizer import psi_orthogonalize

    result = psi_orthogonalize(
        new_factor=fv_df,           # DataFrame: date × stock
        existing_pool={...},         # {name: DataFrame}
    )
    print(f"R²={result.r_squared:.3f}, 独立性={result.independence:.3f}")
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings

# ── 常数 ──────────────────────────────────────────────────
RIDGE_LAMBDA = 1e-3           # Ridge 正则化参数
MIN_STOCKS_CROSS_SECTION = 30  # 截面最少股票数
CORR_THRESHOLD = 0.25          # |corr| > 此值才纳入正交化 (省计算)
MIN_R_SQUARED_WARN = 0.80      # R² > 此值: 因子几乎全是冗余
MIN_R_SQUARED_REJECT = 0.95    # R² > 此值: 建议直接拒绝


@dataclass
class PsiResult:
    """Ψ 正交化结果"""
    factor_name: str
    raw_values: pd.DataFrame         # 原始因子值
    orthogonalized_values: pd.DataFrame  # 正交化后因子值
    r_squared: float = 0.0           # 已有因子池对新因子的解释度 R²
    independence: float = 0.0        # 独立性 = 1 - R² (高=好)
    n_orthogonalized_against: int = 0  # 实际正交化使用的因子数
    orthogonalized_factor_names: List[str] = field(default_factory=list)
    per_date_r_squared: pd.Series = field(default_factory=pd.Series)
    rejection_recommended: bool = False
    warnings: List[str] = field(default_factory=list)


def _select_relevant_factors(
    new_factor_values: np.ndarray,   # shape: (n_stocks,)
    existing_pool: Dict[str, pd.DataFrame],
    date_idx: int,
) -> Tuple[List[str], np.ndarray]:
    """
    筛选与当前截面上新因子 |corr| > CORR_THRESHOLD 的已有因子。

    Returns:
        names: 筛选出的因子名称列表
        X: (n_stocks, k) 的因子值矩阵
    """
    selected_names = []
    selected_values = []

    for name, fv_df in existing_pool.items():
        if fv_df.shape[0] <= date_idx:
            continue
        row = fv_df.iloc[date_idx].values
        if not isinstance(row, np.ndarray):
            row = np.array(row)
        # 剔除 NaN
        valid = ~np.isnan(row) & ~np.isnan(new_factor_values)
        if valid.sum() < MIN_STOCKS_CROSS_SECTION:
            continue
        corr = np.corrcoef(row[valid], new_factor_values[valid])[0, 1]
        if abs(corr) >= CORR_THRESHOLD:
            selected_names.append(name)
            selected_values.append(row)

    if not selected_names:
        return [], np.array([]).reshape(len(new_factor_values), 0)

    X = np.column_stack(selected_values)
    return selected_names, X


def _ridge_residual(
    y: np.ndarray,   # (n,)
    X: np.ndarray,   # (n, k)
    lam: float = RIDGE_LAMBDA,
) -> Tuple[np.ndarray, float]:
    """
    对 y ~ X 做 Ridge 回归，返回残差和 R²。

    Ridge: β = (X'X + λI)^(-1) X'y
    """
    n, k = X.shape
    if k == 0:
        return y, 0.0

    # 仅用完整观测
    valid = ~np.isnan(y)
    for j in range(k):
        valid = valid & ~np.isnan(X[:, j])

    if valid.sum() < MIN_STOCKS_CROSS_SECTION:
        return y, 0.0

    y_v = y[valid]
    X_v = X[valid]

    # 标准化 (避免量纲影响 Ridge)
    y_mean, y_std = y_v.mean(), y_v.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_s = (y_v - y_mean) / y_std

    X_mean = X_v.mean(axis=0)
    X_std = X_v.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_s = (X_v - X_mean) / X_std

    # Ridge: β = (X'X + λI)^(-1) X'y
    XtX = X_s.T @ X_s
    XtX_reg = XtX + lam * np.eye(k)
    Xty = X_s.T @ y_s

    try:
        beta = np.linalg.solve(XtX_reg, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtX_reg, Xty, rcond=None)[0]

    # 残差
    y_pred_s = X_s @ beta
    residual_s = y_s - y_pred_s

    # R²
    ss_res = np.sum(residual_s ** 2)
    ss_tot = np.sum(y_s ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    # 残差还原到原始尺度
    residual = np.full_like(y, np.nan)
    residual[valid] = residual_s * y_std + y_mean  # 保留均值 (维持截面排序)

    return residual, max(0.0, min(r2, 1.0))


def psi_orthogonalize(
    factor_name: str,
    factor_values: pd.DataFrame,
    existing_pool: Dict[str, pd.DataFrame],
    ridge_lambda: float = RIDGE_LAMBDA,
    verbose: bool = False,
) -> PsiResult:
    """
    对因子做 Ψ 正交化。

    Parameters
    ----------
    factor_name: 因子名称
    factor_values: 待正交化的因子值, index=date, columns=stock_code
    existing_pool: 已有因子池 {name: DataFrame(index=date, columns=stock_code)}
    ridge_lambda: Ridge 正则化参数
    verbose: 输出详细日志

    Returns
    -------
    PsiResult: 含正交化后因子值、R²、独立性等
    """
    if not existing_pool:
        result = PsiResult(
            factor_name=factor_name,
            raw_values=factor_values.copy(),
            orthogonalized_values=factor_values.copy(),
            r_squared=0.0,
            independence=1.0,
            n_orthogonalized_against=0,
        )
        if verbose:
            print(f"  [Ψ] {factor_name}: 无已有因子池，跳过正交化")
        return result

    # 对齐已有因子池与 factor_values 的索引
    aligned_pool = {}
    for name, fv in existing_pool.items():
        if isinstance(fv, pd.DataFrame):
            # 对齐日期
            common_dates = factor_values.index.intersection(fv.index)
            if len(common_dates) < 10:
                continue
            # 选取公共日期和列 (对齐最快)
            fv_aligned = fv.loc[common_dates, factor_values.columns.intersection(fv.columns)]
            if fv_aligned.shape[1] >= MIN_STOCKS_CROSS_SECTION:
                aligned_pool[name] = fv

    if not aligned_pool:
        result = PsiResult(
            factor_name=factor_name,
            raw_values=factor_values.copy(),
            orthogonalized_values=factor_values.copy(),
            r_squared=0.0,
            independence=1.0,
            n_orthogonalized_against=0,
        )
        if verbose:
            print(f"  [Ψ] {factor_name}: 无可用对齐因子，跳过正交化")
        return result

    raw = factor_values.copy()
    ortho = factor_values.copy().astype(float)

    r2_series = []
    all_used_factors = set()
    n_dates = len(raw)

    for date_idx in range(n_dates):
        y_raw = raw.iloc[date_idx].values.astype(float)

        # 筛选相关因子
        sel_names, X = _select_relevant_factors(y_raw, aligned_pool, date_idx)

        all_used_factors.update(sel_names)

        # Ridge 正交化
        residual, r2 = _ridge_residual(y_raw, X, ridge_lambda)
        r2_series.append(r2)

        # 只有 nan 部分保持原值，非 nan 部分替换为残差
        if not np.all(np.isnan(residual)):
            ortho.iloc[date_idx] = residual

    # 汇总统计
    r2_avg = float(np.mean(r2_series)) if r2_series else 0.0
    independence = 1.0 - r2_avg

    # 警告
    warnings_list = []
    if r2_avg >= MIN_R_SQUARED_REJECT:
        warnings_list.append(
            f"R²={r2_avg:.3f} ≥ {MIN_R_SQUARED_REJECT}, "
            f"因子几乎完全被已有因子池解释, 建议拒绝"
        )
    elif r2_avg >= MIN_R_SQUARED_WARN:
        warnings_list.append(
            f"R²={r2_avg:.3f} ≥ {MIN_R_SQUARED_WARN}, "
            f"因子冗余度较高"
        )

    result = PsiResult(
        factor_name=factor_name,
        raw_values=raw,
        orthogonalized_values=ortho,
        r_squared=r2_avg,
        independence=independence,
        n_orthogonalized_against=len(all_used_factors),
        orthogonalized_factor_names=sorted(all_used_factors),
        per_date_r_squared=pd.Series(r2_series, index=raw.index[:len(r2_series)]),
        rejection_recommended=r2_avg >= MIN_R_SQUARED_REJECT,
        warnings=warnings_list,
    )

    if verbose:
        status = "✓" if not result.rejection_recommended else "✗ 建议拒绝"
        print(
            f"  [Ψ] {factor_name}: R²={r2_avg:.3f} "
            f"independence={independence:.3f} "
            f"vs {len(all_used_factors)} factors "
            f"{status}"
        )
        for w in warnings_list:
            print(f"       ⚠ {w}")

    return result


def batch_orthogonalize(
    new_factors: Dict[str, pd.DataFrame],
    existing_pool: Dict[str, pd.DataFrame],
    ridge_lambda: float = RIDGE_LAMBDA,
    verbose: bool = True,
) -> Dict[str, PsiResult]:
    """
    批量 Ψ 正交化多个因子。

    按独立性降序依次正交化：最先正交化的因子保留最多原始信息。
    后续因子正交化时，前面已正交化的因子也会加入 existing_pool。

    Parameters
    ----------
    new_factors: {name: factor_values} 待处理的因子
    existing_pool: 已有因子池
    ridge_lambda: Ridge 参数
    verbose: 详细输出

    Returns
    -------
    {name: PsiResult}
    """
    pool = dict(existing_pool)
    results = {}

    # 按 (待注入因子数) 排序 (优先处理重要的)
    sorted_names = list(new_factors.keys())

    for name in sorted_names:
        fv = new_factors[name]
        result = psi_orthogonalize(
            factor_name=name,
            factor_values=fv,
            existing_pool=pool,
            ridge_lambda=ridge_lambda,
            verbose=verbose,
        )
        results[name] = result
        # 正交化后的因子加入池 (为后续因子提供正交化基准)
        pool[name] = result.orthogonalized_values

    return results


# ── 便捷接口 (供 factor_injector.py 调用) ─────────────────

def quick_orthogonalize(
    factor_name: str,
    factor_values: pd.DataFrame,
    pool_csv_path: str,
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    verbose: bool = True,
) -> PsiResult:
    """
    快速 Ψ 正交化：从因子池 CSV 加载已有因子池, 然后正交化。
    这是 factor_injector.py 直接调用的接口。

    Example:
        from psi_orthogonalizer import quick_orthogonalize
        psi = quick_orthogonalize("my_factor", fv, "data/passed_factor_pool.csv", ...)
        factor_values = psi.orthogonalized_values  # 用这个而不是原始值
    """
    from research.factor_alchemy.fri import build_existing_pool_from_csv

    pool = build_existing_pool_from_csv(
        Path(pool_csv_path), close_df, open_df, high_df, low_df, vol_df
    )
    return psi_orthogonalize(
        factor_name=factor_name,
        factor_values=factor_values,
        existing_pool=pool,
        verbose=verbose,
    )


# ── 测试 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # 构建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=50, freq='B')
    stocks = [f'S{i:04d}' for i in range(100)]

    # 已有因子：纯随机噪声
    existing = {
        'momentum': pd.DataFrame(np.random.randn(50, 100), index=dates, columns=stocks),
        'volatility': pd.DataFrame(np.random.randn(50, 100), index=dates, columns=stocks),
    }

    # 新因子：momentum * 0.8 + noise * 0.2 (故意与 momentum 高相关)
    new_factor = existing['momentum'] * 0.8 + pd.DataFrame(
        np.random.randn(50, 100) * 0.2, index=dates, columns=stocks
    )

    print("=" * 60)
    print("Ψ 正交化测试")
    print("=" * 60)

    result = psi_orthogonalize(
        factor_name="momentum_clone",
        factor_values=new_factor,
        existing_pool=existing,
        verbose=True,
    )

    print(f"\n结果:")
    print(f"  R² (冗余度): {result.r_squared:.4f}")
    print(f"  独立性:      {result.independence:.4f}")
    print(f"  正交化对抗:  {result.n_orthogonalized_against} 个因子")
    print(f"  建议拒绝:    {result.rejection_recommended}")

    # 验证：正交化后的因子应与 momentum 不相关
    ortho_val = result.orthogonalized_values.iloc[-1].dropna()
    mom_val = existing['momentum'].iloc[-1].loc[ortho_val.index].dropna()
    post_corr = np.corrcoef(ortho_val, mom_val)[0, 1]
    print(f"  原始 corr(momentum): {np.corrcoef(new_factor.iloc[-1].dropna(), mom_val)[0,1]:.4f}")
    print(f"  正交化后 corr(momentum): {post_corr:.4f}")
    print(f"  ✓ 测试{'通过' if abs(post_corr) < 0.1 else '失败'}")
