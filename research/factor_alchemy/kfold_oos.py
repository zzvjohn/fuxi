# -*- coding: utf-8 -*-
"""
K-fold OOS 评估模块 — 替代 70:30 单切分
=========================================
基于 ODR (Out-of-Distribution Robustness) 论文方法论:
  - 5-fold 时间序列滚动切分 (保持时序, 不打乱)
  - 每 fold: train=t-4~t-1, test=t
  - 输出 avg ± std (ICIR/IC%/hit_rate) 而非单点估计
  - 稳定性评分: 跨 fold 的 CV(ICIR) 越低越好

与旧 70:30 对比:
  70:30 → 单次切分, 切点选择敏感, 无法评估时序稳定性
  K-fold → 多次切分, 报告均值+方差, 检测过拟合

用法:
  from kfold_oos import evaluate_kfold_oos, print_kfold_report
  results = evaluate_kfold_oos(factor_dfs, forward_returns, n_folds=5)
  print_kfold_report(results)
"""

from __future__ import division
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# K-fold OOS 核心评估
# ============================================================

def _compute_ic_metrics(factor_values: np.ndarray, forward_ret: np.ndarray) -> Dict[str, float]:
    """
    计算单期 IC 指标.
    
    Args:
        factor_values: (n_stocks,) 因子值
        forward_ret: (n_stocks,) 前向收益
    
    Returns:
        dict with IC, rank_IC, hit (IC>0 dummy)
    """
    mask = np.isfinite(factor_values) & np.isfinite(forward_ret)
    if mask.sum() < 10:
        return {"IC": np.nan, "rank_IC": np.nan, "hit": np.nan}

    fv = factor_values[mask]
    fr = forward_ret[mask]

    # Pearson IC
    std_fv = np.std(fv)
    std_fr = np.std(fr)
    if std_fv < 1e-12 or std_fr < 1e-12:
        ic = np.nan
    else:
        ic = np.corrcoef(fv, fr)[0, 1]

    # Spearman rank IC
    try:
        rank_ic, _ = spearmanr(fv, fr)
    except Exception:
        rank_ic = np.nan

    return {"IC": float(ic) if np.isfinite(ic) else np.nan,
            "rank_IC": float(rank_ic) if np.isfinite(rank_ic) else np.nan,
            "hit": 1.0 if ic > 0 else 0.0}


def _compute_ic_series(factor_df: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.DataFrame:
    """
    计算完整的时序 IC 序列 (每周一个 IC).
    
    Args:
        factor_df: (n_periods, n_stocks) 因子值
        forward_returns: (n_periods, n_stocks) 前向收益
    
    Returns:
        DataFrame with columns: IC, rank_IC, hit, indexed by period
    """
    common_idx = factor_df.index.intersection(forward_returns.index)
    common_cols = factor_df.columns.intersection(forward_returns.columns)

    if len(common_idx) == 0 or len(common_cols) == 0:
        return pd.DataFrame()

    f_df = factor_df.loc[common_idx, common_cols]
    fr_df = forward_returns.loc[common_idx, common_cols]

    records = []
    for t in common_idx:
        metrics = _compute_ic_metrics(f_df.loc[t].values, fr_df.loc[t].values)
        metrics["period"] = t
        records.append(metrics)

    result = pd.DataFrame(records).set_index("period")
    return result


def _icir_from_ic_series(ic_series: np.ndarray) -> float:
    """ICIR = mean(IC) / std(IC). 需要至少 10 个有效 IC."""
    valid = ic_series[np.isfinite(ic_series)]
    if len(valid) < 10:
        return np.nan
    std = np.std(valid)
    if std < 1e-12:
        return np.nan
    return float(np.mean(valid) / std)


def _hit_rate(ic_series: np.ndarray) -> float:
    """IC > 0 的比例."""
    valid = ic_series[np.isfinite(ic_series)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid > 0))


def _kfold_split_timeseries(dates: pd.DatetimeIndex, n_folds: int = 5) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    时间序列 K-fold 切分 (不打乱, 保持时序).
    
    每 fold: train = 前 4/5, test = 后 1/5 (逐步滚动).
    
    Returns:
        [(train_idx, test_idx), ...] 共 n_folds 组
    """
    n = len(dates)
    if n < n_folds * 2:
        raise ValueError(f"数据点不足: {n} < {n_folds * 2} (需要至少 n_folds×2)")

    fold_size = n // n_folds
    folds = []

    for k in range(n_folds):
        test_start = k * fold_size
        test_end = min((k + 1) * fold_size, n) if k < n_folds - 1 else n
        train_end = test_start

        train_idx = dates[:train_end]
        test_idx = dates[test_start:test_end]

        if len(train_idx) >= 10 and len(test_idx) >= 5:
            folds.append((train_idx, test_idx))

    return folds


def evaluate_kfold_oos(
    factor_dfs: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    n_folds: int = 5,
    min_train_weeks: int = 30,
    min_test_weeks: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    K-fold OOS 评估所有因子.
    
    Args:
        factor_dfs: {name: (n_periods, n_stocks) DataFrame}
        forward_returns: (n_periods, n_stocks) 前向收益
        n_folds: 折数 (默认 5)
        min_train_weeks: 训练集最少周数
        min_test_weeks: 测试集最少周数
        verbose: 是否打印进度
    
    Returns:
        {
            "n_folds": n_folds,
            "total_periods": N,
            "factor_results": {
                factor_name: {
                    "fold_icir": [fold1, fold2, ...],
                    "avg_icir": float,
                    "std_icir": float,
                    "cv_icir": float,        # CV = std/|avg|, 越低越稳定
                    "fold_hit_rate": [...],
                    "avg_hit_rate": float,
                    "stability_grade": "A"/"B"/"C"/"D",
                    "train_icirs": [...],
                }
            },
            "sorted_by_stability": [(name, cv_icir), ...],
        }
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"[K-fold OOS] {n_folds}-fold 时序滚动评估")
        print(f"{'='*70}")

    # 获取共同的时间索引
    all_dates = None
    for name, df in factor_dfs.items():
        if all_dates is None:
            all_dates = pd.DatetimeIndex(df.index)
        else:
            all_dates = all_dates.intersection(df.index)
    all_dates = all_dates.intersection(forward_returns.index)

    if verbose:
        print(f"  共同时间窗口: {all_dates[0].strftime('%Y-%m-%d')} ~ "
              f"{all_dates[-1].strftime('%Y-%m-%d')} ({len(all_dates)} 周)")

    # K-fold 切分
    try:
        folds = _kfold_split_timeseries(all_dates, n_folds)
    except ValueError as e:
        print(f"  [WARN] 数据不足以做 {n_folds}-fold: {e}")
        # 退化为 3-fold
        folds = _kfold_split_timeseries(all_dates, 3)
        print(f"  [FALLBACK] 退化为 {len(folds)}-fold")

    if verbose:
        print(f"  Folds: {len(folds)}")
        for i, (train, test) in enumerate(folds):
            print(f"    Fold {i+1}: train={train[0].strftime('%Y-%m')}~{train[-1].strftime('%Y-%m')} "
                  f"({len(train)}w) → test={test[0].strftime('%Y-%m')}~{test[-1].strftime('%Y-%m')} "
                  f"({len(test)}w)")

    # 对每个因子评估
    factor_results = {}
    total_factors = len(factor_dfs)

    for fi, (name, f_df) in enumerate(factor_dfs.items()):
        fold_icirs = []
        fold_hit_rates = []
        train_icirs = []

        for fold_i, (train_idx, test_idx) in enumerate(folds):
            # 训练集 ICIR (参考)
            train_ic_series = _compute_ic_series(
                f_df.loc[f_df.index.intersection(train_idx)],
                forward_returns.loc[forward_returns.index.intersection(train_idx)]
            )
            if len(train_ic_series) > 0:
                train_ir = _icir_from_ic_series(train_ic_series["IC"].values)
                if np.isfinite(train_ir):
                    train_icirs.append(train_ir)

            # 测试集 ICIR (OOS)
            test_ic_series = _compute_ic_series(
                f_df.loc[f_df.index.intersection(test_idx)],
                forward_returns.loc[forward_returns.index.intersection(test_idx)]
            )

            if len(test_ic_series) >= min_test_weeks:
                oos_icir = _icir_from_ic_series(test_ic_series["IC"].values)
                oos_hit = _hit_rate(test_ic_series["IC"].values)
                if np.isfinite(oos_icir):
                    fold_icirs.append(oos_icir)
                if np.isfinite(oos_hit):
                    fold_hit_rates.append(oos_hit)

        if len(fold_icirs) >= 2:
            avg_icir = float(np.mean(fold_icirs))
            std_icir = float(np.std(fold_icirs, ddof=1))
            cv_icir = abs(std_icir / avg_icir) if abs(avg_icir) > 1e-8 else 999.0
            avg_hit = float(np.mean(fold_hit_rates)) if fold_hit_rates else np.nan

            # 稳定性评级
            if cv_icir < 0.3 and avg_icir > 0.3:
                grade = "A"
            elif cv_icir < 0.5:
                grade = "B"
            elif cv_icir < 1.0:
                grade = "C"
            else:
                grade = "D"

            factor_results[name] = {
                "fold_icirs": [round(x, 4) for x in fold_icirs],
                "avg_icir": round(avg_icir, 4),
                "std_icir": round(std_icir, 4),
                "cv_icir": round(cv_icir, 4),
                "fold_hit_rates": [round(x, 4) for x in fold_hit_rates],
                "avg_hit_rate": round(avg_hit, 4) if np.isfinite(avg_hit) else None,
                "stability_grade": grade,
                "train_icirs": [round(x, 4) for x in train_icirs],
                "n_valid_folds": len(fold_icirs),
            }

        if verbose and (fi + 1) % 20 == 0:
            print(f"  进度: {fi+1}/{total_factors} 因子评估完成")

    if verbose:
        n_valid = len(factor_results)
        print(f"\n  完成: {n_valid}/{total_factors} 因子有足够的 OOS 数据")
        grades = {}
        for r in factor_results.values():
            g = r["stability_grade"]
            grades[g] = grades.get(g, 0) + 1
        print(f"  稳定性分布: {dict(sorted(grades.items()))}")

    # 按稳定性排序
    sorted_stable = sorted(
        factor_results.items(),
        key=lambda x: (x[1]["cv_icir"], -abs(x[1]["avg_icir"]))
    )

    return {
        "n_folds": len(folds),
        "total_periods": len(all_dates),
        "factor_results": factor_results,
        "sorted_by_stability": [(name, r["cv_icir"], r["avg_icir"], r["stability_grade"])
                                for name, r in sorted_stable],
    }


# ============================================================
# 配对复合 K-fold 评估
# ============================================================

def evaluate_pair_kfold_oos(
    factor_a: str,
    factor_b: str,
    factor_dfs: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    n_folds: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    评估一个因子配对 (rank-product composite) 的 K-fold OOS 表现.
    
    配对方式: rank_pct(A) × rank_pct(B) → composite score
    """
    if factor_a not in factor_dfs:
        return {"error": f"因子 {factor_a} 不存在"}
    if factor_b != "__STANDALONE__" and factor_b not in factor_dfs:
        return {"error": f"因子 {factor_b} 不存在"}

    df_a = factor_dfs[factor_a]

    if factor_b == "__STANDALONE__":
        composite_df = df_a.copy()
    else:
        df_b = factor_dfs[factor_b]
        # 对齐
        common_idx = df_a.index.intersection(df_b.index)
        common_cols = df_a.columns.intersection(df_b.columns)
        if len(common_idx) == 0 or len(common_cols) == 0:
            return {"error": "两因子无共同时间/股票"}

        # rank-percentile product
        rp_a = df_a.loc[common_idx, common_cols].rank(axis=1, pct=True)
        rp_b = df_b.loc[common_idx, common_cols].rank(axis=1, pct=True)
        composite_df = rp_a * rp_b

    # 用单因子评估函数
    single_results = evaluate_kfold_oos(
        {"composite": composite_df},
        forward_returns,
        n_folds=n_folds,
        verbose=verbose,
    )

    if "composite" in single_results["factor_results"]:
        return {
            "pair": f"{factor_a} × {factor_b}",
            **single_results["factor_results"]["composite"],
            "n_folds": single_results["n_folds"],
        }
    else:
        return {"error": "评估失败", "pair": f"{factor_a} × {factor_b}"}


# ============================================================
# 报告输出
# ============================================================

def print_kfold_report(results: Dict[str, Any], top_n: int = 30):
    """打印 K-fold OOS 评估报告."""
    print(f"\n{'='*80}")
    print(f"K-fold OOS 评估报告 — {results['n_folds']}-fold | "
          f"{results['total_periods']} 周总量")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'因子':<30} {'avg_ICIR':>9} {'±std':>8} {'CV':>8} "
          f"{'hit%':>7} {'等级':>4}")
    print(f"{'-'*75}")

    for rank, (name, cv, avg, grade) in enumerate(results["sorted_by_stability"][:top_n], 1):
        r = results["factor_results"][name]
        hit_str = f"{r['avg_hit_rate']:.1%}" if r["avg_hit_rate"] is not None else "N/A"
        print(f"{rank:<5} {name:<30} {avg:>+9.4f} ±{r['std_icir']:.4f} "
              f"{cv:>8.4f} {hit_str:>7} {grade:>4}")

    # 统计摘要
    avgs = [r["avg_icir"] for r in results["factor_results"].values()]
    cvs = [r["cv_icir"] for r in results["factor_results"].values()]
    grades = {}
    for r in results["factor_results"].values():
        g = r["stability_grade"]
        grades[g] = grades.get(g, 0) + 1

    print(f"\n{'='*80}")
    print(f"摘要: avg_ICIR={np.mean(avgs):+.4f}±{np.std(avgs):.4f} | "
          f"median_CV={np.median(cvs):.4f} | 等级分布: {dict(sorted(grades.items()))}")
    print(f"A级(cv<0.3 & avg>0.3): {grades.get('A', 0)} | "
          f"B级(cv<0.5): {grades.get('B', 0)} | "
          f"C级(cv<1.0): {grades.get('C', 0)} | "
          f"D级: {grades.get('D', 0)}")


def export_kfold_report(results: Dict[str, Any], output_path: str = None) -> str:
    """导出 K-fold 报告为 JSON."""
    if output_path is None:
        output_path = str(Path(__file__).parent / "output" / "kfold_oos_report.json")

    export = {
        "n_folds": results["n_folds"],
        "total_periods": results["total_periods"],
        "top_by_stability": [
            {"rank": i+1, "factor": name, "avg_icir": r["avg_icir"],
             "std_icir": r["std_icir"], "cv_icir": r["cv_icir"],
             "grade": r["stability_grade"]}
            for i, (name, r) in enumerate(
                sorted(results["factor_results"].items(),
                       key=lambda x: (x[1]["cv_icir"], -abs(x[1]["avg_icir"])))
            )
        ],
        "factor_details": {name: r for name, r in results["factor_results"].items()},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)

    print(f"  K-fold 报告已导出: {output_path}")
    return output_path


# ============================================================
# 配对批量评估
# ============================================================

def evaluate_all_llm_pairs_kfold(
    proposals: List[Dict],
    factor_dfs: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    n_folds: int = 5,
    verbose: bool = True,
) -> List[Dict]:
    """
    对所有 LLM 提案的配对做 K-fold OOS 评估.
    
    Returns:
        按 avg_icir 降序排列的配对评估结果列表
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"[LLM Pair K-fold] 评估 {len(proposals)} 个 LLM 提案配对")
        print(f"{'='*70}")

    pair_results = []
    for i, p in enumerate(proposals):
        if verbose:
            print(f"  [{i+1}/{len(proposals)}] {p['factor_a']} × {p['factor_b']} "
                  f"({p['pairing_type']})", end=" ")

        result = evaluate_pair_kfold_oos(
            p["factor_a"], p["factor_b"],
            factor_dfs, forward_returns,
            n_folds=n_folds, verbose=False,
        )

        if "error" in result:
            if verbose:
                print(f"→ SKIP: {result['error']}")
            continue

        result["proposal_id"] = p["id"]
        result["rationale"] = p["rationale"]
        result["pairing_type"] = p["pairing_type"]

        if verbose:
            print(f"→ avg_ICIR={result.get('avg_icir', 'N/A'):+.4f} "
                  f"cv={result.get('cv_icir', 'N/A'):.4f} "
                  f"grade={result.get('stability_grade', '?')}")

        pair_results.append(result)

    # 排序
    pair_results.sort(key=lambda x: x.get("avg_icir", -999), reverse=True)

    if verbose:
        print(f"\n  有效配对: {len(pair_results)}/{len(proposals)}")
        if pair_results:
            top = pair_results[0]
            print(f"  Top: {top['pair']} → "
                  f"avg_ICIR={top['avg_icir']:+.4f} "
                  f"cv={top['cv_icir']:.4f} "
                  f"({top['proposal_id']})")

    return pair_results


# ============================================================
# 独立运行
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 70)
    print("K-fold OOS 模块测试")
    print("=" * 70)

    # 生成模拟数据验证逻辑
    np.random.seed(42)
    n_stocks = 100
    n_weeks = 150

    dates = pd.date_range("2021-01-01", periods=n_weeks, freq="W-FRI")
    stocks = [f"stock_{i:04d}" for i in range(n_stocks)]

    # 模拟: 因子 A 有真实 alpha (IC~0.05), 因子 B 是纯噪音
    factor_a = pd.DataFrame(
        np.random.randn(n_weeks, n_stocks) * 0.1,
        index=dates, columns=stocks
    )
    forward = factor_a * 0.05 + np.random.randn(n_weeks, n_stocks) * 0.1
    forward = pd.DataFrame(forward, index=dates, columns=stocks)

    factor_b = pd.DataFrame(
        np.random.randn(n_weeks, n_stocks),
        index=dates, columns=stocks
    )

    test_dfs = {"alpha_factor": factor_a, "noise_factor": factor_b}

    results = evaluate_kfold_oos(test_dfs, forward, n_folds=5)
    print_kfold_report(results)
    export_kfold_report(results, "output/kfold_oos_demo.json")
