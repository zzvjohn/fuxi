"""
因子预筛选模块 (v7.4)
=====================
在 NSGA-II 之前将因子空间从 ~100 缩减到 ~20-25,
解决 GA 在 460 万组合空间中仅评估 200 次的严重欠采样问题。

三步筛选:
  1. ICIR 粗筛: |ICIR| >= 0.1 → 约 73 因子
  2. 相关聚类去冗余: 层次聚类 |corr| > 0.5, 每簇选最高 ICIR → 约 25 因子
  3. WFA 存活预检: 单因子 Walk-Forward, >= 3/8 窗口通过 → 约 18 因子

输出: 筛选后的因子名列表, 约 18-25 个
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def screen_factors(
    factor_names: List[str],
    ic_summary: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    factor_dfs: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    icir_threshold: float = 0.1,
    corr_threshold: float = 0.5,
    wfa_windows: int = 8,
    wfa_min_pass: int = 3,
    wfa_icir_threshold: float = 0.1,
    verbose: bool = True,
) -> Tuple[List[str], Dict]:
    """
    三阶段因子预筛选

    Parameters
    ----------
    factor_names : 全量因子名列表
    ic_summary : 单因子 ICIR 汇总 (index=因子名, columns include 'ICIR')
    corr_matrix : 因子间 Pearson 相关性矩阵 (全因子)
    factor_dfs : {因子名: 标准化因子 DataFrame (index=日期, columns=股票)}
    forward_returns : 未来收益 (index=日期, columns=股票)
    icir_threshold : ICIR 粗筛阈值 (default 0.1)
    corr_threshold : 相关性聚类阈值 (default 0.5)
    wfa_windows : WFA 划窗数 (default 8)
    wfa_min_pass : 最少通过窗口数 (default 3)
    wfa_icir_threshold : 单窗口 ICIR 通过阈值 (default 0.1)
    verbose : 是否打印详细日志

    Returns
    -------
    selected : 筛选后的因子名列表
    stats : 各阶段统计信息
    """
    stats = {'input': len(factor_names)}

    # ================================================================
    # Step 1: ICIR 粗筛
    # ================================================================
    icir_ok = []
    icir_rejected = []
    for name in factor_names:
        if name in ic_summary.index:
            icir_val = abs(ic_summary.loc[name, 'ICIR'])
            if np.isfinite(icir_val) and icir_val >= icir_threshold:
                icir_ok.append(name)
            else:
                icir_rejected.append((name, icir_val if np.isfinite(icir_val) else 0.0))
        else:
            icir_rejected.append((name, 0.0))

    stats['after_icir'] = len(icir_ok)
    stats['icir_rejected'] = len(icir_rejected)

    if verbose:
        print(f"\n  [筛选 Step 1] ICIR >= {icir_threshold}: {len(icir_ok)}/{len(factor_names)} 通过")
        if icir_rejected:
            top_rejected = sorted(icir_rejected, key=lambda x: -x[1])[:5]
            print(f"    拒绝(含Top5): {[(n, f'{v:.3f}') for n, v in top_rejected]}")

    if len(icir_ok) < 6:
        print(f"  [WARN] ICIR筛选后仅剩 {len(icir_ok)} 因子, 跳过后续聚类")
        stats['after_wfa'] = len(icir_ok)
        return icir_ok, stats

    # ================================================================
    # Step 2: 相关性聚类去冗余
    # ================================================================
    # 提取筛选后因子的相关性子矩阵
    common = [n for n in icir_ok if n in corr_matrix.index and n in corr_matrix.columns]
    if len(common) < 2:
        stats['after_corr'] = len(icir_ok)
        stats['after_wfa'] = len(icir_ok)
        if verbose:
            print(f"  [筛选 Step 2] 不足2个因子, 跳过聚类")
        return icir_ok, stats

    sub_corr = corr_matrix.loc[common, common]

    # 距离矩阵: d = 1 - |corr|, 高相关 → 近距离 → 同簇
    dist = 1.0 - np.abs(sub_corr.values)
    # ★ NaN 容错: 相关矩阵含NaN(部分因子数据不足)→距离填1.0(=最不相关/最大距离)
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)
    np.fill_diagonal(dist, 0.0)

    # 层次聚类 (Ward)
    try:
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='ward')
        clusters = fcluster(Z, t=1.0 - corr_threshold, criterion='distance')
    except Exception:
        if verbose:
            print(f"  [WARN] 聚类失败, 跳过此步")
        stats['after_corr'] = len(icir_ok)
        stats['after_wfa'] = len(icir_ok)
        return icir_ok, stats

    # 每簇选最高 ICIR 的代表
    cluster_map = {}  # cluster_id → [(name, icir)]
    for name, cid in zip(common, clusters):
        icir_val = abs(ic_summary.loc[name, 'ICIR'])
        if cid not in cluster_map:
            cluster_map[cid] = []
        cluster_map[cid].append((name, icir_val))

    representatives = []
    for cid, members in cluster_map.items():
        best = max(members, key=lambda x: x[1])
        representatives.append(best[0])

    stats['after_corr'] = len(representatives)
    stats['n_clusters'] = len(cluster_map)

    if verbose:
        print(f"  [筛选 Step 2] 相关聚类 |corr|>{corr_threshold}: "
              f"{len(icir_ok)} → {len(representatives)} 因子 ({len(cluster_map)} 簇)")
        # 打印每个簇的代表和成员数
        for cid in sorted(cluster_map.keys()):
            members = cluster_map[cid]
            if len(members) > 1:
                best_name = max(members, key=lambda x: x[1])[0]
                member_names = [m[0] for m in members]
                print(f"    簇#{cid}: {len(members)}因子, 代表={best_name} "
                      f"(成员: {[n for n in member_names if n != best_name][:3]})")

    if len(representatives) < 6:
        if verbose:
            print(f"  [WARN] 聚类后仅剩 {len(representatives)} 因子, 跳过 WFA")
        stats['after_wfa'] = len(representatives)
        return representatives, stats

    # ================================================================
    # Step 3: WFA 存活预检 (单因子 Walk-Forward)
    # ================================================================
    dates = pd.DatetimeIndex(forward_returns.index)
    n_total = len(dates)
    min_train = 52  # 最少 52 周训练

    if n_total < min_train * 2:
        if verbose:
            print(f"  [筛选 Step 3] 数据不足 ({n_total}周), 跳过 WFA")
        stats['after_wfa'] = len(representatives)
        return representatives, stats

    # 划窗边界
    window_edges = np.linspace(min_train, n_total - min(wfa_windows, 26),
                               wfa_windows + 1, dtype=int)

    wfa_pass_count = {}
    for name in representatives:
        if name not in factor_dfs:
            wfa_pass_count[name] = 0
            continue

        factor = factor_dfs[name]
        passes = 0
        for i in range(wfa_windows):
            test_start = window_edges[i]
            test_end = window_edges[i + 1]
            test_dates = dates[test_start:test_end]

            # 提取该窗口的因子值和收益
            common_dates = factor.index.intersection(test_dates)
            if len(common_dates) < 4:
                continue

            f_sub = factor.loc[common_dates]
            r_sub = forward_returns.loc[common_dates]

            # 计算此窗口的截面 IC
            common_stocks = f_sub.columns.intersection(r_sub.columns)
            if len(common_stocks) < 10:
                continue

            ics = []
            for dt in common_dates:
                f_cs = f_sub.loc[dt, common_stocks].dropna()
                r_cs = r_sub.loc[dt, common_stocks].dropna()
                both = f_cs.index.intersection(r_cs.index)
                if len(both) < 10:
                    continue
                ic = np.corrcoef(f_cs.loc[both], r_cs.loc[both])[0, 1]
                if np.isfinite(ic):
                    ics.append(ic)

            if len(ics) >= 4:
                ic_mean = np.mean(ics)
                ic_std = np.std(ics, ddof=1) or 1e-10
                icir_window = ic_mean / ic_std
                if abs(icir_window) >= wfa_icir_threshold:
                    passes += 1

        wfa_pass_count[name] = passes

    wfa_ok = [n for n, p in wfa_pass_count.items() if p >= wfa_min_pass]
    wfa_rejected = [(n, p) for n, p in wfa_pass_count.items() if p < wfa_min_pass]

    stats['after_wfa'] = len(wfa_ok)
    stats['wfa_rejected'] = len(wfa_rejected)
    stats['wfa_pass_counts'] = wfa_pass_count

    if verbose:
        print(f"  [筛选 Step 3] WFA >= {wfa_min_pass}/{wfa_windows} 窗口: "
              f"{len(wfa_ok)}/{len(representatives)} 通过")
        if wfa_rejected:
            top_rej = sorted(wfa_rejected, key=lambda x: -x[1])[:5]
            print(f"    拒绝(含Top5): {[(n, f'{p}/{wfa_windows}') for n, p in top_rej]}")

    if len(wfa_ok) < 4:
        if verbose:
            print(f"  [WARN] WFA 后仅剩 {len(wfa_ok)} 因子, 回退到聚类结果")
        return representatives, stats

    return wfa_ok, stats


def print_screening_summary(stats: Dict, final_names: List[str]):
    """打印筛选摘要"""
    print(f"\n  {'='*50}")
    print(f"  因子预筛选摘要 (v7.4)")
    print(f"  {'='*50}")
    print(f"  输入:     {stats.get('input', '?'):>4} 因子")
    print(f"   ICIR筛:  {stats.get('after_icir', '?'):>4} 因子 (|ICIR| >= 0.1)")
    print(f"   去冗余:  {stats.get('after_corr', '?'):>4} 因子 ({stats.get('n_clusters', '?')} 簇)")
    print(f"   WFA筛:  {stats.get('after_wfa', '?'):>4} 因子 (>= 3/8 窗口)")
    print(f"  缩减率:  {stats.get('input', 1) / max(len(final_names), 1):.1f}x "
          f"({stats.get('input', '?')} → {len(final_names)})")
    print(f"  组合空间: C({len(final_names)},4) = "
          f"{comb(len(final_names), 4) if len(final_names) >= 4 else 0:,}")
    print(f"  {'='*50}\n")


def comb(n, k):
    """组合数"""
    if n < k:
        return 0
    import math
    return math.comb(n, k)
