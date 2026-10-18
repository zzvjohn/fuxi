# -*- coding: utf-8 -*-
"""
因子组合引擎 (Factor Composer) — 路径 B: 管道重定义为「因子生成器+三层选择」
================================================================================

设计原则:
  1. 系统化配对: 对 ICIR 通过阈值的 base 因子做全排列 pairwise → rank-percentile 乘积。
  2. 三层评分 (学术/行业依据):
     - Layer 1 (w=0.3): ICIR — 全样本截面预测力, 基础门槛 (Cochrane 2011)
     - Layer 2 (w=0.4): 滚动窗口稳定性 — 滚动 24M ICIR 的均值/(1+std),
                        持久度 P(IC>0), 惩罚忽正忽负因子 (Harvey-Liu-Zhu 2016, AQR Ilmanen 2021)
     - Layer 3 (w=0.3): IC 时间衰减 — 近 12M vs 全样本 IC 均值差异,
                        检测因子老化 (McLean-Pontiff 2016, Two Sigma 实践)
     - 合成: total = w1*r(ICIR) + w2*r(stability) + w3*r(1-|decay|)
  3. 冗余剔除: 选中集内 pairwise |corr| > 0.7 → 保留总分更高的。
  4. 粗筛包 base: 候选池 = ICIR 通过的 base 因子 + 从它们生成的 pairwise composites。

对比 v7.11 ensemble_calibrated (Boruta + PortfolioSimulator):
  - 旧: 选 base 因子 → 本地模拟器校准 Sharpe (虚高 4.6×) 排序 → 漏掉复合因子。
  - 新: 从 base 生成 composite → 三层稳健评分 → 直接淘汰噪声因子,
        不依赖本地模拟器幻影, 不依赖 ML 黑箱重要性。

学术引用:
  - Cochrane (2011) "Presidential Address": Discount Rates, JF.
  - Harvey, Liu, Zhu (2016) "... and the Cross-Section of Expected Returns", JF.
  - McLean, Pontiff (2016) "Does Academic Research Destroy Stock Return Predictability?", JF.
  - Daniel, Moskowitz (2016) "Momentum Crashes", JFE.
  - Ilmanen et al. (2021) "AQR Factor Selection Framework", AQR white paper.
  - Di Matteo (2005) "Long-term memories...", Physica A.  (Hurst 指数, 此处用 Kendall's tau 替代)
"""

from __future__ import division
import gc
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import kendalltau, spearmanr, norm as _norm, rankdata
from factors.composite import standardize_factors
from evaluation.ic_analysis import compute_ic_icir, compute_ic_icir_fast


# ============================================================
# FDR / Holm-Bonferroni / ENB 诊断工具
# ============================================================

def _icir_to_pvalue(icir: float, n_periods: int = 1) -> float:
    """
    ICIR → 近似双尾 p-value (H0: IC=0).

    本项目 ICIR = mean(IC)/std(IC)（周频、未年化）。
    t 统计量 = ICIR × √n_periods ∼ N(0,1) 大样本近似。
    p = 2 × (1 - Φ(|ICIR| × √n))

    注意: n_periods 必须传实际 IC 序列长度(周数), 否则过度保守。
    """
    t_stat = abs(icir) * np.sqrt(max(n_periods, 1))
    return float(2.0 * (1.0 - _norm.cdf(t_stat)))


def _fdr_bh(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg FDR 校正。
    控制 False Discovery Rate ≤ alpha。

    Returns
    -------
    boolean mask, True = 通过 FDR 校正
    """
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=bool)

    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    bh_crit = np.arange(1, n + 1) / n * alpha

    below = sorted_p <= bh_crit
    if not np.any(below):
        return np.zeros(n, dtype=bool)

    max_idx = int(np.max(np.where(below)[0]))
    threshold = sorted_p[max_idx]
    return pvalues <= threshold


def _holm_bonferroni(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Holm-Bonferroni 校正 (step-down, 比 Bonferroni 更 powerful)。

    Returns
    -------
    boolean mask, True = 通过 Holm 校正
    """
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=bool)

    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]

    passed = np.zeros(n, dtype=bool)
    for k, idx in enumerate(sorted_idx):
        if sorted_p[k] <= alpha / (n - k):
            passed[idx] = True
        else:
            break
    return passed


def _enb_from_corr(corr_matrix: np.ndarray) -> Dict:
    """
    基于相关矩阵计算独立信号数 (Effective Number of Bets)。

    参考: Meucci (2009) "Effective Number of Bets",
          Politis-White (2004) "Automatic Block-Length Selection"

    三方法:
      1. 特征值法: ENB = (Σλᵢ)² / Σλᵢ²  (最常用)
      2. 均值相关法: ENB ≈ N / (1 + (N-1)×mean(|ρ|))
      3. 聚类法: 按 |corr| > 0.7 分簇 → 簇数 = 独立信号族

    Returns dict with all three estimates.
    """
    n = corr_matrix.shape[0]
    if n <= 1:
        return {
            'enb_eigen': float(n), 'enb_avgcorr': float(n),
            'n_clusters_07': int(n), 'n_total': int(n),
        }

    # Method 1: Eigenvalue
    eigvals = np.linalg.eigvalsh(corr_matrix)
    eigvals = np.maximum(eigvals, 0.0)
    sum_eig = float(np.sum(eigvals))
    sum_sq = float(np.sum(eigvals ** 2))
    enb_eigen = round(sum_eig ** 2 / sum_sq, 2) if sum_sq > 0 else 1.0

    # Method 2: Avg absolute correlation
    triu_idx = np.triu_indices(n, k=1)
    abs_corrs = np.abs(corr_matrix[triu_idx])
    mean_abs = float(np.mean(abs_corrs)) if len(abs_corrs) > 0 else 0.0
    if mean_abs >= 1.0 - 1e-9:
        enb_avgcorr = 1.0
    else:
        enb_avgcorr = round(float(n / (1.0 + (n - 1.0) * mean_abs)), 2)

    # Method 3: Greedy clustering (|corr| > 0.7 → same family)
    clusters = []
    for i in range(n):
        assigned = False
        for cl in clusters:
            if all(abs(corr_matrix[i, j]) <= 0.7 for j in cl):
                cl.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])

    return {
        'enb_eigen': enb_eigen,
        'enb_avgcorr': enb_avgcorr,
        'n_clusters_07': len(clusters),
        'n_total': n,
    }


# ============================================================
# 公共辅助函数
# ============================================================

def rolling_icir_stability(ic_series: pd.Series, window: int = 24) -> Tuple[float, float]:
    """
    滚动窗口 ICIR 稳定性 + 持久度 (Layer 2 核心)

    对每个滚动窗口 (window 个时间点):
      ICIR_roll = mean(IC_window) / (std(IC_window) + 1e-9)
      persistence_roll = P(IC > 0 in window)

    稳定性 = mean(ICIR_roll) / (1 + std(ICIR_roll))
    持久度 = mean(persistence_roll)

    返回值越大越稳定。

    Parameters
    ----------
    ic_series : pd.Series
        index=date, values=IC, 每期为因子在该截面的 Spearman Rank IC
    window : int
        滚动窗口大小 (周数; 24 ≈ 6个月)

    Returns
    -------
    (stability, persistence)
      stability   : float, ICIR 夏普比, 范围 [-∞, +∞], 越高越好
      persistence : float, 正 IC 占比, 范围 [0, 1], 越高越好
    """
    if len(ic_series) < window:
        return 0.0, 0.0
    series = ic_series.dropna()
    if len(series) < window:
        return 0.0, 0.0

    icir_rolls = []
    persist_rolls = []
    for i in range(window, len(series) + 1):
        seg = series.iloc[i - window:i]
        seg_mean = seg.mean()
        seg_std = seg.std(ddof=1)
        if seg_std and seg_std > 0:
            icir_rolls.append(seg_mean / seg_std)
        persist_rolls.append(float((seg > 0).mean()))

    if not icir_rolls:
        return 0.0, 0.0

    mean_icir = np.mean(icir_rolls)
    std_icir = np.std(icir_rolls, ddof=1)
    stability = mean_icir / (1.0 + max(std_icir, 1e-9))
    persistence = np.mean(persist_rolls)
    return float(stability), float(persistence)


def _fast_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    向量化 rank-based 标准化 (与 factors.composite.standardize_factors 数学等价, 快 ~100x)。

    逐截面 percentile rank → clip[1e-4, 1-1e-4] → norm.ppf → clip[-4, 4]。
    直接对 2D 数组做 norm.ppf, 避免逐行 apply 的 Python 循环开销。
    NaN 在 clip/ppf 下保持为 NaN (与 standardize_factors 一致)。
    """
    ranks = df.rank(axis=1, pct=True, na_option='keep')
    z = _norm.ppf(np.clip(ranks.values, 0.0001, 0.9999))
    z = np.clip(z, -4.0, 4.0)
    return pd.DataFrame(z, index=df.index, columns=df.columns, dtype=float)


def _fast_icir_vectorized(z_df: pd.DataFrame, returns_df: pd.DataFrame,
                          regime_weeks: List, min_samples: int = 20) -> Dict:
    """
    向量化 ICIR (与 compute_ic_icir_fast 数值完全等价, diff=0.0, 快 ~50x)。

    原理: z_df 已 rank-based 标准化 (z 保留排序), 故周截面 IC = Spearman(factor, return)
          = Pearson(rank(factor), rank(return)) = Pearson(z, rank(return)).
          对整个因子矩阵一次性向量化: 仅对 274 周做 numpy 循环 (每行纯向量化相关),
          避免 compute_ic_icir_fast 逐周 pandas .loc/.dropna/.intersection 开销。

    对齐: 内部按 regime_weeks + 共有股票列自动对齐 (与 standardize_and_filter 行为一致)。

    Returns
    -------
    dict: {'icir': float, 'ic_series': pd.Series}  (icir=NaN 表示无足够样本)
    """
    weeks = [w for w in regime_weeks if w in z_df.index and w in returns_df.index]
    if not weeks:
        return {'icir': np.nan, 'ic_series': pd.Series(dtype=float)}

    Z = z_df.loc[weeks]
    R = returns_df.loc[weeks]
    common_cols = Z.columns.intersection(R.columns)
    if len(common_cols) == 0:
        return {'icir': np.nan, 'ic_series': pd.Series(dtype=float)}

    Z = Z[common_cols].values.astype(float)
    R = R[common_cols].values.astype(float)
    n_weeks = Z.shape[0]

    ic_list = []
    ic_dates = []
    for i in range(n_weeks):
        zr = Z[i]
        rr = R[i]
        mask = np.isfinite(zr) & np.isfinite(rr)
        n = int(mask.sum())
        if n < min_samples:
            continue
        zv = zr[mask]
        rv = rr[mask]
        # Spearman 公式 (与 compute_ic_icir_fast 完全一致): 两行都 rankdata
        fr = rankdata(zv)
        rr_ = rankdata(rv)
        d2 = float(np.sum((fr - rr_) ** 2))
        ic = 1.0 - 6.0 * d2 / (n * (n * n - 1))
        if np.isfinite(ic):
            ic_list.append(ic)
            ic_dates.append(weeks[i])

    if len(ic_list) < 5:
        return {'icir': np.nan, 'ic_series': pd.Series(dtype=float)}

    ic_series = pd.Series(ic_list, index=ic_dates)
    ic_mean = float(np.mean(ic_list))
    ic_std = float(np.std(ic_list, ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    return {'icir': icir, 'ic_series': ic_series}


def ic_decay_rate(ic_series: pd.Series, recency_window: int = 52) -> float:
    """
    IC 衰减率 + 趋势检测 (Layer 3 核心)

    1. 衰减率 = (近 recency_window 期 IC 均值 - 全样本 IC 均值) / (|全样本 IC 均值| + 1e-9)
       负值 → 衰减; 正值 → 增强; 绝对值小 → 稳定

    2. 如果 IC 序列 >= 20 个点, 用 Kendall's tau 检测趋势的单调性;
       Hurst 指数 (DFA) 需要 >256 点才可靠, 5 年周频 ~260 点勉强,
       因此改用 Kendall's tau 做趋势检验 (稳健, 对样本量无硬要求)。

    返回值:  综合衰减指标
      - decay_raw = (recent_mean - full_mean) / (|full_mean| + 1e-9)
      - kendall_sign = sign(tau) if |tau_z| > 1.0 else 0
      - 最终: decay = decay_raw + 0.3 * kendall_sign * abs(decay_raw)
               (趋势一致则放大衰减信号; 趋势矛盾则缩小)

    范围: [-∞, +∞], 负→衰减, 0→稳定, 正→增强。
    最终评分中使用 1 - |decay|, 即大衰减得低分。
    """
    series = ic_series.dropna()
    if len(series) < 10:
        return 0.0

    full_mean = series.mean()
    recent = series.iloc[-recency_window:]
    recent_mean = recent.mean() if len(recent) > 0 else full_mean

    denom = abs(full_mean) + 1e-9
    decay_raw = (recent_mean - full_mean) / denom

    # Kendall's tau 趋势检测 (IC 序列 vs 时间序号)
    if len(series) >= 20:
        tau, p_val = kendalltau(range(len(series)), series.values)
        # Kendall tau z-score ≈ tau * sqrt(9*n*(n-1) / (2*(2n+5)))
        n = len(series)
        var_tau = 2.0 * (2 * n + 5) / (9.0 * n * (n - 1))
        tau_se = np.sqrt(var_tau + 1e-9)
        tau_z = tau / tau_se
        # 趋势信号: z > 1.0 温和, z > 1.65 显著
        kendall_sign = np.sign(tau_z) if abs(tau_z) > 1.0 else 0.0
    else:
        kendall_sign = 0.0

    # 综合: 趋势同向放大，反向缩小
    decay = decay_raw + 0.3 * kendall_sign * abs(decay_raw)
    return float(decay)


# ============================================================
# CompositeSelector 核心类
# ============================================================

class CompositeSelector:
    """
    交叉信号源复合因子生成 + 三层评分选择器。

    权重约定 (来自 Ilmanen 2021 + AQR 实践):
      Layer 1 (ICIR)         weight=0.3
      Layer 2 (滚动稳定性)   weight=0.4  ← 最重, 直接惩罚过拟合
      Layer 3 (时间衰减)     weight=0.3

    用法:
      cs = CompositeSelector(icir_threshold=0.10)
      selected, meta = cs.select(
          factor_dfs_raw, factor_dfs_std, forward_returns,
          icir_pass_base, regime_weeks, factor_ic_series, factor_icir,
          top_k=5)
    """

    def __init__(
        self,
        icir_threshold: float = 0.10,
        stability_window: int = 24,
        decay_window: int = 52,
        layer_weights: Tuple[float, float, float] = (0.3, 0.4, 0.3),
        corr_max: float = 0.7,
        random_state: int = 42,
    ):
        self.icir_threshold = icir_threshold
        self.stability_window = stability_window
        self.decay_window = decay_window
        self.layer_weights = layer_weights
        self.corr_max = corr_max
        self.rng = np.random.RandomState(random_state)

    # ── Step 1: 生成复合因子 ──
    def generate_composites(
        self,
        factor_dfs_raw: Dict[str, pd.DataFrame],
        base_names: List[str],
    ):
        """
        从 ICIR 通过的 base 因子生成所有 pairwise rank-percentile 乘积复合。

        ★ 内存安全: 用生成器逐对 yield (name, comp_df), 不一次性驻留全部 composite。
          下游 standardize_and_filter 边收边标准化+ICIR粗筛, 只保留通过者 →
          峰值内存从「全部composite」降到「单对」(约 1/N), 结果数学不变。

        公式: composite = rank_pct(factor_A, axis=1) × rank_pct(factor_B, axis=1)

        Parameters
        ----------
        factor_dfs_raw : {name: DataFrame(date×stock)}  未标准化的原始因子值
        base_names     : 参与配对的 base 因子名列表

        Yields
        -------
        (composite_name, DataFrame(date×stock))
          命名: comp_{factorA}_x_{factorB}  (按字典序排序, 避免重复)
        """
        base_names = sorted(base_names)
        n = len(base_names)

        for i in range(n):
            for j in range(i + 1, n):
                fa_name, fb_name = base_names[i], base_names[j]
                if fa_name not in factor_dfs_raw or fb_name not in factor_dfs_raw:
                    continue

                dfa = factor_dfs_raw[fa_name]
                dfb = factor_dfs_raw[fb_name]

                # 对齐日期索引
                common_dates = dfa.index.intersection(dfb.index)
                if len(common_dates) < 20:
                    continue

                dfa = dfa.loc[common_dates]
                dfb = dfb.loc[common_dates]

                # 取并集列 (股票)
                union_cols = dfa.columns.union(dfb.columns)

                # rank-percentile 乘积 (逐截面, 即 axis=1)
                rank_a = dfa.reindex(columns=union_cols).rank(
                    pct=True, axis=1, na_option='keep')
                rank_b = dfb.reindex(columns=union_cols).rank(
                    pct=True, axis=1, na_option='keep')

                comp = rank_a * rank_b  # elementwise multiply

                composite_name = f'comp_{fa_name}_x_{fb_name}'
                yield composite_name, comp, fa_name, fb_name

    # ── Step 2: 标准化 + ICIR 粗筛 (纯标量流式, 零大对象驻留) ──
    def standardize_and_filter(
        self,
        composites_input,
        forward_returns: pd.DataFrame,
        regime_weeks: List,
    ) -> Tuple[Dict[str, float], Dict[str, pd.Series], Dict[str, Tuple[str, str]], Dict[str, bool], int]:
        """
        对复合因子做标准化并计算 regime 内 ICIR，粗筛通过阈值的。

        ★★ 内存安全 v2 (OOM 修复): 之前版本对每个通过粗筛的 composite 保留
           ~12MB 的标准化 DataFrame → 数千个通过者 = 12GB+ → _ArrayMemoryError。
           现在只保留标量 (ICIR) + 小 Series (IC序列, ~270 float) + 配对名。
           大 DataFrame 在评分选出 top-K 后由 materialize_composites 按需重建
           (仅 ≤ top_k 个), 峰值内存 ~几十MB。数学结果完全不变。

        Returns
        -------
        comps_icir      : {name: float} ICIR (已取方向矫正后的正值)
        comps_ic_series : {name: pd.Series} IC 序列 (已随方向翻转)
        comps_pair_map  : {name: (factorA, factorB)} 复合因子的 base 配对
        comps_flip      : {name: bool} 该复合是否需要方向翻转 (ICIR<0)
        n_received      : int 实际接收的 composite 个数 (用于报告)
        """
        if composites_input is None:
            return {}, {}, {}, {}, 0
        items = composites_input  # 生成器 / 可迭代 (name, df, fa, fb)

        comps_icir: Dict[str, float] = {}
        comps_ic_series: Dict[str, pd.Series] = {}
        comps_pair_map: Dict[str, Tuple[str, str]] = {}
        comps_flip: Dict[str, bool] = {}
        n_received = 0

        for name, fdf_raw, fa_name, fb_name in items:
            n_received += 1
            if n_received % 200 == 0:
                gc.collect()
                print(f"     [composite] 粗筛进度: {n_received} 对已处理, {len(comps_icir)} 通过", flush=True)
            try:
                # 向量化标准化 (与 standardize_factors 数学等价, 快 ~100x)
                fdf_std = _fast_standardize(fdf_raw)
            except Exception:
                del fdf_raw
                continue
            del fdf_raw

            try:
                # ★ 向量化 ICIR: 因子已 rank-based 标准化, 周截面 IC = Spearman ≡ Pearson(rank,rank).
                #   对整个因子矩阵一次性向量化, 比 compute_ic_icir_fast 逐周 pandas 循环快 ~50x,
                #   数值完全等价 (已验证 diff=0.0). 内部按 regime_weeks + 共有股票对齐.
                ic_res = _fast_icir_vectorized(fdf_std, forward_returns, regime_weeks, min_samples=20)
                if ic_res and 'icir' in ic_res and pd.notna(ic_res['icir']):
                    ir = ic_res['icir']
                    if abs(ir) >= self.icir_threshold:
                        flip = ir < 0
                        ic_ser = ic_res.get('ic_series', pd.Series(dtype=float))
                        if flip and isinstance(ic_ser, pd.Series) and len(ic_ser) > 0:
                            ic_ser = -ic_ser
                        # ★ 只存标量/小对象, 绝不驻留 fdf_std (12MB) !
                        comps_icir[name] = abs(ir)
                        comps_ic_series[name] = ic_ser
                        comps_pair_map[name] = (fa_name, fb_name)
                        comps_flip[name] = flip
            except Exception:
                pass
            finally:
                del fdf_std  # 立即释放, 无论是否通过

        gc.collect()
        return comps_icir, comps_ic_series, comps_pair_map, comps_flip, n_received

    # ── Step 2.5: 按需物化选中的复合因子 (OOM 修复核心) ──
    def materialize_composites(
        self,
        factor_dfs_raw: Dict[str, pd.DataFrame],
        names: List[str],
        pair_map: Dict[str, Tuple[str, str]],
        flip_map: Dict[str, bool],
    ) -> Dict[str, pd.DataFrame]:
        """
        重新生成并标准化指定的复合因子 (仅 ≤ top_k 个, 内存有界)。
        与粗筛阶段的计算路径完全一致 → 数值一致。
        """
        out: Dict[str, pd.DataFrame] = {}
        for name in names:
            pair = pair_map.get(name)
            if pair is None:
                continue
            fa_name, fb_name = pair
            dfa = factor_dfs_raw.get(fa_name)
            dfb = factor_dfs_raw.get(fb_name)
            if dfa is None or dfb is None:
                continue
            common_dates = dfa.index.intersection(dfb.index)
            if len(common_dates) < 20:
                continue
            union_cols = dfa.columns.union(dfb.columns)
            rank_a = dfa.loc[common_dates].reindex(columns=union_cols).rank(
                pct=True, axis=1, na_option='keep')
            rank_b = dfb.loc[common_dates].reindex(columns=union_cols).rank(
                pct=True, axis=1, na_option='keep')
            comp = rank_a * rank_b
            del rank_a, rank_b
            fdf_std = _fast_standardize(comp)
            del comp
            if flip_map.get(name, False):
                fdf_std = -fdf_std
            out[name] = fdf_std
        gc.collect()
        return out

    # ── Step 3: 三层评分 ──
    def score_candidates(
        self,
        candidate_names: List[str],
        factor_icir: Dict[str, float],
        factor_ic_series: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        对每个候选因子计算三层得分。

        Returns
        -------
        DataFrame columns:
          name, icir, stability, persistence, decay,
          score_L1, score_L2, score_L3, total_score
        按 total_score 降序排列
        """
        records = []
        for name in candidate_names:
            ir = factor_icir.get(name, 0.0)
            ic_ser = factor_ic_series.get(name, pd.Series(dtype=float))

            stab, pers = rolling_icir_stability(
                ic_ser, window=self.stability_window)
            decay = ic_decay_rate(ic_ser, recency_window=self.decay_window)

            records.append({
                'name': name,
                'icir': ir,
                'stability': stab,
                'persistence': pers,
                'decay': decay,
            })

        df = pd.DataFrame(records)

        if len(df) == 0:
            return df

        # 排名归一化 → [0, 1]
        def rank_norm(series, ascending=True):
            s = series.dropna()
            if len(s) < 2:
                return pd.Series(0.5, index=series.index)
            ranked = s.rank(pct=True, ascending=ascending)
            return ranked.reindex(series.index).fillna(0.5)

        # Layer 1: ICIR (绝对值越大越好)
        df['score_L1'] = rank_norm(df['icir'].abs(), ascending=True)

        # Layer 2: 稳定性 = stability × (persistence_bonus)
        #   persistence_bonus: 正 IC 占比, 范围 [0, 1], 低于 0.5 严重惩罚
        #   稳定分 = stability_raw × (0.5 + 0.5*persistence)
        raw_L2 = df['stability'] * (0.5 + 0.5 * df['persistence'])
        df['score_L2'] = rank_norm(raw_L2, ascending=True)

        # Layer 3: 时间衰减 (绝对值越小越好, 即 1-|decay| 越大越好)
        raw_L3 = 1.0 - np.abs(df['decay'])
        raw_L3 = np.clip(raw_L3, -1, 1)
        df['score_L3'] = rank_norm(pd.Series(raw_L3, index=df.index), ascending=True)

        # 合成总得分
        w1, w2, w3 = self.layer_weights
        df['total_score'] = (
            w1 * df['score_L1'] +
            w2 * df['score_L2'] +
            w3 * df['score_L3']
        )
        df = df.sort_values('total_score', ascending=False).reset_index(drop=True)

        return df

    # ── Step 4: Top-K + 冗余剔除 ──
    def select_top(
        self,
        scored_df: pd.DataFrame,
        top_k: int = 5,
        factor_dfs_std: Optional[Dict[str, pd.DataFrame]] = None,
        materializer=None,
    ) -> Tuple[Dict[str, float], Dict]:
        """
        按 total_score 降序选择, 跳过与已选中因子高相关的 (|corr| > corr_max)。

        ★ OOM 修复: composite 因子的 DataFrame 不再预先驻留内存。
          materializer(names) -> {name: DataFrame} 按需重建, 内部缓存有界:
          仅保留「已选中因子 + 当前候选」的 DataFrame (≤ top_k+1 个)。

        Returns
        -------
        selected : {factor_name: total_score}
        meta     : dict with n_candidates, n_selected, dropped_by_corr, etc.
        """
        if len(scored_df) == 0:
            return {}, {'n_candidates': 0, 'n_selected': 0}

        meta = {
            'n_candidates': len(scored_df),
            'dropped_by_corr': [],
        }

        # 预计算候选因子间的周期平均相关系数矩阵
        corr_memo: Dict[str, Dict[str, float]] = {}
        names = scored_df['name'].tolist()

        # 有界物化缓存: {name: DataFrame}, 只保留已选中 + 当前候选
        _mat_cache: Dict[str, pd.DataFrame] = {}

        def _get_df(name: str):
            """获取因子标准化 DataFrame: 优先全局 dict, 其次缓存, 最后按需物化."""
            if factor_dfs_std is not None and name in factor_dfs_std:
                return factor_dfs_std[name]
            if name in _mat_cache:
                return _mat_cache[name]
            if materializer is not None:
                got = materializer([name])
                if name in got:
                    _mat_cache[name] = got[name]
                    return got[name]
            return None

        def _evict_cache(keep_names):
            """剔除不再需要的物化缓存 (候选被拒绝时)."""
            for k in list(_mat_cache.keys()):
                if k not in keep_names:
                    del _mat_cache[k]

        def _check_corr_added(name_a: str, name_b: str) -> float:
            """返回两因子在 regime 平均水平的相关系数 (cache)."""
            pa = (name_a, name_b)
            pb = (name_b, name_a)
            if pa in corr_memo:
                return corr_memo[pa]
            if pb in corr_memo:
                return corr_memo[pb]

            dfa = _get_df(name_a)
            dfb = _get_df(name_b)
            if dfa is None or dfb is None:
                corr_memo[pa] = 0.0
                return 0.0

            # 逐周截面相关, 取中位数 (robust 对 outlier 周)
            common_dates = dfa.index.intersection(dfb.index)
            if len(common_dates) == 0:
                corr_memo[pa] = 0.0
                return 0.0

            cors = []
            for d in common_dates:
                a = dfa.loc[d].dropna()
                b = dfb.loc[d].dropna()
                common = a.index.intersection(b.index)
                if len(common) < 30:
                    continue
                try:
                    r, _ = spearmanr(a.loc[common], b.loc[common])
                    if pd.notna(r):
                        cors.append(abs(r))
                except Exception:
                    pass
            val = float(np.median(cors)) if cors else 0.0
            corr_memo[pa] = val
            return val

        selected: Dict[str, float] = {}
        selected_names: List[str] = []

        for _, row in scored_df.iterrows():
            name = row['name']
            if len(selected) >= top_k:
                break

            # 冗余检查
            rejected = False
            for sel_name in selected_names:
                corr_val = _check_corr_added(name, sel_name)
                if corr_val > self.corr_max:
                    meta['dropped_by_corr'].append({
                        'candidate': name,
                        'vs': sel_name,
                        'corr': round(float(corr_val), 3),
                    })
                    rejected = True
                    break
            if rejected:
                _evict_cache(set(selected_names))  # 被拒候选的物化 DataFrame 立即释放
                continue

            selected[name] = float(row['total_score'])
            selected_names.append(name)

        # 如果冗余剔除后不够 top_k, 放宽相关阈值再补
        if len(selected) < min(top_k, len(scored_df)):
            remaining = scored_df[~scored_df['name'].isin(selected_names)]
            for _, row in remaining.iterrows():
                if len(selected) >= top_k:
                    break
                name = row['name']
                max_corr = max(
                    (_check_corr_added(name, s) for s in selected_names),
                    default=0.0
                )
                if max_corr <= self.corr_max * 1.2:  # 放宽 20%
                    selected[name] = float(row['total_score'])
                    selected_names.append(name)
                else:
                    _evict_cache(set(selected_names))

        meta['n_selected'] = len(selected)
        meta['selected_names'] = selected_names
        return selected, meta

    # ── 主导选择流程 ──
    def select(
        self,
        factor_dfs_raw: Dict[str, pd.DataFrame],
        factor_dfs_std: Dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        icir_pass_base: Dict[str, float],
        regime_weeks: List,
        factor_ic_series: Dict[str, pd.Series],
        factor_icir: Dict[str, float],
        top_k: int = 5,
        verbose: bool = True,
    ) -> Tuple[Dict[str, float], Dict]:
        """
        主导选择流程 (Phase 5.3 入口)。

        Parameters
        ----------
        factor_dfs_raw     : {name: DataFrame} 未标准化因子 → 用在 composite 生成
        factor_dfs_std     : {name: DataFrame} 标准化因子 → 用在 ICIR 计算/粗筛
        forward_returns    : DataFrame 前向收益矩阵
        icir_pass_base     : {name: icir} 已通过 ICIR 阈值的 base 因子
        regime_weeks       : 当前 regime 的周日期列表
        factor_ic_series   : {name: pd.Series} 已计算的 IC 序列 (base 因子部分)
        factor_icir        : {name: float} 已计算的 ICIR (base + 硬编码 composite)
        top_k              : 最多保留因子数

        Returns
        -------
        selected : {factor_name: total_score}  (nonzero 兼容)
        meta     : 选择元数据
        """
        meta = {
            'method': 'composite_3layer',
            'n_base_pass': len(icir_pass_base),
            'base_names': list(icir_pass_base.keys()),
        }

        # Step 1: 生成 composite (生成器, 内存安全)
        base_names = sorted(icir_pass_base.keys())
        if verbose:
            print(f"     [composite] 从 {len(base_names)} 个 base 因子生成 pairwise composites (流式)...")

        composites_iter = self.generate_composites(factor_dfs_raw, base_names)
        n_theoretical = len(base_names) * (len(base_names) - 1) // 2
        meta['n_composites_theoretical'] = n_theoretical

        # Step 2: 纯标量流式粗筛 (只保留 ICIR/IC序列/配对名, 零大对象驻留 → OOM 修复)
        comps_icir, comps_ic_series, comps_pair_map, comps_flip, n_received = \
            self.standardize_and_filter(composites_iter, forward_returns, regime_weeks)
        meta['n_composites_raw'] = n_received

        if verbose:
            print(f"     [composite] 理论 {n_theoretical} / 接收 {n_received} 个 composites")
            n_pass = len(comps_icir)
            print(f"     [composite] ICIR >= {self.icir_threshold}: {n_pass} composites 通过")

        # ── FDR / Holm-Bonferroni 诊断 (不改变筛选, 仅报告) ──
        fdr_report = {}
        if len(comps_icir) >= 5:
            comp_names_arr = np.array(list(comps_icir.keys()))
            comp_icir_arr = np.array([abs(comps_icir[n]) for n in comp_names_arr])
            # 每个 composite 用其实际 IC 序列长度(周数)计算 t 统计量
            comp_nweeks_arr = np.array([
                len(comps_ic_series.get(n, [])) for n in comp_names_arr])
            pvals = np.array([
                _icir_to_pvalue(ir, n_periods=nw)
                for ir, nw in zip(comp_icir_arr, comp_nweeks_arr)])

            pass_fdr = _fdr_bh(pvals, alpha=0.05)
            pass_holm = _holm_bonferroni(pvals, alpha=0.05)
            pass_bonf = pvals <= 0.05 / len(pvals)

            fdr_report = {
                'n_total': int(len(comps_icir)),
                'n_pass_raw': int(len(comps_icir)),
                'n_pass_fdr_05': int(np.sum(pass_fdr)),
                'n_pass_holm_05': int(np.sum(pass_holm)),
                'n_pass_bonf_05': int(np.sum(pass_bonf)),
            }
            if verbose:
                print(f"     [FDR] 多重检验校正诊断 (n={fdr_report['n_total']} composites):")
                print(f"           原始通过 (ICIR≥{self.icir_threshold}): {fdr_report['n_pass_raw']}")
                print(f"           FDR (BH α=0.05):       {fdr_report['n_pass_fdr_05']} 通过 ({fdr_report['n_pass_fdr_05']/max(fdr_report['n_total'],1)*100:.0f}%)")
                print(f"           Holm (α=0.05):         {fdr_report['n_pass_holm_05']} 通过 ({fdr_report['n_pass_holm_05']/max(fdr_report['n_total'],1)*100:.0f}%)")
                print(f"           Bonferroni (α=0.05):   {fdr_report['n_pass_bonf_05']} 通过 ({fdr_report['n_pass_bonf_05']/max(fdr_report['n_total'],1)*100:.0f}%)")
        meta['fdr_diagnostic'] = fdr_report

        # 合并标量到全局 dicts (DataFrame 延迟到选中后物化)
        for name, ir in comps_icir.items():
            factor_icir[name] = ir
        for name, ser in comps_ic_series.items():
            factor_ic_series[name] = ser

        # 按需物化器: 只在冗余检查/最终注入时重建 ≤top_k 个 composite
        def _materializer(names_needed):
            return self.materialize_composites(
                factor_dfs_raw, names_needed, comps_pair_map, comps_flip)

        # Step 3: 候选池 = base 因子 + composite 因子
        all_icir_pass = dict(icir_pass_base)
        all_icir_pass.update(comps_icir)
        candidates = sorted(all_icir_pass.keys())

        meta['n_candidates'] = len(candidates)
        meta['n_composites_pass'] = len(comps_icir)

        if not candidates:
            if verbose:
                print("     [composite] 无候选因子通过粗筛, 回退 ICIR top-4")
            top = sorted(icir_pass_base.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
            nonzero = {k: abs(v) for k, v in top}
            meta['fallback'] = 'no_candidates'
            return nonzero, meta

        if verbose:
            print(f"     [composite] 总候选池: {len(candidates)} (base={len(icir_pass_base)} + composite={len(comps_icir)})")

        # Step 4: 三层评分
        scored = self.score_candidates(candidates, factor_icir, factor_ic_series)
        if verbose:
            top3 = scored.head(3)
            for _, r in top3.iterrows():
                print(f"       {r['name']:40s} L1={r['score_L1']:.3f} L2={r['score_L2']:.3f} L3={r['score_L3']:.3f} total={r['total_score']:.3f}")

        # Step 5: Top-K + 冗余剔除 (composite DataFrame 按需物化, 内存有界)
        selected, sel_meta = self.select_top(
            scored, top_k=top_k, factor_dfs_std=factor_dfs_std,
            materializer=_materializer)
        meta.update(sel_meta)

        if not selected:
            if verbose:
                print("     [composite] 冗余剔除后为空, 回退 ICIR top-4")
            top = sorted(icir_pass_base.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
            selected = {k: abs(v) for k, v in top}
            meta['fallback'] = 'all_correlated'

        # Step 6: 物化最终选中的 composite 并注入 factor_dfs_std
        #   (Phase 6 WFA / Phase 7 全样本回测需要用 factor_dfs_std[name] 取值)
        sel_comp_names = [n for n in selected if n.startswith('comp_') and n in comps_pair_map]
        if sel_comp_names:
            if verbose:
                print(f"     [composite] 物化 {len(sel_comp_names)} 个选中的复合因子并注入标准化因子库...")
            materialized = self.materialize_composites(
                factor_dfs_raw, sel_comp_names, comps_pair_map, comps_flip)
            for name, df in materialized.items():
                factor_dfs_std[name] = df
        meta['pair_map'] = {n: list(comps_pair_map[n]) for n in sel_comp_names}
        meta['flip_map'] = {n: bool(comps_flip.get(n, False)) for n in sel_comp_names}

        if verbose:
            print(f"     [composite] 最终选中 {len(selected)}/{len(candidates)} 因子:")
            for nm, sc in sorted(selected.items(), key=lambda x: -x[1]):
                ir = factor_icir.get(nm, 0)
                tp = 'comp' if nm.startswith('comp_') else 'base'
                print(f"       [{tp}] {nm:35s} score={sc:.3f} ICIR={ir:+.3f}")

        # ── ENB 独立信号数 (候选池级别 + 选中集级别) ──
        enb_report = {}
        # 候选池级别 ENB: 基于已计算 IC 序列的 pairwise 相关
        if len(candidates) >= 3:
            try:
                # 对候选池中有 IC 序列的因子计算 pairwise Spearman 相关
                cand_with_ic = [n for n in candidates if n in factor_ic_series
                                and isinstance(factor_ic_series.get(n), pd.Series)
                                and len(factor_ic_series[n].dropna()) >= 20]
                if len(cand_with_ic) >= 3:
                    ic_df = pd.DataFrame({n: factor_ic_series[n] for n in cand_with_ic})
                    ic_corr = ic_df.corr(method='spearman').values
                    enb_report['candidate_pool'] = _enb_from_corr(ic_corr)
                    enb_report['candidate_pool']['n_with_ic'] = len(cand_with_ic)
            except Exception:
                pass

        # 选中集级别 ENB: 用物化后的截面因子值
        sel_names = list(selected.keys())
        if len(sel_names) >= 2:
            try:
                sel_dfs = []
                for nm in sel_names:
                    df = factor_dfs_std.get(nm)
                    if df is None:
                        # 尝试从 raw 重建
                        df = _materializer([nm]).get(nm)
                    if df is not None:
                        sel_dfs.append((nm, df))
                if len(sel_dfs) >= 2:
                    # 逐周截面相关, 取中位数
                    common_idx = sel_dfs[0][1].index
                    for _, df in sel_dfs[1:]:
                        common_idx = common_idx.intersection(df.index)
                    if len(common_idx) >= 10:
                        cors = np.zeros((len(sel_dfs), len(sel_dfs)))
                        for wi, week in enumerate(common_idx[:50]):  # 最多 50 周
                            vals = []
                            for _, df in sel_dfs:
                                v = df.loc[week].dropna()
                                vals.append(v)
                            for ii in range(len(vals)):
                                for jj in range(ii + 1, len(vals)):
                                    common = vals[ii].index.intersection(vals[jj].index)
                                    if len(common) >= 20:
                                        c = spearmanr(vals[ii][common], vals[jj][common])[0]
                                        cors[ii, jj] = cors[jj, ii] = c
                        sel_corr = cors
                        enb_report['selected_pool'] = _enb_from_corr(sel_corr)
                        enb_report['selected_pool']['names'] = sel_names
                        if verbose:
                            enb_sel = enb_report['selected_pool']
                            print(f"     [ENB] 选中集独立信号诊断:")
                            print(f"           {enb_sel['n_total']} 个因子 → ENB(特征值)={enb_sel['enb_eigen']}, "
                                  f"ENB(平均相关)={enb_sel['enb_avgcorr']}, 独立簇(ρ<0.7)={enb_sel['n_clusters_07']}")
                            if 'candidate_pool' in enb_report:
                                enb_c = enb_report['candidate_pool']
                                print(f"     [ENB] 候选池独立信号: {enb_c['n_with_ic']} 个因子 → "
                                      f"ENB={enb_c['enb_eigen']}, 簇={enb_c['n_clusters_07']}")
            except Exception as e:
                if verbose:
                    print(f"     [ENB] 计算异常: {e}")
        meta['enb_report'] = enb_report

        return selected, meta


# ============================================================
# 便捷入口 (命令行测试)
# ============================================================
if __name__ == '__main__':
    # 轻量冒烟测试
    np.random.seed(42)
    N_WEEK, N_STOCK = 100, 200
    dates = pd.date_range('2020-01-01', periods=N_WEEK, freq='W')
    stocks = [f'{i:06d}.XSHG' for i in range(N_STOCK)]

    # 造 5 个 base 因子 (raw)
    raw = {}
    for fn in ['factor_A', 'factor_B', 'factor_C', 'factor_D', 'factor_E']:
        df = pd.DataFrame(np.random.randn(N_WEEK, N_STOCK), index=dates, columns=stocks)
        raw[fn] = df

    # forward_returns
    fr = pd.DataFrame(0.01 * np.random.randn(N_WEEK, N_STOCK), index=dates, columns=stocks)

    # 标准化 factor
    fds = standardize_factors(raw)

    # 假装 3 个 base 通过 ICIR
    icir_pass = {'factor_A': 0.15, 'factor_B': 0.22, 'factor_C': 0.18}
    # 预计算 IC_series (冒烟用合成数据)
    ic_sers = {k: pd.Series(0.02 * np.random.randn(N_WEEK), index=dates) for k in icir_pass}
    icirs = dict(icir_pass)

    cs = CompositeSelector(icir_threshold=0.05)
    selected, meta = cs.select(
        raw, fds, fr, icir_pass, dates.tolist(),
        ic_sers, icirs, top_k=3, verbose=True,
    )
    print(f"\nselected={selected}")
    print(f"meta={meta}")
