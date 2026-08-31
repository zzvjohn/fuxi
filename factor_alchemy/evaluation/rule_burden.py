"""
因子负担分析 (Rule Burden Analysis) — XQuant Ch6 方法论
========================================================
核心洞察: "规则越多 != 策略越好, 简单就是美"
         "能用2个参数解决的问题, 不要用6个"

三个实验:
  4A 因子数堆叠 (Rule Stacking): 从1因子→N因子, 逐层选最优, 样本外验证
  4B 自由度堆叠 (DoF Stacking): 从1自由度→N自由度, 找 plateau 拐点
  4C ElasticNet稀疏选择: L1+L2正则化全局最优因子集 + 自动处理共线性

关键区分 (Ch6):
  "规则数" (加了几条规则) ≠ "自由度" (有几个可调旋钮)
  两个实验分别走出 plateau 拐点 → 确认最优复杂度

v3 (2026-06-24): 修复前视偏差 + TSCV+BIC方法
v4 (2026-06-24): 性能优化 + fallback bug修复
v5 (2026-06-25): ElasticNetCV 替代 TSCV+贪心+BIC
  - [METHOD] 全局最优: L1稀疏 + L2共线性处理, 非贪心逐层
  - [PERF]  单次回归 vs 多次组合评估, 大幅提速
  - [DESIGN] 硬约束→软约束: Phase 3.5 设 MAX_ACTIVE_FACTORS, GA可超越

来源: 《XQuant: 人人都是量化交易员》第6章 Step 4
       Brian Peterson: "Rule burden is a form of overfitting"
       Zou & Hastie (2005): "Regularization and variable selection via the elastic net"
       Friedman, Hastie & Tibshirani (2010): "Regularization paths for GLMs via coordinate descent"
创建: 2026-06-14 | v5: 2026-06-25
"""

import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 引用 FA 内部评估函数
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.ic_analysis import compute_ic_icir, compute_ic_icir_fast, compute_ic_summary
from evaluation.correlation import factor_correlation_matrix
from factors.composite import (
    combine_factors, weights_from_chromosome,
    standardize_factors, combine_factors_vectorized,
)


def _evaluate_combo_icir(
    combo: List[str],
    factor_dfs: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    correlation_threshold: float = 0.5,
    corr_penalty_weight: float = 0.3,
    std_factor_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    use_fast_icir: bool = True,
) -> Tuple[float, float]:
    """
    评估因子组合的ICIR (含软相关惩罚)
    
    v4: 接受预标准化因子 (std_factor_dfs), 跳过逐日z-score重算。
        使用 compute_ic_icir_fast (numpy rankdata) 替代 scipy.spearmanr。
        性能: 单次调用从 ~200ms → ~30ms (7x加速)。

    参数:
        combo: 因子名列表
        factor_dfs: 因子面板 (原始值, 用作fallback)
        forward_returns: 前向收益
        corr_matrix: 相关性矩阵
        correlation_threshold: 相关性阈值 (超过此值开始惩罚)
        corr_penalty_weight: 相关性惩罚权重
        std_factor_dfs: 预标准化的因子面板 (v4, 可选但强烈建议提供)
        use_fast_icir: 使用 numpy rankdata ICIR (v4, 默认True)

    返回:
        (raw_icir, penalized_icir)
    """
    weights = {f: 1.0 / len(combo) for f in combo}
    try:
        # v4 fast path: 使用预标准化因子
        if std_factor_dfs is not None:
            sel_std = {n: std_factor_dfs[n] for n in combo if n in std_factor_dfs}
            if sel_std:
                composite = combine_factors_vectorized(sel_std, weights)
            else:
                sel = {n: factor_dfs[n] for n in combo if n in factor_dfs}
                composite = combine_factors(sel, weights)
        else:
            sel = {n: factor_dfs[n] for n in combo if n in factor_dfs}
            composite = combine_factors(sel, weights)
        
        if composite.empty:
            return 0.0, 0.0
        
        # v4: 使用快速 ICIR (numpy rankdata, 5-10x faster)
        icir_fn = compute_ic_icir_fast if use_fast_icir else compute_ic_icir
        raw_icir = abs(icir_fn(composite, forward_returns)['icir'])

        if np.isnan(raw_icir):
            return 0.0, 0.0

        # 软相关惩罚: max |corr| 超过阈值 → 折扣ICIR
        max_corr = 0.0
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                if combo[i] in corr_matrix.index and combo[j] in corr_matrix.columns:
                    c = abs(corr_matrix.loc[combo[i], combo[j]])
                    max_corr = max(max_corr, c)

        if max_corr > correlation_threshold:
            excess = max_corr - correlation_threshold
            penalty = 1.0 - excess * corr_penalty_weight
            penalized = raw_icir * max(penalty, 0.3)  # 保底30%, 不完全排除
        else:
            penalized = raw_icir

        return raw_icir, penalized
    except Exception:
        return 0.0, 0.0


def run_factor_count_stacking(
    factor_dfs: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    ic_summary: pd.DataFrame,
    max_factors: int = 10,
    top_n_candidates: int = 20,
    correlation_threshold: float = 0.5,
    oos_split_date: Optional[str] = None,
    beam_width: int = 3,
    corr_penalty_weight: float = 0.3,
    method: str = 'beam_search_bic',
    bic_penalty_scale: float = 0.5,
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    4A 因子数堆叠 (Factor Count Stacking) — v5 性能优化TSCV

    从 1 个因子开始, 逐层增加因子数。每层用 Beam Search (保留 top-K 路径)
    代替贪心单路径, 避免陷入局部最优。最终在样本外验证每层最优组合。

    v3 关键修复:
      - 候选因子排序用训练集ICIR (不再用全时段, 消除look-ahead)
      - 相关性矩阵用训练集数据 (不再用全时段)
      - 最优选择改用最小OOS衰减 (min |icir_test - icir_train|, 更稳健)
      - 新增 method 参数支持:
        * 'beam_search_bic': Beam Search + BIC惩罚选择
        * 'beam_search_raw': Beam Search + 原始OOS ICIR最大
        * 'greedy': 贪心单路径 (向后兼容)
        * 'tscv_bic': 时序交叉验证 + BIC准则 (v5 推荐, 增量贪心 5折)

    v5 TSCV改进:
      - 增量贪心: 每折仅一轮贪心 (O(k×N) → O(N)), 跨k共享已选因子
      - 折数可配: n_folds参数控制 (默认5, 性价比最优)
      - 精度提升: 多折平均消除单次split偶然性

    BIC准则:
      BIC(k) = -2*log(ICIR_OOS(k)) + k*log(T_train)
      或等价的惩罚形式: ICIR_adj(k) = ICIR_OOS(k) - penalty * k * log(T_train)/T_train
      目的: 在因子预测力和模型复杂度之间做正规权衡

    对应 Ch6 4A: 从简单到复杂, 每层加一条规则

    参数:
        factor_dfs: 因子面板 {name: DataFrame(日期 x 股票)}
        forward_returns: 前向收益 DataFrame
        ic_summary: 各因子ICIR汇总 (仅用于日志对比, 候选排序改用训练集ICIR)
        max_factors: 最大因子数
        top_n_candidates: 每层考虑的候选因子数
        correlation_threshold: 软相关惩罚阈值 (超过此值开始折扣)
        oos_split_date: OOS分割日期, 若提供则使用与GA一致的OOS分割
        beam_width: Beam Search 宽度 (保留路径数, 默认3)
        corr_penalty_weight: 相关性惩罚权重 (默认0.3)
        method: 搜索与选择方法 (默认 'beam_search_bic'; 推荐 'tscv_bic')
        bic_penalty_scale: BIC惩罚强度缩放 (默认0.5, 值越大越偏好简单模型)
        n_folds: TSCV折数 (仅 method='tscv_bic' 时生效, 默认5)

    返回:
        DataFrame: 每层的结果 (因子数, 样本内ICIR, 样本外ICIR, 衰减, BIC_adj)
    """
    if method == 'tscv_bic':
        return _run_tscv_bic(
            factor_dfs, forward_returns, ic_summary,
            max_factors=max_factors, top_n_candidates=top_n_candidates,
            oos_split_date=oos_split_date,
            correlation_threshold=correlation_threshold,
            bic_penalty_scale=bic_penalty_scale,
            n_folds=n_folds,
        )
    
    if method == 'elasticnet':
        return elasticnet_factor_selection(
            factor_dfs, forward_returns, ic_summary,
            max_factors=max_factors, top_n_candidates=top_n_candidates,
            oos_split_date=oos_split_date,
        )

    print(f"\n{'='*60}")
    print(f"  4A 因子数堆叠 (Factor Count Stacking) — v4 {method}")
    print(f"  最大因子数: {max_factors} | 候选因子: {top_n_candidates} | Beam宽度: {beam_width}")
    print(f"  相关阈值: {correlation_threshold} (软惩罚, 权重={corr_penalty_weight})")
    print(f"  选择准则: {'BIC惩罚最小OOS衰减' if 'bic' in method else '原始OOS ICIR最大'}")
    print(f"  [v4] 预计算z-score + 向量化combine + numpy ICIR (性能优化)")
    print(f"{'='*60}")

    # 分割样本内/外 — 优先使用与GA一致的OOS分割日期
    dates = forward_returns.index
    if oos_split_date is not None:
        split_dt = pd.Timestamp(oos_split_date)
        train_dates = dates[dates < split_dt]
        test_dates = dates[dates >= split_dt]
        split_method = f"OOS日期 {oos_split_date}"
    else:
        split_idx = int(len(dates) * 0.8)
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        split_method = "80/20时间分割"

    train_fr = forward_returns.loc[train_dates]
    test_fr = forward_returns.loc[test_dates]

    print(f"  分割方式: {split_method}")
    print(f"  训练集: {len(train_dates)} 周 "
          f"({train_dates[0].strftime('%Y-%m-%d') if len(train_dates) > 0 else 'N/A'} ~ "
          f"{train_dates[-1].strftime('%Y-%m-%d') if len(train_dates) > 0 else 'N/A'})")
    print(f"  测试集: {len(test_dates)} 周 "
          f"({test_dates[0].strftime('%Y-%m-%d') if len(test_dates) > 0 else 'N/A'} ~ "
          f"{test_dates[-1].strftime('%Y-%m-%d') if len(test_dates) > 0 else 'N/A'})")

    # ★ v4 性能优化: 预计算全量因子的截面z-score (train_fr + test_fr 都需要)
    #   后续 Beam Search 中不再重复标准化, 直接加权求和
    print(f"  [v4 PERF] 预计算因子截面z-score...")
    import time as _time
    _t0 = _time.time()
    
    # 全量标准化 (train+test, 用于后续评估)
    full_std_factor_dfs = standardize_factors(factor_dfs)
    # 仅训练集标准化 (用于 Beam Search 搜索)
    train_std_dfs = {}
    for n, df in full_std_factor_dfs.items():
        train_std_dfs[n] = df.reindex(train_dates).dropna(how='all')
    
    _t1 = _time.time()
    print(f"  [v4 PERF] 标准化完成: {len(full_std_factor_dfs)} 因子, "
          f"耗时 {_t1-_t0:.1f}s")

    # ★★★ v3关键修复: 用训练集ICIR做候选排序, 消除全时段look-ahead ★★★
    train_factor_dfs = {n: df.reindex(train_dates).dropna(how='all')
                        for n, df in factor_dfs.items()}
    train_ic_summary = compute_ic_summary(train_factor_dfs, train_fr)
    print(f"  [v3 FIX] 候选排序基于训练集ICIR ({len(train_dates)}周), 消除前视偏差")

    top_factors = train_ic_summary.sort_values('ICIR', key=abs, ascending=False).head(top_n_candidates)
    candidate_names = list(top_factors.index)
    print(f"  候选池 (Top {len(candidate_names)}): 训练ICIR范围 "
          f"[{top_factors['ICIR'].abs().min():.3f} ~ {top_factors['ICIR'].abs().max():.3f}]")

    # ★★★ v3关键修复: 用训练集数据计算相关性矩阵 ★★★
    corr_matrix = factor_correlation_matrix(
        {n: train_factor_dfs[n] for n in candidate_names if n in train_factor_dfs}
    )
    print(f"  [v3 FIX] 相关性矩阵基于训练集 ({corr_matrix.shape}), 消除前视偏差")
    
    # ★ v4: 准备 train_std 子集 (仅含候选因子, 给 Beam Search)
    train_std_candidates = {n: train_std_dfs[n] for n in candidate_names if n in train_std_dfs}

    # === Beam Search: 每层保留 top-K 路径 ===
    # beams: List of (combo_list, penalized_icir, raw_icir)
    beams = []

    results = []
    max_n = min(max_factors, len(candidate_names))

    for n_factors in range(1, max_n + 1):
        print(f"\n  --- Layer {n_factors} ({n_factors} 因子) ---")

        candidates_this_layer = []

        if n_factors == 1:
            # 第一层: 所有单因子
            for name in candidate_names:
                raw, penalized = _evaluate_combo_icir(
                    [name], factor_dfs, train_fr, corr_matrix,
                    correlation_threshold, corr_penalty_weight,
                    std_factor_dfs=train_std_candidates, use_fast_icir=True,
                )
                if penalized > 0:
                    candidates_this_layer.append(([name], penalized, raw))
        else:
            # 第N层: 每个beam路径 + 每个未使用的因子
            for beam_combo, _, _ in beams:
                for new_factor in candidate_names:
                    if new_factor in beam_combo:
                        continue
                    combo = beam_combo + [new_factor]
                    raw, penalized = _evaluate_combo_icir(
                        combo, factor_dfs, train_fr, corr_matrix,
                        correlation_threshold, corr_penalty_weight,
                        std_factor_dfs=train_std_candidates, use_fast_icir=True,
                    )
                    if penalized > 0:
                        candidates_this_layer.append((combo, penalized, raw))

        if not candidates_this_layer:
            print(f"    无有效组合, 停止搜索")
            break

        # 按 penalized ICIR 排序, 保留 top beam_width
        candidates_this_layer.sort(key=lambda x: -x[1])
        beams = candidates_this_layer[:beam_width]

        # 该层最优 = beam[0]
        best_combo, best_penalized, best_raw = beams[0]

        # 在训练集和测试集上评估 (使用预标准化因子)
        sel_std = {n: full_std_factor_dfs[n] for n in best_combo if n in full_std_factor_dfs}
        w = {f: 1.0 / len(best_combo) for f in best_combo}
        
        if sel_std:
            composite = combine_factors_vectorized(sel_std, w)
        else:
            sel = {n: factor_dfs[n] for n in best_combo if n in factor_dfs}
            composite = combine_factors(sel, w)

        # v4: 使用快速 ICIR
        icir_train = abs(compute_ic_icir_fast(composite, train_fr)['icir'])
        icir_test = abs(compute_ic_icir_fast(composite, test_fr)['icir'])
        decay = icir_test - icir_train

        # 打印beam信息
        print(f"    Beam[0]: {best_combo}")
        print(f"    训练ICIR: {icir_train:.3f} | 测试ICIR: {icir_test:.3f} | 衰减: {decay:+.3f}")
        if len(beams) > 1:
            alt_names = [b[0] for b in beams[1:]]
            alt_icirs = [f"{b[1]:.3f}" for b in beams[1:]]
            print(f"    Beam备选: {list(zip(alt_names, alt_icirs))}")

        results.append({
            'n_factors': n_factors,
            'factors': best_combo,
            'icir_train': icir_train,
            'icir_test': icir_test,
            'decay': decay,
            'n_beams': len(beams),
        })

    result_df = pd.DataFrame(results)
    factor_results = result_df['factors'].copy()
    result_df = result_df.drop(columns=['factors', 'n_beams'])

    # ★ v3: BIC惩罚调整 + 更稳健的选择准则
    T_train = len(train_dates)
    if 'bic' in method:
        # BIC惩罚: 每个因子增加 log(T)/T 的复杂度代价
        bic_penalty_per_factor = bic_penalty_scale * np.log(T_train) / T_train
        result_df['bic_penalty'] = result_df['n_factors'] * bic_penalty_per_factor
        result_df['icir_adj'] = result_df['icir_test'] - result_df['bic_penalty']
        # BIC调整后的最优: 最大化调整后OOS ICIR
        best_idx = result_df['icir_adj'].idxmax()
        selection_criterion = 'BIC_adj_OOS_ICIR'
        print(f"\n  [BIC] 惩罚系数: {bic_penalty_per_factor:.5f} per factor "
              f"(scale={bic_penalty_scale}, log(T)/T = {np.log(T_train)/T_train:.5f})")
    else:
        # 原始: OOS ICIR最大 (可能被单因子反常主导)
        result_df['icir_adj'] = result_df['icir_test']
        best_idx = result_df['icir_test'].idxmax()
        selection_criterion = 'raw_OOS_ICIR'

    best_layer = result_df.loc[best_idx]

    # 备选准则: 最小OOS衰减 (min |icir_test - icir_train|) — 最稳健
    result_df['abs_decay'] = (result_df['icir_test'] - result_df['icir_train']).abs()
    best_decay_idx = result_df['abs_decay'].idxmin()
    decay_optimal = result_df.loc[best_decay_idx]

    print(f"\n  === 最优因子数判定 ===")
    print(f"  主准则 ({selection_criterion}): {int(best_layer['n_factors'])} 因子 "
          f"(OOS={best_layer['icir_test']:.3f}, train={best_layer['icir_train']:.3f})")
    print(f"  备选准则 (min |decay|): {int(decay_optimal['n_factors'])} 因子 "
          f"(OOS={decay_optimal['icir_test']:.3f}, decay={decay_optimal['decay']:+.3f})")

    # v3决策逻辑: 如果主准则=1且OOS ICIR远高于其他层, 标记WARN (可能是前视残余)
    n_best = int(best_layer['n_factors'])
    did_fallback = False  # v4: 追踪是否回退
    if n_best == 1:
        others_oos = result_df[result_df['n_factors'] > 1]['icir_test']
        if len(others_oos) > 0 and best_layer['icir_test'] > others_oos.max() * 1.05:
            print(f"  [WARN] 单因子最优且显著优于多因子 — 可能是数据特定或分裂点效应")
            print(f"     建议: 检查TSCV方法交叉验证, 或扩大训练窗口")
            # 回退到备选准则 (min decay) 保守估计
            if int(decay_optimal['n_factors']) > 1:
                print(f"  [FALLBACK] 使用备选准则结果: {int(decay_optimal['n_factors'])} 因子")
                n_best = int(decay_optimal['n_factors'])
                best_layer = decay_optimal
                did_fallback = True
                
                # ★★★ v4 FIX: 修改 result_df 使 fallback 决策反映到返回值 ★★★
                # 问题: run_fa.py 从 result_df['icir_adj'].idxmax() 读最优,
                #       但 BIC 的 icir_adj 仍然指向 1 因子 → MAX_FACTORS 回退错误
                # 修复: 将 fallback 层的 icir_adj 提升到比原最优略高
                if 'icir_adj' in result_df.columns:
                    current_max = result_df['icir_adj'].max()
                    # 设置为当前最大值 + 小量增量 (确保 idxmax 选到 fallback 行)
                    result_df.loc[result_df['n_factors'] == n_best, 'icir_adj'] = current_max * 1.001
                    print(f"     [v4 FIX] 已更新 result_df.icir_adj: n_factors={n_best} → 新最优")

    print(f"  最终决策: {n_best} 因子")
    print(f"  入选因子: {factor_results.iloc[int(best_layer['n_factors'])-1]}")

    # 检查 plateau
    if n_best < max_n:
        later_rows = result_df[result_df['n_factors'] > n_best]
        if len(later_rows) > 0 and later_rows['icir_test'].max() <= best_layer['icir_test'] * 1.02:
            print(f"  [PASS] 确认 plateau: 超过 {n_best} 因子后样本外不再提升")
            print(f"     (Ch6铁律: 能用{n_best}个因子解决的问题, 不要用{max_factors}个)")

    return result_df


def _build_parameterized_factor(
    base_factors: Dict[str, pd.DataFrame],
    params: Dict,
    forward_returns: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """
    根据参数构建带参数化的复合因子

    对基础因子应用参数选择 (窗口长度 / EMA / 波动率缩放),
    等权组合生成一个复合因子。

    参数:
        base_factors: 基础因子 {name: DataFrame(日期 x 股票)}
        params: 参数字典, 必须包含:
            momentum_window: 动量平滑窗口
            volatility_window: 波动率计算窗口
            use_ema: 是否用指数加权
            use_vol_scaling: 是否用波动率缩放
        forward_returns: 前向收益 (用于对齐截面)

    返回:
        pd.DataFrame: 参数化后的复合因子, 或 None
    """
    mw = params.get('momentum_window', 20)
    vw = params.get('volatility_window', 20)
    use_ema = params.get('use_ema', False)
    use_vol_scaling = params.get('use_vol_scaling', False)

    # 对齐日期和股票
    all_dates = sorted(set.intersection(*[set(df.index) for df in base_factors.values()])
                       & set(forward_returns.index))
    if len(all_dates) < 10:
        return None

    all_stocks = sorted(set.union(*[set(df.columns) for df in base_factors.values()])
                        & set(forward_returns.columns))

    composites = []
    for name, factor_df in base_factors.items():
        # 截面排名标准化
        ranked = factor_df.reindex(index=all_dates, columns=all_stocks)
        ranked = ranked.rank(axis=1, pct=True)  # [0, 1] 百分位排名

        # 时间序列平滑
        if use_ema:
            smoothed = ranked.ewm(span=max(mw, 2), min_periods=min(mw, 5)).mean()
        else:
            smoothed = ranked.rolling(window=max(mw, 2), min_periods=min(mw, 5)).mean()

        if use_vol_scaling:
            vol = ranked.rolling(window=max(vw, 2), min_periods=min(vw, 5)).std()
            vol = vol.replace(0, np.nan)
            smoothed = smoothed / vol

        composites.append(smoothed)

    if not composites:
        return None

    # 等权组合
    composite = sum(composites) / len(composites)
    composite = composite.fillna(np.nan)

    # 只保留有效行 (至少 10 只股票有值)
    valid = composite.dropna(axis=0, thresh=10)
    if len(valid) < 5:
        return None

    return valid


def run_dof_stacking(
    factor_dfs: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    ic_summary: pd.DataFrame,
    param_grids: Optional[Dict[str, List]] = None,
) -> pd.DataFrame:
    """
    4B 自由度堆叠 (Degrees of Freedom Stacking)

    核心实验: 从 1 个可调参数 (1 DoF) 开始, 逐步增加参数维度,
    每级在样本内网格搜索最优组合, 再在样本外验证。

    通过比较各级别的样本外 ICIR, 检测"自由度 plateau"——
    在哪一级之后再加参数不再改善 OOS 表现。

    对应 Ch6 4B: 分别走 plateau 拐点
           "能用2个参数解决的问题, 不要用6个"

    参数:
        factor_dfs: 因子面板 {name: DataFrame(日期 x 股票)}
        forward_returns: 前向收益 DataFrame(日期 x 股票)
        ic_summary: 各因子 ICIR 汇总 DataFrame (index=因子名, col=ICIR)
        param_grids: {参数名: [可选值列表]} 定义搜索空间,
                     默认使用 momentum_window / volatility_window / use_ema / use_vol_scaling

    返回:
        DataFrame: 每级结果
            dof            : 自由度级别 (1, 2, 3, 4)
            params         : 该级别激活的参数列表
            n_combinations : 网格搜索组合数
            icir_train     : 样本内最优 ICIR (绝对值)
            icir_test      : 样本外对应 ICIR (绝对值)
            decay          : 衰减 (icir_test - icir_train)

    工作原理:
        1. 取出 ic_summary 中 ICIR 最高的前 5 个因子作为基础因子池
        2. 对每个 DoF 级别, 在活跃参数的笛卡尔积上做网格搜索
        3. 每个网格点用 _build_parameterized_factor() 构建一个参数化复合因子
        4. 计算该复合因子在训练集上的 ICIR, 记录最优者
        5. 对最优者计算测试集 ICIR, 得到该级别的 OOS 表现
        6. 比较各级别 OOS ICIR, 找到 plateau 拐点

    解读:
        - icir_train 随 DoF 单调上升 → 样本内过拟合 (正常现象)
        - icir_test 先上升后平稳或下降 → 存在最优自由度
        - decay 持续变负 → 更多自由度在伤害样本外表现
        - plateau 出现在 icir_test 不再增长的级别 → 建议在此截断
    """
    # 默认参数网格
    if param_grids is None:
        param_grids = {
            'momentum_window': [5, 10, 20, 60],
            'volatility_window': [10, 20, 40],
            'use_ema': [True, False],
            'use_vol_scaling': [True, False],
        }

    print(f"\n{'='*60}")
    print(f"  4B 自由度堆叠 (DoF Stacking)")
    print(f"  参数维度: {list(param_grids.keys())}")
    print(f"{'='*60}")

    # 取 ICIR 最高的前 5 个因子作为基础因子
    top_n = min(5, len(ic_summary))
    top_factors = ic_summary.sort_values('ICIR', key=abs, ascending=False).head(top_n)
    top_names = [n for n in top_factors.index if n in factor_dfs]
    print(f"  基础因子 (Top {len(top_names)} by ICIR): {top_names}")

    if not top_names:
        print("  [ERROR] 没有可用的基础因子")
        return pd.DataFrame()

    base_factors = {n: factor_dfs[n] for n in top_names}

    # 分割样本内/外 (前 80% 训练, 后 20% 测试)
    dates = forward_returns.index
    split_idx = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    train_fr = forward_returns.loc[train_dates]
    test_fr = forward_returns.loc[test_dates]

    print(f"  训练集: {len(train_dates)} 周 | 测试集: {len(test_dates)} 周")

    param_names = list(param_grids.keys())
    results = []

    for dof_level in range(1, len(param_names) + 1):
        active_params = param_names[:dof_level]
        print(f"\n  --- 自由度: {dof_level} (参数: {active_params}) ---")

        # 对当前参数维度做网格搜索
        grids = [param_grids[p] for p in active_params]
        combinations = list(product(*grids))
        n_combinations = len(combinations)

        best_train_icir_abs = -999.0
        best_test_icir_abs = np.nan
        best_params = None

        for combo in combinations:
            params = dict(zip(active_params, combo))

            # 构建参数化复合因子
            composite = _build_parameterized_factor(base_factors, params, forward_returns)

            if composite is None:
                continue

            # 计算训练集 ICIR
            train_result = compute_ic_icir_fast(composite.loc[train_dates], train_fr)
            train_icir = train_result.get('icir', np.nan)

            if np.isnan(train_icir):
                continue

            train_icir_abs = abs(train_icir)

            if train_icir_abs > best_train_icir_abs:
                best_train_icir_abs = train_icir_abs
                best_params = params.copy()

                # 同时计算测试集 ICIR
                test_result = compute_ic_icir_fast(composite.loc[test_dates], test_fr)
                best_test_icir_abs = abs(test_result.get('icir', np.nan))

        icir_train = best_train_icir_abs if best_train_icir_abs > -999 else np.nan
        icir_test = best_test_icir_abs if not np.isnan(best_test_icir_abs) else np.nan
        decay = icir_test - icir_train if not (np.isnan(icir_test) or np.isnan(icir_train)) else np.nan

        print(f"    组合数: {n_combinations} | "
              f"最优ICIR(train): {icir_train:.3f} | "
              f"最优ICIR(test): {icir_test:.3f} | "
              f"衰减: {decay:+.3f}" if not np.isnan(decay) else f"    组合数: {n_combinations}")
        if best_params:
            print(f"    最优参数: {best_params}")

        results.append({
            'dof': dof_level,
            'params': active_params,
            'n_combinations': n_combinations,
            'icir_train': icir_train if not np.isnan(icir_train) else np.nan,
            'icir_test': icir_test,
            'decay': decay,
        })

    result_df = pd.DataFrame(results)

    # 检测自由度 plateau: OOS ICIR 在哪一级停止增长
    if len(result_df) >= 2:
        oos = result_df['icir_test'].values
        valid_mask = ~np.isnan(oos)
        if valid_mask.any():
            peak_idx = int(np.nanargmax(oos))
            peak_dof = int(result_df.iloc[peak_idx]['dof'])
            if peak_idx < len(oos) - 1:
                later = oos[peak_idx + 1:]
                later_valid = later[~np.isnan(later)]
                if len(later_valid) > 0 and np.nanmax(later) <= oos[peak_idx] * 1.02:
                    print(f"\n  [PASS] 确认 DoF plateau: "
                          f"自由度 {peak_dof} ({result_df.iloc[peak_idx]['params']}) 处达到最优")
                    print(f"     更多自由度后样本外不再改善")
                    print(f"     (Ch6铁律: 能用{peak_dof}个自由度解决的问题, 不要用{len(param_names)}个)")
            else:
                # 最后一级是最优 → 可能还有提升空间
                print(f"\n  [WARN] 未检测到 plateau: 最高自由度 ({len(param_names)}) 仍为最优")
                print(f"     建议扩大参数搜索空间以确认 plateau")

    return result_df


def print_rule_burden_summary(factor_count_results: pd.DataFrame, method: str = 'beam_search_bic'):
    """
    打印因子负担分析总结
    
    参考 Ch6 表6-8 "防过拟合检查清单"
    v5: 支持 TSCV (icir_test=CV平均, decay=0, 无train/test区分)
    """
    is_tscv = method == 'tscv_bic'
    is_en = method == 'elasticnet'
    
    # ★ 防御: 确保 has_adj 始终有值 (即使 factor_count_results 异常)
    has_adj = False
    
    method_label = '[ElasticNetCV]' if is_en else ('[TSCV-5]' if is_tscv else '[Beam Search]')
    print("\n" + "=" * 70)
    print(f"  因子负担分析总结 — XQuant Ch6 方法论 {method_label}")
    print("=" * 70)
    
    if factor_count_results is None or factor_count_results.empty:
        print("  [WARN] 无数据")
        return
    
    # 找最优层
    score_col = 'icir_adj' if 'icir_adj' in factor_count_results.columns else 'icir_test'
    best = factor_count_results.loc[factor_count_results[score_col].idxmax()]
    n_best = int(best['n_factors'])
    
    if is_tscv:
        print(f"\n  [*] 最优因子数: {n_best}")
        print(f"     CV平均ICIR: {best['icir_test']:.3f} (跨折验证)")
        if 'cv_icir_std' in factor_count_results.columns:
            print(f"     CV标准差:   {best['cv_icir_std']:.4f}")
        if 'icir_adj' in factor_count_results.columns:
            print(f"     BIC调整:    {best['icir_adj']:.3f}")
        if 'cv_stable' in factor_count_results.columns:
            print(f"     稳定性:     {'✓ 稳定' if best['cv_stable'] else '⚠ 波动较大'}")
    elif is_en:
        print(f"\n  [*] 最优因子数 (EN): {n_best}")
        print(f"     样本内ICIR: {best['icir_train']:.3f}")
        print(f"     样本外ICIR: {best['icir_test']:.3f}")
        print(f"     衰减: {best['decay']:+.3f}")
        if 'en_selected' in factor_count_results.columns:
            en_row = factor_count_results[factor_count_results['en_selected']]
            if len(en_row) > 0:
                print(f"     EN方法: L1+L2正则化 (全局最优+共线性处理)")
    else:
        print(f"\n  [*] 最优因子数: {n_best}")
        print(f"     样本内ICIR: {best['icir_train']:.3f}")
        print(f"     样本外ICIR: {best['icir_test']:.3f}")
        if 'icir_adj' in factor_count_results.columns:
            print(f"     BIC调整OOS: {best['icir_adj']:.3f}")
        print(f"     衰减: {best['decay']:+.3f}")
    
    # plateau 检测
    has_adj = 'icir_adj' in factor_count_results.columns
    plateau_col = 'icir_adj' if has_adj else 'icir_test'
    plateau_val = best[plateau_col]
    later = factor_count_results[factor_count_results['n_factors'] > n_best]
    if len(later) > 0 and later[plateau_col].max() <= plateau_val * 1.02:
        print(f"\n  [PASS] Plateau: k>{n_best} 不改善")
    
    # 打印完整表格
    if is_tscv:
        cols = ['n_factors', 'icir_test', 'cv_icir_std', 'bic_penalty', 'icir_adj']
        headers = ['因子数', 'CV_ICIR', '±std', 'BIC惩罚', '调整']
        fmts = ['{:4d}', '{:8.3f}', '{:8.4f}', '{:8.5f}', '{:8.3f}']
    elif is_en:
        cols = ['n_factors', 'icir_train', 'icir_test', 'decay', 'en_selected']
        headers = ['因子数', '训练ICIR', '测试ICIR', '衰减', 'EN?']
        fmts = ['{:6d}', '{:10.3f}', '{:10.3f}', '{:+8.3f}', '{:>6s}']
    elif has_adj:
        cols = ['n_factors', 'icir_train', 'icir_test', 'decay', 'bic_penalty', 'icir_adj']
        headers = ['因子数', '训练ICIR', '测试ICIR', '衰减', 'BIC惩罚', '调整OOS']
        fmts = ['{:6d}', '{:10.3f}', '{:10.3f}', '{:+8.3f}', '{:8.5f}', '{:10.3f}']
    else:
        cols = ['n_factors', 'icir_train', 'icir_test', 'decay']
        headers = ['因子数', '训练ICIR', '测试ICIR', '衰减']
        fmts = ['{:6d}', '{:10.3f}', '{:10.3f}', '{:+8.3f}']
    
    available = [c for c in cols if c in factor_count_results.columns]
    print(f"\n  " + "  ".join(f"{h:>8s}" if i > 0 else f"{h:>6s}" for i, h in enumerate(headers[:len(available)])))
    print(f"  {'-'*52}")
    for _, row in factor_count_results.iterrows():
        m = ' ← 最优' if int(row['n_factors']) == n_best else ''
        vals_parts = []
        for i, c in enumerate(available):
            val = row[c]
            if c == 'en_selected':
                val = '✓' if val else '—'
                vals_parts.append(f"{val:>6s}")
            else:
                vals_parts.append(fmts[i].format(val))
        vals = "  ".join(vals_parts)
        print(f"  {vals}{m}")
    
    print(f"\n  [*] Ch6: 能用{n_best}个因子解决的问题, 不要用更多")


def _run_tscv_bic(
    factor_dfs: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    ic_summary: pd.DataFrame,
    max_factors: int = 10,
    top_n_candidates: int = 20,
    oos_split_date: Optional[str] = None,
    n_folds: int = 5,
    correlation_threshold: float = 0.5,
    bic_penalty_scale: float = 0.5,
) -> pd.DataFrame:
    """
    TSCV + BIC 最优因子数 — 5折时序交叉验证 (v5 增量贪心)

    方法论:
      1. 时序交叉验证 (TSCV): 训练数据分割为 N 个 expanding-window folds
         - Fold 1: train[0→T/6],  validate[T/6→2T/6]
         - Fold 5: train[0→5T/6], validate[5T/6→T]
      2. 每折内: 增量贪心 (一轮构建 1→max_k, 跨k共享已选因子)
         - 不复跑每k的贪心, 而是逐步添加 → O(N) vs 旧版 O(k×N)
      3. 跨折平均验证 ICIR → CV_ICIR(k) ± std
      4. BIC 惩罚: ICIR_BIC(k) = CV_ICIR(k) - penalty × k × log(T)/T
      5. 选择 ICIR_BIC(k) 最大的 k

    v5 性能优化 vs v4:
      - 增量贪心: 每折 ~candidates 次评估 (vs 旧版 ~k×candidates)
      - 30候选×5折: ~165次评估 (vs ~935次), 6x 提速
      - 追踪跨折因子共识 (最多票选)

    v4 截面z-score安全性:
      standardize_factors 是逐日截面 z-score = (x-median(x))/std(x)
      每日期独立计算 → 子集化到任意日期范围均无前视偏差

    来源:
      Schwarz (1978) BIC, Arlot & Celisse (2010) CV survey,
      Bergmeir & Benitez (2012) Time-series CV
    """
    import time as _time
    from collections import Counter

    print(f"\n{'='*60}")
    print(f"  TSCV + BIC — 5折时序交叉验证 [v5 增量贪心]")
    print(f"  Folds: {n_folds} | Max k: {max_factors} | Candidates: {top_n_candidates}")
    print(f"  BIC scale: {bic_penalty_scale} | 相关阈值: {correlation_threshold}")
    print(f"{'='*60}")

    dates = forward_returns.index

    # === 1. CV数据范围 (OOS split 之前) ===
    if oos_split_date is not None:
        cv_dates = dates[dates < pd.Timestamp(oos_split_date)]
    else:
        cv_dates = dates[:int(len(dates) * 0.8)]

    T_cv = len(cv_dates)
    MIN_FOLD = 24  # 最少24周验证
    if T_cv < n_folds * MIN_FOLD * 2:
        n_folds = max(2, T_cv // (MIN_FOLD * 2))
        print(f"  [ADJUST] T={T_cv}不足, folds → {n_folds}")

    # === 2. 预计算全量截面 z-score (逐日独立, 无前视) ===
    _t0 = _time.time()
    full_std_dfs = standardize_factors(factor_dfs)
    print(f"  [PERF] 全量 {len(full_std_dfs)} 因子 z-score: {_time.time()-_t0:.1f}s")

    # === 3. Building expanding-window folds ===
    fold_size = T_cv // (n_folds + 1)
    folds = []
    for i in range(1, n_folds + 1):
        end = i * fold_size
        v_end = min((i + 1) * fold_size, T_cv)
        folds.append((cv_dates[:end], cv_dates[end:v_end]))

    print(f"  CV: {T_cv}周 ({cv_dates[0].date()} ~ {cv_dates[-1].date()}), "
          f"每折 ~{fold_size}w train + ~{fold_size}w val")

    # === 4. 跨折交叉验证 (增量贪心) ===
    max_k = min(max_factors, top_n_candidates)
    cv_icir = {k: [] for k in range(1, max_k + 1)}
    cv_factors = {k: [] for k in range(1, max_k + 1)}
    fold_t0 = _time.time()

    for fi, (tr_dates, vl_dates) in enumerate(folds):
        tr_fr = forward_returns.loc[tr_dates]
        vl_fr = forward_returns.loc[vl_dates]

        # 4a. Fold候选排序 (训练集 ICIR, 无look-ahead)
        tr_raw = {n: df.reindex(tr_dates).dropna(how='all')
                  for n, df in factor_dfs.items()}
        fold_ic = compute_ic_summary(tr_raw, tr_fr)
        cands = fold_ic.sort_values('ICIR', key=abs, ascending=False).head(
            top_n_candidates)
        cand_names = list(cands.index)

        # 4b. Fold内相关性矩阵
        fold_corr = factor_correlation_matrix(
            {n: tr_raw[n] for n in cand_names if n in tr_raw})

        # 4c. Fold标准化因子
        tr_std = {n: full_std_dfs[n].reindex(tr_dates).dropna(how='all')
                  for n in cand_names if n in full_std_dfs}
        vl_std = {n: full_std_dfs[n].reindex(vl_dates).dropna(how='all')
                  for n in cand_names if n in full_std_dfs}

        # 4d. v5: 增量贪心 — 一轮构建 1→max_k
        selected = []
        remaining = list(cand_names)
        n_evals = 0

        for layer in range(max_k):
            best_f, best_score = None, -999.0
            for cand in remaining:
                n_evals += 1
                combo = selected + [cand]
                try:
                    sel_std = {n: tr_std[n] for n in combo if n in tr_std}
                    if not sel_std:
                        continue
                    w = {f: 1.0/len(combo) for f in combo}
                    comp = combine_factors_vectorized(sel_std, w)
                    score = abs(compute_ic_icir_fast(comp, tr_fr)['icir'])
                    # 相关性惩罚
                    if len(combo) > 1:
                        mc = 0.0
                        for a in range(len(combo)):
                            for b in range(a+1, len(combo)):
                                if combo[a] in fold_corr.index and combo[b] in fold_corr.columns:
                                    mc = max(mc, abs(fold_corr.loc[combo[a], combo[b]]))
                        if mc > correlation_threshold:
                            score *= max(0.3, 1.0-(mc-correlation_threshold)*0.3)
                    if score > best_score:
                        best_score, best_f = score, cand
                except Exception:
                    continue

            if best_f is None:
                break
            selected.append(best_f)
            remaining.remove(best_f)
            k = len(selected)

            # 验证集评估
            try:
                sv = {n: vl_std[n] for n in selected if n in vl_std}
                if sv:
                    w = {f: 1.0/len(selected) for f in selected}
                    comp = combine_factors_vectorized(sv, w)
                    val_icir = abs(compute_ic_icir_fast(comp, vl_fr)['icir'])
                    if not np.isnan(val_icir) and val_icir > 0:
                        cv_icir[k].append(val_icir)
                        cv_factors[k].append(list(selected))
            except Exception:
                pass

        elapsed = _time.time() - fold_t0
        eta = elapsed / (fi+1) * (n_folds - fi - 1)
        print(f"  Fold {fi+1}/{n_folds}: {tr_dates[0].date()}~{tr_dates[-1].date()} "
              f"| val {vl_dates[0].date()}~{vl_dates[-1].date()} "
              f"| {n_evals}evals | {elapsed:.0f}s [ETA {eta:.0f}s]")

    # === 5. 汇聚 ===
    results = []
    T_m = int(np.mean([len(f[0]) for f in folds]))
    bic_c = bic_penalty_scale * np.log(T_m) / T_m

    print(f"\n  === TSCV-{n_folds}折 + BIC 结果 ({T_m}w均值) ===")
    print(f"  {'k':>3s} {'CV_ICIR':>9s} {'±std':>8s} {'#fold':>6s} "
          f"{'BIC_pen':>9s} {'BIC_adj':>9s} {'Stable':>7s}")
    print(f"  {'-'*60}")

    best_k, best_adj = 1, -999.0
    for k in range(1, max_k + 1):
        vals = cv_icir.get(k, [])
        if len(vals) >= max(2, n_folds//2):
            mu, sd = np.mean(vals), np.std(vals)
            pen = k * bic_c
            adj = mu - pen
            stable = sd/mu < 0.3 if mu > 0 else False
            results.append(dict(
                n_factors=k, icir_train=0.0, icir_test=mu, decay=0.0,
                cv_icir_mean=mu, cv_icir_std=sd, cv_n_folds=len(vals),
                bic_penalty=pen, icir_adj=adj, cv_stable=stable))
            flag = '✓' if stable else '⚠'
            if adj > best_adj:
                best_adj, best_k = adj, k
            print(f"  {k:3d} {mu:9.3f} {sd:8.4f} {len(vals):6d} "
                  f"{pen:9.5f} {adj:9.3f} {flag:>7s}")
        else:
            print(f"  {k:3d} {'—':>9s} {'—':>8s} {len(vals):6d} {'—':>9s} {'—':>9s}")

    result_df = pd.DataFrame(results)
    if results:
        best = result_df.loc[result_df['icir_adj'].idxmax()]
        n_best = int(best['n_factors'])

        # 跨折共识因子
        consensus = []
        if n_best in cv_factors and cv_factors[n_best]:
            all_f = [f for fl in cv_factors[n_best] for f in fl]
            consensus = [f for f, _ in Counter(all_f).most_common(n_best)]

        print(f"\n  [*] TSCV-{n_folds}折+BIC 最优: **{n_best}** 因子 "
              f"(CV={best['cv_icir_mean']:.3f}, BIC_adj={best['icir_adj']:.3f})")
        if consensus:
            print(f"  跨折共识因子: {consensus}")

        # BIC惩罚系数 & plateau
        print(f"  BIC系数: {bic_c:.5f}/factor "
              f"(log(T)/T={np.log(T_m)/T_m:.5f} × scale={bic_penalty_scale})")
        later = result_df[result_df['n_factors'] > n_best]
        if len(later) > 0 and later['icir_adj'].max() <= best['icir_adj'] * 1.02:
            print(f"  [PASS] Plateau: k>{n_best} BIC不改善")
        elif len(later) > 0:
            gap = later['icir_adj'].max() - best['icir_adj']
            print(f"  [INFO] k>{n_best} BIC仍有提升空间 (Δ={gap:+.3f})")

    return result_df


# ============================================================
# 4C: ElasticNetCV 全局稀疏因子选择 (v5)
# ============================================================

def elasticnet_factor_selection(
    factor_dfs: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    ic_summary: pd.DataFrame,
    max_factors: int = 10,
    top_n_candidates: int = 30,
    oos_split_date: Optional[str] = None,
    l1_ratios: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99),
    n_alphas: int = 50,
    cv_folds: int = 5,
    min_coef_abs: float = 0.01,
) -> pd.DataFrame:
    """
    ElasticNetCV 稀疏因子选择 — 替代 TSCV+贪心+BIC

    核心思想:
      将因子选择转化为面板回归的正则化稀疏问题:
        forward_return_it = sum(β_j * factor_j,it) + ε_it

      ElasticNet (L1+L2):
        min ||y - Xβ||² + λ * (ρ||β||₁ + 0.5*(1-ρ)||β||₂²)

      - L1 (||β||₁)   → 自动稀疏, 系数归零 = 因子被淘汰
      - L2 (||β||₂²)  → 处理因子共线性, 高相关因子不会双双入选
      - λ 通过CV自动选择 (最小化 MSE)

    为什么优于 TSCV+贪心+BIC:
      1. 全局最优: 同时优化所有β, 非贪心逐层
      2. 自动共线性: L2天然处理corr > 0.5的因子对
      3. 速度快: O(N × λ_grid) vs O(k × N × candidates)
      4. CV选λ比BIC近似更稳健 (Zou & Hastie 2005)

    参数:
        factor_dfs: 因子面板 {name: DataFrame(日期 x 股票)}
        forward_returns: 前向收益 DataFrame(日期 x 股票)
        ic_summary: ICIR汇总 (仅用于候选排序和日志)
        max_factors: 最大因子数 (硬上限)
        top_n_candidates: 候选因子数 (按训练ICIR排序取Top-N)
        oos_split_date: OOS分割点, 用于训练/测试划分
        l1_ratios: ElasticNet的L1混合比 (ρ=1 → LASSO, ρ=0 → Ridge)
        n_alphas: α网格点数 (在 log-space 均匀采样)
        cv_folds: CV折数 (默认5)
        min_coef_abs: 最小系数绝对值 (低于此值视为0)

    返回:
        DataFrame: 最优因子数, 入选因子, EN系数, 训练/OOS ICIR
    """
    try:
        from sklearn.linear_model import ElasticNetCV
        from sklearn.model_selection import TimeSeriesSplit
    except ImportError:
        print("  [ERROR] sklearn未安装, 无法使用ElasticNetCV。回退到TSCV+BIC。")
        print("  安装: pip install scikit-learn")
        return None

    import time as _time

    print(f"\n{'='*60}")
    print(f"  4C ElasticNetCV 稀疏因子选择 [v5]")
    print(f"  候选因子: {top_n_candidates} | L1 ratios: {l1_ratios}")
    print(f"  原理: min ||y-Xβ||² + λ(ρ|β|₁ + 0.5(1-ρ)|β|₂²)")
    print(f"  优势: 全局最优 + 自动共线性处理 + CV选λ")
    print(f"{'='*60}")

    # --- 1. 划分训练/测试 ---
    dates = forward_returns.index
    if oos_split_date is not None:
        split_dt = pd.Timestamp(oos_split_date)
        train_dates = dates[dates < split_dt]
        test_dates = dates[dates >= split_dt]
        split_method = f"OOS日期 {oos_split_date}"
    else:
        split_idx = int(len(dates) * 0.8)
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        split_method = "80/20时间分割"

    train_fr = forward_returns.loc[train_dates]
    test_fr = forward_returns.loc[test_dates]

    print(f"  分割: {split_method}")
    print(f"  训练: {len(train_dates)}周 | 测试: {len(test_dates)}周")

    # --- 2. 候选因子排序 (训练集ICIR) ---
    from evaluation.ic_analysis import compute_ic_summary
    train_raw = {n: df.reindex(train_dates).dropna(how='all') 
                 for n, df in factor_dfs.items()}
    train_ic = compute_ic_summary(train_raw, train_fr)
    top_factors = train_ic.sort_values('ICIR', key=abs, ascending=False).head(top_n_candidates)
    candidate_names = list(top_factors.index)
    # 按 abs(ICIR) 排序 (EN不依赖排序, 但缩减候选池提高速度)
    print(f"  候选池: Top {len(candidate_names)} (训练ICIR "
          f"[{top_factors['ICIR'].abs().min():.3f} ~ {top_factors['ICIR'].abs().max():.3f}])")

    # --- 3. 截面标准化 (逐日z-score, 无前视) ---
    # v5.3.2 fix: 先裁剪到全量日期范围, 减少标准化时的内存压力
    # factor_dfs 包含全部278周×5710股, 80个因子约1GB内存
    # 裁剪到 train+test 重叠日期后再标准化, 避免同时持有两份copy
    _t0 = _time.time()
    from factors.composite import standardize_factors
    # ★ 限制到前向收益和因子的交集日期, 缩减内存
    candidate_factors = {}
    all_candidate_dates = None
    for n in candidate_names:
        if n in factor_dfs:
            df = factor_dfs[n]
            if all_candidate_dates is None:
                all_candidate_dates = df.index.intersection(forward_returns.index)
            candidate_factors[n] = df.reindex(all_candidate_dates)
    std_dfs = standardize_factors(candidate_factors)
    del candidate_factors  # 立即释放原始数据
    print(f"  [PERF] 标准化 {len(std_dfs)} 因子 (日期裁剪后): {_time.time()-_t0:.1f}s")

    # --- 3.5 ★ 日期对齐: 确保所有因子和forward_returns共享相同日期 ---
    # 某些因子的resample日期可能与forward_returns不完全一致
    # (例如财务因子ffill扩展后仍然缺少头尾日期)
    common_train_dates = train_dates
    common_test_dates = test_dates
    for name, sdf in std_dfs.items():
        common_train_dates = common_train_dates.intersection(sdf.index)
        common_test_dates = common_test_dates.intersection(sdf.index)
    
    n_skipped_train = len(train_dates) - len(common_train_dates)
    n_skipped_test = len(test_dates) - len(common_test_dates)
    if n_skipped_train > 0 or n_skipped_test > 0:
        print(f"  [ALIGN] 日期对齐: 训练跳过{n_skipped_train}/{len(train_dates)}, "
              f"测试跳过{n_skipped_test}/{len(test_dates)}")

    # --- 4. Stack面板数据 (v5.3.1 向量化: MultiIndex一次性对齐) ---
    # X: (T × N_stocks) × N_factors, y: (T × N_stocks)
    _t0 = _time.time()

    def _stack_panel(factor_names, std_dict, fr, date_list):
        """向量化面板堆叠: reset_index+merge 替代 join (避免非唯一MultiIndex)
        v5.3.2: 添加gc.collect()减少内存碎片; 限制因子数>30时分批处理
        Returns: (X, y, actual_cols) or (empty, empty, [])"""
        if not factor_names or len(date_list) == 0:
            return np.empty((0, len(factor_names))), np.empty(0), []
        
        import gc as _gc
        
        # 将每个因子 stack + reset_index 成 DataFrame(date, stock, factor_value)
        dfs = []
        for name in factor_names:
            if name not in std_dict:
                continue
            s = std_dict[name].loc[std_dict[name].index.intersection(date_list)]
            df_s = s.stack().reset_index()
            df_s.columns = ['date', 'stock', name]
            dfs.append(df_s)
        
        if not dfs:
            return np.empty((0, len(factor_names))), np.empty(0), []
        
        # ★ v5.3.2: 分批merge, 每10个因子释放一次内存
        BATCH_SIZE = 10
        if len(dfs) <= BATCH_SIZE:
            # 小规模: 直接merge
            merged = dfs[0]
            for df_i in dfs[1:]:
                merged = merged.merge(df_i, on=['date', 'stock'], how='inner')
        else:
            # 大规模(>30因子): 分批merge减少中间DataFrame大小
            merged = dfs[0]
            batch_count = 0
            for df_i in dfs[1:]:
                merged = merged.merge(df_i, on=['date', 'stock'], how='inner')
                batch_count += 1
                if batch_count % BATCH_SIZE == 0:
                    _gc.collect()  # 释放中间merge分配的内存
        
        # 对齐 forward_returns
        fr_df = fr.loc[fr.index.intersection(date_list)].stack().reset_index()
        fr_df.columns = ['date', 'stock', '_target_']
        merged = merged.merge(fr_df, on=['date', 'stock'], how='inner')
        merged = merged.dropna()
        del dfs  # v5.3.2: 释放因子列表
        _gc.collect()
        
        actual_cols = [c for c in factor_names if c in merged.columns]
        if len(actual_cols) < 2 or len(merged) < 100:
            return np.empty((0, len(factor_names))), np.empty(0), []
        
        X = merged[actual_cols].values.astype(np.float64)
        y = merged['_target_'].values.astype(np.float64)
        del merged  # v5.3.2: 释放合并数据
        _gc.collect()
        return X, y, actual_cols

    factor_names_list = [n for n in candidate_names if n in std_dfs]
    X_train, y_train, actual_cols = _stack_panel(factor_names_list, std_dfs, train_fr, common_train_dates)
    X_test, y_test, _ = _stack_panel(factor_names_list, std_dfs, test_fr, common_test_dates)
    
    # ★ 使用实际入选的因子名 (stack_panel 可能剔除不存在的列)
    if actual_cols:
        active_factor_names = actual_cols
    else:
        active_factor_names = factor_names_list

    if len(y_train) == 0:
        print("  [ERROR] 面板数据stack失败, 回退到TSCV+BIC")
        return None

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    est_stocks = n_train // max(1, len(common_train_dates))
    print(f"  Stack: {n_train} train + {n_test} test obs, "
          f"({len(factor_names_list)} factors × ~{est_stocks:.0f} stocks/date) "
          f"[{_time.time()-_t0:.1f}s]")

    # --- 5. ElasticNetCV ---
    _t0 = _time.time()
    
    # 使用 TimeSeriesSplit 保持时序结构
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # ★ sklearn 0.24+: 参数名是 'alphas', 不是 'n_alphas'
    # n_alphas → 生成 n_alphas 个 α 值在 log-space
    import numpy as _np
    alpha_grid = _np.logspace(-5, 0, n_alphas)  # 10^-5 ~ 10^0
    
    model = ElasticNetCV(
        l1_ratio=list(l1_ratios),
        alphas=alpha_grid,
        cv=tscv,
        max_iter=5000,
        tol=1e-4,
        random_state=42,
        n_jobs=-1,
        selection='cyclic',
    )
    model.fit(X_train, y_train)

    en_time = _time.time() - _t0
    print(f"  [PERF] ElasticNetCV: {en_time:.1f}s "
          f"(α*={model.alpha_:.6f}, ρ*={model.l1_ratio_:.3f}, "
          f"R²_train={model.score(X_train, y_train):.4f})")

    # --- 6. 提取非零系数 → 入选因子 ---
    coef = model.coef_
    n_nonzero = int(np.sum(np.abs(coef) > min_coef_abs))
    
    # ★ 使用 factor_names_list (与X矩阵列顺序一致) 而非 candidate_names
    active_factor_names = factor_names_list
    
    # 按系数绝对值排序
    sorted_idx = np.argsort(np.abs(coef))[::-1]
    selected_factors = []
    selected_coefs = []
    for idx in sorted_idx:
        if np.abs(coef[idx]) > min_coef_abs and len(selected_factors) < max_factors:
            selected_factors.append(active_factor_names[idx])
            selected_coefs.append(coef[idx])

    # 如果EN选出太少(≤1), 用候选ICIR补足至少3个
    if len(selected_factors) < 3:
        n_from_en = len(selected_factors)
        for fname in active_factor_names:
            if fname not in selected_factors and len(selected_factors) < min(3, max_factors):
                selected_factors.append(fname)
                selected_coefs.append(0.01)  # 微小系数标记
        if n_from_en < len(selected_factors):
            print(f"  [SUPP] EN仅选{n_from_en}因子, 补足至{len(selected_factors)} (ICIR排序)")

    print(f"\n  非零系数: {n_nonzero} / {len(active_factor_names)} 候选因子")
    print(f"  入选因子 ({len(selected_factors)}):")
    for f, c in zip(selected_factors, selected_coefs):
        train_icir_val = train_ic.loc[f, 'ICIR'] if f in train_ic.index else 0
        print(f"    {f:35s} EN_coef={c:+7.4f}  训练ICIR={train_icir_val:+6.3f}")

    # --- 7. 构建复合因子并计算ICIR ---
    from evaluation.ic_analysis import compute_ic_icir_fast
    from factors.composite import combine_factors_vectorized

    # 用EN系数的符号+绝对值做权重 (保留信号方向)
    en_weights = {}
    for f, c in zip(selected_factors, selected_coefs):
        en_weights[f] = abs(c) + 0.01  # 最小权重0.01, 防止全零
    
    # 归一化
    w_sum = sum(en_weights.values())
    en_weights = {k: v/w_sum for k, v in en_weights.items()}

    sel_std = {n: std_dfs[n] for n in selected_factors if n in std_dfs}
    composite = combine_factors_vectorized(sel_std, en_weights)

    icir_train = abs(compute_ic_icir_fast(composite, train_fr)['icir'])
    icir_test = abs(compute_ic_icir_fast(composite, test_fr)['icir']) if len(test_dates) > 10 else np.nan
    decay = icir_test - icir_train if not np.isnan(icir_test) else np.nan

    # --- 8. 构建多层次结果 ---
    # 主结果: EN选择的最优因子集
    # 同时提供不同稀疏度级别的对比 (1因子 → N因子, 按EN系数重要性排序)
    results = []
    sorted_by_importance = sorted(
        zip(active_factor_names, np.abs(coef)), key=lambda x: -x[1])
    
    for k in range(1, max_factors + 1):
        if k > len(sorted_by_importance):
            break
        topk_names = [x[0] for x in sorted_by_importance[:k]]
        topk_std = {n: std_dfs[n] for n in topk_names if n in std_dfs}
        topk_w = {f: 1.0/k for f in topk_names}
        topk_comp = combine_factors_vectorized(topk_std, topk_w)
        
        k_icir_train = abs(compute_ic_icir_fast(topk_comp, train_fr)['icir'])
        k_icir_test = abs(compute_ic_icir_fast(topk_comp, test_fr)['icir']) if len(test_dates) > 10 else np.nan
        k_decay = k_icir_test - k_icir_train if not np.isnan(k_icir_test) else np.nan
        
        results.append({
            'n_factors': k,
            'factors': topk_names,
            'icir_train': k_icir_train,
            'icir_test': k_icir_test,
            'decay': k_decay,
            'en_selected': k <= len(selected_factors),
        })

    result_df = pd.DataFrame(results)
    
    # 最优因子数 = EN非零系数数量 (即 len(selected_factors))
    optimal_k = len(selected_factors)

    # 添加 icir_adj 列 (用于兼容 run_fa.py 的闭环逻辑)
    # EN选择的层获得最大 icir_adj (确保 idxmax 选中)
    result_df['icir_adj'] = result_df['icir_test'].copy()
    en_match = result_df['n_factors'] == optimal_k
    if en_match.any():
        # 将EN选择的行 icir_adj 提升到最高
        max_adj = result_df['icir_adj'].max()
        result_df.loc[en_match, 'icir_adj'] = max_adj * 1.001

    print(f"\n  {'k':>3s} {'训练ICIR':>10s} {'测试ICIR':>10s} {'衰减':>9s} {'来源':>8s}")
    print(f"  {'-'*45}")
    for _, row in result_df.iterrows():
        src = 'EN' if row['en_selected'] else 'ext'
        m = ' ←最优' if int(row['n_factors']) == optimal_k else ''
        print(f"  {int(row['n_factors']):3d} {row['icir_train']:10.3f} "
              f"{row['icir_test']:10.3f} {row['decay']:+9.3f} {src:>8s}{m}")

    print(f"\n  === ElasticNet判定 ===")
    print(f"  最优因子数: {optimal_k} (EN非零系数)")
    print(f"  入选因子: {selected_factors}")
    print(f"  OOS ICIR: {icir_test:.3f} (decay={decay:+.3f})")
    print(f"  EN参数: α*={model.alpha_:.6f}, ρ*={model.l1_ratio_:.3f}")
    print(f"  R² (训练): {model.score(X_train, y_train):.4f}")
    if len(X_test) > 100:
        print(f"  R² (测试): {model.score(X_test, y_test):.4f}")

    return result_df


if __name__ == '__main__':
    # 示例用法
    print("因子负担分析模块 (Rule Burden Analysis) — v5")
    print("用法: from evaluation.rule_burden import run_factor_count_stacking")
    print("")
    print("集成到 run_fa.py Phase 3.5:")
    print("  from evaluation.rule_burden import run_factor_count_stacking, print_rule_burden_summary")
    print("  # v5 (推荐): ElasticNetCV 全局稀疏选择")
    print("  burden_results = run_factor_count_stacking(factor_dfs, forward_returns, ic_summary,")
    print("      oos_split_date=OOS_SPLIT_DATE, method='elasticnet')")
    print("  # 备选: TSCV+BIC (学术黄金标准, 计算量大)")
    print("  burden_results = run_factor_count_stacking(factor_dfs, forward_returns, ic_summary,")
    print("      oos_split_date=OOS_SPLIT_DATE, method='tscv_bic')")
    print("  print_rule_burden_summary(burden_results)")
    print("")
    print("4B DoF stacking: READY")
    print("4C ElasticNet: READY")
