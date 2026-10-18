"""
Walk-forward 因子稳定性评估 — XQuant Ch6 方法论
================================================
实现锚定式(Anchored)和滚动式(Rolling) Walk-forward 分析,
评估因子组合的跨窗口参数稳定性。

核心机制 (Ch6 Step 2):
  1. 锚定式 (Anchored WF): 训练窗口从起点开始不断扩展
     - 窗口1: [start, split1] 训练 → [split1, split2] 测试
     - 窗口2: [start, split2] 训练 → [split2, split3] 测试
     - 每一步向前添加数据, 训练窗口越来越大

  2. 滚动式 (Rolling WF): 训练窗口固定长度向前滑动
     - 窗口1: [d1, dN] 训练 → [dN+1, dN+M] 测试
     - 窗口2: [d2, dN+1] 训练 → [dN+2, dN+M+1] 测试

  评估指标:
  - 参数稳定性: 每个因子在各窗口中被选中的频率
  - ICIR 稳定性: 各窗口样本外 ICIR 的均值和标准差
  - 最优参数一致性: 跨窗口出现频率最高的参数组合

核心结论 (Ch6):
  "参数稳定性比参数最优更重要"
  "好参数跨窗口稳定, 坏参数到处变"

来源: 《XQuant: 人人都是量化交易员》第6章 Step 2
创建: 2026-06-14
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time

# FA 内部引用
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.ic_analysis import compute_ic_icir
from factors.composite import combine_factors


def anchored_walkforward(
    factor_dfs: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    ic_summary: pd.DataFrame,
    n_windows: int = 8,
    min_train_periods: int = 52,
    top_n_factors: int = 10,
    n_select: int = 5,
) -> Dict:
    """
    锚定式 Walk-forward 分析

    训练窗口从起点开始不断扩展, 每步在最长的可用历史上优化,
    然后在下一段样本外验证。

    返回:
        dict: {
            'windows': [{train_start, train_end, test_start, test_end, ...}],
            'top_params': [(mom, vol, freq)] 每个窗口的最优参数,
            'param_frequency': {param: count} 各参数组合出现次数,
            'factor_frequency': {factor: count} 各因子出现次数,
            'oos_icir': [float] 每窗口样本外ICIR,
            'oos_icir_mean': float,
            'oos_icir_std': float,
            'stability_score': float,  # 0-100 综合稳定性评分
        }
    """
    print(f"\n{'='*60}")
    print(f"  Walk-forward 分析 — Anchored (锚定式)")
    print(f"  窗口数: {n_windows} | 最少训练期: {min_train_periods} 周")
    print(f"  候选因子: {top_n_factors} | 每窗口选: {n_select}")
    print(f"{'='*60}")

    dates = pd.DatetimeIndex(forward_returns.index)
    n_total = len(dates)

    if n_total < min_train_periods * 2:
        print(f"  [ERROR] 数据不足: {n_total} 周 < 最少需要 {min_train_periods*2}")
        return None

    # 划窗: 每个窗口的训练集不断扩展
    window_edges = np.linspace(min_train_periods, n_total - min(n_windows, 26), 
                               n_windows + 1, dtype=int)

    # 取 ICIR 最高的 top_n 因子作为候选池
    top_candidates = ic_summary.nlargest(top_n_factors, 'ICIR')
    candidate_names = [n for n in top_candidates.index if n in factor_dfs]
    print(f"  候选因子池: {len(candidate_names)} 个 (ICIR top {top_n_factors})")

    window_results = []
    all_params = []
    all_factors_selected = []
    oos_icirs = []

    for i in range(n_windows):
        train_end_idx = window_edges[i]
        test_start_idx = window_edges[i]
        test_end_idx = min(window_edges[i + 1], n_total)

        if test_end_idx - test_start_idx < 4:
            continue  # 跳过太短的测试窗口

        train_dates = dates[:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]

        train_fr = forward_returns.loc[train_dates]
        test_fr = forward_returns.loc[test_dates]

        # 贪心选因子 (基于训练集 ICIR)
        selected = []
        remaining = list(candidate_names)

        for _ in range(n_select):
            best_factor = None
            best_icir = -999

            for f in remaining:
                if f not in factor_dfs:
                    continue
                combo = selected + [f]
                w = {fi: 1.0/len(combo) for fi in combo}
                try:
                    sel_dfs = {n: factor_dfs[n] for n in combo if n in factor_dfs}
                    composite = combine_factors(sel_dfs, w)
                    icir_val = abs(compute_ic_icir(composite, train_fr)['icir'])
                    if not np.isnan(icir_val) and icir_val > best_icir:
                        best_icir = icir_val
                        best_factor = f
                except Exception:
                    continue

            if best_factor:
                selected.append(best_factor)
                remaining.remove(best_factor)

        # 样本外评估
        w_final = {f: 1.0/len(selected) for f in selected}
        sel_dfs = {n: factor_dfs[n] for n in selected if n in factor_dfs}
        composite = combine_factors(sel_dfs, w_final)

        icir_train = abs(compute_ic_icir(composite, train_fr)['icir'])
        icir_test = abs(compute_ic_icir(composite, test_fr)['icir'])

        window_results.append({
            'window': i,
            'train_start': str(train_dates[0].date()),
            'train_end': str(train_dates[-1].date()),
            'test_start': str(test_dates[0].date()),
            'test_end': str(test_dates[-1].date()),
            'train_weeks': len(train_dates),
            'test_weeks': len(test_dates),
            'factors': selected,
            'icir_train': icir_train,
            'icir_test': icir_test,
            'decay': icir_test - icir_train,
        })

        all_params.append(tuple(sorted(selected)))
        all_factors_selected.extend(selected)
        oos_icirs.append(icir_test)

        print(f"  Win {i:2d}: 训练 {train_dates[0].date()}~{train_dates[-1].date()} "
              f"({len(train_dates)}周) → 测试 {test_dates[0].date()}~{test_dates[-1].date()} "
              f"({len(test_dates)}周) ICIR={icir_test:.3f} "
              f"因子={selected}")

    # ---- 汇总分析 ----
    if not window_results:
        print("  [ERROR] 无有效窗口")
        return None

    # 参数频率
    from collections import Counter
    param_freq = Counter(all_params)
    factor_freq = Counter(all_factors_selected)

    oos_mean = np.mean(oos_icirs) if oos_icirs else np.nan
    oos_std = np.std(oos_icirs) if oos_icirs else np.nan

    # 稳定性评分 (0-100)
    # 1. 参数一致性 (40%): 最频繁参数的出现频率
    most_common_pct = param_freq.most_common(1)[0][1] / n_windows if param_freq else 0

    # 2. OOS 稳定性 (40%): 1 - CV (变异系数越小越稳定)
    cv = oos_std / max(abs(oos_mean), 0.001)
    oos_stability = max(0, 1 - cv) if not np.isnan(cv) else 0

    # 3. 衰减稳定性 (20%): 平均衰减接近0
    avg_decay = np.mean([r['decay'] for r in window_results if not np.isnan(r['decay'])])
    decay_score = max(0, 1 - abs(avg_decay)) if not np.isnan(avg_decay) else 0

    stability_score = (most_common_pct * 40 + oos_stability * 40 + decay_score * 20)
    stability_score = min(100, max(0, stability_score))

    # ---- 打印报告 ----
    print(f"\n  === WF-Anchored 汇总 ===")
    print(f"  OOS ICIR: {oos_mean:.3f} ± {oos_std:.3f} (CV={cv:.2f})")
    print(f"  衰减均值: {avg_decay:+.3f}")
    print(f"  稳定性评分: {stability_score:.0f}/100")

    # 因子频率
    print(f"\n  --- 因子跨窗口频率 ---")
    print(f"  {'因子':25s} {'出现次数':>8s} {'频率':>8s}")
    print(f"  {'-'*43}")
    for f, count in factor_freq.most_common(10):
        pct = count / n_windows
        marker = " ***" if pct >= 0.7 else ""
        print(f"  {f:25s} {count:8d} {pct:7.1%}{marker}")

    # 参数组合频率
    print(f"\n  --- 参数组合一致性 ---")
    print(f"  {'组合':40s} {'次数':>5s} {'频率':>7s}")
    print(f"  {'-'*54}")
    for combo, count in param_freq.most_common(5):
        pct = count / n_windows
        print(f"  {str(combo):40s} {count:5d} {pct:6.1%}")

    # 稳定性判断
    if stability_score >= 70:
        verdict = "PASS 参数稳定, 可推广"
    elif stability_score >= 50:
        verdict = "WARN 参数中等稳定, 注意监控"
    else:
        verdict = "FAIL 参数不稳定, 可能过拟合"
    print(f"\n  === 结论: {verdict} ===")

    return {
        'window_results': window_results,
        'factor_freq': factor_freq,
        'param_freq': param_freq,
        'oos_icir_mean': oos_mean,
        'oos_icir_std': oos_std,
        'avg_decay': avg_decay,
        'stability_score': stability_score,
        'verdict': verdict,
    }


def rolling_walkforward(
    factor_dfs: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    ic_summary: pd.DataFrame,
    window_size: int = 104,   # 2年
    step_size: int = 26,      # 半年
    test_size: int = 26,
    top_n_factors: int = 10,
    n_select: int = 5,
) -> Dict:
    """
    滚动式 Walk-forward 分析

    训练窗口固定长度向前滑动, 模拟 "每半年重新优化" 的真实操作场景。

    返回: 同 anchored_walkforward
    """
    print(f"\n{'='*60}")
    print(f"  Walk-forward 分析 — Rolling (滚动式)")
    print(f"  窗口: {window_size}周训练 | {step_size}周步长 | {test_size}周测试")
    print(f"{'='*60}")

    dates = pd.DatetimeIndex(forward_returns.index)
    n_total = len(dates)

    if n_total < window_size + test_size:
        print(f"  [ERROR] 数据不足: {n_total} < {window_size + test_size}")
        return None

    top_candidates = ic_summary.nlargest(top_n_factors, 'ICIR')
    candidate_names = [n for n in top_candidates.index if n in factor_dfs]

    window_results = []
    all_params = []
    all_factors_selected = []
    oos_icirs = []

    for train_start in range(0, n_total - window_size - test_size, step_size):
        train_end = train_start + window_size
        test_start = train_end
        test_end = min(test_start + test_size, n_total)

        if test_end - test_start < 4:
            continue

        train_dates = dates[train_start:train_end]
        test_dates = dates[test_start:test_end]
        train_fr = forward_returns.loc[train_dates]
        test_fr = forward_returns.loc[test_dates]

        # 贪心选因子
        selected = []
        remaining = list(candidate_names)
        for _ in range(n_select):
            best_factor = None
            best_icir = -999
            for f in remaining:
                if f not in factor_dfs:
                    continue
                combo = selected + [f]
                w = {fi: 1.0/len(combo) for fi in combo}
                try:
                    sel_dfs = {n: factor_dfs[n] for n in combo if n in factor_dfs}
                    composite = combine_factors(sel_dfs, w)
                    icir_val = abs(compute_ic_icir(composite, train_fr)['icir'])
                    if not np.isnan(icir_val) and icir_val > best_icir:
                        best_icir = icir_val
                        best_factor = f
                except Exception:
                    continue
            if best_factor:
                selected.append(best_factor)
                remaining.remove(best_factor)

        # 样本外
        w_final = {f: 1.0/len(selected) for f in selected}
        sel_dfs = {n: factor_dfs[n] for n in selected if n in factor_dfs}
        composite = combine_factors(sel_dfs, w_final)
        icir_test = abs(compute_ic_icir(composite, test_fr)['icir'])

        window_results.append({
            'window': len(window_results),
            'train_start': str(train_dates[0].date()),
            'train_end': str(train_dates[-1].date()),
            'test_start': str(test_dates[0].date()),
            'test_end': str(test_dates[-1].date()),
            'factors': selected,
            'icir_test': icir_test,
        })

        all_params.append(tuple(sorted(selected)))
        all_factors_selected.extend(selected)
        oos_icirs.append(icir_test)

        if len(window_results) <= 10:  # 只打印前10个窗口
            print(f"  Win {len(window_results)-1:2d}: "
                  f"训练 {train_dates[0].date()}~{train_dates[-1].date()} → "
                  f"测试 ICIR={icir_test:.3f} 因子={selected}")

    if not window_results:
        print("  [ERROR] 无有效窗口")
        return None

    from collections import Counter
    param_freq = Counter(all_params)
    factor_freq = Counter(all_factors_selected)
    oos_mean = np.mean(oos_icirs)
    oos_std = np.std(oos_icirs)

    n_w = len(window_results)
    most_common_pct = param_freq.most_common(1)[0][1] / n_w if param_freq else 0
    cv = oos_std / max(abs(oos_mean), 0.001)
    stability_score = min(100, max(0, most_common_pct * 40 + max(0, 1 - cv) * 40 + 20))

    print(f"\n  === WF-Rolling 汇总 ===")
    print(f"  总窗口: {n_w}")
    print(f"  OOS ICIR: {oos_mean:.3f} ± {oos_std:.3f}")
    print(f"  稳定性评分: {stability_score:.0f}/100")

    if stability_score >= 70:
        verdict = "PASS"
    elif stability_score >= 50:
        verdict = "WARN"
    else:
        verdict = "FAIL"
    print(f"  结论: {verdict}")

    return {
        'window_results': window_results,
        'factor_freq': factor_freq,
        'param_freq': param_freq,
        'oos_icir_mean': oos_mean,
        'oos_icir_std': oos_std,
        'stability_score': stability_score,
        'verdict': verdict,
    }


def run_full_walkforward_validation(
    factor_dfs: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    ic_summary: pd.DataFrame,
) -> Dict:
    """
    完整 Walk-forward 验证 (Anchored + Rolling)

    Ch6 核心: "两种独立的切法都把概率最高的位置指向同一组参数 →
              切法换了, 结论没变 → 可信度高"
    """
    results = {}

    # Anchored
    t0 = time.time()
    results['anchored'] = anchored_walkforward(
        factor_dfs, forward_returns, ic_summary,
        n_windows=8, min_train_periods=52,
    )
    print(f"  Anchored 耗时: {time.time()-t0:.0f}s")

    # Rolling
    t0 = time.time()
    results['rolling'] = rolling_walkforward(
        factor_dfs, forward_returns, ic_summary,
        window_size=104, step_size=26, test_size=26,
    )
    print(f"  Rolling 耗时: {time.time()-t0:.0f}s")

    # 交叉验证一致性
    if results['anchored'] and results['rolling']:
        a_factor_freq = results['anchored']['factor_freq']
        r_factor_freq = results['rolling']['factor_freq']

        # 两方法共同确认的因子
        common_factors = set(a_factor_freq.keys()) & set(r_factor_freq.keys())
        print(f"\n  === 跨方法一致性 ===")
        print(f"  Anchored 使用因子: {len(a_factor_freq)}")
        print(f"  Rolling  使用因子: {len(r_factor_freq)}")
        print(f"  共同因子: {len(common_factors)}")

        if common_factors:
            # 共同因子的平均排名
            a_ranked = sorted(a_factor_freq.keys(), key=lambda x: -a_factor_freq[x])
            r_ranked = sorted(r_factor_freq.keys(), key=lambda x: -r_factor_freq[x])
            common_ranked = [f for f in a_ranked if f in common_factors]
            print(f"  共同核心因子: {common_ranked[:5]}")

        # 综合结论
        a_ok = results['anchored']['stability_score'] >= 50
        r_ok = results['rolling']['stability_score'] >= 50
        if a_ok and r_ok:
            print(f"  === WF 综合: PASS (双方法通过) ===")
        elif a_ok or r_ok:
            print(f"  === WF 综合: PARTIAL ({'Anchored' if a_ok else 'Rolling'}通过) ===")
        else:
            print(f"  === WF 综合: FAIL (双方法未通过) === 参数不稳定, 需要简化策略 ===")

    return results


if __name__ == '__main__':
    print("Walk-forward 稳定性评估模块")
    print("用法: from evaluation.walkforward_stability import run_full_walkforward_validation")
    print("集成到 run_fa.py Phase 5.5 之后:")
    print("  from evaluation.walkforward_stability import run_full_walkforward_validation")
    print("  wf_results = run_full_walkforward_validation(factor_dfs, forward_returns, ic_summary)")
