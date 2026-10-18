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
并行化: 2026-07-14 (ProcessPoolExecutor, 15窗口并发, FA_WF_WORKERS 环境变量)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# FA 内部引用
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
# ★ 2026-07-15 修复: 原用慢版 compute_ic_icir(Python逐日期循环+每日期重建set()),
#   在 Phase7 被高频调用(8窗口×~42次/窗口≈336次×286日期)→ 集合操作达数十亿次,
#   表现为"卡死/挂起"(ProcessPool版6.7h卡死, ThreadPool版7min无输出)。
#   改用 numpy向量化版 compute_ic_icir_fast(同返回结构, 快5-10x, 无p-value开销)。
from evaluation.ic_analysis import compute_ic_icir_fast as compute_ic_icir
# ★ 2026-07-15 修复: 原 combine_factors 是逐日 Python 循环(286日期×因子数),
#   在 WF 中被调用 ~328次(8+7窗口×41次), 单窗口需1-2分钟 → 表现为"卡死/挂起"。
#   改用向量化 combine_factors_vectorized(预标准化后整DataFrame加权求和, 快50-200x)。
from factors.composite import standardize_factors, combine_factors_vectorized


def _combine(sel_dfs, w):
    """向量化合成(替代逐日循环的 combine_factors), 供 WF 高频调用"""
    if not sel_dfs:
        return pd.DataFrame()
    std = standardize_factors(sel_dfs)
    return combine_factors_vectorized(std, w)


# ============================================================
# ProcessPool  worker 支持 (v7.8+ 并行化)
# ============================================================
# 每个窗口的贪心选因子 + ICIR 评估相互独立, 可并行。
# 为避免把 91 个因子全量 pickle 给每个 worker, 父进程只把
# 候选因子子集(ic_summary top_n, 默认10个)通过 initializer 传入 worker。
_WF_FACTOR_DFS = None
_WF_FORWARD_RETURNS = None
_WF_IC_SUMMARY = None


def _init_wf_worker(factor_dfs, forward_returns, ic_summary):
    """在每个 worker 进程里恢复全局数据 (spawn 模式下父进程内存不共享)"""
    global _WF_FACTOR_DFS, _WF_FORWARD_RETURNS, _WF_IC_SUMMARY
    _WF_FACTOR_DFS = factor_dfs
    _WF_FORWARD_RETURNS = forward_returns
    _WF_IC_SUMMARY = ic_summary


def _select_factors(candidate_names, n_select, train_fr):
    """贪心选因子: 每步加入使训练集组合 |ICIR| 最大的因子 (worker 内调用)"""
    selected = []
    remaining = list(candidate_names)
    for _ in range(n_select):
        best_factor = None
        best_icir = -999
        for f in remaining:
            if f not in _WF_FACTOR_DFS:
                continue
            combo = selected + [f]
            w = {fi: 1.0 / len(combo) for fi in combo}
            try:
                sel_dfs = {n: _WF_FACTOR_DFS[n] for n in combo if n in _WF_FACTOR_DFS}
                composite = _combine(sel_dfs, w)
                icir_val = abs(compute_ic_icir(composite, train_fr)['icir'])
                if not np.isnan(icir_val) and icir_val > best_icir:
                    best_icir = icir_val
                    best_factor = f
            except Exception:
                continue
        if best_factor:
            selected.append(best_factor)
            remaining.remove(best_factor)
    return selected


def _anchored_window_task(spec):
    """单个 Anchored 窗口的评估 (在 worker 进程执行)"""
    i = spec['i']
    train_dates = spec['train_dates']
    test_dates = spec['test_dates']
    candidate_names = spec['candidate_names']
    n_select = spec['n_select']

    train_fr = _WF_FORWARD_RETURNS.loc[train_dates]
    test_fr = _WF_FORWARD_RETURNS.loc[test_dates]

    selected = _select_factors(candidate_names, n_select, train_fr)

    w_final = {f: 1.0 / len(selected) for f in selected}
    sel_dfs = {n: _WF_FACTOR_DFS[n] for n in selected if n in _WF_FACTOR_DFS}
    composite = _combine(sel_dfs, w_final)

    icir_train = abs(compute_ic_icir(composite, train_fr)['icir'])
    icir_test = abs(compute_ic_icir(composite, test_fr)['icir'])
    print(f"  [Anchored win {i}] done: icir_test={icir_test:.3f} factors={selected}")

    return {
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
    }


def _rolling_window_task(spec):
    """单个 Rolling 窗口的评估 (在 worker 进程执行)"""
    i = spec['i']
    train_dates = spec['train_dates']
    test_dates = spec['test_dates']
    candidate_names = spec['candidate_names']
    n_select = spec['n_select']

    train_fr = _WF_FORWARD_RETURNS.loc[train_dates]
    test_fr = _WF_FORWARD_RETURNS.loc[test_dates]

    selected = _select_factors(candidate_names, n_select, train_fr)

    w_final = {f: 1.0 / len(selected) for f in selected}
    sel_dfs = {n: _WF_FACTOR_DFS[n] for n in selected if n in _WF_FACTOR_DFS}
    composite = _combine(sel_dfs, w_final)
    icir_test = abs(compute_ic_icir(composite, test_fr)['icir'])
    print(f"  [Rolling win {i}] done: icir_test={icir_test:.3f} factors={selected}")

    return {
        'window': i,
        'train_start': str(train_dates[0].date()),
        'train_end': str(train_dates[-1].date()),
        'test_start': str(test_dates[0].date()),
        'test_end': str(test_dates[-1].date()),
        'factors': selected,
        'icir_test': icir_test,
    }


def _run_windows_parallel(specs, task_fn, n_workers):
    """统一入口: 并行或串行跑一批窗口, 返回有序结果列表

    ★ 2026-07-14 修复 (Windows spawn 全局变量/内存爆炸 bug):
        原代码 initargs 传了全量 _WF_FACTOR_DFS (91因子, ~1.2GB/worker),
        8 个 ProcessPool 子进程各自反序列化 1.2GB → 内存爆炸 + 序列化死锁,
        主进程永久 block (实测卡死 6.7h, 进程 CPU 零增长)。
        修复: 只从 specs 提取"实际用到的候选因子名并集" (~10-20个),
        构建子集传给 worker, 避免全量 pickle。子集 / worker ≈ 130-260MB, 安全。
    """
    global _WF_FACTOR_DFS, _WF_FORWARD_RETURNS, _WF_IC_SUMMARY
    # 只把 specs 中实际使用的候选因子子集传给 worker (而非全量 91 因子)
    cand_names = set()
    for s in specs:
        cand_names.update(s.get('candidate_names', []) or [])
    cand_subset = (
        {n: _WF_FACTOR_DFS[n] for n in cand_names if n in _WF_FACTOR_DFS}
        if cand_names else None
    )
    factor_dfs_arg = cand_subset if cand_subset else _WF_FACTOR_DFS
    if n_workers and n_workers > 1 and len(specs) > 1:
        try:
            # ★ 改用 ThreadPool: 避免 Windows spawn 子进程的内存峰值(OOM kill)。
            #   父进程已运行数小时, spawn 8 worker 各反序列化候选子集易触发 OOM,
            #   致父进程被系统直接终止(无 Python 机会打印, 表现为"静默死亡")。
            #   线程池共享父进程内存(全局 _WF_FACTOR_DFS 等), 无 spawn/pickle 开销;
            #   worker 内 numpy 计算已释放 GIL, 可真正并行。
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                return list(ex.map(task_fn, specs))
        except Exception as e:
            print(f"  [WARN] ThreadPool 并行失败({e}), 回退串行")
    # 串行回退: 全局变量已在父函数(anchored/rolling)中设置, 直接调用
    return [task_fn(s) for s in specs]


def anchored_walkforward(
    factor_dfs: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    ic_summary: pd.DataFrame,
    n_windows: int = 8,
    min_train_periods: int = 52,
    top_n_factors: int = 10,
    n_select: int = 5,
    n_workers: int = 1,
) -> Dict:
    """
    锚定式 Walk-forward 分析

    训练窗口从起点开始不断扩展, 每步在最长的可用历史上优化,
    然后在下一段样本外验证。
    """
    print(f"\n{'='*60}")
    print(f"  Walk-forward 分析 — Anchored (锚定式)")
    print(f"  窗口数: {n_windows} | 最少训练期: {min_train_periods} 周")
    print(f"  候选因子: {top_n_factors} | 每窗口选: {n_select} | workers={n_workers}")
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

    # ---- 构建窗口规格 (spec) ----
    specs = []
    for i in range(n_windows):
        train_end_idx = window_edges[i]
        test_start_idx = window_edges[i]
        test_end_idx = min(window_edges[i + 1], n_total)

        if test_end_idx - test_start_idx < 4:
            continue  # 跳过太短的测试窗口

        train_dates = dates[:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]
        specs.append({
            'i': i,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'candidate_names': candidate_names,
            'n_select': n_select,
        })

    if not specs:
        print("  [ERROR] 无有效窗口")
        return None

    # ---- 并行 / 串行执行 ----
    t0 = time.time()
    # 父进程侧设置全局(供串行回退 + 打印用)
    global _WF_FACTOR_DFS, _WF_FORWARD_RETURNS, _WF_IC_SUMMARY
    _WF_FACTOR_DFS = factor_dfs
    _WF_FORWARD_RETURNS = forward_returns
    _WF_IC_SUMMARY = ic_summary

    window_results = _run_windows_parallel(specs, _anchored_window_task, n_workers)
    print(f"  [Anchored] {len(window_results)} 窗口评估完成, 耗时 {time.time()-t0:.1f}s")

    # 打印每个窗口结果 (并行后统一打印)
    for r in window_results:
        print(f"  Win {r['window']:2d}: 训练 {r['train_start']}~{r['train_end']} "
              f"({r['train_weeks']}周) → 测试 {r['test_start']}~{r['test_end']} "
              f"({r['test_weeks']}周) ICIR={r['icir_test']:.3f} "
              f"因子={r['factors']}")

    # ---- 汇总分析 ----
    if not window_results:
        print("  [ERROR] 无有效窗口")
        return None

    from collections import Counter
    all_params = [tuple(sorted(r['factors'])) for r in window_results]
    all_factors_selected = [f for r in window_results for f in r['factors']]
    oos_icirs = [r['icir_test'] for r in window_results]

    param_freq = Counter(all_params)
    factor_freq = Counter(all_factors_selected)

    oos_mean = np.mean(oos_icirs) if oos_icirs else np.nan
    oos_std = np.std(oos_icirs) if oos_icirs else np.nan

    most_common_pct = param_freq.most_common(1)[0][1] / n_windows if param_freq else 0

    cv = oos_std / max(abs(oos_mean), 0.001)
    oos_stability = max(0, 1 - cv) if not np.isnan(cv) else 0

    avg_decay = np.mean([r['decay'] for r in window_results if not np.isnan(r['decay'])])
    decay_score = max(0, 1 - abs(avg_decay)) if not np.isnan(avg_decay) else 0

    stability_score = (most_common_pct * 40 + oos_stability * 40 + decay_score * 20)
    stability_score = min(100, max(0, stability_score))

    print(f"\n  === WF-Anchored 汇总 ===")
    print(f"  OOS ICIR: {oos_mean:.3f} ± {oos_std:.3f} (CV={cv:.2f})")
    print(f"  衰减均值: {avg_decay:+.3f}")
    print(f"  稳定性评分: {stability_score:.0f}/100")

    print(f"\n  --- 因子跨窗口频率 ---")
    print(f"  {'因子':25s} {'出现次数':>8s} {'频率':>8s}")
    print(f"  {'-'*43}")
    for f, count in factor_freq.most_common(10):
        pct = count / n_windows
        marker = " ***" if pct >= 0.7 else ""
        print(f"  {f:25s} {count:8d} {pct:7.1%}{marker}")

    print(f"\n  --- 参数组合一致性 ---")
    print(f"  {'组合':40s} {'次数':>5s} {'频率':>7s}")
    print(f"  {'-'*54}")
    for combo, count in param_freq.most_common(5):
        pct = count / n_windows
        print(f"  {str(combo):40s} {count:5d} {pct:6.1%}")

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
    window_size: int = 104,
    step_size: int = 26,
    test_size: int = 26,
    top_n_factors: int = 10,
    n_select: int = 5,
    n_workers: int = 1,
) -> Dict:
    """
    滚动式 Walk-forward 分析

    训练窗口固定长度向前滑动, 模拟 "每半年重新优化" 的真实操作场景。
    """
    print(f"\n{'='*60}")
    print(f"  Walk-forward 分析 — Rolling (滚动式)")
    print(f"  窗口: {window_size}周训练 | {step_size}周步长 | {test_size}周测试 | workers={n_workers}")
    print(f"{'='*60}")

    dates = pd.DatetimeIndex(forward_returns.index)
    n_total = len(dates)

    if n_total < window_size + test_size:
        print(f"  [ERROR] 数据不足: {n_total} < {window_size + test_size}")
        return None

    top_candidates = ic_summary.nlargest(top_n_factors, 'ICIR')
    candidate_names = [n for n in top_candidates.index if n in factor_dfs]

    # ---- 构建窗口规格 ----
    specs = []
    win_idx = 0
    for train_start in range(0, n_total - window_size - test_size, step_size):
        train_end = train_start + window_size
        test_start = train_end
        test_end = min(test_start + test_size, n_total)

        if test_end - test_start < 4:
            continue

        train_dates = dates[train_start:train_end]
        test_dates = dates[test_start:test_end]
        specs.append({
            'i': win_idx,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'candidate_names': candidate_names,
            'n_select': n_select,
        })
        win_idx += 1

    if not specs:
        print("  [ERROR] 无有效窗口")
        return None

    # ---- 并行 / 串行执行 ----
    t0 = time.time()
    global _WF_FACTOR_DFS, _WF_FORWARD_RETURNS, _WF_IC_SUMMARY
    _WF_FACTOR_DFS = factor_dfs
    _WF_FORWARD_RETURNS = forward_returns
    _WF_IC_SUMMARY = ic_summary

    window_results = _run_windows_parallel(specs, _rolling_window_task, n_workers)
    print(f"  [Rolling] {len(window_results)} 窗口评估完成, 耗时 {time.time()-t0:.1f}s")

    for r in window_results:
        print(f"  Win {r['window']:2d}: "
              f"训练 {r['train_start']}~{r['train_end']} → "
              f"测试 ICIR={r['icir_test']:.3f} 因子={r['factors']}")

    if not window_results:
        print("  [ERROR] 无有效窗口")
        return None

    from collections import Counter
    param_freq = Counter([tuple(sorted(r['factors'])) for r in window_results])
    factor_freq = Counter([f for r in window_results for f in r['factors']])
    oos_icirs = [r['icir_test'] for r in window_results]
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
    wf_workers: Optional[int] = None,
) -> Dict:
    """
    完整 Walk-forward 验证 (Anchored + Rolling)

    Ch6 核心: "两种独立的切法都把概率最高的位置指向同一组参数 →
              切法换了, 结论没变 → 可信度高"

    wf_workers: 并行 worker 数. 默认读 FA_WF_WORKERS 环境变量,
                否则取 min(8, cpu_count), 至少 2.
    """
    # 解析 worker 数
    if wf_workers is None:
        try:
            wf_workers = int(os.environ.get('FA_WF_WORKERS', '0')) or None
        except Exception:
            wf_workers = None
    if wf_workers is None:
        # ★ 07-30 修复: 默认串行。原因: ThreadPool(8 workers) 在 Anchored WF
        #    处理 10 候选因子 × 8 窗口时, 父进程内存已高位(OOM kill),
        #    OS 级 SIGKILL 无法被 run_fa.py 的 try/except 捕获。
        #    如需并行, 通过环境变量 FA_WF_WORKERS 显式指定。
        wf_workers = 1

    # 仅向 worker 传递候选因子子集 (ic_summary top_n, 默认10),
    # 避免把 91 个因子全量 pickle 给每个子进程, 显著降低启动开销.
    top_n = 10
    try:
        cand = [n for n in ic_summary.nlargest(top_n, 'ICIR').index if n in factor_dfs]
    except Exception:
        cand = list(factor_dfs.keys())[:top_n]
    factor_sub = {n: factor_dfs[n] for n in cand}
    if not factor_sub:
        factor_sub = factor_dfs

    results = {}

    # Anchored
    t0 = time.time()
    results['anchored'] = anchored_walkforward(
        factor_sub, forward_returns, ic_summary,
        n_windows=8, min_train_periods=52, n_workers=wf_workers,
    )
    print(f"  Anchored 总耗时: {time.time()-t0:.0f}s (workers={wf_workers})")

    # Rolling — 强制串行 (Windows ThreadPool 在 Anchored 线程池结束后复用失败: [Errno 22])
    rolling_wf_workers = 1
    t0 = time.time()
    results['rolling'] = rolling_walkforward(
        factor_sub, forward_returns, ic_summary,
        window_size=104, step_size=26, test_size=26, n_workers=rolling_wf_workers,
    )
    print(f"  Rolling 总耗时: {time.time()-t0:.0f}s (串行)")

    # 交叉验证一致性
    if results['anchored'] and results['rolling']:
        a_factor_freq = results['anchored']['factor_freq']
        r_factor_freq = results['rolling']['factor_freq']

        common_factors = set(a_factor_freq.keys()) & set(r_factor_freq.keys())
        print(f"\n  === 跨方法一致性 ===")
        print(f"  Anchored 使用因子: {len(a_factor_freq)}")
        print(f"  Rolling  使用因子: {len(r_factor_freq)}")
        print(f"  共同因子: {len(common_factors)}")

        if common_factors:
            a_ranked = sorted(a_factor_freq.keys(), key=lambda x: -a_factor_freq[x])
            common_ranked = [f for f in a_ranked if f in common_factors]
            print(f"  共同核心因子: {common_ranked[:5]}")

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
    print("Walk-forward 稳定性评估模块 (v7.8+ ProcessPool 并行)")
    print("用法: from evaluation.walkforward_stability import run_full_walkforward_validation")
    print("集成到 run_fa.py Phase 5.5 之后:")
    print("  wf_results = run_full_walkforward_validation(factor_dfs, forward_returns, ic_summary)")
    print("并行控制: 设置环境变量 FA_WF_WORKERS=N (N>1 启用 ProcessPool)")
