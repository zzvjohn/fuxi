"""
NSGA-II 多目标遗传算法 (v7.8)
=======================
v7.8 (2026-07-13): 多线程并行评估 (ThreadPoolExecutor, FA_GA_WORKERS=4), 
    numpy rankdata 替代 scipy.spearmanr (释GIL), 每代耗时打印
v7.2-v7.3: 三目标(净质量/纯稳定/成本后Sharpe), 真加法链

核心机制:
1. 非支配排序 (Non-Dominated Sort): 将种群按支配关系分层
2. 拥挤距离 (Crowding Distance): 保证前沿面的多样性
3. 二元锦标赛选择: 优先低级Front, 同级优先高拥挤距离

Reference: Deb et al. (2002) "A Fast and Elitist Multiobjective GA: NSGA-II"
"""

import numpy as np


# ============================================================
# v7.8: 并行评估辅助函数 (模块级, 供 ThreadPoolExecutor 调用)
# ============================================================

def _eval_one_safe(args):
    """
    单个染色体安全评估 — 供线程池调用 (v7.8)

    参数打包成 args tuple 以便线程传递。
    numpy C 扩展在计算时释放 GIL, 线程池可有效利用多核。
    """
    (i, chromo, factor_names, factor_dict,
     forward_returns, mcap_df, corr_matrix,
     is_size_map, ic_summary, factor_ic_series,
     factor_dfs_std, factor_dfs_rp, debug_obj3) = args

    from ga.fitness import compute_multi_objective

    result = compute_multi_objective(
        chromo, factor_names, factor_dict,
        forward_returns, mcap_df,
        corr_matrix=corr_matrix,
        is_size_map=is_size_map,
        ic_summary=ic_summary,
        factor_ic_series=factor_ic_series,
        factor_dfs_std=factor_dfs_std,
        factor_dfs_rp=factor_dfs_rp,
        debug_obj3=debug_obj3
    )
    return i, result, None


def _evaluate_parallel(population, factor_names, factor_dict,
                       forward_returns, mcap_df, corr_matrix,
                       is_size_map, ic_summary, factor_ic_series,
                       factor_dfs_std, factor_dfs_rp, debug_obj3, gen, n_workers):
    """
    v7.8: 多线程并行评估种群

    使用 ThreadPoolExecutor 并行评估种群中的所有个体。
    限制 numpy BLAS 为单线程, 避免 threads × numpy 多线程 = 过度订阅。

    Returns
    -------
    obj_matrix : np.ndarray (n_pop, N_OBJECTIVES)
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
    from config import N_OBJECTIVES

    n_pop = len(population)
    obj_matrix = np.zeros((n_pop, N_OBJECTIVES))

    # 限制 numpy BLAS 线程数
    _old_omp = os.environ.get('OMP_NUM_THREADS', None)
    _old_mkl = os.environ.get('MKL_NUM_THREADS', None)
    _old_openblas = os.environ.get('OPENBLAS_NUM_THREADS', None)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    try:
        n_failed = 0
        effective_workers = min(n_workers, n_pop)

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {}
            for i, chromo in enumerate(population):
                packed = (i, chromo, factor_names, factor_dict,
                          forward_returns, mcap_df, corr_matrix,
                          is_size_map, ic_summary, factor_ic_series,
                          factor_dfs_std, factor_dfs_rp, debug_obj3)
                future = executor.submit(_eval_one_safe, packed)
                futures[future] = i

            # ★ 防死循环: 单个体评估超时即标记失败, 不让整代卡死 (曾卡在 Gen2→3)
            _EVAL_TIMEOUT = 180
            _pending = set(futures.keys())
            while _pending:
                _done, _pending = wait(_pending, timeout=_EVAL_TIMEOUT,
                                       return_when=FIRST_COMPLETED)
                for future in _done:
                    i = futures[future]
                    try:
                        i_result, result, _error = future.result()
                        for j in range(N_OBJECTIVES):
                            obj_matrix[i, j] = result[j]
                    except Exception as e:
                        n_failed += 1
                        import traceback
                        chromo_preview = (population[i][:16].tolist()
                                        if hasattr(population[i], 'tolist')
                                        else str(population[i])[:80])
                        print(f"  [GA-CRASH] gen={gen} idx={i} chromo[:16]={chromo_preview} | "
                              f"{type(e).__name__}: {e}")
                        traceback.print_exc()
                        obj_matrix[i, :] = 0.0
                if _pending:
                    # 超时未完成 -> 标记失败, 放弃等待 (疑似线程内死循环/死锁)
                    for future in _pending:
                        i = futures[future]
                        chromo_preview = (population[i][:16].tolist()
                                        if hasattr(population[i], 'tolist')
                                        else str(population[i])[:80])
                        print(f"  [GA-TIMEOUT] gen={gen} idx={i} 评估>{_EVAL_TIMEOUT}s "
                              f"疑似死循环, 置零 | chromo[:16]={chromo_preview}")
                        obj_matrix[i, :] = 0.0
                        n_failed += 1
                        future.cancel()
                    _pending = set()

        if n_failed > 0:
            print(f"  [WARN] evaluate_parallel: {n_failed}/{n_pop} 个体评估失败(gen={gen}), 已置零")

        return obj_matrix

    finally:
        # 恢复 BLAS 线程设置
        for key, old_val in [('OMP_NUM_THREADS', _old_omp),
                              ('MKL_NUM_THREADS', _old_mkl),
                              ('OPENBLAS_NUM_THREADS', _old_openblas)]:
            if old_val is not None:
                os.environ[key] = old_val
            elif key in os.environ:
                del os.environ[key]


def fast_non_dominated_sort(objectives):
    """
    快速非支配排序 (Deb's O(MN^2) algorithm)

    Parameters
    ----------
    objectives : np.ndarray
        (N, M) 目标矩阵, N=种群大小, M=目标数
        所有目标均为最大化方向

    Returns
    -------
    fronts : list of list of int
        fronts[i] = 第i层前沿中个体的索引列表
    """
    n_pop = len(objectives)
    if n_pop == 0:
        return []

    # domination_count[i] = 支配个体i的个体数
    domination_count = np.zeros(n_pop, dtype=int)
    # dominated_set[i] = 被个体i支配的个体索引列表
    dominated_set = [[] for _ in range(n_pop)]

    # O(M * N^2) pairwise comparison
    for i in range(n_pop):
        for j in range(i + 1, n_pop):
            obj_i = objectives[i]
            obj_j = objectives[j]

            # Check if i dominates j (all objectives >= and at least one >)
            i_dominates_j = np.all(obj_i >= obj_j) and np.any(obj_i > obj_j)
            j_dominates_i = np.all(obj_j >= obj_i) and np.any(obj_j > obj_i)

            if i_dominates_j:
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif j_dominates_i:
                dominated_set[j].append(i)
                domination_count[i] += 1

    # Front 1: all individuals with domination_count == 0
    fronts = []
    current_front = [i for i in range(n_pop) if domination_count[i] == 0]
    fronts.append(current_front)

    # Iteratively peel off fronts
    front_idx = 0
    while len(fronts[front_idx]) > 0:
        next_front = []
        for i in fronts[front_idx]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        front_idx += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break

    return fronts


def crowding_distance(objectives, front_indices):
    """
    计算一个Front内所有个体的拥挤距离

    拥挤距离 = 在各目标维度上, 相邻两点归一化间距之和
    距离越大 → 个体越"孤立" → 越值得保留 (维持多样性)

    Parameters
    ----------
    objectives : np.ndarray
        (N, M) 目标矩阵
    front_indices : list of int
        同一Front内个体的索引

    Returns
    -------
    distances : np.ndarray
        长度为N, 不在该Front内的个体距离为0
    """
    n_pop = len(objectives)
    distances = np.zeros(n_pop)

    if len(front_indices) <= 2:
        # 边界个体赋予无限距离 (优先保留)
        for idx in front_indices:
            distances[idx] = np.inf
        return distances

    n_obj = objectives.shape[1]
    front_obj = objectives[front_indices]

    for m in range(n_obj):
        # 按第m个目标排序
        sorted_idx = np.argsort(front_obj[:, m])
        obj_range = front_obj[sorted_idx[-1], m] - front_obj[sorted_idx[0], m]

        if obj_range == 0:
            continue

        # 边界=无穷
        global_idx_first = front_indices[sorted_idx[0]]
        global_idx_last = front_indices[sorted_idx[-1]]
        distances[global_idx_first] = np.inf
        distances[global_idx_last] = np.inf

        # 内部点: 相邻间距 / 范围
        for k in range(1, len(front_indices) - 1):
            global_idx = front_indices[sorted_idx[k]]
            prev_obj = front_obj[sorted_idx[k - 1], m]
            next_obj = front_obj[sorted_idx[k + 1], m]
            distances[global_idx] += (next_obj - prev_obj) / obj_range

    return distances


def nsga2_tournament_select(population, objectives, fronts, crowding, k=3):
    """
    NSGA-II 锦标赛选择

    规则:
    1. 优先选 Front 编号更低的个体
    2. 同一 Front 内选拥挤距离更大的个体

    Parameters
    ----------
    population : list
        种群列表
    objectives : np.ndarray
        目标矩阵
    fronts : list of list of int
    crowding : np.ndarray
    k : int
        锦标赛规模

    Returns
    -------
    selected : selected individual (Chromosome object)
    """
    n_pop = len(population)

    # Build front rank map: individual_idx → front_number
    front_rank = np.full(n_pop, np.inf)
    for rank, front_indices in enumerate(fronts):
        for idx in front_indices:
            front_rank[idx] = rank

    # Tournament
    candidates = np.random.choice(n_pop, k, replace=False)

    best = candidates[0]
    best_rank = front_rank[best]
    best_crowd = crowding[best]

    for c in candidates[1:]:
        c_rank = front_rank[c]
        c_crowd = crowding[c]

        if c_rank < best_rank:
            best, best_rank, best_crowd = c, c_rank, c_crowd
        elif c_rank == best_rank and c_crowd > best_crowd:
            best, best_crowd = c, c_crowd

    return population[best]


def nsga2_select_next_generation(population, objectives, pop_size):
    """
    从当前种群 (parents + offsprings = 2*pop_size) 选出下一代

    按 Front 逐层填充, 最后一层按拥挤距离截断

    Parameters
    ----------
    population : list
        2*pop_size 个体的联合种群
    objectives : np.ndarray
        联合种群的目标矩阵
    pop_size : int
        目标种群大小

    Returns
    -------
    next_pop : list
    next_fronts : list of list of int
    next_crowding : np.ndarray
    """
    fronts = fast_non_dominated_sort(objectives)

    selected_indices = []
    for front_indices in fronts:
        if len(selected_indices) + len(front_indices) <= pop_size:
            selected_indices.extend(front_indices)
        else:
            # 最后一层: 按拥挤距离截断
            remaining = pop_size - len(selected_indices)
            crowding = crowding_distance(objectives, front_indices)
            # 选拥挤距离最大的remaining个
            front_crowding = [(idx, crowding[idx]) for idx in front_indices]
            front_crowding.sort(key=lambda x: -x[1])
            selected_indices.extend([idx for idx, _ in front_crowding[:remaining]])
            break

    next_pop = [population[i] for i in selected_indices]

    # Recompute fronts and crowding for returned population
    next_obj = objectives[selected_indices]
    next_fronts = fast_non_dominated_sort(next_obj)
    next_crowding = crowding_distance(next_obj, list(range(len(next_pop))))

    return next_pop, next_fronts, next_crowding, next_obj


def evaluate_multi_objective(population, factor_names, factor_dict,
                              forward_returns, mcap_df, corr_matrix=None,
                              is_size_map=None, ic_summary=None,
                              factor_ic_series=None,
                              factor_dfs_std=None, factor_dfs_rp=None,
                              debug_obj3=False, gen=None,
                              n_workers=1):
    """
    评估种群多目标适应度 (v7.2 3D: 净质量/纯稳定/成本后夏普)
    返回 (N, N_OBJECTIVES) 矩阵

    v7.3: 每个 chromosome 独立 try/except，失败个体设 [0,0,0] 不阻断全代
    v7.3+: 崩溃个体打印 gen/idx/chromo/异常类型+traceback, 方便后续修复
           (compute_multi_objective 内部已将未捕获异常 re-raise 到此层)
    v7.8: n_workers > 1 时自动切换到多线程并行评估 (ThreadPoolExecutor)
    """
    from config import N_OBJECTIVES

    # v7.8: 并行路径
    if n_workers > 1 and len(population) > 1:
        return _evaluate_parallel(
            population, factor_names, factor_dict,
            forward_returns, mcap_df, corr_matrix,
            is_size_map, ic_summary, factor_ic_series,
            factor_dfs_std, factor_dfs_rp, debug_obj3, gen, n_workers
        )

    # --- 串行路径 (原有代码, n_workers=1 或单个体) ---
    from ga.fitness import compute_multi_objective

    n_pop = len(population)
    obj_matrix = np.zeros((n_pop, N_OBJECTIVES))
    n_failed = 0

    for i, chromo in enumerate(population):
        try:
            result = compute_multi_objective(
                chromo, factor_names, factor_dict,
                forward_returns, mcap_df,
                corr_matrix=corr_matrix,
                is_size_map=is_size_map,
                ic_summary=ic_summary,
                factor_ic_series=factor_ic_series,
                factor_dfs_std=factor_dfs_std,
                factor_dfs_rp=factor_dfs_rp,
                debug_obj3=debug_obj3
            )
            for j in range(N_OBJECTIVES):
                obj_matrix[i, j] = result[j]
        except Exception as e:
            n_failed += 1
            # 失败染色体给零适应度，不阻断全代；记录识别信息便于修复
            chromo_preview = chromo[:16].tolist() if hasattr(chromo, 'tolist') else str(chromo)[:80]
            import traceback
            print(f"  [GA-CRASH] gen={gen} idx={i} chromo[:16]={chromo_preview} | "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()
            obj_matrix[i, :] = 0.0

    if n_failed > 0:
        print(f"  [WARN] evaluate_multi_objective: {n_failed}/{n_pop} 个体评估失败(gen={gen}), 已置零并记录")

    return obj_matrix


def nsga2_run(population, factor_names, factor_dict, forward_returns,
              mcap_df, n_generations=10, pop_size=None,
              crossover_prob=0.7, mutation_prob=0.3, mutation_sigma=0.3,
              corr_matrix=None, is_size_map=None, ic_summary=None,
              factor_ic_series=None, max_factors=None, verbose=True,
              factor_dfs_std=None, factor_dfs_rp=None, debug_obj3=False,
              n_workers=1):
    """
    运行 NSGA-II 多目标优化 (v8.0: rank-product + OOS + regime鲁棒)

    返回: (final_population, final_objectives, pareto_front)
       pareto_front = {
           'objectives': (k, 3) 帕累托前沿目标值,
           'chromosomes': 对应的染色体,
           'weights': 解析后的因子权重,
           'details': 详细评估信息,
       }
    
    v7.2新增:
      - factor_dfs_std: 预标准化因子 (加速组合)
      - debug_obj3: 强制Obj3诊断输出
    v7.8新增:
      - n_workers: 并行评估线程数 (ThreadPoolExecutor, 设 FA_GA_WORKERS 环境变量)
    """
    from ga.chromosome import random_chromosome, Chromosome
    from ga.operators import uniform_crossover, gaussian_mutation
    import time

    if pop_size is None:
        pop_size = len(population)

    n_factors = len(factor_names)

    for gen in range(n_generations):
        t_gen_start = time.time() if (verbose and n_workers > 1) else 0
        # Evaluate
        objectives = evaluate_multi_objective(
            population, factor_names, factor_dict,
            forward_returns, mcap_df,
            corr_matrix=corr_matrix,
            is_size_map=is_size_map,
            ic_summary=ic_summary,
            factor_ic_series=factor_ic_series,
            factor_dfs_std=factor_dfs_std,
            factor_dfs_rp=factor_dfs_rp,
            debug_obj3=(debug_obj3 and gen == 0),  # 仅第1代输出诊断
            gen=gen,
            n_workers=n_workers,  # v7.8
        )

        # Clean NaN/Inf before sort (prevent silent hang)
        objectives = np.nan_to_num(objectives, nan=0.0, posinf=0.0, neginf=0.0)

        # Non-dominated sort
        fronts = fast_non_dominated_sort(objectives)
        crowding = crowding_distance(objectives, list(range(len(population))))

        # Report progress
        if verbose:
            import sys
            front1 = fronts[0] if len(fronts) > 0 else []
            front1_obj = objectives[front1] if len(front1) > 0 else np.zeros((0, objectives.shape[1]))
            n_pareto = len(front1)

            if n_pareto > 0:
                # v7: 三目标 [(净质量Q), (纯稳定S), (成本后Sharpe)]
                q_range = (front1_obj[:, 0].min(), front1_obj[:, 0].max())
                s_range = (front1_obj[:, 1].min(), front1_obj[:, 1].max())
                c_range = (front1_obj[:, 2].min(), front1_obj[:, 2].max())
                elapsed = f" {time.time()-t_gen_start:.1f}s" if t_gen_start else ""
                print(f"  Gen {gen+1}/{n_generations} | Pareto={n_pareto} "
                      f"Q[{q_range[0]:.3f},{q_range[1]:.3f}] "
                      f"S[{s_range[0]:.3f},{s_range[1]:.3f}] "
                      f"C[{c_range[0]:.3f},{c_range[1]:.3f}]{elapsed}")
            else:
                print(f"  Gen {gen+1}/{n_generations} | Pareto={n_pareto}")

            # ★ P2.1: 种群多样性诊断
            if gen % 5 == 0 or gen == n_generations - 1:
                unique_chromos = len(set(tuple(np.round(c, 3)) for c in population))
                diver_ratio = unique_chromos / pop_size
                diver_flag = ""
                if diver_ratio < 0.3:
                    diver_flag = " [WARN: 种群坍缩! unique<30%]"
                elif diver_ratio < 0.5:
                    diver_flag = " [低多样性]"
                # 目标空间分散度
                obj_spread = np.std(objectives, axis=0)
                spread_norm = np.mean(obj_spread) if np.mean(np.abs(objectives)) > 0 else 0
                print(f"       Diversity: unique={unique_chromos}/{pop_size}({diver_ratio:.0%})"
                      f" spread={spread_norm:.3f}{diver_flag}")
            sys.stdout.flush()

        # Generate offspring
        offspring = []
        while len(offspring) < pop_size:
            p1 = nsga2_tournament_select(population, objectives, fronts, crowding, k=3)
            p2 = nsga2_tournament_select(population, objectives, fronts, crowding, k=3)

            if np.random.random() < crossover_prob:
                c1, c2 = uniform_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if np.random.random() < mutation_prob:
                c1 = gaussian_mutation(c1, mutation_sigma)
            if np.random.random() < mutation_prob:
                c2 = gaussian_mutation(c2, mutation_sigma)

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        # Merge parents + offspring, select next generation
        combined_pop = population + offspring[:pop_size]
        try:
            offspring_obj = evaluate_multi_objective(
                offspring[:pop_size], factor_names, factor_dict,
                forward_returns, mcap_df,
                corr_matrix=corr_matrix,
                is_size_map=is_size_map,
                ic_summary=ic_summary,
                factor_ic_series=factor_ic_series,
                factor_dfs_std=factor_dfs_std,
                debug_obj3=False,  # 子代不输出诊断
                gen=f"{gen}(offspring)",
                n_workers=n_workers,  # v7.8
            )
        except Exception as e:
            import traceback
            print(f"  [ERROR] 子代评估崩溃: {e}")
            traceback.print_exc()
            # Fallback: 保留父代
            offspring_obj = np.zeros((len(offspring[:pop_size]), 3))
        offspring_obj = np.nan_to_num(offspring_obj, nan=0.0, posinf=0.0, neginf=0.0)
        combined_obj = np.vstack([objectives, offspring_obj])

        population, _, _, final_objectives = nsga2_select_next_generation(
            combined_pop, combined_obj, pop_size
        )

    # v7.3: Skip redundant final evaluation — reuse objectives from last gen
    print(f"  [v7.3] 跳过重复最终评估，复用末代目标值")

    final_fronts = fast_non_dominated_sort(final_objectives)

    # Build Pareto front
    from factors.composite import weights_from_chromosome, combine_factors
    from ga.fitness import compute_multi_objective

    pareto_front = {
        'objectives': [],
        'chromosomes': [],
        'weights': [],
        'details': [],
    }

    if len(final_fronts) > 0 and len(final_fronts[0]) > 0:
        front1_idx = final_fronts[0]
        for idx in front1_idx:
            chromo = population[idx]
            obj = final_objectives[idx].tolist()
            w = weights_from_chromosome(chromo, factor_names, max_factors=max_factors)

            pareto_front['objectives'].append(obj)
            pareto_front['chromosomes'].append(chromo)
            pareto_front['weights'].append({k: float(v) for k, v in w.items()})

    pareto_front['objectives'] = np.array(pareto_front['objectives'])

    if verbose:
        print(f"\n  === NSGA-II (v7 三目标) 完成 ===")
        print(f"  帕累托前沿: {len(pareto_front['weights'])} 个解")
        if len(pareto_front['objectives']) > 0:
            print(f"  净质量(Q)范围:   [{pareto_front['objectives'][:, 0].min():.3f}, "
                  f"{pareto_front['objectives'][:, 0].max():.3f}]")
            print(f"  纯稳定(S)范围:   [{pareto_front['objectives'][:, 1].min():.3f}, "
                  f"{pareto_front['objectives'][:, 1].max():.3f}]")
            print(f"  成本后Sharpe(C): [{pareto_front['objectives'][:, 2].min():.3f}, "
                  f"{pareto_front['objectives'][:, 2].max():.3f}]")

    return population, final_objectives, pareto_front
