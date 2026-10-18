"""
NSGA-II 多目标遗传算法
=======================
在 ICIR 与 稳定性 之间寻找帕累托前沿

核心机制:
1. 非支配排序 (Non-Dominated Sort): 将种群按支配关系分层
2. 拥挤距离 (Crowding Distance): 保证前沿面的多样性
3. 二元锦标赛选择: 优先低级Front, 同级优先高拥挤距离

Reference: Deb et al. (2002) "A Fast and Elitist Multiobjective GA: NSGA-II"
"""

import numpy as np


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

    return next_pop, next_fronts, next_crowding


def evaluate_multi_objective(population, factor_names, factor_dict,
                              forward_returns, mcap_df, corr_matrix=None,
                              is_size_map=None, ic_summary=None,
                              factor_ic_series=None,
                              factor_dfs_std=None, debug_obj3=False):
    """
    评估种群多目标适应度 (v7.2 3D: 净质量/纯稳定/成本后夏普)
    返回 (N, N_OBJECTIVES) 矩阵
    
    v7.2: 支持 factor_dfs_std (预标准化因子) 和 debug_obj3 (诊断模式)
    """
    from config import N_OBJECTIVES
    from ga.fitness import compute_multi_objective

    n_pop = len(population)
    obj_matrix = np.zeros((n_pop, N_OBJECTIVES))

    for i, chromo in enumerate(population):
        result = compute_multi_objective(
            chromo, factor_names, factor_dict,
            forward_returns, mcap_df,
            corr_matrix=corr_matrix,
            is_size_map=is_size_map,
            ic_summary=ic_summary,
            factor_ic_series=factor_ic_series,
            factor_dfs_std=factor_dfs_std,
            debug_obj3=debug_obj3
        )
        for j in range(N_OBJECTIVES):
            obj_matrix[i, j] = result[j]

    return obj_matrix


def nsga2_run(population, factor_names, factor_dict, forward_returns,
              mcap_df, n_generations=10, pop_size=None,
              crossover_prob=0.7, mutation_prob=0.3, mutation_sigma=0.3,
              corr_matrix=None, is_size_map=None, ic_summary=None,
              factor_ic_series=None, max_factors=None, verbose=True,
              factor_dfs_std=None, debug_obj3=False):
    """
    运行 NSGA-II 多目标优化 (v7.2: 三目标 + 方向感知Obj3)

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
    """
    from ga.chromosome import random_chromosome, Chromosome
    from ga.operators import uniform_crossover, gaussian_mutation

    if pop_size is None:
        pop_size = len(population)

    n_factors = len(factor_names)

    for gen in range(n_generations):
        # Evaluate
        objectives = evaluate_multi_objective(
            population, factor_names, factor_dict,
            forward_returns, mcap_df,
            corr_matrix=corr_matrix,
            is_size_map=is_size_map,
            ic_summary=ic_summary,
            factor_ic_series=factor_ic_series,
            factor_dfs_std=factor_dfs_std,
            debug_obj3=(debug_obj3 and gen == 0)  # 仅第1代输出诊断
        )

        # Non-dominated sort
        fronts = fast_non_dominated_sort(objectives)
        crowding = crowding_distance(objectives, list(range(len(population))))

        # Report progress
        if verbose:
            front1 = fronts[0] if len(fronts) > 0 else []
            front1_obj = objectives[front1] if len(front1) > 0 else np.zeros((0, objectives.shape[1]))
            n_pareto = len(front1)

            if n_pareto > 0:
                # v7: 三目标 [(净质量Q), (纯稳定S), (成本后Sharpe)]
                q_range = (front1_obj[:, 0].min(), front1_obj[:, 0].max())
                s_range = (front1_obj[:, 1].min(), front1_obj[:, 1].max())
                c_range = (front1_obj[:, 2].min(), front1_obj[:, 2].max())
                print(f"  Gen {gen+1}/{n_generations} | Pareto={n_pareto} "
                      f"Q[{q_range[0]:.3f},{q_range[1]:.3f}] "
                      f"S[{s_range[0]:.3f},{s_range[1]:.3f}] "
                      f"C[{c_range[0]:.3f},{c_range[1]:.3f}]")
            else:
                print(f"  Gen {gen+1}/{n_generations} | Pareto={n_pareto}")

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
        combined_obj = np.vstack([
            objectives,
            evaluate_multi_objective(
                offspring[:pop_size], factor_names, factor_dict,
                forward_returns, mcap_df,
                corr_matrix=corr_matrix,
                is_size_map=is_size_map,
                ic_summary=ic_summary,
                factor_ic_series=factor_ic_series,
                factor_dfs_std=factor_dfs_std,
                debug_obj3=False  # 子代不输出诊断
            )
        ])

        population, _, _ = nsga2_select_next_generation(
            combined_pop, combined_obj, pop_size
        )

    # Final evaluation
    final_objectives = evaluate_multi_objective(
        population, factor_names, factor_dict,
        forward_returns, mcap_df,
        corr_matrix=corr_matrix,
        is_size_map=is_size_map,
        ic_summary=ic_summary,
        factor_ic_series=factor_ic_series,
        factor_dfs_std=factor_dfs_std,
        debug_obj3=False
    )

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
