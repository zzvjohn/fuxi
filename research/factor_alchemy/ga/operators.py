"""
GA 算子: 选择、交叉、变异
"""
import numpy as np


def tournament_select(population, fitness_scores, tournament_size=5):
    """
    锦标赛选择
    
    Returns
    -------
    np.ndarray
        选中的染色体
    """
    n = len(population)
    candidates = np.random.choice(n, size=tournament_size, replace=False)
    best = candidates[np.argmax([fitness_scores[i] for i in candidates])]
    return population[best].copy()


def uniform_crossover(parent1, parent2):
    """
    均匀交叉: 每个基因位独立从父1或父2随机选择
    """
    mask = np.random.random(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2


def blend_crossover(parent1, parent2, alpha=0.5):
    """
    混合交叉 (BLX-alpha): 子代 = 父1 + alpha*(父2-父1)
    """
    child1 = parent1 + alpha * (parent2 - parent1)
    child2 = parent2 + alpha * (parent1 - parent2)
    return child1, child2


def gaussian_mutation(chromosome, sigma=0.1, mutation_rate=0.3):
    """
    高斯变异: 以 mutation_rate 概率对每个基因加 N(0, sigma^2) 噪声
    
    Parameters
    ----------
    chromosome : np.ndarray
    sigma : float
        变异幅度
    mutation_rate : float
        每个基因的变异概率
    
    Returns
    -------
    np.ndarray
    """
    mutant = chromosome.copy()
    mask = np.random.random(len(chromosome)) < mutation_rate
    noise = np.random.normal(0, sigma, len(chromosome))
    mutant[mask] += noise[mask]
    return mutant
