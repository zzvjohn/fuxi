"""
遗传算法引擎
=============
染色体: 因子权重向量 + 入选阈值
适应度: 四维度复合得分 (ICIR/单调性/低相关性/双重排序)
"""
import numpy as np
from .chromosome import Chromosome, random_chromosome
from .operators import tournament_select, uniform_crossover, gaussian_mutation
from .fitness import compute_fitness, evaluate_population


class FactorGA:
    """因子组合遗传算法"""
    
    def __init__(self, factor_names, factor_dict, forward_returns, mcap_df, is_size_map,
                 population_size=200, generations=100,
                 crossover_prob=0.7, mutation_prob=0.2, mutation_sigma=0.1,
                 elite_count=5, tournament_size=5,
                 random_seed=42):
        """
        Parameters
        ----------
        factor_names : list
            候选因子名列表
        factor_dict : dict
            {name: pd.DataFrame(index=date, columns=stocks)}
        forward_returns : pd.DataFrame
            前向收益
        mcap_df : pd.DataFrame
            市值
        is_size_map : dict
            {name: True/False} 标记规模因子
        """
        self.factor_names = factor_names
        self.factor_dict = factor_dict
        self.forward_returns = forward_returns
        self.mcap_df = mcap_df
        self.is_size_map = is_size_map
        
        self.pop_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_sigma
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        
        np.random.seed(random_seed)
        
        self.n_factors = len(factor_names)
        self.population = None
        self.best_chromosome = None
        self.best_fitness = -np.inf
        self.history = []  # 每代最佳适应度
    
    def initialize(self):
        """初始化种群"""
        self.population = [
            random_chromosome(self.n_factors) for _ in range(self.pop_size)
        ]
    
    def evolve(self, verbose=True):
        """进化主循环"""
        if self.population is None:
            self.initialize()
        
        for gen in range(self.generations):
            # 评估
            fitness_scores = evaluate_population(
                self.population, self.factor_names,
                self.factor_dict, self.forward_returns, self.mcap_df,
                self.is_size_map
            )
            
            # 记录最佳
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_chromosome = Chromosome(self.population[best_idx].copy())
            
            gen_best = np.max(fitness_scores)
            gen_mean = np.mean(fitness_scores)
            gen_std = np.std(fitness_scores)
            self.history.append((gen_best, gen_mean, gen_std))
            
            if verbose and gen % 10 == 0:
                print(f"  Gen {gen:3d}/{self.generations} | Best={gen_best:.4f} "
                      f"Mean={gen_mean:.4f} Std={gen_std:.4f}")
            
            # 选择、交叉、变异 → 新一代
            new_pop = []
            
            # 精英保留
            elite_indices = np.argsort(fitness_scores)[-self.elite_count:]
            for idx in elite_indices:
                new_pop.append(self.population[idx].copy())
            
            # 填充剩余
            while len(new_pop) < self.pop_size:
                # 锦标赛选择
                p1 = tournament_select(self.population, fitness_scores, self.tournament_size)
                p2 = tournament_select(self.population, fitness_scores, self.tournament_size)
                
                # 交叉
                if np.random.random() < self.crossover_prob:
                    c1, c2 = uniform_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # 变异
                if np.random.random() < self.mutation_prob:
                    c1 = gaussian_mutation(c1, self.mutation_sigma)
                if np.random.random() < self.mutation_prob:
                    c2 = gaussian_mutation(c2, self.mutation_sigma)
                
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)
            
            self.population = new_pop[:self.pop_size]
        
        if verbose:
            print(f"\n  === GA进化完成 ===")
            print(f"  最佳适应度: {self.best_fitness:.4f}")
            best_weights = self.get_best_weights()
            print(f"  最佳权重: {best_weights}")
        
        return self.best_chromosome, self.best_fitness
    
    def get_best_weights(self):
        """获取最佳染色体对应的归一化权重"""
        from factors.composite import weights_from_chromosome
        if self.best_chromosome is None:
            return {}
        return weights_from_chromosome(self.best_chromosome, self.factor_names)
    
    def to_dataframe(self):
        """进化历史 → DataFrame"""
        import pandas as pd
        return pd.DataFrame(self.history, columns=['best', 'mean', 'std'])
