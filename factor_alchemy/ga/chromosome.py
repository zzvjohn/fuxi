"""
GA 染色体: 因子权重向量 [w1, w2, ..., wn]
"""
import numpy as np


class Chromosome(np.ndarray):
    """因子权重向量 (继承 ndarray)"""
    
    def __new__(cls, data):
        obj = np.asarray(data, dtype=float).view(cls)
        return obj
    
    @property
    def weights(self):
        """Softmax 归一化权重"""
        w = np.exp(np.clip(self, -10, 10))
        return w / w.sum()
    
    @property
    def active_factors(self):
        """权重 > 0.05 的因子索引"""
        w = self.weights
        return np.where(w > 0.05)[0]


def random_chromosome(n_factors):
    """随机初始化染色体"""
    # 宽范围均匀分布 [-3, 3], 确保 softmax 产生显著权重差异
    # 63因子时: max_weight ≈ e^3/(62*e^{-3}+e^3) ≈ 0.87
    # 即使均值附近也能稳定越过动态阈值 1/(2*N) ≈ 0.008
    return Chromosome(np.random.uniform(-3, 3, n_factors))


def icir_biased_chromosome(n_factors, icir_values, bias_strength=1.5):
    """
    ICIR偏置初始化染色体
    
    利用单因子ICIR作为先验知识, 给高ICIR因子正的初始权重偏差,
    低ICIR因子负的初始权重偏差, 使种群从"更可能有解"的区域开始搜索。
    
    NSGA-II的拥挤距离多样化机制会保护多样性, 不会过早收敛。
    
    Parameters
    ----------
    n_factors : int
        因子数量
    icir_values : array-like, shape (n_factors,)
        每个因子的单因子 ICIR 值 (已取绝对值, 越高越好)
    bias_strength : float
        偏置强度, 范围建议 [1.0, 2.0]
        - 1.0: 温和偏置, 高ICIR因子均值移至+1左右
        - 1.5: 适中偏置, 高ICIR因子均值移至+1.5左右 (推荐)
        - 2.0: 激进偏置, 高ICIR因子均值移至+2左右
    
    Returns
    -------
    Chromosome
        带有ICIR偏置的权重向量
    
    Algorithm
    ---------
    1. 将ICIR值rank化 → percentile → 映射到 [-1, 1]
    2. bias = bias_strength * normalized_rank
    3. 权重 ~ U(-3 + bias, 3 + bias), clipped to [-3, 3]
    
    效果:
    - Top ICIR因子: 权重分布在 [bias-3, bias+3] ≈ [bias, 3] (正值区)
    - Bottom ICIR因子: 权重分布在 [-3, bias+3] ≈ [-3, bias] (负值区)
    - 中等ICIR因子: 接近原始uniform分布
    """
    icir_arr = np.asarray(icir_values, dtype=float)
    if len(icir_arr) != n_factors:
        raise ValueError(f"icir_values length ({len(icir_arr)}) != n_factors ({n_factors})")
    
    # Rank化: 将ICIR映射到 [0, 1] percentile, 再映射到 [-1, 1]
    # 使用argsort + argsort 实现rank
    ranks = np.argsort(np.argsort(icir_arr))  # [0, n-1]
    if n_factors > 1:
        normalized = 2.0 * ranks / (n_factors - 1) - 1.0  # [-1, 1]
    else:
        normalized = np.zeros(1)
    
    bias = bias_strength * normalized
    
    # 生成偏置后的权重
    low = np.clip(-3.0 + bias, -5.0, 5.0)
    high = np.clip(3.0 + bias, -5.0, 5.0)
    weights = np.random.uniform(low, high)
    
    return Chromosome(weights)
