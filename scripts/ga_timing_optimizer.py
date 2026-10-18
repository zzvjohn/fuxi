# -*- coding: utf-8 -*-
"""
GA择时优化引擎 — 浙商《技术指标优化择时》完整实现
============================================================
流程:
  1. 加载 position_factor.csv → 计算每日组合收益
  2. 加载 HS300指数数据 → 计算技术指标
  3. GA 优化各指标参数 (Phase 1)
  4. 0/1规划 + GA 筛选最佳指标组合 (Phase 2)
  5. 输出最优参数 → 硬编码到聚宽策略

择时逻辑 (投票制):
  - 对每个已选指标, 计算 buy/sell 信号
  - buy_count > sell_count → 满仓 (state=1)
  - sell_count > buy_count → 空仓 (state=0)
  - 平票 → 维持前一日状态
"""
import numpy as np
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing_indicators import INDICATOR_REGISTRY, get_all_signals


# ========== 参数空间定义 ==========
# 每个指标的参数范围: (name, [(param_name, low, high, is_int), ...])
PARAM_SPACE = [
    ('SMA',    [('n', 5, 60, True)]),
    ('EMA',    [('n', 5, 60, True)]),
    ('EXPMA',  [('n', 5, 60, True)]),
    ('EWMA',   [('n', 5, 60, True), ('lambda', 0.85, 0.99, False)]),
    ('MA_CROSS', [('short', 2, 30, True), ('long', 10, 120, True)]),
    ('MACD',   [('n1', 5, 20, True), ('n2', 15, 50, True), ('n3', 5, 15, True)]),
    ('MACD_V2',[('n1', 5, 20, True), ('n2', 15, 50, True), ('n3', 5, 15, True)]),
    ('TRIX',   [('n1', 8, 20, True), ('n2', 10, 30, True)]),
    ('BOLL',   [('n', 10, 40, True), ('k', 1.5, 3.0, False)]),
    ('BOLL_V2',[('n', 10, 40, True), ('k', 1.5, 3.0, False)]),
    ('MTM',    [('n1', 5, 30, True), ('n2', 3, 15, True)]),
    ('SAR',    [('n', 3, 7, True), ('af0', 0.01, 0.05, False),
                ('step', 0.01, 0.05, False), ('af_max', 0.1, 0.3, False)]),
    ('DMI',    [('n', 5, 20, True)]),
    ('HL',     [('n', 10, 60, True)]),
    ('WMS',    [('n', 7, 30, True), ('ub', 70, 90, False), ('lb', 10, 30, False)]),
    ('KDJ',    [('n', 5, 20, True), ('m1', 2, 5, True), ('m2', 2, 5, True)]),
    ('KDJ_V2', [('n', 5, 20, True), ('m1', 2, 5, True), ('m2', 2, 5, True),
                ('ub', 75, 90, False), ('lb', 10, 25, False)]),
    ('RSI',    [('n', 7, 30, True), ('ub', 65, 85, False), ('lb', 15, 35, False)]),
    ('RSI_CR', [('ns', 4, 12, True), ('nl', 18, 36, True)]),
    ('PSY',    [('n', 8, 24, True), ('lb', 15, 35, False), ('ub', 65, 85, False)]),
    ('ARBR',   [('n', 14, 40, True), ('ar_lb', 40, 80, False), ('ar_ub', 120, 170, False),
                ('br_lb', 30, 70, False), ('br_ub', 130, 180, False)]),
    ('MFI',    [('n', 7, 28, True), ('ub', 75, 90, False), ('lb', 10, 25, False)]),
    ('VRSI',   [('n', 4, 14, True), ('ub', 75, 95, False), ('lb', 5, 25, False)]),
]


def params_to_tuple(name, param_dict):
    """将参数字典转换为函数需要的元组"""
    spec = dict(PARAM_SPACE)[name]
    return tuple(param_dict[p[0]] for p in spec)


def load_position_data(filepath):
    """加载持仓数据, 返回每日组合收益序列和总资产序列
    总资产 = 股票市值之和 + 现金"""
    cols = ['日期','品种','标的','多空','数量','可用数量',
            '收盘价','市值','盈亏','开仓均价','持仓均价','保证金',
            '当日盈亏','今手数','盈亏占比','总资产','仓位占比']
    df = pd.read_csv(filepath, encoding='gbk', names=cols, header=None, skiprows=1)
    df['日期'] = pd.to_datetime(df['日期'])
    
    # 股票部分
    stocks = df[df['品种'] == '股票'].copy()
    stocks['mv'] = pd.to_numeric(stocks['市值'], errors='coerce').fillna(0)
    daily_stock_mv = stocks.groupby('日期')['mv'].sum()
    
    # 现金部分
    cash_mask = df['标的'].astype(str).str.strip() == 'Cash'
    daily_cash = {}
    if cash_mask.any():
        for _, r in df[cash_mask].iterrows():
            d = r['日期']
            cash_val = pd.to_numeric(r['市值'], errors='coerce')  # 现金在市值列
            if pd.notna(cash_val) and cash_val > 0:
                daily_cash[d] = float(cash_val)
    
    cash_series = pd.Series(daily_cash).reindex(daily_stock_mv.index, fill_value=0)
    total_asset = daily_stock_mv + cash_series
    daily_ret = total_asset.pct_change().fillna(0)
    return daily_ret, total_asset


def load_index_data(filepath):
    """加载HS300指数日线数据, 返回标准OHLCV DataFrame"""
    df = pd.read_csv(filepath)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    # 列名映射: vol → volume
    if 'vol' in df.columns and 'volume' not in df.columns:
        df = df.rename(columns={'vol': 'volume'})
    return df


def simulate_timing(daily_ret, index_data, indicator_names, params_dict, buy_votes=None):
    """
    模拟择时增强后的收益
    
    daily_ret: Series, 日期索引, 每日组合收益
    index_data: DataFrame, HS300日线
    indicator_names: 已选指标列表
    params_dict: {name: param_tuple}
    buy_votes: 买入所需票数阈值 (None=简单多数)
    
    Returns: (total_return, sharpe, max_dd, annual_ret, win_rate)
    """
    n_indicators = len(indicator_names)
    if n_indicators == 0:
        return 0, 0, 0, 0, 0
    
    if buy_votes is None:
        buy_votes = n_indicators // 2 + 1  # 简单多数
    
    # 只取daily_ret日期范围内的数据
    valid_dates = daily_ret.index
    idx_dates = pd.to_datetime(index_data['trade_date'])
    
    # 建立日期到索引的映射
    idx_mask = idx_dates.isin(valid_dates)
    idx_subset = index_data[idx_mask].reset_index(drop=True)
    
    if len(idx_subset) < 100:
        return 0, 0, 0, 0, 0
    
    # 计算所有指标信号
    signals = get_all_signals(idx_subset, indicator_names, params_dict)
    
    # 投票制择时
    net_signals = np.zeros(len(idx_subset))
    for name in indicator_names:
        buy, sell = signals[name]
        net_signals += buy.astype(float) - sell.astype(float)
    
    # 状态机: 1=持仓, 0=空仓
    # ★ 修复前视偏差: 信号[t]使用了close[t], 只能决定 state[t], 作用于 ret[t+1]
    # 即 state[t] 使用的是状态机基于 signal[t] 新计算的值, 但应用于 ret[t+1]
    st = 1  # 初始满仓
    states = np.ones(len(idx_subset))
    for i in range(len(idx_subset)):
        states[i] = st  # 今日状态 = 昨日决策的结果
        if net_signals[i] >= buy_votes * 1.0:
            st = 1
        elif net_signals[i] <= -buy_votes * 1.0:
            st = 0
        # 否则维持
    
    # 计算择时后收益
    # 需要把index日期映射回daily_ret日期
    ret_dates = idx_subset['trade_date'].values
    timing_ret = []
    for i, d in enumerate(ret_dates):
        d_ts = pd.Timestamp(d)
        if d_ts in daily_ret.index:
            r = daily_ret.loc[d_ts]
            timing_ret.append(r * states[i])
    
    timing_ret = np.array(timing_ret)
    
    if len(timing_ret) < 50:
        return 0, 0, 0, 0, 0
    
    # 计算指标
    cum_ret = float(np.prod(1 + timing_ret) - 1)
    ann_ret = float((1 + cum_ret) ** (252 / len(timing_ret)) - 1)
    ann_vol = float(np.std(timing_ret) * np.sqrt(252))
    sharpe = float((ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0)
    
    # 最大回撤
    cum = np.cumprod(1 + timing_ret)
    peak = np.maximum.accumulate(cum)
    dd_vals = (cum - peak) / peak
    max_dd = float(np.min(dd_vals))
    
    win_rate = float((timing_ret > 0).mean())
    
    return cum_ret, sharpe, max_dd, ann_ret, win_rate


def simulate_timing_benchmark(daily_ret):
    """无择时的基准收益"""
    ret = daily_ret.values
    cum_ret = float(np.prod(1 + ret) - 1)
    ann_ret = float((1 + cum_ret) ** (252 / len(ret)) - 1)
    ann_vol = float(np.std(ret) * np.sqrt(252))
    sharpe = float((ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0)
    cum = np.cumprod(1 + ret)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum - peak) / peak))
    return cum_ret, sharpe, max_dd, ann_ret


# ========== 参数编码 ==========
def encode_params(indicator_names, params_dict):
    """将参数编码为向量 [0,1]^n"""
    vec = []
    bounds = []
    for name in indicator_names:
        spec = dict(PARAM_SPACE)[name]
        for p in spec:
            pname, lo, hi, is_int = p
            val = params_dict[name][pname]
            normalized = (val - lo) / (hi - lo)
            vec.append(normalized)
            bounds.append((lo, hi, is_int))
    return np.array(vec), bounds


def decode_params(vec, bounds):
    """将向量解码为参数"""
    params = []
    for i, (norm_val, (lo, hi, is_int)) in enumerate(zip(vec, bounds)):
        val = lo + norm_val * (hi - lo)
        if is_int:
            val = int(round(val))
        params.append(val)
    return params


def build_params_dict(indicator_names, flat_params):
    """将扁平参数列表重组为 {name: {pname: val}}"""
    params_dict = {}
    idx = 0
    for name in indicator_names:
        spec = dict(PARAM_SPACE)[name]
        p_dict = {}
        for p in spec:
            p_dict[p[0]] = flat_params[idx]
            idx += 1
        params_dict[name] = p_dict
    return params_dict


# ========== 遗传算法 ==========
class GATimingOptimizer:
    """GA择时优化器"""
    
    def __init__(self, daily_ret, index_data, indicator_names,
                 pop_size=100, generations=50, crossover_prob=0.8,
                 mutation_prob=0.1, elite_ratio=0.1):
        self.daily_ret = daily_ret
        self.index_data = index_data
        self.indicator_names = list(indicator_names)
        
        # 计算参数维度
        self.n_params = sum(len(dict(PARAM_SPACE)[n]) for n in self.indicator_names)
        self.n_indicators = len(self.indicator_names)
        
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_size = max(1, int(pop_size * elite_ratio))
        
        self.best_fitness = -np.inf
        self.best_individual = None
        self.history = []
    
    def decode_individual(self, individual):
        """将GA个体(参数向量)解码为参数字典和signal元组"""
        flat_params = decode_params(individual[:self.n_params], self.bounds)
        params_dict = build_params_dict(self.indicator_names, flat_params)
        params_tuples = {}
        for name in self.indicator_names:
            params_tuples[name] = params_to_tuple(name, params_dict[name])
        return params_tuples, params_dict
    
    def fitness(self, individual):
        """适应度函数: 择时后总收益"""
        try:
            params_tuples, _ = self.decode_individual(individual)
            cum_ret, sharpe, max_dd, ann_ret, win_rate = simulate_timing(
                self.daily_ret, self.index_data,
                self.indicator_names, params_tuples
            )
            # 综合评分: 总收益 + 夏普惩罚
            score = cum_ret * 0.7 + sharpe * 0.3
            return score
        except Exception as e:
            return -1e6
    
    def initialize_bounds(self):
        """初始化参数边界"""
        self.bounds = []
        for name in self.indicator_names:
            for p in dict(PARAM_SPACE)[name]:
                self.bounds.append((p[1], p[2], p[3]))
    
    def create_individual(self):
        """创建随机个体"""
        return np.random.random(self.n_params)
    
    def selection_tournament(self, population, fitnesses, k=3):
        """锦标赛选择"""
        selected = []
        for _ in range(len(population)):
            idx = np.random.choice(len(population), k)
            winner = idx[np.argmax([fitnesses[i] for i in idx])]
            selected.append(population[winner].copy())
        return selected
    
    def crossover(self, parent1, parent2):
        """模拟二进制交叉 (SBX-like)"""
        if np.random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = parent1.copy(), parent2.copy()
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                alpha = 0.5 + np.random.random() * 0.5
                child1[i] = alpha * parent1[i] + (1-alpha) * parent2[i]
                child2[i] = alpha * parent2[i] + (1-alpha) * parent1[i]
        return np.clip(child1, 0, 1), np.clip(child2, 0, 1)
    
    def mutation(self, individual):
        """高斯变异"""
        mutant = individual.copy()
        for i in range(len(mutant)):
            if np.random.random() < self.mutation_prob:
                mutant[i] += np.random.normal(0, 0.1)
        return np.clip(mutant, 0, 1)
    
    def optimize(self, verbose=True):
        """运行GA优化"""
        self.initialize_bounds()
        
        # 初始化种群
        population = [self.create_individual() for _ in range(self.pop_size)]
        
        for gen in range(self.generations):
            # 计算适应度
            fitnesses = [self.fitness(ind) for ind in population]
            
            # 记录最佳
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_individual = population[best_idx].copy()
            
            # 精英保留
            elite_idx = np.argsort(fitnesses)[-self.elite_size:]
            elites = [population[i].copy() for i in elite_idx]
            
            # 选择
            selected = self.selection_tournament(population, fitnesses)
            
            # 交叉
            children = []
            for i in range(0, self.pop_size - self.elite_size, 2):
                c1, c2 = self.crossover(selected[i], selected[i+1])
                children.extend([c1, c2])
            children = children[:self.pop_size - self.elite_size]
            
            # 变异
            children = [self.mutation(c) for c in children]
            
            # 新一代
            population = elites + children
            
            avg_fit = np.mean(fitnesses)
            self.history.append((gen, self.best_fitness, avg_fit))
            
            if verbose and gen % 5 == 0:
                print(f"  Gen {gen:3d}: Best={self.best_fitness:.4f} Avg={avg_fit:.4f}")
        
        return self.best_individual, self.best_fitness


# ========== 0/1规划 (指标选择) ==========
class ZOSelection:
    """0/1整数规划: 选择最佳指标组合"""
    
    def __init__(self, daily_ret, index_data, indicator_names, optimized_params,
                 pop_size=80, generations=30):
        self.daily_ret = daily_ret
        self.index_data = index_data
        self.indicator_names = list(indicator_names)
        self.optimized_params = optimized_params  # {name: param_tuple}
        self.n_indicators = len(indicator_names)
        self.pop_size = pop_size
        self.generations = generations
        
        self.best_fitness = -np.inf
        self.best_selection = None
        self.history = []
    
    def fitness(self, selection):
        """适应度: 选择subset后的择时收益"""
        selected_names = [self.indicator_names[i] for i, s in enumerate(selection) if s > 0.5]
        if len(selected_names) == 0:
            return -1e6
        
        params_dict = {name: self.optimized_params[name] for name in selected_names}
        cum_ret, sharpe, max_dd, ann_ret, win_rate = simulate_timing(
            self.daily_ret, self.index_data, selected_names, params_dict
        )
        # 惩罚过少指标 (至少3个)
        if len(selected_names) < 3:
            return cum_ret * 0.5 + sharpe * 0.3 - 0.1 * (3 - len(selected_names))
        return cum_ret * 0.6 + sharpe * 0.3 - max_dd * 0.1
    
    def optimize(self, verbose=True):
        """运行GA选择最佳指标组合"""
        # 初始化: 随机选择
        population = [np.random.random(self.n_indicators) for _ in range(self.pop_size)]
        elite_size = max(1, self.pop_size // 10)
        
        for gen in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in population]
            
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_selection = population[best_idx].copy()
            
            elites = [population[i].copy() for i in np.argsort(fitnesses)[-elite_size:]]
            
            # 选择 + 交叉 + 变异
            selected = []
            for _ in range(self.pop_size):
                idx = np.random.choice(len(population), 3)
                winner = idx[np.argmax([fitnesses[i] for i in idx])]
                selected.append(population[winner].copy())
            
            children = []
            for i in range(0, self.pop_size - elite_size, 2):
                if np.random.random() < 0.8:
                    alpha = np.random.random(self.n_indicators)
                    c1 = alpha * selected[i] + (1-alpha) * selected[i+1]
                    c2 = alpha * selected[i+1] + (1-alpha) * selected[i]
                else:
                    c1, c2 = selected[i].copy(), selected[i+1].copy()
                # 变异
                for c in [c1, c2]:
                    for j in range(self.n_indicators):
                        if np.random.random() < 0.1:
                            c[j] = np.random.random()
                children.extend([c1, c2])
            
            population = elites + children[:self.pop_size - elite_size]
            
            if verbose and gen % 5 == 0:
                print(f"  ZO Gen {gen:3d}: Best={self.best_fitness:.4f}")
        
        # 提取最终选择 (0/1)
        final_selection = (self.best_selection > 0.5).astype(int)
        selected_names = [self.indicator_names[i] for i, s in enumerate(final_selection) if s]
        return selected_names, final_selection, self.best_fitness


def main():
    """主流程"""
    print("=" * 60)
    print("GA择时优化引擎")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    daily_ret, daily_mv = load_position_data('E:/position2/position_factor.csv')
    index_data = load_index_data('e:/quant/data/raw/index_000300_daily.csv')
    
    print(f"  组合数据: {len(daily_ret)} 交易日, {daily_ret.index[0].strftime('%Y-%m-%d')} ~ {daily_ret.index[-1].strftime('%Y-%m-%d')}")
    print(f"  指数数据: {len(index_data)} 条")
    
    # 2. 基线
    print("\n[2/4] 计算基线 (无择时)...")
    baseline = simulate_timing_benchmark(daily_ret)
    print(f"  总收益: {baseline[0]*100:.2f}%")
    print(f"  夏普: {baseline[1]:.3f}")
    print(f"  最大回撤: {baseline[2]*100:.2f}%")
    print(f"  年化: {baseline[3]*100:.2f}%")
    
    # 3. GA参数优化 (所有均线类指标)
    print("\n[3/4] GA参数优化...")
    trend_indicators = ['SMA', 'EMA', 'EXPMA', 'MACD', 'TRIX', 'BOLL', 'MTM', 'SAR', 'DMI', 'HL',
                        'WMS', 'KDJ', 'RSI', 'PSY', 'MFI', 'VRSI']
    
    optimizer = GATimingOptimizer(
        daily_ret, index_data, trend_indicators,
        pop_size=100, generations=40
    )
    best_ind, best_fit = optimizer.optimize(verbose=True)
    
    params_tuples, params_dict = optimizer.decode_individual(best_ind)
    print(f"\n  最佳参数:")
    for name in trend_indicators:
        print(f"    {name}: {params_dict[name]}")
    
    # 4. 0/1规划筛选组合
    print("\n[4/4] 0/1规划筛选指标组合...")
    zo = ZOSelection(
        daily_ret, index_data, trend_indicators, params_tuples,
        pop_size=80, generations=30
    )
    selected_names, selection, zo_fit = zo.optimize(verbose=True)
    
    print(f"\n  {'='*60}")
    print(f"  最佳指标组合 ({len(selected_names)}个):")
    for name in selected_names:
        print(f"    {name}: {params_dict[name]}")
    
    # 最终验证
    final_params = {name: params_tuples[name] for name in selected_names}
    final_ret, final_sharpe, final_dd, final_ann, final_wr = simulate_timing(
        daily_ret, index_data, selected_names, final_params
    )
    
    print(f"\n  {'='*60}")
    print(f"  最终结果:")
    print(f"    总收益: {final_ret*100:.2f}% (基线: {baseline[0]*100:.2f}%)")
    print(f"    夏普: {final_sharpe:.3f} (基线: {baseline[1]:.3f})")
    print(f"    最大回撤: {final_dd*100:.2f}% (基线: {baseline[2]*100:.2f}%)")
    print(f"    年化: {final_ann*100:.2f}%")
    print(f"    胜率: {final_wr*100:.1f}%")
    
    # 保存结果
    result = {
        'selected_indicators': selected_names,
        'params': {name: params_tuples[name] for name in selected_names},
        'total_return': final_ret,
        'sharpe': final_sharpe,
        'max_dd': final_dd,
        'annual_return': final_ann,
        'win_rate': final_wr,
        'baseline_return': baseline[0],
        'baseline_sharpe': baseline[1],
        'baseline_max_dd': baseline[2],
    }
    
    import json
    with open('e:/quant/reports/ga_timing_result.json', 'w', encoding='utf-8') as f:
        # 自定义JSON序列化
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tuple):
                return list(obj)
            return obj
        json.dump(result, f, default=convert, indent=2, ensure_ascii=False)
    
    print(f"\n  结果已保存: e:/quant/reports/ga_timing_result.json")
    return result


if __name__ == '__main__':
    main()
