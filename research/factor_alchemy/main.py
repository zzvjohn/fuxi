"""
因子炼金术 — 主流水线
======================
1. 数据加载
2. 因子计算
3. 单因子评估
4. GA 因子组合
5. 虚拟投资组合验证
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def run_pipeline(start_date='2021-01-01', end_date='2026-04-30', 
                  run_ga=True, verbose=True):
    """
    运行完整流水线
    
    Parameters
    ----------
    start_date : str
    end_date : str
    run_ga : bool
        是否运行GA (False=仅做单因子评估)
    verbose : bool
    
    Returns
    -------
    dict
        {
            'single_factor_results': pd.DataFrame,
            'ga_results': dict (if run_ga),
            'portfolio_results': dict (if run_ga),
        }
    """
    from config import FACTOR_DEFS
    from factors import ALL_FACTORS
    from data.loader import load_all_data
    
    print("=" * 70)
    print("  因子炼金术 (Factor Alchemy) v1.0")
    print(f"  日期范围: {start_date} ~ {end_date}")
    print("=" * 70)
    
    # ==================== Phase 1: 数据加载 ====================
    print("\n[Phase 1] 加载数据...")
    data = load_all_data(start_date, end_date)
    
    price_data = data['price_data']       # dict of DataFrames
    financial_data = data['financial_data']
    valuation_data = data['valuation_data']
    industry_map = data.get('industry_map', {})
    
    close = price_data.get('close')
    if close is None or close.empty:
        print("[ERROR] 无价格数据, 退出")
        return None
    
    print(f"  价格: {close.shape}")
    print(f"  财务: {financial_data.shape if hasattr(financial_data, 'shape') else 'N/A'}")
    print(f"  估值: {valuation_data.shape if hasattr(valuation_data, 'shape') else 'N/A'}")
    
    # 构建周频前向收益
    close_weekly = close.resample('W').last()
    close_weekly.index = pd.to_datetime(close_weekly.index)
    forward_returns = close_weekly.shift(-1) / close_weekly - 1
    
    # 市值数据
    mcap_df = None
    if valuation_data is not None and hasattr(valuation_data, 'columns'):
        if 'market_cap' in valuation_data.columns:
            mcap_raw = valuation_data[['code', 'trade_date', 'market_cap']].copy()
            mcap_raw['trade_date'] = pd.to_datetime(mcap_raw['trade_date'])
            mcap_df = mcap_raw.pivot(index='trade_date', columns='code', values='market_cap')
            mcap_df = mcap_df.resample('W').last()
    
    # ==================== Phase 2: 因子计算 ====================
    print("\n[Phase 2] 计算因子...")
    
    factor_dfs = {}
    is_size_map = {}
    
    for name in FACTOR_DEFS:
        if name not in ALL_FACTORS:
            continue
        
        factor_def = FACTOR_DEFS[name]
        factor_obj = ALL_FACTORS[name]()
        
        print(f"  {name:25s} ({factor_def['label']})...", end=' ')
        try:
            df = factor_obj.compute(
                price_data=price_data,
                financial_data=financial_data,
                valuation_data=valuation_data,
                industry_map=industry_map,
            )
            if df is not None and not df.empty:
                # 重采样到周频
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df_weekly = df.resample('W').last()
                factor_dfs[name] = df_weekly
                is_size_map[name] = (factor_def['category'] == 'size')
                print(f"OK ({df_weekly.shape[0]}周 × {df_weekly.shape[1]}股)")
            else:
                print(f"SKIP (无数据)")
        except Exception as e:
            print(f"FAIL ({str(e)[:50]})")
    
    if len(factor_dfs) == 0:
        print("[ERROR] 无有效因子")
        return None
    
    print(f"\n  成功计算 {len(factor_dfs)}/{len(FACTOR_DEFS)} 个因子")
    
    # ==================== Phase 3: 单因子评估 ====================
    print("\n[Phase 3] 单因子评估...")
    
    from evaluation.ic_analysis import compute_ic_summary
    from evaluation.decile_test import decile_portfolio_test, test_monotonicity
    from evaluation.double_sort import independent_double_sort, size_single_sort
    from evaluation.correlation import factor_correlation_matrix
    from evaluation.stability import compute_stability_comprehensive, print_stability_report
    
    # IC/ICIR
    ic_summary = compute_ic_summary(factor_dfs, forward_returns)
    print(f"\n  === IC/ICIR 排名 (Top 10) ===")
    top_ic = ic_summary.head(10)
    for idx, row in top_ic.iterrows():
        print(f"  {idx:25s} ICIR={row['ICIR']:+.3f}  IC_mean={row['IC_mean']:+.4f}  +IC%={row['+IC%']:.1%}")
    
    # 相关性矩阵
    corr_matrix = factor_correlation_matrix(factor_dfs)
    print(f"\n  因子相关性矩阵: {corr_matrix.shape}")
    
    # 双重排序 (选几个做示例)
    if mcap_df is not None:
        print(f"\n  === 独立双重排序 (示例) ===")
        for name in list(factor_dfs.keys())[:3]:
            if is_size_map.get(name, False):
                ds = size_single_sort(mcap_df, forward_returns)
                print(f"  {name:25s} (规模单变量) size_premium={ds['size_premium']:.4%}  t={ds['t_stat']:+.2f}  p={ds['p_value']:.3f}")
            else:
                ds = independent_double_sort(factor_dfs[name], forward_returns, mcap_df)
                print(f"  {name:25s} spread={ds['factor_quintile_spread']:.4%}  t={ds['t_stat']:+.2f}  p={ds['p_value']:.3f}")
    
    # ==================== Phase 4: GA 因子组合 ====================
    ga_result = None
    port_result = None
    
    if run_ga and mcap_df is not None and len(factor_dfs) >= 3:
        print("\n[Phase 4] 遗传算法因子组合...")
        
        from ga import FactorGA
        
        factor_names = list(factor_dfs.keys())
        
        ga = FactorGA(
            factor_names=factor_names,
            factor_dict=factor_dfs,
            forward_returns=forward_returns,
            mcap_df=mcap_df,
            is_size_map=is_size_map,
            population_size=200,
            generations=50,  # 先用50代测试
        )
        
        best_chromo, best_fitness = ga.evolve(verbose=verbose)
        
        # 最佳权重
        from factors.composite import weights_from_chromosome
        best_weights = weights_from_chromosome(best_chromo, factor_names)
        
        print(f"\n  === 最佳因子组合 ===")
        print(f"  适应度: {best_fitness:.4f}")
        print(f"  入选因子 ({len(best_weights)}个):")
        for name, w in sorted(best_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name:25s} = {w:.3f}")
        
        ga_result = {
            'best_chromosome': best_chromo,
            'best_fitness': best_fitness,
            'best_weights': best_weights,
            'history': ga.to_dataframe(),
        }
        
        # ==================== Phase 5: 虚拟投资组合 ====================
        print("\n[Phase 5] 虚拟投资组合验证...")
        
        from factors.composite import combine_factors
        from portfolio.simulator import PortfolioSimulator
        
        # 用最佳权重合成因子
        selected = {name: factor_dfs[name] for name in best_weights if name in factor_dfs}
        composite = combine_factors(selected, best_weights)
        
        simulator = PortfolioSimulator(
            factor_df=composite,
            price_df=close_weekly,
            top_n=30,
        )
        
        port_result = simulator.run()
        stats = port_result.get('stats', {})
        
        print(f"\n  === 虚拟组合统计 ===")
        print(f"  总收益:   {stats.get('total_return', 0):.2%}")
        print(f"  年化收益: {stats.get('cagr', 0):.2%}")
        print(f"  Sharpe:   {stats.get('sharpe', 0):.2f}")
        print(f"  MaxDD:    {stats.get('max_drawdown', 0):.2%}")
        print(f"  周胜率:   {stats.get('win_rate', 0):.1%}")
        print(f"  平均换手: {stats.get('avg_turnover', 0):.1%}")
    
    # 保存结果
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    ic_summary.to_csv(output_dir / 'ic_summary.csv', encoding='utf-8-sig')
    if not corr_matrix.empty:
        corr_matrix.to_csv(output_dir / 'factor_correlation.csv', encoding='utf-8-sig')
    
    print(f"\n  结果已保存到: {output_dir}")
    
    return {
        'ic_summary': ic_summary,
        'factor_dfs': factor_dfs,
        'corr_matrix': corr_matrix,
        'ga_result': ga_result,
        'portfolio_result': port_result,
    }


if __name__ == '__main__':
    result = run_pipeline()
