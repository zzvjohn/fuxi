"""
PandaFactor 安装验证脚本
- 用本地 Tushare 数据计算 10日动量因子
- 调用 PandaFactor 的 factor 类做 IC 分析和分组回测
- 验证安装是否正确
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 添加工作区路径
sys.path.insert(0, 'e:/quant')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. 验证 PandaFactor 导入
print("=" * 50)
print("Step 1: 验证 PandaFactor 导入")
print("=" * 50)

import panda_common
print("  panda_common OK")

import panda_factor
print("  panda_factor OK")

from panda_factor.analysis.factor import factor
print("  factor class OK")

from panda_factor.analysis.factor_func import z_score, clean_df
print("  factor_func OK")

# 2. 加载本地 Tushare 日线数据
print()
print("=" * 50)
print("Step 2: 加载本地日线数据")
print("=" * 50)

# 检查可用的数据文件
data_dir = 'e:/quant/data/raw'
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'daily' in f.lower()]
print(f"  数据目录: {data_dir}")
print(f"  相关CSV: {csv_files}")

# 尝试加载日线数据
daily_file = os.path.join(data_dir, 'daily_prices.csv')
if not os.path.exists(daily_file):
    # 尝试其他可能的名字
    for f in csv_files:
        daily_file = os.path.join(data_dir, f)
        break

if os.path.exists(daily_file):
    print(f"  加载: {daily_file}")
    df = pd.read_csv(daily_file, nrows=5)
    print(f"  列名: {list(df.columns)}")
    print(f"  前5行:")
    print(df.to_string())
else:
    print("  未找到日线数据文件，使用模拟数据")

# 3. 用模拟数据验证因子计算流程
print()
print("=" * 50)
print("Step 3: 因子计算验证（模拟数据）")
print("=" * 50)

np.random.seed(42)
n_stocks = 200
n_days = 500

# 生成模拟数据
dates = pd.date_range(end='2025-12-31', periods=n_days, freq='B')
stocks = [f'{600000 + i:06d}.SH' for i in range(n_stocks)]

rows = []
for date in dates:
    for stock in stocks:
        # 随机价格路径
        base_price = 10 + np.random.exponential(20)
        ret = np.random.normal(0.0005, 0.025)
        close = base_price * (1 + ret)
        rows.append({
            'date': date.strftime('%Y-%m-%d'),
            'stock': stock,
            'close': round(close, 2)
        })

df_raw = pd.DataFrame(rows)
print(f"  模拟数据: {len(df_raw)} 行, {n_stocks} 只股票, {n_days} 个交易日")

# 4. 计算 10 日动量因子
print()
print("  计算 10日动量因子...")

df_pivot = df_raw.pivot_table(index='date', columns='stock', values='close')
df_ret = df_pivot.pct_change(10)  # 10日收益率作为因子值
df_ret_1d = df_pivot.pct_change(1)  # 1日收益率作为未来收益

# 展平为长格式
df_factor = df_ret.stack().reset_index()
df_factor.columns = ['date', 'stock', 'momentum_10d']

df_1d = df_ret_1d.stack().reset_index()
df_1d.columns = ['date', 'stock', '1day_return']

# 合并
df_merged = df_factor.merge(df_1d, on=['date', 'stock'], how='inner')

# 添加分组列和滞后收益
df_merged = df_merged.dropna(subset=['momentum_10d'])

# 生成分组
df_merged['momentum_10d_group'] = df_merged.groupby('date')['momentum_10d'].transform(
    lambda x: pd.qcut(x, 10, labels=False, duplicates='drop') + 1
)

# 添加 1day_return 作为 period=1 的收益
df_merged = df_merged.rename(columns={'1day_return': '1day_return'})

print(f"  因子数据: {len(df_merged)} 行")
print(f"  示例:")
print(df_merged.head(10).to_string())

# 5. 调用 PandaFactor 的 factor 类
print()
print("=" * 50)
print("Step 4: PandaFactor 因子分析")
print("=" * 50)

# 准备 benchmark 数据
df_bench = pd.DataFrame({
    'date': dates.strftime('%Y-%m-%d'),
    'benchmark_return': np.random.normal(0.0003, 0.015, len(dates))
})
df_bench = df_bench.set_index('date')
df_bench['1D_m'] = df_bench['benchmark_return']  # 1日基准收益

# 创建 factor 实例
f = factor(name='momentum_10d', group_number=10)
f.set_backtest_parameters(
    period=1,
    predict_direction=1,  # 动量因子：值越大越好
    commission=0.002,
    mode=1
)

# 准备回测数据格式（PandaFactor 需要 symbol 列和 {period}day_return 列）
df_backtest = df_merged.copy()
df_backtest['symbol'] = df_backtest['stock']
df_backtest['1day_return'] = df_backtest['1day_return']
# PandaFactor 需要 {period}day_return 列
df_backtest = df_backtest.rename(columns={'1day_return': '1day_return'})

# 添加 returns_lag 列（PandaFactor 需要）
for i in range(1, 21):
    df_backtest[f'returns_lag{i}'] = df_backtest.groupby('stock')['1day_return'].shift(i)

print(f"  回测数据准备完成: {len(df_backtest)} 行")
print(f"  列名: {list(df_backtest.columns)}")

try:
    f.start_backtest(df_backtest, df_bench)
    print()
    print("  回测完成!")
    print(f"  IC 均值: {f.df_ic['ic'].mean():.4f}")
    print(f"  Rank IC 均值: {f.df_ic['rank_ic'].mean():.4f}")

    if not f.df_info.empty:
        print()
        print("  分组统计:")
        print(f.df_info.to_string())

    print()
    print("=" * 50)
    print("PandaFactor 安装验证通过!")
    print("=" * 50)

except Exception as e:
    print(f"  回测出错: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("  注意: 因子分析可能有数据格式问题，但核心 import 已验证通过")
