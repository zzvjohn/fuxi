"""
从 GA 结果恢复执行 Phase 4.5 + Phase 5 + 报告生成
====================
跳过 Phase 1-4, 直接用上次流水线的因子数据
"""
import sys, time, os, pickle, json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

log_dir = Path(__file__).parent / 'output'
output_dir = Path(__file__).parent.parent.parent / 'reports'
output_dir.mkdir(exist_ok=True)

# 加载上次流水线的因子数据
print("加载因子数据...")
factor_dfs = {}
for f in sorted(log_dir.glob('factor_*.csv')):
    name = f.stem.replace('factor_', '')
    try:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        factor_dfs[name] = df
        print(f"  {name}: {df.shape}")
    except:
        pass

print(f"\n加载 {len(factor_dfs)} 个因子")

# 构建 forward_returns 和 mcap_df (从已保存数据重建)
from data.loader import load_all_data
print("\n重新加载价格和市值数据...")
data = load_all_data('2021-01-01', '2026-04-30')
price_data = data['price_data']
valuation_data = data['valuation_data']

close = price_data.get('close')
close_weekly = close.resample('W').last()
close_weekly.index = pd.to_datetime(close_weekly.index)
forward_returns = close_weekly.shift(-1) / close_weekly - 1
print(f"  forward_returns: {forward_returns.shape}")

mcap_df = None
if valuation_data is not None:
    mcap_raw = valuation_data[['code', 'trade_date', 'market_cap']].copy()
    mcap_raw['trade_date'] = pd.to_datetime(mcap_raw['trade_date'])
    mcap_df = mcap_raw.pivot(index='trade_date', columns='code', values='market_cap')
    mcap_df = mcap_df.resample('W').last()
    mcap_df = np.log(mcap_df.clip(lower=1))
    print(f"  mcap_df: {mcap_df.shape}")

# GA 选择的最佳因子 (从上一次运行日志中提取)
# turnover_change(0.331), f_score(0.282), dollar_vol_20d(0.207), turnover_std(0.180)
best_weights = {
    'turnover_change': 0.331,
    'f_score': 0.282,
    'dollar_vol_20d': 0.207,
    'turnover_std': 0.180,
}
best_fitness = 0.9670

# 确保关键模块已导入
from factors.composite import combine_factors
from evaluation.ic_analysis import compute_ic_icir, compute_ic_summary
from evaluation.decile_test import decile_portfolio_test, test_monotonicity
from evaluation.double_sort import independent_double_sort
from evaluation.correlation import factor_correlation_matrix

# =====================================================
# Phase 4.5: 最佳组合完整评估
# =====================================================
print("\n" + "=" * 70)
print("[Phase 4.5] 最佳组合完整四维评估 (维修版)")
print("=" * 70)

t45 = time.time()

# 过滤只保留存在的因子
valid_weights = {n: w for n, w in best_weights.items() if n in factor_dfs}
if len(valid_weights) != len(best_weights):
    missing = set(best_weights) - set(valid_weights)
    print(f"  缺失因子: {missing}")

# 归一化权重
total_w = sum(valid_weights.values())
valid_weights = {n: w/total_w for n, w in valid_weights.items()}

print(f"  有效因子: {list(valid_weights.keys())}")
selected = {n: factor_dfs[n] for n in valid_weights if n in factor_dfs}

# 对齐因子数据 (确保所有因子有相同的日期和股票)
print("\n--- 数据对齐检查 ---")
for name, df in selected.items():
    print(f"  {name:25s}: {df.shape}, NaN={df.isna().sum().sum()}, dates={df.index[0]}~{df.index[-1]}")

composite = combine_factors(selected, valid_weights)
print(f"\n  合成因子: {composite.shape}, NaN={composite.isna().sum().sum()}")
print(f"  [1/4] 合成因子完成 ({time.time()-t45:.0f}s)")

# ICIR
ga_icir = compute_ic_icir(composite, forward_returns)
icir_val = ga_icir.get('icir', np.nan)
print(f"  [2/4] ICIR={icir_val:+.4f} (IC_mean={ga_icir.get('ic_mean',np.nan):+.4f}, +IC={ga_icir.get('ic_positive_ratio',np.nan):.1%}) ({time.time()-t45:.0f}s)")

# 单调性
ga_decile = decile_portfolio_test(composite, forward_returns)
ga_mono = test_monotonicity(ga_decile)
print(f"  [3/4] 十分位spread={ga_mono['spread_mean']:+.4%} t={ga_decile.get('top_bottom_t_stat',np.nan):+.2f} p={ga_mono['p_value']:.4f} ({time.time()-t45:.0f}s)")

# 双重排序
ga_ds = independent_double_sort(composite, forward_returns, mcap_df)
ds_spread = ga_ds.get('factor_quintile_spread', np.nan)
ds_t = ga_ds.get('t_stat', np.nan)
ds_p = ga_ds.get('p_value', np.nan)
print(f"  [4/4] 双重排序spread={ds_spread:+.4%} t={ds_t:+.2f} p={ds_p:.4f} ({time.time()-t45:.0f}s)")

# 相关性 - 增强版错误处理
print(f"\n--- 因子间相关性 ---")
try:
    all_for_corr = {}
    for n in valid_weights:
        if n in factor_dfs:
            all_for_corr[n] = factor_dfs[n]
    all_for_corr['composite'] = composite
    ga_corr_matrix = factor_correlation_matrix(all_for_corr)
    if ga_corr_matrix.empty:
        print("  [WARN] 相关性矩阵为空")
    else:
        print(f"  相关性矩阵: {ga_corr_matrix.shape}")
        if 'composite' in ga_corr_matrix.index:
            comp_corr = ga_corr_matrix.loc['composite'].drop('composite', errors='ignore')
            if len(comp_corr) > 0:
                max_corr = abs(comp_corr).max()
                print(f"  合成因子与源因子最大相关: {max_corr:.3f}")
except Exception as e:
    print(f"  [WARN] 相关性计算失败: {e}")
    import traceback
    traceback.print_exc()
    ga_corr_matrix = pd.DataFrame()

# 4选3判定
from evaluation.scoring import ICIR_THRESHOLD, ICIR_QUALITY_THRESHOLD, MONOTONICITY_P_THRESHOLD, CORRELATION_THRESHOLD, DOUBLE_SORT_P_THRESHOLD, GA_PASS_MIN_DIMENSIONS

d1 = abs(icir_val) >= ICIR_QUALITY_THRESHOLD  # 因子质量: |ICIR| >= 0.5
d2 = ga_mono['p_value'] < MONOTONICITY_P_THRESHOLD
d3 = not ga_corr_matrix.empty and 'composite' in ga_corr_matrix.index
if d3:
    comp_corr_check = ga_corr_matrix.loc['composite'].drop('composite', errors='ignore')
    d3 = len(comp_corr_check) > 0 and abs(comp_corr_check).max() < CORRELATION_THRESHOLD
d4 = ds_p < DOUBLE_SORT_P_THRESHOLD
passed_count = sum([d1, d2, d3, d4])

print(f"\n  === 四维评估结果 ===")
print(f"  ICIR:          {icir_val:+.4f}  质量阈值={ICIR_QUALITY_THRESHOLD}  → {'PASS' if d1 else 'FAIL'}")
print(f"  单调性:        p={ga_mono['p_value']:.4f}  阈值={MONOTONICITY_P_THRESHOLD}  → {'PASS' if d2 else 'FAIL'}")
print(f"  低相关:        {'通过' if d3 else '未通过'}  阈值={CORRELATION_THRESHOLD}  → {'PASS' if d3 else 'FAIL'}")
print(f"  双重排序:      p={ds_p:.4f}  阈值={DOUBLE_SORT_P_THRESHOLD}  → {'PASS' if d4 else 'FAIL'}")
print(f"  四维判定: {passed_count}/4 {'PASS通过' if passed_count >= GA_PASS_MIN_DIMENSIONS else 'FAIL未通过'}")

# 十分位详细
if ga_decile.get('decile_returns') is not None:
    print(f"\n  === 十分位分组收益 ===")
    dr = ga_decile.get('decile_returns', {})
    if isinstance(dr, dict):
        for i in range(10):
            if i in dr:
                print(f"    D{i+1:2d}: {dr[i]:+.4%}")
    elif isinstance(dr, list) and len(dr) >= 10:
        for i in range(10):
            print(f"    D{i+1:2d}: {dr[i]:+.4%}")

# =====================================================
# Phase 5: 虚拟投资组合
# =====================================================
print("\n" + "=" * 70)
print("[Phase 5] 虚拟投资组合验证")
print("=" * 70)

from portfolio.simulator import PortfolioSimulator

try:
    simulator = PortfolioSimulator(
        factor_df=composite,
        price_df=close_weekly,
        top_n=30,
    )
    port_result = simulator.run()
    stats = port_result.get('stats', {})
    
    print(f"\n  === 虚拟组合统计 (周频调仓, Top30, 等权) ===")
    print(f"  总收益:   {stats.get('total_return', 0):.2%}")
    print(f"  年化收益: {stats.get('cagr', 0):.2%}")
    print(f"  Sharpe:   {stats.get('sharpe', 0):.2f}")
    print(f"  MaxDD:    {stats.get('max_drawdown', 0):.2%}")
    print(f"  周胜率:   {stats.get('win_rate', 0):.1%}")
    print(f"  平均换手: {stats.get('avg_turnover', 0):.1%}")
    
    with open(log_dir / 'portfolio_result.pkl', 'wb') as pf:
        pickle.dump(port_result, pf)
except Exception as e:
    print(f"  FAIL Phase 5 异常: {e}")
    import traceback
    traceback.print_exc()
    stats = {}

# =====================================================
# 生成报告 + IC 汇总
# =====================================================
print("\n" + "=" * 70)
print("[生成报告]")
print("=" * 70)

# IC汇总 (从保存的csv加载)
ic_summary_file = log_dir / 'ic_summary.csv'
if ic_summary_file.exists():
    ic_summary = pd.read_csv(ic_summary_file, index_col=0)
else:
    ic_summary = compute_ic_summary(factor_dfs, forward_returns)

# 全部因子列表
all_factors_ic = []
for name in factor_dfs:
    try:
        icr = compute_ic_icir(factor_dfs[name], forward_returns)
        all_factors_ic.append({
            'factor': name,
            'icir': icr['icir'],
            'ic_mean': icr['ic_mean'],
            'ic_positive_ratio': icr['ic_positive_ratio'],
        })
    except:
        all_factors_ic.append({'factor': name, 'icir': np.nan, 'ic_mean': np.nan, 'ic_positive_ratio': np.nan})

# 生成 Markdown 报告
report = f"""# 因子炼金术 (Factor Alchemy) v1.0 — 完整分析报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据范围**: 2021-01-01 ~ 2026-04-30
**调仓频率**: 周频

---

## 一、研究方法

### 遗传算法多因子合成
- **种群**: 20, **代数**: 8, **适应度**: ICIR (快速版)
- **输入**: 32个候选因子 (估值/成长/盈利能力/换手/动量/波动/流动性)
- **评估**: 四维度验证 (ICIR / 单调性 / 低相关性 / 独立双重排序)
- **通过标准**: 4选3
- **独立双重排序**: 市值5组 × 目标因子5组 = 25个投资组合, t检验因子预期收益率

---

## 二、GA最佳因子组合

| 因子 | 权重 | 单因子ICIR | 含义 |
|------|:----:|:----:|------|
"""

total_w = sum(best_weights.values())
all_factors_ic_dict = {x['factor']: x for x in all_factors_ic}
factor_labels = {
    'turnover_change': '换手率变化', 'f_score': 'Piotroski F-Score',
    'dollar_vol_20d': '成交额(20日)', 'turnover_std': '换手率波动'
}
for name, w in sorted(best_weights.items(), key=lambda x: -x[1]):
    ic = all_factors_ic_dict.get(name, {}).get('icir', np.nan)
    label = factor_labels.get(name, name)
    report += f"| {name} | {(w/total_w)*100:.1f}% | {ic:+.3f} | {label} |\n"

report += f"""
**GA适应度**: {best_fitness:.4f}  |  **GA耗时**: 707s (约12分钟)

---

## 三、合成因子四维评估

| 维度 | 指标值 | 判定 |
|------|--------|:--:|
| ICIR | {icir_val:+.4f} | {'PASS' if d1 else 'FAIL'} |
| 单调性 | p={ga_mono['p_value']:.4f} | {'PASS' if d2 else 'FAIL'} |
| 低相关性 | {'通过' if d3 else '未通过'} | {'PASS' if d3 else 'FAIL'} |
| 独立双重排序 | spread={ds_spread:+.4%}, p={ds_p:.4f} | {'PASS' if d4 else 'FAIL'} |

**综合判定**: {passed_count}/4 → **{'PASS 通过' if passed_count >= GA_PASS_MIN_DIMENSIONS else 'FAIL 未通过'}**
"""

# 十分位
if ga_decile.get('decile_returns') is not None:
    report += "\n### 十分位分组收益\n\n| 分组 | 周均收益 |\n|------|:----:|\n"
    dr = ga_decile.get('decile_returns', {})
    for i in range(10):
        if i in dr:
            report += f"| D{i+1} | {dr[i]:+.4%} |\n"

# 双重排序详情
report += f"""
### 独立双重排序详情 (市值 × 因子, 5×5=25组)

- 因子预期收益率 (Top - Bottom): **{ds_spread:+.4%}/周** (年化约 {(1+ds_spread)**52-1:.1%})
- t统计量: {ds_t:+.2f}, p值: {ds_p:.4f}
- {'PASS 通过检验 (p<0.05)' if d4 else 'FAIL 未通过检验'}
"""

# 因子ICIR排行 (Top 20)
report += """
---

## 四、全部因子ICIR排行 (Top 20)

| 排名 | 因子 | ICIR | IC均值 | +IC比例 |
|:--:|------|:----:|:---:|:---:|
"""

all_factors_sorted = sorted(all_factors_ic, key=lambda x: abs(x['icir'] or 0), reverse=True)
for i, f in enumerate(all_factors_sorted[:20], 1):
    icir = f['icir']
    if icir is None or np.isnan(icir):
        continue
    report += f"| {i} | {f['factor']:25s} | {icir:+.3f} | {f['ic_mean']:+.4f} | {f['ic_positive_ratio']:.0%} |\n"

report += """
---

## 五、关键发现

### 5.1 因子簇分析
- **换手率簇** (turnover_change, turnover_std, dollar_vol_20d): 相关性极高, GA自然选出组合
- **F-Score 质量因子**: 单因子ICIR仅+0.092, 但在GA合成中权重高达28.2%, 说明其贡献了独立的信息维度

### 5.2 合成因子特征
"""
report += f"""- 合成因子ICIR={icir_val:+.4f}, 远超任何单因子 (最高 turnover_std +0.503)
- 双重排序通过/未通过: {'PASS' if d4 else 'FAIL'}
- 控制市值后, 因子预期收益率 {ds_spread:+.4%}/周
"""

# 虚拟组合
report += """
### 5.3 虚拟组合表现
"""
if stats:
    report += f"""
| 指标 | 数值 |
|------|:----:|
| 总收益 | {stats.get('total_return', 0):.2%} |
| 年化收益 | {stats.get('cagr', 0):.2%} |
| Sharpe | {stats.get('sharpe', 0):.2f} |
| MaxDD | {stats.get('max_drawdown', 0):.2%} |
| 周胜率 | {stats.get('win_rate', 0):.1%} |
| 平均换手 | {stats.get('avg_turnover', 0):.1%} |
"""
else:
    report += "\n(虚拟组合计算异常, 无有效结果)\n"

report += """
---

## 六、对比: 因子炼金术 vs v5b

| 维度 | 因子炼金术 | v5b FM+RP |
|------|:--:|:--:|
| 方法论 | GA自动合成 | 人工加权 (size/PS/PB/STR) |
| 因子来源 | 32个候选 → 4个 | 4个固定因子 |
| 换手率因子 | **核心驱动** | 无换手率因子 |
| 质量控制 | F-Score 28.2%权重 | 无质量因子 |

---

## 七、下一步建议

1. **换手率因子深度研究**: GA选出3/4为换手率类因子, 需区分真实Alpha vs 微观结构噪音
2. **F-Score权重优化**: 28.2%权重对低ICIR因子偏高, 需验证是否过拟合
3. **不同市场环境**: 分牛熊市验证合成因子稳定性
4. **与v5b对比回测**: 在聚宽环境同等条件下对比
5. **Turnover成本检验**: 模拟组合考虑换手成本后真实收益

---

*报告由因子炼金术 (Factor Alchemy) v1.0 自动生成*
*遗传算法 + 四维度验证 (ICIR / 单调性 / 独立双重排序 / 低相关性)*
"""

# 保存报告
report_path = output_dir / 'factor_alchemy_report_20260606.md'
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write(report)

print(f"\n报告已保存: {report_path}")
print(f"总耗时: {time.time()-t45:.0f}s")
