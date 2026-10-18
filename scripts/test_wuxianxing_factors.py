"""
吴先兴五维成长因子测试 (2026-06-14研报)
========================================
测试 novel_044~048 五个量价代理因子在日频和周频的IC表现
数据: daily_prices.csv / weekly data
复现: 吴先兴成长股量化分析核心因子 — 五维成长框架
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
REPORT_DIR = PROJECT_ROOT / 'reports'
REPORT_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("  吴先兴五维成长因子测试")
print("=" * 60)

# 读取日频价格
prices = pd.read_csv(DATA_DIR / 'daily_prices.csv')
prices['trade_date'] = pd.to_datetime(prices['trade_date'].astype(str), format='mixed')
prices = prices.sort_values(['ts_code', 'trade_date'])
print(f"  日频数据: {prices.shape[0]} 行, {prices['ts_code'].nunique()} 股票")
print(f"  日期范围: {prices['trade_date'].min().strftime('%Y-%m-%d')} ~ {prices['trade_date'].max().strftime('%Y-%m-%d')}")

# 透视: 日期 x 股票
close_p = prices.pivot(index='trade_date', columns='ts_code', values='close')
open_p  = prices.pivot(index='trade_date', columns='ts_code', values='open')
high_p  = prices.pivot(index='trade_date', columns='ts_code', values='high')
low_p   = prices.pivot(index='trade_date', columns='ts_code', values='low')
volume_p = prices.pivot(index='trade_date', columns='ts_code', values='vol')
amount_p = prices.pivot(index='trade_date', columns='ts_code', values='amount')

# 过滤: 仅保留至少有 200 个交易日数据的股票
min_days = 200
valid_stocks = close_p.count()[close_p.count() >= min_days].index
print(f"  有效股票 (>= {min_days} 天): {len(valid_stocks)}")

for df_name, df in [('close', close_p), ('open', open_p), ('high', high_p),
                     ('low', low_p), ('volume', volume_p), ('amount', amount_p)]:
    locals()[f'{df_name}_p'] = df[valid_stocks].copy()

# ============================================================
# 2. 吴先兴五维成长因子 — 公式定义
# ============================================================
WUXIANXING_FACTORS = [
    {
        'id': 'novel_044', 'name': 'earnings_quality_proxy',
        'label': '盈利质量代理',
        'category': 'growth_quality',
        'hypothesis': '盈利质量高的股票(收益稳定性好)长期Alpha更高',
        'logic': '吴先兴五维之一: 盈利质量。量价代理=收益波动率低+正收益占比高',
        'formula': '-(close_p.pct_change().rolling(20).std() / (close_p.pct_change().rolling(20).mean().abs() + 0.001)) * '
                   '(close_p.pct_change() > 0).astype(float).rolling(20).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_045', 'name': 'cashflow_matching_proxy',
        'label': '现金流匹配度代理',
        'category': 'growth_quality',
        'hypothesis': '量价趋势与成交额趋势一致的股票(量价匹配)趋势更可靠',
        'logic': '吴先兴五维之二: 现金流匹配度。量价代理=价格趋势*成交额趋势同向性',
        'formula': '(close_p.pct_change(20) * (volume_p * close_p).pct_change(20) > 0).astype(float).rolling(20).mean()',
        'direction': 'long',
    },
    {
        'id': 'novel_046', 'name': 'capital_efficiency_proxy',
        'label': '资本投入效率代理',
        'category': 'growth_quality',
        'hypothesis': '单位成交额产生的价格变动效率高的股票资金利用效率高',
        'logic': '吴先兴五维之三: 资本投入效率。量价代理=收益/成交额比(Amihud的反面)',
        'formula': 'close_p.pct_change(20).abs() / (volume_p * close_p).rolling(20).mean().replace(0, np.nan) * 1e8',
        'direction': 'long',
    },
    {
        'id': 'novel_047', 'name': 'operational_efficiency_proxy',
        'label': '营运效率代理',
        'category': 'growth_quality',
        'hypothesis': '价格调整效率高的股票(快速消化信息)长期收益更高',
        'logic': '吴先兴五维之四: 营运效率。量价代理=价格自相关低(信息消化快)',
        'formula': '-close_p.pct_change().rolling(20).apply('
                   'lambda x: x.autocorr(lag=1) if len(x.dropna()) > 5 else np.nan, raw=True)',
        'direction': 'long',
    },
    {
        'id': 'novel_048', 'name': 'bargaining_power_proxy',
        'label': '议价能力代理',
        'category': 'growth_quality',
        'hypothesis': '在行业下跌时跌幅更小的股票(定价权强)质量更高',
        'logic': '吴先兴五维之五: 议价能力。量价代理=下行捕获率(跌时跌得少)',
        'formula': '-close_p.pct_change().clip(upper=0).rolling(20).mean()',
        'direction': 'long',
    },
]

# ============================================================
# 3. 计算因子值
# ============================================================
print("\n" + "-" * 40)
print("  计算因子值...")

factor_dfs = {}
for factor in WUXIANXING_FACTORS:
    name = factor['name']
    formula_str = factor['formula']
    try:
        # 在安全的命名空间中执行公式
        local_vars = {
            'close_p': close_p, 'open_p': open_p, 'high_p': high_p,
            'low_p': low_p, 'volume_p': volume_p, 'amount_p': amount_p,
            'np': np, 'pd': pd, 'float': float, 'int': int,
            'len': len, 'abs': abs, 'max': max, 'min': min,
            'sum': sum, 'round': round, 'sorted': sorted,
        }
        result = eval(formula_str, {'__builtins__': {}}, local_vars)
        if isinstance(result, pd.DataFrame):
            # 缩尾: 1%/99%
            q1, q99 = result.quantile(0.01), result.quantile(0.99)
            result = result.clip(q1, q99, axis=1)
            factor_dfs[name] = result
            d0 = result.index[0]
            d1 = result.index[-1]
            d0_str = d0.strftime('%Y-%m-%d') if hasattr(d0, 'strftime') else str(d0)[:10]
            d1_str = d1.strftime('%Y-%m-%d') if hasattr(d1, 'strftime') else str(d1)[:10]
            print(f"  {name:30s} OK  shape={result.shape}  dates={d0_str} ~ {d1_str}")
        else:
            print(f"  {name:30s} FAIL (result type: {type(result).__name__})")
    except Exception as e:
        print(f"  {name:30s} ERROR: {e}")

# ============================================================
# 4. IC 分析 (日频)
# ============================================================
print("\n" + "-" * 40)
print("  日频 IC 分析...")

# 前向收益 (次日)
forward_ret_1d = close_p.shift(-1) / close_p - 1

daily_ic_results = []
for name, factor_df in factor_dfs.items():
    ic_series = []
    common_idx = factor_df.index.intersection(forward_ret_1d.index)
    for date in common_idx:
        try:
            fv = factor_df.loc[date].dropna()
            fr = forward_ret_1d.loc[date].dropna()
            common = fv.index.intersection(fr.index)
            if len(common) < 50:
                continue
            ic = fv[common].rank().corr(fr[common].rank())
            if not np.isnan(ic):
                ic_series.append(ic)
        except Exception:
            continue

    if ic_series:
        ic_arr = np.array(ic_series)
        ic_mean = np.mean(ic_arr)
        ic_std = np.std(ic_arr)
        ir = ic_mean / ic_std if ic_std > 0 else 0.0
        # FA convention: raw IR (not annualized)
        icir_daily = ir
        plus_ic_pct = np.mean(ic_arr > 0) if len(ic_arr) > 0 else 0.5
        daily_ic_results.append({
            'name': name,
            'icir_daily': icir_daily,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            '+ic%': plus_ic_pct,
            'n_obs': len(ic_arr),
        })
        print(f"  {name:30s} ICIR(日频)={icir_daily:+.3f}  IC_mean={ic_mean:+.4f}  +IC%={plus_ic_pct:.1%}  n={len(ic_arr)}")
    else:
        daily_ic_results.append({
            'name': name, 'icir_daily': np.nan, 'ic_mean': np.nan,
            'ic_std': np.nan, '+ic%': np.nan, 'n_obs': 0,
        })
        print(f"  {name:30s} NO DATA")

# ============================================================
# 5. IC 分析 (周频)
# ============================================================
print("\n" + "-" * 40)
print("  周频 IC 分析...")

# 周频聚合: 取每周最后一天的因子值
close_weekly = close_p.resample('W').last()
factor_weekly = {}
for name, df in factor_dfs.items():
    factor_weekly[name] = df.resample('W').last()

forward_ret_1w = close_weekly.shift(-1) / close_weekly - 1

weekly_ic_results = []
for name, factor_df_w in factor_weekly.items():
    ic_series = []
    common_idx = factor_df_w.index.intersection(forward_ret_1w.index)
    for date in common_idx:
        try:
            fv = factor_df_w.loc[date].dropna()
            fr = forward_ret_1w.loc[date].dropna()
            common = fv.index.intersection(fr.index)
            if len(common) < 50:
                continue
            ic = fv[common].rank().corr(fr[common].rank())
            if not np.isnan(ic):
                ic_series.append(ic)
        except Exception:
            continue

    if ic_series:
        ic_arr = np.array(ic_series)
        ic_mean = np.mean(ic_arr)
        ic_std = np.std(ic_arr)
        ir = ic_mean / ic_std if ic_std > 0 else 0.0
        # FA convention: raw IR (not annualized)
        icir_weekly = ir
        plus_ic_pct = np.mean(ic_arr > 0)
        weekly_ic_results.append({
            'name': name,
            'icir_weekly': icir_weekly,
            'ic_mean': ic_mean,
            '+ic%': plus_ic_pct,
            'n_obs': len(ic_arr),
        })
        print(f"  {name:30s} ICIR(周频)={icir_weekly:+.3f}  IC_mean={ic_mean:+.4f}  +IC%={plus_ic_pct:.1%}  n={len(ic_arr)}")
    else:
        weekly_ic_results.append({
            'name': name, 'icir_weekly': np.nan, 'ic_mean': np.nan,
            '+ic%': np.nan, 'n_obs': 0,
        })
        print(f"  {name:30s} NO DATA")

# ============================================================
# 6. 汇总报告
# ============================================================
print("\n" + "=" * 60)
print("  吴先兴五维成长因子 — 测试结果汇总")
print("=" * 60)

print(f"\n  {'因子':30s} {'日频ICIR':>10s} {'+IC%':>8s} {'周频ICIR':>10s} {'+IC%':>8s} {'判定':>12s}")
print(f"  {'-'*82}")

all_results = []
for f_def in WUXIANXING_FACTORS:
    name = f_def['name']
    label = f_def['label']
    d_ic = next((r for r in daily_ic_results if r['name'] == name), {})
    w_ic = next((r for r in weekly_ic_results if r['name'] == name), {})

    d_icir = d_ic.get('icir_daily', np.nan)
    d_pct  = d_ic.get('+ic%', np.nan)
    w_icir = w_ic.get('icir_weekly', np.nan)
    w_pct  = w_ic.get('+ic%', np.nan)

    # 判定: 双重门坎 |ICIR|>=0.3 AND +IC%>=55%
    d_pass = abs(d_icir) >= 0.3 and d_pct >= 0.55
    w_pass = abs(w_icir) >= 0.3 and w_pct >= 0.55

    if d_pass:
        verdict = '✅ 日频PASS'
    elif not np.isnan(d_icir):
        verdict = '📦 储备'
    else:
        verdict = '❌ 无数据'

    print(f"  {name:30s} {d_icir:+10.3f} {d_pct:7.1%} {w_icir:+10.3f} {w_pct:7.1%} {verdict:>12s}")

    all_results.append({
        'name': name, 'label': label, 'category': f_def['category'],
        'hypothesis': f_def['hypothesis'], 'logic': f_def['logic'],
        'icir_daily': d_icir, '+ic%_daily': d_pct,
        'icir_weekly': w_icir, '+ic%_weekly': w_pct,
        'daily_pass': d_pass, 'weekly_pass': w_pass,
    })

# 保存 CSV
df_results = pd.DataFrame(all_results)
csv_path = REPORT_DIR / 'wuxianxing_5factor_test.csv'
df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n  结果已保存: {csv_path}")

# ============================================================
# 7. 对比: 与FA现有Top因子的ICIR
# ============================================================
fa_top_icir = {
    'dollar_vol_stability': 0.670,
    'earnings_growth_yoy': 0.514,
    'max_ret_1m': 0.503,
    'turnover_std': 0.503,
}

print(f"\n  === FA Top4 因子周频ICIR参考 ===")
for fname, icir in fa_top_icir.items():
    print(f"  {fname:30s} ICIR(周频)={icir:+.3f}")

print(f"\n  === 完成 ===")
