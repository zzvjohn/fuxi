"""
因子炼金术 v7.4 — 完整流水线执行脚本 (两阶段: 预筛选 + 浓缩GA)
====================================
v7.4 (2026-07-08):
    因子预筛选: ICIR粗筛(0.1) → 相关聚类去冗余 → WFA存活预检
    浓缩GA: 104→~20因子, 动态预算(60-120pop × 10-20gen), 覆盖率 10-100%
v7.2 (2026-06-27):
    三目标 NSGA-II: Obj1(净质量,真加法链) + Obj2(纯稳定) + Obj3(成本后Sharpe)
    ICIR: clip ∈ [0,1] 与 Obj2 同量纲
    惩罚链: icir_score - Σpenalties + bonus (真加法, 非乘法链)
    MAX_FACTORS=4 (硬约束+惩罚起点统一)
    Obj3 增加成本拆解诊断
因子池: 104 因子 (全量, v7.4 第一步预筛选到 ~20)
NSGA-II: 动态预算 (60-120 pop * 10-20 gen), 覆盖率优化 (3D优化)
结果保存到 output/
"""
import sys, time, os, pickle, traceback
from pathlib import Path
from datetime import datetime

# 修复Windows GBK编码问题 (emoji等Unicode字符导致'gbk' codec错误)
# 注意: 用reconfigure而非新建TextIOWrapper, 避免覆盖-u unbuffered模式
# v3: 同时设置 buffering=1 确保每行立即 flush (防止后台跑时输出丢失)
_encoding_fixed = False
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        _encoding_fixed = True
    except Exception:
        pass

# ★ v7.3: 强制 line-buffering — 防止后台运行进程"消失"后输出全丢
try:
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8',
                       errors='replace', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8',
                       errors='replace', buffering=1)
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from math import comb
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# 命令行参数解析
# =====================================================
INIT_METHOD = 'icir_biased'  # default (v6.1: ICIR偏置优于随机, OOS ICIR 0.504 vs 0.287)
LOG_SUFFIX = ''
NO_WRITE = False  # 沙箱兼容模式: 不写任何文件, 仅stdout
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == '--init':
        if i < len(sys.argv) - 1:
            val = sys.argv[i + 1]
            if val in ('random', 'icir_biased'):
                INIT_METHOD = val
                LOG_SUFFIX = f'_{val}'
            else:
                print(f"[WARN] 未知 --init 值 '{val}', 使用默认 'random'")
    elif arg == '--init-suffix':
        if i < len(sys.argv) - 1:
            LOG_SUFFIX = f"_{sys.argv[i + 1]}"
    elif arg == '--no-write':
        NO_WRITE = True

# 日志重定向
if NO_WRITE:
    # 沙箱兼容模式: 不写文件, 直接 stdout
    log_dir = Path('.')
    log_file = None
else:
    base_output = Path(__file__).parent / 'output'
    base_output.mkdir(exist_ok=True)
    if LOG_SUFFIX:
        log_dir = base_output / LOG_SUFFIX.lstrip('_')
        log_dir.mkdir(exist_ok=True)
    else:
        log_dir = base_output
    log_file = log_dir / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

class TeeOutput:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()

if NO_WRITE:
    pass  # 沙箱模式: 不重定向, 直接stdout
else:
    log_f = open(log_file, 'w', encoding='utf-8')
    sys.stdout = TeeOutput(log_f)

t_total = time.time()

# =====================================================
from config import FACTOR_DEFS, OOS_SPLIT_DATE
from factors import ALL_FACTORS
from ga.domain_split import compute_ds_panel, combine_domain_scores
N_DOMAINS = 3  # 分域数: 低DS / 中DS / 高DS

print("=" * 70)
print(f"  因子炼金术 (Factor Alchemy) v7.5 — 分域版 TEST (06-27 真加法链+clip+Top30 + DS分域)")
print(f"  Obj1=净质量(真加法:ICIR-Σpen+bonus) | Obj2=纯稳定 | Obj3=成本后Sharpe(Top30对齐)")
print(f"  MAX_FACTORS: 4 | 分域: {N_DOMAINS}域(DS偏离显著性) | ICIR: clip | 初始化: {INIT_METHOD}")
print(f"  OOS: {OOS_SPLIT_DATE}")
print(f"  初始化方法: {INIT_METHOD}")
print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if log_file:
    print(f"  日志文件: {log_file}")
print("=" * 70)

# =====================================================
# Phase 1: 数据加载
# =====================================================
print("\n" + "=" * 70)
print("[Phase 1] 加载数据 (2021-01-01 ~ 2026-06-25)")
print("=" * 70)

from data.loader import load_all_data
t1 = time.time()
data = load_all_data('2021-01-01', '2026-06-25')
print(f"  耗时: {time.time()-t1:.0f}s")

price_data = data['price_data']
financial_data = data['financial_data']
valuation_data = data['valuation_data']
industry_map = data.get('industry_map', {})

close = price_data.get('close')
print(f"  价格矩阵: {close.shape}")

# 构建周频前向收益
close_weekly = close.resample('W').last()
close_weekly.index = pd.to_datetime(close_weekly.index)
forward_returns = close_weekly.shift(-1) / close_weekly - 1
print(f"  周频收益: {forward_returns.shape} ({forward_returns.index[0].strftime('%Y-%m-%d')} ~ {forward_returns.index[-1].strftime('%Y-%m-%d')})")

# 市值数据
mcap_df = None
if valuation_data is not None and hasattr(valuation_data, 'columns'):
    if 'market_cap' in valuation_data.columns:
        t2 = time.time()
        mcap_raw = valuation_data[['code', 'trade_date', 'market_cap']].copy()
        mcap_raw['trade_date'] = pd.to_datetime(mcap_raw['trade_date'])
        mcap_df = mcap_raw.pivot(index='trade_date', columns='code', values='market_cap')
        mcap_df = mcap_df.resample('W').last()
        # 取log
        mcap_df = np.log(mcap_df.clip(lower=1))
        print(f"  市值(log): {mcap_df.shape} ({time.time()-t2:.0f}s)")

# =====================================================
# Phase 1.5: 因子池同步检查 — 确保每次运行基于最新最全因子库
# =====================================================
print("\n" + "=" * 70)
print("[Phase 1.5] 因子池同步检查")
print("=" * 70)

# 读取因子池CSV, 与 FACTOR_DEFS 交叉比对
factor_pool_path = Path(__file__).parent.parent.parent / 'data' / 'passed_factor_pool.csv'
if factor_pool_path.exists():
    pool_df = pd.read_csv(factor_pool_path)
    pool_candidates = set(pool_df[pool_df['status'] == 'candidate']['name'].values) if 'status' in pool_df.columns else set()
    pool_all = set(pool_df['name'].values) if 'name' in pool_df.columns else set()
    
    config_factors = set(FACTOR_DEFS.keys())
    
    # candidate中不在FACTOR_DEFS的
    missing_candidates = pool_candidates - config_factors
    # 全部因子池中不在FACTOR_DEFS的
    missing_all = pool_all - config_factors
    
    print(f"  因子池总因子: {len(pool_all)} | candidate: {len(pool_candidates)}")
    print(f"  FACTOR_DEFS注册: {len(config_factors)}")
    print(f"  池中但未注册: {len(missing_all)}")
    
    if missing_candidates:
        print(f"  [WARN] 未注册的candidate因子 ({len(missing_candidates)}):")
        for name in sorted(missing_candidates):
            row = pool_df[pool_df['name'] == name]
            icir_val = row['daily_icir'].values[0] if 'daily_icir' in row.columns else '?'
            print(f"    {name:35s} 日频ICIR={icir_val}")
        print(f"  [TIP] 这些因子在池中标记为candidate, 但未加入FACTOR_DEFS")
        print(f"     运行 scripts/factor_pool.py --sync 同步, 或手动更新 config.py")
    else:
        print(f"  [OK] 所有candidate因子已注册, 因子库最新")
else:
    print(f"  [WARN] 因子池文件不存在: {factor_pool_path}")
    print(f"     请运行 daily_factor_hypothesis.py 生成因子池")

# =====================================================
# Phase 2: 因子计算
# =====================================================
print("\n" + "=" * 70)
print("[Phase 2] 因子计算")
print("=" * 70)

factor_dfs = {}
is_size_map = {}
t2 = time.time()

for name in FACTOR_DEFS:
    if name not in ALL_FACTORS:
        continue
    
    factor_def = FACTOR_DEFS[name]
    factor_obj = ALL_FACTORS[name]()
    
    try:
        df = factor_obj.compute(
            price_data=price_data,
            financial_data=financial_data,
            valuation_data=valuation_data,
            industry_map=industry_map,
        )
        if df is not None and not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            # ★ 先 resample 到周频
            df_weekly = df.resample('W').last()
            
            # ★ 关键修复: 财务类因子 (仅1-5行) 需要 ffill 扩展到所有周
            # 方法: reindex 到 close_weekly 的全部日期, 然后前向填充
            is_financial = df_weekly.shape[0] <= 10 and factor_def['category'] in \
                ('profitability', 'value', 'growth')
            
            if is_financial:
                # 财务因子: 全周扩展
                df_weekly = df_weekly.reindex(close_weekly.index).ffill()
                # 季度开始时前向也有NaN → bfill (用后期数据)
                df_weekly = df_weekly.bfill()
            else:
                # 量价因子: 只保留交集
                common_idx = close_weekly.index.intersection(df_weekly.index)
                if len(common_idx) > 0:
                    df_weekly = df_weekly.reindex(common_idx)
            factor_dfs[name] = df_weekly
            is_size_map[name] = (factor_def['category'] == 'size')
            print(f"  {name:25s} ({factor_def['label']:15s}) OK  {df_weekly.shape[0]}周 × {df_weekly.shape[1]}股")
        else:
            print(f"  {name:25s} ({factor_def['label']:15s}) SKIP (无数据)")
    except Exception as e:
        print(f"  {name:25s} ({factor_def['label']:15s}) FAIL ({str(e)[:60]})")

print(f"\n  成功: {len(factor_dfs)}/{len(FACTOR_DEFS)} 个因子 ({time.time()-t2:.0f}s)")
print(f"  有效因子列表: {list(factor_dfs.keys())}")

if len(factor_dfs) < 3:
    print("[FATAL] 有效因子不足3个, 无法继续")
    sys.exit(1)

# =====================================================
# Phase 3: 单因子评估
# =====================================================
print("\n" + "=" * 70)
print("[Phase 3] 单因子评估 (ICIR + 单调性 + 双重排序 + 相关性 + 稳定性)")
print("=" * 70)

from evaluation.ic_analysis import compute_ic_summary
from evaluation.decile_test import decile_portfolio_test, test_monotonicity
from evaluation.double_sort import independent_double_sort, size_single_sort
from evaluation.correlation import factor_correlation_matrix
from evaluation.scoring import evaluate_factor_comprehensive
from evaluation.stability import (                    # ★ 新增
    compute_stability_comprehensive,
    print_stability_report,
)

t3 = time.time()

# 3.1 IC/ICIR
print("\n--- IC/ICIR 分析 ---")
ic_summary = compute_ic_summary(factor_dfs, forward_returns)

print(f"\n{'因子':25s} {'ICIR':>8s} {'IC_mean':>8s} {'+IC%':>7s} {'方向':>4s}")
print("-" * 56)
for idx, row in ic_summary.iterrows():
    direc = '多' if row['IC_mean'] > 0 else '空'
    print(f"{idx:25s} {row['ICIR']:+7.3f} {row['IC_mean']:+7.4f} {row['+IC%']:6.1%} {direc:>3s}")

# 3.2 相关性矩阵
print("\n--- 因子相关性矩阵 ---", flush=True)
try:
    corr_matrix = factor_correlation_matrix(factor_dfs)
    print(f"  形状: {corr_matrix.shape}", flush=True)
except Exception as e:
    print(f"  [WARN] 相关性计算失败: {e}", flush=True)
    import traceback
    traceback.print_exc()
    corr_matrix = pd.DataFrame()

# 标记高相关对 (>0.5)
high_corr = []
for i in range(len(corr_matrix.index)):
    for j in range(i+1, len(corr_matrix.index)):
        val = abs(corr_matrix.iloc[i, j])
        if val > 0.5:
            high_corr.append((corr_matrix.index[i], corr_matrix.columns[j], val))
if high_corr:
    print(f"  高相关对 (|r|>0.5): {len(high_corr)}")
    for a, b, v in sorted(high_corr, key=lambda x: -x[2])[:10]:
        print(f"    {a:20s} <-> {b:20s}  r={v:+.3f}")

# 3.3 十分位单调性 + 双重排序 (全部因子)
print("\n--- 十分位单调性 & 双重排序 (前12因子) ---")
comprehensive = {}
for name in list(factor_dfs.keys())[:12]:
    try:
        from evaluation.ic_analysis import compute_ic_icir
        icr = compute_ic_icir(factor_dfs[name], forward_returns)
        icir_val = icr.get('icir', 0)
        
        result = evaluate_factor_comprehensive(
            name, factor_dfs[name], forward_returns, mcap_df,
            icir_result={'icir': icir_val},
            correlation_val=0.5,  # placeholder
            is_size_factor=is_size_map.get(name, False)
        )
        comprehensive[name] = result
        
        ds_info = result['double_sort']
        if is_size_map.get(name, False):
            print(f"  {name:25s} ICIR={icir_val:+.3f} | 规模10分位: spread={ds_info.get('size_premium',0):+.4%} t={ds_info.get('t_stat',0):+.2f} p={ds_info.get('p_value',0):.3f}")
        else:
            print(f"  {name:25s} ICIR={icir_val:+.3f} | 双重排序25组: spread={ds_info.get('factor_quintile_spread',0):+.4%} t={ds_info.get('t_stat',0):+.2f} p={ds_info.get('p_value',0):.3f}")
    except Exception as e:
        print(f"  {name:25s} FAIL ({str(e)[:60]})")

# ★ 3.4 时序稳定性评估
print("\n--- 时序稳定性评估 (Top 10 非规模因子) ---")
stability_results = {}
non_size_factors = [n for n in list(factor_dfs.keys())[:10]
                    if not is_size_map.get(n, False)]
for name in non_size_factors:
    try:
        # 获取IC序列
        from evaluation.ic_analysis import compute_ic_icir
        icr = compute_ic_icir(factor_dfs[name], forward_returns)
        ic_series = icr.get('ic_series', pd.Series(dtype=float))
        
        stab = compute_stability_comprehensive(
            factor_dfs[name],
            forward_returns=forward_returns,
            ic_series=ic_series,
            top_n=30,
        )
        stability_results[name] = stab
    except Exception as e:
        print(f"  {name:25s} 稳定性: FAIL ({str(e)[:50]})")

# 打印稳定性排名
if stability_results:
    scores = [(n, r['total_stability_score']) for n, r in stability_results.items()
              if not np.isnan(r['total_stability_score'])]
    scores.sort(key=lambda x: -x[1])
    print(f"\n  {'因子':25s} {'稳定性分数':>10s} {'通过':>6s}")
    print(f"  {'-'*43}")
    for name, score in scores[:8]:
        passed = stability_results[name]['passed']
        print(f"  {name:25s} {score:10.3f} {'PASS' if passed else 'FAIL':>6s}")
    # 打印最稳定因子的详细报告
    if scores:
        print(f"\n  === 最稳定因子: {scores[0][0]} (分数={scores[0][1]:.3f}) ===")
        print_stability_report(stability_results[scores[0][0]])

print(f"\n  单因子评估完成 ({time.time()-t3:.0f}s)")

# ★ 预计算所有因子的 IC 序列 (Phase 3.4 稳定性评估使用, v6.1 NSGA-II不再需要)
print("\n  预计算因子 IC 序列...")
from evaluation.ic_analysis import compute_ic_icir
factor_ic_series = {}
for name in factor_dfs:
    if name not in ic_summary.index:
        continue
    try:
        icr = compute_ic_icir(factor_dfs[name], forward_returns)
        ic_s = icr.get('ic_series', pd.Series(dtype=float))
        if len(ic_s) > 50:
            factor_ic_series[name] = ic_s
    except Exception:
        pass
print(f"  预计算完成: {len(factor_ic_series)}/{len(factor_dfs)} 个因子")

# ★ 预标准化所有因子 (供 NSGA-II Pareto 解码加速使用)
print("\n  预标准化因子 (Pareto解码加速)...")
from factors.composite import standardize_factors
factor_dfs_std = standardize_factors(factor_dfs)
print(f"  标准化完成: {len(factor_dfs_std)} 个因子")

# =====================================================
# Phase 3.5: 跳过 — 06-20 前版本没有 Phase 3.5
#           规则负担分析在 Phase 6 独立执行 (post-hoc BIC), 不反馈到 GA
# =====================================================
print("\n" + "=" * 70)
print("[Phase 3.5] 跳过 — v6.1架构: 规则负担在 Phase 6 独立分析, 不参与流水线")
print("=" * 70)

# 直接用 config.MAX_FACTORS (硬约束8), 让 GA 自由探索 1~8 因子
# 最优因子数由 Phase 6 BIC 分析后人为判断
import sys as _sys
_fa_config_mod = _sys.modules.get('config')
if _fa_config_mod is None:
    import config as _fa_config_mod
effective_max_factors = _fa_config_mod.MAX_FACTORS
recommended_n = 4  # 默认推荐 (基于历史 Phase 6 结论)
burden_results = None  # Phase 7.5 需要此变量

print(f"\n  {'='*50}")
print(f"  [v7.4] 三目标NSGA-II | clip ICIR | 加法链:ICIR - Σpen + bonus")
print(f"    MAX_FACTORS:           {effective_max_factors} (硬约束+惩罚起点统一)")
print(f"    COMPLEXITY_PENALTY:    linear (50% cap, 超{effective_max_factors}个每+1扣{0.50/effective_max_factors:.3f})")
print(f"    Obj1:                  净质量 = clip(|ICIR|/1.5,0,1) - Σpen + bonus (真加法, ∈[0,1])")
print(f"    Obj2:                  纯稳定 = IC秩自相关/0.5 (∈[0,1])")
print(f"    Obj3:                  成本后Sharpe = Top30等权扣QMT成本后年化 (v7.2 诊断增强)")
print(f"  {'='*50}")

# =====================================================
# Phase 3.6: ICIR raw vs clip 区分度对比 (v7.2新增)
# =====================================================
print("\n" + "=" * 70)
print("[Phase 3.6] ICIR raw vs clip 区分度对比")
print("           核心问题: raw模式叠加惩罚链后是否有足够区分度?")
print("=" * 70)

from ga.fitness import icir_to_objective

# ★ 提前定义 valid_factor_names (原本在 Phase 4, 此处 Phase 3.6 也需要)
_valid_names_36 = [n for n in factor_dfs if n in ic_summary.index
                   and not np.isnan(ic_summary.loc[n, 'ICIR'])]

# 收集所有因子ICIR
icir_values = []
for name in _valid_names_36:
    if name in ic_summary.index:
        icir_values.append(abs(ic_summary.loc[name, 'ICIR']))
icir_arr = np.array(icir_values)

if len(icir_arr) > 5:
    # raw模式
    raw_scores = np.array([icir_to_objective(v, mode='raw') for v in icir_arr])
    # clip模式
    clip_scores = np.array([icir_to_objective(v, mode='clip', clip_threshold=1.5) for v in icir_arr])
    
    # 区分度指标
    def discrimination_metrics(scores, name):
        cv = np.std(scores) / (np.mean(scores) + 1e-10)
        spread = np.max(scores) - np.min(scores)
        p10, p90 = np.percentile(scores, [10, 90])
        iqr = p90 - p10
        return {
            'mean': np.mean(scores), 'std': np.std(scores),
            'cv': cv, 'spread': spread, 'iqr': iqr,
            'min': np.min(scores), 'max': np.max(scores),
            'p10': p10, 'p90': p90, 'top3_mean': np.mean(np.sort(scores)[-3:]),
        }
    
    raw_metrics = discrimination_metrics(raw_scores, 'raw')
    clip_metrics = discrimination_metrics(clip_scores, 'clip')
    
    print(f"\n  === ICIR 原始分布 ===")
    print(f"  ICIR范围: [{icir_arr.min():.3f}, {icir_arr.max():.3f}]")
    print(f"  ICIR均值: {icir_arr.mean():.3f} ± {icir_arr.std():.3f}")
    print(f"  ICIR>1.5的因子数: {(icir_arr>1.5).sum()}/{len(icir_arr)} ({(icir_arr>1.5).mean():.0%})")
    
    print(f"\n  === 评分区分度对比 ===")
    print(f"  {'指标':>12s} {'raw':>10s} {'clip':>10s} {'比值(raw/clip)':>15s} {'解读':>20s}")
    print(f"  {'-'*72}")
    
    for key, label, interpretation in [
        ('mean', '均值', 'clip会压缩均值'),
        ('std', '标准差', '越小→区分度越低'),
        ('cv', '变异系数CV', 'CV=σ/μ, 越大区分度越好'),
        ('spread', '极差(max-min)', '总跨度'),
        ('iqr', 'IQR(P90-P10)', '核心区分度(排除极端值)'),
        ('top3_mean', 'Top3均值', '最强3因子平均分'),
    ]:
        rv = raw_metrics[key]
        cv_val = clip_metrics[key]
        ratio = rv / (cv_val + 1e-10) if cv_val > 0 else float('inf')
        print(f"  {label:>12s} {rv:10.3f} {cv_val:10.3f} {ratio:>14.1f}x "
              f"{interpretation:>20s}")
    
    # ★ 模拟惩罚链叠加后的区分度
    print(f"\n  === 惩罚链叠加后区分度 (模拟: ICIR_score - 0.25惩罚) ===")
    pen = 0.25  # 典型惩罚值
    raw_after_pen = np.clip(raw_scores - pen, 0.0, None)
    clip_after_pen = np.clip(clip_scores - pen, 0.0, 1.0)
    
    raw_ap = discrimination_metrics(raw_after_pen, 'raw+pen')
    clip_ap = discrimination_metrics(clip_after_pen, 'clip+pen')
    
    print(f"  {'指标':>12s} {'raw+pen':>10s} {'clip+pen':>10s} {'比值':>10s}")
    print(f"  {'-'*46}")
    for key, label in [('mean', '均值'), ('std', '标准差'), ('cv', 'CV'),
                        ('iqr', 'IQR'), ('spread', '极差')]:
        rv = raw_ap[key]
        cv_val = clip_ap[key]
        ratio = rv / (cv_val + 1e-10) if cv_val > 0 else float('inf')
        print(f"  {label:>12s} {rv:10.3f} {cv_val:10.3f} {ratio:>9.1f}x")
    
    # 结论
    raw_cv = raw_ap['cv']
    clip_cv = clip_ap['cv']
    zero_after_pen = (clip_after_pen <= 0.01).sum()
    
    print(f"\n  === 结论 ===")
    if raw_cv > clip_cv * 1.5:
        print(f"  raw模式区分度({raw_cv:.2f}) >> clip模式({clip_cv:.2f})")
        print(f"  → raw更能区分好因子和差因子, 但需要担心量纲不一致")
    elif clip_cv > raw_cv * 1.5:
        print(f"  clip模式区分度({clip_cv:.2f}) >> raw模式({raw_cv:.2f})")
        print(f"  → clip更适合, Obj1∈[0,1]与Obj2同量纲")
    else:
        print(f"  两模式区分度接近 (raw={raw_cv:.2f}, clip={clip_cv:.2f})")
    
    if zero_after_pen > 0:
        print(f"  [WARN] clip模式有{zero_after_pen}个因子惩罚后归零 (ICIR太小), 区分度可能不足")
    else:
        print(f"  [OK] 惩罚后所有因子仍>0, clip模式有足够区分度")
    
    # ★ 推荐
    print(f"\n  ★ 推荐: {'raw' if raw_cv > clip_cv * 1.3 else 'clip'} 模式")
    print(f"     理由: {'raw区分度显著更高, 惩罚链已提供足够区分' if raw_cv > clip_cv * 1.3 else 'clip与Obj2同量纲, Pareto更均衡, 且惩罚链提供区分度'}")
else:
    print("  有效因子不足5个, 跳过对比")

# =====================================================
# Phase 3.7: DS 分域预处理 (v7.5新)
# =====================================================
print("\n" + "=" * 70)
print(f"[Phase 3.7] DS 分域预处理 (N_DOMAINS={N_DOMAINS}, 控制因子=市值)")
print("=" * 70)

t37 = time.time()
ds_panel, domain_panel = compute_ds_panel(
    forward_returns, {'size': mcap_df}, n_domains=N_DOMAINS
)
domain_coverage = domain_panel.notna().mean().mean()
dom_share_last = domain_panel.iloc[-1].value_counts(normalize=True).to_dict()
print(f"  分域覆盖度: {domain_coverage:.1%}")
print(f"  各域占比(末周): {dom_share_last}")
print(f"  耗时: {time.time()-t37:.0f}s")

# 对齐日期索引 (确保分域面板覆盖所有收益日期)
fr_dates = forward_returns.index.intersection(domain_panel.index)
domain_panel = domain_panel.reindex(fr_dates)
print(f"  对齐后分域面板: {domain_panel.shape}")

# =====================================================
# Phase 4: NSGA-II 三目标遗传算法 (v7.5 分域版 — 全域筛选 + 域内GA)
# =====================================================
print("\n" + "=" * 70)
print(f"[Phase 4] NSGA-II 三目标优化 (v7.5: 全域筛选统一因子池 + {N_DOMAINS}域内独立GA)")
print(f"          MAX_FACTORS={effective_max_factors} (硬约束+惩罚起点) | "
      f"ICIR: clip | 初始化: {INIT_METHOD}")
print(f"          目标: Obj1(净质量) vs Obj2(纯稳定) vs Obj3(成本后Sharpe) [训练集×每域]")
print("=" * 70)

# 过滤有效因子
valid_factor_names = [n for n in factor_dfs if n in ic_summary.index
                      and not np.isnan(ic_summary.loc[n, 'ICIR'])]
print(f"  有效因子: {len(valid_factor_names)}/{len(factor_dfs)}")
skipped = [n for n in factor_dfs if n not in valid_factor_names]
if skipped:
    print(f"  跳过: {skipped}")

n_factors = len(valid_factor_names)

# 预计算相关性矩阵
from evaluation.correlation import factor_correlation_matrix
try:
    corr_matrix_full = factor_correlation_matrix(
        {n: factor_dfs[n] for n in valid_factor_names if n in factor_dfs}
    )
    print(f"  相关性矩阵: {corr_matrix_full.shape}")
except Exception as e:
    print(f"  [WARN] Phase 4 相关性矩阵计算失败: {e}")
    print(f"  [FALLBACK] 使用单位矩阵替代 (NSGA-II 无相关性惩罚)")
    corr_matrix_full = pd.DataFrame(
        np.eye(n_factors),
        index=valid_factor_names, columns=valid_factor_names
    )

# ================================================================
# v7.4 新增: 因子预筛选 (三刀: ICIR → 聚类 → WFA)
# ================================================================
print(f"\n  --- 因子预筛选 (v7.4) ---")
from ga.factor_screening import screen_factors, print_screening_summary

_screened, _scr_stats = screen_factors(
    valid_factor_names, ic_summary, corr_matrix_full,
    factor_dfs, forward_returns,
    icir_threshold=0.1,
    corr_threshold=0.5,
    wfa_windows=8,
    wfa_min_pass=3,
    wfa_icir_threshold=0.1,
    verbose=True,
)
print_screening_summary(_scr_stats, _screened)

# 使用筛选后的因子 (而非全部有效因子)
ga_factor_names = _screened
n_factors = len(ga_factor_names)
print(f"  GA 因子空间: {n_factors} 因子")

# 重建筛选后因子的相关性子矩阵 (用于相关性惩罚)
if len(ga_factor_names) >= 2:
    _common = [n for n in ga_factor_names if n in corr_matrix_full.index]
    corr_matrix_full = corr_matrix_full.loc[_common, _common]
    print(f"  筛选后相关性矩阵: {corr_matrix_full.shape}")

# ===== OOS 数据分割 (v5.3 Step 3) =====
print(f"\n  --- OOS 验证 ---")
print(f"  分割日期: < {OOS_SPLIT_DATE} (训练集) | >= {OOS_SPLIT_DATE} (OOS验证集)")

# 训练/验证集的 forward_returns
fr_index = pd.DatetimeIndex(forward_returns.index)
train_mask = fr_index < pd.Timestamp(OOS_SPLIT_DATE)
oos_mask = fr_index >= pd.Timestamp(OOS_SPLIT_DATE)

train_fr = forward_returns.loc[train_mask]
oos_fr   = forward_returns.loc[oos_mask]
print(f"  训练集: {train_fr.shape[0]} 周 | OOS集: {oos_fr.shape[0]} 周")
print(f"  训练跨度: {train_fr.index[0].strftime('%Y-%m-%d') if len(train_fr) > 0 else 'N/A'} ~ "
      f"{train_fr.index[-1].strftime('%Y-%m-%d') if len(train_fr) > 0 else 'N/A'}")

# 训练集 IC series (从已预计算的全量 IC 中按日期切分)
oos_dt = pd.Timestamp(OOS_SPLIT_DATE)
train_ic = {}
for name, ic_series in factor_ic_series.items():
    if isinstance(ic_series, pd.Series) and len(ic_series) > 0:
        ic_train = ic_series[ic_series.index < oos_dt].dropna()
        if len(ic_train) > 10:
            train_ic[name] = ic_train

print(f"  训练集有效IC序列: {len(train_ic)} / {len(factor_ic_series)}")

# ================================================================
# v7.5: Per-Domain NSGA-II (全域统一因子池 → 域内独立GA)
# ================================================================

# ICIR arr (全域, 用于偏置初始化)
if INIT_METHOD == 'icir_biased':
    from ga.chromosome import icir_biased_chromosome
    icir_list = []
    for name in ga_factor_names:
        if name in ic_summary.index:
            icir_list.append(abs(ic_summary.loc[name, 'ICIR']))
        else:
            icir_list.append(0.0)
    icir_arr = np.array(icir_list)
    print(f"\n  [ICIR偏置初始化] 偏置强度=1.0")
    print(f"  ICIR范围: [{icir_arr.min():.3f}, {icir_arr.max():.3f}]")
else:
    from ga.chromosome import random_chromosome
    icir_arr = None

from ga.nsga2 import nsga2_run
from ga.domain_split import split_universe_by_domain
from factors.composite import combine_factors_vectorized
from ga.fitness import _estimate_weekly_turnover

# 训练期分域面板 (对齐 train_fr)
domain_train = domain_panel.reindex(train_fr.index)
train_domain_mask = domain_train.notna()
print(f"\n  训练期分域有效覆盖率: {train_domain_mask.mean().mean():.1%}")

# 各域独立 NSGA-II
domain_balanced = {}  # {domain_id: {composite, weights, icir_train, ...}}

for d in range(N_DOMAINS):
    domain_tag = ['低DS', '中DS', '高DS'][d]
    print(f"\n  {'='*60}")
    print(f"  [域{d}] {domain_tag} NSGA-II")
    print(f"  {'='*60}")

    # 域掩码 —— 仅该域 stock-date 参与 IC 计算
    train_fr_d = split_universe_by_domain(train_fr, domain_train, d)
    n_stocks_d = (domain_train == d).sum(axis=1).median()
    print(f"  域内中位数股票数: {n_stocks_d:.0f}")
    if n_stocks_d < 30:
        print(f"  [SKIP] 有效股票<30, 跳过域{d}")
        continue

    # GA 预算 (域内股票少, 适当减小)
    _combos = comb(n_factors, 4) if n_factors >= 4 else 1
    if _combos <= 5000:
        pop_size_d = min(80, max(40, _combos // 5))
        n_gen_d = max(8, min(10, 4000 // pop_size_d))
    else:
        pop_size_d, n_gen_d = 50, 8
    print(f"  GA: {pop_size_d}pop × {n_gen_d}gen, C({n_factors},4)={_combos}")

    # 初始化 (ICIR偏置或随机)
    if INIT_METHOD == 'icir_biased':
        pop = [icir_biased_chromosome(n_factors, icir_arr, bias_strength=1.0)
               for _ in range(pop_size_d)]
    else:
        pop = [random_chromosome(n_factors) for _ in range(pop_size_d)]

    # ======= NSGA-II =======
    t4d = time.time()
    try:
        final_pop, final_obj, pareto_front_d = nsga2_run(
            pop, ga_factor_names, factor_dfs,
            train_fr_d, mcap_df,
            n_generations=n_gen_d, pop_size=pop_size_d,
            crossover_prob=0.7, mutation_prob=0.2, mutation_sigma=0.1,
            corr_matrix=corr_matrix_full,
            ic_summary=ic_summary,
            factor_ic_series=train_ic,
            max_factors=effective_max_factors,
            verbose=False,
            factor_dfs_std=factor_dfs_std,
            debug_obj3=False,
        )
        n_pareto_d = len(pareto_front_d['weights'])
        print(f"  耗时: {time.time()-t4d:.0f}s, 前沿解: {n_pareto_d}")
    except Exception as e:
        import traceback
        print(f"  [WARN] NSGA-II 崩溃: {e}, 跳过域{d}")
        continue

    # 解码前沿 (仅域内有效)
    pareto_d = []
    for i, (obj, w) in enumerate(zip(pareto_front_d['objectives'], pareto_front_d['weights'])):
        try:
            sel = {n: factor_dfs_std[n] for n in w if n in factor_dfs_std}
            comp = combine_factors_vectorized(sel, w)
            if comp.empty or comp.shape[0] < 5:
                continue
            icir_tr = abs(compute_ic_icir(comp, train_fr_d)['icir'])
            est_to = _estimate_weekly_turnover(comp, top_n=30, buffer_mult=1.8)
            pareto_d.append({
                'quality_obj': float(obj[0]),
                'stability_obj': float(obj[1]),
                'sharpe_obj': float(obj[2]),
                'icir_train': icir_tr,
                'weights': w,
                'estimated_to': est_to,
                'composite': comp,
                'domain_id': d,
            })
        except Exception:
            continue

    print(f"  有效解码: {len(pareto_d)}/{n_pareto_d} 个解")
    if not pareto_d:
        print(f"  [WARN] 域{d} 无有效解, 跳过")
        continue

    # 选 Balanced (调和公式, 同全域版)
    bal = max(pareto_d, key=lambda x:
        (max(x['quality_obj'], 0.001) ** 0.4) *
        (max(x['stability_obj'], 0.001) ** 0.3) *
        (max(x['sharpe_obj'], 0.001) ** 0.3))

    print(f"  Balanced: ICIR_tr={bal['icir_train']:.3f}, N={len(bal['weights'])}, "
          f"TO={bal['estimated_to']:.1%}")
    for fn, fw in sorted(bal['weights'].items(), key=lambda x: -x[1]):
        fn_label = fn[:30]
        print(f"    {fn_label:30s} w={fw:.3f}")

    domain_balanced[d] = bal

# ================================================================
# Phase 4.1: 域间组合 (v7.5新)
# ================================================================
print("\n" + "=" * 70)
print(f"[Phase 4.1] 域间组合 — {len(domain_balanced)}域 Balanced composite → 统一截面打分")
print("=" * 70)

if len(domain_balanced) >= 2:
    composites_by_domain = [v['composite'] for v in domain_balanced.values()]
    unified_composite = combine_domain_scores(composites_by_domain, method='equal')
    print(f"  统一 composite: {unified_composite.shape}")
    for d, bal in domain_balanced.items():
        domain_tag = ['低DS', '中DS', '高DS'][d]
        fn_list = [f"{n}({fw:.3f})" for n, fw in sorted(bal['weights'].items(), key=lambda x: -x[1])]
        print(f"  域{d}({domain_tag}): {' + '.join(fn_list[:4])}")
elif len(domain_balanced) == 1:
    unified_composite = list(domain_balanced.values())[0]['composite']
    print(f"  [WARN] 仅1域有效, 回退为全域版")
else:
    print(f"  [FATAL] 所有域均无有效解")
    unified_composite = None

# 构造 dummy pareto_solutions (仅含 unified, 兼容后续 Phase)
pareto_solutions = [{
    'idx': 0, 'quality_obj': 0.0, 'stability_obj': 0.0, 'sharpe_obj': 0.0,
    'icir_train': 0.0, 'weights': {}, 'estimated_to': 0.0,
    'composite': unified_composite,
}] if unified_composite is not None else []

# =====================================================
# =====================================================
# Phase 4.5: 域组合统一评估 (v7.5: 分域版, 单策略)
# =====================================================
print("\n" + "=" * 70)
print("[Phase 4.5] 域组合统一评估 + 成本归因 (v7.5分域版)")
print("=" * 70)

best_weights_all = {}

if unified_composite is not None and not unified_composite.empty:
    composite = unified_composite

    # 汇总各域因子权重 (用于展示)
    all_weights = {}
    for d, bal in domain_balanced.items():
        domain_tag = ['低DS', '中DS', '高DS'][d]
        for fn, fw in bal['weights'].items():
            all_weights[f"{fn}({domain_tag})"] = fw
    print(f"\n  分域因子组成 ({len(domain_balanced)}域):")
    for name, wt in sorted(all_weights.items(), key=lambda x: -x[1]):
        print(f"    {name:35s} w={wt:7.3f}")

    # 五维评估 (同全域版)
    from evaluation.scoring import (
        ICIR_THRESHOLD, ICIR_QUALITY_THRESHOLD, MONOTONICITY_P_THRESHOLD,
        CORRELATION_THRESHOLD, DOUBLE_SORT_P_THRESHOLD, STABILITY_THRESHOLD,
        GA_PASS_MIN_DIMENSIONS,
    )
    from evaluation.stability import print_stability_report
    from evaluation.decile_test import decile_portfolio_test, test_monotonicity
    from evaluation.double_sort import independent_double_sort

    t45 = time.time()
    ga_icir = compute_ic_icir(composite, forward_returns)
    ga_decile = decile_portfolio_test(composite, forward_returns)
    ga_mono = test_monotonicity(ga_decile)
    ga_ds = independent_double_sort(composite, forward_returns, mcap_df)
    # 相关性 (取训练期因子, 找 composite 相关性最高的几个因子)
    try:
        subset_factors = {n: factor_dfs[n] for n in list(factor_dfs.keys())[:30]
                         if n in factor_dfs}
        all_for_corr = {**subset_factors, 'composite': composite}
        ga_corr_matrix = factor_correlation_matrix(all_for_corr)
    except Exception:
        ga_corr_matrix = pd.DataFrame()

    icr = compute_ic_icir(composite, forward_returns)
    ga_stab = compute_stability_comprehensive(
        composite, forward_returns=forward_returns,
        ic_series=icr.get('ic_series', pd.Series(dtype=float)), top_n=30,
    )

    d1 = abs(ga_icir['icir']) >= ICIR_QUALITY_THRESHOLD
    d2 = ga_mono['p_value'] < MONOTONICITY_P_THRESHOLD
    if not ga_corr_matrix.empty and 'composite' in ga_corr_matrix.index:
        max_corr = abs(ga_corr_matrix.loc['composite']
                       .drop('composite', errors='ignore')).max()
        d3 = max_corr < CORRELATION_THRESHOLD
    else:
        max_corr = 1.0; d3 = False
    d4 = ga_ds['p_value'] < DOUBLE_SORT_P_THRESHOLD
    d5 = ga_stab['total_stability_score'] >= STABILITY_THRESHOLD
    passed_count = sum([d1, d2, d3, d4, d5])

    from ga.fitness import _estimate_weekly_turnover, _compute_turnover_penalty
    est_to = _estimate_weekly_turnover(composite, top_n=30, buffer_mult=1.8)
    to_pen = _compute_turnover_penalty(est_to)

    print(f"\n  五维: {'PASS' if passed_count>=GA_PASS_MIN_DIMENSIONS else 'FAIL'} "
          f"({passed_count}/5通过) ICIR={ga_icir['icir']:+.3f} "
          f"稳定性={ga_stab['total_stability_score']:.3f}")
    print(f"  换手: 估算TO={est_to:.1%} penalty={to_pen:.2f}")
    print(f"  耗时: {time.time()-t45:.0f}s")

    best_weights_all['balanced'] = {
        'weights': all_weights, 'composite': composite,
        'icir': ga_icir, 'stab': ga_stab,
        'estimated_to': est_to, 'turnover_penalty': to_pen,
        'passed_count': passed_count,
    }
else:
    print("\n  [WARN] unified_composite 为空, 跳过 Phase 4.5")

# =====================================================
# Phase 5: 虚拟投资组合 (含 QMT 双边成本)
# =====================================================
print("\n" + "=" * 70)
print("[Phase 5] 虚拟投资组合验证 (QMT成本: 万1佣金+千1印花税+0.3%滑点)")
print("=" * 70)

# QMT 标准成本
QMT_COMMISSION = 0.0001   # 万1 佣金
QMT_STAMP_TAX = 0.001     # 千1 印花税 (仅卖出)
QMT_SLIPPAGE  = 0.003     # 0.3% 滑点

portfolio_results_all = {}

if pareto_solutions and 'best_weights_all' in dir():
    for strat_name in ['balanced']:  # v7.5: 分域版仅 unified 策略
        if strat_name not in best_weights_all:
            continue

        info = best_weights_all[strat_name]
        composite = info['composite']
        w = info['weights']

        try:
            from portfolio.simulator import PortfolioSimulator

            simulator = PortfolioSimulator(
                factor_df=composite,
                price_df=close_weekly,
                top_n=30,
                weighting='equal',
                max_turnover=0.50,
                commission=QMT_COMMISSION,
                stamp_tax=QMT_STAMP_TAX,
                slippage=QMT_SLIPPAGE,
            )

            port_result = simulator.run()
            stats = port_result.get('stats', {})

            # 计算成本归因
            est_to = info.get('estimated_to', 0)
            round_trip_cost = 2 * QMT_COMMISSION + QMT_STAMP_TAX + 2 * QMT_SLIPPAGE
            annual_cost_estimated = est_to * 52 * round_trip_cost

            print(f"\n  [{strat_name}] 成本归因:")
            print(f"    估算周换手: {est_to:.1%}")
            print(f"    单边成本: comm={QMT_COMMISSION:.4f}({QMT_COMMISSION*100:.2f}%) "
                  f"stamp={QMT_STAMP_TAX:.4f}({QMT_STAMP_TAX*100:.1f}%) "
                  f"slip={QMT_SLIPPAGE:.4f}({QMT_SLIPPAGE*100:.1f}%)")
            print(f"    双边成本: {round_trip_cost:.4f} ({round_trip_cost*100:.2f}%)")
            print(f"    年化成本: {annual_cost_estimated:.1%}")
            print(f"    总收益:   {stats.get('total_return', 0):.2%}")
            print(f"    年化收益: {stats.get('cagr', 0):.2%}")
            print(f"    Sharpe:   {stats.get('sharpe', 0):.2f}")
            print(f"    MaxDD:    {stats.get('max_drawdown', 0):.2%}")
            print(f"    平均换手: {stats.get('avg_turnover', 0):.1%}")
            print(f"    净收益(扣成本): {stats.get('total_return', 0):.2%}")
            if stats.get('cagr', 0) > 0:
                net_cagr = stats.get('cagr', 0)
                print(f"    净年化:   {net_cagr:.2%}")

            portfolio_results_all[strat_name] = port_result

        except Exception as e:
            print(f"\n  [{strat_name}] FAIL 回测异常: {e}")
            traceback.print_exc()

    # 对比总结
    if len(portfolio_results_all) >= 2:
        print(f"\n  === 成本调整后对比 ===")
        print(f"  {'策略':>12s} {'总收益':>8s} {'年化':>8s} {'Sharpe':>7s} {'MaxDD':>7s} {'平均换手':>8s}")
        print(f"  {'-'*56}")
        for sn, pr in portfolio_results_all.items():
            s = pr.get('stats', {})
            print(f"  {sn:>12s} {s.get('total_return',0):>7.1%} {s.get('cagr',0):>7.1%} "
                  f"{s.get('sharpe',0):>6.2f} {s.get('max_drawdown',0):>6.1%} "
                  f"{s.get('avg_turnover',0):>7.1%}")

        # 保存
        if not NO_WRITE:
            with open(log_dir / 'portfolio_results_all.pkl', 'wb') as pf:
                pickle.dump(portfolio_results_all, pf)
else:
    print("\n  [WARN] 无可用策略, 跳过 Phase 5")

# =====================================================
# Phase 5.5: OOS 验证 — 训练最优解在 OOS 上的 ICIR
# =====================================================
print("\n" + "=" * 70)
print("[Phase 5.5] OOS 验证 (训练集最优解在 OOS 上的表现)")
print("=" * 70)

if pareto_solutions and len(oos_fr) > 0:
    print(f"\n  OOS 期间: {oos_fr.index[0].strftime('%Y-%m-%d')} ~ {oos_fr.index[-1].strftime('%Y-%m-%d')} "
          f"({oos_fr.shape[0]} 周)")
    print(f"\n  {'策略':>12s} {'ICIR_train':>12s} {'ICIR_OOS':>12s} {'ΔICIR':>8s} {'稳定性':>6s}")
    print(f"  {'-'*56}")
    
    oos_results = {}
    # v7.5: 分域版仅评估 unified composite
    if unified_composite is not None:
        oos_icir_result = compute_ic_icir(unified_composite, oos_fr)
        oos_icir = abs(oos_icir_result['icir'])
        train_icir = abs(compute_ic_icir(unified_composite, train_fr)['icir'])
        delta = oos_icir - train_icir
        robust = "PASS" if oos_icir > 0 else "FAIL"
        oos_results['分域_unified'] = {
            'icir_train': train_icir, 'icir_oos': oos_icir,
            'delta': delta, 'robust': robust,
        }
        print(f"  {'分域_unified':>12s} {train_icir:12.3f} {oos_icir:12.3f} "
              f"{delta:+8.3f} {robust:>6s}")
    
    # OOS 结论
    all_pass = all(r['robust'] == 'PASS' for r in oos_results.values())
    any_pass = any(r['robust'] == 'PASS' for r in oos_results.values())
    print(f"\n  === OOS ICIR 结论 ===")
    if all_pass:
        print(f"  PASS 全部通过 OOS 验证 — GA 无过拟合, 策略可推广")
    elif any_pass:
        best_oos = max(oos_results.items(), key=lambda x: x[1]['icir_oos'])
        print(f"  PARTIAL 部分通过 — {best_oos[0]} 策略 OOS ICIR={best_oos[1]['icir_oos']:.3f} 最优")
        for sn, r in oos_results.items():
            if r['robust'] == 'FAIL':
                print(f"  FAIL {sn}策略 OOS ICIR<=0 ({r['icir_oos']:.3f}), 样本内过拟合")
    else:
        print(f"  FAIL 全部未通过 OOS — 整个 GA 框架可能过拟合, 需要重新审视")
    
    # === Phase 5.5+: OOS 收益回测 (虚拟组合, QMT成本) ===
    print(f"\n  --- OOS 收益回测 (Top30 等权, QMT成本) ---")
    
    oos_close = close_weekly.loc[oos_fr.index] if len(oos_fr) > 0 else None
    
    print(f"  {'策略':>12s} {'年化收益':>9s} {'Sharpe':>7s} {'MaxDD':>8s} {'Calmar':>7s} {'换手':>6s}")
    print(f"  {'-'*52}")
    
    # v7.5: 分域版用 unified_composite
    if unified_composite is not None:
        try:
            oos_composite = unified_composite.reindex(oos_fr.index).dropna(how='all')
            if not oos_composite.empty and oos_close is not None:
                oos_composite = oos_composite.where(oos_close.notna(), np.nan)
                from portfolio.simulator import PortfolioSimulator
                oos_sim = PortfolioSimulator(
                    factor_df=oos_composite, price_df=oos_close,
                    top_n=30, weighting='equal', max_turnover=0.50,
                    commission=QMT_COMMISSION, stamp_tax=QMT_STAMP_TAX, slippage=QMT_SLIPPAGE,
                )
                oos_port = oos_sim.run()
                oos_stats = oos_port.get('stats', {})
                cagr = oos_stats.get('cagr', 0)
                sharpe = oos_stats.get('sharpe', 0)
                maxdd = oos_stats.get('max_drawdown', 0)
                calmar = cagr / abs(maxdd) if maxdd != 0 else 0
                avg_to = oos_stats.get('avg_turnover', 0)
                print(f"  {'分域_unified':>12s} {cagr:>8.1%} {sharpe:>6.2f} {maxdd:>7.1%} {calmar:>6.2f} {avg_to:>5.1%}")
        except Exception as e:
            print(f"  {'分域_unified':>12s} ERROR: {e}")
    
else:
    print("\n  [WARN] OOS 数据为空, 跳过")

# =====================================================
# Phase 5.6: Obj3 2026时效性验证 (v7.2新增)
# =====================================================
print("\n" + "=" * 70)
print("[Phase 5.6] Obj3 2026时效性验证 (仅2026年数据)")
print("          目标: 确认因子组合在当前市场环境下是否仍有效")
print("=" * 70)

obj3_2026_results = {}
if pareto_solutions and len(factor_dfs_std) > 0:
    # 构建2026专用 forward_returns
    fr_2026_mask = fr_index >= pd.Timestamp('2026-01-01')
    fr_2026 = forward_returns.loc[fr_2026_mask]
    
    if len(fr_2026) >= 8:
        print(f"  2026数据: {fr_2026.index[0].strftime('%Y-%m-%d')} ~ "
              f"{fr_2026.index[-1].strftime('%Y-%m-%d')} ({fr_2026.shape[0]}周)")
        
        # v7.5: 分域版用 unified_composite
        strategies_2026 = [('balanced', unified_composite, '分域_unified')] if unified_composite is not None else []

        # 判断是否不足1年，不足则只报原始总收益（不做年化）
        is_short_period = len(fr_2026) < 52
        if is_short_period:
            header_fmt = f"  {{'策略':>12s}} {{'2026_Gross({len(fr_2026)}w)':>16s}} {{'Cost':>10s}} {{'2026_Net({len(fr_2026)}w)':>16s}} {{'SR':>7s}} {{'方向':>4s}} {{'换手':>6s}} {{'标记':>10s}}"
            print(f"\n{header_fmt}")
            print(f"  {'-'*82}")
        else:
            print(f"\n  {'策略':>12s} {'2026_Gross':>10s} {'2026_Cost':>10s} "
                  f"{'2026_Net':>10s} {'2026_SR':>8s} {'Train_SR':>9s} {'Delta':>8s} "
                  f"{'方向':>4s} {'周换手':>7s}")
            print(f"  {'-'*82}")
        
        from ga.fitness import _compute_top30_cost_adjusted_sharpe
        from factors.composite import combine_factors_vectorized, weights_from_chromosome
        
        for strat_name, composite_obj, label in strategies_2026:
            try:
                # v7.5: 分域版 composite 直接是 DataFrame
                if isinstance(composite_obj, pd.DataFrame):
                    comp_2026 = composite_obj
                else:
                    w = composite_obj['weights']
                    sel = {n: factor_dfs_std[n] for n in w if n in factor_dfs_std}
                    comp_2026 = combine_factors_vectorized(sel, w)
                
                # 只取2026年数据
                comp_2026_only = comp_2026.reindex(fr_2026.index).dropna(how='all')
                if comp_2026_only.empty or comp_2026_only.shape[0] < 5:
                    print(f"  {label:>12s} SKIP (2026数据不足)")
                    continue
                
                # 2026 Obj3计算
                obj3_26 = _compute_top30_cost_adjusted_sharpe(
                    comp_2026_only, fr_2026, force_direction=1, debug=False)
                
                # v7.5: 分域版无单独的 train_sharpe_obj, 用 0 占位
                train_sharpe = composite_obj.get('sharpe_obj', 0.0) if isinstance(composite_obj, dict) else 0.0
                delta = obj3_26['annual_sharpe'] - train_sharpe * 2.0
                
                dir_label = '多' if obj3_26['direction'] >= 0 else '空'
                
                if is_short_period:
                    # ★ FIX: 不足52周，报原始总收益(非线性年化)
                    n_wks = obj3_26['n_weeks']
                    # 从线性年化反推原始总收益: annual = mean*52, total ≈ mean*n = annual*n/52
                    raw_gross = obj3_26['gross_annual'] * n_wks / 52 if n_wks > 0 else 0
                    raw_net = obj3_26['net_annual'] * n_wks / 52 if n_wks > 0 else 0
                    cost_total = obj3_26['cost_annual'] * n_wks / 52 if n_wks > 0 else 0
                    flag = f"{n_wks}w非年化"
                    print(f"  {label:>12s} {raw_gross:>15.1%} {cost_total:>9.1%} {raw_net:>15.1%} "
                          f"{obj3_26['annual_sharpe']:>6.2f} {dir_label:>4s} "
                          f"{obj3_26['avg_turnover']:>5.1%} {flag:>10s}")
                else:
                    print(f"  {label:>12s} {obj3_26['gross_annual']:>9.1%} "
                          f"{obj3_26['cost_annual']:>9.1%} {obj3_26['net_annual']:>9.1%} "
                          f"{obj3_26['annual_sharpe']:>7.2f} {train_sharpe*2.0:>8.2f} "
                          f"{delta:>+7.2f} {dir_label:>4s} {obj3_26['avg_turnover']:>6.1%}")
                
                obj3_2026_results[strat_name] = obj3_26
            except Exception as e:
                print(f"  {label:>12s} ERROR: {e}")
        
        # 结论
        if obj3_2026_results:
            has_positive = any(r['annual_sharpe'] > 0 for r in obj3_2026_results.values())
            avg_sr = np.mean([r['annual_sharpe'] for r in obj3_2026_results.values()])
            avg_cost = np.mean([r['cost_annual'] for r in obj3_2026_results.values()])
            avg_wks = np.mean([r['n_weeks'] for r in obj3_2026_results.values()])
            print(f"\n  === 2026 Obj3 结论 ===")
            if is_short_period:
                print(f"  [INFO] 仅{len(fr_2026)}周数据(<52周), 未做年化, 表中为原始总收益")
                print(f"  [INFO] Sharpe仍用年化公式(无调整), 仅供参考")
            if has_positive:
                print(f"  [OK] 存在2026正Sharpe策略, 因子在最近市场仍有效")
            else:
                print(f"  [WARN] 2026所有策略Sharpe≤0, 成本({avg_cost:.1%}/年)过大")
            print(f"  2026平均Sharpe: {avg_sr:+.2f} | 平均年化成本: {avg_cost:.1%} | 数据{avg_wks:.0f}周")
    else:
        print(f"  2026数据不足 ({len(fr_2026)}周), 跳过时效性验证")
else:
    print("  [WARN] 无标准化因子或无前沿解, 跳过")

# =====================================================
# Phase 7: Walk-forward 因子稳定性 — XQuant Ch6
# =====================================================
print("\n" + "=" * 70)
print("[Phase 7] Walk-forward 因子稳定性 — XQuant Ch6 方法论")
print("          Anchored(锚定式) + Rolling(滚动式) 双验证")
print("          核心: 参数稳定性比参数最优更重要")
print("=" * 70)

try:
    from evaluation.walkforward_stability import run_full_walkforward_validation

    wf_results = run_full_walkforward_validation(
        factor_dfs, forward_returns, ic_summary,
    )

    # 保存
    if wf_results:
        import json
        wf_summary = {}
        for method, result in wf_results.items():
            if result:
                # 序列化非JSON类型
                freq = result.get('factor_freq', {})
                if hasattr(freq, 'most_common'):
                    freq = dict(freq)
                wf_summary[method] = {
                    'stability_score': result.get('stability_score', 0),
                    'verdict': result.get('verdict', 'N/A'),
                    'oos_icir_mean': result.get('oos_icir_mean', 0),
                    'oos_icir_std': result.get('oos_icir_std', 0),
                    'factor_freq': {str(k): int(v) for k, v in freq.items()},
                }

        if not NO_WRITE:
            with open(log_dir / 'walkforward_results.json', 'w', encoding='utf-8') as jf:
                json.dump(wf_summary, jf, indent=2, ensure_ascii=False, default=str)
            print(f"  结果已保存: {log_dir / 'walkforward_results.json'}")

        # 综合健康检查
        a_score = wf_summary.get('anchored', {}).get('stability_score', 0)
        r_score = wf_summary.get('rolling', {}).get('stability_score', 0)
        if a_score >= 70 and r_score >= 70:
            print(f"\n  [PASS] WF 双验证通过 — 因子组合时序稳定, 可进入实盘阶段")
        elif a_score >= 50 and r_score >= 50:
            print(f"\n  [WARN] WF 部分通过 — 建议增加监控频率, 缩短复盘周期")
        else:
            print(f"\n  [FAIL] WF 未通过 — 因子组合可能过拟合, 建议简化或重新设计")
    else:
        print("  [WARN] WF 数据不足, 跳过")
except Exception as e:
    print(f"  [WARN] Phase 7 异常: {e}")
    import traceback; traceback.print_exc()
    wf_results = None

# =====================================================
# Phase 7.5: 防过拟合检查清单 (XQuant Ch6)
# =====================================================
print("\n" + "=" * 70)
print("[Phase 7.5] 防过拟合检查清单 — Brian Peterson '四把铲子'")
print("=" * 70)

try:
    from evaluation.overfitting_checklist import OverfittingChecklist
    from config import MAX_FACTORS

    checklist = OverfittingChecklist()

    # 收集各阶段结果 (部分变量可能在 try 块外未定义)
    oos_data = None
    try:
        if 'oos_results' in dir() and oos_results:
            oos_data = oos_results
    except NameError:
        pass

    wf_data = None
    try:
        if 'wf_results' in dir() and wf_results:
            wf_data = wf_results
    except NameError:
        pass

    burden_data = None
    try:
        if ('burden_results' in dir() and burden_results is not None
                and not (hasattr(burden_results, 'empty') and burden_results.empty)):
            burden_data = burden_results
    except NameError:
        pass

    # 运行全部检查
    checklist_results = checklist.run_all(
        oos_results=oos_data,
        wf_results=wf_data,
        burden_results=burden_data,
        ic_summary=ic_summary,
        factor_dfs=factor_dfs,
        forward_returns=forward_returns,
        current_max_factors=MAX_FACTORS,
    )

    # 打印格式化清单
    checklist.print_checklist(checklist_results)

    # 综合判决
    summary = checklist.generate_summary(checklist_results)

    # 保存到 JSON
    import json
    save_data = {}
    for key, val in checklist_results.items():
        if isinstance(val, dict):
            clean = {}
            for k, v in val.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    clean[k] = v
                elif isinstance(v, list):
                    clean[k] = [str(x) if not isinstance(x, (int, float, bool, type(None))) else x for x in v]
                elif isinstance(v, dict):
                    clean[k] = {
                        str(sk): (str(sv) if not isinstance(sv, (int, float, bool, type(None))) else sv)
                        for sk, sv in v.items()
                    }
                else:
                    clean[k] = str(v)
            save_data[key] = clean

    save_data['summary'] = summary

    if not NO_WRITE:
        with open(log_dir / 'overfitting_checklist.json', 'w', encoding='utf-8') as jf:
            json.dump(save_data, jf, indent=2, ensure_ascii=False, default=str)
        print(f"  结果已保存: {log_dir / 'overfitting_checklist.json'}")

except Exception as e:
    print(f"  [WARN] Phase 7.5 异常: {e}")
    import traceback; traceback.print_exc()

# =====================================================
# 保存结果
# =====================================================
print("\n" + "=" * 70)
print("[保存结果]")
print("=" * 70)

# 保存帕累托前沿
if 'pareto_front' in dir() and len(pareto_front.get('weights', [])) > 0:
    if not NO_WRITE:
        with open(log_dir / 'pareto_front.pkl', 'wb') as pf:
            pf_serializable = {
                'objectives': pareto_front['objectives'].tolist(),
                'weights': pareto_front['weights'],
            }
            pickle.dump(pf_serializable, pf)
    print(f"  帕累托前沿: {len(pareto_front['weights'])} 个解")

if not NO_WRITE:
    ic_summary.to_csv(log_dir / 'ic_summary.csv', encoding='utf-8-sig')
    if not corr_matrix.empty:
        corr_matrix.to_csv(log_dir / 'factor_correlation.csv', encoding='utf-8-sig')

    for name, df in factor_dfs.items():
        df.to_csv(log_dir / f'factor_{name}.csv', encoding='utf-8-sig')

    print(f"  结果目录: {log_dir}")
    print(f"  文件列表: {sorted([str(f.name) for f in log_dir.glob('*')])}")

total_time = time.time() - t_total
m, s = divmod(total_time, 60)
h, m = divmod(m, 60)
print(f"\n  总耗时: {int(h)}h{int(m)}m{int(s)}s")
print(f"  日志: {log_file}")

if not NO_WRITE:
    log_f.close()
sys.stdout = sys.__stdout__

print(f"\n流水线完成! 耗时 {int(h)}h{int(m)}m{int(s)}s")
print(f"结果: {log_dir}")
