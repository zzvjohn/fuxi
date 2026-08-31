#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regime-aware Lasso 因子选择 + ICIR 加权
=========================================
替代 NSGA-II 的确定性方法。

流程:
  1. 加载数据 + 计算因子 (复用 factor_alchemy 模块)
  2. 标准化 (cross-sectional z-score)
  3. 加载 regime 标签, 对齐到因子周频日期
  4. 按 regime 切分:
     - ICIR >= 0.1 预筛选 (regime 内)
     - LassoCV 5-fold 特征选择 (因子 -> forward_return)
     - ICIR 加权 (|ICIR| 归一化为权重, 方向由 Lasso 系数符号决定)
  5. Walk-forward OOS 验证 (前70%训练, 后30%验证)
  6. 产出 JQ 策略 (仅对 JQ_FACTOR_REGISTRY 中有映射的因子)

用法:
  cd <项目根目录> && python factor_alchemy/regime_lasso.py

输出:
  - output/regime_lasso_result.json  (策略权重 + 回测指标)
  - strategies/drafts/fa_v79_lasso_*.py (JQ 策略文件)
"""
import sys, os, time, json, pickle, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ── 版本 / 选择器 (v7.10) ──
# 版本管理: 当前文件为 v7.10 (Step 1 = ElasticNet + XGBoost + Boruta 替代 Lasso)。
# 回退基线: archive/v7.9_lasso/regime_lasso.py (纯 LassoCV)。
# 运行时回退: --selector lasso 可精确复现 v7.9 行为 (用全量数据, 不做子采样)。
import argparse
_PARSER = argparse.ArgumentParser(description='Regime-aware factor selection (v7.10)')
_PARSER.add_argument('--selector', type=str, default='ensemble',
                     choices=['lasso', 'elasticnet', 'xgboost', 'boruta', 'ensemble', 'ensemble_calibrated', 'composite'],
                     help='因子选择方法 (默认 ensemble; composite = 配对乘积复合 + 三层ICIR/稳定性/衰减评分)')
_PARSER.add_argument('--tag', type=str, default='v710',
                     help='输出文件标签 (默认 v710, 用于区分不同 selector 的结果)')
_PARSER.add_argument('--max-factors', type=int, default=5,
                     help='每个 regime 最多保留因子数 (默认 5)')
_PARSER.add_argument('--no-regime', action='store_true',
                     help='不分区 regime, 全样本训练单一因子组合')
_PARSER.add_argument('--icir-threshold', type=float, default=0.10,
                     help='ICIR 阈值 (base 因子和 composite 粗筛共用, 默认 0.10)')
_ARGS, _ = _PARSER.parse_known_args()
SELECTOR = _ARGS.selector
OUT_TAG = _ARGS.tag
MAX_FACTORS_CLI = _ARGS.max_factors
NO_REGIME = _ARGS.no_regime
ICIR_THRESHOLD_CLI = _ARGS.icir_threshold

# ── 路径 ──
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data.loader import load_all_data
from factors import ALL_FACTORS, get_factor_instance
from config import FACTOR_DEFS
from config import COMPOSITE_FACTORS as _COMPOSITE_DEFS
from config import CAUSAL_FACTORS_TOP20
from factors.composite import standardize_factors
from evaluation.ic_analysis import compute_ic_icir, compute_ic_icir_fast

# ── JQ 生成 ──
try:
    from jq_generator import JQGenerator
    from jq_generator.registry import JQ_FACTOR_REGISTRY
    JQ_AVAILABLE = True
except Exception as e:
    print(f"[WARN] JQ 生成器不可用: {e}")
    JQ_AVAILABLE = False

# =====================================================
# 配置
# =====================================================
OUTPUT_DIR = SCRIPT_DIR / 'output'
STRATEGIES_DIR = SCRIPT_DIR.parent / 'strategies' / 'drafts'
REGIME_CSV = SCRIPT_DIR / 'data' / 'regime_labels.csv'

# 因子计算跳过列表 (已知崩溃)
SKIP_FACTORS = {'beta', 'idio_vol'}  # duplicate-index reindex error
SKIP_PREFIX = ()  # 不跳过 _neut (但在 Lasso 路径中会过滤)

# ★ 因果因子子集模式 (注入因果发现推荐的 20 因子, 走 V7.11 ensemble_calibrated)
CAUSAL_FACTORS_ENV = os.environ.get('CAUSAL_FACTORS')
_CAUSAL_FACTORS = [f.strip() for f in CAUSAL_FACTORS_ENV.split(',') if f.strip()] if CAUSAL_FACTORS_ENV else None
if _CAUSAL_FACTORS:
    print(f"[CAUSAL MODE] 仅使用因果因子子集 ({len(_CAUSAL_FACTORS)} 个): {_CAUSAL_FACTORS}")

# Lasso 参数
LASSO_CV = 5
LASSO_MAX_ITER = 5000
ICIR_THRESHOLD = ICIR_THRESHOLD_CLI  # ICIR 阈值, 默认 0.10, CLI 可覆盖

# Walk-forward
WF_TRAIN_RATIO = 0.70  # 前 70% 训练, 后 30% OOS (默认单窗口)
WF_PURGE_WEEKS = 1     # Purge: 训练集尾部剔除周数 (防止因子回望窗口泄露到测试集)
WF_EMBARGO_WEEKS = 1   # Embargo: 训练/测试间间隔周数
# 多起点 WFA (OOS 协议加固): 多个 train:test 切分比例 → 取均值做最终评估
WF_MULTI_RATIOS = [0.50, 0.60, 0.70]  # 对应 50:50, 60:40, 70:30 三种切分

# Top-N 组合回测
TOP_N = 30
QMT_COST_WEEKLY = 0.0072 / 5  # QMT 双边成本 0.72%, 周频 ~0.144%

# ★ FIX (v7.10.1, 2026-07-29): 收益截幅 — 过滤数据毛刺
# 未复权/退市/复牌等异常会让单只股票周收益出现 ±90%+ 的伪值,
# 直接取组合均值会污染净值曲线 (曾导致 Phase 7 MaxDD=-99.7%).
# 截幅到 ±0.6 = 覆盖 6 个连续跌停仍远未触及, 仅剔除数据错误。
def _winsor_ret(s):
    """对单只股票周收益序列做截幅 (clip) 后返回."""
    return s.clip(-0.6, 0.6)

print("=" * 70)
print("  Regime-aware Lasso + ICIR 因子选择 (确定性方法)")
print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# =====================================================
# Phase 1: 数据加载
# =====================================================
print("\n[Phase 1] 数据加载")
t1 = time.time()
data = load_all_data('2021-01-01', '2026-06-25')
price_data = data['price_data']
financial_data = data['financial_data']
valuation_data = data['valuation_data']
industry_map = data.get('industry_map', {})
close = price_data.get('close')
print(f"  价格矩阵: {close.shape} | 耗时: {time.time()-t1:.0f}s")

# ★ v7.10 校准: 提取流通市值用于 calibrated proxy
mcap_weekly = None
if valuation_data is not None:
    _mcap_col = None
    if 'circ_mv' in valuation_data.columns:
        _mcap_col = 'circ_mv'
    elif 'market_cap' in valuation_data.columns:
        _mcap_col = 'market_cap'
    if _mcap_col:
        _mcap_raw = valuation_data[['code', 'trade_date', _mcap_col]].drop_duplicates().copy()
        _mcap_raw['trade_date'] = pd.to_datetime(_mcap_raw['trade_date'])
        _mcap_piv = _mcap_raw.pivot(index='trade_date', columns='code', values=_mcap_col)
        mcap_weekly = _mcap_piv.resample('W').last()
        mcap_weekly.index = pd.to_datetime(mcap_weekly.index)
        print(f"  流通市值(周频, {_mcap_col}): {mcap_weekly.shape}")

# 周频前向收益
close_weekly = close.resample('W').last()
close_weekly.index = pd.to_datetime(close_weekly.index)
forward_returns = close_weekly.shift(-1) / close_weekly - 1
print(f"  周频收益: {forward_returns.shape} ({forward_returns.index[0].strftime('%Y-%m-%d')} ~ {forward_returns.index[-1].strftime('%Y-%m-%d')})")

# =====================================================
# Phase 2: 因子计算
# =====================================================
# 因子缓存 (避免重跑 7 分钟因子计算)
FACTOR_CACHE = SCRIPT_DIR / 'output' / 'factor_cache.pkl'

if FACTOR_CACHE.exists():
    print(f"\n[Phase 2] 因子计算 [CACHE HIT]")
    t2 = time.time()
    with open(FACTOR_CACHE, 'rb') as f:
        factor_dfs = pickle.load(f)
    print(f"  从缓存加载: {len(factor_dfs)} 个因子 | 耗时: {time.time()-t2:.1f}s")
else:
    print("\n[Phase 2] 因子计算")
    t2 = time.time()
    factor_dfs = {}

    for name in FACTOR_DEFS:
        if name not in ALL_FACTORS:
            continue
        if name in SKIP_FACTORS:
            continue
        if name.endswith('_neut'):
            continue
        if name in _COMPOSITE_DEFS:
            continue  # 复合因子在后处理阶段从 base factor 生成, 跳过独立 compute
        if _CAUSAL_FACTORS is not None and name not in _CAUSAL_FACTORS:
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
                df_weekly = df.resample('W').last()

                is_financial = df_weekly.shape[0] <= 10 and factor_def['category'] in \
                    ('profitability', 'value', 'growth')
                if is_financial:
                    df_weekly = df_weekly.reindex(close_weekly.index).ffill().bfill()
                else:
                    common_idx = close_weekly.index.intersection(df_weekly.index)
                    if len(common_idx) > 0:
                        df_weekly = df_weekly.reindex(common_idx)
                factor_dfs[name] = df_weekly
        except Exception as e:
            pass

    print(f"  成功: {len(factor_dfs)} 个因子 | 耗时: {time.time()-t2:.0f}s")
    # 缓存
    with open(FACTOR_CACHE, 'wb') as f:
        pickle.dump(factor_dfs, f)
    print(f"  缓存已保存: {FACTOR_CACHE}")

# ★ 因果因子子集过滤 (确保 factor_dfs 仅含因果因子, 命中缓存/重算均生效)
if _CAUSAL_FACTORS is not None:
    _before = len(factor_dfs)
    factor_dfs = {k: v for k, v in factor_dfs.items() if k in _CAUSAL_FACTORS}
    _missing = set(_CAUSAL_FACTORS) - set(factor_dfs.keys())
    print(f"  [CAUSAL] factor_dfs 过滤: {_before} -> {len(factor_dfs)} | 缺失: {_missing}")

# =====================================================
# ★ 跨周期稳健因子: 从缓存 base factor 生成交叉信号源复合因子
# 注入点: CAUSAL 过滤之后, Phase 3 标准化之前
# 复合因子 = rank_pct(A) × rank_pct(B), 每截面独立 rank→[0,1]
# 设计目标: 两个不相关信号源需同向才触发, 比单一因子更跨周期稳健
# =====================================================
def compute_composite_factors(factor_dfs, composite_defs):
    """从已加载的 base factor 生成 rank-percentile 交叉乘积复合因子

    向量化实现: 两因子先对齐到 union 日期, 再按截面(axis=1)同时 rank→[0,1] 后相乘。
    避免逐行 loc 因日期索引错位抛 KeyError。
    """
    new_factors = {}
    for cname, cdef in composite_defs.items():
        fa, fb = cdef['factors']
        invert = cdef.get('invert', [])
        if fa not in factor_dfs or fb not in factor_dfs:
            print(f"    [Composite SKIP] {cname}: 缺少 {fa if fa not in factor_dfs else fb}")
            continue
        dfa = factor_dfs[fa].astype(float)
        dfb = factor_dfs[fb].astype(float)
        if fa in invert:
            dfa = -dfa
        if fb in invert:
            dfb = -dfb
        # 对齐到 union 日期 (缺失日期该截面全 NaN, rank 时自然排末尾)
        _common = dfa.index.union(dfb.index)
        dfa = dfa.reindex(_common)
        dfb = dfb.reindex(_common)
        # 同时 rank (两因子同一截面) → 乘积
        comp = dfa.rank(pct=True, axis=1, na_option='keep') * dfb.rank(pct=True, axis=1, na_option='keep')
        comp = comp.astype(float)
        new_factors[cname] = comp
        print(f"    [Composite] {cname} = rank({fa})×rank({fb}) | {comp.shape}")
    return new_factors

try:
    from config import COMPOSITE_FACTORS
except Exception:
    COMPOSITE_FACTORS = {}

if COMPOSITE_FACTORS:
    _t_comp = time.time()
    _new = compute_composite_factors(factor_dfs, COMPOSITE_FACTORS)
    for _cn, _cf in _new.items():
        factor_dfs[_cn] = _cf
        if _cn not in FACTOR_DEFS:
            FACTOR_DEFS[_cn] = {'category': 'composite', 'label': COMPOSITE_FACTORS[_cn]['label']}
    print(f"  [Composite] 新增 {len(_new)} 个交叉信号复合因子 | 总因子: {len(factor_dfs)} | 耗时 {time.time()-_t_comp:.1f}s")


# =====================================================
# Phase 2.5: 行业/市值中性化 (V5 核心改进)
# =====================================================
# 在标准化前剥离行业和市值影响 → composer 选出的 composite 是"纯 alpha"
# 控制开关: NEUTRALIZE_V5 = False 可跳过
# =====================================================
NEUTRALIZE_V5 = True
_NEUTRALIZE_R2_THRESHOLD = 0.80  # 若平均 R² > 0.80, 说明因子可能被行业/市值过度解释, 记录警告但继续

if NEUTRALIZE_V5 and industry_map and mcap_weekly is not None and len(industry_map) > 0:
    print(f"\n[Phase 2.5] 行业/市值中性化 (预处理层)")
    _t_neu = time.time()
    from neutralize import neutralize_factor_df as _neutralize_factor_df
    from neutralize import diagnose_neutralization as _diagnose_neut

    n_neutralized = 0
    n_skipped = 0
    neu_r2_list = []

    # 对齐 mcap_weekly 到因子日期范围
    _all_dates = pd.DatetimeIndex([])
    for _fdf in factor_dfs.values():
        _all_dates = _all_dates.union(_fdf.index)
    _mcap_aligned = mcap_weekly.reindex(_all_dates).ffill().bfill()
    # 只保留有市值数据的日期
    _mcap_valid_dates = _mcap_aligned.dropna(how='all').index
    if len(_mcap_valid_dates) < 20:
        print(f"  [WARN] 市值数据不足 ({len(_mcap_valid_dates)} 周), 跳过中性化")
        NEUTRALIZE_V5 = False
    else:
        print(f"  行业映射: {len(industry_map)} 只股票 | 市值矩阵: {_mcap_aligned.shape} | 有效周: {len(_mcap_valid_dates)}")

    if NEUTRALIZE_V5:
        for _fn in list(factor_dfs.keys()):
            _fdf = factor_dfs[_fn]
            # 筛选有效的股票列 (非 NaN 列名)
            _valid_cols = [c for c in _fdf.columns if isinstance(c, str)]
            if len(_valid_cols) < 30:
                n_skipped += 1
                continue
            _fdf = _fdf[_valid_cols]

            try:
                _neut, _info = _neutralize_factor_df(
                    _fdf, industry_map, _mcap_aligned,
                    min_stocks=30, min_industries=2, verbose=False,
                )
                n_neutralized += 1
                r2_vals = [pi.get('r_squared', float('nan')) for pi in _info
                           if pi.get('status') == 'ok' and 'r_squared' in pi]
                if r2_vals:
                    _avg_r2 = np.nanmean(r2_vals)
                    neu_r2_list.append((_fn, _avg_r2))
                factor_dfs[_fn] = _neut
            except Exception as _e:
                n_skipped += 1
                if n_skipped <= 3:
                    print(f"    [NEUT SKIP] {_fn}: {_e}")

    print(f"  中性化完成: {n_neutralized}/{len(factor_dfs)} 个因子, 跳过 {n_skipped}")
    if neu_r2_list:
        _avg_all_r2 = np.mean([v[1] for v in neu_r2_list])
        print(f"  平均中性化 R²: {_avg_all_r2:.4f}")
        if _avg_all_r2 > _NEUTRALIZE_R2_THRESHOLD:
            print(f"  [WARN] R² > {_NEUTRALIZE_R2_THRESHOLD}, 因子可能被行业/市值过度解释")
        # 显示 Top-5 最高 R² 的因子 (行业/市值暴露最严重)
        _top5 = sorted(neu_r2_list, key=lambda x: -x[1])[:5]
        print(f"  Top-5 行业暴露因子 (最高 R²):")
        for _fn, _r2 in _top5:
            print(f"    {_fn:35s} R²={_r2:.4f}")
    print(f"  耗时: {time.time()-_t_neu:.1f}s")
elif NEUTRALIZE_V5:
    print("\n[Phase 2.5] 行业/市值中性化 [SKIP] — industry_map={} mcap_weekly={}".format(
        'OK' if industry_map else 'MISSING',
        'OK' if mcap_weekly is not None else 'MISSING'))
    NEUTRALIZE_V5 = False

_mid_tag = "V5_NEUT" if NEUTRALIZE_V5 else "V5_NONEUT"

# =====================================================
# Phase 3: 标准化
# =====================================================
print("\n[Phase 3] 标准化 (cross-sectional z-score)")
factor_dfs_std = standardize_factors(factor_dfs)
print(f"  标准化完成: {len(factor_dfs_std)} 个因子")

# =====================================================
# Phase 4: Regime 标签加载 + 对齐
# =====================================================
print("\n[Phase 4] Regime 标签加载")
regime_df = pd.read_csv(REGIME_CSV, parse_dates=['date'])
regime_df = regime_df.sort_values('date').reset_index(drop=True)

# ffill 合并 neutral
regime_df['regime_merged'] = regime_df['regime'].replace('neutral', np.nan).ffill()
regime_df['regime_merged'] = regime_df['regime_merged'].fillna('small_bull')  # 头部兜底

# 对齐到 forward_returns 的周频日期 (W-SUN -> W-FRI, 容差 10 天)
fr_index = pd.DatetimeIndex(forward_returns.index)
regime_series = regime_df.set_index('date')['regime_merged']

# nearest match + ffill (与 run_fa.py Phase 2.6 一致)
_aligned = regime_series.reindex(fr_index, method='nearest', tolerance=pd.Timedelta(days=10))
_aligned = _aligned.ffill().bfill()
regime_map = {d: _aligned.loc[d] for d in fr_index if pd.notna(_aligned.loc[d])}

# 确保所有周在 forward_returns 和所有因子 DataFrame 中都有数据 (避免 KeyError)
_valid_weeks = forward_returns.index
for _fdf in factor_dfs_std.values():
    _valid_weeks = _valid_weeks.intersection(_fdf.index)
_n_before = len(regime_map)
regime_map = {d: r for d, r in regime_map.items() if d in _valid_weeks}
if _n_before != len(regime_map):
    print(f"  [INFO] 过滤到全因子共有周: {_n_before} -> {len(regime_map)}")

regime_set = sorted(set(regime_map.values()))
print(f"  对齐: {len(regime_map)}/{len(fr_index)} 周 (覆盖率 {100*len(regime_map)/len(fr_index):.0f}%)")
for r in regime_set:
    cnt = sum(1 for v in regime_map.values() if v == r)
    print(f"    {r}: {cnt} 周 ({100*cnt/len(regime_map):.0f}%)")

# ★ --no-regime: 全样本合并为单一 regime
if NO_REGIME:
    print("\n  [--no-regime] 全样本合并为单一 'all' regime")
    # 使用所有有效周, 覆盖 regime_map
    regime_map = {d: 'all' for d in _valid_weeks if d in forward_returns.index}
    regime_set = ['all']
    print(f"    all: {len(regime_map)} 周")

# =====================================================
# Phase 5: Per-regime Lasso + ICIR 因子选择
# =====================================================
print("\n" + "=" * 70)
print("[Phase 5] Per-regime 因子选择 + ICIR 加权")
print(f"  选择器: {SELECTOR} | 标签: {OUT_TAG}")
print("=" * 70)


# =====================================================
# v7.10 因子选择器: Lasso / ElasticNet / XGBoost / Boruta / Ensemble
# =====================================================
def select_factors(method, X, y, selected_factors, factor_icir, regime_name,
                   lasso_max_iter=5000, subsample_cap=40000):
    """返回 (nonzero: dict[factor]->signed_value, meta: dict)

    signed_value 用途:
      - lasso / elasticnet: = 模型系数 (带符号, 直接反映方向)
      - xgboost / boruta / ensemble: = sign(ICIR) 代理 (树模型不分方向, 方向由 ICIR 决定,
        与 Phase 5.4 ICIR 加权 + 方向校验逻辑一致)

    所有方法最终都走 Phase 5.4 的 ICIR 加权, 保证输出接口与 v7.9 完全一致,
    JQ 生成器 / regime_adaptive 策略无需任何改动即可复用。
    """
    n = len(selected_factors)
    meta = {'method': method, 'n_input': n}

    # 子采样 (仅 ML 方法; lasso 用全量以精确复现 v7.9 基线)
    if method == 'lasso':
        Xs, ys = X, y
    else:
        if len(X) > subsample_cap:
            Xs = X.sample(subsample_cap, random_state=42)
            ys = y.loc[Xs.index]
            meta['subsampled'] = True
        else:
            Xs, ys = X, y

    def _sign_from_icir(fname):
        v = factor_icir.get(fname, 0)
        return v if v != 0 else 1.0

    if method == 'lasso':
        from sklearn.linear_model import LassoCV
        m = LassoCV(cv=5, max_iter=lasso_max_iter, n_jobs=-1, random_state=42)
        m.fit(Xs, ys)
        coefs = pd.Series(m.coef_, index=selected_factors)
        nonzero = coefs[coefs != 0]
        meta['alpha'] = float(m.alpha_)
        if len(nonzero) == 0:
            nonzero = pd.Series({k: v for k, v in
                                sorted(factor_icir.items(), key=lambda x: abs(x[1]), reverse=True)[:4]})
        return nonzero.to_dict(), meta

    if method == 'elasticnet':
        from sklearn.linear_model import ElasticNetCV
        m = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                         cv=5, max_iter=lasso_max_iter, n_jobs=-1, random_state=42)
        m.fit(Xs, ys)
        coefs = pd.Series(m.coef_, index=selected_factors)
        nonzero = coefs[coefs != 0]
        meta['alpha'] = float(m.alpha_)
        meta['l1_ratio'] = float(m.l1_ratio_)
        if len(nonzero) == 0:
            nonzero = pd.Series({k: v for k, v in
                                sorted(factor_icir.items(), key=lambda x: abs(x[1]), reverse=True)[:4]})
        return nonzero.to_dict(), meta

    if method == 'xgboost':
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                             n_jobs=-1, random_state=42, verbosity=0)
        m.fit(Xs, ys)
        imp = pd.Series(m.feature_importances_, index=selected_factors).sort_values(ascending=False)
        meta['xgb_top'] = {k: round(float(v), 4) for k, v in imp.head(8).items()}
        thr = imp.mean()
        chosen = imp[imp > thr].index.tolist()
        if len(chosen) < 3:
            chosen = imp.head(max(3, min(5, n))).index.tolist()
        nonzero = {f: _sign_from_icir(f) for f in chosen}
        meta['n_chosen'] = len(chosen)
        return nonzero, meta

    if method == 'boruta':
        from sklearn.ensemble import RandomForestRegressor
        from boruta import BorutaPy
        rf = RandomForestRegressor(n_estimators=200, max_depth=7, n_jobs=-1, random_state=42)
        boruta = BorutaPy(rf, n_estimators=50, random_state=42, verbose=0)
        boruta.fit(Xs.values, ys.values.ravel())
        mask = boruta.support_
        chosen = [selected_factors[i] for i in range(n) if mask[i]]
        if len(chosen) == 0:
            chosen = [selected_factors[i] for i in range(n) if boruta.support_weak_[i]]
        if len(chosen) == 0:
            chosen = selected_factors[:max(3, min(5, n))]
        nonzero = {f: _sign_from_icir(f) for f in chosen}
        meta['n_confirmed'] = len(chosen)
        return nonzero, meta

    if method == 'ensemble':
        from sklearn.ensemble import RandomForestRegressor
        from boruta import BorutaPy
        import xgboost as xgb
        from sklearn.linear_model import ElasticNetCV
        # 1) Boruta 确认相关因子 (挡住噪声因子, 防过拟合)
        rf = RandomForestRegressor(n_estimators=200, max_depth=7, n_jobs=-1, random_state=42)
        boruta = BorutaPy(rf, n_estimators=50, random_state=42, verbose=0)
        boruta.fit(Xs.values, ys.values.ravel())
        confirmed = [selected_factors[i] for i in range(n) if boruta.support_[i]]
        if len(confirmed) < 3:
            confirmed = [selected_factors[i] for i in range(n) if boruta.support_weak_[i]]
        if len(confirmed) < 3:
            confirmed = selected_factors[:max(3, min(8, n))]
        meta['boruta_confirmed'] = len(confirmed)
        Xc = Xs[confirmed]
        # 2) ElasticNet 排序 (处理高度共线的因子组)
        en = ElasticNetCV(l1_ratio=[0.3, 0.5, 0.7, 0.9], cv=5,
                          max_iter=lasso_max_iter, n_jobs=-1, random_state=42)
        en.fit(Xc, ys)
        en_coef = pd.Series(np.abs(en.coef_), index=confirmed)
        en_coef = en_coef / (en_coef.max() + 1e-9)
        # 3) XGBoost 重要性排序 (捕获非线性交互)
        xg = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                              n_jobs=-1, random_state=42, verbosity=0)
        xg.fit(Xc, ys)
        xg_imp = pd.Series(xg.feature_importances_, index=confirmed)
        xg_imp = xg_imp / (xg_imp.max() + 1e-9)
        # 4) 合并排序 (各 50%)
        combined = (0.5 * en_coef + 0.5 * xg_imp).sort_values(ascending=False)
        meta['ensemble_rank'] = {k: round(float(v), 4) for k, v in combined.head(8).items()}
        topk = combined.head(max(3, min(8, len(confirmed)))).index.tolist()
        nonzero = {f: _sign_from_icir(f) for f in topk}
        meta['n_chosen'] = len(topk)
        return nonzero, meta

    # fallback -> ICIR top-4
    nonzero = pd.Series({k: v for k, v in
                        sorted(factor_icir.items(), key=lambda x: abs(x[1]), reverse=True)[:4]})
    return nonzero.to_dict(), meta


all_regime_strategies = {}

for regime_name in regime_set:
    print(f"\n--- {regime_name} ---")
    regime_weeks = sorted([d for d, r in regime_map.items() if r == regime_name])
    print(f"  训练周数: {len(regime_weeks)}")

    # 最小周数检查: 过少数据量 → ICIR 不可靠 + WFA 窗口无意义
    MIN_REGIME_WEEKS = 30
    if len(regime_weeks) < MIN_REGIME_WEEKS:
        print(f"  [SKIP] 周数 {len(regime_weeks)} < {MIN_REGIME_WEEKS}, 跳过此 regime (数据不足以做复合因子搜索)")
        all_regime_strategies[regime_name] = {
            'error': f'insufficient_weeks_{len(regime_weeks)}',
            'n_weeks': len(regime_weeks),
        }
        continue

    # --- 5.1 计算 regime 内每因子 ICIR ---
    print(f"  [5.1] 计算因子 ICIR...")
    factor_icir = {}
    factor_ic_series = {}

    for fname, fdf in factor_dfs_std.items():
        # 取 regime 周的数据
        regime_cols = [c for c in fdf.columns if c in forward_returns.columns]
        fdf_regime = fdf.loc[regime_weeks, regime_cols]
        fr_regime = forward_returns.loc[regime_weeks, regime_cols]

        try:
            ic_ir = compute_ic_icir_fast(fdf_regime, fr_regime, min_samples=20)
            if ic_ir and 'icir' in ic_ir and pd.notna(ic_ir['icir']):
                factor_icir[fname] = ic_ir['icir']
                factor_ic_series[fname] = ic_ir.get('ic_series', pd.Series(dtype=float))
            else:
                if fname in COMPOSITE_FACTORS:
                    print(f"    [DEBUG] {fname} ICIR 计算返回空/NaN: {ic_ir}")
        except Exception as _e:
            if fname in COMPOSITE_FACTORS:
                print(f"    [DEBUG] {fname} ICIR 计算异常: {type(_e).__name__}: {_e}")
            pass

    print(f"  ICIR 计算: {len(factor_icir)} 因子")
    # [DEBUG] dump 完整 ICIR 含复合因子
    import json as _json
    _dump = {k: round(float(v), 4) for k, v in factor_icir.items()}
    try:
        _json.dump(_dump, open(SCRIPT_DIR / 'output' / f'_debug_icir_{regime_name}.json', 'w'), ensure_ascii=False, indent=2)
        print(f"    [DEBUG] 已 dump factor_icir ({len(_dump)} 因子) → _debug_icir_{regime_name}.json")
    except PermissionError:
        print(f"    [WARN] 无法写入 _debug_icir_{regime_name}.json (文件被锁定), 跳过")

    # ICIR 预筛选
    icir_pass = {k: v for k, v in factor_icir.items() if abs(v) >= ICIR_THRESHOLD}
    print(f"  ICIR >= {ICIR_THRESHOLD}: {len(icir_pass)} 因子")

    # ── [Causal Injection] 因果因子强制注入 composer base 池 ──
    # 因果 Top20 中的因子即使 ICIR 不达标也允许进入配对搜索。
    # 理由: Double ML 证实这些因子有显著因果效应但被 ICIR 低估;
    #       配合 FDR 校正后的 composer 粗筛可安全扩展搜索空间。
    # 注意: bargaining_power_proxy 在禁投令中仅禁"独立使用", 
    #       允许作为 composer 配对成分 (其与互补信号配对可能释放隐藏 alpha)。
    causal_injected = []
    if SELECTOR == 'composite':
        for fn in CAUSAL_FACTORS_TOP20:
            if fn in factor_dfs and fn not in icir_pass:
                icir_pass[fn] = factor_icir.get(fn, 0.0)
                causal_injected.append(fn)
    if causal_injected:
        print(f"  [Causal] 注入 {len(causal_injected)} 个因果因子到 base 池: {causal_injected}")
        print(f"  [Causal] base 池扩至: {len(icir_pass)} (原 ICIR={len(icir_pass)-len(causal_injected)} + 因果={len(causal_injected)})")

    # ── [Composer Base 扩容] 基本面×量价定向交叉 ──
    # 当前 base 池偏向量价/技术因子。手动注入基本面品类中被 ICIR 低估的因子,
    # 丰富搜索空间中的"基本面×量价"交叉配对 (已被 V2 验证为赢家结构)。
    _fundamental_boost = [
        'roe', 'roa', 'roic', 'gross_margin', 'net_margin',       # 盈利能力
        'f_score', 'accruals', 'earnings_quality_proxy',          # 质量
        'asset_turnover', 'debt_coverage', 'earnings_stability',  # 效率/风险
        'cashflow_matching_proxy', 'operational_efficiency_proxy', # 运营
        'ocf_quality',                                             # 现金流
    ]
    _fund_boosted = []
    if SELECTOR == 'composite':
        for fn in _fundamental_boost:
            if fn in factor_dfs and fn not in icir_pass and fn not in SKIP_FACTORS:
                icir_pass[fn] = factor_icir.get(fn, 0.0)
                _fund_boosted.append(fn)
    if _fund_boosted:
        print(f"  [FundBoost] 注入 {len(_fund_boosted)} 个基本面因子: {_fund_boosted}")

    if len(icir_pass) < 3:
        print(f"  [WARN] ICIR 通过因子 < 3, 跳过此 regime")
        all_regime_strategies[regime_name] = {'error': 'insufficient_factors', 'n_pass': len(icir_pass)}
        continue

    # --- 5.2 构建回归矩阵 ---
    print(f"  [5.2] 构建回归矩阵 (Lasso 输入)...")
    selected_factors = sorted(icir_pass.keys())

    # 拉平: (N_stocks * N_weeks, N_factors) -> y = forward_return
    X_parts = []
    y_parts = []
    for week in regime_weeks:
        if week not in forward_returns.index:
            continue
        fr_week = forward_returns.loc[week]
        row_data = {}
        for fname in selected_factors:
            fdf = factor_dfs_std[fname]
            if week in fdf.index:
                row_data[fname] = fdf.loc[week]

        if len(row_data) < len(selected_factors) * 0.5:
            continue  # 缺太多因子

        # 构建 DataFrame
        week_df = pd.DataFrame(row_data)
        week_y = fr_week.reindex(week_df.index)

        # 丢弃 NaN
        mask = week_df.notna().all(axis=1) & week_y.notna()
        X_parts.append(week_df[mask])
        y_parts.append(week_y[mask])

    if not X_parts:
        print(f"  [WARN] 无法构建回归矩阵, 跳过此 regime")
        all_regime_strategies[regime_name] = {'error': 'no_valid_data'}
        continue

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    print(f"  回归矩阵: X={X.shape}, y={y.shape} | 因子: {selected_factors}")

    if len(X) == 0 or len(y) == 0:
        print(f"  [WARN] 回归矩阵有 0 行 (对齐后无有效样本), 跳过此 regime")
        all_regime_strategies[regime_name] = {'error': 'zero_samples_after_align', 'n_weeks': len(regime_weeks)}
        continue

    # --- 5.3 因子选择 (v7.10 选择器) ---
    print(f"  [5.3] 选择器={SELECTOR} 特征选择...")
    t_sel = time.time()
    
    if SELECTOR == 'composite':
        # ── 路径 B: 交叉信号源复合 + 三层稳健评分 (学术依据: Harvey-Liu-Zhu 2016, McLean-Pontiff 2016, Ilmanen 2021) ──
        from factor_composer import CompositeSelector
        cs = CompositeSelector(icir_threshold=ICIR_THRESHOLD)
        nonzero, sel_meta = cs.select(
            factor_dfs_raw=factor_dfs,
            factor_dfs_std=factor_dfs_std,
            forward_returns=forward_returns,
            icir_pass_base=icir_pass,
            regime_weeks=regime_weeks,
            factor_ic_series=factor_ic_series,
            factor_icir=factor_icir,
            top_k=MAX_FACTORS_CLI,
            verbose=True,
        )
    elif SELECTOR == 'ensemble_calibrated':
        # ── v7.11: Boruta + 校准 PortfolioSimulator 评分 ──
        from ga.fast_calibrated_proxy import calibrated_ensemble_select
        nonzero, sel_meta = calibrated_ensemble_select(
            factor_names=selected_factors,
            factor_dfs_std=factor_dfs_std,
            close_weekly=close_weekly,
            mcap_weekly=mcap_weekly,
            regime_weeks=regime_weeks,
            factor_icir_dict=factor_icir,
            X=X, y=y,
            top_k=MAX_FACTORS_CLI,
            verbose=True,
        )
    else:
        nonzero, sel_meta = select_factors(
            SELECTOR, X, y, selected_factors, factor_icir, regime_name,
            lasso_max_iter=LASSO_MAX_ITER)
    print(f"  选择完成: {len(nonzero)}/{len(selected_factors)} 因子 | 耗时: {time.time()-t_sel:.1f}s")
    for fname, val in sorted(nonzero.items(), key=lambda x: abs(x[1]), reverse=True):
        icir_val = factor_icir.get(fname, 0)
        direction = "+" if val > 0 else "-"
        print(f"    {fname:35s} val={val:+.4f} ICIR={icir_val:+.3f} dir={direction}")
    if 'ensemble_rank' in sel_meta:
        print(f"    ensemble_rank={sel_meta['ensemble_rank']}")
    elif 'xgb_top' in sel_meta:
        print(f"    xgb_top={sel_meta['xgb_top']}")
    if 'boruta_confirmed' in sel_meta:
        print(f"    boruta_confirmed={sel_meta['boruta_confirmed']}/{len(selected_factors)}")

    if len(nonzero) == 0:
        print(f"  [WARN] 未选中任何因子, 回退到 ICIR top-4")
        nonzero = {k: v for k, v in sorted(icir_pass.items(), key=lambda x: abs(x[1]), reverse=True)[:4]}

    # --- 5.4 加权 ---
    if SELECTOR == 'composite':
        _wlabel = '三层评分加权'
    elif SELECTOR == 'ensemble_calibrated':
        _wlabel = '校准绩效加权'
    else:
        _wlabel = 'ICIR 加权'
    print(f"  [5.4] {_wlabel}...")
    weights = {}
    
    if SELECTOR == 'composite':
        # 三层评分复合模式: nonzero 的 value = 三层合成分数 [0,1], 直接用做权重
        for fname, score in nonzero.items():
            if abs(score) > 0.001:
                weights[fname] = abs(score)
            else:
                print(f"    [SKIP] {fname}: composite score≈0, 丢弃")
    elif SELECTOR == 'ensemble_calibrated':
        # 校准模式下 nonzero 的 value = signed calibrated Sharpe
        # 直接用 abs(value) 做权重（因子方向已由 ICIR 符号嵌入 signed score）
        for fname, signed_score in nonzero.items():
            if abs(signed_score) > 0.001:
                weights[fname] = abs(signed_score)
            else:
                print(f"    [SKIP] {fname}: calibrated score≈0, 丢弃")
    else:
        for fname, coef in nonzero.items():
            icir_val = factor_icir.get(fname, 0)
            coef_sign = 1 if coef > 0 else -1

            # 方向一致性检查
            if coef_sign * icir_val > 0:
                # 方向一致: ICIR 正 + Lasso 正 -> 正向因子
                weights[fname] = abs(icir_val)
            elif abs(icir_val) > 0.3:
                # 方向矛盾但 ICIR 强: 信任 ICIR 方向
                weights[fname] = abs(icir_val)
                print(f"    [WARN] {fname}: Lasso={coef:+.4f} ICIR={icir_val:+.3f} 方向矛盾, 信任 ICIR")
            else:
                # 方向矛盾且 ICIR 弱: 丢弃
                print(f"    [SKIP] {fname}: Lasso={coef:+.4f} ICIR={icir_val:+.3f} 方向矛盾且弱, 丢弃")

    if len(weights) == 0:
        print(f"  [WARN] 所有因子方向矛盾, 回退到 ICIR top-4")
        for fname, icir_val in sorted(icir_pass.items(), key=lambda x: abs(x[1]), reverse=True)[:4]:
            weights[fname] = abs(icir_val)

    # --- Top-K 截断 (优先有 JQ 映射的因子) ---
    MAX_FACTORS = MAX_FACTORS_CLI
    if len(weights) > MAX_FACTORS:
        jq_avail = {k: v for k, v in weights.items() if k in JQ_FACTOR_REGISTRY} if JQ_AVAILABLE else {}
        jq_missing = {k: v for k, v in weights.items() if k not in jq_avail}

        sorted_jq = sorted(jq_avail.items(), key=lambda x: -x[1])
        sorted_no = sorted(jq_missing.items(), key=lambda x: -x[1])

        selected = dict(sorted_jq[:MAX_FACTORS])
        for k, v in sorted_no:
            if len(selected) >= MAX_FACTORS:
                break
            selected[k] = v

        dropped = set(weights.keys()) - set(selected.keys())
        print(f"  [Top-{MAX_FACTORS}] 截断: {len(weights)} -> {len(selected)} 因子 (JQ映射: {len(jq_avail)}有/{len(jq_missing)}缺)")
        if dropped:
            print(f"           丢弃: {dropped}")
        weights = selected

    # 归一化
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    print(f"\n  ★ {regime_name} 最终策略:")
    for fname, w in sorted(weights.items(), key=lambda x: -x[1]):
        icir_val = factor_icir.get(fname, 0)
        print(f"    {fname:35s} weight={w:.3f} ICIR={icir_val:+.3f}")

    all_regime_strategies[regime_name] = {
        'weights': weights,
        'factor_icir': {k: v for k, v in factor_icir.items() if k in weights},
        'selector': SELECTOR,
        'selector_meta': {k: v for k, v in sel_meta.items()
                          if not isinstance(v, dict) or k in ('pair_map', 'flip_map')},
        'n_nonzero': len(nonzero),
        'n_train_weeks': len(regime_weeks),
        'n_factors_input': len(selected_factors),
    }

# =====================================================
# Phase 6: Walk-forward OOS 验证
# =====================================================
print("\n" + "=" * 70)
print(f"[Phase 6] Multi-window Walk-forward OOS 验证 (ratios={WF_MULTI_RATIOS}, purge={WF_PURGE_WEEKS}w, embargo={WF_EMBARGO_WEEKS}w)")
print("=" * 70)

wf_results = {}

for regime_name, strat in all_regime_strategies.items():
    if 'error' in strat:
        wf_results[regime_name] = {'error': strat['error']}
        continue

    weights = strat['weights']
    regime_weeks = sorted([d for d, r in regime_map.items() if r == regime_name])
    n_total = len(regime_weeks)

    # ── Multi-window WFA: 多起点滚动验证 ──
    window_metrics = []
    for wf_ratio in WF_MULTI_RATIOS:
        # 切分: 前 wf_ratio 为训练, 后 (1-wf_ratio) 为测试
        n_train_raw = int(n_total * wf_ratio)
        # Purge: 训练集尾端剔除 PURGE_WEEKS (防因子计算回望窗口泄露)
        n_train = max(10, n_train_raw - WF_PURGE_WEEKS)
        # Embargo: 训练/测试间间隔 EMBARGO_WEEKS
        oos_start = n_train_raw + WF_EMBARGO_WEEKS
        if oos_start >= n_total:
            continue  # 切分无 OOS 区间

        train_weeks = regime_weeks[:n_train]
        oos_weeks = regime_weeks[oos_start:]

        if len(oos_weeks) < 5:
            continue

        # --- WFA 子窗口内: 6.1 训练集重算 ICIR 权重 ---
        train_icir = {}
        for fname in weights:
            fdf = factor_dfs_std.get(fname)
            if fdf is None:
                continue
            cols = [c for c in fdf.columns if c in forward_returns.columns]
            fdf_t = fdf.loc[fdf.index.intersection(train_weeks), cols]
            fr_t = forward_returns.loc[forward_returns.index.intersection(train_weeks), cols]
            common_idx = fdf_t.index.intersection(fr_t.index)
            if len(common_idx) < 10:
                continue
            try:
                ic_ir = compute_ic_icir(fdf_t.loc[common_idx], fr_t.loc[common_idx], min_samples=10)
                if ic_ir and 'icir' in ic_ir and pd.notna(ic_ir['icir']):
                    train_icir[fname] = ic_ir['icir']
            except:
                pass

        train_weights = {}
        for fname, w in weights.items():
            icir_val = train_icir.get(fname, 0)
            if abs(icir_val) > 0:
                train_weights[fname] = abs(icir_val)
            else:
                train_weights[fname] = w

        total_tw = sum(train_weights.values())
        if total_tw > 0:
            train_weights = {k: v / total_tw for k, v in train_weights.items()}

        # --- WFA 子窗口内: 6.2 OOS 组合回测 ---
        portfolio_returns = []
        for week in oos_weeks:
            if week not in forward_returns.index:
                continue
            composite = pd.Series(0.0, index=forward_returns.columns)
            for fname, w in train_weights.items():
                fdf = factor_dfs_std.get(fname)
                if fdf is not None and week in fdf.index:
                    composite += w * fdf.loc[week].reindex(composite.index).fillna(0)
            composite = composite.dropna()
            if len(composite) < TOP_N:
                continue
            top_stocks = composite.nlargest(TOP_N).index
            fr_week = forward_returns.loc[week]
            valid = [s for s in top_stocks if s in fr_week.index and pd.notna(fr_week[s])]
            if len(valid) < TOP_N // 2:
                continue
            port_ret = _winsor_ret(fr_week[valid]).mean()
            portfolio_returns.append({'week': week, 'return': port_ret, 'n_stocks': len(valid)})

        if not portfolio_returns or len(portfolio_returns) < 5:
            continue

        wf_df = pd.DataFrame(portfolio_returns).set_index('week')
        wf_df['cum_ret'] = (1 + wf_df['return']).cumprod() - 1
        weekly_mean = wf_df['return'].mean()
        weekly_std = wf_df['return'].std()
        annual_ret = weekly_mean * 52
        annual_std = weekly_std * np.sqrt(52)
        sharpe = annual_ret / annual_std if annual_std > 0 else 0
        max_dd = (wf_df['cum_ret'] - wf_df['cum_ret'].cummax()).min()

        # OOS ICIR
        oos_icir_vals = []
        for fname in train_weights:
            fdf = factor_dfs_std.get(fname)
            if fdf is None:
                continue
            cols = [c for c in fdf.columns if c in forward_returns.columns]
            fdf_oos = fdf.loc[fdf.index.intersection(oos_weeks), cols]
            fr_oos = forward_returns.loc[forward_returns.index.intersection(oos_weeks), cols]
            common_idx = fdf_oos.index.intersection(fr_oos.index)
            if len(common_idx) < 5:
                continue
            try:
                ic_ir = compute_ic_icir(fdf_oos.loc[common_idx], fr_oos.loc[common_idx], min_samples=5)
                if ic_ir and 'icir' in ic_ir and pd.notna(ic_ir['icir']):
                    oos_icir_vals.append(ic_ir['icir'])
            except:
                pass
        oos_icir_mean = np.mean(oos_icir_vals) if oos_icir_vals else 0

        window_metrics.append({
            'ratio': wf_ratio,
            'train_weeks': len(train_weeks),
            'oos_weeks': len(oos_weeks),
            'train_start': train_weeks[0].strftime('%Y-%m-%d') if train_weeks else '',
            'train_end': train_weeks[-1].strftime('%Y-%m-%d') if train_weeks else '',
            'oos_start': oos_weeks[0].strftime('%Y-%m-%d') if oos_weeks else '',
            'oos_end': oos_weeks[-1].strftime('%Y-%m-%d') if oos_weeks else '',
            'annual_return': float(annual_ret),
            'sharpe': float(sharpe),
            'max_dd': float(max_dd),
            'oos_icir': float(oos_icir_mean),
            'train_weights': train_weights,
        })

    if not window_metrics:
        wf_results[regime_name] = {'error': 'no_valid_windows'}
        continue

    # ── 汇总各窗口指标 ──
    annual_returns = [m['annual_return'] for m in window_metrics]
    sharpes = [m['sharpe'] for m in window_metrics]
    max_dds = [m['max_dd'] for m in window_metrics]
    oos_icirs = [m['oos_icir'] for m in window_metrics]

    print(f"\n  ★ {regime_name} Multi-WFA 汇总 ({len(window_metrics)} 窗口):")
    for m in window_metrics:
        print(f"    ratio={m['ratio']:.2f}  train={m['train_start']}~{m['train_end']} ({m['train_weeks']}w)  "
              f"oos={m['oos_start']}~{m['oos_end']} ({m['oos_weeks']}w)  "
              f"ret={m['annual_return']:+.1%}  SR={m['sharpe']:.2f}  DD={m['max_dd']:.1%}")

    print(f"    ── Pooled (mean): ret={np.mean(annual_returns):+.1%}  SR={np.mean(sharpes):.2f}  "
          f"DD={np.mean(max_dds):.1%}  ICIR={np.mean(oos_icirs):+.3f}")
    print(f"    ── Range: ret=[{np.min(annual_returns):+.1%}, {np.max(annual_returns):+.1%}]  "
          f"SR=[{np.min(sharpes):.2f}, {np.max(sharpes):.2f}]  "
          f"DD=[{np.min(max_dds):.1%}, {np.max(max_dds):.1%}]")

    # 使用 70:30 窗口作为主记录 (保持向后兼容), 其余作为子窗口详情
    primary = next((m for m in window_metrics if m['ratio'] == 0.70), window_metrics[-1])
    wf_results[regime_name] = {
        'annual_return': float(primary['annual_return']),
        'sharpe': float(primary['sharpe']),
        'max_dd': float(primary['max_dd']),
        'oos_icir': float(primary['oos_icir']),
        'n_oos_weeks': primary['oos_weeks'],
        'n_train_weeks': primary['train_weeks'],
        'train_weights': primary['train_weights'],
        'multi_window': {
            'n_windows': len(window_metrics),
            'ratios': [m['ratio'] for m in window_metrics],
            'mean_annual_return': float(np.mean(annual_returns)),
            'min_annual_return': float(np.min(annual_returns)),
            'max_annual_return': float(np.max(annual_returns)),
            'mean_sharpe': float(np.mean(sharpes)),
            'min_sharpe': float(np.min(sharpes)),
            'max_sharpe': float(np.max(sharpes)),
            'mean_max_dd': float(np.mean(max_dds)),
            'min_max_dd': float(np.min(max_dds)),
            'max_max_dd': float(np.max(max_dds)),
            'mean_oos_icir': float(np.mean(oos_icirs)),
            'window_details': [{k: v for k, v in m.items() if k != 'train_weights'} for m in window_metrics],
        },
    }

# =====================================================
# Phase 6b: 分年 ICIR 一致性表 (OOS 协议加固)
# =====================================================
print("\n" + "=" * 70)
print("[Phase 6b] 分年 ICIR 一致性诊断")
print("=" * 70)

# 收集所有 regime 中选中的因子 (去重)
_all_selected_factors = set()
for regime_name, strat in all_regime_strategies.items():
    if 'error' not in strat and 'weights' in strat:
        _all_selected_factors.update(strat['weights'].keys())
_selected_list = sorted(_all_selected_factors)
print(f"  所有 regime 选中因子 (去重): {len(_selected_list)} 个")

# 按年份 (2021-2026) 分别计算每个因子的 ICIR
year_labels = [2021, 2022, 2023, 2024, 2025, 2026]
_sample_weeks = forward_returns.index

yearly_icir = {}  # {factor_name: {year: icir}}
yearly_ic_mean = {}  # {factor_name: {year: mean_IC}}
yearly_n = {}  # {factor_name: {year: n_weeks}}

for fname in _selected_list:
    fdf = factor_dfs_std.get(fname)
    if fdf is None:
        continue
    yearly_icir[fname] = {}
    yearly_ic_mean[fname] = {}
    yearly_n[fname] = {}

    for yr in year_labels:
        yr_weeks = [w for w in _sample_weeks if getattr(w, 'year', None) == yr]
        if not yr_weeks:
            yr_weeks = [w for w in _sample_weeks
                        if hasattr(w, 'year') and w.year == yr]
        if len(yr_weeks) < 5:
            continue

        fdf_yr = fdf.loc[fdf.index.intersection(yr_weeks)]
        fr_yr = forward_returns.loc[forward_returns.index.intersection(yr_weeks)]
        if len(fdf_yr) < 5:
            continue

        try:
            ic_res = compute_ic_icir_fast(fdf_yr, fr_yr, min_samples=5)
            if ic_res and 'icir' in ic_res and pd.notna(ic_res['icir']):
                yearly_icir[fname][yr] = round(float(ic_res['icir']), 3)
                yearly_ic_mean[fname][yr] = round(float(ic_res.get('ic_mean', 0)), 4)
                yearly_n[fname][yr] = len(fdf_yr)
        except Exception:
            pass

# 构建 DataFrame
if yearly_icir:
    yr_df_data = {}
    for fname, yr_dict in yearly_icir.items():
        if len(yr_dict) >= 2:  # 至少 2 年有数据
            yr_df_data[fname] = yr_dict

    if yr_df_data:
        yr_df = pd.DataFrame(yr_df_data).T
        yr_df.columns = [str(c) for c in yr_df.columns]

        # 计算一致性指标
        yr_df['mean_icir'] = yr_df.mean(axis=1)
        yr_df['std_icir'] = yr_df.std(axis=1)
        yr_df['min_icir'] = yr_df.min(axis=1)
        yr_df['max_icir'] = yr_df.max(axis=1)
        # 一致性 = 1 - (std/|mean|)  (0=不稳定, 1=完美一致)
        yr_df['consistency'] = np.where(
            np.abs(yr_df['mean_icir']) > 0.001,
            np.clip(1.0 - yr_df['std_icir'] / (np.abs(yr_df['mean_icir']) + 0.001), -5, 1),
            0.0
        )
        # 正值年份占比
        yr_df['positive_years'] = yr_df[[str(y) for y in year_labels
                                         if str(y) in yr_df.columns]].gt(0).sum(axis=1)
        yr_df['total_years'] = yr_df[[str(y) for y in year_labels
                                      if str(y) in yr_df.columns]].notna().sum(axis=1)
        yr_df['positive_ratio'] = np.where(
            yr_df['total_years'] > 0,
            yr_df['positive_years'] / yr_df['total_years'],
            0.0
        )
        yr_df = yr_df.sort_values('mean_icir', ascending=False)

        print(f"\n  ★ 选中因子分年 ICIR 表 ({len(yr_df)} 因子, {len(year_labels)} 年):")
        print(f"  {'Factor':40s} {'ICIR':>6s} {'一致性':>6s} {'正年%':>5s}", end='')
        for yr in year_labels:
            print(f" {' ' + str(yr) + ' ':>7s}", end='')
        print()
        print("  " + "-" * 110)

        for fname, row in yr_df.iterrows():
            short = fname[:38] + '..' if len(fname) > 40 else fname
            con_flag = '⚠' if row['consistency'] < 0.3 else ('★' if row['consistency'] > 0.7 else ' ')
            print(f"  {con_flag}{short:39s} {row['mean_icir']:+6.3f} {row['consistency']:+6.2f} {row['positive_ratio']:5.0%}", end='')
            for yr in year_labels:
                val = row.get(str(yr), np.nan)
                if pd.notna(val):
                    print(f" {val:+7.3f}", end='')
                else:
                    print(f" {'-':>7s}", end='')
            print()

        # 标记"仅在特定年份有效"的脆弱因子
        fragile = yr_df[(yr_df['consistency'] < 0.3) & (yr_df['total_years'] >= 3)]
        if len(fragile) > 0:
            print(f"\n  ⚠ 年份一致性不足 (consistency<0.3) 的因子 ({len(fragile)} 个):")
            for fname in fragile.index:
                row = fragile.loc[fname]
                print(f"     {fname:45s} 均值ICIR={row['mean_icir']:+.3f} 一致性={row['consistency']:+.2f} "
                      f"范围=[{row['min_icir']:+.3f}, {row['max_icir']:+.3f}]")

        # 保存 CSV
        try:
            yr_df.to_csv(OUTPUT_DIR / '_yearly_icir_table.csv', encoding='utf-8-sig')
            print(f"\n  分年 ICIR 表已保存: output/_yearly_icir_table.csv")
        except Exception as e:
            print(f"  [WARN] 保存分年 ICIR 表失败: {e}")
    else:
        print("  [WARN] 无足够年份数据生成 ICIR 表")
else:
    print("  [WARN] 无选中因子可计算分年 ICIR")

# =====================================================
# Phase 7: 全样本回测 (所有 regime 合并)
# =====================================================
print("\n" + "=" * 70)
print("[Phase 7] 全样本回测 (regime-aware 合成)")
print("=" * 70)

# 合成: 每周根据 regime 标签选择对应的权重
all_weeks = sorted(regime_map.keys())
portfolio_returns_all = []

for week in all_weeks:
    if week not in forward_returns.index:
        continue

    regime = regime_map[week]
    strat = all_regime_strategies.get(regime, {})
    if 'error' in strat or 'weights' not in strat:
        continue

    weights = strat['weights']
    composite = pd.Series(0.0, index=forward_returns.columns)
    for fname, w in weights.items():
        fdf = factor_dfs_std.get(fname)
        if fdf is not None and week in fdf.index:
            composite += w * fdf.loc[week].reindex(composite.index).fillna(0)

    composite = composite.dropna()
    if len(composite) < TOP_N:
        continue

    top_stocks = composite.nlargest(TOP_N).index
    fr_week = forward_returns.loc[week]
    valid = [s for s in top_stocks if s in fr_week.index and pd.notna(fr_week[s])]
    if len(valid) < TOP_N // 2:
        continue

    port_ret = _winsor_ret(fr_week[valid]).mean()
    # 扣除 QMT 成本
    port_ret_net = port_ret - QMT_COST_WEEKLY

    portfolio_returns_all.append({
        'week': week,
        'regime': regime,
        'return_gross': port_ret,
        'return_net': port_ret_net,
        'n_stocks': len(valid),
    })

if portfolio_returns_all:
    bt_df = pd.DataFrame(portfolio_returns_all).set_index('week')
    bt_df['cum_gross'] = (1 + bt_df['return_gross']).cumprod() - 1
    bt_df['cum_net'] = (1 + bt_df['return_net']).cumprod() - 1

    weekly_mean_net = bt_df['return_net'].mean()
    weekly_std_net = bt_df['return_net'].std()
    annual_ret_net = weekly_mean_net * 52
    annual_std_net = weekly_std_net * np.sqrt(52)
    sharpe_net = annual_ret_net / annual_std_net if annual_std_net > 0 else 0
    max_dd_net = (bt_df['cum_net'] - bt_df['cum_net'].cummax()).min()

    # 分 regime 统计
    regime_stats = bt_df.groupby('regime')['return_net'].agg(['mean', 'std', 'count'])
    regime_stats['annual'] = regime_stats['mean'] * 52
    regime_stats['sharpe'] = regime_stats['annual'] / (regime_stats['std'] * np.sqrt(52))

    print(f"\n  全样本 (regime-aware 合成, 扣 QMT 成本):")
    print(f"  周数: {len(bt_df)} | 年化收益: {annual_ret_net:+.1%} | Sharpe: {sharpe_net:.2f} | MaxDD: {max_dd_net:.1%}")
    print(f"\n  分 regime:")
    for r, row in regime_stats.iterrows():
        print(f"    {r:20s} {row['count']:>3.0f}周 | 年化: {row['annual']:+.1%} | Sharpe: {row['sharpe']:.2f}")

    # 2026 时效验证
    bt_2026 = bt_df[bt_df.index >= '2026-01-01']
    if len(bt_2026) > 0:
        w26 = bt_2026['return_net'].mean()
        s26 = bt_2026['return_net'].std()
        a26 = w26 * 52
        sh26 = a26 / (s26 * np.sqrt(52)) if s26 > 0 else 0
        print(f"\n  2026 时效验证 ({len(bt_2026)}周): 年化 {a26:+.1%} | Sharpe {sh26:.2f}")

    # 保存回测结果
    bt_result = {
        'annual_return_net': float(annual_ret_net),
        'sharpe_net': float(sharpe_net),
        'max_dd_net': float(max_dd_net),
        'n_weeks': len(bt_df),
    }
else:
    bt_result = {'error': 'no_portfolio_returns'}
    print("  [WARN] 无组合收益")

# =====================================================
# Phase 8: 保存结果 + JQ 策略生成
# =====================================================
print("\n" + "=" * 70)
print("[Phase 8] 保存结果 + JQ 策略生成")
print("=" * 70)

# 保存 JSON (带版本/选择器标签, 便于回退对比)
result = {
    'timestamp': datetime.now().isoformat(),
    'method': f'{SELECTOR}_v7.10',
    'selector': SELECTOR,
    'out_tag': OUT_TAG,
    'regimes': {},
    'walk_forward': {},
    'backtest': bt_result,
}

for rname, strat in all_regime_strategies.items():
    if 'error' in strat:
        result['regimes'][rname] = {'error': strat['error']}
    else:
        result['regimes'][rname] = {
            'weights': strat['weights'],
            'factor_icir': strat['factor_icir'],
            'selector': strat.get('selector'),
            'selector_meta': strat.get('selector_meta'),
            'n_train_weeks': strat['n_train_weeks'],
        }

for rname, wf in wf_results.items():
    result['walk_forward'][rname] = wf

result_path = OUTPUT_DIR / f'regime_lasso_result_{OUT_TAG}.json'
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
print(f"  结果保存: {result_path}")

# JQ 策略生成
if JQ_AVAILABLE:
    print(f"\n  JQ 策略生成:")
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

    for rname, strat in all_regime_strategies.items():
        if 'error' in strat:
            print(f"    {rname}: SKIP (error)")
            continue

        weights = strat['weights']
        # 只保留 JQ_FACTOR_REGISTRY 中有映射的因子
        jq_weights = {k: v for k, v in weights.items() if k in JQ_FACTOR_REGISTRY}
        missing = set(weights.keys()) - set(JQ_FACTOR_REGISTRY.keys())

        if len(jq_weights) < 2:
            print(f"    {rname}: SKIP (仅 {len(jq_weights)} 因子有 JQ 映射, 需 >= 2)")
            if missing:
                print(f"           缺映射: {missing}")
            continue

        # 重新归一化
        total = sum(jq_weights.values())
        jq_weights = {k: v / total for k, v in jq_weights.items()}

        # 生成
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        fname = f'fa_{OUT_TAG}_{SELECTOR}_{rname}.py'
        fpath = STRATEGIES_DIR / fname

        try:
            gen = JQGenerator(
                weights=jq_weights,
                name=f'{SELECTOR}_{rname}',
                mcap_min=5,
            )
            gen.generate(str(fpath))
            print(f"    {rname}: ✅ {fname}")
            if missing:
                print(f"           (缺 JQ 映射: {missing})")
        except Exception as e:
            print(f"    {rname}: ❌ {e}")

# 汇总
print("\n" + "=" * 70)
print("  完成!")
print("=" * 70)
print(f"  方法: selector={SELECTOR} (v7.10) + ICIR 加权")
print(f"  标签: {OUT_TAG}")
print(f"  Regimes: {list(all_regime_strategies.keys())}")
for rname, strat in all_regime_strategies.items():
    if 'weights' in strat:
        print(f"  {rname}: {len(strat['weights'])} 因子")
print(f"  结果文件: {result_path}")
