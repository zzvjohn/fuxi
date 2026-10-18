# -*- coding: utf-8 -*-
"""
XGBoost 因子交互发现
════════════════════════════════════════════════
用途: 在 V3 80 base 因子上训练 XGBoost, 提取:
  1. 特征重要性 (gain) → 排名, 发现"漏网"高价值因子
  2. SHAP 交互值     → 最强的因子对交互, 与 V3 已选的 5 对交叉对比
  3. 新复合候选      → V3 pairwise ICIR 漏掉的高阶/非线性组合

设计原则:
  - XGBoost 仅做**发现**, 不直接生成交易信号
  - 发现的交互对 → 候选 rank 乘积 → V3 框架验证 → JQ 真回测

参考: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
"""
from __future__ import annotations

import gc, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# ==================== 配置 ====================
CACHE       = SCRIPT_DIR / 'output' / 'factor_cache.pkl'
OUT_TOP     = SCRIPT_DIR / 'output' / 'xgb_top_interactions_v1.json'      # 最强交互对
OUT_FULL    = SCRIPT_DIR / 'output' / 'xgb_full_interactions_v1.json'     # 完整交互矩阵摘要

# V3 已选的 5 对复合 (来自 regime_lasso_result_composite_v3.json)
V3_SELECTED_PAIRS = {
    ('overnight_5d',        'tvma_20'),
    ('dollar_vol_20d',      'turnover_std'),
    ('money_flow_20',       'value_momentum'),
    ('ret_open_2d_proxy',   'skewness_20'),
    ('gap_up',              'relative_spread_proxy'),
}

# V3 80 base 因子 (from result JSON)
V3_BASE_80 = [
    "bp","f_score","gross_margin","avg_turnover_1m","avg_turnover_3m",
    "turnover_std","turnover_change","abnormal_turnover","ret_1m","ret_3m",
    "ret_6m","max_ret_1m","vol_1m","vol_3m","dollar_vol_20d","dollar_vol_stability",
    "turnover_cv_20d","turnover_to_vol","earnings_growth_yoy","overnight_5d",
    "min_ret_1m","high_low_range","vpt","boll_pct_b","gap_up","opening_gap_momentum",
    "accrual_quality_proxy","attention_decay","volume_stability","bias_20",
    "roc_20","cci_14","price_1m","vr_26","vroc_12","psy_12","money_flow_20",
    "davol_20","skewness_20","atr_14","bull_power","rank_1m","vm_diff",
    "tvma_20","skew_1m","intraday_reversal","range_consistency",
    "volatility_of_volatility","relative_spread_proxy","volume_climax_reversal",
    "ret_open_2d_proxy","earnings_season_vol_div","post_earnings_stability",
    "earnings_volume_drift","bargain_stability","active_quality","value_momentum",
    "lowvol_trend","efficient_growth","high_52w_dist","high_52w_rank",
    "capital_efficiency_proxy","bargaining_power_proxy","ln_mcap",
    "max_drawdown_duration","ln_circulating_mcap","rev_growth_yoy",
    "amihud_illiq","roe","roa","roic","net_margin","accruals",
    "earnings_quality_proxy","asset_turnover","debt_coverage",
    "earnings_stability","cashflow_matching_proxy",
    "operational_efficiency_proxy","ocf_quality",
]

# 嵌套复合因子 (本身不是 base): value_momentum = rank(1/PB) × rank(-ret_3m)
NESTED = {'value_momentum': ('bp', 'ret_3m')}

# ==================== 数据加载 ====================
def load_data():
    print(f'[1/5] 加载因子缓存 {CACHE} ...')
    t0 = time.time()
    cache = pickle.load(open(CACHE, 'rb'))
    print(f'     {len(cache)} 因子, {time.time()-t0:.1f}s')

    # 筛选 V3 base 因子 (排除嵌套复合 — 它们由 base 组成, 已覆盖)
    NESTED_KEYS = {'value_momentum', 'active_quality', 'efficient_growth',
                   'lowvol_trend', 'bargain_stability'}
    available = [f for f in V3_BASE_80 if f in cache and f not in NESTED_KEYS]
    nested_found = [f for f in V3_BASE_80 if f in cache and f in NESTED_KEYS]
    print(f'     V3 base: 可用 {len(available)}, 嵌套已排除 {len(nested_found)}({nested_found})')

    return cache, available


def build_dataset(cache, available):
    """构建 X(factors) + y(forward returns) 完全基于因子缓存, 避免 CSV 对齐问题."""
    print(f'\n[2/5] 构建训练数据...')
    t0 = time.time()

    # 统一索引和列
    first = cache[available[0]]
    weeks = sorted(first.index)
    raw_cols = first.columns.tolist()
    # 过滤 NaN 列名 (缓存中第一个列名是 NaN)
    stocks = sorted([c for c in raw_cols if isinstance(c, str)])

    # 前向周收益: 用 ret_1m 前移1周的差 → 近似周收益
    # ret_1m ≈ 4周累计收益; (next_ret_1m - ret_1m)/4 ≈ 下周收益
    if 'ret_1m' in cache:
        ret_df = cache['ret_1m'].reindex(index=weeks, columns=stocks)
        # 前向收益 = next_week_ret_1m - current_ret_1m (除以4约当周收益)
        fwd_ret = (ret_df.shift(-1) - ret_df) / 4.0
        y = fwd_ret.values.ravel().astype(np.float32)
        print(f'     前向收益: ret_1m 差分 (ret_df={ret_df.shape})')
    else:
        raise KeyError('因子缓存中缺少 ret_1m, 无法计算前向收益')

    # 构建 X: 所有因子展平
    X_list = []
    for fname in available:
        df = cache[fname].reindex(index=weeks, columns=stocks)
        X_list.append(df.values.ravel())
    X = np.column_stack(X_list).astype(np.float32)
    del X_list, df
    gc.collect()

    # 清洗
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    # winsorize y at ±0.6 (6 个跌停)
    y = np.clip(y, -0.6, 0.6)
    print(f'     有效样本: {len(y):,} | 因子: {X.shape[1]} | {time.time()-t0:.1f}s')
    return X, y, available


def train_xgb(X, y, factor_names):
    """训练 XGBoost."""
    print(f'\n[3/5] 训练 XGBoost...')
    t0 = time.time()

    import xgboost as xgb

    n = len(y)
    split = int(n * 0.70)
    np.random.seed(42)
    idx = np.random.permutation(n)
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_val, y_val = X[idx[split:]], y[idx[split:]]
    print(f'     训练 {len(y_train):,} | 验证 {len(y_val):,}')

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=[f'f_{i}' for i in range(X.shape[1])])
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=[f'f_{i}' for i in range(X.shape[1])])

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.0,
        'reg_alpha': 0.1,
        'min_child_weight': 10,
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'verbosity': 1,
        'nthread': 4,
    }

    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(params, dtrain, num_boost_round=200,
                      evals=evals, early_stopping_rounds=20,
                      verbose_eval=20)
    print(f'     best iter={model.best_iteration}, val RMSE={model.best_score:.6f}')
    print(f'     耗时: {time.time()-t0:.1f}s')
    return model, factor_names


def extract_interactions(model, factor_names, X, y):
    """提取特征重要性 + 树路径共现交互."""
    print(f'\n[4/5] 提取特征重要性 + 树路径交互...')
    t0 = time.time()

    n_f = len(factor_names)

    # 特征重要性 (gain)
    importance = model.get_score(importance_type='gain')
    imp_map = {}
    for k, v in importance.items():
        idx = int(k.replace('f_', ''))
        imp_map[factor_names[idx]] = float(v)
    total_gain = sum(imp_map.values())

    # 树路径共现 — 统计因子对在同一路径上的共现次数
    # 方法: 解析每棵树的每条路径, 看哪些因子在同一条决策路径上
    booster = model  # xgb.train() returns Booster directly in xgboost 3.x
    df_trees = booster.trees_to_dataframe()

    # 共现计数矩阵
    cooccur = np.zeros((n_f, n_f), dtype=np.float64)

    # 按 (Tree, Node) 分组: 同一条从根到叶的路径上出现的因子
    # trees_to_dataframe 有 Tree, Node, ID, Feature, Split, Yes, No, Missing 列
    # 我们遍历每棵树, 提取每条根→叶路径上的因子集合

    for tree_id in df_trees['Tree'].unique():
        tree = df_trees[df_trees['Tree'] == tree_id]

        # 找所有叶子节点
        leaves = tree[tree['Feature'] == 'Leaf']

        for _, leaf in leaves.iterrows():
            # 回溯从叶到根的路径
            path_nodes = []
            node_id = leaf['ID']
            # 构建 parent map
            parent_of = {}
            for _, row in tree.iterrows():
                if row['Feature'] != 'Leaf':
                    yes_node = row['Yes']
                    no_node  = row['No']
                    parent_of[yes_node] = row['ID']
                    parent_of[no_node]  = row['ID']

            while node_id in parent_of:
                pid = parent_of[node_id]
                parent_row = tree[tree['ID'] == pid]
                if not parent_row.empty:
                    feat = parent_row['Feature'].values[0]
                    if feat != 'Leaf':
                        path_nodes.append(feat)
                node_id = pid

            # 路径上的因子两两共现 +1
            for i in range(len(path_nodes)):
                fi = path_nodes[i].replace('f_', '')
                if not fi.isdigit():
                    continue
                fi = int(fi)
                for j in range(i + 1, len(path_nodes)):
                    fj = path_nodes[j].replace('f_', '')
                    if not fj.isdigit():
                        continue
                    fj = int(fj)
                    cooccur[fi, fj] += 1
                    cooccur[fj, fi] += 1

    # 归一化: 除以总叶子路径数
    total_paths = len(df_trees[df_trees['Feature'] == 'Leaf'])
    if total_paths > 0:
        cooccur /= total_paths

    print(f'     交互矩阵: {n_f}×{n_f} | 树数={df_trees["Tree"].nunique()} | 叶路径={total_paths} | {time.time()-t0:.1f}s')

    return cooccur, imp_map, total_gain, n_f


def report_interactions(inter_abs, imp_map, total_gain, factor_names, y):
    """报告最强交互对."""
    print(f'\n[5/5] 报告交互发现...')

    n_f = len(factor_names)

    # 1. 特征重要性 Top 30
    ranked = sorted(imp_map.items(), key=lambda x: x[1], reverse=True)
    print(f'\n{"="*70}')
    print(f'  特征重要性 Top 30 (gain, 占比)')
    print(f'{"="*70}')
    for i, (name, g) in enumerate(ranked[:30], 1):
        pct = 100 * g / total_gain
        in_v3 = '★V3' if any(name in p for p in V3_SELECTED_PAIRS) else ''
        print(f'  {i:2d}. {name:32s}  gain={g:8.1f} ({pct:5.1f}%) {in_v3}')

    # 2. 最强交互对 (排除对角线)
    print(f'\n{"="*70}')
    print(f'  SHAP 交互值 Top 30 (因子对, 绝对值均值)')
    print(f'{"="*70}')

    interactions = []
    for i in range(n_f):
        for j in range(i + 1, n_f):
            interactions.append((factor_names[i], factor_names[j], float(inter_abs[i, j])))

    interactions.sort(key=lambda x: x[2], reverse=True)

    # 标注 V3 已选/嵌套
    for rank, (fa, fb, val) in enumerate(interactions[:30], 1):
        pair = (fa, fb)
        pair_r = (fb, fa)

        v3_status = ''
        if pair in V3_SELECTED_PAIRS or pair_r in V3_SELECTED_PAIRS:
            v3_status = ' ✓ V3已选'
        elif fa in NESTED or fb in NESTED:
            v3_status = ' ⚠嵌套'

        print(f'  {rank:2d}. {fa:30s} × {fb:30s}   SHAP={val:.6f}{v3_status}')

    # 3. 新候选 (排除 V3 已选和嵌套)
    print(f'\n{"="*70}')
    print(f'  ★ 新复合候选 (V3 未选, 非嵌套, 高交互 + 高重要性)')
    print(f'{"="*70}')

    candidates = []
    for fa, fb, shap_val in interactions:
        pair = (fa, fb)
        pair_r = (fb, fa)
        if pair in V3_SELECTED_PAIRS or pair_r in V3_SELECTED_PAIRS:
            continue
        if fa in NESTED or fb in NESTED:
            continue

        imp_score = imp_map.get(fa, 0) + imp_map.get(fb, 0)
        candidates.append((fa, fb, shap_val, imp_score))

    # 按交互值 × 重要性 综合排序
    candidates.sort(key=lambda x: x[2] * (1 + x[3] / total_gain), reverse=True)

    for rank, (fa, fb, shap_val, imp_sum) in enumerate(candidates[:20], 1):
        print(f'  {rank:2d}. {fa:30s} × {fb:30s}   SHAP={shap_val:.6f}  imp_sum={imp_sum:.0f}')

    # 4. 存文件
    top_interactions = []
    for fa, fb, shap_val, imp_sum in candidates[:50]:
        top_interactions.append({
            'pair': [fa, fb],
            'shap_interaction': round(shap_val, 6),
            'importance_sum': round(float(imp_sum), 1),
            'importance_pct': round(100 * float(imp_sum) / total_gain, 1),
        })

    result = {
        'method': 'xgb_interaction_v1',
        'n_factors': n_f,
        'n_effective_samples': len(y),
        'top_importance': [(name, round(float(g), 1), round(100*float(g)/total_gain, 1))
                           for name, g in ranked[:30]],
        'top_interactions': top_interactions,
        'v3_selected': [list(p) for p in V3_SELECTED_PAIRS],
    }

    json.dump(result, open(OUT_TOP, 'w'), ensure_ascii=False, indent=2)
    print(f'\n  结果已存: {OUT_TOP}')
    print(f'  候选交互: {len(top_interactions)} 对')

    return top_interactions


# ==================== 主流程 ====================
def main():
    print('XGBoost 因子交互发现 v1')
    print('=' * 60)

    cache, available = load_data()
    X, y, factor_names = build_dataset(cache, available)

    model, factor_names = train_xgb(X, y, factor_names)
    inter_abs, imp_map, total_gain, n_f = extract_interactions(model, factor_names, X, y)

    top = report_interactions(inter_abs, imp_map, total_gain, factor_names, y)

    print(f'\n{"="*60}')
    print(f'  完成! 发现 {len(top)} 对新复合候选')
    print(f'  下一步: 在 V3 框架中验证 → rank_pct 乘积 → ICIR 筛选 → JQ 回测')


if __name__ == '__main__':
    main()
