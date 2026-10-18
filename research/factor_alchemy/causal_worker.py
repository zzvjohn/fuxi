# -*- coding: utf-8 -*-
"""
因果发现多进程 Worker (无副作用, 供 ProcessPoolExecutor 安全 import)
================================================================
- double_ml_ate: 单因子 Double ML 因果效应 (XGBoost CUDA 第一阶段)
- process_factor: 包装单因子, 返回 (name, result_dict)
主脚本 causal_discovery.py 通过 ProcessPoolExecutor 并行调用本模块。
"""
import gc
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold

# XGBoost 3.x 默认启用 CUDA async 内存池, 跨多次 fit 不释放显存 → 单进程连跑数百次
# GBT fit 会把 12GB VRAM 吃满, 触发 CUDA 驱动层硬崩 (无 Python traceback)。
# 关闭后每次 fit 直接 cudaMalloc/cudaFree, 跑完即归还, 彻底消除累积泄漏。
try:
    xgb.set_config(use_cuda_async_pool=False)
except Exception:
    pass


def _make_model():
    """第一阶段去混杂模型: XGBoost on CUDA (RTX 4080)."""
    return xgb.XGBRegressor(
        n_estimators=100, max_depth=3,
        tree_method='hist', device='cuda',
        n_jobs=1, random_state=42, verbosity=0,
    )


def double_ml_ate(T, Y, W, n_bootstrap, n_splits=5):
    """
    交叉拟合 (cross-fitting) Double ML — 比朴素 bootstrap 重训快 ~6x, 且统计更严谨:
      1. K 折交叉拟合: 在训练折上训 E[Y|W] / E[T|W], 在验证折上产出 OOF 残差
         → 仅 K×2 次 GBT 拟合 (而非每 bootstrap 重训)
      2. ATE = cov(T_res, Y_res) / var(T_res)  (在 OOF 残差上, 一次计算)
      3. Bootstrap 在残差对 (T_res, Y_res) 上重采样重算 ATE → 廉价 OLS, 近零成本
         → n_bootstrap 可提到 200+ 得到紧 CI, 几乎不增加耗时
    """
    n = len(T)
    if n < 50:
        return {'ate': 0, 'p_value': 1.0, 'ci_low': 0, 'ci_high': 0}

    T = np.asarray(T, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)

    # 1. 交叉拟合 → OOF 残差
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    Y_res = np.empty(n, dtype=np.float32)
    T_res = np.empty(n, dtype=np.float32)
    for tr, va in kf.split(W):
        my = _make_model()
        mt = _make_model()
        my.fit(W[tr], Y[tr])
        mt.fit(W[tr], T[tr])
        Y_res[va] = Y[va] - my.predict(W[va])
        T_res[va] = T[va] - mt.predict(W[va])
        # 每折释放模型引用; CUDA 显存池已在模块顶层关闭(use_cuda_async_pool=False),
        # fit 结束后显存即归还驱动, 不再累积。
        del my, mt
    gc.collect()

    # 2. 点估计 ATE
    ate = float(np.cov(T_res, Y_res)[0, 1] / (np.var(T_res) + 1e-10))

    # 3. Bootstrap 仅在残差上 (廉价)
    rng = np.random.default_rng(42)
    ate_b = np.empty(n_bootstrap, dtype=np.float32)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        cov_val = np.cov(T_res[idx], Y_res[idx])[0, 1]
        ate_b[b] = cov_val / (np.var(T_res[idx]) + 1e-10)

    ci_low = float(np.percentile(ate_b, 2.5))
    ci_high = float(np.percentile(ate_b, 97.5))
    # 双边 p-value: min(低于0比例, 高于0比例) × 2
    frac_below = float(np.mean(ate_b <= 0))
    frac_above = float(np.mean(ate_b >= 0))
    p_val = float(min(frac_below, frac_above) * 2)
    if p_val < 1e-8:
        p_val = 1e-8

    return {'ate': ate, 'p_value': p_val,
            'ci_low': ci_low, 'ci_high': ci_high}


def process_factor(args):
    """
    单因子处理。args = (fname, T_array, Y, W, factor_icir, p_thresh, n_bootstrap)
    T_array: 该因子已提取的治疗变量列 (1D array), 避免每个 task 重复 pickle 整个 X_all。
    返回 (fname, result_dict)。
    """
    fname, Ti, Y, W, factor_icir, p_thresh, n_bootstrap = args

    Ti = np.asarray(Ti, dtype=float)
    mask = np.isfinite(Ti)
    if mask.sum() < 100:
        return fname, {'ate': 0, 'p_value': 1.0, 'ci_low': 0, 'ci_high': 0,
                       'significant': False, 'icir': factor_icir.get(fname, 0)}

    try:
        res = double_ml_ate(Ti[mask], Y[mask], W[mask], n_bootstrap)
        res['significant'] = res['p_value'] < p_thresh
        res['icir'] = factor_icir.get(fname, 0)
        return fname, res
    except Exception as e:
        return fname, {'ate': 0, 'p_value': 1.0, 'ci_low': 0, 'ci_high': 0,
                       'significant': False, 'error': str(e),
                       'icir': factor_icir.get(fname, 0)}
