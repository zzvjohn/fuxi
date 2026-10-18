"""
因子综合评分 / GA适应度计算
=============================
5维度评估:
  1. ICIR (ICIR_QUALITY_THRESHOLD = 0.5 质量判断 | ICIR_THRESHOLD = 1.5 NSGA-II校准)
  2. 单调性 (MONOTONICITY_P_THRESHOLD = 0.05)
  3. 低相关性 (CORRELATION_THRESHOLD = 0.5)
  4. 双重排序检验 (DOUBLE_SORT_P_THRESHOLD = 0.05)
  5. 时序稳定性 (STABILITY_THRESHOLD = 0.5) ★ 新增

通过标准: 5选3 (GA_PASS_MIN_DIMENSIONS = 3)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np

# 从项目级 config 导入阈值
import importlib.util
_config_path = Path(__file__).parent.parent / 'config.py'
spec = importlib.util.spec_from_file_location('factor_alchemy_config', _config_path)
config_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_mod)

ICIR_THRESHOLD = config_mod.ICIR_THRESHOLD
ICIR_QUALITY_THRESHOLD = getattr(config_mod, 'ICIR_QUALITY_THRESHOLD', 0.5)
MONOTONICITY_P_THRESHOLD = config_mod.MONOTONICITY_P_THRESHOLD
CORRELATION_THRESHOLD = config_mod.CORRELATION_THRESHOLD
DOUBLE_SORT_P_THRESHOLD = config_mod.DOUBLE_SORT_P_THRESHOLD
STABILITY_THRESHOLD = config_mod.STABILITY_THRESHOLD
GA_WEIGHT_ICIR = config_mod.GA_WEIGHT_ICIR
GA_WEIGHT_MONOTONICITY = config_mod.GA_WEIGHT_MONOTONICITY
GA_WEIGHT_CORRELATION = config_mod.GA_WEIGHT_CORRELATION
GA_WEIGHT_DOUBLE_SORT = config_mod.GA_WEIGHT_DOUBLE_SORT
GA_WEIGHT_STABILITY = config_mod.GA_WEIGHT_STABILITY
GA_PASS_MIN_DIMENSIONS = config_mod.GA_PASS_MIN_DIMENSIONS


def score_factor(icir_result, decile_result, correlation_score, double_sort_result,
                 stability_result=None):
    """
    五维度评分 (含时序稳定性)
    
    Parameters
    ----------
    stability_result : dict or None
        时序稳定性综合评估结果 (不含时取0分, 向后兼容)
    
    Returns
    -------
    dict
        {
            'icir_score': float (0-1),
            'monotonicity_score': float (0-1),
            'correlation_score': float (0-1),
            'double_sort_score': float (0-1),
            'stability_score': float (0-1),     # ★ 新增
            'total_score': float (0-1),
            'pass_details': dict,
            'passed': bool,
        }
    """
    # 1. ICIR 维度
    icir = abs(icir_result.get('icir', 0))
    icir_score = np.clip(icir / ICIR_THRESHOLD, 0, 1)  # NSGA-II 校准用 1.5
    # 因子质量通过标准: |ICIR| >= 0.5 (行业共识), 不在NSGA-II校准时做额外折扣
    icir_passed = abs(icir) >= ICIR_QUALITY_THRESHOLD
    
    # 2. 单调性维度
    mono_result = decile_result.get('monotonicity_result', {})
    mono_pval = mono_result.get('p_value', 1.0)
    mono_score = 1.0 - np.clip(mono_pval / MONOTONICITY_P_THRESHOLD, 0, 1)
    if np.isnan(mono_score):
        mono_score = 0
    mono_passed = mono_score >= 0.7
    
    # 3. 低相关性维度
    corr_score = np.clip(1.0 - correlation_score / CORRELATION_THRESHOLD, 0, 1)
    corr_passed = corr_score >= 0.5
    
    # 4. 双重排序维度
    ds_pval = double_sort_result.get('p_value', 1.0)
    ds_score = 1.0 - np.clip(ds_pval / DOUBLE_SORT_P_THRESHOLD, 0, 1)
    if np.isnan(ds_score):
        ds_score = 0
    ds_passed = ds_score >= 0.7
    
    # 5. 时序稳定性 ★ 新增
    if stability_result is not None:
        stab_total = stability_result.get('total_stability_score', 0)
        stab_score = np.clip(stab_total / STABILITY_THRESHOLD, 0, 1)
        stab_passed = stab_score >= 0.7
    else:
        stab_score = 0.5  # 无数据时给中性分
        stab_passed = True  # 向后兼容: 不阻止
    
    # 加权总分
    total = (
        GA_WEIGHT_ICIR * icir_score +
        GA_WEIGHT_MONOTONICITY * mono_score +
        GA_WEIGHT_CORRELATION * corr_score +
        GA_WEIGHT_DOUBLE_SORT * ds_score +
        GA_WEIGHT_STABILITY * stab_score       # ★ 新增
    )
    
    # 5选3通过
    pass_count = sum([icir_passed, mono_passed, corr_passed, ds_passed, stab_passed])
    passed = pass_count >= GA_PASS_MIN_DIMENSIONS
    
    return {
        'icir_score': icir_score,
        'monotonicity_score': mono_score,
        'correlation_score': corr_score,
        'double_sort_score': ds_score,
        'stability_score': stab_score,          # ★ 新增
        'total_score': total,
        'pass_details': {
            'icir_passed': icir_passed,
            'mono_passed': mono_passed,
            'corr_passed': corr_passed,
            'ds_passed': ds_passed,
            'stab_passed': stab_passed,          # ★ 新增
            'pass_count': pass_count,
        },
        'passed': passed,
    }


def check_pass_criteria(score_result):
    """
    简洁的通过/不通过检查
    
    Returns
    -------
    tuple
        (passed: bool, pass_count: int, details: str)
    """
    details = score_result['pass_details']
    passed = score_result['passed']
    pass_count = details['pass_count']
    
    dims = []
    if details['icir_passed']: dims.append('ICIR')
    if details['mono_passed']: dims.append('单调性')
    if details['corr_passed']: dims.append('低相关性')
    if details['ds_passed']: dims.append('双重排序')
    if details.get('stab_passed', True): dims.append('稳定性')
    
    desc = f"通过 {pass_count}/5: [{', '.join(dims)}]" if passed else f"未通过 {pass_count}/5: [{', '.join(dims)}]"
    
    return passed, pass_count, desc


def evaluate_factor_comprehensive(factor_name, factor_df, forward_returns, mcap_df,
                                   icir_result, correlation_val, is_size_factor=False):
    """
    完整评估一个因子 (在GA适应度计算中调用)
    
    Parameters
    ----------
    factor_name : str
    factor_df : pd.DataFrame
    forward_returns : pd.DataFrame
    mcap_df : pd.DataFrame
    icir_result : dict
    correlation_val : float
    is_size_factor : bool
    
    Returns
    -------
    dict
    """
    from .decile_test import decile_portfolio_test, test_monotonicity
    from .double_sort import independent_double_sort, size_single_sort
    
    # 十分位检验
    decile_result = decile_portfolio_test(factor_df, forward_returns)
    mono_result = test_monotonicity(decile_result)
    
    # 独立双重排序 (规模因子用单变量)
    if is_size_factor:
        ds_result = size_single_sort(mcap_df, forward_returns)
    else:
        ds_result = independent_double_sort(factor_df, forward_returns, mcap_df)
    
    # 评分
    icir_result_wrapped = {'icir': icir_result.get('icir', 0)}
    decile_wrapped = {'monotonicity_result': mono_result}
    
    score_result = score_factor(
        icir_result_wrapped,
        decile_wrapped,
        correlation_val,
        ds_result,
    )
    
    return {
        'factor': factor_name,
        'icir': icir_result,
        'decile': decile_result,
        'monotonicity': mono_result,
        'double_sort': ds_result,
        'score': score_result,
        'correlation': correlation_val,
    }
