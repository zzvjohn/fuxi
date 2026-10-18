"""
因子评估模块初始化
"""
from .ic_analysis import compute_ic_icir
from .decile_test import decile_portfolio_test, test_monotonicity
from .double_sort import independent_double_sort, size_single_sort
from .correlation import factor_correlation_matrix, check_multicollinearity
from .scoring import score_factor, check_pass_criteria
from .stability import (                         # ★ 新增
    compute_rank_autocorrelation,
    compute_top_quantile_retention,
    compute_ic_stability,
    compute_ic_decay,
    compute_mean_turnover,
    compute_stability_comprehensive,
    print_stability_report,
)
