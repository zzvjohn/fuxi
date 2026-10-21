"""types_core —— 伏羲 v2.0 量纲/语法类型系统核心（纯 stdlib，跨 fuxi_v1 / factor_mining 共享）。

模块：
- spec.py    TypeSpec 数据结构 + 类型格运算（量纲指数代数 + 形态/族/档合并）
- registry.py 叶子注册表（字段级 + 因子级）+ 名称模式推断 + resolve 三级解析
- rules.py   算子规则表（hard/soft 三档 + 违规码）
- infer.py   类型推断双前端（AST 递归 + preorder 前缀栈式）

用法：
    from fuxi_v1.types_core import TypeSpec, resolve, infer_ast, check_tree

    leaf = resolve("ep", kind="factor")
    spec, violations = infer_ast(ast, lambda name: resolve(name, kind="field"))
    hard = [v for v in violations if v.severity == "hard"]
"""

from .spec import (
    TypeSpec, FORMS, FAMILIES, HORIZONS, SHAPES, DIRECTIONS,
    dim_add, dim_sub, dim_mul, dim_div, dim_square, is_dimensionless,
    merge_shape, merge_family, merge_horizon,
)
from .registry import (
    REGISTRY_VERSION, FIELD_REGISTRY, FACTOR_REGISTRY, INFER_PATTERNS,
    resolve, factor_names, field_names, validate, registry_digest,
    load_overrides, overrides,
)
from .rules import (
    Violation,
    V_DIM_MISMATCH, V_FORM_MISMATCH, V_DENOM_SIGNED, V_SELF_DIV,
    V_LOG_DIM, V_LOG_SHAPE, V_SCORE_ARITH, V_FAMILY_MIXED,
    V_HORIZON_SKIP, V_DENOM_RETURN, V_TEMPORAL_NESTED,
    BINARY_ARITH, TEMPORAL, ALL_OPERATORS, render_for_prompt,
)
from .infer import (
    infer_ast, infer_prefix, check_tree, hard_violations, soft_violations,
    window_horizon, TypeState, mask_actions,
)

__all__ = [
    "TypeSpec", "FORMS", "FAMILIES", "HORIZONS", "SHAPES", "DIRECTIONS",
    "dim_add", "dim_sub", "dim_mul", "dim_div", "dim_square", "is_dimensionless",
    "merge_shape", "merge_family", "merge_horizon",
    "REGISTRY_VERSION", "FIELD_REGISTRY", "FACTOR_REGISTRY", "INFER_PATTERNS",
    "resolve", "factor_names", "field_names", "validate", "registry_digest",
    "load_overrides", "overrides",
    "Violation",
    "V_DIM_MISMATCH", "V_FORM_MISMATCH", "V_DENOM_SIGNED", "V_SELF_DIV",
    "V_LOG_DIM", "V_LOG_SHAPE", "V_SCORE_ARITH", "V_FAMILY_MIXED",
    "V_HORIZON_SKIP", "V_DENOM_RETURN", "V_TEMPORAL_NESTED",
    "BINARY_ARITH", "TEMPORAL", "ALL_OPERATORS", "render_for_prompt",
    "infer_ast", "infer_prefix", "check_tree", "hard_violations",
    "soft_violations", "window_horizon", "TypeState", "mask_actions",
]
