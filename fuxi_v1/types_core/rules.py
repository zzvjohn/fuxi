"""types_core.rules —— 算子规则表（hard / soft 三档 + 违规码）。

表驱动：加算子 = 本文件加一个条目 + 一条测试，不改 infer 内核。

硬约束（hard）只禁"任何经济叙事下都无意义的组合"（可证伪的类型错误）；
软约束（soft）管"可疑"（跨 family 相加、horizon 跨档、序数参与算术），
只产出诊断违规码，不拒绝 —— 留给 reward 面与 E 层裁决。

div 自除禁令在此类型化为 V_SELF_DIV（既有教训：div(X,X) 恒等因子 ic_std≈0
穿透 ICIR 门，见 qlib_daily 台账比值型指标陷阱铁律）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .spec import (
    TypeSpec, dim_div, dim_mul, dim_neg, dim_square, is_dimensionless,
    merge_family, merge_horizon, merge_shape,
)

# ---------------------------------------------------------------------------
# 违规码（供日志 / 统计 / 回归）
# ---------------------------------------------------------------------------

V_DIM_MISMATCH = "V_DIM_MISMATCH"        # add/sub 两侧量纲不同
V_FORM_MISMATCH = "V_FORM_MISMATCH"      # add/sub 两侧测度形态不同
V_DENOM_SIGNED = "V_DENOM_SIGNED"        # div 分母可穿零（shape=signed）
V_SELF_DIV = "V_SELF_DIV"                # div 自除（分子分母同一子表达式）
V_LOG_DIM = "V_LOG_DIM"                  # log 输入非无量纲
V_LOG_SHAPE = "V_LOG_SHAPE"              # log 输入非正（shape != positive）
V_SCORE_ARITH = "V_SCORE_ARITH"          # 序数分参与乘/加（序数不可运算）
V_FAMILY_MIXED = "V_FAMILY_MIXED"        # add/sub 跨 family（软）
V_HORIZON_SKIP = "V_HORIZON_SKIP"        # add/sub horizon 跨档（软）
V_DENOM_RETURN = "V_DENOM_RETURN"        # div 分母是 return（软：收益比收益不稳定）
V_TEMPORAL_NESTED = "V_TEMPORAL_NESTED"  # 时序算子输入非字段叶子


@dataclass(frozen=True)
class Violation:
    code: str
    op: str
    detail: str
    severity: str = "hard"   # hard | soft


# ---------------------------------------------------------------------------
# 算子分类
# ---------------------------------------------------------------------------

BINARY_ARITH = ("add", "sub", "mul", "div")
UNARY_NORMALIZER = ("rank", "zscore")       # 类型归一器：输出无条件标准化
UNARY_PRESERVING = ("neg", "abs")           # 保量纲方向/符号变换
UNARY_SHAPING = ("square", "log")           # 量纲整形
TEMPORAL = ("lag", "return", "mean", "std", "min", "max", "delta")
ALL_OPERATORS = BINARY_ARITH + UNARY_NORMALIZER + UNARY_PRESERVING + UNARY_SHAPING + ("ts_zscore",) + TEMPORAL


def _horizon_gap(a: str, b: str) -> int:
    """horizon 档位差（跨 2 档 = soft 扣分）。none 不算档。"""
    order = {"fast": 0, "medium": 1, "slow": 2}
    if a == "none" or b == "none":
        return 0
    return abs(order[a] - order[b])


# ---------------------------------------------------------------------------
# 检查函数：返回 Violation 列表（空 = 通过）
# ---------------------------------------------------------------------------

def _check_add_sub(op, left, right):
    """add/sub：两侧 form 相同（硬）；dim 相同（硬，沿用 dsl unit 纪律）。
    跨 family、horizon 跨 2 档为软。"""
    out = []
    if left.dim != right.dim:
        out.append(Violation(V_DIM_MISMATCH, op,
                              f"dim {left.dim} vs {right.dim}", "hard"))
    if left.form != right.form:
        out.append(Violation(V_FORM_MISMATCH, op,
                              f"form {left.form} vs {right.form}", "hard"))
    if left.family != right.family and "none" not in (left.family, right.family):
        out.append(Violation(V_FAMILY_MIXED, op,
                              f"family {left.family} vs {right.family}", "soft"))
    if _horizon_gap(left.horizon, right.horizon) >= 2:
        out.append(Violation(V_HORIZON_SKIP, op,
                              f"horizon {left.horizon} vs {right.horizon}", "soft"))
    return out


def _check_mul(op, left, right):
    """mul：无硬约束。序数分参与乘法为软（序数不可乘）。"""
    out = []
    if left.form == "score" and right.form == "score":
        out.append(Violation(V_SCORE_ARITH, op, "score x score", "soft"))
    return out


def _check_div(op, left, right):
    """div：分母穿零风险（硬）；自除（硬）；分母是 return（软）。"""
    out = []
    if right.shape == "signed":
        out.append(Violation(V_DENOM_SIGNED, op, "denominator shape=signed", "hard"))
    if left is right:  # 同一对象（同一叶子 / 同一子树实例）
        out.append(Violation(V_SELF_DIV, op, "self division", "hard"))
    if right.form == "return":
        out.append(Violation(V_DENOM_RETURN, op, "denominator form=return", "soft"))
    return out


def _check_log(op, x):
    out = []
    if not is_dimensionless(x.dim):
        out.append(Violation(V_LOG_DIM, op, f"dim {x.dim} not dimensionless", "hard"))
    if x.shape != "positive":
        out.append(Violation(V_LOG_SHAPE, op, f"shape {x.shape} not positive", "hard"))
    return out


def _check_score_arith(op, left, right):
    """add/sub 里序数分参与为软（序数与连续形态相加无叙事）。"""
    out = []
    if left.form == "score" or right.form == "score":
        out.append(Violation(V_SCORE_ARITH, op,
                              f"score arithmetic {left.form}+{right.form}", "soft"))
    return out


def _check_temporal(op, arg_form, arg_is_leaf):
    out = []
    if not arg_is_leaf:
        out.append(Violation(V_TEMPORAL_NESTED, op, "temporal input not a field leaf", "hard"))
    return out


# ---------------------------------------------------------------------------
# 输出推导：给定输入 TypeSpec，返回输出 TypeSpec
# ---------------------------------------------------------------------------

def output_add_sub(op, left, right):
    return TypeSpec(
        dim=left.dim,                 # dim 已要求相同
        form=left.form,               # form 已要求相同
        family=merge_family(left.family, right.family),
        horizon=merge_horizon(left.horizon, right.horizon),
        shape=merge_shape(left.shape, right.shape),
        direction=left.direction if left.direction == right.direction else 0,
    )


def output_mul(op, left, right):
    return TypeSpec(
        dim=dim_mul(left.dim, right.dim),
        form="ratio",                 # 交互项默认比率形态
        family=merge_family(left.family, right.family),
        horizon=merge_horizon(left.horizon, right.horizon),
        shape="positive" if left.shape == right.shape == "positive" else "signed",
        direction=left.direction * right.direction if left.direction and right.direction else 0,
    )


def output_div(op, left, right):
    return TypeSpec(
        dim=dim_div(left.dim, right.dim),
        form="ratio",
        family=merge_family(left.family, right.family),
        horizon=merge_horizon(left.horizon, right.horizon),
        shape=left.shape,             # 分母恒正（hard 已约束），符号由分子决定
        direction=left.direction * right.direction if left.direction and right.direction else 0,
    )


def output_neg(op, x):
    return x.with_(direction=-x.direction)


def output_abs(op, x):
    return x.with_(shape="positive", direction=0)


def output_square(op, x):
    return x.with_(dim=dim_square(x.dim), shape="positive", direction=0)


def output_log(op, x):
    return TypeSpec(dim=(0, 0), form="ratio", family=x.family,
                    horizon=x.horizon, shape="signed", direction=x.direction)


def output_rank(op, x):
    return TypeSpec(dim=(0, 0), form="ranked", family=x.family,
                    horizon=x.horizon, shape="bounded", direction=x.direction)


def output_zscore(op, x):
    return TypeSpec(dim=(0, 0), form="ranked", family=x.family,
                    horizon=x.horizon, shape="signed", direction=x.direction)


def output_ts_zscore(op, x):
    return TypeSpec(dim=(0, 0), form=x.form, family=x.family,
                    horizon=x.horizon, shape="signed", direction=x.direction)


def output_temporal_return(op, x):
    return TypeSpec(dim=(0, 0), form="return", family="momentum",
                    horizon=x.horizon, shape="signed", direction=x.direction)


def output_temporal_std(op, x):
    return TypeSpec(dim=x.dim, form="dispersion", family="volatility",
                    horizon=x.horizon, shape="positive", direction=x.direction)


def output_temporal_identity(op, x):
    """lag/mean/min/max/delta：量纲与形态继承叶子。"""
    return x


# ---------------------------------------------------------------------------
# 规则表：op -> (check, output)
# ---------------------------------------------------------------------------

_RULES = {
    "add": (_check_add_sub, output_add_sub),
    "sub": (_check_add_sub, output_add_sub),
    "mul": (_check_mul, output_mul),
    "div": (_check_div, output_div),
    "neg": (None, output_neg),
    "abs": (None, output_abs),
    "square": (None, output_square),
    "log": (_check_log, output_log),
    "rank": (None, output_rank),
    "zscore": (None, output_zscore),
    "ts_zscore": (None, output_ts_zscore),
    "lag": (_check_temporal, output_temporal_identity),
    "return": (_check_temporal, output_temporal_return),
    "mean": (_check_temporal, output_temporal_identity),
    "std": (_check_temporal, output_temporal_std),
    "min": (_check_temporal, output_temporal_identity),
    "max": (_check_temporal, output_temporal_identity),
    "delta": (_check_temporal, output_temporal_identity),
}


def rule_for(op):
    return _RULES.get(op)


def render_for_prompt(registry=None):
    """把当前规则表渲染成 LLM prompt 用的自然语言约束（治理 C3 手工副本）。

    registry 参数保留以兼容方案签名；当前渲染不依赖注册表内容。
    """
    lines = [
        "Type rules (hard = rejected at compile time, soft = penalized/diagnostic only):",
        "- add/sub: both sides must have identical unit dimension AND identical form;",
        "  crossing family or skipping horizon tiers is soft-flagged, not rejected.",
        "- mul: no hard constraint (cross-family interaction is allowed); score x score is soft-flagged.",
        "- div: denominator must NOT be signed (sign-changing denominators risk division by zero);",
        "  self-division (same subtree on both sides) is hard-rejected; return-form denominator is soft-flagged.",
        "- log: input must be dimensionless AND strictly positive.",
        "- rank/zscore/ts_zscore: normalizers — output is dimensionless/standardized, always allowed.",
        "- neg: flips direction, preserves everything else; abs/sign: output is non-negative.",
        "- time operators (lag/return/mean/std/min/max/delta): accept a single field leaf only, never nested.",
    ]
    return "\n".join(lines)
