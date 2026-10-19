"""Deterministic factor-family label for reporting (2026-10-19).

The daily report must name each JQ candidate's formula and family. The project
already stores a ``structural_family`` hash, which is not human-readable, and the
archetype mapper falls back to ``fundamental`` for combined price hypotheses.
This module derives a compact, auditable label from the AST itself: the source
fields and the operator set. It is descriptive only and never gates a decision.
"""

FAMILY_ZH = {
    "reversal_volatility_composite": "反转×波动率复合",
    "volatility_cv_composite": "波动率·多窗变异系数复合",
    "volatility_cv": "波动率·变异系数反转",
    "reversal": "反转",
    "momentum": "动量",
    "volume_price": "量价",
    "price_level": "价格水平",
    "price_only": "价格派生",
    "unknown": "未分类",
}

FIELD_ZH = {"close": "收盘", "high": "最高", "low": "最低", "open": "开盘",
            "volume": "成交量", "amount": "成交额", "vwap": "均价"}


def _walk(node, fields, ops):
    if not isinstance(node, dict):
        return
    if "field" in node:
        fields.add(node["field"])
    if "op" in node:
        ops.add(node["op"])
    for child in node.get("args") or ():
        _walk(child, fields, ops)


def classify(ast):
    fields, ops = set(), set()
    _walk(ast, fields, ops)
    has_std, has_mean, has_return = "std" in ops, "mean" in ops, "return" in ops
    volume_like = bool(fields & {"volume", "amount", "vwap"})
    if volume_like:
        family = "volume_price"
    elif has_return and (has_std or has_mean):
        family = "reversal_volatility_composite"
    elif has_std and has_mean:
        family = "volatility_cv_composite" if "mul" in ops or "add" in ops else "volatility_cv"
    elif has_return:
        family = "reversal"
    elif has_std:
        family = "volatility_cv"
    elif fields & {"close", "high", "low", "open"}:
        family = "price_level"
    else:
        family = "unknown"
    return {
        "family": family,
        "family_zh": FAMILY_ZH.get(family, family),
        "fields": sorted(fields),
        "fields_zh": [FIELD_ZH.get(f, f) for f in sorted(fields)],
        "operators": sorted(ops),
    }
