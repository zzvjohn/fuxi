"""Conservative, data-free precheck for LLM factor expressions.

Only mechanically provable defects reject a candidate.  Natural-language
mechanism checks request review; they never certify economic usefulness.
"""
from __future__ import annotations

import ast
import hashlib
import re


def _rank_axis(formula: str):
    tree = ast.parse(formula, mode="eval")
    changes, errors = [], []

    class Normalize(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "rank":
                return node
            receiver = node.func.value
            if (isinstance(receiver, ast.Call)
                    and isinstance(receiver.func, ast.Attribute)
                    and receiver.func.attr in ("rolling", "expanding", "groupby")):
                return node
            axes = [node.args[0]] if node.args else [k.value for k in node.keywords if k.arg == "axis"]
            if axes:
                axis = axes[0]
                if not isinstance(axis, ast.Constant) or axis.value not in (1, "columns"):
                    errors.append("rank_time_axis")
            else:
                node.keywords.append(ast.keyword(arg="axis", value=ast.Constant(1)))
                changes.append("rank_axis_explicit_cross_section")
            return node

    tree = Normalize().visit(tree)
    return (ast.unparse(ast.fix_missing_locations(tree)) if changes else formula,
            sorted(set(changes)), sorted(set(errors)))


def _algebraic_issues(formula: str, hypothesis: str):
    """Treat non-arithmetic pandas calls as repeated opaque symbols.

    This proves cancellation without claiming to understand rolling/rank
    semantics.  Unsupported syntax returns an explicit review observation.
    """
    try:
        import sympy as sp
    except ImportError:
        return [], ["symbolic_checker_unavailable"]
    tree = ast.parse(formula, mode="eval")
    if len(list(ast.walk(tree))) > 180:
        return [], ["symbolic_budget_exceeded"]
    eps = sp.Symbol("__epsilon__")

    def convert(node):
        if isinstance(node, ast.Name):
            return sp.Symbol(node.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return eps if float(node.value) == 1e-8 else sp.Rational(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -convert(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return convert(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = convert(node.left), convert(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
                return a ** b
        if isinstance(node, (ast.Call, ast.Attribute)):
            return sp.Symbol(ast.unparse(node))
        raise ValueError("unsupported symbolic node")

    try:
        outer = tree.body
        if (isinstance(outer, ast.Call) and isinstance(outer.func, ast.Attribute)
                and outer.func.attr == "rank"):
            outer = outer.func.value
        expr = sp.cancel(convert(outer))
        hard = []
        if expr == 0:
            hard.append("algebraic_zero")
        elif eps in expr.free_symbols and sp.cancel(expr.subs(eps, 0)) == 0:
            hard.append("epsilon_dominated_cancellation")
        review = []
        if "正交" in hypothesis:
            close, open_ = sp.Symbol("close"), sp.Symbol("open")
            prior = sp.Symbol("close.shift(1)")
            intraday = (close - open_) / (prior + eps)
            if sp.cancel(expr - intraday) == 0:
                review.append("claimed_orthogonality_reduces_to_intraday_return")
        return hard, review
    except (ValueError, TypeError, ZeroDivisionError):
        return [], ["symbolic_check_inconclusive"]


def _denominator_issues(formula: str) -> list[str]:
    """Request domain review for uncertain denominators, never infer a defect.

    A valid A-share price is strictly positive. Missing observations are dealt
    with by the existing numeric contract; an absent epsilon is not a proof of
    division by zero. The narrow syntactic check here avoids discarding valid
    open/previous-close ratios before evaluation.
    """
    tree = ast.parse(formula, mode="eval")
    price_fields = {"open", "high", "low", "close", "pre_close"}

    def positive_price(node):
        if isinstance(node, ast.Name):
            return node.id in price_fields
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "shift" and isinstance(node.func.value, ast.Name)
                and node.func.value.id in price_fields
                and (not node.args or (isinstance(node.args[0], ast.Constant)
                                       and isinstance(node.args[0].value, int)
                                       and node.args[0].value >= 0)))

    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        denominator = node.right
        if positive_price(denominator):
            continue
        rendered = ast.unparse(denominator)
        if re.search(r"1e-0?8|\.clip\s*\(|\.replace\s*\(", rendered, re.I):
            continue
        return ["denominator_domain_unverified"]
    return []


def _causality_issues(formula: str) -> tuple[list[str], list[str]]:
    """Identify literal future access; leave dynamic periods for human review.

    The expression grammar permits pandas method calls. A negative *literal*
    period or an explicit backward fill is mechanically non-causal. Indexing
    and dynamic period expressions need the execution-level PIT probe instead.
    """
    hard, review = [], []
    lag_methods = {"shift", "diff", "pct_change"}

    def literal_negative(node: ast.AST) -> bool:
        return (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, (int, float))
                and not isinstance(node.operand.value, bool)
                and node.operand.value > 0)

    for node in ast.walk(ast.parse(formula, mode="eval")):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        keywords = {item.arg: item.value for item in node.keywords if item.arg}
        if method in lag_methods:
            period = node.args[0] if node.args else keywords.get("periods")
            if period is not None and literal_negative(period):
                hard.append("negative_lag")
            elif period is not None and not isinstance(period, ast.Constant):
                review.append("lag_direction_unverified")
        if method in {"bfill", "backfill"}:
            hard.append("future_filling_or_centered_window")
        if method == "rolling" and isinstance(keywords.get("center"), ast.Constant):
            if keywords["center"].value is True:
                hard.append("future_filling_or_centered_window")
        if method == "fillna" and (node.args or "method" in keywords):
            fill_method = node.args[0] if node.args else keywords.get("method")
            if isinstance(fill_method, ast.Constant) and fill_method.value in {"bfill", "backfill"}:
                hard.append("future_filling_or_centered_window")
    return sorted(set(hard)), sorted(set(review))


def assess_candidate(candidate: dict) -> dict:
    formula = str(candidate.get("formula") or candidate.get("expression") or "").strip()
    hypothesis = " ".join(str(candidate.get(k) or "")
                          for k in ("factor_name", "hypothesis", "logic"))
    raw_sha = hashlib.sha256(formula.encode("utf-8")).hexdigest()
    if not formula:
        return {"status": "reject", "hard_codes": ["empty_formula"],
                "review_codes": [], "repairs": [], "raw_formula_sha256": raw_sha,
                "canonical_formula": formula}
    try:
        canonical, repairs, hard = _rank_axis(formula)
    except SyntaxError:
        return {"status": "reject", "hard_codes": ["invalid_syntax"],
                "review_codes": [], "repairs": [], "raw_formula_sha256": raw_sha,
                "canonical_formula": formula}
    causal_hard, causal_review = _causality_issues(canonical)
    hard.extend(causal_hard)
    algebra_hard, review = _algebraic_issues(canonical, hypothesis)
    hard.extend(algebra_hard)
    review.extend(causal_review)
    review.extend(_denominator_issues(canonical))
    if "日内" in hypothesis and ("动量" in hypothesis or "波动" in hypothesis):
        if "close.pct_change" in formula and not re.search(r"close\s*-\s*open", formula):
            review.append("intraday_claim_uses_close_to_close_change")
    if str(candidate.get("direction") or "").lower() == "short":
        review.append("short_direction_not_executable_by_long_only_s5")
    hard, review = sorted(set(hard)), sorted(set(review))
    return {"status": "reject" if hard else "review" if review else "pass",
            "hard_codes": hard, "review_codes": review, "repairs": repairs,
            "raw_formula_sha256": raw_sha, "canonical_formula": canonical}
