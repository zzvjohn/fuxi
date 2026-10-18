"""Pure adapters for the factor generation chain.

No memory, candidate, or promotion ledger is written here.  In particular, a
high-ranked SuccessPattern with no formula is not itself a GP parent.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List


_META = ("verification_level", "success_rate", "occurrence_count",
         "sample_factor_ids", "paradigm", "source")


def gp_template_dict(template: Any, attempts_by_name: Dict[str, str]) -> Dict:
    """Preserve parent quality and resolve a formula from recorded sample IDs."""
    get = template.get if isinstance(template, dict) else lambda k, d=None: getattr(template, k, d)
    formula = str(get("formula") or get("expression") or "").strip()
    sample_ids = get("sample_factor_ids") or []
    if not formula:
        formula = next((attempts_by_name.get(str(fid), "") for fid in sample_ids
                        if attempts_by_name.get(str(fid))), "")
    name = str(get("factor_name") or get("pattern_id") or get("name") or "")
    row = {"formula": formula, "factor_name": name}
    row.update({key: get(key) for key in _META if get(key) is not None})
    return row


def resolve_gp_parent_pool(retrieved: Iterable[Any], memory: Dict,
                           limit: int = 20) -> List[Dict]:
    """Fill an under-cap pool from memory without counting empty heads as parents.

    The ordered retrieved list retains priority.  Full memory is a fallback, not
    extra attempts: GP still receives the same requested child budget.
    """
    if limit < 2:
        raise ValueError("GP requires at least two parent slots")
    try:
        from factor_expression_tree import FactorExpressionParser, GPBreeder
    except ImportError:
        from .factor_expression_tree import FactorExpressionParser, GPBreeder
    parser = FactorExpressionParser()
    attempts = {str(a.get("factor_name")): str(a.get("formula") or a.get("expression") or "")
                for a in memory.get("attempts", []) if isinstance(a, dict)}
    out, seen = [], set()
    for raw in list(retrieved) + list(memory.get("success_templates", [])):
        row = gp_template_dict(raw, attempts)
        formula = row["formula"]
        if (formula and formula not in seen and parser.parse(formula) is not None
                and GPBreeder._validate_field_refs(formula)[0]):
            out.append(row)
            seen.add(formula)
        if len(out) >= limit:
            break
    return out


def s5_streak_event(results: Iterable[Any]) -> str:
    """Return pass, economic_failure, or unassessed for generator routing.

    Only candidates that reached S5 count. Missing prices, evaluator errors and
    precheck rejections must not be learned as an economic failure.
    """
    reached = [r for r in results if bool(getattr(r, "s4_passed", False))]
    if not reached or any(getattr(r, "s5_eval_status", "") != "valid" for r in reached):
        return "unassessed"
    return "pass" if any(bool(getattr(r, "s5_passed", False)) for r in reached) else "economic_failure"
