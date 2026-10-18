"""Truthful, candidate-level feedback for the Fuxi research selector.

Proxy IC is a diagnostic, never an economic reward. Historical Memory and MAB
entries are left untouched; consumers can replay the typed observations later.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


STATUS = frozenset({"not_evaluated", "data_insufficient", "error", "precheck_rejected", "valid"})


def _status(value: Any) -> str:
    text = str(value or "not_evaluated").strip().lower()
    return text if text in STATUS else "error"


def candidate_feedback(candidate: Mapping[str, Any], result: Any) -> dict:
    """Describe one evaluation without promoting a missing reading to failure.

    The actual candidate source wins over the selected round generator. A
    successful S1 reading is a proxy observation only. An S5 ``valid`` flag
    still requires the external trusted-ledger provenance before reward use.
    """
    source = str(candidate.get("_source") or candidate.get("source") or "unknown")
    generator = {
        "factorforge": "forge", "forge": "forge", "llm": "llm",
        "gp_breed": "gp_breed", "gp_evolve": "gp_evolve",
    }.get(source, source)
    s1 = _status(getattr(result, "s1_eval_status", None))
    s4 = _status(getattr(result, "s4_eval_status", None))
    s5 = _status(getattr(result, "s5_eval_status", None))
    s6 = _status(getattr(result, "s6_eval_status", None))
    icir = getattr(result, "s1_icir", None)
    try:
        proxy = float(icir) if s1 == "valid" and math.isfinite(float(icir)) else None
    except (TypeError, ValueError):
        proxy = None
    return {
        "research_run_id": str(candidate.get("_research_run_id") or ""),
        "research_decision_id": str(candidate.get("_research_decision_id") or ""),
        "candidate_id": str(candidate.get("candidate_id") or candidate.get("factor_id") or ""),
        "research_action_id": str(candidate.get("research_action_id") or ""),
        "factor_name": str(candidate.get("factor_name") or candidate.get("name") or ""),
        "selected_generator": str(candidate.get("_selected_generator") or ""),
        "actual_generator": generator,
        "candidate_source": source,
        "semantic_plan_id": str(candidate.get("semantic_plan_id") or ""),
        "statuses": {"s1": s1, "s4": s4, "s5": s5, "s6": s6},
        "proxy_icir": proxy,
        "economic_reward": None,
        "economic_reward_status": "not_evaluated",
        "economic_reward_reason": "trusted trade-ledger evidence not attached",
    }
