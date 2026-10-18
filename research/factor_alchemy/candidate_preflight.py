"""Generator-neutral, opt-in formula preflight.

The caller supplies the mode, so this adapter has no environment or storage
side effects. A pass is a formula check, never an economic or S5 verdict.
"""
from __future__ import annotations

import hashlib
from typing import Any

try:
    from formula_quality_precheck import assess_candidate
    from semantic_plan import tag_candidates
except ImportError:  # package import in focused unit tests
    from .formula_quality_precheck import assess_candidate
    from .semantic_plan import tag_candidates


def precheck_candidates(
    candidates: list[dict[str, Any]],
    *,
    source: str,
    mode: str = "off",
    semantic_plan: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return copied candidates and a count audit for any generation source.

    ``off`` is byte-for-byte behavior preserving at the candidate-field level.
    ``shadow`` attaches evidence only. ``enforce`` removes proven hard defects
    and repairs plain unqualified rank to cross-sectional rank. Review codes
    only downlock promotion in the existing validator; they do not reject.
    Pre-existing raw/canonical quality evidence is reused on a repeated gate.
    """
    if mode not in {"off", "shadow", "enforce"}:
        raise ValueError("preflight mode must be off, shadow or enforce")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source is required for preflight attribution")
    rows = [dict(item) for item in candidates]
    if mode == "off":
        return rows, {"source": source, "mode": mode, "input": len(rows),
                      "kept": len(rows), "rejected": 0, "review": 0}
    if semantic_plan is not None:
        rows = tag_candidates(rows, semantic_plan)
    kept: list[dict[str, Any]] = []
    rejected = reviewed = 0
    rejected_candidates: list[dict[str, Any]] = []
    for candidate in rows:
        current = candidate.get("formula") or candidate.get("expression")
        raw = candidate.get("raw_formula")
        prior = candidate.get("formula_quality") or {}
        reused = (bool(raw) and prior.get("canonical_formula") == current
                  and prior.get("raw_formula_sha256")
                  == hashlib.sha256(str(raw).encode("utf-8")).hexdigest())
        quality = dict(prior) if reused else assess_candidate(candidate)
        candidate["formula_quality"] = quality
        candidate["formula_quality_review_required"] = bool(quality["review_codes"])
        if quality["review_codes"]:
            reviewed += 1
        if mode == "enforce" and quality["hard_codes"]:
            rejected += 1
            rejected_candidates.append({
                "factor_name": str(candidate.get("factor_name") or candidate.get("name") or ""),
                "formula_sha256": hashlib.sha256(str(current).encode("utf-8")).hexdigest(),
                "hard_codes": list(quality["hard_codes"]),
            })
            continue
        if mode == "enforce" and quality["repairs"] and not reused:
            candidate["raw_formula"] = current
            candidate["formula"] = quality["canonical_formula"]
            candidate["expression"] = quality["canonical_formula"]
        kept.append(candidate)
    return kept, {"source": source, "mode": mode, "input": len(rows),
                  "kept": len(kept), "rejected": rejected, "review": reviewed,
                  "rejected_candidates": rejected_candidates}
