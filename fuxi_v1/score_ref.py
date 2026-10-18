"""Shared ScoreRef and content-addressed signal cache (P1, v2.0 landing).

R2 blueprint section 7: a signal cache key must contain at least the AST, the
operator version, the data snapshot, the universe, the actual evaluation window,
the point-in-time / adjustment / missing rules and the evaluation implementation.
Sub-expressions may share cache entries; formula names or file times must never
be used to judge reusability. This module lands that contract with a strict
content-addressed cache: identical keys return byte-identical values and the
compute path is never re-entered (waste reduction acceptance: "共同已计算项
逐值一致").
"""

from .contracts import ContractError, digest, keys, require

SCORE_REF_SCHEMA = "fuxi.score_ref.1"


def score_key(*, canonical_ast, operator_catalog_hash, snapshot_ref, universe_ref,
              eval_window, point_in_time_rule, adjustment_rule, missing_rule,
              eval_impl_ref):
    """Build + validate the nine-component signal cache key; returns (key_dict, score_id)."""
    key = {"schema_version": SCORE_REF_SCHEMA,
           "canonical_ast": canonical_ast,
           "operator_catalog_hash": operator_catalog_hash,
           "snapshot_ref": snapshot_ref,
           "universe_ref": universe_ref,
           "eval_window": eval_window,
           "point_in_time_rule": point_in_time_rule,
           "adjustment_rule": adjustment_rule,
           "missing_rule": missing_rule,
           "eval_impl_ref": eval_impl_ref}
    require(isinstance(canonical_ast, dict), "canonical_ast must be a dict")
    require(isinstance(operator_catalog_hash, str) and len(operator_catalog_hash) == 64,
            "operator_catalog_hash must be sha256 hex")
    for field in ("snapshot_ref", "universe_ref", "eval_impl_ref"):
        require(isinstance(key[field], str) and len(key[field]) == 64, field + " must be sha256 hex")
    require(isinstance(eval_window, list) and len(eval_window) == 2
            and all(isinstance(s, str) for s in eval_window), "eval_window must be [start, end]")
    for field in ("point_in_time_rule", "adjustment_rule", "missing_rule"):
        require(isinstance(key[field], str) and key[field], field + " required")
    digest(key)
    return key, digest(key)


def score_ref(*, candidate_id, score_id, key, eval_status, n_sections, support_daily,
              rank_ic_mean=None, icir_unannualized=None, coverage=None, stability=None,
              turnover_proxy=None, values_ref=None):
    """Build + validate a ScoreRef. Descriptive readings may be null (never 0-by-default)."""
    require(isinstance(support_daily, list), "support_daily must be a list of ints")
    for n in support_daily:
        require(type(n) is int and n >= 0, "daily support counts must be non-negative ints")
    ref = {"schema_version": SCORE_REF_SCHEMA, "candidate_id": candidate_id,
           "score_id": score_id, "key": key, "eval_status": eval_status,
           "n_sections": n_sections, "support_daily": support_daily,
           "rank_ic_mean": rank_ic_mean, "icir_unannualized": icir_unannualized,
           "coverage": coverage, "stability": stability, "turnover_proxy": turnover_proxy,
           "values_ref": values_ref}
    require(ref["eval_status"] in ("valid", "data_insufficient", "precheck_rejected",
                                   "error", "not_evaluated"), "invalid eval status")
    require(type(n_sections) is int and n_sections >= 0, "invalid n_sections")
    for metric in ("rank_ic_mean", "icir_unannualized", "coverage", "stability",
                   "turnover_proxy"):
        value = ref[metric]
        require(value is None or (type(value) in (int, float) and value == value),
                "invalid " + metric)
    require(isinstance(score_id, str) and len(score_id) == 64, "score_id must be sha256 hex")
    digest(ref)
    return ref


class ScoreCache:
    """Content-addressed score cache: identical key -> identical values, no recompute.

    Tracks ``computes`` (actual compute calls) and ``hits`` so the P1 waste
    acceptance ("共同已计算项逐值一致；省略项明确 not_evaluated") can be
    verified with counters rather than wall-clock flakiness.
    """

    def __init__(self):
        self._entries = {}
        self.computes = 0
        self.hits = 0

    def get(self, score_id):
        entry = self._entries.get(score_id)
        if entry is not None:
            self.hits += 1
            return entry
        return None

    def put(self, ref, computed=True):
        require(ref["score_id"] not in self._entries, "cache put must not overwrite")
        if computed:
            self.computes += 1
        self._entries[ref["score_id"]] = ref
        return ref

    def get_or_compute(self, key_dict, compute):
        """``compute(key_dict) -> ScoreRef`` is invoked at most once per content key."""
        _, score_id = score_key(**key_dict)
        cached = self.get(score_id)
        if cached is not None:
            return cached
        ref = compute(key_dict)
        require(ref["score_id"] == score_id, "computed score_id does not match the cache key")
        return self.put(ref)

    def __len__(self):
        return len(self._entries)
