"""Point-in-time cohort construction and regeneration from a declared rule.

``history.select_cohort`` already draws its sample from a hash of ``ts_code``
alone, over a frame that *retains* names which delist inside the window, so the
sample is invariant to delisting and to any return outcome. That property is
worth nothing while it is only a docstring: nothing in the kernel re-derived the
cohort, so a later edit could delete members and the deletion was invisible
except as a hand-written ``bias`` string.

This module makes the property mechanical. ``reproduce_cohort`` rebuilds the
cohort from the lifecycle table and the declared rule; certification then
requires the rebuild to equal the frozen member set exactly. Any retrospective
deletion shows up as a non-empty symmetric difference and refuses the universe
flag with a named gap instead of a narrative.
"""
import hashlib
import re

from .contracts import digest, require

MAINBOARD = re.compile(r"(?:00\d{4}\.SZ|60\d{4}\.SH)")
EXCHANGES = ("SH", "SZ")

# The sampling frame, written once and reused. A caller that declares a frame
# predicate is checked against this exact string, so a batch cannot describe its
# population as something wider than the predicate the code actually applies.
FRAME_PREDICATE = "mainboard and listed<=window_end and not delisted<=window_start"
ALL_A_FRAME_PREDICATE = "common_A_share_SH_SZ_BJ and listed<=window_end and not delisted<=window_start"

# One-sided alpha for the survivor test. A cohort that draws no delisting name
# while the population rate predicts several is refused; the threshold is a
# declared statistical convention, not a judgement call.
SURVIVOR_ALPHA = 0.01


def canonical_code(code):
    """Normalise a security identifier to the vendor form ``NNNNNN.SZ`` / ``NNNNNN.SH``.

    Historical snapshots store JoinQuant-style suffixes while the vendor security
    master uses Tushare-style ones. Comparing them without normalising silently
    makes every lifecycle lookup miss, which would report a lifecycle violation
    on every member cell. Normalising here keeps that failure impossible.
    """
    text = (code or "").strip().upper()
    if text.endswith(".XSHE"):
        return text[:-5] + ".SZ"
    if text.endswith(".XSHG"):
        return text[:-5] + ".SH"
    if text.endswith(".SS"):
        return text[:-3] + ".SH"
    return text


def jq_code(code):
    """Inverse of :func:`canonical_code`, for values written into a snapshot."""
    text = canonical_code(code)
    if text.endswith(".SZ"):
        return text[:-3] + ".XSHE"
    if text.endswith(".SH"):
        return text[:-3] + ".XSHG"
    return text


def lifecycle_from_rows(rows):
    """Build ``code -> {list_date, delist_date, list_status}`` from security-master rows."""
    life = {}
    for row in rows:
        code = canonical_code(row.get("ts_code"))
        if not code:
            continue
        life[code] = {"list_date": (row.get("list_date") or "").strip() or None,
                      "delist_date": (row.get("delist_date") or "").strip() or None,
                      "list_status": (row.get("list_status") or "").strip() or None}
    return life


def lifecycle_from_plan(plan):
    """Lifecycle table recorded in a frozen plan, if one was embedded."""
    embedded = (plan or {}).get("lifecycle")
    if embedded:
        return dict(embedded)
    return None


def read_lifecycle(raw_root, name="delist_dates.csv"):
    """Read the vendor security master from the raw store (read-only)."""
    import pandas as pd
    from pathlib import Path
    frame = pd.read_csv(Path(raw_root) / name, dtype=str).astype(object)
    return lifecycle_from_rows(frame.where(frame.notna(), None).to_dict("records"))


def read_lifecycle_file(path):
    """Read a security master from an explicit file path."""
    import pandas as pd
    from pathlib import Path
    frame = pd.read_csv(Path(path), dtype=str).astype(object)
    return lifecycle_from_rows(frame.where(frame.notna(), None).to_dict("records"))


def _iso(value):
    if not value:
        return None
    text = str(value).strip().replace("-", "")
    return text if len(text) == 8 and text.isdigit() else None


def score(salt, code):
    """The selection score: a function of the identifier and the salt only."""
    return hashlib.sha256((salt + ":" + code).encode()).hexdigest()


def frame_rows(life, start, end, *, all_a=False):
    """The sampling frame: mainboard names that existed at any point in the window.

    The predicate uses only *lifecycle* facts, and it is one-sided: a name is kept
    when it was listed at or before the window end and had not already delisted
    before the window start. Names that delist inside the window are retained.
    """
    start, end = _iso(start), _iso(end)
    rows = []
    for code, row in life.items():
        listed, delisted = _iso(row.get("list_date")), _iso(row.get("delist_date"))
        from .a_share import board
        in_scope = board(code) is not None if all_a else bool(MAINBOARD.fullmatch(code))
        if not in_scope or not listed or listed > end:
            continue
        if delisted and delisted <= start:
            continue
        rows.append({"ts_code": code, "list_date": listed, "delist_date": delisted,
                     "list_status": row.get("list_status")})
    return sorted(rows, key=lambda r: r["ts_code"])


def per_exchange_hash_cutoff(rows, start, salt, per_exchange):
    """Frozen-cutoff per-exchange sampling: initial names set thresholds, later IPOs use them.

    This is the rule ``history.select_cohort`` implements. The cutoffs are derived
    from the population that existed at the window start, so a name that IPOs
    later can only enter by clearing a threshold fixed before it existed.

    The threshold is taken from a total order ``(score, ts_code)``. A 256-bit hash
    has no collisions to speak of, so this cannot change which names are drawn;
    it does make the result provably independent of the order rows arrive in,
    which is what :func:`reproduce_cohort` then asserts.
    """
    start = _iso(start)
    picked, cuts, detail = [], {}, {}
    for exchange in EXCHANGES:
        pool = [r for r in rows if r["ts_code"].endswith(exchange)]
        initial = sorted([r for r in pool if r["list_date"] <= start],
                         key=lambda r: (score(salt, r["ts_code"]), r["ts_code"]))
        if len(initial) < per_exchange:
            return None, {"blocking_gaps": ["insufficient_frame_for_%s_exchange" % exchange]}
        cut = score(salt, initial[per_exchange - 1]["ts_code"])
        cuts[exchange] = cut
        kept = [r for r in pool if score(salt, r["ts_code"]) <= cut]
        detail[exchange] = {"frame": len(pool), "initial_frame": len(initial),
                            "cutoff": cut, "kept": len(kept)}
        picked.extend(kept)
    return picked, {"rule_kind": "per_exchange_hash_cutoff", "hash_cutoffs": cuts,
                    "per_exchange": detail}


def full_population_hash_sample(rows, salt, target_size):
    """Whole-population hash sample: no exchange split, no cap on delisted names."""
    ordered = sorted(rows, key=lambda r: (score(salt, r["ts_code"]), r["ts_code"]))
    if len(ordered) < target_size:
        return None, {"blocking_gaps": ["frame_smaller_than_target_size"]}
    picked = ordered[:target_size]
    return picked, {"rule_kind": "full_population_hash_sample", "frame": len(ordered),
                    "target_size": target_size,
                    "marginal_cutoff": score(salt, picked[-1]["ts_code"])}


def _pick(rows, kind, declaration, start, salt):
    """Evaluate the declared rule against one ordering of the frame."""
    if kind == "per_exchange_hash_cutoff":
        return per_exchange_hash_cutoff(rows, start, salt,
                                        int(declaration.get("per_exchange_initial") or 48))
    return full_population_hash_sample(rows, salt, int(declaration.get("target_size") or 0))


def reproduce_cohort(life, declaration, sessions):
    """Rebuild the frozen cohort from the declared rule alone.

    Returns ``(codes, basis)``. ``codes`` is ``None`` when the rule cannot be
    evaluated, in which case ``basis['blocking_gaps']`` explains why.

    The result is also re-derived from a ``reversed`` frame. Reproduction alone
    would pass for a rule that was secretly positional ("the first 103 rows"), so
    invariance under a deterministic reordering is what actually demonstrates the
    selection is a function of ``(salt, identifier)`` and nothing else.
    """
    if not sessions:
        return None, {"blocking_gaps": ["no_sessions_to_reproduce_over"]}
    kind = (declaration or {}).get("rule_kind")
    salt = (declaration or {}).get("salt")
    if not salt:
        return None, {"blocking_gaps": ["sampling_salt_undeclared"]}
    rows = frame_rows(life, sessions[0], sessions[-1], all_a=kind=='all_a_share_census')
    if not rows:
        return None, {"blocking_gaps": ["empty_sampling_frame"]}
    if kind == 'all_a_share_census':
        codes = sorted(r['ts_code'] for r in rows)
        return codes, {'rule_kind':kind,'frame_size':len(rows),'sampled':len(codes),
                       'sampling_frame_is_full_lifecycle_table':True,'permutation_invariant':True,
                       'sampling_keys':['ts_code'],'frame_predicate':ALL_A_FRAME_PREDICATE}
    if kind not in ("per_exchange_hash_cutoff", "full_population_hash_sample"):
        return None, {"blocking_gaps": ["unsupported_cohort_rule:" + str(kind)]}
    picked, basis = _pick(rows, kind, declaration, sessions[0], salt)
    if picked is None:
        return None, basis
    declared_cuts = declaration.get("hash_cutoffs")
    if declared_cuts and basis.get("hash_cutoffs") and declared_cuts != basis["hash_cutoffs"]:
        return None, dict(basis, blocking_gaps=["declared_hash_cutoffs_do_not_match_rule"])
    codes = sorted({canonical_code(r["ts_code"]) for r in picked})
    alt, _ = _pick(list(reversed(rows)), kind, declaration, sessions[0], salt)
    invariant = alt is not None and sorted({canonical_code(r["ts_code"]) for r in alt}) == codes
    return codes, dict(basis, frame_size=len(rows), sampled=len(codes),
                       sampling_frame_is_full_lifecycle_table=True,
                       permutation_invariant=invariant,
                       sampling_keys=["ts_code"],
                       frame_predicate=FRAME_PREDICATE)


def delisted_inside_window(life, sessions):
    """Codes that end their listing inside the window: the survivor test set."""
    start, end = _iso(sessions[0]), _iso(sessions[-1])
    return sorted(c for c, r in life.items()
                  if _iso(r.get("delist_date")) and start < _iso(r["delist_date"]) <= end)


def neutrality_audit(life, sessions, cohort_codes, *, all_a=False):
    """Structural check that the sample cannot encode any outcome.

    Reports, for the drawn cohort and for the population it came from, how many
    names delist inside the window. A rule drawn from ``ts_code`` alone must keep
    delisting names; a survivor-filtered rule would show zero here.

    Zero is not by itself evidence. A cohort that is a small share of the frame
    predicts well under one delisting name, so drawing none is unremarkable. The
    test is therefore the one-sided probability of observing zero under the
    frame's own delisting rate: refused only when that probability falls below
    :data:`SURVIVOR_ALPHA`. The statistic, the threshold and the observed value
    are all reported, so the refusal is reproducible rather than narrative.
    """
    import math
    cohort = {canonical_code(c) for c in cohort_codes}
    dying = delisted_inside_window(life, sessions)
    in_frame = {r["ts_code"] for r in frame_rows(life, sessions[0], sessions[-1],all_a=all_a)}
    kept = sorted(set(cohort) & set(dying))
    dropped = sorted(set(dying) - set(cohort))
    share = (len(cohort) / len(in_frame)) if in_frame else None
    expected = round(len(dying) * share, 4) if share is not None else None
    zero_probability = round(math.exp(-expected), 6) if expected is not None else None
    detected = bool(zero_probability is not None and len(kept) == 0 and zero_probability < SURVIVOR_ALPHA)
    return {
        "window": [sessions[0], sessions[-1]],
        "frame_size": len(in_frame) or None,
        "cohort_size": len(cohort),
        "cohort_share_of_frame": round(share, 6) if share is not None else None,
        "population_names_delisting_inside_window": len(dying),
        "cohort_names_delisting_inside_window": len(kept),
        "cohort_delisting_names": kept[:20],
        "population_delisting_names_not_drawn": len(dropped),
        "expected_cohort_delisting_count": expected,
        "observed_zero_probability": zero_probability,
        "alpha": SURVIVOR_ALPHA,
        "test": "poisson_tail_probability_of_zero_delisting_names_in_cohort",
        "survivor_filter_detected": detected,
        "meaning": "A rule keyed on the identifier alone draws the delisting names at the "
                   "population rate; a rule that had seen the delisting dates would draw none.",
    }


def build_pit_cohort(life, sessions, *, salt, target_size=None, per_exchange=48,
                     rule_kind="per_exchange_hash_cutoff"):
    """Declare a new point-in-time cohort and return (codes, declaration, audit)."""
    if rule_kind == "per_exchange_hash_cutoff":
        declaration = {"rule_kind": rule_kind, "salt": salt, "per_exchange_initial": per_exchange,
                       "sampling_key_fields": ["ts_code"], "sampling_frame": "full_lifecycle_table",
                       "frame_predicate": FRAME_PREDICATE,
                       "posthoc_exclusion": False,
                       "rule": "per-exchange hash sample with thresholds frozen at the window start; "
                               "later listings enter only by clearing the frozen threshold; "
                               "names that delist inside the window are retained until their "
                               "effective delisting date"}
    elif rule_kind == 'all_a_share_census':
        declaration = {'rule_kind':rule_kind,'salt':salt,'sampling_key_fields':['ts_code'],
                       'sampling_frame':'full_lifecycle_table','frame_predicate':ALL_A_FRAME_PREDICATE,
                       'posthoc_exclusion':False,'rule':'all common A-share lifecycle members intersecting the explicit window; no sample cap'}
    else:
        declaration = {"rule_kind": rule_kind, "salt": salt, "target_size": target_size,
                       "sampling_key_fields": ["ts_code"], "sampling_frame": "full_lifecycle_table",
                       "frame_predicate": FRAME_PREDICATE,
                       "posthoc_exclusion": False,
                       "rule": "whole-population hash sample keyed on the identifier alone"}
    codes, basis = reproduce_cohort(life, declaration, sessions)
    require(codes is not None, "cohort rule is not evaluable: " + str(basis.get("blocking_gaps")))
    return codes, declaration, neutrality_audit(life, sessions, codes,all_a=rule_kind=='all_a_share_census')


def membership(life, sessions, codes):
    """Per-session eligible set under the lifecycle rule, for a fixed cohort."""
    from .certification import listed_at
    result = {}
    for day in sessions:
        present = [c for c in codes if listed_at(life, c, day) is True]
        require(len(present) >= 10, "auditable universe below 10 instruments on " + day)
        result[day] = present
    return result


def cohort_hash(declaration):
    return digest(declaration)
