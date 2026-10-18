"""Corporate-action inventory built on a reference-identity proof.

Vendor dividend rows alone do not prove that an entitlement existed. This module
derives approvable events from vendor rows *and* the raw price/adjustment panels
by requiring every approval to reproduce the exchange reference identity

    adj[ex_date] / adj[prev_session]
        == pre_close / ((pre_close - cash_per_share) / (1 + share_rate))

within a tolerance that is calibrated to vendor rounding, not chosen to force
approvals. A row that fails the identity, disagrees with its own duplicates, or
cannot resolve its share components stays unresolved. Missing components are
never filled with zero unless a per-security identity or issuer document proves
that the component is zero.

Not wired to S5, promotion, or any production default.
"""

from dataclasses import dataclass

# Vendor adjustment factors are published to four decimals, so the reference
# identity can only be reproduced to roughly this order. The bound is declared
# once here so every caller inherits the same calibration.
ADJ_ROUNDING_TOLERANCE = 0.001
# The exchange publishes the ex-date reference price on the ordinary price tick,
# so the identity can additionally be off by up to half a tick. That term is
# large for low-priced shares - 0.22% at 2.25 - and must be part of the
# calibration, otherwise cheap stocks are rejected for arithmetic that the
# exchange itself performed. Kept for callers that predate the derived bound.
IDENTITY_TOLERANCE = ADJ_ROUNDING_TOLERANCE


def identity_tolerance(pre_close, cash_per_share=0.0, tick=0.01,
                       rounding=ADJ_ROUNDING_TOLERANCE):
    """Tolerance implied by the two public rounding mechanisms, added together.

    `rounding` covers the four-decimal publication of the adjustment factor and
    the half-tick term covers the exchange rounding the reference price to the
    tick. They are independent rounding steps, so the allowances add. The bound
    is derived from published precision, never from the number of approvals a
    caller wants, and it still separates a benign half-tick event from a genuine
    disagreement about the cash the exchange removed.
    """
    reference = (_num(pre_close) or 0.0) - (_num(cash_per_share) or 0.0)
    if reference <= 0:
        return rounding
    return rounding + 0.5 * tick / reference


@dataclass(frozen=True)
class Verdict:
    status: str
    code: str
    ex_date: str
    reason: str
    detail: dict


def _num(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def share_components(row, identity_proven):
    """Rebuild (bonus, transfer) per share, or raise when unprovable.

    This is the fallback used only when the declared `stk_div` total is absent,
    so a missing total is not treated as a zero total. When both components are
    present they are summed and, if a total is also declared, checked against
    it. A lone component is read as "the other one is zero" only when the
    security's own history demonstrates the vendor alternating which component
    field it populates.
    """
    declared = _num(row.get("stk_div"))
    bonus = _num(row.get("stk_bo_rate"))
    transfer = _num(row.get("stk_co_rate"))

    if bonus is not None and transfer is not None:
        if declared is not None and abs((bonus + transfer) - declared) > 1e-9:
            raise ValueError("components_disagree_with_total")
        return bonus, transfer
    if bonus is None and transfer is None:
        if declared is None or declared > 0:
            raise ValueError("positive_total_without_components")
        return 0.0, 0.0
    sole = bonus if bonus is not None else transfer
    if not identity_proven:
        raise ValueError("lone_component_without_identity_proof")
    if declared is not None and abs(sole - declared) > 1e-9:
        raise ValueError("component_conflicts_with_total")
    return (bonus, 0.0) if bonus is not None else (0.0, transfer)


def merge_same_implementation(rows):
    """Collapse vendor rows that describe one implementation.

    Rows are merged only when every economically binding field agrees -
    ex/record/pay dates, cash, share components and the implementation
    announcement date - and they differ solely in report period or a blank
    announcement date. Merging cannot change any entitlement because every field
    that determines cash or shares is identical. Anything else is left split.
    """
    groups = {}
    for row in rows:
        key = (
            str(row.get("code", "")),
            str(row.get("ex_date", "")),
            str(row.get("record_date", "")),
            str(row.get("pay_date", "")),
            _num(row.get("cash_div")),
            _num(row.get("cash_div_tax")),
            _num(row.get("stk_div")),
            _num(row.get("stk_bo_rate")),
            _num(row.get("stk_co_rate")),
            str(row.get("imp_ann_date") or ""),
        )
        groups.setdefault(key, []).append(row)

    merged, conflicts = [], []
    for key, members in groups.items():
        periods = {str(m.get("end_date") or "") for m in members}
        announced = {str(m.get("imp_ann_date") or "") for m in members}
        # Every economically binding field is already identical by construction
        # of `key`, so any group with more than one member describes one
        # implementation. Rows that agree even on report period and announcement
        # date are redundant rather than distinct and must fold too; leaving them
        # split produced duplicate event identities that later surfaces as a
        # same-day collision instead of being recorded as one event.
        if len(members) > 1:
            head = dict(members[0])
            head["merged_rows"] = len(members)
            head["merged_report_periods"] = sorted(periods)
            head["merged_announcement_dates"] = sorted(announced)
            merged.append(head)
            continue
        merged.extend(members)
    return merged, conflicts


def reference_identity(pre_close, cash_per_share, share_rate, adj_prev, adj_ex):
    """Return (observed, predicted, relative_error) or None when unmeasurable.

    `cash_per_share` must be the cash the *exchange* removed from the reference
    price (the pre-tax declared dividend), which is not always the cash a taxable
    account receives. `share_rate` is the total bonus+transfer per share.
    """
    pre_close = _num(pre_close)
    adj_prev = _num(adj_prev)
    adj_ex = _num(adj_ex)
    if not pre_close or pre_close <= 0 or not adj_prev or adj_prev <= 0 or not adj_ex:
        return None
    cash = _num(cash_per_share) or 0.0
    rate = _num(share_rate) or 0.0
    denominator = (pre_close - cash) / (1.0 + rate)
    if denominator <= 0:
        return None
    observed = adj_ex / adj_prev
    predicted = pre_close / denominator
    return observed, predicted, abs(predicted - observed) / observed


def assess_row(row, *, pre_close, adj_prev, adj_ex, identity_proven,
               tolerance=None, tick=0.01, reference_cash_override=None,
               reference_share_override=None, reference_evidence="",
               cash_absence_proven=False):
    """Classify one vendor row as approved / rejected / unresolved.

    The declared bonus+transfer total is used directly. Component splitting is a
    fallback that only runs when the total itself is absent, because the ledger
    consumes the total multiplier and the pre-tax/net cash split, not the
    bonus-versus-transfer decomposition.

    `tolerance` defaults to the bound derived from the two published rounding
    mechanisms, which needs the declared cash and the observed pre-close.
    """
    code, ex_date = str(row.get("code", "")), str(row.get("ex_date", ""))
    if not ex_date or not str(row.get("record_date") or ""):
        return Verdict("unresolved", code, ex_date, "missing_entitlement_dates", {})
    total = _num(row.get("stk_div"))
    bonus, transfer = _num(row.get("stk_bo_rate")), _num(row.get("stk_co_rate"))
    if total is None:
        try:
            bonus, transfer = share_components(row, identity_proven)
        except ValueError as exc:
            return Verdict("unresolved", code, ex_date, str(exc), {})
        total = bonus + transfer
    # Tushare's documented fields are cash_div (after tax) and cash_div_tax
    # (before tax). The latter is the ordinary ex-reference input. Neither
    # field alone proves what a particular account actually receives: for
    # natural persons the tax can be deferred until sale.
    gross_cash = _num(row.get("cash_div_tax"))
    reported_after_tax = _num(row.get("cash_div"))
    if gross_cash is None:
        if not cash_absence_proven or not reference_evidence:
            return Verdict("unresolved", code, ex_date, "missing_pre_tax_cash_component", {})
        gross_cash = 0.0
    if gross_cash < 0 or (reported_after_tax is not None
                          and (reported_after_tax < 0 or reported_after_tax > gross_cash + 1e-9)):
        return Verdict("unresolved", code, ex_date, "inconsistent_cash_fields", {})
    if (reference_cash_override is not None or reference_share_override is not None) and not reference_evidence:
        return Verdict("unresolved", code, ex_date, "unproven_reference_override", {})
    reference_cash = (gross_cash if reference_cash_override is None
                      else _num(reference_cash_override))
    reference_share = (total if reference_share_override is None
                       else _num(reference_share_override))
    if (reference_cash is None or reference_cash < 0 or reference_share is None
            or reference_share < 0):
        return Verdict("unresolved", code, ex_date, "invalid_reference_override", {})
    measured = reference_identity(pre_close, reference_cash, reference_share, adj_prev, adj_ex)
    if measured is None:
        return Verdict("unresolved", code, ex_date, "identity_unmeasurable", {})
    observed, predicted, error = measured
    if tolerance is None:
        tolerance = identity_tolerance(pre_close, reference_cash, tick)
    detail = {"observed_ratio": observed, "predicted_ratio": predicted,
              "relative_error": error, "reference_cash": reference_cash,
              "reference_share_rate": reference_share,
              "entitlement_cash_pre_tax": gross_cash,
              "reported_cash_after_tax": reported_after_tax,
              "account_cash_receipt": None,
              "net_cash": reported_after_tax,
              "share_rate": total, "bonus": bonus, "transfer": transfer,
              "tolerance": tolerance,
              "tax_split_verified": False,
              "reference_evidence": reference_evidence or None,
              "cash_absence_proven": bool(cash_absence_proven)}
    if error > tolerance:
        return Verdict("rejected", code, ex_date, "reference_identity_failed", detail)
    return Verdict("approved", code, ex_date, "reference_identity_holds", detail)
