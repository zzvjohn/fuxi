"""A1-only corporate-action adjudication; never grants a production S5 label."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from corporate_action_inventory import assess_row


@dataclass(frozen=True)
class BundleVerdict:
    status: str
    reason: str
    detail: dict
    action: dict | None


def _num(value):
    if value is None or value == "":
        return None
    number = float(value)
    if number != number:
        return None
    return number


def _iso(value):
    value = str(value or "")
    if len(value) == 8 and value.isdigit():
        return value[:4] + "-" + value[4:6] + "-" + value[6:]
    return value


def _unique_implementations(rows):
    """Fold only identical entitlements; independent same-day rows stay split."""
    unique = OrderedDict()
    for row in rows:
        if row.get("div_proc") != "实施":
            continue
        identity = tuple(row.get(k) for k in (
            "ts_code", "record_date", "ex_date", "pay_date", "div_listdate",
            "imp_ann_date", "cash_div", "cash_div_tax", "stk_div",
            "stk_bo_rate", "stk_co_rate"))
        unique.setdefault(identity, row)
    return list(unique.values())


def adjudicate_bundle(rows, *, code, ex_date, pre_close, adj_prev, adj_ex,
                      issuer_exception=None, source_evidence=""):
    """Separate holder rights from exchange reference parameters.

    Multiple independent cash-only implementations may be added when their
    record/payment dates agree and an issuer notice proves the combined amount.
    Shares, cash absent from the provider, differential distributions, and
    virtual reference inputs require a dated issuer exception.
    """
    merged = _unique_implementations(rows)
    if not merged:
        return BundleVerdict("unresolved", "no_implemented_row", {}, None)
    if any(r.get("ts_code") != code or _iso(r.get("ex_date")) != ex_date for r in merged):
        return BundleVerdict("unresolved", "event_identity_mismatch", {}, None)
    notice = issuer_exception or {}
    if notice and (notice.get("code") != code or notice.get("ex_date") != ex_date
                   or not notice.get("notice_url") or not notice.get("notice_date")
                   or notice["notice_date"] > ex_date):
        return BundleVerdict("unresolved", "invalid_issuer_exception", {}, None)
    if notice.get("reason") == "restructuring_shares_not_distributed_to_existing_holders":
        # A capital increase shown by a vendor as stk_div can instead fund a
        # restructuring investor or creditor.  The old holder gets no shares;
        # its price-reference event must be modeled separately from entitlement.
        if (notice.get("holder_cash_pre_tax") != 0.0 or
                notice.get("holder_share_rate") != 0.0):
            return BundleVerdict("unresolved", "invalid_non_holder_rights", {}, None)
        return BundleVerdict("not_holder_distribution", notice["reason"],
                             {"issuer_notice_url": notice["notice_url"],
                              "source_rows": len(rows)}, None)
    if len(merged) > 1:
        dates = {tuple(r.get(k) for k in ("record_date", "pay_date", "div_listdate"))
                 for r in merged}
        shares = [_num(r.get("stk_div")) for r in merged]
        gross = [_num(r.get("cash_div_tax")) for r in merged]
        reported = [_num(r.get("cash_div")) for r in merged]
        if (len(dates) != 1 or any(x != 0 for x in shares)
                or any(x is None or x <= 0 for x in gross)
                or any(x is None for x in reported) or not notice):
            return BundleVerdict("unresolved", "unproven_same_day_aggregation", {}, None)
        row = dict(merged[0], cash_div_tax=sum(gross), cash_div=sum(reported),
                   stk_div=0.0)
    else:
        row = dict(merged[0])
    row["code"], row["ex_date"] = code, ex_date
    if notice:
        gross = _num(row.get("cash_div_tax"))
        shares = _num(row.get("stk_div"))
        if gross is None and notice.get("reason") == "issuer_explicit_no_cash_pure_transfer":
            gross = 0.0
        if (gross is None or shares is None
                or abs(gross - notice["holder_cash_pre_tax"]) > 1e-8
                or abs(shares - notice["holder_share_rate"]) > 1e-8):
            return BundleVerdict("unresolved", "issuer_holder_rights_disagree_with_vendor", {}, None)
    evidence = (notice.get("notice_url") or source_evidence)
    verdict = assess_row(
        row, pre_close=pre_close, adj_prev=adj_prev, adj_ex=adj_ex,
        identity_proven=False,
        reference_cash_override=notice.get("reference_cash") if notice else None,
        reference_share_override=notice.get("reference_share_rate") if notice else None,
        reference_evidence=evidence if notice else "",
        cash_absence_proven=notice.get("reason") == "issuer_explicit_no_cash_pure_transfer",
    )
    if verdict.status != "approved":
        return BundleVerdict(verdict.status, verdict.reason, verdict.detail, None)
    detail = dict(verdict.detail)
    detail.update(source_rows=len(rows), economically_unique_implementations=len(merged),
                  issuer_notice_url=notice.get("notice_url"),
                  account_cash_receipt_known=False,
                  tax_settlement_model="unspecified",
                  holder_class=notice.get("account_class_required", "unspecified"))
    record = _iso(row.get("record_date"))
    pay = _iso(row.get("pay_date")) or ex_date
    new_shares = _iso(row.get("div_listdate")) or ex_date
    if not record or not record < ex_date <= pay or (detail["share_rate"] > 0 and new_shares < ex_date):
        return BundleVerdict("unresolved", "invalid_implementation_dates", detail, None)
    action = {
        "event_id": "a1_" + code + "_" + ex_date,
        "code": code, "record_date": record, "ex_date": ex_date,
        "pay_date": pay, "cash_per_share_net": detail["entitlement_cash_pre_tax"],
        "share_multiplier": 1.0 + detail["share_rate"],
        "new_shares_date": new_shares,
        "evidence": evidence + "; A1 gross-cash stress only; tax settlement unspecified",
    }
    return BundleVerdict("approved_for_gross_cash_stress_only", "reference_identity_holds",
                         detail, action)
