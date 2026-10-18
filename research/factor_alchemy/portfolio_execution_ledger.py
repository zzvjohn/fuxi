"""Isolated long-only execution ledger. Not wired to S5 or promotion.

Orders are sized from the preceding signal close and attempted at subsequent
session opens. Open-price fills and aggregate bps costs are research assumptions,
not an auction, liquidity, tax, or broker simulator. Unknown events fail closed.
"""

from dataclasses import dataclass, field
from datetime import date as Date
from math import ceil, floor, isclose, isfinite, nan


def positive(value):
    return isinstance(value, (int, float)) and isfinite(value) and value > 0


@dataclass(frozen=True)
class LotRule:
    minimum: int = 100
    step: int = 100

    def round_order(self, quantity):
        value = floor(max(0, quantity) / self.step) * self.step
        return value if value >= self.minimum else 0


@dataclass(frozen=True)
class CorporateAction:
    event_id: str
    code: str
    record_date: str
    ex_date: str
    pay_date: str
    cash_per_share_net: float = 0.0
    share_multiplier: float = 1.0
    new_shares_date: str = ""
    evidence: str = ""


@dataclass
class LedgerResult:
    days: list = field(default_factory=list)
    journal: list = field(default_factory=list)
    orders: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    collapsed_events: list = field(default_factory=list)
    excluded: list = field(default_factory=list)
    status: str = "complete_under_declared_assumptions"
    formal_eligibility: bool = False


def entitlement_signature(action):
    """Economic identity of an action: what it pays and when it is claimable.

    `new_shares_date` gates share delivery and is therefore binding only when an
    action actually issues shares. For a pure cash distribution it carries no
    entitlement, so two assemblies that disagree on it describe one event.
    """
    shares_binding = action.new_shares_date if action.share_multiplier > 1 else None
    return (action.code, action.ex_date, action.record_date, action.pay_date,
            action.cash_per_share_net, action.share_multiplier, shares_binding)


def consolidate_identical_events(actions, policy="block"):
    """Collapse same-day actions that carry one entitlement.

    Two assemblers can name the same distribution differently, so name-level
    deduplication cannot see that it is one event. Groups whose members all share
    an entitlement signature collapse to a single best-documented instance and
    the removed ids are returned for the audit trail. Members that disagree
    economically are never merged - a genuine multi-entitlement day still needs
    an explicitly consolidated event.

    `policy="block"` keeps the strict behaviour and refuses every duplicate.
    """
    if policy not in ("block", "collapse_identical"):
        raise ValueError("invalid duplicate event policy")
    groups = {}
    for index, action in enumerate(actions):
        groups.setdefault((action.code, action.ex_date), []).append((index, action))
    kept, collapsed = [], []
    for (code, ex_date), members in groups.items():
        if len(members) == 1:
            kept.append(members[0][1])
            continue
        signatures = {entitlement_signature(a) for _, a in members}
        if len(signatures) > 1:
            raise ValueError("conflicting same-day entitlements for %s on %s" % (code, ex_date))
        if policy == "block":
            raise ValueError("multiple same-day actions require an explicit consolidated event")
        ordered = sorted(members, key=lambda item: (-len(item[1].evidence or ""), item[1].event_id))
        head = ordered[0][1]
        kept.append(head)
        fields = ("record_date", "pay_date", "cash_per_share_net",
                  "share_multiplier", "new_shares_date", "evidence")
        collapsed.append({
            "code": code, "ex_date": ex_date, "kept_id": head.event_id,
            "collapsed_ids": sorted(a.event_id for _, a in members),
            "differing_fields": sorted(f for f in fields
                                       if len({getattr(a, f) for _, a in members}) > 1),
        })
    kept.sort(key=lambda a: (a.ex_date, a.code, a.event_id))
    return kept, collapsed


def run_ledger(dates, bars, targets, *, initial_cash=10_000_000.0, cost_bps=15.0,
               rules=None, actions=(), fractional_share_policy="block",
               delist_evidence=None, duplicate_event_policy="block",
               missing_limit_policy="block", unresolved_action_policy="block",
               halted_session_check_policy="unavailable", reference_only_events=(),
               gross_cash_tax_rate=0.0, delist_mark_fraction=1.0,
               share_exchanges=()):
    """bars[(ISO date, code)] contain raw open/close/pre_close and explicit state.

    targets[signal_date] = {code: weight}; None expires pending orders without
    liquidating holdings. signal_date must precede fill date.
    Halts must be explicitly evidenced; absent prices are never inferred halts.
    Exact limits or an explicit no_limit flag are required for each order.
    Corporate action entitlements use record-date holdings; payments remain
    receivables until pay_date. Unsupported/missing events block the experiment.

    Three research policies are opt-in and all default to the strict behaviour:

    fractional_share_policy="floor" or "ceil" models the two adjacent integer
    allotments of a fractional entitlement. These are account scenarios, not
    recovered clearing results. Neither creates cash in lieu.

    delist_evidence={code: {"date": ISO, "evidence": str}} marks a position as
    unlisted from a documented date. The position is frozen at its last observed
    mark with no synthetic return and no revaluation; it cannot be traded. This
    is a declared accounting policy, not a recovered market price.

    duplicate_event_policy="collapse_identical" folds same-day actions that carry
    one entitlement signature into a single instance and records the removed ids.
    One entitlement is still granted once; only redundant re-descriptions of it
    are dropped. Same-day actions that differ economically always block.

    missing_limit_policy="defer" records an unfilled order instead of failing the
    day when a session has no published price limits. Deriving the limits is not
    an option: a derived limit could admit a fill that the real book refused, so
    the strict direction is to refuse execution and retry on the next session.

    unresolved_action_policy="exclude_held_security_day" narrows the halt from the
    whole arm to one security-session. A held name whose reference broke **and for
    which no action was supplied on that session** is carried at its last observed
    mark, cannot be traded, and is journalled as an exclusion; every other name, and
    that name on every other session, is processed normally. A security-session that
    did receive an action is never excluded: its entitlement is already journalled,
    and an exclusion there would grant and exclude one event at the same time. The
    entitlement is still not supplied for an excluded session - this policy never
    grants shares or cash - and the carried mark is a stale price, so the session's
    return carries the mark-to-reference error of the excluded weight and the day is
    flagged. The strict default remains to fail the experiment.

    halted_session_check_policy="vacuous" declines to treat a published full-day
    halt as a session whose entitlement check is merely unreadable. A full-day
    halt publishes no reference price, so no reference adjustment can take place
    on it; the check is vacuous rather than unavailable. This never supplies an
    entitlement: an event whose ex-date falls inside the halt still surfaces as a
    factor move on the first session after the halt that carries an adjustment
    row, and that session is checked normally, so the move cannot hide. Each
    relaxed session is journalled as halted_session_check_vacuous, so the
    relaxation is counted rather than inferred.
    """
    dates = list(dates)
    if not dates or dates != sorted(set(dates)):
        raise ValueError("calendar must be nonempty, sorted and unique")
    for value in dates + list(targets):
        if Date.fromisoformat(value).isoformat() != value:
            raise ValueError("dates must be ISO dates")
    if not positive(initial_cash) or not isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("invalid capital/cost")
    if fractional_share_policy not in ("block", "floor", "ceil"):
        raise ValueError("invalid fractional share policy")
    if not (0 <= gross_cash_tax_rate <= 1 and 0 <= delist_mark_fraction <= 1):
        raise ValueError("invalid model tax or delist mark fraction")
    if duplicate_event_policy not in ("block", "collapse_identical"):
        raise ValueError("invalid duplicate event policy")
    if missing_limit_policy not in ("block", "defer"):
        raise ValueError("invalid missing limit policy")
    if unresolved_action_policy not in ("block", "exclude_held_security_day"):
        raise ValueError("invalid unresolved action policy")
    if halted_session_check_policy not in ("unavailable", "vacuous"):
        raise ValueError("invalid halted session check policy")
    delist_evidence = dict(delist_evidence or {})
    reference_only_events = set(reference_only_events)
    share_exchanges = list(share_exchanges)
    for exchange in share_exchanges:
        if (not isinstance(exchange, dict) or not exchange.get("evidence")
                or exchange.get("old_code") == exchange.get("new_code")
                or exchange.get("date") not in dates
                or not positive(exchange.get("ratio"))):
            raise ValueError("invalid share exchange")
    for code, fact in delist_evidence.items():
        if (not isinstance(fact, dict) or not fact.get("evidence")
                or Date.fromisoformat(str(fact.get("date", ""))).isoformat() != fact.get("date")
                or fact["date"] not in dates):
            raise ValueError("unproven delisting fact")
    for weights in targets.values():
        if weights is None:
            continue
        if any(not isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) > 1 + 1e-12:
            raise ValueError("long-only weights must sum to at most one")
    rules = rules or {}
    if any(r.minimum <= 0 or r.step <= 0 for r in rules.values()):
        raise ValueError("invalid lot rule")
    if len({a.event_id for a in actions}) != len(actions):
        raise ValueError("duplicate action id")
    for a in actions:
        if (not a.evidence or a.record_date not in dates or a.ex_date not in dates
                or not (a.record_date < a.ex_date <= a.pay_date)
                or not isfinite(a.share_multiplier) or a.share_multiplier < 1
                or not isfinite(a.cash_per_share_net) or a.cash_per_share_net < 0
                or (a.share_multiplier > 1 and a.new_shares_date < a.ex_date)):
            raise ValueError("unsupported/unproven corporate action")
    result = LedgerResult()
    actions, collapses = consolidate_identical_events(actions, policy=duplicate_event_policy)
    result.collapsed_events = collapses
    cash, receivable = float(initial_cash), 0.0
    shares, marks, desired, entitlements = {}, {}, {}, {}
    locked, payments = [], []
    target_date = None
    processed_targets = set()
    previous_nav = initial_cash

    def failure(day, code, reason):
        result.errors.append({"date": day, "code": code, "reason": reason})

    def journal(day, code, quantity=0, cash_delta=0.0, receivable_delta=0.0, **details):
        result.journal.append(dict(date=day, code=code, quantity=quantity,
                                   cash_delta=cash_delta, receivable_delta=receivable_delta, **details))

    for day in dates:
        error_start = len(result.errors)
        fresh = [d for d in targets if d < day and d not in processed_targets]
        if fresh:
            target_date = max(fresh)
            processed_targets.update(fresh)
            weights = targets[target_date]
            desired = {} if weights is not None else dict(shares)
            if weights is None:
                result.orders.append(dict(date=day, signal_date=target_date, code='', side='cancel',
                                          requested=0, filled=0, reason='signal_expired'))
            for code, weight in sorted((weights or {}).items()):
                if weight == 0:
                    continue
                price = bars.get((target_date, code), {}).get("close")
                if not positive(price):
                    failure(day, code, "missing_signal_close")
                    continue
                rule = rules.get(code, LotRule())
                desired[code] = rule.round_order(previous_nav * weight / (1 + cost_bps / 10000) / price)

        for exchange in share_exchanges:
            if exchange["date"] != day:
                continue
            old, new = exchange["old_code"], exchange["new_code"]
            old_count = shares.get(old, 0)
            if not old_count:
                continue
            theoretical = old_count * exchange["ratio"]
            if not isclose(theoretical, round(theoretical), abs_tol=1e-8):
                if fractional_share_policy == "block":
                    failure(day, old, "fractional_share_exchange")
                    continue
                new_count = (floor if fractional_share_policy == "floor" else ceil)(theoretical)
            else:
                new_count = round(theoretical)
            shares[old] = 0
            shares[new] = shares.get(new, 0) + new_count
            desired.pop(old, None)
            journal(day, old, -old_count, kind="share_exchange_out",
                    evidence=exchange["evidence"])
            journal(day, new, new_count, kind="share_exchange_in",
                    theoretical=theoretical, granted=new_count,
                    evidence=exchange["evidence"])
        todays_actions = {a.code: a for a in actions if a.ex_date == day}
        if len(todays_actions) != sum(a.ex_date == day for a in actions):
            raise ValueError("multiple same-day actions require an explicit consolidated event")
        for a in todays_actions.values():
            entitlement = entitlements.get(a.event_id, 0)
            extra = entitlement * (a.share_multiplier - 1)
            if not isclose(extra, round(extra), abs_tol=1e-8):
                if fractional_share_policy not in ("floor", "ceil"):
                    failure(day, a.code, "fractional_action_requires_cash_in_lieu")
                    continue
                whole = (floor if fractional_share_policy == "floor" else ceil)(extra)
                journal(day, a.code,
                        kind=("fractional_retired" if fractional_share_policy == "floor"
                              else "fractional_rounded_up"), event_id=a.event_id,
                        theoretical_extra=extra, granted_extra=whole,
                        retired_fraction=max(0.0, extra - whole),
                        allocation_difference=whole - extra)
                extra = whole
            else:
                extra = int(round(extra))
            due = entitlement * a.cash_per_share_net * (1 - gross_cash_tax_rate)
            shares[a.code] = shares.get(a.code, 0) + extra
            receivable += due
            payments.append((a.pay_date, a.code, due, a.event_id))
            locked.append((a.new_shares_date, a.code, extra))
            if a.code in desired:
                desired[a.code] = int(floor(desired[a.code] * a.share_multiplier))
            journal(day, a.code, extra, receivable_delta=due, kind="entitlement", event_id=a.event_id)
        remaining_payments = []
        for pay_date, code, amount, event_id in payments:
            if pay_date <= day:
                cash += amount
                receivable -= amount
                journal(day, code, cash_delta=amount, receivable_delta=-amount,
                        kind="payment", event_id=event_id)
            else:
                remaining_payments.append((pay_date, code, amount, event_id))
        payments = remaining_payments
        locked = [item for item in locked if item[0] > day and item[2]]

        # A price-reference break is evidence of a missing event, not permission
        # to synthesize shares or cash from an adjustment factor.
        excluded_today = set()
        for code in sorted(set(shares) | set(desired)):
            quantity = shares.get(code, 0)
            if quantity <= 0 and desired.get(code, 0) <= 0:
                continue
            bar = bars.get((day, code), {})
            state = bar.get("state", "unknown")
            fact = delist_evidence.get(code)
            if fact and day >= fact["date"]:
                # Documented unlisted position: no quote, no action check, and
                # no tradability. The freeze itself is journalled once.
                if day == fact["date"]:
                    journal(day, code, kind="delisted_freeze", evidence=fact["evidence"])
                continue
            settled_today = (code in todays_actions or (code, day) in reference_only_events)
            halted_full_day = (state == "halted" and bool(bar.get("evidence"))
                               and "close" not in bar)
            if bar.get("action_check_available") is not True:
                if halted_full_day and halted_session_check_policy == "vacuous":
                    # No reference is published for a full-day halt, so no
                    # reference adjustment can occur on the session. Recorded,
                    # not assumed: the relaxation is journalled and the session
                    # is counted by the caller.
                    journal(day, code, kind="halted_session_check_vacuous",
                            evidence=bar.get("evidence"),
                            settled_by_supplied_action=settled_today)
                else:
                    failure(day, code, "missing_action_check")
            needs_action = bool(bar.get("action_required")) and not settled_today
            # A reference may only "break" for a name whose entitlement was not
            # supplied this session. When an action was supplied the entitlement is
            # already journalled above, so the reference move is exactly the
            # adjustment that action describes - not an unsettled event. Excluding
            # it would grant and exclude the same security-session at once, and the
            # grant is the thing that must never be traded away for a longer run.
            reference_broke = (not settled_today and quantity > 0
                               and positive(marks.get(code))
                               and positive(bar.get("pre_close"))
                               and not isclose(bar["pre_close"], marks[code],
                                               abs_tol=.011, rel_tol=0))
            if ((needs_action or reference_broke)
                    and unresolved_action_policy == "exclude_held_security_day"):
                # Narrow the halt to this security-session: the entitlement is
                # still not supplied, the carried mark is stale, and the name
                # cannot be traded. Every other name, and this name on every
                # other session, is processed normally.
                if not positive(marks.get(code)):
                    failure(day, code, "exclusion_without_prior_mark")
                    continue
                excluded_today.add(code)
                journal(day, code, kind="excluded_security_day",
                        action_required=needs_action, reference_break=reference_broke,
                        carried_mark=marks[code], reference_field=bar.get("pre_close"))
                continue
            if needs_action:
                failure(day, code, "missing_corporate_action_event")
            if quantity <= 0:
                continue
            if state not in ("trading", "halted", "partial_halt"):
                failure(day, code, "unknown_held_state")
            elif state == "partial_halt":
                # The session was halted across the open but still published a
                # close, so the position is valued from that close and the
                # reference break is checked normally. A halt that hides the open
                # is never treated as a tradable open.
                if not positive(bar.get("close")) or not positive(bar.get("pre_close")):
                    failure(day, code, "partial_halt_without_quote")
                elif not settled_today and (
                        bar.get("action_required") or
                        (positive(marks.get(code)) and
                         not isclose(bar["pre_close"], marks[code], abs_tol=.011, rel_tol=0))):
                    failure(day, code, "corporate_action_or_reference_break")
            elif state == "halted":
                if not bar.get("evidence") or not positive(marks.get(code)):
                    failure(day, code, "unproven_halt_mark")
                if code in todays_actions or bar.get("action_required"):
                    failure(day, code, "halt_action_requires_explicit_valuation")
            else:
                if not positive(bar.get("open")) or not positive(bar.get("pre_close")):
                    failure(day, code, "missing_open_or_reference")
                elif not settled_today and (bar.get("action_required") or
                        (positive(marks.get(code)) and not isclose(bar["pre_close"], marks[code], abs_tol=.011, rel_tol=0))):
                    failure(day, code, "corporate_action_or_reference_break")

        if len(result.errors) == error_start:
            for side in ("sell", "buy"):
                for code in sorted(set(shares) | set(desired)):
                    held = shares.get(code, 0)
                    delta = desired.get(code, 0) - held
                    if (side == "sell" and delta >= 0) or (side == "buy" and delta <= 0):
                        continue
                    if code in excluded_today:
                        # The reference already reflects a break this session, so
                        # an open-price fill cannot be evidenced at a price the
                        # entitlement never established. The pending target is
                        # retried next session; the name is not frozen.
                        result.orders.append(dict(date=day, signal_date=target_date,
                                                  code=code, side=side, requested=abs(delta),
                                                  filled=0, reason="excluded_security_day"))
                        continue
                    fact = delist_evidence.get(code)
                    if fact and day >= fact["date"]:
                        # An unlisted position cannot be sold or added to.
                        if held and side == "sell":
                            result.orders.append(dict(date=day, signal_date=target_date,
                                                      code=code, side=side, requested=abs(delta),
                                                      filled=0, reason="unlisted_frozen"))
                        continue
                    bar = bars.get((day, code), {})
                    reason = None
                    state = bar.get("state", "unknown")
                    if state == "halted" and bar.get("evidence"):
                        reason = "halted"
                    elif state == "partial_halt":
                        # The open fell inside a suspension window, so an
                        # open-price fill cannot be evidenced. The pending target
                        # is retried on the next tradable session.
                        reason = "partial_halt_open"
                    elif state != "trading" or not positive(bar.get("open")):
                        reason = "unknown_order_state"
                        failure(day, code, reason)
                    elif not bar.get("no_limit", False) and not (positive(bar.get("up_limit")) and positive(bar.get("down_limit"))):
                        if missing_limit_policy == "defer":
                            # No published limit means execution cannot be shown to
                            # have been possible, so the order is refused rather
                            # than priced. The pending target is retried next
                            # session, which delays the rebalance by one session
                            # instead of inventing a limit or a fill.
                            reason = "no_limit_evidence"
                        else:
                            reason = "missing_exact_limits"
                            failure(day, code, reason)
                    elif not bar.get("no_limit", False):
                        if side == "buy" and bar["open"] >= bar["up_limit"] - 1e-6:
                            reason = "limit_up"
                        if side == "sell" and bar["open"] <= bar["down_limit"] + 1e-6:
                            reason = "limit_down"
                    rule = rules.get(code, LotRule())
                    quantity = abs(delta)
                    if reason is None:
                        if side == "sell":
                            unavailable = sum(q for _, c, q in locked if c == code)
                            quantity = min(quantity, max(0, held - unavailable))
                            quantity = quantity if quantity == held else rule.round_order(quantity)
                        else:
                            affordable = floor((cash + 1e-8) / (bar["open"] * (1 + cost_bps / 10000)))
                            quantity = rule.round_order(min(quantity, affordable))
                        if quantity == 0:
                            reason = "cash_lot_or_locked_shares"
                    order = dict(date=day, signal_date=target_date, code=code, side=side,
                                 requested=abs(delta), filled=0, reason=reason)
                    if reason is None:
                        assert target_date is not None and target_date < day
                        signed = quantity if side == "buy" else -quantity
                        notional = quantity * bar["open"]
                        fee = notional * cost_bps / 10000
                        movement = -signed * bar["open"] - fee
                        cash += movement
                        shares[code] = held + signed
                        if shares[code] == 0:
                            # A later reacquisition must not compare today's
                            # reference with this former holding's stale mark.
                            marks.pop(code, None)
                        if cash < -1e-6 or shares[code] < 0:
                            raise AssertionError("cash/position conservation violated")
                        journal(day, code, signed, movement, kind="fill", price=bar["open"],
                                fee=fee, signal_date=target_date)
                        order.update(filled=quantity, price=bar["open"], fee=fee)
                    result.orders.append(order)

        position_rows = {}
        mark_ok = len(result.errors) == error_start
        for code, quantity in sorted(shares.items()):
            if not quantity:
                continue
            bar = bars.get((day, code), {})
            fact = delist_evidence.get(code)
            if fact and day >= fact["date"] and positive(marks.get(code)):
                exchange = next((x for x in share_exchanges
                                 if x["old_code"] == code and day < x["date"]), None)
                if exchange:
                    new_bar = bars.get((day, exchange["new_code"]), {})
                    mark = (exchange["ratio"] * new_bar["close"]
                            if positive(new_bar.get("close")) else nan)
                    if not isfinite(mark):
                        mark_ok = False
                        failure(day, code, "missing_exchange_receivable_mark")
                else:
                    # Explicit model haircut, never a recovered tradable price.
                    mark = marks[code] * delist_mark_fraction
            elif bar.get("state") in ("trading", "partial_halt") and positive(bar.get("close")):
                mark = bar["close"]
                marks[code] = mark
            elif code in excluded_today and positive(marks.get(code)):
                # Unsettled event on a held session: the name is carried at its
                # last observed mark and flagged, so the session still has a NAV
                # instead of stopping the arm.
                mark = marks[code]
            elif bar.get("state") == "halted" and bar.get("evidence") and positive(marks.get(code)):
                mark = marks[code]
            else:
                mark = nan
                mark_ok = False
                failure(day, code, "missing_close_mark")
            position_rows[code] = {"shares": quantity, "mark": mark,
                                   "locked": sum(q for _, c, q in locked if c == code)}
            if code in excluded_today:
                position_rows[code]["excluded"] = "excluded_security_day"
        nav = cash + receivable + sum(row["shares"] * row["mark"] for row in position_rows.values()) if mark_ok else nan
        result.days.append(dict(date=day, cash=cash, receivable=receivable, nav=nav,
                                positions=position_rows,
                                excluded_security_days=sorted(excluded_today),
                                return_=nav / previous_nav - 1 if mark_ok else nan))
        for code in sorted(excluded_today):
            result.excluded.append({"date": day, "code": code,
                                    "policy": unresolved_action_policy})
        for action in actions:
            if action.record_date == day:
                entitlements[action.event_id] = shares.get(action.code, 0)
        if not mark_ok:
            result.status = "data_insufficient"
            break
        previous_nav = nav
    return result


def reconcile_journal(result, initial_cash):
    """Independent accounting replay from journal, not the engine's transitions."""
    cash, receivable, positions = initial_cash, 0.0, {}
    cursor, compared, nav_compared = 0, 0, 0
    for row in result.days:
        while cursor < len(result.journal) and result.journal[cursor]["date"] <= row["date"]:
            entry = result.journal[cursor]
            if entry["kind"] == "fill":
                expected = -entry["quantity"] * entry["price"] - entry["fee"]
                assert isclose(expected, entry["cash_delta"], abs_tol=1e-7)
                assert entry["signal_date"] < entry["date"]
            cash += entry["cash_delta"]
            receivable += entry["receivable_delta"]
            positions[entry["code"]] = positions.get(entry["code"], 0) + entry["quantity"]
            cursor += 1
        expected_positions = {code: count for code, count in positions.items() if count}
        assert expected_positions == {code: value["shares"] for code, value in row["positions"].items()}
        assert isclose(cash, row["cash"], abs_tol=1e-6)
        assert isclose(receivable, row["receivable"], abs_tol=1e-6)
        if isfinite(row["nav"]):
            nav = cash + receivable + sum(positions[c] * v["mark"] for c, v in row["positions"].items())
            assert isclose(nav, row["nav"], abs_tol=1e-6)
            nav_compared += 1
        compared += 1
    assert compared > 0 and cursor == len(result.journal)
    return {"compared_days": compared, "finite_nav_days": nav_compared,
            "journal_entries": cursor, "independent_execution_or_data_validation": False}


def assess_economic_book(result, expected_dates, initial_cash):
    """Fail closed before an isolated cash book can become an economic label.

    Journal parity checks arithmetic only. They do not certify event coverage,
    auction fills, source completeness, or the investment benchmark.
    """
    expected_dates = list(expected_dates)
    if not expected_dates or expected_dates != sorted(set(expected_dates)):
        raise ValueError("expected calendar must be nonempty, sorted and unique")
    parity = reconcile_journal(result, initial_cash)
    seen = [row["date"] for row in result.days]
    reasons = []
    if seen != expected_dates:
        reasons.append("incomplete_window")
    if result.status != "complete_under_declared_assumptions" or result.errors:
        reasons.append("ledger_data_insufficient")
    if result.excluded or any(row.get("excluded_security_days") for row in result.days):
        reasons.append("unresolved_corporate_action_or_stale_mark")
    if any(not isfinite(row.get("nav", nan)) for row in result.days):
        reasons.append("nonfinite_nav")
    if parity["finite_nav_days"] != len(expected_dates):
        reasons.append("incomplete_daily_parity")
    return {
        "status": "book_complete_under_declared_assumptions" if not reasons else "data_insufficient",
        "reasons": sorted(set(reasons)), "expected_days": len(expected_dates),
        "observed_days": len(seen), "parity": parity,
        "independent_execution_or_data_validation": False,
        "formal_eligibility": False,
    }
