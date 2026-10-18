"""Native next-session execution under explicit research assumptions and market evidence."""

from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from .account import Account, money
from .contracts import digest, require, timestamp, finite, MARKET_CONTENT_EXCLUSIONS
from .portfolio import target_weights
from .action_accounting import SUPPORTED,pending_shares,post_entitlements,sell_quantity
from .a_share import instrument_rule, desired_shares, buy_quantity, sell_quantity as board_sell_quantity,venue_code_at


class ResearchAccountContext:
    """Reusable indexes only for uncertified, frozen research markets."""
    def __init__(self,market):
        require(not market.get('investment_assurance') and not market.get('certification_hash')
                and not market.get('event_coverage_certified') and not market.get('tradability_certified'),
                'cached research context cannot carry formal market assurance')
        if getattr(market, "_columnar_market", False):
            from .columnar import populate_research_context
            populate_research_context(self, market)
            return
        self.market=market
        self.execution_hash=digest(market['execution'])
        self.rules={c:instrument_rule(market['execution'],c) for c in {r['instrument'] for r in market['bars']}}
        self.bars={(r['session'],r['instrument']):r for r in market['bars']}
        require(len(self.bars)==len(market['bars']),'duplicate market bar')
        self.by_day={d:{} for d in market['sessions']}
        for row in market['bars']:self.by_day[row['session']][row['instrument']]=row
        self.events=sorted(market['actions'],key=lambda x:timestamp(x['timestamp']))


def run_account(ir, market, *, cost_multiplier=1., participation_multiplier=1., start=None, end=None,
                fractional_resolver=None,research_context=None,restricted_policy=None,restricted_scenario=None):
    require(digest({k: v for k, v in ir.items() if k != "ir_hash"}) == ir["ir_hash"], "IR content mismatch")
    execution = market["execution"]
    require(digest(execution) == ir["execution_hash"], "execution/IR mismatch")
    require(execution["settlement_sessions"] >= 1 and execution["buy_lot"] >= 1 and execution["sell_lot"] >= 1,
            "invalid market lot/settlement rule")
    sessions = [d for d in market["sessions"] if (start is None or d >= start) and (end is None or d <= end)]
    require(len(sessions) >= 2, "empty/short account window")
    restricted = None
    require((restricted_policy is None)==(restricted_scenario is None),'restricted policy/scenario must be paired')
    if restricted_policy is not None:
        from .restricted_assets import RestrictedResearch,public_mark
        restricted=RestrictedResearch(market,restricted_policy,restricted_scenario)
        require(sessions[-1]<=restricted_policy['horizon_end'],'restricted scenario horizon exceeded')
    direct_indexes = False
    if research_context is None and getattr(market, "_columnar_market", False):
        if market.get('investment_assurance'):
            # P-20261015-012: a certified columnar market uses direct indexes;
            # the research-context cache stays reserved for uncertified markets.
            direct_indexes = True
        else:
            research_context = market.research_context()
    if research_context is None:
        if direct_indexes:
            rules = {code: instrument_rule(execution, code) for code in market.instruments}
            bars, by_day = market.bar_index, market.by_day
        else:
            rules = {code: instrument_rule(execution,code) for code in {r['instrument'] for r in market['bars']}}
            bars = {(r["session"], r["instrument"]): r for r in market["bars"]}
            by_day = {day: {} for day in sessions}
            for row in market["bars"]:
                if row["session"] in by_day:by_day[row["session"]][row["instrument"]] = row
            require(len(bars) == len(market["bars"]), "duplicate market bar")
    else:
        require(type(research_context) is ResearchAccountContext and research_context.market is market
                and research_context.execution_hash==digest(execution),'research context market/execution mismatch')
        require(not market.get('investment_assurance') and not market.get('certification_hash')
                and not market.get('event_coverage_certified') and not market.get('tradability_certified'),
                'cached research context cannot carry formal market assurance')
        rules,bars,by_day=research_context.rules,research_context.bars,research_context.by_day
    decisions = {r["execution_session"]: r for r in ir["decisions"]}
    account, daily, orders, gaps = Account(execution["initial_cash"]), [], [], []
    unresolved_codes = set()
    events = (research_context.events if research_context is not None else
              sorted(market["actions"], key=lambda x: timestamp(x["timestamp"])))
    terms = market.get('action_terms', [])
    require(all(t.get('status') not in SUPPORTED or t['record_date'] < t['ex_date'] and
                (not t['gross_cash_per_share'] or t['pay_date'] and t['ex_date']<=t['pay_date']) for t in terms), 'invalid action dates')
    indexed = getattr(market, "_columnar_market", False)
    if indexed:
        from collections import defaultdict
        event_days, ex_days, record_days, pay_days = [defaultdict(list) for _ in range(4)]
        for event in events: event_days[event["timestamp"][:10]].append(event)
        for term in terms:
            ex_days[term["ex_date"]].append(term)
            record_days[term["record_date"]].append(term)
            if term["pay_date"]: pay_days[term["pay_date"]].append(term)
    record_holdings = {}
    entitlements = {}
    initial = {"session": sessions[0]+":initial", "nav": execution["initial_cash"], "cash": execution["initial_cash"],
               "holdings": {}, "receivables": "0", "valuation_identifiable": True}
    for index, day in enumerate(sessions):
        day_events = event_days[day] if indexed else events
        day_ex_terms = ex_days[day] if indexed else terms
        for event in day_events:
            if event["timestamp"][:10] == day and event["timestamp"][11:19] < "09:30:00":
                account.apply(event)
        # Entitlement quantity belongs to record-date closing ownership, even if
        # shares are sold on ex-date before cash is paid. It is never today's holding.
        post_entitlements(account,day_ex_terms,day,record_holdings,entitlements,ir['ir_hash'],execution,gaps,
                          fractional_resolver=fractional_resolver)
        held = {s: sum(lot["quantity"] for lot in lots) for s, lots in account.positions.items() if lots}
        pending=pending_shares(account)
        # P-20261015-004: sorted() makes unknown-event journal/gaps order
        # deterministic across processes (raw set iteration is hash-seed
        # dependent; the journal chain hash feeds account_hash).
        for code in sorted(set(held)|set(pending)):
            bar = bars.get((day, code), {})
            if bar.get("corporate_action_known") is not True and code not in unresolved_codes:
                account.apply({"event_id": "unknown_"+day+code, "kind": "unknown", "timestamp": day+"T09:29:59.999999+08:00",
                               "evidence_id": bar.get("evidence_id", "missing_bar"),
                               "payload": {"instrument": code, "reason": "corporate_action_coverage_unknown"}})
                unresolved_codes.add(code)
                gaps.append({"session": day, "instrument": code, "reason": "corporate_action_coverage_unknown"})
        decision = decisions.get(day)
        if decision is not None:
            require(timestamp(decision["decision_time"]) < timestamp(decision["first_order_time"]), "same-time/future signal trade")
            # Position sizing uses the prior signal close, never today's open or future NAV.
            prior_bars = {s: bars.get((decision["signal_session"], s), {}) for s in set(held) | set(pending) | set(decision["scores"])}
            changed = {e["payload"].get("instrument") for e in day_events
                       if e["timestamp"][:10] == day and e["timestamp"][11:19] < "09:30:00"}
            changed.update(t['instrument'] for t in day_ex_terms if t['ex_date']==day and t['status'] in SUPPORTED)
            # Post-action shares/cash cannot be valued with an unadjusted pre-action close.
            # A certified adapter must supply the ex-action sizing reference and its evidence.
            for code in sorted((set(held) | set(pending) | set(decision["scores"])) & changed):
                bar = bars.get((day,code),{})
                reference = bar.get("sizing_reference_close")
                if reference is None or reference <= 0 or not bar.get("sizing_reference_evidence"):
                    prior_bars[code] = dict(prior_bars[code],close=None)
                    gaps.append({"session":day,"instrument":code,"reason":"action_sizing_reference_missing"})
                else:
                    prior_bars[code] = dict(prior_bars[code],close=reference)
            identifiable = not account.unresolved and all(prior_bars[s].get("close") is not None for s in set(held)|set(pending))
            projected=None
            if restricted is not None:
                sizing_quotes={s:{'price':b['close'],'asof':decision['signal_session']+'T18:00:00+08:00',
                                  'evidence_id':b.get('evidence_id')} for s,b in prior_bars.items() if b.get('close') is not None}
                projected=restricted.project(account,day+'T09:30:00+08:00',sizing_quotes,
                                              quote_cutoff=decision['signal_session']+'T18:00:00+08:00')
                identifiable=projected['nominal_sizing_available']
            if not identifiable:
                gaps.append({"session": day, "reason": "pretrade_NAV_unidentifiable"})
            else:
                if projected is not None:
                    nav=money(projected['conditional_total_nav'])
                else:
                    nav = account.cash + sum((money(str(prior_bars[s]["close"]))*q for s, q in held.items()), Decimal(0))
                    nav += sum((v for v in account.receivables.values() if v is not None), Decimal(0))
                    nav += sum((money(str(prior_bars[s]['close']))*q for s,q in pending.items()),Decimal(0))
                sizing_prices=projected['_prices'] if projected is not None else {s:money(str(prior_bars[s]['close'])) for s in set(held)|set(pending)}
                prior = {s: float(sizing_prices[s]*(held.get(s,0)+pending.get(s,0))/nav) for s in set(held)|set(pending)}
                locked=projected['_active'] if projected is not None else set()
                scores={s:v for s,v in decision['scores'].items() if s not in locked} if projected is not None else decision['scores']
                weights = target_weights(scores, ir["strategy"], prior,
                                         market.get("industries", {}).get(decision["signal_session"]),
                                         frozen_prior=locked & set(prior))
                if weights is not None:
                    desired = {}
                    for code in sorted(set(weights) | set(held)):
                        if code in locked:
                            desired[code]=held.get(code,0)
                            continue
                        close = prior_bars.get(code, {}).get("close")
                        if close is None or close <= 0:
                            gaps.append({"session": day, "instrument": code, "reason": "sizing_close_missing"})
                            desired[code] = held.get(code, 0)
                        else:
                            rule = rules.get(code) or instrument_rule(execution,code)
                            target=desired_shares(nav*money(str(weights.get(code, 0))),close,rule)
                            desired[code] = max(0,target-pending.get(code,0))
                    intents = [(code, desired[code]-held.get(code, 0)) for code in desired if desired[code] != held.get(code, 0)]
                    # Sell first; unfilled cash is not reassigned to other desired weights.
                    for number, (code, delta) in enumerate(sorted(intents, key=lambda x: (x[1] > 0, x[0]))):
                        side, amount = ("buy" if delta > 0 else "sell"), abs(delta)
                        bar = bars.get((day, code), {})
                        reason = None
                        unlimited = bar.get('limit_status')=='unbounded_evidenced' and bool(bar.get('limit_evidence_id'))
                        needed = ("open", "volume", "paused", "corporate_action_known") + (() if unlimited else ("up_limit","down_limit"))
                        # A recorded pause rejects the order without inventing an
                        # opening quote, volume or price limit on that session.
                        if bar.get("paused") is True:
                            reason = "paused"
                        elif any(bar.get(k) is None for k in needed):
                            reason = "market_rule_or_quote_missing"
                        elif not unlimited and (not finite(bar['up_limit']) or not finite(bar['down_limit']) or
                            bar['down_limit']<=0 or bar['up_limit']<bar['down_limit']):
                            reason = "market_rule_or_quote_invalid"
                        elif bar["corporate_action_known"] is not True:
                            reason = "unknown_event"
                        elif bar["open"] <= 0 or bar["volume"] <= 0:
                            reason = "no_liquidity"
                        elif not unlimited and side == "buy" and bar["open"] >= bar["up_limit"]:
                            reason = "limit_up"
                        elif not unlimited and side == "sell" and bar["open"] <= bar["down_limit"]:
                            reason = "limit_down"
                        slip = money(execution["slippage_bps"])*money(str(cost_multiplier))/Decimal(10000)
                        price = money(str(bar["open"]))* (1+slip if side == "buy" else 1-slip) if reason is None else None
                        if price is not None and not rules[code].get('legacy'):
                            tick=money(rules[code]['tick'])
                            price=(price/tick).to_integral_value(rounding=ROUND_CEILING if side=='buy' else ROUND_FLOOR)*tick
                        if price is not None and not unlimited and (price > money(str(bar["up_limit"])) or price < money(str(bar["down_limit"]))):
                            reason = "slipped_price_outside_limits"
                        requested = amount
                        if reason is None:
                            rule = rules[code]
                            lot = rule['buy_step'] if side == 'buy' else rule['sell_step']
                            prior_bar = prior_bars.get(code, {})
                            prior_volume = prior_bar.get("volume")
                            full_halt = (prior_bar.get('paused') is True and prior_bar.get('mark_source') in
                                         {'declared_frozen_last_close_on_vendor_halt','declared_frozen_last_close_on_issuer_halt','declared_ex_cash_reference_on_vendor_halt'})
                            if prior_volume is None and not full_halt:
                                gaps.append({"session": day, "instrument": code, "reason": "lagged_volume_missing"})
                            # Capacity is zero after an evidenced full-day halt;
                            # raw missing signal volume remains missing in the input.
                            raw_capacity = int((prior_volume or 0)*float(execution["participation"])*participation_multiplier)
                            cap = raw_capacity//lot*lot if side=='buy' else raw_capacity
                            amount = min(amount, cap)//lot*lot
                            if not rule.get('legacy') and side == 'buy': amount = buy_quantity(amount,rule,order_type=execution.get('order_type','limit'))
                            if side == "sell":
                                settled = sum(l["quantity"] for l in account.positions.get(code, [])
                                              if l["sellable_at"] <= timestamp(day+"T09:30:00+08:00"))
                                amount = sell_quantity(requested,held.get(code,0),settled,cap,lot,
                                                       execution.get('odd_lot_exit')=='whole_residual')
                                if not rule.get('legacy'):
                                    amount = board_sell_quantity(requested,held.get(code,0),settled,cap,rule,order_type=execution.get('order_type','limit'))
                            commission = money(execution[side+"_commission"])*money(str(cost_multiplier))
                            minimum = money(execution["minimum_commission"])*money(str(cost_multiplier))
                            tax_rate = execution['sell_tax']
                            for effective, rate in execution.get('sell_tax_schedule', []):
                                if effective <= day: tax_rate = rate
                            tax = money(tax_rate)*money(str(cost_multiplier)) if side == "sell" else Decimal(0)
                            if side == "buy":
                                if rule.get('legacy'):
                                    while amount > 0 and price*amount + max(minimum, price*amount*commission) > account.cash:
                                        amount -= lot
                                else:
                                    affordable=int(min((account.cash-minimum)/price,
                                        account.cash/(price*(1+commission))).to_integral_value(rounding=ROUND_FLOOR))
                                    amount=buy_quantity(max(0,min(amount,affordable)),rule,order_type=execution.get('order_type','limit'))
                            if amount <= 0:
                                reason = "prior_full_halt_no_capacity" if full_halt and prior_volume is None else "cash_volume_or_settlement_constraint"
                        order = {"session": day, "instrument": code, "side": side, "requested": requested,
                                 "filled": 0 if reason else amount, "reason": reason,
                                 "price": str(price) if price is not None else None,
                                 "signal_time": decision["decision_time"]}
                        if not rules[code].get('legacy'):
                            order['venue_instrument']=venue_code_at(code,day,execution.get('identity_aliases',{}))
                        orders.append(order)
                        event_id = digest([ir["ir_hash"], cost_multiplier, participation_multiplier, day, number, code, side])
                        if reason is None:
                            fee = max(minimum, price*amount*commission) + price*amount*tax
                            settle_index = index + execution["settlement_sessions"]
                            if side == "buy" and settle_index >= len(sessions):
                                # Last-day holdings stay open; actual future calendar required for later sell.
                                sellable = "9999-12-31T09:30:00+08:00"
                            else:
                                sellable = (sessions[settle_index] if side == "buy" else day)+"T09:30:00+08:00"
                            account.apply({"event_id": event_id, "kind": "fill", "timestamp": day+"T09:30:00+08:00",
                                           "evidence_id": bar["evidence_id"], "payload": {"instrument": code, "side": side,
                                           "quantity": amount, "price": str(price), "fee": str(fee), "sellable_at": sellable}})
                            order["fee"] = str(fee)
                        elif reason in {"market_rule_or_quote_missing", "market_rule_or_quote_invalid", "unknown_event"}:
                            gaps.append({"session": day, "instrument": code, "reason": reason})
        for event in day_events:
            if event["timestamp"][:10] == day and event["timestamp"][11:19] >= "09:30:00":
                account.apply(event)
        for term in record_days[day] if indexed else terms:
            if term['record_date'] == day:
                record_holdings[term['evidence_id']] = sum(l['quantity'] for l in account.positions.get(term['instrument'], []))
        for term in pay_days[day] if indexed else terms:
            if term['pay_date'] == day and term['evidence_id'] in entitlements:
                entitlement = entitlements[term['evidence_id']]
                account.apply({'event_id':entitlement+':payment','kind':'payment','timestamp':day+'T18:00:00+08:00',
                    'evidence_id':term['evidence_id'],'payload':{'entitlement_id':entitlement}})
        when = day+"T18:00:00+08:00"
        quotes = {} if getattr(market, "_columnar_market", False) else {
            code: {"price": str(bar["close"]), "asof": when, "evidence_id": bar["evidence_id"]}
            for code, bar in by_day[day].items() if bar.get("close") is not None and bar["close"] > 0}
        if getattr(market, "_columnar_market", False):
            from .columnar import account_quotes
            quotes = account_quotes(market, account, day)
        marked = account.mark(when, quotes)
        if restricted is not None:
            marked['restricted_research']=public_mark(restricted.project(account,when,quotes))
        marked["session"] = day
        daily.append(marked)
    identifiable = all(row["valuation_identifiable"] for row in daily) and not gaps
    # Account evidence is only formal when the certificate that authorised the market
    # block is the one bound into this market: a flag copied from elsewhere, or one
    # whose binding hash no longer matches, is not evidence.
    review=market.get('investment_assurance') or {}
    # P-20261015-012: annotation fields (certification write-back + assurance
    # binding) are excluded from the content digest, matching the certificate
    # and assurance market_digest computation.
    source_digest=digest({k:v for k,v in market.items() if k not in MARKET_CONTENT_EXCLUSIONS}) if research_context is None else None
    certified = (review.get('schema_version')=='fuxi.market_assurance.1'
                 and review.get('snapshot_hash')==ir['snapshot_hash']
                 and review.get('market_digest')==source_digest
                 and bool(review.get('receipt_id')) and bool(review.get('assurance_hash'))
                 and market.get("event_coverage_certified") is True
                 and market.get("tradability_certified") is True
                 and market.get("certification_hash") is not None
                 and market.get("certification_snapshot_hash") is not None)
    return {"schema_version": "fuxi.account_run.1", "ir_hash": ir["ir_hash"], "market_hash": market["market_hash"],
            "eval_status": "valid" if identifiable else "data_insufficient", "daily": [initial]+daily,
            "orders": orders, "journal": account.journal, "gaps": gaps,
            "cost_multiplier": cost_multiplier, "participation_multiplier": participation_multiplier,
            "formal_evidence": identifiable and certified and fractional_resolver is None and restricted is None,
            "investment_assurance":review,
            "certification_hash": market.get("certification_hash"),
            "certification_binding": {"event_coverage_certified": market.get("event_coverage_certified"),
                                      "tradability_certified": market.get("tradability_certified"),
                                      "snapshot_hash": market.get("certification_snapshot_hash")},
            "assumptions": execution,
            **({'restricted_policy_hash':restricted_policy['policy_hash'],'restricted_scenario':restricted_scenario,
                'restricted_evidence_scope':'conditional_research_only','formal_eligibility':False} if restricted is not None else {})}
