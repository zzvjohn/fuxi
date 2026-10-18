"""Decimal execution reference with integer distributions and listing locks.

Recomputes from targets, quotes, limits, halts and validated events, without
reading the primary journal, orders, cash, or positions to determine any trade.
"""
from decimal import Decimal, ROUND_FLOOR

D = lambda x: Decimal(str(x))


def simulate(days, targets, quotes, limits, halts, actions, initial=10000000, bps=15):
    cash, due, nav = D(initial), D(0), D(initial)
    owned, marks, plan, entitlement = {}, {}, {}, {}
    seen, payments, locks = set(), [], []
    output, fills = [], []
    rate = D(bps) / 10000

    def lot(code, value):
        minimum, step = (200, 1) if code.startswith(('688', '689')) else (100, 1) if code.endswith('.BJ') else (100, 100)
        count = int((D(value) / step).to_integral_value(rounding=ROUND_FLOOR)) * step
        return count if count >= minimum else 0

    by_ex, by_record = {}, {}
    for action in actions:
        by_ex.setdefault(action['ex_date'], []).append(action)
        by_record.setdefault(action['record_date'], []).append(action)
    for day in days:
        new_targets = sorted(t for t in targets if t < day and t not in seen)
        if new_targets:
            stamp = new_targets[-1]
            seen.update(new_targets)
            weights = targets[stamp]
            plan = dict(owned) if weights is None else {
                c: lot(c, nav * D(w) / (1 + rate) / D(quotes[(stamp, c)]['close']))
                for c, w in weights.items() if w > 0}
        for action in by_ex.get(day, []):
            code = action['code']
            entitled = entitlement.get(action['event_id'], 0)
            extra = D(entitled) * (D(action['share_multiplier']) - 1)
            integral = extra.to_integral_value()
            if abs(extra - integral) > D('0.00000001'):
                raise ValueError('fractional_entitlement_has_no_cash_in_lieu_evidence')
            added = int(integral)
            owned[code] = owned.get(code, 0) + added
            amount = D(entitled) * D(action['cash_per_share_net'])
            due += amount
            payments.append((action['pay_date'], amount))
            if added:
                locks.append((action['new_shares_date'], code, added))
            if code in plan:
                plan[code] = int((D(plan[code]) * D(action['share_multiplier'])).to_integral_value(rounding=ROUND_FLOOR))
        pending = []
        for when, amount in payments:
            if when <= day:
                cash += amount
                due -= amount
            else:
                pending.append((when, amount))
        payments = pending
        locks = [(when, c, q) for when, c, q in locks if when > day]
        for buying in (False, True):
            for code in sorted(set(owned) | set(plan)):
                held = owned.get(code, 0)
                delta = plan.get(code, 0) - held
                if not delta or (delta > 0) != buying or (day, code) in halts:
                    continue
                quote, boundary = quotes[(day, code)], limits[(day, code)]
                price = D(quote['open'])
                assert price > 0
                if buying and price >= D(boundary['up_limit']) - D('0.000001'):
                    continue
                if not buying and price <= D(boundary['down_limit']) + D('0.000001'):
                    continue
                quantity = abs(delta)
                if buying:
                    affordable = int(((cash + D('0.00000001')) / (price * (1 + rate))).to_integral_value(rounding=ROUND_FLOOR))
                    quantity = lot(code, min(quantity, affordable))
                else:
                    locked = sum(q for _, c, q in locks if c == code)
                    quantity = min(quantity, max(0, held - locked))
                    if quantity != held:
                        quantity = lot(code, quantity)
                if quantity == 0:
                    continue
                signed = quantity if buying else -quantity
                fee = D(quantity) * price * rate
                cash -= D(signed) * price + fee
                owned[code] = held + signed
                assert cash >= D('-0.000001') and owned[code] >= 0
                fills.append({'date': day, 'code': code, 'quantity': signed, 'price': float(price), 'fee': float(fee)})
        positions = {c: q for c, q in owned.items() if q}
        for code in positions:
            if (day, code) not in halts:
                marks[code] = D(quotes[(day, code)]['close'])
            assert code in marks and marks[code] > 0
        nav = cash + due + sum(D(q) * marks[c] for c, q in positions.items())
        output.append({'date': day, 'cash': float(cash), 'receivable': float(due),
                       'nav': float(nav), 'positions': positions,
                       'locked': {c: sum(q for _, symbol, q in locks if c == symbol) for c in positions}})
        for action in by_record.get(day, []):
            entitlement[action['event_id']] = owned.get(action['code'], 0)
    return output, fills
