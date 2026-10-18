"""Fixed-share cash/receivable ledger for explicit evidenced events.

Not an exchange simulator. Settlements and entitlements are supplied evidence;
unsupported/unknown events cannot produce identifiable NAV.
"""

from copy import deepcopy, copy
from decimal import Decimal, InvalidOperation
from .contracts import digest, require, nonempty, keys, timestamp


def money(value):
    require(isinstance(value, str), "account decimals must be strings")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal") from exc
    require(number.is_finite(), "nonfinite decimal")
    return number


class Account:
    def __init__(self, initial_cash):
        self.cash = money(initial_cash)
        require(self.cash >= 0, "negative initial cash")
        self.initial_cash = self.cash
        self.positions = {}
        self.receivables = {}
        self.share_receivables = {}
        self.events = {}
        self.journal = []
        self.clock = None
        self.unresolved = set()
        self.journal_hash = digest({'initial_cash':initial_cash})

    def apply(self, event):
        keys(event, ("event_id", "kind", "timestamp", "evidence_id", "payload"))
        require(nonempty(event["event_id"]) and nonempty(event["evidence_id"]), "event identity/evidence required")
        event_hash = digest(event)
        if event["event_id"] in self.events:
            require(self.events[event["event_id"]] == event_hash, "conflicting duplicate event")
            return "already_applied"
        when = timestamp(event["timestamp"])
        require(self.clock is None or when >= self.clock, "out-of-order ledger event")
        # Validate/mutate a copy: failures never leave partially posted journal entries.
        # Past journal entries are immutable. Copy the transaction's mutable state,
        # not the entire growing journal on every fill (quadratic deep-copy cost).
        trial = copy(self)
        trial.positions = deepcopy(self.positions)
        trial.receivables = self.receivables.copy()
        trial.share_receivables = deepcopy(self.share_receivables)
        trial.events = self.events.copy()
        trial.journal = self.journal.copy()
        trial.unresolved = self.unresolved.copy()
        trial._post(event, when)
        trial.clock = when
        trial.events[event["event_id"]] = event_hash
        trial.journal.append(deepcopy(event))
        trial.journal_hash = digest([self.journal_hash,event_hash])
        self.__dict__.update(trial.__dict__)
        return "applied"

    def _post(self, event, when):
        kind, p = event["kind"], event["payload"]
        if kind == "fill":
            keys(p, ("instrument", "side", "quantity", "price", "fee", "sellable_at"))
            code, quantity = p["instrument"], p["quantity"]
            require(nonempty(code) and type(quantity) is int and quantity > 0, "invalid fill quantity")
            require(p["side"] in {"buy", "sell"}, "invalid fill side")
            price, fee, sellable = money(p["price"]), money(p["fee"]), timestamp(p["sellable_at"])
            require(price > 0 and fee >= 0, "invalid fill price/fee")
            if p["side"] == "buy":
                require(sellable >= when, "settlement before buy")
                amount = price*quantity + fee
                require(self.cash >= amount, "insufficient cash")
                self.cash -= amount
                self.positions.setdefault(code, []).append({"quantity": quantity, "sellable_at": sellable})
            else:
                require(sellable <= when, "sell fill has future settlement availability")
                lots = self.positions.get(code, [])
                require(sum(lot["quantity"] for lot in lots if lot["sellable_at"] <= when) >= quantity,
                        "insufficient settled shares")
                remaining = quantity
                for lot in lots:
                    if lot["sellable_at"] <= when:
                        take = min(remaining, lot["quantity"])
                        lot["quantity"] -= take
                        remaining -= take
                self.positions[code] = [lot for lot in lots if lot["quantity"] > 0]
                self.cash += price*quantity - fee
                require(self.cash >= 0, "fee exceeds available cash")
        elif kind == "unfilled":
            keys(p, ("instrument", "reason"))
            require(nonempty(p["instrument"]) and nonempty(p["reason"]), "unfilled reason required")
            # An unfilled exit preserves both shares and the full valuation denominator.
        elif kind == "receivable":
            keys(p, ("entitlement_id", "amount"))
            amount = money(p["amount"])
            require(nonempty(p["entitlement_id"]) and amount >= 0 and
                    p["entitlement_id"] not in self.receivables, "duplicate/invalid entitlement")
            self.receivables[p["entitlement_id"]] = amount
        elif kind == "payment":
            keys(p, ("entitlement_id",))
            require(p["entitlement_id"] in self.receivables and self.receivables[p["entitlement_id"]] is not None,
                    "unknown/already paid entitlement")
            self.cash += self.receivables[p["entitlement_id"]]
            self.receivables[p["entitlement_id"]] = None
        elif kind == 'share_entitlement':
            keys(p,('entitlement_id','instrument','quantity','sellable_at'))
            require(p['entitlement_id'] not in self.share_receivables,'duplicate share entitlement')
            require(type(p['quantity']) is int and p['quantity']>0 and nonempty(p['instrument']),'invalid share entitlement')
            require(timestamp(p['sellable_at'])>=when,'share availability before entitlement')
            self.share_receivables[p['entitlement_id']]=dict(p)
        elif kind == 'share_credit':
            keys(p,('entitlement_id',))
            claim=self.share_receivables.get(p['entitlement_id'])
            require(claim is not None,'unknown/already credited shares')
            require(timestamp(claim['sellable_at'])<=when,'share credit before availability')
            self.positions.setdefault(claim['instrument'],[]).append({'quantity':claim['quantity'],
                'sellable_at':timestamp(claim['sellable_at'])})
            self.share_receivables[p['entitlement_id']]=None
        elif kind == "split":
            keys(p, ("instrument", "multiplier", "sellable_at"))
            multiplier, sellable = money(p["multiplier"]), timestamp(p["sellable_at"])
            require(multiplier >= 1 and sellable <= when, "only immediate evidenced integral splits supported")
            require(p["instrument"] in self.positions, "split has no holdings")
            for lot in self.positions[p["instrument"]]:
                quantity = multiplier * lot["quantity"]
                require(quantity == quantity.to_integral_value(), "fractional shares require external evidence")
                lot["quantity"] = int(quantity)
                lot["sellable_at"] = max(lot["sellable_at"], sellable)
        elif kind == "security_transfer":
            keys(p, ("from_instrument","to_instrument","quantity","sellable_at","resolved_event_ids"))
            source,target=p['from_instrument'],p['to_instrument']
            require(nonempty(source) and nonempty(target) and source != target,'invalid security transfer identities')
            quantity=p['quantity'];lots=self.positions.get(source,[])
            require(type(quantity) is int and quantity>0 and quantity==sum(l['quantity'] for l in lots),
                    'transfer requires evidenced full source quantity; no inferred conversion ratio')
            require(not any(c is not None and c['instrument']==source for c in self.share_receivables.values()),
                    'unsettled source entitlements require separate transfer evidence')
            sellable=timestamp(p['sellable_at'])
            require(sellable >= when and sellable >= max(l['sellable_at'] for l in lots),'transfer cannot accelerate share settlement')
            resolved=p['resolved_event_ids']
            require(isinstance(resolved,list) and len(set(resolved))==len(resolved) and set(resolved)<=self.unresolved,
                    'transfer resolution must reference existing unresolved events')
            require(all(self.events.get(e) and next(j for j in self.journal if j['event_id']==e)['payload']['instrument']==source
                        for e in resolved),'transfer cannot resolve another instrument event')
            self.positions.pop(source)
            self.positions.setdefault(target,[]).append({'quantity':quantity,'sellable_at':sellable})
            self.unresolved.difference_update(resolved)
        elif kind == "unknown":
            keys(p, ("instrument", "reason"))
            require(nonempty(p["instrument"]) and nonempty(p["reason"]), "unknown event evidence required")
            self.unresolved.add(event["event_id"])
        else:
            raise ValueError("unsupported account event: " + str(kind))

    def mark(self, when, prices):
        clock = timestamp(when)
        require(self.clock is None or clock >= self.clock, "cannot mark before latest event")
        holdings = {s: sum(lot["quantity"] for lot in lots) for s, lots in self.positions.items()}
        holdings = {s: q for s, q in holdings.items() if q > 0}
        pending={}
        for claim in self.share_receivables.values():
            if claim is not None:pending[claim['instrument']]=pending.get(claim['instrument'],0)+claim['quantity']
        missing, market_value = [], Decimal(0)
        for code in sorted(set(holdings)|set(pending)):  # P-20261015-004 deterministic missing_marks order
            quantity=holdings.get(code,0)+pending.get(code,0)
            quote = prices.get(code)
            if quote is None:
                missing.append(code)
                continue
            keys(quote, ("price", "asof", "evidence_id"))
            require(timestamp(quote["asof"]) == clock and nonempty(quote["evidence_id"]),
                    "mark requires explicit same-cutoff evidence, no silent price carry")
            price = money(quote["price"])
            require(price > 0, "invalid mark")
            market_value += price*quantity
        receivable = sum((v for v in self.receivables.values() if v is not None), Decimal(0))
        identifiable = not missing and not self.unresolved
        return dict(eval_status="valid" if identifiable else "data_insufficient",
                    observation_cutoff=when, valuation_identifiable=identifiable,
                    nav=str(self.cash+market_value+receivable) if identifiable else None,
                    cash=str(self.cash), receivables=str(receivable), holdings=holdings,pending_shares=pending,
                    missing_marks=missing, unresolved_events=sorted(self.unresolved),
                    account_hash=digest({"initial_cash": str(self.initial_cash), "journal_chain": self.journal_hash}),
                    formal_eligibility=False)
