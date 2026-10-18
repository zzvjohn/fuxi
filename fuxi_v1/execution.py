"""Durable paper gateway: intent first, fencing, unknown outcomes and fill deduplication."""

import json
import math
from decimal import Decimal, InvalidOperation
from .contracts import digest, require
from .evidence import now


class PaperGateway:
    def __init__(self, store):
        self.store = store
        self.store.db.executescript('''
          CREATE TABLE IF NOT EXISTS fencing(account TEXT PRIMARY KEY,token INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS orders(id TEXT PRIMARY KEY,account TEXT,token INTEGER,state TEXT,payload TEXT);
          CREATE TABLE IF NOT EXISTS fills(id TEXT PRIMARY KEY,order_id TEXT,payload TEXT);
        ''')

    def _interlock(self,account):
        exists=self.store.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='execution_health'").fetchone()
        if exists:
            health=self.store.db.execute('SELECT state FROM execution_health WHERE account=?',(account,)).fetchone()
            require(health is not None and health[0]=='ready','execution health interlock halted')

    def take_ownership(self, account):
        with self.store.transaction():
            row = self.store.db.execute("SELECT token FROM fencing WHERE account=?", (account,)).fetchone()
            token = row[0]+1 if row else 1
            self.store.db.execute("INSERT INTO fencing VALUES(?,?) ON CONFLICT(account) DO UPDATE SET token=excluded.token", (account, token))
            if self.store.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='execution_health'").fetchone():
                self.store.db.execute("UPDATE execution_health SET state='halted',reason='startup_reconciliation_required',evidence='{}' WHERE account=?",(account,))
            self.store.append("fencing", {"account": account, "token": token, "at": now()})
            return token

    def intent(self, account, token, payload, activation_receipt):
        require(payload.get("mode") == "paper", "gateway has no live trading authority")
        require(type(payload.get("shares")) is int and payload["shares"] > 0, "positive integer order shares required")
        # A caller key identifies the economic instruction across release/restart.
        # Legacy callers retain content identity, but cannot use that path in the controller.
        key = payload.get("client_order_id") or digest(payload)
        require(isinstance(key, str) and key.strip(), "invalid client order identity")
        identity = digest([account, key])
        with self.store.transaction():
            self.store.consume_receipt(activation_receipt, "paper_activation")
            row = self.store.db.execute("SELECT token FROM fencing WHERE account=?", (account,)).fetchone()
            require(row and row[0] == token, "stale execution fencing token")
            self._interlock(account)
            old = self.store.db.execute("SELECT payload FROM orders WHERE id=?", (identity,)).fetchone()
            if old:
                require(json.loads(old[0])["intent"] == payload, "conflicting client order identity")
                return identity
            self.store.db.execute("INSERT INTO orders VALUES(?,?,?,?,?)",
                                  (identity, account, token, "intent", json.dumps({"intent":payload,"activation":activation_receipt}, sort_keys=True)))
            self.store.append("order_intent", {"id": identity, "account": account, "payload": payload,
                                               "token": token, "activation": activation_receipt})
        return identity

    def transition(self, order_id, token, state, *, reconciliation=False):
        allowed = {"intent": {"submitted", "rejected"}, "submitted": {"acknowledged", "unknown", "rejected"},
                   "unknown": {"acknowledged", "filled", "partially_filled", "cancelled", "rejected"},
                   "acknowledged": {"partially_filled", "filled", "cancelled", "unknown"},
                   "partially_filled": {"partially_filled", "filled", "cancelled", "unknown"}}
        with self.store.transaction():
            row = self.store.db.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
            require(row is not None, "unknown order")
            current = self.store.db.execute("SELECT token FROM fencing WHERE account=?", (row["account"],)).fetchone()[0]
            require(token == current, "stale execution fencing token")
            if state == "submitted":
                self._interlock(row['account'])
                self.store.consume_receipt(json.loads(row["payload"])["activation"], "paper_activation")
            require(state in allowed.get(row["state"], set()), "illegal order transition; do not blindly retry")
            require(row["state"] != "unknown" or reconciliation, "unknown outcome requires broker reconciliation")
            self.store.db.execute("UPDATE orders SET state=?,token=? WHERE id=?", (state, token, order_id))
            self.store.append("order_state", {"order_id": order_id, "from": row["state"], "to": state,
                                              "token": token, "reconciled": reconciliation, "at": now()})

    def fill_receipt(self, fill_id, order_id, payload):
        require(type(payload.get("shares")) is int and payload["shares"] > 0,"positive fill shares required")
        with self.store.transaction():
            old = self.store.db.execute("SELECT order_id,payload FROM fills WHERE id=?", (fill_id,)).fetchone()
            raw = json.dumps(payload, sort_keys=True, allow_nan=False)
            if old:
                require(old["order_id"] == order_id and old["payload"] == raw, "conflicting duplicate fill")
                return False
            row = self.store.db.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
            require(row is not None and row["state"] in {"acknowledged", "partially_filled", "unknown"}, "fill without acknowledged/reconcilable order")
            intent = json.loads(row["payload"])["intent"]
            if "account" in payload:
                require(payload["account"] == row["account"], "fill account mismatch")
            if "instrument" in payload:
                require(payload["instrument"] == intent.get("instrument"), "fill instrument mismatch")
            filled = sum(json.loads(r[0])["shares"] for r in self.store.db.execute("SELECT payload FROM fills WHERE order_id=?",(order_id,)))
            require(filled+payload["shares"] <= intent["shares"],"fill exceeds requested shares")
            # Record late facts even after revocation; revocation stops new orders,
            # not truthful reconciliation of fills already executed.
            self.store.db.execute("INSERT INTO fills VALUES(?,?,?)", (fill_id, order_id, raw))
            self.store.append("fill_receipt", {"fill_id": fill_id, "order_id": order_id, "payload": payload})
        return True


def reconcile_rows(expected, observed, fields, tolerance=1e-8):
    require(expected and observed and len(expected) == len(observed), "nonempty same-length parity rows required")
    require(fields and len(set(fields)) == len(fields) and type(tolerance) in (int,float) and math.isfinite(tolerance) and tolerance >= 0,"parity field set and tolerance invalid")
    require(len({r["key"] for r in expected}) == len(expected),"duplicate parity key")
    require(len({r["key"] for r in observed}) == len(observed),"duplicate observed parity key")
    def equal(x, y, *, nullable=False):
        if x is None or y is None:
            return nullable and x is None and y is None
        if type(x) in (int, float) and type(y) in (int, float):
            return math.isfinite(x) and math.isfinite(y) and abs(x-y) <= tolerance
        if type(x) is not type(y):
            return False
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(equal(x[k],y[k],nullable=nullable) for k in x)
        if isinstance(x, list):
            return len(x) == len(y) and all(equal(a,b,nullable=nullable) for a,b in zip(x,y))
        return x == y
    differences = []
    for a, b in zip(expected, observed):
        require(a["key"] == b["key"], "parity keys/order mismatch")
        for name in fields:
            require(name in a and name in b, "parity field missing")
            x, y = a[name], b[name]
            if name in {"cash","receivables","nav"}:
                def financial(v):
                    if v is None or isinstance(v,bool):return None
                    try:
                        number=Decimal(str(v))
                        return number if number.is_finite() else None
                    except InvalidOperation:return None
                nx,ny=financial(x),financial(y)
                matched=nx is not None and ny is not None and abs(nx-ny)<=Decimal(str(tolerance))
            else:
                matched=equal(x,y,nullable=name == "signal")
            # Missing signals may legitimately mean hold; unknown account facts
            # cannot establish parity even when both sources say null.
            if not matched:
                differences.append({"key": a["key"], "field": name, "expected": x, "observed": y})
    return {"n_pairs": len(expected), "accepted": not differences, "differences": differences,
            "fields": fields, "tolerance": tolerance}
