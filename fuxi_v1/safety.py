"""Durable execution interlock. Actual live routing is deliberately unsupported."""
import json
from decimal import Decimal, InvalidOperation
from .contracts import require, timestamp, digest, ContractError
from .execution import PaperGateway, reconcile_rows


def known_nonnegative(value):
    if isinstance(value,bool) or value is None:
        return False
    try:
        number = Decimal(str(value))
        return number.is_finite() and number >= 0
    except (InvalidOperation,ValueError):
        return False


class ExecutionController:
    def __init__(self, store, account, activation, *, max_age_seconds=300):
        require(isinstance(account,str) and account.strip(),"account required")
        require(type(max_age_seconds) is int and max_age_seconds > 0,"positive freshness budget required")
        self.store,self.account,self.activation = store,account,activation
        self.max_age_seconds = max_age_seconds
        self.gateway = PaperGateway(store)
        store.db.execute("CREATE TABLE IF NOT EXISTS execution_health(account TEXT PRIMARY KEY,state TEXT,reason TEXT,evidence TEXT)")
        store.db.execute("INSERT OR IGNORE INTO execution_health VALUES(?,?,?,?)",(account,'halted','startup_reconciliation_required','{}'))
        self.token = self.gateway.take_ownership(account)

    def halt(self, reason, evidence):
        with self.store.transaction():
            self.store.db.execute("UPDATE execution_health SET state='halted',reason=?,evidence=? WHERE account=?",
                                  (reason,json.dumps(evidence,sort_keys=True,allow_nan=False),self.account))
            self.store.append('execution_halt',{'account':self.account,'reason':reason,'evidence':evidence})

    def guard(self, health, now):
        reasons=[]
        try:
            current=timestamp(now)
            require(current.utcoffset() is not None,"explicit execution clock required")
        except (ValueError,TypeError):
            self.halt('invalid_execution_clock',{'clock_repr':repr(now)})
            return {'accepted':False,'blockers':['invalid_execution_clock']}
        if health.get('account') != self.account:reasons.append('account_mismatch')
        for name in ('nav','cash','receivables'):
            if not known_nonnegative(health.get(name)):reasons.append('unknown_'+name)
        for flag in ('account_matched','held_marks_known','events_resolved','data_contract_active'):
            if health.get(flag) is not True:reasons.append(flag+'_not_established')
        try:
            age=(timestamp(now)-timestamp(health['asof'])).total_seconds()
            if not 0 <= age <= self.max_age_seconds:reasons.append('stale_or_future_data')
        except (KeyError,ValueError,TypeError):reasons.append('invalid_data_clock')
        if self.store.db.execute("SELECT 1 FROM orders WHERE account=? AND state IN ('submitted','unknown')",(self.account,)).fetchone():
            reasons.append('unresolved_order_outcome')
        try:self.store.consume_receipt(self.activation,'paper_activation')
        except ContractError:reasons.append('inactive_release')
        if reasons:
            # An invalid numeric fact (NaN/Inf/Decimal) must not crash incident
            # serialization before the durable stop is recorded.
            self.halt(','.join(reasons),{'health_hash':digest({'repr':repr(health)}),'checked_at':now})
        return {'accepted':not reasons,'blockers':reasons}

    def reconcile_and_resume(self, expected, observed, health, now):
        self.halt('reconciliation_in_progress',{'expected_rows':len(expected),'observed_rows':len(observed)})
        result=reconcile_rows(expected,observed,['holdings','cash','receivables','nav'])
        require(all(r.get('account') == self.account for r in expected+observed),'reconciliation account mismatch')
        require(expected[-1]['key'] == health.get('asof'),'health/reconciliation clock mismatch')
        require(all(observed[-1][k] == health.get(k) for k in ('cash','receivables','nav')),
                'health/reconciliation values mismatch')
        verdict=self.guard(health,now)
        if not result['accepted']:
            self.halt('account_parity_mismatch',{'comparison':result})
        require(result['accepted'] and verdict['accepted'],'reconciliation did not resolve execution blockers')
        with self.store.transaction():
            self.store.db.execute("UPDATE execution_health SET state='ready',reason='',evidence=? WHERE account=?",
                                  (json.dumps({'comparison':result,'health':health},sort_keys=True),self.account))
            self.store.append('execution_resumed',{'account':self.account,'token':self.token,'health':health,'comparison':result})
        return result

    def order(self,payload,health,now):
        verdict=self.guard(health,now)
        row=self.store.db.execute('SELECT state FROM execution_health WHERE account=?',(self.account,)).fetchone()
        require(verdict['accepted'] and row[0]=='ready','execution halted; reconcile before new orders')
        require(payload.get('client_order_id'),'stable economic instruction key required')
        return self.gateway.intent(self.account,self.token,payload,self.activation)
