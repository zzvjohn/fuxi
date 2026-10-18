from pathlib import Path
import unittest
from fuxi_v1.evidence import EvidenceStore
from fuxi_v1.safety import ExecutionController
from fuxi_v1.contracts import ContractError
from fuxi_v1.tests.test_kernel import scratch_directory

def setup(store):
    receipt=store.receipt('paper_activation',{'isolated_fixture':True})
    c=ExecutionController(store,'test',receipt)
    h={'account':'test','asof':'2026-10-02T10:00:00+08:00','nav':1000.,'cash':1000.,'receivables':0.,
       'account_matched':True,'held_marks_known':True,'events_resolved':True,'data_contract_active':True}
    row={'account':'test','key':h['asof'],'holdings':{},'cash':1000.,'receivables':0.,'nav':1000.}
    c.reconcile_and_resume([row],[row],h,h['asof'])
    return c,h,row,receipt

def halted(store):return store.db.execute("SELECT state FROM execution_health WHERE account='test'").fetchone()[0]=='halted'

class IncidentTests(unittest.TestCase):
    def test_nonfinite_amount_records_durable_stop(self):
        with scratch_directory() as root:
            s=EvidenceStore(Path(root)/'db.sqlite');c,h,row,_=setup(s)
            for bad in (float('nan'),float('inf'),float('-inf')):
                self.assertFalse(c.guard(dict(h,nav=bad),h['asof'])['accepted']);self.assertTrue(halted(s))
                c.reconcile_and_resume([row],[row],h,h['asof'])
            s.close()
    def test_invalid_execution_clock_records_durable_stop(self):
        with scratch_directory() as root:
            s=EvidenceStore(Path(root)/'db.sqlite');c,h,_,_=setup(s)
            self.assertFalse(c.guard(h,None)['accepted']);self.assertTrue(halted(s));s.close()
    def test_restart_from_ready_requires_fresh_reconciliation(self):
        with scratch_directory() as root:
            path=Path(root)/'db.sqlite';s=EvidenceStore(path);_,h,row,receipt=setup(s);s.close()
            s=EvidenceStore(path);c=ExecutionController(s,'test',receipt)
            self.assertTrue(halted(s))
            with self.assertRaises(ContractError):c.order({'mode':'paper','shares':100,'client_order_id':'x'},h,h['asof'])
            c.reconcile_and_resume([row],[row],h,h['asof']);self.assertFalse(halted(s));s.close()
    def test_malformed_reconciliation_cannot_leave_previous_ready_state(self):
        with scratch_directory() as root:
            s=EvidenceStore(Path(root)/'db.sqlite');c,h,row,_=setup(s)
            with self.assertRaises(ContractError):c.reconcile_and_resume([row],[dict(row,account='other')],h,h['asof'])
            self.assertTrue(halted(s));s.close()
