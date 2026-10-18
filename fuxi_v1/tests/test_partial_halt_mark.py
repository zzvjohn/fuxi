"""Partial halts may carry observed daily marks while all-day and unbound marks fail."""
import unittest
import json
from copy import deepcopy
from fuxi_v1.halt_evidence import partial_halt_mark,valid_partial_halt_mark,partial_halt_intervals
from fuxi_v1.certification import certify_tradability
from fuxi_v1.tests.test_certification import fixture_payload


class PartialHaltMarkTests(unittest.TestCase):
    def fixture(self):
        p=fixture_payload();b=p['market']['bars'][0];day,code=b['session'],b['instrument']
        records=[{'ts_code':'000001.SZ','trade_date':day.replace('-',''),'suspend_type':'R','suspend_timing':None},
            {'ts_code':'000001.SZ','trade_date':day.replace('-',''),'suspend_type':'S','suspend_timing':'11:07-11:17,11:22-13:02'}]
        p['snapshot']['provenance']['source_hashes']={'suspend_d.csv':'source-halt','daily_prices.csv':'source-price'}
        p['snapshot']['columns']['close'][0][0]=b['close'];p['snapshot']['columns']['volume'][0][0]=b['volume']
        b.update(paused=True,mark_source='observed_close_on_vendor_partial_halt')
        b['partial_halt_mark']=partial_halt_mark(records,code,day,b['close'],b['volume'],'source-halt','source-price')
        return p,b,records

    def test_source_bound_intraday_mark_valid_with_order_gate_still_paused(self):
        p,b,_=self.fixture()
        self.assertTrue(valid_partial_halt_mark(b,p['snapshot']))
        self.assertTrue(certify_tradability(p['snapshot'],p['market'])['value'])
        self.assertIs(b['paused'],True)

    def test_frozen_json_round_trip_preserves_source_bound_mark(self):
        p,_,_=self.fixture()
        frozen=json.loads(json.dumps(p))
        self.assertTrue(valid_partial_halt_mark(frozen['market']['bars'][0],frozen['snapshot']))
        self.assertTrue(certify_tradability(frozen['snapshot'],frozen['market'])['value'])

    def test_unbound_or_mismatched_mark_still_refused(self):
        p,b,_=self.fixture()
        for target,key,value in [('proof','records_hash','changed'),('proof','instrument','000002.XSHE'),
                ('proof','price_source_sha256','different'),('bar','close',11.),('bar','volume',0.)]:
            with self.subTest(key=key):
                q=deepcopy(p);r=q['market']['bars'][0]
                (r['partial_halt_mark'] if target=='proof' else r)[key]=value
                self.assertFalse(valid_partial_halt_mark(r,q['snapshot']))
                self.assertIn('halted_member_cells_with_observed_mark:1',certify_tradability(q['snapshot'],q['market'])['blocking_gaps'])

    def test_snapshot_quote_or_source_mutation_refused(self):
        p,b,_=self.fixture()
        p['snapshot']['columns']['close'][0][0]=9.
        self.assertFalse(valid_partial_halt_mark(b,p['snapshot']))
        p,b,_=self.fixture();p['snapshot']['provenance']['source_hashes']['suspend_d.csv']='changed'
        self.assertFalse(valid_partial_halt_mark(b,p['snapshot']))

    def test_full_day_and_bad_timing_never_get_partial_mark(self):
        p,b,records=self.fixture()
        for timing in [None,'','09:30-15:00','09:30-11:30,13:00-15:00','11:07-11:07','25:00-26:00','11:67-12:00','garbage','08:00-09:00']:
            with self.subTest(timing=timing):
                rows=deepcopy(records);rows[1]['suspend_timing']=timing
                self.assertIsNone(partial_halt_mark(rows,b['instrument'],b['session'],b['close'],b['volume'],'s','p'))

    def test_single_digit_hour_timing_accepted(self):
        p,b,records=self.fixture()
        rows=deepcopy(records);rows[1]['suspend_timing']='9:30-9:40'
        proof=partial_halt_mark(rows,b['instrument'],b['session'],b['close'],b['volume'],'source-halt','source-price')
        self.assertIsNotNone(proof)
        self.assertEqual(proof['intervals'],[[570,580]])
        b['partial_halt_mark']=proof
        self.assertTrue(valid_partial_halt_mark(b,p['snapshot']))

    def test_wrong_security_or_session_is_not_timing_evidence(self):
        p,b,records=self.fixture()
        for key,value in [('ts_code','000002.SZ'),('trade_date','20210105')]:
            rows=deepcopy(records);rows[1][key]=value
            self.assertIsNone(partial_halt_intervals(rows,b['instrument'],b['session']))

    def test_legacy_all_day_mark_and_missing_proof_remain_fail_closed(self):
        p,b,_=self.fixture();b.pop('partial_halt_mark')
        self.assertFalse(certify_tradability(p['snapshot'],p['market'])['value'])
        b['mark_source']='declared_frozen_last_close_on_vendor_halt'
        self.assertTrue(certify_tradability(p['snapshot'],p['market'])['value'])
        b['mark_source']='observed_close';b['paused']=None
        self.assertIn('member_cells_with_ambiguous_pause_state:1',certify_tradability(p['snapshot'],p['market'])['blocking_gaps'])


if __name__=='__main__':unittest.main()
