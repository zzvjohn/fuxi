import json
from pathlib import Path
import unittest
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.tests import test_halt_reference
from fuxi_v1.halt_evidence import extract_full_halt_start, load_halt_starts
from fuxi_v1.historical_adapter import build_history_input
from fuxi_v1.datasets import file_hash
from fuxi_v1.contracts import ContractError
import pandas as pd


class IssuerHaltTests(unittest.TestCase):
    def test_explicit_start_only_and_no_duration_inference(self):
        index={'ts_code':'688766.SH','ann_date':'20251126'}
        text='证券代码：688766。自2025年11月26日（星期三）开市起开始停牌，预计停牌时间不超过10个交易日。'
        self.assertEqual(extract_full_halt_start(text,index,'688766.XSHG','2025-11-26')['session'],'2025-11-26')
        for args in [(text,index,'688766.XSHG','2025-11-27'),
                     (text,dict(index,ann_date='20251127'),'688766.XSHG','2025-11-26'),
                     (text,index,'688767.XSHG','2025-11-26'),
                     (text.replace('开市起','10:00起'),index,'688766.XSHG','2025-11-26'),
                     ('取消停牌。'+text,index,'688766.XSHG','2025-11-26')]:
            with self.subTest(args=args),self.assertRaises(ContractError):extract_full_halt_start(*args)

    def test_real_archive_and_hash_tamper_rejection(self):
        path=Path(__file__).resolve().parents[2]/'reports/fuxi_v1_announcements_20261003/postflight_halt_688766/BUNDLE.json'
        e={'path':str(path),'sha256':file_hash(path)}
        result=load_halt_starts(e,{'688766.XSHG'},['2025-11-26','2025-11-27'])
        self.assertEqual(set(result),{('688766.XSHG','2025-11-26')})
        with scratch_directory() as temp:
            b=json.loads(path.read_text(encoding='utf-8'));b['rows'][0]['text_sha256']='0'*64
            bad=Path(temp)/'bad.json';bad.write_text(json.dumps(b),encoding='utf-8')
            with self.assertRaises(ContractError):load_halt_starts({'path':str(bad),'sha256':file_hash(bad)}, {'688766.XSHG'},['2025-11-26'])

    def test_one_date_integration_retains_raw_absence_and_conflict(self):
        # Isolate importer integration from parsing, whose archived proof is tested above.
        from unittest.mock import patch
        with scratch_directory() as temp:
            root=Path(temp);days,iso,_=test_halt_reference.HaltReferenceTests().fixture(root)
            h=pd.read_csv(root/'suspend_d.csv',dtype=str);h['suspend_timing']='09:30-09:30';h.to_csv(root/'suspend_d.csv',index=False)
            codes=['600000.SH']+[f'60000{i}.SH' for i in range(1,10)]+['000001.SZ']
            plan={'window':dict(start=iso(days[0]),end=iso(days[-1]),calibration_end=iso(days[2]),research_start=iso(days[6]),horizon_sessions=2,warmup_sessions=3),
                'cohort':dict(members=[dict(ts_code=c,list_date='20200101',delist_date=None) for c in codes],rule_kind='all_a_share_census',selection='census fixture'),
                'universe':dict(listing_age_calendar_days=90),'raw_sources':{n:dict(path=str(root/n),sha256=file_hash(root/n)) for n in ['daily_prices.csv','adj_factor_daily.csv','stk_limit.csv','suspend_d.csv']},
                'issuer_halt_evidence':{'path':'test proof only'}}
            proof={('600000.XSHG',iso(days[8])):{'evidence_id':'proof','scope':'one date'},
                   ('600000.XSHG',iso(days[6])):{'evidence_id':'conflict','scope':'one date'}}
            with patch('fuxi_v1.halt_evidence.load_halt_starts',return_value=proof):
                p=build_history_input(root,plan,root,root/'issuer_audit')
            bars={(r['session'],r['instrument']):r for r in p['market']['bars']}
            first=bars[(iso(days[8]),'600000.XSHG')];nextday=bars[(iso(days[9]),'600000.XSHG')]
            self.assertEqual(first['close'],10.);self.assertTrue(first['paused'])
            self.assertEqual(first['issuer_full_halt']['evidence_id'],'proof')
            self.assertIsNone(p['snapshot']['columns']['close'][8][0])
            self.assertNotIn('600000.XSHG',p['snapshot']['universe'][iso(days[8])])
            self.assertIsNone(nextday['close']);self.assertNotIn('issuer_full_halt',nextday)
            self.assertIsNone(bars[(iso(days[6]),'600000.XSHG')]['paused'])


if __name__=='__main__':unittest.main()
