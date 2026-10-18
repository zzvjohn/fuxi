from pathlib import Path
import unittest
from fuxi_v1.halt_evidence import extract_closed_halt, load_halt_starts
from fuxi_v1.datasets import file_hash
from fuxi_v1.contracts import ContractError


class ClosedHaltTests(unittest.TestCase):
    def fixture(self):
        start='证券代码：688766。公告编号：2025-082。自2025年11月26日（星期三）开市起开始停牌，预计停牌不超过10个交易日。'
        resume='证券代码：688766。公告编号：2025-088。复牌2025/12/82025/12/9。自2025年11月25日（星期二）开市起停牌，详见公告编号：2025-082。将于2025年12月9日（星期二）开市起复牌。'
        return start,resume,{'ts_code':'688766.SH','ann_date':'20251126'}, {'ts_code':'688766.SH','ann_date':'20251209'},'688766.XSHG'

    def test_intersection_preserves_conflict_and_retrospective_scope(self):
        r=extract_closed_halt(*self.fixture())
        self.assertEqual((r['interval_start'],r['interval_end'],r['resume_session']),('2025-11-26','2025-12-08','2025-12-09'))
        self.assertTrue(r['start_date_conflict']);self.assertFalse(r['independent_assurance'])
        self.assertIn('retrospective',r['scope'])
        a,b,c,d,e=self.fixture();r=extract_closed_halt(a,b.replace('11月25日','11月26日'),c,d,e)
        self.assertFalse(r['start_date_conflict'])

    def test_expectation_and_wrong_incident_never_define_interval(self):
        a,b,c,d,e=self.fixture()
        cases=[(a,b.replace('复牌2025/12/82025/12/9。',''),c,d,e),
               (a,b.replace('详见公告编号：2025-082','详见公告编号：2025-099'),c,d,e),
               (a,b,c,dict(d,ann_date='20251210'),e),
               (a,b.replace('证券代码：688766','证券代码：688767'),c,d,e),
               (a,b.replace('11月25日','12月10日'),c,d,e)]
        for args in cases:
            with self.subTest(args=args),self.assertRaises(ContractError):extract_closed_halt(*args)

    def test_actual_archives_scope_nine_sessions_only(self):
        p=Path(__file__).resolve().parents[2]/'reports/fuxi_v1_announcements_20261003/postflight_halt_interval_688766/BUNDLE.json'
        sessions=['2025-11-25','2025-11-26','2025-11-27','2025-11-28','2025-12-01','2025-12-02','2025-12-03','2025-12-04','2025-12-05','2025-12-08','2025-12-09']
        r=load_halt_starts({'path':str(p),'sha256':file_hash(p)},{'688766.XSHG'},sessions)
        self.assertEqual(len(r),9)
        self.assertNotIn(('688766.XSHG','2025-11-25'),r);self.assertNotIn(('688766.XSHG','2025-12-09'),r)
        self.assertTrue(all(v['start_date_conflict'] and v['independent_assurance'] is False for v in r.values()))


if __name__=='__main__':unittest.main()
