"""Integration regression: cash distributions during a halt cannot double NAV."""
import json
from decimal import Decimal
from pathlib import Path
import unittest
import pandas as pd
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.historical_adapter import build_history_input, HistoricalSnapshot
from fuxi_v1.datasets import file_hash
from fuxi_v1.contracts import digest
from fuxi_v1.portfolio import strategy_spec,compile_strategy
from fuxi_v1.simulation import run_account
from fuxi_v1.action_accounting import reconcile_account


class HaltReferenceTests(unittest.TestCase):
    def fixture(self,root,*,second_cash=False,unsupported_shares=False):
        days=pd.bdate_range('2026-08-03',periods=17).strftime('%Y%m%d').tolist()
        iso=lambda d:d[:4]+'-'+d[4:6]+'-'+d[6:]
        codes=['600000.SH']+[f'60000{i}.SH' for i in range(1,10)]+['000001.SZ']
        cash_dates={10:.03,12:.02} if second_cash else {10:.03}
        # The vendor factor can remain constant throughout a recorded halt.
        prices=[];factors=[];limits=[];halts=[]
        for i,d in enumerate(days):
            total=sum(v for index,v in cash_dates.items() if index<=i)
            for c in codes:
                close=10-total if c==codes[0] else 10.
                if c!=codes[0] or not 8<=i<=13:
                    prices.append(dict(ts_code=c,trade_date=d,open=close,high=close+.01,low=close-.01,close=close,
                        pre_close=close,pct_chg=0,change=0,vol=1000000,amount=1000000*close/10))
                factors.append(dict(ts_code=c,trade_date=d,adj_factor=1.))
                limits.append(dict(ts_code=c,trade_date=d,up_limit=close*1.1,down_limit=close*.9))
                if c==codes[0] and 8<=i<=13:halts.append(dict(ts_code=c,trade_date=d,suspend_type='S',suspend_timing=None))
        files={'daily_prices.csv':prices,'adj_factor_daily.csv':factors,'stk_limit.csv':limits,'suspend_d.csv':halts}
        for name,rows in files.items():pd.DataFrame(rows).to_csv(root/name,index=False,lineterminator='\n')
        pd.DataFrame([dict(exchange='SSE',cal_date=d,is_open=1) for d in days]).to_csv(root/'trade_calendar.csv',index=False)
        pd.DataFrame([dict(ts_code=c,name='样例',start_date='20200101',end_date=None,ann_date='20200101') for c in codes]).to_csv(root/'namechange.csv',index=False)
        actions=[dict(ts_code=codes[0],end_date='20251231',div_proc='实施',ex_date=days[i],record_date=days[i-1],pay_date=days[i],
            stk_div=.1 if unsupported_shares else 0,cash_div_tax=cash,imp_ann_date=days[1]) for i,cash in cash_dates.items()]
        pd.DataFrame(actions).to_csv(root/'dividend.csv',index=False)
        plan={'window':dict(start=iso(days[0]),end=iso(days[-1]),calibration_end=iso(days[2]),research_start=iso(days[6]),horizon_sessions=2,warmup_sessions=3),
            'cohort':dict(members=[dict(ts_code=c,list_date='20200101',delist_date=None) for c in codes],rule_kind='all_a_share_census',selection='census fixture'),
            'universe':dict(listing_age_calendar_days=90),'raw_sources':{n:dict(path=str(root/n),sha256=file_hash(root/n)) for n in files}}
        payload=build_history_input(root,plan,root,root/'audit')
        return days,iso,payload

    def test_cash_halt_and_resumption_preserve_raw_absence_and_tax_only_nav_change(self):
        for second in (False,True):
            with self.subTest(two_events=second),scratch_directory() as temp:
                root=Path(temp);days,iso,p=self.fixture(root,second_cash=second)
                bars={(r['session'],r['instrument']):r for r in p['market']['bars']};code='600000.XSHG'
                for index in range(8,14):
                    self.assertIsNone(p['snapshot']['columns']['close'][index][0])
                    self.assertNotIn(code,p['snapshot']['universe'][iso(days[index])])
                self.assertEqual(bars[(iso(days[10]),code)]['close'],9.97)
                self.assertEqual(bars[(iso(days[13]),code)]['close'],9.95 if second else 9.97)
                self.assertTrue(bars[(iso(days[14]),code)]['corporate_action_known'])
                s=HistoricalSnapshot(p['snapshot'])
                # Buy before the halt and keep the actual held lot; no future selection.
                scores={d:{c:2. if c==code else 0. for c in s.universe[d]} for d in s.sessions[s.start:]}
                ir=compile_strategy(strategy_spec('halt_cash',top_n=1,max_weight=1.,rebalance_sessions=3),scores,s,digest(p['market']['execution']))
                a=run_account(ir,p['market'],start=iso(days[6]),end=iso(days[-1]))
                self.assertEqual(a['eval_status'],'valid',a['gaps']);daily={r['session']:r for r in a['daily']}
                for index,cash in ({10:.03,12:.02} if second else {10:.03}).items():
                    before=daily[iso(days[index-1])];after=daily[iso(days[index])]
                    q=before['holdings'][code]
                    self.assertEqual(Decimal(after['nav'])-Decimal(before['nav']),-Decimal(str(cash))*Decimal('.20')*q)
                replay=reconcile_account(a,p['market']['execution']['initial_cash'],p['market'])
                self.assertTrue(replay['cash_and_share_reconciled']);self.assertTrue(replay['nav_reconciled'])

    def test_share_or_unknown_halt_reference_remains_unidentifiable(self):
        with scratch_directory() as temp:
            days,iso,p=self.fixture(Path(temp),unsupported_shares=True)
            bars={(r['session'],r['instrument']):r for r in p['market']['bars']};code='600000.XSHG'
            self.assertIsNone(bars[(iso(days[10]),code)]['close'])
            self.assertIsNone(bars[(iso(days[13]),code)]['close'])
            self.assertFalse(bars[(iso(days[10]),code)]['corporate_action_known'])
            self.assertEqual(bars[(iso(days[10]),code)]['halt_reference']['status'],'data_insufficient')


if __name__=='__main__':unittest.main()
