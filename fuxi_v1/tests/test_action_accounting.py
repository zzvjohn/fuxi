from copy import deepcopy
from decimal import Decimal
import unittest
from fuxi_v1.account import Account
from fuxi_v1.action_accounting import post_entitlements,sell_quantity,reconcile_account
from fuxi_v1.action_evidence import resolve_action_terms
from fuxi_v1.historical_adapter import dividend_terms
from fuxi_v1.contracts import ContractError,digest


class ActionAccountingTests(unittest.TestCase):
    def event(self,kind,payload,day='2024-01-03',identity=None):
        return {'event_id':identity or kind,'kind':kind,'payload':payload,'timestamp':day+'T09:00:00+08:00','evidence_id':'fixture'}

    def test_share_claim_valuation_credit_and_rollback(self):
        a=Account('100');claim={'entitlement_id':'c','instrument':'stock','quantity':40,'sellable_at':'2024-01-04T09:00:00+08:00'}
        a.apply(self.event('share_entitlement',claim));before=deepcopy(a.__dict__)
        with self.assertRaisesRegex(ContractError,'before availability'):a.apply(self.event('share_credit',{'entitlement_id':'c'}))
        self.assertEqual(before,a.__dict__)
        quote={'stock':{'price':'5','asof':'2024-01-03T18:00:00+08:00','evidence_id':'q'}}
        mark=a.mark('2024-01-03T18:00:00+08:00',quote)
        self.assertEqual(Decimal(mark['nav']),Decimal(300));self.assertEqual(mark['holdings'],{})
        self.assertEqual(mark['pending_shares'],{'stock':40})
        a.apply(self.event('share_credit',{'entitlement_id':'c'},'2024-01-04'))
        self.assertEqual(a.positions['stock'][0]['quantity'],40)
        with self.assertRaisesRegex(ContractError,'already credited'):a.apply(self.event('share_credit',{'entitlement_id':'c'},'2024-01-04','duplicate'))

    def test_claim_belongs_to_record_not_ex_date_holdings(self):
        a=Account('100');t={'instrument':'s','status':'ordinary_distribution_supported','ex_date':'2024-01-03',
          'gross_cash_per_share':.1,'share_distribution':.4,'tax_fraction':'.2','share_credit_date':'2024-01-04','evidence_id':'t'}
        post_entitlements(a,[t],'2024-01-03',{'t':100},{},'ir',{},[])
        self.assertEqual(sum(a.receivables.values()),Decimal('8'));self.assertEqual(next(iter(a.share_receivables.values()))['quantity'],40)
        self.assertFalse(a.positions)
        post_entitlements(a,[t],'2024-01-04',{'t':100},{},'ir',{},[])
        self.assertEqual(a.positions['s'][0]['quantity'],40)

    def test_unknown_fractional_allotment_is_not_silently_rounded(self):
        a=Account('100');gaps=[];t={'instrument':'s','status':'ordinary_distribution_supported','ex_date':'2024-01-03',
          'gross_cash_per_share':0,'share_distribution':.1,'tax_fraction':'.2','share_credit_date':'2024-01-03','evidence_id':'t'}
        post_entitlements(a,[t],'2024-01-03',{'t':105},{},'ir',{},gaps)
        self.assertTrue(a.unresolved);self.assertFalse(a.share_receivables)
        self.assertEqual(gaps[0]['reason'],'fractional_allotment_not_observed')
        for mode,expected in [('floor',10),('ceil',11)]:
            a=Account('100');post_entitlements(a,[t],'2024-01-03',{'t':105},{},'ir',{'fractional_share_policy':mode},[])
            self.assertEqual(a.positions['s'][0]['quantity'],expected)

    def test_odd_lot_is_whole_and_respects_volume_and_settlement(self):
        self.assertEqual(sell_quantity(140,140,140,1000,100,True),140)
        self.assertEqual(sell_quantity(140,140,140,100,100,True),100)
        self.assertEqual(sell_quantity(40,40,40,100,100,True),40)
        self.assertEqual(sell_quantity(40,40,40,50,100,True),40)
        self.assertEqual(sell_quantity(25,40,40,100,100,True),0)
        self.assertEqual(sell_quantity(140,140,100,1000,100,True),100)
        self.assertEqual(sell_quantity(140,140,140,1000,100,False),100)

    def test_resolver_separates_issuer_capital_from_holder_rights(self):
        r={'ts_code':'000001.SZ','div_proc':'实施','ex_date':'20240103','record_date':'20240102','pay_date':None,
           'imp_ann_date':'20231229','stk_div':'1.8','cash_div_tax':None,'div_listdate':'20240103'}
        base=dividend_terms([r],['2024-01-01','2024-01-05'])
        j={'instrument':'000001.XSHE','ex_date':'2024-01-03','record_date':'2024-01-02','ann_date':'2023-12-29',
           'company_share_distribution':1.8,'holder_cash':0,'holder_shares':0,'share_credit_date':None,
           'sources':[{'url':'https://example.org/fixture','sha256':'fixture'}]}
        resolved=resolve_action_terms(base,[r],[j])[0]
        self.assertEqual(resolved['share_distribution'],0);self.assertEqual(resolved['status'],'restructuring_no_holder_distribution')
        self.assertEqual(base[0]['share_distribution'],1.8)
        with self.assertRaises(ContractError):resolve_action_terms(base,[dict(r,stk_div='1.7')],[j])

    def test_resolver_states_zero_cash_for_pure_share_and_never_invents_credit_day(self):
        r={'ts_code':'000001.SZ','div_proc':'实施','ex_date':'20240103','record_date':'20240102','pay_date':'20240103',
           'imp_ann_date':'20231229','stk_div':'0.4','cash_div_tax':'.1','div_listdate':'20240104'}
        def resolved(row):return resolve_action_terms(dividend_terms([row],['2024-01-01','2024-01-05']),[row],[])[0]
        self.assertEqual(resolved(r)['share_credit_date'],'2024-01-04')
        self.assertEqual(resolved(r)['status'],'ordinary_distribution_supported')
        # P-009 (a1): a pure share distribution carries a stated zero cash leg and
        # settles through the vendor credit date; the credit day itself is never invented.
        pure=resolved(dict(r,cash_div_tax=None))
        self.assertEqual(pure['status'],'ordinary_distribution_supported')
        self.assertEqual(pure['gross_cash_per_share'],0.0)
        self.assertEqual(pure['share_credit_date'],'2024-01-04')
        self.assertEqual(resolved(dict(r,div_listdate=None))['status'],'share_distribution_requires_settlement_adapter')

    def test_independent_replay_detects_cash_shares_and_nav_corruption(self):
        from fuxi_v1.tests.test_system import PortfolioExecutionTests
        from fuxi_v1.simulation import run_account
        PortfolioExecutionTests.setUpClass();fixture=PortfolioExecutionTests();market=fixture.payload['market']
        result=run_account(fixture.ir(),market)
        self.assertTrue(reconcile_account(result,market['execution']['initial_cash'],market)['nav_reconciled'])
        changed=deepcopy(result);changed['daily'][-1]['nav']='1'
        with self.assertRaisesRegex(ContractError,'NAV reconciliation'):reconcile_account(changed,market['execution']['initial_cash'],market)

    def test_second_table_must_have_equal_factors_and_agreeing_quotes(self):
        from fuxi_v1.action_evidence import constant_factor_corroborated
        a={'adj_factor':3.1681,'close':10.};b={'adj_factor':3.1681,'close':11.,'pre_close':10.}
        self.assertTrue(constant_factor_corroborated(a,b,10.,10.,11.))
        self.assertFalse(constant_factor_corroborated(a,dict(b,adj_factor=3.1682),10.,10.,11.))
        self.assertFalse(constant_factor_corroborated(a,b,10.,9.9,11.))
        self.assertFalse(constant_factor_corroborated(a,b,10.,10.,11.1))
        self.assertFalse(constant_factor_corroborated(None,b,10.,10.,11.))

    def test_preflight_cannot_be_bypassed_by_empty_or_blocked_receipt(self):
        from fuxi_v1.tests.test_kernel import scratch_directory
        from fuxi_v1.remediation import require_preflight
        from fuxi_v1.datasets import file_hash
        from fuxi_v1.campaign import write_new
        from pathlib import Path
        plan={'remediation':{'search_gate':'all probes'}}
        with scratch_directory() as root:
            path=Path(root)/'market.json';write_new(path,{'synthetic':True});p=Path(root)/'preflight';p.mkdir()
            receipt={'plan_hash':digest(plan),'input_sha256':file_hash(path),'code_hashes':{},'probe_count':12,'search_allowed':True,'readouts':[]}
            write_new(p/'DECISION.json',receipt)
            with self.assertRaisesRegex(ContractError,'probes incomplete'):require_preflight(plan,path,{})

    def test_independent_event_calendar_never_overrides_action_or_price_disagreement(self):
        from fuxi_v1.action_evidence import independent_no_event
        a={'tradestatus':'1','close':'10','volume':'1000'};b={'tradestatus':'1','preclose':'10','close':'11','volume':'2000'}
        args=('2024-01-03',10.,10.,11.)
        self.assertTrue(independent_no_event(a,b,set(),*args))
        self.assertFalse(independent_no_event(a,b,{'2024-01-03'},*args))
        self.assertFalse(independent_no_event(a,b,None,*args))
        self.assertFalse(independent_no_event(a,dict(b,tradestatus='0'),set(),*args))
        self.assertFalse(independent_no_event(a,dict(b,preclose='9.99'),set(),*args))

    def test_uncertified_net_description_cannot_become_reward(self):
        from fuxi_v1.learning import feedback_card
        from fuxi_v1.contracts import evaluation
        result={'snapshot_hash':'s','evaluations':[evaluation('E1','valid','research'),evaluation('E2','valid','similarity',metrics={}),
          evaluation('E3','data_insufficient','market_evidence_not_certified',diagnostics={'paired_mean_increment':.01})]}
        card=feedback_card({'candidate_id':'c'},result,{'policy_hash':'p','request':{'research_start':'a','research_end':'b'}},{},'e','state')
        self.assertEqual(card['descriptive_net_increment'],.01);self.assertIsNone(card['net_increment'])
        self.assertFalse(card['economic_reward_eligible']);self.assertFalse(card['qualification_transfer_allowed'])
        result['evaluations'][1]=evaluation('E2','data_insufficient','zero_pairs')
        self.assertIsNone(feedback_card({'candidate_id':'c'},result,{'policy_hash':'p','request':{'research_start':'a','research_end':'b'}},{},'e','state')['baseline_similarity']['mean_abs_correlation'])

    def test_delayed_share_credit_survives_sale_of_original_position(self):
        from fuxi_v1.system_fixture import synthetic_campaign
        from fuxi_v1.data import Snapshot
        from fuxi_v1.portfolio import strategy_spec,compile_strategy
        from fuxi_v1.simulation import run_account
        p=synthetic_campaign(33,3);s=Snapshot(p['snapshot']);days=s.sessions;codes=s.universe[days[0]];market=p['market']
        market['execution']['participation']='1';market['execution']['odd_lot_exit']='whole_residual'
        scores={d:{c:float(c==(codes[0] if i<22 else codes[1])) for c in codes} for i,d in enumerate(days) if i>=s.start}
        term={'instrument':codes[0],'record_date':days[22],'ex_date':days[23],'pay_date':None,
             'status':'ordinary_distribution_supported','gross_cash_per_share':0,'share_distribution':.4,
             'share_credit_date':days[27],'tax_fraction':'.2','evidence_id':'share_fixture'}
        market['action_terms']=[term]
        for bar in market['bars']:
            if bar['session']==days[23] and bar['instrument']==codes[0]:
                bar.update(sizing_reference_close=bar['close'],sizing_reference_evidence='fixture')
        ir=compile_strategy(strategy_spec('fixture',top_n=1,rebalance_sessions=1,max_weight=.1,gross_exposure=.1),scores,s,digest(market['execution']))
        result=run_account(ir,market);daily={r['session']:r for r in result['daily']}
        quantity=daily[days[22]]['holdings'][codes[0]]
        self.assertNotIn(codes[0],daily[days[24]]['holdings'])
        self.assertEqual(daily[days[24]]['pending_shares'][codes[0]],int(quantity*.4))
        credits=[e for e in result['journal'] if e['kind']=='share_credit']
        self.assertEqual([e['timestamp'][:10] for e in credits],[days[27]])
        self.assertTrue(reconcile_account(result,market['execution']['initial_cash'],market)['nav_reconciled'])

    def test_recorded_pause_does_not_require_an_executable_quote(self):
        from fuxi_v1.tests.test_system import PortfolioExecutionTests
        from fuxi_v1.simulation import run_account
        PortfolioExecutionTests.setUpClass();fixture=PortfolioExecutionTests();market=deepcopy(fixture.payload['market'])
        ir=fixture.ir();day=ir['decisions'][0]['execution_session']
        for bar in market['bars']:
            if bar['session']==day:bar.update(paused=True,open=None,volume=None,up_limit=None,down_limit=None)
        frozen=deepcopy(market);result=run_account(ir,market)
        orders=[o for o in result['orders'] if o['session']==day]
        self.assertTrue(orders);self.assertTrue(all(o['reason']=='paused' and o['filled']==0 for o in orders))
        self.assertFalse([g for g in result['gaps'] if g['session']==day]);self.assertEqual(market,frozen)

    def test_only_evidenced_full_halt_supplies_zero_lagged_capacity(self):
        from fuxi_v1.tests.test_system import PortfolioExecutionTests
        from fuxi_v1.simulation import run_account
        PortfolioExecutionTests.setUpClass();fixture=PortfolioExecutionTests();ir=fixture.ir()
        signal=ir['decisions'][0]['signal_session'];day=ir['decisions'][0]['execution_session']
        for mode in ('full','partial','unknown'):
            market=deepcopy(fixture.payload['market'])
            for bar in market['bars']:
                if bar['session']==signal:bar.update(volume=None,paused=mode!='unknown',
                    mark_source='declared_frozen_last_close_on_vendor_halt' if mode=='full' else 'observed_close')
            result=run_account(ir,market);orders=[o for o in result['orders'] if o['session']==day]
            self.assertTrue(orders);self.assertTrue(all(o['filled']==0 for o in orders))
            gaps=[g for g in result['gaps'] if g['session']==day and g['reason']=='lagged_volume_missing']
            self.assertEqual(bool(gaps),mode!='full')
            if mode=='full':self.assertTrue(all(o['reason']=='prior_full_halt_no_capacity' for o in orders))

    def test_pause_cannot_certify_unknown_events_on_existing_holdings(self):
        from fuxi_v1.tests.test_system import PortfolioExecutionTests
        from fuxi_v1.simulation import run_account
        PortfolioExecutionTests.setUpClass();fixture=PortfolioExecutionTests();market=deepcopy(fixture.payload['market']);ir=fixture.ir()
        baseline=run_account(ir,market);held=next(d for d in baseline['daily'] if d['holdings']);code=next(iter(held['holdings']))
        day=next(d['session'] for d in baseline['daily'] if d['session']>held['session'])
        for bar in market['bars']:
            if bar['session']==day and bar['instrument']==code:bar.update(paused=True,corporate_action_known=False,open=None,volume=None)
        result=run_account(ir,market)
        self.assertEqual(result['eval_status'],'data_insufficient')
        self.assertTrue(any(g['session']==day and g['reason']=='corporate_action_coverage_unknown' for g in result['gaps']))
        self.assertIn(code,next(d for d in result['daily'] if d['session']==day)['holdings'])

if __name__=='__main__':unittest.main()
