from copy import deepcopy
from pathlib import Path
from decimal import Decimal
import unittest
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.data import Snapshot
from fuxi_v1.portfolio import compile_strategy,strategy_spec
from fuxi_v1.contracts import ContractError,digest
from fuxi_v1.simulation import run_account
from fuxi_v1.account import Account
from fuxi_v1.account_reference import reconcile
from fuxi_v1.restricted_assets import freeze_restricted_policy,RestrictedResearch,replay_restricted_pair,run_restricted_study


def held_fixture(source):
    payload=synthetic_campaign(35,3);market=payload['market'];days=market['sessions']
    codes=payload['snapshot']['universe'][days[0]];restricted=codes[0];start=days[24]
    execution=market['execution'];execution.update(initial_cash='10000',slippage_bps='0',minimum_commission='0',
                                                 buy_commission='0',sell_commission='0',sell_tax='0')
    for bar in market['bars']:
        price=10*(codes.index(bar['instrument'])+1)
        bar.update(open=price,close=price,volume=1000000,up_limit=price*1.1,down_limit=price*.9)
    market['bars']=[b for b in market['bars'] if b['instrument']!=restricted or b['session']<start]
    market['market_hash']=digest({k:v for k,v in market.items() if k!='market_hash'})
    snapshot=Snapshot(payload['snapshot'])
    def ir(candidate):
        scores={d:{s:float(10 if (s==restricted if d<start else s==codes[candidate]) else 0) for s in codes} for d in days}
        return compile_strategy(strategy_spec('fixture'+str(candidate),top_n=1,max_weight=.4,rebalance_sessions=1),scores,snapshot,digest(execution))
    policy=freeze_restricted_policy(market,[{'instrument':restricted,'restrict_from':start,
      'known_at':days[23]+'T18:00:00+08:00','known_at_basis':'synthetic declared notice',
      'source_paths':[str(source)]}],horizon_end=days[-1])
    return market,ir(1),ir(2),policy,restricted,start


class RestrictedAssetsTests(unittest.TestCase):
    def fixture(self,folder):
        source=Path(folder)/'notice.txt';source.write_text('synthetic notice, not market evidence','utf8')
        return held_fixture(source)

    def test_nonzero_restricted_quantity_cash_and_strict_unknown_preserved(self):
        with scratch_directory() as folder:
            market,ir,_,p,code,start=self.fixture(folder)
            a=run_account(ir,market,restricted_policy=p,restricted_scenario='0')
            rows=[r for r in a['daily'] if r['session']>=start and 'restricted_research' in r]
            self.assertGreater(len(rows),0)
            for r in rows:
                s=r['restricted_research'];asset=s['restricted_assets'][code]
                self.assertEqual(asset['nominal_quantity'],400);self.assertEqual(asset['scenario_value'],'0')
                self.assertEqual(s['initial_capital'],'10000');self.assertIsNone(r['nav'])
                self.assertTrue(r['unresolved_events']);self.assertIsNotNone(s['conditional_total_nav'])
                self.assertEqual(s['cash'],r['cash']);self.assertFalse(asset['tradable'])
            self.assertFalse(any(o['instrument']==code and o['session']>=start for o in a['orders']))
            ref=reconcile(a,market);self.assertGreater(ref['n_pairs'],0);self.assertFalse(ref['mismatches'])
            self.assertFalse(a['formal_evidence']);self.assertEqual(a['eval_status'],'data_insufficient')

    def test_each_scenario_replays_actual_sizing_without_creating_cash(self):
        with scratch_directory() as folder:
            market,ir,_,p,code,start=self.fixture(folder)
            a=run_account(ir,market,restricted_policy=p,restricted_scenario='0')
            b=run_account(ir,market,restricted_policy=p,restricted_scenario='1')
            self.assertNotEqual(a['orders'],b['orders'])
            self.assertNotEqual(a['daily'][-1]['cash'],b['daily'][-1]['cash'])
            for account in [a,b]:
                self.assertTrue(all(Decimal(r['cash'])>=0 for r in account['daily']))
                self.assertEqual(account['daily'][-1]['holdings'][code],400)
                paid=sum((Decimal(e['payload']['price'])*e['payload']['quantity']*(1 if e['payload']['side']=='buy' else -1)
                          +Decimal(e['payload']['fee'])) for e in account['journal'] if e['kind']=='fill')
                self.assertLessEqual(paid,Decimal(10000))

    def test_pair_replays_all_scenarios_and_does_not_grant_qualification(self):
        with scratch_directory() as folder:
            market,a,b,p,_,_=self.fixture(folder)
            result=replay_restricted_pair(a,b,market,p)
            self.assertEqual(result['scenario_count'],3);self.assertFalse(result['formal_eligibility'])
            self.assertTrue(all(r['paired_return_count']>0 for r in result['rows']))
            self.assertTrue(all(r['paired_return_count']==len(r['accounts']['baseline']['daily'])-1 for r in result['rows']))

    def test_unknown_on_other_asset_still_blocks_conditional_account(self):
        with scratch_directory() as folder:
            market,ir,_,p,_,start=self.fixture(folder)
            other=next(b['instrument'] for b in market['bars'] if b['session']==start)
            market['actions'].append({'event_id':'unknown_other','kind':'unknown','timestamp':start+'T09:00:00+08:00',
              'evidence_id':'synthetic','payload':{'instrument':other,'reason':'unknown rights'}})
            p['market_digest']=digest(market);p['policy_hash']=digest({k:v for k,v in p.items() if k!='policy_hash'})
            a=run_account(ir,market,restricted_policy=p,restricted_scenario='0.5')
            last=a['daily'][-1]['restricted_research']
            self.assertIsNone(last['conditional_total_nav']);self.assertIn('unknown_other',last['unrestricted_unresolved_events'])

    def test_policy_source_market_horizon_and_qualification_tampering_rejected(self):
        with scratch_directory() as folder:
            market,ir,_,p,_,_=self.fixture(folder)
            for key,value in [('formal_eligibility',True),('market_digest','fake'),('horizon_end',market['sessions'][22])]:
                altered=deepcopy(p);altered[key]=value;altered['policy_hash']=digest({k:v for k,v in altered.items() if k!='policy_hash'})
                with self.assertRaises(ContractError):run_account(ir,market,restricted_policy=altered,restricted_scenario='0')
            with self.assertRaises(ContractError):run_account(ir,market,restricted_policy=p,restricted_scenario='0.8')
            Path(p['entries'][0]['sources'][0]['path']).write_text('changed','utf8')
            with self.assertRaisesRegex(ContractError,'source changed'):run_account(ir,market,restricted_policy=p,restricted_scenario='0')

    def test_known_at_prevents_early_restriction_and_no_future_backfill(self):
        with scratch_directory() as folder:
            market,_,_,p,code,start=self.fixture(folder)
            r=RestrictedResearch(market,p,'0.5');a=Account('10000')
            before=p['entries'][0]['known_at'];self.assertNotIn(code,r.active(before,a))
            self.assertIn(code,r.active(start+'T09:00:00+08:00',a))
            future=deepcopy(market);future['bars'].append({'instrument':code,'session':market['sessions'][-1],
                'close':999,'evidence_id':'future'})
            newer=freeze_restricted_policy(future,[{'instrument':code,'restrict_from':start,
              'known_at':before,'known_at_basis':'fixture','source_paths':[p['entries'][0]['sources'][0]['path']]}],
              horizon_end=market['sessions'][-1])
            self.assertEqual(newer['entries'][0]['anchor'],p['entries'][0]['anchor'])

    def test_real_quote_values_but_does_not_release_or_resolve_unknown_rights(self):
        with scratch_directory() as folder:
            market,_,_,p,code,start=self.fixture(folder);a=Account('10000')
            a.apply({'event_id':'buy','kind':'fill','timestamp':market['sessions'][22]+'T09:30:00+08:00','evidence_id':'fixture',
              'payload':{'instrument':code,'side':'buy','quantity':400,'price':'10','fee':'0',
                         'sellable_at':market['sessions'][23]+'T09:30:00+08:00'}})
            a.apply({'event_id':'u','kind':'unknown','timestamp':start+'T09:00:00+08:00','evidence_id':'fixture',
                     'payload':{'instrument':code,'reason':'registration unknown'}})
            when=start+'T18:00:00+08:00';q={code:{'price':'12','asof':when,'evidence_id':'real fixture quote'}}
            result=RestrictedResearch(market,p,'0').project(a,when,q)
            self.assertEqual(result['restricted_scenario_value'],'4800');self.assertEqual(a.cash,Decimal(6000))
            self.assertFalse(result['restricted_assets'][code]['tradable']);self.assertEqual(result['unresolved_events'],['u'])
            self.assertFalse(a.mark(when,q)['valuation_identifiable'])

    def test_quantity_preserving_transfer_inherits_restriction(self):
        with scratch_directory() as folder:
            market,_,_,p,code,start=self.fixture(folder);a=Account('10000')
            a.apply({'event_id':'buy','kind':'fill','timestamp':market['sessions'][22]+'T09:30:00+08:00','evidence_id':'fixture',
              'payload':{'instrument':code,'side':'buy','quantity':400,'price':'10','fee':'0',
                         'sellable_at':market['sessions'][23]+'T09:30:00+08:00'}})
            a.apply({'event_id':'transfer','kind':'security_transfer','timestamp':start+'T09:00:00+08:00','evidence_id':'fixture',
              'payload':{'from_instrument':code,'to_instrument':'otc','quantity':400,
                         'sellable_at':start+'T09:00:00+08:00','resolved_event_ids':[]}})
            result=RestrictedResearch(market,p,'1').project(a,start+'T18:00:00+08:00',{})
            self.assertEqual(result['restricted_assets']['otc']['nominal_quantity'],400)
            self.assertFalse(result['restricted_assets']['otc']['tradable']);self.assertEqual(result['conditional_total_nav'],'10000')

    def test_pending_shares_and_cash_claims_kept_without_spendable_cash(self):
        with scratch_directory() as folder:
            market,_,_,p,code,start=self.fixture(folder);a=Account('10000')
            a.apply({'event_id':'claim','kind':'share_entitlement','timestamp':start+'T09:00:00+08:00','evidence_id':'fixture',
              'payload':{'entitlement_id':'shares','instrument':code,'quantity':40,'sellable_at':start+'T18:00:00+08:00'}})
            a.apply({'event_id':'cash_claim','kind':'receivable','timestamp':start+'T09:00:00+08:00','evidence_id':'fixture',
              'payload':{'entitlement_id':'cash','amount':'80'}})
            result=RestrictedResearch(market,p,'0.5').project(a,start+'T18:00:00+08:00',{})
            self.assertEqual(result['restricted_assets'][code]['pending_share_quantity'],40)
            self.assertEqual(result['cash'],'10000');self.assertEqual(result['cash_receivables'],'80')
            self.assertIsNone(result['conditional_total_nav']);self.assertTrue(result['unsupported_anchor_basis_events'])

    def test_known_split_blocks_stale_anchor_basis_until_a_real_quote_exists(self):
        with scratch_directory() as folder:
            market,_,_,p,code,start=self.fixture(folder);a=Account('10000')
            a.apply({'event_id':'buy','kind':'fill','timestamp':market['sessions'][22]+'T09:30:00+08:00','evidence_id':'fixture',
              'payload':{'instrument':code,'side':'buy','quantity':400,'price':'10','fee':'0',
                         'sellable_at':market['sessions'][23]+'T09:30:00+08:00'}})
            a.apply({'event_id':'split','kind':'split','timestamp':start+'T09:00:00+08:00','evidence_id':'fixture',
                     'payload':{'instrument':code,'multiplier':'2','sellable_at':start+'T09:00:00+08:00'}})
            r=RestrictedResearch(market,p,'1');when=start+'T18:00:00+08:00'
            self.assertIsNone(r.project(a,when,{})['conditional_total_nav'])
            projected=r.project(a,when,{code:{'price':'5','asof':when,'evidence_id':'postsplit'}})
            self.assertEqual(projected['restricted_assets'][code]['nominal_quantity'],800)
            self.assertEqual(projected['conditional_total_nav'],'10000')

    def test_manual_consumer_freezes_then_replays_and_refuses_overwrite(self):
        from fuxi_v1.campaign import write_new
        from fuxi_v1.contracts import load_json
        with scratch_directory() as folder:
            market,a,b,p,_,_=self.fixture(folder);root=Path(folder)
            payload=synthetic_campaign(35,3);payload['market']=market
            for name,obj in [('input',payload),('baseline',a),('candidate',b),('policy',p)]:write_new(root/(name+'.json'),obj)
            args=[root/(n+'.json') for n in ['input','baseline','candidate','policy']]+[root/'result']
            result=run_restricted_study(*args,start=market['sessions'][20],end=market['sessions'][-1])
            self.assertEqual(result['scenario_count'],3)
            self.assertEqual(load_json(root/'result/COMPLETE.json')['result_hash'],digest(load_json(root/'result/RESULT.json')))
            with self.assertRaisesRegex(ContractError,'empty new output'):
                run_restricted_study(*args,start=market['sessions'][20],end=market['sessions'][-1])

    def test_stale_quote_is_not_reported_as_real_current_quote(self):
        with scratch_directory() as folder:
            market,_,_,p,code,start=self.fixture(folder);a=Account('10000')
            a.apply({'event_id':'buy','kind':'fill','timestamp':market['sessions'][22]+'T09:30:00+08:00','evidence_id':'fixture',
              'payload':{'instrument':code,'side':'buy','quantity':400,'price':'10','fee':'0',
                         'sellable_at':market['sessions'][23]+'T09:30:00+08:00'}})
            when=market['sessions'][25]+'T18:00:00+08:00'
            stale={code:{'price':'12','asof':start+'T18:00:00+08:00','evidence_id':'old observed quote'}}
            result=RestrictedResearch(market,p,'0.5').project(a,when,stale)
            self.assertEqual(result['restricted_assets'][code]['mark_kind'],'conditional_anchor_loss_scenario')
            self.assertEqual(result['restricted_scenario_value'],'2000.0')
