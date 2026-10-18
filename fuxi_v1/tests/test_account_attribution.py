import unittest
from copy import deepcopy
from pathlib import Path
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.tests.test_remaining_study import compact
from fuxi_v1.data import snapshot_from_payload
from fuxi_v1.dsl import evaluate_ast
from fuxi_v1.search import node
from fuxi_v1.contracts import ContractError,digest,load_json
from fuxi_v1.incremental_study import account_for
from fuxi_v1.attribution import diagnostic_style_residual
from fuxi_v1.account_attribution import full_account_attribution,residual_account_experiment
from fuxi_v1.simulation import run_account
from fuxi_v1.restricted_assets import freeze_restricted_policy
from fuxi_v1.remaining_study import freeze,run
from fuxi_v1.campaign import write_new


class FullAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload=compact(synthetic_campaign(140,8));cls.snapshot=snapshot_from_payload(cls.payload['snapshot'])
        s=cls.snapshot;d=s.sessions
        cls.request={'research_start':d[85],'research_end':d[-1]}
        cls.scores=evaluate_ast(node('rank',node('return',{'field':'close'},window=20)),s)
        cls.styles={'price':evaluate_ast(node('rank',{'field':'close'}),s)}
        cls.ir,cls.account=account_for(cls.scores,s,cls.payload['market'],cls.request,identity='attribute')
        equal={day:{code:1. if value is not None else None for code,value in row.items()} for day,row in cls.scores.items()}
        cls.pool_ir,cls.pool=account_for(equal,s,cls.payload['market'],cls.request,identity='pool',top_n=8)
        cls.benchmark={'rows':[{'ts_code':'TEST.PRICE','trade_date':day.replace('-',''),
                                'close':100.+i/10} for i,day in enumerate(d)]}

    def full(self,account=None,pool=None,market=None,benchmark=None):
        return full_account_attribution(account or self.account,pool or self.pool,market or self.payload['market'],
                                       self.styles,benchmark or self.benchmark,benchmark_id='TEST.PRICE')

    def test_residual_support_uses_no_future_label_and_exposes_pairs(self):
        fitted=diagnostic_style_residual(self.scores,self.styles,objective='diagnostic_style_residual')
        self.assertGreater(fitted['valid_sections'],0);self.assertFalse(fitted['labels_used_for_fit_or_support'])
        for day in fitted['raw_common_support']:
            self.assertEqual({s for s,v in fitted['raw_common_support'][day].items() if v is not None},
                             {s for s,v in fitted['residual_common_support'][day].items() if v is not None})
        valid=[r for r in fitted['sections'] if r['status']=='valid']
        self.assertLess(max(r['decomposition_max_error'] for r in valid),1e-10)
        self.assertTrue(valid[0]['style_exposures']);self.assertTrue(valid[0]['signal_style_correlations'])

    def test_missing_style_not_zero_filled_and_both_account_supports_match(self):
        style=deepcopy(self.styles);day=self.snapshot.sessions[90];code=next(iter(style['price'][day]))
        style['price'][day][code]=None
        fitted=diagnostic_style_residual(self.scores,style,objective='diagnostic_style_residual')
        self.assertIsNone(fitted['raw_common_support'][day][code]);self.assertIsNone(fitted['residual_common_support'][day][code])

    def test_fully_explained_signal_does_not_rank_numeric_noise(self):
        fitted=diagnostic_style_residual(self.scores,{'same_signal':self.scores},objective='diagnostic_style_residual')
        self.assertEqual(fitted['valid_sections'],0)
        self.assertTrue(all(v is None for row in fitted['residual_common_support'].values() for v in row.values()))

    def test_nonfinite_and_implicit_objective_rejected(self):
        scores=deepcopy(self.scores);scores[self.snapshot.sessions[90]][next(iter(scores[self.snapshot.sessions[90]]))]=float('nan')
        with self.assertRaises(ContractError):diagnostic_style_residual(scores,self.styles,objective='diagnostic_style_residual')
        with self.assertRaises(ContractError):diagnostic_style_residual(self.scores,self.styles,objective='investment_neutralization')

    def test_realized_account_bridge_nonempty_and_adds_exactly(self):
        result=self.full()
        self.assertGreater(result['n_pairs'],0);self.assertGreater(result['valid_style_sections'],0)
        self.assertLess(result['max_abs_bridge_error'],1e-10)
        self.assertEqual(result['n_positions'],len(self.account['daily'])-1)
        self.assertFalse(result['formal_eligibility']);self.assertFalse(result['causal_claim_allowed'])
        valid=[r for r in result['rows'] if r['status']=='valid']
        self.assertTrue(any(r['components']['explicit_fee_and_slippage_contribution']!=0 for r in valid))

    def test_unfilled_proxy_is_separate_from_realized_and_cancels(self):
        constrained=run_account(self.ir,self.payload['market'],start=self.request['research_start'],
                                end=self.request['research_end'],participation_multiplier=.000001)
        result=self.full(constrained,self.account)
        self.assertTrue(any(o['requested']>o['filled'] for o in constrained['orders']))
        self.assertFalse(result['unfilled_proxy_is_executable_counterfactual'])
        self.assertFalse(result['unfilled_proxy_is_realized_account_PnL'])
        self.assertTrue(any(r['status']=='valid' and r['opportunity_proxy_identified'] and
                            r['components']['unfilled_open_to_close_mark_proxy']!=0 for r in result['rows']))
        for row in result['rows']:
            if row['status']=='valid' and row['opportunity_proxy_identified']:
                c=row['components'];self.assertAlmostEqual(c['intended_open_to_close_mark_proxy']+
                    c['unfilled_open_to_close_mark_proxy'],c['filled_open_to_close_contribution'],places=12)

    def test_missing_real_quote_preserves_unknown_dates(self):
        market=deepcopy(self.payload['market']);row=self.account['daily'][6];code=next(iter(row['holdings']))
        for bar in market['bars']:
            if bar['session']==row['session'] and bar['instrument']==code:bar['close']=None
        a=run_account(self.ir,market,start=self.request['research_start'],end=self.request['research_end'])
        b=run_account(self.pool_ir,market,start=self.request['research_start'],end=self.request['research_end'])
        result=self.full(a,b,market)
        self.assertTrue(result['unknown_dates_by_reason']);self.assertLess(result['n_pairs'],result['n_positions'])
        self.assertTrue(any(r['status']=='data_insufficient' for r in result['rows']))

    def test_reference_tamper_and_wrong_benchmark_rejected(self):
        bad=deepcopy(self.account);bad['daily'][4]['cash']='999999999'
        with self.assertRaises(ContractError):self.full(bad)
        bad_index=deepcopy(self.benchmark);bad_index['rows'][0]['ts_code']='ANOTHER.PRICE'
        with self.assertRaises(ContractError):self.full(benchmark=bad_index)

    def test_isolated_residual_accounts_replay_with_same_support(self):
        result=residual_account_experiment(self.scores,self.styles,self.snapshot,self.payload['market'],self.request,
                                           objective='diagnostic_style_residual')
        self.assertGreater(result['scenarios'][0]['n_pairs'],0)
        self.assertFalse(result['future_labels_used_to_select_support']);self.assertFalse(result['production_signal_substitution'])
        self.assertFalse(result['investment_neutralization_certified'])
        self.assertTrue(all(not r['mismatches'] for r in result['scenarios'][0]['references'].values()))

    def test_native_optional_full_consumer_and_declared_pool_size(self):
        payload=deepcopy(self.payload);snap=payload['snapshot'];codes=snap['instruments']
        snap['fields']['free_shares']={'unit':'shares','frequency':'daily','adjustment':'raw'}
        snap['columns']['free_shares']=[[10000. for code in codes] for day in snap['sessions']]
        payload['labels']['feature_snapshot_hash']=digest(snap)
        industry={'captured_at':'2026-01-01T00:00:00+08:00','queries':[
            {'code':code[:6]+('.SH' if code.endswith('XSHG') else '.SZ'),
             'rows':[{'ts_code':code[:6]+('.SH' if code.endswith('XSHG') else '.SZ'),
                      'l1_code':'801010.SI','in_date':snap['sessions'][0].replace('-',''),'out_date':None}]} for code in codes]}
        benchmark=deepcopy(self.benchmark)
        for row in benchmark['rows']:row['ts_code']='000985.CSI'
        request=dict(self.request,pool_top_n=8,full_attribution=True,neutralization_objective='diagnostic_style_residual',
                     alpha=.05,max_trials=3,block_length=6,bootstrap_replicates=99,minimum_net_increment_daily=0.,
                     cost_stress_multiplier=2.,capacity_stress_multiplier=.5)
        with scratch_directory() as temp:
            root=Path(temp);source=root/'input.json';styles=root/'industry.json';index=root/'index.json'
            write_new(source,payload);write_new(styles,industry);write_new(index,benchmark)
            freeze(source,root/'run',request,industry=styles,benchmark=index)
            result=run(root/'run');self.assertGreater(result['full_attribution_n_pairs'],0)
            self.assertTrue(all(n>0 for n in result['residual_experiment_n_pairs']))
            accounts=load_json(root/'run'/'ACCOUNT_REPAIR.json')['result']
            self.assertEqual(accounts['pool_ir']['strategy']['top_n'],8)
            self.assertFalse(result['investment_neutralization_certified'])
            self.assertEqual(run(root/'run'),result)

    def test_restricted_carry_changes_equity_without_cash_and_retains_unknown_rights(self):
        market=deepcopy(self.payload['market']);day=self.snapshot.sessions[95]
        prior=next(r for r in self.account['daily'] if r['session']==self.snapshot.sessions[94])
        code=next(iter(prior['holdings']))
        for bar in market['bars']:
            if bar['instrument']==code and bar['session']>=day:bar['close']=None
        with scratch_directory() as temp:
            evidence=Path(temp)/'synthetic_restriction.txt';evidence.write_bytes(b'explicit synthetic test evidence')
            policy=freeze_restricted_policy(market,[{'instrument':code,'restrict_from':day,'known_at':day+'T00:00:00+08:00',
                'known_at_basis':'synthetic test declaration','source_paths':[evidence]}],horizon_end=self.request['research_end'])
            a=run_account(self.ir,market,start=self.request['research_start'],end=self.request['research_end'],
                          restricted_policy=policy,restricted_scenario='0')
            b=run_account(self.pool_ir,market,start=self.request['research_start'],end=self.request['research_end'],
                          restricted_policy=policy,restricted_scenario='0')
            result=self.full(a,b,market)
            self.assertTrue(any(r['restricted_holding_count']>0 for r in result['rows']))
            self.assertTrue(any(r['status']=='valid' and not r['candidate_strict_nav_known'] for r in result['rows']))
            self.assertFalse(result['formal_eligibility'])
            self.assertTrue(any(r['status']=='valid' and r['components']['restricted_carried_contribution']!=0
                                for r in result['rows']))


if __name__=='__main__':unittest.main()
