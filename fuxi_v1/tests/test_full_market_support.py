"""Contracts at the all-market worker transport and legacy dispatch boundary."""
from pathlib import Path
import hashlib,json,unittest
import numpy as np
from fuxi_v1.contracts import canonical,digest,ContractError
from fuxi_v1.data import Snapshot,Labels
from fuxi_v1.dsl import evaluate_ast
from fuxi_v1.full_market_support import pack_grids,unpack_grids,ordinal_identity,diagnostic_jq,execution_options,complete_generation,grid_bundle_files,load_grid_bundle
from fuxi_v1.tests.test_columnar import fixture
from fuxi_v1.tests.test_kernel import scratch_directory


class FullMarketSupportTests(unittest.TestCase):
    def test_restricted_anchor_and_independent_account_replay_match_scalar(self):
        from fuxi_v1.restricted_assets import freeze_restricted_policy,RestrictedResearch
        from fuxi_v1.portfolio import compile_strategy,strategy_spec
        from fuxi_v1.simulation import run_account
        from fuxi_v1.account_reference import reconcile
        with scratch_directory() as root:
            cs,ns,cl,nl,market=fixture(root)
            try:
                source=Path(root)/'restriction_source.txt';source.write_text('fixture retrospective restriction')
                entries=[{'instrument':cs.instruments[0],'restrict_from':cs.sessions[15],
                    'known_at':cs.sessions[15]+'T18:00:00+08:00','known_at_basis':'declared fixture historical scenario',
                    'source_paths':[str(source)]}]
                scalar=dict(market,bars=[dict(row) for row in market['bars']])
                a=freeze_restricted_policy(market,entries,horizon_end=cs.sessions[-1])
                b=freeze_restricted_policy(scalar,entries,horizon_end=cs.sessions[-1])
                self.assertEqual(a['entries'],b['entries'])
                self.assertEqual(a['entries'][0]['anchor']['session'],cs.sessions[14])
                RestrictedResearch(market,a,'0.5')
                scores=evaluate_ast({'field':'close'},cs)
                ir=compile_strategy(strategy_spec('restricted_fixture',top_n=2,max_weight=.5),scores,cs,digest(market['execution']))
                account=run_account(ir,market,restricted_policy=a,restricted_scenario='0.5')
                native=run_account(ir,scalar,restricted_policy=b,restricted_scenario='0.5')
                # Only frozen policy identities differ because scalar market embeds all bars.
                for result in (account,native):
                    result.pop('restricted_policy_hash')
                    for row in result['daily']:
                        if 'restricted_research' in row:row['restricted_research'].pop('policy_hash')
                self.assertEqual(account,native)
                ref=reconcile(account,market)
                self.assertGreater(ref['n_pairs'],0);self.assertFalse(ref['mismatches'])
                self.assertEqual(ref,reconcile(native,scalar))
            finally:cs.close();cl.close();market.close()

    def test_unknown_tristate_remains_an_evidence_gap(self):
        from fuxi_v1.reachability import audit_reachability
        class Market(dict):_columnar_market=True
        market=Market(event_coverage_certified=False,tradability_certified=False)
        market.status={'corporate_action_known':np.array([[-1,0,1]],dtype=np.int8)}
        market.categorical={'event_coverage':np.array([[0,1,2]],dtype=np.uint8)}
        market.category_values={'event_coverage':{'0':'missing','1':'unknown','2':'observed'}}
        payload={'snapshot':{'sessions':['2024-01-02','2024-01-03'],'quality':{}},'market':market}
        policy={'policy_hash':'frozen','minimum_effective_sections':2,
                'request':{'research_start':'2024-01-02','research_end':'2024-01-03',
                           'bootstrap_replicates':199,'alpha':.05,'max_trials':40}}
        result=audit_reachability(payload,policy)
        self.assertEqual(result['unknown_action_cells'],2)
        self.assertEqual(result['action_gap_counts'],{'missing':1,'unknown':1})

    def test_score_bundle_survives_transport_location_loss(self):
        from fuxi_v1.artifacts import publish
        with scratch_directory() as root:
            cs,ns,cl,nl,market=fixture(root)
            try:
                scores=evaluate_ast({'field':'close'},cs)
                transport=Path(root)/'transport'
                reference=pack_grids(scores,transport)
                bundle=publish(Path(root)/'bundles',grid_bundle_files(scores,reference),{'purpose':'transport_contract_fixture'})
                transport.rename(Path(root)/'old_transport')
                actual=load_grid_bundle(bundle)
                self.assertEqual(digest(actual),digest(scores))
                actual.matrix._mmap.close();actual.membership._mmap.close()
            finally:cs.close();cl.close();market.close()

    def test_empty_finite_emitter_produces_real_novel_quota(self):
        from fuxi_v1.dsl import compile_ast
        fields={'close':{'unit':'price','frequency':'daily','adjustment':'raw'},
                'volume':{'unit':'shares','frequency':'daily','adjustment':'raw'}}
        seen=set()
        for i in range(20):
            proposals,receipt=complete_generation([],fields,2,seen,100029+i)
            self.assertEqual(len(proposals),2);self.assertEqual(receipt['supplemented'],2)
            for p in proposals:
                self.assertEqual(compile_ast(p['canonical_ast'],fields)['formula_hash'],p['formula_hash'])
                self.assertNotIn(p['formula_hash'],seen);seen.add(p['formula_hash'])
        self.assertEqual(len(seen),40)

    def test_streaming_canonical_preserves_existing_json_bytes(self):
        values=[{'中文':[-0.0,None,True,1.23456789123456],'x':{'z':'é\n'}},[1,False,'quotes"\\'],{}]
        for value in values:
            expected=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
            self.assertEqual(canonical(value),expected)
            self.assertEqual(digest(value),hashlib.sha256(expected).hexdigest())
        with self.assertRaises(ValueError):canonical({'v':float('nan')})

    def test_factories_and_hash_bound_grid_transport(self):
        with scratch_directory() as root:
            cs,ns,cl,nl,market=fixture(root)
            loaded=[]
            try:
                new_s=Snapshot(cs.payload);new_l=Labels(cl.payload,new_s)
                loaded.extend([new_s,new_l])
                self.assertEqual(new_s.hash,cs.hash);self.assertEqual(new_l.hash,cl.hash)
                scores=evaluate_ast({'op':'rank','args':[{'op':'return','window':2,'args':[{'field':'close'}]}]},new_s)
                packed=pack_grids({'first':scores,'same':scores},Path(root)/'transport')
                self.assertEqual(packed['first'],packed['same'])
                actual=unpack_grids(packed)
                self.assertIs(actual['first'],actual['same'])
                self.assertGreater(sum(v is not None for row in actual['first'].values() for v in row.values()),0)
                self.assertEqual(digest(scores),digest(actual['first']))
                self.assertEqual(ordinal_identity(scores,cs.sessions[5],cs.sessions[-1]),
                                 ordinal_identity(actual['first'],cs.sessions[5],cs.sessions[-1]))
                actual['first'].matrix._mmap.close();actual['first'].membership._mmap.close()
                reference=packed['first'];path=Path(reference['values']['path']);raw=path.read_bytes()
                path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
                with self.assertRaises(ContractError):unpack_grids(packed)
            finally:
                for row in loaded:row.close()
                cs.close();cl.close();market.close()

    def test_unsupported_venue_preserved_and_never_granted(self):
        ir={'ir_hash':'frozen','decisions':[{'scores':{'830001.BJ':1.0,'600000.XSHG':2.0}}]}
        script,result=diagnostic_jq(ir,{}, {'jq_candidate_allowed':False,'receipt_id':None},None)
        self.assertEqual(result['unsupported_instruments'],['830001.BJ'])
        self.assertEqual(result['research_members_removed'],0)
        self.assertFalse(result['backtest_orders_allowed']);compile(script,'jq_strategy.py','exec')

    def test_qualified_ir_gets_explicit_jq_universe_adapter(self):
        from fuxi_v1.evidence import EvidenceStore
        from fuxi_v1.portfolio import compile_strategy,strategy_spec
        with scratch_directory() as root:
            cs,ns,cl,nl,market=fixture(root)
            try:
                scores=evaluate_ast({'field':'close'},cs)
                ir=compile_strategy(strategy_spec('adapter_fixture',top_n=2,max_weight=.5),
                                    scores,cs,digest(market['execution']))
                first=ir['decisions'][0]
                first['scores']={**first['scores'],'830001.BJ':1.0}
                first['universe']=list(first['universe'])+['830001.BJ']
                ir['ir_hash']=digest({k:v for k,v in ir.items() if k!='ir_hash'})
                store=EvidenceStore(Path(root)/'authority.sqlite')
                try:
                    rid=store.receipt('research_acceptance',{'ir_hash':ir['ir_hash'],
                                                             'jq_candidate_allowed':True})
                    script,compiler=diagnostic_jq(ir,market,
                                                  {'jq_candidate_allowed':True,'receipt_id':rid},store)
                    self.assertTrue(compiler['backtest_orders_allowed'])
                    self.assertEqual(compiler['purpose'],'research_backtest')
                    adapter=compiler['universe_adapter']
                    self.assertEqual(adapter['schema_version'],'fuxi.jq_universe_adapter.1')
                    self.assertEqual(adapter['unsupported_instruments'],['830001.BJ'])
                    self.assertEqual(adapter['research_members_removed'],1)
                    self.assertEqual(adapter['research_ir_hash'],ir['ir_hash'])
                    self.assertNotEqual(adapter['execution_ir_hash'],ir['ir_hash'])
                    self.assertEqual(compiler['ir_hash'],adapter['execution_ir_hash'])
                    self.assertNotIn(b'830001.BJ',script)
                    compile(script,'jq_strategy.py','exec')
                finally:
                    store.close()
            finally:cs.close();cl.close();market.close()

    def test_resource_options_cannot_change_legacy_batch(self):
        self.assertIsNone(execution_options({'snapshot':{'schema_version':'fuxi.snapshot.2'}}))
        with self.assertRaises(ContractError):execution_options({'snapshot':{'schema_version':'fuxi.snapshot.2'},
                  'full_market_execution':{'top_n':80}})

    def test_partial_halt_mark_stays_bound_to_raw_columns(self):
        from fuxi_v1.halt_evidence import partial_halt_mark,valid_partial_halt_mark
        with scratch_directory() as root:
            cs,ns,cl,nl,market=fixture(root)
            try:
                day=cs.sessions[7];code=cs.instruments[0];bar=dict(market.bar_index[(day,code)])
                records=[{'ts_code':'000001.SZ','trade_date':day.replace('-',''),
                          'suspend_type':'S','suspend_timing':'09:30-10:30'}]
                proof=partial_halt_mark(records,code,day,bar['close'],bar['volume'],'halt_hash','price_hash')
                self.assertIsNotNone(proof)
                bar.update(paused=True,mark_source='observed_close_on_vendor_partial_halt',partial_halt_mark=proof)
                snapshot=dict(cs.payload,provenance={'source_hashes':{'suspend_d.csv':'halt_hash','daily_prices.csv':'price_hash'}},
                    arrays=dict(cs.payload['arrays'],fields={'close':cs.payload['arrays']['fields']['close'],
                    'volume':market.manifest_payload['arrays']['numeric']['volume']}))
                self.assertTrue(valid_partial_halt_mark(bar,snapshot))
                altered=dict(bar,close=bar['close']*2)
                self.assertFalse(valid_partial_halt_mark(altered,snapshot))
            finally:cs.close();cl.close();market.close()


if __name__=='__main__':unittest.main()
