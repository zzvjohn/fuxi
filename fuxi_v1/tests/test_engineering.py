from copy import deepcopy
from pathlib import Path
import sqlite3
from threading import Barrier, Thread
import unittest
from fuxi_v1.contracts import ContractError, digest, MARKET_CONTENT_EXCLUSIONS
from fuxi_v1.evidence import EvidenceStore, _transient_write_error
from fuxi_v1.execution import PaperGateway, reconcile_rows
from fuxi_v1.safety import ExecutionController
from fuxi_v1.historical_adapter import HistoricalSnapshot
from fuxi_v1.online import daily_decision
from fuxi_v1.portfolio import strategy_spec
from fuxi_v1.account import Account
from fuxi_v1.readiness import assess
from fuxi_v1.operations import operate
from fuxi_v1.acceptance import AcceptanceService
from fuxi_v1.contracts import evaluation
from fuxi_v1.simulation import run_account
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.data import Snapshot
from fuxi_v1.dsl import evaluate_ast
from fuxi_v1.portfolio import compile_strategy
from fuxi_v1.tests.test_certification import reviewed_certification_fixture
from fuxi_v1.search import node
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.tests.test_certification import fixture_payload, run, LIVE, jq_code


def fault_checks(path):
    """Isolated simulated receipts: software verification, never real release evidence."""
    store=EvidenceStore(path)
    health={'account':'test','asof':'2026-10-02T10:00:00+08:00','nav':1000.,'cash':1000.,'receivables':0.,
            'account_matched':True,'held_marks_known':True,'events_resolved':True,'data_contract_active':True}
    row={'account':'test','key':health['asof'],'holdings':{},'cash':1000.,'receivables':0.,'nav':1000.}
    receipt=store.receipt('paper_activation',{'mode':'paper','fixture_only':True})
    controller=ExecutionController(store,'test',receipt)
    payload={'mode':'paper','client_order_id':'decision:test:buy:A','instrument':'A','shares':100}
    checks={}
    def refused(name,fn):
        try:fn()
        except ContractError:checks[name]=True
        else:checks[name]=False
    refused('startup_halted',lambda:controller.order(payload,health,health['asof']))
    controller.reconcile_and_resume([row],[row],health,health['asof'])
    order=controller.order(payload,health,health['asof'])
    checks['duplicate_intent']=order==controller.order(payload,health,health['asof'])
    refused('conflicting_intent',lambda:controller.order(dict(payload,shares=200),health,health['asof']))
    controller.gateway.transition(order,controller.token,'submitted')
    refused('lost_ack_no_retry',lambda:controller.order(payload,health,health['asof']))
    controller.gateway.transition(order,controller.token,'unknown')
    refused('unknown_needs_reconciliation',lambda:controller.gateway.transition(order,controller.token,'acknowledged'))
    old_token=controller.token
    store.close()
    store=EvidenceStore(path);controller=ExecutionController(store,'test',receipt)
    refused('stale_owner',lambda:controller.gateway.transition(order,old_token,'acknowledged',reconciliation=True))
    refused('restart_stays_halted',lambda:controller.order(payload,health,health['asof']))
    controller.gateway.transition(order,controller.token,'acknowledged',reconciliation=True)
    checks['fill']=controller.gateway.fill_receipt('fill-test',order,{'shares':100,'account':'test','instrument':'A'})
    checks['duplicate_fill']=not controller.gateway.fill_receipt('fill-test',order,{'shares':100,'account':'test','instrument':'A'})
    refused('overfill',lambda:controller.gateway.fill_receipt('over',order,{'shares':1}))
    controller.gateway.transition(order,controller.token,'filled')
    controller.reconcile_and_resume([row],[row],health,health['asof'])
    for name,changed in [('unknown_nav',dict(health,nav=None)),('account_mismatch',dict(health,account='another')),
                          ('stale_data',dict(health,asof='2026-10-02T09:00:00+08:00')),
                          ('held_mark_missing',dict(health,held_marks_known=False))]:
        refused(name,lambda changed=changed:controller.order(payload,changed,health['asof']))
        controller.reconcile_and_resume([row],[row],health,health['asof'])
    store.revoke(receipt,'fixture revocation')
    refused('revoked_release',lambda:controller.order(payload,health,health['asof']))
    store.close()
    assert checks and all(checks.values()),checks
    return {'checks':checks,'scope':'isolated_simulated_execution_contract','actual_broker_observed':False}


class EngineeringTests(unittest.TestCase):
    def test_legacy_four_booleans_do_not_grant_investment_eligibility(self):
        with scratch_directory() as root:
            store=EvidenceStore(Path(root)/'legacy.sqlite')
            data=store.receipt('data_contract',{'snapshot_hash':'snapshot',**{k:True for k in
                ('pit_certified','universe_certified','account_events_certified','execution_rules_certified')}})
            result={'candidate_id':'test','source_scope':'research_exposed','snapshot_hash':'snapshot',
                    'policy_hash':'policy','strategy_ir':{'ir_hash':'ir'},
                    'evaluations':[evaluation(stage,'valid','fixture_only',decision='pass') for stage in ('E0','E1','E2','E3','E4')]}
            admission=AcceptanceService(store).research(store.append('evaluation',result),data)
            self.assertFalse(admission['jq_candidate_allowed'])
            self.assertIn('derived_validation_cannot_grant_investment_eligibility',admission['blockers'])
            store.close()

    def test_daily_input_hash_binds_instrument_column_mapping(self):
        p=fixture_payload()['snapshot'];ast=node('rank',{'field':'close'});spec=strategy_spec('mapping')
        d=p['sessions'][-1];clock=p['decision_times'][d]
        a=daily_decision(ast,HistoricalSnapshot(p),d,clock,spec)
        q=deepcopy(p);q['instruments']=list(reversed(q['instruments']))
        b=daily_decision(ast,HistoricalSnapshot(q),d,clock,spec)
        self.assertNotEqual(a['input_hash'],b['input_hash'])

    def test_presence_of_certification_hashes_is_not_account_authority(self):
        p=synthetic_campaign(40,6);s=Snapshot(p['snapshot']);m=p['market']
        m.update(event_coverage_certified=True,tradability_certified=True,
                 certification_hash='fixture-certificate',certification_snapshot_hash=s.hash)
        scores=evaluate_ast(node('rank',node('return',{'field':'close'},window=2)),s)
        ir=compile_strategy(strategy_spec('fixture',top_n=3),scores,s,digest(m['execution']))
        self.assertFalse(run_account(ir,m)['formal_evidence'])
        binding={'schema_version':'fuxi.market_assurance.1','snapshot_hash':s.hash,
                 'market_digest':digest({k:v for k,v in m.items() if k not in MARKET_CONTENT_EXCLUSIONS}),
                 'receipt_id':'isolated-test-receipt','assurance_hash':'isolated-test-review'}
        m['investment_assurance']=binding
        self.assertTrue(run_account(ir,m)['formal_evidence'])
        changed=deepcopy(m);changed['bars'][0]['close']+=.01
        self.assertFalse(run_account(ir,changed)['formal_evidence'])
        other=deepcopy(ir);other['snapshot_hash']='other-snapshot'
        other['ir_hash']=digest({k:v for k,v in other.items() if k!='ir_hash'})
        self.assertFalse(run_account(other,m)['formal_evidence'])

    def test_intent_and_fact_commit_atomically(self):
        with scratch_directory() as root:
            s=EvidenceStore(Path(root)/'atomic.sqlite');g=PaperGateway(s)
            receipt=s.receipt('paper_activation',{'fixture':True});token=g.take_ownership('a')
            append=s.append
            def fail(kind,*args,**kwargs):
                if kind=='order_intent':raise OSError('injected journal failure')
                return append(kind,*args,**kwargs)
            s.append=fail
            with self.assertRaises(OSError):g.intent('a',token,{'mode':'paper','shares':100},receipt)
            self.assertEqual(s.db.execute('SELECT count(*) FROM orders').fetchone()[0],0)
            s.close()

    def test_derived_clean_certificate_cannot_mint_investment_receipt(self):
        with scratch_directory() as root:
            s=EvidenceStore(Path(root)/'data.sqlite')
            result=operate(s,'register-data',reviewed_certification_fixture(Path(root)))
            self.assertIsNone(result['receipt_id'])
            self.assertTrue(result['validation_receipt_id'])
            self.assertEqual(s.db.execute("SELECT count(*) FROM receipts WHERE kind='data_contract'").fetchone()[0],0)
            s.close()

    def test_ten_software_checks_do_not_enable_live_release(self):
        with scratch_directory() as root:
            s=EvidenceStore(Path(root)/'readiness.sqlite')
            r=assess(s,{str(i):True for i in range(10)})
            self.assertTrue(r['software_ready']);self.assertFalse(r['paper_release_ready'])
            self.assertFalse(r['live_execution_allowed']);self.assertEqual(len(r['blockers']),4)
            s.close()

    def test_security_transfer_needs_exact_evidence_and_target_mark(self):
        a=Account('10000');clock='2026-10-01T09:30:00+08:00'
        def post(key,kind,payload,at=clock):
            return a.apply({'event_id':key,'kind':kind,'timestamp':at,'evidence_id':'fixture:'+key,'payload':payload})
        post('buy','fill',{'instrument':'old','side':'buy','quantity':100,'price':'10','fee':'0','sellable_at':clock})
        post('unknown','unknown',{'instrument':'old','reason':'transfer_confirmation_missing'})
        body={'from_instrument':'old','to_instrument':'new','quantity':100,'sellable_at':clock,'resolved_event_ids':['unknown']}
        with self.assertRaises(ContractError):post('bad','security_transfer',dict(body,quantity=99))
        self.assertIn('old',a.positions);self.assertEqual(a.cash,9000)
        post('transfer','security_transfer',body)
        self.assertIsNone(a.mark(clock,{})['nav'])
        nav=a.mark(clock,{'new':{'price':'10','asof':clock,'evidence_id':'target_quote'}})
        self.assertEqual(nav['nav'],'10000');self.assertEqual(nav['holdings'],{'new':100})
        self.assertEqual(post('transfer','security_transfer',body),'already_applied')

    def test_durable_fault_campaign(self):
        with scratch_directory() as root:
            self.assertTrue(all(fault_checks(Path(root)/'fault.sqlite')['checks'].values()))

    def test_nested_parity_and_unknown_account_facts(self):
        a={'key':'d','holdings':{'A':100},'nav':100.}
        self.assertTrue(reconcile_rows([a],[dict(a,nav=100.+1e-10)],['holdings','nav'])['accepted'])
        for value in (None,float('nan'),float('inf'),True):
            x=dict(a,nav=value)
            self.assertFalse(reconcile_rows([x],[x],['nav'])['accepted'])
        self.assertFalse(reconcile_rows([a],[dict(a,holdings={'A':True})],['holdings'])['accepted'])

    def test_simultaneous_fills_cannot_overfill(self):
        with scratch_directory() as root:
            path=Path(root)/'concurrent.sqlite';store=EvidenceStore(path);gate=PaperGateway(store)
            receipt=store.receipt('paper_activation',{'fixture_only':True});token=gate.take_ownership('a')
            order=gate.intent('a',token,{'mode':'paper','shares':100},receipt)
            gate.transition(order,token,'submitted');gate.transition(order,token,'acknowledged');store.close()
            barrier=Barrier(2);results=[]
            def fill(i):
                s=EvidenceStore(path);g=PaperGateway(s);barrier.wait()
                try:results.append(g.fill_receipt(str(i),order,{'shares':100}))
                except ContractError:results.append(False)
                finally:s.close()
            threads=[Thread(target=fill,args=(i,)) for i in range(2)]
            for thread in threads:thread.start()
            for thread in threads:thread.join(10);self.assertFalse(thread.is_alive())
            self.assertEqual(sorted(results),[False,True])

    def test_wal_init_race_is_healed_during_construction(self):
        # Connections opening one fresh WAL database race to initialize the
        # shared-memory index; the loser is left permanently read-only and must
        # be repaired by discarding that connection, never surfaced raw.
        # Four simultaneous constructions fired the race in every one of 15
        # stress trials and every trial healed with zero failures.
        with scratch_directory() as root:
            path=Path(root)/'race.sqlite';EvidenceStore(path).close()  # -wal/-shm removed
            barrier=Barrier(4);results=[]
            def worker(i):
                store=EvidenceStore(path)
                barrier.wait()
                with store.transaction():store.append('race',{'worker':i,'heals':store.self_heals})
                store.close();results.append(i)
            threads=[Thread(target=worker,args=(i,)) for i in range(4)]
            for thread in threads:thread.start()
            for thread in threads:thread.join(30);self.assertFalse(thread.is_alive())
            self.assertEqual(sorted(results),[0,1,2,3])
            check=EvidenceStore(path)
            rows=check.db.execute("SELECT count(*) FROM facts WHERE kind='race'").fetchone()[0]
            check.close();self.assertEqual(rows,4)

    def test_transient_write_error_matches_only_the_wal_race_signature(self):
        self.assertTrue(_transient_write_error(sqlite3.OperationalError("attempt to write a readonly database")))
        self.assertTrue(_transient_write_error(sqlite3.OperationalError("ATTEMPT TO WRITE A READONLY DATABASE")))
        self.assertFalse(_transient_write_error(sqlite3.OperationalError("database is locked")))
        self.assertFalse(_transient_write_error(sqlite3.OperationalError("no such table: facts")))
        self.assertFalse(_transient_write_error(sqlite3.ProgrammingError("attempt to write a readonly database")))

    def test_closed_store_rejects_transactions_instead_of_reopening(self):
        with scratch_directory() as root:
            store=EvidenceStore(Path(root)/'closed.sqlite');store.close()
            with self.assertRaises(sqlite3.ProgrammingError):
                with store.transaction():pass
            self.assertEqual(store.self_heals,0)

    def test_removed_member_still_requires_event_and_quote_evidence(self):
        p=fixture_payload(universe_overrides={'2021-01-05':LIVE[1:],'2021-01-06':LIVE[1:]})
        for b in p['market']['bars']:
            if b['session']=='2021-01-05' and b['instrument']==jq_code(LIVE[0]):
                b['corporate_action_known']=False;b['paused']=None
        c=run(p)
        self.assertFalse(c['flags']['account_events_certified']['value'])
        self.assertFalse(c['flags']['tradability_certified']['value'])

    def test_missing_bar_is_not_silent_full_coverage(self):
        p=fixture_payload();p['market']['bars']=p['market']['bars'][1:]
        c=run(p)
        self.assertFalse(c['flags']['account_events_certified']['value'])

    def test_clock_comparison_uses_timezone(self):
        p=fixture_payload()
        for d in p['snapshot']['sessions']:p['snapshot']['decision_times'][d]=d+'T11:00:00+00:00'
        self.assertTrue(run(p)['flags']['pit_certified']['value'])  # 19:00 in Shanghai

    def test_daily_scores_cannot_read_future_or_use_batch_cache(self):
        p=fixture_payload()['snapshot'];p['evaluation_start']=p['sessions'][1]
        for index,m in enumerate(p['columns']['close']):
            for j in range(len(m)):m[j]=10+index*(j+1)
        ast=node('rank',node('return',{'field':'close'},window=1));spec=strategy_spec('daily')
        s=HistoricalSnapshot(p);s._score_cache[digest(ast)]={'poison':{}}
        d=p['sessions'][1];clock=p['decision_times'][d]
        a=daily_decision(ast,s,d,clock,spec)
        future=deepcopy(p);future['columns']['close'][-1]=[9999.]*len(p['instruments'])
        b=daily_decision(ast,HistoricalSnapshot(future),d,clock,spec)
        self.assertEqual(a,b)
        with self.assertRaises(ContractError):daily_decision(ast,s,d,d+'T17:00:00+08:00',spec)
