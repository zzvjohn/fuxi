"""Controller integration with real leases/SQLite/processes; costly research is mocked.

All temporary workspaces live in the explicitly writable visualization tree.
Fixtures never edit production state, acquire market data or grant qualification.
"""
import ast
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from datetime import datetime as RealDatetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from fuxi_v1.contracts import ContractError, digest, load_json
from fuxi_v1.governance import governed_writer, SAFE_AUTOMATIC_ACTIONS, LEASE_ENV
from fuxi_v1 import unified_runner as controller
from fuxi_v1.tests.test_fusion_study import _fixture


WORKSPACE=Path(__file__).resolve().parents[2]
TEMP_BASE=Path('C:/Users/Zhong/.codex/visualizations/2026/10/09/01a12085-dfb6-76e3-bda3-164d4e9a37b9/unified_integration')
CODE_IDENTITY={'native':{'fixture.py':digest('controller-integration-fixture')}, 'legacy_boundaries':{}}


def fixture_policy():
    return {'single_writer_confirmed':True, 'current_mode':'controlled_manual',
        'autonomous_writes_enabled':False, 'bounded_research':{
            'enabled':True, 'allowed_actions':sorted(SAFE_AUTOMATIC_ACTIONS),
            'budgets':deepcopy(controller.DEFAULT_LIMITS),
            'authorization':{'authorized':True, 'source':'isolated controller integration fixture',
                             'recorded_at':'2026-10-09T20:00:00+08:00'},
            'approved_code_identity':digest(CODE_IDENTITY), 'approved_request_hashes':[]}}


@contextmanager
def workspace():
    TEMP_BASE.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='controller_',dir=TEMP_BASE) as tmp:
        root=Path(tmp);(root/'codex_automations').mkdir()
        policy=fixture_policy()
        controller.atomic(root/'codex_automations/fuxi_runtime_policy.json',policy)
        source=root/'input.json';controller.atomic(source,_fixture())
        request={'input':str(source),'input_sha256':controller.file_sha(source),
                 'rounds':1,'candidates_per_round':2,'include_catalog':False}
        normalized=controller.normalize_request(root,request)
        policy['bounded_research']['approved_request_hashes']=[digest(normalized)]
        controller.atomic(root/'codex_automations/fuxi_runtime_policy.json',policy)
        yield root,request,policy


def child(code,env=None):
    selected=dict(os.environ if env is None else env)
    selected['TEMP']=selected['TMP']=str(TEMP_BASE)
    selected['PYTHONPATH']=''
    return subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=WORKSPACE,
        env=selected,capture_output=True,text=True,encoding='utf8',timeout=20)


def events(root):
    db=sqlite3.connect((root/'data/fuxi_unified/runs.sqlite').as_uri()+'?mode=ro',uri=True)
    try:return [(kind,json.loads(payload)) for kind,payload in db.execute('SELECT kind,payload FROM events ORDER BY id')]
    finally:db.close()


@contextmanager
def consumers(root,code_identity=None):
    """Replace only expensive research/feedback, retain real controller worker tree."""
    counters={'workers':0,'feedback':0};identity=code_identity or deepcopy(CODE_IDENTITY)
    real_process=controller.bounded_process
    def consumer_process(command,directory,env,*,timeout,memory_mb):
        action=command[command.index('fuxi_v1.unified_runner')+1]
        path=Path(command[command.index('--request')+1])
        if action=='_campaign':
            counters['workers']+=1;writer_action='candidate_search'
            body=(" report=out/'fixture_report.json'\n"
                " atomic(report,{'engineering_fixture_only':True,'campaign_id':spec['campaign_id']})\n"
                " atomic(out/'RECEIPT.json',{'report_path':str(report),'report_sha256':file_sha(report),"
                "'status':'completed','attempted_candidates':2,'valid_descriptive_factors':2,"
                "'engineering_errors':0,'formal_account_valid':0,'economic_reward_count':0,'jq_candidate_count':0})\n")
        elif action=='_feedback':
            counters['feedback']+=1;writer_action='feedback'
            body=(" feedback={'report_sha256':file_sha(spec['report_path']),'qualification_transferred':False,'fixture_only':True}\n"
                  " atomic(out/'projection.json',feedback)\n atomic(out/'RECEIPT.json',feedback)\n")
        else:raise AssertionError('include_catalog=False fixture has unexpected phase: '+action)
        code=("import time\nfrom pathlib import Path\nfrom fuxi_v1.contracts import load_json\n"
            "from fuxi_v1.unified_runner import atomic,file_sha\n"
            "from fuxi_v1.governance import governed_writer\n"
            f"spec=load_json({str(path)!r})\n"
            f"with governed_writer({str(root)!r},{writer_action!r},trigger={json.loads(env[LEASE_ENV])['trigger']!r}) as owner:\n"
            " out=Path(spec['output']);out.mkdir(parents=True,exist_ok=True)\n"
            +body+
            " print(owner['inherited'],owner['action'],flush=True)\n time.sleep(.25)\n")
        return real_process([sys.executable,'-B','-X','utf8','-c',code],directory,env,timeout=timeout,memory_mb=memory_mb)
    with ExitStack() as stack:
        stack.enter_context(patch('fuxi_v1.unified_runner.source_identity',side_effect=lambda _:deepcopy(identity)))
        stack.enter_context(patch('fuxi_v1.legacy_writers.inventory',return_value={'all_declared_boundaries_present':True,'fixture_only':True}))
        stack.enter_context(patch('fuxi_v1.unified_runner.bounded_process',side_effect=consumer_process))
        yield counters,identity


class UnifiedIntegrationTests(unittest.TestCase):
    def test_manual_automatic_and_later_tick_are_one_verified_work(self):
        class LaterDate(RealDatetime):
            @classmethod
            def now(cls,tz=None):return cls(2031,1,12,tzinfo=timezone.utc)
        with workspace() as (root,request,policy),consumers(root) as (calls,_):
            first=controller.run(root,request,trigger='manual')
            self.assertFalse(first['cached']);self.assertEqual(calls,{'workers':1,'feedback':1})
            with patch('fuxi_v1.unified_runner.datetime',LaterDate):
                second=controller.run(root,request,trigger='automatic')
                controller.atomic(root/'data/fuxi_unified/inbox/frozen.json',request)
                tick=controller.tick(root)
            self.assertEqual(first['run_id'],second['run_id'])
            self.assertTrue(second['cached']);self.assertEqual(tick['state'],'idle')
            self.assertEqual(calls,{'workers':1,'feedback':1})
            observed=events(root)
            self.assertEqual(sum(k=='completed' for k,_ in observed),1)
            cached=[p for k,p in observed if k=='cached_trigger']
            self.assertEqual(len(cached),1);self.assertEqual(cached[0]['new_workers'],0)
            self.assertFalse((root/'data/fuxi_runtime.lock').exists())

    def test_completed_artifact_tamper_refuses_without_another_worker(self):
        with workspace() as (root,request,_),consumers(root) as (calls,_):
            first=controller.run(root,request)
            report=Path(first['research']['report_path']);report.write_bytes(b'{"tampered":true}\n')
            with self.assertRaisesRegex(ContractError,'content changed'):controller.run(root,request)
            self.assertEqual(calls['workers'],1);self.assertEqual(calls['feedback'],1)
            self.assertTrue((report.parent.parent/'LAST_FAILURE.json').is_file())
            self.assertTrue(any(k=='failed' and 'content changed' in p['reason'] for k,p in events(root)))
            # The failed verification changes the SQLite state; COMPLETE must
            # still bar another worker on every later trigger.
            with self.assertRaisesRegex(ContractError,'content changed'):controller.run(root,request)
            self.assertEqual(calls,{'workers':1,'feedback':1})
            self.assertEqual(report.read_bytes(),b'{"tampered":true}\n')

    def test_new_code_and_unfrozen_request_auto_refuse_with_zero_worker_receipts(self):
        with workspace() as (root,request,_),consumers(root) as (calls,identity):
            identity['native']['fixture.py']=digest('unapproved-code')
            with self.assertRaisesRegex(ContractError,'approved release'):controller.run(root,request,trigger='automatic')
            identity.clear();identity.update(deepcopy(CODE_IDENTITY))
            changed=dict(request,seed=18)
            with self.assertRaisesRegex(ContractError,'not frozen'):controller.run(root,changed,trigger='automatic')
            failures=[load_json(p) for p in (root/'data/fuxi_unified/rejections').glob('*.json')]
            self.assertEqual(len(failures),2)
            self.assertTrue(all(r['new_workers']==r['new_research_facts']==0 for r in failures))
            self.assertEqual(calls['workers'],0);self.assertFalse((root/'data/fuxi_unified/runs.sqlite').exists())

    def test_each_budget_rejection_persists_without_launch_or_qualification(self):
        cases=[('max_rounds',4),('max_candidates',7),('max_worker_seconds',60),
               ('max_total_seconds',500),('max_memory_mb',1024),('max_external_requests',1),
               ('max_llm_calls',1),('max_output_tokens',1)]
        with workspace() as (root,request,_),consumers(root) as (calls,_):
            for axis,value in cases:
                limits={**controller.DEFAULT_LIMITS,axis:value}
                with self.subTest(axis=axis),self.assertRaises(ContractError):
                    controller.run(root,{**request,'limits':limits},trigger='automatic')
            receipts=[load_json(p) for p in (root/'data/fuxi_unified/rejections').glob('*.json')]
            self.assertEqual(len(receipts),len(cases));self.assertEqual(calls['workers'],0)
            self.assertTrue(all(r['state']=='refused' and r['reason'] and r['new_workers']==0 for r in receipts))
            self.assertFalse((root/'data/fuxi_runtime.lock').exists())

    def test_real_limit_monitor_launch_and_cleanup_errors_keep_resource_evidence(self):
        import psutil
        from fuxi_v1.controller_processes import ControllerTreeGuard
        original_stop=ControllerTreeGuard.stop
        def stopped_but_error(guard,*args,**kwargs):
            original_stop(guard,*args,**kwargs)
            raise RuntimeError('fixture_cleanup_observability_failure')
        with workspace() as (root,_,__):
            for kind in ('timeout','memory','monitor','launch','cleanup','kill','diagnostics'):
                directory=root/kind
                env=dict(os.environ,TEMP=str(TEMP_BASE),TMP=str(TEMP_BASE),PYTHONPATH='')
                with ExitStack() as stack:
                    if kind=='monitor':stack.enter_context(patch.object(ControllerTreeGuard,'sample',side_effect=RuntimeError('fixture_original_monitor_failure')))
                    if kind=='launch':stack.enter_context(patch('fuxi_v1.unified_runner.subprocess.Popen',side_effect=OSError('fixture_original_launch_failure')))
                    if kind=='cleanup':stack.enter_context(patch.object(ControllerTreeGuard,'stop',stopped_but_error))
                    if kind=='kill':stack.enter_context(patch.object(ControllerTreeGuard,'stop',side_effect=RuntimeError('fixture_kill_failure')))
                    if kind=='diagnostics':stack.enter_context(patch.object(ControllerTreeGuard,'diagnostics',side_effect=RuntimeError('fixture_diagnostics_failure')))
                    with self.subTest(kind=kind),self.assertRaises((ContractError,RuntimeError,OSError)):
                        controller.bounded_process([sys.executable,'-B','-c','import time;time.sleep(10)'],
                            directory,env,timeout=.1 if kind in ('timeout','cleanup','kill','diagnostics') else 3,
                            memory_mb=0 if kind=='memory' else 2048)
                resource=load_json(directory/'RESOURCE.json')
                self.assertTrue(resource['reason'])
                if kind=='monitor':self.assertEqual(resource['original_error']['reason'],'fixture_original_monitor_failure')
                if kind=='launch':self.assertEqual(resource['original_error']['reason'],'fixture_original_launch_failure')
                if kind=='cleanup':self.assertEqual(resource['cleanup_error']['reason'],'fixture_cleanup_observability_failure')
                if kind=='kill':
                    self.assertEqual(resource['original_error']['reason'],'fixture_kill_failure')
                    self.assertEqual(resource['cleanup_error']['reason'],'fixture_kill_failure')
                if kind!='launch':self.assertIsNotNone(resource['exit_code'],'owned launcher was not reaped')
                if kind=='diagnostics':
                    self.assertIn('fixture_diagnostics_failure',json.dumps(resource))
                    self.assertTrue(resource['tree']['descendants_unverified'])
                for observed in (resource.get('tree') or {}).get('observed_processes',[]):
                    try:process=psutil.Process(observed['pid'])
                    except psutil.NoSuchProcess:continue
                    self.assertNotEqual(process.create_time(),observed['create_time'],'fixture process survived cleanup')

    def test_actual_legacy_save_rejects_missing_forged_and_other_thread_owners(self):
        import psutil
        from research.factor_alchemy.experience_memory import ExperienceMemory
        with workspace() as (root,_,__):
            memory=ExperienceMemory.__new__(ExperienceMemory)
            memory.path=root/'data/nested/experience_memory.json';memory.data={'attempts':[]}
            with self.assertRaises(ContractError):memory._save()
            self.assertFalse(memory.path.parent.exists())
            controller.atomic(root/'data/fuxi_runtime.lock',{'pid':os.getpid(),'process_create_time':psutil.Process().create_time()})
            with self.assertRaises(ContractError):memory._save()
            self.assertFalse(memory.path.exists())
            (root/'data/fuxi_runtime.lock').unlink()
            with governed_writer(root,'feedback'):
                memory._save();before=controller.file_sha(memory.path)
                exceptions=[]
                def stale_save():
                    try:memory._save()
                    except Exception as exc:exceptions.append(exc)
                thread=threading.Thread(target=stale_save);thread.start();thread.join(5)
                self.assertFalse(thread.is_alive());self.assertEqual(len(exceptions),1)
                self.assertIsInstance(exceptions[0],ContractError);self.assertEqual(controller.file_sha(memory.path),before)
            with self.assertRaises(ContractError):memory._save()

    def test_actual_backfill_cli_branch_checks_guard_before_dry_apply_write(self):
        with workspace() as (root,_,policy):
            policy['single_writer_confirmed']=False
            controller.atomic(root/'codex_automations/fuxi_runtime_policy.json',policy)
            entry=root/'research/factor_alchemy/trigger_d_plus.py';entry.parent.mkdir(parents=True)
            entry.write_bytes((WORKSPACE/'research/factor_alchemy/trigger_d_plus.py').read_bytes())
            marker=root/'backfill_wrote.json'
            code=("import ast,sys\nfrom pathlib import Path\n"
                f"source=Path({str(entry)!r})\n"
                "tree=ast.parse(source.read_bytes())\n"
                "block=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')\n"
                "sys.argv=[str(source),'--dry','--backfill','--apply']\n"
                f"def write_fixture(**kwargs):Path({str(marker)!r}).write_text(str(kwargs),encoding='utf8')\n"
                "scope={'__name__':'__main__','__file__':str(source),'sys':sys,'run_backfill':write_fixture,'run':write_fixture}\n"
                "exec(compile(ast.Module(body=[block],type_ignores=[]),str(source),'exec'),scope)\n")
            result=child(code)
            self.assertNotEqual(result.returncode,0,result.stdout)
            self.assertIn('single writer authorization missing',result.stderr)
            self.assertFalse(marker.exists(),'--dry hid the actual --apply backfill write')

    def test_actual_module_guard_precedes_credential_or_data_side_effect(self):
        with workspace() as (root,_,policy):
            policy['single_writer_confirmed']=False
            controller.atomic(root/'codex_automations/fuxi_runtime_policy.json',policy)
            entry=root/'scripts/stage0_daily_update_v3.py';entry.parent.mkdir()
            entry.write_bytes((WORKSPACE/'scripts/stage0_daily_update_v3.py').read_bytes())
            marker=root/'credential_side_effect.json'
            code=("import ast\nfrom pathlib import Path\n"
                f"source=Path({str(entry)!r});tree=ast.parse(source.read_bytes());prefix=[]\n"
                "for node in tree.body:\n"
                " if isinstance(node,ast.ImportFrom) and node.module=='credentials':break\n"
                " prefix.append(node)\n"
                "scope={'__name__':'__main__','__file__':str(source)}\n"
                "exec(compile(ast.Module(body=prefix,type_ignores=[]),str(source),'exec'),scope)\n"
                f"Path({str(marker)!r}).write_text('unexpected credential/data work',encoding='utf8')\n")
            result=child(code)
            self.assertNotEqual(result.returncode,0,result.stdout)
            self.assertIn('single writer authorization missing',result.stderr)
            self.assertFalse(marker.exists(),'data/credential work happened before actual module guard')


if __name__=='__main__':unittest.main()
