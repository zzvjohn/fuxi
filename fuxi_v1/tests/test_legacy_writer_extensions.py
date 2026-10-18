"""Actual added legacy boundaries in isolated policy/data-shaped workspaces."""
import argparse
import ast
import builtins
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from fuxi_v1.contracts import ContractError, load_json
from fuxi_v1.governance import lock_module
from fuxi_v1.unified_runner import atomic, file_sha
from fuxi_v1.tests.test_unified_integration import TEMP_BASE, WORKSPACE, child, fixture_policy


@contextmanager
def isolated_workspace(*,authorized=True):
    TEMP_BASE.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='legacy_extensions_',dir=TEMP_BASE) as tmp:
        root=Path(tmp);policy=fixture_policy();policy['single_writer_confirmed']=authorized
        atomic(root/'codex_automations/fuxi_runtime_policy.json',policy)
        yield root


def copy_entry(root,name):
    target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes((WORKSPACE/name).read_bytes())
    return target


@contextmanager
def actual_savers(root):
    from research.factor_alchemy.library_orthogonality import LibraryOrthogonalityManager
    from research.factor_alchemy.subtree_fingerprinter import SubtreeFingerprinter
    from research.factor_alchemy.factor_model_cooptim import FactorModelCoOptimizer
    from research.factor_alchemy import llm_evolution
    library=LibraryOrthogonalityManager.__new__(LibraryOrthogonalityManager)
    library.data_dir=root/'data/library';library._state_path=library.data_dir/'state.json'
    library.factors={};library.clusters=[];library.forbidden_regions=[];library._factor_names=[]
    library._last_recluster_time=None;library._factor_count_at_recluster=0;library.auto_rebuild_log=[]
    library._corr_matrix=None;library.get_library_stats=lambda:{'fixture_only':True}
    subtree=SubtreeFingerprinter.__new__(SubtreeFingerprinter)
    subtree.state_path=root/'data/subtree/state.json';subtree.threshold=.15
    subtree.min_factors_to_trigger=2;subtree.skeletons={}
    cooptim=FactorModelCoOptimizer.__new__(FactorModelCoOptimizer)
    cooptim.data_dir=root/'data/cooptim';cooptim._results_path=cooptim.data_dir/'results.json'
    cooptim.results=[]
    evolution_path=root/'data/evolution/state.json';evolution_state={'rounds':list(range(23))}
    # The unchanged legacy function hands json.dump an open handle. Retain and
    # close that real handle in the fixture so its existing IO style is scoped.
    handles=[]
    def tracked_open(*args,**kwargs):
        handle=builtins.open(*args,**kwargs);handles.append(handle);return handle
    def save_evolution():
        llm_evolution.save_state(evolution_state)
        for handle in handles:
            if not handle.closed:handle.flush()
    with patch.object(llm_evolution,'STATE_PATH',evolution_path),patch.object(llm_evolution,'open',tracked_open,create=True):
        cases={
            'library':(library._state_path,library.save_state,None),
            'evolution':(evolution_path,save_evolution,evolution_state),
            'subtree':(subtree.state_path,subtree._save_state,None),
            'cooptim_results':(cooptim._results_path,cooptim._save_results,None)}
        try:yield cases
        finally:
            for handle in handles:handle.close()


class LegacyWriterExtensionTests(unittest.TestCase):
    def test_run_v4_actual_writer_branches_guard_before_consumer(self):
        with isolated_workspace(authorized=False) as root:
            source=copy_entry(root,'research/factor_alchemy/run_v4_pipeline.py')
            for flag in ('--diagnose','--explore','--ralph'):
                marker=root/('unexpected_'+flag[2:]+'.json')
                code=("import ast,os,sys\nfrom pathlib import Path\n"
                    f"source=Path({str(source)!r});tree=ast.parse(source.read_bytes())\n"
                    "block=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')\n"
                    f"sys.argv=[str(source),{flag!r}]\n"
                    f"def consumer(*args,**kwargs):Path({str(marker)!r}).write_text('consumer reached',encoding='utf8')\n"
                    "scope={'__name__':'__main__','__file__':str(source),'Path':Path,'os':os,'run_ralph':consumer,'run_explore':consumer,'run_diagnose':consumer,'run_view':consumer}\n"
                    "exec(compile(ast.Module(body=[block],type_ignores=[]),str(source),'exec'),scope)\n")
                with self.subTest(flag=flag):
                    result=child(code)
                    self.assertNotEqual(result.returncode,0,result.stdout)
                    self.assertIn('single writer authorization missing',result.stderr)
                    self.assertFalse(marker.exists())

    def test_run_v4_actual_view_remains_read_only_without_writer_authorization(self):
        with isolated_workspace(authorized=False) as root:
            source=copy_entry(root,'research/factor_alchemy/run_v4_pipeline.py')
            code=("import ast,os,sys\nfrom pathlib import Path\n"
                f"source=Path({str(source)!r});tree=ast.parse(source.read_bytes())\n"
                "block=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')\n"
                "sys.argv=[str(source),'--view']\n"
                "def view():print('fixture_view_reached')\n"
                "scope={'__name__':'__main__','__file__':str(source),'Path':Path,'os':os,'run_view':view}\n"
                "exec(compile(ast.Module(body=[block],type_ignores=[]),str(source),'exec'),scope)\n")
            result=child(code)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('fixture_view_reached',result.stdout)
            self.assertFalse((root/'data/fuxi_runtime.lock').exists())

    def test_stage2_actual_main_blocks_direct_pool_and_memory_writes(self):
        with isolated_workspace(authorized=False) as root:
            source=copy_entry(root,'scripts/stage2_pass_to_stage3.py');marker=root/'unexpected_bridge_load.json'
            code=("import argparse,ast,sys\nfrom pathlib import Path\n"
                f"source=Path({str(source)!r});tree=ast.parse(source.read_bytes())\n"
                "main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')\n"
                "sys.argv=[str(source)]\n"
                f"def load_before_guard():Path({str(marker)!r}).write_text('state read/modify started',encoding='utf8');return set()\n"
                "scope={'__name__':'fixture','__file__':str(source),'argparse':argparse,'load_existing_pool_names':load_before_guard}\n"
                "exec(compile(ast.Module(body=[main],type_ignores=[]),str(source),'exec'),scope);scope['main']()\n")
            result=child(code)
            self.assertNotEqual(result.returncode,0,result.stdout)
            self.assertIn('single writer authorization missing',result.stderr)
            self.assertFalse(marker.exists());self.assertFalse((root/'data').exists())

    def test_stage2_actual_dry_branch_reads_fixture_without_direct_rewrites(self):
        with isolated_workspace(authorized=False) as root:
            source=copy_entry(root,'scripts/stage2_pass_to_stage3.py')
            pool=root/'data/pool.csv';pool.parent.mkdir();pool.write_bytes(b'factor_name,formula\n')
            memory=root/'data/memory.json';atomic(memory,{'attempts':[],'success_templates':[],'stats':{}})
            before={p:file_sha(p) for p in (pool,memory)}
            tree=ast.parse(source.read_bytes());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
            calls=[]
            def memory_inject(factor,attempts,templates,state):calls.append('memory');state['attempts'].append(factor)
            scope={'__name__':'fixture','__file__':str(source),'argparse':argparse,'csv':__import__('csv'),'json':json,
                'POOL_CSV':pool,'MEMORY_FILE':memory,'load_existing_pool_names':lambda:set(),
                'load_existing_memory_names':lambda:(set(),set()),'load_pass_factors':lambda:[{'factor_name':'fixture','formula':'close_p'}],
                'inject_to_pool':lambda factor,names,fields:{'factor_name':'fixture','formula':'close_p'},
                'inject_to_memory':memory_inject}
            with patch('sys.argv',[str(source),'--dry-run']),patch('sys.path',list(sys.path)):
                exec(compile(ast.Module(body=[main],type_ignores=[]),str(source),'exec'),scope);scope['main']()
            self.assertEqual(calls,['memory'])
            self.assertEqual(before,{p:file_sha(p) for p in (pool,memory)})
            self.assertFalse((root/'data/fuxi_runtime.lock').exists())

    def _assert_saver_requires_owner(self,name):
        with isolated_workspace() as root,actual_savers(root) as cases:
            target,save,mutable=cases[name];before=deepcopy(mutable)
            with self.assertRaisesRegex(ContractError,'actual current-thread lock owner'):save()
            self.assertFalse(target.parent.exists());self.assertEqual(mutable,before)

    def test_library_save_refuses_before_state_or_directory_creation(self):self._assert_saver_requires_owner('library')
    def test_llm_evolution_refuses_before_mutation_and_nonblocking_try(self):self._assert_saver_requires_owner('evolution')
    def test_subtree_save_refuses_before_state_or_directory_creation(self):self._assert_saver_requires_owner('subtree')
    def test_cooptim_results_refuses_before_state_or_directory_creation(self):self._assert_saver_requires_owner('cooptim_results')

    def test_actual_cross_module_raw_lock_and_run_contract_admit_only_owner_thread(self):
        from research.factor_alchemy import run_contract as raw
        self.assertIsNot(raw,lock_module(),'fixture must exercise separate legacy/governance imports')
        for mode in ('FuxiWriterLock','RunContract'):
            with self.subTest(mode=mode),isolated_workspace() as root,actual_savers(root) as cases:
                lock=root/'data/fuxi_runtime.lock'
                owner=(raw.FuxiWriterLock(lock,'extension_fixture') if mode=='FuxiWriterLock' else
                    raw.RunContract(parameters={'fixture_only':True},run_id='extension_fixture',
                        experiment_id='extension_fixture',runs_dir=root/'data/runs',lock_path=lock))
                with owner:
                    self.assertIsNotNone(lock_module().current_thread_writer(lock))
                    for name,(target,save,_) in cases.items():
                        with self.subTest(saver=name):
                            save();self.assertTrue(target.is_file());load_json(target)
                            before=file_sha(target);errors=[]
                            def other_thread():
                                try:save()
                                except Exception as exc:errors.append(exc)
                            thread=threading.Thread(target=other_thread);thread.start();thread.join(5)
                            self.assertFalse(thread.is_alive());self.assertEqual(len(errors),1)
                            self.assertIsInstance(errors[0],ContractError);self.assertEqual(file_sha(target),before)
                self.assertFalse(lock.exists());self.assertIsNone(lock_module().current_thread_writer(lock))
                for target,save,_ in cases.values():
                    before=file_sha(target)
                    with self.assertRaises(ContractError):save()
                    self.assertEqual(file_sha(target),before)
                if mode=='RunContract':self.assertEqual(load_json(root/'data/runs/extension_fixture/manifest.json')['status'],'completed')


if __name__=='__main__':unittest.main()
