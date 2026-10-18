from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

from fuxi_v1.contracts import ContractError
from fuxi_v1.governance import (
    LEASE_ENV, SAFE_AUTOMATIC_ACTIONS, child_environment, governed_writer,
    guard_cli, inherited_identity, lock_module, native_cli_action, validate_native_budget)
from fuxi_v1.runtime import controlled_writer


WORKSPACE = Path(__file__).resolve().parents[2]
MANUAL = {'single_writer_confirmed': True, 'current_mode': 'controlled_manual',
          'autonomous_writes_enabled': False}
BOUNDED = {'single_writer_confirmed': True, 'current_mode': 'bounded_research',
           'autonomous_writes_enabled': True, 'bounded_research': {
    'allowed_actions': sorted(SAFE_AUTOMATIC_ACTIONS),
    'budgets': {'max_rounds': 3, 'max_candidates': 6, 'max_worker_seconds': 120,
               'max_total_seconds': 1800, 'max_memory_mb': 2048, 'max_external_requests': 0,
               'max_llm_calls': 0, 'max_output_tokens': 0},
    'authorization': {'authorized': True, 'source': 'isolated fixture authorization',
                      'recorded_at': '2026-10-09T20:00:00+08:00'}}}


@contextmanager
def workspace(policy=MANUAL):
    base = WORKSPACE/'reports/fuxi_fusion_20261009/test_tmp'
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='governance_', dir=base) as tmp:
        root = Path(tmp)
        (root/'codex_automations').mkdir()
        (root/'codex_automations/fuxi_runtime_policy.json').write_text(json.dumps(policy), encoding='utf8')
        yield root


def child(code, env=None):
    selected = dict(os.environ if env is None else env)
    selected.setdefault('PYTHONIOENCODING', 'utf-8')
    return subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', code],
                          cwd=WORKSPACE, env=selected, text=True, encoding='utf8',
                          capture_output=True, timeout=20)


def native_input(root, *, execution=None):
    payload = {'schema_version': 'fuxi.campaign_input.1', 'snapshot': {'exposure': 'research_exposed'}}
    if execution is not None:
        payload['full_market_execution'] = execution
    path = root/'frozen_input.json'; path.write_text(json.dumps(payload), encoding='utf8')
    return str(path)


class GovernanceTests(unittest.TestCase):
    def test_manual_authorized_and_failure_releases_owner(self):
        with workspace() as root:
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                with governed_writer(root, 'version_record') as identity:
                    self.assertTrue(Path(identity['lock_path']).is_file())
                    self.assertEqual(identity['owner_pid'], os.getpid())
                    raise RuntimeError('fixture failure')
            self.assertFalse((root/'data/fuxi_runtime.lock').exists())

    def test_release_idempotent_when_lock_file_already_absent(self):
        # 4 次长跑事故实证: 退出路径锁文件已不存在时 release 应幂等返回,
        # 不再把已完成的运行误报为失败 (P-20261015-001)。
        with workspace() as root:
            lock = root/'data/fuxi_runtime.lock'
            with governed_writer(root, 'version_record') as identity:
                self.assertTrue(lock.is_file())
                self.assertEqual(identity['owner_pid'], os.getpid())
                lock.unlink()  # 模拟外部删除
            self.assertFalse(lock.exists())

    def test_release_still_refuses_tampered_lock_file(self):
        # 幂等只适用于文件已不存在; 文件存在但 token 被篡改时严格校验保持,
        # 防止把他人锁静默放行 (P-20261015-001)。
        with workspace() as root:
            lock = root/'data/fuxi_runtime.lock'
            with self.assertRaisesRegex(lock_module().ActiveFuxiRunError, 'identity differs'):
                with governed_writer(root, 'version_record'):
                    self.assertTrue(lock.is_file())
                    data = json.loads(lock.read_text(encoding='utf-8'))
                    data['token'] = 'tampered_token'
                    lock.write_text(json.dumps(data), encoding='utf-8')

    def test_manual_policy_refuses_automatic(self):
        with workspace() as root:
            with self.assertRaises(ContractError):
                with governed_writer(root, 'unified_run', trigger='automatic'):
                    self.fail('autonomous manual writer admitted')
            self.assertFalse((root/'data').exists())

    def test_unconfirmed_writer_refused(self):
        with workspace(dict(MANUAL, single_writer_confirmed=False)) as root:
            with self.assertRaises(ContractError):
                with governed_writer(root, 'report'):
                    self.fail('unconfirmed writer admitted')

    def test_bounded_actions_and_budgets_required(self):
        variants = []
        for key in ('max_rounds', 'max_candidates', 'max_memory_mb'):
            policy = deepcopy(BOUNDED)
            policy['bounded_research']['budgets'][key] = 0
            variants.append(policy)
        unsafe = deepcopy(BOUNDED)
        unsafe['bounded_research']['allowed_actions'].append('activate_paper')
        variants.append(unsafe)
        missing = deepcopy(BOUNDED)
        del missing['bounded_research']['authorization']
        variants.append(missing)
        for policy in variants:
            with self.subTest(policy=policy), workspace(policy) as root:
                with self.assertRaises(ContractError):
                    with governed_writer(root, 'unified_run', trigger='automatic'):
                        self.fail('invalid bounded policy admitted')

    def test_bounded_manual_operator_compatibility_and_automatic_prohibition(self):
        with workspace(BOUNDED) as root:
            with governed_writer(root, 'operator_evidence'):
                pass
            for action in ('operator_evidence', 'version_record', 'activate_paper', 'arbitrary_command'):
                with self.subTest(action=action), self.assertRaises(ContractError):
                    with governed_writer(root, action, trigger='automatic'):
                        self.fail('unsafe automatic action admitted')

    def test_automatic_phase_needs_controller(self):
        with workspace(BOUNDED) as root:
            with self.assertRaises(ContractError):
                with governed_writer(root, 'candidate_search', trigger='automatic'):
                    self.fail('standalone automatic phase admitted')

    def test_scoped_automatic_retains_top_level_manual_contract(self):
        scoped = deepcopy(MANUAL)
        scoped['bounded_research'] = dict(deepcopy(BOUNDED['bounded_research']), enabled=True)
        with workspace(scoped) as root:
            with governed_writer(root, 'unified_run', trigger='automatic') as identity:
                self.assertEqual(identity['budgets']['max_candidates'], 6)
            with governed_writer(root, 'operator_evidence') as identity:
                self.assertEqual(identity['budgets'], {})
            actual = json.loads((root/'codex_automations/fuxi_runtime_policy.json').read_text())
            self.assertEqual(actual['current_mode'], 'controlled_manual')
            self.assertFalse(actual['autonomous_writes_enabled'])

    def test_original_frozen_manual_runtime_can_borrow_new_owner_with_random_alias(self):
        scoped = deepcopy(MANUAL)
        scoped['bounded_research'] = dict(deepcopy(BOUNDED['bounded_research']), enabled=True)
        original = WORKSPACE/'reports/fuxi_fusion_20261009/before/code/fuxi_v1/runtime.py'
        self.assertTrue(original.is_file())
        with workspace(scoped) as root:
            contract = root/'research/factor_alchemy/run_contract.py'
            contract.parent.mkdir(parents=True)
            contract.write_bytes((WORKSPACE/'research/factor_alchemy/run_contract.py').read_bytes())
            with governed_writer(root, 'unified_run', run_id='frozen_owner') as owner:
                env = child_environment(owner, 'manual_native')
                result = child(
                    "import importlib.util\n"
                    f"spec=importlib.util.spec_from_file_location('fuxi_v1.original_runtime',{str(original)!r})\n"
                    "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n"
                    f"with module.controlled_writer({str(root/'output')!r}): print('original-compatible')\n", env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('original-compatible', result.stdout)
                self.assertEqual(json.loads(Path(owner['lock_path']).read_text())['run_id'], 'frozen_owner')

    def test_same_thread_nested_and_legacy_contract_keep_parent_lock(self):
        with workspace(BOUNDED) as root:
            with governed_writer(root, 'unified_run', trigger='automatic', run_id='fixture_owner') as owner:
                with governed_writer(root, 'candidate_search', trigger='automatic') as nested:
                    self.assertTrue(nested['inherited'])
                    with lock_module().RunContract(parameters={}, run_id='fixture_legacy',
                            runs_dir=root/'legacy_runs', lock_path=root/'data/fuxi_runtime.lock') as legacy:
                        self.assertTrue(legacy.lock.inherited)
                self.assertEqual(json.loads((root/'data/fuxi_runtime.lock').read_text())['token'], owner['token'])
                with self.assertRaises(ContractError):
                    with governed_writer(root, 'report', trigger='manual'):
                        self.fail('trigger downgrade admitted')
            self.assertFalse((root/'data/fuxi_runtime.lock').exists())

    def test_another_thread_cannot_borrow_same_process_lock(self):
        with workspace() as root:
            errors = []
            with governed_writer(root, 'unified_run'):
                def compete():
                    try:
                        with governed_writer(root, 'report'):
                            errors.append('admitted')
                    except lock_module().ActiveFuxiRunError:
                        errors.append('blocked')
                thread = Thread(target=compete)
                thread.start(); thread.join(10)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, ['blocked'])

    def test_raw_owner_registry_rejects_other_thread_state_save_and_release(self):
        with workspace() as root:
            path = root/'data/fuxi_runtime.lock'
            lock = lock_module().FuxiWriterLock(path, 'raw_thread_owner')
            lock.acquire()
            try:
                self.assertEqual(lock_module().current_thread_writer(path)['token'], lock.token)
                observations = []
                def other_thread_save():
                    observations.append(lock_module().current_thread_writer(path))
                    try:
                        lock.release()
                    except lock_module().ActiveFuxiRunError:
                        observations.append('release_refused')
                thread = Thread(target=other_thread_save)
                thread.start(); thread.join(10)
                self.assertFalse(thread.is_alive())
                self.assertEqual(observations, [None, 'release_refused'])
                self.assertTrue(path.exists())
            finally:
                lock.release()
            self.assertIsNone(lock_module().current_thread_writer(path))

    def test_default_raw_lock_retains_acquire_release_compatibility(self):
        with workspace() as root:
            path = root/'data/fuxi_runtime.lock'
            with lock_module().FuxiWriterLock(path) as lock:
                self.assertTrue(lock.run_id)
                self.assertEqual(lock_module().current_thread_writer(path)['token'], lock.token)
            self.assertFalse(path.exists())

    def test_nested_failure_does_not_release_owner(self):
        with workspace() as root:
            with governed_writer(root, 'unified_run') as owner:
                with self.assertRaises(ValueError):
                    with governed_writer(root, 'report'):
                        raise ValueError('nested failure')
                self.assertEqual(json.loads(Path(owner['lock_path']).read_text())['token'], owner['token'])

    def test_real_child_inherits_ancestor_and_does_not_release_owner(self):
        with workspace(BOUNDED) as root:
            with governed_writer(root, 'unified_run', trigger='automatic') as owner:
                env = child_environment(owner, 'candidate_search')
                code = (
                    "from fuxi_v1.governance import governed_writer\n"
                    f"with governed_writer({str(root)!r},'candidate_search',trigger='automatic') as lease:\n"
                    " print(lease['inherited'],lease['run_id'])\n")
                result = child(code, env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('True '+owner['run_id'], result.stdout)
                self.assertTrue(Path(owner['lock_path']).exists())

    def test_child_forged_token_run_id_or_create_time_refused(self):
        with workspace(BOUNDED) as root:
            with governed_writer(root, 'unified_run', trigger='automatic') as owner:
                for key, value in (('token', 'forged'), ('run_id', 'forged'), ('process_create_time', 1.)):
                    with self.subTest(key=key):
                        env = child_environment(owner, 'candidate_search')
                        lease = json.loads(env[LEASE_ENV]); lease[key] = value
                        env[LEASE_ENV] = json.dumps(lease)
                        result = child(f"from fuxi_v1.governance import inherited_identity; inherited_identity({str(root)!r})", env)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('identity differs', result.stderr)
                        self.assertTrue(Path(owner['lock_path']).exists())

    def test_same_process_env_alone_cannot_borrow(self):
        with workspace() as root:
            lock = lock_module().FuxiWriterLock(root/'data/fuxi_runtime.lock', 'raw_owner')
            lock.acquire()
            lease = {'lock_path': str(lock.path.resolve()), **lock.meta}
            try:
                with patch.dict(os.environ, {LEASE_ENV: json.dumps(lease)}):
                    with self.assertRaises(lock_module().InvalidFuxiLeaseError):
                        lock_module().FuxiWriterLock(lock.path, 'pretend_nested').acquire()
            finally:
                lock.release()

    def test_live_competitor_and_sibling_inheritance_refused(self):
        with workspace() as root:
            env = dict(os.environ); env.pop(LEASE_ENV, None)
            code = (
                "import time\nfrom fuxi_v1.governance import governed_writer\n"
                f"with governed_writer({str(root)!r},'unified_run') as owner:\n"
                " print('READY',flush=True)\n time.sleep(15)\n")
            holder = subprocess.Popen([sys.executable, '-B', '-c', code], cwd=WORKSPACE,
                                      env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(holder.stdout.readline().strip(), 'READY')
                with self.assertRaises(lock_module().ActiveFuxiRunError):
                    with governed_writer(root, 'report'):
                        self.fail('live competitor admitted')
                owner = json.loads((root/'data/fuxi_runtime.lock').read_text())
                lease = {'lock_path': str((root/'data/fuxi_runtime.lock').resolve()), **owner}
                env[LEASE_ENV] = json.dumps(lease)
                result = child(f"from fuxi_v1.governance import lock_module; lock_module().FuxiWriterLock({str(root/'data/fuxi_runtime.lock')!r}).acquire()", env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('not an ancestor', result.stderr)
            finally:
                holder.terminate(); holder.communicate(timeout=10)

    def test_dead_owner_recovered_and_original_lock_archived(self):
        with workspace() as root:
            result = child(
                "import os\nfrom fuxi_v1.governance import lock_module\n"
                f"lock_module().FuxiWriterLock({str(root/'data/fuxi_runtime.lock')!r},'dead').acquire()\n"
                "os._exit(0)\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            with governed_writer(root, 'report'):
                self.assertEqual(len(list((root/'data').glob('fuxi_runtime.lock.stale.*'))), 1)
            self.assertFalse((root/'data/fuxi_runtime.lock').exists())

    def test_malformed_existing_lock_is_not_deleted(self):
        with workspace() as root:
            (root/'data').mkdir(); path = root/'data/fuxi_runtime.lock'
            path.write_bytes(b'{')
            with self.assertRaises(lock_module().ActiveFuxiRunError):
                with governed_writer(root, 'report'):
                    self.fail('unreadable lock reclaimed')
            self.assertEqual(path.read_bytes(), b'{')

    def test_legacy_manifest_start_failure_releases_only_owned_lock(self):
        with workspace() as root:
            manifest = root/'legacy_runs/broken/manifest.json'
            manifest.parent.mkdir(parents=True); manifest.write_bytes(b'{')
            with self.assertRaises(json.JSONDecodeError):
                with lock_module().RunContract(parameters={}, run_id='broken', resume=True,
                        runs_dir=root/'legacy_runs', lock_path=root/'data/fuxi_runtime.lock'):
                    self.fail('broken manifest admitted')
            self.assertFalse((root/'data/fuxi_runtime.lock').exists())

    def test_child_environment_requires_active_verified_scope(self):
        with workspace() as root:
            with governed_writer(root, 'unified_run') as owner:
                self.assertNotIn(LEASE_ENV, os.environ)
            with self.assertRaises(ContractError):
                child_environment(owner)

    def test_cli_automatic_budget_and_operator_action_cannot_be_disguised(self):
        with workspace(BOUNDED) as root:
            with governed_writer(root, 'unified_run', trigger='automatic') as owner:
                env = child_environment(owner, 'candidate_search')
                input_path = native_input(root)
                for args, accepted in ((['campaign', '--rounds', '2', '--candidates', '2',
                                        '--models', 'ridge', '--input', input_path], True),
                                       (['campaign'], False), (['campaign', '--llm'], False),
                                       (['operate', 'activate-paper'], False)):
                    with self.subTest(args=args):
                        result = child(
                            "import sys\nfrom fuxi_v1.runtime import controlled_writer\n"
                            f"sys.argv=['fuxi_v1']+{args!r}\n"
                            f"with controlled_writer({str(root/'output')!r}) as identity: print(identity['action'])\n", env)
                        self.assertEqual(result.returncode == 0, accepted, result.stderr)
                self.assertEqual(native_cli_action(['work-packages', 'prepare-jq']), 'operator_evidence')

    def test_automatic_unbudgeted_native_commands_are_refused(self):
        with workspace(BOUNDED) as root:
            with governed_writer(root, 'unified_run', trigger='automatic') as owner:
                for command in ('blueprint', 'history-study', 'snapshot-history', 'work-packages'):
                    action = native_cli_action([command, 'run'])
                    env = child_environment(owner, action)
                    result = child(
                        "import sys\nfrom fuxi_v1.runtime import controlled_writer\n"
                        f"sys.argv=['fuxi_v1',{command!r},'run']\n"
                        f"with controlled_writer({str(root/'output')!r}): pass\n", env)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('no bounded adapter', result.stderr)

    def test_study_protocol_models_and_resource_input_cannot_expand_automatic_budget(self):
        with workspace(BOUNDED) as root:
            with governed_writer(root, 'unified_run', trigger='automatic') as owner:
                with governed_writer(root, 'candidate_search', trigger='automatic') as identity:
                    input_path = native_input(root)
                    base = ['campaign', '--rounds=2', '--candidates=2', '--models=ridge', '--input', input_path]
                    validate_native_budget(identity, base)
                    for extra in (['--rounds', '30'], ['--models', 'ridge,ridge'], ['--data-receipt-id', 'receipt']):
                        with self.subTest(extra=extra), self.assertRaises(ContractError):
                            validate_native_budget(identity, base+extra)
                    with self.assertRaises(ContractError):
                        validate_native_budget(identity, [x.replace('--models=ridge','--models=unregistered') for x in base])
                    protocol = {'schema_version': 'fuxi.controlled_study.1', 'arm_schedule': ['fixed', 'refine'],
                                'max_trials': 4, 'checkpoint_every': 1, 'memory_sources': []}
                    path = root/'study_protocol.json'; path.write_text(json.dumps(protocol))
                    validate_native_budget(identity, base+['--study-protocol', str(path)])
                    variants = [dict(protocol, max_trials=999), dict(protocol, memory_sources=['unbounded.sqlite']),
                                dict(protocol, arm_schedule=['llm', 'llm_patch'])]
                    for changed in variants:
                        path.write_text(json.dumps(changed))
                        with self.subTest(protocol=changed), self.assertRaises(ContractError):
                            validate_native_budget(identity, base+['--study-protocol', str(path)])
                    full = {'schema_version': 'fuxi.full_market_execution.1', 'top_n': 80,
                            'worker_timeout': 120, 'worker_memory_mb': 2048, 'model_timeout': 60,
                            'portfolio_timeout': 120, 'budget_units': 780}
                    for key, value in (('worker_timeout', 600), ('model_timeout', 600),
                                       ('portfolio_timeout', 600), ('worker_memory_mb', 8192), ('budget_units', 99999)):
                        native_input(root, execution=dict(full, **{key: value}))
                        with self.subTest(resource=key), self.assertRaises(ContractError):
                            validate_native_budget(identity, base)
                    native_input(root, execution=full)
                    validate_native_budget(identity, base)

    def test_scoped_automatic_native_child_matches_real_action_and_budgets(self):
        scoped = deepcopy(MANUAL)
        scoped['bounded_research'] = dict(deepcopy(BOUNDED['bounded_research']), enabled=True)
        with workspace(scoped) as root:
            with governed_writer(root, 'unified_run', trigger='automatic') as owner:
                input_path = native_input(root)
                env = child_environment(owner, 'candidate_search')
                args = ['campaign', '--rounds', '1', '--candidates', '1', '--models', 'ridge', '--input', input_path]
                result = child(
                    "import sys\nfrom fuxi_v1.runtime import controlled_writer\n"
                    f"sys.argv=['fuxi_v1']+{args!r}\n"
                    f"with controlled_writer({str(root/'output')!r}) as lease: print(lease['trigger'],lease['inherited'])\n", env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('automatic True', result.stdout)
                self.assertTrue(Path(owner['lock_path']).exists())

    def test_guard_cli_native_manual_compatibility_and_readonly(self):
        with workspace() as root:
            entry = root/'scripts/entry.py'; entry.parent.mkdir(); entry.write_text('# fixture')
            with guard_cli(entry, 'factor_import') as identity:
                self.assertEqual(identity['trigger'], 'manual')
                self.assertTrue(Path(identity['lock_path']).exists())
            with guard_cli(entry, 'factor_import', read_only=True) as identity:
                self.assertIsNone(identity)
            with patch.object(sys, 'argv', ['fuxi_v1', 'operate', 'admit-research']):
                with controlled_writer(root/'authority.sqlite') as identity:
                    self.assertEqual(identity['action'], 'operator_evidence')

    def test_policy_change_invalidates_active_and_child_lease(self):
        with workspace(BOUNDED) as root:
            with governed_writer(root, 'unified_run', trigger='automatic') as owner:
                env = child_environment(owner, 'report')
                policy = deepcopy(BOUNDED); policy['bounded_research']['budgets']['max_rounds'] = 2
                (root/'codex_automations/fuxi_runtime_policy.json').write_text(json.dumps(policy))
                with self.assertRaises(ContractError):
                    child_environment(owner, 'report')
                result = child(f"from fuxi_v1.governance import inherited_identity; inherited_identity({str(root)!r})", env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('policy changed', result.stderr)


if __name__ == '__main__':
    unittest.main()
