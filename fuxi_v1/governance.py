"""Shared writer authority for manual operations and bounded research orchestration.

An inherited lease is checked against the real ancestor process and lock bytes.
It grants no investment acceptance, platform authority or permission to edit defaults.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import atexit
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import threading
import uuid
from .contracts import digest, load_json, require, timestamp


SAFE_AUTOMATIC_ACTIONS = frozenset({
    'unified_run', 'data_refresh', 'source_scan', 'factor_import',
    'candidate_search', 'evaluate', 'feedback', 'index_refresh', 'report'})
BUDGET_KEYS = frozenset({
    'max_rounds', 'max_candidates', 'max_worker_seconds', 'max_total_seconds',
    'max_memory_mb', 'max_external_requests', 'max_llm_calls', 'max_output_tokens'})
LEASE_ENV = 'FUXI_WRITER_LEASE'
_ACTIVE = ContextVar('fuxi_governed_writer', default=None)
_LOCK_MODULE = None
_STARTED_GUARDS = []


def lock_module():
    """Load the existing lock implementation once; no legacy factor state is imported."""
    global _LOCK_MODULE
    if _LOCK_MODULE is None:
        workspace = find_workspace(__file__)
        require(workspace is not None, 'shared writer implementation requires a declared workspace')
        path = workspace / 'research/factor_alchemy/run_contract.py'
        spec = importlib.util.spec_from_file_location('fuxi_shared_writer_contract', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOCK_MODULE = module
    return _LOCK_MODULE


def find_workspace(path):
    resolved = Path(path).resolve()
    return next((p for p in (resolved, *resolved.parents)
                 if (p/'codex_automations/fuxi_runtime_policy.json').is_file()), None)


def _authorization(policy, action, trigger):
    require(trigger in {'manual', 'automatic'}, 'unknown writer trigger')
    require(isinstance(action, str) and re.fullmatch(r'[a-z][a-z0-9_]{0,63}', action),
            'explicit symbolic writer action required')
    require(policy.get('single_writer_confirmed') is True, 'single writer authorization missing')
    mode = policy.get('current_mode')
    if mode == 'controlled_manual':
        require(policy.get('autonomous_writes_enabled') is False, 'controlled_manual top-level policy mismatch')
        if trigger == 'manual':
            return {}
        require(isinstance(policy.get('bounded_research'), dict)
                and policy['bounded_research'].get('enabled') is True,
                'controlled_manual refuses autonomous writes without a scoped bounded authorization')
    else:
        require(mode == 'bounded_research' and policy.get('autonomous_writes_enabled') is True,
                'workspace writer authorization mismatch')
    bounded = policy.get('bounded_research')
    require(isinstance(bounded, dict), 'bounded research policy required')
    allowed, budgets, authorization = (bounded.get(k) for k in ('allowed_actions', 'budgets', 'authorization'))
    require(isinstance(allowed, list) and allowed and len(allowed) == len(set(allowed))
            and set(allowed) <= SAFE_AUTOMATIC_ACTIONS, 'unsafe or ambiguous automatic action whitelist')
    require(isinstance(budgets, dict) and set(budgets) == BUDGET_KEYS
            and all(type(v) is int and v >= 0 for v in budgets.values()),
            'explicit bounded integer budgets required')
    require(all(budgets[k] > 0 for k in BUDGET_KEYS - {
        'max_external_requests', 'max_llm_calls', 'max_output_tokens'}), 'positive compute budgets required')
    require(isinstance(authorization, dict) and authorization.get('authorized') is True
            and isinstance(authorization.get('source'), str) and authorization['source'].strip(),
            'explicit bounded research authorization required')
    timestamp(authorization.get('recorded_at'))
    if trigger == 'automatic':
        require(action in allowed and action in SAFE_AUTOMATIC_ACTIONS,
                'automatic writer action is not authorized')
    return dict(budgets)


def _lease(identity):
    return {key: identity[key] for key in (
        'schema_version', 'workspace', 'lock_path', 'run_id', 'pid', 'process_create_time',
        'token', 'action', 'trigger', 'policy_hash', 'budgets')}


def inherited_identity(workspace):
    """Read an active local scope or verify the complete environment lease, never trust it alone."""
    workspace = Path(workspace).resolve()
    active = _ACTIVE.get()
    if active is not None:
        require(Path(active['workspace']) == workspace, 'nested writer names another workspace')
        lock_module().validate_writer_lease(workspace/'data/fuxi_runtime.lock', _lease(active), allow_self=True)
        policy = load_json(workspace/'codex_automations/fuxi_runtime_policy.json')
        require(active['policy_hash'] == digest(policy), 'writer policy changed inside active scope')
        require(active['budgets'] == _authorization(policy, active['action'], active['trigger']),
                'active writer budgets differ from authorization')
        return active
    inherited = lock_module().environment_writer_lease()
    if inherited is None:
        return None
    require(inherited.get('schema_version') == 'fuxi.writer_lease.1'
            and Path(inherited.get('workspace', '')).resolve() == workspace,
            'inherited writer workspace/schema mismatch')
    owner = lock_module().validate_writer_lease(workspace/'data/fuxi_runtime.lock', inherited)
    policy = load_json(workspace/'codex_automations/fuxi_runtime_policy.json')
    require(inherited.get('policy_hash') == digest(policy), 'writer policy changed during inherited execution')
    budgets = _authorization(policy, inherited.get('action'), inherited.get('trigger'))
    require(inherited.get('budgets') == budgets, 'inherited writer budgets differ from authorization')
    scope = owner.get('governance_scope')
    require(isinstance(scope, dict) and scope.get('policy_hash') == inherited['policy_hash']
            and scope.get('trigger') == inherited['trigger'], 'lock governance scope differs from lease')
    require(scope.get('action') == 'unified_run' or scope.get('action') == inherited['action'],
            'inherited writer action exceeds owner scope')
    return dict(inherited, owner_pid=owner['pid'], owner_create_time=owner['process_create_time'], inherited=True)


@contextmanager
def governed_writer(workspace, action, *, trigger='manual', run_id=None):
    workspace = Path(workspace).resolve()
    policy = load_json(workspace/'codex_automations/fuxi_runtime_policy.json')
    budgets = _authorization(policy, action, trigger)
    parent = inherited_identity(workspace)
    if parent is not None:
        require(parent['trigger'] == trigger, 'nested writer cannot change its trigger')
        require(parent['policy_hash'] == digest(policy), 'writer policy changed inside active scope')
        require(parent['action'] == 'unified_run' or parent['action'] == action,
                'nested writer action exceeds parent scope')
        require(run_id is None or run_id == parent['run_id'], 'nested writer cannot replace owner run_id')
    elif trigger == 'automatic':
        require(action == 'unified_run', 'automatic phase requires a verified controller lease')
    selected_id = parent['run_id'] if parent else run_id or 'fuxi_governed_' + uuid.uuid4().hex
    require(isinstance(selected_id, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', selected_id),
            'invalid writer run_id')
    module = lock_module()
    with module.FuxiWriterLock(workspace/'data/fuxi_runtime.lock', selected_id,
              inheritance=_lease(parent) if parent else None, allow_self=_ACTIVE.get() is not None,
              lease_scope={'action': action, 'trigger': trigger, 'policy_hash': digest(policy)}) as lock:
        identity = {'schema_version': 'fuxi.writer_lease.1', 'workspace': str(workspace),
                    'lock_path': str((workspace/'data/fuxi_runtime.lock').resolve()),
                    **{k: lock.meta[k] for k in ('run_id', 'pid', 'process_create_time', 'token')},
                    'owner_pid': lock.meta['pid'], 'owner_create_time': lock.meta['process_create_time'],
                    'action': action, 'trigger': trigger, 'policy_hash': digest(policy),
                    'budgets': budgets, 'inherited': lock.inherited}
        token = _ACTIVE.set(identity)
        thread = threading.current_thread()
        attr = module.LOCAL_LEASE_ATTRIBUTE
        previous = getattr(thread, attr, {})
        setattr(thread, attr, {**previous, identity['lock_path']: _lease(identity)})
        try:
            yield identity
        finally:
            setattr(thread, attr, previous)
            _ACTIVE.reset(token)


def child_environment(identity, action=None, trigger=None, base=None):
    """Return a subprocess env after validating this thread's active owner; never mutate os.environ."""
    active = _ACTIVE.get()
    require(active is not None and _lease(active) == _lease(identity), 'active local writer scope required for child env')
    verified = inherited_identity(identity['workspace'])
    selected_action, selected_trigger = action or identity['action'], trigger or identity['trigger']
    policy = load_json(Path(identity['workspace'])/'codex_automations/fuxi_runtime_policy.json')
    _authorization(policy, selected_action, selected_trigger)
    require(selected_trigger == identity['trigger']
            and (identity['action'] == 'unified_run' or selected_action == identity['action']),
            'child action/trigger exceeds parent scope')
    lease = _lease(verified)
    lease.update(action=selected_action, trigger=selected_trigger)
    env = dict(os.environ if base is None else base)
    env[LEASE_ENV] = json.dumps(lease, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return env


def native_cli_action(argv=None):
    """Bind unmodified native CLI writer calls to their actual parsed command vocabulary."""
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else ''
    if command in {'operate', 'certify-data', 'forward-archive'}:
        return 'operator_evidence'
    if command == 'work-packages':
        return 'operator_evidence' if any(x in args for x in ('prepare-jq', 'return-jq')) else 'evaluate'
    return {'campaign': 'candidate_search', 'blueprint': 'evaluate', 'history-study': 'evaluate',
            'history-preflight': 'evaluate', 'snapshot-history': 'data_refresh',
            'snapshot-local': 'data_refresh', 'pit-cohort': 'factor_import',
            'history-export': 'report'}.get(command, 'manual_native')


def validate_native_budget(identity, argv=None):
    if identity['trigger'] != 'automatic':
        return
    args = list(sys.argv[1:] if argv is None else argv)
    require(args and args[0] == 'campaign',
            'automatic native command has no bounded adapter; only controller-managed campaign is admitted')
    budgets = identity['budgets']
    # The native provider does not settle an independent policy token/call budget.
    # Keep automatic campaigns local until such an adapter exists, even if a policy cap is nonzero.
    require('--llm' not in args, 'native LLM has no bounded automatic adapter')
    def option(name, default=None):
        matches = [(i, x) for i, x in enumerate(args) if x == name or x.startswith(name+'=')]
        require(len(matches) <= 1, 'duplicate native budget option')
        if not matches:
            return default
        index, value = matches[0]
        if value.startswith(name+'='):
            return value.split('=', 1)[1]
        require(index+1 < len(args) and not args[index+1].startswith('--'), 'native option is missing: '+name)
        return args[index+1]
    rounds, candidates = int(option('--rounds', 5)), int(option('--candidates', 3))
    require(0 < rounds <= budgets['max_rounds'] and 0 < candidates
            and rounds*candidates <= budgets['max_candidates'], 'native campaign exceeds automatic candidate budget')
    require(option('--data-receipt-id') is None, 'automatic campaign cannot consume formal qualification authority')
    models = option('--models', 'ridge,lightgbm,xgboost').split(',')
    models = [x for x in models if x]
    require(len(models) == len(set(models)) and set(models) <= {'ridge', 'lightgbm', 'xgboost'}
            and len(models) <= rounds, 'automatic model action set is unknown, duplicated or exceeds rounds')
    workspace = Path(identity['workspace'])
    def local_file(name):
        value = option(name)
        require(value is not None, 'automatic campaign requires explicit '+name)
        path = Path(value).resolve()
        require(path.is_relative_to(workspace) and path.is_file(), 'automatic input must be a frozen workspace file: '+name)
        return path
    payload = load_json(local_file('--input'))
    require(payload.get('schema_version') == 'fuxi.campaign_input.1'
            and payload.get('snapshot', {}).get('exposure') == 'research_exposed',
            'automatic campaign requires declared research-exposed input')
    require('full_market_options' not in payload, 'unregistered resource option alias')
    full = payload.get('full_market_execution')
    if full is None:
        native = {'worker_timeout': 300, 'worker_memory_mb': 2048, 'model_timeout': 60,
                  'portfolio_timeout': 120, 'budget_units': rounds*candidates*300+420}
    else:
        require(isinstance(full, dict) and set(full) == {'schema_version', 'top_n', 'worker_timeout',
                'worker_memory_mb', 'model_timeout', 'portfolio_timeout', 'budget_units'}
                and full['schema_version'] == 'fuxi.full_market_execution.1'
                and all(type(v) is int and v > 0 for k, v in full.items() if k != 'schema_version'),
                'automatic full-market execution contract differs')
        native = full
        required_reservation = rounds*candidates*full['worker_timeout']
        required_reservation += (len(models)+2 if models else 0)*full['model_timeout'] + full['portfolio_timeout']
        require(full['budget_units'] >= required_reservation, 'automatic full-market reservation is insufficient')
    require(max(native[k] for k in ('worker_timeout', 'model_timeout', 'portfolio_timeout'))
            <= budgets['max_worker_seconds'] and native['worker_memory_mb'] <= budgets['max_memory_mb']
            and native['budget_units'] <= budgets['max_total_seconds'],
            'native resource reservation exceeds automatic policy')
    if option('--study-protocol') is not None:
        protocol = load_json(local_file('--study-protocol'))
        require(set(protocol) == {'schema_version', 'arm_schedule', 'max_trials', 'checkpoint_every', 'memory_sources'}
                and protocol['schema_version'] == 'fuxi.controlled_study.1', 'automatic study protocol differs')
        require(type(protocol['max_trials']) is int and rounds*candidates <= protocol['max_trials']
                <= budgets['max_candidates'] and type(protocol['checkpoint_every']) is int
                and 1 <= protocol['checkpoint_every'] <= rounds,
                'automatic study protocol exceeds trial budget')
        require(isinstance(protocol['arm_schedule'], list) and len(protocol['arm_schedule']) == rounds
                and set(protocol['arm_schedule']) <= {'fixed', 'random', 'refine', 'combine', 'random_ast', 'refine_local'}
                and protocol['memory_sources'] == [], 'automatic study has unbounded memory or external generation')


@contextmanager
def guard_cli(entry_file, action, *, read_only=False):
    """Guard a statically declared legacy entry point before its first write."""
    workspace = find_workspace(entry_file)
    if workspace is None or read_only:
        yield None
        return
    parent = inherited_identity(workspace)
    trigger = parent['trigger'] if parent else 'manual'
    with governed_writer(workspace, action, trigger=trigger) as identity:
        yield identity


def start_cli_guard(entry_file, action, *, read_only=False):
    """Hold a guard to process exit for historical module-level scripts without a main function."""
    context = guard_cli(entry_file, action, read_only=read_only)
    identity = context.__enter__()
    _STARTED_GUARDS.append(context)
    def close():
        if context in _STARTED_GUARDS:
            _STARTED_GUARDS.remove(context)
            context.__exit__(None, None, None)
    atexit.register(close)
    return identity
