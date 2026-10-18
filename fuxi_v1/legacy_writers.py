"""Declared legacy writer boundaries; archival scripts are not production entries."""
import os
from pathlib import Path
import sys
from .contracts import require
from .governance import find_workspace, inherited_identity, start_cli_guard, lock_module

ENTRIES = {
    'scripts/stage0_daily_update_v3.py': ('data_refresh', 'module', False),
    'scripts/stage0_daily_collect.py': ('data_refresh', 'main', False),
    'scripts/stage0_collect_moneyflow.py': ('data_refresh', 'main', False),
    'scripts/stage0_macro_collect.py': ('data_refresh', 'main', False),
    'scripts/stage0_fund_share_collect.py': ('data_refresh', 'main', False),
    'scripts/collect_index_daily.py': ('data_refresh', 'main', False),
    'scripts/collect_sw_daily.py': ('data_refresh', 'main', False),
    'scripts/collect_adj_factor_daily.py': ('data_refresh', 'main', False),
    'scripts/daily_factor_hypothesis.py': ('evaluate', 'main', False),
    'scripts/stage1_to_factor_proposals.py': ('factor_import', 'main', False),
    'scripts/stage2_pass_to_stage3.py': ('factor_import', 'function', False),
    'scripts/factor_injector.py': ('factor_import', 'main', False),
    'scripts/build_dashboard.py': ('index_refresh', 'main', True),
    'scripts/version_bump.py': ('version_record', 'function', False),
    'research/factor_alchemy/regime_lasso.py': ('evaluate', 'module', False),
    'research/factor_alchemy/trigger_d_plus.py': ('feedback', 'main', False),
    'research/factor_alchemy/ralph_loop.py': ('candidate_search', 'main', False),
    'research/factor_alchemy/run_v4_pipeline.py': ('candidate_search', 'function', False),
}
STATE_BOUNDARIES = {
    'research/factor_alchemy/experience_memory.py': (('_save', 'self.path'),),
    'research/factor_alchemy/factor_model_cooptim.py': (('save_state', 'self._state_path'),('_save_results','self._results_path')),
    'research/factor_alchemy/trajectory_logger.py': (('_save', 'self.path'),),
    'research/factor_alchemy/library_orthogonality.py': (('save_state','self._state_path'),),
    'research/factor_alchemy/llm_evolution.py': (('save_state','STATE_PATH'),),
    'research/factor_alchemy/subtree_fingerprinter.py': (('_save_state','self.state_path'),),
}

def legacy_cli_guard(entry_file, *, read_only=False):
    workspace = find_workspace(entry_file)
    require(workspace is not None, 'legacy writer requires a declared workspace')
    name = Path(entry_file).resolve().relative_to(workspace).as_posix()
    require(name in ENTRIES, 'legacy writer is not registered: '+name)
    action, _, auto = ENTRIES[name]
    parent = inherited_identity(workspace)
    if parent is not None and parent['trigger'] == 'automatic' and not read_only:
        require(auto, 'legacy producer has no bounded automatic adapter: '+name)
    require('--full' not in sys.argv, 'destructive legacy full replacement is not admitted by fusion governance')
    return start_cli_guard(entry_file, action, read_only=read_only)

def assert_state_writer(target):
    """Require the owner before saving production state; isolated tests stay isolated.

    Acquiring a lock only at save time would allow stale read/modify/write state.
    The enclosing entry must own the lock from before its read.
    """
    path = Path(target).resolve()
    workspace = find_workspace(path)
    if workspace is None or not path.is_relative_to(workspace/'data'):
        return
    active = inherited_identity(workspace)
    if active is not None:
        require(active['action'] in {'unified_run','feedback','candidate_search','evaluate','factor_import'},
                'state write exceeds the current action')
        return
    module = lock_module()
    lock = workspace/'data/fuxi_runtime.lock'
    require(module.current_thread_writer(lock) is not None,
            'production state write requires the actual current-thread lock owner')

def inventory(workspace):
    import hashlib, ast
    root = Path(workspace).resolve()
    rows = []
    for name, (action, boundary, auto) in ENTRIES.items():
        path = root/name
        raw = path.read_bytes()
        rows.append({'path':name,'action':action,'boundary':boundary,
                     'automatic_adapter':auto,'sha256':hashlib.sha256(raw).hexdigest(),
                     'guard_present':b'legacy_cli_guard' in raw})
    for name, methods in STATE_BOUNDARIES.items():
        raw=(root/name).read_bytes()
        tree=ast.parse(raw)
        for method,target in methods:
            definitions=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name==method]
            guarded=any(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and
                            node.func.id=='assert_state_writer' and node.args and ast.unparse(node.args[0])==target
                            for node in ast.walk(definition)) for definition in definitions)
            rows.append({'path':name,'action':'feedback','boundary':method,'target':target,
                         'sha256':hashlib.sha256(raw).hexdigest(),'guard_present':guarded})
    rows.append({'path':'research/factor_alchemy/run_v4_pipeline.py','action':'candidate_search',
                 'boundary':'RunContract','guard_present':b'with RunContract(' in (root/'research/factor_alchemy/run_v4_pipeline.py').read_bytes()})
    return {'schema_version':'fuxi.writer_inventory.1','entries':rows,
            'coverage':'declared active production entries and shared state savers',
            'historical_once_scripts_and_arbitrary_file_edits_are_not_intercepted':True,
            'external_RAG':'D:/RAG/rag_ingest_stage1.py is not modified or automatically invoked',
            'all_declared_boundaries_present':all(r['guard_present'] for r in rows)}
