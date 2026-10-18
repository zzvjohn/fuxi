"""Bounded, isolated evidence for the 2026-09-23 Fuxi drought campaign."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import copy
from datetime import datetime
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
ALCHEMY = ROOT / 'research' / 'factor_alchemy'
OUT = ROOT / 'reports' / 'drought_20260923'
sys.path.insert(0, str(ALCHEMY))

CODE = [
    'forge/primitives.py', 'forge/expression.py', 'forge/__init__.py',
    'ralph_loop.py', 's5_joint_filter.py', 'weekly_panel.py',
    'multi_stage_validator.py', 'run_contract.py', 'llm_generator.py',
]
STATE = [
    'experience_memory.json', 'mab_scheduler_state.json', 'trajectory_log.json',
    'breed_fail_streak.json', 'jq_codegen_queue.json', 'version_registry.json',
    'stage1_optimization_proposals.json',
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin_harness(step):
    source = Path(__file__).resolve()
    destination = OUT / 'execution_scripts' / (step + '_' + datetime.now().strftime('%H%M%S_%f') + '.py')
    destination.parent.mkdir(exist_ok=True)
    shutil.copy2(source, destination)
    return {'path': str(destination), 'sha256': digest(destination)}


def assert_reusable_row(row):
    recorded = row.get('code_hashes') or row.get('costs', {}).get('15', {}).get('fingerprint', {}).get('code_hashes')
    current = {name: digest(ALCHEMY / name) for name in CODE}
    if recorded != current:
        raise RuntimeError('cannot reuse row with missing or different core code fingerprint')


def write_json(path, payload):
    from run_contract import _atomic_json
    def scalar(value):
        if hasattr(value, 'item'):
            return value.item()
        if hasattr(value, 'isoformat'):
            return value.isoformat()
        raise TypeError(type(value).__name__)
    _atomic_json(Path(path), json.loads(json.dumps(payload, default=scalar)))


@contextmanager
def switches(**values):
    before = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def repair_cs_broadcast(formula):
    def is_row_series(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            return node.func.attr in ('mean', 'std', 'min', 'max', 'sum') and any(
                k.arg == 'axis' and isinstance(k.value, ast.Constant) and k.value.value == 1
                for k in node.keywords)
        if isinstance(node, ast.BinOp):
            return ((is_row_series(node.left) and (is_row_series(node.right) or isinstance(node.right, ast.Constant)))
                    or (is_row_series(node.right) and isinstance(node.left, ast.Constant)))
        return False

    class Fix(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, (ast.Sub, ast.Div)) and is_row_series(node.right) and not is_row_series(node.left):
                return ast.copy_location(ast.Call(
                    func=ast.Attribute(value=node.left, attr='sub' if isinstance(node.op, ast.Sub) else 'div', ctx=ast.Load()),
                    args=[node.right], keywords=[ast.keyword(arg='axis', value=ast.Constant(0))]), node)
            return node

    original = ast.parse(formula, mode='eval')
    fixed = Fix().visit(ast.parse(formula, mode='eval'))
    if ast.dump(original) == ast.dump(fixed):
        return None
    return ast.unparse(ast.fix_missing_locations(fixed))


def freeze():
    path = OUT / 'protocol.json'
    if path.exists():
        raise RuntimeError('protocol is immutable; use a new campaign for revisions')
    baseline = OUT / 'baseline' / 'data'
    memory = json.loads((baseline / 'experience_memory.json').read_text(encoding='utf-8'))['attempts']
    trajectories = json.loads((baseline / 'trajectory_log.json').read_text(encoding='utf-8'))['trajectories']
    by_name = {item['factor_name']: item for item in memory if item.get('formula')}
    groups = {'compute_failure': [], 's2_rejected': [], 's5_precheck': [], 's5_measured': []}
    for traj in sorted(trajectories, key=lambda t: t.get('created_at', ''), reverse=True):
        if traj.get('created_at', '') < '2026-09-17':
            continue
        events = traj.get('_eval_log') or []
        if not events:
            continue
        event = events[-1]
        name = event.get('factor_name')
        record = by_name.get(name)
        if not record:
            continue
        formula = record['formula']
        try:
            ast.parse(formula, mode='eval')
        except SyntaxError:
            continue
        reason = event.get('rejection_reason', '')
        group = None
        if event.get('stage') == 'S1' and '无周频 IC 数据' in reason:
            group = 'compute_failure'
        elif event.get('stage') == 'S2':
            group = 's2_rejected'
        elif event.get('stage') == 'S5':
            group = 's5_precheck' if event.get('excess_25') is None else 's5_measured'
        if group:
            groups[group].append({
                'memory_id': record['id'], 'factor_name': name, 'formula': formula,
                'formula_sha256': hashlib.sha256(formula.encode()).hexdigest(),
                'natural_freq': traj.get('natural_freq') or 'daily',
                'frequency_source': 'trajectory' if traj.get('natural_freq') else 'historical_default_daily',
                'stratum': group, 'trajectory_id': traj['trajectory_id'],
                'historical_event': event, 'paradigm': record.get('paradigm'),
                'broadcast_repair_formula': repair_cs_broadcast(formula),
            })
    selected, seen = [], set()
    quotas = {'compute_failure': 2, 's2_rejected': 2, 's5_precheck': 1, 's5_measured': 3}
    for group, quota in quotas.items():
        count = 0
        for item in groups[group]:
            if item['formula_sha256'] in seen:
                continue
            selected.append(item)
            seen.add(item['formula_sha256'])
            count += 1
            if count == quota:
                break
        if count != quota:
            raise RuntimeError(f'insufficient complete candidates for {group}: {count}/{quota}')
    protocol = {
        'created_at': datetime.now().astimezone().isoformat(),
        'selection_rule': 'latest unique full formulas by failure stratum; no return-based selection',
        'history_is_research_exposed': True,
        'development_cutoff': '2024-12-31', 'historical_diagnostic_years': [2025, 2026],
        'date_ceil': 'auto', 'universe': 'all_available_A_shares', 'benchmark': '000985',
        'cost_bps': [0, 15, 30], 'official_cost_bps': 15, 'top_n': 80, 'min_coverage': 0.9,
        'formal_threshold': {'vol_scaled': 0.7, 'excess_each_year': 0},
        'candidate_budget': 20, 'historical_candidate_count': 8,
        'llm_budget': {'calls': 3, 'candidates_per_call': 2, 'max_output_tokens_per_call': 4096},
        'llm_families': ['动量反转', '波动率适应', '价量关系'],
        'llm_fields': ['open', 'high', 'low', 'close', 'volume', 'amount'],
        'llm_limit': 'OHLCV families only; event/PIT fundamental families deferred until availability audit',
        'r4': {'combiners': ['equal_rank', 'ridge_alpha_100'], 'fit_end': '2024-12-31',
               'folds': ['2025H1', '2025H2', '2026YTD'], 'promotion': 'no automatic eligibility'},
        'r5': {'status': 'protocol_frozen_no_candidate_promoted',
               'prospective_start': 'after_candidate_freeze',
               'unknown_sample': 'future unobserved observations only; no 2025/2026 OOS claim',
               'changes_restart_validation': True, 's5_pass_is_not_live_trading_approval': True},
        'candidate_groups_available': {key: len(value) for key, value in groups.items()},
        'candidates': selected,
        'code_hashes': {name: digest(ALCHEMY / name) for name in CODE},
        'data_hashes': {name: digest(ROOT / 'data' / 'raw' / name)
                        for name in ['daily_prices.csv', 'index_000985_daily.csv']},
    }
    write_json(path, protocol)
    print(json.dumps({'protocol': str(path), 'candidates': [
        {key: c[key] for key in ['memory_id', 'stratum', 'factor_name', 'natural_freq']}
        for c in selected]}, ensure_ascii=False))


def engine():
    from s5_joint_filter import S5JointFilter, _s5_min_coverage, _s5_adj_price, _s5_tradable_gate, _s5_tie_rule
    inventory = OUT / 'runtime_inventory.json'
    if inventory.exists():
        for name, record in json.loads(inventory.read_text(encoding='utf-8'))['inputs'].items():
            if digest(ROOT / name) != record['sha256']:
                raise RuntimeError('frozen execution input changed: ' + name)
    obj = S5JointFilter(cache_dir=OUT / 'cache')
    assert obj.top_n == 80 and obj.sample_stocks is None
    assert _s5_min_coverage() == .9 and _s5_adj_price() == 'on' and _s5_tradable_gate()
    assert _s5_tie_rule() == 'on' and obj._effective_calmar_threshold() == .7
    assert obj.index_path.name == 'index_000985_daily.csv'
    obj._load_data()
    if (OUT / 'replay.json').exists():
        expected = json.loads((OUT / 'replay.json').read_text(encoding='utf-8'))['data_window']
        actual = {'first': str(obj._price_df.trade_date.min()), 'last': str(obj._price_df.trade_date.max()),
                  'stocks': len(obj._universe), 'rows': len(obj._price_df)}
        if actual != expected:
            raise RuntimeError('market window/universe differs from frozen replay')
    return obj


def runtime_inventory():
    import platform
    import numpy as np
    import pandas as pd
    import scipy
    target = OUT / 'runtime_inventory.json'
    if target.exists():
        raise RuntimeError('runtime inventory is immutable')
    names = ['data/raw/daily_prices.csv', 'data/raw/index_000985_daily.csv', 'data/raw/stk_limit.csv']
    write_json(target, {
        'recorded_at': datetime.now().astimezone().isoformat(),
        'timing': 'supplement during historical replay, before LLM market evaluation; primary price/index hashes also in initial protocol',
        'inputs': {name: {'sha256': digest(ROOT / name), 'bytes': (ROOT / name).stat().st_size,
                         'mtime_ns': (ROOT / name).stat().st_mtime_ns} for name in names},
        'runtime': {'executable': sys.executable, 'python': platform.python_version(),
                    'pandas': pd.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__},
        'limitations': 'auxiliary fields loaded but not used in OHLCV formulas; limit table captured after replay began, not a pre-run snapshot',
    })
    print('Input inventory and runtime versions recorded; late supplementary timing disclosed.')


def cost_diagnostics(obj, factor_df, prefix):
    import pandas as pd
    result = {}
    fingerprint = {'protocol_sha256': digest(OUT / 'protocol.json'),
                   'code_hashes': {name: digest(ALCHEMY / name) for name in CODE},
                   'signal_sha256': digest(OUT / 'signals' / (prefix + '.parquet'))}
    for cost in (0, 15, 30):
        checkpoint = OUT / 'cost_rows' / (prefix + f'_bps{cost}.json')
        if checkpoint.exists():
            result[str(cost)] = json.loads(checkpoint.read_text(encoding='utf-8'))
            if result[str(cost)].get('fingerprint') != fingerprint:
                raise RuntimeError('cost checkpoint fingerprint mismatch: ' + prefix)
            continue
        started = time.time()
        with switches(FUXI_S5_COST_BPS=cost, FUXI_S5_RETURN_ALIGN='on'):
            returns, m25, m26 = obj._backtest_and_metrics(factor_df)
            diag = copy.deepcopy(obj._last_backtest_diag)
            passed, reason = obj._verdict(m25, m26)
            m25_legacy = m26_legacy = None
            if cost == 15:
                with switches(FUXI_S5_RETURN_ALIGN='off'):
                    m25_legacy = obj.compute_metrics(returns, 2025)
                    m26_legacy = obj.compute_metrics(returns, 2026)
        return_file = OUT / 'returns' / (prefix + f'_bps{cost}.parquet')
        return_file.parent.mkdir(exist_ok=True)
        returns.rename('return').to_frame().to_parquet(return_file)
        result[str(cost)] = {
            'fingerprint': fingerprint,
            'metrics_2025': m25, 'metrics_2026': m26, 'diag': diag,
            'economic_pass': bool(passed and (diag or {}).get('coverage_ok', False)),
            'reason': reason, 'returns_file': str(return_file),
            'old_unmatched_metrics_2025': m25_legacy,
            'old_unmatched_metrics_2026': m26_legacy,
            'elapsed_seconds': round(time.time() - started, 2),
        }
        write_json(checkpoint, result[str(cost)])
        print(f'  {prefix} cost={cost} ex25={m25["excess_return"]:.4f} ex26={m26["excess_return"]:.4f} '
              f'windows={m25["n_periods"]}/{m26["n_periods"]} pass={result[str(cost)]["economic_pass"]}', flush=True)
    return result


def evaluate_candidate(obj, candidate, formula, freq, label):
    import numpy as np
    import pandas as pd
    from weekly_panel import eval_weekly_formula, align_weekly_to_daily, weekly_ic
    out = {'candidate_id': candidate['memory_id'], 'label': label,
           'natural_freq': candidate['natural_freq'], 'evaluated_freq': freq,
           'formula': formula, 'formula_sha256': hashlib.sha256(formula.encode()).hexdigest(),
           'formal_eligibility': False, 'purpose': 'isolated_failure_attribution'}
    for mode in ('legacy', 'enforce'):
        with switches(FUXI_S5_SANITY_AST=mode):
            out['precheck_' + mode] = obj._check_formula_sanity(formula, candidate['factor_name'])
    started = time.time()
    panel = obj._get_weekly_panel() if freq == 'weekly' else None
    if freq == 'weekly':
        ok, wide, long, reason = eval_weekly_formula(formula, panel)
        if not ok:
            return {**out, 'eval_status': 'error', 'reason': reason}
        # The old broadcast bug can create date-labelled columns; reject rather
        # than silently intersecting them away and calling it a valid signal.
        if not wide.index.equals(panel['weeks']) or set(wide.columns) != set(panel['universe']):
            return {**out, 'eval_status': 'error', 'reason': 'signal_axis_mismatch'}
        out['ic'] = weekly_ic(panel, wide)
        out['ic_by_fold'] = {name: weekly_ic(panel, wide.loc[start:end])
                             for name, start, end in [
                                 ('development', '2021-01-01', '2024-12-31'),
                                 ('2025H1', '2025-01-01', '2025-06-30'),
                                 ('2025H2', '2025-07-01', '2025-12-31'),
                                 ('2026YTD', '2026-01-01', '2026-12-31')]}
        out['ic_label_basis'] = 'weekly_panel raw weekly close forward return, distinct from adjusted S5 portfolio returns'
        out['finite_values'] = int(np.isfinite(wide.to_numpy()).sum())
        factor_df = align_weekly_to_daily(long, panel, obj._price_df[['ts_code', 'trade_date']])
    else:
        factor_df = obj.evaluate_factor_formula(formula, candidate['factor_name'])
        if factor_df is None or not len(factor_df):
            return {**out, 'eval_status': 'error', 'reason': 'daily_formula_empty'}
        out['finite_values'] = int(len(factor_df))
    out['value_guard'] = obj._value_guards(factor_df)
    signal_path = OUT / 'signals' / (candidate['memory_id'] + '_' + label + '.parquet')
    signal_path.parent.mkdir(exist_ok=True)
    factor_df.to_parquet(signal_path, index=False)
    out['signal_file'] = str(signal_path)
    out['eval_status'] = 'valid'
    out['costs'] = cost_diagnostics(obj, factor_df, candidate['memory_id'] + '_' + label)
    out['elapsed_seconds'] = round(time.time() - started, 2)
    return out


def replay():
    from run_contract import RunContract
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    for filename, expected in protocol['data_hashes'].items():
        if digest(ROOT / 'data' / 'raw' / filename) != expected:
            raise RuntimeError('frozen data changed: ' + filename)
    with RunContract(parameters={'step': 'drought_replay', 'protocol_sha256': digest(OUT / 'protocol.json'),
                                 's5_return_align': 'on', 'costs': [0, 15, 30]},
                     run_id='drought_20260923_replay_' + datetime.now().strftime('%H%M%S'),
                     runs_dir=OUT / 'runs'):
        obj = engine()
        output = {'protocol_sha256': digest(OUT / 'protocol.json'), 'rows': [],
                  'data_window': {'first': str(obj._price_df.trade_date.min()),
                                  'last': str(obj._price_df.trade_date.max()),
                                  'stocks': len(obj._universe), 'rows': len(obj._price_df)},
                  'code_hashes': {name: digest(ALCHEMY / name) for name in CODE},
                  'thresholds_unchanged': True}
        for candidate in protocol['candidates']:
            formula = candidate['formula']
            arms = [(formula, 'daily', 'original_daily')]
            if candidate['natural_freq'] == 'weekly':
                arms.append((formula, 'weekly', 'original_weekly'))
            if candidate['broadcast_repair_formula']:
                arms.append((candidate['broadcast_repair_formula'], candidate['natural_freq'], 'broadcast_repaired'))
            for formula, freq, label in arms:
                print(f'START {candidate["memory_id"]} {candidate["stratum"]} {label}', flush=True)
                checkpoint = OUT / 'replay_rows' / f'{candidate["memory_id"]}_{label}.json'
                if checkpoint.exists():
                    row = json.loads(checkpoint.read_text(encoding='utf-8'))
                    if row.get('code_hashes') != output['code_hashes']:
                        raise RuntimeError('cannot resume checkpoint made by different code')
                else:
                    try:
                        row = evaluate_candidate(obj, candidate, formula, freq, label)
                    except Exception as error:
                        row = {'candidate_id': candidate['memory_id'], 'label': label,
                               'eval_status': 'error', 'reason': f'{type(error).__name__}: {error}'}
                    row['code_hashes'] = output['code_hashes']
                    write_json(checkpoint, row)
                output['rows'].append(row)
                write_json(OUT / 'replay.json', output)
        assert len(output['rows']) > 0
        print(json.dumps({'rows': len(output['rows']), 'valid': sum(r['eval_status'] == 'valid' for r in output['rows'])}))


def snapshot(label):
    target = OUT / label
    if target.exists():
        raise RuntimeError('snapshot already exists: ' + str(target))
    paths = [ALCHEMY / name for name in CODE]
    paths += [ROOT / 'data' / name for name in STATE]
    paths += [ROOT / 'scripts' / 'version_bump.py', ROOT / 'CHANGELOG.md']
    if label == 'caliber_baseline':
        paths = [ROOT / 'scripts' / 's5_fullpool_rerun.py']
    elif label == 'release':
        paths += [ROOT / 'scripts' / name for name in [
            's5_fullpool_rerun.py', 'fuxi_drought_campaign_20260923.py',
            'fuxi_drought_readout_20260923.py', 'verify_drought_release_20260923.py',
            'register_drought_20260923.py', 'fuxi_drought_signal_attribution_20260923.py',
            'fuxi_drought_release_20260923.py']]
        paths += [ALCHEMY / 'tests' / name for name in [
            'test_full_forge_contract.py', 'test_drought_s5_contract.py',
            'test_drought_experiment_controls.py']]
        paths += [ROOT / name for name in [
            'AGENTS.md', '.workbuddy/memory/MEMORY.md', '.workbuddy/memory/_fuxi_landing_details.md',
            '.workbuddy/memory/2026-09-23.md', 'reports/fuxi_drought_contracts_20260923.md',
            'reports/fuxi_drought_campaign_20260923.md']]
    result = {'created_at': datetime.now().astimezone().isoformat(), 'files': {}}
    for src in paths:
        if not src.exists():
            continue
        rel = src.relative_to(ROOT)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        assert digest(src) == digest(dest)
        result['files'][str(rel)] = {'sha256': digest(src), 'bytes': src.stat().st_size}
    write_json(target / 'manifest.json', result)
    print(json.dumps({'snapshot': str(target), 'files': len(result['files'])}))


def experiment_formula_issues(formula):
    fields = {'open', 'high', 'low', 'close', 'volume', 'amount', 'np'}
    methods = {'rolling', 'mean', 'std', 'sum', 'min', 'max', 'median', 'var',
               'corr', 'cov', 'rank', 'shift', 'diff', 'pct_change', 'clip',
               'replace', 'abs', 'pow', 'add', 'sub', 'mul', 'div'}
    numpy = {'log', 'log1p', 'abs', 'sqrt', 'sign', 'maximum', 'minimum', 'nan', 'inf'}
    try:
        tree = ast.parse(formula, mode='eval')
    except SyntaxError:
        return ['invalid_syntax']
    issues = []
    if len(list(ast.walk(tree))) > 500:
        issues.append('expression_budget')
    for node in ast.walk(tree):
        if isinstance(node, (ast.Lambda, ast.Subscript, ast.ListComp, ast.DictComp,
                             ast.GeneratorExp, ast.NamedExpr, ast.IfExp)):
            issues.append('unsupported_ast')
        if isinstance(node, ast.Name) and node.id not in fields:
            issues.append('unknown_name:' + node.id)
        if isinstance(node, ast.Attribute):
            is_np = isinstance(node.value, ast.Name) and node.value.id == 'np'
            if node.attr not in (numpy if is_np else methods):
                issues.append('unknown_attribute:' + node.attr)
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            issues.append('bare_call')
            continue
        method = node.func.attr
        kws = {kw.arg: kw.value for kw in node.keywords}
        if method in ('rolling', 'shift', 'diff', 'pct_change'):
            arg = node.args[0] if node.args else kws.get('window' if method == 'rolling' else 'periods')
            if arg is None and method != 'rolling':
                arg = ast.Constant(1)
            if not isinstance(arg, ast.Constant) or not isinstance(arg.value, int) or not 1 <= arg.value <= 52:
                issues.append('invalid_window:' + method)
        if method == 'pct_change' and not (isinstance(kws.get('fill_method'), ast.Constant)
                                         and kws['fill_method'].value is None):
            issues.append('pct_change_missing_fill_method_none')
        if 'center' in kws and not (isinstance(kws['center'], ast.Constant) and kws['center'].value is False):
            issues.append('centered_window')
        if method == 'rank' and not (isinstance(node.func.value, ast.Call)
                                    and isinstance(node.func.value.func, ast.Attribute)
                                    and node.func.value.func.attr == 'rolling'):
            if not (isinstance(kws.get('axis'), ast.Constant) and kws['axis'].value == 1):
                issues.append('rank_axis_not_cross_sectional')
    return sorted(set(issues))


def prepare_llm():
    from llm_generator import _schema_system_prompt
    path = OUT / 'llm_requests.json'
    if path.exists():
        raise RuntimeError('LLM requests are immutable')
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    requests = []
    for index, family in enumerate(protocol['llm_families'], 1):
        user = (
            f'研究族：{family}。独立提出恰好 2 个简单、机制不同、可证伪的 A 股周频选股因子。'
            '仅探索研究假设，不宣称预测收益。没有本批回测反馈，不追逐任何特定年度表现。'
            '输出裸 JSON 数组，不要 markdown；每项必须包含 name, expression, hypothesis, '
            'refutation_conditions, direction (long 或 short)。公式值为高值买入；'
            'direction=short 将统一乘以 -1，禁止在看到回测后改变方向。'
            '输入是日期行、股票列的周频 DataFrame，真实字段仅 open/high/low/close/volume/amount。'
            '收盘后得到本周数据，最早下周交易；窗口单位为周，整数 1 至 52。'
            '不允许外部数据、自定义函数、条件切换、lambda、pd、索引操作、填零、ffill、未来值。'
            '方法限定 rolling/mean/std/sum/min/max/median/var/corr/cov/rank/shift/diff/'
            'pct_change/clip/replace/abs/pow/add/sub/mul/div；numpy 仅 log/log1p/abs/sqrt/sign/'
            'maximum/minimum/nan/inf。不要使用 ewm/expanding。'
            '所有 pct_change 显式 fill_method=None；rolling 必须完整窗口；'
            '截面排名必须 .rank(axis=1,pct=True)，截面 Series 广播必须 .sub/.div(axis=0)。'
            '分母用 replace(0,np.nan) 或 1e-8 保护。不要裸价格量级、VWAP 代理或复杂算子堆叠。'
            '机制说明需说明为什么该量价行为可能有补偿，以及什么观测会推翻它。')
        requests.append({'id': f'llm_family_{index}', 'family': family,
                         'system_prompt': _schema_system_prompt(weekly_mode=True),
                         'user_prompt': user, 'temperature': 0.2, 'max_tokens': 4096})
    write_json(path, {'frozen_at': datetime.now().astimezone().isoformat(), 'requests': requests})
    print('Frozen 3 requests / 6 candidates; no return feedback included.')


def generate_llm():
    import llm_client
    from llm_generator import LLMGenerator, validate_llm_schema
    # One HTTP attempt per frozen request: retries would consume the bounded budget.
    llm_client.MAX_RETRIES = 1
    client = llm_client.get_llm_client()
    generator = LLMGenerator(output_dir=OUT / 'llm')
    requests = json.loads((OUT / 'llm_requests.json').read_text(encoding='utf-8'))['requests']
    rows = []
    for request in requests:
        checkpoint = OUT / 'llm' / (request['id'] + '.json')
        if checkpoint.exists():
            row = json.loads(checkpoint.read_text(encoding='utf-8'))
        else:
            row = {'request_id': request['id'], 'family': request['family'],
                   'request_sha256': digest(OUT / 'llm_requests.json'),
                   'model': client.model, 'created_at': datetime.now().astimezone().isoformat(),
                   'status': 'started', 'candidates': []}
            write_json(checkpoint, row)
            try:
                raw = client.chat_with_system(**{key: request[key] for key in
                    ['system_prompt', 'user_prompt', 'temperature', 'max_tokens']})
                row['raw_response'] = raw
                row['status'] = 'response_received'
                write_json(checkpoint, row)
                # Preserve the raw response even if parsing/validation fails.
                items = json.loads(raw.strip())
                if not isinstance(items, list) or len(items) != 2:
                    raise ValueError('expected exactly two JSON candidates')
                for number, item in enumerate(items, 1):
                    candidate = generator._normalize_candidate(item)
                    if not candidate:
                        raise ValueError('empty candidate')
                    candidate['memory_id'] = request['id'] + '_' + str(number)
                    candidate['natural_freq'] = 'weekly'
                    candidate['stratum'] = 'fresh_llm'
                    candidate['raw_formula'] = candidate['formula']
                    candidate['schema_observations'] = validate_llm_schema(candidate['formula'], weekly_mode=True)
                    candidate['safety_issues'] = experiment_formula_issues(candidate['formula'])
                    if not candidate.get('hypothesis') or not candidate.get('refutation_conditions'):
                        candidate['safety_issues'].append('missing_mechanism_or_refutation')
                    if item.get('direction') not in ('long', 'short'):
                        candidate['safety_issues'].append('direction_not_declared')
                    if candidate['direction'] == 'short':
                        candidate['formula'] = '-(' + candidate['formula'] + ')'
                    row['candidates'].append(candidate)
                row['status'] = 'parsed'
            except Exception as error:
                row['status'] = 'failed'
                row['error'] = type(error).__name__ + ': ' + str(error)
            write_json(checkpoint, row)
        rows.append(row)
        print(json.dumps({'request_id': row['request_id'], 'status': row['status'],
                          'candidates': len(row['candidates'])}), flush=True)
    write_json(OUT / 'llm_generation.json', {'rows': rows})


def evaluate_llm():
    from run_contract import RunContract
    generation = json.loads((OUT / 'llm_generation.json').read_text(encoding='utf-8'))
    candidates = [c for row in generation['rows'] for c in row['candidates']]
    assert len(candidates) <= 6
    with RunContract(parameters={'step': 'bounded_llm_evaluation',
                                 'generation_sha256': digest(OUT / 'llm_generation.json'),
                                 'harness': pin_harness('llm_evaluation')},
                     run_id='drought_llm_' + datetime.now().strftime('%H%M%S'), runs_dir=OUT / 'runs'):
        obj = engine()
        rows = []
        for candidate in candidates:
            issues = experiment_formula_issues(candidate['formula'])
            issues += [item for item in candidate['safety_issues'] if item != 'invalid_window:pct_change']
            checkpoint = OUT / 'llm_rows' / (candidate['memory_id'] + '.json')
            if checkpoint.exists():
                row = json.loads(checkpoint.read_text(encoding='utf-8'))
                assert_reusable_row(row)
            elif issues:
                row = {'candidate_id': candidate['memory_id'], 'eval_status': 'precheck_rejected',
                       'issues': sorted(set(issues)), 'formal_eligibility': False}
            else:
                print('START ' + candidate['memory_id'], flush=True)
                try:
                    row = evaluate_candidate(obj, candidate, candidate['formula'], 'weekly', 'fresh_llm')
                except Exception as error:
                    row = {'candidate_id': candidate['memory_id'], 'eval_status': 'error',
                           'reason': type(error).__name__ + ': ' + str(error)}
            row['code_hashes'] = {name: digest(ALCHEMY / name) for name in CODE}
            write_json(checkpoint, row)
            rows.append(row)
            write_json(OUT / 'llm_evaluation.json', {'rows': rows})
        print(json.dumps({'llm_rows': len(rows), 'valid': sum(r['eval_status'] == 'valid' for r in rows)}))


def prepare_r4():
    path = OUT / 'r4_protocol.json'
    if path.exists():
        raise RuntimeError('R4 protocol is immutable')
    write_json(path, {
        'frozen_at': datetime.now().astimezone().isoformat(),
        'sources': ['llm_family_1_1', 'llm_family_2_1', 'llm_family_3_1'],
        'selection_rule': 'first response in each predeclared family, before LLM market evaluation',
        'baseline': 'first source on the identical three-source complete-case universe; research only',
        'arms': ['baseline_common', 'equal_rank', 'ridge_alpha_100',
                 'equal_without_1', 'equal_without_2', 'equal_without_3'],
        'fit_end': '2024-12-31', 'alpha': 100.0,
        'objective': 'mean squared cross-sectional forward-return rank error + alpha * squared weights',
        'label': 'next complete market week adjusted close/pre_close return; not next-open execution',
        'missing': 'common complete-case universe for every arm; no zero filling',
        'cost_bps': [0, 15, 30], 'candidate_budget_accounting': '8 historical + 6 LLM + 6 R4 = 20',
        'folds': ['2025H1', '2025H2', '2026YTD'],
        'abort_if': 'any source lacks a finite signal or training data is insufficient',
        'promotion': 'none; equal-rank ablations do not validate ridge attribution',
    })
    print('R4 selection, baseline, masks, weights protocol and six arms frozen.')


def fit_frozen_ridge(values, target, next_event, fit_end, alpha):
    import numpy as np
    import pandas as pd
    common = np.isfinite(values).all(axis=2)
    train_weeks = next_event.notna() & (next_event <= pd.Timestamp(fit_end))
    train = common & np.isfinite(target.to_numpy()) & train_weeks.to_numpy()[:, None]
    x, y = values[train], target.to_numpy()[train]
    if len(y) < 1000:
        raise RuntimeError('R4 training data insufficient')
    coef = np.linalg.solve(x.T @ x / len(y) + alpha * np.eye(values.shape[2]), x.T @ y / len(y))
    latest = next_event.loc[train.any(axis=1)].max()
    assert latest <= pd.Timestamp(fit_end)
    return coef, len(y), str(latest)


def evaluate_r4():
    import numpy as np
    import pandas as pd
    from run_contract import RunContract
    from weekly_panel import align_weekly_to_daily
    from s5_joint_filter import _board_limit_vector
    config = json.loads((OUT / 'r4_protocol.json').read_text(encoding='utf-8'))
    sources = config['sources']
    paths = [OUT / 'signals' / (name + '_fresh_llm.parquet') for name in sources]
    if not all(path.exists() for path in paths):
        write_json(OUT / 'r4_evaluation.json', {'status': 'not_evaluated', 'reason': 'missing_source_signal'})
        return
    with RunContract(parameters={'step': 'r4_bounded_combos', 'protocol_sha256': digest(OUT / 'r4_protocol.json'),
                                 'harness': pin_harness('r4_evaluation')},
                     run_id='drought_r4_' + datetime.now().strftime('%H%M%S'), runs_dir=OUT / 'runs'):
        obj = engine()
        panel = obj._get_weekly_panel()
        weeks = panel['weeks']
        events = pd.DatetimeIndex(panel['week_last_day'].reindex(weeks).values)
        columns = panel['universe']
        frames = []
        for path in paths:
            signal = pd.read_parquet(path)
            signal = signal[signal.trade_date.isin(events)]
            wide = signal.pivot(index='trade_date', columns='ts_code', values='factor_value').reindex(index=events, columns=columns)
            wide.index = weeks
            frames.append(wide.rank(axis=1, pct=True) - .5)
        values = np.stack([frame.to_numpy(dtype=float) for frame in frames], axis=2)
        common = np.isfinite(values).all(axis=2)
        prices = obj._price_df[['ts_code', 'trade_date', 'close_p', 'pre_close_p']].copy()
        daily = prices.close_p / prices.pre_close_p.replace(0, np.nan) - 1
        daily = daily.where(daily.abs() <= _board_limit_vector(prices.ts_code) + .005)
        prices['week'] = prices.trade_date.dt.to_period('W-SUN').dt.end_time.dt.normalize()
        growth = (daily + 1).groupby([prices.week, prices.ts_code]).prod(min_count=1)
        counts = daily.groupby([prices.week, prices.ts_code]).count()
        expected = prices.groupby('week').trade_date.nunique()
        growth = growth.where(counts == growth.index.get_level_values('week').map(expected))
        adjusted_fwd = (growth.unstack('ts_code') - 1).reindex(index=weeks, columns=columns).shift(-1)
        target = adjusted_fwd.rank(axis=1, pct=True) - .5
        from weekly_panel import weekly_ic
        raw_fwd = panel['fwd_ret'].reindex_like(adjusted_fwd)
        paired_labels = np.isfinite(raw_fwd) & np.isfinite(adjusted_fwd) & common
        label_rows = []
        for source, frame in zip(sources, frames):
            for label, start, end in [('development', '2021-01-01', '2024-12-31'),
                                      ('2025H1', '2025-01-01', '2025-06-30'),
                                      ('2025H2', '2025-07-01', '2025-12-31'),
                                      ('2026YTD', '2026-01-01', '2026-12-31')]:
                signal = frame.where(paired_labels).loc[start:end]
                raw_ic = weekly_ic({'fwd_ret': raw_fwd.where(paired_labels)}, signal)
                adj_ic = weekly_ic({'fwd_ret': adjusted_fwd.where(paired_labels)}, signal)
                assert raw_ic['weekly_n'] == adj_ic['weekly_n'] > 0
                label_rows.append({'source': source, 'fold': label, 'raw_label_ic': raw_ic,
                                   'adjusted_label_ic': adj_ic,
                                   'paired_stock_weeks': int(signal.notna().to_numpy().sum())})
        write_json(OUT / 'ic_label_alignment.json', {
            'rows': label_rows, 'purpose': 'paired label-basis diagnosis, not an execution or OOS pass',
            'adjusted_label': 'next full market week close/pre_close compounded, same board guard as S5',
            'all_copies_use_identical_valid_pairs': True})
        next_event = pd.Series(events, index=weeks).shift(-1)
        coef, n_train, label_last = fit_frozen_ridge(values, target, next_event, config['fit_end'], config['alpha'])
        model = {'source_sha256': {p.name: digest(p) for p in paths}, 'sources': sources,
                 'weights': coef.tolist(), 'training_observations': n_train,
                 'training_label_last_available': label_last,
                 'fit_end': config['fit_end'], 'alpha': config['alpha'],
                 'baseline_kind': 'research_not_production', 'formal_eligibility': False}
        write_json(OUT / 'r4_model.json', model)
        arrays = {'baseline_common': values[:, :, 0], 'equal_rank': values.mean(axis=2),
                  'ridge_alpha_100': np.einsum('ijk,k->ij', values, coef)}
        for index in range(3):
            arrays['equal_without_' + str(index + 1)] = np.delete(values, index, axis=2).mean(axis=2)
        rows = []
        for name in config['arms']:
            checkpoint = OUT / 'r4_rows' / (name + '.json')
            if checkpoint.exists():
                row = json.loads(checkpoint.read_text(encoding='utf-8'))
                assert_reusable_row(row)
                if row.get('model') != model:
                    raise RuntimeError('R4 checkpoint model mismatch')
            else:
                wide = pd.DataFrame(np.where(common, arrays[name], np.nan), index=weeks, columns=columns)
                wide.index.name = 'trade_date'
                wide.columns.name = 'stock_code'
                long = wide.stack().rename('factor_value').reset_index()
                daily_signal = align_weekly_to_daily(long, panel, obj._price_df[['ts_code', 'trade_date']])
                prefix = 'r4_' + name
                daily_signal.to_parquet(OUT / 'signals' / (prefix + '.parquet'), index=False)
                print('START ' + prefix, flush=True)
                row = {'candidate_id': prefix, 'label': name, 'eval_status': 'valid', 'model': model,
                       'common_finite_stock_weeks': int(common.sum()), 'formal_eligibility': False,
                       'code_hashes': {name: digest(ALCHEMY / name) for name in CODE},
                       'costs': cost_diagnostics(obj, daily_signal, prefix)}
                write_json(checkpoint, row)
            rows.append(row)
            write_json(OUT / 'r4_evaluation.json', {
                'status': 'evaluated' if len(rows) == len(config['arms']) else 'partial', 'rows': rows})
        assert len(rows) == 6


def restore_line_endings():
    baseline = json.loads((OUT / 'baseline' / 'manifest.json').read_text(encoding='utf-8'))
    extra = json.loads((OUT / 'caliber_baseline' / 'manifest.json').read_text(encoding='utf-8'))
    changed = []
    for rel in {**baseline['files'], **extra['files']}:
        src = ROOT / rel
        folder = 'baseline' if rel in baseline['files'] else 'caliber_baseline'
        old = (OUT / folder / rel).read_bytes()
        if b'\r\n' not in old or old.replace(b'\r\n', b'').count(b'\n'):
            continue
        raw = src.read_bytes()
        fixed = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        if raw != fixed:
            src.write_bytes(fixed)
            changed.append(rel)
    print(json.dumps({'crlf_restored': changed}))


def test_contracts():
    import io
    import unittest
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ALCHEMY / 'tests'))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    write_json(OUT / 'tests.json', {
        'run_at': datetime.now().astimezone().isoformat(), 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'successful': result.wasSuccessful(), 'output': stream.getvalue(),
        'code_hashes': {name: digest(ALCHEMY / name) for name in CODE},
    })
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['baseline', 'caliber_baseline', 'release', 'line-endings', 'freeze', 'replay',
                                       'prepare-llm', 'generate-llm', 'evaluate-llm',
                                       'prepare-r4', 'evaluate-r4', 'runtime', 'tests'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.step == 'line-endings':
        restore_line_endings()
    elif args.step == 'freeze':
        freeze()
    elif args.step == 'replay':
        replay()
    elif args.step == 'prepare-llm':
        prepare_llm()
    elif args.step == 'generate-llm':
        generate_llm()
    elif args.step == 'evaluate-llm':
        evaluate_llm()
    elif args.step == 'prepare-r4':
        prepare_r4()
    elif args.step == 'evaluate-r4':
        evaluate_r4()
    elif args.step == 'runtime':
        runtime_inventory()
    elif args.step == 'tests':
        test_contracts()
    else:
        snapshot(args.step)


if __name__ == '__main__':
    main()
