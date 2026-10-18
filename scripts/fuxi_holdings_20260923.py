"""Isolated, versioned holdings-marking repair and fixed-signal counterfactuals."""

import argparse
import copy
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from fuxi_drought_campaign_20260923 import digest, write_json, switches, STATE

OUT = ROOT / 'reports/holdings_20260923'
OLD = ROOT / 'reports/drought_20260923'
FILES = ['research/factor_alchemy/s5_joint_filter.py', 'scripts/s5_fullpool_rerun.py',
         'scripts/version_bump.py', 'AGENTS.md', 'CHANGELOG.md',
         '.workbuddy/memory/MEMORY.md', '.workbuddy/memory/_fuxi_landing_details.md',
         '.workbuddy/memory/2026-09-23.md'] + ['data/' + name for name in STATE]


def snapshot(label):
    target = OUT / label
    if target.exists():
        raise RuntimeError('immutable snapshot exists')
    names = list(FILES)
    if label == 'release':
        names += ['scripts/fuxi_holdings_20260923.py', 'scripts/fuxi_drought_research_20260923.py',
                  'scripts/register_holdings_20260923.py', 'research/factor_alchemy/tests/test_holdings_mark.py',
                  'reports/fuxi_holdings_and_drought_review_20260923.md',
                  'scripts/fuxi_holdings_diagnose_20260923.py',
                  'scripts/verify_holdings_20260923.py',
                  'research/factor_alchemy/multi_stage_validator.py']
    elif label == 'consumer_baseline':
        names = ['research/factor_alchemy/multi_stage_validator.py']
    elif label == 'marking_v1':
        names = FILES[:2]
    records = {}
    for name in names:
        path = ROOT / name
        if not path.exists():
            continue
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        records[name] = {'sha256': digest(dest), 'bytes': dest.stat().st_size}
    write_json(target / 'manifest.json', {'recorded_at': datetime.now().astimezone().isoformat(), 'files': records})
    if label == 'baseline':
        selected = ['mem_0626_broadcast_repaired', 'llm_family_1_1_fresh_llm',
                    'llm_family_2_1_fresh_llm', 'r4_equal_rank']
        write_json(OUT / 'protocol.json', {
            'created_at': datetime.now().astimezone().isoformat(), 'signals': selected,
            'selection': 'one repaired historical formula, first two hypothesis families, fixed equal-rank R4; no new search',
            'signal_hashes': {name: digest(OLD / 'signals' / (name + '.parquet')) for name in selected},
            'data': json.loads((OLD / 'runtime_inventory.json').read_text(encoding='utf-8'))['inputs'],
            'costs': {'legacy': [15], 'enforce': [15, 0]},
            'scope': 'holdings price/calendar independence, fixed denominator, explicit incomplete marking; not next-open/JQ execution',
            'thresholds': {'cost_bps': 15, 'coverage': .9, 'vol_scaled': .7, 'top_n': 80},
            'formal_promotion': False, 'history_is_research_exposed': True})
    print(json.dumps({'snapshot': label, 'files': len(records)}))


def normalize():
    files = json.loads((OUT / 'baseline/manifest.json').read_text(encoding='utf-8'))['files']
    manifests = [('baseline', files)]
    extra = OUT / 'consumer_baseline/manifest.json'
    if extra.exists():
        manifests.append(('consumer_baseline', json.loads(extra.read_text(encoding='utf-8'))['files']))
    for label, entries in manifests:
        for name in entries:
            old = (OUT / label / name).read_bytes()
            if b'\r\n' in old and b'\n' not in old.replace(b'\r\n', b''):
                path = ROOT / name
                raw = path.read_bytes()
                fixed = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
                if fixed != raw:
                    path.write_bytes(fixed)
    print('Original pure CRLF files normalized only.')


def test_all():
    import io
    import unittest
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'research/factor_alchemy/tests'))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    write_json(OUT / 'tests.json', {'tests_run': result.testsRun, 'successful': result.wasSuccessful(),
                                  'output': stream.getvalue(), 'core_sha256': digest(ROOT / FILES[0]),
                                  'code_hashes': {name: digest(ROOT / name) for name in
                                                 FILES[:2] + ['research/factor_alchemy/multi_stage_validator.py',
                                                             'research/factor_alchemy/tests/test_holdings_mark.py']}})
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise SystemExit(1)


def market():
    import numpy as np
    import pandas as pd
    from s5_joint_filter import S5JointFilter
    from run_contract import RunContract
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    results_dir = OUT / 'market_final'
    results_dir.mkdir(exist_ok=True)
    for name, entry in protocol['data'].items():
        assert digest(ROOT / name) == entry['sha256'], 'frozen data changed: ' + name
    pinned = OUT / ('execution_' + datetime.now().strftime('%H%M%S') + '.py')
    shutil.copy2(Path(__file__), pinned)
    core = {name: digest(ROOT / name) for name in FILES[:2]}
    with RunContract(parameters={'step': 'holdings_mark_ab', 'protocol_sha256': digest(OUT / 'protocol.json'),
                                 'harness': str(pinned), 'harness_sha256': digest(pinned)},
                     run_id='holdings_' + datetime.now().strftime('%H%M%S'), runs_dir=OUT / 'runs'):
        obj = S5JointFilter(cache_dir=OUT / 'cache')
        obj._load_data()
        assert len(obj._price_df) == 7049516 and len(obj._universe) == 5772
        assert str(obj._price_df.trade_date.max().date()) == '2026-09-09'
        rows = []
        for name in protocol['signals']:
            path = OLD / 'signals' / (name + '.parquet')
            assert digest(path) == protocol['signal_hashes'][name]
            signal = pd.read_parquet(path)
            outputs = {}
            for mode, costs in protocol['costs'].items():
                for cost in costs:
                    key = mode + '_' + str(cost)
                    dest = results_dir / 'rows' / (name + '_' + key + '.json')
                    ret_path = results_dir / 'returns' / (name + '_' + key + '.parquet')
                    if dest.exists():
                        row = json.loads(dest.read_text(encoding='utf-8'))
                        assert row['code_hashes'] == core and row['signal_sha256'] == digest(path)
                    else:
                        print('START ' + name + ' ' + key, flush=True)
                        started = time.time()
                        with switches(FUXI_S5_HOLDINGS_MARK=mode, FUXI_S5_COST_BPS=cost):
                            returns, m25, m26 = obj._backtest_and_metrics(signal)
                            passed, reason = obj._verdict(m25, m26)
                            diag = copy.deepcopy(obj._last_backtest_diag)
                        ret_path.parent.mkdir(exist_ok=True)
                        returns.rename('return').to_frame().to_parquet(ret_path)
                        row = {'signal': name, 'mode': mode, 'cost_bps': cost, 'metrics_2025': m25,
                               'metrics_2026': m26, 'diag': diag, 'reason': reason,
                               'economic_pass': bool(passed and diag.get('coverage_ok') and diag.get('accounting_ok', True)),
                               'evaluation_status': 'valid' if not (m25.get('error') or m26.get('error')) else 'data_insufficient',
                               'formal_eligibility': False, 'code_hashes': core, 'signal_sha256': digest(path),
                               'elapsed_seconds': time.time() - started, 'returns_file': str(ret_path)}
                        write_json(dest, row)
                    outputs[key] = row
                    print(json.dumps({'signal': name, 'arm': key, 'status': row['evaluation_status'],
                                      'ex25': row['metrics_2025']['excess_return'],
                                      'ex26': row['metrics_2026']['excess_return'],
                                      'accounting_ok': row['diag'].get('accounting_ok'),
                                      'missing': row['diag'].get('missing_held_price_cells'),
                                      'pass': row['economic_pass']}), flush=True)
            legacy = pd.read_parquet(outputs['legacy_15']['returns_file'])['return']
            old = pd.read_parquet(OLD / 'returns' / (name + '_bps15.parquet'))['return']
            pd.testing.assert_series_equal(legacy, old, check_names=False, check_freq=False)
            new = pd.read_parquet(outputs['enforce_15']['returns_file'])['return']
            paired = pd.concat([legacy.rename('legacy'), new.rename('enforce')], axis=1, sort=True).dropna()
            assert len(paired) > 0
            delta = paired.enforce - paired.legacy
            rows.append({'signal': name, 'arms': outputs, 'legacy_matches_frozen_v55': True,
                         'paired_observed_days': len(paired), 'changed_paired_days': int((delta.abs() > 1e-8).sum()),
                         'max_abs_daily_delta': float(delta.abs().max()),
                         'new_calendar_days': int(len(new.index.difference(legacy.index))),
                         'new_nan_days': int(new.isna().sum()), 'delta_is_alpha_evidence': False})
            write_json(results_dir / 'market_ab.json', {'rows': rows, 'status': 'complete' if len(rows) == 4 else 'partial'})
        # Actual default through the formal weekly consumer, with identical frozen formula.
        generation = json.loads((OLD / 'llm_generation.json').read_text(encoding='utf-8'))
        candidate = generation['rows'][0]['candidates'][0]
        with switches(FUXI_S5_HOLDINGS_MARK='enforce', FUXI_S5_COST_BPS=15):
            actual = obj.validate_factor(candidate['factor_name'], candidate['formula'], natural_freq='weekly')
        expected = pd.read_parquet(results_dir / 'returns/llm_family_1_1_fresh_llm_enforce_15.parquet')['return']
        pd.testing.assert_series_equal(actual.portfolio_returns, expected, check_names=False, check_freq=False)
        write_json(results_dir / 'e2e.json', {'actual_compared_days': len(expected), 'passed': actual.passed,
                                    'error': actual.error, 'reason': actual.reason, 'formal_eligibility': False,
                                    'code_hashes': core, 'core_unchanged_during_run': core == {n: digest(ROOT / n) for n in core}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['baseline', 'consumer_baseline', 'marking_v1', 'release', 'normalize', 'tests', 'market'])
    step = parser.parse_args().step
    OUT.mkdir(parents=True, exist_ok=True)
    if step in ('baseline', 'consumer_baseline', 'marking_v1', 'release'):
        snapshot(step)
    else:
        {'normalize': normalize, 'tests': test_all, 'market': market}[step]()
