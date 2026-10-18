"""Seal holdings evidence and verify historical assets were not rewritten."""

import json
from datetime import datetime

import pandas as pd

from fuxi_holdings_20260923 import ROOT, OUT, OLD, FILES, STATE, digest, write_json
from multi_stage_validator import MultiStageValidator, ValidationResult
from s5_joint_filter import S5JointFilter, S5Result


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    destination = OUT / 'release_audit.json'
    if destination.exists():
        raise RuntimeError('release audit already sealed')
    protocol = read(OUT / 'protocol.json')
    attribution = read(OUT / 'marking_attribution.json')
    data_hashes = {name: row['sha256'] for name, row in protocol['data'].items()}
    data_hashes.update(attribution['source_sha256'])
    for name, expected in data_hashes.items():
        assert digest(ROOT / name) == expected, name
    snapshots = {}
    for label in ['baseline', 'consumer_baseline', 'marking_v1', 'release']:
        manifest = read(OUT / label / 'manifest.json')['files']
        for name, entry in manifest.items():
            assert digest(OUT / label / name) == entry['sha256'], name
            if label == 'release':
                assert digest(ROOT / name) == entry['sha256'], name
        snapshots[label] = len(manifest)
    operating = {}
    for name in STATE[:5]:
        old = OUT / 'baseline/data' / name
        current = ROOT / 'data' / name
        assert digest(old) == digest(current), name
        operating[name] = digest(current)
    for name in ['.workbuddy/memory/2026-09-23.md', '.workbuddy/memory/_fuxi_landing_details.md']:
        assert (ROOT / name).read_bytes().startswith((OUT / 'baseline' / name).read_bytes()), name
    old_proposals = read(OUT / 'baseline/data/stage1_optimization_proposals.json')
    proposals = read(ROOT / 'data/stage1_optimization_proposals.json')
    assert proposals[:len(old_proposals)] == old_proposals
    assert len(proposals) == len(old_proposals) + 1
    assert proposals[-1]['proposal_id'] == 'P-20260923-004'
    old_versions = read(OUT / 'baseline/data/version_registry.json')['versions']
    versions = read(ROOT / 'data/version_registry.json')['versions']
    assert len(versions) == len(old_versions) + 1 == 57
    assert versions[-1]['version'] == 'v0.7.56'
    for before, after in zip(old_versions, versions):
        before, after = dict(before), dict(after)
        old_landmark = before.pop('landmark', '')
        new_landmark = after.pop('landmark', '')
        assert new_landmark.startswith(old_landmark)
        assert before == after, before['version']
    historical = read(OLD / 'release_audit.json')['evidence_sha256']
    for name, expected in historical.items():
        assert digest(OLD / name) == expected, name
    core = {name: digest(ROOT / name) for name in FILES[:2]}
    ab = read(OUT / 'market_final/market_ab.json')
    first = {row['signal']: row for row in read(OUT / 'market_ab.json')['rows']}
    assert ab['status'] == 'complete' and len(ab['rows']) == 4
    consumer, repeat_comparisons = [], []
    for row in ab['rows']:
        assert row['paired_observed_days'] > 0 and row['legacy_matches_frozen_v55']
        assert set(row['arms']) == {'legacy_15', 'enforce_15', 'enforce_0'}
        for key, arm in row['arms'].items():
            assert arm['code_hashes'] == core
            assert not arm['formal_eligibility']
            original = first[row['signal']]['arms'][key]
            old_returns = pd.read_parquet(original['returns_file'])['return']
            new_returns = pd.read_parquet(arm['returns_file'])['return']
            assert len(new_returns) > 0
            pd.testing.assert_series_equal(new_returns, old_returns)
            assert arm['diag'] == original['diag']
            for year in (2025, 2026):
                assert arm[f'metrics_{year}'] == original[f'metrics_{year}']
            repeat_comparisons.append({'signal': row['signal'], 'arm': key,
                                       'compared_dates_including_nan': len(new_returns)})
        observed = row['arms']['enforce_15']
        assert observed['evaluation_status'] == 'data_insufficient'
        result = S5Result(factor_name=row['signal'])
        S5JointFilter._fill_metrics(result, observed['metrics_2025'], observed['metrics_2026'])
        result.error = observed['metrics_2025'].get('error') or observed['metrics_2026'].get('error')
        result.reason = observed['reason']
        validation = ValidationResult(row['signal'])
        MultiStageValidator()._apply_stage5_from_s5result(result, [validation], 0)
        assert validation.s5_eval_status == 'data_insufficient' and not validation.s5_passed
        consumer.append({'signal': row['signal'], 'status': validation.s5_eval_status,
                         'window_days_2025': validation.s5_window_days_2025,
                         'window_days_2026': validation.s5_window_days_2026})
    e2e = read(OUT / 'market_final/e2e.json')
    assert e2e['actual_compared_days'] == 1354 and e2e['core_unchanged_during_run']
    assert e2e['error'] == 'held_price_observations_missing' and e2e['code_hashes'] == core
    assert all(row['independent_count_matches_frozen_ab'] for row in attribution['rows'])
    tests = read(OUT / 'tests.json')
    assert tests['successful'] and tests['tests_run'] == 53
    for name, expected in tests['code_hashes'].items():
        assert digest(ROOT / name) == expected
    memory_chars = len(''.join((ROOT / '.workbuddy/memory/MEMORY.md').read_text(encoding='utf-8').split()))
    assert memory_chars <= 3000, memory_chars
    line_endings = {}
    for label in ['baseline', 'consumer_baseline']:
        for name in read(OUT / label / 'manifest.json')['files']:
            old = (OUT / label / name).read_bytes()
            new = (ROOT / name).read_bytes()
            old_crlf, new_crlf = old.count(b'\r\n'), new.count(b'\r\n')
            old_lf = old.replace(b'\r\n', b'').count(b'\n')
            new_lf = new.replace(b'\r\n', b'').count(b'\n')
            if old_crlf:
                assert new_lf <= old_lf, name
            else:
                assert new_crlf == 0, name
            line_endings[name] = {'crlf': new_crlf, 'bare_lf': new_lf}
    statuses = {}
    for path in (OUT / 'runs').glob('*/manifest.json'):
        run = read(path)
        statuses[path.parent.name] = run['status']
        assert run['status'] != 'running'
    assert statuses
    for path in (OUT / 'release/scripts').glob('*.py'):
        compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
    evidence = {}
    for path in OUT.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.parquet', '.py', '.md'):
            evidence[path.relative_to(OUT).as_posix()] = digest(path)
    write_json(destination, {
        'recorded_at': datetime.now().astimezone().isoformat(), 'version': versions[-1]['version'],
        'verified_snapshots': snapshots, 'unchanged_data_sha256': data_hashes,
        'unchanged_operating_state': operating, 'previous_sealed_evidence_unchanged': len(historical),
        'historical_proposals_unchanged': len(old_proposals), 'historical_versions_preserved': len(old_versions),
        'dated_logs_append_only': True, 'memory_nonwhitespace_chars': memory_chars,
        'tests': {'run': tests['tests_run'], 'successful': tests['successful']}, 'code_sha256': tests['code_hashes'],
        'market_ab_signals': 4, 'market_ab_arms': 12, 'new_valid_economic_failures': 0,
        'compatibility_repeat_return_pairs': repeat_comparisons,
        'data_insufficient_representatives': 4, 'new_formally_eligible_candidates': 0,
        'consumer_frozen_result_replay': consumer,
        'consumer_replay_note': 'current consumer replays frozen market metrics; not another full pipeline market run',
        'e2e_compared_dates_including_nan': 1354, 'run_statuses': statuses,
        'line_endings': line_endings, 'evidence_sha256': evidence,
        'limits': ['fixed-share NAV/next-session execution not implemented',
                   'unknown price/lifecycle events remain; confirmed halts were classified, not filled',
                   '2025/2026 research-exposed, not independent OOS',
                   '53 tests and four representatives do not certify every execution path']})
    print(json.dumps({'sealed': True, 'version': versions[-1]['version'], 'evidence_files': len(evidence),
                      'tests': tests['tests_run'], 'data_insufficient': 4, 'new_jq': 0,
                      'memory_chars': memory_chars}, ensure_ascii=False))


if __name__ == '__main__':
    main()
