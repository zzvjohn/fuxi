"""Snapshot documentation and seal the isolated campaign's evidence."""

import argparse
import ast
import json
import shutil
from datetime import datetime

from fuxi_drought_campaign_20260923 import ROOT, OUT, ALCHEMY, CODE, STATE, digest, write_json


DOCS = ['AGENTS.md', '.workbuddy/memory/MEMORY.md', '.workbuddy/memory/_fuxi_landing_details.md',
        '.workbuddy/memory/2026-09-23.md', 'reports/fuxi_drought_contracts_20260923.md',
        'reports/fuxi_drought_campaign_20260923.md']


def snapshot_docs():
    folder = OUT / 'documentation_before_release'
    if folder.exists():
        raise RuntimeError('documentation snapshot already exists')
    files = {}
    for name in DOCS:
        source = ROOT / name
        if not source.exists():
            continue
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[name] = {'sha256': digest(target), 'bytes': target.stat().st_size,
                       'pure_crlf': b'\r\n' in target.read_bytes()
                       and b'\n' not in target.read_bytes().replace(b'\r\n', b'')}
    write_json(folder / 'manifest.json', {'created_at': datetime.now().astimezone().isoformat(), 'files': files})
    print('Documentation snapshot recorded.')


def format_docs():
    files = json.loads((OUT / 'documentation_before_release' / 'manifest.json').read_text(encoding='utf-8'))['files']
    changed = []
    for name in DOCS:
        path = ROOT / name
        if not path.exists() or (name in files and not files[name]['pure_crlf']):
            continue
        before = path.read_bytes()
        after = before.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        if before != after:
            path.write_bytes(after)
            changed.append(name)
    print(json.dumps({'crlf_restored': changed}))


def seal():
    target = OUT / 'release_audit.json'
    if target.exists():
        raise RuntimeError('release audit is immutable')
    snapshots = {}
    for folder in ('baseline', 'caliber_baseline', 'replay_engine', 'documentation_before_release', 'release'):
        manifest = json.loads((OUT / folder / 'manifest.json').read_text(encoding='utf-8'))['files']
        for name, entry in manifest.items():
            assert digest(OUT / folder / name) == entry['sha256'], (folder, name)
        snapshots[folder] = len(manifest)
    baseline = json.loads((OUT / 'baseline' / 'manifest.json').read_text(encoding='utf-8'))['files']
    operating = {}
    for name in STATE[:5]:
        key = str((ROOT / 'data' / name).relative_to(ROOT))
        operating[name] = digest(ROOT / 'data' / name) == baseline[key]['sha256']
    assert all(operating.values()), operating
    inventory = json.loads((OUT / 'runtime_inventory.json').read_text(encoding='utf-8'))['inputs']
    for name, entry in inventory.items():
        assert digest(ROOT / name) == entry['sha256'], name
    tests = json.loads((OUT / 'tests.json').read_text(encoding='utf-8'))
    current = {name: digest(ALCHEMY / name) for name in CODE}
    assert tests['successful'] and tests['code_hashes'] == current
    statuses = {p.parent.name: json.loads(p.read_text(encoding='utf-8'))['status']
                for p in (OUT / 'runs').glob('*/manifest.json')}
    assert all(value in ('completed', 'failed') for value in statuses.values()), statuses
    readout = json.loads((OUT / 'readout.json').read_text(encoding='utf-8'))
    assert len(readout['details']) == 28
    assert sum(row['eval_status'] == 'valid' for row in readout['details']) == 24
    economic_pass = sum(row.get('economic_pass_15', False) for row in readout['details'])
    memory = (ROOT / '.workbuddy/memory/MEMORY.md').read_text(encoding='utf-8')
    memory_chars = len(''.join(memory.split()))
    assert memory_chars <= 3000, memory_chars
    for script in [ROOT / 'scripts' / name for name in [
            'fuxi_drought_campaign_20260923.py', 'fuxi_drought_readout_20260923.py',
            'fuxi_drought_signal_attribution_20260923.py', 'fuxi_drought_release_20260923.py',
            'register_drought_20260923.py', 'verify_drought_release_20260923.py']]:
        ast.parse(script.read_text(encoding='utf-8-sig'))
    for name in DOCS:
        raw = (ROOT / name).read_bytes()
        previous = OUT / 'documentation_before_release' / name
        old = previous.read_bytes() if previous.exists() else b'\r\n'
        if b'\r\n' in old:
            assert raw.replace(b'\r\n', b'').count(b'\n') <= old.replace(b'\r\n', b'').count(b'\n'), name
        else:
            assert b'\r\n' not in raw, name
        if name == '.workbuddy/memory/2026-09-23.md':
            assert raw.startswith(old), 'daily history was rewritten'
    old_registry = json.loads((OUT / 'baseline/data/version_registry.json').read_text(encoding='utf-8'))
    current_registry = json.loads((ROOT / 'data/version_registry.json').read_text(encoding='utf-8'))
    assert current_registry['versions'][:len(old_registry['versions'])] == old_registry['versions']
    old_proposals = json.loads((OUT / 'baseline/data/stage1_optimization_proposals.json').read_text(encoding='utf-8'))
    current_proposals = json.loads((ROOT / 'data/stage1_optimization_proposals.json').read_text(encoding='utf-8'))
    assert current_proposals[:len(old_proposals)] == old_proposals
    evidence = {}
    for path in OUT.rglob('*'):
        if not path.is_file() or path.suffix not in ('.json', '.parquet', '.py', '.md'):
            continue
        if any(part in ('baseline', 'caliber_baseline', 'replay_engine', 'release',
                        'documentation_before_release', 'test_tmp') for part in path.relative_to(OUT).parts):
            continue
        evidence[str(path.relative_to(OUT))] = digest(path)
    r5 = {'recorded_at': datetime.now().astimezone().isoformat(),
          'status': 'protocol_frozen_no_candidate_promoted',
          'reason': 'No isolated diagnostic grants formal S1-S5 or JQ eligibility.',
          'economic_pass_count_15bps': economic_pass,
          'candidate_budget_exhausted': True, 'additional_model_calls_authorized': False,
          'eligible_candidates': [], 'prospective_observations': 0,
          'future_start_rule': 'Only observations unknown after a later eligible-candidate freeze count.',
          'required_freeze': ['full formula and direction', 'weights and execution', 'data and availability contract',
                              'code/runtime hash', 'minimum sample and success/failure criteria', 'complete trial ledger'],
          'rules': ['Changing any candidate restarts its independent validation.',
                    'Previously realized but not locally loaded dates are not automatically prospective.',
                    'No unattended collection, promotion, JQ submission, or live trading is enabled.'],
          'protocol_sha256': digest(OUT / 'protocol.json'),
          'r4_protocol_sha256': digest(OUT / 'r4_protocol.json'),
          'full_r5_acceptance': False}
    write_json(OUT / 'r5_status.json', r5)
    evidence['r5_status.json'] = digest(OUT / 'r5_status.json')
    write_json(target, {'recorded_at': datetime.now().astimezone().isoformat(),
                        'verified_snapshots': snapshots, 'unchanged_operating_state': operating,
                        'unchanged_input_files': list(inventory), 'run_statuses': statuses,
                        'tests': tests['tests_run'], 'core_sha256': current,
                        'memory_nonwhitespace_chars': memory_chars, 'evidence_sha256': evidence,
                        'economic_pass_count_15bps': economic_pass,
                        'new_formally_eligible_candidates': 0,
                        'limitations': 'snapshot versioning, not Git commits; diagnostic campaign, not full-pool or live-execution validation'})
    print(json.dumps({'sealed': True, 'evidence_files': len(evidence), 'tests': tests['tests_run'],
                      'operating_states_unchanged': len(operating), 'memory_chars': memory_chars}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['snapshot-docs', 'format-docs', 'seal'])
    step = parser.parse_args().step
    {'snapshot-docs': snapshot_docs, 'format-docs': format_docs, 'seal': seal}[step]()
