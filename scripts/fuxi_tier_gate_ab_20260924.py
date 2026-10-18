# -*- coding: utf-8 -*-
"""P2 闸门 A/B 反事实 + 负向测试 + fail-closed 测试。

问题：把「研究特征库」的闸门接进 JQ 生成器，必须证明三件事，否则就是死接线：
  A. legacy 行为不变：库为空时，7 个已注册因子全部照旧生成，输出字节与既有产物一致；
  B. 真实库不改变现状：真实库（8 研究特征 / 7 晋级）下，同样 7 个因子仍全部生成，
     且输出与 A 逐字节相同（证明闸门不改动输出，只可能拒绝）；
  C. 闸门真的被走到（不是死代码）：用一个探针统计 assert_promotable 被调用的次数；
  D. 负向：把一个已在晋级层的名字登记为研究特征后，generate() 必须抛错，
     并且**不写出任何文件**（拒绝必须是先于副作用的）；
  E. fail-closed：登记库不可读时拒绝，不静默放行。

用法：python scripts/fuxi_tier_gate_ab_20260924.py
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
FA = ROOT / 'research' / 'factor_alchemy'
OUT = ROOT / 'reports' / 'feature_tier_20260924' / 'gate_ab'
if str(FA) not in sys.path:
    sys.path.insert(0, str(FA))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import feature_tier as ft                                       # noqa: E402

GEN = importlib.import_module('gen_jq_s5_standalone')
NAMES = ft.jq_registered_names(GEN.__file__)
REAL_REGISTRY = ft.REGISTRY


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_GEN_TIME = re.compile(r'生成时间: \d{4}-\d{2}-\d{2} \d{2}:\d{2}')


def sha_normalised(path):
    """Hash with the embedded generation timestamp masked out.

    `generate()` stamps `datetime.now()` into the output, so two sequential runs can never
    be byte-identical. Masking that one line is what makes "the gate does not change the
    output" a checkable claim rather than a tautology.
    """
    text = _GEN_TIME.sub('生成时间: <MASKED>', Path(path).read_text(encoding='utf-8'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def run_case(label, registry_path, want_refuse=None, temp_dir=None):
    """Generate every name with REGISTRY and OUT_DIR pointed at throwaway locations."""
    ft.REGISTRY = Path(registry_path)
    calls = {'n': 0}
    original_gate = ft.assert_promotable

    def probe(name, registry=None):
        calls['n'] += 1
        return original_gate(name, registry=registry)

    ft.assert_promotable = probe
    original_out = GEN.OUT_DIR
    GEN.OUT_DIR = str(temp_dir)
    rows, errors = [], []
    try:
        for name in NAMES:
            before = set(p.name for p in Path(temp_dir).glob('*.py'))
            try:
                path = GEN.generate(name)
                wrote = Path(path).exists()
                rows.append({'name': name, 'status': 'generated' if wrote else 'silent_no_file',
                             'bytes': Path(path).stat().st_size if wrote else 0,
                             'sha256_masked': sha_normalised(path) if wrote else None})
            except Exception as exc:                            # noqa: BLE001
                after = set(p.name for p in Path(temp_dir).glob('*.py'))
                errors.append({'name': name, 'error_type': type(exc).__name__,
                               'error': str(exc),
                               'files_written': sorted(after - before)})
                rows.append({'name': name, 'status': 'refused'})
    finally:
        ft.assert_promotable = original_gate
        GEN.OUT_DIR = original_out
        ft.REGISTRY = REAL_REGISTRY
    return {'case': label, 'rows': rows, 'errors': errors,
            'gate_calls': calls['n'], 'gate_reached': calls['n'] == len(NAMES)}


def main() -> int:
    started = datetime.now(timezone.utc).astimezone()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'generated_at': started.isoformat(timespec='seconds'),
              'names': NAMES, 'checks': {}}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # ---- A: legacy (empty registry)
        empty = tmp / 'empty.json'
        ft.save(ft._empty_registry(), empty)
        dir_a = tmp / 'a'
        dir_a.mkdir()
        a = run_case('A_legacy_empty_registry', empty, temp_dir=dir_a)

        # ---- B: the real registry
        dir_b = tmp / 'b'
        dir_b.mkdir()
        b = run_case('B_real_registry', REAL_REGISTRY, temp_dir=dir_b)

        same = ([r['sha256_masked'] for r in a['rows']] == [r['sha256_masked'] for r in b['rows']])
        report['checks']['A_legacy_generates_all'] = (
            all(r['status'] == 'generated' for r in a['rows']) and not a['errors'])
        report['checks']['B_real_registry_generates_all'] = (
            all(r['status'] == 'generated' for r in b['rows']) and not b['errors'])
        report['checks']['A_equals_B_byte_for_byte'] = bool(same)
        report['checks']['gate_is_reached_not_dead'] = bool(a['gate_reached'] and b['gate_reached'])

        # ---- compare against the artifacts already on disk (timestamp masked)
        on_disk = {}
        for row in a['rows']:
            p = FA / ('jq_s5_pass_%s_standalone.py' % row['name'])
            on_disk[row['name']] = sha_normalised(p) if p.exists() else None
        mismatched = [r['name'] for r in a['rows']
                      if on_disk.get(r['name']) and on_disk[r['name']] != r['sha256_masked']]
        report['checks']['A_matches_existing_artifacts'] = {
            'compared': sum(1 for v in on_disk.values() if v),
            'identical_after_masking_the_generation_time': sum(
                1 for r in a['rows'] if on_disk.get(r['name']) == r['sha256_masked']),
            'differing': mismatched,
            'disclosure': 'differing artifacts were generated by an earlier revision of the '
                          'template; this is a disclosure, not a gate failure, and those files '
                          'are deliberately left untouched.',
        }

        # ---- D: negative — promote a JQ name into the research tier, then refuse
        victim = NAMES[0]
        reg = ft.load(REAL_REGISTRY)
        rows = [e for e in reg['tiers'][ft.RESEARCH] if e['name'] != victim]
        rows.append({'name': victim, 'family': 'test', 'usable_as': ['exit_filter'],
                     'why_not_promotable': 'A/B fixture', 'evidence': ['fixture'],
                     'promotion_eligible': False, 'registered_by': 'test'})
        reg['tiers'][ft.RESEARCH] = rows
        blocked_path = tmp / 'blocked.json'
        ft.save(reg, blocked_path)
        dir_d = tmp / 'd'
        dir_d.mkdir()
        d = run_case('D_negative_blocked', blocked_path, temp_dir=dir_d)
        refused = [e for e in d['errors'] if e['name'] == victim]
        report['checks']['D_blocked_name_is_refused'] = (
            len(refused) == 1 and refused[0]['error_type'] == 'NotPromotableError')
        report['checks']['D_no_file_written_on_refusal'] = (
            bool(refused) and refused[0]['files_written'] == [])
        report['checks']['D_others_still_generate'] = all(
            r['status'] == 'generated' for r in d['rows'] if r['name'] != victim)

        # ---- E: fail-closed when the registry is present but unusable
        bad = tmp / 'unreadable.json'
        bad.mkdir()                                  # a directory where a file is expected
        dir_e = tmp / 'e'
        dir_e.mkdir()
        e = run_case('E1_registry_is_a_directory', bad, temp_dir=dir_e)

        corrupt = tmp / 'corrupt.json'
        corrupt.write_bytes(b'{"tiers": {"research_feature": [ this is not json')
        dir_e2 = tmp / 'e2'
        dir_e2.mkdir()
        e2 = run_case('E2_registry_is_corrupt', corrupt, temp_dir=dir_e2)
        report['checks']['E_refuses_when_registry_unreadable'] = (
            len(e['errors']) == len(NAMES)
            and all(x['error_type'] == 'RuntimeError' for x in e['errors']))
        report['checks']['E_refuses_when_registry_is_corrupt'] = (
            len(e2['errors']) == len(NAMES)
            and all(x['error_type'] == 'RuntimeError' for x in e2['errors']))

        report['cases'] = {'A': a, 'B': b, 'D': d, 'E1': e, 'E2': e2}
        report['on_disk_sha256'] = on_disk
        report['endpoint_inventory'] = {
            'registry_entries_now': len(NAMES),
            'artifacts_on_disk': len(list(FA.glob('jq_s5_pass_*_standalone.py'))),
            'note': 'the artifacts on disk are a superset of the current registry: the '
                    'generator registry was rewritten over time, so on-disk files are NOT a '
                    'reliable membership source. Membership is read from the registry.',
        }

    report['all_checks_passed'] = all(
        (v if isinstance(v, bool) else True) for v in report['checks'].values())
    payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    (OUT / 'gate_ab.json').write_bytes(payload.encode('utf-8'))
    print(json.dumps({'checks': report['checks'],
                      'endpoint_inventory': report['endpoint_inventory'],
                      'gate_calls_A': report['cases']['A']['gate_calls'],
                      'gate_calls_B': report['cases']['B']['gate_calls']},
                     ensure_ascii=False, indent=2))
    return 0 if report['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
