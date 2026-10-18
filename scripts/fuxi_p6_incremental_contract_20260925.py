# -*- coding: utf-8 -*-
"""P6 批次：固化增量对照的冻结设计（不同时改多个维度）。

本脚本只读封存协议与源码字面量, 只写 reports/incremental_contract_20260925/。
不跑任何实验, 不读任何标签, 不翻任何默认, 不产生任何经济裁决。

用法:
    python scripts/fuxi_p6_incremental_contract_20260925.py --tests
    python scripts/fuxi_p6_incremental_contract_20260925.py --run
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/incremental_contract_20260925'
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))

from incremental_contract import (  # noqa: E402
    BOOKKEEPING_KEYS,
    MODEL_COMPARE_AXES,
    MODEL_COMPARE_BASELINE,
    check_increment,
    classify_key,
    design_fields_from_source,
    feature_delta,
    protocol_sha256,
)

PROTOCOL_GLOBS = ('reports/*/protocol.json',)
MODEL_COMPARE_SCRIPT = ROOT / 'scripts/fuxi_model_compare_20260924.py'
MODEL_COMPARE_PROTOCOL = ROOT / MODEL_COMPARE_BASELINE
MODEL_COMPARE_HISTORY = (ROOT / 'reports/model_compare_20260924/protocol_history'
                         / 'protocol_00.json')
SPEC = {'axes': MODEL_COMPARE_AXES}

# 与 P6 的落点直接相关的两条纪律（写在这里以便报告引用同一份字符串）
DISCIPLINE = (
    '一个增量对照相对其基准只允许改**一个轴**；当该轴是特征集时，一次只允许触碰'
    '**一个**特征（增/删/改 direction|kind|source 各算触碰一次）。'
)


# ------------------------------------------------------------------ 工具

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, payload):
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + '\n'
    data = raw.encode('utf-8')
    assert data[:3] != b'\xef\xbb\xbf'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(data)
    return len(data)


def write_csv(path, header, rows):
    import csv
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(header)
    for row in rows:
        w.writerow(row)
    data = buf.getvalue().encode('utf-8')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(data)
    return len(data)


def protocol_paths():
    paths = []
    for pattern in PROTOCOL_GLOBS:
        paths.extend(sorted(ROOT.glob(pattern)))
    return sorted(set(paths))


# ------------------------------------------------------------------ 普查

def census():
    """全仓协议的键分类普查。

    🔴 两类数都要报：有修订通道的**数目**，以及**没有**修订通道的数目与名单。
    只报前者会让「13 份协议没有任何『相对基准改了什么』的记账通道」这件事消失。
    """
    rows, with_channel, without = [], [], []
    for path in protocol_paths():
        data = read_json(path)
        keys = sorted(data)
        design = [k for k in keys if classify_key(k, SPEC) == 'DESIGN']
        bookkeeping = [k for k in keys if classify_key(k, SPEC) == 'BOOKKEEPING']
        reading = [k for k in keys if classify_key(k, SPEC) == 'READING']
        has_revisions = 'revisions' in data
        row = {
            'path': path.relative_to(ROOT).as_posix(),
            'file_sha256': sha256_file(path),
            'canonical_sha256': protocol_sha256(data),
            'keys': len(keys),
            'design_keys': design,
            'bookkeeping_keys': bookkeeping,
            'reading_keys': reading,
            'has_a_revision_channel': has_revisions,
        }
        rows.append(row)
        (with_channel if has_revisions else without).append(row['path'])
    return {
        'protocol_count': len(rows),
        'with_a_revision_channel': with_channel,
        'without_a_revision_channel': without,
        'rows': rows,
    }


# ------------------------------------------------------------------ 历史修订复核

def recheck_the_recorded_revision():
    """独立复核 model_compare 的那一条修订声明。

    它记的是「只有 plumbing 修复、design_fields_changed 为空」。这里重新推导一遍，
    并额外查一件旧机制没保证的事：**记的摘要是文件摘要还是内容摘要**。
    """
    base = read_json(MODEL_COMPARE_HISTORY)
    cand = read_json(MODEL_COMPARE_PROTOCOL)
    recorded = list(cand.get('revisions', []))
    if not recorded:
        raise RuntimeError('model_compare 没有 revisions 记录，复核无从谈起')
    first = recorded[0]
    out = check_increment(base, cand, spec=SPEC,
                          baseline_sha256=protocol_sha256(base),
                          candidate_sha256=protocol_sha256(cand))
    file_digest = sha256_file(MODEL_COMPARE_HISTORY)
    content_digest = protocol_sha256(base)
    # 重排键序 + 改缩进：文件摘要必变，内容摘要必不变
    reserialized = json.dumps(base, ensure_ascii=False, indent=4, sort_keys=True) + '\n'
    reserialized_digest = hashlib.sha256(reserialized.encode('utf-8')).hexdigest()
    return {
        'recorded_claim': {
            'design_fields_changed': first.get('design_fields_changed'),
            'code_hash_changed': first.get('code_hash_changed'),
            'reason': first.get('reason'),
            'previous_protocol_sha256': first.get('previous_protocol_sha256'),
        },
        'rederived': {
            'verdict': out['verdict'],
            'accepted': out['accepted'],
            'changed_design_fields': out['changed_design_fields'],
            'changed_bookkeeping_fields': out['changed_bookkeeping_fields'],
            'axis_declaration_size': len(out['axis_declaration']),
        },
        'claim_is_reverified': (list(first.get('design_fields_changed') or []) == []
                                and out['changed_design_fields'] == []
                                and out['verdict'] == 'IDENTICAL'),
        'digest_scope': {
            'recorded_digest_equals_the_file_sha256':
                first.get('previous_protocol_sha256') == file_digest,
            'recorded_digest_equals_the_content_sha256':
                first.get('previous_protocol_sha256') == content_digest,
            'file_sha256': file_digest,
            'content_sha256': content_digest,
            'reserialized_file_sha256': reserialized_digest,
            'reserialization_keeps_the_content_digest':
                content_digest == protocol_sha256(json.loads(reserialized)),
            'note': ('旧机制记的是**文件**摘要 ⇒ 同一内容换个缩进/键序就换了身份，'
                     '不能当「基准是谁」的判据；protocol_sha256 给的是**内容**摘要。'),
        },
    }


# ------------------------------------------------------------------ 反事实矩阵

def _features(protocol):
    return [dict(f) for f in protocol['features']]


def counterfactuals():
    """8 个对照，全部由**真实封存协议**派生（不手工搭 dict）。"""
    base = read_json(MODEL_COMPARE_PROTOCOL)
    rows = []

    def probe(tag, intent, build, expectation, spec=None):
        cand = build(copy.deepcopy(base))
        out = check_increment(base, cand, spec=SPEC if spec is None else spec,
                              baseline_sha256=protocol_sha256(base),
                              candidate_sha256=protocol_sha256(cand))
        row = {
            'tag': tag,
            'intent': intent,
            'expected': expectation,
            'verdict': out['verdict'],
            'accepted': out['accepted'],
            'refuses': out['refuses'],
            'is_an_increment': out['is_an_increment'],
            'axes_changed': out['axes_changed'],
            'changed_design_fields': out['changed_design_fields'],
            'changed_bookkeeping_fields': out['changed_bookkeeping_fields'],
            'changed_reading_fields': out['changed_reading_fields'],
            'feature_delta': out['feature_delta'],
            'flags': [f['code'] for f in out['flags']],
            'reasons': out['reasons'],
        }
        row['matches_the_expectation'] = (row['verdict'] == expectation)
        rows.append(row)
        return out

    probe('C1_identical', '同一份协议（JSON 往返）', lambda p: p, 'IDENTICAL')
    probe('C2_bookkeeping_only', '只换摘要与时间戳（plumbing 重冻结的合法情形）',
          lambda p: {**p, 'hashes': {**p['hashes'],
                                     'data/raw/daily_prices.csv': '0' * 64}},
          'IDENTICAL')
    probe('C3_single_feature_added', '把 pe_ttm 加回特征集（**合法的单维增量**）',
          lambda p: {**p,
                     'features': sorted(_features(p) + [{
                         'name': 'pe_ttm', 'kind': 'snapshot', 'direction': -1,
                         'source': 'pe_ttm',
                         'coverage': {'median_rate': 0.7232, 'min_rate': 0.0,
                                      'sections': 89}}], key=lambda f: f['name']),
                     'dropped_by_coverage': []},
          'SINGLE_AXIS')
    probe('C4_single_feature_plus_topn', '在 C3 之上同时把 Top80 改成 Top100（污染）',
          lambda p: {**p,
                     'features': sorted(_features(p) + [{
                         'name': 'pe_ttm', 'kind': 'snapshot', 'direction': -1,
                         'source': 'pe_ttm', 'coverage': {}}], key=lambda f: f['name']),
                     'dropped_by_coverage': [],
                     'top_n': 100,
                     'selection': p['selection'].replace('Top80', 'Top100')},
          'MULTI_AXIS')
    probe('C5_two_features_added', '一次加两个特征（pe_ttm + sf1）',
          lambda p: {**p,
                     'features': sorted(_features(p) + [
                         {'name': 'pe_ttm', 'kind': 'snapshot', 'direction': -1,
                          'source': 'pe_ttm', 'coverage': {}},
                         {'name': 'sf1', 'kind': 'snapshot', 'direction': -1,
                          'source': 'sf1', 'coverage': {}}], key=lambda f: f['name']),
                     'dropped_by_coverage': []},
          'AXIS_ITEM_BATCH')
    probe('C6_coverage_only', '数据延伸后重读 coverage（**不是设计变更**）',
          lambda p: {**p,
                     'features': [{**f,
                                   'coverage': {'median_rate': 0.5, 'min_rate': 0.0,
                                                'sections': 137}}
                                  for f in p['features']]},
          'IDENTICAL')
    probe('C7_undeclared_key', '多出一个未写下来的键（一个名字背后可以装多个维度）',
          lambda p: {**p, 'whatever_new_knob': 1}, 'UNDECLARED_KEY')
    probe('C8_rename_one_feature', '把一个特征改名（1 增 + 1 删 = 触碰 2 个）',
          lambda p: {**p,
                     'features': sorted([{**f, 'name': 'renamed_feature'}
                                         if f['name'] == 'pb' else f
                                         for f in p['features']],
                                        key=lambda f: f['name'])},
          'AXIS_ITEM_BATCH')
    probe('C9_same_key_now_written_down',
          '把 C7 那个键写进轴声明之后（写下来是唯一的合法化路径）',
          lambda p: {**p, 'whatever_new_knob': 1}, 'SINGLE_AXIS',
          spec={'axes': {**MODEL_COMPARE_AXES, 'misc': ('whatever_new_knob',)}})
    return {'discipline': DISCIPLINE, 'rows': rows,
            'vacuous': not (any(r['accepted'] for r in rows)
                            and any(r['refuses'] for r in rows))}


def monotonicity_probe():
    """从 C3 这个**被接受**的单轴增量出发，逐个加第二处设计改动。

    断言：一旦被拒，之后任何超集都必须保持被拒（单向下锁）。
    """
    base = read_json(MODEL_COMPARE_PROTOCOL)
    accepted = copy.deepcopy(base)
    accepted['features'] = sorted(_features(base) + [{
        'name': 'pe_ttm', 'kind': 'snapshot', 'direction': -1, 'source': 'pe_ttm',
        'coverage': {}}], key=lambda f: f['name'])
    accepted['dropped_by_coverage'] = []
    seed = check_increment(base, accepted, spec=SPEC)
    extra_edits = [
        ('top_n', 100),
        ('primary_cost_bps', 30),
        ('reference_arm', 'equal_rank'),
        ('segments', 'halves by index over the trained sections'),
        ('arms', ['equal_rank', 'ridge']),
    ]
    trace, refusals, violation = [], 0, False
    for field, value in extra_edits:
        probe = copy.deepcopy(accepted)
        probe[field] = value
        out = check_increment(base, probe, spec=SPEC)
        if out['refuses']:
            refusals += 1
        if refusals and not out['refuses']:
            violation = True
        trace.append({'added_change': field, 'verdict': out['verdict'],
                      'refuses': out['refuses']})
    return {'seed_verdict': seed['verdict'], 'seed_accepted': seed['accepted'],
            'probes': len(trace), 'refusals': refusals,
            'acceptance_came_back': violation, 'trace': trace}


def legacy_vs_contract():
    """旧守卫（0 与非 0 两档）与契约在同一组反事实上的差别。

    旧守卫在 model_compare 里是这样写的：重冻结时 ``design_fields_changed`` 非空即抛错。
    于是它**拒绝一切**设计变更 —— 包括合法的单维增量。听上去更安全，但真实的后果是：
    想跑一次增量实验，唯一的路是**另写一个批次脚本**，而那样就一份守卫都没有了。
    普查数就是这个后果的度量：14 份协议里 13 份没有任何修订通道。
    """
    rows = []
    for row in counterfactuals()['rows']:
        changed = len(row['changed_design_fields'])
        rows.append({
            'tag': row['tag'],
            'design_dimensions_moved': changed,
            'legacy_guard': 'refuses' if changed else 'accepts',
            'contract': 'accepts' if row['accepted'] else 'refuses',
            'legacy_and_contract_disagree': (('refuses' if changed else 'accepts')
                                             != ('accepts' if row['accepted'] else 'refuses')),
        })
    return {'rows': rows,
            'legacy_would_block_a_legitimate_increment': [
                r['tag'] for r in rows
                if r['legacy_guard'] == 'refuses' and r['contract'] == 'accepts'],
            'both_refuse_the_polluted_one': [
                r['tag'] for r in rows
                if r['legacy_guard'] == 'refuses' and r['contract'] == 'refuses']}


def axis_coverage():
    production = tuple(design_fields_from_source(MODEL_COMPARE_SCRIPT))
    covered = tuple(m for members in MODEL_COMPARE_AXES.values() for m in members)
    return {
        'production_design_fields': list(production),
        'production_design_field_count': len(production),
        'declared_axis_members': list(covered),
        'declared_axis_member_count': len(covered),
        'axes': {axis: list(members) for axis, members in sorted(MODEL_COMPARE_AXES.items())},
        'covers_every_production_design_field': sorted(covered) == sorted(production),
        'no_field_declared_twice': len(covered) == len(set(covered)),
        'bookkeeping_fallback_keys': sorted(BOOKKEEPING_KEYS),
    }


# ------------------------------------------------------------------ 单测

class CensusTests(unittest.TestCase):
    def test_every_protocol_is_censused(self):
        got = census()
        self.assertEqual(got['protocol_count'], len(protocol_paths()))
        self.assertGreaterEqual(got['protocol_count'], 14)

    def test_the_census_reports_both_sides_of_the_channel_count(self):
        """🔴 分母必须两个都报：被排除的项不得从输出里消失。"""
        got = census()
        self.assertEqual(got['protocol_count'],
                         len(got['with_a_revision_channel'])
                         + len(got['without_a_revision_channel']))
        self.assertEqual(got['without_a_revision_channel'],
                         [r['path'] for r in got['rows'] if not r['has_a_revision_channel']])
        self.assertTrue(got['without_a_revision_channel'],
                        '若没有「无通道」的协议，这条纪律就没人需要')

    def test_only_model_compare_has_a_revision_channel(self):
        got = census()
        self.assertEqual(got['with_a_revision_channel'],
                         ['reports/model_compare_20260924/protocol.json'])

    def test_the_batch_is_read_only_for_every_protocol(self):
        before = {str(p): sha256_file(p) for p in protocol_paths()}
        census()
        counterfactuals()
        recheck_the_recorded_revision()
        after = {str(p): sha256_file(p) for p in protocol_paths()}
        self.assertEqual(before, after, '本批不得改动任何封存协议')


class HistoricalRevisionTests(unittest.TestCase):
    def test_the_recorded_claim_is_reverified(self):
        got = recheck_the_recorded_revision()
        self.assertEqual(got['recorded_claim']['design_fields_changed'], [])
        self.assertTrue(got['claim_is_reverified'])
        self.assertEqual(got['rederived']['verdict'], 'IDENTICAL')

    def test_the_recorded_digest_is_a_file_digest_not_a_content_digest(self):
        got = recheck_the_recorded_revision()['digest_scope']
        self.assertTrue(got['recorded_digest_equals_the_file_sha256'])
        self.assertFalse(got['recorded_digest_equals_the_content_sha256'])
        self.assertNotEqual(got['file_sha256'], got['content_sha256'])

    def test_re_serializing_the_baseline_changes_its_file_digest_only(self):
        got = recheck_the_recorded_revision()['digest_scope']
        self.assertNotEqual(got['reserialized_file_sha256'], got['file_sha256'])
        self.assertTrue(got['reserialization_keeps_the_content_digest'])


class CounterfactualTests(unittest.TestCase):
    def test_the_matrix_is_not_vacuous(self):
        got = counterfactuals()
        self.assertFalse(got['vacuous'])
        self.assertGreaterEqual(len(got['rows']), 8)

    def test_every_probe_matches_its_expectation(self):
        got = counterfactuals()
        bad = [r['tag'] for r in got['rows'] if not r['matches_the_expectation']]
        self.assertEqual(bad, [])

    def test_a_legitimate_single_feature_increment_is_accepted(self):
        row = [r for r in counterfactuals()['rows']
               if r['tag'] == 'C3_single_feature_added'][0]
        self.assertEqual(row['verdict'], 'SINGLE_AXIS')
        self.assertTrue(row['accepted'])
        self.assertTrue(row['is_an_increment'])
        self.assertEqual(row['axes_changed'], ['feature_set'])
        self.assertEqual(row['feature_delta']['touched'], 1)

    def test_adding_one_extra_dimension_turns_an_acceptance_into_a_refusal(self):
        """这正是 P6 存在的理由：单维增量是合法的，顺手再改 Top80 就把它污染了。"""
        rows = {r['tag']: r for r in counterfactuals()['rows']}
        self.assertTrue(rows['C3_single_feature_added']['accepted'])
        self.assertTrue(rows['C4_single_feature_plus_topn']['refuses'])
        self.assertEqual(sorted(rows['C4_single_feature_plus_topn']['axes_changed']),
                         ['feature_set', 'selection'])
        self.assertEqual(rows['C4_single_feature_plus_topn']['feature_delta']['touched'], 1,
                         '特征只加了一个 —— 污染来自第二个轴，不是特征数')

    def test_two_features_at_once_is_refused_on_the_same_axis(self):
        rows = {r['tag']: r for r in counterfactuals()['rows']}
        self.assertEqual(rows['C5_two_features_added']['verdict'], 'AXIS_ITEM_BATCH')
        self.assertEqual(rows['C5_two_features_added']['feature_delta']['touched'], 2)
        self.assertEqual(rows['C5_two_features_added']['axes_changed'], ['feature_set'])

    def test_a_rename_counts_as_two_touched_features(self):
        row = [r for r in counterfactuals()['rows']
               if r['tag'] == 'C8_rename_one_feature'][0]
        self.assertEqual(row['feature_delta']['touched'], 2)
        self.assertEqual(row['verdict'], 'AXIS_ITEM_BATCH')

    def test_a_data_extension_is_not_a_design_change(self):
        """🔴 [mem:AD5]：补数据会让 coverage 变；它不得被读成一次设计变更。"""
        row = [r for r in counterfactuals()['rows']
               if r['tag'] == 'C6_coverage_only'][0]
        self.assertEqual(row['verdict'], 'IDENTICAL')
        self.assertTrue(row['accepted'])
        self.assertFalse(row['is_an_increment'])
        self.assertEqual(row['feature_delta']['touched'], 0)
        self.assertTrue(row['feature_delta']['readings_changed'])
        self.assertIn('features[].coverage', row['changed_reading_fields'])

    def test_an_unregistered_key_is_refused_and_flagged(self):
        """🔴 一个未写下来的键名背后可以装任意多个维度 ⇒ 必须拒绝，不是洁癖。"""
        row = [r for r in counterfactuals()['rows']
               if r['tag'] == 'C7_undeclared_key'][0]
        self.assertEqual(row['verdict'], 'UNDECLARED_KEY')
        self.assertTrue(row['refuses'])
        self.assertIn('whatever_new_knob', row['changed_design_fields'])
        self.assertIn('UNDECLARED_KEY', row['flags'])

    def test_writing_the_name_down_is_the_only_legalisation_path(self):
        row = [r for r in counterfactuals()['rows']
               if r['tag'] == 'C9_same_key_now_written_down'][0]
        self.assertEqual(row['verdict'], 'SINGLE_AXIS')
        self.assertTrue(row['accepted'])
        self.assertEqual(row['axes_changed'], ['misc'])

    def test_a_bookkeeping_only_pair_is_not_an_increment(self):
        row = [r for r in counterfactuals()['rows']
               if r['tag'] == 'C2_bookkeeping_only'][0]
        self.assertEqual(row['verdict'], 'IDENTICAL')
        self.assertFalse(row['is_an_increment'])

    def test_acceptance_never_comes_back(self):
        got = monotonicity_probe()
        self.assertTrue(got['seed_accepted'])
        self.assertEqual(got['seed_verdict'], 'SINGLE_AXIS')
        self.assertGreater(got['refusals'], 0, '探针必须至少触发一次拒绝，否则空转')
        self.assertFalse(got['acceptance_came_back'])

    def test_the_contract_is_stricter_than_the_old_guard_on_nothing_and_looser_on_one_thing(self):
        got = legacy_vs_contract()
        self.assertTrue(got['legacy_would_block_a_legitimate_increment'],
                        '旧守卫必须确实拒掉过合法增量，否则「它只有 0/非 0 两档」是无据之说')
        self.assertTrue(got['both_refuse_the_polluted_one'])


class AxisCoverageTests(unittest.TestCase):
    def test_the_declared_axes_cover_every_production_design_field(self):
        got = axis_coverage()
        self.assertTrue(got['covers_every_production_design_field'])
        self.assertTrue(got['no_field_declared_twice'])
        self.assertEqual(got['production_design_field_count'],
                         got['declared_axis_member_count'])


# ------------------------------------------------------------------ 运行

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tests', action='store_true')
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args(argv)

    if args.tests:
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        for cls in (CensusTests, HistoricalRevisionTests, CounterfactualTests,
                    AxisCoverageTests):
            suite.addTests(loader.loadTestsFromTestCase(cls))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    if not args.run:
        ap.error('pass --tests or --run')

    read_only_before = {str(p): sha256_file(p) for p in protocol_paths()}

    cen = census()
    hist = recheck_the_recorded_revision()
    cf = counterfactuals()
    mono = monotonicity_probe()
    legacy = legacy_vs_contract()
    axes = axis_coverage()

    # --- fail-closed 断言：读数不成立就不要写出来
    assert not cf['vacuous'], '反事实矩阵全accept或全refuse ⇒ 空转'
    bad = [r['tag'] for r in cf['rows'] if not r['matches_the_expectation']]
    assert not bad, '反事实与预期不符: %s' % bad
    assert hist['claim_is_reverified'], '历史修订声明复核失败'
    assert hist['digest_scope']['recorded_digest_equals_the_file_sha256']
    assert not hist['digest_scope']['recorded_digest_equals_the_content_sha256']
    assert mono['seed_accepted'] and mono['refusals'] > 0
    assert not mono['acceptance_came_back'], '接受了翻回被拒 ⇒ 单向下锁被破坏'
    assert axes['covers_every_production_design_field']
    assert cen['protocol_count'] == (len(cen['with_a_revision_channel'])
                                    + len(cen['without_a_revision_channel']))
    assert cen['without_a_revision_channel'], '必须报出没有修订通道的协议名单'
    assert legacy['legacy_would_block_a_legitimate_increment']
    assert legacy['both_refuse_the_polluted_one']
    after = {str(p): sha256_file(p) for p in protocol_paths()}
    assert after == read_only_before, '本批改动了封存协议'

    stamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S+08:00')
    payload = {
        'batch': 'incremental_contract_20260925',
        'generated_at': stamp,
        'purpose': 'P6：把「一个增量对照相对基准只允许改一个维度」从靠自觉变成可机检的契约',
        'discipline': DISCIPLINE,
        'module': 'research/factor_alchemy/incremental_contract.py',
        'module_sha256': sha256_file(ROOT / 'research/factor_alchemy/incremental_contract.py'),
        'axis_coverage': axes,
        'protocol_census': {
            'protocol_count': cen['protocol_count'],
            'with_a_revision_channel_count': len(cen['with_a_revision_channel']),
            'with_a_revision_channel': cen['with_a_revision_channel'],
            'without_a_revision_channel_count': len(cen['without_a_revision_channel']),
            'without_a_revision_channel': cen['without_a_revision_channel'],
            'rows': cen['rows'],
        },
        'historical_revision_reverified': hist,
        'counterfactuals': cf,
        'monotonicity': mono,
        'legacy_versus_contract': legacy,
        'read_only_guarantee': {
            'protocol_files_unchanged': True,
            'before': read_only_before,
            'after': after,
        },
        'claims_not_made': [
            '本批不跑任何实验、不读任何标签、不产生任何 IC 或收益读数。',
            '契约只对「两份已冻结协议的字典」做形状判定；它不判断某个维度该不该改，也不判断改动是否合理。',
            '契约不阻止任何人改两个维度 —— 它只是让这种改法**不能被报成单维增量**。',
            '本批没有给其余 13 份协议装上修订通道；普查只把「没有通道」这件事量化并列出名单。',
            '「一次只允许触碰一个特征」是**契约设定的阈值**（默认 1），不是从数据里估出来的；放宽阈值需要显式写下来。',
            '本批对盈亏平衡 IC 缺口（+0.108 vs 实测 +0.054/+0.063）没有直接贡献。',
            'model_compare 的 protocol.json 一字未改；历史协议与历史摘要一律不改写。',
        ],
    }
    size = write_json(OUT / 'readings.json', payload)
    write_csv(OUT / 'counterfactuals.csv',
              ['tag', 'intent', 'expected', 'verdict', 'accepted', 'refuses',
               'axes_changed', 'features_touched', 'changed_design_fields', 'flags'],
              [[r['tag'], r['intent'], r['expected'], r['verdict'], r['accepted'],
                r['refuses'], '|'.join(r['axes_changed']),
                r['feature_delta']['touched'],
                '|'.join(r['changed_design_fields']), '|'.join(r['flags'])]
               for r in cf['rows']])
    write_csv(OUT / 'protocol_census.csv',
              ['path', 'keys', 'design', 'bookkeeping', 'has_revision_channel',
               'file_sha256', 'canonical_sha256'],
              [[r['path'], r['keys'], len(r['design_keys']), len(r['bookkeeping_keys']),
                r['has_a_revision_channel'], r['file_sha256'], r['canonical_sha256']]
               for r in cen['rows']])

    print('protocols %d  with a revision channel %d  without %d'
          % (cen['protocol_count'], len(cen['with_a_revision_channel']),
             len(cen['without_a_revision_channel'])))
    print('counterfactuals %d  accepted %d  refused %d'
          % (len(cf['rows']), sum(1 for r in cf['rows'] if r['accepted']),
             sum(1 for r in cf['rows'] if r['refuses'])))
    for row in cf['rows']:
        print('  %-28s %-16s %s' % (row['tag'], row['verdict'],
                                    'accept' if row['accepted'] else 'REFUSE'))
    print('monotonicity probes %d refusals %d came_back %s'
          % (mono['probes'], mono['refusals'], mono['acceptance_came_back']))
    print('readings.json %d bytes' % size)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
