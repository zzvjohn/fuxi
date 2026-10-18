"""Fusion contracts: causal prefix preservation and method-level paired support."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from fuxi_v1.artifacts import publish
from fuxi_v1.campaign import write_new
from fuxi_v1.contracts import ContractError, digest, load_json
from fuxi_v1.data import Snapshot, Labels
from fuxi_v1.datasets import validate_label_calendar
from fuxi_v1.fusion_study import derive_prefix, freeze, status, _checked, _method_pairs, _request
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.tests.test_kernel import scratch_directory


def _fixture():
    payload = synthetic_campaign(200)
    snapshot = payload['snapshot']
    snapshot['schema_version'] = 'fuxi.snapshot.2'
    snapshot['exposure'] = 'research_exposed'
    snapshot['provenance'] = {'original_barriers_hash': 'synthetic-null-mask-contract-fixture'}
    for row in snapshot['observations']:
        row.update(event_time=row['available_at'], published_at=row['available_at'],
                   ingested_at=row['available_at'], quality_status='research_limited', source_id='test_fixture')
    snapshot['observations'][0]['value'] = None
    payload['labels']['feature_snapshot_hash'] = digest(snapshot)
    return payload


class FusionPrefixTests(unittest.TestCase):
    def test_prefix_preserves_null_barrier_availability_and_label_horizon(self):
        original = _fixture()
        before = deepcopy(original)
        prefix = derive_prefix(original, 180)
        self.assertEqual(original, before)
        self.assertEqual(prefix['snapshot']['observations'][0], original['snapshot']['observations'][0])
        self.assertIsNone(prefix['snapshot']['observations'][0]['value'])
        self.assertEqual(prefix['snapshot']['provenance']['original_barriers_hash'],
                         original['snapshot']['provenance']['original_barriers_hash'])
        self.assertEqual(prefix['market']['execution'], original['market']['execution'])
        retained = {(r['session'], r['instrument']): r for r in original['labels']['rows']}
        self.assertTrue(prefix['labels']['rows'])
        self.assertTrue(all(r == retained[(r['session'], r['instrument'])] for r in prefix['labels']['rows']))
        snapshot = Snapshot(prefix['snapshot'])
        labels = Labels(prefix['labels'], snapshot)
        validate_label_calendar(labels.payload, snapshot.sessions)
        self.assertFalse(any(r['session'] == snapshot.sessions[-1] for r in labels.payload['rows']))

    def test_later_listing_identity_retained_without_earlier_membership(self):
        payload = _fixture()
        later = '688999.XSHG'
        for day in payload['snapshot']['sessions'][180:]:
            payload['snapshot']['universe'][day].append(later)
        payload['labels']['feature_snapshot_hash'] = digest(payload['snapshot'])
        prefix = derive_prefix(payload, 180)
        self.assertIn(later, prefix['snapshot']['provenance']['fusion_prefix']['original_declared_instruments'])
        self.assertFalse(any(later in members for members in prefix['snapshot']['universe'].values()))

    def test_declared_never_eligible_member_is_not_removed_from_identity(self):
        payload = _fixture()
        codes = list(payload['snapshot']['universe'].values())[0]
        payload['snapshot']['provenance']['original_cohort_declaration'] = {
            'members': [{'ts_code': code.replace('.XSHG', '.SH')} for code in codes] +
                       [{'ts_code': '688999.SH'}]}
        payload['labels']['feature_snapshot_hash'] = digest(payload['snapshot'])
        prefix = derive_prefix(payload, 180)
        declared = prefix['snapshot']['provenance']['fusion_prefix']['original_declared_instruments']
        self.assertEqual(len(declared), len(codes) + 1)
        self.assertIn('688999.XSHG', declared)
        self.assertFalse(any('688999.XSHG' in members for members in prefix['snapshot']['universe'].values()))

    def test_reject_unknown_axis_budget_and_protected_source(self):
        for request in ({'rounds': 6}, {'candidates_per_round': 3}, {'native_threshold': 0}, {'seed': -1}):
            with self.assertRaises(ContractError):
                _request(request)
        for exposure in ('synthetic', 'forward_sealed'):
            payload = _fixture()
            payload['snapshot']['exposure'] = exposure
            with self.assertRaises(ContractError):
                derive_prefix(payload, 180)

    def test_frozen_source_and_native_input_tampering_fail_closed(self):
        with scratch_directory() as directory:
            root = Path(directory)
            source = root / 'source.json'
            write_new(source, _fixture())
            protocol = freeze(source, root / 'study', {})
            self.assertEqual(protocol['budget']['candidate_limit_total'], 30)
            self.assertEqual(status(root / 'study')['state'], 'frozen_or_partial')
            changed = load_json(root / 'study' / 'INPUT.json')
            changed['market']['execution']['slippage_bps'] = '0'
            (root / 'study' / 'INPUT.json').write_bytes(__import__('json').dumps(changed).encode())
            with self.assertRaises(ContractError):
                _checked(root / 'study')


class FusionMethodPairTests(unittest.TestCase):
    def _reports(self, root, payload, *, missing=False):
        reports = {}
        codes = list(payload['snapshot']['universe'].values())[0]
        for arm, direction in (('fixed_memory', 1), ('adaptive_memory', -1)):
            scores = {day: {code: (None if missing else direction * i) for i, code in enumerate(codes)}
                      for day in payload['snapshot']['sessions']}
            bundle = publish(root / 'campaigns' / arm / 'bundles', {'scores.json': scores}, {'test': True})
            reports[arm] = {'candidate_results': [{'candidate_id': arm, 'round': 0,
                'bundle_path': str(bundle.relative_to(root / 'campaigns'))}]}
        return reports

    def test_method_pair_uses_common_cells_without_inflating_calendar(self):
        with scratch_directory() as directory:
            root = Path(directory)
            payload = derive_prefix(_fixture(), 180)
            write_new(root / 'INPUT.json', payload)
            reports = self._reports(root, payload)
            for arm in reports:
                reports[arm]['candidate_results'].append(dict(reports[arm]['candidate_results'][0], round=1))
            protocol = {'native_window': {'research_start': payload['snapshot']['sessions'][100],
                                        'research_end': payload['snapshot']['sessions'][-1]}}
            result = _method_pairs(root, protocol, reports, 'fixed_memory', 'adaptive_memory', 'scheduler')
            self.assertGreater(result['n_security_pairs_repeated_across_rounds'], 0)
            self.assertEqual(result['n_round_day_pairs'], 2 * result['distinct_calendar_sections'])
            self.assertFalse(result['independent_sample_count_asserted'])
            self.assertFalse(result['economic_reward_eligible'])
            self.assertFalse(result['winner_declared'])

    def test_zero_common_support_is_rejected(self):
        with scratch_directory() as directory:
            root = Path(directory)
            payload = derive_prefix(_fixture(), 180)
            write_new(root / 'INPUT.json', payload)
            reports = self._reports(root, payload, missing=True)
            protocol = {'native_window': {'research_start': payload['snapshot']['sessions'][100],
                                        'research_end': payload['snapshot']['sessions'][-1]}}
            with self.assertRaises(ContractError):
                _method_pairs(root, protocol, reports, 'fixed_memory', 'adaptive_memory', 'scheduler')


if __name__ == '__main__':
    unittest.main()
