"""Source-bound shared feedback, read consumers and native dependency invalidation."""
from copy import deepcopy
from pathlib import Path
import sqlite3
import unittest

from fuxi_v1.campaign import run_campaign, write_new, _memory_source_contract
from fuxi_v1.contracts import ContractError, digest, load_json
from fuxi_v1.evidence import EvidenceStore
from fuxi_v1.data import Snapshot
from fuxi_v1.learning import compatibility, export_memory, valid_imports, refresh_import_dependencies, feedback_dependencies
from fuxi_v1.policy import calibration_request
from fuxi_v1.search import factor_spec, node
from fuxi_v1.shared_feedback import import_feedback, query_feedback, native_context, legacy_context, feedback_path
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.tests.test_kernel import scratch_directory


def source_fixture(root, *, e1='valid'):
    source = root / 'native'
    source.mkdir()
    payload = synthetic_campaign()
    snapshot = Snapshot(payload['snapshot'])
    horizon = payload['labels']['label_spec']['horizon_sessions']
    cal_index = max(snapshot.start + 25, min(80, len(snapshot.sessions) // 2))
    request = calibration_request(snapshot, calibration_end=snapshot.sessions[cal_index],
        research_start=snapshot.sessions[cal_index + horizon + 2], max_trials=6, horizon=horizon)
    policy = {'request': request, 'scope': 'synthetic_contract_fixture'}
    policy['policy_hash'] = digest(policy)
    key = compatibility(payload, request)
    config = {'campaign_id': 'campaign_test', 'learning_compatibility': key,
              'input_hash': digest(payload), 'budget_units': 10}
    candidate = factor_spec(node('rank', node('mean', {'field': 'close'}, window=17)),
                            payload['snapshot']['fields'], 'fixture')
    report = {'campaign_id': config['campaign_id'], 'input_hash': config['input_hash'], 'policy': policy}
    card = {'schema_version': 'fuxi.feedback.2', 'candidate_id': candidate['candidate_id'], 'candidate': candidate,
        'state_id': 'source_state', 'snapshot_hash': digest(payload['snapshot']), 'policy_hash': policy['policy_hash'],
        'window': [request['research_start'], request['research_end']],
        'observations': [{'stage': 'E1', 'status': e1, 'decision': 'pass' if e1 == 'valid' else 'not_applicable',
                          'reason': 'test_only', 'causes': []},
                         {'stage': 'E3', 'status': 'data_insufficient', 'decision': 'not_applicable',
                          'reason': 'unknown_rights', 'causes': ['data_insufficient']}],
        'mean_rank_ic': .03, 'descriptive_net_increment': .25,
        'economic_reward_eligible': False, 'qualification_transfer_allowed': False,
        'parent_comparisons': [], 'valid_sections': 30, 'n_pairs': 180}
    store = EvidenceStore(source / 'evidence.sqlite')
    try:
        store.campaign(config['campaign_id'], config)
        evaluation_id = store.append('evaluation', {'test_only': True})
        slot = ['source', 'candidate', 'capability', 'window', 'baseline']
        reward = store.reward(slot, 'capability', 1., evaluation_id)
        store.register_consumer('source_state', [reward, evaluation_id], 'rebuild_fixture')
        card['evaluation_id'] = evaluation_id
        store.append('feedback_v2', card)
        store.append('campaign_report', report)
    finally:
        store.close()
    for name, value in [('input.json', payload), ('config.json', config), ('policy.json', policy), ('report.json', report)]:
        write_new(source / name, value)
    return {'path': source / 'report.json', 'database': source / 'evidence.sqlite', 'key': key,
            'slot': slot, 'reward': reward, 'evaluation_id': evaluation_id, 'candidate_id': candidate['candidate_id']}


class SharedFeedbackTests(unittest.TestCase):
    def test_independent_source_keyword_rejects_duplicate_unbound_mode_and_two_channels(self):
        with scratch_directory() as directory:
            root = Path(directory)
            source = source_fixture(root)
            path = str(source['database'].resolve())
            descriptor = {'path': path, 'manifest_hash': digest(export_memory(path, source['key']))}
            self.assertEqual(_memory_source_contract(()), [])
            self.assertEqual(_memory_source_contract([descriptor]), [descriptor])
            with self.assertRaisesRegex(ContractError, 'duplicate'):
                _memory_source_contract([descriptor, path])
            with self.assertRaisesRegex(ContractError, 'bound'):
                _memory_source_contract([descriptor] * 9)
            with self.assertRaisesRegex(ContractError, 'manifest hash'):
                _memory_source_contract([dict(descriptor, manifest_hash='changing-file')])
            with self.assertRaisesRegex(ContractError, 'no-memory'):
                run_campaign('not_read.json', root / 'no_memory', mode='no_memory', memory_sources=[descriptor])
            with self.assertRaisesRegex(ContractError, 'both'):
                run_campaign('not_read.json', root / 'two_channels', memory_sources=[descriptor],
                             study_protocol={'memory_sources': [path]})

    def test_selection_v2_campaign_consumes_shared_source_and_tracks_real_children(self):
        with scratch_directory() as directory:
            root = Path(directory)
            source = source_fixture(root)
            import_feedback(source['path'], root / 'import.json', root)
            context = native_context(root, source['key'])
            path = context['memory_sources'][0]
            manifest = export_memory(path, source['key'])
            descriptor = {'path': path, 'manifest_hash': digest(manifest)}
            report_path, report = run_campaign(source['path'].parent / 'input.json', root / 'consumer',
                rounds=3, candidates_per_round=2, seed=17, llm_enabled=False, model_algorithms=(),
                scheduler_policy='selection_v2', memory_sources=[descriptor], campaign_id='campaign_shared_child')
            config = load_json(report_path.parent / 'config.json')
            self.assertIsNone(config['study_protocol'])
            self.assertEqual(config['scheduler_policy'], 'selection_v2')
            self.assertEqual(config['learning_compatibility'], source['key'])
            self.assertEqual(config['memory_source_bindings'],
                [dict(descriptor, compatibility_hash=source['key'])])
            target = EvidenceStore(report_path.parent / 'evidence.sqlite')
            try:
                decisions = [r for _, r in target.facts('research_decision')]
                self.assertEqual([r['arm'] for r in decisions], ['fixed', 'random', 'refine'])
                self.assertEqual([r['imported_feedback_count'] for r in decisions], [1, 1, 1])
                self.assertTrue(all(source['candidate_id'] in r['parent_ids'] for r in decisions))
                children = [r for _, r in target.facts('candidate') if source['candidate_id'] in r['parent_ids']]
                self.assertGreater(len(children), 0)
                state = 'import_state_' + digest(manifest['records'][0]['token'])
                states = [r['state_id'] for _, r in target.facts('feedback_v2')
                          if r['candidate_id'] in {c['candidate_id'] for c in children}]
                self.assertEqual(len(states), len(children))
                for state_id in states:
                    dependencies = __import__('json').loads(target.db.execute(
                        'SELECT dependencies FROM consumers WHERE id=?', (state_id,)).fetchone()[0])
                    self.assertIn(state, dependencies)
                    self.assertTrue(target.state_usable(state_id))
                print({'native_shared_consumer': {'imported_per_round': [1, 1, 1],
                       'source_parent_children': len(children), 'actual_child_dependency_states': len(states),
                       'selection_v2_without_study_protocol': True,
                       'manifest_hash': descriptor['manifest_hash']}})
                producer = EvidenceStore(source['database'])
                try:
                    producer.reward(source['slot'], 'capability', None, source['evaluation_id'], supersedes=source['reward'])
                finally:
                    producer.close()
                self.assertEqual(refresh_import_dependencies(target, config['memory_imports']), [])
                self.assertTrue(all(not target.state_usable(state_id) for state_id in states))
            finally:
                target.close()

    def test_import_is_idempotent_and_context_does_not_grant_legacy_verdict(self):
        with scratch_directory() as directory:
            root = Path(directory)
            source = source_fixture(root)
            first = import_feedback(source['path'], root / 'import.json', root)
            second = import_feedback(source['path'], root / 'import.json', root)
            self.assertEqual(first, second)
            db = sqlite3.connect(feedback_path(root))
            try:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM facts').fetchone()[0], 1)
            finally:
                db.close()
            row = query_feedback(root, source['key'])[0]
            self.assertEqual(row['stage_states']['E3']['eval_status'], 'data_insufficient')
            self.assertIsNone(row['descriptive_net_increment'])
            self.assertFalse(row['economic_reward_eligible'])
            self.assertIsNone(row['qualification_receipt_id'])
            legacy = legacy_context(root, source['key'])[0]
            self.assertEqual(legacy['statuses']['s5'], 'not_evaluated')
            self.assertEqual(legacy['current_qualification'], 'unknown')
            self.assertIsNone(legacy['proxy_icir'])
            self.assertIsNone(legacy['economic_reward'])
            self.assertEqual(first['read_consumers'], {'native_context': 1, 'legacy_context': 1})

    def test_nonvalid_e1_cannot_export_reading_as_metric(self):
        with scratch_directory() as directory:
            root = Path(directory)
            source = source_fixture(root, e1='data_insufficient')
            import_feedback(source['path'], root / 'import.json', root)
            row = query_feedback(root, source['key'])[0]
            self.assertIsNone(row['descriptive_rank_ic'])
            self.assertTrue(all(s['metrics'] is None for s in row['stage_states'].values()))

    def test_true_native_dependency_consumer_and_source_revision_revocation(self):
        with scratch_directory() as directory:
            root = Path(directory)
            source = source_fixture(root)
            import_feedback(source['path'], root / 'import.json', root)
            context = native_context(root, source['key'])
            self.assertEqual(context['memory_sources'], [str(source['database'].resolve())])
            manifests = [export_memory(path, source['key']) for path in context['memory_sources']]
            target = EvidenceStore(root / 'native_consumer.sqlite')
            try:
                cards = refresh_import_dependencies(target, manifests)
                self.assertEqual(cards[0]['candidate_id'], source['candidate_id'])
                child = {'parent_ids': [source['candidate_id']]}
                dependencies = feedback_dependencies(child, [], manifests)
                self.assertTrue(dependencies)
                target.register_consumer('child', dependencies, 'actual_native_consumer_contract')
                producer = EvidenceStore(source['database'])
                try:
                    producer.reward(source['slot'], 'capability', None, source['evaluation_id'], supersedes=source['reward'])
                finally:
                    producer.close()
                self.assertEqual(query_feedback(root, source['key']), [])
                self.assertEqual(refresh_import_dependencies(target, manifests), [])
                self.assertFalse(target.state_usable('child'))
                with self.assertRaisesRegex(ContractError, 'no currently usable feedback'):
                    import_feedback(source['path'], root / 'revoked.json', root)
            finally:
                target.close()

    def test_incompatible_context_and_tampered_native_input_fail_closed(self):
        with scratch_directory() as directory:
            root = Path(directory)
            source = source_fixture(root)
            import_feedback(source['path'], root / 'import.json', root)
            self.assertEqual(query_feedback(root, 'different_window_or_cost'), [])
            self.assertEqual(native_context(root, 'different_window_or_cost')['memory_sources'], [])
            with self.assertRaisesRegex(ContractError, 'incompatible'):
                export_memory(source['database'], 'different_window_or_cost')
            payload = load_json(source['path'].parent / 'input.json')
            payload['market']['execution']['slippage_bps'] = '0'
            (source['path'].parent / 'input.json').write_text(__import__('json').dumps(payload), encoding='utf-8')
            with self.assertRaises(ContractError):
                query_feedback(root, source['key'])
            with self.assertRaisesRegex(ContractError, 'input mismatch'):
                import_feedback(source['path'], root / 'changed.json', root)


if __name__ == '__main__':
    unittest.main()
