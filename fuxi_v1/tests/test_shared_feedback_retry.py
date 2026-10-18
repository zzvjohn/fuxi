"""An import summary is an immutable first-import snapshot, not live library size."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import unittest

from fuxi_v1.campaign import write_new
from fuxi_v1.contracts import ContractError, load_json
from fuxi_v1.evidence import EvidenceStore
from fuxi_v1.learning import read_source
from fuxi_v1.search import factor_spec, node
from fuxi_v1.shared_feedback import feedback_path, import_feedback, native_context, query_feedback
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.tests.test_shared_feedback import source_fixture


def two_sources(root):
    """Use the existing authentic SQLite fixture, then append a distinct second source."""
    first_root = root / 'first'
    first_root.mkdir()
    first = source_fixture(first_root)
    target = root / 'second' / 'native'
    target.mkdir(parents=True)
    payload = load_json(first['path'].parent / 'input.json')
    policy = load_json(first['path'].parent / 'policy.json')
    config = load_json(first['path'].parent / 'config.json')
    config['campaign_id'] = 'campaign_second'
    report = load_json(first['path'])
    report['campaign_id'] = config['campaign_id']
    with read_source(first['database']) as db:
        card = json.loads(db.execute("SELECT payload FROM facts WHERE kind='feedback_v2'").fetchone()[0])
    candidate = factor_spec(node('rank', node('mean', {'field': 'close'}, window=19)),
                            payload['snapshot']['fields'], 'distinct_second_fixture')
    card.update(candidate=candidate, candidate_id=candidate['candidate_id'], state_id='source_state_second')
    store = EvidenceStore(target / 'evidence.sqlite')
    slot = ['second', candidate['candidate_id'], 'capability', policy['policy_hash'], 'baseline']
    try:
        store.campaign(config['campaign_id'], config)
        evaluation_id = store.append('evaluation', {'test_only': True, 'source': 'second'})
        reward = store.reward(slot, 'capability', 1., evaluation_id)
        store.register_consumer(card['state_id'], [reward, evaluation_id], 'rebuild_second_fixture')
        card['evaluation_id'] = evaluation_id
        store.append('feedback_v2', card)
        store.append('campaign_report', report)
    finally:
        store.close()
    for name, value in [('input.json', payload), ('config.json', config), ('policy.json', policy), ('report.json', report)]:
        write_new(target / name, value)
    second = {'path': target / 'report.json', 'database': target / 'evidence.sqlite', 'key': first['key'],
              'slot': slot, 'reward': reward, 'evaluation_id': evaluation_id, 'candidate_id': candidate['candidate_id']}
    return first, second


def count_facts(root):
    db = sqlite3.connect(feedback_path(root))
    try:
        return db.execute('SELECT COUNT(*) FROM facts').fetchone()[0]
    finally:
        db.close()


class SharedFeedbackRetryTests(unittest.TestCase):
    def test_retry_keeps_first_snapshot_after_other_compatible_source(self):
        with scratch_directory() as directory:
            root = Path(directory)
            first, second = two_sources(root)
            self.assertEqual(first['key'], second['key'])
            self.assertNotEqual(first['candidate_id'], second['candidate_id'])
            output = root / 'first_import.json'
            original = import_feedback(first['path'], output, root)
            original_bytes = output.read_bytes()
            self.assertEqual(original['read_consumers'], {'native_context': 1, 'legacy_context': 1})
            later = import_feedback(second['path'], root / 'second_import.json', root)
            self.assertEqual(later['read_consumers'], {'native_context': 2, 'legacy_context': 2})
            retry = import_feedback(first['path'], output, root)
            self.assertEqual(retry, original)
            self.assertEqual(output.read_bytes(), original_bytes)
            self.assertEqual(count_facts(root), 2)
            self.assertEqual(len(query_feedback(root, first['key'])), 2)
            current = native_context(root, first['key'])
            self.assertEqual(len(current['cards']), 2)
            self.assertEqual(len(current['memory_sources']), 2)
            self.assertEqual(retry['native_memory_sources'], [str(first['database'].resolve())])

    def test_retry_refuses_revoked_source_after_other_source_import(self):
        with scratch_directory() as directory:
            root = Path(directory)
            first, second = two_sources(root)
            output = root / 'first_import.json'
            import_feedback(first['path'], output, root)
            original_bytes = output.read_bytes()
            import_feedback(second['path'], root / 'second_import.json', root)
            store = EvidenceStore(first['database'])
            try:
                store.reward(first['slot'], 'capability', None, first['evaluation_id'], supersedes=first['reward'])
            finally:
                store.close()
            with self.assertRaises(ContractError):
                import_feedback(first['path'], output, root)
            self.assertEqual(output.read_bytes(), original_bytes)
            self.assertEqual(count_facts(root), 2)
            self.assertEqual([r['candidate_id'] for r in query_feedback(root, first['key'])], [second['candidate_id']])

    def test_retry_refuses_changed_source_without_rewriting_snapshot(self):
        with scratch_directory() as directory:
            root = Path(directory)
            first, second = two_sources(root)
            output = root / 'first_import.json'
            import_feedback(first['path'], output, root)
            original_bytes = output.read_bytes()
            import_feedback(second['path'], root / 'second_import.json', root)
            path = first['path'].parent / 'input.json'
            payload = load_json(path)
            payload['market']['execution']['slippage_bps'] = '0'
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ContractError, 'input mismatch'):
                import_feedback(first['path'], output, root)
            self.assertEqual(output.read_bytes(), original_bytes)
            self.assertEqual(count_facts(root), 2)

    def test_retry_refuses_tampered_persistent_summary(self):
        with scratch_directory() as directory:
            root = Path(directory)
            first, _ = two_sources(root)
            output = root / 'first_import.json'
            original = import_feedback(first['path'], output, root)
            changed = deepcopy(original)
            changed['read_consumers']['native_context'] = 999
            output.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaises(ContractError):
                import_feedback(first['path'], output, root)
            self.assertEqual(count_facts(root), 1)


if __name__ == '__main__':
    unittest.main()
