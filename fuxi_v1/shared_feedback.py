"""Shared, immutable descriptive feedback with live source invalidation.

Importers run under the common controller's writer lock.  Read projections never
mint rewards, receipts or legacy economic rejections.  Native memory paths are
consumed by campaign.run_campaign through its independent memory_sources input.
"""
from pathlib import Path
import json
import sqlite3

from .campaign import write_new
from .contracts import digest, load_json, load_fact, require
from .datasets import file_hash
from .learning import compatibility, export_memory, read_source, source_usable


STATUS = frozenset(('valid', 'data_insufficient', 'error', 'not_evaluated', 'precheck_rejected'))


def feedback_path(workspace):
    return Path(workspace).resolve() / 'data' / 'fuxi_unified' / 'feedback.sqlite'


def _source(report_path):
    path = Path(report_path).resolve()
    require(path.is_file(), 'native feedback report missing')
    root = path.parent
    report = load_json(path)
    config, payload, policy = [load_json(root / name) for name in ('config.json', 'input.json', 'policy.json')]
    require(report.get('campaign_id') == config.get('campaign_id'), 'native feedback campaign mismatch')
    require(digest(payload) == config['input_hash'] == report['input_hash'], 'native feedback input mismatch')
    require(policy == report['policy'] and policy['policy_hash'] ==
            digest({k: v for k, v in policy.items() if k != 'policy_hash'}), 'native feedback policy mismatch')
    key = compatibility(payload, policy['request'])
    require(config.get('learning_compatibility') == key, 'native feedback compatibility mismatch')
    database = root / 'evidence.sqlite'
    with read_source(database) as db:
        configs = db.execute('SELECT config FROM campaigns').fetchall()
        require(len(configs) == 1 and json.loads(configs[0][0]) == config, 'native feedback config fact mismatch')
        reports = db.execute("SELECT payload,hash FROM facts WHERE kind='campaign_report'").fetchall()
        require(any(row['hash'] == digest(report) and load_fact(row['payload']) == report for row in reports),
                'native report lacks immutable source fact')
    manifest = export_memory(database, key)
    require(manifest['records'], 'native report contains no currently usable feedback')
    binding = {'report_path': str(path), 'report_sha256': file_hash(path), 'report_hash': digest(report),
        'config_path': str(root / 'config.json'), 'config_sha256': file_hash(root / 'config.json'),
        'input_path': str(root / 'input.json'), 'input_sha256': file_hash(root / 'input.json'),
        'policy_path': str(root / 'policy.json'), 'policy_sha256': file_hash(root / 'policy.json'),
        'input_hash': config['input_hash'], 'policy_hash': policy['policy_hash'], 'compatibility_hash': key,
        'snapshot_hash': digest(payload['snapshot']), 'label_hash': digest(payload['labels']),
        'window': [policy['request']['research_start'], policy['request']['research_end']],
        'cost_profile_hash': digest(payload['market']['execution']), 'campaign_id': config['campaign_id']}
    return report, manifest, binding


def project_feedback(card, binding, token):
    """Projection is descriptive; nonvalid stages never carry formal metrics."""
    require(card.get('schema_version') == 'fuxi.feedback.2', 'native feedback_v2 required')
    require(card['policy_hash'] == binding['policy_hash'] and card['window'] == binding['window']
            and card['snapshot_hash'] == binding['snapshot_hash'],
            'feedback window/policy differs from source report')
    require(card.get('qualification_transfer_allowed') is False, 'source permits forbidden qualification transfer')
    stages = {}
    for observation in card['observations']:
        state = observation['status']
        require(state in STATUS, 'unknown native feedback five-state value')
        require(state == 'valid' or observation['decision'] == 'not_applicable', 'nonvalid feedback decision')
        stages[observation['stage']] = {'eval_status': state, 'decision': observation['decision'],
            'reason_code': observation['reason'], 'causes': observation['causes'], 'metrics': None}
    e1 = stages.get('E1', {'eval_status': 'not_evaluated'})
    e3 = stages.get('E3', {'eval_status': 'not_evaluated'})
    return {'schema_version': 'fuxi.shared_feedback.1', 'source_system': 'native_v1',
        'candidate_id': card['candidate_id'], 'formula_hash': card['candidate']['formula_hash'],
        'candidate': card['candidate'], 'compatibility_hash': binding['compatibility_hash'],
        'source_binding': binding, 'source_token': token, 'stage_states': stages,
        'descriptive_rank_ic': card.get('mean_rank_ic') if e1['eval_status'] == 'valid' else None,
        'descriptive_net_increment': card.get('descriptive_net_increment') if e3['eval_status'] == 'valid' else None,
        'parent_comparisons': card.get('parent_comparisons', []), 'gap_counts': card.get('gap_counts', []),
        'valid_sections': card.get('valid_sections', 0), 'n_pairs': card.get('n_pairs', 0),
        'source_economic_reward_eligible': bool(card.get('economic_reward_eligible')),
        'economic_reward': None, 'economic_reward_eligible': False,
        'qualification_receipt_id': None, 'qualification_transfer_allowed': False,
        'interpretation': 'same-caliber research description; no statistical/economic/qualification transfer',
        'legacy_current_evaluation_status': 'not_evaluated', 'live_execution_allowed': False}


def _open_writer(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS facts(id TEXT PRIMARY KEY,payload TEXT NOT NULL,hash TEXT NOT NULL);
      CREATE TRIGGER IF NOT EXISTS facts_no_update BEFORE UPDATE ON facts
        BEGIN SELECT RAISE(ABORT,'immutable feedback'); END;
      CREATE TRIGGER IF NOT EXISTS facts_no_delete BEFORE DELETE ON facts
        BEGIN SELECT RAISE(ABORT,'immutable feedback'); END;
    ''')
    return db


def import_feedback(report_path, output, workspace):
    """Append source feedback; the persisted summary describes first-import readers.

    A retry revalidates its own source and facts, then returns that first snapshot.
    query_feedback/native_context remain the consumers of current library state.
    """
    report, manifest, binding = _source(report_path)
    rows = [project_feedback(record['card'], binding, record['token']) for record in manifest['records']]
    require(rows, 'feedback import has zero records')
    destination = Path(output).resolve()
    if destination.suffix.lower() != '.json':
        destination = destination / 'IMPORT.json'
    fact_ids = [digest(row) for row in rows]
    if destination.exists():
        previous = load_json(destination)
        require(previous.get('schema_version') == 'fuxi.shared_feedback_import.1'
                and previous.get('summary_hash') == digest({k: v for k, v in previous.items() if k != 'summary_hash'}),
                'feedback import summary hash/schema changed')
        require(previous.get('native_report_hash') == digest(report) and previous.get('source') == binding
                and previous.get('feedback_db') == str(feedback_path(workspace))
                and previous.get('fact_ids') == fact_ids and previous.get('record_count') == len(fact_ids),
                'feedback import source/fact identity changed')
        require(previous.get('economic_rewards_imported') == 0 and previous.get('qualification_receipts_imported') == 0
                and previous.get('live_execution_allowed') is False, 'feedback import forbidden transfer')
        path = feedback_path(workspace)
        require(path.is_file(), 'persisted shared feedback database missing')
        db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        try:
            for identity, expected in zip(fact_ids, rows):
                fact = db.execute('SELECT payload,hash FROM facts WHERE id=?', (identity,)).fetchone()
                require(fact is not None and fact[1] == identity and digest(json.loads(fact[0])) == identity
                        and json.loads(fact[0]) == expected, 'persisted shared feedback fact changed')
                require(_active(expected), 'persisted shared feedback source revoked')
        finally:
            db.close()
        return previous
    db = _open_writer(feedback_path(workspace))
    try:
        with db:
            for identity, row in zip(fact_ids, rows):
                body = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
                old = db.execute('SELECT payload,hash FROM facts WHERE id=?', (identity,)).fetchone()
                require(old is None or (old[1] == identity and json.loads(old[0]) == row), 'feedback identity collision')
                if old is None:
                    db.execute('INSERT INTO facts VALUES(?,?,?)', (identity, body, identity))
    finally:
        db.close()
    # A controller-visible true read consumer is exercised after import.  These
    # are live-checked pointers, not a copied native memory/reward database.
    native = native_context(workspace, binding['compatibility_hash'])
    legacy = legacy_context(workspace, binding['compatibility_hash'])
    summary = {'schema_version': 'fuxi.shared_feedback_import.1', 'native_report_hash': digest(report),
        'source': binding, 'feedback_db': str(feedback_path(workspace)), 'fact_ids': fact_ids,
        'record_count': len(fact_ids), 'idempotent_content_identity': True,
        'read_consumers': {'native_context': len(native['cards']), 'legacy_context': len(legacy)},
        'native_memory_sources': native['memory_sources'], 'economic_rewards_imported': 0,
        'qualification_receipts_imported': 0, 'live_execution_allowed': False}
    summary['summary_hash'] = digest(summary)
    write_new(destination, summary)
    return summary


def _active(row):
    source = row['source_binding']
    for prefix in ('report', 'config', 'input', 'policy'):
        require(file_hash(source[prefix + '_path']) == source[prefix + '_sha256'], 'shared feedback source file drift')
    token = row['source_token']
    with read_source(token['source_path']) as db:
        original = db.execute('SELECT payload,hash FROM facts WHERE id=?', (token['source_event'],)).fetchone()
        if original is None or original['hash'] != token['source_hash']:
            return False
        require(digest(load_fact(original['payload'])) == original['hash'], 'shared feedback original fact changed')
        return source_usable(db, token)


def query_feedback(workspace, compatibility_hash=None):
    """Read valid source facts only; missing/revoked evidence never supplies memory."""
    path = feedback_path(workspace)
    if not path.exists():
        return []
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        rows = []
        for identity, body, expected in db.execute('SELECT id,payload,hash FROM facts ORDER BY id'):
            row = json.loads(body)
            require(identity == expected == digest(row), 'shared feedback content changed')
            if compatibility_hash is not None and row['compatibility_hash'] != compatibility_hash:
                continue
            if _active(row):
                rows.append({'fact_id': identity, **row})
        return rows
    finally:
        db.close()


def native_context(workspace, compatibility_hash):
    """Supply genuine native memory sources to campaign's export_memory consumer.

    Call run_campaign(..., memory_sources=result['memory_sources']).  The common
    controller freezes path/manifest_hash descriptors before calling the campaign.
    The campaign rechecks compatibility, source facts, revisions and dependencies.
    Its scheduler/search/feedback consumers then use the validated imported cards.
    """
    require(isinstance(compatibility_hash, str) and bool(compatibility_hash), 'explicit native feedback compatibility required')
    rows = query_feedback(workspace, compatibility_hash)
    paths = sorted({row['source_token']['source_path'] for row in rows})
    manifests = [export_memory(path, compatibility_hash) for path in paths]
    require(all(m['compatibility'] == compatibility_hash for m in manifests), 'native memory projection incompatible')
    return {'compatibility_hash': compatibility_hash, 'cards': rows, 'memory_sources': paths,
        'exported_native_records': sum(len(m['records']) for m in manifests),
        'consumer': 'campaign.run_campaign memory_sources -> learning.export_memory/valid_imports',
        'reuse_rejected_reason': None if rows else 'no_same_caliber_valid_feedback',
        'rewards_and_receipts_transferred': False, 'live_execution_allowed': False}


def legacy_context(workspace, compatibility_hash=None):
    """Describe native observations to legacy hypothesis generation without verdict mapping."""
    rows = query_feedback(workspace, compatibility_hash)
    return [{'fact_id': row['fact_id'], 'candidate_id': row['candidate_id'],
        'hypothesis': row['candidate']['hypothesis'], 'canonical_ast': row['candidate']['canonical_ast'],
        'native_stage_states': row['stage_states'], 'descriptive_rank_ic': row['descriptive_rank_ic'],
        'window': row['source_binding']['window'], 'cost_profile_hash': row['source_binding']['cost_profile_hash'],
        'statuses': {'s1': 'not_evaluated', 's4': 'not_evaluated', 's5': 'not_evaluated', 's6': 'not_evaluated'},
        'proxy_icir': None, 'economic_reward': None, 'economic_reward_status': 'not_evaluated',
        'current_qualification': 'unknown', 'qualification_receipt_id': None,
        'usage': 'descriptive hypothesis context; native IC is not legacy ICIR; no S5 verdict/reward inheritance'}
        for row in rows]
