"""Content-addressed research cards and a read-only index; qualification is receipt-derived."""
from pathlib import Path
from .contracts import digest, load_json, require


def card(candidate, result, scope, protocol, *, admission=None, source_path=None, training=None):
    stages = result.get('evaluations', [result] if result.get('stage') else [])
    admission = admission or {}
    body = {'schema_version':'fuxi.factor_card.1', 'candidate_id':candidate['candidate_id'],
            'formula_hash':candidate['formula_hash'], 'catalog_hash':candidate['catalog_hash'],
            'canonical_ast':candidate['canonical_ast'], 'hypothesis':candidate['hypothesis'],
            'parent_ids':candidate['parent_ids'], 'mutation_patch':candidate['mutation_patch'],
            'required_fields':candidate['required_fields'], 'decision_time_rule':candidate['decision_time_rule'],
            'missing_policy':candidate['missing_policy'], 'scope':scope, 'protocol_hash':digest(protocol),
            'stage_states':{r['stage']:{k:r[k] for k in ('eval_status','decision','reason_code','n_pairs','economic_reward_eligible')} for r in stages},
            'source_path':source_path, 'exposure':'research_exposed' if scope.get('exposure') != 'synthetic' else 'synthetic',
            'library':'research_components', 'jq_candidate_allowed':False,
            'historical_declared_jq_candidate_allowed':admission.get('jq_candidate_allowed', False),
            'qualification_receipt_id':admission.get('receipt_id'), 'live_execution_allowed':False,
            'revocation_status':'not_qualified' if not admission.get('receipt_id') else 'must_check_authority_at_use',
            'trial_count':protocol.get('max_trials'), 'investment_claim':False}
    body['training_direction'] = training
    body['card_hash'] = digest(body)
    return body


def query(root, *, candidate=None, status=None, start=None, end=None, protocol_hash=None, version=None):
    """No cache writes or stale receipt projection: inspect current immutable card bytes."""
    result = []
    for path in sorted(Path(root).glob('cards/*.json')):
        row = load_json(path)
        require(row['card_hash'] == digest({k:v for k,v in row.items() if k != 'card_hash'}), 'card hash changed')
        if candidate and row['candidate_id'] != candidate:
            continue
        if status and not any(r['eval_status'] == status for r in row['stage_states'].values()):
            continue
        window = row['scope']['window']
        if start and window['end'] < start or end and window['start'] > end:
            continue
        if protocol_hash and row['protocol_hash'] != protocol_hash:
            continue
        if version and row['scope'].get('system_version') != version:
            continue
        result.append({'path':str(path.resolve()), **row})
    return result


def legacy_projection(rows, source_hash):
    """Legacy metadata is an index entry; booleans never mint independent receipts."""
    return [{'candidate_id':r.get('factor_id',r.get('id')), 'library':'legacy_factor_metadata',
             'source_hash':source_hash, 'historical_declared_status':r.get('status'),
             'qualification_receipt_id':None, 'current_qualification':'unknown_requires_structured_authority',
             'live_execution_allowed':False} for r in rows]
