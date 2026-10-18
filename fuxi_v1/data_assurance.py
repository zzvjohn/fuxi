"""Import independently reviewed data evidence; derived declarations are insufficient."""
from .contracts import load_json, require, digest, timestamp, MARKET_CONTENT_EXCLUSIONS
from .certification import certify, derived_data_receipt, account_scope
from .pit_universe import read_lifecycle_file

CLAIMS=('historical_availability_and_revisions','lifecycle_and_selection',
        'actions_marks_and_carried_holdings','execution_rules_and_costs')


def import_assurance(store,spec,review):
    for key in ('snapshot_path','lifecycle_path','assurance_path'):
        require(spec.get(key) in review['evidence_files'],key+' must be independently reviewed')
    payload=load_json(spec['snapshot_path']);assurance=load_json(spec['assurance_path'])
    require(assurance.get('schema_version')=='fuxi.independent_data_assurance.1','unsupported independent assurance schema')
    require(assurance.get('reviewer')==review['reviewer'],'assurance/review identity mismatch')
    require(assurance.get('generator_identity') and assurance['generator_identity'] != review['reviewer'],
            'generator cannot certify its own investment data')
    timestamp(assurance['reviewed_at'])
    require(assurance['reviewed_at']==review['reviewed_at'],'assurance/review clock mismatch')
    require(assurance.get('snapshot_hash')==digest(payload['snapshot']),'assurance snapshot mismatch')
    require(assurance.get('market_digest')==digest({k:v for k,v in payload['market'].items()
                                                    if k not in MARKET_CONTENT_EXCLUSIONS}),
            'assurance market mismatch')
    require(assurance.get('account_scope')=='ever_eligible_holdings_and_entitlements',
            'assurance must cover carried holdings, not just current selection')
    life=read_lifecycle_file(spec['lifecycle_path'])
    certificate=certify(payload,declaration=assurance['declaration'],lifecycle=life)
    require(not certificate['blocking_gaps'],'independent review cannot erase derived data contradictions')
    require(assurance.get('claims') and set(assurance['claims'])==set(CLAIMS),'complete independent claim set required')
    for name in CLAIMS:
        claim=assurance['claims'][name]
        require(claim.get('decision')=='accepted' and claim.get('limitations')==[],name+' has unresolved limits')
        files=claim.get('primary_evidence_files') or {}
        require(files and all(review['evidence_files'].get(k)==v for k,v in files.items()),
                name+' requires reviewed primary evidence hashes')
        require(not set(files)&{spec['snapshot_path'],spec['lifecycle_path'],spec['assurance_path']},
                'derived snapshot/declaration is not independent primary evidence')
    required_features = 0
    required_holdings = 0
    if payload['snapshot'].get('schema_version') == 'fuxi.columnar_snapshot.1':
        # P-20261015-006: columnar coverage counts come from the hash-bound arrays,
        # not the row-form ``snapshot['columns']`` matrix. NaN is the storage null.
        from .columnar import ColumnarSnapshot
        columnar_snap = ColumnarSnapshot(payload['snapshot'])
        import numpy as np
        required_features = sum(int(np.isfinite(array).sum()) for array in columnar_snap.columns.values())
        required_holdings = int(columnar_snap.membership.sum())
        columnar_snap.close()
    else:
        required_features = sum(v is not None for matrix in payload['snapshot']['columns'].values() for row in matrix for v in row)
        required_holdings = sum(map(len, account_scope(payload['snapshot'], life).values()))
    coverage=assurance.get('coverage') or {}
    require(required_features>0 and coverage.get('availability_revision_cells')==required_features,
            'availability/revision assurance coverage incomplete')
    require(required_holdings>0 and coverage.get('potential_holding_cells')==required_holdings,
            'carry/mark/action assurance coverage incomplete')
    body={**derived_data_receipt(certificate),**review,'schema_version':'fuxi.independent_data_assurance.1',
          'investment_eligibility_certified':True,'assurance_hash':digest(assurance),
          'claim_scope':{'kind':'independently_reviewed_investment_data','account_scope':assurance['account_scope'],
                         'claims':list(CLAIMS),'investment_eligibility_certified':True},
          'assurance':assurance,'trust_boundary':'independent_review_of_primary_evidence_not_automatic_proof'}
    return {'receipt_id':store.receipt('data_contract',body),'live_execution_allowed':False}


def bind_assurance(payload,store,receipt_id):
    """Bind a reviewed receipt without changing prices, selection or source hashes."""
    from copy import deepcopy
    body=store.consume_receipt(receipt_id,'data_contract')
    require(body.get('schema_version')=='fuxi.independent_data_assurance.1' and body.get('investment_eligibility_certified') is True,
            'independent investment assurance required')
    require(body['snapshot_hash']==digest(payload['snapshot']),'review snapshot changed')
    require(body['assurance']['market_digest']==digest({k:v for k,v in payload['market'].items()
                                                       if k not in MARKET_CONTENT_EXCLUSIONS}),
            'review market changed')
    require(payload['market'].get('event_coverage_certified') is True and payload['market'].get('tradability_certified') is True,
            'reviewed build must establish event/tradability facts; binder cannot rewrite false facts')
    result=deepcopy(payload)
    result['market']['investment_assurance']={'schema_version':'fuxi.market_assurance.1',
          'snapshot_hash':body['snapshot_hash'],'market_digest':body['assurance']['market_digest'],
          'receipt_id':receipt_id,'assurance_hash':body['assurance_hash']}
    return result
