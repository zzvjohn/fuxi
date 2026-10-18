"""Evidence-based readiness separates software checks from external release authority."""
from .contracts import require, ContractError


def assess(store,software_checks,*,research_receipt=None,parity_receipt=None,
           forward_receipt=None,activation_receipt=None):
    require(software_checks and all(type(v) is bool for v in software_checks.values()),'nonempty explicit software verdicts required')
    blockers=['software:'+k for k,v in software_checks.items() if not v]
    receipts={'research_acceptance':research_receipt,'jq_parity':parity_receipt,
              'forward_acceptance':forward_receipt,'paper_activation':activation_receipt}
    bodies={}
    for kind,identity in receipts.items():
        if not identity:
            blockers.append(kind+'_missing');continue
        try:bodies[kind]=store.consume_receipt(identity,kind)
        except ContractError:blockers.append(kind+'_inactive')
    if 'jq_parity' in bodies:
        p=bodies['jq_parity']
        if not(p.get('observed_on_jq') is True and p.get('accepted') is True and p.get('n_pairs',0)>0):
            blockers.append('actual_JQ_parity_not_established')
    hashes={b.get('ir_hash') for b in bodies.values() if b.get('ir_hash')}
    if len(hashes)>1:blockers.append('release_IR_mismatch')
    return {'software_checks':software_checks,'software_ready':all(software_checks.values()),
            'paper_release_ready':not blockers,'blockers':blockers,
            'live_execution_allowed':False,'live_readiness_blockers':['live_backend_not_implemented',
                'broker_account_integration_not_observed','live_authorization_missing'],
            'statement':'Ten diagnostic rounds establish software behavior; they do not replace external or forward evidence.'}
