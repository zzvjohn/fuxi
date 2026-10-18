"""Protocol feasibility is reported separately from economic rejection."""
from collections import Counter


def audit_reachability(payload, policy, receipt_present=False):
    request = policy['request']
    sessions = [d for d in payload['snapshot']['sessions'] if request['research_start'] <= d <= request['research_end']]
    tail_mass = (request['bootstrap_replicates']+1)*request['alpha']/request['max_trials']/2
    if getattr(payload['market'], "_columnar_market", False):
        import numpy as np
        market=payload['market'];mask=np.asarray(market.status['corporate_action_known']) != 1
        values,counts=np.unique(market.categorical['event_coverage'][mask],return_counts=True)
        gaps=Counter({market.category_values['event_coverage'][str(int(v))]:int(n) for v,n in zip(values,counts)})
    else:
        gaps = Counter(b.get('event_coverage','unspecified') for b in payload['market']['bars']
                       if b.get('corporate_action_known') is not True)
    quality=payload['snapshot'].get('quality',{})
    # A certificate, when one was derived, names the exact cell classes that refused.
    # Repeating the blanket flag name would hide which of six independent claims failed.
    certificate=(payload.get('audit') or {}).get('data_certification') or {}
    blockers=[]
    if len(sessions) < policy['minimum_effective_sections']:
        blockers.append('power_plan_unreachable_on_declared_window')
    if tail_mass < 1:
        blockers.append('bootstrap_tail_resolution_inadequate')
    if not all(quality.get(k) is True for k in ('pit_certified','universe_certified','account_events_certified')):
        blockers.append('snapshot_certification_incomplete')
    if not all(payload['market'].get(k) is True for k in ('event_coverage_certified','tradability_certified')):
        blockers.append('account_evidence_uncertified')
    if not receipt_present:
        blockers.append('independent_data_contract_receipt_missing')
    return {'schema_version':'fuxi.reachability.1','policy_hash':policy['policy_hash'],
            'maximum_requested_daily_sections':len(sessions),'power_planned_sections':policy['minimum_effective_sections'],
            'bootstrap_expected_tail_draws':tail_mass,'blockers':blockers,
            'certification_hash':certificate.get('certification_hash'),
            'certified_claims':certificate.get('flags'),'certification_gaps':certificate.get('blocking_gaps'),
            'known_necessary_conditions_satisfied':not blockers,'sufficient_for_acceptance':False,
            'unknown_action_cells':sum(gaps.values()),'action_gap_counts':dict(gaps),
            'meaning':'Current-window structural/data feasibility; not a verdict on existence of alpha.',
            'protocol_disposition':'Keep frozen formal gates; audit power approximation before any separately registered replacement.'}
