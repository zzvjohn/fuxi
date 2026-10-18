"""Manual monitoring and future-calendar preflight reuse the existing receipt services."""
from .contracts import digest, require, timestamp, finite
from .monitoring import drift_report


def forward_plan(spec):
    """Calendar-backed draft; registering/protecting remains native operate actions."""
    days=spec['sessions'];clocks=spec['decision_times'];freeze=timestamp(spec['frozen_at'])
    require(days and days==sorted(set(days)) and set(clocks)==set(days),'forward calendar identity invalid')
    require(spec.get('calendar_evidence_sha256') and len(spec['calendar_evidence_sha256'])==64,'calendar source hash required')
    from pathlib import Path
    import hashlib
    require(spec.get('calendar_evidence_path') and Path(spec['calendar_evidence_path']).is_file(), 'actual calendar evidence required')
    require(hashlib.sha256(Path(spec['calendar_evidence_path']).read_bytes()).hexdigest()==spec['calendar_evidence_sha256'],
            'calendar evidence changed')
    require(all(timestamp(clocks[d]).date().isoformat()==d for d in days),'calendar decision date mismatch')
    future=[d for d in days if timestamp(clocks[d])>freeze]
    offsets=spec.get('look_sessions',[60,120]);horizon=spec.get('horizon_sessions',5)
    require(type(horizon)is int and horizon>0 and offsets==sorted(set(offsets))
            and offsets and all(type(n)is int and n>0 for n in offsets),'look/horizon design invalid')
    looks=[]
    for n in offsets:
        if len(future)>=n+horizon:
            looks.append({'sections':n,'decision_end':future[n-1],
                          'mature_at':clocks[future[n-1+horizon]],'status':'calendar_resolved'})
        else:looks.append({'sections':n,'decision_end':None,'mature_at':None,'status':'calendar_missing'})
    receipts={k:spec.get(k) for k in ('research_receipt','protocol_receipt','parity_receipt')}
    missing=[k for k,v in receipts.items() if not v]
    return {'schema_version':'fuxi.forward_calendar_preflight.1','frozen_at':spec['frozen_at'],
            'calendar_hash':digest({'sessions':days,'clocks':clocks}),'looks':looks,
            'required_native_actions':['register-protocol','freeze-forward','observe-forward','build-release','activate-paper'],
            'receipt_references':receipts,'missing_receipts':missing,
            'status':'await_receipt_verification' if not missing and all(r['mature_at'] for r in looks) else 'pending',
            'activation_performed':False,'future_feedback_to_same_candidate_allowed':False,
            'rollback_requires':['unknown_orders=0','account_matched=true','frozen_rollback_bundle'],
            'live_execution_allowed':False}


def monitor(spec, store=None):
    require(spec.get('schema_version')=='fuxi.operational_observation.1','operational observation schema invalid')
    clock=timestamp(spec['observation_clock']);issues=[]
    expected=spec['expected_dependencies'];actual=spec['actual_dependencies']
    require(expected and actual,'dependency comparison cannot be empty')
    for name in sorted(set(expected)|set(actual)):
        if name not in expected or name not in actual or expected.get(name)!=actual.get(name):
            issues.append({'kind':'dependency_drift','name':name})
    for item in spec.get('data_channels',[]):
        if item.get('latest_available_at') is None or item.get('expected_latest_at') is None:
            issues.append({'kind':'data_freshness_unknown','name':item['name']});continue
        if timestamp(item['latest_available_at'])<timestamp(item['expected_latest_at']):
            issues.append({'kind':'stale_data','name':item['name']})
    for row in spec.get('signals',[]):
        decision,available,execution=(timestamp(row[k]) for k in ('decision_time','available_at','execution_time'))
        if available>decision or execution<=decision or decision>clock:
            issues.append({'kind':'signal_clock_violation','instrument':row.get('instrument')})
    orders=spec.get('orders')
    require(isinstance(orders,list),'explicit order observation list required')
    known={'pending','acknowledged','partial','filled','cancelled','rejected'}
    if any(r.get('status') not in known or not r.get('order_id') for r in orders):
        issues.append({'kind':'unknown_order'})
    account=spec.get('reconciliation') or {}
    if account.get('unknown_orders')!=0 or account.get('account_matched') is not True:
        issues.append({'kind':'account_reconciliation_incomplete'})
    for name in ('cash','nav','receivables','holdings'):
        if name not in account or account[name] is None:issues.append({'kind':'account_field_unknown','name':name})
    if account.get('nav') is not None:
        from decimal import Decimal, InvalidOperation
        try:valid=Decimal(str(account['nav'])).is_finite() and Decimal(str(account['nav']))>0
        except InvalidOperation:valid=False
        if not valid:issues.append({'kind':'NAV_invalid'})
    drift=None
    if spec.get('reference_scores') is not None and spec.get('current_scores') is not None:
        drift=drift_report(spec['reference_scores'],spec['current_scores'])
    revoked=[]
    if spec.get('required_receipts'):
        require(store is not None,'receipt authority required for active monitor')
        for receipt,kind in spec['required_receipts'].items():
            try:store.consume_receipt(receipt,kind)
            except ValueError:revoked.append(receipt);issues.append({'kind':'receipt_invalid','receipt_id':receipt})
    registered=spec.get('registered_looks',[])
    require(registered==sorted(set(registered),key=timestamp),'registered monitor looks invalid')
    action='halt_new_submissions_and_reconcile' if issues else 'continue_manual_observation'
    return {'schema_version':'fuxi.operational_monitor.1','observation_hash':digest(spec),'issues':issues,
            'recommendation':action,'drift':drift,'invalid_receipts':revoked,
            'statistical_evaluation_due':spec['observation_clock'] in registered,
            'E5_evaluation_performed':False,'automatic_resubmit':False,
            'automatic_champion_switch':False,'live_execution_allowed':False}
