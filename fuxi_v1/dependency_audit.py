"""Narrow feature dependencies and conservative closure of possible carried rights.

This produces a review request, never an independent investment certificate.
"""
from collections import Counter
from .contracts import digest, require, timestamp, finite
from .dsl import compile_ast


def dependency_scope(snapshot, candidates, start, end, market):
    require(start in snapshot.sessions and end in snapshot.sessions and start <= end and candidates,
            'nonempty dependency scope required')
    compiled = [compile_ast(c['canonical_ast'], snapshot.payload['fields']) for c in candidates]
    fields = sorted(set().union(*(set(c['required_fields']) for c in compiled)))
    lookback = max(c['lookback'] for c in compiled)
    first, last = snapshot.sessions.index(start), snapshot.sessions.index(end)
    selected = snapshot.sessions[first:last+1]
    ever = set().union(*(set(snapshot.universe[d]) for d in selected))
    rights = set(ever)
    # Include issued/transferred securities even when they never become eligible selections.
    changed = True
    while changed:
        before = len(rights)
        for event in market.get('actions', []):
            p = event.get('payload', {})
            if p.get('instrument',p.get('from_instrument')) in rights:
                rights.update(x for x in (p.get('target_instrument'), p.get('new_instrument'),p.get('to_instrument')) if x)
        for term in market.get('action_terms', market.get('corporate_action_terms', [])):
            if term.get('instrument') in rights:
                rights.update(x for x in (term.get('target_instrument'), term.get('new_instrument')) if x)
        changed = len(rights) != before
    feature_cells, cutoffs = set(), {}
    for i in range(first, last+1):
        for c in compiled:
            for day in snapshot.sessions[max(0, i-c['lookback']):i+1]:
                for code in snapshot.universe[snapshot.sessions[i]]:
                    for field in c['required_fields']:
                        key = (day,code,field)
                        feature_cells.add(key)
                        cutoffs[key] = min(cutoffs.get(key,snapshot.clocks[snapshot.sessions[i]]),snapshot.clocks[snapshot.sessions[i]])
    good = late = missing = 0
    # Use the earliest requesting clock; later availability must not hide early gaps.
    session_index = {d:i for i,d in enumerate(snapshot.sessions)}
    for (day, code, field), clock in cutoffs.items():
        item = snapshot.cells.get((day, code, field))
        value = snapshot.value(session_index[day], code, field, clock)
        if value is not None:
            good += 1
        elif item:
            late += 1
        else:
            missing += 1
    # Holding closure deliberately survives departures from the selection universe.
    holding_cells = [(day, code) for day in selected for code in sorted(rights)]
    bars = {(r['session'], r['instrument']):r for r in market['bars']}
    problems = Counter()
    for key in holding_cells:
        bar = bars.get(key)
        if bar is None or bar.get('close') is None:
            problems['potential_holding_mark_missing'] += 1
        if bar is None or bar.get('corporate_action_known') is not True:
            problems['potential_holding_event_coverage_unknown'] += 1
    quality = snapshot.payload.get('quality', {})
    blockers = list(quality.get('gaps', []))
    blockers += ['independent_primary_evidence_review_missing', 'revision_vintage_review_missing']
    if not quality.get('universe_certified'):
        blockers.append('lifecycle_and_selection_review_missing')
    if not market.get('tradability_certified'):
        blockers.append('execution_rules_review_missing')
    blockers += list(problems)
    result = {'schema_version':'fuxi.dependency_review.1', 'snapshot_hash':snapshot.hash,
              'market_hash':market['market_hash'], 'window':{'start':start,'end':end},
              'fields':fields, 'max_lookback':lookback, 'universe_rule':snapshot.payload.get('provenance', {}).get('selection'),
              'feature_dependency_cells':len(feature_cells), 'available_feature_cells':good,
              'unavailable_or_null_cells':late, 'absent_feature_cells':missing,
              'feature_dependency_hash':digest(sorted(feature_cells)),
              'ever_eligible':sorted(ever), 'holdings_and_rights_closure':sorted(rights),
              'potential_holding_cells':len(holding_cells), 'holding_dependency_hash':digest(holding_cells),
              'label_horizon_handled_by_explicit_labels':True,
              'gaps':dict(problems), 'blocking_gaps':sorted(set(blockers)),
              'investment_eligibility_certified':False, 'receipt_kind':'derived_review_request',
              'scope_policy':'window/fields narrowed; carried holdings and entitlements never dropped'}
    result['scope_hash'] = digest(result)
    return result


def pit_styles(spec, snapshot):
    """Select a version known at each decision. Current classifications cannot fill history."""
    require(spec.get('schema_version') == 'fuxi.pit_styles.1', 'style schema invalid')
    require(spec.get('source_id') and spec.get('primary_evidence_sha256'), 'style primary evidence required')
    require(set(spec['style_names']) >= {'industry','float_market_cap'}, 'industry and float cap required')
    index = {}
    for row in spec['rows']:
        require(row['style'] in spec['style_names'] and row['evidence_id'] and row['revision_id'], 'style lineage missing')
        require(isinstance(row['value'],str) and row['value'] if row['style']=='industry' else
                row['value'] is None or finite(row['value']), 'style value invalid')
        times = [timestamp(row[k]) for k in ('published_at','available_at','ingested_at')]
        require(times[0] <= times[1] and times[0] <= times[2], 'style chronology invalid')
        require(row['effective_session'] in snapshot.sessions, 'style effective session outside scope')
        bucket = index.setdefault((row['instrument'],row['style']), [])
        require(not any(r['revision_id'] == row['revision_id'] for r in bucket), 'duplicate style revision')
        bucket.append(row)
    raw = {name:{d:{s:None for s in snapshot.universe[d]} for d in snapshot.sessions[snapshot.start:]}
           for name in spec['style_names']}
    source = {}
    for name, days in raw.items():
        for day, cross in days.items():
            for code in cross:
                choices = [r for r in index.get((code,name), []) if r['effective_session'] <= day and
                           timestamp(r['available_at']) <= snapshot.clocks[day] and
                           timestamp(r['ingested_at']) <= snapshot.clocks[day]]
                if choices:
                    latest = max(choices,key=lambda r:(r['effective_session'],timestamp(r['available_at'])))
                    cross[code] = latest['value']
                    source[(name,day,code)] = latest['evidence_id']
    numeric = {k:v for k,v in raw.items() if k != 'industry'}
    categories = sorted({v for cross in raw['industry'].values() for v in cross.values() if v is not None})
    require(categories,'no PIT industry observations available; current industries cannot substitute')
    # Use one omitted category; common support still excludes an unknown classification.
    for group in categories[1:]:
        numeric['industry:'+str(group)] = {d:{s:None if v is None else float(v == group) for s,v in cross.items()}
                                         for d,cross in raw['industry'].items()}
    if len(categories) == 1:
        numeric['industry_known'] = {d:{s:None if v is None else 1. for s,v in cross.items()} for d,cross in raw['industry'].items()}
    return numeric, {'source_id':spec['source_id'], 'primary_evidence_sha256':spec['primary_evidence_sha256'],
                     'source_cell_count':len(source), 'investment_neutralization_certified':False}
