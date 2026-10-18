"""Independent aggregate journal arithmetic for NAV reconciliation.

It verifies cash, total shares and rights. It does not certify fills, settlement,
corporate-action completeness, or price vintages.
"""
from collections import Counter
from decimal import Decimal
from .contracts import require, timestamp


def reconcile(account, market):
    cash = Decimal(market['execution']['initial_cash'])
    shares, claims, share_claims, unknown = {}, {}, {}, set()
    journal = iter(sorted(account['journal'],key=lambda r:timestamp(r['timestamp'])))
    event = next(journal,None)
    bars = market.bar_index if getattr(market,"_columnar_market",False) else {(r['session'],r['instrument']):r for r in market['bars']}
    rows, unsupported = [], Counter()
    for row in account['daily']:
        if row['session'].endswith(':initial'):
            rows.append({'session':row['session'],'matched':Decimal(row['cash']) == cash,
                         'identifiable':True,'reference_nav':str(cash),'missing_marks':[]})
            continue
        while event is not None and timestamp(event['timestamp']) <= timestamp(row['observation_cutoff']):
            p, kind = event['payload'], event['kind']
            if kind == 'fill':
                code, q = p['instrument'], p['quantity']
                change = Decimal(p['price'])*q
                cash += (-change if p['side'] == 'buy' else change)-Decimal(p['fee'])
                shares[code] = shares.get(code,0)+(q if p['side']=='buy' else -q)
            elif kind == 'receivable':
                claims[p['entitlement_id']] = Decimal(p['amount'])
            elif kind == 'payment':
                cash += claims.pop(p['entitlement_id'])
            elif kind == 'share_entitlement':
                share_claims[p['entitlement_id']] = (p['instrument'],p['quantity'])
            elif kind == 'share_credit':
                code, q = share_claims.pop(p['entitlement_id'])
                shares[code] = shares.get(code,0)+q
            elif kind == 'split':
                quantity = Decimal(p['multiplier'])*shares.get(p['instrument'],0)
                require(quantity == quantity.to_integral_value(), 'reference fractional event unsupported')
                shares[p['instrument']] = int(quantity)
            elif kind == 'security_transfer':
                require(shares.get(p['from_instrument'],0) == p['quantity'], 'reference transfer amount mismatch')
                shares.pop(p['from_instrument'])
                shares[p['to_instrument']] = shares.get(p['to_instrument'],0)+p['quantity']
                unknown.difference_update(p['resolved_event_ids'])
            elif kind == 'unknown':
                unknown.add(event['event_id'])
            elif kind != 'unfilled':
                unsupported[kind] += 1
            event = next(journal,None)
        held = {s:q for s,q in shares.items() if q > 0}
        pending = {}
        for code,q in share_claims.values():
            pending[code] = pending.get(code,0)+q
        missing = [s for s in sorted(set(held)|set(pending)) if bars.get((row['session'],s),{}).get('close') is None]  # P-20261015-004 deterministic order
        identifiable = not unknown and not missing and not unsupported
        nav = None if not identifiable else cash+sum(claims.values(),Decimal(0))+sum(
            Decimal(str(bars[(row['session'],s)]['close']))*(held.get(s,0)+pending.get(s,0)) for s in set(held)|set(pending))
        matched = (Decimal(row['cash']) == cash and Decimal(row['receivables']) == sum(claims.values(),Decimal(0))
                   and row['holdings'] == held and row.get('pending_shares',{}) == pending
                   and sorted(row.get('unresolved_events',[])) == sorted(unknown)
                   and row['valuation_identifiable'] == identifiable
                   and (row['nav'] is None if nav is None else abs(Decimal(row['nav'])-nav) <= Decimal('0.000001')))
        rows.append({'session':row['session'],'matched':matched,'identifiable':identifiable,
                     'reference_nav':str(nav) if nav is not None else None,'missing_marks':missing})
    require(rows, 'zero reference reconciliation rows')
    return {'eval_status':'valid' if all(r['matched'] for r in rows) and not unsupported else 'error',
            'n_pairs':len(rows),'mismatches':[r for r in rows if not r['matched']],
            'unidentifiable_rows':sum(not r['identifiable'] for r in rows),'unsupported_kinds':dict(unsupported),
            'formal_certification':False,'scope':'independent_journal_arithmetic_not_independent_market_evidence'}


def account_diagnostics(candidate, baseline):
    require([r['session'] for r in candidate['daily']] == [r['session'] for r in baseline['daily']], 'attribution window mismatch')
    rows = []
    for i, (a,b) in enumerate(zip(candidate['daily'],baseline['daily'])):
        prior = candidate['daily'][i-1] if i else None
        base_prior = baseline['daily'][i-1] if i else None
        fees = sum((Decimal(o.get('fee','0')) for o in candidate['orders'] if o['session']==a['session']),Decimal(0))
        ra = float(Decimal(a['nav'])/Decimal(prior['nav'])-1) if prior and a['nav'] and prior['nav'] else None
        rb = float(Decimal(b['nav'])/Decimal(base_prior['nav'])-1) if base_prior and b['nav'] and base_prior['nav'] else None
        fee_drag = float(fees/Decimal(prior['nav'])) if prior and prior['nav'] else None
        rows.append({'session':a['session'],'net_return':ra,'pool_benchmark_return':rb,
                     'relative_net_return':ra-rb if ra is not None and rb is not None else None,
                     'explicit_fee_drag':fee_drag,'slippage_drag':None,'style_component':None,
                     'selection_and_execution_residual':None,
                     'attribution_residual':ra-rb+fee_drag if None not in (ra,rb,fee_drag) else None})
    refusals = Counter(o['reason'] for o in candidate['orders'] if o.get('reason'))
    return {'scope':'pool_relative_diagnostic_not_causal_alpha_attribution','daily':rows,
            'unfilled_by_reason':dict(refusals),'explicit_fees':str(sum((Decimal(o.get('fee','0')) for o in candidate['orders']),Decimal(0))),
            'unidentified_components':['PIT industry/float cap style','slippage and selection causal split'],
            'legacy_S5_cost_bps':15,'native_cost_model':candidate['assumptions'],
            'cost_models_equivalent':False,'unidentifiable_daily_slots':sum(r['relative_net_return'] is None for r in rows)}
