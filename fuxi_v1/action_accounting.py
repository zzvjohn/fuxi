"""Entitlements from record-date ownership, separate cash and share settlement."""
from decimal import Decimal,ROUND_FLOOR,ROUND_CEILING
from .contracts import digest,require
from .account import money

SUPPORTED={'cash_terms_supported','ordinary_distribution_supported','restructuring_no_holder_distribution'}


def pending_shares(account):
    result={}
    for claim in account.share_receivables.values():
        if claim is not None:result[claim['instrument']]=result.get(claim['instrument'],0)+claim['quantity']
    return result


def post_entitlements(account,terms,day,record_holdings,entitlements,ir_hash,execution,gaps,*,fractional_resolver=None):
    for term in terms:
        if term['ex_date']!=day or term['status'] not in SUPPORTED:continue
        quantity=record_holdings.get(term['evidence_id'],0)
        if quantity<=0:continue
        base=digest([ir_hash,term['evidence_id'],'entitlement'])
        cash=money(str(term['gross_cash_per_share']))*quantity*(1-money(execution.get('cash_tax_fraction',term['tax_fraction'])))
        if cash>0:
            account.apply({'event_id':base,'kind':'receivable','timestamp':day+'T09:00:00+08:00',
                'evidence_id':term['evidence_id'],'payload':{'entitlement_id':base,'amount':str(cash)}})
            entitlements[term['evidence_id']]=base
        extra=Decimal(str(term.get('share_distribution',0)))*quantity
        if extra<=0:continue
        if extra!=extra.to_integral_value():
            if fractional_resolver is not None:
                resolved=fractional_resolver(term,quantity,extra)
                if resolved is not None:
                    require(type(resolved) is int and resolved in {int(extra.to_integral_value(rounding=ROUND_FLOOR)),
                            int(extra.to_integral_value(rounding=ROUND_CEILING))},'fractional resolver left supported integer bounds')
                    extra=Decimal(resolved)
            if extra==extra.to_integral_value():
                policy=None
            else:
                policy=execution.get('fractional_share_policy','block')
            if policy=='block':
                account.apply({'event_id':base+':fractional','kind':'unknown','timestamp':day+'T09:00:00+08:00',
                    'evidence_id':term['evidence_id'],'payload':{'instrument':term['instrument'],
                    'reason':'fractional_allotment_not_observed'}})
                gaps.append({'session':day,'instrument':term['instrument'],'reason':'fractional_allotment_not_observed'})
                continue
            if policy is not None:
                require(policy in {'floor','ceil'},'unsupported fractional share policy')
                extra=extra.to_integral_value(rounding=ROUND_FLOOR if policy=='floor' else ROUND_CEILING)
        if extra>0:
            account.apply({'event_id':base+':shares','kind':'share_entitlement','timestamp':day+'T09:00:00+08:00',
                'evidence_id':term['evidence_id'],'payload':{'entitlement_id':base+':shares','instrument':term['instrument'],
                    'quantity':int(extra),'sellable_at':term['share_credit_date']+'T09:00:00+08:00'}})
    # Grant and credit all claims at the same cutoff after the entitlement loop.
    for key,claim in list(account.share_receivables.items()):
        if claim is not None and claim['sellable_at'][:10]==day:
            account.apply({'event_id':key+':credit','kind':'share_credit','timestamp':day+'T09:00:00+08:00',
                'evidence_id':key,'payload':{'entitlement_id':key}})


def sell_quantity(requested,held,settled,capacity,lot,allow_odd=False):
    """The odd remainder is indivisible; capacity and settlement still constrain it."""
    upper=max(0,min(requested,settled,capacity))
    regular=upper//lot*lot
    if not allow_odd or held%lot==0:return regular
    residual=held%lot
    with_residual=(upper-residual)//lot*lot+residual if upper>=residual else 0
    # Keep any remaining odd shares whole; never split a residual into smaller lots.
    return max(regular,with_residual)


def reconcile_account(result,initial_cash,market=None):
    """Independent arithmetic replay, without Account.apply or Account.mark."""
    cash=Decimal(initial_cash);held={};receivable={};pending={};events={};days=0;unknown=set()
    quotes={(b['session'],b['instrument']):b.get('close') for b in market['bars']} if market else None
    for row in result['journal']:events.setdefault(row['timestamp'][:10],[]).append(row)
    for row in result['daily'][1:]:
        for event in events.get(row['session'],[]):
            p=event['payload'];kind=event['kind']
            if kind=='fill':
                q=p['quantity'] if p['side']=='buy' else -p['quantity']
                held[p['instrument']]=held.get(p['instrument'],0)+q
                cash-=Decimal(p['price'])*q+Decimal(p['fee'])
            elif kind=='receivable':receivable[p['entitlement_id']]=Decimal(p['amount'])
            elif kind=='payment':cash+=receivable.pop(p['entitlement_id'])
            elif kind=='share_entitlement':pending[p['entitlement_id']]=p
            elif kind=='share_credit':
                claim=pending.pop(p['entitlement_id']);held[claim['instrument']]=held.get(claim['instrument'],0)+claim['quantity']
            elif kind=='security_transfer':
                require(held.get(p['from_instrument'])==p['quantity'],'independent transfer quantity mismatch')
                held.pop(p['from_instrument'])
                held[p['to_instrument']]=held.get(p['to_instrument'],0)+p['quantity']
                unknown.difference_update(p['resolved_event_ids'])
            elif kind=='unknown':unknown.add(event['event_id'])
            else:require(kind=='unfilled','unsupported reconciliation event')
        require(cash==Decimal(row['cash']),'independent cash reconciliation failed')
        require({k:v for k,v in held.items() if v}==row['holdings'],'independent share reconciliation failed')
        require(sum(receivable.values(),Decimal(0))==Decimal(row['receivables']),'independent cash entitlement reconciliation failed')
        bycode={}
        for claim in pending.values():bycode[claim['instrument']]=bycode.get(claim['instrument'],0)+claim['quantity']
        require(bycode==row.get('pending_shares',{}),'independent pending-share reconciliation failed')
        if quotes is not None:
            positions={c:held.get(c,0)+bycode.get(c,0) for c in set(held)|set(bycode)}
            identifiable=not unknown and all(quotes.get((row['session'],c)) is not None and
                quotes[(row['session'],c)]>0 for c,q in positions.items() if q)
            require(identifiable==row['valuation_identifiable'] and (row['nav'] is not None)==identifiable,
                    'independent valuation status reconciliation failed')
        if quotes is not None and row['nav'] is not None:
            marked=cash+sum(receivable.values(),Decimal(0))
            for code in set(held)|set(bycode):
                quantity=held.get(code,0)+bycode.get(code,0)
                if quantity:
                    price=quotes.get((row['session'],code));require(price is not None,'reconciliation missing quote')
                    marked+=Decimal(str(price))*quantity
            require(marked==Decimal(row['nav']),'independent NAV reconciliation failed')
        days+=1
    require(days>0,'zero reconciled sessions')
    return {'sessions':days,'journal_entries':len(result['journal']),'cash_and_share_reconciled':True,
            'nav_reconciled':quotes is not None,'scope':'arithmetic reconciliation; not external broker or JQ observation'}
