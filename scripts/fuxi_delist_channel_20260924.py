# -*- coding: utf-8 -*-
"""The vendor's explicit delisting channel, alongside the existing derivation.

Why a second channel is needed
------------------------------
`derive_freeze_facts` (fuxi_ledger_advance_20260923) proves a delisting from two
signs: the name is gone from the *current* security master, and the suspend table
covers its quote-free tail contiguously. The first sign is unusable here: the
local master is a snapshot that keeps names which have since left the market, so
23 of the 52 quote-free tails among held names never even reached the test. The
one arm that still stops in the ladder stops for exactly that reason.

This module adds the vendor's own `stock_basic(list_status='D')` delist_date as
an explicit channel. It never *replaces* a proof; it supplies one where the
master is stale, and where both channels speak it demands that they be compared
and the disagreement published rather than resolved silently.

Direction of the change
-----------------------
Every fact added here moves a position from `unknown`/`halted` to `delisted`,
which means: frozen at the last observed mark, no synthetic return, no
revaluation, cannot be bought or sold, and no action check. It cannot create an
entitlement and it cannot create a fill. The caller is required to prove that the
net-asset path is unchanged on every session both readings computed; if the
channel ever changed a NAV it would be a revaluation, not a relabelling.
"""
import csv
import io

FACT_EVIDENCE = ('vendor stock_basic list_status=D delist_date {date}; '
                 'last quote in the window {last}')


def load_delist_dates(path):
    """{ts_code: ISO delist_date} for every name the vendor marks as delisted."""
    with io.open(path, 'rb') as handle:
        payload = handle.read()
    assert payload[:3] == b'\xef\xbb\xbf', 'delist table lost its BOM'
    assert payload.count(b'\n') - payload.count(b'\r\n') == 0, 'delist table has bare LF'
    reader = csv.DictReader(io.StringIO(payload[3:].decode('utf-8'), newline=''))
    out = {}
    for row in reader:
        code = (row.get('ts_code') or '').strip()
        raw = (row.get('delist_date') or '').strip()
        if not code or not raw or row.get('list_status') != 'D':
            continue
        assert len(raw) == 8 and raw.isdigit(), 'unexpected delist_date %r' % raw
        out[code] = '%s-%s-%s' % (raw[:4], raw[4:6], raw[6:])
    assert out, 'delist table yielded no rows'
    return out


def explicit_facts(prices, sessions, delisted):
    """Derive a freeze fact from the vendor's delist_date alone.

    Accepted only when the delist date is a session inside the window and the
    name's last quote in the window strictly precedes it. A name whose quotes run
    past the vendor's delist date is a vendor inconsistency, not a freeze, and is
    refused with a reason instead of being forced into the evidence map.
    """
    index = {day: position for position, day in enumerate(sessions)}
    last_quote = prices.groupby('ts_code').trade_date.max().to_dict()
    accepted, refused = {}, []
    for code in sorted(last_quote):
        vendor = delisted.get(code)
        if not vendor:
            continue
        last = last_quote[code]
        if last >= vendor:
            refused.append({'code': code, 'last_quote': last, 'vendor_delist_date': vendor,
                            'reason': 'the window still quotes the name on or after the '
                                      'vendor delist date'})
            continue
        if vendor not in index:
            refused.append({'code': code, 'last_quote': last, 'vendor_delist_date': vendor,
                            'reason': 'the vendor delist date is not a session of the window'})
            continue
        accepted[code] = {
            'date': vendor,
            'evidence': FACT_EVIDENCE.format(date=vendor, last=last),
            'channel': 'vendor_delist_date',
            'last_quote': last,
        }
    return accepted, refused


def merge_freeze_facts(legacy, explicit):
    """Union, with the vendor's date preferred where both channels speak.

    A preference, not a proof: the disagreement is returned so the caller has to
    publish it. The vendor date wins because it is the market's own record of the
    name leaving, while the derived date is a proxy read off the suspend table's
    tail - and that table is what lags.
    """
    facts = {code: dict(fact) for code, fact in legacy.items()}
    agreement, overrides, additions = [], [], []
    for code, fact in sorted(explicit.items()):
        previous = legacy.get(code)
        if previous is None:
            facts[code] = dict(fact)
            additions.append({'code': code, 'date': fact['date'],
                              'last_quote': fact['last_quote'],
                              'why': 'absent from the current master test was unusable'})
            continue
        if previous['date'] == fact['date']:
            agreement.append({'code': code, 'date': fact['date'], 'channels': 'both'})
            continue
        facts[code] = dict(fact)
        overrides.append({'code': code, 'vendor_delist_date': fact['date'],
                          'suspend_derived_freeze': previous['date'],
                          'last_quote': fact['last_quote'],
                          'moved_earlier': fact['date'] < previous['date'],
                          'suspend_derived_evidence': previous.get('evidence')})
    return facts, {'additions': additions, 'agreement': agreement, 'overrides': overrides}
