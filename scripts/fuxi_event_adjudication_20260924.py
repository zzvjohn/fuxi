# -*- coding: utf-8 -*-
"""Item 1 next action: adjudicate the held-day corporate-action blockers through
independent published channels, without collecting anything new.

Why this exists
---------------
The inventory batch ended with eight corporate actions that sit on a held day and
whose vendor rows fail the reference identity, and it recorded the next action as
"put the four newly discovered events through the independent-channel
adjudication the earlier batch used on the original four". This script does that
for all eight at once, and it does it through three channels rather than one:

  A  the vendor `pre_close` field on the ex-date row, which is the exchange's own
     published reference price for that session;
  B  the ex-date `stk_limit` pair, from which the reference price and the limit
     rate are both recoverable without assuming the board;
  C  the `adj_factor` ratio across the ex-date, which is a vendor series rather
     than an exchange publication and therefore ranks below A and B.

What the channels do and do not buy
-----------------------------------
They determine the exchange reference price. They do not by themselves split that
price into a cash part and a share part, because one price is one equation in two
unknowns. What makes the reconstruction admissible is that the split is
economically second order: for any pair consistent with the reference price the
value of the position on the ex-date is the previous close exactly, so the
reconstruction reclassifies between cash and shares without creating value. The
reclassification is measured against the declared row and refused when it exceeds
a declared fraction of the ex-date basis. One of the eight events is refused on
that basis because the two channel families disagree by 22 percent, which is a
first-order ambiguity and must not be bet on.

No collection, no writes to raw data, sealed evidence, production state or any
prior report. Every accepted event is reported with its value-preservation
identity and its reclassification size so a later batch can falsify it.
"""

import argparse
import hashlib
import io
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from fuxi_event_inventory_plan_20260924 import (ARMS, WINDOW_HI, WINDOW_LO,  # noqa: E402
                                                held_pairs, holding_calendar,
                                                load_targets, sessions)
from fuxi_event_inventory_20260923 import load_sealed_rows  # noqa: E402

OUT = ROOT / 'reports/event_adjudication_20260924'
SEALED = ROOT / 'reports/event_inventory_20260923/inventory.json'
PLAN_DIR = ROOT / 'reports/event_inventory_plan_20260924'
MISSING_ROWS = PLAN_DIR / 'missing_price_rows.csv'
CHECKPOINTS = ROOT / 'reports/ledger_advance_20260923/checkpoints'
EVIDENCE = ROOT / 'reports/window_evidence_20260923'

TICK = 0.01
# Two tick-rounded publications - the limit pair average and the published
# `pre_close` field - can each be off by half a tick, so they may differ by a
# full tick without either being wrong. Observed disagreement is at most 0.5 tick.
LIMIT_AGREEMENT_TICKS = 1.0
# The adjustment factor carries four decimals on two legs, so the ratio it
# implies is reproducible to about 2e-4/adj. For every held-day event the
# observed factor is at least 1.3, which puts the derivation at 0.015 percent; the
# bound below is the loose reading of that and is still two orders of magnitude
# below the smallest contradiction actually observed (22 percent).
ADJ_REL_BOUND = 1e-3
# Refuse rather than reclassify when the admissible pair moves the declared
# entitlement by more than this fraction of the ex-date basis. Value is preserved
# by construction at any size, so the bound guards the split, not the total.
RECLASS_LIMIT = 0.01

APPROVED_BUCKET = 'approved'
SEALED_BUCKETS = ('approved', 'rejected', 'unresolved')

# The ledger reports the blocker on the ex-date itself, so an event blocks an arm
# exactly when the security is held on its ex-date. Recorded here so the
# enumeration can be tied back to the four checkpoints instead of asserted.
CHECKPOINT_BLOCKERS = {
    'r4_equal_rank': ('600256.SH', '2025-07-18'),
    'llm_family_1_1_fresh_llm': ('002713.SZ', '2025-12-30'),
    'llm_family_2_1_fresh_llm': ('000333.SZ', '2026-06-29'),
    'mem_0626_broadcast_repaired': ('301023.SZ', '2025-05-23'),
}


def num(value):
    if value is None or value == '':
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def iso(value):
    text = str(value or '').strip()
    if len(text) == 8 and text.isdigit():
        return '%s-%s-%s' % (text[:4], text[4:6], text[6:8])
    return text


def iso_from(text):
    """Sealed row dates are already ISO; keep the helper for both shapes."""
    return iso(text) if text else None


def solve_reference(up_limit, down_limit):
    """Recover (reference price, limit rate) from one published limit pair.

    The exchange publishes `round(ref * (1 + rate))` and `round(ref * (1 - rate))`
    on the price tick. Their average recovers the reference price to within half a
    tick and their spread over their sum recovers the limit rate without the
    caller having to assume a board, which matters because the same code can be
    a 5 percent, a 10 percent or a 20 percent name depending on its status.
    """
    up, down = num(up_limit), num(down_limit)
    if not up or not down or up <= 0 or down <= 0 or up <= down:
        return None, None
    total = up + down
    return total / 2.0, (up - down) / total


def rate_resolution(up_limit, down_limit):
    """How finely one limit pair can pin the rate.

    Each published limit can be up to half a tick from its exact value, so the
    recovered rate carries at most `tick / (up + down)`, which is `tick / 2 / ref`.
    The rate is therefore read as an order of magnitude - five, ten, twenty or
    thirty percent - and never as a precise number.
    """
    up, down = num(up_limit), num(down_limit)
    if not up or not down or up + down <= 0:
        return None
    return TICK / (up + down)


def channel_readings(pre_close, adj_prev, adj_ex, up_limit, down_limit, reference_field):
    """Read every local channel that speaks to the ex-date reference price."""
    limit_ref, limit_rate = solve_reference(up_limit, down_limit)
    adj_ref = None
    if num(pre_close) and num(adj_prev) and num(adj_ex) and num(adj_prev) > 0:
        adj_ref = num(pre_close) / (num(adj_ex) / num(adj_prev))
    return {
        'prior_close': num(pre_close),
        'reference_field': num(reference_field),
        'limit_reference': limit_ref,
        'limit_rate': limit_rate,
        'adjustment_reference': adj_ref,
        'adjustment_ratio': None if adj_ref is None else num(pre_close) / adj_ref,
    }


def consensus(readings):
    """The reference price the exchange published, plus the agreement evidence.

    Channel A is the authority whenever it exists, because it is the exchange's
    own published reference price to the tick and needs no inversion. Channels B
    and C must then corroborate it inside their derived bounds; a channel outside
    its bound is reported as a disagreement rather than averaged away.
    """
    field = readings.get('reference_field')
    prior = readings.get('prior_close')
    if not field:
        return {'reference': None, 'agreeing': [], 'disagreeing': [],
                'limit_gap_ticks': None, 'adjustment_gap': None,
                'reason': 'no_published_reference_field'}
    agreeing, disagreeing = ['pre_close_field'], []
    limit_gap = None
    limit_ref = readings.get('limit_reference')
    if limit_ref:
        limit_gap = abs(limit_ref - field) / TICK
        (agreeing if limit_gap <= LIMIT_AGREEMENT_TICKS else disagreeing).append('limit_pair')
    adj_ref = readings.get('adjustment_reference')
    adj_gap = None
    if adj_ref:
        adj_gap = abs(adj_ref - field) / field
        (agreeing if adj_gap <= ADJ_REL_BOUND else disagreeing).append('adjustment_ratio')
    return {'reference': field, 'prior_close': prior, 'agreeing': agreeing,
            'disagreeing': disagreeing, 'limit_gap_ticks': limit_gap,
            'adjustment_gap': adj_gap,
            'adjusted': prior is not None and abs(field - prior) > LIMIT_AGREEMENT_TICKS * TICK,
            'reason': 'channels_agree' if not disagreeing else 'channels_disagree'}


def adjudicate(event, readings, consensus_result):
    """Classify one held-day event into an action, a refusal, or a no-entitlement.

    Three outcomes, in order of increasing assumption:

    `no_entitlement`   the exchange's reference price equals the previous close,
                       so the exchange removed nothing. Combined with
                       non-negativity of both a cash and a share component this
                       pins both at zero, which is a derivation rather than a
                       choice: a non-zero cash or a non-zero share rate would both
                       lower the reference price.
    `reference_reconstructed`  the reference price is pinned by consensus and the
                       adjudicable pair stays inside the reclassification bound.
                       The declared share rate is kept and the cash solved, or, if
                       that returns a negative cash, the declared cash is kept and
                       the rate solved - in both directions the ex-date value is
                       the previous close by construction.
    `refused_*`        anything else. A refusal is a result, not a failure.
    """
    code, ex_date = event['code'], event['ex_date']
    declared_rate = num(event.get('declared_total')) or 0.0
    declared_cash = num(event.get('declared_cash'))
    prior = consensus_result.get('prior_close')
    ref = consensus_result.get('reference')
    base = {'code': code, 'ex_date': ex_date, 'rows': event.get('rows'),
            'buckets': event.get('buckets'), 'arms': event.get('arms'),
            'declared_cash': declared_cash, 'declared_total': declared_rate,
            'readings': readings, 'consensus': consensus_result}

    if ref is None or prior is None or prior <= 0:
        return {**base, 'verdict': 'refused_no_local_reference_channel',
                'action': None, 'detail': {}}
    if consensus_result['reason'] != 'channels_agree':
        return {**base, 'verdict': 'refused_channels_disagree', 'action': None,
                'detail': {'disagreeing': consensus_result['disagreeing'],
                           'limit_gap_ticks': consensus_result['limit_gap_ticks'],
                           'adjustment_gap': consensus_result['adjustment_gap']}}
    if not consensus_result['adjusted']:
        # The exchange published an ex-date reference equal to the prior close.
        action = {'event_id': 'adj_' + code + '_' + ex_date, 'code': code,
                  'record_date': event.get('record_date'), 'ex_date': ex_date,
                  'pay_date': event.get('pay_date') or ex_date,
                  'cash_per_share_net': 0.0, 'share_multiplier': 1.0,
                  'new_shares_date': ex_date,
                  'evidence': ('exchange reference equals the prior close on %s from '
                               'channels %s; a non-negative cash or share component '
                               'would both move it, so both are zero'
                               % (ex_date, ','.join(consensus_result['agreeing'])))}
        return {**base, 'verdict': 'no_entitlement_reference_equals_prior_close',
                'action': action,
                'detail': {'identity': 'reference == prior_close', 'cash': 0.0,
                           'rate': 0.0, 'reclassification': 0.0}}

    cash_implied = prior - ref * (1.0 + declared_rate)
    rate_implied = (prior - (declared_cash or 0.0)) / ref - 1.0
    if cash_implied >= 0:
        cash, rate, direction = cash_implied, declared_rate, 'keep_declared_rate'
    elif rate_implied >= 0:
        cash, rate, direction = (declared_cash or 0.0), rate_implied, 'keep_declared_cash'
    else:
        return {**base, 'verdict': 'refused_no_admissible_pair', 'action': None,
                'detail': {'cash_implied': cash_implied, 'rate_implied': rate_implied}}
    reclassification = max(abs(rate - declared_rate) * prior,
                           abs(cash - (declared_cash or 0.0))) / prior
    if reclassification > RECLASS_LIMIT:
        return {**base, 'verdict': 'refused_reclassification_too_large', 'action': None,
                'detail': {'cash_implied': cash_implied, 'rate_implied': rate_implied,
                           'reclassification': reclassification,
                           'limit': RECLASS_LIMIT}}
    tax_factor = 1.0
    if declared_cash and event.get('declared_net') is not None:
        tax_factor = event['declared_net'] / declared_cash
    net_cash = cash * tax_factor
    # Value preservation is an identity of the reconstruction, so it is asserted
    # rather than approximated: (1 + rate) * reference + cash == prior close.
    identity_gap = abs((1.0 + rate) * ref + cash - prior)
    assert identity_gap <= 1e-9, (code, ex_date, identity_gap)
    action = {'event_id': 'adj_' + code + '_' + ex_date, 'code': code,
              'record_date': event.get('record_date'), 'ex_date': ex_date,
              'pay_date': event.get('pay_date') or ex_date,
              'cash_per_share_net': net_cash, 'share_multiplier': 1.0 + rate,
              'new_shares_date': event.get('new_shares_date') or ex_date,
              'evidence': ('reference %s reconstructed from channels %s; %s; '
                           'value preserved to %s'
                           % (ref, ','.join(consensus_result['agreeing']), direction,
                              identity_gap))}
    return {**base, 'verdict': 'reference_reconstructed', 'action': action,
            'detail': {'cash_implied': cash_implied, 'rate_implied': rate_implied,
                       'direction': direction, 'cash': cash, 'rate': rate,
                       'reclassification': reclassification,
                       'tax_factor': tax_factor, 'identity_gap': identity_gap}}


def load_events(held):
    """Held-day events, from the sealed inventory plus its truncated tail.

    The sealed inventory file keeps at most 400 rows per bucket, so twenty of the
    four hundred and twenty unresolved rows are not in it. Those are recomputed
    here with the inventory's own module so the held-day enumeration is complete
    rather than silently clipped.
    """
    payload = json.loads(SEALED.read_text(encoding='utf-8'))
    summary = payload['summary']
    rows = {}
    for bucket in ('rejected', 'unresolved'):
        for row in payload.get(bucket, []):
            key = (row['code'], row['ex_date'])
            entry = rows.setdefault(key, {'code': row['code'], 'ex_date': row['ex_date'],
                                          'buckets': set(), 'rows': 0})
            entry['buckets'].add(bucket)
            entry['rows'] += 1
    stored = {b: len(payload.get(b, [])) for b in SEALED_BUCKETS}
    recomputed = recompute_unresolved()
    known = set(rows)
    hidden = [r for r in recomputed if (r['code'], r['ex_date']) not in known]
    for row in hidden:
        key = (row['code'], row['ex_date'])
        entry = rows.setdefault(key, {'code': row['code'], 'ex_date': row['ex_date'],
                                      'buckets': set(), 'rows': 0})
        entry['buckets'].add('unresolved')
    return rows, {'summary': summary, 'stored_bucket_sizes': stored,
                  'recomputed_unresolved': len(recomputed),
                  'unresolved_rows_absent_from_the_file': len(hidden),
                  'hidden_rows': hidden}


def recompute_unresolved():
    """Reproduce the inventory's unresolved set to recover the truncated tail."""
    rows, _files = load_sealed_rows()
    from corporate_action_inventory import assess_row, merge_same_implementation
    from fuxi_event_inventory_20260923 import price_and_adjustment_panels, proven_alternation
    proven = proven_alternation(rows)
    merged, _ = merge_same_implementation(rows)
    prev_close, legs, _panels = price_and_adjustment_panels()
    out = []
    for row in merged:
        key = (row['code'], row['ex_date'])
        previous, pair = prev_close.get(key), legs.get(key)
        verdict = assess_row(row, identity_proven=row['code'] in proven,
                             pre_close=previous,
                             adj_prev=pair[0] if pair else None,
                             adj_ex=pair[1] if pair else None)
        if verdict.status == 'unresolved':
            out.append({'code': row['code'], 'ex_date': row['ex_date'],
                        'reason': verdict.reason})
    return out


def group_sealed_rows(rows):
    """Index the sealed dividend rows by (code, ex-date)."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row['code'], row['ex_date']), []).append(row)
    return grouped


def declared_components(sealed_rows):
    """Declared cash and share total for the held-day events.

    Two sealed rows can describe one event and disagree on the cash field, so the
    declared cash is taken as the largest declared value and the row count is
    carried forward: a disagreement inside the vendor data is evidence about the
    row, not something to average.
    """
    out = {}
    for (code, ex_date), members in sealed_rows.items():
        cash = [num(m.get('cash_div')) for m in members]
        cash = [c for c in cash if c is not None]
        net = [num(m.get('cash_div_tax')) for m in members]
        net = [n for n in net if n is not None]
        totals = [num(m.get('stk_div')) for m in members]
        totals = [t for t in totals if t is not None]
        out[(code, ex_date)] = {
            'declared_cash': max(cash) if cash else None,
            'declared_net': max(net) if net else None,
            'declared_total': max(totals) if totals else None,
            'record_date': iso_from(members[0].get('record_date')),
            'pay_date': iso_from(members[0].get('pay_date')),
            'new_shares_date': iso_from(members[0].get('div_listdate')),
            'sealed_rows': len(members),
        }
    return out


def price_channels():
    """Prior close, both adjustment legs, the limit pair and the reference field."""
    frames = {}
    prices = pd.read_csv(ROOT / 'data/raw/daily_prices.csv',
                         usecols=['ts_code', 'trade_date', 'close', 'pre_close'],
                         dtype={'ts_code': str, 'trade_date': str})
    prices['trade_date'] = (pd.to_datetime(prices.trade_date, format='mixed')
                            .dt.strftime('%Y-%m-%d'))
    prices = prices.drop_duplicates(['ts_code', 'trade_date']).sort_values(
        ['ts_code', 'trade_date'])
    prices['prior_close'] = prices.groupby('ts_code').close.shift()
    frames['prices'] = prices
    adj = pd.read_csv(ROOT / 'data/raw/adj_factor_daily.csv',
                      usecols=['ts_code', 'trade_date', 'adj_factor'],
                      dtype={'ts_code': str, 'trade_date': str})
    adj['trade_date'] = (pd.to_datetime(adj.trade_date, format='mixed')
                         .dt.strftime('%Y-%m-%d'))
    adj = adj.drop_duplicates(['ts_code', 'trade_date']).sort_values(
        ['ts_code', 'trade_date'])
    adj['prior_adj'] = adj.groupby('ts_code').adj_factor.shift()
    frames['adj'] = adj
    limits = pd.read_csv(ROOT / 'data/raw/stk_limit.csv',
                         usecols=['ts_code', 'trade_date', 'up_limit', 'down_limit'],
                         dtype={'ts_code': str, 'trade_date': str})
    limits['trade_date'] = (pd.to_datetime(limits.trade_date, format='mixed')
                            .dt.strftime('%Y-%m-%d'))
    frames['limits'] = limits
    return frames


def held_day_map(days):
    """Per arm: the held sessions, the held-day blockers, and the missing rows."""
    per_arm = {}
    for arm in ARMS:
        frames, _insufficient = load_targets(arm)
        spans = holding_calendar(frames, days)
        per_arm[arm] = {'pairs': held_pairs(spans, days), 'spans': spans,
                        'stamps': len(frames)}
    return per_arm


def missing_row_blockers(per_arm):
    """Held-day (security, session) pairs the price panel has no row for.

    Reported for cross-reference only. The ledger does not fail on all of them:
    a session that the suspension table marks as halted is valued from the carried
    mark, so the count of absent rows is not the count of blockers.
    """
    table = pd.read_csv(MISSING_ROWS, dtype=str)
    out = {}
    for record in table.itertuples():
        out.setdefault(record.arms, set()).add((record.security, record.session))
    return out


def arm_walk(arm, per_arm, blocking, resolved, refused):
    """Walk one arm's held days to its first remaining corporate-action blocker.

    Only events the adjustment panel marks as implemented, and which the sealed
    inventory never approved, can block a day: the ledger raises its failure on
    `action_required and code not in todays_actions`. An unapproved vendor row on
    a session the exchange did not adjust is therefore carried as a non-blocker
    and reported as such rather than counted.
    """
    pairs = per_arm[arm]['pairs']
    days = sorted({day for _code, day in pairs})
    events = sorted([(code, day) for code, day in blocking if (code, day) in pairs],
                    key=lambda pair: pair[1])
    walk = [{'date': day, 'code': code,
             'kind': 'event_resolved' if (code, day) in resolved else 'event_refused'}
            for code, day in events]
    remaining = [item for item in walk if item['kind'] != 'event_resolved']
    bound = None
    if remaining:
        first = remaining[0]['date']
        earlier = [d for d in days if d < first]
        bound = earlier[-1] if earlier else None
    else:
        bound = days[-1] if days else None
    return {'first_remaining_blocker': remaining[0] if remaining else None,
            'bound_end': bound, 'walk': walk,
            'blocking_events_on_held_days': len(events),
            'note': ('bound is the last held session before the first remaining blocker; '
                     'it is an arithmetic walk, not a ledger run, and it says nothing '
                     'about blocker classes other than corporate actions')}


def non_blocking_rows(frames, sheets, union):
    """Held-day unapproved rows the exchange never implemented.

    Kept separate because they are the value-manufacturing trap: a row that
    declares a share increase on a session whose reference price did not move
    would double the holder's shares for free if it were ever approved.
    """
    out = []
    for (code, ex) in sorted(sheets, key=lambda pair: pair[1]):
        if (code, ex) not in union:
            continue
        if action_required(frames, code, ex) is False:
            out.append({'code': code, 'ex_date': ex,
                        'buckets': sorted(sheets[(code, ex)]['buckets']),
                        'why': 'adjustment panel shows no implementation on this session, so '
                               'the ledger raises no failure here; approving the row would '
                               'credits shares or cash the exchange did not create'})
    return out


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_digest(directory):
    """One hash over the sorted (name, sha256) pairs of a directory."""
    digest = hashlib.sha256()
    for path in sorted(directory.glob('*')):
        if path.is_file():
            digest.update(path.name.encode('utf-8'))
            digest.update(hash_file(path).encode('utf-8'))
    return digest.hexdigest()


class AdjudicationTests(unittest.TestCase):
    """Properties, not frozen readings."""

    def test_solve_reference_recovers_a_twenty_percent_pair(self):
        ref, rate = solve_reference(56.44, 37.62)
        self.assertLessEqual(abs(rate - 0.2), rate_resolution(56.44, 37.62))
        self.assertAlmostEqual(ref, 47.03, places=2)

    def test_solve_reference_recovers_a_five_percent_pair(self):
        ref, rate = solve_reference(8.55, 7.73)
        self.assertLessEqual(abs(rate - 0.05), rate_resolution(8.55, 7.73))
        self.assertAlmostEqual(ref, 8.14, places=2)

    def test_solve_reference_refuses_a_degenerate_pair(self):
        self.assertEqual(solve_reference(10.0, 10.0), (None, None))
        self.assertEqual(solve_reference(None, 5.0), (None, None))

    def test_limit_average_is_within_half_a_tick_of_the_true_reference(self):
        for ref, rate in ((47.0288, 0.2), (5.47, 0.1), (8.14, 0.05), (41.1, 0.2)):
            solved, solved_rate = solve_reference(round(ref * (1 + rate), 2),
                                                  round(ref * (1 - rate), 2))
            self.assertLessEqual(abs(solved - ref), TICK)

    def test_consensus_names_the_disagreeing_channel(self):
        readings = channel_readings(11.0, 3.755, 3.9441, 8.55, 7.73, 8.14)
        result = consensus(readings)
        self.assertEqual(result['reason'], 'channels_disagree')
        self.assertIn('adjustment_ratio', result['disagreeing'])
        self.assertIn('limit_pair', result['agreeing'])

    def test_consensus_accepts_a_consistent_triple(self):
        readings = channel_readings(66.3, 1.633, 2.3021, 56.44, 37.62, 47.03)
        result = consensus(readings)
        self.assertEqual(result['reason'], 'channels_agree')
        self.assertEqual(len(result['agreeing']), 3)

    def test_no_entitlement_branch_requires_an_unchanged_reference(self):
        readings = channel_readings(1.45, 7.43, 7.43, 1.52, 1.38, 1.45)
        result = consensus(readings)
        outcome = adjudicate({'code': '000656.SZ', 'ex_date': '2025-09-12',
                              'declared_total': 1.0, 'declared_cash': None},
                             readings, result)
        self.assertEqual(outcome['verdict'], 'no_entitlement_reference_equals_prior_close')
        self.assertEqual(outcome['action']['share_multiplier'], 1.0)
        self.assertEqual(outcome['action']['cash_per_share_net'], 0.0)

    def test_accepted_events_preserve_the_ex_date_value_exactly(self):
        prior, ref, rate, cash = 66.3, 47.03, 0.4, 66.3 - 47.03 * 1.4
        self.assertAlmostEqual((1 + rate) * ref + cash, prior, places=10)

    def test_negative_implied_cash_falls_back_to_the_declared_cash(self):
        readings = channel_readings(57.48, 3.6352, 5.084, 49.32, 32.88, 41.1)
        result = consensus(readings)
        outcome = adjudicate({'code': '300806.SZ', 'ex_date': '2026-07-27',
                              'declared_total': 0.4, 'declared_cash': 0.015,
                              'declared_net': 0.015}, readings, result)
        self.assertEqual(outcome['verdict'], 'reference_reconstructed')
        self.assertEqual(outcome['detail']['direction'], 'keep_declared_cash')
        self.assertLess(outcome['detail']['rate'], 0.4)

    def test_reclassification_guard_refuses_a_ninety_percent_move(self):
        readings = channel_readings(11.0, 3.755, 3.9441, 8.55, 7.73, 8.14)
        result = consensus(readings)
        result = {**result, 'reason': 'channels_agree', 'disagreeing': []}
        outcome = adjudicate({'code': '002713.SZ', 'ex_date': '2025-12-30',
                              'declared_total': 1.26775047, 'declared_cash': 0.0},
                             readings, result)
        self.assertEqual(outcome['verdict'], 'refused_reclassification_too_large')

    def test_an_event_without_a_reference_field_is_refused_not_guessed(self):
        readings = channel_readings(10.0, 1.0, 2.0, 5.0, 4.0, None)
        outcome = adjudicate({'code': 'X', 'ex_date': '2026-01-01',
                              'declared_total': 0.0, 'declared_cash': None},
                             readings, consensus(readings))
        self.assertEqual(outcome['verdict'], 'refused_no_local_reference_channel')

    def test_zero_cash_is_not_read_as_a_missing_cash(self):
        readings = channel_readings(6.08, 54.7793, 60.8881, 6.02, 4.92, 5.47)
        result = consensus(readings)
        outcome = adjudicate({'code': '600256.SH', 'ex_date': '2025-07-18',
                              'declared_total': 0.0, 'declared_cash': 0.622},
                             readings, result)
        self.assertGreater(outcome['action']['cash_per_share_net'], 0.6)


def run():
    days = sessions()
    per_arm = held_day_map(days)
    union = set().union(*[entry['pairs'] for entry in per_arm.values()])
    sheets, sealed_meta = load_events(union)
    sealed_rows, files = load_sealed_rows()
    sealed_rows = group_sealed_rows(sealed_rows)
    declared = declared_components(sealed_rows)
    frames = price_channels()

    # The adjudicated set is the unapproved rows that sit on a held day, not every
    # held-day ex-date: the inventory already approves eight thousand five hundred
    # and ninety-five events, and those are not blockers.
    held_events = {(code, ex) for (code, ex) in sheets if (code, ex) in union}

    results = []
    for code, ex in sorted(held_events, key=lambda pair: pair[1]):
        components = declared.get((code, ex), {})
        readings = read_one(frames, code, ex)
        cs = consensus(readings)
        required = action_required(frames, code, ex)
        event = {'code': code, 'ex_date': ex, 'rows': components.get('sealed_rows'),
                 'buckets': sorted(sheets.get((code, ex), {}).get('buckets', [])),
                 'arms': sorted({arm for arm, entry in per_arm.items()
                                 if (code, ex) in entry['pairs']}),
                 **{k: components.get(k) for k in
                    ('declared_cash', 'declared_net', 'declared_total',
                     'record_date', 'pay_date', 'new_shares_date')}}
        outcome = adjudicate(event, readings, cs)
        outcome['action_required_on_held_day'] = required
        outcome['blocks_a_held_day'] = bool(required)
        # Provenance: which sealed response file carried the vendor row.
        outcome['sealed_sources'] = sorted({row['source'] for row in sealed_rows.get((code, ex), [])})
        outcome['sealed_rows_read'] = len(sealed_rows.get((code, ex), []))
        results.append(outcome)

    resolved = {(item['code'], item['ex_date']) for item in results
                if item['verdict'] in ('reference_reconstructed',
                                       'no_entitlement_reference_equals_prior_close')}
    refused = {(item['code'], item['ex_date']) for item in results
               if item['verdict'].startswith('refused')}
    blocking = {(item['code'], item['ex_date']) for item in results if item['blocks_a_held_day']}
    walk = {arm: arm_walk(arm, per_arm, blocking, resolved, refused) for arm in ARMS}

    for arm, entry in walk.items():
        checkpoint = json.loads((CHECKPOINTS / (arm + '.json')).read_text(encoding='utf-8'))
        expected = CHECKPOINT_BLOCKERS[arm]
        blockers = [(item['code'], item['date']) for item in entry['walk']]
        assert blockers and blockers[0] == expected, (arm, blockers[:3], expected)
        entry['reproduces_checkpoint_blocker'] = {
            'arm': arm, 'code': expected[0], 'date': expected[1],
            'checkpoint_end': checkpoint['end'],
            'checkpoint_finite_days': checkpoint['finite_days'],
            'checkpoint_error_reasons': checkpoint.get('error_reasons')}
        entry['resolved_before_it'] = []
        entry['remaining_blockers'] = [item for item in entry['walk']
                                       if item['kind'] != 'event_resolved']

    payload = {
        'scope': 'unapproved held-day corporate-action rows adjudicated through three local '
                 'published channels; no collection, no adjudication change to the sealed '
                 'inventory',
        'window': [WINDOW_LO, WINDOW_HI],
        'arms': ARMS,
        'response_files_read': len(files),
        'sealed_inventory': sealed_meta,
        'unapproved_held_day_rows': len(results),
        'blocking_rows': len(blocking),
        'non_blocking_rows': non_blocking_rows(frames, sheets, union),
        'held_day_absent_price_rows': {arm: len(rows)
                                       for arm, rows in missing_row_blockers(per_arm).items()},
        'verdict_counts': _counts(item['verdict'] for item in results),
        'results': results,
        'arm_walk': walk,
        'bounds': {'limit_agreement_ticks': LIMIT_AGREEMENT_TICKS,
                   'adjustment_relative_bound': ADJ_REL_BOUND,
                   'reclassification_limit': RECLASS_LIMIT,
                   'tick': TICK},
        'channel_authority': ['pre_close field on the ex-date row: the exchange publishes '
                              'the ex-date reference price, so this channel is the '
                              'reference itself rather than an inversion of it',
                              'stk_limit pair on the ex-date: recovers the reference to '
                              'within half a tick and the limit rate without assuming a board',
                              'adj_factor ratio across the ex-date: a vendor series, ranked '
                              'below the two exchange publications and falsifiable by them'],
        'ledger_blocker_condition': ('the ledger raises missing_corporate_action_event only '
                                     'when the adjustment panel shows an implementation on '
                                     'that session and the code has no approved action; an '
                                     'unapproved row on an unadjusted session is carried, '
                                     'not counted'),
        'claims_not_made': [
            'the cash-versus-share split is not recovered, only one value-preserving pair '
            'inside the reclassification bound is chosen and disclosed',
            'a refusal is not evidence that the vendor row is wrong; it is evidence that '
            'the local channels cannot separate it',
            'the arm bounds are arithmetic walks over held days, not a ledger run, and they '
            'are silent about blocker classes other than corporate actions',
            'the tax factor is carried from the vendor row and is not re-derived; all '
            'held-day rows carry cash_div equal to cash_div_tax, so the reconstruction '
            'is optimistic for a taxable account and that is disclosed, not corrected',
            'no production default, feedback state or candidate queue is touched',
        ],
        'formal_eligibility': False,
        'production_unchanged': True,
        'adjudication_changed': False,
        'hashes': {
            'data/raw/daily_prices.csv': hash_file(ROOT / 'data/raw/daily_prices.csv'),
            'data/raw/adj_factor_daily.csv': hash_file(ROOT / 'data/raw/adj_factor_daily.csv'),
            'data/raw/stk_limit.csv': hash_file(ROOT / 'data/raw/stk_limit.csv'),
            'reports/event_inventory_20260923/inventory.json': hash_file(SEALED),
            'reports/ledger_advance_20260923/checkpoints': _dir_digest(CHECKPOINTS),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'adjudication.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str),
        encoding='utf-8', newline='\n')
    _write_csv(OUT / 'adjudication.csv', results)
    _write_actions(payload)
    return payload


def _write_actions(payload):
    """Export the reconstructed actions as a separate overlay, never as an approval.

    Two exclusions are deliberate. The no-entitlement row is not exported because
    its session shows no implementation in the adjustment panel, so the ledger
    raises no failure there and exporting a no-op action would only switch off the
    ledger's own reference check on that session. Refused rows are not exported
    because a refusal is a result.
    """
    actions = [item['action'] for item in payload['results']
               if item['verdict'] == 'reference_reconstructed']
    exported = {(item['code'], item['ex_date']) for item in payload['results']
                if item['verdict'] == 'reference_reconstructed'}
    skipped = [{'code': item['code'], 'ex_date': item['ex_date'], 'verdict': item['verdict'],
                'why': ('session shows no implementation, so the row blocks nothing and a '
                        'no-op action would disable the ledger reference check'
                        if item['verdict'].startswith('no_entitlement') else
                        'refused: the local channels cannot separate the entitlement')}
               for item in payload['results'] if (item['code'], item['ex_date']) not in exported]
    assert len(actions) == len(exported)
    for action in actions:
        assert action['event_id'].startswith('adj_')
        assert action['cash_per_share_net'] >= 0
        assert action['share_multiplier'] >= 1.0
    (OUT / 'adjudicated_actions.json').write_text(
        json.dumps({'actions': actions, 'not_exported': skipped,
                    'adjudication_changed': False, 'formal_eligibility': False,
                    'scope': ('an overlay for a shadow replay, not an amendment to the sealed '
                              'inventory: every exported row keeps its sealed rejected or '
                              'unresolved status'),
                    'source_verdicts': {item['code'] + ' ' + item['ex_date']: item['verdict']
                                        for item in payload['results']},
                    'hashes': payload['hashes']},
                   ensure_ascii=False, indent=1, default=str),
        encoding='utf-8', newline='\n')


def read_one(frames, code, ex_date):
    prices, adj, limits = frames['prices'], frames['adj'], frames['limits']
    row = prices[(prices.ts_code == code) & (prices.trade_date == ex_date)]
    prior = row.iloc[0]['prior_close'] if not row.empty else None
    field = row.iloc[0]['pre_close'] if not row.empty else None
    leg = adj[(adj.ts_code == code) & (adj.trade_date == ex_date)]
    adj_prev = leg.iloc[0]['prior_adj'] if not leg.empty else None
    adj_ex = leg.iloc[0]['adj_factor'] if not leg.empty else None
    lim = limits[(limits.ts_code == code) & (limits.trade_date == ex_date)]
    up = lim.iloc[0]['up_limit'] if not lim.empty else None
    down = lim.iloc[0]['down_limit'] if not lim.empty else None
    return channel_readings(prior, adj_prev, adj_ex, up, down, field)


def action_required(frames, code, ex_date):
    """Whether the adjustment panel shows an implementation on that session.

    This is the ledger's own blocker condition, read from the same panel with the
    same relative tolerance, and it is the difference between an event that
    blocks a day and a vendor row that merely sits on one.
    """
    adj = frames['adj']
    hit = adj[(adj.ts_code == code) & (adj.trade_date == ex_date)]
    if hit.empty:
        return None
    prev, cur = num(hit.iloc[0]['prior_adj']), num(hit.iloc[0]['adj_factor'])
    if prev is None or cur is None:
        return None
    return abs(cur - prev) > 1e-7 * max(abs(cur), abs(prev))


def _counts(values):
    out = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _write_csv(path, results):
    import csv
    columns = ['code', 'ex_date', 'arms', 'rows', 'verdict', 'declared_cash',
               'declared_total', 'reference', 'prior_close', 'limit_gap_ticks',
               'adjustment_gap', 'cash_implied', 'rate_implied', 'direction',
               'reclassification', 'cash_per_share_net', 'share_multiplier']
    with io.open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(columns)
        for item in results:
            detail = item.get('detail') or {}
            action = item.get('action') or {}
            cs = item.get('consensus') or {}
            writer.writerow([
                item['code'], item['ex_date'], '|'.join(item.get('arms') or []),
                item.get('rows'), item['verdict'], item.get('declared_cash'),
                item.get('declared_total'), cs.get('reference'), cs.get('prior_close'),
                _round(cs.get('limit_gap_ticks')), _round(cs.get('adjustment_gap'), 8),
                _round(detail.get('cash_implied')), _round(detail.get('rate_implied'), 8),
                detail.get('direction'), _round(detail.get('reclassification'), 8),
                _round(action.get('cash_per_share_net')), _round(action.get('share_multiplier'), 8),
            ])


def _round(value, places=6):
    return None if value is None else round(float(value), places)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--tests', action='store_true')
    args = parser.parse_args()
    if args.tests:
        suite = unittest.TestLoader().loadTestsFromTestCase(AdjudicationTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if not args.run:
        raise SystemExit('pass --run or --tests')
    for name in ('adjudication.json', 'adjudication.csv'):
        if (OUT / name).exists():
            raise SystemExit('refusing to overwrite %s' % (OUT / name))
    payload = run()
    print(json.dumps({'unapproved_held_day_rows': payload['unapproved_held_day_rows'],
                      'blocking_rows': payload['blocking_rows'],
                      'verdict_counts': payload['verdict_counts'],
                      'non_blocking_rows': [r['code'] + ' ' + r['ex_date']
                                            for r in payload['non_blocking_rows']],
                      'sealed_bucket_sizes': payload['sealed_inventory']['stored_bucket_sizes'],
                      'unresolved_absent_from_file':
                          payload['sealed_inventory']['unresolved_rows_absent_from_the_file'],
                      'arm_bounds': {arm: entry['bound_end']
                                     for arm, entry in payload['arm_walk'].items()},
                      'arm_first_remaining': {arm: entry['first_remaining_blocker']
                                              for arm, entry in payload['arm_walk'].items()},
                      'held_day_absent_price_rows': payload['held_day_absent_price_rows']},
                     ensure_ascii=False, indent=1))
    for item in payload['results']:
        print('%-11s %s %-42s ref=%-8s prior=%-8s %s' % (
            item['code'], item['ex_date'], item['verdict'],
            _round((item['consensus'] or {}).get('reference')),
            _round((item['consensus'] or {}).get('prior_close')),
            json.dumps(item.get('detail') or {}, ensure_ascii=False, default=str)))


if __name__ == '__main__':
    main()
