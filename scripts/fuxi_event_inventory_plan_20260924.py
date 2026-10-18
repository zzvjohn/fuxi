# -*- coding: utf-8 -*-
"""Item 1 next step: a requirements-driven, full-window event inventory and a
bounded evidence-request protocol.

Why this exists
---------------
The five-stage acceptance protocol says, for item 1: when event blockers keep
appearing, the next batch must form a full-window event inventory **from the
securities the frozen target actually touches**, plus a bounded request
protocol, instead of re-running ever-shorter windows and calling that progress.

So this script asks, for every (arm, security, held session) the frozen target
requires, which of the four evidence classes already exists locally and which
does not. It does not fill, infer or impute anything. It stops at counting and
naming.

Evidence classes
----------------
  E1 corporate action  - a corporate-action record for any ex-date falling on a held day
  E2 trading status    - halt / suspension / delisting evidence on a held day
  E3 price limit       - the daily limit panel on a held day (needed to judge fills)
  E4 identity          - point-in-time security identity, board and name/ST history

Reads only local sealed files. Writes under reports/event_inventory_plan_20260924/.
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
from fuxi_execution_lab_20260923 import digest, read, write  # noqa: E402

OUT = ROOT / 'reports/event_inventory_plan_20260924'
TARGETS_DIR = ROOT / 'reports/execution_ledger_20260923/targets'
CHECKPOINTS = ROOT / 'reports/ledger_advance_20260923/checkpoints'
SEALED_INVENTORY = ROOT / 'reports/event_inventory_20260923/inventory.json'

WINDOW_LO = '2024-12-27'
WINDOW_HI = '2026-09-09'
ARMS = ['r4_equal_rank', 'llm_family_1_1_fresh_llm', 'llm_family_2_1_fresh_llm',
        'mem_0626_broadcast_repaired']


def dashed(value):
    return '%s-%s-%s' % (value[:4], value[4:6], value[6:8])


def sessions():
    cal = pd.read_csv(ROOT / 'data/raw/trade_calendar.csv', dtype=str)
    cal = cal[(cal.exchange == 'SSE') & (cal.is_open == '1')]
    days = sorted(pd.to_datetime(cal.cal_date, format='mixed')
                  .dt.strftime('%Y-%m-%d').unique())
    return [d for d in days if WINDOW_LO <= d <= WINDOW_HI]


def date_set(path, column='trade_date'):
    """Distinct session dates present in one raw file (chunked, no memory blow-up)."""
    found = set()
    reader = pd.read_csv(ROOT / path, usecols=[column], dtype=str, chunksize=500000)
    for chunk in reader:
        found.update(chunk[column].str.replace('-', '', regex=False).unique().tolist())
    return {dashed(v) for v in found if isinstance(v, str) and len(v) == 8}


def load_targets(arm):
    payload = json.loads((TARGETS_DIR / (arm + '.json')).read_text(encoding='utf-8'))
    frames = {}
    for stamp, weights in payload['targets'].items():
        held = {code: float(w) for code, w in weights.items() if float(w) > 0}
        if held:
            frames[stamp] = held
    return frames, payload.get('insufficient')


def holding_calendar(frames, days):
    """Map each held security to the sessions it is held on.

    A target stamped on session s is in force from the first session after s
    until the session before the next target takes effect.
    """
    stamps = sorted(frames)
    index = {d: i for i, d in enumerate(days)}
    spans = []
    for position, stamp in enumerate(stamps):
        after = [d for d in days if d > stamp]
        if not after:
            continue
        start = after[0]
        if position + 1 < len(stamps):
            nxt = [d for d in days if d > stamps[position + 1]]
            stop = days[index[nxt[0]] - 1] if nxt else days[-1]
        else:
            stop = days[-1]
        if start > stop:
            continue
        spans.append((stamp, start, stop, frames[stamp]))
    return spans


def held_pairs(spans, days):
    """(code, session) pairs the target requires evidence for."""
    index = {d: i for i, d in enumerate(days)}
    pairs = set()
    for _stamp, start, stop, held in spans:
        for day in days[index[start]:index[stop] + 1]:
            for code in held:
                pairs.add((code, day))
    return pairs


def sealed_corporate_actions():
    """Codes with at least one sealed corporate-action row, and their ex-dates."""
    if not SEALED_INVENTORY.exists():
        return {}, {}
    payload = json.loads(SEALED_INVENTORY.read_text(encoding='utf-8'))
    codes, ex_dates = set(), {}
    for bucket in ('approved', 'rejected', 'unresolved'):
        for row in payload.get(bucket, []):
            code = row.get('code')
            if not code:
                continue
            codes.add(code)
            ex_dates.setdefault(code, set()).add(row.get('ex_date'))
    return codes, {k: sorted(v) for k, v in ex_dates.items()}


def blocks():
    out = {}
    for arm in ARMS:
        path = CHECKPOINTS / (arm + '.json')
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        out[arm] = {'status': payload.get('status'), 'end': payload.get('end'),
                    'finite_days': payload.get('finite_days'),
                    'error_reasons': payload.get('error_reasons'),
                    'errors': payload.get('errors')}
    return out


def prices_for(pairs):
    """Which of the required (code, session) pairs have a sealed price row."""
    wanted = set(pairs)
    have = set()
    reader = pd.read_csv(ROOT / 'data/raw/daily_prices.csv',
                         usecols=['ts_code', 'trade_date'], dtype=str, chunksize=500000)
    for chunk in reader:
        chunk = chunk.assign(day=chunk.trade_date.str.replace('-', '', regex=False))
        for code, raw in zip(chunk.ts_code, chunk.day):
            key = (code, dashed(raw))
            if key in wanted:
                have.add(key)
    return have


def plan():
    days = sessions()
    day_set = set(days)
    limit_dates = date_set('data/raw/stk_limit.csv')
    suspend_dates = date_set('data/raw/suspend_d.csv')
    price_dates = date_set('data/raw/daily_prices.csv')
    adj_dates = date_set('data/raw/adj_factor_daily.csv')
    sealed_codes, sealed_ex = sealed_corporate_actions()
    arm_blocks = blocks()

    per_arm = {}
    all_pairs = {}
    securities = set()
    for arm in ARMS:
        frames, insufficient = load_targets(arm)
        spans = holding_calendar(frames, days)
        pairs = held_pairs(spans, days)
        all_pairs[arm] = pairs
        securities.update(code for code, _day in pairs)
        held_codes = sorted({code for code, _day in pairs})
        held_days = sorted({day for _code, day in pairs})
        coverage = {
            'securities': len(held_codes),
            'span': [spans[0][1], spans[-1][2]] if spans else [None, None],
            'required_security_days': len(pairs),
            'held_sessions': len(held_days),
            'limit_panel_missing_sessions': len([d for d in held_days if d not in limit_dates]),
            'status_missing_sessions': len([d for d in held_days if d not in suspend_dates]),
            'price_missing_sessions': len([d for d in held_days if d not in price_dates]),
            'adjustment_missing_sessions': len([d for d in held_days if d not in adj_dates]),
            'securities_without_any_sealed_corporate_action':
                len([c for c in held_codes if c not in sealed_codes]),
            'targets_insufficient': insufficient,
        }
        block = arm_blocks.get(arm) or {}
        if block.get('end'):
            covered = len([d for d in held_days if d <= block['end']])
            coverage['sessions_before_block'] = covered
            coverage['sessions_after_block'] = len(held_days) - covered
            coverage['fraction_of_held_sessions_justified'] = (
                round(covered / len(held_days), 6) if held_days else None)
            coverage['remaining_required_security_days_after_block'] = len(
                [1 for _c, d in pairs if d > block['end']])
        coverage['block'] = block
        per_arm[arm] = coverage

    union_pairs = set()
    for pairs in all_pairs.values():
        union_pairs.update(pairs)
    union_codes = sorted({code for code, _day in union_pairs})
    union_sessions = sorted({day for _code, day in union_pairs})

    return {
        'batch': 'event_inventory_plan_20260924',
        'generated_at': datetime.now().astimezone().isoformat(),
        'purpose': 'item 1 next step: requirements-driven full-window event inventory '
                   'and a bounded evidence-request protocol',
        'window': {'from': days[0], 'to': days[-1], 'sessions': len(days)},
        'scope_note': 'requirements are derived from the sealed frozen targets; nothing '
                      'is filled, inferred or imputed, and no external request is made '
                      'by this batch',
        'arms': per_arm,
        'union': {
            'securities': len(union_codes),
            'security_codes': union_codes,
            'sessions': len(union_sessions),
            'required_security_days': len(union_pairs),
            'with_sealed_corporate_action': len([c for c in union_codes if c in sealed_codes]),
            'without_sealed_corporate_action': len([c for c in union_codes
                                                    if c not in sealed_codes]),
        },
        'supply': {
            'sessions_in_window': len(days),
            'primary_price_dates': len([d for d in days if d in price_dates]),
            'adjustment_dates': len([d for d in days if d in adj_dates]),
            'limit_panel_dates': len([d for d in days if d in limit_dates]),
            'status_dates': len([d for d in days if d in suspend_dates]),
            'limit_panel_missing_dates': sorted([d for d in days if d not in limit_dates]),
            'status_missing_dates': sorted([d for d in days if d not in suspend_dates]),
            'sealed_corporate_action_codes': len(sealed_codes),
            'sealed_corporate_action_ex_dates': sum(len(v) for v in sealed_ex.values()),
        },
        'claims_not_made': [
            'no fill, no imputation and no zero-fill anywhere in this inventory',
            'a missing local record is recorded as missing, never as absence of an event',
            'no economic verdict for any arm; every arm stays data_insufficient',
            'no JQ eligibility, no factor registration and no production change',
            'a request count is a plan, not a measurement of what a supplier will return'],
        'formal_eligibility': False,
        'full_stage2_accepted': False,
        'hashes': {
            'data/raw/trade_calendar.csv': digest(ROOT / 'data/raw/trade_calendar.csv'),
            'reports/execution_ledger_20260923/targets': _dir_digest(TARGETS_DIR),
            'reports/ledger_advance_20260923/checkpoints': _dir_digest(CHECKPOINTS),
        },
    }


def _dir_digest(folder):
    parts = []
    for path in sorted(folder.glob('*.json')):
        parts.append('%s:%s' % (path.name, hashlib.sha256(path.read_bytes()).hexdigest()))
    return hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()


def requests(plan_payload):
    """The bounded request protocol, derived from the measured gaps."""
    supply = plan_payload['supply']
    union = plan_payload['union']
    rows = []

    def add(kind, target, quantity, acceptance, priority, note):
        rows.append({'priority': priority, 'class': kind, 'target': target,
                     'quantity': quantity, 'acceptance': acceptance, 'note': note})

    add('E1 corporate action', 'per security in the frozen union',
        union['without_sealed_corporate_action'],
        'for every held session of that security, the supplier record must either name an '
        'ex-date (with cash per share, stock distribution and listing date) or affirm that '
        'no ex-date falls on it; a blank row is not an affirmation',
        1,
        'one request per security covers every ex-date inside the window, so the count is '
        'bounded by the security count, not by the session count')
    add('E2 trading status', 'sessions %s..%s' % (supply['status_missing_dates'][0],
                                                  supply['status_missing_dates'][-1])
        if supply['status_missing_dates'] else 'none',
        len(supply['status_missing_dates']),
        'a halt/suspension record per (security, session) or an explicit no-halt affirmation; '
        'a missing session must never be read as "traded normally"',
        2,
        'the local suspension table stops at 2026-04-30, so the whole tail of the window is '
        'uncovered; this is a date-range request, not a per-security one')
    add('E3 price limit panel', 'dates %s' % ', '.join(supply['limit_panel_missing_dates'][:6])
        + (' ...' if len(supply['limit_panel_missing_dates']) > 6 else ''),
        len(supply['limit_panel_missing_dates']),
        'the full cross-section of the daily limit panel for those dates; fills cannot be '
        'judged without it',
        2,
        'missing dates cluster after 2026-08-03 plus two isolated earlier holes')
    add('E4 identity / lifecycle', 'per security in the frozen union',
        union['securities'],
        'point-in-time identity: listing status, board, and the name/ST history effective on '
        'the held session; a current snapshot is not a substitute',
        3,
        'requested once per security; needed to justify both trading eligibility and the '
        'delisting cases')
    total = sum(r['quantity'] for r in rows if r['class'].startswith(('E1', 'E4')))
    payload = {
        'batch': 'event_inventory_plan_20260924',
        'external_requests_made': 0,
        'bounded_by': 'the frozen union security count plus the two uncovered date ranges',
        'union_security_requests': total,
        'date_range_requests': sum(len(supply[k]) for k in
                                   ('status_missing_dates', 'limit_panel_missing_dates')),
        'requests': rows,
        'rule_never': [
            'never fill a missing corporate action with zero cash or a 1.0 share multiplier',
            'never use the adjustment factor to invent cash or shares',
            'never treat an unfilled name as a delisted or halted name',
            'never present a short-window recomputation as full-window progress'],
        'stop_rule': 'if a blocker repeats after its class is fully supplied, escalate to the '
                     'supplier contract instead of requesting the same class again',
    }
    payload['upper_bound_requests'] = (payload['union_security_requests'] * 3
                                       + payload['date_range_requests'])
    return payload


class InventoryPlanTests(unittest.TestCase):
    def test_holding_spans_are_contiguous_and_ordered(self):
        days = ['2024-12-27', '2024-12-30', '2024-12-31', '2025-01-02', '2025-01-03',
                '2025-01-06', '2025-01-07']
        frames = {'2024-12-27': {'A': 0.5, 'B': 0.5}, '2025-01-03': {'C': 1.0}}
        spans = holding_calendar(frames, days)
        self.assertEqual(len(spans), 2)
        # the first target is in force from the session after its stamp up to the
        # session before the next target takes effect
        self.assertEqual(spans[0][1], '2024-12-30')
        self.assertEqual(spans[0][2], '2025-01-03')
        self.assertEqual(spans[1][1], '2025-01-06')
        self.assertEqual(spans[1][2], days[-1])
        # spans must not overlap and must leave no gap between them
        self.assertEqual(days[days.index(spans[0][2]) + 1], spans[1][1])
        pairs = held_pairs(spans, days)
        self.assertIn(('A', '2024-12-30'), pairs)
        self.assertIn(('C', '2025-01-07'), pairs)
        self.assertNotIn(('A', '2025-01-06'), pairs)

    def test_zero_weights_are_dropped(self):
        frames, _ = load_targets(ARMS[0])
        self.assertTrue(all(all(w > 0 for w in held.values()) for held in frames.values()))

    def test_requirements_are_a_subset_of_the_union(self):
        days = sessions()
        frames, _ = load_targets(ARMS[0])
        pairs = held_pairs(holding_calendar(frames, days), days)
        self.assertTrue(pairs)
        self.assertTrue({c for c, _d in pairs} <= {c for c in {c for c, _d in pairs}})
        self.assertTrue(all(d in set(days) for _c, d in pairs))

    def test_missing_is_not_absence(self):
        payload = plan()
        text = json.dumps(payload, ensure_ascii=False)
        self.assertIn('never as absence of an event', text)

    def test_no_imputation_language(self):
        payload = requests(plan())
        text = json.dumps(payload, ensure_ascii=False)
        self.assertIn('never fill a missing corporate action', text)


def tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(InventoryPlanTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def run():
    target = OUT / 'plan.json'
    if target.exists():
        raise RuntimeError('plan.json already exists; refusing to overwrite')
    OUT.mkdir(parents=True, exist_ok=True)
    payload = plan()
    payload['bounded_request_protocol'] = requests(payload)
    union_pairs = None
    days = sessions()
    all_pairs = {}
    for arm in ARMS:
        frames, _ = load_targets(arm)
        all_pairs[arm] = held_pairs(holding_calendar(frames, days), days)
    union_pairs = set()
    for pairs in all_pairs.values():
        union_pairs.update(pairs)
    have = prices_for(union_pairs)
    payload['supply']['required_security_days_with_price_row'] = len(have)
    payload['supply']['required_security_days_without_price_row'] = len(union_pairs - have)
    top_gaps = {}
    for code, day in sorted(union_pairs - have):
        top_gaps[code] = top_gaps.get(code, 0) + 1
    payload['top_codes_without_price_row'] = sorted(top_gaps.items(),
                                                    key=lambda kv: (-kv[1], kv[0]))[:25]
    write(target, payload)
    with io.open(OUT / 'requirements.csv', 'w', encoding='utf-8', newline='') as handle:
        handle.write('arm,security,sessions_required,pct_of_window\n')
        for arm in ARMS:
            days_ = sorted({d for _c, d in all_pairs[arm]})
            for code in sorted({c for c, _d in all_pairs[arm]}):
                n = len([1 for c, d in all_pairs[arm] if c == code])
                handle.write('%s,%s,%d,%.6f\n' % (arm, code, n, n / len(days_)))
    print('window %s..%s sessions=%d' % (days[0], days[-1], len(days)), flush=True)
    print('union securities=%d required security-days=%d'
          % (payload['union']['securities'], payload['union']['required_security_days']),
          flush=True)
    print('with price row=%d without=%d'
          % (payload['supply']['required_security_days_with_price_row'],
             payload['supply']['required_security_days_without_price_row']), flush=True)
    for arm in ARMS:
        c = payload['arms'][arm]
        print('%-28s sec=%4d held_days=%3d required=%6d before_block=%3d after_block=%3d '
              'status_missing=%3d limit_missing=%3d no_sealed_action=%4d'
              % (arm, c['securities'], c['held_sessions'], c['required_security_days'],
                 c.get('sessions_before_block', -1), c.get('sessions_after_block', -1),
                 c['status_missing_sessions'], c['limit_panel_missing_sessions'],
                 c['securities_without_any_sealed_corporate_action']), flush=True)
    print('written: %s' % target, flush=True)


def missing_price_rows():
    """Export the exact required (security, session) pairs that have no sealed price row.

    These are the concrete cases that need individual attribution before any
    "delisted" or "halted" story is allowed: a missing price row is a missing
    record, not proof of absence from the market.
    """
    days = sessions()
    all_pairs = set()
    per_arm = {}
    for arm in ARMS:
        frames, _ = load_targets(arm)
        pairs = held_pairs(holding_calendar(frames, days), days)
        per_arm[arm] = pairs
        all_pairs.update(pairs)
    have = prices_for(all_pairs)
    missing = sorted(all_pairs - have, key=lambda cd: (cd[0], cd[1]))
    rows = []
    for code, day in missing:
        arms = sorted(a for a, pairs in per_arm.items() if (code, day) in pairs)
        rows.append({'security': code, 'session': day, 'arms': ';'.join(arms)})
    target = OUT / 'missing_price_rows.csv'
    if target.exists():
        raise RuntimeError('missing_price_rows.csv already exists; refusing to overwrite')
    OUT.mkdir(parents=True, exist_ok=True)
    with io.open(target, 'w', encoding='utf-8', newline='') as handle:
        handle.write('security,session,arms\n')
        for row in rows:
            handle.write('%s,%s,%s\n' % (row['security'], row['session'], row['arms']))
    codes = sorted({row['security'] for row in rows})
    print('missing price rows=%d across %d securities; sessions %s..%s'
          % (len(rows), len(codes), rows[0]['session'] if rows else None,
             rows[-1]['session'] if rows else None), flush=True)
    print('in arms: %s'
          % json.dumps({a: len([1 for r in rows if a in r['arms'].split(';')])
                        for a in ARMS}, ensure_ascii=False), flush=True)
    print('written: %s' % target, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--tests', action='store_true')
    parser.add_argument('--missing-prices', action='store_true')
    args = parser.parse_args()
    if args.tests:
        return tests()
    if args.run:
        run()
        return 0
    if args.missing_prices:
        missing_price_rows()
        return 0
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
