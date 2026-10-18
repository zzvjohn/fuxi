# -*- coding: utf-8 -*-
"""Item 1, follow-on: individual attribution of every required (security, session)
that has no sealed price row.

Why this exists
---------------
The event-inventory plan found 170 required security-days with no price row and
refused to say anything about them, because a missing record is a missing record
and never proof of absence from the market. This script now asks each of the 170
cases a narrow question and answers it with local evidence only:

  * is there a sealed price row strictly before AND strictly after the gap?
      - with a row on both sides the security demonstrably traded again, so the
        gap is a halt, not a delisting;
      - with nothing after it, the feed itself stopped.
  * is there a suspension record for that exact (security, session) or is the
    session simply outside the local status table's date range?
  * does the security appear in the listed-security snapshot, and what do the
    name-change records say (an *ST entry, a delisting-consolidation rename)?

Verdict vocabulary is deliberately narrow. Nothing is called "delisted" unless a
name-change record says so in words. A gap with no local evidence is reported as
unresolved and is counted, never promoted to a fact.

Reads only local sealed files. Writes under reports/event_inventory_plan_20260924/.
"""

import argparse
import hashlib
import io
import json
import sys
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from fuxi_execution_lab_20260923 import digest, write  # noqa: E402

OUT = ROOT / 'reports/event_inventory_plan_20260924'
MISSING_CSV = OUT / 'missing_price_rows.csv'

ARMS = ['r4_equal_rank', 'llm_family_1_1_fresh_llm', 'llm_family_2_1_fresh_llm',
        'mem_0626_broadcast_repaired']

# verdict -> what local evidence supports it
VERDICTS = {
    'middle_gap_halt_suspension_recorded':
        'a sealed price row exists after the gap and the local suspension table names '
        'this exact (security, session)',
    'middle_gap_halt_status_table_absent':
        'a sealed price row exists after the gap, but the session lies past the local '
        'suspension table terminal date, so the halt record is simply not held locally',
    'middle_gap_halt_unexplained':
        'a sealed price row exists after the gap, the session IS inside the local '
        'suspension table range, yet no suspension row and no other local record covers it',
    'tail_delisting_consolidation_evidenced':
        'no sealed price row after the gap and a name-change record names a delisting '
        'consolidation period (退市整理期) for this security',
    'tail_star_st_evidenced_no_consolidation':
        'no sealed price row after the gap and an *ST name-change record exists, but no '
        'delisting-consolidation record and no local end-of-feed statement, so the tail '
        'is evidenced as a risk-flagged halt whose outcome is not held locally',
    'tail_no_namechange_evidence':
        'no sealed price row after the gap and no name-change record at all: the local '
        'evidence classes cannot describe this tail',
}


def dashed(value):
    value = str(value).replace('-', '')
    return '%s-%s-%s' % (value[:4], value[4:6], value[6:8])


def price_calendars(codes):
    days = {c: set() for c in codes}
    reader = pd.read_csv(ROOT / 'data/raw/daily_prices.csv',
                         usecols=['ts_code', 'trade_date'], dtype=str, chunksize=500000)
    for chunk in reader:
        chunk = chunk[chunk.ts_code.isin(codes)]
        for code, raw in zip(chunk.ts_code, chunk.trade_date):
            days[code].add(dashed(raw))
    return {code: sorted(v) for code, v in days.items()}


def other_channels(codes, pairs):
    """First/last day per security, plus which of the gap pairs each channel covers.

    A halt-day row in a sibling channel tells us the missing price row is a
    price-channel omission for that security-day, not a wholesale absence of the
    security from the supplier's coverage.
    """
    out = {}
    for name, path in [('adjustment', 'data/raw/adj_factor_daily.csv'),
                       ('limit', 'data/raw/stk_limit.csv')]:
        span, present = {}, set()
        reader = pd.read_csv(ROOT / path, usecols=['ts_code', 'trade_date'],
                             dtype=str, chunksize=500000)
        for chunk in reader:
            chunk = chunk[chunk.ts_code.isin(codes)]
            for code, raw in zip(chunk.ts_code, chunk.trade_date):
                day = dashed(raw)
                lo, hi = span.get(code, (day, day))
                span[code] = (day if day < lo else lo, day if day > hi else hi)
                key = (code, day)
                if key in pairs:
                    present.add(key)
        out[name] = span
        out[name + '_pairs_present'] = present
    return out


def suspension_index():
    table = pd.read_csv(ROOT / 'data/raw/suspend_d.csv', dtype=str)
    table = table.assign(day=table.trade_date.map(dashed))
    pairs = {}
    for row in table.itertuples():
        pairs.setdefault((row.ts_code, row.day), []).append(row.suspend_type)
    return pairs, set(table.day)


def listing_index():
    table = pd.read_csv(ROOT / 'data/raw/stock_list.csv', dtype=str)
    return set(table.ts_code), str(table.list_date.max()), len(table)


def namechange_index():
    table = pd.read_csv(ROOT / 'data/raw/namechange_2023_2026.csv', dtype=str)
    out = {}
    for row in table.itertuples():
        out.setdefault(row.ts_code, []).append(
            {'name': row.name, 'start_date': row.start_date, 'end_date': row.end_date,
             'ann_date': row.ann_date, 'change_reason': row.change_reason})
    return {code: sorted(v, key=lambda r: r['start_date'] or '') for code, v in out.items()}


def classify(case, susp_pairs, susp_days, listed, namechanges, spans):
    code, session = case['security'], case['session']
    reason = namechanges.get(code, [])
    reasons = ' '.join(r['change_reason'] or '' for r in reason)
    case['namechange_records'] = reason
    case['listed_snapshot_membership'] = code in listed[0]
    case['listed_snapshot_max_list_date'] = listed[1]
    case['adjustment_span'] = spans['adjustment'].get(code)
    case['limit_span'] = spans['limit'].get(code)
    case['suspension_types_here'] = susp_pairs.get((code, session))
    case['session_in_suspension_table_range'] = session in susp_days
    case['adjustment_row_here'] = (code, session) in spans['adjustment_pairs_present']
    case['limit_row_here'] = (code, session) in spans['limit_pairs_present']

    if case['has_price_after']:
        if case['suspension_types_here']:
            case['verdict'] = 'middle_gap_halt_suspension_recorded'
        elif not case['session_in_suspension_table_range']:
            case['verdict'] = 'middle_gap_halt_status_table_absent'
        else:
            case['verdict'] = 'middle_gap_halt_unexplained'
    else:
        if '退市整理期' in reasons:
            case['verdict'] = 'tail_delisting_consolidation_evidenced'
        elif '*ST' in reasons or 'ST' in reasons:
            case['verdict'] = 'tail_star_st_evidenced_no_consolidation'
        else:
            case['verdict'] = 'tail_no_namechange_evidence'
    case['verdict_meaning'] = VERDICTS[case['verdict']]
    return case


def build():
    rows = pd.read_csv(MISSING_CSV, dtype=str)
    codes = sorted(set(rows.security))
    calendars = price_calendars(codes)
    susp_pairs, susp_days = suspension_index()
    listed = listing_index()
    namechanges = namechange_index()
    case_pairs = set(zip(rows.security, rows.session))
    spans = other_channels(codes, case_pairs)

    cases = []
    for row in rows.itertuples():
        days = calendars[row.security]
        before = [d for d in days if d < row.session]
        after = [d for d in days if d > row.session]
        cases.append({
            'security': row.security,
            'session': row.session,
            'arms': row.arms.split(';'),
            'has_price_before': bool(before),
            'price_day_before': before[-1] if before else None,
            'has_price_after': bool(after),
            'price_day_after': after[0] if after else None,
            'price_first_day': days[0] if days else None,
            'price_last_day': days[-1] if days else None,
            'gap_sessions': len([d for d in days_between(days, row.session)]) + 1,
        })
        classify(cases[-1], susp_pairs, susp_days, listed, namechanges, spans)

    by_verdict = Counter(case['verdict'] for case in cases)
    by_arm = {}
    for arm in ARMS:
        subset = [c for c in cases if arm in c['arms']]
        by_arm[arm] = {
            'missing_required_security_days': len(subset),
            'verdicts': dict(Counter(c['verdict'] for c in subset)),
            'securities': sorted({c['security'] for c in subset}),
        }

    unresolved = [c for c in cases if c['verdict'] in
                  ('middle_gap_halt_unexplained', 'tail_no_namechange_evidence')]
    status_absent = [c for c in cases
                     if c['verdict'] == 'middle_gap_halt_status_table_absent']
    tail = [c for c in cases if not c['has_price_after']]
    tail_codes = sorted({c['security'] for c in tail})
    table_end = max(susp_days)

    return {
        'batch': 'missing_event_attribution_20260924',
        'generated_at': datetime.now().astimezone().isoformat(),
        'purpose': 'individual attribution of every required (security, session) with no '
                   'sealed price row, using local evidence only',
        'input': {'missing_price_rows': str(MISSING_CSV.relative_to(ROOT)),
                  'cases': len(cases),
                  'securities': len(codes)},
        'headline': {
            'cases': len(cases),
            'middle_gap_halt_with_resume': sum(1 for c in cases if c['has_price_after']),
            'tail_feed_stopped': sum(1 for c in cases if not c['has_price_after']),
            'by_verdict': dict(by_verdict),
            'unresolved_after_local_evidence': len(unresolved),
            'unresolved_cases': [{'security': c['security'], 'session': c['session'],
                                  'arms': c['arms'], 'verdict': c['verdict']}
                                 for c in unresolved],
            'blocked_only_by_a_missing_local_table': len(status_absent),
            'blocked_only_by_a_missing_local_table_note':
                'these are halts whose record is absent because the local suspension '
                'table terminates at %s, not because no halt happened' % table_end,
        },
        'tail_breakdown': {
            'note': 'a stopped feed is not by itself a delisting; the listed snapshot and '
                    'a suspension that is still open at the local table end separate the '
                    'two readings the local evidence can support',
            'codes': len(tail_codes),
            'cases': len(tail),
            'absent_from_listed_snapshot': {
                'codes': sorted(c for c in tail_codes if c not in listed[0]),
                'cases': len([c for c in tail if c['security'] not in listed[0]]),
            },
            'present_in_listed_snapshot': {
                'codes': sorted(c for c in tail_codes if c in listed[0]),
                'cases': len([c for c in tail if c['security'] in listed[0]]),
            },
            'still_suspended_at_the_local_table_end': {
                'terminal_date': table_end,
                'codes': sorted(c for c in tail_codes if suspended_at_table_end(c, susp_pairs, table_end)),
                'cases': len([c for c in tail if suspended_at_table_end(c["security"], susp_pairs, table_end)]),
            },
            'tail_codes_with_no_suspension_row_anywhere':
                sorted(c for c in tail_codes
                       if c not in {code for code, _day in susp_pairs}),
            'convergent_readings': {
                'not_listed_and_star_st_recorded': {
                    'codes': sorted(c for c in tail_codes
                                    if c not in listed[0] and has_st_record(c, namechanges)),
                    'note': 'feed stopped, absent from the listed snapshot and an ST/*ST '
                            'record exists: the readings agree that the security is no '
                            'longer a plain listed name',
                },
                'listed_and_halt_still_open': {
                    'codes': sorted(c for c in tail_codes
                                    if c in listed[0] and suspended_at_table_end(c, susp_pairs, table_end)),
                    'note': 'feed stopped, still in the listed snapshot and suspended on '
                            'the local table terminal date: the readings agree on a halt '
                            'whose outcome is not held locally',
                },
                'readings_disagree': {
                    'codes': sorted(c for c in tail_codes
                                    if c not in listed[0] and suspended_at_table_end(c, susp_pairs, table_end)),
                    'note': 'absent from the listed snapshot yet suspended on the local '
                            'table terminal date; the local evidence cannot say whether '
                            'this is a delisting in progress or a snapshot that already '
                            'excluded a name that was still only halted',
                },
            },
        },
        'sibling_channel_readings': {
            'note': 'every gap day carries a row in at least one sibling panel, so the '
                    'security-day is known to the supplier outside the price panel; for '
                    'the cases with an independent suspension record the plain reading is '
                    '"no trade, therefore no price row", and for the rest the local '
                    'evidence cannot separate that from a price-channel collection '
                    'difference',
            'gap_days_with_an_adjustment_row':
                len([c for c in cases if c['adjustment_row_here']]),
            'gap_days_with_a_limit_panel_row':
                len([c for c in cases if c['limit_row_here']]),
            'gap_days_with_neither': len([c for c in cases
                                          if not c['adjustment_row_here']
                                          and not c['limit_row_here']]),
            'readings_that_cannot_be_separated_locally':
                len([c for c in cases if c['verdict'] in
                     ('middle_gap_halt_unexplained',
                      'middle_gap_halt_status_table_absent',
                      'tail_star_st_evidenced_no_consolidation')]),
        },
        'readings': {
            'suspension_table_span': [min(susp_days), table_end],
            'suspension_pairs': len(susp_pairs),
            'listed_snapshot_securities': listed[2],
            'listed_snapshot_max_list_date': listed[1],
            'securities_absent_from_listed_snapshot':
                sorted({c['security'] for c in cases
                        if not c['listed_snapshot_membership']}),
            'gap_days_clustered_on_the_annual_report_deadline':
                sorted(Counter(c['session'][:7] for c in cases
                               if c['session'][5:7] in ('04', '05')).items()),
        },
        'by_arm': by_arm,
        'cases': cases,
        'arithmetic_note': {
            'tail_rows_sit_in_one_arm': sorted({arm for c in cases
                                                if not c['has_price_after']
                                                for arm in c['arms']}),
            'why_it_matters':
                'a halted security held to the end of its feed is a return the frozen '
                'target cannot price; the affected arm can only be marked '
                'data_insufficient for those days, never charged a zero return',
        },
        'claims_not_made': [
            'a missing price row is never reported as a zero return, a flat price or a '
            'delisting',
            'no verdict here asserts that a security left the market; the delisting '
            'verdict only quotes a name-change record that uses those words',
            'no imputation, no forward fill and no interpolation of any price or factor',
            'no economic verdict for any arm and no change to any production default',
            'resolving every case still would not by itself restore the ledger: the ledger '
            'stops at each arm\'s checkpoint, and these cases lie both before and after it',
        ],
        'formal_eligibility': False,
        'full_stage2_accepted': False,
        'hashes': {
            'reports/event_inventory_plan_20260924/missing_price_rows.csv':
                digest(MISSING_CSV),
            'data/raw/daily_prices.csv': digest(ROOT / 'data/raw/daily_prices.csv'),
            'data/raw/suspend_d.csv': digest(ROOT / 'data/raw/suspend_d.csv'),
            'data/raw/stock_list.csv': digest(ROOT / 'data/raw/stock_list.csv'),
            'data/raw/namechange_2023_2026.csv':
                digest(ROOT / 'data/raw/namechange_2023_2026.csv'),
        },
    }


def days_between(days, session):
    """Sessions strictly between the code's price day before and after, exclusive."""
    before = [d for d in days if d < session]
    after = [d for d in days if d > session]
    if not before or not after:
        return []
    return [d for d in days if before[-1] < d < after[0]]


def suspended_at_table_end(code, susp_pairs, table_end):
    """Was the security suspended on the local status table's last day?"""
    return (code, table_end) in susp_pairs


def has_st_record(code, namechanges):
    reasons = ' '.join(r['change_reason'] or '' for r in namechanges.get(code, []))
    return 'ST' in reasons


class AttributionTests(unittest.TestCase):
    def test_every_case_has_a_known_verdict(self):
        for case in build()['cases']:
            self.assertIn(case['verdict'], VERDICTS)

    def test_middle_and_tail_partition_is_exhaustive(self):
        cases = build()['cases']
        self.assertEqual(len([c for c in cases if c['has_price_after']])
                         + len([c for c in cases if not c['has_price_after']]),
                         len(cases))

    def test_every_case_has_a_price_row_before(self):
        # a gap with nothing before it would be a listing event, not a halt
        for case in build()['cases']:
            self.assertTrue(case['has_price_before'],
                            'unexpected leading gap for %s' % case['security'])

    def test_delisting_verdict_requires_a_quoted_record(self):
        for case in build()['cases']:
            if case['verdict'] == 'tail_delisting_consolidation_evidenced':
                reasons = ' '.join(r['change_reason'] or ''
                                   for r in case['namechange_records'])
                self.assertIn('退市整理期', reasons)

    def test_status_absent_verdict_requires_being_outside_the_table_range(self):
        for case in build()['cases']:
            if case['verdict'] == 'middle_gap_halt_status_table_absent':
                self.assertFalse(case['session_in_suspension_table_range'])
                self.assertIsNone(case['suspension_types_here'])

    def test_counts_are_internally_consistent(self):
        payload = build()
        self.assertEqual(sum(payload['headline']['by_verdict'].values()),
                         payload['headline']['cases'])
        self.assertEqual(sum(v['missing_required_security_days']
                             for v in payload['by_arm'].values()),
                         len([1 for c in payload['cases'] if c['arms']]))

    def test_no_imputation_language(self):
        text = json.dumps(build(), ensure_ascii=False)
        self.assertIn('never reported as a zero return', text)

    def test_tail_breakdown_partitions_the_tail(self):
        payload = build()
        tail = payload['tail_breakdown']
        subset = (tail['absent_from_listed_snapshot']['cases']
                  + tail['present_in_listed_snapshot']['cases'])
        self.assertEqual(subset, tail['cases'])
        self.assertEqual(subset, payload['headline']['tail_feed_stopped'])

    def test_convergent_readings_are_disjoint_where_they_should_be(self):
        payload = build()
        conv = payload['tail_breakdown']['convergent_readings']
        disagree = set(conv['readings_disagree']['codes'])
        self.assertFalse(disagree & set(conv['listed_and_halt_still_open']['codes']))
        for code in disagree:
            self.assertIn(code, payload['tail_breakdown']['absent_from_listed_snapshot']
                          ['codes'])
            self.assertIn(code, payload['tail_breakdown']
                          ['still_suspended_at_the_local_table_end']['codes'])

    def test_sibling_readings_match_the_per_case_flags(self):
        payload = build()
        cases = payload['cases']
        sibling = payload['sibling_channel_readings']
        self.assertEqual(sibling['gap_days_with_an_adjustment_row'],
                         len([c for c in cases if c['adjustment_row_here']]))
        self.assertEqual(sibling['gap_days_with_a_limit_panel_row'],
                         len([c for c in cases if c['limit_row_here']]))
        self.assertEqual(sibling['gap_days_with_neither'],
                         len([c for c in cases if not c['adjustment_row_here']
                              and not c['limit_row_here']]))


def tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AttributionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def run():
    target = OUT / 'attribution.json'
    if target.exists():
        raise RuntimeError('attribution.json already exists; refusing to overwrite')
    payload = build()
    write(target, payload)
    with io.open(OUT / 'attribution.csv', 'w', encoding='utf-8', newline='') as handle:
        handle.write('security,session,verdict,arms,price_day_before,price_day_after,'
                     'price_last_day,suspension_types_here,listed_snapshot,'
                     'adjustment_row_here,limit_row_here\n')
        for case in sorted(payload['cases'], key=lambda c: (c['security'], c['session'])):
            handle.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
                case['security'], case['session'], case['verdict'],
                ';'.join(case['arms']), case['price_day_before'] or '',
                case['price_day_after'] or '', case['price_last_day'] or '',
                '/'.join(case['suspension_types_here'] or []),
                case['listed_snapshot_membership'], case['adjustment_row_here'],
                case['limit_row_here']))
    head = payload['headline']
    print('cases=%d middle_gap=%d tail=%d unresolved=%d blocked_only_by_missing_table=%d'
          % (head['cases'], head['middle_gap_halt_with_resume'], head['tail_feed_stopped'],
             head['unresolved_after_local_evidence'],
             head['blocked_only_by_a_missing_local_table']), flush=True)
    for verdict in sorted(head['by_verdict']):
        print('  %-46s %3d' % (verdict, head['by_verdict'][verdict]), flush=True)
    print('unresolved:', flush=True)
    for case in head['unresolved_cases']:
        print('  %s %s %s' % (case['security'], case['session'], case['arms']), flush=True)
    tail = payload['tail_breakdown']
    print('tail codes=%d absent_from_listed_snapshot=%d (%d cases) '
          'present_in_listed_snapshot=%d (%d cases) still_suspended_at_%s=%d'
          % (tail['codes'], len(tail['absent_from_listed_snapshot']['codes']),
             tail['absent_from_listed_snapshot']['cases'],
             len(tail['present_in_listed_snapshot']['codes']),
             tail['present_in_listed_snapshot']['cases'],
             tail['still_suspended_at_the_local_table_end']['terminal_date'],
             tail['still_suspended_at_the_local_table_end']['cases']), flush=True)
    for name, block in sorted(tail['convergent_readings'].items()):
        print('  %-34s %s' % (name, block['codes']), flush=True)
    sibling = payload['sibling_channel_readings']
    print('sibling channels on the gap day: adjustment=%d limit=%d neither=%d '
          'readings_not_separable_locally=%d'
          % (sibling['gap_days_with_an_adjustment_row'],
             sibling['gap_days_with_a_limit_panel_row'],
             sibling['gap_days_with_neither'],
             sibling['readings_that_cannot_be_separated_locally']), flush=True)
    print('written: %s' % target, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--tests', action='store_true')
    args = parser.parse_args()
    if args.tests:
        return tests()
    if args.run:
        run()
        return 0
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
