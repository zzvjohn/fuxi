# -*- coding: utf-8 -*-
"""Work item 5 of the drought review: R5 pre-registration and access-log skeleton.

This batch DOES NOT run an R5 experiment and does NOT create any unknown sample.
It freezes, before any future data exists:
  * the observation start (first trading session whose data did not exist yet)
  * the first weekly signal date and its label maturity
  * the descriptive checkpoint and the single decisive endpoint
  * the frozen evaluation contract inherited from the item-4 design
  * an append-only access log so every future read is timestamped

Nothing here grants economic eligibility. The candidate list is empty by design:
no candidate held economic eligibility when this was written.
"""

import argparse
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

OUT = ROOT / 'reports/r5_preregistration_20260924'
LOG = OUT / 'access_log.jsonl'
PROTOCOL = OUT / 'protocol.json'

# Sessions at or before this date already existed when the batch was written.
PREREGISTRATION_CEILING = '2026-09-23'
DESCRIPTIVE_SECTIONS = 8
DECISIVE_SECTIONS = 26
HORIZON = 5

INHERITED = {
    'label': 'adj_open[exit] / adj_open[entry] - 1, entry = first session after the signal, '
             'exit = five sessions after entry, no forward fill',
    'benchmark': 'index 000985 open[exit] / open[entry] - 1 over the same window',
    'pool': 'positive raw close at the signal date with the frozen features finite',
    'selection': 'Top80 by score descending, contemporaneous amount descending, code ascending',
    'top_n': 80,
    'cost_bps': [0, 15, 30],
    'primary_cost_bps': 15,
    'cost_model': 'cost fraction = 2 * (names replaced / top_n) * bps / 1e4',
    'model_compare_design': 'reports/model_compare_20260924/protocol.json',
}


def sessions():
    cal = pd.read_csv(ROOT / 'data/raw/trade_calendar.csv', dtype=str)
    cal = cal[(cal.exchange == 'SSE') & (cal.is_open == '1')]
    return sorted(pd.to_datetime(cal.cal_date, format='mixed')
                  .dt.strftime('%Y-%m-%d').unique())


def weekly_stamps(days):
    series = pd.Series(pd.to_datetime(list(days)))
    return series.groupby(series.dt.to_period('W-SUN')).max().dt.strftime('%Y-%m-%d').tolist()


def observation_window(days, ceiling=PREREGISTRATION_CEILING, horizon=HORIZON,
                       descriptive=DESCRIPTIVE_SECTIONS, decisive=DECISIVE_SECTIONS):
    """The statistical requirement is fixed first; calendar coverage is reported, not obeyed.

    If the trade calendar does not reach far enough, the decisive endpoint date stays
    undefined. The requirement is never shrunk to fit the available calendar.
    """
    future = [d for d in days if d > ceiling]
    if not future:
        raise ValueError('no future session in the calendar beyond ' + ceiling)
    start = future[0]
    stamps = [s for s in weekly_stamps(future) if s >= start]
    if len(stamps) < descriptive:
        raise ValueError('calendar holds only %d future weekly sections' % len(stamps))
    windows = {}
    for stamp in stamps:
        later = [d for d in days if d > stamp]
        if len(later) > horizon:
            windows[stamp] = (later[0], later[horizon])

    def endpoint(count):
        if len(stamps) < count:
            return {'sections': count, 'signal': None, 'matured_on': None,
                    'calendar_covers': False,
                    'calendar_covers_sections': len(stamps)}
        signal = stamps[count - 1]
        return {'sections': count, 'signal': signal,
                'matured_on': windows[signal][1] if signal in windows else None,
                'calendar_covers': True, 'calendar_covers_sections': len(stamps)}

    return {'observation_start': start, 'stamps': stamps,
            'calendar_last_session': days[-1],
            'calendar_future_sections': len(stamps),
            'descriptive_endpoint': endpoint(descriptive),
            'decisive_endpoint': endpoint(decisive),
            'first_window': {'signal': stamps[0],
                             'entry': windows[stamps[0]][0] if stamps[0] in windows else None,
                             'exit': windows[stamps[0]][1] if stamps[0] in windows else None}}


def freeze():
    if PROTOCOL.exists():
        raise RuntimeError('R5 already pre-registered; refusing to overwrite')
    OUT.mkdir(parents=True, exist_ok=True)
    days = sessions()
    window = observation_window(days)
    model_protocol = ROOT / INHERITED['model_compare_design']
    protocol = {
        'batch': 'r5_preregistration_20260924',
        'preregistered_at': datetime.now().astimezone().isoformat(),
        'status': 'skeleton_only_no_unknown_sample',
        'purpose': 'work item 5: pre-register the future sample before it exists',
        'preregistration_ceiling': PREREGISTRATION_CEILING,
        'observation_start': window['observation_start'],
        'first_signal_date': window['first_window']['signal'],
        'first_window': window['first_window'],
        'calendar_last_session': window['calendar_last_session'],
        'calendar_future_sections': window['calendar_future_sections'],
        'calendar_note': 'the shipped trade calendar ends at %s, covering only %d future weekly '
                         'sections; the decisive endpoint needs %d, so its date stays undefined '
                         'until the calendar is extended. The requirement is not shrunk to fit '
                         'the calendar.' % (window['calendar_last_session'],
                                            window['calendar_future_sections'],
                                            DECISIVE_SECTIONS),
        'descriptive_checkpoint': window['descriptive_endpoint'],
        'decisive_endpoint': window['decisive_endpoint'],
        'descriptive_checkpoint_rule': 'run-and-decay signs only; this is descriptive and '
                                       'never a pass/fail verdict',
        'decisive_endpoint_rule': 'the single pre-registered decision point; no verdict is '
                                  'recorded before it and no repeated peeking may move it',
        'candidates': [],
        'candidates_note': 'empty by design: no candidate held economic eligibility when this '
                           'was written; the four historical signals remain research_exposed '
                           'and data_insufficient, so they cannot be pre-registered as R5 '
                           'candidates either',
        'unknown_sample_effective_count': 0,
        'historical_status': 'every read 2024-12-27..2026-09-23 stays research_exposed',
        'inherited_contract': INHERITED,
        'restart_rule': 'if a model, rule or candidate list changes during observation, the '
                        'unknown-sample start moves to the first session after the change and '
                        'the change is logged; the previous accumulation is void as evidence',
        'access_rule': 'every read of future data appends one record to access_log.jsonl with '
                       'its timestamp, purpose and the matured section count',
        'data_ceiling_rule': 'a ceiling is trusted only when it is a real session and was '
                             'explicitly measured or explicitly logged; no probe is inferred',
        'hashes': {'data/raw/trade_calendar.csv': digest(ROOT / 'data/raw/trade_calendar.csv'),
                   'reports/model_compare_20260924/protocol.json': digest(model_protocol)
                   if model_protocol.exists() else None},
        'jq_eligibility': False,
        'unlock_rule': 'R5 does not unlock any registration, threshold or production default',
    }
    write(PROTOCOL, protocol)
    append({'kind': 'preregistration', 'purpose': 'freeze the R5 skeleton before any future '
                                                  'data exists',
            'observation_start': protocol['observation_start'],
            'decisive_endpoint': protocol['decisive_endpoint'],
            'model_changed': False,
            'note': 'no future data was read; unknown sample effective count stays 0'})
    print('pre-registered: observation start %s, first signal %s'
          % (protocol['observation_start'], protocol['first_signal_date']), flush=True)
    print('descriptive checkpoint: %d sections, matured on %s'
          % (protocol['descriptive_checkpoint']['sections'],
             protocol['descriptive_checkpoint']['matured_on']), flush=True)
    print('decisive endpoint: %d sections, signal %s, matured on %s'
          % (protocol['decisive_endpoint']['sections'],
             protocol['decisive_endpoint']['signal'],
             protocol['decisive_endpoint']['matured_on']), flush=True)
    print('calendar note: %s' % protocol['calendar_note'], flush=True)


def load_log():
    if not LOG.exists():
        return []
    with io.open(LOG, encoding='utf-8', newline='') as handle:
        raw = handle.read()
    if '\r' in raw:
        raise ValueError('access log must stay LF-only')
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def append(record):
    """Append one record. Never rewrites, never reorders."""
    OUT.mkdir(parents=True, exist_ok=True)
    existing = load_log()
    record = dict(record)
    record['logged_at'] = datetime.now().astimezone().isoformat()
    record['sequence'] = len(existing) + 1
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with io.open(LOG, 'a', encoding='utf-8', newline='') as handle:
        handle.write(line + '\n')
    payload = io.open(LOG, encoding='utf-8', newline='').read()
    if '\r' in payload:
        raise ValueError('access log must stay LF-only')
    if payload.splitlines()[-1] != line:
        raise ValueError('access log append verification failed')
    return record


def probe_ceiling():
    """Authoritative fresh ceiling: the real maximum date in the primary price file."""
    maximum = ''
    for part in pd.read_csv(ROOT / 'data/raw/daily_prices.csv', usecols=['trade_date'],
                            dtype=str, chunksize=500000):
        maximum = max(maximum, part.trade_date.str.replace('-', '', regex=False).max())
    return '%s-%s-%s' % (maximum[:4], maximum[4:6], maximum[6:8])


def status(ceiling=None, fresh=False):
    protocol = read(PROTOCOL)
    days = sessions()
    entries = load_log()
    logged = [e for e in entries if e.get('ceiling')]
    if fresh:
        ceiling = probe_ceiling()
        source = 'probe_measured'
    elif ceiling:
        source = 'explicit_argument'
    elif logged:
        ceiling = logged[-1]['ceiling']
        source = 'last_logged_not_remeasured'
    else:
        ceiling = protocol['preregistration_ceiling']
        source = 'preregistration_ceiling_not_remeasured'
    if ceiling not in days:
        raise ValueError('ceiling is not a real session: ' + ceiling)
    window = observation_window(days)
    matured = []
    for stamp in window['stamps']:
        later = [d for d in days if d > stamp]
        if len(later) <= HORIZON:
            continue
        if later[HORIZON] <= ceiling:
            matured.append(stamp)
    # [!] 2026-09-25 (P5): 原实现把 protocol 里**冻结的** descriptive_checkpoint /
    #     decisive_endpoint 直接当「日历覆盖」报出去。日历一旦延伸（P5 把它推到 2027-12-31），
    #     这两个字段就变成**预注册当时的快照**，而日历覆盖是「当前事实」——
    #     继续混着报会制造一个影子结论：工具一直说「判定端点不可覆盖」，而它早已可覆盖。
    #     规范做法（不改统计要求、不改冻结件）：两件事分开报，各自标注时点。
    report = {'ceiling': ceiling, 'ceiling_source': source,
              'observation_start': protocol['observation_start'],
              'sessions_observed': len([d for d in days
                                        if protocol['observation_start'] <= d <= ceiling]),
              'sections_matured': len(matured),
              'last_matured_signal': matured[-1] if matured else None,
              # ---- 冻结件里的原样快照（历史记录，不随日历变化）----
              'as_recorded_at_preregistration': {
                  'as_of': 'preregistration',
                  'note': '预注册当时写死的快照；日历延伸后它不再描述当前事实，但也不改写',
                  'descriptive_checkpoint': protocol['descriptive_checkpoint'],
                  'decisive_endpoint': protocol['decisive_endpoint'],
                  'calendar_last_session': protocol['calendar_last_session'],
                  'calendar_future_sections': protocol['calendar_future_sections'],
              },
              # ---- 按当前随包日历重算的覆盖（统计要求仍从冻结件读入）----
              'coverable_with_the_current_calendar': {
                  'as_of': 'now',
                  'note': ('按当前 data/raw/trade_calendar.csv 重算；段数要求从冻结件读入，'
                           '从不重算、从不缩小'),
                  'calendar_last_session': window['calendar_last_session'],
                  'calendar_future_sections': window['calendar_future_sections'],
                  'descriptive_checkpoint': window['descriptive_endpoint'],
                  'decisive_endpoint': window['decisive_endpoint'],
                  'requirement_estimated_here': False,
              },
              # ---- 向下兼容: 保留原键名, 但显式标注它是快照 ----
              'descriptive_checkpoint': protocol['descriptive_checkpoint'],
              'decisive_endpoint': protocol['decisive_endpoint'],
              'checkpoint_keys_are_preregistration_snapshots': True,
              'descriptive_reached': len(matured) >= protocol['descriptive_checkpoint']['sections'],
              'decisive_reached': len(matured) >= protocol['decisive_endpoint']['sections'],
              'access_entries': len(entries),
              'unknown_sample_effective_count': protocol['unknown_sample_effective_count'],
              'verdict': 'not_evaluated' if len(matured) <
                         protocol['decisive_endpoint']['sections'] else 'decision_point_reached'}
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
    return report


class PreregistrationTests(unittest.TestCase):
    def test_observation_start_is_after_the_ceiling(self):
        days = ['2026-09-22', '2026-09-23', '2026-09-24', '2026-09-28', '2026-09-29',
                '2026-09-30', '2026-10-08', '2026-10-09', '2026-10-12', '2026-10-13',
                '2026-10-14', '2026-10-15', '2026-10-16', '2026-10-19']
        window = observation_window(days, ceiling='2026-09-23', horizon=5,
                                    descriptive=1, decisive=2)
        self.assertEqual(window['observation_start'], '2026-09-24')
        self.assertEqual(window['first_window']['signal'], '2026-09-24')
        self.assertEqual(window['first_window']['entry'], '2026-09-28')
        self.assertEqual(window['first_window']['exit'], '2026-10-12')

    def test_ceiling_must_be_a_real_session(self):
        days = ['2026-09-23', '2026-09-24']
        self.assertNotIn('2026-09-25', days)

    def test_weekly_last_session_wins(self):
        days = ['2026-09-21', '2026-09-22', '2026-09-23', '2026-09-24']
        self.assertEqual(weekly_stamps(days)[-1], '2026-09-24')

    def test_decisive_endpoint_is_not_shrunk_to_fit_the_calendar(self):
        days = ['2026-09-23', '2026-09-24', '2026-09-28', '2026-09-29', '2026-09-30',
                '2026-10-08', '2026-10-09', '2026-10-12']
        window = observation_window(days, ceiling='2026-09-23', horizon=5,
                                    descriptive=1, decisive=4)
        self.assertEqual(window['calendar_future_sections'], 4)
        self.assertTrue(window['decisive_endpoint']['calendar_covers'])
        short = observation_window(days[:6], ceiling='2026-09-23', horizon=5,
                                   descriptive=1, decisive=4)
        self.assertEqual(short['calendar_future_sections'], 3)
        self.assertIsNone(short['first_window']['exit'])
        self.assertFalse(short['decisive_endpoint']['calendar_covers'])
        self.assertIsNone(short['decisive_endpoint']['matured_on'])
        self.assertEqual(short['decisive_endpoint']['sections'], 4)

    def test_candidate_list_is_empty(self):
        protocol = read(PROTOCOL) if PROTOCOL.exists() else None
        if protocol is not None:
            self.assertEqual(protocol['candidates'], [])
            self.assertEqual(protocol['unknown_sample_effective_count'], 0)


def tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(PreregistrationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--access', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--tests', action='store_true')
    parser.add_argument('--ceiling', default=None)
    parser.add_argument('--purpose', default=None)
    parser.add_argument('--model-change', default=None,
                        help='declare a mid-observation model change; restarts the sample')
    args = parser.parse_args()
    if args.tests:
        return tests()
    if args.freeze:
        freeze()
        return 0
    if args.status:
        status(args.ceiling, fresh=args.probe)
        return 0
    if args.access:
        protocol = read(PROTOCOL)
        record = {'kind': 'access', 'purpose': args.purpose,
                  'ceiling': args.ceiling or protocol['preregistration_ceiling'],
                  'model_changed': bool(args.model_change),
                  'note': args.model_change or ''}
        if args.model_change:
            days = sessions()
            days_after = [d for d in days if d > record['ceiling']]
            record['restarted_at'] = days_after[0] if days_after else None
            record['previous_accumulation'] = 'void as evidence'
        entry = append(record)
        print(json.dumps(entry, ensure_ascii=False, indent=1), flush=True)
        return 0
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
