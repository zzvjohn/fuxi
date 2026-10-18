# -*- coding: utf-8 -*-
"""Advance4: separate the two data repairs and test the explicit delist channel.

Three readings, each isolating one change:

  D_reproduce_old_suspend  the backup of the suspension table as it stood before
                           this batch, legacy freeze derivation. Must reproduce
                           advance3's last rung day for day. If it does not, the
                           batch moved a reading it did not intend to move.
  F_legacy_extended        the extended suspension table, legacy derivation. This
                           isolates the data repair alone: same definitions, a
                           table that no longer stops in the middle of the window.
  G_explicit_delist        the extended table plus the vendor's own delist_date,
                           which the stale security master had made unreachable.

Anti-smuggling invariants, asserted rather than described
---------------------------------------------------------
1. `D` reproduces advance3's `D_halt_session_vacuous` day for day (finite days,
   terminal session and blocker list, per arm).
2. The fact map under `G` is a superset of the one under `F`, and every fact the
   explicit channel adds or overrides has a date after the name's last quote and
   inside the window sessions.
3. `G` grants every event `F` granted, and each event `G` grants beyond that has
   an ex-date later than the arm's stop under `F`.
4. For every arm, every net-asset value `F` computed up to its own stop is
   *identical* under `G`. A relabelling may not revalue a position.
5. Every delisted held name that has a quote in the window is either frozen or
   refused with a written reason; nothing is dropped silently.
6. No entitlement is granted on the freeze date of a fact.
7. No frozen name receives a fill on or after its freeze date.
8. `action_conflicts = 0`.

No collection, no mutation of sealed evidence or of the raw tables, no production
S1-S5 state, no feedback state, no candidate queue.
"""

import argparse
import glob
import io
import json
import sys
from dataclasses import asdict
from datetime import datetime
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
sys.path.insert(0, str(ROOT / 'scripts'))

import fuxi_ledger_advance3_20260924 as prior_ladder  # noqa: E402
from fuxi_delist_channel_20260924 import (  # noqa: E402
    explicit_facts, load_delist_dates, merge_freeze_facts,
)

OUT = ROOT / 'reports/ledger_advance4_20260924'
PRIOR = prior_ladder.PRIOR
SEALED_OUT = prior_ladder.SEALED_OUT
ADVANCE3_OUT = ROOT / 'reports/ledger_advance3_20260924'
ADJUDICATED = prior_ladder.ADJUDICATED
INVENTORY = prior_ladder.INVENTORY
POLICY = prior_ladder.POLICY
DELIST_TABLE = ROOT / 'data/raw/delist_dates.csv'
COVERAGE_MANIFEST = ROOT / 'reports/suspend_coverage_20260924/coverage_manifest.json'

CONFIGS = [
    ('D_reproduce_old_suspend', 'old', 'legacy'),
    ('F_legacy_extended', 'extended', 'legacy'),
    ('G_explicit_delist', 'extended', 'explicit'),
]
EVIDENCE_KEYS = ('signal', 'status', 'finite_days', 'end', 'error_reasons')


def main(configs):
    import numpy as np
    import pandas as pd
    # `retrigger` and `unresolved_events` are reused from advance3, whose own
    # `main` injects `pd` into its module globals before calling them. Reusing the
    # functions means reproducing that injection here; editing the earlier batch's
    # frozen script instead would move a reading this batch has to reproduce.
    prior_ladder.pd = pd
    from portfolio_execution_ledger import CorporateAction, LotRule, run_ledger, reconcile_journal
    from fuxi_execution_lab_20260923 import write, digest
    from fuxi_window_replay_20260923 import Book, keyed, merge_overlay, identity_key
    from fuxi_ledger_advance_20260923 import derive_freeze_facts
    from run_contract import RunContract

    policy = json.loads(POLICY.read_text(encoding='utf-8'))
    assert policy['single_writer_confirmed'] and policy['current_mode'] == 'controlled_manual', \
        'runtime policy does not permit this run'

    before = digest(INVENTORY)
    adjudicated = json.loads(ADJUDICATED.read_text(encoding='utf-8'))
    assert adjudicated['adjudication_changed'] is False
    assert before == adjudicated['hashes']['reports/event_inventory_20260923/inventory.json'], \
        'sealed inventory changed since the adjudication batch'
    unsettled, bucket_overlaps = prior_ladder.unresolved_events()
    adjudicated_pairs = {(a['code'], a['ex_date']) for a in adjudicated['actions']}
    pending = unsettled - adjudicated_pairs
    print('pending unsettled %d; rows in both buckets %d'
          % (len(pending), len(bucket_overlaps)), flush=True)

    coverage = json.loads(COVERAGE_MANIFEST.read_text(encoding='utf-8'))
    backups = sorted(glob.glob(str(ROOT / 'data/raw/suspend_d.csv.bak_*')))
    assert backups, 'the pre-extension suspension table backup is missing'
    old_suspend_path = Path(backups[-1])
    assert digest(old_suspend_path) == coverage['sha256_before'], \
        'the backup is not the table the extension started from'
    print('old suspension table %s (%s)' % (old_suspend_path.name,
                                            digest(old_suspend_path)[:12]), flush=True)

    prior = json.loads((PRIOR / 'protocol.json').read_text(encoding='utf-8'))
    source = ROOT / 'reports/window_evidence_20260923'
    OUT.mkdir(parents=True, exist_ok=True)
    actions, inventory_count, conflicts, injected_ids = prior_ladder.load_actions(
        adjudicated['actions'])
    assert conflicts == [], 'inventory conflicts: %s' % conflicts[:2]

    with RunContract(parameters={'actions': len(actions), 'configs': [c[0] for c in configs]},
                     run_id='ledger_advance4_' + datetime.now().strftime('%H%M%S_%f'),
                     runs_dir=OUT / 'runs', lock_path=OUT / 'advance4.lock'):
        target_sets = {name: json.loads((PRIOR / 'targets' / (name + '.json'))
                                        .read_text(encoding='utf-8'))['targets']
                       for name in prior['signals']}
        earliest = min([prior['start']] + [d for t in target_sets.values() for d in t])

        def panel(path, columns):
            frame = pd.read_csv(path, usecols=columns, dtype={'ts_code': str, 'trade_date': str})
            frame['trade_date'] = pd.to_datetime(frame.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
            return frame[(frame.trade_date >= earliest) & (frame.trade_date <= prior['end'])]

        prices = panel(ROOT / 'data/raw/daily_prices.csv',
                       ['ts_code', 'trade_date', 'open', 'close', 'pre_close'])
        limits = panel(ROOT / 'data/raw/stk_limit.csv',
                       ['ts_code', 'trade_date', 'up_limit', 'down_limit'])
        adj_raw = panel(ROOT / 'data/raw/adj_factor_daily.csv',
                        ['ts_code', 'trade_date', 'adj_factor'])
        print('panels loaded', len(prices), len(limits), len(adj_raw), flush=True)

        def overlay(name):
            frame = pd.read_csv(source / name, dtype={'ts_code': str, 'trade_date': str}).fillna('')
            frame['trade_date'] = pd.to_datetime(frame.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
            return frame

        conf = []
        quotes = keyed(prices[['ts_code', 'trade_date', 'open', 'close', 'pre_close']],
                       ['open', 'close', 'pre_close'], conf)
        overlay_limits = overlay('stk_limit_overlay.csv')
        boundaries = merge_overlay(keyed(limits, ['up_limit', 'down_limit'], conf),
                                   keyed(overlay_limits, ['up_limit', 'down_limit'], conf),
                                   'stk_limit', conf)

        def read_suspend(path):
            frame = pd.read_csv(path, dtype=str).fillna('')
            frame['trade_date'] = pd.to_datetime(
                frame.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
            return frame

        susp_old = read_suspend(old_suspend_path)
        susp_extended = read_suspend(ROOT / 'data/raw/suspend_d.csv')
        suspension = {'old': susp_old, 'extended': susp_extended}
        print('suspend rows old=%d extended=%d' % (len(susp_old), len(susp_extended)), flush=True)

        calendar = pd.read_csv(ROOT / 'data/raw/trade_calendar.csv', dtype=str)
        dates = sorted(set(pd.to_datetime(
            calendar.loc[(calendar.exchange == 'SSE') & (calendar.is_open == '1'), 'cal_date'],
            format='mixed').dt.strftime('%Y-%m-%d')))
        dates = [d for d in dates if prior['start'] <= d <= prior['end']]
        held_codes = {c for t in target_sets.values() for w in t.values() if w for c in w}
        listed = set(pd.read_csv(ROOT / 'data/raw/stock_list.csv', dtype=str).ts_code)
        delisted = load_delist_dates(DELIST_TABLE)
        print('master %d names; vendor delisted %d' % (len(listed), len(delisted)), flush=True)

        held_prices = prices[prices.ts_code.isin(held_codes)]
        quoted_held = set(held_prices.ts_code.unique())

        facts_by_channel = {}
        for table in ('old', 'extended'):
            legacy, blocked = derive_freeze_facts(held_prices, suspension[table], dates, listed)
            facts_by_channel[(table, 'legacy')] = (legacy, blocked)
        accepted, refused = explicit_facts(held_prices, dates, delisted)
        union_facts, reconciliation = merge_freeze_facts(
            facts_by_channel[('extended', 'legacy')][0], accepted)
        facts_by_channel[('extended', 'explicit')] = (union_facts, [])

        held_delisted = {c for c in held_codes if c in delisted and c in quoted_held}
        covered = set(accepted) | {row['code'] for row in refused}
        assert covered == held_delisted, \
            ('a delisted held name was neither frozen nor refused: %s'
             % sorted(held_delisted - covered))
        no_quote = sorted(c for c in held_codes if c in delisted and c not in quoted_held)
        old_facts = facts_by_channel[('old', 'legacy')][0]
        print('legacy facts old=%d extended=%d; explicit accepted=%d refused=%d; union=%d'
              % (len(old_facts), len(facts_by_channel[('extended', 'legacy')][0]),
                 len(accepted), len(refused), len(union_facts)), flush=True)

        for code, fact in accepted.items():
            assert fact['date'] in dates, 'fact outside the window: %s' % code
            assert fact['date'] > fact['last_quote'], 'fact not after the last quote: %s' % code
        for row in reconciliation['overrides']:
            assert row['vendor_delist_date'] in dates
            assert row['vendor_delist_date'] > row['last_quote']
            assert row['suspend_derived_freeze'] != row['vendor_delist_date']

        calendar_set = set(dates)
        advance3_d = {name: json.loads((ADVANCE3_OUT / 'D_halt_session_vacuous' / 'checkpoints'
                                        / (name + '.json')).read_text(encoding='utf-8'))
                      for name in prior['signals']}

        def adjust(mode):
            frame = adj_raw.drop_duplicates(['ts_code', 'trade_date']).sort_values(
                ['ts_code', 'trade_date']).copy()
            frame['ts_code'] = [identity_key(d, c, conf)[1]
                                for d, c in zip(frame.trade_date, frame.ts_code)]
            keyed(frame, ['adj_factor'], conf)
            frame = frame.drop_duplicates(['trade_date', 'ts_code']).sort_values(
                ['ts_code', 'trade_date'])
            previous = frame.groupby('ts_code').adj_factor.shift()
            frame['action_check_available'] = (frame.adj_factor > 0) & (previous > 0)
            frame['action_required'] = ~np.isclose(frame.adj_factor, previous, rtol=1e-7, atol=0)
            frame, suppressed = prior_ladder.retrigger(frame, quotes, mode)
            frame = frame.drop(columns=['reference_break'], errors='ignore')
            return frame, suppressed

        rows, ladder, baselines = [], [], {}
        frame, suppressed = adjust('reference_break')
        adjustments = keyed(frame, ['action_check_available', 'action_required'], conf)
        print('trigger=reference_break suppressed=%d' % len(suppressed), flush=True)

        for label, table, channel in configs:
            susp = suspension[table]
            state = merge_overlay(_states(susp, conf, identity_key),
                                  _states(overlay('suspend_d_overlay.csv'), conf, identity_key),
                                  'suspend_d', conf)
            halts = {k for k, payload in state.items() if ('S', '') in payload}
            partial = {k for k, payload in state.items()
                       if any(t == 'S' and timing for t, timing in payload)}
            evidence = facts_by_channel[(table, channel)][0]
            book = Book(quotes, boundaries, adjustments, halts, partial,
                        [(r['date'], r['code']) for r in conf], partial_halt_state=True)
            print('== %s suspend=%s channel=%s facts=%d'
                  % (label, table, channel, len(evidence)), flush=True)
            cfg_rows = []
            for name in prior['signals']:
                targets = target_sets[name]
                rules = {c: LotRule(200, 1) if c.startswith(('688', '689')) else LotRule(100, 1)
                         if c.endswith('.BJ') else LotRule()
                         for w in targets.values() if w for c in w}
                runnable = [a for a in actions
                            if a['record_date'] in calendar_set and a['ex_date'] in calendar_set]
                result = run_ledger(dates, book, targets, initial_cash=prior['initial_cash'],
                                    cost_bps=prior['cost_bps'], rules=rules,
                                    actions=[CorporateAction(**a) for a in runnable],
                                    fractional_share_policy='floor',
                                    duplicate_event_policy='collapse_identical',
                                    missing_limit_policy='defer',
                                    delist_evidence=evidence,
                                    unresolved_action_policy='exclude_held_security_day',
                                    halted_session_check_policy='vacuous')
                write(OUT / label / 'ledgers' / (name + '.json'), asdict(result))
                parity = reconcile_journal(result, prior['initial_cash'])
                finite = [d for d in result.days if d['nav'] is not None and isfinite(d['nav'])]
                last_day = finite[-1]['date'] if finite else None
                reasons = {}
                for error in result.errors:
                    reasons[error['reason']] = reasons.get(error['reason'], 0) + 1
                granted = prior_ladder.granted_ids(result.journal)
                loadable = {a['event_id']: (a['code'], a['ex_date']) for a in runnable}
                assert set(granted) <= set(loadable), 'granted an event outside the loaded set'
                assert not ({loadable[e] for e in granted} & pending), \
                    'granted an event the inventory still cannot settle'
                violations = [{'code': entry['code'], 'date': entry['date']}
                              for entry in result.journal
                              if entry['kind'] == 'fill' and entry.get('code') in evidence
                              and entry.get('date')
                              and entry['date'] >= evidence[entry['code']]['date']]
                assert not violations, ('a frozen name was filled: %s' % violations[:3])
                granted_pairs = {loadable[e] for e in granted}
                on_freeze = sorted(
                    '%s %s' % (code, ex_date) for code, ex_date in granted_pairs
                    if evidence.get(code, {}).get('date') == ex_date)
                assert not on_freeze, 'an entitlement was granted on a freeze date: %s' % on_freeze
                in_force = sorted(code for code, fact in evidence.items()
                                  if last_day is not None and last_day >= fact['date'])
                row = {'config': label, 'suspend': table, 'channel': channel, 'signal': name,
                       'status': result.status, 'finite_days': len(finite), 'end': last_day,
                       'terminal_nav': finite[-1]['nav'] if finite else None,
                       'fills': sum(1 for e in result.journal if e['kind'] == 'fill'),
                       'excluded_security_days': result.excluded,
                       'errors': result.errors[:20], 'error_reasons': reasons,
                       'parity': parity, 'granted_event_ids': granted,
                       'frozen_names_in_force': in_force,
                       'nav_series': [{'date': d['date'], 'nav': d['nav']} for d in finite],
                       'formal_eligibility': False,
                       'advance3_D': {'finite_days': advance3_d[name]['finite_days'],
                                      'end': advance3_d[name]['end']},
                       'delta_vs_advance3_D': len(finite) - advance3_d[name]['finite_days']}
                cfg_rows.append(row)
                write(OUT / label / 'checkpoints' / (name + '.json'), row)
                print('  %-28s %-34s days=%3d end=%s frozen=%d %s'
                      % (name, row['status'], row['finite_days'], row['end'],
                         len(in_force), reasons), flush=True)
            ladder.append({'config': label, 'suspend': table, 'channel': channel,
                           'finite_days': sum(r['finite_days'] for r in cfg_rows),
                           'of_possible': len(dates) * len(prior['signals']),
                           'arms': [{k: r[k] for k in EVIDENCE_KEYS} for r in cfg_rows]})
            baselines[label] = {r['signal']: r for r in cfg_rows}

        # 1. the old table plus the old derivation must reproduce advance3's last rung.
        reproduction = []
        for name in prior['signals']:
            got = baselines['D_reproduce_old_suspend'][name]
            expected = advance3_d[name]
            assert got['finite_days'] == expected['finite_days'], \
                ('D no longer reproduces advance3', name, got['finite_days'],
                 expected['finite_days'])
            assert got['end'] == expected['end'], ('D end moved', name)
            assert got['errors'] == expected['errors'], ('D blockers moved', name)
            reproduction.append({'signal': name, 'finite_days': got['finite_days'],
                                 'end': got['end'], 'reproduces_advance3': True})

        # 2. the evidence map only grows under the explicit channel.
        assert set(facts_by_channel[('extended', 'legacy')][0]) <= set(union_facts), \
            'the explicit channel dropped a legacy fact'
        for code in sorted(set(union_facts) - set(facts_by_channel[('extended', 'legacy')][0])):
            assert union_facts[code]['date'] > union_facts[code]['last_quote'], code

        nav_report, grant_report = [], []
        for name in prior['signals']:
            f_row = baselines['F_legacy_extended'][name]
            g_row = baselines['G_explicit_delist'][name]
            assert set(f_row['granted_event_ids']) <= set(g_row['granted_event_ids']), \
                (name, 'G dropped a grant F made')
            for event_id in sorted(set(g_row['granted_event_ids'])
                                   - set(f_row['granted_event_ids'])):
                extra_date = event_id[-10:]
                assert f_row['end'] is None or extra_date > f_row['end'], \
                    (name, 'G granted an event on or before F\'s stop', event_id)
            grant_report.append({'signal': name,
                                 'F_grants': len(f_row['granted_event_ids']),
                                 'G_grants': len(g_row['granted_event_ids']),
                                 'extra_grants_all_after_F_stop': True})
            f_nav = {r['date']: r['nav'] for r in f_row['nav_series']}
            g_nav = {r['date']: r['nav'] for r in g_row['nav_series']}
            shared = sorted(set(f_nav) & set(g_nav))
            worst = max((abs(f_nav[d] - g_nav[d]) for d in shared), default=0.0)
            assert worst == 0.0, (name, 'the relabelling moved a net asset value', worst)
            nav_report.append({'signal': name, 'shared_sessions': len(shared),
                               'max_abs_nav_difference': worst,
                               'F_end': f_row['end'], 'G_end': g_row['end'],
                               'F_finite_days': f_row['finite_days'],
                               'G_finite_days': g_row['finite_days'],
                               'nav_identical_up_to_F_stop': True})

        summary = {
            'window': [prior['start'], prior['end']],
            'sessions': len(dates),
            'held_names': len(held_codes),
            'master_names': len(listed),
            'vendor_delisted_total': len(delisted),
            'delisted_held_names_with_a_quote': len(held_delisted),
            'delisted_held_names_without_any_quote_in_the_window': no_quote,
            'legacy_facts_old_table': len(old_facts),
            'legacy_facts_extended_table': len(facts_by_channel[('extended', 'legacy')][0]),
            'legacy_blocked_extended_table': facts_by_channel[('extended', 'legacy')][1],
            'explicit_accepted': len(accepted),
            'explicit_refused': refused,
            'legacy_facts_replaced_by_vendor_date': reconciliation['overrides'],
            'explicit_facts_without_a_legacy_twin': reconciliation['additions'],
            'facts_where_both_channels_agree': reconciliation['agreement'],
            'suspend_extension': {
                'old_sha256': coverage['sha256_before'],
                'new_sha256': coverage['sha256_after'],
                'rows_appended': coverage['rows_appended'],
                'max_date_before': coverage['max_date_before'],
                'max_date_after': coverage['max_date_after'],
            },
            'delist_table': {'sha256': digest(DELIST_TABLE), 'names': len(delisted),
                            'file': 'data/raw/delist_dates.csv'},
            'ladder': ladder,
            'reproduction_of_advance3': reproduction,
            'nav_identity': nav_report,
            'grants': grant_report,
            'anti_smuggling': {
                'config_D_reproduces_advance3_D': True,
                'evidence_map_is_monotone': True,
                'extra_grants_are_later_than_F_stop': True,
                'nav_identical_up_to_F_stop': True,
                'every_delisted_held_name_is_frozen_or_refused': True,
                'no_grant_on_a_freeze_date': True,
                'no_fill_for_a_frozen_name': True,
                'action_conflicts_zero': True,
            },
            'claims_not_made': [
                'reaching the full window is arithmetic; it is not evidence that the arm is '
                'tradable, profitable, or out of sample',
                'a frozen position is carried at its last observed mark with no synthetic '
                'return: the freeze is a declared accounting policy, not a recovered price',
                'the vendor\'s suspend table and its delist table disagree for four held '
                'names; the batch publishes both dates and claims neither is exact',
                'the suspension extension is a data repair, not a definition change: it '
                'converts unknown states into evidenced halts, it does not relax a check',
                'no production default, feedback state or candidate queue is touched',
            ],
            'formal_eligibility': False,
            'production_unchanged': True,
            'history_exposed': True,
        }
        write(OUT / 'advance4.json', summary)
        print(json.dumps([{k: l[k] for k in ('config', 'finite_days', 'of_possible')}
                          for l in ladder], ensure_ascii=False, indent=1))


def _states(frame, conf, identity_key):
    result = {}
    for record in frame.itertuples():
        key = identity_key(record.trade_date, record.ts_code, conf)
        result.setdefault(key, set()).add((record.suspend_type, record.suspend_timing))
    return {k: sorted(v) for k, v in result.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', default=None)
    args = parser.parse_args()
    chosen = CONFIGS
    if args.only:
        wanted = {w.strip() for w in args.only.split(',') if w.strip()}
        chosen = [c for c in CONFIGS if c[0] in wanted]
        assert chosen, 'no known config selected'
    main(chosen)
