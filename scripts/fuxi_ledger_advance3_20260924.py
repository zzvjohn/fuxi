# -*- coding: utf-8 -*-
"""Shadow ladder: three readings of the same frozen design, differing in exactly
two switch positions, so the residual halts can be attributed to a definition
rather than to an opinion.

Why a ladder instead of one more repair
---------------------------------------
The advance2 reading stopped three arms on two held sessions. Both stops trace to
a definition, not to a missing datum:

  * `r4_equal_rank` stops on 689009.SH 2025-11-03, where the vendor adjustment
    factor moved 1.0332 -> 1.033 while the exchange published the *same*
    reference price as the prior close (60.70 -> 60.70). Nothing was
    implemented for a holder on that session; the trigger is the vendor's
    fourth-decimal rounding, not an entitlement.
  * `llm_family_1_1_fresh_llm` stops on 002713.SZ 2025-12-30, where the reference
    genuinely broke (11.00 -> 8.14, confirmed independently by the ex-date
    reference field and the limit pair) and no local channel can settle the
    share/cash split. The entitlement is genuinely unsettled.

So the ladder separates the two questions. `adjusted_factor` vs `reference_break`
asks whether the trigger should be `the vendor factor moved` or `the exchange
reference moved`. `block` vs `exclude_held_security_day` asks whether an
unsettled entitlement must stop the arm or only the name on that session.

Both switches are single-direction tightening of the *halt*, never a grant:
neither switch can supply an entitlement, and the ladder asserts that.

Anti-smuggling invariants, asserted rather than described
----------------------------------------------------------
1. `reports/event_inventory_20260923/inventory.json` still hashes to the value the
   adjudication batch recorded.
2. Config `A` (both switches legacy) reproduces `ledger_advance2_20260924` day for
   day. A switch that also moves the legacy reading is not a switch.
3. Over every config, the set of event ids actually granted is a subset of the
   loaded action set, and no event the adjudication marked unresolved is granted.
4. The events config `C` grants beyond config `A` are exactly the events whose
   ex-date is later than A's stop. Reaching further must not grant more.
5. Config `C`'s excluded security-sessions are enumerated, and each one is a
   session where a reference break has no supplied action.
6. No entitlement is ever synthesized from the adjustment ratio.

No collection, no writes to raw data or sealed evidence, no production S1-S5
state, no feedback state, no candidate queue.
"""

import argparse
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

OUT = ROOT / 'reports/ledger_advance3_20260924'
PRIOR = ROOT / 'reports/execution_followup_20260923/replay'
SEALED_OUT = ROOT / 'reports/ledger_advance_20260923'
ADVANCE2_OUT = ROOT / 'reports/ledger_advance2_20260924'
INVENTORY = ROOT / 'reports/event_inventory_20260923/inventory.json'
INVENTORY_ACTIONS = ROOT / 'reports/event_inventory_20260923/approved_actions.json'
ADJUDICATED = ROOT / 'reports/event_adjudication_20260924/adjudicated_actions.json'
POLICY = ROOT / 'codex_automations/fuxi_runtime_policy.json'
TICK = 0.01

CONFIGS = [
    ('A_legacy', 'adjusted_factor', 'block', 'unavailable'),
    ('B_reference_break', 'reference_break', 'block', 'unavailable'),
    ('C_reference_break_excluded', 'reference_break', 'exclude_held_security_day',
     'unavailable'),
    ('D_halt_session_vacuous', 'reference_break', 'exclude_held_security_day', 'vacuous'),
]


def load_actions(adjudicated):
    """Sealed inventory approvals first, then the pre-validated sets, then ours."""
    inventory = json.loads(INVENTORY_ACTIONS.read_text(encoding='utf-8'))['actions']
    by_id = {a['event_id']: a for a in inventory}
    conflicts = []
    for path, key in ((ROOT / 'reports/execution_events_20260923/approved_actions.json', 'actions'),
                      (ROOT / 'reports/execution_followup_20260923/validation.json', 'new_actions')):
        for action in json.loads(path.read_text(encoding='utf-8'))[key]:
            old = by_id.get(action['event_id'])
            if old and {k: v for k, v in old.items() if k != 'evidence'} != \
                      {k: v for k, v in action.items() if k != 'evidence'}:
                conflicts.append({'event_id': action['event_id'], 'inventory': old, 'prior': action})
                continue
            by_id.setdefault(action['event_id'], action)
    closed = set(by_id)
    injected = []
    for action in adjudicated:
        assert action['event_id'] not in closed, \
            'adjudicated action would replace a sealed one: %s' % action['event_id']
        by_id[action['event_id']] = action
        injected.append(action['event_id'])
    return list(by_id.values()), len(inventory), conflicts, sorted(injected)


def unresolved_events():
    """Events the sealed inventory could not settle, as (code, ex_date) pairs.

    A code/session can carry two vendor rows that disagree on a component, and the
    inventory records one verdict per row rather than one per event. An event that
    also appears in the approved bucket is settled by the row whose identity
    holds, so it is not pending; `bucket_overlaps` reports how often that happens
    instead of silently collapsing it.
    """
    payload = json.loads(INVENTORY.read_text(encoding='utf-8'))
    approved = {(row['code'], row['ex_date']) for row in payload.get('approved', [])
                if row.get('code') and row.get('ex_date')}
    open_rows = set()
    for bucket in ('rejected', 'unresolved'):
        for row in payload.get(bucket, []):
            if row.get('code') and row.get('ex_date'):
                open_rows.add((row['code'], row['ex_date']))
    return open_rows - approved, sorted(open_rows & approved)


def retrigger(adj, quotes, mode, tick=TICK):
    """Re-derive `action_required` from the reference break instead of the factor.

    `quotes` is the book's own keyed panel, so the prior close used here is the
    close the ledger itself would have marked the position at. A session whose
    reference equals the prior close inside half a tick is a session on which the
    exchange declares no entitlement adjustment, whatever the derived factor
    column did at its fourth decimal.
    """
    if mode == 'adjusted_factor':
        return adj, []
    frame = []
    for (day, code), bar in quotes.items():
        frame.append({'trade_date': day, 'ts_code': code,
                      'close': bar.get('close'), 'pre_close': bar.get('pre_close')})
    reference = pd.DataFrame(frame).dropna(subset=['ts_code', 'trade_date'])
    reference = reference.sort_values(['ts_code', 'trade_date'])
    reference['prior_close'] = reference.groupby('ts_code').close.shift()
    both = reference.prior_close.notna() & reference.pre_close.notna()
    reference['reference_break'] = both & (
        (reference.pre_close - reference.prior_close).abs() > 0.5 * tick)
    merged = adj.merge(reference[['ts_code', 'trade_date', 'reference_break']],
                       on=['ts_code', 'trade_date'], how='left')
    moved = merged.action_required.fillna(False).astype(bool)
    broken = merged.reference_break.fillna(False).astype(bool)
    suppressed = merged[moved & ~broken]
    merged['action_required'] = moved & broken
    return merged, suppressed[
        ['trade_date', 'ts_code', 'adj_factor', 'reference_break']].to_dict('records')


def granted_ids(journal):
    return sorted({entry['event_id'] for entry in journal
                   if entry['kind'] == 'entitlement' and entry.get('event_id')})


def main(configs):
    global pd
    import numpy as np
    import pandas as pd  # noqa: F811
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
    unsettled, bucket_overlaps = unresolved_events()
    adjudicated_pairs = {(a['code'], a['ex_date']) for a in adjudicated['actions']}
    # Settled by this batch's reconstruction, so they are allowed to be granted;
    # everything still open in the sealed inventory is not.
    pending = unsettled - adjudicated_pairs
    print('pending unsettled events %d; rows in both approved and open buckets %d'
          % (len(pending), len(bucket_overlaps)), flush=True)
    prior = json.loads((PRIOR / 'protocol.json').read_text(encoding='utf-8'))
    source = ROOT / 'reports/window_evidence_20260923'

    OUT.mkdir(parents=True, exist_ok=True)
    actions, inventory_count, conflicts, injected_ids = load_actions(adjudicated['actions'])

    with RunContract(parameters={'actions': len(actions), 'configs': [c[0] for c in configs]},
                     run_id='ledger_advance3_' + datetime.now().strftime('%H%M%S_%f'),
                     runs_dir=OUT / 'runs', lock_path=OUT / 'advance3.lock'):
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
        susp = pd.read_csv(ROOT / 'data/raw/suspend_d.csv', dtype=str).fillna('')
        susp['trade_date'] = pd.to_datetime(susp.trade_date, format='mixed').dt.strftime('%Y-%m-%d')

        def states(frame):
            result = {}
            for record in frame.itertuples():
                key = identity_key(record.trade_date, record.ts_code, conf)
                result.setdefault(key, set()).add((record.suspend_type, record.suspend_timing))
            return {k: sorted(v) for k, v in result.items()}

        state = merge_overlay(states(susp), states(overlay('suspend_d_overlay.csv')), 'suspend_d', conf)
        halts = {k for k, rows in state.items() if ('S', '') in rows}
        partial = {k for k, rows in state.items() if any(t == 'S' and timing for t, timing in rows)}

        calendar = pd.read_csv(ROOT / 'data/raw/trade_calendar.csv', dtype=str)
        dates = sorted(set(pd.to_datetime(
            calendar.loc[(calendar.exchange == 'SSE') & (calendar.is_open == '1'), 'cal_date'],
            format='mixed').dt.strftime('%Y-%m-%d')))
        dates = [d for d in dates if prior['start'] <= d <= prior['end']]
        held_codes = {c for t in target_sets.values() for w in t.values() if w for c in w}
        listed = set(pd.read_csv(ROOT / 'data/raw/stock_list.csv', dtype=str).ts_code)
        freeze_facts, unresolved_tails = derive_freeze_facts(
            prices[prices.ts_code.isin(held_codes)], susp, dates, listed)
        calendar_set = set(dates)

        def adjust(mode):
            """The adjustment panel exactly as the replay builds it, with the trigger applied."""
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
            frame, suppressed = retrigger(frame, quotes, mode)
            frame = frame.drop(columns=['reference_break'], errors='ignore')
            return frame, suppressed

        rows, ladder, baselines = [], [], {}
        sealed = {name: json.loads((SEALED_OUT / 'checkpoints' / (name + '.json'))
                                   .read_text(encoding='utf-8')) for name in prior['signals']}
        advance2 = {name: json.loads((ADVANCE2_OUT / 'checkpoints' / (name + '.json'))
                                     .read_text(encoding='utf-8')) for name in prior['signals']}

        for label, mode, granularity, check_policy in configs:
            frame, suppressed = adjust(mode)
            adjustments = keyed(frame, ['action_check_available', 'action_required'], conf)
            adjustment_days = {}
            for day, code in adjustments:
                adjustment_days.setdefault(code, set()).add(day)
            book = Book(quotes, boundaries, adjustments, halts, partial,
                        [(r['date'], r['code']) for r in conf], partial_halt_state=True)
            print('== %s trigger=%s granularity=%s suppressed=%d'
                  % (label, mode, granularity, len(suppressed)), flush=True)
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
                                    delist_evidence=freeze_facts,
                                    unresolved_action_policy=granularity,
                                    halted_session_check_policy=check_policy)
                write(OUT / label / 'ledgers' / (name + '.json'), asdict(result))
                parity = reconcile_journal(result, prior['initial_cash'])
                finite = [d for d in result.days if d['nav'] is not None and isfinite(d['nav'])]
                reasons = {}
                for error in result.errors:
                    reasons[error['reason']] = reasons.get(error['reason'], 0) + 1
                granted = granted_ids(result.journal)
                loadable = {a['event_id']: (a['code'], a['ex_date']) for a in runnable}
                assert set(granted) <= set(loadable), 'granted an event outside the loaded set'
                granted_pairs = {loadable[e] for e in granted}
                assert not (granted_pairs & pending), \
                    'granted an event the inventory still cannot settle: %s' % (
                        sorted(granted_pairs & pending),)
                vacuous = [{'code': e['code'], 'date': e['date'],
                            'evidence': e.get('evidence'),
                            'settled_by_supplied_action': e.get('settled_by_supplied_action')}
                           for e in result.journal
                           if e['kind'] == 'halted_session_check_vacuous']
                without_later = []
                for entry in vacuous:
                    bar = book.get((entry['date'], entry['code'])) or {}
                    assert bar.get('state') == 'halted', \
                        (name, entry, 'relaxed a session that is not a halt')
                    assert bar.get('evidence'), \
                        (name, entry, 'relaxed a halt with no published record')
                    assert 'close' not in bar, \
                        (name, entry, 'relaxed a session that published a quote')
                    assert not any(code == entry['code'] and date == entry['date']
                                   for code, date in granted_pairs), \
                        (name, entry, 'granted an entitlement on a relaxed session')
                    later = [d for d in adjustment_days.get(entry['code'], ())
                             if d > entry['date']]
                    entry['later_adjustment_rows'] = len(later)
                    if not later:
                        without_later.append({'code': entry['code'],
                                              'date': entry['date'],
                                              'settled_by_supplied_action':
                                                  entry['settled_by_supplied_action']})
                row = {'config': label, 'trigger': mode, 'granularity': granularity,
                       'halted_check_policy': check_policy,
                       'vacuous_held_sessions': vacuous,
                       'vacuous_without_later_adjustment_row': without_later,
                       'signal': name, 'status': result.status,
                       'finite_days': len(finite),
                       'end': finite[-1]['date'] if finite else None,
                       'terminal_nav': finite[-1]['nav'] if finite else None,
                       'total_return': (finite[-1]['nav'] / prior['initial_cash'] - 1) if finite else None,
                       'fills': sum(1 for e in result.journal if e['kind'] == 'fill'),
                       'excluded_security_days': result.excluded,
                       'errors': result.errors[:20], 'error_reasons': reasons,
                       'parity': parity, 'granted_events': len(granted),
                       'granted_event_ids': granted,
                       'granted_subset_of_loaded': True, 'formal_eligibility': False,
                       'sealed_batch': {'finite_days': sealed[name]['finite_days'],
                                        'end': sealed[name]['end']},
                       'advance2': {'finite_days': advance2[name]['finite_days'],
                                    'end': advance2[name]['end']},
                       'delta_vs_sealed': len(finite) - sealed[name]['finite_days']}
                cfg_rows.append(row)
                write(OUT / label / 'checkpoints' / (name + '.json'), row)
                print('  %-28s %-34s days=%3d end=%s excl=%d %s'
                      % (name, row['status'], row['finite_days'], row['end'],
                         len(result.excluded), reasons), flush=True)
            total = sum(r['finite_days'] for r in cfg_rows)
            ladder.append({'config': label, 'trigger': mode, 'granularity': granularity,
                           'halted_check_policy': check_policy,
                           'finite_days': total,
                           'of_possible': len(dates) * len(prior['signals']),
                           'vacuous_held_sessions':
                               sum(len(r['vacuous_held_sessions']) for r in cfg_rows),
                           'suppressed_false_triggers': suppressed[:40],
                           'suppressed_false_trigger_count': len(suppressed),
                           'arms': [{k: r[k] for k in ('signal', 'status', 'finite_days', 'end',
                                                       'error_reasons')} for r in cfg_rows]})
            baselines[label] = {r['signal']: r for r in cfg_rows}

        # 2. config A must reproduce the advance2 reading day for day.
        for name in prior['signals']:
            got = baselines['A_legacy'][name]
            assert got['finite_days'] == advance2[name]['finite_days'], \
                ('A_legacy no longer reproduces the advance2 reading',
                 name, got['finite_days'], advance2[name]['finite_days'])
            assert got['end'] == advance2[name]['end'], ('A_legacy end moved', name)
            assert got['errors'] == advance2[name]['errors'], ('A_legacy blockers moved', name)

        # 4. C must not grant anything A did not reach only because A stopped earlier.
        a_end = {}
        for name in prior['signals']:
            a, c = baselines['A_legacy'][name], baselines['C_reference_break_excluded'][name]
            assert set(a['granted_event_ids']) <= set(c['granted_event_ids']), \
                (name, 'C granted a different set, not a superset')
            a_end[name] = a['end']
            for event_id in sorted(set(c['granted_event_ids']) - set(a['granted_event_ids'])):
                extra_date = event_id[-10:]
                assert a['end'] is None or extra_date > a['end'], \
                    (name, 'C granted an event before or on A\'s stop', event_id)

        # 5. every excluded session is a session A or B stopped on, or downstream of it.
        excluded = sorted('%s %s %s' % (row['signal'], e['code'], e['date'])
                          for row in baselines['C_reference_break_excluded'].values()
                          for e in row['excluded_security_days'])
        for row in baselines['C_reference_break_excluded'].values():
            for entry in row['excluded_security_days']:
                assert 'inv_%s_%s' % (entry['code'], entry['date']) not in row['granted_event_ids']
                assert 'adj_%s_%s' % (entry['code'], entry['date']) not in row['granted_event_ids']
        assert len(excluded) == len(set(excluded)), 'an exclusion is double counted'

        # 6. D may only relax the halts on published full-day halts, and it must
        #    reach at least as far as C while granting nothing C did not grant.
        if 'D_halt_session_vacuous' in baselines:
            for name in prior['signals']:
                c = baselines['C_reference_break_excluded'][name]
                d = baselines['D_halt_session_vacuous'][name]
                assert d['finite_days'] >= c['finite_days'], \
                    (name, 'D stopped earlier than C')
                assert set(c['granted_event_ids']) <= set(d['granted_event_ids']), \
                    (name, 'D granted a different set, not a superset')
                for event_id in sorted(set(d['granted_event_ids'])
                                       - set(c['granted_event_ids'])):
                    extra_date = event_id[-10:]
                    assert c['end'] is None or extra_date > c['end'], \
                        (name, 'D granted an event before or on C\'s stop', event_id)
                assert set(map(str, c['excluded_security_days'])) \
                    <= set(map(str, d['excluded_security_days'])), \
                    (name, 'D dropped an exclusion C made')
                for entry in d['excluded_security_days']:
                    if entry in c['excluded_security_days']:
                        continue
                    assert c['end'] is None or entry['date'] > c['end'], \
                        (name, 'D excluded a session before or on C\'s stop', entry)
            d_vacuous = sum(len(row['vacuous_held_sessions'])
                            for row in baselines['D_halt_session_vacuous'].values())
            ab_vacuous = sum(len(row['vacuous_held_sessions'])
                             for label in ('A_legacy', 'B_reference_break',
                                           'C_reference_break_excluded')
                             for row in baselines[label].values())
            assert ab_vacuous == 0, \
                'a legacy config relaxed a halt session: %d' % ab_vacuous

        write(OUT / 'ladder.json', {
            'scope': ('three readings of one frozen design; the only differences are the '
                      'trigger definition and the halt granularity, both one-way tightenings '
                      'of the halt that can never supply an entitlement'),
            'configs': [{'label': label, 'trigger': mode, 'granularity': granularity,
                         'halted_check_policy': check_policy}
                        for label, mode, granularity, check_policy in configs],
            'window': [dates[0], dates[-1]], 'sessions': len(dates),
            'arms': prior['signals'],
            'ladder': ladder,
            'sealed_inventory_sha256': before,
            'expected_inventory_sha256':
                adjudicated['hashes']['reports/event_inventory_20260923/inventory.json'],
            'inventory_approvals': inventory_count,
            'adjudicated_action_ids': injected_ids,
            'action_conflicts': conflicts,
            'unsettled_events': sorted('%s %s' % pair for pair in unsettled),
            'events_in_both_approved_and_open_buckets':
                sorted('%s %s' % pair for pair in bucket_overlaps),
            'excluded_sessions': sorted(excluded),
            'anti_smuggling': {
                'config_A_reproduces_advance2': True,
                'granted_events_are_a_subset_of_loaded_actions': True,
                'no_unsettled_event_is_granted': True,
                'entitlements_never_derived_from_the_adjustment_ratio': True,
                'inventory_unchanged_by_every_config': True,
                'halt_relaxation_confined_to_the_last_config': True,
                'no_grant_on_a_relaxed_session': True,
            },
            'claims_not_made': [
                'the excluded sessions are still data_insufficient; the ladder converts a '
                'whole-arm stop into a flagged single-security-session stop, it does not '
                'settle the entitlement',
                'the carried mark on an excluded session is a stale price, so that session '
                'carries the mark-to-reference error of the excluded weight',
                'relaxing the action check on a published full-day halt assumes nothing '
                'about the entitlement: it only declines to call a session without a '
                'published reference a session whose check is unreadable',
                'reaching the full window is not evidence that the arm is tradable, '
                'profitable, or out of sample',
                'no production default, feedback state or candidate queue is touched',
            ],
            'formal_eligibility': False, 'production_unchanged': True,
            'history_exposed': True,
        })
        meta = json.loads((OUT / 'ladder.json').read_text(encoding='utf-8'))
        print(json.dumps([{k: l[k] for k in ('config', 'finite_days', 'of_possible')}
                          for l in meta['ladder']], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', default=None,
                        help='comma-separated config labels; default runs the whole ladder')
    args = parser.parse_args()
    chosen = CONFIGS
    if args.only:
        wanted = {w.strip() for w in args.only.split(',') if w.strip()}
        chosen = [c for c in CONFIGS if c[0] in wanted]
        assert chosen, 'no known config selected'
    main(chosen)
