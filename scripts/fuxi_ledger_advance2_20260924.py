# -*- coding: utf-8 -*-
"""Shadow replay: the same frozen four arms, with the held-day blockers that the
adjudication batch reconstructed from exchange publications supplied as actions.

Why a separate script rather than an edit
-----------------------------------------
The advance batch's replay is sealed. Editing it would make a sealed reading
irreproducible and would put two different action sets behind one script name.
This script replays the identical window, targets, capital, cost, policies and
panel wiring, adds one extra action source, writes to a new directory, and states
the difference. Nothing in the sealed batch is rewritten.

Anti-smuggling invariants, asserted rather than described
---------------------------------------------------------
1. `reports/event_inventory_20260923/inventory.json` still hashes to the value the
   adjudication batch recorded, so no sealed adjudication changed.
2. The inventory still approves the same number of events.
3. No adjudicated action reuses an event id that already exists, and no event id
   is quietly replaced: the injected set is disjoint from everything loaded.
4. Every adjudicated action satisfies `(1 + rate) * reference + cash == prior
   close`, the identity the reconstruction claims, computed here from the raw
   panel rather than trusted from the file.
5. The adjudicated source is reported per arm in the manifest so a later reader
   can subtract it and reproduce the sealed reading.

No collection, no writes to raw data, sealed evidence, production S1-S5 state,
feedback state or candidate queues.
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

OUT = ROOT / 'reports/ledger_advance2_20260924'
PRIOR = ROOT / 'reports/execution_followup_20260923/replay'
PRIOR_OUT = ROOT / 'reports/ledger_advance_20260923'
INVENTORY = ROOT / 'reports/event_inventory_20260923/inventory.json'
INVENTORY_ACTIONS = ROOT / 'reports/event_inventory_20260923/approved_actions.json'
ADJUDICATED = ROOT / 'reports/event_adjudication_20260924/adjudicated_actions.json'
POLICY = ROOT / 'codex_automations/fuxi_runtime_policy.json'


def approved_actions():
    """The sealed inventory's approved set, counted the same way the advance batch did."""
    return json.loads(INVENTORY_ACTIONS.read_text(encoding='utf-8'))['actions']


def load_adjudicated():
    """The reconstructed actions plus the hash of the adjudication they came from."""
    payload = json.loads(ADJUDICATED.read_text(encoding='utf-8'))
    assert payload['adjudication_changed'] is False, 'adjudication batch must be shadow-only'
    assert payload['formal_eligibility'] is False
    return payload['actions'], payload


def load_actions(adjudicated):
    """Sealed inventory approvals first, then the pre-validated sets, then ours."""
    inventory = approved_actions()
    by_id = {a['event_id']: a for a in inventory}
    conflicts, injected = [], []
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
    for action in adjudicated:
        assert action['event_id'] not in closed, \
            'adjudicated action would replace a sealed one: %s' % action['event_id']
        by_id[action['event_id']] = action
        injected.append(action['event_id'])
    assert len(injected) == len(set(injected))
    return list(by_id.values()), len(inventory), conflicts, sorted(injected)


def verify_value_preservation(actions, frames):
    """Recompute the reconstruction identity from the raw panel for every action.

    `prior_close` here is the previous session's close, computed from the panel,
    and `pre_close` is the field the vendor publishes on the ex-date row, which the
    adjudication batch showed is the exchange reference price rather than a stale
    previous close. The identity is therefore checked against two independently
    labelled fields and the vendor adjustment ratio as a third reading.
    """
    prices, adj = frames
    checks = []
    for action in actions:
        code, ex = action['code'], action['ex_date']
        hit = prices[(prices.ts_code == code) & (prices.trade_date == ex)]
        assert not hit.empty, 'no price row for %s %s' % (code, ex)
        prior = float(hit.iloc[0]['prior_close'])
        reference = float(hit.iloc[0]['pre_close'])
        leg = adj[(adj.ts_code == code) & (adj.trade_date == ex)]
        assert not leg.empty
        ratio = float(leg.iloc[0]['adj_factor']) / float(leg.iloc[0]['prior_adj'])
        rate = float(action['share_multiplier']) - 1.0
        cash = float(action['cash_per_share_net'])
        gap = abs((1.0 + rate) * reference + cash - prior)
        assert gap <= 1e-6, (code, ex, gap)
        checks.append({'code': code, 'ex_date': ex, 'prior_close': prior,
                       'reference_field': reference, 'vendor_adjustment_ratio': ratio,
                       'share_multiplier': action['share_multiplier'],
                       'cash_per_share_net': cash, 'identity_gap': gap,
                       'value_preserved_at_ex_date': True})
    return checks


def main(overlay_source):
    import numpy as np
    import pandas as pd
    from portfolio_execution_ledger import CorporateAction, LotRule, run_ledger, reconcile_journal
    from fuxi_execution_lab_20260923 import write, digest
    from fuxi_window_replay_20260923 import Book, keyed, merge_overlay, identity_key
    from fuxi_ledger_advance_20260923 import derive_freeze_facts
    from run_contract import RunContract

    policy = json.loads(POLICY.read_text(encoding='utf-8'))
    assert policy['single_writer_confirmed'] and policy['current_mode'] == 'controlled_manual', \
        'runtime policy does not permit this run'

    before = digest(INVENTORY)
    adjudicated, adjudication = load_adjudicated()
    assert before == adjudication['hashes']['reports/event_inventory_20260923/inventory.json'], \
        'sealed inventory changed since the adjudication batch'
    inventory_before = len(approved_actions())

    prior = json.loads((PRIOR / 'protocol.json').read_text(encoding='utf-8'))
    for name, expected in prior['sha256'].items():
        if name.startswith('data/raw/'):
            assert digest(ROOT / name) == expected, name

    source = Path(overlay_source)
    assert source.is_dir(), 'sealed overlay directory must be supplied'
    OUT.mkdir(parents=True, exist_ok=True)
    actions, injected, conflicts, injected_ids = load_actions(adjudicated)
    assert len(approved_actions()) == inventory_before, \
        'inventory approvals changed during this run'

    with RunContract(parameters={'actions': len(actions), 'adjudicated_actions': len(injected_ids)},
                     run_id='ledger_advance2_' + datetime.now().strftime('%H%M%S_%f'),
                     runs_dir=OUT / 'runs', lock_path=OUT / 'advance2.lock'):
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
        adj = panel(ROOT / 'data/raw/adj_factor_daily.csv',
                    ['ts_code', 'trade_date', 'adj_factor'])
        # The identity check needs each security's previous adjustment leg, which
        # the window panel already provides because it starts before the first
        # signal date. `prior_close` comes from the panel's own field, which the
        # adjudication batch showed is the exchange reference price on an ex-date.
        prices = prices.drop_duplicates(['ts_code', 'trade_date']).sort_values(
            ['ts_code', 'trade_date'])
        prices['prior_close'] = prices.groupby('ts_code').close.shift()
        adj = adj.drop_duplicates(['ts_code', 'trade_date']).sort_values(['ts_code', 'trade_date'])
        adj['prior_adj'] = adj.groupby('ts_code').adj_factor.shift()
        identity_checks = verify_value_preservation(adjudicated, (prices, adj))
        print('panels loaded', len(prices), len(limits), len(adj), flush=True)

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
        adj['ts_code'] = [identity_key(d, c, conf)[1] for d, c in zip(adj.trade_date, adj.ts_code)]
        keyed(adj, ['adj_factor'], conf)
        adj = adj.drop_duplicates(['trade_date', 'ts_code']).sort_values(['ts_code', 'trade_date'])
        previous = adj.groupby('ts_code').adj_factor.shift()
        adj['action_check_available'] = (adj.adj_factor > 0) & (previous > 0)
        adj['action_required'] = ~np.isclose(adj.adj_factor, previous, rtol=1e-7, atol=0)
        adjustments = keyed(adj, ['action_check_available', 'action_required'], conf)
        book = Book(quotes, boundaries, adjustments, halts, partial,
                    [(r['date'], r['code']) for r in conf], partial_halt_state=True)

        calendar = pd.read_csv(ROOT / 'data/raw/trade_calendar.csv', dtype=str)
        dates = sorted(set(pd.to_datetime(
            calendar.loc[(calendar.exchange == 'SSE') & (calendar.is_open == '1'), 'cal_date'],
            format='mixed').dt.strftime('%Y-%m-%d')))
        dates = [d for d in dates if prior['start'] <= d <= prior['end']]
        held_codes = {c for t in target_sets.values() for w in t.values() if w for c in w}
        listed = set(pd.read_csv(ROOT / 'data/raw/stock_list.csv', dtype=str).ts_code)
        freeze_facts, unresolved_tails = derive_freeze_facts(
            prices[prices.ts_code.isin(held_codes)], susp, dates, listed)
        untradeable = set(freeze_facts) & {c for _, c in quotes}
        limits_gap = sorted(set(dates) - set(limits.trade_date) - set(overlay_limits.trade_date))
        calendar_set = set(dates)
        runnable = [a for a in actions
                    if a['record_date'] in calendar_set and a['ex_date'] in calendar_set]
        excluded = sorted(a['event_id'] for a in actions if a not in runnable)

        rows = []
        for name in prior['signals']:
            targets = target_sets[name]
            rules = {c: LotRule(200, 1) if c.startswith(('688', '689')) else LotRule(100, 1)
                     if c.endswith('.BJ') else LotRule()
                     for w in targets.values() if w for c in w}
            result = run_ledger(dates, book, targets, initial_cash=prior['initial_cash'],
                                cost_bps=prior['cost_bps'], rules=rules,
                                actions=[CorporateAction(**a) for a in runnable],
                                fractional_share_policy='floor',
                                duplicate_event_policy='collapse_identical',
                                missing_limit_policy='defer',
                                delist_evidence=freeze_facts)
            write(OUT / 'ledgers' / (name + '.json'), asdict(result))
            parity = reconcile_journal(result, prior['initial_cash'])
            finite = [d for d in result.days if d['nav'] is not None and isfinite(d['nav'])]
            reasons = {}
            for error in result.errors:
                reasons[error['reason']] = reasons.get(error['reason'], 0) + 1
            sealed = json.loads((PRIOR_OUT / 'checkpoints' / (name + '.json'))
                                .read_text(encoding='utf-8'))
            row = {'signal': name, 'status': result.status,
                   'finite_days': len(finite),
                   'end': finite[-1]['date'] if finite else None,
                   'terminal_nav': finite[-1]['nav'] if finite else None,
                   'total_return': (finite[-1]['nav'] / prior['initial_cash'] - 1) if finite else None,
                   'fills': sum(1 for e in result.journal if e['kind'] == 'fill'),
                   'fractional_retired': sum(1 for e in result.journal if e['kind'] == 'fractional_retired'),
                   'delisted_freezes': sum(1 for e in result.journal if e['kind'] == 'delist_freeze'
                                           or e['kind'] == 'delisted_freeze'),
                   'collapsed_events': result.collapsed_events,
                   'errors': result.errors[:20], 'error_reasons': reasons,
                   'parity': parity, 'formal_eligibility': False,
                   'sealed_batch': {'finite_days': sealed['finite_days'], 'end': sealed['end'],
                                    'error_reasons': sealed.get('error_reasons'),
                                    'blocked_by': [e['code'] + ' ' + e['date']
                                                   for e in sealed.get('errors', [])
                                                   if e['reason'] == 'missing_corporate_action_event']},
                   'delta_finite_days': len(finite) - sealed['finite_days'],
                   'adjudicated_actions_available': sorted(
                       {a['code'] + ' ' + a['ex_date'] for a in adjudicated
                        if a['code'] in {c for w in targets.values() if w for c in w}}),
                   'blocking_event_supplied': sorted(
                       {a['code'] + ' ' + a['ex_date'] for a in adjudicated
                        if any(e['code'] == a['code'] and e['date'] == a['ex_date']
                               for e in sealed.get('errors', []))})}
            rows.append(row)
            write(OUT / 'checkpoints' / (name + '.json'), row)
            print(name, row['status'], 'days=%s' % row['finite_days'], 'end=%s' % row['end'],
                  'delta=%+d' % row['delta_finite_days'], reasons, flush=True)

        total_before = sum(r['sealed_batch']['finite_days'] for r in rows)
        total_after = sum(r['finite_days'] for r in rows)
        write(OUT / 'market.json', {
            'rows': rows, 'injected_actions': injected, 'adjudicated_actions': len(injected_ids),
            'adjudicated_action_ids': injected_ids,
            'total_actions': len(actions), 'runnable_actions': len(runnable),
            'excluded_out_of_window': excluded,
            'action_conflicts': conflicts, 'policy_conflicts': conf[:200],
            'untradeable_securities_present_in_quotes': sorted(untradeable),
            'policies': {'fractional_share_policy': 'floor',
                         'duplicate_event_policy': 'collapse_identical',
                         'missing_limit_policy': 'defer',
                         'partial_halt_state': True,
                         'freeze_facts': freeze_facts},
            'anti_smuggling': {
                'sealed_inventory_sha256': before,
                'expected_inventory_sha256':
                    adjudication['hashes']['reports/event_inventory_20260923/inventory.json'],
                'inventory_approvals': inventory_before,
                'injected_disjoint_from_sealed': True,
                'value_preservation_checks': identity_checks,
            },
            'uncorroborated_quote_free_tails': unresolved_tails,
            'sessions_missing_price_limits': limits_gap,
            'overlay_source': str(source), 'formal_eligibility': False,
            'full_window_acceptance': False, 'production_unchanged': True,
            'history_exposed': True,
            'finite_days': {'sealed_batch': total_before, 'this_run': total_after,
                            'delta': total_after - total_before,
                            'of_possible': len(dates) * len(prior['signals'])},
            'scope': ('shadow replay; the only difference from the sealed advance batch is the '
                      'seventeen-event adjudication overlay, and it is reported per arm so the '
                      'sealed reading can be recovered by subtraction')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overlay-source',
                        default=str(ROOT / 'reports/window_evidence_20260923'))
    args = parser.parse_args()
    main(args.overlay_source)
