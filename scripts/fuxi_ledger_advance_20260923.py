"""Advance the frozen four-signal ledger with the proven-action inventory.

Replays the same four frozen signals, targets, capital, cost and window as the
sealed batches, but supplies corporate actions sourced from the reference-
identity inventory instead of the narrower legacy assembler. Two opt-in research
policies are enabled and reported explicitly: whole-share retirement for the
fractional part of a share entitlement, and a frozen accounting mark for a
documented unlisted position. Neither produces cash, a recoverable price, or a
promotion. Production S5, defaults, feedback state and candidate queues are not
touched.
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from math import isclose, isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
sys.path.insert(0, str(ROOT / 'scripts'))

OUT = ROOT / 'reports/ledger_advance_20260923'
INVENTORY = ROOT / 'reports/event_inventory_20260923/approved_actions.json'
PRIOR = ROOT / 'reports/execution_followup_20260923/replay'
POLICY = ROOT / 'codex_automations/fuxi_runtime_policy.json'

# Independently documented anchor. The SSE delisting announcement 2025-01-27
# (no delisting-arrangement period) and the security-list delist date both place
# the last tradable session before 2025-02-10. This is used only to certify that
# the derivation below reproduces a known date; it is never the sole basis for a
# freeze.
DOCUMENTED_FREEZE = {'600898.SH': '2025-02-10'}


def derive_freeze_facts(prices, susp, dates, listed):
    """Derive non-tradability only where a quote-free tail is corroborated.

    A missing quote is not proof of delisting: the raw price panel has ragged
    tails for still-listed securities, so gaps alone would manufacture freezes.
    Two independent conditions are required instead - the security is absent from
    the current security master (it no longer has a market) and the suspend table
    covers every session between its last quote and the freeze date. Cases with a
    quote-free tail but no suspend coverage stay blocked and are reported.
    """
    facts, blocked = {}, []
    last_quote = prices.groupby('ts_code').trade_date.max()
    halted = {code: sorted(group.trade_date)
              for code, group in susp[susp.suspend_type == 'S'].groupby('ts_code')}
    for code, last in sorted(last_quote.items()):
        after = [d for d in dates if d > last]
        if not after or code in listed:
            continue
        covered = [d for d in halted.get(code, []) if d > last]
        if not covered or covered != after[:len(covered)] or covered[-1] == dates[-1]:
            blocked.append({'code': code, 'last_quote': last, 'sessions_without_quote': len(after),
                            'suspend_sessions_after': len(covered),
                            'reason': 'quote_free_tail_without_contiguous_suspend_evidence'})
            continue
        freeze = dates[dates.index(covered[-1]) + 1]
        facts[code] = {
            'date': freeze,
            'evidence': ('derived: last quote %s; suspend_d covers %s..%s contiguously; absent '
                         'from the %d-name current security master'
                         % (last, covered[0], covered[-1], len(listed)))}
    for code, day in DOCUMENTED_FREEZE.items():
        assert facts.get(code, {}).get('date') == day, \
            'derived freeze for %s must reproduce the documented date %s' % (code, day)
        facts[code]['evidence'] += ('; reproduces the documented delisting date from the SSE '
                                    'announcement 2025-01-27')
    return facts, blocked


def load_actions():
    """Merge the proven inventory with the previously validated action sets."""
    inventory = json.loads(INVENTORY.read_text(encoding='utf-8'))['actions']
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
    return list(by_id.values()), len(inventory), conflicts


def main(source_dir):
    import numpy as np
    import pandas as pd
    from portfolio_execution_ledger import CorporateAction, LotRule, run_ledger, reconcile_journal
    from fuxi_execution_lab_20260923 import read, write, digest
    from fuxi_window_replay_20260923 import Book, keyed, merge_overlay, identity_key
    from run_contract import RunContract

    policy = json.loads(POLICY.read_text(encoding='utf-8'))
    assert policy['single_writer_confirmed'] and policy['current_mode'] == 'controlled_manual', \
        'runtime policy does not permit this run'

    prior = json.loads((PRIOR / 'protocol.json').read_text(encoding='utf-8'))
    for name, expected in prior['sha256'].items():
        if name.startswith('data/raw/'):
            assert digest(ROOT / name) == expected, name

    source = Path(source_dir)
    assert source.is_dir(), 'sealed overlay directory must be supplied'
    OUT.mkdir(parents=True, exist_ok=True)
    actions, injected, conflicts = load_actions()

    with RunContract(parameters={'actions': len(actions)},
                     run_id='ledger_advance_' + datetime.now().strftime('%H%M%S_%f'),
                     runs_dir=OUT / 'runs', lock_path=OUT / 'advance.lock'):
        # The earliest signal date can precede the replay window start, and the
        # ledger sizes orders from the signal-session close, so the quote panel
        # must retain it. Targets are read once here and reused below.
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
        print('panels loaded', len(prices), len(limits), len(adj), flush=True)

        def overlay(name):
            frame = pd.read_csv(source / name, dtype={'ts_code': str, 'trade_date': str}).fillna('')
            frame['trade_date'] = pd.to_datetime(frame.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
            return frame

        conf = []
        quotes = keyed(prices, ['open', 'close', 'pre_close'], conf)
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
        # Only securities that appear in some target can ever be held, so the
        # freeze derivation is scoped to them rather than to the whole panel.
        held_codes = {c for t in target_sets.values() for w in t.values() if w for c in w}
        listed = set(pd.read_csv(ROOT / 'data/raw/stock_list.csv', dtype=str).ts_code)
        freeze_facts, unresolved_tails = derive_freeze_facts(
            prices[prices.ts_code.isin(held_codes)], susp, dates, listed)
        untradeable = set(freeze_facts) & {c for _, c in quotes}
        # Sessions the published limit panel does not cover. Orders cannot be
        # shown to have been executable there, so they are refused, not invented.
        limits_gap = sorted(set(dates) - set(limits.trade_date) - set(overlay_limits.trade_date))
        # An entitlement registered before the replay window belongs to a period
        # in which the book is empty by construction: it cannot attach to a
        # position. Such events are excluded explicitly, never granted blindly.
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
            row = {'signal': name, 'status': result.status,
                   'finite_days': len(finite),
                   'end': finite[-1]['date'] if finite else None,
                   'terminal_nav': finite[-1]['nav'] if finite else None,
                   'total_return': (finite[-1]['nav'] / prior['initial_cash'] - 1) if finite else None,
                   'fills': sum(1 for e in result.journal if e['kind'] == 'fill'),
                   'fractional_retired': sum(1 for e in result.journal if e['kind'] == 'fractional_retired'),
                   'delisted_freezes': sum(1 for e in result.journal if e['kind'] == 'delisted_freeze'),
                   'collapsed_events': result.collapsed_events,
                   'deferred_no_limit': sum(1 for o in result.orders
                                            if o.get('reason') == 'no_limit_evidence'),
                   'refused_partial_halt_open': sum(1 for o in result.orders
                                                    if o.get('reason') == 'partial_halt_open'),
                   'errors': result.errors[:20], 'error_reasons': reasons,
                   'parity': parity, 'formal_eligibility': False}
            rows.append(row)
            write(OUT / 'checkpoints' / (name + '.json'), row)
            print(name, row['status'], 'days=%s' % row['finite_days'],
                  'end=%s' % row['end'], reasons, flush=True)

        write(OUT / 'market.json', {
            'rows': rows, 'injected_actions': injected, 'total_actions': len(actions),
            'runnable_actions': len(runnable), 'excluded_out_of_window': excluded,
            'action_conflicts': conflicts, 'policy_conflicts': conf[:200],
            'untradeable_securities_present_in_quotes': sorted(untradeable),
            'policies': {'fractional_share_policy': 'floor',
                         'duplicate_event_policy': 'collapse_identical',
                         'missing_limit_policy': 'defer',
                         'partial_halt_state': True,
                         'freeze_facts': freeze_facts},
            'uncorroborated_quote_free_tails': unresolved_tails,
            'sessions_missing_price_limits': limits_gap,
            'overlay_source': str(source), 'formal_eligibility': False,
            'full_window_acceptance': False, 'production_unchanged': True,
            'history_exposed': True,
            'scope': ('proven-action inventory extends the finite prefix; the two research '
                      'policies are declared assumptions, not recovered market facts')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overlay-source',
                        default=str(ROOT / 'reports/window_evidence_20260923'))
    args = parser.parse_args()
    main(args.overlay_source)
