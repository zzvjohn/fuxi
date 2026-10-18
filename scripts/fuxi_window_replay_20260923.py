"""Isolated four-signal overlay replay; never modifies raw or sealed evidence."""
import argparse
from dataclasses import asdict
from datetime import datetime
from math import isclose, isfinite
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from fuxi_execution_lab_20260923 import read, write, digest, load_panel
from fuxi_window_reference_20260923 import simulate
from portfolio_execution_ledger import CorporateAction, LotRule, run_ledger, reconcile_journal
from run_contract import RunContract

OUT = ROOT / 'reports/window_execution_20260923'
EVIDENCE = ROOT / 'reports/window_evidence_20260923'
PRIOR = ROOT / 'reports/execution_followup_20260923/replay'
CUTOVER = '2025-10-09'


def canonical(day, code):
    """Accounting identity is constant; historical trading identity is date-bound."""
    if code == '839680.BJ':
        if day >= CUTOVER:
            raise ValueError('old_identity_after_official_cutover')
        return '920680.BJ'
    return code


def identity_key(day, code, conflicts):
    try:
        return (day, canonical(day, code))
    except ValueError:
        key = (day, '920680.BJ')
        conflicts.append({'source': 'identity', 'date': day, 'code': key[1],
                          'original': code, 'resolution': 'block_conflicting_key_in_book'})
        return key


def keyed(frame, columns, conflicts=None):
    conflicts = [] if conflicts is None else conflicts
    output = {}
    for record in frame.to_dict('records'):
        key = identity_key(record['trade_date'], record['ts_code'], conflicts)
        value = {c: record[c] for c in columns}
        if key in output and output[key] != value:
            conflicts.append({'source': 'duplicate_identity', 'date': key[0], 'code': key[1],
                              'original': output[key], 'overlay': value,
                              'resolution': 'block_conflicting_key_in_book'})
            continue
        output[key] = value
    return output


def merge_overlay(base, overlay, label, conflicts):
    result = dict(base)
    for key, value in overlay.items():
        if key in base and base[key] != value:
            conflicts.append({'source': label, 'date': key[0], 'code': key[1],
                              'original': base[key], 'overlay': value,
                              'resolution': 'block_conflicting_key_in_book'})
        result[key] = value
    return result


class Book:
    def __init__(self, quotes, limits, adjustments, halts, partial, conflicts=(),
                 partial_halt_state=False):
        self.quotes, self.limits, self.adjustments = quotes, limits, adjustments
        self.halts, self.partial = halts, partial
        self.conflicts = set(conflicts)
        # A key can carry a quote and a suspension record at once when the halt
        # covered the open only. Left at the default the state stays 'unknown',
        # which is what every previously sealed batch saw; enabling it separates
        # that evidenced case from a genuine source disagreement.
        self.partial_halt_state = partial_halt_state

    def get(self, key, default=None):
        day, code = key
        key = (day, canonical(day, code))
        action = self.adjustments.get(key, {'action_check_available': False})
        if key in self.conflicts:
            return {**action, 'state': 'unknown', 'action_check_available': False,
                    'evidence': 'conflicting_original_overlay_evidence'}
        # No last-price cash recovery or inferred settlement of delisted holdings.
        if code == '600898.SH' and day >= '2025-02-10':
            return {**action, 'state': 'unknown', 'evidence': 'delisted_equity_valuation_missing'}
        if key in self.quotes:
            # A full-day suspension cannot also produce a quote, so that pairing
            # is a source disagreement and stays fail-closed and unevidenced.
            state, evidence = 'trading', {}
            if key in self.halts:
                state = 'unknown'
            elif key in self.partial and self.partial_halt_state:
                state = 'partial_halt'
                evidence = {'evidence': 'intraday_suspension_window_with_published_quote'}
            elif key in self.partial:
                state = 'unknown'
            return {**self.quotes[key], **self.limits.get(key, {}), **action,
                    **evidence, 'state': state}
        if key in self.halts:
            return {**action, 'state': 'halted', 'evidence': 'original_or_overlay_explicit_full_day_S'}
        return {**action, 'state': 'unknown'}


def compare_reference(result, targets, quotes, limits, halts, actions):
    share_days = [a['ex_date'] for a in actions if a['share_multiplier'] != 1]
    finite = [r for r in result.days if r['nav'] is not None and isfinite(r['nav'])]
    assert len(finite) > 0
    independent, fills = simulate([r['date'] for r in finite], targets, quotes, limits, halts, actions)
    for actual, reference in zip(finite, independent):
        assert actual['date'] == reference['date']
        assert {c: p['shares'] for c, p in actual['positions'].items()} == reference['positions'], (actual['date'], 'positions')
        assert {c: p['locked'] for c, p in actual['positions'].items()} == reference['locked'], (actual['date'], 'locked')
        for key in ['cash', 'receivable', 'nav']:
            assert isclose(actual[key], reference[key], abs_tol=1e-6, rel_tol=0), (actual['date'], key)
    primary = [r for r in result.journal if r['kind'] == 'fill' and r['date'] <= finite[-1]['date']]
    assert len(primary) == len(fills) > 0, ('fill_counts', len(primary), len(fills))
    for actual, reference in zip(primary, fills):
        for key in ['date', 'code', 'quantity']:
            assert actual[key] == reference[key], ('fill', key, actual, reference)
        for key in ['price', 'fee']:
            assert isclose(actual[key], reference[key], abs_tol=1e-6, rel_tol=0)
    return {'days': len(finite), 'fills': len(fills), 'end': finite[-1]['date'],
            'independent_execution': True, 'independent_data_source': False,
            'first_supplied_share_event': min(share_days) if share_days else None,
            'coverage': 'finite prefix including integer share distributions/listing locks; excludes unknown terminal days'}


def compare_prior(result, old):
    current = {r['date']: r for r in result.days if r['nav'] is not None and isfinite(r['nav'])}
    previous = [r for r in old['days'] if r['nav'] is not None]
    differences = []
    for row in previous:
        now = current.get(row['date'])
        changed = ['no_longer_finite'] if now is None else [
            k for k in ['cash', 'receivable', 'nav']
            if not isclose(row[k], now[k], abs_tol=1e-6, rel_tol=0)]
        if now is not None and row['positions'] != now['positions']:
            changed.append('positions')
        if changed:
            differences.append({'date': row['date'], 'fields': changed, 'before': row, 'after': now})
    assert len(previous) > 0
    return {'compared_days': len(previous), 'unchanged': not differences, 'differences': differences}


def delisting_sidecar():
    path = ROOT / 'reports/delisting_evidence_20260923/400257_daily_raw.json'
    raw = read(path)
    prior = read(PRIOR / 'ledgers/llm_family_1_1_fresh_llm.json')
    quantity = prior['days'][-1]['positions']['600898.SH']['shares']
    prices = []
    for line in raw['data']['klines']:
        d, opening, closing, high, low, hands, amount, *_ = line.split(',')
        prices.append({'date': d, 'open': float(opening), 'close': float(closing),
                       'high': float(high), 'low': float(low), 'volume_hands': float(hands),
                       'volume_shares_assuming_100_per_hand': float(hands) * 100,
                       'turnover_yuan': float(amount)})
    assert len(prices) > 0 and prices[0]['date'] == '2025-04-11'
    first = prices[0]
    write(OUT / 'delisting_sidecar.json', {
        'accounting_identity': '600898.SH', 'successor_market_identity': '400257',
        'delist_date': '2025-02-10', 'transfer_start': '2025-04-11', 'held_shares': quantity,
        'source_sha256': digest(path), 'observed_quotes': prices,
        'first_session_total_market_shares': first['volume_shares_assuming_100_per_hand'],
        'position_exceeds_entire_first_session_volume': quantity > first['volume_shares_assuming_100_per_hand'],
        'all_shares_first_open_sale_supported': False, 'cash_recovery_booked': False,
        'unvalued_interval': ['2025-02-10', '2025-04-10'],
        'remaining': ['account-level entitlement confirmation', 'transition-period PIT valuation policy',
                      'historical transfer calendar and capacity/fees', 'complete corporate actions'],
        'scope': 'equity identity and observed-price sidecar; no synthetic NAV, sell, or full-window qualification'})


def run(identity_source, assembly_name='assembly_distribution.json'):
    import pandas as pd
    import numpy as np
    assert not (OUT / 'protocol.json').exists(), 'immutable run already started; use a new output directory for a retry'
    assembly = read(EVIDENCE / assembly_name)
    cash_assembly = read(EVIDENCE / 'assembly.json')
    prior_protocol = read(PRIOR / 'protocol.json')
    policy = read(ROOT / 'codex_automations/fuxi_runtime_policy.json')
    assert policy['single_writer_confirmed'] and policy['current_mode'] == 'controlled_manual'
    source = Path(identity_source)
    assert source.is_file(), 'official identity evidence must be supplied'
    data_names = [n for n in prior_protocol['sha256'] if n.startswith('data/raw/')]
    for name in data_names:
        assert digest(ROOT / name) == prior_protocol['sha256'][name], name
    artifacts = [EVIDENCE / n for n in [assembly_name, 'suspend_d_overlay.csv',
                 'stk_limit_overlay.csv', 'dividend_overlay.csv']] + [source, Path(__file__),
                 EVIDENCE / 'assembly.json',
                 ROOT / 'scripts/fuxi_window_reference_20260923.py',
                 ROOT / 'research/factor_alchemy/portfolio_execution_ledger.py',
                 ROOT / 'research/factor_alchemy/tests/test_window_overlay.py',
                 ROOT / 'reports/execution_events_20260923/approved_actions.json',
                 ROOT / 'reports/execution_followup_20260923/validation.json',
                 ROOT / 'reports/delisting_evidence_20260923/400257_daily_raw.json',
                 PRIOR / 'protocol.json']
    hashes = {str(p): digest(p) for p in artifacts}
    hashes.update({str(ROOT / n): digest(ROOT / n) for n in data_names})
    for name, expected in prior_protocol['signals'].items():
        assert digest(ROOT / 'reports/drought_20260923/signals' / (name + '.parquet')) == expected
        target_path = PRIOR / 'targets' / (name + '.json')
        hashes[str(target_path)] = digest(target_path)
        ledger_path = PRIOR / 'ledgers' / (name + '.json')
        hashes[str(ledger_path)] = digest(ledger_path)
    protocol = {**prior_protocol, 'revision': 'EL-20260923.window-overlay',
                'sha256': hashes, 'signals': prior_protocol['signals'],
                'identity': {'canonical': '920680.BJ', 'historical_before': '839680.BJ',
                             'cutover': CUTOVER, 'evidence': str(source),
                             'contract': 'vendor historical quotes may be normalized; old-code state maps before cutover'},
                'assumptions': 'validated cash/share actions; next-open proxy; 15bps aggregate; no PIT certification; independent Decimal finite-prefix reference',
                'history_exposed': True, 'formal_eligibility': False}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'protocol.json', protocol)
    delisting_sidecar()
    with RunContract(parameters={'protocol_sha256': digest(OUT / 'protocol.json')},
                     run_id='window_replay_' + datetime.now().strftime('%H%M%S_%f'),
                     runs_dir=OUT / 'runs', lock_path=OUT / 'replay.lock'):
        prices, pinfo = load_panel('data/raw/daily_prices.csv', ['ts_code', 'trade_date', 'open', 'close', 'pre_close'])
        limits, linfo = load_panel('data/raw/stk_limit.csv', ['ts_code', 'trade_date', 'up_limit', 'down_limit'])
        adj, ainfo = load_panel('data/raw/adj_factor_daily.csv', ['ts_code', 'trade_date', 'adj_factor'])
        print('Loaded raw panels', flush=True)
        def overlay(name):
            frame = pd.read_csv(EVIDENCE / name, dtype={'ts_code': str, 'trade_date': str}).fillna('')
            frame['trade_date'] = pd.to_datetime(frame.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
            return frame
        conflicts = []
        quotes = keyed(prices, ['open', 'close', 'pre_close'], conflicts)
        boundaries = merge_overlay(keyed(limits, ['up_limit', 'down_limit'], conflicts),
            keyed(overlay('stk_limit_overlay.csv'), ['up_limit', 'down_limit'], conflicts), 'stk_limit', conflicts)
        susp = pd.read_csv(ROOT / 'data/raw/suspend_d.csv', dtype=str).fillna('')
        susp['trade_date'] = pd.to_datetime(susp.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
        # Multiple partial sessions can exist on one stock day, so use sets of records.
        def states(frame):
            result = {}
            for r in frame.itertuples():
                key = identity_key(r.trade_date, r.ts_code, conflicts)
                result.setdefault(key, set()).add((r.suspend_type, r.suspend_timing))
            return {k: sorted(v) for k, v in result.items()}
        state = merge_overlay(states(susp), states(overlay('suspend_d_overlay.csv')), 'suspend_d', conflicts)
        halts = {k for k, rows in state.items() if ('S', '') in rows}
        partial = {k for k, rows in state.items() if any(t == 'S' and timing for t, timing in rows)}
        quote_codes = {c for _, c in quotes}
        unmapped_bse = sorted({c for _, c in state if c.endswith('.BJ') and c not in quote_codes})
        adj['ts_code'] = [identity_key(d, c, conflicts)[1] for d, c in zip(adj.trade_date, adj.ts_code)]
        # Record duplicate values before deduplication; conflicted cells are blocked.
        keyed(adj, ['adj_factor'], conflicts)
        adj = adj.drop_duplicates(['trade_date', 'ts_code'])
        adj = adj.sort_values(['ts_code', 'trade_date'])
        previous = adj.groupby('ts_code').adj_factor.shift()
        adj['action_check_available'] = (adj.adj_factor > 0) & (previous > 0)
        adj['action_required'] = ~np.isclose(adj.adj_factor, previous, rtol=1e-7, atol=0)
        adjustments = keyed(adj, ['action_check_available', 'action_required'], conflicts)
        book = Book(quotes, boundaries, adjustments, halts, partial,
                    [(r['date'], r['code']) for r in conflicts])
        calendar = pd.read_csv(ROOT / 'data/raw/trade_calendar.csv', dtype=str)
        dates = sorted(set(pd.to_datetime(calendar.loc[(calendar.exchange == 'SSE') & (calendar.is_open == '1'), 'cal_date'], format='mixed').dt.strftime('%Y-%m-%d')))
        dates = [d for d in dates if protocol['start'] <= d <= protocol['end']]
        prior_actions = read(ROOT / 'reports/execution_events_20260923/approved_actions.json')['actions']
        prior_actions += read(ROOT / 'reports/execution_followup_20260923/validation.json')['new_actions']
        actions_by_id = {a['event_id']: a for a in prior_actions}
        for action in assembly.get('approved_actions', assembly.get('approved_cash_actions', [])):
            old = actions_by_id.get(action['event_id'])
            if old:
                if {k:v for k,v in old.items() if k != 'evidence'} != {k:v for k,v in action.items() if k != 'evidence'}:
                    key = (action['ex_date'], action['code'])
                    conflicts.append({'source': 'corporate_action', 'date': key[0], 'code': key[1],
                                      'original': old, 'overlay': action,
                                      'resolution': 'block_conflicting_key_in_book'})
                    book.conflicts.add(key)
                    continue
            actions_by_id[action['event_id']] = action
        actions = list(actions_by_id.values())
        write(OUT / 'approved_actions.json', {'actions': actions})
        write(OUT / 'overlay_conflicts.json', {'conflicts': conflicts, 'quote_halt_conflict_count': len(set(quotes) & (halts | partial))})
        write(OUT / 'input_inventory.json', {'price': pinfo, 'limits': linfo, 'adj': ainfo, 'actions': len(actions),
              'request_gaps': cash_assembly['request_gaps'], 'rejected_cash_events': cash_assembly['rejected_events'],
              'rejected_share_events': assembly.get('rejected_share_actions', []),
              'bse_state_codes_absent_from_quote_identity_universe': unmapped_bse,
              'identity_coverage_complete': False,
              'note': 'A rejected cash event may be separately validated as a share event; consult approved IDs.'})
        rows = []
        for name in protocol['signals']:
            targets = read(PRIOR / 'targets' / (name + '.json'))['targets']
            rules = {c: LotRule(200, 1) if c.startswith(('688', '689')) else LotRule(100, 1)
                     if c.endswith('.BJ') else LotRule() for w in targets.values() if w for c in w}
            result = run_ledger(dates, book, targets, initial_cash=protocol['initial_cash'],
                                cost_bps=protocol['cost_bps'], rules=rules,
                                actions=[CorporateAction(**a) for a in actions])
            payload = asdict(result)
            write(OUT / 'ledgers' / (name + '.json'), payload)
            parity = reconcile_journal(result, protocol['initial_cash'])
            try:
                reference = {'status': 'matched', **compare_reference(payload_to_namespace(payload),
                    targets, quotes, boundaries, halts, actions)}
            except (AssertionError, ValueError, KeyError) as exc:
                reference = {'status': 'mismatch_or_insufficient', 'error_type': type(exc).__name__,
                             'error': str(exc), 'independent_certification': False}
            prior = compare_prior(payload_to_namespace(payload), read(PRIOR / 'ledgers' / (name + '.json')))
            write(OUT / 'prior_comparison' / (name + '.json'), prior)
            row = {'signal': name, 'status': result.status, 'errors': result.errors,
                   'parity': parity, 'reference': reference, 'prior_compared_days': prior['compared_days'],
                   'prior_unchanged': prior['unchanged'], 'formal_eligibility': False}
            rows.append(row)
            write(OUT / 'checkpoints' / (name + '.json'), row)
            print(name, row['status'], reference, result.errors, flush=True)
        for path, expected in hashes.items():
            assert digest(path) == expected, path
        write(OUT / 'market.json', {'rows': rows, 'formal_eligibility': False,
              'full_window_acceptance': False, 'production_unchanged': True,
              'scope': 'independent reference certifies finite cash/share prefixes, not source completeness'})


def payload_to_namespace(payload):
    from types import SimpleNamespace
    return SimpleNamespace(**payload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity-source', required=True)
    parser.add_argument('--assembly', default='assembly_distribution.json')
    parser.add_argument('--output', default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.output)
    run(args.identity_source, args.assembly)
