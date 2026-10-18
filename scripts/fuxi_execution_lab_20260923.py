"""Read-only market inputs, immutable isolated execution-ledger experiments."""

import argparse
import hashlib
import io
import json
import math
import shutil
import sys
import unittest
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from run_contract import RunContract, _atomic_json
from portfolio_execution_ledger import LotRule, run_ledger, reconcile_journal

OUT = ROOT / 'reports/execution_ledger_20260923'
OLD = ROOT / 'reports/holdings_20260923'
CODE = ['research/factor_alchemy/portfolio_execution_ledger.py',
        'research/factor_alchemy/tests/test_execution_ledger.py',
        'scripts/fuxi_execution_lab_20260923.py']
INPUTS = ['data/raw/' + name for name in ['daily_prices.csv', 'index_000985_daily.csv',
          'stk_limit.csv', 'suspend_d.csv', 'adj_factor_daily.csv', 'trade_calendar.csv',
          'stock_list.csv', 'kuaidao_gaps/dividend.csv']]
STATE = ['data/' + name for name in ['experience_memory.json', 'mab_scheduler_state.json',
         'trajectory_log.json', 'breed_fail_streak.json', 'jq_codegen_queue.json',
         'version_registry.json', 'stage1_optimization_proposals.json']]
PRODUCTION = ['research/factor_alchemy/s5_joint_filter.py', 'scripts/s5_fullpool_rerun.py',
              'research/factor_alchemy/multi_stage_validator.py',
              'codex_automations/fuxi_runtime_policy.json']
DOCS = ['reports/fuxi_execution_ledger_checkpoint_20260923.md',
        '.workbuddy/memory/2026-09-23.md', '.workbuddy/memory/_fuxi_landing_details.md']


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, payload):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if hasattr(value, 'item'):
            return clean(value.item())
        return None if isinstance(value, float) and not math.isfinite(value) else value
    _atomic_json(path, clean(payload))


def freeze():
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('frozen protocol already exists')
    source = read(OLD / 'protocol.json')
    for name, item in source['data'].items():
        assert digest(ROOT / name) == item['sha256'], name
    files = INPUTS + STATE + PRODUCTION
    hashes = {name: digest(ROOT / name) for name in files}
    for name in STATE + PRODUCTION + ['CHANGELOG.md', 'scripts/version_bump.py',
                 '.workbuddy/memory/2026-09-23.md', '.workbuddy/memory/_fuxi_landing_details.md']:
        dest = OUT / 'baseline' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    write(OUT / 'protocol.json', {
        'revision': 'EL-20260923.1', 'production_version': 'v0.7.56',
        'created_at': datetime.now().astimezone().isoformat(), 'sha256': hashes,
        'signals': source['signal_hashes'], 'initial_cash': 10000000, 'top_n': 80,
        'cost_bps': 15, 'start': '2024-12-30', 'end': '2026-09-09',
        'sizing': 'previous signal close NAV; fixed quantities; sells before buys',
        'execution': 'next session open proxy; exact daily limits; retry until superseded',
        'selection': 'finite signal/raw close, value desc/amount desc/code asc; no fill-day reranking',
        'unknown_state': 'terminal data_insufficient; no invented mark/action',
        'actions': 'explicit record-date entitlements only; local dividend schema not execution-complete',
        'formal_eligibility': False, 'production_defaults_changed': False,
        'comparability': 'fresh cash start and next-open quantities differ from S5; not a single-switch causal A/B',
        'history_exposed': True, 'market_trials': 4,
        'limits': ['open-price fill is an execution proxy, not guaranteed auction fill',
                   'no liquidity/market-impact model', '15bps aggregate cost, not exact tax/commission schedule',
                   'stock lifecycle coverage incomplete; no delisting cash or code-map inferred',
                   'missing adj-factor event checks block rather than imply no action']})
    print('Frozen EL-20260923.1: four existing signals, no production registration.')


def tests():
    if (OUT / 'tests.json').exists():
        archive = OUT / 'test_attempts' / (datetime.now().strftime('%H%M%S_%f') + '.json')
        archive.parent.mkdir(exist_ok=True)
        shutil.copy2(OUT / 'tests.json', archive)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'research/factor_alchemy/tests'))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    write(OUT / 'tests.json', {'successful': result.wasSuccessful(), 'tests_run': result.testsRun,
                             'code_sha256': {n: digest(ROOT / n) for n in CODE},
                             'output': stream.getvalue()})
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise SystemExit(1)


def load_panel(name, columns, *, date_column='trade_date', lower='20241220'):
    import pandas as pd
    pieces = []
    maximum, minimum, count = '', '99999999', 0
    for part in pd.read_csv(ROOT / name, usecols=columns, dtype={date_column: str, 'ts_code': str},
                            chunksize=300000, low_memory=False):
        parsed = pd.to_datetime(part[date_column], format='mixed', errors='raise')
        if parsed.isna().any():
            raise ValueError('missing date in ' + str(name))
        values = parsed.dt.strftime('%Y%m%d')
        if len(values):
            maximum, minimum = max(maximum, values.max()), min(minimum, values.min())
        count += len(part)
        part[date_column] = parsed.dt.strftime('%Y-%m-%d')
        part = part[values.between(lower, '20260909')].copy()
        pieces.append(part)
    frame = pd.concat(pieces, ignore_index=True)
    if frame.duplicated(['ts_code', date_column]).any():
        raise ValueError('duplicate date/code in ' + name)
    return frame, {'rows_total': count, 'source_min': minimum, 'source_max': maximum,
                   'window_rows': len(frame), 'columns': columns}


def market(destination=None, actions=()):
    import numpy as np
    import pandas as pd
    protocol = read(OUT / 'protocol.json')
    destination = destination or OUT
    destination.mkdir(exist_ok=True)
    if (destination / 'market.json').exists():
        raise RuntimeError('market evidence already exists; do not overwrite')
    for name, expected in protocol['sha256'].items():
        assert digest(ROOT / name) == expected, name
    run_id = 'execution_' + datetime.now().strftime('%H%M%S_%f')
    code_hashes = {name: digest(ROOT / name) for name in CODE}
    for name in CODE:
        dest = OUT / 'execution_source' / run_id / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    with RunContract(parameters={'protocol_sha256': digest(OUT / 'protocol.json'), 'code': code_hashes,
                                 'data': protocol['sha256'], 'signals': protocol['signals'],
                                 'destination': str(destination),
                                 'corporate_actions': [asdict(a) for a in actions]},
                     run_id=run_id, runs_dir=OUT / 'runs', lock_path=OUT / 'execution.lock'):
        price, pinfo = load_panel(INPUTS[0], ['ts_code', 'trade_date', 'open', 'close', 'pre_close', 'amount'])
        print('Loaded raw prices', len(price), flush=True)
        limits, linfo = load_panel('data/raw/stk_limit.csv', ['ts_code', 'trade_date', 'up_limit', 'down_limit'])
        adj, ainfo = load_panel('data/raw/adj_factor_daily.csv', ['ts_code', 'trade_date', 'adj_factor'])
        adj = adj.sort_values(['ts_code', 'trade_date'])
        adj['previous_adj'] = adj.groupby('ts_code').adj_factor.shift()
        adj['action_check_available'] = (adj.adj_factor > 0) & (adj.previous_adj > 0)
        adj['action_required'] = ~np.isclose(adj.adj_factor, adj.previous_adj, rtol=1e-7, atol=0)
        susp = pd.read_csv(ROOT / 'data/raw/suspend_d.csv', dtype=str)
        suspend_dates = pd.to_datetime(susp.trade_date, format='mixed', errors='raise')
        suspend_max = suspend_dates.max().strftime('%Y%m%d')
        susp['trade_date'] = suspend_dates.dt.strftime('%Y-%m-%d')
        full = susp[(susp.suspend_type == 'S') & susp.suspend_timing.fillna('').eq('')]
        halt_keys = set(zip(full.trade_date, full.ts_code))
        partial = susp[(susp.suspend_type == 'S') & ~susp.suspend_timing.fillna('').eq('')]
        partial_keys = set(zip(partial.trade_date, partial.ts_code))
        dividends = pd.read_csv(ROOT / 'data/raw/kuaidao_gaps/dividend.csv', dtype=str)
        stock_columns = pd.read_csv(ROOT / 'data/raw/stock_list.csv', nrows=0).columns.tolist()
        calendar = pd.read_csv(ROOT / 'data/raw/trade_calendar.csv', dtype=str)
        calendar = calendar[(calendar.exchange == 'SSE') & (calendar.is_open == '1')]
        ds = pd.to_datetime(calendar.cal_date, format='mixed', errors='raise').sort_values().drop_duplicates()
        ds = ds[(ds >= '2024-12-23') & (ds <= protocol['end'])]
        dates = [d.strftime('%Y-%m-%d') for d in ds if d.strftime('%Y-%m-%d') >= protocol['start']]
        week_ends = pd.Series(ds.to_numpy(), index=ds.dt.to_period('W-SUN')).groupby(level=0).max()
        signal_dates = {d.strftime('%Y-%m-%d') for d in week_ends}
        merged = price.merge(limits, on=['trade_date', 'ts_code'], how='left', validate='one_to_one')
        merged = merged.merge(adj[['trade_date', 'ts_code', 'action_required', 'action_check_available']],
                              on=['trade_date', 'ts_code'], how='left', validate='one_to_one')
        indexed = merged.set_index(['trade_date', 'ts_code']).sort_index()
        adj_index = adj.set_index(['trade_date', 'ts_code']).sort_index()

        class Book:
            def __init__(self):
                self.cache = {}

            def get(self, key, default=None):
                if key in self.cache:
                    return self.cache[key]
                if key in indexed.index:
                    row = indexed.loc[key].to_dict()
                    row['state'] = 'unknown' if key in halt_keys or key in partial_keys else 'trading'
                    row['action_check_available'] = bool(row['action_check_available'] == True)
                    row['action_required'] = bool(row['action_required'] == True)
                elif key in halt_keys:
                    row = {'state': 'halted', 'evidence': 'suspend_d.csv/full-day-S',
                           'action_check_available': False}
                    if key in adj_index.index:
                        ar = adj_index.loc[key]
                        row.update(action_check_available=bool(ar.action_check_available),
                                   action_required=bool(ar.action_required))
                else:
                    row = {'state': 'unknown', 'action_check_available': False}
                self.cache[key] = row
                return row

        book = Book()
        source_inventory = {'price': pinfo, 'limits': linfo, 'adjustments': ainfo,
                            'suspend_max': suspend_max, 'full_halt_rows': len(full),
                            'dividend_columns': list(dividends.columns), 'dividend_rows': len(dividends),
                            'dividend_implemented_rows': int(dividends.div_proc.eq('实施').sum()),
                            'missing_action_fields': [c for c in ['pay_date', 'div_listdate', 'stk_div'] if c not in dividends],
                            'stock_columns': stock_columns, 'has_delist_date': 'delist_date' in stock_columns,
                            'calendar_days': len(dates), 'start': dates[0], 'end': dates[-1]}
        write(destination / 'input_inventory.json', source_inventory)
        old_anomalies = []
        for old_row in read(OLD / 'marking_attribution.json')['rows']:
            for cell in old_row['cells']:
                observation = book.get((cell['date'], cell['ts_code']))
                old_anomalies.append({**cell, 'signal': old_row['signal'],
                                      'ledger_state': observation.get('state'),
                                      'action_check_available': observation.get('action_check_available'),
                                      'action_required': observation.get('action_required'),
                                      'resolved_as_economic_evidence': False})
        assert len(old_anomalies) == 165
        write(destination / 'old_anomaly_followup.json', {'cells': old_anomalies,
              'note': 'State checks only; no replacement returns, no changed historical verdicts.'})
        rows = []
        for name, expected in protocol['signals'].items():
            signal_path = ROOT / 'reports/drought_20260923/signals' / (name + '.parquet')
            assert digest(signal_path) == expected
            signal = pd.read_parquet(signal_path)
            signal['trade_date'] = pd.to_datetime(signal.trade_date).dt.strftime('%Y-%m-%d')
            signal = signal[signal.trade_date.isin(signal_dates)]
            if signal.duplicated(['trade_date', 'ts_code']).any():
                raise ValueError('duplicate signal keys')
            frame = signal.merge(price[['trade_date', 'ts_code', 'close', 'amount']],
                                 on=['trade_date', 'ts_code'], how='inner', validate='one_to_one')
            frame = frame[np.isfinite(frame.factor_value) & (frame.close > 0)]
            targets, insufficient = {}, []
            groups = {day: group for day, group in frame.groupby('trade_date')}
            for day in sorted(signal_dates):
                group = groups.get(day, frame.iloc[:0])
                if len(group) < protocol['top_n']:
                    insufficient.append({'date': day, 'available': len(group)})
                    targets[day] = None
                    continue
                selected = group.sort_values(['factor_value', 'amount', 'ts_code'], ascending=[False, False, True]).head(protocol['top_n'])
                targets[day] = {c: 1 / protocol['top_n'] for c in selected.ts_code}
            assert targets
            rules = {c: LotRule(200, 1) if c.startswith(('688', '689')) else LotRule(100, 1)
                     if c.endswith('.BJ') else LotRule() for weights in targets.values() if weights for c in weights}
            write(destination / 'targets' / (name + '.json'), {'targets': targets, 'insufficient': insufficient})
            result = run_ledger(dates, book, targets, initial_cash=protocol['initial_cash'],
                                cost_bps=protocol['cost_bps'], rules=rules, actions=actions)
            parity = reconcile_journal(result, protocol['initial_cash'])
            assert parity['finite_nav_days'] > 0
            write(destination / 'ledgers' / (name + '.json'), asdict(result))
            row = {'signal': name, 'status': result.status, 'formal_eligibility': False,
                   'processed_days': len(result.days), 'end': result.days[-1]['date'],
                   'errors': result.errors, 'parity': parity, 'code_sha256': code_hashes,
                   'filled_orders': sum(o['filled'] > 0 for o in result.orders),
                   'order_reasons': dict(Counter(o['reason'] for o in result.orders if o['reason'])),
                   'signal_coverage_insufficient_periods': insufficient}
            rows.append(row)
            write(destination / 'checkpoints' / (name + '.json'), row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        for name, expected in {**protocol['sha256'], **code_hashes}.items():
            assert digest(ROOT / name) == expected, name
        event_needs = []
        dividend_dates = pd.to_datetime(dividends.ex_date, format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
        for day, code in sorted({(e['date'], e['code']) for row in rows for e in row['errors']}):
            matches = dividends[(dividends.ts_code == code) & (dividend_dates == day)]
            event_needs.append({'date': day, 'code': code, 'local_dividend_records': matches.to_dict('records'),
                                'required': ['implementation status', 'cash entitlement and payment date',
                                             'stock distribution and listing date', 'lifecycle/capital-event check'],
                                'resolved': False})
        write(destination / 'event_data_requests.json', {'requests': event_needs, 'external_requests_made': 0})
        write(destination / 'market.json', {'rows': rows, 'scope': 'isolated execution engineering, no economic qualification',
                                   'all_inputs_and_production_unchanged': True, 'code_sha256': code_hashes})


def seal():
    if (OUT / 'release_audit.json').exists():
        raise RuntimeError('already sealed')
    protocol = read(OUT / 'protocol.json')
    for name, expected in protocol['sha256'].items():
        assert digest(ROOT / name) == expected, name
    test = read(OUT / 'tests.json')
    assert test['successful']
    for name, expected in test['code_sha256'].items():
        assert digest(ROOT / name) == expected, name
    market = read(OUT / 'market_final/market.json')
    assert len(market['rows']) == 4
    for row in market['rows']:
        assert row['code_sha256'] == test['code_sha256']
        assert row['parity']['compared_days'] > 0 and not row['formal_eligibility']
        name = row['signal']
        assert read(OUT / 'market_final/ledgers' / (name + '.json')) == read(OUT / 'ledgers' / (name + '.json'))
    old_audit = read(OLD / 'release_audit.json')
    for name, expected in old_audit['evidence_sha256'].items():
        assert digest(OLD / name) == expected, name
    for path in (OUT / 'runs').glob('*/manifest.json'):
        assert read(path)['status'] != 'running'
    for name in DOCS[1:]:
        old = (OUT / 'baseline' / name).read_bytes()
        path = ROOT / name
        new = path.read_bytes()
        if b'\r\n' in old and b'\n' not in old.replace(b'\r\n', b''):
            fixed = new.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
            if fixed != new:
                path.write_bytes(fixed)
            new = fixed
        assert new.startswith(old), name
    for name in CODE + DOCS:
        dest = OUT / 'release' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    evidence = {p.relative_to(OUT).as_posix(): digest(p) for p in OUT.rglob('*') if p.is_file()}
    write(OUT / 'release_audit.json', {'revision': protocol['revision'], 'production_version': 'v0.7.56',
          'production_unchanged': True, 'formal_eligibility': False, 'registered_as_landed': False,
          'tests_run': test['tests_run'], 'old_evidence_files_unchanged': len(old_audit['evidence_sha256']),
          'state_and_data_sha256': protocol['sha256'], 'evidence_sha256': evidence,
          'registration_blocker': 'single_writer_confirmed=false; explicit user confirmation pending'})
    print(json.dumps({'sealed': True, 'tests': test['tests_run'], 'files': len(evidence)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['freeze', 'tests', 'market', 'market-final', 'seal'])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.step == 'market-final':
        market(OUT / 'market_final')
    else:
        globals()[args.step]()
