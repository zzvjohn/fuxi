"""Frozen four-signal diagnostics; historical labels are not executable returns."""
import argparse
import math
import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from run_contract import RunContract
from fuxi_execution_lab_20260923 import read, write, digest, load_panel
from fuxi_reference_cash_20260923 import simulate

OUT = ROOT / 'reports/target_diagnostics_20260923'
PRIOR = ROOT / 'reports/execution_followup_20260923'


def label_dates(calendar, stamp, horizon=5):
    """Entry strictly after signal; exit five market sessions after entry."""
    later = [d for d in calendar if d > stamp]
    return (later[0], later[horizon]) if len(later) > horizon else None


def cross_section(frame, top_n=80):
    import numpy as np
    # Select before observing label availability: missing labels never backfill Top80.
    eligible = frame[np.isfinite(frame.factor_value) & (frame.close > 0)].copy()
    selected = eligible.sort_values(['factor_value', 'amount', 'ts_code'],
                                   ascending=[False, False, True]).head(top_n)
    finite = eligible[np.isfinite(eligible.label)]
    top = selected[np.isfinite(selected.label)]
    ic = finite.factor_value.corr(finite.label, method='spearman') if len(finite) > 1 else float('nan')
    return {'eligible_n': len(eligible), 'label_n': len(finite),
            'missing_labels': len(eligible) - len(finite), 'selected_n': len(selected),
            'top_label_n': len(top), 'top_missing_labels': len(selected) - len(top),
            'rank_ic': ic, 'pool_label_mean': finite.label.mean(),
            'top_label_mean': top.label.mean(),
            'top_minus_pool': top.label.mean() - finite.label.mean(),
            'top_complete': len(selected) == top_n and len(top) == top_n}


def quarantine_duplicate_sizes(frame):
    duplicated = frame.duplicated(['trade_date', 'ts_code'], keep=False)
    return frame[~duplicated].copy(), {'quarantined_rows': int(duplicated.sum()),
        'quarantined_keys': int(frame.loc[duplicated, ['trade_date', 'ts_code']].drop_duplicates().shape[0]),
        'input_rows': len(frame), 'remaining_rows': int((~duplicated).sum()),
        'policy': 'exclude every duplicate key, even equal values; never choose first/last'}


def freeze():
    assert not (OUT / 'protocol.json').exists()
    source = read(PRIOR / 'replay/protocol.json')
    names = list(source['signals'])
    books = {n: read(PRIOR / 'replay/ledgers' / (n + '.json')) for n in names}
    common = sorted(set.intersection(*[{r['date'] for r in b['days'] if r['nav'] is not None}
                                      for b in books.values()]))
    assert len(common) > 0
    paths = ['scripts/fuxi_target_diagnostics_20260923.py',
             'scripts/fuxi_reference_cash_20260923.py', 'scripts/fuxi_execution_lab_20260923.py',
             'research/factor_alchemy/run_contract.py',
             'reports/execution_followup_20260923/replay/protocol.json',
             'reports/execution_followup_20260923/validation.json',
             'reports/execution_events_20260923/approved_actions.json']
    paths += ['data/raw/' + x for x in ['daily_prices.csv', 'adj_factor_daily.csv',
              'stk_limit.csv', 'suspend_d.csv', 'trade_calendar.csv', 'daily_basic.csv']]
    for n in names:
        paths += ['reports/drought_20260923/signals/' + n + '.parquet',
                  'reports/execution_followup_20260923/replay/targets/' + n + '.json',
                  'reports/execution_followup_20260923/replay/ledgers/' + n + '.json']
    hashes = {p: digest(ROOT / p) for p in paths}
    for n, h in source['signals'].items():
        assert hashes['reports/drought_20260923/signals/' + n + '.parquet'] == h
    write(OUT / 'protocol.json', {
        'created_at': datetime.now().astimezone().isoformat(), 'signals': source['signals'],
        'hashes': hashes, 'common_ledger_dates': common, 'cost_bps': [0, 15, 30],
        'initial_cash': 10000000, 'top_n': 80, 'label_horizon_market_sessions': 5,
        'prediction_start': '2024-12-27', 'prediction_end': '2026-09-09',
        'selection': 'signal descending / contemporaneous amount descending / code ascending',
        'label': 'next market-session adjusted open to open five market sessions later; no forward fill',
        'pool': 'each frozen signal finite cross-section with positive same-day raw close; no present-day universe filter',
        'missing': 'no replacement of missing Top80 labels; report partial and complete-only pairs separately',
        'size': 'previous market-session total_mv, log cross-sectional means; quarantine all duplicate keys; historical vendor snapshot, not proven vintage PIT',
        'industry': 'not_evaluated: no verified PIT industry history',
        'execution': 'same Decimal independent cash engine, next-open fill proxy; costs rerun, not linear subtraction',
        'scope': 'partial stage2 diagnosis, no search, no production write, no eligibility',
        'research_exposed': True, 'r5_unknown_samples': 0,
        'full_window_execution_valid': False})
    print('Frozen protocol:', len(common), 'common cash-ledger dates', common[0], common[-1], flush=True)


def run():
    import numpy as np
    import pandas as pd
    protocol = read(OUT / 'protocol.json')
    assert not (OUT / 'summary.json').exists()
    for p, h in protocol['hashes'].items():
        assert digest(ROOT / p) == h, p
    with RunContract(parameters={'protocol_sha256': digest(OUT / 'protocol.json'),
                                 'code_data_signal_hashes': protocol['hashes']},
                     run_id='target_' + datetime.now().strftime('%H%M%S_%f'),
                     runs_dir=OUT / 'runs', lock_path=OUT / 'diagnostic.lock'):
        price, _ = load_panel('data/raw/daily_prices.csv',
                             ['ts_code', 'trade_date', 'open', 'close', 'amount'])
        adj, _ = load_panel('data/raw/adj_factor_daily.csv', ['ts_code', 'trade_date', 'adj_factor'])
        prices = price.merge(adj, on=['trade_date', 'ts_code'], validate='one_to_one', how='left')
        prices['adj_open'] = prices.open * prices.adj_factor
        prices.loc[(prices.open <= 0) | (prices.adj_factor <= 0), 'adj_open'] = np.nan
        print('Loaded price and adjustments:', len(prices), flush=True)
        limits, _ = load_panel('data/raw/stk_limit.csv', ['ts_code', 'trade_date', 'up_limit', 'down_limit'])
        cal = pd.read_csv(ROOT / 'data/raw/trade_calendar.csv', dtype=str)
        cal = cal[(cal.exchange == 'SSE') & (cal.is_open == '1')]
        dates = sorted(pd.to_datetime(cal.cal_date, format='mixed').dt.strftime('%Y-%m-%d').unique())
        dates = [d for d in dates if '2024-12-20' <= d <= protocol['prediction_end']]
        series = pd.Series(pd.to_datetime(dates))
        weekly = series.groupby(series.dt.to_period('W-SUN')).max().dt.strftime('%Y-%m-%d').tolist()
        weekly = [d for d in weekly if d >= protocol['prediction_start']]
        targets = {n: read(PRIOR / 'replay/targets' / (n + '.json'))['targets'] for n in protocol['signals']}
        codes = {c for t in targets.values() for d, w in t.items()
                 if d <= protocol['common_ledger_dates'][-1] and w for c in w}
        short = price[price.ts_code.isin(codes) & (price.trade_date <= protocol['common_ledger_dates'][-1])]
        quotes = short.set_index(['trade_date', 'ts_code']).to_dict('index')
        boundaries = limits[limits.ts_code.isin(codes) & (limits.trade_date <= protocol['common_ledger_dates'][-1])]
        boundaries = boundaries.set_index(['trade_date', 'ts_code']).to_dict('index')
        susp = pd.read_csv(ROOT / 'data/raw/suspend_d.csv', dtype=str).fillna('')
        susp['trade_date'] = pd.to_datetime(susp.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
        halts = {(r.trade_date, r.ts_code) for r in susp.itertuples() if r.suspend_type == 'S' and not r.suspend_timing}
        actions = read(ROOT / 'reports/execution_events_20260923/approved_actions.json')['actions']
        actions += read(PRIOR / 'validation.json')['new_actions']
        action_keys = {(a['ex_date'], a['code']) for a in actions}
        adj = adj.sort_values(['ts_code', 'trade_date'])
        adj['prior_adj'] = adj.groupby('ts_code').adj_factor.shift()
        adjidx = adj.set_index(['trade_date', 'ts_code'])
        ledger_rows = []
        for name in protocol['signals']:
            for bps in protocol['cost_bps']:
                book, fills = simulate(protocol['common_ledger_dates'], targets[name], quotes, boundaries,
                                       halts, actions, bps=bps)
                assert len(book) == len(protocol['common_ledger_dates']) > 0 and len(fills) > 0
                # Reject unmodelled adjustment events even for a counterfactual quantity path.
                previous = {}
                for day in book:
                    for code in set(previous) | set(day['positions']):
                        ar = adjidx.loc[(day['date'], code)]
                        assert ar.adj_factor > 0 and ar.prior_adj > 0
                        if not np.isclose(ar.adj_factor, ar.prior_adj, rtol=1e-7, atol=0):
                            assert (day['date'], code) in action_keys, (name, bps, day['date'], code)
                    previous = day['positions']
                if bps == 15:
                    primary = {r['date']: r for r in read(PRIOR / 'replay/ledgers' / (name + '.json'))['days']}
                    for d in book:
                        p = primary[d['date']]
                        assert d['positions'] == {c: v['shares'] for c, v in p['positions'].items()}
                        assert all(math.isclose(d[k], p[k], abs_tol=1e-6, rel_tol=0) for k in ['nav', 'cash', 'receivable'])
                nav_before = dict(zip([d['date'] for d in book], [10000000] + [d['nav'] for d in book[:-1]]))
                gross_turnover = sum(abs(f['quantity']) * f['price'] / nav_before[f['date']] for f in fills)
                fees = sum(f['fee'] for f in fills)
                row = {'signal': name, 'cost_bps': bps, 'paired_days': len(book),
                       'start': book[0]['date'], 'end': book[-1]['date'],
                       'total_return': book[-1]['nav'] / 10000000 - 1,
                       'fees': fees, 'fees_initial_cash_fraction': fees / 10000000,
                       'gross_two_way_turnover': gross_turnover,
                       'mean_cash_fraction': sum(d['cash'] / d['nav'] for d in book) / len(book),
                       'fills': len(fills), 'fifteen_bps_primary_parity_days': len(book) if bps == 15 else None,
                       'status': 'valid_short_prefix_only'}
                ledger_rows.append(row)
                write(OUT / 'cost_ledgers' / (name + '_' + str(bps) + '.json'), {'days': book, 'fills': fills, 'summary': row})
                print('Cost ledger', name, bps, row['total_return'], flush=True)
        # Lagged size is descriptive only: archived vendor history is not verified vintage PIT.
        prior_date = {d: dates[i-1] for i, d in enumerate(dates) if i > 0}
        wanted_size_dates = {prior_date[d] for d in weekly if d in prior_date}
        pieces = []
        for part in pd.read_csv(ROOT / 'data/raw/daily_basic.csv', usecols=['ts_code', 'trade_date', 'total_mv'],
                                dtype={'trade_date': str}, chunksize=300000):
            part['trade_date'] = pd.to_datetime(part.trade_date, format='mixed', errors='raise').dt.strftime('%Y-%m-%d')
            pieces.append(part[part.trade_date.isin(wanted_size_dates)])
        size, size_quality = quarantine_duplicate_sizes(pd.concat(pieces, ignore_index=True))
        write(OUT / 'size_quality.json', size_quality)
        print('Size duplicate quarantine:', size_quality, flush=True)
        sizes = {d: g.set_index('ts_code').total_mv for d, g in size.groupby('trade_date')}
        byday = {d: g.set_index('ts_code') for d, g in prices.groupby('trade_date')}
        sections, summaries = [], []
        for name in protocol['signals']:
            signal = pd.read_parquet(ROOT / 'reports/drought_20260923/signals' / (name + '.parquet'))
            signal['trade_date'] = pd.to_datetime(signal.trade_date).dt.strftime('%Y-%m-%d')
            signal = signal[signal.trade_date.isin(weekly)]
            assert not signal.duplicated(['trade_date', 'ts_code']).any()
            for stamp, group in signal.groupby('trade_date'):
                window = label_dates(dates, stamp)
                if window is None:
                    sections.append({'signal': name, 'date': stamp, 'status': 'label_not_matured'})
                    continue
                entry, exit_day = window
                assert stamp < entry < exit_day
                frame = group.merge(byday[stamp][['close', 'amount']], left_on='ts_code', right_index=True, how='left', validate='one_to_one')
                frame['label'] = frame.ts_code.map(byday[exit_day].adj_open) / frame.ts_code.map(byday[entry].adj_open) - 1
                # Missing labels keep their membership; the reported partial-label mean is conditional.
                stats = cross_section(frame)
                chosen = frame[np.isfinite(frame.factor_value) & (frame.close > 0)].sort_values(
                    ['factor_value', 'amount', 'ts_code'], ascending=[False, False, True]).head(80)
                frozen = targets[name].get(stamp)
                if frozen:
                    assert set(chosen.ts_code) == set(frozen), (name, stamp)
                lag_size = sizes.get(prior_date[stamp], pd.Series(dtype=float))
                pool_size = frame.loc[np.isfinite(frame.factor_value) & (frame.close > 0), 'ts_code'].map(lag_size)
                top_size = chosen.ts_code.map(lag_size)
                pool_size, top_size = pool_size[pool_size > 0], top_size[top_size > 0]
                stats.update({'signal': name, 'date': stamp, 'entry': entry, 'exit': exit_day,
                              'status': 'valid_prediction_label_only', 'raw_signal_n': len(group),
                              'nonfinite_signal_n': int((~np.isfinite(group.factor_value)).sum()),
                              'size_lag_date': prior_date[stamp], 'size_pool_n': len(pool_size),
                              'size_top_n': len(top_size),
                              'top_minus_pool_log_size': float(np.log(top_size).mean() - np.log(pool_size).mean())})
                sections.append(stats)
            rows = [r for r in sections if r['signal'] == name and r['status'] == 'valid_prediction_label_only']
            assert len(rows) > 0
            print('Prediction sections', name, len(rows), flush=True)
            del signal
        common_dates = sorted(set.intersection(*[{r['date'] for r in sections if r['signal'] == n and
                       r['status'] == 'valid_prediction_label_only' and math.isfinite(r['rank_ic'])}
                       for n in protocol['signals']]))
        assert len(common_dates) > 0
        for name in protocol['signals']:
            rows = [r for r in sections if r['signal'] == name and r['date'] in common_dates]
            complete = [r for r in rows if r['top_complete']]
            assert len(rows) > 0
            summaries.append({'signal': name, 'paired_sections': len(rows),
                'first_signal': rows[0]['date'], 'last_signal': rows[-1]['date'],
                'rank_ic_mean': float(np.mean([r['rank_ic'] for r in rows])),
                'top_label_mean': float(np.mean([r['top_label_mean'] for r in rows])),
                'pool_label_mean': float(np.mean([r['pool_label_mean'] for r in rows])),
                'top_minus_pool_mean': float(np.mean([r['top_minus_pool'] for r in rows])),
                'complete_top_sections': len(complete),
                'complete_top_minus_pool_mean': float(np.mean([r['top_minus_pool'] for r in complete])) if complete else None,
                'label_observations': sum(r['label_n'] for r in rows),
                'missing_label_observations': sum(r['missing_labels'] for r in rows),
                'missing_top_label_observations': sum(r['top_missing_labels'] for r in rows),
                'top_minus_pool_log_size_mean': float(np.mean([r['top_minus_pool_log_size'] for r in rows]))})
        write(OUT / 'cross_sections.json', {'rows': sections, 'common_signal_dates': common_dates})
        write(OUT / 'summary.json', {'protocol_sha256': digest(OUT / 'protocol.json'),
              'cost_ledgers': ledger_rows, 'prediction': summaries,
              'industry': 'not_evaluated_no_verified_PIT_history',
              'size': 'lagged_vendor_snapshot_only_not_certified_PIT',
              'execution_baseline': 'four fixed Top80 portfolios; full-pool execution baseline not evaluated',
              'prediction_baseline': 'same-signal finite universe equal-weight label mean, not executable cash NAV',
              'full_stage2_acceptance': False, 'full_stage1_acceptance': False, 'formal_eligibility': False,
              'limitations': ['Historical research-exposed sample; no R5 unknown observations',
                              'Five-session adjusted-open labels do not certify tradability or cash distribution reinvestment',
                              'Missing endpoint labels create selection bias; partial and complete Top80 reported separately',
                              '15bps parity established on common prefix only; no full-window performance conclusion',
                              'No market impact/capacity/auction guarantee or complete account tax model',
                              'Signal original field timing not re-certified; no new signal or model search']})
        for p, h in protocol['hashes'].items():
            assert digest(ROOT / p) == h, p
        print('Completed immutable target diagnostics', flush=True)


class DiagnosticTests(unittest.TestCase):
    def test_duplicate_sizes_all_quarantined(self):
        import pandas as pd
        f = pd.DataFrame({'trade_date': ['d','d','d'], 'ts_code': ['A','A','B'], 'total_mv': [1,2,3]})
        clean, audit = quarantine_duplicate_sizes(f)
        self.assertEqual(clean.ts_code.tolist(), ['B'])
        self.assertEqual(audit['quarantined_keys'], 1)

    def test_next_open_and_market_session_exit(self):
        cal = ['2025-01-03', '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10', '2025-01-13']
        self.assertEqual(label_dates(cal, '2025-01-03'), ('2025-01-06', '2025-01-13'))
        self.assertIsNone(label_dates(cal, '2025-01-06'))

    def test_top_missing_not_backfilled(self):
        import pandas as pd
        f = pd.DataFrame({'ts_code': ['A','B','C'], 'factor_value': [3,2,1],
                          'close': [1,1,1], 'amount': [1,1,1], 'label': [float('nan'), .2, .1]})
        r = cross_section(f, 2)
        self.assertEqual(r['top_label_n'], 1)
        self.assertEqual(r['top_missing_labels'], 1)
        self.assertFalse(r['top_complete'])
        self.assertAlmostEqual(r['top_label_mean'], .2)

    def test_cost_rerun_changes_quantity_not_just_nav(self):
        q = {('d0','A'): {'close': 10}, ('d1','A'): {'open': 10,'close': 11}}
        limits = {('d1','A'): {'up_limit': 12,'down_limit': 8}}
        free, _ = simulate(['d1'], {'d0': {'A': 1}}, q, limits, set(), [], initial=10000, bps=0)
        paid, fills = simulate(['d1'], {'d0': {'A': 1}}, q, limits, set(), [], initial=10000, bps=30)
        self.assertEqual(free[0]['positions']['A'], 1000)
        self.assertEqual(paid[0]['positions']['A'], 900)
        self.assertEqual(fills[0]['fee'], 27)


def tests():
    import io
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticTests))
    write(OUT / 'tests.json', {'tests': result.testsRun, 'successful': result.wasSuccessful(),
                             'output': stream.getvalue(), 'code_sha256': digest(Path(__file__))})
    print(stream.getvalue(), flush=True)
    assert result.wasSuccessful()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['freeze', 'run', 'tests'])
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.out)
    assert OUT.resolve().is_relative_to((ROOT / 'reports/target_diagnostics_20260923').resolve())
    OUT.mkdir(exist_ok=True, parents=True)
    globals()[args.step]()
