"""Attribute frozen incomplete marks; no price filling or performance reassessment."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from s5_joint_filter import S5JointFilter, _board_limit_vector
from fuxi_drought_campaign_20260923 import digest, write_json

OUT = ROOT / 'reports/holdings_20260923'


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    paths = {name: info['sha256'] for name, info in protocol['data'].items()}
    paths['data/raw/suspend_d.csv'] = digest(ROOT / 'data/raw/suspend_d.csv')
    for name, expected in paths.items():
        assert digest(ROOT / name) == expected
    parts = []
    for chunk in pd.read_csv(ROOT / 'data/raw/daily_prices.csv',
                             usecols=['ts_code', 'trade_date', 'close', 'pre_close', 'amount'],
                             dtype={'trade_date': str}, chunksize=400000):
        chunk['trade_date'] = pd.to_datetime(chunk.trade_date, format='mixed')
        chunk = chunk[chunk.trade_date.between('2024-12-01', '2026-09-09')].copy()
        for col in ['close', 'pre_close', 'amount']:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('float32')
        parts.append(chunk)
    price = pd.concat(parts, ignore_index=True)
    assert not price.duplicated(['ts_code', 'trade_date']).any()
    calendar = pd.Series(sorted(price.trade_date.unique()))
    rebalances = pd.DatetimeIndex(calendar.groupby(calendar.dt.to_period('W-SUN')).max())
    by_date = {d: frame.set_index('ts_code') for d, frame in price.groupby('trade_date')}
    obj = S5JointFilter(cache_dir=OUT / 'cache')
    obj._price_df = price
    obj._universe = list(price.ts_code.unique())
    limits = obj._get_limit_up_wide()
    suspend = pd.read_csv(ROOT / 'data/raw/suspend_d.csv', dtype={'trade_date': str})
    suspend['trade_date'] = pd.to_datetime(suspend.trade_date, format='mixed')
    full_halt = suspend[(suspend.suspend_type == 'S') & suspend.suspend_timing.isna()]
    halt_keys = set(zip(full_halt.trade_date, full_halt.ts_code))
    summary = {'scope': '2025/2026 incomplete-return dates only; no substitution into a return series',
               'source_sha256': paths, 'suspension_table_max_date': str(suspend.trade_date.max().date()),
               'rows': []}
    for name in protocol['signals']:
        result = json.loads((OUT / 'rows' / (name + '_enforce_15.json')).read_text(encoding='utf-8'))
        returns = pd.read_parquet(result['returns_file'])['return']
        bad_dates = returns[(returns.index.year >= 2025) & returns.isna()].index
        assert len(bad_dates) > 0
        entries = {day: rebalances[rebalances.searchsorted(day, side='left') - 1] for day in bad_dates}
        signal_path = ROOT / 'reports/drought_20260923/signals' / (name + '.parquet')
        assert digest(signal_path) == protocol['signal_hashes'][name]
        signal = pd.read_parquet(signal_path, filters=[('trade_date', 'in', list(set(entries.values())))])
        selections, unknown = {}, []
        for day, entry in entries.items():
            if entry not in selections:
                ranked = signal[signal.trade_date == entry].merge(by_date[entry].reset_index(), on='ts_code')
                ranked = ranked[np.isfinite(ranked.factor_value)].copy()
                up = ranked.ts_code.map(limits.loc[entry]) if entry in limits.index else pd.Series(np.nan, index=ranked.index)
                exact = up.notna() & (ranked.close.to_numpy() >= up.to_numpy() - 1e-6)
                daily = ranked.close / ranked.pre_close.replace(0, np.nan) - 1
                fallback = up.isna() & (daily >= _board_limit_vector(ranked.ts_code) - .005)
                ranked = ranked[~(exact | fallback)].sort_values(['factor_value', 'amount', 'ts_code'],
                                                               ascending=[False, False, True], kind='mergesort')
                assert len(ranked) >= 80
                selections[entry] = ranked.head(80).ts_code.tolist()
            codes = selections[entry]
            quotes = by_date[day].reindex(codes)
            daily = quotes.close / quotes.pre_close.replace(0, np.nan) - 1
            guarded = daily.abs() > _board_limit_vector(pd.Series(codes, index=quotes.index)) + .005
            invalid = ~np.isfinite(daily) | guarded
            assert invalid.any(), 'independent reconstruction must reproduce the frozen unknown day'
            for code in quotes.index[invalid]:
                present = code in by_date[day].index
                halted = (day, code) in halt_keys
                reason = ('outlier_guard' if bool(guarded.loc[code]) else
                          'confirmed_full_day_suspension' if not present and halted else
                          'unclassified_missing_quote' if not present else 'invalid_price_reference')
                unknown.append({'date': str(day.date()), 'entry_date': str(entry.date()), 'ts_code': code,
                                'reason': reason, 'quote_present': present, 'explicit_full_day_halt': halted})
        by_year = Counter(row['date'][:4] for row in unknown)
        for year in ('2025', '2026'):
            assert by_year[year] == result['diag']['missing_held_price_cells_by_year'].get(year, 0)
        summary['rows'].append({'signal': name, 'incomplete_days': len(bad_dates),
                                'unknown_cells_by_year': dict(by_year),
                                'reasons': dict(Counter(row['reason'] for row in unknown)), 'cells': unknown,
                                'independent_count_matches_frozen_ab': True})
        print(json.dumps({k: v for k, v in summary['rows'][-1].items() if k != 'cells'}, ensure_ascii=False))
    assert all(digest(ROOT / name) == value for name, value in paths.items())
    write_json(OUT / 'marking_attribution.json', summary)


if __name__ == '__main__':
    main()
