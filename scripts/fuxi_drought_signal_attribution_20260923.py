"""Post-freeze measurement only; no new candidates, tuning, or promotion."""

import json
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from fuxi_drought_campaign_20260923 import OUT, engine, digest, write_json, switches
from run_contract import RunContract
from s5_joint_filter import _board_limit_vector, S5JointFilter


def holdings_marking_probe():
    dates = pd.bdate_range('2026-01-05', '2026-01-16')
    prices = pd.DataFrame([{'trade_date': day, 'ts_code': code, 'close': 100.0,
                           'close_p': 101.0 if day == dates[-1] and code == '000001.SZ' else 100.0,
                           'pre_close': 100.0, 'pre_close_p': 100.0, 'amount': 100000.0}
                          for day in dates for code in ('000001.SZ', '000002.SZ')])
    signals = prices[['trade_date', 'ts_code']].copy()
    signals['factor_value'] = np.where(signals.ts_code == '000001.SZ', 2.0, 1.0)
    missing = signals.loc[~((signals.trade_date == dates[-1]) & (signals.ts_code == '000001.SZ'))]
    obj = S5JointFilter(cache_dir=OUT / 'cache')
    obj.top_n = 1
    obj._universe = ['000001.SZ', '000002.SZ']
    obj._price_df = prices
    obj._get_limit_up_wide = lambda: None
    with switches(FUXI_S5_COST_BPS=0, FUXI_S5_ADJ_PRICE='on', FUXI_S5_TRADABLE_GATE='on'):
        complete_returns, _ = obj.backtest_factor(signals)
        missing_returns, _ = obj.backtest_factor(missing)
    assert dates[-1] in complete_returns.index and dates[-1] not in missing_returns.index
    assert np.isclose(complete_returns.loc[dates[-1]], .01)
    write_json(OUT / 'holdings_marking_probe.json', {
        'status': 'reproduced_unfixed', 'kind': 'synthetic_failure_probe_not_market_performance',
        'only_change': 'remove selected stock signal on the exit Friday, keep its real price',
        'complete_return_dates': [str(x.date()) for x in complete_returns.index],
        'missing_signal_return_dates': [str(x.date()) for x in missing_returns.index],
        'real_exit_day_return': float(complete_returns.loc[dates[-1]]),
        'problem': 'future signal availability changes held-position marking samples',
        'market_impact': 'not_quantified', 'promotion': False})


def main():
    target = OUT / 'signal_attribution.json'
    if target.exists():
        raise RuntimeError('immutable attribution already exists')
    config = json.loads((OUT / 'r4_protocol.json').read_text(encoding='utf-8'))
    model = json.loads((OUT / 'r4_model.json').read_text(encoding='utf-8'))
    snapshot = OUT / 'execution_scripts' / ('attribution_' + datetime.now().strftime('%H%M%S_%f') + '.py')
    shutil.copy2(Path(__file__), snapshot)
    with RunContract(parameters={'step': 'post_freeze_signal_attribution',
                                 'harness': {'path': str(snapshot), 'sha256': digest(snapshot)},
                                 'r4_protocol_sha256': digest(OUT / 'r4_protocol.json')},
                     run_id='drought_attribution_' + datetime.now().strftime('%H%M%S'),
                     runs_dir=OUT / 'runs'):
        holdings_marking_probe()
        obj = engine()
        panel = obj._get_weekly_panel()
        weeks = panel['weeks']
        events = pd.DatetimeIndex(panel['week_last_day'].reindex(weeks).values)
        columns = sorted(panel['universe'])
        signals = {}
        for name in config['sources']:
            path = OUT / 'signals' / (name + '_fresh_llm.parquet')
            assert digest(path) == model['source_sha256'][path.name]
            long = pd.read_parquet(path)
            wide = long[long.trade_date.isin(events)].pivot(
                index='trade_date', columns='ts_code', values='factor_value')
            wide = wide.reindex(index=events, columns=columns)
            wide.index = weeks
            signals[name] = wide
        common = np.isfinite(np.stack([x.to_numpy() for x in signals.values()])).all(axis=0)
        prices = obj._price_df[['ts_code', 'trade_date', 'close_p', 'pre_close_p']].copy()
        daily = prices.close_p / prices.pre_close_p.replace(0, np.nan) - 1
        daily = daily.where(daily.abs() <= _board_limit_vector(prices.ts_code) + .005)
        prices['week'] = prices.trade_date.dt.to_period('W-SUN').dt.end_time.dt.normalize()
        group = [prices.week, prices.ts_code]
        growth = (1 + daily).groupby(group).prod(min_count=1)
        counts = daily.groupby(group).count()
        expected = prices.groupby('week').trade_date.nunique()
        growth = growth.where(counts == growth.index.get_level_values('week').map(expected))
        fwd = (growth.unstack('ts_code') - 1).reindex(index=weeks, columns=columns).shift(-1)
        # Load the exact frozen benchmark path; no inferred or forward-filled returns.
        index = pd.read_csv(obj.index_path, dtype={'trade_date': str})
        index['trade_date'] = pd.to_datetime(index.trade_date, format='%Y%m%d')
        index = index.set_index('trade_date').pct_chg.div(100)
        index = index.reindex(sorted(prices.trade_date.unique()))
        assert index.notna().all()
        by_week = index.index.to_period('W-SUN').end_time.normalize()
        bench = (1 + index).groupby(by_week).prod().sub(1).reindex(weeks).shift(-1)
        next_event = pd.Series(events, index=weeks).shift(-1)
        rows = []
        folds = [('development', '2021-01-01', '2024-12-31'),
                 ('2025H1', '2025-01-01', '2025-06-30'),
                 ('2025H2', '2025-07-01', '2025-12-31'),
                 ('2026YTD', '2026-01-01', '2026-12-31')]
        for name, original in signals.items():
            signal = original.where(common)
            deciles = np.ceil(signal.rank(axis=1, pct=True) * 10)
            top = signal.rank(axis=1, ascending=False, method='first') <= 80
            top_observed = fwd.where(top).count(axis=1)
            top_return = fwd.where(top).mean(axis=1).where(top_observed >= 72)
            universe_return = fwd.where(signal.notna()).mean(axis=1)
            quantiles = pd.DataFrame({str(q): fwd.where(deciles == q).mean(axis=1)
                                      for q in range(1, 11)})
            for label, start, end in folds:
                valid = (next_event >= pd.Timestamp(start)) & (next_event <= pd.Timestamp(end))
                matched = pd.concat([top_return.rename('top80'), universe_return.rename('universe'),
                                     bench.rename('benchmark'), quantiles], axis=1).loc[valid].dropna()
                assert len(matched) > 0
                rows.append({'source': name, 'fold': label, 'weeks': len(matched),
                             'first_label_available': str(next_event.reindex(matched.index).min()),
                             'last_label_available': str(next_event.reindex(matched.index).max()),
                             'mean_weekly_top80': float(matched.top80.mean()),
                             'mean_weekly_universe': float(matched.universe.mean()),
                             'mean_weekly_benchmark': float(matched.benchmark.mean()),
                             'mean_top80_minus_universe': float((matched.top80 - matched.universe).mean()),
                             'mean_top80_minus_benchmark': float((matched.top80 - matched.benchmark).mean()),
                             'mean_q10_minus_q1': float((matched['10'] - matched['1']).mean()),
                             'mean_weekly_deciles': {str(q): float(matched[str(q)].mean()) for q in range(1, 11)},
                             'mean_top80_label_count': float(top_observed.reindex(matched.index).mean())})
        write_json(target, {'created_at': datetime.now().astimezone().isoformat(), 'rows': rows,
                            'source_sha256': model['source_sha256'],
                            'selection': 'fixed common signal mask; selection never uses future labels',
                            'fold_assignment': 'label availability date, not signal date',
                            'limitations': ['post-freeze explanatory diagnostic, not preregistered performance selection',
                                            'observed-label means; missing future outcomes are not imputed',
                                            'top80 ties use sorted stock code, not the S5 amount-based tie rule',
                                            'no costs, tradability, drift, or execution; not S5 portfolio returns',
                                            'last observed market week may be partial at data ceiling',
                                            'no sector/size neutralization; no independent OOS'],
                            'formal_eligibility': False})
        print(json.dumps({'attribution_rows': len(rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
