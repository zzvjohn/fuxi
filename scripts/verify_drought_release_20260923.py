"""Verify new default routing against frozen, actual loaded-market diagnostics."""

import dataclasses
import json
import os
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd

from fuxi_drought_campaign_20260923 import ALCHEMY, CODE, OUT, digest, engine, switches, write_json
from run_contract import RunContract


def main():
    replay = json.loads((OUT / 'replay.json').read_text(encoding='utf-8'))
    paired = []
    by_id = {(r['candidate_id'], r['label']): r for r in replay['rows']}
    for candidate_id, label in by_id:
        if label != 'original_daily':
            continue
        daily = by_id[(candidate_id, label)]
        weekly = by_id.get((candidate_id, 'original_weekly'))
        if not weekly or daily['eval_status'] != 'valid' or weekly['eval_status'] != 'valid':
            continue
        assert daily['formula_sha256'] == weekly['formula_sha256']
        paired.append(candidate_id)
    assert len(paired) > 0, 'no counterfactual pair'
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    frozen_row = next(row for row in replay['rows'] if row.get('label') == 'broadcast_repaired'
                      and row['eval_status'] == 'valid' and not row.get('precheck_enforce')
                      and not row.get('value_guard'))
    chosen = dict(next(c for c in protocol['candidates'] if c['memory_id'] == frozen_row['candidate_id']))
    chosen['formula'] = frozen_row['formula']
    expected = pd.read_parquet(frozen_row['costs']['15']['returns_file'])['return']
    with patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith('FUXI_')}, clear=True):
        with RunContract(parameters={'step': 'release_actual_default_e2e', 'candidate_id': chosen['memory_id']},
                         run_id='drought_release_' + datetime.now().strftime('%H%M%S'), runs_dir=OUT / 'runs'):
            obj = engine()
            assert obj.weekly_channel == 'enforce'
            assert obj._effective_calmar_threshold() == .7
            actual = obj.validate_factor(factor_name=chosen['factor_name'], formula=chosen['formula'], natural_freq='weekly')
            assert actual.signal_freq == 'weekly'
            assert actual.weekly_channel_mode == 'enforce'
            assert actual.portfolio_returns is not None and len(actual.portfolio_returns) > 0
            pd.testing.assert_series_equal(actual.portfolio_returns, expected, check_names=False,
                                           check_dtype=False, check_freq=False, rtol=1e-7, atol=1e-8)
            for year in (2025, 2026):
                old = frozen_row['costs']['15'][f'metrics_{year}']
                np.testing.assert_allclose(getattr(actual, f'excess_{year}'), old['excess_return'], rtol=1e-7, atol=1e-8)
                np.testing.assert_allclose(getattr(actual, f'calmar_{year}'), old['calmar'], rtol=1e-7, atol=1e-8)
            assert actual.passed == frozen_row['costs']['15']['economic_pass']
            partial = expected.loc['2025'].iloc[-60:]
            with switches(FUXI_S5_RETURN_ALIGN='on'):
                aligned = obj.compute_metrics(partial, 2025)
            with switches(FUXI_S5_RETURN_ALIGN='off'):
                unmatched = obj.compute_metrics(partial, 2025)
            assert aligned['n_periods'] == 60
            assert abs(aligned['benchmark_return'] - unmatched['benchmark_return']) > 1e-10
            from weekly_panel import eval_weekly_formula, align_weekly_to_daily
            panel = obj._get_weekly_panel()
            ok, _, long, reason = eval_weekly_formula(chosen['formula'], panel)
            assert ok, reason
            with switches(FUXI_WEEKLY_MISSING_CONTRACT='enforce'):
                fresh = align_weekly_to_daily(long, panel, obj._price_df[['ts_code', 'trade_date']])
            with switches(FUXI_WEEKLY_MISSING_CONTRACT='legacy'):
                stale = align_weekly_to_daily(long, panel, obj._price_df[['ts_code', 'trade_date']])
            common = fresh.merge(stale, on=['ts_code', 'trade_date'], suffixes=('_new', '_old'))
            assert len(common) == len(fresh) > 0
            np.testing.assert_allclose(common.factor_value_new, common.factor_value_old)
            fields = dataclasses.asdict(actual)
            fields.pop('portfolio_returns', None)
            fields = {key: value for key, value in fields.items() if not isinstance(value, (pd.Series, pd.DataFrame))}
            write_json(OUT / 'release_e2e.json', {
                'candidate_id': chosen['memory_id'], 'candidate_selection': 'first broadcast-repaired representative, not return-selected',
                'counterfactual_pairs': paired, 'actual_paired_daily_returns': len(expected),
                'default_matches_frozen_enforce': True, 'result': fields,
                'partial_market_return_alignment_ab': {'aligned': aligned, 'unmatched': unmatched,
                                                        'purpose': 'deliberate 60-date diagnostic slice, not candidate promotion'},
                'missing_signal_ab': {'new_rows': len(fresh), 'legacy_rows': len(stale),
                                      'paired_values': len(common), 'stale_rows_removed': len(stale) - len(fresh)},
                'code_hashes': {name: digest(ALCHEMY / name) for name in CODE},
                'new_formally_eligible_candidates': 0,
            })
            print(json.dumps({'paired_returns': len(expected), 'weekly_mode': actual.weekly_channel_mode,
                              'signal_freq': actual.signal_freq, 'passed': actual.passed}, ensure_ascii=False))
            rank_rows = []
            for candidate_id in ('mem_0617', 'mem_0504'):
                source = next(c for c in protocol['candidates'] if c['memory_id'] == candidate_id)
                old = by_id[(candidate_id, 'original_daily')]
                formula = source['formula']
                normalized = obj._normalize_rank_axis(formula)
                assert normalized != formula
                print('START rank-axis counterfactual ' + candidate_id, flush=True)
                signal = obj.evaluate_factor_formula(formula, source['factor_name'])
                assert signal is not None and len(signal) > 0
                guard = obj._value_guards(signal)
                returns, m25, m26 = obj._backtest_and_metrics(signal)
                passed, reason = obj._verdict(m25, m26)
                returns_path = OUT / 'returns' / (candidate_id + '_rank_axis_enforce_bps15.parquet')
                returns.rename('return').to_frame().to_parquet(returns_path)
                row = {'candidate_id': candidate_id, 'old_formula': formula, 'evaluated_formula': normalized,
                       'original_daily_causal': False, 'value_guard': guard,
                       'metrics_2025': m25, 'metrics_2026': m26,
                       'old_metrics_2025': old['costs']['15']['metrics_2025'],
                       'old_metrics_2026': old['costs']['15']['metrics_2026'],
                       'economic_pass': bool(passed and obj._last_backtest_diag['coverage_ok'] and not guard),
                       'reason': reason, 'formal_eligibility': False, 'returns_file': str(returns_path),
                       'diag': obj._last_backtest_diag,
                       'code_hashes': {name: digest(ALCHEMY / name) for name in CODE}}
                rank_rows.append(row)
                write_json(OUT / 'daily_rank_axis_ab.json', {'rows': rank_rows})
                print(json.dumps({'candidate_id': candidate_id, 'excess_2025': float(m25['excess_return']),
                                  'excess_2026': float(m26['excess_return']), 'economic_pass': row['economic_pass']}), flush=True)


if __name__ == '__main__':
    main()
