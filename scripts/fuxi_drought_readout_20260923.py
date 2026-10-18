"""Read frozen diagnostic outputs without promoting candidates or changing history."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fuxi_drought_campaign_20260923 import OUT, ROOT, write_json


FOLDS = [('2025H1', '2025-01-01', '2025-06-30'),
         ('2025H2', '2025-07-01', '2025-12-31'),
         ('2026YTD', '2026-01-01', '2026-12-31')]


def read_returns(row, cost=15):
    return pd.read_parquet(row['costs'][str(cost)]['returns_file'])['return'].dropna()


def paired(a, b):
    frame = pd.concat([a.rename('treatment'), b.rename('baseline')], axis=1).dropna()
    assert len(frame) > 0, 'no actual paired observations'
    delta = (frame.treatment - frame.baseline).to_numpy(dtype=float)
    rng = np.random.default_rng(20260923)
    indices = (rng.integers(0, len(delta), size=(500, (len(delta) + 4) // 5, 1))
               + np.arange(5)) % len(delta)
    means = delta[indices.reshape(500, -1)[:, :len(delta)]].mean(axis=1)
    return {'paired_days': len(frame), 'mean_daily_delta': float(delta.mean()),
            'ci95_block5': np.quantile(means, [.025, .975]).tolist(),
            'total_return_delta': float((1 + frame.treatment).prod() - (1 + frame.baseline).prod())}


def main():
    index = pd.read_csv(ROOT / 'data' / 'raw' / 'index_000985_daily.csv', dtype={'trade_date': str})
    index['trade_date'] = pd.to_datetime(index.trade_date, format='%Y%m%d')
    benchmark = index.set_index('trade_date').pct_chg / 100
    groups = {}
    for name, filename in [('historical', 'replay.json'), ('llm', 'llm_evaluation.json'), ('r4', 'r4_evaluation.json')]:
        path = OUT / filename
        if not path.exists():
            raise RuntimeError('required batch unfinished: ' + filename)
        groups[name] = json.loads(path.read_text(encoding='utf-8')).get('rows', [])
    assert {name: len(rows) for name, rows in groups.items()} == {'historical': 16, 'llm': 6, 'r4': 6}
    details = []
    for group, rows in groups.items():
        for row in rows:
            item = {'group': group, 'candidate_id': row['candidate_id'], 'arm': row.get('label'),
                    'eval_status': row['eval_status'], 'formal_eligibility': False}
            if group == 'historical' and row['candidate_id'] in ('mem_0617', 'mem_0504') and row.get('label') == 'original_daily':
                item['causality_warning'] = 'legacy whole-time Series.rank uses future observations; error-location evidence only'
            if row['eval_status'] != 'valid':
                item['reason'] = row.get('reason', row.get('issues'))
                details.append(item)
                continue
            returns = {cost: read_returns(row, cost) for cost in (0, 15, 30)}
            item['signal_guard'] = row.get('value_guard')
            item['precheck'] = row.get('precheck_enforce')
            item['economic_pass_15'] = row['costs']['15']['economic_pass']
            item['calmar_2025_15'] = row['costs']['15']['metrics_2025']['calmar']
            item['calmar_2026_15'] = row['costs']['15']['metrics_2026']['calmar']
            item['avg_turnover_notional'] = row['costs']['15']['diag']['avg_turnover']
            item['coverage_rate'] = row['costs']['15']['diag']['coverage_rate']
            item['actual_start'] = str(returns[15].index.min().date())
            item['actual_end'] = str(returns[15].index.max().date())
            item['annual_excess'] = {str(cost): {
                str(year): row['costs'][str(cost)][f'metrics_{year}']['excess_return']
                for year in (2025, 2026)} for cost in (0, 15, 30)}
            fee_frame = pd.concat(returns, axis=1).dropna()
            assert len(fee_frame) > 0
            fee = fee_frame[0] - fee_frame[15]
            assert fee.min() >= -1e-7, 'cost path increased daily returns'
            linear_error = float(np.max(np.abs(fee_frame[30] - (fee_frame[0] - 2 * fee))))
            assert linear_error < 1e-6
            item['three_actual_cost_runs_linearity_max_error'] = linear_error
            item['folds'] = {}
            for label, start, end in FOLDS:
                part = returns[15].loc[start:end]
                matched = benchmark.reindex(part.index)
                assert len(part) > 0 and matched.notna().all()
                item['folds'][label] = {
                    'days': len(part), 'start': str(part.index.min().date()), 'end': str(part.index.max().date()),
                    'gross_excess': float((1 + returns[0].reindex(part.index)).prod() - (1 + matched).prod()),
                    'net_excess': float((1 + part).prod() - (1 + matched).prod()),
                    'cost_return_drag': float((1 + returns[0].reindex(part.index)).prod() - (1 + part).prod()),
                    'fee_sum': float(fee.reindex(part.index).sum()),
                }
            details.append(item)
    freq_pairs = []
    historical = {(r['candidate_id'], r['label']): r for r in groups['historical']}
    for key, daily in historical.items():
        weekly = historical.get((key[0], 'original_weekly'))
        if key[1] != 'original_daily' or not weekly or daily['eval_status'] != 'valid' or weekly['eval_status'] != 'valid':
            continue
        assert daily['formula_sha256'] == weekly['formula_sha256']
        freq_pairs.append({'candidate_id': key[0], **paired(read_returns(weekly), read_returns(daily))})
    assert len(freq_pairs) > 0
    combo_pairs = []
    combos = {r['label']: r for r in groups['r4'] if r['eval_status'] == 'valid'}
    for label, row in combos.items():
        baseline = combos['baseline_common'] if label in ('equal_rank', 'ridge_alpha_100') else combos.get('equal_rank')
        if label == 'baseline_common' or baseline is None:
            continue
        a, b = read_returns(row), read_returns(baseline)
        folds = {name: paired(a.loc[start:end], b.loc[start:end]) for name, start, end in FOLDS}
        combo_pairs.append({'arm': label, 'baseline': baseline['label'], 'folds': folds,
                            'improves_all_fold_total_returns': all(v['total_return_delta'] > 0 for v in folds.values())})
    result = {'details': details, 'frequency_pairs': freq_pairs, 'r4_pairs': combo_pairs,
              'daily_rank_axis_ab': json.loads((OUT / 'daily_rank_axis_ab.json').read_text(encoding='utf-8'))['rows'],
              'history_is_research_exposed': True, 'new_formally_eligible_candidates': 0,
              'interpretation': 'economic diagnostics are not S1-S5/JQ qualification or independent OOS validation'}
    write_json(OUT / 'readout.json', result)
    lines = ['# 冻结实验读数', '', '15bps 为正式成本；经济指标达标不等于取得正式候选资格。', '',
             '| 候选 | 实验臂 | 2025 净超额 | 2026 净超额 | 2026 风险比 | 经济门槛 |',
             '|---|---|---:|---:|---:|---|']
    for row in details:
        if row['eval_status'] != 'valid':
            lines.append(f"| {row['candidate_id']} | {row['arm']} | 未求值 | 未求值 | 未求值 | {row['reason']} |")
        else:
            metrics = row['annual_excess']['15']
            lines.append(f"| {row['candidate_id']} | {row['arm']} | {metrics['2025']:.2%} | {metrics['2026']:.2%} | {row['calmar_2026_15']:.3f} | {row['economic_pass_15']} |")
    lines += ['', '全部收益窗以实际 return parquet 的日期为准；2026 年不外推至年末。',
              'mem_0617/mem_0504 的旧日频臂含全时序 rank 前视，仅用于定位错误；纠正结果另见 daily_rank_axis_ab.json。',
              'R4 使用研究基线，不是现有生产组合。区间估计为 5 日循环块 bootstrap，500 次，仅作探索性配对诊断。']
    # Generated artifact, not a manual source edit.
    (OUT / 'readout.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\r\n')
    print(json.dumps({'historical_arms': len(groups['historical']), 'llm': len(groups['llm']),
                      'r4': len(groups['r4']), 'valid_arms': sum(x['eval_status'] == 'valid' for x in details),
                      'economic_pass_15': sum(x.get('economic_pass_15', False) for x in details),
                      'frequency_pairs': len(freq_pairs)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
