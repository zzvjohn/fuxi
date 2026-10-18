"""Neutralised residual attribution for the four frozen signals: size and PIT industry.

This batch answers one question: when the fixed Top80 already lags its own equal-weight
pool in the five-session label, how much of that gap is explained by a size or industry
exposure, and how much survives neutralisation?

Method. The pool itself is the regression sample, so an OLS with an intercept puts the
mean-pool residual at exactly zero. That makes the following an identity, not an
approximation:

    mean(Top label) - mean(pool label)
        = beta . (mean(Top basis) - mean(pool basis))     [explained by exposure]
        + mean(Top residual)                              [unexplained]

Two non-parametric cross-checks run alongside: a size-quintile stratified pair weighted by
pool shares (so the size mix is held fixed), and an explicitly rebuilt size-neutral Top80
whose quotas reproduce the pool's size distribution.

Nothing here is an economic verdict. Labels are uncharged, unexecuted, research-exposed
adjusted-open comparisons; the industry map is PIT by membership date but is not a
certified vendor vintage; and no significance claim is made beyond reported dispersion.
"""
import argparse
import csv
import math
import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from fuxi_execution_lab_20260923 import digest, load_panel, read, write  # noqa: E402
from run_contract import RunContract  # noqa: E402
from fuxi_target_diagnostics_20260923 import label_dates  # noqa: E402

OUT = ROOT / 'reports/target_neutral_20260923'
PRIOR = ROOT / 'reports/target_diagnostics_20260923'
SIGNAL_DIR = ROOT / 'reports/drought_20260923/signals'

TOP_N = 80
MIN_INDUSTRY_N = 20
STRATA = 5
OTHER = '__other__'
PREDICTION_END = '2026-09-09'


# ---------------------------------------------------------------- shared primitives

def fold_size(path, wanted):
    """Frozen size rule: every value finite, positive and exactly equal, else quarantine.

    A conflicted key is dropped whole. Taking a first or last row would silently invent a
    vintage, so the rule is all-or-nothing and quarantined keys are counted, not patched.
    """
    parts = []
    for chunk in pd.read_csv(path, usecols=['trade_date', 'ts_code', 'total_mv'],
                             dtype={'trade_date': str}, chunksize=300000):
        chunk['trade_date'] = pd.to_datetime(chunk.trade_date, format='mixed',
                                             errors='raise').dt.strftime('%Y-%m-%d')
        parts.append(chunk[chunk.trade_date.isin(wanted)])
    frame = pd.concat(parts, ignore_index=True)
    frame['valid'] = np.isfinite(frame.total_mv) & (frame.total_mv > 0)
    grouped = frame.groupby(['trade_date', 'ts_code']).agg(
        rows=('total_mv', 'size'), valid=('valid', 'sum'),
        low=('total_mv', 'min'), high=('total_mv', 'max'))
    safe = grouped[(grouped.rows == grouped.valid) & (grouped.low == grouped.high)]
    clean = safe[['low']].rename(columns={'low': 'total_mv'}).reset_index()
    stats = {'rows_read': len(frame), 'keys': len(grouped), 'safe_keys': len(clean),
             'quarantined_keys': len(grouped) - len(clean)}
    return clean, stats


def public_calendar():
    cal = pd.read_csv(ROOT / 'data/raw/trade_calendar.csv', dtype=str)
    cal = cal[(cal.exchange == 'SSE') & (cal.is_open == '1')]
    days = sorted(pd.to_datetime(cal.cal_date, format='mixed').dt.strftime('%Y-%m-%d').unique())
    return [d for d in days if d <= PREDICTION_END]


def load_industry():
    """SW L1 membership table with a strict conflict audit before any row is kept."""
    table = pd.read_csv(ROOT / 'data/sw_classification_L1.csv', dtype=str)
    table['in'] = pd.to_datetime(table.in_date, format='mixed', errors='coerce')
    table['out'] = pd.to_datetime(table.out_date, format='mixed', errors='coerce')
    table = table.dropna(subset=['in', 'industry_name'])
    assert not table.industry_name.eq(OTHER).any(), 'industry name collides with the pool sentinel'
    keys = ['con_code', 'in', 'out']
    # dropna=False matters: an open-ended membership has out_date = NaT, and pandas would
    # otherwise drop every current holding from the audit and from deduplication.
    variants = table.groupby(keys, dropna=False).industry_name.nunique()
    conflicting = int((variants > 1).sum())
    assert conflicting == 0, 'same membership window carries two industry names'
    audit = {'rows': len(table), 'codes': table.con_code.nunique(),
             'industries': table.industry_name.nunique(),
             'open_ended_rows': int(table['out'].isna().sum()),
             'duplicate_window_rows': int(table.duplicated(keys).sum()),
             'conflicting_windows': conflicting}
    table = table.drop_duplicates(keys, keep='first')
    return table, audit


def pit_lookup(table):
    """stamp -> Series(code -> industry). Asserts a single industry per code and date."""
    def lookup(stamp):
        ts = pd.Timestamp(stamp)
        rows = table[(table['in'] <= ts) & (table['out'].isna() | (table['out'] > ts))]
        counts = rows.groupby('con_code').industry_name.nunique()
        assert len(counts) == 0 or int(counts.max()) == 1, 'two industries in force at once'
        return rows.drop_duplicates('con_code', keep='first').set_index('con_code').industry_name
    return lookup


def ols(y, design):
    """Least squares. Returns beta, residual, numeric rank and the singular-value ratio."""
    beta, _, rank, sv = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    ratio = float(sv[0] / sv[-1]) if len(sv) and sv[-1] > 0 else float('inf')
    return beta, residual, int(rank), ratio


def industry_basis(codes, industry_of, min_n):
    """Pool-level industry levels, pooling thin cells into one reference group.

    Returns the remapped pool levels, the reference level, the dummy levels in order, and a
    report. Codes with no industry are reported, never guessed, and never receive a dummy.
    """
    labels = np.array([industry_of.get(c) for c in codes], dtype=object)
    known = np.array([isinstance(v, str) and bool(v) for v in labels])
    counts = pd.Series(labels[known]).value_counts()
    kept = sorted(k for k in counts.index if counts[k] >= min_n)
    remap = np.where(known & np.isin(labels, kept), labels, OTHER)
    levels = sorted(set(remap))
    reference = OTHER if OTHER in levels else levels[0]
    columns = [lv for lv in levels if lv != reference]
    report = {'reference': reference, 'kept_industries': len(kept), 'dummy_columns': len(columns),
              'pooled_into_reference': int(np.sum(remap == OTHER)) - int((~known).sum()),
              'missing_industry': int((~known).sum()),
              'missing_treated_as_reference': True}
    return remap, reference, columns, report


def dummy_matrix(levels, columns, reference):
    if not columns:
        return np.zeros((len(levels), 0))
    mapped = np.where(np.isin(levels, columns), levels, reference)
    return np.column_stack([(mapped == lv).astype(float) for lv in columns])


def quintile_strata(values, q=STRATA):
    """Stratum id in [0, n) that stays well defined under heavy ties."""
    edges = np.unique(np.quantile(values, np.linspace(0, 1, q + 1)))
    if len(edges) < 3:
        low, high = float(values.min()), float(values.max())
        if low == high:
            return np.zeros(len(values), dtype=int), 1
        edges = np.linspace(low, high, q + 1)
    ids = np.clip(np.searchsorted(edges, values, side='right') - 1, 0, len(edges) - 2)
    return ids, len(edges) - 1


def stratified_neutral_difference(labels, x, is_top, q=STRATA):
    """Pool-share-weighted mean of within-stratum gaps, so the size mix cancels out."""
    stratum, planned = quintile_strata(x, q)
    total = len(x)
    weighted = 0.0
    strata_used = 0
    top_used = 0
    for level in np.unique(stratum):
        member = stratum == level
        top = member & is_top
        if not top.any():
            continue
        share = member.sum() / total
        weighted += share * (labels[top].mean() - labels[member].mean())
        strata_used += 1
        top_used += int(top.sum())
    return weighted, strata_used, planned, top_used


def size_neutral_rebuild(frame, x, quota=TOP_N, q=STRATA):
    """Rebuild a Top-N whose size composition reproduces the pool's, largest-remainder quotas."""
    stratum, _ = quintile_strata(x, q)
    counts = pd.Series(stratum).value_counts().sort_index()
    if counts.empty or counts.sum() == 0:
        return frame.iloc[:0], {}
    raw = counts / counts.sum() * quota
    base = np.floor(raw).astype(int)
    shortfall = int(quota - base.sum())
    for level in (raw - base).sort_values(ascending=False).index[:shortfall]:
        base[level] += 1
    picked = []
    for level, need in base.items():
        if need <= 0:
            continue
        block = frame[stratum == level]
        picked.extend(block.sort_values(['factor_value', 'amount', 'ts_code'],
                                        ascending=[False, False, True]).head(int(need)).index)
    return frame.loc[picked], {int(k): int(v) for k, v in base.items()}


# ------------------------------------------------------------------------ pipeline

def rebuild_sections():
    """Recompute pool, Top80, labels, lagged size and PIT industry for every section."""
    protocol = read(OUT / 'protocol.json')
    calendars = public_calendar()
    prior_date = {d: calendars[i - 1] for i, d in enumerate(calendars) if i > 0}

    price, _ = load_panel('data/raw/daily_prices.csv',
                          ['ts_code', 'trade_date', 'open', 'close', 'amount'])
    adj, _ = load_panel('data/raw/adj_factor_daily.csv', ['ts_code', 'trade_date', 'adj_factor'])
    merged = price.merge(adj, on=['trade_date', 'ts_code'], validate='one_to_one', how='left')
    merged['adj_open'] = merged.open * merged.adj_factor
    merged.loc[(merged.open <= 0) | (merged.adj_factor <= 0), 'adj_open'] = np.nan
    byday = {d: g.set_index('ts_code') for d, g in merged.groupby('trade_date')}
    print('Loaded prices and adjustments:', len(merged), flush=True)

    stamps = sorted(protocol['signal_dates'])
    size_clean, size_stats = fold_size(ROOT / 'data/raw/daily_basic.csv',
                                       {prior_date[d] for d in stamps if d in prior_date})
    sizes = {d: g.set_index('ts_code').total_mv for d, g in size_clean.groupby('trade_date')}
    print('Size keys:', size_stats, flush=True)

    industry, industry_audit = load_industry()
    lookup = pit_lookup(industry)
    print('Industry audit:', industry_audit, flush=True)

    prior_rows = {(r['signal'], r['date']): r
                  for r in read(PRIOR / 'retry_v2/cross_sections.json')['rows']
                  if r['status'] == 'valid_prediction_label_only'}
    size_prior = {(r['signal'], r['date']): r
                  for r in read(PRIOR / 'size_supplement/cross_sections.json')['rows']}

    rows, micro = [], []
    for name in protocol['signals']:
        signal = pd.read_parquet(SIGNAL_DIR / (name + '.parquet'))
        signal['trade_date'] = pd.to_datetime(signal.trade_date).dt.strftime('%Y-%m-%d')
        signal = signal[signal.trade_date.isin(stamps)]
        assert not signal.duplicated(['trade_date', 'ts_code']).any()
        for stamp, group in signal.groupby('trade_date'):
            window = label_dates(calendars, stamp)
            if window is None:
                continue
            entry, exit_day = window
            frame = group.merge(byday[stamp][['close', 'amount']], left_on='ts_code',
                                right_index=True, how='left', validate='one_to_one')
            frame['label'] = (frame.ts_code.map(byday[exit_day].adj_open)
                              / frame.ts_code.map(byday[entry].adj_open) - 1)
            eligible = frame[np.isfinite(frame.factor_value) & (frame.close > 0)].copy()
            selected = eligible.sort_values(['factor_value', 'amount', 'ts_code'],
                                           ascending=[False, False, True]).head(TOP_N)
            finite = eligible[np.isfinite(eligible.label)]
            top = selected[np.isfinite(selected.label)]

            # Parity with the frozen batch before anything is attributed.
            prior = prior_rows[(name, stamp)]
            assert len(eligible) == prior['eligible_n'], (name, stamp)
            assert np.isclose(finite.label.mean(), prior['pool_label_mean'], atol=1e-12), (name, stamp)
            assert np.isclose(top.label.mean(), prior['top_label_mean'], atol=1e-12), (name, stamp)

            lag = prior_date[stamp]
            lag_size = sizes.get(lag, pd.Series(dtype=float))
            industry_of = lookup(stamp)

            # Parity with the frozen size batch: its basis is the whole eligible pool and the
            # whole selected Top80, before label availability is observed. That is a different
            # sample from the labelled regression pool used below, so both are reported.
            pool_basis = eligible.ts_code.map(lag_size).dropna()
            top_basis = selected.ts_code.map(lag_size).dropna()
            raw_size_difference = float(np.log(top_basis).mean() - np.log(pool_basis).mean()) \
                if len(pool_basis) and len(top_basis) else None
            prior_size = size_prior.get((name, stamp))
            assert prior_size is not None and prior_size.get('log_size_difference') is not None
            assert raw_size_difference is not None
            assert np.isclose(raw_size_difference, prior_size['log_size_difference'], atol=1e-9), \
                (name, stamp, raw_size_difference, prior_size['log_size_difference'])

            frame['log_size'] = np.log(frame.ts_code.map(lag_size))
            frame['industry'] = frame.ts_code.map(industry_of)
            eligible = frame[np.isfinite(frame.factor_value) & (frame.close > 0)].copy()
            finite = eligible[np.isfinite(eligible.label)]
            selected = eligible.sort_values(['factor_value', 'amount', 'ts_code'],
                                            ascending=[False, False, True]).head(TOP_N)
            top = selected[np.isfinite(selected.label)]

            row = {'signal': name, 'date': stamp, 'entry': entry, 'exit': exit_day, 'lag_date': lag,
                   'eligible_n': len(eligible), 'label_n': len(finite), 'top_label_n': len(top),
                   'top_complete': len(top) == TOP_N,
                   'raw_top_minus_pool': float(top.label.mean() - finite.label.mean()),
                   'raw_log_size_difference': raw_size_difference,
                   'industry_missing_pool': int(finite.industry.isna().sum()),
                   'industry_missing_top': int(top.industry.isna().sum())}

            usable = finite[np.isfinite(finite.log_size) & finite.industry.notna()]
            top_usable = top[np.isfinite(top.log_size) & top.industry.notna()]
            dropped_no_size = int((~np.isfinite(finite.log_size)).sum())
            dropped_no_industry = int((np.isfinite(finite.log_size) & finite.industry.isna()).sum())
            assert dropped_no_size + dropped_no_industry + len(usable) == len(finite)
            row.update({'dropped_no_size': dropped_no_size,
                        'dropped_no_industry': dropped_no_industry,
                        'reg_pool_n': len(usable), 'reg_top_n': len(top_usable)})
            if len(usable) < MIN_INDUSTRY_N or len(top_usable) < 2:
                row['status'] = 'data_insufficient'
                rows.append(row)
                micro.append({'signal': name, 'date': stamp, 'status': 'data_insufficient',
                              'usable_pool_n': len(usable), 'usable_top_n': len(top_usable)})
                continue

            labels = usable.label.to_numpy()
            x = usable.log_size.to_numpy()
            x_top = top_usable.log_size.to_numpy()
            label_top = top_usable.label.to_numpy()
            mask_top = np.isin(usable.ts_code, top_usable.ts_code)
            assert int(mask_top.sum()) == len(top_usable), 'Top member missing from the regression pool'
            total = float(label_top.mean() - labels.mean())

            one = np.ones(len(x))
            beta_size, resid_size, rank_s, ratio_s = ols(labels, np.column_stack([one, x]))
            explained_size = float(beta_size[1] * (x_top.mean() - x.mean()))
            # Subtract the pool mean too: the intercept makes it numerically near zero, not
            # exactly zero, and the identity must close to machine precision.
            residual_size = float(resid_size[mask_top].mean() - resid_size.mean())

            ind_map = industry_of.to_dict()
            pool_levels, reference, columns, ind_report = industry_basis(
                usable.ts_code.tolist(), ind_map, MIN_INDUSTRY_N)
            top_levels = np.array([ind_map.get(c) for c in top_usable.ts_code], dtype=object)
            design = np.column_stack([one, x, dummy_matrix(pool_levels, columns, reference)])
            top_design = np.column_stack([np.ones(len(x_top)), x_top,
                                          dummy_matrix(top_levels, columns, reference)])
            beta_both, resid_both, rank_b, ratio_b = ols(labels, design)
            explained_both = float(beta_both @ (top_design.mean(axis=0) - design.mean(axis=0)))
            residual_both = float(resid_both[mask_top].mean() - resid_both.mean())

            stratified, strata_used, strata_planned, strata_top = stratified_neutral_difference(
                labels, x, mask_top)
            rebuilt, quotas = size_neutral_rebuild(usable.reset_index(drop=True), x)
            rebuilt = rebuilt[np.isfinite(rebuilt.label)]
            row.update({
                'status': 'valid_descriptive',
                'reg_top_minus_pool': total,
                'size_explained': explained_size, 'size_residual': residual_size,
                'size_identity_error': float(total - explained_size - residual_size),
                'size_rank': rank_s, 'size_condition': ratio_s, 'size_beta': float(beta_size[1]),
                'industry_columns': ind_report['dummy_columns'],
                'industry_kept': ind_report['kept_industries'],
                'industry_pooled_into_reference': ind_report['pooled_into_reference'],
                'both_explained': explained_both, 'both_residual': residual_both,
                'both_identity_error': float(total - explained_both - residual_both),
                'both_rank': rank_b, 'both_condition': ratio_b,
                'stratified_neutral_diff': float(stratified), 'strata_used': strata_used,
                'strata_planned': strata_planned, 'strata_top_used': strata_top,
                'rebuilt_neutral_top_minus_pool': float(rebuilt.label.mean() - labels.mean())
                if len(rebuilt) else None,
                'rebuilt_count': len(rebuilt), 'rebuilt_quotas': str(quotas),
                'top_share': float(mask_top.mean()),
            })
            assert abs(row['size_identity_error']) < 1e-9, (name, stamp, row['size_identity_error'])
            assert abs(row['both_identity_error']) < 1e-9, (name, stamp, row['both_identity_error'])
            rows.append(row)
            micro.append({'signal': name, 'date': stamp, 'status': 'valid_descriptive',
                          'usable_pool_n': len(usable), 'usable_top_n': len(top_usable),
                          'raw_log_size_difference': row['raw_log_size_difference'],
                          'top_share': float(mask_top.mean()),
                          'size_identity_error': row['size_identity_error'],
                          'both_identity_error': row['both_identity_error']})
    return rows, micro, {'size': size_stats, 'industry': industry_audit}


def summarise(rows):
    def values(name, key):
        out = [r[key] for r in rows if r['signal'] == name and r['status'] == 'valid_descriptive'
               and r.get(key) is not None]
        return np.array(out, dtype=float)

    def mean(name, key):
        arr = values(name, key)
        return float(arr.mean()) if len(arr) else None

    def tstat(name, key):
        arr = values(name, key)
        if len(arr) < 2 or arr.std(ddof=1) == 0:
            return None
        return float(arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr))))

    out = []
    for name in sorted({r['signal'] for r in rows}):
        valid = [r for r in rows if r['signal'] == name and r['status'] == 'valid_descriptive']
        planned = [r for r in rows if r['signal'] == name]
        if not valid:
            out.append({'signal': name, 'planned_sections': len(planned),
                        'valid_sections': 0, 'status': 'data_insufficient'})
            continue
        out.append({
            'signal': name, 'planned_sections': len(planned), 'valid_sections': len(valid),
            'complete_top_sections': sum(1 for r in valid if r['top_complete']),
            'raw_top_minus_pool_mean': mean(name, 'raw_top_minus_pool'),
            'raw_log_size_difference_mean': mean(name, 'raw_log_size_difference'),
            'reg_top_minus_pool_mean': mean(name, 'reg_top_minus_pool'),
            'size_explained_mean': mean(name, 'size_explained'),
            'size_residual_mean': mean(name, 'size_residual'),
            'size_residual_t': tstat(name, 'size_residual'),
            'size_identity_error_max': float(np.abs(values(name, 'size_identity_error')).max()),
            'both_explained_mean': mean(name, 'both_explained'),
            'both_residual_mean': mean(name, 'both_residual'),
            'both_residual_t': tstat(name, 'both_residual'),
            'both_identity_error_max': float(np.abs(values(name, 'both_identity_error')).max()),
            'stratified_neutral_diff_mean': mean(name, 'stratified_neutral_diff'),
            'stratified_neutral_diff_t': tstat(name, 'stratified_neutral_diff'),
            'rebuilt_neutral_top_minus_pool_mean': mean(name, 'rebuilt_neutral_top_minus_pool'),
            'rebuilt_neutral_t': tstat(name, 'rebuilt_neutral_top_minus_pool'),
            'reg_pool_observations': sum(r['reg_pool_n'] for r in valid),
            'reg_top_observations': sum(r['reg_top_n'] for r in valid),
            'dropped_no_size': sum(r['dropped_no_size'] for r in valid),
            'dropped_no_industry': sum(r['dropped_no_industry'] for r in valid),
            'industry_missing_pool': sum(r['industry_missing_pool'] for r in valid),
            'industry_missing_top': sum(r['industry_missing_top'] for r in valid),
            'max_condition': float(max(r['both_condition'] for r in valid)),
            'rank_deficient_sections': sum(1 for r in valid
                                           if r['both_rank'] < 2 + r['industry_columns']),
            'strata_top_dropped': sum(r['reg_top_n'] - r['strata_top_used'] for r in valid),
        })
    return out


def write_table(path, content):
    """Write a list of dicts as CSV, quoting values that carry commas or quotes.

    Manual comma joins corrupt the table as soon as any value contains a comma, which
    rebuilt_quotas always does. Columns are ordered with the identifying keys first so a
    reader can align rows to sections without trusting position inside the value set.
    """
    keys = sorted({k for r in content for k in r})
    lead = [k for k in ('signal', 'date', 'entry', 'exit', 'status') if k in keys]
    order = lead + [k for k in keys if k not in lead]
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        sink = csv.writer(handle, lineterminator='\n')
        sink.writerow(order)
        for r in content:
            sink.writerow(['' if r.get(k) is None else r[k] for k in order])
    return order


class NeutralTests(unittest.TestCase):
    def test_ols_recovers_slope_and_annihilates_residual(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=400)
        labels = 3.0 + 0.5 * x
        beta, resid, _, _ = ols(labels, np.column_stack([np.ones(len(x)), x]))
        self.assertAlmostEqual(float(beta[1]), 0.5, places=10)
        self.assertLess(float(np.abs(resid).max()), 1e-10)

    def test_identity_decomposition_is_exact(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=600)
        labels = 1.0 + 0.4 * x + rng.normal(scale=0.1, size=600)
        beta, resid, _, _ = ols(labels, np.column_stack([np.ones(len(x)), x]))
        top = np.zeros(len(x), dtype=bool)
        top[:60] = True
        total = labels[top].mean() - labels.mean()
        explained = beta[1] * (x[top].mean() - x.mean())
        unexplained = resid[top].mean()
        self.assertLess(abs(total - explained - unexplained), 1e-12)

    def test_equal_size_leaves_no_explained_part(self):
        rng = np.random.default_rng(2)
        x = np.zeros(300)
        labels = rng.normal(size=300)
        beta, _, _, _ = ols(labels, np.column_stack([np.ones(len(x)), x]))
        top = np.zeros(300, dtype=bool)
        top[:30] = True
        self.assertAlmostEqual(beta[1] * (x[top].mean() - x.mean()), 0.0, places=12)

    def test_thin_industry_is_pooled_and_counted(self):
        codes = ['A%d' % i for i in range(30)]
        industry = {c: ('big1' if i < 15 else 'big2' if i < 28 else 'thin')
                    for i, c in enumerate(codes)}
        levels, reference, columns, report = industry_basis(codes, industry, min_n=5)
        self.assertEqual(report['kept_industries'], 2)
        self.assertEqual(sorted(columns), ['big1', 'big2'])
        self.assertEqual(report['pooled_into_reference'], 2)
        self.assertEqual(reference, OTHER)
        self.assertEqual(report['missing_industry'], 0)

    def test_missing_industry_is_never_filled(self):
        levels, reference, columns, report = industry_basis(['A', 'B', 'C'], {'A': 'bank'}, 1)
        self.assertEqual(report['missing_industry'], 2)
        self.assertEqual(columns, ['bank'])
        # Unknown industries are never guessed; they fall into the reference group.
        self.assertEqual(report['reference'], OTHER)
        self.assertEqual([v for v in levels], ['bank', OTHER, OTHER])

    def test_strata_hold_the_size_mix_fixed(self):
        x = np.array([1.0] * 100 + [10.0] * 100)
        labels = np.array([0.0] * 100 + [0.05] * 10 + [0.0] * 90)
        top = np.zeros(200, dtype=bool)
        top[100:110] = True
        value, used, planned, top_used = stratified_neutral_difference(labels, x, top)
        self.assertEqual((used, top_used), (1, 10))
        self.assertGreater(planned, 1)
        self.assertAlmostEqual(value, 0.5 * (0.05 - 0.005), places=10)

    def test_rebuilt_quotas_reproduce_the_pool_mix(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=500)
        frame = pd.DataFrame({'factor_value': rng.normal(size=500), 'amount': 1.0,
                              'ts_code': ['C%03d' % i for i in range(500)],
                              'label': rng.normal(size=500), 'log_size': x})
        picked, quotas = size_neutral_rebuild(frame, x, quota=80)
        self.assertEqual(len(picked), 80)
        self.assertEqual(sum(quotas.values()), 80)
        stratum, _ = quintile_strata(x, STRATA)
        pool_share = pd.Series(stratum).value_counts(normalize=True).sort_index()
        got = pd.Series(stratum[picked.index]).value_counts(normalize=True).sort_index()
        self.assertLess(float((pool_share - got).abs().max()), 0.02)

    def test_open_ended_windows_survive_the_conflict_audit(self):
        """pandas drops NaN group keys by default; a still-open membership must stay audited."""
        frame = pd.DataFrame({'con_code': ['A', 'A'], 'in': pd.to_datetime(['2020-01-01'] * 2),
                              'out': pd.to_datetime([None, None]),
                              'industry_name': ['bank', 'media']})
        silently_dropped = frame.groupby(['con_code', 'in', 'out']).industry_name.nunique()
        audited = frame.groupby(['con_code', 'in', 'out'], dropna=False).industry_name.nunique()
        self.assertEqual(len(silently_dropped), 0)
        self.assertEqual(int((audited > 1).sum()), 1)

    def test_table_round_trip_survives_commas(self):
        """A comma inside a value must not shift columns, and output must stay LF."""
        import tempfile
        content = [{'signal': 'a', 'date': 'd', 'rebuilt_quotas': '{0: 15, 1: 16}'},
                   {'signal': 'b', 'date': 'e', 'rebuilt_quotas': '{0: 16, 1: 15}'}]
        path = Path(tempfile.mkdtemp()) / 't.csv'
        order = write_table(path, content)
        raw = path.read_bytes()
        self.assertEqual(raw.count(b'\r\n'), 0)
        with open(path, encoding='utf-8', newline='') as handle:
            back = list(csv.DictReader(handle))
        self.assertEqual(len(back), 2)
        self.assertEqual([r['signal'] for r in back], ['a', 'b'])
        self.assertEqual([r['rebuilt_quotas'] for r in back],
                         ['{0: 15, 1: 16}', '{0: 16, 1: 15}'])
        # DictReader must have produced exactly the declared header, nothing merged in.
        self.assertEqual(sorted(back[0]), sorted(order))
        # A pathological quote inside a value must also survive.
        write_table(path, [{'signal': 'x', 'note': 'a,"b"'}])
        with open(path, encoding='utf-8', newline='') as handle:
            again = list(csv.DictReader(handle))
        self.assertEqual(again[0]['note'], 'a,"b"')


def freeze():
    assert not (OUT / 'protocol.json').exists()
    sections = read(PRIOR / 'retry_v2/cross_sections.json')
    source = read(PRIOR / 'retry_v2/protocol.json')
    stamps = sorted(sections['common_signal_dates'])
    paths = ['data/raw/daily_prices.csv', 'data/raw/adj_factor_daily.csv',
             'data/raw/daily_basic.csv', 'data/raw/trade_calendar.csv',
             'data/sw_classification_L1.csv',
             'reports/target_diagnostics_20260923/retry_v2/cross_sections.json',
             'reports/target_diagnostics_20260923/size_supplement/cross_sections.json',
             'scripts/fuxi_target_neutral_attribution_20260923.py',
             'scripts/fuxi_target_diagnostics_20260923.py']
    paths += ['reports/drought_20260923/signals/' + n + '.parquet' for n in source['signals']]
    hashes = {p: digest(ROOT / p) for p in paths}
    for n, h in source['signals'].items():
        assert hashes['reports/drought_20260923/signals/' + n + '.parquet'] == h
    write(OUT / 'protocol.json', {
        'created_at': datetime.now().astimezone().isoformat(),
        'signals': source['signals'], 'hashes': hashes, 'signal_dates': stamps,
        'top_n': TOP_N, 'min_industry_n': MIN_INDUSTRY_N, 'strata': STRATA,
        'label': source['label'], 'pool': source['pool'], 'selection': source['selection'],
        'size_rule': 'same frozen identical-value collapse; conflicted keys quarantined whole',
        'industry_rule': 'SW L1 membership by in_date <= signal date < out_date; industries '
                         'with fewer than min_industry_n pool members pooled into one '
                         'reference group; missing industry never filled',
        'decomposition': 'pool-internal OLS with intercept; mean-pool residual is zero by '
                         'construction, so total = explained + unexplained exactly',
        'crosschecks': 'size-quintile stratified pair and size-neutral rebuilt Top80',
        'scope': 'descriptive attribution only, no search, no production write, no eligibility',
        'limitations': ['labels are uncharged adjusted-open comparisons, not cash NAV',
                        'industry map is PIT by membership date but not a certified vendor vintage',
                        'no significance claim beyond reported dispersion',
                        'research-exposed sample; no R5 unknown observations'],
        'full_stage2_accepted': False, 'formal_eligibility': False})


def run():
    import json
    protocol = read(OUT / 'protocol.json')
    for p, h in protocol['hashes'].items():
        assert digest(ROOT / p) == h, p
    assert not (OUT / 'summary.json').exists()
    with RunContract(parameters={'protocol_sha256': digest(OUT / 'protocol.json'),
                                 'code_data_hashes': protocol['hashes']},
                     run_id='neutral_' + datetime.now().strftime('%H%M%S_%f'),
                     runs_dir=OUT / 'runs', lock_path=OUT / 'neutral.lock'):
        rows, micro, sources = rebuild_sections()
        for p, h in protocol['hashes'].items():
            assert digest(ROOT / p) == h, p
        summaries = summarise(rows)
        valid = [r for r in rows if r['status'] == 'valid_descriptive']
        def extreme(key):
            return max((abs(r[key]) for r in valid), default=None)
        quality = {
            'sources': sources,
            'sections_planned': {n: sum(1 for r in rows if r['signal'] == n)
                                 for n in protocol['signals']},
            'sections_valid': {n: sum(1 for r in rows if r['signal'] == n
                                      and r['status'] == 'valid_descriptive')
                               for n in protocol['signals']},
            'reg_pool_observations': sum(r['reg_pool_n'] for r in valid),
            'reg_top_observations': sum(r['reg_top_n'] for r in valid),
            'dropped_no_size': sum(r['dropped_no_size'] for r in valid),
            'dropped_no_industry': sum(r['dropped_no_industry'] for r in valid),
            'industry_missing_pool': sum(r['industry_missing_pool'] for r in valid),
            'industry_missing_top': sum(r['industry_missing_top'] for r in valid),
            'size_identity_error_max': extreme('size_identity_error'),
            'both_identity_error_max': extreme('both_identity_error'),
            'rank_deficient_sections': sum(1 for r in valid
                                           if r['both_rank'] < 2 + r['industry_columns']),
            'max_condition': max((r['both_condition'] for r in valid), default=None),
            'strata_top_dropped': sum(r['reg_top_n'] - r['strata_top_used'] for r in valid),
        }
        write(OUT / 'summary.json', {
            'protocol_sha256': digest(OUT / 'protocol.json'), 'rows': summaries, 'quality': quality,
            'reading': 'size_explained is the part of the Top-vs-pool label gap carried by the '
                       'size basis; size_residual survives neutralisation. both_* adds PIT '
                       'industry dummies. stratified_* and rebuilt_* hold the size mix fixed '
                       'without assuming a linear form.',
            'no_significance_claim': True, 'industry_PIT_certified': False,
            'full_stage2_accepted': False, 'formal_eligibility': False})
        for name, content in (('per_section.csv', rows), ('micro_check.csv', micro)):
            write_table(OUT / name, content)
        print(json.dumps(summaries, ensure_ascii=False, indent=1), flush=True)
        print('Quality:', json.dumps(quality, ensure_ascii=False), flush=True)


def tests():
    import io
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(NeutralTests))
    write(OUT / 'tests.json', {'tests': result.testsRun, 'successful': result.wasSuccessful(),
                              'output': stream.getvalue(), 'code_sha256': digest(Path(__file__))})
    print(stream.getvalue(), flush=True)
    assert result.wasSuccessful()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['freeze', 'run', 'tests'])
    OUT.mkdir(exist_ok=True)
    globals()[parser.parse_args().step]()
