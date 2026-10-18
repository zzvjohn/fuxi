"""Real information diversity: three non-price-volume families measured against the
frozen pool, with single-channel ablation and per-section pairing.

Why this batch exists. The drought review's third step asks for genuinely different
economic information rather than more formulas over the same OHLCV. This script takes
three families that are already in the local store, audits their disclosure timing, and
then asks of each candidate factor the only question that matters:

    1. does it carry information the pool does not already have (event contrast)?
    2. does that survive removing size, industry and the four frozen signals' own ranks?
    3. is the direction stable across pre-registered sub-periods?
    4. does selecting on it pick a different portfolio than we already hold?

Discipline that is not negotiable here:

* Sparse families are NEVER filled with zero. An absent observation is not a zero
  observation; filling would manufacture a cross-sectional value out of no information.
  Instead the covered set is compared against the uncovered set, and coverage is
  reported per section.
* A family's value is only used when it was knowable before the entry open. For the two
  exchange-published daily statistics the data date is the signal date (published that
  evening, well before the next session's open). For announcement-type data the
  announcement must precede the signal date's own previous session, which is the
  conservative direction: it discards observations rather than inventing them.
* Nothing here is an economic verdict. Labels are uncharged adjusted-open comparisons
  on a research-exposed window.
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

OUT = ROOT / 'reports/external_info_20260924'
NEUTRAL = ROOT / 'reports/target_neutral_20260923'
PRIOR = ROOT / 'reports/target_diagnostics_20260923'
SIGNAL_DIR = ROOT / 'reports/drought_20260923/signals'

TOP_N = 80
MIN_COVERED = 10          # below this a cross-section cannot be paired at all
MIN_SIDE = 5              # each side of an event contrast needs at least this many names
SEGMENTS = 3              # pre-registered sub-periods, fixed at freeze time
FORECAST_WINDOW_SESSIONS = 60   # a disclosure stays the factor value for this many sessions
PREDICTION_END = '2026-09-09'
OTHER = '__other__'
MIN_INDUSTRY_N = 20


# --------------------------------------------------------------- small statistics

def spearman(a, b):
    """Rank correlation, computed from ranks so no scipy dependency is needed."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return float('nan')
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def mean_ci(values):
    """Mean and a two-sided 95% t interval from the sample's own dispersion.

    Assumes nothing beyond finite variance. Returns None for the interval when fewer
    than two observations exist, rather than inventing a width.
    """
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return None, None, None, 0
    mean = float(vals.mean())
    if len(vals) < 2:
        return mean, None, None, len(vals)
    sd = float(vals.std(ddof=1))
    se = sd / math.sqrt(len(vals))
    half = 1.96 * se
    return mean, mean - half, mean + half, len(vals)


def t_stat(values):
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(vals) < 2:
        return None
    sd = float(vals.std(ddof=1))
    if sd == 0:
        return None
    return float(vals.mean() / (sd / math.sqrt(len(vals))))


def jaccard(left, right):
    left, right = set(left), set(right)
    if not left and not right:
        return None
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


# ------------------------------------------------------------------- data loading

def load_industry():
    table = pd.read_csv(ROOT / 'data/sw_classification_L1.csv', dtype=str)
    table['in'] = pd.to_datetime(table.in_date, format='mixed', errors='coerce')
    table['out'] = pd.to_datetime(table.out_date, format='mixed', errors='coerce')
    table = table.dropna(subset=['in', 'industry_name'])
    assert not table.industry_name.eq(OTHER).any()
    keys = ['con_code', 'in', 'out']
    # dropna=False: pandas would otherwise drop every open-ended membership from the audit.
    variants = table.groupby(keys, dropna=False).industry_name.nunique()
    assert int((variants > 1).sum()) == 0, 'same membership window carries two names'
    table = table.drop_duplicates(keys, keep='first')
    return table


def pit_industry(table):
    def lookup(stamp):
        ts = pd.Timestamp(stamp)
        rows = table[(table['in'] <= ts) & (table['out'].isna() | (table['out'] > ts))]
        counts = rows.groupby('con_code').industry_name.nunique()
        assert len(counts) == 0 or int(counts.max()) == 1, 'two industries in force at once'
        return rows.drop_duplicates('con_code', keep='first').set_index('con_code').industry_name
    return lookup


def fold_size(path, wanted):
    """Frozen size rule: identical, finite, positive or the key is quarantined whole."""
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
    return clean, {'rows_read': len(frame), 'keys': len(grouped), 'safe_keys': len(clean),
                   'quarantined_keys': len(grouped) - len(clean)}


def _daykey(series):
    return series.astype(str).str.strip().str.replace('-', '', regex=False).str.slice(0, 8)


# ------------------------------------------------------------ family construction

def build_top_list(stamps):
    """Dragon-tiger list: an event family, one row per listed name per day.

    Values exist only for names that were listed. Absence is preserved as absence.
    """
    df = pd.read_csv(ROOT / 'data/raw/top_list.csv', dtype=str)
    df['day'] = _daykey(df.trade_date)
    df = df[df.day.isin(stamps)].copy()
    for col in ('net_amount', 'amount', 'l_buy', 'l_sell', 'l_amount'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # same name twice on one day (two reasons) is not two events: sum the money and keep
    # the widest seat book, then assert the collapse is unambiguous per reason set.
    dup = df.groupby(['day', 'ts_code']).size()
    events = df.groupby(['day', 'ts_code']).agg(
        net=('net_amount', 'sum'), amount=('amount', 'first'),
        l_buy=('l_buy', 'sum'), l_sell=('l_sell', 'sum'), l_amount=('l_amount', 'sum'))
    events = events.reset_index()
    events['lh_net_rate'] = np.where(events.amount > 0, events.net / events.amount, np.nan)
    denom = events.l_amount
    events['lh_seat_asym'] = np.where(denom > 0, (events.l_buy - events.l_sell) / denom, np.nan)
    audit = {'rows': len(df), 'collapsed_pairs': len(df) - len(events),
             'multi_row_days': int((dup > 1).sum()), 'events': len(events)}
    out = {d: g.set_index('ts_code') for d, g in events.groupby('day')}
    return out, audit


def build_margin(stamps, lag_of):
    """Margin financing: a dense family (most large names report daily) with a window gap.

    The five-session change is computed on each code's own observed series, so a missing
    session stays missing rather than being interpolated across.
    """
    df = pd.read_csv(ROOT / 'data/raw/margin_detail.csv',
                     dtype=str, usecols=['trade_date', 'ts_code', 'rzye', 'rzmre'])
    df['day'] = _daykey(df.trade_date)
    for col in ('rzye', 'rzmre'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    dup = int(df.duplicated(['day', 'ts_code']).sum())
    df = df.drop_duplicates(['day', 'ts_code'], keep='first')
    df = df.sort_values(['ts_code', 'day'])
    df['rzye_lag5'] = df.groupby('ts_code').rzye.shift(5)
    df['margin_rz_chg5'] = np.where(df.rzye_lag5 > 0, df.rzye / df.rzye_lag5 - 1.0, np.nan)
    df['margin_rz_buyrate'] = np.where(df.rzye > 0, df.rzmre / df.rzye, np.nan)
    window = df[df.day.isin(stamps)]
    audit = {'rows': len(df), 'duplicate_rows_dropped': dup,
             'window_rows': len(window), 'days_in_window': window.day.nunique()}
    out = {d: g.set_index('ts_code')[['margin_rz_chg5', 'margin_rz_buyrate']]
           for d, g in window.groupby('day')}
    return out, audit


def build_forecast(stamps, safe_of, calendars, window_sessions=FORECAST_WINDOW_SESSIONS):
    """Earnings pre-announcements: announcement-dated, so only what precedes the signal.

    `safe_of[stamp]` is the latest day whose announcements were certainly public before
    the signal date. Taking the newest announcement at or before that day is the
    point-in-time view, not the latest revision.

    A disclosure window is applied on the other side too. Without one, a pre-announcement
    published months earlier would remain the factor value indefinitely and the column
    would measure stale guidance memory rather than a disclosure event. The window is
    counted in trading sessions and is frozen in the protocol.
    """
    df = pd.read_csv(ROOT / 'data/raw/forecast.csv', dtype=str)
    df['ann'] = _daykey(df.ann_date)
    df['end'] = _daykey(df.end_date)
    for col in ('p_change_min', 'p_change_max'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df.ann.notna() & (df.ann != '')]
    df = df.sort_values(['ts_code', 'end', 'ann'])
    dup = int(df.duplicated(['ts_code', 'end', 'ann']).sum())
    df = df.drop_duplicates(['ts_code', 'end', 'ann'], keep='last')
    rows = []
    for stamp in stamps:
        cutoff = safe_of[stamp]
        index = calendars.index(stamp) if stamp in calendars else None
        if index is None:
            continue
        open_from = calendars[max(0, index - window_sessions)].replace('-', '')
        sub = df[(df.ann <= cutoff) & (df.ann >= open_from)]
        if sub.empty:
            continue
        # newest announcement per (code, period) that was public inside the window
        latest = sub.groupby(['ts_code', 'end']).tail(1)
        latest = latest.groupby('ts_code').tail(1)
        latest = latest.copy()
        latest['fcst_yoy_mid'] = (latest.p_change_min + latest.p_change_max) / 2.0
        scale = latest[['p_change_min', 'p_change_max']].abs().max(axis=1).clip(lower=1.0)
        latest['fcst_range_ratio'] = (latest.p_change_max - latest.p_change_min) / scale
        # the lookup key must match what rebuild_sections uses: the eight-digit day form
        latest['day'] = stamp.replace('-', '')
        rows.append(latest[['day', 'ts_code', 'fcst_yoy_mid', 'fcst_range_ratio']])
    stacked = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=['day', 'ts_code', 'fcst_yoy_mid', 'fcst_range_ratio'])
    audit = {'rows': len(df), 'duplicate_rows_dropped': dup, 'window_rows': len(stacked),
             'days_with_any': int(stacked.day.nunique()) if len(stacked) else 0}
    out = {d: g.set_index('ts_code')[['fcst_yoy_mid', 'fcst_range_ratio']]
           for d, g in stacked.groupby('day')}
    return out, audit


FAMILY_COLUMNS = {
    'top_list': ['lh_net_rate', 'lh_seat_asym'],
    'margin_detail': ['margin_rz_chg5', 'margin_rz_buyrate'],
    'forecast': ['fcst_yoy_mid', 'fcst_range_ratio'],
}


# -------------------------------------------------------------------- attribution

def industry_basis(codes, industry_of):
    """Industry dummy levels for one cross-section.

    `industry_of` must be keyed by ts_code. Passing a plain dict keyed by position would
    make every lookup miss and silently reduce the design to size-only, so the caller
    passes a code-indexed Series and the miss rate is checked by the caller.
    """
    labels = np.array([industry_of.get(c) for c in codes], dtype=object)
    known = np.array([isinstance(v, str) and bool(v) for v in labels])
    counts = pd.Series(labels[known]).value_counts()
    kept = sorted(k for k in counts.index if counts[k] >= MIN_INDUSTRY_N)
    remap = np.where(known & np.isin(labels, kept), labels, OTHER)
    levels = sorted(set(remap))
    reference = OTHER if OTHER in levels else levels[0]
    columns = [lv for lv in levels if lv != reference]
    return remap, reference, columns


def design_matrix(frame, basis, reference, columns):
    blocks = [np.ones(len(frame)), frame.log_size.to_numpy()]
    for level in columns:
        blocks.append((basis == level).astype(float))
    return np.column_stack(blocks)


def residualise(y, design):
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta, beta, int(rank)


def public_calendar():
    cal = pd.read_csv(ROOT / 'data/raw/trade_calendar.csv', dtype=str)
    cal = cal[(cal.exchange == 'SSE') & (cal.is_open == '1')]
    days = sorted(pd.to_datetime(cal.cal_date, format='mixed').dt.strftime('%Y-%m-%d').unique())
    return [d for d in days if d <= PREDICTION_END]


def evaluate_factor(frame, labels, column, resid_base, resid_full, existing):
    """Everything one candidate factor is allowed to claim, on one cross-section.

    Two residual vectors arrive precomputed: `resid_base` removes size and industry,
    `resid_full` additionally removes the four frozen signals' own ranks. The contrast
    under `resid_full` is therefore "information the existing system did not already
    carry", which is the only version of the claim worth making.
    """
    values = frame[column].to_numpy(dtype=float)
    covered = np.isfinite(values)
    row = {'pool_n': int(len(frame)), 'covered_n': int(covered.sum()),
           'coverage': float(covered.mean())}
    if row['covered_n'] == 0:
        row['status'] = 'no_coverage'
        return row
    if row['covered_n'] < MIN_SIDE:
        row['status'] = 'covered_side_too_thin'
        return row
    if not covered.all() and (len(frame) - row['covered_n']) < MIN_SIDE:
        row['status'] = 'uncovered_side_too_thin'
        return row

    dense = bool(covered.all())
    row['dense_family'] = dense
    row['contrast_raw'] = None if dense else float(labels[covered].mean() - labels[~covered].mean())
    row['contrast_resid'] = None if dense else float(resid_base[covered].mean()
                                                    - resid_base[~covered].mean())
    row['contrast_resid_existing'] = None if dense else float(resid_full[covered].mean()
                                                              - resid_full[~covered].mean())
    row['ic_raw'] = spearman(values[covered], labels[covered])
    row['ic_resid'] = spearman(values[covered], resid_base[covered])
    row['ic_resid_existing'] = spearman(values[covered], resid_full[covered])
    row['status'] = 'valid_descriptive'

    # does selecting on it pick a different book than the ranked existing signals?
    if row['covered_n'] >= TOP_N:
        idx = np.where(covered)[0]
        order = idx[np.argsort(-values[idx], kind='stable')][:TOP_N]
        row['overlap_existing'] = jaccard(order, np.argsort(-existing, kind='stable')[:TOP_N])
    else:
        row['overlap_existing'] = None
    return row


def rebuild_sections():
    protocol = read(OUT / 'protocol.json')
    calendars = public_calendar()
    prior_date = {d: calendars[i - 1] for i, d in enumerate(calendars) if i > 0}
    stamps = sorted(protocol['signal_dates'])

    price, _ = load_panel('data/raw/daily_prices.csv',
                          ['ts_code', 'trade_date', 'open', 'close', 'amount'])
    adj, _ = load_panel('data/raw/adj_factor_daily.csv', ['ts_code', 'trade_date', 'adj_factor'])
    merged = price.merge(adj, on=['trade_date', 'ts_code'], validate='one_to_one', how='left')
    merged['adj_open'] = merged.open * merged.adj_factor
    merged.loc[(merged.open <= 0) | (merged.adj_factor <= 0), 'adj_open'] = np.nan
    byday = {d: g.set_index('ts_code') for d, g in merged.groupby('trade_date')}
    print('prices', len(merged), flush=True)

    size_clean, size_stats = fold_size(ROOT / 'data/raw/daily_basic.csv',
                                       {prior_date[d] for d in stamps if d in prior_date})
    sizes = {d: g.set_index('ts_code').total_mv for d, g in size_clean.groupby('trade_date')}
    industry_all = load_industry()
    lookup = pit_industry(industry_all)
    print('size', size_stats, flush=True)

    stamp_keys = [s.replace('-', '') for s in stamps]
    families, audits = {}, {}
    families['top_list'], audits['top_list'] = build_top_list(set(stamp_keys))
    families['margin_detail'], audits['margin_detail'] = build_margin(set(stamp_keys), prior_date)
    safe_of = {s: prior_date[s].replace('-', '') for s in stamps if s in prior_date}
    families['forecast'], audits['forecast'] = build_forecast(stamps, safe_of, calendars)
    for name, audit in audits.items():
        print('family', name, audit, flush=True)

    prior_rows = {(r['signal'], r['date']): r
                  for r in read(PRIOR / 'retry_v2/cross_sections.json')['rows']
                  if r['status'] == 'valid_prediction_label_only'}
    size_prior = {(r['signal'], r['date']): r
                  for r in read(PRIOR / 'size_supplement/cross_sections.json')['rows']}

    rows, drops = [], []
    geometry = []
    for name in protocol['signals']:
        signal = pd.read_parquet(SIGNAL_DIR / (name + '.parquet'))
        signal['trade_date'] = pd.to_datetime(signal.trade_date).dt.strftime('%Y-%m-%d')
        signal = signal[signal.trade_date.isin(stamps)]
        assert not signal.duplicated(['trade_date', 'ts_code']).any()
        peer = {}
        for other in protocol['signals']:
            table = pd.read_parquet(SIGNAL_DIR / (other + '.parquet'))
            table['trade_date'] = pd.to_datetime(table.trade_date).dt.strftime('%Y-%m-%d')
            peer[other] = table[table.trade_date.isin(stamps)]

        for stamp, group in signal.groupby('trade_date'):
            window = label_dates(calendars, stamp)
            if window is None:
                continue
            entry, exit_day = window
            frame = group.merge(byday[stamp][['close', 'amount']], left_on='ts_code',
                                right_index=True, how='left', validate='one_to_one')
            frame['label'] = (frame.ts_code.map(byday[exit_day].adj_open)
                              / frame.ts_code.map(byday[entry].adj_open) - 1)
            frame['log_size'] = np.log(frame.ts_code.map(sizes[prior_date[stamp]]))
            frame['industry'] = frame.ts_code.map(lookup(stamp))

            # parity with both frozen batches before anything is attributed
            prior = prior_rows[(name, stamp)]
            eligible = frame[np.isfinite(frame.factor_value) & (frame.close > 0)].copy()
            selected = eligible.sort_values(['factor_value', 'amount', 'ts_code'],
                                            ascending=[False, False, True]).head(TOP_N)
            finite = eligible[np.isfinite(eligible.label)]
            top = selected[np.isfinite(selected.label)]
            assert len(eligible) == prior['eligible_n'], (name, stamp)
            assert np.isclose(finite.label.mean(), prior['pool_label_mean'], atol=1e-12), (name, stamp)
            assert np.isclose(top.label.mean(), prior['top_label_mean'], atol=1e-12), (name, stamp)
            pool_basis = eligible.ts_code.map(sizes[prior_date[stamp]]).dropna()
            top_basis = selected.ts_code.map(sizes[prior_date[stamp]]).dropna()
            raw_diff = float(np.log(top_basis).mean() - np.log(pool_basis).mean())
            assert np.isclose(raw_diff, size_prior[(name, stamp)]['log_size_difference'],
                              atol=1e-9), (name, stamp)

            # the pool used for attribution is the labelled one, same as the neutral batch
            pool = finite[np.isfinite(finite.log_size)].copy()
            if len(pool) < 50:
                drops.append({'signal': name, 'date': stamp, 'reason': 'pool_too_small',
                              'pool_n': len(pool)})
                continue
            basis, reference, columns = industry_basis(
                pool.ts_code.tolist(), pool.set_index('ts_code').industry)
            known_share = float(pool.industry.notna().mean())
            # A code-keyed map that silently misses would leave a size-only design and
            # every residual would then be misattributed. Fail loudly instead.
            assert not (known_share > 0.5 and len(columns) == 0), \
                'industry map missed silently (%s, %s)' % (name, stamp)

            # existing information: the equal-weight rank of the four frozen signals.
            # A signal that did not cover a name leaves NaN, which nanmean ignores, so a
            # name is never credited with a rank it was never given.
            ranks = []
            for other in protocol['signals']:
                table = peer[other]
                table = table[table.trade_date == stamp].set_index('ts_code').factor_value
                aligned = pool.ts_code.map(table)
                ranks.append(pd.Series(np.asarray(aligned, dtype=float)).rank().to_numpy())
            stacked_ranks = np.column_stack(ranks)
            with np.errstate(invalid='ignore'):
                existing = np.nanmean(stacked_ranks, axis=1)
            pool = pool.assign(existing=existing)
            label_np = pool.label.to_numpy(dtype=float)

            base_design = design_matrix(pool, basis, reference, columns)
            full_design = np.column_stack([base_design, pool.existing.to_numpy(dtype=float)])
            resid_base, _, rank_base = residualise(label_np, base_design)
            resid_full, _, rank_full = residualise(label_np, full_design)
            geometry.append({'signal': name, 'date': stamp,
                             'base_columns': base_design.shape[1], 'base_rank': rank_base,
                             'full_columns': full_design.shape[1], 'full_rank': rank_full,
                             'existing_missing': int(np.sum(~np.isfinite(existing)))})
            assert rank_base == base_design.shape[1], (name, stamp, rank_base)
            assert rank_full == full_design.shape[1], (name, stamp, rank_full)

            for family, cols in FAMILY_COLUMNS.items():
                table = families[family].get(stamp.replace('-', ''))
                for column in cols:
                    frame_v = pool.copy()
                    if table is None or column not in table.columns:
                        frame_v[column] = np.nan
                    else:
                        frame_v[column] = pool.ts_code.map(table[column])
                    row = evaluate_factor(frame_v, label_np, column,
                                          resid_base, resid_full,
                                          pool.existing.to_numpy(dtype=float))
                    row.update({'factor': column, 'signal': name, 'date': stamp, 'entry': entry,
                                'family': family, 'lag_date': prior_date[stamp],
                                'industry_columns': len(columns), 'zero_filled': 0})
                    rows.append(row)
            print('  %s %s pool=%d industries=%d' % (name, stamp, len(pool), len(columns)),
                  flush=True)
    return rows, drops, geometry, {'size': size_stats, 'families': audits,
                                   'industry_rows': len(industry_all),
                                   'industry_codes': int(industry_all.con_code.nunique())}


# ---------------------------------------------------------------------- summarising

def segment_bounds(protocol):
    """Pre-registered sub-period boundaries, fixed at freeze time and never recomputed."""
    return protocol['segments']


def segment_of(protocol, stamp):
    for index, (lo, hi) in enumerate(segment_bounds(protocol)):
        if lo <= stamp <= hi:
            return index
    return None


def summarise(rows, protocol):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['factor'], []).append(row)
    out = {}
    for factor, mine in sorted(grouped.items()):
        valid = [r for r in mine if r.get('status') == 'valid_descriptive']
        entry = {'sections_total': len(mine), 'sections_valid': len(valid),
                 'sections_planned': len(protocol['signal_dates']) * len(protocol['signals']),
                 'coverage_mean': float(np.mean([r['coverage'] for r in valid])) if valid else None,
                 'coverage_min': float(np.min([r['coverage'] for r in valid])) if valid else None,
                 'coverage_max': float(np.max([r['coverage'] for r in valid])) if valid else None}
        for key in ('contrast_raw', 'contrast_resid', 'contrast_resid_existing',
                    'ic_raw', 'ic_resid', 'ic_resid_existing', 'overlap_existing'):
            vals = [r.get(key) for r in valid]
            mean, lo, hi, n = mean_ci(vals)
            entry[key] = {'mean': mean, 'ci_low': lo, 'ci_high': hi, 'n': n,
                          't': t_stat(vals), 'positive_sections': int(sum(1 for v in vals
                                                                          if v is not None and np.isfinite(v) and v > 0))}
        segs = []
        for index in range(len(segment_bounds(protocol))):
            subset = [r.get('contrast_resid_existing') for r in valid
                      if segment_of(protocol, r['date']) == index]
            subset = [v for v in subset if v is not None and np.isfinite(v)]
            segs.append({'segment': index, 'n': len(subset),
                         'mean': float(np.mean(subset)) if subset else None,
                         'positive': int(sum(1 for v in subset if v > 0))})
        entry['segments'] = segs
        signs = [s['mean'] for s in segs if s['mean'] is not None]
        entry['segment_sign_consistent'] = bool(signs) and all(s > 0 for s in signs) or \
            (bool(signs) and all(s < 0 for s in signs))
        out[factor] = entry
    return out


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    lead = [k for k in ('factor', 'family', 'signal', 'date', 'entry', 'status') if k in keys]
    order = lead + [k for k in keys if k not in lead]
    # csv.writer, never manual joins: some values are dict-shaped strings containing commas.
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        sink = csv.writer(handle, lineterminator='\n')
        sink.writerow(order)
        for r in rows:
            sink.writerow(['' if r.get(k) is None else r[k] for k in order])


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'protocol already frozen; remove it explicitly'
    neutral = read(NEUTRAL / 'protocol.json')
    stamps = sorted(neutral['signal_dates'])
    assert len(stamps) == 87, len(stamps)
    # segments are fixed here, by count, and never recomputed downstream
    size = math.ceil(len(stamps) / SEGMENTS)
    segments = [(stamps[i * size], stamps[min((i + 1) * size, len(stamps)) - 1])
                for i in range(SEGMENTS) if i * size < len(stamps)]
    sources = ['data/raw/top_list.csv', 'data/raw/margin_detail.csv', 'data/raw/forecast.csv',
               'data/raw/daily_prices.csv', 'data/raw/adj_factor_daily.csv',
               'data/raw/daily_basic.csv', 'data/raw/trade_calendar.csv',
               'data/sw_classification_L1.csv',
               'scripts/fuxi_external_info_20260924.py']
    sources += [str(p.relative_to(ROOT).as_posix()) for p in sorted(SIGNAL_DIR.glob('*.parquet'))]
    hashes = {p: digest(ROOT / p) for p in sources}
    protocol = {
        'batch': 'external_info_20260924',
        'purpose': 'work item 3 of the drought review: real information diversity with '
                   'single-channel ablation and per-section pairing',
        'signals': neutral['signals'],
        'signal_dates': stamps,
        'segments': segments,
        'families': FAMILY_COLUMNS,
        'disclosure_rule': {
            'top_list': 'data date == signal date; exchange publishes that evening, before the '
                        'next session open used as entry',
            'margin_detail': 'data date == signal date; exchange publishes that evening, before '
                             'the next session open used as entry',
            'forecast': 'ann_date must be <= the signal date\'s own previous session, so the '
                        'announcement was certainly public before the signal, and no more than '
                        '%d sessions old, so a stale pre-announcement is not read as a current '
                        'event. Conservative on both sides: discards observations rather than '
                        'inventing them.' % FORECAST_WINDOW_SESSIONS,
        },
        'fill_policy': 'sparse families are never zero-filled; the covered set is compared '
                       'against the uncovered set and coverage is reported per section',
        'residual_design': 'resid_base = label ~ 1 + log_size + SW L1 dummies; '
                           'resid_full = resid_base design + equal-weight rank of the four '
                           'frozen signals',
        'thresholds': {'MIN_COVERED': MIN_COVERED, 'MIN_SIDE': MIN_SIDE, 'TOP_N': TOP_N,
                       'MIN_INDUSTRY_N': MIN_INDUSTRY_N, 'SEGMENTS': SEGMENTS,
                       'FORECAST_WINDOW_SESSIONS': FORECAST_WINDOW_SESSIONS},
        'hashes': hashes,
    }
    write(OUT / 'protocol.json', protocol)
    print('frozen segments:', segments)
    print('hashes:', len(hashes))


# --------------------------------------------------------------------------- tests

class ExternalInfoTests(unittest.TestCase):
    """Each test pins a mechanism whose failure would be silent in production."""

    def test_sparse_family_absence_is_never_filled(self):
        """A name with no event must stay absent, not become a zero observation."""
        n = 12
        frame = pd.DataFrame({'log_size': np.arange(n, dtype=float)})
        values = np.array([0.1, np.nan, 0.2, np.nan, np.nan, 0.3,
                           0.4, np.nan, 0.5, np.nan, np.nan, np.nan])
        frame['lh_net_rate'] = values
        labels = np.linspace(0.01, 0.12, n)
        resid = labels.copy()
        row = evaluate_factor(frame, labels, 'lh_net_rate', resid, resid,
                              np.arange(n, dtype=float))
        covered = np.isfinite(values)
        self.assertEqual(row['covered_n'], int(covered.sum()))
        self.assertAlmostEqual(row['coverage'], covered.mean())
        self.assertFalse(row['dense_family'])
        self.assertAlmostEqual(row['contrast_raw'],
                               labels[covered].mean() - labels[~covered].mean())
        self.assertEqual(row['status'], 'valid_descriptive')

    def test_uncovered_side_is_refused_when_too_thin(self):
        frame = pd.DataFrame({'log_size': np.arange(20, dtype=float)})
        frame['x'] = np.nan
        frame.loc[:17, 'x'] = 1.0
        row = evaluate_factor(frame, np.zeros(20), 'x', np.zeros(20), np.zeros(20),
                              np.arange(20, dtype=float))
        self.assertEqual(row['status'], 'uncovered_side_too_thin')

    def test_no_coverage_section_is_reported_not_dropped(self):
        frame = pd.DataFrame({'log_size': np.arange(20, dtype=float)})
        frame['x'] = np.nan
        row = evaluate_factor(frame, np.zeros(20), 'x', np.zeros(20), np.zeros(20),
                              np.arange(20, dtype=float))
        self.assertEqual(row['status'], 'no_coverage')
        self.assertEqual(row['covered_n'], 0)

    def test_dense_family_has_no_event_contrast(self):
        """A family that covers every name has no uncovered side to contrast against."""
        n = 30
        frame = pd.DataFrame({'log_size': np.arange(n, dtype=float)})
        frame['margin_rz_chg5'] = np.linspace(-1, 1, n)
        labels = np.linspace(-0.05, 0.05, n)
        row = evaluate_factor(frame, labels, 'margin_rz_chg5', labels, labels,
                              np.arange(n, dtype=float))
        self.assertTrue(row['dense_family'])
        self.assertIsNone(row['contrast_raw'])
        self.assertEqual(row['status'], 'valid_descriptive')

    def test_default_groupby_drops_open_ended_membership(self):
        """pandas drops NaN group keys by default; a still-open membership is NaT."""
        frame = pd.DataFrame({'con_code': ['A', 'A'], 'in': pd.to_datetime(['2020-01-01'] * 2),
                              'out': pd.to_datetime([None, None]),
                              'industry_name': ['bank', 'media']})
        dropped = frame.groupby(['con_code', 'in', 'out']).industry_name.nunique()
        audited = frame.groupby(['con_code', 'in', 'out'], dropna=False).industry_name.nunique()
        self.assertEqual(len(dropped), 0)
        self.assertEqual(int((audited > 1).sum()), 1)

    def test_csv_writer_survives_a_comma_bearing_value(self):
        """Manual comma joins shift a whole row; csv.writer must be used instead."""
        import io
        import tempfile
        import os
        rows = [{'factor': 'a', 'note': "{'a': 1, 'b': 2}", 'n': 3},
                {'factor': 'b', 'note': 'plain', 'n': 4}]
        handle = tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False,
                                             encoding='utf-8', newline='')
        handle.close()
        try:
            write_csv(handle.name, rows)
            with open(handle.name, encoding='utf-8', newline='') as fh:
                back = list(csv.DictReader(fh))
            self.assertEqual(len(back), 2)
            self.assertEqual(back[0]['note'], "{'a': 1, 'b': 2}")
            self.assertEqual(back[0]['factor'], 'a')
            self.assertEqual(back[1]['n'], '4')
            raw = open(handle.name, 'rb').read()
            self.assertEqual(raw.count(b'\r\n'), 0)
        finally:
            os.unlink(handle.name)

    def test_segments_partition_the_whole_window_without_overlap(self):
        stamps = ['2024-12-%02d' % d for d in range(1, 11)]
        size = math.ceil(len(stamps) / SEGMENTS)
        segs = [(stamps[i * size], stamps[min((i + 1) * size, len(stamps)) - 1])
                for i in range(SEGMENTS) if i * size < len(stamps)]
        self.assertEqual(segs[0][0], stamps[0])
        self.assertEqual(segs[-1][1], stamps[-1])
        for (_, hi), (lo, _) in zip(segs, segs[1:]):
            self.assertLess(hi, lo)

    def test_mean_ci_refuses_to_invent_a_width_from_one_point(self):
        mean, lo, hi, n = mean_ci([0.5])
        self.assertEqual(mean, 0.5)
        self.assertIsNone(lo)
        self.assertIsNone(hi)
        self.assertEqual(n, 1)

    def test_spearman_is_sign_flipped_for_a_reversed_order(self):
        x = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(spearman(x, x), 1.0)
        self.assertAlmostEqual(spearman(x, x[::-1]), -1.0)

    def test_jaccard_is_none_for_two_empty_sets(self):
        self.assertIsNone(jaccard([], []))
        self.assertAlmostEqual(jaccard([1, 2], [2, 3]), 1 / 3)


def tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExternalInfoTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    payload = {'tests': result.testsRun, 'failures': len(result.failures),
               'errors': len(result.errors), 'successful': result.wasSuccessful(),
               'ran_at': datetime.now().isoformat()}
    write(OUT / 'tests.json', payload)
    if not result.wasSuccessful():
        raise SystemExit(1)


def run():
    protocol = read(OUT / 'protocol.json')
    for p, h in protocol['hashes'].items():
        assert digest(ROOT / p) == h, p
    assert not (OUT / 'summary.json').exists()
    with RunContract(parameters={'protocol_sha256': digest(OUT / 'protocol.json'),
                                 'code_data_hashes': protocol['hashes']},
                     run_id='extinfo_' + datetime.now().strftime('%H%M%S_%f'),
                     runs_dir=OUT / 'runs', lock_path=OUT / 'extinfo.lock'):
        rows, drops, geometry, sources = rebuild_sections()
        for p, h in protocol['hashes'].items():
            assert digest(ROOT / p) == h, p
        summaries = summarise(rows, protocol)
        valid = [r for r in rows if r.get('status') == 'valid_descriptive']
        quality = {
            'sources': sources,
            'sections_evaluated': len(valid),
            'factors_per_section': sum(len(v) for v in FAMILY_COLUMNS.values()),
            'sections_planned': (len(protocol['signal_dates']) * len(protocol['signals'])
                                 * sum(len(v) for v in FAMILY_COLUMNS.values())),
            'status_counts': {s: sum(1 for r in rows if r.get('status') == s)
                              for s in sorted({r.get('status') for r in rows})},
            'zero_filled_values': int(sum(1 for r in rows if r.get('zero_filled'))),
            'geometry_min_rank_deficit': min((g['base_columns'] - g['base_rank']
                                               for g in geometry), default=None)
                                          }
        quality['geometry_min_rank_deficit_full'] = min(
            (g['full_columns'] - g['full_rank'] for g in geometry), default=None)
        quality['design_rank_deficit_total'] = int(sum(
            (g['base_columns'] - g['base_rank']) + (g['full_columns'] - g['full_rank'])
            for g in geometry))
        quality['existing_rank_missing_names'] = int(sum(g['existing_missing'] for g in geometry))
        write(OUT / 'summary.json', {'signal_dates': protocol['signal_dates'],
                                     'segments': protocol['segments'],
                                     'thresholds': protocol['thresholds'],
                                     'summaries': summaries,
                                     'quality': quality,
                                     'claims_not_made': [
                                         'labels are uncharged adjusted-open comparisons, not cash NAV',
                                         'no economic verdict for any family',
                                         'the existing-signal rank is a descriptive composite, not a production score',
                                         'research-exposed window; no R5 unknown observations'],
                                     'full_stage2_accepted': False,
                                     'formal_eligibility': False})
        write_csv(OUT / 'per_section.csv', rows)
        write_csv(OUT / 'drops.csv', drops)
        write_csv(OUT / 'geometry.csv', geometry)
        print('done: rows=%d valid=%d drops=%d' % (len(rows), len(valid), len(drops)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['tests', 'freeze', 'run'])
    args = parser.parse_args()
    {'tests': tests, 'freeze': freeze, 'run': run}[args.command]()


if __name__ == '__main__':
    main()
