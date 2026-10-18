"""P0 of the drought follow-up: measure each un-attempted information family's IC
ceiling BEFORE paying for a full pipeline in it.

The drought diagnosis says the binding constraint is information breadth, not model
capacity: break-even IC is a known number (+0.108 for the equal-weight book, report
section 4) while the frozen system delivers +0.054..0.063. So the right first move is
not to build five pipelines but to measure how much IC each of the five never-attempted
families can possibly supply.

What this batch does:

* re-measures the three families already wired in the previous batch as the reference
  point (forecast), and adds the two that were never touched (block trade, industry);
* measures the northbound family's availability honestly: `hk_hold.csv` ends
  2024-12-31 while the evaluation window starts 2024-12-27, so the family has ZERO
  in-window observations. That is "not measurable", which is a different claim from
  "small ceiling", and it is reported as such;
* treats the benchmark family for what it is: an index time series has no
  cross-sectional variance, so it can never be a stock-level factor. Its ceiling is
  measured as a TIMING overlay instead, and reported in its own block.

Discipline carried over unchanged from `fuxi_external_info_20260924`:

* the pool, the label, the residual design and the clustering rule are IMPORTED, not
  re-implemented, and every section is reconciled against the frozen batch before any
  new number is produced;
* sparse families are never zero-filled; covered vs uncovered is compared and both
  sides must be at least MIN_SIDE deep;
* inference is clustered over dates, because the four signals share one market period.

The decisive criterion is frozen in `protocol.json` before any number is produced:

  A family is certified CEILING_BELOW (and downgraded to a conditional feature or an
  exit filter, out of the pipeline) only when the clustered 95% upper bound of its
  ceiling vector's magnitude is below +0.108, its sign is consistent across all three
  pre-registered sub-periods, and it was measurable on at least half the 87 dates. Any
  other outcome -- including "no data" and "too few dates" -- keeps the family in play
  and sends it to P1. The asymmetry is deliberate: this batch is allowed to kill a
  family, never to promote one.
"""
import argparse
import collections
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
import fuxi_external_info_20260924 as ext  # noqa: E402  -- reuse, never re-implement

OUT = ROOT / 'reports/ic_ceiling_20260924'
PRIOR = ROOT / 'reports/external_info_20260924'
NEUTRAL = ROOT / 'reports/target_neutral_20260923'
SIGNAL_DIR = ROOT / 'reports/drought_20260923/signals'

TOP_N = ext.TOP_N
MIN_SIDE = ext.MIN_SIDE
MIN_INDUSTRY_N = ext.MIN_INDUSTRY_N
SEGMENTS = ext.SEGMENTS
MIN_COVERED = ext.MIN_COVERED
FORECAST_WINDOW_SESSIONS = ext.FORECAST_WINDOW_SESSIONS
PREDICTION_END = ext.PREDICTION_END
OTHER = ext.OTHER

# ---------------------------------------------------------------- frozen criterion

# Report section 4: top-minus-pool excess = -0.00843 + 0.0915 * IC, so the equal-weight
# book breaks even at +0.108. Frozen here, before any number is measured.
BREAK_EVEN_IC = 0.108
# Half of the 87 frozen dates. Below this the "below" verdict would rest on a
# non-random subset of dates and is refused rather than qualified.
MIN_DATES_FOR_BELOW = 44
# Industry index levels persist, so a collection gap is forward-filled; the level may
# not be older than this many sessions before it stops describing the current state.
MAX_INDUSTRY_STALENESS = 10
INDEX_MOMENTUM_SESSIONS = 20
MEMBER_MOMENTUM_SESSIONS = 20

VERDICT_BELOW = 'CEILING_BELOW'
VERDICT_UNCLEAR = 'CEILING_UNCLEAR'
VERDICT_ABOVE = 'CEILING_AT_OR_ABOVE'
VERDICT_UNMEASURABLE = 'NOT_MEASURABLE'

# Pre-registered expected sign of each definition, fixed before any number was seen.
# `+1` means "larger value should mean a larger forward residual return". The ceiling
# vector averages the ranks after orienting them by these signs, so a family cannot be
# made to look informative by choosing the orientation after the fact; the per-column
# readings are then a genuine test of the pre-registered direction.
FAMILY_DEFS = {
    'forecast': [
        {'column': 'fcst_yoy_mid', 'sign': +1,
         'meaning': 'midpoint of the pre-announced year-on-year change'},
        {'column': 'fcst_range_ratio', 'sign': -1,
         'meaning': 'width of the pre-announced range, scaled by its own magnitude; '
                    'wider guidance is less certain'},
    ],
    'northbound': [
        {'column': 'nb_ratio', 'sign': +1,
         'meaning': 'last published northbound holding as a share of the float'},
        {'column': 'nb_ratio_chg', 'sign': +1,
         'meaning': 'change in that share against the previous publication of the same '
                    'code, computed on its own observed series'},
    ],
    'block_trade': [
        {'column': 'bt_premium', 'sign': +1,
         'meaning': 'volume-weighted block trade price against that session close; a '
                    'deep discount is selling pressure'},
        {'column': 'bt_share', 'sign': -1,
         'meaning': 'block trade turnover as a share of the session turnover'},
    ],
    'industry': [
        {'column': 'ind_mom20_mem', 'sign': +1,
         'meaning': 'equal-weight 20-session return of the members of the stock\'s SW L1 '
                    'industry, aggregated from prices'},
        {'column': 'ind_mom20_idx', 'sign': +1,
         'meaning': '20-session return of the stock\'s SW L1 industry index itself'},
    ],
}

DISCLOSURE_RULE = {
    'forecast': 'ann_date must be <= the signal date\'s own previous session and no more '
                'than %d sessions old' % FORECAST_WINDOW_SESSIONS,
    'northbound': 'publication date must be <= the signal date\'s own previous session; '
                  'the onshore exchange publishes the holding after the close',
    'block_trade': 'trade date == signal date; the exchange publishes the block trade '
                   'that evening, before the next session\'s open used as entry',
    'industry': 'membership is point-in-time from sw_classification_L1 with its in_date '
                'and out_date; the industry return uses only sessions up to the signal '
                'date',
    'benchmark': 'index close of the signal date or earlier; used as a timing state, '
                 'never as a cross-sectional value',
}


def _daykey(series):
    return series.astype(str).str.strip().str.replace('-', '', regex=False).str.slice(0, 8)


def _iso(day8):
    return '%s-%s-%s' % (day8[:4], day8[4:6], day8[6:8])


# ------------------------------------------------------------------ family builders

def build_northbound(stamps, prior_day_of):
    """Northbound holdings. Published after the close, so only what precedes the signal.

    The table ends 2024-12-31 while the window starts 2024-12-27, so in this batch the
    builder is expected to return nothing. It is implemented anyway: an absent family has
    to be reported as absent, and the moment the gap is backfilled the same code answers
    the ceiling question without a rewrite.
    """
    df = pd.read_csv(ROOT / 'data/raw/hk_hold.csv', dtype=str,
                     usecols=['trade_date', 'ts_code', 'vol', 'ratio', 'exchange'])
    df['day'] = _daykey(df.trade_date)
    df['ratio'] = pd.to_numeric(df.ratio, errors='coerce')
    df['vol'] = pd.to_numeric(df.vol, errors='coerce')
    df = df[df.exchange.isin(['SH', 'SZ'])]
    dup = int(df.duplicated(['day', 'ts_code']).sum())
    df = df.drop_duplicates(['day', 'ts_code'], keep='first')
    df = df.sort_values(['ts_code', 'day'])
    df['ratio_lag1'] = df.groupby('ts_code').ratio.shift(1)
    df['nb_ratio'] = df.ratio
    df['nb_ratio_chg'] = df.ratio - df.ratio_lag1
    wanted = {prior_day_of[s] for s in stamps if s in prior_day_of}
    window = df[df.day.isin(wanted)]
    audit = {'rows': len(df), 'duplicate_rows_dropped': dup, 'window_rows': len(window),
             'days_in_window': int(window.day.nunique()) if len(window) else 0,
             'source_max_day': str(df.day.max()) if len(df) else None,
             'stamps_usable': sorted({s for s in stamps if s in prior_day_of
                                      and prior_day_of[s] in set(window.day)})}
    out = {d: g.set_index('ts_code')[['nb_ratio', 'nb_ratio_chg']]
           for d, g in window.groupby('day')}
    return out, audit


def build_block(stamps, byday):
    """Block trades: an event family, one row per participating name per session.

    The discount is weighted by volume inside the name-day, because a single small print
    should not outweigh a large one. Absence of a name is left absent.

    The session key is normalised to the ISO form used by the price panel before the
    lookup. An eight-digit key against an ISO-keyed map misses on every row, which would
    have produced an empty family that reads exactly like "this family carries nothing".
    """
    df = pd.read_csv(ROOT / 'data/raw/block_trade_full.csv', dtype=str)
    df['day'] = _daykey(df.trade_date)
    df = df[df.day.isin(stamps)].copy()
    for col in ('price', 'vol', 'amount'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[np.isfinite(df.price) & np.isfinite(df.vol) & (df.vol > 0)]
    rows_in_window = len(df)
    close_of, amount_of = {}, {}
    for day in df.day.unique():
        table = byday.get(_iso(day))
        if table is None:
            continue
        close_of[day] = table['close']
        amount_of[day] = table['amount']
    assert rows_in_window == 0 or close_of, \
        'block trade rows exist in window but no session matched the price panel'
    rows = []
    for day, group in df.groupby('day'):
        table = close_of.get(day)
        if table is None:
            continue
        group = group.copy()
        group['close'] = group.ts_code.map(table)
        group = group[np.isfinite(group.close) & (group.close > 0)]
        if group.empty:
            continue
        group['w'] = group.vol
        group['rel'] = group.price / group.close - 1.0
        agg = group.groupby('ts_code').apply(
            lambda g: pd.Series({'bt_premium': float((g.w * g.rel).sum() / g.w.sum()),
                                 'bt_amount': float(g.amount.sum())}),
            include_groups=False)
        agg['bt_share'] = np.where(
            agg.bt_amount > 0,
            agg.bt_amount * 10.0 / agg.index.to_series().map(amount_of[day]),
            np.nan)
        agg['day'] = day
        rows.append(agg.reset_index())
    stacked = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=['ts_code', 'bt_premium', 'bt_amount', 'bt_share', 'day'])
    audit = {'rows_in_window': rows_in_window, 'events': len(stacked),
             'days_with_any': int(stacked.day.nunique()) if len(stacked) else 0}
    assert rows_in_window == 0 or len(stacked) > 0, \
        'block trade builder produced nothing from %d non-empty rows' % rows_in_window
    out = {d: g.set_index('ts_code')[['bt_premium', 'bt_share']]
           for d, g in stacked.groupby('day')}
    return out, audit


def industry_momentum_index(calendar, name_of_code):
    """SW L1 index momentum, forward-filled across collection gaps with a staleness gate.

    The industry index level exists every session; the store simply does not contain
    every session. Forward-filling a level is not the same as zero-filling an event, but
    a level that has not been observed for MAX_INDUSTRY_STALENESS sessions no longer
    describes the present and is dropped instead of carried.

    Columns are renamed from index code to industry name, because the point-in-time
    membership lookup returns names. Leaving the codes in place would make the mapping
    miss on every name and turn a working channel into a silent zero-coverage column.
    """
    codes = sorted(name_of_code)
    df = pd.read_csv(ROOT / 'data/raw/sw_daily.csv', dtype=str,
                     usecols=['ts_code', 'trade_date', 'close'])
    df['day'] = _daykey(df.trade_date).str.slice(0, 8)
    df = df[df.ts_code.isin(codes)]
    df['d'] = pd.to_datetime(df.day, format='%Y%m%d')
    df['close'] = pd.to_numeric(df.close, errors='coerce')
    df = df.drop_duplicates(['ts_code', 'd'], keep='first')
    wide = df.pivot(index='d', columns='ts_code', values='close')
    cal = pd.DatetimeIndex([pd.Timestamp(d) for d in calendar])
    observed_any = wide.notna().any(axis=1)
    last_observed = observed_any[observed_any].index.max()
    # per code, sessions elapsed since its own last observation, evaluated on the calendar
    ages = {code: pd.Series(age_of_observation(wide[code].notna(), cal).to_numpy(), index=cal)
            for code in wide.columns}
    age = pd.DataFrame(ages)
    wide = wide.reindex(wide.index.union(cal)).sort_index().ffill().reindex(cal)
    mom = wide / wide.shift(INDEX_MOMENTUM_SESSIONS) - 1.0
    # a momentum that spans a gap is computed on a level, not on a return; that is the
    # honest reading of a level series, and the staleness gate bounds how stale it is
    mom = mom.where(age.reindex(mom.index) <= MAX_INDUSTRY_STALENESS)
    mom = mom.rename(columns=name_of_code)
    mom.index = [d.strftime('%Y-%m-%d') for d in mom.index]
    unnamed = [c for c in mom.columns if c not in set(name_of_code.values())]
    assert not unnamed, 'industry index column left unmapped: %s' % unnamed[:3]
    # the gate must actually bite where the store ends, or it is decoration. The first
    # MAX_INDUSTRY_STALENESS sessions after the last observation are inside the grace
    # window and legitimately finite; everything beyond it must be empty.
    tail = [d.strftime('%Y-%m-%d') for d in cal if d > last_observed]
    tail = [d for d in tail if d in set(mom.index)]
    grace = tail[:MAX_INDUSTRY_STALENESS]
    beyond = tail[MAX_INDUSTRY_STALENESS:]
    grace_finite = int(mom.loc[grace].notna().to_numpy().sum()) if grace else 0
    beyond_finite = int(mom.loc[beyond].notna().to_numpy().sum()) if beyond else 0
    assert beyond_finite == 0, \
        ('staleness gate did not fire: %d finite values in the %d sessions beyond the '
         '%d-session grace window after the last observation on %s'
         % (beyond_finite, len(beyond), MAX_INDUSTRY_STALENESS, last_observed.date()))
    audit = {'rows': len(df), 'index_codes': int(df.ts_code.nunique()),
             'calendar_sessions': len(cal),
             'last_observed_session': str(last_observed.date()),
             'tail_sessions_after_last_observation': len(tail),
             'gate_grace_window_finite_values': grace_finite,
             'gate_beyond_grace_finite_values': beyond_finite,
             'mom_sessions_available': int(mom.notna().any(axis=1).sum()),
             'staleness_gate_sessions': MAX_INDUSTRY_STALENESS}
    return mom, audit


def member_momentum(merged, stamps, sessions=MEMBER_MOMENTUM_SESSIONS):
    """Equal-weight member return per SW L1 industry, from prices we already have.

    This is the coverage-complete counterpart of the index-based definition: the store's
    sw_daily is a partial-day extract, prices are not.
    """
    px = merged[['trade_date', 'ts_code', 'close']].copy()
    px = px[np.isfinite(px.close) & (px.close > 0)]
    px = px.sort_values(['ts_code', 'trade_date'])
    px['lag'] = px.groupby('ts_code').close.shift(sessions)
    px['mom'] = px.close / px.lag - 1.0
    px = px[px.trade_date.isin(stamps)][['trade_date', 'ts_code', 'mom']]
    return px[['trade_date', 'ts_code', 'mom']]


# ----------------------------------------------------------------------- attribution

def pearson(a, b):
    """Linear correlation over the finite pairs, or None when either side is flat.

    Read next to `spearman`: the break-even arithmetic in report section 4 was derived on
    a RAW rank IC, so a family's raw reading is the one directly comparable to +0.108,
    while the residual reading answers the different question "what does this family add
    that the four frozen signals do not already carry". Reporting both, and the Pearson
    form too, is what makes it visible when a design choice annihilates a family rather
    than the family being empty.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return None
    x, y = x[keep], y[keep]
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ceiling_vector(frame, columns, signs):
    """The family's own best linear summary, oriented by the pre-registered signs.

    Ranks are taken inside the section over the covered names only, so a name the family
    never observed is left out of its own summary rather than given a rank.
    """
    parts = []
    for column, sign in zip(columns, signs):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).any():
            continue
        ranks = pd.Series(values).rank().to_numpy()
        mean = np.nanmean(ranks)
        parts.append(sign * (ranks - mean))
    if not parts:
        return None
    stacked = np.column_stack(parts)
    finite = np.isfinite(stacked)
    counts = finite.sum(axis=1)
    totals = np.where(finite, stacked, 0.0).sum(axis=1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


# --------------------------------------------------------------------------- verdict

def verdict_of(entry, dates_measured):
    """Frozen decision rule. Asymmetric on purpose: may kill, may not promote."""
    if entry is None or not entry.get('n') or dates_measured == 0:
        return VERDICT_UNMEASURABLE, 'no in-window observation'
    mean = entry.get('mean')
    low, high = entry.get('ci_low'), entry.get('ci_high')
    if mean is None:
        return VERDICT_UNMEASURABLE, 'no finite observation'
    magnitude = abs(mean)
    if low is None or high is None:
        bound = None
    else:
        bound = max(abs(low), abs(high))
    if magnitude >= BREAK_EVEN_IC:
        return VERDICT_ABOVE, 'point magnitude %.4f >= %.3f' % (magnitude, BREAK_EVEN_IC)
    if bound is None or bound >= BREAK_EVEN_IC:
        return VERDICT_UNCLEAR, 'upper bound not below break-even'
    if dates_measured < MIN_DATES_FOR_BELOW:
        return VERDICT_UNCLEAR, ('measured on %d dates < %d, so a "below" verdict would '
                                 'rest on a non-random subset of dates'
                                 % (dates_measured, MIN_DATES_FOR_BELOW))
    return VERDICT_BELOW, ('magnitude %.4f with upper bound %.4f below %.3f'
                           % (magnitude, bound, BREAK_EVEN_IC))


def segment_sign_consistent(segments):
    signs = [s['mean'] for s in segments if s['mean'] is not None]
    if not signs:
        return None
    return bool(all(v > 0 for v in signs) or all(v < 0 for v in signs))


# ------------------------------------------------------------------------- main loop

def rebuild_sections():
    protocol = read(OUT / 'protocol.json')
    calendars = ext.public_calendar()
    prior_date = {d: calendars[i - 1] for i, d in enumerate(calendars) if i > 0}
    stamps = sorted(protocol['signal_dates'])

    price, _ = load_panel('data/raw/daily_prices.csv',
                          ['ts_code', 'trade_date', 'open', 'close', 'amount'],
                          lower='20240801')
    adj, _ = load_panel('data/raw/adj_factor_daily.csv',
                        ['ts_code', 'trade_date', 'adj_factor'], lower='20240801')
    merged = price.merge(adj, on=['trade_date', 'ts_code'], validate='one_to_one', how='left')
    merged['adj_open'] = merged.open * merged.adj_factor
    merged.loc[(merged.open <= 0) | (merged.adj_factor <= 0), 'adj_open'] = np.nan
    byday = {d: g.set_index('ts_code') for d, g in merged.groupby('trade_date')}
    print('prices', len(merged), flush=True)

    size_clean, size_stats = ext.fold_size(ROOT / 'data/raw/daily_basic.csv',
                                           {prior_date[d] for d in stamps if d in prior_date})
    sizes = {d: g.set_index('ts_code').total_mv for d, g in size_clean.groupby('trade_date')}
    industry_all = ext.load_industry()
    lookup = ext.pit_industry(industry_all)
    print('size', size_stats, flush=True)

    stamp_keys = set(s.replace('-', '') for s in stamps)
    families, audits = {}, {}
    families['forecast'], audits['forecast'] = ext.build_forecast(
        stamps, {s: prior_date[s].replace('-', '') for s in stamps if s in prior_date}, calendars)
    families['northbound'], audits['northbound'] = build_northbound(
        stamps, {s: prior_date[s].replace('-', '') for s in stamps if s in prior_date})
    families['block_trade'], audits['block_trade'] = build_block(stamp_keys, byday)
    cls = pd.read_csv(ROOT / 'data/sw_classification_L1.csv', dtype=str)
    pairs = cls[['index_code', 'industry_name']].dropna().drop_duplicates()
    variants = pairs.groupby('index_code').industry_name.nunique()
    assert int((variants > 1).sum()) == 0, 'one L1 code carries two names'
    name_of_code = dict(zip(pairs.index_code, pairs.industry_name))
    idx_mom, idx_audit = industry_momentum_index(calendars, name_of_code)
    mem_mom = member_momentum(merged, stamps)
    audits['industry'] = {'index': idx_audit, 'member_rows': len(mem_mom),
                          'member_days': int(mem_mom.trade_date.nunique())}
    for name, audit in audits.items():
        print('family', name, audit, flush=True)

    # index series for the benchmark block
    bench = {}
    for code, name in (('000300.SH', 'index_000300_daily.csv'),
                       ('000905.SH', 'index_000905_daily.csv')):
        frame = pd.read_csv(ROOT / 'data/raw' / name, dtype=str)
        frame['day'] = _daykey(frame.trade_date)
        frame['close'] = pd.to_numeric(frame.close, errors='coerce')
        frame = frame.dropna(subset=['close']).drop_duplicates('day', keep='last')
        bench[code] = frame.set_index('day').close.sort_index()

    prior_rows = {(r['signal'], r['date']): r
                  for r in read(ext.PRIOR / 'retry_v2/cross_sections.json')['rows']
                  if r['status'] == 'valid_prediction_label_only'}
    size_prior = {(r['signal'], r['date']): r
                  for r in read(ext.PRIOR / 'size_supplement/cross_sections.json')['rows']}
    frozen_rows = {(r['signal'], r['date']): r
                   for r in _read_csv(PRIOR / 'per_section.csv')}

    rows, drops, geometry, timing, coverage = [], [], [], [], []
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
            entry_day, exit_day = window
            frame = group.merge(byday[stamp][['close', 'amount']], left_on='ts_code',
                                right_index=True, how='left', validate='one_to_one')
            frame['label'] = (frame.ts_code.map(byday[exit_day].adj_open)
                              / frame.ts_code.map(byday[entry_day].adj_open) - 1)
            frame['log_size'] = np.log(frame.ts_code.map(sizes[prior_date[stamp]]))
            frame['industry'] = frame.ts_code.map(lookup(stamp))

            # parity with the frozen batch before anything new is attributed
            prior = prior_rows[(name, stamp)]
            eligible = frame[np.isfinite(frame.factor_value) & (frame.close > 0)].copy()
            selected = eligible.sort_values(['factor_value', 'amount', 'ts_code'],
                                            ascending=[False, False, True]).head(TOP_N)
            finite = eligible[np.isfinite(eligible.label)]
            top = selected[np.isfinite(selected.label)]
            assert len(eligible) == prior['eligible_n'], (name, stamp)
            assert np.isclose(finite.label.mean(), prior['pool_label_mean'], atol=1e-12)
            assert np.isclose(top.label.mean(), prior['top_label_mean'], atol=1e-12)
            pool_basis = eligible.ts_code.map(sizes[prior_date[stamp]]).dropna()
            top_basis = selected.ts_code.map(sizes[prior_date[stamp]]).dropna()
            raw_diff = float(np.log(top_basis).mean() - np.log(pool_basis).mean())
            assert np.isclose(raw_diff, size_prior[(name, stamp)]['log_size_difference'],
                              atol=1e-9), (name, stamp)

            pool = finite[np.isfinite(finite.log_size)].copy()
            if len(pool) < 50:
                drops.append({'signal': name, 'date': stamp, 'reason': 'pool_too_small',
                              'pool_n': len(pool)})
                continue
            frozen = frozen_rows.get((name, stamp))
            if frozen is not None and frozen['status'] == 'valid_descriptive':
                assert len(pool) == int(frozen['pool_n']), (name, stamp, len(pool))
            basis, reference, columns = ext.industry_basis(
                pool.ts_code.tolist(), pool.set_index('ts_code').industry)
            known_share = float(pool.industry.notna().mean())
            assert not (known_share > 0.5 and len(columns) == 0), \
                'industry map missed silently (%s, %s)' % (name, stamp)

            ranks = []
            for other in protocol['signals']:
                table = peer[other]
                table = table[table.trade_date == stamp].set_index('ts_code').factor_value
                aligned = pool.ts_code.map(table)
                ranks.append(pd.Series(np.asarray(aligned, dtype=float)).rank().to_numpy())
            with np.errstate(invalid='ignore'):
                existing = np.nanmean(np.column_stack(ranks), axis=1)
            pool = pool.assign(existing=existing)
            label_np = pool.label.to_numpy(dtype=float)

            base_design = ext.design_matrix(pool, basis, reference, columns)
            full_design = np.column_stack([base_design, pool.existing.to_numpy(dtype=float)])
            resid_base, _, rank_base = ext.residualise(label_np, base_design)
            resid_full, _, rank_full = ext.residualise(label_np, full_design)
            geometry.append({'signal': name, 'date': stamp, 'base_columns': base_design.shape[1],
                             'base_rank': rank_base, 'full_columns': full_design.shape[1],
                             'full_rank': rank_full,
                             'existing_missing': int(np.sum(~np.isfinite(existing)))})
            assert rank_base == base_design.shape[1], (name, stamp)
            assert rank_full == full_design.shape[1], (name, stamp)

            day8 = stamp.replace('-', '')
            lag8 = prior_date[stamp].replace('-', '')
            attaches = _attach_families(pool, families, idx_mom, mem_mom, lookup, stamp,
                                        day8, lag8)
            existing_np = pool.existing.to_numpy(dtype=float)
            for family, defs in FAMILY_DEFS.items():
                cols = [d['column'] for d in defs]
                signs = [d['sign'] for d in defs]
                frame_v = pool.copy()
                for column in cols:
                    frame_v[column] = attaches[family][column]
                for column in cols:
                    values = frame_v[column].to_numpy(dtype=float)
                    coverage.append({'signal': name, 'date': stamp, 'family': family,
                                     'column': column, 'pool_n': len(pool),
                                     'covered_n': int(np.isfinite(values).sum()),
                                     'coverage': float(np.isfinite(values).mean())})
                for column in cols:
                    row = ext.evaluate_factor(frame_v, label_np, column, resid_base,
                                              resid_full, existing_np)
                    row.update(_linear_readings(frame_v[column].to_numpy(dtype=float),
                                                label_np, resid_base, resid_full))
                    row.update({'factor': column, 'family': family, 'signal': name,
                                'date': stamp, 'entry': entry_day, 'lag_date': prior_date[stamp],
                                'industry_columns': len(columns), 'zero_filled': 0})
                    rows.append(row)
                vec = ceiling_vector(frame_v, cols, signs)
                if vec is not None:
                    frame_v['__ceiling__'] = vec
                    row = ext.evaluate_factor(frame_v, label_np, '__ceiling__', resid_base,
                                              resid_full, existing_np)
                    row.update(_linear_readings(vec, label_np, resid_base, resid_full))
                    row.update({'factor': family + '__ceiling', 'family': family,
                                'signal': name, 'date': stamp, 'entry': entry_day,
                                'lag_date': prior_date[stamp],
                                'industry_columns': len(columns), 'zero_filled': 0})
                    rows.append(row)

            # benchmark block: a timing state, never a cross-sectional value
            excess = (float(top.label.mean()) - float(finite.label.mean())) if len(top) else None
            timing.append({'signal': name, 'date': stamp, 'entry': entry_day,
                           'excess': excess,
                           'idx300_ma20': _ma20_state(bench['000300.SH'], calendars, lag8),
                           'idx300_mom20': _mom(bench['000300.SH'], calendars, lag8, 20),
                           'idx905_mom20': _mom(bench['000905.SH'], calendars, lag8, 20)})
            print('  %s %s pool=%d industries=%d' % (name, stamp, len(pool), len(columns)),
                  flush=True)
    selfcheck = builder_selfcheck(audits, coverage)
    for name, audit in audits.items():
        if name == 'northbound':
            audit['measurability'] = ('no_source_rows_in_window'
                                      if audit['rows'] > 0 and audit['window_rows'] == 0
                                      else 'source_rows_present')
        elif name == 'block_trade':
            audit['measurability'] = ('no_source_rows_in_window'
                                      if audit['rows_in_window'] == 0
                                      else 'source_rows_present')
    return (rows, drops, geometry, timing, coverage,
            {'size': size_stats, 'families': audits, 'selfcheck': selfcheck,
             'industry_codes': int(industry_all.con_code.nunique())})


def builder_selfcheck(audits, coverage):
    """A builder that quietly produced nothing is a defect, not a finding.

    Both defects found while freezing this batch had the same shape: a key-format
    mismatch turned a non-empty family into an empty one, which the verdict rule would
    have reported as NOT_MEASURABLE -- that is, as a fact about the family rather than
    about our code. The family-level audits above refuse that for the two builders whose
    input is non-empty in this window; this check covers the derived industry column.
    """
    idx = [c for c in coverage if c.get('column') == 'ind_mom20_idx']
    total = int(sum(c['covered_n'] for c in idx))
    available = audits['industry']['index']['mom_sessions_available']
    assert available == 0 or total > 0, \
        ('industry index has %d sessions of momentum in the calendar but zero covered '
         'names in every section' % available)
    return {'industry_index_covered_names_total': total}


def _read_csv(path):
    with open(path, encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _linear_readings(values, labels, resid_base, resid_full):
    """Same three contrasts as `evaluate_factor`, in linear form.

    A rank correlation is not zero when the linear one is, because ties inside a group
    still produce rank differences. Measured on the industry family, whose value is
    constant within an SW L1 industry, the two forms disagree in a way that matters: the
    residual design contains the industry dummies, so the residual sums to zero inside
    every industry and the linear correlation with a within-industry-constant factor is
    exactly zero. The rank reading is then an artifact of the ties, not a return relation,
    and only the raw column is comparable to the break-even IC.
    """
    return {'pearson_raw': pearson(values, labels),
            'pearson_resid': pearson(values, resid_base),
            'pearson_resid_existing': pearson(values, resid_full)}


def _attach_families(pool, families, idx_mom, mem_mom, lookup, stamp, day8, lag8):
    """Per-family column vectors aligned to the pool, absence preserved as absence."""
    out = {}
    for family, defs in FAMILY_DEFS.items():
        cols = [d['column'] for d in defs]
        block = {}
        if family == 'industry':
            member = lookup(stamp)
            industry_of = pool.ts_code.map(member)
            day_mem = mem_mom[mem_mom.trade_date == stamp]
            by_ind = day_mem.groupby(day_mem.ts_code.map(member).rename('ind')).mom.mean()
            by_ind_idx = idx_mom.loc[stamp] if stamp in idx_mom.index else None
            block['ind_mom20_mem'] = industry_of.map(by_ind)
            block['ind_mom20_idx'] = (industry_of.map(by_ind_idx)
                                      if by_ind_idx is not None else pd.Series(
                                          np.nan, index=pool.index))
        else:
            table = families[family].get(lag8 if family == 'northbound' else day8)
            for column in cols:
                if table is None or column not in table.columns:
                    block[column] = pd.Series(np.nan, index=pool.index)
                else:
                    block[column] = pool.ts_code.map(table[column])
        out[family] = {k: np.asarray(v, dtype=float) for k, v in block.items()}
    return out


def _mom(series, calendar, day8, sessions):
    days = [d for d in calendar if d.replace('-', '') <= day8]
    if len(days) <= sessions:
        return None
    here, then = days[-1], days[-1 - sessions]
    key_here, key_then = here.replace('-', ''), then.replace('-', '')
    if key_here not in series.index or key_then not in series.index:
        return None
    a, b = series[key_here], series[key_then]
    return float(a / b - 1.0) if b else None


def _ma20_state(series, calendar, day8, sessions=20):
    days = [d for d in calendar if d.replace('-', '') <= day8]
    if len(days) < sessions:
        return None
    keys = [d.replace('-', '') for d in days[-sessions:]]
    if any(k not in series.index for k in keys):
        return None
    window = series[keys]
    return int(float(window.iloc[-1]) > float(window.mean()))


# ---------------------------------------------------------------------- summarising

def cluster(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {'n': 0, 'mean': None, 'ci_low': None, 'ci_high': None, 't': None,
                'positive': 0}
    n = len(vals)
    mean = float(np.mean(vals))
    if n < 2:
        return {'n': 1, 'mean': mean, 'ci_low': None, 'ci_high': None, 't': None,
                'positive': int(vals[0] > 0)}
    sd = float(np.std(vals, ddof=1))
    se = sd / math.sqrt(n)
    half = 1.96 * se if se > 0 else 0.0
    return {'n': n, 'mean': mean, 'ci_low': mean - half, 'ci_high': mean + half,
            't': (mean / se) if se > 0 else None,
            'positive': int(sum(1 for v in vals if v > 0))}


def by_date(rows, metric=('ic_resid_existing', 'ic_resid')):
    if isinstance(metric, str):
        return _by_date_one(rows, metric)
    return _by_date_one(rows, metric[0])


def _by_date_one(rows, metric):
    per_date = collections.defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            per_date[row['date']].append(value)
    return cluster([sum(v) / len(v) for v in per_date.values()]), len(per_date)


def summarise(rows, protocol, coverage):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row['factor']].append(row)
    out = {}
    for factor, mine in sorted(grouped.items()):
        valid = [r for r in mine if r.get('status') == 'valid_descriptive']
        entry = {'family': mine[0]['family'], 'sections_total': len(mine),
                 'sections_valid': len(valid),
                 'sections_planned': len(protocol['signal_dates']) * len(protocol['signals'])}
        for key in ('coverage', 'contrast_raw', 'contrast_resid', 'contrast_resid_existing',
                    'ic_raw', 'ic_resid', 'ic_resid_existing', 'overlap_existing',
                    'pearson_raw', 'pearson_resid', 'pearson_resid_existing'):
            vals = [r.get(key) for r in valid]
            entry[key] = _plain(vals)
        for metric in ('ic_resid_existing', 'ic_resid', 'ic_raw', 'pearson_resid_existing',
                       'pearson_raw'):
            stats, dates = _by_date_one(valid, metric)
            entry['clustered_' + metric] = stats
            entry['clustered_' + metric + '_dates'] = dates
        segs = []
        for index, (low, high) in enumerate(protocol['segments']):
            block = [r for r in valid if low <= r['date'] <= high]
            stats, dates = _by_date_one(block, 'ic_resid_existing')
            segs.append({'segment': index, 'window': [low, high], 'dates': dates,
                         'ic_resid_existing': stats,
                         'contrast_resid_existing': _by_date_one(
                             block, 'contrast_resid_existing')[0]})
        entry['segments'] = segs
        entry['segment_sign_consistent'] = segment_sign_consistent(
            [{'mean': s['ic_resid_existing']['mean']} for s in segs])
        cov = [c for c in coverage if c['family'] == entry['family']]
        entry['coverage_mean'] = float(np.mean([c['coverage'] for c in cov])) if cov else None
        entry['coverage_min'] = float(np.min([c['coverage'] for c in cov])) if cov else None
        out[factor] = entry
    return out


def _plain(vals):
    clean = [float(v) for v in vals if v is not None and isinstance(v, (int, float))
             and np.isfinite(v)]
    if not clean:
        return {'n': 0, 'mean': None, 't': None, 'positive': 0}
    return {'n': len(clean), 'mean': float(np.mean(clean)), 't': ext.t_stat(clean),
            'positive': int(sum(1 for v in clean if v > 0))}


def timing_block(timing, protocol):
    """The benchmark family's only usable form: a market state, not a stock score."""
    per_date = collections.defaultdict(list)
    meta = {}
    for row in timing:
        if row['excess'] is None:
            continue
        per_date[row['date']].append(float(row['excess']))
        meta[row['date']] = row
    dates = sorted(per_date)
    book = {d: float(np.mean(per_date[d])) for d in dates}
    out = {'unit': 'clustered over dates; the four signals share one market period',
           'dates': len(dates), 'mean_excess': cluster(list(book.values())),
           'states': {}}
    for key in ('idx300_ma20', 'idx300_mom20', 'idx905_mom20'):
        if key.endswith('ma20'):
            groups = {1: [], 0: []}
            for d in dates:
                state = meta[d].get(key)
                if state in groups:
                    groups[state].append(book[d])
            if not groups[0] or not groups[1]:
                out['states'][key] = {'n': {k: len(v) for k, v in groups.items()}}
                continue
            diff = float(np.mean(groups[1]) - np.mean(groups[0]))
            # Welch interval on the difference of two date clusters
            a, b = groups[1], groups[0]
            va = float(np.var(a, ddof=1) / len(a)) if len(a) > 1 else None
            vb = float(np.var(b, ddof=1) / len(b)) if len(b) > 1 else None
            if va is None or vb is None:
                out['states'][key] = {'above': cluster(a), 'below': cluster(b),
                                      'n': {1: len(a), 0: len(b)}}
                continue
            se = math.sqrt(va + vb)
            out['states'][key] = {
                'above': cluster(a), 'below': cluster(b),
                'difference': {'mean': diff, 'se': se,
                               't': (diff / se) if se > 0 else None,
                               'ci_low': diff - 1.96 * se, 'ci_high': diff + 1.96 * se},
                'n': {1: len(a), 0: len(b)}}
        else:
            ranked = [(meta[d].get(key), book[d]) for d in dates]
            ranked = [(r, e) for r, e in ranked if r is not None]
            values = [r for r, _ in ranked]
            excess = [e for _, e in ranked]
            signs = [1 if e > 0 else 0 for e in excess]
            out['states'][key] = {
                'n': len(ranked),
                'rank_correlation_with_excess': ext.spearman(values, excess),
                'share_positive_excess': float(np.mean(signs)) if signs else None}
    return out


# ------------------------------------------------------------------------------ run

def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'protocol already frozen; remove explicitly'
    neutral = read(NEUTRAL / 'protocol.json')
    stamps = sorted(neutral['signal_dates'])
    assert len(stamps) == 87, len(stamps)
    prior = read(PRIOR / 'protocol.json')
    sources = ['data/raw/block_trade_full.csv', 'data/raw/hk_hold.csv', 'data/raw/sw_daily.csv',
               'data/sw_classification_L1.csv', 'data/raw/index_000300_daily.csv',
               'data/raw/index_000905_daily.csv', 'data/raw/daily_prices.csv',
               'data/raw/adj_factor_daily.csv', 'data/raw/daily_basic.csv',
               'data/raw/trade_calendar.csv', 'data/raw/forecast.csv',
               'scripts/fuxi_ic_ceiling_20260924.py',
               'scripts/fuxi_external_info_20260924.py']
    sources += [str(p.relative_to(ROOT).as_posix()) for p in sorted(SIGNAL_DIR.glob('*.parquet'))]
    hashes = {p: digest(ROOT / p) for p in sources}
    protocol = {
        'batch': 'ic_ceiling_20260924',
        'purpose': 'P0 of the drought follow-up: measure each never-attempted information '
                   'family\'s attainable IC before paying for a pipeline in it',
        'signals': neutral['signals'],
        'signal_dates': stamps,
        'segments': prior.get('segments') or read(ext.OUT / 'protocol.json')['segments'],
        'family_definitions': {k: v for k, v in FAMILY_DEFS.items()},
        'benchmark_note': 'an index time series has no cross-sectional variance, so the '
                          'benchmark family cannot be a stock-level factor; its ceiling is '
                          'measured as a timing overlay in timing.csv and excluded from the '
                          'cross-sectional table',
        'fill_policy': 'sparse families are never zero-filled; covered and uncovered sides '
                       'are contrasted and each side needs at least MIN_SIDE names',
        'residual_design': 'imported unchanged from external_info_20260924: resid_base = '
                           'label ~ 1 + log_size + SW L1 dummies; resid_full = resid_base '
                           'design + equal-weight rank of the four frozen signals',
        'inference': 'clustered over dates: each factor\'s per-date statistic is averaged '
                     'across the four signals first, then t-tested over dates only',
        'criterion': {
            'break_even_ic': BREAK_EVEN_IC,
            'break_even_source': 'report section 4: excess = -0.00843 + 0.0915 * IC for the '
                                 'equal-weight book, so IC must reach +0.108 to break even',
            'min_dates_for_below': MIN_DATES_FOR_BELOW,
            'rule': 'CEILING_AT_OR_ABOVE if |point| >= break-even; CEILING_BELOW only if the '
                    'clustered upper bound of |mean| is below break-even AND the sign is '
                    'consistent across all three segments AND at least %d dates were '
                    'measured; otherwise CEILING_UNCLEAR; NOT_MEASURABLE if the family has '
                    'no finite in-window reading' % MIN_DATES_FOR_BELOW,
            'asymmetry': 'this batch may downgrade a family, never promote one: only '
                         'CEILING_BELOW takes a family out of P1',
        },
        'thresholds': {'TOP_N': TOP_N, 'MIN_SIDE': MIN_SIDE, 'MIN_COVERED': MIN_COVERED,
                       'MIN_INDUSTRY_N': MIN_INDUSTRY_N, 'SEGMENTS': SEGMENTS,
                       'FORECAST_WINDOW_SESSIONS': FORECAST_WINDOW_SESSIONS,
                       'INDEX_MOMENTUM_SESSIONS': INDEX_MOMENTUM_SESSIONS,
                       'MEMBER_MOMENTUM_SESSIONS': MEMBER_MOMENTUM_SESSIONS,
                       'MAX_INDUSTRY_STALENESS': MAX_INDUSTRY_STALENESS},
        'disclosure_rule': DISCLOSURE_RULE,
        'hashes': hashes,
    }
    write(OUT / 'protocol.json', protocol)
    print('frozen segments:', protocol['segments'])
    print('hashes:', len(hashes))


def run():
    protocol = read(OUT / 'protocol.json')
    for p, h in protocol['hashes'].items():
        assert digest(ROOT / p) == h, p
    assert not (OUT / 'summary.json').exists()
    with RunContract(parameters={'protocol_sha256': digest(OUT / 'protocol.json'),
                                 'code_data_hashes': protocol['hashes'],
                                 'break_even_ic': BREAK_EVEN_IC,
                                 'min_dates_for_below': MIN_DATES_FOR_BELOW},
                     run_id='icceil_' + datetime.now().strftime('%H%M%S_%f'),
                     runs_dir=OUT / 'runs', lock_path=OUT / 'icceil.lock'):
        rows, drops, geometry, timing, coverage, sources = rebuild_sections()
        for p, h in protocol['hashes'].items():
            assert digest(ROOT / p) == h, p
        summaries = summarise(rows, protocol, coverage)
        ceilings = {}
        for family, defs in FAMILY_DEFS.items():
            factor = family + '__ceiling'
            entry = summaries.get(factor)
            if entry is None:
                ceilings[family] = {'verdict': VERDICT_UNMEASURABLE,
                                    'reason': 'no ceiling vector produced',
                                    'definitions': [d['column'] for d in defs]}
                continue
            stats = entry['clustered_ic_resid_existing']
            dates = entry['clustered_ic_resid_existing_dates']
            verdict, reason = verdict_of(stats, dates)
            if verdict == VERDICT_ABOVE and not entry['segment_sign_consistent']:
                verdict, reason = VERDICT_UNCLEAR, 'point magnitude at break-even but the ' \
                                                   'sign flips across segments'
            ceilings[family] = {
                'verdict': verdict, 'reason': reason,
                'definitions': [d['column'] for d in defs],
                'signs': [d['sign'] for d in defs],
                'dates_measured': dates,
                'coverage_mean': entry['coverage_mean'],
                'ceiling_ic_resid_existing': stats,
                'ceiling_ic_resid': entry['clustered_ic_resid'],
                'ceiling_ic_raw': entry['clustered_ic_raw'],
                'ceiling_pearson_resid_existing': entry['clustered_pearson_resid_existing'],
                'ceiling_pearson_raw': entry['clustered_pearson_raw'],
                'raw_sensitivity': {
                    'note': 'the frozen verdict above uses the residual reading; the '
                            'break-even arithmetic of report section 4 was derived on a RAW '
                            'rank IC, so this is the reading that is directly comparable. It '
                            'is reported as a sensitivity and did not decide the verdict.',
                    'stats': entry['clustered_ic_raw'],
                    'dates_measured': entry['clustered_ic_raw_dates'],
                    'magnitude': _magnitude(entry['clustered_ic_raw']),
                    'upper_bound_on_magnitude': _bound(entry['clustered_ic_raw']),
                    'verdict_if_raw_were_primary': verdict_of(
                        entry['clustered_ic_raw'], entry['clustered_ic_raw_dates'])[0]},
                'segment_sign_consistent': entry['segment_sign_consistent'],
                'per_definition': {d['column']: {
                    'sign_expected': d['sign'],
                    'sign_observed': _sign(summaries.get(d['column'], {}).get(
                        'clustered_ic_resid_existing', {}).get('mean')),
                    'clustered_ic_resid_existing': summaries.get(d['column'], {}).get(
                        'clustered_ic_resid_existing'),
                    'clustered_ic_raw': summaries.get(d['column'], {}).get('clustered_ic_raw'),
                    'clustered_pearson_resid_existing': summaries.get(d['column'], {}).get(
                        'clustered_pearson_resid_existing'),
                    'dates_measured': summaries.get(d['column'], {}).get(
                        'clustered_ic_resid_existing_dates'),
                    'coverage_mean': summaries.get(d['column'], {}).get('coverage_mean')}
                    for d in defs}}
            ceilings[family]['enters_p1'] = verdict != VERDICT_BELOW
        valid = [r for r in rows if r.get('status') == 'valid_descriptive']
        quality = {
            'sources': sources,
            'sections_evaluated': len(valid),
            'status_counts': {s: sum(1 for r in rows if r.get('status') == s)
                              for s in sorted({r.get('status') for r in rows})},
            'zero_filled_values': int(sum(1 for r in rows if r.get('zero_filled'))),
            'dates_with_any_family_data': {
                family: len({r['date'] for r in rows if r['family'] == family
                             and r.get('status') == 'valid_descriptive'})
                for family in FAMILY_DEFS},
            'geometry_min_rank_deficit': min((g['base_columns'] - g['base_rank']
                                              for g in geometry), default=None),
            'geometry_min_rank_deficit_full': min((g['full_columns'] - g['full_rank']
                                                   for g in geometry), default=None),
            'design_rank_deficit_total': int(sum(
                (g['base_columns'] - g['base_rank']) + (g['full_columns'] - g['full_rank'])
                for g in geometry)),
            'existing_rank_missing_names': int(sum(g['existing_missing'] for g in geometry)),
            'parity_assertions_passed': True,
        }
        write(OUT / 'summary.json', {
            'signal_dates': protocol['signal_dates'], 'segments': protocol['segments'],
            'break_even_ic': BREAK_EVEN_IC, 'thresholds': protocol['thresholds'],
            'summaries': summaries, 'ceilings': ceilings,
            'timing': timing_block(timing, protocol), 'quality': quality,
            'claims_not_made': [
                'labels are uncharged adjusted-open comparisons, not cash NAV',
                'no economic verdict for any family; a ceiling is not a strategy',
                'the ceiling vector is a descriptive linear summary of one family\'s own '
                'columns, oriented by pre-registered signs; it is not a production score',
                'a CEILING_BELOW verdict is a statement about our data and this window under '
                'these definitions, not a claim that the family carries no information',
                'research-exposed window; no R5 unknown observations',
                'the northbound reading is an availability statement about our store, not '
                'about the factor'],
            'full_stage2_accepted': False, 'formal_eligibility': False})
        ext.write_csv(OUT / 'per_section.csv', rows)
        ext.write_csv(OUT / 'drops.csv', drops)
        ext.write_csv(OUT / 'geometry.csv', geometry)
        ext.write_csv(OUT / 'timing.csv', timing)
        ext.write_csv(OUT / 'coverage.csv', coverage)
        print('done: rows=%d valid=%d drops=%d' % (len(rows), len(valid), len(drops)))
        print('ceilings:', {k: v['verdict'] for k, v in sorted(ceilings.items())})


def _sign(value):
    if value is None:
        return None
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _magnitude(stats):
    mean = (stats or {}).get('mean')
    return None if mean is None else abs(mean)


def _bound(stats):
    """Upper bound on |mean| implied by the clustered interval, or None."""
    stats = stats or {}
    low, high = stats.get('ci_low'), stats.get('ci_high')
    if low is None or high is None:
        return None
    return max(abs(low), abs(high))


# --------------------------------------------------------------------------- tests

class IcCeilingTests(unittest.TestCase):
    """Each test pins a mechanism whose failure would be silent in production."""

    def test_below_verdict_needs_the_upper_bound_under_break_even(self):
        entry = {'n': 80, 'mean': 0.02, 'ci_low': 0.01, 'ci_high': 0.03}
        verdict, _ = verdict_of(entry, 80)
        self.assertEqual(verdict, VERDICT_BELOW)

    def test_a_near_break_even_mean_is_not_certified_below(self):
        """A point estimate under break-even is not enough: the bound must clear it too."""
        entry = {'n': 80, 'mean': 0.09, 'ci_low': 0.02, 'ci_high': 0.16}
        verdict, _ = verdict_of(entry, 80)
        self.assertEqual(verdict, VERDICT_UNCLEAR)

    def test_a_few_dates_cannot_produce_a_below_verdict(self):
        """Measured on a non-random subset of dates, "below" would be a false certainty."""
        entry = {'n': 38, 'mean': 0.01, 'ci_low': 0.005, 'ci_high': 0.015}
        verdict, reason = verdict_of(entry, 38)
        self.assertEqual(verdict, VERDICT_UNCLEAR)
        self.assertIn('non-random', reason)

    def test_no_observation_is_unmeasurable_not_small(self):
        """An absent family must not be reported as a low ceiling."""
        verdict, _ = verdict_of(None, 0)
        self.assertEqual(verdict, VERDICT_UNMEASURABLE)
        verdict, _ = verdict_of({'n': 0, 'mean': None}, 0)
        self.assertEqual(verdict, VERDICT_UNMEASURABLE)

    def test_a_negative_ceiling_is_read_by_magnitude(self):
        """A directionally usable family is measured by |IC|; the sign is reported apart."""
        entry = {'n': 80, 'mean': -0.14, 'ci_low': -0.18, 'ci_high': -0.10}
        verdict, _ = verdict_of(entry, 80)
        self.assertEqual(verdict, VERDICT_ABOVE)

    def test_ceiling_vector_orients_by_the_pre_registered_sign(self):
        """Flipping the registered sign must flip the summary, not the other way round."""
        frame = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [4.0, 3.0, 2.0, 1.0]})
        plus = ceiling_vector(frame, ['a', 'b'], [+1, +1])
        minus = ceiling_vector(frame, ['a', 'b'], [+1, -1])
        self.assertAlmostEqual(float(plus[3]), 0.0)
        self.assertGreater(float(minus[3]), 0.0)

    def test_ceiling_vector_ignores_a_column_without_any_observation(self):
        frame = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [np.nan] * 3})
        vec = ceiling_vector(frame, ['a', 'b'], [+1, +1])
        self.assertTrue(np.isfinite(vec).all())

    def test_ceiling_vector_leaves_an_uncovered_name_uncovered(self):
        frame = pd.DataFrame({'a': [1.0, np.nan, 3.0, 4.0]})
        vec = ceiling_vector(frame, ['a'], [+1])
        self.assertTrue(np.isnan(vec[1]))

    def test_by_date_clustering_averages_signals_before_the_t_test(self):
        """Treating four same-day signals as independent inflates t by about sqrt(4)."""
        rows = []
        for date in ('2025-01-03', '2025-01-10', '2025-01-17', '2025-01-24'):
            for signal in ('a', 'b', 'c', 'd'):
                rows.append({'date': date, 'ic_resid_existing': 0.10})
        stats, dates = _by_date_one(rows, 'ic_resid_existing')
        self.assertEqual(stats['n'], 4)
        self.assertEqual(dates, 4)
        self.assertIsNone(stats['t'])          # zero dispersion across dates
        naive = cluster([r['ic_resid_existing'] for r in rows])
        self.assertEqual(naive['n'], 16)

    def test_timing_state_splits_dates_not_sections(self):
        timing = [{'signal': s, 'date': '2025-01-03', 'excess': 0.01, 'idx300_ma20': 1,
                   'idx300_mom20': 0.0, 'idx905_mom20': 0.0}
                  for s in ('a', 'b', 'c', 'd')]
        timing += [{'signal': s, 'date': '2025-01-10', 'excess': -0.02, 'idx300_ma20': 0,
                    'idx300_mom20': 0.0, 'idx905_mom20': 0.0}
                   for s in ('a', 'b', 'c', 'd')]
        block = timing_block(timing, {'segments': [('2025-01-01', '2025-01-31')]})
        self.assertEqual(block['dates'], 2)
        self.assertAlmostEqual(block['mean_excess']['mean'], -0.005)
        self.assertAlmostEqual(block['states']['idx300_ma20']['above']['mean'], 0.01)
        self.assertAlmostEqual(block['states']['idx300_ma20']['below']['mean'], -0.02)

    def test_a_stale_industry_level_is_dropped_not_carried(self):
        """A level unobserved for longer than the gate no longer describes the present.

        The input has to look like production: `observed` carries ONLY the observed
        dates and the calendar is longer. A series with an explicit False on every
        calendar date passes even when the age computation is broken, which is how the
        gate stayed dead for two rounds.
        """
        calendar = pd.DatetimeIndex(['2025-01-%02d' % d for d in range(1, 26)])
        observed_only = pd.Series([True, True], index=calendar[[0, 1]])   # only two dates
        age = age_of_observation(observed_only, calendar)
        self.assertEqual(int(age.iloc[0]), 0)
        self.assertEqual(int(age.iloc[1]), 0)
        self.assertEqual(int(age.iloc[2]), 1)
        self.assertEqual(int(age.iloc[10]), 9)
        self.assertEqual(int(age.iloc[24]), 23)
        self.assertGreater(int(age.iloc[24]), MAX_INDUSTRY_STALENESS)
        self.assertFalse(bool(np.isnan(age.iloc[0])))    # the first observation is age 0

    def test_industry_momentum_uses_a_level_ratio(self):
        self.assertAlmostEqual(_ratio(110.0, 100.0), 0.10)

    def test_missing_side_gate_rejects_a_one_name_family(self):
        frame = pd.DataFrame({'log_size': np.arange(20, dtype=float)})
        frame['x'] = np.nan
        frame.loc[:18, 'x'] = 1.0
        row = ext.evaluate_factor(frame, np.zeros(20), 'x', np.zeros(20), np.zeros(20),
                                  np.arange(20, dtype=float))
        self.assertEqual(row['status'], 'uncovered_side_too_thin')

    def test_block_premium_weights_by_volume_inside_the_day(self):
        """One small print must not outweigh one large one."""
        ts = pd.DataFrame({'price': [10.0, 11.0], 'vol': [100.0, 1.0], 'rel': [-0.10, 0.10],
                           'w': [100.0, 1.0]})
        value = float((ts.w * ts.rel).sum() / ts.w.sum())
        self.assertLess(value, -0.09)
        self.assertGreater(value, -0.10)

    def test_eight_digit_session_keys_are_normalised_before_a_panel_lookup(self):
        """The panel is keyed by ISO dates; an eight-digit key misses on every row."""
        self.assertEqual(_iso('20250103'), '2025-01-03')
        self.assertNotEqual(_iso('20250103'), '20250103')

    def test_a_builder_that_swallowed_a_non_empty_family_is_refused(self):
        """An empty family and a failed builder must not read the same."""
        audits = {'industry': {'index': {'mom_sessions_available': 1346}}}
        coverage = [{'column': 'ind_mom20_idx', 'covered_n': 0},
                    {'column': 'ind_mom20_mem', 'covered_n': 54}]
        with self.assertRaises(AssertionError):
            builder_selfcheck(audits, coverage)
        coverage[0]['covered_n'] = 12
        self.assertEqual(builder_selfcheck(audits, coverage)[
            'industry_index_covered_names_total'], 12)

    def test_measurability_reason_separates_no_data_from_no_finding(self):
        """`source absent` and `all sections too thin` must not collapse into one label."""
        audits = {'industry': {'index': {'mom_sessions_available': 0}}}
        self.assertEqual(builder_selfcheck(audits, [])['industry_index_covered_names_total'], 0)

    def test_a_group_constant_factor_is_annihilated_by_group_dummies(self):
        """The residual design carries the industry dummies, so the residual sums to zero
        inside every industry. A factor constant within industry therefore has EXACTLY zero
        linear correlation with it -- the rank reading survives only through the ties. This
        is why a rank-only reading can make a whole family look empty, and why the raw
        reading is reported alongside it.
        """
        rows = []
        rng = np.random.default_rng(0)
        for g in range(3):
            for i in range(6):
                rows.append({'group': 'g%d' % g, 'log_size': float(i),
                             'label': 0.3 * g + float(rng.normal(scale=0.05))})
        frame = pd.DataFrame(rows)
        basis = frame.group.to_numpy()
        design = np.column_stack([np.ones(len(frame)), frame.log_size.to_numpy(),
                                  (basis == 'g1').astype(float), (basis == 'g2').astype(float)])
        resid, _, rank = ext.residualise(frame.label.to_numpy(dtype=float), design)
        self.assertEqual(rank, design.shape[1])
        self.assertGreater(float(np.std(resid)), 1e-6)   # the label is not in the design span
        group_constant = frame.group.map({'g0': 1.0, 'g1': 2.0, 'g2': 3.0}).to_numpy()
        self.assertLess(abs(pearson(group_constant, resid)), 1e-10)
        within = rng.normal(size=len(frame))
        self.assertGreater(abs(pearson(within, resid)), 1e-3)


def age_of_observation(observed, calendar):
    """Sessions since the last observation, measured on the calendar.

    `observed` normally carries only the dates that were observed, so every calendar
    date missing from it is an ABSENCE and must become False before the scan. Reindexing
    to the union of the two would leave the shorter index untouched, every value True,
    and the age identically zero -- a gate that is present in the code, referenced in
    the audit and never fires. That is what happened on the first two rounds of this
    batch, and it is why the assertion above the call site now checks the tail.
    """
    seen = observed.reindex(calendar).fillna(False).astype(bool)
    positions = pd.Series(np.arange(len(calendar), dtype=float), index=calendar)
    last = positions.where(seen).ffill()
    return positions - last


def _ratio(now, then):
    return now / then - 1.0


def tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IcCeilingTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write(OUT / 'tests.json', {'tests': result.testsRun, 'failures': len(result.failures),
                               'errors': len(result.errors),
                               'successful': result.wasSuccessful(),
                               'ran_at': datetime.now().isoformat()})
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['tests', 'freeze', 'run'])
    args = parser.parse_args()
    {'tests': tests, 'freeze': freeze, 'run': run}[args.command]()


if __name__ == '__main__':
    main()
