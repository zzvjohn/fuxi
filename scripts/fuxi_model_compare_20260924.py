# -*- coding: utf-8 -*-
"""Item 4 of the drought review: a fair model comparison on one frozen design.

Frozen before the first evaluation:
  * feature set (coverage rule + redundancy rule, both mechanical)
  * a-priori direction per feature
  * label, pool, Top80 selection, costs, benchmark
  * expanding-window training with a purge, tree seeds and hyper-parameters
Nothing but the combiner differs between arms.

The four existing frozen signals are NOT in the feature set; `r4_equal_rank` is
carried as a descriptive reference arm put through the identical pipeline.
"""

import argparse
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
from fuxi_execution_lab_20260923 import digest, read, write  # noqa: E402

OUT = ROOT / 'reports/model_compare_20260924'
CACHE = OUT / 'cache'
PRIOR = ROOT / 'reports/drought_20260923'

PREDICTION_START = '2024-12-27'
PREDICTION_END = '2026-09-09'

# name, kind, a-priori direction, snapshot column
# Directions are frozen priors, not fitted: small size / high turnover / high
# valuation / high volatility / overbought = negative; dividend yield = positive;
# short-horizon reversal = negative; 60-session momentum = positive.
CANDIDATES = (
    ('log_size', 'snapshot', -1, 'total_mv'),
    ('log_circ_size', 'snapshot', -1, 'circ_mv'),
    ('turnover_rate', 'snapshot', -1, 'turnover_rate'),
    ('turnover_rate_f', 'snapshot', -1, 'turnover_rate_f'),
    ('volume_ratio', 'snapshot', -1, 'volume_ratio'),
    ('pb', 'snapshot', -1, 'pb'),
    ('ps_ttm', 'snapshot', -1, 'ps_ttm'),
    ('dv_ratio', 'snapshot', 1, 'dv_ratio'),
    ('pe_ttm', 'snapshot', -1, 'pe_ttm'),
    ('ret_5', 'price', -1, None),
    ('ret_20', 'price', -1, None),
    ('ret_60', 'price', 1, None),
    ('rv20', 'price', -1, None),
    ('amount_ratio_20', 'price', -1, None),
    ('close_position_20', 'price', -1, None),
)

MIN_MEDIAN_COVERAGE = 0.90
MAX_ABS_RANK_CORR = 0.95
MIN_POOL = 300
TOP_N = 80
COST_BPS = (0, 15, 30)
PRIMARY_COST_BPS = 15
RIDGE_ALPHA = 1.0
TREE_PARAMS = {'n_estimators': 100, 'max_depth': 3, 'min_samples_leaf': 500,
               'max_features': 'sqrt'}
TREE_SEEDS = (0, 1, 2, 3, 4)
MIN_TRAIN_SECTIONS = 26

ARMS = ('equal_rank', 'ridge', 'tree')
REFERENCE = 'r4_equal_rank'

SOURCES = ['data/raw/daily_prices.csv', 'data/raw/daily_basic.csv',
           'data/raw/adj_factor_daily.csv', 'data/raw/trade_calendar.csv',
           'data/raw/index_000985_daily.csv']
SNAPSHOT_COLUMNS = ['total_mv', 'circ_mv', 'turnover_rate', 'turnover_rate_f',
                    'volume_ratio', 'pb', 'ps_ttm', 'dv_ratio', 'pe_ttm']


# --------------------------------------------------------------------- calendar
def calendar():
    cal = pd.read_csv(ROOT / 'data/raw/trade_calendar.csv', dtype=str)
    cal = cal[(cal.exchange == 'SSE') & (cal.is_open == '1')]
    dates = sorted(pd.to_datetime(cal.cal_date, format='mixed').dt.strftime('%Y-%m-%d').unique())
    return [d for d in dates if d <= PREDICTION_END]


def weekly_stamps(dates):
    series = pd.Series(pd.to_datetime(dates))
    weekly = series.groupby(series.dt.to_period('W-SUN')).max().dt.strftime('%Y-%m-%d').tolist()
    return [d for d in weekly if d >= PREDICTION_START]


def label_dates(dates, stamp, horizon=5):
    """Entry strictly after the signal; exit `horizon` market sessions later."""
    later = [d for d in dates if d > stamp]
    return (later[0], later[horizon]) if len(later) > horizon else None


def lag_map(dates, stamps):
    index = {d: i for i, d in enumerate(dates)}
    return index, {s: dates[index[s] - 1] for s in stamps if index[s] > 0}


# ----------------------------------------------------------------- panel loading
def fold_snapshot(frame, columns):
    """Per-column safe fold: identical finite duplicates fold, others are quarantined."""
    keys = ['trade_date', 'ts_code']
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    folded = frame[keys].drop_duplicates().copy()
    audit = {}
    for column in columns:
        valid = np.isfinite(frame[column].to_numpy())
        work = frame.assign(_valid=valid, _value=frame[column])
        grouped = work.groupby(keys, dropna=False, sort=False).agg(
            rows=('_valid', 'size'), valid=('_valid', 'sum'),
            low=('_value', 'min'), high=('_value', 'max'))
        safe = (grouped.rows == grouped.valid) & (grouped.low == grouped.high)
        audit[column] = {'keys': int(grouped.shape[0]),
                         'duplicate_keys': int((grouped.rows > 1).sum()),
                         'unsafe_keys': int((~safe).sum())}
        folded = folded.merge(grouped.low.where(safe).rename(column).reset_index(),
                              on=keys, how='left', validate='one_to_one', sort=False)
    return folded, audit


def _chunked(path, columns, wanted, date_column='trade_date'):
    pieces = []
    for part in pd.read_csv(ROOT / path, usecols=columns, dtype={date_column: str},
                            chunksize=400000):
        part[date_column] = (pd.to_datetime(part[date_column], format='mixed', errors='raise')
                             .dt.strftime('%Y-%m-%d'))
        keep = part[part[date_column].isin(wanted)]
        if len(keep):
            pieces.append(keep)
    return pd.concat(pieces, ignore_index=True)


def build_cache():
    dates = calendar()
    stamps = weekly_stamps(dates)
    index, lag = lag_map(dates, stamps)
    needed = sorted({d for s in stamps for d in dates[max(0, index[s] - 60):index[s] + 7]})
    needed = set(d for d in needed if d <= PREDICTION_END)
    price = _chunked('data/raw/daily_prices.csv',
                     ['ts_code', 'trade_date', 'close', 'amount'], needed)
    if price.duplicated(['trade_date', 'ts_code']).any():
        raise ValueError('duplicate price keys')
    wide_close = price.pivot_table(index='trade_date', columns='ts_code',
                                   values='close', aggfunc='first')
    wide_amount = price.pivot_table(index='trade_date', columns='ts_code',
                                    values='amount', aggfunc='first')
    raw = _chunked('data/raw/daily_basic.csv',
                   ['ts_code', 'trade_date'] + SNAPSHOT_COLUMNS, set(lag.values()))
    snapshot, audit = fold_snapshot(raw, SNAPSHOT_COLUMNS)
    CACHE.mkdir(parents=True, exist_ok=True)
    wide_close.to_parquet(CACHE / 'wide_close.parquet')
    wide_amount.to_parquet(CACHE / 'wide_amount.parquet')
    snapshot.to_parquet(CACHE / 'folded_snapshot.parquet')
    write(CACHE / 'fold_audit.json', {'raw_rows': int(len(raw)), 'folded_keys': int(len(snapshot)),
                                      'audit': audit, 'lag_dates': int(len(set(lag.values())))})
    print('cache built: price rows %d, snapshot raw %d -> folded %d'
          % (len(price), len(raw), len(snapshot)), flush=True)
    return dates, stamps, index, lag, wide_close, wide_amount, snapshot


def load_cache():
    if not (CACHE / 'folded_snapshot.parquet').exists():
        return build_cache()
    wide_close = pd.read_parquet(CACHE / 'wide_close.parquet')
    wide_amount = pd.read_parquet(CACHE / 'wide_amount.parquet')
    snapshot = pd.read_parquet(CACHE / 'folded_snapshot.parquet')
    dates = calendar()
    stamps = weekly_stamps(dates)
    index, lag = lag_map(dates, stamps)
    return dates, stamps, index, lag, wide_close, wide_amount, snapshot


# ----------------------------------------------------------------------- features
def derived_frame(dates, index, stamp, wide_close, wide_amount, snapshot, wanted, pool):
    position = index[stamp]
    table = snapshot[snapshot.trade_date == wanted].set_index('ts_code')
    history = dates[max(0, position - 60):position + 1]
    sub_close = wide_close.reindex(history).reindex(columns=pool)
    sub_amount = wide_amount.reindex(history).reindex(columns=pool)
    last = sub_close.iloc[-1]
    rows = {
        'log_size': np.log(table.reindex(pool).total_mv.where(table.reindex(pool).total_mv > 0)),
        'log_circ_size': np.log(table.reindex(pool).circ_mv.where(table.reindex(pool).circ_mv > 0)),
        'turnover_rate': table.reindex(pool).turnover_rate,
        'turnover_rate_f': table.reindex(pool).turnover_rate_f,
        'volume_ratio': table.reindex(pool).volume_ratio,
        'pb': table.reindex(pool).pb.where(table.reindex(pool).pb > 0),
        'ps_ttm': table.reindex(pool).ps_ttm,
        'dv_ratio': table.reindex(pool).dv_ratio,
        'pe_ttm': table.reindex(pool).pe_ttm.where(table.reindex(pool).pe_ttm > 0),
        'ret_5': last / sub_close.iloc[-6] - 1 if len(history) >= 6 else np.nan,
        'ret_20': last / sub_close.iloc[-21] - 1 if len(history) >= 21 else np.nan,
        'ret_60': last / sub_close.iloc[-61] - 1 if len(history) >= 61 else np.nan,
        'rv20': sub_close.iloc[-21:].pct_change().std(ddof=0) if len(history) >= 21 else np.nan,
        'amount_ratio_20': (sub_amount.iloc[-1] / sub_amount.iloc[-21:-1].mean()
                            if len(history) >= 21 else np.nan),
        'close_position_20': ((last - sub_close.iloc[-21:].min()) /
                              (sub_close.iloc[-21:].max() - sub_close.iloc[-21:].min())
                              if len(history) >= 21 else np.nan),
        'amount': wide_amount.loc[stamp].reindex(pool) if stamp in wide_amount.index else np.nan,
    }
    frame = pd.DataFrame(rows)
    frame.index.name = 'ts_code'
    return frame


def features_at(dates, index, stamp, wide_close, wide_amount, snapshot, wanted, names):
    if stamp not in wide_close.index or wanted is None:
        return None
    close = wide_close.loc[stamp]
    pool = close[close > 0].index
    table = snapshot[snapshot.trade_date == wanted]
    if len(table) == 0:
        return None
    pool = pool.intersection(table.ts_code)
    if len(pool) < MIN_POOL:
        return None
    frame = derived_frame(dates, index, stamp, wide_close, wide_amount, snapshot, wanted, pool)
    keep = np.ones(len(frame), dtype=bool)
    for name in names:
        keep &= np.isfinite(pd.to_numeric(frame[name], errors='coerce').to_numpy())
    frame = frame[keep]
    return None if len(frame) < MIN_POOL else frame


def normalise(frame, features):
    """Cross-sectional rank percentile in [-1, 1], oriented by the frozen direction."""
    out = pd.DataFrame(index=frame.index)
    for name, _kind, direction, _column in features:
        values = pd.to_numeric(frame[name], errors='coerce')
        out[name] = values.rank(pct=True, method='average') * 2 - 1
        out[name] = out[name] * direction
    return out


# ---------------------------------------------------------------------- freezing
def coverage_of(dates, stamps, index, lag, wide_close, wide_amount, snapshot, names):
    rates = {name: [] for name in names}
    pools = {}
    for stamp in stamps:
        if stamp not in lag or stamp not in wide_close.index or index[stamp] < 60:
            continue
        close = wide_close.loc[stamp]
        pool = close[close > 0].index
        table = snapshot[snapshot.trade_date == lag[stamp]]
        if len(table) == 0:
            continue
        pool = pool.intersection(table.ts_code)
        pools[stamp] = int(len(pool))
        if len(pool) == 0:
            continue
        frame = derived_frame(dates, index, stamp, wide_close, wide_amount, snapshot,
                              lag[stamp], pool)
        for name in names:
            rates[name].append(float(np.isfinite(
                pd.to_numeric(frame[name], errors='coerce')).mean()))
    return rates, pools


def redundancy(dates, stamps, index, lag, wide_close, wide_amount, snapshot, features):
    names = [f[0] for f in features]
    samples = []
    for stamp in stamps:
        frame = features_at(dates, index, stamp, wide_close, wide_amount, snapshot,
                            lag.get(stamp), names)
        if frame is not None:
            samples.append(normalise(frame, features))
    matrix = pd.DataFrame(np.nan, index=names, columns=names)
    for i, left in enumerate(names):
        for right in names[i:]:
            values = [abs(frame[left].corr(frame[right], method='spearman'))
                      for frame in samples]
            values = [v for v in values if math.isfinite(v)]
            matrix.loc[left, right] = matrix.loc[right, left] = (
                float(np.median(values)) if values else float('nan'))
    return matrix


DESIGN_FIELDS = ('features', 'dropped_by_coverage', 'dropped_by_redundancy',
                 'selection_rules', 'directions_note', 'label', 'benchmark',
                 'selection', 'top_n', 'cost_bps', 'primary_cost_bps', 'cost_model',
                 'arms', 'ridge', 'tree', 'training', 'segments', 'reference_arm')


def freeze(reason=None):
    """Freeze the design once. A re-freeze is allowed only for plumbing fixes."""
    history = OUT / 'protocol_history'
    previous = None
    if (OUT / 'protocol.json').exists():
        if not reason:
            raise RuntimeError('protocol already frozen; pass --reason to re-freeze')
        previous = read(OUT / 'protocol.json')
        history.mkdir(parents=True, exist_ok=True)
        revision = len(list(history.glob('protocol_*.json')))
        write(history / ('protocol_%02d.json' % revision), previous)
    OUT.mkdir(parents=True, exist_ok=True)
    dates, stamps, index, lag, wide_close, wide_amount, snapshot = load_cache()
    names = [c[0] for c in CANDIDATES]
    rates, pools = coverage_of(dates, stamps, index, lag, wide_close, wide_amount,
                               snapshot, names)
    coverage = {}
    for name in names:
        values = rates.get(name, [])
        coverage[name] = {'sections': len(values),
                          'median_rate': float(np.median(values)) if values else None,
                          'min_rate': float(min(values)) if values else None}
    kept = [c for c in CANDIDATES if (coverage[c[0]]['median_rate'] or 0) >= MIN_MEDIAN_COVERAGE]
    dropped_coverage = [c[0] for c in CANDIDATES if c not in kept]
    matrix = redundancy(dates, stamps, index, lag, wide_close, wide_amount, snapshot, kept)
    kept_names, dropped_redundant, pairs = [], [], []
    for candidate in kept:
        name = candidate[0]
        clash = [other for other in kept_names if matrix.loc[name, other] >= MAX_ABS_RANK_CORR]
        if clash:
            dropped_redundant.append(name)
            pairs.append({'dropped': name, 'kept_because_correlated_with': clash[0],
                          'median_abs_rank_corr': float(matrix.loc[name, clash[0]])})
        else:
            kept_names.append(name)
    features = tuple(c for c in kept if c[0] in kept_names)
    write(OUT / 'redundancy.json',
          {'threshold': MAX_ABS_RANK_CORR, 'dropped': pairs,
           'matrix': {a: {b: float(matrix.loc[a, b]) for b in matrix.columns}
                      for a in matrix.index}})
    write(OUT / 'coverage.json',
          {'rule': 'keep iff median finite rate >= %.2f' % MIN_MEDIAN_COVERAGE,
           'per_feature': coverage, 'dropped_by_coverage': dropped_coverage,
           'pool_sections': len(pools),
           'pool_median': float(np.median(list(pools.values()))),
           'pool_min': int(min(pools.values())), 'pool_max': int(max(pools.values()))})
    hashes = {p: digest(ROOT / p) for p in SOURCES}
    code = ['scripts/fuxi_model_compare_20260924.py']
    hashes.update({p: digest(ROOT / p) for p in code})
    reference = PRIOR / 'signals' / (REFERENCE + '.parquet')
    hashes[str(reference.relative_to(ROOT)).replace('\\', '/')] = digest(reference)
    protocol = {
        'batch': 'model_compare_20260924',
        'purpose': 'work item 4 of the drought review: one frozen design, three combiners',
        'features': [{'name': n, 'kind': k, 'direction': d, 'source': c,
                      'coverage': coverage[n]} for n, k, d, c in features],
        'dropped_by_coverage': dropped_coverage,
        'dropped_by_redundancy': dropped_redundant,
        'selection_rules': {
            'coverage': 'keep iff median finite rate >= %.2f on the price/snapshot pool'
                        % MIN_MEDIAN_COVERAGE,
            'redundancy': 'in frozen order, drop a candidate whose median cross-sectional '
                          '|spearman| against an already-kept candidate is >= %.2f'
                          % MAX_ABS_RANK_CORR,
            'imputation': 'none; complete case over the kept features',
            'pool': 'positive raw close at the signal date, present in the lagged snapshot, '
                    'all kept features finite; minimum %d names' % MIN_POOL},
        'directions_note': 'frozen a-priori directions, applied to every arm identically',
        'label': 'adj_open[exit] / adj_open[entry] - 1; entry = first session after the '
                 'signal, exit = five sessions after entry; no forward fill',
        'benchmark': {'path': 'data/raw/index_000985_daily.csv',
                      'definition': 'index open[exit] / index open[entry] - 1, the same '
                                    'window as the label'},
        'selection': 'Top%d by score descending, contemporaneous amount descending, '
                     'code ascending' % TOP_N,
        'top_n': TOP_N, 'cost_bps': list(COST_BPS), 'primary_cost_bps': PRIMARY_COST_BPS,
        'cost_model': 'cost fraction = 2 * (names replaced / top_n) * bps / 1e4; name '
                      'turnover only, no market impact, no capacity, no tax',
        'arms': list(ARMS),
        'ridge': {'alpha': RIDGE_ALPHA, 'fit_intercept': True},
        'tree': {'estimator': 'sklearn.ensemble.RandomForestRegressor', 'params': TREE_PARAMS,
                 'seeds': list(TREE_SEEDS), 'score': 'mean prediction across seeds'},
        'training': {'scheme': 'expanding window over strictly earlier sections',
                     'purge': 'a training section is excluded unless its label exit is '
                              'strictly before the test section entry',
                     'min_train_sections': MIN_TRAIN_SECTIONS},
        'segments': 'thirds by index over the trained sections, frozen before the run',
        'reference_arm': REFERENCE,
        'hashes': hashes,
    }
    protocol['revisions'] = []
    if previous is not None:
        changed = [field for field in DESIGN_FIELDS
                   if protocol.get(field) != previous.get(field)]
        moved = [p for p in previous['hashes']
                 if p not in code and previous['hashes'][p] != hashes.get(p)]
        if moved:
            raise RuntimeError('source hashes changed; this is a new experiment: %s' % moved)
        protocol['revisions'] = list(previous.get('revisions', [])) + [{
            'previous_protocol_sha256': digest(OUT / 'protocol.json'),
            'reason': reason, 'design_fields_changed': changed,
            'code_hash_changed': previous['hashes'].get('scripts/fuxi_model_compare_20260924.py')
            != hashes.get('scripts/fuxi_model_compare_20260924.py')}]
        if changed:
            raise RuntimeError('design fields changed during a plumbing re-freeze: %s' % changed)
    write(OUT / 'protocol.json', protocol)
    print('frozen features (%d): %s' % (len(kept_names), kept_names), flush=True)
    print('dropped by coverage: %s' % dropped_coverage, flush=True)
    print('dropped by redundancy: %s' % pairs, flush=True)
    print('pool median %d min %d max %d' % (np.median(list(pools.values())),
                                            min(pools.values()), max(pools.values())), flush=True)


# ------------------------------------------------------------------------ running
def top_selection(frame, score):
    """Frozen rule: score desc, contemporaneous amount desc, code asc."""
    work = frame.assign(score=score)
    if work.index.name == 'ts_code' and 'ts_code' not in work.columns:
        work = work.reset_index()
    return work.sort_values(['score', 'amount', 'ts_code'],
                            ascending=[False, False, True]).head(TOP_N)


def cluster(values):
    values = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if len(values) < 2:
        return None
    spread = float(values.std(ddof=1))
    return {'n': int(len(values)), 'mean': float(values.mean()), 'std': spread,
            't': float(values.mean() / (spread / math.sqrt(len(values)))) if spread > 0 else None}


def run():
    protocol = read(OUT / 'protocol.json')
    if (OUT / 'summary.json').exists():
        raise RuntimeError('summary already exists; refusing to overwrite')
    for path, expected in protocol['hashes'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('hash mismatch: ' + path)
    features = tuple((f['name'], f['kind'], f['direction'], f['source'])
                     for f in protocol['features'])
    names = [f[0] for f in features]
    dates, stamps, index, lag, wide_close, wide_amount, snapshot = load_cache()

    sections = []
    wanted = set()
    for stamp in stamps:
        window = label_dates(dates, stamp)
        if window is None:
            sections.append({'stamp': stamp, 'status': 'label_not_matured'})
            continue
        entry, exit_day = window
        wanted.update({entry, exit_day})
        sections.append({'stamp': stamp, 'entry': entry, 'exit': exit_day, 'status': 'pending'})
    opens = _chunked('data/raw/daily_prices.csv', ['ts_code', 'trade_date', 'open'], wanted)
    if opens.duplicated(['trade_date', 'ts_code']).any():
        raise ValueError('duplicate open keys')
    open_wide = opens.pivot_table(index='trade_date', columns='ts_code', values='open',
                                  aggfunc='first')
    adj = _chunked('data/raw/adj_factor_daily.csv', ['ts_code', 'trade_date', 'adj_factor'],
                   wanted)
    if adj.duplicated(['trade_date', 'ts_code']).any():
        raise ValueError('duplicate adjustment keys')
    adj_wide = adj.pivot_table(index='trade_date', columns='ts_code', values='adj_factor',
                               aggfunc='first')
    bench = pd.read_csv(ROOT / 'data/raw/index_000985_daily.csv', dtype={'trade_date': str})
    bench['trade_date'] = pd.to_datetime(bench.trade_date, format='mixed').dt.strftime('%Y-%m-%d')
    bench = bench.set_index('trade_date').open

    records, pools = {}, {}
    for section in sections:
        if section['status'] != 'pending':
            continue
        stamp, entry, exit_day = section['stamp'], section['entry'], section['exit']
        frame = features_at(dates, index, stamp, wide_close, wide_amount, snapshot,
                            lag.get(stamp), names)
        if frame is None:
            section['status'] = 'pool_below_minimum'
            continue
        if entry not in open_wide.index or exit_day not in open_wide.index \
                or entry not in adj_wide.index or exit_day not in adj_wide.index:
            section['status'] = 'open_or_adjustment_missing'
            continue
        entry_open = open_wide.loc[entry] * adj_wide.loc[entry]
        exit_open = open_wide.loc[exit_day] * adj_wide.loc[exit_day]
        label = exit_open / entry_open - 1
        frame = frame.assign(label=frame.index.map(label))
        frame = frame[np.isfinite(pd.to_numeric(frame.label, errors='coerce'))]
        if len(frame) < MIN_POOL:
            section['status'] = 'labelled_pool_below_minimum'
            continue
        benchmark = None
        if entry in bench.index and exit_day in bench.index and bench.loc[entry] > 0:
            benchmark = float(bench.loc[exit_day] / bench.loc[entry] - 1)
        records[stamp] = frame
        pools[stamp] = int(len(frame))
        section['status'] = 'evaluated'
        section['pool'] = int(len(frame))
        section['benchmark'] = benchmark

    evaluated = [s for s in sections if s['status'] == 'evaluated']
    order = [s['stamp'] for s in evaluated]
    span = {s['stamp']: s for s in evaluated}
    print('sections %d total / %d evaluated' % (len(stamps), len(evaluated)), flush=True)

    scores = {arm: {} for arm in ARMS}
    seed_scores = {}
    schedule = {}
    # Normalise once per section; every arm consumes the identical matrix.
    panels = {}
    for stamp in order:
        normalised = normalise(records[stamp], features)[names]
        normalised = normalised.assign(
            label=pd.to_numeric(records[stamp].label, errors='coerce'))
        panels[stamp] = normalised
        scores['equal_rank'][stamp] = normalised[names].mean(axis=1)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    for position, stamp in enumerate(order):
        entry = span[stamp]['entry']
        training = [earlier for earlier in order[:position] if span[earlier]['exit'] < entry]
        schedule[stamp] = len(training)
        if len(training) < MIN_TRAIN_SECTIONS:
            continue
        X = pd.concat([panels[e][names] for e in training], ignore_index=True)
        y = pd.concat([panels[e].label for e in training], ignore_index=True)
        valid = np.isfinite(y.to_numpy()) & np.isfinite(X.to_numpy()).all(axis=1)
        X, y = X[valid], y[valid]
        if len(X) < MIN_POOL * 5:
            continue
        normalised = panels[stamp][names]
        ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        ridge.fit(X.to_numpy(), y.to_numpy())
        scores['ridge'][stamp] = pd.Series(ridge.predict(normalised.to_numpy()),
                                           index=normalised.index)
        predictions, per_seed = [], []
        for seed in TREE_SEEDS:
            forest = RandomForestRegressor(random_state=seed, n_jobs=1, **TREE_PARAMS)
            forest.fit(X.to_numpy(), y.to_numpy())
            values = forest.predict(normalised.to_numpy())
            predictions.append(values)
            per_seed.append(pd.Series(values, index=normalised.index))
        seed_scores[stamp] = per_seed
        scores['tree'][stamp] = pd.Series(np.mean(predictions, axis=0),
                                          index=normalised.index)
        if (position + 1) % 20 == 0:
            print('  trained through %d/%d' % (position + 1, len(order)), flush=True)

    trained = [s for s in order if scores['ridge'].get(s) is not None]
    print('trained sections %d' % len(trained), flush=True)

    reference = pd.read_parquet(PRIOR / 'signals' / (REFERENCE + '.parquet'))
    reference['trade_date'] = pd.to_datetime(reference.trade_date).dt.strftime('%Y-%m-%d')
    reference = reference[reference.trade_date.isin(trained)]
    reference_panels = {d: g.set_index('ts_code').factor_value
                        for d, g in reference.groupby('trade_date')}

    rows, seed_rows = [], []
    for arm in list(ARMS) + [REFERENCE]:
        if arm == REFERENCE:
            arm_stamps = [s for s in trained if s in reference_panels]
        else:
            arm_stamps = [s for s in trained if scores[arm].get(s) is not None]
        previous = set()
        for stamp in arm_stamps:
            frame = records[stamp]
            if arm == REFERENCE:
                series = frame.index.map(reference_panels[stamp])
                series = pd.Series(series, index=frame.index)
                finite = np.isfinite(series.to_numpy())
                score = series[finite]
                work = frame.loc[score.index]
            else:
                score = scores[arm][stamp]
                work = frame
            selected = top_selection(work, score)
            chosen = set(selected.ts_code)
            replaced = len(previous - chosen) if previous else TOP_N
            previous = chosen
            turnover = replaced / TOP_N
            section = span[stamp]
            benchmark = section['benchmark']
            gross = float(selected.label.mean())
            row = {'arm': arm, 'stamp': stamp, 'entry': section['entry'],
                   'exit': section['exit'], 'pool': int(len(work)),
                   'selected': int(len(selected)), 'complete': len(selected) == TOP_N,
                   'one_way_turnover': float(turnover), 'gross_return': gross,
                   'pool_return': float(work.label.mean()),
                   'gross_excess_vs_pool': gross - float(work.label.mean()),
                   'benchmark': benchmark,
                   'rank_ic': float(work.label.corr(score, method='spearman')),
                   'train_sections': schedule.get(stamp)}
            if benchmark is not None:
                row['gross_excess_vs_benchmark'] = gross - benchmark
                for bps in COST_BPS:
                    cost = 2 * turnover * bps / 1e4
                    row['cost_%d' % bps] = cost
                    row['net_excess_%d' % bps] = gross - benchmark - cost
                    row['net_return_%d' % bps] = gross - cost
            rows.append(row)
            if arm == 'tree' and stamp in seed_scores:
                for seed, series in zip(TREE_SEEDS, seed_scores[stamp]):
                    pick = top_selection(frame, series)
                    seed_rows.append({'stamp': stamp, 'seed': seed,
                                      'gross_return': float(pick.label.mean()),
                                      'benchmark': benchmark})

    summary = {'protocol_sha256': digest(OUT / 'protocol.json'),
               'sections_total': len(stamps), 'sections_evaluated': len(evaluated),
               'sections_trained': len(trained),
               'median_pool': float(np.median(list(pools.values()))) if pools else None,
               'status_counts': {}, 'arms': {}, 'reference': {},
               'seed_dispersion': {}, 'segments': {}, 'segment_readouts': {}}
    for section in sections:
        key = section['status']
        summary['status_counts'][key] = summary['status_counts'].get(key, 0) + 1
    for arm in list(ARMS) + [REFERENCE]:
        subset = [r for r in rows if r['arm'] == arm]
        entry = {'sections': len(subset),
                 'complete_tops': sum(1 for r in subset if r['complete']),
                 'mean_one_way_turnover': float(np.mean([r['one_way_turnover'] for r in subset])),
                 'rank_ic': cluster([r['rank_ic'] for r in subset]),
                 'gross_return': cluster([r['gross_return'] for r in subset]),
                 'gross_excess_vs_pool': cluster([r['gross_excess_vs_pool'] for r in subset])}
        for bps in COST_BPS:
            entry['net_excess_%d' % bps] = cluster([r.get('net_excess_%d' % bps) for r in subset])
            entry['net_return_%d' % bps] = cluster([r.get('net_return_%d' % bps) for r in subset])
        summary['arms' if arm in ARMS else 'reference'][arm] = entry

    if seed_rows:
        seed_frame = pd.DataFrame(seed_rows)
        by_seed = seed_frame.groupby('seed').gross_return.mean()
        summary['seed_dispersion'] = {
            'seeds': {int(k): float(v) for k, v in by_seed.items()},
            'across_seed_std': float(by_seed.std(ddof=1)),
            'across_seed_min': float(by_seed.min()), 'across_seed_max': float(by_seed.max()),
            'arm_uses': 'mean prediction across seeds'}
    count = len(trained)
    thirds = {'S0': trained[:count // 3], 'S1': trained[count // 3: 2 * count // 3],
              'S2': trained[2 * count // 3:]}
    summary['segments'] = {key: {'from': value[0], 'to': value[-1], 'sections': len(value)}
                           for key, value in thirds.items() if value}
    for arm in list(ARMS) + [REFERENCE]:
        for key, value in thirds.items():
            subset = [r for r in rows if r['arm'] == arm and r['stamp'] in set(value)]
            if not subset:
                continue
            summary['segment_readouts'].setdefault(arm, {})[key] = {
                'sections': len(subset),
                'gross_excess_vs_benchmark': cluster(
                    [r.get('gross_excess_vs_benchmark') for r in subset]),
                'net_excess_%d' % PRIMARY_COST_BPS: cluster(
                    [r.get('net_excess_%d' % PRIMARY_COST_BPS) for r in subset])}
    write(OUT / 'sections.json', rows)
    write_csv(OUT / 'sections.csv', rows)
    write(OUT / 'summary.json', summary)
    write(OUT / 'run_log.json', {s['stamp']: s for s in sections})
    print(json.dumps({'evaluated': len(evaluated), 'trained': len(trained),
                      'net_excess_15': {a: summary['arms'][a]['net_excess_%d' % PRIMARY_COST_BPS]
                                        for a in ARMS},
                      'reference_net_excess_15': summary['reference'][REFERENCE]
                      ['net_excess_%d' % PRIMARY_COST_BPS]},
                     ensure_ascii=False, indent=1), flush=True)


def write_csv(path, rows):
    import csv as _csv
    columns = sorted({k for row in rows for k in row})
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = _csv.writer(handle, lineterminator='\n')
        writer.writerow(columns)
        for row in rows:
            writer.writerow(['' if row.get(c) is None else row.get(c) for c in columns])


class ModelCompareTests(unittest.TestCase):
    def test_label_uses_next_session_entry(self):
        cal = ['2025-01-03', '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09',
               '2025-01-10', '2025-01-13']
        self.assertEqual(label_dates(cal, '2025-01-03'), ('2025-01-06', '2025-01-13'))
        self.assertIsNone(label_dates(cal, '2025-01-06'))

    def test_normalise_orients_by_frozen_direction(self):
        frame = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'amount': [1.0, 1.0, 1.0]},
                             index=['x', 'y', 'z'])
        out = normalise(frame, (('a', 'price', -1, None),))
        self.assertLess(out.loc['z', 'a'], out.loc['x', 'a'])
        self.assertLessEqual(float(out.a.abs().max()), 1.0)

    def test_fold_quarantines_a_disagreeing_duplicate(self):
        frame = pd.DataFrame({'trade_date': ['d', 'd', 'd'],
                              'ts_code': ['A', 'A', 'B'], 'pb': [1.0, 2.0, 3.0]})
        folded, audit = fold_snapshot(frame, ['pb'])
        values = folded.set_index('ts_code').pb
        self.assertTrue(np.isnan(values.loc['A']))
        self.assertEqual(float(values.loc['B']), 3.0)
        self.assertEqual(audit['pb']['unsafe_keys'], 1)

    def test_fold_keeps_an_identical_duplicate(self):
        frame = pd.DataFrame({'trade_date': ['d', 'd'], 'ts_code': ['A', 'A'],
                              'pb': [1.0, 1.0]})
        folded, audit = fold_snapshot(frame, ['pb'])
        self.assertEqual(float(folded.pb.iloc[0]), 1.0)
        self.assertEqual(audit['pb']['unsafe_keys'], 0)

    def test_cluster_refuses_a_single_point(self):
        self.assertIsNone(cluster([0.1]))

    def test_top_selection_breaks_ties_by_amount_then_code(self):
        frame = pd.DataFrame({'amount': [5.0, 9.0, 9.0], 'label': [0.0, 0.1, 0.2]},
                             index=pd.Index(['c1', 'c2', 'c3'], name='ts_code'))
        score = pd.Series([1.0, 1.0, 1.0], index=frame.index)
        self.assertEqual(list(top_selection(frame, score).ts_code), ['c2', 'c3', 'c1'])

    def test_cost_scales_with_replaced_names(self):
        self.assertAlmostEqual(2 * 0.5 * PRIMARY_COST_BPS / 1e4, 0.0015)

    def test_purge_rule_excludes_the_overlapping_predecessor(self):
        cal = ['2025-01-02', '2025-01-03', '2025-01-06', '2025-01-07', '2025-01-08',
               '2025-01-09', '2025-01-10', '2025-01-13', '2025-01-14', '2025-01-15',
               '2025-01-16', '2025-01-17', '2025-01-20', '2025-01-21', '2025-01-22']
        previous_entry, previous_exit = label_dates(cal, '2025-01-03')
        entry, _exit = label_dates(cal, '2025-01-10')
        self.assertEqual(previous_entry, '2025-01-06')
        self.assertEqual(entry, '2025-01-13')
        self.assertFalse(previous_exit < entry)  # purge removes this predecessor
        first_entry, first_exit = label_dates(cal, '2025-01-02')
        self.assertTrue(first_exit < entry)


def tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(ModelCompareTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--reason', default=None,
                        help='required to re-freeze; only plumbing fixes are allowed')
    parser.add_argument('--build-cache', action='store_true')
    parser.add_argument('--tests', action='store_true')
    args = parser.parse_args()
    if args.tests:
        return tests()
    if args.build_cache:
        build_cache()
        return 0
    if args.freeze:
        freeze(args.reason)
        return 0
    run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
