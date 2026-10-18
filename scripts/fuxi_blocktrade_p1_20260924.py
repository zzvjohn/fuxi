# -*- coding: utf-8 -*-
"""P1 of the drought follow-up: give the block-trade family a complete sample, then let
it be judged -- including one hypothesis that P0 could not test.

What changed since P0
---------------------
P0 measured five never-attempted families and returned `CEILING_UNCLEAR` for block trade
for one reason: the store only carried 37 of 87 sections, and the reason was coverage,
not signal. `scripts/fuxi_collect_blocktrade_20260924.py` now pulls the vendor's own
per-session table over every SSE open day, so the family can finally be read on all 87.

P0 also left one thing open. Its pre-registered `bt_share` sign was -1 ("a larger block
share is distribution") and the measured sign came out positive, on 37 sections. Flipping
a sign after seeing the number is not evidence. So this batch states the positive
direction as its own hypothesis, *H1*, before looking at the newly collected sections,
and tests it primarily on the sections P0 never saw.

H1 (stated before the new sections are read)
--------------------------------------------
  A block print is a negotiated transfer, not a market-order imbalance. The seller pays
  a price concession for immediacy or for size; the counterparty is typically an
  institution assembling a position away from the tape. The *concession* is the
  distribution signal, and it is already measured by `bt_premium`. The *share* dimension
  is a different thing: a name whose block turnover is a large fraction of its session
  turnover is receiving concentrated institutional accumulation, and should earn a
  positive forward residual return.
  Therefore H1: sign(bt_share) = +1.

Contamination, disclosed
------------------------
H1's direction was motivated by P0's 37-section reading, so those 37 sections are
in-sample for H1 and cannot test it. The PRIMARY H1 test is restricted to the sections
whose dates carried no rows in `block_trade_full.csv` at all. The all-section reading is
reported as secondary and flagged. The definition of `bt_share` itself is unchanged; only
the registered direction and the sample split are new.

The frozen verdict rule, the pool, the label, the residual design, the clustering rule,
the fold for size, the point-in-time industry map and the signal parquet inputs are all
imported from the P0 module and the frozen framework, never re-implemented. The one
function that has to be re-stated is the block builder, because its source path changes;
equivalence on the shared dates is asserted below rather than assumed.

Usage
-----
  python fuxi_blocktrade_p1_20260924.py tests
  python fuxi_blocktrade_p1_20260924.py freeze
  python fuxi_blocktrade_p1_20260924.py run
"""
import argparse
import collections
import csv
import json
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
import fuxi_external_info_20260924 as ext  # noqa: E402
import fuxi_ic_ceiling_20260924 as ic  # noqa: E402

P0_OUT = ROOT / 'reports/ic_ceiling_20260924'
OUT = ROOT / 'reports/blocktrade_p1_20260924'
BLOCK_DAILY = ROOT / 'data/raw/block_trade_daily.csv'
BLOCK_OLD = ROOT / 'data/raw/block_trade_full.csv'
COLLECTOR = ROOT / 'scripts/fuxi_collect_blocktrade_20260924.py'

TOP_N = ic.TOP_N
MIN_SIDE = ic.MIN_SIDE
MIN_COVERED = ic.MIN_COVERED
MIN_INDUSTRY_N = ic.MIN_INDUSTRY_N
SEGMENTS = ic.SEGMENTS
FORECAST_WINDOW_SESSIONS = ic.FORECAST_WINDOW_SESSIONS
BREAK_EVEN_IC = ic.BREAK_EVEN_IC
MIN_DATES_FOR_BELOW = ic.MIN_DATES_FOR_BELOW
VERDICT_BELOW = ic.VERDICT_BELOW
VERDICT_UNCLEAR = ic.VERDICT_UNCLEAR
VERDICT_ABOVE = ic.VERDICT_ABOVE
VERDICT_UNMEASURABLE = ic.VERDICT_UNMEASURABLE

H1 = {
    'id': 'H1',
    'statement': 'sign(bt_share) = +1: a name whose block turnover is a large share of its '
                 'session turnover is receiving concentrated negotiated accumulation, and '
                 'earns a positive forward residual return',
    'direction_was_p0_minus_one': True,
    'primary_sample': 'sections whose date carried no rows at all in block_trade_full.csv, '
                      'i.e. the sections P0 never read; the other 37 sections are in-sample '
                      'for this hypothesis and are reported as secondary',
    'stated_when': 'before the newly collected sections were read',
}


def p1_family_definitions():
    """P0's definitions with exactly one change: bt_share's registered direction."""
    defs = {name: [dict(d) for d in items] for name, items in ic.FAMILY_DEFS.items()}
    defs['block_trade'] = [
        {'column': 'bt_premium', 'sign': +1,
         'meaning': 'volume-weighted block trade price against that session close; a deep '
                    'discount is selling pressure',
         'carried_from': 'v0.7.67 (P0), unchanged, direction not re-specified'},
        {'column': 'bt_share', 'sign': +1,
         'meaning': 'block trade turnover as a share of the session turnover; large share '
                    'means concentrated negotiated accumulation rather than distribution',
         'carried_from': 'v0.7.68 (P1/H1), direction re-specified with its own mechanism; '
                         'the column definition is byte-identical to P0'},
    ]
    return defs


FAMILY_DEFS = p1_family_definitions()


# ----------------------------------------------------------------- builders (source swap)

def build_block_from(path, stamps, byday):
    """`ic.build_block` with the source path lifted out.

    Kept textually identical to the P0 builder otherwise; `test_builder_equivalence`
    asserts the two produce the same object on the dates both sources cover, so the only
    difference is which sessions exist.
    """
    df = pd.read_csv(path, dtype=str)
    df['day'] = ic._daykey(df.trade_date)
    df = df[df.day.isin(stamps)].copy()
    for col in ('price', 'vol', 'amount'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[np.isfinite(df.price) & np.isfinite(df.vol) & (df.vol > 0)]
    rows_in_window = len(df)
    close_of, amount_of = {}, {}
    for day in df.day.unique():
        table = byday.get(ic._iso(day))
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


def load_block_rows(path, stamps):
    """One source file, restricted to the signal dates and to usable prints.

    `stamps` may arrive in either shape: the protocol carries ISO dates and the raw files
    carry eight-digit keys. Normalising here, plus the guard below, is the fix for defect
    round 2: an ISO key against an eight-digit column matched nothing, and the resulting
    empty frame turned a 0.86 coverage reading into 0.000000 -- a code defect wearing the
    costume of a data finding. The guard refuses that outcome instead of reporting it.
    """
    df = pd.read_csv(path, dtype=str)
    df['day'] = ic._daykey(df.trade_date)
    keys = {str(s).replace('-', '')[:8] for s in stamps}
    available = set(df.day)
    if available and keys:
        spans = min(keys) <= max(available) and max(keys) >= min(available)
    else:
        spans = False
    df = df[df.day.isin(keys)].copy()
    if not len(df) and available and spans:
        raise AssertionError(
            'day-key shape mismatch against %s: %d raw sessions and the requested window '
            'sits inside them, yet nothing matched keys like %s'
            % (Path(path).name, len(available), sorted(keys)[:3]))
    for column in ('price', 'vol', 'amount'):
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df[np.isfinite(df.price) & np.isfinite(df.vol) & (df.vol > 0)]
    return df


def priceable_index(byday):
    """(ISO day, code) pairs the price panel can actually price.

    `build_block` maps each print's `ts_code` to that session's close and drops the row
    when the map is empty, so this set -- not the raw row count -- is the binding
    coverage constraint.
    """
    pairs = set()
    for day, table in byday.items():
        close = table['close']
        for code in table.index[(close > 0).to_numpy()]:
            pairs.add((day, code))
    return pairs


NON_STOCK_PREFIXES = ('11', '12', '13', '15', '16', '18', '50', '51', '52', '56', '58')


def universe_of_price_panel(byday):
    """Every code the equity panel knows about, on any session."""
    codes = set()
    for table in byday.values():
        codes.update(table.index)
    return codes


def audit_source_swap(byday, stamps, universe=None):
    """Three decision-relevant claims, replacing an equality that cannot hold.

    Round 1 asserted "on dates both sources carry, the rows are identical" and failed.
    The diagnosis (kept in defect_round1_subset_and_codes/) is that the old file is stale
    in three separate ways -- missing sessions, truncated name sets on the sessions it
    does carry, and pre-migration Northbound codes that no longer join the price panel --
    so equality was never available.

    The first attempt at the replacement used an invented "99.5% of prints must be
    priceable" threshold and failed at 0.86 on real data. The reason was not a defect:
    `block_trade` covers convertible bonds (11/12/13xxxx), funds and ETFs (15/16/18/50/51/
    52/56/58) and REITs as well as stocks, while the label panel is equities only. So the
    scope is defined from the panel's own universe and the assertion is exact inside it:

      1. inside the universe, EVERY print on a signal date must be priceable that session;
      2. every name-day the OLD file could price must appear in the new file;
      3. on the name-days both can price, the new amount must not be smaller, because the
         old file folded real counterparty legs away.
    """
    stamp_keys = set(stamps)
    old = load_block_rows(BLOCK_OLD, stamp_keys)
    new = load_block_rows(BLOCK_DAILY, stamp_keys)
    codes = universe if universe is not None else universe_of_price_panel(byday)
    pairs = priceable_index(byday)

    def scope_and_price(frame, label):
        in_universe = frame.ts_code.isin(codes).to_numpy()
        keys = list(zip(frame.day.map(ic._iso), frame.ts_code))
        priceable = np.fromiter((k in pairs for k in keys), dtype=bool, count=len(keys))
        outside = frame[~in_universe]
        inside = frame[in_universe]
        inside_priceable = int(priceable[in_universe].sum())
        share = (inside_priceable / len(inside)) if len(inside) else 0.0
        report = {
            label + '_rows_in_window': int(len(frame)),
            label + '_rows_in_universe': int(len(inside)),
            label + '_rows_outside_universe': int(len(outside)),
            label + '_universe_priceable_share': round(float(share), 8),
            label + '_outside_prefixes': outside.ts_code.str.slice(0, 2)
                                                .value_counts().head(12).to_dict(),
        }
        return frame[priceable & in_universe], report

    old_priced, old_rep = scope_and_price(old, 'old')
    new_priced, new_rep = scope_and_price(new, 'new')

    def name_days(frame):
        return frame.groupby(['day', 'ts_code']).amount.sum()

    nd_old, nd_new = name_days(old_priced), name_days(new_priced)
    shared = nd_old.index.intersection(nd_new.index)
    lost_from_old = nd_old.index.difference(nd_new.index)
    gained = nd_new.index.difference(nd_old.index)
    comparison = pd.DataFrame({'old': nd_old.loc[shared], 'new': nd_new.loc[shared]})
    comparison['delta'] = comparison['new'] - comparison['old']
    smaller = comparison[comparison.delta < -1e-9]
    bigger = comparison[comparison.delta > 1e-9]

    report = dict(old_rep)
    report.update(new_rep)
    report.update({
        'universe_codes': len(codes),
        'stamps': len(stamps),
        'old_name_days_priceable': int(len(nd_old)),
        'new_name_days_priceable': int(len(nd_new)),
        'shared_name_days': int(len(shared)),
        'name_days_only_in_old': int(len(lost_from_old)),
        'name_days_only_in_new': int(len(gained)),
        'name_days_only_in_old_examples': [list(map(str, x)) for x in list(lost_from_old)[:8]],
        'cells_new_below_old': int(len(smaller)),
        'cells_new_above_old': int(len(bigger)),
        'amount_recovered_on_shared_cells': float(bigger.delta.sum()) if len(bigger) else 0.0,
        'old_amount_shared': float(comparison.old.sum()) if len(shared) else 0.0,
        'new_amount_shared': float(comparison['new'].sum()) if len(shared) else 0.0,
        'scope_note': 'prints on bonds/funds/REITs are outside the equity label universe '
                      'and are excluded by scope, not dropped silently',
    })
    # 1: inside the equity universe, a print must be priceable on its own session. A store
    #    that answers but cannot be joined would inflate "coverage" while changing nothing.
    assert new_rep['new_universe_priceable_share'] == 1.0, (
        'only %.6f of the in-universe new prints are priceable'
        % new_rep['new_universe_priceable_share'])
    # 2: nothing the old file could price may vanish.
    assert len(lost_from_old) == 0, \
        '%d name-days the old file could price are missing from the new file: %s' % (
            len(lost_from_old), report['name_days_only_in_old_examples'])
    # 3: the new amount must never be smaller on a shared name-day.
    assert len(smaller) == 0, \
        '%d shared name-days lost amount: %s' % (
            len(smaller), [list(map(str, x)) for x in list(smaller.index)[:5]])
    return report


# ------------------------------------------------------------------------------- freeze

def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'protocol already frozen; remove explicitly'
    p0 = read(P0_OUT / 'protocol.json')
    assert p0['batch'] == 'ic_ceiling_20260924'
    assert len(p0['signal_dates']) == 87
    assert BLOCK_DAILY.exists(), 'run the collector first'

    sources = sorted(set(p0['hashes']) | {
        'data/raw/block_trade_daily.csv',
        'scripts/fuxi_blocktrade_p1_20260924.py',
        'scripts/fuxi_collect_blocktrade_20260924.py',
    })
    hashes = {p: digest(ROOT / p) for p in sources}

    protocol = {
        'batch': 'blocktrade_p1_20260924',
        'purpose': 'P1 of the drought follow-up: repair the block-trade family\'s coverage '
                   'to all 87 sections, re-run the frozen ceiling rule on the complete '
                   'sample, and test H1 (the positive bt_share direction) on the sections '
                   'P0 never saw',
        'supersedes': {'batch': 'ic_ceiling_20260924', 'applies_to': 'block_trade only',
                       'other_families': 'P0 verdicts stand; industry, forecast and the '
                                         'benchmark are downgraded to the research feature '
                                         'library (P2) and northbound waits on data'},
        'signals': p0['signals'],
        'signal_dates': p0['signal_dates'],
        'segments': p0['segments'],
        'family_definitions': json.loads(json.dumps(FAMILY_DEFS)),
        'hypotheses': {'H1': H1},
        'sample_split': {
            'rule': 'a section belongs to the primary H1 sample when its date carried no '
                    'rows at all in data/raw/block_trade_full.csv, the file P0 read',
            'note': 'the split is derived from the old file, not chosen, and the membership '
                    'is written to p1_analysis.json',
        },
        'reuse': 'pool, label, residual design, size fold, point-in-time industry map, '
                 'signal parquets, clustering rule and verdict rule are imported from '
                 'fuxi_ic_ceiling_20260924 / fuxi_external_info_20260924; the block builder '
                 'is re-stated only because its source path changes, and equivalence on the '
                 'shared dates is asserted',
        'fill_policy': p0['fill_policy'] + '; a section with fewer than MIN_SIDE covered '
                                           'names is refused, never zero-filled',
        'inference': p0['inference'],
        'criterion': dict(p0['criterion'], primary_for_H1='the new-only sub-sample; the '
                                                          'all-section reading is '
                                                          'contaminated by H1\'s origin'),
        'thresholds': dict(p0['thresholds'], TOP_N=TOP_N, MIN_SIDE=MIN_SIDE,
                           MIN_COVERED=MIN_COVERED, MIN_INDUSTRY_N=MIN_INDUSTRY_N,
                           SEGMENTS=SEGMENTS),
        'disclosure_rule': p0['disclosure_rule'],
        'hashes': hashes,
    }
    write(OUT / 'protocol.json', protocol)
    print('frozen blocktrade_p1_20260924: %d hashes' % len(hashes))
    print('block_trade definitions:', [(d['column'], d['sign'])
                                       for d in protocol['family_definitions']['block_trade']])


# ---------------------------------------------------------------------------------- run

def run():
    protocol = read(OUT / 'protocol.json')
    for path, expected in protocol['hashes'].items():
        assert digest(ROOT / path) == expected, path
    assert not (OUT / 'summary.json').exists()

    # --- the only re-wiring: point the P0 machinery at the new protocol and the new file
    patched_builder = lambda stamps, byday: build_block_from(BLOCK_DAILY, stamps, byday)
    ic.OUT = OUT
    ic.FAMILY_DEFS = protocol['family_definitions']
    ic.build_block = patched_builder
    assert ic.build_block is patched_builder, 'builder patch did not take'
    assert ic.OUT == OUT, 'protocol path patch did not take'
    assert ic.FAMILY_DEFS['block_trade'][1]['sign'] == +1, 'sign patch did not take'

    with RunContract(parameters={'protocol_sha256': digest(OUT / 'protocol.json'),
                                 'code_data_hashes': protocol['hashes'],
                                 'break_even_ic': BREAK_EVEN_IC,
                                 'min_dates_for_below': MIN_DATES_FOR_BELOW,
                                 'hypothesis': 'H1'},
                     run_id='btp1_' + datetime.now().strftime('%H%M%S_%f'),
                     runs_dir=OUT / 'runs', lock_path=OUT / 'btp1.lock'):
        rows, drops, geometry, timing, coverage, sources = ic.rebuild_sections()
        for path, expected in protocol['hashes'].items():
            assert digest(ROOT / path) == expected, path

        summaries = ic.summarise(rows, protocol, coverage)
        ceilings = {}
        for family, defs in protocol['family_definitions'].items():
            factor = family + '__ceiling'
            entry = summaries.get(factor)
            if entry is None:
                ceilings[family] = {'verdict': VERDICT_UNMEASURABLE,
                                    'reason': 'no ceiling vector produced',
                                    'definitions': [d['column'] for d in defs]}
                continue
            stats = entry['clustered_ic_resid_existing']
            dates = entry['clustered_ic_resid_existing_dates']
            verdict, reason = ic.verdict_of(stats, dates)
            if verdict == VERDICT_ABOVE and not entry['segment_sign_consistent']:
                verdict, reason = VERDICT_UNCLEAR, 'point magnitude at break-even but the ' \
                                                   'sign flips across segments'
            ceilings[family] = {
                'verdict': verdict, 'reason': reason,
                'definitions': [d['column'] for d in defs],
                'signs': [d['sign'] for d in defs],
                'dates_measured': dates,
                'coverage_mean': entry['coverage_mean'],
                'coverage_min': entry['coverage_min'],
                'ceiling_ic_resid_existing': stats,
                'ceiling_ic_raw': entry['clustered_ic_raw'],
                'ceiling_ic_resid': entry['clustered_ic_resid'],
                'ceiling_pearson_resid_existing': entry['clustered_pearson_resid_existing'],
                'ceiling_pearson_raw': entry['clustered_pearson_raw'],
                'raw_sensitivity': {
                    'note': 'the frozen verdict uses the residual reading; the break-even '
                            'arithmetic was derived on a RAW rank IC, so this is the '
                            'directly comparable reading, reported as a sensitivity',
                    'stats': entry['clustered_ic_raw'],
                    'dates_measured': entry['clustered_ic_raw_dates'],
                    'magnitude': ic._magnitude(entry['clustered_ic_raw']),
                    'upper_bound_on_magnitude': ic._bound(entry['clustered_ic_raw']),
                    'verdict_if_raw_were_primary': ic.verdict_of(
                        entry['clustered_ic_raw'], entry['clustered_ic_raw_dates'])[0]},
                'segment_sign_consistent': entry['segment_sign_consistent'],
                'per_definition': {d['column']: {
                    'sign_expected': d['sign'],
                    'sign_observed': ic._sign(summaries.get(d['column'], {}).get(
                        'clustered_ic_resid_existing', {}).get('mean')),
                    'clustered_ic_resid_existing': summaries.get(d['column'], {}).get(
                        'clustered_ic_resid_existing'),
                    'clustered_ic_raw': summaries.get(d['column'], {}).get(
                        'clustered_ic_raw'),
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
                for family in protocol['family_definitions']},
            'design_rank_deficit_total': int(sum(
                (g['base_columns'] - g['base_rank']) + (g['full_columns'] - g['full_rank'])
                for g in geometry)),
            'refused_thin_sections': int(sum(
                1 for r in rows if r.get('status') == 'covered_side_too_thin')),
            'parity_assertions_passed': True,
        }

        # ---- the part P1 adds: audit the source swap, split the sample, test H1
        p1 = p1_analysis(protocol, rows, summaries)

        write(OUT / 'summary.json', {
            'signal_dates': protocol['signal_dates'], 'segments': protocol['segments'],
            'break_even_ic': BREAK_EVEN_IC, 'thresholds': protocol['thresholds'],
            'hypotheses': protocol['hypotheses'],
            'summaries': summaries, 'ceilings': ceilings,
            'timing': ic.timing_block(timing, protocol), 'quality': quality,
            'p1': p1,
            'claims_not_made': [
                'labels are uncharged adjusted-open comparisons, not cash NAV',
                'no economic verdict for any family; a ceiling is not a strategy',
                'a CEILING_BELOW verdict is a statement about our data and this window '
                'under these definitions, not a claim that the family carries no information',
                'H1\'s direction was motivated by P0\'s 37 sections, so the all-section H1 '
                'reading is contaminated; only the new-only sub-sample is a test',
                'research-exposed window; no R5 unknown observations',
                'the coverage repair is a data-store statement, not a factor statement',
            ],
            'full_stage2_accepted': False, 'formal_eligibility': False})
        ext.write_csv(OUT / 'per_section.csv', rows)
        ext.write_csv(OUT / 'drops.csv', drops)
        ext.write_csv(OUT / 'geometry.csv', geometry)
        ext.write_csv(OUT / 'timing.csv', timing)
        ext.write_csv(OUT / 'coverage.csv', coverage)
        print('done: rows=%d valid=%d drops=%d' % (len(rows), len(valid), len(drops)))
        print('ceilings:', {k: v['verdict'] for k, v in sorted(ceilings.items())})
        print('block_trade coverage: %s' % json.dumps(
            p1['coverage'], ensure_ascii=False)[:400])
        print('H1 primary (new-only): %s' % json.dumps(
            p1['H1']['primary']['bt_share'], ensure_ascii=False))
        print('source swap audit: %s' % json.dumps(p1['source_swap'], ensure_ascii=False))


def p1_analysis(protocol, rows, summaries):
    """Coverage-repair evidence, the sample split, and H1's primary/secondary readings."""
    stamps = sorted(protocol['signal_dates'])
    calendars = ext.public_calendar()
    price, _ = load_panel('data/raw/daily_prices.csv',
                          ['ts_code', 'trade_date', 'open', 'close', 'amount'],
                          lower='20240801')
    adj, _ = load_panel('data/raw/adj_factor_daily.csv',
                        ['ts_code', 'trade_date', 'adj_factor'], lower='20240801')
    merged = price.merge(adj, on=['trade_date', 'ts_code'], validate='one_to_one', how='left')
    merged['adj_open'] = merged.open * merged.adj_factor
    byday = {d: g.set_index('ts_code') for d, g in merged.groupby('trade_date')}

    swap = audit_source_swap(byday, stamps)

    # The split is READ OFF P0's own output, not chosen here: a section belongs to the
    # primary sample when P0 never obtained a reading for it.
    p0_rows = [r for r in csv.DictReader(
        open(P0_OUT / 'per_section.csv', encoding='utf-8-sig'))
        if r['family'] == 'block_trade']
    p0_read = {r['date'] for r in p0_rows
               if r['factor'] == 'block_trade__ceiling'
               and r['status'] == 'valid_descriptive'}
    assert p0_read <= set(stamps), 'P0 read a date outside the frozen section list'
    primary_days = sorted(d for d in stamps if d not in p0_read)
    polluted_days = sorted(d for d in stamps if d in p0_read)

    block_rows = [r for r in rows if r['family'] == 'block_trade']
    new_valid = {r['date'] for r in block_rows if r['factor'] == 'bt_premium'
                 and r.get('status') == 'valid_descriptive'}
    new_valid &= {r['date'] for r in block_rows if r['factor'] == 'bt_share'
                  and r.get('status') == 'valid_descriptive'}
    coverage = {
        'sections_total': len(stamps),
        'p0_sections_read': len(p0_read),
        'primary_sample_sections': len(primary_days),
        'contaminated_sections': len(polluted_days),
        'sections_now_usable_both_definitions': len(new_valid),
        'new_rows_in_window': swap['new_rows_in_window'],
        'old_rows_in_window': swap['old_rows_in_window'],
        'old_rows_in_universe': swap['old_rows_in_universe'],
        'new_rows_in_universe': swap['new_rows_in_universe'],
        'old_rows_outside_universe': swap['old_rows_outside_universe'],
        'new_rows_outside_universe': swap['new_rows_outside_universe'],
        'old_universe_priceable_share': swap['old_universe_priceable_share'],
        'new_universe_priceable_share': swap['new_universe_priceable_share'],
        'universe_codes': swap['universe_codes'],
        'old_name_days_priceable': swap['old_name_days_priceable'],
        'new_name_days_priceable': swap['new_name_days_priceable'],
        'primary_first_last': [primary_days[0], primary_days[-1]] if primary_days else None,
        'definition': 'primary = the sections P0 obtained no reading for; usable = at least '
                      'MIN_SIDE names covered by BOTH definitions; a section below that is '
                      'refused by evaluate_factor, never filled',
    }

    def readings(factor, dates):
        keep = set(dates)
        block = [r for r in rows if r['factor'] == factor
                 and r.get('status') == 'valid_descriptive' and r['date'] in keep]
        resid, n_resid = ic._by_date_one(block, 'ic_resid_existing')
        raw, n_raw = ic._by_date_one(block, 'ic_raw')
        pearson, n_pearson = ic._by_date_one(block, 'pearson_resid_existing')
        return {'sections': len({r['date'] for r in block}), 'dates': n_resid,
                'ic_resid_existing': resid, 'ic_raw': raw,
                'raw_dates': n_raw, 'pearson_resid_existing': pearson,
                'pearson_dates': n_pearson,
                'verdict_if_raw_were_primary': ic.verdict_of(raw, n_raw)[0],
                'verdict_resid_frozen_rule': ic.verdict_of(resid, n_resid)[0]}

    h1 = {'statement': protocol['hypotheses']['H1'],
          'primary': {
              'label': 'the %d sections P0 obtained no reading for' % len(primary_days),
              'dates': len(primary_days),
              'bt_share': readings('bt_share', primary_days),
              'bt_premium': readings('bt_premium', primary_days),
              'ceiling': readings('block_trade__ceiling', primary_days)},
          'secondary_all_sections': {
              'label': 'all %d sections; contaminated for H1 because its direction came '
                       'from the %d P0 read' % (len(stamps), len(p0_read)),
              'bt_share': readings('bt_share', stamps),
              'ceiling': readings('block_trade__ceiling', stamps)},
          'segment_signs_primary': []}
    for low, high in protocol['segments']:
        block = [r for r in rows if r['factor'] == 'bt_share'
                 and r.get('status') == 'valid_descriptive' and low <= r['date'] <= high
                 and r['date'] in set(primary_days)]
        stats, dates = ic._by_date_one(block, 'ic_resid_existing')
        h1['segment_signs_primary'].append({'window': [low, high], 'dates': dates,
                                            'ic_resid_existing': stats})
    means = [s['ic_resid_existing']['mean'] for s in h1['segment_signs_primary']
             if s['ic_resid_existing']['mean'] is not None]
    h1['primary_segment_sign_consistent'] = (
        None if not means else bool(all(v > 0 for v in means) or all(v < 0 for v in means)))
    h1['primary_verdict'] = _h1_verdict(h1['primary']['bt_share'],
                                        h1['primary']['ceiling'], primary_days)
    return {'coverage': coverage, 'source_swap': swap, 'H1': h1,
            'primary_sample_dates': primary_days, 'reused_p0_sections': sorted(p0_read)}


def _h1_verdict(share, ceiling, primary_days):
    """H1 is a directional claim, so it is read on the registered sign, then bounded.

    `supported` needs the registered sign in the primary sample and the two aggregated
    readings to agree in sign. `refuted` needs the opposite sign. Anything with an interval
    that straddles zero, or fewer than MIN_DATES_FOR_BELOW dates, is `unresolved` -- the
    same refusal-to-overclaim that P0 applied to families.
    """
    stats = share['ic_resid_existing']
    if not stats.get('n') or stats.get('mean') is None or len(primary_days) < MIN_SIDE:
        return {'verdict': 'unresolved', 'reason': 'no usable primary sample'}
    mean = stats['mean']
    low, high = stats.get('ci_low'), stats.get('ci_high')
    if low is None or high is None or low <= 0 <= high:
        return {'verdict': 'unresolved',
                'reason': 'cluster interval on the primary sample straddles zero',
                'mean': mean, 'interval': [low, high], 'dates': stats['n']}
    direction = 'supported' if mean > 0 else 'refuted'
    ceiling_mean = ceiling['ic_resid_existing'].get('mean')
    disagree = (ceiling_mean is not None and (ceiling_mean > 0) != (mean > 0))
    return {'verdict': direction, 'mean': mean, 'interval': [low, high], 'dates': stats['n'],
            'direction_registered': +1, 'ceiling_same_sign': not disagree,
            'reason': 'registered sign in the primary sample with an interval away from '
                      'zero; magnitude is not a promotion'}


# ------------------------------------------------------------------------------- tests

def _stub_byday(days=('2025-03-17', '2025-03-18', '2025-03-19')):
    """A short slice of the real panel, so the tests read real vendor rows."""
    price, _ = load_panel('data/raw/daily_prices.csv',
                          ['ts_code', 'trade_date', 'open', 'close', 'amount'],
                          lower='20250301')
    adj, _ = load_panel('data/raw/adj_factor_daily.csv',
                        ['ts_code', 'trade_date', 'adj_factor'], lower='20250301')
    merged = price.merge(adj, on=['trade_date', 'ts_code'], validate='one_to_one', how='left')
    merged = merged[merged.trade_date.isin(list(days))]
    return {d: g.set_index('ts_code') for d, g in merged.groupby('trade_date')}


def tests():
    class BlockTradeP1(unittest.TestCase):
        def test_builder_equivalence(self):
            old, _ = ic.build_block({'20250318'}, _stub_byday())
            self.assertTrue(old, 'the old file carries no row on 20250318; pick another day')
            new, _ = build_block_from(BLOCK_OLD, {'20250318'}, _stub_byday())
            self.assertEqual(sorted(old), sorted(new))
            for day in old:
                frame_old, frame_new = old[day], new[day].loc[old[day].index]
                for column in ('bt_premium', 'bt_share'):
                    np.testing.assert_allclose(frame_old[column].to_numpy(dtype=float),
                                               frame_new[column].to_numpy(dtype=float),
                                               rtol=0, atol=1e-12)

        def test_swap_audit_refuses_unpriceable_new_rows(self):
            # ISO stamps on purpose: that is the production shape, and passing the
            # eight-digit form here is exactly what let defect round 2 through.
            days = {'2025-03-17', '2025-03-18', '2025-03-19'}
            stub = _stub_byday()
            universe = universe_of_price_panel(stub)
            report = audit_source_swap(stub, days, universe=universe)
            self.assertEqual(report['new_universe_priceable_share'], 1.0)
            self.assertEqual(report['name_days_only_in_old'], 0)
            trimmed = {k: v for k, v in stub.items() if k != '2025-03-18'}
            with self.assertRaises(AssertionError):
                audit_source_swap(trimmed, days, universe=universe)

        def test_swap_audit_scopes_out_non_stocks(self):
            days = {'2025-03-17', '2025-03-18', '2025-03-19'}
            report = audit_source_swap(_stub_byday(), days)
            self.assertGreater(report['new_rows_outside_universe'], 0)
            self.assertTrue(all(str(p)[:2] in NON_STOCK_PREFIXES
                                for p in report['new_outside_prefixes']))

        def test_swap_audit_accepts_both_day_shapes(self):
            days_iso = {'2025-03-17', '2025-03-18', '2025-03-19'}
            days_digits = {d.replace('-', '') for d in days_iso}
            a = load_block_rows(BLOCK_DAILY, days_iso)
            b = load_block_rows(BLOCK_DAILY, days_digits)
            self.assertGreater(len(a), 0)
            self.assertEqual(len(a), len(b))
            self.assertTrue(set(a.day) == set(b.day))

        def test_swap_audit_reports_recovered_legs(self):
            days = {'2025-03-17', '2025-03-18', '2025-03-19'}
            report = audit_source_swap(_stub_byday(), days)
            self.assertEqual(report['cells_new_below_old'], 0)
            self.assertGreater(report['new_rows_in_window'], report['old_rows_in_window'])

        def test_p1_only_changes_one_sign(self):
            p0 = ic.FAMILY_DEFS
            for family in p0:
                for left, right in zip(p0[family], FAMILY_DEFS[family]):
                    self.assertEqual(left['column'], right['column'])
                    if family == 'block_trade' and left['column'] == 'bt_share':
                        self.assertEqual((left['sign'], right['sign']), (-1, +1))
                    else:
                        self.assertEqual(left['sign'], right['sign'])

        def test_h1_verdict_requires_an_interval_away_from_zero(self):
            share = {'ic_resid_existing': {'n': 40, 'mean': 0.02, 'ci_low': -0.01,
                                           'ci_high': 0.05}}
            out = _h1_verdict(share, {'ic_resid_existing': {'mean': 0.01}}, list(range(40)))
            self.assertEqual(out['verdict'], 'unresolved')
            share['ic_resid_existing'] = {'n': 40, 'mean': 0.02, 'ci_low': 0.005,
                                          'ci_high': 0.035}
            out = _h1_verdict(share, {'ic_resid_existing': {'mean': 0.01}}, list(range(40)))
            self.assertEqual(out['verdict'], 'supported')
            share['ic_resid_existing'] = {'n': 40, 'mean': -0.02, 'ci_low': -0.035,
                                          'ci_high': -0.005}
            out = _h1_verdict(share, {'ic_resid_existing': {'mean': -0.01}}, list(range(40)))
            self.assertEqual(out['verdict'], 'refuted')

        def test_h1_flags_a_ceiling_that_disagrees(self):
            share = {'ic_resid_existing': {'n': 40, 'mean': 0.02, 'ci_low': 0.005,
                                           'ci_high': 0.035}}
            out = _h1_verdict(share, {'ic_resid_existing': {'mean': -0.02}}, list(range(40)))
            self.assertEqual(out['verdict'], 'supported')
            self.assertFalse(out['ceiling_same_sign'])

    suite = unittest.TestLoader().loadTestsFromTestCase(BlockTradeP1)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'tests.json', {'tests': result.testsRun,
                               'failures': len(result.failures),
                               'errors': len(result.errors),
                               'ok': result.wasSuccessful()})
    assert result.wasSuccessful(), 'tests failed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['tests', 'freeze', 'run'])
    args = parser.parse_args()
    if args.command == 'tests':
        tests()
    elif args.command == 'freeze':
        freeze()
    else:
        run()


if __name__ == '__main__':
    main()
