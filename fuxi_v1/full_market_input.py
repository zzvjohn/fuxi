"""Hash-bound full A-share columnar inputs from existing immutable vendor panels.

Input sharding is an IO concern only. Every declared member remains in the final
arrays, and cross-sectional research uses the entire per-session membership.
Current vendor versions and publication-hour proxies are never PIT assurance.
"""
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
import json
import re

from .contracts import digest, require
from .datasets import execution_profile, file_hash, read_calendar
from .history import iso, jq_code, name_state
from .pit_universe import build_pit_cohort, read_lifecycle


FIELDS = ('open', 'high', 'low', 'close', 'volume', 'amount', 'turnover_rate',
          'total_shares', 'float_shares', 'free_shares')
CLASSIFICATIONS = (
    'missing_evidence', 'unchanged_reference_and_factor',
    'tiny_drift_consistent_reference_research_only', 'factor_or_reference_conflict',
    'published_corporate_action', 'unchanged_factor_on_recorded_full_halt',
    'initial_observation_no_prior_holdings', 'constant_factor_corroborated_by_stk_factor')
MARK_SOURCES = ('observed_close', 'declared_frozen_last_close_on_vendor_halt',
                'declared_ex_cash_reference_on_vendor_halt',
                'halt_action_reference_not_identifiable', 'observed_close_on_vendor_partial_halt')
NAME_STATES = ('normal', 'st', 'unknown', 'not_yet_published', 'conflicting')
REASONS = ('not_listed_or_delisted', 'listing_age', 'name_not_normal',
           'halt_or_tradability_unknown', 'quote_missing', 'corporate_action_unknown')


def save_new(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write((json.dumps(body, ensure_ascii=False, sort_keys=True,
                                 separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8'))


def save_array(path, values):
    import numpy as np
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.save(handle, values, allow_pickle=False)
    return {'path': str(path), 'sha256': file_hash(path),
            'dtype': values.dtype.str, 'shape': list(values.shape)}


def verify_catalog(root):
    """Verify the parent metadata binding of a declared derivative overlay."""
    root = Path(root).resolve()
    manifest = json.loads((root / 'CATALOG.json').read_text(encoding='utf-8'))
    metadata = json.loads((root / 'CATALOG_METADATA.json').read_text(encoding='utf-8'))
    require(manifest['manifest_hash'] == digest({k: v for k, v in manifest.items() if k != 'manifest_hash'}),
            'full market catalog manifest changed')
    require(metadata['metadata_hash'] == digest({k: v for k, v in metadata.items() if k != 'metadata_hash'}),
            'full market catalog metadata changed')
    parent_hash = manifest.get('parent_catalog_hash')
    require(metadata['manifest_hash'] == manifest['manifest_hash'] or
            parent_hash and metadata['manifest_hash'] == parent_hash,
            'full market metadata is not bound to catalog or declared parent')
    for filename, expected in metadata['files'].items():
        require(file_hash(root / filename) == expected, 'metadata file changed: ' + filename)
    evidence = {'catalog_path': str(root / 'CATALOG.json'),
                'catalog_sha256': file_hash(root / 'CATALOG.json'),
                'manifest_hash': manifest['manifest_hash'],
                'metadata_hash': metadata['metadata_hash'], 'parent_metadata_binding': parent_hash,
                'historical_revision_archive_certified': False}
    if parent_hash:
        # The derivative freezes only explicitly appended queries. The parent
        # path is project-owned and its partitions are verified below.
        project = Path(__file__).resolve().parents[1]
        parent = project / 'data/fuxi_v1/supplements/20261003/catalog_v3'
        parent_manifest = json.loads((parent / 'CATALOG.json').read_text(encoding='utf-8'))
        parent_metadata = json.loads((parent / 'CATALOG_METADATA.json').read_text(encoding='utf-8'))
        require(parent_manifest['manifest_hash'] == parent_hash and
                digest({k: v for k, v in parent_manifest.items() if k != 'manifest_hash'}) == parent_hash,
                'full market overlay parent changed')
        require(parent_manifest['tables'] == manifest['tables'] and parent_metadata == metadata,
                'overlay changed inherited partitions or metadata')
        for api, old_queries in parent_manifest['queries'].items():
            require(all(row in manifest['queries'].get(api, []) for row in old_queries),
                    'overlay removed inherited query: ' + api)
        evidence['parent_catalog_path'] = str(parent / 'CATALOG.json')
        evidence['parent_catalog_sha256'] = file_hash(parent / 'CATALOG.json')
    part_count = 0
    for api, parts in manifest['tables'].items():
        require(parts, 'empty full market channel: ' + api)
        for part in parts:
            require(part['source_hash'] == manifest['sources'][api]['sha256'] and
                    file_hash(root / part['file']) == part['sha256'], 'changed partition: ' + api)
            part_count += 1
    evidence.update(verified_partitions=part_count, raw_files_consumed=False,
                    vintage=manifest['vintage'])
    return manifest, metadata, evidence


def derive_market_arrays(sessions, codes, members, numbers, masks, names, terms, halt_records,
                         *, listing_age_calendar_days=60, crosscheck=None, source_hashes=None):
    """Equivalent daily proxy/barrier semantics, without millions of bar dicts."""
    import numpy as np
    from datetime import date
    from .action_accounting import SUPPORTED
    from .action_evidence import constant_factor_corroborated
    from .halt_evidence import partial_halt_mark
    require(sessions and codes and len(codes) == len(set(codes)), 'empty or repeated full market identities')
    shape = (len(sessions), len(codes))
    ci = {code: j for j, code in enumerate(codes)}
    di = {day: i for i, day in enumerate(sessions)}
    instruments = [jq_code(code) for code in codes]
    short_days = [day.replace('-', '') for day in sessions]
    active = np.zeros(shape, bool)
    age = active.copy()
    by_member = {member['ts_code']: member for member in members}
    dates = np.array([date.fromisoformat(day).toordinal() for day in sessions])
    for j, code in enumerate(codes):
        row = by_member[code]
        listed = iso(row['list_date'])
        if code.endswith('.BJ'):
            listed = max(listed, '2021-11-15')
        delisted = iso(row.get('delist_date'))
        active[:, j] = [(listed <= day and (not delisted or day < delisted)) for day in sessions]
        age[:, j] = dates - date.fromisoformat(listed).toordinal() >= listing_age_calendar_days
    name_codes = np.full(shape, NAME_STATES.index('unknown'), np.uint8)
    for code, records in names.items():
        if code not in ci:
            continue
        # interval state must honour effective start, publication and conflicts;
        # it is never backfilled from today's name or ST status.
        j = ci[code]
        for i, day in enumerate(sessions):
            name_codes[i, j] = NAME_STATES.index(name_state(records, day)[0])
    full_halt = np.zeros(shape, bool)
    partial_halt = full_halt.copy()
    recorded_halt_status = full_halt.copy()
    for (code, day), rows in halt_records.items():
        if code in ci and day in di:
            i, j = di[day], ci[code]
            recorded_halt_status[i, j] = bool(rows)
            full_halt[i, j] = any(r['suspend_type'] == 'S' and not r.get('suspend_timing') for r in rows)
            partial_halt[i, j] = any(r['suspend_type'] == 'S' and bool(r.get('suspend_timing')) for r in rows)
    columns = {f: np.array(numbers['vol' if f == 'volume' else f], dtype=np.float64, copy=True)
               for f in ('open', 'high', 'low', 'close', 'volume', 'amount')}
    columns['volume'] *= 100
    columns['amount'] *= 1000
    for target, source in [('turnover_rate', 'turnover_rate'), ('total_shares', 'total_share'),
                           ('float_shares', 'float_share'), ('free_shares', 'free_share')]:
        columns[target] = np.array(numbers[source], dtype=np.float64, copy=True)
        columns[target] *= .01 if target == 'turnover_rate' else 10000
    # NEEQ quotations before a BJ A-share listing are not A-share feature history.
    for j, code in enumerate(codes):
        if code.endswith('.BJ'):
            before = np.asarray([day < iso(by_member[code]['list_date']) or day < '2021-11-15' for day in sessions])
            for values in columns.values():
                values[before, j] = np.nan
    conflict = full_halt & np.isfinite(columns['volume']) & (columns['volume'] > 0)
    paused = np.full(shape, -1, np.int8)
    paused[columns['volume'] > 0] = 0
    paused[recorded_halt_status] = 0
    paused[full_halt | partial_halt] = 1
    paused[conflict] = -1
    known = np.zeros(shape, bool)
    category = np.zeros(shape, np.uint8)
    mark_source = np.zeros(shape, np.uint8)
    marks = columns['close'].copy()
    sizing = np.full(shape, np.nan)
    barriers = np.full(shape, -1, np.int32)
    latest_barrier = np.full(len(codes), -1, np.int32)
    last_price = np.full(len(codes), np.nan)
    last_adj = last_price.copy()
    last_day = [None] * len(codes)
    seen_quote = np.zeros(len(codes), bool)
    sparse = {}
    terms_at = defaultdict(list)
    for term in terms:
        if term['instrument'] in instruments and term['ex_date'] in di:
            terms_at[di[term['ex_date']]].append(term)
    inst_ci = {code: j for j, code in enumerate(instruments)}
    accepted_crosscheck = 0
    for i, day in enumerate(sessions):
        current = numbers['adj_factor'][i]
        close, reference = columns['close'][i], numbers['pre_close'][i]
        have = np.isfinite(current) & np.isfinite(last_adj) & np.isfinite(last_price) & np.isfinite(reference)
        equal = np.abs(reference - last_price) < .000001
        same = have & equal & (current == last_adj)
        with np.errstate(invalid='ignore', divide='ignore'):
            # P-20261015-011: vendor adj-factor revisions drift up to ~0.76% relative on
            # days whose pre_close chain stays continuous (no ex-event, no price gap).
            # A 1% relative bound separates this declared research sensitivity from real
            # event gaps (which always break the reference chain and stay conflicts).
            tiny = have & equal & (np.abs(current / last_adj - 1) <= .01)
        category[i, have] = CLASSIFICATIONS.index('factor_or_reference_conflict')
        category[i, tiny] = CLASSIFICATIONS.index('tiny_drift_consistent_reference_research_only')
        category[i, same] = CLASSIFICATIONS.index('unchanged_reference_and_factor')
        freeze = full_halt[i] & ~conflict[i] & ~np.isfinite(close) & np.isfinite(last_price)
        marks[i, freeze] = last_price[freeze]
        mark_source[i, freeze] = MARK_SOURCES.index('declared_frozen_last_close_on_vendor_halt')
        halt_known = freeze & (current == last_adj) & np.isfinite(current)
        category[i, halt_known] = CLASSIFICATIONS.index('unchanged_factor_on_recorded_full_halt')
        initial = (i == 0) | (active[i] & ~seen_quote)
        initial = initial & np.isfinite(current) & np.isfinite(close)
        category[i, initial] = CLASSIFICATIONS.index('initial_observation_no_prior_holdings')
        if crosscheck:
            for j in np.flatnonzero(category[i] == CLASSIFICATIONS.index('factor_or_reference_conflict')):
                before = crosscheck.get((codes[j], last_day[j]))
                after = crosscheck.get((codes[j], short_days[i]))
                values = [last_price[j], reference[j], close[j]]
                if constant_factor_corroborated(before, after, *[float(x) for x in values]):
                    category[i, j] = CLASSIFICATIONS.index('constant_factor_corroborated_by_stk_factor')
                    accepted_crosscheck += 1
        for term in terms_at[i]:
            j = inst_ci[term['instrument']]
            category[i, j] = CLASSIFICATIONS.index('published_corporate_action')
            key = day + '|' + term['instrument']
            patch = sparse.setdefault(key, {})
            patch['action_ids'] = [term['evidence_id']]
            if term['status'] in SUPPORTED:
                sizing[i, j] = reference[j]
                patch['sizing_reference_evidence'] = digest([None if not np.isfinite(reference[j]) else float(reference[j]), term])
            if freeze[j]:
                cash = term.get('gross_cash_per_share')
                supported = (term['status'] == 'cash_terms_supported' and term.get('share_distribution') == 0
                             and isinstance(cash, (int, float)) and cash >= 0)
                adjusted = Decimal(str(float(last_price[j]))) - Decimal(str(cash)) if supported else None
                mark = float(adjusted) if adjusted is not None and adjusted > 0 else None
                proof = {'session': day, 'instrument': term['instrument'],
                         'previous_declared_mark': float(last_price[j]), 'gross_cash_per_share': cash,
                         'action_evidence_id': term['evidence_id'], 'reference_mark': mark,
                         'status': 'declared_cash_reference' if mark is not None else 'data_insufficient',
                         'scope': 'declared halted cash-ex valuation; not observed quote or independently certified price'}
                patch['halt_reference'] = proof
                marks[i, j] = np.nan if mark is None else mark
                last_price[j] = marks[i, j]
                mark_source[i, j] = MARK_SOURCES.index('declared_ex_cash_reference_on_vendor_halt' if mark is not None else 'halt_action_reference_not_identifiable')
                if mark is not None:
                    sizing[i, j] = mark
                    patch['sizing_reference_evidence'] = digest(proof)
        known_codes = [CLASSIFICATIONS.index(x) for x in (
            'unchanged_reference_and_factor', 'tiny_drift_consistent_reference_research_only',
            'unchanged_factor_on_recorded_full_halt', 'initial_observation_no_prior_holdings',
            'constant_factor_corroborated_by_stk_factor')]
        known[i] = active[i] & np.isin(category[i], known_codes)
        for term in terms_at[i]:
            j = inst_ci[term['instrument']]
            known[i, j] = bool(active[i, j] and term['status'] in SUPPORTED)
        barrier_safe = [value for value in known_codes
                        if value != CLASSIFICATIONS.index('initial_observation_no_prior_holdings')]
        latest_barrier[~np.isin(category[i], barrier_safe)] = i
        barriers[i] = latest_barrier
        for j in np.flatnonzero(partial_halt[i] & ~full_halt[i] & (paused[i] == 1)):
            source_hashes = source_hashes or {}
            proof = partial_halt_mark(halt_records.get((codes[j], day), []), instruments[j], day,
                                      float(close[j]), float(columns['volume'][i, j]),
                                      source_hashes.get('suspend_d.csv'), source_hashes.get('daily_prices.csv'))
            if proof is not None:
                sparse.setdefault(day + '|' + instruments[j], {})['partial_halt_mark'] = proof
                mark_source[i, j] = MARK_SOURCES.index('observed_close_on_vendor_partial_halt')
        observed = np.isfinite(close)
        last_price[observed] = close[observed]
        seen_quote[observed] = True
        for j in np.flatnonzero(observed):
            last_day[j] = short_days[i]
        have_adj = np.isfinite(current) & active[i]
        last_adj[have_adj] = current[have_adj]
    reasons = np.zeros(shape, np.uint16)
    failing = (~active, ~age, name_codes != NAME_STATES.index('normal'), paused != 0,
               ~np.isfinite(columns['close']), ~known)
    for bit, mask in enumerate(failing):
        reasons[mask] |= 1 << bit
    membership = reasons == 0
    require((membership.sum(axis=1) >= 10).all(), 'full market auditable universe below 10')
    market_numbers = {f: columns[f] for f in ('open', 'high', 'low', 'volume')}
    market_numbers.update(close=marks, up_limit=numbers['up_limit'], down_limit=numbers['down_limit'],
                          sizing_reference_close=sizing)
    return {'columns': columns, 'barriers': barriers, 'membership': membership,
            'numeric': market_numbers, 'status': {'paused': paused, 'corporate_action_known': known},
            'categorical': {'event_coverage': category, 'mark_source': mark_source}, 'sparse_bars': sparse,
            'active': active, 'full_halt': full_halt, 'partial_halt': partial_halt,
            'name_states': name_codes, 'reason_masks': reasons,
            'crosscheck_accepted_cells': accepted_crosscheck}


def build_full_market_input(catalog_root, output, *, start='2024-01-02', end='2026-09-24',
                            warmup=20, horizon=5, listing_age_calendar_days=60,
                            latest_industry_path=None, action_term_exemptions=None,
                            fractional_share_policy='block'):
    import numpy as np
    import pandas as pd
    import time
    from .panel_audit import accumulate_panel, catalog_frames
    from .market_catalog import load_table, factor_corroboration, TABLE_APIS
    from .historical_adapter import dividend_terms
    from .action_evidence import resolve_action_terms
    started = time.monotonic()
    root, out = Path(catalog_root).resolve(), Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / 'input.json').exists(), 'use a new full market build output')
    manifest, metadata, source = verify_catalog(root)
    sessions, calendar_audit = read_calendar(root / 'trade_calendar.csv', start, end)
    require(len(sessions) > warmup + horizon + 10, 'full market calendar too short')
    life = read_lifecycle(root)
    codes, declaration, neutrality = build_pit_cohort(life, sessions,
        salt='fuxi-full-a-census-20261009', rule_kind='all_a_share_census')
    members = [dict(ts_code=code, **life[code]) for code in codes]
    instruments = [jq_code(code) for code in codes]
    save_new(out / 'MEMBERSHIP.json', {'declaration': declaration, 'members': members,
        'sessions': sessions, 'neutrality_audit': neutrality, 'no_sample_cap': True})
    numbers, masks, channel_stats = {}, {}, {}
    aliases = manifest.get('identity_aliases', {})
    short = [day.replace('-', '') for day in sessions]
    for api, keep in [('daily', ('open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount')),
                      ('adj_factor', ('adj_factor',)), ('stk_limit', ('up_limit', 'down_limit')),
                      ('daily_basic', ('turnover_rate', 'total_share', 'float_share', 'free_share'))]:
        panel = accumulate_panel(api, catalog_frames(root, manifest, api, aliases, codes), short, codes, keep=keep)
        masks[api], masks[api + '_conflicts'] = panel['valid'], panel['conflict']
        numbers.update(panel['values'])
        channel_stats[api] = panel['stats']
        print(json.dumps({'full_market_input_channel': api, **panel['stats']}), flush=True)
        del panel
    records = load_table(root, 'suspend_d.csv', codes, set(short))
    halts = defaultdict(list)
    for row in records.to_dict('records'):
        halts[(row['ts_code'], iso(row['trade_date']))].append(row)
    observed_halt_dates = {day for part in manifest['queries']['suspend_d'] for day in short
                          if part['params']['start_date'] <= day <= part['params']['end_date']}
    require(set(short) <= observed_halt_dates, 'sparse halt dates were not queried')
    names_frame = pd.read_csv(root / 'namechange.csv', dtype=str).astype(object).where(lambda f: f.notna(), None)
    names = defaultdict(list)
    for row in names_frame.to_dict('records'):
        names[aliases.get(row['ts_code'], {}).get('stable_code', row['ts_code'])].append(row)
    dividends = pd.read_csv(root / 'dividend.csv', dtype=str).astype(object).where(lambda f: f.notna(), None)
    dividend_records = dividends.to_dict('records')
    for row in dividend_records:
        row['ts_code'] = aliases.get(row['ts_code'], {}).get('stable_code', row['ts_code'])
    terms = resolve_action_terms(dividend_terms(dividend_records, sessions), dividend_records, [])
    if action_term_exemptions is not None:
        # P-009 (a3): hash-bound registry declaring why a specific unresolved
        # vendor event stays unresolved. Certification treats bound terms as
        # declared residual gaps, never as unobserved contradictions.
        from .historical_adapter import apply_action_term_exemptions
        terms = apply_action_term_exemptions(terms, action_term_exemptions)
    wanted = set(instruments)
    terms = [term for term in terms if term['instrument'] in wanted]
    crosscheck, crosscheck_sources = factor_corroboration(root, manifest, codes, set(short))
    source_hashes = {filename: manifest['sources'][api]['sha256'] for filename, api in TABLE_APIS.items()}
    arrays = derive_market_arrays(sessions, codes, members, numbers, masks, names, terms, halts,
        listing_age_calendar_days=listing_age_calendar_days, crosscheck=crosscheck, source_hashes=source_hashes)
    del numbers
    provenance = {'source_hashes': source_hashes, 'catalog': source, 'calendar_parse_audit': calendar_audit,
                  'factor_crosscheck_sources': crosscheck_sources,
                  'availability': 'daily 18:00 conservative vendor proxy; no original publication-hour proof',
                  'latest_version_policy': 'use current accessible frozen vendor versions; no historical PIT archive claim',
                  'field_units': {'volume': 'hands*100 -> shares', 'amount': 'thousand CNY*1000 -> CNY',
                                  'turnover_rate': 'percentage*.01 -> ratio', 'shares': 'ten-thousand shares*10000'},
                  'source_metadata_hash': metadata['metadata_hash']}
    fields = {f: {'unit': 'shares' if f == 'volume' or f.endswith('_shares') else
                         'currency' if f == 'amount' else 'ratio' if f == 'turnover_rate' else 'price',
                  'frequency': 'daily', 'adjustment': 'raw'} for f in FIELDS}
    snapshot = {'schema_version': 'fuxi.columnar_snapshot.1', 'source_id': 'full_A_existing_catalog_current_vintage',
                'exposure': 'research_exposed', 'sessions': sessions, 'instruments': instruments,
                'evaluation_start': sessions[warmup], 'decision_times': {d: d + 'T18:30:00+08:00' for d in sessions},
                'fields': fields, 'availability_times': {f: [d + 'T18:00:00+08:00' for d in sessions] for f in FIELDS},
                'arrays': {'fields': {f: save_array(out / 'arrays' / ('field_' + f + '.npy'), arrays['columns'][f]) for f in FIELDS},
                           'barriers': save_array(out / 'arrays/barriers.npy', arrays['barriers']),
                           'membership': save_array(out / 'arrays/membership.npy', arrays['membership'])},
                'quality': {'pit_certified': False, 'universe_certified': False, 'account_events_certified': False,
                            'gaps': ['historical_revision_archive_unverified', 'name_announcement_times_daily_proxy',
                                     'corporate_actions_vendor_terms_not_holder_settlement_receipts',
                                     'tiny_adjustment_drift_declared_research_sensitivity', 'incomplete_event_and_retired_asset_marks']},
                'cohort_declaration': dict(declaration, members=members), 'provenance': provenance}
    execution = execution_profile()
    from .a_share import rule_for
    require(fractional_share_policy in ('block', 'floor', 'ceil'),
            'fractional_share_policy must be one of block|floor|ceil')
    execution.update(instrument_rules={code: rule_for(code) for code in codes}, order_type='market',
                     scope='all_a_share_quantity_contract_research_only', odd_lot_exit='whole_residual',
                     fractional_share_policy=fractional_share_policy, identity_aliases=aliases,
                     sell_tax_schedule=[['0001-01-01', '0.001'], ['2023-08-28', '0.0005']],
                     dividend_tax_model='flat_20_percent_declared_research_not_actual_tax_receipt')
    market = {'schema_version': 'fuxi.columnar_market.1', 'sessions': sessions, 'instruments': instruments,
              'arrays': {group: {name: save_array(out / 'arrays' / ('market_' + name + '.npy'), v)
                                for name, v in arrays[group].items()}
                         for group in ('numeric', 'status', 'categorical')},
              'category_values': {'event_coverage': {str(i): v for i, v in enumerate(CLASSIFICATIONS)},
                                  'mark_source': {str(i): v for i, v in enumerate(MARK_SOURCES)}},
              'sparse_bars': arrays['sparse_bars'], 'actions': [], 'action_terms': terms, 'execution': execution,
              'event_coverage_certified': False, 'tradability_certified': False, 'provenance': provenance}
    market['market_hash'] = digest(market)
    label_values = np.full(arrays['membership'].shape, np.nan)
    label_status = np.zeros(arrays['membership'].shape, np.uint8)
    mature_at = [None] * len(sessions)
    event_present = np.zeros(arrays['membership'].shape, bool)
    instrument_index = {code: j for j, code in enumerate(instruments)}
    session_index = {day: i for i, day in enumerate(sessions)}
    for term in terms:
        if term['ex_date'] in session_index:
            event_present[session_index[term['ex_date']], instrument_index[term['instrument']]] = True
    for i in range(len(sessions) - horizon - 1):
        a, b = i + 1, i + horizon + 1
        present = arrays['membership'][i]
        entry, exit_mark = arrays['numeric']['open'][a], arrays['numeric']['close'][b]
        event_free = (arrays['status']['corporate_action_known'][a:b + 1].all(axis=0)
                      & ~event_present[a:b + 1].any(axis=0))
        complete = present & np.isfinite(entry) & (entry > 0) & np.isfinite(exit_mark) & event_free
        complete &= arrays['status']['paused'][a] == 0
        complete &= arrays['categorical']['mark_source'][b] == MARK_SOURCES.index('observed_close')
        label_status[i, present] = 2
        label_status[i, complete] = 1
        label_values[i, complete] = exit_mark[complete] / entry[complete] - 1
        mature_at[i] = sessions[b] + 'T18:00:00+08:00'
    labels = {'schema_version': 'fuxi.columnar_labels.1', 'feature_snapshot_hash': digest(snapshot),
              'sessions': sessions, 'instruments': instruments,
              'arrays': {'values': save_array(out / 'arrays/label_values.npy', label_values),
                         'status': save_array(out / 'arrays/label_status.npy', label_status)},
              'status_codes': {'valid': 1, 'data_insufficient': 2, 'not_mature': 3, 'invalid': 4, 'absent': 0},
              'mature_at': mature_at, 'evidence_namespace': market['market_hash'],
              'label_spec': {'definition': 'next-session raw open to horizon-session raw close; corporate-action spans excluded; price proxy, no transaction costs',
                             'horizon_sessions': horizon, 'execution_contract_hash': digest(execution),
                             'kind': 'prediction_proxy', 'price_basis': 'declared_in_definition'}}
    audit_arrays = {'active': arrays['active'], 'reason_masks': arrays['reason_masks'],
                    'name_states': arrays['name_states'], 'full_halt': arrays['full_halt'],
                    'partial_halt': arrays['partial_halt'], **masks}
    audit_refs = {name: save_array(out / 'audit_arrays' / (name + '.npy'), values)
                  for name, values in audit_arrays.items()}
    expected = arrays['active'] & ~arrays['full_halt']
    missing = {api: {'missing_expected_cells': int((expected & ~masks[api]).sum()),
                     'members_with_missing': int((expected & ~masks[api]).any(axis=0).sum())}
               for api in ('daily', 'adj_factor', 'stk_limit', 'daily_basic')}
    industry = {'not_used_for_alpha_or_neutralization': True, 'historical_PIT': False}
    if latest_industry_path:
        path = Path(latest_industry_path).resolve()
        frame = pd.read_csv(path, dtype=str).astype(object).where(lambda x: x.notna(), None)
        rows = [row for row in frame.to_dict('records') if row['ts_code'] in set(codes)]
        industry.update(path=str(path), sha256=file_hash(path), observed_members=len(rows),
                        missing_members=len(codes) - len(rows), policy='latest available observed mapping; unknown retained')
        save_new(out / 'LATEST_INDUSTRY.json', dict(industry, rows=rows))
    from .a_share import board
    audit = {'schema_version': 'fuxi.full_market_input_build.1', 'window': [start, end],
             'sessions': len(sessions), 'members': len(codes), 'board_counts': dict(Counter(board(c) for c in codes)),
             'active_end_members': int(arrays['active'][-1].sum()),
             'within_window_delisted': len([m for m in members if m['delist_date'] and
                                           start.replace('-', '') < m['delist_date'] <= end.replace('-', '')]),
             'active_cells': int(arrays['active'].sum()), 'eligible_cells': int(arrays['membership'].sum()),
             'eligible_per_session_min': int(arrays['membership'].sum(axis=1).min()),
             'eligible_per_session_max': int(arrays['membership'].sum(axis=1).max()),
             'member_deletion_for_missing_data': 0, 'channel_stats': channel_stats, 'channel_missing': missing,
             'reason_counts': {reason: int(((arrays['reason_masks'] & (1 << bit)) != 0).sum()) for bit, reason in enumerate(REASONS)},
             'corporate_action_unknown_active_cells': int((arrays['active'] & ~arrays['status']['corporate_action_known']).sum()),
             'unknown_active_mark_cells': int((arrays['active'] & ~np.isfinite(arrays['numeric']['close'])).sum()),
             'classification_counts': {label: int((arrays['categorical']['event_coverage'] == i).sum()) for i, label in enumerate(CLASSIFICATIONS)},
             'action_term_status': dict(Counter(t['status'] for t in terms)),
             'action_term_exemptions': None if action_term_exemptions is None else {
                 'path': action_term_exemptions['path'],
                 'sha256': action_term_exemptions['sha256'],
                 'bound_terms': sum(1 for t in terms if t.get('exemption'))},
             'labels_status_counts': {name: int((label_status == value).sum()) for name, value in labels['status_codes'].items()},
             'label_tail_absent_sessions': horizon + 1, 'formal_data_assured': False,
             'historical_PIT_assurance': False, 'new_market_requests': 0, 'new_future_observations': 0,
             'latest_industry': industry, 'array_refs': audit_refs,
             'listing_age_calendar_days': listing_age_calendar_days,
             'elapsed_seconds': time.monotonic() - started, 'source_binding': source}
    payload = {'schema_version': 'fuxi.campaign_input.1', 'snapshot': snapshot, 'labels': labels,
               'market': market, 'research_clock': sessions[-1] + 'T20:00:00+08:00', 'audit': audit}
    save_new(out / 'input.json', payload)
    save_new(out / 'AUDIT.json', audit)
    save_new(out / 'RECEIPT.json', {'input_sha256': file_hash(out / 'input.json'), 'snapshot_hash': digest(snapshot),
        'market_hash': market['market_hash'], 'label_hash': digest(labels), 'audit_hash': digest(audit),
        'members': len(codes), 'sessions': len(sessions), 'arrays_bytes': sum(p.stat().st_size for p in out.rglob('*.npy')),
        'action_term_exemptions': None if action_term_exemptions is None else
            {'path': action_term_exemptions['path'], 'sha256': action_term_exemptions['sha256'],
             'bound_terms': sum(1 for t in terms if t.get('exemption'))},
        'builder_sha256': file_hash(Path(__file__)), 'member_deletion': 0, 'new_market_requests': 0})
    print(json.dumps({'full_market_input_completed': True, 'members': len(codes),
                      'sessions': len(sessions), 'elapsed_seconds': audit['elapsed_seconds']}), flush=True)
    return payload
