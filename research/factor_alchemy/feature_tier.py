# -*- coding: utf-8 -*-
"""两层因子分层：研究特征库（无 JQ 晋级资格）+ 策略晋级库（必须完整通过 S5）。

背景
----
P0/P1 的结论是：行业、业绩预告、大宗交易、基准（以及早先的龙虎榜掩码、被复现的融资
买入占比）在单因子口径下**上限达不到盈亏平衡 IC**，但它们作为**条件特征 / 出门口过滤 /
风险口径**仍然可用。把它们与「能独立成为多头策略的因子」混在同一张表里，会让
「覆盖 1% 的掩码」看起来像「一个候选因子」，并诱导用绕开 S5 的方式制造产出数字。

本模块把这件事做成一个**单向下锁**：
  * 登记为 `research_feature` 的名字 **一律禁止**走 JQ 晋级（`assert_promotable` 抛错）；
  * 本登记库**永远不授予**晋级资格（`registry_grants_promotion()` 恒 False 并有断言），
    晋级只能由既有的 S5 完整裁决路径产生；
  * 未登记的名字**不受本库影响**（保持 legacy 行为）—— 这样新库只会减少、不会增加
    任何东西，符合「只删不加 ⇒ 可翻」的默认值纪律。

研究口径
--------
研究特征不按 standalone IC 评估互补性，而按**固定基准组合上的增量净收益**：
基准组合固定为「现有系统按既有信号排序取 Top-N 等权」，特征只作为叠加（出门口过滤 /
轻度倾斜），报 `delta_net` 并按**日期聚类**读 t（同一日多信号共享市场期间，朴素 t 会
被放大）。`standalone_ic_is_not_promotion_evidence` 显式拒绝把单因子 IC 当晋级证据。

用法
----
    python research/factor_alchemy/feature_tier.py validate
    python research/factor_alchemy/feature_tier.py gate <name>
    python research/factor_alchemy/feature_tier.py promotion-view
    python research/factor_alchemy/feature_tier.py seed          # 幂等：写入预置条目
    python research/factor_alchemy/feature_tier.py measure       # 研究口径读数（重，需数据）
    python research/factor_alchemy/feature_tier.py tests
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'data' / 'factor_tier_registry.json'
JQ_GENERATOR = ROOT / 'scripts' / 'gen_jq_s5_standalone.py'
OUT = ROOT / 'reports' / 'feature_tier_20260924'

SCHEMA = 'fuxi_factor_tier_registry/v1'
RESEARCH = 'research_feature'
PROMOTION = 'promotion_candidate'
TIERS = (RESEARCH, PROMOTION)

USES = ('conditional_feature', 'exit_filter', 'risk_caliber', 'attribution_only')

TOP_N = 80
COST_BPS = 15.0
MIN_BASE = 20            # a base book thinner than this cannot carry an overlay claim
MIN_DATES = 20           # below this, report the reading as not yet usable


class NotPromotableError(RuntimeError):
    """Raised when a name is asked to enter the promotion path while it is a research feature."""


# ------------------------------------------------------------------ registry I/O

def _empty_registry():
    return {
        'schema': SCHEMA,
        'policy': {
            'power': 'downgrade_only',
            'grants_promotion': False,
            'unregistered_names': 'unaffected_legacy_behaviour',
            'promotion_path': 'full S5 adjudication; this registry never substitutes for it',
            'research_feature_evaluation': 'incremental_net_return_on_a_fixed_base_portfolio',
            'standalone_ic': 'not accepted as promotion evidence',
        },
        'tiers': {RESEARCH: [], PROMOTION: []},
        'pending_measurement': [],
        'generated_at': None,
    }


def load(path=None):
    """Read the registry. `path` is resolved at call time, not bound as a default argument —
    a default bound at definition time silently ignores a later change of `REGISTRY`, which
    is exactly how a gate ends up reading the wrong file (or a test ends up passing vacuously).

    Absence and breakage are different: a missing file means the library was never adopted,
    so legacy behaviour stands; a path that exists but cannot be read or parsed raises, and
    the caller turns that into a refusal. Treating a broken registry as "empty" would turn a
    configuration fault into a silent permission.
    """
    path = Path(path or REGISTRY)
    if not path.exists():
        return _empty_registry()
    if not path.is_file():
        raise OSError('registry path exists but is not a file: %s' % path)
    return json.loads(path.read_text(encoding='utf-8'))


def save(registry, path=None):
    """UTF-8, no BOM, LF, serialised from bytes (never truncate-then-encode)."""
    path = Path(path or REGISTRY)
    registry = dict(registry)
    registry['schema'] = SCHEMA
    registry['generated_at'] = datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')
    text = json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=False) + '\n'
    payload = text.encode('utf-8')
    assert payload[:3] != b'\xef\xbb\xbf'
    assert payload.count(b'\r\n') == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(payload)
    assert Path(path).read_bytes() == payload
    return path


def _index(registry):
    """Name -> (tier, entry), with the research tier taking precedence.

    A name present in both tiers must not be able to unblock itself. Promotion records are
    written by a *different* code path than the gate, so if promotion simply overwrote
    research here, a name that is both would silently become promotable — the gate would
    report ALLOW on a research feature. Research wins, deterministically.
    """
    out = {}
    for tier in (PROMOTION, RESEARCH):          # research applied last => it wins
        for entry in registry.get('tiers', {}).get(tier, []):
            out[entry['name']] = (tier, entry)
    return out


# ------------------------------------------------------------------ the gate

def blocked(name, registry=None):
    """The only power this registry has: block a research feature from promotion."""
    registry = registry or load()
    tier, _ = _index(registry).get(name, (None, None))
    return tier == RESEARCH


def promotion_allowed(name, registry=None):
    """True = the registry does not object. It never means the registry grants promotion."""
    return not blocked(name, registry=registry)


def assert_promotable(name, registry=None):
    if blocked(name, registry=registry):
        raise NotPromotableError(
            '%s is registered as a research feature: no JQ promotion eligibility. '
            'Promotion requires a full S5 pass, not a tier change.' % name)
    return True


def registry_grants_promotion():
    """Structural property, asserted in tests: the registry cannot grant promotion."""
    return False


def standalone_ic_is_not_promotion_evidence(evidence):
    """Guard for callers tempted to cite a standalone IC as a promotion reason."""
    kind = (evidence or {}).get('kind')
    if kind in ('standalone_ic', 'ic', 'rank_ic', 'single_factor_ic'):
        raise ValueError(
            'standalone IC is not promotion evidence: research features are measured by '
            'incremental net return on a fixed base portfolio')
    return True


# ------------------------------------------------------------------ promotion view

def jq_registered_names(path=JQ_GENERATOR):
    """Names that already have a JQ standalone generator, parsed from its registry."""
    import re
    src = Path(path).read_text(encoding='utf-8')
    body = re.search(r'FACTORS = \{(.*?)\n\}\n', src, re.S)
    if not body:
        raise AssertionError('could not parse FACTORS from %s' % path)
    return re.findall(r'\n    "([A-Za-z0-9_]+)": \{', body.group(1))


def parse_generator_evidence(path=JQ_GENERATOR):
    """Extract the S5 readings that today exist only as prose in a Python comment.

    This is a *transcription with provenance*, not a quotation of a structured artifact:
    every record carries `structured: false` and the exact field it came from, so no
    downstream reader can mistake it for an independent re-read.
    """
    import re
    src = Path(path).read_text(encoding='utf-8')
    out = {}
    for match in re.finditer(r'\n    "([A-Za-z0-9_]+)": \{\n(.*?)\n    \},', src, re.S):
        name, body = match.group(1), match.group(2)
        note = re.search(r'"local_note": "(.*?)",', body, re.S)
        if not note:
            continue
        text = note.group(1)
        nums = {}
        for label, pattern in (('excess_25', r'excess_25=(-?[\d.]+)%'),
                               ('excess_26', r'excess_26=(-?[\d.]+)%'),
                               ('calmar_25', r'Calmar_25=(-?[\d.]+)'),
                               ('calmar_26', r'Calmar_26=(-?[\d.]+)')):
            found = re.search(pattern, text)
            if found:
                nums[label] = float(found.group(1))
        icir = re.search(r'ICIR=([\d.]+)', text)
        out[name] = {
            'kind': 's5_readings',
            'structured': False,
            'parsed_from': '%s::FACTORS.%s.local_note' % (
                Path(path).name, name),
            'note': text,
            'icir_note': float(icir.group(1)) if icir else None,
            **nums,
        }
    return out


def historical_pass_ledger(path=None):
    """The other S5 record system, kept in a JSON ledger with disjoint names."""
    import json as _json
    path = Path(path or (ROOT / 'data' / '_s5_historical_pass_formulas.json'))
    if not path.exists():
        return {}
    rows = _json.loads(path.read_text(encoding='utf-8'))
    return {r['name']: {'kind': 's5_readings', 'structured': True,
                        'parsed_from': path.name,
                        'excess_25': r.get('ex25'), 'excess_26': r.get('ex26'),
                        'calmar_25': r.get('cal25'), 'calmar_26': r.get('cal26'),
                        'ic': r.get('ic'), 'icir': r.get('icir')}
            for r in rows if r.get('name')}


def sync_promotion_evidence(registry=None, path=None):
    """Record the promotion layer's evidence, never its membership.

    Membership is observed from the JQ generator (the endpoint that actually exists);
    this function only attaches evidence pointers and reports every conflict instead of
    resolving it. A name that is blocked by the research tier is reported as a conflict,
    not quietly dropped, and a name with no evidence ends up `evidence_not_recorded`.
    """
    registry = registry or load(path)
    jq_names = jq_registered_names()
    evidence = parse_generator_evidence()
    ledger = historical_pass_ledger()
    blocked_names = {e['name'] for e in registry['tiers'][RESEARCH]}
    existing = {e['name']: e for e in registry['tiers'][PROMOTION]}

    recorded, conflicts = [], []
    for name in jq_names:
        ev = evidence.get(name) or ledger.get(name)
        if name in blocked_names:
            conflicts.append({'name': name, 'kind': 'blocked_name_has_jq_generator',
                              'note': 'the research tier says no promotion; the JQ endpoint '
                                      'already exists. Report only: the registry does not '
                                      'delete or rewrite the endpoint, and it does not add a '
                                      'promotion record for a blocked name either.'})
            continue                       # one-way: never write a promotion row for it
        if name not in existing:
            registry['tiers'][PROMOTION].append({
                'name': name,
                'registered_by': 'observed:scripts/gen_jq_s5_standalone.py',
                's5_evidence': ev,
                'jq_generator': 'research/factor_alchemy/jq_s5_pass_%s_standalone.py' % name,
            })
        else:
            existing[name]['s5_evidence'] = ev
            existing[name]['jq_generator'] = (
                existing[name].get('jq_generator')
                or 'research/factor_alchemy/jq_s5_pass_%s_standalone.py' % name)
        if ev:
            recorded.append(name)

    registry['tiers'][PROMOTION].sort(key=lambda e: e['name'])
    registry['promotion_ledger_crosscheck'] = {
        'jq_registered': len(jq_names),
        'with_evidence': len(recorded),
        'without_evidence': len(jq_names) - len(recorded),
        'structured_evidence': sum(1 for n in recorded
                                   if (evidence.get(n) or ledger.get(n) or {}).get('structured')),
        'unstructured_evidence': sum(1 for n in recorded
                                     if not (evidence.get(n) or ledger.get(n) or {})
                                     .get('structured')),
        'second_ledger_names': len(ledger),
        'second_ledger_overlap_with_jq': len(set(ledger) & set(jq_names)),
        'note': 'two disjoint S5 record systems exist: prose in the JQ generator, and the '
                'JSON ledger. Neither is a substitute for the other, and the overlap is empty.',
    }
    save(registry, path)
    return {'jq_registered': len(jq_names), 'with_evidence': len(recorded),
            'conflicts': conflicts,
            'crosscheck': registry['promotion_ledger_crosscheck']}


def promotion_view(registry=None, jq_names=None, evidence=None):
    """Report the promotion layer as a *view*: it never rewrites the truth it reads."""
    registry = registry or load()
    jq_names = list(jq_names if jq_names is not None else jq_registered_names())
    evidence = evidence if evidence is not None else parse_generator_evidence()
    ledger = historical_pass_ledger()
    recorded = {e['name']: e for e in registry.get('tiers', {}).get(PROMOTION, [])}
    rows = []
    for name in jq_names:
        ev = (recorded.get(name) or {}).get('s5_evidence') or evidence.get(name) or ledger.get(name)
        if blocked(name, registry=registry):
            status = 'blocked_by_registry'
        elif not ev:
            status = 'evidence_not_recorded'
        elif ev.get('structured'):
            status = 'evidence_recorded_structured'
        else:
            status = 'evidence_recorded_unstructured'
        rows.append({'name': name, 'status': status,
                     's5_evidence': ev,
                     'registered_proposal': (recorded.get(name) or {}).get('registered_by')})
    return {
        'jq_registered': len(jq_names),
        'evidence_recorded_structured': sum(1 for r in rows
                                            if r['status'] == 'evidence_recorded_structured'),
        'evidence_recorded_unstructured': sum(1 for r in rows
                                             if r['status'] == 'evidence_recorded_unstructured'),
        'evidence_not_recorded': sum(1 for r in rows if r['status'] == 'evidence_not_recorded'),
        'blocked_by_registry': sum(1 for r in rows if r['status'] == 'blocked_by_registry'),
        'rows': rows,
        'crosscheck': registry.get('promotion_ledger_crosscheck'),
        'note': 'evidence_not_recorded is a disclosure, not a demotion: those names were '
                'promoted before this registry existed and keep their existing status.',
    }


# ------------------------------------------------------------------ validate

def validate(registry=None):
    registry = registry or load()
    issues, notes = [], []
    seen = {}
    for tier in TIERS:
        for entry in registry.get('tiers', {}).get(tier, []):
            name = entry.get('name')
            if not name:
                issues.append({'kind': 'missing_name', 'tier': tier})
                continue
            if name in seen:
                issues.append({'kind': 'name_in_two_tiers', 'name': name,
                               'tiers': [seen[name], tier]})
            seen[name] = tier
            if tier == RESEARCH:
                if entry.get('promotion_eligible') is not False:
                    issues.append({'kind': 'research_feature_marked_promotable', 'name': name})
                bad = [u for u in entry.get('usable_as', []) if u not in USES]
                if bad:
                    issues.append({'kind': 'unknown_use', 'name': name, 'uses': bad})
                if not entry.get('usable_as'):
                    issues.append({'kind': 'research_feature_without_a_use', 'name': name})
                if not entry.get('why_not_promotable'):
                    issues.append({'kind': 'missing_reason', 'name': name})
                if not entry.get('evidence'):
                    issues.append({'kind': 'missing_evidence_pointer', 'name': name})
            else:
                if not entry.get('s5_evidence'):
                    notes.append({'kind': 'promotion_candidate_without_s5_evidence',
                                  'name': name})
    if registry.get('policy', {}).get('grants_promotion') is not False:
        issues.append({'kind': 'policy_claims_to_grant_promotion'})
    return {'ok': not issues, 'issues': issues, 'notes': notes,
            'counts': {t: len(registry.get('tiers', {}).get(t, [])) for t in TIERS},
            'pending_measurement': len(registry.get('pending_measurement', []))}


# ------------------------------------------------------------------ seed

SEED_RESEARCH = [
    dict(name='ind_mom20_idx', family='industry',
         evaluated_columns=['ind_mom20_idx'],
         measured_in_batch='reports/ic_ceiling_20260924 (P0)',
         usable_as=['risk_caliber', 'conditional_feature', 'attribution_only'],
         why_not_promotable='P0 CEILING_BELOW: raw upper bound 0.0507 < break-even 0.108; '
                            'index channel is also gated by a 10-session staleness rule.',
         evidence=['reports/ic_ceiling_20260924/summary.json',
                   'reports/fuxi_ic_ceiling_20260924.md']),
    dict(name='ind_mom20_mem', family='industry',
         evaluated_columns=['ind_mom20_mem'],
         measured_in_batch='reports/ic_ceiling_20260924 (P0)',
         usable_as=['conditional_feature', 'risk_caliber'],
         why_not_promotable='P0 CEILING_BELOW; the interpretable reading is raw -0.0278 '
                            '(t=-1.70). Industry membership is a conditioner, not an alpha.',
         evidence=['reports/ic_ceiling_20260924/summary.json',
                   'reports/fuxi_ic_ceiling_20260924.md']),
    dict(name='fcst_yoy_mid', family='forecast',
         evaluated_columns=['fcst_yoy_mid', 'fcst_range_ratio'],
         measured_in_batch='both the frozen external-info builder and reports/ic_ceiling_20260924',
         usable_as=['conditional_feature', 'exit_filter'],
         why_not_promotable='P0 CEILING_BELOW: upper bound 0.0385; the third item had already '
                            'found it insignificant after date clustering.',
         evidence=['reports/ic_ceiling_20260924/summary.json',
                   'reports/fuxi_external_info_20260924.md']),
    dict(name='bt_premium', family='block_trade',
         evaluated_columns=['bt_premium'],
         measured_in_batch='reports/blocktrade_p1_20260924 (P1)',
         usable_as=['exit_filter', 'conditional_feature'],
         why_not_promotable='P1: independent-sample residual +0.0480 (t=2.86) but coverage is '
                            '1.13% of the pool, i.e. a conditioner magnitude, not a book.',
         evidence=['reports/blocktrade_p1_20260924/summary.json',
                   'reports/fuxi_blocktrade_p1_20260924.md']),
    dict(name='bt_share', family='block_trade',
         evaluated_columns=['bt_share'],
         measured_in_batch='reports/blocktrade_p1_20260924 (P1)',
         usable_as=['conditional_feature'],
         why_not_promotable='P1 H1 unresolved: primary-sample interval [-0.0162, +0.0759] '
                            'straddles zero and the three segments disagree in sign.',
         evidence=['reports/blocktrade_p1_20260924/summary.json',
                   'reports/fuxi_blocktrade_p1_20260924.md']),
    dict(name='lh_event_mask', family='top_list',
         evaluated_columns=['lh_net_rate', 'lh_seat_asym'],
         measured_in_batch='both the frozen external-info builder and reports/fuxi_external_info_20260924.md',
         usable_as=['exit_filter'],
         why_not_promotable='Coverage 1.10% and the middle segment returns to zero; -75.2bps '
                            't=-2.65 survives every control but cannot carry a book.',
         evidence=['reports/fuxi_external_info_20260924.md']),
    dict(name='margin_rz_buyrate', family='margin_detail',
         evaluated_columns=['margin_rz_buyrate', 'margin_rz_chg5'],
         measured_in_batch='both the frozen external-info builder and reports/fuxi_external_info_20260924.md',
         usable_as=['conditional_feature', 'attribution_only'],
         why_not_promotable='Reproduced by the single standard field turnover_rate (difference '
                            '4e-5, cross-sectional rank correlation 0.62~0.70) => redundant, '
                            'not a new information cluster.',
         evidence=['reports/fuxi_external_info_20260924.md']),
    dict(name='benchmark_timing_overlay', family='benchmark',
         evaluated_columns=[],
         measured_in_batch='reports/ic_ceiling_20260924 (P0); no per-name column exists here',
         usable_as=['risk_caliber', 'attribution_only'],
         why_not_promotable='An index has no cross-sectional variance, so it can never be a '
                            'stock-level factor; its timing overlay upper bound is 0.0042.',
         evidence=['reports/ic_ceiling_20260924/summary.json',
                   'reports/fuxi_ic_ceiling_20260924.md']),
]

SEED_PENDING = [
    dict(name='northbound_hold_share', family='northbound',
         reason='NOT_MEASURABLE, not low: hk_hold.csv ends 2024-12-31 and the window carries '
                '0 observations. Registering it as a research feature would be an unearned '
                'downgrade, so it stays in pending until a replay is available.',
         evidence=['reports/ic_ceiling_20260924/summary.json']),
]


def seed(registry=None, path=None):
    """Idempotent: adds the pre-registered research features and refreshes their declared
    column mapping (`evaluated_columns`) without ever touching the tier or the reason."""
    registry = registry or load(path)
    by_name = {e['name']: e for e in registry['tiers'][RESEARCH]}
    promoted = {e['name'] for e in registry['tiers'][PROMOTION]}
    added, refused, refreshed = [], [], []
    for entry in SEED_RESEARCH:
        if entry['name'] in by_name:
            target = by_name[entry['name']]
            for key in ('evaluated_columns', 'measured_in_batch', 'family'):
                if target.get(key) != entry[key]:
                    target[key] = entry[key]
                    refreshed.append({'name': entry['name'], 'key': key})
            continue
        if entry['name'] in promoted:
            refused.append(entry['name'])          # one-way: never demote an existing promotion
            continue
        row = dict(entry)
        row['promotion_eligible'] = False
        row['registered_by'] = 'P-20260924-012'
        registry['tiers'][RESEARCH].append(row)
        added.append(entry['name'])
    pending = {p['name'] for p in registry.get('pending_measurement', [])}
    for item in SEED_PENDING:
        if item['name'] not in pending:
            registry.setdefault('pending_measurement', []).append(dict(item))
    registry['tiers'][RESEARCH].sort(key=lambda e: (e['family'], e['name']))
    save(registry, path)
    return {'added': added, 'refused_one_way': refused, 'refreshed': refreshed,
            'research_features': len(registry['tiers'][RESEARCH]),
            'pending': len(registry.get('pending_measurement', []))}


def declared_columns(registry=None):
    """Map each research feature to the concrete columns the frozen builder must expose.

    The registry names are family-level concepts; the evaluation layer keys on columns. The
    mapping is declared explicitly so a name can never silently evaluate an empty column set
    and report "measured" on nothing.
    """
    registry = registry or load()
    return {e['name']: list(e.get('evaluated_columns', []))
            for e in registry['tiers'][RESEARCH]}


# ------------------------------------------------------------------ research measurement

def _ei():
    """The frozen section builder, imported (not rewritten)."""
    scripts = str(ROOT / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module('fuxi_external_info_20260924')


def capture_frames(columns, ei=None):
    """Run the frozen section builder once, capturing the labelled pool per section.

    `evaluate_factor` is wrapped, never replaced: the returned rows must equal the frozen
    run, so any drift shows up as a parity failure rather than as a quiet new reading.

    The date is recovered by call index: `rebuild_sections` appends exactly one row per
    `evaluate_factor` call, in order, so the i-th call owns `rows[i]`. The mapping is
    asserted column by column rather than trusted.
    """
    ei = ei or _ei()
    original = ei.evaluate_factor
    captured, calls = [], {'n': 0}

    def wrapper(frame, labels, column, resid_base, resid_full, existing):
        index = calls['n']
        calls['n'] += 1
        row = original(frame, labels, column, resid_base, resid_full, existing)
        if column in columns:
            # the evaluated column itself must travel with the copy: keeping only the
            # structural fields silently produced a frame that could no longer be evaluated
            keep = ['ts_code', 'label', 'existing', 'industry', 'log_size', column]
            keep = list(dict.fromkeys(keep))
            assert column in frame.columns, (
                'the frozen builder handed us %r without the column it asked for' % (column,))
            captured.append({'column': column, 'call_index': index,
                             'frame': frame[keep].copy(),
                             'labels': np.asarray(labels, dtype=float)})
        return row

    ei.evaluate_factor = wrapper
    try:
        rows, drops, geometry, sources = ei.rebuild_sections()
    finally:
        ei.evaluate_factor = original
    assert ei.evaluate_factor is original, 'the frozen builder was left patched'
    assert len(rows) == calls['n'], 'one row per evaluate_factor call expected: %d vs %d' % (
        len(rows), calls['n'])
    for item in captured:
        row = rows[item['call_index']]
        assert row['factor'] == item['column'], 'call index misaligned'
        item['date'] = row['date']
        item['signal'] = row['signal']
    return captured, {'rows': rows, 'drops': drops, 'geometry': geometry,
                      'sources': sources, 'evaluate_calls': calls['n']}


def _base_book(frame, labels, top_n=TOP_N):
    """The fixed base portfolio: the existing composite's own ranking, equal weighted."""
    ok = np.isfinite(labels) & np.isfinite(frame.existing.to_numpy(dtype=float))
    idx = np.where(ok)[0]
    if len(idx) < MIN_BASE:
        return None, None
    order = idx[np.argsort(-frame.existing.to_numpy(dtype=float)[idx], kind='stable')][:min(top_n, len(idx))]
    return order, labels[order]


def _overlay_weights(base_values, feature, mode, quantile=0.2):
    """Return (kept_mask, weight_change_l1_half) for the requested overlay on the base book."""
    n = len(base_values)
    w_before = np.full(n, 1.0 / n)
    f = feature
    if mode == 'exit_filter':
        finite = np.isfinite(f)
        if finite.sum() < 5:
            return None, None
        cut = np.nanquantile(f[finite], quantile)
        keep = ~(finite & (f <= cut))
        if keep.sum() < MIN_BASE:
            return None, None
        kept = keep
    elif mode == 'tilt':
        finite = np.isfinite(f)
        if finite.sum() < 5:
            return None, None
        # missing values keep a neutral tilt: absence is never imputed as a signal
        r = np.full(n, 0.5)
        r[finite] = (np.argsort(np.argsort(f[finite])) + 0.5) / finite.sum()
        raw = np.maximum(r, 1e-6)
        w_after = raw / raw.sum()
        return w_after, float(np.abs(w_after - w_before).sum() / 2.0)
    else:
        raise ValueError('unknown overlay mode %r' % mode)
    w_after = np.zeros(n)
    w_after[kept] = 1.0 / kept.sum()
    return w_after, float(np.abs(w_after - w_before).sum() / 2.0)


def _mean_ci(ei, values):
    """Read the frozen module's interval, asserting its shape instead of trusting it.

    `mean_ci` returns (mean, lo, hi, n). Unpacking it as a pair raised only at the very end
    of a two-minute run; the shape check makes the same mistake fail immediately.
    """
    got = ei.mean_ci(np.asarray(values, dtype=float))
    assert isinstance(got, (tuple, list)) and len(got) == 4, (
        'mean_ci must return (mean, lo, hi, n); got %r' % (got,))
    lo, hi = got[1], got[2]
    return float(got[0]), (None if lo is None else float(lo)), (
        None if hi is None else float(hi)), int(got[3])


def incremental_net_return(captured, cost_bps=COST_BPS, modes=('exit_filter', 'tilt'),
                           top_n=TOP_N, ei=None):
    """Incremental net return of a research feature on a fixed base book, clustered by date.

    For each (column, section) we build the base book from the existing composite, apply the
    overlay, and record gross and net deltas. Rows are then averaged over the four signals
    within a date before any t statistic is read, because same-day signals share the market
    window and a naive t would be inflated by roughly sqrt(4).
    """
    ei = ei or _ei()
    per_date = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    refusals = defaultdict(list)
    refusal_counts = defaultdict(lambda: defaultdict(int))
    per_column_sections = defaultdict(int)

    for item in captured:
        column, frame, labels = item['column'], item['frame'], item['labels']
        base_order, base_labels = _base_book(frame, labels, top_n=top_n)
        if base_order is None:
            refusals[column].append('base_book_too_thin')
            refusal_counts[column]['base_book_too_thin'] += 1
            continue
        per_column_sections[column] += 1
        date = str(item.get('date') or '')
        feature = frame[column].to_numpy(dtype=float)[base_order]
        base_ret = float(np.mean(base_labels))
        for mode in modes:
            w_after, turnover = _overlay_weights(base_labels, feature, mode)
            if w_after is None:
                refusals[column].append(mode + ':overlay_too_thin')
                refusal_counts[column][mode + ':overlay_too_thin'] += 1
                continue
            overlay_ret = float(np.sum(w_after * base_labels))
            delta_gross = overlay_ret - base_ret
            delta_net = delta_gross - (cost_bps / 1e4) * turnover
            slot = per_date[column][mode]
            slot['delta_gross'].append(delta_gross)
            slot['delta_net'].append(delta_net)
            slot['turnover'].append(turnover)
            if date:
                per_date[column]['__date__'].setdefault(date, []).append(
                    (mode, delta_net))

    out = {}
    # every column that entered the loop must leave a record: a column that vanished from the
    # output would be indistinguishable from one that was never requested
    for column in sorted({item['column'] for item in captured}):
        modes_d = per_date.get(column, {})
        # the count must be this column's own sections; a running total over all columns
        # would report the same number for every feature and overstate coverage
        base_sections = per_column_sections[column]
        entry = {'column': column,
                 'base_book_sections': base_sections,
                 'cost_bps': cost_bps}
        by_date = modes_d.get('__date__', {})
        for mode in modes:
            slot = modes_d.get(mode)
            read = len(slot['delta_net']) if slot else 0
            # the denominator travels with every reading: "348 sections examined, 92 read"
            # must never be flattenable into "348 sections read"
            denom = {'base_book_sections': base_sections,
                     'sections_read': read,
                     'sections_refused': base_sections - read,
                     'refusals': refusal_counts[column].get(mode + ':overlay_too_thin', 0)}
            if not slot or not slot['delta_net']:
                # only this mode's own refusals: a merged list reads as if both modes failed
                # for the same reason when it is being read to explain one of them
                mine = sorted({r for r in refusals[column]
                               if not r.endswith(':overlay_too_thin') or r.startswith(mode + ':')})
                entry[mode] = {'status': 'not_measured',
                               'reason': '; '.join(mine) or 'no rows',
                               **denom}
                continue
            # cluster by date: one number per date, then a t statistic over dates
            clustered = []
            for date, pairs in by_date.items():
                vals = [v for m, v in pairs if m == mode]
                if vals:
                    clustered.append(float(np.mean(vals)))
            if len(clustered) < 2:
                entry[mode] = {'status': 'not_measured', 'reason': 'fewer than 2 clustered dates',
                               **denom}
                continue
            mean_delta = float(np.mean(clustered))
            sd = float(np.std(clustered, ddof=1))
            se = sd / np.sqrt(len(clustered))
            t = float(mean_delta / se) if se > 0 else float('nan')
            _m, lo, hi, _n = _mean_ci(ei, clustered)
            entry[mode] = {
                'status': 'measured' if len(clustered) >= MIN_DATES else 'provisional',
                'dates': len(clustered),
                **denom,
                'delta_net_mean': mean_delta,
                'delta_net_ci': [float(lo), float(hi)],
                't_clustered_by_date': t,
                'delta_gross_mean': float(np.mean(slot['delta_gross'])),
                'turnover_mean': float(np.mean(slot['turnover'])),
                'positive_dates': int(sum(1 for v in clustered if v > 0)),
            }
        entry['refusals'] = sorted(set(refusals[column]))
        entry['refusals_are_counted_per_mode_below'] = True
        if base_sections == 0:
            entry['status'] = 'rejected_before_any_reading'
            entry['why'] = ('no section produced a base book for this column: %s'
                            % (sorted(set(refusals[column])) or ['no captured rows'],))
        out[column] = entry
    return out


def measure(columns=None, out_dir=OUT):
    """Full run: capture the frozen sections, then read the research口径 on the registry.

    The frozen external-info builder only exposes some families, so the declared mapping is
    split into measurable columns and columns this framework cannot see. The second group is
    reported as `not_measured_in_this_framework` with a pointer to the batch that did measure
    it — "cannot be read here" is not the same claim as "was measured and was low".
    """
    registry = load()
    declared = declared_columns(registry)
    ei = _ei()
    available = {c for cols in ei.FAMILY_COLUMNS.values() for c in cols}
    if columns is None:
        columns = sorted({c for cols in declared.values() for c in cols} & available)
    not_here = {name: [c for c in cols if c not in available]
                for name, cols in declared.items() if any(c not in available for c in cols)}

    started = datetime.now(timezone.utc).astimezone()
    captured, frozen = capture_frames(set(columns), ei=ei)
    captured_columns = sorted({item['column'] for item in captured})
    assert set(captured_columns) == set(columns), (
        'declared columns and captured columns disagree: %r vs %r' % (columns, captured_columns))
    readings = incremental_net_return(captured, ei=ei)

    per_feature = {}
    for name, cols in declared.items():
        here = [c for c in cols if c in available]
        readable = [c for c in here if (readings.get(c) or {}).get('exit_filter', {}).get(
            'status') in ('measured', 'provisional')
            or (readings.get(c) or {}).get('tilt', {}).get('status') in ('measured', 'provisional')]
        rejected = [c for c in here
                    if (readings.get(c) or {}).get('status') == 'rejected_before_any_reading']
        if not here:
            status = 'not_readable_here'
        elif readable:
            status = 'measured'
        elif rejected and len(rejected) == len(here):
            # distinct from "not_yet_usable": nothing was read because no section yielded a
            # base book. Reported separately rather than folded into a single empty verdict.
            status = 'rejected_before_any_reading'
        else:
            status = 'not_yet_usable'
        per_feature[name] = {
            'declared_columns': cols,
            'measurable_columns': here,
            'readable_columns': readable,
            'readings': {c: readings.get(c) for c in here},
            'status': status,
            'not_readable_here': [c for c in cols if c not in available],
        }

    payload = {
        'generated_at': started.isoformat(timespec='seconds'),
        'cost_bps': COST_BPS, 'top_n': TOP_N,
        'base_portfolio': 'the existing composite top-%d, equal weighted (fixed, not tuned)' % TOP_N,
        'columns_requested': columns,
        'columns_available_from_the_frozen_builder': sorted(available),
        'sections_captured': len(captured),
        'frozen_rows': len(frozen['rows']),
        'frozen_drops': len(frozen['drops']),
        'evaluate_calls': frozen['evaluate_calls'],
        'readings': readings,
        'per_feature': per_feature,
        'not_readable_here': not_here,
        'claims_not_made': [
            'no tier change grants promotion',
            'a research feature with a measured increment is still not promotable',
            'readings are in-sample with respect to the research-exposed window',
            'a column reported as not_readable_here was not measured here; that is not a '
            'statement about its magnitude',
            'rejected_before_any_reading means no section produced a base book, not that the '
            'feature was empty or that its effect was zero',
            'a negative incremental net return is a reading on the pre-registered direction; it '
            'is not by itself permission to run the feature inverted',
        ],
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + '\n'
    (out_dir / 'summary.json').write_bytes(text.encode('utf-8'))
    return payload


# ------------------------------------------------------------------ tests

def _fake_frame(n=60, seed=7):
    import pandas as pd
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({
        'ts_code': ['T%03d' % i for i in range(n)],
        'label': rng.normal(0.01, 0.05, n),
        'existing': rng.normal(0, 1, n),
        'industry': ['A'] * n,
        'log_size': rng.normal(10, 1, n),
    })
    f.attrs['date'] = '2026-01-05'
    return f


def tests():
    import tempfile
    results = []

    def check(name, fn):
        try:
            fn()
            results.append((name, 'ok', ''))
        except Exception as exc:                                   # noqa: BLE001
            results.append((name, 'FAIL', '%s: %s' % (type(exc).__name__, exc)))

    def t_research_feature_cannot_be_promoted():
        reg = _empty_registry()
        reg['tiers'][RESEARCH].append(dict(name='x', family='f', usable_as=['exit_filter'],
                                          why_not_promotable='r', evidence=['e'],
                                          promotion_eligible=False))
        assert not promotion_allowed('x', registry=reg)
        try:
            assert_promotable('x', registry=reg)
        except NotPromotableError:
            return
        raise AssertionError('gate did not refuse a research feature')

    def t_unregistered_names_are_unaffected():
        reg = _empty_registry()
        assert promotion_allowed('anything_else', registry=reg)
        assert_promotable('anything_else', registry=reg)

    def t_registry_never_grants():
        assert registry_grants_promotion() is False
        reg = load()
        assert reg['policy']['grants_promotion'] is False

    def t_promotion_entry_alone_does_not_grant():
        reg = _empty_registry()
        reg['tiers'][PROMOTION].append(dict(name='y', s5_evidence={'kind': 's5',
                                                                   'excess_25': 1.0}))
        # a promotion_candidate entry is a record, not a grant: the gate only ever subtracts
        assert promotion_allowed('y', registry=reg) is True
        assert registry_grants_promotion() is False

    def t_one_way_never_demotes_an_existing_promotion():
        reg = _empty_registry()
        reg['tiers'][PROMOTION].append(dict(name='bt_premium', s5_evidence={'kind': 's5'}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reg.json'
            save(reg, path)
            got = seed(load(path), path)
        assert 'bt_premium' in got['refused_one_way'], got
        assert all(e['name'] != 'bt_premium' for e in load(path)['tiers'][RESEARCH])

    def t_validate_flags_a_promotable_research_feature():
        reg = _empty_registry()
        reg['tiers'][RESEARCH].append(dict(name='z', family='f', usable_as=['exit_filter'],
                                           why_not_promotable='r', evidence=['e'],
                                           promotion_eligible=True))
        rep = validate(reg)
        assert not rep['ok']
        assert any(i['kind'] == 'research_feature_marked_promotable' for i in rep['issues'])

    def t_validate_flags_two_tiers():
        reg = _empty_registry()
        row = dict(name='dup', family='f', usable_as=['exit_filter'], why_not_promotable='r',
                   evidence=['e'], promotion_eligible=False)
        reg['tiers'][RESEARCH].append(row)
        reg['tiers'][PROMOTION].append(dict(name='dup', s5_evidence={'kind': 's5'}))
        assert any(i['kind'] == 'name_in_two_tiers' for i in validate(reg)['issues'])

    def t_standalone_ic_is_refused_as_evidence():
        for kind in ('standalone_ic', 'rank_ic', 'ic'):
            try:
                standalone_ic_is_not_promotion_evidence({'kind': kind})
            except ValueError:
                continue
            raise AssertionError('accepted %s' % kind)
        standalone_ic_is_not_promotion_evidence({'kind': 'incremental_net_return'})

    def t_overlay_costs_turnover_and_nothing_is_imputed():
        frame, labels = _fake_frame(), None
        labels = frame.label.to_numpy(dtype=float)
        feature = np.full(len(frame), np.nan)
        feature[:10] = -5.0                       # a clean worst decile to filter out
        w, turnover = _overlay_weights(labels, feature, 'exit_filter')
        assert w is not None and turnover > 0
        assert abs(w.sum() - 1.0) < 1e-12, 'overlay must remain a fully invested book'
        assert abs(turnover - 10.0 / 60.0) < 1e-9, turnover
        w2, turn2 = _overlay_weights(labels, np.full(len(frame), np.nan), 'exit_filter')
        assert w2 is None, 'a feature with no values must refuse, never impute'

    def t_net_is_below_gross_by_the_charged_cost():
        labels = np.linspace(-0.02, 0.04, 60)
        feature = np.concatenate([np.full(12, -9.0), np.full(48, np.nan)])
        w, turnover = _overlay_weights(labels, feature, 'exit_filter')
        base = labels.mean()
        overlay = float(np.sum(w * labels))
        net = (overlay - base) - (COST_BPS / 1e4) * turnover
        assert net < (overlay - base)

    def t_clustering_averages_within_a_date():
        # four sections on two dates: 4 rows, 2 dates. Same-day signals must not quadruple n.
        cap = []
        for day in ('2026-01-05', '2026-01-06'):
            for seed in (7, 8):
                f = _fake_frame(seed=seed)
                v = np.full(len(f), np.nan)
                v[:12] = -9.0
                f['c'] = v
                cap.append({'column': 'c', 'date': day, 'frame': f,
                            'labels': f.label.to_numpy(dtype=float)})

        class _StubEi:
            @staticmethod
            def mean_ci(v):
                v = np.asarray(v, dtype=float)
                return float(v.mean()), float(v.mean()), float(v.mean()), len(v)

        out = incremental_net_return(cap, ei=_StubEi())
        got = out['c']['exit_filter']
        assert got['status'] == 'provisional', got    # 2 dates < MIN_DATES
        assert got['sections_read'] == 4, got
        assert got['sections_refused'] == 0, got
        assert got['dates'] == 2, 'sections were not collapsed to dates: %r' % (got,)
        assert np.isnan(got['t_clustered_by_date']), 'a flat stub gives no dispersion'
        assert np.isclose(got['turnover_mean'], 1.0 / 5.0), got['turnover_mean']

    def t_a_single_date_cannot_be_read_at_all():
        cap = []
        for seed in (7, 8, 9):
            f = _fake_frame(seed=seed)
            v = np.full(len(f), np.nan)
            v[:12] = -9.0
            f['c'] = v
            cap.append({'column': 'c', 'date': '2026-01-05', 'frame': f,
                        'labels': f.label.to_numpy(dtype=float)})

        class _StubEi:
            @staticmethod
            def mean_ci(v):
                v = np.asarray(v, dtype=float)
                return float(v.mean()), float(v.mean()), float(v.mean()), len(v)

        got = incremental_net_return(cap, ei=_StubEi())['c']['exit_filter']
        assert got['status'] == 'not_measured'
        assert got['sections_read'] == 3
        assert 'fewer than 2 clustered dates' in got['reason'], got

    def t_captured_frame_carries_the_evaluated_column():
        # regression guard: the captured copy must still be evaluable. Keeping only the
        # structural fields produced frames that blew up later inside
        # incremental_net_return -- a non-empty input that yielded nothing, which is the
        # same shape of defect as the two pass-parameter bugs already recorded.
        f = _fake_frame(seed=11)
        f['alpha'] = np.linspace(-1.0, 1.0, len(f))

        class _StubEi:
            FAMILY_COLUMNS = {'fam': ['alpha']}

            @staticmethod
            def evaluate_factor(frame, labels, column, resid_base, resid_full, existing):
                return {'factor': column,
                        'coverage': float(np.isfinite(frame[column]).mean())}

            @staticmethod
            def mean_ci(v):
                v = np.asarray(v, dtype=float)
                return float(v.mean()), float(v.mean()), float(v.mean()), len(v)

            @staticmethod
            def rebuild_sections():
                rows = []
                for day in ('2026-01-05', '2026-01-06'):
                    rows.append(dict(_StubEi.evaluate_factor(
                        f, f.label.to_numpy(dtype=float), 'alpha', None, None,
                        f.existing.to_numpy(dtype=float)), date=day, signal='s'))
                return rows, [], {}, {}

        captured, frozen = capture_frames({'alpha'}, ei=_StubEi)
        assert len(captured) == 2, captured
        assert frozen['evaluate_calls'] == 2, frozen
        cols = list(captured[0]['frame'].columns)
        assert 'alpha' in cols, 'the captured frame lost the column it was captured for: %r' % cols
        got = incremental_net_return(captured, ei=_StubEi)['alpha']['exit_filter']
        assert got['status'] == 'provisional', got   # 2 dates < MIN_DATES, but it did read
        assert got['sections_read'] == 2, got

    def t_mean_ci_shape_is_asserted_not_unpacked_blindly():
        # the frozen module returns (mean, lo, hi, n). A two-value stub used to survive every
        # test and only explode at the end of a two-minute production run.
        class _Bad:
            @staticmethod
            def mean_ci(v):
                v = np.asarray(v, dtype=float)
                return float(v.mean()), float(v.mean())

        class _Good:
            @staticmethod
            def mean_ci(v):
                v = np.asarray(v, dtype=float)
                return float(v.mean()), float(v.mean()), float(v.mean()), len(v)

        for stub, should_raise in ((_Bad, True), (_Good, False)):
            try:
                _mean_ci(stub, [1.0, 2.0, 3.0])
            except AssertionError:
                assert should_raise, 'a correct 4-tuple was rejected'
            else:
                assert not should_raise, 'a wrong-shaped mean_ci was accepted'

    def t_sections_are_counted_per_column_not_globally():
        # a running total over all columns printed the same section count for every feature,
        # which silently overstated coverage for the thin ones
        def _frame_with(seed, col):
            f = _fake_frame(seed=seed)
            v = np.full(len(f), np.nan)
            v[:12] = -9.0
            f[col] = v
            return f

        class _StubEi:
            @staticmethod
            def mean_ci(v):
                v = np.asarray(v, dtype=float)
                return float(v.mean()), float(v.mean()), float(v.mean()), len(v)

        cap = []
        for day, seed in (('2026-01-05', 7), ('2026-01-06', 8)):
            f = _frame_with(seed, 'a')
            cap.append({'column': 'a', 'date': day, 'frame': f,
                        'labels': f.label.to_numpy(dtype=float)})
        f = _frame_with(9, 'b')
        cap.append({'column': 'b', 'date': '2026-01-07', 'frame': f,
                    'labels': f.label.to_numpy(dtype=float)})

        out = incremental_net_return(cap, ei=_StubEi)
        assert out['a']['base_book_sections'] == 2, out['a']
        assert out['b']['base_book_sections'] == 1, out['b']
        assert out['b']['exit_filter']['sections_read'] == 1, out['b']['exit_filter']
        assert out['b']['exit_filter']['sections_refused'] == 0, out['b']['exit_filter']

    def t_refused_sections_stay_in_the_denominator():
        # "348 sections examined, 92 read" must never be flattenable into "348 read"
        class _StubEi:
            @staticmethod
            def mean_ci(v):
                v = np.asarray(v, dtype=float)
                return float(v.mean()), float(v.mean()), float(v.mean()), len(v)

        cap = []
        for day, seed, dense in (('2026-01-05', 7, True), ('2026-01-06', 8, False)):
            f = _fake_frame(seed=seed)
            v = np.full(len(f), np.nan)
            if dense:
                v[:12] = -9.0
            f['c'] = v
            cap.append({'column': 'c', 'date': day, 'frame': f,
                        'labels': f.label.to_numpy(dtype=float)})

        got = incremental_net_return(cap, ei=_StubEi)['c']['exit_filter']
        assert got['base_book_sections'] == 2, got
        assert got['sections_read'] == 1, got
        assert got['sections_refused'] == 1, got
        assert got['refusals'] == 1, got

    def t_a_fully_rejected_column_still_appears():
        # a column that never yields a base book used to vanish from the output entirely,
        # which is indistinguishable from a column that was never asked for
        class _StubEi:
            @staticmethod
            def mean_ci(v):
                v = np.asarray(v, dtype=float)
                return float(v.mean()), float(v.mean()), float(v.mean()), len(v)

        thin = _fake_frame(n=8, seed=7)          # below MIN_BASE, so no base book exists
        thin['thin'] = 1.0
        fat = _fake_frame(seed=8)
        v = np.full(len(fat), np.nan)
        v[:12] = -9.0
        fat['fat'] = v
        cap = [
            {'column': 'thin', 'date': '2026-01-05', 'frame': thin,
             'labels': thin.label.to_numpy(dtype=float)},
            {'column': 'fat', 'date': '2026-01-05', 'frame': fat,
             'labels': fat.label.to_numpy(dtype=float)},
        ]
        out = incremental_net_return(cap, ei=_StubEi)
        assert set(out) == {'thin', 'fat'}, sorted(out)
        assert out['thin']['status'] == 'rejected_before_any_reading', out['thin']
        assert out['thin']['base_book_sections'] == 0, out['thin']
        assert 'base_book_too_thin' in out['thin']['refusals'], out['thin']
        assert out['fat']['base_book_sections'] == 1, out['fat']
        assert out['fat']['exit_filter']['sections_read'] == 1, out['fat']['exit_filter']

    def t_seed_is_idempotent():
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reg.json'
            save(_empty_registry(), path)
            first = seed(load(path), path)
            second = seed(load(path), path)
            assert len(first['added']) == len(SEED_RESEARCH)
            assert second['added'] == [], second
            assert second['research_features'] == len(SEED_RESEARCH)

    def t_generator_evidence_is_transcribed_and_flagged_unstructured():
        ev = parse_generator_evidence()
        assert ev, 'no evidence parsed from the JQ generator'
        for name, rec in ev.items():
            assert rec['structured'] is False, name
            assert rec['parsed_from'].endswith('.local_note'), rec['parsed_from']
            assert rec['kind'] == 's5_readings'
        assert any('excess_25' in r for r in ev.values()), 'no S5 numbers extracted'

    def t_the_two_s5_record_systems_are_disjoint():
        ledger = historical_pass_ledger()
        assert ledger, 'the JSON ledger is missing'
        assert all(r['structured'] is True for r in ledger.values())
        overlap = set(ledger) & set(jq_registered_names())
        assert overlap == set(), 'unexpected overlap: %r' % sorted(overlap)

    def t_sync_records_evidence_but_never_membership():
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reg.json'
            save(_empty_registry(), path)
            before = load(path)
            assert before['tiers'][PROMOTION] == []
            got = sync_promotion_evidence(load(path), path)
            after = load(path)
            assert got['jq_registered'] == len(jq_registered_names())
            assert len(after['tiers'][PROMOTION]) == got['jq_registered']
            # membership came from observing the endpoint, not from the registry deciding
            assert got['crosscheck']['structured_evidence'] == 0
            assert got['crosscheck']['unstructured_evidence'] == got['with_evidence']

    def t_duplicate_membership_cannot_unblock_a_research_feature():
        # the failure this guards against: promotion overwriting research in the index,
        # which made the gate answer ALLOW for a research feature (silent, no error).
        reg = _empty_registry()
        row = dict(name='both', family='f', usable_as=['exit_filter'],
                   why_not_promotable='r', evidence=['e'], promotion_eligible=False)
        reg['tiers'][RESEARCH].append(row)
        reg['tiers'][PROMOTION].append(dict(name='both', s5_evidence={'kind': 's5'}))
        assert blocked('both', registry=reg) is True
        assert promotion_allowed('both', registry=reg) is False
        try:
            assert_promotable('both', registry=reg)
        except NotPromotableError:
            pass
        else:
            raise AssertionError('a promotion row unblocked a research feature')

    def t_sync_reports_a_conflict_instead_of_resolving_it():
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reg.json'
            reg = _empty_registry()
            victim = jq_registered_names()[0]
            reg['tiers'][RESEARCH].append(dict(name=victim, family='f',
                                              usable_as=['exit_filter'],
                                              why_not_promotable='r', evidence=['e'],
                                              promotion_eligible=False))
            save(reg, path)
            got = sync_promotion_evidence(load(path), path)
            assert any(c['name'] == victim for c in got['conflicts']), got['conflicts']
            view = promotion_view(load(path))
            assert view['blocked_by_registry'] == 1
            # the endpoint file is untouched: the registry reports, it does not rewrite
            assert victim in jq_registered_names()

    for name, fn in sorted(
            (k, v) for k, v in locals().items() if k.startswith('t_') and callable(v)):
        check(name, fn)
    ok = sum(1 for _, s, _ in results if s == 'ok')
    return {'passed': ok, 'total': len(results),
            'results': [{'test': n, 'status': s, 'error': e} for n, s, e in results]}


# ------------------------------------------------------------------ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('command', choices=['validate', 'gate', 'promotion-view', 'seed',
                                        'sync-evidence', 'measure', 'tests', 'show'])
    ap.add_argument('name', nargs='?')
    ap.add_argument('--registry', default=str(REGISTRY))
    args = ap.parse_args(argv)
    path = Path(args.registry)

    if args.command == 'tests':
        rep = tests()
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0 if rep['passed'] == rep['total'] else 1
    if args.command == 'seed':
        print(json.dumps(seed(load(path), path), ensure_ascii=False, indent=2))
        return 0
    if args.command == 'sync-evidence':
        print(json.dumps(sync_promotion_evidence(load(path), path), ensure_ascii=False,
                         indent=2))
        return 0
    if args.command == 'validate':
        rep = validate(load(path))
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0 if rep['ok'] else 1
    if args.command == 'gate':
        if not args.name:
            raise SystemExit('gate requires a name')
        try:
            assert_promotable(args.name, registry=load(path))
            print('ALLOW  %s (the registry does not object; promotion is still S5-only)' % args.name)
            return 0
        except NotPromotableError as exc:
            print('REFUSE %s\n  %s' % (args.name, exc))
            return 2
    if args.command == 'promotion-view':
        print(json.dumps(promotion_view(load(path)), ensure_ascii=False, indent=2))
        return 0
    if args.command == 'measure':
        payload = measure()
        print(json.dumps({k: payload[k] for k in
                          ('columns_requested', 'sections_captured', 'frozen_rows',
                           'evaluate_calls')}, ensure_ascii=False, indent=2))
        print(json.dumps(payload['readings'], ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(load(path), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
