# -*- coding: utf-8 -*-
"""P3: 上游入口可交易性检查 —— 读数、A/B 反事实、四关验收、接线点清单。

背景
----
`reports/fuxi_event_inventory_plan_20260924.md` 第 4 节尾部首次记录:
    冻结目标曾在不可交易日建仓 —— llm1 有 11 个、mem 有 2 个入场事件落在无行情行日,
    合计 13 / 13701。属上游入口可交易性检查缺口, 应在上游修, 不该在账本层吸收。
该读数当时**没有留下任何脚本**, 本批先把口径重建到逐位可复算, 再把门做出来。

口径重建 (与 `_probe_entry_during_halt_20260924.py` 逐位一致)
------------------------------------------------------------
  * 入场事件 = 每个 (臂, 证券) **连续持有段的首个交易日**; 同一证券退出后再进算两次;
  * 入场事件的交易日 = 目标 stamp 之后**第一个开市日** (目标自次日起生效);
  * 无行情行 = `(证券, 生效日)` 不在 `data/raw/daily_prices.csv` 里;
  * 全窗 = 2024-12-27..2026-09-09, 四臂各 88 个生效 stamp。

本批交付
--------
1. `research/factor_alchemy/entry_tradability.py` —— 门本体 (8 类穷尽互斥判定, 22 项单测);
2. 本脚本 —— 四关验收读数 + A/B 反事实 + 接线点清单 + 历史记录对账。

**本批不接进任何生产调用点** (见 `site_inventory`): 冻结构建器
(`scripts/fuxi_execution_lab_20260923.py`) 已被 `protocol.json` 逐字节钉死, 改它会使
封存产物对不上; 其余 6 个调用点属既有批次产物, 改它们等于改那些批次的读数。
接线需要**新协议 / 新批次**, 本批只把"缺口已可判定、可计数、可逐条列举"这一步做实。

只读封存/本地文件; 只写 `reports/entry_tradability_20260924/`。
"""

import argparse
import hashlib
import io
import json
import sys
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))
import entry_tradability as ET  # noqa: E402

OUT = ROOT / 'reports/entry_tradability_20260924'
TARGETS_DIR = ROOT / 'reports/execution_ledger_20260923/targets'

WINDOW_LO = '2024-12-27'
WINDOW_HI = '2026-09-09'
ARMS = ['r4_equal_rank', 'llm_family_1_1_fresh_llm', 'llm_family_2_1_fresh_llm',
        'mem_0626_broadcast_repaired']

# 端到端对账用的账本世代, 按"越新越全"排序; 取第一个四臂齐全的世代并如实署名。
LEDGER_GENERATIONS = [
    'reports/ledger_advance4_20260924/D_reproduce_old_suspend',
    'reports/ledger_advance3_20260924/D_halt_session_vacuous',
    'reports/ledger_advance2_20260924',
]

# 上游把证券写进目标的所有调用点。逐条在运行时回读源码校验 (行号漂移即报错),
# 不靠 grep 推断 —— 字面白名单会漏变体, 行号断言能在漂移时马上失败。
SITES = [
    {'file': 'scripts/fuxi_execution_lab_20260923.py', 'line': 252,
     'pattern': 'selected = group.sort_values(',
     'kind': 'frozen_protocol_builder',
     'note': '被封存 protocol.json 与 code_hashes 逐字节钉死; 接线需新协议'},
    {'file': 'scripts/fuxi_external_info_20260924.py', 'line': 422,
     'pattern': 'ascending=[False, False, True]).head(TOP_N)',
     'kind': 'research_book_builder',
     'note': 'P-20260924-009/010 批次产物; 改动会改该批读数'},
    {'file': 'scripts/fuxi_ic_ceiling_20260924.py', 'line': 493,
     'pattern': 'ascending=[False, False, True]).head(TOP_N)',
     'kind': 'research_book_builder',
     'note': 'P0 信息族 IC 上限批次; 同上'},
    {'file': 'scripts/fuxi_model_compare_20260924.py', 'line': 413,
     'pattern': 'ascending=[False, False, True]).head(TOP_N)',
     'kind': 'research_book_builder',
     'note': 'R4 公平模型对照批次; 同上'},
    {'file': 'scripts/fuxi_target_diagnostics_20260923.py', 'line': 30,
     'pattern': 'ascending=[False, False, True]).head(top_n)',
     'kind': 'diagnostic',
     'note': '只读诊断, 不改产物'},
    {'file': 'scripts/fuxi_target_neutral_attribution_20260923.py', 'line': 256,
     'pattern': 'ascending=[False, False, True]).head(TOP_N)',
     'kind': 'diagnostic',
     'note': '目标中性化归因批次; 同上'},
    {'file': 'scripts/fuxi_target_neutral_attribution_20260923.py', 'line': 288,
     'pattern': 'ascending=[False, False, True]).head(TOP_N)',
     'kind': 'diagnostic',
     'note': '目标中性化归因批次缓冲组合; 同上'},
    {'file': 'research/factor_alchemy/feature_tier.py', 'line': 554,
     'pattern': 'def _base_book(frame, labels, top_n=TOP_N):',
     'kind': 'measurement_base_book',
     'note': 'P2 研究特征库的固定基准组合; 它用 isfinite(label) 过滤, '
             '这是**标签侧**过滤, 不是生效日可交易性检查'},
]


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def dashed(value):
    value = str(value).replace('-', '')
    return '%s-%s-%s' % (value[:4], value[4:6], value[6:8])


def sessions():
    cal = pd.read_csv(ROOT / 'data/raw/trade_calendar.csv', dtype=str)
    cal = cal[(cal.exchange == 'SSE') & (cal.is_open == '1')]
    days = sorted(pd.to_datetime(cal.cal_date, format='mixed')
                  .dt.strftime('%Y-%m-%d').unique())
    return [d for d in days if WINDOW_LO <= d <= WINDOW_HI]


def load_frames(arm):
    payload = json.loads((TARGETS_DIR / (arm + '.json')).read_text(encoding='utf-8'))
    frames = {}
    for stamp, weights in payload['targets'].items():
        held = {code: float(w) for code, w in weights.items() if float(w) > 0}
        frames[stamp] = held
    return frames, payload


def sealed_bytes(arm):
    return (TARGETS_DIR / (arm + '.json')).read_bytes()


def serialise(payload):
    """与 `run_contract._atomic_json` 同式 —— 这样"逐字节相同"才是可证的。"""
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + '\n').encode('utf-8')


def needed_keys(frames, days):
    """门真正需要查的 (证券, 生效日) 键集合 —— 据此裁剪价格面板, 不全量载入。"""
    keys = {}
    for stamp in sorted(frames):
        session = ET.effective_session(stamp, days)
        if session is None:
            continue
        for code in frames[stamp]:
            keys.setdefault(code, set()).add(session)
    return keys


def build_panel(keys, days):
    """装配门需要的封存面板。只取需要的键, 不做任何填补。"""
    wanted = {(code, session) for code, ss in keys.items() for session in ss}
    codes = set(keys)
    priced = set()
    reader = pd.read_csv(ROOT / 'data/raw/daily_prices.csv',
                         usecols=['ts_code', 'trade_date'], dtype=str,
                         chunksize=500000)
    for chunk in reader:
        chunk = chunk[chunk.ts_code.isin(codes)]
        for code, raw in zip(chunk.ts_code, chunk.trade_date):
            key = (code, dashed(raw))
            if key in wanted:
                priced.add(key)

    susp = pd.read_csv(ROOT / 'data/raw/suspend_d.csv', dtype=str)
    susp = susp.assign(day=susp.trade_date.map(dashed))
    halt_full_all = susp[(susp.suspend_type == 'S') & susp.suspend_timing.fillna('').eq('')]
    halt_partial_all = susp[(susp.suspend_type == 'S') & ~susp.suspend_timing.fillna('').eq('')]
    halt_full = {(c, d) for c, d in zip(halt_full_all.ts_code, halt_full_all.day)
                 if (c, d) in wanted}
    halt_partial = {(c, d) for c, d in zip(halt_partial_all.ts_code, halt_partial_all.day)
                    if (c, d) in wanted}
    status_days = set(susp.day)

    delist = pd.read_csv(ROOT / 'data/raw/delist_dates.csv', dtype=str)
    delist = delist[delist.list_status.fillna('').eq('D') & delist.delist_date.notna()]
    delist_dates = {c: dashed(d) for c, d in zip(delist.ts_code, delist.delist_date)}

    panel = {'priced': priced, 'halt_full': halt_full, 'halt_partial': halt_partial,
             'status_days': status_days, 'status_terminal': max(status_days),
             'sessions': set(days), 'delist_dates': delist_dates}
    panel_meta = {
        'wanted_keys': len(wanted),
        'priced_keys_found': len(priced),
        'halt_full_keys_found': len(halt_full),
        'halt_partial_keys_found': len(halt_partial),
        'status_table_terminal': max(status_days),
        'status_table_rows': int(len(susp)),
        'status_table_full_day_s_rows': int(len(halt_full_all)),
        'status_table_partial_s_rows': int(len(halt_partial_all)),
        'delisted_securities': len(delist_dates),
        'panel_is_restricted_to_the_asked_keys': True,
    }
    return panel, panel_meta


def price_extents(codes, want_full_for=()):
    """一次扫描同时拿到: 每只证券的**行情序列末端** + 指定证券的完整有价日清单。

    末端用来判"目标买进了一只行情序列已经结束的证券"; 完整清单用来做门的对照臂。
    """
    last = {}
    full = {c: set() for c in want_full_for}
    reader = pd.read_csv(ROOT / 'data/raw/daily_prices.csv',
                         usecols=['ts_code', 'trade_date'], dtype=str, chunksize=500000)
    for chunk in reader:
        chunk = chunk[chunk.ts_code.isin(codes)]
        for code, raw in zip(chunk.ts_code, chunk.trade_date):
            day = dashed(raw)
            if day > last.get(code, ''):
                last[code] = day
            if code in full:
                full[code].add(day)
    return last, {c: sorted(v) for c, v in full.items()}


def ledger_orders():
    """端到端对账: 取第一个四臂齐全的账本世代, 并记录选择了哪个。"""
    for base in LEDGER_GENERATIONS:
        folder = ROOT / base / 'ledgers'
        if not folder.is_dir():
            continue
        if all((folder / (a + '.json')).exists() for a in ARMS):
            out = {}
            for arm in ARMS:
                payload = json.loads((folder / (arm + '.json')).read_text(encoding='utf-8'))
                days = payload.get('days') or []
                out[arm] = {
                    'generation': base,
                    'days': len(days),
                    'first_day': days[0]['date'] if days else None,
                    'last_day': days[-1]['date'] if days else None,
                    'status': payload.get('status'),
                    'orders': payload.get('orders') or [],
                }
            return out
    raise RuntimeError('no ledger generation with all four arms was found')


def site_inventory():
    rows = []
    for site in SITES:
        path = ROOT / site['file']
        text = path.read_text(encoding='utf-8').splitlines()
        if site['line'] > len(text):
            raise AssertionError('%s has only %d lines, site claims %d'
                                 % (site['file'], len(text), site['line']))
        actual = text[site['line'] - 1]
        if site['pattern'] not in actual:
            raise AssertionError('site drifted: %s:%d no longer contains %r (got %r)'
                                 % (site['file'], site['line'], site['pattern'], actual))
        rows.append({**site, 'line_text': actual.strip()[:120],
                     'has_entry_tradability_check': False})
    return {
        'note': '逐条回读源码校验行号与模式; 漂移即抛错, 不用 grep 推断。'
                '"有没有门"按实现判, 不按文件名/注释判。',
        'sites_total': len(rows),
        'sites_with_an_entry_tradability_check': 0,
        'sites_without_an_entry_tradability_check': len(rows),
        'sites': rows,
        'only_reader_that_needs_no_wiring':
            'research/factor_alchemy/entry_tradability.py 本身: 它是纯函数库, 无宿主。',
        'wiring_requires': {
            'frozen_protocol_builder': '新协议 (protocol.json 与 code_hashes 逐字节钉死, '
                                       '改它会使封存产物对不上)',
            'research_book_builder': '新批次 (改它等于改 P0/P1/R4 已发布批次的读数)',
            'diagnostic': '可在下一批一并接, 只读, 不影响封存产物',
            'measurement_base_book': '随 P2 的下一批 (它已有标签侧过滤, 缺的是生效日侧)',
        },
    }


def historical_record_reconciliation(days, entry_events):
    """旧记录写的是 "13 个中 11 个落在停牌日"。本批复算是 13/13, 差异必须解释清楚。"""
    backup = ROOT / 'data/raw/suspend_d.csv.bak_20260924_104558'
    current = pd.read_csv(ROOT / 'data/raw/suspend_d.csv', dtype=str)
    cur_rows = {(c, dashed(d)) for c, d in zip(current.ts_code, current.trade_date)}
    out = {
        'recorded': 'llm1 11 + mem 2 = 13/13701 入场事件落在无行情行日, "13 个中 11 个落在停牌日"',
        'recomputed_entry_events': len(entry_events),
        'recomputed_without_a_price_row': len(entry_events),
    }
    if backup.exists():
        old = pd.read_csv(backup, dtype=str)
        old_rows = {(c, dashed(d)) for c, d in zip(old.ts_code, old.trade_date)}
        old_span = [str(old.trade_date.min()), str(old.trade_date.max())]
        covered_old = [e for e in entry_events if (e['code'], e['session']) in old_rows]
        out.update({
            'local_status_table_backup': str(backup.relative_to(ROOT)),
            'status_table_span_when_the_record_was_written': old_span,
            'entry_events_covered_by_that_table': len(covered_old),
            'entry_events_outside_that_table': [
                {'code': e['code'], 'session': e['session']}
                for e in entry_events if (e['code'], e['session']) not in old_rows],
            'explanation':
                '旧记录里的 "11" 是**当时的停牌表末端 (2026-04-30) 造成的覆盖数字**, '
                '不是判定分歧: 有 2 例 (2026-05-06) 落在表外, 因此当时既不能证停牌、'
                '也不能证成交。停牌表已于 2026-09-24 扩到 2026-09-23, 复算这 2 例'
                '同样是 suspend_type=S 的整日停牌 ⇒ 现读数 13/13, 且**11 与 13 都是'
                '对的, 只是分母不同**。不得只说其中任何一个。',
        })
    out['recomputed_covered_by_current_table'] = len(
        [e for e in entry_events if (e['code'], e['session']) in cur_rows])
    out['current_table_span'] = [str(current.trade_date.min()), str(current.trade_date.max())]
    return out, cur_rows


def readings():
    days = sessions()
    panel = None
    frames_by_arm = {}
    per_arm = {}
    for arm in ARMS:
        frames, payload = load_frames(arm)
        frames_by_arm[arm] = frames
        per_arm[arm] = {'stamps': len(frames),
                        'names_per_stamp': sorted({len(v) for v in frames.values()}),
                        'sealed_top_n': payload.get('top_n'),
                        'targets_insufficient': len(payload.get('insufficient') or [])}
    keys = {}
    for arm in ARMS:
        for code, ss in needed_keys(frames_by_arm[arm], days).items():
            keys.setdefault(code, set()).update(ss)
    panel, panel_meta = build_panel(keys, days)

    # ---------------- 关一: getter 实测 ----------------
    events_by_arm = {arm: ET.entry_events(frames_by_arm[arm], days) for arm in ARMS}
    total_events = sum(len(v) for v in events_by_arm.values())
    skipped = {}
    for arm in ARMS:
        found = ET.skipped_stamps(frames_by_arm[arm], days)
        if found:
            skipped[arm] = found
    stamp_of = {}
    for arm in ARMS:
        for stamp in sorted(frames_by_arm[arm]):
            session = ET.effective_session(stamp, days)
            if session is None:
                continue
            for code in frames_by_arm[arm][stamp]:
                stamp_of[(arm, code, session)] = stamp
    no_price = []
    for arm in ARMS:
        for event in events_by_arm[arm]:
            if (event['code'], event['session']) not in panel['priced']:
                no_price.append({'arm': arm, 'stamp': stamp_of[(arm, event['code'],
                                                                event['session'])],
                                 **event})
    gate1 = {'what_it_proves': '门是不是真的在按封存数据判定, 而不是在返常量',
             'entry_events_total': total_events,
             'entry_events_without_a_price_row': len(no_price),
             'stamps_whose_target_never_takes_effect_inside_the_calendar': skipped,
             'stamps_skipped_count': sum(len(v) for v in skipped.values()),
             'skipped_stamp_note': '探针原先静默跳过这些 stamp (13701 不含其持有日); '
                                   '本批逐条列出, 不再让跳过不可见。',
             'by_arm': {arm: {'entry_events': len(events_by_arm[arm]),
                              'without_a_price_row': len(
                                  [e for e in no_price if e['arm'] == arm])}
                        for arm in ARMS}}
    # 门的判定 vs 直接查价格面板, 两类必须逐个一致
    mismatched = []
    for row in no_price:
        got = ET.certify(panel, row['code'], row['session'])
        row['verdict'] = got['verdict']
        row['category'] = got['category']
        if got['admitted']:
            mismatched.append({**row, 'why': 'no price row but the gate admitted it'})
    for arm in ARMS:
        for event in events_by_arm[arm]:
            if (event['code'], event['session']) in panel['priced']:
                got = ET.certify(panel, event['code'], event['session'])
                if not got['admitted']:
                    mismatched.append({'arm': arm, **event, 'verdict': got['verdict'],
                                       'why': 'a price row exists but the gate refused it'})
    gate1['gate_verdicts_for_the_no_price_row_events'] = dict(
        Counter(r['verdict'] for r in no_price))
    gate1['direct_lookup_and_gate_disagreements'] = mismatched
    gate1['disagreements_count'] = len(mismatched)
    assert not mismatched, mismatched

    # 被拒准入的完整台账 (annotate 档)。关一的归因附读与最终落盘都用这一份,
    # 不重复跑门 —— 两处各自跑一遍会让"读到的是哪一次"变得不可说。
    all_rejects = []
    for arm in ARMS:
        got = ET.gate(frames_by_arm[arm], days, panel, mode='annotate')
        for row in got['rejects']:
            all_rejects.append({'arm': arm, **row})

    # 对照臂: 同一证券顺延到之后第一个有行情行的交易日 -> 必须判成可成交。
    # 只有一处与主读数不同: 把那个 (证券, 该日) 加进"有行情行"的集合。
    # 顺带对**全部被拒准入**(不只是被拒入场)做一次"行情序列是否已经结束"的归因,
    # 用同一趟扫描拿到每只证券的行情末端。
    refused_codes = sorted({r['code'] for r in all_rejects})
    control_codes = sorted({r['code'] for r in no_price})
    last_price, calendars = price_extents(set(refused_codes) | set(control_codes)
                                          | {r['code'] for r in all_rejects},
                                          set(control_codes))
    controls = []
    for row in no_price:
        span = calendars.get(row['code']) or []
        later = [d for d in span if d > row['session']]
        if not later:
            controls.append({**row, 'next_priced_session': None, 'verdict': None,
                             'control': 'the_price_series_itself_ends_here'})
            continue
        nxt = later[0]
        probe = dict(panel, priced=panel['priced'] | {(row['code'], nxt)})
        got = ET.certify(probe, row['code'], nxt)
        controls.append({**row, 'next_priced_session': nxt, 'verdict': got['verdict'],
                         'admitted': got['admitted'],
                         'control': 'only_difference_is_that_one_price_row'})
    gate1['control_arm'] = {
        'what_it_proves': '同一证券在之后第一个有行情行的交易日必须判成可成交 —— '
                          '若对照臂也判拒, 说明门返的是常量而不是数据',
        'cases': controls,
        'admitted': len([c for c in controls if c.get('admitted') is True]),
        'refused': len([c for c in controls if c.get('admitted') is False]),
        'feed_ended_so_no_control_possible':
            len([c for c in controls if c['next_priced_session'] is None]),
    }
    assert gate1['control_arm']['refused'] == 0, gate1['control_arm']
    ended = [r for r in all_rejects
             if last_price.get(r['code']) and (r['session'] or '') > last_price[r['code']]]
    gate1['refusals_where_the_price_series_had_already_ended'] = {
        'denominator': '全部被拒准入 (%d 条), 不只 13 个被拒入场事件' % len(all_rejects),
        'count': len(ended),
        'by_arm': dict(Counter(r['arm'] for r in ended)),
        'cases': [{'arm': r['arm'], 'code': r['code'], 'session': r['session'],
                   'stamp': r['stamp'], 'last_price_day': last_price.get(r['code']),
                   'verdict': r['verdict']} for r in ended],
        'note': '这些不是"那天刚好停牌", 而是该证券的本地行情序列**已经结束** —— '
                '目标却仍在下一个交易日把它买进。它们同样落进门的 refused 集合, '
                '但归因不同, 必须分开报。',
    }
    gate1['refusals_of_entry_events_where_the_price_series_had_already_ended'] = {
        'count': len([r for r in ended
                      if (r['code'], r['session']) in
                      {(x['code'], x['session']) for x in no_price}]),
    }

    # ---------------- 关二: A/B 反事实 ----------------
    modes = {}
    gated = {}
    for mode in ('off', 'annotate', 'enforce'):
        arm_out = {}
        for arm in ARMS:
            got = ET.gate(frames_by_arm[arm], days, panel, mode=mode,
                          outcome_label='P3 A/B arm=%s mode=%s' % (arm, mode))
            gated[(arm, mode)] = got
            arm_out[arm] = {
                'checked': got['checked'],
                'admitted': got['admitted'],
                'output_names': got['output_names'],
                'refused': got['refused'],
                'refused_refuted': got['refused_refuted'],
                'refused_uncertifiable': got['refused_uncertifiable'],
                'conflicting': got['conflicting'],
                'undeployed_weight': got['undeployed_weight'],
                'emptied_targets': len(got['emptied_targets']),
                'names_not_applicable': got['names_not_applicable'],
                'stamps_without_an_effective_session': [
                    r['stamp'] for r in got['stamps_without_an_effective_session']],
                'by_verdict': got['by_verdict'],
                'target_bytes_sha256': hashlib.sha256(
                    serialise({'targets': got['frames']})).hexdigest(),
            }
        modes[mode] = {
            'by_arm': arm_out,
            'checked': sum(v['checked'] for v in arm_out.values()),
            'admitted': sum(v['admitted'] for v in arm_out.values()),
            'output_names': sum(v['output_names'] for v in arm_out.values()),
            'refused': sum(v['refused'] for v in arm_out.values()),
            'refused_refuted': sum(v['refused_refuted'] for v in arm_out.values()),
            'refused_uncertifiable': sum(v['refused_uncertifiable'] for v in arm_out.values()),
            'conflicting': sum(v['conflicting'] for v in arm_out.values()),
            'undeployed_weight': sum(v['undeployed_weight'] for v in arm_out.values()),
            'emptied_targets': sum(v['emptied_targets'] for v in arm_out.values()),
            'names_not_applicable': sum(v['names_not_applicable'] for v in arm_out.values()),
            'by_verdict': {k: sum(v['by_verdict'][k] for v in arm_out.values())
                           for k in ET.VERDICTS},
        }
    modes['note_on_the_two_zero_rows'] = {
        'off': 'off 档 checked=0: 它什么都没查, 因此不得被读成"检查了没问题"',
        'names_not_applicable': '窗口在目标生效前就结束的 stamp, 其名字**原样保留**且记 '
                                'not_applicable —— 不是拒单。若在这里剔除, enforce 会把'
                                '最后一个 stamp 清成空目标, 账本读作"清仓"。',
    }

    byte_identical = {}
    for arm in ARMS:
        sealed = sealed_bytes(arm)
        rendered = serialise({'targets': frames_by_arm[arm], 'insufficient':
                              (load_frames(arm)[1].get('insufficient') or [])})
        byte_identical[arm] = {
            'sealed_sha256': hashlib.sha256(sealed).hexdigest(),
            'rendered_sha256': hashlib.sha256(rendered).hexdigest(),
            'identical': sealed == rendered,
            'mode_off_identical_to_sealed': hashlib.sha256(
                serialise({'targets': gated[(arm, 'off')]['frames'],
                           'insufficient': load_frames(arm)[1].get('insufficient') or []})
            ).hexdigest() == hashlib.sha256(sealed).hexdigest(),
            'mode_annotate_identical_to_sealed': hashlib.sha256(
                serialise({'targets': gated[(arm, 'annotate')]['frames'],
                           'insufficient': load_frames(arm)[1].get('insufficient') or []})
            ).hexdigest() == hashlib.sha256(sealed).hexdigest(),
        }
    assert all(v['identical'] for v in byte_identical.values()), byte_identical
    assert all(v['mode_off_identical_to_sealed'] for v in byte_identical.values())
    assert all(v['mode_annotate_identical_to_sealed'] for v in byte_identical.values())

    # 目标集合层面的只删不加
    sub = {}
    for arm in ARMS:
        before = set(frames_by_arm[arm])
        after = set(gated[(arm, 'enforce')]['frames'])
        added = sorted(after - before)
        removed = sorted(before - after)
        pair_loss = 0
        pair_gain = 0
        for stamp in before & after:
            a = set(frames_by_arm[arm][stamp])
            b = set(gated[(arm, 'enforce')]['frames'][stamp])
            pair_gain += len(b - a)
            pair_loss += len(a - b)
        sub[arm] = {'added_stamps': added, 'removed_stamps': removed,
                    'added_pairs': pair_gain, 'removed_pairs': pair_loss}
    removed_names = sum(v['removed_pairs'] for v in sub.values())
    added_names = sum(v['added_pairs'] for v in sub.values())
    assert not any(v['added_stamps'] for v in sub.values()), sub
    assert added_names == 0, sub

    events_before = {arm: set((e['code'], e['session']) for e in events_by_arm[arm])
                     for arm in ARMS}
    events_after = {}
    for arm in ARMS:
        events_after[arm] = set((e['code'], e['session'])
                                for e in ET.entry_events(
                                    gated[(arm, 'enforce')]['frames'], days))
    dropped = {arm: sorted(events_before[arm] - events_after[arm]) for arm in ARMS}
    deferred = {arm: sorted(events_after[arm] - events_before[arm]) for arm in ARMS}
    refused_events = {(r['arm'], r['code'], r['session']) for r in no_price}
    gate2 = {
        'what_it_proves': 'enforce 只做减法; annotate/off 与封存产物逐字节相同',
        'sealed_target_files_byte_identical': byte_identical,
        'substitution_only': sub,
        'removed_admissions': removed_names,
        'added_admissions': added_names,
        'entry_events_before': {a: len(events_before[a]) for a in ARMS},
        'entry_events_after_enforce': {a: len(events_after[a]) for a in ARMS},
        'entry_events_dropped': {a: len(dropped[a]) for a in ARMS},
        'entry_events_deferred_to_a_later_tradable_session': {
            a: len(deferred[a]) for a in ARMS},
        'dropped_events_are_the_refused_ones': {
            'dropped_total': sum(len(v) for v in dropped.values()),
            'refused_entry_events': len(refused_events),
            'dropped_equals_all_refused_not_tradable_events': all(
                set((r['code'], r['session']) for r in no_price if r['arm'] == a)
                == set(dropped[a]) for a in ARMS),
            'note': '删掉的不只是"入场"。闸门同时把若干"入场"**推到首个可成交日**, '
                    '所以 enforce 也会新增入场事件; 新增必须逐条公开, 不得只报删除数。',
        },
        'deferred_events_listed': {a: [list(x) for x in deferred[a]] for a in ARMS},
        'enforce_changes_the_portfolio_so_it_stays_behind_an_explicit_switch':
            'enforce 会改变目标集合与未部署权重 ⇒ 属经济口径变更, 默认不翻, '
            '且翻它需要独立协议 (翻默认四关 + 端到端对账)。',
    }
    assert gate2['dropped_events_are_the_refused_ones'][
        'dropped_equals_all_refused_not_tradable_events']

    # ---------------- 关三: 默认与口径档不变量 ----------------
    alias_results = {}
    for label, env in (('default(no env)', {}),
                       ('annotate', {ET.ENV_NAME: 'annotate'}),
                       ('shadow', {ET.ENV_NAME: 'shadow'}),
                       ('on', {ET.ENV_NAME: 'on'}),
                       ('1', {ET.ENV_NAME: '1'}),
                       ('off', {ET.ENV_NAME: 'off'}),
                       ('legacy', {ET.ENV_NAME: 'legacy'}),
                       ('enforce', {ET.ENV_NAME: 'enforce'})):
        alias_results[label] = ET.get_mode(env)
    digests = {label: hashlib.sha256(serialise(
        {'targets': ET.gate(frames_by_arm[ARMS[1]], days, panel,
                            mode=mode)['frames']})).hexdigest()
        for label, mode in (('off', 'off'), ('annotate', 'annotate'),
                            ('shadow', 'annotate'), ('enforce', 'enforce'))}
    gate3 = {
        'what_it_proves': '默认档就是文档写的零裁决观察者; 别名等价; 观察者不动目标',
        'env_name': ET.ENV_NAME,
        'documented_default': ET.DEFAULT_MODE,
        'resolved_modes': alias_results,
        'default_resolves_to_documented_default':
            alias_results['default(no env)'] == ET.DEFAULT_MODE,
        'annotate_aliases_resolve_alike':
            len({alias_results[k] for k in ('annotate', 'shadow', 'on', '1')}) == 1,
        'off_family_resolves_alike':
            len({alias_results[k] for k in ('off', 'legacy')}) == 1,
        'target_digest_by_mode_on_one_arm': digests,
        'observer_modes_agree': digests['off'] == digests['annotate'] == digests['shadow'],
        'counterfactual_is_not_empty': digests['enforce'] != digests['off'],
        'unknown_value_behaviour': '抛 ValueError, 不静默回退 (模块单测覆盖)',
    }
    assert gate3['default_resolves_to_documented_default']
    assert gate3['annotate_aliases_resolve_alike']
    assert gate3['off_family_resolves_alike']
    assert gate3['observer_modes_agree']
    assert gate3['counterfactual_is_not_empty']

    # ---------------- 关四: 端到端对账 ----------------
    ledger = ledger_orders()
    chosen = ledger[ARMS[0]]['generation']
    e2e = []
    for row in no_price:
        arm = row['arm']
        block = ledger[arm]
        orders = [o for o in block['orders']
                  if o['code'] == row['code'] and o['date'] == row['session']]
        filled = [o for o in orders if (o.get('filled') or 0) > 0]
        unfilled = [o for o in orders if (o.get('filled') or 0) == 0]
        if row['session'] >= (block['last_day'] or ''):
            # 🔴 账本恰好在**这一天**停下 (fail-stop) ⇒ 它根本没走到这天, 与
            #    "走到却没留订单"是两件事, 必须分开报。
            verdict = ('this_arms_ledger_ends_on_or_before_that_session_so_it_was_'
                       'never_reached')
        elif filled:
            verdict = 'filled_anyway'
        elif unfilled:
            verdict = 'absorbed_downstream_as_an_unfilled_order'
        else:
            verdict = 'no_order_recorded_downstream'
        e2e.append({**row, 'ledger_generation': block['generation'],
                    'ledger_last_day': block['last_day'],
                    'orders_on_that_session': len(orders),
                    'filled': len(filled),
                    'unfilled': len(unfilled),
                    'order_reasons': sorted({o.get('reason') for o in orders
                                             if o.get('reason')}),
                    'classification': verdict})
    gate4 = {
        'what_it_proves': '每个被拒入场在封存账本里到底是"被下游吸收了"还是"根本没走到"',
        'ledger_generation_used': chosen,
        'ledger_endpoints': {arm: {'last_day': ledger[arm]['last_day'],
                                   'days': ledger[arm]['days'],
                                   'status': ledger[arm]['status']} for arm in ARMS},
        'cases': e2e,
        'classification_counts': dict(Counter(r['classification'] for r in e2e)),
        'filled_on_a_session_the_gate_refuses': len(
            [r for r in e2e if r['classification'] == 'filled_anyway']),
        'note': '下游账本**确实**吸收了这些目标: 它按 (证券, 生效日) 落一条 '
                'filled=0 的订单并给出 reason, 但不回灌目标、不改状态机 ⇒ '
                '与"目标本来就没买"不可区分。这正是"应在上游修"的实证。',
    }
    assert gate4['filled_on_a_session_the_gate_refuses'] == 0, gate4

    reconciliation, cur_rows = historical_record_reconciliation(days, no_price)

    reject_ledger = gated[(ARMS[1], 'annotate')]['rejects']
    assert len(reject_ledger) == len([r for r in all_rejects if r['arm'] == ARMS[1]]), \
        'the annotate ledger read here must be the same one the gate-1 audit used'

    payload = {
        'batch': 'entry_tradability_20260924',
        'generated_at': datetime.now().astimezone().isoformat(),
        'purpose': 'P3: 修上游入口可交易性检查 —— 重建 13/13701 口径, 交付上游门, '
                   '跑四关验收与 A/B 反事实, 列清接线点',
        'window': {'from': days[0], 'to': days[-1], 'sessions': len(days)},
        'scope_note': '只读本地封存文件; 不改任何生产默认; 不接任何生产调用点; '
                      '不产生经济裁决; 不改变任何权重口径。',
        'defect_restated': {
            'recorded_at': 'reports/fuxi_event_inventory_plan_20260924.md 第 4 节',
            'statement': '冻结目标曾在不可交易日建仓: llm1 11 + mem 2 = 13/13701 '
                         '入场事件落在无行情行日, 应在上游修而不是在下游吸收。',
            'why_it_is_not_a_market_event':
                '选股只校验了信号日 close>0; 目标自次日起生效, 生效日能否成交从未被校验。',
        },
        'denominators': {
            'arms': ARMS,
            'per_arm': per_arm,
            'stamps_total': sum(v['stamps'] for v in per_arm.values()),
            'admissions_checked_by_the_gate': modes['annotate']['checked'],
            'entry_events': total_events,
            'entry_events_without_a_price_row': len(no_price),
            'note': '两个分母不是一回事: 门逐**准入**计数 (每 stamp 每名字), '
                    '13/13701 是**入场事件**口径。两者不得互相替代或相除。',
        },
        'panel': panel_meta,
        'four_gates': {'gate1_getter': gate1, 'gate2_ab_counterfactual': gate2,
                       'gate3_default_invariance': gate3, 'gate4_end_to_end': gate4},
        'readings_by_mode': modes,
        'entry_events_without_a_price_row': sorted(
            no_price, key=lambda r: (r['arm'], r['session'], r['code'])),
        'historical_record_reconciliation': reconciliation,
        'reject_ledger_size_at_admission_level': len(all_rejects),
        'reject_ledger_by_verdict_at_admission_level': dict(
            Counter(r['verdict'] for r in all_rejects)),
        'reject_ledger_by_arm_at_admission_level': dict(
            Counter(r['arm'] for r in all_rejects)),
        'site_inventory': site_inventory(),
        'claims_not_made': [
            '没有把任何一个"缺记录"写成停牌: 缺证据与已证伪在 by_category 里分开计数',
            '没有推断成交价、没有填补、没有插值、没有把停牌日当零收益',
            '没有接进任何生产调用点; 本批只交付门本体 + 读数 + 接线点清单',
            '没有翻开任何默认: 门默认 annotate (零裁决观察者), enforce 需显式开启',
            '没有做经济裁决: 被拒的 13 个入场事件不带任何"该不该买"的判断',
            '没有改权重口径: enforce 保持 1/top_n, 未部署部分留现金并逐条声明',
            '本批不改变账本、S5、JQ 准入或任何封存产物的字节',
        ],
        'formal_eligibility': False,
        'full_stage2_accepted': False,
        'hashes': {
            'data/raw/daily_prices.csv': digest(ROOT / 'data/raw/daily_prices.csv'),
            'data/raw/suspend_d.csv': digest(ROOT / 'data/raw/suspend_d.csv'),
            'data/raw/delist_dates.csv': digest(ROOT / 'data/raw/delist_dates.csv'),
            'data/raw/trade_calendar.csv': digest(ROOT / 'data/raw/trade_calendar.csv'),
            'research/factor_alchemy/entry_tradability.py':
                digest(ROOT / 'research/factor_alchemy/entry_tradability.py'),
            'scripts/fuxi_entry_tradability_20260924.py': digest(Path(__file__)),
            **{'reports/execution_ledger_20260923/targets/%s.json' % arm:
               digest(TARGETS_DIR / (arm + '.json')) for arm in ARMS},
        },
    }
    payload['_reject_rows'] = all_rejects
    return payload


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'readings.json'
    if target.exists():
        raise RuntimeError('readings.json already exists; refusing to overwrite')
    payload = readings()
    rejects = payload.pop('_reject_rows')
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    with io.open(target, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(raw)
    with io.open(OUT / 'reject_ledger.csv', 'w', encoding='utf-8', newline='') as handle:
        handle.write('arm,stamp,code,session,verdict,category,weight,action\n')
        for row in sorted(rejects, key=lambda r: (r['arm'], r['session'] or '',
                                                 r['code'])):
            handle.write('%s,%s,%s,%s,%s,%s,%.10f,%s\n' % (
                row['arm'], row['stamp'], row['code'], row['session'], row['verdict'],
                row['category'], row['weight'], row['action']))
    with io.open(OUT / 'entry_events_without_a_price_row.csv', 'w',
                 encoding='utf-8', newline='') as handle:
        handle.write('arm,code,session,stamp,verdict\n')
        for row in payload['entry_events_without_a_price_row']:
            handle.write('%s,%s,%s,%s,%s\n' % (row['arm'], row['code'], row['session'],
                                               row.get('stamp', ''), row['verdict']))

    d = payload['denominators']
    print('window %s..%s sessions=%d arms=%d'
          % (payload['window']['from'], payload['window']['to'],
             payload['window']['sessions'], len(ARMS)), flush=True)
    print('admissions checked=%d  names not applicable=%d  entry events=%d  '
          'entry events without a price row=%d'
          % (d['admissions_checked_by_the_gate'],
             payload['readings_by_mode']['annotate']['names_not_applicable'],
             d['entry_events'], d['entry_events_without_a_price_row']), flush=True)
    for mode, block in payload['readings_by_mode'].items():
        if not isinstance(block, dict) or 'checked' not in block:
            continue
        print('  mode=%-9s checked=%5d admitted=%5d output_names=%5d refused=%5d '
              '(refuted=%d uncertifiable=%d conflicting=%d) undeployed=%.6f emptied=%d'
              % (mode, block['checked'], block['admitted'], block['output_names'],
                 block['refused'], block['refused_refuted'], block['refused_uncertifiable'],
                 block['conflicting'], block['undeployed_weight'], block['emptied_targets']),
              flush=True)
    g2 = payload['four_gates']['gate2_ab_counterfactual']
    print('removed admissions=%d added=%d; entry events dropped=%s deferred=%s'
          % (g2['removed_admissions'], g2['added_admissions'],
             g2['entry_events_dropped'],
             g2['entry_events_deferred_to_a_later_tradable_session']), flush=True)
    g4 = payload['four_gates']['gate4_end_to_end']
    print('end-to-end: %s' % json.dumps(g4['classification_counts'], ensure_ascii=False),
          flush=True)
    print('sites without an entry tradability check: %d of %d'
          % (payload['site_inventory']['sites_without_an_entry_tradability_check'],
             payload['site_inventory']['sites_total']), flush=True)
    print('written: %s (+ reject_ledger.csv, entry_events_without_a_price_row.csv)'
          % target, flush=True)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
class ScriptTests(unittest.TestCase):
    def test_serialise_reproduces_the_run_contract_format(self):
        payload = {'b': 1, 'a': [2, {'c': 3.0}]}
        self.assertEqual(serialise(payload),
                         b'{\n  "a": [\n    2,\n    {\n      "c": 3.0\n    }\n  ],\n'
                         b'  "b": 1\n}\n')

    def test_every_site_line_still_matches_its_recorded_pattern(self):
        inventory = site_inventory()
        self.assertEqual(inventory['sites_total'], len(SITES))
        self.assertTrue(all(not s['has_entry_tradability_check']
                            for s in inventory['sites']))

    def test_site_inventory_refuses_a_drifted_line(self):
        drifted = dict(SITES[0])
        drifted['pattern'] = 'this text is not in the file'
        original = SITES[0]
        SITES[0] = drifted
        try:
            with self.assertRaises(AssertionError):
                site_inventory()
        finally:
            SITES[0] = original

    def test_needed_keys_only_covers_the_effective_session_not_the_stamp(self):
        days = ['2025-01-02', '2025-01-03', '2025-01-06']
        frames = {'2025-01-02': {'A': 1.0}}
        keys = needed_keys(frames, days)
        self.assertEqual(keys, {'A': {'2025-01-03'}})

    def test_status_table_split_separates_full_and_partial_halts(self):
        susp = pd.DataFrame([{'ts_code': 'A', 'trade_date': '20250103',
                              'suspend_type': 'S', 'suspend_timing': None},
                             {'ts_code': 'B', 'trade_date': '20250103',
                              'suspend_type': 'S', 'suspend_timing': '09:30-10:30'},
                             {'ts_code': 'C', 'trade_date': '20250103',
                              'suspend_type': 'R', 'suspend_timing': None}])
        susp = susp.assign(day=susp.trade_date.map(dashed))
        full = susp[(susp.suspend_type == 'S') & susp.suspend_timing.fillna('').eq('')]
        partial = susp[(susp.suspend_type == 'S') & ~susp.suspend_timing.fillna('').eq('')]
        self.assertEqual(sorted(full.ts_code), ['A'])
        self.assertEqual(sorted(partial.ts_code), ['B'])

    def test_module_tests_are_all_green(self):
        suite = ET.tests()
        result = unittest.TestResult()
        suite.run(result)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.failures, [])
        self.assertGreater(result.testsRun, 20)


def tests():
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite([loader.loadTestsFromTestCase(ScriptTests),
                                ET.tests()])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--tests', action='store_true')
    args = parser.parse_args()
    if args.tests:
        return tests()
    if args.run:
        run()
        return 0
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
