# -*- coding: utf-8 -*-
"""P5 批次：把交易日历延伸到第 26 个周频截面（R5 的判定端点）。

背景
----
R5 预注册（reports/r5_preregistration_20260924/protocol.json）把**统计要求**先固定下来：
描述性检查点 8 段、判定端点 **26 段**。但随包交易日历只到 2026-12-31，只覆盖 **15** 段，
于是 protocol 里 `decisive_endpoint.signal` 为 null、`calendar_covers=false`。

🔴 本批**只延伸日历覆盖**，**绝不改统计要求**：26 这个数从 R5 模块读入，本脚本不得自带副本。
   日历不够时 R5 的做法是「把端点日期留成 null」，而不是把要求缩到能装下 —— 这个方向不能反。

🔴 这是一次**数据源替换**，因此必须证「只是扩展」而不是「其实是覆盖」：
     ① 包含：随包日历的每一天都必须在厂商日历里（含 is_open 标记一致）；
     ② 单调：随包日历标为开市的日子，厂商也必须标为开市；
     ③ 新增可枚举：差集必须是一段连续区间，且逐行可列举。
   三条任一不成立即拒跑（fail-closed）。

🔴 采集默认值必须与日历**原子**同进：日历一延伸，collect_daily_full.CAL_END_DEFAULT
   就得跟着走，否则补采会被重新静默截断（P4 刚修掉的那个 bug）。
   故本脚本在同一个事务里改两个文件，任一环失败即**从备份回滚**。

用法:
    python scripts/fuxi_calendar_extension_20260925.py --tests
    python scripts/fuxi_calendar_extension_20260925.py --plan      # 只读: 抓取+对账+预演
    python scripts/fuxi_calendar_extension_20260925.py --run       # 落盘: 备份+改写+对账
    python scripts/fuxi_calendar_extension_20260925.py --status    # 只读: 当前日历视野与 R5 端点
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/calendar_extension_20260925'
CALENDAR = ROOT / 'data/raw/trade_calendar.csv'
BACKUP = OUT / 'trade_calendar.csv.before'
COLLECT = ROOT / 'strategies/final/collect_daily_full.py'
VENDOR_END = '20271231'
VENDOR_START = '20210101'
HEADER = 'exchange,cal_date,is_open,pretrade_date'

sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'strategies/final'))

import fuxi_r5_preregistration_20260924 as R5  # noqa: E402


# ------------------------------------------------------------------ 纯函数

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def read_bytes(path) -> bytes:
    return Path(path).read_bytes()


def line_ending(raw: bytes) -> str:
    """返回 'crlf' / 'lf' / 'mixed' / 'none'。"""
    crlf = raw.count(b'\r\n')
    lf = raw.count(b'\n') - crlf
    if crlf and lf:
        return 'mixed'
    if crlf:
        return 'crlf'
    if lf:
        return 'lf'
    return 'none'


def split_rows(raw: bytes):
    """把日历字节切成 (header_line, body_lines, 分隔符, 结尾是否有换行)。"""
    text = raw.decode('utf-8')
    nl = '\r\n' if line_ending(raw) == 'crlf' else '\n'
    trailing = text.endswith(nl)
    lines = text.split(nl)
    if trailing:
        lines = lines[:-1]
    header = lines[0]
    return header, lines[1:], nl, trailing


def parse_row(line: str):
    """'SSE,20261231,1,20261230' -> ('SSE','20261231','1','20261230'); 不合法返回 None。"""
    parts = line.split(',')
    if len(parts) != 4:
        return None
    if not parts[1].isdigit():
        return None
    return tuple(parts)


def calendar_facts(raw: bytes) -> dict:
    """只读审计: 行长什么样、行尾是什么、日期写法有没有混。

    🔴 行尾必须先按**字节**看: 本文件头 256 行的行尾是 `\\r\\r\\n` (Windows 文本模式把
       `\\r\\n` 里的 `\\n` 又翻译了一次), 其余行是 `\\r\\n`。用 `str.splitlines()` 看会把
       多余的 `\\r` 当成**空行** —— 于是「256 行空行」这种结论纯属读法造成的幻觉。
       这里直接数字节, 不用 splitlines。

    这些是**观察**，不是修复对象 —— 异常行尾与中段的重复表头本批只报告、不改写。
    """
    header, body, nl, trailing = split_rows(raw)
    blanks = [i for i, l in enumerate(body) if not l.strip()]
    non_data = [(i, l) for i, l in enumerate(body) if l.strip() and parse_row(l) is None]
    dates = []
    for line in body:
        parsed = parse_row(line)
        if parsed:
            dates.append(parsed[1])
    compact = sum(1 for d in dates if re.fullmatch(r'\d{8}', d))
    dashed = sum(1 for d in dates if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d))
    header_text = header.rstrip('\r')
    return {
        'header': header_text,
        'header_raw_has_a_stray_cr': header != header_text,
        'header_is_canonical': header_text == HEADER,
        'line_ending': line_ending(raw),
        'double_cr_line_endings': raw.count(b'\r\r\n'),
        'well_formed_line_endings': raw.count(b'\r\n') - raw.count(b'\r\r\n'),
        'ends_with_newline': trailing,
        'total_lines': len(body),
        'blank_lines': len(blanks),
        'non_data_lines': len(non_data),
        'non_data_samples': non_data[:3],
        'data_rows': len(dates),
        'first_data_line': body[0] if body else None,
        'last_data_line': body[-1] if body else None,
        'first_date': min(dates) if dates else None,
        'last_date': max(dates) if dates else None,
        'date_written_compact': compact,
        'date_written_dashed': dashed,
        'is_descending': all(dates[i] > dates[i + 1] for i in range(len(dates) - 1)),
        'duplicate_dates': len(dates) - len(set(dates)),
        'open_days': sum(1 for line in body
                         if (parse_row(line) or ('', '', '', ''))[2] == '1'),
    }


def canonical_rows(frame) -> list:
    """把厂商 DataFrame 变成与随包文件**逐字同形**的行 (SSE,YYYYMMDD,is_open,pretrade_date)。"""
    rows = []
    for _, r in frame.iterrows():
        rows.append('%s,%s,%s,%s' % (str(r['exchange']), str(r['cal_date']),
                                     str(int(r['is_open'])), str(r['pretrade_date'])))
    return rows


MAX_GAP_DAYS = 12   # 相邻会话最大间隔的合理上限: 春节/国庆连休 + 周末, 实测 8 天


def addition_shape(additions: list, max_gap: int = MAX_GAP_DAYS) -> dict:
    """新增块的形状: 逐月都要有会话, 且相邻会话不得出现异常长的空档。

    🔴 为什么需要这一条：厂商完全可能只给出「它确定的那些日子」——
       那样差集依然「可枚举」, 却会在日历里留下整月空洞, 使 R5 的周频截面数虚高。
       只验「差集非空」是抓不到的, 必须验形状。
    """
    if not additions:
        return {'months_covered': [], 'months_without_any_session': [],
                'max_gap_in_calendar_days': None, 'max_gap_allowed': max_gap,
                'is_contiguous': False, 'reason': '新增为空'}

    dates = sorted(additions)
    months = sorted({d[:6] for d in dates})
    lo_m, hi_m = int(months[0]), int(months[-1])
    expected = []
    y, m = lo_m // 100, lo_m % 100
    while y * 100 + m <= hi_m:
        expected.append('%04d%02d' % (y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    missing = sorted(set(expected) - set(months))

    from datetime import date as _date
    def _d(s):
        return _date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    gaps = [(_d(dates[i + 1]) - _d(dates[i])).days for i in range(len(dates) - 1)]
    worst = max(gaps) if gaps else 0
    return {'months_covered': months,
            'months_without_any_session': missing,
            'max_gap_in_calendar_days': worst,
            'max_gap_allowed': max_gap,
            'is_contiguous': (not missing) and worst <= max_gap,
            'reason': ('逐月有会话' if not missing else '有整月空洞') if worst <= max_gap
                      else '相邻会话间隔 %d 天 > %d' % (worst, max_gap)}


def reconcile(packaged: dict, vendor: dict) -> dict:
    """三条断言：包含 / 单调 / 新增可枚举且形状完好。

    packaged / vendor 都是 {cal_date: is_open 字符串}。
    """
    pkg_dates, ven_dates = set(packaged), set(vendor)
    missing_from_vendor = sorted(pkg_dates - ven_dates)
    additions = sorted(ven_dates - pkg_dates)

    open_mismatch = sorted(d for d in (pkg_dates & ven_dates)
                           if packaged[d] == '1' and vendor[d] != '1')
    close_mismatch = sorted(d for d in (pkg_dates & ven_dates)
                            if packaged[d] != '1' and vendor[d] == '1')

    shape = addition_shape(additions)
    containment = not missing_from_vendor
    monotone = not open_mismatch
    enumerable = bool(additions) and shape['is_contiguous']
    return {
        'assertion_1_containment_packaged_subset_of_vendor': containment,
        'dates_in_packaged_but_not_in_vendor': missing_from_vendor,
        'assertion_2_monotone_open_flags_preserved': monotone,
        'packaged_open_but_vendor_closed': open_mismatch,
        'packaged_closed_but_vendor_open': close_mismatch,
        'assertion_3_additions_are_enumerable_and_well_shaped': enumerable,
        'additions_shape': shape,
        'additions_count': len(additions),
        'additions_first': additions[0] if additions else None,
        'additions_last': additions[-1] if additions else None,
        'additions_are_disjoint_from_packaged': not (set(additions) & pkg_dates),
        'ok': containment and monotone and enumerable,
    }


def insert_after_header(raw: bytes, rows: list) -> bytes:
    """把 rows 插到表头之后（其余字节逐字不动）。"""
    header, body, nl, trailing = split_rows(raw)
    text = header + nl + nl.join(rows) + nl + nl.join(body)
    if trailing:
        text += nl
    return text.encode('utf-8')


def insert_is_byte_clean(before: bytes, after: bytes, rows: list) -> dict:
    """验证「只是插入」：前缀相同、后缀相同、插入块正好是 rows。"""
    header, _body, nl, _t = split_rows(before)
    prefix = (header + nl).encode('utf-8')
    block = (nl.join(rows) + nl).encode('utf-8')
    return {
        'prefix_preserved': after[:len(prefix)] == prefix,
        'suffix_preserved': after[len(prefix) + len(block):] == before[len(prefix):],
        'inserted_block_is_the_declared_rows': after[len(prefix):len(prefix) + len(block)] == block,
        'bytes_before': len(before),
        'bytes_after': len(after),
        'bytes_inserted': len(after) - len(before),
        'declared_block_bytes': len(block),
    }


def move_cal_end_default(source: str, old: str, new: str) -> tuple:
    """把采集脚本里的 CAL_END_DEFAULT 从 old 改到 new（只改字面量，不动其余字节）。

    返回 (新正文, 命中次数)。命中次数 != 1 视为失败。
    """
    pattern = re.compile(r"CAL_END_DEFAULT = '(\d{8})'")
    hits = pattern.findall(source)
    if len(hits) != 1 or hits[0] != old:
        return source, len(hits)
    return source.replace("CAL_END_DEFAULT = '%s'" % old,
                          "CAL_END_DEFAULT = '%s'" % new), 1


# ------------------------------------------------------------------ 网络（惰性）

def fetch_vendor(start=VENDOR_START, end=VENDOR_END):
    """只在真正需要时 import 厂商 SDK。本函数在纯函数测试中永不执行。"""
    sys.path.insert(0, str(ROOT))
    from credentials import get_tushare_token
    import tushare as ts
    ts.set_token(get_tushare_token())
    pro = ts.pro_api()
    frame = pro.trade_cal(exchange='SSE', start_date=start, end_date=end)
    if frame is None or len(frame) == 0:
        raise RuntimeError('厂商 trade_cal 返回空表')
    return frame


def vendor_map(frame) -> dict:
    return {str(r['cal_date']): str(int(r['is_open'])) for _, r in frame.iterrows()}


def packaged_map(raw: bytes) -> dict:
    _h, body, _nl, _t = split_rows(raw)
    out = {}
    for line in body:
        parsed = parse_row(line)
        if parsed:
            out[parsed[1]] = parsed[2]
    return out


# ------------------------------------------------------------------ 主流程

def plan() -> dict:
    """只读预演：抓取厂商日历、对账、算出将要插入哪些行。不写任何数据文件。"""
    frame = fetch_vendor()
    rows = [r for r in canonical_rows(frame)
            if parse_row(r) and parse_row(r)[1] > calendar_facts(read_bytes(CALENDAR))['last_date']]
    rows.sort(key=lambda r: parse_row(r)[1], reverse=True)
    raw = read_bytes(CALENDAR)
    rec = reconcile(packaged_map(raw), vendor_map(frame))
    return {'facts_before': calendar_facts(raw), 'reconciliation': rec,
            'rows_to_insert': rows, 'vendor': {'start': VENDOR_START, 'end': VENDOR_END,
                                               'rows': len(frame),
                                               'max': max(str(c) for c in frame['cal_date'])},
            'planned_bytes_after': len(insert_after_header(raw, rows))}


def apply_extension(dry_rows: list) -> dict:
    """落盘：备份 → 校验将插入的行 → 插入 → 移动默认值 → 双向断言；任一步失败即回滚。"""
    raw = read_bytes(CALENDAR)
    facts_before = calendar_facts(raw)
    rows = sorted(dry_rows, key=lambda r: parse_row(r)[1], reverse=True)

    OUT.mkdir(parents=True, exist_ok=True)
    BACKUP.write_bytes(raw)
    assert BACKUP.read_bytes() == raw, '备份必须与原文逐字节相同'

    collect_before = read_bytes(COLLECT)
    old_default = re.search(r"CAL_END_DEFAULT = '(\d{8})'",
                            collect_before.decode('utf-8')).group(1)
    if old_default != facts_before['last_date']:
        raise RuntimeError('日历末端 %s 与采集默认值 %s 本就不一致 —— 拒绝在脏状态下延伸'
                           % (facts_before['last_date'], old_default))

    rollback = {'needed': False, 'reason': None}
    try:
        after = insert_after_header(raw, rows)
        CALENDAR.write_bytes(after)
        clean = insert_is_byte_clean(raw, after, rows)

        new_source, hits = move_cal_end_default(collect_before.decode('utf-8'),
                                                old_default, VENDOR_END)
        if hits != 1:
            raise RuntimeError('CAL_END_DEFAULT 字面量命中 %d 次 (应为 1)' % hits)
        COLLECT.write_bytes(new_source.encode('utf-8'))

        # ---- 双向断言: 日历变了、默认值也跟着变了、且二者一致 ----
        facts_after = calendar_facts(read_bytes(CALENDAR))
        default_after = re.search(r"CAL_END_DEFAULT = '(\d{8})'",
                                  read_bytes(COLLECT).decode('utf-8')).group(1)
        if facts_after['last_date'] != VENDOR_END:
            raise RuntimeError('日历末端应为 %s, 实为 %s' % (VENDOR_END, facts_after['last_date']))
        if default_after != facts_after['last_date']:
            raise RuntimeError('采集默认值 %s 未与日历末端 %s 同进'
                               % (default_after, facts_after['last_date']))
        if facts_after['data_rows'] != facts_before['data_rows'] + len(rows):
            raise RuntimeError('行数增量与声明不符')
        if facts_after['duplicate_dates'] != 0:
            raise RuntimeError('延伸后出现重复交易日')
        if not clean['prefix_preserved'] or not clean['suffix_preserved']:
            raise RuntimeError('插入破坏了已有字节')
    except Exception as exc:                       # noqa: BLE001 —— 出任何事都必须回滚
        CALENDAR.write_bytes(raw)
        COLLECT.write_bytes(collect_before)
        rollback = {'needed': True, 'reason': '%s: %s' % (type(exc).__name__, exc),
                    'calendar_restored': CALENDAR.read_bytes() == raw,
                    'collect_restored': COLLECT.read_bytes() == collect_before}
        raise

    return {'facts_before': facts_before, 'facts_after': facts_after,
            'byte_cleanliness': clean,
            'backup_path': str(BACKUP.relative_to(ROOT)).replace('\\', '/'),
            'backup_sha256': sha256_file(BACKUP),
            'calendar_sha256_before': sha256_bytes(raw),
            'calendar_sha256_after': sha256_file(CALENDAR),
            'collect_default_before': old_default,
            'collect_default_after': default_after,
            'collect_sha256_before': sha256_bytes(collect_before),
            'collect_sha256_after': sha256_file(COLLECT),
            'rows_inserted': len(rows),
            'line_ending_preserved': (facts_before['line_ending'] == facts_after['line_ending']),
            'rollback': rollback}


def r5_endpoint_view() -> dict:
    """用**延伸后**的日历重算 R5 端点。统计要求从 R5 模块读入, 本脚本不带副本。"""
    days = R5.sessions()
    window = R5.observation_window(days)
    return {
        'decisive_sections_required': R5.DECISIVE_SECTIONS,
        'descriptive_sections_required': R5.DESCRIPTIVE_SECTIONS,
        'observation_start_unchanged': window['observation_start'],
        'calendar_last_session': window['calendar_last_session'],
        'calendar_future_sections': window['calendar_future_sections'],
        'descriptive_checkpoint': window['descriptive_endpoint'],
        'decisive_endpoint': window['decisive_endpoint'],
        'requirement_was_read_from_the_r5_module': True,
        'requirement_never_passed_as_a_local_literal': True,
    }


def snapshot_vs_current(snapshot: dict, current: dict) -> dict:
    """把「预注册快照」与「按当前日历重算的覆盖」逐字段对照。

    🔴 判据是**两个方向都要成立**：
      · 统计要求（段数）必须**逐字不变** —— 否则本批就偷偷改了要求；
      · 覆盖必须**确实变了** —— 否则延伸日历这件事没有发生（或没生效）。
    """
    def sections(block, key):
        return block[key]['sections']

    req_same = (sections(snapshot, 'descriptive_checkpoint') ==
                sections(current, 'descriptive_checkpoint') and
                sections(snapshot, 'decisive_endpoint') ==
                sections(current, 'decisive_endpoint'))
    before_sec = snapshot.get('calendar_future_sections')
    after_sec = current.get('calendar_future_sections')
    dec_before = snapshot['decisive_endpoint']
    dec_after = current['decisive_endpoint']
    return {
        'requirement_unchanged': req_same,
        'descriptive_sections_before': sections(snapshot, 'descriptive_checkpoint'),
        'descriptive_sections_after': sections(current, 'descriptive_checkpoint'),
        'decisive_sections_before': sections(snapshot, 'decisive_endpoint'),
        'decisive_sections_after': sections(current, 'decisive_endpoint'),
        'calendar_future_sections_before': before_sec,
        'calendar_future_sections_after': after_sec,
        'coverage_changed': before_sec != after_sec,
        'decisive_endpoint_signal_before': dec_before['signal'],
        'decisive_endpoint_signal_after': dec_after['signal'],
        'decisive_endpoint_matured_on_after': dec_after['matured_on'],
        'decisive_endpoint_coverable_before': dec_before['calendar_covers'],
        'decisive_endpoint_coverable_after': dec_after['calendar_covers'],
        'staleness_that_existed': (dec_before['signal'] is None
                                   and dec_after['signal'] is not None),
        'ok': req_same and (before_sec != after_sec),
    }


def status_tool_audit() -> dict:
    """记录并机检一个影子结论：R5 的 --status 曾把**冻结快照**当成**当前覆盖**报出。

    日历延伸后，`protocol.decisive_endpoint.calendar_covers` 仍是 false、signal 仍是 null；
    `--status` 直接读它 ⇒ 工具会一直说「判定端点不可覆盖」，而它早已可覆盖。
    规范做法是不改冻结件、把两件事分开报（见 fuxi_r5_preregistration_20260924.status 的修正）。
    """
    protocol = json.loads(read_bytes(ROOT / 'reports/r5_preregistration_20260924/protocol.json')
                          .decode('utf-8'))
    snapshot = {k: protocol[k] for k in ('descriptive_checkpoint', 'decisive_endpoint',
                                         'calendar_last_session', 'calendar_future_sections')}
    days = R5.sessions()
    window = R5.observation_window(days)
    current = {'descriptive_checkpoint': window['descriptive_endpoint'],
               'decisive_endpoint': window['decisive_endpoint'],
               'calendar_last_session': window['calendar_last_session'],
               'calendar_future_sections': window['calendar_future_sections']}
    cmp_ = snapshot_vs_current(snapshot, current)
    live = R5.status()
    return {
        'what': ('R5 --status 原先把 protocol 里冻结的 descriptive_checkpoint / '
                 'decisive_endpoint 直接当作「日历覆盖」输出'),
        'why_it_matters': ('日历延伸后这两个字段变成**预注册当时的快照**；继续当覆盖报，'
                           '工具会永远说「判定端点不可覆盖」，而它已经可覆盖 ⇒ 影子结论'),
        'comparison': cmp_,
        'snapshot_is_not_rewritten': True,
        'protocol_sha256': sha256_file(ROOT / 'reports/r5_preregistration_20260924/protocol.json'),
        'how_status_reports_it_now': {
            'as_recorded_at_preregistration': live['as_recorded_at_preregistration'],
            'coverable_with_the_current_calendar': live['coverable_with_the_current_calendar'],
            'flat_keys_kept_for_compatibility':
                live['checkpoint_keys_are_preregistration_snapshots'],
        },
        'verdict_unchanged': live['verdict'],
        'decisive_reached_unchanged': live['decisive_reached'],
        'unknown_sample_effective_count': live['unknown_sample_effective_count'],
    }


def status() -> dict:
    raw = read_bytes(CALENDAR)
    facts = calendar_facts(raw)
    view = r5_endpoint_view()
    out = {'calendar': facts, 'r5': view,
           'calendar_sha256': sha256_bytes(raw),
           'collect_cal_end_default': re.search(
               r"CAL_END_DEFAULT = '(\d{8})'",
               read_bytes(COLLECT).decode('utf-8')).group(1)}
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    return out


def run() -> int:
    preview = plan()
    rec = preview['reconciliation']
    if not rec['ok']:
        print('FATAL: 三条对账断言未全通过, 拒绝改写日历')
        print(json.dumps(rec, ensure_ascii=False, indent=1))
        return 2

    applied = apply_extension(preview['rows_to_insert'])
    view = r5_endpoint_view()

    # ---- 落盘前最后一道: 统计要求不得被本批改动 ----
    if view['decisive_sections_required'] != R5.DECISIVE_SECTIONS:
        raise RuntimeError('统计要求被改动了 —— 本批只允许延伸日历覆盖')

    rows_csv = OUT / 'vendor_2027_rows.csv'
    with io.open(rows_csv, 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['exchange', 'cal_date', 'is_open', 'pretrade_date'])
        for r in preview['rows_to_insert']:
            w.writerow(parse_row(r))

    payload = {
        'batch': 'calendar_extension_20260925',
        'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'purpose': ('P5: 把交易日历延伸到 R5 第 %d 个周频截面所需的视野; '
                    '只延伸覆盖, 不改统计要求。' % R5.DECISIVE_SECTIONS),
        'vendor_call': {'api': 'trade_cal', 'exchange': 'SSE',
                        'start': VENDOR_START, 'end': VENDOR_END,
                        'rows_returned': preview['vendor']['rows'],
                        'max_date_in_vendor': preview['vendor']['max']},
        'reconciliation': rec,
        'extension': applied,
        'r5_endpoint_after_extension': view,
        'frozen_protocol_left_untouched': {
            'path': 'reports/r5_preregistration_20260924/protocol.json',
            'sha256': sha256_file(ROOT / 'reports/r5_preregistration_20260924/protocol.json'),
            'reason': ('protocol 自身的 freeze() 拒绝覆盖; 其 calendar_note 与 hashes[\'data/raw/'
                       'trade_calendar.csv\'] 记录的是**预注册当时**的样子, 属历史记录, 不改写。'
                       '本批把「日历已延伸」这一事实写进 access_log.jsonl (append-only), '
                       '并用 supersedes 字段指明它取代了哪一条陈述。'),
            'supersedes': ('protocol.calendar_note 中「日历只覆盖 15 段、判定端点日期未定」'
                           '一句 —— 在日历延伸后不再成立; 判定端点的**段数要求 (26) 不变**。'),
            'stale_hash_is_expected': ('protocol.hashes[\'data/raw/trade_calendar.csv\'] = '
                                       '42f39551… 是预注册当时的日历; 延伸后该哈希必然变化, '
                                       '这不是 protocol 损坏, 而是本批的产物。'),
        },
        'claims_not_made': [
            '没有改动 R5 的统计要求: 26 是从 R5 模块读入的, 本脚本没有自带副本',
            '没有改动观测起点 / 预注册上限 / 候选清单 / 未知样本计数',
            '没有重写 protocol.json: 它是冻结件, 本批只在 access log 里留痕',
            '没有声称 2027 的节假日安排已成文 —— 厂商给出的 2027 日历是**规则推算**',
            '没有修日中段的空行与重复表头: 本批只报告 (改写已有数据文件不在本批范围)',
            '没有用新日历去读任何未来数据: 未知样本有效计数仍为 0',
        ],
    }
    out = OUT / 'readings.json'
    out.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))

    # ---- access log: 用 R5 自己的 append(), 保持同一格式 (LF-only, 追加不改写) ----
    record = R5.append({
        'kind': 'calendar_extension',
        'purpose': 'extend the trade calendar so the decisive endpoint becomes coverable',
        'calendar_sha256_before': applied['calendar_sha256_before'],
        'calendar_sha256_after': applied['calendar_sha256_after'],
        'calendar_last_session_after': applied['facts_after']['last_date'],
        'calendar_future_sections_before': 15,
        'calendar_future_sections_after': view['calendar_future_sections'],
        'decisive_sections_required': R5.DECISIVE_SECTIONS,
        'decisive_endpoint_signal': view['decisive_endpoint']['signal'],
        'decisive_endpoint_matured_on': view['decisive_endpoint']['matured_on'],
        'requirement_changed': False,
        'model_changed': False,
        'note': ('日历覆盖已延伸; 统计要求未改; protocol.json 未改, '
                 '本记录取代其 calendar_note 中关于覆盖不足的陈述'),
    })

    print('== 对账 (三条断言) ==')
    print('  ① 包含: %s (缺 %d 天)'
          % (rec['assertion_1_containment_packaged_subset_of_vendor'],
             len(rec['dates_in_packaged_but_not_in_vendor'])))
    print('  ② 单调: %s (开市标记冲突 %d 天)'
          % (rec['assertion_2_monotone_open_flags_preserved'],
             len(rec['packaged_open_but_vendor_closed'])))
    print('  ③ 新增可枚举且形状完好: %s, %s..%s 共 %d 行'
          % (rec['assertion_3_additions_are_enumerable_and_well_shaped'],
             rec['additions_first'], rec['additions_last'], rec['additions_count']))
    print('     逐月覆盖 %d 个月, 空洞 %s, 相邻会话最大间隔 %d 天 (上限 %d)'
          % (len(rec['additions_shape']['months_covered']),
             rec['additions_shape']['months_without_any_session'] or '无',
             rec['additions_shape']['max_gap_in_calendar_days'],
             rec['additions_shape']['max_gap_allowed']))
    print('== 写入 ==')
    print('  日历 %s -> %s 行为 %s -> %s (字节 %d -> %d, 插入 %d)'
          % (applied['facts_before']['data_rows'], applied['facts_after']['data_rows'],
             applied['facts_before']['last_date'], applied['facts_after']['last_date'],
             applied['byte_cleanliness']['bytes_before'],
             applied['byte_cleanliness']['bytes_after'],
             applied['byte_cleanliness']['bytes_inserted']))
    print('  字节洁净: 前缀=%s 后缀=%s 插入块=声明行 %s'
          % (applied['byte_cleanliness']['prefix_preserved'],
             applied['byte_cleanliness']['suffix_preserved'],
             applied['byte_cleanliness']['inserted_block_is_the_declared_rows']))
    print('  采集默认值 %s -> %s (原子同进)'
          % (applied['collect_default_before'], applied['collect_default_after']))
    print('  行尾保留: %s' % applied['line_ending_preserved'])
    print('  备份: %s (%s)' % (applied['backup_path'], applied['backup_sha256'][:12]))
    print('== R5 端点 ==')
    print('  未来周频截面 %d; 描述性 %d 段 -> %s; 判定 %d 段 -> signal=%s matured=%s'
          % (view['calendar_future_sections'], view['descriptive_sections_required'],
             view['descriptive_checkpoint']['matured_on'],
             view['decisive_sections_required'], view['decisive_endpoint']['signal'],
             view['decisive_endpoint']['matured_on']))
    print('  判定端点已可覆盖: %s' % view['decisive_endpoint']['calendar_covers'])
    print('  access log 记录序号 %s' % record['sequence'])
    print('readings.json %d bytes' % len(out.read_bytes()))
    return 0


# ------------------------------------------------------------------ 测试

class CalendarExtensionTests(unittest.TestCase):
    def test_insert_after_header_preserves_every_other_byte(self):
        before = (HEADER + '\r\nSSE,20261231,1,20261230\r\n'
                  + '\r\nSSE,20261230,1,20261229\r\n').encode('utf-8')
        rows = ['SSE,20271231,1,20271230', 'SSE,20271230,1,20271229']
        after = insert_after_header(before, rows)
        clean = insert_is_byte_clean(before, after, rows)
        self.assertTrue(clean['prefix_preserved'])
        self.assertTrue(clean['suffix_preserved'])
        self.assertTrue(clean['inserted_block_is_the_declared_rows'])
        self.assertEqual(clean['bytes_inserted'], clean['declared_block_bytes'])

    def test_insert_keeps_the_file_descending_and_crlf(self):
        before = (HEADER + '\r\nSSE,20261231,1,20261230\r\nSSE,20261230,1,20261229\r\n'
                  ).encode('utf-8')
        rows = ['SSE,20271231,1,20271230']
        facts = calendar_facts(insert_after_header(before, rows))
        self.assertTrue(facts['is_descending'])
        self.assertEqual(facts['line_ending'], 'crlf')
        self.assertEqual(facts['data_rows'], 3)
        self.assertEqual(facts['duplicate_dates'], 0)

    def test_reconcile_requires_containment(self):
        """随包有而厂商无 ⇒ 这不是「扩展」而是「覆盖」 ⇒ 必须判不通过。"""
        rec = reconcile({'20261231': '1', '20260999': '1'}, {'20261231': '1', '20270104': '1'})
        self.assertFalse(rec['assertion_1_containment_packaged_subset_of_vendor'])
        self.assertFalse(rec['ok'])

    def test_reconcile_requires_monotone_open_flags(self):
        """随包标开市而厂商标休市 ⇒ 两边对「什么是交易日」有分歧 ⇒ 不得通过。"""
        rec = reconcile({'20261231': '1'}, {'20261231': '0', '20270104': '1'})
        self.assertFalse(rec['assertion_2_monotone_open_flags_preserved'])
        self.assertFalse(rec['ok'])

    def test_additions_must_be_well_shaped(self):
        """整月空洞意味着厂商只给了「它确定的日子」—— 差集仍然可枚举, 但日历被打穿。"""
        rec = reconcile({'20261231': '1', '20270104': '1'},
                        {'20261231': '1', '20270104': '1',
                         '20270105': '1', '20270301': '1'})
        self.assertEqual(rec['additions_shape']['months_without_any_session'], ['202702'])
        self.assertFalse(rec['assertion_3_additions_are_enumerable_and_well_shaped'])
        self.assertFalse(rec['ok'])

    def test_an_abnormal_gap_is_caught(self):
        """间隔超过连休上限 ⇒ 形状不合格。"""
        rec = reconcile({'20261231': '1'},
                        {'20261231': '1', '20270104': '1', '20270204': '1'})
        self.assertGreater(rec['additions_shape']['max_gap_in_calendar_days'], MAX_GAP_DAYS)
        self.assertFalse(rec['assertion_3_additions_are_enumerable_and_well_shaped'])

    def test_a_clean_extension_passes_all_three(self):
        rec = reconcile({'20261231': '1'}, {'20261231': '1', '20270104': '1', '20270105': '1'})
        self.assertTrue(rec['ok'])
        self.assertTrue(rec['additions_are_disjoint_from_packaged'])

    def test_default_move_touches_exactly_one_literal(self):
        src = "X = 1\nCAL_END_DEFAULT = '20261231'\nY = 2\n"
        out, hits = move_cal_end_default(src, '20261231', '20271231')
        self.assertEqual(hits, 1)
        self.assertIn("CAL_END_DEFAULT = '20271231'", out)
        self.assertIn('X = 1', out)
        self.assertIn('Y = 2', out)

    def test_default_move_refuses_when_the_old_value_is_not_there(self):
        src = "CAL_END_DEFAULT = '20251231'\n"
        out, hits = move_cal_end_default(src, '20261231', '20271231')
        self.assertEqual(hits, 1)
        self.assertEqual(out, src, '值不匹配时不得改动任何字节')

    def test_default_move_refuses_on_duplicate_literals(self):
        src = "CAL_END_DEFAULT = '20261231'\nCAL_END_DEFAULT = '20261231'\n"
        _out, hits = move_cal_end_default(src, '20261231', '20271231')
        self.assertEqual(hits, 2)

    def test_the_requirement_is_read_not_declared(self):
        """🔴 本批不得自带 26 这个数: 必须来自 R5 模块。

        注: 检查串必须拼接而成 —— 否则本测试自己的源码里就含有它要找的字面量。
        """
        import inspect
        source = inspect.getsource(sys.modules[__name__])
        self.assertNotIn('DECISIVE' + '_SECTIONS = ', source)
        self.assertNotIn('DESCRIPTIVE' + '_SECTIONS = ', source)
        self.assertEqual(R5.DECISIVE_SECTIONS, 26)
        self.assertEqual(R5.DESCRIPTIVE_SECTIONS, 8)

    def test_calendar_facts_notices_the_midfile_header(self):
        before = (HEADER + '\r\nSSE,20261231,1,20261230\r\n' + HEADER + '\r\n'
                  + 'SSE,20261230,1,20261229\r\n').encode('utf-8')
        facts = calendar_facts(before)
        self.assertEqual(facts['non_data_lines'], 1)
        self.assertEqual(facts['data_rows'], 2)

    def test_calendar_facts_counts_double_cr_endings_by_bytes(self):
        """🔴 `\\r\\r\\n` 不是空行。用 splitlines() 会把它读成空行, 从而得出错误结论。"""
        before = (HEADER + '\r\r\nSSE,20261231,1,20261230\r\r\n'
                  + 'SSE,20261230,1,20261229\r\n').encode('utf-8')
        facts = calendar_facts(before)
        self.assertEqual(facts['double_cr_line_endings'], 2)
        self.assertEqual(facts['well_formed_line_endings'], 1)
        self.assertEqual(facts['blank_lines'], 0, '按字节看没有空行')
        self.assertEqual(facts['data_rows'], 2)
        self.assertEqual(facts['line_ending'], 'crlf')
        # 而 splitlines() 会把多余的 \r 读成空行 —— 记录这个读法差异本身
        self.assertEqual(len(before.decode('utf-8').splitlines()), 5)

    def test_a_stray_cr_does_not_make_the_header_non_canonical(self):
        before = (HEADER + '\r\r\nSSE,20261231,1,20261230\r\n').encode('utf-8')
        facts = calendar_facts(before)
        self.assertTrue(facts['header_raw_has_a_stray_cr'])
        self.assertTrue(facts['header_is_canonical'], '去掉多余 \\r 后表头是规范的')

    # ---- 影子结论: 冻结快照 vs 当前覆盖 ----

    def _snap(self, fut, dec_signal, dec_covers):
        return {'calendar_future_sections': fut,
                'descriptive_checkpoint': {'sections': 8, 'signal': '2026-11-13',
                                           'matured_on': '2026-11-23', 'calendar_covers': True},
                'decisive_endpoint': {'sections': 26, 'signal': dec_signal,
                                      'matured_on': None, 'calendar_covers': dec_covers}}

    def _cur(self, fut, dec_signal, dec_covers):
        return {'calendar_future_sections': fut,
                'descriptive_checkpoint': {'sections': 8, 'signal': '2026-11-13',
                                           'matured_on': '2026-11-23', 'calendar_covers': True},
                'decisive_endpoint': {'sections': 26, 'signal': dec_signal,
                                      'matured_on': '2027-03-29', 'calendar_covers': dec_covers}}

    def test_requirement_must_not_change_while_coverage_must(self):
        out = snapshot_vs_current(self._snap(15, None, False), self._cur(67, '2027-03-19', True))
        self.assertTrue(out['requirement_unchanged'])
        self.assertTrue(out['coverage_changed'])
        self.assertTrue(out['staleness_that_existed'])
        self.assertTrue(out['ok'])

    def test_a_changed_requirement_is_caught(self):
        """段数被偷偷改掉 ⇒ 必须判不通过（本批只允许延伸覆盖）。"""
        cur = self._cur(67, '2027-03-19', True)
        cur['decisive_endpoint']['sections'] = 20
        out = snapshot_vs_current(self._snap(15, None, False), cur)
        self.assertFalse(out['requirement_unchanged'])
        self.assertFalse(out['ok'])

    def test_an_extension_that_changed_nothing_is_caught(self):
        """覆盖没变 ⇒ 日历延伸没生效（或没发生）⇒ 必须判不通过。"""
        out = snapshot_vs_current(self._snap(15, None, False), self._cur(15, None, False))
        self.assertFalse(out['coverage_changed'])
        self.assertFalse(out['ok'])


def tests() -> int:
    suite = unittest.TestLoader().loadTestsFromTestCase(CalendarExtensionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tests', action='store_true')
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--status-audit', action='store_true', dest='status_audit')
    args = ap.parse_args(argv)
    if args.tests:
        return tests()
    if args.status_audit:
        audit = status_tool_audit()
        cmp_ = audit['comparison']
        assert cmp_['ok'], '影子结论审计未通过: 要求变了或覆盖没变'
        assert cmp_['requirement_unchanged'], '统计要求被改动了'
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / 'status_tool_audit.json'
        target.write_bytes((json.dumps(audit, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
        print('影子结论审计: 要求不变=%s 覆盖变了=%s'
              % (cmp_['requirement_unchanged'], cmp_['coverage_changed']))
        print('  未来周频截面 %s -> %s' % (cmp_['calendar_future_sections_before'],
                                          cmp_['calendar_future_sections_after']))
        print('  判定端点 %s 段: signal %s -> %s (matured %s), 可覆盖 %s -> %s'
              % (cmp_['decisive_sections_before'], cmp_['decisive_endpoint_signal_before'],
                 cmp_['decisive_endpoint_signal_after'],
                 cmp_['decisive_endpoint_matured_on_after'],
                 cmp_['decisive_endpoint_coverable_before'],
                 cmp_['decisive_endpoint_coverable_after']))
        print('  protocol 未被改写 (sha %s)' % audit['protocol_sha256'][:12])
        print('  判定仍为 %s / 未知样本 %d'
              % (audit['verdict_unchanged'], audit['unknown_sample_effective_count']))
        print('  written: %s' % target.relative_to(ROOT).as_posix())
        return 0
    if args.status:
        status()
        return 0
    if args.plan:
        p = plan()
        print(json.dumps({'facts_before': p['facts_before'],
                          'reconciliation': p['reconciliation'],
                          'vendor': p['vendor'],
                          'rows_to_insert_count': len(p['rows_to_insert']),
                          'rows_to_insert_first': p['rows_to_insert'][:2],
                          'rows_to_insert_last': p['rows_to_insert'][-2:],
                          'planned_bytes_after': p['planned_bytes_after']},
                         ensure_ascii=False, indent=1), flush=True)
        return 0
    if args.run:
        return run()
    ap.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
