# -*- coding: utf-8 -*-
"""_bj_uni_6b_census_20261006.py — §6-B annotate 台账普查（只读）

预注册 §6-B 原文:
  「以现默认 `annotate` 全量跑一次 `entry_tradability`，统计 `uncertifiable_*`
    的板块/年份分布（若 BJ 占比与 M6a 的 80.3% 同构，则方案 B 的收益面被确认）。」

双面普查设计（判据锚在实现，不锚在文件名）
------------------------------------------
  面 1「入口面」(entry face): S5 建仓事件 = 每期 `selected_codes` × `entry_date`。
    数据源 = j17_readings.json 的 `_MEMBERS` 插桩（逐期成员，含 entry/exit 日）。
    **机理预期**: `entry_data = merged[merged.trade_date == entry_date]` 且
    `merged` 为 factor_df ⋈ price 的 inner join ⇒ 建仓日必有行情行 ⇒
    `uncertifiable_*` 恒为 0。若实测为 0，则方案 B（建仓日门控）对 BJ 缺格
    **无收益面**，其预注册机制描述与根因不匹配。
  面 2「缺格面」(cell face): 持仓期缺失格 = `held.loc[day].isna()` 的 (交易日, 证券)。
    数据源 = j17_readings.json 的 `_MISS_CELLS` 插桩（M17 同源）。
    此面把 M17 的 before_list 分类**跨接到** entry_tradability 的判决词表，
    给出"若把门改挂在持有期格上"的收益面与板块/年份分布。

输出: reports/fuxi_v1_bj_universe_20261006/annotate_census.json
"""
import io
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(r'E:\quant')
sys.path.insert(0, str(ROOT / 'research' / 'factor_alchemy'))
import entry_tradability as ET  # noqa: E402

OUT = ROOT / 'reports/fuxi_v1_bj_universe_20261006'
READINGS = OUT / 'j17_readings.json'
CENSUS = OUT / 'annotate_census.json'
RAW = ROOT / 'data' / 'raw'

M6A_BJ_SHARE = 0.803          # M6a 全窗普查: 北交所 920xxx 占内部缺格 51,568/64,255
PRIMARY_ARM = 'A1_legacy_enforce'      # 变更前口径 = 现状面
SECONDARY_ARM = 'A2_listed_enforce'    # 方案 A 生效后口径 (披露)


def dashed(value):
    v = str(value).replace('-', '')
    return '%s-%s-%s' % (v[:4], v[4:6], v[6:8])


def board_of(code):
    c = str(code)
    if c.endswith('.BJ'):
        return 'BJ' if c[:2] == '92' else 'BJ_43xx'
    if c[:3] in ('688', '689'):
        return 'STAR'
    if c[:2] == '30':
        return 'GEM'
    if c[:1] in ('8', '4'):
        return 'NEEQ_prefix'
    return 'MAIN'


def year_of(day):
    return str(day)[:4]


def sessions():
    cal = pd.read_csv(RAW / 'trade_calendar.csv', dtype=str)
    cal = cal[(cal.exchange == 'SSE') & (cal.is_open == '1')]
    days = sorted(pd.to_datetime(cal.cal_date, format='mixed')
                  .dt.strftime('%Y-%m-%d').unique())
    return days


def build_panel(wanted):
    """装配门需要的封存面板（只取需要的键, 不做任何填补）。"""
    codes = {c for c, _ in wanted}
    priced = set()
    reader = pd.read_csv(RAW / 'daily_prices.csv',
                         usecols=['ts_code', 'trade_date'], dtype=str,
                         chunksize=500000)
    for chunk in reader:
        chunk = chunk[chunk.ts_code.isin(codes)]
        for code, raw in zip(chunk.ts_code, chunk.trade_date):
            key = (code, dashed(raw))
            if key in wanted:
                priced.add(key)

    susp = pd.read_csv(RAW / 'suspend_d.csv', dtype=str)
    susp = susp.assign(day=susp.trade_date.map(dashed))
    hf_all = susp[(susp.suspend_type == 'S') & susp.suspend_timing.fillna('').eq('')]
    hp_all = susp[(susp.suspend_type == 'S') & ~susp.suspend_timing.fillna('').eq('')]
    halt_full = {(c, d) for c, d in zip(hf_all.ts_code, hf_all.day) if (c, d) in wanted}
    halt_partial = {(c, d) for c, d in zip(hp_all.ts_code, hp_all.day) if (c, d) in wanted}
    status_days = set(susp.day)

    dl = pd.read_csv(RAW / 'delist_dates.csv', dtype=str)
    dl = dl[dl.list_status.fillna('').eq('D') & dl.delist_date.notna()]
    delist_dates = {c: dashed(d) for c, d in zip(dl.ts_code, dl.delist_date)}

    panel = {'priced': priced, 'halt_full': halt_full, 'halt_partial': halt_partial,
             'status_days': status_days, 'status_terminal': max(status_days),
             'sessions': None, 'delist_dates': delist_dates}
    meta = {'keys_asked': len(wanted), 'priced_found': len(priced),
            'halt_full_found': len(halt_full), 'halt_partial_found': len(halt_partial),
            'status_table_terminal': max(status_days), 'status_rows': int(len(susp)),
            'delisted_securities': len(delist_dates)}
    return panel, meta


def collect_events(readings, arm):
    """从插桩读数抽出两个面的待判键。"""
    entry, cell = [], []
    for cand, rec in (readings['arms'][arm]['runs'] or {}).items():
        for m in rec.get('members') or []:
            for code in m['codes']:
                entry.append({'candidate': cand, 'code': code, 'session': m['entry']})
        for c in rec.get('cells') or []:
            cell.append({'candidate': cand, 'code': c['ts_code'], 'session': c['date']})
    return entry, cell


def tally(rows, panel):
    """逐事件出判决 + 多维分布。"""
    out = []
    for r in rows:
        v = ET.certify(panel, r['code'], r['session'])
        out.append({**r, 'verdict': v['verdict'], 'category': v['category'],
                    'admitted': bool(v['admitted']),
                    'board': board_of(r['code']), 'year': year_of(r['session'])})
    return out


def profile(rows, label):
    """一张面的读数: 总量 / 判决分布 / 类别分布 / 板块×年份 / uncertifiable 的 BJ 占比。"""
    n = len(rows)
    by_verdict = Counter(r['verdict'] for r in rows)
    by_category = Counter(r['category'] for r in rows)
    unc = [r for r in rows if r['category'] == 'uncertifiable']
    bj_unc = sum(1 for r in unc if r['board'].startswith('BJ'))
    board_year_unc = Counter((r['board'], r['year']) for r in unc)
    board_year_all = Counter((r['board'], r['year']) for r in rows)
    return {
        'label': label, 'n_events': n,
        'by_verdict': dict(by_verdict.most_common()),
        'by_category': dict(by_category.most_common()),
        'uncertifiable_n': len(unc),
        'uncertifiable_bj_n': bj_unc,
        'uncertifiable_bj_share': round(bj_unc / len(unc), 4) if unc else None,
        'm6a_reference_bj_share': M6A_BJ_SHARE,
        'isomorphic_to_m6a': (None if not unc else abs(bj_unc / len(unc) - M6A_BJ_SHARE) <= 0.05),
        'board_year_uncertifiable': {'%s|%s' % k: v for k, v in sorted(board_year_unc.items())},
        'board_year_all': {'%s|%s' % k: v for k, v in sorted(board_year_all.items())},
    }


def main():
    print('=== §6-B annotate 台账普查 ===', flush=True)
    print('time:', datetime.now().isoformat(), flush=True)
    print('entry_tradability default mode:', ET.get_mode(), flush=True)
    rd = json.loads(READINGS.read_text(encoding='utf-8'))

    days = sessions()
    print('sessions: %d (%s ~ %s)' % (len(days), days[0], days[-1]), flush=True)

    faces = {}
    all_wanted = set()
    for arm in (PRIMARY_ARM, SECONDARY_ARM):
        entry, cell = collect_events(rd, arm)
        faces[arm] = {'entry': entry, 'cell': cell}
        for r in entry + cell:
            all_wanted.add((r['code'], r['session']))
        print('[%s] entry_events=%d cell_events=%d uniq_keys=%d'
              % (arm, len(entry), len(cell), len(all_wanted)), flush=True)

    panel, meta = build_panel(all_wanted)
    panel['sessions'] = set(days)
    print('panel:', meta, flush=True)

    report = {'schema': 'fuxi_bj_universe_6b_census/v1',
              'generated_at': datetime.now().isoformat(),
              'pre_registration': 'PRE_REGISTRATION_bj_universe.md §6-B',
              'entry_tradability_mode': ET.get_mode(),
              'entry_tradability_modes': list(ET.MODES),
              'verdicts_vocabulary': sorted(ET.VERDICTS),
              'panel_meta': meta, 'sessions_n': len(days),
              'faces': {}}

    for arm in (PRIMARY_ARM, SECONDARY_ARM):
        report['faces'][arm] = {}
        for face in ('entry', 'cell'):
            rows = tally(faces[arm][face], panel)
            prof = profile(rows, '%s/%s' % (arm, face))
            report['faces'][arm][face] = prof
            print('\n[%s · %s面] n=%d' % (arm, face, prof['n_events']), flush=True)
            for k, v in prof['by_category'].items():
                print('    %-22s %7d' % (k, v), flush=True)
            if prof['uncertifiable_n']:
                print('    uncertifiable BJ share = %.4f (M6a ref %.3f, isomorphic=%s)'
                      % (prof['uncertifiable_bj_share'], M6A_BJ_SHARE,
                         prof['isomorphic_to_m6a']), flush=True)

    # ── 逐候选明细 (主面) ──
    per_cand = {}
    for face in ('entry', 'cell'):
        c = Counter()
        for r in tally(faces[PRIMARY_ARM][face], panel):
            c[(r['candidate'], r['category'])] += 1
        per_cand[face] = {'%s|%s' % k: v for k, v in sorted(c.items())}
    report['per_candidate_primary_arm'] = per_cand

    CENSUS.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\ndone ->', CENSUS, flush=True)


if __name__ == '__main__':
    main()
