# -*- coding: utf-8 -*-
"""_bj_uni_j17_analyze_20261006.py — J1–J7 判据读数与对拍（只读后处理）

输入: reports/fuxi_v1_bj_universe_20261006/j17_readings.json（5 臂冻结重放产出）
输出: reports/fuxi_v1_bj_universe_20261006/j17_verdict.json（判据逐条读数）
      reports/fuxi_v1_bj_universe_20261006/gap_ledger_crosswalk.json（与 M17 逐类对拍表）

判据口径（预注册 §3，不得事后改）
--------------------------------
  J1 机制有效性: `北交所无成交日`(= before_list) 格数 228/561/40 → 0；
                 `真停牌`(suspended_full_S) 与 `值级守卫裁剪`(row_present_guard_clip) 逐位不变
  J2 只减不加  : added == 0 且 目标键 (date,ts_code) 集合 ⊆ 变更前；逐期成员仅减
  J3 解耦披露  : 3 候选状态变化只披露不验收
  J4 负对照    : ≥3 个不含 .BJ 暴露的已通过候选，读数逐位不变
  J5 非 BJ     : 非 .BJ 成员的逐期成员关系与读数逐位一致（stable 口径）
  J6 开关化    : legacy 与变更前引擎产物逐位一致（A0 vs A1）
  J7 可回退    : 备份 + sha + 行尾恒 0 裸 LF
"""
import hashlib
import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(r'E:\quant')
OUT = ROOT / 'reports' / 'fuxi_v1_bj_universe_20261006'
RAW = ROOT / 'data' / 'raw'

READINGS = OUT / 'j17_readings.json'
VERDICT = OUT / 'j17_verdict.json'
CROSSWALK = OUT / 'gap_ledger_crosswalk.json'
BAK = OUT / 'pre_bjcut' / 's5_joint_filter.py.bak'
ENG = ROOT / 'research' / 'factor_alchemy' / 's5_joint_filter.py'

M17 = {'w_flow_price_divergence': dict(before_list=228, suspended_full_S=88,
                                       row_present_guard_clip=2),
       'w_flow_trend_orthogonal': dict(before_list=561, suspended_full_S=151,
                                       row_present_guard_clip=12, after_delist=6),
       'weekly_ret_skew_8w_rev': dict(before_list=40, suspended_full_S=91,
                                      row_present_guard_clip=5)}

A0, A1, A2 = 'A0_bak_enforce', 'A1_legacy_enforce', 'A2_listed_enforce'
B1, B2 = 'B1_legacy_shadow', 'B2_listed_shadow'

VOLATILE = {'engine_switches', 'computation_time', 'generated_at', 'elapsed_s'}


def stable(d):
    return {k: v for k, v in (d or {}).items() if k not in VOLATILE}


def dashed(v):
    v = str(v).replace('-', '')
    return '%s-%s-%s' % (v[:4], v[4:6], v[6:8])


# ── M17 原样归因器（逐类对拍的可比性要求: 判定序与源与原版逐位一致） ──
class Attributor:
    """判定序与原版逐位一致; 行情行改为**一次性 prime**（原版每候选扫一次全表,
    本批需 ~18 次分类 ⇒ 改为单次分块扫描建立 {(code,YYYYMMDD): (close, pre_close)} 查表）。"""

    def __init__(self):
        sus = pd.read_csv(RAW / 'suspend_d.csv', dtype=str, encoding='utf-8-sig',
                          keep_default_na=False)
        sus['trade_date'] = sus['trade_date'].astype(str).str.replace(r'\.0$', '', regex=True)
        self.S = set(zip(sus.loc[sus.suspend_type == 'S', 'ts_code'],
                         sus.loc[sus.suspend_type == 'S', 'trade_date']))
        self.any_halt = set(zip(sus.ts_code, sus.trade_date))
        dl = pd.read_csv(RAW / 'delist_dates.csv', dtype=str, encoding='utf-8-sig',
                         keep_default_na=False)
        self.dl_meta = {str(r['ts_code']): (str(r.get('list_date') or ''),
                                            str(r.get('delist_date') or ''))
                        for _, r in dl.iterrows()}
        self.px = {}

    def prime(self, cells, chunk=500000):
        want, codes = set(), set()
        for c in cells:
            ts, d8 = str(c['ts_code']), str(c['date']).replace('-', '')
            want.add((ts, d8))
            codes.add(ts)
        reader = pd.read_csv(RAW / 'daily_prices.csv',
                             usecols=['ts_code', 'trade_date', 'close', 'pre_close'],
                             dtype={'trade_date': 'str'}, encoding='utf-8-sig',
                             low_memory=False, chunksize=chunk)
        for ch in reader:
            ch = ch[ch.ts_code.isin(codes)]
            for code, raw, close, pre in zip(ch.ts_code, ch.trade_date,
                                             ch.close, ch.pre_close):
                # 🔴 key 归一化到与 classify 同一口径 (无横线 8 位), 否则查表恒空
                d8 = str(raw).strip().replace('-', '')
                if d8.endswith('.0'):
                    d8 = d8[:-2]
                key = (code, d8)
                if key in want:
                    self.px[key] = (close, pre)
        return {'asked': len(want), 'found': len(self.px), 'codes': len(codes)}

    def classify(self, cells):
        """cells: [{date:'YYYY-MM-DD', ts_code}] → Counter of M17 类别（纯查表, 无 IO）。"""
        out = Counter()
        for c in cells or []:
            ts, d8 = str(c['ts_code']), str(c['date']).replace('-', '')
            key = (ts, d8)
            if (ts, d8) in self.S:
                out['suspended_full_S'] += 1
                continue
            if (ts, d8) in self.any_halt:
                out['suspended_other_R'] += 1
                continue
            if key in self.px:
                close, pre = self.px[key]
                close = None if pd.isna(close) else close
                pre = None if pd.isna(pre) else pre
                if close is None:
                    out['row_present_close_nan'] += 1
                elif pre is None or float(pre) == 0:
                    out['row_present_preclose_bad'] += 1
                else:
                    r = float(close) / float(pre) - 1.0
                    if ts.endswith('.BJ') or ts[:1] in ('4', '8'):
                        lim = 0.30
                    elif ts[:2] in ('30', '68'):
                        lim = 0.20
                    else:
                        lim = 0.10
                    out['row_present_guard_clip' if abs(r) > lim + 0.005
                        else 'row_present_other'] += 1
                continue
            ld, dd = self.dl_meta.get(ts, ('', ''))
            if dd and d8 >= dd:
                out['after_delist'] += 1
            elif ld and d8 < ld:
                out['before_list'] += 1
            else:
                out['price_row_missing'] += 1
        return out


def cellset(rec):
    return {(c['date'], c['ts_code']) for c in rec.get('cells') or []}


def members_map(rec):
    return {m['entry']: set(m['codes']) for m in rec.get('members') or []}


def board_of(code):
    c = str(code)
    if c.endswith('.BJ'):
        return 'BJ_92' if c[:2] == '92' else 'BJ_43'
    if c[:3] in ('688', '689'):
        return 'STAR'
    if c[:2] == '30':
        return 'GEM'
    if c[:1] in ('8', '4'):
        return 'NEEQ_prefix'
    return 'MAIN'


def measure_cut_surface(chunk=500000):
    """J5a 行面量化: 直接调用生产实现 `_s5_bj_cut_pre_listing` 扫全窗行情面板。

    同时给出"非 .BJ 行零触碰"的**全窗实证**（不只靠逻辑与的同义反复）。
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'bj_j17_replay_mod', str(ROOT / 'scripts' / '_bj_uni_j17_replay_20261006.py'))
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    mod = rp.build_module(rp.ENG_CUR, [], [])
    lds = mod._bj_listing_date_map()

    tot = bj = dropped = nonbj_in = nonbj_out = 0
    by_year, dropped_codes, by_year_kept = Counter(), set(), Counter()
    reader = pd.read_csv(RAW / 'daily_prices.csv', usecols=['ts_code', 'trade_date'],
                         dtype={'ts_code': 'str', 'trade_date': 'str'},
                         encoding='utf-8-sig', chunksize=chunk)
    for ch in reader:
        ch = ch.assign(trade_date=pd.to_datetime(ch.trade_date, format='mixed',
                                                 errors='coerce'))
        tot += len(ch)
        is_bj = ch.ts_code.str.endswith('.BJ')
        bj += int(is_bj.sum())
        nonbj_in += int((~is_bj).sum())
        out = mod._s5_bj_cut_pre_listing(ch)          # ← 生产实现原样调用
        nonbj_out += int((~out.ts_code.str.endswith('.BJ')).sum())
        d = ch[is_bj]
        d = d[d.trade_date < d.ts_code.map(lds)]
        d = d[d.trade_date.notna()]
        dropped += len(d)
        dropped_codes.update(d.ts_code.unique())
        for y, n in d.trade_date.dt.year.value_counts().items():
            by_year[str(y)] += int(n)
    return {'generated_from': 'daily_prices.csv (full panel, chunked)',
            'listing_codes_loaded': len(lds),
            'rows_total': tot, 'bj_rows': bj, 'nonbj_rows': nonbj_in,
            'dropped_rows': dropped, 'dropped_codes': len(dropped_codes),
            'dropped_by_year': dict(sorted(by_year.items())),
            'nonbj_dropped': nonbj_in - nonbj_out,
            'nonbj_rows_pass_through_identical': nonbj_in == nonbj_out}


def strict_j4():
    """J4 严格判定: 合并 j4/j4b 两批专项复跑, 只认"真跑出回测(periods>0) ∧ 零 .BJ 暴露
    ∧ 两臂读数逐位一致"者 —— 首轮主重放的 3 个价格水平族候选属**前置门早退**, 无读数,
    不构成负对照 (须计入"不可用"而非"通过")。"""
    files = [OUT / n for n in ('j4_readings.json', 'j4b_readings.json', 'j4c_readings.json')]
    cands, blocked, na = {}, {}, []
    for f in files:
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding='utf-8'))
        leg = (d.get('arms') or {}).get('legacy') or {}
        lis = (d.get('arms') or {}).get('listed_only') or {}
        for n, r in leg.items():
            lr = lis.get(n)
            if r.get('members_n', 0) <= 0 or r['ledger'].get('missing_held_price_cells') is None:
                blocked[n] = {'reason': r.get('reason_head') or r.get('error'),
                              'ran': r.get('ran')}
                continue
            if r.get('held_bj'):
                na.append({'candidate': n, 'held_bj_n': len(r['held_bj']),
                           'missing': r['ledger']['missing_held_price_cells']})
                continue
            if not lr:
                na.append({'candidate': n, 'note': 'listed_only 臂缺读数'})
                continue
            ra, rb = stable(r['readings']), stable(lr['readings'])
            diffs = sorted(k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k))
            s1 = {(c['date'], c['ts_code']) for c in (r.get('cells') or [])}
            s2 = {(c['date'], c['ts_code']) for c in (lr.get('cells') or [])}
            cands[n] = {'bj_exposure_free': True,
                        'periods_legacy': r['members_n'], 'periods_listed': lr.get('members_n'),
                        'missing_legacy': r['ledger']['missing_held_price_cells'],
                        'missing_listed': lr['ledger']['missing_held_price_cells'],
                        'readings_identical': len(diffs) == 0, 'diff_keys': diffs,
                        'diff_values': {k: [ra.get(k), rb.get(k)] for k in diffs[:10]},
                        'cells_identical': s1 == s2,
                        'cells_legacy': len(s1), 'cells_listed': len(s2)}
    ok = [n for n, v in cands.items() if v['readings_identical']]
    return {'policy': '真跑出回测(periods>0) ∧ 零 .BJ 暴露 ∧ 两臂读数逐位一致',
            'n_valid_controls': len(cands), 'n_readings_identical': len(ok),
            'qualifying': ok, 'verdict': len(ok) >= 3,
            'controls': cands,
            'excluded_bj_exposed': na,
            'excluded_blocked_no_readings': blocked}


def main():
    rd = json.loads(READINGS.read_text(encoding='utf-8'))
    arms = rd['arms']
    att = Attributor()
    _all = []
    for _a in arms.values():
        for _rec in (_a.get('runs') or {}).values():
            _all.extend(_rec.get('cells') or [])
    print('[prime] 一次性装载 %d 格行情明细...' % len(_all), flush=True)
    print('        ', att.prime(_all), flush=True)
    report = {'schema': 'fuxi_bj_universe_j17/v1',
              'generated_at': datetime.now().isoformat(),
              'source': str(READINGS), 'criteria': {}, 'crosswalk': {}}

    print('=== J1–J7 判据读数 ===', flush=True)

    # ═══ 接线自查（开关"写了但没生效"是本项目的已知坑） ═══
    wire = {}
    for armid in (A0, A1, A2, B1, B2):
        a = arms.get(armid) or {}
        for cname in sorted(a.get('runs') or {}):
            sw = (a['runs'][cname].get('ledger') or {}).get('switches') or {}
            wire.setdefault(armid, {})[cname] = {
                'holdings_mark': sw.get('holdings_mark'),
                'bj_universe': sw.get('bj_universe'),
                'missing': a['runs'][cname]['ledger'].get('missing_held_price_cells')}
    ok_judged = {}
    for armid, key, want in ((A1, 'bj_universe', 'legacy'), (A2, 'bj_universe', 'listed_only'),
                             (A1, 'holdings_mark', 'enforce'), (B2, 'holdings_mark', 'shadow_carry_forward')):
        vs = [v for v in (wire.get(armid) or {}).values() if v.get(key) is not None]
        # 🔴 早退候选不产生 diag (switches 空) ⇒ 未发生项不得计入判定, 单列披露
        ok_judged['%s.%s' % (armid, key)] = bool(vs) and all(v.get(key) == want for v in vs)
    ok_a1 = ok_judged[A1 + '.bj_universe']
    ok_a2 = ok_judged[A2 + '.bj_universe']
    ok_mark = ok_judged[A1 + '.holdings_mark']
    ok_shadow = ok_judged[B2 + '.holdings_mark']
    report['criteria']['wire_check'] = {
        'A1_bj_universe_all_legacy': ok_a1, 'A2_bj_universe_all_listed_only': ok_a2,
        'A1_holdings_mark_all_enforce': ok_mark, 'B2_holdings_mark_all_shadow': ok_shadow,
        'verdict': bool(ok_a1 and ok_a2 and ok_mark and ok_shadow),
        'n_judged': {k: sum(1 for v in (wire.get(k.split('.')[0]) or {}).values()
                            if v.get(k.split('.')[1]) is not None) for k in ok_judged},
        'excluded_no_diag': {k: sorted(n for n, v in (wire.get(k) or {}).items()
                                       if v.get('bj_universe') is None
                                       and v.get('holdings_mark') is None)
                             for k in (A1, A2, B1, B2)},
        'detail': wire}
    print('\n[wire] A1 legacy=%s A2 listed_only=%s enforce=%s shadow=%s'
          % (ok_a1, ok_a2, ok_mark, ok_shadow), flush=True)

    # ═══ J1 机制有效性 ═══
    j1 = {}
    print('\n[J1] 机制有效性 (A1 legacy → A2 listed_only)', flush=True)
    for cname, exp in M17.items():
        a1 = arms[A1]['runs'][cname]
        a2 = arms[A2]['runs'][cname]
        cells1, cells2 = a1.get('cells') or [], a2.get('cells') or []
        c1 = att.classify(cells1)
        c2 = att.classify(cells2)
        s1, s2 = cellset(a1), cellset(a2)
        rem, add = s1 - s2, s2 - s1
        rem_c = att.classify([{'date': d, 'ts_code': t} for d, t in sorted(rem)])
        add_c = att.classify([{'date': d, 'ts_code': t} for d, t in sorted(add)])
        rem_board = Counter(board_of(t) for _, t in rem)
        add_board = Counter(board_of(t) for _, t in add)
        keep_ok, keep_detail = True, {}
        for k in ('suspended_full_S', 'row_present_guard_clip', 'row_present_other',
                  'row_present_close_nan', 'row_present_preclose_bad', 'after_delist',
                  'suspended_other_R'):
            v1, v2 = c1.get(k, 0), c2.get(k, 0)
            # 「不变」只对 M17 中实际出现过的类作硬约束；其余类披露
            if k in exp:
                keep_detail[k] = {'m17': exp[k], 'legacy': v1, 'listed': v2,
                                  'unchanged': v1 == v2}
                keep_ok &= (v1 == v2)
            else:
                keep_detail[k] = {'m17': exp.get(k), 'legacy': v1, 'listed': v2,
                                  'unchanged': v1 == v2}
        j1[cname] = {
            'cells_legacy': len(a1.get('cells') or []), 'cells_listed': len(a2.get('cells') or []),
            'before_list_m17': exp['before_list'],
            'before_list_legacy': c1.get('before_list', 0),
            'before_list_listed': c2.get('before_list', 0),
            'target_class_zeroed': c2.get('before_list', 0) == 0,
            'preserved_class_counts_unchanged': bool(keep_ok),
            'class_detail': keep_detail,
            'legacy_cats_full': dict(c1.most_common()),
            'listed_cats_full': dict(c2.most_common()),
            'cell_diff': {
                'removed_n': len(rem), 'added_n': len(add),
                'removed_by_class': dict(rem_c.most_common()),
                'added_by_class': dict(add_c.most_common()),
                'removed_by_board': dict(rem_board.most_common()),
                'added_by_board': dict(add_board.most_common()),
                # 目标类是否解释全部 removed（BJ 无成交日 + .BJ 代码的目标代码段位移）
                'removed_all_bj': all(t.endswith('.BJ') for _, t in rem),
                'removed_target_class_n': rem_c.get('before_list', 0),
                'removed_other_class_n': len(rem) - rem_c.get('before_list', 0),
                'added_samples': [{'date': d, 'ts_code': t} for d, t in sorted(add)][:10],
                'removed_nonbj_samples': [
                    {'date': d, 'ts_code': t} for d, t in sorted(rem)
                    if not t.endswith('.BJ')][:12],
            },
            'missing_legacy': a1['ledger']['missing_held_price_cells'],
            'missing_listed': a2['ledger']['missing_held_price_cells'],
            'days_legacy': a1['ledger']['incomplete_marking_days'],
            'days_listed': a2['ledger']['incomplete_marking_days'],
        }
        v = j1[cname]
        print('  %-28s before_list %4d→%4d (m17 %4d) | preserved_ok=%s | cells %4d→%4d'
              % (cname, v['before_list_legacy'], v['before_list_listed'],
                 v['before_list_m17'], v['preserved_class_counts_unchanged'],
                 v['cells_legacy'], v['cells_listed']), flush=True)
    report['criteria']['J1'] = {
        'verdict': all(v['target_class_zeroed'] and v['preserved_class_counts_unchanged']
                       for v in j1.values()),
        'per_candidate': j1}

    # ═══ J2 只减不加 ═══
    j2 = {}
    print('\n[J2] 只减不加', flush=True)
    for cname in arms[A1]['runs']:
        r1, r2 = arms[A1]['runs'].get(cname), arms[A2]['runs'].get(cname)
        if not r1 or not r2:
            continue
        s1, s2 = cellset(r1), cellset(r2)
        removed, added = s1 - s2, s2 - s1
        nb1 = {c for c in s1 if not str(c[1]).endswith('.BJ')}
        nb2 = {c for c in s2 if not str(c[1]).endswith('.BJ')}
        m1, m2 = members_map(r1), members_map(r2)
        periods = sorted(set(m1) | set(m2))
        grew, shrank, same_periods = [], [], 0
        bj_removed_members, nonbj_added_members = Counter(), Counter()
        for p in periods:
            A, B = m1.get(p, set()), m2.get(p, set())
            if B - A:
                grew.append({'entry': p, 'added': sorted(B - A)})
                for x in (B - A):
                    if x.endswith('.BJ'):
                        bj_removed_members['__BJ_ADDED__'] += 1
                    else:
                        nonbj_added_members[x] += 1
            elif A - B:
                shrank.append({'entry': p, 'removed': sorted(A - B)})
                for x in (A - B):
                    if x.endswith('.BJ'):
                        bj_removed_members['BJ_removed'] += 1
            else:
                same_periods += 1
        j2[cname] = {
            'cells_removed': len(removed), 'cells_added': len(added),
            'cells_subset_ok': len(added) == 0,
            'nonbj_cells_legacy_n': len(nb1), 'nonbj_cells_listed_n': len(nb2),
            'nonbj_cells_identical': nb1 == nb2,
            'nonbj_cells_removed': len(nb1 - nb2), 'nonbj_cells_added': len(nb2 - nb1),
            'membership_periods_legacy': len(m1), 'membership_periods_listed': len(m2),
            'periods_identical_membership': same_periods,
            'periods_shrank': len(shrank), 'periods_grew': len(grew),
            'bj_removed_member_slots': bj_removed_members.get('BJ_removed', 0),
            'bj_added_member_slots': bj_removed_members.get('__BJ_ADDED__', 0),
            'nonbj_added_member_slots': sum(nonbj_added_members.values()),
            'nonbj_added_codes': nonbj_added_members.most_common(20),
            'grew_samples': grew[:5], 'shrank_samples': shrank[:5],
            'added_cells_samples': sorted(added)[:20],
        }
        print('  %-28s cells removed=%d added=%d subset_ok=%s | periods same=%d grew=%d shrank=%d'
              % (cname, len(removed), len(added), len(added) == 0, same_periods,
                 len(grew), len(shrank)), flush=True)
    _j2_act = [k for k, v in j2.items() if v['membership_periods_legacy'] > 0]
    report['criteria']['J2'] = {
        'caliber_note': (
            '预注册 J2 原文「全窗统计 removed>0 且 added==0；每个目标键证券集合 ⊆ 变更前集合'
            '（与 entry_tradability 不变量 3 同构）」。entry_tradability 的不变量 3 锚在'
            '**逐期目标帧的证券集合**；而方案 A 改的是**池成员行**，且 `selected = '
            'entry_data.head(top_n)` 的头部选择会由第 N+1 名递补 ⇒ 成员集合层面按字母面必然'
            '出现 added>0。两类口径须分列：J2a=台账键/数据行面（成立预期）、'
            'J2b=逐期成员面（不成立预期, 机制性）。早退候选 (无回测, periods=0) 单列, '
            '不计入"通过", 也不计为反例。'),
        'J2a_ledger_key_level_verdict': all(v['cells_subset_ok'] for v in j2.values()),
        'J2a_removed_gt0_among_active': all(j2[k]['cells_removed'] > 0 for k in _j2_act),
        'J2b_membership_level_verdict': all(j2[k]['periods_grew'] == 0 for k in _j2_act),
        'J2c_nonbj_ledger_slice_verdict': all(v['nonbj_cells_identical'] for v in j2.values()),
        'judged_candidates': _j2_act,
        'no_readings_candidates': [k for k in j2 if k not in _j2_act],
        'removed_cells_total': sum(v['cells_removed'] for v in j2.values()),
        'added_cells_total': sum(v['cells_added'] for v in j2.values()),
        'refill_slots_total': sum(v['nonbj_added_member_slots'] for v in j2.values()),
        'per_candidate': j2}

    # ═══ J3 解耦披露 ═══
    j3 = {}
    for cname in M17:
        a1, a2 = arms[A1]['runs'][cname], arms[A2]['runs'][cname]
        j3[cname] = {'legacy': {'passed': a1['readings'].get('passed'),
                                'error': a1['error'],
                                'reason': (a1['readings'].get('reason') or '')[:160],
                                'economic_readings_status': a1['readings'].get('economic_readings_status')},
                     'listed_only': {'passed': a2['readings'].get('passed'),
                                     'error': a2['error'],
                                     'reason': (a2['readings'].get('reason') or '')[:160],
                                     'economic_readings_status': a2['readings'].get('economic_readings_status')}}
        j3[cname]['status_changed'] = (
            j3[cname]['legacy']['error'] != j3[cname]['listed_only']['error']
            or j3[cname]['legacy']['passed'] != j3[cname]['listed_only']['passed'])
    print('\n[J3] 解耦披露 (只披露不验收)', flush=True)
    for k, v in j3.items():
        print('  %-28s status_changed=%s | legacy err=%r → listed err=%r'
              % (k, v['status_changed'], v['legacy']['error'][:40], v['listed_only']['error'][:40]),
              flush=True)
    report['criteria']['J3'] = {'note': '只作披露, 不作验收; 若变可裁决须单独归档并声明不构成因子有效性证据',
                                'per_candidate': j3}

    # ═══ J4 负对照 ═══
    #   主重放臂内的候选只作**尝试记录**; 权威判定取专项复跑 (j4/j4b) 的严格口径。
    j4_try = {}
    print('\n[J4] 负对照: 主重放臂内尝试记录', flush=True)
    for cname in [c for c in arms[A1]['runs'] if c.startswith('gp_')]:
        a1, a2 = arms[A1]['runs'][cname], arms[A2]['runs'][cname]
        held1 = {x for m in (a1.get('members') or []) for x in m['codes']}
        held2 = {x for m in (a2.get('members') or []) for x in m['codes']}
        bj1 = sorted(x for x in held1 if x.endswith('.BJ'))
        bj2 = sorted(x for x in held2 if x.endswith('.BJ'))
        s1, s2 = cellset(a1), cellset(a2)
        r1, r2 = stable(a1['readings']), stable(a2['readings'])
        diff_keys = sorted(k for k in set(r1) | set(r2) if r1.get(k) != r2.get(k))
        j4_try[cname] = {
            'periods_legacy': a1.get('members_n_periods'),
            'ran_backtest': bool(a1.get('members_n_periods')),
            'bj_held_legacy': bj1[:20], 'bj_held_listed': bj2[:20],
            'bj_held_n_legacy': len(bj1), 'bj_held_n_listed': len(bj2),
            'bj_exposure_free': (len(bj1) == 0 and len(bj2) == 0),
            'cells_identical': s1 == s2,
            'readings_identical': len(diff_keys) == 0,
            'diff_keys': diff_keys,
            'diff_values': {k: [r1.get(k), r2.get(k)] for k in diff_keys[:12]},
            'passed_legacy': a1['readings'].get('passed'),
            'error_legacy': a1['error'][:80],
            'reason_head': (a1['readings'].get('reason') or '')[:120],
            'usable_as_control': bool(a1.get('members_n_periods')) and len(bj1) == 0,
        }
        print('  %-16s periods=%-4s bj_held=%d/%d exposure_free=%-5s readings_same=%-5s | %s'
              % (cname, a1.get('members_n_periods'), len(bj1), len(bj2),
                 j4_try[cname]['bj_exposure_free'], len(diff_keys) == 0,
                 j4_try[cname]['reason_head'][:56] or '-'), flush=True)
    j4 = strict_j4()
    print('  [J4-strict] 有效对照=%d 读数一致=%d 判定=%s | %s'
          % (j4['n_valid_controls'], j4['n_readings_identical'], j4['verdict'],
             j4['qualifying']), flush=True)
    report['criteria']['J4'] = {**j4, 'main_replay_attempts': j4_try}

    # ═══ J5 非 BJ：三口径读数（口径差异必须显式，不得混排） ═══
    #   J5a 行面(数据通道): 裁剪只 drop .BJ 行 ⇒ 非 .BJ 行/值逐位不变 —— 实现级同义反复
    #                        (`_drop = _bj & ...` 逻辑与) + 全窗量化验证
    #   J5b 成员面(字面):   非 .BJ 逐期成员关系是否逐位一致 —— 受 `head(top_n)`
    #                        头部选择递补影响, 须量化递补槽位与来源板
    #   J5c 读数面:         组合读数是否逐位一致 —— 零 .BJ 暴露候选预期一致(=J4)
    j5 = {}
    print('\n[J5] 非 .BJ 三口径', flush=True)
    for cname in arms[A1]['runs']:
        m1, m2 = members_map(arms[A1]['runs'][cname]), members_map(arms[A2]['runs'][cname])
        periods = sorted(set(m1) & set(m2))
        bad = []
        refill_boards = Counter()
        for p in periods:
            A = {x for x in m1[p] if not x.endswith('.BJ')}
            B = {x for x in m2[p] if not x.endswith('.BJ')}
            if A != B:
                added = sorted(B - A)
                bad.append({'entry': p, 'only_legacy': sorted(A - B)[:6],
                            'only_listed': added[:6]})
                for x in added:
                    refill_boards[board_of(x)] += 1
        bj1 = {x for m in m1.values() for x in m if x.endswith('.BJ')}
        j5[cname] = {
            'periods_compared': len(periods),
            'nonbj_membership_differ_periods': len(bad),
            'nonbj_membership_identical': len(bad) == 0,
            'refill_slots': sum(refill_boards.values()),
            'refill_boards': dict(refill_boards.most_common()),
            'diff_samples': bad[:6],
            'bj_member_slots_legacy': sum(1 for m in m1.values() for x in m if x.endswith('.BJ')),
            'bj_codes_legacy': sorted(bj1)[:20],
            'readings_identical': (stable(arms[A1]['runs'][cname]['readings'])
                                   == stable(arms[A2]['runs'][cname]['readings'])),
        }
        print('  %-28s periods=%d nonbj_diff=%d refill_slots=%d identical=%-5s'
              % (cname, len(periods), len(bad), j5[cname]['refill_slots'],
                 len(bad) == 0), flush=True)

    # ── J5a: 全窗裁剪面量化（行面, 直接锚在实现） ──
    cut_surface = measure_cut_surface()
    print('  [J5a] 全窗裁剪面: 总行=%d | .BJ 行=%d | 裁掉(上市前)=%d | 非 .BJ 被裁=%d'
          % (cut_surface['rows_total'], cut_surface['bj_rows'],
             cut_surface['dropped_rows'], cut_surface['nonbj_dropped']), flush=True)

    report['criteria']['J5'] = {
        'caliber_note': (
            '预注册 J5 原文「非 .BJ 代码的所有逐日成员关系与因子读数逐位一致」。'
            '实现层面 `_s5_bj_cut_pre_listing` 的 `_drop = _bj & _cut.notna() & (date<cut)` '
            '是逻辑与 ⇒ 非 .BJ 行**结构性不可被裁**, J5a 成立(行面)。'
            '但 `selected = entry_data.head(top_n)` 的头部选择会在 .BJ 出局后由第 N+1 名'
            '**递补**入组合 ⇒ 非 .BJ 逐期成员与组合读数在**成员面/读数面**必然变化, '
            'J5b/J5c 按字母面不成立, 属机制性后果而非泄漏。三口径须分列读数, 不得混排。'),
        'J5a_row_level_verdict': cut_surface['nonbj_dropped'] == 0,
        # 🔴 口径纪律: 早退候选 (periods=0) 的空集相等**不得**计入"通过"(未发生项须单列)
        'J5b_membership_level_verdict': all(j5[k]['nonbj_membership_identical']
                                            for k in j5 if j5[k]['periods_compared'] > 0),
        'J5c_readings_level_verdict': all(j5[k]['readings_identical']
                                          for k in j5 if j5[k]['periods_compared'] > 0),
        'judged_candidates': [k for k in j5 if j5[k]['periods_compared'] > 0],
        'no_readings_candidates': [k for k in j5 if j5[k]['periods_compared'] == 0],
        'cut_surface': cut_surface,
        'ic_channel_note': (
            'IC/ICIR 由 self._price_df 直算 (compute_rank_ic / _daily_ic_series), '
            '不经过被裁剪的 price 面板 ⇒ 结构性免疫于本开关。'),
        'per_candidate': j5}

    # ═══ J6 开关化默认 legacy ═══
    j6 = {}
    print('\n[J6] legacy 与变更前引擎逐位一致', flush=True)
    for cname in M17:
        a0, a1 = arms[A0]['runs'][cname], arms[A1]['runs'][cname]
        r0, r1 = stable(a0['readings']), stable(a1['readings'])
        diffs = sorted(k for k in set(r0) | set(r1) if r0.get(k) != r1.get(k))
        s0, s1 = cellset(a0), cellset(a1)
        j6[cname] = {
            'readings_identical': len(diffs) == 0, 'diff_keys': diffs,
            'diff_values': {k: [r0.get(k), r1.get(k)] for k in diffs[:10]},
            'cells_identical': s0 == s1,
            'cells_a0': len(s0), 'cells_a1': len(s1),
            'missing_a0': a0['ledger']['missing_held_price_cells'],
            'missing_a1': a1['ledger']['missing_held_price_cells'],
            'm17_exp': a1.get('exp_missing'),
            'pair_match_a0': a0.get('pair_match_m17'), 'pair_match_a1': a1.get('pair_match_m17'),
            'switches_a0': a0['ledger'].get('switches'), 'switches_a1': a1['ledger'].get('switches'),
        }
        print('  %-28s readings_same=%-5s cells_same=%-5s | missing %s/%s (m17 %s)'
              % (cname, len(diffs) == 0, s0 == s1, a0['ledger']['missing_held_price_cells'],
                 a1['ledger']['missing_held_price_cells'], a1.get('exp_missing')), flush=True)
    report['criteria']['J6'] = {
        'verdict': all(v['readings_identical'] and v['cells_identical'] for v in j6.values()),
        'note': 'engine_switches 含 bj_universe 元数据差异, 按 volatile 剔除; 读数与台账逐位比对',
        'per_candidate': j6}

    # ═══ 影子档披露 (现行默认) ═══
    sh = {}
    for cname in M17:
        b1, b2 = arms[B1]['runs'][cname], arms[B2]['runs'][cname]
        r1, r2 = stable(b1['readings']), stable(b2['readings'])
        d = sorted(k for k in set(r1) | set(r2) if r1.get(k) != r2.get(k))
        sh[cname] = {
            'legacy': {'error': b1['error'][:120], 'passed': b1['readings'].get('passed'),
                       'missing': b1['ledger']['missing_held_price_cells'],
                       'days': b1['ledger']['incomplete_marking_days'],
                       'carried': b1['ledger']['carried_forward_cells'],
                       'exceeded': b1['ledger']['carry_forward_exceeded_cells']},
            'listed_only': {'error': b2['error'][:120], 'passed': b2['readings'].get('passed'),
                            'missing': b2['ledger']['missing_held_price_cells'],
                            'days': b2['ledger']['incomplete_marking_days'],
                            'carried': b2['ledger']['carried_forward_cells'],
                            'exceeded': b2['ledger']['carry_forward_exceeded_cells']},
            'readings_identical': len(d) == 0, 'diff_keys': d, 'n_diff_keys': len(d),
        }
    print('\n[影子档·现行默认 shadow_carry_forward]', flush=True)
    for k, v in sh.items():
        print('  %-28s missing %s→%s | carried %s→%s | readings_same=%s (diffs %d)'
              % (k, v['legacy']['missing'], v['listed_only']['missing'],
                 v['legacy']['carried'], v['listed_only']['carried'],
                 v['readings_identical'], v['n_diff_keys']), flush=True)
    report['criteria']['shadow_disclosure'] = sh

    # ═══ J7 可字节级回退 ═══
    bak_raw = io.open(BAK, 'rb').read()
    eng_raw = io.open(ENG, 'rb').read()
    j7 = {'backup_exists': BAK.exists(),
          'backup_sha256': hashlib.sha256(bak_raw).hexdigest(),
          'current_sha256': hashlib.sha256(eng_raw).hexdigest(),
          'backup_bytes': len(bak_raw), 'current_bytes': len(eng_raw),
          'backup_bare_lf': io.open(BAK, 'r', encoding='utf-8', newline='').read().count('\n')
                            - io.open(BAK, 'r', encoding='utf-8', newline='').read().count('\r\n'),
          'current_bare_lf': io.open(ENG, 'r', encoding='utf-8', newline='').read().count('\n')
                             - io.open(ENG, 'r', encoding='utf-8', newline='').read().count('\r\n')}
    j7['verdict'] = (j7['backup_exists'] and j7['backup_bare_lf'] == 0
                     and j7['current_bare_lf'] == 0)
    print('\n[J7] 字节级回退: bak=%s sha=%s bareLF(bak/cur)=%d/%d'
          % (j7['backup_exists'], j7['backup_sha256'][:16], j7['backup_bare_lf'],
             j7['current_bare_lf']), flush=True)
    report['criteria']['J7'] = j7

    # ═══ 与 M17 逐类对拍表 ═══
    cw = {'schema': 'fuxi_bj_universe_m17_crosswalk/v1',
          'generated_at': datetime.now().isoformat(),
          'm17_source': 'reports/fuxi_p2m4_m17_reeval_20260928/stage2_gap_ledger.json',
          'note': '归因器与 M17 原版逐序一致; delist_dates 为 list_date 源 (不含 overlay 补录)',
          'rows': {}}
    for cname in M17:
        c1 = att.classify(arms[A1]['runs'][cname].get('cells') or [])
        c2 = att.classify(arms[A2]['runs'][cname].get('cells') or [])
        cats = sorted(set(M17[cname]) | set(c1) | set(c2))
        cw['rows'][cname] = {k: {'m17': M17[cname].get(k, 0), 'legacy': c1.get(k, 0),
                                 'listed_only': c2.get(k, 0),
                                 'preserved': M17[cname].get(k, 0) == c1.get(k, 0)}
                             for k in cats}
    CROSSWALK.write_text(json.dumps(cw, ensure_ascii=False, indent=1), encoding='utf-8')
    report['crosswalk'] = cw['rows']

    VERDICT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n--- 判据汇总 ---', flush=True)
    for k in ('J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'wire_check'):
        v = report['criteria'][k]
        keys = [x for x in v if any(t in x.lower()
                                    for t in ('verdict', 'qualify', 'identi', 'n_valid'))]
        print('  %-10s %s' % (k, {x: v[x] for x in keys}), flush=True)
    print('\ndone ->', VERDICT, flush=True)


if __name__ == '__main__':
    main()
