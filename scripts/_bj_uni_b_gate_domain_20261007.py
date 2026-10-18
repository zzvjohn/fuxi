# -*- coding: utf-8 -*-
"""_bj_uni_b_gate_domain_20261007.py — 方案 B 的实际收益面实测（只读）

为什么必须单独测
----------------
`annotate_census.json`（§6-B）测的是 **S5 重放的持码格**（缺格自身日期作 session），
得到 `uncertifiable = 876 / bj_share = 1.0`。但 `ET.gate()` 在生产上的判定域是
**执行账本冻结目标帧**：
    frames: {stamp: {code: weight}}  →  session = effective_session(stamp, days)
（`entry_tradability.py:140-146, 200-271`）

两个域的差异是**实质的**:
  · S5 窗口约 2021–2026（含 2021–2023 北交所上市潮 ⇒ BJ 上市前缺格 876 格全部落此）；
  · 冻结目标帧窗口 = **2024-12-27 .. 2026-09-09**（= 2025/2026 全部为 BJ 上市后）。
⇒ 若直接拿 876 当 B 的拒绝量，会把"历史窗口的读数"挂到"执行边界的数据列"上。

本脚本按 `gate()` **自身判定域**逐 `(stamp, code)` 实测三档读数，给出:
  1. 各臂 domain 规模 / 判决分布 / 类别分布（等价于 enforce 的拒绝量）；
  2. BJ 与非 BJ 的**分列**计数（K5）;
  3. `emptied_targets` 预判（某 stamp 是否会被全拒 ⇒ 清仓误读红线，K3）;
  4. 与 §6-B 的 876 并列对照（**不作相等引用**，K1）。

🔴 只读: 不写生产数据、不改任何默认、不产裁决读数。
输出: reports/fuxi_v1_bj_universe_20261006/b_gate_domain.json
"""
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import datetime

ROOT = r'E:\quant'
OUT = os.path.join(ROOT, 'reports', 'fuxi_v1_bj_universe_20261006')
TARGETS = os.path.join(ROOT, 'reports', 'execution_ledger_20260923', 'targets')
DEST = os.path.join(OUT, 'b_gate_domain.json')
ENG_DIR = os.path.join(ROOT, 'research', 'factor_alchemy')
if ENG_DIR not in sys.path:
    sys.path.insert(0, ENG_DIR)
import entry_tradability as ET  # noqa: E402


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    print('=== 方案 B: gate() 自身判定域实测 ===', flush=True)
    print('time:', datetime.now().isoformat(), flush=True)
    cen = load_mod(os.path.join(ROOT, 'scripts', '_bj_uni_6b_census_20261006.py'), 'cen6b')
    days = cen.sessions()
    print('  交易日历 %d 天 (%s..%s)' % (len(days), days[0], days[-1]), flush=True)

    arms, wanted = {}, set()
    for fn in sorted(os.listdir(TARGETS)):
        tg = json.load(open(os.path.join(TARGETS, fn), encoding='utf-8'))['targets']
        pairs = []
        missing_stamp = []
        for stamp in sorted(tg):
            try:
                sess = ET.effective_session(stamp, days)
            except KeyError:
                missing_stamp.append(stamp)
                continue
            for code in sorted(tg[stamp]):
                pairs.append((stamp, sess, code))
                wanted.add((code, sess))
        arms[fn[:-5]] = {'pairs': pairs, 'stamps': len(tg),
                         'stamps_not_in_calendar': missing_stamp}
        print('  %-34s stamps=%-3d pairs=%-6d not-in-calendar=%d'
              % (fn[:-5], len(tg), len(pairs), len(missing_stamp)), flush=True)

    print('  装配门面板: %d 个 (code, session) 待判键 ...' % len(wanted), flush=True)
    panel, meta = cen.build_panel(wanted)
    print('  面板: %s' % json.dumps(meta, ensure_ascii=False), flush=True)

    assert panel.get('sessions') is None
    report = {
        'schema': 'fuxi_bj_gate_domain/v1',
        'generated_at': datetime.now().isoformat(),
        'purpose': '方案 B 的 gate() 自身判定域实测 (K1/K3/K5 前置)',
        'domain_note': ('session = effective_session(stamp, days) —— 目标 stamp 之后'
                        '第一个开市日; 与 §6-B 的「缺格自身日期」口径不同, 不得相等引用'),
        'window': [days[0], days[-1]],
        'panel_meta': meta,
        'arms': {},
    }
    for arm, blob in arms.items():
        cat = Counter()
        ver = Counter()
        by_board = Counter()
        by_board_unc = Counter()
        by_year_unc = Counter()
        per_stamp_unc = Counter()
        per_stamp_n = Counter()
        for stamp, sess, code in blob['pairs']:
            row = ET.certify(panel, code, sess)
            cat[row['category']] += 1
            ver[row['verdict']] += 1
            b = cen.board_of(code)
            by_board[b] += 1
            per_stamp_n[stamp] += 1
            if row['category'] == 'uncertifiable':
                by_board_unc[b] += 1
                by_year_unc[cen.year_of(sess)] += 1
                per_stamp_unc[stamp] += 1
        unc_n = cat.get('uncertifiable', 0)
        bj_unc = sum(v for k, v in by_board_unc.items() if k.startswith('BJ'))
        emptied = [{'stamp': s, 'n': per_stamp_n[s]}
                   for s in sorted(per_stamp_unc)
                   if per_stamp_unc[s] == per_stamp_n[s] and per_stamp_n[s] > 0]
        report['arms'][arm] = {
            'stamps': blob['stamps'],
            'stamps_not_in_calendar': blob['stamps_not_in_calendar'][:5],
            'pairs_checked': len(blob['pairs']),
            'by_category': dict(cat.most_common()),
            'by_verdict': dict(ver.most_common()),
            'uncertifiable_n': unc_n,
            'uncertifiable_bj_n': bj_unc,
            'uncertifiable_bj_share': round(bj_unc / unc_n, 4) if unc_n else None,
            'uncertifiable_by_board': dict(by_board_unc.most_common()),
            'uncertifiable_by_year': dict(sorted(by_year_unc.items())),
            'board_all': dict(by_board.most_common()),
            'emptied_targets_if_enforce': emptied,
            'refused_n_if_enforce': (cat.get('uncertifiable', 0) + cat.get('refuted', 0)
                                     + cat.get('conflicting', 0)),
            'refused_refuted_if_enforce': cat.get('refuted', 0),
            'refused_uncertifiable_if_enforce': unc_n,
        }
        print('  --- %s: pairs=%d uncertifiable=%d (BJ %d, share=%s) refuted=%d '
              'emptied_if_enforce=%d'
              % (arm, len(blob['pairs']), unc_n, bj_unc,
                 report['arms'][arm]['uncertifiable_bj_share'],
                 cat.get('refuted', 0), len(emptied)), flush=True)

    tot_unc = sum(a['uncertifiable_n'] for a in report['arms'].values())
    tot_pairs = sum(a['pairs_checked'] for a in report['arms'].values())
    report['totals'] = {
        'pairs_checked': tot_pairs,
        'uncertifiable_n': tot_unc,
        'uncertifiable_bj_n': sum(a['uncertifiable_bj_n'] for a in report['arms'].values()),
        'census_6b_reference': {
            'note': '§6-B 缺格面读数 (S5 重放持码格, 缺格自身日期口径) —— 并列对照, '
                    '不相等引用',
            'cell_face_uncertifiable': 876, 'cell_face_bj_share': 1.0,
            's5_window_years_of_uncertifiable': {'2021': 428, '2022': 444, '2023': 4}},
    }
    tmp = DEST + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
        fh.write('\n')
    os.replace(tmp, DEST)
    print('\n[TOTAL] pairs=%d uncertifiable=%d -> %s' % (tot_pairs, tot_unc, DEST), flush=True)


if __name__ == '__main__':
    main()
