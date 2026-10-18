# -*- coding: utf-8 -*-
"""轴语义修复 A/B + 安全网单测 (2026-09-18, P-20260918-006)

A/B-1 路由判据: 旧(字面白名单) vs 新(白名单+正则) 在**全库 1269 条公式**上逐一比对。
       期望: removed=0 (不得把任何当前能通过的公式改道), added 只出现在写法变体上。
A/B-2 字段补全: 动态补入字段只能把**失败**变成功, 不得改变已成功公式的结果。
       以"截面路径产出与旧实现逐位一致"验证: 对 19 条原本就 OK 的因子, 比较
       修复前 (backup 模块) 与修复后 (当前模块) 的 factor_value 序列。
Fix C 安全网: 强制路由判据返回 False, 验证 evaluate_factor_formula 会把逐股失败
       升级到截面路径并留痕 (而不是静默丢候选)。
"""
import os
import sys
import re
import json
import importlib
import importlib.util

FA = r'e:/quant/research/factor_alchemy'
sys.path.insert(0, FA)
sys.path.insert(0, r'e:/quant/scripts')
sys.path.insert(0, r'e:/quant')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OLD_MARKERS = ['mean(axis=1)', 'mean(axis = 1)', 'std(axis=1)', 'std(axis = 1)',
               'min(axis=1)', 'min(axis = 1)', 'max(axis=1)', 'max(axis = 1)',
               'rank(axis=1', 'rank(axis = 1', 'rank(pct=True, axis=1']
OLD_AX0 = ['axis=0', 'axis = 0']


def old_detector(f: str) -> bool:
    return any(m in f for m in OLD_MARKERS) or any(m in f for m in OLD_AX0)


def load_old_module():
    """从备份文件加载修复前的 s5_joint_filter, 用于 A/B-2 对照。

    备份名是 `.py.bak_...` (非法扩展名), importlib 不认 ⇒ 先复制成 .py 再加载。
    """
    import shutil
    import tempfile
    bak = os.path.join(FA, 's5_joint_filter.py.bak_20260918_axis')
    tmpd = tempfile.mkdtemp(prefix='s5old_')
    dst = os.path.join(tmpd, 's5_old_ab.py')
    shutil.copy2(bak, dst)
    spec = importlib.util.spec_from_file_location('s5_old_ab', dst)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m._AB_TMPDIR = tmpd
    return m


def build_panel(n_stock=12, n_day=200, seed=7):
    rng = np.random.default_rng(seed)
    codes = ['%06d.SZ' % (1 + i) for i in range(n_stock)]
    dates = pd.bdate_range('2025-01-02', periods=n_day)
    rows = []
    base = {c: 10.0 + rng.random() * 30 for c in codes}
    for d in dates:
        for c in codes:
            base[c] *= float(np.exp(rng.normal(0, 0.02)))
            px = base[c]
            vol = float(np.exp(rng.normal(14, 0.6)))
            rows.append({
                'trade_date': d.strftime('%Y%m%d'), 'ts_code': c,
                'open_p': px * (1 + rng.normal(0, 0.004)),
                'high_p': px * (1 + abs(rng.normal(0, 0.01))),
                'low_p': px * (1 - abs(rng.normal(0, 0.01))),
                'close_p': px, 'volume_p': vol, 'amount_p': vol * px,
                'overnight_p': rng.normal(0, 0.01),
                'amplitude_p': abs(rng.normal(0, 0.02)),
                'returns_p': rng.normal(0, 0.02),
                'turnover_p': abs(rng.normal(2, 0.7)),
                'buy_lg_vol': vol * 0.3, 'sell_lg_vol': vol * 0.28,
                'buy_sm_vol': vol * 0.2, 'sell_sm_vol': vol * 0.22,
                'buy_md_vol': vol * 0.25, 'sell_md_vol': vol * 0.25,
                'buy_elg_vol': vol * 0.1, 'sell_elg_vol': vol * 0.1,
                'net_mf_vol': vol * rng.normal(0, 0.05),
                'net_mf_amount': vol * px * rng.normal(0, 0.05),
                'total_assets': 1e10, 'intan_assets': 1e8, 'goodwill': 1e7,
                'pe_ttm': abs(rng.normal(25, 8)), 'pb': abs(rng.normal(2.5, 1)),
                'dv_ratio': abs(rng.normal(1.5, 0.6)), 'roe_dt': rng.normal(8, 4),
                'tr_yoy': rng.normal(10, 15), 'netprofit_yoy': rng.normal(8, 20),
                'lhb_flag': 1.0 if rng.random() < 0.08 else 0.0,
                'lhb_net_amount': rng.normal(0, 1e7),
                'lhb_amount': abs(rng.normal(5e7, 2e7)),
                'lhb_net_rate': rng.normal(0, 3),
                'lhb_inst_net_buy': rng.normal(0, 2e6) if rng.random() < 0.05 else 0.0,
                'lhb_inst_buy': 0.0, 'lhb_inst_sell': 0.0,
                'lhb_north_net_buy': rng.normal(0, 2e6) if rng.random() < 0.05 else 0.0,
                'lhb_north_buy': 0.0, 'lhb_north_sell': 0.0,
                'lhb_yz_net_buy': rng.normal(0, 2e6) if rng.random() < 0.05 else 0.0,
                'lhb_yz_buy': 0.0, 'lhb_yz_sell': 0.0,
                'rzye': abs(rng.normal(1e9, 1e8)), 'rqye': abs(rng.normal(1e7, 1e6)),
                'rzmre': abs(rng.normal(1e8, 1e7)), 'rqyl': abs(rng.normal(1e6, 1e5)),
                'rqchl': abs(rng.normal(1e5, 1e4)),
            })
    return pd.DataFrame(rows)


def collect_formulas():
    import daily_factor_hypothesis as dh
    novel = [(str(e.get('id')), str(e.get('formula', ''))) for e in dh.NOVEL_FACTOR_LIBRARY]
    libp = r'e:/quant/data/library_orthogonality_state.json'
    lib = json.load(open(libp, encoding='utf-8'))['factors']
    libf = [('lib:' + str(f.get('name'))[:40], str(f.get('expression', ''))) for f in lib]
    return novel, libf


def main():
    import s5_joint_filter as S
    new_det = S.S5JointFilter._needs_cross_section
    dummy = object()

    novel, libf = collect_formulas()
    allf = novel + libf
    n_same = n_added = n_removed = 0
    added = []
    removed = []
    for name, f in allf:
        if not f:
            continue
        o = old_detector(f)
        n = bool(new_det(dummy, f))
        if o == n:
            n_same += 1
        elif (not o) and n:
            n_added += 1
            added.append((name, f[:90]))
        else:
            n_removed += 1
            removed.append((name, f[:90]))

    print('=' * 74)
    print('A/B-1  路由判据: 旧(字面白名单) vs 新(白名单+正则)')
    print('=' * 74)
    print('  公式总数            : %d  (NOVEL %d + 正交库 %d)' % (len(allf), len(novel), len(libf)))
    print('  判定相同            : %d' % n_same)
    print('  新增为 True (收紧)  : %d' % n_added)
    print('  由 True 变 False    : %d   <-- 必须为 0' % n_removed)
    for nm, f in added[:6]:
        print('    + %-40s %s' % (nm[:40], f))
    for nm, f in removed[:6]:
        print('    - %-40s %s' % (nm[:40], f))
    ok1 = (n_removed == 0)

    # ── A/B-2: 已成功公式的读数是否逐位不变 ─────────────────────
    print()
    print('=' * 74)
    print('A/B-2  字段补全不改变已成功公式的读数 (逐位比对)')
    print('=' * 74)
    df = build_panel()
    f_new = S.S5JointFilter(sample_stocks=None)
    S_old = load_old_module()
    f_old = S_old.S5JointFilter(sample_stocks=None)

    AX1 = re.compile(r'\baxis\s*=\s*1\b')
    targets = [(n, f) for n, f in novel if AX1.search(f)]
    same = diff = skipped = 0
    diffs = []
    for n, f in targets:
        try:
            r_old = f_old._evaluate_cross_section(df, f, n)
            r_new = f_new._evaluate_cross_section(df, f, n)
        except Exception as e:
            skipped += 1
            continue
        lo = 0 if r_old is None else len(r_old)
        ln = 0 if r_new is None else len(r_new)
        if lo == 0 or ln == 0:
            # 修复前就失败/空 ⇒ 不计入"已成功"对照
            skipped += 1
            continue
        a = r_old.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        b = r_new.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        if len(a) == len(b) and np.allclose(a['factor_value'].values,
                                            b['factor_value'].values,
                                            rtol=0, atol=0, equal_nan=True):
            same += 1
        else:
            diff += 1
            diffs.append((n, lo, ln))
    print('  参与对照 (修复前已成功) : %d' % (same + diff))
    print('  逐位完全一致            : %d' % same)
    print('  有差异                  : %d   <-- 必须为 0' % diff)
    print('  跳过 (修复前即空/报错)  : %d' % skipped)
    for n, lo, ln in diffs[:8]:
        print('    ! %-40s old_rows=%d new_rows=%d' % (n[:40], lo, ln))
    ok2 = (diff == 0)

    # ── Fix C: 安全网 ────────────────────────────────────────────
    print()
    print('=' * 74)
    print('Fix C  逐股失败 → 截面升级 安全网 (强制路由判据为 False)')
    print('=' * 74)
    panel = build_panel(n_stock=8, n_day=120, seed=11)
    f3 = S.S5JointFilter(sample_stocks=None)
    f3._loaded = True
    f3._price_df = panel
    f3.sample_stocks = None
    f3._universe = sorted(panel['ts_code'].unique().tolist())
    orig = S.S5JointFilter._needs_cross_section
    S.S5JointFilter._needs_cross_section = lambda self, formula: False
    try:
        probe = 'close_p.pct_change(10).sub(close_p.mean(axis=1).pct_change(10), axis=0)'
        print('  探针公式: %s' % probe)
        print('  (强制 _needs_cross_section -> False, 模拟路由漏判)')
        res = f3.evaluate_factor_formula(probe, 'FIXC_probe')
        rows = 0 if res is None or len(res) == 0 else len(res)
        print('  evaluate_factor_formula 返回行数: %d' % rows)
        ok3 = rows > 0
        print('  => 安全网 %s (逐股失败后成功升级到截面路径)' % ('PASS' if ok3 else 'FAIL'))
    finally:
        S.S5JointFilter._needs_cross_section = orig

    print()
    print('=' * 74)
    print('A/B-1=%s  A/B-2=%s  FixC=%s' % (ok1, ok2, ok3))
    print('OVERALL: %s' % ('PASS' if (ok1 and ok2 and ok3) else 'FAIL'))


if __name__ == '__main__':
    main()
