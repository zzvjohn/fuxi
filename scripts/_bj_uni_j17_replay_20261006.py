# -*- coding: utf-8 -*-
"""_bj_uni_j17_replay_20261006.py — BJ universe 方案 A · J1–J7 冻结重放（只读）

判据来源: reports/fuxi_v1_bj_universe_20261006/PRE_REGISTRATION_bj_universe.md §3
冻结重放面: M17 的 3 候选（stage2_gap_ledger.json 同源）+ J4 负对照候选。

五臂设计
--------
  A0_bak_enforce      : 变更前引擎(备份, sha e244b2b8) + HOLDINGS_MARK=enforce   → J6 基线
  A1_legacy_enforce   : patched 引擎 + BJ_UNIVERSE=legacy      + enforce        → J6 复现
  A2_listed_enforce   : patched 引擎 + BJ_UNIVERSE=listed_only + enforce        → J1/J2/J5
  B1_legacy_shadow    : patched 引擎 + legacy      + shadow_carry_forward       → 现行默认披露
  B2_listed_shadow    : patched 引擎 + listed_only + shadow_carry_forward       → 现行默认披露

工程纪律
--------
- 生产文件零改动: 插桩仅在内存 (字符串替换 → exec 临时模块); 唯一改动 =
  marking_samples 12 条截断 + 全量缺失格采集 + 逐期成员采集（纯诊断, 不含逻辑变更）。
- 自证: enforce 臂的 missing/days 须与 M17 记录一致, 否则 FAIL（引擎已演化时记为 drift 并披露）。
- 只写 reports/fuxi_v1_bj_universe_20261006/ 下的产出。
"""
import hashlib
import io
import json
import os
import sys
import time
import types
from datetime import datetime

ROOT = r'E:\quant'
FA = os.path.join(ROOT, 'research', 'factor_alchemy')
sys.path.insert(0, FA)

import numpy as np
import pandas as pd

from pathlib import Path

RAW = os.path.join(ROOT, 'data', 'raw')
DATA_ROOT = Path(ROOT) / 'data'
RAW_DIR_CANON = DATA_ROOT / 'raw'
OUT = os.path.join(ROOT, 'reports', 'fuxi_v1_bj_universe_20261006')

ENG_CUR = os.path.join(FA, 's5_joint_filter.py')
ENG_BAK = os.path.join(OUT, 'pre_bjcut', 's5_joint_filter.py.bak')

# ── 冻结重放面: M17 的 3 候选 (exp = M17 stage2_gap_ledger.json 记录) ──
M17_CANDS = [
    dict(name='w_flow_price_divergence',
         formula='(close.pct_change(4) - volume.pct_change(4)).rolling(4).mean()',
         freq='weekly', exp_missing=318, exp_days=244),
    dict(name='w_flow_trend_orthogonal',
         formula='(close.pct_change(4).rank(pct=True, axis=1) * (1 - (volume.pct_change(4).rank(pct=True, axis=1))))',
         freq='weekly', exp_missing=730, exp_days=407),
    dict(name='weekly_ret_skew_8w_rev',
         formula='-close.pct_change().rolling(8).skew()',
         freq='weekly', exp_missing=136, exp_days=110),
]

# ── J4 负对照候选 (历史通过; 选"有 S5 读数(ex25/ex26 非 null)"且族上 BJ 暴露先验低者) ──
J4_CANDS = [
    dict(name='gp_breed_004', formula='close.rolling(60).max().rank(pct=True)', freq='daily'),
    dict(name='gp_breed_001', formula='close.rolling(60).min().rank(pct=True)', freq='daily'),
    dict(name='gp_breed_007',
         formula='(high.rolling(20).mean() / hl_ratio.rolling(20).mean()).rank(pct=True)',
         freq='daily'),
    dict(name='gp_evo_000',
         formula='(-((high - low) / (returns.rolling(20).mean() + 1)).rank(pct=True))',
         freq='daily'),
    dict(name='gp_breed_005',
         formula='(((close_p.shift(80) - amount_p.rolling(120).corr(close_p)) / open_p)'
                 '.rolling(20).mean() / low.rolling(20).mean()).rank(pct=True)',
         freq='daily'),
]

ARMS = [
    dict(id='A0_bak_enforce', eng='bak', cands='m17',
         env={'FUXI_S5_HOLDINGS_MARK': 'enforce'}),
    dict(id='A1_legacy_enforce', eng='cur', cands='all',
         env={'FUXI_S5_HOLDINGS_MARK': 'enforce', 'FUXI_S5_BJ_UNIVERSE': 'legacy'}),
    dict(id='A2_listed_enforce', eng='cur', cands='all',
         env={'FUXI_S5_HOLDINGS_MARK': 'enforce', 'FUXI_S5_BJ_UNIVERSE': 'listed_only'}),
    dict(id='B1_legacy_shadow', eng='cur', cands='m17',
         env={'FUXI_S5_HOLDINGS_MARK': 'shadow_carry_forward',
              'FUXI_S5_BJ_UNIVERSE': 'legacy'}),
    dict(id='B2_listed_shadow', eng='cur', cands='m17',
         env={'FUXI_S5_HOLDINGS_MARK': 'shadow_carry_forward',
              'FUXI_S5_BJ_UNIVERSE': 'listed_only'}),
]

# ── 插桩 A: 缺失格全量采集 (M17 同款) ──
ANCHOR_CELLS = """                for day in missing[missing > 0].index:
                    if len(_mark_diag['marking_samples']) < 12:
                        _mark_diag['marking_samples'].append({'date': str(day.date()), 'missing': int(missing.loc[day])})"""
PATCH_CELLS = """                for day in missing[missing > 0].index:
                    if len(_mark_diag['marking_samples']) < 12:
                        _mark_diag['marking_samples'].append({'date': str(day.date()), 'missing': int(missing.loc[day])})
                    _s2row = held.loc[day]
                    for _s2c in _s2row.index[_s2row.isna()]:
                        _MISS_CELLS.append({'date': str(day.date()), 'ts_code': str(_s2c)})"""

# ── 插桩 B: 逐期成员采集 (J2/J5 的"成员关系"读数面) ──
ANCHOR_MEMBERS = """                held = held.where(np.isfinite(held))
                missing = held.isna().sum(axis=1)"""
PATCH_MEMBERS = """                held = held.where(np.isfinite(held))
                missing = held.isna().sum(axis=1)
                _MEMBERS.append({'entry': str(entry_date.date()),
                                 'exit': str(exit_date.date()),
                                 'codes': sorted(str(_c) for _c in selected_codes)})"""

VOLATILE_KEYS = {'engine_switches', 'computation_time', 'generated_at', 'elapsed_s'}

READINGS_PATH = os.path.join(OUT, 'j17_readings.json')
RESUME = os.environ.get('BJ_J17_RESUME') == '1'


def _jdefault(o):
    """numpy 标量/数组 → 原生类型 (读数里夹带 float32 会让 json.dump 抛错)。"""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def dump_atomic(payload, path):
    tmp = path + '.tmp'
    with io.open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, default=_jdefault)
    os.replace(tmp, path)


def stable(d):
    """剔除易变字段后的可复现读数 (J5/J6 对账口径)。"""
    return {k: v for k, v in (d or {}).items() if k not in VOLATILE_KEYS}


def build_module(engine_path, cells, members):
    """构建插桩模块。

    🔴 路径归位: 引擎以 __file__ 推导 DATA_DIR/RAW_DIR/scripts 目录
    (DATA_DIR = __file__.parent.parent.parent/'data')。备份副本身处 reports 深层
    目录 ⇒ 会解析到 E:\\quant\\reports\\data (不存在)。为让 A/B 读**同一份数据**,
    两个引擎的 __file__ 统一归位到规范引擎路径; 归位后显式断言数据根。
    """
    src = io.open(engine_path, 'r', encoding='utf-8', newline='').read()
    patched = src
    for tag, a, p in (('cells', ANCHOR_CELLS, PATCH_CELLS),
                      ('members', ANCHOR_MEMBERS, PATCH_MEMBERS)):
        a_c, p_c = a.replace('\n', '\r\n'), p.replace('\n', '\r\n')
        assert '\r' not in p_c.replace('\r\n', ''), f'{tag}: bare CR in patch'
        n = patched.count(a_c)
        if n != 1:
            raise RuntimeError(f'{tag}: anchor count = {n} (expect 1)')
        patched = patched.replace(a_c, p_c)
    assert '\r' not in patched.replace('\r\n', ''), 'patched source has bare CR'
    assert patched.count('\n') == patched.count('\r\n'), 'patched source not pure CRLF'
    mod = types.ModuleType('s5jf_j17_' + hashlib.sha256(
        engine_path.encode()).hexdigest()[:6])
    mod.__file__ = ENG_CUR                      # 路径归位 (两引擎共享同一数据根)
    mod.__package__ = ''
    mod.__dict__['_MISS_CELLS'] = cells
    mod.__dict__['_MEMBERS'] = members
    exec(compile(patched, engine_path, 'exec'), mod.__dict__)
    # belt-and-braces: 显式归位 + 断言 (exec 后 __file__ 推导可能被缓存常量覆盖)
    mod.__dict__['DATA_DIR'] = DATA_ROOT
    mod.__dict__['RAW_DIR'] = RAW_DIR_CANON
    assert mod.__dict__['RAW_DIR'] == RAW_DIR_CANON, 'RAW_DIR relocation failed'
    assert mod.__dict__['DATA_DIR'] == DATA_ROOT, 'DATA_DIR relocation failed'
    return mod


def run_arm(arm):
    env_keys = ('FUXI_S5_HOLDINGS_MARK', 'FUXI_S5_BJ_UNIVERSE')
    for k in env_keys:
        os.environ.pop(k, None)
    os.environ.update(arm['env'])

    eng = ENG_BAK if arm['eng'] == 'bak' else ENG_CUR
    eng_sha = hashlib.sha256(open(eng, 'rb').read()).hexdigest()
    cands = M17_CANDS if arm['cands'] == 'm17' else (M17_CANDS + J4_CANDS)

    print('\n' + '=' * 78, flush=True)
    print('[arm] %s | engine=%s sha=%s' % (arm['id'], os.path.basename(eng), eng_sha[:16]), flush=True)
    print('      env=%s | cands=%d' % (arm['env'], len(cands)), flush=True)

    cells, members = [], []
    t0 = time.time()
    mod = build_module(eng, cells, members)
    s5 = mod.S5JointFilter(top_n=80)
    s5._load_data()
    n_univ = len(s5._universe)
    n_bj = sum(1 for c in s5._universe if str(c).endswith('.BJ'))
    print('      loaded universe=%d (.BJ=%d) | %.0fs' % (n_univ, n_bj, time.time() - t0), flush=True)

    arm_rec = {'arm': arm['id'], 'engine': os.path.basename(eng), 'engine_sha256': eng_sha,
               'env': dict(arm['env']), 'n_universe': n_univ, 'n_universe_bj': n_bj,
               'runs': {}}

    for c in cands:
        cells.clear()
        members.clear()
        t1 = time.time()
        try:
            r = s5.validate_factor(c['name'], c['formula'], natural_freq=c['freq'])
            rd = r.to_dict()
            err = str(getattr(r, 'error', '') or '')
            mdl = r.marking_diagnostics
        except Exception as e:  # noqa: BLE001
            rd, err, mdl = {}, 'EXCEPTION: %r' % e, {}
        diag = s5._last_backtest_diag or {}
        uniq = sorted({(x['date'], x['ts_code']) for x in cells})
        rec = {
            'candidate': c['name'], 'formula': c['formula'], 'freq': c['freq'],
            'elapsed_s': round(time.time() - t1, 1),
            'error': err,
            'readings': rd,
            'marking_diagnostics': mdl,
            'ledger': {
                'missing_held_price_cells': diag.get('missing_held_price_cells'),
                'incomplete_marking_days': diag.get('incomplete_marking_days'),
                'missing_by_year': diag.get('missing_held_price_cells_by_year'),
                'carried_forward_cells': diag.get('carried_forward_cells'),
                'carry_forward_exceeded_cells': diag.get('carry_forward_exceeded_cells'),
                'coverage_rate': diag.get('coverage_rate'),
                'n_periods_possible': diag.get('n_periods_possible'),
                'n_periods_used': diag.get('n_periods_used'),
                'n_periods_skipped': diag.get('n_periods_skipped'),
                'accounting_ok': diag.get('accounting_ok'),
                'switches': diag.get('switches'),
                'adj_outlier_clipped': diag.get('n_adj_outlier_clipped'),
                'n_blocked_limit_up': diag.get('n_blocked_limit_up'),
            },
            'cells': [{'date': d, 'ts_code': t} for d, t in uniq],
            'cells_n': len(uniq),
            # 🔴 必须深拷贝: `members` 是跨轮复用的 list, 下一轮 clear() 会清空已存历史
            'members': [dict(m) for m in members],
            'members_n_periods': len(members),
            'exp_missing': c.get('exp_missing'), 'exp_days': c.get('exp_days'),
            'pair_match_m17': (c.get('exp_missing') is not None and
                               diag.get('missing_held_price_cells') == c.get('exp_missing') and
                               diag.get('incomplete_marking_days') == c.get('exp_days')),
        }
        arm_rec['runs'][c['name']] = rec
        print('  --- %-28s %5.0fs | missing=%s days=%s cells=%d periods=%d | err=%s'
              % (c['name'], rec['elapsed_s'], diag.get('missing_held_price_cells'),
                 diag.get('incomplete_marking_days'), len(uniq), len(members),
                 err[:60] or '-'), flush=True)
        del r

    del s5
    return arm_rec


def main():
    os.makedirs(OUT, exist_ok=True)
    print('=== BJ universe 方案 A · J1-J7 冻结重放 ===', flush=True)
    print('time:', datetime.now().isoformat(), flush=True)

    report = {'schema': 'fuxi_bj_universe_j17/v1',
              'generated_at': datetime.now().isoformat(),
              'pre_registration': 'PRE_REGISTRATION_bj_universe.md',
              'frozen_replay': 'M17 stage2_gap_ledger.json 同源 (3 候选)',
              'instruments': 'marking_samples 12-cap kept; full missing-cell ledger; per-period membership',
              'engines': {'cur': {'path': ENG_CUR,
                                  'sha256': hashlib.sha256(open(ENG_CUR, 'rb').read()).hexdigest()},
                          'bak': {'path': ENG_BAK,
                                  'sha256': hashlib.sha256(open(ENG_BAK, 'rb').read()).hexdigest()}},
              'arms': {}}

    if RESUME and os.path.exists(READINGS_PATH):
        try:
            prev = json.load(io.open(READINGS_PATH, encoding='utf-8'))
            report['arms'] = prev.get('arms', {}) or {}
            print('[resume] carried %d arm(s): %s'
                  % (len(report['arms']), ', '.join(report['arms'])), flush=True)
        except Exception as e:  # noqa: BLE001
            print('[resume] ignored unreadable previous readings (%r)' % e, flush=True)

    for arm in ARMS:
        eng_path = ENG_BAK if arm['eng'] == 'bak' else ENG_CUR
        want_sha = hashlib.sha256(open(eng_path, 'rb').read()).hexdigest()
        prev = report['arms'].get(arm['id'])
        if prev and prev.get('engine_sha256') == want_sha:
            print('\n[arm] %s | SKIP (resume: sha %s already recorded)'
                  % (arm['id'], want_sha[:16]), flush=True)
            continue
        rec = run_arm(arm)
        report['arms'][rec['arm']] = rec
        dump_atomic(report, READINGS_PATH)
        print('  [dump] %s written' % READINGS_PATH, flush=True)

    print('\nall arms done. ->', READINGS_PATH, flush=True)


if __name__ == '__main__':
    main()
