# -*- coding: utf-8 -*-
"""_bj_uni_j4_replay_20261006.py — J4 负对照专项复跑（只读）

背景
----
首轮 J4 候选 (gp_breed_004/001/007) 在**前置门**即被拦截 (`cells=0 periods=0`,
`_last_backtest_diag` 未生成) ⇒ 无读数可比, 不构成有效负对照。选拔纪律修正:
J4 候选必须 ①在**当前**引擎下真出读数 (非早退), ②不含 .BJ 暴露。

候选族修正: 弃"价格水平"族 (`close.rolling(60).max().rank` 类 —— 会命中
P-025 价格水平代理拦截), 改用**标度无关**族 (收益率/区间/波动/相关性), 且全部取
`_s5_historical_pass_formulas.json` 中 `ex25/ex26` 非 null 者 (证其 S5 曾出读数)。

流程
----
  SCREEN: 全部候选在 legacy 臂各跑一次 → 记录 是否出读数 / reason / .BJ 暴露 / 台账
  PAIR  : 通过筛查者 (出读数 ∧ .BJ 暴露=0) 在 listed_only 臂各跑一次 → 逐位对拍
  两臂各用**独立 S5 实例** (避免跨臂状态污染), 两实例共享规范数据根。

输出: reports/fuxi_v1_bj_universe_20261006/j4_readings.json
"""
import hashlib
import importlib.util
import io
import json
import os
import sys
import time
from datetime import datetime

ROOT = r'E:\quant'
OUT = os.path.join(ROOT, 'reports', 'fuxi_v1_bj_universe_20261006')
READINGS = os.path.join(OUT, 'j4_readings.json')

spec = importlib.util.spec_from_file_location(
    'rp_j4', os.path.join(ROOT, 'scripts', '_bj_uni_j17_replay_20261006.py'))
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

CANDIDATES = [
    dict(name='gp_breed_056', tag='量价相关',
         formula='volume.rolling(30).corr(close.pct_change(20))', freq='daily'),
    dict(name='v3_overreaction_reversal', tag='收益极值反转',
         formula='-close_p.pct_change(1).rolling(20).max()', freq='daily'),
    dict(name='gp_evo_000', tag='区间/均值收益',
         formula='(-((high - low) / (returns.rolling(20).mean() + 1)).rank(pct=True))',
         freq='daily'),
    dict(name='gp_evo_003', tag='区间/协方差',
         formula='(-((high - low) / (amplitude.rolling(10).mean().rolling(100).cov(close_p)'
                 '.rolling(20).mean() + 1)).rank(pct=True))', freq='daily'),
    dict(name='gp_evo_002', tag='高低比/量',
         formula='(-((high.rolling(20).mean() / low.rolling(20).mean()) / '
                 '(volume.rolling(20).mean() + hl_ratio)).rank(pct=True))', freq='daily'),
    dict(name='gp_breed_047', tag='振幅/换手/EMA价 取小',
         formula='(-np.maximum(np.minimum(amplitude, turnover), '
                 'close_p.ewm(span=10).mean()).rank(pct=True))', freq='daily'),
]


def run_arm(arm_id, env, cands, mod_cache):
    if arm_id not in mod_cache:
        cells, members = [], []
        mod_cache[arm_id] = (rp.build_module(rp.ENG_CUR, cells, members), cells, members)
    mod, cells, members = mod_cache[arm_id]
    for k in ('FUXI_S5_HOLDINGS_MARK', 'FUXI_S5_BJ_UNIVERSE'):
        os.environ.pop(k, None)
    os.environ.update(env)
    if 's5' not in mod_cache[arm_id]:
        t0 = time.time()
        s5 = mod.S5JointFilter(top_n=80)
        s5._load_data()
        print('[%s] loaded universe=%d | %.0fs'
              % (arm_id, len(s5._universe), time.time() - t0), flush=True)
        mod_cache[arm_id] = (mod, cells, members, s5)
    s5 = mod_cache[arm_id][3]

    out = {}
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
        held = sorted({x for m in members for x in m['codes']})
        rec = {'candidate': c['name'], 'tag': c['tag'], 'formula': c['formula'],
               'elapsed_s': round(time.time() - t1, 1), 'error': err,
               'readings': rd, 'marking_diagnostics': mdl,
               'ledger': {k: diag.get(k) for k in (
                   'missing_held_price_cells', 'incomplete_marking_days',
                   'missing_held_price_cells_by_year', 'coverage_rate',
                   'n_periods_possible', 'n_periods_used', 'n_periods_skipped',
                   'switches', 'n_adj_outlier_clipped', 'n_blocked_limit_up',
                   'carried_forward_cells', 'carry_forward_exceeded_cells')},
               'cells': sorted({(x['date'], x['ts_code']) for x in cells}),
               'members_n': len(members),
               'held_codes_n': len(held),
               'held_bj': [x for x in held if x.endswith('.BJ')],
               'ran': bool(diag) or bool(rd),
               'reason_head': (rd.get('reason') or '')[:140]}
        rec['cells'] = [{'date': d, 'ts_code': t} for d, t in rec['cells']]
        out[c['name']] = rec
        print('  --- %-26s %5.0fs | ran=%-5s missing=%-6s periods=%-4d held=%d bj=%d | %s'
              % (c['name'], rec['elapsed_s'], rec['ran'],
                 rec['ledger']['missing_held_price_cells'], rec['members_n'],
                 rec['held_codes_n'], len(rec['held_bj']),
                 rec['reason_head'][:50] or rec['error'][:40] or '-'), flush=True)
        del r
    return out


def main():
    print('=== J4 负对照专项复跑 ===', flush=True)
    print('time:', datetime.now().isoformat(), flush=True)
    report = {'schema': 'fuxi_bj_universe_j4/v1',
              'generated_at': datetime.now().isoformat(),
              'engine': {'path': rp.ENG_CUR,
                         'sha256': hashlib.sha256(open(rp.ENG_CUR, 'rb').read()).hexdigest()},
              'screen_policy': '当前引擎下真出读数 ∧ 零 .BJ 暴露',
              'candidates': [c['name'] for c in CANDIDATES],
              'arms': {}}

    cache = {}
    # ── SCREEN: legacy ──
    os.environ.pop('FUXI_S5_HOLDINGS_MARK', None)
    report['arms']['legacy'] = run_arm(
        'legacy', {'FUXI_S5_HOLDINGS_MARK': 'enforce', 'FUXI_S5_BJ_UNIVERSE': 'legacy'},
        CANDIDATES, cache)
    rp.dump_atomic(report, READINGS)

    # ── 筛查（加严: 必须真跑出回测/台账, 而非仅"未早退"） ──
    def _valid(n):
        r = report['arms']['legacy'][n]
        return (r['ran'] and r['members_n'] > 0
                and r['ledger']['missing_held_price_cells'] is not None
                and not r['held_bj'])
    valid = [c for c in CANDIDATES if _valid(c['name'])]
    print('\n[screen] 合格负对照候选: %s' % [c['name'] for c in valid], flush=True)
    report['valid_controls'] = [c['name'] for c in valid]

    # ── PAIR: listed_only（只对**屏幕合格者**；screen=0 时显式跳过） ──
    # 🔴 禁用 `valid or CANDIDATES` 形式的**静默回退**：screen=0 时它会退化为
    #    "对全部候选跑 PAIR"，与屏幕结论矛盾（j4c 批次 2026-10-07 实测触发）。
    #    若确实想对**含 BJ 暴露**的候选做效应测量，须走 `DIAG_PAIR_DESIGNATED`
    #    （显式声明的诊断模式，不是回退）。
    DIAG_PAIR_DESIGNATED = []
    _pair = valid if valid else DIAG_PAIR_DESIGNATED
    if not _pair:
        report['arms']['listed_only'] = {}
        report['comparison'] = {}
        report['J4_verdict'] = {
            'n_qualifying': 0, 'qualifying': [], 'pass': False,
            'note': 'screen=0 ⇒ PAIR explicitly SKIPPED (no silent fallback)'}
        rp.dump_atomic(report, READINGS)
        print('\n[pair] SKIPPED explicitly: screen returned 0 candidates -> %s'
              % READINGS, flush=True)
        return
    report['arms']['listed_only'] = run_arm(
        'listed_only',
        {'FUXI_S5_HOLDINGS_MARK': 'enforce', 'FUXI_S5_BJ_UNIVERSE': 'listed_only'},
        _pair, cache)
    rp.dump_atomic(report, READINGS)

    # ── 对拍 ──
    cmp_out = {}
    for c in _pair:
        n = c['name']
        a = report['arms']['legacy'][n]
        b = report['arms']['listed_only'][n]
        ra, rb = rp.stable(a['readings']), rp.stable(b['readings'])
        diffs = sorted(k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k))
        sa = {(x['date'], x['ts_code']) for x in a['cells']}
        sb = {(x['date'], x['ts_code']) for x in b['cells']}
        cmp_out[n] = {
            'ran_both': a['ran'] and b['ran'],
            'bj_exposure_free': (not a['held_bj']) and (not b['held_bj']),
            'readings_identical': len(diffs) == 0, 'diff_keys': diffs,
            'diff_values': {k: [ra.get(k), rb.get(k)] for k in diffs[:12]},
            'cells_identical': sa == sb,
            'cells_legacy': len(sa), 'cells_listed': len(sb),
            'missing_legacy': a['ledger']['missing_held_price_cells'],
            'missing_listed': b['ledger']['missing_held_price_cells'],
            'periods_legacy': a['members_n'], 'periods_listed': b['members_n'],
            'held_codes_n': a['held_codes_n'],
        }
    report['comparison'] = cmp_out
    ok = [k for k, v in cmp_out.items()
          if v['readings_identical'] and v['bj_exposure_free'] and v['ran_both']
          and v['periods_legacy'] > 0]
    report['J4_verdict'] = {'n_qualifying': len(ok), 'qualifying': ok,
                            'pass': len(ok) >= 3}
    rp.dump_atomic(report, READINGS)

    print('\n--- J4 汇总 ---', flush=True)
    for k, v in cmp_out.items():
        print('  %-26s ran=%-5s bj_free=%-5s readings_same=%-5s cells_same=%-5s diffs=%d'
              % (k, v['ran_both'], v['bj_exposure_free'], v['readings_identical'],
                 v['cells_identical'], len(v['diff_keys'])), flush=True)
    print('\n[J4] qualifying=%d pass=%s -> %s'
          % (len(ok), report['J4_verdict']['pass'], READINGS), flush=True)


if __name__ == '__main__':
    main()
