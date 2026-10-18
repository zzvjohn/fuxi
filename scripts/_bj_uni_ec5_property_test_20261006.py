# -*- coding: utf-8 -*-
"""_bj_uni_ec5_property_test_20261006.py — J4 替代取证: 机制级性质测试 (只读)

背景
----
J4 预注册要求「≥3 个**不含 .BJ 暴露**的已通过候选，在变更前后读数逐位不变」。
实测命中率 **1/23 (≈4.3%)** —— 批次1 1/6、批次2 0/6、批次3 0/11（全部含 BJ 暴露）。
结构性原因: S5 池 BJ 占比 **6.10%** (353/5784) × `top_n=80` × ~286 期 ≈ **2.3 万槽位**;
零暴露要求因子在**每一期**都把全部 353 只 BJ 排在 80 名之外 —— 非一般因子可达。

故以**机制级性质测试**补强 J4 的意图（「变更不外溢到非 BJ 路径」）:

  P1 非 .BJ 行逐位恒等 (值 / 行序 / 索引 / dtype)
  P2 目标类必删 (.BJ ∧ trade_date < list_date)
  P3 上市后必留 (.BJ ∧ trade_date >= list_date)
  P4 fail-safe: 无 list_date 的 .BJ 不裁
  P5 幂等: cut(cut(x)) ≡ cut(x)
  P6 边界: 空帧 / 无 ts_code 列 / 全 BJ 无映射
  P7 纯函数: 不改输入 (输入帧内容摘要前后一致)
  P8 fail-closed: FUXI_S5_LEDGER=fixed_shares ∧ listed_only ⇒ ValueError
  P9 getter 值域实测 (默认 legacy / listed_only / 非法值 fail-fast)
  P10 上市日映射覆盖 (353 只, 全 .BJ, 时间戳合法)

🔴 本脚本**只读**: 不写生产数据、不改任何默认、不产经济裁决读数。
输出: reports/fuxi_v1_bj_universe_20261006/ec5_property_test.json
"""
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = r'E:\quant'
ENG = os.path.join(ROOT, 'research', 'factor_alchemy', 's5_joint_filter.py')
OUT = os.path.join(ROOT, 'reports', 'fuxi_v1_bj_universe_20261006')
DEST = os.path.join(OUT, 'ec5_property_test.json')

RESULTS = []
ENV_KEYS = ('FUXI_S5_BJ_UNIVERSE', 'FUXI_S5_LEDGER', 'FUXI_S5_HOLDINGS_MARK')


def rec(pid, name, ok, evidence):
    RESULTS.append({'id': pid, 'name': name, 'ok': bool(ok), 'evidence': evidence})
    print('  %-4s %-46s %s | %s' % (pid, name, 'PASS' if ok else 'FAIL', evidence), flush=True)


def load_engine():
    """按真实路径加载引擎模块 (DATA_DIR 由 __file__ 推导, 必须用真实路径)。"""
    # 引擎目录入 sys.path: 使 s5_ledger_bridge / portfolio_execution_ledger
    # 等延迟 import 可解析 (否则 P8b 会止于 harness 侧的 ModuleNotFoundError)。
    _eng_dir = os.path.dirname(ENG)
    if _eng_dir not in sys.path:
        sys.path.insert(0, _eng_dir)
    spec = importlib.util.spec_from_file_location('ec5_engine', ENG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert os.path.samefile(mod.__file__, ENG), mod.__file__
    return mod


def build_frame(lds):
    """合成帧: 真实 BJ 上市日 + 非 .BJ 对照 + 一个无映射的 .BJ 代码。"""
    codes = sorted(lds.items(), key=lambda kv: kv[1])
    pick = [codes[0][0], codes[len(codes) // 2][0], codes[-1][0]]
    no_map = '920999.BJ'
    probes = {c: lds[c] for c in pick}
    probes[no_map] = None
    nonbj = ['600000.SH', '000001.SZ', '300750.SZ', '688981.SH']

    dates = pd.bdate_range('2020-01-06', '2025-12-26', freq='7B')
    rows = []
    for code, ld in probes.items():
        for d in dates:
            rows.append((code, d, 10.0 + (d.day % 7), 10.0, 1.0e5))
    for code in nonbj:
        for d in dates:
            rows.append((code, d, 20.0 + (d.day % 5), 20.0, 5.0e6))
    frame = pd.DataFrame(rows, columns=['ts_code', 'trade_date', 'close', 'close_p', 'amount'])
    return frame, probes, nonbj


def digest(frame):
    """内容摘要 (与行序/索引无关的稳定指纹, 用于 P7 输入不变性)。"""
    h = pd.util.hash_pandas_object(frame, index=True).values
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def main():
    print('=== EC5 机制级性质测试 (J4 替代取证) ===', flush=True)
    print('time:', datetime.now().isoformat(), flush=True)
    saved_env = {k: os.environ.get(k) for k in ENV_KEYS}
    for k in ENV_KEYS:
        os.environ.pop(k, None)

    mod = load_engine()
    lds = dict(mod._bj_listing_date_map())
    frame, probes, nonbj = build_frame(lds)
    n_bj_probe = sum(1 for c in probes
                     if frame['ts_code'].eq(c).any() and c.endswith('.BJ'))
    print('  映射 %d 只 | 合成帧 %d 行 | BJ 探针 %d 只 (含 1 只无映射) | 非BJ 对照 %d 只'
          % (len(lds), len(frame), n_bj_probe, len(nonbj)), flush=True)

    before = digest(frame)
    out = mod._s5_bj_cut_pre_listing(frame)

    bj_mask_in = frame['ts_code'].astype(str).str.endswith('.BJ')
    bj_mask_out = out['ts_code'].astype(str).str.endswith('.BJ')

    # ── P1 非 .BJ 逐位恒等 ──
    try:
        pd.testing.assert_frame_equal(out[~bj_mask_out].reset_index(drop=True),
                                      frame[~bj_mask_in].reset_index(drop=True))
        idx_same = out[~bj_mask_out].index.equals(frame[~bj_mask_in].index)
        rec('P1', 'non-BJ rows bit-identical (value/order/dtype/index)',
            idx_same, 'rows_in=%d rows_out=%d index_preserved=%s'
            % (int((~bj_mask_in).sum()), int((~bj_mask_out).sum()), idx_same))
    except AssertionError as exc:
        rec('P1', 'non-BJ rows bit-identical', False, str(exc)[:200])

    # ── P2 / P3 目标类必删 / 上市后必留 ──
    bad_drop, bad_keep = [], []
    for code, ld in probes.items():
        if ld is None:
            continue
        sub = frame[frame['ts_code'].eq(code)]
        osub = out[out['ts_code'].eq(code)]
        exp_drop = int((sub['trade_date'] < ld).sum())
        exp_keep = int((sub['trade_date'] >= ld).sum())
        if len(osub) != exp_keep:
            bad_keep.append([code, int(len(osub)), exp_keep])
        elif int((osub['trade_date'] < ld).sum()) != 0:
            bad_drop.append([code, int((osub['trade_date'] < ld).sum())])
        if exp_drop == 0:
            bad_drop.append([code, 'probe-has-no-pre-window'])
    rec('P2', 'target class (.BJ & date<list_date) fully removed', not bad_drop,
        'violations=%s' % (bad_drop or 'none'))
    rec('P3', 'post-listing (.BJ & date>=list_date) fully kept', not bad_keep,
        'violations=%s' % (bad_keep or 'none'))

    # ── P4 fail-safe ──
    nm = out[out['ts_code'].eq('920999.BJ')]
    nm_in = int(frame['ts_code'].eq('920999.BJ').sum())
    rec('P4', 'fail-safe: .BJ without list_date not cut', len(nm) == nm_in,
        'no-map rows kept %d/%d' % (len(nm), nm_in))

    # ── P5 幂等 ──
    out2 = mod._s5_bj_cut_pre_listing(out)
    try:
        pd.testing.assert_frame_equal(out.reset_index(drop=True), out2.reset_index(drop=True))
        rec('P5', 'idempotent: cut(cut(x)) == cut(x)', True, 'rows=%d' % len(out))
    except AssertionError as exc:
        rec('P5', 'idempotent', False, str(exc)[:200])

    # ── P6 边界 ──
    e1 = pd.DataFrame()
    e2 = pd.DataFrame({'a': [1, 2]})
    e3 = pd.DataFrame({'ts_code': ['920999.BJ'], 'trade_date': [pd.Timestamp('2021-01-04')]})
    ok6 = (mod._s5_bj_cut_pre_listing(e1) is e1 and mod._s5_bj_cut_pre_listing(e2) is e2
           and len(mod._s5_bj_cut_pre_listing(e3)) == 1)
    rec('P6', 'boundary: empty / no-ts_code / all-unmapped-BJ', ok6,
        'identity-return on 2 cases + 1 row preserved')

    # ── P7 纯函数 ──
    after = digest(frame)
    rec('P7', 'pure: input frame unmodified', before == after,
        'digest %s -> %s' % (before, after))

    # ── P8 fail-closed (免 __init__ 数据加载: 用 __new__ 造壳) ──
    eng = load_engine()
    shell = eng.S5JointFilter.__new__(eng.S5JointFilter)
    shell.top_n = 80

    def try_call(bj, ledger):
        old = {k: os.environ.get(k) for k in ENV_KEYS}
        os.environ['FUXI_S5_BJ_UNIVERSE'] = bj
        os.environ['FUXI_S5_LEDGER'] = ledger
        try:
            shell.backtest_factor(pd.DataFrame())
            outcome = ['no-exception', '']
        except Exception as exc:  # noqa: BLE001
            outcome = [type(exc).__name__, str(exc)[:300]]
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return outcome

    a = try_call('listed_only', 'fixed_shares')
    refu = (a[0] == 'ValueError' and 'refusing to run the fixed-shares path' in a[1])
    rec('P8a', 'fail-closed: ledger!=proxy & listed_only refuses', refu, '%s: %s' % (a[0], a[1][:150]))
    b = try_call('legacy', 'fixed_shares')
    rec('P8b', 'ledger!=proxy & legacy: refusal NOT triggered',
        not (b[0] == 'ValueError' and 'refusing to run' in b[1]),
        '%s: %s' % (b[0], b[1][:150]))
    c = try_call('listed_only', 'proxy')
    rec('P8c', 'ledger=proxy & listed_only: no refusal',
        not (c[0] == 'ValueError' and 'refusing to run' in c[1]),
        '%s: %s' % (c[0], c[1][:150]))

    # ── P9 getter 值域 ──
    g = {}
    os.environ.pop('FUXI_S5_BJ_UNIVERSE', None)
    g['default'] = eng._s5_bj_universe()
    os.environ['FUXI_S5_BJ_UNIVERSE'] = 'listed_only'
    g['explicit'] = eng._s5_bj_universe()
    os.environ['FUXI_S5_BJ_UNIVERSE'] = 'bogus'
    try:
        eng._s5_bj_universe()
        g['invalid'] = 'no-raise'
    except ValueError:
        g['invalid'] = 'ValueError'
    os.environ.pop('FUXI_S5_BJ_UNIVERSE', None)
    rec('P9', 'getter: default=legacy / explicit ok / invalid fail-fast',
        g['default'] == 'legacy' and g['explicit'] == 'listed_only'
        and g['invalid'] == 'ValueError', json.dumps(g, ensure_ascii=False))

    # ── P10 映射覆盖 ──
    all_bj = all(c.endswith('.BJ') for c in lds)
    all_ts = all(isinstance(v, pd.Timestamp) for v in lds.values())
    dmin = min(lds.values()).strftime('%Y-%m-%d') if lds else None
    dmax = max(lds.values()).strftime('%Y-%m-%d') if lds else None
    rec('P10', 'listing-date map: 353 .BJ codes, valid timestamps',
        len(lds) == 353 and all_bj and all_ts,
        'n=%d all_bj=%s all_ts=%s range=%s..%s' % (len(lds), all_bj, all_ts, dmin, dmax))

    for k, v in saved_env.items():
        if v is not None:
            os.environ[k] = v

    verdict = all(r['ok'] for r in RESULTS)
    report = {
        'schema': 'fuxi_bj_universe_ec5_property_test/v1',
        'generated_at': datetime.now().isoformat(),
        'purpose': 'J4 替代取证: 变更不外溢到非 BJ 路径 (机制级性质测试)',
        'engine_sha256': hashlib.sha256(open(ENG, 'rb').read()).hexdigest(),
        'n_universe_bj_ref': {'n_universe': 5784, 'n_universe_bj': 353, 'share': 353 / 5784.0},
        'top_n_default': 80,
        'synthetic_frame_rows': int(len(frame)),
        'results': RESULTS,
        'verdict': verdict,
        'n_pass': sum(1 for r in RESULTS if r['ok']),
        'n_total': len(RESULTS),
    }
    tmp = DEST + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
        fh.write('\n')
    os.replace(tmp, DEST)
    print('\n[EC5] %d/%d PASS | verdict=%s -> %s'
          % (report['n_pass'], report['n_total'], verdict, DEST), flush=True)
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
