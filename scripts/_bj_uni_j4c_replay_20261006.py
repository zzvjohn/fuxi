# -*- coding: utf-8 -*-
"""_bj_uni_j4c_replay_20261006.py — J4 负对照批次 3（只读）

背景: 批次 1 命中 `gp_evo_002`（零 .BJ 暴露，读数逐位一致）＝ 合格负对照 #1；
批次 2 全部含 .BJ 暴露或早退。历史通过池 (23 条) 已测 17 条，剩 9 条几乎全是
价格水平族（预测早退）。本批改从 **forge round1 S5 结果** 扩池：
  data/forge_round1_s5_results.json — 18 条中 11 条有完整经济读数（4 条 passed=True）。
  其中「volume 时序 rank」族（小成交量⇒低分⇒BJ 排不进 TOP）机制上与 gp_evo_002
  同构，零暴露期望高：73961(passed) / 94631 / 99498。

🔴 保真纪律: 公式**运行时从 JSON 读 pandas_expression**, 不手抄（防截断/转写失真）。
   name 仅作标识 (ASCII slug); 原 name 保留在 tag 与 provenance 中以供追溯。

流程复用批次 1: SCREEN(legacy, enforce) → 取 ran ∧ 零 .BJ 暴露者 → PAIR(listed_only) → 对拍。
输出: reports/fuxi_v1_bj_universe_20261006/j4c_readings.json
"""
import importlib.util
import json
import os

ROOT = r'E:\quant'
spec = importlib.util.spec_from_file_location(
    'j4mod', os.path.join(ROOT, 'scripts', '_bj_uni_j4_replay_20261006.py'))
j4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(j4)

FORGE = os.path.join(ROOT, 'data', 'forge_round1_s5_results.json')
_rows = json.load(open(FORGE, encoding='utf-8'))['rows']


def _has_readings(r):
    """有完整经济读数或 passed=True 者 (排除 ex25/ex26 双 0 占位)。"""
    ex25, ex26 = r.get('excess_2025'), r.get('excess_2026')
    if ex25 is None:
        return False
    return bool(r.get('passed')) or (ex25 or 0) != 0.0 or (ex26 or 0) != 0.0


def _slug(name):
    """forge_r1_筹码分布_53790 → forge_r1_cb_53790 (ASCII 标识; 原名入 tag)。"""
    parts = str(name).split('_')
    tail = parts[-1] if parts[-1].isdigit() else ''
    kind = {'筹码分布': 'cb', '流动性×微观结构': 'liq', '价量关系': 'pv',
            '尾部风险': 'tail', '下行保护': 'dp', '波动率适应': 'vol'}
    tag = next((v for k, v in kind.items() if k in str(name)), 'x')
    return 'forge_%s_%s' % (tag, tail) if tail else 'forge_%s' % tag


CAND = []
for _r in _rows:
    if not _has_readings(_r):
        continue
    _f = (_r.get('pandas_expression') or '').strip()
    if not _f:
        print('[skip-no-formula]', _r.get('factor_name'))
        continue
    CAND.append(dict(name=_slug(_r['factor_name']), tag=str(_r['factor_name']),
                     formula=_f, freq='daily',
                     forge_passed=bool(_r.get('passed')),
                     forge_ex25=_r.get('excess_2025'), forge_ex26=_r.get('excess_2026')))

j4.CANDIDATES = CAND
j4.READINGS = os.path.join(j4.OUT, 'j4c_readings.json')

if __name__ == '__main__':
    print('[j4c] %d forge candidates; formulas read live from %s'
          % (len(CAND), os.path.basename(FORGE)), flush=True)
    for c in CAND:
        print('   %-22s <- %-30s passed=%-5s' % (c['name'], c['tag'], c['forge_passed']), flush=True)
    j4.main()
