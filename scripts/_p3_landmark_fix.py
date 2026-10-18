# -*- coding: utf-8 -*-
"""一次性脚本（P3 修正）：把 P3 批次声明并入 '2026-09-24' 的 LANDMARK（同日只做追加），
并移除误插的 '2026-09-25' 键。

背景（本批的工程发现）：
    ``version_bump.load_proposals()`` **不读提案的 date 字段** —— 它用正则从
    proposal_id ``P-YYYYMMDD-NNN`` 里取日期。因此 ``--bump`` 的版本日期恒为**批次日**，
    与墙上时钟无关。'2026-09-25' 键不会被任何版本引用 ⇒ 属孤儿，必须撤掉。

纪律：
  ① 同日 landmark 只允许**追加**（新值必须以前缀方式包含旧值），不得改写历史；
  ② 先 encode('utf-8') 再 open('wb')；
  ③ 备份 + 回读断言 + import 后 LANDMARK 可解析。
"""
from __future__ import annotations

import importlib.util
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'scripts' / 'version_bump.py'
BATCH_DATE = '2026-09-24'
STRAY_DATE = '2026-09-25'

BODY = [
    '本日 P3 批次 v0.7.70（同日追加）：修上游入口可交易性检查。',
    '缺陷复算成立：13/13701 个入场事件落在无行情行日（llm1 11 + mem 2），13/13 在本地停牌表里是整日停牌（suspend_type=S、suspend_timing 空），'
    '入场事件总数 13701 逐臂复现（4771/2114/2582/4234）。',
    '历史对账：旧记录「11 个落在停牌日」是当时停牌表末端 2026-04-30 造成的覆盖数字，不是判定分歧 —— '
    '002731.SZ 与 688121.SH（均 2026-05-06）当时落在表外，既不能证停牌也不能证成交；停牌表已扩到 2026-09-23，复算这 2 例同为整日停牌 ⇒ 13 = 11 + 2。',
    '🔴 真读数修正：门按「每 stamp 每名字」计数，准入级拒绝 30 个（llm1 22 / mem 7 / r4 1，30/30 全是 refused_halted_full_day），'
    '其中 13 个是入场事件、另 17 个是 carry-over（该证券早已在册，却因仍在目标里被下一个 stamp 继续重提）—— '
    '17 条此前从未被任何一批报出，只在准入级显形 ⇒ 门必须装在准入级，只修入场级会把这 17 条原样留下。',
    '两处此前静默的读数逐条列出：4 个 stamp 的生效日落在交易日历之外（四臂各 1 个、均 2026-09-09，此前被探针静默跳过且 13701 不含其持有日）；'
    '4 例被拒准入其本地行情序列已结束（全在 llm1），与「那天刚好停牌」分开报。',
    '交付：research/factor_alchemy/entry_tradability.py（门本体，8 档穷尽互斥判定表、准入集合={tradable}、fail-closed、缺证据与已证伪分开计数、25 条测试）；'
    '开关 env FUXI_ENTRY_TRADABILITY 默认 annotate（零裁决观察者，含 shadow/on/1 别名）、enforce 需显式开启且权仍按 1/top_n 不重归一化、off/legacy 完全禁用，'
    '未知取值抛 ValueError 不静默降级；scripts/fuxi_entry_tradability_20260924.py（批次脚本 + 四关验收，31 条测试）。',
    '四关：关一 getter 逐事件直查与门判定分歧 0，对照臂（同一证券顺延到之后第一个有行情行日、只差一行行情）13/13 判 tradable ⇒ 门不是返常量；'
    '关二 A/B removed 30 / added 0，入场事件删除 13、另有 7 个被推到首个可成交日（逐条列出，「删除」不是全部事实），'
    '封存目标在 off 与 annotate 两档对四臂全部逐字节相同；'
    '关三 默认即文档所写、四别名同解、off=annotate=shadow 目标摘要相同而 enforce 不同（观察者一个字节都没动）、counterfactual 非空；'
    '关四 端到端 11 例被下游吸收成 filled=0/reason=[halted] 的订单、2 例因该臂账本止于 2026-05-06（data_insufficient）未被触达、'
    '0 例在门判拒的日子成交（这个「11」与旧记录的「11 个落在停牌日」是两个不同的 11，纯属巧合，必须分开记）。',
    '接线点清单 8 处逐条回读源码校验行号与模式（漂移即抛错、不用 grep 推断、「有没有门」按实现判），has_entry_tradability_check 全为 false —— '
    '1 处冻结构建器被封存 protocol.json/code_hashes 逐字节钉死（接线需新协议）、3 处研究账簿构建器、2 处诊断 + 1 处中性化归因、1 处 P2 研究口径基准组合，'
    '改它们等于改那些批次的读数 ⇒ 本批不接任何生产调用点（「缺口已可判定、可计数、可逐条列举」≠「已修好建仓」）。',
    '对干旱的实质帮助（限度优先）：13/13701 = 0.09%，不足以把实测 IC 从 +0.054 推到盈亏平衡 +0.108，本批不生产因子，'
    '属「必要不充分」的评价器输入完整性修复 —— ① 把「市场拒绝成交」与「目标想持有」分开（干旱三因之「反馈噪音」的具体来源）；'
    '② 暴露准入级 30 ≠ 入场级 13 这个此前不可见的量级，「测试形状必须与生产一致」在计数口径上再一次复现；'
    '③ 交付可复用入口校验件，供大宗交易/研究特征库/R5 新截面在进组合层读数前先摘掉结构上不可能成交的样本。',
    '本批未解旱、未产生解旱新因子、未授予 JQ 资格、未翻任何默认值、未改任何封存产物字节；R5 未知样本仍为 0；'
    '2024-12-27~2026-09-09 全为研究已暴露窗口，不得重新包装成样本外。详见 reports/fuxi_entry_tradability_20260924.md。',
]

PROBES = ('P3 批次 v0.7.70', 'entry_tradability', 'carry-over', '第三项', 'FUXI_ENTRY_TRADABILITY')


def emit_body_lines():
    lines = []
    for i, frag in enumerate(BODY):
        tail = ',' if i == len(BODY) - 1 else ''
        lines.append("                  '%s'%s" % (frag, tail))
    return lines


def main() -> int:
    raw = TARGET.read_bytes()
    assert raw[:3] != b'\xef\xbb\xbf', 'unexpected BOM'
    src = raw.decode('utf-8')
    lines = src.split('\n')

    # --- 步骤 1: 移除误插的 STRAY_DATE 块 ---
    stray = None
    for i, line in enumerate(lines):
        if line.startswith("    '%s':" % STRAY_DATE):
            stray = i
            break
    if stray is not None:
        close = None
        for j in range(stray + 1, len(lines)):
            if lines[j] == '}':
                close = j
                break
        assert close is not None, 'closing brace not found after the stray block'
        lines = lines[:stray] + lines[close:]
        print('[fix] removed stray LANDMARK key %s (%d lines)' % (STRAY_DATE, close - stray))
    else:
        print('[fix] no stray key to remove')

    # --- 步骤 2: 追加到 BATCH_DATE 的条目 ---
    anchor = None
    for i, line in enumerate(lines):
        if line.startswith("    '%s':" % BATCH_DATE):
            anchor = i
            break
    assert anchor is not None, 'anchor %s not found' % BATCH_DATE

    close = None
    for j in range(anchor + 1, len(lines)):
        if lines[j] == '}':
            close = j
            break
    assert close is not None, 'closing brace not found'
    last = lines[close - 1]
    assert last.rstrip().endswith("',"), 'unexpected tail: %r' % last[:100]

    old_value_len = len(''.join(lines[anchor:close]))
    # 把该条目最后一行结尾的 ',' 去掉（保留闭合引号），再续上 P3 正文
    lines[close - 1] = last.rstrip()[:-1]
    lines = lines[:close] + emit_body_lines() + lines[close:]
    text = '\n'.join(lines)
    payload = text.encode('utf-8')
    assert payload[:3] != b'\xef\xbb\xbf'

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = TARGET.with_name(TARGET.name + '.bak_%s_p3fix' % stamp)
    shutil.copy2(TARGET, backup)

    with open(TARGET, 'wb') as fh:
        fh.write(payload)

    got = TARGET.read_bytes()
    assert got == payload, 'byte-level round-trip mismatch'
    back = got.decode('utf-8')
    assert "'%s':" % STRAY_DATE not in back, 'stray key still present'
    assert "'%s':" % BATCH_DATE in back, 'batch-date landmark lost'
    assert back.count("    '2026-09-23':") == src.count("    '2026-09-23':"), 'history mutated'
    for probe in PROBES:
        assert probe in back, 'probe missing: %r' % probe

    spec = importlib.util.spec_from_file_location('version_bump_check2', str(TARGET))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert STRAY_DATE not in mod.LANDMARK, 'stray key still parses'
    assert 'P3 批次 v0.7.70' in mod.LANDMARK[BATCH_DATE], 'P3 declaration missing'
    assert mod.LANDMARK[BATCH_DATE].startswith('第三项'), 'historical text corrupted'
    print('ok: bytes %d -> %d  条目块 %d -> %d chars  LANDMARK dates %d  rendered value len %d'
          % (len(raw), len(got), old_value_len,
             len(''.join(emit_body_lines())) + old_value_len - 1,
             len(mod.LANDMARK), len(mod.LANDMARK[BATCH_DATE])))
    print('backup %s' % backup.name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
