# -*- coding: utf-8 -*-
"""一次性脚本：向 data/stage1_optimization_proposals.json 追加 P-20261006-005（M4 翻默认）。

行尾纪律：当前文件实测为纯 CRLF（10051 CRLF / 0 裸 LF，2026-10-06），写回必须保持 CRLF。
内容：M4 FITNESS_TRIAGE 翻默认 off→enforce（四关齐；同锚/无 memory 零行为变更）。
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'data' / 'stage1_optimization_proposals.json'
NEW_ID = 'P-20261006-005'

ENTRY = {
    'proposal_id': NEW_ID,
    'title': 'M4 FITNESS_TRIAGE 翻默认：FUXI_V1_FITNESS_TRIAGE 默认 off→enforce（四关齐；同锚/无 memory 零行为变更）',
    'category': '系统开关/默认值变更',
    'domains': ['T3模块/通道'],
    'risk_level': 'controlled_research_only',
    'status': 'landed',
    'proposal_change_kind': 'behavior',
    'generator_hint': 'manual',
    'trigger': (
        '用户指令『M4 翻默认』（2026-10-06 拍板，M2 判读后）。前置证据：M4 enforce A/B 已闭环'
        '（R1–R6 全绿、failures=[]）+ 本批翻默认验证链 F1–F5 全绿。'
    ),
    'current_state': (
        '开关 FUXI_V1_FITNESS_TRIAGE=off|shadow|enforce（campaign.py:284，默认 off）。唯一 enforce '
        '消费者 = campaign.py:396-403 imported-memory 排序：off/shadow=sorted(-mean_rank_ic)，'
        'enforce=sorted(fitness_key)（ED 优先 / 缺 ED 回退 IC 沉底 / 无读数沉底 / tie=candidate_id）。'
        '台账：fitness_triage facts（off 不写；shadow/enforce 每轮写）。'
    ),
    'proposed_change': (
        'campaign.py:284 默认值 "off"→"enforce"（受控 bytes patch：44905→44909 bytes，+4；纯 LF 不变，'
        '裸 LF 640→640；sha f4a3c945…→c11d0f3c…）。值集不变（off/shadow/enforce 均合法）；回退=显式 '
        'FUXI_V1_FITNESS_TRIAGE=off。测试同步 2 处（test_fitness_triage.py：默认档断言 off→enforce+'
        'facts=2；shadow 测试的 off 对照改显式 env）。'
    ),
    'expected_impact': (
        '同锚全 ED 批次排序零差异（ED=IC/breakeven 单调线性 + A/B 实证）；无 memory 场景结构等价'
        '（A/B S1 + 本批 F3b 四档签名一致）。收益：①排序语义升级（缺 ED 卡沉底、无读数沉底；未来多锚'
        '场景按 ED 正确排序）；②默认档每轮写 fitness_triage 台账（+1 条/轮，纯新增）。成本：混合批次'
        '（legacy 缺-ED 卡/无读数卡）排序变化 by design（已披露）；历史 off campaign resume 须显式旧值'
        '（fail-closed，R0 契约本义）。'
    ),
    'implementation_scope': (
        'fuxi_v1/campaign.py 默认值一行（受控 bytes patch + 四项对账 + pre_m4flip 备份）；'
        'fuxi_v1/tests/test_fitness_triage.py 2 处同步；验证链 '
        'reports/fuxi_v1_m4_flip_20261006/{FLIP_PLAN.md, verify_m4_flip.py, verify_m4_flip.json}。'
    ),
    'four_gates': (
        '①getter实测=pass（F1：默认档=enforce；off/shadow/enforce 合法；bogus ContractError）；'
        '②A/B反事实=pass（M4 A/B R1–R6 全绿 failures=[] + F2a 引用断言）；③默认↔口径不变量=pass'
        '（F3a config 行为面全等〔身份/时间键除外〕+ F3b 无 memory 四档签名一致 + F3c 默认≡显式 off）；'
        '④端到端对账=pass（F4a 全量 280 项 OK；F4b resume 拒绝+回退可用；F4c code_hashes 冻结校验通过）。'
        '2026-10-06 用户签核翻默认。'
    ),
    'landed_detail': (
        '翻默认落地 2026-10-06（用户签核『M4翻默认』）。①代码 patch：campaign.py:284 默认值 off→enforce'
        '（44905→44909 bytes；裸 LF 640 恒定；sha f4a3c945…→c11d0f3c…）；备份 '
        'reports/fuxi_v1_m4_flip_20261006/pre_m4flip/campaign.py.bak。②测试同步：'
        'test_default_run_records_triage_facts（断言 facts=2 / mode=enforce / config=enforce）+ shadow '
        '测试 off 对照改显式 env。③验证链 F1–F5 全绿（21 项检查；F3a 探针自纠：初版把臂身份/时间键纳入'
        '比较面 → 排除后行为面全等，diff 仅 campaign_id/registered_at/authority_path）。④全量回归 280 项 '
        'OK（基线解释器 envs/default；Python314 缺 pypdf 属环境差异非代码问题）。'
    ),
    'verification_scope': (
        '落地后现场实测：F1 getter 4 项 / F2b 默认 vs enforce 逐位（decisions parent_ids + candidate_results '
        '+ config）/ F3b 无 memory 四档 / F4b resume 语义（默认档拒绝 + 显式 off 通过）/ F5a/F5c 台账；'
        'verify_m4_flip.json failures=[]。'
    ),
    'rollback_plan': (
        '显式设 FUXI_V1_FITNESS_TRIAGE=off 即回退（值集不变）；历史 campaign resume 须显式旧值'
        '（fail-closed，R0 契约本义）；字节级 revert 由 pre_m4flip 备份保证。'
    ),
    'jq_validation_required': False,
    'jq_validation_status': 'no_jq_admission_granted_research_only',
    'references': [
        'reports/fuxi_v1_m4_flip_20261006/FLIP_PLAN.md',
        'reports/fuxi_v1_m4_flip_20261006/verify_m4_flip.json',
        'reports/fuxi_v1_m4_enforce_ab_20261006/m4_ab_comparison.json',
        'fuxi_v1/campaign.py',
    ],
    'cross_verified': False,
    'source': '研究批次（M4 A/B + 翻默认验证链 F1–F5）',
}


def main():
    raw = TARGET.read_bytes()
    crlf_before = raw.count(b'\r\n')
    bare_before = raw.count(b'\n') - crlf_before
    assert crlf_before > 0 and bare_before == 0, 'source file is not pure CRLF'
    data = json.loads(raw.decode('utf-8'))
    assert isinstance(data, list)
    ids = [x.get('proposal_id') for x in data]
    assert len(ids) == len(set(ids)), 'duplicate ids present'
    if NEW_ID in ids:
        raise SystemExit('refusing to append: %s already present' % NEW_ID)
    before = len(data)
    last_before = ids[-1]

    data.append(ENTRY)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    payload = text.replace('\n', '\r\n').encode('utf-8')
    assert payload[:3] != b'\xef\xbb\xbf'
    assert payload.count(b'\n') == payload.count(b'\r\n'), 'output is not pure CRLF'

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = TARGET.with_name(TARGET.name + '.bak_%s_p005' % stamp)
    shutil.copy2(TARGET, backup)

    with open(TARGET, 'wb') as fh:
        fh.write(payload)

    got = TARGET.read_bytes()
    assert got == payload, 'byte-level round-trip mismatch'
    assert got.count(b'\n') - got.count(b'\r\n') == 0, 'bare LF after write'
    again = json.loads(got.decode('utf-8'))
    assert isinstance(again, list) and len(again) == before + 1, 'entry count wrong'
    assert again[-1]['proposal_id'] == NEW_ID
    got_ids = [x.get('proposal_id') for x in again]
    assert got_ids[:-1] == ids, 'prior entries mutated'
    assert last_before in got_ids, 'previous last entry lost'
    assert again[-1]['status'] == 'landed'
    assert again[-1]['proposal_change_kind'] == 'behavior'

    print('ok: %s -> %s entries (was %s, new last = %s)'
          % (before, len(again), last_before, NEW_ID))
    print('bytes %d (was %d)  backup %s' % (len(got), len(raw), backup.name))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
