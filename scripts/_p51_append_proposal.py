# -*- coding: utf-8 -*-
"""一次性脚本：向 data/stage1_optimization_proposals.json 追加 P-20261013-014（P5.1）。"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'data' / 'stage1_optimization_proposals.json'
NEW_ID = 'P-20261013-014'

ENTRY = {
    'proposal_id': NEW_ID,
    'date': '2026-10-13',
    'created_at': '2026-10-13T17:00:00+08:00',
    'title': (
        'P5.1 shadow UCB 状态累积修复（arms_state_rr/ucb 双账本 + pending 记录 + resolve 闭环 + replay 快照恢复）'
        '+ run_campaign 轮数上限 50→200 + 20 轮回归 + 120 轮统计级影子对比（74.43s 零 error，'
        'ucb_priority 92/120）+ JQ 因子产出评估（74 候选脚本管线通、合成数据 admission 全拦设计如此、'
        'TOP10 样本包交付）'
    ),
    'category': '伏羲 v2.0 落地 / P5 影子管线',
    'domains': ['T3模块/通道'],
    'risk_level': 'low',
    'status': 'landed',
    'proposal_change_kind': 'engineering',
    'generator_hint': 'manual',
    'trigger': (
        'P5_TRIAL_20R 发现 1：hook_arm_selection 每轮新建 arms_state={}，shadow UCB 永远冷启动选 A_template'
        '（20/20 无区分度）。蓝图审计结论：统计级验收（§9 ③）需 ≥100 轮 + 修复状态累积。'
        '另发现 run_campaign 的 range(rounds) 结构下 resume 无法续轮（round_index 每次从 0 重数），'
        '而 require 上限 50 阻断了 100+ 轮单次调用。'
    ),
    'current_state': (
        'P5.1 之前：ShadowState 只持有 dedup/cooldown/feedback 三个跨轮状态；arms.py 的 select_action'
        ' 原地更新 arms_state 但调用方每次新建 dict；resolve_microbatch 无任何调用点 → UCB 奖励通道'
        '（r_dev_sum/r_dev_count）恒空。'
    ),
    'proposed_change': (
        '① landing_shadow.py：ShadowState 新增 arms_state_rr / arms_state_ucb / pending_shadow；'
        'hook_arm_selection 用累积账本并记录 pending；hook_dedup_cooldown 以生产轮新行为占比为 r_dev，'
        'resolve_microbatch 关闭 microbatch（循环清 pending 兼容崩溃恢复）；两个 hook 的 fact payload'
        ' 均附双臂账本快照；replay 按 rowid 顺序覆盖式恢复（同轮 dedup 快照覆盖 arm_selection 快照）。'
        '② campaign.py：rounds 上限 50→200（活跃测试无锁定；rounds 相关下游逻辑无 ≤50 假设）。'
        '③ test_p5_shadow_wiring.py 新增 StatefulUcbTests 4 项。'
        '④ p5_shadow_stat_120r.py（120 轮统计运行/重分析）+ p5_jq_extract.py（TOP10 样本提取）。'
    ),
    'implementation_scope': (
        'fuxi_v1/landing_shadow.py（三处）; fuxi_v1/campaign.py（require 一处）; '
        'fuxi_v1/tests/test_p5_shadow_wiring.py（+4 项，16/16）; '
        'reports/fuxi_v2_landing_20261013/{p5_shadow_stat_120r.py, p5_shadow_stat_120r_analysis.json, '
        'p5_jq_extract.py, p5_shadow_jq_samples/（TOP10 + MANIFEST）, P5_SHADOW_STAT_120R.md}'
    ),
    'expected_impact': (
        '生产决策路径零变化（影子账本字段只增；两开关默认维持 v0.7.186 翻 on 状态，本批不翻新开关）。'
        '修复验证：20 轮回归 ucb chosen 不再恒 A（cold 4/forced 4/priority 12）；120 轮 cold 4/'
        'forced 24/priority 92，账本 dispatch=resolve 逐臂平衡 pending 0，shadow_error 0。'
        '统计读数（合成数据，r_dev 含零产出轮计 0 分）：ucb C 0.367 / A 0.364 / E 0.355 / B 0.154，'
        'B 与 C 的 Wilson 95% 不相交——但归因分析确认差异源于 82/120 零产出轮与 forced_explore 时序耦合'
        '（B 26 轮中 22 空），不构成臂能力裁决（报告发现 A 如实记录）。'
        'JQ 因子：74 valid 候选的 jq_strategy.py（296 行自包含聚宽脚本）全部生成；合成数据 admission '
        '恒拦（synthetic_data + 独立合同缺失 + E 阶段 reject，设计如此）；TOP10 样本包随附 blockers 透明清单。'
        '全量回归 678 项：零新增回归（227 errors 全为环境性 trash guard/pyarrow/外部数据，1 FAIL 预存在）。'
    ),
    'landing_note': (
        '本批只改影子侧（landing_shadow.py）+ 一处契约常量（rounds 上限）。生产决策对象零改动，'
        'shadow 读数仍只记录不裁决。统计口径如实：r_dev 是「生产产出归因给影子臂」的离线反事实近似，'
        '合成数据 68% 空轮使臂间差异被时序污染，报告明示「不构成对 B_edit 的裁决」。'
        'campaign.py 是 frozen code snapshot 的一部分，本批新 campaign 用新代码哈希，旧目录不受影响。'
    ),
    'verification': (
        'python -m unittest fuxi_v1.tests.test_p5_shadow_wiring => 16/16 OK（含新增 4 项）; '
        'test_p2_five_arms + test_p3_complement_feedback + test_p4_export => 46/46 OK; '
        'p5_trial_20r.py 复跑 => ucb 序列不再恒 A、shadow_error 0; '
        'p5_shadow_stat_120r.py => 120 轮 74.43s、attempted 74、五类钩子 296 facts 零 error; '
        'p5_shadow_stat_120r.py --reanalyze => 口径修正后重出统计; '
        'p5_jq_extract.py => TOP10 样本包 + MANIFEST.json; '
        '全量 unittest discover => 678 项，1 FAIL（预存在 _limits 80≠6）+ 227 errors 全环境性，零新增回归。'
    ),
    'artifacts': [
        'fuxi_v1/landing_shadow.py',
        'fuxi_v1/campaign.py',
        'fuxi_v1/tests/test_p5_shadow_wiring.py',
        'reports/fuxi_v2_landing_20261013/p5_shadow_stat_120r.py',
        'reports/fuxi_v2_landing_20261013/p5_shadow_stat_120r_analysis.json',
        'reports/fuxi_v2_landing_20261013/p5_jq_extract.py',
        'reports/fuxi_v2_landing_20261013/p5_shadow_jq_samples/',
        'reports/fuxi_v2_landing_20261013/P5_SHADOW_STAT_120R.md',
        'reports/fuxi_v2_landing_20261013/p5_trial_20r_analysis.json（复跑覆盖）',
    ],
    'zero_adjudication': True,
    'default_on': False,
    'ab_evidence': (
        '修复前后对照：20 轮 pilot v0.7.186 时 ucb chosen 恒 A（20/20 cold_start）vs 修复后同配置复跑'
        'cold 4/forced 4/priority 12。rr/ucb 双账本独立性：rr 恒 fixed_rotation 96+forced 24，'
        'ucb 账本不受轮转 dispatch 污染（StatefulUcbTests 断言两账本对象不同、计数各自平衡）。'
        'replay 一致性：8 轮事实重建 vs 直跑账本逐字段相等。'
        '全量回归对照：本次改动文件与全部 error 测试零交集（206 trash guard/10 pyarrow/8 unified_runner'
        '/4 外部数据/1 预存在 FAIL），判定零新增回归。'
    ),
    'known_limit': (
        '① r_dev 是生产产出归因给影子臂的离线反事实，真实数据下同样适用，解释须保留此限定；'
        '② 合成数据 AST 空间前 ~20 轮枯竭 → 120 轮中 82 轮零产出，臂间 r_dev 差异被时序/生产臂偶合污染，'
        '统计方法（Wilson/终值排序）已验证但因果比较需真实数据；'
        '③ 120 轮为单次运行，独立重复（≥3 次）留待真实数据批次；'
        '④ contract_gap（生产 strategy_ir vs P4 合同 13 字段）未动，export 74 行仍全 contract_gap；'
        '⑤ JQ 样本包 admission 未放行，仅作待回测样本，不冒充资格。'
    ),
    'next_step': (
        '下一批候选：①真实数据 full-market columnar 快照 + 独立数据合同跑 100+ 轮 × 3 重复'
        '（真正产出 jq_candidate_allowed=True 的因子）；或 ②先修 contract_gap 让 export 钩子出现'
        '第一行 valid。'
    ),
    'rollback': (
        'landing_shadow.py 回退至 v0.7.186 版本（三处改动逐块还原）; campaign.py require 上限改回 50; '
        'test_p5_shadow_wiring.py 删除 StatefulUcbTests 类即可；报告与样本包为纯产物，删除目录即可。'
    ),
}


def main() -> int:
    raw = TARGET.read_bytes()
    assert raw[:3] != b'\xef\xbb\xbf', 'target unexpectedly carries a BOM'
    newline = b'\r\n' if raw.count(b'\r\n') > 0 else b'\n'
    assert newline == b'\r\n' or raw.count(b'\n') == raw.count(newline), 'mixed newlines unsupported'
    data = json.loads(raw.decode('utf-8'))
    assert isinstance(data, list)
    ids = [x.get('proposal_id') for x in data]
    if NEW_ID in ids:
        raise SystemExit('refusing to append: %s already present' % NEW_ID)
    before = len(data)
    last_before = ids[-1]

    data.append(ENTRY)
    text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    if newline == b'\r\n':
        text = text.replace('\n', '\r\n')
    payload = text.encode('utf-8')
    assert payload[:3] != b'\xef\xbb\xbf'

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = TARGET.with_name(TARGET.name + '.bak_%s_p51' % stamp)
    shutil.copy2(TARGET, backup)

    with open(TARGET, 'wb') as fh:
        fh.write(payload)

    got = TARGET.read_bytes()
    assert got == payload, 'byte-level round-trip mismatch'
    assert got[:3] != b'\xef\xbb\xbf'
    assert got.count(newline) == payload.count(newline), 'newline style drifted'
    again = json.loads(got.decode('utf-8'))
    assert isinstance(again, list) and len(again) == before + 1, 'entry count wrong'
    assert again[-1]['proposal_id'] == NEW_ID
    got_ids = [x.get('proposal_id') for x in again]
    assert got_ids[:-1] == ids, 'prior entries mutated'
    assert last_before in got_ids, 'previous last entry lost'
    assert again[-1]['status'] == 'landed'
    assert again[-1]['proposal_change_kind'] == 'engineering'
    assert again[-1]['zero_adjudication'] is True and again[-1]['default_on'] is False

    print('ok: %s -> %s entries (was %s, new last = %s)'
          % (before, len(again), last_before, NEW_ID))
    print('bytes %d (was %d)  backup %s' % (len(got), len(raw), backup.name))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
