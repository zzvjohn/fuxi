# -*- coding: utf-8 -*-
"""一次性脚本：向 data/stage1_optimization_proposals.json 追加 P-20260924-012。

纪律（与 _p1_append_proposal.py 同构）：
  ① 先查占用，占用即拒绝；
  ② 先 encode('utf-8') 拿字节，再 open('wb') 落盘（避免 open('w') 先截断后编码失败清空文件）；
  ③ 落盘前备份；
  ④ 写后回读字节级断言（相等 + 无 BOM + CRLF=0 + 条数 + 对方条目仍在）。
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'data' / 'stage1_optimization_proposals.json'
NEW_ID = 'P-20260924-012'

ENTRY = {
    'proposal_id': NEW_ID,
    'date': '2026-09-24',
    'created_at': '2026-09-24T15:10:00+08:00',
    'title': (
        'P2 建「研究特征库 + 策略晋级库」分层：闸门单向下锁、A/B 空库与真实库逐字节相同；'
        '研究口径首次把 IC 层读数换算为组合层净收益——margin_rz_buyrate 净减 -8.5bps(t=-3.56) / -9.6bps(t=-3.69)，'
        '旧结论「IC 冗余即弃」被证不足以判死外族候选'
    ),
    'category': '因子研究/信息面',
    'domains': ['T1闸门/裁决', 'T2数据口径', 'T3模块/通道'],
    'risk_level': 'low',
    'status': 'landed',
    'proposal_change_kind': 'research',
    'generator_hint': 'manual',
    'trigger': (
        'P0 与 P1 联合结论：五族 IC 上限全不达标（大宗交易补全覆盖后由 0.1062 降为 0.0666、闭合为 CEILING_BELOW），'
        '而盈亏平衡 IC 为 +0.108（等权）/ +0.083（R4），缺口等于「IC 翻倍」，靠新增信息族做不到。'
        'P1 的 next_step 明确要求：把这些族统一归入只供研究、无 JQ 晋级资格的研究特征库，'
        '并按「固定基准组合上的增量净收益」评估互补性，而不是继续用单因子 IC 做代理。'
    ),
    'current_state': (
        '在 P2 之前，系统里没有「研究可用」与「能晋级」的区分：任何名字只要名字出现在晋级终点，就可能被生成 JQ 回测代码。'
        '五族 IC 上限全不达标，但它们并非无用——问题在于「不达标的族还能怎么用、以什么口径评估」没有制度化的答案；'
        '同时 P0/P1 只能给出 IC 层读数，而盈亏平衡点是组合层概念（水平项 -72~-84bps/5 日），两者之间没有换算通道。'
    ),
    'proposed_change': (
        '建两层登记（research_feature 8 条 / promotion_candidate 7 条 / pending_measurement 1 条）与一条单向下锁的闸门，并把闸门接进真正的晋级终点：'
        '① research/factor_alchemy/feature_tier.py（新建）：登记表读写、单向下锁闸门、研究口径评估器、23 项单元测试；'
        'policy 逐字为 power=downgrade_only / grants_promotion=false / unregistered_names=unaffected_legacy_behaviour / '
        'standalone_ic=not accepted as promotion evidence。'
        '② 闸门接在 scripts/gen_jq_s5_standalone.py::generate() 首行（新增 _tier_gate），失败处理 fail-closed：'
        '被登记为研究特征抛 NotPromotableError 且异常原样透传；登记库是目录/JSON 损坏/模块不可导入抛 RuntimeError 仍然拒绝。'
        '③ 研究口径（事前固定、不调参）：固定基准组合 = 现有复合信号 Top80 等权；两个叠加模式 exit_filter（剔除最低 20% 分位，其余等权）'
        '与 tilt（按横截面秩加权，缺失值取中性 0.5、不插补为信号）；成本 15bps × 单边换手；'
        '同一天 4 个臂的信号先聚合成 1 个数再对日期读 t（否则 n 被放大 sqrt(4)）。'
        '④ 分母纪律：每个读数同时报 base_book_sections / sections_read / sections_refused / refusals，'
        '「348 个截面被检查、92 个被读到」不得被压平成「读了 348 个截面」。'
        '⑤ 三种「没有读数」分开报：not_readable_here（本框架看不见该族，附指向实际测过它的批次指针）、'
        'not_yet_usable（列能取到但每截面有效值不足 5 只、叠加被拒）、rejected_before_any_reading（连基准组合都建不起来）。'
    ),
    'implementation_scope': (
        'research/factor_alchemy/feature_tier.py（新建，分层模块 + 23 项测试）; '
        'data/factor_tier_registry.json（新建，登记表）; '
        'scripts/gen_jq_s5_standalone.py（修改，新增 _tier_gate 并接在 generate() 首行）; '
        'scripts/fuxi_tier_gate_ab_20260924.py（新建，A/B + 负向 + fail-closed 证据）; '
        'reports/feature_tier_20260924/summary.json（研究口径读数）; '
        'reports/feature_tier_20260924/gate_ab/gate_ab.json（闸门证据）; '
        'reports/fuxi_feature_tier_20260924.md（报告）'
    ),
    'expected_impact': (
        '闸门（11 项检查全绿，all_checks_passed=true）：'
        'A_legacy_generates_all 空库 7/7 生成；B_real_registry_generates_all 真实库 7/7 生成；'
        'A_equals_B_byte_for_byte 掩掉「生成时间:」一行后逐字节相同（空库 = legacy 行为，可证）；'
        'gate_is_reached_not_dead 探针计数 7+7 次，闸门非死代码；'
        'D_blocked_name_is_refused 抛 NotPromotableError、D_no_file_written_on_refusal 零文件写出、D_others_still_generate 其余仍生成；'
        'E1_registry_is_a_directory / E2_registry_is_corrupt 均 7/7 全拒并抛 RuntimeError（fail-closed 覆盖到「读」而不只是「import」）。'
        'A_matches_existing_artifacts：7 个全部 differing，披露为模板演进（盘上文件由更早模板生成）、非闸门失败，且刻意不覆盖盘上文件。'
        '研究口径读数（2,088 次评估调用 = 4 臂 x 87 周频截面 x 6 列，frozen_drops=0，成本 15bps，按日期聚类）：'
        'margin_rz_buyrate 唯一显著：exit_filter 净 -8.5bps（t=-3.56）、tilt 净 -9.6bps（t=-3.69），176/348 截面可读、44 个日期，两模式同号；'
        'margin_rz_chg5 -2.5bps(t=-0.76) / -2.6bps(t=-0.62)；fcst_yoy_mid -1.7bps(t=-0.86) / -0.4bps(t=-0.19)；'
        'fcst_range_ratio -2.7bps(t=-1.03) / -1.5bps(t=-0.53)（后三列区间全跨零，不构成任何方向的读数）。'
        'lh_event_mask（lh_net_rate / lh_seat_asym）：348/348 个截面全部 overlay_too_thin（每截面有效值不足 5 只）=> 判 not_yet_usable，'
        '这是关于口径可用性的更正（旧结论「只能做出门过滤」要再退一步为「当前稀疏度下连出门口过滤都做不了」），不是效应为零。'
        'ind_mom20_idx/ind_mom20_mem（P0）、bt_premium/bt_share（P1）、benchmark_timing_overlay（P0，无逐名可用列）判 not_readable_here，附批次指针。'
        '两条工程发现：① 存在两个互不相交的 S5 记录系统（生成器散文注释 7 条 structured=false；'
        'data/_s5_historical_pass_formulas.json 23 条 structured=true；名字交集为 0）；'
        '② research/factor_alchemy/jq_s5_pass_*.py 盘上 70 个文件是登记表（7 个）的超集，盘上文件不是可靠成员来源，成员一律读登记表。'
        '对干旱的实质帮助：① 首次把「IC 层读数」换算成「组合层的钱」，从此可直接问「它能不能补上那 72~84bps」而不必靠 IC 做代理；'
        '② margin_rz_buyrate 在 IC 层曾被判 redundant（被 turnover_rate 单字段复现，差 4e-5、秩相关 0.62~0.70），'
        '组合层却是显著负（t≈3.6）——一个 IC 层被复现的字段在组合层可以显著有害、也因此反向后可能显著有益，'
        '⇒「IC 冗余」不足以判死外族候选，这是对 P0 的一条修正；'
        '③ 负读数把外族扩展的期望值进一步向下修正（五族 IC 上限全不达标、其中一族组合层还显著负），资源应转向组合层水平项本身；'
        '④ 把「条件特征」的门槛从覆盖率推进到截面最小样本数（每截面不足 5 只有值即无法叠加）。'
    ),
    'landing_note': (
        '零裁决、零影子开关、未翻任何默认值；闸门是纯减法（只拒晋级、从不授予），未登记的名字行为完全不变。'
        '不翻符号：margin_rz_buyrate 的负读数是在预注册方向上的读数，「看到负号就反用」正是 H1 当初被要求写成独立假说的原因，'
        '反用须新预注册 + 独立截面，本批不做。'
        '本批暴露并留证五轮缺陷，均未用成功轮覆盖：'
        '轮 1 capture_frames 的 keep 白名单丢掉被评估列本身 => 捕获副本不可评估，生产跑满 2 分钟后才 KeyError（非空输入产出空结果，同型第 3 次）；'
        '轮 2 把 mean_ci 当二元组解包，实际返回 (mean, lo, hi, n)，且测试桩写了二元组因此全程掩盖（测试形状 != 生产形状，同型第 4 次）；'
        '轮 3 读数里的 sections 是跨列累加值（每列都报 2088，真实 176/92）= 高报覆盖；'
        '轮 4 被全拒的列（lh_net_rate/lh_seat_asym）从输出里整个消失 => 与「从未被请求」不可区分（静默 skip）；'
        '轮 5 分母只报 sections，被拒截面不可见。'
        '另有 1 处真实逻辑缺陷由测试暴露并修掉：_index() 原先让 promotion_candidate 覆盖 research_feature，'
        '同名同时落两层时闸门静默返回「允许」；改为 research 后应用即胜出。'
        '单元测试 18 -> 23 项，23/23 全绿。'
    ),
    'verification': (
        'feature_tier.py tests => passed 21/21 后扩至 23/23；'
        'validate => ok: true，counts {research_feature: 8, promotion_candidate: 7}，pending_measurement 1；'
        'seed => 幂等（第二次 added=[] / refreshed=[]）；'
        'sync-evidence => jq_registered 7 / with_evidence 7 / conflicts []，并披露 7 条证据全为非结构化、第二台账 23 条、交集 0；'
        'promotion-view => evidence_recorded_structured 0 / unstructured 7 / not_recorded 0 / blocked_by_registry 0；'
        'fuxi_tier_gate_ab_20260924.py => all_checks_passed true（11 项）；'
        'measure => exit=0，产出 summary.json（LF、无 BOM、18,436 bytes、sha256 ce2ea2943e4c...）。'
    ),
    'artifacts': [
        'research/factor_alchemy/feature_tier.py',
        'data/factor_tier_registry.json',
        'scripts/fuxi_tier_gate_ab_20260924.py',
        'reports/feature_tier_20260924/summary.json',
        'reports/feature_tier_20260924/gate_ab/gate_ab.json',
        'reports/fuxi_feature_tier_20260924.md',
    ],
    'zero_adjudication': True,
    'default_on': False,
    'ab_evidence': (
        'A/B 反事实：同一生成器在「空登记库」与「真实登记库」两个输入下各生成 7/7 个文件，'
        '掩掉生成时嵌入的「生成时间:」一行后逐字节相同（A_equals_B_byte_for_byte=true）=> 本批闸门对生产产物零影响，'
        '空库等价于 legacy 行为。负向证据：被登记为研究特征的名字抛 NotPromotableError 且零文件写出，其余名字仍生成。'
        'fail-closed 证据：登记库是目录 / JSON 损坏时 7/7 全拒并抛 RuntimeError。'
        '探针证据：assert_promotable 被调用 7+7 次，闸门确实被走到而非死代码。'
    ),
    'known_limit': (
        '① 研究口径读数全部落在 2024-12-27~2026-09-09 研究已暴露窗口，是样本内读数，无样本外证据；'
        '② 固定基准组合只取现有复合信号 Top80 等权，未做基准敏感性扫描，读数绑定该基准；'
        '③ 本框架只能看到冻结构建器暴露的 6 列（fcst_yoy_mid/fcst_range_ratio/lh_net_rate/lh_seat_asym/margin_rz_buyrate/margin_rz_chg5），'
        '行业/大宗/基准三族在本框架读不到，not_readable_here 不是关于量级的陈述；'
        '④ lh_event_mask 的 not_yet_usable 只说明当前口径（单截面 20% 分位、最小 5 只）下无法叠加，不代表该族无效；'
        '⑤ 登记表本身没有防篡改保护，可被直接改写；闸门只在 load 失败时 fail-closed；'
        '⑥ 闸门只接在 JQ 生成端一处，其他可能的晋级入口未被穷举。'
    ),
    'next_step': (
        '转入 P3：修上游入口可交易性检查——已冻结目标的 13/13701 个入场事件落在无行情行日（其中 11 个落在停牌日），'
        '属上游入口可交易性检查缺口，应在上游修而不是在下游过滤。'
        '随后 P4 结算两项工程欠账（抬高 FUXI_COLLECT_CAL_END 默认值 + 补 R0 锁与 manifest 覆盖）、'
        'P5 把 R5 等日历延伸到第 26 个周频截面、P6 固化增量对照的冻结设计。'
        '同时保留最高优先级：组合层水平项本身的测量（IC=0 时 Top80 仍少赚 72~84bps/5 日）。'
    ),
    'rollback': (
        '删除 research/factor_alchemy/feature_tier.py、data/factor_tier_registry.json、'
        'scripts/fuxi_tier_gate_ab_20260924.py、reports/feature_tier_20260924/ 与报告，'
        '并从 scripts/gen_jq_s5_standalone.py::generate() 首行移除 _tier_gate 调用即可；'
        '本批未翻任何默认值、未接入生产裁决，无需回退数据。'
    ),
}


def main() -> int:
    raw = TARGET.read_bytes()
    assert raw[:3] != b'\xef\xbb\xbf', 'target unexpectedly carries a BOM'
    assert raw.count(b'\r\n') == 0, 'target unexpectedly carries CRLF'
    data = json.loads(raw.decode('utf-8'))
    assert isinstance(data, list)
    ids = [x.get('proposal_id') for x in data]
    if NEW_ID in ids:
        raise SystemExit('refusing to append: %s already present' % NEW_ID)
    before = len(data)
    last_before = ids[-1]

    data.append(ENTRY)
    text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    payload = text.encode('utf-8')          # 先编码，再落盘
    assert payload[:3] != b'\xef\xbb\xbf'
    assert payload.count(b'\r\n') == 0

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = TARGET.with_name(TARGET.name + '.bak_%s_p2' % stamp)
    shutil.copy2(TARGET, backup)

    with open(TARGET, 'wb') as fh:
        fh.write(payload)

    # 回读断言
    got = TARGET.read_bytes()
    assert got == payload, 'byte-level round-trip mismatch'
    assert got[:3] != b'\xef\xbb\xbf'
    assert got.count(b'\r\n') == 0
    again = json.loads(got.decode('utf-8'))
    assert isinstance(again, list) and len(again) == before + 1, 'entry count wrong'
    assert again[-1]['proposal_id'] == NEW_ID
    got_ids = [x.get('proposal_id') for x in again]
    assert got_ids[:-1] == ids, 'prior entries mutated'
    assert last_before in got_ids, 'previous last entry lost'
    assert again[-1]['status'] == 'landed'
    assert again[-1]['proposal_change_kind'] == 'research'
    assert again[-1]['zero_adjudication'] is True and again[-1]['default_on'] is False

    print('ok: %s -> %s entries (was %s, new last = %s)' % (before, len(again), last_before, NEW_ID))
    print('bytes %d (was %d)  backup %s' % (len(got), len(raw), backup.name))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
