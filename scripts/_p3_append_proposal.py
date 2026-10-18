# -*- coding: utf-8 -*-
"""一次性脚本：向 data/stage1_optimization_proposals.json 追加 P-20260924-013。

纪律（与 _p1_/_p2_append_proposal.py 同构）：
  ① 先查占用，占用即拒绝；
  ② 先 encode('utf-8') 拿字节，再 open('wb') 落盘；
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
NEW_ID = 'P-20260924-013'

ENTRY = {
    'proposal_id': NEW_ID,
    'date': '2026-09-25',
    'created_at': '2026-09-25T00:40:00+08:00',
    'title': (
        'P3 修上游入口可交易性检查：13/13701 入场事件落在无行情行日复算成立（13/13 整日停牌），'
        '但准入级真读数是 30 个拒绝 = 13 个入场事件 + 17 个已在册名字被反复重提；'
        '交付零裁决观察者门（8 档判定表、默认 annotate、fail-closed）与四关验收，本批不接任何生产调用点'
    ),
    'category': '因子研究/信息面',
    'domains': ['T1闸门/裁决', 'T2数据口径', 'T3模块/通道'],
    'risk_level': 'low',
    'status': 'landed',
    'proposal_change_kind': 'research',
    'generator_hint': 'manual',
    'trigger': (
        '缺陷首次记录于 reports/fuxi_event_inventory_plan_20260924.md 第 4 节：'
        '冻结目标曾在不可交易日建仓 —— llm_family_1_1 有 11 个入场事件落在无行情行日、mem_0626 有 2 个，'
        '合计 13/13701；原文结论是「这不是市场事件，是入口可交易性检查缺口，应在上游修，不该在账本层吸收」。'
        '该条在 reports/fuxi_delist_channel_20260924.md 被列为「残留与欠账」第 4 项，'
        'P2 的 next_step 明确把修它定为下一批。'
    ),
    'current_state': (
        '在 P3 之前：选股只校验信号日的 close>0，目标自下一个交易日起生效，'
        '生效日能否成交从未被任何一处校验；全仓 8 个选股点全部没有该检查。'
        '后果不是「账本填错了价」，而是三件更隐蔽的事：'
        '① 该权重从未被部署，成为静默现金拖累且无任何决策可归因；'
        '② 下游账本把它吸收成一条 filled=0/reason=halted 的订单，这条订单既不影响状态机也不回灌目标，'
        '与「目标本来就没买」不可区分；'
        '③ 全窗证据库存把「目标想持有」与「市场拒绝成交」混在一张表里（170 例无行情行）。'
        '此外全仓没有任何现成的 13701 计数实现（grep 只命中记忆/报告/CHANGELOG 文本），该口径本身需要重建。'
    ),
    'proposed_change': (
        '① research/factor_alchemy/entry_tradability.py（新建）：上游入口可交易性门本体，'
        '纯函数库、无宿主、8 档穷尽互斥判定表 —— tradable / conflicting_price_and_halt / refused_halted_full_day / '
        'refused_halted_partial_day / refused_delisted / refused_not_a_trading_session / '
        'uncertifiable_status_table_absent / uncertifiable_no_price_row；准入集合 = {tradable}，其余一律不准入（fail-closed）。'
        '「缺证据」与「已证伪」在 by_category 里分开计数，最后两档不得被写成「停牌」。25 条单元测试。'
        '② 开关 env FUXI_ENTRY_TRADABILITY：annotate（默认，含 shadow/on/1 别名）/ enforce / off（含 legacy/0 别名），'
        '未知取值直接抛 ValueError 不静默降级；默认 annotate = 零裁决观察者（只报告不改目标），'
        'enforce 必须显式开启（会改变目标集合与权重，属经济口径变更，需翻默认四关 + 独立协议，本批不翻开）。'
        '③ 权重口径不重归一化：enforce 下权仍按 1/top_n，未部署部分留现金并逐条声明（undeployed_weight 逐臂报出）。'
        '④ scripts/fuxi_entry_tradability_20260924.py（新建）：重建入场事件口径 + 四关验收 + A/B 反事实 + 接线点清单，31 条测试。'
        '⑤ 不变量全部有断言：checked == admitted + refused_refuted + refused_uncertifiable + conflicting；'
        '各档返回同一套键（全零也要在）；enforce 下 admitted == output_names，annotate 下 checked == output_names 且逐 stamp 与入参逐键相同。'
    ),
    'implementation_scope': (
        'research/factor_alchemy/entry_tradability.py（新建，门本体 + 25 项测试）; '
        'scripts/fuxi_entry_tradability_20260924.py（新建，批次脚本 + 31 项测试含模块 25 项）; '
        'reports/entry_tradability_20260924/readings.json（四关读数全文）; '
        'reports/entry_tradability_20260924/entry_events_without_a_price_row.csv（13 例）; '
        'reports/entry_tradability_20260924/reject_ledger.csv（30 条准入级拒绝）; '
        'reports/fuxi_entry_tradability_20260924.md（报告）'
    ),
    'expected_impact': (
        '复算：入场事件总数 13701 逐臂复现（llm1 4771 / llm2 2114 / mem 2582 / r4 4234）；'
        '13/13701 成立，llm1 11 + mem 2，13/13 判 refused_halted_full_day，uncertifiable 计数为 0（无一例是「缺证据」）。'
        '历史对账：旧记录的「11 个落在停牌日」是当时停牌表末端 2026-04-30 造成的覆盖数字，不是判定分歧 —— '
        '002731.SZ 与 688121.SH（均 2026-05-06）当时落在表外，既不能证停牌也不能证成交；'
        '停牌表已扩到 2026-09-23，复算这 2 例同样是整日停牌 ⇒ 13 = 11 + 2。'
        '🔴 真读数修正：门按「每 stamp 每名字」计数，准入级拒绝 30 个（llm1 22 / mem 7 / r4 1，30/30 全是 refused_halted_full_day），'
        '其中 13 个是入场事件、另 17 个是 carry-over —— 该证券早已在册，却因仍在目标里被下一个 stamp 继续重提。'
        '17 条此前从未被任何一批报出（它们不违反「连续持有段首日」的定义，只在准入级显形）'
        '⇒ 门必须装在准入级，只修入场级会把这 17 条原样留下。'
        '两处此前静默的读数被逐条列出：4 个 stamp 的生效日落在交易日历之外（四臂各 1 个，均 2026-09-09，此前被探针静默跳过、13701 不含其持有日）；'
        '4 例被拒准入其本地行情序列已经结束（全在 llm1），与「那天刚好停牌」分开报。'
        '四关验收：'
        '关一 getter——逐事件直接查价格面板与门判定 disagreements_count = 0；'
        '对照臂把同一证券顺延到之后第一个有行情行的交易日（与主读数只差那一行行情），admitted 13 / refused 0 ⇒ 13/13 判 tradable，门不是返常量。'
        '关二 A/B 反事实——removed_admissions 30、added_admissions 0（四臂 added_pairs 全 0）；'
        '入账事件删除 11/0/2/0 = 13（= 全部被拒入场事件）；另有 6/0/1/0 = 7 个入场被推到首个可成交日，逐条列出（「删除」不是全部事实）；'
        '封存目标文件在 off 与 annotate 两档对四臂全部 identical=true（sha256 逐臂相同）。'
        '关三 默认不变量——default_resolves_to_documented_default true、四别名同解、off/legacy 同解、'
        'target_digest_by_mode 里 off = annotate = shadow 同摘要而 enforce 不同 ⇒ 观察者一个字节都没动；counterfactual_is_not_empty true。'
        '关四 端到端（账本世代 reports/ledger_advance4_20260924/D_reproduce_old_suspend）——'
        '11 例被下游吸收成 filled=0/reason=[halted] 的订单、2 例因该臂账本止于 2026-05-06（data_insufficient）根本没被触达、'
        '0 例在门判拒的日子里被成交。（注意：这个「11」与旧记录的「11 个落在停牌日」是两个不同的 11，纯属巧合，必须分开记。）'
        '接线点清单：逐条回读源码校验行号与模式（漂移即抛错、不用 grep 推断、「有没有门」按实现判），'
        '8 处选股点 has_entry_tradability_check 全为 false——1 处冻结构建器（被封存 protocol.json 与 code_hashes 逐字节钉死，改它会让封存产物对不上 ⇒ 需新协议）、'
        '3 处研究账簿构建器、2 处诊断 + 1 处中性化归因、1 处 P2 研究口径基准组合（改它们等于改那些批次的读数）。'
        '对干旱的实质帮助（诚实评估，限度优先）：13/13701 = 0.09%，单看数量不足以把实测 IC 从 +0.054 推到盈亏平衡 +0.108，'
        'P3 不生产任何因子，是「必要不充分」的评价器输入完整性修复，作用有三：'
        '① 把「市场拒绝成交」与「目标想持有」分开 —— 这正是干旱三因里「反馈噪音」的一个具体来源；'
        '② 暴露了准入级 30 ≠ 入场级 13 这个此前完全不可见的量级（17 个 carry-over），是「测试形状必须与生产一致」在计数口径上的再一次复现；'
        '③ 交付可复用的入口校验件，供 P1 大宗交易 / P2 研究特征库 / P5 新截面在进入组合层读数前先摘掉结构上不可能成交的样本。'
        '定位：P3 属地基 —— 它保证标尺刻度本身没错，地基不产生收益读数，但没有它后面所有读数都带 0.09% 不可归因噪音和一个看不见的 17 条口径洞。'
    ),
    'landing_note': (
        '零裁决、零影子开关、未翻任何默认值、未接任何生产调用点、不改变账本/S5/JQ 准入或任何封存产物的字节。'
        '门默认 annotate，enforce 需显式开启。'
        '本批暴露并留证多轮形状缺陷（均未用成功轮覆盖）：'
        '① 入场事件口径初稿按「每证券一段」算，与探针的「连续持有段首日」不一致，导致总数对不上 13701（测试形状 != 生产形状，同型第 5 次）；'
        '② 门面板一度被裁剪到「只含被问到的键」，使对照臂拿不到「之后第一个有行情行日」，'
        '四个行情序列已结束的案例被误放进对照臂的 admit 断言 ⇒ 改为分层：对照臂排除「自身行情序列已结束」的案例，另设独立读数分开报；'
        '③ 观察者档与 enforce 档的 admitted / output_names 两个口径一度混排，会把观察者读成闸门 ⇒ 加断言钉死；'
        '④ 各档返回的键集一度不一致（off 档少键），会让「没检查」被读成「检查了没发现」⇒ 统一键集；'
        '⑤ 对照臂最初允许「无更晚有价日」的案例存在，会把「feed 结束」读成「门拒了」⇒ 显式分出 feed_ended_so_no_control_possible。'
        '门本体 25/25 全绿；批次脚本 31/31 全绿。'
    ),
    'verification': (
        'research/factor_alchemy/entry_tradability.py => Ran 25 tests, OK；'
        'scripts/fuxi_entry_tradability_20260924.py --tests => Ran 31 tests, OK；'
        'scripts/fuxi_entry_tradability_20260924.py --run => exit 0，'
        '产出 readings.json（51,131 bytes，LF、无 BOM，含全部输入文件 sha256）、'
        'entry_events_without_a_price_row.csv（13 行 + 表头）、reject_ledger.csv（30 行 + 表头）；'
        '四关关键断言全部为等式断言（含 assert not mismatched / assert refused == 0 / 各档键集与逐 stamp 逐键相等）。'
    ),
    'artifacts': [
        'research/factor_alchemy/entry_tradability.py',
        'scripts/fuxi_entry_tradability_20260924.py',
        'reports/entry_tradability_20260924/readings.json',
        'reports/entry_tradability_20260924/entry_events_without_a_price_row.csv',
        'reports/entry_tradability_20260924/reject_ledger.csv',
        'reports/fuxi_entry_tradability_20260924.md',
    ],
    'zero_adjudication': True,
    'default_on': False,
    'ab_evidence': (
        'A/B 反事实：off 档与 annotate 档对四臂的封存目标文件渲染结果逐字节相同（sha256 逐臂相等，identical=true）；'
        'enforce 档 removed_admissions=30、added_admissions=0，入账事件删除 13、顺延 7（逐条列出）。'
        '对照臂（关一）：把同一证券顺延到之后第一个有行情行的交易日，13/13 判 tradable、0 判拒 ⇒ 门在读数据不是返常量。'
        '默认不变量（关三）：off = annotate = shadow 目标摘要相同，enforce 不同摘要。'
        '端到端（关四）：11 例在下游被吸收成 filled=0 的订单、2 例未被触达、0 例在门判拒的日子成交。'
    ),
    'known_limit': (
        '① 本批不接任何生产调用点，8 个选股点全部仍无该检查 —— 「缺口已可判定、可计数、可逐条列举」≠「已经修好建仓」；'
        '② 门只读本地封存数据（daily_prices / suspend_d / stock_list / trade_calendar），停牌表末端 2026-09-23，'
        '生效日超出该末端的只能判 uncertifiable_status_table_absent，不能判停牌；'
        '③ 门只回答「能不能成交」，不推断成交价、不填补、不做零收益假设；被拒事件不带任何「该不该买」的判断；'
        '④ enforce 档会改变目标集合与权重，本批刻意不翻开，其经济后果尚未测；'
        '⑤ 窗口 2024-12-27~2026-09-09 全为研究已暴露窗口，是样本内读数，无样本外证据；R5 未知样本仍为 0；'
        '⑥ 只枚举了「有选股动作」的 8 处调用点，其他可能的入口未被穷举；'
        '⑦ 0.09% 量级说明它不可能单独解释 IC 缺口，不得把本批当成解旱成果。'
    ),
    'next_step': (
        '转入 P4：结算两项工程欠账 —— ① 抬高 FUXI_COLLECT_CAL_END 默认值（独立批次 + 对照）；'
        '② 补 R0 锁与 manifest 覆盖：data_fingerprints 不含 suspend_d.csv / delist_dates.csv，code_fingerprints 不含当批脚本。'
        '随后 P5 把 R5 等日历延伸到第 26 个周频截面（不改统计要求，decisive_endpoint 保持 null），'
        'P6 固化增量对照的冻结设计（不同时改多个维度）。'
        '同时保留最高优先级：组合层水平项本身的测量（IC=0 时 Top80 仍少赚 72~84bps/5 日）。'
    ),
    'rollback': (
        '删除 research/factor_alchemy/entry_tradability.py、scripts/fuxi_entry_tradability_20260924.py、'
        'reports/entry_tradability_20260924/ 与报告即可；'
        '本批未接任何生产调用点、未翻任何默认值、未改任何封存产物，无需回退数据。'
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
    backup = TARGET.with_name(TARGET.name + '.bak_%s_p3' % stamp)
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
