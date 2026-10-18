# -*- coding: utf-8 -*-
"""一次性脚本：向 data/stage1_optimization_proposals.json 追加 P-20260924-015（P5）。"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'data' / 'stage1_optimization_proposals.json'
NEW_ID = 'P-20260924-015'

ENTRY = {
    'proposal_id': NEW_ID,
    'date': '2026-09-25',
    'created_at': '2026-09-25T01:00:00+08:00',
    'title': (
        'P5 延伸交易日历到 R5 判定端点：随包日历 2026-12-31→2027-12-31（纯插入 365 行、字节级可证「只是插入」、'
        '三条对账断言）+ 采集默认值原子同进；判定端点首次可覆盖（26 段 signal 2027-03-19 / matured 2027-03-29），'
        '统计要求逐字未改；并修掉 R5 --status 把冻结快照当当前覆盖报出的影子结论'
    ),
    'category': '因子研究/数据基座',
    'domains': ['T2数据口径', 'T1闸门/裁决', 'T3模块/通道'],
    'risk_level': 'medium',
    'status': 'landed',
    'proposal_change_kind': 'engineering',
    'generator_hint': 'manual',
    'trigger': (
        'R5 预注册把统计要求先固定为描述性 8 段 / 判定 26 段，'
        '而随包交易日历末端 2026-12-31 只覆盖 15 个未来周频截面 ⇒ decisive_endpoint.signal 为 null、'
        'calendar_covers=false。R5 自己的规矩是「要求永不缩小以迁就日历」'
        '（单测 test_decisive_endpoint_is_not_shrunk_to_fit_the_calendar 钉死）⇒ 唯一出路是延伸日历。'
        '延伸的合法性：交易日历是已公布的排期，不是未来数据，读它不消耗未知样本。'
    ),
    'current_state': (
        'P5 之前：日历只到 2026-12-31（2191 行 / 1454 开市日），判定端点日期无法计算；'
        '且日历是**全局降序**（20261231 → 20210101），任何延伸都必须维持该约定，否则「最后一行 = 最新交易日」失效。'
        '另有一个此前未被发现的读取幻觉：文件头 256 行的行尾是 \\r\\r\\n（Windows 文本模式把 \\r\\n 里的 \\n 又翻译一次），'
        '用 str.splitlines() 看会把多余的 \\r 读成 256 行空行 —— 空行根本不存在。'
    ),
    'proposed_change': (
        '① data/raw/trade_calendar.csv：在表头后插入 365 行（20270101..20271231，厂商 trade_cal / SSE），'
        '行尾 CRLF、格式与既有行逐字同形、维持全局降序；字节 55113 → 64238（+9125 = 声明的插入块字节数）。'
        '② scripts/fuxi_calendar_extension_20260925.py（新建）：--tests / --plan（只读预演）/ --run（落盘 + 回滚保护）/ '
        '--status / --status-audit；17 条单测。核心纯函数：calendar_facts（按字节审计行尾与非数据行）、'
        'canonical_rows、reconcile（三条断言）、addition_shape（新增块的形状）、insert_after_header + '
        'insert_is_byte_clean（前缀/后缀/插入块三判）、move_cal_end_default、snapshot_vs_current。'
        '③ 🔴 三条对账断言（fail-closed）：包含（随包 ⊆ 厂商）/ 单调（开市标记不被翻转）/ '
        '新增可枚举**且形状完好**（逐月无空洞、相邻会话间隔 ≤ 12 天）。'
        '只验「差集非空」抓不到「厂商只给它确定的日子」，那会让日历被打穿、周频截面数虚高 ⇒ 必须验形状。'
        '④ 原子同进：collect_daily_full.py 的 CAL_END_DEFAULT 20261231 → 20271231，与日历在**同一事务**内改，'
        '任一步断言失败即从备份双文件回滚（本批未触发）。P4 装的守卫（日历延伸而默认值没跟上即 exit 3）'
        '正是这次同进的验收条件。'
        '⑤ 🔴 影子结论修复：R5 的 --status 原先把 protocol 里冻结的 descriptive_checkpoint / decisive_endpoint '
        '直接当作「日历覆盖」输出 —— 日历延伸后这两个字段变成预注册当时的快照，'
        '继续当覆盖报会让工具**永远**说「判定端点不可覆盖」。改为分两块报：'
        'as_recorded_at_preregistration（历史快照）+ coverable_with_the_current_calendar'
        '（按当前日历重算，段数要求仍从冻结件读入、从不重算、从不缩小）；'
        '保留原扁平键并显式标注 checkpoint_keys_are_preregistration_snapshots=true。'
        'protocol.json 一字未改（freeze 守卫本就拒绝覆盖）；「日历已延伸」写进 access_log.jsonl（append-only，序号 2）。'
    ),
    'implementation_scope': (
        'data/raw/trade_calendar.csv（纯插入 365 行，备份于 reports/calendar_extension_20260925/trade_calendar.csv.before）; '
        'strategies/final/collect_daily_full.py（CAL_END_DEFAULT 一处）; '
        'scripts/fuxi_r5_preregistration_20260924.py（status() 拆分快照与当前覆盖）; '
        'scripts/fuxi_calendar_extension_20260925.py（新建 + 17 条单测）; '
        'reports/calendar_extension_20260925/readings.json（6372 bytes）; '
        'reports/calendar_extension_20260925/status_tool_audit.json; '
        'reports/calendar_extension_20260925/vendor_2027_rows.csv（365 行 + 表头）; '
        'reports/calendar_extension_20260925/trade_calendar.csv.before（备份）; '
        'reports/r5_preregistration_20260924/access_log.jsonl（追加 1 条，序号 2）; '
        'reports/fuxi_calendar_extension_20260925.md（报告）'
    ),
    'expected_impact': (
        '结果：未来周频截面 15 → 67；判定 26 段 signal null → **2027-03-19**、matured null → **2027-03-29**、'
        'calendar_covers false → **true**；描述性 8 段（2026-11-13 / 2026-11-23）与段数要求逐字不变。'
        '判定仍为 not_evaluated、未知样本有效计数仍为 0（本批不读任何未来数据）。'
        '零回归（实跑对照）：P3 批次重跑一次与封存读数逐字段 diff —— 仅 2 处差异（generated_at 与'
        'hashes[data/raw/trade_calendar.csv]），**实质读数差异 0**；两个 CSV 逐字节相同；'
        'P3 封存读数随后已还原（sha 429dab22…，仍记录 42f39551… 与原始时间戳）。'
        '为什么可预期零回归：新增日期全部 ≥ 20270101，而现存所有读数窗口末端 ≤ 2026-09-23 '
        '⇒ 任何以 2026-12-31 为界的窗口在延伸前后逐字相同（结构性，非巧合）。'
        '测试总览（延伸后重跑）：p4_debts 9/9、calendar_extension 17/17、r5_preregistration 5/5、'
        'entry_tradability 31/31、event_inventory 5/5、event_adjudication 12/12、'
        'missing_event_attribution 10/10、model_compare 8/8、alchemy 全量 184/184 ⇒ 合计 281 条全绿。'
        '对干旱的实质帮助（诚实评估）：本批**不产生任何因子、不改任何判定基线**，'
        '它解决的是「什么时候能有定论」这个元问题 —— R5 的判定端点此前是 null，'
        '意味着干旱的最终裁决**在日历上无法到期**；现在到期日可算（2027-03-29），'
        '整个干旱定位链条第一次有了闭合的时间轴。属「解旱的先决条件」而非解旱手段。'
    ),
    'landing_note': (
        '🔴 这是本会话唯一一次**写数据文件**（trade_calendar.csv）。风险控制：备份 + 三判字节洁净 + '
        '前后 sha256 + 双文件事务 + 失败回滚 + 可精确还原。'
        '未翻开任何开关、未改账本/S5/JQ 准入、未改任何封存产物的实质读数。'
        '本批暴露并留证两类读法缺陷（均未用成功轮覆盖）：'
        '① 我此前把「256 行空行」当成文件事实，实际是 str.splitlines() 在 \\r 处多切一刀造成的幻觉 —— '
        '真实空行 0 行，是头 256 行用了 \\r\\r\\n 行尾（已按字节写进 calendar_facts 并加 3 条单测钉死）；'
        '② 断言 ③ 第一版写成「新增是一段连续区间」，结果**恒真**（因为 additions 按定义就是 vendor − packaged），'
        '什么也抓不到 ⇒ 改为验新增块的形状（逐月覆盖 + 相邻会话间隔上限），并用「整月空洞」「异常间隔」两个反例钉死。'
        '另有一处测试自我指涉：检查源码里有没有 ``DECISIVE_SECTIONS = `` 这个字面量时，'
        '测试自己就含该字面量 ⇒ 检查串改为拼接而成。'
    ),
    'verification': (
        'python scripts/fuxi_calendar_extension_20260925.py --tests => Ran 17 tests, OK；'
        '--plan => 三条断言全 True（包含缺 0 天 / 开市标记冲突 0 天 / 新增 365 行、12 个月无空洞、最大间隔 1 天）；'
        '--run => exit 0，字节洁净三判全 True、行尾保留 True、回滚未触发；'
        '--status-audit => 要求不变=True、覆盖变了=True、影子结论存在过=True、protocol 未改写、'
        '判定与未知样本未变 ⇒ 产出 status_tool_audit.json；'
        'python scripts/fuxi_r5_preregistration_20260924.py --status => coverable_with_the_current_calendar'
        '.decisive_endpoint = {sections 26, signal 2027-03-19, matured_on 2027-03-29, calendar_covers true}；'
        'python scripts/fuxi_r5_preregistration_20260924.py --tests => Ran 5 tests, OK；'
        'collect_daily_full.default_is_in_sync_with_the_packaged_calendar() => (True, 一致 (20271231))。'
    ),
    'artifacts': [
        'data/raw/trade_calendar.csv',
        'strategies/final/collect_daily_full.py',
        'scripts/fuxi_r5_preregistration_20260924.py',
        'scripts/fuxi_calendar_extension_20260925.py',
        'reports/calendar_extension_20260925/readings.json',
        'reports/calendar_extension_20260925/status_tool_audit.json',
        'reports/calendar_extension_20260925/vendor_2027_rows.csv',
        'reports/fuxi_calendar_extension_20260925.md',
    ],
    'zero_adjudication': True,
    'default_on': False,
    'ab_evidence': (
        'A/B 对照：同一套判据下延伸前后的日历视野（15 → 67 段）与判定端点（null → 2027-03-19），'
        '同时证明统计要求逐字不变（snapshot_vs_current 的两个方向断言：要求必须不变 ∧ 覆盖必须变）。'
        '零回归 A/B：P3 批次重跑 vs 封存读数，实质差异 0 处、CSV 逐字节相同。'
        '回滚 A/B：备份 sha256 = 原文 sha256（42f39551…），可精确还原。'
    ),
    'known_limit': (
        '① 2027 的节假日安排是厂商**规则推算**，正式排期通常 2026 年底才公布 —— '
        '若微调，判定端点只会**日期**平移一两天，段数要求（26）不变；'
        '② 厂商日历可用上限就是 2027-12-31（请求 2028 返回同一份），本批没有延到 2028；'
        '③ 本批只把「什么时候到期」算出来，判定仍未评估、未知样本仍为 0，没有任何样本外证据；'
        '④ 文件里中段的**重复表头**（第 256 行 body）与头 256 行的 \\r\\r\\n 行尾**未被修复**，只被报告；'
        '⑤ 零回归的实跑对照只做了 P3 一个批次，其余靠「窗口末端 ≤ 2026-12-31」的结构性论证 + 各自测试套件；'
        '⑥ 各历史批次产物里记录的 42f39551… 一律**不改写**（改写即伪造历史），'
        '因此仓库里会长期存在指向旧日历的哈希，需要读者知道它是历史值；'
        '⑦ 本批对 IC 缺口没有直接贡献，不得当成解旱成果。'
    ),
    'next_step': (
        '转入 P6：固化增量对照的冻结设计（不同时改多个维度）。'
        '起点已经具备：reports/model_compare_20260924/protocol.json 的 revisions 机制'
        '（design_fields_changed: []）是这条纪律的雏形，本批已独立复核该声明属实；'
        'P6 要把它从「靠自觉」变成「靠契约」—— 一个增量实验相对其基准只允许改一个维度，且 features 维度一次只允许增删一个。'
        '同时保留最高优先级：组合层水平项本身的测量（IC=0 时 Top80 仍少赚 72~84bps/5 日）。'
    ),
    'rollback': (
        'data/raw/trade_calendar.csv <- reports/calendar_extension_20260925/trade_calendar.csv.before（逐字节还原）；'
        'collect_daily_full.py 的 CAL_END_DEFAULT <- 20261231；'
        'fuxi_r5_preregistration_20260924.status() 可回退为直接报 protocol 字段（但会恢复影子结论）；'
        '删除 scripts/fuxi_calendar_extension_20260925.py 与 reports/calendar_extension_20260925/ 即可。'
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
    payload = text.encode('utf-8')
    assert payload[:3] != b'\xef\xbb\xbf'
    assert payload.count(b'\r\n') == 0

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = TARGET.with_name(TARGET.name + '.bak_%s_p5' % stamp)
    shutil.copy2(TARGET, backup)

    with open(TARGET, 'wb') as fh:
        fh.write(payload)

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
    assert again[-1]['proposal_change_kind'] == 'engineering'
    assert again[-1]['zero_adjudication'] is True and again[-1]['default_on'] is False

    print('ok: %s -> %s entries (was %s, new last = %s)'
          % (before, len(again), last_before, NEW_ID))
    print('bytes %d (was %d)  backup %s' % (len(got), len(raw), backup.name))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
