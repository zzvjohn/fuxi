# -*- coding: utf-8 -*-
"""一次性脚本：把 P5 批次声明**追加**进 scripts/version_bump.py 的 LANDMARK['2026-09-24']。

🔴 键必须是 `2026-09-24`：version_bump.load_proposals() 用正则从 proposal_id 取日期
   （P-20260924-015 → 2026-09-24），不读提案的 date 字段。新建 '2026-09-25' 键会变成孤儿。
   同日只做**追加**，不改已有文本。
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'scripts' / 'version_bump.py'
KEY = "    '2026-09-24':"
ANCHOR_TAIL = "详见 reports/fuxi_p4_engineering_debts_20260925.md。',"

P5_TEXT = (
    "'P5：把交易日历延伸到 R5 判定端点（只延伸覆盖，不改统计要求）。'\n"
    "                  '随包日历 2026-12-31 → 2027-12-31：在表头后纯插入 365 行（厂商 trade_cal / SSE，"
    "20270101..20271231），行尾 CRLF、格式与既有行逐字同形、维持文件自身的全局降序；"
    "字节 55113 → 64238（+9125 = 声明的插入块字节数）。"
    "🔴 这是本会话唯一一次写数据文件，风险控制为：备份 + 三判字节洁净（前缀相同 ∧ 后缀相同 ∧ 插入块=声明的行）"
    "+ 前后 sha256 + 双文件事务 + 失败回滚 + 可精确还原（备份 42f39551… 即原文）。'\n"
    "                  '数据源替换的三条 fail-closed 对账断言：① 包含（随包 ⊆ 厂商，差集 0）；"
    "② 单调（开市标记未被翻转，冲突 0）；③ 新增可枚举**且形状完好**。"
    "第 ③ 条第一版写成「新增是一段连续区间」，结果**恒真**（additions 按定义就是 vendor − packaged），什么都抓不到 ⇒ "
    "改为验形状（逐月无空洞、相邻会话间隔 ≤ 12 天，实测 12 个月、空洞无、最大间隔 1 天）；"
    "只验「差集非空」抓不到「厂商只给出它确定的日子」，那会让日历被打穿、周频截面数虚高、判定端点日期算错。'\n"
    "                  '原子同进：collect_daily_full.CAL_END_DEFAULT 20261231 → 20271231 与日历在同一事务内改 —— "
    "日历一延伸，P4 刚装的守卫会立刻 exit 3 拒绝补采；若不修，一次正确的日历延伸会立刻把补采打回静默截断。"
    "独立复核：default_is_in_sync_with_the_packaged_calendar() => (True, 一致 (20271231))。'\n"
    "                  '结果：未来周频截面 15 → 67；判定 26 段 signal null → **2027-03-19**、matured null → "
    "**2027-03-29**、calendar_covers false → true；描述性 8 段（2026-11-13 / 2026-11-23）与段数要求逐字不变。"
    "🔴 这是整个干旱定位链条第一次有了**可闭合的时间轴** —— 此前判定端点在日历上是 null，"
    "意味着最终裁决无法到期。判定仍为 not_evaluated、未知样本仍为 0（本批不读任何未来数据）。'\n"
    "                  '🔴 修掉一个影子结论：R5 的 --status 原先把 protocol 里冻结的 descriptive_checkpoint / "
    "decisive_endpoint 直接当作「日历覆盖」输出，日历延伸后它们变成预注册当时的**快照**，"
    "继续当覆盖报会让工具永远说「判定端点不可覆盖」。改为分两块报：as_recorded_at_preregistration（历史快照）"
    "+ coverable_with_the_current_calendar（按当前日历重算，段数要求仍从冻结件读入、从不重算、从不缩小），"
    "保留原扁平键并显式标注 checkpoint_keys_are_preregistration_snapshots=true；"
    "protocol.json 一字未改（freeze 守卫本就拒绝覆盖），「日历已延伸」写进 access_log.jsonl（append-only，序号 2）。"
    "机检两方向必须同时成立：统计要求逐字不变（8/26）∧ 覆盖确实变了（15→67）∧ 影子结论确实存在过（signal null→非 null）。'\n"
    "                  '另一条被更正的读法幻觉：此前把「该文件混入 256 行空行」当成事实，实际是 str.splitlines() "
    "在 \\r 处多切一刀 —— 真实空行 0 行，是**头 256 行用了 \\r\\r\\n 行尾**（Windows 文本模式把 \\r\\n 里的 \\n 又翻译一次）。"
    "已按字节写进 calendar_facts 并加单测钉死；文件里中段的重复表头（真缺陷）只报告、不改写。'\n"
    "                  '零回归（实跑对照）：P3 批次重跑一次与封存读数逐字段 diff —— 仅 2 处差异（generated_at 与 "
    "hashes[data/raw/trade_calendar.csv]），**实质读数差异 0**，两个 CSV 逐字节相同；P3 封存读数随后已还原"
    "（sha 429dab22…，仍记录 42f39551… 与原始时间戳）。为什么可预期零回归：新增日期全部 ≥ 20270101，"
    "而现存所有读数窗口末端 ≤ 2026-09-23 ⇒ 任何以 2026-12-31 为界的窗口在延伸前后逐字相同（结构性，非巧合）。"
    "各历史批次里记录的旧日历哈希一律**不改写**（改写即伪造历史）。"
    "测试：calendar_extension 17/17、r5_preregistration 5/5、p4_debts 9/9、entry_tradability 31/31、"
    "event_inventory 5/5、event_adjudication 12/12、missing_event_attribution 10/10、model_compare 8/8、"
    "alchemy 全量 184/184 ⇒ 合计 281 条全绿。"
    "限度：2027 节假日安排是厂商**规则推算**（正式排期通常 2026 年底才公布），若微调只会让端点**日期**平移一两天，"
    "段数要求 26 不变；厂商日历上限就是 2027-12-31（请求 2028 返回同一份）；本批对 IC 缺口无直接帮助，"
    "属「解旱的先决条件」。详见 reports/fuxi_calendar_extension_20260925.md。',"
)


def main() -> int:
    raw = TARGET.read_bytes()
    src = raw.decode('utf-8')
    if 'P5：把交易日历延伸到 R5 判定端点' in src:
        raise SystemExit('refusing: P5 declaration already present in LANDMARK')

    i = src.index(KEY)
    j = src.index('\n}', i)
    entry_before = src[i:j]
    if ANCHOR_TAIL not in entry_before:
        raise SystemExit('anchor not found inside the 2026-09-24 entry')

    new_entry = entry_before.replace(
        ANCHOR_TAIL,
        ANCHOR_TAIL[:-1] + '\n                  ' + P5_TEXT,
        1)
    assert new_entry != entry_before
    assert new_entry.startswith(entry_before[:200]), 'entry head drifted'

    out = src[:i] + new_entry + src[j:]
    assert out.count(KEY) == src.count(KEY) == 1, 'LANDMARK key count changed'

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = TARGET.with_name(TARGET.name + '.bak_%s_p5' % stamp)
    shutil.copy2(TARGET, backup)

    payload = out.encode('utf-8')
    assert payload.count(b'\r\n') == raw.count(b'\r\n'), 'line endings changed'
    with open(TARGET, 'wb') as fh:
        fh.write(payload)

    got = TARGET.read_bytes()
    assert got == payload, 'byte-level round-trip mismatch'
    again = got.decode('utf-8')
    assert again.count(KEY) == 1
    assert 'P5：把交易日历延伸到 R5 判定端点' in again
    assert entry_before[:1500] in again, 'pre-existing text mutated'

    import importlib.util
    spec = importlib.util.spec_from_file_location('vb_probe_p5', TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = mod.LANDMARK['2026-09-24']
    assert 'P5：把交易日历延伸到 R5 判定端点' in text, 'LANDMARK value lacks the P5 text'
    assert 'P4：结算两项工程欠账' in text, 'P4 declaration lost'
    assert '2026-09-25' not in mod.LANDMARK, 'a stray LANDMARK key was created'

    print('ok: bytes %d -> %d (+%d)' % (len(raw), len(got), len(got) - len(raw)))
    print('    entry block %d -> %d chars' % (len(entry_before), len(new_entry)))
    print('    LANDMARK dates %d  rendered value len %d'
          % (len(mod.LANDMARK), len(mod.LANDMARK['2026-09-24'])))
    print('    backup %s' % backup.name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
