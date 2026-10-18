# -*- coding: utf-8 -*-
"""一次性脚本：把 P4 批次声明**追加**进 scripts/version_bump.py 的 LANDMARK['2026-09-24']。

🔴 纪律（P3 批次踩过的坑）：
   version_bump.load_proposals() **不读提案的 date 字段**，而是用正则从 proposal_id
   （P-YYYYMMDD-NNN）取日期 —— 所以 LANDMARK 的键必须是 `2026-09-24`（P-20260924-014 的日期），
   新建一个 `2026-09-25` 键会变成**孤儿**（--bump 永远不引用它，版本正文里看不到本批）。
   P4 的 proposal_id 仍是 P-20260924-014 ⇒ 同样是 `2026-09-24`，且同日只做**追加**，不改已有文本。

纪律（与 _p1_/_p2_/_p3 同构）：先查占用 → 备份 → 先 encode 再 open('wb') → 字节级回读断言。
"""
from __future__ import annotations

import io
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'scripts' / 'version_bump.py'
KEY = "    '2026-09-24':"
ANCHOR_TAIL = "详见 reports/fuxi_entry_tradability_20260924.md。',"

P4_TEXT = (
    "'P4：结算两项工程欠账，并在结算中暴露并修掉三个此前不可见的静默缺陷。'\n"
    "                  '① FUXI_COLLECT_CAL_END 默认值由字面量 20260430 抬到 CAL_END_DEFAULT=20261231"
    "（= 随包交易日历末端）；对照：旧默认落后 5 个参考物（packaged_trade_calendar/prices/basic/suspend/limit）"
    "⇒ 拒跑，新默认落后 0 ⇒ 通过；并新增两条 fail-closed 守卫 —— 参考物晚于 cal_end 即 exit 2"
    "（把「静默截断」变成「拒绝运行」）、日历延伸而默认值没跟上即 exit 3。'\n"
    "                  '② 缺陷 A（此前完全不可见）：stk_limit.csv 的 trade_date 混着紧凑写法 8433700 行与"
    "短横线写法 352991 行（首行第 8433702 行），不规范化会重复追加 46 天 / 352991 行 —— "
    "旧默认把这些日期挡在采集窗口外，所以「判成缺失」没有任何后果；抬到 20261231 后它们落进窗口，"
    "日历里是 20260528 而集合里是 2026-05-28，两边永远对不上 ⇒ 属「修一个 bug 才会显形的 bug」；"
    "写入口（append_df）改为只允许一种形式、不可解析即抛错（不填默认值、不静默跳过）。'\n"
    "                  '③ 缺陷 B：同一句 pd.read_csv(usecols=[trade_date]) 在 pandas 3.0.3（默认 env）"
    "抛 IndexError、在生产解释器（pandas 2.x）只发一条 DtypeWarning —— 一个崩、一个静默误判；"
    "改 dtype=str, low_memory=False + 逐值规范化。'\n"
    "                  '④ 缺陷 C（本批新发现，原缺陷的镜像）：交易日历本身延伸进未来 —— 1454 个开市日 = "
    "已发生 1390 + 尚未发生 64（20260928..20261231）；不区分则每轮空转 256 次 API、永久误报缺 64 天、"
    "last_update.txt 的 last_trade_date 被写成未来日期 20261231。新增 not_yet_available_sessions(all_days, today) "
    "拆口径且两个数都进读数（today 显式传入，可用 FUXI_COLLECT_TODAY 覆盖）。"
    "老 bug 是「把过去静默截断」，这一条是「把未来当成缺失」，同一失败模式、方向相反、同样静默。'\n"
    "                  '⑤ manifest 加锁：_DATA_SENTINELS 5→9 键（新增 suspend_d / delist_dates / stock_list / "
    "trade_calendar），code_fingerprints 由 9 个固定模块扩展为可声明追加（extra_code / extra_data —— "
    "旧结构下批次自己的脚本**结构上无法登记**，那不是偶发遗漏），键统一 posix（原来在 Windows 产出反斜杠），"
    "声明即须存在（不存在即抛 ValueError，不允许记一行 exists=False），fingerprint_coverage 把声明分三类 "
    "standard / extra_*_newly_added / extra_*_already_standard（第一版把「重复声明」误算成新增、"
    "修正后又误从标准集剔除，两个方向都错，三分类才自洽）。读数：standard_data 9 键、keys_added_count 4、"
    "posix 键=True、声明缺失被拒=True、复用 run_id 触发 RunReuseError（R0 契约本来就该如此）。'\n"
    "                  '本批未调用任何厂商 API、未改写任何数据文件（352991 行短横线日期今天仍在盘上）、"
    "未翻任何开关、未改账本/S5/JQ 准入或任何封存产物字节；对 IC 缺口无直接帮助，属「解旱的先决条件」"
    "（盈亏平衡线 -0.00843 + 0.0915×IC 的输入就是这四个采集文件，采集被静默截断则所有「某因子无效」的结论"
    "都可能只是数据没到）；它交付的「参考物是否落后」判据正是 P5 延伸日历的验收条件。"
    "测试 run_contract 9/9、fuxi_p4_debts 9/9、alchemy 全量 184/184 全绿。"
    "详见 reports/fuxi_p4_engineering_debts_20260925.md。',"
)


def main() -> int:
    raw = TARGET.read_bytes()
    src = raw.decode('utf-8')
    if 'P-20260924-014' in src:
        raise SystemExit('refusing: P4 declaration already present in LANDMARK')

    i = src.index(KEY)
    j = src.index('\n}', i)
    entry_before = src[i:j]
    if ANCHOR_TAIL not in entry_before:
        raise SystemExit('anchor not found inside the 2026-09-24 entry')

    # 只做追加：把锚点尾部替换成「锚点内容去掉收尾的逗号 + P4 文本」
    new_entry = entry_before.replace(
        ANCHOR_TAIL,
        ANCHOR_TAIL[:-1] + '\n                  ' + P4_TEXT,
        1)
    assert new_entry != entry_before
    assert new_entry.startswith(entry_before[:200]), 'entry head drifted'
    assert "'2026-09-24'" in new_entry

    out = src[:i] + new_entry + src[j:]
    assert out.count(KEY) == src.count(KEY) == 1, 'LANDMARK key count changed'

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = TARGET.with_name(TARGET.name + '.bak_%s_p4' % stamp)
    shutil.copy2(TARGET, backup)

    payload = out.encode('utf-8')          # 先编码，再落盘
    assert payload.count(b'\r\n') == raw.count(b'\r\n'), 'line endings changed'
    with open(TARGET, 'wb') as fh:
        fh.write(payload)

    got = TARGET.read_bytes()
    assert got == payload, 'byte-level round-trip mismatch'
    again = got.decode('utf-8')
    assert again.count(KEY) == 1
    assert 'P4：结算两项工程欠账' in again
    # 追加性质：新正文必须能由旧正文去掉插入块后逐字还原
    assert again.replace('\n                  ' + P4_TEXT, '') == src.replace(ANCHOR_TAIL, ANCHOR_TAIL), \
        'not a pure append'
    assert entry_before[:1500] in again, 'pre-existing text mutated'

    # 编译校验：LANDMARK 常量必须仍可被 import 并含本批声明
    import importlib.util
    spec = importlib.util.spec_from_file_location('vb_probe', TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = mod.LANDMARK['2026-09-24']
    assert 'P4：结算两项工程欠账' in text, 'LANDMARK value does not carry the P4 text'
    assert '2026-09-25' not in mod.LANDMARK, 'a stray LANDMARK key was created'

    print('ok: bytes %d -> %d (+%d)' % (len(raw), len(got), len(got) - len(raw)))
    print('    entry block %d -> %d chars' % (len(entry_before), len(new_entry)))
    print('    LANDMARK dates %d  rendered value len %d'
          % (len(mod.LANDMARK), len(mod.LANDMARK['2026-09-24'])))
    print('    backup %s' % backup.name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
