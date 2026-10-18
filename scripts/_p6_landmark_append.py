# -*- coding: utf-8 -*-
"""一次性脚本：把 P6 批次声明**追加**进 scripts/version_bump.py 的 LANDMARK['2026-09-24']。

🔴 键必须是 `2026-09-24`：version_bump.load_proposals() 用正则从 proposal_id 取日期
   （P-20260924-016 → 2026-09-24），不读提案的 date 字段。新建 '2026-09-25' 键会变成孤儿。
   同日只做**追加**，不改已有文本。
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'scripts' / 'version_bump.py'
KEY = "    '2026-09-24':"
ANCHOR_TAIL = "详见 reports/fuxi_calendar_extension_20260925.md。',"

P6_TEXT = (
    "'P6：把「增量对照相对基准只允许改一个轴」从手写守卫变成可机检契约。'\n"
    "                  '交付 research/factor_alchemy/incremental_contract.py（纯函数、零副作用）："
    "键三分类（DESIGN / BOOKKEEPING / READING，未登记名字一律按 DESIGN 计）+ 轴声明 + 叶级上限 + 单调性。"
    "判定值 IDENTICAL / SINGLE_AXIS / MULTI_AXIS / AXIS_ITEM_BATCH / UNDECLARED_KEY；"
    "accepted 只认前两者，且**对变更集单调不增**（往一个已拒绝的对照里再加改动，永不变成可接受）。'\n"
    "                  '为什么要「轴」：加一个候选特征会同时动 features / dropped_by_coverage / "
    "dropped_by_redundancy / selection_rules 四个字段。把「按构造一起动」的字段声明成一个轴，"
    "「一个决策」才不会被计数成「四处改动」—— 否则合法的单维增量永远不可能通过。"
    "七个轴（feature_set / combiner / label / selection / cost / training / reference）"
    "覆盖生产脚本 DESIGN_FIELDS 的 18/18 个字段、无重复声明，且用 AST 读源码字面量 ⇒ "
    "契约声明的字段集与生产源码是**同一个字符串**，不是靠人抄一遍（单测钉死）。'\n"
    "                  '🔴 features[].coverage 是**读数**不是设计：数据一延伸它就会变。按名扁平比较会把它当设计字段 ⇒ "
    "每次补数据都把「同一次增量」读成一次设计变更（与 mem:AD5 同一根刺）。契约对特征表按「除读数键之外的全部键」投影，"
    "读数单独报出（反事实 C6：coverage 变 ⇒ IDENTICAL、touched=0、readings_changed 非空）。'\n"
    "                  '🔴 一处被测试逼出来的加强（留痕）：初版把「多出一个未登记的键」判成 SINGLE_AXIS 并接受。"
    "想清楚后改为 UNDECLARED_KEY 直接拒绝 —— 一个未写下来的键名（比如 `cfg`）可以在内部装下任意多个维度，"
    "而按「名字」看它只动一处，这是真实的夹带口子。合法化路径唯一：写进某个轴，或写进 BOOKKEEPING_KEYS / Spec "
    "证明它只是记账/读数；放宽须显式传 allow_new_design_keys=True（默认 False）。'\n"
    "                  '补了两条单测（写下来才合法 / 放宽须显式）与两个反事实（C7 拒绝 → C9 写下来之后接受），"
    "把改动两端都钉住。'\n"
    "                  '普查（14 份 reports/*/protocol.json）：带修订通道的 **1** 份（model_compare_20260924），"
    "**不带 13** 份（名单逐条列出）。⇒ 真实的比较风险不是「守卫太严」，而是 13/14 处**根本没有守卫**。"
    "旧守卫只有 0 / 非 0 两档：加一个特征在它眼里是「2 处改动」⇒ 必被拒（C3），"
    "于是要跑一次增量实验唯一的路是另写一个批次脚本，而那样就一份守卫都没有 —— 这就是 13/14 的成因。"
    "刻度对比：C3（合法单维增量）旧守卫按名计数 2 → 拒、契约按轴计数 1 → **接受**；"
    "C4（C3 ＋ Top80→Top100）旧守卫 4 → 拒、契约 2 轴 → **拒**；两者的**特征触碰数都是 1** "
    "⇒ 拒绝理由被正确归因到「多了一个轴」，不会被误报成「特征加太多」。"
    "九个反事实 5 接受 / 4 拒绝，逐个命中事前预期；单调性探针 5/5 被拒且接受从未翻回。'\n"
    "                  '历史修订声明独立复核：protocol_history/protocol_00.json → protocol.json 重推导 "
    "verdict=IDENTICAL、changed_design_fields=[] ⇒ 声明属实。并查出一处旧机制没保证的口径："
    "记录的 previous_protocol_sha256=da92a0eb… 是**文件**摘要（= protocol_00.json 的 sha256），"
    "而**内容**摘要（键序无关）是 c7c31a7b…；同内容重排键序 + 改缩进后文件摘要变 87d4e973…、内容摘要不变 "
    "⇒ 文件摘要不能当「基准是谁」的判据，契约补 protocol_sha256()（内容摘要）。'\n"
    "                  '本批**只读**：protocol_files_unchanged=true（前后逐文件 sha256 相同，单测钉死），"
    "model_compare 的 protocol.json 一字未改，历史协议与历史摘要一律不改写。无生产调用点、无开关、无数据文件被改。"
    "测试：契约单测 27/27、批次单测 20/20（合计 47），alchemy 全量 184 → **211/211**。"
    "限度：契约只判形状、不判该不该改；不阻止改两个维度，只让这种改法不能被报成单维增量；"
    "其余 13 份协议未装修订通道；七个轴只对 model_compare 声明；"
    "对 IC 缺口无直接贡献 —— 属「让解旱结论可信」的基础设施。"
    "详见 reports/fuxi_p6_incremental_freeze_20260925.md。',"
)


def main() -> int:
    raw = TARGET.read_bytes()
    src = raw.decode('utf-8')
    if 'P6：把「增量对照相对基准只允许改一个轴」' in src:
        raise SystemExit('refusing: P6 declaration already present in LANDMARK')

    i = src.index(KEY)
    j = src.index('\n}', i)
    entry_before = src[i:j]
    if ANCHOR_TAIL not in entry_before:
        raise SystemExit('anchor not found inside the 2026-09-24 entry')

    new_entry = entry_before.replace(
        ANCHOR_TAIL,
        ANCHOR_TAIL[:-1] + '\n                  ' + P6_TEXT,
        1)
    assert new_entry != entry_before
    assert new_entry.startswith(entry_before[:200]), 'entry head drifted'

    out = src[:i] + new_entry + src[j:]
    assert out.count(KEY) == src.count(KEY) == 1, 'LANDMARK key count changed'

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = TARGET.with_name(TARGET.name + '.bak_%s_p6' % stamp)
    shutil.copy2(TARGET, backup)

    payload = out.encode('utf-8')
    assert payload.count(b'\r\n') == raw.count(b'\r\n'), 'line endings changed'
    with open(TARGET, 'wb') as fh:
        fh.write(payload)

    got = TARGET.read_bytes()
    assert got == payload, 'byte-level round-trip mismatch'
    again = got.decode('utf-8')
    assert again.count(KEY) == 1
    assert 'P6：把「增量对照相对基准只允许改一个轴」' in again
    assert entry_before[:1500] in again, 'pre-existing text mutated'

    import importlib.util
    spec = importlib.util.spec_from_file_location('vb_probe_p6', TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = mod.LANDMARK['2026-09-24']
    assert 'P6：把「增量对照相对基准只允许改一个轴」' in text, 'LANDMARK value lacks the P6 text'
    assert 'P5：把交易日历延伸到 R5 判定端点' in text, 'P5 declaration lost'
    assert 'P4：结算两项工程欠账' in text, 'P4 declaration lost'
    assert 'P3 批次 v0.7.70' in text, 'P3 declaration lost'
    assert '2026-09-25' not in mod.LANDMARK, 'a stray LANDMARK key was created'

    print('ok: bytes %d -> %d (+%d)' % (len(raw), len(got), len(got) - len(raw)))
    print('    entry block %d -> %d chars' % (len(entry_before), len(new_entry)))
    print('    LANDMARK dates %d  rendered value len %d'
          % (len(mod.LANDMARK), len(mod.LANDMARK['2026-09-24'])))
    print('    backup %s' % backup.name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
