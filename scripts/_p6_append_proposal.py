# -*- coding: utf-8 -*-
"""一次性脚本：向 data/stage1_optimization_proposals.json 追加 P-20260924-016（P6）。"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'data' / 'stage1_optimization_proposals.json'
NEW_ID = 'P-20260924-016'

ENTRY = {
    'proposal_id': NEW_ID,
    'date': '2026-09-25',
    'created_at': '2026-09-25T01:15:00+08:00',
    'title': (
        'P6 增量对照的冻结设计契约：把「相对基准只允许改一个轴」从手写守卫变成可机检纯函数'
        '（键三分类 + 七轴声明 + 叶级上限 + accepted 对变更集单调不增）；普查出 14 份协议里只有 1 份'
        '有修订通道、13 份一个都没有；并发现旧机制记的是文件摘要而非内容摘要'
    ),
    'category': '因子研究/方法与纪律',
    'domains': ['T1闸门/裁决', 'T3模块/通道', 'T2数据口径'],
    'risk_level': 'low',
    'status': 'landed',
    'proposal_change_kind': 'engineering',
    'generator_hint': 'manual',
    'trigger': (
        '干旱复核第 4 项把「一个冻结设计、三个组合器」落成了 scripts/fuxi_model_compare_20260924.py，'
        '其冻结守卫是手写的 DESIGN_FIELDS 逐项比对。这条守卫有两处结构性不足：'
        '① 只在 0 与非 0 两档 ⇒ 合法的单维增量与「同时改两个维度」在它眼里是同一件事，都抛错；'
        '② 只存在于一个批次里 ⇒ 换一个批次就不存在了。'
        '后果是真实的：要跑一次增量实验，唯一的路是另写一个批次脚本，而那样就一份守卫都没有。'
        '干旱的出口是「扩信息面 + 标尺准入」，而扩信息面的每一步都必须能回答「这一步值多少」；'
        '增量结论一旦可以被多维度同时改动污染，这个回答就不可读。'
    ),
    'current_state': (
        'P6 之前：全仓 14 份 reports/*/protocol.json 中，只有 reports/model_compare_20260924/protocol.json '
        '带 revisions 通道（design_fields_changed），其余 13 份没有任何「相对基准改了什么」的记账通道。'
        'model_compare 的 design_fields_changed 是**按字段名**逐项比对的：加一个候选特征会同时动 '
        'features / dropped_by_coverage / dropped_by_redundancy / selection_rules 四个字段，'
        '于是「一个决策」被计数成「四处改动」，合法的单维增量永远不可能通过。'
        '另有一处口径未定义：features[i].coverage 是数据算出来的**读数**，'
        '按名扁平比较会把它当成设计字段 ⇒ 每次补数据都会把「同一次增量」读成一次设计变更（与 mem:AD5 同一根刺）。'
    ),
    'proposed_change': (
        '① research/factor_alchemy/incremental_contract.py（新建，纯函数、零副作用）：'
        'BOOKKEEPING_KEYS 全局兜底（只收身份/时间/摘要/审计轨迹），未登记名字一律按 DESIGN 计（fail-closed）；'
        'axes_of 把「按构造一起动」的设计字段声明成**轴**，Spec 里声明了却不存在的名字即抛 IncrementSpecError；'
        'feature_delta 把 features[].coverage 归为**读数**并从设计比较里投影剔除；'
        'check_increment 返回 verdict / accepted / axes_changed / feature_delta / flags / reasons。'
        '判定值：IDENTICAL / SINGLE_AXIS / MULTI_AXIS / AXIS_ITEM_BATCH / UNDECLARED_KEY；'
        'accepted 只在前两者为真，且**对变更集单调不增**。'
        '② 🔴 未写下来的新键直接拒绝：一个未登记的名字背后可以装任意多个维度，按「名字」看却只动一处 —— '
        '这是真实的夹带口子。合法化路径唯一：写进某个轴，或写进 BOOKKEEPING_KEYS / Spec 证明它只是记账/读数。'
        '放宽须显式传 allow_new_design_keys=True（默认 False）。'
        '③ MODEL_COMPARE_AXES 七个轴覆盖生产脚本 DESIGN_FIELDS 的 18/18 个字段、无重复声明；'
        '契约声明的字段集与生产源码字面量**同一个字符串**（AST 读源码，不 import、不触发重依赖），单测钉死。'
        '④ scripts/fuxi_p6_incremental_contract_20260925.py（新建）：--tests / --run；'
        '普查全仓协议、独立复核历史修订声明、九个反事实（全部由真实封存协议派生）、单调性探针、'
        '旧守卫 vs 契约对照；fail-closed 断言全部通过才写盘。'
    ),
    'implementation_scope': (
        'research/factor_alchemy/incremental_contract.py（新建）; '
        'research/factor_alchemy/tests/test_incremental_contract.py（新建，27 条）; '
        'scripts/fuxi_p6_incremental_contract_20260925.py（新建，20 条）; '
        'reports/incremental_contract_20260925/readings.json（33581 bytes）; '
        'reports/incremental_contract_20260925/counterfactuals.csv（9 行）; '
        'reports/incremental_contract_20260925/protocol_census.csv（14 行）; '
        'reports/fuxi_p6_incremental_freeze_20260925.md（报告）'
    ),
    'expected_impact': (
        '契约单测 27/27、批次单测 20/20（合计 47），alchemy 全量 184 → **211/211** 全绿。'
        '九个反事实：5 接受 / 4 拒绝，逐个命中事前写下的预期。'
        '关键刻度（C3 合法单维增量 vs C4 污染）：旧守卫按名计数 2 → 拒、4 → 拒（分不出）；'
        '契约按轴计数 1 个轴 → 接受、2 个轴 → 拒（分得出）；两者的特征触碰数都是 1 '
        '⇒ 拒绝理由被正确归因到「多了一个轴」，不会被误报成「特征加太多」。'
        '单调性探针：从被接受的 C3 出发逐个加第二处改动，5 个探针全被拒、接受从未翻回（acceptance_came_back=false）。'
        '普查：14 份协议、带修订通道 1 份、不带 13 份（名单逐条列出）。'
        '历史修订声明复核：protocol_00 → protocol.json 重推导 verdict=IDENTICAL、changed_design_fields=[] '
        '⇒ 声明属实；并查出记录的 previous_protocol_sha256=da92a0eb… 是**文件**摘要（= protocol_00.json 的 sha256），'
        '而**内容**摘要是 c7c31a7b…；同内容重排键序+改缩进后文件摘要变 87d4e973…、内容摘要不变 '
        '⇒ 文件摘要不能当「基准是谁」的判据，契约补 protocol_sha256()（内容摘要）。'
        '对干旱的实质帮助（诚实评估）：本批不产生任何因子、不改任何判定基线、不测任何 IC。'
        '它解的是归因闸门 —— 契约把「增量结论只有在只动了一个东西时才可读」'
        '（与 mem:AE5「IC 冗余不足以判死外族候选，须用固定基准组合上的增量净收益」同一方法论）'
        '从靠自觉变成机检；并量化出 13/14 份协议之间的版本变化在账上不记「改了什么」。'
        '属「让解旱结论可信」的基础设施，不属解旱手段本身。'
    ),
    'landing_note': (
        '本批**只读**协议与源码字面量：read_only_guarantee.protocol_files_unchanged=true，'
        '前后逐文件 sha256 相同（单测钉死）。model_compare 的 protocol.json 一字未改；'
        '历史协议与历史摘要一律不改写（改写即伪造历史）。'
        '未翻开任何开关、未改账本/S5/JQ 准入、未改任何封存产物的实质读数、未接任何生产调用点。'
        '一处被测试逼出来的加强（留痕）：初版契约把「多出一个未登记的键」判成 SINGLE_AXIS 并接受；'
        '想清楚后改为 UNDECLARED_KEY 直接拒绝 —— 一个未写下来的键名可以在内部装下任意多个维度，'
        '按名字看却只动一处，这是真实的夹带口子。相应地补了两条单测（写下来才合法 / 放宽须显式）'
        '与两个反事实（C7 拒绝、C9 写下来之后接受），把这条改动的两端都钉住。'
    ),
    'verification': (
        'cd research/factor_alchemy && PYTHONPATH=/e/quant/research/factor_alchemy '
        '<py> -X utf8 -m unittest tests.test_incremental_contract => Ran 27 tests, OK；'
        '<py> -X utf8 scripts/fuxi_p6_incremental_contract_20260925.py --tests => Ran 20 tests, OK；'
        '--run => exit 0，fail-closed 断言全通过（矩阵非空转、9 个反事实逐个命中预期、'
        '历史声明复核属实、摘要口径三种相等关系、单调性探针 refusals>0 且 acceptance_came_back=False、'
        '轴覆盖 18/18、普查分母两侧相加等于总数、旧守卫确实拒过合法增量、协议文件前后未变）；'
        '<py> -X utf8 -m unittest discover -s tests -p "test_*.py" => Ran 211 tests, OK。'
    ),
    'artifacts': [
        'research/factor_alchemy/incremental_contract.py',
        'research/factor_alchemy/tests/test_incremental_contract.py',
        'scripts/fuxi_p6_incremental_contract_20260925.py',
        'reports/incremental_contract_20260925/readings.json',
        'reports/incremental_contract_20260925/counterfactuals.csv',
        'reports/incremental_contract_20260925/protocol_census.csv',
        'reports/fuxi_p6_incremental_freeze_20260925.md',
    ],
    'zero_adjudication': True,
    'default_on': False,
    'ab_evidence': (
        '反事实 A/B（九个，全部由真实封存协议派生）：C3 单特征增量 接受 vs C4 加第二个轴 拒绝；'
        'C5 同轴两特征 拒绝 vs C3 同轴一特征 接受；C6 只变 coverage 判为 IDENTICAL（不是增量）'
        'vs C3 真设计改动判为 SINGLE_AXIS；C7 未写下来的键 拒绝 vs C9 同一键写进轴后 接受。'
        '旧守卫 vs 契约对照：C3/C9 旧守卫拒、契约接受（旧守卫拒过合法增量，有据）；'
        'C4/C5/C7/C8 两者都拒。单调性 A/B：5 个第二处改动探针全被拒、接受从未翻回。'
    ),
    'known_limit': (
        '① 契约只判**形状**，不判断某个维度该不该改、改动是否合理；'
        '② 它不阻止任何人改两个维度，只让这种改法不能被报成单维增量；'
        '③ 其余 13 份协议**没有**装上修订通道，本批只把这件事量化并列出名单，接线需新协议/新批次；'
        '④ 「一次只允许触碰一个特征」是契约设定的阈值（默认 1），不是从数据估出来的；'
        '⑤ BOOKKEEPING_KEYS 是按**名字**的全局兜底，跨层重名的字段必须由 Spec 显式声明 —— '
        '名字匹配跨层本身就是 mem:AE4 记过的坑，契约只能把它显式化，不能自动消掉；'
        '⑥ 七个轴只对 model_compare_20260924 声明；其余 13 份协议要用契约须先写各自轴声明；'
        '⑦ 本批对盈亏平衡 IC 缺口没有直接贡献，不得当成解旱成果。'
    ),
    'next_step': (
        'P0~P7 全部落地后转入**结论报告**：重点回答「之后怎么解决因子干旱」。'
        '已具备的素材：P4 把「静默截断」变成拒跑（盈亏平衡线 -0.00843 + 0.0915×IC 的四个采集输入进入契约）；'
        'P5 让干旱的最终裁决在日历上可到期（判定端点 signal 2027-03-19 / matured 2027-03-29，26 段要求逐字未改）；'
        'P6 保证增量结论不被多维度同时改动污染。'
        '仍未解的最高优先级：组合层**水平项本身**的测量 —— IC=0 时 Top80 仍少赚（水平项 -0.00843、t=-6.35，'
        '剔 4 个重叠截面后 -0.00720、t=-5.78）；盈亏平衡 IC +0.108/+0.083 vs 实测 +0.054/+0.063，'
        '缺口 = IC 需翻倍。'
    ),
    'rollback': (
        '删除 research/factor_alchemy/incremental_contract.py、'
        'research/factor_alchemy/tests/test_incremental_contract.py、'
        'scripts/fuxi_p6_incremental_contract_20260925.py 与 reports/incremental_contract_20260925/ '
        '即可；无生产调用点、无开关、无数据文件被改，回滚不需要还原任何产物。'
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
    backup = TARGET.with_name(TARGET.name + '.bak_%s_p6' % stamp)
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
