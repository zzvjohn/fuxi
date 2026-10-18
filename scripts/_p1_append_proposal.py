# -*- coding: utf-8 -*-
"""一次性脚本：向 data/stage1_optimization_proposals.json 追加 P-20260924-011。

纪律：
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
NEW_ID = 'P-20260924-011'

ENTRY = {
    'proposal_id': NEW_ID,
    'date': '2026-09-24',
    'created_at': '2026-09-24T14:40:00+08:00',
    'title': (
        'P1 大宗交易覆盖修复 + H1 预注册检验：族上限由 CEILING_UNCLEAR 闭合为 CEILING_BELOW（87/87 截面）；'
        'P0 的 +0.1131 拆解为数据效应 -0.0218 与样本效应 -0.0348；bt_premium 在独立样本上站住 +0.0480，bt_share 未决'
    ),
    'category': '因子研究/信息面',
    'domains': ['T1闸门/裁决', 'T2数据口径', 'T3模块/通道'],
    'risk_level': 'low',
    'status': 'landed',
    'proposal_change_kind': 'research',
    'generator_hint': 'manual',
    'trigger': (
        'P0 判定的 next_step 明确：不再按五族平均用力，只对大宗交易投入（补全逐交易日覆盖 + 把 bt_share 的正向写成新预注册假说 + '
        '覆盖 <5 只的截面拒绝产数）。P0 唯一接近盈亏平衡的读数就出自该族（37 个截面上 raw +0.1131），且恰好是 P0 中符号为 -1 的那一个，'
        '因此必须先把覆盖补齐、再把方向作为新假说预注册，而不是沿用 P0 的符号继续读。'
    ),
    'current_state': (
        'P0 使用 data/raw/block_trade_full.csv：窗口内仅 155/426 个交易日有数据、存在 14~28 日连续缺口，'
        '单一族口径下 87 个截面里只有 37 个读出有限值，故 P0 只判到 CEILING_UNCLEAR；'
        '其 bt_share 方向在 P0 协议里注册为 -1（事前预注册，但建立在「掩护性大宗成交 = 出货」的先验上），'
        '而该先验与逐笔明细里呈现的协议转让机制（同一笔按对手方逐腿输出，6~17 组/日）不相容。'
    ),
    'proposed_change': (
        '建两个新脚本与一个新批次：① 采集器按供应商逐交易日接口补齐 2024-12-01 起的全部交易日，'
        '范围由本地价格面板末交易日派生（可用 env 覆盖，不钉死字面量）；去重只折叠七字段全同的 feed 重复行、'
        '保留不同对手方的真实腿，并在 manifest 里披露折叠数/保留腿数/口径/逐日行数/sha256。'
        '② 评估器复用（import，不重写）P0 的池子、标签、残差设计、簇推断与判定器，只做一处重接线：把大宗交易族的源文件换成新表。'
        '③ 把 bt_share 的 +1 方向写成新假说 H1 并在读新截面之前写进协议，同时把「P0 从未读到读数的截面」定为主样本、'
        '已读过的 37 个截面明确定义为样本内并只作次要披露（禁止用样本内读数冒充验证）。'
        '④ 判定规则与阈值完全沿用 P0（不可在读数后放宽）：|点估计| >= +0.108 为 AT_OR_ABOVE；只有簇区间上界 < +0.108、三段同号且 >= 44 个截面才可判 CEILING_BELOW。'
    ),
    'implementation_scope': (
        'scripts/fuxi_collect_blocktrade_20260924.py（新建，采集器）; '
        'scripts/fuxi_blocktrade_p1_20260924.py（新建，评估器，import 复用 P0）; '
        'data/raw/block_trade_daily.csv + data/raw/block_trade_daily.manifest.json（新数据资产）; '
        'data/raw/block_trade_daily.csv.legfolded_evidence（有意保留的中间产物留证）; '
        'reports/blocktrade_p1_20260924/（protocol.json/summary.json/per_section.csv/coverage.csv/geometry.csv/timing.csv/drops.csv/tests.json/runs/）; '
        'reports/blocktrade_p1_20260924/defect_round1_subset_and_codes/（缺陷轮留证）; '
        'reports/blocktrade_p1_20260924/defect_round2_iso_daykey/（缺陷轮留证）; '
        'reports/fuxi_blocktrade_p1_20260924.md（报告）'
    ),
    'expected_impact': (
        '数据资产：442 个交易日全有数据、0 空日；供应商返回 71,451 行，折叠 5,760 条七字段全同行，写入 65,691 行并保留 7,640 条对手方腿'
        '（在共享名字-日上找回成交额 20.25 亿元，旧表系统性少计）。'
        '覆盖：窗口内旧 3,615 行 / 1,916 个可定价名字-日 -> 新 12,432 行 / 5,334 个可定价名字-日；'
        '单一族口径下可读截面由 37/87 升到 87/87，每截面覆盖名字均值 60.5（25~211）。'
        '源替换审计三条断言全部通过：股票宇宙内新表每笔成交当日可定价的比例精确为 1.0；旧表可定价的名字-日一条未丢（name_days_only_in_old = 0）；'
        '同名-日新表成交额无一处更小（cells_new_below_old = 0，另有 206 处更大）。1709 行非股票标的（可转债/ETF/LOF/REIT）按范围剔除而非静默丢弃。'
        '族判定：block_trade 闭合为 CEILING_BELOW（87/87 截面、三段同号）；残差口径上限 +0.0559（t=3.30）、raw 口径上限 +0.0666（t=3.47），'
        '两者都 < 盈亏平衡 +0.108。P0 的 CEILING_UNCLEAR 至此闭合为「否」。'
        '逐定义：bt_premium 残差 +0.0457（t=3.25）、raw +0.0338（t=2.11），方向与预注册一致；'
        'bt_share 残差 +0.0311（区间 [-0.0020, +0.0642] 跨零）、raw +0.0565（t=2.76）。'
        'H1 主检验（P0 从未读过的 50 个截面）：bt_share 残差 +0.0299（区间 [-0.0162, +0.0759]，t=1.27）、raw +0.0308、Pearson -0.0110，'
        '三段符号不一致（+0.0127 / -0.0203 / +0.1035）=> 判 unresolved（既未支持也未否证），不得宣称方向已确认。'
        '唯一在独立样本上站住的是 bt_premium：主样本残差 +0.0480（[+0.0151, +0.0809]，t=2.86）、raw +0.0429（[+0.0068, +0.0789]，t=2.33）、'
        '方向沿用 P0 预注册的 +1 未做任何修改、样本独立、区间不跨零。'
        'P0 亮眼读数的正式拆解（两项效应可分离）：bt_share raw 0.1131 ->（同 37 截面换新数据）0.0913 ->（全 87 截面）0.0565，'
        '族上限 raw 0.1062 -> 0.0892 -> 0.0666，即数据效应 -0.0218 + 样本效应 -0.0348 合计腰斩。'
        '对干旱的实质帮助：①「换信息族」这条路本批闭合，五族无一族上限达到平衡点，靠新族做不到 IC 翻倍；'
        '② 缺口位置因此更确定 = 组合层水平项（IC=0 时 Top80 仍少赚 72~84bps/5 日），下一优先级应测水平项本身；'
        '③ bt_premium 是条件特征量级（+0.0480、覆盖 1.13%），只能做出门过滤或条件特征，不做多头因子；'
        '④ 对 P2 的直接影响：大宗交易不再需要完整流水线，与行业/业绩预告/基准一起进研究特征库。'
    ),
    'landing_note': (
        '新批次、新协议、新 run_id；零裁决、零影子开关、未翻任何默认值；不改动已冻结的 P0 与第 3 项脚本（只 import 复用），'
        '与之逐截面三重断言（eligible_n / pool_label_mean / top_label_mean）对账通过。'
        'P0 定义集只允许一处改动（bt_share 符号 -1 -> +1），并有单元测试 test_p1_only_changes_one_sign 断言其余三族符号逐字未变。'
        '本批暴露并留证两轮缺陷，均未用成功轮覆盖：'
        '缺陷轮 1（defect_round1_subset_and_codes）：原假设「新表只是旧表的覆盖扩展」是错的——旧文件在两个维度都残缺'
        '（名字-日 9,005 vs 31,407；共享日每天多出中位 12、最大 86 个名字），且 61 个「只在旧表出现」的名字全是北交所代码迁移'
        '（836247.BJ -> 920247.BJ，本地价格面板用 920xxx），旧表那些行根本无法连接价格面板；厂商接口无行数上限，采集完整（442/442、0 空日）。'
        '修法：去重策略改为只折叠七字段全同、保留腿；把「等式式对账」改为三条包含/单调式审计断言。'
        '缺陷轮 2（defect_round2_iso_daykey）：单元测试把日键传成 8 位、生产传 ISO，load_block_rows 内部归一后 isin 全不匹配，'
        '窗口内 0 行导致分母为 0、可定价比例报 0.000000（同一数据只读诊断为 0.8625）；与 P0 轮 2 同一类问题——测试把生产的边界条件补平了。'
        '修法：内部同时接受两种日键 + 形状守卫（有原始行、窗口落在已覆盖范围内、过滤后为 0 即抛 day-key shape mismatch），'
        '并把测试改为传 ISO、另加 test_swap_audit_accepts_both_day_shapes。'
    ),
    'verification': (
        '单元测试 8/8 通过；3696 行 / 有效 2664 / 丢弃 0 / 零填充 0 / 设计秩亏 0 / 因太薄被拒 0；'
        'R0 运行契约 manifest 记录 9 项代码与 5 项数据指纹、status=completed，run 前后各校验一次；'
        '判据与 H1 全文写在 protocol.json（早于任何新截面读数）；失败轮与缺陷轮 manifest 全部保留（status=failed / 独立目录），未被成功轮覆盖。'
    ),
    'artifacts': [
        'scripts/fuxi_collect_blocktrade_20260924.py',
        'scripts/fuxi_blocktrade_p1_20260924.py',
        'data/raw/block_trade_daily.csv',
        'data/raw/block_trade_daily.manifest.json',
        'reports/blocktrade_p1_20260924',
        'reports/fuxi_blocktrade_p1_20260924.md',
    ],
    'zero_adjudication': True,
    'default_on': False,
    'ab_evidence': '无 A/B：本批只采集数据与产出测量结论，不接入生产、不改任何默认值，也不改变 P0 的判定阈值。',
    'known_limit': (
        '① 全部读数落在研究已暴露窗口（2025-01-03~2026-08-28），无样本外证据；'
        '② H1 主样本 50 个截面是 87 个截面中「P0 未读过」的那一半，仍属同一已暴露窗口；'
        '③ bt_premium 覆盖率仅 1.13%，量级属条件特征，不足以做多头因子；'
        '④ 新表是本地采集快照（sha256 已记录），供应商后续可能修订，读数绑定该快照；'
        '⑤ 本批只回答「大宗交易这一族的上限」，不外推到其他族或全库。'
    ),
    'next_step': (
        '转入 P2：按本批与 P0 的结论建「研究特征库 + 策略晋级库」分层——大宗交易（bt_premium，条件特征量级）、行业、业绩预告、基准'
        '统一归入研究特征库（条件特征/出门口过滤），策略晋级库另立并以可裁决性为门槛；北向仍是唯一「等数据」的族。'
        '同时把「组合层水平项本身的测量」列为最高优先级（IC=0 时 Top80 仍少赚 72~84bps/5 日，这个水平项才是主要缺口）。'
    ),
    'rollback': '删除 reports/blocktrade_p1_20260924/、报告与 data/raw/block_trade_daily.csv 即可；未接入生产、未改默认值，无需回退代码。',
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
    backup = TARGET.with_name(TARGET.name + '.bak_%s_p1' % stamp)
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
