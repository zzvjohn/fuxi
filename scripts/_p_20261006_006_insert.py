# -*- coding: utf-8 -*-
"""_p_20261006_006_insert.py — 把 P-20261006-006 (BJ universe 方案 A 落地) 写入提案库。

纪律: 幂等 (已存在则拒绝重复插入); 写前备份; 写后回读校验。
"""
import json
import os
import shutil
import sys

ROOT = r'E:\quant'
PATH = os.path.join(ROOT, 'data', 'stage1_optimization_proposals.json')

ENTRY = {
    "proposal_id": "P-20261006-006",
    "title": "BJ universe 时点裁剪（方案 A）落地：新增开关 FUXI_S5_BJ_UNIVERSE（默认 legacy；零默认变化）",
    "category": "系统开关/默认值变更",
    "domains": ["T2数据口径", "T3模块/通道"],
    "risk_level": "controlled_research_only",
    "status": "landed",
    "proposal_change_kind": "engineering",
    "generator_hint": "manual",
    "trigger": "用户指令『串行 A→B→C开跑』（2026-10-06）。母协议 PRE_REGISTRATION_bj_universe.md（M17 §4 独立预注册纪律：不与解锁候选捆绑）。方案 A 先行（面窄、只减、可回退）。",
    "current_state": "无 FUXI_S5_BJ_UNIVERSE 开关。920xxx.BJ 的 list_date 是北交所上市日，而行情行自新三板挂牌期（2021-01-04 起）已存在；挂牌期『无成交日』不生成行情行 ⇒ S5 池内该股被建仓/持有后遇缺格 ⇒ held_price_observations_missing 阻断（M17 2026-09-28 已裁决根因：市场微观结构事实，非数据洞；补数不能解锁）。M17 三候选缺格 {228,561,40}，另 +值级守卫 2/12/5 与退市后 0/6/0。",
    "proposed_change": "s5_joint_filter.py 新增 5 处（受控 bytes patch，patch 前先备份）：①:73-77 _s5_bj_universe() 开关 getter（默认 legacy；值域 legacy|listed_only；非法值 fail-fast）；②:83-109 _bj_listing_date_map() 上市日映射（delist_dates.csv 351 + data/raw/bj_listing_date_overlay.csv 2 = 353/353 覆盖）；③:112-134 _s5_bj_cut_pre_listing(frame) 时点裁剪（_drop 为逻辑与、首项即 .BJ 掩码 ⇒ 非 .BJ 行结构性不可裁；无 list_date 的 .BJ fail-safe 不裁）；④:2549-2557 账本分流 fail-closed（FUXI_S5_LEDGER≠proxy 且 listed_only ⇒ raise ValueError，拒绝静默失效）；⑤:2619-2626 主路径在 price 面板构建后裁剪（merged 为 inner join ⇒ 一处裁剪同时覆盖 entry 与 marking 两路）。",
    "expected_impact": "零默认变化：默认 legacy ⇒ 现行生产行为逐位不变（J6 实测：备份副 A0 与现行 A1 在 3 候选上 readings+cells 逐位一致、pair_match 全真；差异仅 engine_switches 元数据）。listed_only 档实测效益（J1/J5）：目标类 {228,561,40}→0（3/3，与 M17 逐位吻合）；全窗裁 19,844 行 / 166 只 .BJ（2021:12052 / 2022:7792）；非 .BJ 零穿越（6,836,161 行 nonbj_dropped=0）；removed 侧 895 格 removed_all_bj=true。🔴 不改善经济读数：裁决窗口 2025/2026 缺格两臂逐位相同（BJ 上市前缺格全在 2021-2023），excess/calmar/return 逐位一致，差异仅 marking_diagnostics；3 候选 enforce 状态不变（另有真停牌类缺格未消，不属本议题范围）。IC/ICIR 由 self._price_df 直算，对本开关结构性免疫。",
    "implementation_scope": "research/factor_alchemy/s5_joint_filter.py 5 处（210418→215580 bytes；sha e244b2b8e778ed86…→c42b88e08a08763d…；纯 CRLF 保持，裸 LF 0→0）。附带只读取证脚本 7 个（reports/fuxi_v1_bj_universe_20261006/ + scripts/_bj_uni_*）。不改任何其它开关默认、不改数据、不改 suspend_d 语义、不触标记路径分类学。",
    "landed_detail": "证据链（全部只读重放）：①§6-A 上市日通道审计 pass（353/353 覆盖 + 东财 clist f26 逐只一致 0 mismatch）；②§6-B annotate 台账普查（入口面 91,920 事件 100% tradable；缺格面 uncertifiable 876 全为 BJ 且全在 2021-2023）；③J1–J7 冻结重放判据读数（5 臂 × 8 候选）；④EC5 机制级性质测试 12/12（含非 BJ 逐位恒等、纯函数、幂等、fail-closed 拒绝实测）；⑤J4 负对照 1/3（结构性稀缺：19 条真出回测者中仅 1 条零 BJ 暴露 = 5.3%，因 BJ 占池 6.10% × top_n=80 × ~286 期 ≈ 2.3 万槽位；已以 EC5 替代取证并如实登记不达标）。产物：PLAN_A_J1_J7_READINGS_REPORT.md / j17_readings.json / j17_verdict.json / gap_ledger_crosswalk.json / annotate_census.json / ec5_property_test.json / j4{,b,c}_readings.json。",
    "rollback_plan": "默认即 legacy（回退无需动作）；若曾显式设 FUXI_S5_BJ_UNIVERSE=listed_only，取消该 env 即回退。字节级回退备份 reports/fuxi_v1_bj_universe_20261006/pre_bjcut/s5_joint_filter.py.bak（sha256 e244b2b8e778ed865e6ffdce8335a22d46040e9ec3a42a83dbf6222e2c3630fc，210418 bytes，裸 LF 0）。",
    "jq_validation_required": False,
    "jq_validation_status": "not_applicable",
    "references": [
        "reports/fuxi_v1_bj_universe_20261006/PRE_REGISTRATION_bj_universe.md",
        "reports/fuxi_v1_bj_universe_20261006/PLAN_A_J1_J7_READINGS_REPORT.md",
        "reports/fuxi_v1_bj_universe_20261006/EXECUTION_ORDER_plan_B.md",
        "reports/fuxi_p2m4_m17_reeval_20260928/M17_REEVAL_REPORT.md",
    ],
    "cross_verified": True,
    "source": "fuxi_v1_bj_universe_20261006 (方案 A；B 阶段以『不予开工』结项)",
}


def main():
    with open(PATH, 'r', encoding='utf-8') as fh:
        rows = json.load(fh)
    assert isinstance(rows, list), type(rows)
    ids = [r.get('proposal_id') for r in rows]
    if ENTRY['proposal_id'] in ids:
        print('[skip] %s already present' % ENTRY['proposal_id'])
        return 0
    bak = PATH + '.bak_p20261006_006'
    shutil.copy2(PATH, bak)
    rows.append(ENTRY)
    tmp = PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
        fh.write('\n')
    os.replace(tmp, PATH)
    # 回读校验
    back = json.load(open(PATH, encoding='utf-8'))
    hit = [r for r in back if r.get('proposal_id') == ENTRY['proposal_id']]
    assert len(hit) == 1, len(hit)
    assert hit[0]['status'] == 'landed'
    print('[ok] inserted %s | n_rows %d -> %d | backup %s'
          % (ENTRY['proposal_id'], len(rows) - 1, len(back), os.path.basename(bak)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
