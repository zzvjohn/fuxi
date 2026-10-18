# -*- coding: utf-8 -*-
"""P-20260918-001/002/003 审核结论与落地状态写回提案账本 (CRLF 保真)。

🔴 本文件所有写回一律 newline='\r\n': 用 newline='' 会把整文件转裸 LF (静默)。
   写回后必须回读并断言 bare_LF == 0。
"""
import json
import os
import shutil
from datetime import datetime

REPO = r"e:\quant"
P_PATH = os.path.join(REPO, "data", "stage1_optimization_proposals.json")

# ── P-001: 通道断裂修复 + 分歧度因子 ──────────────────────────
N001 = (
    "[实现核查+落地 2026-09-18] 提案前提「缺乏三路资金方向分歧度的截面合成」**部分高估**: "
    "昨日 P-20260917-022 已入库 novel_234/235/236 三条三路因子 "
    "(lhb_triple_resonance_5d / _persist_20d / _agree_minus_disagree_10d), "
    "均消费 lhb_north_net_buy + lhb_inst_net_buy + lhb_yz_net_buy。逐行核对后判定 "
    "**真实瓶颈是数据通道断裂, 不是缺因子**: "
    "(1) lhb_yz_* 在 factor_ic_computer.py 全文出现 0 次 (对照 inst 6 次 / north 7 次) "
    "⇒ 该通道从未被加载 ⇒ novel_234/235/236 在 IC/S5 路径下 eval 报 "
    "NameError: name 'lhb_yz_net_buy' is not defined → **静默丢候选**; "
    "(2) s5_joint_filter._build_series_context 此前连 lhb_inst_*/lhb_north_* 都没有 "
    "(只有 top_list 4 个汇总字段), 同族 bug 在该文件 L1381 有前科记录 "
    "(「S5 环境缺 lhb_flag → 库内 lhb 类模板 eval 报 NameError 全军覆没」); "
    "(3) forge_translate.S5_KNOWN_TERMINALS 自称「与 s5_joint_filter/factor_ic_computer "
    "宽表字段对齐」, 实测**三方漂移**: 缺 lhb_north_*(ic 自 v0.9.2 已有) 与 lhb_yz_* "
    "⇒ find_unknown_identifiers 把实际可求值的公式判为不可求值 (fail-closed 假阴性), "
    "与「判据锚在实现」铁律相悖。 "
    "⇒ 落地范围定为「先修通道, 再谈分歧度」: "
    "D1 factor_ic_computer.py 新增 YZ_INST_FIELDS + 游资营业部 loader "
    "(与 inst/north 同构, 口径与 data_loader_ext.load_top_list_wide P-20260904-004 严格同式); "
    "D2 s5_joint_filter.py 新增 top_inst 三路 loader + _LHB_TRI_COLS(9 字段) 注入 per-stock 上下文; "
    "D3 forge_translate.py 白名单补齐 north/yz (含 _p); "
    "D4 NOVEL_FACTOR_LIBRARY 新增 novel_237~240 四条**分歧度/集中度**变体 "
    "(仅上榜样本, 关键差异是**不用 fillna(0) 摊平** —— 既有 234/235/236 非上榜日三路全 0 "
    "⇒ 截面 rank 被大量 0 值占位稀释)。 "
    "开关 FUXI_LHB_TRI_CHANNEL = on(默认) | off(=修正前, yz 缺失)。 "
    "端到端 A/B (2026-09-18, FactorICComputer lookback=500, 全市场 1,893,814 行): "
    "OFF: novel_234/235/236 → None(NameError) 静默丢弃; ctrl_inst_north → uniq=379。 "
    "ON : novel_234 uniq=67 / 235 uniq=172 / 236 uniq=123; 面板 lhb_yz_net_buy 非零 22,260 行; "
    "**ctrl_inst_north uniq=379 逐位不变** ⇒ 严格增量, 不扰动既有通道 (默认↔口径档不变量过)。 "
    "新因子端到端: novel_237 uniq=1305/std=0.152 | 238 uniq=94557/std=0.096 | "
    "239 uniq=1054/std=0.155 | 240 uniq=249/std=0.061, 全部非近常量。 "
    "🔴 施工中自查发现 novel_238 首版为**死因子** (uniq=1, std=0.0): 根因是 "
    "pd.concat([宽表×3], axis=1).max(axis=1) 会塌掉整个截面 (退化为按日期的 Series), "
    "已改用 np.maximum 元素级取最大 ⇒ 修后 uniq=94557。 "
    "🔴 **另发现独立缺陷 (建议单独立项)**: 全库 270 条中有 31 条 (11.5%) 公式以 "
    ".rank(pct=True, axis=1) 结尾, 而 pandas Series.rank(axis=1) 直接抛 "
    "ValueError: No axis named 1 for object type Series ⇒ 这些因子在 S5 "
    "**逐股 Series 上下文**下必然失败 (novel_234/235/236 与本次 237~240 同式, 属既有口径; "
    "本次沿用既有写法未扩大差异, 但该轴语义跨层不一致是真缺陷)。"
)

# ── P-002: 收缩版影子模块 + 成本敏感性门候选 ────────────────
N002 = (
    "[落地 2026-09-18] 收缩版已落地: 新增 research/factor_alchemy/trade_gate_shadow.py "
    "(成本敏感性影子诊断, 零裁决), 并接线到 ralph_loop.py Phase E (StyleGuard 之后, "
    "try/except 非阻塞), summary['trade_gate'] 落读数。env FUXI_TRADE_GATE = shadow(默认, 生效) | off。 "
    "口径: 本地 5 日 fwd 代理面板, top_n=30, 年化 = 期均 × (252/5); turnover 与 "
    "s5_joint_filter P0-g 严格同式 = (len(prev-cur)+len(cur-prev))/top_n, 首期 1.0; "
    "成本档位 15/30/45 bp; 落盘 data/trade_gate_shadow.json。 "
    "判据双轨 (缺一必漏): fragile_30bp (15bp>0 且 30bp≤0, **仅多头方向有效**) + "
    "high_drag_30bp (drag30/|gross| > 0.5, **方向无关**) + min_periods=12 才出结论。 "
    "🔴 施工中自查: 首版只有 fragile 判据 (要求 15bp 档为正) ⇒ 对 gross<0 的因子 "
    "(如 slow_rev_60 gross=-10.0%) 恒不触发, 属 fail-open 漏报; 已补 high_drag 方向无关判据。 "
    "冒烟 (2026-09-18, standalone lookback=750, 4 因子): "
    "forge_gen4_r1 turn=1.23 gross=+8.3% → 15bp -1.0%, drag30=224% [陡降·高拖累]; "
    "fast_mom_2 turn=1.92 gross=+17.7% → 15bp +3.2% / 30bp -11.3% [脆断·陡降·高拖累]; "
    "slow_rev_60 turn=0.91 gross=-10.0% → 仅 [高拖累] (验证方向无关判据生效)。 "
    "🔴 读数**窗口敏感**: 同一公式 lookback 270→750 日, forge_gen4 r1 毛年化由 -38.9% 翻到 "
    "+8.3%, 期数 19→83 ⇒ caliber 已记录 standalone_lookback_days 并标注「跨轮比较必须同窗」。 "
    "成本敏感性门候选 (仅提案, **未生效**): S1 通过后对 high_drag 因子在 S2/S5 改用 "
    "15bp/30bp 双档对账, 要求 net(30bp)>0 才进 JQ 队列。证据: (1) 本模块 fast_mom_2 脆断实测 "
    "(15bp +3.2% → 30bp -11.3%); (2) research/qlib_daily/p4_cost_capacity_report.json 日频 "
    "cost×1/×2/×3 → 年化 18.95%/9.51%/0.82% (turnover 33%); (3) forge_gen4_r1 drag30=224%。 "
    "样本仅 3 个因子偏薄 ⇒ 先影子跑若干轮, 用 data/trade_gate_shadow.json 的 "
    "n_high_drag/n_checked 比例定阈值, 再按「翻默认四关」执行。"
)

# ── P-003: 半衰期 —— 前提量纲错配, 仅落影子 ──────────────
N003 = (
    "[实现核查+影子落地 2026-09-18] 提案**前提存在量纲错配**, 判为 deferred "
    "(仅落影子报告, **不改任何阈值**)。逐行核对 scripts/factor_decay_profile.py: "
    "classify() 的分段锚在 half_life_days = FactorICComputer.compute_decay 在 lag 1-20 "
    "**交易日**上指数拟合的 IC 自衰减半衰期 (日尺度, 度量「信号平滑度/微观持续性」); "
    "而提案引用的行业证据 (投顾猫 09-14「半衰期从 2-3 年压缩至 6-12 个月」) 指的是 "
    "**因子有效生命周期** (alpha 在年尺度上被套利掉的时间; 同源 McLean & Pontiff 2016 "
    "公开后平均衰减 58%)。两者**不是同一个量**, 不存在「把 fast 线从 5 日改成 120 日」的映射。 "
    "实测证据 (data/factor_decay_profile.json, generated 2026-09-06, n=15): "
    "可识别性: half_life=inf (λ≤1e-6, IC 在 lag 1-20 内不衰减) 12/15 | 有限 2/15 | "
    "缺失 1/15 | fit_r2<0 1/15; 有限读数仅 89.3 / 100.7 / 112.2 日; "
    "现行 fast<5 日线 → 触发 0/15 (该阈值实际上已**失效**, 从不触发); "
    "阈值扫描: <90 日→1/15(7%) | <120 日→2/15(13%) | <180 / <250 / <365 日 仍→2/15(13%)。 "
    "⇒ 提案建议的 5→120 日改动属单向收紧 (理论可翻), 但作用域只落在 2 条有限读数上, "
    "其余 12 条 inf 读数对**任何有限阈值都不响应** ⇒ 改阈值等于放大一个**不可识别**的度量, "
    "属**增加任意性**, 违反项目「去任意性才可翻默认」的判据。 "
    "落地: 仅新增 --half-life-report 影子口径 (零裁决, 落 data/factor_half_life_report.json, CRLF), "
    "输出可识别性 + 全阈值扫描表 + 量纲对照。**classified 阈值一行未动。** "
    "建议: 若要跟进行业「半衰期压缩」, 应**新建年尺度生命周期度量** "
    "(按年度分组 IC 趋势 / 滚动 12 个月 IC 斜率), 而非改本档日尺度阈值 —— 建议立项为新提案。"
)


def main():
    print("=" * 78)
    print("  P-20260918-001/002/003 审核结论写回账本")
    print("=" * 78)
    bak = P_PATH + ".bak_20260918_audit"
    if not os.path.exists(bak):
        shutil.copy2(P_PATH, bak)
        print("  备份 ->", os.path.basename(bak))
    else:
        print("  备份已存在, 跳过:", os.path.basename(bak))

    with open(P_PATH, "r", encoding="utf-8", newline="") as f:
        d = json.load(f)
    print("  读入提案数 =", len(d))

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    plan = {
        "P-20260918-001": {
            "status": "landed",
            "audit_verdict": "landed_dead_channel_fix",
            "audit_date": "2026-09-18",
            "audit_note": N001,
            "landed_at": now,
            "landed_files": [
                "research/factor_alchemy/factor_ic_computer.py",
                "research/factor_alchemy/s5_joint_filter.py",
                "research/factor_alchemy/forge_translate.py",
                "scripts/daily_factor_hypothesis.py",
            ],
            "env_switch": "FUXI_LHB_TRI_CHANNEL = on(默认) | off(=修正前)",
        },
        "P-20260918-002": {
            "status": "landed",
            "audit_verdict": "landed_shadow_default_on",
            "landed_at": now,
            "landing_note": N002,
            "landed_files": [
                "research/factor_alchemy/trade_gate_shadow.py (新建)",
                "research/factor_alchemy/ralph_loop.py (接线 Phase E)",
            ],
            "env_switch": "FUXI_TRADE_GATE = shadow(默认) | off",
            "gate_candidate": "cost_robust_30bp (仅提案, 未生效)",
        },
        "P-20260918-003": {
            "status": "deferred",
            "audit_verdict": "deferred_premise_flawed",
            "audit_date": "2026-09-18",
            "audit_note": N003,
            "landed_at": now,
            "landed_files": ["scripts/factor_decay_profile.py (--half-life-report 影子口径)"],
            "env_switch": "(无开关: 仅新增 CLI 影子模式, 不改裁决)",
        },
    }

    seen = set()
    for e in d:
        pid = e.get("proposal_id")
        if pid in plan:
            for k, v in plan[pid].items():
                e[k] = v
            seen.add(pid)
            print("  已写回 %s -> status=%s verdict=%s"
                  % (pid, e.get("status"), e.get("audit_verdict")))
    missing = set(plan) - seen
    if missing:
        print("  ⚠️ 未找到:", missing)

    with open(P_PATH, "w", encoding="utf-8", newline="\r\n") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

    # ── 回读复核 (不能只信"写入成功") ──
    with open(P_PATH, "rb") as f:
        b = f.read()
    crlf = b.count(b"\r\n")
    bare = b.count(b"\n") - crlf
    d2 = json.loads(b.decode("utf-8"))
    print()
    print("  回读: 提案数=%d  CRLF=%d  bare_LF=%d" % (len(d2), crlf, bare))
    for e in d2:
        if e.get("proposal_id") in plan:
            nm = e.get("audit_note") or e.get("landing_note") or ""
            print("    %-18s status=%-9s len(note)=%d"
                  % (e.get("proposal_id"), e.get("status"), len(nm)))
    assert bare == 0, "🔴 行尾被破坏: bare_LF=%d" % bare
    print("  ✅ 行尾 CRLF 保真")


if __name__ == "__main__":
    main()
