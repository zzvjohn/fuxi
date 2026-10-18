# -*- coding: utf-8 -*-
"""追加本会话记录到当日日志 + 自动化记忆。

🔴 两个文件行尾约定不同 (日志 CRLF / 自动化记忆 LF): 本脚本**按原文件自身**
   的行尾风格追加, 追加后回读断言「新引入的裸 LF == 0 (对 CRLF 文件)」。
"""
import os

DAILY = r"e:\quant\.workbuddy\memory\2026-09-18.md"
AUTO = r"e:\quant\.workbuddy\memory\automations\automation-1786597632320\memory.md"


def nl_of(path):
    b = open(path, "rb").read()
    crlf = b.count(b"\r\n")
    bare = b.count(b"\n") - crlf
    return "\r\n" if crlf >= bare else "\n"


def append(path, text):
    nl = nl_of(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        old = f.read()
    if not old.endswith(nl):
        old += nl
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(old + text.replace("\r\n", "\n").replace("\n", nl))
    b = open(path, "rb").read()
    crlf = b.count(b"\r\n")
    bare = b.count(b"\n") - crlf
    print("%-34s nl=%-4r 追加 %d 字符 -> CRLF=%d bare_LF=%d"
          % (os.path.basename(path), nl, len(text), crlf, bare))
    if nl == "\r\n":
        assert bare == 0, "🔴 裸 LF 非 0: %d" % bare
    return bare


DAILY_TXT = """

---

## 会话追加 2026-09-18（提案审核落地 + D+ 闭环 + 死接线修复）

**D+ 闭环（forge_gen4）**：用户回填 JQ 单因子回测读数（2021-01-01~2026-07-30，week=285，5.6 年）
累计 **64.57% / Sharpe 0.266 / maxDD 29.49%**，滞后 Rank IC 0.0280 / ICIR 0.3695 / IC>0 62.7%（255 期）；
本地 ICIR 0.4649 ⇒ JQ 侧衰减。按 `ralph_loop._jq_outcome` 四档判定：收益 >50% 但 Sharpe 0.266 < 0.4
⇒ 正确落 **JQ_MARGINAL**（防「只看收益>50 就标 PASS」）。已做**单因子范围**闭环（不批量跑
`trigger_d_plus.py`）：trajectory `traj_2561b7d3c712` 写入 jq_* 富字段 + `d_plus_applied=True`；
`jq_codegen_queue` 目标键 `status=jq_marginal`；`lane_calibration` 新增 1 锚点（84 points，
最新 `traj_2561b7d3c712` / JQ_MARGINAL / 64.57）⇒ τ_w 经防抖后**维持 0.15**（Δ=0.0002 < 0.02）。
🔴 发现独立缺陷：`ralph_loop.jq_feedback()` 写回 trajectory 时**只保留 jq_return/jq_share，
清空 jq_maxdd/jq_ic/jq_icir/jq_rating** ⇒ 富字段必须在 jq_feedback **之后**补写；这是
`trigger_d_plus.py` 的**顺序缺陷**，建议单独立项。
🔴 同时发现 D+ 欠账：全库 2298 条 trajectory 中 `d_plus_applied` 键**从未落盘**，
`jq_validated=True 且 not d_plus_applied` = **235 条**（最早可追至 2026-08-05）未被消化。

**P-20260918-001（龙虎榜三路资金分歧度）→ landed**：提案前提**部分高估**（昨日 P-022 已入库
novel_234/235/236 三条三路因子）。逐行核对后判定**真实瓶颈是数据通道断裂**：
`lhb_yz_*` 在 `factor_ic_computer.py` 全文出现 **0 次**（inst 6 / north 7）⇒ 该通道从未加载
⇒ 三条因子 eval 报 `NameError: name 'lhb_yz_net_buy'` **静默丢候选**；
`s5_joint_filter._build_series_context` 此前连 inst/north 都没有（该文件 L1381 有同族前科）；
`forge_translate.S5_KNOWN_TERMINALS` 自称三方对齐但实测**漂移**（缺 north/yz，fail-closed 假阴性）。
落地：D1 `factor_ic_computer` 新增 `YZ_INST_FIELDS`+游资 loader；D2 `s5_joint_filter` 新增 top_inst
三路 loader + `_LHB_TRI_COLS`(9 字段)；D3 `forge_translate` 白名单补齐 north/yz；D4 因子库新增
**novel_237~240** 四条分歧度/集中度变体（**仅上榜样本**，不用 fillna(0) 摊平）。
开关 `FUXI_LHB_TRI_CHANNEL`（默认 on / off=修正前）。
端到端 A/B（lookback=500，1,893,814 行）：OFF → 234/235/236 全 None；ON → uniq 67/172/123，
面板 `lhb_yz_net_buy` 非零 22,260 行；**对照因子 ctrl_inst_north 两模式 uniq=379 逐位不变**
⇒ 严格增量。新因子 uniq 1305 / 94557 / 1054 / 249 全部非近常量。
🔴 施工中自查抓到 **novel_238 首版是死因子**（uniq=1, std=0.0）：`pd.concat([宽表×3],axis=1).max(axis=1)`
会**塌掉整个截面**，改 `np.maximum` 元素级取最大后 uniq=94557。

**P-20260918-002（四道验收门）→ landed（收缩版）**：新增 `trade_gate_shadow.py`（成本敏感性影子，
零裁决）并接线 `ralph_loop` Phase E；`summary["trade_gate"]` 落读数；`data/trade_gate_shadow.json`。
判据**双轨**：`fragile_30bp`（多头方向）+ `high_drag_30bp`（**方向无关**，drag30/|gross|>0.5）。
🔴 自查：首版只有 fragile（要求 15bp 为正）⇒ 对 gross<0 的因子恒不触发 = fail-open，已补 drag 轨。
冒烟：fast_mom_2 turn=1.92 gross=+17.7% → 15bp +3.2% / 30bp **-11.3%** [脆断]；
slow_rev_60 gross=-10.0% → 仅[高拖累]（验证方向无关轨生效）；forge_gen4_r1 drag30=224%。
🔴 读数**窗口敏感**：同一公式 lookback 270→750，毛年化 -38.9%→+8.3%，期数 19→83 ⇒ 已记口径。
门候选 `cost_robust_30bp`（15bp/30bp 双档对账）**仅提案未生效**（样本仅 3 因子偏薄）。

**P-20260918-003（半衰期阈值复核）→ deferred（前提量纲错配）**：`classify()` 锚在
`half_life_days` = IC 在 lag 1-20 **交易日**的自衰减（日尺度），而提案证据「2-3 年→6-12 个月」
指**因子有效生命周期**（年尺度）⇒ **不是同一个量**，无映射关系。实测（n=15）：
`inf` **12/15**（λ≤1e-6，IC 在 1-20 内不衰减）、有限仅 2 条（89.3/100.7/112.2 日）、缺失 1、r2<0 1；
现行 `fast<5 日` 线触发 **0/15**（阈值实际已失效）；阈值扫描 <90日→7% / **<120日→13%** / ≥180日仍 13%。
⇒ 改动虽属单向收紧，但作用域只落 2 条有限读数，其余 12 条对任何有限阈值不响应 ⇒ 改它等于
**放大一个不可识别的度量（增加任意性）**，违反「去任意性才可翻默认」。仅落 `--half-life-report`
影子口径（`data/factor_half_life_report.json`），**阈值一行未动**。建议另立年尺度生命周期度量新提案。

**🔴 新发现（建议单独立项）**：全库 270 条因子中 **31 条（11.5%）**公式以 `.rank(pct=True, axis=1)`
结尾，而 pandas `Series.rank(axis=1)` 直接抛 `ValueError: No axis named 1 for object type Series`
⇒ 这些因子在 S5 **逐股 Series 上下文**下必然失败（novel_234/235/236 与本次 237~240 同式，属既有口径）。
本次沿用既有写法**未扩大差异**，但该轴语义跨层不一致是真缺陷。

**版本**：`version_bump --bump` → **v0.7.38**（2026-09-16 遗留 P-010）+
**v0.7.39**（2026-09-18 批次：P-001 / P-002 / P-004）⇒ **40 版本 / 140 落地提案**，
`--check` 退出码 **0**。测试产物 `_p001_seat_audit_20260918.py` / `_p001_e2e_20260918.py` /
`_p001_new_factors_20260918.py` / `_tg_test_20260918.py` / `_tg_hook_smoke_20260918.py` /
`_verify_dplus_20260918.py` / `_p001_003_audit_writeback_20260918.py` 已落 `research/factor_alchemy/`。

**坑位复现**：本会话 bash coreutils 仍损坏（`ls`/`dirname`/`cd`/`head`/`tail` 全不可用）
⇒ 一律绝对路径直跑 Python + 输出重定向到文件再读。`factor_decay_profile.py` 经 Edit 后为裸 LF
（**原文件本就是 LF**，`scripts/` 下 8 个抽样中 6 个是 LF），但两个新 JSON 产物首版误用
`newline=''` 落裸 LF ⇒ 已改 `newline='\\r\\n'` 并回读断言（两份现均为 CRLF / 裸 LF=0）。
"""

AUTO_TXT = """
### 2026-09-18 下午（交互追加，第 2 段）— 3 条提案审核落地 + D+ 闭环 + 死接线修复 ✅
- 用户指示「今天的3个提案开始审核落地，但 P-002 收缩为一个影子模块 + 一项有证据的成本敏感性门候选」，并回填 forge_gen4 JQ 读数。
- **D+ 闭环**：forge_gen4_gp_gen4_r1（2021-01-01~2026-07-30，5.6年）64.57% / Sharpe 0.266 / maxDD 29.49%
  ⇒ 按四档判定落 **JQ_MARGINAL**（收益>50 但 Sharpe<0.4）。单因子范围闭环（未批量跑 trigger_d_plus）：
  trajectory `traj_2561b7d3c712` + jq 富字段 + `d_plus_applied=True`；queue `jq_marginal`；
  lane_calibration 84 points；τ_w 维持 0.15。
  🔴 两个待办立项：① `jq_feedback()` 写回时**清空** jq_maxdd/jq_ic/jq_icir/jq_rating（富字段须事后补写）；
  ② D+ 欠账 **235 条**（`d_plus_applied` 键从未落盘）。
- **P-001 → landed**：真实瓶颈**不是缺因子**而是**通道断裂** —— `lhb_yz_*` 在 factor_ic_computer 出现 0 次
  ⇒ 昨日 P-022 的 novel_234/235/236 三条三路因子一直 **NameError 静默丢弃**。
  修 D1 factor_ic_computer(yz loader) / D2 s5_joint_filter(top_inst 三路 + 9 字段) / D3 forge_translate 白名单 /
  D4 新增 novel_237~240 分歧度变体（仅上榜样本）。开关 `FUXI_LHB_TRI_CHANNEL`(默认 on)。
  A/B：OFF→3条全 None；ON→uniq 67/172/123 + yz 非零 22,260 行；**对照 ctrl_inst_north uniq=379 逐位不变**
  ⇒ 严格增量。自查抓到 novel_238 首版死因子（concat+max(axis=1) 塌截面）已修。
- **P-002 → landed（收缩版）**：`trade_gate_shadow.py` + ralph_loop Phase E 接线；判据双轨
  fragile(多头) + **high_drag(方向无关)**；自查首版缺 drag 轨 = fail-open 已补。
  冒烟 fast_mom_2 15bp +3.2%→30bp **-11.3%** 脆断；读数窗口敏感（270→750 毛年化反号）。
- **P-003 → deferred**：前提**量纲错配**（本地是 IC 日尺度自衰减 vs 证据是年尺度生命周期）；
  n=15 中 inf 12/15，现行 <5日线触发 0/15，改 <120日只影响 2 条 ⇒ 放大不可识别度量=增加任意性。
  仅落 `--half-life-report` 影子（`data/factor_half_life_report.json`），阈值未动。
- **新发现（建议立项）**：全库 270 条中 **31 条(11.5%)** 用 `.rank(..., axis=1)`，
  在 S5 逐股 Series 上下文必抛 `ValueError`（既有口径缺陷）。
- 版本：`--bump` → **v0.7.38**（09-16 遗留）+ **v0.7.39**（09-18：P-001/002/004）⇒ **40 版本 / 140 落地**，`--check`=0。
- 坑位：bash coreutils 仍全废（绝对路径直跑）；新 JSON 产物首版 `newline=''` 落裸 LF 已改 `'\\r\\n'`。
"""

print("=" * 78)
append(DAILY, DAILY_TXT)
append(AUTO, AUTO_TXT)
print("=" * 78)
print("完成")
