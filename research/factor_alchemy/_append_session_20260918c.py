# -*- coding: utf-8 -*-
"""追加 2026-09-18 第二段收尾记录：技能沉淀 + 状态复核。
daily log 用 CRLF，automation memory 用 LF（各按文件自身约定）。"""
import io, os

DAILY = r"e:\quant\.workbuddy\memory\2026-09-18.md"
AUTO  = r"e:\quant\.workbuddy\memory\automations\automation-1786597632320\memory.md"

daily_note = """
## 2026-09-18 傍晚（第三段）— 技能沉淀 §24 + 收尾状态复核

**背景**: 承接当日「3 条提案审核落地 + D+ 闭环」会话，把本轮新验证有效的方法论固化进技能。

**技能更新 `fuxi-dead-wiring-audit`（+§24，10 个子节）**
- 文件: `C:\\Users\\Zhong\\.workbuddy\\skills\\fuxi-dead-wiring-audit\\SKILL.md`，82345 → **90657 字节**，
  CRLF 保真（1240 CRLF / **裸 LF=0**），备份 `SKILL.md.bak_20260918_155751`。
- 手法: 按技能自身 §23.5 纪律，**绕开 Edit 工具**（会把 CRLF 整体转 LF），改用
  Write 落盘片段 + Python 组装写回 + 写后断言行尾与关键串存在。
- 新增内容:
  1. §24.1 死通道的**三源字段集合求差**（库 / 求值器 / 翻译器）—— 库里有条目 ≠ 通道存在；
     入库前须对**全部求值路径**各跑一次 + 各 grep 一次计数
  2. §24.2 端到端 A/B 的**对照因子纪律** —— 四步法，第 4 步「ctrl 因子 uniq 逐位不变」（379=379）
     才是「严格增量、零副作用」的证据
  3. §24.3 新因子入库必 eval 看 **uniq/std** 排死因子（`novel_238` 首版 uniq=1/std=0.0 塌截面）
     + 死因按 uniq 分档表
  4. §24.4 事件型因子的样本口径: 只取上榜样本，**禁 fillna(0)**（同类因子 ICIR 0.342 vs 0.277）
  5. §24.5 **单边判据 = fail-open 漏报**（`fragile_30bp` 对 gross<0 因子恒不触发）；
     凡方向敏感判据必须问「反方向样本落到哪里」
  6. §24.6 影子读数的**窗口依赖**必须落 caliber（lookback 270→750 ⇒ 毛年化 −38.9% 反号 +8.3%）
  7. §24.7 前提**量纲错配**的提案 ⇒ `deferred` 而非调参（P-003 半衰期：年尺度生命周期 vs 日尺度自衰减）
  8. §24.8 `.rank(axis=1)` **跨层轴语义**缺陷类（全库 31/270 = 11.5%）
  9. §24.9 D+ 闭环 `jq_feedback()` **写回顺序**缺陷（静默清空 jq_maxdd/ic/icir/rating）
  10. §24.10 处置顺序（承接 §23.6）
- frontmatter 同步扩充: description 增补一段 + 触发词新增 28 个（死通道 / 对照因子 / 严格增量 /
  uniq / 死因子 / 塌截面 / 单边判据 / 漏反方向样本 / 窗口依赖 / 量纲错配 / 跨层轴 等）。

**收尾状态复核（全部一致）**
- `version_bump.py --check` **EXIT=0**；`data/version_registry.json` **40 版本**，末条 `v0.7.39`（2026-09-18）。
- `data/stage1_optimization_proposals.json` **240 条目**；
  P-20260918-001 landed/`landed_dead_channel_fix`｜002 landed/`landed_shadow_default_on`｜
  003 deferred/`deferred_premise_flawed`｜004 landed。
- 临时片段文件已清理；测试脚本 `_skill24_append_20260918.py` 按项目惯例保留为证据。

**待用户裁决的 4 个独立缺陷（已在当日报告 §4 列明，未立项未执行）**
1. D+ 欠账 235 条（`jq_validated=True 且 not d_plus_applied`，最早 2026-08-05）—— 批量回填会改
   Memory / hard_forbidden / MAB，须授权
2. `jq_feedback()` 写回顺序缺陷（清空 jq_maxdd/jq_ic/jq_icir/jq_rating）
3. 全库 31 条 (11.5%) `.rank(axis=1)` 跨层轴缺陷 —— 建议单独立项统一轴语义
4. `cost_robust_30bp` 门候选（仅提案未生效，需影子跑若干轮后用 `n_high_drag/n_checked` 比例定阈值）
"""

auto_note = """
### 2026-09-18 傍晚（交互追加，第 3 段）— 技能沉淀 §24 + 收尾复核 ✅
- 承接当日「3 提案审核落地 + D+ 闭环」，把本轮新方法论固化进 `fuxi-dead-wiring-audit` 技能。
- **SKILL.md 82345 → 90657 字节（+§24 共 10 子节）**，CRLF 保真（1240 CRLF / 裸 LF=0），已备份。
  关键新增: 死通道三源字段核对 / 端到端 A/B **对照因子「逐位不变」**证严格增量 / 新因子 eval 看 uniq·std 排死因子 /
  事件型因子禁 fillna(0) / 单边判据=fail-open 漏报 / 影子读数窗口依赖落 caliber / 前提量纲错配应 deferred /
  `.rank(axis=1)` 跨层轴缺陷类 / D+ 写回顺序缺陷。frontmatter description + 28 个触发词已同步。
- 手法: 按技能自身 §23.5 纪律**绕开 Edit**（会静默转 LF），用 Write 片段 + Python 组装 + 写后断言。
- 复核: `version_bump --check` EXIT=0；**40 版本 / v0.7.39**；**240 提案**；P-001/002/004 landed、P-003 deferred。
- 遗留（未立项，待用户裁决）: ① D+ 欠账 235 条 ② `jq_feedback()` 清空富字段 ③ 31 条 axis=1 跨层轴
  ④ `cost_robust_30bp` 门候选定阈值。详见 `reports/提案落地_20260918_三条审核与D+闭环.md` §4。
"""

def append(path, text, eol):
    b = open(path, "rb").read()
    has_bom = b.startswith(b"\xef\xbb\xbf")
    s = b.decode("utf-8-sig" if has_bom else "utf-8").replace("\r\n", "\n")
    if not s.endswith("\n"):
        s += "\n"
    add = text.strip("\n").replace("\r\n", "\n") + "\n"
    out = s + add
    data = out.replace("\n", eol).encode("utf-8")
    if has_bom:
        data = b"\xef\xbb\xbf" + data
    open(path, "wb").write(data)
    nb = open(path, "rb").read()
    bare = nb.count(b"\n") - nb.count(b"\r\n")
    print("%-46s bytes=%d CRLF=%d bareLF=%d" % (os.path.basename(path), len(nb), nb.count(b"\r\n"), bare))

append(DAILY, daily_note, "\r\n")
append(AUTO,  auto_note,  "\n")
print("DONE")
