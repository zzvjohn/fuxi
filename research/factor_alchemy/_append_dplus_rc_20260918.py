# -*- coding: utf-8 -*-
"""2026-09-18 晚: D+ 欠账根因会话的记账写入。
- _fuxi_landing_details.md : CRLF, 追加 K 节 (完整根因)
- 2026-09-18.md            : CRLF, 追加日志
- automation memory.md     : LF,   追加执行摘要
- MEMORY.md                : CRLF, 追加 1 行指针
"""
import io, os

DET  = r"e:\quant\.workbuddy\memory\_fuxi_landing_details.md"
DAILY= r"e:\quant\.workbuddy\memory\2026-09-18.md"
AUTO = r"e:\quant\.workbuddy\memory\automations\automation-1786597632320\memory.md"
MEM  = r"e:\quant\.workbuddy\memory\MEMORY.md"

det = """
## K. D+ 欠账 235 条的根因：`to_dict()` 白名单静默抹除（2026-09-18 晚查清）

**问题**: 为什么 `jq_validated=True 且 not d_plus_applied` 会累积到 235 条？
**答案**: 不是「没人跑 trigger_d_plus」，是**标记存不住** + **一条注入路径根本不写标记**。
⇒ 235 ≈ **100% 的历史 jq_validated**（236 条中唯一 1 条是 9-18 手工直写的），
   那不是待办队列深度，是**结构性伪影**。

### K.1 决定性证据（键频普查）

`trajectory_log.json` / 2298 条：
- 22 个 `to_dict()` 白名单键 → **2298 / 2298** 全覆盖
- `d_plus_applied` / `d_plus_at` / `jq_maxdd` / `jq_ic` / `jq_icir` / `jq_rating` /
  `jq_plus_ic_pct` / `jq_backtest_period` / `jq_notes` → **各恰好 1 条**
- 那 1 条 = `traj_2561b7d3c712`（`d_plus_at=2026-09-18T15:21:12`）—— 当天直写 JSON 且此后未再跑流水线。

🔴 **「1/N」= 被抹签名**：非白名单键恰好出现在 1 条 ⇒ 不是「很少用」，是「每次都被抹、只剩最后一次写入」。

### K.2 机理：两个写入者，schema 不对称

| 写入者 | 写什么 | 后果 |
| --- | --- | --- |
| `trigger_d_plus.run()` → `save_trajectories()` | 原始 dict 直写，富字段**保得住** | 当次落盘 ✓ |
| `TrajectoryLogger._save()` → `to_dict()` | **22 键白名单**重建 | 🔴 **抹掉全部非白名单键** |

`_save()` **8 个调用点**（`start_trajectory` / `log_hypothesis` / `log_expression` /
`log_backtest`×2 / L451 / `log_lesson` / `finalize`）⇒ 每跑一轮 Ralph 流水线触发多次全量重写。
⇒ **`d_plus_applied` 存活上限 = 到下一次 `_save()` 为止**。

### K.3 第二条独立成因：直连注入路径从不写标记

`jq_feedback()` 函数体内**零处**引用 `d_plus_applied`（全库 grep：`ralph_loop.py` **不在**
含该标识符的文件清单内）。只有 `trigger_d_plus.run()` 的 `mark_d_plus_applied()` 会标记。
⇒ 直接调 `ralph.jq_feedback()` 的一次性脚本（如 `_dplus_3factors_20260829.py`，
注释明写"不设 d_plus_applied — 留给 trigger_d_plus 标记"）产出的条目**从出生起就是欠账**。

### K.4 零副作用复现（`_dplus_wipe_repro_20260918.py`）

```
输入 31 键 → to_dict() 输出 22 键
丢弃: d_plus_applied, d_plus_at, jq_maxdd, jq_ic, jq_icir, jq_rating, jq_notes
保留的 jq 键: jq_validated, jq_return, jq_sharpe（白名单里只有这 3 个）
经真实 TrajectoryLogger._save() 往返后回读: 7 个富字段全部 False
```
手法: `TrajectoryLogger.__new__` 绕过 `_load()`、`path` 指向 temp 文件，**绝不碰 data/**。

### K.5 危害：重放导致计数器虚增（数据没坏，权重被灌水）

`form_from_jq()` **有 name 级去重**（`jq_failure::<name>` / `jq_warning::<name>` /
`jq_confirmed::<paradigm>`）⇒ **不新建重复条目** ✓；但去重分支里带 `+= 1`：
- `forbidden_directions[].failed_attempts += 1`、`warning_directions[].failed_attempts += 1` ⇒ 虚增
- `success_templates[].occurrence_count += 1` ⇒ 虚增（影响模式置信度）
- `attempts[].lessons.append(...)` ⇒ 列表膨胀（逐轮追加同一条 `JQ反馈: ...`）

### K.6 🔴 更正：原判「`jq_feedback()` 写回顺序缺陷」是误诊

原判「先 finalize 后补写 ⇒ 富字段被清空 ⇒ 调换写入顺序」**错误**：
真机理是 **schema 缺列**，与顺序**无关**。调换顺序后**下一次 `_save()` 照样抹掉** ⇒ **假修复**。
🔴 通用判据: 凡「A 写入被 B 覆盖」，先问 **B 是不是在重建整个对象**；是 ⇒ schema 问题，不是顺序问题。

### K.7 正解（待裁决，未执行）

1. 9 个富字段提升为 `MiningTrajectory` dataclass 字段，补进 `to_dict()` **与** `from_dict()`
   （🔴 **必须双向**：只补 `to_dict()`，下一轮 `_load()` 又丢）
2. `jq_feedback()` 补标记能力，消除 K.3 的直连路径欠账源
3. **先扩 schema 再回填** —— 顺序反了会白做（9-18 的富字段就是这么白写的）
4. 落地后**跑一轮主流水线再回读**才算验证（只验证写入当刻 = 假验证）

⚠️ 单纯跑一次全量 `trigger_d_plus` 只会在抹除前**短暂清零**，并顺手把 K.5 计数器推高一轮。
"""

daily = """
## 2026-09-18 晚（第四段）— D+ 欠账 235 条根因查清（含对前次定性的更正）

**用户提问**: 「D+ 欠账为什么会有这么多以及 jq_feedback() 写回顺序缺陷又是因为什么导致」

**结论**: 两者是**同一个根因** —— `MiningTrajectory.to_dict()` 的**白名单序列化**。

**决定性证据（键频普查）**: `trajectory_log.json` 2298 条中，
22 个白名单键 **2298/2298** 全覆盖；`d_plus_applied`/`d_plus_at`/`jq_maxdd`/`jq_ic`/`jq_icir`/
`jq_rating` 等 9 个**非白名单键各恰好 1 条**（= 当天直写、此后未跑流水线的那条 `traj_2561b7d3c712`）。
🔴 **「1/N」= 被抹签名**。

**机理**: `_save()`（**8 个调用点**）经 `to_dict()` 全量重写 ⇒ 非白名单键存活不过下一轮
⇒ 标记被逐轮抹除 ⇒ 欠账 **235 ≈ 100% 历史 jq_validated**（不是队列深度，是结构性伪影）。

**第二条成因**: `jq_feedback()` 函数体**零处**引用 `d_plus_applied`（全库 grep 已证
`ralph_loop.py` 不在含该标识符文件清单内）⇒ 直连注入路径**根本不写标记**。

**🔴 更正前次误诊**: 报告 §4 #2 原判「`jq_feedback()` 写回**顺序**缺陷 ⇒ 修 `trigger_d_plus` 顺序」
是**误诊 + 假修复**（真机理是 schema 缺列，调顺序修不好，下轮照样抹）。报告已改，新增 §6 详述。

**危害量化**: `form_from_jq()` **有 name 级去重** ⇒ 不新建重复条目；但去重分支带 `+= 1`
⇒ `failed_attempts` / `occurrence_count` **虚增**、`attempts[].lessons` **膨胀**。数据没坏，权重被灌水。

**产物**:
- `research/factor_alchemy/_dplus_rootcause_20260918.py`（键频普查，只读）
- `research/factor_alchemy/_dplus_wipe_repro_20260918.py`（零副作用复现，temp 路径）
- `reports/提案落地_20260918_三条审核与D+闭环.md` §4 已更正 + 新增 §6（根因全文）
- 技能 `fuxi-dead-wiring-audit` **+§25「白名单序列化」**（90657 → 96627 字节，CRLF 保真）

**正解（未执行，待裁决）**: ① 扩 dataclass schema（`to_dict`+`from_dict` **双向**）
② `jq_feedback()` 补标记能力 ③ **先扩 schema 再回填** ④ 落地后跑一轮流水线再回读。
"""

auto = """
### 2026-09-18 晚（交互追加，第 4 段）— D+ 欠账根因查清 + 更正前次误诊 ✅
- 用户问「D+ 欠账为什么这么多 / jq_feedback() 顺序缺陷成因」→ 查清为**同一根因**:
  `MiningTrajectory.to_dict()` **白名单序列化**（22 键）。
- **决定性证据**: 全库 2298 条中白名单键 2298/2298；非白名单键（d_plus_applied/d_plus_at/
  jq_maxdd/jq_ic/jq_icir/jq_rating 等 9 个）**各恰好 1 条** ⇒ 🔴「1/N」= 被抹签名。
- **机理**: `_save()` 8 个调用点经 to_dict 全量重写 ⇒ 标记存活不过下一轮 ⇒
  欠账 235 ≈ **100% 历史 jq_validated**（结构性伪影，非队列深度）。
  **第二条成因**: `jq_feedback()` 零处引用 d_plus_applied ⇒ 直连注入路径从不标记。
- 🔴 **更正**: 报告 §4 #2 原判「写回顺序缺陷」是**误诊 + 假修复**，已改写并新增 §6 详述。
- **危害**: 下游有 name 级去重（不重复建条目），但 `+= 1` 计数器（failed_attempts /
  occurrence_count）**虚增** + lessons 膨胀 ⇒ 数据没坏、权重被灌水。
- **正解（未执行）**: 扩 schema（to_dict+from_dict 双向）→ jq_feedback 补标记 → **先扩再回填**
  → 跑一轮流水线后回读验证。
- **技能**: `fuxi-dead-wiring-audit` **+§25「白名单序列化」**（+5970 字节，CRLF 保真，已备份）。
- 产物: `_dplus_rootcause_20260918.py`(键频普查) + `_dplus_wipe_repro_20260918.py`(零副作用复现)。
"""

def append(path, text, eol):
    b = open(path, "rb").read()
    bom = b.startswith(b"\xef\xbb\xbf")
    s = b.decode("utf-8-sig" if bom else "utf-8").replace("\r\n", "\n")
    if not s.endswith("\n"):
        s += "\n"
    out = s + text.strip("\n").replace("\r\n", "\n") + "\n"
    data = out.replace("\n", eol).encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    open(path, "wb").write(data)
    nb = open(path, "rb").read()
    print("%-32s bytes=%6d CRLF=%4d bareLF=%d" % (
        os.path.basename(path), len(nb), nb.count(b"\r\n"),
        nb.count(b"\n") - nb.count(b"\r\n")))

append(DET,   det,   "\r\n")
append(DAILY, daily, "\r\n")
append(AUTO,  auto,  "\n")

# MEMORY.md: 只加 1 行指针 (保持索引精简)
b = open(MEM, "rb").read()
s = b.decode("utf-8").replace("\r\n", "\n")
anchor = "- **文案**：非格式化串多写 `%%` ⇒ 不报错；格式化串漏转义 ⇒ `ValueError`。"
assert s.count(anchor) == 1, "MEMORY.md 锚点 %d" % s.count(anchor)
line = ("- 🔴 **非 schema 键存不住**：`_save()` 经 `to_dict()` **白名单全量重写** ⇒ 外部直写的键"
        "（`d_plus_applied`/`jq_maxdd`/`jq_ic`…）**存活不过下一轮**；键频 **「1/N」= 被抹签名**；"
        "修须 `to_dict`+`from_dict` **双向补**。「写回顺序」属误诊。详见 `_fuxi_landing_details.md` §K。")
s = s.replace(anchor, anchor + "\n" + line)
open(MEM, "wb").write(s.replace("\n", "\r\n").encode("utf-8"))
nb = open(MEM, "rb").read()
print("%-32s bytes=%6d CRLF=%4d bareLF=%d" % ("MEMORY.md", len(nb),
      nb.count(b"\r\n"), nb.count(b"\n") - nb.count(b"\r\n")))
print("DONE")
