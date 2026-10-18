# -*- coding: utf-8 -*-
"""MEMORY.md 压缩：3101 -> <=3000，保持 CRLF。"""
import io

P = r"e:\quant\.workbuddy\memory\MEMORY.md"

EDITS = [
    # 1) 外链行精简
    ("> 外链：`_fuxi_landing_details.md`(D1~D8+附录 A~C)｜`CHANGELOG.md`·`data/version_registry.json`｜`kuaidao.md`｜每日 `YYYY-MM-DD.md`",
     "> 外链：`_fuxi_landing_details.md`｜`CHANGELOG.md`·`data/version_registry.json`｜`kuaidao.md`｜每日 `YYYY-MM-DD.md`"),
    # 2) 文案生成行精简
    ("- **文案生成**：非格式化串多写 `%%` ⇒ **不报错**（原样输出）；格式化串漏转义 ⇒ `ValueError`。⇒ 内建扫 `%%` 自检。",
     "- **文案生成**：非格式化串多写 `%%` ⇒ **不报错**（原样输出）；格式化串漏转义 ⇒ `ValueError`。⇒ 内建扫 `%%`。"),
    # 3) 归因行精简
    ("⇒ 直读 CSV 取裁决窗。单条 validate 实测 **71s**（充裕）/~180s（压力）。",
     "⇒ 直读 CSV 取裁决窗。单条 validate 实测 **71s**/~180s。"),
    # 4) JQ 文件名行 -> 外链（欠账记录，非静默坑）
    ("- **JQ 文件名 86.8% 不可信**：68 个 `jq_s5_pass_*` 真通过仅 **8** ⇒ 未回改 = **欠账**。\n", ""),
    # 5) 翻不翻判据行精简
    ("→ 第 2 批全池：60 条 legacy/p0 各 60/60、翻盘 PASS 0/0（可检验 54）⇒ 放行。",
     "→ 第 2 批全池：legacy/p0 各 60/60、翻盘 PASS 0/0（可检验 54）⇒ 放行。"),
    # 6) 补压（第二轮，需再省 ~40）
    ("（JQ `jq_s5_pass_*` 86.8% 名不副实 = 欠账）", "（JQ `jq_s5_pass_*` 86.8% 名不副实）"),
    ("- **解释器**：研究=`C:/Users/Zhong/.workbuddy/binaries/python/envs/default/Scripts/python.exe`；",
     "- **解释器**：研究=`~/.workbuddy/binaries/python/envs/default/Scripts/python.exe`；"),
    ("裸 `python` 缺库 ⇒ **一律绝对路径**。", "裸 `python` 缺库 ⇒ **绝对路径**。"),
    ("锚在**实现**（真实数据列）", "锚在**实现**（数据列）"),
    ("（crowded 大单通道消融后净拖累 t=−5.72）", "（crowded 大单通道消融净拖累 t=−5.72）"),
    ("- **读基线前先打印顶层 key 探结构**", "- **读基线先打印顶层 key 探结构**"),
    # 7) 第三轮（再省 ~15，目标 <=3000）
    ("# 量化工作区长期记忆 (2026-09-17 v38 · 索引版)", "# 量化记忆 (2026-09-17 v38 · 索引版)"),
    ("> ⚠️ 限额 3000 字符 ⇒ 只放「违反即**静默**出错」的铁律，详情外链。",
     "> ⚠️ 限额 3000 ⇒ 只放「违反即**静默**出错」铁律，详情外链。"),
    ("⇒ Write 落盘后由 Python 追加；勿 heredoc；写完回读自检。",
     "⇒ Write 落盘后 Python 追加；勿 heredoc；写完回读。"),
]

with io.open(P, "rb") as f:
    b = f.read()
t = b.decode("utf-8").replace("\r\n", "\n")   # 归一为 LF 再改，写回时统一转 CRLF
before = len(t) + t.count("\n")               # 还原为 CRLF 口径的字符数
for old, new in EDITS:
    if old not in t:
        print("[SKIP] 已应用过: %r" % old[:36])   # 幂等：重复执行不报错
        continue
    t = t.replace(old, new, 1)

# 保留 JQ 欠账一行信息（极简）挂到外链行尾
anchor = "｜每日 `YYYY-MM-DD.md`"
suffix = "（JQ `jq_s5_pass_*` 86.8% 名不副实）"
if suffix not in t:
    t = t.replace(anchor + "（JQ `jq_s5_pass_*` 86.8% 名不副实 = 欠账）", anchor + suffix, 1)
if suffix not in t:
    t = t.replace(anchor, anchor + suffix, 1)

t = t.replace("\r\n", "\n").replace("\n", "\r\n")
with io.open(P, "wb") as f:
    f.write(t.encode("utf-8"))

b2 = t.encode("utf-8")
bare = b2.replace(b"\r\n", b"").count(b"\n")
print("chars %d -> %d  余量=%d" % (before, len(t), 3000 - len(t)))
print("CRLF=%d bare_LF=%d" % (b2.count(b"\r\n"), bare))
for k in ["v38", "数据窗三坑", "SPARSE_DIAG=on", "32 版本", "225 条目", "71s",
          "flags 须用裁决窗", "指纹须含窗口", "86.8%"]:
    print("  ", k, "=", k in t)
