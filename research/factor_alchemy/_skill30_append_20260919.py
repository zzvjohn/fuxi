# -*- coding: utf-8 -*-
"""2026-09-19 技能更新: fuxi-dead-wiring-audit 追加 §30 + 扩 description 与触发词。

⚠️ 绕开 Edit 工具 (会把 CRLF 整体转 LF)。全部断言 + 写后回读。
"""
import io
import shutil
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SK = Path(r'C:/Users/Zhong/.workbuddy/skills/fuxi-dead-wiring-audit/SKILL.md')
FRAG = Path(r'e:/quant/research/factor_alchemy/_frag_skill30_20260919.md')

raw0 = SK.read_bytes()
crlf0 = raw0.count(b'\r\n')
bare0 = raw0.count(b'\n') - crlf0
print(f"原始: {len(raw0)} B  CRLF={crlf0} 裸LF={bare0}")
assert bare0 == 0, "原文件应为纯 CRLF"

bak = SK.with_suffix('.md.bak_20260919_e2e')
shutil.copy2(SK, bak)
print(f"备份: {bak.name}")

txt = raw0.decode('utf-8')

# ── 1. description 补一段 ──
ANCHOR1 = "「顺序」类症状先普查否证再定修法以防假修复、标记失效导致重放使 `+= 1` 计数器虚增）。"
assert txt.count(ANCHOR1) == 1, f"anchor1 命中 {txt.count(ANCHOR1)} 次"
NEW1 = (ANCHOR1[:-1] +
        "；并含**端到端回读验证**（临时路径只证「写入当刻」算假验证，须跑一轮真实入口后回读；"
        "判「是否真走 `_save()`」看 `to_dict` 独有键频与行首缩进而**非** mtime；"
        "**留痕字段同样会被白名单抹除**，须把全部直连 JSON 写方一并排查）。")
txt = txt.replace(ANCHOR1, NEW1, 1)

# ── 2. 触发词补一行 ──
ANCHOR2 = "数据面实验, 真实命名空间, 命名空间exec, 合成面板, 逻辑层通过, 同名方法, 分段取错, 探针漏接收者, 验证脚本自身的bug, 先排除自身缺陷, AST解析, 跨右括号。"
assert txt.count(ANCHOR2) == 1, f"anchor2 命中 {txt.count(ANCHOR2)} 次"
NEW2 = ANCHOR2 + (
    "\r\n  端到端回读验证, 假验证, 只验证写入当刻, 跑一轮流水线再回读, 真实验证, lessons键频, "
    "独有键频, 写方二, 第二个写方, indent翻转, 缩进随写方变, 体积摆动, freq_audit, _freq_audit, "
    "留痕字段被抹, 审计留痕丢失, 直连写方, 键频1/N, 跑前先量键频。")
txt = txt.replace(ANCHOR2, NEW2, 1)

# ── 3. 末尾追加 §30 ──
body = FRAG.read_text(encoding='utf-8').replace('\r\n', '\n').replace('\r', '\n')
body = body.replace('\n', '\r\n')
if not txt.endswith('\r\n'):
    txt += '\r\n'
txt = txt + body

out = txt.encode('utf-8')
SK.write_bytes(out)

# ── 写后断言 ──
raw1 = SK.read_bytes()
crlf1 = raw1.count(b'\r\n')
bare1 = raw1.count(b'\n') - crlf1
t1 = raw1.decode('utf-8')
checks = [
    ("纯 CRLF 保持", bare1 == 0, f"CRLF={crlf1} 裸LF={bare1}"),
    ("体积增长", len(raw1) > len(raw0), f"{len(raw0)} -> {len(raw1)} (+{len(raw1)-len(raw0)})"),
    ("旧内容为前缀 (纯追加 + 前置替换区保留)",
     t1.startswith(raw0.decode('utf-8').split('## 三十')[0][:2000]),
     "首 2000 字保留"),
    ("§30 标题存在且唯一",
     t1.count("## 三十、端到端回读验证") == 1, f"{t1.count('## 三十、端到端回读验证')} 次"),
    ("description 补句存在", "并含**端到端回读验证**" in t1, ""),
    ("触发词补行存在", "端到端回读验证, 假验证" in t1, ""),
    ("frontmatter 未破", t1.startswith("---\r\nname: fuxi-dead-wiring-audit"), ""),
    ("agent_created 保留", "agent_created: true" in t1, ""),
]
ok = True
for name, cond, det in checks:
    print(f"   [{'PASS' if cond else '*** FAIL'}] {name}  {det}")
    ok &= cond

# frontmatter 结构: 第二个 --- 位置
lines = t1.split('\r\n')
print(f"   行数 = {len(lines)}  (原 {len(raw0.decode('utf-8').splitlines())})")
print(f"   第 2 个 '---' 行号 = {[i+1 for i,l in enumerate(lines[:60]) if l.strip()=='---']}")
print()
print(f"技能更新: {'ALL PASS' if ok else 'HAS FAILURE'}")
sys.exit(0 if ok else 1)
