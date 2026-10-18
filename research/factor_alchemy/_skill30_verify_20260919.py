# -*- coding: utf-8 -*-
"""2026-09-19 技能更新的**正确**验证: 逆变换往返 (而不是"前缀"这种窄断言)。

判据: 把新文本中的 §30 段剥掉 + 把两处补丁逆转 -> 必须**逐字符等于**备份原文。
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SK = Path(r'C:/Users/Zhong/.workbuddy/skills/fuxi-dead-wiring-audit/SKILL.md')
BAK = SK.with_suffix('.md.bak_20260919_e2e')

old = BAK.read_bytes().decode('utf-8')
new = SK.read_bytes().decode('utf-8')

HDR = "## 三十、端到端回读验证"

ANCHOR1 = "「顺序」类症状先普查否证再定修法以防假修复、标记失效导致重放使 `+= 1` 计数器虚增）。"
NEW1 = (ANCHOR1[:-1] +
        "；并含**端到端回读验证**（临时路径只证「写入当刻」算假验证，须跑一轮真实入口后回读；"
        "判「是否真走 `_save()`」看 `to_dict` 独有键频与行首缩进而**非** mtime；"
        "**留痕字段同样会被白名单抹除**，须把全部直连 JSON 写方一并排查）。")

ANCHOR2 = "数据面实验, 真实命名空间, 命名空间exec, 合成面板, 逻辑层通过, 同名方法, 分段取错, 探针漏接收者, 验证脚本自身的bug, 先排除自身缺陷, AST解析, 跨右括号。"
NEW2 = ANCHOR2 + (
    "\r\n  端到端回读验证, 假验证, 只验证写入当刻, 跑一轮流水线再回读, 真实验证, lessons键频, "
    "独有键频, 写方二, 第二个写方, indent翻转, 缩进随写方变, 体积摆动, freq_audit, _freq_audit, "
    "留痕字段被抹, 审计留痕丢失, 直连写方, 键频1/N, 跑前先量键频。")

print(f"备份 {len(old)} 字符 / 新 {len(new)} 字符  差 {len(new)-len(old)}")

# ── 逆变换 ──
checks = []

idx = new.find(HDR)
checks.append(("§30 段起点可定位且唯一", idx > 0 and new.count(HDR) == 1, f"idx={idx}, count={new.count(HDR)}"))
core = new[:idx].rstrip('\r\n')

checks.append(("description 补丁可逆且唯一", core.count(NEW1) == 1, f"{core.count(NEW1)} 次"))
rev = core.replace(NEW1, ANCHOR1, 1) if core.count(NEW1) == 1 else core

checks.append(("触发词补丁可逆且唯一", rev.count(NEW2) == 1, f"{rev.count(NEW2)} 次"))
if rev.count(NEW2) == 1:
    rev = rev.replace(NEW2, ANCHOR2, 1)

rev = rev.rstrip('\r\n')
old_c = old.rstrip('\r\n')
checks.append(("逆变换后逐字符等于原文（含行尾）", rev == old_c,
               f"len {len(rev)} vs {len(old_c)}; 首个差异 idx=" +
               (str(next((i for i, (a, b) in enumerate(zip(rev, old_c)) if a != b), 'N/A'))
                if rev != old_c else '无')))

# ── 另一组独立断言: 唯一实际差异就是这 3 处 ──
import difflib
diffs = [l for l in difflib.unified_diff(old_c.split('\r\n'), (core.rstrip('\r\n') + '\r\n' + new[idx:]).split('\r\n'),
                                         lineterm='', n=0) if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))]
n_plus = sum(1 for l in diffs if l.startswith('+'))
n_minus = sum(1 for l in diffs if l.startswith('-'))
print(f"\n统一 diff 行数: +{n_plus} / -{n_minus}  (预期: -3/+3 再加 §30 新增行)")
print("   --- 非 §30 的新增行 ---")
for l in diffs:
    if l.startswith('+') and not l.startswith('+++'):
        if any(k in l for k in ('端到端回读验证', '假验证', 'lessons键频', '留痕字段同样会被白名单抹除')):
            print("   ", l[:150])

# ── 其余结构断言 ──
front = new.split('\r\n')[0:54]
checks.append(("frontmatter 边界完好 (1 与 53 行)", new.split('\r\n')[0] == '---'
               and new.split('\r\n')[52] == '---', f"L53={new.split(chr(13)+chr(10))[52]!r}"))
checks.append(("纯 CRLF 无裸 LF", new.count('\n') - new.count('\r\n') == 0,
               f"CRLF={new.count(chr(13)+chr(10))} 裸LF={new.count(chr(10))-new.count(chr(13)+chr(10))}"))
checks.append(("agent_created 保留", 'agent_created: true' in new, ""))
checks.append(("§30.1~30.6 六个子节齐", all(f"### 30.{i} " in new for i in range(1, 7)),
               str([f"30.{i}" for i in range(1, 7) if f"### 30.{i} " not in new])))

print()
ok = True
for n, c, d in checks:
    print(f"   [{'PASS' if c else '*** FAIL'}] {n}  {d}")
    ok &= c
print()
print(f"OVERALL: {'ALL PASS' if ok else 'HAS FAILURE'}")
sys.exit(0 if ok else 1)
