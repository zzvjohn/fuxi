# -*- coding: utf-8 -*-
"""2026-09-19 记账写回: 报告/日志/详情三处追加。带可证断言。

行尾约定(实测): 9-18 报告=纯LF | 2026-09-19.md 新建=CRLF | _fuxi_landing_details.md=CRLF
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(r'e:/quant')
FRAG = ROOT / "research" / "factor_alchemy"

JOBS = [
    # (fragment, target, eol, create?)
    (FRAG / "_frag_report_dplus_e2e_20260919.md",
     ROOT / "reports" / "提案落地_20260918_D+字段根治与轴语义修复.md", "\n", False),
    (FRAG / "_frag_log_20260919.md",
     ROOT / ".workbuddy" / "memory" / "2026-09-19.md", "\r\n", True),
    (FRAG / "_frag_details_M_20260919.md",
     ROOT / ".workbuddy" / "memory" / "_fuxi_landing_details.md", "\r\n", False),
]

ok_all = True
for frag, tgt, eol, create in JOBS:
    print("=" * 70)
    print(f"{frag.name}  ->  {tgt.name}   (eol={'CRLF' if eol == chr(13)+chr(10) else 'LF'})")
    ftxt = frag.read_text(encoding='utf-8')
    # 归一化片段行尾
    norm = ftxt.replace('\r\n', '\n').replace('\r', '\n')
    body = norm.replace('\n', eol)
    if not body.endswith(eol):
        body += eol

    old = b''
    if tgt.exists() and not create:
        old = tgt.read_bytes()
    elif tgt.exists() and create:
        print("   *** 目标已存在, 拒绝覆盖 (append-only)")
        ok_all = False
        continue

    new = old + body.encode('utf-8')
    before = tgt.stat().st_size if tgt.exists() else 0

    tgt.parent.mkdir(parents=True, exist_ok=True)
    with open(tgt, 'wb') as f:
        f.write(new)

    # ── 断言 ──
    raw = tgt.read_bytes()
    a1 = raw.startswith(old) if old else True          # 纯追加
    crlf = raw.count(b'\r\n')
    bare = raw.count(b'\n') - crlf
    a2 = (bare == 0) if eol == "\r\n" else (crlf == 0)
    a3 = norm.split('\n')[1].strip()[:0] == ''  # placeholder
    tail = raw.decode('utf-8')

    print(f"   追加 {len(body.encode('utf-8'))} B   (文件 {before} -> {len(raw)} B)")
    print(f"   [{'PASS' if a1 else '*** FAIL'}] 纯追加 (旧内容为新内容前缀)")
    print(f"   [{'PASS' if a2 else '*** FAIL'}] 行尾: CRLF={crlf} 裸LF={bare}")
    # 关键串留存
    key = {
        "_frag_report_dplus_e2e_20260919.md": "端到端回读验证（2026-09-19 收口",
        "_frag_log_20260919.md": "# 2026-09-19 研究轨迹",
        "_frag_details_M_20260919.md": "## 附录 M — D+ 字段根治的端到端回读验证",
    }[frag.name]
    a4 = key in tail
    print(f"   [{'PASS' if a4 else '*** FAIL'}] 关键串存在: {key!r}")
    # 只出现 1 次 (防重复追加)
    a5 = tail.count(key) == 1
    print(f"   [{'PASS' if a5 else '*** FAIL'}] 关键串出现次数 = {tail.count(key)} (应为 1)")
    ok_all &= (a1 and a2 and a4 and a5)

print()
print("=" * 70)
print(f"记账写回: {'ALL PASS' if ok_all else 'HAS FAILURE'}")
print("=" * 70)
sys.exit(0 if ok_all else 1)
