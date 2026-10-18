# -*- coding: utf-8 -*-
"""2026-09-19 状态实测: 不信任历史日志文本, 直接从磁盘复核 1/2/3/5 项。

只读, 不写任何文件。
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

TRAJ = ROOT / "data" / "trajectory_log.json"

print("=" * 72)
print("A. trajectory_log.json 结构实测 (只读)")
print("=" * 72)
with open(TRAJ, 'r', encoding='utf-8') as f:
    data = json.load(f)

trajs = data.get("trajectories", [])
n = len(trajs)
print(f"顶层 key: {sorted(data.keys())}")
print(f"总轨迹数: {n}")

# 全量键频
keyfreq = Counter()
for t in trajs:
    for k in t.keys():
        keyfreq[k] += 1

RICH9 = ["jq_maxdd", "jq_ic", "jq_icir", "jq_rating",
         "jq_plus_ic_pct", "jq_backtest_period", "jq_notes",
         "d_plus_applied", "d_plus_at"]
EXTRA = ["d_plus_source", "lessons"]

print("\n-- 原先「1/N 被抹签名」的 9 个富字段 当前键频 --")
for k in RICH9 + EXTRA:
    print(f"   {k:24s} {keyfreq.get(k, 0):5d} / {n}")

# 白名单键 (22) 是否仍全覆盖
WHITE22 = ["trajectory_id", "paradigm", "seed_factor", "generation", "phase",
           "hypothesis", "expression", "code", "backtest", "overall_score",
           "final_ic", "final_icir", "final_outcome", "jq_validated",
           "jq_return", "jq_sharpe", "natural_freq", "interpretation",
           "created_at", "completed_at", "high_reward_segments", "_eval_log"]
missing_white = [k for k in WHITE22 if keyfreq.get(k, 0) != n]
print(f"\n-- 原白名单 22 键覆盖: {'全部 100%' if not missing_white else missing_white}")

print("\n-- 全库键集合 (%d 个) --" % len(keyfreq))
print("   " + ", ".join(sorted(keyfreq.keys())))

# D+ 欠账
jq_val = sum(1 for t in trajs if t.get("jq_validated"))
d_plus = sum(1 for t in trajs if t.get("d_plus_applied"))
debt = sum(1 for t in trajs if t.get("jq_validated") and not t.get("d_plus_applied"))
src = Counter(t.get("d_plus_source", "") for t in trajs if t.get("d_plus_applied"))
print(f"\n-- D+ 账 --")
print(f"   jq_validated          = {jq_val}")
print(f"   d_plus_applied        = {d_plus}")
print(f"   🔴 欠账 (val & !applied) = {debt}")
print(f"   d_plus_source 分布      = {dict(src)}")

# 富字段真实覆盖 (披露纪律: 必须与「已标记」同时说)
rich_any = sum(1 for t in trajs
               if any(t.get(k) not in (None, "", [], {}) for k in
                      ["jq_maxdd", "jq_ic", "jq_icir", "jq_rating",
                       "jq_plus_ic_pct", "jq_backtest_period", "jq_notes"]))
print(f"   🔴 7 个富字段任一非空覆盖 = {rich_any} / {n}")

print()
print("=" * 72)
print("B. dataclass 字段集 vs 库键集合")
print("=" * 72)
from trajectory_logger import MiningTrajectory
import dataclasses
dc_fields = {f.name for f in dataclasses.fields(MiningTrajectory)}
lib_keys = set(keyfreq.keys())
print(f"   dataclass 字段数 = {len(dc_fields)}")
print(f"   库内键数         = {len(lib_keys)}")
print(f"   库键 - dataclass = {sorted(lib_keys - dc_fields)}")
print(f"   dataclass - 库键 = {sorted(dc_fields - lib_keys)}")
print(f"   完全对齐: {lib_keys == dc_fields}")

print()
print("=" * 72)
print("C. 轴语义修复: _AXIS1_CALL_RE 实测")
print("=" * 72)
from s5_joint_filter import _AXIS1_CALL_RE
probes = [
    ("x.rank(pct=True, axis=1)", True),
    ("x.rank(pct=True, axis=1, ascending=False)", False),  # 既有边界: 嵌套括号
    ("x.rank(axis = 1)", True),
    ("x.sub(s, axis=0)", False),
    ("x.rank(axis=0)", False),
]
for s, exp in probes:
    got = bool(_AXIS1_CALL_RE.search(s))
    tag = "OK " if got == exp else "*** 差异"
    print(f"   {tag} {s!r:52s} hit={got}")
print("   (第二行预期 False = 已披露的 `[^()]*` 跨括号边界)")

print()
print("=" * 72)
print("D. jq_feedback 打标接线实测 (AST, 不 import)")
print("=" * 72)
import ast
src = (ROOT / "research" / "factor_alchemy" / "ralph_loop.py").read_text(encoding='utf-8', errors='replace')
tree = ast.parse(src)


def find_func(node, name):
    for ch in ast.walk(node):
        if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) and ch.name == name:
            return ch
    return None


fb = find_func(tree, "jq_feedback")
if fb is None:
    print("   *** 未找到 jq_feedback")
else:
    src_fb = ast.get_source_segment(src, fb)
    print(f"   jq_feedback @ line {fb.lineno}, {len(src_fb.splitlines())} 行")
    print(f"   mark_d_plus=True      出现 {src_fb.count('mark_d_plus=True')} 次")
    print(f"   d_plus_source 出现     {src_fb.count('d_plus_source')} 次")
    print(f"   finalize(...) 调用数   {src_fb.count('finalize(')}")
    for k in ["jq_maxdd", "jq_ic", "jq_icir", "jq_rating", "jq_plus_ic_pct",
              "jq_backtest_period", "jq_notes"]:
        print(f"     透传 {k:20s} {src_fb.count(k)} 次")
print("\nDONE (只读)")
