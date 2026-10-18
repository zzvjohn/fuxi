# -*- coding: utf-8 -*-
"""方案 A patch 落盘后精确对账 + 接线复核 (只读)。

1) 用字节口径重算 5 处替换的期望 delta, 与实测 delta 对账
2) 关键标记出现次数复核
3) 与备份做逐处 diff 定位 (确认只有 5 处变动)
"""
import difflib
import hashlib

SRC = r"E:\quant\research\factor_alchemy\s5_joint_filter.py"
BAK = r"E:\quant\reports\fuxi_v1_bj_universe_20261006\pre_bjcut\s5_joint_filter.py.bak"

cur = open(SRC, "r", encoding="utf-8", newline="").read()
old = open(BAK, "r", encoding="utf-8", newline="").read()

pre_b = len(old.encode("utf-8"))
post_b = len(cur.encode("utf-8"))

print("[1] bytes: %d -> %d (delta %+d)" % (pre_b, post_b, post_b - pre_b))
print("    crlf : %d -> %d (delta %+d)" % (old.count("\r\n"), cur.count("\r\n"),
                                           cur.count("\r\n") - old.count("\r\n")))
print("    bareLF: %d -> %d | rrn: %d -> %d" % (
    old.count("\n") - old.count("\r\n"), cur.count("\n") - cur.count("\r\n"),
    old.count("\r\r\n"), cur.count("\r\r\n")))

# ── 逐行 diff 定位变动块 ──
old_lines = old.split("\r\n")
cur_lines = cur.split("\r\n")
sm = difflib.SequenceMatcher(a=old_lines, b=cur_lines, autojunk=False)
blocks = []
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag != "equal":
        blocks.append((tag, i1, i2, j1, j2))
print("\n[2] changed blocks: %d" % len(blocks))
tot_add = tot_del = 0
for tag, i1, i2, j1, j2 in blocks:
    n_add, n_del = j2 - j1, i2 - i1
    tot_add += n_add
    tot_del += n_del
    head = (cur_lines[j1] if n_add else old_lines[i1])[:70]
    print("    %-8s old[%d:%d] new[%d:%d]  -%d/+%d | %s" % (tag, i1, i2, j1, j2, n_del, n_add, head))
print("    total: -%d/+%d lines" % (tot_del, tot_add))

# ── 关键标记 ──
print("\n[3] markers (count in patched file):")
for mk in ("FUXI_S5_BJ_UNIVERSE", "def _s5_bj_universe", "def _bj_listing_date_map",
           "def _s5_bj_cut_pre_listing", "_s5_bj_cut_pre_listing(price)",
           '"bj_universe": _bj', "_bj = _s5_bj_universe()",
           "bj_listing_date_overlay.csv", "_BJ_LISTING_DATE_CACHE"):
    print("    %-32s %d" % (mk, cur.count(mk)))

# ── 与原始 anchor 对照: 确认 legacy 路径形状未变 (除新增块) ──
print("\n[4] sha256: old=%s" % hashlib.sha256(old.encode("utf-8")).hexdigest()[:16])
print("           new=%s" % hashlib.sha256(cur.encode("utf-8")).hexdigest()[:16])

# ── 5) 确认 price 面板挂点周围上下文 ──
idx = cur.find("_s5_bj_cut_pre_listing(price)")
if idx > 0:
    seg = cur[max(0, idx - 700):idx + 220]
    print("\n[5] P5 context:\n" + "\n".join("    " + l for l in seg.split("\r\n")))
