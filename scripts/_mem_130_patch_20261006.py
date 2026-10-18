# -*- coding: utf-8 -*-
"""_mem_130_patch_20261006.py — MEMORY.md 刷新 v0.7.129 → v0.7.130（同量压缩）

背景: BJ universe 方案 A 落地（P-20261006-006 / v0.7.130）⇒ 开关清单新增
      FUXI_S5_BJ_UNIVERSE（默认 legacy）+ 新增判据「BJ 裁剪：只减上市前段；
      head(top_n) 递补 ⇒ 成员面不减（非泄漏）」。

纪律（fuxi-memory-rag 铁律 #4/#6/#15）:
  ① 先备份 → ② 写前算「去空白字符数」（≤2990 且余量≥10）→ ③ 二进制写入
  → ④ 四项对账（bytes / CRLF / 裸LF / rrn）+ 逐替换回读断言。
用法: python _mem_130_patch_20261006.py          # dry-run（只算不写）
      python _mem_130_patch_20261006.py --apply  # 落地
"""
import re
import shutil
import sys
from pathlib import Path

P = Path(r'E:\quant\.workbuddy\memory\MEMORY.md')
BAK = Path(r'E:\quant\.workbuddy\memory\MEMORY.md.bak_p130_20261006')

# (old, new) —— 每条必须唯一命中；纯 ASCII/中文混合，不含 shell 敏感字符
EDITS = [
    # ── 头部 + 状态行 ──
    ('# 伏羲量化记忆（精炼版 v0.7.129 · 10-06）',
     '# 伏羲量化记忆（精炼版 v0.7.130 · 10-06）'),
    ('v0.7.129｜10-06：🔴M6b/M4翻默认；M2维持pareto；三通道补齐。[mem:BE.16][mem:BE.19]',
     'v0.7.130｜10-06：🔴M6b/M4翻默认；M2维持pareto；BJ方案A落地。[mem:BE.19][mem:BE.22]'),
    # ── 同量压缩（语义判据零删除；读数/细节留 RAG） ──
    ('；三通道未入日更链=停更根因。[mem:AD3][mem:BE.16]',
     '；三通道未入日更链=停更根因。[mem:AD3]'),
    ('增量对照只改一轴（叶级只碰1特征，改名=2⇒拒；未写新键拒）。',
     '增量对照只改一轴（叶级1特征；改名=2⇒拒）。'),
    ('S5=0≠经济失败（144对拍142阻断）。[mem:BA]',
     'S5=0≠经济失败。[mem:BA]'),
    ('；triage不授格；10-06翻enforce。[mem:BE.1][mem:BE.19]',
     '；triage不授格；已翻enforce。[mem:BE.1]'),
    ('（空库=legacy）；fail-closed须包住「读」',
     '；fail-closed须包住「读」'),
    ('；pulls零漂移shadow。[mem:AO.3]',
     '；pulls零漂移。[mem:AO.3]'),
    ('⇒niche_mix不达标维持pareto。',
     '⇒niche不达标维持pareto。'),
    ('；数学错硬拒、机制锁JQ；静态质量≠S5。',
     '；数学错硬拒；静态质量≠S5。'),
    ('09-23/24：禁vwap代理；rank须截面轴。[mem:D5]',
     '禁vwap代理；rank须截面轴。[mem:D5]'),
    # ── 新增（BJ 方案 A） ──
    # 🔴 新增行内换行必须写 '\r\n'（写 '\n' 会引入裸 LF，2026-10-06 首跑实测 bareLF=1）
    ('- 🔴**M1校准层**：只读不裁决/授格/改阈值。[mem:BC]',
     '- 🔴**BJ裁剪**：只减上市前段；head(top_n)递补⇒成员面不减（非泄漏）。[mem:BE.22]\r\n'
     '- 🔴**M1校准层**：只读不裁决/授格/改阈值。[mem:BC]'),
    ('｜MOTIF_MEMORY=off。订单流盘后不得同日建仓',
     '｜MOTIF_MEMORY=off｜BJ_UNIVERSE=legacy。订单流盘后不得同日建仓'),
]


def nws(t):
    return len(re.sub(r'\s', '', t))


def main():
    apply = '--apply' in sys.argv
    raw = P.read_bytes()
    t = raw.decode('utf-8')
    before = nws(t)

    for old, new in EDITS:
        n = t.count(old)
        if n != 1:
            print('[FAIL] 命中 %d 次（须为 1）: %s' % (n, old[:48]))
            return 2
        t = t.replace(old, new)

    after = nws(t)
    print('[plan] 替换 %d 条 | 去空白 %d -> %d (Δ%+d) | 余量 %d'
          % (len(EDITS), before, after, after - before, 3000 - after))
    if after > 2990:
        print('[FAIL] 超限（>2990），拒绝写入')
        return 3

    if not apply:
        print('[dry-run] 未写盘。加 --apply 落地。')
        return 0

    if not BAK.exists():
        shutil.copy2(P, BAK)
        print('[backup] %s' % BAK)
    blob = t.encode('utf-8')
    with open(P, 'wb') as f:
        f.write(blob)

    # ── 四项对账 + 回读断言 ──
    rb = P.read_bytes()
    rt = rb.decode('utf-8')
    bare = rb.count(b'\n') - rb.count(b'\r\n')
    print('[post] bytes %d -> %d | crlf %d | bareLF %d | rrn %d | 去空白 %d'
          % (len(raw), len(rb), rb.count(b'\r\n'), bare, rb.count(b'\r\r\n'), nws(rt)))
    assert bare == 0 and rb.count(b'\r\r\n') == 0, '行尾异常 (bareLF=%d rrn=%d)' % (bare, rb.count(b'\r\r\n'))
    nrm = rt.replace('\r\n', '\n')
    for old, new in EDITS:
        o, n = old.replace('\r\n', '\n'), new.replace('\r\n', '\n')
        # 插入型条目（new 包含 old）⇒ old 保留 1 次；替换型 ⇒ old 必须消失
        want_old = 1 if o in n else 0
        assert nrm.count(o) == want_old, 'old 次数 %d (期望 %d): %s' % (nrm.count(o), want_old, o[:30])
        assert nrm.count(n) == 1, 'new 非唯一/缺失: %s' % n[:30]
    print('[verify] 全部 %d 条回读断言通过（行尾 CRLF-only）' % len(EDITS))
    return 0


if __name__ == '__main__':
    sys.exit(main())
