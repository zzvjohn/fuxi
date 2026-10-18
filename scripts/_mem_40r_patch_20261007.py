# -*- coding: utf-8 -*-
"""MEMORY.md 40R 批增量补丁（同量压缩 + 新增 2 项判据）。

- 目标: .workbuddy/memory/MEMORY.md（CRLF-only）
- 新增: ①干旱行前沿-2.5% ②40R空转死锁+数据版本轴 ③状态行 C 结项
- 压缩: 13 处低损删减（信息在 details/RAG 可查）
- 纪律: dry-run 先算去空白字数（>3000 拒绝写盘）; --apply 备份+四项对账+回读。
"""
import os
import re
import sys

M = '.workbuddy/memory/MEMORY.md'
LIMIT = 3000

EDITS = [
    # ── 标题日期 ──
    ('# 伏羲量化记忆（精炼版 v0.7.130 · 10-06）',
     '# 伏羲量化记忆（精炼版 v0.7.130 · 10-07）'),
    # ── 状态行 ──
    ('v0.7.130｜10-06：🔴M6b/M4翻默认；M2维持pareto；BJ方案A落地。[mem:BE.19][mem:BE.22]',
     'v0.7.130｜10-06：🔴M6b/M4翻默认；M2维持pareto；BJ-A落地/C结项。[mem:BE.22]'),
    # ── 干旱行：加前沿 ──
    ('- **干旱**：三层归因(U段错配／经济失败／覆盖层阻断)；R5须重启时钟，不沿用旧端点。[mem:AQ.5][mem:BE.15]',
     '- **干旱**：三层归因(U段错配／经济失败／覆盖层阻断)；R5须重启时钟；10-07前沿-2.5%(新探索区)。[mem:BE.15][mem:BE.25]'),
    # ── BJ 裁剪行：压缩 + 新增 40R 行 ──
    ('- 🔴**BJ裁剪**：只减上市前段；head(top_n)递补⇒成员面不减（非泄漏）。[mem:BE.22]',
     '- 🔴**BJ裁剪**：只减上市前段；head(top_n)递补⇒成员面不减。[mem:BE.22]\n'
     '- 🔴**40R空转死锁**：空转轮tries不增⇒该臂永久锁定(24/40)；跨批<1pp差归数据版本。[mem:BE.25]'),
    # ── 压缩 A 批（原 8 处） ──
    ('41臂1/16过门⇒翻on非瓶颈；pulls零漂移。', '41臂1/16过门⇒翻on非瓶颈。'),
    ('ED=IC/breakeven；triage不授格；已翻enforce。', 'ED=IC/breakeven；已翻enforce。'),
    ('42阻断点全需公告；A5→A6=登记+新协议。', '42阻断点全需公告。'),
    ('E1valid20→19⇒niche不达标维持pareto。', 'E1valid20→19⇒维持pareto。'),
    ('看冻结基15bps配对净增量。', '看冻结基15bps净增量。'),
    ('🔴**M1校准层**：只读不裁决/授格/改阈值。', '🔴**M1校准层**：只读不裁决/改阈值。'),
    ('（漏参⇒error永不落；形参须AST对账）', '（漏参⇒error不落；形参AST对账）'),
    ('被全拒项与未发生项不得从输出消失', '被全拒/未发生项不得消失'),
    # ── 压缩 B 批（40R 追加 5 处） ──
    ('ICIR=0.370∈(均值,0.07)⇒limited_gain_close', 'ICIR=0.370⇒limited_gain_close'),
    ('回退须显式旧值；不设env≠旧默认。', '回退须显式旧值。'),
    ('10-06翻默认shadow_carry_forward', '翻默认shadow_carry_forward'),
    ('；基准身份用内容摘要。', '；基准身份用摘要。'),
    ('2025/2026已暴露非独立OOS', '2025/2026非独立OOS'),
]


def nws(t):
    return len(re.sub(r'\s', '', t))


def main():
    apply = '--apply' in sys.argv
    raw = open(M, 'rb').read()
    t = raw.decode('utf-8').replace('\r\n', '\n')
    before = nws(t)
    print('[dry] 起点去空白: %d | 余量 %d' % (before, LIMIT - before))
    for i, (old, new) in enumerate(EDITS, 1):
        cnt = t.count(old)
        assert cnt == 1, 'count=%d for %r' % (cnt, old[:50])
        d = nws(new) - nws(old)
        t = t.replace(old, new)
        print('  #%02d Δ=%+d  %s' % (i, d, old[:34].replace('\n', ' ')))
    after = nws(t)
    print('[dry] 终点去空白: %d | Δ=%+d | 余量 %d' % (after, after - before, LIMIT - after))
    if after > LIMIT:
        print('[ABORT] 超限: %d > %d' % (after, LIMIT))
        return 1
    if not apply:
        print('[dry] 未写盘（加 --apply 落盘）')
        return 0
    # 备份
    bak = M + '.bak_40r_20261007'
    if not os.path.exists(bak):
        with open(bak, 'wb') as fh:
            fh.write(raw)
    out = t.replace('\n', '\r\n').encode('utf-8')
    with open(M, 'wb') as fh:
        fh.write(out)
    # 四项对账 + 回读
    rb = open(M, 'rb').read()
    rt = rb.decode('utf-8')
    print('[post] bytes %d -> %d | crlf %d | bareLF %d | rrn %d | 去空白 %d'
          % (len(raw), len(rb), rb.count(b'\r\n'),
             rb.count(b'\n') - rb.count(b'\r\n'), rb.count(b'\r\r\n'), nws(rt)))
    nrm = rt.replace('\r\n', '\n')
    for old, new in EDITS:
        assert old not in nrm, 'old still present: %s' % old[:40]
        assert new in nrm, 'new missing: %s' % new[:40]
    print('[verify] 全部 %d 条回读断言通过（CRLF-only）' % len(EDITS))
    return 0


if __name__ == '__main__':
    sys.exit(main())
