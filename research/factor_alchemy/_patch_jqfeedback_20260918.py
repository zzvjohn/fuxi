# -*- coding: utf-8 -*-
"""补丁: ralph_loop.jq_feedback() 传递 JQ 富字段 + 打 D+ 标记 (2026-09-18)

背景: jq_feedback() 是 D+ 蒸馏本体, 但
  (1) 它只把 ic/icir/sharpe/maxdd 写进 backtest.meta, 轨迹级 jq_* 恒 None;
  (2) 它**从不**打 d_plus_applied 标记 ⇒ 直接调 jq_feedback 的一次性脚本
      产出的条目从出生起就是欠账 (全库 grep 已证 ralph_loop.py 无该标识符)。
本补丁把两处 finalize() 调用补齐富字段并置 mark_d_plus=True。

CRLF 保真: 读 → LF 归一 → 替换 → 整体写回 CRLF (文件本就 0 裸 LF, 无损)。
"""
import io
import py_compile

P = r'e:\quant\research\factor_alchemy\ralph_loop.py'

OLD1 = (
    "                        traj_id,\n"
    "                        outcome=self._jq_outcome(jq_ret, jq_sharpe),\n"
    "                        jq_return=jq_ret,\n"
    "                        jq_sharpe=jq_sharpe,\n"
    "                    )"
)
NEW1 = (
    "                        traj_id,\n"
    "                        outcome=self._jq_outcome(jq_ret, jq_sharpe),\n"
    "                        jq_return=jq_ret,\n"
    "                        jq_sharpe=jq_sharpe,\n"
    "                        # 2026-09-18 (P-20260918-005): 富字段落到轨迹一级字段,\n"
    "                        # 而非只进 backtest.meta; 并**就地打 D+ 标记**, 消除\n"
    "                        # 「直接调 jq_feedback 的条目天生欠账」这一独立欠账源。\n"
    "                        jq_maxdd=f.get('jq_maxdd'),\n"
    "                        jq_ic=f.get('jq_ic'),\n"
    "                        jq_icir=f.get('jq_icir'),\n"
    "                        jq_rating=f.get('jq_rating'),\n"
    "                        jq_plus_ic_pct=f.get('jq_plus_ic_pct'),\n"
    "                        jq_backtest_period=f.get('jq_backtest_period'),\n"
    "                        jq_notes=(f.get('root_cause') or None),\n"
    "                        mark_d_plus=True,\n"
    "                        d_plus_source='jq_feedback',\n"
    "                    )"
)

OLD2 = (
    "                    traj.trajectory_id,\n"
    "                    outcome=self._jq_outcome(jq_ret, jq_sharpe),\n"
    "                    jq_return=jq_ret,\n"
    "                    jq_sharpe=jq_sharpe,\n"
    "                )"
)
NEW2 = (
    "                    traj.trajectory_id,\n"
    "                    outcome=self._jq_outcome(jq_ret, jq_sharpe),\n"
    "                    jq_return=jq_ret,\n"
    "                    jq_sharpe=jq_sharpe,\n"
    "                    # 2026-09-18 (P-20260918-005): 同上 — 新建轨迹分支同步补富字段+打标\n"
    "                    jq_maxdd=f.get('jq_maxdd'),\n"
    "                    jq_ic=f.get('jq_ic'),\n"
    "                    jq_icir=f.get('jq_icir'),\n"
    "                    jq_rating=f.get('jq_rating'),\n"
    "                    jq_plus_ic_pct=f.get('jq_plus_ic_pct'),\n"
    "                    jq_backtest_period=f.get('jq_backtest_period'),\n"
    "                    jq_notes=(f.get('root_cause') or None),\n"
    "                    mark_d_plus=True,\n"
    "                    d_plus_source='jq_feedback',\n"
    "                )"
)

OLD3 = '        print(f"  [Trajectory] 更新了 {trajectories_updated} 条轨迹")'
NEW3 = ('        print(f"  [Trajectory] 更新了 {trajectories_updated} 条轨迹 '
        '(含 d_plus_applied 标记, source=jq_feedback)")')


def main():
    raw = open(P, 'rb').read()
    s = raw.decode('utf-8')
    n_crlf_before = s.count('\r\n')
    n_lf_before = s.count('\n') - n_crlf_before
    print('before: bytes=%d CRLF=%d bareLF=%d' % (len(raw), n_crlf_before, n_lf_before))

    lf = s.replace('\r\n', '\n')
    for i, (o, n) in enumerate([(OLD1, NEW1), (OLD2, NEW2), (OLD3, NEW3)], 1):
        cnt = lf.count(o)
        print('  pattern %d occurrences: %d' % (i, cnt))
        if cnt != 1:
            raise SystemExit('ABORT: pattern %d expected exactly 1, got %d' % (i, cnt))
        lf = lf.replace(o, n, 1)

    out = lf.replace('\n', '\r\n')
    open(P, 'wb').write(out.encode('utf-8'))

    raw2 = open(P, 'rb').read()
    s2 = raw2.decode('utf-8')
    n_crlf = s2.count('\r\n')
    n_lf = s2.count('\n') - n_crlf
    print('after : bytes=%d CRLF=%d bareLF=%d' % (len(raw2), n_crlf, n_lf))
    assert n_lf == 0, 'bare LF introduced!'
    assert n_crlf > n_crlf_before

    py_compile.compile(P, doraise=True)
    print('py_compile OK')

    # 回读复核关键串
    for probe in ["mark_d_plus=True,", "d_plus_source='jq_feedback',",
                  "jq_maxdd=f.get('jq_maxdd'),", 'd_plus_applied 标记, source=jq_feedback']:
        print('  probe %-46s -> %d' % (probe[:46], s2.count(probe)))
    assert s2.count('mark_d_plus=True,') == 2
    print('PATCH OK')


if __name__ == '__main__':
    main()
