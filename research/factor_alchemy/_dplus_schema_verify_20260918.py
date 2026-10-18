# -*- coding: utf-8 -*-
"""D+ schema 修复的决定性验证 (2026-09-18, P-20260918-005)

三层证据:
  T1 dataclass 往返     : 11 个新字段 to_dict→from_dict 逐字段相等
  T2 多次 _save 存活    : 走完 8 个 save 点后回读, 字段仍在 (这是原缺陷的杀手锏)
  T3 jq_feedback 真调用 : 把**真实** jq_feedback 函数体绑到 stub 实例上,
                          全部写入重定向 temp; 断言 d_plus_applied/d_plus_source 落盘

零副作用哨兵: 全程对真实 data/ 下 *.json/*.csv 做 (mtime,size) 前后比对。
判据锚在实现 (回读磁盘字节), 不锚"写入当刻的内存对象" —— 只验证写入当刻 = 假验证。
"""
import os
import sys
import json
import shutil
import tempfile
import glob as _glob
from pathlib import Path

FA = r'e:/quant/research/factor_alchemy'
DATA = r'e:/quant/data'
sys.path.insert(0, FA)
sys.path.insert(0, r'e:/quant/scripts')
sys.path.insert(0, r'e:/quant')

import pandas as pd  # noqa: E402

RICH = ['jq_maxdd', 'jq_ic', 'jq_icir', 'jq_rating', 'jq_plus_ic_pct',
        'jq_backtest_period', 'jq_notes', 'd_plus_applied', 'd_plus_at',
        'd_plus_source', 'lessons']


def snapshot():
    snap = {}
    for pat in ('*.json', '*.csv'):
        for fp in _glob.glob(os.path.join(DATA, pat)):
            try:
                st = os.stat(fp)
                snap[os.path.basename(fp)] = (st.st_mtime_ns, st.st_size)
            except OSError:
                pass
    return snap


def diff(before, after):
    changed = []
    for k in set(before) | set(after):
        if before.get(k) != after.get(k):
            changed.append((k, before.get(k), after.get(k)))
    return changed


# ── T1 ──────────────────────────────────────────────────────────
def t1_roundtrip():
    from trajectory_logger import MiningTrajectory, TrajectoryStep
    print('=' * 74)
    print('T1  dataclass 往返 (to_dict → from_dict 逐字段相等)')
    print('=' * 74)
    t = MiningTrajectory(trajectory_id='traj_T1', paradigm='P1', seed_factor='f1')
    t.jq_validated = True
    t.jq_return = 64.57
    t.jq_sharpe = 0.266
    t.jq_maxdd = 29.49
    t.jq_ic = -0.0312
    t.jq_icir = -0.145
    t.jq_rating = 'MARGINAL'
    t.jq_plus_ic_pct = 52.3
    t.jq_backtest_period = '2021-01-01~2021-07-30'
    t.jq_notes = 'root_cause_demo'
    t.d_plus_applied = True
    t.d_plus_at = '2026-09-18T15:21:12'
    t.d_plus_source = 'jq_feedback'
    t.lessons = [{'lesson': 'JQ验证: 64.6%/0.27', 'timestamp': 'x'}]
    d = t.to_dict()
    missing = [k for k in RICH if k not in d]
    print('  to_dict 缺失字段: %s' % (missing or '无'))
    t2 = MiningTrajectory.from_dict(d)
    bad = []
    for k in RICH:
        a, b = getattr(t, k), getattr(t2, k)
        if a != b:
            bad.append((k, a, b))
    print('  from_dict 往返不一致: %s' % (bad or '无'))
    print('  to_dict 键数: %d' % len(d))
    ok = (not missing) and (not bad)
    print('  => T1 %s' % ('PASS' if ok else 'FAIL'))
    return ok


# ── T2 ──────────────────────────────────────────────────────────
def t2_multi_save():
    from trajectory_logger import TrajectoryLogger
    print()
    print('=' * 74)
    print('T2  多次 _save() 存活 (走完全部 save 点后从磁盘回读)')
    print('=' * 74)
    tmp = Path(tempfile.mkdtemp(prefix='trajT2_'))
    p = tmp / 'trajectory_log.json'
    lg = TrajectoryLogger(log_path=p)
    tid = lg.start_trajectory(paradigm='P1', seed_factor='f_demo').trajectory_id
    lg.log_hypothesis(tid, 'hypo', score=0.5)                 # save 1
    lg.log_expression(tid, 'close_p/volume_p', score=0.5)     # save 2
    lg.log_backtest(tid, score=0.3, ic=-0.0312, icir=-0.145,
                    sharpe=0.266, maxdd=29.49)                # save 3
    lg.log_lesson(tid, 'JQ验证: 64.6%/0.27')                  # save 4
    lg.log_evaluation(tid, factor_name='f_demo', stage='S5',
                      passed=False, ic=0.01)                  # save 5
    lg.finalize(tid, outcome='JQ_MARGINAL', jq_return=64.57, jq_sharpe=0.266,
                jq_maxdd=29.49, jq_ic=-0.0312, jq_icir=-0.145,
                jq_rating='MARGINAL', jq_plus_ic_pct=52.3,
                jq_backtest_period='2021-01-01~2021-07-30',
                jq_notes='rc', mark_d_plus=True,
                d_plus_source='jq_feedback')                  # save 6
    # 🔴 关键: D+ 之后**再触发一次全量重写** (原缺陷就是在这里抹掉标记)
    lg.log_evaluation(tid, factor_name='f_demo', stage='S5_PASS', passed=True,
                      ic=0.02)                                # save 7 — 复现抹除点
    lg2 = TrajectoryLogger(log_path=p)
    lg2.finalize(tid, outcome='JQ_MARGINAL', jq_return=64.57,
                 jq_sharpe=0.266)                             # save 8 — 复现抹除点
    # 回读**磁盘字节** (不信内存对象)
    raw = json.load(open(p, encoding='utf-8'))
    rec = [r for r in raw['trajectories'] if r['trajectory_id'] == tid][0]
    print('  磁盘记录键数: %d' % len(rec))
    lost = []
    expect = {'jq_maxdd': 29.49, 'jq_ic': -0.0312, 'jq_icir': -0.145,
              'jq_rating': 'MARGINAL', 'jq_plus_ic_pct': 52.3,
              'jq_backtest_period': '2021-01-01~2021-07-30', 'jq_notes': 'rc',
              'd_plus_applied': True, 'd_plus_at': None,
              'd_plus_source': 'jq_feedback'}
    for k, v in expect.items():
        got = rec.get(k)
        mark = 'OK ' if (v is None and got) or got == v else 'LOST'
        if mark == 'LOST':
            lost.append((k, v, got))
        print('    %-22s %-10s got=%s' % (k, mark, got))
    n_lessons = len(rec.get('lessons') or [])
    print('    %-22s %-10s got=%d 条' % ('lessons', 'OK ' if n_lessons else 'LOST', n_lessons))
    if not n_lessons:
        lost.append(('lessons', '>=1', 0))
    ok = not lost
    print('  => T2 %s' % ('PASS' if ok else 'FAIL'))
    return ok


# ── T3 ──────────────────────────────────────────────────────────
def t3_jq_feedback():
    from trajectory_logger import TrajectoryLogger
    from experience_memory import ExperienceMemory
    import ralph_loop as RL
    print()
    print('=' * 74)
    print('T3  jq_feedback() 真调用 (真实函数体 + stub 实例 + temp 写入)')
    print('=' * 74)
    tmp = Path(tempfile.mkdtemp(prefix='trajT3_'))

    # 关闭会写真实库的旁路 (只放行被验证的轨迹写路径)
    import seed_injector
    _real_seed = getattr(seed_injector, '_add_champion_factor', None)
    seed_injector._add_champion_factor = lambda **kw: None
    print('  [stub] seed_injector._add_champion_factor -> no-op')

    class _MabStub:
        directions = []

        def resolve_arms(self, *a, **k):
            return []

        def save_state(self):
            print('  [stub] mab.save_state() 被调用 (应为 0 次, streak=0 时应短路)')

    class _Stub:
        jq_feedback = RL.RalphLoop.jq_feedback
        _jq_outcome = staticmethod(RL.RalphLoop._jq_outcome)
        _lookup_traj_freq = RL.RalphLoop._lookup_traj_freq

        def __init__(self):
            self.memory = ExperienceMemory(memory_path=tmp / 'experience_memory.json')
            self.traj_logger = TrajectoryLogger(log_path=tmp / 'trajectory_log.json')
            self.fsa = None
            self.mab = _MabStub()
            self.data_dir = tmp
            self._last_evo_trajectories = None
            self.stats = {}

    st = _Stub()
    # 预置一条轨迹, 让 name 能匹配上 seed_factor (走 matched 分支)
    tg = st.traj_logger
    tg.start_trajectory(paradigm='动量反转', seed_factor='T3_demo_factor')
    res = st.jq_feedback(
        jq_backtest_result={
            'batch_id': 'T3', 'composite_return': 30.0,
            'composite_sharpe': 0.2, 'composite_maxdd': 12.0,
            'factors': [{
                'factor_name': 'T3_demo_factor',
                'formula': 'close_p.pct_change(5)', 'hypothesis': 'h',
                'paradigm': '动量反转', 'category': 'momentum',
                'operators_used': ['pct_change'], 'local_ic': 0.02,
                'local_icir': 0.3, 'jq_return': 30.0, 'jq_sharpe': 0.2,
                'jq_maxdd': 12.0, 'jq_ic': 0.0135, 'jq_icir': 0.42,
                'natural_freq': 'daily',
                'jq_composite_contribution': 'neutral', 'root_cause': 'rc_T3',
            }],
        },
        # streak=0 → Step 3.5 短路, 不碰 MAB
        evo_trajectories=[{'paradigm': '__t3__', 'streak': 0}],
        update_lane=False,
    )
    print('  trajectories_updated = %s' % res.get('trajectories_updated'))
    raw = json.load(open(tmp / 'trajectory_log.json', encoding='utf-8'))
    rec = raw['trajectories'][0]
    checks = {
        'd_plus_applied': rec.get('d_plus_applied') is True,
        'd_plus_source': rec.get('d_plus_source') == 'jq_feedback',
        'd_plus_at': bool(rec.get('d_plus_at')),
        'jq_ic': rec.get('jq_ic') == 0.0135,
        'jq_icir': rec.get('jq_icir') == 0.42,
        'jq_maxdd': rec.get('jq_maxdd') == 12.0,
        'jq_notes': rec.get('jq_notes') == 'rc_T3',
        'jq_validated': rec.get('jq_validated') is True,
        'lessons': len(rec.get('lessons') or []) >= 1,
    }
    for k, v in checks.items():
        print('    %-18s %s  (%s)' % (k, 'OK ' if v else 'FAIL', rec.get(k)))
    # 🔴 再触发一次全量重写, 证明标记不会被抹 (原缺陷的复现点)
    tg.log_evaluation(rec['trajectory_id'], factor_name='T3_demo_factor',
                      stage='S5', passed=False)
    raw2 = json.load(open(tmp / 'trajectory_log.json', encoding='utf-8'))
    rec2 = raw2['trajectories'][0]
    survive = rec2.get('d_plus_applied') is True and rec2.get('jq_ic') == 0.0135
    print('    再触发一次 _save 后标记存活: %s' % ('OK' if survive else 'FAIL'))
    if _real_seed is not None:
        seed_injector._add_champion_factor = _real_seed
    ok = all(checks.values()) and survive
    print('  => T3 %s' % ('PASS' if ok else 'FAIL'))
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


def main():
    b = snapshot()
    print('真实 data/ 哨兵快照: %d 个文件' % len(b))
    print()
    r1 = t1_roundtrip()
    r2 = t2_multi_save()
    r3 = t3_jq_feedback()
    a = snapshot()
    ch = diff(b, a)
    print()
    print('=' * 74)
    print('零副作用哨兵')
    print('=' * 74)
    if ch:
        print('  ⚠️ 真实 data/ 发生 %d 处变化:' % len(ch))
        for k, x, y in ch:
            print('    %-46s %s -> %s' % (k, x, y))
    else:
        print('  ✅ 真实 data/ 下 json/csv 全部 (mtime,size) 未变 — 零副作用')
    print()
    print('T1=%s  T2=%s  T3=%s  side_effect_free=%s' % (r1, r2, r3, not ch))
    print('OVERALL: %s' % ('PASS' if (r1 and r2 and r3 and not ch) else 'FAIL'))


if __name__ == '__main__':
    main()
