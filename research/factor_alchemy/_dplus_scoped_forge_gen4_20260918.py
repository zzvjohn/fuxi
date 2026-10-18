# -*- coding: utf-8 -*-
"""定向 D+ 闭环: forge_gen4_gp_gen4_r1_ts_rank 单因子 JQ 反馈注入。

背景
====
用户回测反馈 (2026-09-18):
    因子: forge_gen4_gp_gen4_r1_ts_rank(ts_mean(ts_std(log(ts...
    区间: 2021-01-01 ~ 2026-07-30 (日志实际记录到 2026-07-27 week=285, 5.6 年)
    累计收益 64.57% | Sharpe 0.266 | maxDD 29.49%
    日志: E:\\log\\forge_gen4_gp_gen4_r1_ts_rank.txt

四档判定 (ralph_loop.RalphLoop._jq_outcome, L5091):
    ret < -20 或 sharpe < -0.5        -> JQ_FAILED
    ret > 50 且 sharpe > 0.4          -> JQ_PASSED
    ret < 0                           -> JQ_WEAK_NEGATIVE
    else                              -> JQ_MARGINAL
=> 64.57 > 50 但 sharpe 0.266 < 0.4  ==> **JQ_MARGINAL**
   (正是该守卫设计要防的「仅收益>50 就误标 PASSED」情形)

🔴 范围说明 (为何不走 trigger_d_plus.py 全量)
    trigger_d_plus 的 find_pending_jq_entries 会一次捞起全部
    `jq_validated and not d_plus_applied` 条目 —— 当前实测 **234 条 / 101 组**
    (2026-08-05 起积压, 且 d_plus_applied 键在 2297 条中全部不存在)。
    全量蒸馏会在无用户授权下批量改写 Memory / hard_forbidden / MAB,
    远超本次「单个因子首次 JQ 验证」的范围 ⇒ 本脚本只定向处理目标条目。
    积压与 mark 未持久化缺陷单独上报, 待用户决策。

🔴 同时**不**调用 mark_d_plus_applied(): 它会把那 234 条一并标为已处理,
   等于静默谎报「已蒸馏」。只手工标记目标条目。
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(REPO, "data")
TRAJ = os.path.join(DATA, "trajectory_log.json")
QUEUE = os.path.join(DATA, "jq_codegen_queue.json")

TARGET_TRAJ_ID = "traj_2561b7d3c712"
TARGET_EXPR_FRAG = "rolling(10).sum()"
QUEUE_KEY_PREFIX = "forge_gen4_gp_gen4_r1_ts_rank"

# ── 用户回测读数 (jq_return 用百分比格式, 非小数) ──
JQ_RETURN = 64.57
JQ_SHARPE = 0.266
JQ_MAXDD = 29.49
JQ_IC = 0.0280          # Rank IC mean (255 samples, 2026-01-26 报告)
JQ_ICIR = 0.3695        # ICIR (mean/std)
JQ_PLUS_IC_PCT = 62.7   # IC > 0 ratio
BACKTEST_PERIOD = "2021-01-01~2026-07-30"
JQ_NOTES = (
    f"JQ 单因子回测 {BACKTEST_PERIOD} (日志至 2026-07-27 week=285, 5.6年): "
    f"累计 {JQ_RETURN}% / Sharpe {JQ_SHARPE} / maxDD {JQ_MAXDD}%; "
    f"滞后 Rank IC {JQ_IC}/ICIR {JQ_ICIR}/IC>0 {JQ_PLUS_IC_PCT}% (255期); "
    f"local ICIR 0.4649 -> JQ ICIR 0.3695 有衰减; "
    f"基准 000905 / topn80 / run_weekly / 周频语义(rolling 保持周口径); "
    f"暴露波动率 2021-02~06 曾报 28.37x 拥挤预警后恒为 0.00x(读数存疑, 见日报)"
)


def _load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(p, obj, crlf=True):
    """按目标文件既有行尾写回 (crlf=False -> 保持 LF)。"""
    txt = json.dumps(obj, ensure_ascii=False, indent=2)
    if crlf:
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(txt.replace("\r\n", "\n").replace("\n", "\r\n"))
    else:
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)


def main():
    print("=" * 62)
    print("  定向 D+ 闭环: forge_gen4_gp_gen4_r1_ts_rank")
    print("=" * 62)

    # ── 1. 定位 + 写入 JQ 字段 ──
    data = _load(TRAJ)
    trajs = data.get("trajectories", [])
    target = None
    for t in trajs:
        if t.get("trajectory_id") == TARGET_TRAJ_ID:
            target = t
            break
    if target is None:
        print(f"  [ERR] 未找到 trajectory_id={TARGET_TRAJ_ID}")
        return 1

    ex = target.get("expression") or {}
    exs = ex.get("content", "") if isinstance(ex, dict) else str(ex)
    assert TARGET_EXPR_FRAG in exs, f"表达式不匹配, 实际: {exs[:120]}"
    print(f"  目标条目: {target['trajectory_id']} | paradigm={target.get('paradigm')} "
          f"| natural_freq={target.get('natural_freq')}")
    print(f"  local: IC={target.get('final_ic')} ICIR={target.get('final_icir')} "
          f"outcome={target.get('final_outcome')}")

    outcome = "JQ_MARGINAL"
    assert not (JQ_RETURN > 50 and JQ_SHARPE > 0.4), "档位判定前提变化, 请复核 _jq_outcome"
    print(f"  JQ 读数: ret={JQ_RETURN}% sharpe={JQ_SHARPE} maxdd={JQ_MAXDD}% -> {outcome}")

    target["jq_validated"] = True
    target["jq_return"] = JQ_RETURN
    target["jq_sharpe"] = JQ_SHARPE
    target["jq_maxdd"] = JQ_MAXDD
    target["jq_ic"] = JQ_IC
    target["jq_icir"] = JQ_ICIR
    target["jq_rating"] = "MARGINAL"
    target["jq_notes"] = JQ_NOTES
    target["final_outcome"] = outcome
    target["completed_at"] = datetime.now().isoformat()
    # 保留 S1 阶段已有 natural_freq (weekly); 仅当缺失才设 weekly
    target.setdefault("natural_freq", "weekly")
    _save(TRAJ, data, crlf=True)
    print("  [1/5] 轨迹 JQ 字段已写入 (CRLF)")

    # ── 2. 队列状态回写 (LF 保持) ──
    q = _load(QUEUE)
    qk = None
    for k in q:
        if k.startswith(QUEUE_KEY_PREFIX):
            qk = k
            break
    if qk:
        e = q[qk]
        e["status"] = "jq_marginal"
        e["jq_return"] = JQ_RETURN
        e["jq_sharpe"] = JQ_SHARPE
        e["jq_maxdd"] = JQ_MAXDD
        e["jq_ic"] = JQ_IC
        e["jq_icir"] = JQ_ICIR
        e["jq_plus_ic_pct"] = JQ_PLUS_IC_PCT
        e["jq_verified_at"] = datetime.now().isoformat()
        e["jq_notes"] = JQ_NOTES
        e["backtest_period"] = BACKTEST_PERIOD
        _save(QUEUE, q, crlf=False)
        print(f"  [2/5] 队列条目 {qk[:46]}... -> jq_marginal (LF 保持)")
    else:
        print("  [2/5] ⚠️ 队列未匹配到条目, 跳过")

    # ── 3. jq_feedback 注入 (Memory + Trajectory) ──
    from trigger_d_plus import construct_jq_feedback  # 复用, 避免逻辑分叉
    data2 = _load(TRAJ)
    tgt2 = next(t for t in data2["trajectories"] if t["trajectory_id"] == TARGET_TRAJ_ID)
    groups = {exs[:80].strip(): [tgt2]}
    jq_feedback = construct_jq_feedback(groups)
    if not jq_feedback:
        print("  [3/5] [ERR] construct_jq_feedback 返回空")
        return 1
    print(f"  [3/5] jq_feedback: {len(jq_feedback['factors'])} 因子 | "
          f"composite_ret={jq_feedback['composite_return']:.2f}% | "
          f"sharpe={jq_feedback['composite_sharpe']:.3f}")

    from ralph_loop import RalphLoop
    from experience_memory import get_memory
    memory = get_memory()
    rl = RalphLoop(memory=memory)
    res = rl.jq_feedback(jq_backtest_result=jq_feedback, update_lane=False)
    print(f"  [3/5] jq_feedback 返回: 轨迹更新={res.get('trajectories_updated', 0)} "
          f"/ 硬禁止新增={res.get('hard_forbidden_added', 0)} "
          f"/ JQ确认成功={res.get('jq_success_confirmed', 0)}")

    # ── 4. 只标记目标条目 (不碰 234 条积压) ──
    data3 = _load(TRAJ)
    n_mark = 0
    for t in data3.get("trajectories", []):
        if t.get("trajectory_id") == TARGET_TRAJ_ID:
            t["d_plus_applied"] = True
            t["d_plus_at"] = datetime.now().isoformat()[:19]
            n_mark += 1
    if n_mark:
        _save(TRAJ, data3, crlf=True)
    print(f"  [4/5] 定向标记 d_plus_applied: {n_mark} 条 (积压 234 条未触碰)")

    # ── 5. lane 校准滚动 + pool 回写 ──
    try:
        from lane_calibration import update_from_jq_feedback, recalibrate_tau_w
        up = update_from_jq_feedback(jq_feedback.get("factors", []))
        print(f"  [5/5] 校准集: 新增 {up['n_new']} / 更新 {up['n_updated']} / 跳过 {up['n_skipped']}")
        cr = recalibrate_tau_w()
        print(f"        τ_w: current={cr['current_tau_w']} recommended={cr['recommended']} "
              f"effective={cr['effective']} action={cr['action']}")
        print(f"        reason: {cr['reason']}")
    except Exception as e:
        print(f"  [5/5] ⚠️ lane 校准失败 (不阻塞): {e}")

    try:
        from trigger_d_plus import _write_back_pool_status
        n_pool = _write_back_pool_status(jq_feedback.get("factors", []))
        print(f"        pool 回写: {n_pool} 条")
    except Exception as e:
        print(f"        pool 回写失败: {e}")

    print("=" * 62)
    print("  定向 D+ 闭环完成")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
