"""P-20260907-006: S5 方向符号守卫 (2026-09-11 转正 enforce)

背景 (同坑三次复现):
  - gp_evo_001 JQ 证伪 (2026-09-07): 本地 attempt icir=-0.205 与 JQ 同号同量级,
    因子仍以【原方向】进入 JQ 队列 → 组合 -25.8% / Sharpe -0.454 / MDD 50.16%。
  - ralph_loop L1942 R3 已知坑: 最佳 ICIR=-0.890 原方向进 S5 → 全灭。
  - 2026-09-11 Stage2: lhb_inst_netsell_intensity_5d 假设 short 被实证证伪
    (实证 IC=+0.0332 为正), 再次暴露方向盲区。

根因:
  S1 / S5 fallback 用 abs(IC) / abs(ICIR) 绝对值做代理判定, 无方向意识;
  s5_joint_filter.validate_factor 无方向参数 → 负 ICIR 公式原样回测必灭
  (框架按因子值排序做多, 负向因子等于做空信号被当多头用)。

守卫逻辑 (前置在 S1 之前, 一因子一决策):
  1. 读候选在【本通道】的本地 IC/ICIR:
       daily  lane → cand["ic"] / cand["icir"]
       weekly lane → cand["weekly_ic"] / cand["weekly_icir"]
  2. ICIR >= 0  → 方向正确, 放行 (action="sign_ok")
  3. ICIR <  0  → 生成方向翻转副本 "-(<formula>)", 用【同通道】重算 IC:
       - 翻转版 ICIR > 0 且 |翻转 ICIR| >= |原 ICIR|
           → enforce: 用翻转版替换 formula/ic/icir, 标 flipped_direction=True
           → shadow : 只标注, 不改 (零行为变化)
       - 否则 → 标 sign_guard_reject=True (负向且翻转无效; 下游符号感知 S1 会拒)
  4. 无 IC 数据 (ic==0 且 icir==0) → 跳过 (交下游 "无IC数据" 分支拒绝)

开关: env FUXI_S5_SIGN_GUARD = enforce(默认) | shadow | off
回退: export FUXI_S5_SIGN_GUARD=off (或 shadow, 一行回退, 零行为变化)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

MODES = ("off", "shadow", "enforce")


def get_mode() -> str:
    """读取守卫模式 (env FUXI_S5_SIGN_GUARD), 默认 enforce (2026-09-11 转正)。"""
    m = str(os.environ.get("FUXI_S5_SIGN_GUARD", "enforce") or "enforce").strip().lower()
    return m if m in MODES else "enforce"


def _flip_formula(formula: str) -> str:
    """方向翻转: 公式整体取负。括号包裹保证与运算符优先级无关。"""
    f = (formula or "").strip()
    if not f:
        return ""
    return f"-({f})"


def _is_weekly(cand: Dict) -> bool:
    return str(cand.get("natural_freq", "") or "") == "weekly"


def _read_signal(cand: Dict) -> tuple:
    """返回 (channel, ic, icir)。weekly lane 读周频字段, 其余读日频。"""
    if _is_weekly(cand):
        ic = float(cand.get("weekly_ic", 0) or 0)
        icir = float(cand.get("weekly_icir", 0) or 0)
        return "weekly", ic, icir
    ic = float(cand.get("ic", cand.get("daily_ic", 0)) or 0)
    icir = float(cand.get("icir", cand.get("daily_icir", 0)) or 0)
    return "daily", ic, icir


def _recompute(formula: str, name: str, channel: str,
               ic_comp: Any, weekly_judge: Any) -> Optional[Dict[str, float]]:
    """用与候选相同的通道重算翻转公式的 IC。返回 {ic, icir} 或 None。"""
    if channel == "weekly":
        if weekly_judge is None:
            return None
        try:
            r = weekly_judge.judge(formula, name)
        except Exception:
            return None
        if not r.get("eval_ok"):
            return None
        return {"ic": float(r.get("weekly_ic", 0) or 0),
                "icir": float(r.get("weekly_icir", 0) or 0)}
    if ic_comp is None:
        return None
    try:
        r = ic_comp.compute(formula, name)
    except Exception:
        return None
    if not r or int(r.get("n_days", 0) or 0) <= 0:
        return None
    return {"ic": float(r.get("ic", 0) or 0),
            "icir": float(r.get("icir", 0) or 0)}


def apply(candidates: List[Dict], ic_comp: Any = None,
          weekly_judge: Any = None, verbose: bool = True) -> Dict[str, int]:
    """对候选列表施加方向符号守卫 (就地修改 enforce 模式下的候选)。

    Parameters
    ----------
    candidates : list[dict]
        候选列表 (含 formula / ic / icir 或 weekly_ic / weekly_icir)。
    ic_comp : FactorICComputer | None
        daily lane 重算用。
    weekly_judge : WeeklyLaneJudge | None
        weekly lane 重算用。
    verbose : bool
        是否打印读数行。

    Returns
    -------
    dict 统计: {"mode","n_checked","n_negative","n_flipped","n_reject","n_skipped"}
    """
    mode = get_mode()
    stats = {"mode": mode, "n_checked": 0, "n_negative": 0,
             "n_flipped": 0, "n_reject": 0, "n_skipped": 0}
    if mode == "off" or not candidates:
        return stats

    for cand in candidates:
        formula = cand.get("formula", cand.get("expression", ""))
        channel, ic, icir = _read_signal(cand)
        # 无 IC 数据 → 交下游拒绝
        if abs(ic) < 1e-8 and abs(icir) < 1e-8:
            stats["n_skipped"] += 1
            continue
        stats["n_checked"] += 1

        if icir >= 0:
            cand["sign_guard_action"] = "sign_ok"
            continue

        # ── 负向候选 ──
        stats["n_negative"] += 1
        name = cand.get("factor_name", cand.get("name", "unknown"))
        flipped = _flip_formula(formula)
        fl = _recompute(flipped, str(name) + "_signflip", channel,
                        ic_comp, weekly_judge) if flipped else None

        if fl and fl["icir"] > 0 and abs(fl["icir"]) >= abs(icir):
            if mode == "enforce":
                cand["formula"] = flipped
                cand["_orig_formula_before_flip"] = formula
                if channel == "weekly":
                    cand["weekly_ic"] = fl["ic"]
                    cand["weekly_icir"] = fl["icir"]
                else:
                    cand["ic"] = fl["ic"]
                    cand["icir"] = fl["icir"]
                    cand["daily_ic"] = fl["ic"]
                    cand["daily_icir"] = fl["icir"]
            cand["flipped_direction"] = True
            cand["sign_guard_action"] = ("flipped" if mode == "enforce"
                                         else "would_flip")
            cand["sign_guard_icir"] = round(icir, 4)
            cand["sign_guard_flip_icir"] = round(fl["icir"], 4)
            if mode == "enforce":
                stats["n_flipped"] += 1
        else:
            cand["sign_guard_action"] = ("reject" if mode == "enforce"
                                         else "would_reject")
            cand["sign_guard_icir"] = round(icir, 4)
            cand["flipped_direction"] = False
            if mode == "enforce":
                stats["n_reject"] += 1
                cand["_sign_guard_reject"] = True

    if verbose and (stats["n_negative"] or mode != "enforce"):
        print(f"  [S5-SignGuard] {mode} 模式: 检查 {stats['n_checked']} 候选 | "
              f"负向 {stats['n_negative']} → 翻转 {stats['n_flipped']} / "
              f"拒 {stats['n_reject']} | 无IC跳过 {stats['n_skipped']}")
    return stats
