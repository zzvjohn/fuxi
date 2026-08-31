# -*- coding: utf-8 -*-
"""
Lane Calibration — v0.7 P4: τ_w 校准集滚动更新 (2026-08-29)
================================================================================
职责 (方案 v07_forge_s1_calibration_plan_20260829.md §6.4 P4):

  1. D+ 闭环时 (trigger_d_plus), 每个 JQ 反馈因子自动复评 pandas 周频 ICIR,
     幂等 upsert 进 data/lane_calibration.json (校准集)。
  2. 基于 JQ 锚点重估 τ_w (周频合格线), 防误杀优先 + hysteresis 双批投票。
  3. get_effective_tau_w(): 裁决器读 τ_w 的唯一入口 —
     lane_calibration.tau_w_effective (已确认) > config.V07_DUAL_LANE.weekly_icir_threshold。

校准规则 (2026-08-29 P3 实证锚定):
  - JQ 锚点 (pandas 周频口径 ICIR → JQ 结果):
      PASS      ts_std = 0.183
      MARGINAL  square_square_tsmin = 0.230 / vol_zscore_inv = 0.013
      FAILED    volume_dual_rank = 0.509  ← FAILED 比 PASS 还高!
  - 实证: pandas 周频 ICIR 与 JQ 结果相关性弱 (0.509→FAILED vs 0.013→MARGINAL),
    τ_w 无分离力 → 唯一可靠约束是"不高于 PASS 锚点下界" (防误杀)。
  - 铁则: S1 只是粗筛, 主力过滤在 S2-S6/JQ; 误杀成本 > 放水成本。
    → τ_w 只允许自动下调或持平, 自动上调禁止 (上调 = 误杀已知 PASS 因子)。
  - 公式: recommended = max(FLOOR, min(pandas_icir of JQ_PASSED) × SAFETY)
            FLOOR = 0.10 (低于此阈值无筛选意义), SAFETY = 0.82 (留 18% 余量)
    当前锚点: min(PASS) = 0.183 → 0.183×0.82 = 0.150 ≈ config 基线 0.15 ✓
  - hysteresis: 同一更低推荐值连续出现 2 个独立 D+ 批次才写 tau_w_effective
    (防单点噪声抖动 τ_w)。推荐 ≥ 当前值时清空投票 (防陈旧票误生效)。
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 项目根 = 本文件上三级 (research/factor_alchemy -> research -> quant)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CALIBRATION_PATH = _PROJECT_ROOT / "data" / "lane_calibration.json"

# τ_w 校准参数
_TAU_FLOOR = 0.10        # 下限: 低于此阈值筛选无意义
_TAU_SAFETY = 0.82       # 余量系数: PASS 锚点下界 × 0.82
_TAU_MIN_STEP = 0.02     # 防抖: |Δτ_w| < 0.02 视为噪声不调线 (ICIR 0.15 vs 0.1498 无筛选差异)
_OUTCOME_PRIORITY = {"JQ_PASSED": 3, "JQ_MARGINAL": 2, "JQ_FAILED": 1}


# ═══════════════════════════════════════════════════════
# 1. 读写
# ═══════════════════════════════════════════════════════

def load_calibration(path=None) -> Dict:
    p = Path(path) if path else _DEFAULT_CALIBRATION_PATH
    if not p.exists():
        return {
            "version": 2, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "description": "周频 lane 校准集 (P4: JQ 反馈滚动更新)",
            "n_points": 0, "points": [], "tau_w_votes": [],
        }
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [LaneCalib] ⚠️ 校准集加载失败, 用空集: {e}")
        return {
            "version": 2, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "description": "周频 lane 校准集 (加载失败重建)",
            "n_points": 0, "points": [], "tau_w_votes": [],
        }


def save_calibration(cal: Dict, path=None) -> None:
    p = Path(path) if path else _DEFAULT_CALIBRATION_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cal, f, ensure_ascii=False, indent=1)
    tmp.replace(p)


def _norm_formula(s) -> str:
    return re.sub(r"\s+", "", str(s or ""))


# ═══════════════════════════════════════════════════════
# 2. JQ 评级 → 校准集 outcome 映射
# ═══════════════════════════════════════════════════════

def _map_outcome(factor: Dict) -> Optional[str]:
    rating = str(factor.get("jq_rating") or "").strip().upper()
    if rating in ("BEST", "PASS", "PASSED", "JQ_PASSED"):
        return "JQ_PASSED"
    if rating in ("MARGINAL", "BORDERLINE", "JQ_MARGINAL"):
        return "JQ_MARGINAL"
    if rating in ("FAIL", "FAILED", "BROKEN", "DUPLICATE", "JQ_FAILED"):
        return "JQ_FAILED"
    contrib = str(factor.get("jq_composite_contribution") or "").lower()
    if contrib == "positive":
        return "JQ_PASSED"
    if contrib == "negative":
        return "JQ_FAILED"
    if contrib == "neutral":
        return "JQ_MARGINAL"
    return None


def _is_weekly_candidate(factor: Dict) -> bool:
    """校准集只收周频 lane 相关因子 (natural_freq=weekly 或 forge 来源)。"""
    freq = str(factor.get("natural_freq") or "").strip().lower()
    if freq == "weekly":
        return True
    name = str(factor.get("factor_name") or "").lower()
    return "forge" in name


# ═══════════════════════════════════════════════════════
# 3. 复评 judge (与 P3a 双口径复评口径完全一致)
# ═══════════════════════════════════════════════════════

def _make_zscore(w: int):
    def f(x):
        if isinstance(x, (__import__("pandas").DataFrame, __import__("pandas").Series)):
            return (x - x.rolling(w).mean()) / (x.rolling(w).std() + 1e-10)
        return x
    return f


def _make_roll_min(w: int):
    def f(x):
        if isinstance(x, (__import__("pandas").DataFrame, __import__("pandas").Series)):
            return x.rolling(w).min()
        return x
    return f


_review_judge_cache = {"judge": None}


def _build_review_judge():
    """构造复评 judge (含旧 forge_r1 公式 helper 注入, 与 P3a 口径一致)。"""
    if _review_judge_cache["judge"] is not None:
        return _review_judge_cache["judge"]
    import sys
    from pathlib import Path as _P
    _mod_dir = _P(__file__).resolve().parent
    if str(_mod_dir) not in sys.path:
        sys.path.insert(0, str(_mod_dir))
    from weekly_lane import WeeklyLaneJudge

    judge = WeeklyLaneJudge(
        parquet_path=_PROJECT_ROOT / "output" / "ap_batch" / "cache" / "weekly_prices.parquet"
    )
    judge._load()
    for _w in (12, 20, 26, 30, 40, 60):
        judge._wide_env[f"zscore{_w}"] = _make_zscore(_w)
        judge._wide_env[f"rolling_min_{_w}"] = _make_roll_min(_w)
    _review_judge_cache["judge"] = judge
    return judge


def review_pandas_icir(formula: str, factor_name: str = "") -> Tuple[Optional[float], Optional[str]]:
    """pandas 周频口径复评 → (icir, note)。eval 失败返回 (None, reason)。"""
    try:
        judge = _build_review_judge()
    except Exception as e:
        return None, f"judge 构造失败: {type(e).__name__}: {str(e)[:60]}"
    r = judge.judge(formula, factor_name)
    if r.get("eval_ok"):
        return float(r["weekly_icir"]), None
    return None, r.get("reason", "eval fail")


# ═══════════════════════════════════════════════════════
# 4. D+ 反馈 → 校准集 upsert (幂等)
# ═══════════════════════════════════════════════════════

def update_from_jq_feedback(factors: List[Dict], path=None) -> Dict:
    """
    把 JQ 反馈因子复评后 upsert 进校准集。

    factors: trigger_d_plus.construct_jq_feedback 产出的 factors 列表,
             每元素含 factor_name/formula/natural_freq/jq_rating/jq_return/...
    返回: {n_new, n_updated, n_skipped, n_eval_ok}
    """
    cal = load_calibration(path)
    points = cal.setdefault("points", [])
    by_key = {_norm_formula(p.get("formula")): p for p in points if p.get("formula")}

    n_new = n_updated = n_skipped = n_eval_ok = 0
    for f in factors:
        formula = str(f.get("formula") or "").strip()
        if not formula or formula.startswith("# TODO"):
            n_skipped += 1
            continue
        if not _is_weekly_candidate(f):
            n_skipped += 1  # daily 因子不进周频校准集
            continue
        outcome = _map_outcome(f)
        if outcome is None:
            n_skipped += 1
            continue

        key = _norm_formula(formula)
        p_icir, p_note = review_pandas_icir(formula, f.get("factor_name", ""))
        if p_icir is not None:
            n_eval_ok += 1

        new_jq = {
            "jq_outcome": outcome,
            "jq_return": f.get("jq_return"),
            "jq_sharpe": f.get("jq_sharpe"),
            "jq_maxdd": f.get("jq_maxdd"),
            # P-001 (2026-08-29): 补存 JQ 单因子 IC/ICIR, 是 τ_w 校准最宝贵的 JQ 口径锚点
            "jq_ic": f.get("jq_ic"),
            "jq_icir": f.get("jq_icir"),
        }
        existing = by_key.get(key)
        if existing is None:
            point = {
                "name": str(f.get("factor_name") or "")[:120],
                "formula": formula[:1000],
                "pandas_icir": round(p_icir, 4) if p_icir is not None else None,
                "forge_icir": None,
                "natural_freq": str(f.get("natural_freq") or "weekly"),
                "note": f"D+ 滚动更新; {p_note}" if p_note else "D+ 滚动更新",
            }
            point.update(new_jq)
            points.append(point)
            by_key[key] = point
            n_new += 1
        else:
            changed = False
            # pandas_icir 只升不降 (同公式确定性复评, 数据滚动下防噪声把强因子标弱)
            if p_icir is not None and (existing.get("pandas_icir") is None
                                       or p_icir > existing["pandas_icir"]):
                existing["pandas_icir"] = round(p_icir, 4)
                changed = True
            # JQ 证据只升不降: (outcome 优先级, jq_return) 更强才整体替换
            old_pri = _OUTCOME_PRIORITY.get(existing.get("jq_outcome"), 0)
            new_pri = _OUTCOME_PRIORITY.get(outcome, 0)
            old_ret = existing.get("jq_return") or -1e18
            new_ret = f.get("jq_return") or -1e18
            if (new_pri, new_ret) > (old_pri, old_ret):
                existing.update(new_jq)
                changed = True
            if p_note and p_note not in str(existing.get("note", "")):
                existing["note"] = str(existing.get("note", "")) + f"; {p_note}"
                changed = True
            if changed:
                n_updated += 1

    cal["n_points"] = len(points)
    cal["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_calibration(cal, path)
    return {"n_new": n_new, "n_updated": n_updated, "n_skipped": n_skipped, "n_eval_ok": n_eval_ok}


# ═══════════════════════════════════════════════════════
# 5. τ_w 重估 (防误杀优先 + hysteresis)
# ═══════════════════════════════════════════════════════

def _config_tau_w() -> float:
    try:
        from config import V07_DUAL_LANE
        return float(V07_DUAL_LANE.get("weekly_icir_threshold", 0.15))
    except Exception:
        return 0.15


def recalibrate_tau_w(path=None, dry: bool = False) -> Dict:
    """
    基于校准集 JQ 锚点重估 τ_w。

    规则 (铁则: 防误杀优先, 自动上调禁止):
      recommended = clamp(min(pandas_icir of JQ_PASSED) × SAFETY, FLOOR, current)
      - 无 PASS 锚点 → 不动作
      - rec < current → 记投票; 连续 2 个独立批次同值 → 写 tau_w_effective
      - rec ≥ current → 清空投票 (防陈旧票误生效), effective 维持
    返回报告 dict。
    """
    cal = load_calibration(path)
    points = cal.get("points", [])
    anchors = [
        p for p in points
        if p.get("jq_outcome") in ("JQ_PASSED", "JQ_MARGINAL", "JQ_FAILED")
        and p.get("pandas_icir") is not None
    ]
    passed = sorted(p["pandas_icir"] for p in anchors if p["jq_outcome"] == "JQ_PASSED")
    failed = sorted(p["pandas_icir"] for p in anchors if p["jq_outcome"] == "JQ_FAILED")
    current = _config_tau_w()

    rec = current
    reason = "无 JQ_PASSED 锚点, 维持现值"
    if passed:
        raw = max(_TAU_FLOOR, min(passed) * _TAU_SAFETY)
        rec = round(min(raw, current), 4)  # 只降不升
        if rec < current - _TAU_MIN_STEP:
            reason = (f"min(PASS)={min(passed):.4f}×{_TAU_SAFETY}="
                      f"{raw:.4f} → 建议下调 τ_w {current} → {rec}")
        else:
            reason = (f"min(PASS)={min(passed):.4f}×{_TAU_SAFETY}="
                      f"{raw:.4f} → |Δτ_w|={current - rec:.4f} < {_TAU_MIN_STEP} 防抖, 维持")
            rec = current  # 防抖: 钳回现值, 不投票

    votes = cal.get("tau_w_votes", [])
    action = "maintain"
    if rec < current - _TAU_MIN_STEP:
        # 需要下调 → 投票
        last_val = votes[-1].get("value") if votes else None
        if last_val is not None and abs(float(last_val) - rec) < 1e-4:
            # 连续第 2 批同值 → 生效
            if not dry:
                cal["tau_w_effective"] = rec
                cal["tau_w_votes"] = []
                cal["tau_w_effective_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                cal["updated_at"] = cal["tau_w_effective_at"]
                save_calibration(cal, path)
            action = "applied"
            reason += f"; hysteresis 连续 2 批同值 → τ_w_effective={rec} 已生效"
        else:
            if not dry:
                votes.append({"value": rec, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "reason": reason})
                cal["tau_w_votes"] = votes[-2:]
                cal["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                save_calibration(cal, path)
            action = "voted"
            reason += "; 首票已记, 下批同值才生效 (hysteresis)"
    else:
        if votes and not dry:
            cal["tau_w_votes"] = []
            cal["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            save_calibration(cal, path)
        if votes:
            reason += "; 陈旧下调票已清除"

    # 高 ICIR FAILED 观察 (仅提示, 不参与自动调线)
    hi_failed = [v for v in failed if v >= 2 * current]
    if hi_failed:
        reason += (f"; ⚠️ {len(hi_failed)} 个 FAILED 锚点 pandas_icir≥2×τ_w "
                   f"(max={max(hi_failed):.4f}) — 阈值无分离力实证, 不自动上调")

    return {
        "action": action,
        "current_tau_w": current,
        "recommended": rec,
        "effective": cal.get("tau_w_effective"),
        "n_anchors": len(anchors),
        "n_pass": len(passed),
        "n_failed": len(failed),
        "min_pass_icir": min(passed) if passed else None,
        "reason": reason,
    }


def get_effective_tau_w(path=None) -> float:
    """裁决器读 τ_w 唯一入口: 校准集 effective (已确认) > config 基线。"""
    try:
        cal = load_calibration(path)
        eff = cal.get("tau_w_effective")
        if eff is not None:
            return float(eff)
    except Exception:
        pass
    return _config_tau_w()
