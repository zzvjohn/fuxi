# -*- coding: utf-8 -*-
"""
FreqAudit — P-20260903-001: natural_freq 标记审计与自动重标定
================================================================================
背景 (2026-09-03 晨检确证):
  vwap_dev_ratio_5_60 系两条轨迹 natural_freq=daily 错标, 而其 JQ 回测为周频口径
  (285 周)、P-001 复评亦为周频 (weekly_icir 0.2721)。v0.7 分频裁决 (weekly XOR
  daily) 完全依赖该字段, 错标即错杀: 周频 alpha 会被路由进 daily lane 门槛
  (fwd5 ic>=0.02 & icir>=0.1), 可能被日频噪声误拒。

设计红线 (防新误杀):
  1. 铁证才自动修正 — 仅当轨迹 JQ 已验证 且 lane 校准集存在同名/同式 weekly 锚点
     (校准集只收周频候选, P-001 复评为周频口径, 是当前最硬的周频铁证)。
  2. 疑似项只标注报告, 绝不修改 (公式含慢窗口启发式误判空间大, 见 2026-09-03 审核)。
  3. 幂等: freq 已是 weekly 的条目跳过; 修正写 _freq_audit 字段留痕。
     🔴 该键必须同时在 MiningTrajectory 三处 (dataclass/to_dict/from_dict)
        出现, 否则下一轮 _save() 会**静默抹除** (P-20260919-001, 已修)。
  4. 审计失败绝不阻塞主流程 (调用方 try/except)。

输出: [FreqAudit] 行, 供日报/日志消费。
"""
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

# 项目根 = 本文件上两级 (research/factor_alchemy -> research -> quant)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRAJ_PATH = _PROJECT_ROOT / "data" / "trajectory_log.json"
_CAL_PATH = _PROJECT_ROOT / "data" / "lane_calibration.json"

_SLOW_ROLL_RE = re.compile(r"rolling\s*\(\s*(\d+)\s*\)")
_FAST_SIGNAL_RE = re.compile(
    r"rolling\s*\(\s*(?:[1-4])\s*\)|pct_change\s*\(\s*1\s*\)|shift\s*\(\s*1\s*\)|diff\s*\(\s*1\s*\)"
)


def _norm(s) -> str:
    return re.sub(r"\s+", "", str(s or ""))


def _traj_formula(e: Dict) -> str:
    """轨迹公式藏在 expression dict 里 (trajectory_logger 结构)。"""
    expr = e.get("expression")
    if isinstance(expr, dict):
        return str(expr.get("content") or "")
    return str(expr or "")


def _traj_freq(e: Dict) -> str:
    return str(e.get("natural_freq") or "").strip().lower()


def _is_jq_validated(e: Dict) -> bool:
    return bool(e.get("jq_validated")) or e.get("jq_return") is not None


def _build_anchor_index(cal: Dict) -> tuple:
    """校准集锚点索引: {norm_formula: anchor}, {name: [anchor]} (仅 weekly 锚点)。"""
    by_formula = {}
    by_name: Dict[str, list] = {}
    for p in cal.get("points", []):
        if str(p.get("natural_freq") or "").strip().lower() != "weekly":
            continue
        f = _norm(p.get("formula"))
        if f:
            by_formula[f] = p
        nm = str(p.get("name") or "").strip()
        if nm:
            by_name.setdefault(nm, []).append(p)
    return by_formula, by_name


def _slow_window_report(formula: str) -> Optional[Dict]:
    """疑似 weekly 启发式 (只报告不改): 含 rolling(w>=20) 慢算子且无 <5 日快信号。

    注意: 该启发式仅用于统计级观察 (大量日频因子也含 20+ 日窗口, 如
    dollar_vol_20d), 绝不作为修改依据 — 铁证修正只看校准集锚点。
    """
    if not formula:
        return None
    rolls = [int(m) for m in _SLOW_ROLL_RE.findall(formula)]
    if not rolls:
        return None
    max_w = max(rolls)
    min_w = min(rolls)
    if max_w < 20 or min_w < 5:
        return None
    if _FAST_SIGNAL_RE.search(formula):
        return None
    return {"max_roll": max_w, "min_roll": min_w}


def audit_trajectory_freqs(traj_path=None, cal_path=None, fix: bool = True,
                           max_suspect_lines: int = 10) -> Dict:
    """
    审计 trajectory_log 的 natural_freq 并(铁证)自动修正。

    返回 report dict: {n_total, n_weekly, n_empty, n_fixed, fixed_names,
                       n_suspect, suspect_lines, cal_anchors}
    """
    traj_path = Path(traj_path) if traj_path else _TRAJ_PATH
    cal_path = Path(cal_path) if cal_path else _CAL_PATH

    cal = {}
    if cal_path.exists():
        try:
            cal = json.load(open(cal_path, encoding="utf-8"))
        except Exception:
            cal = {}
    by_formula, by_name = _build_anchor_index(cal)

    if not traj_path.exists():
        return {"error": f"trajectory_log 不存在: {traj_path}"}

    traj = json.load(open(traj_path, encoding="utf-8"))
    entries = traj.get("trajectories", [])
    n_total = len(entries)
    n_weekly = sum(1 for e in entries if _traj_freq(e) == "weekly")
    n_empty = sum(1 for e in entries if _traj_freq(e) == "")

    n_fixed = 0
    n_suspect = 0
    fixed_names: List[str] = []
    suspect_lines: List[str] = []
    n_traj_weekly_updated = 0

    for e in entries:
        freq = _traj_freq(e)
        name = str(e.get("seed_factor") or "")
        formula = _traj_formula(e)

        # ── 铁证修正: JQ 已验证 + 校准集 weekly 锚点 ──
        if freq in ("daily", "") and _is_jq_validated(e):
            anchor = by_formula.get(_norm(formula)) if formula else None
            if anchor is None and name:
                # 公式未匹配 → name 窄口径铁证: 该 name 下 JQ 验证轨迹恰好 1 条
                cands = by_name.get(name, [])
                jq_same_name = sum(
                    1 for x in entries
                    if x.get("seed_factor") == name and _is_jq_validated(x)
                )
                if len(cands) == 1 and jq_same_name == 1:
                    anchor = cands[0]
            if anchor is not None:
                if fix:
                    e["natural_freq"] = "weekly"
                    e["_freq_audit"] = (
                        f"{freq or 'empty'}->weekly "
                        f"(lane 锚点 {anchor.get('name','?')} weekly 铁证, "
                        f"P-20260903-001)"
                    )
                    n_traj_weekly_updated += 1
                n_fixed += 1
                fixed_names.append(name or "(unnamed)")
                continue

        # ── 疑似标注 (只读, 仅统计有名字的条目避免 unnamed 老轨迹刷屏) ──
        if freq == "daily" and name and not _is_jq_validated(e):
            rep = _slow_window_report(formula)
            if rep is not None:
                n_suspect += 1
                if len(suspect_lines) < max_suspect_lines:
                    suspect_lines.append(
                        f"{name} (roll[{rep['min_roll']},{rep['max_roll']}])"
                    )

    if fix and n_traj_weekly_updated:
        traj["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        json.dump(traj, open(traj_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    return {
        "n_total": n_total,
        "n_weekly": n_weekly + n_fixed,
        "n_empty": n_empty,
        "n_fixed": n_fixed,
        "fixed_names": fixed_names,
        "n_suspect": n_suspect,
        "suspect_lines": suspect_lines,
        "cal_weekly_anchors": len(by_formula),
        "traj_updated": n_traj_weekly_updated > 0,
    }


def run_freq_audit(verbose: bool = True, fix: bool = True) -> Dict:
    """主入口: 审计 + [FreqAudit] 行输出。"""
    rep = audit_trajectory_freqs(fix=fix)
    if verbose:
        if rep.get("error"):
            print(f"  [FreqAudit] ⚠️ {rep['error']}")
            return rep
        print(f"  [FreqAudit] 扫描 {rep['n_total']} 条 | 已 weekly {rep['n_weekly'] - rep['n_fixed']} "
              f"| 铁证修正 {rep['n_fixed']} | 疑似 weekly {rep['n_suspect']} | 空 freq {rep['n_empty']}")
        if rep["fixed_names"]:
            print(f"  [FreqAudit] 修正: {', '.join(rep['fixed_names'])}")
        if rep["suspect_lines"]:
            print(f"  [FreqAudit] 疑似(只标注不改): {', '.join(rep['suspect_lines'])}")
    return rep


if __name__ == "__main__":
    # 独立运行: 默认 dry (只报告不修改), --fix 才执行修正
    import sys
    do_fix = "--fix" in sys.argv
    rep = run_freq_audit(verbose=True, fix=do_fix)
    if not do_fix:
        print("  [FreqAudit] dry 模式, 未落盘; 加 --fix 执行铁证修正")
