"""M2 (2026-09-27, P-20260927-005): 生成器轴学习「写+读双开关」与写入端。

背景（方案 §M2，E1 证据）:
  · 旧写入块注释写「写入恒开」，实现却默认 raise RuntimeError 并被 except
    静默吞掉 ⇒ 注释与实现矛盾，且"关"状态不可见;
  · 生成器轴消费挂在 FUXI_EVO_REFINER 上（该开关本意 = gp_evolve 精炼器
    补路），两义耦合 ⇒ 无法单独观测/启用轴学习。

本模块提供:
  1. resolve_write_mode / resolve_read_mode: 双开关解析（纯函数，可测）;
  2. s2a_generator_axis_write: 写入端三态（off | shadow(默认) | on）;
     shadow 追加观测到 data/generator_axis_write_shadow.jsonl（不动 MAB）;
  3. append_shadow_entry: 观测条目落盘（JSONL, 逐行）。

开关语义（默认即契约）:
  FUXI_GENERATOR_AXIS_WRITE : off | shadow(默认) | on   —— 写入端
  FUXI_GENERATOR_AXIS       : off | shadow(默认) | on   —— 读端
                              （在 ralph_loop._mab_select_generator 内消费）
  FUXI_LEGACY_GENERATOR_PROXY_REWARD=on 为写入端旧别名（⇒ on），仅兼容，
  新代码不应依赖（M2 起以 FUXI_GENERATOR_AXIS_WRITE 为准）。

硬边界（与 M1 的 learning_feedback 通道互补、不重叠）:
  · off/shadow 绝不对 MAB 状态写入（resolve_arms 仅只读计数）;
  · 不触碰裁决字段（stage5_passed / eligible_for_jq 仅只读做 success 布尔）;
  · 任何异常显式返回 status=error 并入 lines，不抛出、不静默
    （即修 E1 的「静默吞异常」）。
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = "fuxi_generator_axis_write/v1"

WRITE_ENV = "FUXI_GENERATOR_AXIS_WRITE"
READ_ENV = "FUXI_GENERATOR_AXIS"
LEGACY_WRITE_ALIAS = "FUXI_LEGACY_GENERATOR_PROXY_REWARD"

WRITE_MODES = ("off", "shadow", "on")
READ_MODES = ("off", "shadow", "on")
DEFAULT_WRITE_MODE = "shadow"
DEFAULT_READ_MODE = "shadow"

# 口径与 evaluate 内 L2 本地信号一致 (ICIR→tanh 压缩, 上限 ±0.3),
# 与 JQ 终裁 (±1.0) 分属不同量级, 仅作引擎选择的引导。
ICIR_GAIN = 3.0
REWARD_CAP = 0.3

_LEDGER_NAME = "generator_axis_write_shadow.jsonl"
DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "data" / _LEDGER_NAME

__all__ = [
    "SCHEMA", "WRITE_ENV", "READ_ENV", "LEGACY_WRITE_ALIAS",
    "WRITE_MODES", "READ_MODES", "DEFAULT_WRITE_MODE", "DEFAULT_READ_MODE",
    "DEFAULT_LEDGER", "resolve_write_mode", "resolve_read_mode",
    "append_shadow_entry", "s2a_generator_axis_write",
]


def resolve_write_mode(env: Optional[Dict[str, str]] = None) -> str:
    """写入端开关解析。非法值回落默认 shadow（调用方负责可见性提示）。"""
    e = os.environ if env is None else env
    raw = str(e.get(WRITE_ENV, "") or "").strip().lower()
    if raw in WRITE_MODES:
        return raw
    if not raw:
        # 旧别名兼容: FUXI_LEGACY_GENERATOR_PROXY_REWARD=on ⇒ 写 MAB
        if str(e.get(LEGACY_WRITE_ALIAS, "off") or "").strip().lower() == "on":
            return "on"
        return DEFAULT_WRITE_MODE
    return DEFAULT_WRITE_MODE


def resolve_read_mode(env: Optional[Dict[str, str]] = None) -> str:
    """读端开关解析（M2 起与 FUXI_EVO_REFINER 解耦，不再读取后者）。"""
    e = os.environ if env is None else env
    raw = str(e.get(READ_ENV, "") or "").strip().lower()
    return raw if raw in READ_MODES else DEFAULT_READ_MODE


def _icir_abs_list(results: Any) -> List[float]:
    """从 evaluate results 抽取 |s1_icir| > 0 的列表（只读，兼容属性/字典）。"""
    out: List[float] = []
    for r in results or []:
        v = getattr(r, "s1_icir", None)
        if v is None and isinstance(r, dict):
            v = r.get("s1_icir")
        try:
            f = abs(float(v or 0.0))
        except (TypeError, ValueError):
            f = 0.0
        if f > 0:
            out.append(f)
    return out


def append_shadow_entry(entry: Dict, path: Optional[Any] = None) -> str:
    """追加一行 JSON 到影子账本（append-only, 单行 LF）。返回实际路径。"""
    p = Path(path) if path else DEFAULT_LEDGER
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
    return str(p)


def s2a_generator_axis_write(
    mab: Any,
    generator: str,
    paradigm: str,
    evaluate_result: Optional[Dict],
    *,
    loop_id: str = "",
    ledger_path: Optional[Any] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict:
    """M2 写入端：生成器轴奖励 / 观测写入（三态开关，绝不抛出）。

    返回 dict:
      mode    : 生效模式 (off | shadow | on)
      status  : skipped | no_data | shadow_logged | recorded | error
      reward  : tanh(max|ICIR|*3)*0.3（口径与历史一致）
      n_arms  : on 模式下实际写入的臂数
      lines   : 逐行日志（调用方原样 print，保证"不静默"）

    off   → 显式跳过（可见状态行）;
    shadow→ 追加观测账本（默认；只写账本, 绝不动 MAB）;
    on    → 写 MAB（兼容别名 FUXI_LEGACY_GENERATOR_PROXY_REWARD=on）。
    """
    e = os.environ if env is None else env
    mode = resolve_write_mode(e)
    rep: Dict[str, Any] = {"schema": SCHEMA, "mode": mode, "status": "skipped",
                           "lines": []}
    lines: List[str] = rep["lines"]

    raw = str(e.get(WRITE_ENV, "") or "").strip().lower()
    if raw and raw not in WRITE_MODES:
        lines.append(f"  [S2a] ⚠️ {WRITE_ENV}={raw!r} 非法 → 按默认 "
                     f"{DEFAULT_WRITE_MODE} 处理")

    try:
        er = evaluate_result if isinstance(evaluate_result, dict) else {}
        icirs = _icir_abs_list(er.get("results"))
        if not icirs or not paradigm:
            rep["status"] = "no_data"
            lines.append(f"  [S2a] 生成器轴写入[{mode}]: 本轮无 |ICIR|>0 读数 "
                         f"或 paradigm 缺失, 跳过 (不记 0)")
            return rep

        r_max = max(icirs)
        reward = float(math.tanh(r_max * ICIR_GAIN) * REWARD_CAP)
        success = bool(er.get("stage5_passed", 0) > 0
                       or er.get("eligible_for_jq", 0) > 0)
        rep.update({"reward": round(reward, 6), "icir_max": round(r_max, 6),
                    "n_icir": len(icirs), "success": success})

        if mode == "off":
            rep["status"] = "skipped"
            lines.append(f"  [S2a] 生成器轴写入=off (显式跳过; "
                         f"{WRITE_ENV}=shadow 启用观测账本)")
            return rep

        if mode == "shadow":
            n_arms_expected: Optional[int] = None
            try:
                n_arms_expected = len(mab.resolve_arms(paradigm))
            except Exception as _rae:  # 只读计数失败不阻塞, 但显式留痕
                lines.append(f"  [S2a] 生成器轴影子: resolve_arms 计数失败 "
                             f"(不阻塞): {_rae}")
            entry = {
                "schema": SCHEMA,
                "ts": datetime.now().isoformat(timespec="seconds"),
                "generator": str(generator),
                "paradigm": str(paradigm),
                "reward": round(reward, 6),
                "success": success,
                "icir_max": round(r_max, 6),
                "n_icir": len(icirs),
                "n_arms_expected": n_arms_expected,
                "source": "model_proxy",
                "mode": "shadow",
                "run_id": str(loop_id or ""),
            }
            path = append_shadow_entry(entry, ledger_path)
            rep["status"] = "shadow_logged"
            rep["ledger"] = path
            rep["n_arms_expected"] = n_arms_expected
            lines.append(f"  [S2a] 生成器轴影子记账: {generator} [{paradigm}] "
                         f"reward={reward:+.3f} (max|ICIR|={r_max:.3f}, "
                         f"应命中 {n_arms_expected} 臂未实际写入) → {path}")
            return rep

        # ── mode == "on": 写 MAB ─────────────────────────────
        try:
            n_arms = int(mab.record_generator_reward(
                paradigm, generator, reward,
                success=success, source="model_proxy") or 0)
        except Exception as _mwe:
            rep["status"] = "error"
            rep["error"] = f"{type(_mwe).__name__}: {_mwe}"
            lines.append(f"  [S2a] ⚠️ 生成器轴 MAB 写入失败 (不阻塞, 显式报告): "
                         f"{_mwe}")
            return rep

        rep["n_arms"] = n_arms
        rep["status"] = "recorded"
        if n_arms:
            lines.append(f"  [S2a] 生成器轴记账: {generator} → {n_arms} 臂 "
                         f"[{paradigm}] reward={reward:+.3f} "
                         f"(max|ICIR|={r_max:.3f})")
            # 🔴 P-20260917-023: 范式广播完整性**逐臂**核验（硬读数化，原样保留）。
            try:
                arms = mab.resolve_arms(paradigm)
                missing = [getattr(d, "name", "?") for d in arms
                           if not (getattr(d, "generator_rewards", None) or {})
                           .get(generator)]
                rep["expected_arms"] = len(arms)
                rep["missing_arms"] = missing
                lines.append(f"  [S2a] 范式广播: 本应 {len(arms)} 臂 / 实际 "
                             f"{len(arms) - len(missing)} 臂" +
                             ("" if not missing else
                              f" ⚠️ 未落臂 {missing[:3]} — 广播缺陷复现, "
                              f"应立即上报"))
            except Exception as _bce:
                lines.append(f"  [S2a] 范式广播核验跳过 (不阻塞): {_bce}")
        else:
            exp_n = 0
            try:
                exp_n = len(mab.resolve_arms(paradigm))
            except Exception:
                pass
            lines.append(f"  [S2a] ⚠️ 生成器轴记账未命中臂: paradigm={paradigm!r} "
                         f"(应命中 {exp_n} 臂 — 键不匹配或臂已移除)")
        return rep

    except Exception as e_outer:  # 顶层兜底：显式返回, 不抛出、不静默
        rep["status"] = "error"
        rep["error"] = f"{type(e_outer).__name__}: {e_outer}"
        rep["lines"] = [f"  [S2a] ⚠️ 生成器轴写入异常 (不阻塞, 显式报告): "
                        f"{e_outer}"]
        return rep
