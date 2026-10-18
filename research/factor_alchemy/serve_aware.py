"""P2-1 (M5 最小切口, 2026-09-28): 生成器路由服务矩阵感知 + 空轮归因 + 逃生舱补空轮。

背景（五轮连续采样实证, reports/fuxi_5rounds_drought_20260928/）:
  · MAB 41 臂跨 16 范式, Forge 仅 8 个 profile (MATURE_PARADIGM_PROFILES)
    ⇒ 24 臂 (59%) 结构性不可服务; 轮转到无 profile 范式时
    `_generate_via_forge` 直接 return [] ([Forge/R3] 明确回退 GP Breed)
    ⇒ 空轮 (R3-R5 连续 3 轮 gen=0, source=gp_breed_fallback);
  · 26 臂 @11p 确定性前瞻: 6 可服务 + 20 不可服务 ⇒ 预期 77% 空轮;
  · `s5_streak_event` 把「0 候选 / 未达 S5 / eval≠valid」一律归 unassessed
    ⇒ `_breed_fail_streak` 连续空轮不增 (自 09-24 streak=0) ⇒ 逃生舱盲区。

四个独立开关（各默认 = 零行为变化; 一次只翻一门）:
  FUXI_ROUTE_SERVE_AWARE : off | shadow | on(默认, 2026-10-10 翻)
      on ⇒ MAB 选定生成器后做服务性检查: 目标范式无 Forge profile 且
      生成器为 forge 时, 降级 llm (LLM 生成不依赖 profile; 既有设计语言
      「未覆盖范式 → LLM 擅长冷启动」见 ralph_loop.run() 退避块)。
  FUXI_SERVE_AWARE_SCOPE : global | whitelist(默认, P2-1b 2026-09-28)
      on 档的降级范围。whitelist ⇒ 仅对 SERVE_AWARE_WHITELIST 内范式
      降级 (P2-1 实证范式异质性大、全局 on 不安全); global ⇒ P2-1
      原全局语义 (仅复现用)。off/shadow 档不受影响 (shadow 打印区分)。
  FUXI_EMPTY_ROUND_ATTR  : off | shadow(默认) | on
      空轮 (n_candidates==0) 时落 data/empty_round_attribution.jsonl 归因,
      reason 分档: no_forge_profile / forge_empty / gp_empty / llm_empty /
      fallback_empty / other。(纯观测, 不改变任何裁决字段)
  FUXI_BREED_FAIL_EMPTY  : off | shadow(默认) | on
      on ⇒ 空轮计入 breed_empty_streak (与 breed_fail_streak 平行, 宿于
      同一状态文件 data/breed_fail_streak.json), 达到
      FUXI_BREED_FAIL_EMPTY_THRESHOLD (默认 3) 时参与 LLM 退避触发;
      有产出即重置。
      🔴 与经济失败连击分离: 空轮 ≠ 经济失败 (后者语义为「到达 S5 且
      全部有效经济失败」, 阈值 2); 两条触发线并行、不互相污染计数。

硬边界:
  · 不降阈值 / 不动 S5 口径 / 不写 MAB 状态 / 不改裁决字段;
  · off 档 100% 历史行为; shadow 只读 + 只打印 (+ 空轮归因落盘);
  · 全部函数纯逻辑, 异常由调用方兜底 (不抛、不静默降级为错误结论)。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

SERVE_ENV = "FUXI_ROUTE_SERVE_AWARE"
EMPTY_ATTR_ENV = "FUXI_EMPTY_ROUND_ATTR"
BREED_EMPTY_ENV = "FUXI_BREED_FAIL_EMPTY"
BREED_EMPTY_TH_ENV = "FUXI_BREED_FAIL_EMPTY_THRESHOLD"
SERVE_SCOPE_ENV = "FUXI_SERVE_AWARE_SCOPE"

MODES = ("off", "shadow", "on")
DEFAULT_MODE = "shadow"
DEFAULT_EMPTY_THRESHOLD = 3

# ── P2-1b (2026-09-28): 范式级白名单降级 ─────────────────────────────
# P2-1 门评估实证（reports/fuxi_p2m1_gate_20260928/P2M1_GATE_REPORT.md §4.3/§6）:
# 降级效果**依赖范式**，全局 on 不安全 —— 资金流 favor llm（0 → 15 候选 /
# S1 66.7% / S2+ 26.7%），事件驱动 oppose（llm S1 9.1% < forge fallback 25.0%）。
# §6.3 建议「把全局降级改造为**范式级白名单降级**（per-paradigm），
#       白名单须**预注册**（范式集合写死在配置中，不在运行期动态扩展）」。
#
# 🔴 预注册纪律（修改本集合 = 一次正式变更，须伴代码审查 + 报告登记）:
#   · 准入判定（P2-1b 预注册）: 仅当某范式完成 **≥2 轮 forge vs llm 严格
#     配对**（同窗、唯一变量 = 生成器）且 llm 侧在「候选数 / S1 通过率 /
#     S2+ 进入数」全维不劣、且 S2+ > 0 时，方可加入本白名单；
#   · 已准入: 资金流（P2-1 r3-r5 forge 0 候选×3 vs 轮 B llm 15 候选/
#     S1 66.7%/S2+ 26.7%）;
#   · 反向证据范式不得加入: 事件驱动（P2-1 轮 A vs C 严格配对 oppose）;
#   · 无配对证据的范式不得加入（宁缺毋滥）: 北向资金 / 情绪×日内 / 截面交互 /
#     行业轮动 / cross_asset / 行为金融 / 高频微观结构（7 个无 profile 范式）;
#   · 空集合 = 等价关闭降级（fail-safe）。
SERVE_AWARE_WHITELIST = ("资金流",)

SCOPES = ("global", "whitelist")
# 默认 whitelist = 收紧方向（P2-1 已淘汰全局语义为「不安全」; on 档现未在
# 生产启用 ⇒ 本默认不触任何生产行为; 需要复现 P2-1 全局语义时显式
# FUXI_SERVE_AWARE_SCOPE=global）。
DEFAULT_SCOPE = "whitelist"

ATTR_LEDGER_NAME = "empty_round_attribution.jsonl"

__all__ = [
    "SERVE_ENV", "EMPTY_ATTR_ENV", "BREED_EMPTY_ENV", "BREED_EMPTY_TH_ENV",
    "SERVE_SCOPE_ENV", "SERVE_AWARE_WHITELIST", "SCOPES", "DEFAULT_SCOPE",
    "MODES", "DEFAULT_MODE", "DEFAULT_EMPTY_THRESHOLD", "ATTR_LEDGER_NAME",
    "resolve_serve_aware_mode", "resolve_serve_aware_scope",
    "resolve_empty_attr_mode",
    "resolve_breed_empty_mode", "resolve_breed_empty_threshold",
    "serve_aware_adjust", "classify_empty_reason", "record_empty_round",
    "format_empty_readout",
]


def _resolve_mode(env_name: str, env: Optional[Dict[str, str]] = None) -> str:
    """通用三档开关解析 (非法值回落默认 shadow; off/shadow/on 精确匹配)。"""
    e = os.environ if env is None else env
    raw = str(e.get(env_name, "") or "").strip().lower()
    if raw in MODES:
        return raw
    return DEFAULT_MODE


def resolve_serve_aware_mode(env: Optional[Dict[str, str]] = None) -> str:
    """FUXI_ROUTE_SERVE_AWARE 解析 (off | shadow | on(默认))。

    2026-10-10 翻默认 (P-20261010-011): 未设置回落 on (原 shadow)。显式
    off/shadow/on 原样返回; 非法值仍回落 DEFAULT_MODE=shadow (fail-safe,
    不与「未设置」混淆)。FUXI_EMPTY_ROUND_ATTR / FUXI_BREED_FAIL_EMPTY 仍
    共用 DEFAULT_MODE=shadow, 本翻默认不外溢到那两开关。
    """
    e = os.environ if env is None else env
    raw = str(e.get(SERVE_ENV, "") or "").strip().lower()
    if raw in MODES:
        return raw
    if raw == "":
        return "on"
    return DEFAULT_MODE


def resolve_serve_aware_scope(env: Optional[Dict[str, str]] = None) -> str:
    """FUXI_SERVE_AWARE_SCOPE 解析 (global | whitelist(默认))。

    P2-1b (2026-09-28): on 档的降级**范围**。
      global    → 对所有「无 Forge profile」范式降级 (P2-1 实验原语义,
                  已被 P2-1 判定为不安全; 保留仅为复现);
      whitelist → 仅对 SERVE_AWARE_WHITELIST 内范式降级, 其余保持 forge
                  (P2-1b 改造后的默认语义; fail-safe)。
    非法值回落默认 whitelist。
    """
    e = os.environ if env is None else env
    raw = str(e.get(SERVE_SCOPE_ENV, "") or "").strip().lower()
    if raw in SCOPES:
        return raw
    return DEFAULT_SCOPE


def _whitelist(wl) -> tuple:
    """白名单解析: 调用方显式传入 (单测/隔离) 优先; 否则用预注册常量。"""
    if wl is None:
        return tuple(SERVE_AWARE_WHITELIST)
    return tuple(str(x) for x in wl)


def resolve_empty_attr_mode(env: Optional[Dict[str, str]] = None) -> str:
    """FUXI_EMPTY_ROUND_ATTR 解析 (off | shadow(默认) | on)。"""
    return _resolve_mode(EMPTY_ATTR_ENV, env)


def resolve_breed_empty_mode(env: Optional[Dict[str, str]] = None) -> str:
    """FUXI_BREED_FAIL_EMPTY 解析 (off | shadow(默认) | on)。"""
    return _resolve_mode(BREED_EMPTY_ENV, env)


def resolve_breed_empty_threshold(env: Optional[Dict[str, str]] = None) -> int:
    """FUXI_BREED_FAIL_EMPTY_THRESHOLD 解析 (默认 3, 下限 1)。"""
    e = os.environ if env is None else env
    try:
        v = int(str(e.get(BREED_EMPTY_TH_ENV, "") or "").strip() or DEFAULT_EMPTY_THRESHOLD)
    except Exception:
        v = DEFAULT_EMPTY_THRESHOLD
    return max(1, v)


def _profiles(profiles: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """服务矩阵来源: 优先调用方传入; 否则惰性 import (便于隔离单测)。"""
    if profiles is not None:
        return profiles
    from forge.paradigm_profiles import MATURE_PARADIGM_PROFILES
    return MATURE_PARADIGM_PROFILES


def serve_aware_adjust(direction: Any, generator: str,
                       mode: Optional[str] = None,
                       profiles: Optional[Dict[str, Any]] = None,
                       scope: Optional[str] = None,
                       whitelist=None) -> str:
    """服务矩阵感知降级 (纯函数)。

    仅当 generator == "forge" 且目标范式不在 profiles 内时动作:
      off    → 返回原 generator (不动作, 100% 历史行为);
      shadow → 打印「本应降级」读数, 返回值不变 (零回归观测);
      on     → scope="global" 返回 "llm"; scope="whitelist" 仅当范式在
               SERVE_AWARE_WHITELIST 内才返回 "llm", 白名单外保持 forge。

    🔴 P2-1b (2026-09-28) 语义变更记录: 改造前 on 档 = 全局降级 (P2-1
    实验语义); 改造后默认 scope=whitelist ⇒ on 档 = 白名单降级。P2-1
    全局语义可用 FUXI_SERVE_AWARE_SCOPE=global 显式复现。
    (on 档此前未在生产启用 ⇒ 本变更不触任何生产行为路径。)

    其余生成器 (llm / gp_breed / gp_evolve / cross_breed) 恒不变 —
    gp_breed 走模板变异, 不依赖 profile。
    """
    mode = mode if mode in MODES else resolve_serve_aware_mode()
    if mode == "off" or generator != "forge":
        return generator
    para = str(getattr(direction, "paradigm", "") or "")
    if not para:
        return generator
    if para in _profiles(profiles):
        return generator
    scope = scope if scope in SCOPES else resolve_serve_aware_scope()
    wl = _whitelist(whitelist)
    in_wl = para in wl
    name = str(getattr(direction, "name", "?") or "?")
    if mode == "shadow":
        if scope == "whitelist" and not in_wl:
            print(f"  [M5/serve/shadow] 方向={name} 范式={para!r} 无 Forge profile "
                  f"⇒ 本应降级但白名单外 ⇒ 拦截 (P2-1b; wl={list(wl)})")
        else:
            print(f"  [M5/serve/shadow] 方向={name} 范式={para!r} 无 Forge profile "
                  f"⇒ 本应降级 forge→llm (FUXI_ROUTE_SERVE_AWARE=on 启用)")
        return generator
    if scope == "whitelist" and not in_wl:
        print(f"  [M5/serve] 方向={name} 范式={para!r} 无 profile 且白名单外 "
              f"⇒ 保持 forge (P2-1b; wl={list(wl)})")
        return generator
    print(f"  [M5/serve] 方向={name} 范式={para!r} 无 Forge profile "
          f"⇒ 生成器降级 forge→llm")
    return "llm"


def classify_empty_reason(paradigm: str, generator: str, source: str,
                          profiles: Optional[Dict[str, Any]] = None) -> str:
    """空轮归因分档 (纯函数)。

    层序: source=="gp_breed_fallback" 表示层 1 (forge) 已空且层 2 (GP
    fallback) 亦空 ⇒ reason 同时标注两层; 否则按 generator 判层 1 原因。
    """
    p = str(paradigm or "")
    gen = str(generator or "")
    src = str(source or "")
    no_profile = bool(p) and p not in _profiles(profiles)
    if src == "gp_breed_fallback":
        return ("no_forge_profile+fallback_empty" if no_profile
                else "forge_empty+fallback_empty")
    if gen == "forge":
        return "no_forge_profile" if no_profile else "forge_empty"
    if gen in ("gp_breed", "gp_breed_fallback", "gp_evolve", "cross_breed"):
        return "gp_empty"
    if gen == "llm":
        return "llm_empty"
    return "other"


def record_empty_round(loop_id: str, paradigm: str, generator: str,
                       source: str, n_candidates: int,
                       empty_streak: int = 0,
                       data_dir: Optional[Path] = None,
                       mode: Optional[str] = None,
                       profiles: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """空轮归因落盘 (append, jsonl)。

    仅当 mode != "off" 且 n_candidates <= 0 时动作; 返回所落记录 (供调用方
    打印), 否则 None。落盘路径 data_dir/empty_round_attribution.jsonl —
    纯新增观测文件, 不消费、不改任何既有账本。
    """
    mode = mode if mode in MODES else resolve_empty_attr_mode()
    if mode == "off" or int(n_candidates or 0) > 0:
        return None
    reason = classify_empty_reason(paradigm, generator, source, profiles)
    rec = {
        "ts": datetime.now().isoformat(),
        "loop_id": str(loop_id or ""),
        "paradigm": str(paradigm or ""),
        "generator": str(generator or ""),
        "source": str(source or ""),
        "reason": reason,
        "n_candidates": int(n_candidates or 0),
        "empty_streak_before": int(empty_streak or 0),
        "attr_mode": mode,
    }
    d = Path(data_dir) if data_dir is not None else (Path(__file__).resolve().parents[2] / "data")
    path = d / ATTR_LEDGER_NAME
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"  [M5/空轮归因] 落盘失败 (非阻塞): {type(exc).__name__}: {exc}")
        return None
    if mode == "on":
        print(f"  [M5/空轮归因] ⚠️ 空轮已记录: reason={reason} "
              f"generator={generator} source={source} → {path.name}")
    return rec


def format_empty_readout(would_be: int, threshold: int) -> str:
    """shadow 档空轮读数文案 (零回归: 只打印不改状态)。"""
    return (f"  [退避/shadow] 本轮空轮 (gen=0) ⇒ 本应计入空轮连击 "
            f"{int(would_be)}/{int(threshold)} (FUXI_BREED_FAIL_EMPTY=on 启用)")
