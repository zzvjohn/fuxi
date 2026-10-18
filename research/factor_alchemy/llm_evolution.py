"""P-20260912-001: CogAlpha 认知进化助手 (llm 引擎的代际学习层)。

来源: Liu et al., "Cognitive Alpha Mining via LLM-Driven Code-Based Evolution",
      ACL 2026 Main Conference (Oral), arXiv:2511.18850 / aclanthology 2026.acl-long.538
      实证: CSI300 IC 0.0591 vs LightGBM 0.0269; 消融 +E(进化) / +A(自适应) / +G(多样引导)
      / +H(七级层级) 逐步提升 RankICIR。

本地化两个机制 (均只作用于 llm 引擎的**提示层**, 不改 MAB 主键/不改裁决逻辑/
不改公式 DSL, 与已有 P-20260910-005 五种转述模式叠加而非替代):
  (1) 思维进化 (thought evolution) — 在"思维/假设"空间而非参数空间做
      MUTATION(机制载体替换) 与 CROSSOVER(两父本机制复合)。
  (2) 自适应生成 (adaptive generation) — 上一代最好/最差因子归因,
      把"为什么好/为什么差"回灌下一轮生成 prompt。

开关: env FUXI_LLM_COG_EVOLVE = off | shadow(默认) | enforce
  - off     : 完全关闭, 零行为变化 (rollback 一行)
  - shadow  : 只记录, **不注入** prompt (零副作用); 状态落盘 + stdout 读数
  - enforce : 把两个提示块注入 llm 生成 prompt (仍是提示层, 无数据/裁决副作用)

状态: data/llm_evolution_state.json (跨轮持久: 父本池 / 归因记忆 / 轮次历史)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE = Path(__file__).resolve().parent.parent.parent       # e:/quant
# env 覆盖仅供测试隔离, 生产不设置
_STATE_ENV = os.environ.get("FUXI_COG_STATE_PATH", "").strip()
STATE_PATH = Path(_STATE_ENV) if _STATE_ENV else (BASE / "data" / "llm_evolution_state.json")
KEEP_ROUNDS = 20

_MODES = ("off", "shadow", "enforce")


def get_mode() -> str:
    m = str(os.environ.get("FUXI_LLM_COG_EVOLVE", "shadow") or "shadow").strip().lower()
    return m if m in _MODES else "shadow"


# ═══════════════════════════════════════════════════════════
# 状态读写
# ═══════════════════════════════════════════════════════════

def _empty_state() -> Dict[str, Any]:
    return {"version": "0.1", "total_rounds": 0, "parents": [],
            "parent_source": "none", "attribution": {}, "rounds": []}


def load_state() -> Dict[str, Any]:
    try:
        if STATE_PATH.exists():
            d = json.load(open(STATE_PATH, encoding="utf-8"))
            if isinstance(d, dict):
                base = _empty_state()
                base.update(d)
                return base
    except Exception:
        pass
    return _empty_state()


def save_state(state: Dict[str, Any]) -> None:
    from fuxi_v1.legacy_writers import assert_state_writer
    assert_state_writer(STATE_PATH)
    try:
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if len(state.get("rounds", [])) > KEEP_ROUNDS:
            state["rounds"] = state["rounds"][-KEEP_ROUNDS:]
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        json.dump(state, open(STATE_PATH, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"  [P-001] ⚠️ 状态落盘失败 (非阻塞): {e}")


# ═══════════════════════════════════════════════════════════
# 提示块构建
# ═══════════════════════════════════════════════════════════

def build_thought_evolution_block(parents: List[Dict], n_hint: int = 3) -> str:
    """思维进化块: 从父本做 MUTATION / CROSSOVER 派生。父本为空返回 ''。"""
    parents = [p for p in (parents or []) if p.get("formula") or p.get("hypothesis")]
    if not parents:
        return ""
    parts: List[str] = []
    parts.append("## 🧬 思维进化 (thought evolution — 在\"思维/假设\"空间进化, 与上方转述模式叠加不冲突)")
    parts.append("上一代父本假设 (可作为进化起点的\"基因\"):")
    for i, p in enumerate(parents[:3], 1):
        hypo = str(p.get("hypothesis") or "").strip()[:120] or "(无假设文本)"
        para = str(p.get("paradigm") or "").strip()
        tag = f" [范式={para}]" if para else ""
        parts.append(f"- P{i}{tag}: {hypo}")
        if p.get("formula"):
            parts.append(f"  `{str(p['formula'])[:150]}`")
    parts.append("")
    n_mut = max(1, min(2, n_hint))
    parts.append(f"请把本批候选中的**至少 {n_mut} 条**按下述\"思维算子\"从上述父本派生 "
                 "(其余候选自由设计; 转述模式仍按上文分配, 两者是不同维度):")
    parts.append("- MUTATION (思维变异): 保留父本的经济机制, 只替换其载体 — "
                 "换观测窗口 / 换变换函数 (rank、zscore、波动率标准化) / 换字段族 (价→量→资金流)。")
    parts.append("  ❌ 不得退化为参数微调 (那是 Light 转述模式); 必须改变实现路径而不只是数字。")
    if len(parents) >= 2:
        parts.append("- CROSSOVER (思维复合): 取两个**不同**父本的机制各一部分, "
                     "复合成此前不存在的新假设, 且必须能用单一 pandas 表达式求值。")
    parts.append("⚠️ 派生候选同样受硬禁止方向 / forbid motif / 前视禁令约束。")
    parts.append("")
    return "\n".join(parts)


def build_adaptive_block(attribution: Dict) -> str:
    """自适应生成块: 上一代最好/最差归因回灌。无归因返回 ''。"""
    if not attribution:
        return ""
    best, worst = attribution.get("best"), attribution.get("worst")
    if not best and not worst:
        return ""
    parts: List[str] = []
    parts.append(f"## 🔄 自适应生成 (adaptive generation — 上一代归因回灌, 第 "
                 f"{attribution.get('round_idx', '?')} 轮)")
    if best:
        parts.append(f"- ✅ 最好: {str(best.get('name', '?'))[:40]} "
                     f"(gate_level={best.get('gate_level', 0)}, "
                     f"IC={float(best.get('ic', 0) or 0):+.4f}, "
                     f"ICIR={float(best.get('icir', 0) or 0):.3f})")
        parts.append(f"  为什么好: {str(best.get('why', ''))[:200]}")
        if best.get("formula"):
            parts.append(f"  公式: `{str(best['formula'])[:130]}`")
    if worst:
        parts.append(f"- ❌ 最差: {str(worst.get('name', '?'))[:40]} "
                     f"(gate_level={worst.get('gate_level', 0)}, "
                     f"IC={float(worst.get('ic', 0) or 0):+.4f}, "
                     f"ICIR={float(worst.get('icir', 0) or 0):.3f})")
        parts.append(f"  为什么差: {str(worst.get('why', ''))[:200]}")
        if worst.get("formula"):
            parts.append(f"  公式: `{str(worst['formula'])[:130]}`")
    parts.append("")
    parts.append("请据此调整本批设计: 强化\"为什么好\"指出的**机制层**有效结构 "
                 "(不是复制公式 —— 库内相似因子会被行为冗余门降级); "
                 "规避\"为什么差\"指出的失效结构。")
    parts.append("")
    return "\n".join(parts)


def consume_blocks() -> Dict[str, Any]:
    """取上一代状态构建两个提示块。enforce 才应注入; shadow 仅供审计/打印。"""
    st = load_state()
    blocks = {
        "thought_evolution": build_thought_evolution_block(st.get("parents", [])),
        "adaptive": build_adaptive_block(st.get("attribution", {})),
    }
    return {
        "blocks": blocks,
        "n_parents": len(st.get("parents", []) or []),
        "parent_source": st.get("parent_source", "none"),
        "has_content": bool(blocks["thought_evolution"] or blocks["adaptive"]),
        "last_round": st.get("total_rounds", 0),
    }


# ═══════════════════════════════════════════════════════════
# 归因 (启发式优先, enforce 可选 LLM 精修)
# ═══════════════════════════════════════════════════════════

def _heuristic_why(row: Dict, good: bool) -> str:
    """归因文本。

    2026-09-12 修正: 初版"最差"归因直接引用 s1_reason, 当该候选**已通过 S1**
    时会出现自相矛盾 (e.g. "止步 gate_level=4; 拒因: [周频] IC=... ICIR=... ✅")。
    该文本在 enforce 下会喂给 LLM, 矛盾归因会误导生成 → 改为按 gate_level
    定位**第一个失败阶段**的拒因 (S1 用 s1_reason, S2/S3/S4/S5 各取其 reason)。
    """
    ic, icir = row.get("ic", 0.0), row.get("icir", 0.0)
    g = int(row.get("gate_level", 0) or 0)
    if good:
        s = f"gate_level={g} 深度通过; IC={ic:+.4f} / ICIR={icir:.3f} 截面排序稳定"
        if g >= 3:
            s += "; 且通过库内相关性/批次多样性筛查 (机制与存量因子正交)"
        return s
    if g <= 0:
        r = row.get("reason") or "未记录拒因"
        return f"止步 S1; 拒因: {str(r)[:160]}"
    stage_key = {1: "s2_reason", 2: "s3_reason", 3: "s4_reason", 4: "s5_reason"}.get(g)
    r = ((row.get(stage_key) if stage_key else "") or row.get("s5_reason")
         or row.get("s2_reason") or row.get("reason") or "未记录拒因")
    return f"止步 gate_level={g} (S{g + 1} 未通过); 拒因: {str(r)[:160]}"


def _llm_attribute(best: Optional[Dict], worst: Optional[Dict]) -> Optional[Dict[str, str]]:
    """enforce 模式下的 LLM 归因精修 (1 次调用, 失败返回 None → 回退启发式)。"""
    try:
        from llm_client import get_llm_client
        lines = []
        if best:
            lines.append(f"最好候选: {best.get('name')} | gate_level={best.get('gate_level')} "
                         f"| IC={best.get('ic'):+.4f} | ICIR={best.get('icir'):.3f} "
                         f"| 公式 {str(best.get('formula'))[:130]}")
        if worst:
            lines.append(f"最差候选: {worst.get('name')} | gate_level={worst.get('gate_level')} "
                         f"| IC={worst.get('ic'):+.4f} | ICIR={worst.get('icir'):.3f} "
                         f"| 公式 {str(worst.get('formula'))[:130]}")
        if not lines:
            return None
        user = ("以下是上一代 LLM 生成因子的评估结果。请分别用一句话回答: "
                "①最好那条为什么好 (机制层面, 不是\"IC 高\"这种同义反复); "
                "②最差那条为什么差。只输出两行, 格式: GOOD: ... / BAD: ...\n\n"
                + "\n".join(lines))
        cli = get_llm_client()
        resp = cli.chat_with_system(
            system_prompt="你是 A 股量化因子研究员, 擅长归因因子成败的机制原因。",
            user_prompt=user, temperature=0.3, max_tokens=256)
        out: Dict[str, str] = {}
        for ln in str(resp).splitlines():
            u = ln.strip()
            if u.upper().startswith("GOOD:"):
                out["good"] = u[5:].strip()
            elif u.upper().startswith("BAD:"):
                out["bad"] = u[4:].strip()
        return out or None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
# 主入口: 记录本轮 (Ralph Phase E 之后调用)
# ═══════════════════════════════════════════════════════════

def record_round(round_idx: int, results: List[Any],
                 mode: Optional[str] = None,
                 use_llm_attr: Optional[bool] = None,
                 verbose: bool = True) -> Dict[str, Any]:
    """把本轮评估结果写成下一代父本 + 归因记忆 (跨轮持久)。零裁决。"""
    mode = mode or get_mode()
    if mode == "off":
        return {"mode": "off", "n_parents": 0, "injected": False}

    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(results or []):
        f = str(getattr(r, "formula", "") or "")
        rows.append({
            "name": str(getattr(r, "factor_name", f"r{i}"))[:60],
            "formula": f,
            "hypothesis": str(getattr(r, "hypothesis", "") or ""),
            "paradigm": str(getattr(r, "paradigm", "") or ""),
            "ic": float(getattr(r, "s1_ic", 0.0) or 0.0),
            "icir": float(getattr(r, "s1_icir", 0.0) or 0.0),
            "s1_passed": bool(getattr(r, "s1_passed", False)),
            "gate_level": int(getattr(r, "gate_level", 0) or 0),
            "reason": str(getattr(r, "s1_reason", "") or ""),
            "s2_reason": str(getattr(r, "s2_reason", "") or ""),
            "s3_reason": str(getattr(r, "s3_reason", "") or ""),
            "s4_reason": str(getattr(r, "s4_reason", "") or ""),
            "s5_reason": str(getattr(r, "s5_reason", "") or ""),
        })
    rows = [r for r in rows if r["formula"]]
    if not rows:
        return {"mode": mode, "n_parents": 0, "injected": False, "reason": "无候选"}

    parents = [r for r in rows if r["s1_passed"]]
    parent_source = "s1_passed"
    if not parents:
        # 退化路径: 无 S1 通过时, 用 |ICIR| top-2 作父本 (本轮信息量最大者)
        parents = sorted(rows, key=lambda x: -abs(x["icir"]))[:2]
        parent_source = "top_icir_fallback"
    parents = parents[:3]

    ranked = sorted(rows, key=lambda x: (-x["gate_level"], -abs(x["icir"])))
    best = ranked[0] if ranked else None
    worst = ranked[-1] if len(ranked) > 1 else None

    attr: Dict[str, Any] = {"round_idx": round_idx, "source": "heuristic"}
    if best:
        attr["best"] = {**{k: best[k] for k in
                           ("name", "formula", "gate_level", "ic", "icir")},
                        "why": _heuristic_why(best, True)}
    if worst:
        attr["worst"] = {**{k: worst[k] for k in
                            ("name", "formula", "gate_level", "ic", "icir")},
                         "why": _heuristic_why(worst, False)}

    if use_llm_attr is None:
        use_llm_attr = (mode == "enforce"
                        and os.environ.get("FUXI_COG_ATTR_LLM", "on") != "off")
    if use_llm_attr:
        refined = _llm_attribute(attr.get("best"), attr.get("worst"))
        if refined:
            if refined.get("good") and attr.get("best"):
                attr["best"]["why"] = refined["good"]
            if refined.get("bad") and attr.get("worst"):
                attr["worst"]["why"] = refined["bad"]
            attr["source"] = "llm"

    st = load_state()
    st["total_rounds"] = int(st.get("total_rounds", 0)) + 1
    st["parents"] = parents
    st["parent_source"] = parent_source
    st["attribution"] = attr
    st["rounds"] = (st.get("rounds", []) + [{
        "round_idx": round_idx,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "n_results": len(rows),
        "n_s1_passed": sum(1 for r in rows if r["s1_passed"]),
        "n_parents": len(parents),
        "parent_source": parent_source,
        "best": (attr.get("best") or {}).get("name"),
        "worst": (attr.get("worst") or {}).get("name"),
        "attr_source": attr.get("source"),
    }])
    save_state(st)

    if verbose:
        print(f"  [P-001] CogAlpha 认知进化 {mode}: 本轮 {len(rows)} 结果 / "
              f"S1 过 {sum(1 for r in rows if r['s1_passed'])} → "
              f"父本 {len(parents)} 条 ({parent_source}), 归因源={attr.get('source')} "
              f"(零裁决{'·已在下一轮注入' if mode == 'enforce' else '·shadow 未注入'})")
    return {"mode": mode, "n_parents": len(parents), "parent_source": parent_source,
            "attr_source": attr.get("source"),
            "injected": bool(mode == "enforce"),
            "best": (attr.get("best") or {}).get("name"),
            "worst": (attr.get("worst") or {}).get("name")}


def preview(verbose: bool = True) -> Dict[str, Any]:
    """审计入口: 打印当前将注入的两个提示块 (shadow 用)。"""
    c = consume_blocks()
    if verbose:
        print(f"  [P-001] 认知进化预览 (mode={get_mode()}): 父本 {c['n_parents']} 条 "
              f"({c['parent_source']}), 上轮 #{c['last_round']}, "
              f"内容={'有' if c['has_content'] else '无'}")
        for k, v in c["blocks"].items():
            print(f"      - {k}: {len(v)} chars")
    return c


if __name__ == "__main__":
    import sys
    if "--preview" in sys.argv:
        preview()
    elif "--record-selftest" in sys.argv:
        class _R:
            def __init__(s, n, ic, icir, g, ok):
                s.factor_name, s.formula = n, f"close.rolling(20).mean() - close"
                s.hypothesis = "20日均线偏离"
                s.paradigm = "均值回复"
                s.s1_ic, s.s1_icir, s.gate_level, s.s1_passed = ic, icir, g, ok
                s.s1_reason = s.s2_reason = s.s5_reason = ""
        res = [_R("a", 0.03, 0.25, 2, True), _R("b", 0.005, 0.04, 0, False)]
        print(record_round(99, res, verbose=True))
        c = preview(verbose=False)
        print(c["blocks"]["thought_evolution"][:600])
    else:
        print(f"mode={get_mode()}  (--preview | --record-selftest)")
