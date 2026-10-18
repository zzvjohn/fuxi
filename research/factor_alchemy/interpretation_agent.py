# -*- coding: utf-8 -*-
"""
L1 解读层 — JQ 回测结果的 LLM 深度归因
========================================
伏羲 agentic 演进 L1: 在 D+ 闭环中, 对每个 JQ 回测因子做结构化归因。

设计原则 (2026-08-14):
- 只读"注释者", 不做"裁判": 不修改 form_from_jq 数值结论、不动 distill_motif 规则
- 产出必须是结构化 verdict 枚举, 否则下游无法消费
- LLM 不可用时启发式降级, 绝不阻塞 D+ 主流程

verdict 四枚举 (下游分流器):
  execution_failure   JQ 崩但 local 信号有效 → 方向不罚, 修复执行 (L2)
  direction_falsified JQ 证伪方向 → MAB 降权 + motif 惩罚
  direction_confirmed JQ 正向验证 → 方向奖励
  data_issue          JQ 数据异常/无法归因 → 不进入任何决策

下游消费三路:
  ① llm_generator.build_prompt    → 避坑提示 (软消费)
  ② experience_memory.distill_motif_knowledge → 两级蒸馏 (硬消费, 走既有门槛)
  ③ ralph_loop.jq_feedback Step 3.6 → MAB 方向级精细化

用法:
  from interpretation_agent import interpret_batch
  result = interpret_batch(jq_backtest_result, memory, use_llm=True)
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent.parent / "data"
INTERP_DIR = DATA_DIR / "interpretations"

VERDICTS = (
    "execution_failure",
    "direction_falsified",
    "direction_confirmed",
    "data_issue",
)

# 灰区边界 (与 form_from_jq 阈值一致)
HARD_FAIL_RET = -20.0
HARD_FAIL_SHARPE = -0.5
PASS_RET = 50.0
PASS_SHARPE = 0.4
# local ICIR 证据阈值: 超过此值认为"方向有 local 证据, JQ 崩属于执行问题"
LOCAL_ICIR_EVIDENCE = 0.3


# ═══════════════════════════════════════════════════════════
# 启发式预分类 (LLM 离线降级)
# ═══════════════════════════════════════════════════════════

def classify_by_rules(factor: Dict) -> Dict:
    """纯规则分类 — 保证 LLM 不可用时管道不断"""
    jq_ret = float(factor.get("jq_return", 0) or 0)
    jq_sharpe = float(factor.get("jq_sharpe", 0) or 0)
    local_icir = abs(float(factor.get("local_icir", 0) or 0))

    if jq_ret < HARD_FAIL_RET or jq_sharpe < HARD_FAIL_SHARPE:
        if local_icir >= LOCAL_ICIR_EVIDENCE:
            verdict = "execution_failure"
            conf = 0.7
            hints = [
                "IC 有效但 JQ 执行崩溃: 检查换手率构造/停牌处理/滑点敏感度",
                "保留方向, 优先修复执行层 (L2)",
            ]
        else:
            verdict = "direction_falsified"
            conf = 0.7
            hints = ["local 与 JQ 双重否定, 方向被证伪", "降低该方向探索预算"]
    elif -20.0 <= jq_ret < 0:
        verdict = "direction_falsified"
        conf = 0.5
        hints = ["软负收益: 未达硬禁止但持续亏损", "观察是否命中已知软警告"]
    elif 0 <= jq_ret <= PASS_RET:
        verdict = "direction_confirmed"
        conf = 0.5
        hints = ["弱正向: 未达通过阈但方向不亏", "可尝试低换手变体冲击通过线"]
    else:
        verdict = "direction_confirmed"
        conf = 0.9
        hints = ["JQ 正向验证, 方向确认", "可围绕该结构做变体育种"]

    return {
        "verdict": verdict,
        "confidence": conf,
        "motif_hits": [],
        "motif_falsified": [],
        "narrative": _rule_narrative(verdict, jq_ret, jq_sharpe, local_icir),
        "next_round_hints": hints,
        "source": "rule_fallback",
    }


def _rule_narrative(verdict: str, jq_ret: float, jq_sharpe: float, local_icir: float) -> str:
    return (
        f"[规则降级] verdict={verdict}; JQ {jq_ret:.1f}%/Sharpe {jq_sharpe:.2f}; "
        f"local ICIR={local_icir:.3f}"
    )


# ═══════════════════════════════════════════════════════════
# LLM 归因
# ═══════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "你是 A 股量化研究主管，负责对聚宽(JQ)回测结果做深度归因。"
    "你的输出必须是严格 JSON，字段: verdict, confidence, motif_hits, motif_falsified, "
    "narrative, next_round_hints。"
    "verdict 只能取以下四个值之一: execution_failure / direction_falsified / "
    "direction_confirmed / data_issue。"
    "判据: local ICIR 明显有效但 JQ 大幅亏损 → execution_failure; "
    "local 与 JQ 双重否定 → direction_falsified; JQ 正向 → direction_confirmed; "
    "数据异常无法归因 → data_issue。"
    "next_round_hints 是给下一轮因子生成器的具体避坑建议 (2-4 条, 每条 ≤30 字)。"
    "narrative 用 2-3 句中文说明归因逻辑。不要输出 JSON 以外的任何内容。"
)


def _build_llm_user_prompt(factor: Dict, paradigm_history: List[Dict]) -> str:
    lines = ["请对以下 JQ 回测结果做深度归因。", ""]
    lines.append(f"因子名: {factor.get('factor_name', '?')}")
    lines.append(f"范式: {factor.get('paradigm', '?')}")
    lines.append(f"公式: {str(factor.get('formula', ''))[:200]}")
    lines.append(
        f"JQ 结果: return={factor.get('jq_return', 0)}% / "
        f"Sharpe={factor.get('jq_sharpe', 0)} / MaxDD={factor.get('jq_maxdd', 0)}%"
    )
    lines.append(
        f"Local 信号: IC={factor.get('local_ic', 0):.4f} / "
        f"ICIR={factor.get('local_icir', 0):.4f}"
    )
    lines.append(f"JQ 平台根因记录: {factor.get('root_cause', '无')}")
    lines.append(f"评级: {factor.get('jq_rating', factor.get('jq_composite_contribution', '?'))}")
    jq_ic = factor.get("jq_ic")
    jq_icir = factor.get("jq_icir")
    if jq_ic is not None or jq_icir is not None:
        lines.append(f"JQ 单因子 IC={jq_ic} / ICIR={jq_icir} (P-001)")
    lines.append("")
    if paradigm_history:
        lines.append(f"该方向历史 JQ 表现 (n={len(paradigm_history)}):")
        for h in paradigm_history[:8]:
            lines.append(
                f"  - {h.get('factor_name', '?')}: {h.get('jq_return', 0)}% / "
                f"Sharpe {h.get('jq_sharpe', 0)}"
            )
    lines.append("")
    lines.append("请输出归因 JSON。")
    return "\n".join(lines)


def _parse_llm_json(text: str, fallback: Dict) -> Dict:
    """从 LLM 响应中提取 JSON, 校验 verdict 合法性, 失败回退启发式"""
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return fallback
        obj = json.loads(m.group(0))
        verdict = str(obj.get("verdict", ""))
        if verdict not in VERDICTS:
            return fallback
        return {
            "verdict": verdict,
            "confidence": float(obj.get("confidence", 0.5)),
            "motif_hits": [str(x) for x in obj.get("motif_hits", [])][:5],
            "motif_falsified": [str(x) for x in obj.get("motif_falsified", [])][:5],
            "narrative": str(obj.get("narrative", ""))[:400],
            "next_round_hints": [str(x)[:60] for x in obj.get("next_round_hints", [])][:4],
            "source": "llm",
        }
    except Exception:
        return fallback


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def _collect_paradigm_history(memory, paradigm: str) -> List[Dict]:
    """从 memory attempts 收集同范式历史 JQ 结果"""
    history = []
    try:
        for a in memory.data.get("attempts", []):
            if a.get("paradigm") == paradigm and a.get("jq_return") is not None:
                history.append(a)
    except Exception:
        pass
    history.sort(key=lambda a: str(a.get("jq_verified_at", "")), reverse=True)
    return history


def interpret_factor(factor: Dict, memory=None, use_llm: bool = True) -> Dict:
    """
    对单个 factor 做归因。返回结构化报告 (verdict 分流器)。

    use_llm=False 时走纯规则降级 (测试/离线模式)。
    """
    fallback = classify_by_rules(factor)

    if not use_llm:
        return fallback

    try:
        from llm_client import get_llm_client
        client = get_llm_client()
        history = _collect_paradigm_history(memory, str(factor.get("paradigm", ""))) if memory else []
        user_prompt = _build_llm_user_prompt(factor, history)
        text = client.chat_with_system(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        return _parse_llm_json(text, fallback)
    except Exception as e:
        print(f"  [L1] ⚠️ LLM 归因失败 ({factor.get('factor_name', '?')}): {e} → 规则降级")
        return fallback


def interpret_batch(jq_backtest_result: Dict, memory=None, use_llm: bool = True) -> Dict:
    """
    对 jq_backtest_result 中每个 factor 做归因, 结果挂回 factor["interpretation"],
    并落盘 data/interpretations/interpretation_<batch_id>.json。
    """
    factors = jq_backtest_result.get("factors", [])
    for f in factors:
        f["interpretation"] = interpret_factor(f, memory=memory, use_llm=use_llm)

    # 落盘
    try:
        INTERP_DIR.mkdir(parents=True, exist_ok=True)
        batch_id = str(jq_backtest_result.get("batch_id", datetime.now().strftime("%Y%m%d_%H%M%S")))
        safe_id = re.sub(r"[^\w\-]", "_", batch_id)
        out_path = INTERP_DIR / f"interpretation_{safe_id}.json"
        payload = {
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "factors": [
                {
                    "factor_name": f.get("factor_name", ""),
                    "paradigm": f.get("paradigm", ""),
                    "jq_return": f.get("jq_return", 0),
                    "interpretation": f.get("interpretation", {}),
                }
                for f in factors
            ],
        }
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        print(f"  [L1] 归因报告落盘: {out_path}")
    except Exception as e:
        print(f"  [L1] ⚠️ 落盘失败: {e}")

    verdict_counts = {}
    for f in factors:
        v = f.get("interpretation", {}).get("verdict", "?")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    print(f"  [L1] verdict 分布: {verdict_counts}")

    return {"status": "ok", "n_factors": len(factors), "verdict_counts": verdict_counts}


def attach_to_memory(memory, factors: List[Dict]) -> int:
    """把 interpretation 写回 memory.attempts 对应条目 (按 factor_name 匹配)"""
    n = 0
    for f in factors:
        name = f.get("factor_name", "")
        interp = f.get("interpretation") or {}
        if not name or not interp:
            continue
        for a in memory.data.get("attempts", []):
            if a.get("factor_name") == name or a.get("name") == name:
                a["interpretation"] = interp
                n += 1
                break
    if n:
        memory._save()
    return n


def summarize(factors: List[Dict]) -> str:
    """生成一行归因概览, 供 trigger_d_plus 打印"""
    parts = []
    for f in factors:
        interp = f.get("interpretation") or {}
        v = interp.get("verdict", "?")
        parts.append(f"{f.get('factor_name', '?')}={v}")
    return ", ".join(parts)


if __name__ == "__main__":
    demo = {
        "batch_id": "demo",
        "factors": [
            {
                "factor_name": "breed008_demo",
                "formula": "close.rolling(20).mean()",
                "paradigm": "活跃资金流",
                "jq_return": -66.4,
                "jq_sharpe": -0.9,
                "jq_maxdd": -70.0,
                "local_ic": 0.03,
                "local_icir": 0.38,
                "root_cause": "IC有效但执行崩溃",
            }
        ],
    }
    result = interpret_batch(demo, use_llm=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
