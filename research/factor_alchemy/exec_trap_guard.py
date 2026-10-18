"""P-20260911-003: 执行陷阱守卫影子 (GoAnt Execution Trap 本地化)。

背景 (两条独立来源相互论证):
  1) GoAnt (arXiv:2609.08719, 2026): Execution Trap — 因子高 RankIC 但计入执行成本/
     滑点/冲击后净收益为负; 单代理贪心优化预测代理会塌缩到局部次优。
  2) PandaAI (2026, 金融毒性): 关闭换手约束的 Factor6 原始 ICIR 0.248 高于完整版
     0.193, 但日换手 >80%, 计入 15bp 佣金 + 5bp 滑点后净表现骤降 —— 独立印证
     「统计指标高 ≠ 可交易」。

定位: 纯诊断维度 (shadow), 不改 S5 门槛主逻辑, 零裁决。对 JQ-eligible (S5 通过)
      候选计算「成本后净收益」估计并标注 exec_trap 标签, 落 data/exec_trap_shadow.json。

估计口径 (假设透明, 可 env 覆盖):
  - gross_excess_floor = min(excess_2025, excess_2026)  ← 两年稳健下限
  - annual_turns: 周频 10 次 / 日频 50 次完整换仓 (启发式; 无因子值时序无法精确测换手)
  - cost_bps_roundtrip: 单次完整换仓成本 (默认 40bp = 15bp 佣金+印花+25bp 冲击滑点)
  - breakeven_cost_bps = gross_excess_floor / annual_turns * 10000
      → 该因子每年换仓一次能承受的成本上限
  - exec_trap = (gross_excess_floor > 0) and (breakeven_cost_bps < cost_bps_roundtrip)
      → 统计超额为正但不足以覆盖真实成本 = 执行陷阱

开关: env FUXI_EXEC_TRAP_GUARD = shadow(默认) | off  (enforce 预留, 当前不拦裁决)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_BASE = Path(__file__).resolve().parent.parent.parent
OUT = _BASE / "data" / "exec_trap_shadow.json"

# 换仓次数启发式 (完整换仓次数 / 年)
ANNUAL_TURNS = {"weekly": 10.0, "daily": 50.0}
DEFAULT_COST_BPS = 40.0  # 单次完整换仓成本 (bp)


def get_mode() -> str:
    m = str(os.environ.get("FUXI_EXEC_TRAP_GUARD", "shadow") or "shadow").strip().lower()
    return m if m in ("off", "shadow", "enforce") else "shadow"


def evaluate(candidates: List[Dict], results: List[Any],
             verbose: bool = True) -> Dict[str, Any]:
    """对 S5 通过 (eligible_for_jq) 候选做执行陷阱影子诊断。"""
    mode = get_mode()
    if mode == "off":
        return {"mode": "off", "n_checked": 0, "n_exec_trap": 0, "rows": []}

    cost_bps = float(os.environ.get("FUXI_EXEC_TRAP_COST_BPS", DEFAULT_COST_BPS) or DEFAULT_COST_BPS)
    rows = []
    for i, cand in enumerate(candidates):
        res = results[i] if i < len(results) else None
        if res is None or not getattr(res, "eligible_for_jq", False):
            continue
        freq = str(getattr(res, "s1_freq", "daily") or "daily")
        e25 = float(getattr(res, "s5_year1_excess", 0) or 0)
        e26 = float(getattr(res, "s5_year2_excess", 0) or 0)
        gross = min(e25, e26)
        turns = ANNUAL_TURNS.get(freq, ANNUAL_TURNS["daily"])
        breakeven_bps = (gross / turns * 10000.0) if turns > 0 else 0.0
        trap = bool(gross > 0 and breakeven_bps <= cost_bps)
        rows.append({
            "factor_name": cand.get("factor_name", cand.get("name", "")),
            "freq": freq,
            "excess_2025": round(e25, 4),
            "excess_2026": round(e26, 4),
            "gross_excess_floor": round(gross, 4),
            "annual_turns_assumed": turns,
            "breakeven_cost_bps": round(breakeven_bps, 1),
            "cost_bps_roundtrip": cost_bps,
            "net_excess_est": round(gross - turns * cost_bps / 10000.0, 4),
            "exec_trap": trap,
        })

    n_trap = sum(1 for r in rows if r["exec_trap"])
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "cost_bps_roundtrip": cost_bps,
        "annual_turns_assumed": ANNUAL_TURNS,
        "n_checked": len(rows),
        "n_exec_trap": n_trap,
        "rows": rows,
    }
    try:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump(payload, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"  [ExecTrap] ⚠️ 影子落盘失败 (非阻塞): {e}")

    if verbose:
        print(f"  [ExecTrap] {mode} 模式: 检查 {len(rows)} S5通过候选 | "
              f"执行陷阱 {n_trap} (成本后净收益转负, 仅标注不裁决)")
        for r in rows[:3]:
            print(f"      - {str(r['factor_name'])[:34]}: 毛利 {r['gross_excess_floor']:+.2%} "
                  f"/ 盈亏平衡成本 {r['breakeven_cost_bps']:.0f}bp "
                  f"{'🔴 执行陷阱' if r['exec_trap'] else '✅'}")
    return payload
