# -*- coding: utf-8 -*-
"""
手动 D+ 蒸馏: lhb_freq_20d JQ 回测结果注入 (无轨迹记录的新因子, trigger_d_plus 扫不到)
用法: cd research/factor_alchemy && python manual_jq_feedback_lhb.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
DATA_DIR = Path(__file__).parent.parent.parent / "data"

JQ_RESULT = {
    "batch_id": "jq_lhb_freq_20d_20260815",
    "timestamp": datetime.now().isoformat(),
    "composite_return": 67.0,
    "composite_sharpe": 0.2662,
    "composite_maxdd": -35.25,
    "factors": [
        {
            "factor_name": "lhb_freq_20d",
            "formula": "-lhb_flag.rolling(20).sum()",
            "hypothesis": (
                "龙虎榜上榜频率反转: 近20日上榜次数取负。游资炒作聚集的股票未来5日收益"
                "显著更差=A股炒作后反转。本地ICIR=-0.495 (2023-2026, 856天, 400股, "
                "10/20/60日窗口交叉验证全显著)。"
            ),
            "paradigm": "资金流",
            "category": "money_flow",
            "operators_used": ["rolling", "sum", "negation"],
            "local_ic": 0.0,
            "local_icir": -0.495,
            "jq_return": 67.0,
            "jq_sharpe": 0.2662,
            "jq_maxdd": -35.25,
            "jq_composite_contribution": "positive",
            "root_cause": (
                "2021-01-01~2026-07-30 窗口累计+67%, 方向(游资活跃度反转)获JQ确认, "
                "但Sharpe 0.27未达PASS阈值(0.4)→MARGINAL; 2018-2021窗口未回测; "
                "TOP50周频+高频换手下成本侵蚀明显"
            ),
            "jq_window": "2021-01-01~2026-07-30",
        }
    ],
}


def main():
    # ── L1 解读层 (先于 distill) ──
    try:
        from interpretation_agent import interpret_batch
        from experience_memory import get_memory as _get_mem_l1
        mem_l1 = _get_mem_l1()
        print("[L1] 解读层启动 (LLM 归因, 失败自动降级规则)...")
        interpret_batch(JQ_RESULT, memory=mem_l1, use_llm=True)
        for f in JQ_RESULT["factors"]:
            itp = f.get("interpretation") or {}
            print("  [L1] %s -> verdict=%s confidence=%s"
                  % (f["factor_name"], itp.get("verdict", "?"), itp.get("confidence", "?")))
    except Exception as e:
        print("[L1] 解读层失败 (不阻塞 D+): %r" % e)

    # ── D+ 闭环 ──
    from experience_memory import get_memory
    from ralph_loop import RalphLoop
    memory = get_memory()
    rl = RalphLoop(memory=memory)
    result = rl.jq_feedback(jq_backtest_result=JQ_RESULT)
    print("\n  D+ 结果:", json.dumps(result, ensure_ascii=False, indent=2))

    # ── L1 写回 memory ──
    try:
        from interpretation_agent import attach_to_memory
        n = attach_to_memory(memory, JQ_RESULT["factors"])
        print("  [L1] attach_to_memory: %d 条" % n)
    except Exception as e:
        print("  [L1] attach 失败: %r" % e)

    # ── 更新因子队列 ──
    qpath = DATA_DIR / "stage1_factor_proposals.json"
    q = json.load(open(qpath, encoding="utf-8"))
    items = q if isinstance(q, list) else q.get("proposals", q.get("factors", []))
    for it in items:
        if str(it.get("factor_name") or "") == "lhb_freq_20d":
            it["jq_status"] = "jq_tested"
            it["jq_result"] = {
                "window": "2021-01-01~2026-07-30",
                "return_pct": 67.0,
                "sharpe": 0.2662,
                "maxdd_pct": -35.25,
                "outcome": "JQ_MARGINAL",
                "note": "收益正但Sharpe<0.4未达PASS; 建议补跑2018-01-01全窗口对比",
                "tested_at": datetime.now().isoformat(),
            }
    json.dump(q, open(qpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("  [Queue] lhb_freq_20d 已标记 jq_tested / JQ_MARGINAL")


if __name__ == "__main__":
    main()
