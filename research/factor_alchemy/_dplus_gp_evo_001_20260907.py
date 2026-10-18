# -*- coding: utf-8 -*-
"""一次性 D+ 闭环: gp_evo_001 JQ 回测反馈 (2026-09-07 用户反馈)
JQ: 2021-01-01 ~ 2026-07-30 | 收益 -25.80% | Sharpe -0.454 | MDD 50.16%
日志: E:/log/gp_evo_001.txt (281 周 IC 全样本重算)
"""
import sys
from ralph_loop import RalphLoop
from experience_memory import get_memory

jq_backtest_result = {
    "batch_id": "jq_20260907_gp_evo_001",
    "composite_return": -25.80,   # % (百分比格式)
    "composite_sharpe": -0.454,
    "composite_maxdd": 50.16,     # %
    "factors": [
        {
            "factor_name": "gp_evo_001",
            "formula": "(-(close_p.pct_change() + ((open_p / close_p.shift(1) - 1) - low_p)))",
            "hypothesis": "EvoTraj 变异自 (-hl_ratio): 隔夜跳空与日内低位关系的负向复合",
            "paradigm": "auto_breed",
            "category": "price_volume",
            "operators_used": ["close", "open", "low", "pct_change", "shift", "negation", "subtraction"],
            "local_ic": None,
            "local_icir": -0.2047,     # mem_0179 当前值
            "jq_return": -25.80,
            "jq_sharpe": -0.454,
            "jq_maxdd": 50.16,
            # 日志全样本重算 (281 周): mean=-0.0309 std=0.1645
            "jq_ic": -0.0309,
            "jq_icir": -0.1879,
            "jq_composite_contribution": "negative",
            "root_cause": (
                "方向反转型弱信号: JQ 281 周全样本 IC=-0.031/ICIR=-0.188/+IC%=43.1%, "
                "与本地 ICIR=-0.205 同号同量级(本地代理有效例); 原方向组合 -25.8%/Sharpe -0.454/MDD 50.16%; "
                "即使取负 ICIR 也仅 +0.19 不达 0.3 门槛, 非'方向错了的好因子'而是真弱因子; "
                "分段: 前半(21-23) ICIR=-0.312 → 后半(24-26) -0.083, |ICIR| 衰减 73% 信号消退; "
                "已知坑复现: S5 validate_factor 无方向参数, 负 ICIR 因子原样进 S5 (R3 同坑)"
            ),
            "source": "gp_breed",
            "engine": "gp_breed",
            "natural_freq": "daily",
        }
    ],
}

if __name__ == "__main__":
    memory = get_memory()
    rl = RalphLoop(memory=memory)
    result = rl.jq_feedback(jq_backtest_result=jq_backtest_result, update_lane=True)
    print("\n==== D+ 汇总 ====")
    for k in ("trajectories_updated", "jq_success_confirmed", "hard_forbidden_added"):
        print(f"  {k}: {result.get(k)}")
    print("  memory_formation:", result.get("memory_formation"))
