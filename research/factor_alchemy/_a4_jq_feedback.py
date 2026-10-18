# -*- coding: utf-8 -*-
"""A4 指数刹车 JQ A/B 验证 → D+ 反馈注入 (组件级, 非因子级)
True : +105.14% / Sharpe 0.494 / maxDD 31.8%
False: +95.22%  / Sharpe 0.420 / maxDD 32.1%
刹车增量: +9.92pp 收益, +0.074 Sharpe, -0.3pp maxDD
归因: 7波触发事件中 2022-04 / 2024-01~02 / 2025-01 为正向主要贡献;
      2026-07 净负 (错过 7-27~30 反弹 +4.5%)。
formula 留空 → 不触发 seed_injector 自动入库, 不污染因子库。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experience_memory import get_memory
from ralph_loop import RalphLoop

jq_result = {
    "batch_id": "a4_index_brake_ab_20260824",
    "composite_return": 105.14,
    "composite_sharpe": 0.494,
    "composite_maxdd": 31.8,
    "factors": [
        {
            "factor_name": "a4_index_brake_component",
            "formula": "",  # 组件非因子, 留空防自动入库
            "hypothesis": "中证1000 15日收益<=-12% 触发清仓空仓一周 (指数刹车, GATE_INDEX A/B)",
            "paradigm": "allweather_risk_gate",
            "category": "component_validation",
            "operators_used": [],
            "local_ic": None,
            "local_icir": None,
            "jq_return": 9.92,  # 刹车增量 (True-False)
            "jq_sharpe": 0.074,  # Sharpe 增量
            "jq_maxdd": -0.3,  # maxDD 改善
            "jq_composite_contribution": "baseline +95.22% -> gated +105.14% (+9.92pp)",
            "root_cause": (
                "7波触发事件: 2022-04(Δ+2.5~4.6pp/日,躲过1000指数4月崩盘)/"
                "2024-01~02(Δ+8.57pp@01-31,躲过微盘股流动性危机)/2025-01(Δ+2.48pp) 正向;"
                "2026-07 净负(-1.07pp年化): 07-13触发空仓一周躲过下跌但错过07-27~30反弹+4.5%。"
                "maxDD 31.8 vs 32.1 几乎无改善: 刹车躲瞬时崩盘, 非最大回撤段。"
                "触发时点与本地预验证偏差1-3交易日(15日收益口径差异), 方向一致。"
            ),
        }
    ],
}

if __name__ == "__main__":
    memory = get_memory()
    rl = RalphLoop(memory=memory)
    result = rl.jq_feedback(jq_backtest_result=jq_result)
    print("\n=== A4 D+ 反馈结果 ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
