# -*- coding: utf-8 -*-
"""A4 指数刹车 → AlphaAgent v3 推广验证 JQ A/B → D+ 反馈注入 (组件级, 非因子级)
True : +198.24% / Sharpe 0.78  / maxDD 27.59%
False: +182.57% / Sharpe 0.689 / maxDD 35.41%
刹车增量: +15.67pp 收益, +0.091 Sharpe, maxDD 改善 -7.82pp
归因: 4波刹车事件3波正向 (2022-04-25 +4.27pp / 2024-02-05微盘崩盘 +2.56pp / 2025-01-06 +0.10pp),
      2026-07-20 期末净负 (-1.93pp, 空仓错过 07-27~30 V反 = 固有代价)。
      增量大头=躲亏的复利放大 (窗口直接Δ+4.29pp, 经2~4年复利放大至期末+15.67pp)。
      maxDD 改善机制: False版maxDD谷=2024-02-07微盘崩盘日(-35.41%), True版同段仅-27.48% → 刹车直接砍掉最大回撤段。
formula 留空 → 不触发 seed_injector 自动入库, 不污染因子库。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experience_memory import get_memory
from ralph_loop import RalphLoop

jq_result = {
    "batch_id": "a4_index_brake_alpha_agent_v3_ab_20260825",
    "composite_return": 198.24,
    "composite_sharpe": 0.78,
    "composite_maxdd": 27.59,
    "factors": [
        {
            "factor_name": "a4_index_brake_component_v3",
            "formula": "",  # 组件非因子, 留空防自动入库
            "hypothesis": "中证1000 15日收益<=-12% 触发清仓空仓一周 (指数刹车, GATE_INDEX A/B), 推广到 AlphaAgent v3 周频复合",
            "paradigm": "allweather_risk_gate",
            "category": "component_validation",
            "operators_used": [],
            "local_ic": None,
            "local_icir": None,
            "jq_return": 15.67,  # 刹车增量 (True-False)
            "jq_sharpe": 0.091,  # Sharpe 增量
            "jq_maxdd": -7.82,  # maxDD 改善 (27.59 vs 35.41)
            "jq_composite_contribution": "baseline +182.57% -> gated +198.24% (+15.67pp), 基底=终版 fa_alpha_agent_v3_jq.py (GATE_INDEX=False 精确复现 +182.57%/0.689/35.41%)",
            "root_cause": (
                "4波刹车事件3波正向: 2022-04-25~05-09(Δ+4.27pp, 躲过4月底急跌)/"
                "2024-02-05~02-19(Δ+2.56pp, 躲过微盘股崩盘, False版maxDD谷2024-02-07恰在窗口内)/"
                "2025-01-06~01-13(Δ+0.10pp, 躲跌与错过反弹抵消)/"
                "2026-07-20~07-30期末(Δ-1.93pp, 空仓错过07-27~30 V反=固有代价, 与打散v1b的-1.07pp同构)。"
                "分年: 2022+4.96/2023+0.10/2024+2.83/2025+0.11/2026-1.62pp。"
                "maxDD 35.41->27.59 (-7.82pp): False版最大回撤段恰为刹车窗口(峰2024-01-04谷2024-02-07), 刹车直接砍段; 打散v1b maxDD无改善因回撤段不在刹车窗口。"
                "增量大头=复利放大: 窗口直接Δ+4.29pp 经2~4年复利放大至期末+15.67pp (交叉项~+10pp)。"
            ),
        }
    ],
}

if __name__ == "__main__":
    memory = get_memory()
    rl = RalphLoop(memory=memory)
    result = rl.jq_feedback(jq_backtest_result=jq_result)
    print("\n=== A4→v3 D+ 反馈结果 ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
