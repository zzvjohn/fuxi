# -*- coding: utf-8 -*-
"""伏羲 v2.0 factor_c9a6fb22 全市场 standalone JQ 验证 -> D+ 反馈注入 (因子级)

factor_c9a6fb22 = neg(std(close,5))  5日收盘价标准差反转(低波动异象)
伏羲内部口径 (frozen cohort 64只, daily, 1119截面):
  E1 mean rank IC = 0.06691 / ICIR = 0.3511 (E0~E4 全 pass)
全市场 standalone 口径 (0/3/6开头, 剔ST/停牌/涨停, top-30等权, 周频, 滞后IC):
  回测区间 2021-01-01 ~ 2026-09-30
  累计收益 +93.96% / Sharpe 0.414 / maxDD 42.38%
  单因子 Rank IC mean +0.0724 / ICIR +0.4523 / IC>0 66.2% / n=281

判定: JQ_PASSED (ret 93.96>50 且 sharpe 0.414>0.4, 但 sharpe 仅微超 0.4 阈值,
       maxDD 42.38% 偏大, 属低波动因子的风格切换脆弱性)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experience_memory import get_memory
from ralph_loop import RalphLoop

jq_result = {
    "batch_id": "fuxi_v2_factor_c9a6fb22_standalone_20261018",
    "composite_return": 93.96,
    "composite_sharpe": 0.414,
    "composite_maxdd": 42.38,
    "factors": [
        {
            "factor_name": "factor_c9a6fb22",
            "formula": "neg(std(close,5))",
            "hypothesis": (
                "5日收盘价标准差反转(低波动异象): 低波动股票未来收益更高。"
                "伏羲 v2.0 campaign round 11 combine 臂产出, "
                "parent=factor_d9155(变异系数反转 neg(rank(std(close,5)/mean(close,5)))) "
                "x factor_588b9974 subtree_crossover。"
                "claim_type=correlational_hypothesis, mechanism=combined_independent_hypotheses。"
            ),
            "paradigm": "lowvol_reversal",
            "category": "factor_validation",
            "operators_used": ["std", "neg"],
            "local_ic": 0.06691,   # 伏羲 E1 (frozen cohort daily)
            "local_icir": 0.3511,  # 伏羲 E1
            "jq_return": 93.96,
            "jq_sharpe": 0.414,
            "jq_maxdd": 42.38,
            "jq_ic": 0.0724,       # standalone 滞后 Rank IC (P-001 真实 per-factor)
            "jq_icir": 0.4523,     # standalone 滞后 ICIR
            "jq_composite_contribution": (
                "全市场 top-30 等权周频 standalone 验证: 累计 +93.96% / Sharpe 0.414 / maxDD 42.38%"
            ),
            "root_cause": (
                "因子本质=5日低波动反转, 全程 IC 稳定为正: 伏羲 E1 逐年 IC "
                "2021=+0.0365/2022=+0.0914/2023=+0.0703/2024=+0.0757/2025=+0.0495; "
                "standalone 滞后 IC mean +0.0724, IC>0 66.2% (281周), 方向稳健。"
                "风险: maxDD 42.38% 偏大, 低波动因子在风格切换(小盘/微盘波动放大)时回撤深; "
                "2026 下半年单周 IC 波动加大(出现 -0.41/-0.28/-0.25 等单周大负值)。"
                "Sharpe 0.414 仅微超 0.4 阈值, 处于 JQ_PASSED 下沿, 需警惕阈值边缘的稳定性。"
                "口径差异: 本收益为全市场 top-30 周频(非伏羲 frozen cohort 64只 top-6 daily)。"
            ),
        }
    ],
}

if __name__ == "__main__":
    # 伏羲 v2.0 生产状态写入锁: 直接脚本须经 governed_writer 持有写入锁,
    # 否则 trajectory_logger._save / experience_memory._save 被 assert_state_writer 拦截。
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    from fuxi_v1.governance import governed_writer, find_workspace

    workspace = find_workspace(__file__)
    with governed_writer(workspace, "feedback", trigger="manual"):
        memory = get_memory()
        rl = RalphLoop(memory=memory)
        result = rl.jq_feedback(jq_backtest_result=jq_result)
    print("\n=== 伏羲 factor_c9a6fb22 D+ 反馈结果 ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
