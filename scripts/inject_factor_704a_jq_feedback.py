# -*- coding: utf-8 -*-
"""
factor_704a 真实 JQ 回测反馈注入 (D+ 阶段, formal admission 候选)
将 -85.53% 的真实回测结果注入 Experience Memory + Trajectory Logger, 形成 R→G→E→D→JQ→D+ 闭环。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FA_DIR = ROOT / "research" / "factor_alchemy"
sys.path.insert(0, str(FA_DIR))
sys.path.insert(0, str(ROOT))

from fuxi_v1.governance import governed_writer

jq_result = {
    "batch_id": "factor_704a_formal_admission_jq_20261016",
    "composite_return": -85.53,
    "composite_sharpe": -1.59,
    "composite_maxdd": 86.61,
    "notes": (
        "factor_704a (neg(rank(return(close,2))) = 短期反转, formal admission 准入候选) 真实 JQ 回测 "
        "2021-01-04~2026-09-30: 累计 -85.53% / Sharpe -1.59 / maxDD 86.61%, 持续下行. "
        "5700 条委托失败 (科创板市价单需限价 1428 + 买入数量<100股 3298 + 停牌 950 + 资金不足 7). "
        "本地 formal admission: E1 IC 0.0157 / ICIR 0.136 (极弱), E3 paired +0.00163/日. "
        "真实 JQ 大幅亏损 → 短期反转因子失效, 不满足 B11 对拍, 不进 B12."
    ),
    "factors": [
        {
            "factor_name": "factor_704a",
            "formula": "neg(rank(return(close,2)))",
            "hypothesis": "短期反转: 买近 2 日跌幅最大的股票",
            "paradigm": "price_volume_reversal",
            "category": "量价-反转",
            "operators_used": ["rank", "return", "neg"],
            "local_ic": 0.0157,
            "local_icir": 0.136,
            "jq_return": -85.53,
            "jq_sharpe": -1.59,
            "jq_maxdd": 86.61,
            "jq_composite_contribution": "single_factor",
            "root_cause": (
                "短期反转因子在 2021-2026 A 股真实市场失效; 叠加执行 bug "
                "(科创板市价单 1428 次买不进 + 最小 100 股 3298 次)"
            ),
        }
    ],
}

with governed_writer(ROOT, "feedback", trigger="manual") as identity:
    from ralph_loop import RalphLoop
    ralph = RalphLoop()
    result = ralph.jq_feedback(jq_result)

    print("\n" + "=" * 60)
    print("  D+ Feedback 完成 (factor_704a)")
    print("  轨迹更新: %s" % result.get("trajectories_updated"))
    print("  硬禁止方向: %s" % result.get("hard_forbidden_added"))
    print("  JQ 确认成功: %s" % result.get("jq_success_confirmed"))
    print("=" * 60)
