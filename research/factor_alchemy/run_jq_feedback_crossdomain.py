# -*- coding: utf-8 -*-
"""
JQ 回测反馈脚本: fa_ralph_crossdomain_20260807
将 JQ 回测结果注入 Experience Memory + Trajectory Logger
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ralph_loop import RalphLoop

# ── 构建 JQ 回测结果 ──
jq_result = {
    "batch_id": "fa_ralph_crossdomain_20260807_v2f",
    "composite_return": -27.3,
    "composite_sharpe": -0.35,
    "composite_maxdd": 65.69,
    "notes": "12因子(9量价+3跨域), Top-80等权周频, 2021-01~2026-07. 峰值+70%(2021-01), 谷底-39.77%(2024-02), 此后缓慢震荡回升. vs v2纯量价(+54.27%) 差了81.6pp",
    "factors": [
        # ── F1-F9: 继承 v2 的量价因子 (已验证有正alpha) ──
        {
            "factor_name": "max_drawdown_duration",
            "formula": "max_drawdown_over_60d / running_max_duration",
            "hypothesis": "最大回撤持续天数: 浅短回撤的股票有韧性",
            "paradigm": "price_volume_tail",
            "category": "量价-尾部",
            "operators_used": ["rolling_max", "cummax"],
        },
        {
            "factor_name": "hl_vol_spread_regime",
            "formula": "corr(high_low_spread, volume, 20d)",
            "hypothesis": "高低价差与成交量的相关性捕捉波动-量能同步性",
            "paradigm": "price_volume_volatility",
            "category": "量价-波动",
            "operators_used": ["corr", "rolling_std"],
        },
        {
            "factor_name": "volume_crowding",
            "formula": "volume_pct / volume_ma20 — 成交拥挤度",
            "hypothesis": "极端成交量拥挤预示反转",
            "paradigm": "price_volume_momentum",
            "category": "量价-成交量",
            "operators_used": ["pct_rank", "ma"],
        },
        {
            "factor_name": "event_convexity",
            "formula": "max_ret_5d / avg_ret_20d — 事件冲击凸性",
            "hypothesis": "正向事件冲击凸性强的股票有动量延续",
            "paradigm": "price_volume_event",
            "category": "量价-事件",
            "operators_used": ["rolling_max", "ma"],
        },
        {
            "factor_name": "lottery_inverse",
            "formula": "-(max_ret_5d / volatility_20d) — 逆彩票偏好",
            "hypothesis": "高偏度低波动=规避散户追涨的泡沫股",
            "paradigm": "price_volume_behavioral",
            "category": "量价-行为",
            "operators_used": ["rolling_max", "rolling_std"],
        },
        {
            "factor_name": "earnings_drift",
            "formula": "ret_around_earnings — 财报后漂移",
            "hypothesis": "财报后股价漂移捕捉基本面信息缓慢扩散",
            "paradigm": "price_volume_event",
            "category": "量价-事件",
            "operators_used": ["shift", "rolling_sum"],
        },
        {
            "factor_name": "tail_hedge",
            "formula": "CVaR_95 / VaR_95 — 尾部风险比",
            "hypothesis": "尾部风险比率低的股票左尾保护更好",
            "paradigm": "price_volume_tail",
            "category": "量价-尾部",
            "operators_used": ["percentile", "rolling_std"],
        },
        {
            "factor_name": "coskew_regime",
            "formula": "coskewness_to_market_60d",
            "hypothesis": "与市场的共偏度区分顺周期vs逆周期股票",
            "paradigm": "price_volume_regime",
            "category": "量价-市场状态",
            "operators_used": ["corr", "rolling_std"],
        },
        {
            "factor_name": "small_liquidity",
            "formula": "log(market_cap) / amihud_illiq",
            "hypothesis": "小盘流动性溢价: 小市值+低流动性被低估",
            "paradigm": "price_volume_microstructure",
            "category": "量价-微观结构",
            "operators_used": ["log", "ma"],
        },
        # ── F10-F12: 跨域增强因子 (首次引入基本面/估值/资金流) ──
        {
            "factor_name": "smart_money_flow",
            "formula": "(up_vol - dn_vol) / total_vol — 日内买卖压力",
            "hypothesis": "机构买入时上涨日成交量大于下跌日成交量, 量价背离捕捉聪明钱流向",
            "paradigm": "cross_domain_moneyflow",
            "category": "跨域-资金流",
            "operators_used": ["sum", "boolean_mask"],
        },
        {
            "factor_name": "roe_quality",
            "formula": "ROE (TTM, JQ indicator.roe) — 基本面质量",
            "hypothesis": "高ROE=优质企业, 基本面高质量有长期正alpha",
            "paradigm": "cross_domain_fundamental",
            "category": "跨域-基本面",
            "operators_used": ["get_fundamentals"],
        },
        {
            "factor_name": "value_safety",
            "formula": "1/PE + 1/PB — 价值安全边际",
            "hypothesis": "低PE+低PB=便宜且安全, 价值因子在A股有长期正alpha",
            "paradigm": "cross_domain_valuation",
            "category": "跨域-估值",
            "operators_used": ["get_fundamentals"],
        },
    ],

    # ── 根因分析 ──
    "root_cause": (
        "v2纯量价9因子(+54.27%)加入3个跨域因子后变为-27.3%(-81.6pp差距). "
        "三个可能根因: "
        "(1)跨域因子(F10资金流/F11 ROE/F12估值)在A股2021-2026期间alpha为负或零, "
        "rank-product复合中稀释了9个正alpha因子的信号; "
        "(2)12因子rank-product乘积噪声过大(12个独立因子同时正确的概率低), "
        "信号衰减严重 vs 9因子; "
        "(3)F11/F12基本面数据覆盖率仅~78-84%, "
        "per-factor neutral fill(0.5)使缺失数据的股票在11-12因子上获得中性分, "
        "但基本面数据存在的selection bias(有ROE的=好公司)可能导致反向选择. "
        "峰值+70%→谷底-39.77%的路径是经典的因子衰减曲线: "
        "初始幸运选股后, 跨域因子在后续市场环境中系统性地拖累选股."
    ),
}

# ── 执行 ──
ralph = RalphLoop()
result = ralph.jq_feedback(jq_result)

print(f"\n{'='*60}")
print(f"  D+ Feedback 完成")
print(f"  轨迹更新: {result['trajectories_updated']}")
print(f"  硬禁止方向: {result['hard_forbidden_added']}")
print(f"  JQ确认成功: {result['jq_success_confirmed']}")
print(f"{'='*60}")
