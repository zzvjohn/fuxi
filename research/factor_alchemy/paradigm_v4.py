# -*- coding: utf-8 -*-
"""
Paradigm v4: 扩展金融范式系统 + A股特定算子库 + 多频率支持
================================================================
v3 已有 13 范式

v4 新增 7+ 范式

v4 新增 7+ 范式:
  14. 事件驱动 —
  15. 行业轮动 —
  16. 北向资金 —
  17. 两融信号 —
  18. 大宗交易 —
  19. 高频微观结构 —
  20. 跨资产联动 —
  21. 市场情绪综合 —

新增A股特定算子类:
  - 资金流类: 北向资金流入率/融资买入占比/大宗折价率
  - 事件类: 距财报天数/业绩预告超预期幅度/分红率
  - 市场结构类: 涨停板强度/板块轮动速度/风格因子偏离
  - 高频类: 日内成交量占比/开盘集合竞价量/尾盘异动

多频率支持:
  - 日频因子 (1-5天窗口, 适用于短线策略)
  - 周频因子 (1-10周窗口, 当前主力)
  - 月频因子 (1-6月窗口, 长线价值)
  - 两周频因子 (2-4周窗口, 中短线平衡)
"""

from typing import Dict, List

# ═══════════════════════════════════════════════════════════
# 一、完整范式定义 (v4: 21个)
# ═══════════════════════════════════════════════════════════

PARADIGMS_V4 = {
    # ── v3 保留 (13个) ─────────────────────────────────────
    "流动性×微观结构": {
        "id": 1,
        "description": "A股alpha主源。成交量、换手率、买卖价差、市场深度等流动性指标，"
                       "以及订单流、高频价量等微观结构特征。流动性差的股票存在流动性溢价。",
        "a_share_relevance": "极高 — A股换手率高、流动性分层明显、散户为主的订单流特征。",
        "economic_logic": "流动性差的股票承担更高的交易成本和流动性风险 → 要求补偿 → alpha。",
        "typical_window": [5, 10, 20, 60],
        "keyword_triggers": ["成交量", "换手率", "价差", "买卖盘", "流动性", "市场深度"],
        "coverage_status": "templates_ready",  # v0.6: 5个量价种子模板已注入
    },
    "资金流": {
        "id": 2,
        "description": "资金流入流出、量价关系、主力资金动向。成交量与价格的交互信号。",
        "a_share_relevance": "极高 — A股资金驱动特征明显，资金流领先价格。",
        "economic_logic": "资金持续流入=机构/聪明钱建仓 → 后续上涨概率大。",
        "typical_window": [5, 10, 20],
        "keyword_triggers": ["资金流", "净流入", "主力", "量价", "放量", "缩量"],
        "coverage_status": "templates_ready",  # v0.6: 4个资金流+5个大单种子模板已注入
    },
    "动量反转": {
        "id": 3,
        "description": "短期反转+中期动量+长期反转。A股1-4周反转、3-12月动量特征。",
        "a_share_relevance": "高 — 但A股动量相对美股弱，反转效应更显著。",
        "economic_logic": "行为金融: 反应不足→动量, 过度反应→反转。",
        "typical_window": [5, 10, 20, 60],
        "keyword_triggers": ["动量", "反转", "趋势", "惯性", "超买", "超卖", "回撤"],
        "coverage_status": "templates_ready",  # v0.6: 5个动量反转种子模板已注入
    },
    "行为金融": {
        "id": 4,
        "description": "锚定效应、处置效应、羊群行为、过度自信等行为偏差的系统性定价。",
        "a_share_relevance": "极高 — A股散户占比高，行为偏差更显著。",
        "economic_logic": "投资者系统性行为偏差 → 错误定价 → alpha。",
        "typical_window": [5, 10, 20],
        "keyword_triggers": ["行为", "偏差", "锚定", "过度反应", "羊群", "心理", "情绪"],
        "coverage_status": "templates_ready",  # v0.6: 3+6+8=V型+财务附注种子模板已注入
    },
    "尾部风险": {
        "id": 5,
        "description": "崩盘风险溢价、极端事件定价、偏度/峰度因子。Harvey-Siddique协偏度。",
        "a_share_relevance": "极高 — A股高波动+政策冲击频繁，尾部风险是核心收益来源。",
        "economic_logic": "投资者厌恶崩盘风险 → 对尾部事件过度定价 → 做多被过度惩罚的股票收取premium。",
        "typical_window": [20, 60, 120],
        "keyword_triggers": ["尾部", "崩盘", "偏度", "极端", "crash", "tail", "skew"],
        "coverage_status": "templates_ready",  # v0.6: 2个偏度+3个回撤种子模板已注入
    },
    "情绪×日内": {
        "id": 6,
        "description": "开盘跳空、日内反转、盘前盘后价格行为。隔夜与日内收益率的分离定价。",
        "a_share_relevance": "高 — A股T+1制度+集合竞价机制产生独特的隔夜-日内定价。",
        "economic_logic": "隔夜信息融入 vs 日内交易行为 → 信息不对称溢价。",
        "typical_window": [1, 5, 10],
        "keyword_triggers": ["开盘", "跳空", "日内", "隔夜", "盘前", "集合竞价", "尾盘"],
        "coverage_status": "templates_ready",  # v0.6 P-003: 5个LLM情绪模板已注入
    },
    "截面交互": {
        "id": 7,
        "description": "两个或多个因子维度的截面交互项。如量价交互、动量-波动率交互。",
        "a_share_relevance": "中高 — 单一维度信号噪声大，交互项可提升信噪比。",
        "economic_logic": "不同经济维度的交叉定价 → 非线性alpha。",
        "typical_window": [5, 10, 20],
        "keyword_triggers": ["交互", "乘积", "交叉", "modulation", "非线性"],
        "coverage_status": "templates_ready",  # v0.6: 6个概念/注意力种子模板已注入
    },
    "趋势": {
        "id": 8,
        "description": "价格趋势的持续性和强度。均线系统、通道突破、趋势质量。",
        "a_share_relevance": "中 — A股趋势持续性弱于美股，但结合资金流后有效。",
        "economic_logic": "信息渐进扩散 → 趋势持续性 → alpha。",
        "typical_window": [10, 20, 60],
        "keyword_triggers": ["趋势", "均线", "突破", "通道", "MA", "方向"],
        "coverage_status": "templates_ready",  # v0.6: 4个趋势/突破种子模板已注入
    },
    "下行保护": {
        "id": 9,
        "description": "回撤控制、尾部对冲、下行风险定价。捕捉下跌后的修复机会。",
        "a_share_relevance": "极高 — A股高波动+急跌特征，下行保护是降回撤的核心。",
        "economic_logic": "过度恐慌→超跌→均值回归修复。",
        "typical_window": [5, 20, 60],
        "keyword_triggers": ["下行", "回撤", "跌幅", "保护", "对冲", "急跌", "修复"],
        "coverage_status": "templates_ready",  # v0.6: 尾部增强因子覆盖下行保护 (via 尾部风险)
    },
    "波动率适应": {
        "id": 10,
        "description": "波动率周期、波动率聚集(GARCH)、波动率状态切换。",
        "a_share_relevance": "中高 — 市场波动率水平是策略表现的关键调节变量。",
        "economic_logic": "低波动=趋势友好, 高波动=反转策略更有效 → 状态依赖alpha。",
        "typical_window": [5, 10, 20, 40, 60],
        "keyword_triggers": ["波动率", "volatility", "GARCH", "聚集", "低波", "高波"],
        "coverage_status": "templates_ready",  # v0.6: 4个波动率种子模板已注入
    },
    "市场宽度": {
        "id": 11,
        "description": "截面离散度、相对强度扩散、市场参与度。个股相对市场整体的偏离。",
        "a_share_relevance": "中 — A股同涨同跌特征强，但结构性行情中宽度信号有效。",
        "economic_logic": "宽度收缩=趋势行情, 宽度扩张=震荡行情 → 状态识别alpha。",
        "typical_window": [5, 10, 20],
        "keyword_triggers": ["宽度", "离散度", "扩散", "参与度", "涨跌比", "新高"],
        "coverage_status": "uncovered",  # v0.6: 暂无种子模板
    },
    "结构突变": {
        "id": 12,
        "description": "价格/成交量的结构性断点。突破/跌破关键位、量能突变。",
        "a_share_relevance": "中高 — A股技术分析文化浓厚，关键位突破有自实现效应。",
        "economic_logic": "突破信号 → 技术派跟风 → 自实现预言。",
        "typical_window": [5, 10, 20, 60],
        "keyword_triggers": ["突破", "突变", "结构", "支撑", "阻力", "关键位"],
        "coverage_status": "uncovered",  # v0.6: 暂无种子模板
    },
    "筹码分布": {
        "id": 13,
        "description": "持仓成本分布、筹码集中度、浮筹比例。A股特有的筹码理论。",
        "a_share_relevance": "极高 — A股核心降回撤维度。筹码锁定=底部, 筹码分散=顶部。",
        "economic_logic": "筹码集中=主力控盘=抗跌+易拉升 → alpha。",
        "typical_window": [10, 20, 60],
        "keyword_triggers": ["筹码", "持仓成本", "集中", "分散", "浮筹", "锁定", "惜售"],
        "coverage_status": "templates_ready",  # v0.6: 3+6=9个筹码分布+分层种子模板已注入
    },

    # ── v4 新增 (8个) ─────────────────────────────────────
    "事件驱动": {
        "id": 14,
        "description": "财报公告、业绩预告、分红除权、ST摘帽、回购增持、大股东增减持、"
                       "股权激励、重大合同公告等事件效应的系统化因子化。",
        "a_share_relevance": "高 — A股公告效应显著(预增公告跳涨、ST摘帽炒作、高送转)。",
        "economic_logic": "信息事件 → 投资者注意力分配不对称 → 公告后漂移(PEAD)/反转 → alpha。",
        "typical_window": [1, 5, 20, 60],
        "keyword_triggers": ["公告", "财报", "业绩", "分红", "除权", "回购", "增持", "减持",
                           "ST", "摘帽", "股权激励", "预告", "超预期", "PEAD"],
        "coverage_status": "templates_ready",  # v0.6: 8个跳空趋势背离种子模板已注入 (P-012)
    },
    "行业轮动": {
        "id": 15,
        "description": "申万一级行业的相对强度、行业动量、行业拥挤度、"
                       "行业间资金流向。捕捉行业层面的alpha。",
        "a_share_relevance": "高 — A股行业轮动明显(2019-2021消费→2023-2024科技/ChatGPT→高股息)。",
        "economic_logic": "经济周期+政策+产业趋势 → 行业轮动 → 行业层面alpha。",
        "typical_window": [5, 20, 60, 120],
        "keyword_triggers": ["行业", "板块", "轮动", "申万", "SW", "相对强度", "拥挤度"],
        "coverage_status": "templates_ready",  # v0.6: 6个行业轮动种子模板已注入 (P-016)
    },
    "北向资金": {
        "id": 16,
        "description": "沪深港通北向资金净流入/流出、持仓市值变化、"
                       "增持/减持比例。聪明钱跟踪信号。",
        "a_share_relevance": "极高 — 北向资金对A股定价权日益增强，持仓变化有显著预测力。",
        "economic_logic": "外资=信息优势投资者 → 持仓方向反映基本面判断 → alpha。",
        "typical_window": [1, 5, 20, 60],
        "keyword_triggers": ["北向", "沪深港通", "港资", "外资", "smart money",
                           "connect", "净流入", "QFII", "陆股通"],
        "coverage_status": "templates_ready",  # v0.5 P-002: 8个北向因子模板已注入
    },
    "两融信号": {
        "id": 17,
        "description": "融资余额变化、融资买入占比、融券余额。杠杆资金行为信号。",
        "a_share_relevance": "高 — A股融资余额波动大，融资买入占比与短期走势高度相关。",
        "economic_logic": "融资买入=看多情绪+杠杆 → 过度乐观→短期反转/过度悲观→反弹。",
        "typical_window": [1, 5, 20],
        "keyword_triggers": ["融资", "融券", "两融", "杠杆", "margin", "保证金", "余额"],
        "coverage_status": "templates_ready",  # v0.6: 6个两融信号种子模板已注入 (P-019, moneyflow代理)
    },
    "大宗交易": {
        "id": 18,
        "description": "大宗交易的折溢价率、成交量占比、交易对手特征。"
                       "大额交易的信号提取。",
        "a_share_relevance": "中 — A股大宗交易折价率高(5-10%常见)，折溢价含信息。",
        "economic_logic": "折价大宗=减持压力=短期利空→但深度折价后反弹。溢价大宗=积极信号。",
        "typical_window": [1, 5, 20],
        "keyword_triggers": ["大宗", "block trade", "折价", "溢价", "大宗交易", "减持"],
        "coverage_status": "uncovered",  # v0.6: 暂无种子模板
    },
    "高频微观结构": {
        "id": 19,
        "description": "分钟级别价量、订单流不平衡、买卖价差、订单簿深度。"
                       "更高频率的市场微观结构信号。",
        "a_share_relevance": "中 — 数据获取难度大(需要Level-2/Tick数据)，但信号质量高。",
        "economic_logic": "微观结构 → 信息不对称和流动性需求的实时定价 → alpha。",
        "typical_window": [1, 5],
        "keyword_triggers": ["高频", "tick", "分钟", "订单流", "买卖盘", "order book",
                           "价差", "level-2", "LOB"],
        "coverage_status": "templates_ready",  # v0.6 P-010: 8个高频微观结构模板已注入 (OHLCV代理)
    },
    "跨资产联动": {
        "id": 20,
        "description": "股债相关性、商品-股票传导、汇率敏感度。跨市场的alpha传导机制。",
        "a_share_relevance": "中 — 人民币汇率→北向资金→A股, 油价→中石油/化工, 美债→成长股。",
        "economic_logic": "跨市场信息传导 → 滞后定价 → alpha。",
        "typical_window": [5, 20, 60],
        "keyword_triggers": ["跨资产", "联动", "汇率", "商品", "债券", "相关性",
                           "传导", "global macro", "spillover"],
        "coverage_status": "uncovered",  # v0.6: 暂无种子模板 (P-023 待落地)
    },
    "市场情绪综合": {
        "id": 21,
        "description": "涨停板强度、跌停板数量、涨跌比、创新高/新低比例、"
                       "VIX风格波动率指数。市场整体的情绪温度计。",
        "a_share_relevance": "高 — A股情绪驱动特征明显(涨停板文化、恐贪指数)。",
        "economic_logic": "市场情绪极端化 → 反转 → alpha。巴菲特: 别人恐惧时贪婪。",
        "typical_window": [1, 5, 10, 20],
        "keyword_triggers": ["涨停", "跌停", "涨跌比", "情绪", "恐贪", "创新高", "新低",
                           "limit up", "limit down", "VIX"],
        "coverage_status": "uncovered",  # v0.6: 暂无种子模板
    },
}

# ═══════════════════════════════════════════════════════════
# 二、A股特定算子库 (v4 扩展)
# ═══════════════════════════════════════════════════════════

# 标准 Forge 算子 (从 alpha_agent.py 继承)
STANDARD_OPERATORS = {
    "时间序列": ["ts_mean(x, d)", "ts_std(x, d)", "ts_sum(x, d)",
                  "ts_min(x, d)", "ts_max(x, d)", "ts_delta(x, d)",
                  "ts_pct(x, d)", "ts_zscore(x, d)", "ts_skew(x, d)",
                  "ts_kurtosis(x, d)", "ts_rank(x, d)", "ts_argmax(x, d)",
                  "ts_argmin(x, d)", "ts_corr(x, y, d)"],
    "截面": ["rank(x)", "scale(x)", "cs_rank(x)", "cs_zscore(x)"],
    "算术": ["add(x, y)", "sub(x, y)", "mul(x, y)", "div(x, y)",
              "neg(x)", "abs(x)", "log(x)", "sqrt(x)", "pow(x, n)", "sigmoid(x)"],
    "逻辑": ["if_else(cond, t, f)", "greater(x, y)", "less(x, y)"],
    "价格": ["returns", "close", "open", "high", "low", "volume", "amount", "vwap"],
}

# A股特定算子 — 需要在JQ策略中实现真实计算
A_SHARE_SPECIFIC_OPERATORS = {
    "北向资金": {
        "description": "沪深港通北向资金相关算子。注意: JQ聚宽有 north_finance 等API可获取。",
        "operators": [
            ("north_flow_daily(x, d)", "北向资金d日净流入(亿元)"),
            ("north_flow_cum(x, d)", "北向资金d日累计净流入"),
            ("north_holding_pct(x, d)", "北向资金持仓占流通市值比例"),
            ("north_holding_change(x, d)", "北向资金d日持仓变化率"),
        ],
    },
    "融资融券": {
        "description": "融资融券余额及变化。JQ有 margin 相关API。",
        "operators": [
            ("margin_balance(_, d)", "融资余额d日变化"),
            ("margin_buy_ratio(_, d)", "融资买入占总成交比例"),
            ("short_balance(_, d)", "融券余额"),
        ],
    },
    "事件驱动": {
        "description": "事件日期和公告数据。JQ有 get_fundamentals 和财务日期API。",
        "operators": [
            ("days_since_report(x, d)", "距最近财报公告天数"),
            ("earnings_surprise(x, d)", "实际EPS相对分析师预期的超预期幅度"),
            ("dividend_yield(x, d)", "近12月股息率"),
            ("buyback_intensity(x, d)", "d日内回购金额/市值"),
            ("insider_net_buy(x, d)", "d日内大股东净增持/总市值"),
        ],
    },
    "行业轮动": {
        "description": "行业相对强度指标。JQ有 get_industry 和 申万行业分类。",
        "operators": [
            ("industry_rel_strength(x, d)", "个股所属行业d日相对强度(行业指数/大盘指数)"),
            ("industry_dispersion(_, d)", "d日行业内截面离散度"),
            ("sector_rotation_signal(x, d)", "板块轮动信号(动量+拥挤度复合)"),
        ],
    },
    "大宗交易": {
        "description": "大宗交易数据。JQ/数据源可获取。",
        "operators": [
            ("block_trade_discount(x, d)", "d日内大宗交易加权平均折价率"),
            ("block_trade_volume_ratio(x, d)", "d日内大宗交易量/总成交量"),
        ],
    },
    "市场情绪": {
        "description": "市场整体情绪指标。",
        "operators": [
            ("limit_up_count(_, d)", "d日全市场涨停个股数/总个股数"),
            ("limit_down_count(_, d)", "d日全市场跌停个股数"),
            ("new_high_ratio(_, d)", "d日创d日新高个股比例"),
            ("advance_decline_ratio(_, d)", "d日涨跌比"),
        ],
    },
    "筹码分布": {
        "description": "A股筹码理论相关算子(增强版)。",
        "operators": [
            ("turnover_concentration(x, d)", "d日内换手率分布的集中度(低换手天数占比)"),
            ("volume_climax(x, d)", "量能高潮检测(当日量>前d日均量2倍)"),
            ("price_volume_divergence(x, d)", "价格与成交量的d日背离度"),
        ],
    },
    "跨资产": {
        "description": "跨市场联动算子。",
        "operators": [
            ("corr_equity_bond(x, y, d)", "个股与国债期货d日相关性"),
            ("fx_sensitivity(x, d)", "个股对人民币汇率d日敏感度(beta_CNY)"),
            ("commodity_beta(x, d)", "个股对商品指数d日暴露"),
        ],
    },
}

# ═══════════════════════════════════════════════════════════
# 三、多频率因子时间窗口建议
# ═══════════════════════════════════════════════════════════

FREQUENCY_CONFIGS = {
    "daily": {
        "name": "日频",
        "suitable_for": "短线交易/高频因子",
        "lookback_windows": [1, 3, 5, 10, 20],
        "warmup_days": 60,
        "note": "⚠️ 已证实20-60日lookback因子不适合日频(信号=噪声)。"
                "仅1-5日窗口的高频微观结构因子可能有效。",
        "status": "仅限实验，非主力",
    },
    "biweekly": {
        "name": "两周频",
        "suitable_for": "中短线均衡",
        "lookback_windows": [5, 10, 15, 20, 30],
        "warmup_weeks": 4,
        "note": "两周频介于日频噪声和月频钝化之间，可能是未被探索的甜区。",
        "status": "建议新增探索",
    },
    "weekly": {
        "name": "周频",
        "suitable_for": "主力频率",
        "lookback_windows": [5, 10, 20, 60, 120],
        "warmup_weeks": 5,
        "note": "王者频率。5.5年回测窗口(268周)。20-60日lookback因子适配最好。",
        "status": "🏆 主力",
    },
    "monthly": {
        "name": "月频",
        "suitable_for": "长线价值/基本面",
        "lookback_windows": [1, 3, 6, 12, 24],
        "warmup_months": 3,
        "note": "更适合基本面因子(估值/质量/成长)。短期技术因子钝化。",
        "status": "辅助频率",
    },
}

# ═══════════════════════════════════════════════════════════
# 四、LLM Prompt 生成工具
# ═══════════════════════════════════════════════════════════

def get_all_paradigms_for_llm() -> str:
    """生成所有范式的LLM prompt描述文本"""
    lines = []
    for name, info in PARADIGMS_V4.items():
        lines.append(f"### {name} (ID={info['id']})")
        lines.append(f"  {info['description']}")
        lines.append(f"  A股相关: {info['a_share_relevance']}")
        lines.append(f"  经济逻辑: {info['economic_logic']}")
        lines.append(f"  典型窗口: {info['typical_window']}")
    return "\n".join(lines)


def get_operators_for_llm() -> str:
    """生成算子的LLM prompt描述文本"""
    lines = ["## 标准Forge算子"]
    for cat, ops in STANDARD_OPERATORS.items():
        lines.append(f"### {cat}: {', '.join(ops)}")

    lines.append("")
    lines.append("## A股特定算子 (JQ中需实现真实计算)")
    for cat, info in A_SHARE_SPECIFIC_OPERATORS.items():
        lines.append(f"### {cat}: {info['description']}")
        for op_name, op_desc in info["operators"]:
            lines.append(f"  - {op_name}: {op_desc}")

    return "\n".join(lines)


def get_paradigm_categories() -> dict:
    """返回范式名称列表(用于category映射)"""
    return {name: info["keyword_triggers"] for name, info in PARADIGMS_V4.items()}


def get_frequency_windows(freq: str = "weekly") -> list:
    """获取指定频率的推荐lookback窗口"""
    freq_config = FREQUENCY_CONFIGS.get(freq, FREQUENCY_CONFIGS["weekly"])
    return freq_config["lookback_windows"]


# ═══════════════════════════════════════════════════════════
# 五、Library-level 正交性管理 (FactorMiner 风格)
# ═══════════════════════════════════════════════════════════

# Correlation Red Sea 阈值
CORRELATION_RED_SEA_THRESHOLD = 0.70  # 最大允许的 pairwise corr
JACCARD_DIMENSION_THRESHOLD = 0.50    # 维度Jaccard上限

# 范式禁止区域 — 每个范式的"撞墙"方向
# 格式: [范式] → [与哪个范式高度冗余, 避免同时出现的因子结构]
FORBIDDEN_REGIONS = {
    "动量反转": ["趋势"],  # 动量反转与趋势维度高度重叠
    "波动率适应": ["波动率"],  # 避免纯波动率因子的变体
    "尾部风险": ["下行保护"],  # 注意尾部vs下行保护的区别(前者=anticipating,后者=repairing)
    "资金流": ["流动性×微观结构"],  # 量价类易重叠
    "事件驱动": ["情绪×日内"],  # 公告日效应 vs 日内效应区分
}

# 成功因子模板库 — 已验证有效的范式配对
SUCCESSFUL_PARADIGM_PAIRS = [
    ("流动性×微观结构", "趋势"),       # overngiht × tvma (王者comp1)
    ("流动性", "资金流"),              # dollar_vol × turnover_std (王者comp2)
    ("资金流", "趋势"),                # money_flow × ret_3m (王者comp3)
    ("趋势", "筹码分布"),              # tvma × LLM因子 (王者comp4, 筹码维度)
    ("微观结构", "资金流"),            # LLM因子 × dollar_vol (王者comp5)
    ("尾部风险", "流动性"),            # 尾部增强因子验证有效
    ("下行保护", "波动率适应"),        # 降回撤配对
]

def get_paradigm_coverage_map() -> Dict[str, str]:
    """返回所有范式的 coverage_status 映射 — 供 MAB 和 Memory Bridge 查询"""
    return {
        name: info.get("coverage_status", "uncovered")
        for name, info in PARADIGMS_V4.items()
    }


def get_templates_ready_paradigms() -> List[str]:
    """返回已有种子模板注入的范式名称列表"""
    return [name for name, info in PARADIGMS_V4.items()
            if info.get("coverage_status") == "templates_ready"]


# ── test ──────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Paradigm v4: {len(PARADIGMS_V4)} 个范式")
    print(f"A股特定算子: {len(A_SHARE_SPECIFIC_OPERATORS)} 类")
    print(f"频率支持: {list(FREQUENCY_CONFIGS.keys())}")
    print()
    for name in PARADIGMS_V4:
        info = PARADIGMS_V4[name]
        tag = " [NEW]" if info["id"] >= 14 else ""
        print(f"  {info['id']:2d}. {name}{tag}")
