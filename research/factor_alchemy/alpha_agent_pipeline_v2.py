# -*- coding: utf-8 -*-
"""
AlphaAgent Pipeline v2: 批量80+ → 三重约束 → EFS 3代进化 → V3集成 → JQ
==========================================================================
vs v1变化:
  1. 因子生成: 16 → 80+候选 (10+金融范式, 每范式6-12个变体)
  2. EFS进化: Gen1→Gen2→Gen3, 每代幸存者模式蒸馏→下一轮模板生成
  3. 更精准的JQ因子实现 (不再占位)

架构:
  Gen1: 80+候选 → 三重约束 → ICIR评估 → Top-15幸存者
  Gen2: 知识蒸馏(模式提取) → 40+变异候选 → 三重约束 → ICIR → Top-10
  Gen3: 第二次蒸馏 → 30+候选 → 三重约束 → ICIR → Top-5
  最终: V3集成(3个V3原复合 + 3-4个Gen3最佳复合) → JQ
"""

import sys, os, json, time, re, gc, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from alpha_agent import (
    AlphaAgentConfig, AlphaAgent, AlphaCandidate,
    init_v3_pool, V3_FACTOR_POOL,
    check_originality, check_complexity, ast_similarity,
    parse_expression, ExprNode,
)

# ═══════════════════════════════════════════════════════════
# 阶段0: 大规模因子模板 (80+ 候选, 10+范式)
# ═══════════════════════════════════════════════════════════

def generate_gen1_factors() -> List[dict]:
    """Gen1: 80+ 因子候选, 覆盖所有关键金融范式"""

    factors = []

    # ── 范式1: 流动性×微观结构 (A股alpha主源, 12个) ──
    f = factors.extend
    f([
        {"expression": "neg(rank(div(sub(high, low), add(ts_mean(volume, 20), 1e-6))))",
         "name": "振幅归一化(量)_v1", "direction": "+",
         "rationale": "日内振幅/成交量均值排名取负。高振幅低量=流动性差=流动性溢价。截面排名消除量纲。",
         "paradigm": "流动性x微观结构", "gen": 1},
        {"expression": "neg(div(sub(high, low), add(ts_mean(volume, 10), 1e-6)))",
         "name": "振幅归一化(量)_v2_10d", "direction": "+",
         "rationale": "10日版振幅/成交量。更短窗口捕捉近期流动性变化。",
         "paradigm": "流动性x微观结构", "gen": 1},
        {"expression": "neg(rank(div(ts_std(close, 5), add(ts_mean(volume, 20), 1e-6))))",
         "name": "价格波动/量(截面)", "direction": "+",
         "rationale": "5日价格波动/成交量均值的截面排名取负。波动大但量小=虚假波动, 真正的低流动性信号。V3无同类。",
         "paradigm": "流动性x微观结构", "gen": 1},
        {"expression": "neg(rank(div(div(ts_std(close, 20), ts_mean(close, 20)), add(ts_mean(volume, 20), 1e-6))))",
         "name": "波动率/价格比率(截面)", "direction": "+",
         "rationale": "变异系数/量均值。复合流动性质量指标。高CV低量=最差流动性=最大溢价。",
         "paradigm": "流动性x微观结构", "gen": 1},
        {"expression": "rank(div(sub(close, ts_min(close, 20)), add(sub(ts_max(close, 20), ts_min(close, 20)), 1e-6)))",
         "name": "价格区间位置(Stochastic)", "direction": "-",
         "rationale": "价格在20日区间中的位置。高位=超买, 低位=超卖。经典Stochastic。",
         "paradigm": "流动性x微观结构", "gen": 1},
        {"expression": "neg(ts_zscore(div(sub(high, low), add(ts_mean(sub(high, low), 20), 1e-6)), 40))",
         "name": "振幅异常度(长期zscore)", "direction": "-",
         "rationale": "日内振幅相对20日均振幅的zscore。异常放大=分歧加剧=方向即将确定。",
         "paradigm": "流动性x微观结构", "gen": 1},
        {"expression": "neg(div(ts_std(sub(high, low), 10), add(ts_mean(close, 10), 1e-6)))",
         "name": "振幅稳定性", "direction": "+",
         "rationale": "10日振幅标准差/10日均价。低振幅稳定性=筹码锁定, 上涨前常见特征。",
         "paradigm": "流动性x微观结构", "gen": 1},
        {"expression": "rank(mul(neg(div(close, ts_mean(close, 20))), neg(div(ts_mean(volume, 5), ts_mean(volume, 20)))))",
         "name": "低位缩量(截面)", "direction": "+",
         "rationale": "截面rank(价格低于20日均线 × 量低于20日均量)。底部双重确认——跌到位+抛压衰竭。",
         "paradigm": "流动性x微观结构", "gen": 1},
        {"expression": "neg(rank(div(sub(high, low), sub(ts_max(high, 20), ts_min(low, 20)))))",
         "name": "相对振幅(范围归一化)", "direction": "+",
         "rationale": "日内振幅/20日价格范围。相对波动率——高相对振幅=关注度高=潜在alpha信号。",
         "paradigm": "流动性x微观结构", "gen": 1},
        {"expression": "neg(rank(div(ts_mean(sub(high, low), 5), add(ts_mean(close, 5), 1e-6))))",
         "name": "5日振幅/价格比(截面)", "direction": "+",
         "rationale": "5日均振幅/5日均价, 取负排名。短期振幅压缩后突破是经典技术信号。",
         "paradigm": "流动性x微观结构", "gen": 1},
        {"expression": "neg(rank(div(close, ts_max(high, 20))))",
         "name": "价格/20日高点比", "direction": "+",
         "rationale": "收盘价/20日最高价排名取负。越低越可能反弹, 阻力位+锚定效应。",
         "paradigm": "流动性x微观结构", "gen": 1},
        {"expression": "rank(div(ts_mean(volume, 3), add(ts_mean(volume, 60), 1e-6)))",
         "name": "短期/长期量比", "direction": "-",
         "rationale": "3日/60日均量排名。短期放量=过度关注=短期顶部风险。拥挤度代理。",
         "paradigm": "流动性x微观结构", "gen": 1},
    ])

    # ── 范式2: 资金流×量价背离 (8个) ──
    f([
        {"expression": "sub(ts_pct(close, 10), ts_pct(volume, 10))",
         "name": "量价背离(10日)", "direction": "-",
         "rationale": "价格增速-量增速。价涨量缩=上涨乏力(顶), 价跌量缩=抛压衰竭(底)。",
         "paradigm": "资金流x量价背离", "gen": 1},
        {"expression": "sub(ts_pct(close, 5), ts_pct(volume, 5))",
         "name": "量价背离(5日)", "direction": "-",
         "rationale": "5日版量价背离。更敏感, 捕捉短期资金异动。",
         "paradigm": "资金流x量价背离", "gen": 1},
        {"expression": "neg(sub(ts_pct(close, 20), ts_pct(volume, 20)))",
         "name": "量价背离(20日)_取负", "direction": "+",
         "rationale": "20日量价背离取负。中周期背离反转——背离终将收敛。",
         "paradigm": "资金流x量价背离", "gen": 1},
        {"expression": "neg(div(ts_pct(close, 10), add(ts_pct(volume, 10), 1e-6)))",
         "name": "量价弹性", "direction": "+",
         "rationale": "价格变化率/量变化率, 取负。量弹性大=少量资金推动大价格=筹码集中度高=上涨潜力。",
         "paradigm": "资金流x量价背离", "gen": 1},
        {"expression": "rank(mul(ts_zscore(close, 20), neg(ts_zscore(volume, 20))))",
         "name": "放量下跌(截面)", "direction": "+",
         "rationale": "rank(价格zscore × 负量zscore)。放量下跌=恐慌=超跌反弹。V3无直接量价交互。",
         "paradigm": "资金流x量价背离", "gen": 1},
        {"expression": "rank(mul(neg(ts_zscore(close, 20)), neg(ts_zscore(volume, 20))))",
         "name": "缩量下跌(截面)", "direction": "+",
         "rationale": "rank(负价格zscore × 负量zscore)。缩量下跌=抛压衰竭=更健康底部。不同于放量下跌。",
         "paradigm": "资金流x量价背离", "gen": 1},
        {"expression": "rank(mul(ts_zscore(close, 10), ts_zscore(volume, 10)))",
         "name": "放量上涨(截面)", "direction": "+",
         "rationale": "rank(价格zscore × 量zscore)。放量上涨=量价配合=趋势确认, 10日窗口捕捉中短期趋势。",
         "paradigm": "资金流x量价背离", "gen": 1},
        {"expression": "neg(rank(div(ts_mean(volume, 5), add(ts_mean(volume, 20), 1e-6))))",
         "name": "缩量强度(取负)", "direction": "+",
         "rationale": "5日/20日均量排名取负。缩量=筹码锁定=观望后突破。A股缩量后上涨概率高于放量。",
         "paradigm": "资金流x量价背离", "gen": 1},
    ])

    # ── 范式3: 波动率×尾部风险 (8个) ──
    f([
        {"expression": "sub(neg(ts_std(returns, 20)), ts_std(returns, 5))",
         "name": "波动率扩张", "direction": "+",
         "rationale": "短期波动-长期波动(均取负)。波动率扩张=趋势启动信号。",
         "paradigm": "波动率x尾部风险", "gen": 1},
        {"expression": "neg(div(ts_min(returns, 20), add(ts_std(returns, 20), 1e-6)))",
         "name": "极端负收益/波动率", "direction": "+",
         "rationale": "最小日收益(取负)/波动率。极端下跌后反弹=尾部风险溢价。",
         "paradigm": "波动率x尾部风险", "gen": 1},
        {"expression": "sub(ts_max(returns, 20), neg(ts_min(returns, 20)))",
         "name": "收益不对称性", "direction": "-",
         "rationale": "最大正收益-最大负收益(取正)。正偏度>负偏度=上涨潜力>下跌风险。负向取负排名做多。",
         "paradigm": "波动率x尾部风险", "gen": 1},
        {"expression": "neg(div(ts_std(returns, 60), ts_std(returns, 20)))",
         "name": "波动率衰减率", "direction": "+",
         "rationale": "长期波动/短期波动取负。波动率收缩=风险释放=安全边际。仿GARCH波动率聚集效应。",
         "paradigm": "波动率x尾部风险", "gen": 1},
        {"expression": "neg(div(ts_mean(neg(ts_min(returns, 20)), 10), add(ts_std(returns, 20), 1e-6)))",
         "name": "下跌幅度/波动率", "direction": "+",
         "rationale": "10日均极端负收益/波动率取负。越痛反弹越强。尾部风险定价。",
         "paradigm": "波动率x尾部风险", "gen": 1},
        {"expression": "neg(rank(div(ts_std(returns, 20), add(ts_mean(returns, 60), 1e-4))))",
         "name": "波动率/长期收益比", "direction": "+",
         "rationale": "截面rank(波动率/60日均收益)取负。高波动低收益=风险定价错误=价值。",
         "paradigm": "波动率x尾部风险", "gen": 1},
        {"expression": "rank(sub(ts_std(returns, 5), ts_std(returns, 20)))",
         "name": "波动率突变(截面)", "direction": "-",
         "rationale": "截面rank(5日波动-20日波动)。波动骤升=不确定性=风险。取负=低波动异象。",
         "paradigm": "波动率x尾部风险", "gen": 1},
        {"expression": "neg(div(ts_std(sub(close, ts_delta(close, 60)), 20), ts_std(close, 20)))",
         "name": "均值回归波动率", "direction": "+",
         "rationale": "均值偏离波动/价格波动取负。高均值回归波动=趋势不确定性=反转机会。",
         "paradigm": "波动率x尾部风险", "gen": 1},
    ])

    # ── 范式4: 动量生命周期 (8个) ──
    f([
        {"expression": "neg(ts_delta(ts_zscore(close, 60), 20))",
         "name": "动量减速", "direction": "+",
         "rationale": "长期zscore的20日变化取负。动量加速→拥挤→减速→超额回归。动量生命周期。",
         "paradigm": "动量生命周期", "gen": 1},
        {"expression": "sub(ts_pct(close, 5), ts_pct(close, 20))",
         "name": "短期/中期动量差", "direction": "-",
         "rationale": "5日收益率-20日收益率。短快长慢=加速(过热),短慢长快=减速(冷却)。",
         "paradigm": "动量生命周期", "gen": 1},
        {"expression": "neg(rank(sub(ts_mean(close, 20), ts_mean(close, 60))))",
         "name": "中期趋势弱度", "direction": "+",
         "rationale": "截面取负的(20日均-60日均)。趋势减弱=反转前兆。A股短周期特征。",
         "paradigm": "动量生命周期", "gen": 1},
        {"expression": "neg(sub(div(close, ts_mean(close, 20)), div(close, ts_mean(close, 60))))",
         "name": "双均线偏离差", "direction": "+",
         "rationale": "20日偏离-60日偏离取负。短期偏离小于长期偏离=趋势衰竭中=抄底机会。",
         "paradigm": "动量生命周期", "gen": 1},
        {"expression": "neg(ts_pct(close, 5))",
         "name": "短期反转(5日)", "direction": "+",
         "rationale": "5日收益率取负。短期反转=经典A股alpha。A股短线资金T+1→T+5效应。",
         "paradigm": "动量生命周期", "gen": 1},
        {"expression": "sub(rank(ts_pct(close, 20)), rank(ts_pct(close, 5)))",
         "name": "动量切换(截面)", "direction": "+",
         "rationale": "截面rank(20日动量-5日动量)。长期动量股转向短期走弱=转势信号。",
         "paradigm": "动量生命周期", "gen": 1},
        {"expression": "neg(rank(div(ts_mean(close, 10), ts_mean(close, 40))))",
         "name": "10/40均线比(取负)", "direction": "+",
         "rationale": "截面取负(10日/40日均线)。跌破中期均线=超卖=均值回归。",
         "paradigm": "动量生命周期", "gen": 1},
        {"expression": "neg(ts_mean(sub(div(close, ts_delta(close, 1)), 1), 10))",
         "name": "10日累计反转", "direction": "+",
         "rationale": "10日日均收益率取负。连续阴跌后反弹。情绪修复效应。",
         "paradigm": "动量生命周期", "gen": 1},
    ])

    # ── 范式5: 情绪×开盘效应 (7个) ──
    f([
        {"expression": "rank(div(sub(open, ts_delta(close, 1)), add(ts_std(close, 20), 1e-6)))",
         "name": "开盘跳空标准化(截面)", "direction": "+",
         "rationale": "开盘跳空/20日波动率截面排名。标准化区分真跳空vs正常波动。V3的gap_up无波动率调整。",
         "paradigm": "情绪x开盘效应", "gen": 1},
        {"expression": "sub(div(sub(open, ts_delta(close, 1)), add(ts_std(close, 20), 1e-6)), div(sub(open, ts_delta(close, 1)), add(ts_std(close, 60), 1e-6)))",
         "name": "跳空异常度(多窗口)", "direction": "-",
         "rationale": "20日窗口z跳空 - 60日窗口z跳空。跳空显著高于历史=过度乐观=回调风险。",
         "paradigm": "情绪x开盘效应", "gen": 1},
        {"expression": "ts_mean(sub(div(open, ts_delta(close, 1)), 1), 5)",
         "name": "5日开盘动量累计", "direction": "+",
         "rationale": "5日开盘跳空均值。持续高开=隔夜资金持续流入=机构建仓痕迹。V3只有2日版。",
         "paradigm": "情绪x开盘效应", "gen": 1},
        {"expression": "neg(rank(sub(div(open, ts_delta(close, 1)), 1)))",
         "name": "跳空反转(截面取负)", "direction": "+",
         "rationale": "截面取负(开盘跳空)。跳空高开-次日回调,跳空低开-次日反弹。均值回归。",
         "paradigm": "情绪x开盘效应", "gen": 1},
        {"expression": "sub(div(close, open), 1)",
         "name": "日内收益率", "direction": "+",
         "rationale": "(收盘-开盘)/开盘。日内走势=隔夜信息消化效率。正向=信息有效吸收。",
         "paradigm": "情绪x开盘效应", "gen": 1},
        {"expression": "neg(rank(div(sub(close, open), add(sub(high, low), 1e-6))))",
         "name": "收盘位置(日内)", "direction": "+",
         "rationale": "截面取负((收盘-开盘)/振幅)。收盘在日内低位=尾盘杀跌=次日修复。",
         "paradigm": "情绪x开盘效应", "gen": 1},
        {"expression": "rank(div(sub(open, ts_min(close, 5)), add(ts_std(close, 20), 1e-6)))",
         "name": "跳空突破(5日最低)", "direction": "+",
         "rationale": "(开盘价-5日最低)/波动率排名。开盘突破近期低点后反弹=破底翻。",
         "paradigm": "情绪x开盘效应", "gen": 1},
    ])

    # ── 范式6: 筹码×机构行为 (7个) ──
    f([
        {"expression": "neg(rank(div(ts_std(volume, 10), add(ts_mean(volume, 10), 1e-6))))",
         "name": "量变异系数(取负截面)", "direction": "+",
         "rationale": "截面取负(10日量CV)。低量CV=筹码稳定=机构持仓=上涨潜力。高量CV=筹码松动=风险。",
         "paradigm": "筹码x机构行为", "gen": 1},
        {"expression": "rank(div(ts_mean(volume, 5), add(ts_std(volume, 60), 1e-6)))",
         "name": "短期量/长期量波动", "direction": "-",
         "rationale": "5日均量/60日量波动。短期放量+长期稳定=异动=可能是出货。",
         "paradigm": "筹码x机构行为", "gen": 1},
        {"expression": "neg(sub(ts_zscore(volume, 60), ts_zscore(volume, 10)))",
         "name": "量趋势衰减", "direction": "+",
         "rationale": "长期量zscore-短期量zscore取负。量能从高位回落=清洗浮筹=再次拉升前兆。",
         "paradigm": "筹码x机构行为", "gen": 1},
        {"expression": "rank(div(sub(close, ts_min(close, 60)), add(sub(ts_max(close, 60), ts_min(close, 60)), 1e-6)))",
         "name": "年度位置(Stochastic)", "direction": "-",
         "rationale": "价格在60日区间位置排名。极度高位=拥挤+获利盘=回调风险。",
         "paradigm": "筹码x机构行为", "gen": 1},
        {"expression": "neg(rank(div(ts_sum(volume, 20), add(ts_sum(volume, 60), 1e-6))))",
         "name": "20/60日量比(取负)", "direction": "+",
         "rationale": "截面取负(20日量/60日量)。低近期量=缩量洗盘=机构吸筹特征。",
         "paradigm": "筹码x机构行为", "gen": 1},
        {"expression": "neg(div(ts_mean(close, 10), add(ts_mean(close, 60), 1e-6)))",
         "name": "10/60均线乖离", "direction": "+",
         "rationale": "10/60日均线比取负。中短期均线死亡交叉后回调=超卖。金叉死叉信号。",
         "paradigm": "筹码x机构行为", "gen": 1},
        {"expression": "rank(mul(neg(ts_zscore(close, 60)), neg(div(ts_mean(volume, 5), ts_mean(volume, 60)))))",
         "name": "低位缩量(60日确认)", "direction": "+",
         "rationale": "截面rank(负长期zscore × 负量比)。长期低位+缩量=双重底部确认。",
         "paradigm": "筹码x机构行为", "gen": 1},
    ])

    # ── 范式7: 信息扩散×滞后反应 (6个) ──
    f([
        {"expression": "neg(rank(div(ts_mean(sub(close, ts_delta(close, 1)), 10), add(ts_std(sub(close, ts_delta(close, 1)), 10), 1e-6))))",
         "name": "信息效率比率(取负)", "direction": "+",
         "rationale": "截面取负(日均收益/日收益波动)。低信噪比=信息未充分扩散=滞后反应=alpha。",
         "paradigm": "信息扩散x滞后", "gen": 1},
        {"expression": "neg(rank(sub(ts_max(close, 5), ts_min(close, 5))))",
         "name": "5日价格范围(截面)", "direction": "+",
         "rationale": "截面取负(5日价差)。价格窄幅波动=筹码锁定=即将突破。盘整信号。",
         "paradigm": "信息扩散x滞后", "gen": 1},
        {"expression": "sub(ts_std(close, 20), ts_std(close, 60))",
         "name": "波动率期限结构", "direction": "-",
         "rationale": "20日波动-60日波动。近端波动<远端波动=波动率期限贴水=风险偏好上升=涨。",
         "paradigm": "信息扩散x滞后", "gen": 1},
        {"expression": "neg(div(ts_std(sub(close, ts_delta(close, 1)), 10), ts_std(sub(close, ts_delta(close, 1)), 60)))",
         "name": "收益自相关衰减", "direction": "+",
         "rationale": "10日收益波动/60日收益波动取负。收益自相关衰减=方向即将明确。",
         "paradigm": "信息扩散x滞后", "gen": 1},
        {"expression": "rank(sub(div(close, ts_delta(close, 5)), div(close, ts_delta(close, 20))))",
         "name": "短期/中期动量差(截面)", "direction": "-",
         "rationale": "截面rank(5日收益率-20日收益率)。短期补涨=滞后反应=超额即将消失。",
         "paradigm": "信息扩散x滞后", "gen": 1},
        {"expression": "rank(div(sub(high, close), add(sub(high, low), 1e-6)))",
         "name": "上影线比率(截面)", "direction": "-",
         "rationale": "(高-收盘)/振幅排名。长上影线=冲高回落=抛压=次日大概率调整。经典K线形态。",
         "paradigm": "信息扩散x滞后", "gen": 1},
    ])

    # ── 范式8: 趋势质量×动量增强 (6个) ──
    f([
        {"expression": "rank(sub(div(ts_mean(close, 5), ts_mean(close, 20)), ts_std(returns, 20)))",
         "name": "趋势强度-波动率(截面)", "direction": "+",
         "rationale": "截面rank(趋势比-波动率)。高趋势低波动=高质量趋势=持续性强。",
         "paradigm": "趋势质量x动量", "gen": 1},
        {"expression": "neg(rank(mul(ts_zscore(close, 20), ts_std(returns, 20))))",
         "name": "动量x波动惩罚(截面)", "direction": "+",
         "rationale": "截面取负(zscore × 波动率)。惩罚高波动的动量——低波动动量更可靠。",
         "paradigm": "趋势质量x动量", "gen": 1},
        {"expression": "rank(mul(ts_pct(close, 20), neg(ts_std(close, 20))))",
         "name": "动量x稳定性(截面)", "direction": "+",
         "rationale": "截面rank(20日收益率 × 负价格波动)。稳健上涨>剧烈上涨。V3无此组合。",
         "paradigm": "趋势质量x动量", "gen": 1},
        {"expression": "sub(rank(ts_pct(close, 60)), rank(ts_pct(close, 10)))",
         "name": "长期/短期动量差", "direction": "+",
         "rationale": "截面rank(60日动量-10日动量)。长期强短期弱=暂时回调=好的买点。",
         "paradigm": "趋势质量x动量", "gen": 1},
        {"expression": "neg(div(ts_std(close, 20), add(ts_mean(sub(close, ts_delta(close, 20)), 20), 1e-6)))",
         "name": "趋势信噪比", "direction": "+",
         "rationale": "波动率/(20日平均涨幅)取负。信号/噪声比越高越好, 低噪趋势=可靠。",
         "paradigm": "趋势质量x动量", "gen": 1},
        {"expression": "rank(sub(ts_sum(div(sub(close, ts_delta(close, 1)), close), 10), ts_sum(div(sub(close, ts_delta(close, 1)), close), 5)))",
         "name": "10/5日累计收益差", "direction": "-",
         "rationale": "截面rank(10日累计-5日累计)。短期减速=中期趋势持续中暂停=正常调整。",
         "paradigm": "趋势质量x动量", "gen": 1},
    ])

    # ── 范式9: 隔夜×日内分解 (5个) ──
    f([
        {"expression": "sub(div(open, ts_delta(close, 1)), sub(div(close, open), 1))",
         "name": "隔夜/日内收益差", "direction": "-",
         "rationale": "跳空-日内收益。隔夜强日内弱=高开低走=短线资金离场=需回避。",
         "paradigm": "隔夜x日内分解", "gen": 1},
        {"expression": "rank(div(sub(open, ts_min(low, 10)), add(ts_std(close, 20), 1e-6)))",
         "name": "开盘/10日最低距", "direction": "+",
         "rationale": "(开盘-10日最低)/波动率排名。开盘已从底部反弹=上周已跌透=本周修复。",
         "paradigm": "隔夜x日内分解", "gen": 1},
        {"expression": "neg(ts_mean(sub(div(open, ts_delta(close, 1)), 1), 10))",
         "name": "10日隔夜收益(取负)", "direction": "+",
         "rationale": "10日开盘跳空均值取负。持续低开=过度悲观=均值回归反弹。",
         "paradigm": "隔夜x日内分解", "gen": 1},
        {"expression": "sub(div(close, open), div(open, ts_delta(close, 1)))",
         "name": "日内/隔夜强度比", "direction": "+",
         "rationale": "日内收益/跳空。日内强于隔夜=真实买盘=非单纯情绪脉冲。",
         "paradigm": "隔夜x日内分解", "gen": 1},
        {"expression": "neg(rank(div(sub(close, open), add(ts_mean(sub(high, low), 20), 1e-6))))",
         "name": "日内涨跌幅/振幅比(取负)", "direction": "+",
         "rationale": "截面取负((收盘-开盘)/平均振幅)。低日内涨幅=回调日=次日反弹。",
         "paradigm": "隔夜x日内分解", "gen": 1},
    ])

    # ── 范式10: 行业相对强度×截面离散 (5个) ──
    # 注意: JQ无行业数据, 以下用价格相对偏差代理
    f([
        {"expression": "neg(rank(sub(ts_zscore(close, 60), ts_zscore(volume, 60))))",
         "name": "价格/量相对强度差", "direction": "+",
         "rationale": "截面取负(价格zscore-量zscore)。价格走势弱于量能=滞后=补涨潜力。",
         "paradigm": "相对强度x截面", "gen": 1},
        {"expression": "rank(sub(div(close, ts_mean(close, 20)), div(volume, ts_mean(volume, 20))))",
         "name": "价量相对偏离", "direction": "+",
         "rationale": "截面rank(价格偏离-量偏离)。价格弱于量=量升价不升=即将补涨。",
         "paradigm": "相对强度x截面", "gen": 1},
        {"expression": "neg(rank(sub(ts_pct(close, 20), ts_pct(close, 60))))",
         "name": "中期/长期动量差(截面)", "direction": "+",
         "rationale": "截面取负(20日收益率-60日收益率)。中期弱于长期=暂时性回调。",
         "paradigm": "相对强度x截面", "gen": 1},
        {"expression": "rank(div(ts_sum(sub(close, ts_delta(close, 1)), 5), ts_std(close, 10)))",
         "name": "5日累计/10日波动", "direction": "+",
         "rationale": "截面rank(5日收益/10日波动)。高收益低波动=有效趋势。风险调整收益。",
         "paradigm": "相对强度x截面", "gen": 1},
        {"expression": "neg(div(sub(ts_max(close, 20), close), add(ts_std(close, 20), 1e-6)))",
         "name": "回调深度/波动率", "direction": "+",
         "rationale": "(20日最高-收盘)/波动率, 取负。深度回调+低波动=假跌=反弹。",
         "paradigm": "相对强度x截面", "gen": 1},
    ])

    # ── 范式11: 极端事件×涨跌停效应 (4个, 注意JQ无涨跌停, 用极值代理) ──
    f([
        {"expression": "neg(rank(sub(div(close, ts_delta(close, 1)), 1.095)))",
         "name": "接近涨停距(取负)", "direction": "+",
         "rationale": "截面取负(日收益率-涨停阈值)。远离涨停=安全边际。靠近涨停=次日大概率低开。",
         "paradigm": "极端事件", "gen": 1},
        {"expression": "rank(div(ts_max(returns, 20), ts_std(returns, 20)))",
         "name": "最大收益/波动率", "direction": "+",
         "rationale": "20日最大日收益/波动率排名。最大涨幅大=上涨弹性强。弹性筛选。",
         "paradigm": "极端事件", "gen": 1},
        {"expression": "neg(rank(div(neg(ts_min(returns, 20)), ts_std(returns, 20))))",
         "name": "最大跌幅/波动率(取负)", "direction": "+",
         "rationale": "最大负收益/波动率排名取负。极端下跌后=反弹机会。弹性+反转。",
         "paradigm": "极端事件", "gen": 1},
        {"expression": "sub(ts_max(close, 5), ts_min(close, 5))",
         "name": "5日振幅", "direction": "-",
         "rationale": "5日最高减最低。短期振幅扩大=分歧=不确定性=风险。",
         "paradigm": "极端事件", "gen": 1},
    ])

    return factors


# ═══════════════════════════════════════════════════════════
# EFS 进化: 幸存者模式提取 → 下一轮变异生成
# ═══════════════════════════════════════════════════════════

def extract_survivor_patterns(survivors: List[AlphaCandidate]) -> Dict:
    """从幸存因子中提取设计模式, 用于知识蒸馏"""

    patterns = {
        "paradigms": defaultdict(int),      # 范式分布
        "operators": defaultdict(int),      # 算子使用频率
        "inputs": defaultdict(int),         # 输入使用频率
        "window_sizes": [],                 # 窗口大小
        "direction": defaultdict(int),      # 方向偏好
        "complexity_range": [0, 0],         # 复杂度范围
        "expr_hashes": set(),               # 表达式去重
        "paradigm_operators": defaultdict(set),  # 范式→算子映射
    }

    for s in survivors:
        # 范式
        rationale = s.rationale.lower()
        paradigm_map = {
            "流动性": "liquidity", "微观": "microstructure",
            "资金流": "moneyflow", "量价": "volprice",
            "波动": "volatility", "尾部": "tail",
            "动量": "momentum", "反转": "reversal",
            "情绪": "sentiment", "开盘": "opening",
            "筹码": "position", "机构": "institutional",
            "信息": "information", "滞后": "lag",
            "趋势": "trend", "隔夜": "overnight",
            "日内": "intraday", "极端": "extreme",
            "相对": "relative", "截面": "crosssection",
        }
        for kw, paradigm in paradigm_map.items():
            if kw in rationale:
                patterns["paradigms"][paradigm] += 1
                break
        else:
            patterns["paradigms"]["other"] += 1

        # 算子
        expr = s.expression
        op_list = ["rank(", "ts_zscore(", "ts_std(", "ts_mean(", "ts_min(", "ts_max(",
                    "ts_sum(", "ts_delta(", "ts_pct(", "div(", "sub(", "add(", "mul(",
                    "neg(", "log(", "sqrt(", "abs(", "inv("]
        for op in op_list:
            if op in expr:
                patterns["operators"][op] += 1

        # 输入
        for inp in ["close", "open", "high", "low", "volume", "returns"]:
            if inp in expr:
                patterns["inputs"][inp] += 1

        # 窗口
        w_matches = re.findall(r',\s*(\d+)\)', expr)
        for w in w_matches:
            patterns["window_sizes"].append(int(w))

        # 方向
        patterns["direction"][s.direction] += 1

        # 复杂度
        n = s.complexity_nodes
        if n > 0:
            if patterns["complexity_range"][0] == 0 or n < patterns["complexity_range"][0]:
                patterns["complexity_range"][0] = n
            patterns["complexity_range"][1] = max(patterns["complexity_range"][1], n)

        # 表达式hash
        patterns["expr_hashes"].add(s.expression[:60])

    return patterns


def build_distillation_prompt(patterns: Dict, gen: int,
                               all_prev_survivors: List[AlphaCandidate]) -> str:
    """基于幸存者模式, 构建知识蒸馏摘要"""

    lines = [f"## EFS 第{gen}代幸存因子知识蒸馏\n"]

    # 范式分布
    paradigm_sorted = sorted(patterns["paradigms"].items(), key=lambda x: -x[1])
    lines.append("### 成功范式 (按频率)")
    for p, cnt in paradigm_sorted[:5]:
        lines.append(f"- {p}: {cnt}个因子进入幸存")
    lines.append("")

    # 算子频率
    op_sorted = sorted(patterns["operators"].items(), key=lambda x: -x[1])
    lines.append("### 高频算子")
    for op, cnt in op_sorted[:8]:
        lines.append(f"- {op}: {cnt}次")
    lines.append("")

    # 输入频率
    inp_sorted = sorted(patterns["inputs"].items(), key=lambda x: -x[1])
    lines.append("### 高频输入")
    for inp, cnt in inp_sorted:
        lines.append(f"- {inp}: {cnt}次")
    lines.append("")

    # 窗口偏好
    if patterns["window_sizes"]:
        from statistics import median
        ws = patterns["window_sizes"]
        lines.append(f"### 窗口偏好: 中位={int(median(ws))}, 范围=[{min(ws)}, {max(ws)}]")
    lines.append("")

    # 复杂度范围
    lines.append(f"### 复杂度: [{patterns['complexity_range'][0]}, {patterns['complexity_range'][1]}]")
    lines.append("")

    # 方向偏好
    dir_sorted = sorted(patterns["direction"].items(), key=lambda x: -x[1])
    lines.append(f"### 方向偏好: {dict(dir_sorted)}")
    lines.append("")

    # 失败范式 (不在幸存中的范式)
    used_paradigms = set(patterns["paradigms"].keys())
    all_paradigms = {"liquidity", "microstructure", "moneyflow", "volprice",
                     "volatility", "tail", "momentum", "reversal",
                     "sentiment", "opening", "position", "institutional",
                     "information", "lag", "trend", "overnight",
                     "intraday", "extreme", "relative", "crosssection"}
    unused = all_paradigms - used_paradigms
    if unused:
        lines.append(f"### 未产生幸存者的范式: {', '.join(sorted(unused))}")
    lines.append("")

    return "\n".join(lines)


def generate_evolution_factors(gen: int,
                                prev_survivors: List[AlphaCandidate],
                                patterns: Dict) -> List[dict]:
    """
    基于上代幸存者模式, 生成进化变异因子。

    策略:
    1. 对每位幸存者, 生成2-3个变异 (换窗口/换算子/加截面)
    2. 对成功的范式, 生成新的组合变体
    3. 对失败的范式, 生成修正版(换方向/加过滤)
    """
    factors = []

    # 1. 幸存者变异: 每个幸存者生成2-3个
    for s in prev_survivors[:10]:
        expr = s.expression
        name_base = s.name.split('(')[0].strip()
        # 换窗口: e.g. 20→10, 20→40
        for new_w, suffix in [(10, "short"), (40, "long"), (5, "micro")]:
            vary = re.sub(r',\s*20\)', f', {new_w})', expr)
            vary = re.sub(r',\s*60\)', f', {max(new_w*3, 20)})', vary)
            if vary != expr:
                factors.append({
                    "expression": vary,
                    "name": f"{name_base}_gen{gen}_{suffix}",
                    "direction": s.direction,
                    "rationale": f"幸存因子变异(窗口{new_w}): {s.rationale[:60]}",
                    "paradigm": f"进化_窗口变异",
                    "gen": gen,
                })

        # 加截面算子: 原表达式外面包rank()
        if "rank(" not in expr[:20]:
            vary2 = f"rank({expr})"
            factors.append({
                "expression": vary2,
                "name": f"{name_base}_gen{gen}_cs",
                "direction": s.direction,
                "rationale": f"幸存因子加截面排名: {s.rationale[:50]}",
                "paradigm": f"进化_截面增强",
                "gen": gen,
            })

        # 加波动率过滤: mul(原, neg(ts_std(...)))
        if "ts_std" not in expr:
            vary3 = f"mul({expr}, neg(ts_std(close, 20)))"
            factors.append({
                "expression": vary3,
                "name": f"{name_base}_gen{gen}_volfilter",
                "direction": s.direction,
                "rationale": f"幸存因子x低波动过滤: 结合波动率质量的{s.rationale[:40]}",
                "paradigm": f"进化_波动过滤",
                "gen": gen,
            })

    # 2. 成功范式新组合
    top_paradigms = sorted(patterns["paradigms"].items(), key=lambda x: -x[1])[:4]
    top_pnames = [p for p, _ in top_paradigms]

    # 范式交叉: 取top2范式, 生成组合因子
    if len(top_pnames) >= 2:
        cross_expressions = [
            (f"rank(mul(neg(ts_zscore(close, 20)), neg(ts_zscore(volume, 20))))",
             "动量x量能(进化_交叉)", "gen_cross1", "+"),
            (f"rank(sub(ts_pct(close, 10), div(ts_std(close, 20), ts_mean(close, 20))))",
             "动量-波动率(进化_交叉)", "gen_cross2", "-"),
            (f"neg(rank(mul(div(sub(high, low), add(ts_mean(volume, 20), 1e-6)), ts_zscore(close, 10))))",
             "流动性x短期动量(进化_交叉)", "gen_cross3", "+"),
            (f"rank(div(ts_mean(sub(open, ts_delta(close, 1)), 5), add(ts_std(volume, 20), 1e-6)))",
             "开盘动量/量波动(进化_交叉)", "gen_cross4", "+"),
        ]
        for expr, rationale, name, direction in cross_expressions[:3]:
            factors.append({
                "expression": expr, "name": name, "direction": direction,
                "rationale": f"{rationale}: 融合{top_pnames[0]}+{top_pnames[1]}范式",
                "paradigm": f"进化_交叉", "gen": gen,
            })

    # 3. 补全未成功范式: 对Gen1失败的范式, 生成修正版
    # 这些范式在Gen1中没产生幸存但可能有潜力
    unused_paradigms = {"extreme", "relative", "crosssection", "information", "lag"} - set(top_pnames)
    if unused_paradigms:
        # 极端事件补全: 更保守的配置
        remedial_factors = [
            ("neg(rank(sub(div(close, ts_delta(close, 1)), 1.05)))",
             "接近涨停补偿(修正)", "rem_extreme1", "+",
             "负收益排名(修正版极端事件——用5%阈值替代9.5%)"),
            ("neg(rank(div(sub(ts_max(high, 20), close), add(ts_std(close, 20), 1e-6))))",
             "高点回调比例(修正)", "rem_extreme2", "+",
             "距20日高点的回落幅度排名取负——深度回调后在低波动环境中反弹"),
        ]
        for expr, name, fname, dir, rationale in remedial_factors:
            factors.append({
                "expression": expr, "name": fname, "direction": dir,
                "rationale": rationale, "paradigm": f"进化_补救", "gen": gen,
            })

    # 去重
    seen = set()
    unique = []
    for f in factors:
        key = f["expression"][:80]
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


# ═══════════════════════════════════════════════════════════
# V3集成 (进化版)
# ═══════════════════════════════════════════════════════════

def build_v3_integration_v2(all_survivors: List[AlphaCandidate]) -> List[dict]:
    """
    V3集成 — 将EFS进化的最终幸存者集成到V3。

    策略:
    1. 保留V3的3个核心复合 (已验证)
    2. 从所有代幸存者中选最优3-4个, 配对成新复合
    3. 优先选择不同范式的幸存者以避免冗余
    """

    preserved = [
        {"name": "comp1_overnight_tvma", "a_name": "overnight_5d", "b_name": "tvma_20",
         "from_v3": True, "a_type": "v3", "b_type": "v3",
         "rationale": "隔夜累计x量价趋势: V3最强复合"},
        {"name": "comp2_dollar_turnover", "a_name": "dollar_vol_20d", "b_name": "turnover_std_cv",
         "from_v3": True, "a_type": "v3", "b_type": "v3",
         "rationale": "成交额x换手稳定性: V3次强"},
        {"name": "comp3_moneyflow_ret3m", "a_name": "money_flow_20", "b_name": "ret_3m",
         "from_v3": True, "a_type": "v3", "b_type": "v3",
         "rationale": "资金流x动量反转: V3原复合"},
    ]

    new_composites = []

    # 按范式分类幸存者
    paradigm_groups = defaultdict(list)
    for s in all_survivors:
        rationale = s.rationale.lower()
        if "流动性" in rationale or "量" in rationale:
            paradigm_groups["liquidity"].append(s)
        elif "波动" in rationale or "尾部" in rationale:
            paradigm_groups["volatility"].append(s)
        elif "动量" in rationale or "反转" in rationale or "趋势" in rationale:
            paradigm_groups["momentum"].append(s)
        elif "资金流" in rationale or "量价" in rationale:
            paradigm_groups["moneyflow"].append(s)
        elif "情绪" in rationale or "开盘" in rationale or "隔夜" in rationale:
            paradigm_groups["sentiment"].append(s)
        else:
            paradigm_groups["other"].append(s)

    # 配对策略: 每个范式的幸存者 × V3因子
    pairings = [
        ("liquidity", "tvma_20", "流动性质量 × 趋势"),
        ("momentum", "turnover_std_cv", "动量 × 换手稳定性"),
        ("volatility", "money_flow_20", "波动率 × 资金流"),
        ("moneyflow", "ret_3m", "资金流质量 × 动量反转"),
        ("sentiment", "turnover_std_cv", "情绪 × 流动性过滤"),
    ]

    used_v3 = {"tvma_20", "dollar_vol_20d", "turnover_std_cv",
               "money_flow_20", "ret_3m"}
    pair_used = set()

    comp_idx = 5  # start from comp5 (comp1-3 are preserved)
    for paradigm, v3_factor, rationale_template in pairings:
        if paradigm not in paradigm_groups or not paradigm_groups[paradigm]:
            continue
        if v3_factor in pair_used:
            continue

        # 选该范式ICIR最高(绝对值)的幸存者
        best = max(paradigm_groups[paradigm], key=lambda x: abs(x.icir) if hasattr(x, 'icir') and x.icir != 0 else 0)

        new_composites.append({
            "name": f"comp{comp_idx}_{best.id}_{v3_factor}",
            "a_type": "new", "a_name": best.name, "a_expr": best.expression,
            "b_type": "v3", "b_name": v3_factor,
            "rationale": f"{rationale_template}: {best.rationale[:40]}",
            "from_v3": False,
        })
        pair_used.add(v3_factor)
        comp_idx += 1

    # 确保至少有2个新复合
    while len(new_composites) < 2:
        # 从other组选
        if paradigm_groups["other"]:
            s = paradigm_groups["other"][0]
            avail_v3 = list(used_v3 - pair_used)
            if avail_v3:
                v3f = avail_v3[0]
                new_composites.append({
                    "name": f"comp{comp_idx}_{s.id}_{v3f}",
                    "a_type": "new", "a_name": s.name, "a_expr": s.expression,
                    "b_type": "v3", "b_name": v3f,
                    "rationale": f"进化幸存({s.rationale[:30]}) × {v3f}",
                    "from_v3": False,
                })
                pair_used.add(v3f)
                comp_idx += 1
            else:
                break
        else:
            break

    # 如果还不够, 用V3原复合补齐
    if len(new_composites) < 2:
        new_composites.append({
            "name": "comp4_retopen_skew",
            "a_type": "v3", "a_name": "ret_open_2d",
            "b_type": "v3", "b_name": "skewness_20",
            "rationale": "开盘动量x收益偏度: V3原comp4",
            "from_v3": True,
        })
        new_composites.append({
            "name": "comp5_gapup_spread",
            "a_type": "v3", "a_name": "gap_up",
            "b_type": "v3", "b_name": "relative_spread",
            "rationale": "跳空x振幅: V3原comp5",
            "from_v3": True,
        })

    all_composites = preserved + new_composites[:4]
    return all_composites[:7]  # max 7 composites


# ═══════════════════════════════════════════════════════════
# JQ生成 (从v1复制, 略作调整)
# ═══════════════════════════════════════════════════════════

# 复用v1中的JQ生成函数
from alpha_agent_pipeline import (
    generate_jq_strategy as generate_jq_strategy_v1,
    _build_jq_code, _gen_v3_factor_funcs, _gen_new_factor_func,
    _gen_composite_func, _validate_jq, _translate_to_jq_numpy, _translate_expr,
)


def generate_jq_strategy(composites, candidates, output_path=None):
    """包装v1的JQ生成, 使用进化版的candidates"""
    return generate_jq_strategy_v1(composites, candidates, output_path)


# ═══════════════════════════════════════════════════════════
# 主流程: 80+ → 三重约束 → EFS 3代 → V3 → JQ
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("AlphaAgent Pipeline v2: 80+候选 → 三重约束 → EFS 3代 → V3 → JQ")
    print("=" * 70)

    config = AlphaAgentConfig(
        originality_threshold=0.70,
        max_complexity=28,
        hypothesis_min_score=3,
        n_survivors=15,
        icir_threshold=0.20,
        output_dir=str(ROOT / "output"),
    )
    agent = AlphaAgent(config)
    init_v3_pool()

    # ── Gen1: 批量生成80+ ──
    print("\n" + "=" * 60)
    print("[Gen1] 大规模因子生成 (80+ 候选)")
    print("=" * 60)

    gen1_specs = generate_gen1_factors()
    print(f"  生成 {len(gen1_specs)} 个因子候选")

    gen1_candidates = agent.add_manual_factors(gen1_specs, generation=1)
    print(f"  添加 {len(gen1_candidates)} 个候选")

    # 三重约束
    gen1_passed = agent.apply_triple_constraints(gen1_candidates, verbose=False)
    agent.all_candidates = gen1_candidates

    # 统计
    n_orig_fail = sum(1 for c in gen1_candidates if not c.originality_ok)
    n_align_fail = sum(1 for c in gen1_candidates if not c.alignment_ok)
    n_comp_fail = sum(1 for c in gen1_candidates if not c.complexity_ok)
    n_parse_fail = len(gen1_candidates) - sum(1 for c in gen1_candidates if c.ast_node is not None)
    print(f"  三重约束: {len(gen1_passed)}/{len(gen1_candidates)} 通过")
    print(f"    解析失败: {n_parse_fail}")
    print(f"    Originality失败: {n_orig_fail}")
    print(f"    Alignment失败: {n_align_fail}")
    print(f"    Complexity失败: {n_comp_fail}")

    # 范式分布
    paradigm_counts = defaultdict(int)
    for c in gen1_passed:
        for spec in gen1_specs:
            if spec["name"] == c.name:
                paradigm_counts[spec.get("paradigm", "unknown")] += 1
                break
    print(f"\n  通过范式分布 (top-8):")
    for p, cnt in sorted(paradigm_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"    {p}: {cnt}")

    # ── Gen1幸存者选择 (基于启发式: economic rationale + complexity + originality) ──
    # 注意: 无ICIR数据时不跑实际评估, 用结构质量评分
    print(f"\n  [Gen1] 选择幸存者 (结构质量评分)...")

    def structural_score(c: AlphaCandidate) -> float:
        s = 0.0
        # Originality: max_sim越低越好
        s += (1.0 - c.originality_max_sim) * 2.0
        # Alignment
        s += c.alignment_score * 1.0
        # Complexity: 8-18最佳, 太简单或太复杂都扣分
        n = c.complexity_nodes
        if 8 <= n <= 18:
            s += 1.5
        elif 18 < n <= 24:
            s += 0.8
        elif n > 24:
            s += 0.3
        else:
            s -= 1.0
        # 截面算子加分
        if "rank(" in c.expression:
            s += 1.0
        if "neg(" in c.expression:
            s += 0.5
        # 多输入加分
        unique_inputs = sum(1 for inp in ["close", "open", "high", "low", "volume"]
                          if inp in c.expression)
        s += unique_inputs * 0.3
        return s

    scored = [(structural_score(c), c) for c in gen1_passed]
    scored.sort(key=lambda x: -x[0])
    gen1_survivors = [c for _, c in scored[:config.n_survivors]]

    print(f"  Gen1 幸存者: {len(gen1_survivors)}/{len(gen1_passed)}")
    for i, s in enumerate(gen1_survivors[:5]):
        print(f"    {i+1}. {s.name}: score={structural_score(s):.2f}, "
              f"orig={s.originality_max_sim:.2f}, comp={s.complexity_nodes}")

    agent.survivors_by_gen[1] = gen1_survivors

    # ── Gen1 知识蒸馏 ──
    gen1_patterns = extract_survivor_patterns(gen1_survivors)
    gen1_distill = build_distillation_prompt(gen1_patterns, 1, gen1_survivors)
    print(f"\n  [Gen1] 知识蒸馏完成")
    print(f"    成功范式: {dict(sorted(gen1_patterns['paradigms'].items(), key=lambda x: -x[1]))}")
    print(f"    高频算子: {dict(sorted(gen1_patterns['operators'].items(), key=lambda x: -x[1])[:5])}")
    print(f"    中位窗口: {int(__import__('statistics').median(gen1_patterns['window_sizes'])) if gen1_patterns['window_sizes'] else 'N/A'}")

    # ── Gen2: 进化变异 ──
    print("\n" + "=" * 60)
    print("[Gen2] EFS进化变异 (知识蒸馏 → 新候选)")
    print("=" * 60)

    gen2_specs = generate_evolution_factors(2, gen1_survivors, gen1_patterns)
    print(f"  生成 {len(gen2_specs)} 个进化变体")

    gen2_candidates = agent.add_manual_factors(gen2_specs, generation=2)
    gen2_passed = agent.apply_triple_constraints(gen2_candidates, verbose=False)
    agent.all_candidates.extend(gen2_candidates)
    print(f"  三重约束: {len(gen2_passed)}/{len(gen2_candidates)} 通过")

    # 更新因子池 (加入Gen1幸存者用于原始性检查)
    for s in gen1_survivors:
        if s.ast_node:
            agent.pool.append(s.ast_node)

    scored2 = [(structural_score(c), c) for c in gen2_passed]
    scored2.sort(key=lambda x: -x[0])
    gen2_survivors = [c for _, c in scored2[:config.n_survivors]]

    print(f"  Gen2 幸存者: {len(gen2_survivors)}")
    for i, s in enumerate(gen2_survivors[:5]):
        print(f"    {i+1}. {s.name}: orig={s.originality_max_sim:.2f}, "
              f"comp={s.complexity_nodes}, align={s.alignment_score}")

    agent.survivors_by_gen[2] = gen2_survivors

    # ── Gen2 知识蒸馏 ──
    gen2_patterns = extract_survivor_patterns(gen2_survivors)
    gen2_distill = build_distillation_prompt(gen2_patterns, 2, gen2_survivors)
    print(f"\n  [Gen2] 知识蒸馏完成")
    print(f"    成功范式: {dict(sorted(gen2_patterns['paradigms'].items(), key=lambda x: -x[1]))}")

    # ── Gen3: 最终进化 ──
    print("\n" + "=" * 60)
    print("[Gen3] EFS最终进化 (二次蒸馏 → 精选候选)")
    print("=" * 60)

    gen3_specs = generate_evolution_factors(3, gen2_survivors, gen2_patterns)
    print(f"  生成 {len(gen3_specs)} 个进化变体")

    gen3_candidates = agent.add_manual_factors(gen3_specs, generation=3)
    gen3_passed = agent.apply_triple_constraints(gen3_candidates, verbose=False)
    agent.all_candidates.extend(gen3_candidates)
    print(f"  三重约束: {len(gen3_passed)}/{len(gen3_candidates)} 通过")

    # 更新因子池
    for s in gen2_survivors:
        if s.ast_node:
            agent.pool.append(s.ast_node)

    scored3 = [(structural_score(c), c) for c in gen3_passed]
    scored3.sort(key=lambda x: -x[0])
    gen3_survivors = [c for _, c in scored3[:10]]  # top-10 for Gen3

    print(f"  Gen3 幸存者: {len(gen3_survivors)}")
    for i, s in enumerate(gen3_survivors[:8]):
        print(f"    {i+1}. {s.name}: score={scored3[i][0]:.2f}, "
              f"orig={s.originality_max_sim:.2f}, comp={s.complexity_nodes}")

    agent.survivors_by_gen[3] = gen3_survivors

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("[汇总] EFS 3代进化结果")
    print("=" * 60)
    print(f"  Gen1: {len(gen1_candidates)}候选 → {len(gen1_passed)}通过 → {len(gen1_survivors)}幸存")
    print(f"  Gen2: {len(gen2_candidates)}候选 → {len(gen2_passed)}通过 → {len(gen2_survivors)}幸存")
    print(f"  Gen3: {len(gen3_candidates)}候选 → {len(gen3_passed)}通过 → {len(gen3_survivors)}幸存")
    print(f"  总候选: {len(agent.all_candidates)}")

    # ── V3集成 ──
    print("\n" + "=" * 60)
    print("[V3集成] EFS幸存因子 → V3复合")
    print("=" * 60)

    all_survivors = gen1_survivors + gen2_survivors + gen3_survivors
    # 去重
    seen_names = set()
    unique_survivors = []
    for s in all_survivors:
        key = s.name[:30]
        if key not in seen_names:
            seen_names.add(key)
            unique_survivors.append(s)
    print(f"  去重幸存者: {len(unique_survivors)}")

    composites = build_v3_integration_v2(unique_survivors)
    print(f"  {len(composites)} 个复合:")
    for comp in composites:
        tag = "[V3原]" if comp.get("from_v3") else "[进化]"
        print(f"  {tag} {comp['name']}: {comp['rationale'][:60]}")

    # ── JQ生成 ──
    print("\n" + "=" * 60)
    print("[JQ] 策略生成")
    print("=" * 60)

    jq_path = generate_jq_strategy(composites, unique_survivors)

    # ── 导出 ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    meta = {
        "pipeline": "AlphaAgent v2 — EFS 3-Gen Evolution",
        "timestamp": timestamp,
        "gen1": {"candidates": len(gen1_specs), "passed_triple": len(gen1_passed),
                 "survivors": len(gen1_survivors)},
        "gen2": {"candidates": len(gen2_specs), "passed_triple": len(gen2_passed),
                 "survivors": len(gen2_survivors)},
        "gen3": {"candidates": len(gen3_specs), "passed_triple": len(gen3_passed),
                 "survivors": len(gen3_survivors)},
        "total_candidates": len(agent.all_candidates),
        "composites": [{"name": c["name"], "rationale": c["rationale"],
                        "from_v3": c.get("from_v3", False)} for c in composites],
        "jq_file": jq_path,
    }
    meta_path = ROOT / "output" / f"alpha_agent_v2_meta_{timestamp}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"Pipeline v2 完成!")
    print(f"  元数据: {meta_path}")
    print(f"  JQ策略: {jq_path}")
    print(f"{'='*70}")

    return jq_path


if __name__ == "__main__":
    jq_file = main()
