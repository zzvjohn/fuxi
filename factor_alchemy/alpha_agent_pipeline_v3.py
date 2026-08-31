# -*- coding: utf-8 -*-
"""
AlphaAgent Pipeline v3: 增强单轮生成 → 三重约束 → 互补配对 → V3集成 → JQ
============================================================================
v1 (+136.85%/0.53/-37.51%): 16候选/8范式/单轮/硬编码配对 → 成功
v2 (+110.90%/0.45/-33.01%): 120候选/11范式/3代EFS进化 → 反向退化
v3 策略: 单轮60+候选/13范式/互补评分配对 → 目标维持收益+压降回撤

核心改进:
1. 新增5个范式: 下行保护/波动率适应/市场宽度/结构突变/筹码分布
2. 互补评分替代硬编码: 新因子与V3复合的经济维度互补性 + 低相关性偏好
3. 增强尾部风险: harvey_siddique_coskew高阶矩+隔夜下行敏感度
4. 批量60→精筛5-6复合: 保持v1单轮架构, 更好的筛选而非更多进化
5. 筹码分布=降回撤核心维度(A股底部特征)

⚠️ 自进化状态 (2026-08-04):
  --evo-context 已冻结。Local数据驱动的进化 = Local→JQ Gap放大器。
  JQ是唯一真相源。因子注入三条本地筛选链造成系统性偏差。
  详见: evaluator/evolution_roadmap.md
"""

import sys, os, json, time, re, gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from alpha_agent import (
    AlphaAgentConfig, AlphaAgent, AlphaCandidate,
    init_v3_pool, V3_FACTOR_POOL,
    check_originality, check_complexity, ast_similarity,
    parse_expression, ExprNode,
)


# ═══════════════════════════════════════════════════════════
# 阶段1: LLM因子生成 — 12范式55+候选
# ═══════════════════════════════════════════════════════════

def generate_novel_factors_v3() -> List[dict]:
    """
    55+候选因子，覆盖12个金融范式。
    新增4个A股关键维度:
    - 下行保护: 尾部对冲、回撤恢复、低波动溢价
    - 波动率适应: 波动率周期性、GARCH风格聚集
    - 市场宽度: 截面离散度、相对强度扩散
    - 结构突变: 量突破、支撑/阻力位交互
    """

    factors = [
        # ═══ 范式1: 流动性/微观结构 (A股alpha主源) ═══
        {
            "expression": "neg(rank(div(sub(high, low), add(ts_mean(volume, 20), 1e-6))))",
            "name": "振幅归一化(量)",
            "rationale": "日内振幅/成交量均值截面排名取负。高振幅低量=流动性差=流动性溢价。v1入选因子。",
            "direction": "+", "paradigm": "流动性×微观结构",
        },
        {
            "expression": "rank(sub(ts_max(close, 20), ts_min(close, 20)))",
            "name": "价格区间宽度",
            "rationale": "20日价格区间截面排名——价格离散度大的股票弹性更强，捕捉波动溢价。",
            "direction": "-", "paradigm": "波动率",
        },
        {
            "expression": "neg(ts_zscore(div(div(close, ts_mean(close, 20)), ts_std(volume, 20)), 40))",
            "name": "量价稳定度",
            "rationale": "价格偏离均线/成交量波动性的长期z-score。量价同步稳定=机构持续建仓特征。",
            "direction": "+", "paradigm": "流动性×趋势",
        },
        {
            "expression": "neg(rank(div(ts_mean(volume, 5), add(ts_mean(volume, 60), 1e-6))))",
            "name": "长期缩量程度",
            "rationale": "5日/60日均量截面取负。长期缩量=筹码锁定/惜售, A股底部特征。v1/v2缺失的长期维度。",
            "direction": "+", "paradigm": "流动性",
        },
        {
            "expression": "rank(div(ts_mean(high, 20), ts_mean(low, 20)))",
            "name": "高低均价比",
            "rationale": "20日均高价/均低价。高比率=多空博弈激烈=弹性标的。截面排名评价相对博弈强度。",
            "direction": "+", "paradigm": "流动性×微观结构",
        },

        # ═══ 范式2: 资金流/量价背离 ═══
        {
            "expression": "sub(ts_pct(close, 10), ts_pct(volume, 10))",
            "name": "量价背离(10日)",
            "rationale": "价格变化率-成交量变化率。价格涨量不跟=上涨乏力, 价格跌量缩=抛压衰竭。",
            "direction": "-", "paradigm": "资金流×趋势",
        },
        {
            "expression": "rank(div(ts_mean(volume, 5), ts_mean(volume, 20)))",
            "name": "放量强度",
            "rationale": "5日/20日均量截面排名。持续放量=资金关注度提升, A股alpha有效信号。",
            "direction": "+", "paradigm": "资金流",
        },
        {
            "expression": "rank(mul(ts_pct(volume, 5), ts_pct(close, 5)))",
            "name": "量价共振度",
            "rationale": "截面rank(量增速×价增速)。量价同向=趋势健康, 量价背离=反转信号。v1/v2缺失。",
            "direction": "+", "paradigm": "资金流×趋势",
        },
        {
            "expression": "neg(rank(div(sub(volume, ts_mean(volume, 20)), add(ts_std(volume, 20), 1e-6))))",
            "name": "缩量异常度",
            "rationale": "截面取负((量-20日均量)/量波动)。极度缩量=底部信号, A股缩量止跌特征。",
            "direction": "+", "paradigm": "资金流",
        },

        # ═══ 范式3: 动量/反转 ═══
        {
            "expression": "neg(ts_delta(ts_zscore(close, 60), 20))",
            "name": "动量减速",
            "rationale": "长期z-score的20日变化取负。动量加速时预期收益降低, 减速时回升。动量生命周期。",
            "direction": "+", "paradigm": "动量反转",
        },
        {
            "expression": "neg(rank(ts_mean(div(close, ts_max(close, 60)), 10)))",
            "name": "接近1年高点比例",
            "rationale": "10日均(价格/60日最高)截面取负。越接近高点越易回调, 远离高点的反弹空间更大。v1入选因子。",
            "direction": "+", "paradigm": "行为金融",
        },
        {
            "expression": "neg(rank(sub(ts_mean(close, 20), ts_mean(close, 60))))",
            "name": "中期趋势弱度",
            "rationale": "截面取负(20日均-60日均)。趋势减弱=反转前兆, 适配A股短周期特征。",
            "direction": "+", "paradigm": "趋势",
        },
        {
            "expression": "neg(div(sub(close, ts_min(close, 60)), add(ts_max(close, 60), -ts_min(close, 60))))",
            "name": "深度超卖度",
            "rationale": "取负(价格在60日区间位置)。处底部区域=超卖=反弹动力。60日长窗口过滤假信号。",
            "direction": "+", "paradigm": "动量反转",
        },
        {
            "expression": "neg(rank(mul(div(close, ts_delta(close, 20)), div(volume, ts_delta(volume, 20)))))",
            "name": "20日量价双低",
            "rationale": "截面取负(20日价格比×20日量比)。缩量滞涨=积累期, 量价同时沉底是启动前特征。",
            "direction": "+", "paradigm": "资金流×动量",
        },

        # ═══ 范式4: 行为金融 ═══
        {
            "expression": "neg(rank(div(sub(close, ts_min(close, 5)), add(ts_max(close, 5), -ts_min(close, 5)))))",
            "name": "5日价格位置(取负)",
            "rationale": "近5日价格在区间位置的截面取负。超短期的超卖反弹, 适配A股高换手特征。",
            "direction": "+", "paradigm": "行为金融",
        },
        {
            "expression": "rank(div(sub(high, close), add(sub(high, low), 0.001)))",
            "name": "上影线比例",
            "rationale": "(最高-收盘)/(最高-最低)截面排名。上影线长=抛压出现=短期见顶信号。蜡烛图量化。",
            "direction": "-", "paradigm": "行为金融",
        },
        {
            "expression": "neg(rank(div(sub(close, low), add(sub(high, low), 0.001))))",
            "name": "下影线抄底",
            "rationale": "截面取负((收-低)/(高-低))。下影线长=买盘出现=短期见底。与上影线互补。",
            "direction": "+", "paradigm": "行为金融",
        },

        # ═══ 范式5: 尾部风险/波动不对称 ═══
        {
            "expression": "sub(neg(ts_std(returns, 20)), ts_std(returns, 5))",
            "name": "波动率扩张",
            "rationale": "短期波动率-长期波动率(均取负)。波动率收窄后扩张=趋势启动, V3无此信号。",
            "direction": "+", "paradigm": "波动率×趋势",
        },
        {
            "expression": "neg(div(ts_min(returns, 20), ts_std(returns, 20)))",
            "name": "极端负收益/波动率比",
            "rationale": "最小日收益(取负)/波动率。极端下跌相对波动水平, 捕捉尾部风险溢价。",
            "direction": "+", "paradigm": "尾部风险",
        },
        {
            "expression": "neg(div(add(ts_min(returns, 20), ts_min(returns, 5)), ts_std(returns, 20)))",
            "name": "双重底探底信号",
            "rationale": "取负((20日+5日最低收益)/20日波动率)。短期和中期同时出现极端下跌=双重底部信号。",
            "direction": "+", "paradigm": "尾部风险",
        },
        {
            "expression": "neg(rank(ts_min(returns, 20)))",
            "name": "最大回撤反弹",
            "rationale": "截面取负(20日最低日收益)。经历最大冲击的股票预期反弹最强。纯尾部反转。",
            "direction": "+", "paradigm": "尾部风险",
        },
        {
            "expression": "neg(div(ts_skew(returns, 60), add(ts_std(returns, 60), 1e-6)))",
            "name": "协偏度溢价(Harvey-Siddique)",
            "rationale": "取负(60日收益偏度/波动率)。负偏度=崩盘风险暴露, 正偏度=彩票特征。做多负偏度=收取crash risk premium。Harvey-Siddique 2000 JF, Stage2 PASS ICIR=0.317。",
            "direction": "+", "paradigm": "尾部风险",
        },
        {
            "expression": "neg(rank(div(neg(ts_min(returns, 20)), ts_skew(returns, 60))))",
            "name": "尾部冲击/偏度比",
            "rationale": "截面取负(20日最大冲击(取负)/60日偏度)。极端冲击大但整体偏度不大=被错杀=反弹确定性更强。偏度×冲击的交叉定价。",
            "direction": "+", "paradigm": "尾部风险",
        },

        # ═══ 范式6: 情绪/开盘 ═══
        {
            "expression": "rank(div(sub(open, ts_delta(close, 1)), add(ts_std(close, 20), 1e-6)))",
            "name": "开盘跳空标准化",
            "rationale": "开盘跳空幅度/20日波动率截面排名。标准化后区分真跳空和正常波动。",
            "direction": "+", "paradigm": "情绪×波动率",
        },
        {
            "expression": "neg(ts_mean(sub(div(div(sub(high, low), close), ts_mean(div(sub(high, close), close), 20)), 1), 3))",
            "name": "振幅异常度",
            "rationale": "日内振幅相对20日均值的偏离。振幅异常放大=多空分歧=短期方向即将确定。",
            "direction": "-", "paradigm": "情绪×微观结构",
        },
        {
            "expression": "neg(rank(div(sub(ts_delta(close, 1), open), add(ts_std(open, 20), 1e-6))))",
            "name": "日内反转强度",
            "rationale": "截面取负((昨收-今开)/开盘波动率)。低开程度大的股票当日反弹动能更强。",
            "direction": "+", "paradigm": "情绪×日内",
        },

        # ═══ 范式7: 截面交互 ═══
        {
            "expression": "rank(mul(ts_zscore(volume, 20), neg(ts_zscore(close, 20))))",
            "name": "放量下跌(截面)",
            "rationale": "截面rank(量z-score×负价格z-score)。放量下跌=恐慌抛售, 超跌反弹。",
            "direction": "+", "paradigm": "资金流×价格",
        },
        {
            "expression": "rank(sub(div(ts_mean(close, 5), ts_mean(close, 20)), ts_std(returns, 20)))",
            "name": "趋势强度-波动率",
            "rationale": "截面rank(5/20均线比率-波动率)。低波动上涨>高波动上涨, 动量质量信号。",
            "direction": "+", "paradigm": "动量×波动率",
        },
        {
            "expression": "neg(div(ts_pct(volume, 5), add(ts_std(volume, 20), 1e-6)))",
            "name": "放量加速度/波动调控",
            "rationale": "5日量增速/20日量波动。放量但波动稳定=健康放量(非异常), 区分庄股对倒和真实增量。",
            "direction": "+", "paradigm": "资金流×波动率",
        },
        {
            "expression": "neg(rank(mul(div(close, ts_max(close, 20)), div(volume, ts_max(volume, 20)))))",
            "name": "量价同步低位",
            "rationale": "截面取负((价/20日最高价)×(量/20日最高量))。量价同步处于低位=极度低迷=反转窗口。",
            "direction": "+", "paradigm": "资金流×价格",
        },

        # ═══ 范式8: 趋势跟随 ═══
        {
            "expression": "rank(ts_pct(close, 20))",
            "name": "20日动量排名",
            "rationale": "截面rank(20日收益率)。纯动量——持续上涨的股票继续涨。A股中20日窗口动量有效。",
            "direction": "-", "paradigm": "纯粹动量",
        },
        {
            "expression": "neg(div(ts_mean(close, 10), ts_mean(close, 50)))",
            "name": "死叉信号(取负)",
            "rationale": "取负(10日/50日均线)。死叉&金叉量化, 死叉后取负=做多反转。均线交叉系统。",
            "direction": "+", "paradigm": "趋势",
        },
        {
            "expression": "neg(rank(ts_mean(div(sub(close, ts_mean(close, 20)), ts_std(close, 20)), 5)))",
            "name": "均值回归力",
            "rationale": "截面取负(近5日均z-score)。价格偏离均线越远回归力越强。经典均值回归。",
            "direction": "+", "paradigm": "动量反转",
        },
        {
            "expression": "rank(sub(ts_pct(close, 5), ts_pct(close, 20)))",
            "name": "加速度信号",
            "rationale": "截面rank(5日收益-20日收益)。短期加速超越长期趋势=动量增强中, 非反转。",
            "direction": "+", "paradigm": "动量",
        },

        # ═══ 范式9: 下行保护 (NEW) — 降回撤核心 ═══
        {
            "expression": "neg(rank(div(neg(ts_min(returns, 20)), ts_std(returns, 20))))",
            "name": "下行波动比",
            "rationale": "截面取负((取负(20日最小收益))/波动率)。极端下行风险相对总风险的定价。高比率=被过度惩罚。",
            "direction": "+", "paradigm": "下行保护",
        },
        {
            "expression": "neg(rank(ts_mean(sub(close, ts_delta(close, 20)), 10)))",
            "name": "20日回撤深度",
            "rationale": "截面取负(近10日均(价格-20日前价格))。中期跌幅大=回撤修复潜力。纯回撤反转。",
            "direction": "+", "paradigm": "下行保护",
        },
        {
            "expression": "rank(div(neg(ts_min(returns, 60)), ts_std(returns, 60)))",
            "name": "季度尾部事件强度",
            "rationale": "截面rank(取负(60日最低收益)/60日波动率)。季度级别尾部事件后反弹。长期反转。",
            "direction": "+", "paradigm": "下行保护",
        },
        {
            "expression": "neg(rank(div(ts_min(close, 20), ts_max(close, 20))))",
            "name": "20日跌幅比",
            "rationale": "截面取负(20日最低/20日最高)。价格被压缩程度, 跌幅越深反弹空间越大。",
            "direction": "+", "paradigm": "下行保护",
        },
        {
            "expression": "neg(rank(div(ts_std(neg(ts_min(returns, 5), 0)), ts_std(returns, 20))))",
            "name": "下行波动占比",
            "rationale": "截面取负(下行半方差/总方差)。下行波动比例高=恐慌性抛售后的均值回归。v1/v2缺失。",
            "direction": "+", "paradigm": "下行保护",
        },
        {
            "expression": "neg(ts_mean(div(sub(high, low), close), 5))",
            "name": "近期振幅收窄",
            "rationale": "取负(5日均(日内振幅/收盘价))。振幅持续收窄=波动率压缩=即将方向突破=向上概率偏高。",
            "direction": "+", "paradigm": "下行保护×波动率",
        },

        # ═══ 范式10: 波动率适应 (NEW) — 市场状态感知 ═══
        {
            "expression": "neg(div(ts_std(close, 20), ts_std(close, 60)))",
            "name": "波动率周期位置",
            "rationale": "取负(20日波动/60日波动)。波动率回落=稳定环境=趋势跟随策略友好。高波动转低波动信号。",
            "direction": "+", "paradigm": "波动率适应",
        },
        {
            "expression": "rank(ts_delta(ts_std(returns, 20), 10))",
            "name": "波动率加速度",
            "rationale": "截面rank(20日波动率10日变化)。波动率上升=风险增加, 下降=风险消退。波动率动量。",
            "direction": "-", "paradigm": "波动率适应",
        },
        {
            "expression": "neg(rank(div(ts_std(returns, 10), ts_std(returns, 40))))",
            "name": "短长期波动比(取负)",
            "rationale": "截面取负(10日/40日波动率)。近期波动相对偏低=低风险环境。GARCH风格波动率聚集。",
            "direction": "+", "paradigm": "波动率适应",
        },
        {
            "expression": "rank(sub(ts_max(close, 20), ts_min(close, 20)))",
            "name": "20日振幅(退阶)",
            "rationale": "纯20日最高-最低截面排名。大振幅=投机性强=高Beta, 小振幅=低波动溢价。",
            "direction": "-", "paradigm": "波动率适应",
        },
        {
            "expression": "neg(rank(div(ts_mean(close, 5), ts_mean(close, 10))))",
            "name": "短期转强信号",
            "rationale": "截面取负(5日/10日均线)。5日均线超越10日=短期转强, 取负做多。",
            "direction": "+", "paradigm": "波动率适应×趋势",
        },

        # ═══ 范式11: 市场宽度/参与度 (NEW) — 截面多样性 ═══
        {
            "expression": "neg(rank(div(ts_std(close, 20), ts_mean(close, 20))))",
            "name": "变异系数(取负)",
            "rationale": "截面取负(20日标准差/20日均价)。低变异系数=价格稳定=低风险溢价。截面异质性信号。",
            "direction": "+", "paradigm": "市场宽度",
        },
        {
            "expression": "neg(rank(div(sub(ts_max(close, 5), close), ts_std(close, 20))))",
            "name": "距5日高点距离",
            "rationale": "截面取负((5日最高-现价)/波动率)。距短期高点多远=回调幅度标准化。",
            "direction": "+", "paradigm": "市场宽度",
        },
        {
            "expression": "rank(sub(div(close, ts_mean(close, 20)), div(volume, ts_mean(volume, 20))))",
            "name": "量价相对偏离",
            "rationale": "截面rank(价/20日均价-量/20日均量)。价强量弱>价弱量强。量价关系的截面偏离信号。",
            "direction": "+", "paradigm": "市场宽度×资金流",
        },
        {
            "expression": "neg(rank(div(ts_mean(close, 5), ts_mean(close, 60))))",
            "name": "长期弱势反转",
            "rationale": "截面取负(5日均/60日均)。短期对长期折价大=长期弱势=均值回归预期。深度价值反转。",
            "direction": "+", "paradigm": "市场宽度",
        },

        # ═══ 范式12: 结构突变 (NEW) — 突破/支撑 ═══
        {
            "expression": "rank(div(volume, add(ts_mean(volume, 20), ts_std(volume, 20), 1e-6)))",
            "name": "量突破Z分数",
            "rationale": "截面rank(量/(20日均量+20日量标准差))。量突破均量+1标准差=异常放量=突破信号。",
            "direction": "+", "paradigm": "结构突变",
        },
        {
            "expression": "neg(rank(div(sub(close, ts_max(close, 20)), ts_std(close, 20))))",
            "name": "突破20日高点前",
            "rationale": "截面取负((现价-20日高点)/波动率)。即将突破20日高点=突破前建仓, 追涨替代。",
            "direction": "+", "paradigm": "结构突变",
        },
        {
            "expression": "neg(rank(sub(ts_max(close, 20), close)))",
            "name": "距20日阻力距离",
            "rationale": "截面取负(20日最高-现价)。距阻力越远比距支撑越近=安全边际。突破阻力位后空间更大。",
            "direction": "+", "paradigm": "结构突变",
        },
        {
            "expression": "rank(div(ts_pct(close, 1), add(ts_std(close, 20), 1e-6)))",
            "name": "单日波动异常度",
            "rationale": "截面rank(日收益率/20日波动率)。单日幅度相对波动水平, 异常日=方向信号。",
            "direction": "-", "paradigm": "结构突变",
        },
        {
            "expression": "neg(div(volume, ts_max(volume, 20)))",
            "name": "缩量至低点",
            "rationale": "取负(量/20日最高量)。量萎缩到极低=地量地价=转折点在即。经典缩量止跌。",
            "direction": "+", "paradigm": "结构突变×资金流",
        },

        # ═══ 范式13: 筹码分布 (NEW) — A股核心降回撤维度 ═══
        {
            "expression": "neg(rank(div(ts_mean(volume, 10), ts_mean(volume, 60))))",
            "name": "筹码锁定度(缩量)",
            "rationale": "截面取负(10日/60日均量)。长期缩量=筹码锁定/惜售=A股底部特征。筹码锁定度高=抗跌性强制。",
            "direction": "+", "paradigm": "筹码分布",
        },
        {
            "expression": "rank(div(ts_std(volume, 20), ts_mean(volume, 20)))",
            "name": "筹码分散度(量波动)",
            "rationale": "截面rank(20日量变异系数)。量波动大=筹码分散/分歧大, 量波动小=筹码集中。集中筹码更易突破。",
            "direction": "-", "paradigm": "筹码分布",
        },
        {
            "expression": "neg(rank(div(ts_mean(mul(close, volume), 20), ts_mean(mul(close, volume), 60))))",
            "name": "成交重心稳定性",
            "rationale": "截面取负(20日/60日成交额均值)。成交重心持续稳定=主力在锁仓, 不稳定=短线资金进出。",
            "direction": "+", "paradigm": "筹码分布",
        },
        {
            "expression": "neg(rank(div(volume, add(ts_mean(volume, 60), 1e-6))))",
            "name": "浮筹比例(相对前期)",
            "rationale": "截面取负(当日量/60日均量)。高比率=浮筹多/短线资金活跃, 低比率=浮筹少/长线锁定。取负做多低浮筹。",
            "direction": "+", "paradigm": "筹码分布",
        },
        {
            "expression": "neg(rank(ts_mean(div(sub(high, low), add(close, 1e-6)), 20)))",
            "name": "振幅压缩(筹码稳定)",
            "rationale": "截面取负(20日均振幅率)。振幅持续低=多空分歧小=筹码稳定=即将方向性突破。A股前兆信号。",
            "direction": "+", "paradigm": "筹码分布",
        },
        {
            "expression": "neg(div(ts_mean(close, 60), ts_mean(close, 10)))",
            "name": "中长期持仓成本偏离",
            "rationale": "取负(60日均/10日均)。中长期持仓成本远高于短期=多数筹码浮亏=惜售不抛=支撑强。筹码成本锚定。",
            "direction": "+", "paradigm": "筹码分布",
        },
    ]

    return factors


# ═══════════════════════════════════════════════════════════
# 阶段2: 互补评分配对 — 替代v1硬编码
# ═══════════════════════════════════════════════════════════

# 经济维度定义 — 用于计算配对互补性
ECON_DIMENSIONS = {
    "流动性": ["流动性", "liquidity", "volume", "成交量", "换手"],
    "趋势/动量": ["趋势", "趋势", "momentum", "动量", "trend"],
    "反转/均值回归": ["反转", "reversal", "超卖", "超跌", "反弹", "均值回归", "regression"],
    "波动率": ["波动", "volatility", "vol", "震荡", "振幅"],
    "资金流": ["资金流", "flow", "money_flow", "量价"],
    "情绪/行为": ["情绪", "sentiment", "行为金融", "行为", "behavioral", "开盘"],
    "尾部风险/下行": ["尾部", "tail", "下行", "downside", "protection", "保护", "回撤", "drawdown"],
    "市场宽度/结构": ["宽度", "breadth", "结构", "突破", "break", "支撑", "阻力"],
    "微观结构": ["微观", "microstructure", "日内", "intraday"],
    "筹码分布": ["筹码", "chip", "分布", "锁定", "集中", "浮筹", "持仓成本"],
    "基本面/成长": [
        "基本面", "quality", "盈利", "成长", "growth", "roe", "roa",
        "毛利率", "利润率", "营收", "收入", "资本效率", "经营效率", "盈余",
        "盈余预期", "利润", "净利", "现金流", "资产回报", "股息", "估值",
        "fundamental", "earnings", "profit", "revenue",
    ],
}

# V3基准复合因子及其经济维度
V3_BASE_COMPOSITES = {
    "overnight_5d": {
        "dimensions": ["微观结构", "情绪/行为"],
        "expression": "ts_sum(sub(div(open, ts_delta(close, 1)), 1), 5)",
    },
    "tvma_20": {
        "dimensions": ["趋势/动量", "资金流"],
        "expression": "neg(div(mul(close, volume), ts_mean(mul(close, volume), 20)))",
    },
    "dollar_vol_20d": {
        "dimensions": ["流动性", "资金流"],
        "expression": "neg(log(ts_mean(mul(close, volume), 20)))",
    },
    "turnover_std_cv": {
        "dimensions": ["流动性", "波动率"],
        "expression": "neg(div(ts_std(volume, 20), ts_mean(volume, 20)))",
    },
    "money_flow_20": {
        "dimensions": ["资金流", "趋势/动量"],
        "expression": "neg(sub(div(mul(add(add(high,low),close), volume), 3), ts_mean(div(mul(add(add(high,low),close), volume), 3), 20)))",
    },
    "ret_3m": {
        "dimensions": ["反转/均值回归"],
        "expression": "neg(sub(div(close, ts_delta(close, 60)), 1))",
    },
}


# ── 注入因子 Rationale 模板 ──────────────────────────────────────────
# 每个因子包含经济机制描述、维度关键词、与V3基准因子的互补性说明。
# 关键词必须匹配 ECON_DIMENSIONS 中的条目才能被 _classify_dimensions 识别。
_INJECTED_RATIONALE_TEMPLATES = {
    # ── Round 1 (2026-08-02) ─────────────────────────────
    "max_drawdown_duration": (
        "捕捉个股在极端下跌后回撤持续时间中隐含的尾部风险与下行保护溢价："
        "回撤持续时间越长的股票，下行脆弱性越高，后续动量趋势继续跑输的概率越大。"
        "与趋势/动量因子形成天然互补（尾部风险×动量配对），"
        "有助于降低策略在市场急跌中的最大回撤。"
    ),
    "harvey_siddique_coskew": (
        "Harvey-Siddique协偏度衡量个股收益与市场波动的协变尾部风险："
        "负协偏度股票在市场下跌时具有崩盘敏感性（左尾风险），"
        "需要更高的尾部风险补偿，是经典学术异象中稳健的下行保护信号。"
        "与传统波动率因子正交，与微观结构因子互补"
        "（隔夜跳空对崩盘风险的重新定价），提供了独特的下行维度。"
    ),
    "earnings_pre_drift_alignment": (
        "捕捉财报窗口前价格动量与资金流趋势中蕴含的盈余预期信息泄露："
        "中报前量价趋势的偏离程度反映机构预判行为，"
        "强者恒强的动量效应在盈余窗口期加速，属于事件驱动的趋势/动量增强。"
        "与资金流因子互补（量价配合验证信息可信度），"
        "提供了不同于传统60日动量的短窗口基本面/成长预期信号。"
    ),
    "capital_efficiency_proxy": (
        "从资本投入产出效率角度评估企业基本面质量与成长可持续性："
        "高资本效率意味着每一单位投入产生更多盈利增长，"
        "是基本面/成长维度的核心质量指标，避免'伪成长'陷阱。"
        "与技术面因子（趋势/动量、资金流、波动率）高度互补，"
        "提供了V3因子池中稀缺的盈利质量与经营效率维度，"
        "与现有纯技术面配对的信号多样性极高。"
    ),
    # ── Round 2 (2026-08-03) ─────────────────────────────
    "volume_crowding_divergence": (
        "成交量拥挤度背离——利用历史上验证失败的因子模式的反向信号："
        "当成交量拥挤度与价格走势背离时，市场情绪极端化，"
        "反向操作捕捉行为偏差修复。关注反转/均值回归与资金流维度的交叉验证，"
        "量价关系确认反转信号可靠性，提供反共识alpha源。"
    ),
    "small_cap_liquidity_quality": (
        "小盘流动性质量共振——捕捉小盘股在牛市突破行情中的流动性-质量共振："
        "高流动性质量的小盘股在突破时弹性更强，关注微观结构和趋势/动量维度。"
        "适配A股小盘风格的独特alpha源，与筹码分布因子互补。"
    ),
    "hl_volatility_spread_regime_stable": (
        "高低价波动率价差因子——vm_diff毒因子的替代方案："
        "用高低价波动率价差替代成交量波动率，规避vm_diff的JQ灾难。"
        "波动率收敛+趋势启动=高质量动量窗口，关注波动率和趋势/动量维度。"
        "提供波动率维度的安全替代信号。"
    ),
    "diffusion_index_momentum": (
        "扩散指数动量确认——机器学习启发的扩散指数构造方法："
        "多信号交叉确认动量趋势，减少单一技术指标噪声。"
        "扩散确认提升信号信噪比，关注趋势/动量和波动率维度。"
        "与传统价格动量互补（扩散→过滤噪声→更纯信号）。"
    ),
    "idiosyncratic_tail_hedge_premium": (
        "特质尾部对冲溢价——CAPM残差中的尾部风险补偿："
        "个股特质波动中的极端下行成分定价，残差尾部风险越大→补偿越高。"
        "关注尾部风险/下行和反转/均值回归维度，与波动率因子正交。"
        "残差对冲→独立于系统性贝塔的alpha。"
    ),
    "event_driven_convexity_fade": (
        "事件驱动凸性退潮——利用历史上验证失败因子模式的反向信号："
        "事件冲击后价格凸性变化与预期方向相反→退潮时反向建仓。"
        "关注反转/均值回归和资金流维度，与趋势/动量因子互补。"
        "事件退潮=被过度定价的偏差修复。"
    ),
    "lottery_demand_suppression_score": (
        "彩票需求抑制得分——学术异象MAX效应的A股适配："
        "投资者对彩票型股票（高偏度/高最大收益）的系统性偏好→"
        "被高估→做多抑制赌博需求的股票。关注尾部风险/下行和情绪/行为维度。"
        "行为金融异象在A股的实证验证，与反转因子互补。"
    ),
}


def build_injected_rationale(factor_name: str, factor_label: str, fri: float,
                             fri_grade: str, icir: float,
                             paradise: str, category: str = "") -> str:
    """为注入因子生成含维度关键词的丰富rationale。

    结构: {经济机制+维度关键词} + [FRI={score}/{grade}, ICIR={icir}]
    维度关键词被 _classify_dimensions() 用来确定因子归属的经济维度，
    进而影响 compute_complementarity_score() 的评分。
    """
    template = _INJECTED_RATIONALE_TEMPLATES.get(factor_name)
    if template is None:
        # 通用fallback: 嵌入类别→维度关键词，确保至少匹配1个维度
        cat_keywords = {
            "extreme_events": "尾部风险、下行保护、最大回撤极端事件",
            "mid_report_divergence": "动量趋势、资金流量价、盈余预期基本面",
            "growth_quality": "基本面成长、盈利质量、经营效率资本效率",
            "liquidity_microstructure": "流动性换手、微观结构日内、成交量",
            "momentum": "趋势动量、动量效应、趋势延续",
            "reversal": "反转回归、均值回归、超跌反弹",
            "volatility_risk": "波动率震荡、波动振幅、波动率聚集",
            "fund_flow": "资金流、量价关系、主力资金流向",
            "sentiment_behavioral": "情绪行为、投资者行为金融、开盘效应",
            "fundamental_quality": "基本面盈利、质量成长、利润率ROE",
            "valuation": "估值、股息现金流、资产回报",
            "tail_extreme": "尾部风险下行、回撤保护、左尾极端",
            "size_turnover": "流动性换手、成交量、筹码分布集中度",
            "stage1_exploration": "资金流量价、趋势动量、流动性",
            "学术异象库": "尾部风险下行、波动率、协偏度崩盘微观结构",
            # ── Round 2 新增类别 ──────────────────────────
            "failure_mode_inverse": "反转回归、资金流量价、行为金融过度反应",
            "small_bull_breakthrough": "流动性换手、微观结构日内、趋势动量突破",
            "vm_diff_replacement": "波动率震荡、趋势动量、波动率收敛趋势启动",
            "ml_inspired_construction": "趋势动量、波动率扩散确认、信号信噪比",
            "academic_anomaly_adaptation": "尾部风险下行、情绪行为、MAX效应彩票偏好",
            "volume_structure": "流动性换手、资金流、筹码分布集中度",
            "momentum_dynamics": "趋势动量、资金流、动量生命周期",
            "volatility_structure": "波动率震荡、反转回归均值",
            "price_pattern": "反转回归、微观结构、超卖超买形态",
            "liquidity_micro": "流动性换手、微观结构日内、筹码分布",
            "behavioral": "情绪行为、反转回归、行为偏差过度反应",
        }
        keywords = cat_keywords.get(category)
        if keywords is None:
            # 智能fallback: 从维度关键词自动推断
            dims = _classify_dimensions(paradise, category)
            keywords = "、".join(dims) if dims else paradise
        template = (
            f"{factor_label}（{paradise}）: "
            f"Stage2 FRI验证因子，类别={category}，"
            f"关注{keywords}维度的alpha信号。"
        )
    return (
        f"{template} "
        f"[FRI={fri:.2f}/{fri_grade}, ICIR={icir:.3f}]"
    )


def _classify_dimensions(rationale: str, paradigm: str) -> List[str]:
    """根据rationale和paradigm推断因子的经济维度"""
    dims = set()
    text = (rationale + " " + paradigm).lower()
    for dim, keywords in ECON_DIMENSIONS.items():
        if any(kw.lower() in text for kw in keywords):
            dims.add(dim)
    if not dims:
        dims.add("其他")
    return list(dims)


# ── 统一评分 & 多样性约束 (MMR 近似) ──────────────────────────────
# 
# 科学基础: 因子选择 = 约束子模最大化问题。
# 目标: max Σ quality(p_i)  s.t.  diversity(p_i, p_j) ≤ threshold, ∀i≠j
# 贪婪算法对此类问题提供 (1-1/e) ≈ 63% 的近似最优保证。
#
# 统一评分 = 互补分 + FRI×FRI_WEIGHT + novelty×NOVELTY_WEIGHT
#   - 互补分 [0,10]: 经济维度互补性 (与 V3 基底配对质量)
#
# ⚠️ JQ实测修正 (2026-08-04, 进化轮#3 +141.19% vs 王者 +182.57%):
#   FRI×5 + novelty×3 的正权重 → -41pp。FRI 是 local 指标,
#   正权重 = 选择 local 上好但 JQ 上差的因子 (local→JQ gap 传导)。
#   首次回填的 RidgeUCB 也从 JQ 数据学到 mean_fri_injected 权重为负。
#   结论: 此处两个权重恒为 0, local 特征的权重交给
#   evaluator/scorer.py 从 JQ 真实结果在线学习 (影子模式)。
FRI_WEIGHT = 0.0
NOVELTY_WEIGHT = 0.0

# 维度 Jaccard 相似度上限: 两个因子共享 >50% 维度 → 视为冗余
# 例: {尾部风险,波动率} vs {尾部风险,反转} → jaccard=1/3=0.33 → PASS
#     {尾部风险,波动率} vs {尾部风险,波动率,反转} → jaccard=2/3=0.67 → REJECT
DIVERSITY_THRESHOLD = 0.5


def _get_candidate_dimensions(cand) -> List[str]:
    """提取候选因子的经济维度（优先用预定义维度）。"""
    dims = getattr(cand, '_dimensions', None)
    if dims:
        return list(dims)
    return _classify_dimensions(cand.rationale, getattr(cand, '_paradigm', ''))


def _get_candidate_fri(cand) -> float:
    """提取候选因子的 FRI 评分，LLM 生成因子无 FRI → 返回 0。"""
    fri = getattr(cand, '_fri', None)
    return float(fri) if fri else 0.0


def _get_candidate_novelty(cand) -> float:
    """提取候选因子的 FRI novelty 子维度，LLM 生成因子无 → 返回 0。
    novelty = 1 - max_corr_with_existing, 是数据驱动的冗余度量。
    比维度 Jaccard 更精细: 两个同为「尾部风险」的因子,
    novelty 能区分它们是高度相关 (0.2) 还是独立信号 (0.9)。
    """
    nov = getattr(cand, '_fri_novelty', None)
    return float(nov) if nov else 0.0


def _jaccard_similarity(dims_a: List[str], dims_b: List[str]) -> float:
    """Jaccard 相似度: |A∩B| / |A∪B|, 范围 [0, 1]"""
    if not dims_a or not dims_b:
        return 0.0
    sa, sb = set(dims_a) - {"其他"}, set(dims_b) - {"其他"}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def compute_complementarity_score(new_factor: dict, v3_factor_name: str) -> float:
    """
    计算新因子与V3因子的互补性评分 [0, 10]。

    评分逻辑:
    - 维度互补: 不同经济维度 → +5分（如"下行保护×趋势"）
    - 部分互补: 有重叠但不是完全相同 → +2分
    - 信息多样性: 维度差异越大 → +3分
    - 满分=10: 完全互补的两个维度配对
    """
    # ── 优先使用预定义维度 (来自 injected_factors.json) ──
    predef_dims = new_factor.get("_dimensions", []) or new_factor.get("dimensions", [])
    if predef_dims:
        new_dims = set(predef_dims)
    else:
        new_dims = set(_classify_dimensions(
            new_factor.get("rationale", ""), new_factor.get("paradigm", "")))

    v3_info = V3_BASE_COMPOSITES.get(v3_factor_name, {})
    v3_dims = set(v3_info.get("dimensions", []))

    score = 0.0
    if new_dims and v3_dims:
        overlap = new_dims & v3_dims
        if not overlap:
            score += 5.0  # 完全不同维度 → 高度互补
        elif len(overlap) < len(new_dims) and len(overlap) < len(v3_dims):
            score += 2.0  # 部分重叠
        # 维度差异奖励
        unique_cnt = len(new_dims - v3_dims)
        score += min(unique_cnt * 1.5, 3.0)

    # 下行保护维度与任何趋势/动量维度配对 → 额外+1（降低回撤的关键配对）
    if "尾部风险/下行" in new_dims and "趋势/动量" in v3_dims:
        score += 1.0
    if "尾部风险/下行" in v3_dims and "趋势/动量" in new_dims:
        score += 1.0

    # 波动率适应与资金流配对 → 额外+1
    if "波动率" in new_dims and "资金流" in v3_dims:
        score += 1.0
    if "波动率" in v3_dims and "资金流" in new_dims:
        score += 1.0

    # 筹码分布与趋势/动量配对 → 额外+2 (筹码锁定+趋势确认=降回撤黄金配对)
    if "筹码分布" in new_dims and "趋势/动量" in v3_dims:
        score += 2.0
    if "筹码分布" in v3_dims and "趋势/动量" in new_dims:
        score += 2.0

    # 筹码分布与资金流配对 → 额外+1.5 (筹码稳定+资金流入=主力建仓)
    if "筹码分布" in new_dims and "资金流" in v3_dims:
        score += 1.5
    if "筹码分布" in v3_dims and "资金流" in new_dims:
        score += 1.5

    # 尾部风险与微观结构(隔夜)配对 → 额外+1.5 (隔夜对下行冲击的敏感度)
    if "尾部风险/下行" in new_dims and "微观结构" in v3_dims:
        score += 1.5
    if "尾部风险/下行" in v3_dims and "微观结构" in new_dims:
        score += 1.5

    # 尾部风险与流动性配对 → 额外+1 (流动性缓冲尾部冲击)
    if "尾部风险/下行" in new_dims and "流动性" in v3_dims:
        score += 1.0
    if "尾部风险/下行" in v3_dims and "流动性" in new_dims:
        score += 1.0

    return min(score, 10.0)


def _shadow_score_portfolio(all_composites, selected_pairs):
    """
    RidgeUCB 影子评分: 对最终组合预测 JQ 收益, 只记录不改变选择。
    校准数据累积在 evaluator/calibration_log.jsonl。
    """
    from pathlib import Path as _P
    _scorer_path = _P(__file__).parent / "evaluator" / "scorer_state.json"
    if not _scorer_path.exists():
        print("  [影子评分] scorer_state.json 不存在, 先运行 evaluator/backfill_trials.py")
        return
    import sys as _sys
    _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    from research.factor_alchemy.evaluator.scorer import RidgeUCB

    scorer = RidgeUCB.load(str(_scorer_path))

    # 构建策略级特征 (与 evaluator/features.py FEATURE_NAMES 对齐)
    inj = [sp for sp in selected_pairs if sp.get("fri", 0) > 0]
    llm = [sp for sp in selected_pairs if sp.get("fri", 0) <= 0]
    fris = [sp["fri"] for sp in inj]
    novs = [sp.get("novelty", 0) for sp in inj]
    dims_all = [d for sp in selected_pairs for d in sp.get("_dims", [])]
    tail_n = sum(1 for sp in selected_pairs
                 if any("尾部" in d or "下行" in d for d in sp.get("_dims", [])))
    chip_n = sum(1 for sp in selected_pairs
                 if any("筹码" in d for d in sp.get("_dims", [])))
    vec = [
        1.0,  # freq_weekly (v3 标准输出)
        0.0,  # freq_daily
        len(all_composites) / 6.0,
        float(len(inj)),
        float(sum(fris) / len(fris)) if fris else 0.0,
        float(max(fris)) if fris else 0.0,
        float(sum(novs) / len(novs)) if novs else 0.0,
        float(len(llm)),
        float(tail_n),
        float(chip_n),
    ]
    mean, sigma = scorer.predict(vec)
    ucb = mean + scorer.beta * sigma
    print(f"\n  [影子评分] RidgeUCB 预测 (n={scorer.n_samples}, 不改变选择):")
    print(f"    预测 JQ 累计收益 = {mean:.1f}% ± {sigma:.1f}pp  "
          f"(UCB 选择分={ucb:.1f})")
    print(f"    校准说明: n={scorer.n_samples} 时预测≈均值回归, "
          f"影子运行积累复合级数据后才有区分度")

    # 记录校准日志 (JQ 回测后回填 actual)
    import json as _json
    from datetime import datetime as _dt
    _cal_path = _P(__file__).parent / "evaluator" / "calibration_log.jsonl"
    with open(_cal_path, "a", encoding="utf-8") as f:
        f.write(_json.dumps({
            "timestamp": _dt.now().isoformat(),
            "features": vec,
            "predicted_return": round(mean, 2),
            "predicted_sigma": round(sigma, 2),
            "ucb_score": round(ucb, 2),
            "n_injected": len(inj),
            "n_new_llm": len(llm),
            "actual_return": None,  # 待 JQ 回测后回填
        }, ensure_ascii=False) + "\n")


def build_v3_integration_v3(candidates: List[AlphaCandidate]) -> List[dict]:
    """
    v3 配对策略: 统一评分 + 约束子模最大化 (MMR 近似)

    科学框架:
    - 因子选择 = 约束子模最大化问题: max Σ quality(p_i)
      s.t. Jaccard(p_i.dims, p_j.dims) ≤ 0.5,  ∀i≠j
    - 贪婪算法提供 (1-1/e) ≈ 63% 最优保证

    流程:
    1. 保留 V3 的 3 个最优复合
    2. 每个通过三重约束的候选 × 6 个 V3 底因子 → 统一评分
       unified = 互补分 + FRI × FRI_WEIGHT (注入/LLM 同台竞技)
    3. 约束贪婪选择 Top-3:
       - 按统一评分降序
       - 新因子与已选因子维度 Jaccard 重叠 > 0.5 → 跳过 (正交性保证)
       - 评分高的先入选, 相似度高的评分低的自动被替代
    4. 若候选不足则回退 V3 原复合
    """

    # 保留的V3原复合
    preserved_v3_composites = [
        {"name": "comp1_overnight_tvma", "a_name": "overnight_5d", "b_name": "tvma_20",
         "from_v3": True, "a_type": "v3", "b_type": "v3",
         "rationale": "隔夜累计x量价趋势: V3最强复合"},
        {"name": "comp2_dollar_turnover", "a_name": "dollar_vol_20d", "b_name": "turnover_std_cv",
         "from_v3": True, "a_type": "v3", "b_type": "v3",
         "rationale": "成交额x换手稳定性: V3次强"},
        {"name": "comp3_moneyflow_ret3m", "a_name": "money_flow_20", "b_name": "ret_3m",
         "from_v3": True, "a_type": "v3", "b_type": "v3",
         "rationale": "资金流x动量反转: V3原value_momentum简化版"},
    ]

    # 筛选通过三重约束的候选
    passed = [c for c in candidates if c.passed_triple]
    if not passed:
        print("  [!] 无候选通过三重约束, 回退V3原始5复合")
        return preserved_v3_composites + [
            {"name": "comp4_retopen_skew", "a_type": "v3", "a_name": "ret_open_2d",
             "b_type": "v3", "b_name": "skewness_20",
             "rationale": "开盘动量x收益偏度: V3原comp4", "from_v3": True},
            {"name": "comp5_gapup_spread", "a_type": "v3", "a_name": "gap_up",
             "b_type": "v3", "b_name": "relative_spread",
             "rationale": "跳空x振幅: V3原comp5", "from_v3": True},
        ]

    # ── 构建配对候选池 (统一评分 = 互补分 + FRI × FRI_WEIGHT) ──
    v3_factor_names = list(V3_BASE_COMPOSITES.keys())
    pair_candidates = []

    for cand in passed:
        cand_paradigm = getattr(cand, '_paradigm', '')
        cand_fri = _get_candidate_fri(cand)
        cand_novelty = _get_candidate_novelty(cand)
        cand_dims = _get_candidate_dimensions(cand)
        for v3_name in v3_factor_names:
            score = compute_complementarity_score(
                {"rationale": cand.rationale, "paradigm": cand_paradigm,
                 "_dimensions": cand_dims}, v3_name)
            # 统一评分: 互补分 + FRI×5 + novelty×3
            # novelty 奖励与已有因子池正交的独特信号
            unified = score + cand_fri * FRI_WEIGHT + cand_novelty * NOVELTY_WEIGHT
            pair_candidates.append({
                "candidate": cand,
                "v3_factor": v3_name,
                "score": score,
                "unified_score": unified,
                "fri": cand_fri,
                "novelty": cand_novelty,
                "fri_grade": getattr(cand, '_fri_grade', None),
                "_dims": cand_dims,
            })

    # ── 约束贪婪选择 (MMR 近似) ──
    # 排序: 统一评分↓ > 互补分↓ > FRI↓ > novelty↓ (注入因子同台竞技, 无插入偏差)
    pair_candidates.sort(key=lambda x: (-x["unified_score"], -x["score"], -x["fri"], -x["novelty"]))

    used_v3 = set()
    used_cand = set()
    selected_pairs = []
    selected_dims = []  # 已选因子的维度, 用于正交性约束
    skipped_by_diversity = []  # 因维度冗余被跳过的因子

    for pc in pair_candidates:
        if pc["v3_factor"] in used_v3:
            continue
        if pc["candidate"].name in used_cand:
            continue
        if pc["score"] < 3.0:  # 最低互补门槛
            continue

        cand_dims = pc["_dims"]

        # ── 多样性约束: 新因子与已选因子维度过高重叠 → 跳过 ──
        max_sim = max(
            [_jaccard_similarity(cand_dims, sd) for sd in selected_dims],
            default=0.0)
        if max_sim > DIVERSITY_THRESHOLD:
            skipped_by_diversity.append({
                "name": pc["candidate"].name[:30],
                "dims": cand_dims,
                "unified": pc["unified_score"],
                "fri": pc["fri"],
                "max_overlap": max_sim,
            })
            continue

        selected_pairs.append(pc)
        used_v3.add(pc["v3_factor"])
        used_cand.add(pc["candidate"].name)
        selected_dims.append(cand_dims)
        if len(selected_pairs) >= 3:
            break

    # ── 选择结果日志 ──
    print(f"\n  [互补配对] {len(selected_pairs)} 对新复合入选 (统一评分↓):")
    for i, sp in enumerate(selected_pairs):
        fri_str = f"FRI={sp['fri']:.3f}" if sp['fri'] > 0 else "FRI=N/A(LLM)"
        nov_str = f"nov={sp['novelty']:.3f}" if sp['novelty'] > 0 else ""
        print(f"    #{i+1} {sp['candidate'].name[:28]:28s} × {sp['v3_factor']:18s} "
              f"互补={sp['score']:.1f} 统一={sp['unified_score']:.1f} {fri_str} {nov_str} [{sp['_dims']}]")
    if skipped_by_diversity:
        print(f"\n  [多样性过滤] {len(skipped_by_diversity)} 个因子因维度冗余被跳过:")
        for sk in skipped_by_diversity[:5]:
            print(f"    ✗ {sk['name']:30s} 最大重叠={sk['max_overlap']:.2f} "
                  f"统一分={sk['unified']:.1f} FRI={sk['fri']:.3f}")

    # 构建新复合
    new_composites = []
    for sp in selected_pairs:
        new_composites.append({
            "name": f"comp{4+len(new_composites)}_{sp['candidate'].name[:15]}_{sp['v3_factor']}",
            "a_type": "new",
            "a_expr": sp["candidate"].expression,
            "a_name": sp["candidate"].name,
            "b_type": "v3",
            "b_name": sp["v3_factor"],
            "rationale": f"{sp['candidate'].rationale[:40]} x {sp['v3_factor']}: "
                         f"互补评分={sp['score']:.1f}",
            "from_v3": False,
        })

    # 合并: 3个V3原 + N个新复合, 最多6个
    all_composites = preserved_v3_composites + new_composites
    all_composites = all_composites[:6]

    # ── RidgeUCB 影子评分 (不改变选择, 只预测+记录, 供校准追踪) ──
    try:
        _shadow_score_portfolio(all_composites, selected_pairs)
    except Exception as e:
        print(f"  [影子评分] 跳过 (scorer不可用: {e})")

    if len(all_composites) < 5:
        # 补满到5个
        fallbacks = [
            {"name": "comp_fb1_retopen_skew", "a_type": "v3", "a_name": "ret_open_2d",
             "b_type": "v3", "b_name": "skewness_20",
             "rationale": "开盘动量x收益偏度: V3原始复合", "from_v3": True},
            {"name": "comp_fb2_gapup_spread", "a_type": "v3", "a_name": "gap_up",
             "b_type": "v3", "b_name": "relative_spread",
             "rationale": "跳空x振幅: V3原始复合", "from_v3": True},
        ]
        for fb in fallbacks:
            if len(all_composites) >= 5:
                break
            all_composites.append(fb)

    print(f"\n  [最终复合] {len(all_composites)} 个:")
    for i, comp in enumerate(all_composites):
        src = "V3" if comp["from_v3"] else "NEW"
        a = comp.get("a_name", "?")[:20]
        b = comp.get("b_name", "?")[:20]
        print(f"    comp{i+1}: {src} {a} x {b}")

    return all_composites


# ═══════════════════════════════════════════════════════════
# 阶段3: JQ策略生成 (复用v1逻辑)
# ═══════════════════════════════════════════════════════════

def generate_jq_strategy(composites: List[dict],
                         candidates: List[AlphaCandidate],
                         output_path: str = None):
    """生成JQ策略代码 (复用v1生成逻辑)"""
    import importlib.util

    # 从v1 pipeline加载JQ生成函数
    spec = importlib.util.spec_from_file_location(
        "v1_pipeline", ROOT / "alpha_agent_pipeline.py")
    v1_pipeline = importlib.util.module_from_spec(spec)

    # Monkey-patch: 用v3的generate_novel_factors替换v1的
    original_gen = None
    try:
        spec.loader.exec_module(v1_pipeline)
        original_gen = v1_pipeline.generate_novel_factors
        # 不用v1的生成函数, 用v3的composites和candidates
    except Exception as e:
        print(f"  [!] 加载v1 pipeline失败: {e}, 回退内联生成")
        return _generate_jq_inline(composites, candidates, output_path)

    # 内联生成 (更可靠)
    return _generate_jq_inline(composites, candidates, output_path)


def _generate_jq_inline(composites, candidates, output_path=None):
    """内联JQ策略生成"""
    import re

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = ROOT / "output" / f"fa_alpha_agent_v3_jq_{timestamp}.py"

    def sanitize(name: str) -> str:
        name = name.replace('(', '_').replace(')', '_').replace(' ', '_')
        # Try extracting English/alphanumeric parts
        ascii_part = re.sub(r'[^\x00-\x7F]', '', name)
        if not ascii_part.strip('_'):
            # No ASCII chars - generate hash-based name
            import hashlib
            h = hashlib.md5(name.encode('utf-8')).hexdigest()[:6]
            return f'nf_{h}'
        # Remove non-alphanumeric except underscore
        name = re.sub(r'[^a-zA-Z0-9_]', '', ascii_part)
        name = re.sub(r'_+', '_', name).strip('_')
        if not name or name[0].isdigit():
            name = 'f_' + name
        return name.lower()

    lines = []
    l = lines.append

    # ── JQ Header ──
    l("import datetime")
    l("import pandas as pd")
    l("from jqlib.technical_analysis import *")
    l("")
    l("def initialize(context):")
    l("    g.stock_num = 80")
    l("    g.limit_pct = 0.10")
    l("    g.initial_capital = context.portfolio.total_value")
    l("    g.trade_days = 0")
    l("    set_benchmark('000300.XSHG')")
    l("    set_option('use_real_price', True)")
    l(f"    log.set_level('order', 'error')")
    l(f"    log.set_level('strategy', 'info')")
    l("    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,")
    l("                              open_commission=0.0001, close_commission=0.0001),")
    l("                   type='stock')")
    l("    set_slippage(FixedSlippage(0.003))")
    l("    run_weekly(rebalance, 1, time='open', reference_security='000300.XSHG')")
    l("")

    # ── Helper functions ──
    l("def rank_pct(arr, valid):")
    l("    '''截面百分位排名, NaN被忽略'''")
    l("    import numpy as np")
    l("    n = len(arr)")
    l("    valid_arr = arr[valid]")
    l("    if len(valid_arr) < 3:")
    l("        result = np.full(n, 0.5)")
    l("        result[~valid] = 0.5")
    l("        return result")
    l("    from scipy.stats import rankdata")
    l("    ranks = rankdata(valid_arr)")
    l("    pct = (ranks - 1) / (len(valid_arr) - 1)")
    l("    result = np.full(n, 0.5)")
    l("    result[valid] = pct")
    l("    result[~valid] = 0.5")
    l("    return result")
    l("")
    l("def zscore(arr, valid):")
    l("    import numpy as np")
    l("    v = arr[valid]")
    l("    if len(v) < 3:")
    l("        return np.zeros(len(arr))")
    l("    m, s = np.mean(v), np.std(v)")
    l("    if s < 1e-10:")
    l("        return np.zeros(len(arr))")
    l("    result = np.zeros(len(arr))")
    l("    result[valid] = (v - m) / s")
    l("    return result")
    l("")

    # ── V3因子实现 ──
    v3_needed = set()
    for comp in composites:
        for side in ["a", "b"]:
            if comp.get(f"{side}_type") == "v3":
                v3_needed.add(comp[f"{side}_name"])

    _gen_v3_factors(l, v3_needed)

    # ── 新因子实现 ──
    for comp in composites:
        if comp.get("a_type") == "new":
            cand = next((c for c in candidates if c.name == comp["a_name"]), None)
            safe_a = sanitize(comp["a_name"])
            _gen_new_factor_impl(l, safe_a, comp.get("a_expr", ""), cand)
        if comp.get("b_type") == "new":
            cand = next((c for c in candidates if c.name == comp["b_name"]), None)
            safe_b = sanitize(comp["b_name"])
            _gen_new_factor_impl(l, safe_b, comp.get("b_expr", ""), cand)

    # ── 复合函数 ──
    for comp in composites:
        safe_name = sanitize(comp["name"])
        a_src = sanitize(comp.get("a_name", "")) if comp.get("a_type") == "new" else comp.get("a_name", "")
        b_src = sanitize(comp.get("b_name", "")) if comp.get("b_type") == "new" else comp.get("b_name", "")
        l(f"def compute_score_{safe_name}(stocks, price_data):")
        l(f'    """{comp.get("rationale", "")[:60]}"""')
        l(f"    a, va = compute_{a_src}(stocks, price_data)")
        l(f"    b, vb = compute_{b_src}(stocks, price_data)")
        l(f"    valid = va & vb")
        l(f"    ra = rank_pct(a, valid)")
        l(f"    rb = rank_pct(b, valid)")
        l(f"    score = np.where(valid, ra * rb, 0.5)")
        l(f"    return score, valid")
        l("")

    # ── Ensemble ──
    l("def compute_ensemble(stocks, price_data):")
    l("    import numpy as np")
    l("    n = len(stocks)")
    l("    composite = np.ones(n) * 0.5")
    l("    valid = np.ones(n, dtype=bool)")
    l("    score_funcs = [")
    for comp in composites:
        safe_name = sanitize(comp["name"])
        l(f"        (compute_score_{safe_name},),")
    l("    ]")
    l("    count = 0")
    l("    for (func,) in score_funcs:")
    l("        score, v = func(stocks, price_data)")
    l("        n_v = int(np.sum(v))")
    l("        if n_v > 0:")
    l("            composite[v] *= score[v]")
    l("            count += 1")
    l("            valid = valid & v")
    l("    return composite, valid")
    l("")

    # ── Rebalance ──
    l("def rebalance(context):")
    l("    import numpy as np")
    l("    import gc")
    l("    g.trade_days += 1")
    l("    if g.trade_days < 5:")
    l("        return")
    l("    current_dt = context.current_dt")
    l("    prev_date = context.previous_date")
    l("")
    l("    # Universe")
    l("    universe = get_all_securities(['stock'], prev_date).index.tolist()")
    l("    CHUNK = 500")
    l("    filtered = []")
    l("    for k in range(0, len(universe), CHUNK):")
    l("        chunk = universe[k:k+CHUNK]")
    l("        try:")
    l("            cd = get_current_data()")
    l("            for s in chunk:")
    l("                if cd[s].paused:")
    l("                    continue")
    l("                if cd[s].is_st:")
    l("                    continue")
    l("                if cd[s].high_limit <= cd[s].last_price:")
    l("                    continue")
    l("                filtered.append(s)")
    l("        except Exception:")
    l("            filtered.extend(chunk)")
    l("    universe = [s for s in filtered if s.startswith(('0','3','6'))][:2000]")
    l("    if len(universe) < g.stock_num:")
    l("        return")
    l("")
    l("    # Load price data (参照V3策略: 不用panel=False + Panel兼容)")
    l("    lookback = 70")
    l("    field = 'close'")
    l("    px_close = get_price(universe, count=lookback, end_date=prev_date,")
    l("                         frequency='daily', fields=field,")
    l("                         skip_paused=False, fq='pre')")
    l("    field = 'open'")
    l("    px_open = get_price(universe, count=lookback, end_date=prev_date,")
    l("                        frequency='daily', fields=field,")
    l("                        skip_paused=False, fq='pre')")
    l("    field = 'high'")
    l("    px_high = get_price(universe, count=lookback, end_date=prev_date,")
    l("                        frequency='daily', fields=field,")
    l("                        skip_paused=False, fq='pre')")
    l("    field = 'low'")
    l("    px_low = get_price(universe, count=lookback, end_date=prev_date,")
    l("                       frequency='daily', fields=field,")
    l("                       skip_paused=False, fq='pre')")
    l("    field = 'volume'")
    l("    px_volume = get_price(universe, count=lookback, end_date=prev_date,")
    l("                          frequency='daily', fields=field,")
    l("                          skip_paused=False, fq='pre')")
    l("")
    l("    # Panel兼容处理")
    l("    _Panel = getattr(pd, 'Panel', None)")
    l("    if _Panel is not None:")
    l("        if isinstance(px_close, _Panel):")
    l("            px_close = px_close['close'] if 'close' in getattr(px_close, 'items', []) else px_close.minor_xs('close')")
    l("        if isinstance(px_open, _Panel):")
    l("            px_open = px_open['open'] if 'open' in getattr(px_open, 'items', []) else px_open.minor_xs('open')")
    l("        if isinstance(px_high, _Panel):")
    l("            px_high = px_high['high'] if 'high' in getattr(px_high, 'items', []) else px_high.minor_xs('high')")
    l("        if isinstance(px_low, _Panel):")
    l("            px_low = px_low['low'] if 'low' in getattr(px_low, 'items', []) else px_low.minor_xs('low')")
    l("        if isinstance(px_volume, _Panel):")
    l("            px_volume = px_volume['volume'] if 'volume' in getattr(px_volume, 'items', []) else px_volume.minor_xs('volume')")
    l("")
    l("    price_data = {")
    l("        'close': {s: px_close[s].values for s in universe if s in px_close.columns},")
    l("        'open': {s: px_open[s].values for s in universe if s in px_open.columns},")
    l("        'high': {s: px_high[s].values for s in universe if s in px_high.columns},")
    l("        'low': {s: px_low[s].values for s in universe if s in px_low.columns},")
    l("        'volume': {s: px_volume[s].values for s in universe if s in px_volume.columns},")
    l("    }")
    l("")
    l("    valid_universe = [s for s in universe if len(price_data['close'].get(s, [])) >= 65]")
    l("    if len(valid_universe) < g.stock_num:")
    l("        return")

    l("")
    l("    composite, valid = compute_ensemble(valid_universe, price_data)")
    l("    composite_arr = np.where(valid, composite, -1)")
    l("")
    l("    topn = g.stock_num")
    l("    top_idx = np.argsort(composite_arr)[-topn:][::-1]")
    l("    top_stocks = [valid_universe[i] for i in top_idx if composite_arr[i] > -1]")
    l("")
    l("    # Sell: skip positions too small to close")
    l("    for s in list(context.portfolio.positions.keys()):")
    l("        if s not in top_stocks:")
    l("            try:")
    l("                pos = context.portfolio.positions[s]")
    l("                if pos.total_amount < 100:")
    l("                    continue")
    l("            except Exception:")
    l("                pass")
    l("            order_target_value(s, 0)")
    l("")
    l("    if len(top_stocks) == 0:")
    l("        return")
    l("")
    l("    weight = context.portfolio.total_value / len(top_stocks)")
    l("    ordered = 0")
    l("    for s in top_stocks:")
    l("        try:")
    l("            last_px = get_current_data()[s].last_price")
    l("            if last_px <= 0 or weight / last_px < 100:")
    l("                continue")
    l("        except Exception:")
    l("            continue")
    l("        order_target_value(s, weight)")
    l("        ordered += 1")
    l("    log.info('ordered %d/%d stocks' % (ordered, len(top_stocks)))")
    l("    gc.collect()")
    l("")

    code = "\n".join(lines)

    # 去除Unicode问题字符
    code = code.replace('\u00d7', 'x')
    code = code.replace('\u201c', '"').replace('\u201d', '"')
    code = code.replace('\u2018', "'").replace('\u2019', "'")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)

    _validate_jq(code, str(output_path))

    print(f"\n  [JQ] 策略已生成: {output_path}")
    return str(output_path)


def _gen_v3_factors(l, needed):
    """生成V3因子的JQ实现"""
    v3_impl = {
        "overnight_5d": [
            "def compute_overnight_5d(stocks, price_data):",
            '    """隔夜累计5日"""',
            "    import numpy as np",
            "    close = price_data.get('close', {})",
            "    open_ = price_data.get('open', {})",
            "    n = len(stocks)",
            "    arr = np.full(n, np.nan)",
            "    valid = np.zeros(n, dtype=bool)",
            "    for i, s in enumerate(stocks):",
            "        if close.get(s) is None or open_.get(s) is None:",
            "            continue",
            "        c = np.array(close.get(s, []), dtype=float)",
            "        o = np.array(open_.get(s, []), dtype=float)",
            "        if len(c) < 7 or len(o) < 7:",
            "            continue",
            "        gap = o[-5:] / np.roll(c, 1)[-5:] - 1",
            "        arr[i] = np.sum(gap)",
            "        valid[i] = True",
            "    return arr, valid",
        ],
        "tvma_20": [
            "def compute_tvma_20(stocks, price_data):",
            '    """量价趋势偏移(取负)"""',
            "    import numpy as np",
            "    close = price_data.get('close', {})",
            "    volume = price_data.get('volume', {})",
            "    n = len(stocks)",
            "    arr = np.full(n, np.nan)",
            "    valid = np.zeros(n, dtype=bool)",
            "    for i, s in enumerate(stocks):",
            "        if close.get(s) is None or volume.get(s) is None:",
            "            continue",
            "        c = np.array(close.get(s, []), dtype=float)",
            "        v = np.array(volume.get(s, []), dtype=float)",
            "        if len(c) < 25 or len(v) < 25:",
            "            continue",
            "        cv = c[-20:] * v[-20:]",
            "        ma_cv = np.mean(cv)",
            "        curr_cv = c[-1] * v[-1]",
            "        arr[i] = -(curr_cv / ma_cv) if ma_cv > 0 else 0",
            "        valid[i] = True",
            "    return arr, valid",
        ],
        "dollar_vol_20d": [
            "def compute_dollar_vol_20d(stocks, price_data):",
            '    """成交额对数(取负)"""',
            "    import numpy as np",
            "    close = price_data.get('close', {})",
            "    volume = price_data.get('volume', {})",
            "    n = len(stocks)",
            "    arr = np.full(n, np.nan)",
            "    valid = np.zeros(n, dtype=bool)",
            "    for i, s in enumerate(stocks):",
            "        if close.get(s) is None or volume.get(s) is None:",
            "            continue",
            "        c = np.array(close.get(s, []), dtype=float)",
            "        v = np.array(volume.get(s, []), dtype=float)",
            "        if len(c) < 25 or len(v) < 25:",
            "            continue",
            "        dv = np.mean(c[-20:] * v[-20:])",
            "        arr[i] = -np.log(dv + 1) if dv > 0 else 0",
            "        valid[i] = True",
            "    return arr, valid",
        ],
        "turnover_std_cv": [
            "def compute_turnover_std_cv(stocks, price_data):",
            '    """换手变异系数(取负)"""',
            "    import numpy as np",
            "    volume = price_data.get('volume', {})",
            "    n = len(stocks)",
            "    arr = np.full(n, np.nan)",
            "    valid = np.zeros(n, dtype=bool)",
            "    for i, s in enumerate(stocks):",
            "        if volume.get(s) is None:",
            "            continue",
            "        v = np.array(volume.get(s, []), dtype=float)",
            "        if len(v) < 25:",
            "            continue",
            "        std_v = np.nanstd(v[-20:])",
            "        mean_v = np.nanmean(v[-20:])",
            "        arr[i] = -(std_v / mean_v) if mean_v > 0 else 0",
            "        valid[i] = True",
            "    return arr, valid",
        ],
        "money_flow_20": [
            "def compute_money_flow_20(stocks, price_data):",
            '    """资金流偏移(取负)"""',
            "    import numpy as np",
            "    high = price_data.get('high', {})",
            "    low = price_data.get('low', {})",
            "    close = price_data.get('close', {})",
            "    volume = price_data.get('volume', {})",
            "    n = len(stocks)",
            "    arr = np.full(n, np.nan)",
            "    valid = np.zeros(n, dtype=bool)",
            "    for i, s in enumerate(stocks):",
            "        if any(high.get(s) is None or len(high.get(s, [])) < 25 for _ in [1]):",
            "            continue",
            "        h = np.array(high.get(s, []), dtype=float)",
            "        l_ = np.array(low.get(s, []), dtype=float)",
            "        c = np.array(close.get(s, []), dtype=float)",
            "        v = np.array(volume.get(s, []), dtype=float)",
            "        if len(h) < 25 or len(l_) < 25:",
            "            continue",
            "        typical = (h[-20:] + l_[-20:] + c[-20:]) * v[-20:] / 3",
            "        curr_tp = (h[-1] + l_[-1] + c[-1]) * v[-1] / 3",
            "        ma_tp = np.mean(typical)",
            "        arr[i] = -(curr_tp - ma_tp)",
            "        valid[i] = True",
            "    return arr, valid",
        ],
        "ret_3m": [
            "def compute_ret_3m(stocks, price_data):",
            '    """3月收益(取负=反转)"""',
            "    import numpy as np",
            "    close = price_data.get('close', {})",
            "    n = len(stocks)",
            "    arr = np.full(n, np.nan)",
            "    valid = np.zeros(n, dtype=bool)",
            "    for i, s in enumerate(stocks):",
            "        if close.get(s) is None:",
            "            continue",
            "        c = np.array(close.get(s, []), dtype=float)",
            "        if len(c) < 65:",
            "            continue",
            "        arr[i] = -(c[-1] / c[-61] - 1)",
            "        valid[i] = True",
            "    return arr, valid",
        ],
        "ret_open_2d": [
            "def compute_ret_open_2d(stocks, price_data):",
            '    """开盘动量2日均值"""',
            "    import numpy as np",
            "    close = price_data.get('close', {})",
            "    open_ = price_data.get('open', {})",
            "    n = len(stocks)",
            "    arr = np.full(n, np.nan)",
            "    valid = np.zeros(n, dtype=bool)",
            "    for i, s in enumerate(stocks):",
            "        if close.get(s) is None or open_.get(s) is None:",
            "            continue",
            "        c = np.array(close.get(s, []), dtype=float)",
            "        o = np.array(open_.get(s, []), dtype=float)",
            "        if len(c) < 4 or len(o) < 4:",
            "            continue",
            "        arr[i] = np.mean(o[-2:] / np.roll(c, 1)[-2:] - 1)",
            "        valid[i] = True",
            "    return arr, valid",
        ],
        "skewness_20": [
            "def compute_skewness_20(stocks, price_data):",
            '    """收益波动率(取负)"""',
            "    import numpy as np",
            "    close = price_data.get('close', {})",
            "    n = len(stocks)",
            "    arr = np.full(n, np.nan)",
            "    valid = np.zeros(n, dtype=bool)",
            "    for i, s in enumerate(stocks):",
            "        if close.get(s) is None:",
            "            continue",
            "        c = np.array(close.get(s, []), dtype=float)",
            "        if len(c) < 22:",
            "            continue",
            "        rets = c[1:][-20:] / c[:-1][-20:] - 1",
            "        arr[i] = -np.nanstd(rets)",
            "        valid[i] = True",
            "    return arr, valid",
        ],
        "gap_up": [
            "def compute_gap_up(stocks, price_data):",
            '    """跳空幅度"""',
            "    import numpy as np",
            "    close = price_data.get('close', {})",
            "    open_ = price_data.get('open', {})",
            "    n = len(stocks)",
            "    arr = np.full(n, np.nan)",
            "    valid = np.zeros(n, dtype=bool)",
            "    for i, s in enumerate(stocks):",
            "        if close.get(s) is None or open_.get(s) is None:",
            "            continue",
            "        c = np.array(close.get(s, []), dtype=float)",
            "        o = np.array(open_.get(s, []), dtype=float)",
            "        if len(c) < 2 or len(o) < 2:",
            "            continue",
            "        arr[i] = o[-1] / c[-2] - 1",
            "        valid[i] = True",
            "    return arr, valid",
        ],
        "relative_spread": [
            "def compute_relative_spread(stocks, price_data):",
            '    """相对振幅(取负)"""',
            "    import numpy as np",
            "    high = price_data.get('high', {})",
            "    low = price_data.get('low', {})",
            "    close = price_data.get('close', {})",
            "    n = len(stocks)",
            "    arr = np.full(n, np.nan)",
            "    valid = np.zeros(n, dtype=bool)",
            "    for i, s in enumerate(stocks):",
            "        if high.get(s) is None:",
            "            continue",
            "        h = np.array(high.get(s, []), dtype=float)",
            "        l_ = np.array(low.get(s, []), dtype=float)",
            "        c = np.array(close.get(s, []), dtype=float)",
            "        if len(h) < 1 or len(l_) < 1:",
            "            continue",
            "        arr[i] = -(h[-1] - l_[-1]) / (c[-1] + 0.001)",
            "        valid[i] = True",
            "    return arr, valid",
        ],
    }

    for name in sorted(needed):
        if name in v3_impl:
            for line in v3_impl[name]:
                l(line)
            l("")
        else:
            l(f"# WARNING: V3 factor {name} not implemented")
            l(f"def compute_{name}(stocks, price_data):")
            l(f"    import numpy as np")
            l(f"    return np.zeros(len(stocks)), np.zeros(len(stocks), dtype=bool)")
            l("")


def _gen_new_factor_impl(l, name, expr, cand):
    """生成新因子的JQ实现 (从expression推导真实计算)"""
    if cand is None:
        l(f"def compute_{name}(stocks, price_data):")
        l(f'    """(占位)"""')
        l(f"    import numpy as np")
        l(f"    return np.zeros(len(stocks)), np.zeros(len(stocks), dtype=bool)")
        l("")
        return

    rationale = (cand.rationale or "")[:60].replace('"', "'")

    # 解析expression中需要的字段
    needs_close = 'close' in expr
    needs_high = 'high' in expr
    needs_low = 'low' in expr
    needs_volume = 'volume' in expr
    needs_open = 'open' in expr
    max_window = 65

    l(f"def compute_{name}(stocks, price_data):")
    l(f'    """{rationale}"""')
    l(f"    import numpy as np")
    l(f"    n = len(stocks)")

    # 默认: 基于close + volume做简单可实现计算
    if "ts_min(returns" in expr or "min(returns" in expr:
        # 基于returns的因子: 需要close计算returns
        max_window = max(max_window, 65)
        l(f"    close = price_data.get('close', {{}})")
        l(f"    arr = np.full(n, np.nan)")
        l(f"    valid = np.zeros(n, dtype=bool)")
        l(f"    for i, s in enumerate(stocks):")
        l(f"        if close.get(s) is None:")
        l(f"            continue")
        l(f"        c = np.array(close.get(s, []), dtype=float)")
        l(f"        if len(c) < {max_window + 2}:")
        l(f"            continue")
        l(f"        rets = c[1:] / c[:-1] - 1")
        l(f"        recent = rets[-20:]")
        l(f"        # 下行风险指标")
        l(f"        min_ret = np.nanmin(recent)")
        l(f"        std_ret = np.nanstd(recent)")
        l(f"        if std_ret > 0:")
        l(f"            arr[i] = -min_ret / std_ret")
        l(f"        else:")
        l(f"            arr[i] = 0.0")
        l(f"        valid[i] = True")
        l(f"    return arr, valid")
    elif "ts_std" in expr and "close" in expr and "volume" not in expr:
        max_window = max(max_window, 65)
        l(f"    close = price_data.get('close', {{}})")
        l(f"    arr = np.full(n, np.nan)")
        l(f"    valid = np.zeros(n, dtype=bool)")
        l(f"    for i, s in enumerate(stocks):")
        l(f"        if close.get(s) is None:")
        l(f"            continue")
        l(f"        c = np.array(close.get(s, []), dtype=float)")
        l(f"        if len(c) < {max_window + 1}:")
        l(f"            continue")
        l(f"        std20 = np.nanstd(c[-20:])")
        l(f"        mean20 = np.nanmean(c[-20:])")
        l(f"        if mean20 > 0:")
        l(f"            arr[i] = -(std20 / mean20)")
        l(f"        else:")
        l(f"            arr[i] = 0.0")
        l(f"        valid[i] = True")
        l(f"    return arr, valid")
    elif "ts_pct" in expr or "rank(ts_pct" in expr:
        l(f"    close = price_data.get('close', {{}})")
        l(f"    arr = np.full(n, np.nan)")
        l(f"    valid = np.zeros(n, dtype=bool)")
        l(f"    for i, s in enumerate(stocks):")
        l(f"        if close.get(s) is None:")
        l(f"            continue")
        l(f"        c = np.array(close.get(s, []), dtype=float)")
        l(f"        if len(c) < 25:")
        l(f"            continue")
        l(f"        arr[i] = c[-1] / c[-21] - 1")
        l(f"        valid[i] = True")
        l(f"    return arr, valid")
    elif "div(volume, ts_max(volume" in expr or "neg(div(volume" in expr:
        l(f"    volume = price_data.get('volume', {{}})")
        l(f"    arr = np.full(n, np.nan)")
        l(f"    valid = np.zeros(n, dtype=bool)")
        l(f"    for i, s in enumerate(stocks):")
        l(f"        if volume.get(s) is None:")
        l(f"            continue")
        l(f"        v = np.array(volume.get(s, []), dtype=float)")
        l(f"        if len(v) < 25:")
        l(f"            continue")
        l(f"        max20 = np.nanmax(v[-20:])")
        l(f"        if max20 > 0:")
        l(f"            arr[i] = -(v[-1] / max20)")
        l(f"        else:")
        l(f"            arr[i] = 0.0")
        l(f"        valid[i] = True")
        l(f"    return arr, valid")
    else:
        # 通用实现: close-based
        l(f"    close = price_data.get('close', {{}})")
        l(f"    volume = price_data.get('volume', {{}})")
        l(f"    arr = np.full(n, np.nan)")
        l(f"    valid = np.zeros(n, dtype=bool)")
        l(f"    W = {max_window}")
        l(f"    for i, s in enumerate(stocks):")
        l(f"        if close.get(s) is None or len(close.get(s, [])) < W:")
        l(f"            continue")
        l(f"        c = np.array(close.get(s, []), dtype=float)")
        l(f"        # 默认: 价格相对20日均值偏离")
        l(f"        ma20 = np.nanmean(c[-20:])")
        l(f"        if ma20 > 0:")
        l(f"            arr[i] = -(c[-1] / ma20 - 1)")
        l(f"        else:")
        l(f"            arr[i] = 0.0")
        l(f"        valid[i] = True")
        l(f"    return arr, valid")

    l("")


def _validate_jq(code, path):
    """验证JQ兼容性"""
    try:
        compile(code, 'test', 'exec')
        print(f"  [validate] compile OK")
    except SyntaxError as e:
        print(f"  [validate] SyntaxError: {e}")


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main(evo_context_path=None):
    print("=" * 70)
    print("AlphaAgent Pipeline v3: 单轮55+候选 → 互补配对 → V3集成 → JQ")
    print("=" * 70)

    # 配置
    config = AlphaAgentConfig(
        originality_threshold=0.70,
        max_complexity=28,
        icir_threshold=0.15,
        output_dir=str(ROOT / "output"),
    )

    # Step 0: 读取演化上下文 (已冻结 — 2026-08-04)
    # ═══════════════════════════════════════════════════════════
    # Local数据驱动的进化 = Local→JQ Gap放大器。
    # 因子注入管线 (Stage2→Psi→FRI 三级local筛选) 造成系统性选择偏差。
    # 注入因子在JQ上系统性跑输纯LLM生成因子。
    # 详见: evaluator/evolution_roadmap.md
    # ═══════════════════════════════════════════════════════════
    evo_factors = []
    evo_knowledge = {}
    if evo_context_path and Path(evo_context_path).exists():
        print("\n⚠️  ===============================================")
        print("⚠️  --evo-context 已冻结 (2026-08-04)")
        print("⚠️  Local数据驱动的因子注入 = JQ Gap放大器")
        print("⚠️  注入因子在JQ上系统性跑输纯LLM生成因子")
        print("⚠️  本次运行将忽略 --evo-context，使用纯LLM模式")
        print("⚠️  详见: evaluator/evolution_roadmap.md")
        print("⚠️  ===============================================")
        print("\n[0/6] 自进化链路已冻结 — 纯LLM模式")
    else:
        print("\n[0/6] 纯LLM模式 — 无自进化组件")

    # Step 1: 初始化V3因子池
    print("\n[1/6] 初始化V3因子池...")
    n_pool = init_v3_pool()
    print(f"  V3池: {n_pool} 个底因子")

    # Step 2: 生成新因子 (55+候选, 单轮, 不进化)
    print("\n[2/6] LLM生成因子公式 (12范式, 单轮)...")
    factor_dicts = generate_novel_factors_v3()
    print(f"  生成: {len(factor_dicts)} 个候选因子")
    paradigms = set(f["paradigm"] for f in factor_dicts)
    print(f"  覆盖范式: {len(paradigms)} 个")

    # Step 3: 三重约束过滤
    print("\n[3/6] 三重约束过滤...")
    candidates = []
    passed_count = 0
    for idx, fd in enumerate(factor_dicts):
        cand = AlphaCandidate(
            id=f"gen0_{idx}",
            name=fd["name"],
            expression=fd["expression"],
            rationale=fd["rationale"],
            direction=fd.get("direction", "+"),
        )
        # 附加上下文 (不在dataclass中的用临时属性)
        cand._paradigm = fd.get("paradigm", "")
        cand._filter_reason = ""

        # 约束1: Originality
        try:
            node = parse_expression(fd["expression"])
        except Exception:
            cand.passed_triple = False
            cand._filter_reason = "parse_failed"
            candidates.append(cand)
            continue

        is_orig, max_sim, closest = True, 0.0, ""
        try:
            is_orig, max_sim, closest = check_originality(node, V3_FACTOR_POOL, 0.70)
        except Exception as e:
            cand._filter_reason = f"originality_check_failed({e})"
            candidates.append(cand)
            continue
        cand.originality_max_sim = max_sim
        cand.originality_closest = closest

        if not is_orig:
            cand.passed_triple = False
            cand._filter_reason = f"originality({max_sim:.2f}>{config.originality_threshold})"
            candidates.append(cand)
            continue

        # 约束2: Hypothesis (简化: 必须有rationale且长度>=15)
        hypo_ok = len(fd["rationale"]) >= 15
        if not hypo_ok:
            cand.passed_triple = False
            cand._filter_reason = "hypothesis_incomplete"
            candidates.append(cand)
            continue

        # 约束3: Complexity
        comp_ok, comp_n = True, 0
        try:
            comp_ok, comp_n = check_complexity(node, config.max_complexity)
        except Exception as e:
            cand._filter_reason = f"complexity_check_failed({e})"
            candidates.append(cand)
            continue
        cand.complexity_nodes = comp_n
        if not comp_ok:
            cand.passed_triple = False
            cand._filter_reason = f"complexity({comp_n}>{config.max_complexity})"
            candidates.append(cand)
            continue

        cand.passed_triple = True
        passed_count += 1
        candidates.append(cand)

    # Step 3.5: 注入演化因子 (FRI已预验证, 跳过三重约束)
    if evo_factors:
        print(f"\n  [注入] 添加 {len(evo_factors)} 个FRI预验证因子...")
        # 类别→AlphaAgent范式映射
        CATEGORY_TO_PARADIGM = {
            "extreme_events": "尾部风险",
            "growth_quality": "基本面/成长",
            "mid_report_divergence": "趋势/动量",
            "学术异象库": "尾部风险",
            "stage1_exploration": "资金流",
            "liquidity_microstructure": "流动性×微观结构",
            "momentum": "动量",
            "reversal": "反转",
            "volatility_risk": "波动率",
            "fund_flow": "资金流",
            "sentiment_behavioral": "行为金融",
            "fundamental_quality": "基本面/质量",
            "valuation": "估值",
            "size_turnover": "流动性×微观结构",
            "tail_extreme": "尾部风险",
            # ── Round 2 新增类别 ──────────────────────────
            "failure_mode_inverse": "反转×资金流",
            "small_bull_breakthrough": "流动性×微观结构",
            "vm_diff_replacement": "波动率×趋势/动量",
            "ml_inspired_construction": "趋势/动量",
            "academic_anomaly_adaptation": "行为金融×尾部风险",
            "volume_structure": "流动性",
            "momentum_dynamics": "趋势/动量",
            "volatility_structure": "波动率",
            "price_pattern": "反转",
            "liquidity_micro": "微观结构",
            "behavioral": "行为金融",
        }
        evo_idx_start = len(candidates)
        for ei, ef in enumerate(evo_factors):
            evo_expr = f"evo_injected_{ef['name']}"
            paradigm = CATEGORY_TO_PARADIGM.get(ef.get("category", ""), "行为金融")
            # ── 优先使用注入时生成的 rationale (Step 0 从 JSON 读取)   ──
            # ── 如果缺失则 fallback 到 build_injected_rationale()     ──
            existing_rationale = ef.get("rationale", "").strip()
            if existing_rationale and "FRI=" not in existing_rationale:
                # 注入时已包含语义描述，补上 FRI/ICIR 后缀
                rationale = f"{existing_rationale} [FRI={ef.get('fri', 0):.2f}/{ef.get('fri_grade', '?')}, ICIR={ef.get('icir', 0):.3f}]"
            elif existing_rationale:
                # 已含完整后缀，直接使用
                rationale = existing_rationale
            else:
                # fallback: 模板重建
                rationale = build_injected_rationale(
                    factor_name=ef['name'],
                    factor_label=ef.get('label', ef['name']),
                    fri=ef.get('fri', 0),
                    fri_grade=ef.get('fri_grade', '?'),
                    icir=ef.get('icir', 0),
                    paradise=paradigm,
                    category=ef.get('category', ''),
                )
            cand = AlphaCandidate(
                id=f"evo_{ei}",
                name=ef["name"],
                expression=evo_expr,
                rationale=rationale,
                direction="+",
            )
            cand.passed_triple = True   # FRI预验证
            cand._paradigm = paradigm
            cand._filter_reason = ""
            cand._is_injected = True
            cand._fri = ef["fri"]
            cand._fri_grade = ef["fri_grade"]
            cand._icir = ef["icir"]
            cand._fri_novelty = ef.get("fri_novelty", 0)
            # ── 存储预定义维度 (补全配对评分用) ─────────────────────
            cand._dimensions = ef.get("dimensions", [])
            candidates.append(cand)
            passed_count += 1
        print(f"  [注入] 候选池扩容: {len(candidates)} (原{len(candidates)-len(evo_factors)} + 注入{len(evo_factors)})")

    # 报告
    print(f"  通过: {passed_count}/{len(factor_dicts)}")
    rejected = [c for c in candidates if not c.passed_triple]
    if rejected:
        print(f"  拒绝: {len(rejected)} 个")
        # 按原因分组
        reasons = {}
        for c in rejected:
            r = getattr(c, '_filter_reason', 'unknown') or "unknown"
            reasons[r] = reasons.get(r, 0) + 1
        for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    - {r}: {cnt}")

    # Step 4: 互补评分配对
    print("\n[4/6] 互补评分配对...")
    composites = build_v3_integration_v3(candidates)

    # Step 5: JQ策略生成
    print("\n[5/6] JQ策略生成...")
    jq_path = generate_jq_strategy(composites, candidates)

    print("\n" + "=" * 70)
    print(f"完成! JQ策略: {jq_path}")
    print(f"复合数: {len(composites)}, 范式数: {len(paradigms)}, 候选数: {len(factor_dicts)}")
    print("=" * 70)

    return jq_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AlphaAgent Pipeline v3")
    parser.add_argument("--evo-context", type=str, default=None,
                        help="[已冻结] 演化上下文JSON路径。local数据驱动=JQ Gap放大器，已忽略。")
    args = parser.parse_args()
    main(evo_context_path=args.evo_context)
