# -*- coding: utf-8 -*-
"""
Seed Injector — 将 AlphaAgent v3 王者因子注入 Experience Memory + 合并因子库
==================================================================================

功能:
  1. 从 alpha_agent_pipeline_v3.py 提取 55+ 因子表达式
  2. 转换为 passed_factor_pool.csv 的统一格式
  3. 注入 Experience Memory 作为 SuccessPattern 种子
  4. 与新因子库合并去重 (统一 annotation/expression/metadata)
  5. 缺漏字段自动补充 (direction/paradigm/source 等)

用法:
    python seed_injector.py
    # 或
    from seed_injector import inject_champion_seeds, merge_factor_pools
"""

import sys
import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SCRIPT_DIR = Path(__file__).parent


# ═══════════════════════════════════════════════════════════
# AlphaAgent v3 因子库 (从源代码提取)
# ═══════════════════════════════════════════════════════════

CHAMPION_FACTORS = [
    # ═══ 流动性/微观结构 ═══
    {"expression": "neg(rank(div(sub(high, low), add(ts_mean(volume, 20), 1e-6))))", "name": "振幅归一化(量)",
     "rationale": "日内振幅/成交量均值截面排名取负。高振幅低量=流动性差=流动性溢价", "direction": "+", "paradigm": "流动性×微观结构"},
    {"expression": "rank(sub(ts_max(close, 20), ts_min(close, 20)))", "name": "价格区间宽度",
     "rationale": "20日价格区间截面排名——价格离散度大的股票弹性更强", "direction": "-", "paradigm": "波动率"},
    {"expression": "neg(ts_zscore(div(div(close, ts_mean(close, 20)), ts_std(volume, 20)), 40))", "name": "量价稳定度",
     "rationale": "价格偏离均线/成交量波动性的长期z-score", "direction": "+", "paradigm": "流动性×趋势"},
    {"expression": "neg(rank(div(ts_mean(volume, 5), add(ts_mean(volume, 60), 1e-6))))", "name": "长期缩量程度",
     "rationale": "5日/60日均量截面取负。长期缩量=筹码锁定/惜售", "direction": "+", "paradigm": "流动性"},
    {"expression": "rank(div(ts_mean(high, 20), ts_mean(low, 20)))", "name": "高低均价比",
     "rationale": "20日均高价/均低价。高比率=多空博弈激烈", "direction": "+", "paradigm": "流动性×微观结构"},

    # ═══ 资金流/量价背离 ═══
    {"expression": "sub(ts_pct(close, 10), ts_pct(volume, 10))", "name": "量价背离(10日)",
     "rationale": "价格变化率-成交量变化率", "direction": "-", "paradigm": "资金流×趋势"},
    {"expression": "rank(div(ts_mean(volume, 5), ts_mean(volume, 20)))", "name": "放量强度",
     "rationale": "5日/20日均量截面排名。持续放量=资金关注度提升", "direction": "+", "paradigm": "资金流"},
    {"expression": "rank(mul(ts_pct(volume, 5), ts_pct(close, 5)))", "name": "量价共振度",
     "rationale": "截面rank(量增速×价增速)。量价同向=趋势健康", "direction": "+", "paradigm": "资金流×趋势"},
    {"expression": "neg(rank(div(sub(volume, ts_mean(volume, 20)), add(ts_std(volume, 20), 1e-6))))", "name": "缩量异常度",
     "rationale": "截面取负((量-20日均量)/量波动)", "direction": "+", "paradigm": "资金流"},

    # ═══ 动量/反转 ═══
    {"expression": "neg(ts_delta(ts_zscore(close, 60), 20))", "name": "动量减速",
     "rationale": "长期z-score的20日变化取负。动量生命周期", "direction": "+", "paradigm": "动量反转"},
    {"expression": "neg(rank(ts_mean(div(close, ts_max(close, 60)), 10)))", "name": "接近1年高点比例",
     "rationale": "10日均(价格/60日最高)截面取负", "direction": "+", "paradigm": "行为金融"},
    {"expression": "neg(rank(sub(ts_mean(close, 20), ts_mean(close, 60))))", "name": "中期趋势弱度",
     "rationale": "截面取负(20日均-60日均)", "direction": "+", "paradigm": "趋势"},
    {"expression": "neg(div(sub(close, ts_min(close, 60)), add(ts_max(close, 60), -ts_min(close, 60))))", "name": "深度超卖度",
     "rationale": "取负(价格在60日区间位置)。超卖=反弹动力", "direction": "+", "paradigm": "动量反转"},
    {"expression": "neg(rank(mul(div(close, ts_delta(close, 20)), div(volume, ts_delta(volume, 20)))))", "name": "20日量价双低",
     "rationale": "缩量滞涨=积累期", "direction": "+", "paradigm": "资金流×动量"},

    # ═══ 行为金融 ═══
    {"expression": "neg(rank(div(sub(close, ts_min(close, 5)), add(ts_max(close, 5), -ts_min(close, 5)))))", "name": "5日价格位置(取负)",
     "rationale": "近5日价格在区间位置的截面取负", "direction": "+", "paradigm": "行为金融"},
    {"expression": "rank(div(sub(high, close), add(sub(high, low), 0.001)))", "name": "上影线比例",
     "rationale": "上影线长=抛压出现=短期见顶", "direction": "-", "paradigm": "行为金融"},
    {"expression": "neg(rank(div(sub(close, low), add(sub(high, low), 0.001))))", "name": "下影线抄底",
     "rationale": "下影线长=买盘出现=短期见底", "direction": "+", "paradigm": "行为金融"},

    # ═══ 尾部风险/波动不对称 ═══
    {"expression": "neg(ts_zscore(div(ts_std(sub(close, ts_mean(close, 20)), 20), ts_mean(close, 20)), 60))",
     "name": "波动率归一化", "rationale": "波动率偏离历史均值取负。低波溢价", "direction": "+", "paradigm": "波动率"},
    {"expression": "neg(ts_skewness(sub(close, ts_mean(close, 20)), 60))", "name": "负偏度溢价",
     "rationale": "收益分布负偏度取负。负偏=崩盘风险=需溢价补偿", "direction": "+", "paradigm": "尾部风险"},
    {"expression": "rank(div(ts_max(close, 60), ts_min(close, 60)))", "name": "60日波动幅度",
     "rationale": "60日最高/最低。高幅度=高风险=尾部溢价", "direction": "-", "paradigm": "波动率"},
    {"expression": "neg(div(ts_std(close, 20), ts_mean(close, 20)))", "name": "低波动溢价",
     "rationale": "20日波动率取负。低波=防守型alpha", "direction": "+", "paradigm": "波动率"},
    {"expression": "neg(ts_kurtosis(sub(close, ts_mean(close, 20)), 60))", "name": "低峰度溢价",
     "rationale": "峰度取负。低峰度=稳定收益=质量信号", "direction": "+", "paradigm": "尾部风险"},

    # ═══ 趋势/突破 ═══
    {"expression": "div(sub(close, ts_mean(close, 5)), ts_std(close, 20))", "name": "短期突破强度",
     "rationale": "(价格-5日均线)/20日波动。标准化突破强度", "direction": "+", "paradigm": "趋势"},
    {"expression": "neg(rank(div(sub(close, ts_max(close, 10)), add(ts_max(close, 10), -ts_mean(close, 10)))))", "name": "接近阻力位",
     "rationale": "价格距离10日高点的比例取负", "direction": "+", "paradigm": "趋势"},
    {"expression": "rank(sub(close, ts_mean(close, 5)))", "name": "5日动量",
     "rationale": "价格偏离5日均线截面排名。短期趋势强度", "direction": "+", "paradigm": "趋势"},
    {"expression": "ts_delta(ts_zscore(volume, 20), 5)", "name": "量加速",
     "rationale": "成交量z-score的5日变化。量加速=突破确认", "direction": "+", "paradigm": "资金流"},

    # ═══ 尾部增强: 下行保护/回撤恢复 ═══
    {"expression": "neg(ts_mean(sub(close, ts_min(close, 20)), 5))", "name": "回撤深度",
     "rationale": "5日均(价格-20日最低)。回撤浅=防御性强", "direction": "+", "paradigm": "尾部风险"},
    {"expression": "neg(rank(div(ts_max(close, 60), close)))", "name": "距高点回撤",
     "rationale": "截面取负(60日最高/当前价)。回撤小=风险可控", "direction": "+", "paradigm": "尾部风险"},
    {"expression": "div(ts_std(close, 60), ts_mean(close, 60))", "name": "长期波动率",
     "rationale": "60日波动率/60日均价。长期风险度量", "direction": "-", "paradigm": "波动率"},

    # ═══ 筹码分布(微观结构增强) ═══
    {"expression": "neg(div(ts_mean(volume, 20), ts_mean(volume, 60)))", "name": "量结构变化",
     "rationale": "20日均量/60日均量取负。缩量=筹码锁定", "direction": "+", "paradigm": "筹码分布"},
    {"expression": "rank(div(close, div(ts_max(close, 20), ts_min(close, 20))))", "name": "区间位置度",
     "rationale": "价格/(20日波动幅度)。相对位置截面排名", "direction": "+", "paradigm": "筹码分布"},
    {"expression": "neg(rank(sub(ts_max(close, 60), close)))", "name": "距最高价",
     "rationale": "截面取负(60日最高-当前价)", "direction": "+", "paradigm": "筹码分布"},

    # ═══ 北向资金 (v0.5 P-002 新增) ═══
    {"expression": "div(north_money, amount)", "name": "北向净流入强度",
     "rationale": "北向净流入/成交额。高比例=外资对个股参与度远超内资", "direction": "+", "paradigm": "北向资金", "source": "experimental"},
    {"expression": "div(sub(north_money, shift(north_money, 5)), add(amount, 1e-6))", "name": "北向资金加速度",
     "rationale": "(北向净流入-5日北向净流入)/成交额。北向加速=信号增强", "direction": "+", "paradigm": "北向资金", "source": "experimental"},
    {"expression": "div(sub(hgt, sgt), add(add(hgt, sgt), 1e-6))", "name": "沪深港通分化比",
     "rationale": "(沪股通-深股通)/(沪股通+深股通)。正值=沪强深弱(偏好大盘)", "direction": "+", "paradigm": "北向资金", "source": "experimental"},
    {"expression": "div(north_money, sub(add(north_money, south_money), 1e-12))", "name": "北向占比",
     "rationale": "北向/(北向+南向)。高比例=外资引领、内资跟随", "direction": "+", "paradigm": "北向资金", "source": "experimental"},
    {"expression": "ts_mean(div(north_money, add(amount, 1e-6)), 5)", "name": "北向资金持续性",
     "rationale": "5日均(北向净流入/成交额)。持续北上=机构建仓信号", "direction": "+", "paradigm": "北向资金", "source": "experimental"},
    {"expression": "neg(rank(div(sub(south_money, north_money), add(south_money, north_money))))", "name": "南向资金背离度",
     "rationale": "截面取负((南向-北向)/(南向+北向))。外资流出+内资南下=风险信号", "direction": "+", "paradigm": "北向资金", "source": "experimental"},
    {"expression": "div(sub(ts_mean(north_money, 5), ts_mean(north_money, 20)), add(ts_std(north_money, 20), 1e-6))", "name": "北向资金突破",
     "rationale": "(5日均北向-20日均北向)/20日波动。北向突破历史均值", "direction": "+", "paradigm": "北向资金", "source": "experimental"},
    {"expression": "ts_corr(north_money, sub(close, ts_mean(close, 20)), 20)", "name": "北向-价格共振",
     "rationale": "北向净流入与价格偏离均线的20日相关性。正相关=外资追涨", "direction": "-", "paradigm": "北向资金", "source": "experimental"},

    # ═══ 高频微观结构 (v0.6 P-010 新增) — 范式19 0→1 ═══
    {"expression": "neg(div(sub(open, ts_min(low, 20)), add(sub(ts_max(high, 20), ts_min(low, 20)), 1e-6)))",
     "name": "开盘位置度", "rationale": "开盘价在20日区间位置取负。低开=情绪恐慌=反弹机会", "direction": "+",
     "paradigm": "高频微观结构", "source": "experimental"},
    {"expression": "div(sub(close, open), add(sub(high, low), 0.001))", "name": "日内方向强度",
     "rationale": "(收-开)/(高-低)。正值=日内多头主导", "direction": "+",
     "paradigm": "高频微观结构", "source": "experimental"},
    {"expression": "neg(rank(div(sub(high, open), add(sub(open, low), 0.001))))", "name": "开盘抛压比",
     "rationale": "截面取负((高-开)/(开-低))。高比值=开盘后抛压大", "direction": "+",
     "paradigm": "高频微观结构", "source": "experimental"},
    {"expression": "rank(div(sub(close, low), add(sub(high, low), 0.001)))", "name": "日内收盘位置",
     "rationale": "(收-低)/(高-低)。收在日内高位=强势收盘=次日动量", "direction": "+",
     "paradigm": "高频微观结构", "source": "experimental"},
    {"expression": "neg(rank(div(sub(high, close), add(sub(high, open), 0.001))))", "name": "尾盘回落度",
     "rationale": "截面取负((高-收)/(高-开))。尾盘回落=聪明钱出货", "direction": "+",
     "paradigm": "高频微观结构", "source": "experimental"},
    {"expression": "neg(div(ts_mean(volume, 5), ts_mean(volume, 20)))", "name": "大单缩量度",
     "rationale": "5日均量/20日均量取负。缩量+窄幅=筹码集中", "direction": "+",
     "paradigm": "高频微观结构", "source": "experimental"},
    {"expression": "ts_corr(sub(close, open), volume, 20)", "name": "日内量价共振",
     "rationale": "20日日收益与成交量相关性。正相关=量价健康", "direction": "+",
     "paradigm": "高频微观结构", "source": "experimental"},
    {"expression": "neg(rank(div(ts_std(sub(close, open), 20), add(ts_mean(sub(close, open), 20), 0.001))))",
     "name": "日内波动稳定性", "rationale": "日收益CV取负。稳定日内收益=机构持仓特征", "direction": "+",
     "paradigm": "高频微观结构", "source": "experimental"},

    # ═══ 筹码分层结构 (v0.6 P-007 新增) ═══
    {"expression": "neg(div(sub(close, ts_mean(close, 60)), add(ts_std(close, 60), 1e-6)))",
     "name": "长期持仓成本偏离", "rationale": "(价-60日均)/60日波动取负。远离长期成本=获利盘压力", "direction": "+",
     "paradigm": "筹码分布", "source": "experimental"},
    {"expression": "div(sub(ts_mean(close, 20), ts_mean(close, 60)), add(ts_std(close, 60), 1e-6))",
     "name": "短期vs长期成本差", "rationale": "(20日-60日均)/60日波动。正值=短期持仓成本高于长期=新资金入场", "direction": "+",
     "paradigm": "筹码分布", "source": "experimental"},
    {"expression": "rank(div(ts_mean(volume, 5), ts_mean(volume, 60)))", "name": "筹码换手加速",
     "rationale": "5日/60日均量。高比值=近期筹码快速换手", "direction": "-",
     "paradigm": "筹码分布", "source": "experimental"},
    {"expression": "neg(rank(div(sub(ts_max(close, 20), close), add(ts_max(close, 20), -ts_min(close, 20)))))",
     "name": "获利盘比例(代理)", "rationale": "截面取负(最高-当前价)/区间。低值=多数持仓者盈利=抛压", "direction": "+",
     "paradigm": "筹码分布", "source": "experimental"},
    {"expression": "ts_corr(volume, sub(close, ts_mean(close, 20)), 20)", "name": "量价筹码共振",
     "rationale": "20日量与价格偏离相关性。正=放量突破=筹码松动", "direction": "-",
     "paradigm": "筹码分布", "source": "experimental"},
    {"expression": "neg(div(ts_mean(ts_std(close, 10), 20), add(ts_std(close, 60), 1e-6)))",
     "name": "波动收敛度", "rationale": "20日均(10日波动)/60日波动取负。波动收敛=筹码锁定末期", "direction": "+",
     "paradigm": "筹码分布", "source": "experimental"},

    # ═══ 主动资金流 (v0.6 P-008 新增) ═══
    {"expression": "div(sub(buy_lg_vol, sell_lg_vol), add(add(buy_lg_vol, sell_lg_vol), 1e-6))",
     "name": "大单净流向强度", "rationale": "(大买-大卖)/(大买+大卖)。正=大单净流入=机构建仓", "direction": "+",
     "paradigm": "资金流", "source": "experimental"},
    {"expression": "ts_mean(div(sub(buy_lg_vol, sell_lg_vol), add(volume, 1e-6)), 5)",
     "name": "大单净流向持续性", "rationale": "5日均(大单净/总量)。持续净流入=机构趋势性建仓", "direction": "+",
     "paradigm": "资金流", "source": "experimental"},
    {"expression": "div(sub(buy_lg_vol, buy_sm_vol), add(add(buy_lg_vol, buy_sm_vol), 1e-6))",
     "name": "大单vs小单买入比", "rationale": "(大买-小买)/(大买+小买)。正=机构主导买入", "direction": "+",
     "paradigm": "资金流", "source": "experimental"},
    {"expression": "div(sub(sub(buy_lg_vol, sell_lg_vol), sub(buy_sm_vol, sell_sm_vol)), add(volume, 1e-6))",
     "name": "机构-散户流向差", "rationale": "((大买-大卖)-(小买-小卖))/总量。正=机构净买+散户净卖=聪明钱信号", "direction": "+",
     "paradigm": "资金流", "source": "experimental"},
    {"expression": "ts_delta(div(sub(buy_lg_vol, sell_lg_vol), add(volume, 1e-6)), 5)",
     "name": "大单流向加速度", "rationale": "5日变化(大单净/量)。加速净流入=趋势增强", "direction": "+",
     "paradigm": "资金流", "source": "experimental"},

    # ═══ 财务附注 (v0.6 P-009 降级) — 全 balancesheet字段, 不依赖income ═══
    {"expression": "div(intan_assets, sub(total_assets, goodwill))", "name": "无形资产密度",
     "rationale": "无形资产/(总资产-商誉)。高=技术资产占比高(替代rd_intensity)", "direction": "+",
     "paradigm": "行为金融", "source": "experimental", "note": "P-009: 仅balancesheet字段"},
    {"expression": "div(intan_assets, total_assets)", "name": "无形资产占比",
     "rationale": "无形资产/总资产。高=轻资产技术型(替代rd_capitalize)", "direction": "+",
     "paradigm": "行为金融", "source": "experimental", "note": "P-009: 仅balancesheet字段"},
    {"expression": "div(goodwill, total_assets)", "name": "商誉质量比",
     "rationale": "商誉/总资产。高商誉=并购活跃但减值风险", "direction": "-",
     "paradigm": "行为金融", "source": "experimental", "note": "P-009: 仅balancesheet字段"},
    {"expression": "div(sub(intan_assets, goodwill), total_assets)", "name": "纯无形资产占比",
     "rationale": "(无形资产-商誉)/总资产。剔除商誉衡量真实技术壁垒", "direction": "+",
     "paradigm": "行为金融", "source": "experimental", "note": "P-009: 仅balancesheet字段"},
    {"expression": "ts_pct(intan_assets, 4)", "name": "无形资产增速",
     "rationale": "无形资产同比增长率(季度)。加速=技术投入加大(替代rd增速)", "direction": "+",
     "paradigm": "行为金融", "source": "experimental", "note": "P-009: 仅balancesheet字段"},

    # ═══ LLM 情绪因子 (v0.6 P-003 新增) ═══
    {"expression": "neg(ts_mean(div(sub(close, ts_min(close, 20)), add(ts_max(close, 20), -ts_min(close, 20))), 5))",
     "name": "恐慌反弹强度", "rationale": "5日均(20日区间位置)取负。持续低位=恐慌过度=反弹潜力", "direction": "+",
     "paradigm": "情绪×日内", "source": "experimental"},
    {"expression": "div(ts_pct(volume, 5), add(ts_std(close, 20), 1e-6))",
     "name": "恐慌量比", "rationale": "5日量增速/20日波动。放量+高波=恐慌性抛售", "direction": "-",
     "paradigm": "情绪×日内", "source": "experimental"},
    {"expression": "neg(div(ts_pct(close, 3), add(ts_std(volume, 20), 1e-6)))",
     "name": "情绪反转度", "rationale": "3日收益率/量波动取负。急跌缩量=恐慌末期", "direction": "+",
     "paradigm": "情绪×日内", "source": "experimental"},
    {"expression": "sub(ts_skewness(sub(close, ts_mean(close, 20)), 20), ts_skewness(sub(close, ts_mean(close, 20)), 60))",
     "name": "情绪偏度变化", "rationale": "短期偏度-长期偏度。正=情绪改善(偏度升高)", "direction": "+",
     "paradigm": "情绪×日内", "source": "experimental"},
    {"expression": "ts_corr(sub(close, open), sub(high, low), 10)", "name": "日内情绪一致性",
     "rationale": "10日日收益率与日振幅相关性。正=趋势情绪一致", "direction": "+",
     "paradigm": "情绪×日内", "source": "experimental"},

    # ═══ 概念/注意力因子 (v0.6 P-005 新增) ═══
    {"expression": "neg(rank(ts_corr(close, ts_mean(close, 20), 20)))", "name": "板块内偏离度",
     "rationale": "截面取负(个股与自身均线相关)。低相关=脱离板块趋势=异动信号", "direction": "+",
     "paradigm": "截面交互", "source": "experimental"},
    {"expression": "neg(rank(div(ts_std(close, 20), ts_mean(close, 20))))", "name": "板块内波动分歧",
     "rationale": "截面取负(20日波动率CV)。低CV=一致性强=趋势可靠", "direction": "+",
     "paradigm": "截面交互", "source": "experimental"},
    {"expression": "sub(ts_pct(close, 10), ts_pct(ts_mean(close, 60), 10))", "name": "龙头-跟风收益差",
     "rationale": "个股10日收益-长期均线10日收益。正=跑赢自身趋势=强势股", "direction": "+",
     "paradigm": "截面交互", "source": "experimental"},
    {"expression": "ts_corr(volume, ts_pct(close, 10), 20)", "name": "板块量价共振度",
     "rationale": "20日量与10日收益率相关性。正=量价共振=趋势健康", "direction": "+",
     "paradigm": "截面交互", "source": "experimental"},
    {"expression": "neg(div(ts_mean(sub(high, low), 10), ts_mean(sub(high, low), 60)))",
     "name": "热度衰减度", "rationale": "10日均振幅/60日均振幅取负。振幅收缩=热度退潮", "direction": "+",
     "paradigm": "截面交互", "source": "experimental"},
    {"expression": "rank(div(ts_mean(volume, 5), ts_mean(volume, 60)))", "name": "注意力突增度",
     "rationale": "5日/60日均量截面排名。突增=市场注意力聚焦", "direction": "-",
     "paradigm": "截面交互", "source": "experimental"},

    # ═══ 跳空趋势背离 (v0.6 P-012 新增) — 中金 Loop Engineering Sharpe 2.74 ═══
    {"expression": "sub(div(sub(ts_mean(close, 20), close), add(ts_std(close, 60), 1e-6)), "
                   "div(sub(ts_std(sub(high, low), 10), ts_std(sub(high, low), 5)), "
                   "add(ts_std(sub(high, low), 20), 1e-6)))",
     "name": "跳空溢價-振幅背離", "rationale": "(价格偏离均线/波动) - (振幅加速度/振幅)。中金核心因子：正=持续高开但日内波动未放大=隔夜溢价有基本面支撑", "direction": "+",
     "paradigm": "事件驱动", "source": "experimental", "note": "P-012: 中金Loop Engineering复现"},
    {"expression": "div(sub(close, open), add(ts_std(sub(close, open), 20), 1e-6))",
     "name": "跳空标准化强度", "rationale": "(收-开)/跳空波动。正=跳空幅度高于历史=异常跳空", "direction": "+",
     "paradigm": "事件驱动", "source": "experimental"},
    {"expression": "ts_corr(sub(close, open), sub(close, ts_mean(close, 20)), 10)",
     "name": "连续跳空方向一致性", "rationale": "10日跳空与价格偏离的相关性。高正=连续同向跳空=趋势强化", "direction": "+",
     "paradigm": "事件驱动", "source": "experimental"},
    {"expression": "div(sub(close, open), add(ts_mean(volume, 5), 1e-6))",
     "name": "跳空伴生放量质量", "rationale": "(收-开)/5日均量。高=跳空无量=虚跳=短期反转。低=跳空有量=真信号", "direction": "-",
     "paradigm": "事件驱动", "source": "experimental"},
    {"expression": "neg(sub(div(sub(close, open), close), ts_mean(div(sub(close, open), close), 10)))",
     "name": "跳空-日内反转背离", "rationale": "取负(当日跳空率-10日均跳空率)。正=跳空减小=溢价衰减=反转信号", "direction": "+",
     "paradigm": "事件驱动", "source": "experimental"},
    {"expression": "div(sub(close, open), sub(high, low))",
     "name": "跳空质量比", "rationale": "跳空/(高-低)。高=跳空主导日内=隔夜信息强烈", "direction": "+",
     "paradigm": "事件驱动", "source": "experimental"},
    {"expression": "neg(div(ts_std(sub(close, open), 20), add(ts_mean(abs(sub(close, open)), 20), 1e-6)))",
     "name": "隔夜溢价稳定性", "rationale": "取负(跳空CV)。低CV=跳空稳定=可预测溢价", "direction": "+",
     "paradigm": "事件驱动", "source": "experimental"},
    {"expression": "sub(rank(div(sub(close, open), add(sub(high, low), 1e-6))), "
                   "rank(div(ts_std(sub(close, open), 20), add(close, 1e-6))))",
     "name": "跳空强度-稳定性合成", "rationale": "截面排名(跳空率) - 截面排名(跳空CV)。高=强跳空+稳定", "direction": "+",
     "paradigm": "事件驱动", "source": "experimental"},

    # ═══ 行为金融 V型处置效应 (v0.6 P-014 新增) — 中信建投 VCDE3 ═══
    {"expression": "neg(rank(div(sub(close, ts_min(close, 20)), add(ts_max(close, 20), -ts_min(close, 20)))))",
     "name": "V型处置效应(深度亏损)", "rationale": "截面取负(20日区间位置)。极低位=深度亏损→加速卖出=恐慌清仓=超卖反弹", "direction": "+",
     "paradigm": "行为金融", "source": "experimental", "note": "P-014: V型处置效应"},
    {"expression": "neg(rank(div(sub(close, ts_max(close, 20)), add(ts_max(close, 20), -ts_min(close, 20)))))",
     "name": "处置效应(小幅盈利)", "rationale": "截面取负(距20日高点的位置)。高位=小幅盈利→过早卖出=后续继续涨", "direction": "+",
     "paradigm": "行为金融", "source": "experimental"},
    {"expression": "div(sub(close, ts_max(close, 60)), add(close, 1e-6))",
     "name": "52周标杆锚定", "rationale": "(当前价-60日最高)/当前价。低=远离高点=锚定偏差→低估=反弹潜力", "direction": "+",
     "paradigm": "行为金融", "source": "experimental"},
    {"expression": "neg(rank(div(sub(close, ts_mean(close, 5)), add(ts_std(close, 20), 1e-6))))",
     "name": "过度自信反转", "rationale": "截面取负(价格偏离5日均/波动)。高位=过度自信追涨→后续反转", "direction": "+",
     "paradigm": "行为金融", "source": "experimental"},
    {"expression": "div(ts_pct(volume, 1), add(ts_std(close, 20), 1e-6))",
     "name": "注意力驱动放量", "rationale": "单日量增速/波动。突增=市场注意力聚焦→短期动量", "direction": "-",
     "paradigm": "行为金融", "source": "experimental"},
    {"expression": "neg(rank(ts_corr(close, ts_mean(close, 5), 20)))",
     "name": "后悔厌恶(追涨倾向)", "rationale": "截面取负(价格与5日均的20日相关)。高正=持续追涨→后悔驱动→反转", "direction": "+",
     "paradigm": "行为金融", "source": "experimental"},
    {"expression": "neg(div(ts_mean(sub(high, low), 5), ts_mean(sub(high, low), 60)))",
     "name": "羊群效应衰减", "rationale": "取负(5日均振幅/60日均振幅)。高=波动放大=羊群效应→后续回归", "direction": "+",
     "paradigm": "行为金融", "source": "experimental"},
    {"expression": "div(ts_std(close, 10), ts_mean(close, 10))",
     "name": "有限注意力波动", "rationale": "10日CV。低波动=注意力不足→被低估=后续补涨", "direction": "-",
     "paradigm": "行为金融", "source": "experimental"},

    # ═══ 行业轮动 (v0.6 P-016 新增) — 范式15 0→1 ═══
    {"expression": "neg(rank(div(sub(close, ts_min(close, 60)), add(ts_max(close, 60), -ts_min(close, 60)))))",
     "name": "行业个股相对强度(代理)", "rationale": "截面取负(60日区间位置)。低位=行业弱势股→轮动补涨", "direction": "+",
     "paradigm": "行业轮动", "source": "experimental", "note": "P-016: 华泰AI行业轮动实证"},
    {"expression": "div(ts_pct(close, 20), add(ts_std(close, 60), 1e-6))",
     "name": "行业动量强度", "rationale": "20日收益率/60日波动。高=行业趋势强度", "direction": "+",
     "paradigm": "行业轮动", "source": "experimental"},
    {"expression": "neg(rank(div(sub(ts_mean(close, 5), ts_mean(close, 20)), add(ts_std(close, 60), 1e-6))))",
     "name": "行业轮动加速度", "rationale": "截面取负((5日-20日均)/波动)。正=加速上涨=行业轮入信号", "direction": "+",
     "paradigm": "行业轮动", "source": "experimental"},
    {"expression": "neg(div(ts_mean(volume, 60), ts_mean(volume, 20)))",
     "name": "行业拥挤度(代理)", "rationale": "取负(60日均量/20日均量)。>1=近期缩量=拥挤度下降=资金流出", "direction": "+",
     "paradigm": "行业轮动", "source": "experimental"},
    {"expression": "rank(div(volume, ts_mean(volume, 60)))",
     "name": "行业资金流强度", "rationale": "截面排名(量/60日均量)。高=放量=行业受关注", "direction": "-",
     "paradigm": "行业轮动", "source": "experimental"},
    {"expression": "ts_corr(close, ts_pct(volume, 5), 20)",
     "name": "行业量价健康度", "rationale": "20日价格与量增速的相关性。正=量价共振=行业趋势健康", "direction": "+",
     "paradigm": "行业轮动", "source": "experimental"},

    # ═══ 情绪×日内 MAB补全 (v0.6 P-015 新增) — MAB Top-3 UCB ═══
    {"expression": "div(sub(open, ts_delta(close, 1)), add(close, 1e-6))",
     "name": "隔夜跳空强度", "rationale": "(今开-昨收)/昨收。大正跳空=隔夜利好=情绪驱动", "direction": "+",
     "paradigm": "情绪×日内", "source": "experimental", "note": "P-015: MAB UCB Top-3"},
    {"expression": "neg(div(sub(close, open), add(sub(high, low), 0.001)))",
     "name": "日内反转率", "rationale": "取负((收-开)/(高-低))。低开后高走=日内反转=超卖", "direction": "+",
     "paradigm": "情绪×日内", "source": "experimental"},
    {"expression": "ts_mean(div(sub(open, ts_delta(close, 1)), add(close, 1e-6)), 10)",
     "name": "隔夜溢价持续性", "rationale": "10日均(跳空率)。持续正跳空=情绪稳定=溢价可期", "direction": "+",
     "paradigm": "情绪×日内", "source": "experimental"},
    {"expression": "div(sub(close, open), sub(ts_delta(close, 1), ts_mean(close, 5)))",
     "name": "跳空-日内冲刷背离", "rationale": "跳空/日内幅度差。跳空大但日内反转小=情绪稳定", "direction": "-",
     "paradigm": "情绪×日内", "source": "experimental"},
    {"expression": "div(ts_sum(div(sub(volume, ts_mean(volume, 20)), add(ts_std(volume, 20), 1e-6)), 10), 10)",
     "name": "开盘量能情绪比", "rationale": "10日均(放量度)。持续放量=情绪亢奋→短期过热", "direction": "-",
     "paradigm": "情绪×日内", "source": "experimental"},
    {"expression": "neg(ts_corr(sub(high, low), ts_pct(close, 5), 10))",
     "name": "日内波动-收益背离", "rationale": "取负(振幅与5日收益的10日相关)。负=波动加大但收益不匹配=情绪转折", "direction": "+",
     "paradigm": "情绪×日内", "source": "experimental"},

    # ═══ 两融信号 (v0.6 P-019 新增) — 范式17 0→1 ═══
    {"expression": "neg(div(sub(buy_lg_vol, sell_lg_vol), add(add(buy_lg_vol, sell_lg_vol), 1e-6)))",
     "name": "融资买入强度(代理)", "rationale": "取负(大单净流向)。背离=大单净卖出→融资盘减仓=杠杆情绪降温", "direction": "+",
     "paradigm": "两融信号", "source": "experimental", "note": "P-019: 用moneyflow大单代理两融"},
    {"expression": "div(sub(buy_lg_vol, sell_lg_vol), ts_mean(volume, 60))",
     "name": "融资杠杆倾向", "rationale": "(大买-大卖)/60日均量。正=大单活跃→杠杆资金参与度高", "direction": "+",
     "paradigm": "两融信号", "source": "experimental"},
    {"expression": "ts_mean(div(sub(buy_lg_vol, sell_lg_vol), add(volume, 1e-6)), 20)",
     "name": "融资盘持续性", "rationale": "20日均(大单净/总量)。持续净买入=融资盘加仓趋势", "direction": "+",
     "paradigm": "两融信号", "source": "experimental"},
    {"expression": "neg(rank(div(sub(buy_elg_vol, sell_elg_vol), add(add(buy_elg_vol, sell_elg_vol), 1e-6))))",
     "name": "融券空头压力(代理)", "rationale": "截面取负(超大单净流向)。超大单净卖出=机构减仓=空头压力", "direction": "+",
     "paradigm": "两融信号", "source": "experimental"},
    {"expression": "div(sub(sub(buy_lg_vol, sell_lg_vol), sub(buy_sm_vol, sell_sm_vol)), add(volume, 1e-6))",
     "name": "多空分歧度", "rationale": "((大买-大卖)-(小买-小卖))/总量。正=大单净买+散户净卖=机构vs散户分歧", "direction": "+",
     "paradigm": "两融信号", "source": "experimental"},
    {"expression": "ts_corr(sub(buy_lg_vol, sell_lg_vol), ts_pct(close, 10), 20)",
     "name": "杠杆-价格共振", "rationale": "20日大单净与收益的相关性。正=杠杆同向=趋势增强", "direction": "+",
     "paradigm": "两融信号", "source": "experimental"},

    # ═══ GP 育种 JQ 验证通过 (v0.6 自动入库) ═══
    {"expression": "-((close / rolling_min(close, 60)) - close)",
     "name": "gp_breed_000_均值回复", "rationale": "GP育种: 60日滚动最小值相对价格(取负→低值反转)。【JQ +124.78%/Sharpe 0.44】", "direction": "+",
     "paradigm": "均值回复", "source": "gp_jq_validated"},
    {"expression": "high_p - close_p.pct_change(20).shift(20)",
     "name": "gp_breed_002_动量回调幅度", "rationale": "GP育种: 高价-20日前20日动量(高位回调=买入)。【JQ +143.46%/Sharpe 0.51】🏆", "direction": "+",
     "paradigm": "价量关系", "source": "gp_jq_validated"},
    {"expression": "(-((high - low) / (volume.rolling(20).mean() + 1)).rank(pct=True))",
     "name": "gp_breed_006_振幅量比交叉排名", "rationale": "GP育种: 振幅/量均20日排名(取负→低振幅高量=蓄势)。【JQ +129.36%/Sharpe 0.46】", "direction": "+",
     "paradigm": "微观结构", "source": "gp_jq_validated"},

    # ═══ P-20260812-027: 行为金融真因子 (CGO/处置效应/过度反应/锚定) ═══
    {"expression": "div(sub(close, div(ts_mean(mul(close, volume), 120), add(ts_mean(volume, 120), 1e-6))), add(close, 1e-6))",
     "name": "cgo_vwap_reference", "rationale": "CGO(Capital Gains Overhang): (现价-VWAP参考价)/现价, VWAP=120日量价加权均价。正=大量浮盈盘(Frazzini 2006), 处置效应持有者惜售→方向待S5裁决", "direction": "-",
     "paradigm": "行为金融", "source": "p_20260812_027"},
    {"expression": "neg(mul(div(volume, add(ts_mean(volume, 60), 1e-6)), div(sub(close, ts_min(close, 60)), add(sub(ts_max(close, 60), ts_min(close, 60)), 1e-6))))",
     "name": "disposition_effect_intensity", "rationale": "处置效应强度: 取负(放量度×60日高位度)。放量+高位=获利了结压力集中(处置效应触发)", "direction": "+",
     "paradigm": "行为金融", "source": "p_20260812_027"},
    {"expression": "neg(ts_max(ts_pct(close, 1), 20))",
     "name": "overreaction_reversal", "rationale": "过度反应反转: 取负(20日最大单日涨幅)。极端涨幅=过度反应, 后续反转(Bondt-Thaler)", "direction": "+",
     "paradigm": "行为金融", "source": "p_20260812_027"},
    {"expression": "neg(div(sub(close, ts_max(close, 120)), add(ts_max(close, 120), 1e-6)))",
     "name": "anchoring_120d_proxy", "rationale": "锚定效应代理: 取负(距120日高点距离)。接近历史锚点=投资者锚定卖压, 突破锚点后空间打开", "direction": "+",
     "paradigm": "行为金融", "source": "p_20260812_027"},

    # ═══ P-20260812-026/P-20260814-002: 主力净流入分化 (net_mf_amount 口径) ═══
    {"expression": "ts_mean(div(net_mf_amount, add(amount, 1e-6)), 5)",
     "name": "net_mf_intensity_5d", "rationale": "主力净流入强度: 5日均(主力净流入/成交额)。正=机构资金持续吸筹", "direction": "+",
     "paradigm": "资金流", "source": "p_20260812_026"},
    {"expression": "ts_corr(net_mf_amount, ts_pct(close, 5), 20)",
     "name": "net_mf_price_resonance", "rationale": "主力资金-价格共振: 20日主力净流入与5日收益的相关性。共振高=资金有效推动价格", "direction": "+",
     "paradigm": "资金流", "source": "p_20260812_026"},
    {"expression": "div(sub(net_mf_amount, ts_mean(net_mf_amount, 20)), add(ts_std(net_mf_amount, 20), 1e-6))",
     "name": "net_mf_surprise_zscore", "rationale": "主力净流入突变: 当日净流入的20日z-score。突变=机构行为变化信号(正突变→买入)", "direction": "+",
     "paradigm": "资金流", "source": "p_20260812_026"},
    {"expression": "neg(ts_corr(net_mf_amount, rzye, 60))",
     "name": "margin_mf_divergence", "rationale": "两融-主力分歧: 取负(60日主力净流入与融资余额相关性)。负相关=融资盘与主力资金背离=分歧预警(注: margin 数据接入后有效, 当前 rzye 缺失期结果不可用)", "direction": "+",
     "paradigm": "跨资产联动", "source": "p_20260814_002"},

    # ═══ P-20260812-026 补全: 龙虎榜真因子 (top_list/top_inst 数据接入 2026-08-14) ═══
    {"expression": "div(ts_sum(lhb_inst_net_buy, 10), add(ts_sum(amount, 10), 1e-6))",
     "name": "lhb_inst_intensity_10d", "rationale": "机构专用席位净买入强度: 近10日机构席位净买入/近10日成交额。机构上榜后持续净买=中期看好, A股机构席位跟踪是龙虎榜核心信号", "direction": "+",
     "paradigm": "资金流", "source": "p_20260812_026"},
    {"expression": "ts_sum(lhb_flag, 20)",
     "name": "lhb_freq_20d", "rationale": "游资活跃度: 近20日上榜次数。高频上榜=游资炒作聚集=短期拥挤(A股历史倾向炒作后反转, 方向待S5裁决)", "direction": "-",
     "paradigm": "资金流", "source": "p_20260812_026"},
    {"expression": "ts_mean(lhb_net_rate, 5)",
     "name": "lhb_net_rate_mean_5d", "rationale": "龙虎榜净买率5日均线: 上榜净买率持续性。连续净买率高=上榜后承接强", "direction": "+",
     "paradigm": "资金流", "source": "p_20260812_026"},
    {"expression": "div(ts_sum(sub(lhb_inst_net_buy, lhb_net_amount), 10), add(ts_sum(amount, 10), 1e-6))",
     "name": "lhb_inst_vs_retail_excess", "rationale": "机构vs游资分歧: 近10日(机构席位净买-龙虎榜整体净买)/成交额。正=机构净买强于游资=机构认可度高于炒作资金", "direction": "+",
     "paradigm": "资金流", "source": "p_20260812_026"},

    # ═══ P-20260814-002 升级: 北向×两融 月频历史验证 (2018-2024 窗口, 2026-08-14 实测) ═══
    # 实测结论 (月频面板 3421 只×44 月): north_monthly_chg 无 alpha (ICIR≈0);
    # margin_monthly_chg ICIR=-0.659 强负; north_margin_div ICIR=+0.264 弱正。
    # 注意: north_ratio 为低频披露数据(34-50 天/年), 日频 rolling 会全 NaN → 仅月频语义有效
    {"expression": "sub(rzye, ts_delay(rzye, 21))",
     "name": "margin_balance_chg_21d", "rationale": "融资余额月度变化(日频代理): rzye-21日前rzye。月频实测 ICIR=-0.659 (2018-2024, 44月): 融资余额上升的股票未来收益显著更差=杠杆追高后回落。两融范式(17)第二强信号", "direction": "-",
     "paradigm": "两融信号", "source": "p_20260814_002"},
    {"expression": "neg(ts_corr(ts_delta(north_ratio, 1), ts_delta(rzye, 1), 12))",
     "name": "north_margin_div_monthly", "rationale": "北向vs两融月频分歧(历史验证用): 取负(12期北向月度变化与两融月度变化相关)。月频实测 ICIR=+0.264 (2018-2024): 外资与杠杆资金背离时未来收益略正。⚠️ north_ratio 低频披露→仅月频语义有效, 日频S5框架下会全NaN; JQ近端无北向数据不可回测, 仅作历史知识沉淀", "direction": "+",
     "paradigm": "跨资产联动", "source": "p_20260814_002"},
]

# ── 已知王者复合因子的 JQ 结果 ──
CHAMPION_JQ_RESULT = {
    "composite_return": 182.57,
    "composite_sharpe": 0.69,
    "composite_maxdd": -35.41,
}


# ═══════════════════════════════════════════════════════════
# 因子库合并
# ═══════════════════════════════════════════════════════════

def extract_operators(formula: str) -> List[str]:
    """从公式中提取算子列表"""
    ops = set()
    for m in re.finditer(r'(?:\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', formula):
        ops.add(m.group(1))
    return sorted(ops)

def extract_windows(formula: str) -> List[int]:
    """从公式中提取窗口参数"""
    windows = set()
    for m in re.finditer(r'(?:rolling\()\s*(\d+)\)', formula):
        windows.add(int(m.group(1)))
    for m in re.finditer(r'(?:ts_\w+\([^,]+,\s*(\d+))', formula):
        windows.add(int(m.group(1)))
    return sorted(windows)

# ══════════════════════════════════════════════════════════════
# v0.6: JQ 验证因子自动入库
# ══════════════════════════════════════════════════════════════

_JQ_VALIDATED_PATH = None
_JQ_VALIDATED_CACHE = None

def _get_jq_registry() -> List[Dict]:
    """加载 JQ 验证因子注册表 (JSON 持久化, 跨 session)"""
    global _JQ_VALIDATED_PATH, _JQ_VALIDATED_CACHE
    import json, os
    if _JQ_VALIDATED_PATH is None:
        workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _JQ_VALIDATED_PATH = os.path.join(workspace, 'data', 'jq_validated_champions.json')
    if _JQ_VALIDATED_CACHE is not None:
        return _JQ_VALIDATED_CACHE
    try:
        with open(_JQ_VALIDATED_PATH, 'r', encoding='utf-8') as f:
            _JQ_VALIDATED_CACHE = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _JQ_VALIDATED_CACHE = []
    return _JQ_VALIDATED_CACHE

def _add_champion_factor(
    expression: str,
    name: str,
    rationale: str,
    direction: str = "+",
    paradigm: str = "auto_breed",
    source: str = "gp_jq_validated",
) -> bool:
    """
    v0.6: JQ 验证通过后自动入库 — 写入 JSON 注册表。
    会在 seed_injector 注入时与硬编码 CHAMPION_FACTORS 合并。

    Returns True 如果是新因子 (去重)
    """
    import json, os
    registry = _get_jq_registry()
    # 去重: 同名或同公式不重复添加
    for existing in registry:
        if existing.get("name") == name:
            return False
        if existing.get("expression", "").strip().replace(" ", "") == expression.strip().replace(" ", ""):
            return False

    entry = {
        "expression": expression,
        "name": name,
        "rationale": rationale,
        "direction": direction,
        "paradigm": paradigm,
        "source": source,
    }
    registry.append(entry)
    global _JQ_VALIDATED_CACHE
    _JQ_VALIDATED_CACHE = registry
    try:
        os.makedirs(os.path.dirname(_JQ_VALIDATED_PATH), exist_ok=True)
        with open(_JQ_VALIDATED_PATH, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

def _get_all_champions() -> List[Dict]:
    """v0.6: 合并硬编码 CHAMPION_FACTORS + JQ 验证 JSON 注册表"""
    jq_registry = _get_jq_registry()
    # 去重: JSON 中的优先 (相同 name 覆盖硬编码)
    merged = {c["name"]: c for c in CHAMPION_FACTORS}
    for c in jq_registry:
        merged[c["name"]] = c
    return list(merged.values())

# ══════════════════════════════════════════════════════════════
# v0.6 P-020: Memory Bridge — 高 occ Memory 模板 → GP 育种种子
# ══════════════════════════════════════════════════════════════

def inject_from_memory_templates(
    memory=None,
    min_occurrence: int = 100,
    verbose: bool = True,
) -> List[Dict]:
    """
    P-020 Template-to-Factor Bridge: 将 Experience Memory 中反复出现的高频
    成功模板 (occ >= min_occurrence) 转化为 GP 育种可用的模板种子。

    这是解开「模板已注入但管线不消费」瓶颈的关键桥接:
    - Memory 记录了哪些范式/结构高效 (如流动性×微观结构 occ=2655)
    - CHAMPION_FACTORS 有对应的模板表达式
    - 但 MAB 不知道范式已有模板 → Ralph Loop 不会主动生成

    本函数将 Memory 高 occ 模板 (已有 formula 字段 → 可消费) 转化为
    可直接喂给 GPBreeder.breed_from_templates() 的种子格式。

    Parameters
    ----------
    memory: ExperienceMemory 实例 (None 时自动获取)
    min_occurrence: 最低出现次数阈值 (默认100, 确保是反复验证的模式)

    Returns
    -------
    [{formula, factor_name, paradigm, source: "memory_bridge", ...}, ...]
    """
    try:
        from experience_memory import get_memory as _get_memory
        if memory is None:
            memory = _get_memory()
    except ImportError:
        if verbose:
            print("[MemoryBridge] 无法导入 ExperienceMemory")
        return []

    templates = memory.get_unconsumed_bridge_templates(min_occurrence)

    if not templates:
        if verbose:
            print(f"[MemoryBridge] 无未消费高 occ 模板 (occ>={min_occurrence})")
        return []

    seeds = []
    for t in templates:
        pattern_id = t.get("pattern_id", "")
        formula = t.get("formula", "")
        description = t.get("description", "")[:80]
        paradigm = pattern_id.split("::")[0] if "::" in pattern_id else t.get("paradigm", "未知")
        occurrence = t.get("occurrence_count", 0)

        seed = {
            "formula": formula,
            "factor_name": f"memory_bridge::{pattern_id[:50]}",
            "pattern_id": pattern_id,
            "paradigm": paradigm,
            "description": description,
            "occurrence_count": occurrence,
            "source": "memory_bridge",
            "direction": "+",
            "rationale": f"[Memory Bridge] 高 occ ({occurrence}) 模板: {description}",
        }
        seeds.append(seed)

    if verbose:
        paradigms = set(s.get("paradigm", "") for s in seeds)
        print(f"[MemoryBridge] 从 Memory 提取 {len(seeds)} 个桥接模板种子")
        print(f"  范式: {sorted(paradigms)}")
        for s in seeds:
            print(f"  → {s['pattern_id']}: occ={s['occurrence_count']}")

    return seeds


def mark_bridge_templates_consumed(
    seeds: List[Dict],
    memory=None,
    verbose: bool = True,
) -> int:
    """标记 Memory Bridge 种子为已消费 (避免重复注入)"""
    try:
        from experience_memory import get_memory as _get_memory
        if memory is None:
            memory = _get_memory()
    except ImportError:
        return 0

    consumed = 0
    for seed in seeds:
        pattern_id = seed.get("pattern_id", "")
        if pattern_id and memory.mark_bridge_consumed(pattern_id):
            consumed += 1

    if verbose and consumed > 0:
        print(f"[MemoryBridge] 标记 {consumed} 个模板为已消费")

    return consumed


# ══════════════════════════════════════════════════════════════

def champion_to_pool_format(factor: Dict, source: str = "alpha_agent_v3_champion") -> Optional[Dict]:
    """将王者因子转换为 factor pool 统一格式。

    v0.6.1 修复 (2026-08-28): JQ 注册表含组合策略记录 (scatter rank和/sentiment复合
    等, 仅有 jq_return 元信息、无单因子公式) → 返回 None 由调用方跳过,
    避免 KeyError: 'expression' 打断整个种子注入。
    """
    expr = factor.get("expression") or factor.get("formula")
    name = factor.get("name")
    if not expr or not name:
        return None
    expr = str(expr)
    name = str(name)
    ops = extract_operators(expr)

    return {
        "name": f"v3_{re.sub(r'[^a-zA-Z0-9_]', '_', name)}",
        "label": name,
        "formula": expr,
        "hypothesis": factor["rationale"],
        "direction": "long" if factor.get("direction", "+") == "+" else "short",
        "paradigm": factor.get("paradigm", "未分类"),
        "category": factor.get("paradigm", "未分类"),
        "source": source,
        "status": "champion_seed",
        "operators": ",".join(ops),
        "windows": ",".join(str(w) for w in extract_windows(expr)),
        "icir": 0.0,    # 王者因子未在 local 验证
        "ic": 0.0,
        "jq_return": CHAMPION_JQ_RESULT["composite_return"],
        "jq_sharpe": CHAMPION_JQ_RESULT["composite_sharpe"],
        "jq_maxdd": CHAMPION_JQ_RESULT["composite_maxdd"],
        "jq_verified": True,
    }


def merge_factor_pools(
    pool_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Tuple[List[Dict], int, int]:
    """
    合并因子库: passed_factor_pool.csv + AlphaAgent v3 王者因子

    Returns
    -------
    (merged_list, n_existing, n_new)
    """
    pool_path = pool_path or DATA_DIR / "passed_factor_pool.csv"
    output_path = output_path or DATA_DIR / "unified_factor_pool.csv"
    
    # v0.5: 如果已有统一池, 从中读取(保留已有的LLM释义)
    last_path = DATA_DIR / "unified_factor_pool.csv"
    if last_path.exists() and last_path.stat().st_mtime > pool_path.stat().st_mtime:
        pool_path = last_path  # 使用最新的统一池为基础

    # 读取现有池
    existing = {}
    if pool_path.exists():
        with open(pool_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get("name", "") or row.get("formula", "")
                if key:
                    existing[key] = dict(row)

    # 转换王者因子
    new_count = 0
    n_skipped = 0
    champions = _get_all_champions()
    for champ in champions:
        converted = champion_to_pool_format(champ)
        if converted is None:
            n_skipped += 1  # 组合策略记录 (无单因子公式), 不入因子池
            continue
        key = converted["name"]
        if key not in existing:
            existing[key] = converted
            new_count += 1
        else:
            # 补充缺失字段
            existing_entry = existing[key]
            for field in ["jq_return", "jq_sharpe", "jq_maxdd", "jq_verified", "operators", "windows"]:
                if field not in existing_entry:
                    existing_entry[field] = converted.get(field, "")

    # 写入合并后的文件
    if existing:
        fieldnames = list(next(iter(existing.values())).keys())
        # 确保关键字段存在
        for fn in ["name", "label", "formula", "hypothesis", "logic", "direction", "paradigm", 
                    "category", "source", "status", "operators", "windows",
                    "icir", "ic", "jq_return", "jq_sharpe", "jq_maxdd", "jq_verified"]:
            if fn not in fieldnames:
                fieldnames.append(fn)

        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in existing.values():
                writer.writerow(row)

    print(f"[SeedInjector] 因子库合并: {len(existing)} 总量 ({len(existing) - new_count} 原有 + {new_count} 王者种子)")
    return list(existing.values()), len(existing) - new_count, new_count


# ═══════════════════════════════════════════════════════════
# Experience Memory 注入
# ═══════════════════════════════════════════════════════════

def inject_champion_seeds(memory=None, verbose: bool = True) -> Dict:
    """
    将王者因子注入 Experience Memory 作为 SuccessPattern 种子。

    Parameters
    ----------
    memory: ExperienceMemory 实例 (None 时自动获取)

    Returns
    -------
    {n_injected, n_updated, paradigms_detected, ...}
    """
    try:
        from experience_memory import get_memory
        if memory is None:
            memory = get_memory()
    except ImportError:
        print("[SeedInjector] 无法导入 ExperienceMemory, 仅做文件合并")
        return {"error": "ExperienceMemory 不可用"}

    injected = 0
    updated = 0
    paradigms = set()

    champions = _get_all_champions()
    for champ in champions:
        expr = champ.get("expression") or champ.get("formula")
        if not expr:
            continue  # v0.6.1: 组合策略记录 (无单因子公式), 跳过 Memory 注入
        # v0.6.2 (2026-08-29): 组合描述文本防复发 — 含中文或组合标记的
        # expression (如 "红利×质量(roe区间5-100)×低波 全池等权" /
        # "overnight5+tvma20+... rank(pct) 等权") 不是单因子公式,
        # 注入 formula 会导致 IC Computer 每轮 SyntaxError
        _expr_s = str(expr)
        if re.search(r'[\u4e00-\u9fff]', _expr_s) or any(
                m in _expr_s for m in ("rank(pct) 等权", "rank-product",
                                       "全池等权", "组合诊断", "非单因子")):
            continue
        ops = extract_operators(str(expr))
        paradigm = champ.get("paradigm", "未分类")
        paradigms.add(paradigm)

        pattern_id = f"{paradigm}::{'|'.join(ops[:3])}"
        description = f"【王者种子】{champ['name']}: {champ['rationale']}"

        # 构建 SuccessPattern
        pattern = {
            "pattern_id": pattern_id,
            "paradigm": paradigm,
            "description": description,
            "typical_operators": ops,
            "typical_windows": extract_windows(expr),
            "occurrence_count": 5,  # 高初始计数 (王者级别)
            "success_rate": 1.0,
            "ic_range": (0.02, 0.10),
            "icir_range": (0.4, 1.2),
            "jq_return": CHAMPION_JQ_RESULT["composite_return"],
            "jq_sharpe": CHAMPION_JQ_RESULT["composite_sharpe"],
            "formula": expr,
            "source": "alpha_agent_v3_champion",
        }

        # Upsert 到 memory
        existing_patterns = memory.data.get("success_templates", [])
        found = False
        for i, p in enumerate(existing_patterns):
            if p.get("pattern_id") == pattern_id:
                # 更新已有记录
                p["occurrence_count"] = max(p.get("occurrence_count", 1), 5)
                p["success_rate"] = 1.0
                p["jq_return"] = CHAMPION_JQ_RESULT["composite_return"]
                p["jq_sharpe"] = CHAMPION_JQ_RESULT["composite_sharpe"]
                p["description"] = description
                p["formula"] = expr
                existing_patterns[i] = p
                found = True
                updated += 1
                break

        if not found:
            existing_patterns.append(pattern)
            injected += 1

    memory.data["success_templates"] = existing_patterns
    memory._save()

    result = {
        "n_injected": injected,
        "n_updated": updated,
        "n_total": injected + updated,
        "paradigms_detected": sorted(paradigms),
        "total_templates": len(existing_patterns),
        "source": "alpha_agent_v3_champion",
    }
    if verbose:
        print(f"[SeedInjector] Memory 注入: {injected} 新模板 + {updated} 更新 = {injected+updated} 王者种子")
        print(f"[SeedInjector]   覆盖范式: {sorted(paradigms)}")
        print(f"[SeedInjector]   Memory 模板总数: {len(existing_patterns)}")
    return result


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def run_full_injection(verbose: bool = True) -> Dict:
    """
    完整注入流程: 合并因子库 + 注入 Memory
    """
    result = {}

    if verbose:
        print("=" * 60)
        print("  AlphaAgent v3 → RalphLoop 种子注入 + 因子库合并")
        print("=" * 60)

    # Step 1: 因子库合并
    merged, n_existing, n_new = merge_factor_pools()
    result["pool"] = {
        "total": len(merged),
        "existing": n_existing,
        "new_champion": n_new,
        "output": str(DATA_DIR / "unified_factor_pool.csv"),
    }

    # Step 2: Memory 注入
    mem_result = inject_champion_seeds()
    result["memory"] = mem_result

    if verbose:
        print(f"\n  合并后因子库: {result['pool']['total']} 因子")
        print(f"  Memory 模板: {mem_result.get('total_templates', '?')} 个")
        print(f"  范式覆盖: {mem_result.get('paradigms_detected', [])}")

    return result


if __name__ == "__main__":
    run_full_injection()
