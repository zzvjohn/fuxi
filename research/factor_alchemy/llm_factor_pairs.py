# -*- coding: utf-8 -*-
"""
LLM 辅助因子配对提案模块
=========================
基于 AlphaAgent (KDD 2025) 和 EFS (2507.17211) 方法论:
  - 不依赖统计优化器 (GAMP3)
  - 用金融领域知识提出有经济逻辑的因子配对
  - JQ 回测为终极裁判

V2/V3 已验证的配对范式:
  1. 流动性 × 波动率: dollar_vol × turnover_std (不同维度量同一现象)
  2. 微观结构 × 趋势: overnight × tvma (短期异象 × 中长期方向)
  3. 资金流 × 估值动量: money_flow × value_momentum (聪明钱 × 便宜)
  4. 开盘动量 × 尾部风险: ret_open × skewness (方向 × 风险定价)
  5. 跳空 × 振幅: gap_up × relative_spread (事件驱动 × 波动)
  6. 单独复合: earnings_volume_drift (缩量上涨=筹码锁定)

本模块基于 86 因子库, 按上述范式 + 因子类别交叉生成候选配对方便后续 JQ 验证.
"""

from __future__ import division
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# ============================================================
# 因子配对提案 (基于金融领域知识, 非统计优化)
# ============================================================

# 每个提案: (factor_a, factor_b, composite_label, economic_rationale, pairing_type)
# pairing_type: complementary | orthogonal | reinforcement | standalone

LLM_PAIR_PROPOSALS: List[Tuple[str, str, str, str, str]] = [
    # ═══════════════════════════════════════════════════════
    # 范式 1: 流动性 × 波动率 (V2验证: dollar_vol×turnover_std)
    # 逻辑: 高流动性的高波动 = 机构博弈区, 低流动性的高波动 = 散户噪音区
    # ═══════════════════════════════════════════════════════
    ("dollar_vol_20d", "vol_1m",
     "dollar_vol×vol_1m",
     "成交额×短期波动: 高成交额+高波动=机构关注, 低成交额+高波动=散户噪音。区分波动率质量。",
     "complementary"),

    ("amihud_illiq", "downside_vol",
     "illiq×downside",
     "非流动性×下行风险: Amihud测价格冲击成本, 叠加下行波动捕捉流动性危机中的错杀机会。",
     "complementary"),

    ("turnover_cv_20d", "atr_14",
     "turnover_cv×atr",
     "换手变异系数×均幅: 换手不稳定+高波动=投机特征, 应回避。反向做空信号。",
     "complementary"),

    ("dollar_vol_stability", "vol_3m",
     "dvol_stable×vol3m",
     "成交额稳定性×中期波动: 稳定的高成交额+低波动=机构锁仓信号, 类似earnings_volume_drift逻辑。",
     "complementary"),

    # ═══════════════════════════════════════════════════════
    # 范式 2: 微观结构 × 趋势方向 (V2验证: overnight×tvma)
    # 逻辑: 短期异象(隔夜/开盘)提供方向, 中长期趋势提供过滤
    # ═══════════════════════════════════════════════════════
    ("overnight_5d", "bias_20",
     "overnight×bias20",
     "隔夜收益×乖离率: 隔夜反映信息优势, 乖离率提供趋势强度。隔夜正+乖离正=趋势确认的知情交易。",
     "orthogonal"),

    ("short_rev_5d", "boll_pct_b",
     "short_rev×boll",
     "5日反转×布林带位置: 短期超卖+布林下轨=均值回归买点, 短期超买+布林上轨=获利回吐卖点。",
     "reinforcement"),

    ("gap_up", "roc_20",
     "gap_up×roc20",
     "跳空持续性×变动速率: 跳空+趋势加速=突破确认, 跳空+趋势减速=假突破风险。",
     "reinforcement"),

    ("opening_gap_momentum", "tvma_20",
     "open_gap_mom×tvma",
     "开盘跳空动量×趋势MA偏差: 开盘动量+趋势方向一致=高确信度信号。",
     "reinforcement"),

    # ═══════════════════════════════════════════════════════
    # 范式 3: 资金流 × 估值 (V2验证: money_flow×value_momentum)
    # 逻辑: 聪明钱流向×便宜程度, 资金确认估值信号
    # ═══════════════════════════════════════════════════════
    ("money_flow_20", "bp",
     "money_flow×bp",
     "资金流量×账面市值比: 资金流入+低PB=价值发现, 资金流出+低PB=价值陷阱。",
     "orthogonal"),

    ("vpt", "ep",
     "vpt×ep",
     "量价趋势×盈利收益率: VPT量价确认+高E/P=便宜且有资金推动的趋势股。",
     "orthogonal"),

    ("volume_ratio", "cfp",
     "vol_ratio×cfp",
     "量比反转×现金收益率: 缩量+高现金收益率=被低估的现金牛, 类似巴菲特逻辑。",
     "complementary"),

    ("vroc_12", "sp",
     "vroc×sp",
     "量变动速率×营收比: 放量+低PS=营收被低估且有资金关注。",
     "orthogonal"),

    # ═══════════════════════════════════════════════════════
    # 范式 4: 动量 × 收益分布 (V2验证: ret_open×skewness)
    # 逻辑: 趋势方向×风险特征, 区分"好动量"和"坏动量"
    # ═══════════════════════════════════════════════════════
    ("ret_1m", "kurtosis_20",
     "ret1m×kurtosis",
     "月收益×峰度: 正收益+低峰度=稳健上涨, 正收益+高峰度=尾部风险(可能含单日暴涨)。",
     "complementary"),

    ("max_ret_1m", "sharpe_20",
     "max_ret×sharpe",
     "月最大日收益×夏普比率: 排除彩票型股票(高max_ret+低sharpe), 保留真趋势(高max_ret+高sharpe)。",
     "complementary"),

    ("streak", "skewness_20",
     "streak×skewness",
     "连涨天数×偏度: 连续上涨+正偏度=动量持续, 连续上涨+负偏度=可能反转。Bali et al. (2011) 彩票效应。",
     "orthogonal"),

    ("ret_3m", "idio_vol",
     "ret3m×idio_vol",
     "3月收益×特质波动: 高收益+低特质波动=低异质风险的动量(更可持续), 高收益+高特质波动=噪音。",
     "complementary"),

    # ═══════════════════════════════════════════════════════
    # 范式 5: 情绪 × 流动性 (新范式, 未被V2/V3覆盖)
    # 逻辑: 市场情绪信号×流动性条件, 确认情绪是否被资金支持
    # ═══════════════════════════════════════════════════════
    ("vr_26", "turnover_to_vol",
     "vr26×tov_ratio",
     "VR量比×换手波动比: VR反映量能结构(上涨量vs下跌量), 换手/波动比反映交易效率。高VR+高效率=真实需求。",
     "complementary"),

    ("psy_12", "dollar_vol_20d",
     "psy×dollar_vol",
     "心理线×成交额: PSY超卖+高成交额=恐慌放量底, PSY超买+缩量=上涨乏力顶。",
     "orthogonal"),

    ("rsi_14", "turnover_cv_20d",
     "rsi×turn_cv",
     "RSI反转×换手稳定性: RSI超卖+换手稳定=健康回调, RSI超卖+换手异常=恐慌抛售。",
     "complementary"),

    # ═══════════════════════════════════════════════════════
    # 范式 6: 成长 × 质量 (新范式)
    # 逻辑: 高增长+高质量=真正成长, 高增长+低质量=会计操纵风险
    # ═══════════════════════════════════════════════════════
    ("earnings_growth_yoy", "f_score",
     "earn_growth×f_score",
     "盈利增长×Piotroski F-score: 高增长+高F-score=真实成长, 高增长+低F-score=可能盈余管理。",
     "complementary"),

    ("rev_growth_yoy", "accruals",
     "rev_growth×accruals",
     "营收增长×应计利润: 营收增长+低应计=高质量增长(Sloan 1996), 营收增长+高应计=现金转化差。",
     "complementary"),

    ("roe", "asset_growth",
     "roe×asset_growth",
     "ROE×资产增长: 高ROE+适度资产增长=高效扩张(Cooper-Gulen-Schill 2008), 高ROE+激进扩张=回报递减风险。",
     "orthogonal"),

    ("gross_margin", "earnings_quality_proxy",
     "margin×earn_quality",
     "毛利率×盈利质量: 高毛利+高质量盈利=可持续护城河, 高毛利+低质量=一次性因素。",
     "complementary"),

    # ═══════════════════════════════════════════════════════
    # 范式 7: 事件驱动 (V2验证: earnings_volume_drift 单独)
    # 逻辑: 特定事件信号足够强, 不需要配对, 但可加过滤因子
    # ═══════════════════════════════════════════════════════
    ("earnings_volume_drift", "__STANDALONE__",
     "earnings_vol_drift",
     "缩量上涨=筹码锁定: V2已验证, ICIR=1.06。单独使用, 不配对。Jegadeesh-Titman风格, PEAD效应。",
     "standalone"),

    ("volume_climax_reversal", "__STANDALONE__",
     "vol_climax_rev",
     "放量极点反转: ICIR=1.12 (全因子池最高!), 放量极值后反转。单独使用。",
     "standalone"),

    ("trend_persistence_score", "ret_6m",
     "trend_persist×ret6m",
     "趋势持续性×6月收益: 趋势质量(持续性好)+中长期动量方向=强趋势股。过滤假突破。",
     "reinforcement"),

    # ═══════════════════════════════════════════════════════
    # 范式 8: 跨类别交叉 (互补信息源)
    # 逻辑: 两个完全不相关的因子类别交叉, 最大化信息增量
    # ═══════════════════════════════════════════════════════
    ("davol_20", "beta",
     "davol×beta",
     "换手异动×Beta: 换手异常+高Beta=投机热度, 换手异常+低Beta=基本面异动(更可信)。",
     "orthogonal"),

    ("bull_power", "f_score",
     "bull_power×f_score",
     "多头力道×F-score: 技术面强势+基本面优质=戴维斯双击候选。",
     "reinforcement"),

    ("high_low_range", "dp",
     "hl_range×dp",
     "高低价振幅×股息率: 高振幅+高股息=防御性高波动(可能是错杀), 低振幅+高股息=稳健红利。",
     "complementary"),
]


# ============================================================
# 提案管理
# ============================================================

def get_all_proposals() -> List[Dict]:
    """返回所有 LLM 提案的标准化字典列表."""
    proposals = []
    for i, (fa, fb, label, rationale, ptype) in enumerate(LLM_PAIR_PROPOSALS):
        proposals.append({
            "id": f"llm_pair_{i+1:03d}",
            "factor_a": fa,
            "factor_b": fb,
            "composite_label": label,
            "rationale": rationale,
            "pairing_type": ptype,
            "is_standalone": fb == "__STANDALONE__",
        })
    return proposals


def get_proposals_by_type(pairing_type: str) -> List[Dict]:
    """按配对类型过滤提案."""
    return [p for p in get_all_proposals() if p["pairing_type"] == pairing_type]


def get_proposals_by_factor(factor_name: str) -> List[Dict]:
    """查找包含指定因子的所有提案."""
    return [
        p for p in get_all_proposals()
        if p["factor_a"] == factor_name or p["factor_b"] == factor_name
    ]


def print_proposals(proposals: Optional[List[Dict]] = None):
    """打印提案概览."""
    if proposals is None:
        proposals = get_all_proposals()

    type_counts = {}
    for p in proposals:
        t = p["pairing_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n{'='*70}")
    print(f"LLM 因子配对提案 — {len(proposals)} 个候选")
    print(f"{'='*70}")
    print(f"按类型: {type_counts}")
    print()

    for p in proposals:
        tag = "[单因子]" if p["is_standalone"] else f"[{p['pairing_type']}]"
        if p["is_standalone"]:
            print(f"  {p['id']} {tag} {p['factor_a']:30s} → {p['composite_label']}")
        else:
            print(f"  {p['id']} {tag} {p['factor_a']:22s} × {p['factor_b']:22s} → {p['composite_label']}")
        print(f"       {p['rationale'][:90]}...")
        print()


def export_proposals_json(output_path: str = None) -> str:
    """导出提案为 JSON, 供下游 (K-fold OOS / JQ生成器) 使用."""
    if output_path is None:
        output_path = str(Path(__file__).parent / "output" / "llm_pair_proposals.json")

    proposals = get_all_proposals()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_by": "llm_factor_pairs.py (domain-knowledge-driven, AlphaAgent/EFS methodology)",
            "total_pairs": len(proposals),
            "pairing_paradigms": [
                "流动性×波动率 (V2验证)",
                "微观结构×趋势 (V2验证)",
                "资金流×估值 (V2验证)",
                "动量×收益分布 (V2验证)",
                "情绪×流动性 (新范式)",
                "成长×质量 (新范式)",
                "事件驱动/单独 (V2验证)",
                "跨类别交叉 (新范式)",
            ],
            "proposals": proposals,
        }, f, indent=2, ensure_ascii=False)

    print(f"  提案已导出: {output_path}")
    return output_path


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    print_proposals()
    export_proposals_json()
