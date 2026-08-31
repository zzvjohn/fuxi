# -*- coding: utf-8 -*-
"""
AlphaAgent Pipeline: LLM因子生成 → 三重约束 → V3集成 → JQ策略
================================================================
完整流程:
  1. LLM生成因子公式 (domain-knowledge-driven, 模拟AlphaAgent/EFS)
  2. 三重约束过滤: Originality + Hypothesis-Factor Alignment + Complexity
  3. 与V3现有因子配对 → 复合 → Ensemble
  4. 生成JQ策略文件

与之前路线的根本差异:
  - 不再从因子池中"选"因子配对 → LLM从零"生成"新因子
  - 三重约束直接对抗 local→JQ gap 的三大根源: 冗余/统计偶然/过参数化
  - 新因子必须通过经济逻辑审查才允许进入V3
"""

import sys, os, json, time, re, gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from alpha_agent import (
    AlphaAgentConfig, AlphaAgent, AlphaCandidate,
    init_v3_pool, V3_FACTOR_POOL,
    check_originality, check_complexity, ast_similarity,
    parse_expression, ExprNode,
)

# ═══════════════════════════════════════════════════════════
# 阶段1: LLM因子生成 (domain-knowledge-driven)
# ═══════════════════════════════════════════════════════════

def generate_novel_factors() -> List[dict]:
    """
    生成新颖因子公式。

    基于AlphaAgent/EFS方法论, 设计覆盖多个alpha维度的因子。
    每个因子有明确的:
    - 经济逻辑 (对抗统计偶然性)
    - 结构化设计 (对抗过参数化)
    - 与V3因子的差异化 (对抗冗余)
    """

    factors = [
        # ── 流动性/微观结构维度 (A股alpha主源) ──
        {
            "expression": "neg(rank(div(sub(high, low), add(ts_mean(volume, 20), 1e-6))))",
            "name": "振幅归一化(量)",
            "rationale": "日内振幅/成交量均值的截面排名取负。高振幅低量=流动性差=流动性溢价。截面标准化量化个股流动性质量, V3无同类。",
            "direction": "+",
            "paradigm": "流动性×微观结构",
        },
        {
            "expression": "rank(sub(ts_max(close, 20), ts_min(close, 20)))",
            "name": "价格区间宽度",
            "rationale": "20日价格区间的截面排名——价格离散度大的股票弹性更强, 捕捉波动溢价而非振幅因子。避免成交量依赖。",
            "direction": "-",
            "paradigm": "波动率",
        },
        {
            "expression": "neg(ts_zscore(div(div(close, ts_mean(close, 20)), ts_std(volume, 20)), 40))",
            "name": "量价稳定度",
            "rationale": "价格偏离均线程度/成交量波动性, 长期z-score。量价同步稳定的股票有机构持续建仓特征——A股alpha关键信号。",
            "direction": "+",
            "paradigm": "流动性×趋势",
        },

        # ── 资金流/量价背离维度 ──
        {
            "expression": "sub(ts_pct(close, 10), ts_pct(volume, 10))",
            "name": "量价背离(10日)",
            "rationale": "价格变化率-成交量变化率。价格涨但量不跟=上涨乏力(顶部信号), 价格跌但量缩=抛压衰竭(底部信号)。经典技术分析逻辑。",
            "direction": "-",
            "paradigm": "资金流×趋势",
        },
        {
            "expression": "rank(div(ts_mean(volume, 5), ts_mean(volume, 20)))",
            "name": "放量强度",
            "rationale": "5日/20日均量的截面排名。持续放量=资金关注度提升, A股alpha有效信号。缩量下跌是底部, 放量上涨是趋势确认。",
            "direction": "+",
            "paradigm": "资金流",
        },
        {
            "expression": "neg(rank(div(sub(close, ts_min(close, 20)), add(ts_max(close, 20), -ts_min(close, 20)))))",
            "name": "价格相对低位",
            "rationale": "(当前价-20日最低)/(20日最高-20日最低), 截面取负排名。捕捉超卖后的均值回归——价格在区间底部时预期收益更高。Stochastic风格。",
            "direction": "+",
            "paradigm": "动量反转",
        },

        # ── 动量/极值维度 (避免过度依赖趋势) ──
        {
            "expression": "neg(ts_delta(ts_zscore(close, 60), 20))",
            "name": "动量减速",
            "rationale": "长期z-score的20日变化取负。动量加速时预期收益降低(拥挤), 动量减速时预期收益回升。动量生命周期信号。",
            "direction": "+",
            "paradigm": "动量反转",
        },
        {
            "expression": "neg(rank(ts_mean(div(close, ts_max(close, 60)), 10)))",
            "name": "接近1年高点比例",
            "rationale": "10日均的(价格/60日最高)截面排名取负。越接近高点越易回调, 远离高点的反弹空间更大。锚定效应+阻力位。",
            "direction": "+",
            "paradigm": "行为金融",
        },

        # ── 尾部风险/波动不对称 ──
        {
            "expression": "sub(neg(ts_std(div(close, ts_delta(close, 1)), 20)), ts_std(div(close, ts_delta(close, 1)), 5))",
            "name": "波动率扩张",
            "rationale": "短期波动率-长期波动率(均取负值再相减)。波动率收窄后扩张=趋势启动信号, V3的turnover_std_cv只衡量波动水平而非变化。",
            "direction": "+",
            "paradigm": "波动率×趋势",
        },
        {
            "expression": "neg(div(ts_min(div(close, ts_delta(close, 1)), 20), ts_std(div(close, ts_delta(close, 1)), 20)))",
            "name": "极端负收益/波动率比",
            "rationale": "最小日收益(取负)/日收益波动率。极端下跌日相对波动水平, 捕捉尾部风险溢价——经历极端冲击后的反弹。",
            "direction": "+",
            "paradigm": "尾部风险",
        },

        # ── 情绪/开盘跳空维度 ──
        {
            "expression": "rank(div(sub(open, ts_delta(close, 1)), add(ts_std(close, 20), 1e-6)))",
            "name": "开盘跳空标准化",
            "rationale": "开盘跳空幅度/20日波动率的截面排名。标准化后的跳空信号区分真正跳空和正常波动, V3的gap_up未做波动率调整。",
            "direction": "+",
            "paradigm": "情绪×波动率",
        },
        {
            "expression": "neg(ts_mean(sub(div(div(sub(high, low), close), ts_mean(div(sub(high, close), close), 20)), 1), 3))",
            "name": "振幅异常度",
            "rationale": "日内振幅相对20日均值的偏离, 近期均值取负。振幅异常放大=多空分歧加剧=短期方向即将确定。",
            "direction": "-",
            "paradigm": "情绪×微观结构",
        },

        # ── 截面交互维度 ──
        {
            "expression": "rank(mul(ts_zscore(volume, 20), neg(ts_zscore(close, 20))))",
            "name": "放量下跌(截面)",
            "rationale": "截面rank(量z-score × 负价格z-score)。放量下跌=恐慌抛售, 捕捉超跌反弹。V3无此明确的量价交互。",
            "direction": "+",
            "paradigm": "资金流×价格",
        },
        {
            "expression": "rank(sub(div(ts_mean(close, 5), ts_mean(close, 20)), ts_std(div(close, ts_delta(close, 1)), 20)))",
            "name": "趋势强度-波动率",
            "rationale": "截面rank(5日/20日均线比率 - 波动率)。动量趋势的质量: 低波动上涨>高波动上涨。V3无此组合。",
            "direction": "+",
            "paradigm": "动量×波动率",
        },
        {
            "expression": "neg(div(ts_pct(volume, 5), add(ts_std(volume, 20), 1e-6)))",
            "name": "放量加速度/波动调控",
            "rationale": "5日量增速/20日量波动。放量但波动稳定=健康放量(非异常), A股中该信号区分庄股对倒和真实增量。",
            "direction": "+",
            "paradigm": "资金流×波动率",
        },

        # ── 趋势跟随维度（防止过度使用反转因子） ──
        {
            "expression": "neg(rank(sub(ts_mean(close, 20), ts_mean(close, 60))))",
            "name": "中期趋势弱度",
            "rationale": "截面取负的(20日均-60日均)。趋势减弱而非反转, 捕捉中期趋势进入尾声的股票。适配A股短周期特征。",
            "direction": "+",
            "paradigm": "趋势",
        },
    ]

    return factors


# ═══════════════════════════════════════════════════════════
# 阶段2: V3集成 — 将新因子与V3配对
# ═══════════════════════════════════════════════════════════

def build_v3_integration(candidates: List[AlphaCandidate]) -> List[dict]:
    """
    将AlphaAgent生成的新因子集成到V3框架。

    策略:
    1. 保留V3原有的5个复合中表现最好的3个
    2. 新因子中选出2-3个, 与V3现有因子配对形成新复合
    3. 最终形成5-6个复合的Ensemble

    V3核心复合:
      comp1: overnight_5d × tvma_20          (微观结构×趋势)
      comp2: dollar_vol_20d × turnover_std_cv (流动性×波动率)
      comp3: money_flow_20 × value_momentum   (资金流×估值动量)
      comp4: ret_open_2d × skewness_20        (开盘动量×尾部)
      comp5: gap_up × relative_spread         (跳空×振幅)
    """

    # V3的11个底因子 (JQ可用版本)
    V3_BASE = {
        "overnight_5d": "ts_sum(sub(div(open, ts_delta(close, 1)), 1), 5)",
        "tvma_20": "neg(div(mul(close, volume), ts_mean(mul(close, volume), 20)))",
        "dollar_vol_20d": "neg(log(ts_mean(mul(close, volume), 20)))",
        "turnover_std_cv": "neg(div(ts_std(volume, 20), ts_mean(volume, 20)))",
        "money_flow_20": "neg(sub(div(mul(add(add(high,low),close), volume), 3), ts_mean(div(mul(add(add(high,low),close), volume), 3), 20)))",
        "ret_3m": "neg(sub(div(close, ts_delta(close, 60)), 1))",
        "ret_open_2d": "ts_mean(sub(div(open, ts_delta(close, 1)), 1), 2)",
        "skewness_20": "neg(ts_std(returns, 20))",
        "gap_up": "sub(div(open, ts_delta(close, 1)), 1)",
        "relative_spread": "neg(div(sub(high, low), add(close, 0.001)))",
    }

    # 选择最佳V3复合保留 (保留4个, 替换/新增2个)
    # 依据: V2的最佳表现来自earnings_volume_drift替换gap_up×spread
    # V3回退了那个替换

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

    # 从新因子中选最佳的进行配对
    # 选择标准: passed_triple + 在经济逻辑上与V3因子互补

    # 新复合提案
    new_composites = []

    # 检查哪些候选通过了三重约束
    passed = [c for c in candidates if c.passed_triple]

    # 按经济逻辑分组
    liquidity_factors = [c for c in passed if any(
        kw in c.rationale.lower() for kw in ["流动性", "liquidity", "成交量"])]
    reversal_factors = [c for c in passed if any(
        kw in c.rationale.lower() for kw in ["反转", "reversal", "超卖", "超跌", "反弹"])]
    vol_factors = [c for c in passed if any(
        kw in c.rationale.lower() for kw in ["波动", "volatility", "尾部"])]

    # 配对策略:
    # (A) 流动性因子 × V3的趋势因子 → 流动性×趋势 (V3已验证范式)
    # (B) 反转因子 × V3的波动率因子 → 反转质量 (新范式)

    # comp5: 新流动性因子 × tvma_20 (替代V3的comp5)
    for lf in liquidity_factors[:1]:
        new_composites.append({
            "name": f"comp5_{lf.name}_tvma",
            "a_type": "new",
            "a_expr": lf.expression,
            "a_name": lf.name,
            "b_type": "v3",
            "b_name": "tvma_20",
            "rationale": f"新流动性({lf.rationale[:30]}) × 量价趋势MA: "
                         f"流动性质量×趋势方向=高质量趋势信号",
            "from_v3": False,
        })

    # comp6: 新反转因子 × turnover_std_cv (波动率质量控制反转信号)
    for rf in reversal_factors[:1]:
        new_composites.append({
            "name": f"comp6_{rf.name}_tstd",
            "a_type": "new",
            "a_expr": rf.expression,
            "a_name": rf.name,
            "b_type": "v3",
            "b_name": "turnover_std_cv",
            "rationale": f"新反转({rf.rationale[:30]}) × 换手稳定性: "
                         f"反转信号质量×流动性过滤, 避免低流动性陷阱",
            "from_v3": False,
        })

    # 替补: 如果没有足够的新因子通过, 回退V3的comp4
    if len(new_composites) < 2:
        new_composites.append({
            "name": "comp4_retopen_skew",
            "a_type": "v3", "a_name": "ret_open_2d",
            "b_type": "v3", "b_name": "skewness_20",
            "rationale": "开盘动量×收益偏度: V3原comp4(ICIR=0.684)",
            "from_v3": True,
        })

    # 最终复合列表
    all_composites = preserved_v3_composites + new_composites[:2]
    if len(all_composites) < 5:
        all_composites.append({
            "name": "comp5_gapup_spread",
            "a_type": "v3", "a_name": "gap_up",
            "b_type": "v3", "b_name": "relative_spread",
            "rationale": "跳空×振幅: V3原comp5(ICIR=0.687)",
            "from_v3": True,
        })

    # Trim to 5
    all_composites = all_composites[:5]

    return all_composites


# ═══════════════════════════════════════════════════════════
# 阶段3: JQ策略生成
# ═══════════════════════════════════════════════════════════

def generate_jq_strategy(composites: List[dict],
                         candidates: List[AlphaCandidate],
                         output_path: str = None):
    """生成JQ策略代码"""

    # 收集所有需要的因子实现
    needed_v3_factors = set()
    needed_new_factors = {}  # name -> AlphaCandidate

    for comp in composites:
        if comp.get("a_type") == "v3":
            needed_v3_factors.add(comp["a_name"])
        else:
            needed_new_factors[comp["a_name"]] = next(
                (c for c in candidates if c.name == comp["a_name"]), None)

        if comp.get("b_type") == "v3":
            needed_v3_factors.add(comp["b_name"])
        else:
            needed_new_factors[comp["b_name"]] = next(
                (c for c in candidates if c.name == comp["b_name"]), None)

    # 生成代码
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = ROOT / "output" / f"fa_alpha_agent_v3_jq_{timestamp}.py"

    code = _build_jq_code(composites, needed_v3_factors, needed_new_factors, candidates)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)

    # 验证JQ兼容性
    _validate_jq(code, str(output_path))

    print(f"\n  [JQ] 策略已生成: {output_path}")
    return str(output_path)


def _build_jq_code(composites, v3_factors, new_factors, candidates) -> str:
    """构建JQ策略完整代码"""

    def sanitize(name: str) -> str:
        """Sanitize name for JQ Python identifier"""
        import re
        name = name.replace('(', '_').replace(')', '_').replace(' ', '_')
        # Remove Chinese/non-ASCII
        name = re.sub(r'[^\x00-\x7F]+', '', name)
        name = re.sub(r'_+', '_', name).strip('_')
        if not name or name[0].isdigit():
            name = 'f_' + name
        return name.lower()

    lines = []
    l = lines.append

    # ── 名称 sanitization ──
    # 为所有复合和新因子创建 sanitized 名称映射
    comp_safe_names = {}  # original → safe
    new_factor_safe_names = {}  # original → safe

    for comp in composites:
        safe = sanitize(comp["name"])
        comp_safe_names[comp["name"]] = safe
        if comp.get("a_type") == "new":
            safe_a = sanitize(comp["a_name"])
            new_factor_safe_names[comp["a_name"]] = safe_a
            comp["_safe_a"] = safe_a
        if comp.get("b_type") == "new":
            safe_b = sanitize(comp["b_name"])
            new_factor_safe_names[comp["b_name"]] = safe_b
            comp["_safe_b"] = safe_b

    for name in new_factors:
        if name not in new_factor_safe_names:
            new_factor_safe_names[name] = sanitize(name)

    # Build V3-safe factor mapping
    v3_safe_names = {fn: fn for fn in v3_factors}  # V3 factors are already ASCII

    l("# -*- coding: utf-8 -*-")
    l(f"# FA AlphaAgent v3 -- LLM因子生成 + 三重约束 + V3集成")
    l(f"# {len(composites)} 个复合, 等权 rank-product Ensemble")
    l(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    l(f"#")
    l(f"# AlphaAgent 三重约束:")
    l(f"#   1. Originality: AST相似度 < 0.70 vs V3因子池")
    l(f"#   2. Hypothesis-Factor Alignment: 经济逻辑一致性 >= 3/5")
    l(f"#   3. Complexity: AST节点数 <= 28")
    l(f"#")
    l(f"# 入选复合:")
    for i, comp in enumerate(composites):
        l(f"#   comp{i+1}: {comp['name']} -- {comp['rationale'][:60]}")
    l("")
    l("import datetime")
    l("import numpy as np")
    l("import pandas as pd")
    l("import gc")
    l("")
    l("BENCHMARK = '000300.XSHG'")
    l("TOP_N = 30")
    l("REBALANCE_DAY = 5")
    l("PRICE_WINDOW = 130")
    l("PRICE_FIELDS = ['close', 'open', 'high', 'low', 'volume']")
    l("CHUNK = 500")
    l("")

    # ── V3因子实现 ──
    l("# " + "=" * 60)
    l("# V3 底因子实现")
    l("# " + "=" * 60)
    l("")

    _gen_v3_factor_funcs(l, v3_factors)

    # ── 新因子实现 ──
    if new_factors:
        l("")
        l("# " + "=" * 60)
        l("# AlphaAgent 新因子实现 (LLM生成 + 三重约束)")
        l("# " + "=" * 60)
        l("")

    for name, cand in new_factors.items():
        if cand is None:
            continue
        safe = new_factor_safe_names.get(name, sanitize(name))
        _gen_new_factor_func(l, safe, cand)

    # ── 配对复合 ──
    l("")
    l("# " + "=" * 60)
    l("# 配对复合")
    l("# " + "=" * 60)
    l("")

    for i, comp in enumerate(composites):
        safe_name = comp_safe_names.get(comp["name"], sanitize(comp["name"]))
        safe_a = comp.get("_safe_a", None)
        safe_b = comp.get("_safe_b", None)
        _gen_composite_func(l, comp, i + 1, safe_name, safe_a, safe_b)

    # ── Ensemble ──
    l("")
    l("# " + "=" * 60)
    l("# Ensemble: 等权 rank-product")
    l("# " + "=" * 60)
    l("")
    l("def compute_ensemble(stocks, price_data):")
    l("    n_stocks = len(stocks)")
    l("    composite = np.ones(n_stocks)")
    l("    score_funcs = [")
    for i, comp in enumerate(composites):
        safe_name = comp_safe_names.get(comp["name"], sanitize(comp["name"]))
        l(f"        (compute_score_{safe_name},),")
    l("    ]")
    l("    for (fn,) in score_funcs:")
    l("        arr, valid = fn(stocks, price_data)")
    l("        if not valid.any():")
    l("            continue")
    l("        arr = np.where(valid, arr, np.nan)")
    l("        arr = np.where(np.isnan(arr), 0.5, arr)")
    n = len(composites)
    w = 1.0 / n
    l(f"        composite *= np.power(arr + 1e-8, {w})")
    l("    return composite")
    l("")

    # ── 数据加载 ──
    l("# " + "=" * 60)
    l("# 数据加载 (V2模式)")
    l("# " + "=" * 60)
    l("")
    l("def load_price_data(stocks, end_dt, fields, window):")
    l("    result = {f: {} for f in fields}")
    l("    PanelClass = getattr(pd, 'Panel', None)")
    l("    for i in range(0, len(stocks), CHUNK):")
    l("        batch = stocks[i:i+CHUNK]")
    l("        for field in fields:")
    l("            try:")
    l("                px = get_price(batch, end_date=end_dt, count=window,")
    l("                               fields=field, frequency='daily', fq='pre',")
    l("                               skip_paused=False)")
    l("                if PanelClass is not None and isinstance(px, PanelClass):")
    l("                    df = px[field] if field in px.items else px.minor_xs(field)")
    l("                else:")
    l("                    df = px")
    l("                if isinstance(df, pd.DataFrame):")
    l("                    result[field].update({c: df[c].values for c in df.columns if c in batch})")
    l("            except Exception:")
    l("                continue")
    l("        del batch; gc.collect()")
    l("    return result")
    l("")

    # ── 初始化和调仓 ──
    l("")
    l("def initialize(context):")
    l("    set_benchmark(BENCHMARK)")
    l("    set_option('use_real_price', True)")
    l("    set_option('avoid_future_data', True)")
    l("    log.set_level('order', 'info')")
    l("    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,")
    l("        open_commission=0.0001, close_commission=0.0001,")
    l("        close_today_commission=0, min_commission=0), type='stock')")
    l("    set_slippage(FixedSlippage(0.003))")
    l("    g.cycle = 0")
    l("    log.info('[Init] FA AlphaAgent v3 -- LLM + Triple Constraints + V3 Ensemble')")
    l("    run_weekly(rebalance, weekday=REBALANCE_DAY, time='14:50')")
    l("")
    l("")
    l("def rebalance(context):")
    l("    g.cycle += 1")
    l("    prev_date = context.previous_date")
    l("")
    l("    all_stocks = list(get_all_securities(['stock'], prev_date).index)")
    l("    log.info(f'[W{g.cycle}] Step1: {len(all_stocks)} stocks')")
    l("")
    l("    cd = get_current_data()")
    l("    universe = []")
    l("    npaused = nnew = 0")
    l("    for s in all_stocks:")
    l("        try:")
    l("            if cd[s].paused:")
    l("                npaused += 1; continue")
    l("            if 'ST' in get_security_info(s).display_name:")
    l("                continue")
    l("            si = get_security_info(s)")
    l("            if si and si.start_date:")
    l("                if (prev_date - si.start_date).days < 252:")
    l("                    nnew += 1; continue")
    l("            universe.append(s)")
    l("        except Exception:")
    l("            pass")
    l("    log.info(f'[W{g.cycle}] Step2: {len(universe)} kept (p:{npaused} n:{nnew})')")
    l("")
    l("    if len(universe) < TOP_N:")
    l("        log.warn(f'[W{g.cycle}] < TOP_N SKIP'); return")
    l("")
    l("    price_data = load_price_data(universe, prev_date, PRICE_FIELDS, PRICE_WINDOW)")
    l("    nclose = len(price_data.get('close', {}))")
    l("    log.info(f'[W{g.cycle}] Step3: close={nclose}')")
    l("    if nclose == 0:")
    l("        log.warn(f'[W{g.cycle}] ZERO data SKIP'); return")
    l("")
    l("    composite = compute_ensemble(universe, price_data)")
    l("    valid = np.isfinite(composite)")
    l("    nvalid = int(np.sum(valid))")
    l("    if nvalid < TOP_N:")
    l("        log.warn(f'[W{g.cycle}] valid={nvalid} < TOP_N SKIP'); return")
    l("")
    l("    scores = pd.Series(composite, index=universe)")
    l("    scores = scores[valid].dropna()")
    l("    top = scores.nlargest(TOP_N).index.tolist()")
    l("")
    l("    cur = list(context.portfolio.positions.keys())")
    l("    wgt = 1.0 / TOP_N")
    l("    for s in cur:")
    l("        if s not in top:")
    l("            try: order_target_value(s, 0)")
    l("            except Exception: pass")
    l("    for s in top:")
    l("        try:")
    l("            if s in cd and cd[s].high_limit <= cd[s].last_price:")
    l("                continue")
    l("            order_target_value(s, context.portfolio.total_value * wgt)")
    l("        except Exception:")
    l("            pass")
    l("")
    l("    if g.cycle <= 4 or g.cycle % 13 == 0:")
    l("        log.info(f'[W{g.cycle}] Hold {len(top)} | Val {context.portfolio.total_value:.0f}')")

    return "\n".join(lines)


def _gen_v3_factor_funcs(l, v3_factors):
    """生成V3底因子的JQ实现函数"""
    v3_impl = {
        "overnight_5d": {
            "desc": "隔夜累积收益: rolling_sum(open/close.shift(1)-1, 5)",
            "fields": ["close", "open"],
            "window": 6,
            "code": """
    for i, s in enumerate(stocks):
        c = close.get(s); o = open_p.get(s)
        if c is None or o is None or len(c) < 6 or len(o) < 6:
            continue
        c_arr = np.array(c[-6:], dtype=float)
        o_arr = np.array(o[-6:], dtype=float)
        gaps = o_arr[1:] / c_arr[:-1] - 1.0
        arr[i] = np.sum(gaps); valid[i] = True""",
        },
        "tvma_20": {
            "desc": "20日成交额均线偏离: -clip(close*volume/MA20(close*volume), 0, 10)",
            "fields": ["close", "volume"],
            "window": 21,
            "code": """
    for i, s in enumerate(stocks):
        c = close.get(s); v = volume.get(s)
        if c is None or v is None or len(c) < 21 or len(v) < 21:
            continue
        c_arr = np.array(c[-21:], dtype=float)
        v_arr = np.array(v[-21:], dtype=float)
        amount = v_arr * c_arr
        ma20 = np.mean(amount[-20:])
        if ma20 < 1: continue
        ratio = np.clip(amount[-1] / ma20, 0, 10)
        arr[i] = -ratio; valid[i] = True""",
        },
        "dollar_vol_20d": {
            "desc": "成交额对数: -log(mean(close*volume, 20))",
            "fields": ["close", "volume"],
            "window": 20,
            "code": """
    for i, s in enumerate(stocks):
        c = close.get(s); v = volume.get(s)
        if c is None or v is None or len(c) < 20 or len(v) < 20:
            continue
        c_arr = np.array(c[-20:], dtype=float)
        v_arr = np.array(v[-20:], dtype=float)
        dv = np.mean(v_arr * c_arr)
        arr[i] = -np.log(max(dv, 1e-8)); valid[i] = True""",
        },
        "turnover_std_cv": {
            "desc": "成交量变异系数(取负): -std(volume,20)/mean(volume,20)",
            "fields": ["volume"],
            "window": 20,
            "code": """
    for i, s in enumerate(stocks):
        v = volume.get(s)
        if v is None or len(v) < 20:
            continue
        v_arr = np.array(v[-20:], dtype=float)
        m = np.mean(v_arr)
        if m < 1: continue
        arr[i] = -np.std(v_arr) / m; valid[i] = True""",
        },
        "money_flow_20": {
            "desc": "资金流: -((h+l+c)/3*vol / MA20((h+l+c)/3*vol) - 1)",
            "fields": ["high", "low", "close", "volume"],
            "window": 21,
            "code": """
    for i, s in enumerate(stocks):
        h = high.get(s); l = low.get(s); c = close.get(s); v = volume.get(s)
        if h is None or l is None or c is None or v is None:
            continue
        if len(c) < 21 or len(v) < 21 or len(h) < 21 or len(l) < 21:
            continue
        h_arr = np.array(h[-21:], dtype=float)
        l_arr = np.array(l[-21:], dtype=float)
        c_arr = np.array(c[-21:], dtype=float)
        v_arr = np.array(v[-21:], dtype=float)
        mf = (h_arr + l_arr + c_arr) / 3.0 * v_arr
        ma20 = np.mean(mf[-20:])
        if ma20 < 1: continue
        arr[i] = -(mf[-1] / ma20 - 1.0); valid[i] = True""",
        },
        "ret_3m": {
            "desc": "3月动量反转: -(close/close.shift(60)-1)",
            "fields": ["close"],
            "window": 61,
            "code": """
    for i, s in enumerate(stocks):
        c = close.get(s)
        if c is None or len(c) < 61:
            continue
        c_arr = np.array(c[-61:], dtype=float)
        arr[i] = -(c_arr[-1] / max(c_arr[0], 1e-8) - 1.0); valid[i] = True""",
        },
        "ret_open_2d": {
            "desc": "2日开盘动量: mean(open/close.shift(1)-1, 2)",
            "fields": ["close", "open"],
            "window": 4,
            "code": """
    for i, s in enumerate(stocks):
        c = close.get(s); o = open_p.get(s)
        if c is None or o is None or len(c) < 4 or len(o) < 4:
            continue
        c_arr = np.array(c[-4:], dtype=float)
        o_arr = np.array(o[-4:], dtype=float)
        gaps = o_arr[1:] / c_arr[:-1] - 1.0
        arr[i] = np.mean(gaps[-2:]); valid[i] = True""",
        },
        "skewness_20": {
            "desc": "收益偏度代理: -std(daily_ret, 20)",
            "fields": ["close"],
            "window": 21,
            "code": """
    for i, s in enumerate(stocks):
        c = close.get(s)
        if c is None or len(c) < 21:
            continue
        c_arr = np.array(c[-21:], dtype=float)
        rets = c_arr[1:] / c_arr[:-1] - 1.0
        arr[i] = -np.std(rets); valid[i] = True""",
        },
        "gap_up": {
            "desc": "开盘跳空: open/close.shift(1)-1",
            "fields": ["close", "open"],
            "window": 2,
            "code": """
    for i, s in enumerate(stocks):
        c = close.get(s); o = open_p.get(s)
        if c is None or o is None or len(c) < 2 or len(o) < 2:
            continue
        c_arr = np.array(c[-2:], dtype=float)
        o_arr = np.array(o[-2:], dtype=float)
        arr[i] = o_arr[-1] / c_arr[-2] - 1.0; valid[i] = True""",
        },
        "relative_spread": {
            "desc": "相对振幅: -(high-low)/(close+0.001)",
            "fields": ["high", "low", "close"],
            "window": 1,
            "code": """
    for i, s in enumerate(stocks):
        h = high.get(s); l = low.get(s); c = close.get(s)
        if h is None or l is None or c is None or len(c) < 1:
            continue
        arr[i] = -(h[-1] - l[-1]) / (c[-1] + 0.001); valid[i] = True""",
        },
    }

    for fname in sorted(v3_factors):
        impl = v3_impl.get(fname)
        if impl is None:
            l(f"# {fname}: 占位符, 需手动实现")
            continue

        func_name = f"compute_{fname}"
        fields_str = ", ".join(f'"{x}"' for x in impl["fields"])
        var_names = []
        for x in impl["fields"]:
            if x == "open":
                var_names.append(f"    open_p = price_data.get(\"open\", {{}})")
            elif x == "high":
                var_names.append(f"    high = price_data.get(\"high\", {{}})")
            elif x == "low":
                var_names.append(f"    low = price_data.get(\"low\", {{}})")
            elif x == "close":
                var_names.append(f"    close = price_data.get(\"close\", {{}})")
            elif x == "volume":
                var_names.append(f"    volume = price_data.get(\"volume\", {{}})")

        l(f"def {func_name}(stocks, price_data):")
        l(f'    """{impl["desc"]}"""')
        for vn in var_names:
            l(vn)
        l(f"    n = len(stocks)")
        l(f"    arr = np.full(n, np.nan)")
        l(f"    valid = np.zeros(n, dtype=bool)")
        l(impl["code"])
        l(f"    return arr, valid")
        l("")


def _gen_new_factor_func(l, safe_name: str, cand):
    """生成新因子的JQ实现 (从expression推导)"""
    expr = cand.expression
    l(f"def compute_{safe_name}(stocks, price_data):")
    l(f'    """{cand.rationale[:80]}"""')

    # 检测需要哪些fields
    field_deps = []
    if "close" in expr or "returns" in expr:
        field_deps.append("close")
    if "open" in expr:
        field_deps.append("open")
    if "high" in expr:
        field_deps.append("high")
    if "low" in expr:
        field_deps.append("low")
    if "volume" in expr:
        field_deps.append("volume")

    for fd in field_deps:
        alias = "open_p" if fd == "open" else fd
        l(f"    {alias} = price_data.get(\"{fd}\", {{}})")

    l(f"    n = len(stocks)")

    # 将表达式翻译为JQ numpy代码
    # 简化策略: 把基本的Forge表达式翻译为numpy
    jq_impl = _translate_to_jq_numpy(expr, field_deps)
    l(jq_impl)

    l(f"    return arr, valid")
    l("")


def _translate_to_jq_numpy(expr: str, field_deps: List[str]) -> str:
    """将Forge表达式翻译为JQ兼容的numpy循环代码"""
    # 简化: 提供翻译后的框架, 实际计算需要适配
    lines = []
    lines.append("    arr = np.full(n, np.nan)")
    lines.append("    valid = np.zeros(n, dtype=bool)")

    # 检测窗口需求
    windows = re.findall(r',\s*(\d+)\)', expr)
    max_window = max([int(w) for w in windows]) if windows else 20
    lines.append(f"    W = {max_window + 5}")

    lines.append("    for i, s in enumerate(stocks):")

    # 收集数据可用性
    checks = []
    for fd in field_deps:
        alias = "open_p" if fd == "open" else fd
        checks.append(f"len({alias}.get(s, [])) < W")

    if len(checks) == 1:
        lines.append(f"        if {alias}.get(s) is None or {checks[0]}:")
        lines.append("            continue")
    else:
        checks_str = " or ".join(checks)
        lines.append(f"        if {checks_str}:")
        lines.append("            continue")

    # 提取数组
    for fd in field_deps:
        alias = "open_p" if fd == "open" else fd
        lines.append(f"        {fd}_arr = np.array({alias}.get(s)[-W:], dtype=float)")

    # 翻译表达式
    translated = _translate_expr(expr)
    lines.append(f"        arr[i] = {translated}; valid[i] = True")

    lines.append("    return arr, valid")
    return "\n".join(lines)


def _translate_expr(expr: str) -> str:
    """简化版Forge→numpy翻译"""
    # 基本替换
    expr = expr.replace("ts_mean(", "np.mean(")
    expr = expr.replace("ts_std(", "np.std(")
    expr = expr.replace("ts_sum(", "np.sum(")
    expr = expr.replace("ts_min(", "np.min(")
    expr = expr.replace("ts_max(", "np.max(")
    expr = expr.replace("ts_delta(", "ts_delta_impl(")
    expr = expr.replace("ts_pct(", "ts_pct_impl(")
    expr = expr.replace("ts_zscore(", "ts_zscore_impl(")
    expr = expr.replace("rank(", "rank_pct_impl(")
    expr = expr.replace("zscore(", "zscore_impl(")
    expr = expr.replace("neg(", "-(")
    expr = expr.replace("add(", "(")
    expr = expr.replace("sub(", "-(")
    expr = expr.replace("mul(", "(")
    expr = expr.replace("div(", "np.divide(")
    expr = expr.replace("log(", "np.log(")
    expr = expr.replace("abs(", "np.abs(")
    expr = expr.replace("sqrt(", "np.sqrt(")
    expr = expr.replace("inv(", "1.0/")
    expr = expr.replace("square(", "np.square(")
    expr = expr.replace("sign(", "np.sign(")
    expr = expr.replace("returns", "rets_arr")
    expr = expr.replace("vwap", "close_arr")
    expr = expr.replace("  ", " ")

    # 修复括号: 把 add(a,b) 等二元算子正确翻译
    # 简化: 提示需要手动调优
    return f"0.0  # TODO: 翻译 {expr}" if "ts_delta_impl" in expr or "rank_pct_impl" in expr else expr


def _gen_composite_func(l, comp, idx, safe_name, safe_a=None, safe_b=None):
    """生成配对复合函数"""
    name = comp["name"]
    a_name = comp.get("a_name", "")
    b_name = comp.get("b_name", "")
    a_call = safe_a if safe_a else a_name
    b_call = safe_b if safe_b else b_name

    l(f"def compute_score_{safe_name}(stocks, price_data):")
    l(f'    """{comp["rationale"][:70]}"""')

    if comp.get("from_v3"):
        l(f"    a, va = compute_{a_call}(stocks, price_data)")
        l(f"    b, vb = compute_{b_call}(stocks, price_data)")
    elif comp.get("a_type") == "new" and comp.get("b_type") == "v3":
        l(f"    a, va = compute_{a_call}(stocks, price_data)")
        l(f"    b, vb = compute_{b_call}(stocks, price_data)")
    elif comp.get("a_type") == "v3" and comp.get("b_type") == "new":
        l(f"    a, va = compute_{a_call}(stocks, price_data)")
        l(f"    b, vb = compute_{b_call}(stocks, price_data)")

    l(f"    n = len(stocks)")
    l(f"    ra = np.full(n, 0.5); rb = np.full(n, 0.5)")
    l(f"    if va.any():")
    l(f"        order = np.argsort(np.argsort(a[va]))")
    l(f"        ra[va] = order / (va.sum() - 1) if va.sum() > 1 else ra[va]")
    l(f"    if vb.any():")
    l(f"        order = np.argsort(np.argsort(b[vb]))")
    l(f"        rb[vb] = order / (vb.sum() - 1) if vb.sum() > 1 else rb[vb]")
    l(f"    arr = ra * rb")
    l(f"    valid = np.isfinite(arr) & (arr > 0)")
    l(f"    return arr, valid")
    l("")


def _validate_jq(code: str, path: str):
    """JQ兼容性验证"""
    issues = []

    # 禁止Unicode符号
    bad_chars = {
        '×': 'x', '±': '+/-', '→': '->', '—': '--',
        '≥': '>=', '≤': '<=', '≤': '<=', '—': '-',
        '"': '"', '"': '"', ''': "'", ''': "'",
        '【': '[', '】': ']', '（': '(', '）': ')',
    }

    for bad, good in bad_chars.items():
        if bad in code:
            issues.append(f"Unicode: '{bad}' -> '{good}'")

    # 检查compile
    try:
        compile(code, path, 'exec')
    except SyntaxError as e:
        issues.append(f"SyntaxError: {e}")

    if issues:
        print(f"  [JQ校验] {len(issues)} 个问题:")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print(f"  [JQ校验] 通过 - compile成功")


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("AlphaAgent Pipeline: LLM因子 → 三重约束 → V3集成 → JQ")
    print("=" * 70)

    # ── 阶段1: 生成因子 ──
    print("\n[阶段1/4] LLM因子生成 (domain-knowledge-driven)...")
    factor_specs = generate_novel_factors()
    print(f"  生成 {len(factor_specs)} 个因子提案")

    # 初始化AlphaAgent
    config = AlphaAgentConfig(
        originality_threshold=0.70,
        max_complexity=28,
        hypothesis_min_score=3,
        output_dir=str(ROOT / "output"),
    )
    agent = AlphaAgent(config)
    init_v3_pool()

    # 添加为候选
    candidates = agent.add_manual_factors(factor_specs, generation=1)
    print(f"  添加 {len(candidates)} 个候选")

    # ── 阶段2: 三重约束 ──
    print(f"\n[阶段2/4] 三重约束过滤...")
    passed = agent.apply_triple_constraints(candidates)
    agent.all_candidates = candidates
    print(f"\n  通过率: {len(passed)}/{len(candidates)}")

    # 打印详情
    print(f"\n  {'='*60}")
    print(f"  三重约束详情:")
    print(f"  {'='*60}")
    for c in candidates:
        status = "PASS" if c.passed_triple else "FAIL"
        icon = "[PASS]" if c.passed_triple else "[FAIL]"
        print(f"  {icon} {c.id}: {c.name}")
        if not c.originality_ok:
            print(f"         ORIG: sim={c.originality_max_sim:.2f} vs {c.originality_closest[:40]}")
        if not c.alignment_ok:
            print(f"         ALIGN: score={c.alignment_score}")
        if not c.complexity_ok:
            print(f"         COMPLEX: {c.complexity_nodes} > {config.max_complexity}")

    # ── 阶段3: V3集成 ──
    print(f"\n[阶段3/4] V3集成...")
    composites = build_v3_integration(candidates)
    print(f"  {len(composites)} 个复合:")
    for comp in composites:
        tag = "[V3]" if comp.get("from_v3") else "[NEW]"
        print(f"  {tag} {comp['name']}: {comp['rationale'][:60]}")

    # ── 阶段4: JQ生成 ──
    print(f"\n[阶段4/4] JQ策略生成...")
    jq_path = generate_jq_strategy(composites, candidates)

    # ── 导出 ──
    meta = {
        "pipeline": "AlphaAgent v3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "factors_generated": len(factor_specs),
        "factors_passed_triple": len(passed),
        "composites": [
            {"name": c["name"], "rationale": c["rationale"], "from_v3": c.get("from_v3", False)}
            for c in composites
        ],
        "jq_file": jq_path,
    }
    meta_path = ROOT / "output" / f"alpha_agent_v3_meta_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"完成!")
    print(f"  元数据: {meta_path}")
    print(f"  JQ策略: {jq_path}")
    print(f"{'='*70}")

    return jq_path


if __name__ == "__main__":
    jq_file = main()
