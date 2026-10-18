
# -*- coding: utf-8 -*-
"""
V0.3 自动化因子组合策略 — 由 Ralph Loop v0.3 生成
生成时间: 2026-08-07 16:55:08
因子来源: GP 育种 + 因子库精选
"""

import numpy as np
import pandas as pd
from datetime import datetime

def initialize(context):
    """初始化"""
    g.top_n = 80
    g.rebalance_freq = 5  # 周频调仓
    g.day_count = 0
    g.factors = [
        {"name": "gp_breed_000", "formula": r"""mul(sub(div(close, mean(rolling(volume_p, 60))), 1), pct_change(close_p, 5))"""},
        {"name": "gp_breed_001", "formula": r"""mean(rolling(sub(div(open_p, shift(close_p, 1)), 1), 5))"""},
        {"name": "gp_breed_002", "formula": r"""neg(mean(rolling(astype(gt(abs(sub(open_p, shift(close_p, 1))), mul(0.0200, shift(close_p, 1))), flo"""},
        {"name": "gp_breed_003", "formula": r"""mean(rolling(sub(div(open_p, shift(close_p, 1)), rolling(zscore(abs(overnight)), 15)), 5))"""},
        {"name": "gp_breed_004", "formula": r"""pct_change(overnight)"""},
        {"name": "gp_breed_005", "formula": r"""neg(hl_ratio)"""},
    ]
    
    # 热身期
    g.warmup = 5
    
    set_benchmark("000905.XSHG")
    set_option("use_real_price", True)
    log.info("V0.3 策略初始化完成, {} 个因子".format(len(g.factors)))

def compute_factor(data, formula_str):
    """按公式计算因子值 (numpy 实现)"""
    try:
        # 简化计算：使用 pandas eval
        close = data["close"]
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        amount = data.get("money", data.get("amount", volume * close))
        
        # 构建本地命名空间
        context = {
            "close_p": close.values,
            "open_p": open_.values,
            "high_p": high.values,
            "low_p": low.values,
            "volume_p": volume.values,
            "amount_p": amount.values,
            "overnight": open_.values / np.roll(close.values, 1) - 1,
        }
        
        result = eval(formula_str, {"np": np, "abs": np.abs, "sqrt": np.sqrt, "__builtins__": {}}, context)
        if isinstance(result, np.ndarray):
            return pd.Series(result.ravel(), index=close.index)
        return pd.Series(result, index=close.index)
    except Exception as e:
        log.warning("Factor compute error: {}".format(e))
        return pd.Series(0, index=data.index)

def before_trading_start(context):
    """开盘前：计算因子并选股"""
    g.day_count += 1
    
    # 周频调仓
    if g.day_count % g.rebalance_freq != 0 and g.day_count > 1:
        return
    
    try:
        # 获取全市场股票
        stocks = list(get_all_securities(["stock"]).index)
        # 过滤 ST, 新股, 停牌
        current_data = get_current_data()
        stocks = [s for s in stocks 
                  if not current_data[s].is_st 
                  and not current_data[s].paused
                  and s.startswith(("0", "3", "6"))]
        
        if len(stocks) < g.top_n:
            return
        
        # 获取历史价格数据
        end_date = context.previous_date
        start_date = end_date - pd.Timedelta(days=300)
        
        panel = get_price(
            stocks, start_date=start_date, end_date=end_date,
            fields=["close", "open", "high", "low", "volume", "money"],
            frequency="daily", skip_paused=True, fq="pre"
        )
        
        if panel is None or len(panel) < 50:
            return
        
        # 计算每个因子的值
        factor_scores = {}
        for fac in g.factors:
            name = fac["name"]
            formula = fac["formula"]
            
            try:
                # 对每只股票计算因子值
                values = {}
                for stock in stocks:
                    if stock in panel.minor_axis:
                        stock_data = panel.minor_xs(stock)
                        if len(stock_data) >= 50:
                            val = compute_factor(stock_data, formula)
                            if isinstance(val, pd.Series) and len(val) > 0:
                                values[stock] = val.iloc[-1]
                
                if values:
                    factor_scores[name] = values
            except Exception as e:
                log.debug("Factor {} error: {}".format(name, e))
        
        if not factor_scores:
            return
        
        # 综合评分: rank-product
        combined_scores = {}
        for stock in stocks:
            ranks = []
            for name, scores in factor_scores.items():
                if stock in scores:
                    sorted_stocks = sorted(scores.items(), key=lambda x: x[1])
                    rank = next((i for i, (s, _) in enumerate(sorted_stocks) if s == stock), len(sorted_stocks) // 2)
                    ranks.append(rank)
            if ranks:
                combined_scores[stock] = np.prod([r + 1 for r in ranks])  # rank-product
        
        if combined_scores:
            # 选 Top-N
            sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1])
            g.security_list = [s for s, _ in sorted_scores[:g.top_n]]
            
            # 等权配置
            g.weight = {s: 1.0 / len(g.security_list) for s in g.security_list}
            log.info("Selected {} stocks from {} factors".format(len(g.security_list), len(factor_scores)))
            
    except Exception as e:
        log.error("Rebalance error: {}".format(e))

def handle_data(context, data):
    """每日交易"""
    if not hasattr(g, "security_list") or not g.security_list:
        return
    
    if not hasattr(g, "weight"):
        return
    
    # 执行调仓
    for security in list(context.portfolio.positions.keys()):
        if security not in g.security_list:
            order_target(security, 0)
    
    for security in g.security_list:
        target_value = context.portfolio.total_value * g.weight.get(security, 0)
        if target_value > 0:
            order_target_value(security, target_value)
