# -*- coding: utf-8 -*-
"""
V0.3 自动化因子组合策略 — Ralph Loop v0.3 生成
时间: 2026-08-07 16:58:43
因子: 3 个 (S5通过 + 真实因子库)
"""

import numpy as np
import pandas as pd

def initialize(context):
    g.top_n = 80
    g.rebalance_freq = 5
    g.day_count = 0
    g.factors = [
        {"name": "volume_stability", "formula": r"""-(volume_p.rolling(20).std() / volume_p.rolling(20).mean())"""},
        {"name": "dollar_vol_stability", "formula": r"""-(volume_p * close_p).rolling(20).std() / (volume_p * close_p).rolling(20).mean()"""},
        {"name": "trend_smoothness", "formula": r"""close_p.rolling(20).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1]**2 if len(x) > 5 else 0, raw=True)"""},
    ]
    g.warmup = 5
    set_benchmark("000905.XSHG")
    set_option("use_real_price", True)
    log.info("V0.3 Ralph Loop 策略: {} 个因子".format(len(g.factors)))

def compute_factor_np(data, formula):
    """用 numpy 实现因子公式（向量化）"""
    try:
        close = data["close"].values
        open_v = data["open"].values
        high = data["high"].values
        low = data["low"].values
        volume = data["volume"].values
        amount = data.get("money", np.zeros_like(close))
        
        context = {
            "np": np, "pd": pd,
            "close_p": close, "open_p": open_v,
            "high_p": high, "low_p": low,
            "volume_p": volume, "amount_p": amount,
            "close": close, "open": open_v,
            "high": high, "low": low,
            "volume": volume, "amount": amount,
            "overnight": open_v / np.roll(close, 1) - 1,
            "abs": np.abs, "sqrt": np.sqrt, "log": np.log,
            "sign": np.sign, "rank": lambda x: pd.Series(x).rank().values,
        }
        
        result = eval(formula, {"__builtins__": {}}, context)
        if isinstance(result, np.ndarray):
            return pd.Series(result.ravel()[-1:], index=data.index[-1:])
        return pd.Series([float(result)], index=data.index[-1:])
    except:
        return pd.Series([0], index=data.index[-1:])

def before_trading_start(context):
    g.day_count += 1
    if g.day_count % g.rebalance_freq != 0 and g.day_count > 1:
        return
    
    stocks = list(get_all_securities(["stock"]).index)
    cd = get_current_data()
    stocks = [s for s in stocks if not cd[s].is_st and not cd[s].paused
              and s.startswith(("0","3","6"))]
    
    if len(stocks) < g.top_n:
        return
    
    end_date = context.previous_date
    start_date = end_date - pd.Timedelta(days=300)
    
    panel = get_price(stocks, start_date=start_date, end_date=end_date,
                      fields=["close","open","high","low","volume","money"],
                      frequency="daily", skip_paused=False, fq="pre")
    
    if panel is None or len(panel) < 50:
        return
    
    factor_scores = {}
    for fac in g.factors:
        values = {}
        for stock in stocks:
            if stock in panel.minor_axis:
                sd = panel.minor_xs(stock)
                if len(sd) >= 50:
                    val = compute_factor_np(sd, fac["formula"])
                    if len(val) > 0:
                        values[stock] = val.iloc[-1]
        if values:
            factor_scores[fac["name"]] = values
    
    if not factor_scores:
        return
    
    # Rank-product 综合
    combined = {}
    for stock in stocks:
        ranks = []
        for name, scores in factor_scores.items():
            if stock in scores:
                items = sorted(scores.items(), key=lambda x: x[1])
                r = next((i for i,(s,_) in enumerate(items) if s==stock), len(items)//2)
                ranks.append(r+1)
        if ranks:
            combined[stock] = np.prod(ranks)
    
    if combined:
        sorted_stocks = sorted(combined.items(), key=lambda x: x[1])
        g.security_list = [s for s,_ in sorted_stocks[:g.top_n]]
        g.weight = {s: 1.0/len(g.security_list) for s in g.security_list}
        log.info("选股: {}/{} 因子={}".format(len(g.security_list), len(stocks), len(factor_scores)))

def handle_data(context, data):
    if not hasattr(g,"security_list") or not g.security_list:
        return
    for sec in list(context.portfolio.positions.keys()):
        if sec not in g.security_list:
            order_target(sec, 0)
    for sec in g.security_list:
        target = context.portfolio.total_value * g.weight.get(sec, 0)
        if target > 0:
            order_target_value(sec, target)
