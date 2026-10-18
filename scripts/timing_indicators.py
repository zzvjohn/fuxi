# -*- coding: utf-8 -*-
"""
择时技术指标模块 — 浙商《技术指标优化择时》完整实现
============================================================
均线类(10个): SMA, EMA, EXPMA, EWMA, MACD, TRIX, BOLL, MTM, SAR, DMI, HL
超买超卖类(7个): WMS, KDJ, RSI, PSY, ARBR, MFI, VRSI

每个指标返回 (buy_signal, sell_signal):
  buy_signal:  1=买入, 0=中性/持有
  sell_signal: 1=卖出, 0=中性/持有

参数以元组传入，便于 GA 搜索
"""
import numpy as np
import pandas as pd


# ========== 均线类 ==========

def sma_signal(data, param=(20,)):
    """算术移动平均线: 价格>均线做多, <均线做空, 穿越时发信号"""
    n = int(param[0])
    close = data['close'].values
    ma = pd.Series(close).rolling(n).mean().values
    buy = np.where((close > ma) & (np.roll(close, 1) <= np.roll(ma, 1)), 1, 0)
    sell = np.where((close < ma) & (np.roll(close, 1) >= np.roll(ma, 1)), 1, 0)
    buy[0] = sell[0] = 0
    return buy, sell


def ema_signal(data, param=(20,)):
    """指数移动平均线"""
    n = int(param[0])
    close = data['close'].values
    ema = pd.Series(close).ewm(span=n, adjust=False).mean().values
    buy = np.where((close > ema) & (np.roll(close, 1) <= np.roll(ema, 1)), 1, 0)
    sell = np.where((close < ema) & (np.roll(close, 1) >= np.roll(ema, 1)), 1, 0)
    buy[0] = sell[0] = 0
    return buy, sell


def expma_signal(data, param=(20,)):
    """指数平均线 EXPMA = (2*Ct + (N-1)*EXPMA_{t-1}) / (N+1)"""
    n = int(param[0])
    close = data['close'].values
    alpha = 2.0 / (n + 1)
    expma = np.zeros_like(close)
    expma[0] = close[0]
    for i in range(1, len(close)):
        expma[i] = alpha * close[i] + (1 - alpha) * expma[i-1]
    buy = np.where((close > expma) & (np.roll(close, 1) <= np.roll(expma, 1)), 1, 0)
    sell = np.where((close < expma) & (np.roll(close, 1) >= np.roll(expma, 1)), 1, 0)
    buy[0] = sell[0] = 0
    return buy, sell


def ewma_signal(data, param=(20, 0.94)):
    """指数加权移动平均 EWMA = sum((1-lambda)*lambda^(t-i)*Ci) / (1-lambda^N)"""
    n, lam = int(param[0]), param[1]
    close = data['close'].values
    ewma = np.zeros_like(close)
    for i in range(len(close)):
        if i < n:
            ewma[i] = np.mean(close[:i+1])
        else:
            weights = np.array([(1-lam)*lam**j for j in range(n)])
            weights = weights / weights.sum()
            ewma[i] = np.sum(weights * close[i-n+1:i+1][::-1])
    buy = np.where((close > ewma) & (np.roll(close, 1) <= np.roll(ewma, 1)), 1, 0)
    sell = np.where((close < ewma) & (np.roll(close, 1) >= np.roll(ewma, 1)), 1, 0)
    buy[0] = sell[0] = 0
    return buy, sell


def ma_cross_signal(data, param=(5, 20)):
    """双均线交叉: 短期上穿长期=买入, 下穿=卖出"""
    n_short, n_long = int(param[0]), int(param[1])
    close = data['close'].values
    ma_s = pd.Series(close).rolling(n_short).mean().values
    ma_l = pd.Series(close).rolling(n_long).mean().values
    buy = np.where((ma_s > ma_l) & (np.roll(ma_s, 1) <= np.roll(ma_l, 1)), 1, 0)
    sell = np.where((ma_s < ma_l) & (np.roll(ma_s, 1) >= np.roll(ma_l, 1)), 1, 0)
    buy[:n_long] = sell[:n_long] = 0
    return buy, sell


def macd_signal(data, param=(12, 26, 9)):
    """MACD: DIF上穿DEA买入, 下穿卖出; 方法2: DIF>DEA且DIF>0=多头"""
    n1, n2, n3 = int(param[0]), int(param[1]), int(param[2])
    close = data['close'].values
    ema_s = pd.Series(close).ewm(span=n1, adjust=False).mean().values
    ema_l = pd.Series(close).ewm(span=n2, adjust=False).mean().values
    dif = ema_s - ema_l
    dea = pd.Series(dif).ewm(span=n3, adjust=False).mean().values
    buy = np.where((dif > dea) & (np.roll(dif, 1) <= np.roll(dea, 1)), 1, 0)
    sell = np.where((dif < dea) & (np.roll(dif, 1) >= np.roll(dea, 1)), 1, 0)
    buy[:n2] = sell[:n2] = 0
    return buy, sell


def macd_v2_signal(data, param=(12, 26, 9)):
    """MACD方法2: DIF>DEA 且 DIF>0 = 多头"""
    n1, n2, n3 = int(param[0]), int(param[1]), int(param[2])
    close = data['close'].values
    ema_s = pd.Series(close).ewm(span=n1, adjust=False).mean().values
    ema_l = pd.Series(close).ewm(span=n2, adjust=False).mean().values
    dif = ema_s - ema_l
    dea = pd.Series(dif).ewm(span=n3, adjust=False).mean().values
    bull = (dif > dea) & (dif > 0)
    buy = np.where(bull & ~np.roll(bull, 1), 1, 0)
    sell = np.where(~bull & np.roll(bull, 1), 1, 0)
    buy[0] = sell[0] = 0
    return buy, sell


def trix_signal(data, param=(12, 20)):
    """三重指数平滑平均线 TRIX上穿MATRIX买入"""
    n1, n2 = int(param[0]), int(param[1])
    close = data['close'].values
    # 三重EMA
    tr = close.copy()
    for _ in range(3):
        tr = pd.Series(tr).ewm(span=n1, adjust=False).mean().values
    trix = np.zeros_like(close)
    trix[1:] = 100 * (tr[1:] - tr[:-1]) / np.maximum(np.abs(tr[:-1]), 1e-8)
    matrix = pd.Series(trix).rolling(n2).mean().values
    buy = np.where((trix > matrix) & (np.roll(trix, 1) <= np.roll(matrix, 1)), 1, 0)
    sell = np.where((trix < matrix) & (np.roll(trix, 1) >= np.roll(matrix, 1)), 1, 0)
    buy[:n1+n2] = sell[:n1+n2] = 0
    return buy, sell


def boll_signal(data, param=(20, 2.0)):
    """BOLL线: 价穿上轨=卖出, 价跌穿下轨=买入"""
    n, k = int(param[0]), param[1]
    close = data['close'].values
    mid = pd.Series(close).rolling(n).mean().values
    std = pd.Series(close).rolling(n).std().values
    upper = mid + k * std
    lower = mid - k * std
    # 价格从上往下穿中轨或上轨 → 卖出
    sell1 = (close < mid) & (np.roll(close, 1) >= np.roll(mid, 1))
    sell2 = (close < upper) & (np.roll(close, 1) >= np.roll(upper, 1))
    # 价格从下往上穿下轨 → 买入
    buy = (close > lower) & (np.roll(close, 1) <= np.roll(lower, 1))
    sell = sell1 | sell2
    buy[:n] = sell[:n] = 0
    return buy.astype(int), sell.astype(int)


def boll_v2_signal(data, param=(20, 2.0)):
    """BOLL方法2: 价穿上轨卖出 + 价跌穿中轨卖出, 价穿中轨买入"""
    n, k = int(param[0]), param[1]
    close = data['close'].values
    mid = pd.Series(close).rolling(n).mean().values
    std = pd.Series(close).rolling(n).std().values
    upper = mid + k * std
    buy = (close > mid) & (np.roll(close, 1) <= np.roll(mid, 1))
    sell = ((close < upper) & (np.roll(close, 1) >= np.roll(upper, 1))) | \
           ((close < mid) & (np.roll(close, 1) >= np.roll(mid, 1)))
    buy[:n] = sell[:n] = 0
    return buy.astype(int), sell.astype(int)


def mtm_signal(data, param=(12, 6)):
    """动量指标: MTM>0多头; MTM上穿MTMMA买入"""
    n1, n2 = int(param[0]), int(param[1])
    close = data['close'].values
    mtm = np.zeros_like(close)
    mtm[n1:] = close[n1:] - close[:-n1]
    mtmma = pd.Series(mtm).rolling(n2).mean().values
    buy = np.where((mtm > mtmma) & (np.roll(mtm, 1) <= np.roll(mtmma, 1)), 1, 0)
    sell = np.where((mtm < mtmma) & (np.roll(mtm, 1) >= np.roll(mtmma, 1)), 1, 0)
    buy[:n1+n2] = sell[:n1+n2] = 0
    return buy, sell


def sar_signal(data, param=(4, 0.02, 0.02, 0.2)):
    """抛物线SAR: 价格下穿SAR卖出, 上穿SAR买入"""
    n, af0, step, af_max = int(param[0]), param[1], param[2], param[3]
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    
    sar = np.zeros(len(close))
    trend = np.ones(len(close))  # 1=涨势, -1=跌势
    ep = np.zeros(len(close))
    af = np.zeros(len(close))
    
    # 初始化
    sar[0] = low[0]
    trend[0] = 1
    ep[0] = high[0]
    af[0] = af0
    
    for i in range(1, len(close)):
        sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
        trend[i] = trend[i-1]
        ep[i] = ep[i-1]
        af[i] = af[i-1]
        
        if trend[i-1] == 1:  # 涨势
            if sar[i] > low[i]:
                trend[i] = -1
                sar[i] = max(high[max(0,i-n):i+1]) if i >= n else high[:i+1].max()
                ep[i] = low[i]
                af[i] = af0
            else:
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + step, af_max)
        else:  # 跌势
            if sar[i] < high[i]:
                trend[i] = 1
                sar[i] = min(low[max(0,i-n):i+1]) if i >= n else low[:i+1].min()
                ep[i] = high[i]
                af[i] = af0
            else:
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + step, af_max)
    
    buy = np.where((trend == 1) & (np.roll(trend, 1) == -1), 1, 0)
    sell = np.where((trend == -1) & (np.roll(trend, 1) == 1), 1, 0)
    buy[0] = sell[0] = 0
    return buy, sell


def dmi_signal(data, param=(7,)):
    """动向指标DMI: +DI上穿-DI买入, +DI下穿-DI卖出"""
    n = int(param[0])
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    
    tr = np.zeros(len(close))
    pdm = np.zeros(len(close))
    ndm = np.zeros(len(close))
    
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        pdm[i] = up if (up > 0 and up > down) else 0
        ndm[i] = down if (down > 0 and down > up) else 0
    
    tr_n = pd.Series(tr).rolling(n).sum().values
    pdm_n = pd.Series(pdm).rolling(n).sum().values
    ndm_n = pd.Series(ndm).rolling(n).sum().values
    
    pdi = np.where(tr_n > 0, 100 * pdm_n / tr_n, 0)
    ndi = np.where(tr_n > 0, 100 * ndm_n / tr_n, 0)
    
    buy = np.where((pdi > ndi) & (np.roll(pdi, 1) <= np.roll(ndi, 1)), 1, 0)
    sell = np.where((pdi < ndi) & (np.roll(pdi, 1) >= np.roll(ndi, 1)), 1, 0)
    buy[:n] = sell[:n] = 0
    return buy, sell


def hl_signal(data, param=(20,)):
    """高低线: 价格>HL买入, <HL卖出"""
    n = int(param[0])
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    hl = (pd.Series(high).rolling(n).max().values + pd.Series(low).rolling(n).min().values) / 2
    buy = np.where((close > hl) & (np.roll(close, 1) <= np.roll(hl, 1)), 1, 0)
    sell = np.where((close < hl) & (np.roll(close, 1) >= np.roll(hl, 1)), 1, 0)
    buy[:n] = sell[:n] = 0
    return buy, sell


# ========== 超买超卖类 ==========

def wms_signal(data, param=(14, 80, 20)):
    """威廉指标: WMS>80超卖买入, WMS<20超买卖出"""
    n, ub, lb = int(param[0]), param[1], param[2]
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    hh = pd.Series(high).rolling(n).max().values
    ll = pd.Series(low).rolling(n).min().values
    wms = np.where(hh != ll, 100 * (hh - close) / (hh - ll), 50)
    buy = np.where((wms > ub) & (np.roll(wms, 1) <= ub), 1, 0)
    sell = np.where((wms < lb) & (np.roll(wms, 1) >= lb), 1, 0)
    buy[:n] = sell[:n] = 0
    return buy, sell


def kdj_signal(data, param=(9, 3, 3)):
    """KDJ: K上穿D买入, K>80超买, K<20超卖; 使用交叉法"""
    n, m1, m2 = int(param[0]), int(param[1]), int(param[2])
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    hh = pd.Series(high).rolling(n).max().values
    ll = pd.Series(low).rolling(n).min().values
    rsv = np.where(hh != ll, 100 * (close - ll) / (hh - ll), 50)
    k = np.zeros_like(close); k[:n] = 50
    d = np.zeros_like(close); d[:n] = 50
    for i in range(n, len(close)):
        k[i] = (m1-1)/m1 * k[i-1] + 1/m1 * rsv[i]
        d[i] = (m2-1)/m2 * d[i-1] + 1/m2 * k[i]
    buy = np.where((k > d) & (np.roll(k, 1) <= np.roll(d, 1)), 1, 0)
    sell = np.where((k < d) & (np.roll(k, 1) >= np.roll(d, 1)), 1, 0)
    buy[:n] = sell[:n] = 0
    return buy, sell


def kdj_v2_signal(data, param=(9, 3, 3, 80, 20)):
    """KDJ方法2: K>ub超买=卖出, K<lb超卖=买入"""
    n, m1, m2, ub, lb = int(param[0]), int(param[1]), int(param[2]), param[3], param[4]
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    hh = pd.Series(high).rolling(n).max().values
    ll = pd.Series(low).rolling(n).min().values
    rsv = np.where(hh != ll, 100 * (close - ll) / (hh - ll), 50)
    k = np.zeros_like(close); k[:n] = 50
    for i in range(n, len(close)):
        k[i] = (m1-1)/m1 * k[i-1] + 1/m1 * rsv[i]
    buy = np.where((k < lb) & (np.roll(k, 1) >= lb), 1, 0)
    sell = np.where((k > ub) & (np.roll(k, 1) <= ub), 1, 0)
    buy[:n] = sell[:n] = 0
    return buy, sell


def rsi_signal(data, param=(14, 70, 30)):
    """RSI: RSI>70超买卖出, RSI<30超卖买入"""
    n, ub, lb = int(param[0]), param[1], param[2]
    close = data['close'].values
    diff = np.diff(close, prepend=close[0])
    gain = np.where(diff > 0, diff, 0)
    loss = np.where(diff < 0, -diff, 0)
    avg_gain = pd.Series(gain).rolling(n).mean().values
    avg_loss = pd.Series(loss).rolling(n).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - 100 / (1 + rs)
    buy = np.where((rsi < lb) & (np.roll(rsi, 1) >= lb), 1, 0)
    sell = np.where((rsi > ub) & (np.roll(rsi, 1) <= ub), 1, 0)
    buy[:n] = sell[:n] = 0
    return buy, sell


def rsi_cross_signal(data, param=(6, 24)):
    """RSI交叉法: 短期RSI上穿长期RSI买入"""
    ns, nl = int(param[0]), int(param[1])
    close = data['close'].values
    diff = np.diff(close, prepend=close[0])
    gain = np.where(diff > 0, diff, 0)
    loss = np.where(diff < 0, -diff, 0)
    avg_gain_s = pd.Series(gain).rolling(ns).mean().values
    avg_loss_s = pd.Series(loss).rolling(ns).mean().values
    avg_gain_l = pd.Series(gain).rolling(nl).mean().values
    avg_loss_l = pd.Series(loss).rolling(nl).mean().values
    rs_s = np.where(avg_loss_s > 0, avg_gain_s / avg_loss_s, 100)
    rs_l = np.where(avg_loss_l > 0, avg_gain_l / avg_loss_l, 100)
    rsi_s = 100 - 100 / (1 + rs_s)
    rsi_l = 100 - 100 / (1 + rs_l)
    buy = np.where((rsi_s > rsi_l) & (np.roll(rsi_s, 1) <= np.roll(rsi_l, 1)), 1, 0)
    sell = np.where((rsi_s < rsi_l) & (np.roll(rsi_s, 1) >= np.roll(rsi_l, 1)), 1, 0)
    buy[:nl] = sell[:nl] = 0
    return buy, sell


def psy_signal(data, param=(12, 25, 75)):
    """PSY心理线: PSY<25超卖买入, PSY>75超买卖出"""
    n, lb, ub = int(param[0]), param[1], param[2]
    close = data['close'].values
    up_days = pd.Series((close > np.roll(close, 1)).astype(float)).rolling(n).sum().values
    psy = 100 * up_days / n
    buy = np.where((psy < lb) & (np.roll(psy, 1) >= lb), 1, 0)
    sell = np.where((psy > ub) & (np.roll(psy, 1) <= ub), 1, 0)
    buy[:n] = sell[:n] = 0
    return buy, sell


def arbr_signal(data, param=(26, 60, 140, 50, 150)):
    """ARBR情绪指标: AR<60超卖买入, AR>140超买卖出"""
    n, ar_lb, ar_ub, br_lb, br_ub = int(param[0]), param[1], param[2], param[3], param[4]
    high = data['high'].values
    low = data['low'].values
    open_ = data['open'].values
    close = data['close'].values
    
    h_o = pd.Series(np.maximum(high - open_, 0)).rolling(n).sum().values
    o_l = pd.Series(np.maximum(open_ - low, 0)).rolling(n).sum().values
    ar = np.where(o_l > 0, 100 * h_o / o_l, 100)
    
    h_c = pd.Series(np.maximum(high - np.roll(close, 1), 0)).rolling(n).sum().values
    c_l = pd.Series(np.maximum(np.roll(close, 1) - low, 0)).rolling(n).sum().values
    br = np.where(c_l > 0, 100 * h_c / c_l, 100)
    
    buy_ar = np.where(ar < ar_lb, 1, 0)
    sell_ar = np.where(ar > ar_ub, 1, 0)
    buy_br = np.where(br < br_lb, 1, 0)
    sell_br = np.where(br > br_ub, 1, 0)
    
    buy = np.where((buy_ar | buy_br) & ~(np.roll(buy_ar | buy_br, 1)), 1, 0)
    sell = np.where((sell_ar | sell_br) & ~(np.roll(sell_ar | sell_br, 1)), 1, 0)
    buy[:n] = sell[:n] = 0
    return buy, sell


def mfi_signal(data, param=(14, 80, 20)):
    """MFI资金流量: MFI>80超买卖出, MFI<20超卖买入"""
    n, ub, lb = int(param[0]), param[1], param[2]
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    volume = data['volume'].values
    
    tp = (high + low + close) / 3
    mf = tp * volume
    
    pmf = np.zeros(len(close))
    nmf = np.zeros(len(close))
    for i in range(1, len(close)):
        if mf[i] > mf[i-1]:
            pmf[i] = mf[i]
        elif mf[i] < mf[i-1]:
            nmf[i] = mf[i]
    
    pmf_sum = pd.Series(pmf).rolling(n).sum().values
    nmf_sum = pd.Series(nmf).rolling(n).sum().values
    mfi = np.where(nmf_sum > 0, 100 - 100 / (1 + pmf_sum / nmf_sum), 50)
    
    buy = np.where((mfi < lb) & (np.roll(mfi, 1) >= lb), 1, 0)
    sell = np.where((mfi > ub) & (np.roll(mfi, 1) <= ub), 1, 0)
    buy[:n] = sell[:n] = 0
    return buy, sell


def vrsi_signal(data, param=(6, 85, 15)):
    """VRSI量相对强弱: VRSI>85超买卖出, VRSI<15超卖买入"""
    n, ub, lb = int(param[0]), param[1], param[2]
    volume = data['volume'].values
    diff_vol = np.diff(volume, prepend=volume[0])
    gain = np.where(diff_vol > 0, diff_vol, 0)
    loss = np.where(diff_vol < 0, -diff_vol, 0)
    avg_gain = pd.Series(gain).rolling(n).mean().values
    avg_loss = pd.Series(loss).rolling(n).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    vrsi = 100 - 100 / (1 + rs)
    buy = np.where((vrsi < lb) & (np.roll(vrsi, 1) >= lb), 1, 0)
    sell = np.where((vrsi > ub) & (np.roll(vrsi, 1) <= ub), 1, 0)
    buy[:n] = sell[:n] = 0
    return buy, sell


# ========== 指标注册表 ==========
INDICATOR_REGISTRY = {
    # 均线类
    'SMA':    (sma_signal,    (20,),            'SMA均线'),
    'EMA':    (ema_signal,    (20,),            'EMA均线'),
    'EXPMA':  (expma_signal,  (20,),            'EXPMA均线'),
    'EWMA':   (ewma_signal,   (20, 0.94),       'EWMA均线'),
    'MA_CROSS': (ma_cross_signal, (5, 20),      '双均线交叉'),
    'MACD':   (macd_signal,   (12, 26, 9),      'MACD'),
    'MACD_V2':(macd_v2_signal,(12, 26, 9),      'MACD方法2'),
    'TRIX':   (trix_signal,   (12, 20),         '三重指数平滑'),
    'BOLL':   (boll_signal,   (20, 2.0),        'BOLL线'),
    'BOLL_V2':(boll_v2_signal,(20, 2.0),        'BOLL线方法2'),
    'MTM':    (mtm_signal,    (12, 6),          '动量指标'),
    'SAR':    (sar_signal,    (4, 0.02, 0.02, 0.2), '抛物线SAR'),
    'DMI':    (dmi_signal,    (7,),             '动向指标'),
    'HL':     (hl_signal,     (20,),            '高低线'),
    # 超买超卖类
    'WMS':    (wms_signal,    (14, 80, 20),     '威廉指标'),
    'KDJ':    (kdj_signal,    (9, 3, 3),        'KDJ随机'),
    'KDJ_V2': (kdj_v2_signal, (9, 3, 3, 80, 20),'KDJ超买超卖'),
    'RSI':    (rsi_signal,    (14, 70, 30),     'RSI相对强弱'),
    'RSI_CR': (rsi_cross_signal, (6, 24),       'RSI交叉'),
    'PSY':    (psy_signal,    (12, 25, 75),     '心理线'),
    'ARBR':   (arbr_signal,   (26, 60, 140, 50, 150), '情绪指标'),
    'MFI':    (mfi_signal,    (14, 80, 20),     '资金流量'),
    'VRSI':   (vrsi_signal,   (6, 85, 15),      '量相对强弱'),
}


def get_all_signals(data, indicator_names=None, params_dict=None):
    """
    批量计算所有指标信号
    data: DataFrame with columns: open, high, low, close, volume
    indicator_names: list of indicator names, if None use all
    params_dict: {name: param_tuple} override default params
    
    Returns: dict of {name: (buy_array, sell_array)}
    """
    if indicator_names is None:
        indicator_names = list(INDICATOR_REGISTRY.keys())
    
    signals = {}
    for name in indicator_names:
        func, default_params, _ = INDICATOR_REGISTRY[name]
        params = params_dict.get(name, default_params) if params_dict else default_params
        buy, sell = func(data, params)
        signals[name] = (buy, sell)
    return signals
