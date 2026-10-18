"""
中期优化新增因子 (10 个, 纯量价计算, 无需新数据源)
=====================================================
品类覆盖: momentum/turnover/sentiment/pattern/distribution/volatility
全部从 price_data (close/open/high/low/volume) 直接计算
"""

import numpy as np
import pandas as pd
from .base import BaseFactor


class PricePosition(BaseFactor):
    """60日价格位置: (close - 60d_low) / (60d_high - 60d_low)
    经济含义: 价格在60日箱体内的相对位置, 极端位置→反转
    方向: 负向 (A股反转: 高位→超买→未来跌)
    """
    def __init__(self):
        super().__init__("price_position", "momentum", "价格位置")

    def compute(self, price_data, window=60, **kwargs):
        close = price_data.get('close')
        high = price_data.get('high')
        low = price_data.get('low')
        if close is None or high is None or low is None:
            return None
        hh = high.rolling(window=window, min_periods=window).max()
        ll = low.rolling(window=window, min_periods=window).min()
        denom = hh - ll
        denom = denom.where(denom > 0, np.nan)
        position = (close - ll) / denom
        return -position.clip(0, 1)


class VolumeBreakout(BaseFactor):
    """量能突破: volume / MA(volume, 20)
    经济含义: 放量可能是出货信号 (A股特殊性)
    方向: 负向
    """
    def __init__(self):
        super().__init__("volume_breakout", "turnover", "量能突破")

    def compute(self, price_data, window=20, **kwargs):
        volume = price_data.get('volume')
        if volume is None:
            return None
        ma = volume.rolling(window=window, min_periods=window).mean()
        ratio = volume / ma.replace(0, np.nan)
        return -ratio.clip(0, 10)


class GapRatio(BaseFactor):
    """跳空比率: (open - prev_close) / prev_close
    经济含义: 向上跳空→缺口回补概率高
    方向: 负向
    """
    def __init__(self):
        super().__init__("gap_ratio", "sentiment", "跳空比率")

    def compute(self, price_data, **kwargs):
        close = price_data.get('close')
        o = price_data.get('open')
        if close is None or o is None:
            return None
        prev_close = close.shift(1)
        gap = (o - prev_close) / prev_close.replace(0, np.nan)
        return -gap.clip(-0.11, 0.11)


class UpperShadow(BaseFactor):
    """上影线: (high - max(open, close)) / (high - low + eps)
    经济含义: 长上影=卖压重→负向信号
    方向: 负向
    """
    def __init__(self):
        super().__init__("upper_shadow", "pattern", "上影线")

    def compute(self, price_data, window=5, **kwargs):
        close = price_data.get('close')
        high = price_data.get('high')
        low = price_data.get('low')
        o = price_data.get('open')
        if close is None or high is None or low is None or o is None:
            return None
        body_high = o.combine(close, max)
        denom = (high - low).replace(0, np.nan)
        shadow = (high - body_high) / denom
        return -shadow.rolling(window=window, min_periods=window).mean()


class LowerShadow(BaseFactor):
    """下影线: (min(open, close) - low) / (high - low + eps)
    经济含义: 长下影=支撑买盘→正向信号
    方向: 正向
    """
    def __init__(self):
        super().__init__("lower_shadow", "pattern", "下影线")

    def compute(self, price_data, window=5, **kwargs):
        close = price_data.get('close')
        high = price_data.get('high')
        low = price_data.get('low')
        o = price_data.get('open')
        if close is None or high is None or low is None or o is None:
            return None
        body_low = o.combine(close, min)
        denom = (high - low).replace(0, np.nan)
        shadow = (body_low - low) / denom
        return shadow.rolling(window=window, min_periods=window).mean()


class VolumeCV(BaseFactor):
    """成交量变异系数: std(volume, 20d) / mean(volume, 20d)
    经济含义: 成交量不稳定→投机性强→负向
    方向: 负向
    """
    def __init__(self):
        super().__init__("volume_cv", "turnover", "成交量变异")

    def compute(self, price_data, window=20, **kwargs):
        volume = price_data.get('volume')
        if volume is None:
            return None
        mean_v = volume.rolling(window=window, min_periods=window).mean()
        std_v = volume.rolling(window=window, min_periods=window).std()
        cv = std_v / mean_v.replace(0, np.nan)
        return -cv.clip(0, 5)


class RetAsymmetry(BaseFactor):
    """涨跌不对称: |mean(正收益)| / |mean(负收益)| - 1
    经济含义: 正向价格推力 vs 负向价格推力对比
    方向: 正负混合, 以正收益为主 (强势股有惯性但A股反转)
    方向: 负向 (涨停后反转)
    """
    def __init__(self):
        super().__init__("ret_asymmetry", "distribution", "涨跌不对称")

    def compute(self, price_data, window=20, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        ret = close.pct_change()
        pos_ret = ret.where(ret > 0, 0).rolling(window=window, min_periods=window).mean()
        neg_ret = ret.where(ret < 0, 0).abs().rolling(window=window, min_periods=window).mean()
        asym = pos_ret / neg_ret.replace(0, np.nan) - 1.0
        return -asym.clip(-5, 5)


class ClosePosition(BaseFactor):
    """收盘位置: (close - low) / (high - low + eps)
    经济含义: 收盘靠近日内高点→买方主导, 但A股反转→高位收盘次日回撤
    方向: 负向
    """
    def __init__(self):
        super().__init__("close_position", "pattern", "收盘位置")

    def compute(self, price_data, window=5, **kwargs):
        close = price_data.get('close')
        high = price_data.get('high')
        low = price_data.get('low')
        if close is None or high is None or low is None:
            return None
        denom = (high - low).replace(0, np.nan)
        pos = (close - low) / denom
        avg_pos = pos.rolling(window=window, min_periods=window).mean()
        return -avg_pos.clip(0, 1)


class DownsideCapture(BaseFactor):
    """下行捕获: sum(负收益) / sum(|收益|)
    经济含义: 下行参与度, 高下行参与→风险股→负向
    方向: 负向
    """
    def __init__(self):
        super().__init__("downside_capture", "volatility", "下行捕获")

    def compute(self, price_data, window=60, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        ret = close.pct_change()
        neg_ret = ret.where(ret < 0, 0).abs().rolling(window=window, min_periods=window).sum()
        abs_ret = ret.abs().rolling(window=window, min_periods=window).sum()
        capture = neg_ret / abs_ret.replace(0, np.nan)
        return -capture.clip(0, 1)


class RSRating(BaseFactor):
    """相对强度: 个股20日累计收益 - 等权市场20日累计收益
    经济含义: 跑赢市场→超买→反转 (A股特性)
    方向: 负向
    """
    def __init__(self):
        super().__init__("rs_rating", "momentum", "相对强度")

    def compute(self, price_data, window=20, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        cum_ret = close.pct_change().rolling(window=window, min_periods=window).apply(
            lambda x: (1 + x).prod() - 1, raw=False)
        mkt_ret = cum_ret.mean(axis=1)
        mkt_df = pd.DataFrame({col: mkt_ret for col in cum_ret.columns}, index=cum_ret.index)
        rs = cum_ret - mkt_df
        return -rs.clip(-1, 2)
