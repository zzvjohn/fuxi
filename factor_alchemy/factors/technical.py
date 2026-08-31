"""
技术因子 (jqfactor 启发)
==========================
从聚宽内置因子库中挑选，补充现有34因子体系中缺失的量价因子。

新增因子:
  - streak: 连涨/连跌天数 (从聚宽streak启发)
  - rsi_14: 14日RSI相对强弱
  - high_low_range: 高低价振幅 (日内波动范围)
  - vpt: 量价趋势 (Volume-Price Trend)
  - short_rev_5d: 5日短期反转
  - boll_pct_b: 布林带%B位置
  - volume_ratio: 量比 (当日成交量 / 5日均量)
"""
import numpy as np
import pandas as pd
from .base import BaseFactor


class Streak(BaseFactor):
    """
    连涨/连跌天数
    
    计算: 从最近一日向前数，连续同向（收盘>开盘=涨, 收盘<开盘=跌）的天数
    方向: 正向(涨天数多→看好), 负向(跌天数多→看空)
    
    聚宽因子: streak (情绪类)
    我们的Phase 2: ICIR=+2.16, 最强因子
    """
    def __init__(self):
        super().__init__("streak", "momentum", "连涨天数")
    def compute(self, price_data, **kwargs):
        close = price_data.get('close')
        open_ = price_data.get('open')
        
        if close is None or open_ is None:
            return None
        
        df = close.copy()
        df[:] = np.nan
        
        for col in df.columns:
            if col not in open_.columns:
                continue
            c = close[col].values
            o = open_[col].values
            
            streak = np.zeros(len(c))
            streak[0] = 0
            
            for i in range(1, len(c)):
                if c[i] > o[i]:  # 当日收阳
                    if c[i-1] > o[i-1]:
                        streak[i] = streak[i-1] + 1
                    else:
                        streak[i] = 1
                elif c[i] < o[i]:  # 当日收阴
                    if c[i-1] < o[i-1]:
                        streak[i] = streak[i-1] - 1
                    else:
                        streak[i] = -1
                else:
                    streak[i] = 0  # 平盘
            
            df[col] = streak
        
        return df


class RSI14(BaseFactor):
    """
    14日相对强弱指标 (RSI)
    
    计算: RSI = 100 - 100/(1 + RS), RS = 14日平均涨幅 / 14日平均跌幅
    方向: 正向(高RSI→超买, 但动量持续), A股更偏反转
    
    聚宽因子: rsi_14 (技术类)
    """
    def __init__(self):
        super().__init__("rsi_14", "momentum", "RSI相对强弱")
    def compute(self, price_data, window=14, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        df = close.copy()
        df[:] = np.nan
        
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        
        # 反转向: A股高RSI→超买→未来收益低
        return -rsi  # 取负号, 高分=低RSI=超卖


class HighLowRange(BaseFactor):
    """
    高低价振幅
    
    计算: (High - Low) / Close, 20日均值
    方向: 负向(高振幅→高波动→高风险的彩票型特征)
    
    聚宽因子: high_low (波动类)
    """
    def __init__(self):
        super().__init__("high_low_range", "volatility", "高低价振幅")
    def compute(self, price_data, window=20, **kwargs):
        close = price_data.get('close')
        high = price_data.get('high')
        low = price_data.get('low')
        
        if close is None or high is None or low is None:
            return None
        
        daily_range = (high - low) / close
        avg_range = daily_range.rolling(window=window, min_periods=window//2).mean()
        
        # 负向: 高振幅=坏
        return -avg_range


class VPT(BaseFactor):
    """
    Volume-Price Trend (量价趋势)
    
    计算: VPT_t = VPT_{t-1} + Volume_t * (Close_t - Close_{t-1}) / Close_{t-1}
    方向: 正向(量价配合上涨)
    
    聚宽因子: vpt (量价类)
    """
    def __init__(self):
        super().__init__("vpt", "liquidity", "量价趋势")
    def compute(self, price_data, **kwargs):
        close = price_data.get('close')
        volume = price_data.get('volume')
        
        if close is None or volume is None:
            return None
        
        df = close.copy()
        df[:] = np.nan
        
        pct_change = close.pct_change().fillna(0)
        
        for col in df.columns:
            if col not in volume.columns:
                continue
            v = volume[col].values
            pc = pct_change[col].values
            
            vpt = np.zeros(len(v))
            for i in range(1, len(v)):
                vpt[i] = vpt[i-1] + v[i] * pc[i]
            
            df[col] = vpt
        
        return df


class ShortRev5D(BaseFactor):
    """
    5日短期反转
    
    计算: 过去5个交易日的累计收益
    方向: 负向(反转: 短期涨→未来跌)
    
    聚宽因子: reversal_5d (反转类)
    A股典型行为: 短期反转极强
    """
    def __init__(self):
        super().__init__("short_rev_5d", "momentum", "5日反转")
    def compute(self, price_data, window=5, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        ret_5d = close.pct_change(periods=window)
        
        # 反转: 正向收益→负向预测
        return -ret_5d


class BollPctB(BaseFactor):
    """
    布林带 %B 位置
    
    计算: %B = (Close - Lower) / (Upper - Lower)
    其中 Upper/Lower = MA20 ± 2*σ20
    方向: 负向(高位→超买→反转, A股反转逻辑)
    
    聚宽因子: boll (技术类)
    """
    def __init__(self):
        super().__init__("boll_pct_b", "volatility", "布林带位置")
    def compute(self, price_data, window=20, num_std=2.0, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        ma = close.rolling(window=window, min_periods=window).mean()
        std = close.rolling(window=window, min_periods=window).std()
        
        upper = ma + num_std * std
        lower = ma - num_std * std
        
        pct_b = (close - lower) / (upper - lower)
        pct_b = pct_b.clip(-0.5, 1.5)
        
        # 负向: %B高→超买
        return -pct_b


class VolumeRatio(BaseFactor):
    """
    量比 (Volume Ratio)
    
    计算: 当日成交量 / 5日均量
    方向: 负向(异常放量→可能是主力出货)
    
    聚宽因子: volume_ratio (量价类)
    """
    def __init__(self):
        super().__init__("volume_ratio", "turnover", "量比")
    def compute(self, price_data, window=5, **kwargs):
        volume = price_data.get('volume')
        if volume is None:
            return None
        
        avg_vol = volume.rolling(window=window, min_periods=window).mean()
        ratio = volume / avg_vol.replace(0, np.nan)
        
        # 负向: 放量→可能是出货信号
        return -ratio


class Overnight5D(BaseFactor):
    """
    近5日隔夜收益 (close→next_open)
    
    计算: sum(open_t / close_{t-1} - 1) for t in [-5, -1]
    方向: 正向(高隔夜收益→质量信号, 异象因子)
    ICIR均值0.61, CV=0.22 (跨周期最稳定之一)
    """
    def __init__(self):
        super().__init__("overnight_5d", "momentum", "隔夜收益5D")

    def compute(self, price_data, **kwargs):
        close = price_data.get('close')
        open_ = price_data.get('open')
        if close is None or open_ is None:
            return None
        
        df = close.copy()
        df[:] = np.nan
        
        for col in df.columns:
            if col not in open_.columns:
                continue
            c = close[col].values
            o = open_[col].values
            n = min(len(c), len(o))
            if n < 7:
                continue
            
            overnight = np.full(n, np.nan)
            for t in range(1, n):
                if c[t-1] > 0 and o[t] > 0:
                    overnight[t] = o[t] / c[t-1] - 1
            
            df[col] = pd.Series(overnight, index=close.index).rolling(5, min_periods=3).sum()
        
        return df


class MinRet1M(BaseFactor):
    """
    近20日最低日收益
    
    计算: min(daily_ret) over [-20, -1]
    方向: 正向(min_ret本身为负, rank标准化处理方向; 极端下跌少→质量高)
    ICIR均值0.45, 尾部统计因子
    """
    def __init__(self):
        super().__init__("min_ret_1m", "momentum", "月最低日收益")

    def compute(self, price_data, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        rets = close.pct_change()
        min_ret = rets.rolling(20, min_periods=15).min()
        return min_ret
