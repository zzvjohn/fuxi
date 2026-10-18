"""
高级技术因子 (聚宽因子库二次补充)
================================
系统对标聚宽 200+ 因子库, 补全量价/情绪/分布/形态四大缺失品类。

新增 20 因子, 分 5 类:

趋势延伸 (trend):
  - bias_20: 20日乖离率 (价格偏离MA20的程度)
  - roc_20: 20日变动速率 (normalized price change rate)
  - cci_14: 14日顺势指标 (Commodity Channel Index)
  - plrc_12: 价格线性回归斜率
  - price_1m: 价格vs月度均值 (偏离度)

情绪因子 (sentiment) — 全新品类:
  - vr_26: VR成交量比率 (上涨量/下跌量, 26日)
  - vroc_12: 成交量变动速率
  - psy_12: 心理线 (12日上涨天数占比)
  - money_flow_20: 资金流量 (MFI变体)
  - davol_20: 换手率异动 (20日/120日换手比)

收益分布 (distribution) — 全新品类:
  - skewness_20: 收益偏度 (彩票型识别)
  - kurtosis_20: 收益峰度 (极端事件)
  - sharpe_20: 滚动夏普比率
  - atr_14: 均幅指标 (volatility替代)

形态因子 (pattern) — 全新品类:
  - bull_power: 多头力道 (Elder)
  - bear_power: 空头力道 (Elder)
  - high_52w_rank: 52周高点接近度
  - rank_1m: 1月收益排名反转

成交量结构 (volume_structure) — 补全:
  - vm_diff: VMACD差值
  - tvma_20: 20日成交额均线 (量能)

参考: https://www.joinquant.com/help/api/help#name:factor_values
"""
import numpy as np
import pandas as pd
from .base import BaseFactor


# ============================================================
# 趋势延伸因子
# ============================================================

class Bias20(BaseFactor):
    """
    20日乖离率
    
    计算: (Close - MA20) / MA20 * 100
    方向: 负向 (A股反转: 正乖离→超买→未来跌)
    
    聚宽: BIAS20
    """
    def __init__(self):
        super().__init__("bias_20", "momentum", "乖离率")
    def compute(self, price_data, window=20, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        ma = close.rolling(window=window, min_periods=window).mean()
        bias = (close - ma) / ma
        return -bias  # 反转方向


class ROC20(BaseFactor):
    """
    20日变动速率 (Price Rate of Change)
    
    计算: (Close_t - Close_{t-N}) / Close_{t-N}
    与普通收益率不同: ROC 不累积, 只取端点比较
    方向: 负向 (反转)
    
    聚宽: ROC20
    """
    def __init__(self):
        super().__init__("roc_20", "momentum", "变动速率")
    def compute(self, price_data, window=20, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        roc = close.pct_change(periods=window)
        return -roc  # 反转


class CCI14(BaseFactor):
    """
    14日顺势指标 (Commodity Channel Index)
    
    计算: CCI = (TP - MA(TP,14)) / (0.015 * MeanDeviation(TP,14))
    其中 TP = (High + Low + Close) / 3
    方向: 负向 (高CCI→超买→反转)
    
    聚宽: CCI14 (技术因子-动量)
    NOTE: MAD使用向量化 mad = rolling(|TP - rolling_mean(TP)|).mean()
    """
    def __init__(self):
        super().__init__("cci_14", "momentum", "顺势指标")
    def compute(self, price_data, window=14, **kwargs):
        close = price_data.get('close')
        high = price_data.get('high')
        low = price_data.get('low')
        if close is None or high is None or low is None:
            return None
        
        tp = (high + low + close) / 3.0
        ma = tp.rolling(window=window, min_periods=window).mean()
        # 向量化 MAD: mean(|x - mean(x)|)
        mad = (tp - ma).abs().rolling(window=window, min_periods=window).mean()
        
        cci = (tp - ma) / (0.015 * mad.replace(0, np.nan))
        return -cci.clip(-300, 300)  # 反转, 限制极端值


class PLRC12(BaseFactor):
    """
    12日价格线性回归斜率
    
    计算: 对 Close/mean(Close) 与日期序号做OLS, 取斜率beta
    方向: 正向 (价格趋势向上→动量延续)
    
    聚宽: PLRC12 (技术因子-动量)
    NOTE: 使用向量化OLS公式加速
    """
    def __init__(self):
        super().__init__("plrc_12", "momentum", "价格回归斜率")
    def compute(self, price_data, window=12, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        # 滚动mean
        roll_mean = close.rolling(window=window, min_periods=window).mean()
        
        # 标准化: Y = Close / rolling_mean
        y = close / roll_mean
        
        # 向量化OLS斜率: slope = sum((x-x̄)(y-ȳ)) / sum((x-x̄)^2)
        # 其中 x = 1..window, y = 标准化后的close
        t = np.arange(1, window + 1, dtype=float)
        t_mean = t.mean()
        t_demeaned = t - t_mean
        sum_sq_t = np.sum(t_demeaned ** 2)
        
        # 滚动sum of (x-x̄)*(y-ȳ) = sum(x*y) - x̄*sum(y)
        # Since x is the same for each window, we can compute rolling weighted sum
        weights = t_demeaned / sum_sq_t
        weights_rev = weights[::-1]  # most recent gets highest weight
        
        # Apply convolution for slope (raw=True passes numpy array)
        def _conv_slope(arr):
            return np.convolve(arr, weights_rev, mode='full')[:len(arr)]
        
        slope = y.fillna(0).apply(_conv_slope, raw=True)
        slope.iloc[:window-1] = np.nan
        slope.iloc[:window-1] = np.nan
        
        return slope


class Price1M(BaseFactor):
    """
    价格vs月度均值偏离
    
    计算: Close / mean(Close, 21) - 1
    方向: 负向 (高于均值→可能回落)
    
    聚宽: Price1M (动量因子)
    """
    def __init__(self):
        super().__init__("price_1m", "momentum", "价格月度偏离")
    def compute(self, price_data, window=21, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        ma = close.rolling(window=window, min_periods=window).mean()
        dev = close / ma - 1.0
        return -dev  # 反转


# ============================================================
# 情绪因子 (全新品类)
# ============================================================

class VR26(BaseFactor):
    """
    VR成交量比率 (Volume Ratio)
    
    计算: VR = (AVS + 0.5*CVS) / (BVS + 0.5*CVS) * 100
    AVS = N日内上涨日成交量之和
    BVS = N日内下跌日成交量之和
    CVS = N日内平盘日成交量之和
    方向: 负向 (高VR→过度放量上涨→反转)
    
    聚宽: VR (情绪因子)
    """
    def __init__(self):
        super().__init__("vr_26", "sentiment", "VR量比")
    def compute(self, price_data, window=26, **kwargs):
        close = price_data.get('close')
        open_ = price_data.get('open')
        volume = price_data.get('volume')
        
        if close is None or open_ is None or volume is None:
            return None
        
        # 多空判定
        up_mask = close > open_
        down_mask = close < open_
        flat_mask = close == open_
        
        up_vol = volume.where(up_mask, 0)
        down_vol = volume.where(down_mask, 0)
        flat_vol = volume.where(flat_mask, 0)
        
        avs = up_vol.rolling(window=window, min_periods=window).sum()
        bvs = down_vol.rolling(window=window, min_periods=window).sum()
        cvs = flat_vol.rolling(window=window, min_periods=window).sum()
        
        denom = bvs + 0.5 * cvs
        vr_val = (avs + 0.5 * cvs) / denom.replace(0, np.nan)
        
        return -vr_val.clip(0, 500)  # 反转: 高VR→超买


class VROC12(BaseFactor):
    """
    成交量变动速率
    
    计算: (Volume_t - Volume_{t-N}) / Volume_{t-N} * 100
    方向: 负向 (异常放量→出货信号)
    
    聚宽: VROC12 (情绪因子)
    """
    def __init__(self):
        super().__init__("vroc_12", "sentiment", "量变动速率")
    def compute(self, price_data, window=12, **kwargs):
        volume = price_data.get('volume')
        if volume is None:
            return None
        
        vroc = volume.pct_change(periods=window)
        return -vroc.clip(-1, 10)  # 反转, 限制极值


class PSY12(BaseFactor):
    """
    心理线 (Psychological Line)
    
    计算: N日内上涨天数 / N * 100
    方向: 负向 (过度乐观→反转)
    
    聚宽: PSY (情绪因子)
    """
    def __init__(self):
        super().__init__("psy_12", "sentiment", "心理线")
    def compute(self, price_data, window=12, **kwargs):
        close = price_data.get('close')
        ret = close.pct_change().fillna(0)
        
        up_days = (ret > 0).astype(float)
        psy = up_days.rolling(window=window, min_periods=window).mean()
        
        return -psy  # 反转: 高乐观→反转


class MoneyFlow20(BaseFactor):
    """
    资金流量 (MFI变体)
    
    计算: 典型价格 * 成交量 的N日累计
    TP = (High + Low + Close) / 3
    方向: 负向 (资金持续流入→可能过热)
    
    聚宽: money_flow_20 (情绪因子)
    """
    def __init__(self):
        super().__init__("money_flow_20", "sentiment", "资金流量")
    def compute(self, price_data, window=20, **kwargs):
        close = price_data.get('close')
        high = price_data.get('high')
        low = price_data.get('low')
        volume = price_data.get('volume')
        
        if close is None or high is None or low is None or volume is None:
            return None
        
        tp = (high + low + close) / 3.0
        mf = tp * volume
        
        # 标准化: 除以自身滚动均值
        mf_ma = mf.rolling(window=window, min_periods=window).mean()
        mf_norm = mf / mf_ma.replace(0, np.nan) - 1.0
        
        return -mf_norm.clip(-0.9, 5)


class DAVOL20(BaseFactor):
    """
    换手率异动
    
    计算: 20日平均换手率 / 120日平均换手率
    方向: 负向 (放量异动→可能是出货)
    
    聚宽: DAVOL20 (情绪因子)
    """
    def __init__(self):
        super().__init__("davol_20", "sentiment", "换手异动")
    def compute(self, price_data, short=20, long=120, **kwargs):
        volume = price_data.get('volume')
        if volume is None:
            return None
        
        # 需要总股本 → 用日均成交额/价格估计换手
        # 简化: 直接用量比替代
        close = price_data.get('close')
        if close is not None:
            money = volume * close
            short_ma = money.rolling(window=short, min_periods=short).mean()
            long_ma = money.rolling(window=long, min_periods=long).mean()
            davol = short_ma / long_ma.replace(0, np.nan)
        else:
            short_ma = volume.rolling(window=short, min_periods=short).mean()
            long_ma = volume.rolling(window=long, min_periods=long).mean()
            davol = short_ma / long_ma.replace(0, np.nan)
        
        return -davol.clip(0, 10)


# ============================================================
# 收益分布因子 (全新品类)
# ============================================================

class Skewness20(BaseFactor):
    """
    20日收益偏度
    
    计算: 过去20个交易日收益率的三阶中心矩
    方向: 负向 (正偏→彩票型→未来收益低)
    
    聚宽: Skewness20 (风险因子)
    NOTE: 使用pandas内置 rolling().skew() 加速
    """
    def __init__(self):
        super().__init__("skewness_20", "distribution", "收益偏度")
    def compute(self, price_data, window=20, min_p=10, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        ret = close.pct_change()
        skew = ret.rolling(window=window, min_periods=min_p).skew()
        
        return -skew  # 反转: 正偏→彩票→低收益


class Kurtosis20(BaseFactor):
    """
    20日收益峰度
    
    计算: 过去20个交易日收益率的四阶中心矩 (excess kurtosis)
    方向: 负向 (高峰度→极端值多→风险高)
    
    聚宽: Kurtosis20 (风险因子)
    NOTE: 使用pandas内置 rolling().kurt() 加速
    """
    def __init__(self):
        super().__init__("kurtosis_20", "distribution", "收益峰度")
    def compute(self, price_data, window=20, min_p=10, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        ret = close.pct_change()
        kurt = ret.rolling(window=window, min_periods=min_p).kurt()
        
        return -kurt  # 反转: 高峰度→极端风险


class Sharpe20(BaseFactor):
    """
    20日滚动夏普比率
    
    计算: (年化收益 - Rf) / 年化波动
    方向: 正向 (高Sharpe→高质量)
    
    聚宽: sharpe_ratio_20 (风险因子)
    """
    def __init__(self):
        super().__init__("sharpe_20", "distribution", "夏普比率")
    def compute(self, price_data, window=20, rf=0.04, min_p=10, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        ret = close.pct_change()
        mu = ret.rolling(window=window, min_periods=min_p).mean() * 252
        sigma = ret.rolling(window=window, min_periods=min_p).std() * np.sqrt(252)
        
        sharpe = (mu - rf) / sigma.replace(0, np.nan)
        return sharpe.clip(-5, 5)


class ATR14(BaseFactor):
    """
    14日均幅指标 (Average True Range)
    
    计算: TR = max(H-L, |H-C_prev|, |L-C_prev|), ATR = EMA(TR, 14)
    方向: 负向 (高ATR→高波动→高风险的彩票型)
    
    聚宽: ATR14 (情绪因子)
    """
    def __init__(self):
        super().__init__("atr_14", "distribution", "均幅指标")
    def compute(self, price_data, window=14, **kwargs):
        close = price_data.get('close')
        high = price_data.get('high')
        low = price_data.get('low')
        
        if close is None or high is None or low is None:
            return None
        
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.DataFrame(
            np.maximum(np.maximum(tr1.values, tr2.values), tr3.values),
            index=tr1.index, columns=tr1.columns
        )
        
        # 标准化: ATR / Close
        atr = tr.ewm(span=window, min_periods=window).mean()
        atr_norm = atr / close
        
        return -atr_norm  # 反转: 高波动→低收益


# ============================================================
# 形态因子 (全新品类)
# ============================================================

class BullPower(BaseFactor):
    """
    多头力道 (Elder Ray)
    
    计算: (最高价 - EMA(close, 13)) / close
    方向: 负向 (过高→超买→反转)
    
    聚宽: bull_power (动量因子)
    """
    def __init__(self):
        super().__init__("bull_power", "pattern", "多头力道")
    def compute(self, price_data, window=13, **kwargs):
        close = price_data.get('close')
        high = price_data.get('high')
        if close is None or high is None:
            return None
        
        ema = close.ewm(span=window, min_periods=window).mean()
        bp = (high - ema) / close
        
        return -bp  # 反转: 过强多头→回落


class BearPower(BaseFactor):
    """
    空头力道 (Elder Ray)
    
    计算: (最低价 - EMA(close, 13)) / close
    方向: 正向 (过低→超卖→反弹)
    
    聚宽: bear_power (动量因子)
    """
    def __init__(self):
        super().__init__("bear_power", "pattern", "空头力道")
    def compute(self, price_data, window=13, **kwargs):
        close = price_data.get('close')
        low = price_data.get('low')
        if close is None or low is None:
            return None
        
        ema = close.ewm(span=window, min_periods=window).mean()
        bp = (low - ema) / close
        
        return bp  # 正向: 负值大→超卖→反弹


class High52WRank(BaseFactor):
    """
    52周高点接近度
    
    计算: Close / max(Close, 250) 的排名百分位
    方向: 负向 (接近新高→锚定效应→反转)
    
    聚宽: fifty_two_week_close_rank (动量因子)
    """
    def __init__(self):
        super().__init__("high_52w_rank", "pattern", "52周高点接近")
    def compute(self, price_data, window=250, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        high_52w = close.rolling(window=window, min_periods=min(60, window)).max()
        proximity = close / high_52w
        
        return -proximity  # 反转


class Rank1M(BaseFactor):
    """
    1月收益排名反转
    
    计算: 截面百分位排名 (0-1), 高收益→高排名
    方向: 正向 — 用排名百分位本身 (过去涨得好的股票, 排名高)
    
    聚宽: Rank1M (动量因子)
    NOTE: 向量化 rank(axis=1) 加速
    """
    def __init__(self):
        super().__init__("rank_1m", "pattern", "收益排名反转")
    def compute(self, price_data, window=21, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        ret_1m = close.pct_change(periods=window)
        
        # 向量化: 每行做百分位排名
        result = ret_1m.rank(axis=1, pct=True)
        
        return result  # 0-1之间, IC方向由GA自动决定


# ============================================================
# 成交量结构因子
# ============================================================

class VMDiff(BaseFactor):
    """
    VMACD 差值 (Volume MACD)
    
    计算: EMA(Vol, 12) - EMA(Vol, 26)
    方向: 负向 (量MACD金叉放量→可能是拉升出货)
    
    聚宽: VMACD / VDIFF (情绪因子)
    """
    def __init__(self):
        super().__init__("vm_diff", "volume_structure", "量MACD差值")
    def compute(self, price_data, short=12, long=26, **kwargs):
        volume = price_data.get('volume')
        if volume is None:
            return None
        
        ema_s = volume.ewm(span=short, min_periods=short).mean()
        ema_l = volume.ewm(span=long, min_periods=long).mean()
        
        diff = ema_s - ema_l
        # 标准化: 除以长期均值
        diff_norm = diff / ema_l.replace(0, np.nan)
        
        return -diff_norm.clip(-5, 5)


class TVMA20(BaseFactor):
    """
    20日成交额均线比
    
    计算: 当日成交额 / 20日均成交额
    方向: 负向 (放量→出货信号)
    
    聚宽: TVMA20 (情绪因子)
    """
    def __init__(self):
        super().__init__("tvma_20", "volume_structure", "成交额均线比")
    def compute(self, price_data, window=20, **kwargs):
        volume = price_data.get('volume')
        close = price_data.get('close')
        
        if volume is None:
            return None
        
        # 成交额 = volume * close (如果有close)
        if close is not None:
            money = volume * close
        else:
            money = volume
        
        ma = money.rolling(window=window, min_periods=window).mean()
        ratio = money / ma.replace(0, np.nan)
        
        return -ratio.clip(0, 10)


# ============================================================
# Phase 2 独立因子补充 (v5)
# ============================================================

class High52WDist(BaseFactor):
    """
    52周高点距离 (Phase 2 独立因子)
    
    计算: (rolling_high_250 - close) / rolling_high_250
    方向: 正向 (远离高点→锚定修正→反弹)
    
    区别于 high_52w_rank = close/max (接近度), 
    这是距离百分比, 与 Phase 2 的 high_52w_dist 完全一致
    """
    def __init__(self):
        super().__init__("high_52w_dist", "distribution", "52周高点距离")
    def compute(self, price_data, window=250, min_p=60, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        high_52w = close.rolling(window=window, min_periods=min_p).max()
        dist = (high_52w - close) / high_52w.replace(0, np.nan)
        
        return dist.clip(0, 1)  # 0=新高, 1=腰斩; 正向预期


class Skew1M(BaseFactor):
    """
    1月收益偏度 (Phase 2 独立因子)
    
    计算: 过去21个交易日收益率的三阶中心矩
    方向: 负向 (正偏→彩票型→未来收益低)
    
    区别于 skewness_20 (20日窗口), 此为21日, 
    与 Phase 2 的 skew_1m 完全一致
    """
    def __init__(self):
        super().__init__("skew_1m", "distribution", "1月收益偏度")
    def compute(self, price_data, window=21, min_p=10, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        ret = close.pct_change()
        skew = ret.rolling(window=window, min_periods=min_p).skew()
        
        return -skew  # 反转: 正偏→彩票→低收益


class GapUp(BaseFactor):
    """
    跳空缺口持续性 (XQuant Ch9 第1轮通过)
    
    计算: 当日开盘价 / 前日收盘价 - 1
    方向: 正向 (向上跳空→信息冲击→短期动量延续)
    
    来源: weekly_factor_hypothesis #1 PASS (ICIR=+0.437, +IC%=67.3%)
    """
    def __init__(self):
        super().__init__("gap_up", "momentum", "跳空缺口持续性")
    
    def compute(self, price_data, **kwargs):
        close = price_data.get('close')
        open_p = price_data.get('open')
        if close is None or open_p is None:
            return None
        return open_p / close.shift(1) - 1


class PanicSelling(BaseFactor):
    """
    恐慌抛售代理 (XQuant Ch9 第2轮通过)
    
    计算: 高成交(vol ratio > 1) * 价格下跌(down 3d) → 做多恐慌底部
    方向: 正向 (恐慌抛售→筹码换手→短期修复)
    
    来源: weekly_factor_hypothesis #2 PASS (ICIR=+0.454, +IC%=76.0%)
    """
    def __init__(self):
        super().__init__("panic_selling", "behavioral", "恐慌抛售代理")
    
    def compute(self, price_data, **kwargs):
        close = price_data.get('close')
        volume = price_data.get('volume')
        if close is None or volume is None:
            return None
        
        chg_3d = close.pct_change(3)
        is_down = (chg_3d < 0).astype(float)
        is_down.replace(0, np.nan, inplace=True)
        
        vol_ratio = volume / volume.rolling(3).mean()
        panic = vol_ratio * is_down
        
        return panic  # 正向: 恐慌高→未来反弹


# ============================================================
# 日频因子假设实验通过因子 (v2 原创去重)
# ============================================================

class TrendPersistenceScore(BaseFactor):
    """
    趋势持续性评分 (日频因子实验 v2 通过)
    
    计算: 分形效率(Path Ratio) × 20日胜率
      - 分形效率 = |ret_20d| / sum(|ret_daily|, 20d)  — 趋势的"直线度"
      - 胜率 = fraction(positive_daily_ret, 20d)       — 趋势的"稳定性"
    方向: 正向 (高效率+高胜率=强趋势→动量延续)
    
    来源: daily_factor_hypothesis v2 PASS (ICIR=+0.396, +IC%=68.5%)
    日期: 2026-06-15
    """
    def __init__(self):
        super().__init__("trend_persistence_score", "trend_quality", "趋势持续性评分")
    
    def compute(self, price_data, window=20, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        ret = close.pct_change()
        
        # 分形效率: |总收益| / 路径总长度
        total_ret = ret.rolling(window=window, min_periods=window//2).sum().abs()
        path_length = ret.abs().rolling(window=window, min_periods=window//2).sum()
        fractal_eff = total_ret / path_length.replace(0, np.nan)
        
        # 胜率: 正收益占比
        win_rate = (ret > 0).astype(float).rolling(window=window, min_periods=window//2).mean()
        
        # 综合评分 = 效率 × 胜率
        score = fractal_eff * win_rate
        
        return score.clip(0, 1)


class AttentionDecay(BaseFactor):
    """
    注意力衰减 (日频因子实验 v2 通过, 2026-06-20)
    
    计算: -(volume_5d_max / volume_20d_mean - 1).shift(3)
      - 极端放量事件后第3天是反转窗口 (Barinov 2014 注意力驱动理论)
      - 放量吸引散户注意力→短期高估→第3天开始价格修复
    方向: 正向 (放量后第3天的反转: 之前高估→现在回归)
    
    来源: daily_factor_hypothesis v2 PASS (ICIR=+0.335, +IC%=63.4%)
    日期: 2026-06-20
    """
    def __init__(self):
        super().__init__("attention_decay", "behavioral", "注意力衰减")
    
    def compute(self, price_data, **kwargs):
        volume = price_data.get('volume')
        if volume is None:
            return None
        
        vol_5d_max = volume.rolling(window=5, min_periods=5).max()
        vol_20d_mean = volume.rolling(window=20, min_periods=10).mean()
        
        # 极端放量强度 = max(5d) / mean(20d) - 1
        intensity = vol_5d_max / vol_20d_mean.replace(0, np.nan) - 1
        
        # 第3天反转窗口: shift(3) + 负号 (放量→高估→3天后回落→正向选反转)
        reversal = -intensity.shift(3)
        
        return reversal


# ============================================================
# 吴先兴五维成长因子框架 (2026-06-20)
# ============================================================

class EarningsQualityProxy(BaseFactor):
    """
    盈利质量代理 (吴先兴五维成长因子框架 之一)
    
    计算: -(ret_std_20 / (|ret_mean_20| + ε)) * 正收益占比_20
      - CV(变异系数) 的负值: 低CV = 收益更稳定 = 盈利质量高
      - 乘以正收益占比: 既稳定又正向 = 高质量
    方向: 正向 (盈利质量高 → Alpha)
    
    来源: 吴先兴 (2026-06-14) 五维成长因子框架 — 盈利质量
    """
    def __init__(self):
        super().__init__("earnings_quality_proxy", "growth_quality", "盈利质量代理")
    
    def compute(self, price_data, window=20, epsilon=0.001, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        ret = close.pct_change()
        
        ret_std = ret.rolling(window=window, min_periods=window).std()
        ret_mean_abs = ret.rolling(window=window, min_periods=window).mean().abs()
        win_rate = (ret > 0).astype(float).rolling(window=window, min_periods=window).mean()
        
        # CV = std / |mean| → -CV (低CV=高质量)
        neg_cv = -ret_std / (ret_mean_abs + epsilon)
        
        quality = neg_cv * win_rate
        
        return quality


class CashFlowMatchingProxy(BaseFactor):
    """
    现金流匹配度代理 (吴先兴五维成长因子框架 之二)
    
    计算: 过去20日中, 价格趋势方向与成交额趋势方向一致的占比
      - 价格趋势: close涨跌幅(20日)
      - 成交额趋势: 成交额(vol*close)涨跌幅(20日)
      - 两者同向 → 量价匹配 → 趋势更可靠
    方向: 正向 (量价匹配度高 → 趋势可靠)
    
    来源: 吴先兴 (2026-06-14) 五维成长因子框架 — 现金流匹配度
    """
    def __init__(self):
        super().__init__("cashflow_matching_proxy", "growth_quality", "现金流匹配度代理")
    
    def compute(self, price_data, window=20, **kwargs):
        close = price_data.get('close')
        volume = price_data.get('volume')
        if close is None or volume is None:
            return None
        
        # 价格趋势: 20日涨跌幅
        price_trend = close.pct_change(periods=window)
        
        # 成交额趋势: 20日成交额涨跌幅
        dollar_vol = volume * close
        amount_trend = dollar_vol.pct_change(periods=window)
        
        # 同向性: 价格趋势 × 成交额趋势 > 0
        same_direction = (price_trend * amount_trend > 0).astype(float)
        
        # 20日均值 = 同向占比
        matching = same_direction.rolling(window=window, min_periods=window).mean()
        
        return matching


class CapitalEfficiencyProxy(BaseFactor):
    """
    资本投入效率代理 (吴先兴五维成长因子框架 之三)
    
    计算: |20日收益| / 20日均成交额 * 1e8
      - 本质是 Amihud 非流动性的反面: 单位成交额产生的价格变动
      - 效率高 → 资金推动力强 → 高资金利用效率
    方向: 正向 (资本效率高 → Alpha)
    
    来源: 吴先兴 (2026-06-14) 五维成长因子框架 — 资本投入效率
    """
    def __init__(self):
        super().__init__("capital_efficiency_proxy", "growth_quality", "资本投入效率代理")
    
    def compute(self, price_data, window=20, scale=1e8, **kwargs):
        close = price_data.get('close')
        volume = price_data.get('volume')
        if close is None or volume is None:
            return None
        
        ret_abs = close.pct_change(periods=window).abs()
        dollar_vol = volume * close
        avg_dollar_vol = dollar_vol.rolling(window=window, min_periods=window).mean()
        
        # 单位成交额的价格变动 (Amihud反面)
        efficiency = ret_abs / avg_dollar_vol.replace(0, np.nan) * scale
        
        return efficiency


class OperationalEfficiencyProxy(BaseFactor):
    """
    营运效率代理 (吴先兴五维成长因子框架 之四)
    
    计算: -ret_1d_autocorr_20d
      - 日收益的1阶自相关: 自相关高 → 信息消化慢 → 营运效率低
      - 取负号: 自相关低 = 信息消化快 = 营运效率高
      - 使用向量化滚动协方差/方差公式加速
    方向: 正向 (自相关低 → 营运效率高 → 长期Alpha)
    
    来源: 吴先兴 (2026-06-14) 五维成长因子框架 — 营运效率
    """
    def __init__(self):
        super().__init__("operational_efficiency_proxy", "growth_quality", "营运效率代理")
    
    def compute(self, price_data, window=20, min_periods=10, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        ret = close.pct_change()
        ret_lag = ret.shift(1)
        
        # 滚动均值
        mean_ret = ret.rolling(window=window, min_periods=min_periods).mean()
        mean_lag = ret_lag.rolling(window=window, min_periods=min_periods).mean()
        
        # 滚动方差
        var_ret = ret.rolling(window=window, min_periods=min_periods).var()
        
        # 滚动协方差: E[ret * ret_lag] - E[ret] * E[ret_lag]
        cross_mean = (ret * ret_lag).rolling(window=window, min_periods=min_periods).mean()
        cov = cross_mean - mean_ret * mean_lag
        
        # 自相关 = cov / var
        autocorr = cov / var_ret.replace(0, np.nan)
        
        # 负号: 自相关低 → 效率高
        return -autocorr


class BargainingPowerProxy(BaseFactor):
    """
    议价能力代理 (吴先兴五维成长因子框架 之五)
    
    计算: -下行收益的20日均值 (仅取下跌日收益, 再取均值)
      - 下行收益越接近0 (跌幅小) → 议价能力强
      - 取负号: 跌幅小 → 因子值大 → 正向选股
    方向: 正向 (下行抗跌 → 议价能力强 → 质量高)
    
    来源: 吴先兴 (2026-06-14) 五维成长因子框架 — 议价能力
    """
    def __init__(self):
        super().__init__("bargaining_power_proxy", "growth_quality", "议价能力代理")
    
    def compute(self, price_data, window=20, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        ret = close.pct_change()
        
        # 只取下行收益 (ret < 0), 上行收益clip为0
        downside = ret.clip(upper=0)
        
        # 下行收益均值的负值: 跌幅越小 → 值越大(越接近0)
        # 负号翻转: 原值越负 → 乘-1后越正 = 议价能力越强
        power = -downside.rolling(window=window, min_periods=window).mean()
        
        return power


# ============================================================
# 日频因子实验 v2 通过 (2026-06-21)
# ============================================================

class VolumeStability(BaseFactor):
    """
    成交量稳定性 (日频因子实验 v2 通过, 2026-06-21)
    
    计算: -std(volume, 20d) / mean(volume, 20d)
      - 成交量变异系数(CV)的负值: 低CV = 成交量稳定 = 筹码锁定好
      - 与 dollar_vol_stability 区分: 该因子使用纯成交量(股数),
        不受价格波动影响，捕捉纯流动性供给稳定性
      - stable volume → lower information noise → higher trend persistence
    方向: 正向 (成交量稳定 → 信息噪音低 → 未来收益高)
    
    来源: daily_factor_hypothesis v2 PASS (ICIR=+0.989, +IC%=85.0%, n=147截面)
    日期: 2026-06-21
    """
    def __init__(self):
        super().__init__("volume_stability", "volume_structure", "成交量稳定性")
    
    def compute(self, price_data, window=20, min_periods=10, **kwargs):
        volume = price_data.get('volume')
        if volume is None:
            return None
        
        vol_mean = volume.rolling(window=window, min_periods=min_periods).mean()
        vol_std = volume.rolling(window=window, min_periods=min_periods).std()
        
        # CV = std / mean → -CV (低CV = 稳定 = 正向)
        stability = -vol_std / vol_mean.replace(0, np.nan)
        
        return stability


class VolumeClimaxReversal(BaseFactor):
    """
    天量反转 (日频因子实验 v2 通过, 2026-06-21 round 2)
    
    计算: -(volume_lag1 / volume_ma20_lag1 - 1).clip(lower=0)
      - 天量(>均量)次日缩量 = 买方力量耗尽 → 短期反转
      - 仅关注正向天量(>1x均量), 缩量部分自然衰减
      - 与 attention_decay 互补: 天量次日反转 vs 高峰后3日反转
    
    方向: 正向 (天量后缩量 → 买方枯竭 → 反转上行)
    
    来源: daily_factor_hypothesis v2 PASS (ICIR=+0.390, +IC%=64.4%, n=146截面)
    日期: 2026-06-21
    """
    def __init__(self):
        super().__init__("volume_climax_reversal", "volume_structure", "天量反转")
    
    def compute(self, price_data, window=20, min_periods=10, **kwargs):
        volume = price_data.get('volume')
        if volume is None:
            return None
        
        vol_ma = volume.rolling(window=window, min_periods=min_periods).mean()
        vol_ratio_lag = volume.shift(1) / vol_ma.shift(1) - 1.0
        
        # 仅当天量为正(>均量)时产生信号, 缩量为0信号
        climax = -vol_ratio_lag.clip(lower=0)
        
        return climax
