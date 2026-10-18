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


# ============================================================
# 日频因子实验 v2 破格注册 (2026-06-29)
# ============================================================

class OpeningGapMomentum(BaseFactor):
    """
    开盘跳空动量 (日频因子实验 2026-06-29, 破格注册)
    
    计算: (open / close.shift(1) - 1).rolling(3).mean()
      - 连续跳空高开的股票(强势开盘)短期动量延续
      - 与 GapUp (单期跳空) 互补: GapUp捕捉单期冲击, OpeningGapMomentum捕捉连续趋势
      - 3期平滑降低噪音: 连续多日跳空=持续信息流入→动量信号更可靠
    方向: 正向 (连续跳空高开 → 强势开盘动量)
    
    来源: 广发Level-2 ret_open2A系列因子(月频RankIC 3.11%)
          daily_factor_hypothesis v2 (ICIR=+0.294, +IC%=63.2%, n=163截面, 破格)
    日期: 2026-06-29
    """
    def __init__(self):
        super().__init__("opening_gap_momentum", "momentum", "开盘跳空动量")
    
    def compute(self, price_data, window=3, **kwargs):
        close = price_data.get('close')
        open_p = price_data.get('open')
        if close is None or open_p is None:
            return None
        
        overnight_ret = open_p / close.shift(1) - 1
        momentum = overnight_ret.rolling(window=window, min_periods=window).mean()
        return momentum


# ============================================================
# 日频因子实验 v2 通过 (2026-06-30)
# ============================================================

class MaxDrawdownDuration(BaseFactor):
    """
    最大回撤持续时间 (日频因子实验 2026-06-30, PASS)
    
    计算: (close / close.rolling(120).max()).rolling(20).mean()
      - 已从长期回撤中修复的股票(韧性信号)有选股Alpha
      - 回撤持续时间=市场压力测试: 从120日高点回撤后逐步修复=基本面强→Alpha
      - 值越高=越接近120日高点=韧性越强→正向选股
    方向: 正向 (恢复越接近高点 → 韧性越强)
    
    来源: 学术文献 extreme_events / daily_factor_hypothesis v2 (ICIR=+1.665, +IC%=93.3%, n=30截面)
    日期: 2026-06-30
    """
    def __init__(self):
        super().__init__("max_drawdown_duration", "extreme_events", "最大回撤持续时间")
    
    def compute(self, price_data, long_window=120, short_window=20, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        # 计算相对120日高点的位置
        drawdown_position = close / close.rolling(window=long_window, min_periods=long_window).max()
        # 20日均值平滑噪音
        recovery_score = drawdown_position.rolling(window=short_window, min_periods=short_window).mean()
        return recovery_score


class AccrualQualityProxy(BaseFactor):
    """
    应计质量代理 (日频因子实验 2026-07-01, PASS)
    
    计算: -(pct_change.std(20) / pct_change.abs().mean(20) + 0.001)
      - 收益变动系数(CV)的负值: 收益越稳定 = 应计质量越高
      - Sloan(1996)应计异象: 高应计=盈余管理→未来反转; 低应计=真实盈余→高质量
      - 价格趋势平滑=低应计噪音→财务质量好→正向Alpha
    方向: 正向 (低CV → 高应计质量 → 正向选股)
    
    来源: Sloan(1996) / daily_factor_hypothesis v2 (ICIR=+0.565, +IC%=70.9%, n=148截面)
    日期: 2026-07-01
    """
    def __init__(self):
        super().__init__("accrual_quality_proxy", "fundamental_quality_proxy", "应计质量代理")
    
    def compute(self, price_data, window=20, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None
        
        ret = close.pct_change()
        # 收益稳定性 = -CV(ret): 越低越稳定 → 越高质量
        ret_std = ret.rolling(window=window, min_periods=max(window//2, 5)).std()
        ret_abs_mean = ret.abs().rolling(window=window, min_periods=max(window//2, 5)).mean()
        quality_score = -(ret_std / (ret_abs_mean + 0.001))
        return quality_score


# ============================================================
# 日频因子实验 v2 积压 candidate 批量注册 (2026-07-03)
# ============================================================

class IntradayReversal(BaseFactor):
    """
    日内反转 (日频因子实验 v2, 2026-07-03 批量注册)

    计算: -(close / open - 1).rolling(5).mean()
      - 连续多日日内负收益(开盘高走低)的股票短期反弹
      - 日内收益反映散户情绪; 连续日内下跌=散户恐慌→均值回归

    方向: 正向 (日内连续下跌 → 均值回归反弹)

    来源: daily_factor_hypothesis v2 (ICIR=+0.315, +IC%=59.5%)
    日期: 2026年7月3日 批量注册
    """
    def __init__(self):
        super().__init__("intraday_reversal", "return_decomposition", "日内反转")

    def compute(self, price_data, window=5, **kwargs):
        close = price_data.get('close')
        open_p = price_data.get('open')
        if close is None or open_p is None:
            return None

        intraday_ret = close / open_p - 1.0
        reversal = -intraday_ret.rolling(window=window, min_periods=window).mean()
        return reversal


class RangeConsistency(BaseFactor):
    """
    波幅一致性 (日频因子实验 v2, 2026-07-03 批量注册)

    计算: -(high / low - 1).rolling(20).std()
      - 日内波幅稳定的股票有更持续的趋势(非暴涨暴跌)
      - 波幅一致性=价格行为有序: 波幅波动大→分歧大→预测力弱

    方向: 正向 (波幅稳定 → 价格行为有序 → 趋势可信)

    来源: daily_factor_hypothesis v2 (ICIR=+0.444, +IC%=62.8%)
    日期: 2026年7月3日 批量注册
    """
    def __init__(self):
        super().__init__("range_consistency", "price_pattern", "波幅一致性")

    def compute(self, price_data, window=20, **kwargs):
        high = price_data.get('high')
        low = price_data.get('low')
        if high is None or low is None:
            return None

        daily_range = high / low - 1.0
        consistency = -daily_range.rolling(window=window, min_periods=window).std()
        return consistency


class VolatilityOfVolatility(BaseFactor):
    """
    波动率的波动率 (日频因子实验 v2, 2026-07-03 批量注册)

    计算: -ret.std(5).rolling(20).std()
      - 波动率本身波动大的股票(不确定性大)未来收益低
      - Vol-of-Vol = 不确定性中的不确定性 → 价格信号的信噪比极低

    方向: 正向 (低Vol-of-Vol → 信号清晰 → 正向Alpha)

    来源: daily_factor_hypothesis v2 (ICIR=+0.431, +IC%=65.0%)
    日期: 2026年7月3日 批量注册
    """
    def __init__(self):
        super().__init__("volatility_of_volatility", "volatility_structure", "波动率的波动率")

    def compute(self, price_data, short_window=5, long_window=20, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None

        ret = close.pct_change()
        vol_5d = ret.rolling(window=short_window, min_periods=short_window).std()
        vol_of_vol = -vol_5d.rolling(window=long_window, min_periods=long_window).std()
        return vol_of_vol


class RelativeSpreadProxy(BaseFactor):
    """
    相对价差代理 (日频因子实验 v2, 2026-07-03 批量注册)

    计算: -(high - low) / (close + 0.001)
      - 日内相对价差小的股票流动性好、预期收益高
      - 日内价差=买卖价差代理: 价差小→交易成本低→流动性溢价为正

    方向: 正向 (小价差 → 低成本高流动性 → 正向Alpha)

    来源: daily_factor_hypothesis v2 (ICIR=+0.415, +IC%=65.3%)
    日期: 2026年7月3日 批量注册
    """
    def __init__(self):
        super().__init__("relative_spread_proxy", "liquidity_micro", "相对价差代理")

    def compute(self, price_data, **kwargs):
        high = price_data.get('high')
        low = price_data.get('low')
        close = price_data.get('close')
        if high is None or low is None or close is None:
            return None

        spread = -(high - low) / (close + 0.001)
        return spread


class TrendSmoothness(BaseFactor):
    """
    趋势平滑度 (日频因子实验 v2, 2026-07-03 批量注册)

    计算: close.rolling(20).apply(lambda x: corr(range(len(x)), x)^2)
      - 趋势平滑(R²高)的股票比趋势锯齿的股票动量更可靠
      - 高R²→线性趋势→可信; 低R²→锯齿→噪音

    方向: 正向 (高R² → 线性趋势 → 动量信号可靠)

    来源: daily_factor_hypothesis v2 (ICIR=+0.544, +IC%=70.3%)
    日期: 2026年7月3日 批量注册
    """
    def __init__(self):
        super().__init__("trend_smoothness", "trend_quality", "趋势平滑度")

    def compute(self, price_data, window=20, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None

        def r_squared(x):
            n = len(x)
            if n < 5:
                return 0.0
            r = np.corrcoef(np.arange(n), x)[0, 1]
            return r * r

        smoothness = close.rolling(window=window, min_periods=window).apply(
            r_squared, raw=True
        )
        return smoothness


class EarningsConsistencyProxy(BaseFactor):
    """
    盈利一致性代理 (日频因子实验 v2, 2026-07-03 批量注册)

    计算: close.pct_change(5).rolling(60).apply(lambda x: (x > 0).mean())
      - 正收益窗口占比高的股票(隐含盈利稳定)未来Alpha更高
      - Hsu et al.(2018): 持续正收益=隐藏的盈利增长 → 基本面质量信号

    方向: 正向 (高正收益占比 → 盈利稳定 → 正向Alpha)

    来源: Hsu et al.(2018) / daily_factor_hypothesis v2 (ICIR=+0.309, +IC%=63.8%)
    日期: 2026年7月3日 批量注册
    """
    def __init__(self):
        super().__init__("earnings_consistency_proxy", "fundamental_quality_proxy", "盈利一致性代理")

    def compute(self, price_data, ret_window=5, stat_window=60, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None

        ret_5d = close.pct_change(ret_window)
        consistency = ret_5d.rolling(window=stat_window, min_periods=stat_window).apply(
            lambda x: (x > 0).mean(), raw=True
        )
        return consistency


class RetOpen2DProxy(BaseFactor):
    """
    开盘动量2日代理 (日频因子实验 v2, 2026-07-03 批量注册)

    计算: (open / close.shift(1) - 1).rolling(2).mean()
      - 连续2日开盘跳空高开的股票(持续信息流入)短期动量强
      - 广发Level-2 ret_open2A: 开盘跳空=隔夜信息集中反映; 2日连续=信息持续

    方向: 正向 (连续跳空高开 → 持续信息流入 → 强势趋势)

    来源: 广发Level-2 / daily_factor_hypothesis v2 (ICIR=+0.306, +IC%=59.3%)
    日期: 2026年7月3日 批量注册
    """
    def __init__(self):
        super().__init__("ret_open_2d_proxy", "market_microstructure", "开盘动量2日代理")

    def compute(self, price_data, window=2, **kwargs):
        close = price_data.get('close')
        open_p = price_data.get('open')
        if close is None or open_p is None:
            return None

        overnight_ret = open_p / close.shift(1) - 1.0
        momentum = overnight_ret.rolling(window=window, min_periods=window).mean()
        return momentum


class EarningsSeasonVolDiv(BaseFactor):
    """
    中报窗口量价背离度 (Stage 2 通过, 2026-07-13)

    计算: 5日滚动窗口内 close变化 与 volume变化 的 Spearman秩相关系数取负
      - 价涨量缩(负相关)→正值(锁仓看多) → 未来收益高
      - 价跌量增(负相关)→正值 → 但本质是恐慌出逃

    方向: 正向 (量价背离 → 锁仓/惜售 → 正向Alpha)

    来源: Stage1→Stage2 管道 / daily_factor_hypothesis v2 (ICIR=+0.435, +IC%=66.5%)
    日期: 2026年7月13日
    """
    def __init__(self):
        super().__init__("earnings_season_vol_div", "earnings_season_divergence", "中报窗口量价背离度")

    def compute(self, price_data, window=5, **kwargs):
        close = price_data.get('close')
        volume = price_data.get('volume')
        if close is None or volume is None:
            return None

        pct_chg = close.pct_change().fillna(0)
        vol_chg = volume.pct_change().fillna(0)

        def rolling_spearman_neg(xy):
            """5日窗口中价变与量变的Spearman秩相关取负"""
            n = len(xy)
            if n < 5:
                return 0.0
            xp = xy[:n//2]
            yv = xy[n//2:]
            # Spearman rank correlation
            rx = xp.argsort().argsort().astype(float)
            ry = yv.argsort().argsort().astype(float)
            mr = rx.mean()
            my = ry.mean()
            num = ((rx - mr) * (ry - my)).sum()
            den = np.sqrt(((rx - mr)**2).sum() * ((ry - my)**2).sum())
            if den == 0 or np.isnan(den):
                return 0.0
            rho = num / den
            return -rho  # negate: price-vol divergence → positive signal

        # Stack price and volume changes for rolling apply
        combined = pd.DataFrame({
            'p': pct_chg.values.flatten(),
            'v': vol_chg.values.flatten()
        }, index=pd.MultiIndex.from_arrays([pct_chg.index.repeat(pct_chg.shape[1]),
                                              np.tile(pct_chg.columns, pct_chg.shape[0])]))

        # Simpler approach: compute per-column then rebuild
        result_df = pd.DataFrame(index=close.index, columns=close.columns, data=np.nan)
        for col in close.columns:
            p = pct_chg[col].values
            v = vol_chg[col].values
            res = np.full(len(p), np.nan)
            for i in range(window-1, len(p)):
                x = p[i-window+1:i+1]
                y = v[i-window+1:i+1]
                rx = x.argsort().argsort().astype(float)
                ry = y.argsort().argsort().astype(float)
                mr = rx.mean()
                my = ry.mean()
                num = ((rx - mr) * (ry - my)).sum()
                den = np.sqrt(((rx - mr)**2).sum() * ((ry - my)**2).sum())
                if den > 0 and not np.isnan(den):
                    res[i] = -num / den
                else:
                    res[i] = 0.0
            result_df[col] = res

        # Smooth with 5-day rolling mean
        result = result_df.rolling(window=5, min_periods=1).mean()
        return result


class PostEarningsStability(BaseFactor):
    """
    业绩后稳定性 (Stage 2 通过, 2026-07-17)

    计算: 5日波动率(取负) × 量能MA5/MA20对数比
      - 低波动+量能回升 = 业绩预告后机构有序建仓信号
      - 对数变换避免量比正偏分布

    方向: 正向 (低波动+量回升 → 正向Alpha)

    来源: Stage1→Stage2 管道 / daily_factor_hypothesis v2 (ICIR=+0.378, +IC%=58.6%)
    日期: 2026年7月17日
    """
    def __init__(self):
        super().__init__("post_earnings_stability", "earnings_anomaly", "业绩后稳定性")

    def compute(self, price_data, **kwargs):
        close = price_data.get('close')
        volume = price_data.get('volume')
        if close is None or volume is None:
            return None

        ret = close.pct_change()
        # 5日波动率 (取负: 低波动=高信号)
        vol_5d = ret.rolling(5).std()
        # 量比: MA5/MA20, 对数化
        ma5 = volume.rolling(5).mean()
        ma20 = volume.rolling(20).mean()
        vol_ratio = np.log(ma5.replace(0, np.nan) / ma20.replace(0, np.nan) + 1e-6)
        # 合成: 负波动 × 正量比
        raw = (-vol_5d.clip(upper=0.15)) * vol_ratio.clip(lower=-1, upper=2)
        # z-score 标准化
        median = raw.median()
        std = raw.std().clip(lower=1e-6)
        result = (raw - median) / std
        return result


class EarningsVolumeDrift(BaseFactor):
    """
    业绩窗量价漂移 (Stage 2 通过, 2026-07-17)

    计算: 5日窗口中正收益且缩量的股票 = 缩量上涨=筹码锁定 → 正向信号
      - ret_5d > 0 且 vol_chg_5d < 0 → 最强信号
      - 仅保留上涨信号(缩量下跌不纳入)

    方向: 正向 (缩量上涨 → 筹码锁定 → 正向Alpha)

    来源: Stage1→Stage2 管道 / daily_factor_hypothesis v2 (ICIR=+0.448, +IC%=68.3%)
    日期: 2026年7月17日
    """
    def __init__(self):
        super().__init__("earnings_volume_drift", "earnings_anomaly", "业绩窗量价漂移")

    def compute(self, price_data, **kwargs):
        close = price_data.get('close')
        volume = price_data.get('volume')
        if close is None or volume is None:
            return None

        ret_5d = close / close.shift(5) - 1
        ma5 = volume.rolling(5).mean()
        ma5_shifted = volume.shift(5).rolling(5).mean()
        vol_chg_5d = ma5 / ma5_shifted.replace(0, np.nan) - 1
        # 缩量上涨: 收益>0 and 量增<0 → 正向
        signal = ret_5d.clip(lower=-0.3, upper=0.3) * (-vol_chg_5d).clip(lower=-1, upper=2)
        # 仅保留上涨信号
        signal = signal.where(ret_5d > 0, 0)
        # z-score 标准化
        median = signal.median()
        std = signal.std().clip(lower=1e-6)
        result = (signal - median) / std
        return result


# ============================================================
# Stage 2 通过 (2026-07-31)
# ============================================================

class HarveySiddiqueCoskew(BaseFactor):
    """
    Harvey-Siddique 市场协偏度 (Stage 2 通过, 2026-07-31)

    计算: Harvey-Siddique(2000, JF)协偏度 — 个股收益与等权市场收益平方的
      标准化协方差(60日滚窗)。
      - 负协偏度 = 与市场下行共偏 → crash risk premium → 正向补偿
      - 正协偏度 = 彩票特征 → 未来收益低

    方向: 正向 (做多负协偏度 = crash risk premium)

    来源: Harvey & Siddique (2000, JF)
          Stage2 daily_factor_hypothesis v2 (ICIR=+0.317, +IC%=58.9%, n=112截面)
    日期: 2026年7月31日
    """
    def __init__(self):
        super().__init__("harvey_siddique_coskew", "market_microstructure", "Harvey-Siddique市场协偏度")

    def compute(self, price_data, lookback=60, smooth=5, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None

        import pandas as pd
        import numpy as np

        ret = close.pct_change().fillna(0.0)
        mkt_ret_raw = ret.mean(axis=1)
        mkt_ret_sq = (mkt_ret_raw ** 2).values.reshape(-1, 1)

        # Denominator: std(ret_i) * mean(mkt_ret^2)
        ri_std = ret.rolling(lookback).std()

        mkt_idx = pd.Series(mkt_ret_sq.flatten(), index=ret.index)
        rm_sq_mean = mkt_idx.rolling(lookback).mean()
        rm_sq_mean_df = pd.DataFrame(
            np.tile(rm_sq_mean.values.reshape(-1, 1), (1, ret.shape[1])),
            index=ret.index, columns=ret.columns
        )

        # Numerator: mean(ret_i * mkt_ret^2)
        cross_prod = ret.mul(pd.Series(mkt_ret_sq.flatten(), index=ret.index), axis=0)
        num = cross_prod.rolling(lookback).mean()

        # Co-skewness
        denom = (ri_std * rm_sq_mean_df).replace(0, np.nan)
        coskew = num / denom

        # Negative coskew = crash risk premium -> long
        factor = -coskew.rolling(smooth).mean()
        return factor


# ============================================================
# Stage 2 通过 (2026-09-16) — 干旱破冰批次
# ============================================================

class MarketStateRiskAdjustedMomentum(BaseFactor):
    """
    市场状态风险调整动量 (Stage 2 通过, 2026-09-16)

    计算: 趋势偏离 × 动量波动比
      - 项A = (m20 − m20_60mean) / m20_20std_60mean   → 中期动量相对长期均值的偏离,
              以"波动比的长期均值"标准化 = 趋势强度的风险调整值
      - 项B = m20 / m20_20std                        → 当期风险调整动量 (Sharpe 式)
      其中 m20 = close.pct_change(20).rolling(20).mean()

    方向: 正向 (高值 = 强势且低波动的市场状态 → 动量延续)

    来源: daily_factor_hypothesis Stage2 v2 (2026-09-16 日报)
          实测 RankIC +0.0774, ICIR 0.620, +IC% 76.3%, RRE60d 0.998, n=76 截面
          源提案 P-20260821-005; 方向实证与预测一致 (long, 未证伪)
    日期: 2026年9月16日
    """
    def __init__(self):
        super().__init__("market_state_risk_adjusted_momentum",
                         "market_state_strength", "市场状态风险调整动量")

    def compute(self, price_data, fast=20, slow=20, long_win=60, **kwargs):
        close = price_data.get('close')
        if close is None:
            return None

        # 中期动量: 20日收益的20日均值
        #   注意: 严格复刻实验原式 —— 项A/项B 的分母都是 pct_change(20) 的滚动标准差
        #   (sd20), 不是 m20 的标准差; 且一律使用 pandas 默认 min_periods (全窗口)。
        #   写成 m20.rolling(20).std() 会延长预热期并改变数值 (实测 ICIR 0.866 vs 0.620)。
        pct20 = close.pct_change(fast)
        m20 = pct20.rolling(slow).mean()
        sd20 = pct20.rolling(slow).std()

        # 项A: 趋势偏离 / 波动比的长期均值
        dev = m20 - m20.rolling(long_win).mean()
        term_a = dev / (sd20.rolling(long_win).mean() + 1e-8)

        # 项B: 当期风险调整动量
        term_b = m20 / (sd20 + 1e-8)

        return term_a * term_b


class CrowdedReversal5DLargeOrderFlow(BaseFactor):
    """
    拥挤反转·大单资金流 (Stage 2 通过, 2026-09-16)

    计算: 在高拥挤截面 (双 rank 乘积 >0.7) 上取 5 日收益的负值, 其余置 0
      - 拥挤度 = rank_pct(amount / amount_20d_sum) × rank_pct(大单净买占比)
        大单净买占比 = (buy_lg_vol − sell_lg_vol) / (buy_lg_vol + sell_lg_vol)
      - 触发条件: 成交额占比与主动大单失衡同时处于高位 (=机构拥挤交易)
      - 信号 = −ret_5d (仅触发样本), 非触发样本为 0 (=中性不参与)

    方向: 正向 (拥挤且已上涨 → 信号负 → 短期反转; 拥挤且已下跌 → 信号正 → 修复)
          注: 原始预测方向为 short, Stage2 实测 IC 为正 → direction_falsified=True,
          按"方向本地实证翻转铁律"以实证符号为准, 故此处直接注册原式 (不取反)。

    数据依赖: price_data['amount'] + price_data['buy_lg_vol'] + price_data['sell_lg_vol']
      ⚠⚠ 口径与数据限制 (2026-09-16 注册实测量化 + 当晚补采后复评, 必读):
        1. 缺失订单流一律 NaN (不 fillna(0)) —— 原实验口径 fillna(0) 会把"整日无
           订单流"的截面伪造成"按成交额排序的反转信号": lg_imbalance=0 ⇒ lg_rank≈0.5
           常量 ⇒ 门控退化为"按成交额份额排序的反转"。
        2. moneyflow 覆盖缺陷已于 2026-09-16 补采修复 (P-20260916-003, v0.7.25):
           原 2025-08-08~2026-02-05 的 122 个交易日每日仅 2~4 行残片, 现 795,221 →
           1,397,102 行 / 269 个交易日 0 缺口; 缺口率 46.01% → 5.46%
           (残留 90.6% 是北交所 —— Tushare moneyflow 对 .BJ 覆盖仅 3.8%, 属能力边界)。
        3. 🔴 补采后复评 (Stage2 官方 harness, n_days=400, 2025-08-12~2026-09-15):
           诚信 NaN 口径 FULL ICIR 0.370 (IC +0.0278 / +IC% 62.6% / n=243) ⇒ PASS。
           此前"未达门槛"的结论系残缺样本(仅 155 截面 + 46% 填 0)造成的低估, 已推翻。
           分段: NEW(2025-08-12~2026-02-05, n=102) 0.628 | PREV(2026-02-06~, n=142) 0.249
           ⇒ 引用 FULL 必须同时声明分段 (NEW 段是强反转友好期: 纯反转 NEW 0.394 / PREV 0.029)。
        4. 🔴🔴 归因警告 (2026-09-16 通道剥离消融, 见 P-20260916-004): 本因子的
           **大单资金流通道为统计显著净拖累** —— 移除后 FULL ICIR 0.370 → 0.440
           (+IC% 62.6% → 68.3%); 逐截面配对检验 t=−5.72 / p<1e-4, NEW/PREV 三段均显著为负。
           大单通道单独门控 (0.230) 相对纯反转 (0.124) 不显著 (p=0.56) ⇒ 无增量信息。
           有效信息**全部来自成交额份额拥挤度通道** ⇒ 因子名 large_order_flow 名实不符,
           与 v0.7.9 margin_volatility_squeeze 同型 (第二次复现)。重写方案见 P-20260916-004。

    来源: daily_factor_hypothesis Stage2 v2 (2026-09-16 日报) + 补采后复评 (2026-09-16 晚)
          原始实测 RankIC +0.0276, ICIR 0.335, +IC% 60.0%, RRE60d 0.744, n=155 截面
          补采后诚信口径 ICIR 0.370 / +IC% 62.6% / n=243; 通道剥离消融 0.440
          源提案 P-20260823-002 → 复评提案 P-20260916-004
    日期: 2026年9月16日 (注册) / 2026年9月16日晚 (补采复评)
    """
    def __init__(self):
        super().__init__("crowded_reversal_5d_large_order_flow",
                         "conditional_reversal", "拥挤反转·大单资金流")

    def compute(self, price_data, ret_win=5, amount_win=20,
                crowd_threshold=0.7, **kwargs):
        close = price_data.get('close')
        amount = price_data.get('amount')
        buy_lg = price_data.get('buy_lg_vol')
        sell_lg = price_data.get('sell_lg_vol')
        if close is None or amount is None or buy_lg is None or sell_lg is None:
            return None

        # 成交额占比分位 (axis=1: 沿截面排名; 默认 axis=0 会引入前视)
        #   注意: 与实验原式一致, 使用 pandas 默认 min_periods (全窗口)
        amount_share = amount / amount.rolling(amount_win).sum()
        amount_rank = amount_share.rank(pct=True, axis=1)

        # 主动大单净买占比分位
        #   ⚠ 口径声明: 不填 0。缺失订单流 ⇒ 该股该日不参与拥挤度排名 (NaN),
        #     而非伪造"零失衡"。原实验的 load_moneyflow_wide 对缺失 fillna(0),
        #     会把"整日无订单流"的截面伪造成"按成交额排序的反转信号"。
        #     实测 moneyflow_daily.csv 在 2025-08~2025-12 每日仅 3 行(≈空),
        #     2026-02 起才接近满覆盖 ⇒ 填 0 会污染约 1/3 的样本。
        lg_imbalance = (buy_lg - sell_lg) / (buy_lg + sell_lg + 1e-8)
        lg_rank = lg_imbalance.rank(pct=True, axis=1)

        # 拥挤度 = 两个独立维度的 rank 乘积 → 再取分位
        crowd = (amount_rank * lg_rank).rank(pct=True, axis=1)
        mask = crowd > crowd_threshold

        ret5 = close.pct_change(ret_win)
        # 非触发样本置 0 (中性), 触发样本取负收益
        signal = -ret5.where(mask, 0.0)

        # 逐截面守卫: 该截面无任何触发样本 (典型=订单流整段缺失期) → 整行置 NaN。
        #   否则全 0 常量行会作为"有效信号"注入下游截面回归, 制造隐藏暴露。
        no_trigger = mask.sum(axis=1) == 0
        if bool(no_trigger.all()):
            return None
        return signal.mask(no_trigger)


class CrowdedAmountReversal5D(BaseFactor):
    """
    拥挤反转·成交额份额 (P-20260916-004 落地, 2026-09-16)

    计算: 在成交额份额拥挤端 (截面分位 > 0.9) 上取 5 日收益的负值, 其余置 0
      - 拥挤度 = rank_pct(amount / amount_20d_sum)     ← 单通道, 只用成交额
      - 触发条件: 成交额份额处于截面最拥挤的 10%
      - 信号 = −ret_5d (仅触发样本), 非触发样本为 0 (=中性不参与)

    方向: 正向 (拥挤且已上涨 → 信号负 → 短期反转; 拥挤且已下跌 → 信号正 → 修复)

    ── 为什么重写 (P-20260916-004, 通道级消融实证) ──────────────────────────
    原式 `crowded_reversal_5d_large_order_flow` 用了「成交额份额分位 × 大单净买分位」
    的联合分位, 但消融证明**大单通道是统计显著净拖累**:
      · 移除大单通道后 FULL ICIR 0.370 → 0.440 (+IC% 62.6% → 68.3%)
      · 逐截面配对 t 检验 (V3原式 − V1成交额单通道): FULL ΔIC=−0.01218 t=−5.72 p<1e-4;
        NEW t=−5.89; PREV t=−2.60 p=0.0102 —— 三段全显著为负, 非噪声
      · 大单通道单独门控 (0.230) 相对纯反转 (0.124) 不显著 (p=0.562) ⇒ 零增量信息
      · 提高阈值再比较: V1(0.9) − V3(0.7原式) ΔIC=+0.01894 t=+6.50 p=4.5e-10 极显著
      ⇒ 原式因子名 `large_order_flow` 名实不符, 与 v0.7.9 `margin_volatility_squeeze`
        同型 (第二次复现); 本因子即"名实相符"的重写版。

    ── 阈值 0.9 的样本外依据 (不是 argmax, 是显著改进+保守选择) ──────────────
    阈值扫描 (FULL / NEW / PREV 三段, ICIR):
      thr   FULL    NEW    PREV          thr   FULL    NEW    PREV
      0.5   0.330  0.567  0.207          0.9   0.693  1.021  0.508
      0.6   0.369  0.631  0.233          0.92  0.754  1.087  0.569
      0.7   0.440  0.739  0.285          0.95  0.828  1.115  0.666
      0.8   0.525  0.864  0.344          0.98  0.866  1.063  0.748
    ⇒ **单调性在三段全部成立** (非窗口内单点拟合)。样本外四折确认: 面板按时间切
      Q1/Q2/Q3, 4 个 fold (早选后验 ×2 + 反向后验 ×2) 的 IS 最优阈值稳定落在 0.9/0.95,
      OOS 一致击败 thr=0.7 (+0.236 / +0.405 / +0.279 / +0.362)。
    ⇒ 取 0.9 而非更高的 0.95/0.98 的理由:
      (a) 配对检验 V1(0.95) − V1(0.9): ΔIC=−0.00109 t=−0.75 **p=0.455 不显著**
          ⇒ 更高阈值无统计显著增量;
      (b) 触发稀疏化: 日均触发股票数 thr=0.7 为 1502 只 / 0.9 为 **502** 只 / 0.95 为 252 只
          ⇒ 0.95 会让可选股池腰斩而买不到显著好处; 0.9 仍留 502 只, 组合 TOP-N 空间充足。
      (c) 0.96~0.98 的 ICIR 0.855→0.866 属平台区 (0.97 还微降到 0.852), 而 IC 绝对水平
          在 0.95 后开始衰减 (+0.0468 @0.9 → +0.0348 @0.98) ⇒ 极端阈值是"波动被压小"
          而非"信号更强", 端点不可取。

    ── 数据依赖 ─────────────────────────────────────────────────────────────
    price_data['close'] + price_data['amount']   ← 仅此两项, 均来自 daily_prices.csv
      ✅ 与旧式的关键差别: **完全不读 moneyflow** (buy_lg_vol/sell_lg_vol)。
         收益: (1) 对 `FUXI_MF_MISSING` 缺失口径开关**免疫** —— 无论 zero/nan 都不受影响;
               (2) 摆脱 Tushare moneyflow 对北交所覆盖仅 3.8% 的数据源能力边界;
               (3) 无 2025-08~2025-12 的历史空洞风险 (P-20260916-003 补采的区间)。
      ⚠️ 逐截面守卫仍保留: 无触发样本的截面整行 NaN, 防"全 0 常量行"注入下游回归。

    口径: 截面 rank 必须 axis=1 (v0.9.4 铁律; 默认 axis=0 会引入前视)。
          **禁用 fillna(0)** 伪造缺失 (本因子其实已无缺失来源, 守卫为防御性保留)。

    来源: P-20260916-004 (cross_verified=true) ← 归因消融 P-20260916-003 复评
    实证归档: data/crowded_attribution_20260916.json (10 变体消融)
              data/crowded_attrib_test_20260916.json (配对 t + 触发集合 Jaccard)
              data/p004_oos_threshold_20260916.json (三段阈值扫描 + 4 折样本外)
              data/p004_threshold_diag_20260916.json (端点延伸 + 稀疏化 + 配对检验)
    日期: 2026年9月16日
    """
    def __init__(self):
        super().__init__("crowded_amount_reversal_5d",
                         "conditional_reversal", "拥挤反转·成交额份额")

    def compute(self, price_data, ret_win=5, amount_win=20,
                crowd_threshold=0.9, **kwargs):
        close = price_data.get('close')
        amount = price_data.get('amount')
        if close is None or amount is None:
            return None

        # 成交额占比分位 (axis=1: 沿截面排名; 默认 axis=0 会引入前视)
        #   注意: 与实验原式一致, 使用 pandas 默认 min_periods (全窗口)
        amount_share = amount / amount.rolling(amount_win).sum()
        amount_rank = amount_share.rank(pct=True, axis=1)

        # 单通道拥挤度门控 (P-004: 已剥离大单通道 —— 实测为净拖累)
        mask = amount_rank > crowd_threshold

        ret5 = close.pct_change(ret_win)
        # 非触发样本置 0 (中性), 触发样本取负收益
        signal = -ret5.where(mask, 0.0)

        # 逐截面守卫: 该截面无任何触发样本 → 整行置 NaN。
        #   否则全 0 常量行会作为"有效信号"注入下游截面回归, 制造隐藏暴露。
        no_trigger = mask.sum(axis=1) == 0
        if bool(no_trigger.all()):
            return None
        return signal.mask(no_trigger)

# ============================================================
# Stage 2 通过 (2026-10-09) — GitHub 公开快照复现批次 (源: E:\fuxi v0.7.2)
# 注册路径: E:\fuxi Stage2 (daily_factor_hypothesis) → 生产 FA 注册
# 保真口径: 逐值复刻实验原式 + harness rank 语义修正 (裸 rank → axis=1);
#          无数据区间逐截面守卫 (整行 NaN) — 数据覆盖区间内 diff=0 (见注册归档)
# ============================================================

class PostListExternalFlowDays(BaseFactor):
    """
    榜外资金持续流入天数 (Stage 2 通过, 2026-10-09)

    计算: 近 10 日 "大单资金净流入" 天数 → 截面分位
      - 逐日判定 (buy_lg_vol − sell_lg_vol) > 0 记 1 天 (榜外大单净买入)
      - 10 日滚动求和 = 持续天数 ∈ [0,10] (pandas 默认 min_periods=10 全窗口)
      - 截面 rank(pct=True, axis=1) → [0,1] 分位 (高 = 建仓耐心/计划性强)

    方向: long (持续天数越高 → 后续上涨越稳健, 暗仓资金有计划吸筹)

    数据依赖: price_data['buy_lg_vol'] + price_data['sell_lg_vol']
      (data/loader.py 订单流通道; 源 data/raw/moneyflow_daily.csv)
      ⚠ 覆盖限制与逐截面守卫: moneyflow 覆盖自 2025-08-08 起 (实测 279 日、
        每日 5,101~5,572 只, 零稀疏日; Tushare 对北交所覆盖仅 3.8% 属能力边界)。
        该日之前无数据: 原式 (net>0) 会把整段 NaN 压成 "恒定 0.5 排行" (=伪信号),
        直接回测会用 4.6 年假截面稀释真实 IC ⇒ 已加逐截面守卫
        (有效股票 <30 只 → 整行 NaN; 30 = harness 自身截面样本下限)。
        守卫在数据覆盖区间内恒不触发 ⇒ 保真 diff 在全部真实截面上 = 0。
      ⚠ 缺失口径: 原式 (net > 0) 对 NaN 与 fillna(0) 后的 0 同为 False
        ⇒ 本因子对 FUXI_MF_MISSING 口径开关 (zero/nan) 逐值免疫
        (注册保真校验已实测两口径输出逐值一致)。
      ⚠ 语义注意: 无数据个股 (如 .BJ) 被计为 "零净流入天", 与原实验一致;
        该退化系原式固有, 不作修补 (保真优先)。

    来源: daily_factor_hypothesis Stage2 (2026-10-09, GitHub 快照 v0.7.2 复现批)
          实测 RankIC +0.0253, ICIR +0.375, +IC% 63.8%, RRE60d 0.967, n=163 截面
          源提案 P-20260831-003「榜后差额暗仓资金」→ stage1_factor_proposals.json status=PASS
    日期: 2026年10月9日
    """
    def __init__(self):
        super().__init__("post_list_external_flow_days",
                         "moneyflow_persistence", "榜外资金持续流入天数")

    def compute(self, price_data, win=10, **kwargs):
        buy_lg = price_data.get('buy_lg_vol')
        sell_lg = price_data.get('sell_lg_vol')
        if buy_lg is None or sell_lg is None:
            return None

        net = buy_lg - sell_lg
        pos = (net > 0)                      # NaN→False, 与原式一致 (缺失口径免疫)
        days = pos.rolling(win).sum()        # 默认 min_periods=win (全窗口), 与原式一致
        out = days.rank(pct=True, axis=1)    # 截面 rank 必须 axis=1 (v0.9.4 铁律)

        # 逐截面守卫 (数据覆盖): 该截面订单流有效股票 <30 只 → 整行 NaN。
        #   根因: 2025-08-08 前无 moneyflow; 原式对整段 NaN 产出恒定 0.5 行 (伪信号),
        #   会以 4.6 年假截面稀释 IC。30 = 实验 harness 自身的每截面样本下限。
        valid = net.notna().sum(axis=1) >= 30
        if not bool(valid.any()):
            return None
        return out.mask(~valid)


class IdealAmplitudeWCutReversal20D(BaseFactor):
    """
    理想振幅 W 式切割 20 日反转 (Stage 2 通过, 2026-10-09)

    计算: 20 日收益分位 × 振幅分位 → 乘积再取截面分位后取负
      - r20_rank = rank_pct(close.pct_change(20))                     (截面)
      - amp_rank = rank_pct(((high−low)/(close+1e-8)).rolling(20).mean())  (截面)
      - 信号 = − rank_pct( r20_rank × amp_rank )
      语义: 高振幅期间反转信号更强 (过度交易 → 价格修正), 取负 = 反转方向。

    方向: short (高 "涨幅×振幅" 分位 → 信号低 → 预期回落)

    数据依赖: price_data['close'] + price_data['high'] + price_data['low']
      (daily_prices.csv; 全部个股全覆盖, 无订单流依赖 ⇒ 缺失口径开关免疫)
      ⚠ 口径差异声明 (实验 harness vs FA 生产, 必读):
        实验用**不复权** close/high/low (daily_prices 原始列);
        FA loader 输出**后复权** (×adj_factor)。本因子含 pct_change(20),
        除权除息日附近两口径存在差异 (后复权平滑分红跳变) ⇒ ICIR 读数
        不可与实验逐位对齐, 系固有口径差而非实现错误
        (保真校验 = FA 口径内 原式 vs 实现 max|diff|, 见注册归档)。
      ⚠ 使用 pandas 默认 min_periods (全窗口), 不额外加 min_periods
        (加半窗会延长/缩短预热期并稀释 ICIR — skill 记载的已知坑)。

    来源: daily_factor_hypothesis Stage2 (2026-10-09, GitHub 快照 v0.7.2 复现批)
          实测 RankIC +0.0569, ICIR +0.3152, +IC% 58.55%, RRE60d 1.000, n=152 截面
          源提案 P-20260901-003「理想振幅 W 式」→ stage1_factor_proposals.json status=PASS
    日期: 2026年10月9日
    """
    def __init__(self):
        super().__init__("ideal_amplitude_w_cut_reversal_20d",
                         "amplitude_reversal", "理想振幅W式切割20日反转")

    def compute(self, price_data, ret_win=20, amp_win=20, **kwargs):
        close = price_data.get('close')
        high = price_data.get('high')
        low = price_data.get('low')
        if close is None or high is None or low is None:
            return None

        # 20 日收益分位 (axis=1: 沿截面排名; 默认 axis=0 会引入前视)
        r20_rank = close.pct_change(ret_win).rank(pct=True, axis=1)
        # 振幅分位: (high − low)/close 的 20 日均值 → 截面分位
        amp = ((high - low) / (close + 1e-8)).rolling(amp_win).mean()
        amp_rank = amp.rank(pct=True, axis=1)
        # 乘积 → 截面分位 → 取负 (反转方向)
        return -(r20_rank * amp_rank).rank(pct=True, axis=1)
