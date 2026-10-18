"""
市场 Regime 检测器 — 基于可观测指标、零前视

MVP (风格价差+趋势): 4 base regimes
V2  (加波动率):       8 detail regimes

用法:
    detector = RegimeDetector()
    detector.load_index_data(data_dir)
    labels = detector.label_history('2021-01-01', '2026-06-30')
    regime_today = detector.detect(today_str, index_prices_today)

设计原则:
  - 所有指标基于已发生的过去4周数据 (零前视)
  - 阈值用规则门控 (不调参, 避免过拟合)
  - 每个 regime 保证 >= 50 个周样本 (足够 GA 训练)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List


# ── Regime 编码 ──────────────────────────────────────────────
# 实证: A股历史上 small_bear(小盘强势+大盘跌)几乎不存在
# 实际活跃的 base regime:
#   0: small_bull  — 小盘强势+整体上涨 (流动性/量价因子主场)
#   1: large_dominant — 大盘强势 (质量/价值/低波因子主场)
#   2: neutral — 风格中性, 极稀少 (~6%), 合并到邻近regime

BASE_REGIME = {
    0: "small_bull",       # 小盘强势 + 大盘涨
    1: "large_dominant",   # 大盘强势 (不分涨跌)
}

# V2: + 波动率维度 → 4-6 regimes (每regime ≥ 50周)
REGIME_LABELS = {
    "small_bull":      "小盘强势牛 (流动性因子主场, high_quality主战场)",
    "large_dominant":  "大盘强势 (质量/价值因子主场)",
    "neutral":         "风格中性 (极稀少, 合并至邻近regime)",
}


class RegimeDetector:
    """市场 Regime 检测器"""

    def __init__(self, lookback_weeks: int = 4, spread_threshold: float = 0.02):
        """
        lookback_weeks: 回顾窗口 (周), 默认4周
        spread_threshold: 风格价差阈值, CSI500-CSI300 收益差 > 2% 为小盘强势
        """
        self.lookback = lookback_weeks
        self.spread_threshold = spread_threshold
        self.index_weekly: Optional[pd.DataFrame] = None
        self.labels_: Optional[pd.Series] = None
        self.indicators_: Optional[pd.DataFrame] = None

    # ── 数据加载 ──────────────────────────────────────────
    def load_index_data(self, data_dir) -> "RegimeDetector":
        """
        从 tushare 格式的 index_*.csv 加载三大指数日线, 转为周频
        data_dir: Path or str, 如 'data/raw'
        """
        data_dir = Path(data_dir)
        indices = {
            'CSI300':   data_dir / 'index_000300_daily.csv',
            'CSI500':   data_dir / 'index_000905_daily.csv',
            'CSI_ALL':  data_dir / 'index_000985_daily.csv',
        }

        weekly_close = {}
        for name, path in indices.items():
            if not path.exists():
                raise FileNotFoundError(f"缺少指数数据: {path}")
            df = pd.read_csv(path, encoding='utf-8-sig')
            # 日期列: 兼容 int YYYYMMDD 和字符串格式
            if 'trade_date' not in df.columns:
                raise ValueError(f"{path.name}: 缺少 trade_date 列")
            date_col = df['trade_date']
            if date_col.dtype in ('int64', 'int32', 'float64', 'float32'):
                date_col = pd.to_datetime(
                    date_col.fillna(0).astype(int).astype(str), format='%Y%m%d',
                    errors='coerce'
                )
            else:
                date_col = pd.to_datetime(date_col, errors='coerce')
            df['trade_date'] = date_col
            df = df.dropna(subset=['trade_date']).sort_values('trade_date')
            df = df.set_index('trade_date')
            # 周频: 取每周五收盘
            w = df['close'].resample('W-FRI').last().dropna()
            w.name = name
            weekly_close[name] = w

        self.index_weekly = pd.DataFrame(weekly_close).dropna()
        self.index_weekly.index.name = 'date'
        print(f"  指数周线: {len(self.index_weekly)} 周 "
              f"({self.index_weekly.index[0].strftime('%Y-%m-%d')} → "
              f"{self.index_weekly.index[-1].strftime('%Y-%m-%d')})")
        return self

    def load_from_dataframe(self, df: pd.DataFrame) -> "RegimeDetector":
        """
        直接从已构造好的周频 DataFrame 加载
        df: index=date, columns=['CSI300','CSI500','CSI_ALL']
        """
        self.index_weekly = df.copy()
        self.index_weekly.index.name = 'date'
        return self

    # ── 指标计算 ──────────────────────────────────────────
    def compute_indicators(self) -> pd.DataFrame:
        """
        计算 regime 判定所需的三个指标:
          - style_spread: CSI500 4周收益 - CSI300 4周收益
          - trend:        CSI_ALL 4周收益
          - volatility:   CSI_ALL 4周日频收益 std (V2)
        零前视: 每个时间点的指标仅使用该点之前的数据
        """
        if self.index_weekly is None:
            raise RuntimeError("请先 load_index_data()")

        w = self.index_weekly
        pct = w.pct_change().dropna()  # 周收益率

        # 4周累计收益 (用 rolling sum of log-return 近似)
        ret_4w = (1 + pct).rolling(self.lookback, min_periods=3).apply(
            lambda x: np.prod(1 + x) - 1, raw=True
        )  # type: ignore

        style_spread = ret_4w['CSI500'] - ret_4w['CSI300']
        trend = ret_4w['CSI_ALL']

        # V2: 波动率 (CSI_ALL 4周收益率的 std)
        volatility = pct['CSI_ALL'].rolling(self.lookback, min_periods=3).std()

        self.indicators_ = pd.DataFrame({
            'style_spread': style_spread,
            'trend': trend,
            'volatility': volatility,
        }).dropna()
        return self.indicators_

    # ── Regime 分类 ───────────────────────────────────────
    def _classify_base(self, row: pd.Series) -> str:
        """单行 → base regime (仅2个主regime + neutral兜底)"""
        spread = row['style_spread']
        trend = row['trend']

        if spread > self.spread_threshold:
            # 小盘强势: A股历史上趋势几乎都>0, 但防御性保留small_bear
            return "small_bull" if trend > -0.01 else "large_dominant"
        elif spread < -self.spread_threshold:
            return "large_dominant"
        else:
            return "neutral"

    def _classify_detail(self, row: pd.Series) -> str:
        """V2: base regime + vol suffix"""
        base = self._classify_base(row)
        vol_median = self.indicators_['volatility'].median()
        vol_tag = "_highvol" if row['volatility'] > vol_median else "_lowvol"
        return base + vol_tag

    # ── 对外接口 ──────────────────────────────────────────
    def label_history(self, start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      detail: bool = False) -> pd.Series:
        """
        对全历史标注 regime (返回 Series, index=date, values=regime label)
        detail=False → MVP 4 base regimes
        detail=True  → V2 8 regimes (base × vol)
        """
        ind = self.compute_indicators()

        if start_date:
            ind = ind[ind.index >= pd.Timestamp(start_date)]
        if end_date:
            ind = ind[ind.index <= pd.Timestamp(end_date)]

        classifier = self._classify_detail if detail else self._classify_base
        self.labels_ = ind.apply(classifier, axis=1)
        self.labels_.name = 'regime'
        return self.labels_

    def detect(self, index_prices: Dict[str, float]) -> str:
        """
        实盘周度检测: 给定三大指数本周收盘价, 返回当前 regime

        index_prices: {'CSI300': 本周收盘, 'CSI500': ..., 'CSI_ALL': ...}
        内部维护最近4周价格序列, 计算指标后分类
        首次调用需先调 load_index_data() 以初始化序列

        返回: regime label string
        """
        if self.index_weekly is None:
            raise RuntimeError("请先 load_index_data() 以初始化价格序列")

        # 追加本周数据
        today = pd.Timestamp.now().normalize()
        new_row = pd.DataFrame(
            {k: [v] for k, v in index_prices.items()},
            index=[today]
        )
        w = pd.concat([self.index_weekly, new_row])
        # 只保留最近 (lookback+5) 周, 避免无限增长
        w = w.iloc[-(self.lookback + 5):]

        # 临时计算指标
        pct = w.pct_change().dropna()
        ret_4w = (1 + pct).rolling(self.lookback, min_periods=3).apply(
            lambda x: np.prod(1 + x) - 1, raw=True
        )
        spread = ret_4w['CSI500'].iloc[-1] - ret_4w['CSI300'].iloc[-1]
        trend = ret_4w['CSI_ALL'].iloc[-1]

        # 分类
        row = pd.Series({'style_spread': spread, 'trend': trend})
        return self._classify_base(row)

    def stats(self) -> pd.DataFrame:
        """regime 分布统计"""
        if self.labels_ is None:
            self.label_history()

        counts = self.labels_.value_counts()
        pct = (counts / counts.sum() * 100).round(1)

        return pd.DataFrame({
            '周数': counts,
            '占比%': pct,
            '说明': [REGIME_LABELS.get(k, '') for k in counts.index],
        }).rename_axis('Regime')

    def labels_merged(self) -> pd.Series:
        """
        将 neutral (< 5% 出现率) 合并到邻近 regime,
        返回可直接用于 GA 训练切分的标签
        """
        if self.labels_ is None:
            self.label_history()

        labels = self.labels_.copy()
        # neutral → forward-fill 到最近的非neutral regime
        mask = labels == 'neutral'
        labels[mask] = None
        labels = labels.ffill().bfill()
        return labels


def get_regime_splits(labels: pd.Series) -> Dict[str, pd.DatetimeIndex]:
    """
    返回每个 regime 的时间索引集合, 供 GA 训练切分使用

    return: {'small_bull': DatetimeIndex([...]), ...}
    """
    return {regime: labels[labels == regime].index for regime in labels.unique()}


# ── 快捷函数 ──────────────────────────────────────────────
def load_and_label(data_dir, start='2021-01-01', end='2026-06-30',
                   detail: bool = False) -> Tuple[RegimeDetector, pd.Series]:
    """一步加载+标注"""
    detector = RegimeDetector()
    detector.load_index_data(data_dir)
    labels = detector.label_history(start, end, detail=detail)
    return detector, labels
