"""
HMM 市场 Regime 检测器
=====================
用 Hidden Markov Model 自动发现市场隐含状态,
替代基于规则门控的 RegimeDetector (style_spread + trend)。

v1 (2026-07-21): GaussianHMM, 3-5 states, BIC自动选K

用法:
    detector = HMMRegimeDetector()
    detector.load_index_data(data_dir)
    labels, model = detector.fit_predict(n_states=None)  # None=auto BIC
    detector.save_labels('data/regime_labels_hmm.csv')
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings("ignore")


class HMMRegimeDetector:
    """HMM 市场状态检测器"""

    def __init__(self, lookback_weeks: int = 4,
                 volatility_lookback: int = 20):
        """
        lookback_weeks: 收益率回顾窗口 (周)
        volatility_lookback: 波动率回顾窗口 (周)
        """
        self.lookback = lookback_weeks
        self.vol_lookback = volatility_lookback
        self.features_: Optional[pd.DataFrame] = None
        self.labels_: Optional[np.ndarray] = None
        self.model_ = None
        self.state_names_: Dict[int, str] = {}
        self.weekly_dates_: Optional[pd.DatetimeIndex] = None

    # ── 数据加载 ──────────────────────────────────────
    def load_index_data(self, data_dir) -> "HMMRegimeDetector":
        """
        从 tushare 格式的 index_*.csv 加载三大指数日线, 提取周频特征
        """
        data_dir = Path(data_dir)
        indices = {
            'CSI300':  data_dir / 'index_000300_daily.csv',
            'CSI500':  data_dir / 'index_000905_daily.csv',
            'CSI_ALL': data_dir / 'index_000985_daily.csv',
        }
        missing = [n for n, p in indices.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"缺少指数数据: {missing}")

        weekly_returns = {}
        for name, path in indices.items():
            df = pd.read_csv(path, encoding='utf-8-sig')

            # 日期处理
            date_col = df['trade_date']
            if date_col.dtype in ('int64', 'float64'):
                df['dt'] = pd.to_datetime(date_col.astype(int).astype(str),
                                          format='%Y%m%d')
            else:
                df['dt'] = pd.to_datetime(date_col.astype(str), format='mixed')

            df = df.set_index('dt').sort_index()

            # 复权后周收益
            close = df['close'].astype(float)
            w_close = close.resample('W').last()
            w_ret = w_close.pct_change().dropna()
            weekly_returns[name] = w_ret

        # 对齐所有指数
        common_idx = weekly_returns['CSI300'].index
        for name in ['CSI500', 'CSI_ALL']:
            common_idx = common_idx.intersection(weekly_returns[name].index)
        common_idx = common_idx.sort_values()

        self.weekly_dates_ = common_idx

        # 构建特征矩阵
        features = pd.DataFrame(index=common_idx)
        features['csi300_ret'] = weekly_returns['CSI300'].loc[common_idx]
        features['csi500_ret'] = weekly_returns['CSI500'].loc[common_idx]
        features['csi_all_ret'] = weekly_returns['CSI_ALL'].loc[common_idx]

        # 风格价差
        features['style_spread'] = (features['csi500_ret'] -
                                     features['csi300_ret'])

        # 滚动波动率 (年化)
        for col in ['csi300_ret', 'csi500_ret', 'csi_all_ret']:
            features[f'{col}_vol'] = (features[col]
                                      .rolling(self.vol_lookback).std()
                                      * np.sqrt(52))

        # 趋势强度 (4周累计收益)
        for col in ['csi300_ret', 'csi500_ret']:
            features[f'{col}_trend'] = (features[col]
                                        .rolling(self.lookback).mean() * self.lookback)

        features = features.dropna()
        self.features_ = features
        return self

    # ── HMM 训练 ──────────────────────────────────────
    def fit_predict(self, n_states: Optional[int] = None,
                    feature_cols: Optional[List[str]] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    ) -> Tuple[pd.Series, object]:
        """
        Args:
            n_states: 隐含状态数, None=自动BIC选择(3-5)
            feature_cols: 使用的特征列, 默认全部
            start_date/end_date: 训练窗口, None=全部

        Returns:
            labels: Series(index=date, value=state_id)
            model: 训练好的 GaussianHMM
        """
        from hmmlearn import hmm

        if self.features_ is None:
            raise RuntimeError("请先调用 load_index_data()")

        df = self.features_
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        self.filtered_features_ = df  # 保存过滤后的特征用于标签对齐

        if feature_cols is None:
            # 默认特征: 排除趋势(与ret近共线), 保留ret/vol/spread
            exclude = [c for c in df.columns if '_trend' in c]
            feature_cols = [c for c in df.columns if c not in exclude]

        X = df[feature_cols].values
        # 标准化特征
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_norm = (X - X_mean) / X_std

        if n_states is None:
            n_states = self._auto_select_k(X_norm)

        # 训练 GaussianHMM
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=500,
            tol=1e-4,
            random_state=42,
        )
        model.fit(X_norm)

        # 预测
        states = model.predict(X_norm)

        labels = pd.Series(states, index=df.index, name='hmm_state')
        self.labels_ = states
        self.model_ = model
        self.feature_cols_ = feature_cols

        # 命名 states
        self._name_states(df, labels, feature_cols)

        return labels, model

    # ── 自动选 K ─────────────────────────────────────
    def _auto_select_k(self, X: np.ndarray, k_range=(3, 5)) -> int:
        """BIC 自动选择最优状态数"""
        from hmmlearn import hmm

        best_k, best_bic = k_range[0], np.inf
        for k in range(k_range[0], k_range[1] + 1):
            m = hmm.GaussianHMM(
                n_components=k, covariance_type='diag',
                n_iter=300, tol=1e-4, random_state=42,
            )
            m.fit(X)
            bic = m.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_k = k

        return best_k

    # ── 状态命名 ─────────────────────────────────────
    def _name_states(self, df, labels, feature_cols):
        """根据各状态统计特征命名状态"""
        state_names = {}
        for s in sorted(labels.unique()):
            mask = labels == s
            sub = df[mask]

            ret_300 = sub['csi300_ret'].mean()
            ret_500 = sub['csi500_ret'].mean()
            spread = sub['style_spread'].mean()

            # 简单规则命名
            if ret_300 > 0 and ret_500 > 0:
                if spread > 0.005:
                    name = "small_bull"      # 小盘强势牛
                else:
                    name = "broad_bull"       # 普涨牛
            elif ret_300 > 0 and ret_500 < 0:
                name = "large_dominant"      # 大盘强势
            elif ret_300 < 0 and ret_500 < 0:
                if spread > 0:
                    name = "small_resilient" # 小盘抗跌
                else:
                    name = "broad_bear"       # 普跌熊
            else:
                name = f"regime_{s}"

            n_weeks = mask.sum()
            state_names[s] = name
            print(f"  State {s} ({name}): {n_weeks} 周, "
                  f"CSI300_ret={ret_300:.4f}, CSI500_ret={ret_500:.4f}, "
                  f"spread={spread:.4f}")

        self.state_names_ = state_names

    # ── 保存 / 加载 ──────────────────────────────────
    def save_labels(self, path: str):
        """保存 regime labels 到 CSV"""
        idx = self.filtered_features_.index if hasattr(self, 'filtered_features_') else self.features_.index
        labels_s = pd.Series(self.labels_, index=idx, name='hmm_state')
        labels_s.to_csv(path, encoding='utf-8-sig')
        print(f"[HMM] 保存 {len(labels_s)} 周标签到 {path}")

    def labels_merged(self, min_samples: int = 10) -> pd.Series:
        """合并稀有状态 (少于 min_samples 周的合并到最近状态)"""
        idx = self.filtered_features_.index if hasattr(self, 'filtered_features_') else self.features_.index
        labels_s = pd.Series(self.labels_, index=idx, name='hmm_state')
        counts = labels_s.value_counts()
        rare = counts[counts < min_samples].index.tolist()

        if not rare:
            return labels_s

        # ffill 合并稀有状态
        result = labels_s.replace({r: np.nan for r in rare})
        result = result.ffill().bfill().astype(int)
        print(f"[HMM] 合并 {len(rare)} 个稀有状态 "
              f"({', '.join(str(r) for r in rare)}) → {result.nunique()} states")
        return result

    # ── 诊断 ──────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        """各状态统计摘要"""
        if self.labels_ is None or self.features_ is None:
            return pd.DataFrame()

        df = self.features_.copy()
        df['state'] = self.labels_
        return df.groupby('state').agg(['mean', 'std', 'count']).round(4)


# ── 便捷函数 ──────────────────────────────────────────
def detect_regimes_hmm(data_dir, n_states=None,
                       start_date='2021-01-01', end_date='2026-06-30',
                       output_path=None) -> Tuple[pd.Series, HMMRegimeDetector]:
    """一键 HMM regime 检测"""
    detector = HMMRegimeDetector()
    detector.load_index_data(data_dir)
    labels, model = detector.fit_predict(
        n_states=n_states, start_date=start_date, end_date=end_date
    )
    labels = detector.labels_merged()
    if output_path:
        labels.to_csv(output_path, encoding='utf-8-sig')
        print(f"[HMM] 标签已保存: {output_path}")
    return labels, detector
