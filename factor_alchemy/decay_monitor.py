# -*- coding: utf-8 -*-
"""
Decay Monitor — S6 因子衰减动态监控 Stage (v0.5 P-004)
===========================================================

监控因子有效性的时间维度衰减，对标:
  - WorldQuant Brain 过拟合诊断: IC稳定性/暴露偏度/伪因子
  - McLean & Pontiff (2016): 因子发表后衰减 58%
  - FinLab 期间敏感性实证

功能:
  1. IC 滚动稳定性检测 (std_rolling_IC > 0.015 → 预警)
  2. IC 单调性检测 (IC_t+1 > IC_t+2 > IC_t+3 → phantom factor)
  3. 周度全库扫描 + 衰减趋势追踪
  4. 与 library_orthogonality 的 Red Sea 面板联动

设计原则:
  - 独立模块，不接入 S1-S5 主线
  - S5 评估时缓存 IC 序列，DecayMonitor 离线扫描
  - 在 JQ 验证前 1-2 周发现衰减信号

用法:
    from decay_monitor import DecayMonitor
    
    dm = DecayMonitor()
    dm.update_ic_cache("factor_name", ic_series)
    report = dm.scan_all()
    print(f"预警: {report['warnings']}, Phantom: {report['phantoms']}")
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# ── 默认路径 ──────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent.parent / "data"
IC_CACHE_PATH = DATA_DIR / "cache" / "ic_history.json"
DECAY_LOG_PATH = DATA_DIR / "decay_monitor_log.json"


@dataclass
class DecayAlert:
    """衰减告警"""
    factor_name: str
    alert_type: str          # "IC_UNSTABLE" / "PHANTOM" / "DECLINING"
    severity: str            # "warn" / "danger"
    current_ic: float = 0.0
    ic_stability: float = 0.0   # 滚动 IC 标准差
    ic_trend: str = "flat"      # "improving" / "flat" / "declining"
    monotonicity_violated: bool = False
    detail: str = ""
    detected_at: str = ""

    def to_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "current_ic": round(self.current_ic, 6),
            "ic_stability": round(self.ic_stability, 6),
            "ic_trend": self.ic_trend,
            "monotonicity_violated": self.monotonicity_violated,
            "detail": self.detail,
            "detected_at": self.detected_at,
        }


class DecayMonitor:
    """因子衰减动态监控器 — S6 Stage
    
    独立于 S1-S5 主线，可离线/周度运行。
    """

    def __init__(
        self,
        ic_cache_path: Path = IC_CACHE_PATH,
        log_path: Path = DECAY_LOG_PATH,
        stability_threshold: float = 0.015,   # WorldQuant: IC std > 0.015 预警
        monotonicity_window: int = 4,          # 检查最近 4 期 IC
        min_history_days: int = 60,            # 至少 60 天 IC 数据才检测
    ):
        self.ic_cache_path = ic_cache_path
        self.log_path = log_path
        self.stability_threshold = stability_threshold
        self.monotonicity_window = monotonicity_window
        self.min_history_days = min_history_days

        self.ic_cache: Dict[str, List[float]] = self._load_ic_cache()
        self.alerts: List[DecayAlert] = []
        self.scan_history: List[Dict] = self._load_log()

    # ── 存储 ──────────────────────────────────────────────

    def _load_ic_cache(self) -> Dict[str, List[float]]:
        if self.ic_cache_path.exists():
            with open(self.ic_cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_ic_cache(self):
        self.ic_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ic_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.ic_cache, f, ensure_ascii=False, indent=2)

    def _load_log(self) -> List[Dict]:
        if self.log_path.exists():
            with open(self.log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("scans", [])
        return []

    def _save_log(self, scan_result: Dict):
        self.scan_history.append(scan_result)
        # 只保留最近 100 次扫描
        if len(self.scan_history) > 100:
            self.scan_history = self.scan_history[-50:]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "version": "1.0.0",
                "scans": self.scan_history,
                "updated_at": datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)

    # ── IC 缓存更新 ──────────────────────────────────────

    def update_ic_cache(self, factor_name: str, ic_series):
        """更新单个因子的 IC 序列缓存（在 S5 评估后调用）
        
        Parameters
        ----------
        factor_name: 因子名
        ic_series: IC 序列 (list/ndarray/Series)
        """
        if hasattr(ic_series, 'tolist'):
            ic_series = ic_series.tolist()
        elif isinstance(ic_series, (pd.Series,)):
            ic_series = ic_series.tolist()

        # 过滤 NaN
        ic_series = [float(x) for x in ic_series if not (isinstance(x, float) and np.isnan(x))]

        if not ic_series:
            return

        self.ic_cache[factor_name] = ic_series
        self._save_ic_cache()

    def batch_update_from_results(self, results: List):
        """从 S5 验证结果批量更新 IC 缓存
        
        Parameters
        ----------
        results: List[ValidationResult] 或 List[Dict]（需包含 factor_name + s1_ic 序列信息）
        """
        for r in results:
            name = getattr(r, 'factor_name', r.get('factor_name', '')) if not hasattr(r, 'keys') else r.get('factor_name', '')
            if not name:
                continue

            # 尝试从 result 获取 IC 序列
            ic_series = None
            if hasattr(r, 's1_ic'):
                # 单值 IC，缓存为单点（至少有一个数据点）
                ic_series = [r.s1_ic]
            elif isinstance(r, dict):
                ic_series = r.get('ic_series', r.get('ic_history', None))

            if ic_series:
                existing = self.ic_cache.get(name, [])
                existing.extend(ic_series if isinstance(ic_series, list) else [ic_series])
                # 去重尾数限制
                if len(existing) > 500:
                    existing = existing[-500:]
                self.ic_cache[name] = existing

        self._save_ic_cache()

    # ── 衰减检测 ─────────────────────────────────────────

    def check_ic_stability(self, factor_name: str, ic_series: List[float]) -> Optional[DecayAlert]:
        """IC 滚动稳定性检测 — WorldQuant Brain 标准
        
        std(rolling_60d_IC) > 0.015 → 衰减预警
        """
        if len(ic_series) < self.min_history_days:
            return None

        series = pd.Series(ic_series)
        # 滚动均值
        rolling_mean = series.rolling(min(60, len(series)), min_periods=20).mean()
        # 滚动标准差
        rolling_std = series.rolling(min(60, len(series)), min_periods=20).std()

        current_std = rolling_std.iloc[-1]
        current_ic = rolling_mean.iloc[-1] if not pd.isna(rolling_mean.iloc[-1]) else series.iloc[-1]

        if pd.isna(current_std):
            return None

        if current_std > self.stability_threshold:
            # 计算趋势
            if len(rolling_mean) >= 20:
                first_half = rolling_mean.iloc[-20:-10].mean()
                second_half = rolling_mean.iloc[-10:].mean()
                if second_half < first_half * 0.8:
                    trend = "declining"
                elif second_half > first_half * 1.2:
                    trend = "improving"
                else:
                    trend = "flat"
            else:
                trend = "flat"

            return DecayAlert(
                factor_name=factor_name,
                alert_type="IC_UNSTABLE",
                severity="warn" if current_std < 0.03 else "danger",
                current_ic=current_ic,
                ic_stability=current_std,
                ic_trend=trend,
                detail=f"IC滚动标准差={current_std:.4f} > {self.stability_threshold}, 趋势={trend}",
                detected_at=datetime.now().isoformat(),
            )

        return None

    def check_monotonicity(self, factor_name: str, ic_series: List[float]) -> Optional[DecayAlert]:
        """IC 单调性检测 — Phantom Factor 预警
        
        WorldQuant: 83% 的坍缩因子违反 IC(t+1) > IC(t+2) > IC(t+3)
        """
        if len(ic_series) < self.monotonicity_window + 1:
            return None

        # 取最近 N 期的月度均值（或直接用日度 IC 的 N 日窗口均值）
        series = pd.Series(ic_series)

        if len(series) >= 60:
            # 按月聚合
            monthly_ic = series.groupby(series.index // 20).mean()  # ~20 交易日/月
            recent = monthly_ic.tail(self.monotonicity_window + 1)
            if len(recent) >= 4:
                diffs = recent.diff().dropna()
                # 所有差分 < 0 → 严格递减
                if (diffs < 0).all():
                    return DecayAlert(
                        factor_name=factor_name,
                        alert_type="PHANTOM",
                        severity="danger",
                        current_ic=series.iloc[-1],
                        monotonicity_violated=True,
                        detail=f"IC连续{len(diffs)}期递减 → 疑似phantom factor",
                        detected_at=datetime.now().isoformat(),
                    )
        else:
            # 日度检测：最近 N 天的滚动均值递减
            if len(series) >= 20:
                rolling_20d = series.rolling(20).mean().dropna()
                if len(rolling_20d) >= self.monotonicity_window + 1:
                    recent = rolling_20d.tail(self.monotonicity_window + 1)
                    diffs = recent.diff().dropna()
                    if (diffs < 0).all():
                        return DecayAlert(
                            factor_name=factor_name,
                            alert_type="PHANTOM",
                            severity="warn",
                            current_ic=series.iloc[-1],
                            monotonicity_violated=True,
                            detail=f"20日均IC连续{len(diffs)}期递减 → 衰减中",
                            detected_at=datetime.now().isoformat(),
                        )

        return None

    def check_decay_trend(self, factor_name: str, ic_series: List[float]) -> Optional[DecayAlert]:
        """衰减趋势综合检测
        
        结合稳定性和单调性，输出综合判断。
        """
        if len(ic_series) < self.min_history_days:
            return None

        stability_alert = self.check_ic_stability(factor_name, ic_series)
        monotonicity_alert = self.check_monotonicity(factor_name, ic_series)

        # 如果两个都触发 → danger
        if stability_alert and monotonicity_alert:
            stability_alert.alert_type = "DECLINING"
            stability_alert.severity = "danger"
            stability_alert.monotonicity_violated = True
            stability_alert.detail += f"; {monotonicity_alert.detail}"
            return stability_alert

        # 只要有一个触发就返回
        return stability_alert or monotonicity_alert

    # ── 全库扫描 ─────────────────────────────────────────

    def scan_all(self, verbose: bool = True) -> Dict:
        """周度全库扫描 — 检测所有因子的衰减状态
        
        Returns
        -------
        {
            "warnings": int,        # 预警因子数
            "phantoms": int,        # phantom factor 数
            "alerts": List[DecayAlert],
            "scanned_total": int,
            "scan_timestamp": str,
        }
        """
        self.alerts = []
        warnings = 0
        phantoms = 0

        for factor_name, ic_series in self.ic_cache.items():
            alert = self.check_decay_trend(factor_name, ic_series)
            if alert:
                self.alerts.append(alert)
                if alert.alert_type in ("PHANTOM", "DECLINING"):
                    phantoms += 1
                else:
                    warnings += 1

        scan_result = {
            "warnings": warnings,
            "phantoms": phantoms,
            "alerts": [a.to_dict() for a in self.alerts],
            "scanned_total": len(self.ic_cache),
            "scan_timestamp": datetime.now().isoformat(),
        }

        self._save_log(scan_result)

        if verbose:
            print(f"\n[S6 DecayMonitor] 全库扫描完成:")
            print(f"  扫描因子数: {len(self.ic_cache)}")
            print(f"  预警: {warnings} | Phantom: {phantoms}")
            for alert in self.alerts:
                icon = "🔴" if alert.severity == "danger" else "🟡"
                print(f"  {icon} [{alert.alert_type}] {alert.factor_name}: {alert.detail[:80]}")

        return scan_result

    def get_factor_decay_status(self, factor_name: str) -> Dict:
        """获取单个因子的衰减状态
        
        Returns
        -------
        {"status": "OK"|"WARN"|"DANGER", "alert": DecayAlert|None, "ic_series_len": int}
        """
        ic_series = self.ic_cache.get(factor_name, [])
        if not ic_series or len(ic_series) < self.min_history_days:
            return {"status": "NODATA", "alert": None, "ic_series_len": len(ic_series)}

        alert = self.check_decay_trend(factor_name, ic_series)
        if alert:
            return {
                "status": "DANGER" if alert.severity == "danger" else "WARN",
                "alert": alert.to_dict(),
                "ic_series_len": len(ic_series),
            }

        return {"status": "OK", "alert": None, "ic_series_len": len(ic_series)}

    def get_decay_rate(self, factor_name: str) -> float:
        """计算因子衰减率（最近期 IC / 全期 IC）- 用于 Red Sea 面板"""
        ic_series = self.ic_cache.get(factor_name, [])
        if len(ic_series) < 40:
            return 1.0  # 数据不足，假定未衰减

        series = pd.Series(ic_series)
        recent_ic = abs(series.tail(20).mean())
        full_ic = abs(series.mean())

        if full_ic < 1e-8:
            return 1.0

        decay_rate = recent_ic / full_ic
        return max(0.0, min(1.0, decay_rate))  # clamp [0, 1]

    def get_summary_for_log(self) -> str:
        """生成用于日志的报告摘要"""
        alerts = self.alerts
        if not alerts:
            return "[S6] 无衰减预警 — 所有因子 IC 稳定"

        danger = [a for a in alerts if a.severity == "danger"]
        warn = [a for a in alerts if a.severity == "warn"]

        lines = [f"[S6] 衰减扫描: {len(danger)} danger + {len(warn)} warn"]
        for a in danger:
            lines.append(f"  🔴 {a.factor_name}: {a.detail[:60]}")
        for a in warn[:3]:
            lines.append(f"  🟡 {a.factor_name}: {a.detail[:60]}")

        return "\n".join(lines)

    # ── P-018: Red Sea 拥挤度 → 衰减映射 ─────────────────

    def compute_crowding_from_red_sea(self) -> Dict[str, float]:
        """
        从 library_orthogonality 的 Red Sea 相关性矩阵中
        提取每个因子的拥挤度得分，作为衰减预测的输入。

        Returns
        -------
        {factor_name: crowding_score}
        """
        try:
            from library_orthogonality import LibraryOrthogonalityManager
            # 尝试加载已有管理器（共享状态）
            mgr = LibraryOrthogonalityManager()
            scores = mgr.get_all_crowding_scores()
            return scores
        except Exception as e:
            print(f"[DecayMonitor] ⚠️ Red Sea拥挤度获取失败: {e}")
            return {}

    def get_crowding_decay_risk(self, factor_name: str) -> Dict:
        """
        综合因子衰减率 + 拥挤度，输出衰减风险评分。
        risk = 0.6 × decay_rate + 0.4 × crowding_score
        """
        result = {
            "factor_name": factor_name,
            "decay_rate": None,
            "crowding_score": None,
            "risk_score": None,
            "risk_level": "unknown",
        }

        # 衰减率
        decay_rate = self.get_decay_rate(factor_name)
        if decay_rate is not None:
            result["decay_rate"] = decay_rate

        # 拥挤度
        crowding = self.compute_crowding_from_red_sea()
        if factor_name in crowding:
            result["crowding_score"] = crowding[factor_name]

        # 综合风险
        if result["decay_rate"] is not None and result["crowding_score"] is not None:
            result["risk_score"] = 0.6 * (1 - result["decay_rate"]) + 0.4 * result["crowding_score"]
            if result["risk_score"] > 0.5:
                result["risk_level"] = "high"
            elif result["risk_score"] > 0.3:
                result["risk_level"] = "medium"
            else:
                result["risk_level"] = "low"

        return result

    # ── P-022: DecayMonitor v2 — Hyperbolic模型 + 行业衰减 ───

    def fit_hyperbolic_decay(
        self,
        ic_series: List[float],
        time_window: int = None,
    ) -> Dict:
        """
        对 IC 序列拟合双曲线衰减模型 α(t) = K / (1 + λt)

        通过线性化转换 1/α = 1/K + (λ/K)·t 用 OLS 估计 K 和 λ。
        同时拟合线性模型做 R² 对比，区分结构性衰减 vs 线性漂移。

        Parameters
        ----------
        ic_series : 原始 IC 序列（按时间顺序，t=0 为最早）
        time_window : 用最近 N 个周期做滚动拟合。
                      默认 auto: max(20, min(len, 60)) — 至少20, 最多60

        Returns
        -------
        {
            "K": float,            # 初始 IC 水平
            "lambda": float,       # 衰减速率 (λ>0=衰减, λ≈0=平稳, λ<0=改善)
            "half_life": float,    # 半衰期 (时间单位与输入序列一致)
            "r2_hyperbolic": float,  # 双曲线拟合 R²
            "r2_linear": float,     # 线性拟合 R²
            "decay_mode": str,      # "structural"|"linear"|"stable"|"improving"
            "mean_ic": float,       # 平均 IC
            "recent_ic": float,     # 最近期 IC
            "sufficient_data": bool,
        }
        """
        # Auto window: hyperbolic decay is a long-term phenomenon — use most data.
        # Default: 80% of series (min 20, no upper cap) for structural pattern detection.
        if time_window is None:
            series_len = len(ic_series)
            time_window = max(20, int(series_len * 0.8))

        result = {
            "K": 0.0, "lambda": 0.0, "half_life": float('inf'),
            "r2_hyperbolic": 0.0, "r2_linear": 0.0,
            "decay_mode": "stable", "mean_ic": 0.0, "recent_ic": 0.0,
            "sufficient_data": False,
        }

        if len(ic_series) < max(time_window, 10):
            return result

        # 取最近 time_window 个点，取绝对值以便拟合
        series = np.array(ic_series[-time_window:], dtype=np.float64)
        ic_abs = np.abs(series)
        result["mean_ic"] = float(np.mean(ic_abs))
        result["recent_ic"] = float(np.mean(ic_abs[-5:])) if len(ic_abs) >= 5 else float(ic_abs[-1])
        result["sufficient_data"] = True

        t = np.arange(len(ic_abs), dtype=np.float64)
        ic_pos = np.maximum(ic_abs, 1e-8)  # 避免除零

        # ── Hyperbolic fit: 1/α = 1/K + (λ/K)·t ──
        y_inv = 1.0 / ic_pos
        # OLS: y_inv = a + b·t
        A = np.column_stack([np.ones_like(t), t])
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, y_inv, rcond=None)
            a, b = coeffs[0], coeffs[1]  # a=1/K, b=λ/K
        except np.linalg.LinAlgError:
            return result

        if a <= 0:
            return result

        K = 1.0 / a
        lam = b / a  # λ = (λ/K) / (1/K)

        # 计算 R²
        y_pred_hyper = K / (1.0 + lam * t)
        ss_res = np.sum((ic_abs - y_pred_hyper) ** 2)
        ss_tot = np.sum((ic_abs - np.mean(ic_abs)) ** 2)
        r2_hyper = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

        result["K"] = round(K, 6)
        result["lambda"] = round(lam, 6)
        result["r2_hyperbolic"] = round(r2_hyper, 4)

        # ── Half-life: 当 α(t) = K/2 → 1/(1+λt) = 1/2 → t = 1/λ ──
        if lam > 1e-8:
            result["half_life"] = round(1.0 / lam, 2)
        else:
            result["half_life"] = float('inf')

        # ── Linear fit for comparison ──
        try:
            coeffs_lin, _, _, _ = np.linalg.lstsq(A, ic_abs, rcond=None)
            y_pred_lin = coeffs_lin[0] + coeffs_lin[1] * t
            ss_res_lin = np.sum((ic_abs - y_pred_lin) ** 2)
            r2_lin = 1.0 - ss_res_lin / ss_tot if ss_tot > 1e-10 else 0.0
            result["r2_linear"] = round(r2_lin, 4)
        except np.linalg.LinAlgError:
            result["r2_linear"] = 0.0

        # ── Decay mode classification ──
        r2_h = result["r2_hyperbolic"]
        r2_l = result["r2_linear"]

        if lam < -1e-6 and r2_h > 0.3:
            result["decay_mode"] = "improving"
        elif lam > 1e-6:
            # Structural decay: hyperbolic fit captures the decay shape.
            # Multi-tier: at high R² even small advantage matters; at low R² need bigger gap.
            r2_advantage = r2_h - r2_l
            if r2_advantage > 0.02 and r2_h > 0.8 and lam > 0.003:
                result["decay_mode"] = "structural"
            elif r2_advantage > 0.05 and r2_h > 0.5 and lam > 0.005:
                result["decay_mode"] = "structural"
            elif r2_advantage > 0.10 and r2_h > 0.25:
                result["decay_mode"] = "structural"
            elif r2_l > 0.2:
                result["decay_mode"] = "linear"
            else:
                result["decay_mode"] = "linear" if lam > 0.005 else "stable"
        else:
            result["decay_mode"] = "stable"

        return result

    def paradigm_decay_aggregate(
        self,
        paradigm_factor_map: Dict[str, List[str]] = None,
    ) -> Dict:
        """
        按范式汇总所有因子的衰减状态，检测范式级系统性衰减。

        Parameters
        ----------
        paradigm_factor_map : {paradigm_name: [factor_name, ...]}
                              如不传，从 library_orthogonality 获取

        Returns
        -------
        {
            "per_paradigm": {
                paradigm_name: {
                    "n_factors": int,
                    "n_structural": int,     # structural decay
                    "n_linear": int,         # linear decay
                    "n_declining": int,      # any decay mode
                    "pct_declining": float,
                    "avg_half_life": float,  # 平均半衰期 (inf→排除)
                    "alert_level": "red"|"yellow"|"green",
                    "alert_msg": str,
                }
            },
            "systemic_alerts": [{paradigm, level, msg}],
        }
        """
        # 获取范式-因子映射
        if paradigm_factor_map is None:
            paradigm_factor_map = self._get_paradigm_factor_map()

        per_paradigm = {}
        systemic_alerts = []

        for paradigm, factor_names in paradigm_factor_map.items():
            if not factor_names:
                continue

            n_factors = len(factor_names)
            n_structural = 0
            n_linear = 0
            n_improving = 0
            half_lives = []

            for fname in factor_names:
                ic = self.ic_cache.get(fname, [])
                if len(ic) < self.min_history_days:
                    continue
                fit = self.fit_hyperbolic_decay(ic)
                if not fit["sufficient_data"]:
                    continue

                mode = fit["decay_mode"]
                if mode == "structural":
                    n_structural += 1
                elif mode == "linear":
                    n_linear += 1
                elif mode == "improving":
                    n_improving += 1

                hl = fit["half_life"]
                if hl != float('inf') and hl < 1e6:
                    half_lives.append(hl)

            n_declining = n_structural + n_linear
            pct_declining = n_declining / n_factors if n_factors > 0 else 0.0
            avg_hl = np.mean(half_lives) if half_lives else float('inf')

            # ── Alert level ──
            if pct_declining >= 0.50:
                alert_level = "red"
                msg = (
                    f"🔴 范式「{paradigm}」系统性失效: "
                    f"{n_declining}/{n_factors} 因子处于衰减状态 "
                    f"({pct_declining:.0%}), 其中 structural={n_structural}"
                )
            elif pct_declining >= 0.30:
                alert_level = "yellow"
                msg = (
                    f"🟡 范式「{paradigm}」部分衰减: "
                    f"{n_declining}/{n_factors} 因子衰减({pct_declining:.0%}), "
                    f"structural={n_structural}"
                )
            elif n_structural >= 2:
                alert_level = "yellow"
                msg = (
                    f"🟡 范式「{paradigm}」: {n_structural} 个因子检测到结构性衰减"
                )
            else:
                alert_level = "green"
                msg = ""

            per_paradigm[paradigm] = {
                "n_factors": n_factors,
                "n_structural": n_structural,
                "n_linear": n_linear,
                "n_improving": n_improving,
                "n_declining": n_declining,
                "pct_declining": round(pct_declining, 3),
                "avg_half_life": round(avg_hl, 1) if avg_hl != float('inf') else None,
                "alert_level": alert_level,
                "alert_msg": msg,
            }

            if alert_level in ("red", "yellow"):
                systemic_alerts.append({
                    "paradigm": paradigm,
                    "level": alert_level,
                    "msg": msg,
                })

        # 按严重性排序
        systemic_alerts.sort(key=lambda x: 0 if x["level"] == "red" else 1)

        return {
            "per_paradigm": per_paradigm,
            "systemic_alerts": systemic_alerts,
        }

    def predict_decay_trajectory(
        self,
        ic_series: List[float],
        forecast_horizons: List[int] = None,
    ) -> Dict:
        """
        基于当前 IC 序列拟合的双曲线模型，预测未来 IC 路径。

        Parameters
        ----------
        ic_series : 原始 IC 序列
        forecast_horizons : 预测期列表，默认 [1, 3, 6] 个月（按月计算：
                           1月≈20交易日, 3月≈60, 6月≈120）

        Returns
        -------
        {
            "current_ic": float,
            "half_life": float,
            "decay_mode": str,
            "predictions": [
                {
                    "horizon_label": "1M",
                    "horizon_steps": 20,
                    "pred_ic": float,
                    "ci_lower": float,    # 80% 置信下界
                    "ci_upper": float,    # 80% 置信上界
                    "warning": str,       # "归零风险" 或 ""
                },
                ...
            ],
            "time_to_zero": float or None,   # 预计归零时间 (步数)
            "sufficient_data": bool,
        }
        """
        if forecast_horizons is None:
            forecast_horizons = [
                (1, "1M", 20),
                (3, "3M", 60),
                (6, "6M", 120),
            ]  # (月数, 标签, 交易日步数)

        result = {
            "current_ic": 0.0,
            "half_life": float('inf'),
            "decay_mode": "stable",
            "predictions": [],
            "time_to_zero": None,
            "sufficient_data": False,
        }

        fit = self.fit_hyperbolic_decay(ic_series)
        if not fit["sufficient_data"]:
            return result

        K = fit["K"]
        lam = fit["lambda"]
        hl = fit["half_life"]
        mode = fit["decay_mode"]

        result["current_ic"] = fit["recent_ic"]
        result["half_life"] = hl
        result["decay_mode"] = mode
        result["sufficient_data"] = True

        # ── 为置信区间估计残差标准差 ──
        ic_abs = np.abs(np.array(ic_series, dtype=np.float64))
        t_hist = np.arange(len(ic_abs), dtype=np.float64)
        ic_pos = np.maximum(ic_abs, 1e-8)
        y_pred_hist = K / (1.0 + lam * t_hist[-len(ic_pos):])
        # Pad to match lengths
        y_pred_hist = K / (1.0 + lam * t_hist)
        residuals = ic_pos - y_pred_hist
        residual_std = np.std(residuals) if len(residuals) > 1 else 0.01
        # 80% CI: ±1.28σ
        z80 = 1.28

        current_t = len(ic_series)

        # ── Time to zero: 当 α(t)=0.002（可忽略水平）→ K/(1+λt)=0.002 → t=(K/0.002-1)/λ ──
        ic_floor = 0.002
        if lam > 1e-8 and K > ic_floor:
            t_zero = (K / ic_floor - 1) / lam - current_t
            result["time_to_zero"] = round(max(0, t_zero), 1)

        # ── Generate predictions ──
        for months, label, steps in forecast_horizons:
            t_future = current_t + steps
            pred_ic = K / (1.0 + lam * t_future)

            # Confidence interval: 来自残差σ → 预测值±z80·σ
            ci_lower = max(0.0, pred_ic - z80 * residual_std)
            ci_upper = pred_ic + z80 * residual_std

            # Warning
            warning = ""
            if hl != float('inf') and steps >= hl * 0.5:
                if steps >= hl:
                    warning = f"⚠️ 已超过半衰期 {hl:.0f} 步, 预计IC衰减>50%"
                else:
                    warning = f"🐢 {steps}步内将接近半衰期({hl:.0f}步)"

            if pred_ic < ic_floor:
                warning = f"🔴 {label}后预计IC ≈ {pred_ic:.4f}, 濒临归零"

            result["predictions"].append({
                "horizon_label": label,
                "horizon_steps": steps,
                "pred_ic": round(pred_ic, 6),
                "ci_lower": round(ci_lower, 6),
                "ci_upper": round(ci_upper, 6),
                "warning": warning,
            })

        return result

    def _get_paradigm_factor_map(self) -> Dict[str, List[str]]:
        """
        从 library_orthogonality 获取范式→因子名映射。
        """
        paradigm_map = {}
        try:
            from research.factor_alchemy.library_orthogonality import LibraryOrthogonalityManager
            from research.factor_alchemy.paradigm_v4 import PARADIGMS_V4
            mgr = LibraryOrthogonalityManager()
            for fname, fe in mgr.factors.items():
                p = getattr(fe, 'paradigm', 'unknown')
                if p not in paradigm_map:
                    paradigm_map[p] = []
                paradigm_map[p].append(fname)
        except Exception:
            # 回退：从 ic_cache 中简单分组
            for fname in self.ic_cache:
                if 'unknown' not in paradigm_map:
                    paradigm_map['unknown'] = []
                paradigm_map['unknown'].append(fname)
        return paradigm_map

    def scan_all_structural(self, verbose: bool = True) -> Dict:
        """
        v2 增强扫描：结合 P-004 线性检测 + P-022 hyperbolic 检测，
        输出范式级衰减聚合报告。

        Returns
        -------
        {
            "linear_alerts": [...],       # P-004 原有线性检测告警
            "structural_alerts": [...],   # P-022 hyperbolic structural decay
            "paradigm_decay": {...},      # 范式级衰减聚合
            "trajectories": {...},        # 高风险因子的衰减轨迹预测
            "scanned_total": int,
            "scan_timestamp": str,
        }
        """
        # P-004 线性检测
        linear_report = self.scan_all(verbose=False)

        # P-022 hyperbolic 检测
        structural_alerts = []
        trajectories = {}

        for factor_name, ic_series in self.ic_cache.items():
            if len(ic_series) < self.min_history_days:
                continue

            fit = self.fit_hyperbolic_decay(ic_series)
            if not fit["sufficient_data"]:
                continue

            mode = fit["decay_mode"]
            hl = fit["half_life"]
            lam = fit["lambda"]

            if mode == "structural":
                severity = "danger" if hl < 60 else "warn"
                alert = DecayAlert(
                    factor_name=factor_name,
                    alert_type="STRUCTURAL_DECAY",
                    severity=severity,
                    current_ic=fit["recent_ic"],
                    ic_stability=lam,
                    ic_trend="declining",
                    detail=(
                        f"Hyperbolic R²={fit['r2_hyperbolic']:.3f} > Linear R²={fit['r2_linear']:.3f}, "
                        f"λ={lam:.4f}, 半衰期={hl if hl!=float('inf') else '∞'} 步"
                    ),
                    detected_at=datetime.now().isoformat(),
                )
                structural_alerts.append(alert)

                # 轨迹预测
                traj = self.predict_decay_trajectory(ic_series)
                if traj["sufficient_data"]:
                    trajectories[factor_name] = traj

        # 范式级聚合
        paradigm_decay = self.paradigm_decay_aggregate()

        scan_result = {
            "linear_alerts": linear_report["alerts"],
            "structural_alerts": [a.to_dict() for a in structural_alerts],
            "paradigm_decay": paradigm_decay,
            "trajectories": {k: {
                "decay_mode": v["decay_mode"],
                "half_life": v["half_life"],
                "predictions": v["predictions"],
                "time_to_zero": v["time_to_zero"],
            } for k, v in trajectories.items()},
            "scanned_total": len(self.ic_cache),
            "scan_timestamp": datetime.now().isoformat(),
        }

        self._save_log(scan_result)

        if verbose:
            n_struct = len(structural_alerts)
            n_linear = len(linear_report["alerts"])
            n_red = len([a for a in paradigm_decay.get("systemic_alerts", []) if a["level"] == "red"])
            n_yellow = len([a for a in paradigm_decay.get("systemic_alerts", []) if a["level"] == "yellow"])

            print(f"\n[S6 DecayMonitor v2] 增强扫描完成:")
            print(f"  扫描因子数: {len(self.ic_cache)}")
            print(f"  线性告警(P-004): {n_linear} | 结构性衰减(P-022): {n_struct}")
            print(f"  范式级警报: {n_red} RED + {n_yellow} YELLOW")

            # 范式级告警详情
            for sa in paradigm_decay.get("systemic_alerts", []):
                print(f"  {sa['msg']}")

            # 结构性衰减 Top-3
            for a in structural_alerts[:3]:
                print(f"  📉 {a.factor_name}: {a.detail[:100]}")

        return scan_result

    def get_decay_alert_for_diagnose(self) -> Dict:
        """
        为 run_v4_pipeline.py --diagnose 提供轻量衰减告警摘要。
        不修改文件，只读分析 IC 缓存。

        Returns
        -------
        {
            "structural_count": int,
            "linear_count": int,
            "systemic_paradigm_alerts": [{paradigm, level, msg}],
            "top_risk_factors": [{name, half_life, time_to_zero}],
            "overall_health": "green"|"yellow"|"red",
        }
        """
        report = self.scan_all_structural(verbose=False)

        struct_count = len(report["structural_alerts"])
        linear_count = len(report["linear_alerts"])
        systemic = report["paradigm_decay"].get("systemic_alerts", [])

        # 提取 Top-5 风险因子（按半衰期排序）
        risk_factors = []
        for name, traj in report["trajectories"].items():
            hl = traj.get("half_life", float('inf'))
            tz = traj.get("time_to_zero")
            risk_factors.append({
                "name": name,
                "half_life": hl if hl != float('inf') else None,
                "time_to_zero": tz,
                "decay_mode": traj["decay_mode"],
            })
        risk_factors.sort(key=lambda x: x["half_life"] if x["half_life"] is not None else 1e9)

        # Overall health
        n_red = len([a for a in systemic if a["level"] == "red"])
        n_yellow = len([a for a in systemic if a["level"] == "yellow"])
        if n_red > 0:
            overall = "red"
        elif n_yellow > 0 or struct_count > 0:
            overall = "yellow"
        else:
            overall = "green"

        return {
            "structural_count": struct_count,
            "linear_count": linear_count,
            "systemic_paradigm_alerts": systemic,
            "top_risk_factors": risk_factors[:5],
            "overall_health": overall,
        }


# ── 便捷函数 ──────────────────────────────────────────────

_default_monitor: Optional[DecayMonitor] = None


# ══════════════════════════════════════════════════════════════
# P-20260825-004: 拥挤度三阶段衰减模型 (crowding_decay)
# 拥挤 → 族内同质化 → IC 集体衰减 的因果链检测 (7月行业回撤教训)
# 影子模式: 阈值离线回放校准完成前 alert_enabled=False (只记录不报警)
# ══════════════════════════════════════════════════════════════

CROWDING_DECAY_CONFIG_PATH = DATA_DIR / "crowding_decay_config.json"
CROWDING_DECAY_HISTORY_PATH = DATA_DIR / "crowding_decay_history.json"

DEFAULT_CROWDING_DECAY_CONFIG = {
    "version": "P-20260825-004",
    "corr_threshold": 0.25,        # 族内中位 pairwise corr 阈值 (待离线回放校准)
    "ic_drop_threshold": 0.30,     # 族级 IC 均值环比降幅阈值 (待离线回放校准)
    "lookback_days": 5,            # IC 对比窗口: 近N日均值 vs 前N日均值
    "confirm_days": 2,             # 连续满足天数确认 (防单日噪音误报)
    "alert_enabled": False,        # 校准完成前影子模式 (只记录不报警)
    "calibrated_at": None,
}


def load_crowding_decay_config() -> Dict:
    """加载拥挤衰减配置 (不存在则写默认配置)"""
    if CROWDING_DECAY_CONFIG_PATH.exists():
        try:
            with open(CROWDING_DECAY_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # 补齐新增字段
            for k, v in DEFAULT_CROWDING_DECAY_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
        except Exception:
            pass
    save_crowding_decay_config(DEFAULT_CROWDING_DECAY_CONFIG)
    return dict(DEFAULT_CROWDING_DECAY_CONFIG)


def save_crowding_decay_config(cfg: Dict) -> None:
    CROWDING_DECAY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CROWDING_DECAY_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def aggregate_family_ic(orthogonality_path: Optional[Path] = None) -> Dict[str, Dict]:
    """从 library_orthogonality_state.json 聚合族级 IC 统计

    Returns
    -------
    {family_name: {"ic_median": float, "ic_mean": float, "n_factors": int}}
    """
    if orthogonality_path is None:
        orthogonality_path = DATA_DIR / "library_orthogonality_state.json"
    if not orthogonality_path.exists():
        return {}
    with open(orthogonality_path, "r", encoding="utf-8") as f:
        st = json.load(f)
    factors = st.get("factors", [])
    fam: Dict[str, List[float]] = {}
    for x in factors:
        paradigm = x.get("paradigm") or "未知"
        ic = x.get("ic")
        if ic is None:
            continue
        try:
            ic = float(ic)
        except (TypeError, ValueError):
            continue
        fam.setdefault(str(paradigm), []).append(ic)
    return {
        name: {
            "ic_median": float(np.median(vals)),
            "ic_mean": float(np.mean(vals)),
            "n_factors": len(vals),
        }
        for name, vals in fam.items()
    }


def scan_crowding_decay(
    family_corr_map: Dict[str, float],
    family_ic_map: Optional[Dict[str, Dict]] = None,
    verbose: bool = True,
) -> Dict:
    """族级拥挤衰减扫描 + 历史归档 (P-20260825-004)

    Parameters
    ----------
    family_corr_map: {族名: 族内中位corr}  (来自 factor_level_crowding.json family_max_ratio)
    family_ic_map:   {族名: {ic_median, n_factors}}  (来自 aggregate_family_ic)

    Returns
    -------
    {
        "date": str,
        "triggered": List[str],     # 本次触发双阈值的族
        "alerted": List[str],       # 触发且满足连续确认天数 → 进入告警 (影子模式仅记录)
        "shadow_mode": bool,        # True = 校准未完成, 只记录不报警
        "details": Dict[str, Dict],
    }
    """
    cfg = load_crowding_decay_config()
    family_ic_map = family_ic_map or {}

    # 读取历史 (上一快照)
    history: List[Dict] = []
    if CROWDING_DECAY_HISTORY_PATH.exists():
        try:
            with open(CROWDING_DECAY_HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    prev = history[-1] if history else None
    prev_families = (prev.get("families", {}) if prev else {}) or {}

    details: Dict[str, Dict] = {}
    triggered: List[str] = []
    today = datetime.now().strftime("%Y-%m-%d")

    for fam, corr in family_corr_map.items():
        ic_now = family_ic_map.get(fam, {}).get("ic_median")
        ic_prev = prev_families.get(fam, {}).get("ic_now")
        drop_pct = None
        if ic_now is not None and ic_prev not in (None, 0) and ic_prev != 0:
            drop_pct = (ic_prev - ic_now) / abs(ic_prev)

        corr_ok = corr > cfg["corr_threshold"]
        drop_ok = (drop_pct is not None and drop_pct > cfg["ic_drop_threshold"])
        hit = corr_ok and drop_ok

        # 连续确认天数 (防单日噪音)
        prev_consec = prev_families.get(fam, {}).get("consecutive", 0) or 0
        consecutive = (prev_consec + 1) if hit else 0

        if hit:
            triggered.append(fam)
        details[fam] = {
            "corr": round(corr, 4),
            "ic_now": round(ic_now, 6) if ic_now is not None else None,
            "ic_prev": round(float(ic_prev), 6) if ic_prev is not None else None,
            "drop_pct": round(drop_pct, 4) if drop_pct is not None else None,
            "triggered": hit,
            "consecutive": consecutive,
        }

    alerted = [
        fam for fam, d in details.items()
        if d["triggered"] and d["consecutive"] >= cfg["confirm_days"]
    ]

    # 历史归档 (append)
    history.append({
        "date": today,
        "families": details,
        "triggered": triggered,
        "alerted": alerted,
        "alert_enabled": bool(cfg.get("alert_enabled", False)),
        "recorded_at": datetime.now().isoformat(),
    })
    if len(history) > 400:
        history = history[-400:]
    CROWDING_DECAY_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CROWDING_DECAY_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    shadow = not bool(cfg.get("alert_enabled", False))
    if verbose:
        mode = "影子模式(只记录)" if shadow else "告警已启用"
        print(f"\n[P-004 crowding_decay] 扫描 {len(family_corr_map)} 族 | {mode}")
        if triggered:
            for fam in triggered:
                d = details[fam]
                print(f"  ⚠️ {fam}: corr={d['corr']} "
                      f"IC {d['ic_prev']}→{d['ic_now']} (降{d['drop_pct']:.0%}) "
                      f"连续{d['consecutive']}天")
        if alerted:
            icon = "🟡 记录" if shadow else "🔴 告警"
            print(f"  {icon}: {', '.join(alerted)}")
        elif not triggered:
            print("  无族触发双阈值")

    return {
        "date": today,
        "triggered": triggered,
        "alerted": alerted,
        "shadow_mode": shadow,
        "details": details,
    }


def get_decay_monitor() -> DecayMonitor:
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = DecayMonitor()
    return _default_monitor


# ── 测试 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    dm = DecayMonitor()

    # 模拟数据: 正常因子
    ic_stable = np.random.normal(0.03, 0.01, 100)
    dm.update_ic_cache("test_stable_factor", ic_stable)

    # 模拟数据: 衰减因子
    ic_decaying = np.concatenate([
        np.random.normal(0.04, 0.01, 40),
        np.random.normal(0.025, 0.015, 40),  # IC 下降 + 波动变大
        np.random.normal(0.01, 0.02, 40),    # 进一步衰减
    ])
    dm.update_ic_cache("test_decaying_factor", ic_decaying)

    # 扫描
    report = dm.scan_all()
    print(f"\n{dm.get_summary_for_log()}")
