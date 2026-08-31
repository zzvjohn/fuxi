# -*- coding: utf-8 -*-
"""
Weekly Lane Judge — v0.7 频率对称双通道: 周频裁决器 (2026-08-29)
================================================================================

架构 (research/v07_forge_s1_calibration_plan_20260829.md v3):
  生成层频率中立 × 裁决层 S1 分频 XOR 路由 × 执行层 S5/S6/JQ 唯一周频口径

本模块职责: natural_freq=="weekly" 的候选在**周频口径**下裁决 S1。
  - 数据源 = weekly_prices.parquet (与 Forge 建模同源, W-FRI 采样, T≈279)
  - fwd_ret 直接用 parquet 预计算列 (1 周持有), 与 Forge fitness 口径一致
  - OHLC 用 close 代理 (与 Forge 内部一致: 周频 parquet 只有 close/vol/amount)
  - 裁决指标: 全史周频 Rank ICIR + 近 2 年活性校验 (防全史强但近段塌陷死因子)

铁则:
  - V07_DUAL_LANE["enabled"]=False 时本模块完全不被调用 (零回归)
  - 旁证只开闸不赦免: 周频 S1 通过后 S2-S6/JQ 全照走, JQ 唯一真相源不变
  - 一因子一通道 (XOR): weekly 候选不再走日频 FactorICComputer (避免双重检验)

用法:
    from weekly_lane import WeeklyLaneJudge, get_weekly_judge
    r = get_weekly_judge().judge(formula, factor_name)
    # r = {"weekly_ic": ..., "weekly_icir": ..., "weekly_n": ...,
    #      "weekly_icir_recent": ..., "activity_ok": bool}
"""

import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# 项目根 = 本文件上三级 (research/factor_alchemy -> research -> quant)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PARQUET = _PROJECT_ROOT / "output" / "ap_batch" / "cache" / "weekly_prices.parquet"


class WeeklyLaneJudge:
    """周频 S1 裁决器: 公式在周频宽表上 eval → 逐周 Rank IC → 全史 ICIR + 近 2 年活性。"""

    def __init__(
        self,
        parquet_path=None,
        icir_threshold: float = 0.45,
        ic_min: float = 0.005,
        activity_years: float = 2.0,
        activity_min_icir: float = 0.2,
        min_stocks: int = 30,
    ):
        self.parquet_path = Path(parquet_path) if parquet_path else _DEFAULT_PARQUET
        # v0.7 P4 (2026-08-29): 相对路径解析到项目根。流水线脚本常 os.chdir(FA_DIR),
        # config 的 weekly_parquet 是项目根相对路径 → 原样传给 Path 会解析到错误位置。
        if not self.parquet_path.is_absolute():
            self.parquet_path = _PROJECT_ROOT / self.parquet_path
        self.icir_threshold = icir_threshold
        self.ic_min = ic_min
        self.activity_years = activity_years
        self.activity_min_icir = activity_min_icir
        self.min_stocks = min_stocks
        # v0.7 P1 (2026-08-29): 活性校验降级为影子 (只标注不拦截)。
        # 实测 ts_std/tsmin 近 2 年 ICIR 0.074/0.140 仍全期 JQ PASS —
        # 硬拦截会误杀已验证 PASS 因子; 校准集充足以再转正。
        self.activity_gate = False

        self._loaded = False
        self._wide_env: Dict[str, pd.DataFrame] = {}
        self._fwd_ret: Optional[pd.DataFrame] = None  # 宽表 (index=date, cols=stock)
        self._all_dates: Optional[pd.DatetimeIndex] = None
        self._universe = []
        self._mkt_state: Dict = {}  # P-20260830-004: {date: trend|range|down}

    # ── 数据加载 ──────────────────────────────────────────

    def _load(self) -> bool:
        if self._loaded:
            return True
        if not self.parquet_path.exists():
            print(f"  [WeeklyLane] ⚠️ 周频数据不存在: {self.parquet_path}")
            return False
        try:
            w = pd.read_parquet(self.parquet_path)
        except Exception as e:
            print(f"  [WeeklyLane] ⚠️ 周频数据加载失败: {e}")
            return False

        w = w.drop_duplicates(subset=["stock_code", "trade_date"], keep="last")
        w = w.sort_values(["stock_code", "trade_date"])

        self._universe = sorted(w["stock_code"].unique())
        self._all_dates = pd.DatetimeIndex(sorted(w["trade_date"].unique()))

        def _pivot(col: str) -> pd.DataFrame:
            sub = w[["trade_date", "stock_code", col]].dropna(subset=[col])
            wide = sub.set_index(["trade_date", "stock_code"])[col].unstack()
            return wide.reindex(index=self._all_dates, columns=self._universe)

        close_w = _pivot("close")
        vol_w = _pivot("vol")
        amt_w = _pivot("amount")

        # v0.7 P3c (2026-08-29): 接入真实周频 OHLC (weekly_ohlc.parquet, 从日频重建:
        # open=周首日 open, high=周内 max, low=周内 min, 覆盖率 100%)。
        # 无该文件时回退 close 代理 (向后兼容, 与 Forge 旧口径一致)。
        ohlc_path = self.parquet_path.parent / "weekly_ohlc.parquet"
        open_w = high_w = low_w = None
        if ohlc_path.exists():
            try:
                _o = pd.read_parquet(ohlc_path)
                _o = _o.drop_duplicates(subset=["stock_code", "trade_date"], keep="last")

                def _pivot_ohlc(col: str) -> pd.DataFrame:
                    sub = _o[["trade_date", "stock_code", col]].dropna(subset=[col])
                    wide = sub.set_index(["trade_date", "stock_code"])[col].unstack()
                    return wide.reindex(index=self._all_dates, columns=self._universe)

                open_w, high_w, low_w = (_pivot_ohlc(c) for c in ("open", "high", "low"))
                print(f"  [WeeklyLane] 真实周频 OHLC 已接入 (open/high/low 非 close 代理)")
            except Exception as e:
                print(f"  [WeeklyLane] ⚠️ weekly_ohlc 加载失败, 回退 close 代理: {e}")

        # OHLC 用真实值 (P3c 后), vwap 用 vol 代理 (与 Forge 语义一致)
        env: Dict[str, pd.DataFrame] = {
            "close": close_w, "close_p": close_w,
            "open": open_w if open_w is not None else close_w,
            "open_p": open_w if open_w is not None else close_w,
            "high": high_w if high_w is not None else close_w,
            "high_p": high_w if high_w is not None else close_w,
            "low": low_w if low_w is not None else close_w,
            "low_p": low_w if low_w is not None else close_w,
            "volume": vol_w, "volume_p": vol_w,
            "amount": amt_w, "amount_p": amt_w,
            "vwap": vol_w,
            "returns": close_w.pct_change(),
            "returns_p": close_w.pct_change(),
        }
        for extra in ("momentum_12_1", "vol_60", "log_size"):
            ew = _pivot(extra)
            if ew is not None and not ew.isna().all().all():
                env[extra] = ew

        self._wide_env = env
        self._fwd_ret = _pivot("fwd_ret")
        self._loaded = True

        # P-20260830-004: 市场状态分段 (等权市场指数代理, 周频)
        # 状态 = f(4 周动量, 4 周波动): trend(动量>2%) / down(动量<-2%) / range(其余)
        try:
            mkt_close = close_w.mean(axis=1)
            mkt_mom = mkt_close.pct_change(4)
            mkt_vol = mkt_close.pct_change().rolling(4).std()
            state_map = {}
            for _d, _m, _v in zip(mkt_close.index, mkt_mom, mkt_vol):
                if pd.isna(_m):
                    state_map[_d] = "range"  # 早期 warmup 归震荡
                elif _m > 0.02:
                    state_map[_d] = "trend"
                elif _m < -0.02:
                    state_map[_d] = "down"
                else:
                    state_map[_d] = "range"
            self._mkt_state = state_map
            n_st = {}
            for s in ("trend", "range", "down"):
                n_st[s] = sum(1 for v in state_map.values() if v == s)
            print(f"  [WeeklyLane] 市场状态分段就绪 (P-004 影子): trend={n_st['trend']}周 "
                  f"range={n_st['range']}周 down={n_st['down']}周")
        except Exception as e:
            self._mkt_state = {}
            print(f"  [WeeklyLane] ⚠️ 市场状态分段失败 (影子退避): {e}")

        print(f"  [WeeklyLane] 周频数据加载: {len(w)} 行, {len(self._universe)} 股, "
              f"{len(self._all_dates)} 周 ({self._all_dates.min().date()} ~ {self._all_dates.max().date()})")
        return True

    # ── 公式清洗 (对齐 FactorICComputer._eval_formula 语义) ──

    @staticmethod
    def _clean_formula(formula: str) -> Optional[str]:
        clean = (formula or "").strip()
        if not clean or clean.startswith("# TODO"):
            return None
        lines = [ln.strip() for ln in clean.split("\n")
                 if ln.strip() and not ln.strip().startswith("#")]
        if not lines:
            return None
        body_lines = []
        if len(lines) > 1:
            body_lines = lines[:-1]
            final_line = lines[-1]
            m = re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*(.+)$', final_line)
            if m:
                final_line = m.group(1).strip()
            clean = final_line
        else:
            if "=" in clean and not any(op in clean for op in ["<=", ">=", "==", "!="]):
                parts = clean.split("=", 1)
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', parts[0].strip()):
                    clean = parts[1].strip()

        # 裸 rank 轴修正 (v0.9.4/v0.9.5 同款三段式): 宽表裸 .rank(pct=True)
        # 默认 axis=0 沿时间轴=前视; rolling(N).rank 保持原样。
        _roll_rank_re = re.compile(r'(\.rolling\([^)]*\))\.rank\(pct\s*=\s*True\)')
        _plain_rank_re = re.compile(r'\.rank\(pct\s*=\s*True\)')
        _holders = []
        clean = _roll_rank_re.sub(
            lambda m: _holders.append(m.group(0)) or f'{m.group(1)}.__RRK{len(_holders)-1}__',
            clean)
        clean = _plain_rank_re.sub('.rank(pct=True, axis=1)', clean)
        for _i, _h in enumerate(_holders):
            clean = clean.replace(f'__RRK{_i}__', 'rank(pct=True)')
        return clean

    # ── 裁决 ─────────────────────────────────────────────

    def judge(self, formula: str, factor_name: str = "unknown") -> Dict:
        """周频裁决。返回 {weekly_ic, weekly_icir, weekly_n,
        weekly_icir_recent, activity_ok, passed, reason, eval_ok}"""
        empty = {
            "weekly_ic": 0.0, "weekly_icir": 0.0, "weekly_n": 0,
            "weekly_icir_recent": 0.0, "activity_ok": False,
            "passed": False, "reason": "", "eval_ok": False,
        }
        if not self._load():
            empty["reason"] = "周频数据不可用"
            return empty

        clean = self._clean_formula(formula)
        if not clean:
            empty["reason"] = "公式不可解析"
            return empty

        # eval 环境 (对齐 FactorICComputer: 宽表 eval + 基本函数)
        safe_builtins = {
            'range': range, 'len': len, 'int': int, 'float': float,
            'list': list, 'dict': dict, 'tuple': tuple, 'str': str, 'bool': bool,
            'True': True, 'False': False, 'None': None,
            'abs': abs, 'min': min, 'max': max, 'round': round,
            'sum': sum, 'zip': zip, 'sorted': sorted,
            'print': print, 'isinstance': isinstance,
            'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
        }
        base_context = {
            "np": np, "pd": pd,
            "abs": np.abs, "sqrt": np.sqrt, "log": np.log,
            "log1p": np.log1p, "exp": np.exp, "sign": np.sign,
            "maximum": np.maximum, "minimum": np.minimum,
            "where": np.where, "clip": np.clip,
            "range": range, "len": len, "int": int, "float": float,
            "list": list, "dict": dict, "tuple": tuple,
        }
        full_globals = {"__builtins__": safe_builtins, **base_context, **self._wide_env}

        import warnings
        warnings.filterwarnings("ignore")
        try:
            result = eval(clean, full_globals, {})
        except Exception as e:
            empty["reason"] = f"周频 eval 失败: {type(e).__name__}: {str(e)[:90]}"
            return empty

        if isinstance(result, pd.DataFrame):
            factor_df = (result.rename_axis("trade_date")
                         .rename_axis("stock_code", axis=1)
                         .stack().rename("factor_value").reset_index())
        elif isinstance(result, pd.Series):
            wide = result.to_frame().reindex(columns=self._universe)
            factor_df = (wide.rename_axis("trade_date")
                         .rename_axis("stock_code", axis=1)
                         .stack().rename("factor_value").reset_index())
        else:
            empty["reason"] = f"周频 eval 结果类型不支持: {type(result).__name__}"
            return empty

        factor_df = factor_df.dropna(subset=["factor_value"])
        factor_df = factor_df[np.isfinite(factor_df["factor_value"])]
        if len(factor_df) == 0:
            empty["reason"] = "周频因子值全 NaN"
            return empty

        # merge 周频前向收益 (parquet 预计算, 1 周持有, 与 Forge fitness 同口径)
        fwd_long = (self._fwd_ret.rename_axis("trade_date")
                    .rename_axis("stock_code", axis=1)
                    .stack().rename("fwd_ret").reset_index())
        merged = factor_df.merge(fwd_long, on=["stock_code", "trade_date"], how="inner")
        merged = merged.dropna(subset=["factor_value", "fwd_ret"])

        ic_series = []
        for date, group in merged.groupby("trade_date"):
            if len(group) < self.min_stocks:
                continue
            ic = group["factor_value"].rank().corr(group["fwd_ret"].rank())
            if not np.isnan(ic):
                ic_series.append((date, ic))

        if not ic_series:
            empty["reason"] = "周频 IC 序列为空 (样本不足)"
            return empty

        ic_arr = np.array([v for _, v in ic_series])
        mean_ic = float(np.nanmean(ic_arr))
        std_ic = float(np.nanstd(ic_arr))
        icir = abs(mean_ic / std_ic) if std_ic > 0 else 0.0

        # 活性校验: 近 activity_years 年 ICIR
        recent_cut = ic_series[-1][0] - pd.Timedelta(days=int(self.activity_years * 365.25))
        recent = [v for d, v in ic_series if d >= recent_cut]
        recent_icir = 0.0
        if len(recent) >= 12:
            r_arr = np.array(recent)
            recent_icir = abs(float(np.nanmean(r_arr) / np.nanstd(r_arr))) if np.nanstd(r_arr) > 0 else 0.0
        activity_ok = (len(recent) >= 12) and (recent_icir >= self.activity_min_icir)

        # 活性校验 P1 阶段为影子: 不参与 passed 判定 (见 __init__ 注释)
        if self.activity_gate and not activity_ok:
            passed = False
        else:
            passed = (icir >= self.icir_threshold and abs(mean_ic) >= self.ic_min)
        reason_bits = [f"周频 ICIR={icir:.3f}(≥{self.icir_threshold})"
                       if icir >= self.icir_threshold else f"周频 ICIR={icir:.3f}<{self.icir_threshold}"]
        if abs(mean_ic) < self.ic_min:
            reason_bits.append(f"|IC|={abs(mean_ic):.4f}<{self.ic_min}")
        if self.activity_gate and not activity_ok:
            reason_bits.append(f"近{self.activity_years:.0f}年 ICIR={recent_icir:.3f}<{self.activity_min_icir}")
        elif not activity_ok:
            reason_bits.append(f"⚠️活性影子: 近{self.activity_years:.0f}年 ICIR={recent_icir:.3f} 偏低(仅标注)")
        reason = "; ".join(reason_bits) + (f" (n={len(ic_arr)}周)" if passed else f" (n={len(ic_arr)}周)")

        return {
            "weekly_ic": round(mean_ic, 6),
            "weekly_icir": round(icir, 4),
            "weekly_n": len(ic_arr),
            "weekly_icir_recent": round(recent_icir, 4),
            "activity_ok": activity_ok,
            "passed": passed,
            "reason": reason,
            "eval_ok": True,
            # P-20260830-004: 市场状态条件化影子 (只输出不改判定)
            "state_shadow": self._state_conditioned_shadow(ic_series),
        }

    # ── P-20260830-004: 市场状态条件化影子 ────────────────

    def _state_conditioned_shadow(self, ic_series) -> Dict:
        """按市场状态三段 (trend/range/down) 分段统计周频 ICIR。

        纯影子输出: 不改 S1 判定。为周频因子提供状态稳健性旁证,
        识别「只在单一状态有效」的脆弱因子 (方向翻转风险预警数据积累)。

        Returns: {trend: {n, mean_ic, icir}, range: {...}, down: {...},
                  consistency: 最差段ICIR/全史ICIR (无数据=1.0)}
        """
        shadow = {"trend": None, "range": None, "down": None, "consistency": None}
        if not self._mkt_state:
            return shadow

        def _seg_stats(ics):
            if not ics or len(ics) < 6:
                return None
            a = np.array(ics)
            m, s = float(np.nanmean(a)), float(np.nanstd(a))
            return {
                "n": len(ics),
                "mean_ic": round(m, 5),
                "icir": round(abs(m / s) if s > 0 else 0.0, 4),
            }

        for d, ic in ic_series:
            st = self._mkt_state.get(d, "range")
            if st in ("trend", "range", "down"):
                shadow[st] = shadow[st] or []
                shadow[st].append(ic)

        stats = {}
        for st in ("trend", "range", "down"):
            stats[st] = _seg_stats(shadow[st] or [])
            shadow[st] = stats[st]

        seg_icirs = [stats[st]["icir"] for st in ("trend", "range", "down") if stats[st]]
        full_arr = np.array([v for _, v in ic_series])
        full_icir = abs(float(np.nanmean(full_arr) / np.nanstd(full_arr))) \
            if np.nanstd(full_arr) > 0 else 0.0
        if seg_icirs and full_icir > 0:
            shadow["consistency"] = round(min(seg_icirs) / full_icir, 3)
        else:
            shadow["consistency"] = 1.0
        return shadow


# ── 单例 (全流水线共享一次 parquet 加载) ──────────────

_judge_singleton: Optional[WeeklyLaneJudge] = None


def get_weekly_judge(force_reload: bool = False) -> Optional[WeeklyLaneJudge]:
    """按 config.V07_DUAL_LANE 构造周频裁决器单例。V07 关闭时返回 None (零开销)。"""
    global _judge_singleton
    if force_reload:
        _judge_singleton = None
    if _judge_singleton is None:
        try:
            from config import V07_DUAL_LANE
        except Exception:
            return None
        if not bool(V07_DUAL_LANE.get("enabled", False)):
            return None
        # v0.7 P4 (2026-08-29): τ_w 动态读取 — 校准集 effective (JQ 锚点滚动确认)
        # 优先于 config 基线; lane_calibration.get_effective_tau_w 是唯一入口。
        try:
            from lane_calibration import get_effective_tau_w
            _tau_w = get_effective_tau_w()
        except Exception:
            _tau_w = float(V07_DUAL_LANE.get("weekly_icir_threshold", 0.15))
        _judge_singleton = WeeklyLaneJudge(
            parquet_path=V07_DUAL_LANE.get("weekly_parquet"),
            icir_threshold=_tau_w,
            ic_min=float(V07_DUAL_LANE.get("weekly_ic_min", 0.005)),
            activity_years=float(V07_DUAL_LANE.get("weekly_activity_years", 2.0)),
            activity_min_icir=float(V07_DUAL_LANE.get("weekly_activity_min_icir", 0.05)),
            min_stocks=int(V07_DUAL_LANE.get("weekly_min_stocks", 30)),
        )
        _judge_singleton.activity_gate = bool(V07_DUAL_LANE.get("weekly_activity_gate", False))
    return _judge_singleton


def v07_enabled() -> bool:
    try:
        from config import V07_DUAL_LANE
        return bool(V07_DUAL_LANE.get("enabled", False))
    except Exception:
        return False


def infer_natural_freq(candidate: Dict) -> str:
    """natural_freq 推断 (兜底): factorforge 来源 → weekly; 其余默认 daily。
    候选已显式携带 natural_freq 时以候选为准。"""
    freq = str(candidate.get("natural_freq") or "").strip().lower()
    if freq in ("weekly", "daily"):
        return freq
    src = str(candidate.get("_source", candidate.get("source", "")) or "").lower()
    if src == "factorforge" or src == "forge":
        return "weekly"
    return "daily"
