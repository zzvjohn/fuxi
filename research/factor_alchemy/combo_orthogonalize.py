# -*- coding: utf-8 -*-
"""M14 (2026-09-27): 组合层正交化诊断 —— "基组合重复度"纯归因读数 (零裁决)
============================================================================
方案依据: reports/fuxi_alpha_generation_overhaul_20260926.md §M14
依赖: M13 control 固定版本快照 (data/ralph_control_snapshot.json)

🔴🔴 用途边界 (承接用户 2026-09-15 钦定约束 + 2026-07-30 V5 证伪先例, 修改前必读)
================================================================================
本模块**只做归因诊断**, 对外只输出标量读数。它回答的问题是:

    "候选因子的表观增量里, 有多少是**基组合已有信息的重新包装** (base_r2),
     有多少是真正的独立信息 (残差口径 IC)?"

**严禁**将本模块的回归残差、正交化结果或任何中间产物用于:
    × 因子中性化处理 (把残差当作新因子 / 构造"残差组合"/ 回测残差策略)
    × 因子入库 / 候选改写 / S1·S5·增量门裁决输入
    × 生成器上下文注入 (不得提示 LLM/GP "请做正交化")

依据 (库内既证, 非推测):
  1. V5 中性化路线已**证伪** —— JQ 实测 +30.77% / Sharpe 0.06 / DD −26.52%,
     "中性化 + HRP = alpha 杀手", 2026-07-30 正式放弃 (见 MEMORY.md 已证伪清单);
  2. P-20260915-001 (style_exposure_guard) 用户钦定边界:
     **诊断可做, 改因子不可做**;
  3. 07-21 StableAlpha 结论: "风格暴露 ≠ Alpha 来源" ⇒ 正确用法是把暴露
     当**读数**报出来, 而不是把因子改造成"无暴露版本"。

接口层三重防护 (从设计上杜绝误用, 与 style_exposure_guard 同构):
    1. 不 import 任何 neutralize 模块 (独立实现, 无耦合);
    2. `factor_check()` / `evaluate()` 只返回标量字典, **不返回残差面板**,
       落盘只写读数 JSON, 不写任何因子文件;
    3. 残差**在逐期循环内算完即弃** (不缓存、不落盘、不进任何下游消费)。

与 M13 的分工 (避免口径分叉)
============================
  · M13 (incremental_pairing) = **组合层增量门**: rank_sum_combo(ctl+[cand]) 与
    rank_sum_combo(ctl) 的 IR/DD 配对差 —— 回答"并入后组合读数是否改善";
  · M14 (本模块) = **因子级正交旁证**: 候选对基组合因子逐期截面回归的
    base_r2 / 三口径 IC —— 回答"这个增量有多少是基组合重复信息"。
  两者并列读, M14 **绝不进入 M13 的判定** (gate_passed / decision 不受影响)。
  control 公式与 M13 同源 (读同一快照), 缺失时返回 data_insufficient,
  **不自行兜底选公式** (否则两模块口径静默分叉)。

三口径 (K4 铁律: 组内恒定因子×哑变量⇒秩相关造假显著, 必须并列报出)
====================================================================
  1. `rank_raw`     : 候选 (截面 rank) 与前向收益的秩相关 —— 表观口径;
  2. `rank_resid`   : 对基组合因子回归取残差后的秩相关 —— 剔除重复信息;
  3. `pearson_raw`  : 候选原始值与收益的线性相关 —— 防秩变换掩盖假象。
  另报 `ic_decay_ratio = (|rank_raw| − |rank_resid|) / |rank_raw|` (主判读量)。
  逐期截面回归前**剔除零方差控制列** (组内恒定因子的防护), 计数落盘。

成本纪律 (Phase E 每轮跑, 不得引入重 IO)
========================================
  · 只复用 ralph_loop 已加载的 FactorICComputer 面板 (attach_computer);
  · **不读** daily_basic.csv (1377 万行) —— 风格 β 读数由 style_exposure_guard
    承担 (summary.style_guard), 本模块默认不重复计算 (可选 --with-style 显式开);
  · 控制面板进程内缓存 (control 快照冻结 ⇒ 可安全复用)。

开关
====
env `FUXI_COMBO_ORTHO`:
    shadow | on | true | 1 (默认) -> 生效, 零裁决, 落 data/combo_ortho_shadow.json
    off | 0 | false              -> 完全禁用
默认 shadow: 本模块不改变任何裁决 (零裁决类同 trade_gate_shadow / style_guard)。

🔴 未实现的显式扩展位 (需用户授权, 默认不做)
============================================
"组合层残差口径净增量" (用残差因子构造组合回测再与 control 比) **不在本模块**:
该口径实质是把中性化产物送进策略回测, 正是 V5 证伪路径的复发面。
若未来确需, 必须: ① 用户显式授权; ② 独立开关 + 独立预注册; ③ 读数永不进裁决。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent          # e:/quant
DATA_DIR = BASE / "data"
IND_PATH = DATA_DIR / "sw_industry_map.csv"
DEFAULT_SNAPSHOT_PATH = DATA_DIR / "ralph_control_snapshot.json"
OUT_PATH = DATA_DIR / "combo_ortho_shadow.json"

SCHEMA = "fuxi_combo_orthogonalize/v1"
MODES: Tuple[str, ...] = ("off", "shadow")

# ── 判据参数 (env 可覆盖, 全部有安全默认) ──────────────────────────────
DEFAULT_TOP_CTL = 8
DEFAULT_MAX_CAND = 4
DEFAULT_MIN_STOCKS = 50
DEFAULT_MAX_PERIODS = 120
DEFAULT_FWD = 5
R2_REPLICATED = 0.80          # base_r2_median ≥ 此值 ⇒ 候选 ≈ 基组合线性复制
DECAY_RISK = 0.50             # ic_decay_ratio ≥ 此值 且表观 IC 达标 ⇒ 弱增量
MIN_IC_ABS = 0.02             # 表观 IC 的最小可读幅度 (与 StyleGuard 同值)
MIN_PERIODS_OK = 10           # 少于此期数 ⇒ data_insufficient (不硬报)

FORBIDDEN_USE = (
    "仅归因诊断: 本模块输出(含内部回归残差)禁止用于因子中性化处理/因子入库/"
    "候选改写/裁决输入/生成器注入。V5 中性化路线已证伪 "
    "(JQ +30.77% / Sharpe 0.06 / DD -26.52%, 中性化+HRP = alpha 杀手, 2026-07-30 放弃)。"
    "M14 只把 base_r2/三口径 IC 当读数并列报出, 不改因子、不改判定。"
)

_CTX: Dict[str, Any] = {}     # 进程内缓存: computer / price / ctl 面板 / ind_map


def _dump(payload: Dict[str, Any], out_path: Optional[Any] = None) -> None:
    """落盘读数 (只写标量 JSON; 失败非阻塞)。"""
    try:
        _out = Path(out_path) if out_path else OUT_PATH
        _out.parent.mkdir(parents=True, exist_ok=True)
        with open(_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"  [ComboOrtho] ⚠️ 影子落盘失败 (非阻塞): {e}")


# ═══════════════════════════════════════════════════════════
# 开关与参数
# ═══════════════════════════════════════════════════════════

def get_mode() -> str:
    """读取开关。默认 shadow (生效但零裁决)。非法值 fail-safe 回 shadow。"""
    raw = str(os.environ.get("FUXI_COMBO_ORTHO", "shadow") or "shadow").strip().lower()
    if raw in ("off", "0", "false", "no", "disable", "disabled", "none"):
        return "off"
    return "shadow"


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)) or default)
    except Exception:
        return int(default)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)) or default)
    except Exception:
        return float(default)


# ═══════════════════════════════════════════════════════════
# 数据层 (复用已加载面板, 零额外重 IO)
# ═══════════════════════════════════════════════════════════

def _get_computer():
    if "computer" not in _CTX:
        from factor_ic_computer import FactorICComputer
        comp = FactorICComputer(lookback_days=int(os.environ.get("FUXI_STYLE_LOOKBACK_DAYS", "270") or 270))
        comp._load_data()
        _CTX["computer"] = comp
    return _CTX["computer"]


def attach_computer(comp) -> None:
    """复用 ralph_loop 已加载的 FactorICComputer (避免重复读面板)。"""
    if comp is not None:
        _CTX["computer"] = comp


def _price_frame() -> Optional[pd.DataFrame]:
    """从已加载面板派生 fwd_return (与 StyleGuard 同口径: close.shift(-FWD)/close − 1)。

    只在面板原列上做一次 groupby 变换并缓存复用。
    """
    if "px" in _CTX:
        return _CTX["px"]
    comp = _get_computer()
    px = getattr(comp, "_price_df", None)
    if px is None or len(px) == 0:
        _CTX["px"] = None
        return None
    need = [c for c in ("ts_code", "trade_date", "close") if c in px.columns]
    if len(need) < 3:
        _CTX["px"] = None
        return None
    fwd = _env_int("FUXI_COMBO_ORTHO_FWD", DEFAULT_FWD)
    px = (px[need].drop_duplicates(subset=["ts_code", "trade_date"])
          .sort_values(["ts_code", "trade_date"]).reset_index(drop=True))
    g = px.groupby("ts_code", sort=False)["close"]
    px["fwd_return"] = (g.shift(-fwd) / px["close"].clip(lower=1e-9) - 1.0).astype("float32")
    out = px[["ts_code", "trade_date", "fwd_return"]]
    _CTX["px"] = out
    return out


def _industry_map() -> Dict[str, str]:
    """申万一级行业映射 code → industry (静态表, 无时变)。"""
    if "ind_map" in _CTX:
        return _CTX["ind_map"]
    m: Dict[str, str] = {}
    try:
        d = pd.read_csv(IND_PATH, dtype=str)
        ccol = "code" if "code" in d.columns else d.columns[0]
        icol = "industry" if "industry" in d.columns else d.columns[1]
        m = dict(zip(d[ccol].astype(str), d[icol].fillna("(未知)").astype(str)))
    except Exception as e:
        print(f"  [ComboOrtho] ⚠️ 行业映射读取失败 (退化为无行业读数): {e}")
    _CTX["ind_map"] = m
    return m


def _panel(comp, formula: str, name: str) -> Optional[pd.DataFrame]:
    """评估公式 → [ts_code, trade_date, factor_value]; 失败返回 None。"""
    if not formula:
        return None
    try:
        fdf = comp._eval_formula(str(formula), str(name)[:40])
    except Exception:
        return None
    if fdf is None or len(fdf) == 0:
        return None
    if "factor_value" not in fdf.columns:
        return None
    fdf = fdf[["ts_code", "trade_date", "factor_value"]].dropna(subset=["factor_value"])
    return fdf if len(fdf) else None


def _ctl_series(comp, ctl_formulas: List[str], sha_key: str
                ) -> Tuple[List[Tuple[str, pd.Series]], List[Dict[str, str]]]:
    """控制因子面板 → (list[(formula, Series[(ts_code,trade_date)])], skipped)。

    进程内缓存: control 快照冻结 ⇒ 同一 sha_key 下可安全复用。
    """
    key = ("ctl", sha_key)
    if key in _CTX:
        return _CTX[key]
    series: List[Tuple[str, pd.Series]] = []
    skipped: List[Dict[str, str]] = []
    for i, f in enumerate(ctl_formulas or []):
        fdf = _panel(comp, f, f"ctl{i}")
        if fdf is None:
            skipped.append({"i": i, "formula": str(f)[:80], "reason": "no_frame"})
            continue
        s = fdf.set_index(["ts_code", "trade_date"])["factor_value"].astype("float64")
        series.append((str(f), s))
    _CTX[key] = (series, skipped)
    return series, skipped


# ═══════════════════════════════════════════════════════════
# 统计工具
# ═══════════════════════════════════════════════════════════

def _rank01(a: np.ndarray) -> np.ndarray:
    """截面 rank(pct)*2−1 (与 StyleGuard 同口径: [-1,1] 均匀化)。"""
    s = pd.Series(a)
    return (s.rank(pct=True).to_numpy(dtype="float64") * 2.0 - 1.0)


def _z(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype="float64")
    mu = np.nanmean(a) if a.size else 0.0
    sd = np.nanstd(a) if a.size else 0.0
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(a)
    out = (a - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _rank_ic(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """截面秩相关 (与 FactorICComputer 同口径)。"""
    if a.size < 5:
        return None
    sa = pd.Series(a).rank()
    sb = pd.Series(b).rank()
    v = sa.corr(sb)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """截面线性相关 (防秩变换掩盖假象的第三口径)。"""
    if a.size < 5:
        return None
    sa = pd.Series(a)
    sb = pd.Series(b)
    v = sa.corr(sb)          # pandas 默认 pearson
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def _tstat(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    a = np.asarray(xs, dtype="float64")
    m = float(np.nanmean(a))
    s = float(np.nanstd(a))
    n = int(a.size)
    t = (m / (s / np.sqrt(n))) if (s > 0 and n > 1) else 0.0
    return m, float(t)


# ═══════════════════════════════════════════════════════════
# 核心: 单候选正交化归因
# ═══════════════════════════════════════════════════════════

def factor_check(formula: str, name: str, ctl_formulas: List[str], *,
                 comp=None, ctl_sha: str = "", verbose: bool = False) -> Optional[Dict[str, Any]]:
    """对单个候选做"基组合重复度"归因 (只返回标量, 不返回残差)。

    逐期截面: y = rank01(候选), X = [1, rank01(ctl_i)...] → 残差 e;
      · base_r2      = 1 − SS_res/SS_tot (中位数) —— 候选被基组合解释的比例;
      · rank_raw     = rank_corr(y, fwd_return);
      · rank_resid   = rank_corr(e, fwd_return);
      · pearson_raw  = pearson_corr(y, fwd_return);
      · 零方差控制列逐期剔除 (组内恒定因子的防护), 计数落盘。
    """
    comp = comp or _get_computer()
    if comp is None:
        return None

    fdf = _panel(comp, formula, name)
    if fdf is None:
        return None

    series_list = [fdf.set_index(["ts_code", "trade_date"])["factor_value"]
                   .astype("float64").rename("y")]
    ctl_series, ctl_skipped = _ctl_series(
        comp, ctl_formulas or [], ctl_sha or "adhoc")
    if not ctl_series:
        return {"factor_name": str(name)[:60], "status": "data_insufficient",
                "reason": "control_series_empty", "n_ctl_skipped": len(ctl_skipped)}
    for i, (_f, s) in enumerate(ctl_series):
        series_list.append(s.rename(f"x{i}"))

    px = _price_frame()
    mat = pd.concat(series_list, axis=1)

    # 窗口裁剪: 只保留最后 MAX_PERIODS 个交易日 (与 StyleGuard 采样上限一致)
    dts = np.sort(pd.Index(mat.index.get_level_values("trade_date")).unique())
    max_p = _env_int("FUXI_COMBO_ORTHO_MAX_PERIODS", DEFAULT_MAX_PERIODS)
    if len(dts) > max_p:
        dts = dts[-max_p:]
        mat = mat[mat.index.get_level_values("trade_date").isin(dts)]

    if px is not None:
        mat = mat.join(px.set_index(["ts_code", "trade_date"])["fwd_return"], how="left")
    else:
        mat["fwd_return"] = np.nan
    mat = mat.reset_index()
    # 完整性契约: 候选与全部控制非空 (fwd_return 允许缺, 缺期在 IC 处单独剔除)
    need = ["y"] + [c for c in mat.columns if c.startswith("x")]
    mat = mat.dropna(subset=need)
    if len(mat) == 0:
        return {"factor_name": str(name)[:60], "status": "data_insufficient",
                "reason": "no_overlap_rows"}

    min_stocks = _env_int("FUXI_COMBO_ORTHO_MIN_STOCKS", DEFAULT_MIN_STOCKS)
    dates = np.sort(mat["trade_date"].unique())

    xcols_all = [c for c in mat.columns if c.startswith("x")]
    base_r2s: List[float] = []
    ic_raw: List[float] = []
    ic_res: List[float] = []
    ic_pear: List[float] = []
    n_stocks_used: List[int] = []
    n_ctl_used: List[int] = []
    n_zero_var_dropped = 0
    n_periods_skipped_small = 0
    ind_r2s: List[float] = []
    ind_map = _industry_map()

    for d in dates:
        g = mat[mat["trade_date"] == d]
        if len(g) < min_stocks:
            n_periods_skipped_small += 1
            continue
        y = _rank01(g["y"].to_numpy(dtype="float64"))
        if not np.isfinite(y).all():
            continue

        cols = []
        for c in xcols_all:
            v = g[c].to_numpy(dtype="float64")
            if np.nanstd(v) <= 1e-12:            # 组内恒定/零方差 ⇒ 剔除 (防护)
                n_zero_var_dropped += 1
                continue
            cv = _rank01(v)
            if not np.isfinite(cv).all():
                continue
            cols.append(cv)
        if not cols:
            continue
        X = np.column_stack([np.ones(len(g))] + cols)
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            continue
        resid = y - X @ beta
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        base_r2s.append(1.0 - ss_res / max(ss_tot, 1e-12))
        n_stocks_used.append(int(len(g)))
        n_ctl_used.append(int(len(cols)))

        r = g["fwd_return"].to_numpy(dtype="float64")
        ok = np.isfinite(r)
        if ok.sum() >= 5:
            yr, rr, er = y[ok], r[ok], resid[ok]
            v1 = _rank_ic(yr, rr)
            v2 = _rank_ic(er, rr)
            v3 = _pearson(yr, rr)
            if v1 is not None and v2 is not None:
                ic_raw.append(v1)
                ic_res.append(v2)
            if v3 is not None:
                ic_pear.append(v3)

        # 行业解释力: 方差分解 (不用哑变量, 与 StyleGuard 同口径)
        try:
            gm = y.mean()
            sst = float(((y - gm) ** 2).sum())
            if sst > 1e-12 and ind_map:
                ind = pd.Series(g["ts_code"].astype(str)).map(ind_map).fillna("(未知)").to_numpy()
                sw = 0.0
                for _, gg in pd.Series(y).groupby(ind):
                    v = gg.to_numpy(dtype="float64")
                    sw += float(((v - v.mean()) ** 2).sum())
                ind_r2s.append(1.0 - sw / sst)
        except Exception:
            pass

    n_periods = len(base_r2s)
    if n_periods < MIN_PERIODS_OK:
        return {"factor_name": str(name)[:60], "status": "data_insufficient",
                "reason": f"periods({n_periods}<{MIN_PERIODS_OK})",
                "n_periods": int(n_periods),
                "n_periods_skipped_small": int(n_periods_skipped_small)}

    base_r2_med = float(np.median(base_r2s))
    rank_raw_m = float(np.nanmean(ic_raw)) if ic_raw else 0.0
    rank_resid_m = float(np.nanmean(ic_res)) if ic_res else 0.0
    pearson_m = float(np.nanmean(ic_pear)) if ic_pear else 0.0
    decay = (((abs(rank_raw_m) - abs(rank_resid_m)) / abs(rank_raw_m))
             if abs(rank_raw_m) > 1e-9 else 0.0)

    # 读法标签 (纯提示, 零裁决)
    if base_r2_med >= R2_REPLICATED:
        read = "replicated_exposure"
    elif abs(rank_raw_m) >= MIN_IC_ABS and decay > DECAY_RISK:
        read = "weak_increment"
    else:
        read = "increment_candidate"

    row: Dict[str, Any] = {
        "factor_name": str(name)[:60],
        "status": "valid",
        "read": read,                      # 读法提示, 非裁决
        "n_periods": int(n_periods),
        "n_periods_skipped_small": int(n_periods_skipped_small),
        "n_stocks_median": int(np.median(n_stocks_used)) if n_stocks_used else 0,
        "n_ctl_used_median": int(np.median(n_ctl_used)) if n_ctl_used else 0,
        "n_ctl_skipped": int(len(ctl_skipped)),
        "n_zero_var_col_dropped": int(n_zero_var_dropped),
        "base_r2_median": round(base_r2_med, 4),
        "rank_raw": round(rank_raw_m, 6),
        "rank_resid": round(rank_resid_m, 6),
        "pearson_raw": round(pearson_m, 6),
        "ic_decay_ratio": round(float(decay), 4),
        "industry_r2_median": (round(float(np.median(ind_r2s)), 4) if ind_r2s else None),
        "rank_raw_t": round(_tstat(ic_raw)[1], 2),
        "rank_resid_t": round(_tstat(ic_res)[1], 2),
        "evidence_note": ("base_r2=候选被基组合解释的比例; ic_decay=(|raw|-|resid|)/|raw|; "
                          "三者并列读, 不构成裁决"),
    }
    if verbose:
        print(f"  [ComboOrtho] {str(name)[:30]}: base_r2 {base_r2_med:.3f} | "
              f"rank_raw {rank_raw_m:+.4f} → rank_resid {rank_resid_m:+.4f} "
              f"(衰减 {decay:+.1%}) | pearson {pearson_m:+.4f} | "
              f"行业R² {row['industry_r2_median']} | 读法 {read}")
    return row


# ═══════════════════════════════════════════════════════════
# control 快照读取 (与 M13 同源)
# ═══════════════════════════════════════════════════════════

def load_control(snapshot_path: Optional[Any] = None
                 ) -> Tuple[List[str], Dict[str, Any]]:
    """读 M13 control 快照 → (formulas, meta)。

    缺失/损坏 ⇒ ([], {"status": "snapshot_missing"|...}) —— **不兜底自选**
    (自选会与 M13 门静默口径分叉)。
    """
    p = Path(snapshot_path) if snapshot_path else DEFAULT_SNAPSHOT_PATH
    meta: Dict[str, Any] = {"path": str(p), "status": "unknown"}
    if not p.exists():
        meta["status"] = "snapshot_missing"
        return [], meta
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        meta["status"] = f"snapshot_unreadable: {type(e).__name__}: {e}"
        return [], meta
    snap = d.get("snapshot", d) if isinstance(d, dict) else {}
    formulas = [str(f) for f in (snap.get("formulas") or []) if str(f).strip()]
    meta.update({
        "status": "ok" if formulas else "snapshot_empty",
        "revision": snap.get("revision"),
        "snapshot_sha256": snap.get("snapshot_sha256"),
        "rule": snap.get("rule"),
        "n_formulas": len(formulas),
    })
    return formulas, meta


def _pick_candidates(candidates: List[Dict], results: List[Any],
                     max_n: int) -> List[Dict[str, str]]:
    """优先取 S5/JQ 候选; 无则退到 S1 通过; 再退到全部 (上限 max_n)。"""
    picked: List[Dict[str, str]] = []
    for tier in ("jq", "s1", "all"):
        for i, cand in enumerate(candidates or []):
            if len(picked) >= max_n:
                break
            f = cand.get("formula") or cand.get("expression") or ""
            if not f:
                continue
            res = results[i] if i < len(results or []) else None
            if tier == "jq" and not getattr(res, "eligible_for_jq", False):
                continue
            if tier == "s1" and not getattr(res, "s1_passed", False):
                continue
            picked.append({
                "formula": f,
                "name": cand.get("factor_name") or cand.get("name") or f"cand_{i}",
                "paradigm": str(cand.get("paradigm", "")),
            })
        if picked:
            break
    return picked[:max_n]


# ═══════════════════════════════════════════════════════════
# 轮级入口 (Phase E 钩子)
# ═══════════════════════════════════════════════════════════

def evaluate(candidates: List[Dict], results: List[Any], *,
             ctl_formulas: Optional[List[str]] = None,
             snapshot_path: Optional[Any] = None,
             out_path: Optional[Any] = None,
             verbose: bool = True) -> Dict[str, Any]:
    """Ralph Phase E 钩子: 组合层正交化影子诊断 (零裁决, 仅归因)。

    返回只含标量读数; 不返回/不落盘任何残差面板; 不修改 candidates/results。
    """
    mode = get_mode()
    if mode == "off":
        return {"schema": SCHEMA, "mode": "off", "status": "not_evaluated",
                "n_checked": 0, "rows": []}

    ctl_meta: Dict[str, Any] = {}
    if ctl_formulas is None:
        ctl_formulas, ctl_meta = load_control(snapshot_path)
    else:
        ctl_meta = {"status": "injected", "n_formulas": len(ctl_formulas),
                    "snapshot_sha256": None, "revision": None}
    if not ctl_formulas:
        payload = {
            "schema": SCHEMA, "mode": mode, "status": "data_insufficient",
            "reason": "control_unavailable (与 M13 同源快照缺失; 不兜底自选)",
            "control": ctl_meta, "n_checked": 0, "rows": [],
            "forbidden_use": FORBIDDEN_USE,
        }
        _dump(payload, out_path)
        if verbose:
            print(f"  [ComboOrtho] {mode}: control 不可用 "
                  f"({ctl_meta.get('status')}) ⇒ data_insufficient (不硬报)")
        return payload

    sha = str(ctl_meta.get("snapshot_sha256") or "")
    max_n = _env_int("FUXI_COMBO_ORTHO_MAX", DEFAULT_MAX_CAND)
    rows: List[Dict[str, Any]] = []
    for p in _pick_candidates(candidates, results, max_n):
        try:
            row = factor_check(p["formula"], p["name"], ctl_formulas,
                               ctl_sha=sha, verbose=verbose)
            if row:
                rows.append(row)
        except Exception as e:
            rows.append({"factor_name": str(p["name"])[:60], "status": "error",
                         "reason": f"{type(e).__name__}: {str(e)[:120]}"})

    valid = [r for r in rows if r.get("status") == "valid"]
    read_dist: Dict[str, int] = {}
    for r in valid:
        read_dist[r["read"]] = read_dist.get(r["read"], 0) + 1
    base_r2s = [r["base_r2_median"] for r in valid]

    status = "valid" if valid else ("data_insufficient" if rows else "not_evaluated")
    payload = {
        "schema": SCHEMA,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "status": status,
        "control": dict(ctl_meta),
        "params": {
            "max_cand": max_n,
            "min_stocks": _env_int("FUXI_COMBO_ORTHO_MIN_STOCKS", DEFAULT_MIN_STOCKS),
            "max_periods": _env_int("FUXI_COMBO_ORTHO_MAX_PERIODS", DEFAULT_MAX_PERIODS),
            "fwd": _env_int("FUXI_COMBO_ORTHO_FWD", DEFAULT_FWD),
            "r2_replicated": R2_REPLICATED,
            "decay_risk": DECAY_RISK,
        },
        "n_checked": len(valid),
        "n_replicated": int(read_dist.get("replicated_exposure", 0)),
        "n_weak_increment": int(read_dist.get("weak_increment", 0)),
        "n_increment_candidate": int(read_dist.get("increment_candidate", 0)),
        "median_base_r2": (round(float(np.median(base_r2s)), 4) if base_r2s else None),
        "read_distribution": read_dist,
        "n_data_insufficient": sum(1 for r in rows if r.get("status") == "data_insufficient"),
        "n_error": sum(1 for r in rows if r.get("status") == "error"),
        "rows": rows,
        "forbidden_use": FORBIDDEN_USE,
    }
    _dump(payload, out_path)

    if verbose:
        print(f"  [ComboOrtho] {mode} 模式 (仅归因, 零裁决): 检查 {len(valid)} 候选 | "
              f"读法分布 {read_dist} | base_r2 中位 "
              f"{payload['median_base_r2'] if payload['median_base_r2'] is not None else 'NA'}")
    return payload


# ═══════════════════════════════════════════════════════════
# CLI (独立运行 + 真实库取证)
# ═══════════════════════════════════════════════════════════

def _replay_formulas(limit: int = 4) -> List[Dict[str, str]]:
    """回放样本: 取 stage1 因子提案 (多样范式) 优先, 退化到冠军池。"""
    out: List[Dict[str, str]] = []
    p = DATA_DIR / "stage1_factor_proposals.json"
    if p.exists():
        try:
            d = json.load(open(p, encoding="utf-8"))
            arr = d if isinstance(d, list) else d.get("proposals", d.get("factors", []))
            seen: Dict[str, int] = {}
            for it in arr:
                f = it.get("formula") or it.get("expression") or ""
                if not f:
                    continue
                pa = str(it.get("paradigm", "") or "(未标注)")
                if seen.get(pa, 0) >= 2:
                    continue
                seen[pa] = seen.get(pa, 0) + 1
                out.append({"formula": f,
                            "name": it.get("factor_name", it.get("name", f"prop_{len(out)}")),
                            "paradigm": pa})
                if len(out) >= limit:
                    break
        except Exception:
            pass
    return out


def _cli() -> int:
    ap = argparse.ArgumentParser(description="M14 组合层正交化诊断 (仅归因, 零裁决)")
    ap.add_argument("--replay", action="store_true", help="历史回放验证")
    ap.add_argument("--limit", type=int, default=4, help="回放因子数上限")
    ap.add_argument("--formulas-file", dest="formulas_file", default=None,
                    help="JSON 因子清单 (含 formula/name/paradigm)")
    ap.add_argument("--snapshot", dest="snapshot", default=None,
                    help="control 快照路径 (默认 data/ralph_control_snapshot.json)")
    ap.add_argument("--out", dest="out", default=None, help="读数落盘路径 (默认 data/combo_ortho_shadow.json)")
    ap.add_argument("--quiet", action="store_true", help="不打印逐条读数")
    args = ap.parse_args()

    items: List[Dict[str, str]] = []
    if args.formulas_file:
        d = json.load(open(args.formulas_file, encoding="utf-8"))
        arr = d if isinstance(d, list) else d.get("proposals", d.get("factors", []))
        for it in arr:
            f = it.get("formula") or it.get("expression") or ""
            if f:
                items.append({"formula": f,
                              "name": it.get("factor_name", it.get("name", "")),
                              "paradigm": str(it.get("paradigm", ""))})
    elif args.replay:
        items = _replay_formulas(int(args.limit or 4))
    else:
        ap.print_help()
        return 0

    print(f"[ComboOrtho] 归因对象 {len(items)} 因子 (mode={get_mode()})")
    payload = evaluate(items, [None] * len(items),
                       snapshot_path=args.snapshot, out_path=args.out,
                       verbose=not args.quiet)
    print(f"[ComboOrtho] 报告: {args.out or OUT_PATH} | status={payload.get('status')}")
    return 0 if payload.get("status") in ("valid", "data_insufficient") else 1


if __name__ == "__main__":
    sys.exit(_cli())
