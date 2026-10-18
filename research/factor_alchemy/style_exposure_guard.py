# -*- coding: utf-8 -*-
"""P-20260915-001: 风格捷径阴影诊断 (style exposure attribution guard)。

🔴🔴 用途边界 (用户 2026-09-15 钦定, 硬约束, 修改本文件前必读)
================================================================
本模块**仅做归因诊断**。对外只输出标量读数 (暴露 β / IC 衰减率 / R²)。

**严禁**将本模块的回归残差、中性化结果或任何中间产物用于:
    × 因子中性化处理 (把残差当作新因子)
    × 因子入库 / 候选改写 / S1·S5 裁决输入
    × 生成器上下文注入 (不得提示 LLM/GP "请做中性化")

依据: V5 中性化路线已**证伪** —— JQ 实测 +30.77% / Sharpe 0.06 / DD −26.52%,
"中性化 + HRP = alpha 杀手", 2026-07-30 正式放弃 (见 MEMORY.md 已证伪清单)。
而 07-21 的 StableAlpha 结论是"风格暴露 ≠ Alpha 来源"⇒ 正确边界 =
**诊断可做, 改因子不可做**。

接口层三重防护 (从设计上杜绝误用):
    1. 不 import 任何 neutralize 模块 (独立实现, 无耦合);
    2. `factor_check()` / `evaluate()` 只返回标量字典, **不返回残差面板**;
    3. 落盘报告写入 `forbidden_use` 字段, 并只落 JSON 读数, 不写任何因子文件。

背景 (双源相互论证, cross_verified=true)
========================================
  1) 国泰海通量化选股系列(五)《让深度学习因子适应不同市场环境》:
     深度学习因子端到端训练后风格暴露高度集中 (低波 / 小市值 / 反转);
     多粒度因子与市值相关性 −0.3; 正交(中性化)后 IC-IR 显著提升但组合收益下滑
     ⇒ 反证表观 IC 中相当部分由风格暴露贡献。
  2) 广发金工《StableAlpha: 更稳定的深度学习因子》:
     同一现象 —— 中性化后 RankIC 明显下降甚至失去稳定性; 主张用训练期风格约束
     替代事后中性化。

定位
====
纯诊断影子层, **零裁决**: 不改门槛 / 不改生成器 / 不改任何裁决路径。
Ralph Phase E 在 SurvGuard 之后调用; 日报风险预警节报读数。

四项读数
========
  A. 风格暴露 β: 逐期截面 OLS 的 β 时序均值 + t 统计 (size / lowvol / reversal)
  B. IC 衰减率: (|IC_raw| − |IC_resid|) / |IC_raw|  ← 主判据
     衰减率高 ⇒ 表观 IC 主要来自风格暴露, 属"风格捷径"伪 alpha
  C. 风格解释力 R²: 因子截面方差被三项风格解释的比例 (中位数)
  D. 行业解释力: 1 − SS_within / SS_total (申万一级方差分解, 中位数)
     —— 零额外成本, 不用 30+ 行业哑变量 (避免小样本自由度污染)

开关
====
env `FUXI_STYLE_GUARD`:
    on | true | 1 | shadow (默认) → 生效 (零裁决归因)
    off | 0 | false               → 关闭
✅ 默认**开启** (用户 2026-09-15 钦定), 与其它守卫"默认 shadow/off"不同。

风格正交响应分层 (P-20260919-003, 三方论证 cross_verified)
============================================================
在四项读数之外, 为每个因子补一个**先验提示**字段: 按范式/公式/名称机理把因子
分入 动量系/反转系/价值系/盈利系/其他, 引用 Scruggs (JoI 2021-08) 实证 sign ——
反转/价值因子做风格正交预期 IR 增益 (+), 动量/盈利因子预期损害 (−)。
三方独立论证: 国泰海通量化选股(五) (深度学习因子加风格惩罚层反而更不稳定) ×
Scruggs JoI 2021 × 新浪金工 (风格中性化跟踪误差不降反升)。
**纯提示, 零裁决**: 不改变 shortcut 判定 / 不触发任何中性化 / forbidden_use 铁律不变。
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
BASIC_PATH = DATA_DIR / "raw" / "daily_basic.csv"
IND_PATH = DATA_DIR / "sw_industry_map.csv"
OUT_PATH = DATA_DIR / "style_exposure_guard_report.json"

STYLE_COLS: Tuple[str, ...] = ("size", "lowvol", "reversal")

FORBIDDEN_USE = (
    "仅归因诊断: 本模块输出(含内部回归残差)禁止用于因子中性化处理/因子入库/"
    "候选改写/裁决输入/生成器注入。V5 中性化路线已证伪 "
    "(JQ +30.77% / Sharpe 0.06 / DD -26.52%, 中性化+HRP = alpha 杀手, 2026-07-30 放弃)。"
)

# ── P-20260919-003: 风格正交响应分层 (Scruggs JoI 2021 先验, 纯提示零裁决) ──
ORTHO_PRIOR_EVIDENCE = (
    "Scruggs, Does Neutralizing Style Factors Help or Hurt?, JoI 2021-08: "
    "反转/价值因子 industry-neutralization 预期增益(+); 动量/盈利因子预期损害(-)。"
    "交叉论证: 国泰海通量化选股(五)2026-09 风格惩罚层使组合更不稳定; "
    "新浪金工: 风格中性化跟踪误差不降反升。"
)

# 范式/公式/名称 关键词 → 因子类型 (按列表顺序取首个命中; substring 小写匹配)
ORTHO_TYPE_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("reversal", ("reversal", "反转", "rev_", "_rev")),
    ("value", ("value", "估值", "bp_", "_bp", "ep_", "_ep", "pb", "ps_", "dividend", "股息")),
    ("momentum", ("momentum", "动量", "trend", "趋势", "roc(", "vroc", "mom", "breakout", "ma_")),
    ("profitability", ("earnings", "profit", "盈利", "roe", "quality", "质量",
                       "growth", "成长", "accrual", "cashflow", "净利")),
)
# Scruggs 实证 sign: 反转/价值 = 中性化增益; 动量/盈利 = 中性化损害
ORTHO_SIGN: Dict[str, str] = {"reversal": "+", "value": "+",
                              "momentum": "-", "profitability": "-"}


def neutralization_prior(paradigm: str, formula: str, name: str) -> Dict[str, Any]:
    """按机理类型给「若做风格正交, 预期 IR 变化方向」的先验提示。

    纯提示, 零裁决: 结果只进影子报告, 不进任何裁决/生成路径。
    """
    text = f"{paradigm}|{formula}|{name}".lower()
    ftype = "other"
    for t, kws in ORTHO_TYPE_KEYWORDS:
        if any(k.lower() in text for k in kws):
            ftype = t
            break
    sign = ORTHO_SIGN.get(ftype)
    note = ""
    if sign is not None:
        note = (f"{ftype} 因子: 风格正交预期 IR{'增益' if sign == '+' else '损害'} ({sign}); "
                + ORTHO_PRIOR_EVIDENCE)
    return {"factor_type": ftype, "ortho_expected_ir_sign": sign,
            "ortho_prior_note": note}

# ── 判据参数 (env 可覆盖, 全部有安全默认) ─────────────────────
LOOKBACK_DAYS = int(os.environ.get("FUXI_STYLE_LOOKBACK_DAYS", "270") or 270)
MAX_FORMULAS = int(os.environ.get("FUXI_STYLE_MAX_FORMULAS", "8") or 8)
MIN_STOCKS = int(os.environ.get("FUXI_STYLE_MIN_STOCKS", "50") or 50)
MAX_PERIODS = int(os.environ.get("FUXI_STYLE_MAX_PERIODS", "120") or 120)
DECAY_RISK = float(os.environ.get("FUXI_STYLE_DECAY_RISK", "0.50") or 0.50)
MIN_IC_ABS = float(os.environ.get("FUXI_STYLE_MIN_IC", "0.02") or 0.02)
R2_RISK = float(os.environ.get("FUXI_STYLE_R2_RISK", "0.30") or 0.30)
T_RISK = float(os.environ.get("FUXI_STYLE_T_RISK", "2.0") or 2.0)
FORWARD_PERIOD = int(os.environ.get("FUXI_STYLE_FWD", "5") or 5)

_CTX: Dict[str, Any] = {}   # 进程内缓存: computer / style_px / mv / ind_map


def get_mode() -> str:
    """读取开关。默认 on (用户 2026-09-15 钦定)。"""
    raw = str(os.environ.get("FUXI_STYLE_GUARD", "on") or "on").strip().lower()
    if raw in ("off", "0", "false", "no", "disable", "disabled", "none"):
        return "off"
    return "shadow"       # on/true/1/shadow/enforce 一律落到"生效但零裁决"


# ═══════════════════════════════════════════════════════════
# 数据层: 价格衍生风格 + 市值 + 行业 (全部进程内缓存)
# ═══════════════════════════════════════════════════════════

def _get_computer():
    if "computer" not in _CTX:
        from factor_ic_computer import FactorICComputer
        _CTX["computer"] = FactorICComputer(lookback_days=LOOKBACK_DAYS)
        _CTX["computer"]._load_data()
    return _CTX["computer"]


def attach_computer(comp) -> None:
    """复用 ralph_loop 已加载的 FactorICComputer (避免重复读 ~560MB 面板)。"""
    if comp is not None:
        _CTX["computer"] = comp


def _style_price_frame() -> pd.DataFrame:
    """从已加载面板派生风格原料: lowvol(20日收益std) / reversal(20日累计收益) / fwd_return。

    只在面板原列上做一次 groupby 变换, 结果缓存复用 (逐因子不再重算)。
    """
    if "style_px" in _CTX:
        return _CTX["style_px"]

    comp = _get_computer()
    px = comp._price_df
    need = [c for c in ("ts_code", "trade_date", "close") if c in px.columns]
    px = (px[need].drop_duplicates(subset=["ts_code", "trade_date"])
          .sort_values(["ts_code", "trade_date"]).reset_index(drop=True))

    g = px.groupby("ts_code", sort=False)["close"]
    ret = g.pct_change()
    px["lowvol"] = (ret.groupby(px["ts_code"], sort=False)
                    .rolling(20, min_periods=10).std()
                    .reset_index(level=0, drop=True).to_numpy())
    px["reversal"] = (px["close"] / g.shift(20).clip(lower=1e-9) - 1.0).astype("float32")
    px["fwd_return"] = (g.shift(-FORWARD_PERIOD) / px["close"].clip(lower=1e-9)
                        - 1.0).astype("float32")
    px["lowvol"] = pd.to_numeric(px["lowvol"], errors="coerce").astype("float32")
    px = px[["ts_code", "trade_date", "lowvol", "reversal", "fwd_return"]]
    _CTX["style_px"] = px
    return px


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
        print(f"  [StyleGuard] ⚠️ 行业映射读取失败 (退化为无行业读数): {e}")
    _CTX["ind_map"] = m
    return m


def _log_mv_panel() -> Optional[pd.DataFrame]:
    """读 daily_basic 市值 (total_mv, 万元) → log。lookback 窗口 + 去重 + float32。

    🔴 daily_basic.csv 有大规模重复行 (1377万行/703万唯一) ⇒ 必须 drop_duplicates,
       否则 merge 后逐期截面含重复股票, 秩相关与方差分解全部失真。
    """
    if "mv" in _CTX:
        return _CTX["mv"]
    try:
        df = pd.read_csv(BASIC_PATH, usecols=["ts_code", "trade_date", "total_mv"],
                         dtype={"ts_code": str}, low_memory=False)
    except Exception as e:
        print(f"  [StyleGuard] ⚠️ 市值面板读取失败 (size 维度将缺失): {e}")
        _CTX["mv"] = None
        return None

    td = df["trade_date"]
    if pd.api.types.is_integer_dtype(td):
        df["trade_date"] = pd.to_datetime(td.astype(str), format="%Y%m%d", errors="coerce")
    else:
        df["trade_date"] = pd.to_datetime(td, format="mixed", errors="coerce")
    df = df.dropna(subset=["trade_date", "total_mv"])
    df["total_mv"] = pd.to_numeric(df["total_mv"], errors="coerce").astype("float32")
    df = df.dropna(subset=["total_mv"])
    df = df[df["total_mv"] > 0]
    df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    dmax = df["trade_date"].max()
    df = df[df["trade_date"] >= dmax - pd.Timedelta(days=LOOKBACK_DAYS)]
    df["log_mv"] = np.log(df["total_mv"].to_numpy(dtype="float64")).astype("float32")
    df = df[["ts_code", "trade_date", "log_mv"]].reset_index(drop=True)
    _CTX["mv"] = df
    return df


def _z(v: pd.Series) -> np.ndarray:
    a = pd.to_numeric(v, errors="coerce").to_numpy(dtype="float64")
    mu = np.nanmean(a) if a.size else 0.0
    sd = np.nanstd(a) if a.size else 0.0
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(a)
    out = (a - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _rank_ic(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """截面 rank IC (与 FactorICComputer 同口径: 秩相关)。"""
    if a.size < 5:
        return None
    sa = pd.Series(a).rank()
    sb = pd.Series(b).rank()
    v = sa.corr(sb)
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
# 单因子归因 (核心)
# ═══════════════════════════════════════════════════════════

def factor_check(formula: str, name: str, paradigm: str,
                 verbose: bool = False) -> Optional[Dict[str, Any]]:
    """对单个因子做风格暴露归因 (只返回标量读数, 不返回残差)。"""
    comp = _get_computer()
    fdf = comp._eval_formula(formula, str(name)[:40])
    if fdf is None or len(fdf) == 0:
        return None
    fdf = fdf.dropna(subset=["factor_value"])
    if len(fdf) == 0:
        return None

    df = fdf.merge(_style_price_frame(), on=["ts_code", "trade_date"], how="inner")
    if len(df) == 0:
        return None
    mv = _log_mv_panel()
    if mv is not None:
        df = df.merge(mv, on=["ts_code", "trade_date"], how="left")
    else:
        df["log_mv"] = np.nan
    ind_map = _industry_map()
    df["industry"] = df["ts_code"].map(ind_map).fillna("(未知)") if ind_map else "(未知)"

    df = df.dropna(subset=["factor_value", "fwd_return"])
    if len(df) < MIN_STOCKS * 5:
        return None

    dates = np.sort(df["trade_date"].unique())
    if len(dates) > MAX_PERIODS:
        idx = np.unique(np.linspace(0, len(dates) - 1, MAX_PERIODS).round().astype(int))
        dates = dates[idx]

    betas: Dict[str, List[float]] = {c: [] for c in STYLE_COLS}
    r2s: List[float] = []
    ind_r2s: List[float] = []
    ic_raw: List[float] = []
    ic_res: List[float] = []
    n_used = 0
    n_skip_mv = 0

    for d in dates:
        g = df[df["trade_date"] == d]
        if len(g) < MIN_STOCKS:
            continue
        y = (g["factor_value"].rank(pct=True).to_numpy(dtype="float64") * 2.0 - 1.0)
        if not np.isfinite(y).all():
            continue
        r = g["fwd_return"].to_numpy(dtype="float64")

        cols = []
        size_ok = "log_mv" in g.columns and g["log_mv"].notna().sum() >= MIN_STOCKS
        if size_ok:
            cols.append(_z(g["log_mv"]))
        else:
            n_skip_mv += 1
        cols.append(_z(g["lowvol"]))
        cols.append(_z(g["reversal"]))
        X = np.column_stack([np.ones(len(g))] + cols)

        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            continue
        resid = y - X @ beta
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        r2s.append(r2)

        j = 1
        if size_ok:
            betas["size"].append(float(beta[j])); j += 1
        elif cols:
            j += 1
        betas["lowvol"].append(float(beta[j]))
        betas["reversal"].append(float(beta[j + 1]))

        ics_r = _rank_ic(y, r)
        ics_e = _rank_ic(resid, r)
        if ics_r is not None and ics_e is not None:
            ic_raw.append(ics_r)
            ic_res.append(ics_e)

        # 行业解释力: 方差分解 (零额外成本, 不用 30+ 哑变量)
        try:
            gm = y.mean()
            sst = float(((y - gm) ** 2).sum())
            sw = 0.0
            for _, gg in pd.Series(y).groupby(g["industry"].to_numpy()):
                v = gg.to_numpy(dtype="float64")
                sw += float(((v - v.mean()) ** 2).sum())
            if sst > 1e-12:
                ind_r2s.append(1.0 - sw / sst)
        except Exception:
            pass
        n_used += 1

    if n_used < 10:
        return None

    ic_raw_m = float(np.nanmean(ic_raw)) if ic_raw else 0.0
    ic_res_m = float(np.nanmean(ic_res)) if ic_res else 0.0
    decay = ((abs(ic_raw_m) - abs(ic_res_m)) / abs(ic_raw_m)) if abs(ic_raw_m) > 1e-9 else 0.0

    bt: Dict[str, Dict[str, float]] = {}
    for c in STYLE_COLS:
        m, t = _tstat(betas[c])
        bt[c] = {"beta_mean": round(m, 4), "t": round(t, 2),
                 "n_periods": len(betas[c])}
    dom = max(bt.items(), key=lambda kv: abs(kv[1]["t"]))[0] if bt else ""
    r2_med = float(np.median(r2s)) if r2s else 0.0
    ind_r2_med = float(np.median(ind_r2s)) if ind_r2s else None

    shortcut = bool(abs(ic_raw_m) >= MIN_IC_ABS and decay > DECAY_RISK)

    prior = neutralization_prior(paradigm, formula, name)

    return {
        "factor_name": str(name)[:60],
        "paradigm": str(paradigm or ""),
        "ic_raw": round(ic_raw_m, 6),
        "ic_resid": round(ic_res_m, 6),
        "ic_decay_ratio": round(float(decay), 4),
        "style_r2_median": round(r2_med, 4),
        "industry_r2_median": (round(float(ind_r2_med), 4)
                               if ind_r2_med is not None else None),
        "style_beta": bt,
        "dominant_style": dom,
        "n_periods": n_used,
        "n_periods_missing_mv": n_skip_mv,
        "style_shortcut_risk": shortcut,
        "high_style_r2": bool(r2_med > R2_RISK),
        # P-20260919-003: 风格正交响应分层先验 (纯提示, 零裁决)
        "factor_type": prior["factor_type"],
        "ortho_expected_ir_sign": prior["ortho_expected_ir_sign"],
        "ortho_prior_note": prior["ortho_prior_note"],
    }


def _pick_formulas(candidates: List[Dict], results: List[Any]) -> List[Dict[str, str]]:
    """优先取 S5/JQ 候选; 无则退到 S1 通过; 再退到全部 (上限 MAX_FORMULAS)。"""
    picked: List[Dict[str, str]] = []
    for tier in ("jq", "s1", "all"):
        for i, cand in enumerate(candidates or []):
            if len(picked) >= MAX_FORMULAS:
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
                "paradigm": cand.get("paradigm", ""),
            })
        if picked:
            break
    return picked[:MAX_FORMULAS]


def evaluate(candidates: List[Dict], results: List[Any],
             verbose: bool = True) -> Dict[str, Any]:
    """Ralph Phase E 钩子: 风格捷径影子诊断 (零裁决, 仅归因)。

    返回只含标量读数; 不返回/不落盘任何残差面板。
    """
    mode = get_mode()
    if mode == "off":
        return {"mode": "off", "n_checked": 0, "n_shortcut_risk": 0, "rows": []}

    rows: List[Dict[str, Any]] = []
    for p in _pick_formulas(candidates, results):
        try:
            row = factor_check(p["formula"], p["name"], p["paradigm"])
            if row:
                rows.append(row)
        except Exception as e:
            if verbose:
                print(f"  [StyleGuard] {str(p['name'])[:30]} 归因失败 (跳过): "
                      f"{type(e).__name__}: {str(e)[:80]}")

    n_risk = sum(1 for r in rows if r["style_shortcut_risk"])
    n_high_r2 = sum(1 for r in rows if r["high_style_r2"])
    dom_count: Dict[str, int] = {}
    for r in rows:
        dom_count[r["dominant_style"]] = dom_count.get(r["dominant_style"], 0) + 1
    decays = [r["ic_decay_ratio"] for r in rows]

    # P-20260919-003: 因子类型/正交响应先验分布 (纯提示)
    ortho_dist: Dict[str, int] = {}
    for r in rows:
        ft = r.get("factor_type", "other")
        ortho_dist[ft] = ortho_dist.get(ft, 0) + 1

    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "proposal": "P-20260915-001",
        "ortho_prior_layer": "P-20260919-003",
        "mode": mode,
        "forbidden_use": FORBIDDEN_USE,
        "params": {
            "lookback_days": LOOKBACK_DAYS, "min_stocks": MIN_STOCKS,
            "max_periods": MAX_PERIODS, "decay_risk": DECAY_RISK,
            "min_ic_abs": MIN_IC_ABS, "r2_risk": R2_RISK,
            "forward_period": FORWARD_PERIOD,
        },
        "n_checked": len(rows),
        "n_shortcut_risk": n_risk,
        "n_high_style_r2": n_high_r2,
        "median_ic_decay_ratio": round(float(np.median(decays)), 4) if decays else 0.0,
        "dominant_style_dist": dom_count,
        "ortho_prior_dist": ortho_dist,
        "rows": rows,
    }
    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        json.dump(payload, open(OUT_PATH, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"  [StyleGuard] ⚠️ 影子落盘失败 (非阻塞): {e}")

    if verbose:
        print(f"  [StyleGuard] {mode} 模式 (仅归因, 零裁决): 检查 {len(rows)} 因子 | "
              f"风格捷径风险 {n_risk} | 高风格R² {n_high_r2} | "
              f"IC衰减率中位 "
              f"{payload['median_ic_decay_ratio']:+.3f}")
        for r in rows[:3]:
            b = r["style_beta"]
            print(f"      - {str(r['factor_name'])[:28]}: IC_raw {r['ic_raw']:+.4f} "
                  f"→ IC_resid {r['ic_resid']:+.4f} / 衰减 {r['ic_decay_ratio']:+.1%} / "
                  f"R² {r['style_r2_median']:.3f} / 主导 {r['dominant_style']} "
                  f"(β {b.get(r['dominant_style'], {}).get('beta_mean', 0):+.3f}, "
                  f"t {b.get(r['dominant_style'], {}).get('t', 0):+.2f}) | "
                  f"类型 {r.get('factor_type', '?')} → 正交先验 {r.get('ortho_expected_ir_sign', '?')}")
    return payload


# ═══════════════════════════════════════════════════════════
# CLI (独立运行 + 历史回放验证)
# ═══════════════════════════════════════════════════════════

def _replay_formulas(limit: int = 6) -> List[Dict[str, str]]:
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
    if not out:
        pool = DATA_DIR / "unified_factor_pool.csv"
        if pool.exists():
            try:
                df = pd.read_csv(pool, nrows=200)
                for _, r in df.iterrows():
                    f = r.get("formula") or r.get("expression")
                    if isinstance(f, str) and f:
                        out.append({"formula": f, "name": str(r.get("factor_name", "")),
                                    "paradigm": str(r.get("paradigm", ""))})
                    if len(out) >= limit:
                        break
            except Exception:
                pass
    return out


def _cli() -> int:
    ap = argparse.ArgumentParser(description="风格捷径阴影诊断 (P-20260915-001, 仅归因)")
    ap.add_argument("--replay", action="store_true", help="历史回放验证")
    ap.add_argument("--limit", type=int, default=6, help="回放因子数上限")
    ap.add_argument("--formulas-file", dest="formulas_file", default=None,
                    help="JSON 因子清单 (含 formula/name/paradigm)")
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
        items = _replay_formulas(int(args.limit or 6))
    else:
        ap.print_help()
        return 0

    print(f"[StyleGuard] 归因对象 {len(items)} 因子 (mode={get_mode()})")
    payload = evaluate(items, [None] * len(items), verbose=not args.quiet)
    print(f"[StyleGuard] 报告: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
