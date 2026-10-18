"""P-20260912-002: 幸存者偏差守卫影子 (self-evolving 循环的环境自身样本构成审计)。

背景 (三源相互论证, cross_verified=true):
  1) Vadim (2026) "Self-evolving agent: survivorship bias ran the wrong way":
     short-interest flow lane 唯一正向结果在 readmitted universe 下消失
     (IID t 2.47 → 1.69, overlap 校正后 0.86)。关键结论: 幸存者偏差**不是保守方向
     而是逆向运行** —— 被 gate 反复排除的极端名恰好是信号最强处, 抹除它们会让
     假信号"看起来真"。
  2) Gimmer (2026-09-01) point-in-time universe 重建方法论 (crypto 域)。
  3) 叩富网 QMT (2026-09-04) 幸存者偏差规避 (纳入全生命周期标的/动态选股/复盘失效标的)。

定位: 纯诊断 (shadow 默认), **零裁决**, 不改任何门槛/不改生成器。
      输出 data/survivorship_guard_report.json, Ralph Phase E 后调用, 日报风险预警节报读数。

四项子检查
  A. 面板宇宙体检 (静态): 面板区间/标的数、逐年入退场、"非幸存标的"占比、脆弱标的池规模。
  B. 因子级双口径 IC 差分: IC_all / IC_survivor / IC_non_survivor(存续期) →
     依赖度 = |IC_non| / |IC_surv|。依赖度高 ⇒ 表观 alpha 依赖已离场标的 (实盘不可得)。
  C. 脆弱标的剔除敏感性: 剔除 [低价 / 微成交额 / 触板] 后 IC 提升幅度 (uplift)。
     uplift > 阈值 ⇒ 信号靠不可交易标的; uplift 显著为负 ⇒ 信号被脆弱标的侵蚀。
  D. 范式级累积检测: 候选"有效截面覆盖率"(非零信号占比) 中位数 < 0.50 ⇒
     该范式失败可能是"评估环境把它饿死了"(假阴性) 而非真无信号。

开关: env FUXI_SURVIVORSHIP_GUARD = shadow(默认) | off   (enforce 预留: 仅标注不改判)

⚠️ 诚实边界 (下界估计): daily_prices.csv 自 2021-01 起, 不含 2021 年前退市标的;
   且 Tushare daily 对已退市标的只返回其存续期数据。因此本守卫测到的是偏差的**下界**,
   报"未检出"不等于"不存在"。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent          # e:/quant
DATA_DIR = BASE / "data"
PRICE_PATH = DATA_DIR / "raw" / "daily_prices.csv"
OUT_PATH = DATA_DIR / "survivorship_guard_report.json"
AUDIT_CACHE = DATA_DIR / "survivorship_guard_cache.json"

# ── 判据参数 (env 可覆盖, 全部有安全默认) ─────────────────────
DISAPPEAR_GAP_DAYS = int(os.environ.get("FUXI_SURV_GAP_DAYS", "20") or 20)
NON_SURV_MIN_STOCKS = int(os.environ.get("FUXI_SURV_MIN_STOCKS", "6") or 6)
IC_MIN_STOCKS = int(os.environ.get("FUXI_SURV_IC_MIN_STOCKS", "30") or 30)
LOW_POWER_DAYS = int(os.environ.get("FUXI_SURV_LOW_POWER_DAYS", "60") or 60)
FRAGILE_PRICE = float(os.environ.get("FUXI_SURV_FRAGILE_PRICE", "2.0") or 2.0)
FRAGILE_AMT_PCT = float(os.environ.get("FUXI_SURV_FRAGILE_AMT_PCT", "5.0") or 5.0)
FRAGILE_ABS_CHG = float(os.environ.get("FUXI_SURV_FRAGILE_ABS_CHG", "9.5") or 9.5)
# 标的级脆弱判定: 该股脆弱交易日占比 >= 此阈值才算"脆弱标的"
# (2026-09-12 修正: 早期版本用"全历史曾脆弱"并集 → 5738/5774=99% 无意义,
#  因一年内几乎每只股票都会有一两天进成交额最小 5% 或触板)
FRAGILE_DAY_RATIO = float(os.environ.get("FUXI_SURV_FRAGILE_DAY_RATIO", "0.30") or 0.30)
DEP_T_MIN = float(os.environ.get("FUXI_SURV_DEP_T_MIN", "2.0") or 2.0)
DEPENDENCY_RISK = float(os.environ.get("FUXI_SURV_DEP_RISK", "2.0") or 2.0)
DEPENDENCY_MIN_IC = float(os.environ.get("FUXI_SURV_DEP_MIN_IC", "0.02") or 0.02)
UPLIFT_RISK = float(os.environ.get("FUXI_SURV_UPLIFT", "0.30") or 0.30)
COVERAGE_FLOOR = float(os.environ.get("FUXI_SURV_COVERAGE", "0.50") or 0.50)
LOOKBACK_DAYS = int(os.environ.get("FUXI_SURV_LOOKBACK_DAYS", "270") or 270)
MAX_FORMULAS = int(os.environ.get("FUXI_SURV_MAX_FORMULAS", "12") or 12)

_CTX: Dict[str, Any] = {}   # 进程内缓存: {"panel":..., "computer":..., "audit":...}


def get_mode() -> str:
    m = str(os.environ.get("FUXI_SURVIVORSHIP_GUARD", "shadow") or "shadow").strip().lower()
    return m if m in ("off", "shadow", "enforce") else "shadow"


# ═══════════════════════════════════════════════════════════
# 面板读取与宇宙体检 (Check A)
# ═══════════════════════════════════════════════════════════

def _read_price_panel() -> pd.DataFrame:
    """读取全量价格面板 (仅 5 列, float32 降内存)。"""
    cols = ["ts_code", "trade_date", "close", "amount", "pct_chg"]
    df = pd.read_csv(PRICE_PATH, usecols=cols, dtype={"ts_code": str}, low_memory=False)
    td = df["trade_date"]
    # 与 factor_ic_computer 同口径: 纯整数 YYYYMMDD 必须先按 %Y%m%d 解析,
    # 否则 pandas 3.x format='mixed' 会把它当 epoch 纳秒 (静默全错)
    if pd.api.types.is_integer_dtype(td):
        df["trade_date"] = pd.to_datetime(td.astype(str), format="%Y%m%d", errors="coerce")
    else:
        df["trade_date"] = pd.to_datetime(td, format="mixed", errors="coerce")
    df = df.dropna(subset=["trade_date", "close"])
    df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    for c in ("close", "amount", "pct_chg"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df


def panel_audit(df: pd.DataFrame, use_cache: bool = True) -> Dict[str, Any]:
    """Check A: 面板宇宙体检 (全量历史, 揭示长期 attrition 形态)。

    判据: 非幸存标的占比 < 1% ⇒ 面板近乎全幸存 (偏差上界风险高);
          > 3% ⇒ 面板确实含离场标的 (相对健康, 但仍是下界)。
    """
    try:
        if use_cache and AUDIT_CACHE.exists():
            cached = json.load(open(AUDIT_CACHE, encoding="utf-8"))
            _p = [DISAPPEAR_GAP_DAYS, FRAGILE_PRICE, FRAGILE_AMT_PCT,
                  FRAGILE_ABS_CHG, FRAGILE_DAY_RATIO]
            if (cached.get("_price_mtime") == PRICE_PATH.stat().st_mtime
                    and cached.get("_params") == _p):
                return cached
    except Exception:
        pass

    dates = np.sort(df["trade_date"].unique())
    n_dates = len(dates)
    end_ord = n_dates - 1
    ord_of = pd.Series(np.arange(n_dates), index=pd.DatetimeIndex(dates))
    o = ord_of.reindex(df["trade_date"]).to_numpy()

    g = (pd.DataFrame({"ts_code": df["ts_code"].to_numpy(), "_ord": o})
         .groupby("ts_code")["_ord"].agg(["min", "max", "count"]))

    gone = g[g["max"] <= end_ord - DISAPPEAR_GAP_DAYS]
    n_stocks = int(len(g))
    n_gone = int(len(gone))

    def _years(ords):
        return pd.DatetimeIndex(dates[ords]).year

    exit_year = pd.Series(_years(gone["max"].to_numpy())).value_counts().sort_index()
    entry_year = pd.Series(_years(g["min"].to_numpy())).value_counts().sort_index()

    # 脆弱标的池 (逐日: 低价 | 成交额最小分位 | 触板)
    day_rank = df.groupby("trade_date")["amount"].rank(pct=True)
    frag = ((df["close"] < FRAGILE_PRICE)
            | (day_rank <= FRAGILE_AMT_PCT / 100.0)
            | (df["pct_chg"].abs() >= FRAGILE_ABS_CHG)).fillna(False)
    n_obs = int(len(df))
    fragile_obs_ratio = float(frag.mean()) if n_obs else 0.0
    # 标的级脆弱度 = 该股脆弱交易日数 / 该股总交易日数 (修正: 不看"曾脆弱"并集)
    per_stock = frag.groupby(df["ts_code"]).mean()
    frag_stocks = int((per_stock >= FRAGILE_DAY_RATIO).sum())
    frag_day_median = float(per_stock.median()) if len(per_stock) else 0.0
    year = df["trade_date"].dt.year
    # 年内观测级脆弱占比 (比"年内曾脆弱标的数"稳定得多)
    frag_by_year = (pd.DataFrame({"_y": year.to_numpy(), "_f": frag.to_numpy()})
                    .groupby("_y")["_f"].mean())
    univ_by_year = df.assign(_y=year).groupby("_y")["ts_code"].nunique()

    # 与 IC lookback 窗口对齐的窗口内离场标的 (Check B 用)
    win_start_dt = pd.Timestamp(dates[-1]) - pd.Timedelta(days=LOOKBACK_DAYS)
    win_start_ord = int(np.searchsorted(dates, win_start_dt.to_datetime64()))
    gone_in_window = gone[gone["max"] >= win_start_ord]
    in_window = g[g["max"] >= win_start_ord]

    gone_ratio = (n_gone / n_stocks) if n_stocks else 0.0
    ratio_win = (len(gone_in_window) / len(in_window)) if len(in_window) else 0.0

    audit = {
        "panel_start": str(pd.Timestamp(dates[0]).date()),
        "panel_end": str(pd.Timestamp(dates[-1]).date()),
        "n_trading_days": int(n_dates),
        "n_rows": int(len(df)),
        "n_stocks": n_stocks,
        "n_non_survivors_full": n_gone,
        "non_survivor_ratio_full": round(float(gone_ratio), 4),
        "n_non_survivors_in_window": int(len(gone_in_window)),
        "non_survivor_ratio_in_window": round(float(ratio_win), 4),
        "non_survivor_exit_year_dist": {str(k): int(v) for k, v in exit_year.items()},
        "entry_year_dist": {str(k): int(v) for k, v in entry_year.items()},
        "n_fragile_stocks": frag_stocks,
        "fragile_ratio": round(frag_stocks / n_stocks, 4) if n_stocks else 0.0,
        "fragile_obs_ratio": round(fragile_obs_ratio, 4),
        "fragile_day_ratio_median": round(frag_day_median, 4),
        "fragile_day_ratio_threshold": FRAGILE_DAY_RATIO,
        "fragile_obs_ratio_by_year": {str(k): round(float(v), 4)
                                      for k, v in frag_by_year.items()},
        "universe_by_year": {str(k): int(v) for k, v in univ_by_year.items()},
        "mostly_survivor": bool(gone_ratio < 0.01),
        "note": "非幸存 = 最后交易日距面板末 > %d 交易日; 脆弱标的 = 脆弱交易日占比 >= %.2f; "
                "下界估计 (不含 2021 前退市标的)"
                % (DISAPPEAR_GAP_DAYS, FRAGILE_DAY_RATIO),
    }
    try:
        audit["_price_mtime"] = PRICE_PATH.stat().st_mtime
        audit["_params"] = [DISAPPEAR_GAP_DAYS, FRAGILE_PRICE, FRAGILE_AMT_PCT,
                            FRAGILE_ABS_CHG, FRAGILE_DAY_RATIO]
        json.dump(audit, open(AUDIT_CACHE, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    except Exception:
        pass
    return audit


# ═══════════════════════════════════════════════════════════
# IC 工具
# ═══════════════════════════════════════════════════════════

def _ic_stats(sub: pd.DataFrame, min_stocks: int = IC_MIN_STOCKS) -> Dict[str, Any]:
    """逐日 Rank IC (与 FactorICComputer.compute 同口径: 截面 rank 相关)。"""
    if sub is None or len(sub) == 0:
        return {"ic": 0.0, "icir": 0.0, "n_days": 0, "ic_std": 0.0, "t": 0.0}
    ics = []
    for _, grp in sub.groupby("trade_date", sort=True):
        if len(grp) < min_stocks:
            continue
        v = grp["factor_value"].rank().corr(grp["fwd_return"].rank())
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            ics.append(float(v))
    if not ics:
        return {"ic": 0.0, "icir": 0.0, "n_days": 0, "ic_std": 0.0, "t": 0.0}
    a = np.asarray(ics, dtype=float)
    m, s = float(np.nanmean(a)), float(np.nanstd(a))
    n = int(len(ics))
    t = (m / (s / np.sqrt(n))) if s > 0 and n > 1 else 0.0
    return {"ic": round(m, 6), "icir": round(abs(m / s) if s > 0 else 0.0, 4),
            "n_days": n, "ic_std": round(s, 6), "t": round(float(t), 3)}


def _get_computer():
    if "computer" not in _CTX:
        from factor_ic_computer import FactorICComputer
        _CTX["computer"] = FactorICComputer(lookback_days=LOOKBACK_DAYS)
        _CTX["computer"]._load_data()
    return _CTX["computer"]


def attach_computer(comp) -> None:
    """复用外部已加载的 FactorICComputer (避免重复读 ~560MB 面板)。

    ralph_loop 的 _phase_evaluate 已为 S1 建好实例, 透传进来即可。
    注意: 外部实例的 lookback_days 应与本守卫口径一致 (默认 270)。
    """
    if comp is not None:
        _CTX["computer"] = comp


def _get_panel_full():
    if "panel" not in _CTX:
        _CTX["panel"] = _read_price_panel()
    return _CTX["panel"]


# ═══════════════════════════════════════════════════════════
# 因子级检查 (Check B/C/D)
# ═══════════════════════════════════════════════════════════

def factor_check(formula: str, name: str, paradigm: str,
                 non_surv: set, audit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """对单个因子做 B(双口径 IC 差分) + C(脆弱剔除敏感性) + D(覆盖率)。"""
    comp = _get_computer()
    fdf = comp._eval_formula(formula, str(name)[:40])
    if fdf is None or len(fdf) == 0:
        return None

    px = comp._price_df
    need = ["ts_code", "trade_date", "close", "amount", "pct_chg"]
    px = px[[c for c in need if c in px.columns]].drop_duplicates(
        subset=["ts_code", "trade_date"])

    merged = fdf.merge(px, on=["ts_code", "trade_date"], how="inner")
    if len(merged) == 0:
        return None
    merged = merged.sort_values(["ts_code", "trade_date"])
    merged["fwd_return"] = (merged.groupby("ts_code")["close"].shift(-5)
                            / merged["close"] - 1.0)
    merged = merged.dropna(subset=["factor_value", "fwd_return"])
    if len(merged) < IC_MIN_STOCKS * 5:
        return None

    # 脆弱标的标记 (与面板体检同定义, 逐日重算以对齐 lookback 窗口)
    if "amount" in merged.columns:
        r = merged.groupby("trade_date")["amount"].rank(pct=True)
        frag = (r <= FRAGILE_AMT_PCT / 100.0)
    else:
        frag = pd.Series(False, index=merged.index)
    frag = frag | (merged["close"] < FRAGILE_PRICE)
    if "pct_chg" in merged.columns:
        frag = frag | (merged["pct_chg"].abs() >= FRAGILE_ABS_CHG)
    frag = frag.fillna(False)

    all_s = _ic_stats(merged, IC_MIN_STOCKS)
    codes = merged["ts_code"]
    surv_s = _ic_stats(merged[~codes.isin(non_surv)], IC_MIN_STOCKS)
    non_s = _ic_stats(merged[codes.isin(non_surv)], NON_SURV_MIN_STOCKS)
    clean_s = _ic_stats(merged[~frag], IC_MIN_STOCKS)

    dep = abs(non_s["ic"]) / max(abs(surv_s["ic"]), 1e-9)
    uplift = ((abs(clean_s["ic"]) - abs(all_s["ic"]))
              / max(abs(all_s["ic"]), 1e-9))
    # 依赖度风险: 依赖度超阈 + 非幸存 IC 有量级 + 样本足够 + t 显著
    # (加 t 判据防止 n 小 IC 大的噪声被误报)
    dep_risk = bool(dep > DEPENDENCY_RISK
                    and abs(non_s["ic"]) >= DEPENDENCY_MIN_IC
                    and non_s["n_days"] >= 20
                    and abs(non_s.get("t", 0.0)) >= DEP_T_MIN)
    fragile_dep = bool(uplift > UPLIFT_RISK)
    fragile_eroded = bool(uplift < -UPLIFT_RISK)

    # Check D: 有效截面覆盖率 (非零信号占比中位数)
    coverage = None
    try:
        nz = (merged.assign(_nz=merged["factor_value"].abs() > 1e-12)
              .groupby("trade_date")["_nz"].sum())
        nu = merged.groupby("trade_date")["factor_value"].size()
        cov = (nz / nu.replace(0, np.nan)).dropna()
        if len(cov):
            coverage = float(cov.median())
    except Exception:
        coverage = None
    narrow = bool(coverage is not None and coverage < COVERAGE_FLOOR)

    return {
        "factor_name": str(name)[:60],
        "paradigm": str(paradigm or ""),
        "ic_all": all_s,
        "ic_survivor": surv_s,
        "ic_non_survivor": non_s,
        "ic_fragile_excluded": clean_s,
        "non_survivor_dependency": round(float(dep), 3),
        "fragile_uplift": round(float(uplift), 3),
        "effective_coverage": round(coverage, 4) if coverage is not None else None,
        "low_power": bool(non_s["n_days"] < LOW_POWER_DAYS),
        "dependency_risk": dep_risk,
        "fragile_dependent": fragile_dep,
        "fragile_eroded": fragile_eroded,
        "narrow_coverage": narrow,
        "survivorship_risk": bool(dep_risk or fragile_dep),
    }


# ═══════════════════════════════════════════════════════════
# 主入口 (Ralph Phase E 钩子)
# ═══════════════════════════════════════════════════════════

def _pick_formulas(candidates: List[Dict], results: List[Any]) -> List[Dict[str, str]]:
    """优先取 S5/JQ 候选; 无则退到 S1 通过; 再退到全部 (上限 MAX_FORMULAS)。"""
    picked: List[Dict[str, str]] = []
    tiers = ("jq", "s1", "all")
    for tier in tiers:
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
    """Ralph Phase E 钩子: 对候选做幸存者偏差影子诊断 (零裁决)。"""
    mode = get_mode()
    if mode == "off":
        return {"mode": "off", "n_checked": 0, "n_survivorship_risk": 0, "rows": []}

    audit: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    try:
        full = _get_panel_full()
        audit = panel_audit(full)
        # 非幸存集合: 全量面板派生的离场标的 (与 lookback 窗口交集由 IC 侧自然处理)
        dates = np.sort(full["trade_date"].unique())
        end_ord = len(dates) - 1
        om = pd.Series(np.arange(len(dates)), index=pd.DatetimeIndex(dates))
        o = om.reindex(full["trade_date"]).to_numpy()
        gg = (pd.DataFrame({"ts_code": full["ts_code"].to_numpy(), "_ord": o})
              .groupby("ts_code")["_ord"].max())
        non_surv = set(gg[gg <= end_ord - DISAPPEAR_GAP_DAYS].index)

        for p in _pick_formulas(candidates, results):
            try:
                row = factor_check(p["formula"], p["name"], p["paradigm"],
                                   non_surv, audit)
                if row:
                    rows.append(row)
            except Exception as e:
                if verbose:
                    print(f"  [SurvGuard] {str(p['name'])[:30]} 检查失败 (跳过): "
                          f"{type(e).__name__}: {str(e)[:80]}")
    except Exception as e:
        print(f"  [SurvGuard] ⚠️ 面板/宇宙体检失败 (非阻塞): {type(e).__name__}: {e}")

    # Check D 范式级聚合
    para: Dict[str, List[float]] = {}
    for r in rows:
        cv = r.get("effective_coverage")
        if cv is not None:
            para.setdefault(r["paradigm"] or "(未标注)", []).append(cv)
    para_summary = [
        {"paradigm": k, "n": len(v), "median_coverage": round(float(np.median(v)), 4),
         "starved": bool(float(np.median(v)) < COVERAGE_FLOOR)}
        for k, v in sorted(para.items(), key=lambda x: -len(x[1]))
    ]

    n_risk = sum(1 for r in rows if r["survivorship_risk"])
    n_frag = sum(1 for r in rows if r["fragile_dependent"])
    n_starved = sum(1 for p in para_summary if p["starved"])

    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "params": {
            "disappear_gap_days": DISAPPEAR_GAP_DAYS,
            "fragile_price": FRAGILE_PRICE,
            "fragile_amt_pct": FRAGILE_AMT_PCT,
            "fragile_abs_chg": FRAGILE_ABS_CHG,
            "dependency_risk": DEPENDENCY_RISK,
            "uplift_risk": UPLIFT_RISK,
            "coverage_floor": COVERAGE_FLOOR,
            "lookback_days": LOOKBACK_DAYS,
        },
        "panel_audit": {k: v for k, v in audit.items() if not k.startswith("_")},
        "n_checked": len(rows),
        "n_survivorship_risk": n_risk,
        "n_fragile_dependent": n_frag,
        "n_starved_paradigms": n_starved,
        "paradigm_summary": para_summary,
        "rows": rows,
        "boundary": "下界估计: 面板自 2021-01 起, 不含更早退市标的; 未检出 ≠ 不存在",
    }
    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        json.dump(payload, open(OUT_PATH, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"  [SurvGuard] ⚠️ 影子落盘失败 (非阻塞): {e}")

    if verbose:
        a = audit or {}
        print(f"  [SurvGuard] {mode} 模式: 面板 {a.get('panel_start','?')}~{a.get('panel_end','?')} "
              f"/ {a.get('n_stocks','?')} 只 / 非幸存 {a.get('n_non_survivors_full','?')} "
              f"({float(a.get('non_survivor_ratio_full', 0) or 0):.2%}) | "
              f"脆弱池 {a.get('n_fragile_stocks','?')} "
              f"({float(a.get('fragile_ratio', 0) or 0):.2%})")
        print(f"  [SurvGuard] 检查 {len(rows)} 因子 | 幸存依赖风险 {n_risk} | "
              f"脆弱依赖 {n_frag} | 覆盖率不足范式 {n_starved} (零裁决)")
        for r in rows[:3]:
            print(f"      - {str(r['factor_name'])[:30]}: IC_all {r['ic_all']['ic']:+.4f} "
                  f"/ IC_surv {r['ic_survivor']['ic']:+.4f} "
                  f"/ 依赖度 {r['non_survivor_dependency']:.2f} "
                  f"/ 脆弱uplift {r['fragile_uplift']:+.1%} "
                  f"/ 覆盖 {r['effective_coverage']}")
    return payload


# ═══════════════════════════════════════════════════════════
# CLI (独立运行 + 历史回放验证)
# ═══════════════════════════════════════════════════════════

def _replay_formulas(limit: int = 12) -> List[Dict[str, str]]:
    """回放样本: 从 stage1 因子提案 (多样范式) + 冠军因子池各取若干。"""
    out: List[Dict[str, str]] = []
    p = DATA_DIR / "stage1_factor_proposals.json"
    if p.exists():
        try:
            d = json.load(open(p, encoding="utf-8"))
            arr = d if isinstance(d, list) else d.get("proposals", d.get("factors", []))
            seen_para: Dict[str, int] = {}
            for it in arr:
                f = it.get("formula") or it.get("expression") or ""
                if not f:
                    continue
                pa = str(it.get("paradigm", "") or "(未标注)")
                if seen_para.get(pa, 0) >= 2:
                    continue
                seen_para[pa] = seen_para.get(pa, 0) + 1
                out.append({"formula": f,
                            "name": it.get("factor_name", it.get("name", f"prop_{len(out)}")),
                            "paradigm": pa})
                if len(out) >= limit:
                    break
        except Exception as e:
            print(f"  [SurvGuard] stage1 提案读取失败: {e}")
    if len(out) < limit:
        try:
            from seed_injector import CHAMPION_FACTORS
            for f_ in CHAMPION_FACTORS:
                if len(out) >= limit:
                    break
                form = f_.get("expression") or f_.get("formula") or ""
                if form:
                    out.append({"formula": form,
                                "name": f_.get("factor_name", f_.get("name", "champ")),
                                "paradigm": str(f_.get("paradigm", "") or "(champion)")})
        except Exception:
            pass
    return out[:limit]


def _cli() -> int:
    ap = argparse.ArgumentParser(description="P-20260912-002 幸存者偏差守卫影子")
    ap.add_argument("--replay", action="store_true",
                    help="历史回放: 对现有宇宙做一次诊断, 验证读数非平凡")
    ap.add_argument("--formulas-file", default="",
                    help="JSON 文件: [{'formula','name','paradigm'}, ...]")
    ap.add_argument("--limit", type=int, default=MAX_FORMULAS)
    args = ap.parse_args()

    if args.formulas_file:
        try:
            items = json.load(open(args.formulas_file, encoding="utf-8"))
        except Exception as e:
            print(f"读取 formulas-file 失败: {e}")
            return 1
        cands = [{"factor_name": it.get("name", f"f{i}"),
                  "formula": it.get("formula", it.get("expression", "")),
                  "paradigm": it.get("paradigm", "")}
                 for i, it in enumerate(items)]
        payload = evaluate(cands, [], verbose=True)
    else:
        forms = _replay_formulas(args.limit)
        if not forms:
            print("无可用回放公式 (stage1_factor_proposals.json / CHAMPION_FACTORS 均为空)")
            return 1
        print(f"  [SurvGuard] 回放 {len(forms)} 条公式 ...")
        cands = [{"factor_name": p["name"], "formula": p["formula"],
                  "paradigm": p["paradigm"]} for p in forms]
        payload = evaluate(cands, [], verbose=True)

    # 非平凡读数自检 (proposal 要求: 回放必须证明不是 always-safe 摆设)
    rows = payload.get("rows", [])
    nontrivial = [r for r in rows
                  if r["survivorship_risk"] or r["fragile_dependent"]
                  or r["narrow_coverage"]
                  or abs(r["fragile_uplift"]) > 0.10
                  or r["non_survivor_dependency"] > 1.2]
    print(f"  [SurvGuard] 非平凡读数: {len(nontrivial)}/{len(rows)} 条 "
          f"⇒ {'✅ 通过 (非摆设)' if nontrivial else '⚠️ 全绿 (需校准阈值)'}")
    print(f"  [SurvGuard] 报告 → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
