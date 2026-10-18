# -*- coding: utf-8 -*-
"""
P0 持有期 IC 衰减诊断 — 日频/双日频可行性判决
================================================
目的: 判断现有 JQ 验证因子 + 短持有候选的 alpha 是快衰减还是慢积累,
      为"是否立项日频策略"提供判决依据。

方法:
  1. FactorICComputer 全市场向量化数据层 (lookback_days=270, 口径对齐 S5 冠军验证)
  2. 13 因子宽表公式 (10 个 JQ 冠军 1:1 翻译自 jq_scatter_v3 + 3 个短持有候选)
  3. 多持有期 h ∈ {1,2,3,5,10,20} 前向收益截面 Rank IC (Spearman)
  4. 半衰期 (指数拟合 |IC(h)| = a·exp(-λh)) + 形态分类
  5. 成本盈亏平衡分析 (Grinold IR + 日频双边成本 0.72%)

判决标准:
  FAST       : 半衰期 ≤ 2 天 且 |IC(1d)| t≥2        → 日频候选 (信息 1-2 天内消化)
  ACCUMULATE : |IC(5d)| ≥ 3×|IC(1d)| (同号)        → 慢积累, 日频只增成本零新信息
  MID        : 其余                                 → 中间型, 双日频候选
  WEAK       : |IC(1d)| t < 1.5                     → 无短持有信号

口径铁律遵循: 宽表滚动沿时间轴 (无前视); 停牌 NaN 自动剔除;
  log/divide 分母 .where(x>0) 转 NaN (与 JQ fill_paused 语义一致)。
"""
import sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

from factor_ic_computer import FactorICComputer

HORIZONS = [1, 2, 3, 5, 10, 20]
MIN_STOCKS = 30          # 每截面最少有效股票数
LOOKBACK_DAYS = 270      # 口径对齐 S5 冠军验证窗口 (≈185 交易日)

FACTORS = {
    # ── A. 10 个 JQ 验证冠军 (jq_scatter_v3_10factor_voltarget 宽表公式 1:1 翻译) ──
    "dollar20": (
        "dv = (close * volume).rolling(20, min_periods=10).mean()\n"
        "dv = dv.where(dv > 0)\n"
        "-np.log(dv)"
    ),
    "overnight5": "(open / close.shift(1) - 1.0).rolling(5, min_periods=3).sum()",
    "tstd_cv": (
        "den = volume.rolling(20, min_periods=10).mean()\n"
        "den = den.where(den > 0)\n"
        "-(volume.rolling(20, min_periods=10).std() / den)"
    ),
    "tvma20": (
        "num = close * volume\n"
        "den = num.rolling(20, min_periods=10).mean()\n"
        "den = den.where(den > 0)\n"
        "-num / den"
    ),
    "fa_turnover_std": (
        "den = volume.rolling(20, min_periods=10).mean()\n"
        "den = den.where(den > 0)\n"
        "ratio = volume / den\n"
        "-ratio.rolling(20, min_periods=10).std()"
    ),
    "fa_vroc_12": "volume.pct_change(12)",
    "gp002_pricevol": "high - close.pct_change(20).shift(20)",
    "gp_micro_7db3": "(close - low) / (close * volume - open + 0.001)",
    "overreaction": "-close.pct_change(1).rolling(20).max()",
    "gp_evo_003": "-volume.rolling(120).corr(close).pct_change(10)",
    # ── B. 短持有候选 (日频方向新假设) ──
    "overnight_raw": "open / close.shift(1) - 1.0",   # 单日隔夜收益 (最快速信号)
    "intraday_rev": "-(close / open - 1.0)",          # 日内反转 (T+0 日内收益负向)
    "lhb_freq_20d": "-lhb_flag.rolling(20).sum()",    # 龙虎榜事件 (JQ MARGINAL +64.59%)
}


def eval_formula(formula: str, env: dict):
    """多行公式: 中间行 exec 进命名空间, 最后一行 eval (与 _eval_formula 同语义)"""
    lines = [ln.strip() for ln in formula.strip().split("\n") if ln.strip()]
    body, final = lines[:-1], lines[-1]
    for ln in body:
        exec(ln, env)
    return eval(final, env)


def half_life(ic_by_h: dict) -> float:
    """指数拟合 |IC(h)| = a·exp(-λh) → 半衰期 = ln2/λ; λ≤0 返回 inf (不衰减)"""
    hs, vals = [], []
    for h in HORIZONS:
        v = ic_by_h.get(h)
        if v is not None and abs(v) > 1e-6:
            hs.append(float(h))
            vals.append(np.log(abs(v)))
    if len(hs) < 3:
        return float("inf")
    # 线性回归 ln|IC| = a - λh
    slope, intercept = np.polyfit(hs, vals, 1)
    lam = -slope
    if lam <= 1e-4:
        return float("inf")
    return float(np.log(2.0) / lam)


def main():
    t0 = time.time()
    print("=" * 78)
    print("P0 持有期 IC 衰减诊断 — 日频/双日频可行性判决")
    print("=" * 78)

    # ── 1. 数据层 (全市场向量化) ──
    print(f"\n[1/4] 加载数据层 (lookback={LOOKBACK_DAYS}d, 全市场)...")
    comp = FactorICComputer(lookback_days=LOOKBACK_DAYS)
    comp._load_data()
    print(f"    done in {time.time()-t0:.1f}s | {len(comp._price_df)} rows")

    # ── 2. 宽表字段 + 前向收益 ──
    print("[2/4] 构造宽表字段 + 6 持有期前向收益...")
    fields = {}
    for f in ["close", "open", "high", "low", "volume", "lhb_flag"]:
        w = comp._get_wide_field(f)
        if w is None:
            print(f"    ⚠️ 字段 {f} 宽表为空, 相关因子将失败")
        fields[f] = w

    close_w = fields["close"]
    dates = close_w.index
    stocks = close_w.columns
    print(f"    宽表: {len(dates)} 交易日 × {len(stocks)} 只")

    fwd = {}
    for h in HORIZONS:
        fwd[h] = (close_w.shift(-h) / close_w - 1.0)

    # ── 3. 逐因子 × 逐持有期 IC ──
    print("[3/4] 计算 13 因子 × 6 持有期截面 Rank IC...")
    env_base = {
        "np": np, "pd": pd, "log": np.log,
        "close": close_w, "open": fields["open"], "high": fields["high"],
        "low": fields["low"], "volume": fields["volume"],
        "lhb_flag": fields["lhb_flag"],
    }
    results = {}
    for fname, formula in FACTORS.items():
        t_f = time.time()
        env = dict(env_base)
        try:
            fw = eval_formula(formula, env)
        except Exception as e:
            print(f"    ❌ {fname}: eval 失败 {type(e).__name__}: {str(e)[:90]}")
            results[fname] = {"error": str(e)[:200]}
            continue
        if not isinstance(fw, pd.DataFrame):
            print(f"    ❌ {fname}: 返回非 DataFrame")
            results[fname] = {"error": "non-DataFrame"}
            continue
        fw = fw.reindex(index=dates, columns=stocks)
        fw = fw.replace([np.inf, -np.inf], np.nan)

        ic_by_h = {}
        detail = {}
        for h in HORIZONS:
            ics = []
            fr = fwd[h]
            for i in range(len(dates)):
                fv = fw.iloc[i].to_numpy()
                rv = fr.iloc[i].to_numpy()
                m = np.isfinite(fv) & np.isfinite(rv)
                if m.sum() < MIN_STOCKS:
                    continue
                ic = spearmanr(fv[m], rv[m])[0]
                if np.isfinite(ic):
                    ics.append(ic)
            arr = np.array(ics)
            if len(arr) < 10:
                ic_by_h[h] = None
                detail[h] = {"ic": None, "icir": None, "t": None, "n": len(arr)}
                continue
            mean = float(arr.mean())
            std = float(arr.std())
            ic_by_h[h] = mean
            detail[h] = {
                "ic": round(mean, 5),
                "icir": round(abs(mean / std), 3) if std > 1e-9 else None,
                "t": round(mean / std * np.sqrt(len(arr)), 2) if std > 1e-9 else None,
                "n": int(len(arr)),
            }
        hl = half_life(ic_by_h)
        results[fname] = {"ic_by_h": ic_by_h, "detail": detail, "half_life": hl}
        print(f"    {fname:<18} hl={hl:>7}  IC1={ic_by_h.get(1)}")
        del fw

    # ── 4. 形态分类 + 输出 ──
    print("\n[4/4] 形态分类与判决...")
    rows = []
    for fname, r in results.items():
        if "error" in r:
            continue
        d = r["detail"]
        ic1 = r["ic_by_h"].get(1) or 0.0
        ic5 = r["ic_by_h"].get(5) or 0.0
        t1 = (d[1]["t"] if d[1]["t"] is not None else 0.0)
        hl = r["half_life"]
        same_sign = (ic1 != 0) and (ic5 * ic1 > 0)
        ratio5 = abs(ic5 / ic1) if ic1 != 0 else float("inf")
        if abs(t1) < 1.5:
            cls = "WEAK"
        elif hl <= 2.0 and abs(t1) >= 2.0:
            cls = "FAST"
        elif same_sign and ratio5 >= 3.0:
            cls = "ACCUMULATE"
        else:
            cls = "MID"
        r["class"] = cls
        r["ratio_5_1"] = round(ratio5, 2)
        rows.append((fname, cls, hl, r["ratio_5_1"],
                     d[1]["ic"], d[1]["t"], d[2]["ic"], d[5]["ic"], d[20]["ic"]))

    # ── 输出表 ──
    print("\n" + "=" * 110)
    print(f"{'因子':<18}{'形态':<12}{'半衰期(d)':>10}{'IC5/IC1':>8}  "
          f"{'IC(1d)':>9}{'t(1d)':>7}{'IC(2d)':>9}{'IC(5d)':>9}{'IC(20d)':>9}")
    print("-" * 110)
    for (fname, cls, hl, r51, i1, t1, i2, i5, i20) in sorted(rows, key=lambda x: x[1]):
        hl_s = f"{hl:.1f}" if hl < 100 else "inf"
        fmt = lambda v: f"{v:+.4f}" if v is not None else "   nan"
        print(f"{fname:<18}{cls:<12}{hl_s:>10}{r51:>8.2f}  "
              f"{fmt(i1):>9}{t1:>7.1f}{fmt(i2):>9}{fmt(i5):>9}{fmt(i20):>9}")
    print("=" * 110)

    # ── 判决摘要 ──
    fast = [r[0] for r in rows if r[1] == "FAST"]
    acc = [r[0] for r in rows if r[1] == "ACCUMULATE"]
    mid = [r[0] for r in rows if r[1] == "MID"]
    weak = [r[0] for r in rows if r[1] == "WEAK"]
    print(f"\n📋 判决摘要:")
    print(f"  FAST (日频候选)       : {fast if fast else '无'}")
    print(f"  MID  (双日频候选)     : {mid if mid else '无'}")
    print(f"  ACCUMULATE (慢积累)   : {acc if acc else '无'}")
    print(f"  WEAK (无短持有信号)   : {weak if weak else '无'}")

    # ── 成本盈亏平衡 ──
    print("\n💰 日频成本盈亏平衡 (Grinold 近似):")
    print("  双边成本/次全额换手 = 佣金0.01%×2 + 印花税0.05% + 滑点0.3%×2 ≈ 0.72%")
    for h in (0.05, 0.10):
        c = 240 * h * 0.0072
        ir_ic1 = np.sqrt(240)
        print(f"  日换手 {h*100:.0f}% → 年成本 {c*100:.1f}%; IR = IC(1d)×{ir_ic1:.1f}; "
              f"盈亏平衡 IC(1d) ≈ {c / (ir_ic1 * 0.10):.4f} (按主动风险 10%)")

    out = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lookback_days": LOOKBACK_DAYS,
        "horizons": HORIZONS,
        "n_dates": int(len(dates)),
        "n_stocks": int(len(stocks)),
        "results": {k: {kk: (vv if kk != "ic_by_h" else {str(k2): v2 for k2, v2 in vv.items()})
                        for kk, vv in v.items()} for k, v in results.items()},
        "classes": {r[0]: r[1] for r in rows},
    }
    out_path = Path(BASE).parent.parent / "data" / "p0_holding_period_ic_decay.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=lambda o: float(o))
    print(f"\n✅ 结果已保存: {out_path}")
    print(f"⏱️  总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
