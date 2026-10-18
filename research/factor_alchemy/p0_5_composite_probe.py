# -*- coding: utf-8 -*-
"""
P0.5 复合因子探测 — 日频/双日频立项判决 (决定性实验)
========================================================
P0 结论: 单因子短端 IC 全在盈亏平衡线边缘 → 复合是唯一机会。
本实验: 5 个 MID 因子 (overnight_raw / overnight5 / intraday_rev /
        lhb_freq_20d / overreaction) 等权 rank 和复合 (打散 v1 形态),
        回答三个问题:
        ① 复合后 IC(2d) 能否跨过成本线?
        ② 信号是否分半段稳健 (非单一 regime 假象)?
        ③ top50 组合日换手多少 (成本是否可控)?

严格口径 (与 P0 的关键差异):
  信号 t 日收盘定稿 → t+1 开盘买入 → t+1+h 开盘卖出
  fwd_strict(h) = open(t+1+h) / open(t+1) - 1
  对比 P0 宽松口径 close(t+h)/close(t)-1 (含 t收盘→t+1开盘 的隔夜跳跃,
  该段信号不可捕获, 对隔夜动量类因子显著高估)。

判决标准 (联立):
  IC_strict(2d) ≥ 0.05 且 t≥2 且半段均正 且 日换手 ≤ 8%   → PASS 立项 P1
  IC_strict(2d) ∈ [0.03, 0.05)                            → MARGINAL
  其余                                                   → 证伪日频
"""
import sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

from factor_ic_computer import FactorICComputer

HORIZONS = [1, 2, 3, 5, 10, 20]
MIN_STOCKS = 30
LOOKBACK_DAYS = 270
TOP_N = 50               # 换手模拟 top50 (对齐 JQ 池风格)

FACTORS = {
    "overnight_raw": "open / close.shift(1) - 1.0",    # 隔夜动量 (2d峰值, t3.2)
    "overnight5": "(open / close.shift(1) - 1.0).rolling(5, min_periods=3).sum()",
    "intraday_rev": "-(close / open - 1.0)",           # 日内反转 (真快衰减)
    "lhb_freq_20d": "-lhb_flag.rolling(20).sum()",     # 龙虎榜事件 (t4.1 最强)
    "overreaction": "-close.pct_change(1).rolling(20).max()",  # 反转 (短端最强)
}


def eval_formula(formula, env):
    lines = [ln.strip() for ln in formula.strip().split("\n") if ln.strip()]
    body, final = lines[:-1], lines[-1]
    for ln in body:
        exec(ln, env)
    return eval(final, env)


def row_rank_pct(wide):
    """宽表逐行 (截面) rank pct: 每行 rankdata / 有效数, NaN 保留 (铁律: axis=1 语义)"""
    arr = wide.to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan)
    for i in range(arr.shape[0]):
        row = arr[i]
        m = np.isfinite(row)
        n = m.sum()
        if n < MIN_STOCKS:
            continue
        out[i, m] = rankdata(row[m]) / n
    return pd.DataFrame(out, index=wide.index, columns=wide.columns)


def ic_series(factor_wide, fwd_wide):
    """逐日截面 Spearman IC (只保留因子与前向收益均有效的截面)"""
    ics = []
    for i in range(len(factor_wide.index)):
        fv = factor_wide.iloc[i].to_numpy()
        rv = fwd_wide.iloc[i].to_numpy()
        m = np.isfinite(fv) & np.isfinite(rv)
        if m.sum() < MIN_STOCKS:
            continue
        ic = spearmanr(fv[m], rv[m])[0]
        if np.isfinite(ic):
            ics.append(ic)
    return np.array(ics)


def turnover_sim(comp_wide, top_n=TOP_N):
    """top_n 名单日变化 → 单边日换手估计 (等权, 卖出比例)"""
    dates = comp_wide.index
    prev = None
    tos = []
    for i in range(len(dates)):
        row = comp_wide.iloc[i].dropna()
        if len(row) < top_n:
            continue
        cur = set(row.nlargest(top_n).index)
        if prev is not None:
            sold = len(prev - cur) / top_n     # 今天跌出名单 → 昨天持有今天卖出
            tos.append(sold)
        prev = cur
    return np.array(tos) if tos else np.array([np.nan])


def main():
    t0 = time.time()
    print("=" * 78)
    print("P0.5 复合因子探测 — 日频/双日频立项判决")
    print("=" * 78)

    # ── 数据层 ──
    print(f"\n[1/5] 加载数据层 (lookback={LOOKBACK_DAYS}d)...")
    comp = FactorICComputer(lookback_days=LOOKBACK_DAYS)
    comp._load_data()
    print(f"    done in {time.time()-t0:.1f}s")

    # ── 宽表 ──
    print("[2/5] 构造宽表 (open/close/volume/lhb_flag)...")
    close_w = comp._get_wide_field("close")
    open_w = comp._get_wide_field("open")
    volume_w = comp._get_wide_field("volume")
    lhb_w = comp._get_wide_field("lhb_flag")
    dates = close_w.index
    stocks = close_w.columns
    print(f"    {len(dates)} 交易日 × {len(stocks)} 只")

    # ── 严格口径前向收益: fwd(h) = open(t+1+h)/open(t+1) - 1 ──
    open_s1 = open_w.shift(-1)                       # open(t+1)
    fwd_strict = {}
    for h in HORIZONS:
        fwd_strict[h] = open_w.shift(-(h + 1)) / open_s1 - 1.0
    # 宽松口径 (P0 同款, 对照)
    fwd_loose = {h: close_w.shift(-h) / close_w - 1.0 for h in HORIZONS}

    # ── 5 因子宽表 + rank pct ──
    print("[3/5] 5 因子求值 + 截面 rank pct...")
    env = {"np": np, "pd": pd, "log": np.log, "close": close_w, "open": open_w,
           "volume": volume_w, "lhb_flag": lhb_w}
    rp = {}
    for name, formula in FACTORS.items():
        fw = eval_formula(formula, dict(env))
        fw = fw.reindex(index=dates, columns=stocks).replace([np.inf, -np.inf], np.nan)
        rp[name] = row_rank_pct(fw)
        print(f"    {name:<18} valid_days={int(rp[name].notna().any(axis=1).sum())}")

    # ── 因子间相关性 (日截面 spearman 均值) ──
    print("[4/5] 因子相关性矩阵 + 复合 rank 和...")
    names = list(FACTORS.keys())
    print("    " + " ".join(f"{n[:8]:>10}" for n in names))
    for a in names:
        row = []
        for b in names:
            if a == b:
                row.append(1.0)
                continue
            ic = ic_series(rp[a], rp[b])
            row.append(float(ic.mean()) if len(ic) else np.nan)
        print(f"    {a:<10}" + " ".join(f"{v:>10.3f}" for v in row))

    # 复合 = 等权 rank 和 (v1 形态)
    comp_rp = pd.concat([rp[n] for n in names]).groupby(level=0).mean()
    comp_rp = comp_rp.reindex(index=dates, columns=stocks)

    # ── 复合 IC: 严格 + 宽松 ──
    print("[5/5] 复合 IC 衰减 (严格 open-to-open vs 宽松 close-to-close)...")
    out = {"strict": {}, "loose": {}}
    print(f"\n    {'持有期':>6} | {'严格IC':>9} {'t值':>7} {'半段1':>8} {'半段2':>8} | {'宽松IC':>9}")
    print("    " + "-" * 66)
    half = len(dates) // 2
    strict_by_h = {}
    for h in HORIZONS:
        s = ic_series(comp_rp, fwd_strict[h])
        l = ic_series(comp_rp, fwd_loose[h])
        out["strict"][str(h)] = {"ic": round(float(s.mean()), 5), "n": int(len(s))}
        out["loose"][str(h)] = {"ic": round(float(l.mean()), 5), "n": int(len(l))}
        strict_by_h[h] = float(s.mean())
        if h in (1, 2, 3, 5):
            s1 = ic_series(comp_rp.iloc[:half], fwd_strict[h].iloc[:half])
            s2 = ic_series(comp_rp.iloc[half:], fwd_strict[h].iloc[half:])
            tval = s.mean() / s.std() * np.sqrt(len(s)) if len(s) > 5 and s.std() > 1e-9 else 0
            print(f"    {h:>5}d | {s.mean():>+9.4f} {tval:>7.1f} "
                  f"{(s1.mean() if len(s1) else np.nan):>+8.4f} "
                  f"{(s2.mean() if len(s2) else np.nan):>+8.4f} | {l.mean():>+9.4f}")
        else:
            print(f"    {h:>5}d | {s.mean():>+9.4f} {'':>7} {'':>8} {'':>8} | {l.mean():>+9.4f}")

    # ── 换手模拟 ──
    print("\n    换手模拟 (top%d, 严格口径下的名单日变化):" % TOP_N)
    tos = turnover_sim(comp_rp, TOP_N)
    tos = tos[np.isfinite(tos)]
    if len(tos):
        print(f"      单边日换手: 均值 {tos.mean()*100:.1f}% | 中位 {np.median(tos)*100:.1f}% "
              f"| 90分位 {np.percentile(tos, 90)*100:.1f}%")

        # ── 成本后判决 ──
        print("\n    💰 成本后净 alpha 核算 (双日频 = 年 120 次调仓):")
        ic2 = strict_by_h.get(2, 0.0)
        sigma_act = 0.10  # 主动风险 10% 口径
        gross = abs(ic2) * np.sqrt(120) * sigma_act
        for h_day in (0.05, tos.mean()):
            cost = 120 * h_day * 0.0072
            net = gross - cost
            print(f"      日换手 {h_day*100:.1f}% → 年成本 {cost*100:.1f}% | "
                  f"毛 alpha {gross*100:.1f}% | 净 alpha {net*100:+.1f}%")
        ic2_half1 = ic_series(comp_rp.iloc[:half], fwd_strict[2].iloc[:half])
        ic2_half2 = ic_series(comp_rp.iloc[half:], fwd_strict[2].iloc[half:])
        h1 = float(ic2_half1.mean()) if len(ic2_half1) else 0.0
        h2 = float(ic2_half2.mean()) if len(ic2_half2) else 0.0
        mean_to = float(tos.mean())
        if abs(ic2) >= 0.05 and h1 > 0 and h2 > 0 and mean_to <= 0.08:
            verdict = "PASS — 立项 P1 (JQ 双日频骨架)"
        elif abs(ic2) >= 0.03:
            verdict = "MARGINAL — 谨慎立项, 先查换手与信号衰减来源"
        else:
            verdict = "FAIL — 日频路线证伪, 资源投新 alpha 源"
        print(f"\n    📋 判决: {verdict}  (IC_strict(2d)={ic2:+.4f}, "
              f"半段1={h1:+.4f}, 半段2={h2:+.4f}, 日换手={mean_to*100:.1f}%)")

    # ── 落盘 ──
    res = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lookback_days": LOOKBACK_DAYS, "top_n": TOP_N,
        "n_dates": int(len(dates)), "n_stocks": int(len(stocks)),
        "ic": out,
        "turnover": {"mean": round(float(tos.mean()), 4), "median": round(float(np.median(tos)), 4),
                     "p90": round(float(np.percentile(tos, 90)), 4)} if len(tos) else {},
    }
    out_path = BASE.parent.parent / "data" / "p0_5_composite_probe.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存: {out_path}")
    print(f"⏱️  总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
