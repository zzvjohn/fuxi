# -*- coding: utf-8 -*-
"""快刀季频池 regime 叠加 —— 严格可交易版 (lag=1)
关键修正: 第 i 日的仓位必须只依赖第 i-1 日收盘可得信息.
同时输出 lag0(前视) 对照, 量化前视虚增量.
成本: 每次 |Δw| 计 15bp (佣金+印花+滑点, A股单边约0.1~0.15%)
"""
import csv, math

NAV = r"e:\quant\data\kuaidao_test\full_backtest\timing3\nav_variants3.csv"
IDX = r"e:\quant\data\raw\index_000985_daily.csv"
COST_PER_UNIT = 0.0015


def load_nav(p):
    dates, B = [], []
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            dates.append(r["date"])
            B.append(float(r["B_quarter_bh"]))
    return dates, B


def load_idx(p):
    d = {}
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d[r["trade_date"]] = float(r["close"])
    return d


def stats(nav):
    r = [nav[i + 1] / nav[i] - 1 for i in range(len(nav) - 1)]
    n = len(r)
    m = sum(r) / n
    sd = (sum((v - m) ** 2 for v in r) / (n - 1)) ** 0.5
    sharpe = (m - 0.02 / 250) / sd * math.sqrt(250) if sd > 0 else 0.0
    peak, mdd = nav[0], 0.0
    for v in nav:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    return nav[-1] - 1, sharpe, mdd, m * 250


def simulate(B, w, cost=0.0):
    nav = [1.0]
    for i in range(1, len(B)):
        ret = w[i] * (B[i] / B[i - 1] - 1)
        c = cost * abs(w[i] - w[i - 1]) if cost else 0.0
        nav.append(nav[-1] * (1 + ret - c))
    return nav


def ma(px, i, n):
    if i - n + 1 < 0:
        return None
    seg = px[i - n + 1:i + 1]
    if any(v is None for v in seg):
        return None
    return sum(seg) / n


def main():
    dates, B = load_nav(NAV)
    idx = load_idx(IDX)
    px = [idx.get(d) for d in dates]
    n = len(dates)
    b_tot, b_sh, b_mdd, _ = stats(B)
    print(f"[nav] {dates[0]} -> {dates[-1]} rows={n}")
    print(f"[BASE B_quarter_bh] total={b_tot*100:.2f}%  sharpe={b_sh:.3f}  mdd={b_mdd*100:.2f}%")
    wtot, wsh, wmdd, _ = stats(simulate(B, [1.0] * n, COST_PER_UNIT))
    print(f"[BASE 含空转成本]   total={wtot*100:.2f}%  sharpe={wsh:.3f}  mdd={wmdd*100:.2f}%")
    print("-" * 112)

    def row(name, w, lag, tag=""):
        nav = simulate(B, w, COST_PER_UNIT)
        tot, sh, mdd, ann = stats(nav)
        expo = sum(w[1:]) / (n - 1)
        sw = sum(abs(w[i] - w[i - 1]) for i in range(1, n))
        print(f"  {name:<34} lag{lag}  total={tot*100:>7.2f}% ({(tot-b_tot)*100:+7.2f}pp)  "
              f"sh={sh:>6.3f} ({sh-b_sh:+.3f})  mdd={mdd*100:>7.2f}% ({(mdd-b_mdd)*100:+6.2f}pp)  "
              f"expo={expo*100:>5.1f}% sw={sw:>5.1f} {tag}")

    # ---------- A. 趋势过滤: 指数 vs MA_N ----------
    print("\n== A. 趋势过滤: 指数 < MA_N -> 空仓 (lag=1 可交易) ==")
    for N in (5, 10, 20, 30, 60, 120):
        M = [ma(px, i, N) for i in range(n)]
        for lag in (1, 0):
            w = [1.0] * n
            for i in range(1, n):
                j = i - lag
                if j >= 0 and px[j] is not None and M[j] is not None and px[j] < M[j]:
                    w[i] = 0.0
            row(f"指数<MA{N} -> 空仓", w, lag)

    print("\n== A2. 趋势过滤 半仓变体 (lag=1) ==")
    for N in (10, 20, 30, 60):
        M = [ma(px, i, N) for i in range(n)]
        w = [1.0] * n
        for i in range(1, n):
            j = i - 1
            if px[j] is not None and M[j] is not None and px[j] < M[j]:
                w[i] = 0.5
        row(f"指数<MA{N} -> 半仓", w, 1)

    # ---------- B. 动量过滤 ----------
    print("\n== B. 动量过滤: 指数 N 日收益<0 -> 空仓 ==")
    for N in (5, 10, 15, 20, 40, 60):
        for lag in (1, 0):
            w = [1.0] * n
            for i in range(1, n):
                j = i - lag
                if j - N >= 0 and px[j] is not None and px[j - N] is not None:
                    if px[j] / px[j - N] - 1 < 0:
                        w[i] = 0.0
            row(f"指数{N}日动量<0 -> 空仓", w, lag)

    # ---------- C. 双均线 ----------
    print("\n== C. 双均线 MA_f>MA_s 满仓 else 空仓 ==")
    for f, s in ((5, 20), (10, 30), (20, 60), (20, 120)):
        Mf = [ma(px, i, f) for i in range(n)]
        Ms = [ma(px, i, s) for i in range(n)]
        for lag in (1, 0):
            w = [1.0] * n
            for i in range(1, n):
                j = i - lag
                if Mf[j] is not None and Ms[j] is not None and Mf[j] <= Ms[j]:
                    w[i] = 0.0
            row(f"MA{f}>MA{s} 满仓 else 空仓", w, lag)

    # ---------- D. 指数刹车 (本就 lag=1) ----------
    print("\n== D. 指数刹车 15日跌幅<=thr -> 空仓 hold 日 ==")
    r15 = [(px[i] / px[i - 15] - 1) if (i >= 15 and px[i] and px[i - 15]) else None for i in range(n)]
    for thr in (-0.08, -0.10, -0.12):
        for hold in (5, 10, 20):
            w = [1.0] * n
            i = 15
            trig = 0
            while i < n - 1:
                if r15[i] is not None and r15[i] <= thr:
                    trig += 1
                    for k in range(i + 1, min(i + 1 + hold, n)):
                        w[k] = 0.0
                    i += hold
                else:
                    i += 1
            row(f"15d<={thr*100:.0f}% -> 空仓{hold}日", w, 1, tag=f"trig={trig}")


if __name__ == "__main__":
    main()
