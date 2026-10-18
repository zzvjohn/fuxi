# -*- coding: utf-8 -*-
"""
参考信号表业绩敏感性诊断 — 为什么本地复现与原帖 +195.32% 差 90 倍
==================================================================
诊断维度:
  D1 执行时点: D开盘 / D收盘 / D+1开盘 / D+1收盘 (信号表条目 D 的信息截止日是哪个?)
  D2 数据复权: 检查 daily_prices.csv 是否不复权 (除权跳空对微盘股的影响)
  D3 对照: 小市值等权代理 (低价股) 同期 open-to-open, 判断信号表组合收益是否被数据口径拖累
"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

from p0_conditional_line import load_wide_light

SIG_JSON = Path("E:/log/signal_table_meanreverse.json")
TOP_K = 8
COST_REF = {"buy": 0.0003, "sell": 0.0003 + 0.0005}


def load_ref_signal_table():
    import re
    raw = json.load(open(SIG_JSON, encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", k):
            continue
        dt = pd.Timestamp(k)
        day = {}
        for code, info in (v or {}).items():
            if isinstance(info, dict):
                w = float(info.get("weight", 0.0) or 0.0)
            else:
                try:
                    w = float(info)
                except Exception:
                    w = 0.0
            if w <= 0:
                continue
            s = code.strip().upper()
            ts = (s[2:] + ".SH") if s.startswith("SH") else (s[2:] + ".SZ")
            day[ts] = w
        tot = sum(day.values())
        out[dt] = {c: w / tot for c, w in day.items()} if tot > 0 else {}
    return out


def run_exec_timing(ref_sig, dates, open_w, close_w, exec_mode):
    """exec_mode: openD(开盘) / closeD(收盘) / openD1(次日开盘) / closeD1(次日收盘)
    执行日 E 成交价 P(E), 收益 = P(E+1)/P(E)-1 (持有 1 天)。
    信号表条目 D → 执行日 E:
      openD/closeD : E = D        (信号用 D-1 数据, D 日执行)
      openD1/closeD1: E = D+1     (信号用 D 数据, D+1 执行)
    """
    px = open_w if exec_mode.startswith("open") else close_w
    shift = 0 if exec_mode.endswith("D") else 1
    r_arr, s_arr, b_arr = [], [], []
    prev_set = set()
    for i in range(len(dates) - 1):
        d = dates[i]
        e_idx = i + shift
        if e_idx >= len(dates) - 1:
            r_arr.append(0.0)
            s_arr.append(0.0)
            b_arr.append(0.0)
            continue
        target = ref_sig.get(d, None)
        cur_set = set(target.keys()) if target else set()
        r = 0.0
        if target:
            for s, w in target.items():
                if s not in px.columns:
                    continue
                p0 = px.iloc[e_idx][s]
                p1 = px.iloc[e_idx + 1][s]
                if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                    r += w * (p1 / p0 - 1.0)
        r_arr.append(float(r))
        s_arr.append(len(prev_set - cur_set) / TOP_K)
        b_arr.append(len(cur_set - prev_set) / TOP_K)
        prev_set = cur_set
    r_arr = np.array(r_arr)
    cost_arr = np.array(s_arr) * COST_REF["sell"] + np.array(b_arr) * COST_REF["buy"]
    net_arr = r_arr - cost_arr
    gross = float(np.prod(1 + r_arr) - 1)
    net = float(np.prod(1 + net_arr) - 1)
    nav = np.cumprod(1 + net_arr)
    peak = np.maximum.accumulate(nav)
    sharpe = float(net_arr.mean() / net_arr.std() * np.sqrt(240)) if net_arr.std() > 0 else 0.0
    sharpe_g = float(r_arr.mean() / r_arr.std() * np.sqrt(240)) if r_arr.std() > 0 else 0.0
    return {
        "exec": exec_mode,
        "gross": round(100 * gross, 2),
        "net": round(100 * net, 2),
        "cost": round(100 * (gross - net), 2),
        "sharpe_g": round(sharpe_g, 2),
        "sharpe_n": round(sharpe, 2),
        "maxdd": round(100 * float(np.min(nav / peak - 1)), 2),
    }


def main():
    print("=" * 72)
    print("  参考信号表执行时点敏感性诊断 (原帖窗口 2024-01~2025-11)")
    print("=" * 72)
    close_w, open_w, vol_w, lhb_w = load_wide_light()
    dates = close_w.index
    ref_sig = load_ref_signal_table()
    lo, hi = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-11-30")
    wd = [d for d in dates if lo <= d <= hi]

    print(f"\n  {'执行时点':<10} {'毛收益':>9} {'净收益':>9} {'成本':>7} "
          f"{'Sharpe毛':>9} {'Sharpe净':>9} {'MaxDD':>8}")
    for mode in ["openD", "closeD", "openD1", "closeD1"]:
        res = run_exec_timing(ref_sig, wd, open_w, close_w, mode)
        print(f"  {res['exec']:<10} {res['gross']:>8}% {res['net']:>8}% {res['cost']:>6}% "
              f"{res['sharpe_g']:>9} {res['sharpe_n']:>9} {res['maxdd']:>7}%")

    # D2: 数据复权检查 — 找信号表里几只股票, 看除权日价格是否跳空
    print("\n  [D2] 数据复权检查 (信号表股票, 高换手区间前后):")
    all_ref = sorted({s for v in ref_sig.values() for s in v})
    # 取出现次数最多的几只
    from collections import Counter
    cnt = Counter()
    for v in ref_sig.values():
        for s in v:
            cnt[s] += 1
    top_codes = [c for c, _ in cnt.most_common(6)]
    for code in top_codes:
        s = close_w[code].loc[lo:hi]
        d0 = (s.pct_change() < -0.25).sum()
        big = s.index[s.pct_change() < -0.25]
        jump = " | ".join(d.strftime("%Y-%m-%d") for d in big[:4])
        print(f"    {code}: 区间 {len(s)} 日, 单日跌>25% {d0} 次 (疑似除权跳空): {jump if jump else '无'}")

    # D3: 小市值代理对照 — 价格最低 300 只等权 open-to-open
    print("\n  [D3] 低价股等权对照 (小市值代理, 2024-01~2025-11):")
    price_last = close_w.loc[wd[0]]
    low_p = price_last.nsmallest(300).index
    ok = [s for s in low_p if s in open_w.columns]
    r_series = []
    for i in range(len(wd) - 1):
        r = 0.0
        n = 0
        for s in ok:
            p0 = open_w.iloc[i][s]
            p1 = open_w.iloc[i + 1][s]
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                r += p1 / p0 - 1.0
                n += 1
        r_series.append(float(r / n) if n else 0.0)
    r_arr = np.array(r_series)
    cum = float(np.prod(1 + r_arr) - 1)
    sharpe = float(r_arr.mean() / r_arr.std() * np.sqrt(240))
    print(f"    低价300等权 open-to-open 累计: {100 * cum:+.2f}% | 日Sharpe {sharpe:.2f} | n={len(r_arr)}")

    # 全市场等权对照
    r_all = []
    for i in range(len(wd) - 1):
        p0 = open_w.iloc[i]
        p1 = open_w.iloc[i + 1]
        m = np.isfinite(p0.to_numpy()) & np.isfinite(p1.to_numpy()) & (p0.to_numpy() > 0)
        r_all.append(float(((p1[m] / p0[m]) - 1).mean()))
    r_arr2 = np.array(r_all)
    cum2 = float(np.prod(1 + r_arr2) - 1)
    print(f"    全市场等权 open-to-open 累计: {100 * cum2:+.2f}% | 日Sharpe "
          f"{float(r_arr2.mean() / r_arr2.std() * np.sqrt(240)):.2f}")


if __name__ == "__main__":
    main()
