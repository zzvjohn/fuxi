# -*- coding: utf-8 -*-
"""
差距分解 — 参考日频策略信号表 vs 我们的窄池 LGBM 复现 (2026-08-23)
==================================================================
问题: 为什么参考策略 JQ 回测 +195.32%/2.112 (2024-01~2025-11), 而我们复现全区间净 -1.61%?
正交分解四个维度:
  D1 执行层   : 完全跟随信号表 (参考原版, 无 NDrop) vs NDrop=1 (我们之前的模式)
  D2 成本口径 : 参考口径 0.11%/换手 (万3佣金×2+印花0.05%, FixedSlippage(0)) vs 我们 0.67%/换手
  D3 窗口     : 原帖窗口 2024-01~2025-11 vs 全信号期 2024-01~2026-05
  D4 信号源   : 参考信号表 (Qlib+LightGBM 158特征) vs 我们的 LGBM (29特征 walk-forward OOS)

统一口径 (对齐 JQ 执行器):
  信号 D 用 D-1 收盘信息, D 开盘调仓, 收益 = open[D+1]/open[D]-1 (open-to-open)
  参考: signal_table[D]  (JQ 回测模式 use_date = today_td)
  我们: pred[D-1] 池内 top8  (信息截止 D-1 收盘)
成本: 每次换股 = 卖(佣金+印花) + 买(佣金), 参考/我们两套费率
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
PRED_CSV = BASE.parent / "data" / "narrow_pool_lgbm_oos_preds.csv"
OUT_JSON = BASE.parent / "data" / "gap_decomp_ref_vs_ours.json"

TOP_K = 8
POOL_N = 1000

# 两套成本: (买侧费率, 卖侧费率) — 每单位换手
COST_REF = {"buy": 0.0003, "sell": 0.0003 + 0.0005}          # 万3佣金×2 + 印花0.05%, 零滑点
COST_OURS = {"buy": 0.0001 + 0.003, "sell": 0.0001 + 0.0005 + 0.003}  # 万1+滑点0.3%, 卖加印花


def load_ref_signal_table():
    """signal_table {date: {SH600266: {weight}} } → {pd.Timestamp: {ts_code: w}} (归一化)"""
    raw = json.load(open(SIG_JSON, encoding="utf-8"))
    out = {}
    import re
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
            if s.startswith("SH") and len(s) >= 8:
                ts = s[2:] + ".SH"
            elif s.startswith("SZ") and len(s) >= 8:
                ts = s[2:] + ".SZ"
            else:
                continue
            day[ts] = w
        total = sum(day.values())
        if total > 0:
            out[dt] = {c: w / total for c, w in day.items()}
        else:
            out[dt] = {}
    return out


def load_our_pred_signal():
    """pred[D] 池内 top8 等权 → {date: {ts_code: w}}, 信号 = pred[D-1] 的信息"""
    df = pd.read_csv(PRED_CSV)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    pred_map = {}
    for dt, grp in df.groupby("trade_date"):
        pred_map[dt] = grp.set_index("ts_code")["pred"]
    return pred_map


def replay(signal_fn, dates, ndrop):
    """执行层重放: 返回每日生效持仓名单 (d 开盘调仓后, 当日收益暴露的名单)"""
    holdings = {}
    eff = []
    for i in range(len(dates) - 1):
        d = dates[i]
        target = signal_fn(d)          # None → 严格清仓; {} → 空仓
        if target is None:
            holdings = {}
        elif ndrop and holdings:
            cur_set = set(holdings.keys())
            new_set = set(target.keys())
            drop_c = sorted([s for s in cur_set if s not in new_set],
                            key=lambda s: (holdings[s], s))[:ndrop]
            add_c = sorted([s for s in new_set if s not in cur_set], key=lambda s: s)[:ndrop]
            for s in drop_c:
                del holdings[s]
            for s in add_c:
                holdings[s] = 1.0 / TOP_K
        else:
            holdings = dict(target)
        eff.append(dict(holdings))
    return eff


def simulate(signal_fn, dates, open_w, ndrop, cost):
    """open-to-open: d 开盘调仓 (信息截止 d-1), 收益 open[d+1]/open[d]-1。
    换手成本按相邻日生效名单差异 × 费率"""
    eff = replay(signal_fn, dates, ndrop)
    r_arr, s_arr, b_arr = [], [], []
    for i, h in enumerate(eff):
        r = 0.0
        for s, w in h.items():
            if s not in open_w.columns:
                continue
            o0 = open_w.iloc[i][s]
            o1 = open_w.iloc[i + 1][s]
            if np.isfinite(o0) and np.isfinite(o1) and o0 > 0:
                r += w * (o1 / o0 - 1.0)
        r_arr.append(float(r))
        prev = set(eff[i - 1].keys()) if i > 0 else set()
        cur = set(h.keys())
        s_arr.append(len(prev - cur) / TOP_K)
        b_arr.append(len(cur - prev) / TOP_K)
    r_arr = np.array(r_arr)
    s_arr = np.array(s_arr)
    b_arr = np.array(b_arr)
    cost_arr = s_arr * cost["sell"] + b_arr * cost["buy"]
    net_arr = r_arr - cost_arr
    gross = float(np.prod(1 + r_arr) - 1)
    net = float(np.prod(1 + net_arr) - 1)
    nav = np.cumprod(1 + net_arr)
    peak = np.maximum.accumulate(nav)
    maxdd = float(np.min(nav / peak - 1))
    sharpe = (float(net_arr.mean() / net_arr.std() * np.sqrt(240))
              if net_arr.std() > 0 else 0.0)
    sharpe_gross = (float(r_arr.mean() / r_arr.std() * np.sqrt(240))
                    if r_arr.std() > 0 else 0.0)
    turn_daily = float((s_arr + b_arr).mean() / 2)
    return {
        "n_days": len(r_arr),
        "gross": round(100 * gross, 2),
        "net": round(100 * net, 2),
        "cost_cum": round(100 * (gross - net), 2),
        "sharpe_gross": round(sharpe_gross, 2),
        "sharpe_net": round(sharpe, 2),
        "maxdd": round(100 * maxdd, 2),
        "turn_daily_pct": round(100 * turn_daily, 1),
    }


def window_mask(dates, lo, hi):
    return [d for d in dates if pd.Timestamp(lo) <= d <= pd.Timestamp(hi)]


def main():
    print("=" * 72)
    print("  差距分解: 参考信号表 vs 我们的 LGBM (open-to-open 统一口径)")
    print("=" * 72)
    close_w, open_w, vol_w, lhb_w = load_wide_light()
    dates = close_w.index
    print(f"  宽表: {len(dates)} 日 | {dates[0].date()} ~ {dates[-1].date()}")

    ref_sig = load_ref_signal_table()
    print(f"  参考信号表: {len(ref_sig)} 天")

    # 参考信号表的股票在本地数据覆盖度
    all_ref_codes = set()
    for v in ref_sig.values():
        all_ref_codes |= set(v.keys())
    covered = all_ref_codes & set(close_w.columns)
    print(f"  参考信号覆盖 {len(all_ref_codes)} 只, 本地数据命中 {len(covered)} "
          f"({100 * len(covered) / max(len(all_ref_codes), 1):.1f}%)")

    def ref_fn(d):
        return ref_sig.get(d, None)

    # 我们的信号: pred[D-1] 池内 top8
    pred_map = load_our_pred_signal()
    dev = close_w / close_w.rolling(20).mean() - 1.0
    pool_by_day = {}
    date_list = list(dates)
    for i in range(len(date_list)):
        s = dev.iloc[i]
        valid = np.isfinite(s.to_numpy())
        pool_by_day[date_list[i]] = s[valid].nsmallest(POOL_N).index

    def ours_fn(d):
        i = date_list.index(d) if d in date_list else None
        if i is None or i == 0:
            return None
        pred = pred_map.get(date_list[i - 1])
        if pred is None:
            return None
        pool = pool_by_day.get(date_list[i - 1])
        if pool is None:
            return None
        pp = pred.reindex(pool).dropna()
        if len(pp) < TOP_K:
            return {}
        top = pp.nlargest(TOP_K).index
        return {s: 1.0 / TOP_K for s in top}

    windows = {
        "orig_2024_01_2025_11": ("2024-01-01", "2025-11-30"),
        "full_2024_01_2026_05": ("2024-01-01", "2026-05-29"),
    }
    results = {}
    for wname, (lo, hi) in windows.items():
        wd = window_mask(dates, lo, hi)
        for sig_name, sig_fn in [("ref_signal_table", ref_fn), ("ours_lgbm_29f", ours_fn)]:
            for ndrop in (0, 1):
                for cost_name, cost in [("ref_cost_011", COST_REF), ("ours_cost_067", COST_OURS)]:
                    key = f"{wname}|{sig_name}|ndrop{ndrop}|{cost_name}"
                    res = simulate(sig_fn, wd, open_w, ndrop, cost)
                    res["label"] = f"{sig_name}/ndrop{ndrop}/{cost_name}"
                    results[key] = res
                    print(f"\n  [{wname} | {sig_name} | NDrop={ndrop} | {cost_name}]")
                    print(f"    毛 {res['gross']}% | 净 {res['net']}% | 成本 {res['cost_cum']}%")
                    print(f"    Sharpe 毛 {res['sharpe_gross']} / 净 {res['sharpe_net']} | "
                          f"MaxDD {res['maxdd']}% | 日换手 {res['turn_daily_pct']}%")

    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print(f"\n  结果已存: {OUT_JSON}")


if __name__ == "__main__":
    main()
