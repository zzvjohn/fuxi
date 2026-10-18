# -*- coding: utf-8 -*-
"""
M5 窄池 LGBM 执行配置扫描 — EMA 平滑 × 重平衡周期 (2026-08-23)
================================================================
目标: 把日换手从 78.9% 压到 ≤20%, 验证"名单稳定性约束"是否挽救日频 LGBM。
口径 (诚实无前视, 与 JQ 交付对齐):
  - t 日收盘后生成信号 (只用 t 及以前数据)
  - t+1 日开盘执行换仓 (open 价)
  - 持仓收益 = 执行开盘 → 次日开盘 (open-to-open)
  - 成本 = 日换手 × 0.67% (买万1+滑点0.3% + 卖万1+印花0.05%+滑点0.3%)
  - 停牌 (open NaN): 不买不卖, 当日收益按 0
配置网格: 信号源 {fwd3, fwd5} × EMA {0, 0.3, 0.5} × 重平衡周期 {1, 3, 5}
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

DATA_DIR = BASE.parent / "data"
PRED_CSV_FWD3 = DATA_DIR / "narrow_pool_lgbm_oos_preds.csv"
PRED_CSV_FWD5 = DATA_DIR / "narrow_pool_lgbm_fwd5_oos_preds.csv"
OUT_JSON = DATA_DIR / "narrow_pool_lgbm_stable_sim.json"

TOP_K = 8
COST_ROUND_TRIP = 0.0067
REBAL_CYCLES = [1, 3, 5]
EMA_ALPHAS = [0.0, 0.3, 0.5]


def load_pred_map(csv_path):
    """CSV -> {date: {code: score}}, 已按 (date, code) 去重"""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
    out = {}
    for dt, grp in df.groupby("trade_date"):
        out[dt] = grp.set_index("ts_code")["pred"]
    return out


def ema_smooth(pred_map, alpha):
    """对每只股票的时间序列做 EMA(1-alpha)。宽表沿时间轴, 只用过去信息。"""
    if alpha <= 0:
        return pred_map
    s = pd.DataFrame(pred_map).T.sort_index()  # index=date, columns=code
    s = s.ewm(alpha=1 - alpha, adjust=False, axis=0).mean()
    out = {}
    for dt, row in s.iterrows():
        out[dt] = row.dropna()
    return out


def pick_top8(pred_map, dt, pool):
    """确定性 top8: 分数排序 + 股票代码 tie-breaker"""
    pred = pred_map.get(dt)
    if pred is None:
        return []
    pp = pred.reindex(pool).dropna()
    if len(pp) == 0:
        return []
    # sort_values(kind=mergesort) 稳定排序, 同分按代码字典序, 保证确定性
    return list(pp.sort_values(ascending=False, kind="mergesort").nlargest(TOP_K).index)


def simulate(pred_map, dates, open_w, pool_by_day, cycle, alpha):
    """t 收盘定名单 → t+1 开盘执行 → open-to-open 收益。"""
    holdings = {}          # code -> 等权 1/8
    returns, costs, turns = [], [], []
    sig_dates = sorted(pred_map.keys())
    first_d = sig_dates[0] if sig_dates else None
    day_idx = {d: i for i, d in enumerate(dates)}

    for i in range(len(dates) - 2):
        t = dates[i]            # 信号日 (收盘后)
        e = i + 1               # 执行日 (开盘)
        if t not in pred_map:
            returns.append(0.0); costs.append(0.0); turns.append(0.0)
            continue
        pool = pool_by_day.get(t)
        if pool is None or len(pool) < TOP_K * 2:
            returns.append(0.0); costs.append(0.0); turns.append(0.0)
            continue
        # 用交易日序数判断重平衡 (自然日间隔不可靠)
        if t == first_d or (i - day_idx.get(first_d, i)) % cycle == 0:
            target = pick_top8(pred_map, t, pool)
            cur = set(holdings.keys())
            sell = sorted(cur - set(target))
            buy = sorted(set(target) - cur)
            n_chg = len(sell) + len(buy)
            turns.append(min(n_chg, 2 * TOP_K) / (2 * TOP_K))  # 换手率=买卖总股数/(2K)
            # 开盘执行: open[e] 有效才可成交
            op_e = open_w.iloc[e]
            for s in sell:
                holdings.pop(s, None)
            for s in buy:
                if np.isfinite(op_e.get(s, np.nan)) and op_e[s] > 0:
                    holdings[s] = 1.0 / TOP_K
            cost = (len(sell) + len([s for s in buy if s in holdings])) / (2 * TOP_K) * COST_ROUND_TRIP
            costs.append(cost)
        else:
            turns.append(0.0)
            costs.append(0.0)

        # 持仓收益: 执行开盘 e → 次日开盘 e+1
        if holdings:
            op_e = open_w.iloc[e]
            op_e1 = open_w.iloc[e + 1]
            r = 0.0
            n = 0
            for s in holdings:
                p0 = op_e.get(s, np.nan)
                p1 = op_e1.get(s, np.nan)
                if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                    r += (1.0 / TOP_K) * (p1 / p0 - 1.0)
                    n += 1
            returns.append(r)
        else:
            returns.append(0.0)
    r_arr = np.array(returns)
    c_arr = np.array(costs)
    t_arr = np.array(turns)
    gross = np.prod(1 + r_arr) - 1
    net = np.prod(1 + r_arr - c_arr) - 1
    sharpe_g = r_arr.mean() / r_arr.std() * np.sqrt(240) if r_arr.std() > 0 else 0.0
    net_r = r_arr - c_arr
    sharpe_n = net_r.mean() / net_r.std() * np.sqrt(240) if net_r.std() > 0 else 0.0
    nav = np.cumprod(1 + net_r)
    maxdd = float(np.min(nav / np.maximum.accumulate(nav) - 1))
    # 分年 (按执行日年份)
    yrs = {}
    for i in range(len(r_arr)):
        yrs.setdefault(dates[i + 1].year, []).append(net_r[i])
    annual = {str(y): round(100 * (np.prod(1 + np.array(v)) - 1), 2) for y, v in yrs.items()}
    return {
        "gross": round(100 * gross, 2),
        "net": round(100 * net, 2),
        "sharpe_g": round(float(sharpe_g), 2),
        "sharpe_n": round(float(sharpe_n), 2),
        "maxdd": round(100 * maxdd, 2),
        "turn_daily_pct": round(100 * float(t_arr.mean()), 1),
        "yearly_net": annual,
        "n_days": len(r_arr),
    }


def main():
    print("=" * 68)
    print("  窄池 LGBM 执行配置扫描 (EMA × 重平衡周期, open-to-open 诚实口径)")
    print("=" * 68)
    close_w, open_w, vol_w, lhb_w = load_wide_light()
    dates = close_w.index
    dev = close_w / close_w.rolling(20).mean() - 1.0
    pool_by_day = {}
    for i in range(len(dates)):
        s = dev.iloc[i]
        valid = np.isfinite(s.to_numpy())
        pool_by_day[dates[i]] = s[valid].nsmallest(1000).index

    results = {}
    for tag, csv_path in [("fwd3", PRED_CSV_FWD3), ("fwd5", PRED_CSV_FWD5)]:
        pred_map = load_pred_map(csv_path)
        if pred_map is None:
            print(f"\n  [{tag}] 预测文件不存在, 跳过: {csv_path.name}")
            continue
        print(f"\n  [{tag}] OOS 预测覆盖 {len(pred_map)} 交易日")
        for alpha in EMA_ALPHAS:
            sm = ema_smooth(pred_map, alpha)
            for cycle in REBAL_CYCLES:
                key = f"{tag}|ema{alpha}|reb{cycle}"
                res = simulate(sm, dates, open_w, pool_by_day, cycle, alpha)
                results[key] = res
                print(f"    {key:<22} 毛 {res['gross']:>7}% | 净 {res['net']:>7}% "
                      f"| SR毛 {res['sharpe_g']:>5} | SR净 {res['sharpe_n']:>5} "
                      f"| DD {res['maxdd']:>6}% | 日换手 {res['turn_daily_pct']:>5}%")
                for y, v in res["yearly_net"].items():
                    pass
                print(f"      分年净: {res['yearly_net']}")
    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print(f"\n  扫描结果已存: {OUT_JSON}")


if __name__ == "__main__":
    main()
