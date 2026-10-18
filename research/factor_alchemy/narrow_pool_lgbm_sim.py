# -*- coding: utf-8 -*-
"""
M3 窄池 LightGBM 模拟交易 — top8 + NDrop 执行层 + 成本 (2026-08-23)
================================================================
复刻聚宽日频策略的执行层纪律: TopK=8 + NDrop=1 (每日只换 1/8, 平均持有 9.9 天)。
对照:
  - 基线 A: 纯 NDrop 无信号 (随机维持) — 剥离信号只看执行层
  - 基线 B: 线性 rank 和 (B1, 已知 top8 Sharpe 0.05 证伪形态)
  - King 同期: 全区间 +182.57%/0.69 (外部已知, 用于裁决锚)
成本: 0.72%/次全额换手 (佣金万1×2 + 印花税 0.05% + 滑点 0.3%×2), 单边 0.36%。
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

PRED_CSV = BASE.parent / "data" / "narrow_pool_lgbm_oos_preds.csv"
OUT_JSON = BASE.parent / "data" / "narrow_pool_lgbm_sim.json"

TOP_K = 8
NDROP = 1
COST_ROUND_TRIP = 0.0067    # 一次换手完整成本: 买(万1+滑点0.3%) + 卖(万1+印花0.05%+滑点0.3%)
INIT_CAP = 1_000_000


def load_preds():
    df = pd.read_csv(PRED_CSV)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    pred_map = {}
    for dt, grp in df.groupby("trade_date"):
        pred_map[dt] = grp.set_index("ts_code")["pred"]
    return pred_map


def simulate(pred_map, pool_by_day, dates, fwd1, mode="lgbm"):
    """NDrop 执行层模拟: 每日最多换 NDROP 只, t 收盘定名单 → t+1 收盘收益"""
    holdings = {}          # code -> 权重 (等权 1/K)
    returns = []           # 日组合收益 (成本前)
    cost_days = []
    turns = []
    prev_hold = set()

    for i in range(len(dates) - 1):
        dt = dates[i]
        pool = pool_by_day.get(dt)
        if pool is None or len(pool) < TOP_K * 2:
            returns.append(0.0)
            cost_days.append(0.0)
            turns.append(0.0)
            continue

        if mode == "lgbm":
            pred = pred_map.get(dt)
            if pred is None:
                target = []
            else:
                pp = pred.reindex(pool).dropna()
                target = list(pp.nlargest(TOP_K).index)
        elif mode == "random":
            rng = np.random.RandomState(42 + i)
            target = list(rng.choice(np.array(pool), size=TOP_K, replace=False))
        else:
            raise ValueError(mode)

        # 无信号且空仓: 跳过 (不计换手, 收益 0)
        if not target and not holdings:
            returns.append(0.0)
            cost_days.append(0.0)
            turns.append(0.0)
            continue

        # 初始化
        if not holdings:
            if not target:
                returns.append(0.0)
                cost_days.append(0.0)
                turns.append(0.0)
                continue
            for s in target:
                holdings[s] = 1.0 / TOP_K
            prev_hold = set(target)
            turns.append(1.0)
        else:
            cur = set(holdings.keys())
            # NDrop 确定性换仓: 换出信号最低的持仓, 换入信号最高的候选
            # (set 迭代顺序依赖 hash seed, 不能用裸 set 顺序 — 三次运行毛收益漂移实证)
            if mode == "lgbm" and pred is not None:
                score = {s: float(pred.get(s, -np.inf)) for s in set(list(cur) + target)}
                # tie-breaker: 股票代码, 彻底确定性 (score 相同如 -inf 时不依赖 set 顺序)
                drop = sorted([s for s in cur if s not in target],
                              key=lambda s: (score.get(s, -np.inf), s))
                add = sorted([s for s in target if s not in cur],
                             key=lambda s: (-score.get(s, -np.inf), s))
            else:
                drop = sorted([s for s in cur if s not in target])
                add = sorted([s for s in target if s not in cur], reverse=True)
            n_change = min(len(drop), len(add), NDROP)
            for s in drop[:n_change]:
                del holdings[s]
            for s in add[:n_change]:
                holdings[s] = 1.0 / TOP_K
            turns.append(n_change / TOP_K)

        # t+1 收益 (成本前)
        h = list(holdings.keys())
        r = float(fwd1.iloc[i][h].mean()) if h else 0.0
        returns.append(r)
        cost_days.append(turns[-1] * COST_ROUND_TRIP)
        prev_hold = set(h)

    r_arr = np.array(returns)
    c_arr = np.array(cost_days)
    t_arr = np.array(turns)
    gross = np.cumprod(1 + r_arr)
    net = np.cumprod(1 + r_arr - c_arr)
    # 汇总
    yrs = {}
    for i in range(len(r_arr)):
        yrs.setdefault(dates[i].year, []).append(r_arr[i])
    annual = {}
    for y, v in yrs.items():
        prod = float(np.prod(1 + np.array(v)) - 1)
        annual[y] = round(100 * prod, 2)
    n_days = len(r_arr)
    return {
        "n_days": n_days,
        "gross_return": round(100 * (float(gross[-1]) - 1), 2),
        "net_return": round(100 * (float(net[-1]) - 1), 2),
        "ann_cost": round(100 * (float(gross[-1]) - float(net[-1])), 2),
        "daily_mean": round(100 * float(r_arr.mean()), 4),
        "sharpe_gross": round(float(r_arr.mean() / r_arr.std() * np.sqrt(240)), 2),
        "sharpe_net": round(float((r_arr - c_arr).mean() / (r_arr - c_arr).std() * np.sqrt(240)), 2),
        "maxdd": round(100 * float(_max_dd(gross)), 2),
        "turnover": round(100 * float(t_arr.mean()), 1),
        "yearly_net": annual,
        "pos_years": f"{sum(1 for v in annual.values() if v > 0)}/{len(annual)}",
    }


def _max_dd(nav):
    peak = np.maximum.accumulate(nav)
    return np.min(nav / peak - 1)


def main():
    print("=" * 68)
    print("  窄池 LightGBM — NDrop 执行层模拟 (top8 / NDrop=1 / 成本 0.36% 单边)")
    print("=" * 68)
    close_w, open_w, vol_w, lhb_w = load_wide_light()
    dates = close_w.index
    fwd1 = close_w.shift(-1) / close_w - 1.0
    dev = close_w / close_w.rolling(20).mean() - 1.0
    pool_by_day = {}
    for i in range(len(dates)):
        s = dev.iloc[i]
        valid = np.isfinite(s.to_numpy())
        pool_by_day[dates[i]] = s[valid].nsmallest(1000).index

    pred_map = load_preds()
    print(f"  OOS 预测覆盖 {len(pred_map)} 个交易日")
    out = {}
    for mode in ["lgbm", "random"]:
        res = simulate(pred_map, pool_by_day, dates, fwd1, mode=mode)
        out[mode] = res
        print(f"\n  [{mode}]")
        print(f"    毛收益 {res['gross_return']}% | 净收益 {res['net_return']}% "
              f"| 年成本 {res['ann_cost']}%")
        print(f"    Sharpe 毛 {res['sharpe_gross']} / 净 {res['sharpe_net']} | "
              f"MaxDD {res['maxdd']}% | 日换手 {res['turnover']}%")
        print(f"    分年净收益: {res['yearly_net']} ({res['pos_years']} 正)")
    payload = {
        "top_k": TOP_K, "ndrop": NDROP, "cost_round_trip": COST_ROUND_TRIP,
        "results": out,
        "baseline_ref": {
            "king_full": "182.57%/0.69/-35.41% (2021-2026)",
            "linear_rank_top8": "top8 Sharpe 0.05 (已证伪形态)",
        },
    }
    json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print(f"\n  模拟结果已存: {OUT_JSON}")


if __name__ == "__main__":
    main()
