# -*- coding: utf-8 -*-
"""
P1 慢信号复合分 × NDrop 边际调仓 — 日频框架可行性验证
======================================================
背景: 聚宽日频均值回归策略 (post/63846) 用 TopK=8 + NDrop=1 (每日只换1只)
      实现 12.5% 单边换手, 成本后 Sharpe 2.1 (2024-01~2025-11)。
      我们 P0/P0.5 已证伪"快信号日频全换仓" (换手76% → 成本66%)。
      本验证回答: 我们的慢信号因子 (overreaction/lhb/overnight5) 复合后,
      在 NDrop 框架下的真实换手率与成本后净 alpha 是多少。

方法:
  1. FactorICComputer 数据层 (270d, 与 P0 同口径)
  2. 复合分 = overreaction + lhb_freq_20d + overnight5 等权截面 rank 和
     (三个 P0 日频 IC 最强因子, 两两低相关 ρ<0.3, 均为慢/中信号)
  3. 换手模拟 (t 日名单 → t+1 执行):
     - FULL   : 每日全换 (P0.5 证伪场景, 对照)
     - NDROP1 : 每日最多换 1 只 (复刻聚宽策略)
     - BUFFER : NDrop=1 + 缓冲带 (目标排名仍在前 top(K+B) 不换出)
  4. 成本核算: 年成本 = 240 × 换手 × 0.72%; 毛 alpha = topK 等权组合
     相对全池均值的日超额年化; 净 = 毛 - 成本
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")
from factor_ic_computer import FactorICComputer

LOOKBACK_DAYS = 270
K = 8                      # 持仓数 (对齐聚宽策略; 另测 K=10)
BUFFER = 2                 # 缓冲带: 持仓股排名仍在前 K+B 内不换出
COST_RT = 0.0072           # 一次双边换手成本 (佣金万1×2+印花税+滑点0.3%×2)

FORMULAS = {
    "overreaction": "-close.pct_change(1).rolling(20).max()",
    "lhb_freq_20d": "-lhb_flag.rolling(20).sum()",
    "overnight5": "(open / close.shift(1) - 1.0).rolling(5, min_periods=3).sum()",
}


def eval_formula(formula, env):
    lines = [ln.strip() for ln in formula.strip().split("\n") if ln.strip()]
    body, final = lines[:-1], lines[-1]
    for ln in body:
        exec(ln, env)
    return eval(final, env)


def rank_pct_axis1(df):
    """宽表(index=日期, cols=股票) 截面 rank pct — 铁律: 必须 axis=1 (沿股票截面)"""
    return df.rank(axis=1, pct=True, na_option="keep")


def simulate_turnover(dates, comp_rank, fwd_1d, k=K, mode="NDROP1", buffer=BUFFER):
    """
    逐日模拟持仓演进, 返回 (单边换手率序列, 日收益序列)
    mode: FULL 全换 / NDROP1 每日最多换1只 / BUFFER NDrop+缓冲带
    """
    n = len(dates)
    holdings = None      # 当前持仓 set (t 日收盘后生效)
    turns, rets = [], []

    for i in range(n - 1):  # i: 决策日 (t), 收益取自 t+1
        r = comp_rank.iloc[i]
        fv = r.to_numpy()
        fw = fwd_1d.iloc[i].to_numpy()
        m = np.isfinite(fv)
        if m.sum() < 30:
            turns.append(np.nan)
            rets.append(np.nan)
            continue
        valid_stocks = list(comp_rank.columns[m])
        vals = fv[m]
        order = np.argsort(-vals)
        topk = [valid_stocks[j] for j in order[:k]]
        target = set(topk)

        if holdings is None:
            new_holdings = target
            turn = 1.0
        elif mode == "FULL":
            new_holdings = target
            turn = len(target - holdings) / k
        else:
            # NDROP1: 最多换 1 只 — 换出: 持仓中排名最差的1只 (或已不在目标池的)
            cur_rank = {s: r[s] for s in holdings if np.isfinite(r[s])}
            if not cur_rank:
                new_holdings = target
                turn = 1.0
            else:
                worst = min(cur_rank, key=cur_rank.get)
                new_holdings = set(holdings)
                if worst not in target:
                    if mode == "BUFFER":
                        # 缓冲带: worst 掉出 topK 但仍在前 top(K+B) → 不换 (0 换手)
                        topkb = set(valid_stocks[j] for j in order[:k + buffer])
                        if worst in topkb:
                            turn = 0.0
                        else:
                            new_holdings.discard(worst)
                            for s in topk:
                                if s not in new_holdings:
                                    new_holdings.add(s)
                                    break
                            turn = 1.0 / k
                    else:
                        # NDROP1: 换出 worst, 换入目标中排名最高的未持仓股
                        new_holdings.discard(worst)
                        for s in topk:
                            if s not in new_holdings:
                                new_holdings.add(s)
                                break
                        turn = 1.0 / k
                else:
                    turn = 0.0  # worst 仍在目标内: 不换

        turns.append(turn)
        # t+1 日超额收益: topK 组合 - 全池等权均值 (超额口径, 修正绝对收益偏差)
        r_next = fwd_1d.iloc[i]
        bench = float(np.nanmean(r_next.to_numpy())) if np.isfinite(r_next.to_numpy()).any() else 0.0
        hs = [s for s in new_holdings if np.isfinite(r_next[s])]
        rets.append(float(r_next[hs].mean()) - bench if hs else 0.0)
        holdings = new_holdings

    return np.array(turns), np.array(rets)


def main():
    t0 = time.time()
    print("=" * 90)
    print("P1 慢信号复合 × NDrop 边际调仓 — 日频框架可行性验证")
    print("=" * 90)

    print(f"\n[1/4] 加载数据层 (lookback={LOOKBACK_DAYS}d)...")
    comp = FactorICComputer(lookback_days=LOOKBACK_DAYS)
    comp._load_data()
    print(f"    done {time.time()-t0:.0f}s | {len(comp._price_df)} rows")

    print("[2/4] 宽表字段 + 三因子复合分...")
    fields = {}
    for f in ["close", "open", "volume", "lhb_flag"]:
        w = comp._get_wide_field(f)
        fields[f] = w
    close_w = fields["close"]
    dates = close_w.index
    stocks = close_w.columns
    print(f"    宽表: {len(dates)} 交易日 × {len(stocks)} 只")

    env = {
        "np": np, "pd": pd,
        "close": close_w, "open": fields["open"], "volume": fields["volume"],
        "lhb_flag": fields["lhb_flag"],
    }
    rps = {}
    ics = {}
    for fname, formula in FORMULAS.items():
        fw = eval_formula(formula, dict(env))
        fw = fw.reindex(index=dates, columns=stocks).replace([np.inf, -np.inf], np.nan)
        rps[fname] = rank_pct_axis1(fw)
        # 单因子日频 IC 复核 (对齐 P0)
        fwd1 = close_w.shift(-1) / close_w - 1.0
        ic_list = []
        for i in range(len(dates)):
            fv, rv = fw.iloc[i].to_numpy(), fwd1.iloc[i].to_numpy()
            mm = np.isfinite(fv) & np.isfinite(rv)
            if mm.sum() >= 30:
                ic = spearmanr(fv[mm], rv[mm])[0]
                if np.isfinite(ic):
                    ic_list.append(ic)
        ics[fname] = (np.mean(ic_list), np.mean(ic_list) / np.std(ic_list) * np.sqrt(len(ic_list)))
        print(f"    {fname:<16} IC(1d)={ics[fname][0]:+.4f} t={ics[fname][1]:.1f}")
        del fw

    comp_score = sum(rps.values()) / len(rps)      # 等权 rank 和 (打散 v1 形态)
    comp_rank = rank_pct_axis1(comp_score)          # 复合再排名 (只影响相对序)
    fwd_1d = close_w.shift(-1) / close_w - 1.0

    # 复合分日频 IC
    icc = []
    for i in range(len(dates)):
        fv, rv = comp_score.iloc[i].to_numpy(), fwd_1d.iloc[i].to_numpy()
        mm = np.isfinite(fv) & np.isfinite(rv)
        if mm.sum() >= 30:
            ic = spearmanr(fv[mm], rv[mm])[0]
            if np.isfinite(ic):
                icc.append(ic)
    icc = np.array(icc)
    print(f"    复合 rank 和 IC(1d)={icc.mean():+.4f} t={icc.mean()/icc.std()*np.sqrt(len(icc)):.1f}")

    print("[3/4] 换手模拟 (FULL / NDROP1 / BUFFER × K=8,10)...")
    results = {}
    for k in [8, 10]:
        for mode in ["FULL", "NDROP1", "BUFFER"]:
            turns, rets = simulate_turnover(dates, comp_rank, fwd_1d, k=k, mode=mode)
            ok = np.isfinite(turns)
            tr = turns[ok]
            rt = rets[ok]
            ann_turn = 240 * tr.mean()          # 年化换手 (倍)
            ann_cost = ann_turn * COST_RT       # 年化成本
            gross_daily = rt.mean()             # 日毛 alpha (相对 0, 非超额)
            gross_ann = gross_daily * 240
            net_ann = gross_ann - ann_cost
            sr_daily = rt.mean() / rt.std() if rt.std() > 0 else 0
            results[(k, mode)] = {
                "turn_pct": round(100 * tr.mean(), 2),
                "turn_median": round(100 * np.median(tr), 2),
                "ann_turn": round(ann_turn, 1),
                "ann_cost_pct": round(100 * ann_cost, 1),
                "gross_ann_pct": round(100 * gross_ann, 1),
                "net_ann_pct": round(100 * net_ann, 1),
                "sharpe_daily_ann": round(sr_daily * np.sqrt(240), 2),
            }

    print("[4/4] 判决表...")
    print("\n" + "-" * 100)
    print(f"{'K':>2} {'模式':<8}{'单边换手':>9}{'年化换手':>9}{'年成本':>8}  "
          f"{'毛alpha':>9}{'净alpha':>9}{'日Sharpe(年化)':>14}")
    print("-" * 100)
    for k in [8, 10]:
        for mode in ["FULL", "NDROP1", "BUFFER"]:
            r = results[(k, mode)]
            print(f"{k:>2} {mode:<8}{r['turn_pct']:>8}%{r['ann_turn']:>8.0f}x{r['ann_cost_pct']:>7.1f}%  "
                  f"{r['gross_ann_pct']:>+8.1f}%{r['net_ann_pct']:>+8.1f}%{r['sharpe_daily_ann']:>14.2f}")
    print("-" * 100)
    print("\n口径说明: 毛 alpha = topK 等权组合 t->t+1 close 收益年化 (未扣池内超额基准, 上界口径)")
    print("         成本 = 年化换手 × 0.72% (佣金万1×2 + 印花税0.05% + 滑点0.3%×2)")
    print("         严格口径 (open-to-open) 毛 alpha 需再打 8~9 折 (P0.5 实证)")

    out = {
        "factor_ic": {f: {"ic": round(v[0], 5), "t": round(v[1], 1)} for f, v in ics.items()},
        "composite_ic_1d": {"ic": round(float(icc.mean()), 5),
                            "t": round(float(icc.mean() / icc.std() * np.sqrt(len(icc))), 1)},
        "simulation": {f"K{k}_{mode}": r for (k, mode), r in results.items()},
        "config": {"K": K, "BUFFER": BUFFER, "COST_RT": COST_RT,
                   "lookback_days": LOOKBACK_DAYS, "n_dates": len(dates)},
    }
    import json
    outp = BASE.parent.parent / "data" / "p1_ndrop_feasibility.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存: {outp} | 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
