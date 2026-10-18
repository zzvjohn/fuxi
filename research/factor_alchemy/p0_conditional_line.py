# -*- coding: utf-8 -*-
"""
P0 条件化因子线 — 分段/乘性条件公式能否逼近树模型交互项 (2026-08-23)
========================================================================
背景: King 同期实验 (2024-2025 +57.42%/0.77 vs LightGBM +195.32%/2.112) 判决
      条件化非线性是真实方法增量。六轮验证已知: 线性因子在超跌池内 IC 显著
      (+0.035/t8.2) 但 top8 优选失败 (Sharpe 0.05), 信息在空头端 (G0 最差)。

P0 假设: 树模型交互项 ("深超跌×缩量企稳×资金回流") 可用 LLM 分段公式逼近。
任务设定对齐 LightGBM: 池 = MA20 偏离最负 1000 只; 问题 = "池内谁反弹" (top8 优选)。

设计:
  - 信号层 (已验证短持有信号): over / lhb / overn5 / intraday
  - 条件层 (慢变量, 控制换手): volr 量比 / dev 偏离深度 / vol20 波动 / ret1/5 跌势
  - 12 条条件化候选 (乘性交互 + if-else 分段)
  - 3 条基线: B0 纯缩量 / B1 线性3因子rank和 / B2 池等权vs全市场
判决: 候选 top8 超额 Sharpe >= 0.3 (基线 0.05) 且 单边日换手 <= 15% 且
      分年 >=3/5 年为正 → 立项 P1 (信号表架构 + NDrop 执行层 → JQ)。

口径: close-to-close, t 收盘信号 → t+1 收盘收益 (与六轮验证基线同口径)。
成本: 0.72%/次全额换手 (佣金万1×2+印花税0.05%+滑点0.3%×2)。
"""
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

from factor_ic_computer import FactorICComputer, RAW_DIR

OUT_JSON = BASE.parent / "data" / "p0_conditional_line.json"
COST_ROUND_TRIP = 0.0072   # 每次全额换手摩擦
POOL_N = 1000
TOP_K = 8


def load_wide_light():
    """轻量宽表加载 (绕开 FactorICComputer 全量 merge 内存墙, 2026-08-23)
    FactorICComputer._load_data 全量 (697万行×73列) merge top_list 需 5.51GiB
    连续内存 → 六轮验证只能跑 1500 自然日 (2022-07 起)。P0 需全区间 2021-2026,
    这里只读 4 个必要字段 + lhb_flag, 宽表 1365×5760 float64 = 4×63MB 无压力。"""
    import gc as _gc
    price_csv = RAW_DIR / "daily_prices.csv"
    df = pd.read_csv(
        price_csv,
        usecols=["ts_code", "trade_date", "open", "close", "vol"],
        low_memory=False,
    )
    df = df.rename(columns={"vol": "volume"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["trade_date"]).drop_duplicates(
        subset=["ts_code", "trade_date"], keep="last")
    close_w = df.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    open_w = df.pivot(index="trade_date", columns="ts_code", values="open").sort_index()
    vol_w = df.pivot(index="trade_date", columns="ts_code", values="volume").sort_index()
    del df
    _gc.collect()
    # lhb_flag: top_list.csv (YYYYMMDD 数字格式)
    lhb_w = None
    tl_path = RAW_DIR / "top_list.csv"
    if tl_path.exists():
        try:
            tl = pd.read_csv(tl_path, usecols=["ts_code", "trade_date"], low_memory=False)
            tl["trade_date"] = pd.to_datetime(
                tl["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
            tl = tl.dropna(subset=["trade_date"])
            agg = tl.groupby(["ts_code", "trade_date"]).size().reset_index(name="lhb_flag")
            lhb_w = (agg.pivot(index="trade_date", columns="ts_code", values="lhb_flag")
                     .sort_index().reindex(index=close_w.index, columns=close_w.columns)
                     .fillna(0.0))
            print(f"  [Light] lhb_flag 宽表: {len(agg)} stock-days "
                  f"(至 {agg['trade_date'].max().date()})")
        except Exception as e:
            print(f"  [Light] ⚠️ top_list load failed: {e}")
    else:
        print("  [Light] ⚠️ top_list.csv 不存在, lhb 因子不可用")
    return close_w, open_w, vol_w, lhb_w


def _rank_cs(w: pd.DataFrame) -> pd.DataFrame:
    """截面百分位排名 (宽表 axis=1, 铁律 2026-08-21: 裸 rank 默认时间轴=前视)"""
    return w.rank(axis=1, pct=True)


def build_base_signals(close_w, open_w, vol_w, lhb_w):
    """信号层 + 条件层宽表"""
    sig = {}
    sig["over"] = -close_w.pct_change(1).rolling(20).max()
    sig["lhb"] = -lhb_w.rolling(20).sum()
    sig["overn5"] = (open_w / close_w.shift(1) - 1.0).rolling(5, min_periods=3).sum()
    sig["intraday"] = -(close_w / open_w - 1.0)

    cond = {}
    ma20 = close_w.rolling(20).mean()
    cond["dev"] = close_w / ma20 - 1.0
    vpos = vol_w.where(vol_w > 0)                     # 停牌日 volume=0 → NaN 防误判缩量
    cond["volr"] = vol_w / vol_w.rolling(20).mean()
    cond["volr"] = cond["volr"].where(vpos > 0)
    cond["vol20"] = close_w.pct_change(1).rolling(20).std()
    cond["ret1"] = close_w.pct_change(1)
    cond["ret5"] = close_w / close_w.shift(5) - 1.0
    return sig, cond


def build_candidates(sig, cond):
    """12 条条件化候选 (乘性交互 + if-else 分段)"""
    over, lhb, overn5, intraday = sig["over"], sig["lhb"], sig["overn5"], sig["intraday"]
    dev, volr, vol20, ret1, ret5 = (cond["dev"], cond["volr"], cond["vol20"],
                                    cond["ret1"], cond["ret5"])
    shrink = (volr < 0.8)                # 缩量企稳 (抛压衰竭)
    deep = (dev < -0.25)                 # 深超跌
    med_vol = vol20.median(axis=1)       # 每日截面波动率中位数 (慢)
    lowvol = vol20.lt(med_vol, axis=0)   # DataFrame-Series 广播必须 axis=0 (铁律)

    def _where(cond_w, a, b):
        """np.where 保持 DataFrame 结构 (index/columns 不丢), b 可为标量"""
        b_arr = b.to_numpy() if hasattr(b, "to_numpy") else b
        return pd.DataFrame(
            np.where(cond_w.to_numpy(), a.to_numpy(), b_arr),
            index=a.index, columns=a.columns)

    cand = {
        "C01_缩量反转":          over * shrink,
        "C02_缩量连续加权":       over * (1.0 - volr.clip(0, 2) / 2.0),
        "C03_缩量且低波":         over * shrink * lowvol,
        "C04_隔夜缩量":          overn5 * shrink,
        "C05_跌势确认反转":       over * (ret5 < 0),
        "C06_深超跌反转":         over * deep,
        "C07_复合缩量":          (over + overn5) * shrink,
        "C08_三段分段":          _where(volr < 0.8, over, _where(volr < 1.2, overn5, -1.0)),
        "C09_深浅分段":          _where(deep, over, overn5),
        "C10_缩量反转爆量罚":     over * shrink - 2.0 * (volr > 1.5),
        "C11_三信号缩量":         (over + overn5 + intraday) * shrink,
        "C12_缩量跌幅收窄":       over * shrink * (ret1 > -0.03),
    }
    # 基线
    base = {
        "B0_纯缩量":      -volr,                       # 最缩量 top8 (无信号纯条件)
        "B1_线性rank和":   (_rank_cs(over) + _rank_cs(lhb) + _rank_cs(overn5)),
        "B2_池等权vs市场": None,                       # 特殊处理: 池均 vs 全市场均
    }
    return cand, base


def run_day_loop(dates, dev, fwd1, cand_wide, base_wide):
    """逐日: 构造超跌池 → 池内评分 → IC / top8超额 / 换手 / 分年"""
    results = {}
    for name in list(cand_wide.keys()) + ["B0_纯缩量", "B1_线性rank和"]:
        results[name] = {
            "ic": [], "top8_excess": [], "turnover": [],
            "yearly": {}, "half": {"first": [], "second": []},
            "pool_ew_vs_mkt": [],
        }
    results["B2_池等权vs市场"] = {
        "ic": [], "top8_excess": [], "turnover": [],
        "yearly": {}, "half": {"first": [], "second": []},
        "pool_ew_vs_mkt": [], "pool_turnover": [],
    }
    n_days = len(dates) - 1
    half_mark = n_days // 2

    prev_top = {k: None for k in results}

    for i in range(n_days):
        dv = dev.iloc[i]
        r = fwd1.iloc[i]
        m = np.isfinite(dv.to_numpy()) & np.isfinite(r.to_numpy())
        if m.sum() < 500:
            continue
        pool = dv[m].sort_values().head(POOL_N).index   # 最超跌 1000 只
        rp = r[pool]
        base_ex = float(rp.mean() - r[m].mean())        # 池 vs 全市场 (beta 部分)
        half = "first" if i < half_mark else "second"

        # 池名单换手 (B2 结构成本背景)
        pset = set(pool)
        prev_pool = getattr(run_day_loop, "prev_pool", None)
        if prev_pool is not None:
            results["B2_池等权vs市场"]["pool_turnover"].append(1 - len(pset & prev_pool) / len(pset))
        run_day_loop.prev_pool = pset
        results["B2_池等权vs市场"]["pool_ew_vs_mkt"].append(base_ex)
        results["B2_池等权vs市场"]["half"][half].append(base_ex)
        y = dates[i].year
        results["B2_池等权vs市场"]["yearly"].setdefault(y, []).append(base_ex)

        for name, cw in cand_wide.items():
            _eval_candidate(results[name], cw.iloc[i], pool, rp, prev_top[name],
                            dates[i].year, half)
        for name in ["B0_纯缩量", "B1_线性rank和"]:
            _eval_candidate(results[name], base_wide[name].iloc[i], pool, rp,
                            prev_top[name], dates[i].year, half)
        # 更新 prev_top
        for name in results:
            if name == "B2_池等权vs市场":
                continue
            cw = cand_wide.get(name, base_wide.get(name))
            fp = cw.iloc[i][pool]
            valid = np.isfinite(fp.to_numpy())
            if valid.sum() >= 100:
                prev_top[name] = set(fp[valid].nlargest(TOP_K).index)
    run_day_loop.prev_pool = None
    return results


def _eval_candidate(res, fp_series, pool, rp, prev_top, year, half):
    fp = fp_series[pool]
    valid = np.isfinite(fp.to_numpy())
    if valid.sum() < 100:
        return
    fp_v = fp[valid]
    rp_v = rp[valid]
    res["ic"].append(float(spearmanr(fp_v.to_numpy(), rp_v.to_numpy())[0]))
    top8 = fp_v.nlargest(TOP_K).index
    res["top8_excess"].append(float(rp_v[top8].mean() - rp_v.mean()))
    if prev_top is not None:
        res["turnover"].append(1.0 - len(set(top8) & prev_top) / TOP_K)
    res["yearly"].setdefault(year, []).append(res["top8_excess"][-1])
    res["half"][half].append(res["top8_excess"][-1])


def summarize(results):
    out = {}
    for name, res in results.items():
        ic = np.array(res["ic"]) if res["ic"] else np.array([np.nan])
        ex = np.array(res["top8_excess"]) if res["top8_excess"] else np.array([np.nan])
        tr = np.array(res["turnover"]) if res["turnover"] else np.array([np.nan])
        n = len(ex)
        ex_mean = float(ex.mean())
        ex_sr = float(ex_mean / ex.std() * np.sqrt(240)) if ex.std() > 0 else 0.0
        ann = 240 * ex_mean
        tr_mean = float(tr.mean()) if len(tr) else 0.0
        cost = 240 * tr_mean * COST_ROUND_TRIP
        yearly = {str(k): float(np.mean(v)) for k, v in res["yearly"].items()}
        pos_years = sum(1 for v in yearly.values() if v > 0)
        n_years = len(yearly)
        icv = np.array([x for x in ic if np.isfinite(x)])
        ic_mean = float(icv.mean()) if len(icv) else np.nan
        ic_t = float(ic_mean / icv.std() * np.sqrt(len(icv))) if len(icv) > 1 and icv.std() > 0 else 0.0
        half1 = np.array(res["half"]["first"])
        half2 = np.array(res["half"]["second"])
        out[name] = {
            "n_days": n,
            "ic": round(ic_mean, 4), "ic_t": round(ic_t, 1),
            "top8_excess_daily": round(100 * ex_mean, 4),
            "ann_excess": round(100 * ann, 2),
            "sharpe": round(ex_sr, 2),
            "turnover": round(100 * tr_mean, 1),
            "ann_cost": round(100 * cost, 1),
            "net_ann": round(100 * (ann - cost), 1),
            "yearly_daily_excess": {k: round(100 * v, 4) for k, v in yearly.items()},
            "positive_years": f"{pos_years}/{n_years}",
            "half_split_sharpe": [round(float(half1.mean() / half1.std() * np.sqrt(240)), 2) if half1.std() > 0 else 0.0,
                                  round(float(half2.mean() / half2.std() * np.sqrt(240)), 2) if half2.std() > 0 else 0.0],
        }
    # B2 池换手
    ptr = np.array(results["B2_池等权vs市场"]["pool_turnover"])
    if len(ptr):
        out["B2_池等权vs市场"]["pool_turnover"] = round(100 * float(ptr.mean()), 1)
        out["B2_池等权vs市场"]["pool_ann_cost"] = round(100 * 240 * float(ptr.mean()) * COST_ROUND_TRIP, 1)
    return out


def main():
    print("=" * 64)
    print("  P0 条件化因子线 — 分段公式 × 超跌池 top8 优选")
    print("=" * 64)
    close_w, open_w, vol_w, lhb_w = load_wide_light()
    if lhb_w is None:
        raise SystemExit("  [FATAL] lhb_flag 宽表不可用, 中止 (lhb 为信号层核心)")
    dates = close_w.index
    print(f"  宽表: {len(dates)} 交易日 | {close_w.shape[1]} 只 | "
          f"{dates[0].date()} ~ {dates[-1].date()}")
    fwd1 = close_w.shift(-1) / close_w - 1.0

    sig, cond = build_base_signals(close_w, open_w, vol_w, lhb_w)
    cand, base = build_candidates(sig, cond)

    print(f"  候选 {len(cand)} 条 + 基线 {len(base)} 条, 池 {POOL_N} 只 / top{TOP_K}")
    results = run_day_loop(dates, cond["dev"], fwd1, cand, base)
    summary = summarize(results)

    print("\n  {:14s} {:>7s} {:>7s} {:>9s} {:>7s} {:>8s} {:>9s} {:>9s} {:>6s}".format(
        "候选", "IC", "t", "ann超额%", "Sharpe", "换手%", "年成本%", "净年化%", "正年"))
    print("  " + "-" * 80)
    for name in list(cand.keys()) + ["B0_纯缩量", "B1_线性rank和", "B2_池等权vs市场"]:
        s = summary[name]
        print("  {:14s} {:>7.4f} {:>7.1f} {:>9.2f} {:>7.2f} {:>8.1f} {:>9.1f} {:>9.1f} {:>6s}".format(
            name, s["ic"], s["ic_t"], s["ann_excess"], s["sharpe"],
            s.get("turnover", 0), s.get("ann_cost", 0), s.get("net_ann", 0),
            s["positive_years"]))
    print("\n  分年 top8 日超额 (%):")
    for name in list(cand.keys()) + ["B0_纯缩量", "B1_线性rank和"]:
        s = summary[name]
        yrs = " ".join(f"{k}:{v:+.3f}" for k, v in s["yearly_daily_excess"].items())
        print(f"    {name:14s} {yrs}")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "pool_n": POOL_N, "top_k": TOP_K,
        "cost_round_trip": COST_ROUND_TRIP,
        "window": [str(dates[0].date()), str(dates[-1].date())],
        "summary": summary,
    }
    json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print(f"\n  结果已存: {OUT_JSON}")


if __name__ == "__main__":
    main()
