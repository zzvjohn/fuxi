# -*- coding: utf-8 -*-
"""
快刀斩股 skill -> 因子提炼 实证探针 (2026-09-11)

目的: 把 kuaidao-stock-selection 的三层判定 (排雷/价格门控/十项打分) 中
      可量化的条目落成 pandas 因子, 并在全 A 截面上做 PIT 严格对齐的
      月度 IC / ICIR 检验, 回答 "这个 skill 能提炼出什么因子"。

铁律遵守:
 - 公告类数据 (income/bal/fina/dividend/holdertrade/repurchase/holdernumber/top10)
   一律用 ann_date <= t 做时点对齐 (绝不用 end_date), 防前视。
 - daily_basic 面板先 drop_duplicates(['ts_code','trade_date']) (工作区已验证的重复行坑)。
 - 截面 rank 一律 axis=1 / 逐列滚动, 不写净值, 只信 IC/ICIR。

输出: data/kuaidao_test/factor_probe_results.json + 控制台表格
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FM = ROOT / "data" / "kuaidao_test" / "full_market"
OUT = ROOT / "data" / "kuaidao_test" / "factor_probe_results.json"

FWD = 20            # 前瞻交易日
START = "20240131"  # 首个调仓日
IC_MIN = 0.02       # 参考伏羲 v0.7 daily lane 门槛
ICIR_MIN = 0.10


def log(*a):
    print(*a, flush=True)


def build_panel():
    log("[1/6] 读 daily_basic 面板 ...")
    t0 = time.time()
    db = pd.read_csv(FM / "full_db_daily.csv", dtype={"ts_code": str, "trade_date": str})
    n0 = len(db)
    db = db.drop_duplicates(["ts_code", "trade_date"])
    log(f"    daily_basic {n0:,} -> 去重 {len(db):,} 行  ({time.time()-t0:.1f}s)")

    try:
        adj = pd.read_csv(FM / "full_adj_factor.csv", dtype={"ts_code": str, "trade_date": str})
        adj = adj.drop_duplicates(["ts_code", "trade_date"])
        db = db.merge(adj, on=["ts_code", "trade_date"], how="left")
        db["close_adj"] = db["close"] * db["adj_factor"].fillna(1.0)
        log(f"    adj_factor 已并入, 非空 {db['adj_factor'].notna().mean():.1%}")
    except Exception as e:
        log(f"    [WARN] adj_factor 不可用({e}) -> 用原始 close")
        db["close_adj"] = db["close"]

    db = db[(db["close"] > 0) & (db["close_adj"] > 0) & (db["total_mv"] > 0)]
    dates = np.sort(db["trade_date"].unique())
    reb = [d for d in dates if d >= START]
    reb = pd.Series(reb).groupby(pd.Series(reb).str[:6]).max().tolist()
    idx = {d: i for i, d in enumerate(dates)}
    reb = [d for d in reb if d in idx and idx[d] + FWD < len(dates)]
    log(f"    调仓月 {len(reb)} 个: {reb[0]} ~ {reb[-1]}")

    pan = {}
    for col in ["close_adj", "total_mv", "pb", "pe_ttm", "ps_ttm"]:
        pan[col] = db.pivot_table(index="trade_date", columns="ts_code",
                                  values=col, aggfunc="last").reindex(dates).astype("float64")
    pan["ps_pct_own"] = pan["ps_ttm"].rolling(250, min_periods=120).rank(pct=True)
    log(f"    面板 {pan['close_adj'].shape[0]} 日 x {pan['close_adj'].shape[1]} 股")
    return pan, dates, reb


def load_per_stock(kind):
    log(f"[2/6] 读 per_stock_{kind} ...")
    t0 = time.time()
    fs = sorted((FM / f"per_stock_{kind}").glob("*.csv"))
    frames = []
    for f in fs:
        try:
            frames.append(pd.read_csv(f, dtype={"ts_code": str, "ann_date": str, "end_date": str}))
        except Exception:
            pass
    df = pd.concat(frames, ignore_index=True)
    df = df[df["ann_date"].notna() & (df["ann_date"] != "")]
    df = df.drop_duplicates(["ts_code", "end_date", "ann_date"])
    log(f"    {len(fs)} 文件 -> {len(df):,} 行 ({time.time()-t0:.1f}s)")
    return df


def add_yoy(df, valcol):
    d = df.dropna(subset=[valcol]).sort_values(["ts_code", "end_date"]).copy()
    d[f"{valcol}_yoy"] = d.groupby("ts_code")[valcol].pct_change(4)
    return df.merge(d[["ts_code", "end_date", "ann_date", f"{valcol}_yoy"]],
                    on=["ts_code", "end_date", "ann_date"], how="left")


def pit(df, t):
    d = df[df["ann_date"] <= t]
    if d.empty:
        return df.iloc[0:0]
    return d.sort_values(["ts_code", "ann_date"]).groupby("ts_code").last()


def main():
    pan, dates, reb = build_panel()
    close = pan["close_adj"]
    fwd = close.shift(-FWD) / close - 1.0

    inc = load_per_stock("income")
    bal = load_per_stock("bal")
    fin = load_per_stock("fina")
    inc = add_yoy(inc, "total_revenue")
    inc = add_yoy(inc, "n_income")
    bal = add_yoy(bal, "accounts_receiv")

    log("[3/6] 读 dividend / 增减持 / 回购 / 股东户数 / 前十大 / 行业 ...")
    div = pd.read_csv(FM / "full_div.csv", dtype={"ts_code": str, "end_date": str, "ann_date": str})
    trd = pd.read_csv(FM / "full_trades.csv", dtype={"ts_code": str, "ann_date": str},
                      on_bad_lines="skip", engine="python")
    rep = pd.read_csv(FM / "full_rep.csv", dtype={"ts_code": str, "ann_date": str})
    hold = pd.read_csv(FM / "full_holder.csv", dtype={"ts_code": str, "ann_date": str, "end_date": str})
    t10 = pd.read_csv(FM / "full_top10.csv", dtype={"ts_code": str, "ann_date": str, "end_date": str})
    log(f"    div {len(div):,} / trades {len(trd):,} / rep {len(rep):,} / "
        f"holder {len(hold):,} / top10 {len(t10):,}")
    sw = pd.read_csv(FM / "full_sw.csv", dtype={"ts_code": str})
    sw = sw[sw["is_new"] == "Y"].drop_duplicates("ts_code").set_index("ts_code")["l3_name"]
    sw_l1 = pd.read_csv(FM / "full_sw.csv", dtype={"ts_code": str})
    sw_l1 = sw_l1[sw_l1["is_new"] == "Y"].drop_duplicates("ts_code").set_index("ts_code")["l1_name"]

    trd["change_vol"] = pd.to_numeric(trd["change_vol"], errors="coerce")
    trd["avg_price"] = pd.to_numeric(trd["avg_price"], errors="coerce")
    trd["_val"] = trd["change_vol"] * trd["avg_price"]
    trd["_in"] = trd["in_de"].astype(str).str.upper().eq("IN")
    trd["_de"] = trd["in_de"].astype(str).str.upper().eq("DE")
    rep["_amt"] = pd.to_numeric(rep["amount"], errors="coerce")

    rows = []
    log(f"[4/6] 逐月计算因子 ... ({len(reb)} 个调仓日)")
    t0 = time.time()
    for k, t in enumerate(reb):
        d = pit(inc, t)
        b = pit(bal, t)
        f = pit(fin, t)
        mv = pan["total_mv"].loc[t] * 1e4                      # 万元 -> 元
        XL = mv.index
        X = pd.DataFrame(index=XL)

        def A(s):
            """对齐到截面索引"""
            if not isinstance(s, pd.Series):
                return pd.Series(s, index=XL)
            return pd.to_numeric(s, errors="coerce").reindex(XL)

        pb, pe, ps = A(pan["pb"].loc[t]), A(pan["pe_ttm"].loc[t]), A(pan["ps_ttm"].loc[t])
        cl = A(close.loc[t])
        roe = A(f["roe"])
        gm, nm = A(f["grossprofit_margin"]), A(f["netprofit_margin"])
        ta = A(b["total_assets"])
        rev = A(d["total_revenue"])
        ni = A(d["n_income"])
        ts = A(b["total_share"])

        # ── 估值层 (价格门控) ───────────────────────────────
        X["ep_ttm"] = (1.0 / pe).where(pe > 0)
        X["bp"] = (1.0 / pb).where(pb > 0)
        X["sp_ttm"] = (1.0 / ps).where(ps > 0)
        X["ps_own_pct_low"] = -A(pan["ps_pct_own"].loc[t])
        # PR = 100*PB/ROE^2 (ROE 为百分数); 取倒数使"越大越便宜"
        X["pr_inverse"] = (roe.pow(2) / (100.0 * pb)).where((pb > 0) & (roe > 0))

        # ── 质量层 (s2 / s7) ────────────────────────────────
        X["roe"] = roe
        X["gm"] = gm
        X["nm"] = nm
        ind = XL.to_series().map(sw)
        for c in ["gm", "nm", "roe"]:
            X[f"{c}_ind_rank"] = X[c].groupby(ind.values).rank(pct=True)

        # ── 资产负债 (s3) ──────────────────────────────────
        X["net_cash_to_mv"] = ((A(b["money_cap"]) + A(b["trad_asset"]).fillna(0)
                                - A(b["st_borr"]).fillna(0) - A(b["lt_borr"]).fillna(0)
                                - A(b["bond_payable"]).fillna(0)
                                - A(b["non_cur_liab_due_1y"]).fillna(0)) / mv)
        X["goodwill_to_assets"] = -(A(b["goodwill"]).fillna(0) / ta)
        X["asset_heaviness"] = ((A(b["fix_assets"]).fillna(0) + A(b["cip"]).fillna(0)
                                 + A(b["intan_assets"]).fillna(0)) / ta)
        X["ar_to_revenue"] = -(A(b["accounts_receiv"]) / rev)
        X["advance_receipt_ratio"] = ((A(b["contract_liab"]).fillna(0)
                                       + A(b["adv_receipts"]).fillna(0)) / rev)

        # ── 成长 (s1) ──────────────────────────────────────
        X["rev_yoy"] = A(d["total_revenue_yoy"])
        X["ni_yoy"] = A(d["n_income_yoy"])
        X["impair_to_ni"] = -(A(d["assets_impair_loss"]).abs() / ni.abs())

        # ── 分红 (s5) ──────────────────────────────────────
        w = div[(div["ann_date"] <= t) & (div["ann_date"] > str(int(t[:4]) - 1) + t[4:])]
        cd = pd.to_numeric(w["cash_div"], errors="coerce").fillna(0)
        dv = cd.groupby(w["ts_code"]).max().reindex(XL).fillna(0)
        X["div_yield"] = (dv * ts) / mv
        X["payout_ratio"] = ((dv * ts) / ni).where(ni > 0)

        # ── 股权变动 (s8) ──────────────────────────────────
        w = trd[(trd["ann_date"] <= t) & (trd["ann_date"] > str(int(t[:4]) - 1) + t[4:])]
        X["insider_net_buy"] = (w[w["_in"]].groupby("ts_code")["_val"].sum().reindex(XL).fillna(0)
                                - w[w["_de"]].groupby("ts_code")["_val"].sum().reindex(XL).fillna(0)) / mv
        w = rep[(rep["ann_date"] <= t) & (rep["ann_date"] > str(int(t[:4]) - 1) + t[4:])]
        X["buyback_to_mv"] = w.groupby("ts_code")["_amt"].sum().reindex(XL).fillna(0) / mv

        # ── 筹码 (s9) ──────────────────────────────────────
        hn = A(pit(hold, t)["holder_num"])
        X["per_capita_value"] = (((ts - 0) * cl) / hn).where(hn > 0)   # 分母修正见下
        t10c = t10[t10["ann_date"] <= t]
        if len(t10c):
            g = t10c.groupby(["ts_code", "end_date"])["hold_amount"].sum()
            top_sum = g.groupby(level=0).last().reindex(XL).fillna(0)
            X["per_capita_value"] = (((ts - top_sum) * cl) / hn).where(hn > 0)
        hh = hold[hold["ann_date"] <= t].sort_values(["ts_code", "end_date"])
        hh["_h4"] = hh.groupby("ts_code")["holder_num"].shift(4)
        last = hh.groupby("ts_code").last()
        X["holder_num_yoy"] = -(A(last["holder_num"]) / A(last["_h4"]) - 1.0)

        blk = X.copy()
        blk["_fwd"] = fwd.loc[t].reindex(XL)
        blk["_ind"] = XL.to_series().map(sw_l1).values
        rows.append(blk)
        if (k + 1) % 6 == 0 or k == len(reb) - 1:
            log(f"    {k+1}/{len(reb)}  {t}  ({time.time()-t0:.0f}s)")

    log("[5/6] 逐月截面 Spearman IC (全A / 行业中性 / 门控后池内) ...")

    def ic_series(fc, mode, min_n):
        ics, ns = [], []
        for blk in rows:
            s, r = blk[fc], blk["_fwd"]
            if mode == "ind":
                s = s - s.groupby(blk["_ind"]).transform("mean")      # 行业中性(去行业均值)
            m = s.notna() & r.notna()
            if mode == "prgate":
                m &= blk["pr_inverse"] >= (1.0 / 1.5)                  # PR <= 1.5 简化价格门控
            if mode == "psgate":
                # ps_own_pct_low 已取负, <= -0.5 即 ps_ttm 处自身 250 日分位下半区(便宜半区)
                m &= blk["ps_own_pct_low"] <= -0.5
            if m.sum() < min_n:
                continue
            a, b = s[m].rank(), r[m].rank()
            if a.std() == 0 or b.std() == 0:
                continue
            ics.append(a.corr(b))
            ns.append(int(m.sum()))
        return pd.Series(ics).dropna(), ns

    def summarize(fc, ics, ns):
        if len(ics) < 8:
            return None
        sd = ics.std(ddof=1)
        return {"ic_mean": float(ics.mean()), "icir": float(ics.mean() / sd) if sd > 0 else None,
                "t_stat": float(ics.mean() / sd * np.sqrt(len(ics))) if sd > 0 else None,
                "pos_ic_pct": float((ics > 0).mean()),
                "avg_n": int(np.mean(ns)), "n_months": int(len(ics))}

    factors = [c for c in rows[0].columns if not c.startswith("_")]
    recs, cond = {}, {}
    for fc in factors:
        for mode, mn in [("all", 200), ("ind", 200), ("prgate", 120), ("psgate", 120)]:
            ics, ns = ic_series(fc, mode, mn)
            r = summarize(fc, ics, ns[:len(ics)])
            if r:
                if mode == "all":
                    recs[fc] = r
                else:
                    cond.setdefault(fc, {})[mode] = r

    log(f"    因子数 recs={len(recs)} cond={len(cond)}")
    out = pd.DataFrame.from_dict(recs, orient="index")
    if out.empty:
        raise SystemExit("所有因子 IC 样本不足, 检查数据")
    out["abs_icir"] = out["icir"].abs()
    out = out.sort_values("abs_icir", ascending=False)
    out["pass"] = (out["ic_mean"].abs() >= IC_MIN) & (out["abs_icir"] >= ICIR_MIN)

    ctab = []
    for fc, base in recs.items():
        dd = cond.get(fc, {})
        ctab.append({
            "factor": fc,
            "IC_all": base["ic_mean"], "ICIR_all": base["icir"],
            "IC_indn": dd.get("ind", {}).get("ic_mean"), "ICIR_indn": dd.get("ind", {}).get("icir"),
            "IC_prgate": dd.get("prgate", {}).get("ic_mean"), "ICIR_prgate": dd.get("prgate", {}).get("icir"),
            "n_prgate": dd.get("prgate", {}).get("avg_n"),
            "IC_psgate": dd.get("psgate", {}).get("ic_mean"), "ICIR_psgate": dd.get("psgate", {}).get("icir"),
            "n_psgate": dd.get("psgate", {}).get("avg_n"),
        })
    cdf = pd.DataFrame(ctab)
    if cdf.empty:
        raise SystemExit("条件化表为空")
    cdf = cdf.set_index("factor")
    cdf["dIC_prgate"] = cdf["IC_prgate"] - cdf["IC_all"]
    cdf = cdf.sort_values("IC_prgate", key=lambda s: s.abs(), ascending=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    log("\n" + "=" * 100)
    log(f"[表1] 全A 月度 Rank IC/ICIR (fwd{FWD}, {reb[0]}~{reb[-1]}, {len(reb)} 个月)")
    log("=" * 100)
    log(out[["n_months", "avg_n", "ic_mean", "icir", "t_stat", "pos_ic_pct", "pass"]]
        .to_string(float_format=lambda v: f"{v:+.4f}"))
    log("\n" + "=" * 120)
    log("[表2] 条件化检验: 全A vs 行业中性 vs 快刀价格门控后池内 (IC 均值 / 均值样本量)")
    log("=" * 120)
    log(cdf.to_string(float_format=lambda v: f"{v:+.4f}"))
    log("=" * 120)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "window": [reb[0], reb[-1]], "n_months": len(reb), "fwd_days": FWD,
        "universe": "全A (daily_basic), 后复权", "thresholds": {"ic": IC_MIN, "icir": ICIR_MIN},
        "factors": recs, "conditional": cond,
        "conditional_table": cdf.reset_index().to_dict("records"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[6/6] 落盘 {OUT}")


if __name__ == "__main__":
    t = time.time()
    main()
    log(f"总耗时 {time.time()-t:.0f}s")
