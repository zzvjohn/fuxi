# -*- coding: utf-8 -*-
"""
快刀斩股 skill -> 因子提炼 v2 (2026-09-11)

v1 的错误: 市赚率 PR 用了 pr_inverse = ROE^2/(100*PB) —— 这是 **PE_TTM 口径**,
           既没做净现金调整, 也没用预估净利润。

v2 修正口径 (与 skill references/gating.md 一致):
    净现金     = 总现金(货币资金) + 交易性金融资产 - 有息负债
                (有息负债 = 短借 + 长借 + 应付债券 + 一年内到期非流动负债)
    调整后市值 = 总市值 - 净现金
    预估净利润 = 去年年报净利润 x (1 + 今年增速)
    增长市盈率 = 调整后市值 / 预估净利润
    PR         = 增长市盈率 / ROE(%)          <- 越低越便宜
    PS         = 调整后市值 / 营业收入        <- 非稳定盈利公司用

同时把十项打分落成 10 个 s1~s10 分项分, 直接检验"总分"能否排序。

铁律: 公告表一律 ann_date <= t 时点对齐; daily_basic 先 drop_duplicates;
      截面 rank 逐列, 不写净值, 只信 IC/ICIR。

输出: data/kuaidao_test/factor_probe_v2_results.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FM = ROOT / "data" / "kuaidao_test" / "full_market"
OUT = ROOT / "data" / "kuaidao_test" / "factor_probe_v2_results.json"

FWD = 20
START = "20240131"
IC_MIN, ICIR_MIN = 0.02, 0.10
PR_GATE = 1.5          # PR > 1.5 淘汰 (2026-09-11 用户钦定: 门槛=1.5; 曾试 2.0 已回退)
PS_GATE = 0.70         # PS 自身历史分位 > 70% 淘汰


def log(*a):
    print(*a, flush=True)


# ──────────────────────────────────────────────────────────
def build_panel():
    log("[1/7] 读 daily_basic 面板 ...")
    t0 = time.time()
    db = pd.read_csv(FM / "full_db_daily.csv", dtype={"ts_code": str, "trade_date": str})
    n0 = len(db)
    db = db.drop_duplicates(["ts_code", "trade_date"])
    log(f"    daily_basic {n0:,} -> 去重 {len(db):,}")
    try:
        adj = pd.read_csv(FM / "full_adj_factor.csv", dtype={"ts_code": str, "trade_date": str})
        adj = adj.drop_duplicates(["ts_code", "trade_date"])
        db = db.merge(adj, on=["ts_code", "trade_date"], how="left")
        db["close_adj"] = db["close"] * db["adj_factor"].fillna(1.0)
    except Exception as e:
        log(f"    [WARN] adj_factor 不可用({e})")
        db["close_adj"] = db["close"]
    db = db[(db["close"] > 0) & (db["close_adj"] > 0) & (db["total_mv"] > 0)]
    dates = np.sort(db["trade_date"].unique())
    reb = [d for d in dates if d >= START]
    reb = pd.Series(reb).groupby(pd.Series(reb).str[:6]).max().tolist()
    idx = {d: i for i, d in enumerate(dates)}
    reb = [d for d in reb if idx[d] + FWD < len(dates)]
    pan = {}
    for col in ["close_adj", "total_mv", "pb", "pe_ttm", "ps_ttm"]:
        pan[col] = db.pivot_table(index="trade_date", columns="ts_code",
                                  values=col, aggfunc="last").reindex(dates).astype("float64")
    # PS 自身历史分位 (250 日 > 70% = 高估)
    pan["ps_pct_own"] = pan["ps_ttm"].rolling(250, min_periods=120).rank(pct=True)
    log(f"    面板 {pan['close_adj'].shape[0]} 日 x {pan['close_adj'].shape[1]} 股, 调仓 {len(reb)} 月")
    return pan, dates, reb


def load_per_stock(kind):
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
    # 缓存有 ~22% 完全重复行 (per-stock 文件自带重复) -> 先去完全重复, 再按主键去重
    df = df.drop_duplicates().drop_duplicates(["ts_code", "end_date", "ann_date"])
    log(f"    per_stock_{kind}: {len(fs)} 文件 -> {len(df):,} 行 ({time.time()-t0:.0f}s)")
    return df


def num(s, idx):
    if not isinstance(s, pd.Series):
        return pd.Series(s, index=idx)
    return pd.to_numeric(s, errors="coerce").reindex(idx)


def pit(df, t):
    d = df[df["ann_date"] <= t]
    if d.empty:
        return df.iloc[0:0]
    return d.sort_values(["ts_code", "ann_date"]).groupby("ts_code").last()


# ──────────────────────────────────────────────────────────
def main():
    pan, dates, reb = build_panel()
    close = pan["close_adj"]
    fwd = close.shift(-FWD) / close - 1.0

    log("[2/7] 读三表 ...")
    inc = load_per_stock("income")
    bal = load_per_stock("bal")
    fin = load_per_stock("fina")

    # 年报切片 + 同比
    def annual(df, cols):
        a = df[df["end_date"].str[4:] == "1231"].copy()
        a = a.dropna(subset=cols).sort_values(["ts_code", "end_date"])
        for c in cols:
            a[f"{c}_yoy_a"] = a.groupby("ts_code")[c].pct_change()
        return a

    inc_a = annual(inc, ["total_revenue", "n_income"])
    fin_a = annual(fin, ["roe"])
    bal_a = annual(bal, ["accounts_receiv"])

    # 最新同比 (任意期) —— 作为 "今年增速" 代理
    inc_all = inc.dropna(subset=["n_income", "total_revenue"]).sort_values(["ts_code", "end_date"]).copy()
    inc_all["ni_yoy_last"] = inc_all.groupby("ts_code")["n_income"].pct_change(4)
    inc_all["rev_yoy_last"] = inc_all.groupby("ts_code")["total_revenue"].pct_change(4)

    log("[3/7] 读公告表 ...")
    div = pd.read_csv(FM / "full_div.csv", dtype={"ts_code": str, "end_date": str, "ann_date": str})
    trd = pd.read_csv(FM / "full_trades.csv", dtype={"ts_code": str, "ann_date": str},
                      on_bad_lines="skip", engine="python")
    rep = pd.read_csv(FM / "full_rep.csv", dtype={"ts_code": str, "ann_date": str})
    hold = pd.read_csv(FM / "full_holder.csv", dtype={"ts_code": str, "ann_date": str, "end_date": str})
    t10 = pd.read_csv(FM / "full_top10.csv", dtype={"ts_code": str, "ann_date": str, "end_date": str})
    swf = pd.read_csv(FM / "full_sw.csv", dtype={"ts_code": str})
    swf = swf[swf["is_new"] == "Y"].drop_duplicates("ts_code").set_index("ts_code")
    sw_l3, sw_l1 = swf["l3_name"], swf["l1_name"]

    trd["change_vol"] = pd.to_numeric(trd["change_vol"], errors="coerce")
    trd["avg_price"] = pd.to_numeric(trd["avg_price"], errors="coerce")
    trd["_val"] = trd["change_vol"] * trd["avg_price"]
    trd["_de"] = trd["in_de"].astype(str).str.upper().eq("DE")
    trd["_in"] = trd["in_de"].astype(str).str.upper().eq("IN")
    rep["_amt"] = pd.to_numeric(rep["amount"], errors="coerce")
    div["_cd"] = pd.to_numeric(div["cash_div"], errors="coerce")

    rows = []
    log(f"[4/7] 逐月计算 ... ({len(reb)} 月)")
    t0 = time.time()
    for k, t in enumerate(reb):
        d = pit(inc, t)          # 最新报告期 (TTM 近似用累计, 仅取估值用)
        b = pit(bal, t)
        f = pit(fin, t)
        mv = pan["total_mv"].loc[t] * 1e4              # 万元 -> 元
        XL = mv.index
        X = pd.DataFrame(index=XL)
        A = lambda s: num(s, XL)

        pb, pe, ps = A(pan["pb"].loc[t]), A(pan["pe_ttm"].loc[t]), A(pan["ps_ttm"].loc[t])
        cl = A(close.loc[t])
        roe = A(f["roe"])
        rev = A(d["total_revenue"])
        ni = A(d["n_income"])
        ts = A(b["total_share"])

        # ── 公用公式: 净现金 / 调整后市值 ───────────────────
        cash = A(b["money_cap"])
        trad = A(b["trad_asset"]).fillna(0)
        ib = (A(b["st_borr"]).fillna(0) + A(b["lt_borr"]).fillna(0)
              + A(b["bond_payable"]).fillna(0) + A(b["non_cur_liab_due_1y"]).fillna(0))
        net_cash = cash + trad - ib
        adj_mv = mv - net_cash
        X["net_cash_to_mv"] = (net_cash / mv).where(mv > 0)
        X["adj_mv_ratio"] = (adj_mv / mv).where(mv > 0)

        # ── 去年年报净利 + 今年增速 -> 预估净利 ─────────────
        ia = inc_a[inc_a["ann_date"] <= t].sort_values(["ts_code", "end_date"])
        last_ann = ia.groupby("ts_code").last()
        last_np = A(last_ann["n_income"])
        # 近3年报 (s1/s2 用)
        last3 = ia.groupby("ts_code").tail(3)
        n3 = last3.groupby("ts_code").size()
        pos_rev = last3.assign(p=(last3["total_revenue_yoy_a"] > 0).astype(float)).groupby("ts_code")["p"].sum()
        pos_ni = last3.assign(p=(last3["n_income_yoy_a"] > 0).astype(float)).groupby("ts_code")["p"].sum()
        loss3 = last3.assign(p=(last3["n_income"] < 0).astype(float)).groupby("ts_code")["p"].sum()
        rev_grow3 = A((pos_rev == 3) & (n3 == 3)).fillna(0)
        ni_grow3 = A((pos_ni == 3) & (n3 == 3)).fillna(0)
        loss_2y = A((loss3 >= 2)).fillna(0)
        stable = A((loss3 == 0) & (n3 == 3)).fillna(0)          # 近3年报净利全 > 0

        # 今年增速: 最新期净利同比, 缺失回落年报同比
        li = inc_all[inc_all["ann_date"] <= t].sort_values(["ts_code", "end_date"]).groupby("ts_code").last()
        g = A(li["ni_yoy_last"])
        g_ann = A(last_ann["n_income_yoy_a"])
        g = g.where(g.notna(), g_ann)
        g = g.clip(-0.9, 3.0)                                    # 增速限幅, 防爆
        est_np = last_np * (1.0 + g)

        growth_pe = (adj_mv / est_np).where((est_np > 0) & (adj_mv > 0))
        X["growth_pe_inv"] = (est_np / adj_mv).where((est_np > 0) & (adj_mv > 0))   # 越高越便宜
        X["est_np_yield_mv"] = (est_np / mv).where(est_np > 0)                      # 未调整口径, 分解用
        X["pr"] = (growth_pe / roe).where((roe > 0) & growth_pe > 0)
        X["pr_inv_v2"] = (roe / growth_pe).where((roe > 0) & growth_pe > 0)         # 修正后 PR 倒数
        # v1 旧口径 (PE_TTM 版), 保留对照
        X["pr_inv_v1"] = (roe.pow(2) / (100.0 * pb)).where((pb > 0) & (roe > 0))

        # ── 估值族 (对照) ──────────────────────────────────
        X["ep_ttm"] = (1.0 / pe).where(pe > 0)
        X["bp"] = (1.0 / pb).where(pb > 0)
        X["sp_ttm"] = (1.0 / ps).where(ps > 0)
        X["adj_sp_inv"] = (rev / adj_mv).where(adj_mv > 0)      # 调整后市销率倒数
        X["ps_pct_own_low"] = -A(pan["ps_pct_own"].loc[t])

        # ── s1 收入利润 ────────────────────────────────────
        X["rev_grow3"] = rev_grow3
        X["ni_grow3"] = ni_grow3
        X["loss_2y"] = loss_2y
        X["s1_score"] = np.where((rev_grow3 == 1) & (ni_grow3 == 1), 1.0,
                                 np.where(loss_2y == 1, -1.0, 0.0))
        X["rev_yoy_last"] = A(li["rev_yoy_last"])
        X["ni_yoy_last"] = A(li["ni_yoy_last"])

        # ── s2 ROE ─────────────────────────────────────────
        fa = fin_a[fin_a["ann_date"] <= t].sort_values(["ts_code", "end_date"])
        f3 = fa.groupby("ts_code").tail(3)
        roe_min3 = A(f3.groupby("ts_code")["roe"].min())
        roe_cnt3 = A(f3.groupby("ts_code")["roe"].size())
        X["roe_min3"] = roe_min3
        X["s2_score"] = np.where((roe_min3 >= 15) & (roe_cnt3 == 3), 1.0,
                                 np.where((roe_min3 < 10) & (roe_cnt3 == 3), -1.0, 0.0))
        X["roe"] = roe

        # ── s3 资产负债表 (a∧b∧c -> +1; 任意两项不满足 -> -1) ─
        ar = A(b["accounts_receiv"])
        ta = A(b["total_assets"])
        gd = A(b["goodwill"]).fillna(0)
        a_ok = ((ar / rev <= 0.20) | ((A(b["contract_liab"]).fillna(0) + A(b["adv_receipts"]).fillna(0)) / rev >= 0.20))
        b_ok = (net_cash / (cash + trad) > 0.80).where((cash + trad) > 0)
        c_ok = (gd / ta <= 0.05)
        cnt = (a_ok.fillna(False).astype(int) + b_ok.fillna(False).astype(int) + c_ok.fillna(False).astype(int))
        X["s3_score"] = np.where(cnt == 3, 1.0, np.where(cnt <= 1, -1.0, 0.0))
        X["ar_to_revenue"] = -(ar / rev)
        X["advance_receipt_ratio"] = ((A(b["contract_liab"]).fillna(0) + A(b["adv_receipts"]).fillna(0)) / rev)
        X["net_cash_ratio"] = (net_cash / (cash + trad)).where((cash + trad) > 0)
        X["goodwill_to_assets"] = -(gd / ta)

        # ── s4 现金流量表: 缓存缺现金流量表 -> 全 N/A ───────
        X["s4_score"] = np.nan

        # ── s5 股东回报 ────────────────────────────────────
        dv = div[div["ann_date"] <= t].copy()
        dv["yr"] = dv["end_date"].str[:4]
        yrs = [str(int(t[:4]) - i) for i in (1, 2, 3)]
        dvw = dv[dv["yr"].isin(yrs) & (dv["_cd"] > 0)]
        nyears = dvw.groupby("ts_code")["yr"].nunique().reindex(XL).fillna(0)
        any_div3 = dv.groupby("ts_code")["_cd"].max()
        dmax = dvw.groupby("ts_code")["_cd"].max().reindex(XL).fillna(0)
        payout = ((dmax * ts) / ni).where(ni > 0)
        X["div_yield"] = (dmax * ts) / mv
        X["payout_ratio"] = payout
        X["s5_score"] = np.where((nyears == 3) & ((payout.isna()) | (payout <= 1.0)), 1.0,
                                 np.where(nyears == 0, -1.0, 0.0))

        # ── s7 龙头 (gm/nm/roe 至少2项行业内 top3) ──────────
        gm, nm = A(f["grossprofit_margin"]), A(f["netprofit_margin"])
        X["gm"], X["nm"] = gm, nm
        ind3 = XL.to_series().map(sw_l3)
        def top3(s):
            g = s.groupby(ind3.values)
            n = g.transform("size")
            r = g.rank(ascending=False, method="min")
            thr = np.where(n < 10, 3, np.maximum(3, np.ceil(n * 0.03)))
            return (r <= thr).where(s.notna(), other=False)
        hits = (top3(gm).astype(float).fillna(0) + top3(nm).astype(float).fillna(0)
                + top3(roe).astype(float).fillna(0))
        X["s7_score"] = np.where(hits >= 2, 1.0, 0.0)
        for c, s in [("gm", gm), ("nm", nm), ("roe", roe)]:
            X[f"{c}_ind_rank"] = s.groupby(ind3.values).rank(pct=True)

        # ── s8 股权变动 ────────────────────────────────────
        yr_lo = str(int(t[:4]) - 1) + t[4:]
        w = trd[(trd["ann_date"] <= t) & (trd["ann_date"] > yr_lo)]
        ins = (w[w["_in"]].groupby("ts_code")["_val"].sum().reindex(XL).fillna(0)
               - w[w["_de"]].groupby("ts_code")["_val"].sum().reindex(XL).fillna(0))
        X["insider_net_buy"] = ins / mv
        wr = rep[(rep["ann_date"] <= t) & (rep["ann_date"] > yr_lo)]
        bb = wr.groupby("ts_code")["_amt"].sum().reindex(XL).fillna(0) / mv
        X["buyback_to_mv"] = bb
        a_pos = ((bb >= 0.01) | (X["insider_net_buy"] >= 0.0003)).astype(float)
        c_neg = (X["insider_net_buy"] <= -0.0003).astype(float)
        X["s8_score"] = a_pos - c_neg

        # ── s9 人均持股市值 ────────────────────────────────
        hn = A(pit(hold, t)["holder_num"])
        t10c = t10[t10["ann_date"] <= t]
        if len(t10c):
            g2 = t10c.sort_values(["ts_code", "end_date"]).groupby("ts_code").tail(10)
            topsum = g2.groupby("ts_code")["hold_amount"].sum().reindex(XL).fillna(0)
        else:
            topsum = pd.Series(0.0, index=XL)
        pcv = (((ts - topsum) * cl) / hn).where(hn > 0)
        X["per_capita_value"] = pcv
        X["s9_score"] = np.where(pcv >= 300000, 1.0, np.where(pcv < 100000, -1.0, 0.0))

        # ── s10 估值水平 (同行业内 PR/PS 分位) ──────────────
        val = np.where(stable == 1, X["pr"], (1.0 / A(pan["ps_ttm"].loc[t])).where(ps > 0))
        vs = pd.Series(val, index=XL)
        indp = vs.groupby(ind3.values).rank(pct=True)
        X["val_ind_pct_low"] = -indp
        X["s10_score"] = np.where(indp <= 0.30, 1.0, np.where(indp >= 0.70, -1.0, 0.0))

        # ── 总分 (s4/s6 N/A, 不计) ─────────────────────────
        sc = ["s1_score", "s2_score", "s3_score", "s5_score", "s7_score", "s8_score", "s9_score", "s10_score"]
        X["kd_score"] = X[sc].fillna(0).sum(axis=1)
        X["n_neg"] = (X[sc] == -1).sum(axis=1)
        X["n_pos"] = (X[sc] == 1).sum(axis=1)

        # ── 价格门控 v2 (正确口径) ─────────────────────────
        gate_pr = (X["pr"] <= PR_GATE).where(stable == 1)
        gate_ps = (A(pan["ps_pct_own"].loc[t]) <= PS_GATE).where(stable != 1)
        X["gate_pass"] = gate_pr.where(stable == 1, gate_ps).fillna(False)

        blk = X.copy()
        blk["_fwd"] = fwd.loc[t].reindex(XL)
        blk["_ind"] = XL.to_series().map(sw_l1).values
        rows.append(blk)
        if (k + 1) % 6 == 0 or k == len(reb) - 1:
            log(f"    {k+1}/{len(reb)} {t} ({time.time()-t0:.0f}s)")

    log("[5/7] 截面 IC ...")
    def ic_series(fc, mode, min_n):
        ics, ns = [], []
        for blk in rows:
            s, r = blk[fc], blk["_fwd"]
            if mode == "ind":
                s = s - s.groupby(blk["_ind"]).transform("mean")
            m = s.notna() & r.notna()
            if mode == "gatev2":
                m &= blk["gate_pass"].fillna(False)
            if m.sum() < min_n:
                continue
            a, b = s[m].rank(), r[m].rank()
            if a.std() == 0 or b.std() == 0:
                continue
            ics.append(a.corr(b)); ns.append(int(m.sum()))
        return pd.Series(ics).dropna(), ns

    def summarize(ics, ns):
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
        for mode, mn in [("all", 200), ("ind", 200), ("gatev2", 100)]:
            ics, ns = ic_series(fc, mode, mn)
            r = summarize(ics, ns[:len(ics)])
            if r:
                if mode == "all":
                    recs[fc] = r
                else:
                    cond.setdefault(fc, {})[mode] = r

    log(f"    因子 {len(recs)} 个")
    out = pd.DataFrame.from_dict(recs, orient="index")
    out["abs_icir"] = out["icir"].abs()
    out = out.sort_values("abs_icir", ascending=False)
    out["pass"] = (out["ic_mean"].abs() >= IC_MIN) & (out["abs_icir"] >= ICIR_MIN)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    log("\n" + "=" * 100)
    log(f"[表1] 全A 月度 Rank IC/ICIR  (v2 修正口径, fwd{FWD}, {reb[0]}~{reb[-1]})")
    log("=" * 100)
    log(out[["n_months", "avg_n", "ic_mean", "icir", "t_stat", "pos_ic_pct", "pass"]]
        .to_string(float_format=lambda v: f"{v:+.4f}"))

    # 门控后池内
    ctab = []
    for fc in recs:
        dd = cond.get(fc, {})
        ctab.append({"factor": fc, "IC_all": recs[fc]["ic_mean"], "ICIR_all": recs[fc]["icir"],
                     "IC_ind": dd.get("ind", {}).get("ic_mean"), "ICIR_ind": dd.get("ind", {}).get("icir"),
                     "IC_gate": dd.get("gatev2", {}).get("ic_mean"), "ICIR_gate": dd.get("gatev2", {}).get("icir"),
                     "n_gate": dd.get("gatev2", {}).get("avg_n")})
    cdf = pd.DataFrame(ctab).set_index("factor")
    cdf["dIC_gate"] = cdf["IC_gate"] - cdf["IC_all"]
    log("\n" + "=" * 120)
    log("[表2] 全A vs 行业中性 vs 快刀价格门控后池内 (门控= 稳定盈利 PR<=1.5 / 非稳定 PS分位<=70%)")
    log("=" * 120)
    log(cdf.sort_values("IC_all", key=lambda s: s.abs(), ascending=False)
        .to_string(float_format=lambda v: f"{v:+.4f}"))

    log("\n" + "=" * 90)
    log("[表3] PR 口径对照 (v1 旧=PE_TTM 版 vs v2 新=增长市盈率版)")
    log("=" * 90)
    for c in ["pr_inv_v1", "pr_inv_v2", "growth_pe_inv", "est_np_yield_mv", "bp", "ep_ttm", "adj_sp_inv"]:
        if c in recs:
            r = recs[c]
            log(f"    {c:18s} IC {r['ic_mean']:+.4f}  ICIR {r['icir']:+.4f}  t {r['t_stat']:+.2f}  n {r['avg_n']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "window": [reb[0], reb[-1]], "n_months": len(reb), "fwd_days": FWD,
        "universe": "全A (daily_basic), 后复权",
        "pr_definition": "PR = (总市值-净现金)/(去年净利*(1+增速)) / ROE(%), 净现金=货币资金+交易性金融资产-有息负债",
        "gate": {"stable": "近3年报净利全>0 -> PR<=1.5", "unstable": "PS自身历史分位<=70%"},
        "thresholds": {"ic": IC_MIN, "icir": ICIR_MIN},
        "factors": recs, "conditional": cond,
        "conditional_table": cdf.reset_index().to_dict("records"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[7/7] 落盘 {OUT}")


if __name__ == "__main__":
    t = time.time()
    main()
    log(f"总耗时 {time.time()-t:.0f}s")
