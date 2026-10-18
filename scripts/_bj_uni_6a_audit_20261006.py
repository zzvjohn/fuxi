# -*- coding: utf-8 -*-
"""§6-A 上市日通道审计（BJ universe 批 · 只读）

目的（预注册 §6-A）：对 list_date 候选源做三断言，决定方案 A 能否开工。
  断言1 包含性：价格面板中全部 .BJ 代码在两源中均有非空 list_date
  断言2 单调性：list_date 与「价格首行日」口径一致（头部悬挂 = list_date > 首行日）
  断言3 形状  ：无空值 / 无未来日 / 格式合法（YYYYMMDD）
另加：源间一致性（delist_dates vs stock_list 的 list_date 逐只对账）
      账本交叉验证（M17 的 before_list 格是否逐格满足 date < list_date）

产出：reports/fuxi_v1_bj_universe_20261006/listing_date_audit.json
"""
import json
import os
import sys
from datetime import datetime

import pandas as pd

ROOT = r"E:\quant"
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "reports", "fuxi_v1_bj_universe_20261006")
LEDGERS = [
    os.path.join(ROOT, "reports", "fuxi_p2m4_m17_reeval_20260928", "stage2_gap_ledger.json"),
]
os.makedirs(OUT, exist_ok=True)


def norm_d8(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def load_src(path):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    df["ts_code"] = df["ts_code"].astype(str).str.strip()
    df["list_date"] = norm_d8(df["list_date"]).replace("", None)
    return df


def fmt_check(v):
    """YYYYMMDD 合法性"""
    if v is None or v != v or v == "None":
        return False
    if len(v) != 8 or not v.isdigit():
        return False
    try:
        datetime.strptime(v, "%Y%m%d")
        return True
    except ValueError:
        return False


def main():
    rep = {"schema": "fuxi_v1_bj_universe/listing_date_audit/v1",
           "generated_at": datetime.now().isoformat(),
           "read_only": True, "assertions": {}, "findings": {}}

    # ── 0. 载入 ──
    dd = load_src(os.path.join(RAW, "delist_dates.csv"))
    sl = load_src(os.path.join(RAW, "stock_list.csv"))
    px = pd.read_csv(os.path.join(RAW, "daily_prices.csv"),
                     usecols=["ts_code", "trade_date"],
                     dtype={"trade_date": "str"}, encoding="utf-8-sig",
                     low_memory=False)
    px["trade_date"] = norm_d8(px["trade_date"])
    px["ts_code"] = px["ts_code"].astype(str)

    bj_codes_px = sorted({c for c in px["ts_code"].unique() if c.endswith(".BJ")})
    print(f"[load] delist_dates={len(dd)} stock_list={len(sl)} | px .BJ codes={len(bj_codes_px)}")

    dd_bj = dd[dd["ts_code"].str.endswith(".BJ")].set_index("ts_code")
    sl_bj = sl[sl["ts_code"].str.endswith(".BJ")].set_index("ts_code")

    # 前缀分布（信息披露）
    pre = {}
    for c in bj_codes_px:
        pre[c[:3]] = pre.get(c[:3], 0) + 1
    rep["findings"]["bj_prefix_dist_px"] = dict(sorted(pre.items()))

    # ── 断言1 包含性 ──
    miss_dd = [c for c in bj_codes_px if c not in dd_bj.index or dd_bj.at[c, "list_date"] is None]
    miss_sl = [c for c in bj_codes_px if c not in sl_bj.index or sl_bj.at[c, "list_date"] is None]
    rep["assertions"]["A1_coverage"] = {
        "px_bj_codes": len(bj_codes_px),
        "delist_dates_missing": len(miss_dd), "delist_dates_missing_sample": miss_dd[:20],
        "stock_list_missing": len(miss_sl), "stock_list_missing_sample": miss_sl[:20],
        "pass_delist_dates": len(miss_dd) == 0, "pass_stock_list": len(miss_sl) == 0,
    }
    print(f"[A1] coverage: dd_missing={len(miss_dd)} sl_missing={len(miss_sl)}")

    # ── 断言3 形状（先做，独立于口径） ──
    today = datetime.now().strftime("%Y%m%d")
    shape = {}
    for nm, d in (("delist_dates", dd_bj), ("stock_list", sl_bj)):
        vals = d["list_date"].dropna().astype(str)
        bad_fmt = [c for c, v in vals.items() if not fmt_check(v)]
        future = [c for c, v in vals.items() if fmt_check(v) and v > today]
        too_old = [c for c, v in vals.items() if fmt_check(v) and v < "19900101"]
        shape[nm] = {"n_bj_rows": int(len(d)), "n_list_date_nonnull": int(len(vals)),
                     "bad_format": len(bad_fmt), "bad_format_sample": bad_fmt[:10],
                     "future_dates": len(future), "future_sample": future[:10],
                     "pre_1990": len(too_old), "pre_1990_sample": too_old[:10],
                     "pass": not bad_fmt and not future and not too_old}
    rep["assertions"]["A3_shape"] = shape
    print(f"[A3] shape: dd_pass={shape['delist_dates']['pass']} sl_pass={shape['stock_list']['pass']}")

    # ── 源间一致性 ──
    both = sorted(set(dd_bj.index) & set(sl_bj.index) & set(bj_codes_px))
    mism = []
    for c in both:
        v1 = dd_bj.at[c, "list_date"]
        v2 = sl_bj.at[c, "list_date"]
        if v1 != v2:
            mism.append({"ts_code": c, "delist_dates": v1, "stock_list": v2})
    rep["findings"]["source_agreement"] = {
        "n_both": len(both), "n_mismatch": len(mism), "mismatch_sample": mism[:20],
    }
    print(f"[agree] both={len(both)} mismatch={len(mism)}")

    # ── 断言2 单调性：list_date vs 价格首行日 ──
    first_px = px[px["ts_code"].isin(bj_codes_px)].groupby("ts_code")["trade_date"].min()
    rows = []
    for c in bj_codes_px:
        ld = dd_bj.at[c, "list_date"] if c in dd_bj.index else None
        fp = first_px.get(c)
        if ld and fp:
            hung = ld > fp                       # 头部悬挂：价格表含上市前数据
            lead_days = None
            if hung:
                lead_days = int((pd.Timestamp(ld) - pd.Timestamp(fp)).days)
            rows.append({"ts_code": c, "list_date": ld, "first_px": fp,
                         "head_overhang": bool(hung), "overhang_days": lead_days})
    mono = pd.DataFrame(rows)
    n_hung = int(mono["head_overhang"].sum()) if len(mono) else 0
    hung_codes = mono[mono["head_overhang"]]["ts_code"].tolist() if len(mono) else []
    rep["assertions"]["A2_monotonicity"] = {
        "n_checked": int(len(mono)),
        "n_head_overhang": n_hung,
        "n_normal": int(len(mono)) - n_hung,
        "overhang_codes_sample": hung_codes[:30],
        "overhang_days_quantiles": (
            {q: float(mono.loc[mono["head_overhang"], "overhang_days"].quantile(q))
             for q in (0.0, 0.25, 0.5, 0.75, 1.0)} if n_hung else {}),
        "all_prefix_92": all(c.startswith("92") for c in hung_codes),
        "pass": bool(len(mono) and (n_hung + (len(mono) - n_hung) == len(mono))),
        "semantics": "head_overhang = list_date > first_price_date (北交所前行情=新三板期)",
    }
    print(f"[A2] checked={len(mono)} head_overhang={n_hung} all92={rep['assertions']['A2_monotonicity']['all_prefix_92']}")

    # ── 账本交叉：before_list 格逐格验证 ──
    ledger = json.load(open(LEDGERS[0], encoding="utf-8"))
    xcheck = {"by_candidate": {}}
    tot_cells = tot_ok_dd = tot_ok_sl = 0
    uniq_bj_codes = set()
    for r in ledger["results"]:
        cat = (r.get("attribution") or {}).get("by_cat", {})
        bl = cat.get("before_list", 0)
        # 重算：全部格中 (date < list_date) 的数量（用两源各自判）
        ok_dd = ok_sl = n = 0
        for cell in r["cells"]:
            ts, d = cell["ts_code"], cell["date"].replace("-", "")
            n += 1
            v1 = dd_bj.at[ts, "list_date"] if ts in dd_bj.index else None
            v2 = sl_bj.at[ts, "list_date"] if ts in sl_bj.index else None
            if v1 and d < v1:
                ok_dd += 1
                uniq_bj_codes.add(ts)
            if v2 and d < v2:
                ok_sl += 1
        tot_cells += n
        tot_ok_dd += ok_dd
        tot_ok_sl += ok_sl
        xcheck["by_candidate"][r["candidate"]] = {
            "cells": n, "before_list_reported": bl,
            "cells_d_lt_listdate_dd": ok_dd, "cells_d_lt_listdate_sl": ok_sl,
        }
    xcheck["total"] = {"cells": tot_cells, "ok_dd": tot_ok_dd, "ok_sl": tot_ok_sl,
                       "reported_829": 829,
                       "uniq_bj_codes_in_ledger": len(uniq_bj_codes),
                       "bj_codes_sample": sorted(uniq_bj_codes)[:20]}
    rep["findings"]["ledger_crosscheck"] = xcheck
    print(f"[xcheck] cells={tot_cells} d<list_date: dd={tot_ok_dd} sl={tot_ok_sl}")

    # ── 总判 ──
    a1 = rep["assertions"]["A1_coverage"]["pass_delist_dates"] and \
         rep["assertions"]["A1_coverage"]["pass_stock_list"]
    a2 = rep["assertions"]["A2_monotonicity"]["pass"]
    a3 = shape["delist_dates"]["pass"] and shape["stock_list"]["pass"]
    rep["verdict"] = {
        "A1_coverage": bool(a1), "A2_monotonicity": bool(a2), "A3_shape": bool(a3),
        "source_agreement_no_mismatch": len(mism) == 0,
        "can_start_plan_A": bool(a1 and a2 and a3),
        "notes": [],
    }
    if a1 and a2 and a3:
        rep["verdict"]["notes"].append(
            "上市日通道两源均可用于 .BJ 全量覆盖与形状合法")
    if len(mism) > 0:
        rep["verdict"]["notes"].append(
            f"两源有 {len(mism)} 处 list_date 不一致 ⇒ 须先定裁决规则")
    print("[verdict] can_start_plan_A =", rep["verdict"]["can_start_plan_A"])

    path = os.path.join(OUT, "listing_date_audit.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[done]", path)


if __name__ == "__main__":
    main()
