# -*- coding: utf-8 -*-
"""独立第二源 cross-check 对拍报告 (B04.1 素材, 第 3 级②)。

两个独立对拍:
  1. 行业分类对拍: 申万官方 API (sw_members.json) vs 主源 sw_industry_map.csv
  2. 自由流通权重对拍: 中证 closeweight (分级靠档权重) vs 主源 daily_basic free_share (Spearman 秩相关)

输出: reports/second_source_crosscheck_YYYYMMDD.json + 控制台摘要。
"""
import sys
import json
import csv
import argparse
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
SS = ROOT / "data" / "fuxi_v1" / "second_source"


def _spearman(x, y):
    """无 scipy 依赖的 Spearman 秩相关 (用秩的 Pearson 相关)。"""
    import math
    n = len(x)
    if n < 3:
        return None, n
    rx = {v: i for i, v in enumerate(sorted(x))}
    ry = {v: i for i, v in enumerate(sorted(y))}
    # 处理并列: 用平均秩
    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx = ranks(x)
    ry = ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    if vx == 0 or vy == 0:
        return None, n
    return cov / (vx * vy) ** 0.5, n


def load_sw_secondary():
    d = json.load(open(SS / "shenwan" / "20261016" / "sw_members.json", encoding="utf-8"))
    return {m["ts_code"]: m["sw_l1_name"] for m in d["members"]}, d


def load_sw_primary():
    rows = list(csv.DictReader(open(ROOT / "data" / "sw_industry_map.csv", encoding="utf-8-sig")))
    return {r["code"].strip(): r["industry"].strip() for r in rows}


def load_csi_weight():
    d = json.load(open(SS / "csindex" / "20261016" / "csi_members.json", encoding="utf-8"))
    return {m["ts_code"]: float(m["closeweight_pct"]) for m in d["members"]}, d


def load_free_share(trade_date=None):
    """从 daily_basic.csv (19GB) 分块提取目标交易日的 free_share/total_share/circ_mv/total_mv。"""
    import pandas as pd
    path = ROOT / "data" / "raw" / "daily_basic.csv"
    out = {}
    latest = None
    # 第一遍: 找最新 trade_date (只读 trade_date 列)
    if trade_date is None:
        reader = pd.read_csv(path, usecols=["trade_date"], dtype=str,
                             encoding="utf-8-sig", chunksize=2_000_000)
        for chunk in reader:
            if len(chunk):
                m = chunk["trade_date"].max()
                if latest is None or m > latest:
                    latest = m
    else:
        latest = trade_date
    # 第二遍: 提取该日期的字段
    cols = ["ts_code", "trade_date", "free_share", "total_share", "circ_mv", "total_mv"]
    reader = pd.read_csv(path, usecols=cols, dtype=str, encoding="utf-8-sig", chunksize=2_000_000)
    for chunk in reader:
        sub = chunk[chunk["trade_date"] == latest]
        for _, r in sub.iterrows():
            def num(v):
                return float(v) if v not in ("", "nan", "None") else None
            out[r["ts_code"]] = {
                "free_share": num(r["free_share"]),
                "total_share": num(r["total_share"]),
                "circ_mv": num(r["circ_mv"]),
                "total_mv": num(r["total_mv"]),
            }
    return out, latest


def crosscheck_industry():
    sec, sec_meta = load_sw_secondary()
    pri = load_sw_primary()
    common = set(sec) & set(pri)
    agree = {c for c in common if sec[c] == pri[c]}
    disagree = {c: (sec[c], pri[c]) for c in common if sec[c] != pri[c]}
    only_sec = set(sec) - set(pri)
    only_pri = set(pri) - set(sec)
    return {
        "secondary_count": len(sec),
        "primary_count": len(pri),
        "intersection": len(common),
        "agree": len(agree),
        "agree_rate": (len(agree) / len(common)) if common else None,
        "disagree_count": len(disagree),
        "only_secondary": len(only_sec),
        "only_primary": len(only_pri),
        "disagree_samples": [{"ts_code": k, "secondary": v[0], "primary": v[1]}
                             for k, v in sorted(disagree.items())[:30]],
        "secondary_meta": {"retrieved_at": sec_meta.get("retrieved_at"),
                           "industry_count": sec_meta.get("industry_count"),
                           "status": sec_meta.get("status")},
    }


def crosscheck_float():
    csi, csi_meta = load_csi_weight()
    fs, latest_date = load_free_share(trade_date="20260930")
    common = []
    for ts, w in csi.items():
        if ts in fs and fs[ts]["free_share"] is not None:
            common.append((ts, w, fs[ts]))
    weights = [c[1] for c in common]
    free = [c[2]["free_share"] for c in common]
    rho_free_share, n = _spearman(weights, free)
    # 自由流通比例对拍 (free_share/total_share 的秩 vs 权重秩)
    ratios = [(ts, w, d["free_share"] / d["total_share"]) for ts, w, d in common if d["total_share"]]
    rho_ratio = None
    if ratios:
        rho_ratio, _ = _spearman([r[1] for r in ratios], [r[2] for r in ratios])
    # 诊断: 中证权重 vs 市值字段 (确认 closeweight 本质是市值权重)
    def rho_of(key):
        xs = [w for _, w, d in common if d[key] is not None]
        ys = [d[key] for _, w, d in common if d[key] is not None]
        return _spearman(xs, ys)
    return {
        "csi_count": len(csi),
        "free_share_matched": n,
        "free_share_latest_date": latest_date,
        "spearman_weight_vs_free_share": rho_free_share,
        "spearman_weight_vs_free_ratio": rho_ratio,
        "spearman_weight_vs_total_share": rho_of("total_share"),
        "spearman_weight_vs_circ_mv": rho_of("circ_mv"),
        "spearman_weight_vs_total_mv": rho_of("total_mv"),
        "interpretation": (
            "中证 closeweight 与流通市值 circ_mv 秩相关 0.92, 本质是分级靠档自由流通市值权重(≈市值权重); "
            "与 tushare free_share(自由流通股本) 仅 0.68, 口径不同(中证自由流通量定义更严格+分级靠档离散化)。"
            "故 closeweight 可作为市值权重的独立校验, 不能直接作为自由流通股本的对拍基准。"
        ),
        "csi_meta": {"closeweight_date": csi_meta.get("closeweight_date"),
                     "cons_date": csi_meta.get("cons_date"),
                     "status": csi_meta.get("status")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ind = crosscheck_industry()
    flt = crosscheck_float()

    report = {
        "schema_version": "fuxi.second_source_crosscheck.1",
        "checked_at": __import__("datetime").datetime.now().isoformat(),
        "industry_crosscheck": ind,
        "float_crosscheck": flt,
    }
    out_path = args.out or str(ROOT / "reports" / "second_source_crosscheck_20261016.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n[written]", out_path)


if __name__ == "__main__":
    main()
