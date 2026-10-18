# -*- coding: utf-8 -*-
"""中证指数独立第二源采集 (P-20261016-second-source, 第 2 级)。

产出 (data/fuxi_v1/second_source/csindex/{YYYYMMDD}/):
  - csi_000985_cons.csv         中证全指成分股目录 (含交易所, 含北交所)
  - csi_000985_closeweight.csv  样本权重 (分级靠档自由流通权重 %)
  - csi_members.json            结构化成员 (ts_code 映射) + provenance
  - MANIFEST.json               快照 manifest

口径声明:
  - 中证全指(000985)是"自由流通市值加权"指数, closeweight 的权重(%) 与
    个股分级靠档自由流通市值成正比, 是自由流通股本口径的官方第二源锚点;
  - 指数总自由流通市值可由"权重% x 总市值"反推, 或从指数详情页快照补充;
  - 成分股 cons 与权重 closeweight 的发布日期不同步 (官网正常现象),
    两者均以各自文件内"日期"字段为准并如实记录;
  - status = observed_independent_vendor_response (同申万, 独立素材非保证)。
"""
import time

import pandas as pd
import requests

from collect_second_source_common import (
    CSI_BASE, CSI_ALL_SHARE, UA, digest, now_utc, today_tag,
    snapshot_dir, save_json, append_index, ts_code_from_exchange,
    sha256_text,
)

HDR = {"Referer": "https://www.csindex.com.cn/", "User-Agent": UA}
SOURCE = "csindex"


def _download(url, out_path):
    """带重试的中证 OSS 静态文件下载。"""
    for i in range(4):
        try:
            r = requests.get(url, headers=HDR, timeout=30, verify=False)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(out_path, "wb") as fh:
                    fh.write(r.content)
                return out_path
            print("  [HTTP %d/%dB] %s 重试 %d" % (r.status_code, len(r.content), url, i + 1))
        except Exception as e:  # noqa: BLE001
            print("  [ERR] %s 重试 %d: %s" % (url, i + 1, str(e)[:120]))
        time.sleep(1.5 + 1.5 * i)
    raise RuntimeError("中证文件下载失败: " + url)


def _read_xls(path):
    return pd.read_excel(path, engine="xlrd")


def run():
    tag = today_tag()
    d = snapshot_dir(SOURCE, tag)

    # 1. 下载成分股 + 权重
    cons_url = "%s/cons/%scons.xls" % (CSI_BASE, CSI_ALL_SHARE)
    wgt_url = "%s/closeweight/%scloseweight.xls" % (CSI_BASE, CSI_ALL_SHARE)
    cons_raw = _download(cons_url, d / "csi_000985_cons.xls")
    wgt_raw = _download(wgt_url, d / "csi_000985_closeweight.xls")

    cons = _read_xls(cons_raw)
    wgt = _read_xls(wgt_raw)

    # 2. 归一化列名 (中证 XLS 表头含中英文与括号)
    cons_cols = {
        "成份券代码Constituent Code": "code",
        "成份券名称Constituent Name": "name",
        "交易所Exchange": "exchange",
    }
    wgt_cols = {
        "成份券代码Constituent Code": "code",
        "成份券名称Constituent Name": "name",
        "交易所Exchange": "exchange",
        "权重(%)weight": "weight",
    }
    cons = cons.rename(columns=cons_cols)[list(cons_cols.values())]
    wgt = wgt.rename(columns=wgt_cols)[list(wgt_cols.values())]

    cons_date = str(_first_date(cons_raw))
    wgt_date = str(_first_date(wgt_raw))

    # 3. 代码/交易所 -> ts_code
    members, unmapped = [], []
    for _, r in cons.iterrows():
        ts = ts_code_from_exchange(r["code"], r["exchange"])
        rec = {"ts_code": ts, "raw_code": str(r["code"]).zfill(6),
               "name": r["name"], "exchange": r["exchange"]}
        (members if ts else unmapped).append(rec)

    # 权重表 (仅对能映射的 A 股)
    weights = {}
    for _, r in wgt.iterrows():
        ts = ts_code_from_exchange(r["code"], r["exchange"])
        if ts:
            try:
                weights[ts] = float(r["weight"])
            except (TypeError, ValueError):
                weights[ts] = None

    for m in members:
        m["closeweight_pct"] = weights.get(m["ts_code"])
    for m in unmapped:
        m["closeweight_pct"] = None

    # 4. 落盘 CSV (ts_code 契约, 可直接 cross-check)
    members_df = pd.DataFrame(sorted(members, key=lambda x: x["ts_code"]))
    members_df.to_csv(d / "csi_000985_members.csv", index=False)
    cons.to_csv(d / "csi_000985_cons.csv", index=False)
    wgt.to_csv(d / "csi_000985_closeweight.csv", index=False)

    # 5. 结构化成员 + provenance
    payload = {
        "schema_version": "fuxi.second_source_snapshot.1",
        "source": "csindex_csi_all_share", "retrieved_at": now_utc(),
        "source_urls": {"cons": cons_url, "closeweight": wgt_url},
        "status": "observed_independent_vendor_response",
        "index_code": CSI_ALL_SHARE, "index_name": "中证全指",
        "cons_date": cons_date, "closeweight_date": wgt_date,
        "member_count": len(members), "unmapped_count": len(unmapped),
        "weighted_count": sum(1 for m in members if m["closeweight_pct"] is not None),
        "coverage": {"note": "含沪深北交易所 A 股; 权重为分级靠档自由流通权重%"},
        "members": sorted(members, key=lambda x: x["ts_code"]),
        "unmapped": sorted(unmapped, key=lambda x: x["raw_code"]),
    }
    payload["rows_hash"] = digest(payload["members"])
    save_json(d / "csi_members.json", payload)

    # 6. manifest
    manifest = {
        "schema_version": "fuxi.second_source_snapshot.1",
        "source": "csindex_csi_all_share", "tag": tag, "retrieved_at": now_utc(),
        "status": "observed_independent_vendor_response",
        "index_code": CSI_ALL_SHARE, "cons_date": cons_date, "closeweight_date": wgt_date,
        "files": {
            "csi_000985_cons.xls": sha256_text(open(cons_raw, "rb").read().hex()),
            "csi_000985_closeweight.xls": sha256_text(open(wgt_raw, "rb").read().hex()),
            "csi_members.json": sha256_text((d / "csi_members.json").read_text(encoding="utf-8")),
        },
        "member_count": len(members), "unmapped_count": len(unmapped),
        "weighted_count": sum(1 for m in members if m["closeweight_pct"] is not None),
    }
    manifest["manifest_hash"] = digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
    save_json(d / "MANIFEST.json", manifest)

    # 7. 累积索引
    append_index(SOURCE, {
        "tag": tag, "retrieved_at": now_utc(), "manifest_hash": manifest["manifest_hash"],
        "cons_date": cons_date, "closeweight_date": wgt_date,
        "member_count": len(members), "unmapped_count": len(unmapped),
        "weighted_count": sum(1 for m in members if m["closeweight_pct"] is not None),
        "manifest_path": str(d / "MANIFEST.json"),
    })

    print("[中证第二源] 完成: 成分股 %d, 权重覆盖 %d, 未映射 %d (cons日期=%s, 权重日期=%s)"
          % (len(members), sum(1 for m in members if m["closeweight_pct"] is not None),
             len(unmapped), cons_date, wgt_date))
    return manifest


def _first_date(path):
    """读取 XLS 首个'日期'列的首个非空值 (格式 YYYYMMDD 或 date)。"""
    raw = _read_xls(path)
    date_col = next((c for c in raw.columns if "日期" in str(c) or "Date" in str(c)), None)
    if date_col is None:
        return None
    val = raw[date_col].dropna().iloc[0]
    return str(val).replace("-", "")[:8]


if __name__ == "__main__":
    t0 = time.time()
    m = run()
    print("用时 %.1fs, manifest_hash=%s" % (time.time() - t0, m["manifest_hash"]))
