# -*- coding: utf-8 -*-
"""申万 SW2021 一级行业分类独立第二源采集 (P-20261016-second-source, 第 2 级)。

产出 (data/fuxi_v1/second_source/shenwan/{YYYYMMDD}/):
  - sw_l1_industries.json   31 个一级行业 (swindexcode / swindexname / 行情快照)
  - sw_industry_members.csv 全量成分股映射 (ts_code, sw_l1_code, sw_l1_name,
                            sw_weight, effective_date)  -> 可直接作为
                            full_market_input 的 latest_industry_path 输入
  - sw_members.json         结构化成员 + provenance (rows_hash / response_sha256)
  - MANIFEST.json           快照 manifest

口径声明:
  - 申万成分股接口只覆盖沪深 A 股 (代码前缀 0/3/6), 不含北交所与港美股;
  - 每只股票归属唯一一级行业 (SW2021 一级行业互斥);
  - newweight 为申万指数权重 (%), beginningdate 为该行业归属生效日;
  - status = observed_independent_vendor_response: 独立官方源原始观测,
    不构成投资数据保证 (非 B04.1 的独立审核结论本身, 而是其素材)。
"""
import sys
import time

import requests

from collect_second_source_common import (
    SW_API_L1, SW_API_COMP, UA, digest, now_utc, today_tag,
    snapshot_dir, save_json, load_json, append_index,
    ts_code_from_6digits, sha256_text,
)

HDR = {"Referer": "https://www.swsresearch.com/", "User-Agent": UA}
SOURCE = "shenwan"


def _get(url, params):
    """带重试与限速的申万请求 (官方公开接口, SSL 证书链在本地环境不可验证)。"""
    for i in range(4):
        try:
            r = requests.get(url, params=params, headers=HDR, timeout=25, verify=False)
            if r.status_code == 200:
                return r
            print("  [HTTP %d] %s 重试 %d" % (r.status_code, url, i + 1))
        except Exception as e:  # noqa: BLE001
            print("  [ERR] %s 重试 %d: %s" % (url, i + 1, str(e)[:120]))
        time.sleep(1.5 + 1.5 * i)
    raise RuntimeError("申万请求失败: " + url)


def fetch_l1():
    r = _get(SW_API_L1, {"indextype": "一级行业", "page": 1, "page_size": 100})
    body = r.json()
    assert body.get("code") == "200", "申万一级行业接口异常: " + str(body.get("message"))
    rows = body["data"]["results"]
    assert body["data"]["count"] == len(rows), "申万一级行业分页不完整"
    return body, rows


def fetch_components(swindexcode):
    r = _get(SW_API_COMP, {"swindexcode": swindexcode, "page": 1, "page_size": 10000})
    body = r.json()
    assert body.get("code") == "200", "申万成分股接口异常: " + str(body.get("message"))
    rows = body["data"]["results"]
    assert body["data"]["count"] == len(rows), "申万成分股分页不完整: " + swindexcode
    return body, rows


def run():
    tag = today_tag()
    d = snapshot_dir(SOURCE, tag)

    # 1. 一级行业列表
    l1_body, l1_rows = fetch_l1()
    industries = [{"swindexcode": it["swindexcode"], "swindexname": it["swindexname"]}
                  for it in l1_rows]
    save_json(d / "sw_l1_industries.json", {
        "schema_version": "fuxi.second_source_snapshot.1",
        "source": "shenwan_sw2021", "retrieved_at": now_utc(),
        "source_url": SW_API_L1, "industry_count": len(industries),
        "industries": industries})

    # 2. 全量成分股
    members, raw_responses = [], []
    for it in l1_rows:
        comp_body, comp_rows = fetch_components(it["swindexcode"])
        raw_responses.append({"swindexcode": it["swindexcode"], "count": len(comp_rows)})
        for r in comp_rows:
            ts = ts_code_from_6digits(r["stockcode"])
            members.append({
                "ts_code": ts,
                "raw_code": str(r["stockcode"]).zfill(6),
                "stock_name": r.get("stockname"),
                "sw_l1_code": it["swindexcode"],
                "sw_l1_name": it["swindexname"],
                "sw_weight": r.get("newweight"),
                "effective_date": r.get("beginningdate"),
            })
        time.sleep(0.2)

    mapped = [m for m in members if m["ts_code"]]
    unmapped = [m for m in members if not m["ts_code"]]

    # 3. 落盘 CSV (对齐 full_market_input latest_industry_path 的 ts_code 契约)
    import csv
    csv_path = d / "sw_industry_members.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts_code", "raw_code", "stock_name", "sw_l1_code",
                    "sw_l1_name", "sw_weight", "effective_date"])
        for m in sorted(mapped, key=lambda x: (x["sw_l1_code"], x["ts_code"])):
            w.writerow([m["ts_code"], m["raw_code"], m["stock_name"],
                        m["sw_l1_code"], m["sw_l1_name"], m["sw_weight"],
                        m["effective_date"]])

    # 4. 结构化成员 + provenance
    payload = {
        "schema_version": "fuxi.second_source_snapshot.1",
        "source": "shenwan_sw2021", "retrieved_at": now_utc(),
        "source_url": SW_API_COMP,
        "status": "observed_independent_vendor_response",
        "industry_count": len(industries),
        "member_count": len(members), "mapped_count": len(mapped),
        "unmapped_count": len(unmapped),
        "coverage": {"note": "仅沪深 A 股 (代码前缀 0/3/6), 不含北交所/港美股"},
        "industries": industries, "members": sorted(mapped, key=lambda x: (x["sw_l1_code"], x["ts_code"])),
        "unmapped": sorted(unmapped, key=lambda x: x["raw_code"]),
    }
    payload["rows_hash"] = digest(payload["members"])
    save_json(d / "sw_members.json", payload)

    # 5. manifest
    import os
    manifest = {
        "schema_version": "fuxi.second_source_snapshot.1",
        "source": "shenwan_sw2021", "tag": tag, "retrieved_at": now_utc(),
        "status": "observed_independent_vendor_response",
        "files": {
            "sw_l1_industries.json": sha256_text(
                (d / "sw_l1_industries.json").read_text(encoding="utf-8")),
            "sw_industry_members.csv": sha256_text(
                csv_path.read_text(encoding="utf-8")),
            "sw_members.json": sha256_text(
                (d / "sw_members.json").read_text(encoding="utf-8")),
        },
        "industry_count": len(industries), "member_count": len(members),
        "mapped_count": len(mapped), "unmapped_count": len(unmapped),
        "raw_responses": raw_responses,
    }
    manifest["manifest_hash"] = digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
    save_json(d / "MANIFEST.json", manifest)

    # 6. 累积索引
    append_index(SOURCE, {
        "tag": tag, "retrieved_at": now_utc(), "manifest_hash": manifest["manifest_hash"],
        "industry_count": len(industries), "member_count": len(members),
        "mapped_count": len(mapped), "unmapped_count": len(unmapped),
        "manifest_path": str(d / "MANIFEST.json"),
    })

    print("[申万第二源] 完成: 行业 %d, 成分股 %d, 映射 %d, 未映射 %d"
          % (len(industries), len(members), len(mapped), len(unmapped)))
    if unmapped:
        print("  未映射样本: %s" % [m["raw_code"] for m in unmapped[:10]])
    return manifest


if __name__ == "__main__":
    t0 = time.time()
    m = run()
    print("用时 %.1fs, manifest_hash=%s" % (time.time() - t0, m["manifest_hash"]))
