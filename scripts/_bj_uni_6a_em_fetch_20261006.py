# -*- coding: utf-8 -*-
"""拉取东财北交所证券列表快照（含上市日期 f26）—— §6-A 第三方仲裁通道

产出：reports/fuxi_v1_bj_universe_20261006/listing_date_source_eastmoney.json
只读外部接口，不改任何生产文件。可重复执行（幂等覆盖快照）。
"""
import json
import time
import urllib.request
from datetime import datetime

OUT = r"E:\quant\reports\fuxi_v1_bj_universe_20261006\listing_date_source_eastmoney.json"
URL = ("https://push2.eastmoney.com/api/qt/clist/get?pn={pn}&pz=100&po=1&np=1"
       "&fltt=2&invt=2&fid=f12&fs=m:0+t:81+s:2048&fields=f12,f14,f26")


def fetch(pn, retry=3):
    last = None
    for i in range(retry):
        try:
            req = urllib.request.Request(
                URL.format(pn=pn),
                headers={
                    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                                   "Chrome/125.0 Safari/537.36"),
                    "Referer": "https://quote.eastmoney.com/",
                    "Accept": "*/*",
                })
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5)
    raise RuntimeError(f"fetch pn={pn} failed after {retry}: {last}")


def main():
    rows = []
    total = None
    for pn in range(1, 6):
        d = fetch(pn)
        data = d.get("data") or {}
        total = data.get("total", total)
        diff = data.get("diff") or []
        if not diff:
            break
        rows.extend(diff)
        print(f"[pn={pn}] +{len(diff)} (cum={len(rows)} / total={total})")
        if len(rows) >= (total or 0):
            break
        time.sleep(0.4)

    snap = {
        "schema": "fuxi_bj_listing_dates/eastmoney/v1",
        "fetched_at": datetime.now().isoformat(),
        "source": "push2.eastmoney.com clist (fs=m:0+t:81+s:2048 = 北交所板块)",
        "fields": "f12=code, f14=name, f26=list_date(YYYYMMDD)",
        "total_reported": total,
        "n_rows": len(rows),
        "rows": sorted(rows, key=lambda x: str(x.get("f12"))),
    }
    json.dump(snap, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[done] {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
