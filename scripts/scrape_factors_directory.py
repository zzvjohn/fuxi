# -*- coding: utf-8 -*-
"""factors.directory 全站因子采集器 v2 (2026-09-05)

- 从 sitemap 的 zh 因子页 URL 逐个抓取 HTML
- 用 fd_extractor.extract_factor_data (LD+JSON / RSC flight payload 双路径) 提取
- 输出: data/factors_directory_factors.json (精炼索引) + data/_fd_tmp/raw/*.json (原始)
- 礼貌限速 0.4s, 失败重试 2 次, 支持断点续抓
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根
DATA = os.path.join(BASE, "data")
TMP = os.path.join(DATA, "_fd_tmp")
RAW = os.path.join(TMP, "raw")
os.makedirs(RAW, exist_ok=True)
sys.path.insert(0, os.path.join(BASE, "scripts"))
from fd_extractor import extract_factor_data

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def fetch(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(1.5 * (i + 1))
    return None


def main():
    with open(os.path.join(TMP, "urls_clean.txt"), encoding="utf-8") as f:
        urls = [u.strip() for u in f if "/zh/factors/" in u]
    print("zh 因子页总数: %d" % len(urls), flush=True)

    out = []
    ok = fail = 0
    for i, url in enumerate(urls):
        slug = url.rstrip("/").split("/")[-1]
        cat = url.split("/factors/")[1].split("/")[0]
        raw_path = os.path.join(RAW, slug + ".json")
        fac = None
        if os.path.exists(raw_path):
            try:
                fac = json.load(open(raw_path, encoding="utf-8"))
            except Exception:
                fac = None
        if fac is None:
            html = fetch(url)
            if html is None:
                fail += 1
                print("[FAIL] %s" % url, flush=True)
                continue
            fac = extract_factor_data(html)
            if fac is None or not fac.get("formulas"):
                fail += 1
                print("[NO-FORMULA] %s" % url, flush=True)
                continue
            fac["slug"] = slug
            fac["category"] = cat
            fac["url"] = url
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(fac, f, ensure_ascii=False, indent=2)

        out.append({
            "slug": slug,
            "url": url,
            "category": cat,
            "name_zh": fac.get("name", ""),
            "tags": fac.get("tags", []),
            "formulas": fac.get("formulas", []),
            "description": (fac.get("description") or "")[:3000],
        })
        ok += 1
        if i % 50 == 0 or i == len(urls) - 1:
            print("progress %d/%d ok=%d fail=%d" % (i + 1, len(urls), ok, fail),
                  flush=True)
        time.sleep(0.4)

    with open(os.path.join(DATA, "factors_directory_factors.json"), "w",
              encoding="utf-8") as f:
        json.dump({"source": "factors.directory", "scraped_at":
                   time.strftime("%Y-%m-%d %H:%M:%S"), "n": len(out),
                   "n_fail": fail, "factors": out}, f, ensure_ascii=False,
                  indent=1)
    print("DONE ok=%d fail=%d -> data/factors_directory_factors.json"
          % (ok, fail), flush=True)


if __name__ == "__main__":
    main()
