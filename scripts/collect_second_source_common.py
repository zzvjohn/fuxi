# -*- coding: utf-8 -*-
"""伏羲独立第二源采集公共层 (P-20261016-second-source, 第 2 级)。

职责:
  1. 统一 provenance / hash / 归档语义, 对齐 fuxi_v1 的 digest 约定 (canonical JSON sha256);
  2. 独立第二源代码 -> tushare ts_code 的映射 (申万按代码前缀, 中证按交易所字段);
  3. 累积索引 (index.json) 的读写, 支撑"每日快照 -> 时间序列"的 B04.1 独立 cross-check。

独立第二源语义 (与主源 tushare 隔离):
  - 申万 (swsresearch.com)  -> SW2021 一级行业分类, 独立于 tushare 行业口径;
  - 中证 (csindex.com.cn)    -> 中证全指(000985)成分股 + 分级靠档自由流通权重,
                               是自由流通股本口径的官方第二源锚点。

本模块只做采集与归档, 不修改 fuxi_v1 内核任何判决。
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 数据根: 独立第二源累积框架根目录 (勿与主源 data/fuxi_v1/supplements 混用)
SECOND_SOURCE_ROOT = Path("e:/quant/data/fuxi_v1/second_source")

# 申万行业分类 API (官方 swsresearch 域名, 需 Referer + UA)
SW_API_L1 = "https://www.swsresearch.com/institute-sw/api/index_publish/current/"
SW_API_COMP = "https://www.swsresearch.com/institute-sw/api/index_publish/details/component_stocks/"

# 中证 OSS 静态文件 (官方 csindex, 稳定可下载, 无登录)
CSI_BASE = ("https://oss-ch.csindex.com.cn/static/html/csindex/"
            "public/uploads/file/autofile")
CSI_ALL_SHARE = "000985"  # 中证全指

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

SCHEMA = "fuxi.second_source_snapshot.1"


def canonical_json(value):
    """伏羲 digest 同款 canonical 序列化: sort_keys + 紧凑分隔 + 无 NaN。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(value):
    """与 fuxi_v1.contracts.digest 语义一致的 canonical JSON sha256。"""
    return sha256_text(canonical_json(value))


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def today_tag():
    return datetime.now().strftime("%Y%m%d")


def ts_code_from_6digits(code6):
    """6 位纯数字代码 -> tushare ts_code (申万无交易所字段, 用前缀推断)。"""
    c = str(code6).strip().zfill(6)
    if not re.fullmatch(r"\d{6}", c):
        return None
    if c[0] in "68":          # 600/601/603/605/688/689 -> 沪 (含科创板)
        return c + ".SH"
    if c[0] in "03":          # 000/001/002/003, 300/301 -> 深 (含创业板)
        return c + ".SZ"
    if c[0] in "48":          # 北交所 (申万 API 当前不含, 兜底保留)
        return c + ".BJ"
    if c.startswith("92"):    # 北交所 92 段 (兜底)
        return c + ".BJ"
    return None


def ts_code_from_exchange(code, exchange):
    """中证交易所字段 -> tushare 后缀 (权威映射, 含北交所)。"""
    c = str(code).strip().zfill(6)
    if not re.fullmatch(r"\d{6}", c):
        return None
    ex = str(exchange)
    if "深圳" in ex or "Shenzhen" in ex:
        return c + ".SZ"
    if "上海" in ex or "Shanghai" in ex:
        return c + ".SH"
    if "北京" in ex or "Beijing" in ex:
        return c + ".BJ"
    return None


def save_json(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(canonical_json(body) + "\n")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def snapshot_dir(source, tag):
    d = SECOND_SOURCE_ROOT / source / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_path(source):
    return SECOND_SOURCE_ROOT / source / "index.json"


def load_index(source):
    p = _index_path(source)
    return load_json(p) if p.exists() else {"schema_version": SCHEMA, "source": source, "snapshots": []}


def append_index(source, entry):
    """累积索引: 追加/覆盖当日快照条目, 保留历史, 形成时间序列。"""
    idx = load_index(source)
    snaps = [s for s in idx["snapshots"] if s.get("tag") != entry["tag"]]
    snaps.append(entry)
    snaps.sort(key=lambda s: s["tag"])
    idx["snapshots"] = snaps
    idx["latest_tag"] = entry["tag"]
    save_json(_index_path(source), idx)
    return idx
