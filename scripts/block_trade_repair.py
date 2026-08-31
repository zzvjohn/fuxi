# -*- coding: utf-8 -*-
"""
P-20260817-005 落地: block_trade_full.csv 日期列修复
损坏形态: 旧 block_trade.csv 拼接段(前1000行)的 trade_date 被当作纳秒时间戳解析,
   显示为 '1970-01-01 00:00:00.020250530', 原始 YYYYMMDD 藏在小数部分 (0 + YYYYMMDD)。
修复: 提取小数部分去前导0 -> YYYYMMDD; 与正常段拼接后按 (ts_code,trade_date,price,vol) 去重。
输出: data/raw/block_trade_full.csv (覆盖写, 修复前备份 block_trade_full.csv.bak)
"""
import os
import shutil

import pandas as pd

SRC = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "block_trade_full.csv")


def repair():
    path = os.path.abspath(SRC)
    if not os.path.exists(path):
        print(f"[repair] 未找到 {path}")
        return

    shutil.copy2(path, path + ".bak")
    print(f"[repair] 已备份 -> {path}.bak")

    df = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str})
    d = df["trade_date"].astype(str)
    bad = d.str.startswith("1970")

    if bad.sum() == 0:
        print("[repair] 无损坏行, 无需修复")
        return

    # 小数部分: '020250530' -> '20250530'
    fracs = d[bad].str.split(".").str[1].str.lstrip("0")
    repaired = pd.to_datetime(fracs, format="%Y%m%d", errors="coerce")
    n_fail = int(repaired.isna().sum())
    if n_fail:
        print(f"[repair] ⚠️ {n_fail} 行无法解析, 保留原值")
        df.loc[bad & repaired.isna(), "trade_date"] = d[bad & repaired.isna()]
    df.loc[bad & repaired.notna(), "trade_date"] = (
        repaired[repaired.notna()].dt.strftime("%Y%m%d")
    )

    before = len(df)
    df = df.drop_duplicates(subset=["ts_code", "trade_date", "price", "vol"])
    print(f"[repair] 修复 {int(bad.sum())} 行损坏日期 | 去重 -{before - len(df)} 行 | 剩余 {len(df)} 行")

    # 验证: 日期应为合法交易日格式且范围合理
    ok = df["trade_date"].str.match(r"^\d{8}$")
    df = df[ok]
    dts = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    print(f"[repair] 修复后日期范围: {dts.min().date()} ~ {dts.max().date()}")
    print(f"[repair] 修复后唯一日期数: {dts.nunique()}")

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[repair] 已写回 {path}")


if __name__ == "__main__":
    repair()
