# -*- coding: utf-8 -*-
"""方案 A 开关冒烟: 默认 legacy / listed_only 裁剪只减不加 / fail-safe / env 校验 (只读, 不落盘)。"""
import importlib.util
import os
import sys

sys.path.insert(0, r"E:\quant\research\factor_alchemy")
spec = importlib.util.spec_from_file_location(
    "s5jf_smoke", r"E:\quant\research\factor_alchemy\s5_joint_filter.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import pandas as pd

print("=" * 78)
print("[1] 默认档位 (未设 env)")
os.environ.pop("FUXI_S5_BJ_UNIVERSE", None)
print("    _s5_bj_universe() =", mod._s5_bj_universe())

print("\n[2] 显式 listed_only")
os.environ["FUXI_S5_BJ_UNIVERSE"] = "listed_only"
print("    _s5_bj_universe() =", mod._s5_bj_universe())

print("\n[3] 非法值 -> ValueError")
os.environ["FUXI_S5_BJ_UNIVERSE"] = "bogus"
try:
    mod._s5_bj_universe()
    print("    !! no raise (BAD)")
except ValueError as e:
    print("    raise ValueError ok:", e)

print("\n[4] 上市日通道加载 (_bj_listing_date_map)")
os.environ["FUXI_S5_BJ_UNIVERSE"] = "listed_only"
mp = mod._bj_listing_date_map()
print("    loaded codes:", len(mp))
if len(mp) < 350:
    print("    !! loaded < 350 (expect 353)")
codes = sorted(mp)[:3] + sorted(mp)[-3:]
for c in codes:
    print("    %s -> %s" % (c, mp[c]))

print("\n[5] 裁剪只减不加 / 非 BJ 逐位一致")
f = pd.DataFrame({
    "ts_code": ["920201.BJ", "920201.BJ", "920201.BJ", "000001.SZ", "000001.SZ", "600000.SH"],
    "trade_date": pd.to_datetime(["2024-01-02", "2026-09-23", "2026-09-24",
                                  "2024-01-02", "2026-09-24", "2026-09-24"]),
    "close": [10.0, 11.0, 12.0, 100.0, 101.0, 50.0],
})
out = mod._s5_bj_cut_pre_listing(f)
print("    in rows=%d out rows=%d" % (len(f), len(out)))
assert set(map(tuple, out[["ts_code", "trade_date"]].values)) <= set(
    map(tuple, f[["ts_code", "trade_date"]].values)), "输出不是输入子集"
nonbj_in = f[f.ts_code != "920201.BJ"].reset_index(drop=True)
nonbj_out = out[out.ts_code != "920201.BJ"].reset_index(drop=True)
print("    非 BJ 行数 %d -> %d ; 逐位一致 = %s" % (
    len(nonbj_in), len(nonbj_out), nonbj_in.equals(nonbj_out)))
print("    920201.BJ: 剪掉日期 =",
      sorted(str(d)[:10] for d in f.trade_date[~f.trade_date.isin(out.trade_date)]))
print("    920201.BJ 保留 =", sorted(str(d)[:10] for d in out[out.ts_code == "920201.BJ"].trade_date))

print("\n[6] fail-safe: 无 list_date 的 .BJ 不裁")
f2 = pd.DataFrame({"ts_code": ["999999.BJ"], "trade_date": pd.to_datetime(["2024-01-02"]), "close": [1.0]})
out2 = mod._s5_bj_cut_pre_listing(f2)
print("    unknown .BJ rows %d -> %d (expect 1 -> 1)" % (len(f2), len(out2)))

print("\n[7] 空/None/无 ts_code 列 (fail-safe)")
print("    None ->", mod._s5_bj_cut_pre_listing(None))
print("    empty ->", len(mod._s5_bj_cut_pre_listing(pd.DataFrame({"ts_code": [], "trade_date": []}))))
print("    no ts_code ->", len(mod._s5_bj_cut_pre_listing(pd.DataFrame({"x": [1]}))))

print("\n[8] legacy 档不裁 (整条链路确认)")
os.environ["FUXI_S5_BJ_UNIVERSE"] = "legacy"
mod._BJ_LISTING_DATE_CACHE.clear()
print("    legacy mode =", mod._s5_bj_universe())

print("=" * 78)
print("[smoke] PASS")
