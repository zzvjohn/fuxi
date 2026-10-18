# -*- coding: utf-8 -*-
"""AlphaAgent v3 allweather A/B 归因: GATE_INDEX=True vs False 持仓明细"""
import io
import json
import numpy as np
import pandas as pd

COLS17 = ["日期","品种","标的","多空","数量","可用数量","收盘价","市值","盈亏",
          "开仓均价","持仓均价","保证金","当日盈亏","今手数","盈亏占比pct","总资产","仓位占比"]

def load(path):
    df = pd.read_csv(path, encoding="gbk", header=None, names=COLS17, skiprows=1)
    df.index = pd.to_datetime(df["日期"])
    df = df.drop(columns=["日期"])
    df["标的"] = df["标的"].astype(str).str.strip()
    ta = pd.to_numeric(df["总资产"], errors="coerce")
    cash_mv = pd.to_numeric(df["市值"], errors="coerce").where(df["标的"] == "Cash")
    df["_ta"] = ta.fillna(0)
    # Cash 行: 市值位存的是现金总资产
    df.loc[df["标的"] == "Cash", "_ta"] = cash_mv[df["标的"] == "Cash"].fillna(0)
    return df

def daily_series(df):
    ta = df.groupby(level=0)["_ta"].max()
    ta = ta.replace(0, np.nan).ffill()
    n_stock = df.groupby(level=0)["标的"].apply(lambda x: (x != "Cash").sum())
    return ta, n_stock

d_t, ns_t = daily_series(load("E:/log/jq_alpha_agent_v3_allweather_true.csv"))
d_f, ns_f = daily_series(load("E:/log/jq_alpha_agent_v3_allweather_false.csv"))

nav_t = d_t / d_t.iloc[0]
nav_f = d_f / d_f.iloc[0]
print("=== 整体 (复现用户口径) ===")
print(f"True : {d_t.index[0]} ~ {d_t.index[-1]}  累计 {nav_t.iloc[-1]-1:+.2%}  (首 {d_t.iloc[0]:,.0f} -> 末 {d_t.iloc[-1]:,.0f})")
print(f"False: {d_f.index[0]} ~ {d_f.index[-1]}  累计 {nav_f.iloc[-1]-1:+.2%}  (首 {d_f.iloc[0]:,.0f} -> 末 {d_f.iloc[-1]:,.0f})")

# 夏普/回撤重建
def perf(nav):
    r = nav.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    return sharpe, dd
st, ddt = perf(nav_t); sf, ddf = perf(nav_f)
print(f"重建 Sharpe: True {st:.3f} | False {sf:.3f}  (用户: 0.78 / 0.689)")
print(f"重建 MaxDD : True {ddt:+.2%} | False {ddf:+.2%}  (用户: 27.59% / 35.41%)")

# 对齐日历
cal = d_t.index.union(d_f.index).sort_values()
r_t = nav_t.reindex(cal).ffill().pct_change()
r_f = nav_f.reindex(cal).ffill().pct_change()

print("\n=== 分年 ===")
for y in sorted(set(str(d)[:4] for d in cal)):
    m = [str(d).startswith(y) for d in cal]
    ct = (1 + r_t[m].fillna(0)).prod() - 1
    cf = (1 + r_f[m].fillna(0)).prod() - 1
    print(f"{y}: True {ct:+8.2%} | False {cf:+8.2%} | Δ {ct-cf:+8.2%}")

# 空仓段 (true 版) -> 刹车事件
flat = (ns_t.reindex(cal).fillna(0) <= 1)
segs, s = [], None
for d, v in flat.items():
    if v and s is None: s = d
    if not v and s is not None:
        segs.append((s, d)); s = None
if s is not None: segs.append((s, flat.index[-1]))
segs = [(a, b) for a, b in segs if (pd.Timestamp(b) - pd.Timestamp(a)).days >= 2]
print(f"\n=== True 版空仓/半空仓段(<=1只, >=2自然日; =刹车事件+停牌残留): {len(segs)} 波 ===")
for a, b in segs:
    print(f"  {a} ~ {b} ({pd.Timestamp(b)-pd.Timestamp(a)}自然日)")

print("\n=== False 版空仓段(应≈0, 基线无刹车) ===")
flat_f = (ns_f.reindex(cal).fillna(0) <= 1)
s2 = None; segs_f = []
for d, v in flat_f.items():
    if v and s2 is None: s2 = d
    if not v and s2 is not None:
        segs_f.append((s2, d)); s2 = None
if s2 is not None: segs_f.append((s2, flat_f.index[-1]))
segs_f = [(a, b) for a, b in segs_f if (pd.Timestamp(b) - pd.Timestamp(a)).days >= 2]
print(f"  {len(segs_f)} 段: {segs_f[:8]}")

# 刹车窗口分解: 每波事件 ±5 交易日
print("\n=== 刹车窗口分解 (每波事件前后5交易日) ===")
all_win = pd.Series(False, index=cal)
for a, b in segs:
    ia, ib = cal.get_loc(a), cal.get_loc(b)
    all_win.iloc[max(0, ia - 5): min(len(cal), ib + 6)] = True
    ct = (1 + r_t[cal[ia:ib + 1]].fillna(0)).prod() - 1
    cf = (1 + r_f[cal[ia:ib + 1]].fillna(0)).prod() - 1
    print(f"  {a}~{b}: True {ct:+7.2%} | False {cf:+7.2%} | Δ {ct-cf:+7.2%}")
ct_w = (1 + r_t[all_win].fillna(0)).prod() - 1
cf_w = (1 + r_f[all_win].fillna(0)).prod() - 1
ct_o = (1 + r_t[~all_win].fillna(0)).prod() - 1
cf_o = (1 + r_f[~all_win].fillna(0)).prod() - 1
print(f"\n窗口±5日合计: True {ct_w:+.2%} | False {cf_w:+.2%} | Δ {ct_w-cf_w:+.2%}")
print(f"非窗口合计  : True {ct_o:+.2%} | False {cf_o:+.2%} | Δ {ct_o-cf_o:+.2%}")
print(f"非窗口占比(总天数): {all_win.sum()}/{len(cal)} 天在窗口内")

# 月度最差/最佳窗口, 2026 重点
print("\n=== 2026 上半年逐月 (刹车年) ===")
m26 = [str(d)[:7] for d in cal if str(d).startswith("2026")]
for ym in sorted(set(m26)):
    m = [str(d).startswith(ym) for d in cal]
    ct = (1 + r_t[m].fillna(0)).prod() - 1
    cf = (1 + r_f[m].fillna(0)).prod() - 1
    print(f"{ym}: True {ct:+7.2%} | False {cf:+7.2%} | Δ {ct-cf:+7.2%}")
