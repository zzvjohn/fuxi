# -*- coding: utf-8 -*-
"""A4 指数刹车 A/B 对比: GATE_INDEX=True vs False 的 JQ 持仓明细归因"""
import json
import pandas as pd
import numpy as np

def load_positions(path):
    for enc in ("utf-8-sig", "gbk"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    print(f"[{path}] encoding={enc}, rows={len(df)}, cols={list(df.columns)}")
    return df

def nav_series(df):
    """持仓明细 -> 每日总资产 -> 日收益/净值序列。
    聚宽导出表头16列但数据17字段: pandas 把日期吃成 index,
    '盈亏占比'列实际存的是总资产(float), '仓位占比'是百分比字符串。"""
    daily = df.groupby(level=0)["盈亏占比"].max().sort_index()
    daily = daily.replace(0, np.nan).ffill()  # Cash 行总资产=0, 空仓日用前值
    daily = pd.to_numeric(daily)
    daily.name = "total_asset"
    nav = daily / daily.iloc[0]
    return daily, nav

gate = json.load(open("e:/quant/data/daily_freq/a4_gate_dates.json", encoding="utf-8"))
brake_dates = [d[:4] + "-" + d[4:6] + "-" + d[6:] for d in gate["index_brake_1000"]]

d_true, n_true = nav_series(load_positions("E:/log/position_true.csv"))
d_false, n_false = nav_series(load_positions("E:/log/position_false.csv"))

# 对齐到并集日历
cal = d_true.index.union(d_false.index).sort_values()
r_true = n_true.reindex(cal).ffill().pct_change()
r_false = n_false.reindex(cal).ffill().pct_change()
diff_daily = r_true - r_false

print("\n=== 整体对比 ===")
print(f"True : 累计 {n_true.iloc[-1]/n_true.iloc[0]-1:+.2%}  (首日 {d_true.index[0]} -> 末日 {d_true.index[-1]})")
print(f"False: 累计 {n_false.iloc[-1]/n_false.iloc[0]-1:+.2%}  (首日 {d_false.index[0]} -> 末日 {d_false.index[-1]})")
print(f"差异: {n_true.iloc[-1]/n_false.iloc[-1]-1:+.2%} (净值比)")

# 分年对比
print("\n=== 分年收益对比 ===")
years = sorted(set(d[:4] for d in cal))
for y in years:
    rt = r_true[cal.str.startswith(y)]
    rf = r_false[cal.str.startswith(y)]
    cum_t = (1 + rt.fillna(0)).prod() - 1
    cum_f = (1 + rf.fillna(0)).prod() - 1
    print(f"{y}: True {cum_t:+7.2%} | False {cum_f:+7.2%} | Δ {cum_t-cum_f:+7.2%}")

# 刹车触发窗口对比: 触发日前后各5个交易日(实际空仓一周)
print("\n=== 刹车窗口对比 (中证1000 15日-12% 触发日, 前后5交易日) ===")
for bd in brake_dates:
    if bd not in cal:
        continue
    idx = cal.get_loc(bd)
    lo, hi = max(0, idx - 5), min(len(cal), idx + 6)
    win = cal[lo:hi]
    ct = (1 + r_true[win].fillna(0)).prod() - 1
    cf = (1 + r_false[win].fillna(0)).prod() - 1
    print(f"{bd}: True {ct:+6.2%} | False {cf:+6.2%} | Δ {ct-cf:+6.2%}")

# 全部刹车窗口合计
all_win = pd.Series(False, index=cal)
for bd in brake_dates:
    if bd in cal:
        idx = cal.get_loc(bd)
        all_win.iloc[max(0, idx - 5): min(len(cal), idx + 6)] = True
ct_win = (1 + r_true[all_win].fillna(0)).prod() - 1
cf_win = (1 + r_false[all_win].fillna(0)).prod() - 1
ct_out = (1 + r_true[~all_win].fillna(0)).prod() - 1
cf_out = (1 + r_false[~all_win].fillna(0)).prod() - 1
print(f"\n刹车窗口合计: True {ct_win:+.2%} | False {cf_win:+.2%} | Δ {ct_win-cf_win:+.2%}")
print(f"非窗口合计  : True {ct_out:+.2%} | False {cf_out:+.2%} | Δ {ct_out-cf_out:+.2%}")

# 触发日密度: 33个触发日落在几个事件
ev = []
prev = None
for bd in brake_dates:
    if prev is None or (pd.to_datetime(bd) - pd.to_datetime(prev)).days > 10:
        ev.append([bd, bd])
    else:
        ev[-1][1] = bd
    prev = bd
print(f"\n刹车事件(合并相邻触发日): {len(ev)} 波")
for s, e in ev:
    print(f"  {s} ~ {e}")
