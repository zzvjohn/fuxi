# -*- coding: utf-8 -*-
"""季度刷新选股池 (2026-09-10): 复用 v1 全流程 (排雷R1/R2/R3 -> 价格门控 -> 十项打分 -> 裁决)
在 中报(8-31)/三季报(10-31) 披露截止日重选池, 与年报时点(4-30)共同构成"季度更新"的选股池序列。
无前视约束:
  - 全部数据 ann_date <= run_date(披露截止日); run_date 取披露截止日当天(报告最迟当日披露)
  - R1 更名史截断 start_date<=run_date (杜绝未来更名信息)
  - 洗澡重述信号仅年报时点计算 (中报/三季报时点 年报vsQ3营收跨年比较无意义, 强制关闭防误判)
期内冻结项(年报口径): s1/s2趋势/s5分红/s6行业/s7龙头截面/R2中位数 —— 财报数据本身未变
期内刷新项: 估值与价格门控(PR/PS分位/周期PE/金融PB), s8增减持回购(12月窗口滑动), s9股东数(中报更新),
           s3/s4资产负债(缓存含季度报表, 中报时点用H1表), R3减持窗口, s10估值分位
"""
import sys, os, shutil
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
import kuaidao_full_backtest_v1 as V

OUT = V.OUT

QPERIODS = [
    dict(tag='P2024H1', report_end='20240630', run_date='20240831'),
    dict(tag='P2024Q3', report_end='20240930', run_date='20241031'),
    dict(tag='P2025H1', report_end='20250630', run_date='20250831'),
    dict(tag='P2025Q3', report_end='20250930', run_date='20251031'),
    dict(tag='P2026H1', report_end='20260630', run_date='20260831'),
]
# 现金流缓存: 季度刷新只用年报行(end 1231), 直接复用对应年报期缓存
CF_SRC = {'P2024H1': 'P2023', 'P2024Q3': 'P2023', 'P2025H1': 'P2024', 'P2025Q3': 'P2024', 'P2026H1': 'P2025'}


_orig_snapshot = V.stock_snapshot


def stock_snapshot_patched(code, tab, mt, uni, P, r2_med, fina_cross):
    """v1 快照 + 防前视补丁: restated(营收重述)仅年报时点(report_end 以 1231 结尾)计算"""
    s = _orig_snapshot(code, tab, mt, uni, P, r2_med, fina_cross)
    if s is not None and not str(P['report_end']).endswith('1231'):
        s['restated'] = False
    return s


def main():
    for P in QPERIODS:
        src, dst = f"{OUT}/cashflow_{CF_SRC[P['tag']]}_full.pkl", f"{OUT}/cashflow_{P['tag']}_full.pkl"
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy(src, dst)
            print(f"[cache] cashflow {P['tag']} <- {CF_SRC[P['tag']]}", flush=True)
    V.stock_snapshot = stock_snapshot_patched
    print('[load] universe + tables ...', flush=True)
    uni = V.load_universe()
    tab_all = V.load_tables()
    mt = V.load_market_tables()
    print(f'[load] universe={len(uni)}, tables={len(tab_all)}, db rows={len(mt["db"])}', flush=True)
    pools = {}
    for P in QPERIODS:
        if os.path.exists(f"{OUT}/pool_{P['tag']}.csv"):
            print(f"[skip] {P['tag']} pool already exists", flush=True)
            continue
        if len(mt['nm']):
            mt['nm'] = mt['nm'][mt['nm']['start_date'].astype(str) <= P['run_date']].copy()
        pools[P['tag']] = V.run_period(P, tab_all, mt, uni)
    print('[done] pool sizes:', {k: len(v) for k, v in pools.items()}, flush=True)


if __name__ == '__main__':
    main()
