# -*- coding: utf-8 -*-
"""
XGBoost 非线性因子组合训练
============================
用 Factor Alchemy 已有的 5 因子 CSV + Forward Returns
训练 XGBoost → 导出 3^5 = 243 行查表 → 聚宽可用
"""
import pandas as pd
import numpy as np
import pickle, os, sys

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
FACTOR_DIR = OUTPUT_DIR

print("=" * 60)
print("[1] 加载 5 因子 & 价格数据")
print("=" * 60)

factors = {}
fnames = ['max_ret_1m', 'vol_3m', 'cfp', 'sp', 'davol_20']
for fn in fnames:
    fpath = os.path.join(FACTOR_DIR, 'factor_%s.csv' % fn)
    df = pd.read_csv(fpath, index_col=0)  # trade_date as index
    factors[fn] = df
    print("  %s: %d weeks, %d stocks" % (fn, df.shape[0], df.shape[1]))
    print("    date range: %s ~ %s" % (df.index[0], df.index[-1]))

# ---- 统一日期对齐 ----
common_dates = factors['max_ret_1m'].index
for fn in fnames:
    common_dates = common_dates.intersection(factors[fn].index)
print(f"\n  共同日期: {len(common_dates)} 周")

# ---- 取 1 周的 forward return ----
# 用 max_ret_1m 的第一行对齐（所有因子日期一致）
all_stocks = set()
for fn in fnames:
    all_stocks.update(factors[fn].columns.tolist())
all_stocks = sorted(all_stocks)
print(f"  总股票数: {len(all_stocks)}")

# 构建长表
print("\n" + "=" * 60)
print("[2] 构建训练数据集 (long format)")
print("=" * 60)

records = []
date_list = sorted(common_dates)
# 使用 t 周因子 → t+1 周 forward return
for i in range(len(date_list) - 1):
    t_date = date_list[i]
    t1_date = date_list[i + 1]
    
    # 取 5 因子的 t 周截面
    row = {}
    for fn in fnames:
        s = factors[fn].loc[t_date].dropna()
        row[fn] = s
    
    # compute forward return: 需要 close prices
    # 此处暂用 t_date 到 t1_date 的因子值差作为近似 (后续替换为真实收益)
    # TODO: 需要真实 forward return — 先跳过需要价格数据的部分
    # 因为因子 CSV 里只有因子值没有价格

print("\n[WARN] 需要 close 价格数据计算 forward returns")
print("       从因子 CSV 本身无法直接获取 forward return")
print("       需要加载 Tushare 周频价格 或 pipeline 中已有的 forward_returns")

# ---- 方案: 使用 pipeline 中已有的 forward_returns ----
# 检查是否有保存的 forward_returns 文件
fw_paths = [
    os.path.join(OUTPUT_DIR, 'forward_returns.pkl'),
    os.path.join(OUTPUT_DIR, 'forward_returns.csv'),
]
fw_loaded = False
for fwp in fw_paths:
    if os.path.exists(fwp):
        if fwp.endswith('.pkl'):
            fw = pickle.load(open(fwp, 'rb'))
            if isinstance(fw, pd.DataFrame):
                print("  Loaded forward_returns.pkl: %s" % str(fw.shape))
                fw_loaded = True
                break
        else:
            fw = pd.read_csv(fwp, index_col=0)
            print("  Loaded forward_returns.csv: %s" % str(fw.shape))
            fw_loaded = True
            break

if not fw_loaded:
    print("\n  forward_returns not found in output/")
    print("  Attempting to compute from factor_alchemy pipeline...")
    
    # 用 run_fa.py 的部分基础设施计算 forward returns
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        from research.factor_alchemy.pipeline.data_loader import load_data
        print("  Loading data via pipeline...")
        # 这会很慢, 但只需要一周价格
        pass
    except Exception as e:
        print(f"  [ERROR] Cannot load pipeline: {e}")
    
    print("\n  === 改用替代方案: 直接从 Tushare 拉取价格 ===")
    # 简单的替代: 用 factor CSV 的前后两周因子均值变化
    # (这只是一个占位 — 真实训练需要 forward returns)

print("\n[SKIP] XGBoost training requires forward returns which need price data")
print("       Will implement in next iteration")
