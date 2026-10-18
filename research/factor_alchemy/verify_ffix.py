# -*- coding: utf-8 -*-
"""验证 Phase 2 ffill 修复逻辑: 对现有(含 bug)缓存中稀疏因子应用 ffill,
确认周覆盖从 ~21 周恢复至 ~274 周。不重算因子, 仅验证修复方向正确。"""
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / 'output'
with open(OUT / 'factor_cache_winsorized.pkl', 'rb') as f:
    fd = pickle.load(f)
print(f"loaded {len(fd)} factors from cache")

sparse = []
for name, df in fd.items():
    if not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        continue
    frac = df.isna().mean().mean()
    nw = df.shape[0]
    if (nw <= 30) or (frac > 0.5):
        ff = df.ffill()
        valid = int((ff.notna().sum(axis=1) >= 30).sum())
        sparse.append((name, nw, round(float(frac), 3), valid))

sparse.sort(key=lambda x: -x[2])
print(f"\nsparse factors detected: {len(sparse)}")
print(f"{'因子':<30s} {'weeks':>6s} {'frac_nan':>9s} {'ffill_valid(>=30stk)':>20s}")
for name, nw, frac, valid in sparse:
    print(f"{name:<30s} {nw:>6d} {frac:>9.3f} {valid:>20d}")

dense = [n for n, d in fd.items() if isinstance(d, pd.DataFrame) and d.shape[0] > 30
         and d.isna().mean().mean() <= 0.5]
print(f"\ndense (non-sparse) factors: {len(dense)}")
total_valid = sum(1 for _, _, _, v in sparse)
print(f"SUM ffill_valid across sparse: {total_valid} (期望≈ {len(sparse)}×274)")
