# -*- coding: utf-8 -*-
"""批量审计所有 jq_s5_pass_*.py 的 lookback 是否满足公式末行有效性。
2026-08-31: 36fe6513 valid=0 事故复盘 — 旧 infer_lookback(max*2+10) 低估嵌套窗口链需求。"""
import re
import glob
import numpy as np
import pandas as pd

G = {"np": np, "pd": pd, "abs": np.abs, "sqrt": np.sqrt, "log": np.log,
     "log1p": np.log1p, "exp": np.exp, "sign": np.sign,
     "maximum": np.maximum, "minimum": np.minimum,
     "where": np.where, "clip": np.clip}

files = glob.glob("research/factor_alchemy/jq_s5_pass_*.py")
bad, checked = [], 0
for f in files:
    src = open(f, encoding="utf-8").read()
    m_lb = re.search(r"g\.lookback\s*=\s*(\d+)", src)
    m_ex = re.search(r'_EXPR\s*=\s*"(.+)"', src)
    if not m_lb or not m_ex:
        continue
    checked += 1
    lb, expr = int(m_lb.group(1)), m_ex.group(1)
    idx = pd.date_range("2025-01-01", periods=lb, freq="B")
    close = pd.DataFrame(np.random.rand(lb, 5) + 10, index=idx, columns=list("abcde"))
    volume = pd.DataFrame(np.random.rand(lb, 5) * 1e6, index=idx, columns=list("abcde"))
    L = {"close_p": close, "volume_p": volume, "open_p": close, "high_p": close,
         "low_p": close, "amount_p": volume, "close": close, "open": close,
         "high": close, "low": close, "volume": volume, "amount": volume,
         "returns": close.pct_change()}
    try:
        val = eval(expr, G, L)
        if isinstance(val, pd.DataFrame):
            n_ok = int(val.iloc[-1].notna().sum())
            if n_ok == 0:
                bad.append((f.split("\\")[-1], lb, "末行0/5有效"))
    except Exception as e:
        bad.append((f.split("\\")[-1][:55], lb, "EVAL_ERR: " + str(e)[:60]))

print("扫描文件总数:", len(files), "| 含 _EXPR 可验证:", checked)
print("lookback 不足 (末行全 NaN):")
for b in bad:
    print("  ", b)
if not bad:
    print("   (无 — 除已修复的 36fe6513)")
