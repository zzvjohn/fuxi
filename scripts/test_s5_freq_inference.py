# -*- coding: utf-8 -*-
"""
v0.8 自动频次判断 单元测试 + 回归验证。

1. infer_periods_per_year: 日/周/月/双周/字符串/异常/退化 → 正确年化因子
2. compute_metrics 守卫: 月频 12 观测不误杀; 周频 20 观测触发守卫
3. 日频回归: 重跑 small_cap_liquidity_quality, Calmar 与旧口径 (1.26/2.46) 一致
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "research/factor_alchemy")
import numpy as np
import pandas as pd
from s5_joint_filter import S5JointFilter, infer_periods_per_year

PASS, FAIL = 0, 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name} {detail}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {detail}")

print("═══ 1. infer_periods_per_year 频次推断 ═══")
# 日频 (工作日)
idx_d = pd.bdate_range("2025-01-01", "2025-12-31")
check("日频工作日", infer_periods_per_year(idx_d) == 252, f"→ {infer_periods_per_year(idx_d)}")
# 周频 (每周五)
idx_w = pd.date_range("2024-01-05", "2026-06-30", freq="W-FRI")
check("周频", infer_periods_per_year(idx_w) == 52, f"→ {infer_periods_per_year(idx_w)}")
# 月频 (每月末)
idx_m = pd.date_range("2020-01-31", "2026-06-30", freq="ME")
check("月频", infer_periods_per_year(idx_m) == 12, f"→ {infer_periods_per_year(idx_m)}")
# 双周
idx_b = pd.date_range("2024-01-01", "2026-06-30", freq="2W-MON")
check("双周频", infer_periods_per_year(idx_b) == 26, f"→ {infer_periods_per_year(idx_b)}")
# 字符串 index
idx_s = [d.strftime("%Y-%m-%d") for d in idx_w]
check("字符串周频", infer_periods_per_year(idx_s) == 52, f"→ {infer_periods_per_year(idx_s)}")
# 退化: 空 / 单元素 / 非日期
check("空 index 回退", infer_periods_per_year(pd.Index([])) == 252)
check("单元素回退", infer_periods_per_year(pd.Index(["2025-01-01"])) == 252)
check("非日期回退", infer_periods_per_year(pd.Index(["a", "b", "c"])) == 252)
# 含 NaN 日频
idx_nan = idx_d.to_list() + [None, None]
check("含None日频", infer_periods_per_year(pd.Index(idx_nan)) == 252,
      f"→ {infer_periods_per_year(pd.Index(idx_nan))}")

print("\n═══ 2. compute_metrics 守卫自适应 ═══")
filt = S5JointFilter()
filt._index_df = pd.DataFrame({
    "trade_date": pd.date_range("2025-01-01", "2025-12-31", freq="B"),
    "idx_return": [0.0005] * 261,
})
# 月频: 一年 12 个观测 → 不误杀
rng = np.random.default_rng(7)
port_m = pd.Series(rng.normal(0.002, 0.02, 12),
                   index=pd.date_range("2025-01-31", "2025-12-31", freq="ME"))
m = filt.compute_metrics(port_m, 2025)
check("月频12观测不误杀", m["calmar"] != 0 and m["total_return"] != 0,
      f"→ ppy={m['periods_per_year']} calmar={m['calmar']:.3f}")
check("月频ppy=12", m["periods_per_year"] == 12)
# 周频: 一年 20 个观测 (<26) → 守卫触发
port_w = pd.Series(rng.normal(0.001, 0.01, 20),
                   index=pd.date_range("2025-01-03", "2025-05-16", freq="W-FRI"))
w = filt.compute_metrics(port_w, 2025)
check("周频20观测触发守卫", w["calmar"] == 0 and w["total_return"] == 0,
      f"→ ppy={w['periods_per_year']}")
# 周频: 完整一年 52 观测 → 正常
port_w52 = pd.Series(rng.normal(0.001, 0.01, 52),
                     index=pd.date_range("2025-01-03", "2025-12-26", freq="W-FRI"))
w52 = filt.compute_metrics(port_w52, 2025)
check("周频52观测正常", w52["calmar"] != 0 and w52["periods_per_year"] == 52,
      f"→ calmar={w52['calmar']:.3f}")
# 日频: 完整一年 → 正常, ppy=252
idx_yr = pd.bdate_range("2025-01-01", "2025-12-31")
port_d = pd.Series(rng.normal(0.0005, 0.01, len(idx_yr)), index=idx_yr)
d = filt.compute_metrics(port_d, 2025)
check("日频244观测正常", d["calmar"] != 0 and d["periods_per_year"] == 252,
      f"→ calmar={d['calmar']:.3f}")

print("\n═══ 3. 日频回归 (small_cap_liquidity_quality) ═══")
print("  (重跑 S5, 期望 cal25≈1.26 cal26≈2.46, 与旧口径一致)")
pool = pd.read_csv("data/passed_factor_pool.csv")
row = pool[pool["name"] == "small_cap_liquidity_quality"].iloc[0]
filt2 = S5JointFilter()
r = filt2.validate_factor("small_cap_liquidity_quality", str(row["formula"]))
check("回归 cal25", abs(r.calmar_2025 - 1.26) < 0.03, f"→ {r.calmar_2025:.2f}")
check("回归 cal26", abs(r.calmar_2026 - 2.46) < 0.03, f"→ {r.calmar_2026:.2f}")
check("回归 passed", r.passed, f"→ {r.passed}")

print(f"\n{'═'*50}\n结果: {PASS} 通过 / {FAIL} 失败")
sys.exit(1 if FAIL else 0)
