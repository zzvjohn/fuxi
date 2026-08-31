# -*- coding: utf-8 -*-
"""
计算 7 个已送 JQ 因子的「2025-2026 合并区间」Calmar (跨年度合并口径)。

对比三种口径:
  1. 旧分年口径 (bug): annual_factor=252 应用于周频数据 → Calmar 虚高 4.846x
  2. 修复分年口径: annual_factor=52 (周频), 每年单独算
  3. 合并口径: 2025-01-01~2026-08-07 连续区间, 年化收益/最大回撤

注意: 合并口径下 2026 只有约 31 周 (数据截止 2026-08-07),
      年化 = (1+TR)^(52/n) - 1, n≈83 周。
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "research/factor_alchemy")
import numpy as np
import pandas as pd
from s5_joint_filter import S5JointFilter

filt = S5JointFilter()

# ── 收集 7 个因子 ──────────────────────────────
factors = []
pool = pd.read_csv("data/passed_factor_pool.csv")
for n in ["small_cap_liquidity_quality", "event_driven_convexity_fade",
          "max_drawdown_duration"]:
    row = pool[pool["name"] == n].iloc[0]
    factors.append((n, str(row["formula"])))

rows = json.load(open("data/forge_round1_s5_results.json", encoding="utf-8"))["rows"]
forge_map = {r["factor_name"]: r for r in rows}
forge_names = {
    "forge_r1_vol_zscore_inv_stability": "forge_r1_筹码分布_53790",
    "forge_r1_vol_double_zscore_inv_stability": "forge_r1_筹码分布_7495",
    "forge_r1_open_min_vol_squared": "forge_r1_筹码分布_83717",
    "forge_r1_volume_dual_rank": "forge_r1_尾部风险_73961",
}
for jq_n, internal in forge_names.items():
    factors.append((jq_n, forge_map[internal]["pandas_expression"]))

# ── monkey-patch: 捕获 port_ret 序列 ────────────
captured = {}
_orig_cm = S5JointFilter.compute_metrics

def _patched_cm(self, port_ret, year):
    captured["_port_ret"] = port_ret
    captured["_index_df"] = self._index_df
    return _orig_cm(self, port_ret, year)

S5JointFilter.compute_metrics = _patched_cm

PPY = 52          # 周频
OLD_AMP = 252 / 52  # 旧口径放大倍数

def merged_stats(port_ret, idx_df):
    """2025-01-01 ~ 2026-08-07 合并区间的整体统计"""
    seg = port_ret[(port_ret.index >= "2025-01-01") &
                   (port_ret.index <= "2026-08-07")]
    if len(seg) < 20:
        return None
    tr = (1 + seg).prod() - 1
    cum = (1 + seg).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    ann = (1 + tr) ** (PPY / len(seg)) - 1
    calmar = ann / abs(mdd) if mdd < 0 else float("nan")
    idx_seg = idx_df[(idx_df["trade_date"] >= "2025-01-01") &
                     (idx_df["trade_date"] <= "2026-08-07")]
    bench = (1 + idx_seg["idx_return"].dropna()).prod() - 1 if len(idx_seg) else 0.0
    return dict(n=len(seg), total_return=tr, mdd=mdd, ann=ann, calmar=calmar,
                bench=bench, excess=tr - bench)

hdr = (f"{'因子':<42} {'两年收益':>8} {'超额':>7} {'两年MDD':>7} {'年化':>6} "
       f"{'合并Cal':>7} | {'旧cal25':>7} {'旧cal26':>7} {'修复cal25':>7} {'修复cal26':>7}")
print(hdr)
print("-" * len(hdr))
out = {}
for name, formula in factors:
    try:
        r = filt.validate_factor(name, formula)
    except Exception as e:
        print(f"{name:<42} 异常: {str(e)[:60]}")
        continue
    pr = captured.get("_port_ret")
    idx = captured.get("_index_df")
    ms = merged_stats(pr, idx) if pr is not None else None
    fix25 = r.calmar_2025 / OLD_AMP if r.calmar_2025 else 0.0
    fix26 = r.calmar_2026 / OLD_AMP if r.calmar_2026 else 0.0
    if ms:
        print(f"{name:<42} {ms['total_return']*100:>7.1f}% {ms['excess']*100:>6.2f}% "
              f"{abs(ms['mdd'])*100:>6.1f}% {ms['ann']*100:>5.2f}% {ms['calmar']:>7.2f} | "
              f"{r.calmar_2025:>7.2f} {r.calmar_2026:>7.2f} {fix25:>7.2f} {fix26:>7.2f}")
        out[name] = {k: (float(v) if isinstance(v, (int, float, np.floating))
                         else v) for k, v in ms.items()}
        out[name]["old_calmar_2025"] = float(r.calmar_2025)
        out[name]["old_calmar_2026"] = float(r.calmar_2026)
    else:
        print(f"{name:<42} 合并区间观测不足")

json.dump(out, open("data/s5_merged_calmar_2526.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已保存 → data/s5_merged_calmar_2526.json")
