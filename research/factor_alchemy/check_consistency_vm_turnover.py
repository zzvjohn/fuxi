# -*- coding: utf-8 -*-
"""
vm_diff / turnover_std 本地↔JQ 一致性核对
========================================
两层验证:
  A) 快照级: 直接执行 registry 中的 JQ 生成函数, 与本地因子类在最新截面比对
  B) 全样本 ICIR 级: 用 JQ 等价方法复现整段时间序列, 比对 ICIR/截面相关
重点: 暴露 ewm(adjust=True vs False) 约定差异 与 turnover_std 的 volume-CV 代理有效性
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from data.loader import load_all_data
from factors.advanced_technical import VMDiff
from factors.turnover import TurnoverStd
from jq_generator.registry import JQ_FACTOR_REGISTRY

np.set_printoptions(suppress=True)


def weekly_factor(factor_df_daily):
    """日频因子 -> 周频(取每周最后交易日) 截面"""
    return factor_df_daily.resample('W-FRI').last().iloc[1:]


def compute_icir(factor_w, fwd_ret_w):
    """逐周截面 Spearman IC, 返回 (ICIR均值, IC均值, 通过周数)"""
    fwd_ret_w = fwd_ret_w.reindex(index=factor_w.index, columns=factor_w.columns)
    ics = []
    for t in range(len(factor_w)):
        f = factor_w.iloc[t].values
        r = fwd_ret_w.iloc[t].values
        m = np.isfinite(f) & np.isfinite(r)
        if m.sum() < 30:
            continue
        ic, _ = spearmanr(f[m], r[m])
        if np.isfinite(ic):
            ics.append(ic)
    if len(ics) < 10:
        return 0.0, 0.0, 0
    ics = np.array(ics)
    return ics.mean() / max(ics.std(ddof=1), 0.001), ics.mean(), len(ics)


def xcorr(a, b):
    """两序列按共同有限截面算 Spearman 相关"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    return spearmanr(a[m], b[m])[0]


print("=" * 70)
print("加载数据 (2021-2026, 子集以提高速度)...")
data = load_all_data('2021-01-01', '2026-04-30')
price = data['price_data']
vol = price['volume']
to = price['turnover']
close = price['close']

# 取子集: 流动性较好的前 N 只
subset = list(vol.columns[:500])
vol = vol[subset]; to = to[subset] if hasattr(to, 'columns') else to
close = close[subset]
print(f"  子集: {len(subset)} 只, 日期 {vol.shape[0]} 天")

# 周频前向收益 (1周)
close_w = close.resample('W-FRI').last().iloc[1:]
fwd_ret_w = close_w.shift(-1) / close_w - 1
fwd_ret_w = fwd_ret_w.iloc[:-1]

# ============================================================
# vm_diff
# ============================================================
print("\n" + "=" * 70)
print("【vm_diff】")
# 本地 (ewm adjust=True, 实际GA训练用法)
vm_local = VMDiff().compute({'volume': vol})          # 全序列
vm_local_w = weekly_factor(vm_local)                  # 周频
# JQ 等价 (递归EMA == ewm adjust=False) 全序列
vm_jq = -(vol.ewm(span=12, adjust=False).mean() / vol.ewm(span=26, adjust=False).mean() - 1).clip(-5, 5)
vm_jq_w = weekly_factor(vm_jq)

icir_l, ic_l, n_l = compute_icir(vm_local_w, fwd_ret_w)
icir_j, ic_j, n_j = compute_icir(vm_jq_w, fwd_ret_w)
print(f"  本地(adjust=True)  ICIR={icir_l:+.3f}  IC均值={ic_l:+.4f}  n={n_l}")
print(f"  JQ(adjust=False)    ICIR={icir_j:+.3f}  IC均值={ic_j:+.4f}  n={n_j}")
print(f"  ICIR 差异           {icir_l - icir_j:+.3f}  (相对 { (icir_l-icir_j)/max(abs(icir_l),1e-9)*100:+.1f}% )")
# 快照级(最新截面) 相关
last_l = vm_local_w.iloc[-1]; last_j = vm_jq_w.iloc[-1]
print(f"  最新截面 Spearman 相关 = {xcorr(last_l, last_j):+.4f}")

# 直接执行 registry 的 JQ 生成函数 (快照级, 证明生成代码本身正确)
print("  -- 直接执行 registry 生成函数 (最新截面) --")
reg = JQ_FACTOR_REGISTRY['vm_diff']['jq_compute_function']
ns = {'np': np}; exec(reg, ns)
stocks = list(vol.columns)
vol_dict = {s: vol[s].dropna().values.tolist() for s in stocks}
arr, valid = ns['compute_vm_diff'](stocks, {'volume': vol_dict})
jq_snap = pd.Series(arr, index=stocks)
print(f"  registry函数 vs 本地快照 Spearman 相关 = {xcorr(last_l.reindex(stocks), jq_snap):+.4f}  (有效 {valid.sum()}/{len(stocks)})")

# ============================================================
# turnover_std
# ============================================================
print("\n" + "=" * 70)
print("【turnover_std】")
# 本地: TurnoverStd (turnover -> -std20d -> 截面zscore)
ts_local = TurnoverStd().compute({'turnover': to}, None, None)
ts_local_w = weekly_factor(ts_local)
# JQ 等价: volume 的 CV(std/mean) 取负 (registry 实际生成)
cv_vol = (vol.rolling(20).std() / vol.rolling(20).mean()).replace([np.inf, -np.inf], np.nan)
ts_jq = -cv_vol
ts_jq_w = weekly_factor(ts_jq)
# 中间量: 本地原始 -std(turnover) (未zscore) 用于验证代理
std_to = -(to.rolling(20).std())
std_to_w = weekly_factor(std_to)

icir_tl, ic_tl, n_tl = compute_icir(ts_local_w, fwd_ret_w)
icir_tj, ic_tj, n_tj = compute_icir(ts_jq_w, fwd_ret_w)
print(f"  本地(turnover,zscore) ICIR={icir_tl:+.3f}  IC均值={ic_tl:+.4f}  n={n_tl}")
print(f"  JQ(volume CV)         ICIR={icir_tj:+.3f}  IC均值={ic_tj:+.4f}  n={n_tj}")
print(f"  ICIR 差异             {icir_tl - icir_tj:+.3f}")
last_tl = ts_local_w.iloc[-1]; last_tj = ts_jq_w.iloc[-1]
print(f"  最新截面 Spearman 相关 (zscore本地 vs raw JQ) = {xcorr(last_tl, last_tj):+.4f}")
last_std = std_to_w.iloc[-1]
print(f"  代理有效性: -std(turnover) vs -cv(volume) 最新截面相关 = {xcorr(last_std, last_tj):+.4f}")
print(f"  (相关越接近1, 说明 volume-CV 是 turnover-std 的有效标度无关代理)")

# 直接执行 registry 生成函数
print("  -- 直接执行 registry 生成函数 (最新截面) --")
reg2 = JQ_FACTOR_REGISTRY['turnover_std']['jq_compute_function']
ns2 = {'np': np}; exec(reg2, ns2)
vol_dict2 = {s: vol[s].dropna().values.tolist() for s in stocks}
arr2, valid2 = ns2['compute_turnover_std'](stocks, {'volume': vol_dict2})
jq_turn_snap = pd.Series(arr2, index=stocks)
print(f"  registry函数 vs JQ-CV快照 Spearman 相关 = {xcorr(last_tj.reindex(stocks), jq_turn_snap):+.4f}  (有效 {valid2.sum()}/{len(stocks)})")

print("\n" + "=" * 70)
print("结论速览:")
print(f"  vm_diff:      本地/JQ ICIR差 {icir_l-icir_j:+.3f}; 快照相关(本地vs registry)={xcorr(last_l.reindex(stocks), jq_snap):+.3f}")
print(f"  turnover_std: 本地/JQ ICIR差 {icir_tl-icir_tj:+.3f}; 代理有效性={xcorr(last_std, last_tj):+.3f}")
