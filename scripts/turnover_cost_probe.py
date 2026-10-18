"""
P-20260909-001: S5 换手率成本敏感性影子检查 (金融毒性标记)

纯影子零裁决: 对 S5 通过候选估算年化成本拖累,
cost_drag > excess_2026 的 50% → 标记 cost_sensitive 进日报风险预警。
不改变任何裁决结果, S5 照过、JQ 代码照生成。

成本模型 (影子口径, 非实盘精确, 局限见下):
- 信号换手: 因子截面 rank 后逐股 lag1 自相关 rho → 日单边换手 ≈ (1 - rho)
  (信号自相关越低, 每日被替换的持仓越多)
- 年化单边换手 = (1 - rho) * 252
- 单边成本 = 21bp = (开佣金 0.001 + 平税 0.0001 + 平佣金 0.0001)/2 + 滑点 0.003/2
  (对齐 JQ 口径 OrderCost(0, 0.001, 0.0001, 0.0001) + FixedSlippage(0.003))
- cost_drag_annual = 年化单边换手 * 21bp
- 标记阈值: cost_drag > 0.5 * excess_2026 (缺失则回退 excess_2025;
  两者皆缺 → 标记 insufficient_data, 不误报)

口径局限 (观测期 2 周, 阈值校准后再评估升级硬门禁):
1. 换手用信号自相关近似, 未计涨跌停/停牌/容量限制 → 真实成本可能更高;
2. 组合持股数/调仓权重渐变未建模;
3. 与 JQ 面板回测的净口径结果冲突时, 以 JQ 结果为准 (本脚本只是提前预警)。

参考文献:
- PandaAI (arXiv 2026-09): 因子6 原始 ICIR 0.248 > 完整版 0.193, 但日换手 >80%,
  计 15bp 佣金 + 5bp 滑点后净表现显著下降 ("金融毒性");
- Novy-Marx & Velikov (2016) 交易成本过滤器: 400 异象计成本后仅 15-30% 存活。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_FHERE = Path(__file__).resolve()
_PROJ_ROOT = _FHERE.parents[1]
_FA_DIR = _PROJ_ROOT / "research" / "factor_alchemy"
if str(_FA_DIR) not in sys.path:
    sys.path.insert(0, str(_FA_DIR))

# 单边成本率 (对齐 JQ OrderCost + FixedSlippage, 影子口径)
UNIT_COST = 0.0021
# 标记阈值: cost_drag > 该比例 * excess
COST_SENSITIVE_RATIO = 0.5
# 数据窗口 (与 FactorICComputer 标准 270d 对齐)
LOOKBACK_DAYS = 270
TRADING_DAYS_PER_YEAR = 252

_SHADOW_PATH = _PROJ_ROOT / "data" / "turnover_cost_shadow.json"


def _signal_rho(signal_wide: pd.DataFrame) -> float:
    """宽表信号 (index=日期, columns=股票) 的截面平均 lag1 自相关。

    先截面 rank(pct) 再逐股算 lag1 自相关, 截面均值 rho。
    rho 低 = 信号换手快 = 成本高。
    """
    rank_df = signal_wide.rank(axis=1, pct=True)
    # 逐股: rank 序列与 shift(1) 的相关系数 (要求 ≥10 个非空重叠观测)
    rhos = []
    for col in rank_df.columns:
        x = rank_df[col]
        x_lag = x.shift(1)
        valid = x.notna() & x_lag.notna()
        n = int(valid.sum())
        if n < 10:
            continue
        c = x[valid].corr(x_lag[valid])
        if pd.notna(c):
            rhos.append(c)
    if not rhos:
        return 0.0
    return float(np.clip(np.mean(rhos), -1.0, 1.0))


def probe_one(
    formula: str,
    factor_name: str,
    excess_ref: float,
    ic_computer=None,
) -> dict:
    """对单个 S5 通过候选做成本影子测算。

    Returns:
        {factor_name, signal_rho, annual_turnover, cost_drag_annual,
         excess_ref, cost_sensitive, status}
    """
    if not formula:
        return {
            "factor_name": factor_name, "status": "no_formula",
            "cost_sensitive": False,
        }
    if ic_computer is None:
        from factor_ic_computer import FactorICComputer
        ic_computer = FactorICComputer(lookback_days=LOOKBACK_DAYS)
    try:
        long_df = ic_computer._eval_formula(formula, factor_name)
        if long_df is None or len(long_df) == 0:
            return {
                "factor_name": factor_name, "status": "eval_failed",
                "cost_sensitive": False,
            }
        wide = long_df.pivot(
            index="trade_date", columns="ts_code", values="factor_value"
        )
        rho = _signal_rho(wide)
        annual_turnover = max(0.0, (1.0 - rho) * TRADING_DAYS_PER_YEAR)
        cost_drag = annual_turnover * UNIT_COST
        out = {
            "factor_name": factor_name,
            "signal_rho": round(rho, 4),
            "annual_turnover": round(annual_turnover, 3),
            "cost_drag_annual": round(cost_drag, 4),
            "excess_ref": round(float(excess_ref or 0.0), 4),
            "status": "ok",
        }
        out["cost_sensitive"] = bool(
            excess_ref and cost_drag > COST_SENSITIVE_RATIO * excess_ref
        )
        return out
    except Exception as e:
        return {
            "factor_name": factor_name, "status": f"error: {type(e).__name__}: {str(e)[:80]}",
            "cost_sensitive": False,
        }


def probe_candidates(details: list) -> list:
    """对 S5 通过候选批量影子测算 (共享一个 FactorICComputer, 只读不裁决)。

    落盘 data/turnover_cost_shadow.json (按 factor_name 累积最新记录)。
    """
    if not details:
        return []
    s5_passed = [d for d in details if d.get("s5_passed")]
    if not s5_passed:
        return []

    from factor_ic_computer import FactorICComputer
    try:
        comp = FactorICComputer(lookback_days=LOOKBACK_DAYS)
    except Exception as e:
        print(f"  [CostShadow] 数据加载失败 (非阻塞): {e}")
        return []

    report = []
    for d in s5_passed:
        name = d.get("factor_name", "?")
        formula = d.get("formula", "") or ""
        excess = float(
            d.get("s5_year2_excess")
            or d.get("s5_year1_excess")
            or 0.0
        )
        res = probe_one(formula, name, excess, ic_computer=comp)
        report.append(res)

    # 落盘影子记录 (幂等: 同名覆盖为最新)
    try:
        store = {}
        if _SHADOW_PATH.exists():
            try:
                store = json.load(open(_SHADOW_PATH, encoding="utf-8"))
                if not isinstance(store, dict):
                    store = {}
            except Exception:
                store = {}
        for r in report:
            if r.get("status") == "ok":
                store[r["factor_name"]] = r
        json.dump(store, open(_SHADOW_PATH, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"  [CostShadow] 影子落盘失败 (非阻塞): {e}")

    return report


if __name__ == "__main__":
    # 独立冒烟: 读最新 ralph_loop_result_*.json 的 jq_candidate_details
    import glob
    import os

    results = sorted(
        glob.glob(str(_PROJ_ROOT / "data" / "ralph_loop_result_*.json")),
        key=os.path.getmtime, reverse=True,
    )
    if not results:
        print("无 ralph_loop_result 文件")
        sys.exit(0)
    latest = results[0]
    print(f"读取: {os.path.basename(latest)}")
    r = json.load(open(latest, encoding="utf-8"))
    details = r.get("jq_candidate_details", [])
    if not details:
        print("该轮无 S5 通过候选, 无影子标记")
        sys.exit(0)
    rep = probe_candidates(details)
    for c in rep:
        if c.get("status") != "ok":
            print(f"  [{c['factor_name']}] {c['status']}")
        else:
            flag = "COST_SENSITIVE" if c["cost_sensitive"] else "ok"
            print(
                f"  [{flag}] {c['factor_name']} "
                f"cost_drag={c['cost_drag_annual']*100:.2f}%/yr "
                f"vs excess={c['excess_ref']*100:.2f}% "
                f"(rho={c['signal_rho']:.3f}, ann_turnover={c['annual_turnover']:.1f}x)"
            )
