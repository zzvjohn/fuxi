# -*- coding: utf-8 -*-
"""
D+ 蒸馏手动触发 — comb_skew_vot_ar_pvt 4因子组合 JQ 实测结果注入
================================================================
JQ 实测 (2021-01-01 ~ 2026-09-30, 用户聚宽回测, 日志 E:\\log\\comb_skew_vot_ar_pvt.txt):
  组合累计收益 +203.25%, Sharpe 0.827, maxDD 27.13% (聚宽面板口径, 周频/池250/top30/等权rank合成)

4 成员因子无独立 JQ 回测 — jq_return/jq_sharpe/jq_maxdd 记 0 (JQ_MARGINAL, 非失败),
  但日志含 P-001 逐日 Rank IC/ICIR 全期报告 (n=281, 2026-08-03 最后一份):
    skewness(=fd_skewness)                        IC +0.0389  ICIR +0.3626  IC>0 65.5%
    vot(=fd_volatility_coefficient_of_turnover)   IC +0.0584  ICIR +0.5186  IC>0 72.6%
    ar_indicator(=fd_ar_indicator)                IC +0.0639  ICIR +0.4534  IC>0 65.1%
    pvt(=fd_price_volume_trend)                   IC +0.0693  ICIR +0.4714  IC>0 69.4%
  全部满足因子验证标准 (ICIR>=0.3 且 +IC%>=55%) → 跨源交叉印证成功。

组合本身不进 factors (防 seed_injector 自动入库污染: 组合公式非合法单因子表达式,
  历史教训 2026-08-23 king6 诊断伪因子污染); 组合成功由 composite 三字段承载
  (composite_assessment=STRONG_POSITIVE)。

用法: python scripts/run_d_plus_comb_skew_vot_ar_pvt.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

TRAJ_LOG = ROOT / "data" / "trajectory_log.json"

# ── 本地因子公式 (与 reports/fuxi_v1_crossyear_combo_20261011/run_crossyear_combo.py 逐字一致) ──
LOCAL_FORMULA = {
    "fd_skewness": "-(returns.rolling(120).skew())",
    "fd_ar_indicator": "-((high - open).rolling(26).sum() / (open - low).rolling(26).sum().clip(lower=1e-10) * 100)",
    "fd_volatility_coefficient_of_turnover": "amount.rolling(20).mean().div(amount.rolling(20).std())",
    "fd_price_volume_trend": "-((close.pct_change() * volume).cumsum())",
}

# ── 本地因子库读数 (data/stage1_factor_proposals.json) 与 JQ 日志 n=281 全期读数 ──
MEMBER = {
    "fd_skewness": {
        "local_ic": 0.0437, "local_icir": 0.9097,
        "jq_ic": 0.0389, "jq_icir": 0.3626, "jq_ic_pos": 0.655,
        "paradigm": "factors.directory tech", "category": "technical",
        "hypothesis": "120日收益偏度反转: 右偏(急涨)股未来回归, 买入左偏股; 本地6年全正因子, 组合基石",
    },
    "fd_volatility_coefficient_of_turnover": {
        "local_ic": 0.0576, "local_icir": 0.7323,
        "jq_ic": 0.0584, "jq_icir": 0.5186, "jq_ic_pos": 0.726,
        "paradigm": "波动率", "category": "liquidity",
        "hypothesis": "成交额变异系数倒数(成交额均值/标准差): 成交额平稳放量的股票后续收益更高, 惩罚成交额高波动",
    },
    "fd_ar_indicator": {
        "local_ic": 0.068, "local_icir": 0.4,
        "jq_ic": 0.0639, "jq_icir": 0.4534, "jq_ic_pos": 0.651,
        "paradigm": "技术指标反转", "category": "technical",
        "hypothesis": "量价AR人气指标反转: 上攻能量强(high-open占比高)的股票短期过热, 反转做空人气; 2026独立alpha来源",
    },
    "fd_price_volume_trend": {
        "local_ic": 0.0533, "local_icir": 0.3705,
        "jq_ic": 0.0693, "jq_icir": 0.4714, "jq_ic_pos": 0.694,
        "paradigm": "技术指标反转", "category": "technical",
        "hypothesis": "价量累积趋势反转: PVT持续上行的股票拥挤度累积, 反转; 2024专用alpha来源(组合互补)",
    },
}

JQ_COMPOSITE = {"ret": 203.25, "sharpe": 0.827, "maxdd": 27.13}
JQ_WINDOW = "2021-01-01 ~ 2026-09-30"


def infer_operators(formula: str):
    ops = []
    kws = {"pct_change": "pct_change", "rolling": "rolling", "rank": "rank",
           "std": "rolling_std", "mean": "rolling_mean", "skew": "rolling_skew",
           "sum": "rolling_sum", "cumsum": "cumsum", "clip": "clip",
           "div": "division", "abs": "abs", "/": "division",
           "*": "multiplication", "+": "addition", "-": "subtraction"}
    for kw, op in kws.items():
        if kw in formula.lower():
            ops.append(op)
    return ops


def build_jq_result() -> dict:
    factors = []
    for name, meta in MEMBER.items():
        factors.append({
            "factor_name": name,
            "formula": LOCAL_FORMULA[name],
            "hypothesis": meta["hypothesis"],
            "paradigm": meta["paradigm"],
            "category": meta["category"],
            "operators_used": infer_operators(LOCAL_FORMULA[name]),
            "local_ic": meta["local_ic"],
            "local_icir": meta["local_icir"],
            # 组合成员: 无独立 JQ 回测, 记 0 → JQ_MARGINAL (非失败, 非 PASS)
            "jq_return": 0.0,
            "jq_sharpe": 0.0,
            "jq_maxdd": 0.0,
            # P-001: 真实 per-factor IC/ICIR (JQ 日志 n=281 全期提取)
            "jq_ic": meta["jq_ic"],
            "jq_icir": meta["jq_icir"],
            "jq_composite_contribution": "positive",
            "root_cause": (
                f"组合 comb_skew_vot_ar_pvt 成员因子, 无独立 JQ 回测 (jq_return 记 0=MARGINAL, 非失败)。"
                f"JQ 逐日 Rank IC 全期 n=281: IC {meta['jq_ic']:+.4f} / ICIR {meta['jq_icir']:+.4f} "
                f"/ IC>0 {meta['jq_ic_pos']*100:.1f}% — 过验证标准 (ICIR>=0.3 且 +IC%>=55%), "
                f"与本地 IC {meta['local_ic']:+.4f}/{meta['local_icir']:+.4f} 方向一致 → 跨源交叉印证。"
                f"组合层 JQ {JQ_WINDOW}: +{JQ_COMPOSITE['ret']}% / Sharpe {JQ_COMPOSITE['sharpe']} "
                f"/ maxDD {JQ_COMPOSITE['maxdd']}% (STRONG_POSITIVE)。"
            ),
        })
    return {
        "batch_id": f"comb_skew_vot_ar_pvt_jq_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "composite_return": JQ_COMPOSITE["ret"],
        "composite_sharpe": JQ_COMPOSITE["sharpe"],
        "composite_maxdd": JQ_COMPOSITE["maxdd"],
        "factors": factors,
    }


def mark_new_traj_d_plus():
    """jq_feedback 新建的轨迹 (phase=jq_feedback) 标记 d_plus_applied=true"""
    data = json.load(open(TRAJ_LOG, encoding="utf-8"))
    now = datetime.now().isoformat()
    count = 0
    for t in data.get("trajectories", []):
        if t.get("phase") == "jq_feedback" and not t.get("d_plus_applied"):
            t["d_plus_applied"] = True
            t["d_plus_at"] = now[:19]
            count += 1
    data["updated_at"] = now
    json.dump(data, open(TRAJ_LOG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  标记 {count} 条新轨迹 d_plus_applied=true")
    return count


def main():
    print("=" * 64)
    print("  D+ 蒸馏: comb_skew_vot_ar_pvt JQ 实测结果注入")
    print(f"  JQ 窗口 {JQ_WINDOW} | 组合 +{JQ_COMPOSITE['ret']}% / "
          f"Sharpe {JQ_COMPOSITE['sharpe']} / maxDD {JQ_COMPOSITE['maxdd']}%")
    print("=" * 64)

    jq_result = build_jq_result()
    for f in jq_result["factors"]:
        print(f"    {f['factor_name']}: JQ IC {f['jq_ic']:+.4f} / ICIR {f['jq_icir']:+.4f} "
              f"(local {f['local_ic']:+.4f}/{f['local_icir']:+.4f}) | {f['jq_composite_contribution']}")

    from ralph_loop import RalphLoop
    from experience_memory import get_memory

    memory = get_memory()
    rl = RalphLoop(memory=memory)
    result = rl.jq_feedback(jq_backtest_result=jq_result)

    marked = mark_new_traj_d_plus()

    print(f"\n{'='*64}")
    print(f"  D+ 完成")
    print(f"    轨迹更新:      {result.get('trajectories_updated', 0)}")
    print(f"    硬禁止方向:    +{result.get('hard_forbidden_added', 0)}")
    print(f"    软负警告:      +{result.get('soft_warnings_added', 0)}")
    print(f"    JQ确认成功:    +{result.get('jq_success_confirmed', 0)}")
    print(f"    MAB streak:    +{result.get('mab_streak_updates', 0)}")
    print(f"    d_plus 标记:   {marked}")
    print(f"    Memory 记录:   {memory.data['stats'].get('total_attempts')}")
    print(f"{'='*64}")

    summary = {
        "batch_id": jq_result["batch_id"],
        "jq_window": JQ_WINDOW,
        "composite": {
            "return": jq_result["composite_return"],
            "sharpe": jq_result["composite_sharpe"],
            "maxdd": jq_result["composite_maxdd"],
        },
        "result": {k: result.get(k) for k in
                   ("trajectories_updated", "hard_forbidden_added",
                    "soft_warnings_added", "jq_success_confirmed",
                    "mab_streak_updates")},
        "factors": [
            {"factor_name": f["factor_name"], "jq_return": f["jq_return"],
             "jq_sharpe": f["jq_sharpe"], "jq_ic": f["jq_ic"],
             "jq_icir": f["jq_icir"],
             "contribution": f["jq_composite_contribution"]}
            for f in jq_result["factors"]
        ],
    }
    out = ROOT / "data" / "jq_feedback_comb_skew_vot_ar_pvt.json"
    json.dump(summary, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  摘要已存: {out.name}")
    return 0


if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    # 共享研究状态 (trajectory_log.json / experience_memory.json) 写入须持 governed_writer 锁
    # (assert_state_writer: feedback action 在放行集合; 触发=manual, 与 trigger_d_plus 同动作域)
    from fuxi_v1.governance import guard_cli as _guard_cli
    with _guard_cli(__file__, "feedback"):
        _sys.exit(main())
