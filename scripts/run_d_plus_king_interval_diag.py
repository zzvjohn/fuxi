# -*- coding: utf-8 -*-
"""
D+ 蒸馏手动触发 — King 区间版方法归因诊断 JQ 实测注入 (2026-08-23)
==================================================================
JQ 实测 (用户回测, 文件 jq_combo_king6_standalone_2024_2025.py):
  区间 2024-01-01 ~ 2025-11-30 (与 LightGBM 策略同期 23 个月)
  King 线性 rank积 6因子: +57.42%, Sharpe 0.77, MaxDD 29.21%

对照: LightGBM 策略同期 +195.32% / Sharpe 2.112 / MaxDD ~24% (风控后)

判决: 差距真实。+57.42% 落在 +40~60% 判决带 → "窄池条件化非线性"是真增量,
      非行情红利 → 立项条件化因子线 (direction_confirmed)。

关键论证:
  1. 3.4x 收益 / 2.7x Sharpe 差距;
  2. α=0.708 折纯 alpha Sharpe ≈ 1.78, 仍 2.3x King, beta 解释不了;
  3. 2024-2025 恰是价量族友好期 (King 全区间 Sharpe 0.69 → 同期 0.77),
     有利行情下差距依然存在 → 差距非 King 状态差所致;
  4. 剩余折扣项 (集中度 8 vs 30 只 / 微盘池 beta / 幸存者偏差) 无法闭合 2.7x Sharpe 缺口。

注入方式: 诊断型 jq_feedback (裁决 direction_confirmed), 不误入库因子;
          champions 注册表新增 king6_interval_2024_2025_diag 组合记录。

用法: python scripts/run_d_plus_king_interval_diag.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

TRAJ_LOG = ROOT / "data" / "trajectory_log.json"
CHAMPIONS = ROOT / "data" / "jq_validated_champions.json"

COMBO_RET = 57.42
COMBO_SR = 0.77
COMBO_MDD = 29.21

VERDICT = (
    "method_gap_confirmed: 同区间方法对照决定性实验。King 线性 rank积 6因子 "
    "2024-01~2025-11 仅 +57.42%/Sharpe 0.77/MaxDD 29.21%, 对照 LightGBM 条件化非线性 "
    "+195.32%/Sharpe 2.112/MaxDD~24% → 3.4x 收益 / 2.7x Sharpe 差距。"
    "α=0.708 折纯 alpha Sharpe≈1.78 仍 2.3x King, beta 解释不了; "
    "且 2024-2025 是价量族友好期 (King 全区间 0.69→同期 0.77), 有利行情下差距仍存。"
    "→ 窄池条件化非线性优选是真实方法增量, 非行情红利。"
    "立项条件化因子线: LLM 分段公式 (if-else 条件交互) + 超跌池 + NDrop 执行层。"
)


def build_payload() -> dict:
    return {
        "batch_id": f"jq_king6_interval_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "composite_return": COMBO_RET,
        "composite_sharpe": COMBO_SR,
        "composite_maxdd": COMBO_MDD,
        "combo_note": VERDICT,
        "factors": [
            {
                "factor_name": "king6_interval_2024_2025_diagnostic",
                "formula": (
                    "overnight5+tvma20+dollar20+tstd_cv+ret3m+nearhigh "
                    "rank-product (King 组合诊断, 非单因子, 不用于因子入库)"
                ),
                "hypothesis": (
                    "方法归因: 线性 rank积组合在 LightGBM 策略同期区间的表现, "
                    "判定'窄池条件化非线性'增量是否真实。判决带: +40~60% → 差距真实; "
                    "+100%+ → 行情红利"
                ),
                "paradigm": "diagnostic_interval",
                "category": "method_attribution",
                "operators_used": ["rank_product", "rolling_mean", "pct_change"],
                "local_ic": None,
                "local_icir": None,
                "jq_return": COMBO_RET,
                "jq_sharpe": COMBO_SR,
                "jq_maxdd": COMBO_MDD,
                "jq_ic": None,
                "jq_icir": None,
                "jq_composite_contribution": "neutral",
                "root_cause": "direction_confirmed: " + VERDICT,
                "interpretation": {
                    "verdict": "direction_confirmed",
                    "confidence": "high",
                    "summary": (
                        "King 同期 +57.42%/0.77 落判决带 +40~60% → 条件化非线性增量真实。"
                        "下一步: 立项条件化因子线 P0 (LLM 分段公式生成 + 超跌池内求值 + "
                        "top8 优选项验证), 目标证明分段公式可逼近树模型交互项"
                    ),
                },
            }
        ],
    }


def record_combo_in_champions():
    if not CHAMPIONS.exists():
        return
    data = json.load(open(CHAMPIONS, encoding="utf-8"))
    existing = {str(it.get("name", "")) for it in data if isinstance(it, dict)}
    name = "king6_interval_2024_2025_diag"
    if name in existing:
        print("  champions 注册表已存在, 跳过")
        return
    data.append({
        "name": name,
        "paradigm": "diagnostic_interval",
        "jq_return": COMBO_RET,
        "jq_sharpe": COMBO_SR,
        "jq_maxdd": COMBO_MDD,
        "note": (
            "方法归因决定性实验 (区间 2024-01-01~2025-11-30, 与 LightGBM 策略同期)。"
            "King 线性 rank积 +57.42%/0.77/29.21% vs LightGBM 条件化非线性 "
            "+195.32%/2.112/~24% → 3.4x/2.7x 差距, 条件化非线性为真实增量。"
            "判决: 立项条件化因子线 (LLM 分段公式 + 超跌池 + NDrop 执行层)"
        ),
        "recorded_at": datetime.now().isoformat()[:19],
    })
    json.dump(data, open(CHAMPIONS, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("  champions 注册表新增: king6_interval_2024_2025_diag")


def cleanup_auto_injected_pseudo_factor():
    """jq_feedback 自动入库机制 (Sharpe>=0.4 门槛) 会把诊断伪因子注入注册表,
    名称为 '<factor_name>_JQ验证' 且无 jq_return/jq_sharpe 字段。诊断注入后必须清理。"""
    if not CHAMPIONS.exists():
        return
    data = json.load(open(CHAMPIONS, encoding="utf-8"))
    bad = [it for it in data if isinstance(it, dict)
           and str(it.get("name", "")).startswith("king6_interval_2024_2025_diagnostic")
           and it.get("jq_return") is None]
    if bad:
        data = [it for it in data if it not in bad]
        json.dump(data, open(CHAMPIONS, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  清理自动入库伪因子 {len(bad)} 条 (诊断注入防污染)")


def mark_new_traj_d_plus():
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
    print("=" * 60)
    print("  D+ 蒸馏: King 区间版方法归因诊断注入")
    print("=" * 60)

    payload = build_payload()
    print(f"  King interval: {payload['composite_return']}% / "
          f"Sharpe {payload['composite_sharpe']} / MDD {payload['composite_maxdd']}%")

    from ralph_loop import RalphLoop

    rl = RalphLoop()
    result = rl.jq_feedback(jq_backtest_result=payload)
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))

    out = ROOT / "data" / "jq_feedback_king_interval_diag.json"
    json.dump({"input": payload, "feedback_result": result},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"  结果已存: {out}")

    record_combo_in_champions()
    cleanup_auto_injected_pseudo_factor()
    mark_new_traj_d_plus()
    print("D+ 闭环完成")


if __name__ == "__main__":
    main()
