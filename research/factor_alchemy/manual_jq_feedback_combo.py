# -*- coding: utf-8 -*-
"""
手动 D+ 蒸馏: 统一因子池组合探索 (king6 / mmr6 / recon6) JQ 回测结果注入
────────────────────────────────────────────────────────────────────
背景 (2026-08-17~18):
  王者 8 底仓 + JQ PASS/MARGINAL 候选混池 → MMR 贪心组合搜索 (_combo_search.py)
  三个 6 因子组合经 JQ 平台回测 (pool500/top30/lb70/周频/手数下单):
    king6  (王者压缩6):  overnight5+tvma20+dollar20+tstd_cv+ret3m+nearhigh
    mmr6   (全池自由MMR): breed002+overnight5+tstd_cv+nearhigh+dollar20+tvma20
    recon6 (受限重构6):   breed002+overreac+mf20+nearhigh+overnight5+dollar20
  本脚本: 组合级 JQ 结果注入 D+ 闭环 (trajectory + experience memory),
          并做组合探索方法论知识沉淀 (本地排序 vs JQ 排序验证)。
注意: 组合级结果不作为单因子入库 (formula 留空 → 跳过 seed_injector 自动入库)。
用法: cd research/factor_alchemy && python manual_jq_feedback_combo.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# ── 三个组合的定义与本地/JQ 对照 ─────────────────────────
COMBOS = [
    {
        "name": "combo_king6",
        "members": "overnight5 + tvma20 + dollar20 + tstd_cv + ret3m + nearhigh",
        "desc": "王者8因子MMR压缩版 (裁剪共线量能系 mf20/ampvol, 内部|corr|均值0.096)",
        "local_ret": 456.9, "local_sharpe": 1.203, "local_mdd": None,
        "jq_ret": 150.97, "jq_sharpe": 0.58, "jq_maxdd": 35.88,
    },
    {
        "name": "combo_mmr6",
        "members": "breed002 + overnight5 + tstd_cv + nearhigh + dollar20 + tvma20",
        "desc": "全池自由MMR贪心 (候选SR>=0.25门槛, 与已选最大|corr|最小优先)",
        "local_ret": 258.9, "local_sharpe": 0.894, "local_mdd": None,
        "jq_ret": 127.04, "jq_sharpe": 0.50, "jq_maxdd": 36.43,
    },
    {
        "name": "combo_recon6",
        "members": "breed002 + overreac + mf20 + nearhigh + overnight5 + dollar20",
        "desc": "受限重构 (必选最正交候选对 breed002+overreac, 混合王者+候选)",
        "local_ret": 150.3, "local_sharpe": 0.732, "local_mdd": None,
        "jq_ret": 118.63, "jq_sharpe": 0.50, "jq_maxdd": 35.79,
    },
]

KING8 = {"ret": 182.57, "sharpe": 0.69, "mdd": 35.41}  # AlphaAgent v3 完整王者8因子 JQ 历史


def make_feedback(combo: dict) -> dict:
    return {
        "batch_id": f"jq_combo_{combo['name']}_20260818",
        "timestamp": datetime.now().isoformat(),
        "test_method": "jq_backtest_platform",
        "test_window": "2021-01-01~2026-07-30",
        "composite_return": combo["jq_ret"],
        "composite_sharpe": combo["jq_sharpe"],
        "composite_maxdd": combo["jq_maxdd"],
        "factors": [
            {
                "factor_name": combo["name"],
                "formula": "",  # 留空: 组合描述不得作为单因子公式入库
                "hypothesis": (
                    f"统一因子池组合探索: {combo['desc']}. 成员: {combo['members']}. "
                    f"本地MMR模拟 {combo['local_ret']:.1f}%/SR{combo['local_sharpe']:.3f} "
                    f"→ JQ实测 {combo['jq_ret']:.1f}%/SR{combo['jq_sharpe']:.2f}."
                ),
                "paradigm": "combo_exploration",
                "category": "portfolio_combo",
                "operators_used": [],
                "local_ic": 0.0,
                "local_icir": 0.0,
                "jq_return": combo["jq_ret"],
                "jq_sharpe": combo["jq_sharpe"],
                "jq_maxdd": combo["jq_maxdd"],
                "jq_composite_contribution": "positive",
                "root_cause": (
                    f"组合级JQ验证: +{combo['jq_ret']:.2f}%/SR{combo['jq_sharpe']:.2f}/"
                    f"MDD{combo['jq_maxdd']:.2f}% (2021-2026). "
                    f"跑输王者完整8因子 (+{KING8['ret']:.2f}%/SR{KING8['sharpe']:.2f}): "
                    f"现有因子池内重组无法突破王者 (稀释定理 JQ 证实)。"
                ),
                "jq_window": "2021-01-01~2026-07-30",
            }
        ],
    }


def main():
    from experience_memory import get_memory
    from ralph_loop import RalphLoop

    memory = get_memory()
    rl = RalphLoop(memory=memory)

    print("=" * 70)
    print("  D+: 组合探索 JQ 反馈 (king6 / mmr6 / recon6)")
    print("=" * 70)

    # ── 1. 逐个组合走标准 jq_feedback 闭环 ──
    all_results = {}
    for combo in COMBOS:
        print(f"\n>>> {combo['name']}  JQ +{combo['jq_ret']:.2f}%/SR{combo['jq_sharpe']:.2f}")
        res = rl.jq_feedback(jq_backtest_result=make_feedback(combo))
        all_results[combo["name"]] = res
        print(f"    轨迹更新={res['trajectories_updated']} "
              f"硬禁止={res['hard_forbidden_added']} "
              f"确认成功={res['jq_success_confirmed']}")

    # ── 2. 后处理: 修正 SuccessPattern 为组合级验证等级 ──
    #    form_from_jq 默认把成功因子标为 jq_single; 组合探索必须降级为 jq_composite
    #    且三个组合会 merge 成同一个 pattern (paradigm 相同) → 重建为独立组合模式
    templates = memory.data.setdefault("success_templates", [])
    removed = [t for t in templates if t.get("pattern_id") == "jq_confirmed::combo_exploration"]
    templates = [t for t in templates if t.get("pattern_id") != "jq_confirmed::combo_exploration"]
    memory.data["success_templates"] = templates
    print(f"\n[后处理] 移除默认合并组合模式 {len(removed)} 条 → 重建独立 jq_composite 模式")

    for combo in COMBOS:
        pat = {
            "pattern_id": f"combo::{combo['name']}",
            "description": (
                f"[JQ组合验证] {combo['name']}: {combo['members']} → "
                f"+{combo['jq_ret']:.2f}%/SR{combo['jq_sharpe']:.2f}/MDD{combo['jq_maxdd']:.2f}%"
            ),
            "typical_operators": [],
            "typical_windows": [20, 60],
            "applicable_regimes": [],
            "success_rate": 1.0,
            "ic_range": (0.02, 0.08),
            "icir_range": (0.3, 0.8),
            "sample_factor_ids": [combo["name"]],
            "lessons": [
                f"本地MMR模拟 {combo['local_ret']:.1f}%/SR{combo['local_sharpe']:.3f} "
                f"→ JQ {combo['jq_ret']:.1f}%/SR{combo['jq_sharpe']:.2f} "
                f"(折扣 {combo['local_ret'] / combo['jq_ret']:.1f}x, 量级不可信, 排序待验证)",
                f"跑输王者8完整版 (+{KING8['ret']:.2f}%/SR{KING8['sharpe']:.2f}): "
                f"现有因子池内重组无法突破王者",
            ],
            "formula": "",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "occurrence_count": 1,
            "jq_return": combo["jq_ret"],
            "jq_sharpe": combo["jq_sharpe"],
            "bridge_consumed": False,
            "verification_level": "jq_composite",  # 组合级验证, 非单因子
        }
        templates.append(pat)
    memory.data["success_templates"] = templates
    print(f"    已重建 {len(COMBOS)} 个 combo 模式 (verification_level=jq_composite)")

    # ── 3. 组合探索方法论知识沉淀 (结构化实验记录) ──
    combos_rec = []
    for combo in COMBOS:
        combos_rec.append({
            "name": combo["name"],
            "members": combo["members"],
            "desc": combo["desc"],
            "local_ret_pct": combo["local_ret"],
            "local_sharpe": combo["local_sharpe"],
            "jq_ret_pct": combo["jq_ret"],
            "jq_sharpe": combo["jq_sharpe"],
            "jq_maxdd_pct": combo["jq_maxdd"],
            "jq_discount_x": round(combo["local_ret"] / combo["jq_ret"], 2),
            "tested_at": datetime.now().isoformat(),
        })
    record = {
        "batch_id": "combo_exploration_unified_pool_20260818",
        "test_window": "2021-01-01~2026-07-30",
        "king8_jq": KING8,
        "combos": combos_rec,
        "methodology_findings": [
            # 排序验证: 本地 sim 收益排序 vs JQ 收益排序
            "MMR组合搜索排序验证: 本地 king6(456.9)>mmr6(258.9)>recon6(150.3) "
            "与 JQ king6(150.97)>mmr6(127.04)>recon6(118.63) 3/3 一致 "
            "→ 相关性结构MMR可作为组合预排序器(6因子同结构场景)",
            "本地收益量级系统性虚高且折扣非恒定 (king6 3.0x / mmr6 2.0x / recon6 1.3x) "
            "→ 本地只用于排序与相关性结构, 绝对收益一律以 JQ 为准",
            "统一池重组无法突破王者: 三组合全部跑输 King8 (+182.57%/0.69/-35.41%) "
            "→ 稀释定理 JQ 证实: 等权并入 SR<=0.51 的候选因子必稀释组合 SR; "
            "突破王者需 SR>=0.5 且与王者低相关的新因子, 而非池内重组",
            "king6 压缩版跑输 King8 完整版 31.6 个点 (SR 0.58 vs 0.69) "
            "→ 被裁剪的 mf20/ampvol 量能系因子有真实组合贡献, 王者结构不宜压缩",
            "本地 sim 对 King8(352%) vs king6(456.9%) 的相对判断与 JQ 相反 "
            "→ 成分数量不同的组合对比, 本地排序同样不可信; 排序验证结论仅限同结构(6因子)",
        ],
    }
    expl = memory.data.setdefault("combo_explorations", [])
    expl = [e for e in expl if e.get("batch_id") != record["batch_id"]]
    expl.append(record)
    memory.data["combo_explorations"] = expl

    # ── 4. 事件日志 + 保存 ──
    memory._log_event("combo_exploration", {
        "batch_id": record["batch_id"],
        "n_combos": len(COMBOS),
        "conclusion": "池内重组无法突破王者; MMR排序JQ验证3/3",
    })
    memory._save()

    print("\n" + "=" * 70)
    print("  D+ 闭环完成")
    print(f"  组合模式: {len(COMBOS)} 个 (jq_composite)")
    print(f"  方法论记录: combo_explorations[{record['batch_id']}]")
    print("  结论: 现有因子池内重组无法突破王者; 需新正交高SR因子")
    print("=" * 70)

    # 汇总对照表
    print("\n本地 vs JQ 对照:")
    print(f"  {'组合':<10} {'本地收益':>8} {'JQ收益':>8} {'JQ SR':>6} {'JQ MDD':>7} {'折扣':>5}")
    for c in combos_rec:
        print(f"  {c['name']:<10} {c['local_ret_pct']:>7.1f}% {c['jq_ret_pct']:>7.2f}% "
              f"{c['jq_sharpe']:>6.2f} {c['jq_maxdd_pct']:>6.2f}% {c['jq_discount_x']:>4.1f}x")
    print(f"  {'King8完整':<10} {'352.0%':>8} {KING8['ret']:>7.2f}% {KING8['sharpe']:>6.2f} {KING8['mdd']:>6.2f}%")


if __name__ == "__main__":
    main()
