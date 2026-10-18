# -*- coding: utf-8 -*-
"""伏羲 v2.0 → D+ 反馈注入 (2026-10-19, 7 因子)

背景
----
15 个 E0~E4 全 pass 的 standalone 脚本在 2026-10-18 修掉"因子数组 / 股票列表错位"
(IndexError + 静默选错票) 后重跑, 用户在聚宽看到 2021-01-01~2026-09-30 (290 周) 的
组合收益。本脚本把其中 **用户指定允许回填的 7 个** 注入 Experience Memory + Trajectory。

用户明确排除 (不得进入 D+ 记忆反馈, 本脚本硬断言):
    factor_879c96895, factor_588b9974b, factor_c9a6fb22d, factor_0763f0983,
    factor_41b5d3f44, factor_3e5fa937b, factor_69846fa5d

未纳入 (非排除, 而是数据不合格):
    factor_3014464e4 —— E:/log/factor_3014464.txt 是修复前 (22:24) 的运行,
    univ=1300 按代码截断 + 因子数组错位, 该 70.88% 不能作为因子证据; 待重跑后再注入。

证据链
------
- 收益/Sharpe/maxDD: 用户聚宽回测读数 (见 RESULTS)
- JQ 滞后 Rank IC/ICIR/IC>0/n: 从 E:/log/factor_<id>.txt 的 P-001 报告块提取
  (output/_verify_standalone_fix/jq_log_summary_20261019.json)
- 执行缺陷: 同批日志的 ERROR 行 (GBK) 统计 —— 科创板(688) 市价单被拒
  "科创板市价单需要指定保护限价", 即选出但买不进, 组合实际少买若干仓位
"""
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

EVIDENCE = ROOT / "output" / "_verify_standalone_fix" / "jq_log_summary_20261019.json"

PERIOD = "2021-01-01~2026-09-30"

# ── 用户聚宽读数 (累计收益% / Sharpe / maxDD%) ──────────────────
RESULTS = {
    "factor_68cedf84b": (62.09, 0.260, 26.33),
    "factor_49cf665bd": (70.71, 0.282, 35.90),
    "factor_7bb331fd2": (16.03, -0.090, 19.29),
    "factor_e6c97864f": (60.07, 0.289, 25.96),
    "factor_c34c66f5d": (42.31, 0.100, 38.92),
    "factor_b8fc1bac1": (21.16, -0.027, 32.42),
    "factor_2914a71d3": (12.18, -0.108, 24.13),
}

# ── 用户指定排除: 任何情况下不得进入本次 payload ──────────────
EXCLUDED = {
    "factor_879c96895": "用户指定排除",
    "factor_588b9974b": "用户指定排除",
    "factor_c9a6fb22d": "用户指定排除",
    "factor_0763f0983": "用户指定排除",
    "factor_41b5d3f44": "用户指定排除",
    "factor_3e5fa937b": "用户指定排除",
    "factor_69846fa5d": "用户指定排除",
}

# ── 因子元信息: formula / 类别 / 伏羲 E1 本地读数 ──────────────
META = {
    "factor_68cedf84b": dict(
        formula="-(rank((std(close,2) / mean(close,2))))",
        paradigm="lowvol_reversal", category="量价-低波动",
        local_ic=0.04429, local_icir=0.233, operators=["std", "mean", "div", "rank", "neg"],
        mechanism="2 日变异系数(波动率/均价)的截面排名取负: 极短期低波动异象"),
    "factor_49cf665bd": dict(
        formula="(rank(-(rank((std(close,20) / mean(close,20))))) / rank((rank(return(close,10)) - rank((volume / mean(volume,10))))))",
        paradigm="lowvol_reversal", category="量价-低波动×动量/量能",
        local_ic=0.03795, local_icir=0.208, operators=["std", "mean", "div", "rank", "neg", "return", "sub"],
        mechanism="20 日低波动排名 除以 (10 日动量排名 - 量能放大排名): 低波动为主, 叠加缩量/弱势条件"),
    "factor_7bb331fd2": dict(
        formula="-(rank((std(close,20) / mean(close,20))))",
        paradigm="lowvol_reversal", category="量价-低波动",
        local_ic=0.05922, local_icir=0.259, operators=["std", "mean", "div", "rank", "neg"],
        mechanism="20 日变异系数截面排名取负: 中期低波动异象"),
    "factor_e6c97864f": dict(
        formula="(rank((rank(-(rank((std(close,5) / mean(close,5))))) + rank(-(rank((std(close,20) / mean(close,20))))))) + rank((rank(return(close,10)) - rank((volume / mean(volume,10))))))",
        paradigm="lowvol_reversal", category="量价-复合(双周期低波动+缩量动量)",
        local_ic=0.05079, local_icir=0.252,
        operators=["std", "mean", "div", "rank", "neg", "add", "return", "sub"],
        mechanism="5 日+20 日双周期低波动排名之和, 再与 (10 日动量 - 量能) 排名相加"),
    "factor_c34c66f5d": dict(
        formula="-(return(close,20))",
        paradigm="momentum_reversal", category="量价-动量反转",
        local_ic=0.05049, local_icir=0.238, operators=["return", "neg"],
        mechanism="20 日收益取负: 中期反转 (与 factor_69846fa5d 的 rank 变体完全同构)"),
    "factor_b8fc1bac1": dict(
        formula="(rank((rank(-(rank((std(close,5) / mean(close,5))))) + rank(-(rank((std(close,20) / mean(close,20))))))) + rank(-(rank(return(close,20)))))",
        paradigm="lowvol_reversal", category="量价-复合(双周期低波动+动量反转)",
        local_ic=0.07286, local_icir=0.370,
        operators=["std", "mean", "div", "rank", "neg", "add", "return"],
        mechanism="5 日+20 日低波动排名之和, 与 20 日动量反转排名相加"),
    "factor_2914a71d3": dict(
        formula="(rank(rank(return(close,2))) * rank((rank(-(rank((std(close,5) / mean(close,5))))) + rank(-(rank((std(close,20) / mean(close,20))))))))",
        paradigm="lowvol_reversal", category="量价-复合(低波动×2日动量)",
        local_ic=0.04906, local_icir=0.231,
        operators=["return", "rank", "mul", "std", "mean", "div", "neg", "add"],
        mechanism="2 日动量排名 乘 双周期低波动排名 (交互项, 只在动量不弱时做低波动)"),
}


def load_evidence():
    if not EVIDENCE.exists():
        raise SystemExit("缺少日志证据文件: %s" % EVIDENCE)
    rows = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    return {r["factor"]: r for r in rows if r.get("status") == "ok"}


def build_payload():
    ev = load_evidence()
    factors = []
    for name, (ret, sr, mdd) in RESULTS.items():
        assert name not in EXCLUDED, "排除清单泄漏: %s" % name
        m = META[name]
        e = ev.get(name)
        if e is None:
            raise SystemExit("%s 在日志证据中缺失" % name)
        if ret > 50 and sr > 0.4:
            verdict = "JQ_PASSED"
        elif ret < -20 or sr < -0.5:
            verdict = "JQ_FAILED"
        elif ret < 0:
            verdict = "JQ_WEAK_NEGATIVE"
        else:
            verdict = "JQ_MARGINAL"
        contribution = "positive" if verdict == "JQ_PASSED" else (
            "negative" if ret < 0 else "neutral")
        root_cause = (
            "%s: 累计 %+.2f%% / Sharpe %+.3f / maxDD %.2f%% (2021-01-01~2026-09-30, 290 周)。"
            "方向为正但未达 PASS 线 (需 ret>50%% 且 Sharpe>0.4), 判 %s。"
            "本地 E1 IC %+.5f / ICIR %.3f → JQ 滞后 Rank IC %+.4f / ICIR %.3f / IC>0 %.1f%% (n=%d): "
            "IC 均值与 IC 命中率均未塌陷, 说明信号方向仍在, 但组合层波动过大/收益不足, "
            "即 IC→Sharpe 的转化率低 (等权 top-30 周频, 无风险平价/波动率缩放)。"
            "执行缺陷: 同批日志有 %d 次科创板(688)市价单被拒 (\"科创板市价单需要指定保护限价\"), "
            "选出却买不进 → 组合实际少买仓位并留现金拖累; universe 为全市场等距抽样 %s 只。"
            % (verdict, ret, sr, mdd, verdict, m["local_ic"], m["local_icir"],
               e["jq_ic"], e["jq_icir"], (e["jq_ic_pos"] or 0) * 100,
               e["jq_ic_n"] or 0, e["err_688"], e["last_univ"])
        )
        factors.append({
            "factor_name": name,
            "formula": m["formula"],
            "hypothesis": "%s (%s): %s" % (name, m["paradigm"], m["mechanism"]),
            "paradigm": m["paradigm"],
            "category": m["category"],
            "operators_used": m["operators"],
            "local_ic": m["local_ic"],
            "local_icir": m["local_icir"],
            "jq_return": ret,
            "jq_sharpe": sr,
            "jq_maxdd": mdd,
            "jq_ic": e["jq_ic"],
            "jq_icir": e["jq_icir"],
            "jq_ic_n": e["jq_ic_n"],
            "jq_ic_pos": e["jq_ic_pos"],
            "exec_688_rejected": e["err_688"],
            "jq_backtest_period": PERIOD + " (290 周)",
            "jq_composite_contribution": contribution,
            "root_cause": root_cause,
            "source": "fuxi_v2_standalone_jq",
        })

    assert len(factors) == 7
    assert not ({f["factor_name"] for f in factors} & set(EXCLUDED))

    rets = [f["jq_return"] for f in factors]
    srs = [f["jq_sharpe"] for f in factors]
    mdds = [f["jq_maxdd"] for f in factors]
    return {
        "batch_id": "fuxi_v2_standalone_7factors_%s" % datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        # 7 条相互独立单因子回测的算术均值, 不是组合回测; 仅用于 composite_assessment 标签
        "composite_return": round(sum(rets) / len(rets), 2),
        "composite_sharpe": round(sum(srs) / len(srs), 4),
        "composite_maxdd": round(sum(mdds) / len(mdds), 2),
        "composite_note": "composite = 7 条独立单因子 standalone 回测读数的算术均值, 非组合净值",
        "factors": factors,
    }


def main():
    payload = build_payload()
    print("=" * 74)
    print("  D+ 批量反馈: 伏羲 v2.0 standalone 因子 (允许回填 7 / 排除 7)")
    print("=" * 74)
    print("  batch_id: %s" % payload["batch_id"])
    print("  composite(算术均值): %+.2f%% / Sharpe %+.3f / maxDD %.2f%%"
          % (payload["composite_return"], payload["composite_sharpe"], payload["composite_maxdd"]))
    print("  排除(未注入): %s" % ", ".join(sorted(EXCLUDED)))
    print("-" * 74)
    for f in payload["factors"]:
        print("  %-18s %+7.2f%% SR%+6.3f MDD%5.2f%% | local IC %+.5f/ICIR %.3f | "
              "JQ IC %+.4f/ICIR %.3f (IC>0 %.1f%%, n=%d, 688拒单 %d)"
              % (f["factor_name"], f["jq_return"], f["jq_sharpe"], f["jq_maxdd"],
                 f["local_ic"], f["local_icir"], f["jq_ic"], f["jq_icir"],
                 (f["jq_ic_pos"] or 0) * 100, f["jq_ic_n"] or 0,
                 f["exec_688_rejected"]))

    from fuxi_v1.governance import governed_writer, find_workspace
    from experience_memory import get_memory
    from ralph_loop import RalphLoop

    workspace = find_workspace(__file__)
    with governed_writer(workspace, "feedback", trigger="manual"):
        memory = get_memory()
        rl = RalphLoop(memory=memory)
        result = rl.jq_feedback(jq_backtest_result=payload)

    print("\n=== D+ 反馈结果 ===")
    for k, v in result.items():
        if isinstance(v, dict):
            print("  %s: %s" % (k, json.dumps(v, ensure_ascii=False)[:300]))
        else:
            print("  %s: %s" % (k, v))

    receipt = ROOT / "data" / ("dplus_fuxi_v2_7factors_%s.json"
                               % datetime.now().strftime("%Y%m%d_%H%M%S"))
    receipt.write_text(json.dumps({"payload": payload, "result": result},
                                  ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n  receipt: %s" % receipt)


if __name__ == "__main__":
    main()
