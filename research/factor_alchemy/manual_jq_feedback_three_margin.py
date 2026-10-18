# -*- coding: utf-8 -*-
"""
手动 D+ 蒸馏: 三因子 JQ 研究环境纸质回测结果注入
(margin_short_ratio / margin_balance_chg_21d / margin_mf_divergence)
回测方式: 研究环境真实 JQ 数据 (get_mtss/get_money_flow/get_price) 手动复刻引擎周频调仓
          (回测引擎无两融/资金流 API, 纸质回测为替代验证路径)
窗口: 2018-01-01 ~ 2026-08-13, 中证500起点成分池 500 股, TOP50 周频, 成本 0.72%×换手
用法: cd research/factor_alchemy && python manual_jq_feedback_three_margin.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
DATA_DIR = Path(__file__).parent.parent.parent / "data"

JQ_RESULT = {
    "batch_id": "jq_paper_backtest_margin_3f_20260815",
    "timestamp": datetime.now().isoformat(),
    "test_method": "research_env_paper_backtest",
    "test_window": "2018-01-01~2026-08-13",
    "composite_return": -7.23,
    "composite_sharpe": 0.08,
    "composite_maxdd": -56.16,
    "factors": [
        {
            "factor_name": "margin_short_ratio",
            "formula": "fin_value / (max(fin_sec_value - fin_value, 0) + 1e-6)",
            "hypothesis": (
                "两融多空比: 融资余额/融券余额。融资占比高的股票=散户杠杆情绪高, "
                "未来收益弱于融券占比高的股票。本地ICIR=+0.235 (2015-2026)。"
            ),
            "paradigm": "资金流",
            "category": "margin",
            "operators_used": ["div", "clip"],
            "local_ic": 0.0,
            "local_icir": 0.235,
            "jq_return": 15.29,
            "jq_sharpe": 0.20,
            "jq_maxdd": -43.10,
            "jq_ic": 0.0110,
            "jq_icir": 0.0878,
            "jq_composite_contribution": "positive_weak",
            "root_cause": (
                "研究环境纸质回测 2018-2026 全窗: 累计+15.29% 跑输基准中证500(+21.53%), "
                "Sharpe 0.20, MaxDD -43%。方向(融资占比高=弱)获IC确认(ICIR+0.088, 55.3%正) "
                "但强度远低于本地(+0.235), 组合层面不达标 → FAIL。"
            ),
            "jq_window": "2018-01-01~2026-08-13 (研究环境纸质回测)",
        },
        {
            "factor_name": "margin_balance_chg_21d",
            "formula": "-(fin_value - fin_value.shift(21))",
            "hypothesis": (
                "融资余额21日变化取负: 杠杆资金追高后回落。本地月频ICIR=-0.659 (2018-2024)。"
            ),
            "paradigm": "资金流",
            "category": "margin",
            "operators_used": ["diff", "negation"],
            "local_ic": 0.0,
            "local_icir": -0.659,
            "jq_return": -21.42,
            "jq_sharpe": 0.01,
            "jq_maxdd": -56.16,
            "jq_ic": 0.0318,
            "jq_icir": 0.3350,
            "jq_composite_contribution": "negative",
            "root_cause": (
                "BROKEN 型 (复现 breed 008 教训: IC正≠组合正): JQ日频 ICIR=+0.335 "
                "(61.4%周IC为正, 方向获确认) 但组合累计-21.42%, 大幅跑输基准43个点。"
                "IC 是全体横截面信息, TOP50 尾部选择+36%平均换手×0.72%成本侵蚀 "
                "使信号无法转成组合收益 → FAIL。"
            ),
            "jq_window": "2018-01-01~2026-08-13 (研究环境纸质回测)",
        },
        {
            "factor_name": "margin_mf_divergence",
            "formula": "-rolling60_corr(net_amount_main, fin_value)",
            "hypothesis": (
                "主力资金流与融资余额60日相关性取负: 两者背离时定价分歧, 未来收益弱。"
                "本地弱正信号。"
            ),
            "paradigm": "资金流",
            "category": "margin_money_flow",
            "operators_used": ["rolling", "corr", "negation"],
            "local_ic": 0.0,
            "local_icir": 0.1,
            "jq_return": -15.57,
            "jq_sharpe": 0.04,
            "jq_maxdd": -50.28,
            "jq_ic": None,
            "jq_icir": None,
            "jq_composite_contribution": "negative",
            "root_cause": (
                "组合累计-15.57% 大幅跑输基准; 且因子值高度趋同(A股主力资金流与融资余额"
                "共线性强, corr 同值化), 87% 周的 spearmanr 常数输入返回 nan "
                "(IC>0占比仅12.7%), 因子缺横截面区分度 → FAIL。"
            ),
            "jq_window": "2018-01-01~2026-08-13 (研究环境纸质回测)",
        },
    ],
}


def main():
    # ── L1 解读层 (先于 distill) ──
    try:
        from interpretation_agent import interpret_batch
        from experience_memory import get_memory as _get_mem_l1
        mem_l1 = _get_mem_l1()
        print("[L1] 解读层启动 (LLM 归因, 失败自动降级规则)...")
        interpret_batch(JQ_RESULT, memory=mem_l1, use_llm=True)
        for f in JQ_RESULT["factors"]:
            itp = f.get("interpretation") or {}
            print("  [L1] %s -> verdict=%s confidence=%s"
                  % (f["factor_name"], itp.get("verdict", "?"), itp.get("confidence", "?")))
    except Exception as e:
        print("[L1] 解读层失败 (不阻塞 D+): %r" % e)

    # ── D+ 闭环 ──
    from experience_memory import get_memory
    from ralph_loop import RalphLoop
    memory = get_memory()
    rl = RalphLoop(memory=memory)
    result = rl.jq_feedback(jq_backtest_result=JQ_RESULT)
    print("\n  D+ 结果:", json.dumps(result, ensure_ascii=False, indent=2)[:3000])

    # ── L1 写回 memory ──
    try:
        from interpretation_agent import attach_to_memory
        n = attach_to_memory(memory, JQ_RESULT["factors"])
        print("  [L1] attach_to_memory: %d 条" % n)
    except Exception as e:
        print("  [L1] attach 失败: %r" % e)

    # ── 更新因子队列 ──
    qpath = DATA_DIR / "stage1_factor_proposals.json"
    q = json.load(open(qpath, encoding="utf-8"))
    items = q if isinstance(q, list) else q.get("proposals", q.get("factors", []))
    for it in items:
        name = str(it.get("factor_name") or "")
        base = name.split("_v2")[0]
        if base in ("margin_short_ratio", "margin_balance_chg_21d", "margin_mf_divergence"):
            jqr = {
                "method": "research_env_paper_backtest",
                "window": "2018-01-01~2026-08-13",
                "return_pct": {
                    "margin_short_ratio": 15.29,
                    "margin_balance_chg_21d": -21.42,
                    "margin_mf_divergence": -15.57,
                }[base],
                "sharpe": {
                    "margin_short_ratio": 0.20,
                    "margin_balance_chg_21d": 0.01,
                    "margin_mf_divergence": 0.04,
                }[base],
                "outcome": "JQ_FAIL",
                "note": "纸质回测跑输基准, 不进生产候选",
                "tested_at": datetime.now().isoformat(),
            }
            it["jq_status"] = "jq_tested"
            it["jq_result"] = jqr
    json.dump(q, open(qpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("  [Queue] 3 个 margin 因子已标记 jq_tested / JQ_FAIL")


if __name__ == "__main__":
    main()
