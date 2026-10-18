# -*- coding: utf-8 -*-
"""
backfill_trials.py — 回填历史 JQ 试验 → trial_log.jsonl, 训练首版 scorer

数据来源: .workbuddy/memory/MEMORY.md 策略榜单 (用户聚宽实测结果)
只收录文件↔结果映射确定的 AlphaAgent v3 时代试验。
日频策略 (-42.44%) 无留存文件, 无法提取特征, 记录但不入训练集。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.factor_alchemy.evaluator.features import (
    parse_jq_strategy, build_strategy_features, FEATURE_NAMES,
)
from research.factor_alchemy.evaluator.scorer import RidgeUCB

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "research" / "factor_alchemy" / "output"
INJECTED_JSON = ROOT / "data" / "injected_factors.json"
TRIAL_LOG = Path(__file__).parent / "trial_log.jsonl"
SCORER_STATE = Path(__file__).parent / "scorer_state.json"

# (文件时间戳, JQ累计收益%, Sharpe, MaxDD%, 回测录入日期, 映射确定?)
TRIALS = [
    ("20260802_130337", 182.57, 0.69, -35.41, "2026-08-02", True),   # 王者
    ("20260802_230931", 180.91, 0.66, -40.36, "2026-08-02", True),   # 进化#1初版
    ("20260802_231040", 130.50, 0.46, -42.03, "2026-08-02", True),   # 进化#1二版(LLM异常)
    ("20260803_183550", 180.91, 0.66, -40.36, "2026-08-03", True),   # 进化#2
    ("20260804_001621", 141.19, 0.50, -43.65, "2026-08-04", True),   # 进化#3(MMR)
    ("20260801_164126", 133.90, 0.53, -32.64, "2026-08-01", False),  # 周频基础版5复合(文件映射不完全确定)
    ("20260801_230253", 136.85, 0.53, -37.51, "2026-08-01", False),  # 月频(三个文件之一)
]


def main():
    records = []
    for ts, ret, sr, mdd, date, certain in TRIALS:
        path = OUTPUT_DIR / f"fa_alpha_agent_v3_jq_{ts}.py"
        if not path.exists():
            print(f"⚠️ 文件缺失: {path.name}, 跳过")
            continue
        parsed = parse_jq_strategy(path)
        vec, meta = build_strategy_features(parsed, INJECTED_JSON)
        records.append({
            "strategy_file": path.name,
            "date": date,
            "total_return_pct": ret,
            "sharpe": sr,
            "max_drawdown_pct": mdd,
            "features": vec,
            "meta": meta,
            "mapping_certain": certain,
            "logged_at": datetime.now().isoformat(),
        })
        flag = "" if certain else " (映射不确定)"
        print(f"✓ {path.name}: freq={meta['frequency']} comps={meta['n_composites']} "
              f"inj={meta['n_injected']} llm={meta['n_new_llm']} → {ret}%{flag}")

    # 写 trial_log.jsonl (覆盖式回填, 之后影子运行用追加)
    with open(TRIAL_LOG, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n→ {TRIAL_LOG.name}: {len(records)} 条试验记录")

    # 只用映射确定的样本训练
    certain = [r for r in records if r["mapping_certain"]]
    X = np.array([r["features"] for r in certain])
    y = np.array([r["total_return_pct"] for r in certain])
    dates = [r["date"] for r in certain]

    scorer = RidgeUCB(n_features=len(FEATURE_NAMES), lam=10.0, beta=1.0)
    scorer.feature_names = FEATURE_NAMES
    scorer.fit(X, y, dates)
    scorer.save(SCORER_STATE)

    # ── 报告 ──
    print(f"\n{'='*64}")
    print(f"首版 Scorer 训练完成 (n={len(certain)}, 特征={len(FEATURE_NAMES)}, λ=10)")
    print(f"{'='*64}")
    print(f"y_mean={scorer.y_mean:.1f}%  残差σ_e={scorer.sigma_e:.1f}pp")

    print(f"\n权重表 (标准化空间, 按|w|排序):")
    for name, w in scorer.weight_table():
        bar = "█" * int(min(abs(w) * 2, 20))
        print(f"  {name:<24} {w:+7.2f}  {bar}")

    print(f"\n校准 (留一法意义有限, n=5 仅供参考 — 预测 vs 实际):")
    for r in certain:
        mean, sigma = scorer.predict(r["features"])
        actual = r["total_return_pct"]
        err = mean - actual
        print(f"  {r['strategy_file'][-10:-3]}: 预测={mean:6.1f}±{sigma:5.1f}  "
              f"实际={actual:6.1f}  误差={err:+6.1f}pp")

    # 诚实提示: evo#1a vs evo#1b 同特征不同结果 → 特征无法解释的方差
    print(f"\n⚠️ 诚实警告: evo#1初版(+180.9)与#1二版(+130.5)特征完全相同,")
    print(f"   结果差50pp — 当前特征无法捕获 LLM 公式质量这个主导变量。")
    print(f"   这是预期内的: scorer 的真正价值在影子运行产生复合级数据后显现。")


if __name__ == "__main__":
    main()
