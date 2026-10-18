"""
D+ 批量反馈: 2026-08-17 S5 通过 4 因子 JQ 回测结果注入 (用户手动聚宽回测)
- vshape_disposition_effect_v2 / oil_sensitive_cycle_lead_lag
- limited_attention_max_return_5d / herding_industry_deviation
窗口: 2021-01-01 ~ 2026-07-30 (285 周), 日志在 E:/log/<name>.txt
走完整闭环: 日志 IC 提取 → L1 归因 → RalphLoop.jq_feedback → 轨迹写回 → pool 回写
"""
import json
import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

# ── 用户在聚宽平台看到的最终指标 ──
JQ_RESULTS = {
    "vshape_disposition_effect_v2":    {"ret": 39.32,  "sharpe": 0.104,  "maxdd": 37.29},
    "oil_sensitive_cycle_lead_lag":    {"ret": 17.35,  "sharpe": -0.048, "maxdd": 31.36},
    "limited_attention_max_return_5d": {"ret": -14.86, "sharpe": -0.305, "maxdd": 41.02},
    "herding_industry_deviation":      {"ret": 53.68,  "sharpe": 0.199,  "maxdd": 29.73},
}

ROOT_CAUSES = {
    "vshape_disposition_effect_v2":
        "行为金融处置效应: 收益正但强度不足 (SR 0.104), 方向未证伪; 高价股等权下单残留 <100 股委托失败噪声",
    "oil_sensitive_cycle_lead_lag":
        "跨资产原油领先滞后代理: JQ 组合层基本无 alpha (SR 约 0), 周 IC 近零 (0.0026); local ICIR 0.42 → JQ 大幅折扣",
    "limited_attention_max_return_5d":
        "注意力限制 (5日最大收益反转): 组合负收益 -14.86%, 方向在组合层证伪 → 软负警告",
    "herding_industry_deviation":
        "行业偏离度反拥挤: 收益 53.68% 为正但 Sharpe 0.199 过低, 波动大; 行为金融方向最接近证实但未达 PASS",
}


def extract_ic_from_log(name: str):
    """从 E:/log/<name>.txt 提取每周 Rank IC 序列 (剔除 week=0 初始化行), 计算 mean/std/ICIR"""
    text = Path(f"E:/log/{name}.txt").read_text(encoding="utf-8", errors="replace")
    ics = [float(m) for m in re.findall(r"IC=(-?\d+\.\d+)", text)]
    # 剔除第一行 IC=0.0000 (week=0 初始化)
    ics = [ic for i, ic in enumerate(ics) if not (i == 0 and abs(ic) < 1e-9)]
    n = len(ics)
    mean = sum(ics) / n if n else 0.0
    std = statistics.stdev(ics) if n > 1 else 0.0
    icir = mean / std if std > 0 else 0.0
    half = n // 2
    m1 = sum(ics[:half]) / half if half else 0.0
    m2 = sum(ics[half:]) / (n - half) if n > half else 0.0
    return {"n_weeks": n, "jq_ic": mean, "jq_icir": icir,
            "first_half_ic": m1, "second_half_ic": m2}


def load_memory_meta():
    mem = json.loads((DATA_DIR / "experience_memory.json").read_text(encoding="utf-8"))
    meta = {}
    for a in mem.get("attempts", []):
        name = a.get("factor_name") or a.get("name")
        if name in JQ_RESULTS:
            meta[name] = a
    return meta


def build_payload():
    meta = load_memory_meta()
    factors = []
    for name, res in JQ_RESULTS.items():
        ic = extract_ic_from_log(name)
        a = meta.get(name, {})
        paradigm = a.get("paradigm", "未知")
        category = a.get("category", "unknown")
        formula = a.get("formula", "")
        local_icir = a.get("icir") or 0.0
        ret, sharpe, mdd = res["ret"], res["sharpe"], res["maxdd"]
        if ret < 0:
            contribution = "negative"
        elif ret > 50 and sharpe > 0.4:
            contribution = "positive"
        else:
            contribution = "neutral"
        factors.append({
            "factor_name": name,
            "formula": formula,
            "hypothesis": f"{name} ({paradigm}): " + (a.get("tags", [""])[0] if a.get("tags") else ""),
            "paradigm": paradigm,
            "category": category,
            "operators_used": [],
            "local_ic": 0.0,
            "local_icir": local_icir,
            "jq_return": ret,
            "jq_sharpe": sharpe,
            "jq_maxdd": mdd,
            "jq_ic": ic["jq_ic"],        # P-001: 日志提取的真实周频 Rank IC 均值
            "jq_icir": ic["jq_icir"],    # ICIR = mean/std
            "jq_weeks": ic["n_weeks"],
            "jq_composite_contribution": contribution,
            "root_cause": ROOT_CAUSES[name],
        })
    rets = [f["jq_return"] for f in factors]
    sharpes = [f["jq_sharpe"] for f in factors]
    maxdds = [f["jq_maxdd"] for f in factors]
    return {
        "batch_id": f"dplus_s5_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "composite_return": sum(rets) / len(rets),
        "composite_sharpe": sum(sharpes) / len(sharpes),
        "composite_maxdd": sum(maxdds) / len(maxdds),
        "factors": factors,
    }


def write_back_pool(factors):
    pool_path = DATA_DIR / "passed_factor_pool.csv"
    if not pool_path.exists():
        print("  [pool] 未找到 passed_factor_pool.csv, 跳过")
        return 0
    import pandas as pd
    pool = pd.read_csv(pool_path)
    if "status" not in pool.columns or "name" not in pool.columns:
        print("  [pool] 缺 name/status 列, 跳过")
        return 0
    names = {f["factor_name"] for f in factors}
    mask = pool["name"].astype(str).isin(names)
    n = int(mask.sum())
    if n:
        pool.loc[mask, "status"] = "jq_done"
        pool.to_csv(pool_path, index=False, encoding="utf-8-sig")
        print(f"  [pool] 状态回写 {n} 条 → jq_done: {sorted(pool.loc[mask,'name'].tolist())}")
    return n


def main():
    print("=" * 62)
    print("  D+ 批量反馈: S5 通过 4 因子 JQ 结果注入")
    print("=" * 62)
    payload = build_payload()
    print(f"\n  composite: {payload['composite_return']:.1f}% / Sharpe {payload['composite_sharpe']:.2f}")
    for f in payload["factors"]:
        print(f"    {f['factor_name']}: {f['jq_return']:+.2f}% / SR {f['jq_sharpe']:+.3f} / "
              f"MDD {f['jq_maxdd']:.1f}% | JQ IC {f['jq_ic']:+.4f} ICIR {f['jq_icir']:+.3f} "
              f"({f['jq_weeks']}周) | {f['paradigm']}")

    # ── L1 解读层 (与 trigger_d_plus 一致, 失败不阻塞) ──
    l1_result = None
    try:
        from interpretation_agent import interpret_batch
        from experience_memory import get_memory as _get_memory_l1
        _mem_l1 = _get_memory_l1()
        print(f"\n[L1] 解读层: 对 {len(payload['factors'])} 因子深度归因...")
        l1_result = interpret_batch(jq_backtest_result=payload, memory=_mem_l1, use_llm=True)
        for f in payload["factors"]:
            interp = f.get("interpretation") or {}
            print(f"    {f['factor_name']} → {interp.get('verdict', '?')}")
    except Exception as e:
        print(f"  [L1] ⚠️ 解读层失败 (不阻塞): {e}")

    # ── D+ 蒸馏 ──
    from ralph_loop import RalphLoop
    from experience_memory import get_memory
    memory = get_memory()
    rl = RalphLoop(memory=memory)
    result = rl.jq_feedback(jq_backtest_result=payload)

    # ── L1 写回 ──
    if l1_result:
        try:
            from interpretation_agent import attach_to_memory
            n_mem = attach_to_memory(memory, payload["factors"])
            print(f"  [L1] 写回 memory: {n_mem} 条")
        except Exception as e:
            print(f"  [L1] ⚠️ 写回失败: {e}")

    # ── trajectory_log 标记 d_plus_applied (本次 4 因子) ──
    names = {f["factor_name"] for f in payload["factors"]}
    traj_path = DATA_DIR / "trajectory_log.json"
    traj_data = json.loads(traj_path.read_text(encoding="utf-8"))
    marked = 0
    for t in traj_data.get("trajectories", []):
        sf = str(t.get("seed_factor", ""))
        if any(n in sf for n in names) and t.get("jq_validated") and not t.get("d_plus_applied"):
            t["d_plus_applied"] = True
            t["d_plus_at"] = datetime.now().isoformat()[:19]
            marked += 1
    traj_data["updated_at"] = datetime.now().isoformat()
    traj_path.write_text(json.dumps(traj_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [traj] 标记 d_plus_applied: {marked} 条")

    # ── pool 回写 ──
    n_pool = write_back_pool(payload["factors"])

    print(f"\n{'=' * 62}")
    print(f"  D+ 完成: 轨迹 {result.get('trajectories_updated')} | "
          f"硬禁止 +{result.get('hard_forbidden_added')} | 软警告 +{result.get('soft_warnings_added')} | "
          f"确认成功 +{result.get('jq_success_confirmed')} | pool {n_pool}")
    print(f"{'=' * 62}")
    return result


if __name__ == "__main__":
    main()
