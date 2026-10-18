"""
D+ 蒸馏自动触发器 — 检测未处理的 JQ 回测结果并自动蒸馏

用法:
  python trigger_d_plus.py          # 一键检测+蒸馏
  python trigger_d_plus.py --dry     # 仅检测, 不蒸馏
  python trigger_d_plus.py --force   # 强制重蒸馏所有 JQ 已验证条目
  python trigger_d_plus.py --backfill            # 历史欠账回填 (dry, 只统计)
  python trigger_d_plus.py --backfill --apply    # 历史欠账回填 (写回)

架构:
  1. 扫描 trajectory_log.json 中 jq_validated=true 且 d_plus_applied!=true 的条目
  2. 按 breed 分组, 自动构造 jq_backtest_result 字典
  3. 调用 RalphLoop.jq_feedback() 执行 D+ 蒸馏
  4. 兜底核对标记 (jq_feedback 自 P-20260918-021 起已就地打标)
  5. 输出蒸馏报告

集成点: 可在 run_v4_pipeline.py 启动时或自动化 Stage 4 中调用

D+ 欠账根因 (2026-09-18, P-20260918-021):
  TrajectoryLogger._save() → MiningTrajectory.to_dict() 是 22 键**白名单重建**,
  d_plus_applied / jq_* / lessons 等非白名单键每轮流水线被静默抹除 (8 个 save 点)。
  ⇒ 欠账 235 条 ≈ 100% 历史 jq_validated, 是**结构性伪影**而非待办队列深度。
  修复: 9+2 个富字段提升为 dataclass 一级字段 (三处同步) + jq_feedback 就地打标。
  🔴 欠账回填只补字段, **不得**用重跑 trigger_d_plus 代替 (会灌水 Memory 计数器)。
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

# e:\quant\data\ 是实际数据目录 (workspace 根下的 data/)
DATA_DIR = Path(__file__).parent.parent.parent / "data"
TRAJ_LOG = DATA_DIR / "trajectory_log.json"
JQ_FEEDBACK = DATA_DIR / "jq_feedback_summary.json"

# L1 解读层开关 (2026-08-14): JQ 蒸馏前对每个因子做 LLM 深度归因
# 归因结果挂 factor["interpretation"], 供 distill_motif 两级蒸馏 + MAB 精细化消费
ENABLE_L1 = True
L1_USE_LLM = True  # False 时走纯规则降级 (测试模式)


def load_trajectories():
    """加载轨迹日志"""
    with open(TRAJ_LOG, "r", encoding="utf-8") as f:
        return json.load(f)


def save_trajectories(data: Dict):
    """保存轨迹日志"""
    data["updated_at"] = datetime.now().isoformat()
    with open(TRAJ_LOG, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_pending_jq_entries(data: Dict) -> List[Dict]:
    """找到所有等待 D+ 蒸馏的 JQ 条目"""
    trajs = data.get("trajectories", [])
    pending = []
    for t in trajs:
        if t.get("jq_validated") and not t.get("d_plus_applied"):
            pending.append(t)
    return pending


def group_by_breed(pending: List[Dict]) -> Dict[str, List[Dict]]:
    """按因子名称分组, 去重 (相同公式多个轨迹只取代表性一条)"""
    groups = defaultdict(list)
    for t in pending:
        # 提取简洁的 breed 名称
        expr = t.get("expression", {})
        content = expr.get("content", "") if isinstance(expr, dict) else str(expr)

        # 用公式前80字符做去重键
        key = content[:80].strip()
        if key:
            groups[key].append(t)

    return groups


def construct_jq_feedback(
    groups: Dict[str, List[Dict]], force: bool = False
) -> Optional[Dict]:
    """从轨迹分组构造 jq_backtest_result 字典"""
    factors = []
    batch_name = f"dplus_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 记录各因子的范式/类别/operators 使用的映射
    # 从 formula 推断 operators (简化版)
    def infer_operators(formula: str) -> List[str]:
        ops = []
        keywords = {
            "close": "close", "open": "open", "high": "high", "low": "low",
            "volume": "volume", "amount": "amount", "turnover": "turnover",
            "pct_change": "pct_change", "shift": "shift",
            "rolling_mean": "rolling_mean", "rolling_std": "rolling_std",
            "rolling_min": "rolling_min", "rolling_max": "rolling_max",
            "rank": "rank", "ts_rank": "ts_rank",
            "ts_delta": "ts_delta", "ts_corr": "ts_corr",
            "neg": "negation", "/": "division", "*": "multiplication",
            "+": "addition", "-": "subtraction",
            "astype": "astype", "hl_ratio": "hl_ratio",
        }
        for kw, op_name in keywords.items():
            if kw in formula.lower():
                ops.append(op_name)
        return ops

    for key, entries in groups.items():
        t = entries[0]  # 取第一条为代表
        name = t.get("factor_name", t.get("trajectory_id", "unknown"))
        paradigm = t.get("paradigm", "未知")

        # 推断 category (如果 trajectory 里没有)
        paradigm_to_category = {
            "均值回复": "mean_reversion",
            "价量关系": "price_volume",
            "动量": "momentum",
            "动量反转": "momentum_reversal",
            "波动": "volatility",
            "活跃资金流": "money_flow",
            "微观结构": "microstructure",
            "资金流": "money_flow",
            "筹码分布": "position_analysis",
            "情绪": "sentiment",
        }
        category = paradigm_to_category.get(paradigm, "unknown")

        # 提取公式
        expr = t.get("expression", {})
        content = expr.get("content", "") if isinstance(expr, dict) else str(expr)

        jq_return = t.get("jq_return", 0)
        jq_sharpe = t.get("jq_sharpe", 0)
        jq_maxdd = t.get("jq_maxdd", 0)
        jq_rating = t.get("jq_rating", "UNKNOWN")

        # 确定贡献方向
        # 修复 (2026-08-29): jq_rating 为 legacy 字段 (轨迹全 None), 贡献方向改由
        # final_outcome 推导, 否则 JQ_PASSED 因子在 D+ 归因/校准集里被降级为 neutral
        _outcome = str(t.get("final_outcome") or "").strip().upper()
        if jq_rating in ("BEST", "PASS"):
            contribution = "positive"
        elif jq_rating in ("FAIL", "BROKEN", "DUPLICATE"):
            contribution = "negative"
        elif _outcome in ("JQ_PASSED", "JQ_BEST"):
            contribution = "positive"
        elif _outcome in ("JQ_FAILED", "JQ_FAIL", "JQ_BROKEN", "JQ_WEAK_NEGATIVE"):
            contribution = "negative"
        else:
            contribution = "neutral"

        # 获取 root_cause
        root_cause = t.get("jq_notes", "")

        # 获取 local_ic / local_icir (如果有)
        local_ic = t.get("local_ic") or t.get("final_ic") or 0.0
        local_icir = t.get("local_icir") or t.get("final_icir") or 0.0

        # P-001 (2026-08-29): 透传真实 JQ per-factor IC/ICIR (日志 _compute_per_factor_ic 提取),
        # jq_feedback 优先用真实 IC 而非 jq_ret/100 收益代理
        jq_ic = t.get("jq_ic")
        jq_icir = t.get("jq_icir")

        factor = {
            "factor_name": name,
            "formula": content,
            "hypothesis": t.get("hypothesis", "") or name,
            "paradigm": paradigm,
            "category": category,
            "operators_used": infer_operators(content) if content else [],
            "local_ic": local_ic,
            "local_icir": local_icir,
            "jq_return": jq_return,
            "jq_sharpe": jq_sharpe,
            "jq_maxdd": jq_maxdd,
            "jq_ic": jq_ic,
            "jq_icir": jq_icir,
            # v0.7 频率对称: natural_freq 透传 (motif 知识带频率维度, 同 motif 日/周频表现可能相反)
            "natural_freq": t.get("natural_freq", "daily"),
            "jq_composite_contribution": contribution,
            "root_cause": root_cause,
        }
        factors.append(factor)

    if not factors:
        return None

    # 计算复合指标
    returns = [f["jq_return"] for f in factors]
    sharpes = [f["jq_sharpe"] for f in factors]
    maxdds = [f["jq_maxdd"] for f in factors]

    return {
        "batch_id": batch_name,
        "timestamp": datetime.now().isoformat(),
        "composite_return": sum(returns) / len(returns) if returns else 0,
        "composite_sharpe": sum(sharpes) / len(sharpes) if sharpes else 0,
        "composite_maxdd": sum(maxdds) / len(maxdds) if maxdds else 0,
        "factors": factors,
    }


def _summarize_interpretations(factors: List[Dict]) -> str:
    """L1 归因一行概览"""
    parts = []
    for f in factors:
        interp = f.get("interpretation") or {}
        v = interp.get("verdict", "?")
        parts.append(f"{f.get('factor_name', '?')}={v}")
    return ", ".join(parts)


def _attach_to_trajectories(data: Dict, factors: List[Dict]) -> int:
    """把 interpretation 写回 trajectory_log 条目 (按 seed_factor 名匹配)"""
    n = 0
    trajs = data.get("trajectories", [])
    for f in factors:
        name = f.get("factor_name", "")
        interp = f.get("interpretation") or {}
        if not name or not interp:
            continue
        for t in trajs:
            if name in str(t.get("seed_factor", "")):
                t["interpretation"] = interp
                n += 1
                break
    return n


def mark_d_plus_applied(data: Dict, pending: List[Dict]):
    """幂等兜底标记 (2026-09-18 语义更新)

    自 P-20260918-021 起, ``jq_feedback()`` 内部已就地打 d_plus_applied 标记
    (source=jq_feedback)。本函数保留为**兜底**: 只对仍未标记的条目补标,
    并区分报告「已被 jq_feedback 标记」与「本次兜底补标」两种来源。

    ⚠️ 此前该函数的输出 "标记完成: N 条" 曾被误读为 D+ 欠账深度; 实际它
    每次都会把**全部** jq_validated 条目重新标记一遍, 且标记本身存不住
    (schema 缺列) — 二者叠加造成 235 条恒欠账的假象。
    """
    now = datetime.now().isoformat()
    already = 0
    backstop = 0
    for t in data.get("trajectories", []):
        if not t.get("jq_validated"):
            continue
        if t.get("d_plus_applied"):
            already += 1
            continue
        t["d_plus_applied"] = True
        t["d_plus_at"] = now[:19]  # 精确到秒
        t["d_plus_source"] = "trigger_d_plus_backstop"
        backstop += 1
    print(f"  标记核对: 已标记 {already} 条 (jq_feedback/历史) / 本次兜底补标 {backstop} 条")
    return backstop


def backfill_d_plus_marks(data: Dict, source: str = None,
                          apply: bool = False,
                          recover_fields: bool = False) -> Dict:
    """历史欠账回填 — **纯字段补写, 不重跑任何蒸馏** (2026-09-18)

    根因: ``TrajectoryLogger.to_dict()`` 是白名单重建, jq_*/d_plus_* 等非白名单键
    每次 ``_save()`` 被静默抹除 ⇒ ``jq_validated=True and not d_plus_applied``
    累积 235 条 (≈100% 历史 jq_validated)。

    🔴 纪律: **禁止**用跑一次 trigger_d_plus 来"清零"欠账 —— 那会真正重跑蒸馏,
    把 Memory 的 failed_attempts / occurrence_count 再推高一轮 (form_from_jq 的
    去重分支带 ``+= 1``)。本函数只补标记, 不触碰 memory。

    ── 两级策略 ─────────────────────────────────────────────
    Tier 1 (默认, apply=True 即生效)
        d_plus_applied = True
        d_plus_at      = ""            # 真实打标时刻已被抹除 ⇒ 留空, **不臆造**
        d_plus_source  = "backfill_YYYYMMDD"

    Tier 2 (需显式 recover_fields=True)
        从 backtest.meta 恢复 jq_ic / jq_icir / jq_maxdd。**默认关闭**, 因为
        实测 235 条中 153 条的 meta.ic 是**退化代理** ``ic = jq_return/100``
        (jq_feedback 的 ``ic=_ic if _ic is not None else jq_ret/100`` 兜底分支),
        并非真实 per-factor IC。回填会把"收益/100"伪装成"真实 IC", 直接破坏
        P-001(2026-08-29) 建立的"真实 IC 优先"契约。
        开启时加**双筛**: ① 不等于代理值 ② 量纲合理 (|ic|<=0.5, |icir|<=3,
        |maxdd|>0), 两条同时满足才写入。写入后 d_plus_source 追加 "+recover"。

    Parameters
    ----------
    apply: False → 只统计 (dry); True → 写回
    recover_fields: 是否启用 Tier 2 富字段恢复
    """
    src = source or ("backfill_" + datetime.now().strftime("%Y%m%d"))
    trajs = data.get("trajectories", [])
    stats = {"total": len(trajs), "pending": 0, "already": 0,
             "proxy_ic": 0, "proxy_icir": 0, "zero_maxdd": 0,
             "recoverable_ic": 0, "recoverable_icir": 0, "recoverable_maxdd": 0,
             "recovered_ic": 0, "recovered_icir": 0, "recovered_maxdd": 0,
             "recover_enabled": bool(recover_fields),
             "samples": [], "applied": False, "source": src}

    def _plausible_ic(v):
        return v is not None and abs(float(v)) <= 0.5

    def _plausible_icir(v):
        return v is not None and abs(float(v)) <= 3.0

    for t in trajs:
        if not t.get("jq_validated"):
            continue
        if t.get("d_plus_applied"):
            stats["already"] += 1
            continue
        stats["pending"] += 1
        r = t.get("jq_return")
        s = t.get("jq_sharpe")
        bt = t.get("backtest")
        if isinstance(bt, dict):
            meta = bt.get("meta") or {}
        elif hasattr(bt, "meta"):
            meta = getattr(bt, "meta") or {}
        else:
            meta = {}
        m_ic, m_icir, m_md = meta.get("ic"), meta.get("icir"), meta.get("maxdd")

        # 代理识别 (退化兜底分支的指纹)
        is_proxy_ic = (m_ic is not None and r is not None
                       and abs(float(m_ic) - float(r) / 100.0) < 1e-9)
        is_proxy_icir = (m_icir is not None and s is not None
                         and abs(float(m_icir) - float(s)) < 1e-9)
        if is_proxy_ic:
            stats["proxy_ic"] += 1
        if is_proxy_icir:
            stats["proxy_icir"] += 1
        if m_md is None or abs(float(m_md)) < 1e-12:
            stats["zero_maxdd"] += 1

        ok_ic = (not is_proxy_ic) and _plausible_ic(m_ic)
        ok_icir = (not is_proxy_icir) and _plausible_icir(m_icir)
        ok_md = (m_md is not None and abs(float(m_md)) >= 1e-12)

        if len(stats["samples"]) < 12:
            stats["samples"].append({
                "trajectory_id": t.get("trajectory_id"),
                "seed_factor": str(t.get("seed_factor", ""))[:34],
                "final_outcome": t.get("final_outcome"),
                "jq_return": r, "jq_sharpe": s,
                "meta_ic": m_ic, "meta_icir": m_icir, "meta_maxdd": m_md,
                "ic_is_proxy": is_proxy_ic,
                "recoverable_ic": ok_ic, "recoverable_icir": ok_icir,
            })

        # 诊断计数**始终**统计 (与是否写入解耦, 否则 dry 诊断恒 0 = 误导)
        if ok_ic:
            stats["recoverable_ic"] += 1
        if ok_icir:
            stats["recoverable_icir"] += 1
        if ok_md:
            stats["recoverable_maxdd"] += 1

        if recover_fields:
            if t.get("jq_ic") is None and ok_ic:
                stats["recovered_ic"] += 1
                if apply:
                    t["jq_ic"] = float(m_ic)
            if t.get("jq_icir") is None and ok_icir:
                stats["recovered_icir"] += 1
                if apply:
                    t["jq_icir"] = float(m_icir)
            if t.get("jq_maxdd") is None and ok_md:
                stats["recovered_maxdd"] += 1
                if apply:
                    t["jq_maxdd"] = float(m_md)

        if apply:
            t["d_plus_applied"] = True
            t["d_plus_at"] = ""            # 真实时刻不可恢复 ⇒ 留空, 不臆造
            t["d_plus_source"] = (src + "+recover") if recover_fields else src

    stats["applied"] = bool(apply)
    return stats


def _write_back_pool_status(factors: List[Dict]) -> int:
    """D+ 蒸馏后, 把已 JQ 回测因子的 passed_factor_pool.csv status 回写为 jq_done

    治本修复 (2026-08-17): pool 的 status=candidate 是历史遗留状态, JQ 结果此前不回写,
    导致看板/校验把已验证因子仍当作待验证候选。此后每次 D+ 蒸馏同步回写。
    """
    pool_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "passed_factor_pool.csv"))
    if not os.path.exists(pool_path):
        print("  [pool] 未找到 passed_factor_pool.csv, 跳过状态回写")
        return 0
    try:
        import pandas as pd
    except ImportError:
        print("  [pool] pandas 不可用, 跳过状态回写")
        return 0
    pool = pd.read_csv(pool_path)
    if "status" not in pool.columns or "name" not in pool.columns:
        print("  [pool] pool 缺 name/status 列, 跳过")
        return 0
    names = {str(f.get("factor_name", "")) for f in factors if f.get("factor_name")}
    mask = pool["name"].astype(str).isin(names)
    n = int(mask.sum())
    if n:
        pool.loc[mask, "status"] = "jq_done"
        pool.to_csv(pool_path, index=False, encoding="utf-8-sig")
        hit = sorted(pool.loc[mask, "name"].astype(str).tolist())
        print(f"  [pool] passed_factor_pool.csv 状态回写: {n} 条 → jq_done: {hit}")
    return n


def run(dry: bool = False, force: bool = False):
    """
    主入口: 检测 + 蒸馏

    Parameters
    ----------
    dry: 仅检测, 不蒸馏
    force: 强制重蒸馏所有 (忽略 d_plus_applied 标记)
    """
    # ── load ──
    print("=" * 60)
    print("  D+ 蒸馏自动触发器")
    print("=" * 60)

    data = load_trajectories()

    if force:
        # 清除所有 d_plus_applied 标记
        for t in data.get("trajectories", []):
            t.pop("d_plus_applied", None)
            t.pop("d_plus_at", None)
        print(f"  [force] 清除所有 d_plus_applied 标记")

    # ── detect ──
    pending = find_pending_jq_entries(data)
    print(f"\n检出未蒸馏条目: {len(pending)} 条")

    if not pending:
        print("  ✅ 无待蒸馏条目, 退出")
        return {"status": "no_pending", "count": 0}

    # 列出
    groups = group_by_breed(pending)
    print(f"  去重后分组: {len(groups)} 组")
    for key, entries in groups.items():
        t = entries[0]
        name = t.get("factor_name", "?")
        rating = t.get("jq_rating", "?")
        jq_ret = t.get("jq_return", 0)
        print(f"    {name}: {rating} | return={jq_ret} | {len(entries)} 轨迹")
        print(f"      formula: {key[:100]}")

    if dry:
        print("\n  [dry] 仅检测, 跳过蒸馏")
        return {"status": "dry_run", "pending": len(pending), "groups": len(groups)}

    # ── construct ──
    jq_feedback = construct_jq_feedback(groups, force=force)
    if not jq_feedback:
        print("  ❌ 无法构造 jq_backtest_result, 退出")
        return {"status": "error", "reason": "construct_failed"}

    print(f"\n构造 jq_backtest_result: {len(jq_feedback['factors'])} 因子")
    print(f"  composite: {jq_feedback['composite_return']:.1f}% / "
          f"Sharpe {jq_feedback['composite_sharpe']:.2f}")

    # ── L1 解读层: 蒸馏前归因 (2026-08-14) ──
    # 时序: L1 先于 distill, 使 interpretation 可参与两级蒸馏 (execution_failure 不罚 motif)
    l1_result = None
    if ENABLE_L1:
        try:
            from interpretation_agent import interpret_batch
            from experience_memory import get_memory as _get_memory_l1
            _mem_l1 = _get_memory_l1()
            print(f"\n[L1] 解读层启动: 对 {len(jq_feedback['factors'])} 因子做深度归因...")
            l1_result = interpret_batch(
                jq_backtest_result=jq_feedback,
                memory=_mem_l1,
                use_llm=L1_USE_LLM and not dry,
            )
            print(f"  [L1] 归因概览: {_summarize_interpretations(jq_feedback['factors'])}")
        except Exception as e:
            print(f"  [L1] ⚠️ 解读层失败 (不阻塞 D+): {e}")
            l1_result = None

    # ── distill ──
    try:
        from ralph_loop import RalphLoop
        from experience_memory import get_memory

        memory = get_memory()
        rl = RalphLoop(memory=memory)
        # P-20260903-004: trigger_d_plus 自己负责 lane 校准更新 (下方 v0.7 P4 块),
        # 传 update_lane=False 防同批次双更新 (update_from_jq_feedback 幂等但
        # 周频复评很重, 且 recalibrate 双调会污染 hysteresis 双批投票)。
        result = rl.jq_feedback(jq_backtest_result=jq_feedback, update_lane=False)
    except Exception as e:
        print(f"  ❌ D+ 蒸馏失败: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "reason": str(e)}

    # ── 修复 (2026-08-20): jq_feedback 内部已把 JQ 结果写入轨迹并落盘 ──
    # 旧代码用蒸馏前加载的 data 继续 mark/save, 会把 final_outcome/backtest 的
    # JQ 更新覆盖回旧值。此处重新加载, 保留 jq_feedback 的轨迹更新。
    data = load_trajectories()

    # ── L1 写回轨迹 (trajectory_log + memory attempts) ──
    if l1_result:
        try:
            from interpretation_agent import attach_to_memory
            n_mem = attach_to_memory(memory, jq_feedback.get("factors", []))
            n_traj = _attach_to_trajectories(data, jq_feedback.get("factors", []))
            print(f"  [L1] 写回: memory {n_mem} 条 / trajectory {n_traj} 条")
        except Exception as e:
            print(f"  [L1] ⚠️ 写回失败: {e}")

    # ── mark ──
    marked = mark_d_plus_applied(data, pending)
    save_trajectories(data)

    # ── pool 状态回写 (治本: 已验证因子不再被当作待验证候选) ──
    n_pool = _write_back_pool_status(jq_feedback.get("factors", []))

    # ── v0.7 P4: lane 校准滚动更新 (JQ 反馈 → lane_calibration → τ_w 重估) ──
    calib_report = None
    if not dry:
        try:
            from lane_calibration import update_from_jq_feedback, recalibrate_tau_w
            _up = update_from_jq_feedback(jq_feedback.get("factors", []))
            print(f"\n  [v0.7 P4] 校准集滚动更新: 新增 {_up['n_new']} / 更新 {_up['n_updated']} "
                  f"/ 跳过 {_up['n_skipped']} (周频复评成功 {_up['n_eval_ok']})")
            calib_report = recalibrate_tau_w()
            print(f"  [v0.7 P4] τ_w 重估: {calib_report['reason']}")
            print(f"  [v0.7 P4] τ_w 状态: current={calib_report['current_tau_w']} "
                  f"recommended={calib_report['recommended']} "
                  f"effective={calib_report['effective']} action={calib_report['action']}")
        except Exception as e:
            print(f"  [v0.7 P4] ⚠️ 校准集滚动更新失败 (不阻塞 D+): {e}")

    # ── P-20260903-004: D+ 闭环完整性自检 (只读, 只告警) ──
    try:
        from lane_calibration import verify_dplus_closure
        _names = [f.get("factor_name", "") for f in jq_feedback.get("factors", [])]
        _v = verify_dplus_closure(_names)
        if _v.get("unsynced"):
            print(f"  [D+闭环保] ⚠️ {len(_v['unsynced'])} 个 weekly 因子轨迹 JQ 时间"
                  f"晚于校准集 updated_at, 可能未同步: {_v['unsynced']}")
        elif _v.get("not_found"):
            print(f"  [D+闭环保] ℹ️ {len(_v['not_found'])} 个因子无轨迹匹配"
                  f"(新建轨迹无 completed_at): {_v['not_found']}")
        else:
            print(f"  [D+闭环保] ✓ 校准集同步正常 ({len(_v['synced'])} 因子核验, "
                  f"cal_updated_at={_v.get('cal_updated_at', '')[:16]})")
    except Exception as e:
        print(f"  [D+闭环保] ⚠️ 自检失败 (不阻塞): {e}")

    # ── summary ──
    print(f"\n{'='*60}")
    print(f"  D+ 蒸馏完成")
    print(f"    轨迹更新:     {result.get('trajectories_updated', 0)}")
    print(f"    硬禁止方向:   {result.get('hard_forbidden_added', 0)}")
    print(f"    JQ确认成功:   {result.get('jq_success_confirmed', 0)}")
    print(f"    Memory记录:   {memory.data['stats'].get('total_attempts', '?')}")
    print(f"    标记完成:     {marked} 条")
    print(f"    Pool 回写:    {n_pool} 条")
    print(f"{'='*60}")

    return {
        "status": "ok",
        "result": result,
        "marked": marked,
        "pending": len(pending),
    }


def run_backfill(apply: bool = False, recover_fields: bool = False):
    """入口: 历史 D+ 欠账回填 (纯字段补写, 不重跑蒸馏)"""
    print("=" * 68)
    print("  D+ 历史欠账回填 (纯字段补写, 不重跑蒸馏)")
    print("=" * 68)
    data = load_trajectories()
    stats = backfill_d_plus_marks(data, apply=apply, recover_fields=recover_fields)
    print(f"  轨迹总数              : {stats['total']}")
    print(f"  已标记 (真实 D+)      : {stats['already']}")
    print(f"  待回填                : {stats['pending']}")
    print(f"  --- 富字段可恢复性诊断 (Tier 2, 需 --recover-fields) ---")
    print(f"  meta.ic 是退化代理    : {stats['proxy_ic']}  (= jq_return/100, 非真实 IC)")
    print(f"  meta.icir 同代理      : {stats['proxy_icir']}")
    print(f"  meta.maxdd 为 0/缺失  : {stats['zero_maxdd']}")
    print(f"  双筛后 jq_ic 可恢复   : {stats['recoverable_ic']}  (实写 {stats['recovered_ic']})")
    print(f"  双筛后 jq_icir 可恢复 : {stats['recoverable_icir']}  (实写 {stats['recovered_icir']})")
    print(f"  双筛后 jq_maxdd 可恢复: {stats['recoverable_maxdd']}  (实写 {stats['recovered_maxdd']})")
    print(f"  Tier 2 是否启用       : {stats['recover_enabled']}")
    print(f"  d_plus_source         : {stats['source']}")
    print("  样本:")
    for s in stats["samples"]:
        print("    %-22s %-34s %-16s ret=%-8s ic=%-8s proxy=%-5s ok_ic=%s" % (
            str(s["trajectory_id"])[:22], s["seed_factor"][:34],
            str(s["final_outcome"])[:16], s["jq_return"],
            s["meta_ic"], s["ic_is_proxy"], s["recoverable_ic"]))
    if apply:
        save_trajectories(data)
        print("  ✅ 已写回 trajectory_log.json")
    else:
        print("  [dry] 未写回; 加 --apply 生效 (可另加 --recover-fields 启用 Tier 2)")
    return stats


if __name__ == "__main__":
    from pathlib import Path as _FuxiPath
    import sys as _FuxiSys
    _FuxiSys.path.insert(0, str(_FuxiPath(__file__).resolve().parents[2]))
    from fuxi_v1.legacy_writers import legacy_cli_guard as _FuxiGuard
    _fuxi_guard = _FuxiGuard(__file__, read_only=('--apply' not in sys.argv if '--backfill' in sys.argv else '--dry' in sys.argv))
    dry = "--dry" in sys.argv
    force = "--force" in sys.argv
    apply = "--apply" in sys.argv
    recover = "--recover-fields" in sys.argv

    if "--backfill" in sys.argv:
        run_backfill(apply=apply, recover_fields=recover)
    else:
        run(dry=dry, force=force)
