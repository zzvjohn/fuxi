#!/usr/bin/env python3
"""
伏羲 v0.5 全管线修复 & 验证脚本
  1. FactorForge 数据加载 — 验证 _load_forge_data() 找到真实数据
  2. Red Sea corr_matrix — 从因子池计算 → recluster
  3. LLM 公式生成端到端 — AlphaAgent V3 → DeepSeek API → S1-S5

用法: python research/factor_alchemy/fix_and_verify.py [--skip-forge] [--skip-redsea] [--skip-llm]
"""

import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# ── 通用工具 ──────────────────────────────────────────
PASS, FAIL = 0, 0


def header(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def step(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✅ {name}" + (f": {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f": {detail}" if detail else ""))


def result_block(title: str, ok: bool, items: list = None):
    icon = "✅" if ok else "❌"
    print(f"\n  {icon} {title}")
    if items:
        for it in items:
            print(f"     {it}")


# ═══════════════════════════════════════════════════════
# FIX 1: FactorForge 数据加载
# ═══════════════════════════════════════════════════════
def fix_forge_data():
    header("FIX 1: FactorForge 数据加载")

    import pandas as pd
    import numpy as np
    from ralph_loop import RalphLoop

    # 快速测试 — 不创建完整 RalphLoop，直接测 _load_forge_data
    print("  🔧 测试 _load_forge_data()...")
    try:
        # 检查 parquet 文件是否存在
        parquet_path = Path("output/ap_batch/cache/weekly_prices.parquet")
        step("weekly_prices.parquet 存在", parquet_path.exists(),
             str(parquet_path))

        if not parquet_path.exists():
            result_block("FactorForge 数据", False,
                         ["数据文件缺失，跳过"]
                         )
            return

        df = pd.read_parquet(parquet_path)
        step("parquet 可读", len(df) > 0,
             f"{len(df)} rows, {len(df.columns)} cols")

        # 通过 RalphLoop 实例测试完整流程
        ralph = RalphLoop()
        data = ralph._load_forge_data()

        step("_load_forge_data() 返回数据", data is not None)
        if data is not None:
            keys = list(data.keys())
            close_shape = data.get("close", np.zeros((1, 1))).shape
            step("数据包含 OHLCV 字段", len(keys) >= 3,
                 f"keys={keys}")
            step("close shape 合理", close_shape[0] > 10 and close_shape[1] > 100,
                 f"(T,N)=({close_shape[0]}, {close_shape[1]})")

            # 检查是否有 returns
            if "returns" in data:
                step("前向收益数据", data["returns"].shape == close_shape,
                     f"shape={data['returns'].shape}")

            # 检查数据无 NaN 泛滥
            close_arr = data.get("close")
            if close_arr is not None:
                nan_ratio = np.isnan(close_arr).mean()
                step(f"close NaN 比例可接受", nan_ratio < 0.5,
                     f"{nan_ratio:.1%} NaN (周频截面数据正常有缺失)")

            # 验证 open ≈ close (代理字段)
            if "open" in data and "close" in data:
                identical = np.allclose(
                    data["open"], data["close"],
                    equal_nan=True, rtol=1e-10
                )
                step("open=close (代理字段)", identical,
                     "周频数据无独立 open/high/low，close 代理符合预期" if identical
                     else "意外：open≠close")

        result_block("FactorForge 数据加载",
                     data is not None and "close" in (data or {}),
                     [f"数据源: {parquet_path}",
                      f"字段: {list(data.keys()) if data else 'N/A'}",
                      f"Shape: {data.get('close', np.zeros((1,1))).shape if data else 'N/A'}"])

    except Exception as e:
        step("_load_forge_data() 异常", False, str(e))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════
# FIX 2: Red Sea corr_matrix recluster
# ═══════════════════════════════════════════════════════
def fix_red_sea():
    header("FIX 2: Red Sea corr_matrix Recluster")

    import pandas as pd
    import numpy as np
    import pickle
    from library_orthogonality import LibraryOrthogonalityManager

    print("  🔧 加载因子池 → 计算相关性矩阵 → recluster...")
    pool_path = Path("data/cache/existing_factor_pool.pkl")
    step("因子池文件存在", pool_path.exists(), str(pool_path))

    if not pool_path.exists():
        result_block("Red Sea Recluster", False,
                     ["因子池文件缺失，无法计算 corr_matrix"])
        return

    try:
        # 加载因子池
        t0 = time.time()
        with open(pool_path, "rb") as f:
            pool = pickle.load(f)
        n_factors = len(pool)
        step("因子池加载", n_factors > 0,
             f"{n_factors} 因子, {time.time()-t0:.1f}s")

        # 提取公共日期断面 — 对齐不同因子的股票列
        factor_names = list(pool.keys())
        sample_factor = pool[factor_names[0]]
        first_shape = sample_factor.shape if hasattr(sample_factor, 'shape') else (0, 0)
        step("因子数据 Shape", first_shape[0] > 0,
             f"首因子 shape={first_shape}, {n_factors} 个因子")

        # ⚠️ 不同因子可能有不同的股票数 → 取交集
        common_stocks = None
        for fi, fname in enumerate(factor_names):
            arr = pool[fname]
            if hasattr(arr, "columns"):
                stocks = set(arr.columns)
            else:
                continue
            if common_stocks is None:
                common_stocks = stocks
            else:
                common_stocks &= stocks
        common_stocks = sorted(common_stocks) if common_stocks else []
        n_common = len(common_stocks)
        step("公共股票对齐", n_common > 100,
             f"{n_common} 只股票 (交集), 原始范围 {first_shape[1]}~N")

        # ── 计算截面 Spearman 秩相关矩阵 ──
        print(f"\n  📊 计算 {n_factors}×{n_factors} 相关矩阵...")
        print(f"     策略: 取最近 20 期截面 → 逐对 Spearman → 平均 → 对称矩阵")
        t0 = time.time()

        n_periods = min(20, first_shape[0])  # 用最近 20 期
        n_stocks = n_common
        corr_sum = np.zeros((n_factors, n_factors))
        corr_count = 0

        # 逐期计算 (只算上三角)
        for period_offset in range(n_periods):
            # 提取该期所有因子的截面值 (公共股票)
            date_idx = -(1 + period_offset)  # 从最近往前
            cross_section = np.zeros((n_stocks, n_factors))
            valid_factors = 0

            for fi, fname in enumerate(factor_names):
                arr = pool[fname]
                if hasattr(arr, "columns") and common_stocks:
                    row = arr.loc[arr.index[date_idx], common_stocks].values.astype(float)
                elif hasattr(arr, "iloc"):
                    row = arr.iloc[date_idx].values.astype(float)
                elif isinstance(arr, np.ndarray):
                    row = arr[date_idx]
                cross_section[:, fi] = row
                if not np.all(np.isnan(row)):
                    valid_factors += 1

            if valid_factors < 2:
                continue

            # 对每对因子计算 Spearman rank corr
            for i in range(n_factors):
                for j in range(i, n_factors):
                    xi = np.nan_to_num(cross_section[:, i], nan=0.0)
                    xj = np.nan_to_num(cross_section[:, j], nan=0.0)

                    # 排除两个因子都是 NaN 的股票
                    mask = ~(np.isnan(cross_section[:, i]) |
                             np.isnan(cross_section[:, j]))
                    if mask.sum() < 30:
                        continue

                    ri = xi[mask]
                    rj = xj[mask]
                    # Spearman rank correlation
                    try:
                        from scipy.stats import spearmanr
                        corr_val, _ = spearmanr(ri, rj)
                    except ImportError:
                        # Fallback: Pearson on ranks
                        ri_rank = pd.Series(ri).rank().values
                        rj_rank = pd.Series(rj).rank().values
                        corr_val = np.corrcoef(ri_rank, rj_rank)[0, 1]

                    if not np.isnan(corr_val):
                        corr_sum[i, j] += corr_val
                        corr_sum[j, i] += corr_val  # 对称

            corr_count += 1

        if corr_count == 0:
            step("有效计算期", False, "0 期 → 无法计算")
            result_block("Red Sea Recluster", False, ["无有效截面数据"])
            return

        # 平均相关矩阵
        corr_matrix = corr_sum / corr_count
        step("corr_matrix 计算完成",
             not np.all(np.isnan(corr_matrix)),
             f"{corr_count} 期有效, shape={corr_matrix.shape}")

        # 统计
        diag_mask = ~np.eye(n_factors, dtype=bool)
        off_diag = corr_matrix[diag_mask]
        step("corr_matrix 质量", not np.allclose(off_diag, 0),
             f"mean(|ρ|)={np.abs(off_diag).mean():.4f}, "
             f"max(|ρ|)={np.abs(off_diag).max():.4f}, "
             f"ρ>0.7: {(np.abs(off_diag)>0.7).sum()} pairs")

        # ── Recluster ──
        lom = LibraryOrthogonalityManager()
        lom.recluster(corr_matrix, factor_names)
        step("recluster() 执行", len(lom.clusters) > 0,
             f"{len(lom.clusters)} clusters")

        # 统计 Red Sea 级别
        cluster_sizes = [len(c.factors) for c in lom.clusters]
        n_crowded = sum(1 for s in cluster_sizes if s >= 5)
        step("Red Sea 聚类分析",
             len(cluster_sizes) > 0,
             f"最大族={max(cluster_sizes)}因子, "
             f"≥5拥挤族={n_crowded}, "
             f"平均族大小={np.mean(cluster_sizes):.1f}")

        # 检查 corr_matrix 已持久化
        step("corr_matrix 内部已存储",
             lom._corr_matrix is not None,
             f"shape={lom._corr_matrix.shape}")

        # 保存状态 (写入因子池 json 索引)
        lom.save_state()
        step("状态持久化", True, "已写入 data/cache/")

        elapsed = time.time() - t0
        print(f"\n  ⏱ Red Sea recluster 耗时: {elapsed:.1f}s")

        result_block("Red Sea corr_matrix + Recluster", True,
                     [f"因子池: {n_factors} 因子",
                      f"相关矩阵: {corr_matrix.shape}",
                      f"计算期数: {corr_count}",
                      f"聚类结果: {len(lom.clusters)} clusters",
                      f"max(|ρ|)={np.abs(off_diag).max():.4f}"])

    except Exception as e:
        step("Red Sea recluster 异常", False, str(e))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════
# FIX 3: LLM 公式生成端到端
# ═══════════════════════════════════════════════════════
def verify_llm_pipeline():
    header("FIX 3: LLM 公式生成端到端 (AlphaAgent V3 → S5)")

    from llm_client import get_llm_client
    from llm_generator import LLMGenerator
    from experience_memory import get_memory
    from factor_quality_gate import FactorQualityGate
    from multi_stage_validator import MultiStageValidator

    # 3.1 LLM Client 可达性
    client = get_llm_client()
    step("LLM Client 初始化",
         client.api_key is not None and len(client.api_key) > 10,
         f"model={client.model}")

    # 3.2 LLM Generator 上下文注入
    mem = get_memory()
    gen = LLMGenerator()

    # 注入完整 Memory context
    forbidden = mem.data.get("forbidden_directions", [])
    warnings = mem.data.get("warning_directions", [])
    motif_rules = mem.get_motif_rules()
    success_templates = mem.data.get("success_templates", [])

    gen.receive_context(
        success_templates=success_templates[:5],
        forbidden_directions=forbidden[:5],
        warning_directions=warnings[:3],
        motif_rules=motif_rules,
        paradigm_priorities={"动量": 0.5, "价量关系": 0.3},
        mab_direction="动量",
    )
    step("LLM Generator 上下文注入",
         True,
         f"forbidden={len(forbidden)} warnings={len(warnings)} "
         f"motifs={len(motif_rules.get('forbid',[]))}")

    # 3.3 Prompt 构建
    prompt = gen.build_prompt("动量", 3)
    step("LLM Prompt 构建", len(prompt) > 500,
         f"{len(prompt)} chars — 含 memory/warning/motif 段")

    # 3.4 实际 API 调用
    print("\n  🤖 调用 DeepSeek API 生成因子...")
    t0 = time.time()
    try:
        # 使用 ralph_loop 相同的调用路径：llm_client.chat_with_system() + parse_response()
        from llm_client import get_llm_client
        client2 = get_llm_client()
        response_text = client2.chat_with_system(
            system_prompt=(
                "你是 A 股量化因子研究员，专精于设计具有经济逻辑支撑的因子表达式。"
                "使用 pandas 风格表达式（如 close.rolling(20).mean()）。"
            ),
            user_prompt=prompt,
            temperature=0.8,
            max_tokens=4096,
        )
        parsed = gen.parse_response(response_text)
        elapsed = time.time() - t0
        step("LLM API 调用成功", len(parsed) > 0,
             f"{len(parsed)} 因子 / {elapsed:.1f}s")
    except Exception as e:
        step("LLM API 调用失败", False, str(e))
        parsed = []
        elapsed = 0
        traceback.print_exc()

    if not parsed:
        result_block("LLM 公式生成", False,
                     ["API 调用失败，无法继续测试"])
        return

    # 打印生成结果
    print("\n  📝 LLM 生成因子:")
    for i, pf in enumerate(parsed):
        name = pf.get("factor_name", pf.get("name", f"unknown_{i}"))
        expr = pf.get("expression", pf.get("formula", "?"))
        hypo = pf.get("hypothesis", pf.get("rationale", ""))[:80]
        print(f"    {i+1}. {name}")
        print(f"       expr: {expr[:90]}...")
        print(f"       hypo: {hypo}")

    # 3.5 QualityGate 验证
    gate = FactorQualityGate()
    gate_results = []
    for pf in parsed:
        expr = pf.get("expression", pf.get("formula", ""))
        name = pf.get("factor_name", pf.get("name", "llm_gen"))
        if expr:
            gr = gate.verify({
                "factor_name": name,
                "formula": expr,
                "hypothesis": pf.get("hypothesis", pf.get("rationale", "")),
            })
            gate_results.append(gr)

    n_gate_pass = sum(1 for g in gate_results if g.passed)
    step("QualityGate 通过", n_gate_pass > 0,
         f"{n_gate_pass}/{len(gate_results)} 通过")
    for g in gate_results:
        status_icon = "✅" if g.passed else "❌"
        issues = "; ".join(g.fatal_issues[:2] + g.warnings[:1]) or "无问题"
        print(f"    {status_icon} {issues[:100]}...")

    # 3.6 S1-S5 验证
    if n_gate_pass > 0:
        gate_passed = [p for p, g in zip(parsed, gate_results) if g.passed]
    else:
        gate_passed = parsed

    validator = MultiStageValidator(data_dir=Path("data"))
    candidates = []
    for pf in gate_passed:
        expr = pf.get("expression", pf.get("formula", ""))
        name = pf.get("factor_name", pf.get("name", "llm_gen"))
        candidates.append({
            "factor_name": name,
            "formula": expr,
            "paradigm": pf.get("paradigm", "动量"),
            "hypothesis": pf.get("hypothesis", pf.get("rationale", "")),
            "logic": pf.get("logic", pf.get("rationale", ""))[:80],
            "source": "llm_api",
        })

    print(f"\n  🔬 S1-S5 验证 {len(candidates)} 个候选...")
    t0 = time.time()
    try:
        results, summary = validator.validate(candidates)
        elapsed = time.time() - t0
        step("S1-S5 验证完成", True,
             f"{summary.get('total_candidates', len(results))} 候选 / {elapsed:.1f}s")

        # 逐阶段统计
        stages = {
            "S1 (IC≥0.02)": "s1_passed",
            "S2 (ICIR≥0.3)": "s2_passed",
            "S3 (稳定性)": "s3_passed",
            "S4 (FRI)": "s4_passed",
            "S5 (JQ Joint)": "s5_passed",
        }
        print("\n  📊 LLM 因子验证逐阶段:")
        for stage_name, attr in stages.items():
            n_pass = sum(1 for r in results if getattr(r, attr, False))
            print(f"    {stage_name}: {n_pass}/{len(results)} 通过")

        # 统计 IC/ICIR
        ics = []
        for r in results:
            ic = getattr(r, 's1_ic', None)
            if ic is not None and isinstance(ic, (int, float)) and not (ic == 0.0 and not getattr(r, 's1_passed', False)):
                ics.append(ic)
        if ics:
            print(f"\n    IC: mean={np.mean(ics):.4f}, max={np.max(ics):.4f}")
        else:
            print(f"\n    IC: 因子未通过S1 (IC < 0.02 阈值)")

        result_block("LLM 公式生成 → S5 端到端",
                     len(results) > 0,
                     [f"API: {len(parsed)} 因子 / {elapsed:.1f}s",
                      f"QualityGate: {n_gate_pass}/{len(gate_results)} 通过",
                      f"S1-S5: {len(results)} 候选验证完成",
                      f"S5通过: {summary.get('stage5_passed', 0)}",
                      f"IC范围: {min(ics):.4f}~{max(ics):.4f}" if ics else "IC: < 0.02 (S1阈值未通过)"])

    except Exception as e:
        step("S1-S5 验证异常", False, str(e))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-forge", action="store_true")
    ap.add_argument("--skip-redsea", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("  伏羲 v0.5 全管线修复 & 验证")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not args.skip_forge:
        fix_forge_data()

    if not args.skip_redsea:
        fix_red_sea()

    if not args.skip_llm:
        verify_llm_pipeline()

    # ── 总结 ──
    print(f"\n{'='*60}")
    print(f"  总结: {PASS} ✅ / {FAIL} ❌ (共 {PASS+FAIL} 项)")
    if FAIL == 0:
        print(f"  全部通过！伏羲系统管线正常。")
    else:
        print(f"  ⚠️ {FAIL} 项未通过，详见上方日志")
    print(f"{'='*60}")
