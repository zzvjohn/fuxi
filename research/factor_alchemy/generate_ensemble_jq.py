# -*- coding: utf-8 -*-
"""
Ensemble Top-K JQ 策略生成器 (v8.1)

从 ga_result.json 的完整 Pareto 前沿中, 取每个 regime 的 Top-K 个非被支配解,
对其权重取均值 (等权 Ensemble), 生成单个 Ensemble 策略文件。

原理:
  - 单点优化 → 过拟合 local 数据 → JQ 回测坍缩 (v8.0 证实)
  - Top-K Ensemble → 多个 Pareto 最优解权重平均 → 降低方差 → 更鲁棒

论文支撑:
  - Eigen-Portfolios (Avellaneda & Lee 2010): 组合多样化降低过拟合
  - RegimeFolio (2025): Pareto 前沿 Ensemble 优于单点选择
  - Harvey & Liu (2015): 多重检验下的因子选择应保守聚合

用法:
  python generate_ensemble_jq.py [ga_result.json] [--top-k 5] [--strategy-name ensemble_v1]
"""

import argparse
import json
import os
import sys
import datetime
from pathlib import Path
from collections import defaultdict

# 确保 jq_generator 可 import
sys.path.insert(0, str(Path(__file__).parent))

from jq_generator import JQGenerator

# 禁投清单: 这些因子在 JQ 上被证伪, 即使 Pareto 前沿含其权重也必须排除
FORBIDDEN_FACTORS = {
    'vm_diff',          # ☠️ 毒因子 (JQ 死重)
    'bargaining_power_proxy',
    'relative_spread_proxy',
    'avg_turnover_3m',
    'min_ret_1m',
    'atr_14',
}

ENSEMBLE_METHODS = ['mean', 'median']


def compute_domination_counts(objectives_list):
    """计算每个解的 Pareto 被支配数 (0=最优, N-1=最差)"""
    n = len(objectives_list)
    dominated = [0] * n
    for i in range(n):
        qi, si, ci = objectives_list[i]
        for j in range(n):
            if i == j:
                continue
            qj, sj, cj = objectives_list[j]
            # j dominates i: all obj >= i AND at least one strict >
            if qj >= qi and sj >= si and cj >= ci and (qj > qi or sj > si or cj > ci):
                dominated[i] += 1
    return dominated


def select_top_k_pareto(solutions, K=5):
    """从 Pareto 解列表中选 Top-K (按被支配数升序, 同 rank 按 sharpe 降序)"""
    if len(solutions) <= K:
        return solutions.copy()

    objectives = [(s['quality_obj'], s['stability_obj'], s['sharpe_obj'])
                  for s in solutions]
    dominated = compute_domination_counts(objectives)

    # (index, dominated_count, sharpe_obj)
    indexed = [(i, dominated[i], solutions[i]['sharpe_obj']) for i in range(len(solutions))]
    indexed.sort(key=lambda x: (x[1], -x[2]))  # 被支配少 + Sharpe 高优先

    return [solutions[i] for i, _, _ in indexed[:K]]


def ensemble_weights(solutions, method='mean'):
    """对 K 组权重做 Ensemble

    Args:
        solutions: list of dicts, each with 'weights' key
        method: 'mean' (算术平均) or 'median' (中位数, 对异常值更鲁棒)

    Returns:
        dict {factor_name: ensemble_weight}
    """
    if method not in ENSEMBLE_METHODS:
        raise ValueError(f"未知的 ensemble 方法: {method}, 可选: {ENSEMBLE_METHODS}")

    K = len(solutions)
    if K == 0:
        return {}

    # 收集所有因子名
    all_factors = set()
    for sol in solutions:
        all_factors.update(sol['weights'].keys())

    # 对每个因子, 在 K 个解中取 ensemble
    ensemble = {}
    for fname in sorted(all_factors):
        vals = []
        for sol in solutions:
            w = sol['weights'].get(fname, 0.0)
            vals.append(w)

        if method == 'mean':
            ensemble[fname] = float(np_mean(vals))
        elif method == 'median':
            ensemble[fname] = float(np_median(vals))

    # 归一化 (使权重和为 1)
    total = sum(ensemble.values())
    if total > 0:
        ensemble = {k: v / total for k, v in ensemble.items()}

    # 过滤禁投因子
    ensemble = {k: v for k, v in ensemble.items() if k not in FORBIDDEN_FACTORS}

    # 二次归一化
    total = sum(ensemble.values())
    if total > 0:
        ensemble = {k: round(v / total, 6) for k, v in ensemble.items()}

    return ensemble


def np_mean(vals):
    """纯 Python mean, 避免 numpy 依赖"""
    n = len(vals)
    if n == 0:
        return 0.0
    return sum(vals) / n


def np_median(vals):
    """纯 Python median"""
    n = len(vals)
    if n == 0:
        return 0.0
    sorted_vals = sorted(vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    else:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0


def generate_ensemble(ga_json_path, top_k=5, ensemble_method='mean',
                      strategy_name='ensemble_v1', output_dir=None):
    """主流程: 读 ga_result.json → 选 Top-K → Ensemble → 生成 JQ 策略

    Returns:
        dict {regime_name: output_file_path}
    """
    with open(ga_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pareto_all = data.get('pareto_solutions_all', {})
    if not pareto_all:
        print("[ERROR] ga_result.json 中无 pareto_solutions_all 字段")
        print("  请用最新版 run_fa.py (含 v8.1 修改) 重新运行 GA")
        return {}

    print(f"=== Ensemble Top-K JQ 策略生成器 (v8.1) ===")
    print(f"  源文件: {ga_json_path}")
    print(f"  Top-K:  {top_k}")
    print(f"  方法:   {ensemble_method}")
    print(f"  策略名: {strategy_name}")
    print(f"  禁投:   {', '.join(sorted(FORBIDDEN_FACTORS))}")

    if output_dir is None:
        output_dir = Path(ga_json_path).parent.parent / 'strategies' / 'drafts'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    results = {}

    for regime_name, solutions in pareto_all.items():
        if not solutions:
            print(f"\n  [{regime_name}] 无 Pareto 解, 跳过")
            continue

        print(f"\n  [{regime_name}] {len(solutions)} 个 Pareto 解")

        # Step 1: 选 Top-K
        top_solutions = select_top_k_pareto(solutions, K=top_k)
        print(f"    选取 Top-{len(top_solutions)} (被支配数=0 的比例: "
              f"{sum(1 for s in top_solutions if s['dominated_by'] == 0)}/{len(top_solutions)})")

        # Step 2: 打印 Top-K 详细信息
        for i, sol in enumerate(top_solutions):
            w = sol['weights']
            factors_str = ", ".join(f"{f}={v:.3f}" for f, v in
                                   sorted(w.items(), key=lambda x: -x[1]))
            print(f"    #{i+1} rank={sol['rank']} dom={sol['dominated_by']} "
                  f"Q={sol['quality_obj']:.3f} S={sol['stability_obj']:.3f} "
                  f"C={sol['sharpe_obj']:.3f}")
            print(f"        {factors_str}")

        # Step 3: Ensemble 权重
        ens_weights = ensemble_weights(top_solutions, method=ensemble_method)

        # 过滤禁投因子
        filtered_out = set()
        for sol in top_solutions:
            filtered_out.update(set(sol['weights'].keys()) & FORBIDDEN_FACTORS)

        if filtered_out:
            print(f"    ⚠️ 已过滤禁投因子: {', '.join(sorted(filtered_out))}")

        if not ens_weights:
            print(f"    [SKIP] Ensemble 后无有效因子 (可能全部被禁投过滤)")
            continue

        print(f"    Ensemble 权重 ({len(ens_weights)} 因子):")
        for f, w in sorted(ens_weights.items(), key=lambda x: -x[1]):
            # 检查是否是 Top-K 中多解的共有因子
            n_shared = sum(1 for sol in top_solutions if f in sol['weights'])
            print(f"      {f:35s} w={w:.4f}  (共现 {n_shared}/{len(top_solutions)} 解)")

        # Step 4: 生成 JQ 策略
        safe_regime = regime_name.replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"fa_v81_ensemble_{strategy_name}_{safe_regime}_{timestamp}.py"
        filepath = output_dir / filename

        print(f"\n    生成 JQ 策略...")
        try:
            gen = JQGenerator(
                weights=ens_weights,
                name=f"v8.1_ens_{strategy_name}_{safe_regime}",
                mcap_min=5,
                stock_num=30,
                max_turnover=0.60,
            )
            gen.generate(str(filepath))
            print(f"    [OK] {filename}")
            results[regime_name] = str(filepath)
        except Exception as e:
            print(f"    [FAIL] {e}")
            import traceback
            traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble Top-K JQ 策略生成器 (v8.1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate_ensemble_jq.py
  python generate_ensemble_jq.py output/ga_result.json --top-k 7 --method median
  python generate_ensemble_jq.py output/ga_result.json --name test_v2
        """
    )
    parser.add_argument('ga_json', nargs='?',
                        default='output/ga_result.json',
                        help='ga_result.json 路径 (默认: output/ga_result.json)')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                        help='选取 Top-K 个 Pareto 解 (默认: 5)')
    parser.add_argument('--method', '-m', choices=ENSEMBLE_METHODS, default='mean',
                        help='Ensemble 方式: mean(均值) 或 median(中位数) (默认: mean)')
    parser.add_argument('--name', '-n', type=str, default='ensemble_v1',
                        help='策略名称前缀 (默认: ensemble_v1)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='输出目录 (默认: strategies/drafts/)')

    args = parser.parse_args()

    ga_path = Path(args.ga_json)
    if not ga_path.exists():
        print(f"[ERROR] 文件不存在: {ga_path}")
        return 1

    results = generate_ensemble(
        str(ga_path),
        top_k=args.top_k,
        ensemble_method=args.method,
        strategy_name=args.name,
        output_dir=args.output_dir,
    )

    if not results:
        print("\n[FAIL] 未生成任何 Ensemble 策略")
        return 1

    print(f"\n=== 生成完成: {len(results)} 个 Ensemble 策略 ===")
    for regime, path in results.items():
        print(f"  [{regime}] {path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
