#!/usr/bin/env python
"""
Chronicle CLI — 实验编年史查询工具
===================================
用法:
    python chronicle/cli.py summary                          # 查看所有实验
    python chronicle/cli.py experiment <exp_id>              # 查看单次实验详情
    python chronicle/cli.py top-factors [--min-icir 0.3]    # 高ICIR因子排行
    python chronicle/cli.py top-strategies [--min-sharpe 0.5] # 最佳策略排行
    python chronicle/cli.py trend <factor_id>                # 因子ICIR历史趋势
    python chronicle/cli.py diff <exp1> <exp2>               # 两次实验对比
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from chronicle.db import ExperimentChronicle


def cmd_summary(chronicle: ExperimentChronicle, args):
    experiments = chronicle.get_recent_experiments(limit=args.limit)
    if not experiments:
        print("(no experiments recorded yet)")
        return

    print(f"{'Experiment ID':<35s} {'Version':<8s} {'Status':<12s} {'Duration':>8s}  Error")
    print("-" * 100)
    for e in experiments:
        dur = f"{e['total_duration_s']/60:.0f}m" if e.get('total_duration_s') else "-"
        err = (e.get('error_message', '') or '')[:50]
        print(f"{e['id']:<35s} {e['version']:<8s} {e['status']:<12s} {dur:>8s}  {err}")


def cmd_experiment(chronicle: ExperimentChronicle, args):
    summary = chronicle.get_experiment_summary(args.exp_id)
    if not summary or not summary.get("experiment"):
        print(f"Experiment '{args.exp_id}' not found.")
        return

    exp = summary["experiment"]
    print(f"\n=== {exp['id']} ===")
    print(f"  Version: {exp['version']}")
    print(f"  Status:  {exp['status']}")
    print(f"  Started: {exp['started_at']}")
    print(f"  Duration: {exp.get('total_duration_s', 0)/60:.1f} min")

    # Top factors
    factors = summary.get("top_factors", [])
    if factors:
        print(f"\n  Top Factors ({len(factors)}):")
        for f in factors[:15]:
            screening = "PASS" if f.get("passed_screening") else "FAIL"
            print(f"    {f['factor_id']:<30s} ICIR={f['icir_mean']:+.3f}  TO={f.get('estimated_turnover',0):.1%}  {screening}")

    # Strategies
    strategies = summary.get("strategies", [])
    if strategies:
        print(f"\n  Strategies ({len(strategies)}):")
        for s in strategies:
            print(f"    [{s['strategy_label']:<15s}] Sharpe(Phase5)={s.get('phase5_sharpe',0):.2f}  "
                  f"MaxDD={s.get('phase5_max_dd',0):.1%}  Turnover={s.get('phase5_avg_turnover',0):.1%}  "
                  f"Overfit={s.get('overfitting_risk','?')}")

    # GA
    ga = summary.get("ga", {})
    if ga:
        print(f"\n  GA: pop={ga.get('pop_size')} gen={ga.get('generations')} "
              f"pareto={ga.get('final_pareto_size')}")


def cmd_top_factors(chronicle: ExperimentChronicle, args):
    factors = chronicle.get_top_factors(
        experiment_id=args.exp_id, min_icir=args.min_icir, limit=args.limit
    )
    if not factors:
        print("No factors found.")
        return

    print(f"{'Factor':<30s} {'Category':<15s} {'ICIR':>8s} {'TO':>7s}  {'WFA':>6s}")
    print("-" * 75)
    for f in factors:
        icir = f.get("icir_mean", f.get("avg_icir", 0))
        to = f.get("estimated_turnover", f.get("avg_turnover", 0)) or 0
        wfa = f.get("wfa_survival_rate", 0) or 0
        print(f"{f['factor_id']:<30s} {f.get('category',''):<15s} {icir:>+7.3f} {to:>6.1%}  {wfa:>5.1%}")


def cmd_top_strategies(chronicle: ExperimentChronicle, args):
    strategies = chronicle.get_best_strategies(
        min_sharpe=args.min_sharpe, limit=args.limit
    )
    if not strategies:
        print("No strategies found.")
        return

    print(f"{'Experiment':<35s} {'Label':<15s} {'Sharpe':>7s} {'MaxDD':>7s} {'OOS_ICIR':>8s} {'Risk':>6s}")
    print("-" * 90)
    for s in strategies:
        print(f"{s['experiment_id']:<35s} {s['strategy_label']:<15s} "
              f"{s.get('phase5_sharpe',0):>+6.2f} {s.get('phase5_max_dd',0):>6.1%} "
              f"{s.get('oos_icir_mean',0):>+7.3f} {s.get('overfitting_risk','?'):>6s}")


def cmd_trend(chronicle: ExperimentChronicle, args):
    trend = chronicle.get_factor_trend(args.factor_id)
    if not trend:
        print(f"No history for factor '{args.factor_id}'.")
        return

    print(f"\nFactor: {args.factor_id}")
    print(f"{'Experiment':<35s} {'Version':<8s} {'ICIR':>7s} {'Turnover':>9s}")
    print("-" * 65)
    for t in trend:
        print(f"{t['id']:<35s} {t['version']:<8s} {t['icir_mean']:>+6.3f} {t.get('estimated_turnover',0):>8.1%}")


def cmd_diff(chronicle: ExperimentChronicle, args):
    s1 = chronicle.get_experiment_summary(args.exp1)
    s2 = chronicle.get_experiment_summary(args.exp2)

    if not s1.get("experiment"):
        print(f"'{args.exp1}' not found."); return
    if not s2.get("experiment"):
        print(f"'{args.exp2}' not found."); return

    print(f"\nComparing: {args.exp1}  vs  {args.exp2}")
    print()

    # Factor ICIR comparison
    f1_map = {f["factor_id"]: f["icir_mean"] for f in s1.get("top_factors", [])}
    f2_map = {f["factor_id"]: f["icir_mean"] for f in s2.get("top_factors", [])}
    all_factors = sorted(set(list(f1_map.keys()) + list(f2_map.keys())))

    print(f"{'Factor':<30s} {'ICIR('+args.exp1[-8:]+')':>12s} {'ICIR('+args.exp2[-8:]+')':>12s} {'Delta':>8s}")
    print("-" * 68)
    for fid in all_factors:
        v1 = f1_map.get(fid, 0)
        v2 = f2_map.get(fid, 0)
        delta = v2 - v1
        marker = " <<" if abs(delta) > 0.2 else ""
        print(f"{fid:<30s} {v1:>+11.3f} {v2:>+11.3f} {delta:>+7.3f}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Chronicle CLI - 实验编年史查询")
    parser.add_argument("--db", default=None, help="Chronicle DB 路径 (默认 outputs/chronicle.db)")

    sub = parser.add_subparsers(dest="command")

    p_summary = sub.add_parser("summary", help="查看最近实验列表")
    p_summary.add_argument("--limit", type=int, default=10)

    p_exp = sub.add_parser("experiment", help="查看单次实验详情")
    p_exp.add_argument("exp_id")

    p_tf = sub.add_parser("top-factors", help="高ICIR因子排行")
    p_tf.add_argument("--exp-id", default=None)
    p_tf.add_argument("--min-icir", type=float, default=0.3)
    p_tf.add_argument("--limit", type=int, default=20)

    p_ts = sub.add_parser("top-strategies", help="最佳策略排行")
    p_ts.add_argument("--min-sharpe", type=float, default=0.5)
    p_ts.add_argument("--limit", type=int, default=10)

    p_tr = sub.add_parser("trend", help="因子ICIR历史趋势")
    p_tr.add_argument("factor_id")

    p_diff = sub.add_parser("diff", help="两次实验对比")
    p_diff.add_argument("exp1")
    p_diff.add_argument("exp2")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    chronicle = ExperimentChronicle(args.db)
    cmds = {
        "summary": cmd_summary,
        "experiment": cmd_experiment,
        "top-factors": cmd_top_factors,
        "top-strategies": cmd_top_strategies,
        "trend": cmd_trend,
        "diff": cmd_diff,
    }
    cmds[args.command](chronicle, args)


if __name__ == "__main__":
    main()
