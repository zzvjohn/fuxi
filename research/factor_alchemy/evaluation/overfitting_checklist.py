"""
防过拟合检查清单 (Overfitting Checklist) — XQuant Ch6 方法论
============================================================
Formalized "四把铲子" (Four Shovels) from Brian Peterson's methodology.

五大检查维度:
  1. 样本外验证 (OOS) — ICIR 衰减分析
  2. Walk-forward 稳定性 — Anchored + Rolling 双验证
  3. 规则负担 (Rule Burden) — Plateau 检测 & 最优因子数
  4. 交叉验证 — 时序K-fold (TBD)
  5. 参数敏感性 — 高原型 vs 山峰型 检测

Scoring rules 严格按照 Ch6 表6-8 checklist 执行。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class OverfittingChecklist:
    """防过拟合检查清单 — XQuant Ch6 方法论

    实现 Brian Peterson 的 "四把铲子" 框架化 checklist,
    对接 FA 流水线各阶段结果, 输出格式化表格和综合判决。
    """

    def __init__(self):
        self.checks = {}

    # ================================================================
    # Check #1: 样本外验证 (OOS)
    # ================================================================
    def check_oos(self, oos_results: dict) -> dict:
        """OOS 验证: ICIR 衰减分析

        比较训练集 ICIR 和 OOS ICIR, 计算衰减幅度。

        Args:
            oos_results: Phase 5.5 输出的 dict,
                {strategy_name: {icir_train, icir_oos, delta, robust}}

        Returns:
            dict: {status, result, max_decay, details}
        """
        if not oos_results:
            return {
                'status': 'SKIP',
                'result': '[SKIP]',
                'reason': '无OOS数据',
            }

        max_decay = 0.0
        details = []

        for name, r in oos_results.items():
            icir_train = abs(r.get('icir_train', 0))
            icir_oos = r.get('icir_oos', 0)

            if icir_train < 1e-10:
                decay = float('inf') if abs(icir_oos) > 1e-10 else 0.0
            else:
                decay = abs(icir_train - icir_oos) / icir_train

            max_decay = max(max_decay, decay)

            if icir_oos <= 0:
                item_status = 'FAIL'
            elif decay >= 1.0:
                item_status = 'FAIL'
            elif decay >= 0.5:
                item_status = 'WARN'
            else:
                item_status = 'PASS'

            details.append({
                'strategy': name,
                'icir_train': icir_train,
                'icir_oos': icir_oos,
                'decay': decay,
                'status': item_status,
            })

        # 整体判定: 取最差结果
        all_statuses = [d['status'] for d in details]
        if 'FAIL' in all_statuses:
            status = 'FAIL'
            result_str = '[FAIL]'
        elif 'WARN' in all_statuses:
            status = 'WARN'
            result_str = '[WARN]'
        else:
            status = 'PASS'
            result_str = '[PASS]'

        return {
            'status': status,
            'result': result_str,
            'max_decay': max_decay,
            'details': details,
        }

    # ================================================================
    # Check #2: Walk-forward 稳定性
    # ================================================================
    def check_walkforward(self, wf_results: dict) -> dict:
        """Walk-forward 稳定性检查

        检查 Anchored 和 Rolling 两路 stability_score,
        双验证都 >= 70 才算通过。

        Args:
            wf_results: Phase 7 输出的 dict,
                {anchored: {stability_score, verdict, ...},
                 rolling:  {stability_score, verdict, ...}}

        Returns:
            dict: {status, result, anchored_score, rolling_score}
        """
        if not wf_results:
            return {
                'status': 'SKIP',
                'result': '[SKIP]',
                'reason': '无WF数据',
            }

        anchored = wf_results.get('anchored', {})
        rolling = wf_results.get('rolling', {})

        a_score = anchored.get('stability_score', 0) if isinstance(anchored, dict) else 0
        r_score = rolling.get('stability_score', 0) if isinstance(rolling, dict) else 0

        if a_score >= 70 and r_score >= 70:
            status = 'PASS'
            result_str = '[PASS]'
        elif a_score >= 50 and r_score >= 50:
            status = 'WARN'
            result_str = '[WARN]'
        else:
            status = 'FAIL'
            result_str = '[FAIL]'

        return {
            'status': status,
            'result': result_str,
            'anchored_score': a_score,
            'rolling_score': r_score,
        }

    # ================================================================
    # Check #3: 规则负担 (Rule Burden)
    # ================================================================
    def check_rule_burden(self, burden_results,
                          current_max_factors: int = None) -> dict:
        """规则负担检查

        从因子堆叠结果中检测 plateau, 找到最优 n_factors,
        与当前 MAX_FACTORS 比较判断是否过度复杂化。

        Args:
            burden_results: run_factor_count_stacking 返回的 DataFrame,
                columns: n_factors, icir_train, icir_test, selected_factors
            current_max_factors: 当前 MAX_FACTORS 配置值

        Returns:
            dict: {status, result, optimal_n_factors, has_plateau, ...}
        """
        if burden_results is None:
            return {
                'status': 'SKIP',
                'result': '[SKIP]',
                'reason': '无负担分析数据',
            }

        if hasattr(burden_results, 'empty') and burden_results.empty:
            return {
                'status': 'SKIP',
                'result': '[SKIP]',
                'reason': '负担分析DataFrame为空',
            }

        df = burden_results

        if 'icir_test' not in df.columns or 'n_factors' not in df.columns:
            return {
                'status': 'SKIP',
                'result': '[SKIP]',
                'reason': '缺少必要列(icir_test, n_factors)',
            }

        # 最优因子数 = OOS ICIR 最大值
        best_idx = df['icir_test'].idxmax()
        optimal_n = int(df.loc[best_idx, 'n_factors'])
        best_icir = float(df.loc[best_idx, 'icir_test'])

        # Plateau 检测: 连续两步没有改善
        icir_vals = df['icir_test'].values
        if len(icir_vals) >= 2:
            diffs = np.diff(icir_vals)
            has_plateau = any(
                diffs[i] <= 0 and diffs[i + 1] <= 0
                for i in range(len(diffs) - 1)
            )
            if not has_plateau and len(diffs) >= 1 and diffs[-1] <= 1e-4:
                has_plateau = True
        else:
            has_plateau = False

        # 获取当前 MAX_FACTORS
        if current_max_factors is None:
            try:
                from config import MAX_FACTORS as current_max_factors
            except ImportError:
                current_max_factors = optimal_n

        if not has_plateau:
            status = 'FAIL'
            result_str = '[FAIL]'
        elif current_max_factors == optimal_n:
            status = 'PASS'
            result_str = '[PASS]'
        elif current_max_factors > optimal_n:
            status = 'WARN'
            result_str = '[WARN]'
        else:
            # current < optimal: 还有优化空间
            status = 'WARN'
            result_str = '[WARN]'

        return {
            'status': status,
            'result': result_str,
            'optimal_n_factors': optimal_n,
            'optimal_icir': best_icir,
            'current_max_factors': current_max_factors,
            'has_plateau': has_plateau,
        }

    # ================================================================
    # Check #4: 交叉验证 (TBD)
    # ================================================================
    def check_cross_validation(self) -> dict:
        """交叉验证 — 时序K-fold 待实现"""
        return {
            'status': 'NOT_IMPL',
            'result': '[NOT IMPL]',
            'reason': '时序K-fold待实现',
        }

    # ================================================================
    # Check #5: 参数敏感性 (高原型 vs 山峰型)
    # ================================================================
    def check_parameter_sensitivity(self, ic_summary: pd.DataFrame,
                                     factor_dfs: dict = None,
                                     forward_returns: pd.DataFrame = None,
                                     n_subperiods: int = 3) -> dict:
        """参数敏感性: 检测因子 ICIR 是高原型还是山峰型

        将全时段拆成 n_subperiods 个子期, 计算每个因子在各子期的 ICIR,
        用 CV = std/mean 衡量稳定性。
        - 高原型 (CV < 0.5): ICIR 跨期稳定, 参数不敏感 → 好信号
        - 山峰型 (CV >= 0.5): ICIR 剧烈波动, 参数敏感 → 脆弱

        Args:
            ic_summary: ICIR 汇总 DataFrame (index=factor_names)
            factor_dfs: {name: DataFrame} 因子面板 (子期分析需要)
            forward_returns: 前向收益 (子期分析需要)
            n_subperiods: 子期数量

        Returns:
            dict: {status, result, top_factors, sensitivity, ...}
        """
        if ic_summary is None or (hasattr(ic_summary, 'empty') and ic_summary.empty):
            return {
                'status': 'SKIP',
                'result': '[SKIP]',
                'reason': '无ICIR数据',
            }

        if 'ICIR' not in ic_summary.columns:
            return {
                'status': 'SKIP',
                'result': '[SKIP]',
                'reason': 'ICIR列不存在',
            }

        # Top 3 因子 (按 |ICIR|)
        sorted_ic = ic_summary['ICIR'].abs().sort_values(ascending=False)
        top_3 = sorted_ic.head(3).index.tolist()

        # 如果没有因子面板数据, 跳过子期分析
        if not factor_dfs or forward_returns is None:
            return {
                'status': 'NOT_IMPL',
                'result': '[NOT IMPL]',
                'reason': '缺少因子数据, 无法子期分析',
                'top_factors': top_3,
            }

        # 拆子期
        dates = pd.DatetimeIndex(forward_returns.index)
        n_total = len(dates)
        if n_total < n_subperiods * 20:
            n_subperiods = max(2, n_total // 20)

        sub_size = n_total // n_subperiods

        from evaluation.ic_analysis import compute_ic_icir

        sub_period_icirs: Dict[str, list] = {name: [] for name in top_3}

        for i in range(n_subperiods):
            start = i * sub_size
            end = n_total if i == n_subperiods - 1 else (i + 1) * sub_size
            sub_dates = dates[start:end]
            sub_fr = forward_returns.loc[sub_dates]

            for name in top_3:
                if name not in factor_dfs:
                    sub_period_icirs[name].append(np.nan)
                    continue

                factor_df = factor_dfs[name]
                common_idx = sub_dates.intersection(factor_df.index)
                if len(common_idx) < 10:
                    sub_period_icirs[name].append(np.nan)
                    continue

                try:
                    icr = compute_ic_icir(
                        factor_df.reindex(common_idx),
                        sub_fr.reindex(common_idx),
                    )
                    sub_period_icirs[name].append(abs(icr.get('icir', 0)))
                except Exception:
                    sub_period_icirs[name].append(np.nan)

        # 计算 CV = std / mean
        sensitivity = {}
        for name in top_3:
            vals = [v for v in sub_period_icirs[name] if not np.isnan(v)]
            if len(vals) >= 2:
                mu = np.mean(vals)
                cv = np.std(vals) / mu if mu > 1e-10 else 0.5
            else:
                mu = 0.0
                cv = 0.5
            sensitivity[name] = {
                'cv': cv,
                'mean': mu,
                'std': np.std(vals) if len(vals) >= 2 else 0.0,
                'sub_icirs': vals,
            }

        # 评分: 检查 top 3 有多少是高原型 (CV < 0.5)
        cv_list = [s['cv'] for s in sensitivity.values()]
        n_plateau = sum(1 for cv in cv_list if cv < 0.5)
        n_peak = sum(1 for cv in cv_list if cv >= 0.5)

        if n_plateau == len(top_3):
            status = 'PASS'
            result_str = '[PASS]'
        elif n_plateau >= 2:
            status = 'WARN'
            result_str = '[WARN]'
        else:
            status = 'FAIL'
            result_str = '[FAIL]'

        return {
            'status': status,
            'result': result_str,
            'top_factors': top_3,
            'sensitivity': sensitivity,
            'n_plateau': n_plateau,
            'n_peak': n_peak,
        }

    # ================================================================
    # Run All Checks
    # ================================================================
    def run_all(self, oos_results=None, wf_results=None, burden_results=None,
                ic_summary=None, factor_dfs=None, forward_returns=None,
                current_max_factors=None) -> dict:
        """运行全部 5 项检查

        Args:
            oos_results: Phase 5.5 OOS 结果 dict
            wf_results: Phase 7 Walk-forward 结果 dict
            burden_results: Phase 6 因子负担 DataFrame
            ic_summary: Phase 3 ICIR 汇总 DataFrame
            factor_dfs: {name: DataFrame} 因子面板
            forward_returns: 前向收益 DataFrame
            current_max_factors: 当前 MAX_FACTORS

        Returns:
            dict: {check_1_oos, check_2_wf, check_3_burden,
                   check_4_cv, check_5_sensitivity}
        """
        results = {
            'check_1_oos': self.check_oos(oos_results),
            'check_2_wf': self.check_walkforward(wf_results),
            'check_3_burden': self.check_rule_burden(
                burden_results, current_max_factors
            ),
            'check_4_cv': self.check_cross_validation(),
            'check_5_sensitivity': self.check_parameter_sensitivity(
                ic_summary, factor_dfs, forward_returns
            ),
        }

        self.checks = results
        return results

    # ================================================================
    # Print Checklist Table (Ch6 表6-8 格式)
    # ================================================================
    def print_checklist(self, results: dict):
        """打印格式化检查清单 (Ch6 表6-8 Markdown 风格)"""
        print("\n" + "=" * 70)
        print("  防过拟合检查清单 (Overfitting Checklist)")
        print("  方法论: Brian Peterson '四把铲子' 体系 · Ch6 表6-8")
        print("=" * 70)

        header = "| 检查项 | 怎么做 | 好的信号 | 坏的信号 | 本次结果 |"
        sep = "|--------|--------|---------|---------|---------|"

        rows = [
            ('样本外验证', 'OOS数据上跑',
             '衰减小(<30%)', '衰减大(>50%)',
             results.get('check_1_oos', {}).get('result', '[N/A]')),
            ('Walk-forward', '分窗口优化+验证',
             '参数跨窗口稳定', '参数每次都变',
             results.get('check_2_wf', {}).get('result', '[N/A]')),
            ('交叉验证', '多折验证',
             '各折结果一致', '各折差异巨大',
             results.get('check_4_cv', {}).get('result', '[NOT IMPL]')),
            ('规则负担', '从简到繁逐层加',
             'OOS同步改善', 'IS改善但OOS不变',
             results.get('check_3_burden', {}).get('result', '[N/A]')),
            ('参数敏感性', '扫描参数',
             '高原型', '山峰型',
             results.get('check_5_sensitivity', {}).get('result', '[N/A]')),
        ]

        print(header)
        print(sep)
        for name, method, good, bad, result in rows:
            print(f"| {name} | {method} | {good} | {bad} | {result} |")
        print()

    # ================================================================
    # Generate Overall Summary
    # ================================================================
    def generate_summary(self, results: dict) -> dict:
        """生成综合判决

        统计 PASS 项数, 输出四档结论:
          - ALL_PASS:   >=4 项 PASS
          - MOSTLY_PASS: 3 项 PASS
          - NEEDS_WORK:  2 项 PASS
          - HIGH_RISK:   <2 项 PASS
        (NOT_IMPL/SKIP 不计入)

        Returns:
            dict: {verdict, pass_count, total, interpretation, statuses}
        """
        statuses = {
            'check_1_oos': results.get('check_1_oos', {}).get('status', 'N/A'),
            'check_2_wf': results.get('check_2_wf', {}).get('status', 'N/A'),
            'check_3_burden': results.get('check_3_burden', {}).get('status', 'N/A'),
            'check_4_cv': results.get('check_4_cv', {}).get('status', 'NOT_IMPL'),
            'check_5_sensitivity': results.get('check_5_sensitivity', {}).get('status', 'N/A'),
        }

        pass_count = sum(
            1 for s in statuses.values()
            if s not in ('NOT_IMPL', 'SKIP', 'N/A') and s == 'PASS'
        )
        total_executed = sum(
            1 for s in statuses.values()
            if s not in ('NOT_IMPL', 'SKIP', 'N/A')
        )

        if pass_count >= 4:
            verdict = 'ALL_PASS'
            interpretation = '所有检查均通过 — 策略稳健, 可上线实盘'
        elif pass_count >= 3:
            verdict = 'MOSTLY_PASS'
            interpretation = '大部分检查通过 — 建议对警示项进行额外监控'
        elif pass_count >= 2:
            verdict = 'NEEDS_WORK'
            interpretation = '需要改进 — 存在明显过拟合风险'
        else:
            verdict = 'HIGH_RISK'
            interpretation = '高风险 — 策略严重过拟合, 不建议上线'

        label_map = {
            'check_1_oos': '样本外验证',
            'check_2_wf': 'Walk-forward',
            'check_3_burden': '规则负担',
            'check_4_cv': '交叉验证',
            'check_5_sensitivity': '参数敏感性',
        }

        print(f"\n  === 防过拟合检查清单 — 综合判决 ===")
        print(f"  通过: {pass_count}/{total_executed} 项")
        print(f"  判决: {verdict}")
        print(f"  解读: {interpretation}")
        for key, label in label_map.items():
            print(f"    {label}: {statuses.get(key, 'N/A')}")

        return {
            'verdict': verdict,
            'pass_count': pass_count,
            'total': total_executed,
            'interpretation': interpretation,
            'statuses': statuses,
        }


# ================================================================
# 示例运行
# ================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  OverfittingChecklist — 示例运行")
    print("=" * 70)

    checklist = OverfittingChecklist()

    # ---- 模拟 OOS 数据 ----
    oos_dummy = {
        '激进': {'icir_train': 0.45, 'icir_oos': 0.38, 'delta': -0.07, 'robust': 'PASS'},
        '均衡': {'icir_train': 0.40, 'icir_oos': 0.36, 'delta': -0.04, 'robust': 'PASS'},
        '稳健': {'icir_train': 0.35, 'icir_oos': 0.32, 'delta': -0.03, 'robust': 'PASS'},
    }

    # ---- 模拟 WF 数据 ----
    wf_dummy = {
        'anchored': {'stability_score': 75, 'verdict': 'PASS', 'oos_icir_mean': 0.3, 'oos_icir_std': 0.05},
        'rolling': {'stability_score': 68, 'verdict': 'PASS', 'oos_icir_mean': 0.28, 'oos_icir_std': 0.07},
    }

    # ---- 模拟 负担分析 ----
    burden_dummy = pd.DataFrame({
        'n_factors': [1, 2, 3, 4, 5, 6, 7, 8],
        'icir_train': [0.30, 0.38, 0.42, 0.46, 0.47, 0.47, 0.45, 0.43],
        'icir_test': [0.20, 0.30, 0.36, 0.40, 0.40, 0.38, 0.35, 0.32],
        'selected_factors': ['A', 'A,B', 'A,B,C', 'A,B,C,D',
                            'A,B,C,D,E', 'A,B,C,D,E,F', 'A,B,C,D,E,F,G',
                            'A,B,C,D,E,F,G,H'],
    })

    # ---- 模拟 IC 汇总 ----
    ic_summary_dummy = pd.DataFrame({
        'ICIR': [0.45, 0.40, 0.35, 0.30, 0.28],
        'IC_mean': [0.025, 0.022, 0.018, 0.015, 0.012],
        '+IC%': [0.65, 0.62, 0.58, 0.55, 0.53],
    }, index=['momentum', 'volatility', 'value', 'size', 'quality'])

    # ---- 运行所有检查 ----
    results = checklist.run_all(
        oos_results=oos_dummy,
        wf_results=wf_dummy,
        burden_results=burden_dummy,
        ic_summary=ic_summary_dummy,
    )

    # ---- 打印表格 ----
    checklist.print_checklist(results)

    # ---- 综合判决 ----
    summary = checklist.generate_summary(results)
    print(f"\n  判决结果: {summary['verdict']}")
    print(f"  通过: {summary['pass_count']}/{summary['total']}")
    print(f"  说明: {summary['interpretation']}")
