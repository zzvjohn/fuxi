"""
GA超参优化器 (基于 Optuna)
============================
自动搜索 NSGA-II 最优超参，替代手工硬编码配置。

用法:
  from ga_tuner import GAHyperOptimizer
  optimizer = GAHyperOptimizer(n_trials=50, mode="quick")
  best_params = optimizer.optimize()
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class GAHyperOptimizer:
    """NSGA-II 超参优化器"""

    def __init__(self, n_trials: int = 50, mode: str = "quick",
                 storage: str = None, study_name: str = None,
                 chronicle = None):
        """
        Args:
            n_trials: Optuna 试验次数
            mode: 'quick'(轻量GA, ~5min/trial) | 'full'(完整GA, ~90min/trial)
            storage: Optuna存储路径 (默认 sqlite:///outputs/optuna.db)
            study_name: 研究名称
            chronicle: ExperimentChronicle 实例 (自动记录最优)
        """
        if not HAS_OPTUNA:
            raise ImportError("optuna 未安装。pip install optuna")

        self.n_trials = n_trials
        self.mode = mode
        self.chronicle = chronicle

        # Optuna 存储
        if storage is None:
            storage = "sqlite:///" + str(
                Path(__file__).parent.parent / "outputs" / "optuna.db"
            )
        self.storage = storage

        if study_name is None:
            study_name = f"fa_ga_tuning_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.study_name = study_name

        # Phase 1-3 缓存 (避免每个trial重新加载)
        self._cache = {}

    def optimize(self, return_study: bool = False) -> Dict[str, Any]:
        """执行超参优化, 返回最优参数"""
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",
            load_if_exists=True,
        )

        # 如果已有结果, 直接返回最优
        if len(study.trials) >= self.n_trials:
            print(f"[Optuna] 已有 {len(study.trials)} 次试验, 直接返回最优")
            return self._best_params(study)

        print(f"[Optuna] {self.study_name} — {self.n_trials} trials, mode={self.mode}")
        t0 = time.time()

        # 目标函数闭包
        def objective(trial):
            return self._evaluate_trial(trial)

        study.optimize(objective, n_trials=self.n_trials - len(study.trials))

        elapsed = time.time() - t0
        print(f"[Optuna] 优化完成: {elapsed/60:.1f}min, "
              f"最佳值={study.best_value:.4f}")

        best = self._best_params(study)

        # 记录到 Chronicle
        if self.chronicle:
            try:
                self.chronicle.log_methodology_note(
                    category="optuna_tuning",
                    note=(
                        f"Optuna GA 超参调优完成\n"
                        f"  Study: {self.study_name}\n"
                        f"  Trials: {len(study.trials)}\n"
                        f"  Best Value: {study.best_value:.4f}\n"
                        f"  Best Params: {json.dumps(best, ensure_ascii=False)}"
                    ),
                    tags=["optuna", "ga_tuning", "best_params"],
                )
            except Exception:
                pass

        if return_study:
            return best, study
        return best

    def _best_params(self, study) -> dict:
        """提取最优参数"""
        best = study.best_params
        best["best_value"] = study.best_value
        best["n_trials"] = len(study.trials)
        best["study_name"] = self.study_name
        return best

    # ── 搜索空间 ──

    def _suggest_params(self, trial) -> dict:
        """定义超参搜索空间"""
        return {
            "pop_size": trial.suggest_int("pop_size", 20, 120, step=10),
            "n_generations": trial.suggest_int("n_generations", 5, 20),
            "crossover_prob": trial.suggest_float("crossover_prob", 0.6, 0.95, step=0.05),
            "mutation_prob": trial.suggest_float("mutation_prob", 0.05, 0.30, step=0.05),
            "tournament_size": trial.suggest_int("tournament_size", 2, 7),
            "icir_threshold": trial.suggest_float("icir_threshold", 0.3, 1.5, step=0.1),
            "mutation_sigma": trial.suggest_float("mutation_sigma", 0.05, 0.25, step=0.05),
            "elitism_count": trial.suggest_int("elitism_count", 2, 14, step=2),
        }

    # ── 目标函数 ──

    def _evaluate_trial(self, trial) -> float:
        """评估一组超参的质量分数。

        使用 Pareto 前沿的 (Q+S+C)/3 调和均值作为目标,
        并惩罚高换手 (>100% TO)。
        """
        params = self._suggest_params(trial)
        t_start = time.time()

        try:
            if self.mode == "quick":
                score = self._quick_evaluate(params)
            else:
                score = self._full_evaluate(params)

            duration = time.time() - t_start
            trial.set_user_attr("duration_s", duration)

            if self.chronicle:
                try:
                    self.chronicle.log_methodology_note(
                        category="optuna_trial",
                        note=f"Trial {trial.number}: score={score:.4f} "
                             f"params={json.dumps(params, ensure_ascii=False)} "
                             f"duration={duration:.0f}s",
                        tags=["optuna", f"trial_{trial.number}"],
                    )
                except Exception:
                    pass

            return score

        except Exception as e:
            # 失败返回极小值让 Optuna 跳过
            print(f"  [Optuna] Trial {trial.number} FAILED: {e}")
            return -999.0

    def _quick_evaluate(self, params: dict) -> float:
        """轻量评估: 使用预缓存的 Phase 1-3 数据, 只跑 GA"""
        # 提取参数
        pop_size = params["pop_size"]
        n_gen = params["n_generations"]
        crossover_prob = params["crossover_prob"]
        mutation_prob = params["mutation_prob"]
        tournament_size = params["tournament_size"]
        icir_threshold = params["icir_threshold"]
        mutation_sigma = params["mutation_sigma"]
        elitism_count = params["elitism_count"]

        # 加载缓存数据
        if not self._cache:
            self._load_phase1_3_cache()

        if not self._cache:
            return -999.0

        factor_dfs = self._cache["factor_dfs"]
        forward_returns = self._cache["forward_returns"]
        ic_summary = self._cache["ic_summary"]

        # 因子筛选 (可变 ICIR 阈值)
        from ga.factor_screening import screen_factors
        from config import OOS_SPLIT_DATE

        passed, _ = screen_factors(
            factor_dfs, forward_returns, ic_summary,
            icir_min=icir_threshold,
            train_end=OOS_SPLIT_DATE,
        )
        n_factors = len(passed)
        if n_factors < 4:
            return -999.0  # 因子太少无法做GA

        # 构建因子列表
        factor_ids = list(passed.keys())
        n_factors = len(factor_ids)

        # NSGA-II 构建
        from ga.nsga2 import nsga2_run
        from ga.chromosome import random_chromosome

        n_max_factors = min(4, n_factors // 2) if n_factors >= 8 else 3

        # 轻量种群
        pop = [random_chromosome(n_factors, max_factors=n_max_factors)
               for _ in range(pop_size)]

        try:
            _, _, pareto_front = nsga2_run(
                factor_dfs=passed,
                forward_returns=forward_returns,
                population=pop,
                n_generations=n_gen,
                pop_size=pop_size,
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob,
                tournament_size=tournament_size,
                mutation_sigma=mutation_sigma,
                elitism_count=elitism_count,
                max_factors=n_max_factors,
            )
        except Exception:
            return -999.0

        if not pareto_front:
            return -999.0

        # 计算 Pareto 前沿质量分
        return _compute_pareto_score(pareto_front)

    def _full_evaluate(self, params: dict) -> float:
        """完整评估: 完整 FA pipeline (备选, 用于验证)"""
        # 实现较慢的完整评估 — 暂时复用 quick_evaluate
        # (full mode 用在最终验证阶段)
        return self._quick_evaluate(params)

    def _load_phase1_3_cache(self):
        """从最近的 FA 运行加载 Phase 1-3 数据缓存"""
        import pickle
        import pandas as pd
        from pathlib import Path

        output_base = Path(__file__).parent.parent / "output"
        # 找最近的 output 目录
        candidate_dirs = sorted(output_base.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)

        for d in candidate_dirs:
            cache_path = d / "phase3_cache.pkl"
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        self._cache = pickle.load(f)
                    print(f"  [Optuna] 加载 Phase 1-3 缓存: {cache_path}")
                    return
                except Exception:
                    continue

        print("  [Optuna] 未找到 Phase 1-3 缓存, 需先运行一次完整 FA")
        self._cache = {}

    # ── 导出 ──

    def export_config_patch(self, params: dict) -> str:
        """生成可直接粘贴到 config.py 的配置patch"""
        lines = []
        lines.append("# === Optuna 自动优化结果 ===")
        lines.append(f"# Study: {self.study_name}")
        lines.append(f"# Best Value: {params.get('best_value', 'N/A')}")
        lines.append(f"# Trials: {params.get('n_trials', 'N/A')}")
        lines.append("")
        lines.append(f"GA_POPULATION = {params['pop_size']}")
        lines.append(f"GA_GENERATIONS = {params['n_generations']}")
        lines.append(f"GA_CROSSOVER_PROB = {params['crossover_prob']}")
        lines.append(f"GA_MUTATION_PROB = {params['mutation_prob']}")
        lines.append(f"GA_TOURNAMENT_SIZE = {params['tournament_size']}")
        lines.append(f"GA_MUTATION_SIGMA = {params['mutation_sigma']}")
        lines.append(f"GA_ELITISM = {params['elitism_count']}")
        lines.append(f"ICIR_QUALITY_THRESHOLD = {params['icir_threshold']}")
        lines.append("# === 优化结束 ===")
        return "\n".join(lines)

    def save_best_params(self, params: dict, path: str = None):
        """保存最优参数到 JSON"""
        if path is None:
            path = str(
                Path(__file__).parent.parent / "outputs" / "optuna_best.json"
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        print(f"[Optuna] 最优参数已保存: {path}")


def _compute_pareto_score(pareto_front: list) -> float:
    """计算 Pareto 前沿质量分。

    取 top-10 解的 (Q+S+C)/3 调和均值, 惩罚高换手。
    """
    if not pareto_front:
        return -999.0

    # 提取前10个解的三目标值
    solutions = []
    for ps in pareto_front[:10]:
        obj = ps.get("objectives", [0, 0, 0])
        to = ps.get("estimated_turnover", 1.0)
        if isinstance(obj, dict):
            q = obj.get("quality", 0)
            s = obj.get("stability", 0)
            c = obj.get("sharpe", 0)
        elif isinstance(obj, (list, tuple)) and len(obj) >= 3:
            q, s, c = obj[0], obj[1], obj[2]
        else:
            continue

        # 调和均值 (对极端值更敏感)
        hmean = 3.0 / (1.0 / max(q, 0.01) + 1.0 / max(s, 0.01) + 1.0 / max(c, 0.01))

        # 换手惩罚: TO > 100% → 指数衰减
        to_penalty = 1.0 if to < 1.0 else np.exp(-2.0 * (to - 1.0))

        solutions.append(hmean * to_penalty)

    if not solutions:
        return -999.0

    # 取均值 (代表整条前沿质量)
    return float(np.mean(solutions))
