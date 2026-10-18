"""
Chronicle -- 实验编年史
======================
SQLite 结构化实验数据库。
替代散落 .log 文件，为 Meta Learner 提供训练数据。

用法:
    from chronicle import ExperimentChronicle
    chronicle = ExperimentChronicle("outputs/chronicle.db")
    chronicle.start_experiment("fa_v7.9_20260714", {"version": "v7.9", ...})
    chronicle.log_factor_performance(...)
    chronicle.log_strategy_result(...)
    chronicle.complete_experiment(...)
"""

from chronicle.db import ExperimentChronicle
from chronicle.models import (
    ExperimentConfig, FactorPerformance, StrategyResult,
    GARunLog, MethodologyUpdate
)

__all__ = [
    "ExperimentChronicle",
    "ExperimentConfig", "FactorPerformance", "StrategyResult",
    "GARunLog", "MethodologyUpdate",
]
