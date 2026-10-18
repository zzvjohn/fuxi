"""
Pydantic 数据模型 — 实验编年史所有实体的类型定义。
用于验证写入数据、序列化 JSON 字段、生成 SQL 列映射。
"""
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict


@dataclass
class ExperimentConfig:
    """一次实验的完整配置快照"""
    experiment_id: str
    version: str
    description: str = ""
    start_date: str = "2021-01-01"
    end_date: str = "2026-06-25"
    frequency: str = "W"
    ga_pop_size: int = 80
    ga_generations: int = 15
    ga_mutation_rate: float = 0.2
    ga_mutation_sigma: float = 0.1
    ga_workers: int = 8
    wf_workers: int = 8
    n_factors_input: int = 0           # 预筛选前因子数
    n_factors_screened: int = 0        # 预筛选后因子数
    max_factors: int = 4
    icir_threshold: float = 0.1
    wfa_pass_threshold: int = 3
    oos_split_date: str = "2025-01-01" # v7.9: 仅标签, GA不切分
    init_mode: str = "icir_biased"
    icir_mode: str = "clip"
    training_mode: str = "full_sample"  # v7.9: 全样本训练


@dataclass
class FactorPerformance:
    """单因子在本次实验中的性能"""
    experiment_id: str
    factor_id: str
    category: str = ""
    icir_mean: float = 0.0
    icir_std: float = 0.0
    ic_mean: float = 0.0
    rank_autocorr: float = 0.0
    top_decile_retention: float = 0.0
    cv_ic: float = 0.0
    ic_half_life: float = 0.0
    estimated_turnover: float = 0.0
    decile_monotonicity: float = 0.0
    wfa_survival_rate: float = 0.0
    regime_bull_icir: Optional[float] = None
    regime_bear_icir: Optional[float] = None
    regime_volatile_icir: Optional[float] = None
    regime_calm_icir: Optional[float] = None
    passed_screening: bool = False     # 是否通过两阶段预筛选
    status: str = "active"             # active|fail|wfa_rejected


@dataclass
class StrategyResult:
    """策略评估结果 (Phase 4.5 四策略 + Phase 5 组合)"""
    experiment_id: str
    strategy_label: str                # high_quality|high_stable|high_sharpe|balanced
    factor_weights: Dict[str, float] = field(default_factory=dict)
    quality_obj: float = 0.0
    stability_obj: float = 0.0
    sharpe_obj: float = 0.0
    icir_train: float = 0.0
    estimated_turnover: float = 0.0
    turnover_penalty: float = 0.0
    passed_dimensions: int = 0         # 五维通过数
    phase5_total_return: Optional[float] = None
    phase5_cagr: Optional[float] = None
    phase5_sharpe: Optional[float] = None
    phase5_max_dd: Optional[float] = None
    phase5_avg_turnover: Optional[float] = None
    phase55_icir: Optional[float] = None   # 全样本复核 ICIR
    phase56_sharpe_2026: Optional[float] = None  # 2026 时效性
    oos_icir_mean: Optional[float] = None      # Phase 7 WF OOS ICIR
    wf_anchored_score: Optional[float] = None
    wf_rolling_score: Optional[float] = None
    overfitting_risk: str = "UNKNOWN"   # LOW|MEDIUM|HIGH|UNKNOWN


@dataclass
class GARunLog:
    """NSGA-II 运行日志"""
    experiment_id: str
    pop_size: int
    generations: int
    n_factors: int
    final_pareto_size: int = 0
    pareto_q_range: str = ""            # "[min, max]"
    pareto_s_range: str = ""
    pareto_c_range: str = ""
    convergence_q: List[float] = field(default_factory=list)  # 每代最优 Obj1
    convergence_s: List[float] = field(default_factory=list)  # 每代最优 Obj2
    convergence_c: List[float] = field(default_factory=list)  # 每代最优 Obj3
    ga_duration_s: float = 0.0


@dataclass
class MethodologyUpdate:
    """Meta Learner 方法论更新记录"""
    experiment_id: str
    update_type: str                    # threshold|operator_weight|ga_param|rule|structural
    old_value: str
    new_value: str
    reasoning: str
    source_module: str = ""             # 哪个Meta模块发起的(bayesian_opt|symbolic_reg|agent_reflect)
    validated: bool = False
    validated_by: str = ""              # 验证此更新的实验ID
    created_at: str = ""
