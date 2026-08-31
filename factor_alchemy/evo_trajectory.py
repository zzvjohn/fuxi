# -*- coding: utf-8 -*-
"""
EvolutionTrajectory v0.6 — 连续进化轨迹 (对标 AlphaAgentEvo ICLR 2026)
======================================================================

核心思想 (AlphaAgentEvo 降级移植):
  AlphaAgentEvo 将因子挖掘重新定义为 "learn an evolution policy π"，
  而非 "optimize a single factor"。核心差异在于：
    - 传统: seed → GP breed(单次) → evaluate → best 1
    - EvoTraj: seed → GP mutate → S1 eval → best → GP mutate(motif历史) → ... → S5
                └────────────── T 轮进化轨迹 ──────────────┘
                                ↓
                        轨迹蒸馏 → MAB方向更新 (streak 奖励)

伏羲降级策略 (不需要 LLM + GRPO 训练):
  - 用 GP 确定性变异 + P-001 motif 约束替代 LLM 反思
  - 用 MAB streak-reward 替代 GRPO 策略训练
  - 用 FSA 骨架指纹替代 AST 相似度

核心字段:
  - turns[]: 每轮 (种子因子, 评估结果, 改善幅度 delta)
  - streak: 当前连续改善轮数
  - best_icir: 全程最佳 ICIR
  - trajectory_reward: 复合评分

用法:
    traj = EvolutionTrajectory(seed_factor, max_turns=5)
    for turn in range(max_turns):
        children = gp.breed_from_templates(current, motif_avoid=..., motif_prefer=...)
        results = validator.validate(children)
        best = max(results, key=lambda r: r.icir)
        traj.add_turn(best_factor, best_icir, best_calmar)
        if traj.should_stop():
            break
    print(f"轨迹: {traj.streak}轮连续改善, 最佳ICIR {traj.best_icir:.3f}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class TurnRecord:
    """单轮进化记录"""
    turn_idx: int
    factor_name: str
    formula: str
    icir: float
    calmar: float = 0.0
    n_children: int = 0               # 本轮产生多少候选
    n_pass_gate: int = 0              # 多少通过门禁
    motifs: List[str] = field(default_factory=list)
    delta_from_prev: float = 0.0      # 相对上一轮的 ICIR 改善


@dataclass
class EvolutionTrajectory:
    """
    连续进化轨迹 — 封装一个 seed 因子 T 轮进化的全生命周期。

    对标 AlphaAgentEvo 的 "evolution trajectory τ":
      - 传统 Ralph Loop 等价于 T=1 (每轮独立, 不感知历史)
      - EvoTraj 实现 T>1 (每轮基于前一轮最佳继续打磨)
    """

    # 标识
    traj_id: str = ""
    paradigm: str = ""
    seed_factor_name: str = ""
    seed_formula: str = ""

    # 轨迹参数
    max_turns: int = 5
    current_turn: int = 0

    # 核心数据
    turns: List[TurnRecord] = field(default_factory=list)

    # streak 统计
    streak: int = 0                    # 连续改善轮数
    best_icir: float = -float("inf")
    best_calmar: float = -float("inf")
    best_turn: int = -1
    total_improvement: float = 0.0     # 全程累积改善

    # 终止条件
    dead_streak: int = 0               # 连续无改善轮数

    # 轨迹奖励 (复合)
    trajectory_reward: float = 0.0

    # 时间戳
    started_at: str = ""
    finished_at: str = ""

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now().isoformat()
        if not self.traj_id:
            self.traj_id = f"evotraj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ── 生命周期 ───────────────────────────────────────

    def add_turn(
        self,
        factor_name: str,
        formula: str,
        icir: float,
        calmar: float = 0.0,
        n_children: int = 0,
        n_pass_gate: int = 0,
        motifs: Optional[List[str]] = None,
    ) -> TurnRecord:
        """记录一轮完成"""
        prev_icir = self.turns[-1].icir if self.turns else 0.0
        delta = icir - prev_icir

        record = TurnRecord(
            turn_idx=self.current_turn,
            factor_name=factor_name,
            formula=formula,
            icir=round(icir, 6),
            calmar=round(calmar, 4),
            n_children=n_children,
            n_pass_gate=n_pass_gate,
            motifs=motifs or [],
            delta_from_prev=round(delta, 6),
        )
        self.turns.append(record)
        self.current_turn += 1

        # 更新 streak
        if delta > 0.001:
            self.streak += 1
            self.dead_streak = 0
            self.total_improvement += delta
        else:
            self.streak = 0
            self.dead_streak += 1

        # 更新最佳
        if icir > self.best_icir:
            self.best_icir = icir
            self.best_calmar = calmar
            self.best_turn = self.current_turn - 1

        # 更新轨迹奖励
        self._compute_reward()

        return record

    def should_stop(self) -> bool:
        """判断是否应该停止进化"""
        if self.current_turn >= self.max_turns:
            return True
        # 连续 2 轮无改善 → 停止
        if self.dead_streak >= 2:
            return True
        return False

    def _compute_reward(self):
        """
        计算轨迹奖励 (简化版 AlphaAgentEvo 分层奖励)。

        原始 5 层: tool_use → consistency → exploration → performance → streak
        伏羲简化: streak × ICIR × diversity

        Formula:
          reward = best_icir * (1 + 0.2 * streak) * diversity_bonus
        """
        if not self.turns:
            self.trajectory_reward = 0.0
            return

        # ICIR 归一化到 0~1 (假设 max 2.0)
        icir_norm = min(1.0, max(0.0, self.best_icir / 2.0))

        # streak 奖励: 每轮连续改善 +20%
        streak_bonus = 1.0 + 0.2 * self.streak

        # diversity 奖励: turns 中出现了多少不同的 motif
        all_motifs = set()
        for t in self.turns:
            all_motifs.update(t.motifs)
        diversity_bonus = 1.0 + 0.1 * min(len(all_motifs), 10)

        # 衰减惩罚: 每个 dead turn 扣 10%
        dead_penalty = max(0.3, 1.0 - 0.1 * self.dead_streak)

        self.trajectory_reward = round(
            icir_norm * streak_bonus * diversity_bonus * dead_penalty, 4
        )

    # ── 查询 ────────────────────────────────────────────

    def get_best_factor(self) -> Optional[TurnRecord]:
        """返回最佳轮次的记录"""
        if self.best_turn >= 0 and self.best_turn < len(self.turns):
            return self.turns[self.best_turn]
        return self.turns[-1] if self.turns else None

    def get_current_factor(self) -> Optional[TurnRecord]:
        """返回当前(最后一轮)记录"""
        return self.turns[-1] if self.turns else None

    def get_improvement_curve(self) -> List[float]:
        """返回每轮的 ICIR 序列"""
        return [t.icir for t in self.turns]

    def get_delta_curve(self) -> List[float]:
        """返回每轮的 delta 改善序列"""
        return [t.delta_from_prev for t in self.turns]

    def get_summary(self) -> Dict:
        """轨迹摘要 (供 MAB 奖励更新)"""
        return {
            "traj_id": self.traj_id,
            "paradigm": self.paradigm,
            "seed_factor_name": self.seed_factor_name,
            "turns": self.current_turn,
            "streak": self.streak,
            "dead_streak": self.dead_streak,
            "best_icir": self.best_icir,
            "best_calmar": self.best_calmar,
            "best_turn": self.best_turn,
            "total_improvement": round(self.total_improvement, 6),
            "trajectory_reward": self.trajectory_reward,
            "improvement_curve": self.get_improvement_curve(),
        }

    # ── 序列化 ──────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "traj_id": self.traj_id,
            "paradigm": self.paradigm,
            "seed_factor_name": self.seed_factor_name,
            "seed_formula": self.seed_formula,
            "max_turns": self.max_turns,
            "current_turn": self.current_turn,
            "turns": [
                {
                    "turn_idx": t.turn_idx,
                    "factor_name": t.factor_name,
                    "formula": t.formula,
                    "icir": t.icir,
                    "calmar": t.calmar,
                    "n_children": t.n_children,
                    "n_pass_gate": t.n_pass_gate,
                    "motifs": t.motifs,
                    "delta_from_prev": t.delta_from_prev,
                }
                for t in self.turns
            ],
            "streak": self.streak,
            "dead_streak": self.dead_streak,
            "best_icir": self.best_icir,
            "best_calmar": self.best_calmar,
            "best_turn": self.best_turn,
            "total_improvement": self.total_improvement,
            "trajectory_reward": self.trajectory_reward,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EvolutionTrajectory":
        traj = cls(
            traj_id=data.get("traj_id", ""),
            paradigm=data.get("paradigm", ""),
            seed_factor_name=data.get("seed_factor_name", ""),
            seed_formula=data.get("seed_formula", ""),
            max_turns=data.get("max_turns", 5),
        )
        traj.current_turn = data.get("current_turn", 0)
        traj.streak = data.get("streak", 0)
        traj.dead_streak = data.get("dead_streak", 0)
        traj.best_icir = data.get("best_icir", -float("inf"))
        traj.best_calmar = data.get("best_calmar", -float("inf"))
        traj.best_turn = data.get("best_turn", -1)
        traj.total_improvement = data.get("total_improvement", 0.0)
        traj.trajectory_reward = data.get("trajectory_reward", 0.0)
        traj.started_at = data.get("started_at", "")
        traj.finished_at = data.get("finished_at", "")

        for t_data in data.get("turns", []):
            record = TurnRecord(
                turn_idx=t_data["turn_idx"],
                factor_name=t_data["factor_name"],
                formula=t_data["formula"],
                icir=t_data["icir"],
                calmar=t_data.get("calmar", 0.0),
                n_children=t_data.get("n_children", 0),
                n_pass_gate=t_data.get("n_pass_gate", 0),
                motifs=t_data.get("motifs", []),
                delta_from_prev=t_data.get("delta_from_prev", 0.0),
            )
            traj.turns.append(record)

        return traj

    def finish(self):
        """标记轨迹完成"""
        self.finished_at = datetime.now().isoformat()
        self._compute_reward()

    def __repr__(self):
        best = self.get_best_factor()
        best_str = f"ICIR={best.icir:.3f}" if best else "N/A"
        return (
            f"EvolutionTrajectory({self.traj_id}, "
            f"paradigm={self.paradigm}, "
            f"turns={self.current_turn}, "
            f"streak={self.streak}, "
            f"best={best_str}, "
            f"reward={self.trajectory_reward:.3f})"
        )


# ═══════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════

def create_trajectory(
    seed_factor_name: str = "",
    seed_formula: str = "",
    paradigm: str = "",
    max_turns: int = 5,
) -> EvolutionTrajectory:
    """快速创建进化轨迹"""
    return EvolutionTrajectory(
        seed_factor_name=seed_factor_name,
        seed_formula=seed_formula,
        paradigm=paradigm,
        max_turns=max_turns,
    )
