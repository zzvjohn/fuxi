# -*- coding: utf-8 -*-
"""
Trajectory Logger — 对标 QuantaAlpha 的轨迹级进化日志
==========================================================

将每次因子挖掘过程分解为结构化子步骤:
  hypothesis → factor_expression → code → backtest_result

每个子步骤独立评分，支持:
  - Targeted Mutation: 定位到次优步骤做针对性修订
  - Crossover: 重组不同轨迹的互补高奖励段
  - 进化溯源: 追踪因子从初始假设到最终表现的完整链路

设计原则 (对标 QuantaAlpha Section 4.2):
  - 每条轨迹包含完整的 {hypothesis, expression, code, backtest, scores}
  - 子步骤评分让进化从 "盲猜" 变为 "靶向治疗"
  - 轨迹历史供 QuantaAlpha 风格的 trajectory Crossover 使用

文件: data/trajectory_log.json

用法:
    from trajectory_logger import TrajectoryLogger

    logger = TrajectoryLogger()

    # 开始一条新轨迹
    traj = logger.start_trajectory(paradigm="动量反转", seed_factor="alpha_v3_xxx")

    # 记录各阶段
    logger.log_hypothesis(traj.trajectory_id, "价格momentum + 成交量确认...")
    logger.log_expression(traj.trajectory_id, "ts_delta(close,20)/ts_std(close,20)", complexity=2)
    logger.log_code(traj.trajectory_id, "def factor_func(df):...", compilation_success=True)
    logger.log_backtest(traj.trajectory_id, ic=0.045, icir=0.52, sharpe=1.2)

    # 提取模式
    traj = logger.get_trajectory(traj.trajectory_id)
    weak_points = traj.get_weakest_step()  # → "hypothesis"
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ── 默认路径 ──────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent.parent / "data"
TRAJECTORY_PATH = DATA_DIR / "trajectory_log.json"


@dataclass
class TrajectoryStep:
    """单个子步骤"""
    step_name: str              # "hypothesis" / "expression" / "code" / "backtest"
    content: str = ""
    score: float = 0.0          # 0-1 归一化评分
    meta: Dict = field(default_factory=dict)  # 额外元数据
    timestamp: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        _content = self.content
        if not isinstance(_content, str):
            try:
                _content = json.dumps(_content, ensure_ascii=False, default=str)
            except Exception:
                _content = str(_content)
        return {
            "step_name": self.step_name,
            "content": _content[:500],
            "score": self.score,
            "meta": self.meta,
            "timestamp": self.timestamp,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryStep":
        return cls(
            step_name=d.get("step_name", ""),
            content=d.get("content", ""),
            score=d.get("score", 0.0),
            meta=d.get("meta", {}),
            timestamp=d.get("timestamp", ""),
            errors=d.get("errors", []),
            warnings=d.get("warnings", []),
        )


@dataclass
class MiningTrajectory:
    """一条完整的因子挖掘轨迹 — 对标 QuantaAlpha 的 trajectory 概念"""
    trajectory_id: str
    paradigm: str = ""
    seed_factor: str = ""       # 起始因子或父轨迹ID
    generation: int = 0          # 进化代数
    phase: str = ""             # "mutation" / "crossover" / "initial"

    # 子步骤
    hypothesis: Optional[TrajectoryStep] = None
    expression: Optional[TrajectoryStep] = None
    code: Optional[TrajectoryStep] = None
    backtest: Optional[TrajectoryStep] = None

    # 整体评估
    overall_score: float = 0.0
    final_ic: float = 0.0
    final_icir: float = 0.0
    final_outcome: str = ""     # "PASS" / "REJECT" / "WEAK"
    jq_validated: bool = False
    jq_return: Optional[float] = None
    jq_sharpe: Optional[float] = None

    # v0.7 频率对称: S1 裁决口径标签 ("daily" | "weekly"), D+ 蒸馏按频率归因
    natural_freq: str = "daily"

    # L1 解读层归因报告 (2026-08-14, 软字段: 不参与评分)
    interpretation: Dict = field(default_factory=dict)

    # 时间戳
    created_at: str = ""
    completed_at: str = ""

    # 用于 Crossover 的高奖励段标记
    high_reward_segments: List[str] = field(default_factory=list)

    def get_weakest_step(self) -> str:
        """返回评分最低的子步骤 — 用于 Targeted Mutation"""
        steps = [
            ("hypothesis", self.hypothesis),
            ("expression", self.expression),
            ("code", self.code),
            ("backtest", self.backtest),
        ]
        scored = [(name, s.score) for name, s in steps if s is not None]
        if not scored:
            return "unknown"
        return min(scored, key=lambda x: x[1])[0]

    def get_strongest_step(self) -> str:
        """返回评分最高的子步骤 — 用于 Crossover 的贡献段"""
        steps = [
            ("hypothesis", self.hypothesis),
            ("expression", self.expression),
            ("code", self.code),
            ("backtest", self.backtest),
        ]
        scored = [(name, s.score) for name, s in steps if s is not None]
        if not scored:
            return "unknown"
        return max(scored, key=lambda x: x[1])[0]

    def is_complete(self) -> bool:
        """轨迹是否已完成（finalize 已调用）"""
        return bool(self.completed_at)

    def has_all_steps(self) -> bool:
        """所有子步骤是否已记录（不要求 code，因部分管线不产生 code）"""
        return all([
            self.hypothesis is not None,
            self.expression is not None,
            self.backtest is not None,
        ])

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "paradigm": self.paradigm,
            "seed_factor": self.seed_factor,
            "generation": self.generation,
            "phase": self.phase,
            "hypothesis": self.hypothesis.to_dict() if self.hypothesis else None,
            "expression": self.expression.to_dict() if self.expression else None,
            "code": self.code.to_dict() if self.code else None,
            "backtest": self.backtest.to_dict() if self.backtest else None,
            "overall_score": self.overall_score,
            "final_ic": self.final_ic,
            "final_icir": self.final_icir,
            "final_outcome": self.final_outcome,
            "jq_validated": self.jq_validated,
            "jq_return": self.jq_return,
            "jq_sharpe": self.jq_sharpe,
            "natural_freq": self.natural_freq,  # v0.7 频率对称
            "interpretation": self.interpretation,  # L1 (2026-08-14)
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "high_reward_segments": self.high_reward_segments,
            "_eval_log": getattr(self, '_eval_log', []) or [],  # v0.7: 持久化评估日志
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MiningTrajectory":
        traj = cls(
            trajectory_id=d.get("trajectory_id", ""),
            paradigm=d.get("paradigm", ""),
            seed_factor=d.get("seed_factor", ""),
            generation=d.get("generation", 0),
            phase=d.get("phase", ""),
            overall_score=d.get("overall_score", 0.0),
            final_ic=d.get("final_ic", 0.0),
            final_icir=d.get("final_icir", 0.0),
            final_outcome=d.get("final_outcome", ""),
            jq_validated=d.get("jq_validated", False),
            jq_return=d.get("jq_return"),
            jq_sharpe=d.get("jq_sharpe"),
            natural_freq=d.get("natural_freq", "daily"),  # v0.7 频率对称
            interpretation=d.get("interpretation", {}) or {},  # L1 (2026-08-14)
            created_at=d.get("created_at", ""),
            completed_at=d.get("completed_at", ""),
            high_reward_segments=d.get("high_reward_segments", []),
        )
        if d.get("hypothesis"):
            traj.hypothesis = TrajectoryStep.from_dict(d["hypothesis"]) if isinstance(d["hypothesis"], dict) else TrajectoryStep(step_name="hypothesis", content=str(d["hypothesis"]))
        if d.get("expression"):
            traj.expression = TrajectoryStep.from_dict(d["expression"]) if isinstance(d["expression"], dict) else TrajectoryStep(step_name="expression", content=str(d["expression"]))
        if d.get("code"):
            traj.code = TrajectoryStep.from_dict(d["code"]) if isinstance(d["code"], dict) else TrajectoryStep(step_name="code", content=str(d["code"]))
        if d.get("backtest"):
            traj.backtest = TrajectoryStep.from_dict(d["backtest"]) if isinstance(d["backtest"], dict) else TrajectoryStep(step_name="backtest", content=str(d["backtest"]))
        traj._eval_log = d.get("_eval_log", []) or []  # v0.7: 恢复评估日志
        return traj


def _json_safe_default(o):
    """JSON 序列化兜底: numpy 标量 (float32 等) → Python 原生类型。

    8-21 S5 float32 修复后, 评估指标 (ic/icir/excess/calmar) 可能为 np.float32,
    json.dump 默认无法序列化 → 2026-08-22 Ralph 主轮在此崩溃。此处全局兜底。
    """
    if hasattr(o, 'item'):
        try:
            v = o.item()
            if isinstance(v, (int, float, bool, str)) or v is None:
                return v
        except (ValueError, TypeError):
            pass
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')


class TrajectoryLogger:
    """轨迹日志管理器 — 对标 QuantaAlpha 的 trajectory logging"""

    def __init__(self, log_path: Path = TRAJECTORY_PATH):
        self.path = log_path
        self.trajectories: Dict[str, MiningTrajectory] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for t_data in data.get("trajectories", []):
                traj = MiningTrajectory.from_dict(t_data)
                self.trajectories[traj.trajectory_id] = traj

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0.0",
            "total_trajectories": len(self.trajectories),
            "updated_at": datetime.now().isoformat(),
            "trajectories": [t.to_dict() for t in self.trajectories.values()],
        }
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_json_safe_default)

    # ── 轨迹生命周期 ─────────────────────────────────────

    def start_trajectory(
        self,
        paradigm: str = "",
        seed_factor: str = "",
        generation: int = 0,
        phase: str = "initial",
        natural_freq: str = "daily",  # v0.7 频率对称
    ) -> MiningTrajectory:
        """开始一条新的挖掘轨迹"""
        traj_id = f"traj_{uuid.uuid4().hex[:12]}"
        traj = MiningTrajectory(
            trajectory_id=traj_id,
            paradigm=paradigm,
            seed_factor=seed_factor,
            generation=generation,
            phase=phase,
            natural_freq=natural_freq,
            created_at=datetime.now().isoformat(),
        )
        self.trajectories[traj_id] = traj
        self._save()
        return traj

    def get_trajectory(self, traj_id: str) -> Optional[MiningTrajectory]:
        return self.trajectories.get(traj_id)

    # ── 子步骤记录 ───────────────────────────────────────

    def log_hypothesis(
        self, traj_id: str, content: str, score: float = 0.0,
        novelty: float = 0.0, economic_logic: float = 0.0,
    ):
        """记录 hypothesis 步骤"""
        traj = self.trajectories.get(traj_id)
        if not traj:
            return
        traj.hypothesis = TrajectoryStep(
            step_name="hypothesis",
            content=content,
            score=score,
            meta={"novelty": novelty, "economic_logic": economic_logic},
            timestamp=datetime.now().isoformat(),
        )
        self._update_overall(traj)
        self._save()

    def log_expression(
        self, traj_id: str, expression: str, score: float = 0.0,
        complexity: int = 0, node_count: int = 0, ast_depth: int = 0,
        motifs: Optional[List[str]] = None,  # v0.5 P-001: motif 标注
    ):
        """记录 factor expression 步骤"""
        traj = self.trajectories.get(traj_id)
        if not traj:
            return
        meta = {
            "complexity": complexity,
            "node_count": node_count,
            "ast_depth": ast_depth,
        }
        if motifs:
            meta["motifs"] = motifs  # v0.5: 子结构模式列表
        traj.expression = TrajectoryStep(
            step_name="expression",
            content=expression,
            score=score,
            meta=meta,
            timestamp=datetime.now().isoformat(),
        )
        self._update_overall(traj)
        self._save()

    def log_code(
        self, traj_id: str, code: str, score: float = 0.0,
        compilation_success: bool = False,
        errors: Optional[List[str]] = None,
    ):
        """记录 code 实现步骤"""
        traj = self.trajectories.get(traj_id)
        if not traj:
            return
        traj.code = TrajectoryStep(
            step_name="code",
            content=code,
            score=score,
            meta={"compilation_success": compilation_success},
            errors=errors or [],
            timestamp=datetime.now().isoformat(),
        )
        self._update_overall(traj)
        self._save()

    def log_backtest(
        self, traj_id: str, score: float = 0.0,
        ic: float = 0.0, icir: float = 0.0,
        sharpe: float = 0.0, maxdd: float = 0.0,
        warnings: Optional[List[str]] = None,
    ):
        """记录 backtest 结果"""
        traj = self.trajectories.get(traj_id)
        if not traj:
            return
        traj.backtest = TrajectoryStep(
            step_name="backtest",
            content=f"IC={ic:.4f}, ICIR={icir:.3f}, Sharpe={sharpe:.2f}",
            score=score,
            meta={"ic": ic, "icir": icir, "sharpe": sharpe, "maxdd": maxdd},
            warnings=warnings or [],
            timestamp=datetime.now().isoformat(),
        )
        traj.final_ic = ic
        traj.final_icir = icir
        self._update_overall(traj)
        self._save()

    # ── v0.7 Phase 2: 结构化评估日志 ──────────────────────

    def log_evaluation(
        self, traj_id: str,
        factor_name: str = "", formula: str = "", paradigm: str = "",
        stage: str = "",             # "S1" | "S2" | ... | "S5" | "S5_PASS"
        passed: bool = False,
        ic: Optional[float] = None, icir: Optional[float] = None,
        excess_25: Optional[float] = None, excess_26: Optional[float] = None,
        calmar_25: Optional[float] = None, calmar_26: Optional[float] = None,
        rejection_reason: str = "",
        formula_complexity: int = 0,
        warnings: Optional[List[str]] = None,
    ):
        """
        v0.7: 结构化记录单候选的全管线评估结果。

        每条候选独立记录 → analyze_failures() 可做统计蒸馏。
        """
        traj = self.trajectories.get(traj_id)
        if not traj:
            return

        # 将评估数据存入 meta
        if not hasattr(traj, '_eval_log') or traj._eval_log is None:
            traj._eval_log = []
        traj._eval_log.append({
            "factor_name": factor_name,
            "formula": formula[:200],
            "paradigm": paradigm,
            "stage": stage,
            "passed": passed,
            "ic": ic,
            "icir": icir,
            "excess_25": excess_25,
            "excess_26": excess_26,
            "calmar_25": calmar_25,
            "calmar_26": calmar_26,
            "rejection_reason": rejection_reason[:120],
            "formula_complexity": formula_complexity,
            "timestamp": datetime.now().isoformat(),
        })

        # 同时更新 backtest 中的 S5 数据
        if stage.startswith("S5") or stage == "S5_PASS":
            meta = {"ic": ic, "icir": icir,
                    "excess_25": excess_25, "excess_26": excess_26,
                    "calmar_25": calmar_25, "calmar_26": calmar_26,
                    "passed": passed, "rejection_reason": rejection_reason}
            traj.backtest = TrajectoryStep(
                step_name="backtest",
                content=f"IC={ic:.4f}, ICIR={icir:.3f}, ex25={excess_25}, ex26={excess_26}" if ic is not None else "no_data",
                score=0.7 if passed else 0.3,
                meta=meta,
                warnings=warnings or [],
                timestamp=datetime.now().isoformat(),
            )
            traj.final_ic = ic or 0.0
            traj.final_icir = icir or 0.0

        self._save()

    # ═══════════════════════════════════════════════════════════
    # v0.7 Phase 2: 自动失败分析 + 蒸馏提示
    # ═══════════════════════════════════════════════════════════

    def analyze_failures(self, recent_n: int = 200) -> Dict:
        """
        自动分析最近 N 条评估记录的失败模式。

        Returns
        -------
        {
            "paradigm_stats": {paradigm: {mean_ic, mean_icir, pass_rate, n}},
            "rejection_dist": {reason: count},
            "complexity_vs_ic": [(complexity, mean_ic), ...],
            "top_paradigms": [(paradigm, mean_ic), ...],
            "bottleneck_stage": "S1" | "S5" | ...,
            "summary": "一句话总结"
        }
        """
        import numpy as np

        # 收集所有 eval_log
        all_evals = []
        for traj in self.trajectories.values():
            log = getattr(traj, '_eval_log', None) or []
            all_evals.extend(log)

        if not all_evals:
            return {"summary": "无评估数据, 先跑几轮 Ralph Loop"}

        # 只取最近 N 条
        all_evals = all_evals[-recent_n:]

        # ── 按范式统计 ──
        paradigm_data = {}
        for ev in all_evals:
            p = ev.get("paradigm", "") or "unknown"
            if p not in paradigm_data:
                paradigm_data[p] = {"ic_list": [], "icir_list": [], "pass": 0, "total": 0}
            d = paradigm_data[p]
            if ev.get("ic") is not None:
                d["ic_list"].append(ev["ic"])
            if ev.get("icir") is not None:
                d["icir_list"].append(ev["icir"])
            if ev.get("passed"):
                d["pass"] += 1
            d["total"] += 1

        paradigm_stats = {}
        for p, d in paradigm_data.items():
            ic_arr = np.array(d["ic_list"]) if d["ic_list"] else np.array([0.0])
            icir_arr = np.array(d["icir_list"]) if d["icir_list"] else np.array([0.0])
            paradigm_stats[p] = {
                "mean_ic": round(float(np.mean(ic_arr)), 5),
                "max_ic": round(float(np.max(ic_arr)), 5),
                "mean_icir": round(float(np.mean(icir_arr)), 4),
                "pass_rate": round(d["pass"] / max(d["total"], 1), 3),
                "n": d["total"],
            }

        # ── 拒绝原因分布 ──
        rejection_dist = {}
        for ev in all_evals:
            reason = ev.get("rejection_reason", "") or "未记录"
            # 归类合并
            if "过度简化" in reason:
                key = "过度简化(ops<2)"
            elif "公式合理性" in reason:
                key = "公式合理性拦截"
            elif "IC=" in reason or "ICIR=" in reason:
                key = "S1: IC/ICIR不足"
            elif "excess" in reason.lower() or "calmar" in reason.lower():
                key = "S5: 回测指标不达标"
            elif reason == "未记录":
                key = "未记录"
            else:
                key = reason[:40]
            rejection_dist[key] = rejection_dist.get(key, 0) + 1

        # ── 复杂度 vs IC ──
        comp_pairs = []
        for ev in all_evals:
            if ev.get("ic") is not None and ev.get("formula_complexity", 0) > 0:
                comp_pairs.append((ev["formula_complexity"], ev["ic"]))
        # 按复杂度分组
        comp_vs_ic = {}
        for c, ic in comp_pairs:
            bucket = (c // 2) * 2  # 2个一桶
            if bucket not in comp_vs_ic:
                comp_vs_ic[bucket] = []
            comp_vs_ic[bucket].append(ic)
        comp_vs_ic = {k: round(float(np.mean(v)), 5) for k, v in sorted(comp_vs_ic.items())}

        # ── Top 范式 ──
        top_paradigms = sorted(
            [(p, s["mean_ic"]) for p, s in paradigm_stats.items() if s["n"] >= 2],
            key=lambda x: -x[1]
        )[:5]

        # ── 瓶颈定位 ──
        stages = {}
        for ev in all_evals:
            st = ev.get("stage", "?")
            stages[st] = stages.get(st, 0) + 1
        bottleneck = max(stages, key=stages.get) if stages else "?"

        # ── 总结 ──
        total_pass = sum(1 for ev in all_evals if ev.get("passed"))
        total = len(all_evals)
        summary = (
            f"最近{total}候选: 通过率{total_pass/total:.1%}, "
            f"瓶颈在{bottleneck}, "
            f"Top范式={top_paradigms[0][0] if top_paradigms else 'N/A'}(IC={top_paradigms[0][1]:.3f})" 
            if top_paradigms else f"最近{total}候选: 通过率{total_pass/total:.1%}, 瓶颈={bottleneck}"
        )

        return {
            "paradigm_stats": paradigm_stats,
            "rejection_dist": rejection_dist,
            "complexity_vs_ic": comp_vs_ic,
            "top_paradigms": top_paradigms,
            "bottleneck_stage": bottleneck,
            "summary": summary,
            "total_analyzed": total,
        }

    def get_distillation_hints(self) -> Dict:
        """
        v0.7: 从轨迹分析中提取可操作的蒸馏提示。

        这些提示直接反馈到:
        - MAB 方向选择 (prefer_paradigms, avoid_paradigms)
        - GP 参数 (target_complexity_range)
        - S5 阈值调整建议
        """
        analysis = self.analyze_failures(recent_n=200)

        hints = {
            "prefer_paradigms": [],
            "avoid_paradigms": [],
            "target_complexity_range": None,
            "bottleneck": analysis.get("bottleneck_stage", "?"),
            "suggestion": "",
        }

        # 范式偏好: IC > 0.01 且 样本 >= 2
        for p, s in analysis.get("paradigm_stats", {}).items():
            if s["n"] >= 2 and s["mean_ic"] > 0.01:
                hints["prefer_paradigms"].append((p, s["mean_ic"]))
            elif s["n"] >= 3 and s["mean_ic"] < -0.01:
                hints["avoid_paradigms"].append(p)

        hints["prefer_paradigms"].sort(key=lambda x: -x[1])

        # 复杂度范围: 找 IC 最高的复杂度桶
        comp_map = analysis.get("complexity_vs_ic", {})
        if comp_map:
            best_comp = max(comp_map, key=comp_map.get)
            hints["target_complexity_range"] = (best_comp, best_comp + 2)

        # 生成建议
        bottleneck = hints["bottleneck"]
        top_para = hints["prefer_paradigms"][0][0] if hints["prefer_paradigms"] else "?"
        if bottleneck == "S1" or bottleneck.startswith("S1"):
            hints["suggestion"] = (
                f"瓶颈在S1(IC不足): {top_para}范式IC最高, "
                f"建议MAB优先选择{top_para}方向; 增加公式复杂度至≥{best_comp}次ops"
                if comp_map else
                f"瓶颈在S1(IC不足): {top_para}范式IC最高, 建议MAB优先选择{top_para}方向"
            )
        elif bottleneck == "S5" or bottleneck.startswith("S5"):
            hints["suggestion"] = (
                f"瓶颈在S5(回测): IC尚可但excess/calmar不达标, "
                f"需提升因子稳健性, 建议跨范式crossover"
            )
        else:
            hints["suggestion"] = f"瓶颈在{bottleneck}, S5通过率偏低, 继续扩大多样性"

        return hints

    def log_lesson(self, traj_id: str, lesson: str):
        """记录轨迹中的经验教训（JQ验证后附加）"""
        traj = self.trajectories.get(traj_id)
        if not traj:
            return
        if not hasattr(traj, 'lessons') or traj.lessons is None:
            traj.lessons = []
        traj.lessons.append({
            "lesson": lesson,
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def finalize(
        self, traj_id: str, outcome: str,
        jq_return: Optional[float] = None,
        jq_sharpe: Optional[float] = None,
    ):
        """标记轨迹完成"""
        traj = self.trajectories.get(traj_id)
        if not traj:
            return
        traj.final_outcome = outcome
        traj.completed_at = datetime.now().isoformat()
        if jq_return is not None:
            traj.jq_validated = True
            traj.jq_return = jq_return
            traj.jq_sharpe = jq_sharpe

        # 标记高奖励段
        for step_name in ["hypothesis", "expression", "code", "backtest"]:
            step = getattr(traj, step_name)
            if step and step.score >= 0.7:
                traj.high_reward_segments.append(step_name)

        self._save()

    def _update_overall(self, traj: MiningTrajectory):
        """更新整体评分"""
        scores = []
        for step in [traj.hypothesis, traj.expression, traj.code, traj.backtest]:
            if step is not None:
                scores.append(step.score)
        if scores:
            traj.overall_score = sum(scores) / len(scores)

    # ── 查询 ─────────────────────────────────────────────

    def get_by_paradigm(self, paradigm: str, n: int = 10) -> List[MiningTrajectory]:
        """按范式获取轨迹"""
        matched = [t for t in self.trajectories.values() if t.paradigm == paradigm]
        matched.sort(key=lambda t: t.final_icir, reverse=True)
        return matched[:n]

    def get_high_reward_trajectories(self, min_score: float = 0.6, n: int = 10) -> List[MiningTrajectory]:
        """获取高奖励轨迹 — 用于 Crossover 的父轨迹选择"""
        matched = [t for t in self.trajectories.values()
                   if t.overall_score >= min_score and t.is_complete()]
        matched.sort(key=lambda t: t.overall_score, reverse=True)
        return matched[:n]

    def get_weakest_steps_summary(self, paradigm: str = "") -> Dict[str, int]:
        """统计各范式中最常见的弱步骤"""
        trajs = self.trajectories.values()
        if paradigm:
            trajs = [t for t in trajs if t.paradigm == paradigm]

        counter = {}
        for t in trajs:
            weak = t.get_weakest_step()
            counter[weak] = counter.get(weak, 0) + 1
        return counter

    def get_summary(self) -> Dict:
        """获取日志摘要"""
        total = len(self.trajectories)
        completed = sum(1 for t in self.trajectories.values() if bool(t.completed_at))
        passed = sum(1 for t in self.trajectories.values() if t.final_outcome == "PASS")
        jq_validated = sum(1 for t in self.trajectories.values() if t.jq_validated)
        jq_passed = sum(1 for t in self.trajectories.values()
                        if t.jq_validated and t.final_outcome in ("PASS", "JQ_PASSED"))

        # 平均子步骤评分
        h_scores = [t.hypothesis.score for t in self.trajectories.values() if t.hypothesis]
        e_scores = [t.expression.score for t in self.trajectories.values() if t.expression]
        c_scores = [t.code.score for t in self.trajectories.values() if t.code]
        b_scores = [t.backtest.score for t in self.trajectories.values() if t.backtest]

        return {
            "total": total,
            "completed": completed,
            "passed": passed,
            "jq_validated": jq_validated,
            "jq_passed": jq_passed,
            "avg_scores": {
                "hypothesis": sum(h_scores) / len(h_scores) if h_scores else 0,
                "expression": sum(e_scores) / len(e_scores) if e_scores else 0,
                "code": sum(c_scores) / len(c_scores) if c_scores else 0,
                "backtest": sum(b_scores) / len(b_scores) if b_scores else 0,
            },
        }


# ── 便捷函数 ──────────────────────────────────────────────

_default_logger: Optional[TrajectoryLogger] = None


def get_logger() -> TrajectoryLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = TrajectoryLogger()
    return _default_logger


# ── 测试 ──────────────────────────────────────────────────

if __name__ == "__main__":
    logger = TrajectoryLogger()

    traj = logger.start_trajectory(paradigm="动量反转", seed_factor="momentum_v1")

    logger.log_hypothesis(traj.trajectory_id,
        "价格momentum在特定成交量确认下持续有效",
        score=0.75, novelty=0.6, economic_logic=0.8)

    logger.log_expression(traj.trajectory_id,
        "ts_delta(close, 20) / ts_std(close, 20)",
        score=0.80, complexity=2, node_count=5, ast_depth=3)

    logger.log_code(traj.trajectory_id,
        "def factor(df): return df['close'].diff(20) / df['close'].rolling(20).std()",
        score=0.90, compilation_success=True)

    logger.log_backtest(traj.trajectory_id,
        score=0.70, ic=0.045, icir=0.52, sharpe=1.2, maxdd=-0.15)

    logger.finalize(traj.trajectory_id, outcome="PASS")

    print(f"轨迹 {traj.trajectory_id}:")
    print(f"  最弱步骤: {traj.get_weakest_step()}")
    print(f"  最强步骤: {traj.get_strongest_step()}")
    print(f"  高奖励段: {traj.high_reward_segments}")
    print(f"  摘要: {logger.get_summary()}")
