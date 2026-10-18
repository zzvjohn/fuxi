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
import re
from pathlib import Path


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
    # ── S2-c i (2026-09-12): 步骤级归因原料 ──
    n_fsa: int = 0                    # FSA 骨架过滤后剩多少
    n_eval_ok: int = 0                # 成功求值出 IC 的条数
    n_eval_failed: int = 0            # 求值失败条数 (代码段问题)
    degraded_phase: str = ""          # 本轮劣化归因到的阶段 ("" = 未劣化)
    degraded_reason: str = ""


def attribute_phase(n_children: int, n_pass_gate: int, n_fsa: int,
                    n_eval_ok: int, n_eval_failed: int, delta: float):
    """S2-c i (2026-09-12): 把一轮的"劣化"归因到 AlphaSeek 四段式的某一段。

    trajectory_pool.TRAJECTORY_PHASES = [hypothesis, expression, code, backtest]
    伏羲版映射 (可观测代理):
      expression — 变异算子/骨架无解      (n_children == 0)
      code       — 门禁/FSA/求值失败      (表达式不合规 或 DSL 不支持)
      hypothesis — 管线健康但 ICIR 无增量 (方向本身没有 alpha 可挖)
    返回 (phase, reason); 未劣化返回 ("", "")。
    """
    if n_children == 0:
        return "expression", "GP 未产出任何子代 (骨架/变异算子无解)"
    if n_pass_gate == 0:
        return "code", f"子代 {n_children} 条全部未过门禁 (表达式/语义不合规)"
    if n_pass_gate > 0 and n_fsa == 0:
        return "code", f"门禁通过 {n_pass_gate} 条但全被 FSA 骨架过滤"
    if n_eval_ok == 0:
        return "code", (f"求值全部失败 (ok=0, failed={n_eval_failed}) "
                        f"— DSL/字段不支持")
    if delta <= 0:
        return "hypothesis", "管线健康但 ICIR 未改善 (该假设方向无增量)"
    return "", ""



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
        n_fsa: int = 0,
        n_eval_ok: int = 0,
        n_eval_failed: int = 0,
    ) -> TurnRecord:
        """记录一轮完成"""
        prev_icir = self.turns[-1].icir if self.turns else 0.0
        delta = icir - prev_icir

        # S2-c i: 步骤级归因 (仅对非首轮劣化判因)
        _deg_p, _deg_r = "", ""
        if self.turns and delta <= 0:
            _deg_p, _deg_r = attribute_phase(
                n_children, n_pass_gate, n_fsa, n_eval_ok, n_eval_failed, delta)

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
            n_fsa=int(n_fsa or 0),
            n_eval_ok=int(n_eval_ok or 0),
            n_eval_failed=int(n_eval_failed or 0),
            degraded_phase=_deg_p,
            degraded_reason=_deg_r,
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
        # S2-c i: 步骤级归因汇总
        _deg = {}
        for t in self.turns:
            if t.degraded_phase:
                _deg[t.degraded_phase] = _deg.get(t.degraded_phase, 0) + 1
        _n_delta_pos = sum(1 for t in self.turns if t.delta_from_prev > 0.001)
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
            # S2-c i (2026-09-12)
            "n_turns_improved": _n_delta_pos,
            "improve_ratio": round(_n_delta_pos / max(1, self.current_turn), 3),
            "degradation_attribution": _deg,
            "degraded_steps": [
                {"turn": t.turn_idx, "phase": t.degraded_phase,
                 "reason": t.degraded_reason}
                for t in self.turns if t.degraded_phase
            ],
            # S2-c ii: 高奖励片段 (本轮最佳轮次所属阶段)
            "high_reward_turn": self.best_turn,
            "high_reward_formula": (self.get_best_factor().formula
                                    if self.get_best_factor() else ""),
            "high_reward_icir": (self.get_best_factor().icir
                                 if self.get_best_factor() else 0.0),
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
                    # S2-c i (2026-09-12)
                    "n_fsa": t.n_fsa,
                    "n_eval_ok": t.n_eval_ok,
                    "n_eval_failed": t.n_eval_failed,
                    "degraded_phase": t.degraded_phase,
                    "degraded_reason": t.degraded_reason,
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
                n_fsa=t_data.get("n_fsa", 0),
                n_eval_ok=t_data.get("n_eval_ok", 0),
                n_eval_failed=t_data.get("n_eval_failed", 0),
                degraded_phase=t_data.get("degraded_phase", ""),
                degraded_reason=t_data.get("degraded_reason", ""),
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


# ═══════════════════════════════════════════════════════════════
# S2-c ii (2026-09-12): 高奖励片段重组 — 种子池构建
# ═══════════════════════════════════════════════════════════════

def load_high_reward_seed_pool(
    data_dir=None,
    top_k: int = 6,
    min_final_icir: float = 0.10,
    prefer_segment: str = "expression",
    validator=None,
    max_final_icir: float = 3.0,
    max_formula_len: int = 300,
    require_alpha: bool = True,
) -> List[Dict]:
    """从 `data/trajectory_log.json` 提取"高奖励片段"公式作重组父本。

    实测 (2026-09-12): 2194 条轨迹中 **1067 条** 已带非空
    `high_reward_segments` (值取自 TRAJECTORY_PHASES, 如 ["expression","backtest"]),
    但该字段**此前从未被任何消费者读取** ⇒ 这是现成的、零采集成本的资产。

    🔴 2026-09-12 实测: 该池**不是干净的** —— 实测取出的 top 条目为
    `scale(returns)` / `inv(scale(returns))`, 即 LLM 生成的**伪代码/占位描述**
    (标识符 scale/inv 在 S5 eval 上下文中不存在), 且其 `final_icir` 高达
    1000.0 / 23.8 (明显脏记录)。原实现按 `-final_icir` 排序 ⇒ **脏条目恒排最前**,
    重组出的子代 100% 求值失败 (NameError: name 'scale' is not defined)。
    故新增三道清洗闸 (可关):

      · `max_final_icir` 合理性上限 (默认 3.0) — 剔除损坏的 ICIR 记录
      · `validator(formula) -> List[str]` 标识符校验 (注入 ralph_loop 的
        `_find_unknown_identifiers`; 返回非空即非法) — 剔除伪代码
      · `require_alpha` 至少含一个下划线字段或已知算子 — 剔除纯常量/空壳

    返回 [{formula, factor_name, paradigm, final_icir, segments, best_segment}]
    按 (是否含 prefer_segment, final_icir) 降序, 最多 top_k 条。
    """
    d = Path(data_dir) if data_dir else (
        Path(__file__).resolve().parent.parent.parent / "data")
    p = d / "trajectory_log.json"
    if not p.exists():
        return []
    try:
        t = json.load(open(p, encoding="utf-8"))
    except Exception:
        return []
    trajs = t.get("trajectories", t) if isinstance(t, dict) else t
    if not isinstance(trajs, list):
        return []

    pool = []
    n_rej_icir = n_rej_ident = n_rej_len = 0
    for x in trajs:
        if not isinstance(x, dict):
            continue
        segs = list(x.get("high_reward_segments") or [])
        if not segs:
            continue
        _icir = float(x.get("final_icir") or 0)
        if _icir < min_final_icir:
            continue
        # 闸1: ICIR 合理性 (剔除 1000.0 / 23.8 这类损坏记录)
        if max_final_icir > 0 and _icir > max_final_icir:
            n_rej_icir += 1
            continue
        formula = ""
        for ph in ("expression", "code"):
            pd_ = x.get(ph)
            if isinstance(pd_, dict) and pd_.get("content"):
                formula = str(pd_["content"]).strip()
                break
        if not formula and x.get("seed_factor"):
            formula = str(x["seed_factor"]).strip()
        if not formula:
            continue
        # 闸2: 长度闸 (防御性, 超长内容必是描述文本而非公式)
        if max_formula_len > 0 and len(formula) > max_formula_len:
            n_rej_len += 1
            continue
        # 闸3: 标识符校验 (核心闸 — 剔除 scale(returns)/inv(...) 类伪代码)
        if validator is not None:
            try:
                if validator(formula):
                    n_rej_ident += 1
                    continue
            except Exception:
                pass
        # 闸4: 至少含一个面板字段 (下划线命名) 或已知算子调用
        if require_alpha and not re.search(r"[A-Za-z_]\w*_\w*|\w+\s*\(", formula):
            n_rej_len += 1
            continue
        pool.append({
            "formula": formula,
            "factor_name": str(x.get("seed_factor") or x.get("trajectory_id") or "")[:60],
            "paradigm": str(x.get("paradigm") or ""),
            "final_icir": float(x.get("final_icir") or 0),
            "segments": segs,
            "best_segment": (prefer_segment if prefer_segment in segs else segs[0]),
            "trajectory_id": x.get("trajectory_id", ""),
        })

    # 公式去重 (保留 ICIR 最高)
    dedup = {}
    for e in pool:
        k = re.sub(r"\s+", "", e["formula"])
        if k not in dedup or e["final_icir"] > dedup[k]["final_icir"]:
            dedup[k] = e
    pool = list(dedup.values())
    pool.sort(key=lambda e: (0 if prefer_segment in e["segments"] else 1,
                             -e["final_icir"]))
    if (n_rej_icir or n_rej_ident or n_rej_len):
        print(f"  [S2c-ii] 高奖励池清洗: 剔除 ICIR异常 {n_rej_icir} / "
              f"标识符非法 {n_rej_ident} / 长度或空壳 {n_rej_len} ⇒ 保留 {len(pool)}")
    return pool[:top_k]


def recombine_high_reward_seeds(pool: List[Dict], breeder, n_children: int = 6,
                                fsa=None, output_format: str = "pandas"):
    """S2-c ii: 用既有 GP 引擎对"高奖励片段"父本做跨轨迹重组。

    粒度说明 (诚实): 伏羲的 GP 引擎 (`breed_from_templates`) 本身即在
    **子树级**做交叉/变异, 故此处只需把高奖励片段公式喂成父本列表, 重组
    由既有引擎完成 —— 不另造一套易出类型错的拼接器。
    """
    if not pool or breeder is None or n_children <= 0:
        return []
    templates = [{"formula": e["formula"],
                  "factor_name": f"seg_{e['factor_name'][:36]}",
                  "sample_factor_ids": []}
                 for e in pool]
    try:
        kids = breeder.breed_from_templates(
            templates=templates, n_children=n_children,
            fsa=fsa, output_format=output_format)
    except Exception:
        return []
    for c in kids or []:
        if isinstance(c, dict):
            c["_source"] = "evo_segment_recombine"
            c["_segments"] = "|".join(sorted({s for e in pool for s in e["segments"]}))
    return kids or []
