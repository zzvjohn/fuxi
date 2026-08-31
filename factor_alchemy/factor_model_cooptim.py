# -*- coding: utf-8 -*-
"""
因子-模型联合优化框架 (RD-Agent(Q) 风格)
============================================
基于 Microsoft RD-Agent(Q) (NeurIPS 2025) 的核心思想:
  1. MAB自适应调度器 — 多臂老虎机自适应选择研究方向
  2. 因子-模型协同优化 — 不单独优化因子或模型，而是联合优化
  3. 数据驱动设计 — LLM仅与schema级信息交互，避免数据泄露
  4. 五阶段闭环 — Specification → Synthesis → Implementation → Validation → Analysis

与现有架构的关系:
  - 因子层: AlphaAgent v3 LLM生成 → 三重约束 → 互补配对
  - 模型层: 新增，与因子层联合优化
  - 调度层: MAB自适应选择研究方向
  - 反馈层: JQ影子验证 → RidgeUCB scorer

设计原则:
  - JQ是唯一真相源
  - Local仅做否决不做排序
  - LLM不接触raw data，只接触schema级描述
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import random


# ═══════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════

@dataclass
class ResearchDirection:
    """研究方向 — MAB的一个臂"""
    direction_id: str
    name: str
    description: str
    paradigm: str = ""                    # 关联的因子范式
    model_type: str = ""                  # 关联的模型类型
    generator: str = "gp_breed"           # v3.1: 推荐生成器 (gp_breed/llm/forge)
    expected_reward: float = 0.0         # MAB估计奖励
    pulls: int = 0                       # 被选择次数
    successes: int = 0                   # 成功次数(由JQ验证)
    failures: int = 0                    # 失败次数
    last_pulled: str = ""                # 最近一次选择时间
    status: str = "active"              # active / cooling / depleted
    cooldown_until: str = ""            # 冷却截止时间
    jq_results: List[Dict] = field(default_factory=list)
    # v3.1: 多生成器性能追踪
    generator_rewards: Dict[str, List[float]] = field(default_factory=dict)  # {generator: [reward1, ...]}


@dataclass
class ModelConfig:
    """模型配置 — 因子→预测的映射方式"""
    model_id: str
    model_type: str  # LightGBM / XGBoost / Ridge / NN / Ensemble
    description: str
    params: Dict = field(default_factory=dict)
    n_factors_used: int = 0
    performance: Dict = field(default_factory=dict)  # JQ评估结果


@dataclass
class CoOptimResult:
    """一轮联合优化结果"""
    round_id: str
    timestamp: str
    factor_changes: List[str]    # 新增/替换的因子
    model_changes: Dict          # 模型参数变化
    jq_return: Optional[float] = None
    jq_sharpe: Optional[float] = None
    vs_baseline: float = 0.0     # vs 王者基准
    insights: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# MAB 自适应调度器
# ═══════════════════════════════════════════════════════════

class MABScheduler:
    """
    多臂老虎机自适应调度器。

    实现 UCB1 (Upper Confidence Bound) 算法:
      score_i = mean_reward_i + c * sqrt(ln(total_pulls) / pulls_i)

    每个"臂"是一个研究方向（因子范式 × 模型类型的组合）。

    设计精要 (来自 RD-Agent(Q)):
    - 不贪婪选择历史表现最好的方向 → 探索-利用平衡
    - 自适应: JQ验证成功→reward↑, JQ验证失败→reward↓
    - 冷却: 连续3次失败 → 暂停该方向
    """

    def __init__(self, exploration_c: float = 2.0, cooldown_failures: int = 3):
        self.directions: Dict[str, ResearchDirection] = {}
        self.exploration_c = exploration_c
        self.cooldown_failures = cooldown_failures
        self.total_pulls = 0
        self.data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        self._state_path = self.data_dir / "mab_scheduler_state.json"
        self._load_state()

    def add_direction(self, rd: ResearchDirection):
        """注册新的研究方向"""
        if rd.direction_id not in self.directions:
            self.directions[rd.direction_id] = rd
        else:
            # 合并已有方向
            existing = self.directions[rd.direction_id]
            if rd.expected_reward != 0:
                existing.expected_reward = rd.expected_reward

    def select_direction(self, n: int = 1) -> List[ResearchDirection]:
        """
        选择n个研究方向 (UCB1算法)。

        返回列表，按UCB分数降序。
        """
        available = [
            d for d in self.directions.values()
            if d.status == "active"
        ]

        if not available:
            # 所有方向都被冷却 → 返回最近冷却时间最久的方向
            available = [
                d for d in self.directions.values()
                if d.status == "cooling"
            ]
            if not available:
                return []
            # 按冷却剩余时间排序
            available.sort(key=lambda d: d.cooldown_until)

        # UCB1计算
        scores = []
        for d in available:
            if d.pulls == 0:
                ucb_score = float('inf')  # 从未尝试过 → 优先
            else:
                exploration_bonus = self.exploration_c * np.sqrt(
                    np.log(max(1, self.total_pulls)) / d.pulls
                )
                ucb_score = d.expected_reward + exploration_bonus
            scores.append((ucb_score, d))

        scores.sort(key=lambda x: -x[0])
        selected = [d for _, d in scores[:n]]

        # 更新计数
        for d in selected:
            d.pulls += 1
            d.last_pulled = datetime.now().isoformat()
        self.total_pulls += n

        self.save_state()
        return selected

    def update_reward(self, direction_id: str, reward: float, success: bool):
        """
        更新方向奖励 (基于JQ回测结果)。

        reward 范围: -1.0 (极差) 到 +1.0 (极好)
        0 = 与基准持平
        +X = 超越基准
        -X = 跑输基准
        """
        if direction_id not in self.directions:
            return

        d = self.directions[direction_id]

        # 增量更新期望奖励
        if d.pulls > 0:
            d.expected_reward = (
                (d.expected_reward * (d.pulls - 1) + reward) / d.pulls
            )

        if success:
            d.successes += 1
            d.failures = 0  # 重置连续失败计数
        else:
            d.failures += 1

        # 连续失败 → 冷却
        if d.failures >= self.cooldown_failures:
            d.status = "cooling"
            from datetime import timedelta
            d.cooldown_until = (
                datetime.now() + timedelta(days=14 * (d.failures - self.cooldown_failures + 1))
            ).isoformat()
            print(f"[MAB] ⚠️ {d.name}: 连续{d.failures}次失败 → 冷却至{d.cooldown_until}")

        # 成功 → 解冻
        if success and d.status == "cooling":
            d.status = "active"
            d.cooldown_until = ""
            print(f"[MAB] ✅ {d.name}: 解冻")

        self.save_state()

    def get_exploration_report(self) -> Dict:
        """生成探索状态报告"""
        directions_summary = []
        for d in sorted(self.directions.values(),
                        key=lambda x: -x.expected_reward):
            directions_summary.append({
                "name": d.name,
                "paradigm": d.paradigm,
                "model_type": d.model_type,
                "expected_reward": round(d.expected_reward, 4),
                "pulls": d.pulls,
                "success_rate": (
                    round(d.successes / max(1, d.successes + d.failures), 2)
                    if (d.successes + d.failures) > 0 else None
                ),
                "status": d.status,
            })

        return {
            "total_directions": len(self.directions),
            "total_pulls": self.total_pulls,
            "directions": directions_summary,
        }

    def _load_state(self):
        if self._state_path.exists():
            try:
                with open(self._state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                for dd in state.get("directions", []):
                    rd = ResearchDirection(**dd)
                    self.directions[rd.direction_id] = rd
                self.total_pulls = state.get("total_pulls", 0)
            except Exception as e:
                print(f"[MAB] ⚠️ 状态加载失败: {e}")

    def save_state(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "updated_at": datetime.now().isoformat(),
            "total_pulls": self.total_pulls,
            "exploration_c": self.exploration_c,
            "directions": [
                {
                    "direction_id": d.direction_id,
                    "name": d.name,
                    "description": d.description,
                    "paradigm": d.paradigm,
                    "model_type": d.model_type,
                    "expected_reward": d.expected_reward,
                    "pulls": d.pulls,
                    "successes": d.successes,
                    "failures": d.failures,
                    "last_pulled": d.last_pulled,
                    "status": d.status,
                    "cooldown_until": d.cooldown_until,
                }
                for d in self.directions.values()
            ],
        }
        with open(self._state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
# 因子-模型联合优化引擎
# ═══════════════════════════════════════════════════════════

class FactorModelCoOptimizer:
    """
    因子-模型联合优化引擎。

    核心思想 (RD-Agent(Q)):
    - "因子挖掘和模型创新是量化研究的两大关键，它们互相依赖——
       好的因子需要好的模型来验证，好的模型也需要好的因子作为输入。"

    五阶段闭环:
      1. Specification  — 根据当前库状态确定优化目标
      2. Synthesis      — 生成因子假设 × 模型假设
      3. Implementation — 代码实现
      4. Validation     — JQ影子验证
      5. Analysis       — 分析结果 → 更新MAB + 更新库

    数据驱动设计:
      LLM不接触raw data，只接触:
      - 范式描述 (paradigm name + description)
      - 算子签名 (operator name + parameter types)
      - 因子统计摘要 (范式分布/相关性结构/红海状态)
      - JQ验证结果 (匿名化的success/failure + reward)
    """

    def __init__(
        self,
        data_dir: Path = None,
        mab: MABScheduler = None,
    ):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.mab = mab or MABScheduler()

        # 模型库
        self.models: Dict[str, ModelConfig] = {}
        self.results: List[CoOptimResult] = []

        # 初始化默认模型
        self._init_default_models()

        # 初始化默认研究方向
        self._init_default_directions()

        # 状态
        self._results_path = self.data_dir / "cooptim_results.json"

    def _init_default_models(self):
        """初始化默认模型配置"""
        defaults = [
            ModelConfig(
                model_id="equal_weight_rank",
                model_type="Ensemble",
                description="等权rank-product (王者当前方式)",
                params={"method": "rank_product", "weight_strategy": "equal"},
            ),
            ModelConfig(
                model_id="lightgbm_rank",
                model_type="LightGBM",
                description="LightGBM ranking模型, 因子→排名预测",
                params={"objective": "lambdarank", "max_depth": 4, "n_estimators": 100},
            ),
            ModelConfig(
                model_id="ridge_linear",
                model_type="Ridge",
                description="Ridge线性回归, 因子→收益预测",
                params={"alpha": 1.0},
            ),
            ModelConfig(
                model_id="xgb_rank",
                model_type="XGBoost",
                description="XGBoost ranking模型",
                params={"objective": "rank:pairwise", "max_depth": 3},
            ),
        ]
        for m in defaults:
            self.models[m.model_id] = m

    def _init_default_directions(self):
        """初始化默认研究方向 (范式 × 模型 组合)"""
        from paradigm_v4 import PARADIGMS_V4

        key_paradigms = [
            "流动性×微观结构", "资金流", "动量反转", "尾部风险",
            "筹码分布", "下行保护", "事件驱动", "北向资金",
            "行业轮动", "高频微观结构",
            # v0.6 P-003/P-005: 新范式模板已注入, 需要 MAB 方向
            "情绪×日内", "截面交互",
        ]

        key_models = ["equal_weight_rank", "lightgbm_rank", "ridge_linear"]

        for paradigm in key_paradigms:
            if paradigm not in PARADIGMS_V4:
                continue
            info = PARADIGMS_V4[paradigm]
            for model_id in key_models:
                model = self.models.get(model_id)
                if not model:
                    continue
                direction_id = f"{paradigm}__{model_id}"
                rd = ResearchDirection(
                    direction_id=direction_id,
                    name=f"{info['id']}.{paradigm} × {model.model_type}",
                    description=f"探索范式 [{paradigm}] 在 {model.model_type} 模型中的表现",
                    paradigm=paradigm,
                    model_type=model.model_type,
                )
                self.mab.add_direction(rd)

    def specify(self, library_stats: Dict) -> Dict:
        """
        阶段1: Specification — 确定本轮联合优化目标。

        输入: 库的当前状态 (来自 LibraryOrthogonalityManager)
        输出: 优化目标描述 (不接触raw data)
        """
        red_sea = library_stats.get("red_sea", {})
        paradigm_coverage = library_stats.get("paradigm_coverage", {})

        # 识别未被覆盖的范式
        uncovered = [
            p for p, count in paradigm_coverage.items()
            if count == 0
        ]

        # 识别过饱和的范式 (≥5个因子)
        saturated = [
            p for p, count in paradigm_coverage.items()
            if count >= 5
        ]

        # MAB选择方向
        selected_directions = self.mab.select_direction(n=2)

        spec = {
            "round_type": "explore" if uncovered else "exploit",
            "uncovered_paradigms": uncovered,
            "saturated_paradigms": saturated,
            "red_sea_level": red_sea.get("level", "unknown"),
            "selected_directions": [
                {
                    "name": d.name,
                    "paradigm": d.paradigm,
                    "model_type": d.model_type,
                    "ucb_score": "N/A (first pull)" if d.pulls == 0 else
                                 round(d.expected_reward, 4),
                }
                for d in selected_directions
            ],
            "constraints": {
                "max_new_factors": 10,
                "avoid_paradigms": saturated,
                "prefer_paradigms": uncovered,
                "forbidden_regions": [
                    fr for fr in library_stats.get("forbidden_regions", [])
                    if isinstance(fr, dict) and fr.get("severity") == "hard"
                ],
            },
        }
        return spec

    def synthesize_context_for_llm(self, spec: Dict,
                                     library_stats: Dict) -> str:
        """
        阶段2: Synthesis — 生成LLM上下文 (schema-only, 无raw data)。

        这是数据驱动设计的核心: LLM只看到结构描述，不接触具体数据。
        """
        lines = [
            "## 当前库状态 (Schema-level)",
            f"- 总因子数: {library_stats.get('total_factors', 0)}",
            f"- 因子族数: {library_stats.get('n_clusters', 0)}",
            f"- 范式覆盖: {len(library_stats.get('paradigm_coverage', {}))} 个范式",
            f"- Red Sea等级: {library_stats.get('red_sea', {}).get('level', 'unknown')}",
            "",
            "## 本轮优化目标",
            f"- 类型: {spec['round_type']}",
        ]

        if spec["uncovered_paradigms"]:
            lines.append(f"- 未覆盖范式 (高优先级): {', '.join(spec['uncovered_paradigms'])}")
        if spec["saturated_paradigms"]:
            lines.append(f"- 过饱和范式 (避免): {', '.join(spec['saturated_paradigms'])}")

        lines.append("")
        lines.append("## 选定的研究方向")
        for d in spec["selected_directions"]:
            lines.append(f"- {d['name']}")

        lines.append("")
        lines.append("## 可用算子 (Forge语法)")
        from paradigm_v4 import STANDARD_OPERATORS
        for cat, ops in STANDARD_OPERATORS.items():
            lines.append(f"### {cat}: {', '.join(ops)}")

        lines.append("")
        lines.append("## 要求")
        lines.append("1. 生成因子的Forge表达式 (使用上述算子)")
        lines.append("2. 每个因子必须属于指定范式并有经济学rationale")
        lines.append("3. 不重复已知禁止区域的因子结构")
        lines.append("4. 优先覆盖未覆盖范式")

        return "\n".join(lines)

    def record_result(self, result: CoOptimResult):
        """记录一轮联合优化结果"""
        self.results.append(result)

        # 更新MAB
        # 根据实际JQ结果计算reward
        if result.jq_return is not None:
            # vs 王者 (+182.57%)
            baseline_return = 182.57
            relative = (result.jq_return - baseline_return) / 100.0

            # reward映射: 超越王者→正, 持平→0, 跑输→负
            reward = np.clip(relative * 5, -1.0, 1.0)
            success = result.jq_return >= baseline_return

            # 更新所有涉及的方向
            for change in result.factor_changes:
                # 从结果中的因子变化推断方向
                for d in self.mab.directions.values():
                    if d.paradigm in change:
                        self.mab.update_reward(d.direction_id, reward, success)

        self._save_results()

    def get_insights_for_next_round(self) -> List[str]:
        """从历史结果中提取下一轮的建议"""
        insights = []

        if not self.results:
            insights.append("尚未有JQ验证结果 — 首次探索建议从低饱和范式开始")
            return insights

        # 最近5轮
        recent = self.results[-5:]

        # 成功率
        n_success = sum(1 for r in recent if r.vs_baseline >= 0)
        insights.append(f"近5轮成功率: {n_success}/5")

        # 有效范式
        paradigm_performance = defaultdict(list)
        for r in recent:
            for change in r.factor_changes:
                for d in self.mab.directions.values():
                    if d.paradigm in change:
                        paradigm_performance[d.paradigm].append(r.vs_baseline)

        for paradigm, perfs in paradigm_performance.items():
            avg = np.mean(perfs)
            if avg > 0:
                insights.append(f"✅ [{paradigm}] 平均+{avg:.1f}pp vs 王者 → 继续探索")
            else:
                insights.append(f"❌ [{paradigm}] 平均{avg:.1f}pp vs 王者 → 考虑冷却或范式内微调")

        # MAB报告
        report = self.mab.get_exploration_report()
        top_directions = report["directions"][:3]
        insights.append(f"Top-3 UCB方向: {' | '.join(d['name'] for d in top_directions)}")

        return insights

    def _save_results(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "updated_at": datetime.now().isoformat(),
            "n_results": len(self.results),
            "results": [
                {
                    "round_id": r.round_id,
                    "timestamp": r.timestamp,
                    "factor_changes": r.factor_changes,
                    "model_changes": r.model_changes,
                    "jq_return": r.jq_return,
                    "jq_sharpe": r.jq_sharpe,
                    "vs_baseline": r.vs_baseline,
                    "insights": r.insights,
                }
                for r in self.results[-50:]  # 只保留最近50轮
            ],
        }
        with open(self._results_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════

def create_default_cooptimizer() -> FactorModelCoOptimizer:
    """创建默认配置的联合优化器"""
    return FactorModelCoOptimizer()


def print_system_status(cooptim: FactorModelCoOptimizer):
    """打印系统状态"""
    report = cooptim.mab.get_exploration_report()
    print("=" * 60)
    print("  因子-模型联合优化系统状态")
    print("=" * 60)
    print(f"  研究方向数: {report['total_directions']}")
    print(f"  总探索次数: {report['total_pulls']}")
    print(f"  已完成轮次: {len(cooptim.results)}")
    print()
    print("  Top-5 UCB方向:")
    for i, d in enumerate(report["directions"][:5]):
        status_icon = {"active": "🟢", "cooling": "🟡", "depleted": "🔴"}.get(d["status"], "⚪")
        sr = f" {d['success_rate']:.0%}" if d["success_rate"] is not None else " N/A"
        print(f"  {i+1}. {status_icon} {d['name']}")
        print(f"     reward={d['expected_reward']:.3f} pulls={d['pulls']} sr={sr}")
    print("=" * 60)


if __name__ == "__main__":
    cooptim = create_default_cooptimizer()
    print_system_status(cooptim)
