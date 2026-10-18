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


# ── S2-c iii (2026-09-12): 模型代理奖励的量纲基准 (唯一权威定义处) ──
# proxy_reward = clip(ic_mean / PROXY_REWARD_REF_IC, -1, 1)
# 与 JQ 奖励 clip((jq_return-182.57)/100*5, -1, 1) 不同源 ⇒ 禁止混算。
PROXY_REWARD_REF_IC = 0.05


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
    last_pulled: str = ""                # 最近一次"实际执行"时间 (语义见 S1-a)
    status: str = "active"              # active / cooling / depleted
    cooldown_until: str = ""            # 冷却截止时间
    jq_results: List[Dict] = field(default_factory=list)
    # v3.1: 多生成器性能追踪
    generator_rewards: Dict[str, List[float]] = field(default_factory=dict)  # {generator: [reward1, ...]}
    # S1-a a3 (2026-09-12): "被选中"与"被实际执行"分离。
    # 原实现 select_direction 一选中就 pulls += 1 ⇒ 诊断(specify)每跑一次就
    # 虚增 pulls, 实测 total_pulls=317 而 sum(pulls)=484 两本账不一致。
    # 现在: selected_at = 被诊断/规划选中时间 (不计 pull);
    #       last_pulled = 实际执行时间 (才计 pull)。
    selected_at: str = ""
    n_diagnostic_selections: int = 0    # 被选中但未执行的次数 (空计数留痕)
    # S2-c iii (2026-09-12): 奖励量纲隔离。
    # JQ 奖励 = clip((jq_return-182.57)/100*5, -1, 1)  (1.0 ≈ +20pp 超额)
    # 模型代理奖励 = clip(ic_mean/0.05, -1, 1)          (1.0 ≈ IC 0.05)
    # 两者**不同源**, 混算会污染 MAB 先验 ⇒ 记录来源, 换源时重置而非混合。
    reward_source: str = ""             # "" | "jq" | "model_proxy"
    reward_scale: str = ""              # 量纲描述 (留痕)
    n_rewards: int = 0                  # 该臂累计收到的奖励观测数 (期望值的分母)


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
        # ── S1-a (2026-09-12): pulls 语义开关 + 多写者防覆盖 ──
        # FUXI_MAB_PULL_SEMANTICS:
        #   fixed(默认) → 只有"实际执行"才计 pull (选中只记 selected_at)
        #   legacy      → 沿用旧语义 (选中即 +1), 仅供历史复现
        import os as _os
        _ps = _os.environ.get("FUXI_MAB_PULL_SEMANTICS", "fixed").strip().lower()
        self.pull_semantics = "legacy" if _ps == "legacy" else "fixed"
        # 多实例/多进程防覆盖: 记录加载时的 updated_at, save 时检测他方写入
        self._loaded_updated_at: Optional[str] = None
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

    def resolve_arms(self, identifier: str) -> List[ResearchDirection]:
        """
        S2-a (2026-09-12): 权威臂解析器 — 消灭"死匹配"这一类 bug。

        根因: 臂的命名是三段不一致的 ——
          direction_id = "资金流__lightgbm_rank"
          name         = "2.资金流 × LightGBM"
          paradigm     = "资金流"
        而历史调用点普遍写 `d.name == paradigm or d.direction_id == paradigm`
        (paradigm 是纯范式名) ⇒ **恒不命中**, 静默丢奖励/丢归因。
        (实测 `mab_scheduler_state.json` 全部 39 臂的 name 均形如
         "N.范式 × Model", direction_id 均形如 "范式__model", 无一等于范式名。)

        解析优先级 (高→低, 一旦命中即返回):
          1. direction_id 精确
          2. name 精确
          3. paradigm 精确 → 该范式下**全部** model 臂
          4. 子串命中 (identifier ⊂ name / paradigm / direction_id)
             → 用于 "资金流" ⊂ "2.资金流 × LightGBM" 这类历史调用

        返回: 命中的臂列表 (可能为空)。
        """
        if not identifier:
            return []
        _id = str(identifier)
        # 1) direction_id 精确
        d = self.directions.get(_id)
        if d is not None:
            return [d]
        # 2) name 精确
        for _d in self.directions.values():
            if _d.name == _id:
                return [_d]
        # 3) paradigm 精确 (一个范式对多个 model 臂)
        hits = [d for d in self.directions.values() if d.paradigm == _id]
        if hits:
            return hits
        # 4) 双向子串
        hits = [
            d for d in self.directions.values()
            if _id in str(d.name) or _id in str(d.paradigm)
            or _id in str(d.direction_id)
        ]
        return hits

    def select_direction(self, n: int = 1, commit: Optional[bool] = None) -> List[ResearchDirection]:
        """
        选择n个研究方向 (UCB1算法)。

        返回列表，按UCB分数降序。

        S1-a a3 (2026-09-12): `commit` 控制是否把"选中"记为一次 pull。
          - 诊断/规划路径 (FactorModelCoOptimizer.specify) 必须 commit=False,
            否则每跑一次诊断就虚增 pulls (实测两本账 317 vs 484 不一致);
          - 期望"选中即执行"的调用方显式 commit=True;
          - 默认值 None → 由 FUXI_MAB_PULL_SEMANTICS 决定
            (fixed→False / legacy→True, 历史复现用)。
        """
        if commit is None:
            commit = (self.pull_semantics == "legacy")
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
        _now = datetime.now().isoformat()
        for d in selected:
            d.selected_at = _now          # a3: 选中只记 selected_at (零成本)
            if commit:
                d.pulls += 1
                d.last_pulled = _now
            else:
                d.n_diagnostic_selections = int(
                    getattr(d, "n_diagnostic_selections", 0) or 0) + 1
        if commit:
            self.total_pulls += n
            self.save_state()

        return selected

    def mark_executed(self, direction_id: str, save: bool = True) -> bool:
        """
        S1-a a3 (2026-09-12): 把一次**实际执行**记入 pulls / total_pulls。

        这是唯一合法的 pulls 增长点 (legacy 语义除外)。返回是否命中该方向。
        """
        d = self.directions.get(direction_id)
        if d is None:
            for _d in self.directions.values():
                if _d.name == direction_id:
                    d = _d
                    break
        if d is None:
            return False
        d.pulls += 1
        d.last_pulled = datetime.now().isoformat()
        self.total_pulls += 1
        if save:
            self.save_state()
        return True

    def ensure_reward_source(self, d: ResearchDirection, source: str) -> bool:
        """S2-c iii (2026-09-12): 奖励量纲守卫。

        JQ 奖励  = clip((jq_return-182.57)/100*5, -1, 1)   (1.0 ≈ +20pp 超额)
        模型代理 = clip(ic_mean/0.05, -1, 1)               (1.0 ≈ IC 0.05)
        二者**不同源**, 混算会得到一个既不是收益也不是 IC 的无意义期望值
        ⇒ 检测到换源就**重置** (而非平均), 并把来源写进臂上留痕。

        返回 True 表示发生了换源重置。
        """
        _scale = ("jq_vs_182.57" if source == "jq"
                  else f"proxy_ref_ic={PROXY_REWARD_REF_IC}")
        if d.reward_source and d.reward_source != source:
            print(f"[MAB] 🔀 {d.name}: 奖励源 {d.reward_source}→{source} "
                  f"⇒ 重置期望奖励 (禁止跨量纲混算)")
            d.expected_reward = 0.0
            d.n_rewards = 0
            d.reward_source = source
            d.reward_scale = _scale
            return True
        d.reward_source = source
        d.reward_scale = _scale
        return False

    def update_reward(self, direction_id: str, reward: float, success: bool,
                      source: str = "jq"):
        """
        更新方向奖励 (基于JQ回测结果)。

        reward 范围: -1.0 (极差) 到 +1.0 (极好)
        0 = 与基准持平
        +X = 超越基准
        -X = 跑输基准

        S2-c iii (2026-09-12): `source` 标注量纲来源 ("jq" / "model_proxy")。
        换源 ⇒ **重置** (见 `ensure_reward_source`), 且期望值分母改用
        `n_rewards` (原实现用 `pulls` — 而 pulls 计的是"选择/执行次数",
        不是"奖励观测数", 属口径错配)。
        """
        if direction_id not in self.directions:
            return

        d = self.directions[direction_id]

        self.ensure_reward_source(d, source)
        d.n_rewards = int(getattr(d, "n_rewards", 0) or 0) + 1
        _n = d.n_rewards
        d.expected_reward = (d.expected_reward * (_n - 1) + reward) / _n

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

    def record_generator_reward(
        self,
        identifier: str,
        generator: str,
        reward: float,
        success: bool = True,
        source: str = "model_proxy",
        cap: int = 20,
        save: bool = True,
    ) -> int:
        """
        S2-a (2026-09-12): 生成器轴 (engine axis) 的奖励写入点。

        根因: `ResearchDirection.generator_rewards` 此前**全代码库无写入点**
        (仅 `_mab_select_generator` 读它) ⇒ `_mab_select_generator` 里
        "有历史数据: 软最大化选择最优生成器" 分支是**死代码**, 永远走
        `else` 的冷启动启发式 (按 pulls 分档)。也就是说**生成器选择从未学习**。

        本方法是 `generator_rewards` 的唯一合法写入点, 与方向奖励
        (`update_reward`) 分轴并存:
            direction 轴 = 范式 × 模型   (选"研究什么")
            generator 轴 = 引擎/生成器   (选"怎么生成")
        两者共用 `resolve_arms` 解析 + `ensure_reward_source` 量纲守卫。

        identifier: direction_id / name / paradigm (经 resolve_arms 解析)
        generator : gp_breed | llm | forge | gp_evolve | ...
        返回: 实际写入的臂数 (0 = 未命中任何臂 ⇒ 调用方应告警)
        """
        arms = self.resolve_arms(identifier)
        if not arms:
            return 0
        _rw = max(-1.0, min(1.0, float(reward)))
        for d in arms:
            # 生成器轴独立于方向轴量纲: generator_rewards 是纯奖励序列,
            # 不做 ensure_reward_source 重置 (那是方向级期望值的守卫)。
            bucket = d.generator_rewards.setdefault(str(generator), [])
            bucket.append(round(_rw, 6))
            if cap and len(bucket) > cap:
                del bucket[:-cap]
        if save:
            self.save_state()
        return len(arms)

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
                "n_diagnostic_selections": int(
                    getattr(d, "n_diagnostic_selections", 0) or 0),
                "selected_at": d.selected_at,
                "success_rate": (
                    round(d.successes / max(1, d.successes + d.failures), 2)
                    if (d.successes + d.failures) > 0 else None
                ),
                "status": d.status,
            })

        return {
            "total_directions": len(self.directions),
            "total_pulls": self.total_pulls,
            "pull_semantics": self.pull_semantics,
            "n_diagnostic_selections": sum(
                int(getattr(d, "n_diagnostic_selections", 0) or 0)
                for d in self.directions.values()),
            "directions": directions_summary,
        }

    def _load_state(self):
        if self._state_path.exists():
            try:
                with open(self._state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                for dd in state.get("directions", []):
                    # 只取 dataclass 已知字段, 容忍历史 json 缺键/多键
                    _known = ResearchDirection.__dataclass_fields__.keys()
                    rd = ResearchDirection(**{k: v for k, v in dd.items()
                                              if k in _known})
                    self.directions[rd.direction_id] = rd
                self.total_pulls = state.get("total_pulls", 0)
                self._loaded_updated_at = state.get("updated_at")
            except Exception as e:
                print(f"[MAB] ⚠️ 状态加载失败: {e}")

    def _merge_external_state(self, ext: Dict) -> Dict[str, int]:
        """
        S1-a a2 (2026-09-12): 多写者单调合并。

        根因: RalphLoop.__init__ 与 FactorModelCoOptimizer.__init__ 各自 new 一个
        MABScheduler, 共用 data/mab_scheduler_state.json ⇒ 后写者用**自己内存里的
        旧值**覆盖前者 (实测 total_pulls=317 而 sum(pulls)=484, 两本账不一致)。

        策略: 只在检测到"他方已写入"(_loaded_updated_at != 文件 updated_at) 时合并,
        且只对**单调量**取 max (pulls/total_pulls/successes/failures), 列表做并集 —
        永不减少计数 ⇒ 零风险 (不会让任何统计倒退)。
        """
        merged = {"pulls": 0, "total_pulls": 0, "recovered": 0}
        ext_dirs = {d.get("direction_id"): d for d in ext.get("directions", [])
                    if isinstance(d, dict)}
        for did, ed in ext_dirs.items():
            d = self.directions.get(did)
            if d is None:
                continue
            ep = int(ed.get("pulls", 0) or 0)
            if ep > d.pulls:
                merged["recovered"] += ep - d.pulls
                d.pulls = ep
            for _f in ("successes", "failures"):
                ev = int(ed.get(_f, 0) or 0)
                if ev > int(getattr(d, _f, 0) or 0):
                    setattr(d, _f, ev)
            # 列表并集 (jq_results 按内容去重, generator_rewards 按值并集)
            for _r in (ed.get("jq_results") or []):
                if isinstance(_r, dict) and _r not in d.jq_results:
                    d.jq_results.append(_r)
            for _gen, _vals in (ed.get("generator_rewards") or {}).items():
                _cur = d.generator_rewards.setdefault(_gen, [])
                for _v in (_vals or []):
                    if _v not in _cur:
                        _cur.append(_v)
        etp = int(ext.get("total_pulls", 0) or 0)
        if etp > self.total_pulls:
            merged["total_pulls"] = etp - self.total_pulls
            self.total_pulls = etp
        return merged

    def save_state(self):
        from fuxi_v1.legacy_writers import assert_state_writer
        assert_state_writer(self._state_path)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # ── S1-a a2: 写前检测他方写入 → 单调合并 ──
        _merge_info = {"pulls": 0, "total_pulls": 0, "recovered": 0}
        if self._state_path.exists():
            try:
                with open(self._state_path, 'r', encoding='utf-8') as f:
                    ext = json.load(f)
                if ext.get("updated_at") != self._loaded_updated_at:
                    # 无 _loaded_updated_at 且文件存在 ⇒ 他方在本实例 init 之后
                    # 才创建了文件, 同样必须合并 (否则盲覆盖)
                    _merge_info = self._merge_external_state(ext)
                    if _merge_info["recovered"] or _merge_info["total_pulls"]:
                        print(f"[MAB] 🔀 检测到他方写入 → 单调合并: "
                              f"回收 pulls={_merge_info['recovered']}, "
                              f"total_pulls+={_merge_info['total_pulls']}")
            except Exception:
                pass
        state = {
            "updated_at": datetime.now().isoformat(),
            "total_pulls": self.total_pulls,
            "exploration_c": self.exploration_c,
            "pull_semantics": self.pull_semantics,   # S1-a: 留痕
            "merge_recovered": _merge_info,          # S1-a: 留痕
            "directions": [
                {
                    "direction_id": d.direction_id,
                    "name": d.name,
                    "description": d.description,
                    "paradigm": d.paradigm,
                    "model_type": d.model_type,
                    "generator": d.generator,          # a1: 原白名单漏字段
                    "expected_reward": d.expected_reward,
                    "pulls": d.pulls,
                    "successes": d.successes,
                    "failures": d.failures,
                    "last_pulled": d.last_pulled,
                    "selected_at": d.selected_at,      # a3: 新增留痕
                    "n_diagnostic_selections": int(
                        getattr(d, "n_diagnostic_selections", 0) or 0),
                    "reward_source": d.reward_source,  # S2-c iii: 量纲来源
                    "reward_scale": d.reward_scale,
                    "n_rewards": int(getattr(d, "n_rewards", 0) or 0),
                    "status": d.status,
                    "cooldown_until": d.cooldown_until,
                    # ── a1 (2026-09-12): 原 12 字段白名单静默丢弃这两个 ──
                    "jq_results": d.jq_results,
                    "generator_rewards": d.generator_rewards,
                }
                for d in self.directions.values()
            ],
        }
        with open(self._state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        self._loaded_updated_at = state["updated_at"]


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
            # S1-a (2026-09-12) 修复: 原为 .parent.parent/"data" → E:\quant\research\data
            # (仓库根 data 应为 .parent.parent.parent, 与 MABScheduler._state_path 对齐)。
            # 后果: cooptim_results.json 会落到错误的 research/data/ 且从未被消费。
            data_dir = Path(__file__).resolve().parent.parent.parent / "data"
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
        # S1-a a3 (2026-09-12): 显式 commit=False — 规划/诊断阶段的"选中"
        # 不是一次探索, 不得计入 pulls (否则每跑一次诊断就虚增, UCB 被污染)。
        selected_directions = self.mab.select_direction(n=2, commit=False)

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
        from fuxi_v1.legacy_writers import assert_state_writer
        assert_state_writer(self._results_path)
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
