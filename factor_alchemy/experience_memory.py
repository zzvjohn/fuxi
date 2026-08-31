"""
Experience Memory — FactorMiner 经验记忆库 (v2: F/E/R 操作符 + 结构化 P_succ/P_fail)

记录每次因子发现尝试，支持相似历史检索，在生成新因子前提供经验上下文。

设计 (v2 增强, 对标 FactorMiner Section 3.3):
  1. 每次 FRI 评估后自动记录
  2. 按范式/结果/标签/FRI画像检索相似历史
  3. 生成 LLM 经验上下文（用于 AlphaAgent 生成因子时注入）
  4. 自动提取模式（toxic/success 模式检测）
  5. F/E/R 操作符: Formation(从轨迹提取) / Evolution(合并冗余) / Retrieval(上下文检索)
  6. 结构化 P_succ (成功因子模板) 和 P_fail (禁止区域) — FactorMiner 核心

文件: data/experience_memory.json

用法:
    from research.factor_alchemy.experience_memory import ExperienceMemory

    mem = ExperienceMemory()
    mem.record(factor_name="test", formula="...", fri=0.7, outcome="PASS")
    
    # F/E/R 操作符
    mem.form(trajectory)       # F: 从轨迹提取经验
    mem.evolve()               # E: 合并冗余/淘汰低效用
    priors = mem.retrieve(library_context, k=5)  # R: 检索相关 priors
    
    # LLM 上下文
    context = mem.get_llm_context(paradigm="动量")
    # → "过去动量因子尝试 12 次, 成功 8 次, 典型ICIR 0.45..."
    
    # P_succ / P_fail 结构化记忆
    template = mem.get_success_templates(paradigm="流动性")  # 成功因子模板
    forbidden = mem.get_forbidden_regions()                  # 禁止方向
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


# ── 默认路径 ──────────────────────────────────────────────
MEMORY_PATH = Path(__file__).parent.parent.parent / "data" / "experience_memory.json"


# ── v0.8: 模板验证等级排序权重 (retrieve 主键) ──────────
VERIFICATION_RANK = {
    "jq_single": 3,     # 单因子 JQ 验证通过
    "jq_composite": 2,  # 仅组合级 JQ 验证
    "s5_passed": 1,     # 通过 S1-S5 本地校验, 未 JQ
    "stage2_only": 0,   # 仅 Stage2 快筛
}


# ── 结构化数据类型 (v2 新增) ─────────────────────────────

@dataclass
class SuccessPattern:
    """成功因子模板 — 可复用为 LLM 生成约束的结构化知识
    
    对标 FactorMiner Table 4: Recommended mining directions from experience memory.
    """
    pattern_id: str
    description: str           # 如 "Higher Moment Regimes"
    typical_operators: List[str]  # 如 ["Skew", "Kurt", "IfElse"]
    typical_windows: List[int] = field(default_factory=lambda: [20, 60])
    applicable_regimes: List[str] = field(default_factory=list)
    success_rate: float = 0.0   # 该模板生成因子的通过率
    ic_range: Tuple[float, float] = (0.02, 0.08)  # 典型IC范围
    icir_range: Tuple[float, float] = (0.3, 0.8)
    sample_factor_ids: List[str] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)
    formula: str = ""            # v0.5: 模板对应的示例公式 (供 GP 育种)
    created_at: str = ""
    updated_at: str = ""
    occurrence_count: int = 0  # 因子库中出现次数
    jq_return: Optional[float] = None   # JQ回测收益%
    jq_sharpe: Optional[float] = None   # JQ回测Sharpe
    bridge_consumed: bool = False  # v0.6 P-020: 是否已通过 Memory Bridge 传递给 GP 育种
    # v0.8: 模板验证等级 — 决定该模板作为 GP 亲本/LLM 约束时的可信度权重
    #   jq_single     单因子 JQ 回测验证通过 (P-001 有单因子 IC/ICIR)
    #   jq_composite  仅组合级 JQ 验证 (如 AlphaAgent v3 六因子复合 +182.57%)
    #   s5_passed     通过 S1-S5 本地校验但未 JQ
    #   stage2_only   仅 Stage2 快筛 (ICIR>=0.3), 无 S5 无 JQ
    verification_level: str = "stage2_only"

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "typical_operators": self.typical_operators,
            "typical_windows": self.typical_windows,
            "applicable_regimes": self.applicable_regimes,
            "success_rate": self.success_rate,
            "ic_range": list(self.ic_range),
            "icir_range": list(self.icir_range),
            "sample_factor_ids": self.sample_factor_ids,
            "lessons": self.lessons,
            "formula": self.formula,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "occurrence_count": self.occurrence_count,
            "jq_return": self.jq_return,
            "jq_sharpe": self.jq_sharpe,
            "bridge_consumed": self.bridge_consumed,  # v0.6 P-020
            "verification_level": self.verification_level,  # v0.8
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SuccessPattern":
        return cls(
            pattern_id=d.get("pattern_id", ""),
            description=d.get("description", ""),
            typical_operators=d.get("typical_operators", []),
            typical_windows=d.get("typical_windows", [20, 60]),
            applicable_regimes=d.get("applicable_regimes", []),
            success_rate=d.get("success_rate", 0.0),
            ic_range=tuple(d.get("ic_range", [0.02, 0.08])),
            icir_range=tuple(d.get("icir_range", [0.3, 0.8])),
            sample_factor_ids=d.get("sample_factor_ids", []),
            lessons=d.get("lessons", []),
            formula=d.get("formula", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            occurrence_count=d.get("occurrence_count", 0),
            jq_return=d.get("jq_return"),
            jq_sharpe=d.get("jq_sharpe"),
            bridge_consumed=d.get("bridge_consumed", False),  # v0.6 P-020
            verification_level=d.get("verification_level", "stage2_only"),  # v0.8
        )


@dataclass
class ForbiddenDirection:
    """禁止方向 — 已验证无效的因子探索路径
    
    对标 FactorMiner Table 5: Forbidden mining directions (high correlation risk).
    """
    direction_id: str
    description: str           # 如 "VWAP Deviation variants"
    correlated_factors: List[str]  # 与哪些现有因子高相关
    correlation_threshold: float = 0.5
    prototype_expression: str = ""  # 典型失败表达式
    reason: str = ""            # "high_correlation" / "jq_failure" / "local_toxic"
    severity: str = "soft"      # hard=绝对禁止 / soft=低优先级
    failed_attempts: int = 0
    added_at: str = ""
    updated_at: str = ""
    cooldown_until: str = ""
    jq_return: Optional[float] = None   # JQ回测收益%
    jq_sharpe: Optional[float] = None   # JQ回测Sharpe

    def to_dict(self) -> dict:
        return {
            "direction_id": self.direction_id,
            "description": self.description,
            "correlated_factors": self.correlated_factors,
            "correlation_threshold": self.correlation_threshold,
            "prototype_expression": self.prototype_expression,
            "reason": self.reason,
            "severity": self.severity,
            "failed_attempts": self.failed_attempts,
            "added_at": self.added_at,
            "updated_at": self.updated_at,
            "cooldown_until": self.cooldown_until,
            "jq_return": self.jq_return,
            "jq_sharpe": self.jq_sharpe,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ForbiddenDirection":
        return cls(
            direction_id=d.get("direction_id", ""),
            description=d.get("description", ""),
            correlated_factors=d.get("correlated_factors", []),
            correlation_threshold=d.get("correlation_threshold", 0.5),
            prototype_expression=d.get("prototype_expression", ""),
            reason=d.get("reason", ""),
            severity=d.get("severity", "soft"),
            failed_attempts=d.get("failed_attempts", 0),
            added_at=d.get("added_at", ""),
            updated_at=d.get("updated_at", ""),
            cooldown_until=d.get("cooldown_until", ""),
            jq_return=d.get("jq_return"),
            jq_sharpe=d.get("jq_sharpe"),
        )


@dataclass
class MemoryEvent:
    """记忆演化事件日志 — 记录 F/E/R 操作"""
    event_id: str
    event_type: str           # "formation" / "evolution" / "retrieval"
    timestamp: str
    details: Dict = field(default_factory=dict)


class ExperienceMemory:
    """因子发现经验记忆库 (v2: F/E/R 操作符增强)
    
    F (Formation): 从挖掘轨迹中提取成功模板和禁止方向
    E (Evolution): 合并冗余模式，淘汰低效用信息
    R (Retrieval): 基于当前库状态检索上下文相关的记忆信号
    """
    
    def __init__(self, memory_path: Path = MEMORY_PATH):
        self.path = memory_path
        self.data = self._load()
        # F/E/R 操作统计
        self._stats = {
            "formations": 0,
            "evolutions": 0,
            "retrievals": 0,
        }
    
    # ── 存储操作 ──────────────────────────────────────────
    
    def _load(self) -> Dict:
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # v2 兼容: 自动迁移旧格式
            return self._migrate_if_needed(data)
        return self._create_empty()
    
    def _create_empty(self) -> Dict:
        return {
            "version": "3.1.0",  # v3.1: warning_directions 软负收益记录层
            "attempts": [],
            "success_templates": [],    # v2: 结构化 P_succ
            "forbidden_directions": [], # v2: 结构化 P_fail (hard 禁止)
            "warning_directions": [],   # v3.1: 软负收益记录 (severity=soft, jq_ret<0 自动记录)
            "motif_stats": {},          # v3 P-001: {"motif_key": {"pass": N, "fail": N, "jq_pass": N, "jq_fail": N}}
            "patterns": {
                "toxic_patterns": [],
                "successful_patterns": [],
                "regime_insights": {},
            },
            "stats": {
                "total_attempts": 0,
                "total_pass": 0,
                "total_reject": 0,
                "by_paradigm": {},
            },
            "evolution_log": [],         # v2: F/E/R 操作日志
            "forbidden_regions": [],     # v1 兼容
        }
    
    def _migrate_if_needed(self, data: Dict) -> Dict:
        """自动迁移 v1 → v2 数据格式"""
        if data.get("version", "1.0.0") == "1.0.0":
            data["version"] = "2.0.0"
            data.setdefault("success_templates", [])
            data.setdefault("forbidden_directions", [])
            data.setdefault("evolution_log", [])
            # 迁移 v1 forbidden_regions → v2 forbidden_directions
            for fr in data.get("forbidden_regions", []):
                fd = {
                    "direction_id": f"migrated_{fr.get('paradigm', 'unknown')}",
                    "description": f"{fr.get('paradigm', '')}: {fr.get('reason', '')}",
                    "correlated_factors": [],
                    "correlation_threshold": 0.5,
                    "prototype_expression": fr.get("prototype_expression", ""),
                    "reason": "correlation_exhausted",
                    "severity": fr.get("severity", "soft"),
                    "failed_attempts": fr.get("count", 1),
                    "added_at": fr.get("added_at", ""),
                    "cooldown_until": "",
                }
                data["forbidden_directions"].append(fd)
        return data
    
    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def record(
        self,
        factor_name: str,
        formula: str,
        paradigm: str,
        category: str = "",
        fri_score: float = 0.0,
        fri_grade: str = "",
        fri_precision: float = 0.0,
        fri_persistence: float = 0.0,
        fri_consistency: float = 0.0,
        fri_novelty: float = 0.0,
        icir: float = 0.0,
        outcome: str = "PASS",     # PASS / REJECT / WEAK
        psi_r_squared: float = 0.0,
        psi_independence: float = 1.0,
        max_corr_factor: str = "",
        tags: Optional[List[str]] = None,
        lessons: Optional[List[str]] = None,
        motifs: Optional[List[str]] = None,  # v3 P-001: motif 子结构列表
    ) -> str:
        """
        记录一次因子发现尝试。

        Returns: 记忆 ID
        """
        attempt_id = f"mem_{len(self.data['attempts']):04d}"
        
        # ═══ 去重: 按 factor_name 检查是否已存在 ═══
        existing_idx = None
        for idx, existing in enumerate(self.data["attempts"]):
            if existing.get("factor_name") == factor_name:
                existing_idx = idx
                break
        
        if existing_idx is not None:
            # 更新已有记录（保留原ID，覆盖数据）
            existing = self.data["attempts"][existing_idx]
            attempt_id = existing["id"]
            # 更新 fields
            existing["formula"] = formula[:500]
            existing["paradigm"] = paradigm
            existing["category"] = category
            existing["outcome"] = outcome
            existing["fri"] = {
                "score": round(fri_score, 4),
                "grade": fri_grade,
                "precision": round(fri_precision, 4),
                "persistence": round(fri_persistence, 4),
                "consistency": round(fri_consistency, 4),
                "novelty": round(fri_novelty, 4),
            }
            existing["icir"] = round(icir, 4)
            existing["psi"] = {
                "r_squared": round(psi_r_squared, 4),
                "independence": round(psi_independence, 4),
                "max_corr_factor": max_corr_factor,
            }
            existing["tags"] = tags or existing.get("tags", [])
            existing["lessons"] = lessons or existing.get("lessons", [])
            existing["timestamp"] = datetime.now().isoformat()
            existing["updated_count"] = existing.get("updated_count", 0) + 1
            
            self._save()
            return attempt_id  # 更新模式: 返回已有ID, 不重复计数
        
        # 新增记录
        record = {
            "id": attempt_id,
            "factor_name": factor_name,
            "formula": formula[:500],          # 截断以防过长
            "paradigm": paradigm,
            "category": category,
            "outcome": outcome,
            "fri": {
                "score": round(fri_score, 4),
                "grade": fri_grade,
                "precision": round(fri_precision, 4),
                "persistence": round(fri_persistence, 4),
                "consistency": round(fri_consistency, 4),
                "novelty": round(fri_novelty, 4),
            },
            "icir": round(icir, 4),
            "psi": {
                "r_squared": round(psi_r_squared, 4),
                "independence": round(psi_independence, 4),
                "max_corr_factor": max_corr_factor,
            },
            "tags": tags or [],
            "lessons": lessons or [],
            "timestamp": datetime.now().isoformat(),
        }
        
        self.data["attempts"].append(record)
        
        # 更新统计
        stats = self.data["stats"]
        stats["total_attempts"] += 1
        if outcome == "PASS":
            stats["total_pass"] += 1
        elif outcome == "REJECT":
            stats["total_reject"] += 1
        
        by_p = stats["by_paradigm"]
        if paradigm not in by_p:
            by_p[paradigm] = {"total": 0, "pass": 0, "avg_fri": 0.0}
        by_p[paradigm]["total"] += 1
        if outcome == "PASS":
            by_p[paradigm]["pass"] += 1
        # 更新平均 FRI (简单累加再平均)
        n = by_p[paradigm]["total"]
        old_avg = by_p[paradigm]["avg_fri"]
        by_p[paradigm]["avg_fri"] = round((old_avg * (n - 1) + fri_score) / n, 4)
        
        # 自动提取模式
        self._auto_extract_patterns()
        
        # v3 P-001: motif 聚合 — 关联统计
        if motifs:
            self._update_motif_stats(motifs, outcome == "PASS")
        
        self._save()
        return attempt_id
    
    # ── 检索操作 ──────────────────────────────────────────
    
    def find_similar(
        self,
        paradigm: str = "",
        outcome: str = "",
        min_fri: float = 0.0,
        min_icir: float = 0.0,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """检索相似的历史记录"""
        results = []
        
        for attempt in self.data["attempts"]:
            # 范式过滤
            if paradigm and attempt["paradigm"] != paradigm:
                continue
            # 结果过滤
            if outcome and attempt["outcome"] != outcome:
                continue
            # FRI 过滤
            if min_fri > 0 and attempt["fri"]["score"] < min_fri:
                continue
            # ICIR 过滤
            if min_icir > 0 and attempt["icir"] < min_icir:
                continue
            # 标签过滤 (至少一个匹配)
            if tags:
                attempt_tags = set(attempt["tags"])
                if not attempt_tags.intersection(tags):
                    continue
            
            results.append(attempt)
        
        # 按 FRI 降序排列
        results.sort(key=lambda x: x["fri"]["score"], reverse=True)
        return results[:limit]
    
    def get_best_in_paradigm(self, paradigm: str, limit: int = 5) -> List[Dict]:
        """获取某范式下最好的因子"""
        return self.find_similar(paradigm=paradigm, outcome="PASS", limit=limit)
    
    def get_all_failures(self, paradigm: str = "", limit: int = 10) -> List[Dict]:
        """获取失败记录 (用于避免重复错误)"""
        return self.find_similar(paradigm=paradigm, outcome="REJECT", limit=limit)
    
    def get_recent(self, n: int = 10) -> List[Dict]:
        """获取最近 n 条记录"""
        attempts = sorted(
            self.data["attempts"],
            key=lambda x: x["timestamp"],
            reverse=True,
        )
        return attempts[:n]
    
    # ── LLM 上下文生成 ────────────────────────────────────
    
    def get_llm_context(
        self,
        paradigm: str = "",
        max_examples: int = 5,
        include_failures: bool = True,
    ) -> str:
        """
        生成 LLM 经验上下文。
        
        用于在 AlphaAgent v3 生成因子前注入到 prompt 中，
        帮助 LLM 了解"过去什么有效、什么无效"。
        """
        parts = []
        
        # 总体统计
        stats = self.data["stats"]
        parts.append(f"## 因子发现经验库 (共 {stats['total_attempts']} 次尝试)")
        
        if paradigm and paradigm in stats.get("by_paradigm", {}):
            ps = stats["by_paradigm"][paradigm]
            parts.append(f"- {paradigm}范式: {ps['total']} 次尝试, {ps['pass']} 次通过, 平均FRI={ps['avg_fri']:.3f}")
        
        # 成功案例
        successes = self.find_similar(
            paradigm=paradigm, outcome="PASS", min_fri=0.4, limit=max_examples
        )
        if successes:
            parts.append(f"\n### 历史成功因子 ({len(successes)} 个)")
            for s in successes:
                parts.append(
                    f"- **{s['factor_name']}** (FRI={s['fri']['score']:.3f}/{s['fri']['grade']}, "
                    f"ICIR={s['icir']:.3f}) | paradigm={s['paradigm']} | {s['formula'][:80]}"
                )
                if s.get("lessons"):
                    parts.append(f"  经验: {'; '.join(s['lessons'])}")
        
        # 失败案例
        if include_failures:
            failures = self.get_all_failures(paradigm=paradigm, limit=3)
            if failures:
                parts.append(f"\n### ⚠️ 历史失败记录 (避免重复)")
                for f_item in failures:
                    parts.append(
                        f"- ~~{f_item['factor_name']}~~ ({f_item['formula'][:60]}...)"
                    )
        
        # 自动提取的模式
        patterns = self.data.get("patterns", {})
        if patterns.get("toxic_patterns"):
            parts.append(f"\n### 🚫 已知毒性模式")
            for p in patterns["toxic_patterns"][:5]:
                parts.append(f"- {p}")

        # v3.1: 软负收益警告
        warnings = self.data.get("warning_directions", [])
        if warnings:
            parts.append(f"\n### ⚠️ JQ 软负收益警告 ({len(warnings)} 个)")
            for w in warnings[-5:]:
                name = w.get("direction_id", "").replace("jq_warning::", "")
                jq_ret = w.get("jq_return", 0)
                parts.append(f"- {name}: JQ {jq_ret:.1f}% (持续亏损, 降低探索优先级)")
        
        if patterns.get("successful_patterns"):
            parts.append(f"\n### ✅ 已知成功模式")
            for p in patterns["successful_patterns"][:5]:
                parts.append(f"- {p}")
        
        return "\n".join(parts)
    
    def get_paradigm_stats(self) -> Dict:
        """获取各范式的统计摘要"""
        return self.data["stats"].get("by_paradigm", {})
    
    # ── 模式提取 ──────────────────────────────────────────
    
    def _auto_extract_patterns(self):
        """
        自动从累积的尝试中提取模式。
        
        规则:
          - 同一范式连续 3 次 REJECT → toxic_pattern
          - 同一范式连续 3 次 PASS 且 FRI > 0.5 → successful_pattern
          - 同一标签连续 2 次 REJECT → 标签级 poisonous
        """
        attempts = self.data["attempts"]
        if len(attempts) < 3:
            return
        
        patterns = self.data["patterns"]
        
        # 按范式统计最近结果
        paradigm_recent = {}
        for a in attempts[-9:]:  # 最近 9 条
            p = a["paradigm"]
            if p not in paradigm_recent:
                paradigm_recent[p] = []
            paradigm_recent[p].append(a["outcome"])
        
        # 检测 toxic: 连续 3 次 REJECT
        for paradigm, outcomes in paradigm_recent.items():
            if len(outcomes) >= 3 and all(o == "REJECT" for o in outcomes[-3:]):
                pattern = f"{paradigm}范式: 连续 3 次失败"
                if pattern not in patterns["toxic_patterns"]:
                    patterns["toxic_patterns"].append(pattern)
        
        # 检测 successful: 连续 3 次 PASS 且 FRI > 0.5
        for paradigm, outcomes in paradigm_recent.items():
            if len(outcomes) >= 3 and all(o == "PASS" for o in outcomes[-3:]):
                recent_fris = [
                    a["fri"]["score"] for a in attempts[-3:]
                    if a["paradigm"] == paradigm and a["outcome"] == "PASS"
                ]
                if len(recent_fris) >= 3 and np.mean(recent_fris) > 0.5:
                    pattern = f"{paradigm}范式: 连续 3 次成功 (avg FRI={np.mean(recent_fris):.3f})"
                    if pattern not in patterns["successful_patterns"]:
                        patterns["successful_patterns"].append(pattern)
        
        self.data["patterns"] = patterns
    
    # ── Motif 统计 (v3 P-001) ────────────────────────────
    
    def _update_motif_stats(self, motifs: List[str], passed: bool):
        """更新 motif 级 pass/fail 统计
        
        关联统计 (frequentist): 因子通过 → 它的每个组成 motif 的 pass+1
                              因子失败 → 它的每个组成 motif 的 fail+1
        
        这不是因果归因，但在大样本下，真正的好 motif 会因出现在多个通过
        因子中而拉回成功率。我们只需要排名，不需要精确因果。
        """
        motif_stats = self.data.setdefault("motif_stats", {})
        for m in motifs:
            if m not in motif_stats:
                motif_stats[m] = {"pass": 0, "fail": 0, "jq_pass": 0, "jq_fail": 0}
            if passed:
                motif_stats[m]["pass"] += 1
            else:
                motif_stats[m]["fail"] += 1
    
    def get_motif_success(self, min_samples: int = 3) -> List[Tuple[str, Dict]]:
        """获取高成功率 motif 列表
        
        Returns
        -------
        [(motif_key, {pass, fail, rate}), ...] — 按成功率降序
        """
        motif_stats = self.data.get("motif_stats", {})
        scored = []
        for m, stats in motif_stats.items():
            total = stats["pass"] + stats["fail"]
            if total >= min_samples:
                rate = stats["pass"] / total
                scored.append((m, {**stats, "rate": rate, "total": total}))
        scored.sort(key=lambda x: x[1]["rate"], reverse=True)
        return scored
    
    def get_motif_forbidden(self, min_samples: int = 3, max_rate: float = 0.1) -> List[str]:
        """获取低成功率 motif 列表（应避开）"""
        motif_stats = self.data.get("motif_stats", {})
        forbidden = []
        for m, stats in motif_stats.items():
            total = stats["pass"] + stats["fail"]
            if total >= min_samples:
                rate = stats["pass"] / total
                if rate <= max_rate:
                    forbidden.append(m)
        return forbidden
    
    # ── D 阶段 Motif 蒸馏 (v3 P-001) ───────────────────
    
    def distill_motif_knowledge(self, formula: str, jq_passed: bool, fsa=None,
                                interpretation: Optional[Dict] = None,
                                jq_marginal: bool = False):
        """D 阶段: 将 JQ 回测结果蒸馏到 motif 级记忆
        
        在 ralph_loop.jq_feedback() 中每个因子调用一次。
        
        L1 两级蒸馏 (2026-08-14): interpretation 存在时:
          - verdict=execution_failure → motif 不记 jq_fail (执行崩不归咎 motif 结构)
          - verdict=data_issue → motif 不记 jq_fail (数据问题不归咎 motif 结构)
          - verdict=direction_falsified → 正常记 jq_fail (结构被证伪)
        
        MARGINAL 中性通道 (2026-08-15, lhb_freq_20d 案例 +67%/Sharpe 0.27):
          jq_marginal=True (收益正但未达 PASS) → 不记 jq_pass 也不记 jq_fail, 仅记注释
        
        Parameters
        ----------
        formula: 因子表达式
        jq_passed: JQ 回测是否通过 (return>50% + sharpe>0.4)
        fsa: SubtreeFingerprinter 实例 (可选，用于提取 motif)
        interpretation: L1 归因报告 (可选, verdict 分流器)
        jq_marginal: 收益正但未达 PASS 阈值 (中性, 不罚 motif)
        """
        motifs = []
        
        # 尝试从 FSA 提取 motif
        if fsa:
            try:
                fingerprints = fsa.extract_fingerprints(formula)
                motifs = [f.fingerprint_key if hasattr(f, 'fingerprint_key') else str(f) 
                         for f in fingerprints]
            except Exception:
                pass
        
        # 如果没有 FSA，从 formula 文本中提取粗略 motif
        if not motifs:
            motifs = self._extract_rough_motifs(formula)
        
        if not motifs:
            return
        
        # ── L1 两级蒸馏: verdict 分流 ──
        verdict = ""
        if interpretation:
            verdict = str(interpretation.get("verdict", ""))
        # execution_failure: 执行崩不归咎 motif (breed008 型: IC 有效但 JQ 崩)
        # data_issue: 数据问题不归咎 motif (2026-08-15)
        # jq_marginal: 收益正但未达 PASS → 中性, 不记 jq_fail (2026-08-15)
        skip_motif_penalty = (verdict in ("execution_failure", "data_issue")) or jq_marginal
        
        # 更新 motif_stats 中的 JQ 通道
        motif_stats = self.data.setdefault("motif_stats", {})
        for m in motifs:
            if m not in motif_stats:
                motif_stats[m] = {"pass": 0, "fail": 0, "jq_pass": 0, "jq_fail": 0}
            if skip_motif_penalty:
                # 执行失败/数据问题/MARGINAL → 不动 jq_pass/jq_fail, 只记注释
                notes = motif_stats[m].setdefault("interp_notes", [])
                if verdict:
                    note = f"L1:{verdict}@{interpretation.get('confidence', 0) if interpretation else 0}"
                else:
                    note = "JQ_MARGINAL(中性)"
                if note not in notes:
                    notes.append(note)
                    if len(notes) > 5:
                        motif_stats[m]["interp_notes"] = notes[-5:]
            elif jq_passed:
                motif_stats[m]["jq_pass"] += 1
            else:
                motif_stats[m]["jq_fail"] += 1
        
        # 达到统计显著后生成 R 规则
        for m, stats in list(motif_stats.items()):
            total_jq = stats["jq_pass"] + stats["jq_fail"]
            if total_jq >= 3:  # v3.1: 降低门槛 5→3, 让负收益更快触发 forbid
                rate = stats["jq_pass"] / total_jq
                if rate >= 0.6:
                    # 优先组合此 motif
                    self._add_motif_rule("prefer", m, f"JQ成功率{rate:.0%} (n={total_jq})")
                elif rate <= 0.1:
                    # 禁止此 motif
                    self._add_motif_rule("forbid", m, f"JQ成功率{rate:.0%} (n={total_jq})")
        
        self._save()
    
    @staticmethod
    def _extract_rough_motifs(formula: str) -> List[str]:
        """从公式文本中粗略提取 motif（无 FSA 时的 fallback）"""
        import re
        # 提取算子组合模式
        ops = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', formula)
        if len(ops) >= 2:
            return ["|".join(ops[:4])]  # 取前4个算子作为粗略 motif
        return [f"rough::{formula[:40]}"]
    
    def _add_motif_rule(self, rule_type: str, motif_key: str, reason: str):
        """添加 motif 级规则（prefer/forbid）"""
        rules = self.data.setdefault("motif_rules", [])
        # 去重
        for r in rules:
            if r["motif_key"] == motif_key and r["rule_type"] == rule_type:
                r["reason"] = reason
                r["updated_at"] = datetime.now().isoformat()
                return
        
        rules.append({
            "motif_key": motif_key,
            "rule_type": rule_type,
            "reason": reason,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        })
        # 限制规则数量
        if len(rules) > 100:
            self.data["motif_rules"] = rules[-50:]
    
    def get_motif_rules(self) -> Dict[str, List[Dict]]:
        """获取所有 motif 规则"""
        rules = self.data.get("motif_rules", [])
        return {
            "prefer": [r for r in rules if r["rule_type"] == "prefer"],
            "forbid": [r for r in rules if r["rule_type"] == "forbid"],
        }

    # ── L1 解读层: interpretation 存储与检索 (2026-08-14) ──

    def upsert_interpretation(self, factor_name: str, interpretation: Dict) -> bool:
        """把 L1 归因报告写回 attempts 条目 (按 factor_name 匹配)。只增不改数值结论。"""
        if not factor_name or not interpretation:
            return False
        for a in self.data.get("attempts", []):
            if a.get("factor_name") == factor_name or a.get("name") == factor_name:
                a["interpretation"] = interpretation
                self._save()
                return True
        return False

    def get_interpretations(self, paradigm: str = "", limit: int = 5) -> List[Dict]:
        """检索 L1 归因记录 (软消费: 供 llm_generator prompt 注入, 不参与排序主键)。

        Parameters
        ----------
        paradigm: 按范式过滤 (为空取全部)
        limit: 最多返回条数 (按 jq_verified_at 倒序)
        """
        out = []
        for a in self.data.get("attempts", []):
            interp = a.get("interpretation")
            if not interp:
                continue
            if paradigm and paradigm not in str(a.get("paradigm", "")):
                continue
            out.append({
                "factor_name": a.get("factor_name", a.get("name", "")),
                "paradigm": a.get("paradigm", ""),
                "jq_return": a.get("jq_return"),
                "jq_sharpe": a.get("jq_sharpe"),
                "jq_verified_at": a.get("jq_verified_at", ""),
                "interpretation": interp,
            })
        out.sort(key=lambda x: str(x.get("jq_verified_at", "")), reverse=True)
        return out[:limit]

    def build_interpretation_prompt_fragment(self, paradigm: str = "", limit: int = 5) -> str:
        """把归因记录拼成可直接注入 LLM 生成 prompt 的避坑片段 (L1 消费①)。

        返回空串表示无可用归因。格式:
          ## 上轮 JQ 归因提示 (避坑指南)
          - [执行失败] factor_x: 避免高换手构造 ...
        """
        records = self.get_interpretations(paradigm=paradigm, limit=limit)
        if not records:
            return ""
        verdict_label = {
            "execution_failure": "执行失败(方向保留)",
            "direction_falsified": "方向证伪",
            "direction_confirmed": "方向确认",
            "data_issue": "数据问题",
        }
        parts = ["## 🧭 上轮 JQ 归因提示 (避坑指南)", ""]
        for r in records:
            interp = r.get("interpretation", {})
            v = interp.get("verdict", "?")
            label = verdict_label.get(v, v)
            hints = " | ".join(str(h) for h in interp.get("next_round_hints", [])[:3])
            jq_ret = r.get("jq_return")
            ret_s = f"{jq_ret:+.1f}%" if isinstance(jq_ret, (int, float)) else "?"
            parts.append(f"- [{label}] {r.get('factor_name', '?')} (JQ {ret_s}): {hints}")
        parts.append("")
        parts.append("生成新因子时必须规避上述坑, 若命中 execution_failure 应聚焦执行层而非换方向。")
        parts.append("")
        return "\n".join(parts)
    
    # ── 禁止区域管理 (FactorMiner v4 新增) ──────────────

    def add_forbidden_region(
        self,
        paradigm: str,
        reason: str,
        prototype_expression: str = "",
        severity: str = "soft",
    ):
        """
        添加禁止区域 — 标记某个范式/方向已被充分覆盖或验证失败。

        禁止区域用于:
        - 引导LLM避开已证伪的方向
        - 避免重复探索已被覆盖的因子空间
        - 实现 FactorMiner 的 "Correlation Red Sea" 管理
        """
        if "forbidden_regions" not in self.data:
            self.data["forbidden_regions"] = []

        # 去重
        for fr in self.data["forbidden_regions"]:
            if fr["paradigm"] == paradigm and fr.get("reason") == reason:
                fr["count"] = fr.get("count", 0) + 1
                self._save()
                return

        self.data["forbidden_regions"].append({
            "paradigm": paradigm,
            "reason": reason,
            "prototype_expression": prototype_expression[:200],
            "severity": severity,
            "added_at": datetime.now().isoformat(),
            "count": 1,
        })
        self._save()

    def get_forbidden_regions(self, paradigm: str = "") -> List[Dict]:
        """获取当前禁止区域"""
        regions = self.data.get("forbidden_regions", [])
        if paradigm:
            regions = [r for r in regions if r["paradigm"] == paradigm]
        return regions

    # ── v3.1: WarningDirection 软负收益记录层 ────────────

    # ── v0.6 P-020: Memory Bridge — 高 occ 模板 → GP 育种种子 ──

    def get_unconsumed_bridge_templates(
        self, min_occurrence: int = 100
    ) -> List[Dict]:
        """
        获取未消费的高频率成功模板，供 Memory Bridge 转化为 GP 育种种子。

        筛选条件:
        - occurrence_count >= min_occurrence (默认100, Memory中反复出现的成功模式)
        - bridge_consumed = False (未被消费过)
        - 有 formula 字段 (可转化为实际因子表达式)

        Returns
        -------
        [{pattern_id, description, formula, paradigm, occurrence_count, ...}, ...]
        """
        templates = self.data.get("success_templates", [])
        candidates = []
        for t in templates:
            occ = t.get("occurrence_count", 0)
            consumed = t.get("bridge_consumed", False)
            formula = t.get("formula", "")
            if occ >= min_occurrence and not consumed and formula:
                candidates.append(t)
        return candidates

    def mark_bridge_consumed(self, pattern_id: str) -> bool:
        """标记模板已通过 Memory Bridge 消费"""
        templates = self.data.get("success_templates", [])
        for t in templates:
            if t.get("pattern_id") == pattern_id:
                t["bridge_consumed"] = True
                t["bridge_consumed_at"] = datetime.now().isoformat()
                self._save()
                return True
        return False

    # ── v3.1: WarningDirection 软负收益记录层 ────────────

    def get_warning_directions(self, paradigm: str = "") -> List[Dict]:
        """获取软负收益警告记录（severity=soft 层）

        用于 MAB 方向选择时降权：若某范式有活跃 warning，给予负向先验。
        """
        warnings = self.data.get("warning_directions", [])
        if paradigm:
            warnings = [w for w in warnings if paradigm in w.get("description", "")]
        return warnings

    def get_warning_paradigms(self) -> Dict[str, int]:
        """按范式统计 warning 数量 → {paradigm: warning_count}

        供 ralph_loop._mab_select_direction() 使用。
        范式名从 description 字段的 [范式名] 前缀中提取。
        """
        import re
        from collections import Counter
        warnings = self.data.get("warning_directions", [])
        paradigm_counts = Counter()
        for w in warnings:
            desc = w.get("description", "")
            # 提取 [范式名] 前缀 (v3.1 格式: "[动量] JQ软负收益: ...")
            m = re.match(r'\[([^\]]+)\]', desc)
            if m:
                paradigm_counts[m.group(1)] += 1
            else:
                # fallback: 用因子名
                w_name = w.get("direction_id", "").replace("jq_warning::", "")
                paradigm_counts[w_name] += 1
        return dict(paradigm_counts)

    def get_warning_context_for_llm(self) -> str:
        """生成软负收益的 LLM 上下文 — 告知 LLM 哪些方向有持续亏损风险"""
        warnings = self.data.get("warning_directions", [])
        if not warnings:
            return ""

        lines = ["## ⚠️ 软负收益警告 (JQ回测负收益但未触发硬禁止)"]
        lines.append("")
        lines.append("以下因子在JQ回测中产生负收益，建议降低其范式/结构的探索优先级：")

        for w in warnings[-10:]:  # 最近 10 条
            name = w.get("direction_id", "").replace("jq_warning::", "")
            jq_ret = w.get("jq_return", 0)
            jq_sharpe = w.get("jq_sharpe", 0)
            reason = w.get("reason", "")
            lines.append(f"- ⚠️ [{name}] JQ {jq_ret:.1f}% / Sharpe {jq_sharpe:.2f} ({reason})")
            if w.get("prototype_expression"):
                lines.append(f"  公式: `{w['prototype_expression'][:100]}`")

        return "\n".join(lines)

    def get_forbidden_context_for_llm(self) -> str:
        """
        生成禁止区域的LLM上下文 — 告知LLM哪些方向不要碰。

        这是FactorMiner的核心功能: 经验记忆不只是"什么成功过"，
        更要记录"什么方向已经撞墙"。
        """
        regions = self.data.get("forbidden_regions", [])
        toxic = self.data.get("patterns", {}).get("toxic_patterns", [])

        if not regions and not toxic:
            return ""

        lines = ["## ⚠️ 禁止区域 (避免重复已验证无效的方向)"]
        lines.append("")
        lines.append("以下方向已被充分验证为无效或低效，请避免生成类似因子:")

        for fr in regions[-10:]:  # 最近10条
            icon = "🚫" if fr.get("severity") == "hard" else "⚠️"
            lines.append(f"- {icon} [{fr['paradigm']}] {fr['reason']}")
            if fr.get("prototype_expression"):
                lines.append(f"  典型表达式: `{fr['prototype_expression'][:100]}`")

        if toxic:
            lines.append("")
            lines.append("### 已知毒性模式 (连续失败)")
            for t in toxic[-5:]:
                lines.append(f"- {t}")

        return "\n".join(lines)

    # ── 范式覆盖率分析 ──────────────────────────────────

    def get_paradigm_coverage_context(self) -> str:
        """
        生成范式覆盖率的LLM上下文 — 告知LLM哪些范式已饱和、哪些待探索。

        实现全局因子库视角: 每轮生成时考虑整个库的结构。
        """
        stats = self.data["stats"]
        by_p = stats.get("by_paradigm", {})

        lines = ["## 范式覆盖状态"]
        lines.append("")

        from paradigm_v4 import PARADIGMS_V4

        for paradigm, info in PARADIGMS_V4.items():
            ps = by_p.get(paradigm, {"total": 0, "pass": 0, "avg_fri": 0})
            total = ps["total"]
            if total == 0:
                lines.append(f"- 🔵 [{paradigm}] 未探索 — **高优先级**")
            elif total <= 3:
                lines.append(f"- 🟢 [{paradigm}] 探索中 ({total}次, {ps['pass']}通过, avg FRI={ps['avg_fri']:.3f})")
            elif total <= 8:
                lines.append(f"- 🟡 [{paradigm}] 较充分 ({total}次, {ps['pass']}通过)")
            else:
                # 检查是否已有禁止区域
                forbidden = [fr for fr in self.data.get("forbidden_regions", [])
                            if fr["paradigm"] == paradigm and fr.get("severity") == "hard"]
                if forbidden:
                    lines.append(f"- 🔴 [{paradigm}] 已饱和+硬禁止 ({total}次)")
                else:
                    lines.append(f"- 🟠 [{paradigm}] 已饱和 ({total}次) — 低优先级")

        # 添加禁止区域提示
        forbidden = self.get_forbidden_regions()
        if forbidden:
            lines.append("")
            lines.append(f"当前活跃禁止区域: {len(forbidden)}个")
            hard_forbidden = [fr for fr in forbidden if fr.get("severity") == "hard"]
            if hard_forbidden:
                lines.append(f"  🚫 硬禁止: {', '.join(fr['paradigm'] for fr in hard_forbidden)}")

        return "\n".join(lines)


    # ═══════════════════════════════════════════════════════════
    # F/E/R 操作符 (v2 核心 — 对标 FactorMiner Section 3.3)
    # ═══════════════════════════════════════════════════════════

    # ── F: Formation — 从轨迹提取经验 ──────────────────

    def form(
        self,
        trajectory: Dict,
        auto_save: bool = True,
    ) -> Dict:
        """
        F (Formation) 操作符: 从一批挖掘轨迹中提取结构化经验。

        对标 FactorMiner Eq.7: M_form = F(M_t, τ_t)

        从轨迹(batch)中:
        - 通过 IC 筛选 + 无高相关 → 提取 P_succ (SuccessPattern)
        - 高相关被拒 → 提取 P_fail (ForbiddenDirection)
        - IC 不达标 → 记录到 attempts

        Parameters
        ----------
        trajectory: {
            "batch_id": str,
            "candidates": [
                {
                    "factor_name": str,
                    "formula": str,
                    "hypothesis": str,
                    "paradigm": str,
                    "ic": float,
                    "icir": float,
                    "max_corr": float,
                    "max_corr_factor": str,
                    "fri_score": float,
                    "outcome": str,  # PASS/REJECT/WEAK
                    "operators_used": List[str],  # AST 中使用的算子
                },
                ...
            ]
        }
        auto_save: 是否自动保存

        Returns
        -------
        {
            "new_success_patterns": int,
            "new_forbidden_directions": int,
            "new_attempts": int,
        }
        """
        batch_id = trajectory.get("batch_id", f"batch_{self._stats['formations']}")
        candidates = trajectory.get("candidates", [])

        result = {
            "new_success_patterns": 0,
            "new_forbidden_directions": 0,
            "new_attempts": 0,
        }

        for cand in candidates:
            # 记录为 attempt
            self.record(
                factor_name=cand.get("factor_name", "unknown"),
                formula=cand.get("formula", ""),
                paradigm=cand.get("paradigm", "通用"),
                category=cand.get("category", ""),
                fri_score=cand.get("fri_score", 0),
                icir=cand.get("icir", 0),
                outcome=cand.get("outcome", "WEAK"),
                max_corr_factor=cand.get("max_corr_factor", ""),
                tags=cand.get("tags", []),
                lessons=cand.get("lessons", []),
            )
            result["new_attempts"] += 1

            # 提取 SuccessPattern (IC达标 + 无高相关)
            if cand.get("outcome") == "PASS" and cand.get("icir", 0) > 0.3:
                self._upsert_success_pattern(cand)
                result["new_success_patterns"] += 1

            # 提取 ForbiddenDirection (高相关被拒)
            if cand.get("outcome") in ("REJECT", "WEAK") and cand.get("max_corr", 0) > 0.5:
                self._upsert_forbidden_direction(cand)
                result["new_forbidden_directions"] += 1

        # 日志
        self._log_event("formation", {
            "batch_id": batch_id,
            "candidate_count": len(candidates),
            **result,
        })

        if auto_save:
            self._save()

        self._stats["formations"] += 1
        return result

    # ── F-JQ: JQ 回测驱动的模式提取 ────────────────────

    def form_from_jq(
        self,
        jq_results: Dict,
        auto_save: bool = True,
    ) -> Dict:
        """
        JQ 回测结果驱动的模式提取。对标 FactorMiner 的"真实反馈"机制。

        这是 D 阶段最关键的增强: local IC 是代理指标，JQ 回测才是真相源。
        - JQ 正收益 + 合理 Sharpe → 升级为 JQ-confirmed SuccessPattern
        - JQ 负收益 → 创建 hard ForbiddenDirection（权重远高于 local 禁止）

        Parameters
        ----------
        jq_results: {
            "batch_id": str,
            "timestamp": str,
            "composite_return": float,   # 复合策略总收益 %
            "composite_sharpe": float,
            "composite_maxdd": float,
            "factors": [
                {
                    "factor_name": str,
                    "formula": str,
                    "hypothesis": str,
                    "paradigm": str,
                    "category": str,
                    "operators_used": List[str],
                    "local_ic": float,
                    "local_icir": float,
                    "jq_return": float,
                    "jq_sharpe": float,
                    "jq_maxdd": float,
                    "jq_composite_contribution": str,  # "positive"/"negative"/"neutral"
                    "root_cause": str,  # JQ失败根因分析
                },
                ...
            ]
        }
        auto_save: 是否自动保存

        Returns
        -------
        {
            "jq_success_patterns": int,
            "jq_forbidden_directions": int,
            "trajectories_updated": int,
        }
        """
        batch_id = jq_results.get("batch_id", f"jq_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        factors = jq_results.get("factors", [])
        composite = {
            "return": jq_results.get("composite_return", 0),
            "sharpe": jq_results.get("composite_sharpe", 0),
            "maxdd": jq_results.get("composite_maxdd", 0),
        }

        result = {
            "jq_success_patterns": 0,
            "jq_forbidden_directions": 0,
            "trajectories_updated": 0,
            "composite_assessment": "",
        }

        # 复合策略整体评估
        if composite["return"] > 50 and composite["sharpe"] > 0.4:
            result["composite_assessment"] = "STRONG_POSITIVE"
        elif composite["return"] < -20 or composite["sharpe"] < -0.5:
            result["composite_assessment"] = "STRONG_NEGATIVE"
        else:
            result["composite_assessment"] = "MIXED"

        for f in factors:
            name = f.get("factor_name", "unknown")
            jq_ret = f.get("jq_return", 0)
            jq_sharpe = f.get("jq_sharpe", 0)
            jq_mdd = f.get("jq_maxdd", 0)

            # ── 更新 attempt 记录，写入 JQ 结果 ──────────
            existing = None
            for a in self.data.get("attempts", []):
                if a.get("factor_name") == name or a.get("name") == name:
                    existing = a
                    break

            if existing:
                existing["jq_return"] = jq_ret
                existing["jq_sharpe"] = jq_sharpe
                existing["jq_maxdd"] = jq_mdd
                existing["jq_verified"] = True
                existing["jq_verified_at"] = datetime.now().isoformat()
                if f.get("root_cause"):
                    existing.setdefault("lessons", [])
                    existing["lessons"].append(f"JQ反馈: {f['root_cause']}")
                result["trajectories_updated"] += 1

                # 更新 outcome 为 JQ 验证后的最终状态
                if jq_ret < -20 or jq_sharpe < -0.5:
                    existing["outcome"] = "JQ_FAILED"
                    existing["tags"] = existing.get("tags", []) + ["jq_failed"]
                elif jq_ret < 0:
                    # v3.1: 软负收益 — 未达硬禁止阈值但持续亏损
                    existing["outcome"] = "JQ_WEAK_NEGATIVE"
                    existing["tags"] = existing.get("tags", []) + ["jq_soft_negative"]
                elif jq_ret > 50 and jq_sharpe > 0.4:
                    existing["outcome"] = "JQ_PASSED"

            # ── JQ 失败 → Hard ForbiddenDirection ──────────
            if jq_ret < -20 or jq_sharpe < -0.5:
                direction_id = f"jq_failure::{name}"
                # 检查去重
                directions = self.data.setdefault("forbidden_directions", [])
                dup = False
                for d in directions:
                    if d.get("direction_id") == direction_id:
                        d["failed_attempts"] = d.get("failed_attempts", 0) + 1
                        d["severity"] = "hard"
                        d["jq_return"] = jq_ret
                        d["updated_at"] = datetime.now().isoformat()
                        dup = True
                        break

                if not dup:
                    _hyp = f.get("hypothesis", "")
                    _hyp_s = _hyp.get("content", "") if isinstance(_hyp, dict) else str(_hyp)
                    direction = ForbiddenDirection(
                        direction_id=direction_id,
                        description=(
                            f"JQ实盘失败: {name} → {jq_ret:.1f}%/{jq_sharpe:.2f}/MDD{jq_mdd:.1f}%. "
                            f"根因: {f.get('root_cause', '未知')}. "
                            f"{_hyp_s[:80]}"
                        ),
                        correlated_factors=[],
                        correlation_threshold=0.0,
                        prototype_expression=f.get("formula", "")[:200],
                        reason="jq_backtest_failure",
                        severity="hard",
                        failed_attempts=1,
                        jq_return=jq_ret,
                        jq_sharpe=jq_sharpe,
                        added_at=datetime.now().isoformat(),
                        updated_at=datetime.now().isoformat(),
                    )
                    directions.append(direction.to_dict())
                    result["jq_forbidden_directions"] += 1

            # ── v3.1: 软负收益 → WarningDirection (jq_ret < 0, 未达 hard 禁止阈值) ──
            elif jq_ret < 0 and (jq_ret >= -20 and jq_sharpe >= -0.5):
                warning_id = f"jq_warning::{name}"
                warnings = self.data.setdefault("warning_directions", [])
                dup = False
                for w in warnings:
                    if w.get("direction_id") == warning_id:
                        w["failed_attempts"] = w.get("failed_attempts", 0) + 1
                        w["jq_return"] = min(w.get("jq_return", 0), jq_ret)
                        w["updated_at"] = datetime.now().isoformat()
                        dup = True
                        break

                if not dup:
                    paradigm = f.get("paradigm", "未知")
                    _hyp = f.get("hypothesis", "")
                    _hyp_s = _hyp.get("content", "") if isinstance(_hyp, dict) else str(_hyp)
                    warning = ForbiddenDirection(
                        direction_id=warning_id,
                        description=(
                            f"[{paradigm}] JQ软负收益: {name} → {jq_ret:.1f}%/{jq_sharpe:.2f}/MDD{jq_mdd:.1f}%. "
                            f"未达硬禁止阈值但持续亏损. "
                            f"{_hyp_s[:80]}"
                        ),
                        correlated_factors=[],
                        correlation_threshold=0.0,
                        prototype_expression=f.get("formula", "")[:200],
                        reason="jq_soft_negative",
                        severity="soft",
                        failed_attempts=1,
                        jq_return=jq_ret,
                        jq_sharpe=jq_sharpe,
                        added_at=datetime.now().isoformat(),
                        updated_at=datetime.now().isoformat(),
                    )
                    warnings.append(warning.to_dict())
                    result.setdefault("jq_warning_directions", 0)
                    result["jq_warning_directions"] += 1

            # ── JQ 成功 → JQ-confirmed SuccessPattern ──────
            elif jq_ret > 50 and jq_sharpe > 0.4:
                paradigm = f.get("paradigm", "通用")
                operators = f.get("operators_used", [])
                pattern_key = f"jq_confirmed::{paradigm}"

                # 检查去重
                templates = self.data.setdefault("success_templates", [])
                dup = False
                for t in templates:
                    if t.get("pattern_id") == pattern_key:
                        t["occurrence_count"] = t.get("occurrence_count", 0) + 1
                        # 保留最强 JQ 证据: 新证据更弱时不覆盖 (2026-08-29 修复 ts_std 91.91 覆盖 gp_breed_000 173.32 事故)
                        if jq_ret > t.get("jq_return", 0):
                            t["jq_return"] = jq_ret
                            t["jq_sharpe"] = jq_sharpe
                        t["updated_at"] = datetime.now().isoformat()
                        dup = True
                        break

                if not dup:
                    _hyp = f.get("hypothesis", paradigm)
                    _hyp_s = _hyp.get("content", "") if isinstance(_hyp, dict) else str(_hyp)
                    template = SuccessPattern(
                        pattern_id=pattern_key,
                        description=f"[JQ已验证] {_hyp_s} — JQ {jq_ret:.1f}%/Sharpe {jq_sharpe:.2f}",
                        typical_operators=operators[:8],
                        typical_windows=[20, 60],
                        success_rate=1.0,
                        ic_range=(0.03, 0.10),
                        icir_range=(0.4, 0.9),
                        sample_factor_ids=[name],
                        jq_return=jq_ret,
                        jq_sharpe=jq_sharpe,
                        lessons=[f"JQ回测验证通过: {f.get('root_cause', '')}"],
                        created_at=datetime.now().isoformat(),
                        updated_at=datetime.now().isoformat(),
                        occurrence_count=1,
                        verification_level="jq_single",  # v0.8: 单因子 JQ 验证通过
                    )
                    templates.append(template.to_dict())
                    result["jq_success_patterns"] += 1

        # 保存
        self._log_event("jq_formation", {
            "batch_id": batch_id,
            "composite": composite,
            "n_factors": len(factors),
            **result,
        })

        if auto_save:
            self._save()

        self._stats["formations"] += 1
        return result

    def _upsert_success_pattern(self, cand: Dict):
        """插入或更新 SuccessPattern"""
        paradigm = cand.get("paradigm", "通用")
        operators = cand.get("operators_used", [])
        key = f"{paradigm}::{'+'.join(sorted(operators[:3]))}"

        # 检查是否已存在相似模板
        templates = self.data.setdefault("success_templates", [])
        for t in templates:
            if t.get("pattern_id") == key:
                t["occurrence_count"] = t.get("occurrence_count", 0) + 1
                t["success_rate"] = min(1.0, (t["success_rate"] * (t["occurrence_count"] - 1) + 1.0) / t["occurrence_count"])
                t["updated_at"] = datetime.now().isoformat()
                # v0.8: 验证等级只升不降 (jq_single 不被 s5_passed 覆盖)
                cur_rank = VERIFICATION_RANK.get(t.get("verification_level", "stage2_only"), 0)
                new_lvl = cand.get("verification_level", "s5_passed")
                new_rank = VERIFICATION_RANK.get(new_lvl, 1)
                if new_rank > cur_rank:
                    t["verification_level"] = new_lvl
                return

        # 新建模板
        template = SuccessPattern(
            pattern_id=key,
            description=f"{paradigm}: {'+'.join(operators[:3])} 组合模板",
            typical_operators=operators[:5],
            typical_windows=[int(w) for w in cand.get("windows", [20])],
            success_rate=1.0,
            ic_range=(cand.get("ic", 0.02), cand.get("ic", 0.08)),
            icir_range=(cand.get("icir", 0.3), cand.get("icir", 0.8)),
            sample_factor_ids=[cand.get("factor_name", "")],
            lessons=cand.get("lessons", []),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            occurrence_count=1,
            verification_level=cand.get("verification_level", "s5_passed"),  # v0.8
        )
        templates.append(template.to_dict())

    def _upsert_forbidden_direction(self, cand: Dict):
        """插入或更新 ForbiddenDirection"""
        max_corr = cand.get("max_corr", 0)
        corr_factor = cand.get("max_corr_factor", "")

        # 检查是否已存在
        directions = self.data.setdefault("forbidden_directions", [])
        for d in directions:
            if corr_factor and corr_factor in d.get("correlated_factors", []):
                d["failed_attempts"] = d.get("failed_attempts", 0) + 1
                if d["failed_attempts"] >= 3:
                    d["severity"] = "hard"  # 连续失败升级为硬禁止
                return

        direction = ForbiddenDirection(
            direction_id=f"fd_{len(directions):04d}",
            description=f"{cand.get('paradigm', '')}: {cand.get('factor_name', '')} 与 {corr_factor} 高相关",
            correlated_factors=[corr_factor] if corr_factor else [],
            correlation_threshold=max_corr,
            prototype_expression=cand.get("formula", "")[:200],
            reason="high_correlation",
            severity="soft",
            failed_attempts=1,
            added_at=datetime.now().isoformat(),
        )
        directions.append(direction.to_dict())

    def _migrate_v1_patterns(self) -> int:
        """将 v1 格式 patterns 迁移到 v2 结构化格式。
        
        v1 格式: patterns.successful_patterns = ["范式: 描述..."]
        v2 格式: success_templates = [SuccessPattern(...)]
        
        基于 attempts 数据补充统计信息。
        """
        migrated = 0
        attempts = self.data.get("attempts", [])
        v1_patterns = self.data.get("patterns", {}).get("successful_patterns", [])
        
        # 按范式分组统计 success attempts
        by_paradigm = defaultdict(lambda: {"pass": [], "total": 0, "icirs": [], "operators": set()})
        for a in attempts:
            p = a.get("paradigm", "通用")
            by_paradigm[p]["total"] += 1
            if a.get("outcome") == "PASS":
                by_paradigm[p]["pass"].append(a)
                if a.get("icir"):
                    by_paradigm[p]["icirs"].append(a["icir"])
                # 尝试从 formula 中提取算子
                formula = a.get("formula", "")
                ops = self._extract_operators(formula)
                by_paradigm[p]["operators"].update(ops)
        
        # 为每个有成功记录的范式创建 SuccessPattern
        for paradigm, stats in by_paradigm.items():
            if not stats["pass"]:
                continue
            n_pass = len(stats["pass"])
            success_rate = n_pass / stats["total"] if stats["total"] > 0 else 0
            avg_icir = np.mean(stats["icirs"]) if stats["icirs"] else 0.3
            avg_ic = avg_icir * 0.08  # IC ≈ ICIR × σ(IC), 典型σ≈0.08
            
            template = SuccessPattern(
                pattern_id=f"{paradigm}::success_template_v1",
                description=f"{paradigm}范式: {n_pass}次成功, avg_ICIR={avg_icir:.3f}",
                typical_operators=list(stats["operators"])[:8] if stats["operators"] else [],
                typical_windows=[20, 60],
                success_rate=success_rate,
                ic_range=(max(0.01, avg_ic - 0.02), min(0.08, avg_ic + 0.02)),
                icir_range=(max(0.15, avg_icir - 0.2), min(0.8, avg_icir + 0.2)),
                occurrence_count=n_pass,
                lessons=[f"v1迁移: {v1_patterns[i]}" for i in range(min(len(v1_patterns), 2))],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )
            self.data.setdefault("success_templates", []).append(template.to_dict())
            migrated += 1
        
        # 从 toxic_patterns 迁移为 forbidden_directions
        toxic = self.data.get("patterns", {}).get("toxic_patterns", [])
        for t in toxic:
            direction = ForbiddenDirection(
                direction_id=f"v1_migration::{t[:30]}",
                description=f"[v1迁移] {t}",
                severity="soft",
                reason="v1 toxic pattern",
                failed_attempts=1,
                added_at=datetime.now().isoformat(),
            )
            self.data.setdefault("forbidden_directions", []).append(direction.to_dict())
        
        if migrated > 0:
            self._log_event("migration", {"v1_to_v2": migrated, "patterns_count": len(v1_patterns)})
        
        return migrated
    
    @staticmethod
    def _extract_operators(formula: str) -> List[str]:
        """从公式中提取算子名称"""
        import re
        known_ops = [
            "ts_delta", "ts_std", "ts_mean", "ts_sum", "ts_max", "ts_min",
            "ts_rank", "ts_corr", "ts_cov", "ts_argmax", "ts_argmin",
            "rank", "delay", "delta", "correlation", "covariance",
            "scale", "signed_power", "log", "abs", "sign",
            "rolling", "pct_change", "shift", "diff", "expanding",
            "clip", "where", "np.where", "groupby",
        ]
        found = []
        for op in known_ops:
            if op in formula or op.replace("np.", "") in formula:
                found.append(op)
        return found

    # ── E: Evolution — 合并冗余 / 淘汰低效用 ────────────

    def evolve(self, auto_save: bool = True) -> Dict:
        """
        E (Evolution) 操作符: 维护记忆库健康，合并冗余，淘汰低效用信息。

        对标 FactorMiner Eq.8: M_{t+1} = E(M_t, M_form)

        自动操作:
        - SuccessPattern 去重: 合并描述相似的模板 (operator overlap > 67%)
        - ForbiddenDirection 升级: failed_attempts ≥ 3 → severity = "hard"
        - 低效用淘汰: occurrence_count ≤ 1 且 30天未更新 → 归档
        - 范式洞察更新

        Returns
        -------
        {
            "merged_patterns": int,
            "upgraded_forbidden": int,
            "archived_stale": int,
        }
        """
        result = {"merged_patterns": 0, "upgraded_forbidden": 0, "archived_stale": 0, "v1_migrated": 0, "dedup_attempts": 0}

        # 0. v1→v2 迁移: 如果新格式为空但旧 patterns 有数据，自动迁移
        if not self.data.get("success_templates") and self.data.get("patterns", {}).get("successful_patterns"):
            result["v1_migrated"] = self._migrate_v1_patterns()

        # 0.5. 去重: 清理 attempts 中按 factor_name 的重复记录
        seen_names = {}
        dedup_attempts = []
        for attempt in self.data.get("attempts", []):
            name = attempt.get("factor_name", "")
            if name and name in seen_names:
                # 保留最新的（timestamp 更大）
                existing_ts = seen_names[name].get("timestamp", "")
                new_ts = attempt.get("timestamp", "")
                if new_ts > existing_ts:
                    # 替换旧记录
                    dedup_attempts[dedup_attempts.index(seen_names[name])] = attempt
                    seen_names[name] = attempt
            else:
                seen_names[name] = attempt
                dedup_attempts.append(attempt)
        removed = len(self.data.get("attempts", [])) - len(dedup_attempts)
        if removed > 0:
            self.data["attempts"] = dedup_attempts
            # 重新统计
            self.data["stats"]["total_attempts"] = len(dedup_attempts)
            self.data["stats"]["total_pass"] = sum(1 for a in dedup_attempts if a.get("outcome") == "PASS")
            self.data["stats"]["total_reject"] = len(dedup_attempts) - self.data["stats"]["total_pass"]
            result["dedup_attempts"] = removed

        # 1. SuccessPattern 去重合并
        templates = self.data.get("success_templates", [])
        merged_indices = set()
        for i in range(len(templates)):
            if i in merged_indices:
                continue
            for j in range(i + 1, len(templates)):
                if j in merged_indices:
                    continue
                # 计算算子重叠度
                ops_i = set(templates[i].get("typical_operators", []))
                ops_j = set(templates[j].get("typical_operators", []))
                if not ops_i or not ops_j:
                    continue
                overlap = len(ops_i & ops_j) / max(len(ops_i | ops_j), 1)
                if overlap > 0.67:  # 67% 重叠 → 合并
                    # 保留 occurrence 更多的
                    if templates[i].get("occurrence_count", 0) >= templates[j].get("occurrence_count", 0):
                        templates[i]["occurrence_count"] += templates[j].get("occurrence_count", 0)
                        templates[i]["updated_at"] = datetime.now().isoformat()
                        merged_indices.add(j)
                    else:
                        templates[j]["occurrence_count"] += templates[i].get("occurrence_count", 0)
                        templates[j]["updated_at"] = datetime.now().isoformat()
                        merged_indices.add(i)

        if merged_indices:
            new_templates = [t for idx, t in enumerate(templates) if idx not in merged_indices]
            self.data["success_templates"] = new_templates
            result["merged_patterns"] = len(merged_indices)

        # 2. ForbiddenDirection 升级
        for d in self.data.get("forbidden_directions", []):
            if d.get("failed_attempts", 0) >= 3 and d.get("severity") != "hard":
                d["severity"] = "hard"
                result["upgraded_forbidden"] += 1

        # 3. 淘汰 30 天未更新的低效用模板
        now = datetime.now()
        stale_threshold = now.isoformat()
        # 用字符串比较做近似（iso日期时间格式自然有序）
        new_templates = []
        for t in self.data.get("success_templates", []):
            updated = t.get("updated_at", "")
            if updated:
                try:
                    days_ago = (now - datetime.fromisoformat(updated)).days
                    if t.get("occurrence_count", 0) <= 1 and days_ago > 30:
                        result["archived_stale"] += 1
                        continue
                except Exception:
                    pass
            new_templates.append(t)
        self.data["success_templates"] = new_templates

        # 日志
        self._log_event("evolution", result)

        if auto_save:
            self._save()

        self._stats["evolutions"] += 1
        return result

    # ── R: Retrieval — 上下文检索 ────────────────────────

    def retrieve(
        self,
        library_context: Optional[Dict] = None,
        paradigm: str = "",
        k: int = 5,
    ) -> Dict:
        """
        R (Retrieval) 操作符: 基于当前库状态检索上下文相关记忆信号。

        对标 FactorMiner Eq.9: m_t = R(M_t, L_t)

        Parameters
        ----------
        library_context: 当前因子库状态 (来自 LibraryOrthogonalityManager 或手动构建)
            {
                "total_factors": int,
                "paradigm_coverage": Dict[str, int],
                "red_sea_level": str,       # green/elevated/warning/critical
                "n_forbidden_regions": int,
                "uncovered_paradigms": List[str],
            }
        paradigm: 指定检索范式 (为空则检索所有)
        k: 返回 top-k 条结果

        Returns
        -------
        {
            "success_templates": List[SuccessPattern],    # 可复用的成功模板
            "forbidden_directions": List[ForbiddenDirection],  # 需避开的禁止方向
            "exploration_priorities": List[Dict],          # 优先级排序的探索建议
            "llm_prompt_fragment": str,                    # 可直接注入 LLM prompt 的片段
        }
        """
        # 1. 检索成功模板
        templates = self.data.get("success_templates", [])
        if paradigm:
            templates = [t for t in templates if paradigm in t.get("pattern_id", "")]
        # v0.8: 验证等级主键排序 (jq_single > jq_composite > s5_passed > stage2_only),
        #       occurrence 次键 — 防止零验证模板靠 occ 累积顶替高验证模板
        templates.sort(
            key=lambda t: (
                VERIFICATION_RANK.get(t.get("verification_level", "stage2_only"), 0),
                t.get("occurrence_count", 0),
                t.get("success_rate", 0),
            ),
            reverse=True,
        )
        top_templates = [SuccessPattern.from_dict(t) for t in templates[:k]]

        # 2. 检索禁止方向
        directions_raw = self.data.get("forbidden_directions", [])
        # v1 兼容: 也包含旧的 forbidden_regions
        for fr in self.data.get("forbidden_regions", []):
            if paradigm and fr.get("paradigm", "") != paradigm:
                continue
            directions_raw.append({
                "direction_id": fr.get("paradigm", ""),
                "description": f"{fr.get('paradigm', '')}: {fr.get('reason', '')}",
                "severity": fr.get("severity", "soft"),
                "failed_attempts": fr.get("count", 1),
                "reason": fr.get("reason", ""),
            })

        if paradigm:
            directions_raw = [d for d in directions_raw if paradigm in d.get("description", "") or paradigm in d.get("direction_id", "")]
        directions_raw.sort(key=lambda d: (0 if d.get("severity") == "hard" else 1, -d.get("failed_attempts", 0)))
        top_directions = [ForbiddenDirection.from_dict(d) for d in directions_raw[:k]]

        # 3. 生成探索优先级 (结合 library context)
        priorities = self._compute_exploration_priorities(
            library_context or {},
            top_templates,
            top_directions,
        )

        # 4. 生成 LLM prompt 片段
        prompt_fragment = self._build_retrieval_prompt(
            top_templates, top_directions, priorities
        )

        # 5. v3 P-001: motif 级统计
        motif_success = self.get_motif_success(min_samples=3)
        motif_forbidden = self.get_motif_forbidden(min_samples=3)
        motif_rules = self.get_motif_rules()

        # 日志
        self._log_event("retrieval", {
            "paradigm": paradigm,
            "k": k,
            "templates_found": len(top_templates),
            "directions_found": len(top_directions),
            "motif_success_count": len(motif_success),
            "motif_forbidden_count": len(motif_forbidden),
        })
        self._stats["retrievals"] += 1

        return {
            "success_templates": top_templates,
            "forbidden_directions": top_directions,
            "exploration_priorities": priorities,
            "llm_prompt_fragment": prompt_fragment,
            "motif_success": motif_success,          # v3: [(motif, stats), ...]
            "motif_forbidden": motif_forbidden,       # v3: [motif_key, ...]
            "motif_rules": motif_rules,              # v3: {prefer: [...], forbid: [...]}
        }

    def _compute_exploration_priorities(
        self,
        library: Dict,
        templates: List[SuccessPattern],
        directions: List[ForbiddenDirection],
    ) -> List[Dict]:
        """计算因子探索的优先级排序"""
        # 1. 找未覆盖范式 (library)
        uncovered = library.get("uncovered_paradigms", [])
        paradigm_cov = library.get("paradigm_coverage", {})

        priorities = []
        if uncovered:
            for p in uncovered:
                priorities.append({
                    "paradigm": p,
                    "priority": "highest",
                    "reason": "完全未覆盖，新范式开拓",
                    "suggested_templates": [t.pattern_id for t in templates[:2]],
                })

        # 2. 找有成功模板但覆盖不足的范式
        for t in templates:
            paradigm = t.pattern_id.split("::")[0]
            count = paradigm_cov.get(paradigm, 0)
            if count < 5 and paradigm not in uncovered:
                priorities.append({
                    "paradigm": paradigm,
                    "priority": "high",
                    "reason": f"有成功模板(occ={t.occurrence_count})但覆盖不足({count}因子)",
                    "suggested_template": t.pattern_id,
                    "typical_operators": t.typical_operators,
                })

        # 3. Mark hard forbidden 为 blocked
        hard_forbidden_paradigms = set()
        for d in directions:
            if d.severity == "hard":
                # 从 description 中提取 paradigm
                for word in d.description.split(":"):
                    word = word.strip()
                    if word:
                        hard_forbidden_paradigms.add(word)
                        break

        # 去重 + 排序
        seen = set()
        unique_priorities = []
        for p in priorities:
            key = p.get("paradigm", "")
            if key not in seen and key not in hard_forbidden_paradigms:
                seen.add(key)
                unique_priorities.append(p)

        return unique_priorities[:10]

    def _build_retrieval_prompt(
        self,
        templates: List[SuccessPattern],
        directions: List[ForbiddenDirection],
        priorities: List[Dict],
    ) -> str:
        """构建可注入 LLM prompt 的 retrieval 片段"""
        lines = ["## 📚 Experience Memory — 历史经验检索"]

        # 成功模板
        if templates:
            lines.append(f"\n### ✅ 可复用成功模板 ({len(templates)} 个)")
            for t in templates[:3]:
                lines.append(f"- **{t.pattern_id}** (occ={t.occurrence_count}, success_rate={t.success_rate:.0%})")
                if t.typical_operators:
                    lines.append(f"  算子: {', '.join(t.typical_operators[:5])}")
                if t.lessons:
                    lines.append(f"  经验: {'; '.join(t.lessons[:2])}")

        # 禁止方向
        if directions:
            lines.append(f"\n### 🚫 禁止方向 — 避免重复 ({len(directions)} 个)")
            for d in directions[:3]:
                icon = "🔴" if d.severity == "hard" else "🟡"
                lines.append(f"- {icon} [{d.severity.upper()}] {d.description[:100]}")
                if d.prototype_expression:
                    lines.append(f"  典型表达式: `{d.prototype_expression[:80]}`")

        # 探索优先级
        if priorities:
            lines.append(f"\n### 🎯 推荐探索方向 ({len(priorities)} 个)")
            for p in priorities[:5]:
                icon = {"highest": "🔵", "high": "🟢"}.get(p.get("priority", ""), "⚪")
                lines.append(f"- {icon} **{p.get('paradigm', '')}**: {p.get('reason', '')}")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════

    # ── 结构化 Memory API (v2 新增) ──────────────────────

    def get_success_templates(
        self, paradigm: str = "", min_occurrence: int = 1
    ) -> List[SuccessPattern]:
        """获取成功因子模板"""
        templates = self.data.get("success_templates", [])
        if paradigm:
            templates = [t for t in templates if paradigm in t.get("pattern_id", "")]
        templates = [t for t in templates if t.get("occurrence_count", 0) >= min_occurrence]
        result = [SuccessPattern.from_dict(t) for t in templates]
        result.sort(key=lambda t: t.occurrence_count, reverse=True)
        return result

    def get_forbidden_directions_list(
        self, severity: str = ""
    ) -> List[ForbiddenDirection]:
        """获取禁止方向列表"""
        directions = self.data.get("forbidden_directions", [])
        if severity:
            directions = [d for d in directions if d.get("severity") == severity]
        result = [ForbiddenDirection.from_dict(d) for d in directions]
        result.sort(key=lambda d: d.failed_attempts, reverse=True)
        return result

    def get_evolution_log(self, n: int = 20) -> List[Dict]:
        """获取 F/E/R 操作日志"""
        log = self.data.get("evolution_log", [])
        return log[-n:]  # 最近 n 条

    def _log_event(self, event_type: str, details: Dict):
        """记录 F/E/R 事件"""
        self.data.setdefault("evolution_log", []).append({
            "event_id": f"evt_{len(self.data['evolution_log']):05d}",
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "details": details,
        })
        # 限制日志长度
        if len(self.data["evolution_log"]) > 1000:
            self.data["evolution_log"] = self.data["evolution_log"][-500:]

    # ── 已存在方法 ──────────────────────────────────────
    # ── 维护操作 ──────────────────────────────────────────
    
    def get_summary(self) -> Dict:
        """获取记忆库摘要"""
        stats = self.data["stats"]
        motif_stats = self.data.get("motif_stats", {})
        motif_rules = self.data.get("motif_rules", [])
        return {
            "total_attempts": stats["total_attempts"],
            "pass_rate": (
                round(stats["total_pass"] / stats["total_attempts"], 3)
                if stats["total_attempts"] > 0 else 0
            ),
            "paradigms_tracked": list(stats.get("by_paradigm", {}).keys()),
            "toxic_patterns": len(self.data["patterns"].get("toxic_patterns", [])),
            "successful_patterns": len(self.data["patterns"].get("successful_patterns", [])),
            # v3 P-001
            "motif_tracked": len(motif_stats),
            "motif_rules": {"prefer": sum(1 for r in motif_rules if r.get("rule_type")=="prefer"),
                           "forbid": sum(1 for r in motif_rules if r.get("rule_type")=="forbid")},
            "file": str(self.path),
        }
    
    def import_from_csv(
        self,
        pool_csv_path: Path,
        mapping: Optional[Dict[str, str]] = None,
    ):
        """
        批量从 passed_factor_pool.csv 导入历史记录。

        Parameters
        ----------
        pool_csv_path: CSV 路径
        mapping: category → paradigm 映射表
        """
        import pandas as pd
        
        if not pool_csv_path.exists():
            return 0
        
        df = pd.read_csv(pool_csv_path, encoding='utf-8-sig')
        has_formula = df['formula'].notna() & (df['formula'] != '')
        df = df[has_formula]
        
        default_mapping = {
            'volume_structure': '流动性', 'liquidity_micro': '微观结构',
            'return_decomposition': '隔夜', 'momentum_dynamics': '趋势/动量',
            'price_pattern': '反转', 'volatility_structure': '波动率',
            'return_distribution': '尾部风险', 'behavioral': '行为金融',
            'information_flow': '微观结构', 'extreme_events': '尾部风险',
            'event_driven': '基本面/成长', 'multi_period': '趋势/动量',
            'relative_value': '基本面/成长', 'trend_quality': '趋势/动量',
            'growth_quality': '基本面/成长', 'mid_report_divergence': '趋势/动量',
        }
        
        actual_mapping = mapping or default_mapping
        
        imported = 0
        existing_names = {a['factor_name'] for a in self.data['attempts']}
        
        for _, row in df.iterrows():
            name = row.get('name', row.get('factor_name', ''))
            if not name or name in existing_names:
                continue
            
            formula = row.get('formula', '')
            category = row.get('category', '')
            paradigm = actual_mapping.get(category, '通用')
            status = row.get('status', 'reserve')
            icir = float(row.get('icir', row.get('daily_icir', 0))) if pd.notna(row.get('icir', row.get('daily_icir', np.nan))) else 0.0
            
            outcome = "PASS" if status == "candidate" else ("REJECT" if status == "failed" else "WEAK")
            
            self.record(
                factor_name=name,
                formula=formula,
                paradigm=paradigm,
                category=category,
                icir=icir,
                outcome=outcome,
            )
            imported += 1
        
        return imported


# ── 便捷函数 ──────────────────────────────────────────────

_default_memory: Optional[ExperienceMemory] = None


def get_memory() -> ExperienceMemory:
    """获取全局单例 ExperienceMemory"""
    global _default_memory
    if _default_memory is None:
        _default_memory = ExperienceMemory()
    return _default_memory


# ── 测试 ──────────────────────────────────────────────────
if __name__ == "__main__":
    mem = ExperienceMemory()
    
    # 记录一些模拟数据
    mem.record(
        factor_name="test_momentum_v1",
        formula="ts_delta(close, 20) / ts_std(close, 20)",
        paradigm="趋势/动量",
        category="momentum_dynamics",
        fri_score=0.65,
        fri_grade="A",
        icir=0.45,
        outcome="PASS",
        tags=["momentum", "cross_section"],
        lessons=["20日动量在 large_dominant 下表现好"],
    )
    
    mem.record(
        factor_name="test_liquidity_fail",
        formula="ts_rank(volume, 20) * ts_corr(close, volume, 10)",
        paradigm="流动性",
        category="volume_structure",
        fri_score=0.28,
        fri_grade="C",
        icir=0.12,
        outcome="REJECT",
        tags=["volume", "high_turnover"],
    )
    
    # 检索
    print("=" * 60)
    print("Experience Memory 测试")
    print("=" * 60)
    
    print(f"\n摘要: {mem.get_summary()}")
    
    print(f"\n动量范式最佳因子:")
    for s in mem.get_best_in_paradigm("趋势/动量", limit=3):
        print(f"  - {s['factor_name']}: FRI={s['fri']['score']:.3f}, ICIR={s['icir']:.3f}")
    
    print(f"\nLLM 上下文 (动量范式):")
    ctx = mem.get_llm_context(paradigm="趋势/动量")
    print(ctx)
    
    print(f"\n范式统计:")
    for p, s in mem.get_paradigm_stats().items():
        print(f"  {p}: {s['total']} 次, {s['pass']} 通过, avg FRI={s['avg_fri']:.3f}")
