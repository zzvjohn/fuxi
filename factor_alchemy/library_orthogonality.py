# -*- coding: utf-8 -*-
"""
Library-Level 正交性管理引擎 (FactorMiner 风格)
===================================================
实现全局因子库视角的正交性管理:
  1. 因子族聚类 — 基于相关性矩阵的层次聚类
  2. 禁止区域 (Forbidden Regions) — 动态维护已被覆盖的因子空间
  3. Correlation Red Sea 监控 — 库接近饱和程度检测
  4. 库级准入控制 — 全局视角的因子入库决策
  5. 因子血统追踪 — 每个因子的演化谱系

设计原则 (来自 evolution_roadmap.md):
  - P1: JQ是唯一真相源
  - P2: Local仅做否决不做排序
  - P3: 进化=探索不是优化
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════

@dataclass
class FactorEntry:
    """因子条目 — 完整血统信息"""
    name: str
    expression: str
    paradigm: str
    category: str = ""
    dimensions: List[str] = field(default_factory=list)
    hypothesis: str = ""      # v0.5: 投资假设 (LLM生成)
    logic: str = ""           # v0.5: 经济逻辑 (LLM生成)
    source: str = "unknown"  # LLM生成 / 手工设计 / 注入 / 进化变体
    parent_factors: List[str] = field(default_factory=list)
    generation: int = 0
    ic_pct: float = 0.0
    ic: float = 0.0  # mean IC (from csv ic_mean)
    icir: float = 0.0
    fri: float = 0.0
    fri_grade: str = "?"
    psi_r2: float = 0.0  # Psi正交化R²
    jq_shadow_return: Optional[float] = None
    jq_full_return: Optional[float] = None
    jq_sharpe: Optional[float] = None
    status: str = "reserve"  # reserve/candidate/injected/king/dead
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class FactorCluster:
    """因子族 — 一组高度相关的因子"""
    cluster_id: int
    factors: List[str]       # 因子名列表
    representative: str = ""  # 代表性因子
    avg_intra_corr: float = 0.0  # 族内平均相关性
    paradigm_distribution: Dict[str, int] = field(default_factory=dict)
    status: str = "active"   # active / depleted / forbidden


@dataclass
class ForbiddenRegion:
    """禁止区域 — 已被充分覆盖的搜索空间"""
    region_id: str
    description: str
    paradigms_involved: List[str]
    factor_prototype: str     # 典型因子表达式
    reason: str               # 为什么禁止: "correlation_exhausted" / "jq_failure" / "local_toxic"
    severity: str = "soft"    # hard=绝对禁止 / soft=低优先级
    created_at: str = ""
    cooldown_until: str = ""  # soft禁止的冷却期
    failed_attempts: int = 0


# ═══════════════════════════════════════════════════════════
# 核心引擎
# ═══════════════════════════════════════════════════════════

class LibraryOrthogonalityManager:
    """
    库级正交性管理器。

    核心概念:
    - Correlation Red Sea: 当库中已有N个因子时，新因子与库中某个因子
      相关性超过阈值的概率随N增长而指数增长。
    - Forbidden Regions: 已被覆盖的搜索空间区域，标记为禁止或低优先级。
    - Global Library Perspective: 每次入库决策考虑整个库的结构，而非单因子质量。
    """

    def __init__(
        self,
        data_dir: Path = None,
        corr_threshold: float = 0.70,
        jaccard_threshold: float = 0.50,
        red_sea_warning: float = 0.65,  # 库相关性中位数超过此值→警告
    ):
        if data_dir is None:
            # P-20260815-001: 统一指向项目 data 目录, 与 run_v4_pipeline DATA_DIR 一致
            # 旧默认 (research/data) 存在一份 0因子/166族 陈旧状态, 与真相源错位
            data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.corr_threshold = corr_threshold
        self.jaccard_threshold = jaccard_threshold
        self.red_sea_warning = red_sea_warning

        # 内部状态
        self.factors: Dict[str, FactorEntry] = {}
        self.clusters: List[FactorCluster] = []
        self.forbidden_regions: List[ForbiddenRegion] = []
        self._corr_matrix: Optional[np.ndarray] = None
        self._factor_names: List[str] = []
        self._state_path = self.data_dir / "library_orthogonality_state.json"

        # P-018: Red Sea 自动重建追踪
        self._last_recluster_time: Optional[str] = None
        self._factor_count_at_recluster: int = 0

        # P-20260831-001: 自动重建日志 (每日次数上限)
        self.auto_rebuild_log: List[Dict] = []

        self._load_state()

    # ── 持久化 ──────────────────────────────────────────

    @property
    def _corr_matrix_path(self) -> Path:
        return self.data_dir / "library_corr_matrix.npz"

    def _load_state(self):
        """加载持久状态"""
        if self._state_path.exists():
            try:
                with open(self._state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                for fd in state.get("factors", []):
                    fe = FactorEntry(**fd)
                    self.factors[fe.name] = fe
                for cd in state.get("clusters", []):
                    self.clusters.append(FactorCluster(**cd))
                for rd in state.get("forbidden_regions", []):
                    self.forbidden_regions.append(ForbiddenRegion(**rd))
                self._factor_names = state.get("_factor_names", [])
                self._last_recluster_time = state.get("_last_recluster_time")
                self._factor_count_at_recluster = state.get("_factor_count_at_recluster", 0)
                self.auto_rebuild_log = state.get("auto_rebuild_log", [])

                # P-018: 加载 corr_matrix (NPZ格式, 独立于JSON)
                if self._corr_matrix_path.exists():
                    try:
                        loaded = np.load(self._corr_matrix_path)
                        self._corr_matrix = loaded['corr_matrix']
                        loaded.close()
                    except Exception:
                        self._corr_matrix = None

                print(f"[正交性管理] 加载状态: {len(self.factors)}因子, "
                      f"{len(self.clusters)}族, {len(self.forbidden_regions)}禁止区"
                      f"{', +corr_matrix' if self._corr_matrix is not None else ''}")
            except Exception as e:
                print(f"[正交性管理] ⚠️ 状态加载失败: {e}, 从空库开始")

    def save_state(self):
        """持久化当前状态"""
        state = {
            "updated_at": datetime.now().isoformat(),
            "factors": [f.to_dict() for f in self.factors.values()],
            "clusters": [c.__dict__ for c in self.clusters],
            "forbidden_regions": [fr.__dict__ for fr in self.forbidden_regions],
            "_factor_names": self._factor_names,
            "_last_recluster_time": self._last_recluster_time,
            "_factor_count_at_recluster": self._factor_count_at_recluster,
            "auto_rebuild_log": self.auto_rebuild_log,
            "stats": self.get_library_stats(),
        }
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)

        # P-018: corr_matrix 独立持久化 (NPZ格式)
        if self._corr_matrix is not None:
            np.savez_compressed(self._corr_matrix_path, corr_matrix=self._corr_matrix)

    # ── 因子注册 ────────────────────────────────────────

    def register_factor(self, factor: FactorEntry) -> Dict:
        """
        注册新因子到库。返回准入决策。

        流程:
        1. Psi冗余检查 (R² > 0.95 → 拒绝)
        2. 相关性红海检查 (与库中所有因子max corr)
        3. 禁止区域匹配
        4. 入库 + 重新聚类 + 更新禁止区域
        """
        result = {
            "admitted": False,
            "reason": "",
            "warnings": [],
            "suggested_category": "",
        }

        # ── 0. 基础检查 ──
        if factor.name in self.factors:
            result["reason"] = f"因子 {factor.name} 已存在于库中"
            return result

        # ── 1. Psi冗余检查 ──
        if factor.psi_r2 > 0.95:
            result["reason"] = f"Psi R²={factor.psi_r2:.3f} > 0.95, 信息增量不足5%"
            result["warnings"].append("几乎完全冗余 → 直接拒绝")
            return result

        if factor.psi_r2 > 0.80:
            result["warnings"].append(f"Psi R²={factor.psi_r2:.3f} > 0.80, 信息增量不足20%")

        # ── 2. 禁止区域匹配 ──
        forbidden_match = self._check_forbidden_regions(factor)
        if forbidden_match:
            if forbidden_match.severity == "hard":
                result["reason"] = f"命中硬禁止区域: {forbidden_match.description}"
                return result
            else:
                result["warnings"].append(
                    f"命中软禁止区域: {forbidden_match.description} → 降低优先级")

        # ── 3. 相关性红海检查 (如果有相关性矩阵) ──
        if self._corr_matrix is not None and len(self._factor_names) > 0:
            # 这里仅做逻辑检查，实际corr需要从外部传入
            red_sea_status = self.get_red_sea_status()
            if red_sea_status["level"] == "critical":
                result["warnings"].append(
                    f"⚠️ 库已进入Correlation Red Sea: "
                    f"中位相关性={red_sea_status['median_corr']:.3f}")

        # ── 4. 准入 ──
        result["admitted"] = True
        if not result["warnings"]:
            result["reason"] = "通过所有准入检查"
        else:
            result["reason"] = "通过(有警告)"

        # ── 5. 入库 ──
        factor.updated_at = datetime.now().isoformat()
        self.factors[factor.name] = factor
        self._factor_names.append(factor.name)

        # ── 6. 增量更新禁止区域 ──
        self._update_forbidden_regions(factor)

        self.save_state()
        return result

    # ── 禁止区域管理 ─────────────────────────────────────

    def _check_forbidden_regions(self, factor: FactorEntry) -> Optional[ForbiddenRegion]:
        """检查因子是否落入已知禁止区域"""
        for fr in self.forbidden_regions:
            # 匹配条件: 至少共享一个范式
            if factor.paradigm in fr.paradigms_involved:
                return fr
            # 或者维度重叠
            if set(factor.dimensions) & set(fr.paradigms_involved):
                return fr
        return None

    def add_forbidden_region(
        self, description: str, paradigms: List[str],
        prototype: str, reason: str, severity: str = "soft",
        cooldown_days: int = 14,
    ) -> ForbiddenRegion:
        """手动添加禁止区域 (如JQ验证失败后)"""
        region_id = f"FR_{len(self.forbidden_regions):03d}"
        fr = ForbiddenRegion(
            region_id=region_id,
            description=description,
            paradigms_involved=paradigms,
            factor_prototype=prototype,
            reason=reason,
            severity=severity,
            created_at=datetime.now().isoformat(),
            cooldown_until=(
                datetime.now().isoformat() if severity == "hard"
                else ""  # soft需要cooldown
            ),
        )
        if severity == "soft" and cooldown_days > 0:
            from datetime import timedelta
            fr.cooldown_until = (datetime.now() + timedelta(days=cooldown_days)).isoformat()

        self.forbidden_regions.append(fr)
        self.save_state()
        print(f"[禁止区域] 新增: {description} ({severity}, {len(paradigms)}范式)")
        return fr

    def _update_forbidden_regions(self, new_factor: FactorEntry):
        """入库后自动更新禁止区域 — 检测是否范式已被过度覆盖"""
        paradigm_count = defaultdict(int)
        for f in self.factors.values():
            paradigm_count[f.paradigm] += 1

        # 如果某个范式已有5+个因子且最后3个相关性>0.5 → 标记为soft-forbidden
        for paradigm, count in paradigm_count.items():
            if count >= 5:
                paradigm_factors = [
                    n for n, f in self.factors.items()
                    if f.paradigm == paradigm
                ]
                # 检查是否已有此范式的禁止区域
                existing = any(
                    paradigm in fr.paradigms_involved
                    for fr in self.forbidden_regions
                )
                if not existing:
                    self.add_forbidden_region(
                        description=f"范式 [{paradigm}] 已被充分覆盖 ({count}个因子)",
                        paradigms=[paradigm],
                        prototype=f"latest: {new_factor.name}",
                        reason="paradigm_coverage_saturated",
                        severity="soft",
                        cooldown_days=30,
                    )

    # ── 因子族聚类 ───────────────────────────────────────

    def recluster(self, corr_matrix: np.ndarray, factor_names: List[str]):
        """
        基于相关性矩阵重新聚类。

        使用简单的层次聚类/阈值聚类:
        - 与代表因子 corr > threshold → 归入同一族
        - 否则 → 创建新族
        """
        self._corr_matrix = corr_matrix
        self._factor_names = factor_names
        self._last_recluster_time = datetime.now().isoformat()
        self._factor_count_at_recluster = len(factor_names)
        n = len(factor_names)

        assigned = set()
        self.clusters = []
        cluster_id = 0

        # 简单阈值聚类: 以corr最高的因子对开始
        for i in range(n):
            if i in assigned:
                continue
            cluster_factors = [factor_names[i]]
            assigned.add(i)

            for j in range(i + 1, n):
                if j in assigned:
                    continue
                # 与聚类中已有因子的最大corr (简化: 直接用下面的循环)
                max_corr = 0
                # 检查与聚类成员的corr
                high_corr = False
                for cf_name in cluster_factors:
                    cf_idx = factor_names.index(cf_name)
                    if abs(corr_matrix[j, cf_idx]) > self.corr_threshold:
                        high_corr = True
                        break

                if high_corr:
                    cluster_factors.append(factor_names[j])
                    assigned.add(j)

            # 计算族内平均相关性
            intra_corrs = []
            for ci, cf1 in enumerate(cluster_factors):
                for cf2 in cluster_factors[ci + 1:]:
                    idx1 = factor_names.index(cf1)
                    idx2 = factor_names.index(cf2)
                    intra_corrs.append(abs(corr_matrix[idx1, idx2]))

            avg_corr = np.mean(intra_corrs) if intra_corrs else 0.0

            # 找代表性因子 (ICIR最高或JQ表现最好)
            rep = cluster_factors[0]
            best_score = -999
            for cf in cluster_factors:
                fe = self.factors.get(cf)
                if fe:
                    score = fe.icir or 0
                    if fe.jq_sharpe and fe.jq_sharpe > 0:
                        score = fe.jq_sharpe * 10
                    if score > best_score:
                        best_score = score
                        rep = cf

            # 统计范式分布
            paradigm_dist = defaultdict(int)
            for cf in cluster_factors:
                fe = self.factors.get(cf)
                if fe and fe.paradigm:
                    paradigm_dist[fe.paradigm] += 1

            cluster = FactorCluster(
                cluster_id=cluster_id,
                factors=cluster_factors,
                representative=rep,
                avg_intra_corr=avg_corr,
                paradigm_distribution=dict(paradigm_dist),
                status="active",
            )
            self.clusters.append(cluster)
            cluster_id += 1

        self.save_state()
        print(f"[聚类] {n}因子 → {len(self.clusters)}族 (阈值={self.corr_threshold})")
        return self.clusters

    # ── P-018: Red Sea 自动重建 ──────────────────────────

    def schedule_recluster(self, factor_pool: dict = None, max_age_hours: float = 24,
                           min_factor_change: int = 5) -> Dict:
        """
        自动调度 recluster: 满足两个条件之一即触发。
        1. 距离上次 recluster > max_age_hours
        2. 因子池变动 > min_factor_change

        Parameters
        ----------
        factor_pool: {name: FactorEntry} — 当前因子池，用于计算变动数
        max_age_hours: 超时小时数
        min_factor_change: 触发重建的最小因子变动数

        Returns
        -------
        {"triggered": bool, "reason": str, "details": dict}
        """
        result = {"triggered": False, "reason": "", "details": {}}

        if factor_pool is None:
            factor_pool = self.factors

        now = datetime.now()
        current_count = len(factor_pool)

        # 条件1: 超时
        if self._last_recluster_time:
            try:
                last = datetime.fromisoformat(self._last_recluster_time)
                age_hours = (now - last).total_seconds() / 3600
                result["details"]["age_hours"] = round(age_hours, 1)
                if age_hours > max_age_hours:
                    result["triggered"] = True
                    result["reason"] = f"距上次recluster已{age_hours:.0f}h > {max_age_hours}h"
            except (ValueError, TypeError):
                result["details"]["age_hours"] = "unknown"

        # 条件2: 因子变动
        delta = current_count - self._factor_count_at_recluster
        result["details"]["factor_delta"] = delta
        result["details"]["current_count"] = current_count
        result["details"]["count_at_last"] = self._factor_count_at_recluster

        if abs(delta) >= min_factor_change:
            result["triggered"] = True
            if result["reason"]:
                result["reason"] += f" + 因子变动{delta}个 >= {min_factor_change}"
            else:
                result["reason"] = f"因子变动{delta}个 >= {min_factor_change}"

        if not result["triggered"]:
            result["reason"] = "未满足触发条件"
            result["details"]["next_check_after"] = f"after {max_age_hours}h or {min_factor_change} factor changes"

        return result

    def get_crowding_score(self, factor_name: str) -> Optional[float]:
        """
        计算单个因子的拥挤度得分。
        crowding = avg_corr_in_cluster × (cluster_size / n_clusters)
        得分越高 → 与其他因子越"拥挤" → 风险越高
        """
        if self._corr_matrix is None:
            return None
        if factor_name not in self._factor_names:
            return None

        idx = self._factor_names.index(factor_name)
        # 防御: corr_matrix 可能只覆盖部分因子 (如重建失败时矩阵与因子名列表不一致)
        if self._corr_matrix.shape[0] <= idx or self._corr_matrix.shape[1] <= idx:
            return None
        n = len(self._factor_names)
        if n < 2:
            return 0.0

        # 找到该因子所属的族
        target_cluster = None
        for cluster in self.clusters:
            if factor_name in cluster.factors:
                target_cluster = cluster
                break

        # 与所有其他因子的平均相关性
        correlations = []
        m = self._corr_matrix.shape[0]  # 矩阵实际覆盖的因子数
        for j in range(min(n, m)):
            if j != idx:
                correlations.append(abs(self._corr_matrix[idx, j]))
        avg_corr_all = np.mean(correlations) if correlations else 0.0

        # 拥挤度 = avg_corr × 族规模归一化
        if target_cluster and len(self.clusters) > 0:
            cluster_size = len(target_cluster.factors)
            crowding = avg_corr_all * (cluster_size / max(len(self.clusters), 1))
        else:
            crowding = avg_corr_all

        return float(crowding)

    def get_all_crowding_scores(self) -> Dict[str, float]:
        """计算所有因子的拥挤度得分，返回 {factor_name: crowding_score}"""
        scores = {}
        for name in self._factor_names:
            score = self.get_crowding_score(name)
            if score is not None:
                scores[name] = score
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def get_crowding_top_n(self, n: int = 5) -> List[tuple]:
        """返回拥挤度最高的 N 个因子"""
        scores = self.get_all_crowding_scores()
        return list(scores.items())[:n]

    # ── Red Sea 监控 ─────────────────────────────────────

    def get_red_sea_status(self) -> Dict:
        """获取Correlation Red Sea状态"""
        if self._corr_matrix is None or len(self._factor_names) == 0:
            return {
                "level": "no_matrix",
                "median_corr": 0.0,
                "max_corr": 0.0,
                "pct_above_threshold": 0.0,
                "n_factors": len(self.factors),
                "n_clusters": len(self.clusters),
                "threshold": self.corr_threshold,
                "reason": "尚未计算相关性矩阵 — 需先调用 recluster() 或手动加载 corr_matrix",
            }

        n = len(self._factor_names)
        cm_n = self._corr_matrix.shape[0]
        # P-018 修复: _factor_names 与 _corr_matrix 可能不同步 (因子新增后未重建矩阵)
        if n != cm_n:
            n = cm_n
        # 取下三角 (不含对角线)
        if n < 2:
            return {"level": "green", "median_corr": 0, "n_factors": n}

        tril_indices = np.tril_indices(n, k=-1)
        correlations = np.abs(self._corr_matrix[tril_indices])
        median_corr = np.median(correlations)
        pct_above_threshold = np.mean(correlations > self.corr_threshold)  # 小数 (0.48% = 0.0048)

        # 分级
        if median_corr > self.red_sea_warning + 0.10:
            level = "critical"
        elif median_corr > self.red_sea_warning:
            level = "warning"
        elif median_corr > self.red_sea_warning - 0.10:
            level = "elevated"
        else:
            level = "green"

        return {
            "level": level,
            "median_corr": float(median_corr),
            "max_corr": float(np.max(correlations)),
            "pct_above_threshold": float(pct_above_threshold),
            "n_factors": n,
            "n_clusters": len(self.clusters),
            "threshold": self.corr_threshold,
        }

    def get_library_stats(self) -> Dict:
        """获取库的全局统计"""
        total = len(self.factors)
        by_paradigm = defaultdict(int)
        by_status = defaultdict(int)
        by_source = defaultdict(int)
        for f in self.factors.values():
            by_paradigm[f.paradigm] += 1
            by_status[f.status] += 1
            by_source[f.source] += 1

        # 范式覆盖率
        from paradigm_v4 import PARADIGMS_V4
        paradigm_coverage = {
            p: by_paradigm.get(p, 0)
            for p in PARADIGMS_V4
        }

        return {
            "total_factors": total,
            "n_clusters": len(self.clusters),
            "n_forbidden_regions": len(self.forbidden_regions),
            "by_paradigm": dict(by_paradigm),
            "by_status": dict(by_status),
            "by_source": dict(by_source),
            "paradigm_coverage": paradigm_coverage,
            "red_sea": self.get_red_sea_status(),
        }

    def get_suggested_exploration_directions(self) -> List[Dict]:
        """
        基于当前库结构，建议探索方向。
        优先: 未被覆盖的范式 + 与现有因子低相关的方向
        """
        from paradigm_v4 import PARADIGMS_V4, FORBIDDEN_REGIONS

        suggestions = []
        for paradigm, info in PARADIGMS_V4.items():
            count = sum(1 for f in self.factors.values() if f.paradigm == paradigm)

            # 未覆盖的范式 → 高优先级
            if count == 0:
                suggestions.append({
                    "paradigm": paradigm,
                    "priority": "high",
                    "reason": f"范式 [{paradigm}] 尚未被覆盖",
                    "current_count": 0,
                    "description": info["description"],
                    "a_share_relevance": info["a_share_relevance"],
                })
            # 覆盖1-2个 → 中优先级
            elif count <= 2:
                suggestions.append({
                    "paradigm": paradigm,
                    "priority": "medium",
                    "reason": f"范式 [{paradigm}] 覆盖面不足 ({count}个因子)",
                    "current_count": count,
                    "description": info["description"],
                })
            # 禁止区域检查
            forbidden_paradigms = set()
            for fr in self.forbidden_regions:
                if fr.severity == "hard":
                    forbidden_paradigms.update(fr.paradigms_involved)

        # 按优先级排序
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 3))

        return suggestions

    def print_report(self):
        """打印库状态报告"""
        stats = self.get_library_stats()
        rs = stats["red_sea"]

        print("=" * 60)
        print("  Library Orthogonality Report")
        print("=" * 60)
        print(f"  因子总数:        {stats['total_factors']}")
        print(f"  因子族数:        {stats['n_clusters']}")
        print(f"  禁止区域:        {stats['n_forbidden_regions']}")
        print(f"  Red Sea Level:   {rs['level'].upper()}")
        if rs.get("reason"):
            print(f"  说明:            {rs['reason']}")
        print(f"  中位相关性:       {rs.get('median_corr', 0):.3f}")
        if rs.get("pct_above_threshold", 0) > 0:
            print(f"  超阈值比例:      {rs['pct_above_threshold']:.1f}%")
        
        # v0.5 P-004: 衰减率概览
        try:
            from decay_monitor import get_decay_monitor
            dm = get_decay_monitor()
            n_decay = len(dm.ic_cache)
            if n_decay > 0:
                decaying = sum(1 for _ in dm.ic_cache if dm.get_decay_rate(_) < 0.7)
                print(f"  衰减率追踪:      {n_decay} 因子 | {decaying} 衰减率<0.7")
        except ImportError:
            pass
        print()

        print("  范式覆盖:")
        for paradigm, count in stats["paradigm_coverage"].items():
            bar = "█" * min(count, 20) + "░" * max(0, 20 - min(count, 20))
            tag = " [未覆盖]" if count == 0 else ""
            print(f"    {paradigm:20s} {bar} {count}{tag}")

        print()
        print("  建议探索方向 (高优先级):")
        suggestions = self.get_suggested_exploration_directions()
        for s in suggestions[:5]:
            if s["priority"] == "high":
                print(f"    → {s['paradigm']}: {s['reason']}")

        print("=" * 60)


# ═══════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════

def rebuild_from_factor_pool(
    csv_path: Path = None,
    pkl_path: Path = None,
    max_factors: int = 500,
) -> "LibraryOrthogonalityManager":
    """
    P-20260815-001: 从真相源 (passed_factor_pool.csv + existing_factor_pool.pkl)
    全量重建正交性状态。修复 state json 与因子池错位 (0因子/166族 vs 300因子/51族)。

    流程: 备份旧状态 → CSV 构建 FactorEntry → pkl 计算 corr 矩阵 → recluster → 落盘
    """
    import shutil

    mgr = LibraryOrthogonalityManager()

    # 1. 备份旧状态 (P-20260831-001: 时间戳命名, 不再固定覆盖旧档)
    if mgr._state_path.exists():
        bak = mgr._state_path.with_suffix(f".json.bak_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(mgr._state_path, bak)
        print(f"[重建] 旧状态已备份: {bak}")

    # 2. 从 run_v4_pipeline 复用元数据加载与矩阵计算 (函数内 lazy import 避免循环)
    from run_v4_pipeline import (
        load_factor_metadata, build_factor_entries, compute_correlation_matrix,
    )

    if csv_path is None:
        csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "passed_factor_pool.csv"
    if pkl_path is None:
        pkl_path = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "existing_factor_pool.pkl"

    metadata = load_factor_metadata(csv_path)
    entries, cat_map = build_factor_entries(metadata)
    print(f"[重建] CSV 元数据 {len(entries)} 因子")

    if len(entries) < 3:
        print("[重建] ❌ 因子元数据不足, 中止")
        return mgr

    # 3. 计算相关性矩阵 (从 pkl 因子值缓存)
    corr_matrix, factor_names = compute_correlation_matrix(
        entries, pkl_path=pkl_path, max_factors=max_factors,
    )

    # 4. 重置管理器并重建聚类
    mgr.factors = {e.name: e for e in entries}
    mgr._factor_names = [e.name for e in entries]
    mgr._corr_matrix = corr_matrix

    if corr_matrix is not None and len(factor_names) >= 3:
        mgr.recluster(corr_matrix, factor_names)
    else:
        print("[重建] ⚠️ corr 矩阵不可用, 仅刷新因子列表")

    # P-20260831-001: 记录 CSV 全量计数为重建基线 (corr 矩阵 max_factors=500
    # 只是计算上限, 若用 len(factor_names) 会与 CSV 657 永久差 157 → 每日误触发)
    mgr._factor_count_at_recluster = len(entries)

    mgr.save_state()
    print(f"[重建] ✅ 完成: {len(mgr.factors)}因子 / {len(mgr.clusters)}族")
    return mgr


def check_rs_health_and_rebuild(
    max_age_hours: float = 24,
    min_factor_change: int = 5,
    auto_rebuild: bool = True,
    max_rebuilds_per_day: int = 1,
    verbose: bool = True,
) -> Dict:
    """P-20260831-001: RS Health 预检 + 自动重建闭环。

    diagnose 长期显示 NEEDS_REBUILD 但 Ralph Loop 从不自动重建, 正交状态
    与因子库长期失配。此函数供 ralph_loop.run() 每轮开始前调用:

      1. 用真相源 CSV (passed_factor_pool.csv) 计数 vs state 记录的上次重建计数,
         评估 schedule_recluster 触发条件 (修 diagnose 传空 factor_pool={} 的 bug,
         该 bug 导致 delta 恒为负大数);
      2. triggered 且 auto_rebuild=True 时调用 rebuild_from_factor_pool();
      3. 每日自动重建次数上限 max_rebuilds_per_day (状态写 state json 的
         auto_rebuild_log 字段), 防止同一天内反复重建。

    Returns: {health, triggered, rebuilt, reason, csv_count, state_count, ...}
    """
    mgr = LibraryOrthogonalityManager()
    result = {
        "health": "OK",
        "triggered": False,
        "rebuilt": False,
        "reason": "",
        "csv_count": None,
        "state_count": len(mgr.factors),
        "rebuilds_today": 0,
    }

    # ── 1. 真相源 CSV 计数 ──
    csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "passed_factor_pool.csv"
    csv_count = None
    if csv_path.exists():
        try:
            import csv as _csv
            with open(csv_path, encoding="utf-8-sig") as f:
                csv_count = sum(1 for row in _csv.DictReader(f) if row.get("name", "").strip())
            result["csv_count"] = csv_count
        except Exception as e:
            result["reason"] = f"CSV 计数失败: {e}"
            if verbose:
                print(f"[RS-Health] ⚠️ CSV 计数失败 ({e}), 跳过自动重建")
            return result
    else:
        result["reason"] = "passed_factor_pool.csv 不存在, 无法比对"
        if verbose:
            print(f"[RS-Health] ⚠️ {result['reason']}, 跳过自动重建")
        return result

    # ── 2. 触发条件检测 (直接复用 schedule_recluster 的规则, 用 CSV 真源计数) ──
    auto_result = {"triggered": False, "reason": ""}
    now = datetime.now()
    last_t = mgr._last_recluster_time
    age_hours = None
    if last_t:
        try:
            last = datetime.fromisoformat(last_t)
            age_hours = (now - last).total_seconds() / 3600
            if age_hours > max_age_hours:
                auto_result["triggered"] = True
                auto_result["reason"] = f"距上次recluster已{age_hours:.0f}h > {max_age_hours}h"
        except (ValueError, TypeError):
            pass
    delta = csv_count - mgr._factor_count_at_recluster
    if abs(delta) >= min_factor_change:
        auto_result["triggered"] = True
        if auto_result["reason"]:
            auto_result["reason"] += f" + 因子变动{delta}个 >= {min_factor_change}"
        else:
            auto_result["reason"] = f"因子变动{delta}个 >= {min_factor_change}"
    if not auto_result["triggered"]:
        auto_result["reason"] = (f"未满足触发条件 (age={age_hours:.0f}h" if age_hours
                                 else "未满足触发条件 (age=unknown") + \
            f", delta={delta}, need>=5 or >{max_age_hours:.0f}h)"
    if not auto_result["triggered"]:
        result["reason"] = auto_result["reason"]
        if verbose:
            print(f"[RS-Health] OK: {auto_result['reason']}")
        return result

    result["triggered"] = True
    result["health"] = "NEEDS_REBUILD"
    result["reason"] = auto_result["reason"]
    if not auto_rebuild:
        if verbose:
            print(f"[RS-Health] NEEDS_REBUILD ({auto_result['reason']}), auto_rebuild=False 跳过")
        return result

    # ── 3. 每日重建次数上限 ──
    today = time.strftime("%Y-%m-%d")
    auto_log = list(getattr(mgr, "auto_rebuild_log", []) or [])
    rebuilds_today = sum(1 for e in auto_log if str(e.get("date")) == today)
    result["rebuilds_today"] = rebuilds_today
    if rebuilds_today >= max_rebuilds_per_day:
        result["reason"] += f"; 今日已自动重建 {rebuilds_today} 次 (上限 {max_rebuilds_per_day}), 跳过"
        if verbose:
            print(f"[RS-Health] ⚠️ {result['reason']}")
        return result

    # ── 4. 自动重建 ──
    if verbose:
        print(f"[RS-Health] NEEDS_REBUILD ({auto_result['reason']}) → 自动重建中 (约3-4分钟)...")
    try:
        mgr = rebuild_from_factor_pool()
        result["rebuilt"] = True
        result["health"] = "REBUILT"
        result["state_count_after"] = len(mgr.factors)
        # 重建后 mgr 是新实例: 追加 auto_rebuild_log 再保存
        mgr.auto_rebuild_log = (list(mgr.auto_rebuild_log or [])[-20:]
                                + [{
                                    "date": today,
                                    "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "reason": auto_result["reason"],
                                    "csv_count": csv_count,
                                }])
        mgr.save_state()
        if verbose:
            print(f"[RS-Health] ✅ 自动重建完成: {len(mgr.factors)}因子 / {len(mgr.clusters)}族")
    except Exception as e:
        result["reason"] += f"; 自动重建失败: {type(e).__name__}: {str(e)[:100]}"
        if verbose:
            print(f"[RS-Health] ❌ 自动重建失败: {e}")

    return result


def create_from_existing_factors(
    factor_dir: Path = None,
    data_dir: Path = None,
) -> LibraryOrthogonalityManager:
    """从现有因子池创建管理器并初始化 (P-001: 失配时提示 rebuild)"""
    if factor_dir is None:
        factor_dir = Path(__file__).resolve().parent.parent.parent / "data"
    if data_dir is None:
        data_dir = factor_dir

    mgr = LibraryOrthogonalityManager(data_dir=data_dir)

    # P-20260815-001: 与真相源 CSV 比对, 失配>=5 提示重建
    csv_path = factor_dir / "passed_factor_pool.csv"
    if csv_path.exists():
        try:
            import csv as _csv
            with open(csv_path, encoding="utf-8-sig") as f:
                csv_count = sum(1 for row in _csv.DictReader(f) if row.get("name", "").strip())
            diff = csv_count - len(mgr.factors)
            if abs(diff) >= 5:
                print(f"[正交性管理] ⚠️ 状态失配: CSV {csv_count}因子 vs state {len(mgr.factors)}因子 "
                      f"(差 {diff}) → 建议: python library_orthogonality.py --rebuild")
        except Exception as e:
            print(f"[正交性管理] ⚠️ CSV 比对失败: {e}")

    # 尝试从 existing_factor_pool.pkl 加载因子
    pool_path = factor_dir / "cache" / "existing_factor_pool.pkl"
    if pool_path.exists():
        print(f"[正交性管理] 发现已有因子池: {pool_path}")

    return mgr


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--rebuild":
        mgr = rebuild_from_factor_pool()
    else:
        mgr = create_from_existing_factors()
    mgr.print_report()
