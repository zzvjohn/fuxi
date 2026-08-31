# -*- coding: utf-8 -*-
"""
BehaviorHomogeneity — v0.6 P0-2: 行为聚类同质性门禁 (P-007 升级)
==================================================================
设计目标: 因子行为同质性检测 (factor homogeneity + factor behavior):
  行为指纹 = 周频截面百分位 rank 的随机投影 sketch (JL 12维)
  相似度   = 0.65 × positive(信号指纹相关) + 0.35 × positive(残差收益相关)
            残差收益 = 日收益 - 跨因子日度中位数
  门禁     : 在线最近邻相似度 ≥ reject(0.92) → 硬拒
             ≥ substitute(0.82) → SUBSTITUTE 标记
  拥挤簇   : 层次聚类边界 0.74, 簇 size ≥ 8 → FOLD_TO_LEADER 指令

与伏羲嵌合:
  - 输入: 候选信号序列 (n_days,) 与库内因子信号 (复用 FactorICComputer 产出)
  - 生成前: build_research_directive() → must_avoid_cluster_ids/preferred_mechanisms
            → 注入 llm_gen.receive_context(homogeneity_control=...)
  - 评估后: evaluate_candidate() → redundancy_label 附到候选 (S2 阶段并行影子)

数据流: E 阶段 S2 之前调用, 与现有 max_corr 检查并行; 开关默认 False。
"""

from __future__ import annotations
import os

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# v0.6.1: 12→32 维。12 维时独立信号投影 cos 厚尾 (σ≈0.29, 实测尾值 0.83),
# 0.92 硬拒误报风险不可忽略; 32 维 σ≈0.18, 独立信号 cos 实测 ≤0.45, 克隆仍 ≈1.0。
JL_DIM = 32
SIGNAL_WEIGHT = 0.65
RESIDUAL_WEIGHT = 0.35
CLUSTER_SIMILARITY = 0.74
NEAR_DUPLICATE_SIGNAL = 0.95
NEAR_DUPLICATE_RESIDUAL = 0.90


@dataclass
class BehaviorProfile:
    """单因子行为指纹 (可复算/可持久化)"""
    factor_id: str
    signal_fingerprint: np.ndarray          # (JL_DIM,) 周频截面rank随机投影
    residual_fingerprint: np.ndarray        # (JL_DIM,) 残差收益随机投影
    n_weeks: int = 0

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "signal_fp": [round(float(x), 6) for x in self.signal_fingerprint],
            "residual_fp": [round(float(x), 6) for x in self.residual_fingerprint],
            "n_weeks": self.n_weeks,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BehaviorProfile":
        return cls(
            factor_id=d["factor_id"],
            signal_fingerprint=np.asarray(d["signal_fp"], dtype=float),
            residual_fingerprint=np.asarray(d["residual_fp"], dtype=float),
            n_weeks=int(d.get("n_weeks", 0)),
        )


@dataclass
class HomogeneityVerdict:
    factor_id: str = ""
    redundancy_label: str = "PENDING"       # NEAR_DUPLICATE/SUBSTITUTE/RELATED/DISTINCT/PENDING
    nearest_factor_id: str = ""
    nearest_similarity: float = 0.0
    signal_correlation: float = 0.0
    residual_correlation: float = 0.0
    cluster_id: str = ""
    cluster_size: int = 0
    gate_passed: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "redundancy_label": self.redundancy_label,
            "nearest_factor_id": self.nearest_factor_id,
            "nearest_similarity": round(self.nearest_similarity, 4),
            "signal_correlation": round(self.signal_correlation, 4),
            "residual_correlation": round(self.residual_correlation, 4),
            "cluster_id": self.cluster_id,
            "cluster_size": self.cluster_size,
            "gate_passed": self.gate_passed,
            "reason": self.reason,
        }


def _jl_projection(values: np.ndarray, dim: int = JL_DIM, seed: int = 20260828) -> np.ndarray:
    """确定性 JL 随机投影: 信号向量 → dim 维 sketch (归一化到单位长度, 相关=cos相似)。"""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 8:
        return np.zeros(dim)
    # 周频截面百分位 rank → 中心化 (去 DC 分量) → 随机高斯投影
    # v0.6.1 修复: 不中心化的 rank 含均值 0.5 的 DC 分量, 常数信号与随机信号
    # 的投影 cos 相似度被抬到 ~0.75-0.92 (伪阳性硬拒)。中心化后 DC=0。
    ranks = np.argsort(np.argsort(vals)) / max(1.0, vals.size - 1) - 0.5
    rng = np.random.RandomState(seed)
    proj = rng.normal(0.0, 1.0, size=(dim, vals.size))
    sketch = proj @ ranks
    norm = np.linalg.norm(sketch)
    return sketch / norm if norm > 1e-12 else np.zeros(dim)


def residualize_daily_returns(returns_matrix: np.ndarray, axis: int = 1) -> np.ndarray:
    """残差化: 每行(日)减去跨因子日度中位数。returns_matrix: (n_days, n_factors)"""
    m = np.asarray(returns_matrix, dtype=float)
    med = np.nanmedian(m, axis=axis, keepdims=True)
    return m - med


def fingerprint_from_returns(
    returns_matrix: np.ndarray,
    axis: int = 1,
) -> np.ndarray:
    """残差收益指纹 = JL 投影残差化后的日收益序列 (按 axis 方向聚合)。"""
    m = np.asarray(returns_matrix, dtype=float)
    if m.size == 0:
        return np.zeros(JL_DIM)
    if axis == 1:
        series = np.nanmean(m, axis=1)   # 每行(日)的因子均值 → 需先残差化
    else:
        series = np.nanmean(m, axis=0)
    return _jl_projection(series)


def behavior_similarity(a: BehaviorProfile, b: BehaviorProfile) -> Tuple[float, float, float]:
    """返回 (综合相似度, 信号相关, 残差收益相关)。

    残差指纹缺失 (全零) 时退化到纯信号相似度 (w_sig=1.0), 避免换皮因子逃逸。
    """
    sig = float(np.clip(np.dot(a.signal_fingerprint, b.signal_fingerprint), 0.0, 1.0))
    res = float(np.clip(np.dot(a.residual_fingerprint, b.residual_fingerprint), 0.0, 1.0))
    a_has_res = bool(np.any(np.abs(a.residual_fingerprint) > 1e-12))
    b_has_res = bool(np.any(np.abs(b.residual_fingerprint) > 1e-12))
    if a_has_res and b_has_res:
        comp = SIGNAL_WEIGHT * sig + RESIDUAL_WEIGHT * res
    else:
        comp = sig  # 退化: 纯信号
    return comp, sig, res


def stable_cluster_id(members: List[str]) -> str:
    """簇 ID = 排序成员的稳定哈希 (可复现, 与显示顺序无关)。"""
    digest = hashlib.sha256("|".join(sorted(members)).encode("utf-8")).hexdigest()
    return f"B_{digest[:10]}"


class BehaviorHomogeneity:
    """行为聚类 + 准入门禁 + 生成前指令 (P0-2)。"""

    def __init__(
        self,
        *,
        reject_similarity: float = 0.92,
        substitute_similarity: float = 0.82,
        cluster_threshold: float = 0.74,
        crowded_size: int = 8,
        enabled: bool = False,
    ):
        self.reject_similarity = reject_similarity
        self.substitute_similarity = substitute_similarity
        self.cluster_threshold = cluster_threshold
        self.crowded_size = crowded_size
        self.enabled = enabled
        self.library: Dict[str, BehaviorProfile] = {}

    # ── 库管理 ────────────────────────────────────────────────
    def register(
        self,
        factor_id: str,
        signal_series: np.ndarray,
        daily_returns: Optional[np.ndarray] = None,
        residual_returns: Optional[np.ndarray] = None,
    ) -> BehaviorProfile:
        sig_fp = _jl_projection(signal_series)
        if residual_returns is not None:
            res_fp = _jl_projection(np.asarray(residual_returns, dtype=float))
        elif daily_returns is not None:
            res_fp = _jl_projection(np.asarray(daily_returns, dtype=float))
        else:
            res_fp = np.zeros(JL_DIM)
        prof = BehaviorProfile(factor_id, sig_fp, res_fp, int(np.isfinite(signal_series).sum()))
        self.library[factor_id] = prof
        return prof

    def build_library_from_pool(self, pool: List[Dict[str, Any]]) -> int:
        """从因子池构建库: pool 元素含 factor_name + signal 数组 (+ daily_returns 可选)。"""
        n = 0
        for item in pool:
            name = str(item.get("factor_name", item.get("factor_id", "")))
            sig = item.get("signal")
            if not name or sig is None:
                continue
            self.register(name, np.asarray(sig, dtype=float),
                          daily_returns=item.get("daily_returns"),
                          residual_returns=item.get("residual_returns"))
            n += 1
        return n

    # ── 候选评估 (准入) ───────────────────────────────────────
    def evaluate_candidate(
        self,
        factor_id: str,
        signal_series: np.ndarray,
        daily_returns: Optional[np.ndarray] = None,
        residual_returns: Optional[np.ndarray] = None,
        *,
        persist_on_pass: bool = False,
    ) -> HomogeneityVerdict:
        """
        评估候选 vs 库内行为指纹。v0.6 修正: 评估本身不写入库
        (原实现 register 会污染库, 使后续批次与未通过候选比较)。
        通过且 persist_on_pass=True 时才显式入库。
        """
        v = HomogeneityVerdict(factor_id=factor_id)
        if not self.enabled or len(self.library) == 0:
            v.reason = "同质性门禁未启用或库为空"
            return v
        # 构建临时指纹 (不写库)
        sig_fp = _jl_projection(signal_series)
        if residual_returns is not None:
            res_fp = _jl_projection(np.asarray(residual_returns, dtype=float))
        elif daily_returns is not None:
            res_fp = _jl_projection(np.asarray(daily_returns, dtype=float))
        else:
            res_fp = np.zeros(JL_DIM)
        prof = BehaviorProfile(
            factor_id, sig_fp, res_fp, int(np.isfinite(signal_series).sum()))
        best = (0.0, "", 0.0, 0.0)
        for fid, other in self.library.items():
            if fid == factor_id:
                continue
            comp, sig, res = behavior_similarity(prof, other)
            if comp > best[0]:
                best = (comp, fid, sig, res)
        comp, nearest, sig, res = best
        v.nearest_similarity = comp
        v.nearest_factor_id = nearest
        v.signal_correlation = sig
        v.residual_correlation = res

        if comp >= self.reject_similarity:
            v.redundancy_label = "NEAR_DUPLICATE"  # 硬拒优先于替代品语义
            v.gate_passed = False
            v.reason = (f"行为冗余硬拒: 与 {nearest} 相似度 {comp:.3f} ≥ {self.reject_similarity} "
                        f"(信号{ sig:.3f}/残差{res:.3f})")
        elif sig >= NEAR_DUPLICATE_SIGNAL and res >= NEAR_DUPLICATE_RESIDUAL:
            v.redundancy_label = "NEAR_DUPLICATE"
            v.gate_passed = False
            v.reason = (f"信号+残差双重冗余: 与 {nearest} "
                        f"(信号{sig:.3f}≥{NEAR_DUPLICATE_SIGNAL}/残差{res:.3f}≥{NEAR_DUPLICATE_RESIDUAL})")
        elif comp >= self.substitute_similarity:
            v.redundancy_label = "SUBSTITUTE"
            v.gate_passed = True   # 替代品可入库但标记 (性能豁免通道在 S6/增量层)
            v.reason = f"SUBSTITUTE: 与 {nearest} 相似度 {comp:.3f}, 标记冗余"
        else:
            v.redundancy_label = "RELATED" if comp >= self.cluster_threshold else "DISTINCT"
            v.gate_passed = True
            v.reason = f"{v.redundancy_label}: 最近 {nearest} 相似度 {comp:.3f}"

        if persist_on_pass and v.gate_passed:
            self.library[factor_id] = prof
        return v

    # ── 库持久化 (v0.6.1: 跨轮次复用行为指纹) ────────────────
    def save(self, path=None) -> int:
        """库指纹落盘 (12维向量 JSON), 返回保存因子数。"""
        try:
            path = path or str(Path(os.environ.get("FUXI_DATA_DIR") or str(Path(__file__).resolve().parent.parent.parent / "data")) / "behavior_library.json")
            data = {k: p.to_dict() for k, p in self.library.items()}
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"n": len(data), "profiles": data}, f, ensure_ascii=False, indent=1)
            return len(data)
        except Exception:
            return 0

    def load(self, path=None) -> int:
        """从落盘库恢复行为指纹, 返回加载因子数。"""
        try:
            path = path or str(Path(os.environ.get("FUXI_DATA_DIR") or str(Path(__file__).resolve().parent.parent.parent / "data")) / "behavior_library.json")
            if not Path(path).exists():
                return 0
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f).get("profiles", {})
            self.library = {k: BehaviorProfile.from_dict(p) for k, p in data.items()}
            return len(self.library)
        except Exception:
            return 0

    # ── 层次聚类 (库级, 生成前指令用) ──────────────────────────
    def cluster_library(self) -> Dict[str, Any]:
        """平均链接层次聚类 + 拥挤簇识别 + 机制画像 (因子数/簇数)。"""
        ids = sorted(self.library.keys())
        n = len(ids)
        clusters: List[List[str]] = [[i] for i in ids]
        sims: Dict[Tuple[str, str], float] = {}
        for a in ids:
            for b in ids:
                if a >= b:
                    continue
                comp, _, _ = behavior_similarity(self.library[a], self.library[b])
                sims[(a, b)] = comp

        # 平均链接聚合
        while len(clusters) > 1:
            best_pair, best_sim = None, -1.0
            cluster_keys = list(range(len(clusters)))
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    s = 0.0
                    cnt = 0
                    for a in clusters[i]:
                        for b in clusters[j]:
                            key = (a, b) if a < b else (b, a)
                            s += sims.get(key, 0.0)
                            cnt += 1
                    avg = s / cnt if cnt else 0.0
                    if avg > best_sim:
                        best_sim, best_pair = avg, (i, j)
            if best_pair is None or best_sim < self.cluster_threshold:
                break
            i, j = best_pair
            clusters[i] = clusters[i] + clusters[j]
            del clusters[j]

        cluster_map: Dict[str, str] = {}
        crowded: List[Dict[str, Any]] = []
        mechanism_counts: Dict[str, int] = {}
        cluster_mechanisms: Dict[str, Dict[str, int]] = {}
        for cl in clusters:
            cid = stable_cluster_id(cl)
            for m in cl:
                cluster_map[m] = cid
            if len(cl) >= self.crowded_size:
                crowded.append({
                    "cluster_id": cid, "size": len(cl),
                    "leader_factor_id": cl[0],
                    "action": "FOLD_TO_LEADER_AND_AVOID_PARAMETER_VARIANTS",
                    "member_sample": cl[:8],
                })
            # 机制统计 (有机制标签时)
            for m in cl:
                mech = getattr(self, "_mech_map", {}).get(m, "unknown")
                mechanism_counts[mech] = mechanism_counts.get(mech, 0) + 1
                cluster_mechanisms.setdefault(cid, {}).setdefault(mech, 0)
                cluster_mechanisms[cid][mech] += 1
        crowded.sort(key=lambda x: -x["size"])
        return {
            "cluster_map": cluster_map,
            "clusters": [[str(x) for x in c] for c in clusters],
            "crowded_clusters": crowded,
            "n_factors": n,
            "n_clusters": len(clusters),
            "mechanism_counts": mechanism_counts,
        }

    def set_mechanism_map(self, mech_map: Dict[str, str]) -> None:
        """factor_id → 机制/范式标签 (用于机制拥挤画像)。"""
        self._mech_map = dict(mech_map)

    # ── 生成前指令 (注入 LLM prompt) ─────────────────────────
    def build_research_directive(self, cluster_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        report = cluster_report or self.cluster_library()
        must_avoid = [c["cluster_id"] for c in report.get("crowded_clusters", [])][:5]
        mech_counts = report.get("mechanism_counts", {})
        # 稀疏机制优先: 因子数最少的机制 top6
        preferred = sorted(mech_counts.items(), key=lambda kv: kv[1])[:6]
        directive = {
            "protocol": "A_SHARE_FACTOR_HOMOGENEITY_CONTROL_V1_FUXI",
            "must_avoid_cluster_ids": must_avoid,
            "preferred_mechanisms": [m for m, _ in preferred],
            "discouraged_mechanisms": sorted(mech_counts.items(), key=lambda kv: -kv[1])[:3],
            "primary_rule": (
                "必须开启新行为簇或稀疏机制; 仅修改 lookback/rank 包装/平滑窗口/符号的参数变体"
                "不构成研究进展, 会被确定性门禁拒绝。"
            ),
            "n_crowded_clusters": len(must_avoid),
        }
        return directive


# ── smoke ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.RandomState(7)
    n_days = 300
    base_mom = rng.normal(0, 1, n_days)
    # 三个库因子: 动量(基准) + 动量换皮(高度相似) + 反转(独立)
    bh = BehaviorHomogeneity(enabled=True, crowded_size=3)
    bh.register("momentum_v1", base_mom)
    bh.register("momentum_v2", base_mom * 0.9 + rng.normal(0, 0.2, n_days))
    bh.register("reversal_v1", -base_mom + rng.normal(0, 1.5, n_days))
    bh.set_mechanism_map({"momentum_v1": "动量", "momentum_v2": "动量", "reversal_v1": "反转"})

    # 候选1: 动量换皮 → 应硬拒
    v1 = bh.evaluate_candidate("momentum_v3", base_mom * 0.85 + rng.normal(0, 0.25, n_days))
    print("[SMOKE] 换皮候选:", v1.to_dict())
    # 候选2: 独立信号 → DISTINCT
    v2 = bh.evaluate_candidate("volume_shape_v1", rng.normal(0, 1, n_days))
    print("[SMOKE] 独立候选:", v2.to_dict())
    # 聚类 + 指令
    report = bh.cluster_library()
    print("[SMOKE] 簇数:", report["n_clusters"], "拥挤簇:", report["crowded_clusters"])
    print("[SMOKE] 指令:", json.dumps(bh.build_research_directive(report), ensure_ascii=False)[:400])
    print("[SMOKE] OK: BehaviorHomogeneity 全部路径可运行")
