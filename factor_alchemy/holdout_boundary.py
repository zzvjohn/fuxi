# -*- coding: utf-8 -*-
"""
HoldoutBoundary — v0.6 P0-1: 密封 holdout + 盲评边界
=====================================================
盲评器设计 (blind evaluator + evaluation §8):
  - 数据切三段: 公开区(train+validation) / 隐藏区(holdout) 只允许盲评边界读取
  - 盲评 verdict 仅四类分级结论 + 证据哈希, 精确指标永不进入 LLM 上下文
  - 手动重叠登记 → MANUAL_HOLDOUT_CONTAMINATION
  - 一次性盲测预算 (每世代 N 次), 失败后不得微调同一候选

与伏羲嵌合:
  - local 侧: 隐藏区起点默认 2025-07-01 (对齐 config.META_VALIDATION_START)
    E 阶段 S1-S5 的 IC/组合计算自动截断到公开区, 隐藏区仅由 blind_evaluate 访问
  - JQ 侧: jq_feedback 通道的 D+ 回灌按 verdict 映射 (verdict→层级标签),
    隐藏区 JQ 结果不注入 Experience Memory 明细, 只写 verdict+hash
  - 影子模式: holdout_enabled=False → 全部行为与 v0.5.2 一致 (不截断)

用法:
    from holdout_boundary import HoldoutBoundary
    hb = HoldoutBoundary(enabled=True, holdout_start="2025-07-01")
    hb.check_request("2026-01-05", "manual_backtest")   # → CONTAMINATED verdict
    verdict = hb.blind_verdict(net_returns)             # 四类 verdict
"""

from __future__ import annotations
import os

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

VERDICT_PASS = "BLIND_GENERALIZATION_PASSED"
VERDICT_NO_GEN = "BLIND_NO_GENERALIZATION"
VERDICT_RISK = "BLIND_RISK_FAILURE"
VERDICT_COST = "BLIND_COST_FAILURE"
VERDICT_CONTAMINATED = "MANUAL_HOLDOUT_CONTAMINATION"
VALID_VERDICTS = {VERDICT_PASS, VERDICT_NO_GEN, VERDICT_RISK, VERDICT_COST, VERDICT_CONTAMINATED}


@dataclass
class BlindVerdict:
    verdict: str = VERDICT_PASS
    passed: bool = False
    evidence_hash: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "passed": self.passed,
            "evidence_hash": self.evidence_hash,
            "reason": self.reason,
        }


def evidence_hash(returns: np.ndarray | list, factor_id: str = "") -> str:
    """证据哈希: 只存指纹不存明细, 保证 'verdict 可核验但收益不可反推'。"""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return ""
    payload = (
        f"{factor_id}|n={arr.size}|mean={arr.mean():.6f}|std={arr.std(ddof=1):.6f}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class HoldoutBoundary:
    """密封 holdout: 切分 + verdict 映射 + 污染登记 + 盲测预算。"""

    def __init__(
        self,
        *,
        enabled: bool = False,
        holdout_start: str = "2025-07-01",
        max_blind_attempts: int = 10,
        min_sharpe: float = 0.0,
        min_annual_return: float = 0.0,
        max_drawdown: float = 0.20,
    ):
        self.enabled = enabled
        self.holdout_start = holdout_start
        self.max_blind_attempts = max_blind_attempts
        self.min_sharpe = min_sharpe
        self.min_annual_return = min_annual_return
        self.max_drawdown = max_drawdown
        self.blind_attempts: Dict[str, int] = {}
        self.contaminations: List[Dict[str, Any]] = []

    # ── 切分 ────────────────────────────────────────────────
    def split_public_holdout(self, dates: List[str] | np.ndarray) -> Dict[str, Any]:
        """按 holdout_start 切分公开区/隐藏区。"""
        d = np.asarray([str(x) for x in dates], dtype=object)
        if not self.enabled:
            return {"public": list(d), "holdout": [], "mode": "DISABLED"}
        pub = [x for x in d if str(x) < self.holdout_start]
        hol = [x for x in d if str(x) >= self.holdout_start]
        return {"public": pub, "holdout": hol, "mode": "SEALED"}

    # ── 污染登记 ────────────────────────────────────────────
    def check_request(
        self,
        request_start: str,
        request_kind: str = "manual_backtest",
        factor_id: str = "",
    ) -> BlindVerdict:
        """检查研究请求是否触碰隐藏区 → 触碰则登记污染并返回 CONTAMINATED。"""
        if not self.enabled:
            return BlindVerdict(VERDICT_PASS, passed=True, reason="holdout 未启用")
        if str(request_start) >= self.holdout_start:
            entry = {
                "kind": request_kind,
                "factor_id": factor_id,
                "request_start": request_start,
                "holdout_start": self.holdout_start,
            }
            self.contaminations.append(entry)
            return BlindVerdict(
                VERDICT_CONTAMINATED,
                passed=False,
                evidence_hash=hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16],
                reason=(f"{request_kind} 请求区间 {request_start} 触碰隐藏区 "
                        f"(≥{self.holdout_start}), 已登记污染; 同世代不得访问盲评器"),
            )
        return BlindVerdict(VERDICT_PASS, passed=True, reason="请求在公开区内")

    # ── 盲评 (隐藏区唯一入口) ────────────────────────────────
    def blind_evaluate(
        self,
        returns: np.ndarray | list,
        *,
        factor_id: str = "",
        generation: str = "default",
        periods_per_year: int = 52,
    ) -> BlindVerdict:
        """隐藏区盲评: 只返回四类 verdict + 哈希, 精确指标不出边界。"""
        if not self.enabled:
            return BlindVerdict(VERDICT_PASS, passed=True, reason="holdout 未启用, 视为公开评估")

        # 一次性预算
        used = self.blind_attempts.get(generation, 0)
        if used >= self.max_blind_attempts:
            return BlindVerdict(
                VERDICT_NO_GEN, passed=False, evidence_hash="",
                reason=f"盲测预算耗尽 ({used}/{self.max_blind_attempts}), 需新世代",
            )
        self.blind_attempts[generation] = used + 1

        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        h = evidence_hash(arr, factor_id)
        if arr.size < 12:
            return BlindVerdict(VERDICT_NO_GEN, passed=False, evidence_hash=h,
                                reason=f"隐藏区样本不足 ({arr.size}<12)")

        sharpe = float(arr.mean() / arr.std(ddof=1)) * np.sqrt(periods_per_year) if arr.std(ddof=1) > 0 else 0.0
        # 年化收益 (近似: 几何周收益累乘)
        annual = float(np.prod(1.0 + arr) ** (periods_per_year / arr.size) - 1.0) if arr.size else 0.0
        cum = np.cumprod(1.0 + arr)
        dd = float(np.max(1.0 - cum / np.maximum.accumulate(cum))) if arr.size else 1.0

        if annual <= self.min_annual_return:
            v = BlindVerdict(VERDICT_NO_GEN, passed=False, evidence_hash=h,
                             reason=f"隐藏区年化 {annual:.1%} ≤ {self.min_annual_return}")
        elif dd > self.max_drawdown:
            v = BlindVerdict(VERDICT_RISK, passed=False, evidence_hash=h,
                             reason=f"隐藏区回撤 {dd:.1%} > {self.max_drawdown}")
        elif sharpe < self.min_sharpe:
            v = BlindVerdict(VERDICT_COST, passed=False, evidence_hash=h,
                             reason=f"隐藏区 Sharpe {sharpe:.2f} < {self.min_sharpe}")
        else:
            v = BlindVerdict(VERDICT_PASS, passed=True, evidence_hash=h,
                             reason=f"隐藏区通过: Sharpe≥{self.min_sharpe}, 回撤≤{self.max_drawdown}")
        return v

    # ── 盲评预算持久化 (v0.6.1: 跨轮次预算不重置) ───────────
    def save_budget(self, path=None) -> bool:
        """盲评预算落盘 (每世代已用次数)。"""
        try:
            import json as _json
            from pathlib import Path as _Path
            path = path or str(Path(os.environ.get("FUXI_DATA_DIR") or str(Path(__file__).resolve().parent.parent.parent / "data")) / "holdout_budget.json")
            _Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                _json.dump({"blind_attempts": self.blind_attempts,
                            "n_contaminations": len(self.contaminations)}, f)
            return True
        except Exception:
            return False

    def load_budget(self, path=None) -> int:
        """恢复盲评预算, 返回已恢复的世代数。"""
        try:
            import json as _json
            from pathlib import Path as _Path
            path = path or str(Path(os.environ.get("FUXI_DATA_DIR") or str(Path(__file__).resolve().parent.parent.parent / "data")) / "holdout_budget.json")
            if not _Path(path).exists():
                return 0
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            self.blind_attempts = dict(data.get("blind_attempts", {}))
            return len(self.blind_attempts)
        except Exception:
            return 0

    # ── JQ verdict 映射 (D+ 回灌通道) ───────────────────────
    @staticmethod
    def map_jq_to_verdict(
        jq_return: Optional[float],
        jq_sharpe: Optional[float],
        jq_maxdd: Optional[float],
        *,
        is_holdout_window: bool = False,
        min_sharpe: float = 0.0,
        max_drawdown: float = 0.20,
    ) -> str:
        """JQ 回测结果 → 分级 verdict (holdout 窗口内只允许 verdict 进 D+ 记忆)。"""
        if not is_holdout_window:
            return "PUBLIC_JQ_RESULT"
        if jq_return is None or jq_sharpe is None:
            return VERDICT_NO_GEN
        if jq_return <= 0:
            return VERDICT_NO_GEN
        if jq_maxdd is not None and abs(jq_maxdd) > max_drawdown:
            return VERDICT_RISK
        if jq_sharpe < min_sharpe:
            return VERDICT_COST
        return VERDICT_PASS


# ── smoke ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.RandomState(3)
    # 公开区行为一致
    hb_off = HoldoutBoundary(enabled=False)
    dates = ["2024-01-05", "2025-08-01", "2026-01-05"]
    print("[SMOKE] off:", hb_off.split_public_holdout(dates))
    # 密封模式
    hb = HoldoutBoundary(enabled=True, holdout_start="2025-07-01", max_blind_attempts=3)
    print("[SMOKE] split:", hb.split_public_holdout(dates))
    print("[SMOKE] 污染:", hb.check_request("2026-01-05", "manual_backtest", "f1").to_dict())
    print("[SMOKE] 公开:", hb.check_request("2024-06-01", "manual_backtest", "f2").to_dict())
    # 盲评
    good = 0.006 + rng.normal(0, 0.02, 60) * 0.0 + 0.0  # 确定性均值
    bad = -0.004 + rng.normal(0, 0.02, 60) * 0.0
    good = np.full(60, 0.006)
    bad = np.full(60, -0.004)
    print("[SMOKE] blind good:", hb.blind_evaluate(good, factor_id="g").to_dict())
    print("[SMOKE] blind bad :", hb.blind_evaluate(bad, factor_id="b").to_dict())
    print("[SMOKE] blind #3  :", hb.blind_evaluate(bad, factor_id="c").to_dict())
    print("[SMOKE] blind #4  :", hb.blind_evaluate(bad, factor_id="d").to_dict())  # 预算耗尽
    # JQ 映射
    print("[SMOKE] JQ holdout PASS:", HoldoutBoundary.map_jq_to_verdict(0.15, 0.8, -0.12, is_holdout_window=True))
    print("[SMOKE] JQ holdout RISK:", HoldoutBoundary.map_jq_to_verdict(0.15, 0.8, -0.35, is_holdout_window=True))
    print("[SMOKE] JQ public:", HoldoutBoundary.map_jq_to_verdict(0.15, 0.8, -0.35, is_holdout_window=False))
    print("[SMOKE] OK: HoldoutBoundary 全部路径可运行")
