# -*- coding: utf-8 -*-
"""
DirectionCampaign — v0.6 P1-2: 方向微战役 (MAB 上层约束)
=========================================================
自适应方向微战役 (evaluation §8.2):
  系统不得逐轮追逐最近最高夏普; 方向由确定性诊断器选定后冻结为微战役:
    1. 每战役最多 N 候选 (修复/执行失败都消耗次数)
    2. 连续 K 次无公开组合改善 → 提前停止
    3. 目标完全修复 → 提前成功结束
    4. 已结束方向冷却 C 个战役, 不得因"接近门槛"延长
    5. 方向/基线/依据/尝试/结束原因全部进持久账本
    6. 隐藏区结论不参与方向选择或战役成败

与伏羲嵌合:
  - 位置: RalphLoop G 阶段之前 (MAB 选方向后, 生成前), 作为 MAB 的上层约束
  - MAB 建议的方向进入 campaign 状态机: 冷却期/预算耗尽 → 覆盖 MAB 选择
  - 账本: JSONL append-only (data/campaign_ledger.jsonl)

用法:
    from direction_campaign import DirectionCampaign
    dc = DirectionCampaign(max_attempts=3, early_stop_misses=3, cooldown=2)
    direction = dc.resolve(mab_direction="资金流", public_improved=False)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

LEDGER_PATH = Path(__file__).parent / "data" / "campaign_ledger.jsonl"


@dataclass
class CampaignState:
    campaign_id: str = ""
    direction: str = ""
    objective: str = ""
    attempts_used: int = 0
    consecutive_misses: int = 0
    status: str = "ACTIVE"           # ACTIVE / SUCCESS / EARLY_STOP / COOLED / BUDGET_EXHAUSTED
    cooldown_until_round: int = -1
    created_round: int = 0
    ended_round: int = -1
    end_reason: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "direction": self.direction,
            "objective": self.objective,
            "attempts_used": self.attempts_used,
            "consecutive_misses": self.consecutive_misses,
            "status": self.status,
            "cooldown_until_round": self.cooldown_until_round,
            "created_round": self.created_round,
            "ended_round": self.ended_round,
            "end_reason": self.end_reason,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CampaignState":
        return cls(**{k: d.get(k, v) for k, v in cls().to_dict().items()})


class DirectionCampaign:
    """方向微战役状态机 (append-only 账本 + 冷却 + 早停)。"""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        early_stop_misses: int = 3,
        cooldown_campaigns: int = 2,
        ledger_path: Path = LEDGER_PATH,
        enabled: bool = False,
    ):
        self.max_attempts = max_attempts
        self.early_stop_misses = early_stop_misses
        self.cooldown_campaigns = cooldown_campaigns
        self.ledger_path = ledger_path
        self.enabled = enabled
        self.active: Dict[str, CampaignState] = {}
        self.round = 0
        self._load_ledger()

    # ── 账本 ────────────────────────────────────────────────
    def _load_ledger(self) -> None:
        if not self.ledger_path.exists():
            return
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    st = CampaignState.from_dict(rec)
                    if st.status == "ACTIVE":
                        self.active[st.campaign_id] = st
                    if rec.get("round", 0) > self.round:
                        self.round = int(rec["round"])
        except Exception:
            pass  # 账本损坏不阻塞

    def _append_ledger(self, state: CampaignState) -> None:
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            rec = state.to_dict()
            rec["round"] = self.round
            rec["ts"] = __import__("datetime").datetime.now().isoformat()
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 账本写入失败不阻塞研究

    # ── 核心: resolve ───────────────────────────────────────
    def resolve(
        self,
        mab_direction: str,
        *,
        objective: str = "",
        public_improved: Optional[bool] = None,
        generation_id: str = "default",
    ) -> Dict[str, Any]:
        """每轮 G 阶段前调用: 返回 (是否放行, 最终方向, 战役上下文)。"""
        if not self.enabled:
            return {"allow": True, "direction": mab_direction, "mode": "DISABLED", "campaign_id": "UNCONSTRAINED"}
        self.round += 1

        # 1) 若有活跃战役 → 先收尾判定
        active_cid = f"{generation_id}:{mab_direction}"
        if mab_direction in (c.direction for c in self.active.values()) and active_cid in self.active:
            st = self.active[active_cid]
            if public_improved is not None:
                if public_improved:
                    st.consecutive_misses = 0
                else:
                    st.consecutive_misses += 1
                st.attempts_used += 1
                st.history.append({"round": self.round, "improved": public_improved})
                if st.attempts_used >= self.max_attempts:
                    st.status = "BUDGET_EXHAUSTED"
                    st.ended_round = self.round
                    st.end_reason = f"战役预算耗尽 ({self.max_attempts} 候选)"
                    self._append_ledger(st)
                    del self.active[active_cid]
                    return self._fallback(mab_direction, f"预算耗尽: {st.direction}")
                if st.consecutive_misses >= self.early_stop_misses:
                    st.status = "EARLY_STOP"
                    st.ended_round = self.round
                    st.end_reason = f"连续 {self.early_stop_misses} 次无改善提前停止"
                    self._append_ledger(st)
                    del self.active[active_cid]
                    return self._fallback(mab_direction, f"早停: {st.direction}")
                self._append_ledger(st)
                return {"allow": True, "direction": st.direction, "mode": "CAMPAIGN",
                        "campaign_id": active_cid, "remaining": self.max_attempts - st.attempts_used}

        # 2) 冷却检查: 同方向最近结束的战役
        cooled = self._is_cooled(mab_direction, generation_id)
        if cooled:
            return self._fallback(mab_direction, "冷却期内")

        # 3) 新战役
        st = CampaignState(
            campaign_id=active_cid, direction=mab_direction, objective=objective,
            created_round=self.round,
        )
        self.active[active_cid] = st
        self._append_ledger(st)
        return {"allow": True, "direction": mab_direction, "mode": "CAMPAIGN",
                "campaign_id": active_cid, "remaining": self.max_attempts}

    def _is_cooled(self, direction: str, generation_id: str) -> bool:
        if not self.ledger_path.exists():
            return False
        try:
            recent = []
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("direction") == direction and rec.get("status") in ("EARLY_STOP", "BUDGET_EXHAUSTED"):
                        recent.append(rec)
            if not recent:
                return False
            last_end = max(int(r.get("ended_round", r.get("round", 0))) for r in recent)
            return (self.round - last_end) <= self.cooldown_campaigns
        except Exception:
            return False

    def _fallback(self, mab_direction: str, reason: str) -> Dict[str, Any]:
        """方向被拦截 → 返回 None 方向 (由调用方走第二优先级/范式轮转)。"""
        return {"allow": False, "direction": None, "mode": "BLOCKED",
                "reason": reason, "campaign_id": "UNCONSTRAINED"}

    def mark_success(self, campaign_id: str) -> None:
        if campaign_id in self.active:
            st = self.active.pop(campaign_id)
            st.status = "SUCCESS"
            st.ended_round = self.round
            st.end_reason = "公开目标完全修复, 提前成功结束"
            self._append_ledger(st)

    def ledger_summary(self) -> Dict[str, Any]:
        n_total, by_status = 0, {}
        try:
            if self.ledger_path.exists():
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        n_total += 1
                        rec = json.loads(line)
                        by_status[rec.get("status", "?")] = by_status.get(rec.get("status", "?"), 0) + 1
        except Exception:
            pass
        return {"n_records": n_total, "by_status": by_status, "active_campaigns": len(self.active)}


# ── smoke ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os
    tmp = Path(tempfile.mkdtemp()) / "campaign.jsonl"
    dc = DirectionCampaign(max_attempts=3, early_stop_misses=3, cooldown_campaigns=2,
                           ledger_path=tmp, enabled=True)
    # 战役 1: 连续 miss → 早停
    print("[SMOKE] r1:", dc.resolve("资金流", objective="跨期稳定性"))
    for i in range(3):
        r = dc.resolve("资金流", public_improved=False)
        print(f"[SMOKE] miss{i+1}:", {k: r[k] for k in ("allow", "mode", "reason") if k in r})
    # 冷却: 同方向立刻再试 → BLOCKED
    r = dc.resolve("资金流")
    print("[SMOKE] 冷却拦截:", {k: r[k] for k in ("allow", "mode", "reason")})
    # 其他方向不受影响
    r2 = dc.resolve("微观结构", objective="交易效率")
    print("[SMOKE] 新方向:", r2["mode"], r2["direction"])
    # 预算耗尽路径
    dc2 = DirectionCampaign(max_attempts=2, early_stop_misses=5, cooldown_campaigns=0,
                            ledger_path=Path(tempfile.mkdtemp()) / "c2.jsonl", enabled=True)
    dc2.resolve("尾部风险")
    print("[SMOKE] 预算-1:", {k: dc2.resolve("尾部风险", public_improved=True)[k] for k in ("allow", "mode")})
    print("[SMOKE] 预算-2:", {k: dc2.resolve("尾部风险", public_improved=True)[k] for k in ("allow", "mode", "reason")})
    print("[SMOKE] summary:", dc.ledger_summary())
    print("[SMOKE] OK: DirectionCampaign 全部路径可运行")
