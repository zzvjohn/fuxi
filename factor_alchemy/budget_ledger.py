"""
P2: 族级 + 世代预算账本 (family/generation budget)

目的: 防止进化过程在少数"拥挤族"上无限堆候选, 控制每世代总候选规模,
      把稀缺的评估预算(JQ 回测时间/金钱)分配到机制多样的探索上。

铁则: 本模块只做"预算检查/记账", 不参与因子质量判定 (不碰筛选毒药区)。
      开关 V06_EXPERIMENTAL["budget_ledger_enabled"]=False 时所有检查直接放行。

用法:
    ledger = BudgetLedger()
    ok, reason = ledger.can_generate(family_id="reversal", generation=12)
    if ok:
        ledger.consume(family_id="reversal", generation=12, factor_id="f_xxx")
"""
import os
import json
import threading
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(os.environ.get("FUXI_DATA_DIR") or str(Path(__file__).resolve().parent.parent.parent / "data"))

_BUDGET_FILE = DATA_DIR / "budget_ledger.json"
_LOCK = threading.Lock()


class BudgetLedger:
    """族级 + 世代配额账本 (append-only 消费记录 + 可重算余额)"""

    def __init__(
        self,
        path: Path = _BUDGET_FILE,
        max_per_family: int = 100,
        max_per_generation: int = 500,
        enabled: bool = False,
    ):
        self.path = path
        self.max_per_family = max_per_family
        self.max_per_generation = max_per_generation
        self.enabled = enabled
        self._consumption = []  # [{family_id, generation, factor_id, ts}]
        self._load()

    # ── 持久化 ──────────────────────────────────────────
    def _load(self):
        try:
            if self.path.exists():
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._consumption = data.get("consumption", [])
        except Exception:
            self._consumption = []

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _LOCK:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "updated_at": datetime.now().isoformat(),
                            "max_per_family": self.max_per_family,
                            "max_per_generation": self.max_per_generation,
                            "consumption": self._consumption,
                        },
                        f, ensure_ascii=False, indent=1,
                    )
        except Exception:
            pass  # 账本写失败不阻塞流水线 (预算为软约束)

    # ── 查询/消费 ───────────────────────────────────────
    def family_count(self, family_id: str, generation: int = None) -> int:
        if generation is None:
            return sum(1 for c in self._consumption if c["family_id"] == family_id)
        return sum(
            1 for c in self._consumption
            if c["family_id"] == family_id and c["generation"] == generation
        )

    def generation_count(self, generation: int) -> int:
        return sum(1 for c in self._consumption if c["generation"] == generation)

    def can_generate(self, family_id: str, generation: int) -> tuple:
        """
        预算检查。返回 (allow: bool, reason: str)。
        开关关闭时恒放行 (rollback 安全)。
        """
        if not self.enabled:
            return True, "BUDGET_DISABLED"
        if self.generation_count(generation) >= self.max_per_generation:
            return False, f"世代预算耗尽: gen={generation} 已消费 {self.generation_count(generation)}/{self.max_per_generation}"
        if self.family_count(family_id) >= self.max_per_family:
            return False, f"族预算耗尽: {family_id} 已消费 {self.family_count(family_id)}/{self.max_per_family}"
        return True, "OK"

    def consume(self, family_id: str, generation: int, factor_id: str) -> bool:
        """记账一次消费 (生成候选时调用)。返回是否成功写入。"""
        self._consumption.append({
            "family_id": family_id,
            "generation": int(generation),
            "factor_id": factor_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
        })
        self._save()
        return True

    def summary(self, top_families: int = 10) -> dict:
        """账本摘要: 总消费、世代分布、最拥挤族 Top-N (供日报/监控用)"""
        from collections import Counter
        fam = Counter(c["family_id"] for c in self._consumption)
        gen = Counter(c["generation"] for c in self._consumption)
        return {
            "total_consumed": len(self._consumption),
            "n_families": len(fam),
            "top_families": fam.most_common(top_families),
            "generations": dict(sorted(gen.items())),
        }


def get_ledger() -> BudgetLedger:
    """工厂函数: 从 config 读开关与配额 (config 不可用时回退禁用态)"""
    try:
        from config import V06_EXPERIMENTAL
        return BudgetLedger(
            enabled=bool(V06_EXPERIMENTAL.get("budget_ledger_enabled", False)),
            max_per_family=int(V06_EXPERIMENTAL.get("budget_max_per_family", 100)),
            max_per_generation=int(V06_EXPERIMENTAL.get("budget_max_per_generation", 500)),
        )
    except Exception:
        return BudgetLedger(enabled=False)


if __name__ == "__main__":
    import tempfile
    p = Path(tempfile.gettempdir()) / "smoke_budget_ledger.json"
    if p.exists():
        p.unlink()

    # 关闭态: 恒放行
    off = BudgetLedger(path=p, enabled=False, max_per_family=2, max_per_generation=3)
    assert off.can_generate("reversal", 1)[0] is True
    print("[SMOKE] 开关关闭: 恒放行 OK")

    # 开启态: 族级/世代配额生效
    on = BudgetLedger(path=p, enabled=True, max_per_family=2, max_per_generation=3)
    assert on.can_generate("reversal", 1)[0] is True
    on.consume("reversal", 1, "f1")
    on.consume("reversal", 1, "f2")
    allow, reason = on.can_generate("reversal", 1)
    assert allow is False and "族预算耗尽" in reason, reason
    print(f"[SMOKE] 族级配额拦截: {reason}")

    assert on.can_generate("momentum", 1)[0] is True  # 其他族不受影响
    on.consume("momentum", 1, "f3")
    allow, reason = on.can_generate("value", 1)
    assert allow is False and "世代预算耗尽" in reason, reason
    print(f"[SMOKE] 世代配额拦截: {reason}")

    s = on.summary()
    assert s["total_consumed"] == 3
    print(f"[SMOKE] 摘要: total={s['total_consumed']} top={s['top_families']}")

    # 持久化重载
    on2 = BudgetLedger(path=p, enabled=True, max_per_family=2, max_per_generation=3)
    assert on2.family_count("reversal") == 2
    print("[SMOKE] 持久化重载 OK")
    p.unlink()
    print("[SMOKE] OK: BudgetLedger 全部路径可运行")
