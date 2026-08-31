"""
P2: 人工回测污染账本 (MANUAL_HOLDOUT_CONTAMINATION)

目的: 密封 holdout 的完整性依赖"盲评区数据不被人类触碰"。凡有人工(或外部工具)
      查看/使用了 holdout 区数据, 必须登记。登记后:
      - 盲评时该因子/日期段的 verdict 降级 (MARGINAL→NO_GENERALIZATION) 或标记 SUSPECT
      - 升级报告可追溯污染来源

铁则: 只记账, 不拦截。判定降级由 holdout_boundary 消费本账本实现。
      开关 V06_EXPERIMENTAL["manual_contamination_enabled"]=False 时 record() 静默忽略。

事件字段: scope(污染范围: factor_id/date_range/both), factor_id, date_start, date_end,
          actor(操作主体), reason(原因), ts。
"""
import os
import json
import threading
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(os.environ.get("FUXI_DATA_DIR") or str(Path(__file__).resolve().parent.parent.parent / "data"))

_CONTAMINATION_FILE = DATA_DIR / "manual_contamination_ledger.json"
_LOCK = threading.Lock()


class ContaminationLedger:
    """人工回测污染账本 (append-only 登记, 支持范围查询)"""

    def __init__(self, path: Path = _CONTAMINATION_FILE, enabled: bool = False):
        self.path = path
        self.enabled = enabled
        self._events = []
        self._load()

    # ── 持久化 ──────────────────────────────────────────
    def _load(self):
        try:
            if self.path.exists():
                with open(self.path, "r", encoding="utf-8") as f:
                    self._events = json.load(f).get("events", [])
        except Exception:
            self._events = []

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _LOCK:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"updated_at": datetime.now().isoformat(), "events": self._events},
                        f, ensure_ascii=False, indent=1,
                    )
        except Exception:
            pass

    # ── 登记/查询 ───────────────────────────────────────
    def record(
        self,
        scope: str,
        actor: str,
        reason: str,
        factor_id: str = None,
        date_start: str = None,
        date_end: str = None,
    ) -> bool:
        """
        登记一次污染事件。
        scope: "factor" | "date_range" | "both"
        """
        if not self.enabled:
            return False  # 开关关闭: 不记账 (rollback 安全)
        ev = {
            "id": (max((e.get("id", 0) for e in self._events), default=0) + 1),
            "scope": scope,
            "factor_id": factor_id,
            "date_start": date_start,
            "date_end": date_end,
            "actor": actor,
            "reason": reason[:500],
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        self._events.append(ev)
        self._save()
        return True

    def is_factor_contaminated(self, factor_id: str) -> bool:
        return any(
            e["factor_id"] == factor_id and e["scope"] in ("factor", "both")
            for e in self._events
        )

    def is_range_contaminated(self, date_start: str, date_end: str) -> bool:
        """日期段 [date_start, date_end] 是否与任何已登记日期段重叠"""
        for e in self._events:
            if e["scope"] not in ("date_range", "both"):
                continue
            es, ee = e.get("date_start"), e.get("date_end")
            if not es:
                continue
            ee = ee or es
            # 字符串 ISO 日期可直接字典序比较
            overlap = (date_start <= ee) and (es <= date_end)
            if overlap:
                return True
        return False

    def check(self, factor_id: str = None, date_start: str = None, date_end: str = None) -> dict:
        """综合检查: 返回 {contaminated: bool, reasons: [...]}"""
        hits = []
        for e in self._events:
            if factor_id and e["factor_id"] == factor_id and e["scope"] in ("factor", "both"):
                hits.append(e)
            if date_start and date_end and e["scope"] in ("date_range", "both") and e.get("date_start"):
                es, ee = e.get("date_start"), e.get("date_end", e.get("date_start"))
                if (date_start <= ee) and (es <= date_end):
                    hits.append(e)
        # 去重 (同事件可能同时命中因子与日期; 以自增 id 为准, ts 同秒会撞)
        seen, uniq = set(), []
        for h in hits:
            k = h.get("id", h["ts"])
            if k not in seen:
                seen.add(k)
                uniq.append(h)
        return {"contaminated": len(uniq) > 0, "events": uniq}

    def report(self, limit: int = 20) -> str:
        """人类可读报告 (供日报/升级报告引用)"""
        if not self._events:
            return "污染账本: 无登记事件"
        lines = [f"污染账本: 共 {len(self._events)} 次人工触碰登记"]
        for e in self._events[-limit:]:
            rng = ""
            if e.get("date_start"):
                rng = f" [{e['date_start']}~{e.get('date_end', e['date_start'])}]"
            fid = f" factor={e['factor_id']}" if e.get("factor_id") else ""
            lines.append(f"- {e['ts']} {e['actor']}: {e['reason'][:80]}{rng}{fid}")
        return "\n".join(lines)


def get_contamination_ledger() -> ContaminationLedger:
    """工厂函数: 从 config 读开关 (config 不可用时回退禁用态)"""
    try:
        from config import V06_EXPERIMENTAL
        return ContaminationLedger(
            enabled=bool(V06_EXPERIMENTAL.get("manual_contamination_enabled", False))
        )
    except Exception:
        return ContaminationLedger(enabled=False)


if __name__ == "__main__":
    import tempfile
    p = Path(tempfile.gettempdir()) / "smoke_contamination.json"
    if p.exists():
        p.unlink()

    # 关闭态: record 不记账
    off = ContaminationLedger(path=p, enabled=False)
    assert off.record("factor", "zhong", "手滑翻了下数据", factor_id="f1") is False
    assert off.check(factor_id="f1")["contaminated"] is False
    print("[SMOKE] 开关关闭: 不记账 OK")

    # 开启态
    on = ContaminationLedger(path=p, enabled=True)
    on.record("factor", "zhong", "核对某因子 holdout 区收益", factor_id="f1")
    on.record("date_range", "research-agent", "下载了 2026-01 全量面板", date_start="2026-01-01", date_end="2026-01-31")

    assert on.is_factor_contaminated("f1") is True
    assert on.is_factor_contaminated("f2") is False
    assert on.is_range_contaminated("2026-01-15", "2026-01-20") is True
    assert on.is_range_contaminated("2025-06-01", "2025-06-30") is False

    chk = on.check(factor_id="f1", date_start="2026-01-10", date_end="2026-01-20")
    assert chk["contaminated"] is True and len(chk["events"]) == 2
    print(f"[SMOKE] 综合检查: {len(chk['events'])} 条命中")

    # 持久化重载
    on2 = ContaminationLedger(path=p, enabled=True)
    assert on2.is_factor_contaminated("f1") is True
    print("[SMOKE] 持久化重载 OK")
    print(on2.report())
    p.unlink()
    print("[SMOKE] OK: ContaminationLedger 全部路径可运行")
