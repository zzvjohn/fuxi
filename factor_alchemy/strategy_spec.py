"""
P2: 策略规格对象化

目的: 把"研究端信号 + JQ 模板参数"的交付物固化为单一规格对象, 供:
      - 计划执行器 (P4 p4_plan_exec_jq.py 族) 生成时校验完整性
      - 升级报告/审计追踪 (谁、何时、基于什么信号源、什么参数)
      - D+ 阶段 JQ 反馈回灌时精确挂载 (避免手工对不上信号文件)

铁则: 纯交付物规格, 不参与流水线判定 (config 开关 strategy_spec_enabled
      控制的是"是否强制校验", 不影响任何质量门)。

规格字段:
    name / signal_source(信号CSV路径或表名) / template_version(执行模板)
    n_drop(组合持仓数) / rebalance_freq / capacity_estimate
    universe / factor_ids / jq_metrics(回测结果回填) / created_at / status
"""
import os
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

DATA_DIR = Path(os.environ.get("FUXI_DATA_DIR") or str(Path(__file__).resolve().parent.parent.parent / "data"))
SPEC_DIR = DATA_DIR / "strategy_specs"

REQUIRED_FIELDS = [
    "name", "signal_source", "template_version", "n_drop",
    "rebalance_freq", "universe",
]

VALID_STATUS = ("draft", "research_validated", "jq_pending", "jq_validated", "live", "retired")


@dataclass
class StrategySpec:
    """JQ 策略交付物规格 (研究端→执行端的交接契约)"""
    name: str
    signal_source: str                      # 信号 CSV/表 (e.g. "signals/p2_scores_jq_fuse.csv")
    template_version: str                   # 执行模板版本 (e.g. "v1.2")
    n_drop: int                             # 组合持仓数 (topk)
    rebalance_freq: str                     # "weekly" / "monthly"
    universe: str                           # 股票池口径 (e.g. "全A除ST/退市整理")
    factor_ids: List[str] = field(default_factory=list)
    capacity_estimate: str = ""             # 容量区间 (e.g. "3000万~1亿")
    risk_params: Dict = field(default_factory=dict)   # 止损/集中度等
    jq_metrics: Dict = field(default_factory=dict)    # 回测结果回填 {return, sharpe, maxdd, ic...}
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = ""
    status: str = "draft"
    notes: str = ""

    # ── 校验 ────────────────────────────────────────────
    def validate(self) -> tuple:
        """完整性校验。返回 (ok, missing_fields/错误信息)"""
        missing = [f for f in REQUIRED_FIELDS if not getattr(self, f, None)]
        if missing:
            return False, f"缺必备字段: {missing}"
        if self.status not in VALID_STATUS:
            return False, f"非法 status: {self.status} (合法={VALID_STATUS})"
        if self.n_drop <= 0:
            return False, f"n_drop 必须 > 0, got {self.n_drop}"
        if self.rebalance_freq not in ("weekly", "monthly", "daily"):
            return False, f"非法 rebalance_freq: {self.rebalance_freq}"
        return True, "OK"

    # ── 序列化 ──────────────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategySpec":
        known = {f: d.get(f) for f in cls.__dataclass_fields__}
        # 兼容旧文件缺新字段
        for f in ("jq_metrics", "risk_params", "factor_ids"):
            if known[f] is None:
                known[f] = {} if f != "factor_ids" else []
        return cls(**known)

    def save(self, path: Path = None) -> Path:
        path = path or (SPEC_DIR / f"{self.name}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now().isoformat(timespec="seconds")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: Path) -> "StrategySpec":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def enforce_spec(fn=None, *, enabled: bool = True):
    """装饰器: 若 strategy_spec_enabled, 被装饰函数返回的 StrategySpec
    必须通过 validate(), 否则抛 ValueError (fail-fast, 早于 JQ 挂机)。"""
    import functools

    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if enabled and isinstance(result, StrategySpec):
                ok, err = result.validate()
                if not ok:
                    raise ValueError(f"策略规格校验失败: {err}")
            return result
        return wrapper
    return deco(fn) if fn else deco


if __name__ == "__main__":
    import tempfile
    # 完整规格: 通过
    spec = StrategySpec(
        name="p4_topk20_fwd20_fuse",
        signal_source="signals/p4_plan_topk20.csv",
        template_version="v1.3",
        n_drop=20,
        rebalance_freq="weekly",
        universe="全A (除ST/退市整理, 无面值过滤)",
        factor_ids=["fwd20_rank", "fwd40_rank"],
        capacity_estimate="3000万~1亿",
        jq_metrics={"return_pct": 625.39, "maxdd": -12.6},
        status="jq_validated",
    )
    ok, err = spec.validate()
    assert ok, err
    print("[SMOKE] 完整规格校验通过")

    # 缺字段: 拦截
    bad = StrategySpec(name="x", signal_source="", template_version="", n_drop=0, rebalance_freq="weekly", universe="")
    ok, err = bad.validate()
    assert not ok and "缺必备字段" in err
    print(f"[SMOKE] 缺字段拦截: {err}")

    # 非法 status / n_drop / 频率
    bad2 = StrategySpec(name="x", signal_source="s", template_version="t", n_drop=-1, rebalance_freq="hourly", universe="u", status="weird")
    ok, err = bad2.validate()
    assert not ok
    print(f"[SMOKE] 非法参数拦截: {err}")

    # 序列化往返
    p = Path(tempfile.gettempdir()) / "smoke_spec.json"
    spec.save(p)
    loaded = StrategySpec.load(p)
    assert loaded.to_dict() == spec.to_dict()
    print("[SMOKE] 序列化往返 OK")

    # enforce_spec 装饰器
    @enforce_spec(enabled=True)
    def make_bad():
        return bad
    try:
        make_bad()
        raise AssertionError("应抛 ValueError")
    except ValueError as e:
        print(f"[SMOKE] enforce_spec 拦截: {e}")
    p.unlink()
    print("[SMOKE] OK: StrategySpec 全部路径可运行")
