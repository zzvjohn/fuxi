# -*- coding: utf-8 -*-
"""P-20260827-001: SSPM 结构化编辑记忆 (Structured Search Process Memory)

AlphaMemo (2026) 机制落地:
  在 (parent_paradigm, edit_mode) 上下文中维护「编辑模式」的信用统计
  (观测数 n / 残差均值 / Welford M2), 高置信负残差触发非对称否决 (APV)。

与既有两层信用的分工:
  - 全局算子信用: GPBreeder._choose_operator_thompson (Beta 后验, v0.4D)
  - 条件化编辑信用: 本模块 (n>=8 且 mean<veto_threshold → 软否决, 默认仅记录)
  - 全局子结构热度: SubstructurePenalizer (P-007, 高频∧JQ差 罚)

Rollback: hard_veto_enabled=False (默认) 时 should_veto 恒为 False,
          只记录统计不改变任何采样行为; 与改造前完全一致。

残差口径 (双通道):
  - FactorForge:  连续 fitness 残差 = 子代 fitness - 父代 fitness (进化循环内)
  - GPBreeder:    S5 二值残差 = 通过 +1 / 未过 S1 -1 (ralph_loop validate 后回填,
                  由候选 source 字段 gp_{op} 解析编辑模式; S1 过 S5 未过记 0 中性)

状态文件: data/substructure_edit_memory.json
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PATH = _PROJECT_ROOT / "data" / "substructure_edit_memory.json"

# 编辑模式词表 (紧凑; 与 GPBreeder source 字段 gp_{op} 对齐)
EDIT_MODES = ("crossover", "mutate", "perturb")


class SSPMEditMemory:
    """(paradigm, edit_mode) → 残差信用统计 + 非对称否决"""

    def __init__(
        self,
        path: Optional[Path] = None,
        hard_veto_enabled: bool = False,   # rollback: 默认仅记录
        min_samples: int = 8,              # 否决证据下限
        veto_threshold: float = -0.01,     # 残差均值低于此值触发否决
        autosave: bool = True,
    ):
        self.path = Path(path) if path else DEFAULT_PATH
        self.hard_veto_enabled = hard_veto_enabled
        self.min_samples = min_samples
        self.veto_threshold = veto_threshold
        self.autosave = autosave
        self.pairs: Dict[str, Dict[str, float]] = {}  # key "paradigm|mode"
        self.load()

    # ── 状态持久化 ────────────────────────────────────

    @staticmethod
    def _key(paradigm: str, edit_mode: str) -> str:
        return f"{paradigm or 'uniform'}|{edit_mode}"

    def load(self) -> None:
        try:
            if self.path.exists():
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
                self.pairs = data.get("pairs", {}) or {}
        except Exception:
            self.pairs = {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            out = {
                "pairs": self.pairs,
                "meta": {
                    "hard_veto_enabled": self.hard_veto_enabled,
                    "min_samples": self.min_samples,
                    "veto_threshold": self.veto_threshold,
                },
            }
            self.path.write_text(
                json.dumps(out, ensure_ascii=False, indent=1, default=str),
                encoding="utf-8")
        except Exception as e:
            print(f"  [SSPM] ⚠️ 状态保存失败 (非阻塞): {e}")

    # ── 统计更新 (Welford) ────────────────────────────

    def record(self, paradigm: str, edit_mode: str, residual: float) -> None:
        """记录一次编辑的残差 (成功为正, 失败为负)"""
        if edit_mode not in EDIT_MODES:
            return
        try:
            residual = float(residual)
        except (TypeError, ValueError):
            return
        if not (-1e6 < residual < 1e6):
            return
        key = self._key(paradigm, edit_mode)
        st = self.pairs.setdefault(key, {"n": 0.0, "mean": 0.0, "m2": 0.0})
        n = st["n"] + 1.0
        delta = residual - st["mean"]
        st["mean"] = st["mean"] + delta / n
        st["m2"] = st["m2"] + delta * (residual - st["mean"])
        st["n"] = n
        if self.autosave:
            self.save()

    # ── 查询 / 否决 ───────────────────────────────────

    def mean_residual(self, paradigm: str, edit_mode: str) -> Optional[float]:
        st = self.pairs.get(self._key(paradigm, edit_mode))
        return float(st["mean"]) if st and st.get("n") else None

    def n_obs(self, paradigm: str, edit_mode: str) -> int:
        st = self.pairs.get(self._key(paradigm, edit_mode))
        return int(st.get("n", 0)) if st else 0

    def should_veto(self, paradigm: str, edit_mode: str) -> bool:
        """非对称否决: n>=min_samples 且 mean<veto_threshold。
        hard_veto_enabled=False (默认) 时恒 False (仅记录)。
        """
        if not self.hard_veto_enabled:
            return False
        st = self.pairs.get(self._key(paradigm, edit_mode))
        if not st or st.get("n", 0) < self.min_samples:
            return False
        return float(st["mean"]) < self.veto_threshold

    def vetoed_pairs(self) -> List[Tuple[str, str]]:
        out = []
        for key in self.pairs:
            p, m = key.rsplit("|", 1)
            if self.should_veto(p, m):
                out.append((p, m))
        return out

    def summary(self, top_n: int = 8) -> str:
        """日志摘要: 观测充足的对按残差排序"""
        rows = []
        for key, st in self.pairs.items():
            if st.get("n", 0) <= 0:
                continue
            p, m = key.rsplit("|", 1)
            rows.append((st["mean"], st["n"], p, m))
        rows.sort()
        if not rows:
            return "  [SSPM] 编辑记忆为空 (尚无足够观测)"
        head = ", ".join(
            f"{p[:16]}×{m}={mean:+.3f}(n={int(n)})" for mean, n, p, m in rows[:top_n]
        )
        veto = [f"{p}×{m}" for p, m in self.vetoed_pairs()]
        veto_s = f" | 否决: {'; '.join(veto[:5])}" if veto else ""
        return f"  [SSPM] 编辑记忆 {len(rows)} 对 | {head}{veto_s} (hard_veto={self.hard_veto_enabled})"
