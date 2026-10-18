"""
P-20260825-007: 表达式子结构频率库 + JQ 联动降采样
====================================================

对标中金 Loop Engineering 的"频繁子结构规避"机制 (600轮1.5万因子筛69个)。

核心机制:
1. 三元组子结构提取: 公式 → FactorExpressionParser → skeleton → 函数名序列滑动窗口(3)
2. 全局热度表: {triplet: {freq, jq_n, jq_fail}}, 持久化 data/substructure_freq.json
3. JQ 联动 (用户钦定 2026-08-25):
   - 高频 ∧ JQ 失败率高 → 强降采样 (penalty = heat_norm * 1.0)
   - 高频但 JQ 表现好 → 保留     (penalty = heat_norm * 0.1)
   - JQ 数据不足 → 中性基础罚    (penalty = heat_norm * 0.4)
4. 软拒绝 (非硬禁止): gp_breed/forge 生成时按 penalty 概率拒绝并重试
5. 审计: Ralph 每轮输出 Top10 高频子结构

Rollback: penalizer=None 时所有调用方行为与改造前完全一致。
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DATA_DIR = Path(__file__).parent.parent.parent / "data"
FREQ_PATH = DATA_DIR / "substructure_freq.json"

_FUNC_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\(")

# JQ 联动阈值 (离线可调)
JQ_MIN_N = 3          # JQ 样本不足此数视为"数据不足"
JQ_GOOD_FAIL_RATE = 0.40   # fail_rate <= 此值 → JQ 表现好 → 保留
JQ_BAD_FAIL_RATE = 0.60    # fail_rate > 此值 → JQ 表现差 → 强罚
PENALTY_KEEP = 0.1    # 高频但 JQ 好 → 保留 (仅 10% 基础罚)
PENALTY_NEUTRAL = 0.4 # JQ 数据不足 → 中性
PENALTY_HARD = 1.0    # 高频 ∧ JQ 差 → 强罚


class SubstructurePenalizer:
    """表达式子结构频率库 + JQ 联动软降采样器"""

    def __init__(self, enabled: bool = True, window: int = 3):
        self.enabled = enabled
        self.window = window
        self.table: Dict[str, Dict] = {}   # {triplet: {freq, jq_n, jq_fail}}
        self._max_freq = 1
        self._parser = None  # 延迟初始化 (避免循环导入)

    # ── 三元组提取 ──────────────────────────────────────

    def _get_parser(self):
        if self._parser is None:
            from factor_expression_tree import FactorExpressionParser
            self._parser = FactorExpressionParser()
        return self._parser

    def _skeleton_of(self, formula: str) -> Optional[str]:
        """解析公式并返回结构骨架; 解析失败返回 None"""
        try:
            node = self._get_parser().parse(formula)
            return node.to_skeleton()
        except Exception:
            return None

    @staticmethod
    def _func_sequence(skeleton: str) -> List[str]:
        """从 skeleton 字符串提取函数名序列 (保留出现顺序)"""
        return _FUNC_RE.findall(skeleton)

    def extract_triplets(self, formula: str) -> Set[str]:
        """提取公式的三元组子结构集合 (滑动窗口, 字段/数字已抽象)"""
        skel = self._skeleton_of(formula)
        if not skel:
            return set()
        seq = self._func_sequence(skel)
        if len(seq) < self.window:
            return set()
        return {"|".join(seq[i:i + self.window]) for i in range(len(seq) - self.window + 1)}

    # ── 热度表构建 / 持久化 ─────────────────────────────

    def build_table(self, formulas: List[str],
                    jq_records: Optional[List[Tuple[str, bool]]] = None) -> Dict:
        """
        formulas: 库内所有因子公式 (freq 统计源)
        jq_records: [(formula, jq_ok)] — jq_ok = 该公式 JQ 回测是否正收益
        """
        table: Dict[str, Dict] = {}
        for f in formulas:
            for t in self.extract_triplets(f):
                table.setdefault(t, {"freq": 0, "jq_n": 0, "jq_fail": 0})
                table[t]["freq"] += 1
        if jq_records:
            for f, jq_ok in jq_records:
                for t in self.extract_triplets(f):
                    if t in table:
                        table[t]["jq_n"] += 1
                        if not jq_ok:
                            table[t]["jq_fail"] += 1
        self.table = table
        self._max_freq = max((v["freq"] for v in table.values()), default=1)
        return table

    def save(self, path: Path = FREQ_PATH) -> None:
        payload = {
            "window": self.window,
            "max_freq": self._max_freq,
            "triplets": len(self.table),
            "table": self.table,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)

    def load(self, path: Path = FREQ_PATH) -> bool:
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.window = payload.get("window", self.window)
        self.table = payload.get("table", {})
        self._max_freq = payload.get("max_freq", 1)
        return True

    # ── 惩罚计算 ────────────────────────────────────────

    def jq_fail_rate(self, triplet: str) -> Optional[float]:
        e = self.table.get(triplet)
        if not e or e["jq_n"] < JQ_MIN_N:
            return None
        return e["jq_fail"] / e["jq_n"]

    def penalty_for(self, formula: str) -> float:
        """公式级惩罚概率 = max(三元组惩罚)。0.0 = 不降采样"""
        if not self.enabled or not self.table:
            return 0.0
        trips = self.extract_triplets(formula)
        if not trips:
            return 0.0
        worst = 0.0
        for t in trips:
            e = self.table.get(t)
            if not e:
                continue
            heat = min(1.0, e["freq"] / self._max_freq)
            fail = self.jq_fail_rate(t)
            if fail is None:
                p = heat * PENALTY_NEUTRAL       # JQ 数据不足 → 中性基础罚
            elif fail <= JQ_GOOD_FAIL_RATE:
                p = heat * PENALTY_KEEP          # 高频但 JQ 表现好 → 保留
            elif fail > JQ_BAD_FAIL_RATE:
                p = heat * PENALTY_HARD          # 高频 ∧ JQ 失败率高 → 强降采样
            else:
                p = heat * PENALTY_NEUTRAL       # 中间地带 → 中性
            worst = max(worst, p)
        return worst

    def should_reject(self, formula: str, rng: random.Random) -> bool:
        """软拒绝: 按 penalty 概率拒绝 (调用方负责重试)"""
        if not self.enabled or not self.table:
            return False
        p = self.penalty_for(formula)
        if p <= 0:
            return False
        return rng.random() < p

    # ── 2026-08-31: JQ 失败先验 (影子特征, 非否决) ────────

    def fail_prior_for(self, formula: str) -> Dict:
        """候选级 JQ 失败先验: 基于三元组 fail% 聚合。

        设计 (P-20260831 分离力分析落地):
        - 本地周频 pandas_icir 对 JQ 成败无分离力 (p=0.139);
        - 子结构 fail% 是血统级 JQ 信号 (P-007 546 三元组带 jq_n/jq_fail);
        - 本函数只产出影子特征, 不参与任何否决 (调用方决定用途)。

        Returns:
            {n_triplets, n_known, n_high_fail, fail_rate_pooled, coverage, prior_score}
            prior_score = coverage*pooled + (1-coverage)*0.5  (未知三元组按中性 0.5)
        """
        empty = {"n_triplets": 0, "n_known": 0, "n_high_fail": 0,
                 "fail_rate_pooled": None, "coverage": 0.0, "prior_score": 0.5}
        if not self.table:
            return empty
        trips = self.extract_triplets(formula)
        if not trips:
            return empty
        known = []
        n_high = 0
        for t in trips:
            fr = self.jq_fail_rate(t)   # None = jq_n < JQ_MIN_N
            if fr is not None:
                known.append((fr, self.table[t]["jq_n"]))
                if fr > JQ_BAD_FAIL_RATE:
                    n_high += 1
        n_triplets = len(trips)
        n_known = len(known)
        if n_known:
            tot_fail = sum(fr * n for fr, n in known)
            tot_n = sum(n for _, n in known)
            pooled = tot_fail / max(tot_n, 1)
        else:
            pooled = None
        coverage = n_known / n_triplets
        prior = (coverage * pooled + (1 - coverage) * 0.5) if pooled is not None else 0.5
        return {"n_triplets": n_triplets, "n_known": n_known, "n_high_fail": n_high,
                "fail_rate_pooled": round(pooled, 4) if pooled is not None else None,
                "coverage": round(coverage, 3), "prior_score": round(prior, 4)}

    # ── 审计输出 ────────────────────────────────────────

    def top_n(self, n: int = 10) -> List[Tuple[str, Dict]]:
        ranked = sorted(self.table.items(), key=lambda kv: -kv[1]["freq"])
        return ranked[:n]

    def audit_report(self, n: int = 10) -> str:
        if not self.table:
            return "  [Substructure] 热度表为空 (未启用或未构建)"
        lines = [f"  [Substructure] 子结构热度表 Top{n} (共 {len(self.table)} 个三元组):"]
        for t, e in self.top_n(n):
            fail = self.jq_fail_rate(t)
            fail_s = f"{fail:.0%}" if fail is not None else "n/a"
            lines.append(f"    {t:<36} freq={e['freq']:>4}  jq_n={e['jq_n']:>3}  fail={fail_s}")
        return "\n".join(lines)


# ── 便捷入口 ────────────────────────────────────────────

def build_penalizer_from_library(formulas: List[str],
                                 jq_records: Optional[List[Tuple[str, bool]]] = None,
                                 enabled: bool = True) -> SubstructurePenalizer:
    """从因子库公式 + JQ 记录构建 penalizer (含表落盘)"""
    pz = SubstructurePenalizer(enabled=enabled)
    pz.build_table(formulas=formulas, jq_records=jq_records)
    pz.save()
    return pz


def load_or_build_penalizer(formulas: Optional[List[str]] = None,
                            jq_records: Optional[List[Tuple[str, bool]]] = None,
                            enabled: bool = True) -> SubstructurePenalizer:
    """优先加载持久化热度表; 不存在或未提供公式源时返回空 penalizer"""
    pz = SubstructurePenalizer(enabled=enabled)
    if pz.load():
        return pz
    if formulas:
        return build_penalizer_from_library(formulas, jq_records, enabled)
    return pz


def compute_fail_prior_from_library(formula: str,
                                    penalizer: Optional[SubstructurePenalizer] = None) -> Dict:
    """便捷入口: 计算单公式的 JQ 失败先验 (影子特征, 非否决)。

    penalizer=None 时从 data/substructure_freq.json 加载 (表缺失 → 全中性 0.5)。
    调用方 (ralph_loop S1 影子记录) 需自行 try/except 包裹, 保证非阻塞。
    """
    if penalizer is None:
        penalizer = load_or_build_penalizer()
    return penalizer.fail_prior_for(formula)
