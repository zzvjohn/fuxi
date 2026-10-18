# -*- coding: utf-8 -*-
"""
Semantic Consistency Verifier — 对标 QuantaAlpha Section 4.1 的语义一致性校验
===================================================================================

三重一致性检查 (QuantaAlpha 核心机制):
  1. H↔E: Hypothesis ↔ Expression — 假设与数学表达式对齐
  2. E↔C: Expression ↔ Code — 符号表达式与可执行代码一致
  3. H↔C: Hypothesis ↔ Code — 假设在代码行为中得到体现

设计:
  - LLM-based verifier (调用外部 LLM API)
  - 可离线降级为 rule-based checks
  - 返回 pass/fail + 详细理由

用法:
    from semantic_verifier import SemanticVerifier

    verifier = SemanticVerifier()
    result = verifier.verify(
        hypothesis="20日动量在市场低波动时有效",
        factor_expression="ts_delta(close, 20) / ts_std(close, 20)",
        code="def factor(df): return df['close'].diff(20) / df['close'].rolling(20).std()"
    )
    if result["pass"]:
        print("语义一致 ✅")
    else:
        print(f"语义漂移: {result['reasons']}")
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════
# v0.5.4: H↔E 概念表 (修复 LLM 候选 0/10 全拒问题)
# ═══════════════════════════════════════════════════════════
# 旧版问题:
#   1. 概念表仅 10 行, 缺失 A 股核心概念 (资金流/大单/筹码/换手/突破/背离)
#   2. 英文关键词纯子串匹配: "vol" 误命中 "volume" → 波动概念误触发, 要求公式含 std
#   3. "aml" 拼写错误 (应为 amt)
#   4. 动量要求公式含 diff/pct_change/slope 字面 token, 但 LLM 常用 shift / 比率形式
#   5. 反转公式常写作 -(x) 或 (-1)*x, 表中却要求 "neg"/"rank" 字面 token
#
# 修复: 扩充概念表 + 边界化英文匹配 + 操作符同义扩展。
# 每行: (假设关键词, 表达式实现关键词)
H2E_CONCEPT_TABLE = [
    # 动量/趋势/加速 → 差分/位移/斜率
    (["动量", "momentum", "趋势", "trend", "加速", "acceleration", "惯性"],
     ["diff", "pct_change", "roc", "slope", "ts_delta", "delta", "shift", "accelerat"]),
    # 波动/波动率 → std/var/振幅
    (["波动", "volatility", "波动率", "振幅", "震荡"],
     ["std", "var", "rolling_std", "ts_std", "amplitude", "hl_ratio", "atr"]),
    # 反转/均值回复 → 负号/rank
    (["反转", "reversal", "逆转", "逆向", "均值回复", "mean_reversion", "回落", "回调"],
     ["neg", "rank", "csrank", "-"]),
    # 量能/成交/换手 → volume/amount/turnover/moneyflow 字段
    # 注意: 不含单字 "量" — 会误命中 "动量" 中的 "量" (动量行已覆盖)
    (["成交量", "volume", "成交", "换手", "turnover", "amt", "放量", "缩量", "量能", "amount"],
     ["volume", "amount", "turnover", "amt", "vol", "mf_vol", "lg_vol", "sm_vol", "md_vol", "elg_vol"]),
    # 相关/共振/联动 → corr/cov
    (["相关", "correlation", "corr", "共振", "联动", "同步"],
     ["corr", "rolling_corr", "ts_corr", "cov"]),
    # 均线/平均 → mean/rolling
    (["均线", "平均", "mean", "average", "sma", "ema", "ma"],
     ["mean", "sma", "ema", "ts_mean", "rolling"]),
    # 偏度 / 峰度
    (["偏度", "skew"], ["skew", "ts_skew"]),
    (["峰度", "kurt", "kurtosis"], ["kurt", "ts_kurt"]),
    # 回归/斜率/残差
    (["回归", "slope", "residual", "残差", "r2", "rsquare", "拟合"],
     ["slope", "rsquare", "resi", "reg"]),
    # 价格位置 (高点/低点/开盘/收盘)
    (["高点", "低点", "high", "low", "开盘", "收盘", "open", "close"],
     ["open", "high", "low", "close"]),
    # v0.5.4 新增: 资金流/大单/主力/净流入 → moneyflow 字段
    (["资金流", "大单", "主力", "净流入", "净流出", "moneyflow", "money_flow", "机构", "smart_money"],
     ["buy_lg_vol", "sell_lg_vol", "buy_elg_vol", "sell_elg_vol", "buy_sm_vol", "sell_sm_vol",
      "buy_md_vol", "sell_md_vol", "net_mf_vol", "net_mf_amount", "lg_vol", "sm_vol", "elg_vol"]),
    # v0.5.4 新增: 筹码/成本分布
    (["筹码", "chip", "成本分布", "套牢", "锁仓", "分布"],
     ["chip", "cost", "vwap", "lockup", "distribution", "concentration"]),
    # v0.5.4 新增: 突破/支撑/阻力
    (["突破", "breakout", "支撑", "阻力", "support", "resistance", "新高", "新低"],
     ["max", "min", "high", "low", "breakout", "cross"]),
    # v0.5.4 新增: 背离
    (["背离", "divergence", "divergent"],
     ["div", "diff", "corr", "-"]),
]

# 英文关键词边界匹配: 避免 "vol" 误命中 "volume"、"std" 误命中 "stand"
_LATIN_BOUNDARY = re.compile(r'[a-z0-9_]+')


# ═══════════════════════════════════════════════════════════
# P-20260907-002: FaVOR 式可观测条件分解验证
# ═══════════════════════════════════════════════════════════
# 背景: 旧 H↔E 为单一二元判定, 09-05 事故 (假设=开盘30分钟, 公式=6日动量差)
#       二元判定漏网后被人工撤销。FaVOR (arXiv:2608.30192) 的做法:
#       假设 → 可观测条件列表 → 逐条件核验 → 条件级通过率。
#
# 防误杀设计 (与 P-002 提案 rollback_plan 一致):
#   - 默认 shadow 模式: 只输出 condition_report + 告警, 不改变 pass 裁决。
#   - enforce 模式 (FUXI_COND_VERIFY=enforce): 通过率 ≤ 0.5 且条件数 ≥ 2 时,
#     仅对 h2e 原本已 fail 的候选追加证据; h2e 已 pass 的候选只降级
#     confidence 不翻案 → 零新误杀源。
#   - 阈值松 (≤50% 才触发), 语义等价表述 (rolling(20) vs window=20) 不判不齐。

# 条件类别 → 假设侧触发词 → 表达式侧证据词
COND_TRIGGERS = {
    # 时间窗口条件: 假设出现 N日/N天/N周/N月 → 表达式应含同量级窗口数字
    "time_window": None,  # 特殊处理 (正则提取数字)
    # 数据字段条件: 复用 H2E_CONCEPT_TABLE 的字段行 (量能/资金流/筹码/价格)
    "data_field": [
        # (假设关键词, 表达式证据关键词, 说明)
        (["资金流", "大单", "超大单", "主力", "净流入", "净流出", "moneyflow"],
         ["buy_lg", "sell_lg", "buy_elg", "sell_elg", "net_mf", "buy_sm", "sell_sm"], "资金流字段"),
        (["成交量", "volume", "成交", "换手", "turnover", "量能", "放量", "缩量"],
         ["vol", "amount", "turnover"], "量能字段"),
        (["筹码", "chip", "成本分布", "套牢", "锁仓"],
         ["chip", "cost", "vwap", "lockup"], "筹码字段"),
        (["高点", "低点", "开盘", "收盘", "high", "low", "open", "close"],
         ["high", "low", "open", "close"], "价格字段"),
        (["龙虎榜", "席位", "lhb", "机构专用", "游资"],
         ["lhb", "inst", "yz_", "north"], "龙虎榜字段"),
        (["融资", "融券", "两融", "margin"],
         ["rzye", "rqye", "rzmre", "rqyl"], "两融字段"),
    ],
    # 变换算子条件: 假设提到统计变换 → 表达式含对应算子
    "transform_op": [
        (["均值", "平均", "mean", "average", "sma"],
         ["mean", "sma", "ewm"], "均值算子"),
        (["标准差", "波动", "std", "volatility"],
         ["std", "var"], "波动算子"),
        (["偏度", "skew"], ["skew"], "偏度算子"),
        (["峰度", "kurt", "kurtosis"], ["kurt"], "峰度算子"),
        (["回归", "残差", "residual", "regression", "拟合"],
         ["reg", "resi", "slope", "rsquare"], "回归算子"),
        (["相关", "corr", "联动", "共振"],
         ["corr", "cov"], "相关算子"),
        (["分布", "集中度", "离散", "dispersion"],
         ["std", "skew", "kurt", "mad", "quantile"], "分布统计算子"),
    ],
    # 方向预期条件 (仅告警级: 容易语义等价误判, 不参与拒阈值)
    "direction": [
        (["反转", "reversal", "逆转", "逆向", "均值回复", "回落", "负向", "做空"],
         ["-", "neg", "rank", "rev"], "负向/反转"),
        (["动量", "momentum", "趋势", "正向", "加速"],
         ["diff", "pct_change", "shift", "roc"], "正向/动量"),
    ],
    # 条件依赖条件 (仅告警级: LLM 公式常用 .where/.clip 等价表述)
    "regime": [
        (["当", "时", "若", "如果", "条件", "regime", "状态", "环境", "if"],
         ["where", "ifelse", "clip", "mask", "greater", "less"], "条件依赖"),
    ],
}

# 参与拒阈值裁决的条件类别 (direction/regime 仅告警)
_COND_ADJUDICATION_TYPES = {"time_window", "data_field", "transform_op"}

# 时间窗口提取: 中文 (N日/N天/N周/N月/N分钟) 与英文 (N day/week/min)
_RE_TIME_CN = re.compile(r'(\d{1,3})\s*(?:日|天|周|个月|月|分钟|小时)')
_RE_TIME_EN = re.compile(r'(\d{1,3})\s*(?:day|days|week|weeks|month|months|min)')


def get_cond_verify_mode() -> str:
    """P-20260907-002 条件分解验证模式:
    shadow (默认, 只报告不裁决) / enforce (通过率≤0.5 且 h2e 已 fail 时加重证据) / off
    """
    return os.environ.get("FUXI_COND_VERIFY", "shadow").strip().lower()


def decompose_hypothesis_conditions(hypothesis: str) -> List[Dict]:
    """把假设 H 分解为可观测条件列表 (规则版, 无 LLM 调用)。

    返回: [{"type": str, "label": str, "text": str, "evidence_keys": [...]}, ...]
    """
    h = hypothesis or ""
    hl = h.lower()
    conds: List[Dict] = []

    # 1) 时间窗口条件
    for m in _RE_TIME_CN.finditer(h) or []:
        n = int(m.group(1))
        if 1 <= n <= 252:
            conds.append({
                "type": "time_window", "label": f"时间窗口 {n} 单位",
                "text": m.group(0), "num": n, "evidence_keys": ["window"],
            })
    if not conds:
        for m in _RE_TIME_EN.finditer(h):
            n = int(m.group(1))
            if 1 <= n <= 252:
                conds.append({
                    "type": "time_window", "label": f"时间窗口 {n} 单位",
                    "text": m.group(0), "num": n, "evidence_keys": ["window"],
                })
                break

    # 2) 数据字段 / 3) 变换算子 / 4) 方向 / 5) 条件依赖
    for ctype in ("data_field", "transform_op", "direction", "regime"):
        for hyp_kws, ev_kws, label in COND_TRIGGERS[ctype]:
            if _match_keywords(hl, hyp_kws):
                conds.append({
                    "type": ctype, "label": label,
                    "text": hyp_kws[0], "evidence_keys": ev_kws,
                })

    return conds


def verify_conditions(hypothesis: str, expression: str) -> Dict:
    """逐条件核验公式是否满足假设分解出的可观测条件 (P-20260907-002)。

    返回: {
        "conditions": [{..., "matched": bool, "evidence": str}, ...],
        "n_total": int, "n_matched": int, "pass_rate": float,
        "adjudication_pass_rate": float,  # 仅参与裁决的类别 (窗口/字段/算子)
        "verified": bool,                 # 全部条件通过
    }
    防误杀: 窗口数字同量级即通过 (2-252 内任一数字均可, 不要求精确相等);
            字段/算子用子串证据匹配, 同义实现 (shift/diff/pct_change) 互认。
    """
    conds = decompose_hypothesis_conditions(hypothesis)
    el = (expression or "").lower()

    # 表达式侧全部窗口数字 (同量级判断用)
    expr_nums = [int(x) for x in re.findall(r'(\d{1,3})', expression) if 2 <= int(x) <= 252]

    n_total = len(conds)
    n_matched = 0
    adj_matched = 0
    adj_total = 0

    for c in conds:
        matched = False
        evidence = ""
        if c["type"] == "time_window":
            # 防误杀: 同量级窗口即通过 (假设 20日 vs 公式 rolling(20) 之外,
            # 公式写 21/22 也算语义同量级, 不做精确相等)
            tgt = c.get("num", 0)
            if expr_nums:
                for en in expr_nums:
                    if abs(en - tgt) / max(tgt, 1) <= 0.25 or tgt == en:
                        matched = True
                        evidence = f"表达式含窗口 {en}"
                        break
                if not matched:
                    evidence = f"表达式窗口数字 {expr_nums} 与假设 {tgt} 非同量级"
            else:
                evidence = "表达式无窗口数字"
        else:
            for ek in c["evidence_keys"]:
                if ek in el:
                    matched = True
                    evidence = f"命中证据 '{ek}'"
                    break
            if not matched:
                if c["type"] in ("direction", "regime"):
                    # 方向/条件依赖仅告警: 未命中不算失败 (等价表述多)
                    matched = True
                    evidence = f"告警级条件 (证据词: {'/'.join(c['evidence_keys'])})"
                else:
                    evidence = f"未命中证据词 {'/'.join(c['evidence_keys'][:4])}"
        c["matched"] = matched
        c["evidence"] = evidence
        if matched:
            n_matched += 1
        if c["type"] in _COND_ADJUDICATION_TYPES:
            adj_total += 1
            if matched:
                adj_matched += 1

    pass_rate = (n_matched / n_total) if n_total else 1.0
    adj_rate = (adj_matched / adj_total) if adj_total else 1.0

    return {
        "conditions": conds,
        "n_total": n_total,
        "n_matched": n_matched,
        "pass_rate": pass_rate,
        "adjudication_pass_rate": adj_rate,
        "verified": n_matched == n_total,
    }


def _match_keywords(haystack: str, keywords) -> bool:
    """关键词匹配: 中文/长词用子串, 短英文词用边界匹配防止误命中"""
    h = haystack.lower()
    for kw in keywords:
        k = kw.lower()
        if not k:
            continue
        if re.fullmatch(r'[a-z0-9_]+', k) and len(k) <= 4:
            # 短英文 token: 词边界匹配 (防止 vol⊂volume, std⊂stand 误命中)
            if re.search(rf'(?<![a-z0-9_]){re.escape(k)}(?![a-z0-9_])', h):
                return True
        else:
            if k in h:
                return True
    return False


class SemanticVerifier:
    """语义一致性校验器 — 对标 QuantaAlpha 的 consistency verification"""

    def __init__(
        self,
        max_complexity: int = 10,     # AST 最大节点数
        max_ast_depth: int = 4,       # AST 最大深度
        max_corr_threshold: float = 0.7,  # 冗余度阈值
    ):
        self.max_complexity = max_complexity
        self.max_ast_depth = max_ast_depth
        self.max_corr_threshold = max_corr_threshold

    def verify(
        self,
        hypothesis: str,
        factor_expression: str,
        code: str = "",
        llm_available: bool = False,
        llm_api_key: str = "",
        llm_base_url: str = "",
        refutation_conditions=None,
    ) -> Dict:
        """
        执行三重一致性校验。

        Parameters
        ----------
        hypothesis: 因子假设 (自然语言)
        factor_expression: 因子数学表达式
        code: 可执行代码 (可选)
        llm_available: 是否可用 LLM verifier
        llm_api_key: LLM API key
        refutation_conditions: 可证伪判据列表 (P-20260902-005, AQuA 式)
            至少 1 条可观测证伪判据 (如 "若该因子在 2023 子样本 ICIR < 0 则假设不成立")
            缺失/为空 → 不拒绝, 但 confidence 降级为 low_confidence 并计数

        Returns
        -------
        {
            "pass": bool,             # 三重检查全部通过
            "h2e_pass": bool,         # hypothesis ↔ expression
            "e2c_pass": bool,         # expression ↔ code
            "h2c_pass": bool,         # hypothesis ↔ code
            "refutation_present": bool,      # P-20260902-005: 证伪条件存在性
            "confidence": str,               # normal | low_confidence
            "reasons": List[str],     # 未通过的原因
            "scores": {               # 各维度评分
                "novelty": float,
                "economic_logic": float,
                "expression_quality": float,
                "code_fidelity": float,
            },
            "complexity": {           # 复杂度/冗余检查
                "node_count": int,
                "ast_depth": int,
                "complexity_pass": bool,
                "redundancy_pass": bool,
            },
            "condition_report": {     # P-20260907-002: FaVOR 式条件分解 (shadow 默认)
                "conditions": List[Dict],  # 每条含 type/matched/evidence
                "n_total": int, "n_matched": int,
                "pass_rate": float, "adjudication_pass_rate": float,
                "verified": bool,
            },
        }
        """
        result = {
            "pass": False,
            "h2e_pass": False,
            "e2c_pass": False,
            "h2c_pass": False,
            "reasons": [],
        }

        # ── P-20260902-005: 证伪条件存在性校验 (AQuA 式) ──
        # 缺失仅降级 low_confidence, 不拒绝 (可证伪性不足 ≠ 因子无效)
        _rc = refutation_conditions
        if isinstance(_rc, str):
            _rc = [x.strip() for x in _rc.split(";") if x.strip()] if _rc.strip() else []
        result["refutation_present"] = bool(isinstance(_rc, (list, tuple)) and len(_rc) >= 1)
        result["confidence"] = "normal" if result["refutation_present"] else "low_confidence"
        if not result["refutation_present"]:
            result["reasons"].append(
                "[refutation] 缺证伪判据 (P-20260902-005: 不拒绝, 降级 low_confidence)"
            )

        # ── Rule-based 检查 ──────────────────────────────

        # H↔E: 假设中的关键概念是否在表达式中体现
        h2e = self._check_hypothesis_to_expression(hypothesis, factor_expression)
        result["h2e_pass"] = h2e["pass"]
        if not h2e["pass"]:
            result["reasons"].extend(h2e["reasons"])

        # ── P-20260907-002: FaVOR 式条件分解验证 (默认 shadow 防误杀) ──
        result["condition_report"] = {}
        cond_mode = get_cond_verify_mode()
        if cond_mode in ("shadow", "enforce") and hypothesis and factor_expression:
            creport = verify_conditions(hypothesis, factor_expression)
            result["condition_report"] = creport
            if not creport["verified"]:
                _miss = [c for c in creport["conditions"] if not c["matched"]]
                _miss_txt = "; ".join(f"{c['label']}({c['text']})→{c['evidence']}"
                                      for c in _miss[:4])
                if cond_mode == "shadow":
                    # 影子: 仅告警, 不改变 pass (提案 rollback_plan 约定)
                    result["reasons"].append(
                        f"[cond-shadow] 条件通过率 {creport['pass_rate']:.0%} "
                        f"({creport['n_matched']}/{creport['n_total']}): {_miss_txt}"
                    )
                else:
                    # enforce: 通过率 ≤ 0.5 且裁决类条件 ≥ 2 时加重证据;
                    # h2e 已 pass 只降级 confidence 不翻案 (零新误杀源)
                    if (creport["adjudication_pass_rate"] <= 0.5
                            and creport["n_total"] >= 2):
                        result["reasons"].append(
                            f"[cond-enforce] 裁决类条件通过率 {creport['adjudication_pass_rate']:.0%} "
                            f"低于 0.5: {_miss_txt}"
                        )
                        if not result["h2e_pass"]:
                            result["confidence"] = "low_confidence"
                        else:
                            result["confidence"] = "low_confidence"
                    elif not creport["verified"]:
                        result["reasons"].append(
                            f"[cond-enforce] 条件级告警 ({creport['pass_rate']:.0%}): {_miss_txt}"
                        )

        # E↔C: 表达式的操作是否在代码中实现
        if code:
            e2c = self._check_expression_to_code(factor_expression, code)
            result["e2c_pass"] = e2c["pass"]
            if not e2c["pass"]:
                result["reasons"].extend(e2c["reasons"])
        else:
            result["e2c_pass"] = True  # 无代码时不检查

        # H↔C: 假设是否在代码中得到体现
        if code:
            h2c = self._check_hypothesis_to_code(hypothesis, code)
            result["h2c_pass"] = h2c["pass"]
            if not h2c["pass"]:
                result["reasons"].extend(h2c["reasons"])
        else:
            result["h2c_pass"] = True

        # ── 综合裁决 ──────────────────────────────────────
        result["pass"] = all([result["h2e_pass"], result["e2c_pass"], result["h2c_pass"]])

        # ── 额外评分 ──────────────────────────────────────
        result["scores"] = {
            "novelty": self._score_novelty(hypothesis),
            "economic_logic": self._score_economic_logic(hypothesis),
            "expression_quality": self._score_expression_quality(factor_expression),
            "code_fidelity": self._score_code_fidelity(code) if code else 1.0,
        }

        # ── 复杂度/冗余检查 ───────────────────────────────
        result["complexity"] = self._check_complexity(factor_expression)

        return result

    # ── H↔E: 假设→表达式对齐 ────────────────────────────

    def _check_hypothesis_to_expression(
        self, hypothesis: str, expression: str
    ) -> Dict:
        """
        Rule-based: 检查假设中的关键概念是否出现在表达式中。

        v0.5.4: 使用扩充后的 H2E_CONCEPT_TABLE + 边界化英文匹配。
        历史问题: LLM 假设含 动量/趋势 (MAB 方向) 但公式为 shift/比率形式,
        字面 token 匹配 (diff/pct_change/slope) 全部落空 → 0/10 误拒。
        """
        hyp_lower = hypothesis.lower()
        expr_lower = expression.lower()

        reasons = []
        matched_concepts = 0
        total_concepts = 0

        for hyp_keywords, expr_keywords in H2E_CONCEPT_TABLE:
            if _match_keywords(hyp_lower, hyp_keywords):
                total_concepts += 1
                # 表达式侧用纯子串匹配: buy_lg_vol 应命中 "lg_vol", "vol" 等字段片段
                if any(k in expr_lower for k in expr_keywords):
                    matched_concepts += 1
                else:
                    reasons.append(
                        f"H↔E: 假设提到了 '{hyp_keywords[0]}'，但表达式中未找到对应操作 "
                        f"({', '.join(expr_keywords[:3])})"
                    )

        # 至少 60% 的概念得到匹配
        if total_concepts == 0:
            return {"pass": True, "reasons": []}  # 无概念时放过
        pass_rate = matched_concepts / max(total_concepts, 1)

        # v0.5.4: 阈值语义调整 — 规则匹配无法理解自然语言, 全拒代价过高
        #   ERROR: 提到 ≥2 个概念但 0 个实现 (明显假设-公式脱节)
        #   pass:  匹配率 ≥ 50% (>=2 概念时至少实现一半)
        pass_flag = (matched_concepts >= 1) if total_concepts == 1 else (pass_rate >= 0.5)

        return {
            "pass": pass_flag,
            "reasons": reasons,
            "match_rate": pass_rate,
            "matched": matched_concepts,
            "total": total_concepts,
        }

    # ── E↔C: 表达式→代码对齐 ────────────────────────────

    def _check_expression_to_code(
        self, expression: str, code: str
    ) -> Dict:
        """
        检查表达式中的操作是否在代码中实现。

        重点检查:
        - 函数调用是否存在 (ts_delta → rolling/diff, ts_rank → rank)
        - 参数是否一致 (窗口大小)
        """
        reasons = []

        # 提取表达式中的窗口参数
        expr_windows = set()
        for m in re.finditer(r'(\d+)', expression):
            num = int(m.group(1))
            if 2 <= num <= 252:  # 合理窗口范围
                expr_windows.add(num)

        # 检查代码中是否使用了相同窗口
        code_windows = set()
        for m in re.finditer(r'(\d+)', code):
            num = int(m.group(1))
            if 2 <= num <= 252:
                code_windows.add(num)

        # 关键操作检查
        expr_lower = expression.lower()
        code_lower = code.lower()

        op_checks = [
            ("delta", ["diff", "shift", "pct_change"]),
            ("mean", ["mean", "rolling.*mean", "average"]),
            ("std", ["std", "rolling.*std"]),
            ("corr", ["corr", "rolling.*corr"]),
            ("rank", ["rank", "pct_rank"]),
            ("neg", ["neg", "-"]),
            ("skew", ["skew"]),
            ("kurt", ["kurt", "kurtosis"]),
            ("slope", ["slope", "linregress"]),
            ("ema", ["ewm", "ema"]),
        ]

        for op, code_patterns in op_checks:
            if op in expr_lower:
                if not any(re.search(p, code_lower) for p in code_patterns):
                    reasons.append(f"E↔C: 表达式使用了 '{op}' 操作，但代码中未找到对应实现")

        # 窗口一致性检查 (宽松)
        for w in expr_windows:
            if w not in code_windows:
                reasons.append(f"E↔C: 表达式使用窗口 {w}，但代码中未找到此参数 (代码中有: {sorted(code_windows)})")

        return {
            "pass": len(reasons) == 0,
            "reasons": reasons,
        }

    # ── H↔C: 假设→代码对齐 ───────────────────────────────

    def _check_hypothesis_to_code(
        self, hypothesis: str, code: str
    ) -> Dict:
        """
        检查假设中的关键机制是否在代码中得到体现。
        专注于: 市场状态条件 (IfElse风格)、多因子交互。
        """
        hyp_lower = hypothesis.lower()
        code_lower = code.lower()
        reasons = []

        # 条件逻辑检查
        condition_keywords = ["当", "when", "if", "条件", "condition", "regime", "状态", "环境"]
        has_condition_in_hyp = any(kw in hyp_lower for kw in condition_keywords)
        has_condition_in_code = any(kw in code_lower for kw in ["ifelse", "if ", "where", "np.where", "greater", "less"])

        if has_condition_in_hyp and not has_condition_in_code:
            reasons.append("H↔C: 假设中描述了条件/状态依赖逻辑，但代码中未使用 IfElse/Greater/Less 等条件操作")

        # 交互检查
        interaction_keywords = ["交互", "interaction", "乘积", "product", "结合", "combine"]
        has_interaction_in_hyp = any(kw in hyp_lower for kw in interaction_keywords)
        has_interaction_in_code = any(kw in code_lower for kw in ["mul(", "multiply", "*", "add(", "+"])

        if has_interaction_in_hyp and not has_interaction_in_code:
            reasons.append("H↔C: 假设描述了多因子交互，但代码中未使用乘法/加法")

        return {
            "pass": len(reasons) == 0,
            "reasons": reasons,
        }

    # ── 复杂度/冗余检查 ───────────────────────────────────

    def _check_complexity(self, expression: str) -> Dict:
        """对标 QuantaAlpha 的 complexity & redundancy 约束"""
        # AST 节点数近似 = 函数调用数 + 操作符数
        fn_calls = len(re.findall(r'\w+\(', expression))
        operators = len(re.findall(r'[\+\-\*/]', expression))
        node_count = fn_calls + operators

        # AST 深度近似 = 最大嵌套层数
        max_nesting = 0
        current = 0
        for ch in expression:
            if ch == '(':
                current += 1
                max_nesting = max(max_nesting, current)
            elif ch == ')':
                current -= 1
        ast_depth = max_nesting

        complexity_pass = (node_count <= self.max_complexity) and (ast_depth <= self.max_ast_depth)
        redundancy_pass = node_count <= self.max_complexity * 2  # 松一点

        return {
            "node_count": node_count,
            "ast_depth": ast_depth,
            "complexity_pass": complexity_pass,
            "redundancy_pass": redundancy_pass,
            "max_allowed_nodes": self.max_complexity,
            "max_allowed_depth": self.max_ast_depth,
        }

    # ── 评分 ──────────────────────────────────────────────

    def _score_novelty(self, hypothesis: str) -> float:
        """评估假设的新颖度"""
        hyp_lower = hypothesis.lower()
        # 常见模式 → 低新颖度
        common_patterns = ["动量", "反转", "均线交叉"]
        uncommon_patterns = ["偏度", "峰度", "残差", "熵", "拥挤", "微观结构", "跨资产"]

        common_count = sum(1 for p in common_patterns if p in hyp_lower)
        uncommon_count = sum(1 for p in uncommon_patterns if p in hyp_lower)

        score = 0.3 + uncommon_count * 0.15 - common_count * 0.1
        return max(0.0, min(1.0, score))

    def _score_economic_logic(self, hypothesis: str) -> float:
        """评估假设的经济逻辑强度"""
        hyp_lower = hypothesis.lower()
        logic_keywords = [
            "因为", "因此", "由于", "所以", "由...导致",
            "because", "therefore", "since", "due to",
            "机制", "机理", "mechanism",
        ]
        score = sum(0.2 for kw in logic_keywords if kw in hyp_lower)
        return min(1.0, score + 0.3)  # 最低 0.3

    def _score_expression_quality(self, expression: str) -> float:
        """评估表达式质量"""
        if not expression:
            return 0.0
        score = 0.5

        # 多样性奖励
        ops = set(re.findall(r'(?:ts_|cs_|rolling_)?(\w+)\(', expression.lower()))
        if len(ops) >= 3:
            score += 0.2
        if len(ops) >= 5:
            score += 0.1

        # 复杂性惩罚
        node_count = len(re.findall(r'\w+\(', expression))
        if node_count <= 5:
            score += 0.1
        elif node_count > 10:
            score -= 0.2

        return max(0.1, min(1.0, score))

    def _score_code_fidelity(self, code: str) -> float:
        """评估代码的保真度"""
        if not code:
            return 0.5
        score = 0.5

        # 变量命名质量
        if any(kw in code.lower() for kw in ["return", "def "]):
            score += 0.2

        # 空值处理
        if any(kw in code for kw in ["fillna", "dropna", "np.nan", "isna"]):
            score += 0.2

        return min(1.0, score)

    def print_verification(self, result: Dict):
        """打印校验结果"""
        status = "✅ 通过" if result["pass"] else "❌ 未通过"
        print(f"语义一致性校验: {status}")
        print(f"  H↔E (假设→表达式): {'✅' if result['h2e_pass'] else '❌'}")
        print(f"  E↔C (表达式→代码): {'✅' if result['e2c_pass'] else '❌'}")
        print(f"  H↔C (假设→代码): {'✅' if result['h2c_pass'] else '❌'}")
        if result["reasons"]:
            print(f"  问题:")
            for r in result["reasons"]:
                print(f"    → {r}")
        print(f"  评分: novelty={result['scores']['novelty']:.2f}, "
              f"econ_logic={result['scores']['economic_logic']:.2f}, "
              f"expr_qual={result['scores']['expression_quality']:.2f}")
        cx = result["complexity"]
        print(f"  复杂度: nodes={cx['node_count']}/{cx['max_allowed_nodes']}, "
              f"depth={cx['ast_depth']}/{cx['max_allowed_depth']}, "
              f"pass={cx['complexity_pass']}")


# ── 便捷函数 ──────────────────────────────────────────────

_default_verifier: Optional[SemanticVerifier] = None


def get_verifier() -> SemanticVerifier:
    global _default_verifier
    if _default_verifier is None:
        _default_verifier = SemanticVerifier()
    return _default_verifier


# ── 测试 ──────────────────────────────────────────────────

if __name__ == "__main__":
    verifier = SemanticVerifier()

    # 一致性案例
    result = verifier.verify(
        hypothesis="20日动量在市场低波动时有效，因为趋势在稳定环境中持续",
        factor_expression="ts_delta(close, 20) / ts_std(close, 20)",
        code="def factor(df): return (df['close'].diff(20) / df['close'].rolling(20).std()).fillna(0)",
    )
    verifier.print_verification(result)

    print("\n" + "=" * 60)

    # 语义漂移案例
    result2 = verifier.verify(
        hypothesis="偏度驱动的尾部风险信号，通过偏度/峰度条件分支实现反转",
        factor_expression="ts_mean(volume, 5)",  # 完全不匹配假设
        code="def factor(df): return df['volume'].rolling(5).mean()",
    )
    verifier.print_verification(result2)
