# -*- coding: utf-8 -*-
"""
Ralph Loop — Self-Evolving Discovery Engine v0.3 (对标 FactorMiner + QuantaAlpha + CICC Loop Eng)
====================================================================================================

Ralph Loop = Retrieve → Generate → Evaluate → Distill

v0.3 增强 (2026-08-07, 基于中金 CICC Loop Engineering 报告):
  ✅ 集成 FSA (子结构指纹规避) — 结构层面去重，防止同质化
  ✅ 集成 Sub-agent 审查分离 — 独立 FactorReviewAgent 审查
  ✅ 集成检查点系统 — 断点续跑 + 哈希去重
  ✅ 集成 GP 表达式树育种 — 五维演化策略
  ✅ 集成 S5 联合正向过滤 + Calmar — 对标中金 11 项过滤

v0.2 原有:
  ✅ 集成: experience_memory.py (F/E/R) + multi_stage_validator.py + library_orthogonality.py
  ✅ 增强: meta_controller.py 的 auto_cycle (更结构化)
  ✅ 增强: run_v4_pipeline.py (可选的 Ralph Loop 模式)

用法:
    from ralph_loop import RalphLoop

    ralph = RalphLoop()
    
    # 标准模式
    result = ralph.run(candidates=candidate_factors, generator="manual")
    
    # GP 育种模式 (v0.3 新增)
    result = ralph.run(generator="gp_breed", paradigm="动量反转", max_candidates=10)
    
    # 从检查点恢复 (v0.3 新增)
    success, state = ralph.resume_from_checkpoint()
    
    # 跨多轮自动化
    for i in range(100):
        ralph.run_single_round(generator="gp_breed")
"""

import sys
import json
import re
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple, Any


# ═══════════════════════════════════════════════════════════
# v0.9: 种子公式翻译 — S5 终端白名单 + 遗留函数规范化
# ═══════════════════════════════════════════════════════════

# 与 s5_joint_filter._build_series_context 提供的变量名对齐
_S5_KNOWN_TERMINALS = {
    # 价格/量 (含无 _p 别名)
    "close", "open", "high", "low", "volume", "amount",
    "close_p", "open_p", "high_p", "low_p", "volume_p", "amount_p",
    # 衍生
    "returns", "returns_p", "overnight", "overnight_p",
    "amplitude", "amplitude_p", "turnover", "turnover_p",
    "hl_ratio", "hl_ratio_p",
    # v0.6 moneyflow
    "buy_lg_vol", "buy_lg_vol_p", "sell_lg_vol", "sell_lg_vol_p",
    "buy_sm_vol", "buy_sm_vol_p", "sell_sm_vol", "sell_sm_vol_p",
    "buy_md_vol", "buy_md_vol_p", "sell_md_vol", "sell_md_vol_p",
    "buy_elg_vol", "buy_elg_vol_p", "sell_elg_vol", "sell_elg_vol_p",
    "net_mf_vol", "net_mf_vol_p", "net_mf_amount", "net_mf_amount_p",
    # v0.9 P-20260814-001: 两融字段 (margin_detail.csv 接入)
    "rzye", "rzye_p", "rqye", "rqye_p", "rzmre", "rzmre_p",
    "rqyl", "rqyl_p", "rqchl", "rqchl_p",
    # v0.9 P-20260812-026: 龙虎榜字段 (top_list/top_inst 接入)
    "lhb_flag", "lhb_flag_p", "lhb_net_amount", "lhb_net_amount_p",
    "lhb_net_rate", "lhb_net_rate_p", "lhb_amount", "lhb_amount_p",
    "lhb_inst_net_buy", "lhb_inst_net_buy_p",
    "lhb_inst_buy", "lhb_inst_buy_p", "lhb_inst_sell", "lhb_inst_sell_p",
    # v0.9 P-20260814-002: 北向持股字段 (hk_hold SH+SZ 接入, 2018-2024 历史窗口)
    "north_vol", "north_vol_p", "north_ratio", "north_ratio_p",
    # v0.6 balancesheet
    "intan_assets", "intan_assets_p", "goodwill", "goodwill_p",
    "total_assets", "total_assets_p",
}

# S5 eval 上下文中可用的函数名 (base_context + safe_builtins 并集)
_S5_KNOWN_FUNCS = {
    "np", "pd",
    "abs", "sqrt", "log", "log1p", "exp", "sign",
    "maximum", "minimum", "where", "clip",
    "range", "len", "int", "float", "list", "dict", "tuple", "str", "bool",
    "min", "max", "round", "sum", "zip", "sorted", "reversed",
    "enumerate", "map", "filter", "print", "isinstance",
    "Exception", "ValueError", "TypeError",
}

# 遗留 pandas 函数式调用 → 方法链 (rolling_min(x, w) → (x).rolling(w).min())
_LEGACY_ROLLING_METHODS = {
    "rolling_min": "min",
    "rolling_max": "max",
    "rolling_mean": "mean",
    "rolling_std": "std",
    "rolling_sum": "sum",
}

_LEGACY_ROLLING_RE = re.compile(
    r'\b(rolling_min|rolling_max|rolling_mean|rolling_std|rolling_sum)'
    r'\s*\(\s*([^,()]+)\s*,\s*(\d+)\s*\)'
)

# v0.6.2 (2026-08-29): 单参数省略字段形式 rolling_min(60) → (close).rolling(60).min()
# 修种子 traj_943b85a0eadc: -((close / rolling_min(60)) - close) 此前报"未知标识符 rolling_min"
_LEGACY_ROLLING_IMPLICIT_RE = re.compile(
    r'\b(rolling_min|rolling_max|rolling_mean|rolling_std|rolling_sum)'
    r'\s*\(\s*(\d+)\s*\)'
)


def _is_numeric_const(name: str) -> bool:
    """Forge 常量终端判断 (数字/科学计数/负数)。"""
    try:
        float(name)
        return True
    except (ValueError, TypeError):
        return False


# v0.6.2 (2026-08-29): 组合描述公式特征 (champion combo 的 expression 是
# "overnight5+tvma20+... rank(pct) 等权" 或 "红利×质量(roe区间5-100)×低波 全池等权"
# 之类的组合描述文本, 非单因子表达式)
_COMBO_FORMULA_MARKERS = (
    "rank(pct) 等权", "rank-product", "全池等权", "组合诊断", "非单因子",
    "三重交集", "区间5-100",
)

# 结构性翻译失败前缀 — 公式不变则永久失败, 无需每轮重试
_STRUCTURAL_TRANSLATION_FAILURE_PREFIXES = (
    "组合种子", "北向资金数据已停更",
)


def _is_combo_formula(formula: str) -> bool:
    """v0.6.2: 判定公式是否为组合描述文本 (非单因子表达式)。"""
    f = (formula or "").strip()
    if not f:
        return False
    # 含中文字符 → 描述文本 (S5 公式永远不应含中文)
    if re.search(r'[\u4e00-\u9fff]', f):
        return True
    return any(m in f for m in _COMBO_FORMULA_MARKERS)


def _is_structural_translation_failure(reason: str) -> bool:
    """v0.6.2: 结构性失败 (公式不变则永久失败) → 跳过重试。"""
    r = (reason or "").strip()
    return r.startswith(_STRUCTURAL_TRANSLATION_FAILURE_PREFIXES)


def _normalize_legacy_rolling(formula: str) -> str:
    """v0.9: 遗留函数式调用 → 方法链。

    rolling_min(close, 60)            → (close).rolling(60).min()
    rolling_min(60)                   → (close).rolling(60).min()   # v0.6.2 省略字段=close
    -((close / rolling_min(close, 60)) - close) → -((close / (close).rolling(60).min()) - close)
    """
    def _rep(m):
        fn, x, w = m.group(1), m.group(2).strip(), m.group(3)
        return "(%s).rolling(%s).%s()" % (x, w, _LEGACY_ROLLING_METHODS[fn])
    formula = _LEGACY_ROLLING_RE.sub(_rep, formula)
    # v0.6.2: 单参数形式 (隐式作用于 close)
    def _rep_imp(m):
        fn, w = m.group(1), m.group(2)
        return "(close).rolling(%s).%s()" % (w, _LEGACY_ROLLING_METHODS[fn])
    return _LEGACY_ROLLING_IMPLICIT_RE.sub(_rep_imp, formula)


def _canonicalize_neg_inline(formula: str) -> str:
    """v0.9: 内联负号规范化 — Forge 解析器不识别 `-ts_min(x, 5)` 参数形式,
    统一改写为 `neg(ts_min(x, 5))` (仅在 Forge 解析路径使用, 不影响原式)。

    实现: 扫描式匹配 `-` 紧邻 标识符`(` 的模式, 用括号配对找到参数闭合位置,
    整体替换为 neg(<fn>(<args>)) 保证括号平衡。
    中缀减号 (close - x) 与负常数 (-0.5) 不受影响。
    """
    out = []
    i, n = 0, len(formula)
    while i < n:
        ch = formula[i]
        if ch == '-' and i + 1 < n and not (i > 0 and formula[i - 1] in "0123456789."):
            m = re.match(r'\s*([A-Za-z_]\w*)\s*\(', formula[i + 1:])
            if m:
                fn = m.group(1)
                open_pos = i + 1 + m.end() - 1  # '(' 下标
                depth, j = 1, open_pos + 1
                while j < n and depth > 0:
                    if formula[j] == '(':
                        depth += 1
                    elif formula[j] == ')':
                        depth -= 1
                    j += 1
                if depth == 0:
                    inner = formula[open_pos + 1:j - 1]
                    out.append("neg(%s(%s))" % (fn, inner))
                    i = j
                    continue
        out.append(ch)
        i += 1
    return "".join(out)


def _forge_parse_failure_reason(formula: str) -> Optional[str]:
    """v0.9: Forge 前缀风格解析失败时, 定位具体原因 (未知终端/不支持语法)。"""
    from forge.primitives import PRIMITIVE_BY_NAME
    root_m = re.match(r'^([A-Za-z_]\w*)\s*\(', formula)
    if not root_m or root_m.group(1) not in PRIMITIVE_BY_NAME:
        return None
    forge_names = set(PRIMITIVE_BY_NAME.keys())
    # v0.6.1: 负向后顾避免科学计数法 1e-6 中的 e 被误报为未知终端
    tokens = set(re.findall(r'(?<![0-9.])[A-Za-z_]\w*', formula))
    # v0.6.2 (2026-08-29): 北向资金数据已停更 (hk_hold 2026-06 停更) —
    # S5 无 north_money/south_money/hgt/sgt 终端, 给明确停更原因 (结构性, 不再每轮重试)
    _north_stopped = tokens & {"north_money", "south_money", "hgt", "sgt"}
    if _north_stopped:
        return "北向资金数据已停更(hk_hold), S5 无该终端: " + ", ".join(sorted(_north_stopped))
    known = forge_names | _S5_KNOWN_TERMINALS | _S5_KNOWN_FUNCS
    unknown = sorted(
        t for t in tokens
        if t not in known and not _is_numeric_const(t)
    )
    if unknown:
        return "Forge 未知终端: " + ", ".join(unknown)
    return "Forge 公式解析失败 (内嵌语法不受支持)"


def _find_unknown_identifiers(formula: str) -> List[str]:
    """v0.9: AST 遍历找出 S5 eval 上下文中不存在的标识符。

    Returns:
        ["<语法错误>"]   — 表达式本身非法
        [...]            — 未知标识符列表 (已排除 np/pd 属性链与 builtins)
        []               — 全部可解析
    """
    import ast
    import builtins
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return ["<语法错误>"]
    allowed = _S5_KNOWN_TERMINALS | _S5_KNOWN_FUNCS | {"factor", "result"}
    allowed |= set(dir(builtins))
    unknown = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in allowed:
                unknown.add(node.id)
    return sorted(unknown)


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experience_memory import ExperienceMemory, get_memory
from multi_stage_validator import MultiStageValidator, ValidationResult
from factor_quality_gate import FactorQualityGate, GateResult
from trajectory_logger import TrajectoryLogger, MiningTrajectory

# v0.3 新增模块
from subtree_fingerprinter import SubtreeFingerprinter, get_fsa
from loop_checkpoint import CheckpointManager
from factor_expression_tree import GPBreeder, FactorExpressionParser
from factor_model_cooptim import MABScheduler, ResearchDirection

# v0.6: 连续进化轨迹 (EvoTraj)
from evo_trajectory import EvolutionTrajectory, TurnRecord, create_trajectory

# v3.1: LLM API 客户端 (统一 DeepSeek 调用)
from llm_client import get_llm_client

# v3.1: FactorForge 引擎 (完整 GP 进化)
from forge import FactorForge
from forge.paradigm_profiles import MATURE_PARADIGM_PROFILES  # v0.7 P-025

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class RalphLoop:
    """
    Ralph Loop v0.5 — 统一自进化发现引擎

    R → G → E → D → D+ 四阶段闭环。
    
    v0.5 统一架构:
      - GP 育种 + LLM 生成 双引擎
      - MAB UCB1 方向调度
      - 统一 FactorQualityGate
      - MMR 复合选择
      - Experience Memory ↔ LLM 双通道反馈
    """

    def __init__(
        self,
        memory: Optional[ExperienceMemory] = None,
        validator: Optional[MultiStageValidator] = None,
        quality_gate: Optional[FactorQualityGate] = None,
        trajectory_logger: Optional[TrajectoryLogger] = None,
        data_dir: Path = DATA_DIR,
        # v0.3 新增
        fsa: Optional[SubtreeFingerprinter] = None,
        checkpoint: Optional[CheckpointManager] = None,
        breeder: Optional[GPBreeder] = None,
        # v0.5: MAB 调度器
        mab: Optional[MABScheduler] = None,
        # v0.5: LLM 生成器
        llm_gen: Optional["LLMGenerator"] = None,
        # v3.1: FactorForge GP 引擎 + SemanticVerifier
        forge: Optional[FactorForge] = None,
        semantic_verifier: Optional["SemanticVerifier"] = None,
    ):
        # 延迟导入避免循环
        from llm_generator import LLMGenerator as _LLMGen
        from semantic_verifier import SemanticVerifier as _SemanticVerifier
        
        self.memory = memory or get_memory()
        self.validator = validator or MultiStageValidator(data_dir=data_dir)
        self.gate = quality_gate or FactorQualityGate()
        self.traj_logger = trajectory_logger or TrajectoryLogger()
        self.data_dir = data_dir

        # v0.3 组件
        self.fsa = fsa or get_fsa()
        self.checkpoint = checkpoint or CheckpointManager()

        # P-007: 子结构频率惩罚器 (持久化热度表加载; 空表时软拒绝自动失效=无行为变化)
        try:
            from substructure_frequency import load_or_build_penalizer
            self.sub_penalizer = load_or_build_penalizer(enabled=True)
        except Exception:
            self.sub_penalizer = None

        # P-20260827-001: SSPM 结构化编辑记忆 (默认 hard_veto=False=仅记录, 零行为变化)
        try:
            from edit_memory import SSPMEditMemory
            self.edit_memory = SSPMEditMemory(hard_veto_enabled=False)
        except Exception:
            self.edit_memory = None

        self.breeder = breeder or GPBreeder(
            max_depth=7, max_nodes=25,           # v0.6: 提升深度→缓解S5过度简化拦截
            penalizer=self.sub_penalizer,        # P-007: 高频子结构软拒绝
            edit_memory=self.edit_memory,        # P-20260827-001: SSPM 编辑记忆
        )
        self.parser = FactorExpressionParser()

        # v0.7 Phase 2: 轨迹蒸馏提示 (跨 run 持久化)
        self._distillation_hints: Optional[Dict] = None

        # v0.5: MAB 调度器 (控制探索方向)
        self.mab = mab or MABScheduler()
        self._mab_loaded = False
        
        # v0.5: LLM 生成器 (双引擎之第二引擎)
        self.llm_gen = llm_gen or _LLMGen()

        # v3.1: FactorForge GP 引擎 + SemanticVerifier
        self.forge = forge  # 延迟初始化，需要数据时再调用 _init_forge()
        self.semantic_verifier = semantic_verifier or _SemanticVerifier()

        # 运行统计
        self.stats = {
            "total_loops": 0,
            "total_retrieved": 0,
            "total_generated": 0,
            "total_evaluated": 0,
            "total_distilled": 0,
            "last_run": None,
        }
        
        # v0.5.1: 生成器自动退避 — 连续失败追踪
        self._breed_fail_streak = 0      # gp_breed/forge 连续 S5=0 的轮数
        self._last_generator = None       # 上一轮使用的生成器
        self._breed_fail_threshold = 2    # 连续失败阈值 → 触发 LLM 退避
        self._last_llm_paradigm = None    # LLM 退避时使用的范式

        # v0.8: 种子主体重检 — 种子原式混入验证管线的配额与冷却
        self._max_seed_recheck = 5        # 每轮最多混入的种子数
        self._seed_recheck_cooloff_days = 30  # 同一种子重检冷却期

    # ═══════════════════════════════════════════════════════════
    # v0.6 实验接线 (评价宪法)
    # 全部受 V06_EXPERIMENTAL 开关控制; 关闭=零行为变化; 异常一律静默降级
    # ═══════════════════════════════════════════════════════════

    def _v06_generation_guards(self, paradigm: str, generator: str,
                               max_candidates: int) -> Dict:
        """
        G 阶段接线: 方向微战役冻结检查 + 族级预算软警告 + 行为聚类生成指令。

        Returns: {"blocked": bool, "block_reason": str, "notes": [str]}
        """
        out = {"blocked": False, "block_reason": "", "notes": []}
        try:
            from config import V06_EXPERIMENTAL
        except Exception:
            return out

        # P1-2: 方向微战役 — 冻结方向硬拦截 (block 时由调用方降级处理)
        if V06_EXPERIMENTAL.get("direction_campaign_enabled", False) and paradigm:
            try:
                from direction_campaign import DirectionCampaign
                camp = DirectionCampaign(
                    enabled=True,
                    max_attempts=int(V06_EXPERIMENTAL.get("campaign_max_attempts", 3)),
                    early_stop_misses=int(V06_EXPERIMENTAL.get("campaign_early_stop_misses", 3)),
                    cooldown_campaigns=int(V06_EXPERIMENTAL.get("campaign_cooldown", 2)),
                )
                gen_id = f"gen{int(self.stats.get('total_loops', 0) or 0) // 50}"
                decision = camp.resolve(mab_direction=paradigm, generation_id=gen_id)
                if not decision.get("allow", True):
                    out["blocked"] = True
                    out["block_reason"] = decision.get("reason", "方向冻结")
                    out["notes"].append(
                        f"[v0.6-微战役] 方向 '{paradigm}' 被冻结: {out['block_reason']}")
                else:
                    out["notes"].append(
                        f"[v0.6-微战役] {decision.get('mode', '?')} "
                        f"campaign={decision.get('campaign_id', '?')} "
                        f"remaining={decision.get('remaining', '?')}")
            except Exception as e:
                out["notes"].append(f"[v0.6-微战役] 不可用, 降级放行: {e}")

        # P2: 族级预算 — 软警告 (不硬拦截, 避免误杀; 数据仅供监控)
        if V06_EXPERIMENTAL.get("budget_ledger_enabled", False):
            try:
                from budget_ledger import get_ledger
                ledger = get_ledger()
                fam = paradigm or "auto"
                gen = int(self.stats.get("total_loops", 0) or 0)
                ok, reason = ledger.can_generate(fam, gen)
                if not ok:
                    out["notes"].append(f"[v0.6-预算] 软警告: {reason}")
            except Exception:
                pass

        # P0-2: 行为聚类生成指令 — 拥挤簇避开 (软提示, 打印供日报引用)
        if V06_EXPERIMENTAL.get("behavior_homogeneity_enabled", False):
            try:
                from behavior_homogeneity import BehaviorHomogeneity
                bh = BehaviorHomogeneity(
                    enabled=True,
                    reject_similarity=float(V06_EXPERIMENTAL.get("behavior_similarity_reject", 0.92)),
                    substitute_similarity=float(V06_EXPERIMENTAL.get("behavior_similarity_substitute", 0.82)),
                    cluster_threshold=float(V06_EXPERIMENTAL.get("behavior_cluster_threshold", 0.74)),
                    crowded_size=int(V06_EXPERIMENTAL.get("behavior_crowded_cluster_size", 8)),
                )
                directive = bh.build_research_directive()
                avoid = directive.get("must_avoid_cluster_ids", [])
                if avoid:
                    out["notes"].append(
                        f"[v0.6-行为簇] 拥挤簇避开指令: {avoid[:5]} "
                        f"(拥挤簇数={directive.get('n_crowded_clusters', 0)})")
            except Exception:
                pass

        return out

    def _v06_evaluate_guards(
        self,
        candidates: List[Dict],
        results: List,
        *,
        library_signals: Optional[Dict] = None,
        lib_factors: Optional[List[Dict]] = None,
        s5_filter: Any = None,
    ) -> Dict:
        """
        E 阶段接线 (在 validator.validate 之后调用, v0.6 生效态):
          1) 行为同质性门禁 — 库指纹(持久化) vs 候选, NEAR_DUPLICATE 硬拒剥夺 eligible
          2) 密封盲评 — 对 eligible 候选做隐藏区盲评 (预算跨轮持久化)
          3) 配对增量门禁 — control=库组合 vs treatment=库+候选组合 (S5JointFilter 通道)

        硬拒实现: 直接修改 results[i].eligible_for_jq 并同步 summary 计数
        (调用方 _phase_evaluate 负责 summary 重算)。
        """
        out = {"behavior_flagged": 0, "behavior_rejected": 0,
               "holdout_verdicts": 0, "incremental_checked": 0,
               "incremental_rejected": 0, "notes": []}
        try:
            from config import V06_EXPERIMENTAL
        except Exception:
            return out

        # P0-2: 行为同质性门禁 (v0.6.1: 库持久化 + 硬拒生效)
        if V06_EXPERIMENTAL.get("behavior_homogeneity_enabled", False):
            try:
                from behavior_homogeneity import BehaviorHomogeneity
                bh = BehaviorHomogeneity(
                    enabled=True,
                    reject_similarity=float(V06_EXPERIMENTAL.get("behavior_similarity_reject", 0.92)),
                    substitute_similarity=float(V06_EXPERIMENTAL.get("behavior_similarity_substitute", 0.82)),
                    cluster_threshold=float(V06_EXPERIMENTAL.get("behavior_cluster_threshold", 0.74)),
                    crowded_size=int(V06_EXPERIMENTAL.get("behavior_crowded_cluster_size", 8)),
                )
                n_loaded = bh.load()
                # 库信号注入 (每轮刷新, 与持久化库合并)
                if library_signals:
                    pool = [{"factor_name": k, "signal": v}
                            for k, v in library_signals.items() if v is not None]
                    n_added = bh.build_library_from_pool(pool)
                else:
                    n_added = 0
                if n_loaded == 0 and n_added == 0:
                    out["notes"].append(
                        "[v0.6-行为簇] 库指纹为空 (无 library_signals 且无持久化库) → 门禁本轮空转")
                for cand, res in zip(candidates, results):
                    if not getattr(res, "s1_passed", False):
                        continue
                    sig = cand.get("signal")
                    if sig is None:
                        continue
                    _bh_ret = cand.get("return_series")
                    if _bh_ret is None:
                        _bh_ret = cand.get("_combo_returns")
                    v = bh.evaluate_candidate(
                        factor_id=cand.get("factor_name", "?"),
                        signal_series=sig,
                        daily_returns=_bh_ret,
                    )
                    cand["_behavior_label"] = getattr(v, "redundancy_label", "?")
                    cand["_behavior_similarity"] = round(
                        float(getattr(v, "nearest_similarity", 0.0) or 0.0), 3)
                    cand["_behavior_nearest"] = getattr(v, "nearest_factor_id", "")
                    if getattr(v, "redundancy_label", "") in ("NEAR_DUPLICATE", "SUBSTITUTE"):
                        out["behavior_flagged"] += 1
                    if not getattr(v, "gate_passed", True):
                        # 生效态: 行为冗余硬拒 → 剥夺 JQ 资格
                        out["behavior_rejected"] += 1
                        cand["_behavior_rejected"] = True
                        if getattr(res, "eligible_for_jq", False):
                            res.eligible_for_jq = False
                bh.save()  # 库指纹跨轮持久化
                if out["behavior_rejected"]:
                    out["notes"].append(
                        f"[v0.6-行为簇] 硬拒 {out['behavior_rejected']} 个 NEAR_DUPLICATE "
                        f"(库={n_loaded + n_added} 指纹)")
            except Exception as e:
                out["notes"].append(f"[v0.6-行为簇] 门禁异常(降级放行): {e}")

        # P0-1: 密封盲评 (v0.6.1: 预算跨轮持久化)
        if V06_EXPERIMENTAL.get("holdout_enabled", False):
            try:
                from holdout_boundary import HoldoutBoundary
                hb = HoldoutBoundary(
                    enabled=True,
                    holdout_start=str(V06_EXPERIMENTAL.get("holdout_local_start", "2025-07-01")),
                )
                hb.load_budget()
                gen_id = f"gen{int(self.stats.get('total_loops', 0) or 0) // 50}"
                for cand, res in zip(candidates, results):
                    if not getattr(res, "eligible_for_jq", False):
                        continue
                    ret = cand.get("return_series")
                    if ret is None:
                        ret = cand.get("_combo_returns")
                    if ret is None:
                        continue
                    verdict = hb.blind_evaluate(
                        ret, factor_id=cand.get("factor_name", "?"), generation=gen_id)
                    cand["_holdout_verdict"] = verdict.to_dict() if hasattr(verdict, "to_dict") else str(verdict)
                    out["holdout_verdicts"] += 1
                hb.save_budget()
            except Exception:
                pass

        # P1-1: 配对增量门禁 (v0.6.1 接线: S5JointFilter 组合通道)
        if (V06_EXPERIMENTAL.get("incremental_margin_enabled", False)
                and s5_filter is not None and lib_factors):
            try:
                from incremental_margin import IncrementalMarginGate
                gate = IncrementalMarginGate(
                    enabled=True,
                    min_incremental_net_ir=float(V06_EXPERIMENTAL.get("incremental_net_ir_min", 0.10)),
                    max_drawdown_deterioration=float(V06_EXPERIMENTAL.get("incremental_dd_deterioration_max", 0.02)),
                )
                # control = 库组合 (rank 和 top 8 公式), 每轮算一次
                ctl_formulas = [f.get("formula") for f in lib_factors if f.get("formula")][:8]
                control_ret = s5_filter.rank_sum_combo_returns(ctl_formulas)
                if len(control_ret) < 8:
                    out["notes"].append("[v0.6-增量] control 组合收益不足 → 本轮豁免")
                else:
                    max_pair = 3  # 每轮配对预算 (组合模拟成本控制)
                    eligible_idx = [i for i, r in enumerate(results)
                                    if getattr(r, "eligible_for_jq", False)]
                    for i in eligible_idx[:max_pair]:
                        cand = candidates[i]
                        formula = cand.get("formula", cand.get("expression", ""))
                        if not formula:
                            continue
                        treat_ret = s5_filter.rank_sum_combo_returns(ctl_formulas + [formula])
                        if len(treat_ret) < 8:
                            continue
                        v = gate.evaluate(
                            control_ret, treat_ret,
                            factor_id=cand.get("factor_name", "?"),
                        )
                        cand["_incremental_verdict"] = v.to_dict() if hasattr(v, "to_dict") else str(v)
                        out["incremental_checked"] += 1
                        if not getattr(v, "gate_passed", True):
                            out["incremental_rejected"] += 1
                            cand["_incremental_rejected"] = True
                            if getattr(results[i], "eligible_for_jq", False):
                                results[i].eligible_for_jq = False
            except Exception as e:
                out["notes"].append(f"[v0.6-增量] 门禁异常(降级放行): {e}")

        return out

    # ═══════════════════════════════════════════════════════════
    # 主循环
    # ═══════════════════════════════════════════════════════════

    def run(
        self,
        candidates: List[Dict] = None,
        library_state: Optional[Dict] = None,
        library_signals: Optional[Dict] = None,
        library_factors: Optional[Dict] = None,
        generator: str = "manual",
        generator_fn: Optional[Callable] = None,
        paradigm: str = "",
        max_candidates: int = 10,
        # v0.3 新增
        use_fsa: bool = True,
        use_reviewer: bool = True,
        use_checkpoint: bool = True,
        resume: bool = False,
        gp_templates: Optional[List[Dict]] = None,
        gp_priors = None,  # v0.4: GPPriors instance
        evo_turns: int = 5,  # v0.6: EvoTraj 最大进化轮数
        freq: str = "",  # P-20260830-001: "weekly" 时 LLM 周频语境生成
    ) -> Dict:
        """
        执行一次完整的 Ralph Loop。

        Parameters (v0.2 原有)
        ----------
        candidates: 手动提供的候选因子 (generator="manual" 时)
        library_state: 当前因子库状态
        library_signals: 库中因子的信号序列 {name: array}
        library_factors: 库中因子的元数据 {name: {icir, ...}}
        generator: "manual" | "gp_breed" | "llm" | "cross_breed" | "custom"
        paradigm: 指定范式 (v0.5: MAB 自动选择)
        max_candidates: 最大候选数
        freq: "weekly" 时 LLM 生成走周频语境 (P-20260830-001, weekly lane 供给)

        Parameters (v0.3 新增)
        ----------
        use_fsa: 是否启用频繁子树规避
        use_reviewer: 是否启用 Sub-agent 审查
        use_checkpoint: 是否启用检查点
        resume: 是否从检查点恢复
        gp_templates: GP 育种模板 (generator="gp_breed" 时)
        gp_priors: v0.4 Directed GP 先验知识 (GPPriors), 驱动语义配对+显著度+Thompson

        Returns
        -------
        完整循环结果
        """
        loop_id = f"ralph_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.stats['total_loops']:03d}"
        t0 = time.time()

        # v0.3: 检查点恢复
        if resume and use_checkpoint and self.checkpoint.can_resume():
            success, state = self.checkpoint.resume()
            if success:
                loop_id = state.loop_id or loop_id
                print(f"  [Checkpoint] 从第 {state.iteration} 轮恢复")
            else:
                self.checkpoint.start_new(loop_id=loop_id)

        elif use_checkpoint:
            self.checkpoint.start_new(loop_id=loop_id)

        print(f"\n{'='*60}")
        print(f"  Ralph Loop v0.5 #{self.stats['total_loops']}: {loop_id}")
        print(f"  R → G(GP+LLM) → E(S1-S5) → D(F/E/R) → D+(JQ)")
        print(f"{'='*60}")

        # ── P-20260831-001: RS Health 预检 + 自动重建 (每轮开始, 每日限1次) ──
        try:
            from library_orthogonality import check_rs_health_and_rebuild
            rs_check = check_rs_health_and_rebuild(auto_rebuild=True,
                                                   max_rebuilds_per_day=1)
            if rs_check.get("rebuilt"):
                print(f"  [RS-Health] 本轮前已完成自动重建: "
                      f"{rs_check.get('state_count_after')}因子 "
                      f"(触发: {rs_check.get('reason','')[:80]})")
        except Exception as _rs_ex:
            print(f"  [RS-Health] ⚠️ 预检失败 (不阻塞): {type(_rs_ex).__name__}: {_rs_ex}")

        # ── Phase R: Retrieve (v0.3: +FSA forbidden context) ──
        print(f"\n[R] Retrieving memory priors + FSA status...")
        retrieve_result = self._phase_retrieve(library_state, paradigm)
        
        # v0.3: 注入 FSA 禁止列表到检索结果
        if use_fsa:
            fsa_forbidden = self.fsa.get_forbidden_for_generation()
            retrieve_result["fsa_forbidden_context"] = fsa_forbidden

        # v0.4: 注入 Directed GP 先验到 GPBreeder
        if gp_priors is not None:
            self.breeder.priors = gp_priors
            if gp_priors.has_signal():
                print(f"  [v0.4] Directed GP 先验激活: "
                      f"{len(gp_priors.field_weights)} fields, "
                      f"{len(gp_priors.domain_complementarity)} domain pairs, "
                      f"Thompson: {gp_priors.operator_success}")

        # v0.5.1: 自动退避检查 — 即使显式指定 generator 也检查
        breed_like_generators = {"gp_breed", "forge", "gp_evolve", "cross_breed"}
        if (generator in breed_like_generators 
            and self._breed_fail_streak >= self._breed_fail_threshold):
            print(f"  [退避] {generator} 连续 {self._breed_fail_streak} 轮 S5=0 → "
                  f"强制切换到 LLM (忽略 --generator {generator})")
            generator = "llm"
            # 选择未覆盖范式
            uncovered = ["行业轮动", "北向资金", "两融信号", "大宗交易", "高频微观结构", "跨资产联动"]
            if paradigm in (None, ""):
                paradigm = uncovered[0]
                self._last_llm_paradigm = paradigm

        # v3.1: MAB 方向选择 + 生成器推荐
        if paradigm is None or paradigm == "":
            selected_dir = self._mab_select_direction()
            if selected_dir:
                paradigm = selected_dir.paradigm or paradigm
                # v0.5.1: 自动退避 — 连续 gp_breed/forge 失败 → 强制切换 LLM
                if self._breed_fail_streak >= self._breed_fail_threshold:
                    print(f"  [退避] gp_breed/forge 连续 {self._breed_fail_streak} 轮 S5=0 → 强制切换 LLM")
                    generator = "llm"
                    # 偏向未覆盖范式 (LLM 擅长冷启动)
                    uncovered = ["行业轮动", "北向资金", "两融信号", "大宗交易", "高频微观结构", "跨资产联动"]
                    direction_names = {d.name for d in (list(self.mab.directions.values()) if isinstance(self.mab.directions, dict) else self.mab.directions)}
                    for u in uncovered:
                        if u in direction_names:
                            paradigm = u
                            self._last_llm_paradigm = u
                            break
                # 如果未指定 generator, 由 MAB 推荐
                elif generator in (None, "", "gp_breed"):
                    generator = self._mab_select_generator(selected_dir)
                print(f"  [MAB] 方向={selected_dir.name} paradigm={paradigm} "
                      f"generator={generator} pulls={selected_dir.pulls}")

        # ── Phase G: Generate (v0.3: +GP breed + reviewer) ───
        print(f"[G] Generating candidates (generator={generator})...")
        generate_result = self._phase_generate(
            candidates, retrieve_result, generator, generator_fn,
            paradigm, max_candidates,
            use_fsa=use_fsa, use_reviewer=use_reviewer,
            gp_templates=gp_templates, evo_turns=evo_turns,
        )

        # ── Phase E: Evaluate ─────────────────────────────
        print(f"[E] Evaluating {len(generate_result['candidates'])} candidates...")
        evaluate_result = self._phase_evaluate(
            generate_result["candidates"],
            library_signals,
            library_factors,
        )

        # ── Phase D: Distill ──────────────────────────────
        print(f"[D] Distilling experience...")
        distill_result = self._phase_distill(
            generate_result["candidates"],
            evaluate_result,
            paradigm,
        )

        # ── Summary ────────────────────────────────────────
        elapsed = time.time() - t0
        eligible = [r for r in evaluate_result["results"] if r.eligible_for_jq]
        s5_passed = evaluate_result.get("stage5_passed", 
                    sum(1 for r in evaluate_result["results"] if getattr(r, 'grade', 'D') in ('A', 'B', 'C')))

        # v0.5.1: 生成器自动退避 — 更新连续失败计数
        s5_count = evaluate_result.get("stage5_passed", 0)
        # 也检查 total_candidates vs eligible
        n_candidates = len(generate_result.get("candidates", []))
        n_passed_any_stage = sum(1 for r in evaluate_result.get("results", []) 
                                 if getattr(r, 'stage1_passed', False))
        
        if generator in ("gp_breed", "forge", "gp_breed_fallback"):
            if s5_count == 0 and n_candidates > 0:
                self._breed_fail_streak += 1
                print(f"  [退避] gp_breed/forge S5=0 连续 {self._breed_fail_streak} 轮 "
                      f"(阈值={self._breed_fail_threshold})")
            else:
                self._breed_fail_streak = 0
        elif generator == "llm":
            # LLM 成功后重置 (即使 S5=0, LLM 生成的新因子也算注入新血统)
            if n_passed_any_stage > 0:
                self._breed_fail_streak = 0
                print(f"  [退避] LLM 注入成功 → 退避计数重置")
        self._last_generator = generator

        # v0.3: 更新检查点
        if use_checkpoint and self.checkpoint:
            self.checkpoint._state.iteration = self.stats["total_loops"]
            for cand in generate_result["candidates"]:
                self.checkpoint.mark_tested(cand)
                result = next(
                    (r for r in evaluate_result["results"]
                     if r.factor_name == cand.get("factor_name", "")),
                    None
                )
                if result and result.eligible_for_jq:
                    self.checkpoint.mark_approved(cand)
            self.checkpoint.save(iteration=self.stats["total_loops"])
            print(f"  [Checkpoint] 已保存: {self.checkpoint._state.total_tested} 测试, "
                  f"{self.checkpoint._state.total_approved} 入库")

        # v0.8: 记录种子重检结果 (冷却 + S5 结论)
        seed_rechecked = [c for c in generate_result["candidates"] if c.get("_seed_recheck")]
        if seed_rechecked:
            result_map = {r.factor_name: r for r in evaluate_result.get("results", [])}
            n_seed_jq = 0
            for cand in seed_rechecked:
                result = result_map.get(cand.get("factor_name", ""))
                s5_info = {}
                if result is not None:
                    s5_info = {
                        "eligible_for_jq": bool(result.eligible_for_jq),
                        "stage5_passed": bool(getattr(result, "stage5_passed", False)),
                        "calmar_2025": float(getattr(result, "calmar_2025", 0) or 0),
                        "calmar_2026": float(getattr(result, "calmar_2026", 0) or 0),
                    }
                    if result.eligible_for_jq:
                        n_seed_jq += 1
                self.record_seed_recheck(cand.get("factor_name", ""), s5_result=s5_info)
            if n_seed_jq > 0:
                print(f"  [SeedRecheck] {n_seed_jq}/{len(seed_rechecked)} 个种子 S5 通过 → 进入 JQ 候选队列")

        # v0.3: FSA 因子库扫描
        if use_fsa and evaluate_result["results"]:
            # 收集当前已入库的因子用于 FSA 扫描
            if library_factors:
                all_factors = [
                    {"factor_name": name, "formula": info.get("formula", "")}
                    for name, info in library_factors.items()
                ]
                # 添加本轮新入库因子
                for r in eligible:
                    cand = next(
                        (c for c in generate_result["candidates"]
                         if c.get("factor_name") == r.factor_name),
                        None
                    )
                    if cand:
                        all_factors.append({
                            "factor_name": r.factor_name,
                            "formula": cand.get("formula", ""),
                        })
                if len(all_factors) >= self.fsa.min_factors_to_trigger:
                    fsa_report = self.fsa.scan_library(
                        all_factors, persist=True
                    )
                    if fsa_report.forbidden_count > 0:
                        print(f"  [FSA] 冻结 {fsa_report.forbidden_count} 个骨架, "
                              f"最大集中度 {fsa_report.max_concentration:.1%}")

        # v0.9: JQ 候选详情 (公式/来源/种子标记) — 供 run_v4_pipeline 自动生成 JQ 代码
        cand_map = {c.get("factor_name"): c for c in generate_result["candidates"]}
        jq_candidate_details = []
        for r in eligible:
            cand = cand_map.get(r.factor_name, {})
            jq_candidate_details.append({
                "factor_name": r.factor_name,
                "formula": cand.get("formula", getattr(r, "formula", "")),
                "formula_original": cand.get("_formula_original", ""),
                "paradigm": cand.get("paradigm", getattr(r, "paradigm", "")),
                "hypothesis": cand.get("hypothesis", ""),
                "seed_recheck": bool(cand.get("_seed_recheck")),
                "source": cand.get("_source", cand.get("source", "generated")),
                # v0.7 频率对称: S1 裁决口径透传 (JQ 队列/D+ 蒸馏归因)
                "natural_freq": cand.get("natural_freq", "daily"),
                "grade": getattr(r, "final_grade", ""),
                "s5_calmar": float(getattr(r, "s5_calmar", 0) or 0),
                "s5_passed": bool(getattr(r, "s5_passed", False)),
            })

        summary = {
            "loop_id": loop_id,
            "elapsed_seconds": round(elapsed, 1),
            "total_candidates": generate_result["n_candidates"],
            "source": generate_result.get("source", "unknown"),
            "stages_completed": {
                "retrieve": retrieve_result["n_retrieved"],
                "generate": generate_result["n_candidates"],
                "evaluate": evaluate_result["summary"],
                "distill": distill_result["summary"],
            },
            "fsa_status": self.fsa.get_status() if use_fsa else {},
            "checkpoint_progress": (
                self.checkpoint.get_progress() if use_checkpoint else {}
            ),
            "jq_candidates": [r.factor_name for r in eligible],
            "n_jq_candidates": len(eligible),
            "jq_candidate_details": jq_candidate_details,
            # v0.6: EvoTraj 轨迹信息
            "evo_trajectories": self._extract_trajectories(generate_result),
        }

        # 更新统计
        self.stats["total_loops"] += 1
        self.stats["total_retrieved"] += retrieve_result["n_retrieved"]
        self.stats["total_generated"] += generate_result["n_candidates"]
        self.stats["total_evaluated"] += generate_result["n_candidates"]
        self.stats["total_distilled"] += distill_result["summary"].get("new_patterns", 0)
        self.stats["last_run"] = datetime.now().isoformat()

        print(f"\n{'='*60}")
        print(f"  Ralph Loop 完成 ({elapsed:.1f}s)")
        print(f"  检索: {retrieve_result['n_retrieved']} priors")
        print(f"  候选: {generate_result['n_candidates']}")
        print(f"  验证: {evaluate_result['summary']['eligible_for_jq']} JQ候选")
        print(f"  蒸馏: {distill_result['summary'].get('new_patterns', 0)} 新模式")
        print(f"{'='*60}")

        # v0.6: 每轮结束强制保存 MAB state (确保 pulls 跨 run 持久化)
        try:
            self.mab.save_state()
        except Exception:
            pass

        # P-007: 每轮结束刷新子结构热度表 (并入本轮公式) + Top10 审计输出
        try:
            self._refresh_substructure_penalty()
        except Exception as e:
            print(f"  [Substructure] ⚠️ 热度表刷新失败 (不阻塞): {e}")

        return {
            "loop_id": loop_id,
            "phases": {
                "retrieve": retrieve_result,
                "generate": generate_result,
                "evaluate": evaluate_result,
                "distill": distill_result,
            },
            "summary": summary,
            "jq_candidates": [r.factor_name for r in eligible],
            "jq_candidate_details": jq_candidate_details,
            # v0.6: 轨迹摘要 (供 MAB streak 奖励)
            "evo_trajectories": self._extract_trajectories(generate_result),
        }

    # ═══════════════════════════════════════════════════════════
    # P-007: 子结构频率库刷新 (每轮结束调用)
    # ═══════════════════════════════════════════════════════════

    def _refresh_substructure_penalty(self) -> None:
        """重建子结构热度表 (公式源=因子提案库+JQ轨迹, 含 JQ 成败联动) 并打印 Top10"""
        from substructure_frequency import build_penalizer_from_library

        formulas: List[str] = []
        jq_records: List = []
        try:
            with open(self.data_dir / "stage1_factor_proposals.json",
                      encoding="utf-8") as f:
                fp = json.load(f)
            flist = fp if isinstance(fp, list) else fp.get("proposals", fp.get("factors", []))
            formulas.extend(
                x.get("formula_pandas", "") for x in flist if x.get("formula_pandas")
            )
        except Exception:
            pass
        try:
            with open(self.data_dir / "trajectory_log.json", encoding="utf-8") as f:
                tl = json.load(f)
            for t in tl.get("trajectories", []):
                if not t.get("jq_validated"):
                    continue
                expr = t.get("expression", {})
                content = expr.get("content", "") if isinstance(expr, dict) else str(expr)
                if not content:
                    continue
                jq_ret = t.get("jq_return") or 0
                jq_records.append((content, jq_ret >= 0))
                formulas.append(content)
        except Exception:
            pass

        pz = build_penalizer_from_library(formulas, jq_records, enabled=True)
        self.sub_penalizer = pz
        if getattr(self, "breeder", None) is not None:
            self.breeder.penalizer = pz
        if getattr(self, "forge", None) is not None:
            self.forge.penalizer = pz
        print(pz.audit_report(n=10))

    # ═══════════════════════════════════════════════════════════
    # Phase R: Retrieve
    # ═══════════════════════════════════════════════════════════

    def _phase_retrieve(
        self,
        library_state: Optional[Dict],
        paradigm: str,
    ) -> Dict:
        """
        R 阶段: 从 Experience Memory 检索上下文相关 priors。

        检索内容:
        - P_succ (成功因子模板) → 指导 LLM 重用有效模式
        - P_fail (禁止方向) → 避免 LLM 重复已知失败
        - Exploration priorities → 告诉 LLM 优先探索哪些方向
        """
        library_context = library_state or {}

        # 使用 F/E/R 中的 R (Retrieval) 操作符
        retrieval = self.memory.retrieve(
            library_context=library_context,
            paradigm=paradigm,
            k=5,
        )

        # 补充范式覆盖率上下文
        coverage_ctx = self.memory.get_paradigm_coverage_context()

        # 补充禁止区域上下文
        forbidden_ctx = self.memory.get_forbidden_context_for_llm()

        # 合成统一的 memory priors context
        combined_context = [
            retrieval.get("llm_prompt_fragment", ""),
            "",
            coverage_ctx,
            "",
            forbidden_ctx,
        ]

        n_retrieved = (
            len(retrieval.get("success_templates", []))
            + len(retrieval.get("forbidden_directions", []))
            + len(retrieval.get("exploration_priorities", []))
        )

        return {
            "n_retrieved": n_retrieved,
            "success_templates": retrieval.get("success_templates", []),
            "forbidden_directions": retrieval.get("forbidden_directions", []),
            "exploration_priorities": retrieval.get("exploration_priorities", []),
            "llm_prompt_context": "\n".join(combined_context),
            "paradigm_coverage": coverage_ctx,
            "forbidden_context": forbidden_ctx,
        }

    # ═══════════════════════════════════════════════════════════
    # Phase G: Generate
    # ═══════════════════════════════════════════════════════════

    def _phase_generate(
        self,
        candidates: List[Dict],
        retrieve_result: Dict,
        generator: str,
        generator_fn: Optional[Callable],
        paradigm: str,
        max_candidates: int,
        use_fsa: bool = True,
        use_reviewer: bool = True,
        gp_templates: Optional[List[Dict]] = None,
        evo_turns: int = 5,
    ) -> Dict:
        """
        G 阶段: 生成候选因子 (v0.5: MAB 方向选择 + 统一 FactorQualityGate)。

        生成方式:
        - "manual": 使用传入候选项
        - "gp_breed": GP 表达式树育种 (五维演化)
        - "cross_breed": 从记忆模板交叉育种
        - "custom": 调用自定义生成函数
        """
        # v0.5: MAB 方向选择 — 决定本轮探索哪个范式
        if paradigm is None or paradigm == "":
            selected_dir = self._mab_select_direction()
            if selected_dir:
                paradigm = selected_dir.paradigm or paradigm
                print(f"  [MAB] 选择探索方向: {selected_dir.name} (paradigm={paradigm}, "
                      f"pulls={selected_dir.pulls}, reward={selected_dir.expected_reward:.3f})")

        # v0.6 实验接线: 方向微战役 + 族级预算 + 行为聚类指令
        # 开关全关 = 零行为变化 (rollback 安全); 方向被冻结时降级为无范式约束生成
        v06_guard = self._v06_generation_guards(paradigm, generator, max_candidates)
        for _n in v06_guard.get("notes", []):
            print(f"  {_n}")
        if v06_guard.get("blocked"):
            print(f"  [v0.6-微战役] 方向 '{paradigm}' 冻结 → 本轮降级无范式约束生成")
            paradigm = ""

        # v0.6 P-020: Memory Bridge — 高 occ Memory 模板 → GP 育种种子优先消费
        # 在 generator=gp_breed|gp_evolve 且缺少 gp_templates 时自动注入
        if generator in ("gp_breed", "gp_evolve") and not gp_templates:
            try:
                from seed_injector import inject_from_memory_templates, mark_bridge_templates_consumed
                bridge_seeds = inject_from_memory_templates(
                    memory=self.memory,
                    min_occurrence=100,
                    verbose=True,
                )
                if bridge_seeds:
                    # 转换 seed_injector 格式 → gp_templates 格式
                    gp_templates = [
                        {
                            "formula": s.get("formula", s.get("expression", "")),
                            "factor_name": s.get("factor_name", s.get("pattern_id", "")),
                            "paradigm": s.get("paradigm", ""),
                            "source": "memory_bridge",
                        }
                        for s in bridge_seeds
                    ]
                    print(f"  [P-020] Memory Bridge: {len(gp_templates)} 个桥接模板注入 GP 育种")
                    # 不立即标记 consumed — 等 GP 育种完成后再标记
            except Exception as e:
                print(f"  [P-020] Memory Bridge 注入失败 (非阻塞): {e}")

        if generator == "manual":
            gen_candidates = candidates or []
            source = "manual"
        elif generator == "gp_breed":
            templates = gp_templates or retrieve_result.get("success_templates", [])
            if not templates:
                templates_raw = self.memory.data.get("success_templates", [])
                templates = []
                for t in templates_raw:
                    if isinstance(t, dict):
                        templates.append({
                            "formula": t.get("formula", t.get("expression", "")),
                            "factor_name": t.get("pattern_id", t.get("name", "")),
                        })
                    elif hasattr(t, 'formula'):
                        templates.append({
                            "formula": getattr(t, 'formula', ''),
                            "factor_name": getattr(t, 'pattern_id', getattr(t, 'name', '')),
                        })

            # 统一转换: 确保所有模板都是 dict 格式 (适配 GPBreeder)
            converted_templates = []
            for t in templates:
                if isinstance(t, dict):
                    converted_templates.append(t)
                elif hasattr(t, 'formula'):
                    converted_templates.append({
                        "formula": getattr(t, 'formula', getattr(t, 'expression', '')),
                        "factor_name": getattr(t, 'pattern_id', getattr(t, 'name', '')),
                    })
                else:
                    converted_templates.append({"formula": str(t), "factor_name": str(t)[:50]})
            templates = converted_templates

            # P-20260819-005: 轨迹父本注入 gp_breed 主路径 (2026-08-19 修正:
            # 原注入点仅在 gp_evolve 的 _phase_generate_evo, 主线 gp_breed 从未生效)
            try:
                from trajectory_pool import ENABLE_TRAJECTORY_POOL, get_trajectory_parents
                if ENABLE_TRAJECTORY_POOL:
                    tp_par = get_trajectory_parents(
                        data_dir=self.data_dir,
                        n=min(3, max(1, max_candidates // 3)),
                        paradigm=paradigm)
                    if tp_par:
                        for tp in tp_par:
                            templates.append({
                                "formula": tp["formula"],
                                "factor_name": f"traj_{tp['factor_name']}",
                            })
                        print(f"  [TrajPool] 轨迹父本注入 gp_breed 主路径 {len(tp_par)} 条")
                    else:
                        print("  [TrajPool] 无可用轨迹父本 (范式无匹配或全在冷却期)")
            except Exception as e:
                print(f"  [TrajPool] 注入失败(不影响主线): {e}")

            fsa_for_breed = self.fsa if use_fsa else None
            gen_candidates = self.breeder.breed_from_templates(
                templates=templates,
                n_children=max_candidates,
                fsa=fsa_for_breed,
                output_format="pandas",
                paradigm=paradigm,  # v0.9.1: MAB 方向透传 (修复 auto_breed 断链)
            )
            source = "gp_breed"
            print(f"  [GP Breed] 从 {len(templates)} 个模板育种出 {len(gen_candidates)} 个候选")
            # v0.6 P-020: 标记 Memory Bridge 模板已消费
            if gp_templates:
                try:
                    from seed_injector import mark_bridge_templates_consumed
                    for t in gp_templates:
                        if t.get("source") == "memory_bridge":
                            pid = t.get("factor_name", "").replace("memory_bridge::", "")
                            if pid:
                                self.memory.mark_bridge_consumed(pid)
                except Exception:
                    pass
        elif generator == "cross_breed":
            gen_candidates = self._cross_breed_from_templates(
                retrieve_result["success_templates"],
                paradigm,
                max_candidates,
            )
            source = "cross_breed"
        elif generator == "llm":
            # v3.1: LLM 因子生成 (双引擎) — 直接调用 DeepSeek API
            self.llm_gen.receive_context(
                success_templates=retrieve_result.get("success_templates", []),
                forbidden_directions=retrieve_result.get("forbidden_directions", []),
                fsa_forbidden=retrieve_result.get("fsa_forbidden_context", ""),
                library_stats=retrieve_result.get("library_stats", {}),
                paradigm_priorities=retrieve_result.get("exploration_priorities", {}),
                mab_direction=paradigm,
            )
            # 注入 v3.1: Warning 方向 (软负收益) + Motif 规则
            try:
                warnings = self.memory.get_warning_directions()
                if warnings:
                    self.llm_gen.receive_context(warning_directions=warnings)
                motif_rules = self.memory.get_motif_rules()
                if motif_rules:
                    self.llm_gen.receive_context(motif_rules=motif_rules)
            except Exception:
                pass

            # v0.5: RAG 知识注入 — 从本地知识库检索前沿研究
            try:
                from rag_context import retrieve_for_generation
                rag_result = retrieve_for_generation(
                    paradigm=paradigm,
                    mab_direction=getattr(self, '_last_mab_name', ''),
                    top_k=5,
                )
                if rag_result.get("frontier_research") or rag_result.get("failure_patterns"):
                    n_frontier = len(rag_result.get("frontier_research", []))
                    n_fail = len(rag_result.get("failure_patterns", []))
                    print(f"  [RAG] 检索到 {n_frontier} 条前沿 + {n_fail} 条失败模式 (query: {rag_result.get('query', '?')[:60]})")
                    self.llm_gen.receive_context(rag_knowledge=rag_result)
                else:
                    print(f"  [RAG] 检索无结果 (query: {rag_result.get('query', '?')[:60]})")
            except Exception as e:
                print(f"  [RAG] 检索失败 (非阻塞): {e}")

            prompt = self.llm_gen.build_prompt(
                target_paradigm=paradigm, n_factors=max_candidates,
                weekly_mode=(freq == "weekly"),
            )
            print(f"  [LLM Gen] 调用 DeepSeek API (model=deepseek-chat)"
                  + (" [周频语境]" if freq == "weekly" else "") + "...")

            try:
                llm_client = get_llm_client()
                # P1-3 (v0.6 实验): 评价宪法 system_prompt 注入
                # 开关 V06_EXPERIMENTAL["g_prompt_constitution"] 关闭即回退 v0.5.2 (rollback)
                _sys_prompt = (
                    "你是 A 股量化因子研究员，专精于设计具有经济逻辑支撑的因子表达式。"
                    "使用 pandas 风格表达式（如 close.rolling(20).mean()）。"
                )
                try:
                    from config import V06_EXPERIMENTAL
                    if V06_EXPERIMENTAL.get("g_prompt_constitution", False):
                        _sys_prompt += (
                            "你的因子将接受严格的评价宪法审查: 以加入组合后的增量边际为准"
                            "(高IC但无组合增量的因子会被拒), 与库内因子行为高度相似的变体会被降级, "
                            "探索方向有尝试预算, 最终盲评验证区对生成端不可见。"
                            "请优先设计机制正交、经济逻辑清晰、可解释的因子。"
                        )
                except Exception:
                    pass  # config 不可用时回退基础 system_prompt
                response_text = llm_client.chat_with_system(
                    system_prompt=_sys_prompt,
                    user_prompt=prompt,
                    temperature=0.8,
                    max_tokens=4096,
                )
                gen_candidates = self.llm_gen.parse_response(response_text)
                if freq in ("weekly", "daily"):
                    # P-20260830-001: 分频语境生成 → 候选显式打 natural_freq 标签
                    for _c in gen_candidates:
                        _c["natural_freq"] = freq
                source = "llm"
                print(f"  [LLM Gen] API 返回 {len(response_text)} chars → 解析出 {len(gen_candidates)} 个候选"
                      + (f" (natural_freq={freq})" if freq else ""))
            except Exception as e:
                # Fallback: 降级为文件模式
                print(f"  [LLM Gen] API 调用失败: {e}，降级为文件模式")
                prompt_file, _ = self.llm_gen.generate(
                    target_paradigm=paradigm,
                    n_factors=max_candidates,
                    mode="file",
                )
                gen_candidates = []
                source = "llm"
                print(f"  [LLM Gen] Prompt → {prompt_file}，等待手动提交")
        elif generator == "forge":
            # v3.1: FactorForge GP 引擎 — 完整遗传编程进化
            # v0.7 P-025 固化: pop 30→200 (Round1c 实测参数)
            gen_candidates = self._generate_via_forge(
                pop_size=200,
                n_generations=5,
                max_candidates=max_candidates,
            )
            source = "forge"
            if not gen_candidates:
                # 回退到 GP 育种
                print(f"  [Forge] 无候选产出，回退到 GP Breed")
                templates = gp_templates or retrieve_result.get("success_templates", [])
                if templates:
                    gen_candidates = self.breeder.breed_from_templates(
                        templates=templates,
                        n_children=max_candidates,
                        fsa=self.fsa if use_fsa else None,
                        output_format="pandas",
                    )
                    source = "gp_breed_fallback"
        elif generator == "gp_evolve":
            # v0.6: GP 多轮进化 (EvoTraj) — 对标 AlphaAgentEvo
            templates_raw = gp_templates or retrieve_result.get("success_templates", [])
            # 统一转换: SuccessPattern / dict → {formula, factor_name}
            templates = []
            for t in templates_raw:
                if isinstance(t, dict):
                    templates.append({
                        "formula": t.get("formula", t.get("expression", "")),
                        "factor_name": t.get("pattern_id", t.get("factor_name", t.get("name", ""))),
                        "sample_factor_ids": t.get("sample_factor_ids", []),  # P-005 v2
                    })
                elif hasattr(t, 'formula'):
                    templates.append({
                        "formula": getattr(t, 'formula', getattr(t, 'expression', '')),
                        "factor_name": getattr(t, 'pattern_id', getattr(t, 'name', 'unknown')),
                        "sample_factor_ids": getattr(t, 'sample_factor_ids', []),  # P-005 v2
                    })
            source, gen_candidates = self._phase_generate_evo(
                templates=templates,
                retrieve_result=retrieve_result,
                paradigm=paradigm,
                max_candidates=max_candidates,
                max_turns=evo_turns,
                use_fsa=use_fsa,
            )
        elif generator == "custom" and generator_fn:
            gen_candidates = generator_fn(
                memory_priors=retrieve_result,
                paradigm=paradigm,
                max_candidates=max_candidates,
            )
            source = "custom"
        else:
            gen_candidates = []
            source = "unspecified"

        # v0.8: 种子主体混入 — 种子因子不只做模板养料, 原式也作为主体候选
        # 同批走 S1-S5 → 通过即入 jq_candidates (打通"种子→JQ单因子验证"通道)
        if candidates and generator != "manual":
            seed_recheck = self._prepare_seed_recheck_candidates(
                candidates, max_seeds=self._max_seed_recheck
            )
            if seed_recheck:
                gen_candidates = seed_recheck + list(gen_candidates or [])
                print(f"  [SeedRecheck] 混入 {len(seed_recheck)} 个种子原式同批验证 "
                      f"({[c['factor_name'][:28] for c in seed_recheck[:3]]}...)")

        # v0.3: FSA 过滤
        if use_fsa and gen_candidates:
            original_count = len(gen_candidates)
            gen_candidates = [
                c for c in gen_candidates
                if not self.fsa.check_expression(
                    c.get("formula", c.get("expression", ""))
                )
            ]
            removed = original_count - len(gen_candidates)
            if removed > 0:
                print(f"  [FSA] 过滤掉 {removed} 个禁止骨架候选")

        # v0.3: 检查点去重
        if self.checkpoint and gen_candidates:
            # v0.8: seed_recheck 种子豁免 checkpoint 去重 (种子重检是周期性裁决,
            #       不是新因子发现; 冷却由 seed_recheck_state.json 管理)
            seed_cands = [c for c in gen_candidates if c.get("_seed_recheck")]
            breed_cands = [c for c in gen_candidates if not c.get("_seed_recheck")]
            if breed_cands:
                breed_cands = self.checkpoint.get_untested(breed_cands)
            gen_candidates = seed_cands + breed_cands
            if len(gen_candidates) < max_candidates:
                print(f"  [Checkpoint] 去重后剩余 {len(gen_candidates)} 个候选")

        # v0.5: 统一质量门禁 (替代分离的 Reviewer + Verifier)
        if use_reviewer and gen_candidates:
            # P-017: CodeGate 预检 + auto_fix 循环
            n_fixed = 0
            n_code_rejected = 0
            for cand in gen_candidates:
                formula = cand.get("formula", cand.get("expression", ""))
                code = cand.get("code", "")
                cg = FactorQualityGate.preflight_code_check(code, formula)
                if not cg["passed"]:
                    # 尝试自动修复
                    fixed_code = FactorQualityGate.codegate_self_debug(
                        code, cg["errors"][0] if cg["errors"] else ""
                    )
                    if fixed_code:
                        cand["code"] = fixed_code
                        n_fixed += 1
                    else:
                        cand["_codegate_failed"] = True
                        cand["_codegate_errors"] = cg["errors"]
                        n_code_rejected += 1
                elif cg["warnings"]:
                    cand["_codegate_warnings"] = cg["warnings"]
            if n_fixed > 0 or n_code_rejected > 0:
                print(f"  [CodeGate] {n_fixed} fixed, {n_code_rejected} rejected")
            # 移除 CodeGate 失败的候选
            gen_candidates = [c for c in gen_candidates if not c.get("_codegate_failed")]

            for cand in gen_candidates:
                gate_result = self.gate.verify(cand)
                cand["_gate_result"] = gate_result
                if not gate_result.passed:
                    cand["_gate_fatal"] = gate_result.fatal_issues
                    cand["_gate_warnings"] = gate_result.warnings
            n_passed = sum(1 for c in gen_candidates if c.get("_gate_result") and c["_gate_result"].passed)
            print(f"  [QualityGate] {n_passed}/{len(gen_candidates)} 通过统一门禁")
            # v0.5 伏羲: 打印详细拒绝原因 (仅前3个)
            if n_passed < len(gen_candidates):
                rejected = [c for c in gen_candidates if not (c.get("_gate_result") and c["_gate_result"].passed)]
                for i, rc in enumerate(rejected[:3]):
                    fatal = rc.get("_gate_fatal", [])
                    formula_preview = rc.get("formula", "")[:60]
                    print(f"    ❌ #{i+1} [{formula_preview}...]: {'; '.join(fatal[:2])}")

        # 轨迹日志
        for cand in gen_candidates:
            traj = self.traj_logger.start_trajectory(
                paradigm=cand.get("paradigm", paradigm),
                seed_factor=cand.get("seed_factor", ""),
                phase="ralph_loop",
                natural_freq=cand.get("natural_freq", "daily"),  # v0.7 频率对称
            )
            cand["_trajectory_id"] = traj.trajectory_id

            if cand.get("hypothesis"):
                gate = cand.get("_gate_result")
                passed = gate.passed if gate else False
                self.traj_logger.log_hypothesis(
                    traj.trajectory_id, cand["hypothesis"],
                    score=0.6 if passed else 0.3,
                )
            if cand.get("formula"):
                cx = self.gate._check_complexity(cand["formula"])
                self.traj_logger.log_expression(
                    traj.trajectory_id, cand["formula"],
                    score=0.8 if cx["complexity_pass"] else 0.4,
                    node_count=cx["node_count"],
                    ast_depth=cx["ast_depth"],
                )

        # 注入 memory priors 到每个候选
        for cand in gen_candidates:
            cand["_memory_priors"] = retrieve_result["llm_prompt_context"][:2000]
            cand["_source"] = source
            # v0.7 频率对称: natural_freq 兜底注入 (显式标签优先, 缺失按 source 推断)
            if not cand.get("natural_freq"):
                try:
                    from weekly_lane import infer_natural_freq
                    cand["natural_freq"] = infer_natural_freq(cand)
                except Exception:
                    cand["natural_freq"] = "weekly" if source == "factorforge" else "daily"
            if use_fsa:
                cand["_fsa_forbidden"] = retrieve_result.get("fsa_forbidden_context", "")[:500]
            # v0.5: 注入 MAB 选择的范式方向
            if paradigm:
                cand["_mab_paradigm"] = paradigm

        return {
            "n_candidates": len(gen_candidates),
            "candidates": gen_candidates,
            "source": source,
            "mab_direction": paradigm,
        }

    # ── v0.8: 种子主体重检 ─────────────────────────────

    def _prepare_seed_recheck_candidates(
        self,
        candidates: List[Dict],
        max_seeds: int = 5,
    ) -> List[Dict]:
        """
        从 candidates (run_v4_pipeline Step 2 构造的种子列表) 中挑选
        "值得作为主体重测"的种子原式, 混入本轮验证管线。

        设计意图 (v0.8):
          - 种子因子双身份: 模板养料 (亲本) + 主体候选 (原式重测)
          - 组合级 JQ 背书 (champion) ≠ 单因子验证 → 需单因子重测
          - 通过 S1-S5 的种子自然进入 jq_candidates → 用户送 JQ 单因子验证

        过滤规则:
          1. 已单因子 JQ 验证过的 (state 中 jq_single_verified=True) → 跳过
          2. 冷却期内已重检过的 (默认 30 天) → 跳过
          3. 排序: champion_seed 优先, 然后按 icir 降序
          4. 上限 max_seeds 个

        Returns:
          带 _seed_recheck=True 标记的候选列表
        """
        if not candidates:
            return []
        state = self._load_seed_recheck_state()
        now = datetime.now()
        cool_days = self._seed_recheck_cooloff_days

        picked = []
        for c in candidates:
            name = c.get("factor_name", "")
            formula = c.get("formula", "")
            if not name or not formula:
                continue
            st = state.get(name, {})
            # 1. 已单因子 JQ 验证 → 不再重测
            if st.get("jq_single_verified"):
                continue
            # 1b. 已重检且 S5 明确未通过 → 永久跳过 (公式未变, 周期重测无意义)
            s5_prev = st.get("s5_result") or {}
            if s5_prev.get("stage5_passed") is False:
                continue
            # 2. 冷却期内已重检 → 跳过
            last = st.get("last_check", "")
            if last:
                try:
                    last_dt = datetime.fromisoformat(last)
                    if (now - last_dt).days < cool_days:
                        continue
                except Exception:
                    pass
            # v0.6.2 (2026-08-29): 结构性翻译失败缓存 — 组合种子/数据停更类
            # 失败永久有效 (公式未变), 跳过重试 (消 5 个固定失败种子每轮报错)
            tr_prev = st.get("translate") or {}
            if tr_prev.get("ok") is False and _is_structural_translation_failure(
                    tr_prev.get("reason", "")):
                continue
            # v0.6.2 (2026-08-29): 组合种子检测 — 公式是组合描述文本 (含中文/
            # "rank(pct) 等权"/"rank-product"/"全池等权") 而非单因子表达式,
            # 不进入单因子重检 (v3_traj_30fd4a7de4e2/v3_variantA_gpevo006/
            # v3_dividend_quality_lowvol 三个 combo 种子此前每轮报"未知标识符")
            if _is_combo_formula(formula):
                self._note_seed_translation(
                    name, ok=False,
                    reason="组合种子(非单因子公式): " + formula[:80])
                continue
            # v0.9: 公式口径统一 — S5 只能执行 pandas 表达式;
            #       Forge 风格公式 (champion 原式) 经 to_pandas_string 翻译,
            #       翻译失败显式报因落盘 (不再静默跳过)
            translated, reason = self._translate_seed_formula(formula)
            if not translated:
                self._note_seed_translation(name, ok=False, reason=reason)
                print(f"  [SeedRecheck] 翻译失败跳过: {name}: {reason}")
                continue
            self._note_seed_translation(name, ok=True, reason="")
            c2 = dict(c)
            c2["formula"] = translated
            c2["_formula_original"] = formula[:500]
            picked.append(c2)

        # 3. champion_seed 优先, 其次 icir 降序
        picked.sort(
            key=lambda c: (
                0 if c.get("status") == "champion_seed" else 1,
                -(float(c.get("icir") or 0) or 0),
            )
        )
        seeds = picked[:max_seeds]
        return [dict(s, _seed_recheck=True, _source="seed_recheck") for s in seeds]

    def _translate_seed_formula(self, formula: str):
        """种子公式统一为 pandas 风格 (S5 执行要求) — v0.9 三缺口修复。

        Returns:
            (translated: str, reason: str)
              - translated 非空 → 成功, reason=""
              - translated==""  → 失败, reason 说明原因 (不再静默跳过)

        修复历史缺口:
          1. 中缀 + 遗留 pandas 函数风格 (rolling_min(close,60)) → 方法链重写
          2. Forge 前缀风格含未知终端 (north_money) → 显式报因
          3. 所有路径失败都有明确 reason 落盘 (不再静默 "")
        """
        f = (formula or "").strip()
        if not f:
            return "", "空公式"

        # ── 路径 1: Forge 前缀风格 → 表达式树翻译 (带终端白名单校验) ──
        try:
            from forge.expression import parse_expression, EXTRA_TERMINALS
            # v0.6.1 (2026-08-29): 注册 S5 已知终端到 Forge 解析器 —
            # 修复 lhb_flag/pe_ttm 等字段 parse_expression 返回 None
            # → seed 重检翻译失败跳过 (15/58 种子此前被跳过)
            EXTRA_TERMINALS |= _S5_KNOWN_TERMINALS
            # v0.9: 内联负号 -ts_min(x,5) → neg(ts_min(x,5)) (仅解析用)
            tree = parse_expression(_canonicalize_neg_inline(f))
            if tree is not None:
                terms = [t for t in tree.terminals() if not _is_numeric_const(t)]
                unknown = [t for t in terms if t not in _S5_KNOWN_TERMINALS]
                if unknown:
                    return "", f"Forge 未知终端: {', '.join(sorted(set(unknown)))}"
                return tree.to_pandas_string(), ""
        except Exception:
            pass

        # v0.6.1 (2026-08-29): Forge 失败 → 先尝试 DSL 转换器
        # (FactorExpressionParser 覆盖 ts_corr/ts_delay/ts_rank 等 Forge 无的
        # 原语与混合风格公式), 再报 Forge 解析失败原因。原逻辑对 ts_corr 类
        # 种子直接提前 return, 15 个种子永远翻译失败。
        try:
            from factor_expression_tree import dsl_to_pandas_infix
            dsl_converted = dsl_to_pandas_infix(f)
            if dsl_converted and dsl_converted != f:
                unknown_dsl = _find_unknown_identifiers(dsl_converted)
                if not unknown_dsl:
                    return dsl_converted, ""
        except Exception:
            pass

        # v0.9: Forge 风格但解析失败 → 定位具体原因 (未知终端/不支持语法)
        forge_reason = _forge_parse_failure_reason(f)
        if forge_reason:
            return "", forge_reason

        # ── 路径 2: pandas/中缀风格 → 规范化 + AST 标识符校验 ──
        normalized = _normalize_legacy_rolling(f)
        # v0.6.1 (2026-08-29): 再尝试 DSL 转换器 (含 legacy rolling 规范化后)
        try:
            from factor_expression_tree import dsl_to_pandas_infix
            dsl_converted = dsl_to_pandas_infix(normalized)
            if dsl_converted and dsl_converted != normalized:
                normalized = dsl_converted
        except Exception:
            pass
        unknown = _find_unknown_identifiers(normalized)
        if unknown:
            if unknown == ["<语法错误>"]:
                return "", "无法解析 (非 Forge 前缀亦非合法 Python 表达式)"
            return "", f"未知标识符: {', '.join(unknown)}"
        return normalized, ""

    def _note_seed_translation(self, factor_name: str, ok: bool, reason: str) -> None:
        """v0.9: 翻译失败原因落盘 seed_recheck_state (供诊断, 不阻塞)。"""
        p = self._seed_recheck_state_path()
        state = self._load_seed_recheck_state()
        st = state.setdefault(factor_name, {})
        st["translate"] = {"ok": bool(ok), "reason": reason, "at": datetime.now().isoformat()}
        try:
            json.dump(state, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _seed_recheck_state_path(self) -> Path:
        return Path(__file__).parent.parent.parent / "data" / "seed_recheck_state.json"

    def _load_seed_recheck_state(self) -> Dict:
        p = Path(self._seed_recheck_state_path())
        if p.exists():
            try:
                return json.load(open(p, encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def record_seed_recheck(
        self,
        factor_name: str,
        s5_result: Optional[Dict] = None,
        jq_single_verified: bool = False,
    ) -> None:
        """记录种子重检结果 (S5 结论 + 单因子 JQ 验证状态)。

        v0.9: 合并式更新 (setdefault) — 不覆盖 translate/jq_code 等既有键。
        """
        p = self._seed_recheck_state_path()
        state = self._load_seed_recheck_state()
        st = state.setdefault(factor_name, {})
        st["last_check"] = datetime.now().isoformat()
        st["s5_result"] = s5_result or {}
        st["jq_single_verified"] = bool(jq_single_verified)
        try:
            json.dump(state, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    def record_seed_jq_code(
        self,
        factor_name: str,
        jq_file: str,
        generated: bool,
        reason: str = "",
    ) -> None:
        """v0.9: 记录种子单因子 JQ 代码自动生成状态 (落盘 seed_recheck_state)。"""
        p = self._seed_recheck_state_path()
        state = self._load_seed_recheck_state()
        st = state.setdefault(factor_name, {})
        st["jq_code"] = {
            "file": jq_file,
            "generated": bool(generated),
            "reason": reason,
            "at": datetime.now().isoformat(),
        }
        try:
            json.dump(state, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ── v3.1: FactorForge GP 生成 ──────────────────────

    def _generate_via_forge(
        self,
        pop_size: int = 200,
        n_generations: int = 5,
        max_candidates: int = 10,
    ) -> List[Dict]:
        """
        使用 FactorForge 引擎进行 GP 进化生成候选因子。

        FactorForge 使用 grow/full 随机树初始化 + 锦标赛选择 + 交叉/变异，
        与当前 GPBreeder（基于模板变异）互补：Forge 探索全新结构，Breeder 微调已有模式。

        v0.7 P-025 固化 (2026-08-13):
          - pop=200 / gens=5 / depth=6 (Round1c 实测 S5 通过率 22% vs gp_breed 10.4%)
          - FSA 频繁子树规避 (外部冻结 + 运行内集中度冻结)
          - 成熟范式定向 (forge.paradigm_profiles.MATURE_PARADIGM_PROFILES, 8 范式)
          - 质量门: 常数/低基数/价格-规模水平代理检测已内置于 FactorForge
          - 候选标记 experimental+forge_round1, S5 不通过即回退 GP Breed
        """
        # 延迟初始化 FactorForge
        if self.forge is None:
            print("  [Forge] 初始化 FactorForge 引擎...")
            # 尝试加载数据
            data = self._load_forge_data()
            if data is None:
                print("  [Forge] 无可用数据，回退到 GP 育种")
                return []

            self.forge = FactorForge(
                data=data,
                max_depth=6,              # v0.7 P-025: 5 → 6
                max_complexity=12.0,
                icir_threshold=0.1,
                seed=None,                # v0.6.1 (2026-08-29): 修复 seed=42 固定种子
                                          # → 4 轮产出逐字相同个体 (fitness 0.3391 重复)
                forward_returns=data.get("fwd_return"),  # P-022: 前向收益仅用于 ICIR, 不暴露为终端
                fsa=self.fsa,             # v0.7 P-025: FSA 接入
                fsa_retry=3,
                paradigm_profiles=MATURE_PARADIGM_PROFILES,  # v0.7 P-025: 成熟范式定向
                penalizer=self.sub_penalizer,  # P-007: 高频子结构软拒绝
                edit_memory=self.edit_memory,  # P-20260827-001: SSPM 编辑记忆 (连续 fitness 残差)
            )

        # 运行进化
        try:
            best_all = self.forge.evolve(
                n_generations=n_generations,
                pop_size=pop_size,
                tournament_size=3,
                elite_count=2,
                mutation_rate=0.3,
                crossover_rate=0.5,
                verbose=True,
            )
        except Exception as e:
            print(f"  [Forge] 进化异常: {e}")
            return []

        # 转换为标准候选格式 (v3.2: 使用 pandas_direct 表达式供 fri.py 执行)
        candidates = []
        seen_exprs = set()
        for b in best_all:
            expr = b["expression"]
            pandas_expr = b.get("pandas_expression", expr)  # v3.2: Forge→pandas 翻译
            if expr in seen_exprs:
                continue
            seen_exprs.add(expr)
            # v0.7 P3b: 方向归一化 (2026-08-29)
            # Forge fitness=|ICIR| 双向搜索 (合理: neg 原语存在, 负方向表达
            # 取负即正 alpha)。但下游 S1/S5/JQ/入库全链路不消费 direction 字段
            # (S5 validate_factor 无方向参数, 负 ICIR 公式原样回测必灭) —
            # R3 最佳 ICIR=-0.890 原方向进 S5 全灭即此坑。
            # 修复: 输出候选时对负 ICIR 因子统一取负翻转, 全链路 direction="+"。
            # 代价 vs max(icir,0): 保留双向搜索空间 (不砍负方向表达, 免 GP
            # 额外演化 neg 节点), 且一处修复覆盖所有下游。
            icir_out = float(b["icir"])
            direction = "+"
            flipped = False
            if icir_out < 0:
                pandas_expr = f"-({pandas_expr})"
                icir_out = -icir_out
                direction = "+"
                flipped = True
            candidates.append({
                "factor_name": f"forge_gen{b['generation']}_{b['name'][:40]}",
                "formula": pandas_expr,           # fri.py 可执行的 pandas 表达式
                "expression": expr,               # Forge 原始表达式 (日志用)
                "hypothesis": f"Forge GP Gen{b['generation']} | ICIR={icir_out:+.3f} | nodes={b['size']}",
                "logic": (f"Forge GP 进化生成 (gen={b['generation']}, fitness={b['fitness']:.4f}, "
                          f"raw_icir={b['icir']:+.3f}" + (", 方向归一化: 已取负" if flipped else "") + ")"),
                "direction": direction,
                "source": "factorforge",
                "generation": b["generation"],
                "icir": icir_out,
                # v0.7 频率对称: Forge 在周频数据上内生演化 (weekly_prices.parquet),
                # fitness=周频 ICIR → natural_freq=weekly, 保真原周频 ICIR 供 weekly lane S1。
                # V07 关闭时此标签不参与裁决 (daily lane 现状路径, 零回归)。
                "natural_freq": "weekly",
                "weekly_icir": icir_out,
                # v0.7 P-025: 实验标记 — S5 不通过即回退 (rollback_plan)
                "tags": ["experimental", "forge_round1"],
                "paradigm": b.get("paradigm"),
            })

        # 取 Top-N
        candidates = candidates[:max_candidates]
        print(f"  [Forge] 产生 {len(candidates)} 个候选 (过滤重复后)")
        return candidates

    def _load_forge_data(self) -> Optional[Dict]:
        """加载 FactorForge 所需的 OHLCV 数据

        自动扫描多个可能的 parquet/csv 数据源，构建 FactorForge 期望的
        {'open','high','low','close','volume','returns'} (T,N) 格式。
        缺失的 OHLC 字段用 close 作为合理代理。
        """
        try:
            import pandas as pd
            import numpy as np

            # ── 路径扫描：parquet → csv → pickle ──
            # 所有相对路径基于项目根目录 (data_dir 的父目录) 解析，避免 cwd 依赖
            project_root = Path(str(self.data_dir)).parent
            paths_to_try = [
                # 实际存在的 weekly_prices.parquet (relative → project_root)
                project_root / "output" / "ap_batch" / "cache" / "weekly_prices.parquet",
                # 备选路径 (data_dir 下的)
                Path(str(self.data_dir)) / "processed" / "factors_weekly_forward.parquet",
                Path(str(self.data_dir)) / "cache" / "factors_weekly_forward.parquet",
                # 项目内其他可能位置
                project_root / "research" / "factor_alchemy" / "data" / "cache" / "factors_weekly_forward.parquet",
                # CSV 备选 (大文件跳过 schema 推断)
                Path(str(self.data_dir)) / "cache" / "weekly_prices.csv",
                project_root / "output" / "ap_batch" / "cache" / "weekly_prices.csv",
            ]

            df = None
            loaded_path = None
            for p in paths_to_try:
                abs_p = p if p.is_absolute() else project_root / p
                if abs_p.exists():
                    try:
                        if abs_p.suffix == '.parquet':
                            df = pd.read_parquet(abs_p)
                        elif abs_p.suffix == '.csv':
                            df = pd.read_csv(abs_p, low_memory=False)
                        loaded_path = str(abs_p)
                        break
                    except Exception:
                        continue

            if df is None:
                print("  [Forge] 未找到任何可用数据文件，回退到 GP 育种")
                return None

            print(f"  [Forge] 数据源: {loaded_path}, shape={df.shape}")

            # ── 列名标准化 ──
            # weekly_prices.parquet 实际列: stock_code, trade_date, close, vol, amount, fwd_ret, ...
            df = df.copy()

            # v0.7 P3c (2026-08-29): 接入真实周频 OHLC (weekly_ohlc.parquet, 从日频
            # 重建: open=周首日 open, high=周内 max, low=周内 min)。
            # 此前 open/high/low 全部 close 代理 (假终端) → GP 演化出的 OHLC 因子
            # 翻译到 JQ (真实 OHLC) 后行为漂移 (第六个口径错配源)。
            # 无该文件时自动回退 close 代理 (向后兼容)。
            _ohlc_path = project_root / "output" / "ap_batch" / "cache" / "weekly_ohlc.parquet"
            if _ohlc_path.exists() and "stock_code" in df.columns and "trade_date" in df.columns:
                try:
                    _ohlc = pd.read_parquet(_ohlc_path)
                    _ohlc["trade_date"] = pd.to_datetime(
                        _ohlc["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True),
                        format="mixed", errors="coerce")
                    df["trade_date"] = pd.to_datetime(
                        df["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True),
                        format="mixed", errors="coerce")
                    # 防御: drop 与主表重叠的列 (close 等), 避免 merge 产生 close_x/close_y
                    _drop = [c for c in _ohlc.columns
                             if c in df.columns and c not in ("stock_code", "trade_date")]
                    if _drop:
                        _ohlc = _ohlc.drop(columns=_drop)
                    df = df.merge(_ohlc, on=["stock_code", "trade_date"], how="left")
                    _cov = df["open"].notna().mean() if "open" in df.columns else 0.0
                    print(f"  [Forge] 真实周频 OHLC 已接入 (open 覆盖率 {_cov:.3f})")
                except Exception as _e:
                    print(f"  [Forge] weekly_ohlc 接入失败, 回退 close 代理: {_e}")

            col_map = {
                "stock_code": "stock", "trade_date": "date",
                "vol": "volume", "amount": "amount",
                "fwd_ret": "forward_return",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

            # ── 确保 date 列存在 ──
            if "date" not in df.columns:
                if df.index.name in ("date", "trade_date"):
                    df = df.reset_index()
                else:
                    print("  [Forge] 缺少日期列，回退到 GP 育种")
                    return None

            df["date"] = pd.to_datetime(df["date"])

            # ── 构建 OHLCV 矩阵 (T, N) ──
            data = {}
            # 主字段映射: {forge_key: (df_col, fallback_col)}
            field_specs = {
                "close":  ("close", None),
                "volume": ("volume", None),
                "open":   ("open", "close"),    # 无 open → close 代理
                "high":   ("high", "close"),    # 无 high → close 代理
                "low":    ("low", "close"),     # 无 low → close 代理
                # v0.7 P-025: vwap 终端用 volume 代理
                # (与 to_pandas_string 的 vwap→volume_p 语义一致; 旧实现缺 vwap
                #  导致 vwap 终端全 NaN — Forge 中一直是死终端)
                "vwap":   ("volume", None),
            }

            pivot_cache = {}
            for fname, (primary, fallback) in field_specs.items():
                col = primary if primary in df.columns else fallback
                if col is None or col not in df.columns:
                    if fname in ("close", "volume"):
                        # close/volume 是必须的
                        print(f"  [Forge] 缺少必要字段 '{fname}'，回退到 GP 育种")
                        return None
                    continue

                if col not in pivot_cache:
                    pivoted = df.pivot_table(
                        index="date", columns="stock", values=col, aggfunc="last"
                    )
                    pivot_cache[col] = pivoted
                data[fname] = pivot_cache[col].values.astype(float)

            # ── 前向收益 (仅用于 ICIR target, 不作为 Forge 终端变量) ──
            fwd_col = None
            for c in ("forward_return", "forward_return_1w", "fwd_ret"):
                if c in df.columns:
                    fwd_col = c
                    break
            if fwd_col:
                pivoted = df.pivot_table(
                    index="date", columns="stock", values=fwd_col, aggfunc="last"
                )
                data["fwd_return"] = pivoted.values.astype(float)
                # ⚠️ P-022: data["returns"] 必须是历史收益, 不能是 fwd_ret!
                # 旧代码 data["returns"] = fwd_ret 导致 GP 进化出 scale(returns)
                # → ICIR(fwd_ret, fwd_ret) ≈ 1.0 → 虚高 +1000
                # Forge 终端变量用历史 returns (close.pct_change), 目标用 fwd_return

            # ── 历史收益 (Forge 终端变量 — GP 可以用的特征) ──
            if "close" in data:
                close_arr = data["close"]  # (T, N)
                # hist_ret[t] = (close[t] - close[t-1]) / close[t-1], 首行 NaN
                hist_ret = np.full_like(close_arr, np.nan)
                if close_arr.shape[0] > 1:
                    hist_ret[1:] = (close_arr[1:] - close_arr[:-1]) / np.where(
                        close_arr[:-1] != 0, close_arr[:-1], np.nan
                    )
                data["returns"] = hist_ret

            if data:
                close_shape = data.get("close", np.zeros((1, 1))).shape
                n_fields = len(data)
                has_fwd = "fwd_return" in data
                print(f"  [Forge] 数据加载成功: {n_fields} fields (fwd_ret={'✓' if has_fwd else '✗'}), "
                      f"(T,N)=({close_shape[0]},{close_shape[1]})")
                return data
            else:
                print("  [Forge] 数据解析后为空，回退到 GP 育种")

        except Exception as e:
            print(f"  [Forge] 数据加载异常: {e}")
            import traceback
            traceback.print_exc()
        return None

    def _cross_breed_from_templates(
        self,
        templates: List,
        paradigm: str,
        max_candidates: int,
    ) -> List[Dict]:
        """从成功模板中交叉育种生成新候选"""
        candidates = []
        for t in templates[:max_candidates]:
            cand = {
                "factor_name": f"cross_{t.pattern_id.replace('::', '_')}",
                "hypothesis": f"基于成功模板 '{t.description}' 的交叉育种",
                "paradigm": paradigm or t.pattern_id.split("::")[0],
                "operators_used": t.typical_operators,
                "ic": t.ic_range[0],  # 保守估计
                "icir": t.icir_range[0],
                "source": "cross_breed",
                "parent_template": t.pattern_id,
            }
            candidates.append(cand)
        return candidates

    # ═══════════════════════════════════════════════════════════
    # v0.6: GP 多轮进化 (EvoTraj) — 对标 AlphaAgentEvo
    # ═══════════════════════════════════════════════════════════

    def _phase_generate_evo(
        self,
        templates: List[Dict],
        retrieve_result: Dict,
        paradigm: str = "",
        max_candidates: int = 10,
        max_turns: int = 5,
        use_fsa: bool = True,
        motif_filter: bool = True,
    ) -> tuple:
        """
        G 阶段增强: GP 多轮进化轨迹 (EvoTraj v0.6)。

        对标 AlphaAgentEvo (ICLR 2026) 的连续进化思想:
          - 传统: seed → GP breed(单次) → best 1
          - EvoTraj: seed → GP mutate → S1 eval → best → GP mutate → ... → S5
                      └────────────── T 轮 ──────────────┘
                                    ↓
                            轨迹蒸馏 → MAB streak 奖励

        Parameters
        ----------
        templates: 种子因子模板 (from Memory in Retrieve phase)
        retrieve_result: Memory retrieval 结果 (含 motif 约束)
        paradigm: 当前探索范式
        max_candidates: 每轮 GP 育种子代数
        max_turns: 最大进化轮数
        use_fsa: 启用 FSA 过滤
        motif_filter: 启用 P-001 motif 级过滤

        Returns
        -------
        (source, gen_candidates, trajectory) — 轨迹用于后续 MAB 奖励更新
        """
        all_candidates = []
        trajectories = []

        # P-20260819-005: 轨迹父本池注入 (AlphaSeek 对齐, 开关默认关)
        # 成功轨迹按命中率加权分层采样作父本, 与模板种子合并进化
        traj_parents = []
        try:
            from trajectory_pool import ENABLE_TRAJECTORY_POOL, get_trajectory_parents
            if ENABLE_TRAJECTORY_POOL:
                traj_parents = get_trajectory_parents(
                    data_dir=self.data_dir, n=min(3, max(1, max_candidates // 3)),
                    paradigm=paradigm)
                if traj_parents:
                    print(f"  [TrajPool] 轨迹父本池注入 {len(traj_parents)} 条 "
                          f"(命中率加权, 30天冷却)")
        except Exception as e:
            print(f"  [TrajPool] 注入失败(不影响主线): {e}")

        # v0.6: 模板种子不足时, 降级为普通 GP breed
        if len(templates) < 2:
            print(f"  [EvoTraj] 种子模板不足 ({len(templates)}), 降级为普通 gp_breed")
            converted = []
            for t in templates:
                if isinstance(t, dict):
                    converted.append({"formula": t.get("formula", t.get("expression", "")), 
                                     "factor_name": t.get("factor_name", t.get("pattern_id", ""))})
                elif hasattr(t, 'formula'):
                    converted.append({"formula": getattr(t, 'formula', ''), 
                                     "factor_name": getattr(t, 'pattern_id', 'unknown')})
            # P-20260819-005: 轨迹父本并入降级育种 (成功轨迹知识继承)
            for tp in traj_parents:
                converted.append({"formula": tp["formula"],
                                  "factor_name": f"traj_{tp['factor_name']}"})
            fsa_for_breed = self.fsa if use_fsa else None
            all_candidates = self.breeder.breed_from_templates(
                templates=converted, n_children=max_candidates,
                fsa=fsa_for_breed, output_format="pandas",
            )
            return ("gp_evolve_fallback", all_candidates)

        # ── 主循环: 多轮进化 ─────────────────────────────
        # P-20260819-005 v2 (2026-08-19 试跑修正):
        # (a) jq_confirmed 类模板无 formula 字段 → 从 Memory attempts 按
        #     sample_factor_ids 解析真实公式 (否则空种子第1轮即无候选产出)
        # (b) 轨迹父本强制占 1 个 seed 位 (原补位条件 len<3 几乎不触发, 虚注入)
        def _resolve_template_formula(tmpl) -> str:
            if isinstance(tmpl, dict):
                f = (tmpl.get("formula") or tmpl.get("expression") or "").strip()
                if f:
                    return f
                for fid in (tmpl.get("sample_factor_ids") or []):
                    for a in self.memory.data.get("attempts", []):
                        if isinstance(a, dict) and a.get("factor_name") == fid:
                            _f = (a.get("formula") or a.get("expression") or "").strip()
                            if _f:
                                tmpl["formula"] = _f  # 就地写回 (种子循环读 formula 键)
                                return _f
            elif hasattr(tmpl, 'formula'):
                return (getattr(tmpl, 'formula', '') or '').strip()
            return ""

        n_seeds = min(3, len(templates))
        seed_templates = []
        for t in templates[:n_seeds + 2]:
            if _resolve_template_formula(t):
                seed_templates.append(t)
            else:
                _nm = t.get('pattern_id', '?') if isinstance(t, dict) else getattr(t, 'pattern_id', '?')
                print(f"  [EvoTraj] 模板 {_nm} 公式解析失败(空formula且attempts无记录), 跳过种子")
            if len(seed_templates) >= 3:
                break
        # 轨迹父本强制混入: 保留前2个模板种子 + 1条轨迹父本
        if traj_parents and len(seed_templates) >= 2:
            seed_templates = seed_templates[:2] + [{
                "formula": traj_parents[0]["formula"],
                "factor_name": f"traj_{traj_parents[0]['factor_name']}",
                "_from_trajpool": True,
            }]
            print(f"  [EvoTraj] 轨迹父本强制入 seed 位: traj_{traj_parents[0]['factor_name'][:36]}")
        n_seeds = len(seed_templates)
        print(f"  [EvoTraj] 启动多轮进化: {n_seeds} seeds × {max_turns} turns")

        # 2026-08-19 性能修复: FactorICComputer 每实例 _load_data 读全市场 csv (数分钟),
        # 原实现每 turn 新建实例 → 3 seeds × 5 turns × 重复加载 = 数十分钟级。
        # 改为 seed 循环外创建一次, 所有 turn/seed 复用同一加载缓存。
        _shared_ic_comp = None

        for seed_idx in range(n_seeds):
            seed = seed_templates[seed_idx]
            # 兼容 dict 和 object 两种类型
            if isinstance(seed, dict):
                seed_formula = seed.get("formula", seed.get("expression", ""))
                seed_name = seed.get("factor_name", seed.get("pattern_id", f"seed_{seed_idx}"))
            elif hasattr(seed, 'formula'):
                seed_formula = getattr(seed, 'formula', getattr(seed, 'expression', ''))
                seed_name = getattr(seed, 'pattern_id', f"seed_{seed_idx}")
            else:
                continue

            # 创建轨迹对象
            traj = create_trajectory(
                seed_factor_name=seed_name,
                seed_formula=seed_formula,
                paradigm=paradigm,
                max_turns=max_turns,
            )
            print(f"  [EvoTraj] Seed#{seed_idx+1}: {seed_name[:40]} → {max_turns}轮精炼")

            # 当前最佳 = seed
            current_templates = [{"formula": seed_formula, "factor_name": seed_name}]

            for turn in range(max_turns):
                # 准备 motif 约束 (P-001 集成)
                motif_avoid_set = set()
                motif_prefer_set = set()
                if motif_filter:
                    try:
                        from experience_memory import get_memory
                        mem = get_memory()
                        forbidden = mem.get_motif_forbidden(min_samples=3, max_rate=0.1)
                        motif_avoid_set = set(forbidden)
                        # 成功 motif: 仅当有 JQ 验证数据才用于偏好
                        # (初期无 JQ 数据, prefer 为空)
                    except Exception:
                        pass

                # GP 育种 — v0.6: 使用 breed_from_single_seed (连续变异)
                fsa_for_breed = self.fsa if use_fsa else None
                try:
                    children = self.breeder.breed_from_single_seed(
                        formula=current_templates[0]["formula"],
                        n_children=max_candidates,
                        fsa=fsa_for_breed,
                        output_format="pandas",
                        motif_avoid=motif_avoid_set if motif_filter else None,
                        motif_prefer=motif_prefer_set if motif_filter else None,
                    )
                except Exception as e:
                    print(f"    [EvoTraj] GP breed 第{turn+1}轮出错: {e}")
                    break

                if not children:
                    print(f"    [EvoTraj] 第{turn+1}轮无候选产出, 停止")
                    break

                # 门禁过滤
                n_pass_gate = 0
                valid_children = []
                for child in children:
                    gate_result = self.gate.verify(child)
                    child["_gate_result"] = gate_result
                    if gate_result.passed:
                        valid_children.append(child)
                        n_pass_gate += 1
                    elif motif_avoid_set:
                        # 门禁不过的, 不再另行 motif 检查
                        pass

                # 对通过门禁的候选做 FSA 过滤
                fsa_filtered = []
                if use_fsa and valid_children:
                    for c in valid_children:
                        formula = c.get("formula", c.get("expression", ""))
                        if formula and not self.fsa.check_expression(formula):
                            fsa_filtered.append(c)

                if not fsa_filtered and valid_children:
                    fsa_filtered = valid_children

                if not fsa_filtered:
                    print(f"    [EvoTraj] 第{turn+1}轮全部被过滤 (gate={n_pass_gate}/{len(children)})")
                    traj.add_turn(
                        factor_name=seed_name, formula=seed_formula,
                        icir=0.0, calmar=0.0,
                        n_children=len(children), n_pass_gate=n_pass_gate,
                    )
                    break

                # S1 快速评估 — FactorICComputer 实时计算 IC/ICIR
                # (2026-08-19 修复: 原 validate(stages=[1]) 参数不存在→TypeError,
                #  降级 _compute_ic 亦不存在→评估恒空; 且变异子代无预计算 IC,
                #  stage1_fast_ic 会全拒。改为主线 _phase_evaluate 同款实时计算)
                eval_results = []
                try:
                    from factor_ic_computer import FactorICComputer
                    from multi_stage_validator import ValidationResult
                    if _shared_ic_comp is None:
                        _shared_ic_comp = FactorICComputer()  # 首次加载 (全市场 csv, 慢)
                    _ic_comp = _shared_ic_comp
                    for _ci, c in enumerate(fsa_filtered):
                        _f = c.get("formula", c.get("expression", ""))
                        if not _f:
                            continue
                        _r = _ic_comp.compute(_f, c.get("factor_name", f"evotraj_{seed_idx}_{turn}"))
                        if _r.get("n_days", 0) > 0:
                            eval_results.append(ValidationResult(
                                factor_name=c.get("factor_name", f"evotraj_{seed_idx}_{turn}"),
                                formula=_f,
                                s1_ic=_r.get("ic", 0.0),
                                s1_icir=_r.get("icir", 0.0),
                            ))
                except Exception as _e:
                    print(f"    [EvoTraj] IC 实时计算失败: {_e}")

                if not eval_results:
                    print(f"    [EvoTraj] 第{turn+1}轮评估无结果")
                    traj.add_turn(
                        factor_name=seed_name, formula=seed_formula,
                        icir=0.0, calmar=0.0,
                        n_children=len(children), n_pass_gate=n_pass_gate,
                    )
                    break

                # 选最佳 (2026-08-19 修复: ValidationResult 无 icir/calmar 字段,
                # 实时计算值落在 s1_icir/s1_ic; 原 hasattr 检查恒为 0 → 首个子代胜出)
                def _vr_icir(r):
                    v = getattr(r, 's1_icir', 0.0) or 0.0
                    if v == 0.0:
                        v = getattr(r, 'icir', 0.0) or 0.0
                    return v

                # P-20260826-005: 分层奖励亲本选择 (L2 稳健 ICIR 主导 + L3 稀缺调制)
                def _vr_layered(r):
                    base = _vr_icir(r)
                    f = getattr(r, 'formula', '') or ''
                    pen = 0.0
                    pz = getattr(self.breeder, 'penalizer', None)
                    if pz is not None and f:
                        try:
                            pen = float(pz.penalty_for(f))
                        except Exception:
                            pen = 0.0
                    return base * (1.0 - 0.35 * pen)

                best_result = max(eval_results, key=_vr_layered)
                best_icir = _vr_icir(best_result)
                best_calmar = getattr(best_result, 's5_calmar', 0) or 0
                best_formula = getattr(best_result, 'formula', '') or ''
                best_name = getattr(best_result, 'factor_name', f'evotraj_{seed_idx}_t{turn}')

                # v0.6: 提取 motif
                child_motifs = []
                if use_fsa and best_formula:
                    try:
                        fps = self.fsa.extract_fingerprints(best_formula)
                        child_motifs = [fp[:80] for fp in fps[:5]]  # 限制长度
                    except Exception:
                        pass

                # 记录轮次
                traj.add_turn(
                    factor_name=best_name,
                    formula=best_formula,
                    icir=best_icir,
                    calmar=best_calmar,
                    n_children=len(children),
                    n_pass_gate=n_pass_gate,
                    motifs=child_motifs,
                )

                improvement = ""
                if turn > 0 and traj.turns[-2].icir > 0:
                    delta = best_icir - traj.turns[-2].icir
                    improvement = f" Δ={delta:+.4f}"

                print(f"    [EvoTraj] T{turn+1}: {best_name[:25]} ICIR={best_icir:.4f}"
                      f" (gate={n_pass_gate}/{len(children)}) streak={traj.streak}{improvement}")

                # 将该轮的 factor 加入全局候选池
                for c in fsa_filtered:
                    c["_source"] = "gp_evolve"
                    c["_paradigm"] = paradigm
                    c["_traj_id"] = traj.traj_id
                    c["_memory_priors"] = retrieve_result.get("llm_prompt_context", "")[:1000]
                    all_candidates.append(c)

                # 准备下一轮: 用当前最佳作为种子
                current_templates = [{"formula": best_formula, "factor_name": best_name}]

                # 停止条件
                if traj.should_stop():
                    reason = f"max_turns={max_turns}" if traj.current_turn >= max_turns else f"dead_streak={traj.dead_streak}"
                    print(f"    [EvoTraj] 停止: {reason}")
                    break

            # 轨迹完成
            traj.finish()
            trajectories.append(traj)
            print(f"    [EvoTraj] Seed#{seed_idx+1} 完成: streak={traj.streak}, "
                  f"best_ICIR={traj.best_icir:.4f}, reward={traj.trajectory_reward:.3f}")

        # ── 去重 ─────────────────────────────────────────
        seen_formulas = set()
        deduped = []
        for c in all_candidates:
            formula = c.get("formula", c.get("expression", ""))
            if formula and formula not in seen_formulas:
                seen_formulas.add(formula)
                deduped.append(c)
        all_candidates = deduped

        print(f"  [EvoTraj] 总计: {len(trajectories)}条轨迹, {len(all_candidates)}候选")

        # 将 trajectories 附在第一个候选上 (供后续 MAB 奖励更新)
        if all_candidates:
            all_candidates[0]["_evo_trajectories"] = trajectories

        return ("gp_evolve", all_candidates)

    # ═══════════════════════════════════════════════════════════
    # v0.6: EvoTraj 辅助方法
    # ═══════════════════════════════════════════════════════════

    def _extract_trajectories(self, generate_result: Dict) -> List[Dict]:
        """从 G 阶段结果提取轨迹摘要 (供外部使用)"""
        trajectories = []
        candidates = generate_result.get("candidates", [])
        # 轨迹附带在第一个候选上
        if candidates and candidates[0].get("_evo_trajectories"):
            for traj in candidates[0]["_evo_trajectories"]:
                trajectories.append(traj.get_summary())
        return trajectories

    # ═══════════════════════════════════════════════════════════
    # v0.5: MAB 探索方向调度
    # ═══════════════════════════════════════════════════════════

    def _mab_select_direction(self) -> Optional[ResearchDirection]:
        """使用 MAB UCB1 选择本轮探索的范式方向"""
        if not self._mab_loaded:
            try:
                self.mab.load_state()
            except Exception:
                pass
            self._mab_loaded = True

        directions = list(self.mab.directions.values()) if isinstance(self.mab.directions, dict) else self.mab.directions
        if not directions:
            print("  [MAB] 无可用方向, 使用默认")
            return None

        total_pulls = sum(d.pulls for d in directions)

        # v0.6: 强制探索多样性 — 同一范式连续>2次后强制轮转
        if not hasattr(self, '_last_paradigm'):
            self._last_paradigm = None
            self._paradigm_streak = 0

        best = None
        best_score = -float("inf")

        for d in directions:
            if d.status == "cooling":
                continue
            if d.pulls == 0:
                # 未探索方向优先
                best = d
                break
            # UCB1
            avg = d.expected_reward

            # v0.6: MAB 冷启动修复 L1 — Memory motif 先验
            # 有成功模式的范式获得微幅正信号 (0.03)，不主导选择，仅打破全零困局
            memory_prior = 0.0
            try:
                from experience_memory import get_memory as _get_mem
                _mem = _get_mem()
                _ret = _mem.retrieve(k=3)
                for _pat in _ret.get("success_templates", []):
                    _pp = getattr(_pat, "paradigm", "")
                    if _pp and _pp in str(d.paradigm):
                        memory_prior = 0.03
                        break
            except Exception:
                pass

            # v0.7 Phase 2: 轨迹蒸馏先验 — 从历史失败分析中提炼范式偏好
            traj_prior = 0.0
            if self._distillation_hints:
                prefer = {p: ic for p, ic in self._distillation_hints.get("prefer_paradigms", [])}
                avoid = set(self._distillation_hints.get("avoid_paradigms", []))
                if d.paradigm in prefer:
                    traj_prior = min(0.08, prefer[d.paradigm] * 4)  # IC→先验, 上限0.08
                elif d.paradigm in avoid:
                    traj_prior = -0.05  # 微幅负信号, 不主导但倾向避开

            # v3.1: WarningDirection 降权 — JQ 软负收益因子 → MAB 负向先验
            warning_prior = 0.0
            try:
                mem = _get_mem()
                if mem:
                    warn_paras = mem.get_warning_paradigms()
                    n_warn = warn_paras.get(d.paradigm, 0)
                    if n_warn > 0:
                        warning_prior = -min(0.08, n_warn * 0.02)
            except Exception:
                pass

            avg = avg + memory_prior + traj_prior + warning_prior

            bonus = 2.0 * (np.log(max(total_pulls, 1)) / d.pulls) ** 0.5
            ucb = avg + bonus
            if ucb > best_score:
                best_score = ucb
                best = d

        if best:
            # v0.6: 强制轮转 — 基于 pulls 推断 (跨 run 持久化)
            # 找出 pulls 最小的 active 范式，如果当前方向 pulls 明显偏高则切换
            active = [d for d in directions if d.status == 'active' and d.paradigm]
            if active:
                min_pulls = min(d.pulls for d in active)
                # 如果当前方向 pulls 比最小多 >2，说明已被过度探索 → 切换
                if best.pulls > min_pulls + 2:
                    underexplored = [d for d in active if d.pulls <= min_pulls + 1]
                    if underexplored:
                        underexplored.sort(key=lambda d: d.pulls)
                        old_para = best.paradigm
                        best = underexplored[0]
                        print(f"  [MAB] 🔄 轮转: {old_para}({best.pulls}p) → {best.paradigm}({best.pulls}p) [min_pulls={min_pulls}]")

            best.pulls += 1
            best.last_pulled = datetime.now().isoformat()
        return best

    def _mab_select_generator(self, direction: ResearchDirection) -> str:
        """v3.1: 为选定方向选择最优生成器 (基于历史表现)"""
        gen_rewards = getattr(direction, 'generator_rewards', {})
        if gen_rewards:
            # 有历史数据: 软最大化选择 (ε-greedy)
            import random
            if random.random() < 0.2:  # 20% 探索
                return random.choice(["gp_breed", "llm", "forge"])
            # 80% 利用: 选平均奖励最高的生成器
            best_gen = "gp_breed"
            best_avg = -float("inf")
            for gen, rewards in gen_rewards.items():
                if rewards:
                    avg = sum(rewards) / len(rewards)
                    if avg > best_avg:
                        best_avg = avg
                        best_gen = gen
            return best_gen
        else:
            # 无历史数据: 基于 direction.pulls 选择
            # 未探索方向 → LLM (创新); 已探索方向 → GP/Forge (精细化)
            pulls = getattr(direction, 'pulls', 0)
            if pulls <= 1:
                return "llm"  # 冷启动方向: LLM 生成新颖因子
            elif pulls <= 3:
                return "gp_breed"  # 中期: GP 育种
            else:
                return "forge"  # 成熟方向: Forge 完整进化

    def _mab_record_result(
        self, direction_name: str, jq_return: float, jq_sharpe: float,
        streak: int = 0,
    ):
        """
        MAB 记录结果并更新奖励 (v0.6: streak 增强)。

        对标 AlphaAgentEvo 的 streak 奖励:
          reward_base = (jq_return - 182.57) / 100.0  # vs 王者基准
          reward = max(-1, min(1, reward_base * 5 * (1 + 0.2 * streak)))

        连续改善的方向获得更多 MAB 资源倾斜。
        """
        dirs = list(self.mab.directions.values()) if isinstance(self.mab.directions, dict) else self.mab.directions
        for d in dirs:
            if d.name == direction_name or d.direction_id == direction_name:
                relative = (jq_return - 182.57) / 100.0  # vs 王者
                # v0.6: streak bonus
                streak_multiplier = 1.0 + 0.2 * streak
                reward = max(-1.0, min(1.0, relative * 5 * streak_multiplier))
                d.expected_reward = (d.expected_reward * d.pulls + reward) / (d.pulls + 1) if d.pulls > 0 else reward
                if jq_return < -20 or jq_sharpe < -0.5:
                    d.failures += 1
                    if d.failures >= 3:
                        d.status = "cooling"
                else:
                    d.successes += 1
                    d.jq_results.append({"return": jq_return, "sharpe": jq_sharpe, "streak": streak})
                self.mab.save_state()
                if streak > 0:
                    print(f"    [MAB] {d.name}: reward={reward:.3f} (streak_bonus={streak_multiplier:.1f}x)")
                break

    # ═══════════════════════════════════════════════════════════
    # Phase E: Evaluate
    # ═══════════════════════════════════════════════════════════

    def _phase_evaluate(
        self,
        candidates: List[Dict],
        library_signals: Optional[Dict],
        library_factors: Optional[Dict],
    ) -> Dict:
        """
        E 阶段: 多阶段验证 (v0.5.2: S2 库内相关性 + S5 轻量回退)。

        使用 MultiStageValidator 的五阶段管线:
        S1: 快速 IC 筛选（FactorICComputer 实时计算）
        S2: 库内相关性去重（v0.5.2: 与 Memory 模板做 Spearman 横截面相关）
        S3: 批次内去重
        S4: OOS + 替换检查
        S5 (v0.5.2): 联合正向过滤 + Calmar
            - 优先: S5JointFilter 真实回测
            - 回退: FactorICComputer 年拆分 IC 轻量过滤

        v0.5.2 变更:
        - S2 不再静默通过（max_corr 默认 0）。改为从 Memory 加载模板公式，
          用 FactorICComputer 计算横截面 Spearman 秩相关，填充真实 max_corr。
        - S5 不可用时不再默认失败。改用 IC 年拆分做轻量两年正向过滤，
          避免无 S5 时候选仍 eligible_for_jq（4/5 绕过联合过滤）。
        """
        # ── v0.5.1: 轻量级 IC 计算 (所有来源: LLM/GP/Forge) ──
        n_ic_computed = 0
        ic_comp = None
        # v0.7 频率对称: weekly lane 候选走周频裁决器 (XOR, 不重复日频计算)
        _weekly_judge = None
        try:
            from weekly_lane import get_weekly_judge
            _weekly_judge = get_weekly_judge()
        except Exception:
            _weekly_judge = None
        n_weekly_judged = 0
        try:
            from factor_ic_computer import FactorICComputer
            ic_comp = FactorICComputer()
            for cand in candidates:
                formula = cand.get("formula", cand.get("expression", ""))
                # ── v0.7: weekly lane 分频 (V07 开启时) ──
                if _weekly_judge is not None and cand.get("natural_freq") == "weekly":
                    _wj = _weekly_judge.judge(formula, cand.get("factor_name", ""))
                    if _wj.get("eval_ok"):
                        cand["weekly_ic"] = _wj["weekly_ic"]
                        cand["weekly_icir"] = _wj["weekly_icir"]
                        cand["weekly_n"] = _wj["weekly_n"]
                        cand["weekly_icir_recent"] = _wj["weekly_icir_recent"]
                        cand["_weekly_activity_ok"] = _wj["activity_ok"]
                        cand["_weekly_reason"] = _wj["reason"]
                        n_weekly_judged += 1
                    else:
                        cand["weekly_ic"] = 0.0
                        cand["weekly_icir"] = 0.0
                        cand["_weekly_reason"] = _wj.get("reason", "周频裁决失败")
                    # XOR: weekly 候选不再走日频 FactorICComputer (避免双重检验)
                    continue
                has_ic = abs(float(cand.get("ic", cand.get("daily_ic", 0)) or 0)) > 1e-8
                if not has_ic and formula:
                    # P-20260828-004: 同时取 IC 序列供 bootstrap CI 影子模式
                    # P-20260828-001: 同时取 lag1 IC → Δlag 滞后衰减影子模式
                    ic_result = ic_comp.compute(formula, cand.get("factor_name", ""),
                                                return_series=True, also_lag1=True)
                    if ic_result["n_days"] > 0:
                        cand["ic"] = ic_result["ic"]
                        cand["icir"] = ic_result["icir"]
                        cand["daily_ic"] = ic_result["ic"]
                        cand["daily_icir"] = ic_result["icir"]
                        cand["_ic_series"] = ic_result.get("ic_series", [])
                        cand["ic_lag1"] = ic_result.get("ic_lag1", 0.0)
                        cand["delta_lag"] = ic_result.get("delta_lag", 0.0)
                        n_ic_computed += 1
            if n_ic_computed > 0:
                print(f"  [伏羲] IC 实时计算: {n_ic_computed}/{len(candidates)} 候选 "
                      f"(max|IC|={max(abs(float(c.get('ic', 0)) or 0) for c in candidates):.4f})")
            if n_weekly_judged > 0:
                print(f"  [v0.7] 周频裁决器: {n_weekly_judged} 候选走 weekly lane "
                      f"(τ_w 门槛, 不重复日频裁决)")
        except Exception as e:
            print(f"  [伏羲] ⚠️ IC 计算失败: {e}，候选 S1 将被拒绝")

        # ── 2026-08-31: 子结构 fail% 先验影子 (P-007 → 五元组影子特征, 非否决) ──
        # 背景: 分离力分析证明本地周频 pandas_icir 对 JQ 成败无分离力 (p=0.139),
        #   子结构 fail% 是血统级 JQ 信号, 先以影子字段随候选落盘, 4-8 周后归因。
        try:
            from substructure_frequency import compute_fail_prior_from_library
            _pz_shadow = getattr(self.breeder, "penalizer", None)
            _n_ss_known = 0
            for cand in candidates:
                _f = cand.get("formula", "") or ""
                if not _f:
                    cand["ss_fail_prior"] = 0.5
                    cand["ss_coverage"] = 0.0
                    cand["ss_n_known"] = 0
                    cand["ss_n_high_fail"] = 0
                    continue
                try:
                    _sp = compute_fail_prior_from_library(_f, penalizer=_pz_shadow)
                except Exception:
                    _sp = {"prior_score": 0.5, "coverage": 0.0, "n_known": 0, "n_high_fail": 0}
                cand["ss_fail_prior"] = _sp["prior_score"]
                cand["ss_coverage"] = _sp["coverage"]
                cand["ss_n_known"] = _sp["n_known"]
                cand["ss_n_high_fail"] = _sp["n_high_fail"]
                if _sp["n_known"] > 0:
                    _n_ss_known += 1
            if _n_ss_known:
                _ss_top = sorted(
                    (c for c in candidates if c.get("ss_n_known")),
                    key=lambda c: -c["ss_fail_prior"])[:3]
                _ss_s = ", ".join(
                    "%s(%s=%.2f)" % (str(c.get("factor_name", "?"))[:22],
                                     "prior", c["ss_fail_prior"])
                    for c in _ss_top)
                print(f"  [SS-Prior] 子结构失败先验影子 ({_n_ss_known}/{len(candidates)}"
                      f" 有JQ标签) Top: {_ss_s}")
        except Exception as _spe:
            print(f"  [SS-Prior] 先验计算失败 (非阻塞): {_spe}")

        # ── v0.5.2: S2 库内相关性计算 ──
        # 从 Memory 加载模板公式作为"库"，对无 max_corr 的候选计算真实相关性
        n_corr_computed = 0
        n_corr_unavailable = 0
        try:
            if ic_comp is not None:

                # ── v0.5.3: 过滤不可执行公式 ──
                def _is_pandas_compatible(formula: str) -> bool:
                    """公式是否可在 pandas eval 环境执行 (排除 Forge 风格/虚构列名)"""
                    import re as _re
                    if not formula or formula.startswith("# TODO"):
                        return False
                    # Forge 函数调用模式 (ts_mean|ts_std|neg|rank|div|sub|...) 带括号
                    forge_patterns = [
                        r'\bts_mean\(', r'\bts_std\(', r'\bts_pct\(', r'\bts_delta\(',
                        r'\bts_zscore\(', r'\bts_corr\(', r'\bts_rank\(', r'\bts_skew\(',
                        r'\bts_kurtosis\(', r'\bts_argmax\(', r'\bts_argmin\(',
                        r'\bcs_rank\(', r'\bneg\(', r'\bscale\(', r'\breturns\b',
                        r'\brank\(', r'\bdiv\(', r'\bsub\(', r'\badd\(', r'\bmul\(',
                        r'\bif_else\(', r'\brolling_min\b', r'\brolling_max\b',
                        r'\bpow\(', r'\blog\(', r'\bsqrt\(', r'\babs\(', r'\bsigmoid\(',
                    ]
                    for pat in forge_patterns:
                        if _re.search(pat, formula):
                            return False
                    # 虚构列名 (不在 OHLCV/moneyflow 中)
                    # v0.5.4: buy_elg_vol/sell_elg_vol/buy_lg_vol 等已由
                    #   FactorICComputer 从 moneyflow_daily.csv merge 支持
                    fictional_cols = [
                        'north_money', 'south_money',
                        'rd_expense', 'rd_capitalized', 'rd_staff', 'revenue',
                        'goodwill', 'intan_assets', 'concept_label',
                    ]
                    for fc in fictional_cols:
                        if fc in formula:
                            return False
                    return True

                # 收集库因子公式（从 Memory success_templates）
                lib_factors = []
                seen_names = set()
                n_forge_skipped = 0
                templates_raw = self.memory.data.get("success_templates", [])
                if isinstance(templates_raw, dict):
                    templates_raw = list(templates_raw.values())
                for t in templates_raw:
                    if isinstance(t, dict):
                        name = t.get("pattern_id", t.get("factor_name", t.get("name", "")))
                        formula = t.get("formula", t.get("expression", ""))
                    elif hasattr(t, "formula"):
                        name = getattr(t, "pattern_id", getattr(t, "name", ""))
                        formula = getattr(t, "formula", getattr(t, "expression", ""))
                    else:
                        continue
                    if name and formula and name not in seen_names:
                        if _is_pandas_compatible(str(formula)):
                            seen_names.add(name)
                            lib_factors.append({"factor_name": str(name), "formula": str(formula)})
                        else:
                            n_forge_skipped += 1

                # 也为 Memory 中的 pattern_stats 提取公式
                pattern_stats = self.memory.data.get("pattern_stats", {})
                if isinstance(pattern_stats, dict):
                    for pid, pdata in pattern_stats.items():
                        if isinstance(pdata, dict):
                            pf = pdata.get("formula", pdata.get("expression", ""))
                            if pf and pid not in seen_names:
                                if _is_pandas_compatible(str(pf)):
                                    seen_names.add(pid)
                                    lib_factors.append({"factor_name": str(pid), "formula": str(pf)})
                                else:
                                    n_forge_skipped += 1

                if lib_factors:
                    forge_note = f" ({n_forge_skipped} Forge/虚构列名已跳过)" if n_forge_skipped > 0 else ""
                    print(f"  [S2] 库因子: {len(lib_factors)} 个模板公式{forge_note}，"
                          f"正计算 {len(candidates)} 候选相关性...")
                    # v0.6.1: 批量缓存式相关计算 (库模板只 eval 一次, 坏模板预筛剔除)
                    #          seed 重检候选豁免 (本就在库中, 自比较无意义)
                    pending_cands, pending_formulas, pending_names = [], [], []
                    for cand in candidates:
                        raw_corr = float(cand.get("max_corr",
                            cand.get("max_correlation", 0)) or 0)
                        has_corr = abs(raw_corr) > 1e-8
                        formula = cand.get("formula", cand.get("expression", ""))
                        if cand.get("_seed_recheck"):
                            continue
                        if not has_corr and formula:
                            pending_cands.append(cand)
                            pending_formulas.append(formula)
                            pending_names.append(cand.get("factor_name", ""))
                    if pending_formulas:
                        batch_res = ic_comp.compute_max_corr_vs_library_batch(
                            pending_formulas, lib_factors, pending_names)
                        for j, cand in enumerate(pending_cands):
                            key = pending_names[j]
                            if key not in batch_res:
                                continue
                            max_corr, max_factor = batch_res[key]
                            if max_corr >= 0:
                                cand["max_corr"] = max_corr
                                cand["max_corr_factor"] = max_factor
                                n_corr_computed += 1
                            elif max_corr == -1:
                                n_corr_unavailable += 1
                                cand["max_corr"] = 0.0  # 无法计算 → 保守放行
                                cand["max_corr_factor"] = ""

                    if n_corr_computed > 0:
                        max_c = max(float(c.get("max_corr", 0) or 0) for c in candidates)
                        print(f"  [S2] 相关性计算: {n_corr_computed}/{len(candidates)} 完成 "
                              f"(max_corr={max_c:.3f}), {n_corr_unavailable} 跳过")
                else:
                    print(f"  [S2] ⚠️ 无库因子公式可用，S2 将静默通过")
        except Exception as e:
            print(f"  [S2] ⚠️ 相关性计算失败: {e}，S2 将静默通过")

        # ── v0.5.2: S5 回测过滤器 + 轻量回退 ──
        s5_filter = None
        s5_fallback = False
        try:
            from s5_joint_filter import S5JointFilter
            s5_filter = S5JointFilter(top_n=80, sample_stocks=200)
            if not s5_filter.is_ready:
                s5_filter = None
        except Exception:
            pass

        # v0.5.2: S5 不可用时，用 IC 年拆分做轻量两年正向过滤
        if s5_filter is None and ic_comp is not None:
            print(f"  [S5] S5JointFilter 不可用，启用 IC 年拆分轻量过滤...")
            s5_fallback = True
            n_s5_populated = 0
            for cand in candidates:
                formula = cand.get("formula", cand.get("expression", ""))
                if not formula:
                    continue
                yearly = ic_comp.compute_yearly_ic(formula, cand.get("factor_name", ""))
                if yearly:
                    y2025 = yearly.get("2025") or {}
                    y2026 = yearly.get("2026") or {}
                    # 轻量 S5: 用 |IC| 做超额代理, |ICIR| 做 Calmar 代理
                    # 阈值: Calmar ≥ 0.3 (vs 真实 S5 的 1.0)
                    y1_excess = abs(y2025.get("ic", 0))
                    y2_excess = abs(y2026.get("ic", 0))
                    calmar_proxy = abs(yearly.get("all", {}).get("icir", 0))

                    cand["excess_2025"] = y1_excess
                    cand["excess_2026"] = y2_excess
                    cand["calmar"] = calmar_proxy
                    cand["_s5_lightweight"] = True
                    n_s5_populated += 1
            if n_s5_populated > 0:
                print(f"  [S5] 年拆分 IC 填充: {n_s5_populated}/{len(candidates)} 候选")

        # v3.1: SemanticVerifier 预检 — H↔E 假设-表达式对齐
        n_semantic_pass = 0
        for cand in candidates:
            hypothesis = cand.get("hypothesis", cand.get("logic", cand.get("rationale", "")))
            expression = cand.get("expression", cand.get("formula", ""))
            if hypothesis and expression:
                sem_result = self.semantic_verifier.verify(
                    hypothesis=hypothesis,
                    factor_expression=expression,
                    code="",
                    llm_available=False,
                )
                cand["semantic_pass"] = sem_result["pass"]
                cand["semantic_scores"] = sem_result.get("scores", {})
                cand["semantic_reasons"] = sem_result.get("reasons", [])
                if sem_result["pass"]:
                    n_semantic_pass += 1
            else:
                cand["semantic_pass"] = True  # 无假设信息时放行
                n_semantic_pass += 1

        if candidates:
            print(f"  [Semantic] H↔E 对齐检查: {n_semantic_pass}/{len(candidates)} 通过")

        results, summary = self.validator.validate(
            candidates=candidates,
            library_signals=library_signals,
            library_factors=library_factors,
            s5_filter=s5_filter,
            s5_lightweight=s5_fallback,
        )

        # v0.6 实验接线: 行为同质性硬拒 + 密封盲评 + 配对增量门禁
        # (开关全关=零行为变化; v0.6 全开=生效态, 硬拒直接剥夺 eligible)
        v06_eval = self._v06_evaluate_guards(
            candidates, results,
            library_signals=library_signals,
            lib_factors=locals().get("lib_factors") or [],
            s5_filter=s5_filter,
        )
        for _n in v06_eval.get("notes", []):
            print(f"  {_n}")
        if v06_eval.get("behavior_flagged"):
            print(f"  [v0.6-行为簇] 冗余标记: {v06_eval['behavior_flagged']} 候选 "
                  f"(NEAR_DUPLICATE/SUBSTITUTE), 硬拒 {v06_eval.get('behavior_rejected', 0)}")
        if v06_eval.get("holdout_verdicts"):
            print(f"  [v0.6-盲评] 密封盲评: {v06_eval['holdout_verdicts']} 候选 "
                  f"(verdict 附回 candidate['_holdout_verdict'])")
        if v06_eval.get("incremental_checked"):
            print(f"  [v0.6-增量] 配对增量: {v06_eval['incremental_checked']} 候选, "
                  f"拒绝 {v06_eval.get('incremental_rejected', 0)}")
        # v0.6 生效态: 硬拒同步 summary (eligible 被剥夺)
        n_hard_rej = int(v06_eval.get("behavior_rejected", 0)) + int(v06_eval.get("incremental_rejected", 0))
        if n_hard_rej > 0:
            summary["eligible_for_jq"] = max(0, int(summary.get("eligible_for_jq", 0)) - n_hard_rej)
            summary["v06_hard_rejected"] = n_hard_rej

        # P-20260827-001: SSPM 残差回填 (GPBreeder 通道, S5 二值残差)
        # 候选 source 字段 gp_{op} 携带编辑模式; 通过=+1 / S1未过=-1 / S1过S5未过=0(中性)
        if self.edit_memory is not None:
            try:
                _n_sspm = 0
                for _cand, _res in zip(candidates, results):
                    _src = str(_cand.get("source", ""))
                    if not _src.startswith("gp_"):
                        continue
                    _op = _src[3:]
                    if _op not in ("crossover", "mutate", "perturb"):
                        continue
                    _para = _cand.get("paradigm") or "auto_breed"
                    if (getattr(_res, "s5_passed", False)
                            or getattr(_res, "eligible_for_jq", False)):
                        _resid = 1.0
                    elif not getattr(_res, "s1_passed", False):
                        _resid = -1.0
                    else:
                        _resid = 0.0
                    self.edit_memory.record(_para, _op, _resid)
                    _n_sspm += 1
                if _n_sspm:
                    print(self.edit_memory.summary())
            except Exception as _sse:
                print(f"  [SSPM] 残差回填失败 (非阻塞): {_sse}")

        # P-20260826-005: 分层奖励观测 (不改变任何门槛, 仅计算 + 附到候选 + Top-3 摘要)
        try:
            _pz = getattr(self.breeder, 'penalizer', None)
            _layered_rows = []
            for _c, _r in zip(candidates, results):
                _icir = float(getattr(_r, 's1_icir', 0) or 0)
                _pen = 0.0
                if _pz is not None:
                    _f = _c.get('formula', '') or ''
                    if _f:
                        try:
                            _pen = float(_pz.penalty_for(_f))
                        except Exception:
                            _pen = 0.0
                _l2 = min(abs(_icir) / 0.5, 1.0) if abs(_icir) > 0 else 0.0      # 稳健: ICIR 归一
                _l3 = (1.0 - min(float(_c.get('max_corr', 0) or 0), 1.0)) * 0.6 \
                      + (1.0 - _pen) * 0.4                                        # 稀缺: 正交 + 子结构频率
                _score = 0.55 * _l2 + 0.25 * _l3 + 0.2 * (1.0 if getattr(_r, 's1_passed', False) else 0.0)
                _c['_layered_reward'] = round(_score, 3)
                _layered_rows.append((_score, _c.get('factor_name', _c.get('name', '?'))[:34]))
            if _layered_rows:
                _layered_rows.sort(reverse=True)
                _top = ', '.join(f'{n}({s:.2f})' for s, n in _layered_rows[:3])
                print(f"  [Layered] 分层奖励观测 Top-3: {_top}")
        except Exception as _le:
            print(f"  [Layered] 分层奖励计算失败 (非阻塞): {_le}")

        # 更新轨迹日志中的回测信息
        for cand, result in zip(candidates, results):
            traj_id = cand.get("_trajectory_id")
            if traj_id:
                # 记录 code 步骤 (从候选公式生成)
                formula = cand.get("formula", "")
                if formula:
                    self.traj_logger.log_code(
                        traj_id,
                        code=f"factor_func(df) → {formula[:200]}",
                        score=0.6 if result.s1_passed else 0.3,
                        compilation_success=True,
                    )
                self.traj_logger.log_backtest(
                    traj_id,
                    score=0.8 if result.eligible_for_jq else 0.3,
                    ic=result.s1_ic,
                    icir=result.s1_icir,
                )
                # v0.7 Phase 2: 结构化评估日志 (S1-S5 全量数据)
                reject_stage = ""
                reject_reason = ""
                # v3.1: SemanticVerifier 检查
                if not cand.get("semantic_pass", True):
                    reject_reason = f"Semantic H↔E mismatch: {'; '.join(cand.get('semantic_reasons', [])[:3])}"
                elif not result.s1_passed:
                    reject_stage = "S1"
                    reject_reason = result.s1_reason
                elif not result.s2_passed:
                    reject_stage = "S2"
                    reject_reason = result.s2_reason
                elif not result.s3_passed:
                    reject_stage = "S3"
                    reject_reason = result.s3_reason
                elif not result.s4_passed:
                    reject_stage = "S4"
                    reject_reason = result.s4_reason
                elif not result.s5_passed:
                    reject_stage = "S5"
                    reject_reason = result.s5_reason
                else:
                    reject_stage = "S5_PASS"
                # 计算公式复杂度
                try:
                    formula_ops = sum(formula.count(op) for op in
                        ['rolling','pct_change','shift','.mean(','.std(','.rank(','.sub(','.div(','.mul('])
                except Exception:
                    formula_ops = 0

                self.traj_logger.log_evaluation(
                    traj_id,
                    factor_name=cand.get("factor_name", cand.get("name", "")),
                    formula=formula,
                    paradigm=cand.get("paradigm", ""),
                    stage=reject_stage,
                    passed=result.eligible_for_jq,
                    ic=result.s1_ic,
                    icir=result.s1_icir,
                    excess_25=result.s5_year1_excess if result.s5_year1_excess else None,
                    excess_26=result.s5_year2_excess if result.s5_year2_excess else None,
                    calmar_25=result.s5_calmar if result.s5_calmar else None,
                    rejection_reason=reject_reason,
                    formula_complexity=formula_ops,
                )
                self.traj_logger.finalize(
                    traj_id,
                    outcome="PASS" if result.final_grade in ("A", "B") else "WEAK",
                )
                # ── L2 修复回路: 代码类失败分流 (2026-08-14) ──
                if not result.eligible_for_jq and reject_reason:
                    self._maybe_enqueue_repair(
                        cand=cand,
                        stage=reject_stage,
                        reason=reject_reason,
                    )

        # v0.6: MAB 冷启动修复 L2 — 任何有 S1 IC 的候选都给微弱引导信号
        # S5 通过 (JQ候选) → 正常信号 ×1.0; 仅 S1 通过 → 微弱信号 ×0.1
        # reward 上限 ±0.5 < JQ 终裁 ±1.0，本地信号只引导不主导
        if hasattr(self, 'mab') and self.mab:
            for cand, result in zip(candidates, results):
                icir = abs(float(getattr(result, 's1_icir', 0) or 0))
                if icir < 0.03:
                    continue  # ICIR 太低不配引导

                is_jq = getattr(result, 'eligible_for_jq', False)
                signal_mult = 1.0 if is_jq else 0.1  # 非JQ候选 微弱信号
                local_reward = np.tanh(icir * 3) * 0.3 * signal_mult

                cand_paradigm = str(cand.get('paradigm', ''))

                for d in list(self.mab.directions.values()):
                    if d.paradigm == cand_paradigm or cand_paradigm in str(d.name):
                        n = max(d.pulls, 1)
                        new_reward = (d.expected_reward * n + local_reward) / (n + 1)
                        d.expected_reward = max(-0.5, min(0.5, new_reward))
                        self.mab.save_state()
                        break

        return {
            "results": results,
            "summary": summary,
            "total_candidates": len(candidates),
            "stage5_passed": sum(1 for r in results if getattr(r, 's5_passed', False)),
            "eligible_for_jq": sum(1 for r in results if getattr(r, 'eligible_for_jq', False)),
            "grade_distribution": {
                g: sum(1 for r in results if getattr(r, 'grade', 'D') == g)
                for g in ['A', 'B', 'C', 'D']
            },
            "timestamp": datetime.now().isoformat(),
        }

    # ═══════════════════════════════════════════════════════════
    # Phase D: Distill
    # ═══════════════════════════════════════════════════════════

    def _maybe_enqueue_repair(self, cand: Dict, stage: str, reason: str) -> None:
        """L2 修复回路: 代码类失败分流 (2026-08-14)。

        判定逻辑: 失败原因含代码错误关键词 → 写入 repair_queue.json (幂等)。
        数据/公式/表现类失败不走此通道。Semantic H↔E mismatch 属假设问题, 排除。
        """
        if not reason:
            return
        if "Semantic" in str(reason) or "H↔E" in str(reason):
            return  # 假设-表达式不对齐是假设问题, 不是代码问题
        try:
            from repair_agent import is_code_failure, enqueue_repair
        except Exception:
            return  # repair_agent 不可用时不阻塞主线
        if not is_code_failure(reason):
            return
        enqueue_repair(
            factor_name=cand.get("factor_name", cand.get("name", "")),
            formula=str(cand.get("formula", cand.get("expression", ""))),
            stage=stage or "UNKNOWN",
            reason=reason,
            kind="local_python",
        )

    def _phase_distill(
        self,
        candidates: List[Dict],
        evaluate_result: Dict,
        paradigm: str,
    ) -> Dict:
        """
        D 阶段: 将本批经验蒸馏到记忆库。

        使用 F/E/R 中的 F (Formation) 和 E (Evolution) 操作符:
        - F: 从本批轨迹提取 SuccessPattern + ForbiddenDirection
        - E: 合并冗余 + 淘汰低效用

        P-018: 入库前 Red Sea 相关性核验
        """
        # ── P-018: Red Sea 入库核验 ──
        red_sea_rejections = 0
        try:
            from library_orthogonality import LibraryOrthogonalityManager
            mgr = LibraryOrthogonalityManager()
            if mgr._corr_matrix is not None:
                for cand in candidates:
                    factor_name = cand.get("name", "")
                    mgr.register_factor_lite(factor_name)  # 轻量注册
                    crowding = mgr.get_crowding_score(factor_name)
                    if crowding is not None and crowding > 0.8:
                        red_sea_rejections += 1
                        cand["red_sea_rejected"] = True
                        cand["crowding_score"] = crowding
        except Exception:
            pass  # Red Sea不可用时不阻塞主线

        # ── P-20260819-005: 冗余感知融合检测 (AlphaSeek 对齐) ──
        # 新候选与 SOTA 库 AST 结构相似度 >0.7 → 标记 redundancy_fused
        # (纯检测标记, 不改变入库行为; 融合动作由后续版本实现)
        try:
            from trajectory_pool import (ENABLE_TRAJECTORY_POOL,
                                         redundancy_check, collect_sota_formulas)
            if ENABLE_TRAJECTORY_POOL:
                sota_formulas = collect_sota_formulas(self.memory)
                if sota_formulas:
                    for cand in candidates:
                        f = cand.get("formula") or cand.get("expression") or ""
                        if not f:
                            continue
                        res = redundancy_check(f, sota_formulas)
                        if res["redundant"]:
                            cand["redundancy_fused"] = True
                            cand["redundancy_max_sim"] = res["max_sim"]
                            print(f"    [冗余融合] {cand.get('name', '?')[:36]} "
                                  f"与 SOTA 相似 {res['max_sim']:.2f} → 标记融合建议")
        except Exception:
            pass  # 冗余检测不可用时不阻塞主线

        # F: Formation
        formation_result = self.memory.form(
            trajectory={
                "batch_id": f"ralph_{datetime.now().strftime('%Y%m%d')}",
                "candidates": candidates,
            },
            auto_save=False,
        )

        # E: Evolution
        evolution_result = self.memory.evolve(auto_save=True)

        # ── v0.7 Phase 2: 轨迹自动蒸馏 → 指导后续 MAB ──
        distillation_hints = {}
        try:
            analysis = self.traj_logger.analyze_failures(recent_n=200)
            hints = self.traj_logger.get_distillation_hints()
            # 存储为实例变量，供 _mab_select_direction 使用
            self._distillation_hints = hints

            if analysis.get("total_analyzed", 0) >= 5:
                header = f"  [🧬 EvoTraj Phase 2] {analysis.get('summary', '')[:120]}"
                print(header)
                if hints.get("prefer_paradigms"):
                    top3 = hints["prefer_paradigms"][:3]
                    print(f"    → 优先方向: {', '.join(f'{p}(IC={ic:.4f})' for p,ic in top3)}")
                if hints.get("avoid_paradigms"):
                    avoid = hints["avoid_paradigms"][:3]
                    print(f"    → 避开方向: {', '.join(avoid)}")
                if hints.get("target_complexity_range"):
                    lo, hi = hints["target_complexity_range"]
                    print(f"    → 推荐复杂度: {lo}-{hi} ops")
                distillation_hints = hints
            else:
                print(f"  [🧬 EvoTraj Phase 2] 轨迹不足({analysis.get('total_analyzed',0)}条), 等待积累...")
        except Exception as e:
            print(f"  [🧬 EvoTraj Phase 2] ⚠️ 蒸馏失败: {e}")
            self._distillation_hints = None

        return {
            "summary": {
                "new_patterns": formation_result["new_success_patterns"],
                "new_forbidden": formation_result["new_forbidden_directions"],
                "merged": evolution_result.get("merged_patterns", 0),
                "upgraded": evolution_result.get("upgraded_forbidden", 0),
                "archived": evolution_result.get("archived_stale", 0),
                "red_sea_rejections": red_sea_rejections,  # P-018
            },
            "formation": formation_result,
            "evolution": evolution_result,
        }

    # ═══════════════════════════════════════════════════════════
    # Phase D+: JQ 回测反馈闭环
    # ═══════════════════════════════════════════════════════════

    def jq_feedback(
        self,
        jq_backtest_result: Dict,
        trajectory_ids: Optional[List[str]] = None,
        evo_trajectories: Optional[List[Dict]] = None,  # v0.6: EvoTraj summaries
    ) -> Dict:
        """
        JQ 回测反馈: 将真实回测结果注入 D 阶段，实现完整 R→G→E→D→JQ→D+ 闭环。

        对标 FactorMiner 的"真实环境反馈"机制.
        local IC/ICIR 是代理信号，JQ 回测是唯一真相源。
        D 阶段在 JQ 反馈前只能基于 local 数据做弱蒸馏；
        D+ 阶段用 JQ 结果做强蒸馏，产出可真正复用的模式。

        v0.6 增强: 接受 evo_trajectories 参数，用于 MAB streak 奖励更新
          - streak 越高 → MAB reward 越大 → 方向获得更多资源

        Parameters
        ----------
        jq_backtest_result: {
            "batch_id": str,
            "composite_return": float,   # %
            "composite_sharpe": float,
            "composite_maxdd": float,    # %
            "factors": [
                {
                    "factor_name": str,
                    "formula": str,
                    "hypothesis": str,
                    "paradigm": str,
                    "category": str,
                    "operators_used": List[str],
                    "local_ic": float,
                    "local_icir": float,
                    "jq_return": float,
                    "jq_sharpe": float,
                    "jq_maxdd": float,
                    "jq_composite_contribution": str,
                    "root_cause": str,
                },
                ...
            ]
        }
        trajectory_ids: 关联的轨迹 ID 列表（可选，用于更新已有轨迹）
        evo_trajectories: v0.6 EvoTraj 轨迹摘要列表 [{traj_id, paradigm, streak, best_icir, ...}]

        Returns
        -------
        {
            "trajectories_updated": int,
            "memory_formation": Dict,
            "memory_evolution": Dict,
            "hard_forbidden_added": int,
            "jq_success_confirmed": int,
            "mab_streak_updates": int,   # v0.6
        }
        """
        print(f"\n{'='*60}")
        print(f"  Ralph Loop D+: JQ 回测反馈闭环")
        print(f"  Composite: {jq_backtest_result.get('composite_return', 0):.1f}% / "
              f"Sharpe {jq_backtest_result.get('composite_sharpe', 0):.2f}")
        print(f"{'='*60}")

        factors = jq_backtest_result.get("factors", [])

        # ── Step 1: 更新轨迹日志中的 backtest 步骤 ──────
        trajectories_updated = 0
        for f in factors:
            name = f.get("factor_name", "")
            jq_ret = f.get("jq_return", 0)
            jq_sharpe = f.get("jq_sharpe", 0)

            # 尝试从已有轨迹中找到匹配的（通过种子因子名）
            matched = False
            for traj_id, traj in self.traj_logger.trajectories.items():
                # 兼容 hypothesis 为 dict(轨迹结构) 或 str(回测结果) 两种形态
                _hyp = f.get("hypothesis", "")
                _hyp_s = _hyp.get("content", "") if isinstance(_hyp, dict) else str(_hyp)
                if name in traj.seed_factor or (
                    _hyp_s and _hyp_s[:30] in str(traj.paradigm or "")
                ):
                    # 更新 backtest 步骤为 JQ 真实结果
                    jq_score = 0.8 if jq_ret > 50 and jq_sharpe > 0.4 else (
                        0.3 if jq_ret > -20 else 0.05  # JQ 失败 → 极低分
                    )
                    # P-001: 优先使用真实 per-factor IC/ICIR (日志提取), 缺失时才退化到收益代理
                    _ic = f.get("jq_ic")
                    _icir = f.get("jq_icir")
                    self.traj_logger.log_backtest(
                        traj_id,
                        score=jq_score,
                        ic=_ic if _ic is not None else jq_ret / 100,
                        icir=_icir if _icir is not None else jq_sharpe,
                        sharpe=jq_sharpe,
                        maxdd=f.get("jq_maxdd", 0),
                    )
                    self.traj_logger.log_lesson(
                        traj_id,
                        f"JQ验证: {jq_ret:.1f}%/{jq_sharpe:.2f}/MDD{f.get('jq_maxdd',0):.1f}%"
                        + (f" | 根因: {f.get('root_cause', '未知')}" if f.get('root_cause') else ""),
                    )
                    # 标记 JQ 已验证 (四档严格判定, 与 form_from_jq 阈值一致)
                    self.traj_logger.finalize(
                        traj_id,
                        outcome=self._jq_outcome(jq_ret, jq_sharpe),
                        jq_return=jq_ret,
                        jq_sharpe=jq_sharpe,
                    )
                    trajectories_updated += 1
                    matched = True
                    break

            if not matched:
                # 无匹配轨迹，创建新轨迹记录 JQ 结果
                traj = self.traj_logger.start_trajectory(
                    paradigm=f.get("paradigm", "JQ验证"),
                    seed_factor=name,
                    phase="jq_feedback",
                )
                self.traj_logger.log_hypothesis(
                    traj.trajectory_id,
                    f.get("hypothesis", "") or name,
                    score=0.5,
                )
                if f.get("formula"):
                    self.traj_logger.log_expression(
                        traj.trajectory_id,
                        f["formula"],
                        score=0.5,
                    )
                jq_score = 0.8 if (jq_ret > 50 and jq_sharpe > 0.4) else (
                    0.3 if jq_ret > -20 else 0.05
                )
                _ic = f.get("jq_ic")
                _icir = f.get("jq_icir")
                self.traj_logger.log_backtest(
                    traj.trajectory_id,
                    score=jq_score,
                    ic=_ic if _ic is not None else jq_ret / 100,
                    icir=_icir if _icir is not None else jq_sharpe,
                    sharpe=jq_sharpe,
                    maxdd=f.get("jq_maxdd", 0),
                )
                self.traj_logger.finalize(
                    traj.trajectory_id,
                    outcome=self._jq_outcome(jq_ret, jq_sharpe),
                    jq_return=jq_ret,
                    jq_sharpe=jq_sharpe,
                )
                trajectories_updated += 1

        print(f"  [Trajectory] 更新了 {trajectories_updated} 条轨迹")

        # ── Step 2: JQ 驱动 F (Formation) ─────────────────
        print(f"\n  [Memory] 提取 JQ 驱动的模式...")
        formation_result = self.memory.form_from_jq(
            jq_results=jq_backtest_result,
            auto_save=False,
        )

        hard_forbidden = formation_result["jq_forbidden_directions"]
        jq_success = formation_result["jq_success_patterns"]
        jq_warnings = formation_result.get("jq_warning_directions", 0)
        print(f"    JQ 禁止方向: {hard_forbidden} (硬禁止)")
        print(f"    JQ 软负警告: {jq_warnings} (severity=soft)")
        print(f"    JQ 确认成功: {jq_success}")

        # ── Step 2.5: Motif 蒸馏 (v0.5 P-001 + L1 两级蒸馏) ───────────
        print(f"\n  [Memory] Motif 级知识蒸馏...")
        motif_distilled = 0
        for f in factors:
            formula = f.get("formula", "")
            jq_ret = f.get("jq_return", 0)
            jq_sharpe = f.get("jq_sharpe", 0)
            jq_passed = jq_ret > 50 and jq_sharpe > 0.4
            # MARGINAL (2026-08-15): 收益正但未达 PASS 阈值 → 中性, 不记 jq_fail
            jq_marginal = ((not (jq_ret < -20 or jq_sharpe < -0.5))
                           and (not jq_passed) and jq_ret >= 0)
            try:
                self.memory.distill_motif_knowledge(
                    formula=formula,
                    jq_passed=jq_passed,
                    jq_marginal=jq_marginal,
                    fsa=self.fsa,
                    interpretation=f.get("interpretation"),  # L1 (2026-08-14)
                )
                motif_distilled += 1
            except Exception as e:
                pass  # FSA 不可用或 motif 提取失败
        print(f"    蒸馏因子数: {motif_distilled}/{len(factors)}")

        # ── Step 2.6: JQ 通过因子自动入库 seed_injector (v0.6) ──
        auto_入库 = 0
        for f in factors:
            formula = f.get("formula", "")
            factor_name = f.get("factor_name", "")
            jq_ret = f.get("jq_return", 0)
            jq_sharpe = f.get("jq_sharpe", 0)
            paradigm = f.get("paradigm", "auto_breed")
            if jq_ret > 50 and jq_sharpe > 0.4 and formula and factor_name:
                try:
                    from seed_injector import _add_champion_factor as _acf
                    _acf(
                        expression=formula,
                        name=f"{factor_name}_JQ验证",
                        rationale=f"GP育种 JQ自动入库: {factor_name} | JQ +{jq_ret:.1f}%/Sharpe {jq_sharpe:.2f}",
                        direction="+",
                        paradigm=paradigm,
                        source="gp_jq_validated",
                    )
                    auto_入库 += 1
                    print(f"    📦 自动入库: {factor_name} → seed_injector (JQ +{jq_ret:.1f}%)")
                except Exception as e:
                    pass  # 入库失败不阻塞主流程
        if auto_入库 > 0:
            print(f"    共入库 {auto_入库} 个 JQ 验证因子")

        # ── Step 3: E (Evolution) — 合并/去重 ─────────────
        print(f"\n  [Memory] 演化合并...")
        evolution_result = self.memory.evolve(auto_save=True)

        # ── Step 3.5: v0.6 EvoTraj streak MAB 更新 ─────────
        mab_streak_updates = 0
        if evo_trajectories:
            for traj_summary in evo_trajectories:
                paradigm = traj_summary.get("paradigm", "")
                streak = traj_summary.get("streak", 0)
                best_icir = traj_summary.get("best_icir", 0)
                traj_reward = traj_summary.get("trajectory_reward", 0)
                if paradigm and streak > 0:
                    # 找到对应的 MAB 方向
                    dirs = list(self.mab.directions.values()) if isinstance(self.mab.directions, dict) else self.mab.directions
                    for d in dirs:
                        if d.name == paradigm or d.direction_id == paradigm:
                            # 使用 JQ composite return 作为 baseline
                            jq_return = jq_backtest_result.get("composite_return", 0)
                            relative = (jq_return - 182.57) / 100.0
                            streak_multiplier = 1.0 + 0.2 * streak
                            reward = max(-1.0, min(1.0, relative * 5 * streak_multiplier))
                            d.expected_reward = (d.expected_reward * d.pulls + reward) / (d.pulls + 1) if d.pulls > 0 else reward
                            d.jq_results.append({
                                "return": jq_return,
                                "sharpe": jq_backtest_result.get("composite_sharpe", 0),
                                "streak": streak,
                                "traj_reward": traj_reward,
                                "best_icir": best_icir,
                            })
                            self.mab.save_state()
                            mab_streak_updates += 1
                            print(f"  [MAB] EvoTraj → {d.name}: streak={streak}, "
                                  f"reward={reward:.3f} ({streak_multiplier:.1f}x)")
                            break

        # ── Step 3.6: L1 interpretation → MAB 方向级精细化 (2026-08-14) ──
        # 边界: 只许方向级修正, 永不碰因子级筛选 (AlphaAgent v3 毒药区)
        # execution_failure → 撤销失败计数 (方向有 local 证据, 执行崩不把方向打入冷却)
        # direction_falsified → 惩罚 (方向被 JQ 证伪)
        mab_interp_updates = 0
        try:
            dirs = list(self.mab.directions.values()) if isinstance(self.mab.directions, dict) else self.mab.directions
            for f in factors:
                interp = f.get("interpretation") or {}
                verdict = str(interp.get("verdict", ""))
                if verdict not in ("execution_failure", "direction_falsified"):
                    continue
                paradigm = str(f.get("paradigm", ""))
                matched = False
                for d in dirs:
                    if paradigm and (d.name == paradigm or d.direction_id == paradigm or paradigm in str(d.name)):
                        matched = True
                        if verdict == "execution_failure":
                            if getattr(d, "failures", 0) > 0:
                                d.failures -= 1
                                print(f"  [MAB-L1] {d.name}: execution_failure → 失败计数撤销 "
                                      f"({d.failures} 剩余), 方向不惩罚")
                        elif verdict == "direction_falsified":
                            d.failures += 1
                            d.expected_reward = max(-0.5, getattr(d, "expected_reward", 0) - 0.2)
                            print(f"  [MAB-L1] {d.name}: direction_falsified → 惩罚 -0.2, "
                                  f"failures={d.failures}")
                        self.mab.save_state()
                        mab_interp_updates += 1
                        break
                if not matched:
                    # 2026-08-18: 范式未注册 MAB 臂时显式告警 (此前静默 miss 导致归因信号无处落地)
                    print(f"  [MAB-L1] ⚠️ {f.get('factor_name', '?')} 范式 {paradigm!r} "
                          f"无对应 MAB 臂 ({verdict} 信号未写回) — 若该范式持续产出, "
                          f"应在 mab_scheduler_state.json 注册方向")
        except Exception as e:
            print(f"  [MAB-L1] ⚠️ 方向精细化失败 (不阻塞): {e}")

        # ── Step 3.9: JQ 代码生成队列状态回写 (2026-08-19 修复) ──
        # pending_jq_run → jq_run_done, 附回测结果。此前无任何环节回写队列,
        # 导致已验证因子长期挂在 pending_jq_run, 日报反复误报"待回测"。
        jq_queue_synced = 0
        try:
            import json as _json
            from pathlib import Path as _Path
            _qp = _Path(__file__).resolve().parents[1] / "data" / "jq_codegen_queue.json"
            if _qp.exists():
                _q = _json.load(open(_qp, encoding="utf-8"))
                _items = _q.get("queue", _q) if isinstance(_q, dict) else {}
                if isinstance(_items, dict):
                    for _f in factors:
                        _fn = _f.get("factor_name", "")
                        if _fn in _items and _items[_fn].get("status") == "pending_jq_run":
                            _items[_fn]["status"] = "jq_run_done"
                            _items[_fn]["jq_return"] = _f.get("jq_return")
                            _items[_fn]["jq_sharpe"] = _f.get("jq_sharpe")
                            _items[_fn]["jq_maxdd"] = _f.get("jq_maxdd")
                            _items[_fn]["jq_verified_at"] = datetime.now().isoformat()
                            _items[_fn]["root_cause"] = _f.get("root_cause", "")
                            jq_queue_synced += 1
                    _json.dump(_q, open(_qp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                if jq_queue_synced:
                    print(f"  [JQ队列] 状态回写: {jq_queue_synced} 个 pending_jq_run → jq_run_done")
        except Exception as e:
            print(f"  [JQ队列] ⚠️ 状态回写失败 (不阻塞): {e}")

        # ── Step 4: 汇总 ──────────────────────────────────
        self.stats["total_jq_feedbacks"] = self.stats.get("total_jq_feedbacks", 0) + 1
        self.stats["last_jq_feedback"] = datetime.now().isoformat()

        summary = {
            "trajectories_updated": trajectories_updated,
            "memory_formation": formation_result,
            "memory_evolution": evolution_result,
            "hard_forbidden_added": hard_forbidden,
            "soft_warnings_added": jq_warnings,   # v3.1: 软负收益警告
            "jq_success_confirmed": jq_success,
            "mab_streak_updates": mab_streak_updates,  # v0.6
            "mab_interp_updates": mab_interp_updates,  # L1 (2026-08-14)
            "jq_queue_synced": jq_queue_synced,  # (2026-08-19)
        }

        print(f"\n  D+ 完成:")
        print(f"    轨迹更新: {trajectories_updated}")
        print(f"    硬禁止方向: +{hard_forbidden}")
        print(f"    软负警告: +{jq_warnings}")
        print(f"    JQ确认成功: +{jq_success}")
        print(f"    Memory记录: {self.memory.data['stats']['total_attempts']}")

        return summary

    @staticmethod
    def _jq_outcome(jq_ret: float, jq_sharpe: float) -> str:
        """JQ 结果四档判定 (与 form_from_jq 阈值一致, 防止仅收益>0 就被误标 JQ_PASSED)"""
        if jq_ret < -20 or jq_sharpe < -0.5:
            return "JQ_FAILED"
        if jq_ret > 50 and jq_sharpe > 0.4:
            return "JQ_PASSED"
        if jq_ret < 0:
            return "JQ_WEAK_NEGATIVE"
        return "JQ_MARGINAL"

    # ═══════════════════════════════════════════════════════════
    # v0.3 新增: 多轮自动化 + 断点续跑
    # ═══════════════════════════════════════════════════════════

    def run_single_round(
        self,
        generator: str = "gp_breed",
        max_candidates: int = 10,
        **kwargs,
    ) -> Dict:
        """
        执行单轮循环（用于定时自动化任务）。

        简化接口: 自动从 memory 获取库状态。

        Parameters
        ----------
        generator: 生成方式 ("gp_breed" / "alpha_agent_v3")
        max_candidates: 每轮生成候选数
        """
        return self.run(
            generator=generator,
            max_candidates=max_candidates,
            use_fsa=True,
            use_reviewer=True,
            use_checkpoint=True,
            **kwargs,
        )

    def resume_from_checkpoint(self) -> Tuple[bool, Dict]:
        """
        尝试从检查点恢复并继续运行。
        
        Returns
        -------
        (success, state_or_error)
        """
        if not self.checkpoint.can_resume():
            return False, {"error": "无可用检查点"}

        success, state = self.checkpoint.resume()
        if not success:
            return False, {"error": "检查点恢复失败"}

        return True, {
            "iteration": state.iteration,
            "total_tested": state.total_tested,
            "total_approved": state.total_approved,
            "approved_factors": state.approved_factors,
        }

    def scan_library_fsa(
        self, factors: List[Dict] = None
    ) -> Dict:
        """
        对当前因子库执行 FSA 扫描（供 MetaController 调用）。

        Parameters
        ----------
        factors: 因子列表。如果未提供，从 memory 获取。

        Returns
        -------
        FSAReport dict
        """
        if factors is None:
            # 从 experience memory 获取
            memory_factors = self.memory.data.get("attempts", [])
            factors = [
                {"factor_name": a.get("factor_name", ""),
                 "formula": a.get("formula", a.get("expression", ""))}
                for a in memory_factors
            ]

        if not factors:
            return {"total_factors": 0, "message": "无可扫描因子"}

        report = self.fsa.scan_library(factors, persist=True)
        return {
            "total_factors": report.total_factors,
            "total_skeletons": report.total_skeletons,
            "forbidden_count": report.forbidden_count,
            "forbidden_skeletons": report.forbidden_skeletons,
            "max_concentration": report.max_concentration,
            "max_concentration_skeleton": report.max_concentration_skeleton,
            "diversity_score": self.fsa.get_diversity_score(),
            "warnings": report.warnings,
        }

    def get_status(self) -> Dict:
        """获取 Ralph Loop 运行状态 (v0.3 增强)"""
        mem_summary = self.memory.get_summary()
        traj_summary = self.traj_logger.get_summary()
        return {
            "ralph_stats": self.stats,
            "memory": mem_summary,
            "trajectories": traj_summary,
            "fsa": self.fsa.get_status(),
            "checkpoint": self.checkpoint.get_progress(),
            "reviewer": self.reviewer.get_stats(),
            "last_run": self.stats["last_run"],
        }

    def get_jq_readiness(self, n_candidates: int = 0) -> Dict:
        """评估是否准备好提交 JQ"""
        return {
            "memory_ready": self.memory.data["stats"]["total_attempts"] > 0,
            "n_success_templates": len(self.memory.data.get("success_templates", [])),
            "n_forbidden_directions": len(self.memory.data.get("forbidden_directions", [])),
            "recommendation": (
                "ready" if n_candidates > 0 and self.memory.data["stats"]["total_attempts"] >= 10
                else "need_more_attempts"
            ),
        }


# ── 便捷函数 ──────────────────────────────────────────────

_default_ralph: Optional[RalphLoop] = None


def get_ralph() -> RalphLoop:
    global _default_ralph
    if _default_ralph is None:
        _default_ralph = RalphLoop()
    return _default_ralph


# ── 测试 ──────────────────────────────────────────────────

if __name__ == "__main__":
    ralph = RalphLoop()

    mock_candidates = [
        {
            "factor_name": "momentum_vol_adaptive",
            "formula": "ts_delta(close, 20) / ts_std(close, 20)",
            "hypothesis": "20日动量除以波动率，在低波动时放大信号",
            "paradigm": "动量反转",
            "ic": 0.045,
            "icir": 0.52,
            "max_corr": 0.35,
            "max_corr_factor": "existing_momentum",
            "operators_used": ["ts_delta", "ts_std"],
        },
        {
            "factor_name": "liquidity_vol_simple",
            "formula": "ts_mean(volume, 5)",
            "hypothesis": "5日平均成交量",
            "paradigm": "资金流",
            "ic": 0.008,
            "icir": 0.12,
            "max_corr": 0.55,
            "operators_used": ["ts_mean"],
        },
    ]

    # 运行 Ralph Loop
    result = ralph.run(
        candidates=mock_candidates,
        generator="manual",
        paradigm="动量反转",
    )

    print(f"\n完整结果:")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))

    print(f"\nRalph Loop 状态:")
    print(json.dumps(ralph.get_status(), indent=2, ensure_ascii=False))