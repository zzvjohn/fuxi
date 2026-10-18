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
import os
import json
import re
import time
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple, Any


# ═══════════════════════════════════════════════════════════
# v0.9: 种子公式翻译 — S5 终端白名单 + 遗留函数规范化
# ── P-20260918-004: 白名单 / 翻译器 **单一真源** ────────────
# 此前 `_S5_KNOWN_TERMINALS`/`_S5_KNOWN_FUNCS`/`_is_numeric_const`/
# `_canonicalize_neg_inline`/`_find_unknown_identifiers`/
# `_normalize_legacy_rolling` 与 3 个 `_LEGACY_ROLLING_*` 常量都在本文件
# 内联定义, 而 `factor_ic_computer._normalize_to_pandas` 另有一套**正则白名单**
# ⇒ 两套翻译器覆盖范围不一致 = 库模板静默零覆盖 (实测仅 25.4%)。
# 现统一引用 `forge_translate` (叶子模块: 只依赖 forge.expression, 无循环导入)。
# 🔴 导入必须放在 `sys.path.insert(...)` **之后** (下方库导入区), 否则
#    `forge_translate` 尚未可导入 ⇒ ImportError (v3 补丁把它放在此处并因此隐雷)。
# ═══════════════════════════════════════════════════════════






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


def _build_wq101_anchor() -> str:
    """P-20260909-004: WQ101 已验证算子先验锚定 (默认 on, env FUXI_WQ101_ANCHOR=off 关闭)。

    数据源 = data/wq101_coverage_audit.json (P-20260908-002 审计产出, 与审计永远同步):
    - 仅锚定本库已有模板实证的算子类 (coverage > 0);
    - 缺口族 (ts_decay/indneutralize/signedpower/cross_corr) 不锚定;
    - indneutralize 永久排除 (已证伪方向)。
    返回锚定 prompt 片段 (空串 = 不可用/未启用, 不改变原 prompt)。
    """
    import json as _json
    from pathlib import Path as _Path
    try:
        _audit_path = (
            _Path(__file__).resolve().parents[2] / "data" / "wq101_coverage_audit.json"
        )
        if not _audit_path.exists():
            return ""
        _audit = _json.load(open(_audit_path, encoding="utf-8"))
        _classes = _audit.get("operator_classes", {})
        _verified = []
        for _name, _info in _classes.items():
            if not isinstance(_info, dict):
                continue
            if float(_info.get("coverage", 0) or 0) <= 0:
                continue  # 缺口族不锚定 (无本地实证)
            if "indneutralize" in _name:
                continue  # 已证伪方向, 永久排除
            _ex = (_info.get("examples") or ["-"])
            _ex0 = str(_ex[0])[:90] if _ex else "-"
            _verified.append(
                f"- {_name}: 本库已验证 {int(_info.get('hit_templates', 0) or 0)} 个模板"
                f" (例: {_ex0})"
            )
        if not _verified:
            return ""
        return (
            "\n[WQ101 结构化先验锚定] 以下算子模式已在 A 股本地库有实证模板, "
            "优先使用这些已验证算子组合生成公式; "
            "缺口算子(ts_decay/indneutralize/signedpower/cross_corr)本地无实证, 不要使用:\n"
            + "\n".join(_verified)
            + "\n"
        )
    except Exception:
        return ""


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


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experience_memory import ExperienceMemory, get_memory, _norm_engine

# ── P-20260918-004: Forge DSL 翻译层单一真源 (叶子模块, 无循环导入) ──
#    保留下面的旧名别名 ⇒ 全部既有调用点零改动。
from forge_translate import (  # noqa: E402
    S5_KNOWN_TERMINALS as _S5_KNOWN_TERMINALS,
    S5_KNOWN_FUNCS as _S5_KNOWN_FUNCS,
    is_numeric_const as _is_numeric_const,
    canonicalize_neg_inline as _canonicalize_neg_inline,
    find_unknown_identifiers as _find_unknown_identifiers,
    normalize_legacy_rolling as _normalize_legacy_rolling,
)
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


def _cfg_div_discount_w() -> float:
    """P-20260831 P1: 多样性折扣权重 (config.DIV_DISCOUNT_W, 默认 0=关闭生产零变化)。
    config 不可用/无此常量时静默回退 0.0 — 软引导, 绝不影响主流程。"""
    try:
        from config import DIV_DISCOUNT_W
        return float(DIV_DISCOUNT_W or 0.0)
    except Exception:
        return 0.0


def _bh_wire_mode() -> str:
    """P-20260917-024: 行为同质性门禁接线开关。

    off(默认) = 完全沿用历史行为 (library_signals 恒 None ⇒ 门禁零评估, 零回归)
    shadow    = **接线并真实评估**, 打印本应硬拒的条数, 但**不剥夺** eligible
                (为 enforce 转正积累误杀率证据)
    on        = 接线 + 硬拒生效 (v0.6 评价宪法原意)

    🔴 为何默认 off: 接线会把一个**从未生效过的硬拒闸门**首次点亮。虽然方向是
    单向收紧 (不会凭空造出 PASS), 但 `behavior_homogeneity._jl_projection` 会
    丢弃股票身份 (按值分布做 rank), 存在误杀风险 ⇒ 必须先 shadow 拿到误杀读数,
    再按「翻默认过四关」流程决定是否 enforce。
    """
    _m = (os.environ.get("FUXI_BEHAVIOR_GATE_WIRE", "off") or "off").strip().lower()
    return _m if _m in ("off", "shadow", "on") else "off"


# ═══════════════════════════════════════════════════════════════
# M9 (2026-09-26): 统一质量门接线 (off|shadow|enforce) + 纯函数分级
# ═══════════════════════════════════════════════════════════════
# 病灶 (E1): `_gate_fatal` 只在 _phase_generate 内写入与打印, 从未被任何
#   下游消费 ⇒ 未过门候选照常进入 S1-S5 ⇒ 假门禁 (消耗评估预算, 且在
#   账本上与"过门候选"不可区分)。
# 口径: shadow(默认) = 记账留档 (records 随 batch 输出落盘), 不拦截,
#   零行为变化 (为误杀率 A/B 积累读数); enforce = 拒绝未过门候选,
#   被拒项完整进 records (不无声消失 — 被全拒项不得从输出消失)。
# 纪律: shadow→enforce 前须核对 records 分母 (n_fatal/n_checked) 与误杀
#   样例。单向下锁: 只会减少候选, 不会凭空造出通过。

def quality_gate_wire_mode() -> str:
    """解析 FUXI_QUALITY_GATE_WIRE (off|shadow|enforce, 默认 shadow)。

    非法值直接抛 ValueError (fail-fast; 配置错误不得被静默吞成默认)。
    """
    _m = (os.environ.get("FUXI_QUALITY_GATE_WIRE", "shadow") or "shadow").strip().lower()
    if _m not in ("off", "shadow", "enforce"):
        raise ValueError("FUXI_QUALITY_GATE_WIRE must be off, shadow or enforce")
    return _m


def quality_gate_wire_step(
    candidates: List[Dict],
    mode: str,
    stage: str = "",
    max_records: int = 50,
) -> Dict:
    """M9: 统一质量门分级 (纯函数, 零副作用; 基于候选上已有的 _gate_result)。

    输入契约: 候选须已由调用方跑过 FactorQualityGate.verify (写入
    `_gate_result` / `_gate_fatal`)。无 `_gate_result` 的候选视为**未评估**
    — 不计入 n_fatal, 也不被 enforce 误删 (评估口径: 未评估≠拒绝)。
    enforce 仅删除"有判定且未过门"的候选; 其余模式 kept 原样返回。
    """
    fatal_cands = [
        c for c in candidates
        if c.get("_gate_result") is not None and not c["_gate_result"].passed
    ]
    records: List[Dict] = []
    for _c in fatal_cands[:max(0, int(max_records))]:
        records.append({
            "stage": str(stage or ""),
            "name": str(_c.get("factor_name", _c.get("name", "?")))[:80],
            "formula": str(_c.get("formula", _c.get("expression", "")) or "")[:160],
            "fatal_issues": [str(_i)[:200] for _i in (_c.get("_gate_fatal") or [])][:5],
            "removed": bool(mode == "enforce"),
        })
    kept = list(candidates)
    n_removed = 0
    if mode == "enforce":
        kept = [
            c for c in candidates
            if not (c.get("_gate_result") is not None and not c["_gate_result"].passed)
        ]
        n_removed = len(candidates) - len(kept)
    return {
        "mode": str(mode or ""),
        "n_checked": len(candidates),
        "n_fatal": len(fatal_cands),
        "n_removed": n_removed,
        "kept": kept,
        "records": records,
    }


def _bh_panel_to_signal(df, n_days: int = 60):
    """因子值长表面板 → 信号向量 (最近 n_days 交易日每股均值)。

    口径对齐 run_v4_pipeline 的 factor_value_cache (「最近60个交易日 → 每股均值
    向量」), 保持与既有聚类/DIV_DISCOUNT 参照集同一语义, 不新造口径。
    """
    try:
        if df is None or len(df) == 0:
            return None
        need = {"ts_code", "trade_date", "factor_value"}
        if not need.issubset(set(df.columns)):
            return None
        d = df.sort_values("trade_date")
        recent = d.groupby("ts_code").tail(int(n_days))
        s = recent.groupby("ts_code")["factor_value"].mean()
        arr = s.to_numpy(dtype=float)
        return arr if arr.size >= 8 else None
    except Exception:
        return None


def _s2a_refiner_gate(generator: str, pulls: int,
                      env: Optional[Dict] = None) -> Dict:
    """
    S2-a (2026-09-12): gp_evolve 精炼器补路的**纯门控函数** (可独立测试)。

    返回 {"mode", "min_pulls", "every", "mature", "probe", "reason"}
      mode  : off | shadow | on (FUXI_EVO_REFINER, 默认 shadow)
      probe : 是否应当**实际执行**精炼 (仅 on + 成熟 + 周期命中)
      reason: 未触发的可读归因 (诊断用)

    抽成纯函数的原因: 原实现在 `run()` 内联 40+ 行, 不跑完整流水线 (数百秒)
    无法验证; 抽离后单测毫秒级覆盖全分支。
    """
    import os as _os
    _env = env if env is not None else _os.environ
    mode = str(_env.get("FUXI_EVO_REFINER", "shadow")).strip().lower()
    if mode not in ("off", "shadow", "on"):
        mode = "shadow"
    try:
        min_pulls = int(_env.get("FUXI_EVO_REFINER_MIN_PULLS", "3") or 3)
    except Exception:
        min_pulls = 3
    try:
        every = max(1, int(_env.get("FUXI_EVO_REFINER_EVERY", "5") or 5))
    except Exception:
        every = 5
    pulls = int(pulls or 0)
    mature = pulls >= min_pulls
    _base = {"mode": mode, "min_pulls": min_pulls, "every": every}
    if mode == "off":
        return {**_base, "mature": mature, "probe": False, "reason": "switch=off"}
    if generator == "gp_evolve":
        return {**_base, "mature": mature, "probe": False,
                "reason": "主生成器已是 gp_evolve (避免重复)"}
    if not mature:
        return {**_base, "mature": False, "probe": False,
                "reason": f"方向未成熟 (pulls={pulls} < {min_pulls})"}
    if pulls % every != 0:
        return {**_base, "mature": True, "probe": False,
                "reason": f"周期未命中 (pulls={pulls} % {every} != 0)"}
    if mode == "shadow":
        return {**_base, "mature": True, "probe": False,
                "reason": "shadow (只读不执行)"}
    return {**_base, "mature": True, "probe": True,
            "reason": "on + 成熟 + 周期命中"}


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
            diversity_w=_cfg_div_discount_w(),   # P-20260831 P1: 多样性折扣 (0=关闭)
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
        # 2026-09-01: 跨进程持久化 — 原实现每次进程启动 streak=0,
        #   每轮独立进程 → 退避逃生舱永不触发 (8-31 实证: forge 连续 4 轮
        #   S5=0 仍未强制 LLM, 每日自动化每轮新进程)。现落盘
        #   data/breed_fail_streak.json, 跨轮累积。
        self._breed_fail_streak = self._load_breed_fail_streak()
        self._last_generator = None       # 上一轮使用的生成器
        self._breed_fail_threshold = 2    # 连续失败阈值 → 触发 LLM 退避
        # P2-1 (M5, 2026-09-28): 空轮计数器 — 独立于经济失败连击。
        # 病灶: 空轮(gen=0)被 s5_streak_event 归 unassessed ⇒ 连击不增
        # (自 09-24 streak=0; R3-R5 连续 3 空轮仍未触发退避)。语义见
        # serve_aware 模块 docstring; 开关默认 shadow ⇒ 本节零回归。
        self._breed_empty_streak = self._load_breed_empty_streak()
        self._breed_empty_threshold = self._load_breed_empty_threshold()
        self._last_llm_paradigm = None    # LLM 退避时使用的范式

        # v0.8: 种子主体重检 — 种子原式混入验证管线的配额与冷却
        self._max_seed_recheck = 5        # 每轮最多混入的种子数
        self._seed_recheck_cooloff_days = 30  # 同一种子重检冷却期

    # ── 2026-09-01: 退避计数跨进程持久化 ──────────────────────
    def _breed_streak_path(self) -> Path:
        isolated = str(os.environ.get("FUXI_BREED_FAIL_STATE_PATH", "") or "").strip()
        if isolated:
            path = Path(isolated)
            return path if path.is_absolute() else Path(str(self.data_dir)).parent / path
        return DATA_DIR / "breed_fail_streak.json"

    def _load_breed_fail_streak(self) -> int:
        """读入历史连续失败轮数 (异常/缺失 → 0)。"""
        try:
            p = self._breed_streak_path()
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    v = int(data.get("streak", 0))
                    return max(0, v)
        except Exception:
            pass
        return 0

    def _save_breed_fail_streak(self):
        """落盘连续失败轮数 (异常静默, 不阻塞主流程)。"""
        try:
            p = self._breed_streak_path()
            payload = {"streak": int(self._breed_fail_streak),
                       "last_generator": self._last_generator,
                       "threshold": self._breed_fail_threshold,
                       # P2-1 (M5): 空轮连击与阈值 (与 fail 计数同宿;
                       # 旧读者只取 streak 键, 逐键兼容)。
                       "empty_streak": int(getattr(self, "_breed_empty_streak", 0) or 0),
                       "empty_threshold": int(getattr(self, "_breed_empty_threshold", 3) or 3),
                       "updated_at": datetime.now().isoformat()}
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, p)
        except Exception:
            pass

    # ── P2-1 (M5, 2026-09-28): 空轮连击 (逃生舱补空轮) ─────────────
    def _load_breed_empty_streak(self) -> int:
        """读入历史连续空轮数 (异常/缺失 → 0)。与 fail 计数同宿一文件。"""
        try:
            p = self._breed_streak_path()
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return max(0, int(data.get("empty_streak", 0) or 0))
        except Exception:
            pass
        return 0

    def _load_breed_empty_threshold(self) -> int:
        """空轮退避阈值 (env FUXI_BREED_FAIL_EMPTY_THRESHOLD 可覆盖;
        默认 3 — 比经济失败阈值 2 稍宽, 因空轮含结构性因素, 首触发观察为主)。"""
        try:
            from serve_aware import resolve_breed_empty_threshold
            return resolve_breed_empty_threshold()
        except Exception:
            return 3

    def _breed_empty_trigger_active(self) -> bool:
        """P2-1 (M5): 空轮逃生舱是否参与本轮退避触发。

        off/shadow 档恒 False (零回归); on 档且 empty_streak >= threshold
        时为 True, 由 run() 两处退避检查并入触发条件。
        """
        try:
            from serve_aware import resolve_breed_empty_mode
            if resolve_breed_empty_mode() != "on":
                return False
            return self._breed_empty_streak >= self._breed_empty_threshold
        except Exception:
            return False

    # ═══════════════════════════════════════════════════════════
    # v0.6 实验接线 (LLM_QUANT_FACTORY 评价宪法借鉴)
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

        # P1-2: 方向微战役 — 只记账本。🔴 D5(2026-09-26): freeze/block 结构不可达
        #        (resolve 缺 public_improved 输入 ⇒ 战役永不推进/冻结); 此处不是门禁
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

        # P2: 族级预算 — 只读软警告 (不硬拦截, 避免误杀; 数据仅供监控)
        # 🔴 D5(2026-09-26): 开关改名 budget_ledger_monitor_enabled(默认 False); consume() 未接线 ⇒ 无数据可监控
        if V06_EXPERIMENTAL.get("budget_ledger_monitor_enabled", False):
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
               "behavior_would_reject": 0, "behavior_gate_mode": "off",
               "holdout_verdicts": 0, "incremental_checked": 0,
               "incremental_rejected": 0,
               # M13 (2026-09-27): 快照/分母/队列/异常显式落盘
               "incremental_status": "not_evaluated",
               "incremental_control": {},
               "incremental_coverage": {},
               "incremental_deferred": {},
               "incremental_exceptions": 0,
               # P-20260915-014: 门禁有效性显式落盘 (防"零评估"被误读为"零违规")。
               # True 当且仅当: 库指纹非空 且 本轮至少评估了 1 个候选信号。
               "behavior_gate_effective": False,
               "notes": []}
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
                        "[v0.6-行为簇] ⚠️ 门禁未生效: 库指纹为空 → 本轮零评估。"
                        "接线状态 FUXI_BEHAVIOR_GATE_WIRE="
                        f"{_bh_wire_mode()} "
                        "(off ⇒ 预期零评估; shadow/on 仍空 ⇒ 接线故障, 需排查 "
                        "S2 面板捕获与 signal 注入)")
                _bhw = _bh_wire_mode()
                out["behavior_gate_mode"] = _bhw
                _n_bh_eval = 0
                for cand, res in zip(candidates, results):
                    if not getattr(res, "s1_passed", False):
                        continue
                    sig = cand.get("signal")
                    if sig is None:
                        continue
                    _n_bh_eval += 1
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
                        # P-20260917-024: shadow = 只记账不剥夺 (为 enforce 攒误杀读数)
                        if _bhw == "on":
                            out["behavior_rejected"] += 1
                            cand["_behavior_rejected"] = True
                            if getattr(res, "eligible_for_jq", False):
                                res.eligible_for_jq = False
                        else:
                            out["behavior_would_reject"] += 1
                            cand["_behavior_would_reject"] = True
                bh.save()  # 库指纹跨轮持久化
                # P-20260915-014: 有效性判定 (库非空 且 至少评估 1 个信号)
                if (n_loaded + n_added) > 0 and _n_bh_eval > 0:
                    out["behavior_gate_effective"] = True
                if out["behavior_rejected"]:
                    out["notes"].append(
                        f"[v0.6-行为簇] 硬拒 {out['behavior_rejected']} 个 NEAR_DUPLICATE "
                        f"(库={n_loaded + n_added} 指纹)")
                if out["behavior_would_reject"]:
                    out["notes"].append(
                        f"[v0.6-行为簇/shadow] 本应硬拒 {out['behavior_would_reject']} 个 "
                        f"(库={n_loaded + n_added} 指纹, 本轮评估 {_n_bh_eval} 个候选) "
                        f"⇒ 未剥夺 eligible (FUXI_BEHAVIOR_GATE_WIRE=on 才生效)")
                if _n_bh_eval > 0:
                    out["notes"].append(
                        f"[v0.6-行为簇] 接线有效: 库 {n_loaded + n_added} 指纹 / "
                        f"评估 {_n_bh_eval} 候选 (mode={_bhw})")
                else:
                    # P-20260917-024 加固: 库非空但 0 候选可评估 = **覆盖缺口**, 必须显式可见。
                    # 原修复只覆盖「库空 ⇒ 零评估」; 若库非空而候选侧拿不到 signal,
                    # 门禁对候选**不设防**却无任何输出 ⇒ 与被修的 fail-open 同型 (静默)。
                    if (n_loaded + n_added) > 0 and _bhw != "off":
                        out["notes"].append(
                            f"[v0.6-行为簇] ⚠️ 覆盖缺口: 库 {n_loaded + n_added} 指纹 但 **0 候选被评估** "
                            f"(候选侧无 signal ⇒ 公式无法求值 / 注入失败) ⇒ 门禁对候选不设防 "
                            f"(mode={_bhw})")
                    print(f"  [v0.6-行为簇] ⚠️ 覆盖缺口: 库 {n_loaded + n_added} 指纹 / 评估 0 候选 "
                          f"(mode={_bhw}) — 门禁对候选不设防")
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
                    # R2 (2026-09-22): a full S5 vector is already observed and
                    # therefore is not a sealed holdout. Only a boundary-produced
                    # payload may enter blind evaluation.
                    ret = cand.get("_sealed_holdout_returns")
                    if ret is None:
                        cand["_holdout_eval_status"] = "not_evaluated"
                        continue
                    verdict = hb.blind_evaluate(
                        ret,
                        factor_id=cand.get("factor_name", "?"),
                        generation=gen_id,
                        periods_per_year=(
                            52 if cand.get("natural_freq") == "weekly" else 252
                        ),
                    )
                    cand["_holdout_verdict"] = verdict.to_dict() if hasattr(verdict, "to_dict") else str(verdict)
                    cand["_holdout_eval_status"] = "valid"
                    out["holdout_verdicts"] += 1
                hb.save_budget()
            except Exception:
                pass

        # P1-1: 配对增量门禁 (v0.6.1 接线: S5JointFilter 组合通道)
        # 🔴 M13 (2026-09-27) 修复四处静默:
        #   ① control 从"dict 遍历序前 8 条"改为**固定版本快照**
        #      (control_snapshot; 库变了也保持冻结, 漂移只记不切 ⇒ 漂移次数应为 0);
        #   ② max_pair 3 → 可配 (默认 12); 被预算挡下的候选显式落 not_evaluated
        #      并进补数队列 (不再从输出消失);
        #   ③ 异常不再"降级放行": 标 not_evaluated + 计数 + 队列;
        #   ④ 期数不足不再"豁免通过": 五态 data_insufficient + deferred。
        #   门关 = 零行为变化 (连候选都不触碰)。
        if V06_EXPERIMENTAL.get("incremental_margin_enabled", False):
            try:
                from incremental_pairing import run_incremental_pairs
                _inc = run_incremental_pairs(
                    candidates, results,
                    lib_factors=lib_factors, s5_filter=s5_filter,
                    cfg=V06_EXPERIMENTAL,
                )
                out["incremental_checked"] += int(_inc.get("checked", 0) or 0)
                out["incremental_rejected"] += int(_inc.get("rejected", 0) or 0)
                out["incremental_status"] = _inc.get("status", "not_evaluated")
                out["incremental_control"] = _inc.get("control", {}) or {}
                out["incremental_coverage"] = _inc.get("coverage", {}) or {}
                out["incremental_deferred"] = _inc.get("deferred", {}) or {}
                out["incremental_exceptions"] += int(_inc.get("exceptions", 0) or 0)
                for _n in (_inc.get("notes") or []):
                    out["notes"].append(_n)
            except Exception as e:
                # 不得静默放行: 整体异常 ⇒ eligible 候选标 not_evaluated + 队列 + 计数
                out["incremental_exceptions"] += 1
                out["incremental_status"] = "error"
                _deferred_items = []
                for _i, _r in enumerate(results):
                    if not getattr(_r, "eligible_for_jq", False):
                        continue
                    _cand = candidates[_i]
                    _cand["_incremental_eval_status"] = "not_evaluated"
                    _deferred_items.append({
                        "factor_id": str(_cand.get("factor_name", "?")),
                        "status": "not_evaluated",
                        "reason": f"incremental_pairing_error: {type(e).__name__}: {e}",
                    })
                out["incremental_deferred"] = {
                    "n_deferred": len(_deferred_items),
                    "n_listed": len(_deferred_items),
                    "truncated": 0,
                    "items": _deferred_items,
                }
                out["notes"].append(
                    f"[v0.6-增量] 门禁异常 ⇒ {len(_deferred_items)} 个 eligible 候选标 "
                    f"not_evaluated + 进补数队列 (不记为通过, 不剥夺 eligible): {e}")

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
        # 2026-09-12 (W1) forge 产量旋钮 (默认值 = 原硬编码行为, 零回归)
        forge_generations: int = 5,   # 原 ralph_loop.py 内硬编码 n_generations=5
        forge_topk: int = 1,          # 原恒 order[:1]; >1 = 每代记录 k 条
        forge_mine_top_n: int = 0,    # >0 = 额外并入 forge.final_population 前列
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
        self._active_decision_id = "decision_" + uuid.uuid4().hex
        t0 = time.time()

        # P-20260903-001: natural_freq 审计前置 (铁证才修正, 失败不阻塞)
        try:
            from freq_audit import run_freq_audit
            run_freq_audit(verbose=True)
        except Exception:
            pass

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
            # M9: 库级 FSA 判据可达性诊断 (只读; share 不可达时显式暴露
            # "空转非无同质", 不再是静默 0)
            try:
                _fsa_ctx_st = self.fsa.get_context_status()
                retrieve_result["fsa_context_status"] = _fsa_ctx_st
                _reach = _fsa_ctx_st.get("reachable")
                if _reach is False:
                    print(f"  [FSA] ⚠️ {str(_fsa_ctx_st.get('diag', ''))[:200]}")
                elif _reach is not None and _fsa_ctx_st.get("n_forbidden", 0) > 0:
                    print(f"  [FSA] mode={_fsa_ctx_st.get('mode')} 冻结 "
                          f"{_fsa_ctx_st.get('n_forbidden')} 条库内骨架")
            except Exception as _fsa_diag_ex:
                print(f"  [FSA] 可达性诊断失败 (不阻塞): {type(_fsa_diag_ex).__name__}")

        # v0.4: 注入 Directed GP 先验到 GPBreeder
        if gp_priors is not None:
            self.breeder.priors = gp_priors
            if gp_priors.has_signal():
                print(f"  [v0.4] Directed GP 先验激活: "
                      f"{len(gp_priors.field_weights)} fields, "
                      f"{len(gp_priors.domain_complementarity)} domain pairs, "
                      f"Thompson: {gp_priors.operator_success}")

        # v0.5.1: 自动退避检查 — 即使显式指定 generator 也检查
        # P2-1 (M5, 2026-09-28): 并入空轮逃生舱 — 连续空轮(gen=0)此前
        # 永不触发 (unassessed 盲区); 仅在 FUXI_BREED_FAIL_EMPTY=on 时
        # _breed_empty_trigger_active() 才可能为 True (默认 shadow 恒 False)。
        breed_like_generators = {"gp_breed", "forge", "gp_evolve", "cross_breed"}
        _empty_trig = self._breed_empty_trigger_active()
        if (generator in breed_like_generators 
            and (self._breed_fail_streak >= self._breed_fail_threshold or _empty_trig)):
            _trig_why = (f"连续 {self._breed_fail_streak} 轮 S5=0"
                         if self._breed_fail_streak >= self._breed_fail_threshold
                         else f"连续 {self._breed_empty_streak} 轮空轮(gen=0)")
            print(f"  [退避] {generator} {_trig_why} → "
                  f"强制切换到 LLM (忽略 --generator {generator})")
            generator = "llm"
            # 选择未覆盖范式
            uncovered = ["行业轮动", "北向资金", "两融信号", "大宗交易", "高频微观结构", "跨资产联动"]
            if paradigm in (None, ""):
                paradigm = uncovered[0]
                self._last_llm_paradigm = paradigm

        # v3.1: MAB 方向选择 + 生成器推荐
        # S2-a (2026-09-12): 记录本轮 MAB 方向, 供精炼器补路使用 (见下方 Refiner)。
        _refine_dir = None
        if paradigm is None or paradigm == "":
            selected_dir = self._mab_select_direction()
            if selected_dir:
                _refine_dir = selected_dir
                paradigm = selected_dir.paradigm or paradigm
                # v0.5.1: 自动退避 — 连续 gp_breed/forge 失败 → 强制切换 LLM
                # P2-1 (M5): 并入空轮逃生舱 (on 时生效; shadow/off 零回归)。
                if (self._breed_fail_streak >= self._breed_fail_threshold
                        or self._breed_empty_trigger_active()):
                    _mab_trig_why = (f"连续 {self._breed_fail_streak} 轮 S5=0"
                                     if self._breed_fail_streak >= self._breed_fail_threshold
                                     else f"连续 {self._breed_empty_streak} 轮空轮(gen=0)")
                    print(f"  [退避] gp_breed/forge {_mab_trig_why} → 强制切换 LLM")
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
                    # P2-1 (M5, 2026-09-28): 服务矩阵感知 — 无 Forge profile
                    # 的范式路由到 forge 必空转 (R3-R5 实证) ⇒ 按开关降级。
                    generator = self._serve_aware_adjust(selected_dir, generator)
                print(f"  [MAB] 方向={selected_dir.name} paradigm={paradigm} "
                      f"generator={generator} pulls={selected_dir.pulls}")

        # P-20260904-002: weekly lane 机制化补给 — 累计每 WEEKLY_FORCE_INTERVAL 轮
        # 强制一次周频语境 LLM 主生成 (无显式 --freq 时)。开关=config, 0 关闭回退。
        # 计数用独立状态文件 data/weekly_supply_state.json: checkpoint.start_new 每轮
        # 清零 iteration, 进程内 stats 不跨轮, 须独立落盘累计。
        if not freq:
            try:
                from config import WEEKLY_FORCE_INTERVAL as _wfi
                _wfi = int(_wfi or 0)
            except Exception:
                _wfi = 0
            if _wfi > 0 and generator == "llm":
                _n_loops = self._load_weekly_supply_count()
                if _n_loops % _wfi == 0:
                    freq = "weekly"
                    print(f"  [WeeklySupply] 累计第 {_n_loops} 轮 → 强制周频语境主生成 "
                          f"(P-20260904-002)")

        # ── Phase G: Generate (v0.3: +GP breed + reviewer) ───
        print(f"[G] Generating candidates (generator={generator})...")
        # S2-b (2026-09-12): 每轮起手清空轨迹通道 ⇒ 防上一轮 (gp_evolve) 的
        # 轨迹在本轮 (llm/forge) 被 _extract_trajectories 误读为"本轮产物"。
        self._last_evo_trajectories = []
        self._last_evo_downgrade = None
        generate_result = self._phase_generate(
            candidates, retrieve_result, generator, generator_fn,
            paradigm, max_candidates,
            use_fsa=use_fsa, use_reviewer=use_reviewer,
            gp_templates=gp_templates, evo_turns=evo_turns,
            freq=freq,  # P-20260830-001 (2026-09-01 修复漏传 → LLM 路径 NameError)
            # 2026-09-12 (W1): forge 产量旋钮透传
            forge_generations=forge_generations,
            forge_topk=forge_topk,
            forge_mine_top_n=forge_mine_top_n,
        )

        # ══════════════════════════════════════════════════════════
        # S2-a (2026-09-12): gp_evolve 精炼器补路 (可达性修复)
        # ══════════════════════════════════════════════════════════
        # 根因 (决定性证据): 近 40 轮结果文件 `source` 仅 {forge:31, llm:9},
        # evo_trajectories 全空 ⇒ 轨迹级进化在**真实流水线中从未被执行过**。
        # 原因: 自动化恒显式传 `--generator forge`, 而 `_mab_select_generator`
        # 只在 `generator in (None,"","gp_breed")` 时被调用 ⇒ 整条生成器选择
        # 路径 (含 gp_evolve) 被绕过。修开关也无用 —— 必须在**主生成器之外**
        # 增设补路 (与"gp_evolve = 精炼器而非平级引擎"的语义一致):
        # 主路照常出候选, 精炼器在此基础上追加"进化子代", 同批走 S1–S5。
        # 这样零额外裁决成本 (不新增 e/е 阶段), 且轨迹通道不再为空。
        #
        # 触发条件 (全部满足):
        #   · FUXI_EVO_REFINER == on (shadow 只打印读数, 零回归)
        #   · 主生成器不是 gp_evolve (避免重复)
        #   · 方向成熟: pulls >= FUXI_EVO_REFINER_MIN_PULLS (默认 3)
        #   · 周期门: pulls % FUXI_EVO_REFINER_EVERY == 0 (默认 5)
        try:
            _rf_dir = _refine_dir
            if _rf_dir is None and paradigm:
                _cand_arms = [a for a in self.mab.resolve_arms(paradigm)
                              if a.status != "cooling"]
                if _cand_arms:
                    _rf_dir = max(_cand_arms, key=lambda a: a.pulls)
            _rf_pulls = int(getattr(_rf_dir, "pulls", 0) or 0) if _rf_dir else 0
            _rf_gate = _s2a_refiner_gate(generator, _rf_pulls)
            # M6-A (2026-09-28): 影子遥测本地暂存 (块末统一落盘, 保证每轮一行)
            _rf_telem = {"n_seeds": None, "n_kids": None,
                         "seed_sources": "", "downgrade": ""}
            _rf_n_main_before = len(generate_result.get("candidates") or [])
            if _rf_gate["mode"] == "off":
                pass
            elif _rf_gate["reason"].startswith("shadow") or (
                    _rf_gate["mode"] == "shadow" and _rf_gate["mature"]):
                # shadow: 只打印读数 (零回归)。仅当"若无开关本应触发"时才刷屏。
                if _rf_gate["mature"] and _rf_pulls % _rf_gate["every"] == 0:
                    print(f"  [S2a/shadow] 精炼器补路本应触发: "
                          f"方向={_rf_dir.name} pulls={_rf_pulls} "
                          f"(>= {_rf_gate['min_pulls']}, %{_rf_gate['every']}==0) "
                          f"→ 追加 gp_evolve 子代 (FUXI_EVO_REFINER=on 启用)")
            elif _rf_gate["probe"]:
                # 精炼种子 = 本轮主路产出 (优先) + 检索到的成功模板 (补足 ≥2)
                _rf_templates = []
                _seen_f = set()
                _rf_main_src = list(generate_result.get("candidates") or [])[:3]
                _rf_src = (_rf_main_src
                           + list(gp_templates or
                                  retrieve_result.get("success_templates", [])))
                _rf_n_retr = max(0, len(_rf_src) - len(_rf_main_src))
                for _t in _rf_src:
                    if not isinstance(_t, dict):
                        _t = {"formula": getattr(_t, "formula", ""),
                              "factor_name": getattr(_t, "pattern_id", "?")}
                    _f = (_t.get("formula") or _t.get("expression") or "").strip()
                    if _f and _f not in _seen_f:
                        _seen_f.add(_f)
                        _rf_templates.append({
                            "formula": _f,
                            "factor_name": _t.get("factor_name", _t.get("pattern_id", "?")),
                        })
                _rf_telem["n_seeds"] = len(_rf_templates)
                _rf_telem["seed_sources"] = (
                    f"main:{len(_rf_main_src)}|retr:{_rf_n_retr}"
                    f"|dedup:{len(_rf_templates)}")
                if not _rf_templates:
                    print("  [S2a] 精炼器补路: 无可用种子 (主路产出与检索模板均空)")
                else:
                    _rf_ret = self._phase_generate_evo(
                        templates=_rf_templates,
                        retrieve_result=retrieve_result,
                        paradigm=paradigm,
                        max_candidates=max(3, int(max_candidates * 0.5)),
                        max_turns=evo_turns,
                        use_fsa=use_fsa,
                    )
                    _rf_kids = _rf_ret[1] or []
                    _rf_telem["n_kids"] = len(_rf_kids)
                    if len(_rf_ret) > 2 and _rf_ret[2]:
                        self._last_evo_trajectories = _rf_ret[2]
                    if _rf_kids:
                        generate_result["candidates"] = (
                            list(generate_result.get("candidates") or []) + list(_rf_kids))
                        generate_result["n_candidates"] = len(generate_result["candidates"])
                        print(f"  [S2a] 精炼器补路: gp_evolve 追加 {len(_rf_kids)} 子代 "
                              f"(方向={_rf_dir.name} pulls={_rf_pulls}), 同批走 S1–S5")
                    else:
                        _rf_telem["downgrade"] = str(
                            getattr(self, "_last_evo_downgrade", None) or "")
                        print(f"  [S2a] 精炼器补路: gp_evolve 未产子代 (降级 "
                              f"{getattr(self, '_last_evo_downgrade', None)})")
            # M6-A (2026-09-28): 门控遥测落盘 (每轮一行 ⇒ 触发率分母恒存在)。
            # 纯工程读数, 非经济证据; 不碰 MAB / 不碰任何裁决字段。
            try:
                from s2a_refiner_shadow import (
                    s2a_refiner_refine_telemetry as _rf_telem_write)
                _rf_telem_rep = _rf_telem_write(
                    gate=_rf_gate, run_id=loop_id,
                    direction=(getattr(_rf_dir, "name", "") if _rf_dir else ""),
                    paradigm=paradigm or "", generator=generator or "",
                    pulls=_rf_pulls, n_seeds=_rf_telem.get("n_seeds"),
                    n_kids=_rf_telem.get("n_kids"),
                    n_main_candidates=_rf_n_main_before,
                    seed_sources=_rf_telem.get("seed_sources", ""),
                    downgrade=_rf_telem.get("downgrade", ""))
                for _rf_telem_line in _rf_telem_rep.get("lines", []):
                    print(_rf_telem_line)
            except Exception as _rf_te:
                print(f"  [S2a] ⚠️ 精炼器遥测跳过 (不阻塞): {_rf_te}")

        except Exception as _rfe:
            print(f"  [S2a] ⚠️ 精炼器补路跳过 (不阻塞): {_rfe}")
            # M6-A (2026-09-28): 异常轮也落一行 (不静默; 分母含 error 轮)
            try:
                from s2a_refiner_shadow import (
                    s2a_refiner_refine_telemetry as _rf_telem_write2)
                _rf_telem_rep2 = _rf_telem_write2(
                    gate=locals().get("_rf_gate"), run_id=loop_id,
                    direction=(getattr(locals().get("_rf_dir"), "name", "")
                               if locals().get("_rf_dir") else ""),
                    paradigm=paradigm or "", generator=generator or "",
                    pulls=int(locals().get("_rf_pulls", 0) or 0), error=str(_rfe))
                for _rf_telem_line2 in _rf_telem_rep2.get("lines", []):
                    print(_rf_telem_line2)
            except Exception as _rfe3:
                print(f"  [S2a] ⚠️ 异常轮遥测写入失败 (不阻塞): {_rfe3}")

        # ── P-20260907-004: 周频原生模板注入 (weekly 补给轮) ──
        # 周频补给轮 (freq=="weekly") 在 Phase G 产出之外, 追加 config 中的
        # 周频原生模板候选 (周收益偏度/周内高低位序/周换手一致性等),
        # natural_freq=weekly → 走周频裁决器 (τ_w)。开关 config, 失败静默回退。
        _inj_n = 0
        if freq == "weekly":
            try:
                from config import WEEKLY_NATIVE_TEMPLATES as _wnt
                from config import WEEKLY_NATIVE_TEMPLATE_INJECT as _wnt_on
            except Exception:
                _wnt, _wnt_on = [], False
            if _wnt_on and _wnt:
                _existing = {str(c.get("factor_name", "")) for c in
                             generate_result.get("candidates", [])}
                for _t in _wnt:
                    if str(_t.get("factor_name", "")) in _existing:
                        continue  # 防重复注入 (同轮 LLM 已产出同名候选)
                    _cand = {
                        "factor_name": _t.get("factor_name"),
                        "formula": _t.get("formula", ""),
                        "hypothesis": _t.get("hypothesis", ""),
                        "paradigm": _t.get("paradigm", "周频原生"),
                        "natural_freq": "weekly",
                        "_source": "weekly_native_template",
                        "_memory_priors": retrieve_result["llm_prompt_context"][:2000],
                    }
                    generate_result["candidates"].append(_cand)
                    _inj_n += 1
                generate_result["n_candidates"] = len(generate_result["candidates"])
                if _inj_n:
                    print(f"  [WeeklyNative] P-20260907-004 注入 {_inj_n} 条周频原生模板候选")

        # ── Phase E: Evaluate ─────────────────────────────
        for _candidate in generate_result["candidates"]:
            _candidate["_selected_generator"] = generator
            _candidate["_research_run_id"] = loop_id
            _candidate["_research_decision_id"] = self._active_decision_id
        print(f"[E] Evaluating {len(generate_result['candidates'])} candidates...")
        evaluate_result = self._phase_evaluate(
            generate_result["candidates"],
            library_signals,
            library_factors,
        )
        # Candidate-level observations retain the actual engine and measurement
        # states. Proxy IC is deliberately excluded from economic reward.
        try:
            from learning_feedback import candidate_feedback
            evaluate_result["learning_feedback"] = [
                candidate_feedback(_c, _r)
                for _c, _r in zip(generate_result["candidates"],
                                  evaluate_result.get("results", []))
            ]
        except Exception as _feedback_error:
            evaluate_result["learning_feedback_error"] = str(_feedback_error)[:200]

        # ── S2-a (2026-09-12) / M2 (2026-09-27): 生成器轴奖励写入 ──────────
        # 根因: `generator_rewards` 全库无写入点 ⇒ `_mab_select_generator` 的
        # "基于历史表现选生成器" 恒退化 (见 MABScheduler.record_generator_reward)。
        # 口径与 evaluate 内 L2 本地信号一致 (ICIR→tanh 压缩, 上限 ±0.3),
        # 故与 JQ 终裁 (±1.0) 分属不同量级, 仅作引擎选择的引导。
        #
        # 🔴 M2 (P-20260927-005) 修复两处旧缺陷:
        #   (a) 旧注释写"写入恒开", 实现却默认 raise 后 except 静默吞掉 ⇒
        #       注释与实现矛盾, 且"关"状态不可见 (E1 证据);
        #   (b) 写入与 FUXI_EVO_REFINER 缠绕, 无独立开关。
        # 现写入逻辑移交 generator_axis.s2a_generator_axis_write, 三态:
        #   off → 显式跳过; shadow(默认) → 追加观测
        #   (data/generator_axis_write_shadow.jsonl, 绝不动 MAB 状态);
        #   on → 写 MAB (旧别名 FUXI_LEGACY_GENERATOR_PROXY_REWARD=on 兼容)。
        # 读端开关 FUXI_GENERATOR_AXIS 见 _mab_select_generator, 两端解耦。
        try:
            from generator_axis import s2a_generator_axis_write as _s2a_ga_write
            _s2a_ga_rep = _s2a_ga_write(
                self.mab, generator, paradigm, evaluate_result, loop_id=loop_id)
            for _s2a_ga_line in _s2a_ga_rep.get("lines", []):
                print(_s2a_ga_line)
        except Exception as _ge:
            print(f"  [S2a] ⚠️ 生成器轴写入块异常 (不阻塞, 显式报告): {_ge}")

        # ── M6-A (2026-09-28): S2-a 精炼器子代归因遥测 ──────────────
        # 仅当批次内含 `_source=gp_evolve` 候选时落行 (主路/子代分开计数,
        # S1 分母带 read/refused), 为 FUXI_EVO_REFINER 的 A/B 提供同口径读数端点。
        # 边界: 纯工程读数, 非经济证据; 不动 MAB / 不动任何裁决字段。
        try:
            from s2a_refiner_shadow import (
                s2a_refiner_eval_telemetry as _rf_eval_write)
            _rf_eval_rep = _rf_eval_write(
                generate_result.get("candidates") or [],
                evaluate_result.get("results") or [],
                run_id=loop_id, paradigm=paradigm or "",
                generator=generator or "")
            for _rf_eval_line in _rf_eval_rep.get("lines", []):
                print(_rf_eval_line)
        except Exception as _rfe2:
            print(f"  [S2a] ⚠️ 子代归因遥测跳过 (不阻塞): {_rfe2}")

        # ── Phase D: Distill ──────────────────────────────
        print(f"[D] Distilling experience...")
        distill_result = self._phase_distill(
            generate_result["candidates"],
            evaluate_result,
            paradigm,
        )

        # ── P-20260910-007: 范式提案准入判定 (shadow 默认零副作用) ──
        try:
            self._phase_paradigm_proposal(
                generate_result["candidates"], evaluate_result, paradigm)
        except Exception as e:
            print(f"  [P-007] 范式提案判定失败 (非阻塞): {e}")

        # ── P-20260912-001: CogAlpha 认知进化 — 记录本轮父本/归因供下一轮 ──
        # shadow(默认) 只落盘 + 打印读数; enforce 时下一轮生成会注入提示块。
        try:
            import llm_evolution as _cog
            _cr = _cog.record_round(
                int(self.stats.get("total_loops", 0) or 0),
                evaluate_result.get("results", []), verbose=True)
            evaluate_result["cog_evolve"] = _cr
        except Exception as _ce:
            print(f"  [P-001] 认知进化记录失败 (非阻塞): {_ce}")

        # ── P-20260914-003: 经济学先验门槛 — 影子打标 (默认 shadow, 零裁决) ──
        try:
            import prior_gate as _pg
            evaluate_result["prior_gate"] = _pg.run_prior_gate(
                generate_result.get("candidates", []), evaluate_result,
                self.data_dir, generator)
        except Exception as _pe:
            print(f"  [PriorGate] 失败 (非阻塞): {_pe}")

        # ── P-20260915-005: AST 结构去重 — 影子读数 (默认 shadow, 零裁决) ──
        try:
            import structure_dedup as _sd
            _sdr = _sd.shadow_readout(generate_result.get("candidates", []))
            evaluate_result["struct_dedup"] = _sdr
            if _sdr.get("readout") and _sdr.get("mode") != "off":
                print(f"  {_sdr['readout']}")
        except Exception as _sde:
            print(f"  [StructDedup] 失败 (非阻塞): {_sde}")

        # ── Summary ────────────────────────────────────────
        elapsed = time.time() - t0
        eligible = [r for r in evaluate_result["results"] if r.eligible_for_jq]
        s5_passed = evaluate_result.get("stage5_passed", 
                    sum(1 for r in evaluate_result["results"] if getattr(r, 'grade', 'D') in ('A', 'B', 'C')))

        # v0.5.1: 生成器自动退避 — 更新连续失败计数
        s5_count = evaluate_result.get("stage5_passed", 0)
        # 也检查 total_candidates vs eligible
        n_candidates = len(generate_result.get("candidates", []))
        # 2026-09-01 修复: ValidationResult 字段是 s1_passed (原 stage1_passed 恒 False
        # → n_passed_any_stage 恒 0 → LLM 轮退避计数永不重置, 退避逃生舱触发后无法回 MAB)
        n_passed_any_stage = sum(1 for r in evaluate_result.get("results", []) 
                                 if getattr(r, 's1_passed', False))
        
        # ── P2-1 (M5): 空轮归因 (纯观测, 默认 shadow 落盘; 不动任何裁决) ──
        try:
            from serve_aware import record_empty_round
            record_empty_round(
                loop_id=loop_id, paradigm=paradigm or "",
                generator=generator,
                source=str(generate_result.get("source", "") or ""),
                n_candidates=n_candidates,
                empty_streak=self._breed_empty_streak,
                data_dir=DATA_DIR)
        except Exception as _era:
            print(f"  [M5/空轮归因] 失败 (非阻塞): {type(_era).__name__}")

        if generator in ("gp_breed", "forge", "gp_breed_fallback"):
            from generation_contract import s5_streak_event
            _s5_event = s5_streak_event(evaluate_result.get("results", []))
            if _s5_event == "economic_failure":
                self._breed_fail_streak += 1
                self._save_breed_fail_streak()
                print(f"  [退避] gp_breed/forge 有效 S5 经济失败连续 {self._breed_fail_streak} 轮 "
                      f"(阈值={self._breed_fail_threshold}, 已持久化)")
            elif _s5_event == "pass":
                self._breed_fail_streak = 0
                self._save_breed_fail_streak()
            else:
                print("  [退避] 本轮 S5 未形成有效经济裁决，失败连击不变")

            # ── P2-1 (M5): 空轮计数 (独立于经济失败连击) ──
            # 病灶: s5_streak_event 对「0 候选」返回 unassessed ⇒ 连续空轮
            # (R3-R5 gen=0×3) 一律走上面 else 分支, 永不触发退避。空轮独立
            # 计数, 有产出即重置; on 档才写状态/参与触发 (默认 shadow 只打印)。
            try:
                from serve_aware import (
                    resolve_breed_empty_mode as _rbe,
                    format_empty_readout as _fer)
                _be_mode = _rbe()
                if _be_mode != "off":
                    if n_candidates == 0:
                        if _be_mode == "on":
                            self._breed_empty_streak += 1
                            self._save_breed_fail_streak()
                            print(f"  [退避] 空轮连续 {self._breed_empty_streak} 轮 "
                                  f"(阈值={self._breed_empty_threshold}, 已持久化)")
                        else:
                            print(_fer(self._breed_empty_streak + 1,
                                       self._breed_empty_threshold))
                    elif _be_mode == "on" and self._breed_empty_streak:
                        self._breed_empty_streak = 0
                        self._save_breed_fail_streak()
                        print("  [退避] 本轮有产出 → 空轮连击重置")
            except Exception as _bee:
                print(f"  [退避] 空轮计数失败 (非阻塞): {type(_bee).__name__}")
        elif generator == "llm":
            # LLM 成功后重置 (即使 S5=0, LLM 生成的新因子也算注入新血统)
            if n_passed_any_stage > 0:
                self._breed_fail_streak = 0
                self._save_breed_fail_streak()
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
        # 2026-09-01 修复: cand_map 按 factor_name 匹配失败时 (候选缺 factor_name 键,
        # validator 给过 candidate_N 兜底名), 按 validate 的 enumerate 索引回退取原候选,
        # 并从 expression 双键兜底公式 — 根治 JQ codegen "无公式" 全 SKIP
        cand_map = {c.get("factor_name"): c for c in generate_result["candidates"]}
        gen_cand_list = generate_result["candidates"]
        jq_candidate_details = []
        for r in eligible:
            cand = cand_map.get(r.factor_name, {})
            _fn = str(getattr(r, "factor_name", ""))
            if not cand and _fn.startswith("candidate_"):
                try:
                    _idx = int(_fn.split("_", 1)[1])
                    if 0 <= _idx < len(gen_cand_list):
                        cand = gen_cand_list[_idx]
                except (ValueError, IndexError):
                    cand = {}
            jq_candidate_details.append({
                "factor_name": r.factor_name,
                "formula": (
                    cand.get("formula")
                    or cand.get("expression")
                    or getattr(r, "formula", "")
                ),
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
                # P-20260909-001: 透传 S5 分年超额 (换手率成本影子检查基准)
                "s5_year1_excess": float(getattr(r, "s5_year1_excess", 0) or 0),
                "s5_year2_excess": float(getattr(r, "s5_year2_excess", 0) or 0),
                # ── W5 (2026-09-12): S5 口径留痕 (诊断用, 不影响裁决) ──
                # 原缺陷: 只有单一浮点 s5_calmar, 无口径标注 ⇒ 208 个 result json
                #   中 calmar_mode/window_days_*/window_matched 出现 0 次, "不可比"
                #   只能靠人肉考古。现补 8 键。
                "s5_calmar_mode": getattr(r, "s5_calmar_mode", ""),
                "s5_calmar_2025": float(getattr(r, "s5_calmar_2025", 0) or 0),
                "s5_calmar_2026": float(getattr(r, "s5_calmar_2026", 0) or 0),
                "s5_calmar_2025_full": float(getattr(r, "s5_calmar_2025_full", 0) or 0),
                "s5_calmar_2026_full": float(getattr(r, "s5_calmar_2026_full", 0) or 0),
                "s5_window_days_2025": int(getattr(r, "s5_window_days_2025", 0) or 0),
                "s5_window_days_2026": int(getattr(r, "s5_window_days_2026", 0) or 0),
                "s5_window_matched": getattr(r, "s5_window_matched", None),
                "s5_calmar_threshold_effective": float(
                    getattr(r, "s5_calmar_threshold_effective", 1.0) or 1.0),
            })

        # ── W5 (2026-09-12): 全候选 S5 口径明细 (含**未通过**者) ──
        # 原 `jq_candidate_details` 只覆盖 eligible (S5 通过者) ⇒ 208 个 json 中
        # 仅 11 个非空 ⇒ S5 失败原因无法批量归因。此处对**所有**拿到 S5 结论的
        # 候选落盘口径字段 (纯诊断, 不改变裁决, 不参与任何下游判定)。
        s5_caliber_details = []
        try:
            for r in evaluate_result.get("results", []):
                _mode = getattr(r, "s5_calmar_mode", "")
                if not _mode and not getattr(r, "s5_reason", ""):
                    continue      # 未走到 S5 的候选不落盘
                s5_caliber_details.append({
                    "factor_name": getattr(r, "factor_name", ""),
                    "grade": getattr(r, "final_grade", ""),
                    "s5_passed": bool(getattr(r, "s5_passed", False)),
                    "calmar_mode": _mode,
                    "calmar_2025": float(getattr(r, "s5_calmar_2025", 0) or 0),
                    "calmar_2026": float(getattr(r, "s5_calmar_2026", 0) or 0),
                    "calmar_2025_full": float(getattr(r, "s5_calmar_2025_full", 0) or 0),
                    "calmar_2026_full": float(getattr(r, "s5_calmar_2026_full", 0) or 0),
                    "window_days_2025": int(getattr(r, "s5_window_days_2025", 0) or 0),
                    "window_days_2026": int(getattr(r, "s5_window_days_2026", 0) or 0),
                    "window_matched": getattr(r, "s5_window_matched", None),
                    "calmar_threshold_effective": float(
                        getattr(r, "s5_calmar_threshold_effective", 1.0) or 1.0),
                })
        except Exception as _w5e:
            print(f"  [W5-口径] ⚠️ 全候选口径明细生成失败 (非阻塞): {_w5e}")

        summary = {
            "loop_id": loop_id,
            "elapsed_seconds": round(elapsed, 1),
            "total_candidates": generate_result["n_candidates"],
            "source": generate_result.get("source", "unknown"),
            # P-20260917-028: LLM 降级留痕 (账本可区分「无新假设」与「从未调用 API」)
            "llm_api_ok": generate_result.get("llm_api_ok"),
            "llm_degrade_reason": generate_result.get("llm_degrade_reason", ""),
            "semantic_selection": generate_result.get("semantic_selection"),
            "stages_completed": {
                "retrieve": retrieve_result["n_retrieved"],
                "generate": generate_result["n_candidates"],
                "evaluate": evaluate_result["summary"],
                "distill": distill_result["summary"],
            },
            # P-20260902-002: IC 衰减半衰期影子统计 (仅标注, 不参与裁决)
            "decay_shadow": evaluate_result.get("decay_shadow", {}),
            "fsa_status": self.fsa.get_status() if use_fsa else {},
            "checkpoint_progress": (
                self.checkpoint.get_progress() if use_checkpoint else {}
            ),
            "jq_candidates": [r.factor_name for r in eligible],
            "n_jq_candidates": len(eligible),
            "jq_candidate_details": jq_candidate_details,
            # W5 (2026-09-12): S5 口径留痕 (诊断用; 含未通过候选)
            "s5_caliber_details": s5_caliber_details,
            "s5_caliber_summary": evaluate_result.get("summary", {}).get("s5_caliber", {}),
            # v0.6: EvoTraj 轨迹信息
            "evo_trajectories": self._extract_trajectories(generate_result),
        }

        # 更新统计
        self.stats["total_loops"] += 1
        # P-20260904-002: weekly 补给累计轮次落盘 (跨进程; checkpoint.start_new 会清零迭代)
        self._bump_weekly_supply_count()
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
            # W5 (2026-09-12): S5 口径留痕 (诊断用; 含未通过候选)
            "s5_caliber_details": s5_caliber_details,
            "s5_caliber_summary": evaluate_result.get("summary", {}).get("s5_caliber", {}),
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

        # P-20260910-006: 前沿→路径卡片注入 (知识从信息层升级为方向层)
        # 按"当前范式空间缺口"召回相关卡片, 只作 llm 上下文灵感, 不改 MAB 选择主键。
        # 开关 FRONTIER_CARD_INJECT: "0"=关闭 (退化为现有日报流程) / 默认 "1"=注入
        frontier_card_ctx = ""
        n_frontier_cards = 0
        if os.environ.get("FRONTIER_CARD_INJECT", "1") != "0":
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
                from frontier_path_card import recall_cards, format_for_prompt  # type: ignore
                _uncovered = ["高频微观结构", "北向资金", "跨资产联动", "两融信号", "大宗交易"]
                _gaps = [paradigm] if paradigm else []
                _gaps += [g for g in _uncovered if g not in _gaps]
                _cards = recall_cards(_gaps, k=3)
                if _cards:
                    frontier_card_ctx = format_for_prompt(_cards)
                    n_frontier_cards = len(_cards)
                    print(f"  [P-006] 前沿路径卡片注入: {n_frontier_cards} 张 "
                          f"(按范式缺口召回, 仅上下文不改 MAB 主键)")
            except Exception as e:
                print(f"  [P-006] 路径卡片注入失败 (非阻塞): {e}")

        # 合成统一的 memory priors context
        combined_context = [
            retrieval.get("llm_prompt_fragment", ""),
            "",
            coverage_ctx,
            "",
            forbidden_ctx,
            "",
            frontier_card_ctx,
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
            "llm_prompt_context": "\n".join(c for c in combined_context if c is not None),
            "paradigm_coverage": coverage_ctx,
            "forbidden_context": forbidden_ctx,
            "frontier_card_context": frontier_card_ctx,
            "n_frontier_cards": n_frontier_cards,
        }

    # ═══════════════════════════════════════════════════════════
    # P-20260910-007: 范式提案准入判定
    # ═══════════════════════════════════════════════════════════

    def _recent_high_candidates(self, n: int = 5) -> List[Dict]:
        """近 n 条高 |ICIR| 的尝试记录 (反固着参照池)。"""
        try:
            attempts = self.memory.data.get("attempts", []) or []
            recs = [a for a in attempts if a.get("formula")]
            recs.sort(key=lambda a: -abs(float(a.get("icir") or 0.0)))
            return recs[:n]
        except Exception:
            return []

    def _phase_paradigm_proposal(self, candidates: List[Dict],
                                 evaluate_result: Dict, paradigm: str) -> Dict:
        """对标记为范式提案的候选做 S1 快速证伪 → provisional 准入判定。

        shadow(默认): 判定照跑, 结果只落奇影日志, 不入范式空间 (零副作用)
        enforce     : 满足全部条件则正式登记 provisional 范式
        """
        import paradigm_proposal as _pp  # type: ignore
        mode = _pp.get_mode()
        props = [c for c in (candidates or []) if c.get("paradigm_proposal")]
        if not props:
            return {"mode": mode, "n_proposals": 0, "admitted": 0}
        if mode == "off":
            print("  [P-007] 范式提案模式 off → 跳过准入判定")
            return {"mode": mode, "n_proposals": len(props), "admitted": 0}

        results = {getattr(r, "factor_name", ""): r for r in evaluate_result.get("results", [])}
        recent_high = self._recent_high_candidates(5)
        admitted = 0
        for c in props:
            fname = c.get("factor_name") or c.get("name") or ""
            r = results.get(fname)
            icir = float(getattr(r, "s1_icir", 0.0) or 0.0) if r else 0.0
            s1_ic = float(getattr(r, "s1_ic", 0.0) or 0.0) if r else 0.0
            # 方向正确: 预测方向与 S1 IC 符号一致 (long→ic>0 / short→ic<0)
            want_neg = (c.get("direction") == "short")
            direction_ok = (s1_ic < 0) if want_neg else (s1_ic > 0)
            v = _pp.admit_provisional(c, icir=icir, direction_ok=direction_ok,
                                      recent_high=recent_high, mode=mode)
            if mode == "shadow":
                _pp.record_shadow(v)
            if v.get("admitted"):
                admitted += 1
            tag = "✅准入" if v.get("admitted") else ("◻️拟准入(shadow)" if v.get("shadow_record") else "❌未过")
            print(f"  [P-007] {tag} 新范式='{v.get('name') or '?'}' "
                  f"icir={icir:.3f} dir_ok={direction_ok} | {v.get('reason', '')[:80]}")
        return {"mode": mode, "n_proposals": len(props), "admitted": admitted,
                "shadow_record": mode == "shadow"}

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
        freq: str = "",  # P-20260830-001: "weekly" 时 LLM 周频语境 (2026-09-01 修复漏传 NameError)
        forge_generations: int = 5,   # 2026-09-12 (W1)
        forge_topk: int = 1,          # 2026-09-12 (W1)
        forge_mine_top_n: int = 0,    # 2026-09-12 (W1)
    ) -> Dict:
        """
        G 阶段: 生成候选因子 (v0.5: MAB 方向选择 + 统一 FactorQualityGate)。

        生成方式:
        - "manual": 使用传入候选项
        - "gp_breed": GP 表达式树育种 (五维演化)
        - "cross_breed": 从记忆模板交叉育种
        - "custom": 调用自定义生成函数
        """
        # P-20260917-028: LLM 通道降级留痕 (消除 fail-silent)。
        # 病灶: 旧版 except 分支把 gen_candidates 硬置 [] 且 source 仍写 "llm"
        #   (与 API 成功路径同值) ⇒ 结果账本无法区分「API 通但模型没给新假设」
        #   与「凭据缺失从未调用 API」。实测 2026-09-17 E2E-2 即为此假象。
        # 语义: None=本轮未走 LLM 分支 / True=API 成功 / False=已降级文件模式。
        # 必须在此处初始化: 本 return 由全部 generator 分支共用。
        _llm_api_ok = None
        _llm_degrade_reason = ""
        _semantic_selection = None
        # M7: 批内 AST 去重留档 (Ralph 首轮 + reviewer 二轮; 随 return 落盘)。
        # evidence 记录命中项的指纹/首现者/阶段, 误杀可回放 (去重不删证据)。
        _m7_dedup_evidence = []
        _m7_dedup_mode = ""
        # M9: 统一质量门接线留档 (shadow 记账 / enforce 拒绝; 随 return 落盘)。
        _qgw_mode = ""
        _qgw_records = []
        _qgw_n_checked = 0
        _qgw_n_fatal = 0
        _qgw_n_removed = 0

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

            # Only formula-bearing heads can be GP parents. Resolve samples and
            # retain verification metadata; fill the fixed parent pool beyond k=5.
            from generation_contract import resolve_gp_parent_pool
            templates = resolve_gp_parent_pool(templates, self.memory.data, limit=20)

            # P-20260819-005: 轨迹父本注入 gp_breed 主路径 (2026-08-19 修正:
            # 原注入点仅在 gp_evolve 的 _phase_generate_evo, 主线 gp_breed 从未生效)
            try:
                from trajectory_pool import ENABLE_TRAJECTORY_POOL, get_trajectory_parents
                if ENABLE_TRAJECTORY_POOL:
                    tp_par = get_trajectory_parents(
                        data_dir=self.data_dir,
                        n=min(3, max(1, max_candidates // 3)),
                        paradigm=paradigm,
                        # 2026-09-12: 池内含 rolling_min(...) 等未定义函数公式,
                        # 必须校验 (否则种子位直接 NameError)
                        validator=_find_unknown_identifiers)
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
            # M7: 批内 AST 去重开关 fail-fast 校验。放在 API try 之外 ——
            #   配置错误不得落入 except 降级被误报为 api_error (静默降级)。
            from llm_generator import batch_dedup_mode as _m7_bd_mode
            _m7_dedup_mode = _m7_bd_mode()
            self.llm_gen.receive_context(
                success_templates=retrieve_result.get("success_templates", []),
                forbidden_directions=retrieve_result.get("forbidden_directions", []),
                fsa_forbidden=retrieve_result.get("fsa_forbidden_context", ""),
                library_stats=retrieve_result.get("library_stats", {}),
                paradigm_priorities=retrieve_result.get("exploration_priorities", {}),
                mab_direction=paradigm,
                frontier_cards=retrieve_result.get("frontier_card_context", ""),  # P-20260910-006
            )
            # 注入 v3.1: Warning 方向 (软负收益) + Motif 规则
            # P2 (2026-09-05): consumer_engine="llm" — 跨引擎 forbid 降为软警告
            try:
                warnings = self.memory.get_warning_directions()
                if warnings:
                    self.llm_gen.receive_context(warning_directions=warnings)
                motif_rules = self.memory.get_motif_rules(consumer_engine="llm")
                if motif_rules:
                    self.llm_gen.receive_context(motif_rules=motif_rules)
                    _fc = len(motif_rules.get("forbid_cross", []))
                    _pc = len(motif_rules.get("prefer_cross", []))
                    if _fc or _pc:
                        print(f"  [P2-Engine] LLM 消费 motif 规则: 硬禁止 {len(motif_rules.get('forbid', []))} / "
                              f"跨引擎软警告 {_fc} / 跨引擎低权重偏好 {_pc}")
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

            # P-20260910-007: 范式提案模式 — 每 N=3 轮额外生成 1 条偏离既有 21 范式的候选
            try:
                import paradigm_proposal as _pp  # type: ignore
                _round_idx = int(self.stats.get("total_loops", 0) or 0)
                if _pp.should_propose(_round_idx):
                    _space = _pp.load_space()
                    _knows = set(_space.get("base_paradigms", [])) | set(_space.get("promoted", []))
                    _hint = "；".join([p for p in ("多尺度复杂性", "路径几何", "跨期结构断裂",
                                                  "跨资产联动", "高频微观结构")
                                       if p not in _knows])
                    self.llm_gen.set_proposal_mode(True, hint=_hint)
                    print(f"  [P-007] 范式提案模式 ON (第 {_round_idx} 轮, mode={_pp.get_mode()}, "
                          f"提示: {_hint[:40]}...)")
                else:
                    self.llm_gen.set_proposal_mode(False)
            except Exception as e:
                self.llm_gen.set_proposal_mode(False)
                print(f"  [P-007] 范式提案模式装配失败 (非阻塞): {e}")

            # P-20260912-001: CogAlpha 认知进化 — enforce 才注入, shadow(默认) 只报读数。
            # 父本/归因来自上一轮 record_round 落盘 (data/llm_evolution_state.json)。
            try:
                import llm_evolution as _cog  # type: ignore
                _cog_mode = _cog.get_mode()
                _cog_ctx = _cog.consume_blocks()
                if _cog_mode == "enforce" and _cog_ctx.get("has_content"):
                    self.llm_gen.set_cog_evolve(True, _cog_ctx["blocks"])
                    _ch = sum(len(v) for v in _cog_ctx["blocks"].values())
                    print(f"  [P-001] 认知进化注入: 父本 {_cog_ctx['n_parents']} 条 "
                          f"({_cog_ctx['parent_source']}) + 归因块, 共 {_ch} chars")
                else:
                    self.llm_gen.set_cog_evolve(False)
                    if _cog_mode == "shadow":
                        print(f"  [P-001] CogAlpha shadow: 父本 {_cog_ctx['n_parents']} 条 "
                              f"({_cog_ctx['parent_source']}) / 上轮 "
                              f"#{_cog_ctx['last_round']} (未注入 prompt, 零副作用)")
            except Exception as _coge:
                self.llm_gen.set_cog_evolve(False)
                print(f"  [P-001] 认知进化装配失败 (非阻塞): {_coge}")

            # Plan-level search is opt-in. Its evidence has no promotion authority.
            _plan_mode = os.environ.get("FUXI_SEMANTIC_PLAN", "off").strip().lower()
            if _plan_mode not in ("off", "on", "auto"):
                raise ValueError("FUXI_SEMANTIC_PLAN must be off, on or auto")
            if _plan_mode == "auto":
                # M6 (2026-09-28) 频率匹配前置: 计划库无当前 lane 频率的计划 ⇒
                # 显式降级 off (打印原因, 零注入), 不让整轮生成崩掉;
                # 其它 ValueError (schema/axes/重复 id) 仍向上抛。
                _lane_freq_pre = "weekly" if freq == "weekly" else "daily"
                _plan_file_pre = os.environ.get("FUXI_SEMANTIC_PLAN_FILE")
                _evidence_pre = os.environ.get("FUXI_SEMANTIC_EVIDENCE_FILE")
                if _plan_file_pre and _evidence_pre:
                    try:
                        from mechanism_search import load_catalog as _lc_pre
                        _lc_pre(_plan_file_pre, _lane_freq_pre)
                    except ValueError as _pre_err:
                        if "no plans for requested frequency" in str(_pre_err):
                            print(f"  [SemanticSearch] ⚠️ 计划库无 '{_lane_freq_pre}' "
                                  f"频率计划 ⇒ 本轮显式降级 off (零注入, 生成继续)")
                            _plan_mode = "off"
                        else:
                            raise
            if _plan_mode == "on":
                from semantic_plan import load_plan
                _plan_file = os.environ.get("FUXI_SEMANTIC_PLAN_FILE")
                _plan_id = os.environ.get("FUXI_SEMANTIC_PLAN_ID")
                if not _plan_file or not _plan_id:
                    raise ValueError("semantic plan on requires FILE and ID")
                self.llm_gen.set_semantic_plan(load_plan(
                    _plan_file, _plan_id, "weekly" if freq == "weekly" else "daily"))
            elif _plan_mode == "auto":
                from mechanism_search import load_catalog, load_evidence, select_plan
                _plan_file = os.environ.get("FUXI_SEMANTIC_PLAN_FILE")
                _evidence_file = os.environ.get("FUXI_SEMANTIC_EVIDENCE_FILE")
                if not _plan_file or not _evidence_file:
                    raise ValueError("semantic plan auto requires FILE and EVIDENCE_FILE")
                _plans = load_catalog(
                    _plan_file, "weekly" if freq == "weekly" else "daily")
                _ledger_path = os.environ.get("FUXI_SEMANTIC_DECISION_LEDGER", "").strip()
                _seed = int(os.environ.get("FUXI_SEMANTIC_SEARCH_SEED", "20260925"))
                if _ledger_path:
                    _workspace = Path(self.data_dir).parent.resolve()
                    _ledger = Path(_ledger_path).resolve()
                    if _ledger != _workspace and _workspace not in _ledger.parents:
                        raise ValueError("semantic decision ledger must stay in workspace")
                    # Persisted decisions are opt-in and require the same
                    # single-writer ownership as the rest of the pipeline.
                    from run_contract import LOCK_PATH
                    _owner = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
                    if int(_owner.get("pid", -1)) != os.getpid():
                        raise RuntimeError("semantic decision ledger requires Fuxi writer lock")
                    from learning_event_store import LearningEventStore
                    _store = LearningEventStore(_ledger)
                    _today = datetime.now().date().isoformat()
                    _store.ingest_evidence(
                        _evidence_file, Path(self.data_dir).parent, _plans,
                        observed_on=_today)
                    _selected, _semantic_selection = _store.decide(
                        self._active_decision_id, _plans, as_of=_today,
                        seed=_seed)
                else:
                    _observations = load_evidence(
                        _evidence_file, Path(self.data_dir).parent, _plans)
                    _selected, _semantic_selection = select_plan(
                        _plans, _observations, seed=_seed,
                        round_index=int(self.stats.get("total_loops", 0) or 0),
                    )
                self.llm_gen.set_semantic_plan(_selected)
                print(f"  [SemanticSearch] {_semantic_selection['mode']} → "
                      f"{_selected['plan_id']} "
                      f"(valid_economic={_semantic_selection['valid_economic_observations']})")
            else:
                self.llm_gen.set_semantic_plan(None)

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
                            "探索方向有尝试记录(供评审追溯), 最终盲评验证区对生成端不可见。"
                            "请优先设计机制正交、经济逻辑清晰、可解释的因子。"
                        )
                except Exception:
                    pass  # config 不可用时回退基础 system_prompt
                # P-20260909-004: WQ101 已验证算子先验锚定 (默认 on, env=off 关闭;
                # 数据源=wq101_coverage_audit.json, 与审计结果动态同步)
                if os.environ.get("FUXI_WQ101_ANCHOR", "on") == "on":
                    try:
                        _anchor = _build_wq101_anchor()
                        if _anchor:
                            _sys_prompt += _anchor
                    except Exception:
                        pass  # 锚定不可用时静默回退基础 prompt
                response_text = llm_client.chat_with_system(
                    system_prompt=_sys_prompt,
                    user_prompt=prompt,
                    temperature=0.8,
                    max_tokens=4096,
                )
                # M7: 批内 AST 去重接入主线 (顺序与 generate() 内路径一致:
                # 去重 → 语义打标)。此前该阶段只存在于 generate() 内路径。
                gen_candidates = self.llm_gen.parse_response(response_text)
                gen_candidates = self.llm_gen.batch_dedup_ast(
                    gen_candidates, evidence=_m7_dedup_evidence,
                    mode=_m7_dedup_mode, stage="llm_round1")
                gen_candidates = self.llm_gen.tag_semantic_candidates(gen_candidates)
                # P-20260915-012: LLM 路径 paradigm 兜底写入。
                # 根因: parse_response 仅 JSON 路径透传 paradigm (管道分隔格式无该
                # 字段), 且 LLM 常整体缺省 ⇒ Phase E 广播 cand.get('paradigm','')
                # 恒空串 ⇒ _arms=[] 且无任何提示, MAB 本地引导静默跳过 (实测 R4)。
                # 兜底来源 = 本轮生成语境的 target_paradigm (MAB 选中方向, 语义上
                # 候选即在该范式下生成), 比信任 LLM 自填更可靠。
                for _c in gen_candidates:
                    if not str(_c.get("paradigm", "") or "").strip():
                        _c["paradigm"] = paradigm or ""
                if freq in ("weekly", "daily"):
                    # P-20260830-001: 分频语境生成 → 候选显式打 natural_freq 标签
                    for _c in gen_candidates:
                        _c["natural_freq"] = freq
                source = "llm"
                _llm_api_ok = True
                print(f"  [LLM Gen] API 返回 {len(response_text)} chars → 解析出 {len(gen_candidates)} 个候选"
                      + (f" (natural_freq={freq})" if freq else ""))
                # ── P-20260909-004: Reviewer 二轮定向改进 (默认 on, env FUXI_LLM_REVIEWER=off
                # 关闭) ──
                # 首轮候选本地快速 IC 预评 → 最好/最差动机回传 LLM → 第二轮定向改进。
                # 预评不进任何裁决 (仅生成反馈), 合并后候选照走完整 S1-S5 链。
                # 每批最多 2 轮 LLM 调用, 失败自动降级单轮。
                if os.environ.get("FUXI_LLM_REVIEWER", "on") == "on" and len(gen_candidates) >= 2:
                    try:
                        from factor_ic_computer import FactorICComputer
                        _ric = FactorICComputer(lookback_days=270)
                        _pre = []
                        for _c in gen_candidates[:2]:
                            _f = _c.get("formula") or _c.get("expression") or ""
                            if not _f:
                                continue
                            try:
                                _r = _ric.compute(
                                    _f, str(_c.get("factor_name", "rev_pre"))[:40],
                                    also_lag1=True)
                                _pre.append({
                                    "name": str(_c.get("factor_name", "?"))[:40],
                                    "formula": _f[:150],
                                    "ic": float(_r.get("ic", 0.0) or 0.0),
                                    "icir": float(_r.get("icir", 0.0) or 0.0),
                                })
                            except Exception:
                                pass  # 单条预评失败不影响其余
                        if _pre:
                            _pre_sorted = sorted(_pre, key=lambda x: -abs(x["icir"]))
                            _fb_lines = [
                                f"{p['name']}: IC={p['ic']:+.4f}, ICIR={p['icir']:.3f}"
                                f" (公式: {p['formula']})"
                                for p in _pre_sorted
                            ]
                            _reviewer_prompt = (
                                "首轮公式的本地快速 Rank IC 预评结果如下:\n"
                                + "\n".join(_fb_lines)
                                + "\n\n请做第二轮定向改进: 保留 ICIR 最高公式的结构骨架, "
                                "修复 ICIR 最低公式的缺陷。注意 IC 符号方向: 负向信号必须在"
                                "公式里显式取负 (不要依赖方向标注)。输出 2 条改进后的新公式, "
                                "pandas .rolling()/.shift()/.rank(pct=True) 风格, "
                                "输出格式与首轮完全一致。"
                            )
                            try:
                                _resp2 = llm_client.chat_with_system(
                                    system_prompt=_sys_prompt,
                                    user_prompt=_reviewer_prompt,
                                    temperature=0.7,
                                    max_tokens=2048,
                                )
                                # M7: 二轮同样先批内去重 (仅批内; 与首轮的
                                # 同骨架候选属定向改进, 不做跨轮合并去重)
                                _round2 = self.llm_gen.parse_response(_resp2)
                                _round2 = self.llm_gen.batch_dedup_ast(
                                    _round2, evidence=_m7_dedup_evidence,
                                    mode=_m7_dedup_mode, stage="llm_round2")
                                _round2 = self.llm_gen.tag_semantic_candidates(_round2)
                                for _c in _round2:
                                    if freq in ("weekly", "daily"):
                                        _c["natural_freq"] = freq
                                    # P-20260915-012: 二轮候选同样补 paradigm 兜底
                                    if not str(_c.get("paradigm", "") or "").strip():
                                        _c["paradigm"] = paradigm or ""
                                    _c["_llm_reviewer_round2"] = True
                                if _round2:
                                    print(
                                        f"  [LLM-Reviewer] 二轮定向改进产出 {len(_round2)} 条 "
                                        f"(预评 {len(_pre)} 条, 最佳 {_pre_sorted[0]['name']} "
                                        f"ICIR={_pre_sorted[0]['icir']:.3f})"
                                    )
                                    gen_candidates = gen_candidates + _round2
                            except Exception as _r2e:
                                print(f"  [LLM-Reviewer] 二轮调用失败 (降级单轮, 不阻塞): {_r2e}")
                    except Exception as _rev_e:
                        print(f"  [LLM-Reviewer] 预评失败 (降级单轮, 不阻塞): {_rev_e}")
            except Exception as e:
                # Fallback: 降级为文件模式
                # P-20260917-028: 降级必须留痕, 否则 0 候选被误读为「模型没给新假设」
                _emsg = str(e)
                _llm_api_ok = False
                if "429" in _emsg:
                    _llm_degrade_reason = "rate_limited"
                elif "401" in _emsg or "403" in _emsg:
                    _llm_degrade_reason = "credential_rejected"
                elif ("DEEPSEEK" in _emsg.upper() or "API_KEY" in _emsg.upper()):
                    _llm_degrade_reason = "credential_missing"
                else:
                    _llm_degrade_reason = "api_error"
                print(f"  [LLM Gen] ⚠️ 降级文件模式 — 归因={_llm_degrade_reason}, 本轮候选=0 "
                      f"(非「无新假设」, 勿按空回合计入 MAB 奖励)")
                print(f"  [LLM Gen] 原因: {_emsg[:200]}")
                if _llm_degrade_reason.startswith("credential"):
                    try:
                        from llm_client import describe_api_key
                        _dg = describe_api_key()  # 脱敏: 只回 source/长度/前 6 位
                        print(f"  [LLM Gen] 凭据诊断: present={_dg.get('present')} "
                              f"source={_dg.get('source')} len={_dg.get('length')}")
                    except Exception as _dge:
                        print(f"  [LLM Gen] 凭据诊断不可用: {_dge}")
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
            # 2026-09-12 (W1): n_generations / top_k / mine_top_n 参数化。
            #   原 n_generations=5 硬编码 ⇒ best_all 恒 5 条 ⇒ forge 每轮产量 1~5。
            #   top_k>1: 每代记录 k 条; mine_top_n>0: 额外并入 final_population 前列。
            gen_candidates = self._generate_via_forge(
                pop_size=200,          # v0.7 P-025 固化 (2026-08-13)
                n_generations=forge_generations,
                max_candidates=max_candidates,
                top_k=forge_topk,
                mine_top_n=forge_mine_top_n,
                target_paradigm=paradigm,
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
            # S2-b (2026-09-12): 返回值改为 (source, candidates, trajectories) 三元组。
            # 兼容二元组 (历史/降级分支), 避免 arity 变更引发解包错误。
            _evo_ret = self._phase_generate_evo(
                templates=templates,
                retrieve_result=retrieve_result,
                paradigm=paradigm,
                max_candidates=max_candidates,
                max_turns=evo_turns,
                use_fsa=use_fsa,
            )
            source, gen_candidates = _evo_ret[0], _evo_ret[1]
            if len(_evo_ret) > 2:
                self._last_evo_trajectories = _evo_ret[2] or []
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

        # LLM-only deterministic formula precheck before seeds and S1-S5.
        # Raw model responses remain archived; seed/GP formulas are untouched.
        if source == "llm" and gen_candidates:
            gen_candidates = self.llm_gen.filter_quality_candidates(gen_candidates)

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

        # A2: same opt-in static/PIT contract for LLM, GP, Forge and seeds.
        # The historical LLM gate remains in place while this is compared.
        from candidate_preflight import precheck_candidates
        gen_candidates, _preflight_audit = precheck_candidates(
            list(gen_candidates or []), source=source,
            mode=os.environ.get("FUXI_GENERATOR_PREFLIGHT", "off").strip().lower(),
        )

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

            # ── P-20260910-001: 静态因果闭包校验 (shadow 默认: 只记录不拦截) ──
            try:
                from causal_closure_check import (
                    check_causal_closure, record_batch, should_block, MODE as _CC_MODE)
                _cc_entries, _cc_fatal = [], 0
                for cand in gen_candidates:
                    formula = cand.get("formula", cand.get("expression", ""))
                    if not formula:
                        continue
                    cc = check_causal_closure(formula)
                    if cc["verdict"] != "ok":
                        _cc_entries.append({
                            "factor": cand.get("name", cand.get("factor_name", "?")),
                            "formula_preview": formula[:80],
                            "verdict": cc["verdict"],
                            "violations": cc["violations"],
                            "warnings": cc["warnings"],
                        })
                        if should_block(cc["verdict"]):
                            cand["_causal_fatal"] = cc["violations"]
                            _cc_fatal += 1
                if _cc_entries:
                    record_batch(_cc_entries)
                if _CC_MODE == "enforce" and _cc_fatal:
                    gen_candidates = [c for c in gen_candidates if not c.get("_causal_fatal")]
                    print(f"  [CausalClosure] enforce: 拦截 {_cc_fatal} 个前视违规候选")
                elif _cc_entries:
                    print(f"  [CausalClosure] shadow: {len(_cc_entries)} 个候选命中违规/警告 (未拦截, 已记录)")
            except Exception as _cc_ex:
                print(f"  [CausalClosure] 检查失败 (不阻塞): {type(_cc_ex).__name__}: {_cc_ex}")

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

            # ── M9: 统一质量门接线 (shadow 记账 / enforce 拒绝; 证据随 return 落盘) ──
            _qgw_mode = quality_gate_wire_mode()
            if _qgw_mode != "off":
                _qgw = quality_gate_wire_step(
                    gen_candidates, _qgw_mode, stage="ralph_phase_generate")
                _qgw_records = _qgw["records"]
                _qgw_n_checked = _qgw["n_checked"]
                _qgw_n_fatal = _qgw["n_fatal"]
                _qgw_n_removed = _qgw["n_removed"]
                if _qgw_mode == "enforce":
                    if _qgw_n_removed > 0:
                        gen_candidates = _qgw["kept"]
                        print(f"  [QualityGate] enforce: 拒绝 {_qgw_n_removed}/{_qgw_n_checked} "
                              f"个未过门候选 (证据留档, 不进 S1)")
                elif _qgw_n_fatal > 0:
                    print(f"  [QualityGate] shadow: {_qgw_n_fatal}/{_qgw_n_checked} 个"
                          f"未过门候选**未拦截** (已留档; 设 FUXI_QUALITY_GATE_WIRE=enforce 转正)")

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
            cand.setdefault("_source", source)
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
            # R0/R3: immutable candidate and selected research-action identity.
            # The action id names what was actually executed, not merely the arm
            # that will later receive a reward.
            try:
                from run_contract import candidate_identity, research_action_id
                cand.update(candidate_identity(cand))
                cand["research_action_id"] = research_action_id(
                    generator=str(cand.get("_source") or source),
                    paradigm=str(cand.get("paradigm") or paradigm or ""),
                    fitness_mode=(
                        str(getattr(self.forge, "fitness_mode", ""))
                        if source == "forge" else "n/a"
                    ),
                    natural_freq=str(cand.get("natural_freq") or ""),
                )
            except Exception as _identity_error:
                cand["_identity_error"] = str(_identity_error)[:160]

        return {
            "n_candidates": len(gen_candidates),
            "candidates": gen_candidates,
            "source": source,
            "mab_direction": paradigm,
            # P-20260917-028: 降级留痕 (None=非 LLM 分支 / True=API 成功 / False=降级)
            "llm_api_ok": _llm_api_ok,
            "llm_degrade_reason": _llm_degrade_reason,
            "semantic_selection": _semantic_selection,
            "generator_preflight": _preflight_audit,
            # M7: 批内 AST 去重留档 (enforce 移除 / shadow 仅命中 均在 records)
            "batch_ast_dedup": {
                "mode": _m7_dedup_mode,
                "n_flagged": len(_m7_dedup_evidence),
                "n_removed": sum(1 for _r in _m7_dedup_evidence
                                 if _r.get("removed")),
                "records": _m7_dedup_evidence,
            },
            # M9: 统一质量门接线留档 (shadow 记账 / enforce 拒绝均在 records)
            "quality_gate_wire": {
                "mode": _qgw_mode,
                "n_checked": _qgw_n_checked,
                "n_fatal": _qgw_n_fatal,
                "n_removed": _qgw_n_removed,
                "records": _qgw_records,
            },
            # M9: 库级 FSA 判据可达性快照 (只读; share 不可达时 reachable=False)
            "fsa_context": retrieve_result.get("fsa_context_status", {}),
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

        # 3. champion_seed 优先 + weekly 配额 (P-20260904-002) + icir 降序
        # weekly 判定: 显式 natural_freq > 轨迹 freq > 默认 daily
        try:
            from config import WEEKLY_SEED_MIN as _wsm
            _wsm = int(_wsm or 0)
        except Exception:
            _wsm = 0

        def _freq_of(c):
            nf = str(c.get("natural_freq") or "").strip()
            if nf in ("weekly", "daily"):
                return nf
            return self._lookup_traj_freq(c.get("factor_name", "")) or "daily"

        champions = [c for c in picked if c.get("status") == "champion_seed"]
        _ch_ids = {id(c) for c in champions}
        weekly = [c for c in picked if id(c) not in _ch_ids and _freq_of(c) == "weekly"]
        _wk_ids = {id(c) for c in weekly}
        others = [c for c in picked if id(c) not in _ch_ids and id(c) not in _wk_ids]
        for _grp in (champions, weekly, others):
            _grp.sort(key=lambda c: -(float(c.get("icir") or 0) or 0))
        if _wsm > 0:
            weekly = weekly[:_wsm]
        merged = champions + weekly + others
        seeds = merged[:max_seeds]
        if _wsm > 0 and weekly:
            n_wk = sum(1 for s in seeds if any(s is w for w in weekly))
            print(f"  [WeeklySupply] 种子重检 weekly 配额 {n_wk}/{_wsm} (P-20260904-002)")
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
        top_k: int = 1,
        mine_top_n: int = 0,
        target_paradigm: str = "",
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
        # R3: Forge can only receive reward for an action it actually executed.
        # An MAB paradigm without a registered Forge profile falls back explicitly
        # instead of searching every profile and misattributing the outcome.
        if target_paradigm and target_paradigm not in MATURE_PARADIGM_PROFILES:
            print(f"  [Forge/R3] 范式 {target_paradigm!r} 无 Forge profile → 明确回退 GP Breed")
            return []

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
                # 2026-09-12 (W1): 每代记录条数参数化 (默认 1 = 原行为)。
                # 原实现 best_all 长度恒 = n_generations(=5) ⇒ forge 产量被钉死,
                # 与 max_candidates 无关 (max_candidates 只对 llm 路径 binding)。
                top_k=top_k,
                # 挖 final_population 时需相应放大 ICIR 计算条数
                icir_top_n=max(30, mine_top_n),
                target_paradigm=target_paradigm or None,
            )
        except Exception as e:
            print(f"  [Forge] 进化异常: {e}")
            return []

        # ── 2026-09-12 (W1): 挖掘最终种群 ──
        # 旧实现只导出每代最优 (5 条), 而 forge 早已把**整个最终种群**写到
        # self.final_population (见 forge/__init__.py:final_population, ~pop_size 条,
        # 按 fitness 降序, 前 icir_top_n 条带 ICIR)。该资产此前**只被实验脚本
        # forge_round1_experiment.py 消费**, 流水线从未使用。
        # mine_top_n>0 时把种群前列并入候选池 (优先与该代最优去重, 不重复计入)。
        _pool = list(best_all)
        if mine_top_n > 0:
            _mined = 0
            for _e in (getattr(self.forge, "final_population", None) or [])[:mine_top_n]:
                if _e.get("icir") is None:
                    continue  # 未补算 ICIR 的个体不能用 (下游 S1 需要 icir)
                _pool.append({
                    "generation": int(n_generations),   # 视为末代种群成员
                    "name": f"pop_{_e['expression'][:48]}",
                    "rank_in_gen": 99,                  # 99 = 种群挖掘(非代内排名)
                    "expression": _e["expression"],
                    "pandas_expression": _e.get("pandas_expression", _e["expression"]),
                    "fitness": _e.get("fitness", 0.0),
                    "icir": _e["icir"],
                    "complexity": _e.get("complexity", 0),
                    "size": _e.get("size", 0),
                    "depth": _e.get("depth", 0),
                    "terminals": [],
                    "paradigm": _e.get("paradigm"),
                    "_mined": True,
                })
                _mined += 1
            print(f"  [Forge] 种群挖掘: 并入 {_mined} 条 final_population "
                  f"(池 {len(best_all)} → {len(_pool)})")

        # 转换为标准候选格式 (v3.2: 使用 pandas_direct 表达式供 fri.py 执行)
        candidates = []
        seen_exprs = set()
        for b in _pool:
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
                "paradigm": b.get("paradigm") or target_paradigm,
                "forge_fitness_mode": str(getattr(self.forge, "fitness_mode", "legacy")),
                "forge_target_paradigm": target_paradigm or "",
            })

        # 取 Top-N
        _n_before = len(candidates)
        candidates = candidates[:max_candidates]
        print(f"  [Forge] 产生 {len(candidates)} 个候选 "
              f"(池 {_n_before} → 截断 max_candidates={max_candidates})")
        return candidates

    def _load_forge_data(self) -> Optional[Dict]:
        """加载 FactorForge 所需的 OHLCV 数据

        自动扫描多个可能的 parquet/csv 数据源，构建 FactorForge 期望的
        {'open','high','low','close','volume','returns'} (T,N) 格式。
        默认要求真实 OHLC 字段；字段契约 legacy 才允许历史 close 代理。
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
            from forge.primitives import field_contract
            if field_contract() == 'enforce':
                field_specs.pop('vwap')
                for field in ('open', 'high', 'low'):
                    field_specs[field] = (field, None)

            pivot_cache = {}
            for fname, (primary, fallback) in field_specs.items():
                col = primary if primary in df.columns else fallback
                if col is None or col not in df.columns:
                    if field_contract() == 'enforce':
                        raise ValueError(f'Forge real field unavailable: {fname}')
                    if fname in ("close", "volume"):
                        # close/volume 是必须的
                        print(f"  [Forge] 缺少必要字段 '{fname}'，回退到 GP 育种")
                        return None
                    continue

                if col not in pivot_cache:
                    pivoted = df.pivot_table(
                        index="date", columns="stock", values=col, aggfunc="last"
                    )
                    if 'close' in pivot_cache:
                        pivoted = pivoted.reindex_like(pivot_cache['close'])
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
                if 'close' in pivot_cache:
                    pivoted = pivoted.reindex_like(pivot_cache['close'])
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
                    paradigm=paradigm,
                    validator=_find_unknown_identifiers)  # 2026-09-12 公式校验
                if traj_parents:
                    print(f"  [TrajPool] 轨迹父本池注入 {len(traj_parents)} 条 "
                          f"(命中率加权, 30天冷却)")
        except Exception as e:
            print(f"  [TrajPool] 注入失败(不影响主线): {e}")

        # v0.6: 模板种子不足时, 降级为普通 GP breed
        if len(templates) < 2:
            # ── S2-b (2026-09-12): 显式告警 + 归因 ──
            # 更正: 原实现**有**一行 print (报告初稿写的"静默返回且不打印"不准确),
            # 但它不是告警级、且**没有归因**, 无法据此判断上游为何断供。
            _why = []
            if not templates:
                _why.append("success_templates 为空 (R 阶段未产出可用模板)")
            else:
                _why.append(f"仅 {len(templates)} 条 < 2 (阈值)")
            if not traj_parents:
                _why.append("TrajectoryPool 父本 0 条 (冷却中/无合格轨迹)")
            print(f"  [EvoTraj] ⚠️ 降级为普通 gp_breed — 归因: {'; '.join(_why)}")
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
            # S2-b: 显式置空轨迹通道 (不残留上一轮), 并记录降级原因供诊断
            self._last_evo_trajectories = []
            self._last_evo_downgrade = {
                "reason": "templates<2",
                "detail": "; ".join(_why),
                "n_templates": len(templates),
                "n_traj_parents": len(traj_parents),
                "ts": datetime.now().isoformat(),
            }
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
                        # P2 (2026-09-05): gp_breed 消费 — 跨引擎失败样本降权 0.5
                        forbidden = mem.get_motif_forbidden(min_samples=3, max_rate=0.1,
                                                            consumer_engine="gp_breed")
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
                        # S2-c i: FSA 过滤后为 0 ⇒ 归因到 code 段
                        n_fsa=0, n_eval_ok=0, n_eval_failed=0,
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
                        # S2-c i: 有候选但零条求值成功 ⇒ 归因到 code 段
                        n_fsa=len(fsa_filtered), n_eval_ok=0,
                        n_eval_failed=len(fsa_filtered),
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
                    # S2-c i (2026-09-12): 步骤级归因原料
                    n_fsa=len(fsa_filtered),
                    n_eval_ok=len(eval_results),
                    n_eval_failed=max(0, len(fsa_filtered) - len(eval_results)),
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

        # ── S2-c ii (2026-09-12): 高奖励片段重组 ─────────────
        # 实测: trajectory_log 2194 条中 1067 条已带非空 high_reward_segments,
        # 但该字段此前**从未被任何消费者读取**。此处把它作为父本喂给既有 GP
        # 引擎做跨轨迹重组 (子系统已在做子树级交叉, 不另造拼接器)。
        # 开关 FUXI_EVO_SEGMENT_RECOMBINE = off|shadow(默认)|on
        try:
            import os as _os
            _sr = _os.environ.get("FUXI_EVO_SEGMENT_RECOMBINE",
                                  "shadow").strip().lower()
            if _sr not in ("off", "shadow", "on"):
                _sr = "shadow"
            if _sr != "off":
                from evo_trajectory import (load_high_reward_seed_pool,
                                            recombine_high_reward_seeds)
                # 2026-09-12 修复: 注入权威标识符校验器 —— 实测该池 top 条目是
                # `scale(returns)`/`inv(scale(returns))` 伪代码 (ICIR 1000/23.8),
                # 不校验则重组子代 100% 求值失败 (NameError: scale/inv)。
                _seg_pool = load_high_reward_seed_pool(
                    data_dir=self.data_dir, top_k=6, min_final_icir=0.10,
                    prefer_segment="expression",
                    validator=_find_unknown_identifiers)
                if _seg_pool:
                    _kids = recombine_high_reward_seeds(
                        _seg_pool, self.breeder,
                        n_children=max(4, max_candidates // 2),
                        fsa=self.fsa if use_fsa else None)
                    if _sr == "on":
                        all_candidates.extend(_kids)
                        print(f"  [S2c-ii] 高奖励片段重组: 父本 {len(_seg_pool)} "
                              f"→ 子代 {len(_kids)} (已注入)")
                    else:
                        print(f"  [S2c-ii/shadow] 高奖励片段重组可产 {len(_kids)} 子代 "
                              f"(父本 {len(_seg_pool)}), 未注入 "
                              f"(FUXI_EVO_SEGMENT_RECOMBINE=on 启用)")
                else:
                    print(f"  [S2c-ii] 高奖励片段种子池为空 (high_reward_segments 无可用条目)")
        except Exception as _se:
            print(f"  [S2c-ii] 片段重组跳过 (不阻塞): {_se}")

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

        # ── S2-b (2026-09-12): 轨迹改为一等通道 ──────────────
        # 原实现 `if all_candidates: all_candidates[0]["_evo_trajectories"] = ...`
        # 是**单点挂载**: 首候选被 S1/S2 过滤 ⇒ 整批轨迹丢失; all_candidates 为空
        # ⇒ 轨迹直接丢弃。现改为实例级通道, 由 _phase_generate 提升为一等字段。
        self._last_evo_trajectories = list(trajectories)
        self._last_evo_downgrade = None
        if all_candidates:
            all_candidates[0]["_evo_trajectories"] = trajectories  # 兼容旧读法(deprecated)

        return ("gp_evolve", all_candidates, trajectories)

    # ═══════════════════════════════════════════════════════════
    # v0.6: EvoTraj 辅助方法
    # ═══════════════════════════════════════════════════════════

    def _extract_trajectories(self, generate_result: Dict) -> List[Dict]:
        """
        从 G 阶段结果提取轨迹摘要 (供外部使用)。

        S2-b (2026-09-12): 改读**实例级通道** `self._last_evo_trajectories`。
        原实现只读 `candidates[0]["_evo_trajectories"]` —— 单点挂载:
          · 首候选被 S1/S2 过滤掉 ⇒ 整批轨迹丢失 (静默)
          · all_candidates 为空 ⇒ 轨迹直接丢弃 (空候选轮完全不落盘)
        实例通道与候选列表解耦 ⇒ 空候选轮也能取到轨迹 (空列表, 但可判"跑过")。
        旧字段保留为兼容回退 (deprecated)。
        """
        trajectories: List[Dict] = []
        _channel = getattr(self, "_last_evo_trajectories", None)
        if _channel:
            for traj in _channel:
                try:
                    trajectories.append(
                        traj.get_summary() if hasattr(traj, "get_summary") else dict(traj))
                except Exception:
                    continue
            if trajectories:
                return trajectories
        # ── 兼容回退: 旧单点挂载 (deprecated) ──
        candidates = generate_result.get("candidates", []) if generate_result else []
        if candidates and candidates[0].get("_evo_trajectories"):
            for traj in candidates[0]["_evo_trajectories"]:
                try:
                    trajectories.append(
                        traj.get_summary() if hasattr(traj, "get_summary") else dict(traj))
                except Exception:
                    continue
        return trajectories

    # ═══════════════════════════════════════════════════════════
    # v0.5: MAB 探索方向调度
    # ═══════════════════════════════════════════════════════════

    def _mab_select_direction(self) -> Optional[ResearchDirection]:
        """使用 MAB UCB1 选择本轮探索的范式方向"""
        if not self._mab_loaded:
            # S1-a a4 (2026-09-12): 原实现调 `self.mab.load_state()` —— 该方法
            # **不存在** (MABScheduler 只有私有 `_load_state`), AttributeError 被
            # `except: pass` 静默吞掉 ⇒ 看起来"加载成功"实则什么都没做。
            # 且 `_load_state` 是**覆盖式**加载 (非合并), 在 S1-a a2 单实例注入后
            # 再重载会把内存中未落盘的更新冲掉 ⇒ 明确不做重载, 只标记。
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

            # S1-a a4 (2026-09-12): 改为走 mark_executed —— 唯一合法的 pulls 增长点。
            # 原实现 `best.pulls += 1` 只动臂计数、**不动 total_pulls** ⇒ 制造出
            # 第三本账 (实测 total_pulls=317 而 sum(pulls)=484)。选择即执行是本
            # 函数的语义 (调用方随即用 paradigm 生成), 故此处计 pull 是正确的,
            # 但必须两侧同增。
            if not self.mab.mark_executed(best.direction_id, save=False):
                best.pulls += 1
                best.last_pulled = datetime.now().isoformat()
            self.mab.save_state()
        return best

    def _mab_select_generator(self, direction: ResearchDirection) -> str:
        """
        v3.1 / S2-a (2026-09-12): 为选定方向选择生成器 (基于历史表现)。

        S2-a 修正两处:
        1. **生成器轴死代码**: `generator_rewards` 此前全代码库无写入点
           (见 `MABScheduler.record_generator_reward`), 且读入后
           `if gen_rewards:` 恒假 ⇒ "软最大化选择" 分支永不执行, 永远退化为
           按 `pulls` 分档的冷启动启发式。即**生成器选择从未学习**。
        2. **gp_evolve 定位精炼器 (非平级引擎)**: EvoTraj 多轮进化的语义是
           "在已有库知识上做精炼" (需要 ≥2 模板 + 轨迹父本), 不适合冷启动。
           故仅在**成熟方向** (pulls ≥ MIN_PULLS) 才纳入候选, 并按
           "效果优先 + 周期性探针" 两条路路由:
             (a) gp_evolve 已记录平均奖励严格最优 → 直接选它
             (b) 每 REFINE_EVERY 次执行周期探针一次 (防 starve)
          冷启动方向 (pulls < MIN_PULLS) 永不路由 gp_evolve。

        开关 FUXI_GENERATOR_AXIS (M2, 2026-09-27, P-20260927-005: 与
        FUXI_EVO_REFINER 解耦 —— 后者现仅管 gp_evolve 精炼器补路, 见
        _s2a_refiner_gate 与 run() 内补路块):
          off    → 完全沿用旧逻辑 (忽略 generator_rewards), 100% 历史行为
          shadow → **计算但不应用** (打印本应路由到的生成器) ⇒ 零回归观测 (默认)
          on     → 应用生成器轴学习 (含 gp_evolve 探针/效果优先路由)
        **写读双开关** (M2): 写 = FUXI_GENERATOR_AXIS_WRITE (默认 shadow,
        只写观测账本不动 MAB); 读 = 本函数。默认 shadow×shadow 行为与历史
        逐位一致。
        """
        import os as _os
        from generator_axis import resolve_read_mode as _resolve_read_mode
        _mode = _resolve_read_mode()
        if _mode not in ("off", "shadow", "on"):
            _mode = "shadow"
        # M2: 阈值/周期支持新名 (FUXI_GENERATOR_AXIS_*); 旧名 (FUXI_EVO_REFINER_*)
        # 作为回落以兼容历史配置。
        _min_pulls = int(_os.environ.get(
            "FUXI_GENERATOR_AXIS_MIN_PULLS",
            _os.environ.get("FUXI_EVO_REFINER_MIN_PULLS", "3")) or 3)
        _every = int(_os.environ.get(
            "FUXI_GENERATOR_AXIS_EVERY",
            _os.environ.get("FUXI_EVO_REFINER_EVERY", "5")) or 5)
        _every = max(1, _every)

        pulls = int(getattr(direction, 'pulls', 0) or 0)
        mature = pulls >= _min_pulls
        gen_rewards = getattr(direction, 'generator_rewards', None) or {}

        # ── 旧逻辑 (off / shadow 的实际返回值) ──
        def _legacy() -> str:
            # 学习路径**仅对成熟方向**开放 (S2-a): 冷启动方向即使带有残余
            # generator_rewards 也必须走 pulls 分档 (否则会跳过 LLM 冷启动,
            # 而冷启动正是 LLM 的主场 —— 见"未探索方向 → LLM 创新")。
            if gen_rewards and mature and _mode == "on":
                # 有历史数据: 软最大化选择 (ε-greedy)
                import random
                _pool = ["gp_breed", "llm", "forge", "gp_evolve"]
                if random.random() < 0.2:  # 20% 探索
                    return random.choice(_pool)
                # 80% 利用: 选**有效样本数达标**的生成器中平均奖励最高者
                best_gen, best_avg, best_n = "gp_breed", -float("inf"), -1
                for gen in _pool:
                    rewards = [r for r in (gen_rewards.get(gen) or [])
                               if isinstance(r, (int, float))]
                    if len(rewards) < 1:
                        continue
                    avg = sum(rewards) / len(rewards)
                    if (avg, len(rewards)) > (best_avg, best_n):
                        best_avg, best_n, best_gen = avg, len(rewards), gen
                return best_gen
            # 无可用历史 (或 off / 冷启动): 按 pulls 分档
            if pulls <= 1:
                return "llm"        # 冷启动方向: LLM 生成新颖因子
            elif pulls <= 3:
                return "gp_breed"   # 中期: GP 育种
            else:
                return "forge"      # 成熟方向: Forge 完整进化

        legacy_gen = _legacy()

        if _mode in ("off", "shadow"):
            if _mode == "shadow" and mature:
                # 影子读数: 生成器轴学习 与 精炼器路由的**忠实反事实**。
                #
                # 🔴 P-20260917-023 修复 (原实现两处误导, 2026-09-17 实证):
                #   (a) `_would` 初始化为硬编码 "gp_breed" 且**不回落 pulls 分档**
                #       ⇒ 生成器轴尚无累计数据(=正常首轮状态)时打印
                #       「本应路由 gp_breed」, 而 on 模式的真实决策是回落 pulls
                #       分档 (pulls>3 ⇒ forge) ⇒ 正常的「数据未积累」被误读成
                #       「路由 break」, 连续多日被当成缺陷上报。**这才是真缺陷**。
                #   (b) 尾部文案 "(FUXI_EVO_REFINER=on 启用)" 是**硬编码字符串**,
                #       与实际 _mode (默认 shadow) 无关 ⇒ 让人以为 on 模式出了异常。
                # 现改为逐支复刻 on 模式决策链 (探针 → gp_evolve 最优 → 贪婪 →
                # pulls 分档), 并如实打印当前 _mode。**纯读数, 不改任何返回值**。
                _axes = []
                for gen in ("gp_breed", "llm", "forge", "gp_evolve"):
                    _r = [x for x in (gen_rewards.get(gen) or [])
                          if isinstance(x, (int, float))]
                    if _r:
                        _axes.append(f"{gen}={sum(_r)/len(_r):+.3f}(n={len(_r)})")
                _probe = mature and (_every > 0) and (pulls % _every == 0)

                def _avg(gen):
                    r = [x for x in (gen_rewards.get(gen) or [])
                         if isinstance(x, (int, float))]
                    return (sum(r) / len(r)) if r else None

                if not mature:
                    _would, _why = legacy_gen, "冷启动: pulls 分档"
                elif _probe:
                    _would, _why = "gp_evolve", "周期探针"
                elif (_avg("gp_evolve") is not None
                      and max([v for v in (_avg("gp_breed"), _avg("llm"),
                                           _avg("forge")) if v is not None] or
                              [-float("inf")]) < _avg("gp_evolve")):
                    _would, _why = "gp_evolve", "gp_evolve 均值严格最优"
                elif _axes:
                    _best, _bg = -float("inf"), legacy_gen
                    for gen in ("gp_breed", "llm", "forge", "gp_evolve"):
                        v = _avg(gen)
                        if v is not None and v > _best:
                            _best, _bg = v, gen
                    _would, _why = _bg, "生成器轴贪婪支 (on 另含 20% 随机探索)"
                else:
                    # 🔴 关键修复: 无累计数据 ≠ 路由故障
                    _would, _why = legacy_gen, "生成器轴暂无数据 ⇒ 回落 pulls 分档"
                print(f"  [S2a/shadow] 方向={direction.name} pulls={pulls} mature={mature} "
                      f"生成器轴={'/'.join(_axes) if _axes else '空'} ⇒ "
                      f"本应路由 {_would} [{_why}], "
                      f"实际沿用 legacy={legacy_gen} "
                      f"(FUXI_GENERATOR_AXIS 当前={_mode}: 仅观测不应用)")
            return legacy_gen

        # ── on: 生成器轴学习 + gp_evolve 精炼器路由 ──
        if mature:
            # 周期探针 (防 starve): 每 _every 次执行强制一次精炼
            if _every > 0 and pulls % _every == 0:
                print(f"  [S2a] 方向={direction.name} pulls={pulls} → "
                      f"周期精炼探针 gp_evolve (每 {_every} 次)")
                return "gp_evolve"
            # 效果优先: gp_evolve 平均奖励严格最优 → 选它
            _ge = [x for x in (gen_rewards.get("gp_evolve") or [])
                   if isinstance(x, (int, float))]
            if _ge:
                _ge_avg = sum(_ge) / len(_ge)
                _others = [
                    sum(r) / len(r)
                    for g in ("gp_breed", "llm", "forge")
                    for r in [(gen_rewards.get(g) or [])]
                    if any(isinstance(x, (int, float)) for x in r)
                    and [x for x in r if isinstance(x, (int, float))]
                ]
                if _others and _ge_avg > max(_others):
                    print(f"  [S2a] 方向={direction.name} → gp_evolve 精炼器 "
                          f"(avg={_ge_avg:+.3f} > 其他最优 {max(_others):+.3f})")
                    return "gp_evolve"
        return legacy_gen

    def _serve_aware_adjust(self, direction: ResearchDirection, generator: str) -> str:
        """P2-1 (M5, 2026-09-28): 生成器路由「服务矩阵感知」降级。

        Forge 只能服务 MATURE_PARADIGM_PROFILES 内 8 范式; 无 profile 的
        范式进 `_generate_via_forge` 即 return [] ([Forge/R3] 明确回退),
        上层 GP fallback 亦可能空 ⇒ 空轮 (R3-R5 连续 3 轮实证; 26 臂
        @11p 前瞻 20/26 = 77% 预期空轮)。本调整在 MAB 选定生成器后做
        服务性检查, 目标为 forge 且范式无 profile 时降级 llm (LLM 生成
        不依赖 profile; 既有设计语言「未覆盖范式 → LLM 擅长冷启动」)。

        开关 FUXI_ROUTE_SERVE_AWARE (off | shadow | on(默认)):
          off    → 不动作 (100% 历史行为)
          shadow → 只打印「本应降级」读数, 返回值不变 (零回归)
          on     → 应用降级 (forge → llm)
        纯逻辑见 serve_aware.serve_aware_adjust (可单测)。
        """
        try:
            from serve_aware import resolve_serve_aware_mode, serve_aware_adjust
            return serve_aware_adjust(
                direction, generator,
                mode=resolve_serve_aware_mode(),
                profiles=MATURE_PARADIGM_PROFILES)
        except Exception as _sa:
            print(f"  [M5/serve] 服务矩阵检查失败 (非阻塞): {type(_sa).__name__}: {_sa}")
            return generator

    def _mab_record_result(
        self, direction_name: str, jq_return: float, jq_sharpe: float,
        streak: int = 0, evidence: Optional[Dict] = None,
    ):
        """M1(2026-09-26): 「已验证经济事件→奖励」的**唯一通道**（薄封装）。

        实现全部移交 reward_guard.consume_verified_reward：
          - 五态守卫: 非 valid 事件永不奖惩; NaN/缺失 jq_return 视为无读数
            (不得当 0 或负值);
          - 写前强制 ensure_reward_source (换源重置, 禁跨量纲混算);
          - 默认 shadow: 只落 data/reward_guard_ledger.jsonl (只读数不生效),
            与 09-26 A0 的隔离语义合并; 写生效需显式 FUXI_REWARD_GUARD=enforce。

        历史 (改造前): 本函数是**零调用死函数**, 且无任何五态守卫 —— M1 对象。
        奖励量纲: clip((jq_return-182.57)/100*5*(1+0.2*streak), -1, 1)
        """
        from reward_guard import consume_verified_reward
        ev = dict(evidence) if evidence else {}
        ev.setdefault("source", "jq")
        if "value" not in ev:
            ev["value"] = jq_return
        res = consume_verified_reward(
            self.mab, direction_name, evidence=ev,
            jq_return=jq_return, jq_sharpe=jq_sharpe, streak=streak)
        if res.get("status") == "no_arms":
            print(f"  [MAB] ⚠️ JQ 奖励无对应臂: {direction_name!r} "
                  f"(reward={res.get('reward')} 未写回; mode={res.get('mode')}) — "
                  f"若该范式持续产出, 应在 mab_scheduler_state.json 注册方向")
        return res

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
        n_ic_eval_failed = 0   # W2 (2026-09-12): 公式求值失败计数 (归因用, 非因子无效)
        ic_comp = None
        # v0.7 频率对称: weekly lane 候选走周频裁决器 (XOR, 不重复日频计算)
        _weekly_judge = None
        _weekly_load_err = ""
        try:
            from weekly_lane import get_weekly_judge
            _weekly_judge = get_weekly_judge()
        except Exception as _wje:  # noqa: BLE001
            _weekly_judge = None
            _weekly_load_err = f"{type(_wje).__name__}: {_wje}"
        n_weekly_judged = 0
        n_daily_lane = 0   # P-20260915-010: 日频通道计数 (分频统计恒可见)
        # P-20260902-002: IC 衰减半衰期影子计数
        n_decay_computed = 0
        n_decay_fast = 0
        try:
            from factor_ic_computer import FactorICComputer
            ic_comp = FactorICComputer()
            self._last_ic_comp = ic_comp  # P-20260912-002: 供幸存者偏差守卫复用
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
                n_daily_lane += 1   # P-20260915-010: 分频统计 (日频通道)
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
                        # 2026-09-12 (W2): 透传 IC 有效天数 — 供 S1 样本量守卫 + 归因
                        cand["n_days"] = int(ic_result["n_days"])
                        n_ic_computed += 1
                    else:
                        # 2026-09-12 (W2): n_days==0 ⇒ 公式**求值失败**(语法/字段/rolling 不支持),
                        # 与"上游没预先算 IC"是两回事。原实现不给任何标记, S1 会把它报成
                        # 「无IC数据（status=reserve/未评估）— Stage1拒绝」, 导致归因被污染
                        # (fd 审计里 C 类"流程故障"正是此坑)。此处显式打标。
                        cand["_ic_eval_failed"] = True
                        cand["n_days"] = 0
                        n_ic_eval_failed += 1
                        # ── P-20260902-002: IC 衰减半衰期 (影子, 不参与裁决) ──
                        try:
                            _dec = ic_comp.compute_decay(formula, cand.get("factor_name", ""))
                            cand["decay_half_life_days"] = _dec.get("decay_half_life_days")
                            cand["decay_fast"] = _dec.get("decay_fast", False)
                            cand["decay_lambda"] = _dec.get("decay_lambda")
                            cand["_decay_ic_by_lag"] = _dec.get("ic_by_lag", {})
                            n_decay_computed += 1
                            if _dec.get("decay_fast"):
                                n_decay_fast += 1
                        except Exception as _de:
                            cand["decay_half_life_days"] = None
                            cand["decay_fast"] = False
                            cand["_decay_err"] = str(_de)[:80]
            if n_ic_computed > 0:
                print(f"  [伏羲] IC 实时计算: {n_ic_computed}/{len(candidates)} 候选 "
                      f"(max|IC|={max(abs(float(c.get('ic', 0)) or 0) for c in candidates):.4f})")
            if n_ic_eval_failed > 0:
                print(f"  [伏羲][W2] 公式求值失败 {n_ic_eval_failed}/{len(candidates)} 候选 "
                      f"(n_days=0 → S1 归因=流程故障, 非因子无效)")
            if n_decay_computed > 0:
                print(f"  [P-20260902-002 影子] IC 半衰期: {n_decay_computed} 候选已算, "
                      f"decay_fast(<5日) {n_decay_fast} 个 (仅标注, 不裁决)")
            # P-20260915-010: 分频裁决统计**恒输出**。
            # 原实现只在 n_weekly_judged>0 时打印, 且裁决器加载失败静默置 None
            # ⇒ 「本轮无 weekly 候选」与「周频裁决器损坏」产生完全相同的可观测
            #   结果 (均无输出), 流水线会沉默退化成纯日频而无人察觉 (接口层回退
            #   偏执面)。现恒输出裁决器状态 + 双通道计数。
            if _weekly_judge is None:
                print(f"  [v0.7] 🔴 周频裁决器不可用 ⇒ 本轮强制全走日频 lane "
                      f"(原因: {_weekly_load_err or '未初始化'})")
            if n_weekly_judged > 0:
                print(f"  [v0.7] 周频裁决器: {n_weekly_judged} 候选走 weekly lane "
                      f"(τ_w 门槛, 不重复日频裁决)")
            print(f"  [v0.7] 分频裁决统计: weekly lane {n_weekly_judged} / "
                  f"daily lane {n_daily_lane} | 裁决器 "
                  f"{'✅在线' if _weekly_judge is not None else '🔴离线'}")
        except Exception as e:
            print(f"  [伏羲] ⚠️ IC 计算失败: {e}，候选 S1 将被拒绝")

        # ── P-20260907-006: S5 方向符号守卫 (2026-09-11 转正 enforce) ──
        # 背景: gp_evo_001/R3/2026-09-11 Stage2 三次同坑 — 负 ICIR 因子以原方向
        #   进 JQ 队列必灭 (组合负收益)。此处在 S1 之前对负向候选生成方向翻转副本
        #   并用同通道重算, 取 ICIR 更优者; 负向且翻转无效的直接标 reject。
        # 开关: env FUXI_S5_SIGN_GUARD = enforce(默认) | shadow | off (一行回退)。
        try:
            from s5_sign_guard import apply as _apply_sign_guard
            _sg = _apply_sign_guard(candidates, ic_comp=ic_comp,
                                    weekly_judge=_weekly_judge, verbose=True)
            self._sign_guard_stats = _sg
        except Exception as _sge:
            print(f"  [S5-SignGuard] ⚠️ 守卫执行失败 (非阻塞): {_sge}")

        # ── W3 (2026-09-12) / M10 (2026-09-27): forbidden_directions 公式级硬屏蔽门 ──
        # 背景: 51 条 forbidden_directions 全 severity=hard、50 条带 prototype,
        #   但 S1–S5 全链路不消费 (只在入库时经 library_orthogonality 硬拒) ⇒
        #   已被 JQ 证伪的原型反复重生成、反复烧回测额度。
        # 🔴 M10 (2026-09-27) 顺序修复: 本块原位于符号守卫**之前**, 与自身注释
        #   ("必须在符号守卫之后执行") 矛盾 —— 守卫 enforce 会**原地替换**
        #   formula 为 `-(...)` 翻转版 (s5_sign_guard), 守卫前打标会漏检翻转后
        #   的实际提交式。现已移到守卫之后 (与注释一致)。
        # 开关: env FUXI_FORBIDDEN_GATE = off | shadow | enforce (默认, 2026-09-27 转正;
        #   回退: 设 shadow 只标记 / off 无痕)。豁免: env FUXI_FORBIDDEN_EXEMPT
        #   (默认 data/forbidden_gate_exempt.json; off 关闭); 命中豁免 ⇒
        #   exempted=True / enforced=False (标而不拒, 不隐藏信息)。
        # 冲突复测 (2026-09-27): 库内 1308 条唯一公式命中 4 条 (0.3%) ——
        #   3 条 JQ 方向证伪保留拦截 + 1 条 dollar_vol_stability 执行层根因豁免。
        # 裁决在 multi_stage_validator.stage1_fast_ic (读 cand["_forbidden_hit"])。
        self._forbidden_stats = None
        try:
            from forbidden_gate import get_forbidden_gate
            _fg = get_forbidden_gate()
            _n_hit = 0
            _n_enforced = 0
            if _fg.mode != "off":      # off = 真·无痕 (不打标也不算)
                for cand in candidates:
                    _fexpr = cand.get("formula", cand.get("expression", ""))
                    _fr = _fg.check(_fexpr) if _fexpr else None
                    if _fr and _fr.get("blocked"):
                        cand["_forbidden_hit"] = _fr
                        _n_hit += 1
                        if _fr.get("enforced"):
                            _n_enforced += 1
                    else:
                        cand.pop("_forbidden_hit", None)
            self._forbidden_stats = {"mode": _fg.mode, "hits": _n_hit,
                                     "enforced": _n_enforced,
                                     "blacklist": len(_fg.entries)}
            if _n_hit > 0:
                print(f"  [W3-禁止方向] 命中 {_n_hit}/{len(candidates)} 候选 "
                      f"(mode={_fg.mode}, 黑名单 {len(_fg.entries)} 骨架)"
                      + (f" → 硬拒 {_n_enforced}" if _n_enforced else " → 仅标记"))
        except Exception as _fge:
            print(f"  [W3-禁止方向] ⚠️ 屏蔽门异常 (非阻塞): {_fge}")

        # ── M10 (2026-09-27): AST 结构去重 — S1 前置打标 (enforce 通道接线) ──
        # 背景: P-20260915-005 的门此前只有评估后影子读数 (shadow_readout),
        #   check_candidate 已有 would_reject 语义但全仓无裁决消费点 ⇒ 本块把
        #   打标前置到 S1 之前 (与 W3 forbidden 同模式), 裁决消费在
        #   multi_stage_validator.stage1_fast_ic (读 cand["_struct_dup_hit"])。
        # 位置: 符号守卫之后 (守卫会原地替换 formula, 同 W3 理由)。
        # 开关: env FUXI_STRUCT_DEDUP = off | shadow(默认, 零裁决) | enforce(硬拒)。
        #   附加开关 (仅 enforce 生效): FUXI_STRUCT_DEDUP_THRESHOLD /
        #   FUXI_STRUCT_DEDUP_ABSTRACTION / _MIN_SIZE / _MAX_FANOUT。
        self._struct_dup_stats = None
        try:
            _sd_mode = (os.environ.get("FUXI_STRUCT_DEDUP", "shadow") or "shadow").strip().lower()
            if _sd_mode not in ("off", "shadow", "enforce"):
                _sd_mode = "shadow"
            _n_sdh = 0
            _n_sdr = 0
            if _sd_mode != "off":
                import structure_dedup as _sd2
                _sd_ded = _sd2.get_indexed_deductor()
                if _sd_ded is None:
                    print("  [M10-结构去重] ⚠️ 库索引不可用, 本轮跳过 (非阻塞)")
                else:
                    for cand in candidates:
                        _sdexpr = cand.get("formula", cand.get("expression", ""))
                        if not _sdexpr:
                            cand.pop("_struct_dup_hit", None)
                            continue
                        _sdr = _sd_ded.check_candidate(_sdexpr)
                        if _sdr.get("ok") and _sdr.get("is_structural_dup"):
                            _ab = _sdr.get("abstraction", "literal")
                            _ch = (_sdr.get("per_abstraction") or {}).get(_ab, {}) or {}
                            cand["_struct_dup_hit"] = {
                                "matched_factor": _ch.get("matched_factor"),
                                "max_common_size": _ch.get("max_common_size"),
                                "abstraction": _ab,
                                "sign_same": _sdr.get("sign_same"),
                                "would_reject": bool(_sdr.get("would_reject")),
                                "reason": (f"与库内 {_ch.get('matched_factor')} 共享子树 "
                                           f"{_ch.get('max_common_size')} 节点 "
                                           f"(阈值 {_sdr.get('threshold')}, {_ab})"),
                            }
                            _n_sdh += 1
                            if _sdr.get("would_reject"):
                                _n_sdr += 1
                        else:
                            cand.pop("_struct_dup_hit", None)
            self._struct_dup_stats = {"mode": _sd_mode, "hits": _n_sdh, "would_reject": _n_sdr}
            if _n_sdh:
                print(f"  [M10-结构去重] 命中 {_n_sdh}/{len(candidates)} 候选 "
                      f"(mode={_sd_mode})" + (f" → 硬拒 {_n_sdr}" if _n_sdr else " → 仅标记"))
        except Exception as _sdge:
            print(f"  [M10-结构去重] ⚠️ 打标异常 (非阻塞): {_sdge}")

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
        # P-20260917-024: 行为同质性门禁接线所需的因子值面板 (S2 eval 顺带捕获)
        _bh_lib_panels: List = []
        _bh_cand_panels: Dict = {}
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
                        # P-20260917-024: 门禁接线开启时改用「留存面板」版 —
                        # 在同一次 eval 上顺带拿到库/候选因子值面板, 额外成本 ≈ 0
                        _bhw = _bh_wire_mode()
                        if _bhw != "off":
                            batch_res, _bh_lib_panels, _bh_cand_panels = (
                                ic_comp.compute_max_corr_vs_library_batch_with_panels(
                                    pending_formulas, lib_factors, pending_names))
                        else:
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

        # ── P-20260917-024: 行为同质性门禁**接线** (库侧 + 候选侧信号) ──
        # 病灶 (multi_stage_validator.py L470-471 早已记录, 一直未修):
        #   (1) `library_signals` 全流水线从未传入 (run_v4_pipeline 无此参数) ⇒ 恒 None;
        #   (2) `cand["signal"]` 全代码库无任何写入点 (仅读不写) ⇒ 恒 None。
        # ⇒ `_v06_evaluate_guards` 的 _n_bh_eval 恒为 0 ⇒ 门禁**零评估**,
        #   连续多轮打印「库指纹为空」但从未真正拦截过任何候选。
        # 修法: 复用 S2 已经算过的因子值面板 (零额外 eval) → 60 日每股均值向量
        #   (口径对齐 factor_value_cache) → 注入 library_signals / cand["signal"]。
        _bh_mode = _bh_wire_mode()
        if _bh_mode != "off":
            try:
                # ── P-20260917-025: Forge DSL 面板回填 (门禁**覆盖缺口**) ──
                # 病灶 (与 P-024 的「接线断线」是两件事):
                #   `compute_max_corr_vs_library_batch` 只对**可 pandas 求值**的公式
                #   留存面板 (`_eval_formula` 返回 None ⇒ 无面板)。Forge 系候选的
                #   formula 是 Forge 前缀 DSL (ts_rank/ts_std/...) ⇒ 候选侧面板恒空
                #   ⇒ `_v06_evaluate_guards` 对候选**零评估** = 门禁不设防。
                #   库侧同理: `_is_pandas_compatible` 正则把 Forge 模板整体跳过
                #   (E2E 实测「库因子: 20 个模板公式 (97 Forge/虚构列名已跳过)」)。
                # 修法: 复用既有 `self._translate_seed_formula` (Forge 表达式树 →
                #   `to_pandas_string()`, 原本用于种子重检) 把 DSL 翻成 pandas,
                #   再补一次 eval 取面板。**只补缺面板者** ⇒ 已有面板不重复 eval。
                # 🔴 隔离保证: 仅写入 `_bh_lib_panels` / `_bh_cand_panels`
                #   (门禁**专用**参照集), **不触碰** `lib_factors` ⇒
                #   S2 max_corr 的输入逐位不变, 不引入 S2 收紧副作用。
                # 开关: FUXI_BH_DSL_BACKFILL=auto(默认) / off;
                #       FUXI_BH_LIB_BACKFILL_MAX (默认 40, 库侧上限, 控成本)。
                _bh_fb = (os.environ.get("FUXI_BH_DSL_BACKFILL", "auto")
                          or "auto").strip().lower()
                if _bh_fb != "off" and ic_comp is not None:
                    # (a) 候选侧 — 门禁核心缺口 (候选必须被门禁评估)
                    _n_fb_c, _n_fb_cfail = 0, 0
                    _fb_c_why = []   # P-20260917-025 F3: 失败样例 (最多 3 条)
                    _fb_l_why = []
                    for _c in candidates:
                        _cn = str(_c.get("factor_name", "") or "")
                        if (not _cn) or (_cn in _bh_cand_panels):
                            continue
                        if _c.get("_seed_recheck"):
                            continue
                        _cf = _c.get("formula", _c.get("expression", ""))
                        if not _cf:
                            continue
                        try:
                            _px, _why = self._translate_seed_formula(str(_cf))
                        except Exception:
                            _px = ""
                        if not _px:
                            _n_fb_cfail += 1
                            if len(_fb_c_why) < 3:
                                _fb_c_why.append("%s ← %s"
                                                 % (str(_cn)[:26], str(_why)[:70]))
                            continue
                        try:
                            _pdf = ic_comp._eval_formula(str(_px), str(_cn))
                        except Exception:
                            _pdf = None
                        if _pdf is None or len(_pdf) == 0:
                            _n_fb_cfail += 1
                            continue
                        _bh_cand_panels[_cn] = _pdf
                        _n_fb_c += 1
                    # (b) 库侧 — 限量补齐被 _is_pandas_compatible 跳过的 Forge 模板
                    _n_fb_l, _n_fb_lfail = 0, 0
                    _lib_cap = int(os.environ.get("FUXI_BH_LIB_BACKFILL_MAX", "40"))
                    _have = {n for n, _ in _bh_lib_panels}
                    # P-20260917-025 F2: 上限判据原为 `len(_bh_lib_panels) >= len(_have)+cap`,
                    #   而 `_have` 每成功一条就 +1 ⇒ 两侧同步增长 ⇒ 条件**永不成立**
                    #   (实测 +86 > 上限 40, 上限形同虚设)。改锚**循环前基数**。
                    _lib_base = len(_bh_lib_panels)
                    if _lib_cap > 0:
                        for _t in templates_raw:
                            if len(_bh_lib_panels) >= _lib_base + _lib_cap:
                                break
                            if isinstance(_t, dict):
                                _ln2 = _t.get("pattern_id",
                                              _t.get("factor_name",
                                                     _t.get("name", "")))
                                _lf2 = _t.get("formula", _t.get("expression", ""))
                            elif hasattr(_t, "formula"):
                                _ln2 = getattr(_t, "pattern_id",
                                               getattr(_t, "name", ""))
                                _lf2 = getattr(_t, "formula",
                                               getattr(_t, "expression", ""))
                            else:
                                continue
                            _ln2 = str(_ln2 or "")
                            if (not _ln2) or (_ln2 in _have) or (not _lf2):
                                continue
                            if _is_pandas_compatible(str(_lf2)):
                                continue  # 已 pandas 兼容 ⇒ 已在上游处理过
                            try:
                                _px2, _w2 = self._translate_seed_formula(str(_lf2))
                            except Exception:
                                _px2 = ""
                            if not _px2:
                                _n_fb_lfail += 1
                                if len(_fb_l_why) < 3:
                                    _fb_l_why.append("%s ← %s"
                                                     % (str(_ln2)[:26], str(_w2)[:70]))
                                continue
                            try:
                                _pdf2 = ic_comp._eval_formula(str(_px2), _ln2)
                            except Exception:
                                _pdf2 = None
                            if _pdf2 is None or len(_pdf2) == 0:
                                _n_fb_lfail += 1
                                continue
                            _bh_lib_panels.append((_ln2, _pdf2))
                            _have.add(_ln2)
                            _n_fb_l += 1
                    print(f"  [v0.6-行为簇] DSL 回填 (P-025): 候选 +{_n_fb_c} "
                          f"(翻译/求值失败 {_n_fb_cfail}) / 库 +{_n_fb_l} "
                          f"(失败 {_n_fb_lfail}, 上限 {_lib_cap}, "
                          f"基数 {_lib_base})")
                    # P-20260917-025 F3: 失败必须可归因 (否则「失败 N」是黑箱)
                    for _tag, _lst in (("候选", _fb_c_why), ("库", _fb_l_why)):
                        if _lst:
                            print(f"      ↳ {_tag}翻译失败样例: " + " | ".join(_lst))

                _lib_sig = {}
                for _ln, _pdf in _bh_lib_panels:
                    _sv = _bh_panel_to_signal(_pdf)
                    if _sv is not None:
                        _lib_sig[_ln] = _sv
                _cand_pan_sig = {}
                for _cn, _pdf in _bh_cand_panels.items():
                    _sv = _bh_panel_to_signal(_pdf)
                    if _sv is not None:
                        _cand_pan_sig[_cn] = _sv
                if not library_signals and _lib_sig:
                    library_signals = _lib_sig
                _n_csig = 0
                for _c in candidates:
                    _cn = _c.get("factor_name", "")
                    if _c.get("signal") is None and _cn in _cand_pan_sig:
                        _c["signal"] = _cand_pan_sig[_cn]
                        _n_csig += 1
                print(f"  [v0.6-行为簇] 接线 mode={_bh_mode}: 库信号 {len(_lib_sig)} 条 / "
                      f"候选信号 {_n_csig} 条 (库面板 {len(_bh_lib_panels)} / "
                      f"候选面板 {len(_bh_cand_panels)})")
                if not _lib_sig and not _cand_pan_sig:
                    print("  [v0.6-行为簇] ⚠️ 接线开启但未产出任何信号 "
                          "(S2 未走到 eval 或面板列名不符) — 门禁仍将零评估")
            except Exception as _bhwe:
                print(f"  [v0.6-行为簇] ⚠️ 接线异常 (不阻塞): {_bhwe}")

        # ── P-20260831 P1: 多样性折扣影子 (Alpha2 MaxCorr 移植) ──
        # 对候选计算 vs【JQ 正面验证】因子的最大 |rank corr| (jq_max_corr),
        # 只落盘影子字段不改变任何裁决 (铁则: local 仅否决不排序)。
        # 数据用途: 观察哪些候选在复制已验证信号, 为 enforce 转正积累证据。
        try:
            from forge import diversity_discount as _dd
            _div_mode = _dd.get_mode()
            if _div_mode != "off" and ic_comp is not None:
                _dc = _dd.get_shared_cache(ic_comp=ic_comp)
                _n_ref = _dc.reference_size()
                if _n_ref > 0:
                    _div_pending_f, _div_pending_n, _div_pending_c = [], [], []
                    for cand in candidates:
                        _f = cand.get("formula", cand.get("expression", ""))
                        if _f and not cand.get("_seed_recheck"):
                            _div_pending_f.append(_f)
                            _div_pending_n.append(cand.get("factor_name", ""))
                            _div_pending_c.append(cand)
                    if _div_pending_f:
                        _corrs = _dc.get_corrs_batch(_div_pending_f, _div_pending_n)
                        for cand in _div_pending_c:
                            _n = cand.get("factor_name", "")
                            _c = _corrs.get(_n, 0.0)
                            cand["jq_max_corr"] = round(_c, 4)
                            cand["jq_div_discount"] = round(
                                _dd.discount_factor(_c), 4)
                            if abs(float(cand.get("ic", 0)) or 0) > 1e-8:
                                cand["jq_div_adj_score"] = round(
                                    abs(float(cand["ic"])) * _dd.discount_factor(_c), 6)
                        _top = sorted(_div_pending_c,
                                      key=lambda c: -c.get("jq_max_corr", 0))[:3]
                        _ts = ", ".join(
                            "%s(corr=%.2f)" % (str(c.get("factor_name", "?"))[:20],
                                               c.get("jq_max_corr", 0))
                            for c in _top)
                        _snap = _dd.COUNTERS.snapshot()
                        print(f"  [DivDisc] {_div_mode} 模式: JQ参照 {_n_ref} 因子, "
                              f"候选 {len(_div_pending_f)} 已算 jq_max_corr "
                              f"(累计查询 {_snap['n_queries']}, 成功 {_snap['n_corr_computed']}, "
                              f"自命中 {_snap['n_self_hit']}) Top占用: {_ts}")
        except Exception as _de:
            print(f"  [DivDisc] ⚠️ 影子计算失败 (非阻塞): {_de}")

        # ── v0.5.2: S5 回测过滤器 + 轻量回退 ──
        s5_filter = None
        s5_fallback = False
        try:
            from s5_joint_filter import S5JointFilter
            s5_filter = S5JointFilter(top_n=80)  # P-20260909-A: 默认全量池, FUXI_S5_UNIVERSE 可回退
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
                    y2025 = yearly.get("2025")
                    y2026 = yearly.get("2026")
                    cand["_s5_lightweight_window"] = yearly.get("_meta", {})
                    if not y2025 or not y2026:
                        cand["_s5_lightweight_eval_status"] = "data_insufficient"
                        continue
                    # 轻量 S5: 用 IC 做超额代理, ICIR 做 Calmar 代理
                    # 阈值: Calmar ≥ 0.3 (vs 真实 S5 的 1.0)
                    # ── P-20260907-006 (2026-09-11 转正): 符号感知 ──
                    # 原实现 abs(ic)/abs(icir) 无方向意识, 负向强信号被当正向通过
                    # (负向因子按原方向回测必灭)。现改为: IC 与 ICIR 需同为正才计入
                    # 超额/Calmar 代理; 负向一律记 0 → 自然落榜。
                    _y1 = float(y2025.get("ic", 0) or 0)
                    _y2 = float(y2026.get("ic", 0) or 0)
                    _ai = float(yearly.get("all", {}).get("icir", 0) or 0)
                    y1_excess = _y1 if _y1 > 0 else 0.0
                    y2_excess = _y2 if _y2 > 0 else 0.0
                    calmar_proxy = _ai if _ai > 0 else 0.0

                    cand["excess_2025"] = y1_excess
                    cand["excess_2026"] = y2_excess
                    cand["calmar"] = calmar_proxy
                    cand["_s5_lightweight"] = True
                    cand["_s5_sign_aware"] = True
                    cand["_s5_lightweight_eval_status"] = "valid"
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
                # P-20260907-002: 条件分解报告落盘 trajectory (供 D+ 蒸馏归因)
                if sem_result.get("condition_report"):
                    _cr = sem_result["condition_report"]
                    cand["semantic_cond_pass_rate"] = _cr.get("pass_rate")
                    cand["semantic_cond_report"] = [
                        {"type": c.get("type"), "matched": c.get("matched"),
                         "evidence": c.get("evidence", "")[:200]}
                        for c in _cr.get("conditions", [])
                    ]
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

        # ── P-20260907-006: 方向符号守卫读数透传到 summary (日报消费) ──
        _sg_stats = getattr(self, "_sign_guard_stats", None)
        if _sg_stats:
            summary["sign_guard"] = _sg_stats
            if _sg_stats.get("n_flipped") or _sg_stats.get("n_reject"):
                print(f"  [S5-SignGuard] 本轮方向修正: 翻转 {_sg_stats.get('n_flipped',0)} "
                      f"/ 拒负向 {_sg_stats.get('n_reject',0)} "
                      f"(mode={_sg_stats.get('mode')})")

        # ── P-20260911-003: 执行陷阱守卫影子 (GoAnt Execution Trap, 零裁决) ──
        try:
            from exec_trap_guard import evaluate as _exec_trap_eval
            _et = _exec_trap_eval(candidates, results, verbose=True)
            summary["exec_trap"] = {
                "mode": _et.get("mode"),
                "n_checked": _et.get("n_checked", 0),
                "n_exec_trap": _et.get("n_exec_trap", 0),
            }
        except Exception as _ete:
            print(f"  [ExecTrap] ⚠️ 影子计算失败 (非阻塞): {_ete}")

        # ── P-20260912-002: 幸存者偏差守卫影子 (环境自身样本构成审计, 零裁决) ──
        # 复用本轮已加载的 IC 计算器实例, 避免重复读 560MB 面板。
        # env FUXI_SURVIVORSHIP_GUARD = shadow(默认) | off
        try:
            from survivorship_guard import (
                attach_computer as _surv_attach, evaluate as _surv_eval)
            _surv_attach(getattr(self, "_last_ic_comp", None))
            _sv = _surv_eval(candidates, results, verbose=True)
            summary["survivorship_guard"] = {
                "mode": _sv.get("mode"),
                "n_checked": _sv.get("n_checked", 0),
                "n_survivorship_risk": _sv.get("n_survivorship_risk", 0),
                "n_fragile_dependent": _sv.get("n_fragile_dependent", 0),
                "n_starved_paradigms": _sv.get("n_starved_paradigms", 0),
            }
        except Exception as _sve:
            print(f"  [SurvGuard] ⚠️ 影子计算失败 (非阻塞): {_sve}")

        # ── P-20260915-001: 风格捷径阴影诊断 (仅归因, 零裁决) ──
        # 🔴🔴 硬约束 (用户 2026-09-15 钦定): 本守卫**只产出归因读数**, 其回归残差 /
        #     中性化结果严禁用于: 因子中性化处理 / 因子入库 / 候选改写 / 裁决输入 /
        #     生成器上下文注入。V5 中性化路线已证伪 (JQ +30.77% / Sharpe 0.06 /
        #     DD -26.52%, 中性化+HRP = alpha 杀手, 2026-07-30 正式放弃)。
        # 复用本轮已加载的 IC 计算器实例, 避免重复读面板。
        # env FUXI_STYLE_GUARD = on|true|1|shadow (默认, 生效) | off
        try:
            from style_exposure_guard import (
                attach_computer as _sty_attach, evaluate as _sty_eval)
            _sty_attach(getattr(self, "_last_ic_comp", None))
            _st = _sty_eval(candidates, results, verbose=True)
            summary["style_guard"] = {
                "mode": _st.get("mode"),
                "n_checked": _st.get("n_checked", 0),
                "n_shortcut_risk": _st.get("n_shortcut_risk", 0),
                "n_high_style_r2": _st.get("n_high_style_r2", 0),
                "median_ic_decay_ratio": _st.get("median_ic_decay_ratio", 0.0),
                "dominant_style_dist": _st.get("dominant_style_dist", {}),
            }
        except Exception as _ste:
            print(f"  [StyleGuard] ⚠️ 影子计算失败 (非阻塞): {_ste}")

        # ── P-20260918-002 (收缩版): 成本敏感性影子诊断 (仅归因, 零裁决) ──
        # 🔴🔴 硬约束 (用户 2026-09-18 钦定, 收缩自「交易门」提案):
        #     本模块**只输出成本档位梯度读数**, 用的是本地 5 日 fwd 代理面板,
        #     不是 S5 引擎 ⇒ 绝对收益不可与 S5/JQ 比较, 仅跨成本档位梯度有意义。
        #     严禁用于: 因子准入 / S1·S2·S5 裁决输入 / 候选改写 / 生成器注入 /
        #     取代 S5 已有的 P0-g 成本计费 (15bp) 与 TRADABLE_GATE。
        # 复用本轮已加载的 IC 计算器实例, 避免重复读面板。
        # env FUXI_TRADE_GATE = shadow(默认, 生效) | off
        try:
            from trade_gate_shadow import (
                attach_computer as _tg_attach, evaluate as _tg_eval)
            _tg_attach(getattr(self, "_last_ic_comp", None))
            _tgr = _tg_eval(candidates, results, verbose=True)
            summary["trade_gate"] = {
                "mode": _tgr.get("mode"),
                "n_checked": _tgr.get("n_checked", 0),
                "n_fragile": _tgr.get("n_fragile", 0),
                "n_steep": _tgr.get("n_steep", 0),
                "n_high_drag": _tgr.get("n_high_drag", 0),
                "n_unreliable": _tgr.get("n_unreliable", 0),
                "median_turnover": _tgr.get("median_turnover", 0.0),
                "median_drag_ratio_30": _tgr.get("median_drag_ratio_30"),
            }
        except Exception as _tge:
            print(f"  [TradeGate] ⚠️ 影子计算失败 (非阻塞): {_tge}")

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
        # P-20260915-014: 行为簇门禁有效性随 summary 落盘 (结果 JSON 可查)
        summary["behavior_gate_effective"] = bool(
            v06_eval.get("behavior_gate_effective", False))
        # P-20260917-024: 接线状态 + shadow 本应硬拒条数一并落盘 (供日报/看板消费)
        summary["behavior_gate_mode"] = v06_eval.get("behavior_gate_mode", "off")
        summary["behavior_would_reject"] = int(
            v06_eval.get("behavior_would_reject", 0) or 0)
        if v06_eval.get("behavior_flagged"):
            print(f"  [v0.6-行为簇] 冗余标记: {v06_eval['behavior_flagged']} 候选 "
                  f"(NEAR_DUPLICATE/SUBSTITUTE), 硬拒 {v06_eval.get('behavior_rejected', 0)}"
                  f" / 本应硬拒 {v06_eval.get('behavior_would_reject', 0)}"
                  f" (mode={v06_eval.get('behavior_gate_mode', 'off')})")
        if v06_eval.get("holdout_verdicts"):
            print(f"  [v0.6-盲评] 密封盲评: {v06_eval['holdout_verdicts']} 候选 "
                  f"(verdict 附回 candidate['_holdout_verdict'])")
        # M13 (2026-09-27): 增量门快照/分母/队列/异常随 summary 落盘
        # (供"控制组版本漂移次数应为 0""每轮配对覆盖数"两个指标直接对账)
        summary["incremental_status"] = v06_eval.get("incremental_status", "not_evaluated")
        summary["incremental_control"] = v06_eval.get("incremental_control", {}) or {}
        summary["incremental_coverage"] = v06_eval.get("incremental_coverage", {}) or {}
        summary["incremental_deferred"] = v06_eval.get("incremental_deferred", {}) or {}
        summary["incremental_exceptions"] = int(
            v06_eval.get("incremental_exceptions", 0) or 0)
        _inc_probe = v06_eval
        if (_inc_probe.get("incremental_checked")
                or _inc_probe.get("incremental_exceptions")
                or (_inc_probe.get("incremental_deferred", {}) or {}).get("n_deferred")):
            _cov = _inc_probe.get("incremental_coverage", {}) or {}
            _dec = _inc_probe.get("incremental_deferred", {}) or {}
            _ctl = _inc_probe.get("incremental_control", {}) or {}
            print(f"  [v0.6-增量] 配对增量: {_inc_probe['incremental_checked']} 候选, "
                  f"拒绝 {_inc_probe.get('incremental_rejected', 0)}"
                  f" | 分母: eligible={_cov.get('n_eligible', 0)} "
                  f"attempted={_cov.get('n_pairs_attempted', 0)} "
                  f"budget={_cov.get('budget', '?')} "
                  f"| 补数队列 {_dec.get('n_deferred', 0)}"
                  f" | 异常 {_inc_probe.get('incremental_exceptions', 0)}"
                  f" | control={str(_ctl.get('snapshot_sha256') or '')[:12]}"
                  f"(rev={_ctl.get('revision')}, drift={_ctl.get('drift_detected')})")
        # ── M14 (2026-09-27): 组合层正交化影子诊断 (仅归因, 零裁决) ──
        # 🔴🔴 硬约束 (承接用户 2026-09-15 钦定 + 2026-07-30 V5 证伪先例):
        #     本块只产出归因读数 (base_r2 / 三口径 IC), 其回归残差**算完即弃**,
        #     严禁用于因子中性化处理/因子入库/候选改写/裁决输入/生成器注入。
        #     它只回答"候选的表观增量里有多少是基组合重复信息", 不改变任何判定
        #     (不碰 eligible_for_jq / 不进 incremental gate / 不改候选字典)。
        #     control 与 M13 同源 (同一快照); 快照缺失 ⇒ data_insufficient, 不兜底。
        # env FUXI_COMBO_ORTHO = shadow (默认, 生效但零裁决) | off
        try:
            from combo_orthogonalize import (
                attach_computer as _co_attach, evaluate as _co_eval)
            _co_attach(getattr(self, "_last_ic_comp", None))
            _co = _co_eval(candidates, results, verbose=True)
            summary["combo_ortho"] = {
                "mode": _co.get("mode"),
                "status": _co.get("status"),
                "n_checked": int(_co.get("n_checked", 0) or 0),
                "n_replicated": int(_co.get("n_replicated", 0) or 0),
                "n_weak_increment": int(_co.get("n_weak_increment", 0) or 0),
                "n_increment_candidate": int(_co.get("n_increment_candidate", 0) or 0),
                "median_base_r2": _co.get("median_base_r2"),
                "control_sha256": (_co.get("control") or {}).get("snapshot_sha256"),
            }
        except Exception as _coe:
            print(f"  [ComboOrtho] ⚠️ 影子诊断失败 (非阻塞): {_coe}")

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
                    # P-20260901-005: 父因子上下文层 (影子, 只记录不裁决)
                    _em = _cand.get("edit_meta") or {}
                    for _pf in (_em.get("parents") or []):
                        if _pf:
                            self.edit_memory.record_parent(
                                _pf, _op, _resid,
                                child_formula=_cand.get("formula", ""))
                    _n_sspm += 1
                if _n_sspm:
                    print(self.edit_memory.summary())
                    try:
                        print(self.edit_memory.parent_overlap_report())
                    except Exception:
                        pass
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
                    # P-20260902-005: 证伪判据随轨迹落盘 (D+ 与 JQ 结果对拍)
                    refutation_conditions=str(cand.get("refutation_conditions", "") or ""),
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
        if (hasattr(self, 'mab') and self.mab and
                os.environ.get("FUXI_LEGACY_S1_LOCAL_REWARD", "off").strip().lower() == "on"):
            # P-20260915-012: 空范式显式提示 (每轮一次, 不逐候选刷屏)。
            # 缺 paradigm 的候选 _arms=[] 静默跳过, 历史上无任何日志可查。
            _n_no_par = sum(1 for _c in candidates
                            if not str(_c.get('paradigm', '') or '').strip())
            if _n_no_par:
                print(f"  [S2a] {_n_no_par}/{len(candidates)} 候选缺 paradigm 字段, "
                      f"本地引导对这部分跳过 (接线检查: P-20260915-012)")
            for cand, result in zip(candidates, results):
                icir = abs(float(getattr(result, 's1_icir', 0) or 0))
                if icir < 0.03:
                    continue  # ICIR 太低不配引导

                is_jq = getattr(result, 'eligible_for_jq', False)
                signal_mult = 1.0 if is_jq else 0.1  # 非JQ候选 微弱信号
                local_reward = np.tanh(icir * 3) * 0.3 * signal_mult

                cand_paradigm = str(cand.get('paradigm', ''))

                # S2-a (2026-09-12): 口径/作用域修正。
                # 原实现遍历 `self.mab.directions.values()` 命中即 `break` ⇒
                # 范式级信号只落到**字典序首个**臂 (恒为 `__equal_weight_rank`,
                # 因插入顺序固定) ⇒ 三模型臂中只有 Ensemble 收到引导, 另两个
                # 长期饿死, UCB 失去区分度。
                # 范式级信号应广播给该范式下**全部** model 臂。
                # 开关 FUXI_MAB_PARADIGM_BROADCAST (P-20260915-013): 默认从 shadow
                # 升级为 on — 范式级信号广播给该范式下全部 model 臂, 消除 Ensemble
                # 独食偏差 (UCB 区分度恢复)。本地信号上限 ±0.5、非 JQ 再 ×0.1,
                # 且 S5 组合超额终裁不受影响 ⇒ 风险封顶为探索浪费。
                # 回退: 设 FUXI_MAB_PARADIGM_BROADCAST=shadow。
                import os as _os
                _bc = _os.environ.get("FUXI_MAB_PARADIGM_BROADCAST", "on").strip().lower()
                _arms = self.mab.resolve_arms(cand_paradigm) if cand_paradigm else []
                _arms = [d for d in _arms if d.status != "cooling"]
                if _bc != "on":
                    _arms = _arms[:1]
                if _bc == "shadow" and len(_arms) == 1:
                    _all = self.mab.resolve_arms(cand_paradigm) if cand_paradigm else []
                    if len(_all) > 1:
                        print(f"  [S2a/shadow] 范式 {cand_paradigm!r} 本地引导本应广播 "
                              f"{len(_all)} 臂, 实际仅 {_arms[0].name} "
                              f"(FUXI_MAB_PARADIGM_BROADCAST=on 启用)")
                _touched = 0
                for d in _arms:
                    n = max(d.pulls, 1)
                    new_reward = (d.expected_reward * n + local_reward) / (n + 1)
                    d.expected_reward = max(-0.5, min(0.5, new_reward))
                    _touched += 1
                if _touched:
                    self.mab.save_state()

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
            # P-20260902-002: IC 衰减半衰期影子统计 (仅标注, 不参与裁决)
            "decay_shadow": {
                "computed": n_decay_computed,
                "fast": n_decay_fast,
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

        # Legacy Formation defaulted every candidate without an explicit
        # outcome to WEAK, then evolved that fabricated negative memory.
        # A5 stores status-aware observations separately until a verified
        # economic result can be joined to the originating decision.
        if os.environ.get("FUXI_LEGACY_MEMORY_FORMATION", "off").strip().lower() == "on":
            formation_result = self.memory.form(
                trajectory={
                    "batch_id": f"ralph_{datetime.now().strftime('%Y%m%d')}",
                    "candidates": candidates,
                },
                auto_save=False,
            )
            evolution_result = self.memory.evolve(auto_save=True)
        else:
            formation_result = {"new_success_patterns": 0,
                                "new_forbidden_directions": 0,
                                "status": "awaiting_verified_economic_label"}
            evolution_result = {"merged_patterns": 0,
                                "upgraded_forbidden": 0,
                                "archived_stale": 0,
                                "status": "skipped_unverified_memory"}

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

    def _lookup_traj_freq(self, name: str) -> str:
        """P-20260903-004: 从轨迹中查因子 natural_freq (Step 2.7 lane 更新补 freq 用)。"""
        if not name:
            return ""
        try:
            for traj in self.traj_logger.trajectories.values():
                if name in str(getattr(traj, "seed_factor", "") or ""):
                    return str(getattr(traj, "natural_freq", "") or "")
        except Exception:
            return ""
        return ""

    # ── P-20260904-002: weekly 补给跨进程轮次计数 ──
    def _weekly_supply_state_path(self) -> Path:
        return Path(self.data_dir) / "weekly_supply_state.json"

    def _load_weekly_supply_count(self) -> int:
        """读取已完成的 Ralph 轮次累计数 (0 = 尚未执行过任何一轮)。"""
        try:
            p = self._weekly_supply_state_path()
            if p.exists():
                return int(json.load(open(p, encoding="utf-8")).get("total_loops", 0) or 0)
        except Exception:
            pass
        return 0

    def _bump_weekly_supply_count(self) -> int:
        """run() 结束时调用: 累计轮次 +1 落盘 (幂等失败不阻塞)。"""
        n = self._load_weekly_supply_count() + 1
        try:
            json.dump({"total_loops": n,
                       "updated_at": datetime.now().isoformat()},
                      open(self._weekly_supply_state_path(), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        except Exception:
            pass
        return n

    def jq_feedback(
        self,
        jq_backtest_result: Dict,
        trajectory_ids: Optional[List[str]] = None,
        evo_trajectories: Optional[List[Dict]] = None,  # v0.6: EvoTraj summaries
        update_lane: bool = True,  # P-20260903-004: 是否同步 lane 校准集 (trigger_d_plus 传 False 避免同批次双更新)
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

        # ── S2-a/S2-c (2026-09-12): 轨迹通道自动回填 ──────────────
        # 根因: 全库 ~20 处 `jq_feedback(...)` 调用**无一传入 evo_trajectories**
        # ⇒ 下方 Step 3.5 的 "EvoTraj streak → MAB" 恒不执行 (死接线),
        # 轨迹知识永远回不到调度器。此处加自动回填: 未显式传入时, 依次尝试
        #   (1) 实例通道 self._last_evo_trajectories (同进程先后调用)
        #   (2) 最近一个 data/ralph_loop_result_*.json 的 evo_trajectories
        #     (跨进程, automation 走的是 "跑 pipeline → 另行注入 JQ 结果" 模式)
        # 只在 A 链 (轨迹) 上补数据, 不改任何裁决口径。
        if not evo_trajectories:
            try:
                _inst = getattr(self, "_last_evo_trajectories", None)
                if _inst:
                    evo_trajectories = [
                        (t.get_summary() if hasattr(t, "get_summary") else dict(t))
                        for t in _inst
                    ]
                    print(f"  [S2a] 轨迹通道回填 (实例): {len(evo_trajectories)} 条")
            except Exception:
                evo_trajectories = None
        if not evo_trajectories:
            try:
                import glob as _glob
                _dd = Path(self.data_dir)
                _files = sorted(_glob.glob(str(_dd / "ralph_loop_result_*.json")))
                for _fp in reversed(_files[-6:]):
                    _d = json.load(open(_fp, encoding="utf-8"))
                    _et = _d.get("evo_trajectories") or []
                    if _et:
                        evo_trajectories = _et
                        print(f"  [S2a] 轨迹通道回填 (落盘 {Path(_fp).name}): "
                              f"{len(_et)} 条")
                        break
            except Exception as _fe:
                print(f"  [S2a] ⚠️ 轨迹通道落盘回填失败 (不阻塞): {_fe}")

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
                        # 2026-09-18 (P-20260918-021): 富字段落到轨迹一级字段,
                        # 而非只进 backtest.meta; 并**就地打 D+ 标记**, 消除
                        # 「直接调 jq_feedback 的条目天生欠账」这一独立欠账源。
                        jq_maxdd=f.get('jq_maxdd'),
                        jq_ic=f.get('jq_ic'),
                        jq_icir=f.get('jq_icir'),
                        jq_rating=f.get('jq_rating'),
                        jq_plus_ic_pct=f.get('jq_plus_ic_pct'),
                        jq_backtest_period=f.get('jq_backtest_period'),
                        jq_notes=(f.get('root_cause') or None),
                        mark_d_plus=True,
                        d_plus_source='jq_feedback',
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
                    # 2026-09-18 (P-20260918-021): 同上 — 新建轨迹分支同步补富字段+打标
                    jq_maxdd=f.get('jq_maxdd'),
                    jq_ic=f.get('jq_ic'),
                    jq_icir=f.get('jq_icir'),
                    jq_rating=f.get('jq_rating'),
                    jq_plus_ic_pct=f.get('jq_plus_ic_pct'),
                    jq_backtest_period=f.get('jq_backtest_period'),
                    jq_notes=(f.get('root_cause') or None),
                    mark_d_plus=True,
                    d_plus_source='jq_feedback',
                )
                trajectories_updated += 1

        print(f"  [Trajectory] 更新了 {trajectories_updated} 条轨迹 (含 d_plus_applied 标记, source=jq_feedback)")

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
                # P2 (2026-09-05): 引擎溯源 — 优先 JQ 结果自带 source/engine,
                # 否则按因子名回溯 Memory attempts
                _eng = _norm_engine(
                    f.get("source") or f.get("engine")
                    or self.memory.get_engine_of(f.get("factor_name", ""))
                )
                self.memory.distill_motif_knowledge(
                    formula=formula,
                    jq_passed=jq_passed,
                    jq_marginal=jq_marginal,
                    fsa=self.fsa,
                    interpretation=f.get("interpretation"),  # L1 (2026-08-14)
                    engine=_eng,
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

        # ── Step 2.7: v0.7 P4 lane 校准集滚动更新 (P-20260903-004) ──
        # 此前该步骤只在 trigger_d_plus 里, 一次性反馈脚本直接调 jq_feedback 会
        # 静默漏步 (2026-09-02 vwap 反转漏 2 锚点)。现内联到 jq_feedback:
        #   - update_lane=False 时跳过 (trigger_d_plus 自行调用, 防同批次双更新)
        #   - 幂等 upsert (by formula key) + recalibrate 防重守卫 (300s)
        #   - natural_freq 缺失时从轨迹补 (daily 因子被校准集过滤, 保守安全)
        lane_report = None
        if update_lane:
            try:
                from lane_calibration import update_from_jq_feedback, recalibrate_tau_w
                _lane_factors = []
                for _f in factors:
                    _fe = dict(_f)
                    if not _fe.get("natural_freq"):
                        _freq = self._lookup_traj_freq(_fe.get("factor_name", ""))
                        if _freq:
                            _fe["natural_freq"] = _freq
                    _lane_factors.append(_fe)
                _up = update_from_jq_feedback(_lane_factors)
                print(f"  [v0.7 P4] lane 校准集滚动更新: 新增 {_up['n_new']} / "
                      f"更新 {_up['n_updated']} / 跳过 {_up['n_skipped']} "
                      f"(周频复评成功 {_up['n_eval_ok']})")
                lane_report = recalibrate_tau_w()
                print(f"  [v0.7 P4] τ_w 重估: {str(lane_report.get('reason', ''))[:150]}")
            except Exception as e:
                print(f"  [v0.7 P4] ⚠️ lane 校准更新失败 (不阻塞 D+): {e}")

        # ── Step 3: E (Evolution) — 合并/去重 ─────────────
        print(f"\n  [Memory] 演化合并...")
        evolution_result = self.memory.evolve(auto_save=True)

        # ── Step 3.4: (M1 影子半程) 奖励守卫 — 已验证经济事件的影子消费 ──
        # 只读消费: 默认 shadow 只落 data/reward_guard_ledger.jsonl, MAB 臂零改动;
        # 五态守卫: 非 valid 事件永不奖惩; NaN/缺失 jq_return 视为无读数。
        # 写生效需显式 FUXI_REWARD_GUARD=enforce（新增开关, 翻默认前须 A/B）。
        reward_guard_report = {"mode": "off", "consumed": 0, "rewardable": 0,
                               "no_reward": 0, "no_arms": 0, "applied": 0}
        try:
            from reward_guard import guard_mode
            _rg_mode = guard_mode()
            reward_guard_report["mode"] = _rg_mode
            if _rg_mode != "off":
                for _f in factors:
                    _pgn = str(_f.get("paradigm", "") or "")
                    if not _pgn:
                        continue
                    _ev = _f.get("evaluation_status")
                    if isinstance(_ev, dict):
                        _s5 = str(_ev.get("s5", "") or "")
                    else:
                        _s5 = str(_ev or "")
                    _res = self._mab_record_result(
                        _pgn, _f.get("jq_return"), _f.get("jq_sharpe"),
                        evidence={"eval_status": _s5, "source": "jq",
                                  "ref": str(_f.get("factor_name", "") or "")})
                    reward_guard_report["consumed"] += 1
                    _st = _res.get("status")
                    if _st == "applied":
                        reward_guard_report["applied"] += 1
                    elif _st == "shadow_only":
                        reward_guard_report["rewardable"] += 1
                    elif _st == "no_arms":
                        reward_guard_report["no_arms"] += 1
                    elif _st == "no_reward":
                        reward_guard_report["no_reward"] += 1
                print(f"  [M1 奖励守卫/{_rg_mode}] 影子消费 {reward_guard_report['consumed']} 条: "
                      f"可奖惩 {reward_guard_report['rewardable']} / "
                      f"无臂 {reward_guard_report['no_arms']} / "
                      f"五态守卫拦截 {reward_guard_report['no_reward']}")
        except Exception as _rge:
            print(f"  [M1 奖励守卫] ⚠️ 影子消费失败 (不阻塞 D+): {_rge}")
        # ── Step 3.5: v0.6 EvoTraj streak MAB 更新 ─────────
        mab_streak_updates = 0
        if (evo_trajectories and
                os.environ.get("FUXI_LEGACY_EVO_STREAK_REWARD", "off").strip().lower() == "on"):
            for traj_summary in evo_trajectories:
                paradigm = traj_summary.get("paradigm", "")
                streak = traj_summary.get("streak", 0)
                best_icir = traj_summary.get("best_icir", 0)
                traj_reward = traj_summary.get("trajectory_reward", 0)
                if paradigm and streak > 0:
                    # S2-a (2026-09-12): 原实现 `d.name == paradigm or
                    # d.direction_id == paradigm` 是**死匹配** —— 臂 name 形如
                    # "3.动量反转 × LightGBM"、direction_id 形如
                    # "动量反转__lightgbm_rank", 而这里 paradigm 是纯范式名
                    # ⇒ 恒不命中, streak 奖励**从未写回任何臂** (静默)。
                    # 改用 resolve_arms (paradigm 精确 → 该范式下全部 model 臂)。
                    _arms = self.mab.resolve_arms(paradigm)
                    if not _arms:
                        print(f"  [MAB] ⚠️ EvoTraj streak={streak} 范式 {paradigm!r} "
                              f"无对应臂 (奖励未写回)")
                        continue
                    # 使用 JQ composite return 作为 baseline
                    jq_return = jq_backtest_result.get("composite_return", 0)
                    relative = (jq_return - 182.57) / 100.0
                    streak_multiplier = 1.0 + 0.2 * streak
                    reward = max(-1.0, min(1.0, relative * 5 * streak_multiplier))
                    for d in _arms:
                        try:
                            self.mab.ensure_reward_source(d, "jq")  # S2-c iii 量纲
                        except Exception:
                            pass
                        _n = int(getattr(d, "n_rewards", 0) or 0) + 1
                        d.n_rewards = _n
                        d.expected_reward = (d.expected_reward * (_n - 1) + reward) / _n
                        d.jq_results.append({
                            "return": jq_return,
                            "sharpe": jq_backtest_result.get("composite_sharpe", 0),
                            "streak": streak,
                            "traj_reward": traj_reward,
                            "best_icir": best_icir,
                        })
                    self.mab.save_state()
                    mab_streak_updates += 1
                    print(f"  [MAB] EvoTraj → {len(_arms)} 臂 [{paradigm}]: streak={streak}, "
                          f"reward={reward:.3f} ({streak_multiplier:.1f}x)")

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
                if (verdict == "direction_falsified" and
                        os.environ.get("FUXI_LEGACY_L1_REWARD", "off").strip().lower() != "on"):
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
            "reward_guard": reward_guard_report,  # M1 (2026-09-26)
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
    from pathlib import Path as _FuxiPath
    import sys as _FuxiSys
    _FuxiSys.path.insert(0, str(_FuxiPath(__file__).resolve().parents[2]))
    from fuxi_v1.legacy_writers import legacy_cli_guard as _FuxiGuard
    _fuxi_guard = _FuxiGuard(__file__)
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
