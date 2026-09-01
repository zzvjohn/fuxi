"""
LLM 因子生成器: Experience Memory 反馈驱动的 LLM 因子生成。
在统一闭环中作为 GP 育种的补充生成器。
"""
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

DATA_DIR = Path(os.environ.get("FUXI_DATA_DIR") or str(Path(__file__).resolve().parent.parent.parent / "data"))
OUTPUT_DIR = Path(os.environ.get("FUXI_OUTPUT_DIR") or str(Path(__file__).resolve().parent.parent.parent / "output"))

# P-20260821-001: AQuA 密封研究契约开关 (rollback: 置 False 即回退裸 prompt)
_RESEARCH_CONTRACT_ENABLED = True


def _load_v06_flag(key: str, default=False) -> bool:
    """读取 V06_EXPERIMENTAL 开关 (config 导入失败时静默回退默认值)"""
    try:
        from config import V06_EXPERIMENTAL
        return bool(V06_EXPERIMENTAL.get(key, default))
    except Exception:
        return default


def _extract_recent_failures(limit: int = 3, days: int = 7) -> str:
    """从 experience_memory.attempts 提取近期失败原因 (lazy 回灌, 失败时静默)"""
    try:
        from datetime import timedelta
        em_path = DATA_DIR / "experience_memory.json"
        if not em_path.exists():
            return ""
        with open(em_path, "r", encoding="utf-8") as f:
            em = json.load(f)
        attempts = em.get("attempts", [])
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        fails = []
        for a in reversed(attempts):
            if len(fails) >= limit:
                break
            ts = a.get("timestamp", "")
            if ts < cutoff:
                continue
            outcome = a.get("outcome", "")
            if outcome in ("PASS", "PASSED", "MARGINAL"):
                continue
            name = a.get("factor_name", "?")[:50]
            lessons = a.get("lessons", [])
            lesson_txt = "; ".join(str(l)[:80] for l in lessons[:2]) if lessons else "无归因记录"
            fails.append(f"- {name} ({outcome}): {lesson_txt}")
        return "\n".join(fails)
    except Exception:
        return ""


class LLMGenerator:
    """
    基于 Experience Memory 的 LLM 因子生成器。
    
    工作流:
    1. receive_context() 接收 Memory priors (成功模板/禁止方向/FSA/库状态)
    2. build_prompt() 构建结构化 LLM prompt
    3. [手动] 用户将 prompt 提交给 LLM
    4. parse_response() 解析 LLM 响应为候选因子
    5. get_candidates() 返回标准化候选列表
    
    也支持通过 MCP/API 直接调用 (如未来接入 LLM API)。
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._context = {}
        self._last_prompt_file = None
        self._gen_counter = 0
        # P-20260830-001: 周频语境开关 (默认 False = 日频语境, 回退即关闭)
        self._weekly_mode = False
    
    # ── P-20260830-001: 周频语境开关 ──────────────────────
    
    def set_weekly_mode(self, weekly: bool = True) -> None:
        """开启/关闭周频生成语境。开启后 build_prompt 注入周频语义约束,
        且 generate() 解析出的候选自动打 natural_freq=weekly 标签 (走 weekly lane)。"""
        self._weekly_mode = bool(weekly)
    
    # ── 上下文注入 ───────────────────────────────────────
    
    def receive_context(
        self,
        success_templates: List = None,
        forbidden_directions: List = None,
        fsa_forbidden: str = "",
        library_stats: Dict = None,
        paradigm_priorities: Dict = None,
        mab_direction: str = "",
        recent_performance: Dict = None,
        # v3.1: 软负收益警告 + Motif 规则
        warning_directions: List = None,
        motif_rules: Dict = None,
        # v0.5: RAG 知识注入
        rag_knowledge: Dict = None,
        # P-20260821-001: AQuA 研究契约 — 上轮失败原因回灌
        last_round_failures: str = "",
    ):
        """接收 Experience Memory 检索结果 + RAG 知识库注入"""
        self._context = {
            "success_templates": success_templates or [],
            "forbidden_directions": forbidden_directions or [],
            "fsa_forbidden": fsa_forbidden,
            "library_stats": library_stats or {},
            "paradigm_priorities": paradigm_priorities or {},
            "mab_direction": mab_direction,
            "recent_performance": recent_performance or {},
            "warning_directions": warning_directions or [],
            "motif_rules": motif_rules or {},
            "rag_knowledge": rag_knowledge or {},
            "last_round_failures": last_round_failures,
        }
    
    # ── Prompt 构建 ──────────────────────────────────────
    
    def build_prompt(self, target_paradigm: str = "", n_factors: int = 5,
                     weekly_mode: Optional[bool] = None) -> str:
        """
        构建 LLM prompt: 包含成功模板、禁止方向、FSA约束、库状态。
        返回可直接发给 LLM 的文本。

        P-20260830-001: weekly_mode=True 时注入周频语义约束 —
        表达式将在周频宽表 (W-FRI 采样) 上 eval, 窗口口径与日频完全不同,
        fwd_ret = 1 周持有; 生成公式走 weekly lane 裁决 (τ_w)。
        """
        if weekly_mode is None:
            weekly_mode = self._weekly_mode
        ctx = self._context
        parts = []
        
        if weekly_mode:
            parts.append("# A股量化因子生成任务 (周频语境 weekly lane)")
            parts.append("")
            parts.append("你是一位专业量化研究员，请基于以下经验记忆生成新的**周频因子**表达式。")
            parts.append("这些因子将在周频数据 (每周最后一个交易日采样, W-FRI) 上求值, ")
            parts.append("持有周期 = 1 周 (本周信号 → 下周收益)。")
            parts.append("")
            parts.append("## 📐 周频语境硬约束 (与日频语义完全不同, 必须遵守)")
            parts.append("1. 窗口含义: rolling(1)=1周, rolling(4)=约1月, rolling(12)=约1季度, rolling(20)=约5个月")
            parts.append("2. 周频求值环境字段: close open high low volume amount (均为周频宽表 Series)")
            parts.append("   - 周频 OHLC: open=周首日开盘, high=周内最高, low=周内最低 (真实周内路径)")
            parts.append("3. 可设计的周频特有结构: 周内路径形态 (如 (close-open)/(high-low) 光头周线), ")
            parts.append("   周度资金累积 (volume/amount 的周频滚动), 周频均值回复 (周收益的 zscore 反转)")
            parts.append("4. 禁止用日频直觉设计窗口: 周频 rolling(20) 已是 5 个月, 勿套用日频 20 日均线语义")
            parts.append("5. 表达式语法仍为 pandas Rolling 风格 (见下方语法要求)")
            parts.append("")
            # P-20260901-004: 周频原生种子 few-shot 参考 (生成侧与裁决侧频率对齐)
            try:
                from weekly_seed_pool import format_prompt_reference
                parts.append("## 🌱 周频原生种子参考 (结构风格示例, 请设计新的而非复制)")
                parts.append(format_prompt_reference(max_n=6))
                parts.append("")
            except Exception:
                pass
        else:
            parts.append("# A股量化因子生成任务")
            parts.append("")
            parts.append("你是一位专业量化研究员，请基于以下经验记忆生成新的因子表达式。")
            parts.append("表达式必须使用 Forge DSL 语法(如 ts_mean, ts_std, rank, sub, div, mul 等)。")
            parts.append("")
        
        # 目标范式
        if target_paradigm:
            parts.append(f"## 目标探索范式: {target_paradigm}")
            parts.append(f"请优先围绕'{target_paradigm}'范式设计因子。")
        elif ctx.get("mab_direction"):
            parts.append(f"## MAB推荐方向: {ctx['mab_direction']}")
        
        parts.append("")
        
        # 成功模板 (来自 Experience Memory — 曾经有效的模式)
        templates = ctx.get("success_templates", [])
        if templates:
            parts.append("## ✅ 历史成功模板 (建议参考这些模式)")
            parts.append("以下是过去验证有效的因子模式，可以作为变体设计参考:")
            for i, t in enumerate(templates[:10]):
                if isinstance(t, dict):
                    formula = t.get("formula", t.get("expression", ""))
                    desc = t.get("description", t.get("pattern_id", f"template_{i}"))
                    paradigm = t.get("paradigm", "")
                    occ = t.get("occurrence_count", 1)
                    jq_tag = ""
                    if t.get("jq_return"):
                        jq_tag = f" [JQ:{t['jq_return']:+.1f}%]"
                    parts.append(f"- {desc}{jq_tag}: `{formula[:120]}` (范式={paradigm}, occ={occ})")
                elif hasattr(t, 'description'):
                    formula = getattr(t, 'formula', getattr(t, 'expression', ''))
                    desc = getattr(t, 'description', str(t))
                    paradigm = getattr(t, 'pattern_id', '').split('::')[0]
                    occ = getattr(t, 'occurrence_count', 1)
                    jq_tag = ""
                    jq_ret = getattr(t, 'jq_return', None)
                    if jq_ret:
                        jq_tag = f" [JQ:{jq_ret:+.1f}%]"
                    parts.append(f"- {desc}{jq_tag}: `{str(formula)[:120]}` (范式={paradigm}, occ={occ})")
            parts.append("")
        
        # 禁止方向
        forbidden = ctx.get("forbidden_directions", [])
        if forbidden:
            parts.append("## 🚫 硬禁止方向 (绝对不能生成)")
            for d in forbidden[:8]:
                if isinstance(d, dict):
                    parts.append(f"- **{d.get('direction', d.get('name', '?'))}**: {d.get('reason', '')[:150]}")
                elif hasattr(d, 'direction'):
                    parts.append(f"- **{d.direction}**: {getattr(d, 'reason', '')[:150]}")
            parts.append("")
        
        # v3.1: 软负收益警告方向
        warnings = ctx.get("warning_directions", [])
        if warnings:
            parts.append("## ⚠️ 软负收益警告 (倾向避开这些范式)")
            for w in warnings[:5]:
                if isinstance(w, dict):
                    parts.append(f"- {w.get('description', w.get('pattern_id', '?'))[:200]}")
                elif hasattr(w, 'description'):
                    parts.append(f"- {getattr(w, 'description', str(w))[:200]}")
            parts.append("")

        # v3.1: Motif 级别规则
        motif_rules = ctx.get("motif_rules", {})
        if motif_rules:
            forbid_motifs = motif_rules.get("forbid", [])
            prefer_motifs = motif_rules.get("prefer", [])
            if forbid_motifs:
                parts.append("## 🚫 Motif 禁止规则 (JQ验证失败率过高)")
                for r in forbid_motifs[:5]:
                    parts.append(f"- 结构 '{r.get('motif_key', '?')[:60]}': {r.get('reason', '')}")
                parts.append("")
            if prefer_motifs:
                parts.append("## ✅ Motif 推荐规则 (JQ验证成功率≥67%)")
                for r in prefer_motifs[:5]:
                    parts.append(f"- 结构 '{r.get('motif_key', '?')[:60]}': {r.get('reason', '')}")
                parts.append("")

        # FSA 禁止骨架
        fsa = ctx.get("fsa_forbidden", "")
        if fsa and fsa.strip():
            parts.append("## 🚫 FSA 禁止子树骨架 (避免使用这些结构)")
            parts.append(fsa[:1000])
            parts.append("")
        
        # 库状态
        lib = ctx.get("library_stats", {})
        if lib:
            parts.append(f"## 因子库现状")
            parts.append(f"- 因子总数: {lib.get('total_factors', '?')}")
            parts.append(f"- 因子族数: {lib.get('n_clusters', '?')}")
            red_sea = lib.get("red_sea", {})
            parts.append(f"- Red Sea等级: {red_sea.get('level', '?')} (相关性中位数={red_sea.get('median_correlation', '?')})")
            parts.append("")
        
        # 范式优先级
        priorities = ctx.get("paradigm_priorities", {})
        if priorities:
            parts.append("## 探索优先级")
            for p, pri in priorities.items():
                parts.append(f"- {p}: {pri}")
            parts.append("")
        
        # 近期表现
        recent = ctx.get("recent_performance", {})
        if recent:
            parts.append("## 近期表现反馈")
            for k, v in recent.items():
                parts.append(f"- {k}: {v}")
            parts.append("")
        
        # v0.5: RAG 知识注入 — 从本地知识库检索相关前沿研究
        rag = ctx.get("rag_knowledge", {})
        if rag and (rag.get("frontier_research") or rag.get("failure_patterns")):
            from rag_context import format_for_prompt
            rag_text = format_for_prompt(rag)
            if rag_text.strip():
                parts.append(rag_text)

        # L1 解读层: 上轮 JQ 归因避坑提示 (2026-08-14, 软消费)
        # 只影响"怎么生成", 不影响"选谁" — 不碰筛选毒药区
        try:
            from experience_memory import get_memory as _get_mem_l1
            _frag = _get_mem_l1().build_interpretation_prompt_fragment(
                paradigm=target_paradigm,
                limit=5,
            )
            if _frag.strip():
                parts.append(_frag)
        except Exception:
            pass  # 归因不可用时不影响生成

        # ── P-20260821-001: AQuA 密封研究契约 (sealed contract) ──
        # AQuA (arXiv 2608.12841) 实证: 递归自改进核心 = 固定约束声明 +
        # 持久化验证证据回灌。此处把求值环境硬约束 + 上轮失败原因显式注入 prompt。
        # 开关 _RESEARCH_CONTRACT_ENABLED 关闭即回退裸 prompt (rollback)。
        if _RESEARCH_CONTRACT_ENABLED:
            parts.append("## 📜 研究契约 (sealed contract — 固定约束, 违反必被拦截)")
            parts.append("1. 求值上下文仅五变量: close open high low volume (+ 可选资金流字段), 无 amount/无基本面")
            parts.append("2. 方向一律 long; 空头信号用表达式整体负号实现 (系统按负号取反)")
            parts.append("3. .rank(pct=True) 语义 = 横截面排名 (系统强制 axis=1), 勿依赖时间轴排名")
            parts.append("4. 窗口口径: " + (
                "周频数据对齐 — rolling(1)=1周, rolling(4)=1月, rolling(12)=1季度, rolling(20)=5个月; 回测频率周频(1周持有)"
                if weekly_mode else
                "周频对齐: rolling(5)≈一周, rolling(20)≈一月; 回测频率周频"))
            parts.append("5. 禁止 if-else/where 状态切换表达式 (regime 切换族多次 JQ 失败 -79.5%)")
            parts.append("")
            _fail_note = ctx.get("last_round_failures", "")
            if not _fail_note:
                _fail_note = _extract_recent_failures()
            if _fail_note.strip():
                parts.append("## 🔁 上轮失败原因回灌 (避免重复犯错)")
                parts.append(_fail_note)
                parts.append("")

        # ── P1-3 (v0.6 实验): 评价宪法约束注入 ──
        # 让生成端提前知晓最终评价纪律, 约束生成方向
        # (增量边际 > 绝对IC、行为冗余降级、预算意识、holdout 不透明)。
        # 开关 V06_EXPERIMENTAL["g_prompt_constitution"] 关闭即回退 v0.5.2 prompt (rollback)。
        if _load_v06_flag("g_prompt_constitution"):
            parts.append("## 📜 评价宪法约束 (决定因子生死 — 请按此纪律设计)")
            parts.append("1. 【增量边际】新因子以'加入组合后的净提升'为准: 高 IC 但组合增量≈0 的因子会被拒绝; 优先设计与现有因子机制正交的信号")
            parts.append("2. 【行为冗余】与库内因子行为高度相似的变体会被标记为替代品/近似重复并降级; 同一经济机制只保留一个, 变体须改变机制而非微调参数")
            parts.append("3. 【预算纪律】每个探索方向有尝试预算, 连续落空会被冻结冷却; 已被禁止的方向不要变相重试")
            parts.append("4. 【密封盲评】存在最终盲评验证区 (其划分对生成端不可见), 只有类别结论回传; 任何针对验证集调参的尝试都会被污染登记拦截")
            parts.append("5. 【可解释优先】假设必须说清经济机制, 纯统计拟合优先被淘汰")
            parts.append("")

        # 输出格式要求
        parts.append(f"## 输出格式")
        parts.append(f"请生成 {n_factors} 个新因子，每行一个，格式: `表达式 | 名称 | 假设 | 方向`")
        parts.append(f"其中方向为 + (做多) 或 - (做空)")
        parts.append("")
        parts.append("⚠️ 假设 (hypothesis) 必须与表达式严格对齐:")
        parts.append("- 假设必须具体描述表达式中实际使用的操作和字段")
        parts.append("  (如用到 rolling(5).mean 就写均线, 用到 diff/pct_change/shift 才写动量)")
        parts.append("- 禁止写表达式里没有的概念 (表达式不含 volume 时, 假设不得提放量/缩量)")
        parts.append("- 若表达式中使用了 buy_lg_vol/sell_lg_vol 等字段, 假设应提及大单/主力资金流")
        parts.append("")
        parts.append("⚠️ 表达式语法要求 (pandas Rolling 风格):")
        parts.append("- 必须使用 pandas Series.rolling(N).method() 语法, 不允许 forge ts_* 风格")
        parts.append("- 可用算子: .rolling(N).mean() .rolling(N).std() .rolling(N).max() .rolling(N).min() .shift(N) .diff(N) .pct_change(N) .rank(pct=True)")
        parts.append("- 可用函数: np.log np.abs np.sign np.where np.maximum np.minimum")
        parts.append("- 变量名: close open high low volume (均为 pandas Series)")
        parts.append("- 可选资金流字段: buy_lg_vol sell_lg_vol buy_sm_vol sell_sm_vol buy_md_vol sell_md_vol buy_elg_vol sell_elg_vol net_mf_vol")
        parts.append("- 示例正确: `(close - close.rolling(60).min()) / close.rolling(60).min()`")
        parts.append("- 示例正确: `-((high - low) / (volume.rolling(20).mean() + 1)).rank(pct=True)`")
        parts.append("- 示例错误: `ts_mean(close, 20)` `div(a,b)` `neg(x)` `rank(x)` (不允许!)")
        parts.append("")
        parts.append("因子设计原则:")
        parts.append("1. 避免使用已被FSA禁止的骨架结构")
        parts.append("2. 优先在目标范式内设计，也可以跨界创新")
        parts.append("3. 表达式复杂度适中(node_count ≤ 10, depth ≤ 4)")
        parts.append("4. 有清晰的经济逻辑支撑")
        parts.append("5. 考虑A股市场特征(散户为主、高换手、政策敏感)")
        parts.append("6. 如果能基于成功模板做改进/变体，优先该路径")
        parts.append("")
        parts.append("直接输出因子列表(不要额外解释):")
        
        return "\n".join(parts)
    
    # ── 写入 prompt 文件 ─────────────────────────────────
    
    def write_prompt_file(self, prompt: str = "", target_paradigm: str = "") -> Path:
        """将 prompt 写入文件供手动提交LLM"""
        if not prompt:
            prompt = self.build_prompt(target_paradigm)
        
        self._gen_counter += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_gen_prompt_{ts}_r{self._gen_counter:03d}.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        self._last_prompt_file = filepath
        print(f"  [LLMGen] Prompt → {filepath}")
        return filepath
    
    # ── 响应解析 ──────────────────────────────────────────
    
    def parse_response(self, response_text: str) -> List[Dict]:
        """
        解析 LLM 响应为候选因子列表。
        
        支持两种格式:
        1. 管道分隔: expression | name | hypothesis | direction
        2. JSON: [{"expression": ..., "name": ..., "hypothesis": ..., "direction": ...}]
        """
        candidates = []
        
        # 尝试 JSON 格式
        if response_text.strip().startswith("["):
            try:
                items = json.loads(response_text)
                for item in items:
                    cand = self._normalize_candidate(item)
                    if cand:
                        candidates.append(cand)
                if candidates:
                    return candidates
            except json.JSONDecodeError:
                pass
        
        # 管道分隔格式
        for line in response_text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            
            # 跳过 LLM 的"解释"行 (通常不是因子格式)
            if "|" not in line:
                continue
            
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            
            expression = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            hypothesis = parts[2] if len(parts) > 2 else ""
            direction = parts[3].strip() if len(parts) > 3 else "+"
            
            # 2026-09-01 修复: 剥掉 markdown 列表前导符 (LLM 常以 "- expr | name | ..."
            # 或 "1. expr | ..." 输出, 前导 "- " 会被误当成公式取负 → 方向语义翻转)
            _stripped = re.sub(r'^\s*[-*•]\s+', '', expression)
            _stripped = re.sub(r'^\s*\d+[.)、]\s*', '', _stripped)
            # 剥除后非空且仍像表达式 (不以裸运算符结尾/开头异常) 才采用
            if _stripped and len(_stripped) > 2 and _stripped != expression:
                expression = _stripped
            
            # 标准化方向
            if direction in ("-", "short", "做空"):
                direction = "short"
            else:
                direction = "long"
            
            # 2026-09-01 修复: 管道分支与 JSON 分支统一走 _normalize_candidate,
            # 保证候选 dict 含 factor_name/formula 键。
            # (此前管道产物只有 expression/name, 下游 jq_candidate_details 的
            #  cand_map 按 factor_name 匹配失败 → formula 空 → JQ codegen 全 SKIP)
            cand = self._normalize_candidate({
                "expression": expression,
                "name": name,
                "hypothesis": hypothesis,
                "direction": direction,
            })
            if cand:
                candidates.append(cand)
        
        return candidates
    
    def parse_response_from_file(self, filepath: str) -> List[Dict]:
        """从文件读取 LLM 响应并解析"""
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return self.parse_response(text)
    
    # ── 归一化 ───────────────────────────────────────────
    
    def _normalize_candidate(self, item: Dict) -> Optional[Dict]:
        """将各种格式的候选字典归一化为标准格式"""
        formula = item.get("expression") or item.get("formula") or item.get("expr", "")
        if not formula:
            return None
        
        name = item.get("name") or item.get("factor_name") or f"llm_gen_{hash(formula) % 10000}"
        hypothesis = item.get("hypothesis") or item.get("rationale") or item.get("logic") or ""
        direction = item.get("direction", "+")
        
        if direction in ("-", "short", "做空"):
            direction = "short"
        else:
            direction = "long"
        
        return {
            "factor_name": name,
            "formula": formula,
            "expression": formula,
            "hypothesis": hypothesis,
            "logic": hypothesis,  # LLM 生成的可能在同一个字段里
            "direction": direction,
            "source": "llm_generated",
            "paradigm": item.get("paradigm", ""),
            # P-20260830-001: 周频语境标签透传 (weekly → weekly lane 裁决)
            "natural_freq": item.get("natural_freq") or item.get("freq") or "",
        }
    
    # ── 统一入口 ─────────────────────────────────────────
    
    def generate(
        self,
        target_paradigm: str = "",
        n_factors: int = 5,
        mode: str = "file",  # "file" | "api" | None (仅解析已有response)
        response_text: str = "",
        weekly_mode: Optional[bool] = None,  # P-20260830-001
    ) -> Tuple[Path, List[Dict]]:
        """
        生成因子: 构建prompt → API调用或写入文件 → 解析响应。
        
        Args:
            target_paradigm: 目标探索范式
            n_factors: 期望生成的因子数
            mode: "file" 写入文件等待手动提交 | "api" 调用 DeepSeek API
            response_text: 如果已有LLM响应文本，直接解析
            weekly_mode: True 时周频语境 prompt + 候选打 natural_freq=weekly
        """
        if weekly_mode is None:
            weekly_mode = self._weekly_mode
        if response_text:
            candidates = self.parse_response(response_text)
            if weekly_mode:
                for c in candidates:
                    c["natural_freq"] = "weekly"
            return None, candidates
        
        prompt = self.build_prompt(target_paradigm, n_factors, weekly_mode=weekly_mode)
        filepath = self.write_prompt_file(prompt, target_paradigm)
        
        if mode == "file":
            return filepath, []
        
        if mode == "api":
            # v3.1: 直接调用 DeepSeek API
            try:
                from llm_client import get_llm_client
                client = get_llm_client()
                response_text = client.chat_with_system(
                    system_prompt=(
                        "你是 A 股量化因子研究员，专精于设计具有经济逻辑支撑的因子表达式。"
                        "使用 pandas 风格表达式（如 close.rolling(20).mean()）。"
                    ),
                    user_prompt=prompt,
                    temperature=0.8,
                    max_tokens=4096,
                )
                candidates = self.parse_response(response_text)
                if weekly_mode:
                    for c in candidates:
                        c["natural_freq"] = "weekly"
                print(f"  [LLMGen] API 返回 {len(response_text)} chars → 解析出 {len(candidates)} 个候选"
                      + (" (周频语境)" if weekly_mode else ""))
                return filepath, candidates
            except Exception as e:
                print(f"  [LLMGen] API 调用失败: {e}，降级为文件模式")
                return filepath, []
        
        return filepath, []  # 默认: 生成文件, 手动解析

    # ── v0.6 P-021: 多范式批量生成 ────────────────────────

    def generate_paradigm_batch(
        self,
        paradigms: List[str],
        n_per_paradigm: int = 3,
        mode: str = "file",  # "file" | "api"
    ) -> Dict[str, Tuple[Path, List[Dict]]]:
        """
        P-021: 对多个未覆盖范式并行生成因子公式。

        为每个目标范式构建领域专用 prompt (包含经济逻辑+典型表达式)，
        调用 LLM 生成因子公式，返回按范式分组的候选列表。

        Parameters
        ----------
        paradigms: 目标范式名列表, 如 ["行业轮动", "情绪×日内"]
        n_per_paradigm: 每个范式生成因子数
        mode: "file" 写入文件 | "api" 直接调用 DeepSeek API

        Returns
        -------
        {paradigm_name: (prompt_file_path, [candidate_dict, ...]), ...}
        """
        results = {}
        for paradigm in paradigms:
            # 构建领域专用 prompt
            prompt = self._build_domain_prompt(paradigm, n_per_paradigm)
            filepath = self.write_prompt_file(prompt, paradigm)

            candidates = []
            if mode == "api":
                try:
                    from llm_client import get_llm_client
                    client = get_llm_client()
                    response_text = client.chat_with_system(
                        system_prompt=(
                            "你是 A 股量化因子研究员，专精于设计具有经济逻辑支撑的因子表达式。"
                            "使用 pandas 风格表达式（如 close.rolling(20).mean()）。"
                        ),
                        user_prompt=prompt,
                        temperature=0.8,
                        max_tokens=4096,
                    )
                    candidates = self.parse_response(response_text)
                    # 标注 paradigm
                    for c in candidates:
                        c["paradigm"] = paradigm
                    print(f"  [LLMGen] {paradigm}: {len(candidates)} 个候选")
                except Exception as e:
                    print(f"  [LLMGen] {paradigm} API 调用失败: {e}")

            results[paradigm] = (filepath, candidates)

        return results

    def _build_domain_prompt(self, paradigm: str, n_factors: int = 3) -> str:
        """
        构建领域专用 prompt，为每个未覆盖范式注入经济逻辑和参考表达式风格。

        基于 paradigm_v4.py 的 PARADIGMS_V4 定义，自动填充:
        - 范式经济逻辑
        - A股相关性说明
        - 典型表达式风格 (来自 CHAMPION_FACTORS 同范式模板)
        """
        from paradigm_v4 import PARADIGMS_V4

        paradigm_info = PARADIGMS_V4.get(paradigm, {})
        economic_logic = paradigm_info.get("economic_logic", "")
        a_share_relevance = paradigm_info.get("a_share_relevance", "")
        description = paradigm_info.get("description", "")

        # 从 seed_injector CHAMPION_FACTORS 中找同范式示例
        example_formulas = self._get_example_formulas(paradigm)

        ctx = self._context
        parts = []

        parts.append(f"# A股量化因子生成: {paradigm} 范式探索")
        parts.append("")
        parts.append(f"## 范式背景")
        parts.append(f"- **经济逻辑**: {economic_logic}")
        parts.append(f"- **A股相关性**: {a_share_relevance}")
        parts.append(f"- **描述**: {description}")
        parts.append("")

        if example_formulas:
            parts.append(f"## 参考表达式风格 (来自已验证的同范式因子)")
            for ex in example_formulas[:3]:
                parts.append(f"- `{ex['expression'][:120]}` — {ex['rationale'][:80]}")
            parts.append("")

        # 成功模板
        templates = ctx.get("success_templates", [])
        paradigm_templates = [t for t in templates
                            if (isinstance(t, dict) and paradigm in t.get("paradigm", "")) or
                               (hasattr(t, 'pattern_id') and paradigm in str(getattr(t, 'pattern_id', '')))]
        if paradigm_templates:
            parts.append(f"## ✅ 历史成功模板 ({paradigm} 领域)")
            for t in paradigm_templates[:3]:
                if isinstance(t, dict):
                    parts.append(f"- {t.get('description', t.get('pattern_id', ''))[:100]}: `{t.get('formula', '')[:120]}`")
            parts.append("")

        # 禁止方向
        forbidden = ctx.get("forbidden_directions", [])
        if forbidden:
            parts.append("## 🚫 禁止方向")
            for d in forbidden[:3]:
                parts.append(f"- {d.get('description', d.get('direction', ''))[:120]}")
            parts.append("")

        # 输出格式
        parts.append(f"## 输出格式")
        parts.append(f"请为 `{paradigm}` 范式生成 {n_factors} 个新因子，每行: `表达式 | 名称 | 假设 | 方向`")
        parts.append("方向为 + (做多) 或 - (做空)")
        parts.append("")
        parts.append("⚠️ 表达式要求: pandas Rolling 风格 (close.rolling(20).mean()), 不允许 forge ts_* 风格")
        parts.append("表达式复杂度适中 (node_count ≤ 10, depth ≤ 4)")
        parts.append("")
        parts.append("直接输出因子列表:")

        return "\n".join(parts)

    @staticmethod
    def _get_example_formulas(paradigm: str) -> List[Dict]:
        """从 CHAMPION_FACTORS 提取同范式示例公式"""
        try:
            from seed_injector import CHAMPION_FACTORS
        except ImportError:
            return []
        examples = []
        for f in CHAMPION_FACTORS:
            if f.get("paradigm", "") == paradigm:
                examples.append({
                    "expression": f.get("expression", f.get("formula", "")),
                    "rationale": f.get("rationale", ""),
                })
            if len(examples) >= 3:
                break
        return examples


def generate_llm_prompt_file(
    memory_priors: Dict,
    target_paradigm: str = "",
    n_factors: int = 5,
) -> Path:
    """
    快捷函数: 从 Memory priors 生成 LLM prompt 文件。
    
    Args:
        memory_priors: memory.retrieve() 的返回值
        target_paradigm: 目标范式
        n_factors: 期望生��因子数
        
    Returns:
        写入的 prompt 文件路径
    """
    gen = LLMGenerator()
    gen.receive_context(
        success_templates=memory_priors.get("success_templates", []),
        forbidden_directions=memory_priors.get("forbidden_directions", []),
        fsa_forbidden=memory_priors.get("fsa_forbidden_context", ""),
        library_stats=memory_priors.get("library_stats", {}),
    )
    return gen.write_prompt_file(target_paradigm=target_paradigm)


if __name__ == "__main__":
    # 自测: 构建一个示例 prompt
    gen = LLMGenerator()
    gen.receive_context(
        success_templates=[
            {"pattern_id": "筹码分布::[缩量,均值]", "description": "筹码锁定度因子",
             "formula": "neg(rank(div(ts_mean(volume, 10), ts_mean(volume, 60))))",
             "paradigm": "筹码分布", "occurrence_count": 6, "jq_return": 182.57},
        ],
        forbidden_directions=[
            {"direction": "ralph_no_ic_candidates", "reason": "无IC数据的因子不能直接上JQ"},
        ],
        library_stats={"total_factors": 243, "n_clusters": 156,
                       "red_sea": {"level": "green", "median_correlation": 0.038}},
    )
    
    filepath = gen.write_prompt_file(target_paradigm="筹码分布")
    print(f"Prompt written to: {filepath}")
    
    # 测试解析
    test_response = """
rank(div(sub(high, low), ts_mean(volume, 20))) | 振幅量比 | 高振幅低量=流动性溢价 | +
neg(ts_zscore(close, 60)) | 深度超卖 | 长期超卖=均值回归 | +
"""
    candidates = gen.parse_response(test_response)
    print(f"Parsed {len(candidates)} candidates:")
    for c in candidates:
        print(f"  - {c['expression'][:50]} | {c['name']} | {c['direction']}")
