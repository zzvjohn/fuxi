# -*- coding: utf-8 -*-
"""
W3 (2026-09-12): forbidden_directions 公式级硬屏蔽门 — 代码级、S1 前置。

背景 (审计结论):
  - `experience_memory.forbidden_directions` 现有 51 条, 全部 `severity=hard`,
    50 条带 `prototype_expression`, 0 条写 cooldown。
  - 但 S1–S5 全链路**不消费**它: `severity=='hard'` 的代码级硬拒只出现在
    `library_orthogonality._check_forbidden_regions` (仅**入库**时), 生成端与
    验证端完全看不见 ⇒ 已被 JQ 证伪的公式原型会反复被重新生成、反复烧回测额度。
  - FSA (`subtree_fingerprinter`) 的 `forbidden_skeletons` 走的是"库内骨架频次
    超阈值"路径。M9 (2026-09-26) 复测: share 判据 (15% × n_factors) 在现行
    1122 因子库下仍不可达 (max count=5 < 阈值 168)。库级冻结恒 0 的事实已由
    `get_context_status()` 显式暴露 (reachable=False), 不再是静默 0; dominance
    判据 (env FUXI_FSA_MODE=dominance) 实测可产出 15 条冻结, 但**默认未切换**
    (未走「翻默认四关」)。生产链中真实消费冻结的判据是 Forge 运行内 FSA
    (forge/__init__.py: 种群集中度 ≥15% 或 ≥3 次, 冻结骨架直接 -999 淘汰)。
    三者**互补**关系:
      FSA(库级)   = 库内自身同质化(内生, 频次驱动, 当前观察态)
      FSA(运行内) = 种群级集中度(内生, 实时驱动, 已消费)
      本门         = 已证伪方向的黑名单(外生, 失败驱动)

设计原则 (审计要求):
  1. AST 规范化后比较, 不做字符串匹配 ⇒ 复用 `SubtreeFingerprinter.parse_expression`
     (字段→FIELD, 窗口→N1/N2..., 算子别名归一)。
  2. 双重判据: (a) 骨架**精确相等**; (b) 骨架 token 集合 **Jaccard ≥ 0.95**
     (近似等价但写法不同, 如 `ts_mean(close,20)-ts_mean(close,60)` vs
     `sub(ma(FIELD,N1),ma(FIELD,N2))` 的变体)。
  3. **默认 enforce** (2026-09-27 M10 转正; 回退: `FUXI_FORBIDDEN_GATE=shadow`,
     零行为变化)。转正前置 (2026-09-12 版设计要求): 只读冲突扫描的 4 条库内
     冲突须逐个裁决 —— 已于 2026-09-27 (M10) 完成:
       - `max_drawdown_duration` / `chip_cost_zscore_volume_weighted` /
         `vwap_dev_ratio_5_60`: JQ 方向证伪明确 (-39.9% / -45.5% / -46.9%,
         ±MDD 53~62%), 库内残留属"证伪遗留" ⇒ **保留拦截, 不豁免**;
       - `dollar_vol_stability`: 命中条目 `gp_breed_008` 的根因自述为
         "因子alpha有效但执行端崩溃: 选出低价小市值标的, 交易量不足导致
         '开仓数量<100'; 需加市值[过滤]" ⇒ **执行层教训, 不构成公式方向
         证伪**; 且库内 icir=0.862 (池内第二, 与 FA 核心 dollar_vol_20d
         同族) ⇒ **进入豁免白名单 (标而不拒)**。
  4. **豁免白名单** (2026-09-27 M10): env `FUXI_FORBIDDEN_EXEMPT` 指向豁免
     清单 JSON (默认 `data/forbidden_gate_exempt.json`; 设 `off` 关闭豁免)。
     命中豁免的候选返回 `blocked=True, exempted=True, enforced=False`
     —— **仍打标供观测, 但不硬拒** (豁免不隐藏任何信息)。
  5. 开关: env `FUXI_FORBIDDEN_GATE = off | shadow | enforce(默认, 硬拒)`。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MEMORY_PATH = DATA_DIR / "experience_memory.json"
EXEMPT_ENV = "FUXI_FORBIDDEN_EXEMPT"                       # M10 (2026-09-27)
DEFAULT_EXEMPT_PATH = DATA_DIR / "forbidden_gate_exempt.json"

_TOK_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SKEL_NAME_RE = re.compile(r"(FIELD|N\d+)")


def _resolve_mode(mode: Optional[str] = None) -> str:
    if mode is None:
        # M10 (2026-09-27) 转正: 默认 enforce —— 回退显式设 shadow (零行为变化)。
        mode = os.environ.get("FUXI_FORBIDDEN_GATE", "enforce")
    mode = str(mode).strip().lower()
    return mode if mode in ("off", "shadow", "enforce") else "enforce"


def _resolve_exempt_path(exempt_path: Optional[Path] = None) -> Optional[Path]:
    """豁免清单路径解析 (M10): 参数 > env > 默认; env=off/none/0 -> None (关闭豁免)。"""
    if exempt_path is not None:
        return Path(exempt_path)
    env = (os.environ.get(EXEMPT_ENV) or "").strip()
    if env.lower() in ("off", "none", "0"):
        return None
    if env:
        return Path(env)
    return DEFAULT_EXEMPT_PATH


def _skeleton_tokens(skeleton: str) -> Set[str]:
    """骨架 → token 集合 (算子名 + FIELD/N占位符), 仅用于诊断展示。"""
    toks = set(_SKEL_NAME_RE.findall(skeleton))
    toks |= {t for t in _TOK_RE.findall(skeleton) if t not in ("FIELD",)}
    toks |= set(re.findall(r"N\d+", skeleton))
    return toks


# 冗余括号: `(FIELD)` / `(N1)` / `(-x)` → 去掉单 token 括号 (不改结构语义)
_REDUNDANT_PAREN_RE = re.compile(r"\((FIELD|N\d+)\)")


def _canon_skeleton(skeleton: str) -> str:
    """骨架二次规范化 — 只为消除**写法差异**造成的假区分。

    审计教训: 库内 `-(open_p)` 与黑名单 `-open_p` 骨架字符串不同但语义相同;
    反之 token-**集合** Jaccard 会把 `A-B` 与 `B-A` 判为同一 (结构顺序丢失)。
    故此处只做"去冗余括号 + 压缩空白", 保留算子顺序, 供序列相似度比较。
    """
    s = str(skeleton or "").strip()
    prev = None
    while prev != s:                      # 迭代到不动点, 处理嵌套单 token 括号
        prev = s
        s = _REDUNDANT_PAREN_RE.sub(r"\1", s)
    s = re.sub(r"\s+", "", s)
    return s


def _seq_similarity(a: str, b: str) -> float:
    """顺序敏感的骨架相似度 (difflib 序列比) — 不会混淆 A-B 与 B-A。"""
    import difflib
    ca, cb = _canon_skeleton(a), _canon_skeleton(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    return float(difflib.SequenceMatcher(None, ca, cb).ratio())


class ForbiddenGate:
    """已证伪方向的公式级硬屏蔽门 (S1 前置, 默认 off)。"""

    def __init__(
        self,
        memory_path: Path = MEMORY_PATH,
        mode: Optional[str] = None,
        jaccard_threshold: float = 0.95,
        min_failed_attempts: int = 1,
        exempt_path: Optional[Path] = None,
    ):
        self.memory_path = Path(memory_path)
        self.mode = _resolve_mode(mode)
        # 顺序敏感相似度阈值 (difflib 序列比, 非 token 集合 Jaccard —
        # 审计教训: 集合 Jaccard 会把 A-B / B-A 判同, 故改序列比)
        self.jaccard_threshold = float(jaccard_threshold)
        self.min_failed_attempts = int(min_failed_attempts)
        # skeleton -> {"direction_id", "desc", "failed_attempts", "expr"}
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.entry_tokens: Dict[str, Set[str]] = {}
        self.entry_canon: Dict[str, str] = {}
        self.loaded = False
        self.load_error = ""
        # M10 (2026-09-27): 豁免白名单 (canon_skel -> 条目 dict)
        self.exempt_path = _resolve_exempt_path(exempt_path)
        self.exempt_entries: Dict[str, Dict[str, Any]] = {}
        self.exempt_loaded = False
        self.exempt_error = ""
        self._parse = None
        self._stats = {"checked": 0, "blocked": 0, "exact": 0, "jaccard": 0, "exempted": 0}

    # ── 加载 ────────────────────────────────────────────────
    def _get_parser(self):
        if self._parse is None:
            try:
                from subtree_fingerprinter import SubtreeFingerprinter
                self._parse = SubtreeFingerprinter().parse_expression
            except Exception:
                # 降级: 极简规范化 (字段→FIELD, 数字→N), 保证不因依赖缺失而失效
                def _fallback(expr: str) -> str:
                    s = re.sub(r"\b\d+\b", "N", str(expr))
                    s = re.sub(r"\s+", "", s)
                    return s
                self._parse = _fallback
        return self._parse

    def load(self) -> int:
        """读取 forbidden_directions, 构建骨架黑名单。返回条目数。

        M10 (2026-09-27): 同步加载豁免清单, 使 load() 后门状态完整 —
        scan_library.exempt_size / _stats 不再依赖 check() 被动懒加载。
        """
        if not self.exempt_loaded:
            self.load_exempt()
        if not self.memory_path.exists():
            self.load_error = f"memory 文件不存在: {self.memory_path}"
            return 0
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.load_error = f"读取失败: {e}"
            return 0

        parse = self._get_parser()
        dirs = data.get("forbidden_directions", []) or []
        n = 0
        for d in dirs:
            if not isinstance(d, dict):
                continue
            if str(d.get("severity", "")).lower() != "hard":
                continue
            if int(d.get("failed_attempts", 0) or 0) < self.min_failed_attempts:
                continue
            proto = (d.get("prototype_expression") or "").strip()
            if not proto or proto.lower() == "none":
                continue
            try:
                skel = parse(proto)
            except Exception:
                continue
            if not skel:
                continue
            self.entries[skel] = {
                "direction_id": d.get("direction_id", "?"),
                "desc": (d.get("description") or "")[:120],
                "failed_attempts": int(d.get("failed_attempts", 0) or 0),
                "expr": proto,
                "jq_return": d.get("jq_return"),
                "jq_sharpe": d.get("jq_sharpe"),
            }
            self.entry_tokens[skel] = _skeleton_tokens(skel)
            self.entry_canon[skel] = _canon_skeleton(skel)
            n += 1
        self.loaded = True
        return n

    # ── 豁免白名单加载 (M10, 2026-09-27) ────────────────────
    def load_exempt(self) -> int:
        """读豁免清单: {'schema','entries':[{'skeleton','factor_name','reason',...}]}。

        返回豁免条目数。文件不存在 / 路径为 None / 读坏 -> 0 (无豁免, 不阻塞)。
        """
        p = self.exempt_path
        self.exempt_loaded = True
        if p is None or not Path(p).exists():
            return 0
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as e:
            self.exempt_error = f"豁免清单读取失败: {e}"
            return 0
        n = 0
        for e in (data.get("entries", []) or []):
            if not isinstance(e, dict):
                continue
            skel = str(e.get("skeleton") or "").strip()
            if not skel:
                continue
            self.exempt_entries[_canon_skeleton(skel)] = e
            n += 1
        return n

    def _apply_exempt(self, out: Dict[str, Any], cand_canon: str) -> None:
        """命中黑名单后查豁免 —— 豁免时 enforced 强制 False (标而不拒)。

        匹配: 豁免骨架规范化后与候选骨架序列相似度 >= 同阈值 (与 check 一致)。
        豁免不隐藏信息: blocked/matched/direction_id/similarity 全部保留。
        """
        if not self.exempt_loaded:
            self.load_exempt()
        if not self.exempt_entries or not cand_canon:
            return
        best, best_e = 0.0, None
        for csk, e in self.exempt_entries.items():
            s = 1.0 if csk == cand_canon else _seq_similarity(cand_canon, csk)
            if s > best:
                best, best_e = s, e
        if best_e is not None and best >= self.jaccard_threshold:
            out.update(exempted=True, enforced=False,
                       exempt_factor=str(best_e.get("factor_name") or ""),
                       exempt_similarity=round(best, 4),
                       exempt_reason=str(best_e.get("reason") or ""),
                       reason=(str(out.get("reason") or "") +
                               f" [豁免: {best_e.get('factor_name', '?')} 白名单, 不硬拒]"))
            self._stats["exempted"] += 1

    # ── 检查 ────────────────────────────────────────────────
    def check(self, expression: str) -> Dict[str, Any]:
        """检查单个表达式是否命中硬屏蔽名单。

        **总是计算** (不因 mode=off 而短路), 返回 `enforced` 供调用方决定是否硬拒。
        好处: 只读冲突扫描 (scan_library) 与影子归因都无需临时改 mode。
        M10 (2026-09-27): 命中豁免白名单时 `exempted=True, enforced=False` (标而不拒)。
        """
        out = {"blocked": False, "enforced": self.mode == "enforce", "exempted": False,
               "matched": "", "direction_id": "", "similarity": 0.0,
               "mode": self.mode, "reason": ""}
        if not expression:
            return out
        if not self.loaded:
            self.load()
        if not self.entries:
            return out

        try:
            skel = self._get_parser()(expression)
        except Exception:
            return out
        if not skel:
            return out

        self._stats["checked"] += 1
        cand_canon = _canon_skeleton(skel)

        # (a) 骨架精确相等 (含去冗余括号后相等)
        if skel in self.entries or cand_canon in self.entry_canon.values():
            hit_skel = skel if skel in self.entries else next(
                (k for k, v in self.entry_canon.items() if v == cand_canon), skel)
            e = self.entries.get(hit_skel)
            if e is None:
                return out
            out.update(blocked=True, matched=hit_skel, direction_id=e["direction_id"],
                       similarity=1.0, exact=True,
                       reason=(f"命中硬禁止方向 {e['direction_id']} "
                               f"(JQ {e.get('jq_return')}%/{e.get('jq_sharpe')}), "
                               f"失败 {e['failed_attempts']} 次"))
            self._stats["blocked"] += 1
            self._stats["exact"] += 1
            self._apply_exempt(out, cand_canon)     # M10: 豁免 (标而不拒)
            return out

        # (b) 顺序敏感序列相似度 (不会把 A-B 与 B-A 判同)
        best_j, best_skel = 0.0, ""
        for fskel in self.entries:
            j = _seq_similarity(cand_canon, self.entry_canon[fskel])
            if j > best_j:
                best_j, best_skel = j, fskel
        if best_j >= self.jaccard_threshold and best_skel:
            e = self.entries[best_skel]
            out.update(blocked=True, matched=best_skel, direction_id=e["direction_id"],
                       similarity=round(best_j, 4), exact=False,
                       reason=(f"结构近似硬禁止方向 {e['direction_id']} "
                               f"(序列相似度 {best_j:.3f}≥{self.jaccard_threshold})"))
            self._stats["blocked"] += 1
            self._stats["jaccard"] += 1
            self._apply_exempt(out, cand_canon)     # M10: 豁免 (标而不拒)
            return out
        return out

    # ── 只读冲突扫描 (开启前必跑) ───────────────────────────
    def scan_library(self, factors: List[Dict]) -> Dict[str, Any]:
        """扫描现有因子库, 统计会被本门拦下的因子 (只读, 不改任何状态)。"""
        if not self.loaded:
            self.load()
        hits: List[Dict[str, Any]] = []
        for f in factors:
            expr = f.get("formula", f.get("expression", ""))
            name = f.get("factor_name", f.get("name", "?"))
            if not expr:
                continue
            r = self.check(expr)
            if r["blocked"]:
                hits.append({"factor_name": name, "formula": expr[:100],
                             "matched": r["matched"], "direction_id": r["direction_id"],
                             "similarity": r["similarity"], "exact": r.get("exact", False),
                             "exempted": bool(r.get("exempted"))})
        n_exempt = sum(1 for h in hits if h.get("exempted"))
        return {
            "n_scanned": len(factors),
            "n_conflict": len(hits),
            "n_exempted": n_exempt,
            "n_blocking": len(hits) - n_exempt,
            "conflicts": hits[:50],
            "blacklist_size": len(self.entries),
            "exempt_size": len(self.exempt_entries),
            "similarity_threshold": self.jaccard_threshold,
        }

    # ── 生成端注入文本 ──────────────────────────────────────
    def get_for_generation_text(self, limit: int = 8) -> str:
        if self.mode == "off" or not self.entries:
            return ""
        lines = ["## 🚫 已证伪方向黑名单 (公式级硬屏蔽, 请勿生成同构公式)："]
        for i, (skel, e) in enumerate(list(self.entries.items())[:limit], 1):
            lines.append(f"{i}. `{skel}` — {e['direction_id']} "
                         f"(JQ {e.get('jq_return')}%/{e.get('jq_sharpe')}, "
                         f"失败{e['failed_attempts']}次)")
        if len(self.entries) > limit:
            lines.append(f"... 另有 {len(self.entries) - limit} 条")
        return "\n".join(lines)

    def get_status(self) -> Dict[str, Any]:
        return {"mode": self.mode, "loaded": self.loaded, "blacklist_size": len(self.entries),
                "similarity_threshold": self.jaccard_threshold, "stats": dict(self._stats),
                "exempt_size": len(self.exempt_entries),
                "exempt_path": str(self.exempt_path) if self.exempt_path else None,
                "exempt_error": self.exempt_error, "error": self.load_error}


_GATE: Optional[ForbiddenGate] = None


def get_forbidden_gate() -> ForbiddenGate:
    global _GATE
    if _GATE is None:
        _GATE = ForbiddenGate()
        _GATE.load()
    return _GATE
