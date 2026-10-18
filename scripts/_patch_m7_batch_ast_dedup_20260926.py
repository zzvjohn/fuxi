# -*- coding: utf-8 -*-
"""M7 (2026-09-26): 批内 AST 去重接入 Ralph 主线 — 字节级补丁。

背景 (reports/fuxi_alpha_generation_overhaul_20260926.md M7 / 证据 E1):
  `_batch_dedup_ast` 自 P-20260910-005 起只存在于 generate() 的两条内路径
  (llm_generator.py:1086/1115)。Ralph 生产主线逐轮直调 parse_response
  (首轮 + reviewer 二轮), 从未经过批内去重 ⇒ 同批骨架完全相同的候选
  照常进入 S1-S5 ⇒ 评估预算重复消耗。

改动 (两文件, 7 处):
  llm_generator.py
    [A1] 新增模块级公共阶段 batch_dedup_mode / dedup_ast_candidates
         (插在 class LLMGenerator 之前)
    [A2] _batch_dedup_ast 改为委托公共阶段; 新增公共方法 batch_dedup_ast
  ralph_loop.py
    [B1] 生成阶段公共初始化新增 M7 留档变量
    [B0] llm 分支入口 fail-fast 校验开关 (置于 API try 之外,
         防止配置错误落入 except 被误报为 api_error 的静默降级)
    [B2] 首轮 parse → 批内去重 → tag_semantic (顺序与 generate() 一致)
    [B3] reviewer 二轮 parse → 批内去重 → tag_semantic
    [B5] generate 结果字典追加 "batch_ast_dedup" 留档键

范围纪律: 仅批内去重。reviewer 二轮本质是"同骨架定向改进"(提示词要求保留
  最高 ICIR 公式的骨架), 因此**不做跨轮合并去重** — 否则二轮产出会被首轮
  骨架直接杀掉。

开关: FUXI_BATCH_AST_DEDUP = off | shadow | enforce (默认 enforce, 与
  generate() 内历史无条件去重行为一致)。shadow 只留档不移除, 供误杀率 A/B。

文件纪律: 两目标文件均为纯 CRLF (bare LF=0), 字节级替换; 每处要求唯一命中;
  先备份 .bak_m7_20260926; 默认干跑; 写后校验 bare LF == 0 且 LF == CRLF;
  py_compile 两文件。

用法:
  python scripts/_patch_m7_batch_ast_dedup_20260926.py          # 干跑
  python scripts/_patch_m7_batch_ast_dedup_20260926.py --apply  # 应用
"""
import argparse
import hashlib
import py_compile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAK_SUFFIX = ".bak_m7_20260926"


def crlf(text: str) -> str:
    """把 LF 文本规范为 CRLF (目标文件均为纯 CRLF)。"""
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


# ═══════════════════════════════════════════════════════════════
# A1: 模块级公共阶段 (插在 class LLMGenerator 之前)
# ═══════════════════════════════════════════════════════════════
A1_OLD = """class LLMGenerator:
"""

A1_NEW = """# ═══════════════════════════════════════════════════════════════
# M7 (2026-09-26): 批内 AST 骨架去重 — 提升为公共阶段
# ═══════════════════════════════════════════════════════════════
# 病灶 (E1): `_batch_dedup_ast` 自 P-20260910-005 起只出现在 generate() 的
#   两条内路径 (L1086/L1115)。Ralph 生产主线逐轮直调 parse_response
#   (ralph_loop.py 首轮 + reviewer 二轮), 从未经过批内去重 ⇒ 同批骨架
#   完全相同的候选照常进入 S1-S5 ⇒ 评估预算重复消耗。
# 语义: 骨架指纹 = subtree_fingerprinter.parse_expression (字段→FIELD, 窗口→
#   N1/N2 按首现顺序编号, 算子别名归一); 结构重复保留先出现者 (保多样性)。
# 开关: FUXI_BATCH_AST_DEDUP = off | shadow | enforce (默认 enforce, 与
#   generate() 内历史无条件去重行为一致)。shadow 只留档不移除, 供误杀率 A/B。
# 留档: 命中的候选全部写入 evidence_out (含指纹/首现者/阶段), 不静默丢弃。
# 范围: 仅批内。reviewer 二轮本质是"同骨架定向改进", 故意不做跨轮去重。

def batch_dedup_mode() -> str:
    \"\"\"解析 FUXI_BATCH_AST_DEDUP (off|shadow|enforce, 默认 enforce)。

    非法值直接抛 ValueError (fail-fast; 调用方须放在 API 降级 try 之外,
    否则配置错误会被 except 吞掉误报为 api_error)。
    \"\"\"
    mode = str(os.environ.get("FUXI_BATCH_AST_DEDUP", "enforce") or "enforce").strip().lower()
    if mode not in ("off", "shadow", "enforce"):
        raise ValueError("FUXI_BATCH_AST_DEDUP must be off, shadow or enforce")
    return mode


def dedup_ast_candidates(
    candidates: List[Dict],
    evidence_out: Optional[List[Dict]] = None,
    mode: Optional[str] = None,
    stage: str = "",
) -> List[Dict]:
    \"\"\"批内 AST 骨架指纹去重 (结构重复保留先出现者, 保多样性)。

    mode=None ⇒ 读取 FUXI_BATCH_AST_DEDUP。
    enforce: 移除重复项; shadow: 只记录不移除; off: 原样返回。
    evidence_out: 每条命中候选追加留档记录 (去重但不删除证据)。
    \"\"\"
    if mode is None:
        mode = batch_dedup_mode()
    if mode == "off" or len(candidates) < 2:
        return candidates
    try:
        from subtree_fingerprinter import get_fsa
        fsa = get_fsa()
        seen: Dict[str, str] = {}
        out, n_dup = [], 0
        for idx, c in enumerate(candidates):
            expr = c.get("formula", c.get("expression", ""))
            if not expr:
                out.append(c)
                continue
            try:
                fp = fsa.fingerprint(expr)
            except Exception:
                fp = None
            if fp and fp in seen:
                n_dup += 1
                if evidence_out is not None:
                    evidence_out.append({
                        "stage": str(stage or ""),
                        "factor_name": str(c.get("factor_name", c.get("name", "?")))[:80],
                        "formula": str(expr)[:200],
                        "fingerprint": str(fp)[:200],
                        "dup_of": str(seen[fp])[:80],
                        "index": idx,
                        "round2": bool(c.get("_llm_reviewer_round2", False)),
                        "removed": bool(mode == "enforce"),
                    })
                if mode == "enforce":
                    continue
            if fp and fp not in seen:
                seen[fp] = str(c.get("factor_name", c.get("name", "?")))[:80]
            out.append(c)
        if n_dup:
            _suffix = f" (stage={stage})" if stage else ""
            if mode == "enforce":
                print(f"  [LLMGen] 批内 AST 去重: 移除 {n_dup} 条结构重复候选{_suffix}")
            else:
                print(f"  [LLMGen] 批内 AST 去重 (shadow): 命中 {n_dup} 条结构重复候选,"
                      f" 未移除{_suffix}")
        return out
    except Exception as e:
        print(f"  [LLMGen] 批内去重失败 (不阻塞): {e}")
        return candidates


class LLMGenerator:
"""

# ═══════════════════════════════════════════════════════════════
# A2: _batch_dedup_ast 委托公共阶段 + 新增公共方法
# ═══════════════════════════════════════════════════════════════
A2_OLD = """    def _batch_dedup_ast(self, candidates: List[Dict]) -> List[Dict]:
        \"\"\"P-20260910-005: 批内 AST 骨架指纹去重 (结构重复保留先出现者, 保多样性)。\"\"\"
        if len(candidates) < 2:
            return candidates
        try:
            from subtree_fingerprinter import get_fsa
            fsa = get_fsa()
            seen, out, n_dup = set(), [], 0
            for c in candidates:
                expr = c.get("formula", c.get("expression", ""))
                if not expr:
                    out.append(c)
                    continue
                try:
                    fp = fsa.fingerprint(expr)
                except Exception:
                    fp = None
                if fp and fp in seen:
                    n_dup += 1
                    continue
                if fp:
                    seen.add(fp)
                out.append(c)
            if n_dup:
                print(f"  [LLMGen] 批内 AST 去重: 移除 {n_dup} 条结构重复候选")
            return out
        except Exception as e:
            print(f"  [LLMGen] 批内去重失败 (不阻塞): {e}")
            return candidates
"""

A2_NEW = """    def batch_dedup_ast(
        self,
        candidates: List[Dict],
        evidence: Optional[List[Dict]] = None,
        mode: Optional[str] = None,
        stage: str = "",
    ) -> List[Dict]:
        \"\"\"M7: 公共批内 AST 去重入口 (Ralph 主线 / reviewer 二轮 / generate())。

        受 FUXI_BATCH_AST_DEDUP 管辖; evidence 非 None 时命中项追加留档。
        \"\"\"
        return dedup_ast_candidates(candidates, evidence_out=evidence,
                                    mode=mode, stage=stage)

    def _batch_dedup_ast(self, candidates: List[Dict]) -> List[Dict]:
        \"\"\"P-20260910-005 历史入口: 委托公共阶段 (M7 起统一受开关管辖)。\"\"\"
        return dedup_ast_candidates(candidates)
"""

# ═══════════════════════════════════════════════════════════════
# B1: 生成阶段公共初始化 — M7 留档变量
# ═══════════════════════════════════════════════════════════════
B1_OLD = """        _llm_api_ok = None
        _llm_degrade_reason = ""
        _semantic_selection = None
"""

B1_NEW = """        _llm_api_ok = None
        _llm_degrade_reason = ""
        _semantic_selection = None
        # M7: 批内 AST 去重留档 (Ralph 首轮 + reviewer 二轮; 随 return 落盘)。
        # evidence 记录命中项的指纹/首现者/阶段, 误杀可回放 (去重不删证据)。
        _m7_dedup_evidence = []
        _m7_dedup_mode = ""
"""

# ═══════════════════════════════════════════════════════════════
# B0: llm 分支入口 fail-fast 校验 (置于 API try 之外)
# ═══════════════════════════════════════════════════════════════
B0_OLD = """        elif generator == "llm":
            # v3.1: LLM 因子生成 (双引擎) — 直接调用 DeepSeek API
            self.llm_gen.receive_context(
"""

B0_NEW = """        elif generator == "llm":
            # v3.1: LLM 因子生成 (双引擎) — 直接调用 DeepSeek API
            # M7: 批内 AST 去重开关 fail-fast 校验。放在 API try 之外 ——
            #   配置错误不得落入 except 降级被误报为 api_error (静默降级)。
            from llm_generator import batch_dedup_mode as _m7_bd_mode
            _m7_dedup_mode = _m7_bd_mode()
            self.llm_gen.receive_context(
"""

# ═══════════════════════════════════════════════════════════════
# B2: 首轮 parse → 批内去重 → tag_semantic
# ═══════════════════════════════════════════════════════════════
B2_OLD = """                gen_candidates = self.llm_gen.tag_semantic_candidates(
                    self.llm_gen.parse_response(response_text))
"""

B2_NEW = """                # M7: 批内 AST 去重接入主线 (顺序与 generate() 内路径一致:
                # 去重 → 语义打标)。此前该阶段只存在于 generate() 内路径。
                gen_candidates = self.llm_gen.parse_response(response_text)
                gen_candidates = self.llm_gen.batch_dedup_ast(
                    gen_candidates, evidence=_m7_dedup_evidence,
                    mode=_m7_dedup_mode, stage="llm_round1")
                gen_candidates = self.llm_gen.tag_semantic_candidates(gen_candidates)
"""

# ═══════════════════════════════════════════════════════════════
# B3: reviewer 二轮 parse → 批内去重 → tag_semantic
# ═══════════════════════════════════════════════════════════════
B3_OLD = """                                _round2 = self.llm_gen.tag_semantic_candidates(
                                    self.llm_gen.parse_response(_resp2))
"""

B3_NEW = """                                # M7: 二轮同样先批内去重 (仅批内; 与首轮的
                                # 同骨架候选属定向改进, 不做跨轮合并去重)
                                _round2 = self.llm_gen.parse_response(_resp2)
                                _round2 = self.llm_gen.batch_dedup_ast(
                                    _round2, evidence=_m7_dedup_evidence,
                                    mode=_m7_dedup_mode, stage="llm_round2")
                                _round2 = self.llm_gen.tag_semantic_candidates(_round2)
"""

# ═══════════════════════════════════════════════════════════════
# B5: generate 结果字典追加减留档键
# ═══════════════════════════════════════════════════════════════
B5_OLD = """            "generator_preflight": _preflight_audit,
        }
"""

B5_NEW = """            "generator_preflight": _preflight_audit,
            # M7: 批内 AST 去重留档 (enforce 移除 / shadow 仅命中 均在 records)
            "batch_ast_dedup": {
                "mode": _m7_dedup_mode,
                "n_flagged": len(_m7_dedup_evidence),
                "n_removed": sum(1 for _r in _m7_dedup_evidence
                                 if _r.get("removed")),
                "records": _m7_dedup_evidence,
            },
        }
"""

PATCHES = [
    {"file": "research/factor_alchemy/llm_generator.py", "tag": "A1", "old": A1_OLD, "new": A1_NEW},
    {"file": "research/factor_alchemy/llm_generator.py", "tag": "A2", "old": A2_OLD, "new": A2_NEW},
    {"file": "research/factor_alchemy/ralph_loop.py", "tag": "B1", "old": B1_OLD, "new": B1_NEW},
    {"file": "research/factor_alchemy/ralph_loop.py", "tag": "B0", "old": B0_OLD, "new": B0_NEW},
    {"file": "research/factor_alchemy/ralph_loop.py", "tag": "B2", "old": B2_OLD, "new": B2_NEW},
    {"file": "research/factor_alchemy/ralph_loop.py", "tag": "B3", "old": B3_OLD, "new": B3_NEW},
    {"file": "research/factor_alchemy/ralph_loop.py", "tag": "B5", "old": B5_OLD, "new": B5_NEW},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际写入 (默认干跑)")
    args = ap.parse_args()

    by_file = {}
    for p in PATCHES:
        by_file.setdefault(p["file"], []).append(p)

    failed = 0
    for rel, plist in by_file.items():
        path = ROOT / rel
        raw = path.read_bytes()
        sha_before = hashlib.sha256(raw).hexdigest()
        n_crlf_before = raw.count(b"\r\n")
        n_lf_before = raw.count(b"\n")
        print(f"== {rel}")
        print(f"   sha256={sha_before[:16]}  CRLF={n_crlf_before}  "
              f"bareLF={n_lf_before - n_crlf_before}")
        text = raw.decode("utf-8")
        new_text = text
        file_fail = 0
        for p in plist:
            old, new = crlf(p["old"]), crlf(p["new"])
            cnt = new_text.count(old)
            ok = (cnt == 1)
            print(f"   [{p['tag']}] match={cnt} {'OK' if ok else '✗ 非唯一命中'}")
            if not ok:
                file_fail += 1
                continue
            new_text = new_text.replace(old, new, 1)
        if file_fail:
            failed += file_fail
            print("   ⚠ 有未唯一命中, 本文件不写入")
            continue
        nb = new_text.encode("utf-8")
        n_crlf = nb.count(b"\r\n")
        n_lf = nb.count(b"\n")
        if n_lf != n_crlf:
            print(f"   ✗ 行尾校验失败: CRLF={n_crlf} LF={n_lf} — 拒绝写入")
            failed += 1
            continue
        if not args.apply:
            print(f"   (干跑) 行尾校验通过: CRLF 将 {n_crlf_before} → {n_crlf}, bareLF=0")
            continue
        bak = path.with_suffix(path.suffix + BAK_SUFFIX)
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"   备份 → {bak.name}")
        else:
            print(f"   备份已存在, 保留: {bak.name}")
        path.write_bytes(nb)
        sha_after = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"   写入完成: CRLF {n_crlf_before} → {n_crlf}  "
              f"sha256={sha_after[:16]}")

    if failed:
        print(f"\n✗ 共 {failed} 处失败")
        return 1
    if args.apply:
        for rel in by_file:
            py_compile.compile(str(ROOT / rel), doraise=True)
            print(f"py_compile OK: {rel}")
        print("\n✓ 全部补丁应用完成")
    else:
        print("\n(干跑结束, 未写入任何文件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
