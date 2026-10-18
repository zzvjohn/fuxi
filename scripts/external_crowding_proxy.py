# -*- coding: utf-8 -*-
"""P-20260919-004: 因子拥挤「机构使用家数」代理监控 (影子观测, 纯零干预)。

目的
====
为拥挤度监控补上「外部采用」维度: 以本地 RAG 知识库 (27,980 chunks 量化研究存量)
召回片段中提取的因子公式为**外部语料**, 用 AST 结构指纹 Jaccard 相似度
(复用 P-20260915-005 structure_dedup 同源的 trajectory_pool.ast_fingerprint)
估算库内因子 (NOVEL_FACTOR_LIBRARY + stage1_factor_proposals) 的
「被借鉴热度」近似分 (外部语料命中数)。

语义边界 (诚实披露)
====================
  * 本代理度量的是「本地知识库存量研究中与该因子结构相似的公式条数」,
    是「被借鉴热度」的**下界近似**, 不是真实机构使用家数。
  * 100wjjw/雪球/浙商外部实证 (>30 家使用 => 胜率掉两成) 仅作阈值动机,
    本影子不做任何胜率换算、不做任何干预。

输出
====
  data/external_crowding_proxy.json: top 因子 + 命中详情 + 语料规模 + 降级标注
  stdout: Top-N 预警行 (供日报风险预警节引用)

开关/回退
=========
  env FUXI_EXT_CROWD_SIM   相似度阈值 (默认 0.55)
  env FUXI_EXT_CROWD_K     每查询召回片段数 (默认 15)
  纯影子观测: 删脚本即回退, 零裁决零干预。

🔴 解释器约定 (两段式, 2026-09-19 实测 segfault 教训)
====================================================
  RAG 索引 (FAISS + bge embedding) **只能在 D:/RAG/venv 解释器里加载**,
  用研究 venv 直接 load 会 Segfault (ABI 不匹配)。
  ⇒ 语料提取段: 本脚本以 `--dump-corpus` 模式被 D:/RAG/venv python 子进程执行;
  ⇒ 相似度段: 默认入口在研究解释器里跑, 读语料 JSON 做 AST 指纹比对。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent          # e:/quant
ALCHEMY = BASE / "research" / "factor_alchemy"
OUT_PATH = BASE / "data" / "external_crowding_proxy.json"

for p in (str(ALCHEMY), str(BASE)):
    if p not in sys.path:
        sys.path.insert(0, p)

SIM_THRESHOLD = float(os.environ.get("FUXI_EXT_CROWD_SIM", "0.55") or 0.55)
RECALL_K = int(os.environ.get("FUXI_EXT_CROWD_K", "30") or 30)
TOP_N = 10

# 外部语料查询词 (覆盖主要范式族, 与日报 2.2 前沿维度对齐)
CORPUS_QUERIES = [
    "因子 构造 公式 rolling",
    "动量因子 反转因子 构造方法",
    "量价因子 换手率 波动率 公式",
    "龙虎榜 资金流 因子",
    "估值因子 盈利因子 质量因子",
    "机器学习 因子挖掘 alpha",
    "隔夜收益 日内收益 因子",
    "特异度 换手率因子 低波动",
    "大单 主力资金 因子公式",
    "技术指标 因子化 公式",
]

# 公式样态行提取: pandas 风格因子公式的常见特征
FORMULA_PAT = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\([^()]*\))?"
    r"(?:\.(?:rolling|shift|rank|pct_change|diff|mean|std|sum|corr|abs|clip)"
    r"\([^()]*\)){1,}")
MIN_LEN, MAX_LEN = 15, 300


def _extract_formulas_from_text(text: str) -> list:
    """从一段研究文本中提取公式样态字符串 (每行取最长匹配)。"""
    out = []
    for line in str(text).splitlines():
        line = line.strip()
        if not (MIN_LEN <= len(line) <= MAX_LEN):
            continue
        best = ""
        for m in FORMULA_PAT.finditer(line):
            g = m.group(0)
            # 剔除无变量主语的方法链碎片 (rolling(...).mean() 这种抓不全的)
            if g.split("(")[0].split(".")[0] in (
                    "rolling", "shift", "rank", "pct_change", "diff",
                    "mean", "std", "sum", "corr", "abs", "clip"):
                continue
            if len(g) > len(best):
                best = g
        if best and any(k in best for k in ("rolling", "shift", "pct_change")):
            out.append(best)
    return out


def build_external_corpus() -> tuple:
    """(--dump-corpus 模式, 必须在 D:/RAG/venv 解释器下运行)
    从本地 RAG 召回片段提取外部公式语料。失败降级返回空语料。"""
    snippets = []          # [(formula, source_title)]
    degraded = ""
    try:
        store = _get_rag_store()
    except Exception as e:
        return snippets, f"[RAG:不可用] {type(e).__name__}: {str(e)[:100]}"
    seen_idx = set()
    for q in CORPUS_QUERIES:
        try:
            hits = store.query(q, k=RECALL_K)
        except Exception as e:
            degraded = f"[RAG:查询失败] {type(e).__name__}: {str(e)[:100]}"
            continue
        for h in hits:
            idx = h.get("_idx")
            if idx in seen_idx:
                continue
            seen_idx.add(idx)
            title = str(h.get("title", h.get("source_file", "?")))[:60]
            for f in _extract_formulas_from_text(h.get("text", "")):
                snippets.append((f, title))
    return snippets, degraded


_RAG_STORE = None


def _get_rag_store():
    global _RAG_STORE
    if _RAG_STORE is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_rag_store_ext", "D:/RAG/rag_store.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RAG_STORE = mod.RAGStore("D:/RAG/rag_index")
    return _RAG_STORE


def build_library_formulas() -> list:
    """库内因子: NOVEL_FACTOR_LIBRARY (源码正则提取, 避免 import 副作用)
    + stage1_factor_proposals.json 未测提案。"""
    lib = []
    seen = set()
    # (a) NOVEL_FACTOR_LIBRARY 字面列表 (scripts/daily_factor_hypothesis.py)
    src = (BASE / "scripts" / "daily_factor_hypothesis.py").read_text(encoding="utf-8")
    for m in re.finditer(
            r"'name':\s*'([^']+)'.*?'formula':\s*'([^']+)'", src, re.S):
        name, f = m.group(1), m.group(2)
        if name not in seen and len(f) <= 600:
            seen.add(name)
            lib.append((name, f, "NOVEL_LIBRARY"))
    # (b) stage1 因子提案
    p = BASE / "data" / "stage1_factor_proposals.json"
    if p.exists():
        try:
            d = json.load(open(p, encoding="utf-8"))
            arr = d if isinstance(d, list) else d.get("proposals", [])
            for it in arr:
                f = it.get("formula") or it.get("expression") or ""
                name = it.get("factor_name") or it.get("name") or ""
                if name and f and name not in seen and len(f) <= 600:
                    seen.add(name)
                    lib.append((name, f, "stage1_proposals"))
        except Exception:
            pass
    return lib


def main() -> int:
    ap = argparse.ArgumentParser(description="外部拥挤代理影子监控 (P-20260919-004)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--dump-corpus", dest="dump_corpus", default=None,
                    help="语料提取段: 写入指定 JSON 后退出 (须用 D:/RAG/venv python 执行)")
    args = ap.parse_args()

    if args.dump_corpus:
        # ── 段 A: 语料提取 (RAG venv 解释器) ──
        snippets, degraded = build_external_corpus()
        json.dump({"date": datetime.now().strftime("%Y-%m-%d"),
                   "corpus_size": len(snippets), "degraded": degraded,
                   "snippets": [{"formula": f, "source": t} for f, t in snippets]},
                  open(args.dump_corpus, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"[ExtCrowd/dump] 语料 {len(snippets)} 条 -> {args.dump_corpus}"
              + (f" | {degraded}" if degraded else ""))
        return 0

    # ── 段 B: AST 相似度 (研究解释器) ──
    from trajectory_pool import ast_fingerprint  # AST 指纹 (structure_dedup 同源)

    lib = build_library_formulas()

    # 语料文件: 不存在或非今日 -> 调 RAG venv 子进程重提
    corpus_path = BASE / "data" / "external_crowding_corpus.json"
    today = datetime.now().strftime("%Y-%m-%d")
    degraded = ""
    corpus_meta = {}
    need_dump = True
    if corpus_path.exists():
        try:
            corpus_meta = json.load(open(corpus_path, encoding="utf-8"))
            need_dump = corpus_meta.get("date") != today
        except Exception:
            need_dump = True
    if need_dump:
        import subprocess
        rag_py = "D:/RAG/venv/Scripts/python.exe"
        try:
            r = subprocess.run([rag_py, str(Path(__file__).resolve()),
                                "--dump-corpus", str(corpus_path)],
                               capture_output=True, text=True, timeout=600,
                               encoding="utf-8", errors="replace")
            if r.returncode != 0:
                degraded = (f"[RAG:语料提取失败] exit={r.returncode} "
                            f"{(r.stderr or '')[-120:]}")
            corpus_meta = json.load(open(corpus_path, encoding="utf-8"))
        except Exception as e:
            degraded = f"[RAG:语料提取异常] {type(e).__name__}: {str(e)[:100]}"
            corpus_meta = {"snippets": [], "corpus_size": 0, "date": ""}
    snippets = [(s.get("formula", ""), s.get("source", "?"))
                for s in corpus_meta.get("snippets", [])
                if s.get("formula")]
    if corpus_meta.get("degraded") and not degraded:
        degraded = corpus_meta["degraded"]

    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "proposal": "P-20260919-004",
        "mode": "shadow",
        "forbidden_use": "纯观测零干预: 本代理为「被借鉴热度」下界近似, 不是真实机构家数; "
                         "不做胜率换算、不做任何裁决/干预。删脚本即回退。",
        "sim_threshold": SIM_THRESHOLD,
        "n_library": len(lib),
        "corpus_size": len(snippets),
        "corpus_date": corpus_meta.get("date", ""),
        "degraded": degraded,
        "top": [],
    }

    if not snippets:
        payload["note"] = "外部语料为空, 无法计算热度 (RAG 降级), 仅记录不预警。"
    elif lib:
        # 🔴 复合指纹 (2026-09-19 首跑实证修正):
        #   纯 AST op-multiset 对泛型短链 ({rolling,std}) 天然打满 1.0, 命中全是噪音;
        #   叠加归一化字符 bigram (捕捉变量名差异) 后, sim = 0.5*AST + 0.5*bigram。
        from trajectory_pool import _normalize_formula

        def _bigrams(f: str) -> Counter:
            n = _normalize_formula(f)
            return Counter(n[i:i + 2] for i in range(max(1, len(n) - 1)))

        def _jac(c1: Counter, c2: Counter) -> float:
            union = sum((c1 | c2).values())
            return sum((c1 & c2).values()) / union if union else 0.0

        corp = []
        for f, title in snippets:
            ast_fp = ast_fingerprint(f)
            if len(ast_fp) < 2:          # 过小结构不参与 (退化守卫)
                continue
            corp.append((ast_fp, _bigrams(f), title))
        payload["corpus_size_usable"] = len(corp)
        n_generic = 0
        scored = []
        for name, formula, src_tag in lib:
            ast_fp = ast_fingerprint(formula)
            if len(ast_fp) < 2:
                n_generic += 1
                continue
            bg_fp = _bigrams(formula)
            hits = 0
            best = 0.0
            best_title = ""
            for c_ast, c_bg, title in corp:
                sim = 0.5 * _jac(ast_fp, c_ast) + 0.5 * _jac(bg_fp, c_bg)
                if sim >= SIM_THRESHOLD:
                    hits += 1
                    if sim > best:
                        best, best_title = sim, title
            if hits:
                scored.append({"factor_name": name, "source": src_tag,
                               "hits": hits, "best_sim": round(best, 3),
                               "best_match_source": best_title})
        payload["n_library_generic_skipped"] = n_generic
        scored.sort(key=lambda x: (-x["hits"], -x["best_sim"]))
        payload["top"] = scored[:TOP_N]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(OUT_PATH, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    if not args.quiet:
        print(f"[ExtCrowd] shadow 模式 (纯观测零干预): 库内因子 {len(lib)} | "
              f"外部语料 {len(snippets)} 条 | 阈值 {SIM_THRESHOLD} "
              f"| 命中因子 {len(payload['top'])}")
        if degraded:
            print(f"[ExtCrowd] ⚠️ {degraded}")
        for t in payload["top"][:5]:
            print(f"  ⚠️ {t['factor_name']}: 被借鉴热度 {t['hits']} 条 "
                  f"(best_sim {t['best_sim']}, 源: {t['best_match_source']})")
        print(f"[ExtCrowd] 报告: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
