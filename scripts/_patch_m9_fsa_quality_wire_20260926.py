# -*- coding: utf-8 -*-
"""M9 (2026-09-26): FSA 阈值降级为观察 + 统一质量门接线 — 字节级补丁。

背景 (reports/fuxi_alpha_generation_overhaul_20260926.md M9 / 证据 E1):
  两个"假门禁"病灶:
  (1) 库级 FSA: share 判据 (15% × n_factors 绝对次数) 在现行 1122 因子库下
      不可达 (max count=5 < 阈值 168) ⇒ forbidden_skeletons 恒 0, 生成端约束
      空转。dominance 判据 (FUXI_FSA_MODE=dominance) 实测可产出 15 条冻结,
      但默认未切换 (未走「翻默认四关」)。决定: **明确降级为观察** —
      不修阈值不翻默认, 把"不可达"从静默 0 变成显式读数 (reachable=False)。
  (2) 统一质量门 `_gate_fatal`: 自诞生起只在 ralph_loop 写入与打印
      (L2242-2256), 从未被任何下游消费 ⇒ 未过门候选照常进入 S1-S5。
      修复: FUXI_QUALITY_GATE_WIRE = off | shadow(默认) | enforce —
      shadow 记账留档 (零行为变化), enforce 拒绝 (证据留档不无声消失)。

改动 (三文件, 7 处):
  ralph_loop.py (CRLF)
    [C1] 模块级 quality_gate_wire_mode / quality_gate_wire_step (纯函数分级)
    [C2] _phase_generate 公共初始化新增 M9 留档变量
    [C3] R 阶段 FSA 注入后新增可达性诊断 (只读, 打印 + 落 retrieve_result)
    [C4] 质量门打印块之后接 wire 分级 (shadow 记账 / enforce 拒绝)
    [C5] generate 结果字典追加 "quality_gate_wire" / "fsa_context" 留档键
  subtree_fingerprinter.py (CRLF)
    [D1] 新增 get_context_status() — 库级 FSA 判据可达性诊断 (只读)
  forbidden_gate.py (LF)
    [E1] docstring 更新: 从"永不可达 (200/200 轮恒 0)"更新为 M9 复测事实
         (share 不可达 / dominance 可切未切 / forge 运行内 FSA 是唯一消费方)

文件纪律: ralph_loop/subtree 为纯 CRLF (bare LF=0), forbidden_gate 为纯 LF
  (CRLF=0); 字节级替换; 每处要求唯一命中; 先备份 .bak_m9_20260926;
  默认干跑; 写后按文件实测行尾校验 (crlf: LF==CRLF; lf: CRLF 不新增)。

用法:
  python scripts/_patch_m9_fsa_quality_wire_20260926.py          # 干跑
  python scripts/_patch_m9_fsa_quality_wire_20260926.py --apply  # 应用
"""
import argparse
import hashlib
import py_compile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAK_SUFFIX = ".bak_m9_20260926"


def norm(text: str, eol: str) -> str:
    """把文本规范为目标文件的行尾 (crlf/lf)。"""
    t = text.replace("\r\n", "\n")
    if eol == "crlf":
        return t.replace("\n", "\r\n")
    return t


# ═══════════════════════════════════════════════════════════════
# C1: 模块级 M9 helper (插在 _bh_panel_to_signal 之前)
# ═══════════════════════════════════════════════════════════════
C1_OLD = """def _bh_panel_to_signal(df, n_days: int = 60):
"""

C1_NEW = """# ═══════════════════════════════════════════════════════════════
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
    \"\"\"解析 FUXI_QUALITY_GATE_WIRE (off|shadow|enforce, 默认 shadow)。

    非法值直接抛 ValueError (fail-fast; 配置错误不得被静默吞成默认)。
    \"\"\"
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
    \"\"\"M9: 统一质量门分级 (纯函数, 零副作用; 基于候选上已有的 _gate_result)。

    输入契约: 候选须已由调用方跑过 FactorQualityGate.verify (写入
    `_gate_result` / `_gate_fatal`)。无 `_gate_result` 的候选视为**未评估**
    — 不计入 n_fatal, 也不被 enforce 误删 (评估口径: 未评估≠拒绝)。
    enforce 仅删除"有判定且未过门"的候选; 其余模式 kept 原样返回。
    \"\"\"
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
"""

# ═══════════════════════════════════════════════════════════════
# C2: _phase_generate 公共初始化 — M9 留档变量
# ═══════════════════════════════════════════════════════════════
C2_OLD = """        _m7_dedup_evidence = []
        _m7_dedup_mode = ""
"""

C2_NEW = """        _m7_dedup_evidence = []
        _m7_dedup_mode = ""
        # M9: 统一质量门接线留档 (shadow 记账 / enforce 拒绝; 随 return 落盘)。
        _qgw_mode = ""
        _qgw_records = []
        _qgw_n_checked = 0
        _qgw_n_fatal = 0
        _qgw_n_removed = 0
"""

# ═══════════════════════════════════════════════════════════════
# C3: R 阶段 FSA 注入后 — 可达性诊断
# ═══════════════════════════════════════════════════════════════
C3_OLD = """        # v0.3: 注入 FSA 禁止列表到检索结果
        if use_fsa:
            fsa_forbidden = self.fsa.get_forbidden_for_generation()
            retrieve_result["fsa_forbidden_context"] = fsa_forbidden
"""

C3_NEW = """        # v0.3: 注入 FSA 禁止列表到检索结果
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
"""

# ═══════════════════════════════════════════════════════════════
# C4: 质量门打印块之后 — wire 分级 (12 空格缩进, review 分支内)
# ═══════════════════════════════════════════════════════════════
C4_OLD = """            # v0.5 伏羲: 打印详细拒绝原因 (仅前3个)
            if n_passed < len(gen_candidates):
                rejected = [c for c in gen_candidates if not (c.get("_gate_result") and c["_gate_result"].passed)]
                for i, rc in enumerate(rejected[:3]):
                    fatal = rc.get("_gate_fatal", [])
                    formula_preview = rc.get("formula", "")[:60]
                    print(f"    ❌ #{i+1} [{formula_preview}...]: {'; '.join(fatal[:2])}")
"""

C4_NEW = """            # v0.5 伏羲: 打印详细拒绝原因 (仅前3个)
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
"""

# ═══════════════════════════════════════════════════════════════
# C5: generate 结果字典追加 M9 留档键
# ═══════════════════════════════════════════════════════════════
C5_OLD = """            "generator_preflight": _preflight_audit,
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

C5_NEW = """            "generator_preflight": _preflight_audit,
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
"""

# ═══════════════════════════════════════════════════════════════
# D1: subtree_fingerprinter — 新增 get_context_status (只读诊断)
# ═══════════════════════════════════════════════════════════════
D1_OLD = """        if len(forbidden) > 5:
            lines.append(f"... 另有 {len(forbidden) - 5} 个被禁骨架")

        return "\\n".join(lines)
"""

D1_NEW = """        if len(forbidden) > 5:
            lines.append(f"... 另有 {len(forbidden) - 5} 个被禁骨架")

        return "\\n".join(lines)

    def get_context_status(self) -> Dict:
        \"\"\"M9 (2026-09-26): 库级 FSA 判据可达性诊断 (只读, 不改状态)。

        share 判据 = max(int(n_factors × 0.15), 2) 的绝对次数。M9 复测:
        现行 1122 因子库下 max count=5 < 阈值 168 ⇒ forbidden_skeletons
        恒 0, 生成端约束**空转** (≠ 真实无同质)。本方法把该事实显式化,
        供 ralph_loop 打印与 batch 证据 (fsa_context) 消费, 零行为变更。
        dominance 判据 (FUXI_FSA_MODE=dominance) 与库规模无关, 实测可产出
        冻结, 但默认未切换 (未走「翻默认四关」)。

        Returns
        -------
        dict: mode / n_skeletons / n_forbidden / max_count / median_count /
              reachable (True=可达 / False=不可达 / None=无状态) / diag
        \"\"\"
        counts = [s.count for s in self.skeletons.values()]
        n_skel = len(counts)
        forbidden = self.get_forbidden_skeletons()
        out = {
            "mode": self.mode,
            "n_skeletons": n_skel,
            "n_forbidden": len(forbidden),
            "max_count": max(counts) if counts else 0,
            "median_count": (sorted(counts)[n_skel // 2] if counts else 0),
            "reachable": True,
            "diag": "",
        }
        if n_skel == 0:
            out["reachable"] = None
            out["diag"] = "FSA 状态为空 (库未扫描过或状态文件缺失)"
            return out
        if self.mode == "share":
            n_factors = sum(counts)  # 每因子贡献一次骨架计数 ⇒ 参与扫描的因子数
            thr_count = max(int(n_factors * self.threshold), 2)
            out["n_factors_est"] = n_factors
            out["threshold_count"] = thr_count
            if out["max_count"] < thr_count:
                out["reachable"] = False
                out["diag"] = (
                    f"FSA[share] 库级冻结判据不可达: 最大 count={out['max_count']} "
                    f"< 阈值 {thr_count} (15% × {n_factors} 因子)。生成端约束当前为"
                    f"空转 (非真实无同质)。可选: FUXI_FSA_MODE=dominance 启用相对"
                    f"判据 (count ≥ {self.dominance_min_abs} 且 ≥ k × 中位数)。")
        return out
"""

# ═══════════════════════════════════════════════════════════════
# E1: forbidden_gate docstring — 更新为 M9 复测事实
# ═══════════════════════════════════════════════════════════════
E1_OLD = """  - FSA (`subtree_fingerprinter`) 的 `forbidden_skeletons` 走的是"库内骨架频次
    超阈值"路径, 实测阈值 15% × 271 骨架 = 需 ≥40 次才冻结, 而最大 count=3
    ⇒ **永不可达** (200/200 轮恒 0)。它与本门是**互补**关系:
      FSA  = 库内自身同质化(内生, 频次驱动)
      本门 = 已证伪方向的黑名单(外生, 失败驱动)
"""

E1_NEW = """  - FSA (`subtree_fingerprinter`) 的 `forbidden_skeletons` 走的是"库内骨架频次
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
"""

PATCHES = [
    {"file": "research/factor_alchemy/ralph_loop.py", "tag": "C1", "old": C1_OLD, "new": C1_NEW},
    {"file": "research/factor_alchemy/ralph_loop.py", "tag": "C2", "old": C2_OLD, "new": C2_NEW},
    {"file": "research/factor_alchemy/ralph_loop.py", "tag": "C3", "old": C3_OLD, "new": C3_NEW},
    {"file": "research/factor_alchemy/ralph_loop.py", "tag": "C4", "old": C4_OLD, "new": C4_NEW},
    {"file": "research/factor_alchemy/ralph_loop.py", "tag": "C5", "old": C5_OLD, "new": C5_NEW},
    {"file": "research/factor_alchemy/subtree_fingerprinter.py", "tag": "D1",
     "old": D1_OLD, "new": D1_NEW},
    {"file": "research/factor_alchemy/forbidden_gate.py", "tag": "E1",
     "old": E1_OLD, "new": E1_NEW},
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
        bare_before = n_lf_before - n_crlf_before
        eol = "crlf" if n_crlf_before > 0 else "lf"
        print(f"== {rel}  (eol={eol})")
        print(f"   sha256={sha_before[:16]}  CRLF={n_crlf_before}  bareLF={bare_before}")
        text = raw.decode("utf-8")
        new_text = text
        file_fail = 0
        for p in plist:
            old, new = norm(p["old"], eol), norm(p["new"], eol)
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
        if eol == "crlf":
            ok_eol = (n_lf == n_crlf)
            detail = f"CRLF 将 {n_crlf_before} → {n_crlf}, bareLF=0"
        else:
            ok_eol = (n_crlf == n_crlf_before)
            detail = f"LF 保持, CRLF 不新增 ({n_crlf_before} → {n_crlf})"
        if not ok_eol:
            print(f"   ✗ 行尾校验失败: CRLF={n_crlf} LF={n_lf} — 拒绝写入")
            failed += 1
            continue
        if not args.apply:
            print(f"   (干跑) 行尾校验通过: {detail}")
            continue
        bak = path.with_suffix(path.suffix + BAK_SUFFIX)
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"   备份 → {bak.name}")
        else:
            print(f"   备份已存在, 保留: {bak.name}")
        path.write_bytes(nb)
        sha_after = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"   写入完成: {detail}  sha256={sha_after[:16]}")

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
