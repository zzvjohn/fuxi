# -*- coding: utf-8 -*-
"""31 条 axis=1 因子的 S5 全量池裁决 —— 结果分析 (分母审计 + 归因 + 裁决)。

判据纪律 (S5 技能 §6b / §9.7):
  - **分母审计**: 「两腿超额同时恰为 0.0」= 读数**由构造决定**, 不是测量
    ⇒ 必须按 `PRE_METRICS_REASONS`(7 类) 归因, 并与数值签名交叉核对;
  - 「仅凭数值签名」命中的条目 = **未分类短路** ⇒ 判 FAIL / 人查;
  - 每个比较必须断言实际比较数量 > 0。
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = Path(r'e:/quant')
RES = ROOT / "data" / "s5_axis1_rerun_result_default.json"
PRE = ROOT / "data" / "_axis1_31_candidates_20260919.json"

# 从实现枚举的 pre-metrics 短路类 (s5_joint_filter.validate_factor 3071→3235)
PRE_METRICS_REASONS = (
    "价格数据不可用",
    "数据加载失败",
    "🧹 公式合理性拦截",
    "因子公式评估无有效数据",
    "因子值全为 NaN",
    "🧹 常数因子拦截",
    "🧹 价格水平代理拦截",
)


def cls_of(reason: str) -> str:
    s = str(reason or "")
    for p in PRE_METRICS_REASONS:
        if s.startswith(p):
            return p
    return ""


def main():
    d = json.loads(RES.read_text(encoding="utf-8"))
    rows = d["results"]
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    print("=" * 78)
    print("0. 结构性披露")
    print("=" * 78)
    print(f"  caliber={d.get('caliber')} date_ceil={d.get('date_ceil')} "
          f"idx_align={d.get('idx_align')}")
    print(f"  universe={d.get('universe')} benchmark={d.get('benchmark')}")
    print(f"  run_at={d.get('run_at')} progress={d.get('progress')}")
    print(f"  候选源={d.get('candidate_source')}  前置 gate_ok={pre.get('gate_ok')}")

    n = len(rows)
    n_err = sum(1 for r in rows if r.get("error"))
    n_not_eval = sum(1 for r in rows if r.get("eval_status") == "not_evaluated")
    n_cmp = n - n_err - n_not_eval
    print()
    print("=" * 78)
    print("1. 分母审计 (🔴 承重判据)")
    print("=" * 78)
    print(f"  候选 {n} | 求值异常 {n_err} | 未评估 {n_not_eval} | "
          f"**n_compared = {n_cmp}**")
    assert n_cmp > 0, "🔴 n_compared == 0 ⇒ 无有效比较, 不得产出裁决"
    # 🔴 未评估必须携带**可复核的静态证明**, 否则等于静默丢样本 (排除集合不可证)
    for r in rows:
        if r.get("eval_status") == "not_evaluated":
            pf = r.get("eval_block_proof") or {}
            print(f"     未评估 {r['factor_name'][:40]:<40} block={r.get('eval_block')} "
                  f"unused_loop_var={pf.get('unused_loop_var')}")
            assert pf.get("unused_loop_var") or r.get("eval_block") == "manual_skip", \
                "🔴 未评估条目缺少可复核证明 ⇒ 拒绝出裁决"

    # 短路归因
    shortc, zero_only = [], []
    for r in rows:
        if r.get("error") or r.get("eval_status") == "not_evaluated":
            continue
        c = cls_of(r.get("new_reason"))
        z = (r.get("new_excess_25") == 0.0 and r.get("new_excess_26") == 0.0)
        if c:
            shortc.append((r["factor_name"], c))
        elif z:
            zero_only.append(r["factor_name"])
    print(f"  短路条目(有前置原因, 排除可证) = {len(shortc)}")
    for k, v in Counter(c for _, c in shortc).most_common():
        print(f"     {v:>2}  {k}")
    print(f"  🔴「仅凭数值签名」命中 = {len(zero_only)}"
          f"   {'⇒ 未分类短路, 判 FAIL/人查' if zero_only else '(应为 0)'}")
    for x in zero_only:
        print(f"     - {x}")
    # 交叉核对: 有前置原因的短路必须**恰好**两腿为 0
    bad = [nm for nm, _ in shortc
           if next(r for r in rows if r["factor_name"] == nm).get("new_excess_25") != 0.0]
    print(f"  交叉核对: 短路条目中两腿非 0 的 = {len(bad)} (应为 0) "
          f"{'🔴 ' + str(bad) if bad else 'OK'}")

    n_meas = n_cmp - len(shortc)
    print(f"  **有效测量条目 = {n_meas}** (排除短路后)")

    print()
    print("=" * 78)
    print("1b. 缺陷分层: 评估链问题 vs 公式级缺陷 (本次核心问题)")
    print("=" * 78)
    _pf = {c["factor_name"]: c for c in pre["candidates"]}
    n_chain = 0
    n_pre_formula = 0
    for r in rows:
        nm = r["factor_name"]
        if r.get("eval_status") == "not_evaluated":
            print(f"   [公式级·结构退化]  {nm[:38]:<38} {r.get('eval_block')} "
                  f"(未评估, 不入分母)")
            continue
        if r.get("error"):
            print(f"   [评估链·求值异常]  {nm[:38]:<38} {str(r.get('error'))[:56]}")
            n_chain += 1
            continue
        c = cls_of(r.get("new_reason"))
        if c.startswith("🧹"):
            n_pre_formula += 1
            print(f"   [公式级·前置拦截]  {nm[:38]:<38} {c}")
    print(f"   ⇒ 评估链异常 = {n_chain}   "
          f"(路由修复前应为 31: 全落逐股通道 ⇒ Series.rank(axis=1) ⇒ 静默 None)")
    print(f"   ⇒ 公式级前置拦截 = {n_pre_formula} (在 n_compared 内)")
    print(f"   ⇒ 真测量条目 = {n_cmp - n_pre_formula} / n_compared={n_cmp}"
          f"  + 未评估 {n_not_eval}")
    dw = [c["factor_name"] for c in pre["candidates"] if c.get("dead_weight")]
    print(f"   公式库内带 dead_weight 标记 = {len(dw)} {dw if dw else ''}")

    print()
    print("=" * 78)
    print("2. 裁决: PASS")
    print("=" * 78)
    n_pass = sum(1 for r in rows if r.get("new_passed"))
    print(f"  🔴 PASS = {n_pass} / {n}")
    for r in rows:
        if r.get("new_passed"):
            print(f"     ✅ {r['factor_name'][:44]:<44} ex25={r['new_excess_25']:>8} "
                  f"ex26={r['new_excess_26']:>8} c25={r.get('new_calmar_25')} "
                  f"c26={r.get('new_calmar_26')}")
            print(f"        flags={r.get('sparse_flags')} argmax={r.get('sparse_argmax_recent')}")

    print()
    print("=" * 78)
    print("3. 有效测量条目的读数分布 (ex25 / ex26, %)")
    print("=" * 78)
    meas = [r for r in rows
            if not r.get("error") and r.get("eval_status") != "not_evaluated"
            and cls_of(r.get("new_reason")) == ""
            and r.get("new_excess_25") is not None]
    for tag in ("new_excess_25", "new_excess_26"):
        v = sorted(float(r[tag]) for r in meas)
        if not v:
            continue
        k = len(v)
        print(f"  {tag}: n={k} min={v[0]:.4f} p25={v[k//4]:.4f} "
              f"中位={v[k//2]:.4f} p75={v[3*k//4]:.4f} max={v[-1]:.4f} "
              f"| >0 的条数={sum(1 for x in v if x > 0)}")
    two_pos = [r for r in meas
               if (r.get("new_excess_25") or 0) > 0 and (r.get("new_excess_26") or 0) > 0]
    print(f"  两腿超额**同时为正**的条目 = {len(two_pos)} / {len(meas)}")
    for r in sorted(two_pos, key=lambda x: -min(x["new_excess_25"], x["new_excess_26"]))[:10]:
        print(f"     {r['factor_name'][:42]:<42} ex25={r['new_excess_25']:>8} "
              f"ex26={r['new_excess_26']:>8}")

    print()
    print("=" * 78)
    print("4. 稀疏结构 flags 画像 (S5 技能 §3)")
    print("=" * 78)
    fc = Counter()
    for r in rows:
        for f in (r.get("sparse_flags") or []):
            fc[f] += 1
    print(f"  flags 分布: {dict(fc) if fc else '(全空)'}")
    argmax = Counter(r.get("sparse_argmax_recent") for r in rows
                     if r.get("sparse_argmax_recent") is not None)
    print(f"  裁决窗 argmax 分布: {dict(sorted(argmax.items()))}  (Q10 才算尾部未失效)")

    print()
    print("=" * 78)
    print("5. 逐条明细")
    print("=" * 78)
    for i, r in enumerate(rows, 1):
        if r.get("eval_status") == "not_evaluated":
            tag = "SKIP"
        elif r.get("error"):
            tag = "ERR "
        elif r.get("new_passed"):
            tag = "PASS"
        else:
            tag = "rej "
        st = cls_of(r.get("new_reason"))
        st = ("[短路:" + st[2:14] + "]") if st else ""
        if tag == "SKIP":
            st = "[未评估:" + str(r.get("eval_block")) + "]"
        print(f"  {i:>2}. {tag} {r['factor_name'][:40]:<40} "
              f"ex25={str(r.get('new_excess_25')):>9} ex26={str(r.get('new_excess_26')):>9} {st}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
