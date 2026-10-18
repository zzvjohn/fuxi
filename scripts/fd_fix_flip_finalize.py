# -*- coding: utf-8 -*-
"""
FD 翻译结果修复 + 方向翻转重测 + 轻量去重 (一次性脚本)
==================================================================================
1. 修 6 个 'null' 字符串公式 (手工构造可执行 pandas 版)
2. 剔除前视泄漏因子 (open.shift(-1) 类) + 其他不可救 null 归 skip
3. 对 S1 强负 IC (ic<=-0.02 & |icir|>=0.1) 的因子翻转取负重跑
4. 公式文本相似度去重 (vs 库 expression + FD 内部互相)
5. 输出最终清单 data/_fd_tmp/fd_final_passed.json
"""
import json
import re
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "research" / "factor_alchemy"))
sys.path.insert(0, str(BASE / "scripts"))
from factor_ic_computer import FactorICComputer
from daily_factor_hypothesis import validate_and_preprocess_formula

TRANS_PATH = BASE / "data" / "_fd_tmp" / "translations.json"
S1_CSV = BASE / "data" / "_fd_tmp" / "fd_s1_results.csv"
OUT_FINAL = BASE / "data" / "_fd_tmp" / "fd_final_passed.json"

# 手工修复: slug -> pandas 公式 (均为 volatility/liquidity 类, 字段白名单内)
MANUAL_FIX = {
    "up-down-volatility-ratio":
        "returns.where(returns > 0).rolling(20).std().div(returns.where(returns < 0).rolling(20).std())",
    "value-at-risk":
        "returns.rolling(60).quantile(0.05)",
    "turnover-volatility":
        "turnover.rolling(20).std()",
    "turnover-rate-variation":
        "turnover.rolling(20).std().div(turnover.rolling(20).mean())",
    "trading-volume-fluctuation":
        "volume.rolling(20).std().div(volume.rolling(20).mean())",
    "illiquidity-coefficient-of-variation":
        "(returns.abs().div(amount.clip(lower=1e-8))).rolling(20).std().div((returns.abs().div(amount.clip(lower=1e-8))).rolling(20).mean())",
}
# 前视泄漏黑名单: 使用未来数据 (shift(-1) 等)
LEAKAGE_SLUGS = {
    "earnings-announcement-stock-jump": "公式用 open.shift(-1) 即次日开盘价=前视泄漏, IC 0.136/ICIR 1.10 为泄漏特征; 无公告日期数据无法无前视构造, 剔除",
}

# 翻转重测阈值
FLIP_IC_LO = -0.02
FLIP_ICIR_ABS = 0.10


def _norm(f: str) -> str:
    """公式文本规范化 (去空白/数值归一) 用于相似度比较"""
    f = re.sub(r"\s+", "", f)
    return f


def text_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def main():
    trans = json.load(open(TRANS_PATH, encoding="utf-8"))

    # ── 1. 修复 null ─────────────────────────────
    n_manual = 0
    for slug, fix in MANUAL_FIX.items():
        if trans.get(slug, {}).get("formula_pandas") in (None, "null"):
            trans[slug]["formula_pandas"] = fix
            trans[slug]["manual_fixed"] = True
            trans[slug]["skip_reason"] = None
            n_manual += 1
    # 前视剔除 + 其他 null 归 skip
    for slug, reason in LEAKAGE_SLUGS.items():
        if slug in trans:
            trans[slug]["formula_pandas"] = None
            trans[slug]["skip_reason"] = reason
            trans[slug]["leakage"] = True
    for slug, v in trans.items():
        if v.get("formula_pandas") in (None, "null"):
            v["formula_pandas"] = None
            if not v.get("skip_reason"):
                v["skip_reason"] = "无法翻译(缺数据/公式超链式能力)"
    json.dump(trans, open(TRANS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"修复: 手工补公式 {n_manual} 个 | 前视剔除 1 个")

    # ── 2. 读 S1 第一轮结果 ─────────────────────────
    import csv
    s1 = {}
    with open(S1_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            s1[row["slug"]] = row

    # ── 3. 翻转重测 ─────────────────────────────
    ic_comp = FactorICComputer()
    flip_candidates = []
    for slug, row in s1.items():
        ic = row.get("ic")
        icir = row.get("icir")
        if ic is None or icir is None or icir == "" or ic == "":
            continue
        ic, icir = float(ic), float(icir)
        if ic <= FLIP_IC_LO and abs(icir) >= FLIP_ICIR_ABS:
            v = trans.get(slug, {})
            fp = v.get("formula_pandas")
            if fp and not str(fp).startswith("-"):
                flip_candidates.append(slug)

    flip_results = {}
    print(f"翻转重测候选: {len(flip_candidates)} 个")
    for slug in flip_candidates:
        orig = trans[slug]["formula_pandas"]
        flipped = f"-({orig})"
        ok, warns, cleaned = validate_and_preprocess_formula(flipped, slug)
        if not ok:
            print(f"  {slug}: 翻转后预检失败 {warns[:1]}")
            flip_results[slug] = {"verdict": "FLIP_PRECHECK_FAIL"}
            continue
        try:
            r = ic_comp.compute(cleaned, slug)
            ic, icir, n = r["ic"], r["icir"], r["n_days"]
            passed = ic >= 0.02 and icir >= 0.1 and n >= 30
            flip_results[slug] = {
                "verdict": "FLIP_PASS" if passed else "FLIP_FAIL",
                "flipped_ic": ic, "flipped_icir": icir, "n_days": n,
                "flipped_formula": flipped,
            }
            trans[slug]["flip_test"] = flip_results[slug]
            print(f"  {slug}: 原ic={float(s1[slug]['ic']):.4f} → 翻转 {ic:.4f}/{icir:.4f} {'PASS' if passed else ''}")
        except Exception as e:
            flip_results[slug] = {"verdict": "FLIP_EVAL_ERROR", "note": str(e)[:80]}
    json.dump(trans, open(TRANS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ── 4. 组装最终清单 ─────────────────────────
    lib = json.load(open(BASE / "data" / "library_orthogonality_state.json", encoding="utf-8"))
    lib_exprs = [(_norm(str(fx.get("expression", ""))), fx.get("name", "")) for fx in lib.get("factors", []) if fx.get("expression")]

    final = []
    seen_formulas = {}
    for slug, v in trans.items():
        fp = v.get("formula_pandas")
        if not fp or fp == "null":
            continue
        # 决定最终公式: 翻转过线用翻转版, 否则用原版
        ft = v.get("flip_test", {})
        use_formula = fp
        use_ic = None
        use_icir = None
        row = s1.get(slug, {})
        if ft.get("verdict") == "FLIP_PASS":
            use_formula = ft["flipped_formula"]
            use_ic, use_icir = ft["flipped_ic"], ft["flipped_icir"]
        else:
            ic_v = row.get("ic")
            icir_v = row.get("icir")
            if ic_v not in (None, "") and icir_v not in (None, ""):
                use_ic, use_icir = float(ic_v), float(icir_v)

        # 过线判定 (无 S1 值的手工修复因子补测)
        if use_ic is None or use_icir is None:
            ok, warns, cleaned = validate_and_preprocess_formula(use_formula, slug)
            if not ok:
                continue
            try:
                r = ic_comp.compute(cleaned, slug)
                use_ic, use_icir, n = r["ic"], r["icir"], r["n_days"]
            except Exception:
                continue
        passed = (use_ic >= 0.02 and use_icir >= 0.1)

        if not passed:
            continue

        # 轻量去重: 与库 + 已选 FD 公式文本相似
        nfp = _norm(use_formula)
        dup_with = None
        for lexpr, lname in lib_exprs:
            if text_sim(nfp, lexpr) >= 0.88:
                dup_with = f"lib:{lname}"
                break
        if not dup_with and nfp in seen_formulas:
            dup_with = f"fd:{seen_formulas[nfp]}"
        if not dup_with:
            seen_formulas[nfp] = slug

        final.append({
            "slug": slug, "name_zh": v.get("name_zh", ""), "category": v.get("category", ""),
            "formula_pandas": use_formula,
            "ic": round(use_ic, 4), "icir": round(use_icir, 4),
            "direction": -1 if ft.get("verdict") == "FLIP_PASS" else v.get("direction", 1),
            "rationale": v.get("rationale", ""),
            "flipped": ft.get("verdict") == "FLIP_PASS",
            "dup_with": dup_with,
        })

    # 排序: icir 降序
    final.sort(key=lambda x: -x["icir"])
    json.dump({
        "finalized_at": datetime.now().isoformat(),
        "n_final": len(final),
        "n_flip_pass": sum(1 for x in final if x["flipped"]),
        "items": final,
    }, open(OUT_FINAL, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n最终过线清单: {len(final)} 个 (其中方向翻转 {sum(1 for x in final if x['flipped'])} 个)")
    for x in final:
        tag = " [翻转]" if x["flipped"] else ""
        dup = f" [重复:{x['dup_with']}]" if x.get("dup_with") else ""
        print(f"  {x['slug']}: ic={x['ic']:.4f} icir={x['icir']:.4f}{tag}{dup}")


if __name__ == "__main__":
    main()
