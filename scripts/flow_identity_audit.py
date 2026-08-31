# -*- coding: utf-8 -*-
"""
scripts/flow_identity_audit.py — P-20260822-002 落地
资金流族伪动力学审计 (partial-out 检验)
=============================================

背景
----
FinScope 每日因子 (2026-08) 警告: RankCorr(超大单20d净流入, 小单20d净流入) 受恒等式
「超大单+大单+中单+小单 = 全市场净流入」约束, 是伪动力学因子 — 其 alpha 可被
流入绝对量级完全解释, 不携带独立的资金流动力学信息。

方法
----
对每个因子, 截面逐日 partial-out:
    factor_t ~ 1 + ctrl_t     (ctrl = 当日截面 rank of log1p(|net_mf_amount|))
残差后 Rank IC 与原始 IC 对比:
    retention = resid_ic / raw_ic   (符号感知)
    retention < 0.5 且 raw_ic > 0.005  → pseudo_dynamics (伪动力学)
    0.5 <= retention < 0.8              → weak_redundant (弱量级冗余)
    retention >= 0.8                    → clean (alpha 独立于流入量级)

控制变量选择 (审核裁决 2026-08-22):
    主控制 = log1p(|net_mf_amount|) 截面秩 — 「流入绝对量级」窄代理。
    禁用成交额(amount)本身: 资金流 alpha 与成交额天然相关, 会把真 alpha 一起残差掉。

对照基准 (方法有效性自检):
    neg_control_1: 'net_mf_amount.abs()'         纯量级因子 → 必须被标记 (残差 IC ≈ 0)
    neg_control_2: 'buy_lg_vol + sell_lg_vol'    大单毛活动量   → 预期被标记
    pos_control_1/2: 库内非资金流已知好因子       → 残差后 IC 不应崩 (方法不误杀)

只打标不改库: 标记结果写 data/flow_identity_audit_YYYYMMDD.json,
不改 library_orthogonality_state.json, 不降权, 不删除。

用法:
    cd <项目根目录> && python scripts/flow_identity_audit.py            # 全族 51 因子
    cd <项目根目录> && python scripts/flow_identity_audit.py --top-n 8  # 只看 IC 正 Top-N
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

from factor_ic_computer import FactorICComputer  # noqa: E402

ORTHO_PATH = ROOT / "data" / "library_orthogonality_state.json"
STAGE1_FP_PATH = ROOT / "data" / "stage1_factor_proposals.json"
OUT_DIR = ROOT / "data"
PARTIAL_PATH = OUT_DIR / "flow_identity_audit_partial.json"

# ── 审计对象 ──────────────────────────────────────────────
AUDIT_PARADIGM = "资金流"

# 提案点名的关键嫌疑 (8-21 PASS, 不在 454 库内, 从 stage1_factor_proposals 拉公式)
SUSPECT_NAMES = ["large_order_flow_direction_consistency_20d_v2"]

# 对照基准
BENCH_NEG = [
    {"name": "NEG_CTRL_magnitude_only", "formula": "net_mf_amount.abs()",
     "desc": "纯流入量级 (无方向) — 残差后 IC 必须崩塌, 验证方法敏感性"},
    {"name": "NEG_CTRL_gross_lg_activity", "formula": "buy_lg_vol + sell_lg_vol",
     "desc": "大单毛活动量 — 量级代理, 预期被标记"},
]
BENCH_POS = [
    # 库内非资金流已知好因子 (表达式不含 moneyflow 字段)
    {"name": "POS_CTRL_max_drawdown_duration", "library_name": "max_drawdown_duration",
     "desc": "结构突变族 IC +0.1246 — 残差后 IC 不应崩, 验证方法不误杀"},
    {"name": "POS_CTRL_lottery_demand_suppression", "library_name": "lottery_demand_suppression_score",
     "desc": "截面异常族 IC +0.0722 — 同上"},
]

MF_FIELD_MARKERS = (
    "buy_lg_vol", "sell_lg_vol", "buy_sm_vol", "sell_sm_vol",
    "buy_md_vol", "sell_md_vol", "buy_elg_vol", "sell_elg_vol",
    "net_mf_vol", "net_mf_amount",
)


def _json_safe(o):
    """np 标量 → python 原生类型 (2026-08-22 铁律: json.dump 前必须转换)。"""
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def load_library():
    with open(ORTHO_PATH, encoding="utf-8") as f:
        s = json.load(f)
    return s.get("factors", [])


def load_suspects():
    """从 stage1_factor_proposals 拉 PASS 嫌疑因子公式。"""
    out = []
    if not STAGE1_FP_PATH.exists():
        return out
    with open(STAGE1_FP_PATH, encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("proposals", [])
    for it in items:
        if it.get("factor_name") in SUSPECT_NAMES:
            out.append({
                "name": it["factor_name"],
                "expression": it.get("formula_pandas", ""),
                "paradigm": it.get("paradigm", AUDIT_PARADIGM),
                "direction": it.get("direction", "long"),
                "recorded_ic": it.get("ic_mean"),
                "recorded_icir": it.get("icir"),
                "source": "stage1_factor_proposals (suspect)",
            })
    return out


def spearman_ic(factor: pd.Series, fwd: pd.Series) -> float:
    """截面 Rank IC, 不足 3 个有效值返回 NaN。"""
    valid = factor.notna() & fwd.notna()
    if valid.sum() < 3:
        return np.nan
    r1 = factor[valid].rank()
    r2 = fwd[valid].rank()
    if r1.nunique() < 2 or r2.nunique() < 2:
        return np.nan
    return r1.corr(r2)


def audit_one_factor(merged: pd.DataFrame) -> dict:
    """逐日截面: raw IC + partial-out 残差 IC。
    (pandas 3.x groupby.apply 会丢弃 group key 列 → 直接单循环, 顺带省去 apply 开销)
    """
    raw_ics, resid_ics = [], []
    for _, group in merged.groupby("trade_date"):
        g = group.dropna(subset=["factor_value", "ctrl", "fwd_return"])
        if len(g) < 30:
            continue
        raw = spearman_ic(g["factor_value"], g["fwd_return"])
        if not np.isnan(raw):
            raw_ics.append(raw)
        y = g["factor_value"].values.astype(float)
        x_rank = pd.Series(g["ctrl"].values.astype(float)).rank().values
        X = np.column_stack([np.ones(len(y)), x_rank])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = pd.Series(y - X @ beta, index=g.index)
        res = spearman_ic(resid, g["fwd_return"])
        if not np.isnan(res):
            resid_ics.append(res)
    return {"raw_ics": raw_ics, "resid_ics": resid_ics}


def aggregate(ics: list) -> dict:
    if not ics:
        return {"ic": 0.0, "icir": 0.0, "n": 0}
    arr = np.array(ics, dtype=float)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    return {
        "ic": round(mean, 5),
        "icir": round(abs(mean / std), 4) if std > 0 else 0.0,
        "n": int((~np.isnan(arr)).sum()),
    }


def classify(raw_ic: float, resid_ic: float, retention: float) -> str:
    if abs(raw_ic) < 0.005:
        return "no_signal"
    if retention < 0.5:
        return "pseudo_dynamics"
    if retention < 0.8:
        return "weak_redundant"
    return "clean"


def run_audit(top_n: int | None = None) -> dict:
    library = load_library()

    # ── 审计对象: 资金流族 51 + 嫌疑因子 ──
    flow_factors = [
        {"name": f["name"], "expression": f.get("expression", ""),
         "paradigm": f.get("paradigm", ""),
         "recorded_ic": f.get("ic"), "recorded_icir": f.get("icir"),
         "source": "library"}
        for f in library if str(f.get("paradigm", "")) == AUDIT_PARADIGM
    ]
    suspects = load_suspects()
    targets = flow_factors + suspects
    print(f"[审计] 资金流族 {len(flow_factors)} 因子 + 嫌疑 {len(suspects)} 个 = {len(targets)} 对象")

    if top_n:
        targets.sort(key=lambda x: -(abs(x.get("recorded_ic") or 0)))
        targets = targets[:top_n]
        print(f"[审计] --top-n {top_n} → 审计前 {len(targets)} 个")

    # ── 基准 ──
    lib_by_name = {f["name"]: f for f in library}
    benchmarks = []
    for b in BENCH_NEG:
        benchmarks.append({"kind": "neg_control", **b})
    for b in BENCH_POS:
        lib_f = lib_by_name.get(b["library_name"])
        if lib_f is None:
            print(f"[警告] 正对照 {b['library_name']} 不在库, 跳过")
            continue
        benchmarks.append({
            "kind": "pos_control", "name": b["name"],
            "formula": lib_f.get("expression", ""),
            "desc": b["desc"],
        })

    # ── IC Computer (口径对齐: 270 天/前向 5 日/全市场) ──
    comp = FactorICComputer()
    comp._load_data()
    base = comp._price_df[["ts_code", "trade_date", "close", "net_mf_amount"]].copy()
    base = base.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    base["fwd_return"] = base.groupby("ts_code")["close"].shift(-comp.forward_period) / base["close"] - 1
    base["ctrl_raw"] = np.log1p(base["net_mf_amount"].abs())
    base["ctrl"] = base.groupby("trade_date")["ctrl_raw"].rank()
    base = base.dropna(subset=["fwd_return"])

    # ── 逐因子 partial-out (断点续跑: 已完成因子落盘 partial, 重跑自动跳过) ──
    partial: dict = {}
    if PARTIAL_PATH.exists():
        with open(PARTIAL_PATH, encoding="utf-8") as f:
            partial = json.load(f)
        if partial.get("_audit_date") != datetime.now().strftime("%Y-%m-%d"):
            partial = {}   # 跨天检查点失效, 重新审计
        else:
            print(f"[续跑] 已从检查点恢复 {len(partial) - 1} 个已完成对象")
    if "_audit_date" not in partial:
        partial["_audit_date"] = datetime.now().strftime("%Y-%m-%d")
    all_names = [t["name"] for t in targets] + [b["name"] for b in benchmarks]
    results = [v for k, v in partial.items() if not k.startswith("_")]
    for i, spec in enumerate(targets + benchmarks):
        name = spec["name"]
        key = f"{spec.get('kind', 'factor')}::{name}"
        if key in partial:
            print(f"  [{i+1}/{len(all_names)}] {name}: 已在检查点, 跳过")
            continue
        formula = spec.get("expression") or spec.get("formula")
        if not formula:
            print(f"  [{i+1}/{len(all_names)}] {name}: 无公式 SKIP")
            results.append({"name": name, "kind": spec.get("kind", "factor"),
                            "flag": "eval_fail", "note": "无公式",
                            "paradigm": spec.get("paradigm", "")})
            partial[key] = results[-1]
            with open(PARTIAL_PATH, "w", encoding="utf-8") as f:
                json.dump(partial, f, ensure_ascii=False, default=_json_safe)
            continue
        panel = comp._eval_formula(formula, name)
        if panel is None or len(panel) == 0:
            print(f"  [{i+1}/{len(all_names)}] {name}: eval 失败/空 SKIP")
            results.append({"name": name, "kind": spec.get("kind", "factor"),
                            "flag": "eval_fail", "note": "受限 eval 环境不支持 (import/DSL 类公式)",
                            "paradigm": spec.get("paradigm", ""),
                            "recorded_ic": spec.get("recorded_ic"),
                            "recorded_icir": spec.get("recorded_icir")})
            partial[key] = results[-1]
            with open(PARTIAL_PATH, "w", encoding="utf-8") as f:
                json.dump(partial, f, ensure_ascii=False, default=_json_safe)
            continue
        merged = panel.merge(base, on=["ts_code", "trade_date"], how="inner")
        series = audit_one_factor(merged)
        raw = aggregate(series["raw_ics"])
        resid = aggregate(series["resid_ics"])
        if raw["ic"] == 0.0 and raw["n"] == 0:
            retention = float("nan")
        elif abs(raw["ic"]) < 1e-6:
            retention = float("nan")
        else:
            retention = round(resid["ic"] / raw["ic"], 3)
        flag = classify(raw["ic"], resid["ic"], retention)
        results.append({
            "name": name,
            "kind": spec.get("kind", "factor"),
            "paradigm": spec.get("paradigm", ""),
            "raw_ic": raw["ic"], "raw_icir": raw["icir"], "raw_n": raw["n"],
            "resid_ic": resid["ic"], "resid_icir": resid["icir"], "resid_n": resid["n"],
            "retention": retention if not (isinstance(retention, float) and np.isnan(retention)) else None,
            "flag": flag,
            "recorded_ic": spec.get("recorded_ic"),
            "recorded_icir": spec.get("recorded_icir"),
            "source": spec.get("source", ""),
        })
        partial[key] = results[-1]
        with open(PARTIAL_PATH, "w", encoding="utf-8") as f:
            json.dump(partial, f, ensure_ascii=False, default=_json_safe)
        print(f"  [{i+1}/{len(all_names)}] {name:50s} raw_ic={raw['ic']:+.4f} "
              f"resid_ic={resid['ic']:+.4f} retention={retention} → {flag}", flush=True)

    # ── 汇总 ──
    factors = [r for r in results if r["kind"] == "factor"]
    pseudo = [r["name"] for r in factors if r["flag"] == "pseudo_dynamics"]
    weak = [r["name"] for r in factors if r["flag"] == "weak_redundant"]
    clean = [r["name"] for r in factors if r["flag"] == "clean"]
    neg_b = [r for r in results if r["kind"] == "neg_control"]
    pos_b = [r for r in results if r["kind"] == "pos_control"]

    report = {
        "audit_date": datetime.now().strftime("%Y-%m-%d"),
        "proposal": "P-20260822-002",
        "method": {
            "control": "daily cross-sectional rank of log1p(|net_mf_amount|)",
            "residualization": "per-day OLS factor ~ 1 + ctrl_rank",
            "forward_period": comp.forward_period,
            "lookback_days": comp.lookback_days,
            "classify": {
                "pseudo_dynamics": "|raw_ic|>0.005 且 retention<0.5",
                "weak_redundant": "|raw_ic|>0.005 且 0.5<=retention<0.8",
                "clean": "retention>=0.8 或 |raw_ic|<0.004",
            },
        },
        "benchmark_check": {
            "neg_controls": [{k: r.get(k) for k in ("name", "retention", "flag", "raw_ic", "resid_ic")} for r in neg_b],
            "pos_controls": [{k: r.get(k) for k in ("name", "retention", "flag", "raw_ic", "resid_ic")} for r in pos_b],
            # 灵敏度: 纯量级负对照必须被标记; 特异性: 正对照必须不被误杀。
            # 毛活动量代理 (gross_lg_activity) 为探索性对照, 不参与有效性判据。
            "method_valid": (
                any(r["name"] == "NEG_CTRL_magnitude_only"
                    and r["flag"] in ("pseudo_dynamics", "weak_redundant") for r in neg_b)
                and all(r["flag"] == "clean" for r in pos_b if r.get("retention") is not None)
            ) if neg_b and pos_b else None,
        },
        "summary": {
            "audited": len(factors),
            "eval_failed": len([r for r in factors if r["flag"] == "eval_fail"]),
            "pseudo_dynamics": len(pseudo),
            "weak_redundant": len(weak),
            "clean": len(clean),
            "flagged_names": {"pseudo_dynamics": pseudo, "weak_redundant": weak},
        },
        "results": results,
    }

    out_path = OUT_DIR / f"flow_identity_audit_{datetime.now():%Y%m%d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_safe)

    print("\n" + "=" * 78)
    n_fail = len([r for r in factors if r["flag"] == "eval_fail"])
    print(f"审计完成: {len(factors)} 因子 → 伪动力学 {len(pseudo)} / 弱冗余 {len(weak)} / 干净 {len(clean)} / eval失败 {n_fail}")
    print(f"方法自检: 负对照 {[r['flag'] for r in neg_b]} | 正对照 {[r['flag'] for r in pos_b]}")
    if n_fail:
        print(f"⏭️  eval 失败 (受限环境, 保留记录 IC 不标记): "
              f"{[r['name'] for r in factors if r['flag'] == 'eval_fail']}")
    if pseudo:
        print(f"⚠️  伪动力学标记: {pseudo}")
    if weak:
        print(f"🟡 弱量级冗余: {weak}")
    print(f"报告: {out_path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="资金流族伪动力学审计 (P-20260822-002)")
    ap.add_argument("--top-n", type=int, default=None, help="只审计 |recorded_ic| Top-N 个 (调试用)")
    args = ap.parse_args()
    run_audit(top_n=args.top_n)
