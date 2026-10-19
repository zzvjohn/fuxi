"""转正评分卡引擎：把 system_evolution 影子裁决翻译成转正标准逐条判定。

从 FROZEN.json + ASSESSMENT.json 机械评估转正标准，产出评分卡三要素：
  1. 建议 recommendation（RECOMMEND_PROMOTE / HOLD / REJECT）
  2. 理由 rationale（一句话）
  3. 标准逐条满足情况 gates（P 门通用必要条件 + S 门分类别充分条件）

边界：只出建议，不翻默认、不授资格、不切生产；转正永远人工（P7 四关）。
"""
from datetime import datetime, timezone
from pathlib import Path

from fuxi_v1.contracts import load_json

SE_DIR = Path(__file__).resolve().parents[2] / "data" / "system_evolution"
SCORECARD_SCHEMA = "fuxi.system_evolution.1.promotion"
P7_PENDING = ["反事实A/B", "默认口径档不变量", "端到端加载对账", "版本账本登记"]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _gate(record, _id, name, met, value, evidence, note=None):
    entry = {"id": _id, "name": name, "met": bool(met), "value": value, "evidence": evidence}
    if note:
        entry["note"] = note
    record.append(entry)


def build_scorecard(workspace, change_id):
    """从 FROZEN.json + ASSESSMENT.json 机械评估转正标准，产出评分卡。

    三态建议：
      RECOMMEND_PROMOTE  P1–P6 全✓ 且 S 门全✓（仍须人工 P7 四关后落账）
      HOLD               P1–P6 全✓ 但 S 门未满足（或经济类 forward 时长不足）
      REJECT             P1–P6 任一 ✗

    P 门度量随 kind 分支：
      score 型（correctness/research_method/economic_logic）→ 效应在 score（mean_delta）。
      efficiency → 效应在 wall_seconds（节省比例），score 只需"质量非劣"。
    """
    workspace = Path(workspace).resolve()
    directory = workspace / "data" / "system_evolution" / change_id
    frozen = load_json(directory / "FROZEN.json") if (directory / "FROZEN.json").exists() else None
    assessment = load_json(directory / "ASSESSMENT.json") if (directory / "ASSESSMENT.json").exists() else None
    gates = []

    if not frozen:
        return {"change_id": change_id, "recommendation": "REJECT",
                "rationale": "无冻结档案，不可转正",
                "gates": [], "p7_pending": P7_PENDING, "reviewer_required": True,
                "schema_version": SCORECARD_SCHEMA, "generated_at": _now()}

    plan = frozen.get("plan", {})
    kind = plan.get("kind")
    thresholds = plan.get("thresholds", {})
    noise_floor = float(thresholds.get("noise_floor", 0.0))
    margin = float(thresholds.get("quality_noninferiority_margin", 0.0))
    required_mult = 3 if kind == "economic_logic" else 2

    if not assessment:
        return {"change_id": change_id, "kind": kind, "decision": "frozen_not_evaluated",
                "recommendation": "REJECT",
                "rationale": "已冻结但尚未 assess，无影子裁决，不可转正",
                "gates": gates, "p7_pending": P7_PENDING, "reviewer_required": True,
                "schema_version": SCORECARD_SCHEMA, "generated_at": _now()}

    decision = assessment.get("decision")
    effects = assessment.get("effects", {}) or {}
    groups = effects.get("groups", {}) or {}
    case_deltas = effects.get("case_deltas", {}) or {}
    reason_code = assessment.get("reason_code", "")

    # P1 影子裁决通过
    _gate(gates, "P1", "影子裁决通过", decision == "review_shadow",
          str(decision), "decision")

    group_deltas = [g["mean_delta"] for g in groups.values() if "mean_delta" in g]
    min_delta = min(group_deltas) if group_deltas else 0.0
    saved = float(effects.get("wall_seconds_saved", 0.0))
    before = float(effects.get("incumbent_wall_seconds", 0.0))
    wall_ratio = (saved / before) if before > 0 else 0.0

    # P2 效应量显著（度量随 kind 分支）：
    #   efficiency  → wall 节省比例 ≥ 10%
    #   correctness → 存在真实修复（max case_delta > 噪声；稀疏修复不被 0-case 稀释）
    #   research_method / economic_logic → score 效应量 ≥ 2×（经济 ≥ 3×）噪声
    if kind == "efficiency":
        p2 = wall_ratio >= 0.10
        _gate(gates, "P2", "效应量显著(wall节省≥10%)", p2,
              f"{wall_ratio:.0%}", "wall_seconds_saved/incumbent_wall_seconds")
    elif kind == "correctness":
        max_delta = max(case_deltas.values()) if case_deltas else 0.0
        p2 = max_delta > noise_floor
        _gate(gates, "P2", "存在真实修复(maxΔ>噪声)", p2,
              f"maxΔ={max_delta:.4f}", "case_deltas")
    elif noise_floor > 0:
        p2_mult = min_delta / noise_floor
        p2 = p2_mult >= required_mult
        _gate(gates, "P2", f"效应量≥{required_mult}×噪声", p2,
              f"{p2_mult:.2f}×", "min(mean_delta)/noise_floor")
    else:
        _gate(gates, "P2", f"效应量≥{required_mult}×噪声", False,
              "noise_floor=0", "noise_floor")

    # P3 样本充分 n≥8
    n = int(effects.get("n_unique_tasks", 0))
    _gate(gates, "P3", "样本充分 n≥8", n >= 8, f"n={n}", "n_unique_tasks")

    # P4 方向一致性（度量随 kind 分支）：
    #   efficiency          → 质量非劣（score 不退化）
    #   correctness         → 无显著退化（修复稀疏，允许非 bug case 保持 0）
    #   research_method / economic_logic → 全正向（增益型，要求所有 cohort 正向）
    if kind == "efficiency":
        p4 = bool(group_deltas) and all(d >= -margin for d in group_deltas)
        _gate(gates, "P4", "质量非劣(score不退化)", p4, f"minΔ={min_delta:.4f}",
              "groups.mean_delta")
    elif kind == "correctness":
        p4 = bool(group_deltas) and all(d >= -noise_floor for d in group_deltas)
        _gate(gates, "P4", "无显著退化", p4, f"minΔ={min_delta:.4f}",
              "groups.mean_delta")
    elif group_deltas:
        p4 = all(d > 0 for d in group_deltas)
        _gate(gates, "P4", "方向一致(全正向)", p4,
              f"minΔ={min_delta:.4f}", "groups.mean_delta")
    else:
        _gate(gates, "P4", "方向一致(全正向)", False, "无group", "groups")

    # P5 jackknife 稳健性（度量随 kind 分支）：
    #   correctness / efficiency → 质量维度：去掉任一 case 后最差 delta 仍不退化
    #   research_method / economic_logic → 效应维度：去掉任一 case 后效应仍 > 噪声
    if case_deltas and len(case_deltas) > 1:
        vals = list(case_deltas.values())
        loo_means = []
        for i in range(len(vals)):
            rest = vals[:i] + vals[i + 1:]
            loo_means.append(sum(rest) / len(rest))
        min_loo = min(loo_means)
        if kind == "efficiency":
            p5 = min_loo >= -margin
            _gate(gates, "P5", "jackknife稳健(质量)", p5, f"min_loo={min_loo:.4f}",
                  "case_deltas leave-one-out")
        elif kind == "correctness":
            p5 = min_loo >= -noise_floor
            _gate(gates, "P5", "jackknife稳健(质量)", p5, f"min_loo={min_loo:.4f}",
                  "case_deltas leave-one-out")
        else:
            p5 = min_loo > noise_floor
            _gate(gates, "P5", "jackknife稳健(效应)", p5, f"min_loo={min_loo:.4f}",
                  "case_deltas leave-one-out")
    else:
        _gate(gates, "P5", "jackknife稳健", False,
              "无case_deltas" if not case_deltas else "n≤1", "case_deltas",
              note="旧档案缺 case_deltas，需重跑 assess 或补算")

    # P6 无退化 / 无越界
    worst = effects.get("worst_case_delta")
    max_loss = float(thresholds.get("maximum_case_loss", 0.0))
    p6_worst = (worst is not None) and worst >= -max_loss
    p6_budget = not reason_code.startswith("budget_exceeded")
    p6 = p6_worst and p6_budget
    _gate(gates, "P6", "无退化/越界", p6,
          f"worst={worst}, reason={reason_code}", "worst_case_delta/reason_code")

    # S 门（分类别充分条件）
    def s_gate(_id, name, met, value, evidence):
        _gate(gates, _id, name, met, value, evidence)

    if kind == "correctness":
        vals = list(case_deltas.values()) if case_deltas else []
        s_c1 = any(d > noise_floor for d in vals)
        s_gate("S-C1", "识别到正确性修复", s_c1,
               f"{sum(1 for d in vals if d > noise_floor)}个case>噪声", "case_deltas")
        status_counts = assessment.get("status_counts", {})
        challenger_counts = status_counts.get("challenger", {})
        s_c2 = challenger_counts.get("error", 0) == 0
        s_gate("S-C2", "无新增错误case", s_c2,
               f"challenger error={challenger_counts.get('error', 0)}", "status_counts")
        s_met = s_c1 and s_c2
    elif kind == "efficiency":
        min_saving = float(thresholds.get("minimum_wall_saving", 0.0))
        s_e1 = saved > min_saving and wall_ratio >= 0.10
        s_gate("S-E1", "wall节省达标(≥10%)", s_e1,
               f"saved={saved}s, {wall_ratio:.0%}", "wall_seconds_saved")
        s_e2 = all(d >= -margin for d in group_deltas)
        s_gate("S-E2", "质量非劣", s_e2, f"minΔ={min_delta:.4f}", "groups.mean_delta")
        s_met = s_e1 and s_e2
    elif kind == "research_method":
        min_gain = float(thresholds.get("minimum_mean_gain", 0.0))
        s_r1 = min_delta > max(noise_floor, min_gain)
        s_gate("S-R1", "transfer增益beyond噪声", s_r1, f"minΔ={min_delta:.4f}",
               "groups.mean_delta")
        engine_deltas = {}
        for gkey, g in groups.items():
            eng = gkey.split(":")[0]
            engine_deltas.setdefault(eng, []).append(g.get("mean_delta", 0.0))
        s_r2 = len(engine_deltas) >= 2 and all(min(ds) > noise_floor for ds in engine_deltas.values())
        s_gate("S-R2", "跨引擎一致(≥2引擎)", s_r2,
               f"engines={list(engine_deltas)}", "groups")
        s_met = s_r1 and s_r2
    elif kind == "economic_logic":
        forward_meta = load_json(directory / "FORWARD_META.json") \
            if (directory / "FORWARD_META.json").exists() else {}
        fwd_days = forward_meta.get("declared_forward_days", 0)
        min_fwd = forward_meta.get("minimum_forward_days", 60)
        s_g1 = fwd_days >= min_fwd
        s_gate("S-G1", f"forward时长达标(≥{min_fwd}天)", s_g1, f"{fwd_days}天",
               "FORWARD_META.json")
        s_g2 = min_delta > noise_floor and not assessment.get("economic_winner_confirmed", True)
        s_gate("S-G2", "增益beyond噪声且不越界", s_g2, f"minΔ={min_delta:.4f}",
               "groups.mean_delta")
        s_met = s_g1 and s_g2
    else:
        s_met = False

    p_all = all(g["met"] for g in gates if g["id"].startswith("P"))
    if not p_all:
        recommendation = "REJECT"
        rationale = f"P 门未全通过（{reason_code or decision}），不转正"
    elif s_met:
        recommendation = "RECOMMEND_PROMOTE"
        rationale = "P 门全✓ + S 门全✓，建议转正（须人工完成 P7 四关后落账）"
    else:
        recommendation = "HOLD"
        rationale = "P 门全✓ 但 S 门未满足，建议暂缓补证据后复审"

    return {
        "schema_version": SCORECARD_SCHEMA,
        "change_id": change_id,
        "kind": kind,
        "decision": decision,
        "recommendation": recommendation,
        "rationale": rationale,
        "gates": gates,
        "p7_pending": P7_PENDING,
        "generated_at": _now(),
        "reviewer_required": True,
    }
