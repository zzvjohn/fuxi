# -*- coding: utf-8 -*-
"""生成 B04.1 独立数据接受 assurance 素材 (独立第二源校验)。
诚实边界: 数据源独立已达成(申万/中证 vs tushare), reviewer 身份仍为同一 operator。
"""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
SS = ROOT / "data" / "fuxi_v1" / "second_source"

def load(path):
    return json.load(open(path, encoding="utf-8"))

sw_manifest = load(SS / "shenwan" / "20261016" / "MANIFEST.json")
csi_manifest = load(SS / "csindex" / "20261016" / "MANIFEST.json")
crosscheck = load(ROOT / "reports" / "second_source_crosscheck_20261016.json")

assurance = {
    "schema_version": "fuxi.second_source_assurance_material.1",
    "purpose": "B04.1 独立数据接受 — 独立第二源交叉校验素材",
    "produced_at": datetime.now(timezone.utc).isoformat(),
    "independent_second_sources": {
        "shenwan": {
            "vendor": "申万宏源 swsresearch.com 官方 API (SW2021)",
            "role": "行业分类独立校验",
            "snapshot_tag": "20261016",
            "manifest_hash": sw_manifest["manifest_hash"],
            "member_count": sw_manifest["member_count"],
            "mapped_count": sw_manifest["mapped_count"],
            "unmapped_count": sw_manifest["unmapped_count"],
            "status": sw_manifest["status"],
            "retrieved_at": sw_manifest["retrieved_at"],
        },
        "csindex": {
            "vendor": "中证指数 csindex.com.cn 官方 OSS (中证全指 000985)",
            "role": "自由流通权重独立校验",
            "snapshot_tag": "20261016",
            "manifest_hash": csi_manifest["manifest_hash"],
            "member_count": csi_manifest["member_count"],
            "weighted_count": csi_manifest["weighted_count"],
            "closeweight_date": csi_manifest["closeweight_date"],
            "cons_date": csi_manifest["cons_date"],
            "status": csi_manifest["status"],
            "retrieved_at": csi_manifest["retrieved_at"],
        },
    },
    "crosscheck_results": {
        "industry": {
            "method": "申万官方 API vs 主源 sw_industry_map.csv 同股行业一致性",
            "intersection": crosscheck["industry_crosscheck"]["intersection"],
            "agree": crosscheck["industry_crosscheck"]["agree"],
            "agree_rate": crosscheck["industry_crosscheck"]["agree_rate"],
            "disagree_count": crosscheck["industry_crosscheck"]["disagree_count"],
            "conclusion": "89.88% 一致, 10.12% 为申万行业分类调整(主源映射陈旧), 申万第二源可捕获主源行业陈旧",
        },
        "float": {
            "method": "中证 closeweight vs 主源 daily_basic (Spearman 秩相关)",
            "spearman_weight_vs_circ_mv": crosscheck["float_crosscheck"]["spearman_weight_vs_circ_mv"],
            "spearman_weight_vs_total_mv": crosscheck["float_crosscheck"]["spearman_weight_vs_total_mv"],
            "spearman_weight_vs_free_share": crosscheck["float_crosscheck"]["spearman_weight_vs_free_share"],
            "conclusion": crosscheck["float_crosscheck"]["interpretation"],
        },
    },
    "reviewer_identity": {
        "data_source_independence": "achieved (申万/中证 vs 主源 tushare, 不同 vendor)",
        "reviewer_identity_independence": "NOT_achieved (reviewer 仍为同一 operator/agent, 非第三方人工)",
        "honest_note": "本材料是独立第二源的交叉校验素材, 证明数据源独立; 但不构成 fuxi.independent_data_assurance.1 的正式保证——后者要求真第三方 reviewer 身份(import_assurance 的 generator!=reviewer 约束), 需人工/外部承担",
    },
    "trust_boundary": "independent_second_source_material_not_automatic_proof",
}

out = ROOT / "reports" / "second_source_assurance_b04_1_20261016.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(assurance, f, ensure_ascii=False, indent=2)
print(json.dumps(assurance, ensure_ascii=False, indent=2))
print("\n[written]", out)
