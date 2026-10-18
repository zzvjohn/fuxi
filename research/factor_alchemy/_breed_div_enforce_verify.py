# -*- coding: utf-8 -*-
"""
P1 转正配套: gp_breed 父本分层采样路径的生产数据验证 (2026-09-01)
====================================================================
背景: MAB 方向推荐会覆盖 --generator gp_breed (默认值被当作"未指定"),
  近 4 轮方向均推荐 forge → 育种路径 (GPBreeder._layered_template_weights
  批量折扣, 今晚重写) 尚未在真实轮次中运行。本脚本用生产模板池
  (Memory success_templates) 直接驱动该路径, 验证:
    A. enforce 折扣生效且方向正确 (高 corr 模板分数下降, corr=0 不变)
    B. 磁盘缓存 data/jq_div_corr_cache.json 落盘
    C. shadow/w0 基线一致性 (回退安全)
"""
import os
import sys
import json

os.environ["FUXI_DIV_DISCOUNT"] = "enforce"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experience_memory import get_memory
from factor_expression_tree import GPBreeder
from forge import diversity_discount as dd

DATA_DIR = r"e:/quant/data"


def load_templates():
    mem = get_memory()
    raw = mem.data.get("success_templates", []) or []
    out = []
    for t in raw:
        if isinstance(t, dict):
            out.append({
                "formula": t.get("formula", t.get("expression", "")) or "",
                "factor_name": t.get("pattern_id", t.get("name", "")) or "",
                "verification_level": t.get("verification_level", "stage2_only"),
                "success_rate": t.get("success_rate", 0) or 0,
                "occurrence_count": t.get("occurrence_count", 0) or 0,
            })
    return out


def main():
    templates = load_templates()
    print(f"A. 生产模板池: {len(templates)} 个 success_templates")
    if not templates:
        print("   模板池为空, 终止")
        return 1

    # enforce 口径
    b_enforce = GPBreeder(diversity_w=0.10, layered_selection=True)
    w_enforce = b_enforce._layered_template_weights(templates)

    # w=0 基线 (生产回退口径)
    dd.reset_shared_cache()
    b_base = GPBreeder(diversity_w=0.0, layered_selection=True)
    w_base = b_base._layered_template_weights(templates)

    diffs = [(i, w_base[i], w_enforce[i]) for i in range(len(templates))
             if abs(w_base[i] - w_enforce[i]) > 1e-9]
    print(f"B. 被折扣的模板: {len(diffs)}/{len(templates)}")
    for i, wb, we in sorted(diffs, key=lambda x: -(x[1] - x[2]))[:8]:
        nm = templates[i]["factor_name"][:36] or "?"
        fml = templates[i]["formula"][:50]
        print(f"   {wb:.4f} -> {we:.4f}  ({nm} | {fml})")

    # 方向正确性: 折扣只降不升
    assert all(we <= wb + 1e-9 for wb, we in
               [(w_base[i], w_enforce[i]) for i in range(len(templates))]), \
        "折扣出现分数上升 (bug)"
    print("C. 方向正确性: 折扣只降不升 ✓")

    # 磁盘缓存落盘检查
    cache_f = dd.JQDiversityCache._CACHE_FILE
    if cache_f.exists():
        with open(cache_f, encoding="utf-8") as f:
            data = json.load(f)
        n = len(data.get("entries", {}))
        sig = data.get("lib_sig", "")[:8]
        print(f"D. 磁盘缓存已落盘: {n} 条 (lib_sig={sig})")
    else:
        print("D. ⚠️ 磁盘缓存文件不存在 (全部求值失败或未触发保存)")
        return 1

    # 统计窗口
    snap = dd.COUNTERS.snapshot()
    print(f"E. 查询统计: {snap}")
    print("验证完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
