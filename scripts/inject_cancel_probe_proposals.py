# -*- coding: utf-8 -*-
"""P-20260822-001 落地: 撤单博弈日线代理公式注入 stage1_factor_proposals.json

审核裁决 (2026-08-22): 有条件通过, 定位=范式探针。
- 公式与库内 23 个 tail 类因子语义重叠风险已标注 → 检验时若与 tail 族正交性 <0.7 即弃。
- 避开 hard_forbidden: gp_breed_054 上影线做空族 (short_extreme_rank_reversal) → 全部 direction=long,
  不做 -(high-close).rank 做空端极值。
- 公式风格: pandas .rolling()/np.maximum/np.minimum, 与 stage1_factor_proposals 现有惯例一致。
"""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "stage1_factor_proposals.json"

NEW_FACTORS = [
    {
        "factor_name": "cancel_probe_lower_wick_ratio",
        "label": "下影线承接比(撤单试盘代理)",
        "formula_pandas": "((np.minimum(open, close) - low) / (high - low).clip(lower=0.01)).rolling(10).mean()",
        "economic_rationale": (
            "海通撤单博弈代理: 无逐笔数据时, 下影线占比高的K线=盘中卖压(含试盘撤单式砸盘)"
            "被买方承接吸收的痕迹, 承接强的股票短期走强 (买方护盘意愿)。"
            "long: 下影线承接比高 → 买盘占优。"
        ),
        "paradigm": "高频微观结构",
        "direction": "long",
        "source_proposal": "P-20260822-001",
        "generated_at": datetime.now().isoformat(),
    },
    {
        "factor_name": "cancel_probe_wick_asymmetry_20d",
        "label": "影线不对称度20日均值(撤单博弈代理)",
        "formula_pandas": "(((np.minimum(open, close) - low) - (high - np.maximum(open, close))) / (high - low).clip(lower=0.01)).rolling(20).mean()",
        "economic_rationale": (
            "下影线与上影线之差: 正值=下影主导(卖压承接), 负值=上影主导(买方试盘后被压制)。"
            "与硬禁止族区别: 本因子为 long 端影线结构比值, 非上影线极值做空 rank。"
        ),
        "paradigm": "高频微观结构",
        "direction": "long",
        "source_proposal": "P-20260822-001",
        "generated_at": datetime.now().isoformat(),
    },
    {
        "factor_name": "cancel_probe_close_position_rise",
        "label": "收盘位置抬升(日内博弈控盘代理)",
        "formula_pandas": "((close - low) / (high - low).clip(lower=0.01)).rolling(20).mean() - ((close - low) / (high - low).clip(lower=0.01)).rolling(60).mean()",
        "economic_rationale": (
            "收盘在日内振幅中的位置: 20日均值 vs 60日均值的抬升幅度。"
            "收盘位置系统性上移 = 日内买方逐步控盘(承接后封在相对高位), 中期上涨前兆。"
        ),
        "paradigm": "高频微观结构",
        "direction": "long",
        "source_proposal": "P-20260822-001",
        "generated_at": datetime.now().isoformat(),
    },
    {
        "factor_name": "cancel_probe_wick_volume_divergence",
        "label": "缩量下影扩展(卖压衰竭试盘代理)",
        "formula_pandas": "((np.minimum(open, close) - low) / close).rolling(5).mean() * (volume.rolling(60).mean() / volume.rolling(5).mean().clip(lower=1))",
        "economic_rationale": (
            "下影线绝对值放大 × 近期量比收缩: 卖压衰竭型试盘(撤单式砸盘无跟进量) → "
            "卖方力竭信号, 承接成本低, 短期反转向上概率大。long。"
        ),
        "paradigm": "高频微观结构",
        "direction": "long",
        "source_proposal": "P-20260822-001",
        "generated_at": datetime.now().isoformat(),
    },
]


def main():
    with open(PATH, encoding="utf-8") as f:
        items = json.load(f)
    items = items if isinstance(items, list) else items.get("proposals", [])
    existing = {it.get("factor_name") for it in items}
    added = 0
    for nf in NEW_FACTORS:
        if nf["factor_name"] in existing:
            print("SKIP dup:", nf["factor_name"])
            continue
        items.append(nf)
        added += 1
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    with open(PATH, encoding="utf-8") as f:
        json.load(f)  # 写回校验
    print(f"P-001 注入完成: 新增 {added} 条 | 提案库总量 {len(items)}")


if __name__ == "__main__":
    main()
