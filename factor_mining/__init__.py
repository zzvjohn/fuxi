"""因子挖掘方法库 (Factor Mining) — 伏羲 v2.0 的多样性/组合补充工具。

两个独立模块，共用同一份数据与同一套评估口径（`data_loader`）：

- `nsga2_biobjective` : 两目标 NSGA-II 组合层（收益= |ic_mean| 主导复合 + 换手），
  从预定义因子池选 ≤4 个因子做 rank 加权组合，产出收益-换手 Pareto 前沿。
- `gflownet`          : GFlowNet 因子表达式生成器（TB 损失 + GRU 前向策略），
  学习生成策略而非找最优，产出多样化的表达式因子候选。

与伏羲 v2.0（`fuxi_v1/`）的关系：
  - v2.0 生成层用有界 GA（`ga_ops.refine_ga/combine_ga`）+ 五臂（arms），
    父本选择用 `monitoring.pareto_archive`（quality=Spearman rank IC mean vs complexity）。
  - 本库的 GFlowNet 是 v2.0 生成层 **LLM 臂/探索臂旁的多样性补充**，不替代有界 GA。
  - 本库的两目标 NSGA-II 提供 v2.0 目前缺失的「多因子加权组合」能力。

调用入口：
  python -m factor_mining nsga2   --pop 120 --gen 40
  python -m factor_mining gflownet --ep 300 --bs 16 --n 300
  python -m factor_mining compare  --pop 120 --gen 40 --gf_ep 300 --gf_bs 16 --gf_n 300
"""
from .data_loader import (
    load_aligned,
    evaluate_composite,
    ic_strength_return_score,
    linear_combine,
)
from .nsga2_biobjective import NSGA2Biobjective

__all__ = [
    "load_aligned",
    "evaluate_composite",
    "ic_strength_return_score",
    "linear_combine",
    "NSGA2Biobjective",
]

__version__ = "1.0.0"
