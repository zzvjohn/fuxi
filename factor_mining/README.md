# factor_mining — 因子挖掘方法库

伏羲 v2.0 的**多样性/组合补充工具**，独立于 `fuxi_v1` 的治理链，可被每日自动化直接 `import` 或 `python -m` 调用。

## 两个模块

| 模块 | 作用 | 落点 |
|---|---|---|
| `nsga2_biobjective` | 两目标 NSGA-II **组合层**：从 97 因子池选 ≤4 个做 rank 加权组合，产出「收益-换手」Pareto 前沿 | 补充 v2.0 缺失的「多因子加权组合」能力 |
| `gflownet` | GFlowNet **因子表达式生成器**：TB 损失 + GRU 前向策略，学生成策略而非找最优，产出多样化表达式候选 | v2.0 生成层 LLM 臂/探索臂旁的**多样性补充** |

两者共用 `data_loader`（同一数据 + 同一评估口径），保证对比只反映优化器差异。

## 调用方式

```bash
# 两目标 NSGA-II 组合层
python -m factor_mining nsga2 --pop 120 --gen 40

# GFlowNet 因子表达式生成
python -m factor_mining gflownet --ep 300 --bs 16 --n 300

# 同口径对比（复用 run_comparison.py）
python -m factor_mining compare --pop 120 --gen 40 --gf_ep 300 --gf_bs 16 --gf_n 300
```

Python 接口（供自动化脚本 import，无需 subprocess）：

```python
from factor_mining import load_aligned, NSGA2Biobjective
factor_dict, fr = load_aligned()
ga = NSGA2Biobjective(factor_dict, fr, seed=42)
pareto = ga.run(pop_size=120, n_generations=40)
```

数据源默认 `e:/quant/research/factor_alchemy/output/{factor_cache,forward_returns}.pkl`，
可用环境变量覆盖（供每日自动化接入不同数据口径）：

```bash
export FUXI_FACTOR_CACHE=/path/to/factor_cache.pkl
export FUXI_FORWARD_RETURNS=/path/to/forward_returns.pkl
```

## 🔴 方法论铁律（血泪结论，2026-10-20 实验实证）

1. **比值型指标只能做「验收/筛选」，不能做「搜索目标」。**
   `ICIR = ic_mean / ic_std`、Sharpe、信息比这类**分母可被操纵**的指标，一旦被优化器
   追着跑，就会构造 `std→0` 的组合把比值除到无穷大（伪信号），GA 会全收敛到「稳定反向」
   的伪组合。实测：NSGA-II 改 `|ICIR|` 主导后 median ICIR 反恶化到 −2.05、正占比 0%、
   内部相关飙到 0.996（收敛到单一伪组合）。

2. **搜索目标用「分子型」指标**：截面 IC 均值 `ic_mean`（= Spearman rank IC mean，
   即排序强度），方向由 direction 检测自动翻转故取绝对值。伏羲 v2.0 生产代码的
   `pareto_archive` 里 `quality = e1.diagnostics.spearman.mean` 已经是这个口径 ✅。

3. **换手率要独立成目标（或强软约束），不能只做惩罚项**：`|ic_mean|` 只奖排序强度、
   不罚换手，强排序因子天然换手高（实测换手 0.07→0.32）。两目标 NSGA-II 把换手拎出
   独立成 Obj2；GFlowNet reward 用 `clip(1−2·turnover)` 强换手惩罚。

4. **GFlowNet 定位 = 多样性补充，不是替代 NSGA-II/有界 GA**：它学「生成策略」而非
   「找最优」，产出内部相关更低（0.20 vs 0.44）、与 NSGA-II 互补（跨方法相关 0.25）。
   用在生成层的冷门结构覆盖，不进组合权重层。

## 与伏羲 v2.0 的边界

- v2.0 生成层 = 有界 GA（`fuxi_v1/ga_ops.py` 的 `refine_ga`/`combine_ga`）+ 五臂
  （`fuxi_v1/arms.py`），父本选择 = `fuxi_v1/monitoring.py::pareto_archive`
  （quality vs complexity 两目标非支配筛选）。
- 本库**不改、不接管** v2.0 的任何源码/开关/授权。GFlowNet 与两目标 NSGA-II 是
  **旁挂工具**：产出候选表达式/组合后，由调用方决定是否喂进 v2.0 的 E0–E5 验证链。
