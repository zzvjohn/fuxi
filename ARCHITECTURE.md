# 伏羲 v2.0 系统架构

> **版本**：v2.0 ｜ **定版日期**：2026-10-18
> **代码基线**：`fuxi_v1` kernel
> **范围说明**：本文覆盖研究主链路（**E 道：E0–E5**）与执行契约。落地影子层（**G 道：G0–G2，P5 shadow**）仍在验证期、不参与任何决策，故不纳入本文。

---

## 1. 定位与边界

### 1.1 系统定位

伏羲 v2.0 是一套 **A 股量化因子研究的闭环系统**：从假设生成、表达式搜索、证据分级，到 JQ 回测与前瞻归档，全部以「可复现的声明 + 事务性证据」驱动。

核心原则：

- **证据驱动**：任何结论必须锚定在实现、数据列或结构化字段上，不锚定文件名或人工叙述。
- **不做无证据的裁决**：执行失败 ≠ 方向证伪；数据不足 ≠ 因子无效。
- **确定性优先**：同一 seed、同一快照、同一配置，必须复现同一学习轨迹。

### 1.2 能力边界

`unified_runner.run()` 是唯一的单次工作流入口，它**不能**：

- 准入候选（admission 由 E 道证据决定）
- 提交 JQ 实盘（导出与执行分离）
- 修改判官或评估策略
- 发布策略

它**只做**一件事：在冻结的请求与预算内，产出可审计的研究证据。

### 1.3 版本坐标系

| 坐标系 | 含义 | 载体 |
|---|---|---|
| v0.7.x | 旧伏羲的工程迭代账本（逐批提案） | `data/version_registry.json` |
| v1.0 | `fuxi_v1` kernel 语义版本 | `fuxi_v1/__init__.py` |
| **v2.0** | **本文档描述的产品定版**（融合 v0.7 + v1.0 + v1.1 影子验收层） | 本文件 |

三者并存不冲突：v2.0 是产品版本，v0.7.x 是内部工程批次编号。

---

## 2. 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│ 入口层   cycle_orchestrator（日循环） · unified_runner（单次）│
├─────────────────────────────────────────────────────────────┤
│ 数据层   native64 快照 · 历史 PIT 快照 · 列式 ScoreGrid      │
├─────────────────────────────────────────────────────────────┤
│ 研究层   L0 原型归并 → L1 假设选择 → L2 臂调度 → L3 反思     │
├─────────────────────────────────────────────────────────────┤
│ 搜索层   fixed / random_ast / refine / combine（有界 GA）    │
├─────────────────────────────────────────────────────────────┤
│ 评估层   E0 契约 → E1 统计 → E2 独立 → E3 经济 → E4 → E5     │
├─────────────────────────────────────────────────────────────┤
│ 证据层   evidence.sqlite（事务性 facts）· governance（写者锁）│
├─────────────────────────────────────────────────────────────┤
│ 出口层   JQ 策略导出 · 前瞻归档 · 独立数据认证               │
└─────────────────────────────────────────────────────────────┘
```

数据流方向单一：**数据 → 研究 → 搜索 → 评估 → 证据 → 出口**。没有任何一层会反向修改上游的冻结产物。

---

## 3. 模块详解

### 3.1 入口层

| 模块 | 职责 | 关键接口 |
|---|---|---|
| `scripts/cycle_orchestrator.py` | 每日闭环编排：select → generate → evaluate → reflect | 脚本级入口 |
| `fuxi_v1/unified_runner.py` | 单次有界工作流，source-keyed 幂等 | `run(workspace, request, *, trigger='manual')` |
| `fuxi_v1/__main__.py` | CLI 命令族 | `snapshot-history` / `campaign` / `history-study` / `forward-archive` / `certify-data` / `operate` |

`run()` 的四个阶段（按顺序执行，可断点续跑）：

1. `catalog_before` — 冻结因子目录快照
2. `research` — 核心研究阶段（调用 `campaign.run_campaign`）
3. `feedback` — 反馈写回
4. `catalog_after` — 目录后快照

产物：`PHASE_RESEARCH.json`、`COMPLETE.json`、`RECEIPT.json`、`RUN_PROTOCOL.json`。

### 3.2 数据层

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `datasets.py` | 本地 CSV → native64 快照（`local_market_input`） | **分交易所 hash 抽样**（固定 salt，沪深均衡，无幸存者偏差） |
| `historical_adapter.py` | 历史 PIT 快照构建（`build_history_input`） | 消费 `plan['cohort']['members']`，含退市标的 |
| `history.py` | 历史 cohort 抽样与复权分类 | `select_cohort`：per-exchange hash cutoff，`per_exchange=48` |
| `cohort_input.py` | 显式 cohort 声明与校验 | `verify_declared_cohort` |
| `pit_universe.py` | PIT 宇宙构建 | 按 score(salt, ts_code) 抽样 |
| `columnar.py` | 列式快照（ScoreGrid） | 大截面/大宇宙走列式，避免 dict-of-dict 内存爆炸 |
| `full_market_input.py` | 全市场口径输入 | 与 native64 抽样口径严格分离 |
| `data.py` | `Snapshot` / `Labels` 运行时对象 | 单次加载，只读 |

**抽样纪律（v2.0 关键修复）**：禁止用 `sorted(codes)[:N]` 做字符串排序抽样——它会让 `000*.SZ` 垄断样本。必须用 hash 抽样并断言沪深双所均有覆盖。

### 3.3 研究层（L0–L3）

| 层 | 模块 | 职责 |
|---|---|---|
| **L0** | `archetypes.py` + `data/paradigm_archetype_map.json` | 把 731 个深扫 paradigm 归并到 **24 个 archetype**（两阶段确定性匹配） |
| **L1** | `campaign.py::select_hypotheses` | 按 archetype 路由选择假设，避免单一机制垄断 |
| **L2** | `campaign.py::choose_arm` | UCB 臂调度 + **diversity budget**（HHI 集中度降权，`random_ast` 豁免） |
| **L3** | `cycle_orchestrator.py` reflect 阶段 | 自进化生成 `external_search_queries` |

`choose_arm` 的臂列表：`["fixed", "random_ast", "refine", "combine"]`。前 4 轮为冷启动轮扫（每臂各一轮），之后按 UCB 分数选择，并保留最小探索保证。

#### 知识注入：结构化知识卡（`scripts/knowledge_card.py`）

外部知识进入系统的唯一合规通道是**知识卡**，而不是直接写假设：

```
原始材料（论文/开源库/社区/研报）
   ↓  结构化
知识卡 data/knowledge_cards.json   （schema: fuxi.knowledge_card.1，append-only）
   ↓  promote（派生）
因子候选 data/stage1_factor_proposals.json
   ↓  select_hypotheses
external_hypotheses → campaign
```

一张卡必须记录五组信息：

| 组 | 字段 |
|---|---|
| 来源身份 | `title` / `url` / `doi` / `arxiv_id` / `doc_version` / `retrieved_at` / `license` |
| 机制与公式 | `economic_rationale` / `formula_pandas` 或 `formula_latex` / `direction`(1/-1) / `required_fields` |
| 适用范围 | `market` / `universe` / `frequency` / `sample_period` / `disclosure_lag` / `earliest_tradable_at` |
| 证据与局限 | `original_result` / `cost_assumption` / `replication_status` / `failure_conditions` / `contrary_evidence` |
| 本地状态 | `state`（unverified / data_insufficient / replicated / falsified） |

其中 **`knowledge_available_at` 是必填的 PIT 元知识**：它记录「该知识何时可得」，不是「何时检索到」。用 2026 年发表的论文指导 2020–2025 回测，只能称「今天做的历史研究」，不能声称是当年可得知识下的前瞻结果。

命令行：`init` / `add --file` / `list --state` / `promote --id` / `verify`。`promote` 内置查重（`factor_name` 或 `formula_pandas` 重复即跳过），保证候选池不被污染。

### 3.4 搜索层

| 模块 | 职责 |
|---|---|
| `search.py::generate` | 统一生成入口，按 `arm` 分派 |
| `ga_ops.py` | **有界 GA**：`refine_ga`（六类变异）+ `combine_ga`（多算子 + 子树交叉） |
| `edits.py` | 六类编辑原语：window / field / operator / wrap / unwrap / binary |
| `dsl.py` | AST 编译与校验（`max_depth` / `max_nodes` / unit 一致性） |
| `calibration.py` | 随机零假设校准（`random_population`） |
| `models.py` | 可选模型算法（ridge 等），与因子评估解耦 |

四个臂的语义：

| 臂 | 行为 |
|---|---|
| `fixed` | 固定模板基线（可 `shuffle_source`） |
| `random_ast` | 完整合法空间的随机探索 + 零假设校准 |
| `refine` | 对 parents 做局部变异（GA 时为六类有界变异） |
| `combine` | 组合两个 parents（GA 时为多算子 + 子树交叉） |

**基因算子开关**：`FUXI_V1_GA_OPS`（默认 `on`，`off` 可回退旧单叶变异/固定 `add_rank`）。

### 3.5 评估层（E 道）

| 闸门 | 模块 | 判定内容 |
|---|---|---|
| **E0** | `research_evaluation.py` | 数据与 PIT 契约是否支持该候选 |
| **E1** | `research_evaluation.py` + `research_protocol.py::screen` | 方向性、年度稳定性、**沪深双所均正 IC**、`alpha/max_trials` 的 block bootstrap 区间下界 > 0 |
| **E2** | `research_evaluation.py` | 独立性：配对截面相关 + 风格归因 + 单字段子通道消融 |
| **E3** | `research_evaluation.py::paired_accounts` | 经济增量：配对账户净收益差，含成本与容量压力 |
| **E4** | `research_evaluation.py` | 稳健性（成本倍数 / 参与率倍数压力） |
| **E5** | `forward_archive.py` | 前瞻证据（JQ 回传后的预注册 look） |

关键机制——**E1 的统计门槛随试验数变化**：

```
t_hurdle = z(1 - α / (2 × max_trials))
min_detectable_ICIR = t_hurdle / sqrt(n_sections)
```

研究窗 1125 截面时（`sqrt(n)=33.54`）：

| max_trials | t 门槛 | 最低可检测 ICIR |
|---|---|---|
| 12 | 2.87 | 0.085 |
| 60 | 3.34 | 0.100 |
| **80** | **3.42** | **0.102** |
| 120 | 3.53 | 0.105 |
| 240 | 3.71 | 0.111 |

> 注意：`screen_policy.py` 里的 `t_hurdle_ladder`（100→2.0 / 300→2.5 / 1000→3.0）**不参与 E1**，它标注为 confirmatory-only 且未冻结。

### 3.6 证据层

| 模块 | 职责 |
|---|---|
| `evidence.py` | 事务性 facts 存储（SQLite），预算、尝试预留、反馈、暴露隔离 |
| `governance.py` | 写者权威：文件锁 + 真实进程租约校验（防并发写） |
| `monitoring.py` | Pareto archive 维护、`select_parents`、多样性度量 |
| `learning.py` | 经验记忆与 motif 学习 |
| `shared_feedback.py` | 跨 campaign 反馈复用 |
| `attribution.py` | 风格归因（size / lowvol / reversal 等） |

`evidence.sqlite` 表结构：

| 表 | 用途 |
|---|---|
| `facts` | 所有事实（kind + payload），含 candidate / candidate_result / archive_member / round_completed 等 |
| `campaigns` | campaign 级配置与状态 |
| `attempts` | 候选评估的尝试预留（防重复计费） |
| `rewards` | 奖励账本（slot + revision） |
| `consumers` | 依赖消费者（决定事实是否仍被引用） |
| `exposure` / `protection` | 数据暴露隔离与冻结保护 |
| `receipts` | 交付回执 |

### 3.7 出口层

| 模块 | 职责 |
|---|---|
| `export.py` | 导出为可执行策略规格 |
| `jq_strategy.py` | 生成自包含的 JQ 策略 bundle |
| `forward_archive.py` | 前瞻决策归档（实际决策与原生账户配对） |
| `certification.py` / `certification_columnar.py` | 独立数据认证（generator ≠ reviewer 的实质独立性） |
| `lineage.py` / `factor_cards.py` | 因子血缘与卡片 |

---

## 4. 一次 Campaign 的完整生命周期

### 4.1 请求

```json
{
  "schema_version": "fuxi.unified_run.1",
  "input": "<snapshot path>",
  "input_sha256": "<hash>",
  "rounds": 20,
  "candidates_per_round": 4,
  "seed": 77777,
  "mode": "adaptive",
  "scheduler_policy": "legacy",
  "include_catalog": true,
  "purpose": "GA operator A/B",
  "llm_enabled": false,
  "limits": {
    "max_rounds": 40,
    "max_candidates": 120,
    "max_worker_seconds": 120,
    "max_total_seconds": 14400,
    "max_memory_mb": 8192
  }
}
```

### 4.2 轮内循环

```
for round_index in range(rounds):
    1. 刷新导入依赖与反馈历史
    2. choose_arm()            # UCB + diversity budget
    3. select_parents(archive) # Pareto 前沿选父
    4. generate(arm=... )      # 搜索层产出 proposals
    5. 对每个 proposal：
         - experiment digest 缓存命中？→ 复用
         - 否则 reserve attempt → spawn worker（可并行）
    6. settle_candidate()      # 按 proposals 顺序确定性归并
    7. 写 round_completed 事实
```

**确定性硬约束**：并行结果必须按 `candidate_id`（digest 排序）归并，禁止按 worker 完成先后写入，否则学习轨迹会分叉。

### 4.3 产物

| 文件 | 内容 |
|---|---|
| `RUN_PROTOCOL.json` | 代码 hash、effective_config、request、limits、identity |
| `PHASE_RESEARCH.json` | 研究阶段报告（candidate_results、rounds、archive） |
| `COMPLETE.json` | 完成回执 |
| `RECEIPT.json` | 指向 report_path 的交付回执 |
| `evidence.sqlite` | 全部事实 |
| `attempts/attempt_*/` | 每个尝试的 request / result / process_guard |

---

## 5. 关键数据契约

| Schema | 用途 |
|---|---|
| `fuxi.campaign_input.1` | campaign 输入契约 |
| `fuxi.snapshot.2` | native 快照格式 |
| `fuxi.columnar_snapshot.1` | 列式快照格式 |
| `fuxi.campaign.1` | campaign 配置 |
| `fuxi.knowledge_card.1` | 结构化知识卡（外部知识注入） |
| `fuxi.unified_run.1` | 单次工作流请求 |
| `fuxi.calibration_request.1` | 零假设校准请求 |
| `fuxi.account_run.1` | 账户模拟结果 |
| `fuxi.cost_spec.1` / `fuxi.cost_proxy_map.1` | 成本规格与代理映射 |
| `fuxi.data_certification.1` | 独立数据认证 |
| `fuxi.dedup_ledger.1` | 去重账本 |

---

## 6. 配置与开关

### 6.1 搜索与执行

| 变量 | 默认 | 作用 |
|---|---|---|
| `FUXI_V1_GA_OPS` | `on` | refine/combine 走有界 GA；`off` 回退单叶变异 |
| `FUXI_V1_PARALLEL_CANDIDATES` | `off` | 轮内候选并行评估（M1） |
| `FUXI_V1_PERSISTENT_POOL` | `off` | 持久 worker 池（M2），复用快照缓存 |
| `FUXI_V1_POOL_SIZE` | `2` | 池大小（推荐 4） |
| `FUXI_V1_POOL_MAX_REQUESTS` | `8` | 每 slot 处理多少请求后 recycle（推荐 40） |

### 6.2 预算常量（代码内）

`fuxi_v1/ga_ops.py`：

```python
GA_POPULATION_PER_GENERATION = 4      # 每代新个体数
GA_MIN_GENERATIONS = 10               # GA 实际迭代代数下限
GA_TRIALS_BUDGET = (60, 120)          # max_trials 目标区间
```

推荐配置：`rounds=20`、`candidates_per_round=4` → `max_trials=80`、t 门槛 3.42、GA 代数 ≈10。

---

## 7. 性能基准

实测环境：32 逻辑核 / 34 GB 内存，native64 快照（1212 sessions × 64 instruments，249 MB）。

| 模式 | 单候选耗时 | 说明 |
|---|---|---|
| 单次 spawn | 63.7 s | 每次重解析 249 MB 快照 |
| loop cold | 42.5 s | 免进程 spawn，首次仍加载快照 |
| loop warm | **31.2 s** | 命中快照缓存（1.36x） |
| 4 worker 并行 | 墙钟 64.7 s（串行求和 248.4 s） | **3.84x**，峰值 RSS 5.27 GB |

组合效果：**20 轮 × 4 候选**单臂约 25–30 分钟（串行无池约 83 分钟）。

---

## 8. 目录结构

```
fuxi_v1/
├── 入口      unified_runner.py  __main__.py
├── 数据      datasets.py  historical_adapter.py  history.py
│             cohort_input.py  pit_universe.py  columnar.py
│             full_market_input.py  data.py
├── 研究      archetypes.py  campaign.py  scheduler.py  arms.py
├── 搜索      search.py  ga_ops.py  edits.py  dsl.py  calibration.py
├── 评估      research_evaluation.py  research_protocol.py
│             evaluation.py  simulation.py  portfolio.py  statistics.py
├── 证据      evidence.py  governance.py  monitoring.py  learning.py
│             shared_feedback.py  attribution.py
├── 出口      export.py  jq_strategy.py  forward_archive.py
│             certification.py  lineage.py  factor_cards.py
└── 契约      contracts.py  routing.py  run_spec.py  effective_config.py

scripts/
├── cycle_orchestrator.py      # 日循环
├── rebuild_snapshot_1212.py   # 快照重建
├── ab_ga_ops.py               # GA A/B 前验
├── knowledge_card.py          # 知识卡：init/add/list/promote/verify
└── version_bump.py            # 版本账本

data/
├── raw/                        # 原始 CSV 通道
├── fuxi_unified/snapshots/     # 研究快照
├── fuxi_unified/runs/          # 运行产物
└── version_registry.json       # 工程版本账本
```

---

## 9. 核心工程纪律

1. **判据必须锚在实现/数据列/结构化字段**，不锚文件名。
2. **执行失败 ≠ 方向证伪**；数据不足 ≠ 因子无效。
3. **只判所测，不推全族**；S5=0 先查覆盖层再断经济失败。
4. **抽样必须 hash，不得字符串排序**（防单交易所垄断 + 幸存者偏差）。
5. **并行必须确定性归并**（按 candidate_id 排序，不按完成先后）。
6. **翻默认前必须查全部调用方**：凡依赖「未设置=旧默认」的显式传递点，都要改成显式写值。
7. **新机制先影子，再翻默认**；影子期只写 facts，不改主决策路径。

---

## 10. 版本与变更

| 版本 | 日期 | 要点 |
|---|---|---|
| **v2.0** | 2026-10-18 | 产品定版：快照 hash 抽样修复、GA 默认 on、M1/M2 提速 3.2x、E1 通过率 22–25% |

详细工程批次见 `CHANGELOG.md` 与 `data/version_registry.json`。

---

*本文档只描述 v2.0 的研究主链路（E 道）。落地影子层（G 道：G0 合法性预检 / G1 诊断路由 / G2 账户席位）处于 P5 shadow 验证期，不参与决策，待校准后另行成文。*
