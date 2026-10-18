# 伏羲 v2.0 · A股因子研究闭环系统

> **产品版本**：**v2.0.0**（与产品名对齐，2026-10-18 定版）｜ **内核版本**：2.0.0（`fuxi_v1/__init__.py::__version__`）
> **工程账本**：v0.7.218（`data/version_registry.json`，219 个批次条目）
> **语言**：Python 3.10+ ｜**回测平台**：聚宽 JoinQuant（人工上传 / 原始结果回传）
> 英文版见 [README_EN.md](README_EN.md) ｜架构细节见 [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 为什么值得一试

一句话：**把"人肉调因子"变成"可复现、可审计、会积累经验"的自动搜索——而且不吹收益。**

| 做因子研究最常踩的坑 | 伏羲怎么处理 | 可对照的实现 |
|---|---|---|
| 换个 seed 结果就变，没法复现 | 同 seed + 同快照 + 同配置复现同一轨迹；并行结果按 `candidate_id` 归并，不按完成先后 | `unified_runner.py` / `campaign.py` |
| 回测漂亮，上实盘就废 | E0–E4 分级证据：契约 → 统计 → 独立性 → 经济增量 → 压力；JQ 回测是唯一真相源 | `research_evaluation.py` |
| 越挖越像，池子被同一族因子占满 | 跨族配对约束 + E2 池内去冗余（高相关必须拿出正交残差 alpha 才放行） | `ga_ops.py` / `research_evaluation.py` |
| 试了 200 次挑出最好的，显著性虚高 | E1 门槛随试验数自动收紧：`t = z(1 − α/(2·max_trials))` | `research_protocol.py` |
| 拿 2026 年的论文指导 2020 年回测 | 知识卡强制记录 `knowledge_available_at`，区分"知识何时可得"与"我何时检索到" | `scripts/knowledge_card.py` |
| 抽样悄悄偏了，回测却很好看 | hash 抽样 + 沪深双所覆盖断言，禁止 `sorted(codes)[:N]` | `datasets.py` |
| 多进程写状态互相覆盖 | 写者权威 + 真实进程租约锁 | `governance.py` |

**性能不是口号**（32 核 / 34 GB，1212 会话 × 64 标的快照）：

- 单候选 63.7 s → 热循环 **31.2 s**（1.36x）
- 4 worker 并行 **3.84x**（墙钟 64.7 s vs 串行 248.4 s）
- 20 轮 × 4 候选：串行 83 分钟 → **25–30 分钟**

**它明确不做什么**（这恰恰是它能被信任的原因）：不自动准入候选、不碰实盘、不修改判官、不发布策略。
`live_execution_allowed` / `QMT_orders_enabled` 默认 `False`，跨线必须由账户所有者人工签署。

**上手只要三条命令**：

```bash
pip install -r requirements-env-default.txt
python -X utf8 -m fuxi_v1 system-fixture --output reports/_smoke/synthetic.json
python -X utf8 -m fuxi_v1 campaign --input reports/_smoke/synthetic.json \
  --output reports/_smoke/campaigns --rounds 6 --candidates 2
```

> 本项目**不承诺任何收益率**。README 里出现的每个数字，都能在仓库中找到对应实现或实测记录；
> 历史回测不代表未来表现。

---

## 0. 30 秒速览

伏羲是一套 **A股量化因子研究闭环**：从假设生成、表达式搜索、分级证据，到 JQ 回测与前瞻归档，
全部由「可复现的声明 + 事务性证据」驱动，而不是靠人工叙述或文件名判定成败。

三条不可让渡的原则：

1. **证据驱动** —— 结论必须锚定实现、数据列或结构化字段；
2. **不做无证据的裁决** —— 执行失败 ≠ 方向证伪，数据不足 ≠ 因子无效；
3. **确定性优先** —— 同一 seed、同一快照、同一配置，必须复现同一条学习轨迹。

---

## 1. 版本坐标（三者并存，互不冲突）

| 坐标系 | 含义 | 载体 |
|---|---|---|
| **v2.0.0** | **产品版本 = 内核版本**（已对齐，2026-10-18） | `fuxi_v1/__init__.py` · `VERSION` |
| v2.0 | 产品定版架构：本文档描述的完整系统 | `ARCHITECTURE.md` |
| v0.7.x | 工程迭代账本（逐批提案），当前 v0.7.218 | `data/version_registry.json` |

> 版本号已定为 **2.0.0**，与产品名 v2.0 对齐（`__version__` 由 `1.0.0-rc12` 直接跳到 `2.0.0`，无残留旧号）。
> 看到 `v0.7.218` 不等于系统停在 v0.7 架构：v0.7.x 是**工程批次编号**，不是产品版本。
> 两者并行记录，是本项目的既定版本纪律，不是版本回退。

---

## 2. 能力边界：它不做什么

`fuxi_v1.unified_runner.run()` 是唯一的单次工作流入口，它**只做一件事**：
在冻结的请求与预算内产出可审计的研究证据。它**不能**：

- 准入候选（admission 由 E 道证据决定）
- 提交 JQ 实盘（导出与执行严格分离）
- 修改判官或评估策略
- 发布策略

真实交易客户端不在本期范围；`live_execution_allowed` 与 `QMT_orders_enabled` 默认 `False`，
任何跨越都需要账户所有者**手工签署**，系统不会自动解除。

---

## 3. 分层架构

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

数据流单向：**数据 → 研究 → 搜索 → 评估 → 证据 → 出口**，没有任何一层能反向改写上游冻结产物。

---

## 4. 目录结构

```
e:\quant\
├── fuxi_v1/                  # v2.0 内核（研究主链路）
│   ├── unified_runner.py     # 单次有界工作流（source-keyed 幂等）
│   ├── __main__.py           # CLI 命令族
│   ├── campaign.py           # campaign 主循环 / L1-L2 调度
│   ├── search.py ga_ops.py dsl.py edits.py   # 搜索层
│   ├── research_evaluation.py research_protocol.py  # E 道评估
│   ├── evidence.py governance.py monitoring.py      # 证据层
│   ├── export.py jq_strategy.py forward_archive.py  # 出口层
│   └── tests/                # 回归测试
├── scripts/                  # 编排与运维脚本
│   ├── cycle_orchestrator.py # 日循环编排
│   ├── rebuild_snapshot_1212.py
│   ├── knowledge_card.py     # 知识卡：init/add/list/promote/verify
│   └── version_bump.py       # 版本账本登记
├── research/factor_alchemy/  # 旧 v0.7 因子炼金库（LLM 客户端等仍被复用）
├── data/                     # 原始数据与快照（体积大，不入库）
│   ├── raw/                  # 原始 CSV 通道
│   ├── fuxi_unified/snapshots/
│   └── version_registry.json # 工程版本账本
├── reports/                  # 面向决策的成果（日报 / 审计 / 裁决）
├── ARCHITECTURE.md           # v2.0 架构权威文档
├── AGENTS.md                 # 项目铁律与运行纪律
├── .env.example              # 凭据模板（本文件不含真实值）
└── credentials.py            # 统一凭据加载（Tushare / 聚宽 / WorldQuant）
```

---

## 5. 快速开始

### 5.1 环境

Python 3.10+。项目约定使用**既有解释器**，不新建环境（详见 `AGENTS.md`）。
Windows 下含中文输出的脚本建议加 `-X utf8`：

```bash
python -X utf8 -m fuxi_v1 --help
```

### 5.2 依赖

```bash
pip install -r requirements-env-default.txt
```

最小集（人工核对用）：`numpy pandas scipy tushare jqdatasdk`。

### 5.3 配置凭据

**通道一：数据平台（Tushare / 聚宽 / WorldQuant）** —— 由 `credentials.py` 读取 `.env`：

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

`.env` 的 section 名称不可修改：

```ini
[tushare]
token = YOUR_TUSHARE_TOKEN        # 占位符，填入你自己的 40 位 token

[joinquant]
username = YOUR_JQ_USERNAME
password = YOUR_JQ_PASSWORD

[worldquant]
email = YOUR_WQ_EMAIL             # 可选，不参与主链路
password = YOUR_WQ_PASSWORD
```

自检（输出自动脱敏，只显示前 3 + 后 2 位）：

```bash
python -X utf8 credentials.py
```

**通道二：LLM（DeepSeek）** —— 由 `research/factor_alchemy/llm_client.py::resolve_api_key_with_source` 按优先级解析：

| 优先级 | 来源 | 用法 |
|---|---|---|
| 1 | 显式传参 | 代码内 `explicit=` |
| 2 | 进程环境变量 | `DEEPSEEK_API_KEY=<DEEPSEEK_API_KEY>` |
| 3 | 本地凭据文件 | `.env` / `.env.local` / `config/secrets.local.json` |
| 4 | Windows 用户环境变量 | 读取注册表，绕过进程快照 |

```bash
# Linux / macOS
export DEEPSEEK_API_KEY="<DEEPSEEK_API_KEY>"
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "<DEEPSEEK_API_KEY>"
```

未配置 Key 时 LLM 臂自动降级，不影响 GP / Forge 主链路。

### 5.4 准备数据

从本地原始 CSV 构建快照（**必须用 hash 抽样**，禁止字符串排序抽样）：

```bash
python -X utf8 -m fuxi_v1 snapshot-local \
  --raw-root data/raw --start 2024-01-01 --end 2026-09-30 \
  --max-instruments 64 --output data/fuxi_unified/snapshots/mysnap
```

> v2.0 关键修复：旧版 `sorted(codes)[:N]` 会让 `000*.SZ` 垄断样本。
> 现按 `hash(salt, ts_code)` 抽样并断言沪深双所均有覆盖。`--legacy-code-order`
> 仅用于**显式复现**旧的有偏诊断样本，生产路径不要使用。

### 5.5 跑通第一条链路

```bash
# 1) 合成夹具（不需要真实行情，先验证工程链路）
python -X utf8 -m fuxi_v1 system-fixture --output reports/_smoke/synthetic.json

# 2) 跑一个小 campaign（6 轮 × 2 候选）
python -X utf8 -m fuxi_v1 campaign \
  --input reports/_smoke/synthetic.json \
  --output reports/_smoke/campaigns \
  --rounds 6 --candidates 2 --mode adaptive

# 3) 校验产物
python -X utf8 -m fuxi_v1 verify reports/_smoke/campaigns
```

真实数据用 `--input <快照路径>`，并按需加 `--llm`（启用 LLM 臂）、
`--models ridge,lightgbm,xgboost`（空字符串关闭）、`--resume-id campaign_<ID>`（断点续跑）。

### 5.6 常用命令

| 命令 | 用途 |
|---|---|
| `python -m fuxi_v1 demo --output <dir>` | 最小演示链路 |
| `python -m fuxi_v1 system-fixture --output <json>` | 合成夹具 |
| `python -m fuxi_v1 campaign --input --output --rounds --candidates` | 主链路 |
| `python -m fuxi_v1 snapshot-local --raw-root --start --end --output` | 本地快照 |
| `python -m fuxi_v1 snapshot-history --plan --raw-root --supplement --output` | 历史 PIT 快照 |
| `python -m fuxi_v1 certify-data --input --apply` | 派生数据契约标志 |
| `python -m fuxi_v1 pit-cohort --raw-root --start --end --salt` | 声明 PIT cohort |
| `python -m fuxi_v1 forward-archive` | 前瞻决策归档 |
| `python -m fuxi_v1 operate` | 运维动作（受 governance 锁约束） |
| `python -m fuxi_v1 work-packages` / `blueprint` | 工作包与蓝图状态 |
| `python scripts/version_bump.py --bump` | 版本账本登记 |

---

## 6. 配置与开关

### 6.1 搜索与执行（`FUXI_V1_*`）

| 变量 | 默认 | 作用 |
|---|---|---|
| `FUXI_V1_GA_OPS` | `on` | refine/combine 走有界 GA；`off` 回退单叶变异 |
| `FUXI_V1_PARALLEL_CANDIDATES` | `off` | 轮内候选并行评估 |
| `FUXI_V1_PERSISTENT_POOL` | `off` | 持久 worker 池，复用快照缓存 |
| `FUXI_V1_POOL_SIZE` | `2` | 池大小（推荐 4） |
| `FUXI_V1_POOL_MAX_REQUESTS` | `8` | 每 slot 请求数上限后 recycle（推荐 40） |

### 6.2 预算常量（代码内，`fuxi_v1/ga_ops.py`）

```python
GA_POPULATION_PER_GENERATION = 4      # 每代新个体数
GA_MIN_GENERATIONS = 10               # GA 实际迭代代数下限
GA_TRIALS_BUDGET = (60, 120)          # max_trials 目标区间
```

推荐组合：`rounds=20`、`candidates_per_round=4` → `max_trials=80`、t 门槛 3.42、GA 代数约 10。

### 6.3 遗留 S5 通道

旧 v0.7 S5 通道仍由 `FUXI_*` 系列开关控制（`COST_BPS=15`、`MIN_COVERAGE=0.9`、
`TRADABLE_GATE=on`、`ADJ_PRICE=on`、`DATE_CEIL=auto` 等），完整清单与翻默认纪律见 `AGENTS.md`。
**v2.0 的研究主链路（E 道）不读取这些开关**，两者不要混为一谈。

---

## 7. 产物与契约

两个入口的产物不同，不要混：

**`campaign` 命令**（实际运行验证）：

```
<output>/
├── authority.sqlite          # 跨 campaign 的权威账本
└── campaign_<id>/
    ├── config.json input.json policy.json
    ├── code/                 # 冻结的源码副本（运行中改源码会阻断报告）
    ├── evidence.sqlite       # 该 campaign 的事务性事实
    ├── attempts/             # 每次尝试的 request / result / process_guard
    ├── models/ bundles/
    └── report_<hash>.json    # 每轮一行 JSON + 末尾汇总行
```

**`unified_runner.run()`**：

| 文件 | 内容 |
|---|---|
| `RUN_PROTOCOL.json` | 代码 hash、`effective_config`、请求、limits、identity |
| `PHASE_RESEARCH.json` | 研究阶段报告（candidate_results / rounds / archive） |
| `COMPLETE.json` | 完成回执 |
| `research/RECEIPT.json` | 指向 report_path 的交付回执 |
| `evidence.sqlite` | facts / campaigns / attempts / rewards / exposure / receipts |

主要 schema：`fuxi.campaign_input.1`、`fuxi.unified_run.1`、`fuxi.snapshot.2`、
`fuxi.columnar_snapshot.1`、`fuxi.knowledge_card.1`、`fuxi.cost_spec.1`、`fuxi.data_certification.1`。

---

## 8. 评估口径 E0–E5

| 闸门 | 判定内容 |
|---|---|
| E0 | 数据与 PIT 契约是否支持该候选 |
| E1 | 方向性、年度稳定性、**沪深双所均正 IC**、block bootstrap 区间下界 > 0 |
| E2 | 独立性：配对截面相关 + 风格归因 + 单字段子通道消融 |
| E3 | 经济增量：配对账户净收益差（含成本与容量压力） |
| E4 | 稳健性：成本倍数 / 参与率倍数压力 |
| E5 | 前瞻证据：JQ 回传后的预注册 look |

E1 的门槛随试验数自适应收紧：

```
t_hurdle = z(1 - α / (2 × max_trials))
min_detectable_ICIR = t_hurdle / sqrt(n_sections)
```

研究窗 1125 截面时，`max_trials` 80 对应 t=3.42、最低可检测 ICIR=0.102。

> **JQ 是唯一真相源**：本地 IC 只做否决、不做排序。每个 JQ 回测必须输出逐因子 Rank IC/ICIR。

---

## 9. 当前状态与已知边界

写文档最容易骗人的地方，所以这一节只写已经核对过的读数：

- **v2.0 已定版**（2026-10-18），研究主链路 E 道可用；工程批次已登记至 v0.7.218。
- **G 道（G0 合法性预检 / G1 诊断路由 / G2 账户席位）仍在影子验证期**，
  处于 P5 shadow，只写 facts、不参与任何决策，因此未纳入 `ARCHITECTURE.md` 主链路。
- **蓝图 §9 统计级验收仍在积累**（需 ≥100 轮影子 + 独立重复 + 在途项修复）。
  影子读数不等于经济结论，本文档不把它写成已通过。
- **不要相信旧文档的环境变量**：早期 README 声称的 `FUXI_DATA_DIR`、`FUXI_OUTPUT_DIR`、
  `RAG_DIR` 在当前 `fuxi_v1/` 与 `scripts/` 中**没有任何消费点**，已从本文档移除。
  路径一律走 CLI 的 `--output` / `--raw-root` / `--input` 显式传参。
- **成本口径**：账户重放仍用 15bps 代理；`CostSpec` 原生费率映射已在合同层建档，尚未切生产默认。
- **停复牌表**仅至 2026-04-30，公司行动与 PIT 标签仍有缺口，相关结论须显式披露。

---

## 10. 凭据与脱敏规范

本仓库**不含任何真实凭据**。文档、示例、日志中的敏感信息统一按下列方式占位：

| 类型 | 占位符 | 禁止写法 |
|---|---|---|
| Tushare token | `YOUR_TUSHARE_TOKEN` | 40 位十六进制实值 |
| 聚宽账号 / 密码 | `YOUR_JQ_USERNAME` / `YOUR_JQ_PASSWORD` | 手机号 / 明文密码 |
| WorldQuant | `YOUR_WQ_EMAIL` / `YOUR_WQ_PASSWORD` | 真实邮箱与密码 |
| DeepSeek | `<DEEPSEEK_API_KEY>` | `sk-` 开头的实值 |
| 账户 / 本金 | `<ACCOUNT_ID>` / `<INITIAL_CAPITAL>` | 券商账号、真实资金额 |

规则：

1. 真实值只存在于 `.env` / `.env.bt` / `config/secrets.local.json`，这些文件已在 `.gitignore` 中；
2. 提交前自检：`grep -rnE "sk-[A-Za-z0-9]{20,}" --include=*.md --include=*.py .`；
3. 记忆与日志中的历史凭据同样要脱敏——**历史事实可以保留，明文凭据不可以**；
4. 泄露过的 Key 视为已失效，应当在供应商侧轮换，而不是只从文件里删掉。

---

## 11. 免责声明

本项目仅用于量化研究学习。文中任何回测读数均为历史模拟结果，**不构成投资建议**；
历史收益不代表未来表现。A 股市场有风险，投资需谨慎。
