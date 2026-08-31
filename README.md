# 伏羲 (Fuxi) — A股因子自进化挖掘系统

**一个会自己发现 alpha 的流水线。** 伏羲以 109+ 种子因子模板为起点，通过
「生成 → 多级评估 → 组合筛选 → 真回测验证 → 知识蒸馏 → 反馈再进化」的闭环，
**持续自主挖掘新的量化因子**，无需人工干预即可把失败经验蒸馏为约束、把成功模式固化为新种子。

> 版本: v0.7 (频率对称双通道) · 语言: Python 3.10+ · 回测平台: 聚宽 (JoinQuant)
> 中文说明为主文档，英文见 [README_EN.md](README_EN.md)。

---

## 为什么叫"自进化"

伏羲不是一个固定因子的回测工具，而是一个**在因子空间中自主搜索、自我纠错、持续积累经验**的进化系统：

| 能力 | 实现机制 |
|------|---------|
| **自主探索** | 三大生成引擎并行搜索: GP 遗传编程(60%) / LLM 公式生成 / Forge 范式生成器 |
| **调度学习** | MAB 多臂老虎机根据各引擎历史产出动态分配算力，探索-利用自动平衡 |
| **多级自然选择** | S1~S5 五级评估(IC/ICIR → 稳健性 → 拥挤度门控 → 多信号联合筛选)，逐级淘汰 |
| **经验记忆** | 三层 Experience Memory (因子经验 / 范式状态 / 轨迹池)，历史教训自动注入生成 prompt |
| **知识蒸馏** | 每次真回测后自动把结论蒸馏为"禁飞区"约束与新种子模板 |
| **反馈闭环** | 聚宽回测结果自动写回记忆与轨迹 (D+ 阶段)，形成完整的进化闭环 |

每一轮进化都会让下一代因子**更聪明地绕开已证伪的坑**（如回测污染、拥挤 alpha、过度拟合模式），而不是在原地重复随机试错。

---

## 系统架构

```
                    ┌──────────────────────────────────────────────┐
                    │              伏羲自进化主循环 (Ralph Loop)         │
                    └──────────────────────────────────────────────┘
  种子因子库 ──► MAB 引擎调度 ──► R: Experience Memory 检索 ──► G: 生成引擎
 (109+ 模板)    (探索/利用平衡)   (三层记忆蒸馏)            ├─ gp_breed 遗传编程 (60%)
                                                             ├─ LLM 公式生成
                                                             └─ Forge 范式生成器
        ▲                                                            │
        │                                                            ▼
   D+: JQ 反馈注入                                         E: 多级评估 S1~S5
 (结果写回记忆与轨迹)                          (IC/ICIR → 稳健性 → 拥挤度 → 合成)
        ▲                                                            │
        │                                                            ▼
   D: 三层知识蒸馏 ◄── JQ 真回测验证 ◄── MMR 组合筛选 ◄── S5 通过
```

### 核心设计原则

1. **JQ 是唯一真相源** — 本地 IC 只做否决、不做排序，真回测说了算；
2. **进化 ≠ 优化** — 探索新因子空间优先于在旧空间内调参；
3. **LLM 只生成公式，不筛选因子** — 统计筛选与组合优化由规则引擎负责；
4. **领域配对优于统计筛选** — 因子必须绑定投资逻辑与行为金融含义；
5. **单因子 IC/ICIR 强制披露** — 每个 JQ 回测必须输出逐因子 Rank IC/ICIR。

### 流水线各阶段

| 阶段 | 模块 | 职责 |
|------|------|------|
| 种子注入 | `seed_injector.py` | 注入 109+ 种子因子模板, 保持库多样性 |
| MAB 调度 | `factor_model_cooptim.py` | 多臂老虎机在三大生成引擎间调度资源 |
| 记忆检索 | `experience_memory.py` | 三层记忆: 因子经验 / 范式状态 / 轨迹池 |
| 生成 | `gp_breed`(ga/) / `llm_generator.py` / `forge/` | 三大生成引擎并行探索 |
| 评估 S1~S5 | `factor_ic_computer.py`, `s5_joint_filter.py`, `multi_stage_validator.py` | IC/ICIR → 稳健性 → 拥挤度 → 多信号联合筛选 |
| 组合 | `mmr_selector.py` | 最大边际相关组合选择, 去冗余 |
| JQ 验证 | `jq_generator/` | S5 通过后自动生成聚宽回测代码 |
| 知识蒸馏 | `experience_memory.py`, `trajectory_pool.py` | 回测结果蒸馏进因子库与禁飞区 |
| 反馈闭环 | `trigger_d_plus.py`, `trajectory_logger.py` | JQ 结果自动写回记忆与轨迹 |

### v0.6 实验防护模块 (默认关闭, `config.py` 开关控制)

`holdout_boundary` / `behavior_homogeneity` / `multiple_testing_guard`(DSR·PBO·BH-FDR) /
`direction_campaign` / `budget_ledger` / `contamination_ledger` / `strategy_spec`

---

## 自进化产出示例

系统在无人干预的迭代中自主发现并验证过的部分因子（聚宽历史回测，仅作能力演示，**不代表未来表现**）：

| 因子 | 发现引擎 | 因子内涵 | 回测区间收益 |
|------|---------|---------|------------|
| 002 价量关系 | GP 遗传编程 | 价量背离的均值回复 | +143.5% |
| 尾部风险 (ts_std 复合) | Forge 范式生成 | 波动尾部暴露惩罚 | +91.9% |
| 断板回调卫星 v2 | LLM + 拥挤度门控 | 连板断板后的回调动量 | +91.0% |
| 复合组合 (AlphaAgent v3) | 多因子 MMR 组合 | 跨范式去冗余合成 | +198.2% |

---

## 目录结构

```
fuxi/
├── README.md / README_EN.md
├── LICENSE                  # MIT
├── .env.example             # 凭据模板 (复制为 .env 后填写)
├── credentials.py           # 统一凭据加载 (Tushare / 聚宽 / WorldQuant)
├── factor_alchemy/          # 核心系统
│   ├── run_v4_pipeline.py   # v4 管道统一入口 (--ralph 启动自进化)
│   ├── ralph_loop.py        # 自进化主循环
│   ├── config.py            # 全系统配置与实验开关
│   ├── forge/               # Forge 范式生成引擎
│   ├── ga/                  # GP 遗传编程引擎 (NSGA-II)
│   ├── evaluation/          # 单因子评估 (IC/分位/相关性/稳健性)
│   ├── evaluator/           # 影子评估器
│   ├── factors/             # 因子库基础实现
│   ├── portfolio/           # 组合模拟器
│   ├── jq_generator/        # 聚宽回测代码生成器
│   └── ...
├── scripts/                 # 配套流水线脚本 (提案/采集/看板/拥挤度监控)
└── docs/                    # 补充文档
```

---

## 快速开始

### 1. 环境准备

```bash
pip install numpy pandas scipy tushare jqdatasdk
```

### 2. 配置凭据

```bash
cp .env.example .env
# 编辑 .env 填入:
#   [tushare]   token      — Tushare Pro token
#   [joinquant] username/password — 聚宽 JQData 账号
#   [worldquant] email/password   — (可选) WorldQuant BRAIN 凭据
```

LLM 生成引擎需要 DeepSeek API Key (通过环境变量注入):

```bash
export DEEPSEEK_API_KEY="sk-..."
```

### 3. 目录约定

- 行情/因子数据默认存放在 `data/` (可用环境变量 `FUXI_DATA_DIR` 重定向);
- 输出默认在 `output/` (可用 `FUXI_OUTPUT_DIR` 重定向);
- 可选: 设置 `RAG_DIR` 指向本地研究知识库, 自动注入生成 prompt。

### 4. 运行

```bash
cd factor_alchemy

# 系统状态快照
python run_v4_pipeline.py --view

# 全量诊断报告
python run_v4_pipeline.py --diagnose

# 启动自进化主循环 (Ralph Loop)
python run_v4_pipeline.py --ralph
```

---

## 安全说明

- 本项目**不包含任何账户凭据**; 所有密钥通过 `.env` / 环境变量注入;
- `.env` 已在 `.gitignore` 中, 请勿提交。

## 免责声明

本项目仅用于量化研究学习。文中回测数据均为历史模拟结果，**不构成任何投资建议**；
历史收益不代表未来表现。A 股市场有风险, 投资需谨慎。
