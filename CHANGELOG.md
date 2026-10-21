# 更新日志 (Changelog)

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格。
版本号自 v2.0.0 起与产品名对齐；`v0.7.x` 段为旧伏羲的工程批次账本，两者并行记录。

## [2.2.0] - 2026-10-21

### 版本记录（工程批次账本同步）
- 工程批次账本 `data/version_registry.json` 同步登记 **v0.7.219（2026-10-21，2 提案）**：
  `P-20261021-001` 量纲语法系统 types_core 落地、`P-20261021-002` GFlowNet 生产臂（第 6 臂）入 UCB。
  源树 `CHANGELOG.md`（`scripts/version_bump.py --render` 生成）已重渲染至 **220 个版本 / 375 条已落地提案全部编号**。
  发布仓 README 工程账本引用同步至 v0.7.219。

### Added
- **量纲语法系统 `types_core/`**（2026-10-21 引入，量纲 + 语义分层类型）：
  `spec.py`（TypeSpec + 量纲指数代数 + UNIT_DIMENSIONS 单点）、`registry.py`（字段级 10 + 因子级 97 显式注册 +
  名称模式推断 + resolve 四级解析）、`rules.py`（算子规则三档 hard/soft/open + 违规码 + render_for_prompt）、
  `infer.py`（双前端推断 + TypeState 增量状态 + mask_actions 动作掩码）。纯 stdlib，零 fuxi_v1 内部依赖，
  与 factor_mining 两侧复用同一规则表。接线点 W1-W4 均带独立 env 开关（默认 off = 逐位不变）：
  `FUXI_TYPE_GATE` / `FUXI_TYPE_GROUPS` / `FUXI_TYPE_MASK` / `FUXI_TYPE_NOTES` / `FUXI_TYPE_SOFT_PENALTY` /
  `FUXI_TYPES_OVERRIDE`。
- **GFlowNet 生产臂（第 6 臂）**：`gflownet_arm.py` + `gflownet_policy.py`，把 factor_mining 沙箱里的
  GFlowNet 提升为生产 fuxi_v1 的一臂，输出生产 DSL 空间的 AST（叶子 = `{"field": name}`、算子 = 生产
  CATALOG 非时序算子），走与 fixed/refine/combine 相同的 `search.factor_spec → compile_ast → E0-E5` 链路。
  采样侧用 TypeState + mask_actions（同一 rules 规则表），产出侧走 compile_ast 的 UNITS 量纲 +
  FUXI_TYPE_GATE 兜底。UCB 接线：scheduler.choose（selection_v2）+ campaign.choose_arm（legacy）+
  campaign study_protocol + landing_shadow.PRODUCTION_ARM_TO_V2（映射到 E_explore）。
- **旁挂方法库 `factor_mining/`**（2026-10-20 引入）：两目标 NSGA-II 组合层（nsga2_biobjective）+
  GFlowNet 因子表达式生成器（gflownet/）+ E 道移植复评（e_channel）+ JQ 回测代码（jq_backtests/）。
  独立于 fuxi_v1 治理链，可被每日自动化 `import` 或 `python -m` 调用。

### Changed
- **预算提速**：`DEFAULT_WORKER_TIMEOUT_SECONDS` 420→600s（LLM 臂单候选 ~410-440s，600s 留 160-190s 余量）；
  `DEFAULT_LIMITS.max_worker_seconds` 同步 420→600，`max_total_seconds` 36000→50000。
- **L2 prompt 规则同源**：`search.llm_candidates` 的手写 rules 字符串改为 `render_for_prompt()` 拼接
  （消灭 C3 手工副本漂移）。
- **G6 注册携带 output_type**：`acceptance.research` 的 body 在 FUXI_TYPE_NOTES=on 且 output_type 非 None 时
  加 output_type 字段（off 时 body 逐位不变，A/A 安全）。
- **G3 family 冗余诊断**：`research_evaluation.pool_redundancy` 加 candidate_ast 参数 + `_semantic_redundancy_note()`
  （FUXI_TYPE_NOTES=on 时产出，同 family + 相关>0.7 标注 redundancy_semantic，仅诊断不升门控）。
- **W2 GA 变异升 type_groups**：`ga_ops.type_groups_enabled()` + `_group_key`（FUXI_TYPE_GROUPS=on 时按
  TypeSpec (dim,form) 分组，off 时按 unit 分组，原行为逐位不变）。

### Fixed
- `test_generator_bounds.test_valid_deep_parents_reject_oversized_child_and_fall_back` 断言改为
  `counters['compiler_rejected']>=1`（GA 路径 combine_ga 写 counters，非 legacy combine 的 diagnostics 死代码）。
- `landing_shadow.PRODUCTION_ARM_TO_V2` 补 `random_ast → E_explore`（legacy 默认路径臂序用 random_ast，
  但原映射只有 random，导致 P5 shadow 冷启动 round 1 每回合落 landing_shadow_error）。

## [2.1.0] - 2026-10-20

### Added
- **系统慢循环（慢进化）落地**：`scripts/system_evolve.py`（CLI：observe/draft/freeze/assess/history/promote）
  与 `research/system_evolution/`（`observe.py` 观测沉淀 + 规则提炼、`scorecard.py` 转正评分卡
  P 门 7 条 + S 门分类别、`gate.py` 冻结/评估扩展 `case_deltas`）。转正严格锁死人工，
  其余观测/提炼/挑战者开关/freeze-assess/评分卡全自动。
- **因子评价 E1 扩展**：`monotonicity.py`（单调性诊断，Patton & Timmermann (2010, JFE) MR test 口径，
  相邻分组收益差 block bootstrap 单边 p 值，note-only 不设绝对门槛）、`applicability.py`
  （适用性备注：行业/市场状态/市值分段的条件化 Rank IC）、`regime.py`（沪深300 MA200 牛熊划分）、
  `cap_data.py`（tushare 历史流通市值序列，代码转换 XSHE/XSHG ↔ SZ/SH）。
- **组合层风险监控**：`exposure_monitor.py`（行业暴露 + 漂移 + 市值分段）、`factor_family.py`（因子族类归因）。
- **工程裁决**：`final_admission.py`（E2 池内去冗余，IC 降序贪心入池）、`benchmarks.py`
  （E3 双基准 CSI300/CSI500，配对增量与绝对超额分开报告）、`budgets.py`（worker 超时 420s、
  预算单元、2% 容差 overrun 判定）。

### Changed
- **单调性门降级为诊断备注**：移除 `spearman≥0.8` 绝对门槛（A 股单因子日频不可达，
  全量 28 个 JQ pass 因子实测 0 通过）；改为 spread 不显著 → `data_insufficient` fail-open，
  不再翻转 E1 裁决，结果并入 `factor_notes` 与 regime/market_cap/industry 并列。
- **E3 参照系**：新增沪深300/中证500 双基准，绝对收益≤0 时须双基准超额均>0（仅 certified 快照启用）。
- **预算提速**：worker 超时 300s→420s（含 2% 容差），编排器默认并行 M1+M2。

### Fixed
- `residual_rank_ic` / `paired_parent` 的 dict 分支 `clock` 类型错误（datetime vs str 比较）。
- `gen_15_jq_standalone.py` 科创板 688/689 市价单需保护限价（改限价单 + 保护价缓冲）。

## [2.0.0] - 2026-10-18

### Changed
- **版本号与产品名对齐**：`fuxi_v1/__init__.py::__version__` 由 `1.0.0-rc12` 定为 **`2.0.0`**，
  新增根目录 `VERSION` 作为唯一对外版本声明。
- **主链路收敛到 `fuxi_v1` kernel**：数据 → 研究（L0–L3）→ 搜索 → 评估（E0–E5）→ 证据 → 出口。
- **文档重写**：`README.md` / `README_EN.md` / `ARCHITECTURE.md` 全部按 v2.0 实况重写，
  移除与实现不符的历史陈述。

### Fixed
- **快照抽样偏差**：`sorted(codes)[:N]` 会让 `000*.SZ` 垄断样本；改为按 `hash(salt, ts_code)`
  抽样并断言沪深双所均有覆盖，`--legacy-code-order` 仅用于显式复现旧有偏样本。
- **并行归并确定性**：并行候选结果按 `candidate_id`（digest 排序）归并，禁止按完成先后写入。
- **E2 独立性门失效**：原实现把 E2 判定直接镜像 E3，相关性只记录不裁决；改为池内去冗余门
  （高相关时做正交残差 IC，无残差 alpha 才拒绝）。
- **因子库坍缩**：combine/refine 引入跨族配对约束，模板池由 62 扩至 76 个机制。

### Added
- `ARCHITECTURE.md`：v2.0 分层架构、模块职责、数据契约、E1 自适应门槛、性能基准。
- `.env.example` / `VERSION` / `requirements-env-default.txt`。
- 知识卡通道（`scripts/knowledge_card.py`，schema `fuxi.knowledge_card.1`，append-only），
  外部知识只能经知识卡进入系统，且必须记录 `knowledge_available_at`。

### Security
- `.gitignore` 补齐 `.env.bt` / `.env.*`（保留 `.env.example`）/ `.workbuddy/` / `data/` / `output/` / `*.sqlite`。
- 文档与示例中的凭据统一占位化（`YOUR_*` / `<DEEPSEEK_API_KEY>`）。

## [0.7.2] - 2026-09-01

### Fixed
- **LLM 周频通道 NameError**：`freq` 参数未透传给 LLM 生成器，weekly 候选生成路径直接崩溃（P-20260830-001）
- `lane_calibration` 复评接口返回结构完善（eval 失败安全返回 None）

### Added
- **维度预剪枝**（`forge/dimension_rules.py`，P-20260831）：量纲审计 6 类规则，三态开关 `off / shadow / enforce`（默认 shadow 只计数不裁决；`FUXI_DIM_PRUNE=enforce` 启用硬剪枝）
- **多样性折扣**（`forge/diversity_discount.py`，P-20260831 P1）：GP 子代按父代 MaxCorr 打折，缓解生成趋同（权重默认 0 = 生产零变化）
- **top-k 双口径 ICIR 影子**（P-20260831 P2）：集中度比值作为"尖峰噪声探针"，区分"少数尖峰周撑起的 ICIR"与"各周均匀贡献的稳健 ICIR"
- **父因子上下文层**（P-20260901-005，影子）：生成时记录父因子谱系，只记录不裁决
- **breed fail-streak 跨进程落盘**：修复连败计数每轮归零的问题，退避策略真正生效

### Changed
- `weekly_lane` 周频裁决器输出新增 `topk_shadow` 字段（影子，不影响判定）

## [0.7.1] - 2026-08-31

### Fixed
- **双通道周频 trade_date 解析崩溃**：日期带 `.0` 后缀 + `format="mixed"` 显式声明（P-20260830 系列）
- `s5_joint_filter` / `factor_ic_computer` 小修

### Added
- **JQ 单因子 IC 快验批量门**（`scripts/gen_jq_fast_ic.py`，影子模式）：一次 JQ 回测批量计算 N 因子周频 rank IC/ICIR，作为组合回测前的第二道证据门（`FAST_IC_GATE_ENFORCE=False` 默认只记录）
- **P-007 子结构 fail-prior 影子**：`compute_fail_prior_from_library()` 按历史失败率对高频子结构加权

## [0.7.0] - 2026-08-31

### Added
- 首次公开发布（Initial public release）
- 完整自进化流水线：种子注入 → MAB 引擎调度 → R 记忆检索 → G 三引擎生成（GP / LLM / Forge）→ E 多级评估（S1-S5）→ MMR 去冗余 → JQ 聚宽回测裁决 → D 三层知识蒸馏 → D+ 经验回流
- v0.7 双通道：S1 分频 XOR 路由（周频裁决器 ICIR ≥ 0.15 / 日频裁决器 fwd5）
- v0.6 八大防护模块（行为同质性 / Holdout 边界 / 多重检验 DSR·PBO·FDR / 预算账本 / 污染账本 / 方向微战役 / 增量边际 / 策略规格）
- MIT License，双语文档（README.md / README_EN.md）
