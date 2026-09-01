# 更新日志 (Changelog)

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格。
版本号 `v0.7.x` 对应内部 v0.7 双通道架构的补丁序列。

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
