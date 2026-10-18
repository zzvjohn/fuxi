## 当前全A二十轮受控迭代（2026-10-09，v0.7.146）

已实际完成全A5670成员、20轮40候选，工程错误0；研究窗2024-05-17..2026-09-24，最新历史数据、非PIT保证，正式/JQ资格0。每日研究合格成员4484..5318，102窗口退市成员仍保留。453回归通过；生产S5默认与资格门槛不变。

[完整结果](E:/quant/reports/fuxi_full_a_20rounds_20261009/REPORT.md)。运行/恢复入口见本批RUN_GUIDE；旧64证券结果仅为历史样本，不代表本批全A。controlled_manual，无定时写入、QMT或实盘。

## 当前十轮受控迭代（2026-10-09，v0.7.144）

已完成10轮、20候选结果，E1有效描述读数（筛选通过0） 19、工程错误0，新增正式/JQ资格0。416回归通过；修复真实进程树RSS预算/回收，旧蓝图Status/Resume消费核验后的冻结源码。研究窗2024-05-17..2026-09-24、原64证券，已有历史暴露，未生成新市场日期或未知样本。

本批[报告](E:/quant/reports/fuxi_v1_ten_rounds_20261009/REPORT.md)、[每轮结果](E:/quant/reports/fuxi_v1_ten_rounds_20261009/ROUND_RESULTS.csv)、[真实反馈数据库](E:/quant/reports/fuxi_v1_ten_rounds_20261009/campaigns/campaign_fuxi10r20261009/evidence.sqlite)。原完整工程蓝图和操作说明仍在下文；controlled_manual，无定时写入、QMT或实盘。

## 当前完整工程蓝图 R5（2026-10-09）

十二软件工作包已接通，原64×663实际输入与统一8阶段运行/幂等恢复完成；408回归0失败，三受限情景各563/563条件归因、冻结预测57+57误差0。当前入口为scripts/run_fuxi_blueprint.ps1（默认Status，Resume消费冻结批），完整64进化转换及隔离回退已交付。

当前完整版：[BLUEPRINT](E:/quant/reports/fuxi_v1_full_landing_20261009/BLUEPRINT.md)；[运行说明](E:/quant/reports/fuxi_v1_full_landing_20261009/OPERATOR_GUIDE.md)；[工作事项](E:/quant/reports/fuxi_v1_full_landing_20261009/WORK_ITEMS.md)；[实际验收](E:/quant/reports/fuxi_v1_full_landing_20261009/REPORT.md)。软件12/12与金融证据分层：原47子项41项执行、6真实条件未齐；严格权益未知328日，正式资格/JQ/未来0，latest-vintage不恢复历史PIT。controlled_manual、无定时写入/实盘。以下章节均保留各自历史状态，以本节为当前入口。

## 当前账户与蓝图 R4（2026-10-09）

受限资产单独记账＋总账户情景已实现为明确研究通道：simulation.run_account接受restricted_policy/restricted_scenario；python -m fuxi_v1.restricted_assets提供手动冻结/完整成对重放入口。股数、应收、未知和初始本金保留，估计不支付订单或解除交易；三个情景逐一重放，不自动取得正式资格。54相关回归、三个情景各563历史配对和3384严格账本核对通过。旧PIT不可得时继续冻结最新已取得版本做回顾研究并披露限制。

当前完整版：[BLUEPRINT.md](E:/quant/reports/fuxi_v1_restricted_assets_20261009/BLUEPRINT.md)；清单：[WORK_ITEMS.md](E:/quant/reports/fuxi_v1_restricted_assets_20261009/WORK_ITEMS.md)；证据：[REPORT.md](E:/quant/reports/fuxi_v1_restricted_assets_20261009/REPORT.md)。仍39/47、8完整条件；64股输入尚未建、完整归因/审阅与真正未来仍待，合格后手动JQ。controlled_manual，不开启定时写入。以下R3及更早章节为各自历史状态，以本节当前入口为准。

# 伏羲 V1.0：独立研究系统

## 当前研究数据口径 R3（2026-10-09）

行业/自由股本回顾诊断采用每批冻结前最新已取得版本，按真实采集时间及来源哈希选定，历史日期/生效区间不改、缺失不补零。旧PIT档案无法取得时不再作为本期诊断前置；原历史可得/修订、投资中性化与正式资格仍分别披露。新原生请求使用style_data_policy="latest_available_snapshot"，industry_candidates为预声明已登记版本列表；自由股本保持声明冻结输入/目录版本，未自动采集更晚数据。未指定的新模式之外的旧入口保持兼容，不宣称所有生产流程已切换。

当前完整版：[BLUEPRINT.md](E:/quant/reports/fuxi_v1_latest_vintage_20261009/BLUEPRINT.md)；清单：[WORK_ITEMS.md](E:/quant/reports/fuxi_v1_latest_vintage_20261009/WORK_ITEMS.md)；实施与证据：[REPORT.md](E:/quant/reports/fuxi_v1_latest_vintage_20261009/REPORT.md)。受限资产分账/压力方案已具体化，尚未接管账户。以下R2及历史章节保留其当时状态。QMT仍移出，运行controlled_manual，不开启定时写入。


## 当前工作蓝图 R2（2026-10-08）

当前正式路线为关键证据补齐、受控研究迭代、合格候选后用户手动聚宽上传/原始结果回传，以及冻结后未来验收。真实交易客户端衔接已按用户指令移出本期范围；12个工作包共47子项，39项工程/诊断执行、8项完整验收待证据，删项不计通过。历史章节中的平台计划、计数和实验读数保留其当时范围，不是当前依赖。

当前完整版：[BLUEPRINT.md](E:/quant/reports/fuxi_v1_controlled_iteration_20261008/BLUEPRINT.md)；当前清单：[WORK_ITEMS.md](E:/quant/reports/fuxi_v1_controlled_iteration_20261008/WORK_ITEMS.md)；本批真实结果与限制：[REPORT.md](E:/quant/reports/fuxi_v1_controlled_iteration_20261008/REPORT.md)。继续使用controlled_manual，不因本期迭代开启定时写入或增加外部LLM预算。


软件版本 `1.0.0-rc9`，对应 design-0.3。原生研究链、模型与组合实验、受控进化、独立验收及模拟发布接口已实现。真实市场的数据认证、平台对拍和未来 E5 经济验收仍未完成；工程候选版本不表示已有可投资策略。

实施及实际结果：`reports/fuxi_v1_full_20260929/REPORT.md`、`IMPLEMENTATION_MATRIX.md`。旧 bootstrap 报告保留第一批历史事实。

rc2 补记：用户于2026-09-29明确授权DeepSeek联网验证及补跑。LLM提示明确JSON合同，拒绝重复键/非有限值/超额候选/截断回复；解析失败也保留供应商用量、响应身份和错误阶段。实际验证与补跑记录见 `reports/fuxi_v1_deepseek_20260929/REPORT.md`，旧报告的网络阻断状态为当时事实。

## 使用

在 E:/quant 使用已有解释器；没有安装新环境。精确依赖版本见本批 `ENVIRONMENT.json`。

```powershell
$fuxiPython = 'C:/Users/Zhong/.workbuddy/binaries/python/envs/default/Scripts/python.exe'
& $fuxiPython -X utf8 -m unittest discover -s fuxi_v1/tests -v
& $fuxiPython -X utf8 -m fuxi_v1 system-fixture --output reports/my_fuxi_run/synthetic.json
& $fuxiPython -X utf8 -m fuxi_v1 campaign --input reports/my_fuxi_run/synthetic.json --output reports/my_fuxi_run/campaigns --rounds 6 --candidates 2
& $fuxiPython -X utf8 -m fuxi_v1 campaign --input reports/fuxi_v1_full_20260929/market_input.json --output reports/my_fuxi_run/market --rounds 6 --candidates 2 --mode adaptive
```

`--mode adaptive/fixed/no_memory` 控制进化方式；默认模型为 `--models ridge,lightgbm,xgboost`，空字符串关闭。恢复时使用相同参数并追加 `--resume-id campaign_<已有ID>`。输入或代码变化须创建新批次，完成尝试不重复计费/奖励。`demo/research` 为第一批兼容入口，完整流程使用 `campaign`。

`--llm` 明确启用一次至多 1,800 输出 token 的 DeepSeek 请求，仅发送字段目录和少量证据描述，不发送价格面板或密钥。连接失败留存失败记录及未知费用，采用明确标记的类型化随机回退，不冒充 LLM 产出。请求别名为deepseek-chat，实际模型以回执为准；不会把别名当成固定后端版本。结构化回执保留未受信任的生成文本供审计，任何内容仍必须通过本地AST编译器。

本地数据适配器只读 CSV 或哈希绑定的分片补数目录，不改原始数据；输出文件必须尚不存在。

```powershell
& $fuxiPython -X utf8 -m fuxi_v1 snapshot-local --raw-root data/raw --start 2024-01-02 --end 2026-09-24 --max-instruments 24 --output reports/my_fuxi_run/input.json
& $fuxiPython -X utf8 -m fuxi_v1 verify reports/my_fuxi_run/campaigns/campaign_<ID>/bundles/run_<ID>
```

每批目录包含 `config.json / input.json / policy.json / code/ / evidence.sqlite / attempts/ / models/ / bundles/ / report_<hash>.json`。数据、代码、依赖、窗口、成本、基线和种子进入内容身份。运行中源码变更阻断报告；失败尝试和实际费用仍保留。

## 模块分工

| 职责 | 文件 | 实际作用 |
|---|---|---|
| 契约/目录 | contracts, dsl | 五态、哈希、严格 AST、17 算子、量纲/数值/复杂度约束 |
| 数据/标签 | data, datasets | 多版本时点选择、单位归一、会话日历、下一会话执行、标签成熟 |
| 因子研究 | search, campaign | 机制假设、谱系、模板/随机/单窗变异/组合/LLM、实际动作编排 |
| 模型研究 | models | Ridge/LightGBM/XGBoost 真滚动训练、仅训练窗标准化、模型工件 |
| 组合研究 | portfolio | 并列排名权重、仓位/行业/换手约束、组合轴实验、StrategyIR |
| 评价 | evaluation, research_evaluation, statistics, attribution | E0–E4、配对/风格诊断、单通道剥离、账户增量、时间块及压力测试 |
| 账户 | account, simulation | 固定股数、现金/应收、结算、部分成交、停牌/缺事件阻断 |
| 证据/记忆 | evidence | 不可变事实、版本奖励、依赖失效、经验检索、跨批次暴露/资格 |
| 运行 | campaign, workers, runtime | 串行硬预算、超时/内存、独立进程、工作区锁、断点恢复 |
| JQ | jq, jq_strategy | 只读诊断、证券映射、冻结历史策略、候选回测编译、字段对拍 |
| 验收/发布 | policy, acceptance, operations | 校准协议、固定观察时点、E5、ReleaseBundle、模拟激活/回退 |
| 执行/监控 | execution, monitoring | 意图、栅栏、未知回报、幂等成交、漂移分类、研究 Pareto 档案 |
| 可选扩展 | plugins | 能力登记、AST 再校验、方法比较边界；不假称外部优化算法已实现 |

## 因子与策略生成路径

冻结输入/资源/协议 → 选择研究动作及父候选 → HypothesisSpec + AST → 编译与 PIT 求值 → 成熟标签及 E1 → 配对风格诊断/单通道剥离 → 冻结基线加一因子 → StrategyIR → 下一会话订单/固定股数账户 → E3 同窗增量 → E4 双成本及半容量压力 → 裁决/冻结包 → 有效证据更新档案、经验和下一动作。

调度器先探索已准入生成器，再使用实际读数完成率/耗时。只有有效、非合成、可识别账户证据才进入经济奖励；有经济读数后用于动作选择，同时保留探索。未知经济表现不记为零，LLM 自评不作裁判。奖励更正替代旧槽，依赖状态失效并从有效版本重建。

模型轴固定特征，特征轴固定算法和完整样本支持，组合轴固定分数只改调仓周期，均真实运行。优化器没有修改成本、裁判、保护窗、生产默认或交易权限的动作。

## 公式及门槛

五态：`valid / data_insufficient / unsupported / error / not_evaluated`。有效读数不自动等于 `decision=pass`。

```
Coverage_t = 同时有效的因子/成熟标签配对数 / 当日声明股票池数
IC_t = Corr(f,y); RankIC_t = Corr(rank(f),rank(y))
ICIR = mean(IC_t) / sample_std(IC_t)  # 不年化
required_sections = ceil(((z(1-alpha/max_trials)+z(power))*sigma/effect)^2 * dependence_multiplier)
NAV = cash + receivables + sum(fixed_shares * evidenced_mark)
```

并列平均秩，常数截面为缺失；不把面板行数当独立时间样本。rc1兼容协议的历史记录：研究声明 effect=0.03、alpha=0.05、power=0.8、12 次尝试上限、6 会话时间块。校准结果要求 7,075 个截面，实际远不足。

rc1兼容协议曾要求E1时间块区间下界>0且满足功效规划；当前历史研究v2采用≥120截面、配对覆盖≥0.80、6/12/24块各3999次95%区间下界>0、正年度占比≥0.60、沪深分别≥120截面且均值>0，功效只作规划。E3 要求同窗净收益增量下界>预登记最小净增量；E4 要求双倍成本/半容量均具备有效账户证据并通过。E2 还要求可识别组合增量，多字段须完整单字段通道剥离及逐截面/账户配对。动量和价格变异系数残差仅是风格诊断，缺 PIT 行业/市值不能宣称投资级中性化。

功效公式是研究近似；依赖倍率不是精确有效样本估计。199 次 bootstrap 和事后 Holm 表仅描述自适应历史搜索，不证明确认性误报控制。独立前向协议须有校准证据、足够重采样分辨率、固定观察计划；不可看结果后改门槛。

## 冻结与独立验收

1. **诊断 JQ**：支持的 E0 即可编译；不下单、不授资格。
2. **首次正式 JQ 回测**：适用 E0–E4 通过，冻结政策、策略及已核验数据回执绑定。E5 不是前置条件。
3. **前向 E5**：真实 JQ 信号/订单/账户对拍之后重新冻结同一策略，固定未来数据范围及观察时点。未成熟时保持 pending。
4. **模拟发布**：E5 回执 → 代码/模型/策略/数据/政策/状态格式哈希包 → paper_activation。回退先对账未知订单和真实账户。没有实盘券商后端。

操作入口例：

```powershell
& $fuxiPython -X utf8 -m fuxi_v1 operate register-data --request path/to/reviewed_data_request.json --authority data/fuxi_v1/authority.sqlite
```

其余动作：`admit-research / compile-jq / import-jq-parity / register-protocol / freeze-forward / observe-forward / build-release / activate-paper / rollback-paper / revoke`。请求字段见 operations.py、policy.py；tests/test_lifecycle.py 提供完整正向及拒绝路径的可执行合同样例。审理请求需要 reviewer、reviewed_at、绝对证据路径与 SHA256；这是受信任操作者的证据记录，不是对供应商真实性的密码学证明，生成器无权调用。

CLI 统一用 `data/fuxi_v1/authority.sqlite`，不同输出目录不能绕过保护窗。库调用须传入同一 authority_path；默认只共享输出根目录内的登记库。规范市场依赖、证券、时间和标签区间共同定义暴露，改供应商名称不移除规范依赖。

JQ 服务冻结历史计划，不是在线重训。外部平台对拍尚待完成；未认证应收 API 字段导出 null，须用独立事件证据补齐，不得填零取得一致。

## 能力边界及运维

研究范围为日频、long-only、明确资金/费率合同。旧主板回放保留；rc7全市场入口显式声明沪深主板、创业板、科创板和北交所逐证券数量规则。时序算子仅接受字段叶子。支持整股拆分、明确应收/到账；含税分红、配股、碎股、退市及特殊交易情形须有专用证据适配，不支持的事件阻断资格。公司行动后没有除权定仓参考价时不能沿用除权前价。

旧24只起始日样本保留为偏差诊断；rc7使用含退市历史的全市场生命周期名单及实际补充的停复牌、名称和分红记录。供应商当前版本与声明发布时间代理不构成完整 PIT 修订、交易及公司行动独立认证。RankIC 标签是无成本价格代理，账户成本独立冻结。2025/2026 均已研究暴露。

SQLite 触发器与哈希防止意外改写，不是对恶意本机管理员的安全边界。隔离 worker 只执行受控 AST/受信任模型代码，不宣称能沙箱任意敌意代码。

保持 controlled_manual 和既有 FuxiWriterLock；未开计划任务、autonomous_writes 或旧 S5 新默认。停止调用新入口即可停止该研究软件，保留原始资产和全部证据。M7 的 PySR/BoTorch/GEPA/动态系统/研究 LLM 训练为可选后续能力，未实施的插件明确不支持。


## rc3补记：可达性、丰富反馈和检查点

本批见reports/fuxi_v1_evolution_20260929/REPORT.md及PROTOCOL_REVIEW.md。正式数据仍未获完整认证；增加bootstrap尾部解析度不足的准入否决，不以候选数量代替证据。

campaign新增可选--data-receipt-id，直接查询工作区authority当前回执；新增--study-protocol（schema见本批PLAN.json中的protocol）、--pause-after-round。预注册时刻表可允许多次有上限的LLM请求，普通campaign仍最多一次。LLM反馈出站须有明确授权。

learning.py提供失败分类、具体读数/样本、同窗父子代IC及净增量状态、相似性、行为指纹、兼容性导入和源撤销传播。继承经验只用于研究；不继承资格或经济奖励。no_memory在受控时刻表下使用无绩效锚点，提示卡为空。每5轮检查和两块连续无新行为/父子代改善停止规则均在实验前冻结。

旧rc2功效/账户门槛保留，新的可达性报告将功效样本不足与方向不支持拆开。停止调用rc3并保留证据可回退；旧冻结批次不能用新代码强行resume，应使用其代码快照或另建批次。


rc3.1补记：LLM引用证据和逐来源撤销依赖已补强，79测试；正常有效记忆与提示不变。40轮市场试验保留rc3代码身份，见reports/fuxi_v1_evolution_20260929/revocation_followup/REPORT.md。


## 历史研究入口（rc4.1引入）

rc4.1的2021年以来显式窗口、沪深主板逐日样本、语义字段检查、现金事件及三种子对照见 reports/fuxi_v1_history_20260930/REPORT.md。历史数据使用 snapshot-history 与 history-study；旧 snapshot-local 必须显式 --legacy-code-order 才可复现旧样本，不再静默采用24只代码排序样本。

    python -X utf8 -m fuxi_v1 snapshot-history --plan reports/fuxi_v1_history_20260930/PLAN.json --raw-root data/raw --supplement reports/fuxi_v1_history_20260930/supplement --output <新的快照路径>
    python -X utf8 -m fuxi_v1 history-study freeze --plan <事前计划> --candidates <冻结候选> --input <冻结快照> --output <新试验目录>
    python -X utf8 -m fuxi_v1 history-study run --output <已冻结试验目录>
    python -X utf8 -m fuxi_v1 history-export --run <完成试验目录> --formula <完整公式哈希> --output <新导出根目录>

使用项目既有WorkBuddy Python解释器。结果目录可续跑且校验代码/数据/规则哈希；已完成目录校验原冻结代码存档后返回原结果，不重复收费或搜索；未完成批次要求当前代码完全匹配冻结版本。协议或数据改变必须新建冻结批次。功效规划是参考，不是E1普遍硬门槛；首次正式JQ仍要求数据审查与完整账户，未获得未来独立确认。96项测试通过。更多历史数据不自动成为未见样本。

## rc5 账户修复与扩搜前置验证

rc5里程碑版本为1.0.0-rc5；最新rc5.2实测见下节。送转按登记日持仓产生独立股份应收，除权日估值、上市日可售；支持完整零股卖出、独立现金/股份/NAV重放。重整新增股份不能直接乘到原股东持仓。非整数实际分配未知仍阻断，固定分红税负保留为研究情景。

含remediation的计划必须显式提供版本化证据配置；无需改写raw。配置示例与本批全部来源见reports/fuxi_v1_remediation_20260930。独立来源用于核对事件日期和原始行情，不用复权收益替代账户收益。

    python -X utf8 -m fuxi_v1 snapshot-history --plan <计划> --raw-root data/raw --supplement <分红和名称补充目录> --remediation <证据配置JSON> --output <新批次目录/market_input.json>
    python -X utf8 -m fuxi_v1 history-preflight --plan <计划> --input <同一输入JSON> --output <新批次目录/preflight>
    python -X utf8 -m fuxi_v1 history-study freeze --plan <同一计划> --candidates <候选JSON> --input <同一输入JSON> --output <新批次目录/evaluation>
    python -X utf8 -m fuxi_v1 history-study run --output <新批次目录/evaluation>

preflight固定四探针、正常/双倍成本/半容量三情景，必须全部可识别并完成独立重放；history-study直接核验前置回执与全部12份文件哈希。失败时不消费扩搜预算。输入、代码或规则改变须新建批次，不能覆盖旧回执。账户可计算但尚未认证时，描述净增量可写入反馈，经济奖励及JQ资格保持不授予。


rc5本批实跑：111测试通过；12份固定探针独立算术重放全部通过，9份账户完整可计算，反转20混合的3个情景因600077退市后估值缺口仍为data_insufficient。取得400192的674条未复权行情，不替代36个过渡日期的计价或账户确权。未扩大搜索（0/120），正式认证账户0，新增JQ0。实际报告见reports/fuxi_v1_remediation_20260930/REPORT.md；同窗停牌修复复跑存于attempt2/preflight。


## rc5.2 冻结排除诊断与扩搜

已按用户授权冻结排除7只股票，96只诊断池的12/12账户探针全部完整且重放一致；完成60条旧公式复评及120/120个新搜索请求，共121条唯一公式评价，121条账户路径可计算。新搜索带来58条不同于旧库/父代对照的公式，其中43条通过E1。正式新增JQ资格仍为0。补充等权对照中，新公式同时通过E1且三种成本情景名义增量区间下界均为正的数量为0。 三情景成本数值检查通过（全部/新）41/28；原103只/旧裁决保留。事后幸存者池不得授予正式资格，研究数据仍未独立认证，DeepSeek0、JQ提交0。 精确投影上批不可变输入，live raw外部更新的失败构建保留；114测试和真实账户重放通过，未改旧S5默认。详情reports/fuxi_v1_exclusion_20260930/REPORT.md，实际多种子检查点与候选清单同目录。 补充三情景等权对照后，E1且各成本下界为正的全部/新公式为0/0；该对照仅事后解释，不改原门槛。 首跑第13轮AST超限中断已保留；rc5.2修复有界提案拒绝，24历史决策精确复现、12探针字节一致，77评价按同内核审计复用，实际完成目录attempt2/evaluation。


## rc6工程修复（2026-10-02）

182测试、10轮真实历史诊断、12探针、30账户独立重放；正式JQ资格0，实盘未准入。完整结果及数值门槛见`reports/fuxi_v1_engineering_20261002/REPORT.md`。新增online.py逐日重算、safety.py持久停单控制、data_assurance.py独立证据导入及readiness.py分层验收。`operate register-data`仅派生诊断验证，返回validation_receipt_id；`operate import-data-assurance`需审理后的四类主证据及完整持仓延续范围，不能以声明时钟/源布尔值代替。独立数据review信任边界仍需真实审理人负责，不是自动证明。bind_assurance只绑定已审理且本身完整的来源，不补造事实。

在线daily_decision只是同DSL计算接口，通用安全控制器只有paper权限。真实平台路由、生产数据时点/修订认证与未来E5尚未验收；未开启自动化写入。


## rc7 全市场数据与交易合同（2026-10-03）

全市场入口为scripts/fuxi_all_a_20261003.py，原始CSV只读。当前供应商响应落在data/fuxi_v1/supplements/20261003；分片目录通过CATALOG.json及CATALOG_METADATA.json绑定原始来源、每个完整查询、北交所旧新代码和名称/分红/日历/生命周期元数据。整日查询可包含显式空响应；整页响应必须继续请求尾页。缺值、冲突重复行和不明事件保留为未知。

数量合同：沪深主板最低100股、100股步进；创业板限价单最多30万股、市价单最多15万股；科创板最低200股、1股步进、限价10万/市价5万；北交所最低100股、1股步进。动态价格限制、价格申报范围、市场权限、费用与公司行动另需带日期证据。没有涨跌停价不能仅凭上市日期推断不设限制。北交所248条代码关系区分稳定研究标识和当日交易代码，不冒称券商已完成账户权益迁移。

全市场snapshot-history消费实际turnover_rate和total/float/free_shares，分别按百分比和万股转换；缺失free_shares仍缺失，成交量不替代换手率。盘后字段使用声明的保守18:00日频代理，只能次一交易会话执行；历史发布/修订认证仍需独立证据。

临时剔除使用observed_at/effective_at/review_after及证据哈希；只限制此后新进入，保留已有持仓、缺报价风险和应收权益。未来解除事实不改写过去决策。退市前历史仍在全市场研究名单；全局服务/认证问题不能通过剔除全体股票解决。

本批先做2021-01-04至2026-09-30全历史数据逐格审计，另做每股票10轮真实供应商三会话账户算术诊断和近期全市场下游加载；这些不替代完整持仓跨事件延续认证，也不授予因子JQ资格。最终读数以reports/fuxi_v1_all_a_20261003的结构化回执为准；初始合成数量测试保留在根目录，创业板限价上限更正后的十轮位于contracts_v2，显式市价/限价合同并绑定最终代码的十轮位于contracts_v3。全市场native市价回放与JQ市价生成均使用市价申报上限。

本机QMT自带Python3.6可导入SDK，环境探针没有连接账户或下单。研究Python保持既有环境；账户登录、实际权限、模拟对拍、投资数据审理和未来E5仍须分别验收。本地JQ编译器尚无已验证的北交所映射；不能宣称平台全市场可交付。


## rc8 公告原文与全市场验收入口

用户已购买的anns_d权限实测成功。public_notices从哈希绑定的实施PDF文本复算普通股东条款，action_evidence导入时复算核验；historical_adapter通过计划issuer_action_evidence绑定独立批次原文包。北交所供应商历史索引按新旧代码查询，原文发行人代码按公告日和官方映射生效日核对。空值不补零、重整与实际随机零碎股未知仍阻断，不自行生成独立数据回执。rec_time后补录不代表历史首次可得时间。

本批采集、测试、预检和十个冻结全市场候选验收入口为scripts/fuxi_notice_*_20261003.py；结果以reports/fuxi_v1_announcements_20261003/ten_rounds/OUTCOME.json实际结构化读数为准。全生命周期2025-01-02..2026-09-30，十个固定公式、三种成本/容量情景、每日/批量对拍与资金/股份/NAV算术核验。未启动扩张搜索、未把已有暴露当未来E5。全市场构建释放已消费的原始行字典；标签逐日集合核对保持原验证含义并避免线性证券扫描。旧S5默认不变。

用户手动提交聚宽和QMT代码；现有JQ生成器依旧只支持冻结历史回测，实际前向模拟信号适配、北交所平台映射与平台日志尚待分别验证，不能由本机JQData鉴权或软件测试代替。


rc9: supported cash distributions during recorded full-day halts deduct gross cash from the declared frozen valuation reference. Raw prices stay missing; unresolved or noncash references remain blocked. The resumed raw pre-close must still match the adjusted reference. The stopped rc8 attempt and new rc9 freezes are retained separately.

## rc10 公告条款关联与发行人停牌证据（2026-10-03批次）

普通转增公告仅在同一年度、明确零现金、股本未变且实施股东比例及股本基数一致时关联审批章节；单独缺现金仍未知。PDF页码切断“不派发现金红利/股利”的明确跨页格式单独处理，任意内嵌数字或金额不移除。`halt_evidence.py`从原PDF和文本重核发行人及开市停牌起始日，只确认明确一天，不根据预计停牌期限扩展；计划通过`issuer_halt_evidence`绑定原件哈希。报价特征仍缺失，不允许借声明停牌估值新建仓；冲突或未知公司行动继续阻断，声明参考不作为认证的原始报价。

最终代码228项测试通过。首组十轮rc9保留于`reports/fuxi_v1_announcements_20261003/ten_rounds_v2`，源补查和修复后的第二组使用独立目录`ten_rounds_v3`；读取实际OUTCOME及REPORT，不以目录名或测试通过推断JQ/实盘准入。共10个唯一固定公式、两个相同设计的十轮执行，外部LLM与扩张搜索请求均为0。

## rc11 连续停牌区间补证

rc10单日起始补证把反转5持仓覆盖缺口移至第二天，`ten_rounds_v3`在12预检后停止，候选完成数0，保留`STOP_FOR_INTERVAL_CORRECTION.json`。rc11将实际起始公告与明确停止/复牌日的原文配对，用共同覆盖区间提供研究估值参考：688766为2025-11-26至2025-12-08共9个交易日。原文起始日26日与复牌文追述25日的一日冲突保留，不给争议日新增补证，不从预计时长外推。后续复牌文只是历史区间确认，不是各历史日PIT可得性或alpha证据；实际特征仍缺报价、源冲突仍阻断。

rc11实际231项测试通过，完整新复跑入口为`scripts/fuxi_notice_campaign_v4_driver_20261003.py`，冻结目录为`ten_rounds_v4`。两个完整十轮的最终统计以本批REPORT和实际OUTCOME核验，停止的rc8/rc10尝试不混入十轮分母。前向聚宽/QMT平台接口与真实日志仍需分别验收。
