# -*- coding: utf-8 -*-
"""_add_p20260915_011.py — 登记「狮子锚点因子族接入」提案。

变更性质显式声明 proposal_change_kind='behavior' + domains=['T3模块/通道']:
--bump 的 zero_adjudication 是**子串匹配**, 会误判文案 (见 P-20260915-006 教训),
必须显式覆盖。本次新增字段会改变 S1 候选池 ⇒ 真实行为变更, 非影子。
"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

P = 'e:/quant/data/stage1_optimization_proposals.json'
d = json.load(open(P, encoding='utf-8'))
prods = d.get('proposals', d) if isinstance(d, dict) else d

PID = 'P-20260915-011'
if any(str(p.get('proposal_id')) == PID for p in prods):
    print(f'{PID} 已存在, 跳过'); sys.exit(0)

entry = {
    "proposal_id": PID,
    "title": "狮子锚点因子族接入: 最近涨停日→启动点结构锚点通道 (后复权比值口径)",
    "category": "因子接入",
    "trigger": "用户要求把《变种小狮子 V3.0》审计中的 bias 构造提炼进伏羲候选池, 并指定"
               "后复权比值口径。审计: reports/lion_v3_strategy_audit_20260915.md",
    "current_state": "NOVEL_FACTOR_LIBRARY 共 256 条, 其中反转/位置/动量类 105 条**全部**是"
                     "固定窗口构造 (rolling(20)/rolling(60)/rolling(252)/rolling(760)); 无任何"
                     "**事件锚点**型 (以某次具体事件为锚、锚点位置随个股动态变化的) 因子。"
                     "data_loader_ext 亦无涨停→启动点类通道。",
    "proposed_change": "① data_loader_ext.py 新增 load_lion_anchor_wide(daily_index, daily_columns, "
                       "lookback=760) 事件锚点通道, 输出 3 字段: lion_bias (后复权 close / 后复权 "
                       "anchor_low, anchor=最近涨停日之前最后一根阴线 low) / lion_lu_days (距最近涨停"
                       "交易日数, 窗口内无涨停截断为 760) / lion_lu_cnt (760 窗口涨停次数)。"
                       "② daily_factor_hypothesis.load_data 注册三字段入 extra。"
                       "③ NOVEL_FACTOR_LIBRARY 注入 novel_227/228/229 三条。"
                       "④ 后复权实现: 因 adj_factor.csv 仅 46 个季频快照 (每股 ~24 个) 无法覆盖"
                       "快照间除权, 改用 pre_close(t)/close(t-1) 自算**日频**复权调整比再 cumprod "
                       "(已与官方因子交叉校验 600519: 1/1.1679=0.8562 vs 官方 0.8563)。",
    "expected_impact": "① 新增 3 条候选因子。伏羲自身口径实测 (174 有效日, fwd5): lion_bias ic_mean "
                       "-0.0300 / ICIR 0.210 → 过 S1 闸门; lion_lu_days ic_mean +0.0421 / ICIR 0.394 "
                       "→ 过闸且为族内最强 (+IC% 66.7%); lion_lu_cnt ic_mean -0.0132 / ICIR 0.163 "
                       "→ |IC| 未达 0.02 门槛, 作为**方向对照**保留 (社区策略恰取涨停次数最多者, "
                       "实测方向相反)。② **须标注的局限**: 独立实测显示三者与既有反转/流动性因子"
                       "存在重叠 —— lion_bias 与 rev20/rev60 截面相关 -0.400/-0.481, 对 "
                       "rev20+rev60+size+liq 正交后 IC 仅 +0.0090 (但符号由 -0.0250 翻转为 +0.0090, "
                       "t=2.32); lion_lu_days 与 liq 相关 -0.295, 正交后 IC +0.0054 (t=1.36) 不显著。"
                       "③ 与池内既有因子零重名, blocked=False, 可进候选。",
    "implementation_scope": "scripts/data_loader_ext.py (新增 ~150 行通道) + "
                            "scripts/daily_factor_hypothesis.py (import/注册/3 条库条目)",
    "generator_hint": "auto",
    "risk_level": "low",
    "rollback_plan": "三处均为**新增**, 不改动既有字段与裁决层: ① 删 load_data 中 load_lion_anchor_wide "
                     "调用块即彻底回退 (字段消失, 依赖它的因子降级为 NameError 被 S1 记 0); "
                     "② NOVEL_FACTOR_LIBRARY 中按 name 删除 novel_227/228/229 三条即可。无开关。",
    "status": "landed",
    "domains": ["T3模块/通道"],
    "proposal_change_kind": "behavior",
    "landed_detail": "load_lion_anchor_wide 落地于 data_loader_ext.py; load_data 注册 extra['lion_bias'/"
                     "'lion_lu_days'/'lion_lu_cnt']; NOVEL_FACTOR_LIBRARY novel_227/228/229 已注入 "
                     "(compile 自检通过)。验证: scripts/_verify_lion_factors.py → 三字段非空率 "
                     "last 4982/5422/5516 (共 5544 只), first_valid 中位 #0/179 (无 reindex 取交集 bug); "
                     "公式预检 (True, [], name) 全通过; RRE 排序稳定性 0.999~1.000。"
                     "独立复算: research/_verify_bias_adj.py (复权口径) + research/_verify_bias_incremental.py "
                     "(增量价值)。",
}

if isinstance(d, dict):
    d.setdefault('proposals', prods).append(entry)
else:
    d.append(entry)

json.dump(d, open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'✅ {PID} 已登记, 提案库 {len(prods)} 条')
