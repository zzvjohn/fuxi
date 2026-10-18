"""Render actual full-census readouts and append the scoped implementation ledger."""
import json
import re
import sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import digest,require,load_json
from fuxi_v1.datasets import file_hash,execution_profile
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_all_a_20261003'


def new_json(path,body):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:json.dump(body,handle,ensure_ascii=False,indent=2)


def append_native(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    old=path.read_bytes() if path.exists() else b''
    marker=text.splitlines()[0]
    require(marker.encode('utf-8') not in old,'duplicate ledger append')
    eol='\r\n' if not old or old.count(b'\r\n')>=old.count(b'\n')-old.count(b'\r\n') else '\n'
    with path.open('ab') as handle:handle.write((eol+eol+text.replace('\n',eol)+eol).encode('utf-8'))


def main():
    import pandas as pd
    summary=load_json(OUT/'DELIVERY_SUMMARY.json');audit=load_json(OUT/'DATA_AUDIT.json')
    require(summary['current_collection_status']['state']=='complete','collection incomplete')
    require(file_hash(OUT/'DELIVERY_STATUS.csv')==summary['registry_sha256'],'delivery ledger changed')
    checks=load_json(OUT/'contracts_v3/TEN_ROUND_CONTRACTS.json')
    require(checks['code_hashes']=={p.name:file_hash(p) for p in (ROOT/'fuxi_v1').glob('*.py')},'contract code changed')
    tests=load_json(OUT/'TEST_RESULT_FINAL.json')
    require(tests['exit_code']==0 and tests['failures']==0 and tests['errors']==0,'tests failed')
    require(tests['code_hashes']==checks['code_hashes'],'tests do not bind delivered code')
    from fuxi_all_a_20261003 import active_catalog
    catalog_root,full_path=active_catalog()
    full=load_json(full_path);smoke=load_json(OUT/'smoke/preflight/DECISION.json')
    require(full['catalog_manifest_hash']==audit['manifest_hash'],'report/audit catalog mismatch')
    require(smoke['code_hashes']==tests['code_hashes'],'downstream smoke code changed')
    require(smoke['probe_count']==12 and len(smoke['readouts'])==12,'downstream probes incomplete')
    require(smoke['input_sha256']==file_hash(OUT/'smoke/market_input.json'),'downstream smoke input changed')
    require(smoke['plan_hash']==digest(load_json(OUT/'FULL_A_SMOKE_PLAN.json')),'downstream smoke plan changed')
    for row in smoke['readouts']:
        stem=row['name']+'_'+row['scenario']
        require(load_json(OUT/'smoke/preflight'/(stem+'.receipt.json'))==row and
            file_hash(OUT/'smoke/preflight'/(stem+'.json.gz'))==row['sha256'],'downstream probe receipt changed')
    repair=load_json(OUT/'REPAIR_STATUS.json');require(repair['state']=='complete','repair acquisition incomplete')
    core=load_json(OUT/'CORE_REPAIR_STATUS.json');require(core['state']=='complete','core acquisition incomplete')
    jq_access=load_json(OUT/'JQDATA_ACCESS_PROBE.json')
    jq_direct=load_json(OUT/'JQDATA_ACCESS_PROBE_V2.json')
    frame=pd.read_csv(OUT/'DATA_STATUS.csv');registry=pd.read_csv(OUT/'DELIVERY_STATUS.csv')
    history=load_json(OUT/'TEN_ROUND_HISTORY_CHECKS.json');exclusions=load_json(OUT/'EXCLUSIONS.json')
    require(file_hash(OUT/'DIAGNOSTIC_WINDOW_EXCEPTIONS.json')==summary['diagnostic_window_exceptions_sha256'],'diagnostic availability ledger changed')
    cost=execution_profile();cost['order_type']='market';cost['sell_tax_schedule']=[['0001-01-01','0.001'],['2023-08-28','0.0005']]
    new_json(OUT/'COST_PROFILE.json',cost)
    source_files={str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/'fuxi_v1').glob('*.py')}
    source_files.update({str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/'scripts').glob('fuxi_all_a*20261003.py')})
    source_files[str(Path('scripts/fuxi_jq_access_probe_20261003.py'))]=file_hash(ROOT/'scripts/fuxi_jq_access_probe_20261003.py')
    new_json(OUT/'DELIVERED_CODE.json',source_files)
    coverage=load_json(OUT/'LOCAL_COVERAGE.json')
    manifest=load_json(catalog_root/'CATALOG.json')
    raw_recheck={name:{'expected':r['sha256'],'actual':file_hash(Path(r['path']))} for name,r in manifest['sources'].items()}
    require(all(r['expected']==r['actual'] for r in raw_recheck.values()),'raw data changed during task')
    new_json(OUT/'RAW_RECHECK.json',raw_recheck)
    outstanding={'schema_version':'fuxi.all_a_external_requirements.1',
        'automatic_collection_completed':True,'global_gaps_cannot_be_stock_exclusions':True,
        'platform':{'choice':'awaiting_user','qmt_sdk_import':load_json(OUT/'QMT_SDK_PROBE.json'),
            'jqdata_access_probe':jq_access,
            'jqdata_direct_https_probe':jq_direct,
            'needed':['chosen platform and locally logged-in simulation account','SH/SZ/ChiNext/STAR/BSE market permissions',
                'actual signal/order/fill/cash/holding/receivable/NAV exports for identical frozen IR','broker runtime adapter and restart/unknown-order reconciliation']},
        'primary_sources_needed':['historical publication and revision archives','issuer/register-date ownership, share credit, fractional allocations and investor tax receipts',
            'delisted holdings/transfer/valuation chain including 600077 to 400192','independent reviewer and complete primary-evidence coverage'],
        'future_acceptance':'freeze prospective E5 after formal research and actual platform parity; exposed history cannot replace it',
        'credentials':'do not send passwords or tokens; use local account login and existing credential storage'}
    from fuxi_v1.action_accounting import SUPPORTED
    from fuxi_v1.pit_universe import canonical_code
    terms=pd.read_csv(OUT/'ACTION_TERMS.csv');requests=[]
    for row in terms.to_dict('records'):
        if row['status'] in SUPPORTED:continue
        requests.append({'ts_code':canonical_code(row['instrument']),'source_kind':'issuer_and_register_holder_rights',
            'date_start':row['ex_date'],'date_end':row['ex_date'],'exact_dates':row['ex_date'],
            'reason':row['status'],'needed_fields':'original implementation notice/publication clock; record ownership; actual holder cash/shares; share credit/fractional allocation/investor tax',
            'current_evidence_id':row['evidence_id'],'current_source_is_not_independent_certification':True})
    gap_frame=pd.read_csv(OUT/'DATA_GAPS.csv',dtype={'trade_date':str})
    for (code,reason),group in gap_frame.groupby(['ts_code','reason'],sort=True):
        dates=sorted(set(group.trade_date))
        requests.append({'ts_code':code,'source_kind':'original_quote_event_or_suspension_archive',
            'date_start':dates[0],'date_end':dates[-1],'exact_dates':','.join(dates),
            'reason':reason,'needed_fields':'original dated quote/rule/publication/revision evidence; suspension start/resumption if relevant; carried ownership and valuation if held',
            'current_evidence_id':file_hash(OUT/'DATA_GAPS.csv'),'current_source_is_not_independent_certification':True})
    require(requests,'no concrete external source requests')
    pd.DataFrame(requests).to_csv(OUT/'EXTERNAL_SOURCE_REQUESTS.csv',index=False,lineterminator='\n')
    outstanding['concrete_instrument_requests_csv']=str(OUT/'EXTERNAL_SOURCE_REQUESTS.csv')
    outstanding['source_requests_sha256']=file_hash(OUT/'EXTERNAL_SOURCE_REQUESTS.csv')
    new_json(OUT/'DATA_REQUIREMENTS.json',outstanding)
    sections=[]
    for r in history['rounds']:
        sections.append(f"| {r['round']} | {r['instrument_count']} | {r['status_counts'].get('valid',0)} | {r['status_counts'].get('data_insufficient',0)} | {r['filled_orders']} | 0 |")
    gaps='\n'.join(f'| {k} | {v} |' for k,v in sorted(audit['gap_counts'].items()))
    data_counts='\n'.join(f"| {api} | {r['valid_cells']} | {r['conflicting_cells']} |" for api,r in audit['stats'].items())
    probes='\n'.join(f"| {r['name']} | {r['scenario']} | {r['status']} | {r['identified_days']} | {json.dumps(r['gaps'],ensure_ascii=False)} |" for r in smoke['readouts'])
    excluded=', '.join(r['instrument'] for r in exclusions['exclusions'])
    report=f'''# 伏羲 V1.0 rc7 全市场工程接线与十轮核验（2026-10-03）

本批已完成全市场生命周期清点、Tushare补数、数据逐格审计、全市场原生消费核验和两类十轮账户检查。当前仍未满足投资数据独立认证、真实平台及实盘交付条件。软件版本为1.0.0-rc7；正式JQ资格、外部平台运行与实盘订单均为0。

## 实际范围与口径

供应商主表5911条，其中普通A股{summary['instrument_count']}只，按L状态5571只；另外1只CDR及1条过时T代码单列。主板、创业板、科创板及北交所按实际证券类型纳入，未按收益或因子结果选池。2021-01-04至2026-09-30共{audit['sessions']}会话；全史生命周期交集{len(full['cohort']['members'])}只，保留窗口内退市股，不移除既有研究暴露。2021–2026均为research_exposed。

三轮共完成{summary['current_collection_status']['completed_queries']+repair['completed_queries']+core['completed_queries']}个完整补数查询；其中首轮{summary['current_collection_status']['completed_queries']}个，包括69个月停复牌、1393个分红除权日、名称历史和248条北交所代码关系；分页请求数不等于完整查询数。响应投影、原始响应哈希、查询身份、时钟和失败类别分开保留。所有原始CSV哈希复核一致，补数位于data/fuxi_v1/supplements/20261003；CATALOG及元数据绑定每个输入。

首轮审计与对应代码逐字节保留在audit_attempt1，避免把修复后的解释回写成原结论。修复了历史涨跌停缓存没有pre_close而被错误判缺、相同换手率的部分/完整股本记录被误判冲突两项合同问题；仅用真实非空观测合并，真实冲突继续阻断。追加{repair['completed_queries']}个逐股票名称/行情查询，名称整股票替换、行情整股票区间替换，成功空响应也不得沿用旧值。当前消费目录为{catalog_root.name}，原CATALOG与原全史计划保留。不同季/年报告期的已实施纯现金分红，只在登记、支付与公告时钟相同且每期无矛盾时求和；重组、股份或同报告期矛盾不能聚合。

第二轮审计保留在audit_attempt2。再追加{core['completed_queries']}个完整查询（19个日期同表因子、268只每日指标、283只涨跌停），仅按实际报价存在且通道缺失选取。复权精度切换采用既有constant_factor_corroborated规则：第二表前后因子必须精确相同且为正，两个原始收盘和本日参考价全部一致，且没有已声明公司行动；不修改漂移阈值或拼接复权基准。实际比较{audit['factor_corroboration']['compared_cells']}单元，核对通过{audit['factor_corroboration']['accepted_cells']}，同供应商复核不作为独立认证。FACTOR_CORROBORATION.csv绑定每一对源查询。无效涨跌停价格现在同时在审计和执行账户阻断。

普通盘后字段采用声明的保守18:00日频代理，18:30决策、下一会话执行；这不是原始历史披露/修订时点认证。行情与复权、涨跌停、每日指标分别检查；冲突重复行保留为未知，缺值不补零。真实turnover_rate按百分比转比例，total/float/free_shares按万股转股，成交量不冒充换手率。北交所2021-11-15以前的新三板行情不进入A股特征历史。

## 已实现的工程修复

全市场census入口取消24只/主板样本限制，声明全生命周期名单与实际成员；目录读取和审计使用相同数值重复定义，原始文件、每个分片、完整查询、元数据和代码均绑定哈希。北交所稳定研究身份与当日交易代码分开，代码映射不能替代账户确权。

沪深主板100股步进；创业板限价30万/市价15万股；科创板最低200股、1股步进、限价10万/市价5万股；北交所最低100股、1股步进。全市场原生市价回放和本地JQ市价编译使用市价上限。各板块单日价限、价格申报范围、权限、费用及特殊事件仍需要带日期证据；缺涨跌停价不得仅凭上市日期猜测无限制。本地JQ编译器没有已验收的北交所映射。

临时剔除含发现/生效/复审时钟与证据，只限制此后新进入，保留原持仓及应收。未来解除不能改写过去决策。退市后缺确权时账户继续阻断，不冻结最后价冒充真实持仓价值。整日停牌与成交量矛盾单列，不借停牌豁免掩盖冲突。

规则来源：[深交所2026交易规则](https://docs.static.szse.cn/www/lawrules/rule/trade/current/W020260424690713155663.pdf)、[上交所科创板规则说明](https://edu.sse.com.cn/tib/qa/c/4866268.shtml)、[北交所交易规则](https://www.bse.cn/jygl_list/200028217.html)。来源仅支持声明的规则，真实平台能力仍需实测。

## 全史数据逐格读数

活跃生命周期单元{audit['active_lifecycle_cells']}，剔除明确整日停牌后的密集通道预期单元{audit['dense_expected_cells']}。密集通道全窗齐全股票{audit['dense_core_complete_instruments']}只，连同名称/事件/计价代理完整{audit['data_complete_under_vendor_proxy_instruments']}只；独立投资数据认证为0。剩余单元包含实际特殊交易、待证事件、冲突或缺行情，逐格见DATA_GAPS.csv，逐股票见DATA_STATUS.csv。全局修订/主证据不足不通过剔除整个市场解决。

每日指标的基础完整性指换手率、总股本与流通股本核心合同；自由流通股本可选缺失仍留空，依赖它的因子必须另过实际配对覆盖检查。

| 通道 | 有效单元 | 冲突单元 |
| --- | ---: | ---: |
{data_counts}

| 未解决原因 | 单元数 |
| --- | ---: |
{gaps}

公司行动条款状态：{json.dumps(audit['action_status_counts'],ensure_ascii=False)}。支持的供应商日期/条款也只是研究声明，碎股分配、投资者税负、登记确权和跨市场持仓仍需真实来源。缺报价的持仓不得删股重加权。9月30日缺行情的11只L状态普通A股均有明确整日停牌记录；它们留在名单中，不能因停牌假装不存在。

## 十轮实测

最终{tests['tests']}项测试通过。contracts_v3以实际{summary['instrument_count']}只证券标识配合合成报价核验市价/限价上限、T+1、零股、重复成交、缺计价和退出后应收，共{checks['instrument_rounds']}个股票轮次通过。根目录初始合同及contracts_v2保留；前者创业板上限解释已被更正，v2只覆盖限价合同，最终以绑定当前代码的v3为准。

另外10轮逐股票真实供应商行情回放，共{history['instrument_rounds']}个声明股票轮次。各股票采用固定时间分位的完整、无除权事件三会话窗口；这是存在性诊断，不是无偏研究抽样或完整持仓延续认证。无完整窗口明确记录data_insufficient，不计零收益；每轮现金/股份/NAV独立算术重放且非零成交总数大于0。可识别窗口{summary['instruments_with_ten_identifiable_diagnostic_windows']}只，10轮均有非零诊断成交{summary['instruments_with_ten_nonzero_diagnostic_fill_rounds']}只，详见DELIVERY_STATUS.csv。

10个不同历史窗口的股票为{summary['instruments_with_ten_distinct_diagnostic_windows']}只；较短上市历史的窗口可能重复，逐股票distinct_history_windows明确列出，不把重复执行当10份独立市场证据。

无完整窗口原因按实际活跃会话、停牌与通道读数分解：{json.dumps(summary['diagnostic_window_exception_counts'],ensure_ascii=False)}。逐股票见DIAGNOSTIC_WINDOW_EXCEPTIONS.json；不足三会话的新股保留并等待真实未来会话，窗口前已退市身份保留历史记录。全窗停牌及停牌证据缺失也分别记录，不混为补数失败或事后删除。

| 轮 | 声明股票 | 有效诊断账户 | 缺完整窗口 | 非零成交订单 | 正式账户 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(sections)}

费用为声明研究情景：初始资金100万元，买卖佣金各0.03%、最低5元，滑点5bps，容量为前一会话成交量1%；2023-08-28前卖出税0.1%、之后0.05%。按轮交替正常、成本×2、容量×0.5；分红平坦20%税负及碎股阻断均保留。完整合同见COST_PROFILE.json。没有优化公式或放宽经济门槛，也没有新增alpha结论。

## 全市场下游消费与持仓延续核验

smoke使用2026-05-06至2026-09-30全市场生命周期交集，预先声明20会话热身、5会话标签、7月31日校准末端及8月11日起研究，未按结果移除股票。短窗用于控制单体JSON内存和验证下游消费；整段历史仍由全量分片逐格审计，不能把短窗写成六年全账户通过。

首次原生消费在输出阶段因计划遗漏cohort.selection报KeyError，完整失败及部分产物保留在smoke_attempt1。计划生成器现在把显式cohort.rule原文写入selection；新全史计划以consumer_schema_repair绑定父计划，并断言除该描述字段及修复元数据外逐项不变，未变股票、窗口或数据。原计划与旧指针均保留；重跑使用修复计划，核心计算源码未变。

固定4探针×正常/双倍成本/半容量12条账户路径，实测可识别与算术重放状态如下；search_allowed={smoke['search_allowed']}，formal_account_evidence_certified={smoke['formal_account_evidence_certified']}。实际缺口继续保留，本批没有消费扩搜预算。

| 探针 | 情景 | 状态 | 可识别天数 | 缺口 |
| --- | --- | --- | ---: | --- |
{probes}

## 临时剔除及仍需补证

本批前向临时记录{len(exclusions['exclusions'])}只：{excluded}。600077至400192的账户确权及36个过渡计价日期仍未解决；已退市股票本身不能新建A股仓位，此记录也不改变其退市前历史。其他生命周期/上市年龄/当日停牌不可选属于声明的逐日状态，不能被包装成补数失败而事后删股。

本机国金QMT自带Python3.6.8成功导入SDK，6类账户查询接口存在；没有连接账户、订阅或下单。研究环境沿用原WorkBuddy Python。需要用户明确实际交付平台并在本机登录模拟账户、说明沪深/创业板/科创板/北交所权限；无需发送密码或令牌。尚缺券商实盘后端、实际信号/订单/成交/资产对拍、独立PIT修订和公司行动主证据，以及冻结后的未来E5。

本次JQData只读SDK与官方HTTPS通道检查结果：{json.dumps(jq_access,ensure_ascii=False)}。没有据旧过期日期断言当前权限，也没有把数据服务凭据认证当成策略平台账户对拍。首轮SDK参数未正确启用超时及后续重复目标拒绝均保留；修复了本机SDK参数须分别设置的问题，未续费或提交策略。

直接HTTPS路径（仅本次会话关闭环境代理，不改全局设置）结果：{json.dumps(jq_direct,ensure_ascii=False)}。原SDK/HTTP结果及对应源文件保留，传输故障与实际权限返回分开记录。

Tushare当前查询不能证明历史修订档案，也不能证明本账户碎股、税负、退市权益已到账。具体来源与平台协助项见DATA_REQUIREMENTS.json。因子正式JQ仍要求适用E0–E4及有效独立数据回执；history_v2的E1≥120截面、每截面≥10股、配对覆盖≥0.80，6/12/24块3999次95%区间下界>0、正年度≥0.60、沪深各≥120且均值>0；0.03是功效规划效应，不是IC硬阈值。E3/E4需正常及两压力情景同窗净增量下界>0。本批工程/补数结果不授资格。

EXTERNAL_SOURCE_REQUESTS.csv进一步逐证券、逐日期列出未解决行动/报价/规则/停复牌证据需求，包含原观测证据哈希；数据源名称不能替代逐项证据。少数个股问题按根因办理，全局认证/平台依赖继续单独列示。

## 恢复与回退

受控人工执行、单一写入锁，未开定时或autonomous_writes，旧S5默认未改。补数可按内容身份恢复，输入/代码/口径变更须另建批次。停止新入口并用before恢复旧代码，新增模块不再引用；保留全部数据、事实及未决账户事件，切换运行目标前先对账。原始CSV及旧研究、旧证书、旧裁决未覆盖。
'''
    with (OUT/'REPORT.md').open('x',encoding='utf-8') as handle:handle.write(report)
    research=ROOT/'research/fuxi_v1_20261003/ALL_A_ENGINEERING_REVIEW.md';research.parent.mkdir(parents=True,exist_ok=True)
    with research.open('x',encoding='utf-8') as handle:
        handle.write(report+'\n\n逐字节输入、代码和结构化结果以同批reports资产为准。\n')
    note=f'''## AU. 伏羲V1.0 rc7全市场补数与十轮核验（2026-10-03）

- 普通A股{summary['instrument_count']}只，全史生命周期{len(full['cohort']['members'])}只；1393会话保留退市历史，当前观察并不成为PIT修订认证。
- {tests['tests']}测试通过，最终市价/限价合成合同{checks['instrument_rounds']}股票轮次；真实无事件三会话诊断10轮共{history['instrument_rounds']}声明轮次，10轮窗口可识别{summary['instruments_with_ten_identifiable_diagnostic_windows']}只。原始/V2合同保留，最终为contracts_v3。
- 密集四通道全窗齐全{audit['dense_core_complete_instruments']}只，名称/事件/计价代理齐全{audit['data_complete_under_vendor_proxy_instruments']}只；完整主证据认证0。短窗全市场12探针search_allowed={smoke['search_allowed']}，formal为False；不能写成六年全账户通过。
- 支持所有板块数量及显式市价上限、北交所248代码时钟、实际换手率/股本单位、前向禁入且保留持仓应收；不明事件/碎股/退市后确权与停牌冲突继续阻断。
- 前向临时记录{excluded}，不改历史。QMT自带3.6 SDK导入成功但未登录连接；平台选择/账户权限、原始披露修订及权益主证据、真实对拍/券商后端、未来E5仍缺。正式JQ/平台策略运行/实盘均0。
- reports/fuxi_v1_all_a_20261003/REPORT.md及DATA_REQUIREMENTS.json为本批事实；research/fuxi_v1_20261003/ALL_A_ENGINEERING_REVIEW.md。软件1.0.0-rc7，账本版本以新registration为准；controlled_manual、关闭自动写入、旧S5默认保留。'''
    append_native(ROOT/'.workbuddy/memory/_fuxi_landing_details.md',note)
    append_native(ROOT/'.workbuddy/memory/2026-10-03.md',note.replace('## AU.','##'))
    proposals=ROOT/'data/stage1_optimization_proposals.json';raw=proposals.read_bytes();items=json.loads(raw)
    require(not any(p['proposal_id']=='P-20261003-001' for p in items),'proposal already exists')
    items.append({'proposal_id':'P-20261003-001','title':'伏羲V1.0 rc7全市场数据接线、板块交易合同及十轮核验',
        'category':'全市场研究工程','trigger':'用户授权全A股执行交付修复、必要Tushare补数、特殊个股前向临时剔除。',
        'current_state':'原研究入口偏主板样本；缺全市场数据目录和独立逐股票交付状态。',
        'proposed_change':'全市场生命周期census、分片补数与四通道审计；板块数量/市价限价/代码时钟；真实换手和股本；前向禁入且保留权益；逐股票10轮诊断。',
        'expected_impact':'补齐可运行的全市场工程入口，明确仍缺主证据/真实平台/前向条件；不制造因子资格。',
        'implementation_scope':'fuxi_v1、scripts/fuxi_all_a*20261003.py、本批reports及research；旧S5默认不改。',
        'landed_detail':f"{tests['tests']}测试，{summary['instrument_count']}股票合成与真实行情各10轮；密集通道齐全{audit['dense_core_complete_instruments']}只，完整代理{audit['data_complete_under_vendor_proxy_instruments']}只，正式JQ/平台运行/实盘0。",
        'generator_hint':'manual','risk_level':'controlled_research_only','rollback_plan':'停止全市场新入口，用before恢复旧代码；新增模块不引用，全部数据/历史事实保留，原账户先对账。',
        'jq_validation_required':True,'jq_validation_status':'not_run_actual_account_and_formal_data_missing',
        'references':['reports/fuxi_v1_all_a_20261003/REPORT.md','reports/fuxi_v1_all_a_20261003/DELIVERY_SUMMARY.json',str(research.relative_to(ROOT))],
        'cross_verified':False,'verification_scope':'vendor-current full-census audit and native arithmetic; no independent whole-account or actual platform certification',
        'status':'landed','source':'UserAuthorized','proposal_change_kind':'behavior','domains':['T1闸门/裁决','T2数据口径','T3模块/通道']})
    eol='\r\n' if raw.count(b'\r\n')>=raw.count(b'\n')-raw.count(b'\r\n') else '\n'
    proposals.write_bytes((json.dumps(items,ensure_ascii=False,indent=2)+'\n').replace('\n',eol).encode('utf-8'))
    print(json.dumps({'report':str(OUT/'REPORT.md'),'proposal':'P-20261003-001','formal_JQ':0,'live':0}),flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):main()
