"""Report actual ten-round results, append history and register the implementation."""
import csv
import json
import runpy
import sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,require,digest
from fuxi_v1.datasets import file_hash
from fuxi_v1.artifacts import code_manifest
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_announcements_20261003'
TEN=OUT/'ten_rounds_v4'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:json.dump(value,handle,ensure_ascii=False,indent=2)


def append(path,text):
    blob=path.read_bytes() if path.exists() else b''
    require(blob.count(b'\n')==blob.count(b'\r\n'),'history is not uniform CRLF')
    with path.open('ab') as handle:handle.write(('\r\n\r\n'+text.strip().replace('\r\n','\n').replace('\n','\r\n')+'\r\n').encode('utf-8'))


def fmt(value):
    if value is None:return '未知'
    return f'{value:.6f}' if isinstance(value,float) else str(value)


def main():
    require(not (OUT/'FINAL_VERIFICATION.json').exists(),'campaign already finalized')
    outcome=load_json(TEN/'OUTCOME.json');tested=load_json(OUT/'tests_v5/TEST_RESULT.json')
    collection=load_json(OUT/'collection_v6/COLLECTION_RESULT.json');bundle=load_json(OUT/'PUBLIC_IMPLEMENTATION_BUNDLE_v3.json')
    prior=load_json(OUT/'ten_rounds_v2/OUTCOME.json')
    require(prior['round_count']==10,'first ten results not sealed')
    probe=load_json(ROOT/'deliverables/fuxi_v1_20261004/PROBE_RECEIPT_v2.json')
    require(file_hash(probe['path'])==probe['sha256'] and probe['actual_platform_runs']==0 and len(probe['local_fixture_checks'])==4,'manual probe artifact not verified')
    frozen=load_json(TEN/'FREEZE.json');plan=load_json(TEN/'PLAN.json');input_receipt=load_json(TEN/'INPUT_RECEIPT.json')
    require(outcome['round_count']==10 and len(outcome['rows'])==10,'ten rounds not complete')
    require(tested['successful'] and tested['code_hashes']==code_manifest()==frozen['code_hashes'],'tested and run code mismatch')
    require(digest(frozen)==outcome['freeze_hash'] and digest(plan)==frozen['plan_hash'],'result scope mismatch')
    require(file_hash(TEN/'market_input.json')==input_receipt['sha256'],'history input changed')
    require(file_hash(OUT/'PUBLIC_IMPLEMENTATION_BUNDLE_v3.json')==plan['issuer_action_evidence']['sha256'],'issuer bundle changed')
    require(file_hash(plan['issuer_halt_evidence']['path'])==plan['issuer_halt_evidence']['sha256'],'halt bundle changed')
    from fuxi_v1.halt_evidence import load_halt_starts
    require(len(load_halt_starts(plan['issuer_halt_evidence'],{'688766.XSHG'},['2025-11-26','2025-11-27','2025-11-28','2025-12-01','2025-12-02','2025-12-03','2025-12-04','2025-12-05','2025-12-08']))==9,'primary halt proof not consumed')
    require(file_hash(OUT/'collection_v6/COLLECTION_RESULT.json')==bundle['collection_sha256'],'collection changed')
    for row in outcome['rows']:
        path=TEN/'rounds'/('%02d.json.gz'%row['round']);meta=path.with_name('%02d.receipt.json'%row['round'])
        require(file_hash(path)==row['sha256'] and load_json(meta)==row,'round artifact changed')
    source_rechecks={}
    for name,info in plan['raw_sources'].items():
        actual=file_hash(info['path']);require(actual==info['sha256'],'original source changed '+name)
        source_rechecks[name]={'path':info['path'],'sha256':actual}
    require(collection['event_count']==1085,'event queue incomplete')
    for event in collection['events']:
        require(file_hash(event['path'])==event['sha256'],'event receipt changed')
    claims=Counter(b for row in outcome['rows'] for b in row['admission_blockers'])
    stage_counts=Counter((v['stage'],v['eval_status'],v['reason_code']) for r in outcome['rows'] for v in r['stage_readouts'])
    delivery_counts=Counter(r['jq_delivery']['status']+':'+r['jq_delivery'].get('reason','') for r in outcome['rows'])
    current_events=[r for r in bundle['readouts'] if plan['window']['start']<=r['ex_date']<=plan['window']['end']]
    window_notice_stats={'events_in_current_window':len(current_events),'resolved':sum(r['resolved'] for r in current_events),'unresolved':sum(not r['resolved'] for r in current_events),'historical_events_without_original':collection['event_count']-collection['events_with_original']}
    save(OUT/'WINDOW_NOTICE_STATS.json',window_notice_stats)
    roots=sorted({code for row in outcome['rows'] for code in row['root_gap_instruments']})
    supplemental=load_json(OUT/'postflight_sources/RESULT.json')
    require(supplemental['frozen_ten_round_results_unchanged'],'postflight source checks changed results')
    for item in supplemental['readouts']:
        require(file_hash(item['path'])==item['sha256'],'supplemental source record changed')
    gap_rows=[];risk_rows=[]
    from fuxi_v1.historical_study import _load_gzip
    from fuxi_v1.statistics import performance
    for row in outcome['rows']:
        result=_load_gzip(TEN/'rounds'/('%02d.json.gz'%row['round']))
        for role,key in [('candidate','scenario_accounts'),('frozen_baseline','scenario_baseline_accounts')]:
            for scenario,value in result[key].items():
                account=value['account'];reading=performance(account['daily']) if account['eval_status']=='valid' else None
                risk_rows.append({'round':row['round'],'factor':row['name'],'account_role':role,'scenario':scenario,
                    'eval_status':account['eval_status'],'reading':{k:v for k,v in reading.items() if k!='returns'} if reading else None})
                for gap in value['account']['gaps']:
                    if gap.get('instrument'):gap_rows.append(dict(round=row['round'],factor=row['name'],account_role=role,scenario=scenario,**gap))
    fields=sorted({k for g in gap_rows for k in g}) or ['round','factor','scenario','session','instrument','reason']
    with (OUT/'TEN_ROUND_ACCOUNT_BLOCKERS.csv').open('x',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(gap_rows)
    save(OUT/'RISK_DIAGNOSTICS.json',{'scope':'descriptive identified-account cumulative return, volatility and drawdown; not a new admission rule',
        'rows':risk_rows,'source_outcome_sha256':file_hash(TEN/'OUTCOME.json'),'formal_assurance':False})
    actions={'exact_account_blocking_instruments':roots,'admission_blocker_counts':dict(claims),
       'priority':[
        {'order':1,'action':'Resolve the exact held-account event/mark gaps in TEN_ROUND_ACCOUNT_BLOCKERS.csv; separate absent sources, parser ambiguity and actual fractional allocations; preserve existing holdings and receivables.',
         'depends_on':'non-identifiable scenario accounts, if any; no economic rejection inferred from missing NAV'},
        {'order':2,'action':'Review primary data evidence for the actual formula and account dependencies; financial archives are needed only for financial inputs; generator cannot certify its own investment evidence.',
         'depends_on':'independent data receipt remains absent; public parser output is not a certification'},
        {'order':3,'action':'For identifiable accounts, use the three-scenario net increment and single-channel ablation to decide further mechanism research; preregister turnover/portfolio changes before evaluation; preserve statistical/economic thresholds.',
         'depends_on':'E1 and cost/capacity results; bounded search rather than unbounded formula generation'},
        {'order':4,'action':'Implement and verify the manual JoinQuant forward signal-only adapter and QMT signal ingestion using actual account state; verify BSE provider mapping; user submits both codes.',
         'depends_on':'local generator currently historical backtest only; platform exports not yet observed'},
        {'order':5,'action':'Freeze eligible candidates and their future acceptance protocol, then collect genuinely subsequent observations and actual platform parity.',
         'depends_on':'historical 2025/2026 observations cannot replace future E5'}],
       'stage_verdict_counts':[{'stage':k[0],'status':k[1],'reason':k[2],'count':v} for k,v in sorted(stage_counts.items())],
       'postflight_source_result_sha256':file_hash(OUT/'postflight_sources/RESULT.json'),
       'jq_delivery_counts':dict(delivery_counts),'current_window_notice_stats':window_notice_stats,
       'no_additional_retrospective_exclusions':True,'formal_JQ':outcome['formal_JQ_qualifications'],'live':0}
    save(OUT/'NEXT_ACTIONS.json',actions)
    next_protocol={'schema_version':'fuxi.next_mechanism_design.1','status':'planned_not_executed',
        'basis_outcome_sha256':file_hash(TEN/'OUTCOME.json'),'basis_is_research_exposed':True,
        'candidate_names':[r['name'] for r in outcome['rows'] if r['e1_decision']=='pass'],
        'research_question':'Do IC-positive candidates add net value after reducing concentration, without changing costs or granting formal source assurance?',
        'entry_condition':'Resolve or explicitly model every held-path quantity/mark gap; preflight must be identifiable; no historical deletion of blocked holdings.',
        'portfolio_axis':{'control':{'top_n':10,'max_weight':.1},
                          'arms':[{'top_n':30,'max_weight':1/30},{'top_n':50,'max_weight':1/50}],
                          'linked_fields':'top_n and equal-weight ceiling are one concentration axis; candidate and baseline matched'},
        'unchanged_contract':['same candidate AST and direction','frozen momentum20 baseline','50/50 rank blend',
            '5-session horizon and rebalance','same calendar/PIT scope','normal/double-cost/half-capacity','same original E1 and economic gates'],
        'max_new_portfolio_arms':2*outcome['e1_pass'],'external_LLM_requests':0,'threshold_relaxation_allowed':False,
        'attribution_required':'single-channel ablation, section pairing, and existing price-CV/momentum style attribution; positive IC alone is insufficient',
        'uncertain_allotments':'uniform floor/ceil runs are declared sensitivities, not bounds on all dynamic future account paths; no fabricated point allocation',
        'account_branch_design':{'status':'planned_not_implemented',
            'choice_scope':'only source-supported fractional rules whose entitlement is floor or ceil of the recorded hypothetical holding times holder ratio',
            'method':'branch at each actual fractional event; recompute later holdings, fees, NAV, orders and later entitlements separately; keep baseline/cost/window matched',
            'max_branches_per_candidate_scenario':128,'overflow_status':'data_insufficient; incomplete enumeration cannot claim a bound',
            'other_source_gaps':'unsupported amounts/marks do not become bounded merely by enumerating fractional choices',
            'economic_reading':'use the worst paired-increment lower confidence endpoint across fully enumerated paths as a research robustness diagnostic',
            'formal_scope':'branch robustness does not identify an actual investor allocation or replace independent data/platform/forward evidence'},
        'retention':'all requests, arms, failures and valid pairs retained; not stop-at-first-positive',
        'inference':'existing-window results remain nominal research diagnostics; portfolio variants do not repair discovery exposure',
        'future_acceptance':'freeze candidate, data timing, costs, trials and multiplicity control before subsequent observations; E5 uses genuinely later samples and actual platform parity',
        'new_announcement_channel':'retain actual acquisition/publication evidence; use after availability and next tradable session; rec_time backfill is not initial availability'}
    save(OUT/'NEXT_RESEARCH_PROTOCOL.json',next_protocol)
    save(OUT/'MANUAL_JQ_PROBE.json',probe)
    probe_note=f"已生成供用户手动提交的只读聚宽覆盖探针 `{probe['path']}`（哈希 `{probe['sha256']}`），普通A股范围按冻结LOCAL_COVERAGE筛选，预期仍上市{probe['expected_live_instruments']}只、行情样本{probe['sample_count']}只。四类本地夹具检查涵盖完整分母、真实返回后缀、幂等执行、歧义及失败保留、未来访问阻断和全局服务故障分辨；平台实际运行0。策略初始化后在收盘后运行一次，导出fuxi_jq_coverage_probe_20261004.json，用户返回此文件或实际日志即可继续核验映射。探针没有订单接口，也不替代前向因子策略或完整账户认证。"
    rows=[]
    for r in outcome['rows']:
        rows.append('| '+ ' | '.join([str(r['round']),r['name'],fmt(r['rank_ic']),str(r['valid_ic_sections']),
            r['e1_decision'],str(r['identified_scenarios'])+'/3',str(r['paired_identified_scenarios'])+'/3',
            ', '.join(fmt(r['net_increment_lower'][s]) for s,_,_ in [('normal',1,1),('double_cost',2,1),('half_capacity',1,.5)]),
            str(r['jq_candidate_allowed'])])+' |')
    report=f'''# 伏羲V1.0 rc11：已购买公告权限补全与两组十轮全市场验收

完成时间：{datetime.now(timezone.utc).isoformat()}。本轮实际新增正式JQ资格{outcome['formal_JQ_qualifications']}，实盘资格0。整体验收的软件/纸面发布裁决：{json.dumps(outcome['readiness'],ensure_ascii=False)}。

## 公告权限与数据补全

新凭据请求成功，ENTITLEMENT_PROBE.json返回3条300980公告并包含正式实施PDF链接。旧40203保留为当时请求的权限状态观察，本轮以新请求成功为准。[Tushare公告接口](https://tushare.pro/document/2?doc_id=176)提供索引、URL及可选rec_time；购买接口不等于账户结算或历史全部修订已被证实。

原审计固定1,085个事件，2021-01-04至2026-09-30；每事件查前60日至后7日（止于9月30日）的完整索引。实际状态{json.dumps(collection['states'],ensure_ascii=False)}，下载{collection['downloaded_documents']}个不同URL的实施原文，覆盖{collection['events_with_original']}个事件。PDF、文本、请求及实际采集时间保留并绑定哈希。完整索引保留，不从公告标题推断验收通过。初版泛化“实施”筛选及后续传输/原文筛选调整均保留STOP与原件，没有覆盖历史。

已修正北交所历史公告查询：供应商公告索引可能仍用原83xxxx代码，不能只以现在920xxx查询。冻结官方映射目录，对每个已映射事件分别查询新旧索引，保留实际请求及原始ts_code；原文证券身份按公告日与映射生效日另行核对。重叠窗口按证券合并后仍按每个冻结事件前60日至后7日裁剪，已完成V5请求按路径/哈希复用。探针832000.BJ返回113行并有2024-05-29实施原文，证明此前的920000空返回不能解释为无公开源。部分同一ann_date/PDF的rec_time为2025/2026后补录且另有空值，不能当历史首次可得时间；实际时钟计数{json.dumps(bundle['publication_index_clock_diagnostics'],ensure_ascii=False)}。本批仅复算历史账户公司行动，没有用公告文本同日或历史补录时间制造alpha。

已将{bundle['resolved_existing_gap_events']}个原阻断事件转为可复算的普通条款；全史条款当前状态{json.dumps(bundle['current_status_counts'],ensure_ascii=False)}。其余{1085-bundle['resolved_existing_gap_events']}个固定事件仍待审理，原因{json.dumps(bundle['unresolved_event_reasons'],ensure_ascii=False)}。本轮窗口内固定缺口结果为{json.dumps(window_notice_stats,ensure_ascii=False)}。初次复算曾误把供应商全表5,911行当作普通A股5,909行断言，检查在任何条款导入前停止；已改用冻结LOCAL_COVERAGE范围，失败原件和原因保留于adjudicator_attempt_1。解析失败、公开条款有歧义和供应商无数据不能混成同一个“无法补数”。

条款要求发行人身份、实际实施方案、含税金额、登记/除权/派现/记股/流通日期与供应商组件核对；导入时再次从原文复算并核验文件哈希。空现金不能补零；转增与公司总体除权参考比例分开；合计现金需公开明确且与供应商组件一致。零碎股按尾数排序/随机的账户结果继续未知，默认block。假设历史账户按自身登记日收盘持仓计权益，无需用户未曾持有股票的个人结算记录。真实账户结算仍需实际日志。

原文及代理条款不是独立投资认证，本轮独立数据回执0。只补本轮声明来源，原价格/复权/限制/停复牌/股票原始数据未改，见FINAL_VERIFICATION.json。

## 停牌期间分红计价修复

旧ten_rounds冻结批次在12条预检后停止，未计入十轮完成数，STOP_FOR_CORRECTION保留。603221于2026-08-12在全日停牌期间每股现金分红0.03元，8月18日复牌；旧冻结价未扣除含税现金且原始末次价格比较把复牌参考变化误判未知。rc9对可支持的现金条款复算声明停牌除息参考，并逐笔保存halt_reference_adjustments.csv，后续仍无原始观测时只沿用该声明参考；不回填原始字段、不借未来报价。单次/多次分红回归验证净值仅承受声明税额；未知/送转参考继续不可识别。rc9第一组十轮位于ten_rounds_v2，代码与测试、窗口、成本和全部尝试各自绑定哈希；完成后保留原件。

## 首组结果与自我修复复跑

首组rc9十轮已真实完成：候选可识别{prior['identified_scenarios']}/30，E1通过{prior['e1_pass']}/10，E1且三情景净增量下界全正{prior['e1_and_three_scenario_increment_positive']}/10，正式JQ {prior['formal_JQ_qualifications']}。保留其OUTCOME与全部冻结代码、账户及失败路径。15笔定点源补查均成功后发现：688766的2025-11-26无成交行情由发行人停牌公告明确解释；300192/301227的零现金条款跨审批与实施章节，其中301227还被PDF页码切断。

rc10关联同一年度明确零现金、未变股本和同一股东转增比例，跨页处理只移除“不/派发现金红利”间明确独立页码，不删除任意数字。两份实施公告不把缺现金字段补零，不以整体除权比替代股东比例。688766停牌起始日原PDF/text/公告索引逐项哈希核验并再次从PDF提取，计划绑定postflight_halt_interval_688766/BUNDLE.json；rc10旧尝试仅确认2025-11-26，12预检中反转5账户缺口因此移至2025-11-27；已记录ten_rounds_v3/STOP_FOR_INTERVAL_CORRECTION并停止，完成候选轮数0，未与两组完整十轮混计。补归档2025-12-09复牌原文后，rc11只使用两份原文共同覆盖的2025-11-26..2025-12-08九个实际交易日，不从预计时长推导。起始日原文为11月26日，复牌原文追述为11月25日，一日冲突保留，争议的11月25日不由该包补证。停止日12月8日与复牌日12月9日来自明确表格/条款；后续公告用于研究估值区间的回顾性确认，不声明各历史日独立PIT认证，不作为alpha输入。新增issuer_halt计价参考仍是声明估值；特征原始报价继续缺失、股票不得当日新进入，源冲突仍阻断。独立数据认证不得把声明停牌参考计入原始价格证据。

最终第二组十轮在新目录ten_rounds_v4使用相同10个公式、窗口、基准和成本，重新构建全部输入与账户。本批共20次固定候选验收执行、10个唯一公式，不是20个独立因子发现或20个LLM生成循环。第二组最终结果如下。

## 十轮范围与真实结果

全生命周期普通A股{outcome['instruments']}只，{plan['window']['start']}至{plan['window']['end']}，60会话热身；校准截止2025-05-30，研究自2025-06-11，5会话前向标签，保留窗内退市股票，不按本轮结果回删名单。2025/2026均已有研究暴露。十个固定公式只作为本轮候选/控制，重复执行与已见公式不计独立新发现；未调用外部LLM，未启动扩张型或自适应搜索。本轮运行路径为V1原生研究内核的候选验收，不是旧生产S1–S5调度、LLM自主生成及外部平台实测全链路十次运行。

基准为预先冻结动量20，候选与基准按50/50截面秩组合，long-only top10、周换。实际资金/股份账本，下一会话成交；成本合同为初始100万元、双边佣金0.03%且每笔最低5元、滑点5bps、卖出税0.05%、前一会话量1%容量；现金分红声明平坦20%税敏感性。正常、成本×2、容量×0.5同窗对比，不把盘后数据同日交易。旧生产S5的IR0.7/coverage0.90/COST_BPS15及默认开关未改；该生产配置与本轮原生资金账本成本合同分别记账。

{tested['tests']}项测试真实通过；十轮原生不变量通过{outcome['software_invariants_passed']}/10，候选资金账户可识别{outcome['identified_scenarios']}/30，与基准可识别且实际非零配对{outcome['paired_identified_scenarios']}/30，E1通过{outcome['e1_pass']}/10，E1且三成本/容量净增量区间下界全正{outcome['e1_and_three_scenario_increment_positive']}/10。每轮三个预先取定日期做每日/批量信号对拍，并进行未来字段扰动、现金/股份/NAV独立算术重放及故障夹具核验。真实平台运行0，不能把本地对拍写成聚宽或QMT对拍。

| 轮 | 公式 | Rank IC | 有效截面 | E1 | 候选可识别账户 | 候选/基准有效配对情景 | 正常/双成本/半容量净增量95%下界（日） | 正式JQ |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
{chr(10).join(rows)}

preflight search_allowed={outcome['preflight_search_allowed']}。即使预检阻断扩搜，十个预先冻结的验收夹具仍记录其阻断路径，未消耗扩搜预算。账户未可识别时收益及区间留未知，不能归因经济失败。逐轮E0–E5、单通道剥离/截面对配、持仓/应收与精确阻断日期见ten_rounds_v4/rounds及TEN_ROUND_ACCOUNT_BLOCKERS.csv。

RISK_DIAGNOSTICS.json保存候选和冻结基准各三情景的累计收益、最大回撤、日收益波动与年化Sharpe，共60条账户读数；不可识别账户保留未知。表中的净增量是同日收益差的区间，与累计净值差是不同统计量；平均增量为正不等于区间下界为正，也不新增准入规则。

E1数值规则：至少120有效截面、每截面至少10股、配对覆盖至少0.80；6/12/24块各3999次重采样的95%区间下界大于0；具备至少120截面的年度中正IC年度占比至少0.60，沪深各至少120且均值大于0。0.03是功效规划效应，不是IC均值准入硬阈值。本轮E1区间是研究样本的名义95%诊断，multiple_testing_corrected=false，不是扩搜后的确认性错误率保证。正式准入还需要E2单通道剥离及增量、E3/E4正常/双成本/半容量同窗净增量下界均大于0，并有独立数据回执；未来E5及真实平台对拍另行验收。

## 精确阻断点的自动源补查

十轮完成后，仅对实际持仓/预检的源缺口进行15日前后窗口的daily、adj_factor、stk_factor、stk_limit及已购anns_d串行查询。实际物理请求{supplemental['physical_calls']}，查询{supplemental['queries']}，状态{json.dumps(supplemental['statuses'],ensure_ascii=False)}。原始补查行、读取时钟、响应哈希与实际阻断日期见postflight_sources/RESULT.json。空返回不推导零现金或零收益；新源观察不能覆写已冻结十轮，也未自行授予独立认证。若取得新行情/条款，应另行冻结接入与消费者重跑；如果已取得原文却只是解析遗漏，优先修解析关联。

## 干旱裁决与后续阻塞

正式资格仍按实际裁决字段计数，不能用原文已下载、单项IC为正或软件测试通过替代。本轮正式JQ={outcome['formal_JQ_qualifications']}。已识别账户中经济证据与资料未识别的路径分开；实际准入阻断计数{json.dumps(dict(claims),ensure_ascii=False)}；账户根因证券{json.dumps(roots,ensure_ascii=False)}。实际逐阶段裁决汇总{json.dumps([{'stage':k[0],'status':k[1],'reason':k[2],'count':v} for k,v in sorted(stage_counts.items())],ensure_ascii=False)}；JQ交付读数{json.dumps(dict(delivery_counts),ensure_ascii=False)}。这轮只检验固定公式集合，不证明整个可研究因子空间没有alpha。

优先处理逐日期实际持仓阻断及普通公告解析遗漏，再完成适用字段/账户范围的主证据复核。无财务输入的量价因子不要求未使用的财务版本；已有价格、停复牌、公司行动及持仓计价依赖仍需证据。可识别账户若IC通过但三情景增量不成立，下一轮预先冻结低换手/成本与容量设计、基准残差和单通道归因；固定预算、保留全部尝试及未来验证，保持原阈值。新公告文本/事件通道须按真实可得时间设计，新数据权限本身不证明新alpha。

用户手动提交聚宽/QMT代码的路线保持。现有JQ生成器仍仅历史backtest，本轮编译及平台障碍按jq_delivery实际返回记录；前向模拟信号适配、真实QMT账户目标换算/接收、北交所平台映射、真实信号/订单/成交/资产权益对拍、未来E5仍分别办理。无需助理登录或提交用户平台，也不能把本机JQData鉴权当代码交付前置。NEXT_ACTIONS.json给出顺序和依赖。

{probe_note}

NEXT_RESEARCH_PROTOCOL.json给出尚未执行的有界机制方案：从本轮E1通过的候选出发，候选和基准同改持仓分散轴，比较top30/top50与原top10，上限{2*outcome['e1_pass']}个新增组合臂；保持公式方向、50/50混合、5会话标签/换仓和三压力成本。进入实验前须处理实际持仓不可识别问题；零碎股统一floor/ceil只是声明敏感性，不能当所有动态账户路径的上下界。逐事件分支推演设计上限每候选/情景128条，后续权益与交易须沿各自路径重算，未枚举完整或遇未界定源缺口时继续data_insufficient；该设计尚未实现或运行。现有窗仍研究暴露，观察到正值也需独立主证据、真正后续样本及平台对拍，不能将提案写成实验已通过。

本轮没有新增事后剔除，原600077前向禁入记录保持；全部已有持仓、应收、退市历史及失败尝试保留。controlled_manual且autonomous_writes_enabled=false，不启用定时写入或实盘交易。
'''
    with (OUT/'REPORT.md').open('x',encoding='utf-8',newline='\n') as handle:handle.write(report)
    research=ROOT/'research/fuxi_v1_20261003/PAID_ANNOUNCEMENTS_AND_TEN_ROUNDS.md'
    research.parent.mkdir(parents=True,exist_ok=True)
    with research.open('x',encoding='utf-8',newline='\n') as handle:handle.write(report)
    proposal_id='P-20261003-002';p=ROOT/'data/stage1_optimization_proposals.json';blob=p.read_bytes();proposals=json.loads(blob)
    require(not any(r['proposal_id']==proposal_id for r in proposals),'proposal identity already used')
    proposals.append({'proposal_id':proposal_id,'title':'伏羲V1.0 rc11已购公告原文接入与两组十轮全市场候选验收',
        'category':'公告证据与工程验收','trigger':'用户公告权限已购买，授权补全并试跑十轮、定位因子干旱。',
        'current_state':'旧公告探针40203，普通转增空现金/冲突条款未由原文接入；现有平台交付仍仅历史回测。',
        'proposed_change':'源绑定普通实施条款提取、导入重算及哈希核验；现金分红全停牌参考与复牌衔接修复；审批/实施同年度零现金关联及跨PDF页码修复；发行人停复牌原文共同区间接入并保留日期冲突；原固定缺口公告采集；全生命周期两组十轮原生统计/账户/成本压力/对拍/准入读数；供用户手动提交的只读聚宽覆盖探针。',
        'expected_impact':'补公告证据并明确数据、软件、经济与平台阻断；不以权限或测试授资格。',
        'implementation_scope':'fuxi_v1 public_notices/action_evidence/historical_adapter/simulation/halt_evidence/certification; scripts/fuxi_notice*; 本批reports/research',
        'landed_detail':f"{tested['tests']}测试；{collection['event_count']}事件、{collection['downloaded_documents']}原文URL，普通缺口关闭{bundle['resolved_existing_gap_events']}；十轮可识别{outcome['identified_scenarios']}/30、E1 {outcome['e1_pass']}/10、正式JQ {outcome['formal_JQ_qualifications']}。",
        'generator_hint':'manual','risk_level':'controlled_research_only','rollback_plan':'停用新公告计划引用，按before_core恢复被改源文件，保留所有原始公告、补数与研究事实；现有持仓先对账。',
        'jq_validation_required':True,'jq_validation_status':'actual_platform_not_observed',
        'references':[str((OUT/'REPORT.md').relative_to(ROOT)),str(research.relative_to(ROOT))],
        'cross_verified':False,'verification_scope':'hash-bound public issuer/vendor comparison and native arithmetic; no independent whole-account or actual platform assurance',
        'status':'landed','source':'UserAuthorized','proposal_change_kind':'behavior','domains':['T1闸门/裁决','T2数据口径','T3模块/通道']})
    require(blob.count(b'\n')==blob.count(b'\r\n'),'proposal ledger line endings changed')
    p.write_bytes((json.dumps(proposals,ensure_ascii=False,indent=2)+'\n').replace('\n','\r\n').encode('utf-8'))
    for flag in ('--bump','--render','--check'):
        sys.argv=['scripts/version_bump.py',flag]
        try:runpy.run_path(str(ROOT/'scripts/version_bump.py'),run_name='__main__')
        except SystemExit as exc:require(exc.code in (None,0),'version workflow failed')
    registry=load_json(ROOT/'data/version_registry.json');version=registry['versions'][-1]['version']
    note=f'''## AW. 伏羲rc11已购买公告补全与两组十轮全市场验收（2026-10-03批次）

- 新anns_d请求成功，旧40203保留为过去权限观察。固定1085事件索引，下载{collection['downloaded_documents']}个正式原文URL、覆盖{collection['events_with_original']}事件；关闭普通条款缺口{bundle['resolved_existing_gap_events']}。现金空值不补零，股东比例与总股本参考不混用，碎股仍block；原始数据/失败与前轮均保留。
- 原生全市场{outcome['instruments']}只，2025-01-02..2026-09-30，研究自2025-06-11，窗内退市不回删，均研究暴露；10固定候选，不是10个独立新发现。{tested['tests']}测试，原生不变量{outcome['software_invariants_passed']}/10，账户{outcome['identified_scenarios']}/30，E1 {outcome['e1_pass']}/10，E1且三压力净增量下界全正{outcome['e1_and_three_scenario_increment_positive']}/10。正式JQ {outcome['formal_JQ_qualifications']}、真实平台/实盘0。
- software_ready={outcome['readiness']['software_ready']}，paper_release_ready={outcome['readiness']['paper_release_ready']}；精确持仓阻断、独立主证据、前向聚宽信号/QMT接收、北交所平台映射/实测、未来E5按NEXT_ACTIONS办理。公告购买不等于最后所有卡点已关闭。
- core 1.0.0-rc11，登记{version}，{proposal_id}；旧S5默认/阈值不改，controlled_manual、自动写入关闭。reports/fuxi_v1_announcements_20261003/REPORT.md及ten_rounds_v4/OUTCOME.json为本批事实；首组ten_rounds_v2保留；research/fuxi_v1_20261003/PAID_ANNOUNCEMENTS_AND_TEN_ROUNDS.md。'''
    append(ROOT/'.workbuddy/memory/_fuxi_landing_details.md',note)
    append(ROOT/'.workbuddy/memory/2026-10-03.md',note)
    from datetime import timedelta
    completion_date=datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    if completion_date!='2026-10-03':
        append(ROOT/'.workbuddy/memory'/f'{completion_date}.md',note+'\n\n本日志记录实际完成日；原批次仍为2026-10-03，详细结果按冻结范围复核。')
    ledger_files=['data/version_registry.json','data/stage1_optimization_proposals.json','CHANGELOG.md',
                  '.workbuddy/memory/2026-10-03.md','.workbuddy/memory/_fuxi_landing_details.md']
    if completion_date!='2026-10-03':ledger_files.append(f'.workbuddy/memory/{completion_date}.md')
    ledger={}
    for name in ledger_files:
        data=(ROOT/name).read_bytes();require(data.count(b'\n')==data.count(b'\r\n'),'ledger has bare LF '+name);ledger[name]=file_hash(ROOT/name)
    save(OUT/'REGISTRATION.json',{'version':version,'core_version':'1.0.0-rc11','proposal_id':proposal_id})
    save(OUT/'FINAL_VERIFICATION.json',{'completed_at':datetime.now(timezone.utc).isoformat(),'version':version,
       'runner_sha256':file_hash(Path(__file__)),'core_hashes':code_manifest(),'round_count':10,'total_fixed_evaluation_executions':20,
       'first_ten_outcome_sha256':file_hash(OUT/'ten_rounds_v2/OUTCOME.json'),
       'halt_bundle_sha256':file_hash(OUT/'postflight_halt_interval_688766/BUNDLE.json'),
       'manual_probe_sha256':probe['sha256'],'manual_probe_receipt_sha256':file_hash(ROOT/'deliverables/fuxi_v1_20261004/PROBE_RECEIPT_v2.json'),
       'freeze_hash':digest(frozen),'original_sources_unchanged':source_rechecks,'ledger_hashes':ledger,
       'report_sha256':file_hash(OUT/'REPORT.md'),'research_sha256':file_hash(research),
       'postflight_source_result_sha256':file_hash(OUT/'postflight_sources/RESULT.json'),
       'bundle_sha256':file_hash(OUT/'PUBLIC_IMPLEMENTATION_BUNDLE_v3.json'),'outcome_sha256':file_hash(TEN/'OUTCOME.json'),
       'formal_JQ':outcome['formal_JQ_qualifications'],'actual_platform_runs':0,'live':0,
       'controlled_runtime_policy_sha256':file_hash(ROOT/'codex_automations/fuxi_runtime_policy.json')})
    print(json.dumps({'version':version,'report':str(OUT/'REPORT.md'),'formal_JQ':outcome['formal_JQ_qualifications'],
        'software_ready':outcome['readiness']['software_ready'],'paper_release_ready':outcome['readiness']['paper_release_ready']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):main()
