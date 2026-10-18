"""Append user delivery clarification and source findings; no runtime gate changes."""
import json
import runpy
import shutil
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.artifacts import code_manifest
from fuxi_v1.contracts import load_json,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_manual_handoff_20261003'


def new_json(path,body):
    with path.open('x',encoding='utf-8') as handle:json.dump(body,handle,ensure_ascii=False,indent=2)


def append_crlf(path,text):
    old=path.read_bytes();require(old.count(b'\n')==old.count(b'\r\n'),'ledger is not pure CRLF')
    require(text.splitlines()[0].encode() not in old,'duplicate note')
    with path.open('ab') as handle:handle.write(('\r\n\r\n'+text.replace('\n','\r\n')+'\r\n').encode('utf-8'))


def main():
    if (OUT/'MANUAL_DELIVERY_SCOPE.json').exists():
        require(not (OUT/'NOTE_VERIFICATION.json').exists(),'manual clarification already sealed')
        require((OUT/'REPORT.md').exists() and
            '## AV. 手动聚宽模拟→QMT交付与公开数据核验'.encode('utf-8') in (ROOT/'.workbuddy/memory/2026-10-03.md').read_bytes(),
            'partial note cannot resume without report and memory')
        return finish()
    previous=ROOT/'reports/fuxi_v1_all_a_20261003'
    require(code_manifest()==load_json(previous/'FINAL_VERIFICATION.json')['code_hashes'],'core changed since engineering verification')
    probe=load_json(OUT/'TUSHARE_SOURCE_PROBE.json');follow=load_json(OUT/'PUBLIC_SOURCE_FOLLOWUP.json')
    financial=load_json(OUT/'FINANCIAL_COLUMN_VERIFICATION.json')
    pdf=follow['issuer_notice'];require(pdf['state']=='downloaded_issuer_pdf' and file_hash(Path(pdf['path']))==pdf['sha256'],'notice PDF missing or changed')
    for r in financial['queries']:
        if 'query_path' in r:require(file_hash(Path(r['query_path']))==r['query_sha256'],'financial sample changed')
    before=OUT/'bookkeeping_before';before.mkdir()
    files=['data/version_registry.json','CHANGELOG.md','.workbuddy/memory/2026-10-03.md','.workbuddy/memory/_fuxi_landing_details.md']
    for name in files:shutil.copy2(ROOT/name,before/Path(name).name)
    scope={'schema_version':'fuxi.manual_delivery_scope.1','recorded_at':datetime.now(timezone.utc).isoformat(),
        'user_statement':'聚宽代码我手动提交就好，QMT也是我手动提交的（聚宽模拟交易生成交易信号发给QMT）',
        'delivery_flow':['system prepares code and signal contract','user submits code to JoinQuant simulation',
            'JoinQuant simulation emits signals','user submits receiver/execution code to QMT'],
        'local_direct_jqdata_authentication_required_for_code_delivery':False,
        'workspace_direct_broker_login_or_order_submission_required_for_code_delivery':False,
        'actual_platform_validation':'user-run signal/order/fill/cash/holding/rights exports remain necessary to verify runtime behavior',
        'qmt_sizing':'target weights or targets must be recomputed against actual QMT cash, holdings, settled shares and permissions; do not copy simulated account share quantities blindly',
        'implementation_gaps':['current jq_strategy compiler only permits historical backtest and uses frozen historical decisions',
            'forward daily signal-only simulation adapter and signal ingestion/reconciliation contract remain to be implemented and tested',
            'current local JQ compiler has no admitted BSE venue mapping; actual provider capability must be observed'],
        'data_dependency_principles':{
            'announcement_fulltext':'optional for formulas that do not use text; original implementation notices needed for ambiguous event terms',
            'financial_revisions':'complete financial revision dependency can be omitted for formulas using no financial fields; used financial fields require contemporaneously available values',
            'historical_hypothetical_accounts':'public event terms plus explicit settlement/tax/fractional models; no requirement for personal settlement records for stocks the user never owned',
            'actual_account_settlement':'actual held positions, entitlement credits, taxes and carried rights must be reconciled from account observations after execution starts',
            'unavoidable_market_contract':'price, tradability, corporate actions, held-position marks, cash and shares remain necessary',
            'prospective_exclusions':'unknown instrument/event can block future new entry under dated declared rules; carried holdings and entitlements remain on the account',
            'forward_vintages':'preserve raw observations and acquisition clocks; do not backdate a later restatement to the original report date'},
        'runtime_policy_changed':False,'formal_qualification_granted':False,
        'source_receipt_hashes':{p.name:file_hash(p) for p in (OUT/'TUSHARE_SOURCE_PROBE.json',OUT/'PUBLIC_SOURCE_FOLLOWUP.json',OUT/'FINANCIAL_COLUMN_VERIFICATION.json')}}
    new_json(OUT/'MANUAL_DELIVERY_SCOPE.json',scope)
    text='''# 手动交付范围与公开数据替代核验（2026-10-03）

用户确认：系统交付代码，由用户提交聚宽模拟策略和QMT执行代码；聚宽模拟生成信号发给QMT。因此本机JQData直连403、代理代登录或代理直接连券商，不是代码交付的前置条件。实际平台的权限及用户运行后导出的信号、订单、成交、资金、持仓、权益记录仍须验收。人工提交代码不保证聚宽模拟持仓和真实账户持仓相同，QMT需按实际可用资金、可卖数量与权限重算订单。

当前jq_strategy.py只支持历史backtest，决策来自冻结历史IR；目前不能把它称为可每日运行的模拟信号策略。下一实现需要面向未来会话的每日计算、信号有效期/身份/去重、QMT接收和实际账户对账。当前本地JQ编译器也没有已验收的北交所映射。此为范围澄清和审计发现，本批未改变运行代码或准入闸门。

## 来源核验

Tushare有anns_d公告索引，提供原PDF链接及rec_time，但需要单独权限。[官方公告接口](https://tushare.pro/document/2?doc_id=176)。现有凭据小范围查询返回40203、permission；不自动购买。原始实施公告可改从[巨潮资讯](https://www.cninfo.com.cn/new/index?lang=zh)、[上交所](https://www.sse.com.cn/disclosure/listedinfo/announcement/index.shtml)、[深交所](https://www.szse.cn/disclosure/notice/company/index.html)、[北交所](https://www.bse.cn/disclosure/announcement.html)检索。每条仍要核证券身份、实施日期、条款与原文，网页存在不证明已完成全量补证。

Tushare利润表有ann_date、f_ann_date、update_flag及调整前报表类型5。[官方利润表](https://tushare.pro/document/2?doc_id=33)、[修正标志说明](https://tushare.pro/document/1?doc_id=122)。实测600000.SH、2021-2026公告窗正常合并报表返回33条，update_flag包含0/1；类型5成功空返回。只证明该股票该查询的返回，不推断所有公司和所有历史版本齐全。首次诊断误写net_profit；正确字段为n_income，旧失败与修正后成功均保留，不把客户端字段错误归因于供应商缺数据。

Tushare dividend已有登记、除权、派息、红股上市等结构化字段，[官方分红接口](https://tushare.pro/document/2?doc_id=103)。它们描述上市公司安排。某账户实际持有变更、红利税和权益到账是账户数据；[中国结算证券查询服务](https://www.chinaclear.cn/zdjs/scxfw/201306/197c82ddec6a4dc58d5b953f19339c7d.shtml)说明相关个人查询需要网络用户登录。无需为历史假设账户索取用户从未持有证券的个人记录；真实执行后，只对实际账户的持仓、权益和交易进行核对。

## 已取得的公开原文样本

保存300980祥源新材公告2026-054原PDF，SHA256见PUBLIC_SOURCE_FOLLOWUP.json。发行人明确：每10股转增4股，无现金；登记9月29日、除权/流通/记入账户9月30日。这可以补充供应商空字段的条款证据，不能用空值自行推断零现金。回购账户不参加分配，所以符合条件股东的每股转增0.4与公司总体除权参考比例0.3901769不同；不能混用。碎股按尾数排序，同尾数随机，公开原文不能确定具体账户的不足一股分配结果。

该样本只保存事实与出处，未导入生产行动裁决，也未授予全市场数据认证或因子JQ资格。样本公告原件的公共镜像见[发行人实施公告PDF](https://file.finance.sina.com.cn/211.154.219.97%3A9494/MRGG/CNSESZ_STOCK/2026/2026-9/2026-09-22/12616031.PDF)。

## 可以不使用的数据及边界

| 数据 | 可免用的范围 | 保留的要求 |
| --- | --- | --- |
| 全量公告文本 | 因子不使用公告文本时，可免全量文本因子库 | 涉及不明公司行动时核实施条款；文本/公告因子核真实可得时间 |
| 完整财报修订库 | 因子无财务输入时，可免财务修订依赖 | 已使用财务值须按当时披露版本选择，不能把最新修正值倒填历史 |
| 用户历史结算记录 | 历史假设回测无需从未持有股票的个人记录 | 公开事件条款、明确税/碎股/结算模型；真实持仓执行后按实际账户对账 |
| 行情、停复牌、除权/送转、持仓计价 | 资金与股票账本有相关依赖时不可免 | 无报价不得补零或删掉持仓；复权价不能替代实际股数和现金结算 |

量价因子能缩小财务修订依赖，但价格会受公司行动影响，仍需事件及复权口径。近期窗口从现金开始的诊断可免窗口前未持有的权益延续，必须声明起始状态。无法获得的特殊事件可按事前规则阻断新进入，原持仓/应收保留并继续审理。不能用今天知道的失败名单回删过去研究样本。

当前投资数据闸门仍要求完整独立主证据回执；本批没有改动它。按实际因子字段和账户用途拆分来源要求，是后续版本的具体设计方向，应先冻结依赖、模型假设和证据覆盖，再验证读数与裁决。保守延迟不能让财报修正值自动变成历史当时可知值。统计及经济门槛、已暴露历史和未来验收要求保留。
'''
    with (OUT/'REPORT.md').open('x',encoding='utf-8') as handle:handle.write(text)
    research=ROOT/'research/fuxi_v1_20261003/MANUAL_HANDOFF_AND_PUBLIC_SOURCES.md'
    with research.open('x',encoding='utf-8') as handle:handle.write(text)
    note='''## AV. 手动聚宽模拟→QMT交付与公开数据核验（2026-10-03）

- 用户确认手动提交两端代码，聚宽模拟出信号、QMT执行。本机JQData认证不作为代码交付前置；真实运行信号/成交/账户权益对拍仍需用户导出。当前生成器仅历史backtest，前向模拟信号适配仍待实现；北交所JQ映射未验收。
- Tushare anns_d有原文URL/rec_time但现有凭据40203权限不足；公开交易所/巨潮及发行人原PDF可补。已保存300980公告2026-054：转增0.4、现金明确0、9月30日记股；公司总体除权比例0.3901769不同，碎股随机不能确定单账户结果。未导入生产裁决。
- 正确income字段n_income；首次net_profit错误造成合同拒绝，不能归因供应商。600000正常合并报表33条、flag0/1；类型5成功空，不能推全历史版本齐全。
- 量价策略可免财务修订依赖，非文本因子可免全量公告文本；历史假设账户不要求用户未持有股票的个人结算。实际持仓权益/行情/事件/计价仍必需。当前主证据/JQ闸门未改；后续须按依赖与用途设计并验证。reports/fuxi_manual_handoff_20261003/REPORT.md及MANUAL_DELIVERY_SCOPE.json为本批事实。纯研究补记，v0.7.121/core rc7不变。'''
    append_crlf(ROOT/'.workbuddy/memory/_fuxi_landing_details.md',note)
    append_crlf(ROOT/'.workbuddy/memory/2026-10-03.md',note)
    registry_path=ROOT/'data/version_registry.json';registry=load_json(registry_path)
    require(registry['versions'][-1]['version']=='v0.7.121','version changed during note')
    registry['versions'][-1].setdefault('non_bump',[]).append('2026-10-03手动聚宽模拟→QMT范围确认、Tushare公告/财务权限实测及公开实施公告样本；无运行闸门变更，详见reports/fuxi_manual_handoff_20261003/REPORT.md。')
    registry_path.write_bytes((json.dumps(registry,ensure_ascii=False,indent=1)+'\n').replace('\n','\r\n').encode())
    finish()


def finish():
    previous=ROOT/'reports/fuxi_v1_all_a_20261003'
    research=ROOT/'research/fuxi_v1_20261003/MANUAL_HANDOFF_AND_PUBLIC_SOURCES.md'
    files=['data/version_registry.json','CHANGELOG.md','.workbuddy/memory/2026-10-03.md','.workbuddy/memory/_fuxi_landing_details.md']
    registry_path=ROOT/'data/version_registry.json';registry=load_json(registry_path)
    require(registry['versions'][-1]['version']=='v0.7.121' and
        any('手动聚宽模拟→QMT' in n for n in registry['versions'][-1].get('non_bump',[])),'note not registered')
    for args in ('--bump','--render','--check'):
        sys.argv=['scripts/version_bump.py',args]
        try:runpy.run_path(str(ROOT/'scripts/version_bump.py'),run_name='__main__')
        except SystemExit as exc:require(exc.code in (None,0),'version tool failed')
    require(code_manifest()==load_json(previous/'FINAL_VERIFICATION.json')['code_hashes'],'note changed research core')
    after=load_json(registry_path);require(after==registry,'renderer changed version content beyond this note')
    artifacts={str(p.relative_to(ROOT)):file_hash(p) for p in OUT.rglob('*') if p.is_file()}
    artifacts[str(research.relative_to(ROOT))]=file_hash(research)
    for name in files:
        blob=(ROOT/name).read_bytes();require(blob.count(b'\n')==blob.count(b'\r\n'),'ledger line endings changed')
        artifacts[name]=file_hash(ROOT/name)
    new_json(OUT/'NOTE_VERIFICATION.json',{'version':'v0.7.121','runtime_code_unchanged':True,
        'qualification_or_safety_gates_changed':False,'artifacts':artifacts,'note_runner_sha256':file_hash(Path(__file__))})
    print(json.dumps({'version':'v0.7.121','core_unchanged':True,'report':str(OUT/'REPORT.md')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):main()
