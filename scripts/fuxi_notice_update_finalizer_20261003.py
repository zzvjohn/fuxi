"""Prepare final rc10 reporting without touching sealed evaluations."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'scripts/fuxi_notice_finalize_20261003.py'
s=p.read_bytes().decode('utf-8');assert '\r\n' not in s
s=s.replace("OUT/'tests_v3/TEST_RESULT.json'","OUT/'tests_v4/TEST_RESULT.json'")
s=s.replace('PUBLIC_IMPLEMENTATION_BUNDLE_v2.json','PUBLIC_IMPLEMENTATION_BUNDLE_v3.json')
s=s.replace('新结果位于ten_rounds_v2，代码与测试、窗口、成本和全部尝试各自绑定哈希。',
    'rc9第一组十轮位于ten_rounds_v2，代码与测试、窗口、成本和全部尝试各自绑定哈希；完成后保留原件。')
insert='''## 首组结果与自我修复复跑

首组rc9十轮已真实完成：候选可识别{prior['identified_scenarios']}/30，E1通过{prior['e1_pass']}/10，E1且三情景净增量下界全正{prior['e1_and_three_scenario_increment_positive']}/10，正式JQ {prior['formal_JQ_qualifications']}。保留其OUTCOME与全部冻结代码、账户及失败路径。15笔定点源补查均成功后发现：688766的2025-11-26无成交行情由发行人停牌公告明确解释；300192/301227的零现金条款跨审批与实施章节，其中301227还被PDF页码切断。

rc10关联同一年度明确零现金、未变股本和同一股东转增比例，跨页处理只移除“不/派发现金红利”间明确独立页码，不删除任意数字。两份实施公告不把缺现金字段补零，不以整体除权比替代股东比例。688766停牌起始日原PDF/text/公告索引逐项哈希核验并再次从PDF提取，计划绑定postflight_halt_688766/BUNDLE.json；只确认2025-11-26这一天，不从“预计不超过10日”推导后续停牌。新增issuer_halt计价参考仍是声明估值；特征原始报价继续缺失、股票不得当日新进入，源冲突仍阻断。独立数据认证不得把声明停牌参考计入原始价格证据。

最终第二组十轮在新目录ten_rounds_v3使用相同10个公式、窗口、基准和成本，重新构建全部输入与账户。本批共20次固定候选验收执行、10个唯一公式，不是20个独立因子发现或20个LLM生成循环。第二组最终结果如下。

'''
assert s.count('## 十轮范围与真实结果')==1;s=s.replace('## 十轮范围与真实结果',insert+'## 十轮范围与真实结果')
s=s.replace('见ten_rounds_v2/rounds及TEN_ROUND_ACCOUNT_BLOCKERS.csv','见ten_rounds_v3/rounds及TEN_ROUND_ACCOUNT_BLOCKERS.csv')
s=s.replace("'title':'伏羲V1.0 rc9已购公告原文接入与十轮全市场候选验收'", "'title':'伏羲V1.0 rc10已购公告原文接入与两组十轮全市场候选验收'")
s=s.replace('现金分红全停牌参考与复牌衔接修复；原固定缺口公告采集；全生命周期十轮原生统计',
    '现金分红全停牌参考与复牌衔接修复；审批/实施同年度零现金关联及跨PDF页码修复；发行人单日全停牌证据导入；原固定缺口公告采集；全生命周期两组十轮原生统计')
s=s.replace('fuxi_v1 public_notices/action_evidence/historical_adapter/simulation;',
    'fuxi_v1 public_notices/action_evidence/historical_adapter/simulation/halt_evidence/certification;')
s=s.replace('## AW. 伏羲rc9已购买公告补全与十轮全市场验收','## AW. 伏羲rc10已购买公告补全与两组十轮全市场验收')
s=s.replace('core 1.0.0-rc9，登记','core 1.0.0-rc10，登记')
s=s.replace('及ten_rounds_v2/OUTCOME.json为本批事实','及ten_rounds_v3/OUTCOME.json为本批事实；首组ten_rounds_v2保留')
s=s.replace("'core_version':'1.0.0-rc9'", "'core_version':'1.0.0-rc10'")
s=s.replace("'core_hashes':code_manifest(),'round_count':10,", "'core_hashes':code_manifest(),'round_count':10,'total_fixed_evaluation_executions':20,\n       'first_ten_outcome_sha256':file_hash(OUT/'ten_rounds_v2/OUTCOME.json'),\n       'halt_bundle_sha256':file_hash(OUT/'postflight_halt_688766/BUNDLE.json'),")
compile(s,str(p),'exec');p.write_bytes(s.encode('utf-8'))
print('rc10 finalizer references final tests, source bundle and both ten-round histories')
