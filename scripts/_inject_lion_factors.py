# -*- coding: utf-8 -*-
"""_inject_lion_factors.py — 向 NOVEL_FACTOR_LIBRARY 注入狮子锚点因子族

遵循 fuxi-factor-onboarding skill:
  - 用 src.index('\\n]', i) 定位顶层列表结束符 (不可用 rfind, 会被后续列表干扰)
  - 用 repr() 逐键输出 (不用 json.dumps().replace)
  - 写回前 compile() 自检
  - 按 'name' 幂等去重
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PATH = 'e:/quant/scripts/daily_factor_hypothesis.py'

ENTRIES = [
    {
        'id': 'novel_227',
        'name': 'lion_bias',
        'label': '涨停启动点贴近度(结构锚点)',
        'category': 'behavioral_anchor',
        'hypothesis': '现价贴近「最近一次涨停日之前最后一根阴线低点」的股票处于主力成本区, 下行空间有限且修复概率高',
        'logic': '事件锚点(非固定窗口): 最近涨停=主力发力信号; 其前最后一根阴线低点=启动点/成本下沿; '
                 '现价/启动点越低=越贴近成本区→结构型反转。来源: 聚宽社区《变种小狮子 V3.0》bias 构造, '
                 '口径改为后复权比值。实测 RankIC -0.0250 / 日ICIR -0.178 (过闸); '
                 '但与本库 rev20/rev60 截面相关 0.40~0.48, 正交后 IC 仅 +0.0090 → 主要是反转代理',
        'formula': 'lion_bias',
        'direction': 'short',
    },
    {
        'id': 'novel_228',
        'name': 'lion_lu_days',
        'label': '距最近涨停天数',
        'category': 'event_decay',
        'hypothesis': '距最近一次涨停越久(沉寂越深)的股票未来收益越高 —— 涨停后的注意力溢价随时间衰减',
        'logic': '涨停=注意力事件, 其后接力资金与情绪溢价随时间衰减; 距涨停远=情绪已出清未被炒作。'
                 '实测 RankIC +0.0439 / 日ICIR +0.410, 五分位单调性 +0.900 (Q1 -0.168% → Q5 +0.227%), '
                 '小/中/大市值三池 t = +5.65 / +8.35 / +2.64 全部显著。'
                 '与本库 rev20/rev60 相关仅 +0.11/+0.21, 与 size -0.04 → 非反转/市值代理; '
                 '但与同源 lion_lu_cnt 相关 -0.655',
        'formula': 'lion_lu_days',
        'direction': 'long',
    },
    {
        'id': 'novel_229',
        'name': 'lion_lu_cnt',
        'label': '760日涨停次数(涨停基因)',
        'category': 'event_decay',
        'hypothesis': '过去 760 交易日涨停次数越多的股票未来收益越低 —— 与社区「涨停基因」选股方向相反',
        'logic': '涨停次数=历史被炒作程度; 炒作过的票套牢盘重、筹码分散。'
                 '聚宽社区策略取涨停次数前 10% 当优势股池, 实测方向相反 (RankIC -0.0195, 五分位单调性 -1.000)。'
                 '与 lion_lu_days 相关 -0.655 (同源两测度), 二者择一即可',
        'formula': 'lion_lu_cnt',
        'direction': 'short',
    },
]

src = open(PATH, encoding='utf-8').read()
i = src.index('NOVEL_FACTOR_LIBRARY = [')
j = src.index('\n]', i)
block = src[i:j]
print(f'列表区间: [{i}, {j})  现有条目 ~{block.count(chr(10) + "    {")} 个')

added, skipped = [], []
for e in ENTRIES:
    if f"'{e['name']}'" in block:
        skipped.append(e['name'])
        continue
    lines = ['    {']
    for k, v in e.items():
        lines.append(f'        {k!r}: {v!r},')
    lines.append('    },')
    added.append('\n'.join(lines))

print(f'新增 {len(added)}  跳过(已存在) {len(skipped)} {skipped}')
if not added:
    print('无需写入'); sys.exit(0)

out = src[:j] + '\n' + '\n'.join(added) + src[j:]
try:
    compile(out, PATH, 'exec')
except SyntaxError as ex:
    print(f'❌ compile 自检失败, 未落盘: {ex}')
    sys.exit(1)
open(PATH, 'w', encoding='utf-8').write(out)
print(f'✅ 已写入, 文件 {len(src)} → {len(out)} 字节, compile 自检通过')
