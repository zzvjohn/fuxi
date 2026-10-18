# -*- coding: utf-8 -*-
"""一次性脚本：向 data/stage1_optimization_proposals.json 追加 P-20260924-014。

纪律（与 _p1_/_p2_/_p3_append_proposal.py 同构）：
  ① 先查占用，占用即拒绝；
  ② 先 encode('utf-8') 拿字节，再 open('wb') 落盘；
  ③ 落盘前备份；
  ④ 写后回读字节级断言（相等 + 无 BOM + CRLF=0 + 条数 + 对方条目仍在）。
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'data' / 'stage1_optimization_proposals.json'
NEW_ID = 'P-20260924-014'

ENTRY = {
    'proposal_id': NEW_ID,
    'date': '2026-09-25',
    'created_at': '2026-09-25T00:52:00+08:00',
    'title': (
        'P4 结算两项工程欠账：FUXI_COLLECT_CAL_END 默认值 20260430→20261231（与随包日历末端对齐 + 落后即拒跑 exit 2 '
        '+ 默认值与日历同进守卫 exit 3），manifest 加锁（_DATA_SENTINELS 5→9 键、extra_code/extra_data 声明通道、'
        'posix 键、声明即须存在）；结算中暴露并修掉三个此前不可见的静默缺陷（日期写法混用 352991 行 / 读取器在混合类型列上崩 / 抬高默认值会把「尚未发生」当成「缺失」）'
    ),
    'category': '因子研究/数据基座',
    'domains': ['T2数据口径', 'T3模块/通道', 'T1闸门/裁决'],
    'risk_level': 'low',
    'status': 'landed',
    'proposal_change_kind': 'engineering',
    'generator_hint': 'manual',
    'trigger': (
        '两项欠账首次记录于 reports/fuxi_holdings_and_drought_review_20260923.md 与 '
        'reports/fuxi_delist_channel_20260924.md：'
        '① collect_daily_full.py 把 FUXI_COLLECT_CAL_END 硬编码为 20260430，把每一次补采静默截断在该日，'
        'suspend_d.csv 就停在那里，直接造成账本 90 个持有日不可达；'
        '② research/factor_alchemy/run_contract.py 的 _DATA_SENTINELS 只有 5 键，'
        '而判据输入 suspend_d.csv / delist_dates.csv / stock_list.csv / trade_calendar.csv 一个都不在，'
        'code_fingerprints 又是 9 个固定模块 ⇒ 批次自己的脚本结构上无法登记。'
    ),
    'current_state': (
        'P4 之前：'
        '① 采集脚本第 [2] 步「检查已有数据」用 pd.read_csv(usecols=[trade_date]) 读四个大文件；'
        'stk_limit.csv 的 trade_date 列混着紧凑写法（8433700 行）与短横线写法（352991 行，首行第 8433702 行），'
        '该调用在 pandas 3.0.3（默认 env / 评估用）上直接抛 IndexError，在生产解释器 C:/Python314（pandas 2.x）上'
        '只发一条 DtypeWarning —— 也就是同一件事在一个解释器上崩、在另一个上不吭声；'
        '② 旧 get_existing_dates 不指定 dtype，把短横线日期原样存进集合，于是「判成缺失」在两种解释器上都成立；'
        '③ 采集末端落后于本地数据末端时脚本照样跑完并报「所有数据已补全」，没有任何读数能看出它什么都没采；'
        '④ manifest 的 data 标准集只有 5 键，声称覆盖「数据」却对「整日停牌」「退市日期」两条判据的输入一无所知；'
        '⑤ 键用平台分隔符 str(relative_to(...))，在 Windows 产出反斜杠，跨平台无法逐字比较，'
        '且对项目外路径直接抛 ValueError。'
    ),
    'proposed_change': (
        '① strategies/final/collect_daily_full.py（改）：'
        'a) tushare 客户端改惰性初始化 get_pro()（原 ts.set_token()/pro_api() 在 import 时执行，使模块无法被任何离线测试导入，'
        '而「日历末端是否落后」这类纯逻辑恰恰必须能离线单测）；'
        'b) 新增 normalize_day() 与 date_format_audit()；'
        'c) get_existing_dates() 改 dtype=str, low_memory=False + 逐值规范化；'
        'd) append_df() 在写入口对 trade_date/cal_date 逐值规范化，不可解析即 raise ValueError（不填默认值、不静默跳过）；'
        'e) 新增 PACKAGED_CALENDAR / packaged_calendar_end() / stale_horizons()：把「日历末端落后」变成可判定对象，'
        '任一参考物晚于 cal_end 即 sys.exit(2)；'
        'f) 默认值由字面量 20260430 抬到 CAL_END_DEFAULT=20261231（= 随包日历末端），'
        '并新增 default_is_in_sync_with_the_packaged_calendar()：日历延伸而默认值没跟上即 sys.exit(3)；'
        'g) 🔴 新增 not_yet_available_sessions(all_days, today)：把交易日历拆成「已发生(应采)」与「尚未发生(不采、不计缺失)」，'
        '两个数都进读数；today 显式传入（可用 FUXI_COLLECT_TODAY 覆盖）以便离线单测。'
        '② research/factor_alchemy/run_contract.py（改）：'
        '_DATA_SENTINELS 5→9 键（新增 suspend_d.csv / delist_dates.csv / stock_list.csv / trade_calendar.csv）；'
        '新增 _manifest_key()（项目内 posix 相对、项目外 posix 绝对）；'
        '新增 _resolve_declared()（声明的路径不存在即 raise ValueError，fail-closed）；'
        'RunContract.__init__ 新增 extra_code=()/extra_data=()；'
        '新增 fingerprint_coverage 块，把声明分三类：standard / extra_*_newly_added / extra_*_already_standard。'
        '③ research/factor_alchemy/tests/test_run_contract.py（改）：4 条 → 9 条。'
        '④ scripts/fuxi_p4_debts_20260924.py（新建）：对照 + 三缺陷复现 + 四关验收 + 9 条单测。'
    ),
    'implementation_scope': (
        'strategies/final/collect_daily_full.py（CRLF，7381 -> 约 11700 bytes，7 处修改）; '
        'research/factor_alchemy/run_contract.py（LF，_DATA_SENTINELS / _manifest_key / _resolve_declared / '
        'extra_code,extra_data / fingerprint_coverage）; '
        'research/factor_alchemy/tests/test_run_contract.py（LF，4 -> 9 条）; '
        'scripts/fuxi_p4_debts_20260924.py（新建，批次脚本 + 9 条单测）; '
        'reports/p4_debts_20260924/readings.json（11199 bytes）; '
        'reports/p4_debts_20260924/date_format_mixed.csv; '
        'reports/p4_debts_20260924/manifest_keys.csv; '
        'reports/fuxi_p4_engineering_debts_20260925.md（报告）'
    ),
    'expected_impact': (
        '欠账一读数（对照）：旧默认 20260430 落后于 5 个参考物（packaged_trade_calendar / prices / basic / suspend / limit）'
        '⇒ 判 refused；新默认 20261231 落后 0 个 ⇒ runs。'
        '缺陷 A：stk_limit.csv 紧凑 8433700 行 + 短横线 352991 行（首行第 8433702 行）；'
        '不规范化会重复追加 46 天 / 352991 行。🔴 此前完全不可见 —— 旧默认把这些日期挡在采集窗口外，'
        '「判成缺失」没有任何后果；抬到 20261231 后它们落进窗口，日历里是 20260528 而集合里是 2026-05-28，'
        '两边永远对不上 ⇒ 逐日重复追加。这是「修了一个 bug 才会显形的 bug」。'
        '缺陷 B：同一句 pd.read_csv 在 pandas 3.0.3 抛 IndexError、在 pandas 2.x 只发 DtypeWarning（静默误判）。'
        '缺陷 C（本批新发现，原缺陷的镜像）：随包日历 1454 个开市日 = 已发生 1390 + 尚未发生 64（20260928..20261231）；'
        '若不区分，每轮空转 256 次 API、永久误报缺 64 天、last_update.txt 的 last_trade_date 被写成 20261231（未来日期）。'
        '老 bug 是「把过去静默截断」，这一条是「把未来当成缺失」，同一个失败模式方向相反、同样静默。'
        '欠账二读数：standard_data_set_size 9（新增 4）；manifest data 键 10（9 标准 + 1 新声明）；keys_added_count 4；'
        'code 条目带 sha256=True、data 条目不带 sha256=True；declared_but_missing_is_refused=True；'
        'keys_use_posix_separators=True；项目外路径保留 posix 绝对键；R0 复用守卫触发 RunReuseError（run_id 不复用）。'
        '新增键：data/raw/delist_dates.csv, data/raw/stock_list.csv, data/raw/suspend_d.csv, data/raw/trade_calendar.csv。'
        '对干旱的实质帮助（诚实评估）：对 IC 缺口本身没有直接帮助，本批不产生任何因子、不改任何判定基线。'
        '它属于「解旱的先决条件」：盈亏平衡线 -0.00843 + 0.0915×IC 的输入就是这四个采集文件，'
        '采集被静默截断 ⇒ 所有「某因子无效」的结论都可能只是数据没到；'
        'P4 交付的「参考物是否落后」判据 P5 立刻就要用（延伸日历必须同步抬默认值），'
        '否则一次正确的日历延伸会立刻把补采打回静默截断。'
    ),
    'landing_note': (
        '未调用任何厂商 API、未改写任何数据文件、未翻任何开关、未改账本/S5/JQ 准入或任何封存产物。'
        'stk_limit.csv 的 352991 行短横线日期今天仍然在盘上（格式混用被报告，没有被就地修复）。'
        '本批暴露并留证多轮形状缺陷（均未用成功轮覆盖）：'
        '① keys_added_count 断言先失败在 5（把 manifest 全部 data 键都算成标准集新增，把本批声明的 2 条 extra 也算了进去）；'
        '修正后失败在 3（把「重复声明」从标准集里错误剔除 —— 以 delist_dates.csv 为证：它既是新纳入标准集的 4 键之一，又被本批显式声明）'
        '⇒ 最终改为三分类（standard / newly_added / already_standard）才自洽，两个方向都错；'
        '② _manifest_key 用 str(relative_to(...)) 在 Windows 产出反斜杠键，测试期望的是 posix 键；'
        '③ 项目外路径的测试写成 str(blob) 而实际键是 as_posix()（测试形状 != 生产形状，同型第 6 次）；'
        '④ 缺陷 C 是本批在写「只增不改」证据时才发现的 —— 原以为只需抬高默认值，'
        '抬高后才发现交易日历本身延伸进未来；若只按原计划交付，补采会「成功地把 64 个未来日各调 5 次 API 并记成成功」。'
        '门本体与批次脚本测试：run_contract 9/9 全绿；fuxi_p4_debts 9/9 全绿；alchemy 全量 184/184 全绿。'
    ),
    'verification': (
        'python scripts/fuxi_p4_debts_20260924.py --tests => Ran 9 tests, OK；'
        'cd research/factor_alchemy && PYTHONPATH=$PWD python tests/test_run_contract.py => Ran 9 tests, OK；'
        'cd research/factor_alchemy && PYTHONPATH=$PWD python -m unittest discover -s tests -p "test_*.py" => Ran 184 tests, OK；'
        'python scripts/fuxi_p4_debts_20260924.py --run => exit 0，'
        '全部 fail-closed 断言通过（旧默认必须判 refused、新默认必须判 runs、keys_added_count==4、声明缺失必须被拒、'
        '复用 run_id 必须被拒、键必须 posix、not_yet_available>0、空转 API 次数>0、默认值与随包日历一致）；'
        '产出 readings.json（11199 bytes）、date_format_mixed.csv、manifest_keys.csv。'
    ),
    'artifacts': [
        'strategies/final/collect_daily_full.py',
        'research/factor_alchemy/run_contract.py',
        'research/factor_alchemy/tests/test_run_contract.py',
        'scripts/fuxi_p4_debts_20260924.py',
        'reports/p4_debts_20260924/readings.json',
        'reports/p4_debts_20260924/date_format_mixed.csv',
        'reports/p4_debts_20260924/manifest_keys.csv',
        'reports/fuxi_p4_engineering_debts_20260925.md',
    ],
    'zero_adjudication': True,
    'default_on': False,
    'ab_evidence': (
        'A/B 反事实（欠账一）：同一套 stale_horizons 判据下，旧默认 20260430 命中 5 个参考物 ⇒ refused、'
        '新默认 20261231 命中 0 个 ⇒ runs。这是「抬高默认值」这一采集范围变更的对照。'
        '默认不变量：CAL_END_DEFAULT 必须等于 packaged_calendar_end()（单测 + 运行时 exit 3 守卫双向钉死）。'
        'R0 契约：复用 run_id 被拒（RunReuseError），确定性复现（就地造探针 run，不用上次残留）。'
        '旁路诊断与运行读数分开报：所有涉及 all_days 的数都标注为「以随包日历为代理」的旁路诊断。'
    ),
    'known_limit': (
        '① 本批不调用任何厂商 API —— 「抬高默认值后能补到多少天」不是本批的读数，本批不声称补采已经跑通；'
        '② stk_limit.csv 的 352991 行短横线日期没有就地修复，只是被报告；下一次真正的补采才会把它们规范化（写入口）；'
        '③ 缺陷 C 的 64 天是按墙钟（today=20260925）算的，不是「厂商在那 64 天没有数据」的证据；'
        '④ manifest 的 extra_* 是**声明**而非自动发现，未声明的输入依旧不在集内，不得读成「完整覆盖」；'
        '⑤ 本批只是让「登记本批脚本/数据」在结构上成为可能，没有翻开 R0 锁，也没有给任何批次强制声明；'
        '⑥ 抬高默认值本身是采集范围变更，其经济后果（多补的 46 天是否改变任何读数）未测；'
        '⑦ 本批对 IC 缺口没有直接贡献，不得当成解旱成果。'
    ),
    'next_step': (
        '转入 P5：把 R5 等交易日历延伸到第 26 个周频截面所需的视野（约 2027-04）。'
        '🔴 关键耦合：日历一延伸，CAL_END_DEFAULT 必须原子同进，否则 P4 刚修掉的静默截断立刻复发 —— '
        'P5 的验收条件就是 default_is_in_sync_with_the_packaged_calendar() 为真。'
        'P5 只延伸覆盖、不改统计要求（26 段），decisive_endpoint 的段数不变；'
        '随后 P6 固化增量对照的冻结设计（不同时改多个维度）。'
        '同时保留最高优先级：组合层水平项本身的测量（IC=0 时 Top80 仍少赚 72~84bps/5 日）。'
    ),
    'rollback': (
        'collect_daily_full.py：把 CAL_END_DEFAULT 改回 20260430，或删除新增的 packaged_calendar_end / '
        'stale_horizons / not_yet_available_sessions / default_is_in_sync_with_the_packaged_calendar 与调用点；'
        'run_contract.py：_DATA_SENTINELS 回到 5 键、移除 extra_code/extra_data/fingerprint_coverage；'
        '本批未改任何数据文件，无需回退数据。'
    ),
}


def main() -> int:
    raw = TARGET.read_bytes()
    assert raw[:3] != b'\xef\xbb\xbf', 'target unexpectedly carries a BOM'
    assert raw.count(b'\r\n') == 0, 'target unexpectedly carries CRLF'
    data = json.loads(raw.decode('utf-8'))
    assert isinstance(data, list)
    ids = [x.get('proposal_id') for x in data]
    if NEW_ID in ids:
        raise SystemExit('refusing to append: %s already present' % NEW_ID)
    before = len(data)
    last_before = ids[-1]

    data.append(ENTRY)
    text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    payload = text.encode('utf-8')          # 先编码，再落盘
    assert payload[:3] != b'\xef\xbb\xbf'
    assert payload.count(b'\r\n') == 0

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = TARGET.with_name(TARGET.name + '.bak_%s_p4' % stamp)
    shutil.copy2(TARGET, backup)

    with open(TARGET, 'wb') as fh:
        fh.write(payload)

    got = TARGET.read_bytes()
    assert got == payload, 'byte-level round-trip mismatch'
    assert got[:3] != b'\xef\xbb\xbf'
    assert got.count(b'\r\n') == 0
    again = json.loads(got.decode('utf-8'))
    assert isinstance(again, list) and len(again) == before + 1, 'entry count wrong'
    assert again[-1]['proposal_id'] == NEW_ID
    got_ids = [x.get('proposal_id') for x in again]
    assert got_ids[:-1] == ids, 'prior entries mutated'
    assert last_before in got_ids, 'previous last entry lost'
    assert again[-1]['status'] == 'landed'
    assert again[-1]['proposal_change_kind'] == 'engineering'
    assert again[-1]['zero_adjudication'] is True and again[-1]['default_on'] is False

    print('ok: %s -> %s entries (was %s, new last = %s)'
          % (before, len(again), last_before, NEW_ID))
    print('bytes %d (was %d)  backup %s' % (len(got), len(raw), backup.name))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
