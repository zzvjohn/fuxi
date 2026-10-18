# -*- coding: utf-8 -*-
"""上游入口可交易性门 (P3, P-20260924-013)。

缺陷本体
========
冻结目标在**不可交易日建仓**: 目标按信号日排名选出 top-N, 而目标从**下一个交易日**
起生效; 选股时只校验了信号日的 `close > 0`, **没有校验该证券在生效日能否成交**。
实测 13 / 13701 个入场事件落在无行情行日 (llm1 11 + mem 2), 其中 13/13 在本地停牌表
里是整日停牌 (`suspend_type='S'`, `suspend_timing` 为空)。

后果不是"账本填错了价", 而是三件更隐蔽的事:
  1. 该权重**从未被部署**, 变成静默现金拖累, 无任何决策可归因;
  2. 下游账本**吸收了**它 —— 只留一条 `filled=0, reason='halted'` 的订单, 而且这条
     订单既不影响状态机也不回灌目标 ⇒ 与"目标本来就没买"不可区分;
  3. 全窗证据库存 (170 例无行情行) 把"目标想持有"与"市场拒绝成交"混在一张表里。

所以修法只能是**在上游入口显式判定并留证**, 不在下游兜。

模块边界 (修改本文件前必读)
==========================
本模块**只做一件事**: 给定 (证券, 生效日), 用**本地封存数据**回答"能不能成交",
并把不能成交的**逐条**列出来。它
    × 不推断成交价、不填价、不插值、不做零收益假设;
    × 不改变权重口径 (默认 `enforce` 也不重归一化 —— 见下);
    × 不下任何经济裁决, 不产生 PASS, 不触碰 S5 / JQ 准入;
    × 不把"缺记录"说成"停牌"。**缺证据与已证伪必须分开计数**(见 VERDICTS)。

判定表 (穷尽且互斥, 按此顺序求值)
==================================
| 判定 | 含义 | `enforce` 下动作 |
|---|---|---|
| `tradable` | 生效日有行情行 | 准入 |
| `conflicting_price_and_halt` | 生效日既有行情行又被标整日停牌 | 拒(并要求裁决) |
| `refused_halted_full_day` | 本地停牌表 `S` 且 `suspend_timing` 为空 | 拒 |
| `refused_halted_partial_day` | 本地停牌表 `S` 但 `suspend_timing` 非空 | 拒 |
| `refused_delisted` | 厂商 `list_status='D'` 且 `delist_date <= 生效日` | 拒 |
| `refused_not_a_trading_session` | 生效日不是 SSE 开市日 | 拒 |
| `uncertifiable_status_table_absent` | 无行情行, 且生效日**超出**本地停牌表末端 | 拒(计入"缺证据") |
| `uncertifiable_no_price_row` | 无行情行, 生效日在表内却无停牌行、无退市记录 | 拒(计入"缺证据") |

**准入集合 = {tradable}**, 其余一律不准入 (fail-closed)。后两类**不得**被写成"停牌",
它们只能进"缺证据"计数 —— 这是本模块最重要的区分。

开关
====
env `FUXI_ENTRY_TRADABILITY`:
    `annotate` | `shadow` | `on` | `1`   -> **默认**: 逐事件判定 + 落拒绝台账,
                                            **目标与权重一字不改** (零裁决类)。
    `enforce`                            -> 拒绝项从目标中剔除; 权重仍按 1/top_n,
                                            未部署部分留现金并逐条声明 (不重归一化)。
    `off` | `legacy` | `0`               -> 完全禁用 (返回入参, 台账为空)。

**默认取 `annotate`**: 它是零裁决观察者 —— 只报告、不改任何产物, 因此不存在
"凭空造出 PASS" 的风险 (与 `trade_gate_shadow` 同一条理由, 见该模块 §开关)。
`enforce` **必须显式开启**: 它会改变目标集合与权重, 属经济口径变更, 需要
「翻默认四关」+ 独立协议, 不在本批自行翻开。未知取值**直接抛错**, 不做静默降级。

不变量 (全部有断言)
====================
1. `checked == admitted + refused_refuted + refused_uncertifiable + conflicting`
   —— 每一个入场事件都必须落在某一类里, **既不许静默跳过, 也不许重复计数**。
2. `annotate` 与 `off` 返回的 `frames` **逐键逐一相等** (观察者不动目标)。
3. `enforce` 只做减法: `enforce` 的每个目标键的证券集合 `⊆` 入参的证券集合,
   且 `enforce ⊆ legacy` 在全窗统计上体现为 removed>0, added==0。
4. 权重口径不变: `enforce` 下准入项的权重与入参**逐位相同**。
"""

import os

MODES = ('off', 'annotate', 'enforce')
DEFAULT_MODE = 'annotate'

ENV_NAME = 'FUXI_ENTRY_TRADABILITY'

_ALIASES = {
    'annotate': 'annotate', 'shadow': 'annotate', 'on': 'annotate', '1': 'annotate',
    'true': 'annotate', 'yes': 'annotate',
    'enforce': 'enforce', 'block': 'enforce',
    'off': 'off', 'legacy': 'off', '0': 'off', 'false': 'off', 'no': 'off',
}

# verdict -> (是否准入, 类别, 含义)
VERDICTS = {
    'tradable': (
        True, 'admitted',
        '生效日本地有行情行, 该证券当日有可成交记录'),
    'conflicting_price_and_halt': (
        False, 'conflicting',
        '生效日既有行情行又被本地停牌表标为整日停牌: 两条本地证据互相矛盾, '
        '不替它选边, 按不准入处理并要求独立裁决'),
    'refused_halted_full_day': (
        False, 'refuted',
        '本地停牌表命中 (证券, 生效日) 且 suspend_type=S、suspend_timing 为空 '
        '⇒ 当日整日停牌, 不存在可成交价'),
    'refused_halted_partial_day': (
        False, 'refuted',
        '本地停牌表命中但 suspend_timing 非空 ⇒ 当日只部分时段停牌, '
        '不得假设有全日流动性, 不按"可成交"处理'),
    'refused_delisted': (
        False, 'refuted',
        '厂商证券主表 list_status=D 且 delist_date 不晚于生效日 ⇒ 当日已不属于'
        '可交易证券'),
    'refused_not_a_trading_session': (
        False, 'refuted',
        '生效日不在 SSE 开市日集合内 ⇒ 结构上不可能是建仓日'),
    'refused_target_never_takes_effect': (
        False, 'refuted',
        '该 stamp 之后窗口内已无开市日 ⇒ 这个目标从未生效, 谈不上可交易性; '
        '它与"停牌"是两件事, 单独一类, 由调用方按 stamp 逐条公开'),
    'uncertifiable_status_table_absent': (
        False, 'uncertifiable',
        '无行情行, 且生效日超出本地停牌表末端 ⇒ 本地既不能证明可成交、也不能证明'
        '停牌; 记"缺证据", **不得**写成停牌'),
    'uncertifiable_no_price_row': (
        False, 'uncertifiable',
        '无行情行, 生效日落在本地停牌表区间内, 却没有停牌行、也没有退市记录 ⇒ '
        '本地证据无法定性; 记"缺证据", **不得**写成停牌'),
}

ADMITTED_VERDICTS = tuple(v for v, spec in VERDICTS.items() if spec[0])
REFUSED_VERDICTS = tuple(v for v in VERDICTS if v not in ADMITTED_VERDICTS)


def get_mode(environ=None):
    """解析 env 开关。未知取值**抛错**, 绝不静默回退到某一档。"""
    environ = os.environ if environ is None else environ
    raw = environ.get(ENV_NAME)
    if raw is None or str(raw).strip() == '':
        return DEFAULT_MODE
    key = str(raw).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            '%s=%r is not a recognised mode; expected one of %s (aliases: %s). '
            'Refusing to guess: a silent fallback would change which sessions are '
            'treated as executable without anyone noticing.'
            % (ENV_NAME, raw, '|'.join(MODES), '|'.join(sorted(_ALIASES))))
    return _ALIASES[key]


def effective_session(stamp, days):
    """目标在 stamp 日生成, 自**其后第一个开市日**起生效 —— 那才是建仓日。"""
    index = _session_index(days)
    if stamp not in index:
        raise KeyError('stamp %r is not in the supplied session calendar' % (stamp,))
    position = index[stamp] + 1
    return days[position] if position < len(days) else None


def _session_index(days):
    return {d: i for i, d in enumerate(days)}


def certify(panel, code, session):
    """对单个 (证券, 生效日) 出判定。纯函数, 只读 `panel`, 无 I/O。

    `panel` 需要四个集合与两个映射 (全部由调用方从封存文件装配):
        priced        {(code, session)} 有行情行
        halt_full     {(code, session)} suspend_type=S 且 suspend_timing 为空
        halt_partial  {(code, session)} suspend_type=S 且 suspend_timing 非空
        status_days   {session} 本地停牌表覆盖到的日期
        status_terminal / sessions / delist_dates
    """
    sessions = panel.get('sessions')
    if session is None:
        return _verdict(code, session, 'refused_target_never_takes_effect', panel)
    if sessions is not None and session not in sessions:
        return _verdict(code, session, 'refused_not_a_trading_session', panel)
    key = (code, session)
    priced = key in panel['priced']
    if priced:
        if key in panel['halt_full']:
            return _verdict(code, session, 'conflicting_price_and_halt', panel,
                            extra={'has_price_row': True, 'status_says': 'full_day_halt'})
        return _verdict(code, session, 'tradable', panel, extra={'has_price_row': True})
    if key in panel['halt_full']:
        return _verdict(code, session, 'refused_halted_full_day', panel)
    if key in panel['halt_partial']:
        return _verdict(code, session, 'refused_halted_partial_day', panel)
    delist = panel.get('delist_dates', {}).get(code)
    if delist and delist <= session:
        return _verdict(code, session, 'refused_delisted', panel,
                        extra={'delist_date': delist})
    if session not in panel.get('status_days', set()):
        return _verdict(code, session, 'uncertifiable_status_table_absent', panel,
                        extra={'status_table_terminal': panel.get('status_terminal')})
    return _verdict(code, session, 'uncertifiable_no_price_row', panel)


def _verdict(code, session, verdict, panel, extra=None):
    admit, category, meaning = VERDICTS[verdict]
    row = {'code': code, 'session': session, 'verdict': verdict, 'category': category,
           'admitted': admit, 'meaning': meaning,
           'has_price_row': False, 'session_in_status_table':
               session in panel.get('status_days', set())}
    if extra:
        row.update(extra)
    return row


def gate(frames, days, panel, mode=None, outcome_label=None):
    """把门套在目标帧上。

    `frames`: {stamp: {code: weight}} —— 与冻结目标文件同构。
    返回 dict:
        mode, checked, admitted, refused, counts, by_verdict, by_category,
        undeployed_weight, emptied_targets, rejects, frames

    **返回值里 `frames` 才是目标**, `rejects` 是台账; 两者不得混排成一张读数
    (旧稿把"若执行会怎样"与"实际产出"混排会被读成事实)。
    """
    mode = get_mode() if mode is None else mode
    if mode not in MODES:
        raise ValueError('unknown mode %r' % (mode,))
    index = _session_index(days)
    out = {}
    rejects = []
    counts = {'checked': 0}
    by_verdict = {v: 0 for v in VERDICTS}
    by_category = {c: 0 for c in ('admitted', 'refuted', 'uncertifiable', 'conflicting')}
    emptied = []
    undeployed = 0.0
    not_applicable = []
    if mode == 'off':
        for stamp in sorted(frames):
            out[stamp] = dict(frames[stamp] or {})
        # 🔴 各档必须返回**同一套键** (全零也要在)。少键会让"没检查"被读成"检查了没发现"。
        return {'mode': mode, 'checked': 0, 'admitted': 0, 'output_names':
                sum(len(f or {}) for f in out.values()),
                'refused': 0, 'refused_refuted': 0, 'refused_uncertifiable': 0,
                'conflicting': 0, 'counts': counts, 'by_verdict': by_verdict,
                'by_category': by_category, 'undeployed_weight': 0.0,
                'emptied_targets': [], 'rejects': [], 'frames': out,
                'stamps_without_an_effective_session': [],
                'names_not_applicable': 0,
                'outcome_label':
                    outcome_label or 'gate disabled: targets returned unchanged, nothing '
                                     'was checked, and nothing is claimed about '
                                     'tradability'}

    for stamp in sorted(frames):
        held = frames[stamp] or {}
        session = effective_session(stamp, days) if stamp in index else None
        if session is None:
            # 🔴 窗口在这条目标生效前就结束了 ⇒ **没有可判的那一天**, 不是"拒"。
            #    必须原样返回: 若在这里剔除, enforce 会把最后一个 stamp 清成空目标,
            #    账本读作"清仓", 一次窗口边界的结构事实就变成了一笔真实卖出。
            out[stamp] = dict(held)
            not_applicable.append({'stamp': stamp, 'names': len(held),
                                   'why': 'no_session_inside_the_calendar_after_this_stamp'})
            continue
        kept = {}
        for code in sorted(held):
            counts['checked'] += 1
            row = certify(panel, code, session)
            row['stamp'] = stamp
            row['weight'] = held[code]
            by_verdict[row['verdict']] += 1
            by_category[row['category']] += 1
            if row['admitted']:
                kept[code] = held[code]
            else:
                row['action'] = 'refused' if mode == 'enforce' else 'reported_only'
                undeployed += held[code]
                rejects.append(row)
        if mode == 'enforce':
            out[stamp] = kept
            if held and not kept:
                emptied.append({'stamp': stamp, 'session': session,
                                'refused_names': sorted(held)})
        else:
            out[stamp] = dict(held)

    assert counts['checked'] == (by_category['admitted'] + by_category['refuted']
                                 + by_category['uncertifiable']
                                 + by_category['conflicting']), by_category
    assert counts['checked'] == sum(by_verdict.values())
    assert counts['checked'] == len(rejects) + by_category['admitted']
    # 🔴 `admitted` 是**判定**计数, 与"输出目标里实际有多少个名字"不是同一件事:
    #    annotate 一个名字都不删 ⇒ 输出里仍是 checked 个, 而 admitted 只数得通的那个;
    #    另外每条"窗口内从未生效"的目标原样保留, 其名字记 not_applicable。
    #    三者混用会把观察者读成闸门、把窗口边界读成拒单。
    output_names = sum(len(f or {}) for f in out.values())
    names_not_applicable = sum(row['names'] for row in not_applicable)
    if mode == 'enforce':
        assert by_category['admitted'] + names_not_applicable == output_names, (
            by_category['admitted'], names_not_applicable, output_names)
    else:
        assert counts['checked'] + names_not_applicable == output_names, (
            counts['checked'], names_not_applicable, output_names)
        for stamp in frames:
            assert out[stamp] == dict(frames[stamp] or {}), stamp
    return {
        'mode': mode,
        'checked': counts['checked'],
        'admitted': by_category['admitted'],
        'output_names': output_names,
        'refused': len(rejects),
        'refused_refuted': by_category['refuted'],
        'refused_uncertifiable': by_category['uncertifiable'],
        'conflicting': by_category['conflicting'],
        'counts': counts,
        'by_verdict': by_verdict,
        'by_category': by_category,
        'undeployed_weight': undeployed,
        'emptied_targets': emptied,
        'rejects': rejects,
        'frames': out,
        'stamps_without_an_effective_session': not_applicable,
        'names_not_applicable': sum(row['names'] for row in not_applicable),
        'outcome_label':
            outcome_label or ('enforce: refused names removed, weights untouched, the '
                              'undeployed slice stays in cash and is listed per name'
                              if mode == 'enforce' else
                              'annotate: targets and weights are byte-identical to the '
                              'input; the reject ledger is a report, not a decision'),
    }


def entry_events(frames, days):
    """入场事件 = 每个 (证券) 连续持有段的首个交易日。

    这是复算 `13 / 13701` 的口径, 与 `_probe_entry_during_halt_20260924.py` 逐位一致。
    一个证券在同一臂里可以入场多次 (退出后再进) ⇒ 事件数 > 证券数。

    持有段的端点必须与探针逐位相同: 一个 stamp 的目标自**其后第一个开市日**起生效,
    持续到**下一个 stamp 生效日**为止 (含该日)。把端点写成"下一个 stamp 前一天"会
    在每个 stamp 边界多切出一条一秒长的缝, 使每个 (证券, span) 都变成一次入场 ——
    实测会把入场事件从 13701 放大到 27840。这类缝不会报错, 只会把分母换掉。
    """
    index = _session_index(days)
    held_by_code = {}
    for stamp in sorted(frames):
        if stamp not in index:
            raise KeyError('stamp %r is not in the supplied session calendar' % (stamp,))
        held = frames[stamp] or {}
        start = index[stamp] + 1
        if start >= len(days):
            continue
        nxt = _next_stamp(stamp, frames)
        if nxt is None:
            stop = len(days) - 1
        else:
            nxt_effective = effective_session(nxt, days)
            stop = (index[nxt_effective] - 1) if nxt_effective is not None \
                else len(days) - 1
        if start > stop:
            continue
        for code in held:
            held_by_code.setdefault(code, set()).update(days[start:stop + 1])
    events = []
    for code, dayset in held_by_code.items():
        ordered = sorted(dayset)
        for day in ordered:
            i = index[day]
            previous = days[i - 1] if i > 0 else None
            if previous is None or previous not in dayset:
                events.append({'code': code, 'session': day})
    return sorted(events, key=lambda r: (r['code'], r['session']))


def skipped_stamps(frames, days):
    """Would-be spans that never take effect inside the calendar.

    🔴 探针是**静默跳过**这些 stamp 的 (所以 13701 里不含它们的持有日)。静默跳过不得
    留存为不可见: 这里把它们逐条列出来, 调用方必须落盘计数。
    """
    index = _session_index(days)
    out = []
    for stamp in sorted(frames):
        if stamp not in index:
            out.append({'stamp': stamp, 'why': 'not_a_session_in_the_calendar'})
            continue
        start = index[stamp] + 1
        if start >= len(days):
            out.append({'stamp': stamp, 'why': 'the_calendar_ends_before_the_target_'
                                               'takes_effect'})
            continue
        nxt = _next_stamp(stamp, frames)
        if nxt is None:
            continue
        nxt_effective = effective_session(nxt, days)
        stop = (index[nxt_effective] - 1) if nxt_effective is not None else len(days) - 1
        if start > stop:
            out.append({'stamp': stamp, 'why': 'the_next_target_takes_effect_before_'
                                               'this_one_does'})
    return out


def _next_stamp(stamp, frames):
    stamps = sorted(frames)
    position = stamps.index(stamp)
    return stamps[position + 1] if position + 1 < len(stamps) else None


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def _panel(priced=(), halt_full=(), halt_partial=(), status_days=(),
           terminal=None, sessions=(), delist=None):
    return {'priced': set(priced), 'halt_full': set(halt_full),
            'halt_partial': set(halt_partial), 'status_days': set(status_days),
            'status_terminal': terminal, 'sessions': set(sessions),
            'delist_dates': dict(delist or {})}


def tests():
    import unittest

    DAYS = ['2025-01-02', '2025-01-03', '2025-01-06', '2025-01-07', '2025-01-08']

    class GateTests(unittest.TestCase):
        def test_mode_aliases_resolve_to_the_three_documented_modes(self):
            for alias, expected in (('shadow', 'annotate'), ('on', 'annotate'),
                                    ('1', 'annotate'), ('off', 'off'),
                                    ('legacy', 'off'), ('enforce', 'enforce')):
                self.assertEqual(get_mode({ENV_NAME: alias}), expected)

        def test_default_mode_is_the_zero_adjudication_observer(self):
            self.assertEqual(DEFAULT_MODE, 'annotate')
            self.assertEqual(get_mode({}), 'annotate')
            self.assertEqual(get_mode({ENV_NAME: '   '}), 'annotate')

        def test_unknown_mode_raises_instead_of_falling_back(self):
            with self.assertRaises(ValueError):
                get_mode({ENV_NAME: 'maybe'})

        def test_verdict_partition_is_exhaustive_and_mutually_exclusive(self):
            categories = [spec[1] for spec in VERDICTS.values()]
            self.assertEqual(sorted(set(categories)),
                             ['admitted', 'conflicting', 'refuted', 'uncertifiable'])
            self.assertEqual(sorted(ADMITTED_VERDICTS), ['tradable'])
            self.assertEqual(len(ADMITTED_VERDICTS) + len(REFUSED_VERDICTS), len(VERDICTS))

        def test_effective_session_is_the_next_session_not_the_stamp(self):
            self.assertEqual(effective_session('2025-01-03', DAYS), '2025-01-06')
            self.assertIsNone(effective_session('2025-01-08', DAYS))

        def test_price_row_makes_a_name_tradable(self):
            panel = _panel(priced=[('A', '2025-01-06')], sessions=DAYS)
            self.assertEqual(certify(panel, 'A', '2025-01-06')['verdict'], 'tradable')

        def test_full_day_halt_is_refuted_with_its_own_reason(self):
            panel = _panel(halt_full=[('A', '2025-01-06')], sessions=DAYS)
            got = certify(panel, 'A', '2025-01-06')
            self.assertEqual(got['verdict'], 'refused_halted_full_day')
            self.assertEqual(got['category'], 'refuted')

        def test_partial_halt_is_not_treated_as_tradable(self):
            panel = _panel(halt_partial=[('A', '2025-01-06')], sessions=DAYS)
            self.assertEqual(certify(panel, 'A', '2025-01-06')['verdict'],
                             'refused_halted_partial_day')

        def test_price_row_plus_halt_marker_is_a_conflict_not_an_admission(self):
            panel = _panel(priced=[('A', '2025-01-06')], halt_full=[('A', '2025-01-06')],
                           sessions=DAYS)
            got = certify(panel, 'A', '2025-01-06')
            self.assertEqual(got['verdict'], 'conflicting_price_and_halt')
            self.assertFalse(got['admitted'])

        def test_a_missing_record_outside_the_table_is_never_called_a_halt(self):
            panel = _panel(status_days=['2025-01-03'], terminal='2025-01-03', sessions=DAYS)
            got = certify(panel, 'A', '2025-01-06')
            self.assertEqual(got['verdict'], 'uncertifiable_status_table_absent')
            self.assertEqual(got['category'], 'uncertifiable')
            self.assertNotIn('halt', got['verdict'])

        def test_an_unexplained_gap_inside_the_table_is_uncertifiable(self):
            panel = _panel(status_days=['2025-01-06'], terminal='2025-01-06', sessions=DAYS)
            self.assertEqual(certify(panel, 'A', '2025-01-06')['verdict'],
                             'uncertifiable_no_price_row')

        def test_delisting_is_only_asserted_from_the_vendor_date(self):
            panel = _panel(status_days=['2025-01-06'], sessions=DAYS,
                           delist={'A': '2025-01-03'})
            got = certify(panel, 'A', '2025-01-06')
            self.assertEqual(got['verdict'], 'refused_delisted')
            self.assertEqual(got['delist_date'], '2025-01-03')

        def test_a_session_outside_the_calendar_cannot_be_an_entry(self):
            panel = _panel(sessions=DAYS)
            self.assertEqual(certify(panel, 'A', '2025-01-04')['verdict'],
                             'refused_not_a_trading_session')

        def test_off_mode_returns_the_input_untouched_and_claims_nothing(self):
            frames = {'2025-01-03': {'A': 0.5, 'B': 0.5}}
            panel = _panel(halt_full=[('A', '2025-01-06')], sessions=DAYS)
            got = gate(frames, DAYS, panel, mode='off')
            self.assertEqual(got['frames'], frames)
            self.assertEqual(got['rejects'], [])
            self.assertEqual(got['checked'], 0)
            self.assertIn('nothing was checked', got['outcome_label'])

        def test_annotate_moves_no_target_while_still_counting_every_refusal(self):
            frames = {'2025-01-03': {'A': 0.5, 'B': 0.5}}
            panel = _panel(priced=[('B', '2025-01-06')], status_days=['2025-01-06'],
                           terminal='2025-01-06',
                           halt_full=[('A', '2025-01-06')], sessions=DAYS)
            got = gate(frames, DAYS, panel, mode='annotate')
            self.assertEqual(got['frames'], frames)
            self.assertEqual(got['checked'], 2)
            self.assertEqual(got['refused'], 1)
            self.assertEqual(got['rejects'][0]['action'], 'reported_only')
            self.assertEqual(got['rejects'][0]['session'], '2025-01-06')

        def test_enforce_is_subtraction_only_and_keeps_the_weight_convention(self):
            frames = {'2025-01-03': {'A': 0.5, 'B': 0.5}}
            panel = _panel(priced=[('B', '2025-01-06')],
                           halt_full=[('A', '2025-01-06')], sessions=DAYS)
            got = gate(frames, DAYS, panel, mode='enforce')
            self.assertEqual(got['frames'], {'2025-01-03': {'B': 0.5}})
            self.assertLessEqual(set(got['frames']['2025-01-03']), set(frames['2025-01-03']))
            self.assertEqual(got['undeployed_weight'], 0.5)
            self.assertEqual(got['rejects'][0]['action'], 'refused')

        def test_a_stamp_with_no_effective_session_is_not_applicable_not_refused(self):
            frames = {'2025-01-08': {'A': 0.5, 'B': 0.5}}
            panel = _panel(halt_full=[('A', '2025-01-08')], sessions=DAYS)
            for mode in ('annotate', 'enforce'):
                got = gate(frames, DAYS, panel, mode=mode)
                # it must survive untouched: emptying it would be read as "sell everything"
                self.assertEqual(got['frames'], frames, mode)
                self.assertEqual(got['checked'], 0, mode)
                self.assertEqual(got['refused'], 0, mode)
                self.assertEqual(got['emptied_targets'], [], mode)
                self.assertEqual(got['names_not_applicable'], 2, mode)
                self.assertEqual(len(got['stamps_without_an_effective_session']), 1, mode)
            got = gate(frames, DAYS, panel, mode='annotate')
            self.assertEqual(got['output_names'], 2)

        def test_admitted_count_and_output_name_count_are_different_readings(self):
            frames = {'2025-01-03': {'A': 0.5, 'B': 0.5}}
            panel = _panel(priced=[('B', '2025-01-06')],
                           halt_full=[('A', '2025-01-06')], sessions=DAYS)
            observer = gate(frames, DAYS, panel, mode='annotate')
            gatekeeper = gate(frames, DAYS, panel, mode='enforce')
            self.assertEqual(observer['admitted'], gatekeeper['admitted'])
            self.assertNotEqual(observer['output_names'], gatekeeper['output_names'])
            self.assertEqual(observer['output_names'], 2)
            self.assertEqual(gatekeeper['output_names'], 1)

        def test_an_emptied_target_is_reported_not_silently_dropped(self):
            frames = {'2025-01-03': {'A': 0.5}}
            panel = _panel(priced=[('B', '2025-01-06')],
                           halt_full=[('A', '2025-01-06')], sessions=DAYS)
            got = gate(frames, DAYS, panel, mode='enforce')
            self.assertEqual(got['frames'], {'2025-01-03': {}})
            self.assertEqual(len(got['emptied_targets']), 1)
            self.assertEqual(got['emptied_targets'][0]['refused_names'], ['A'])

        def test_every_checked_name_lands_in_exactly_one_category(self):
            frames = {'2025-01-03': {'A': 0.5, 'B': 0.5, 'C': 0.5, 'D': 0.5},
                      '2025-01-06': {'E': 0.5}}
            panel = _panel(priced=[('B', '2025-01-06'), ('E', '2025-01-07')],
                           halt_full=[('C', '2025-01-06')],
                           status_days=['2025-01-06'], terminal='2025-01-06',
                           sessions=DAYS)
            got = gate(frames, DAYS, panel, mode='annotate')
            self.assertEqual(got['checked'], 5)
            self.assertEqual(sum(got['by_verdict'].values()), 5)
            self.assertEqual(len(got['rejects']), 3)
            self.assertEqual(got['output_names'], 5)
            self.assertEqual(got['by_category'],
                             {'admitted': 2, 'refuted': 1, 'uncertifiable': 2,
                              'conflicting': 0})

        def test_uncertifiable_is_counted_apart_from_refuted(self):
            frames = {'2025-01-03': {'A': 0.5, 'B': 0.5}}
            panel = _panel(halt_full=[('A', '2025-01-06')], status_days=[],
                           terminal=None, sessions=DAYS)
            got = gate(frames, DAYS, panel, mode='enforce')
            self.assertEqual(got['refused_refuted'], 1)
            self.assertEqual(got['refused_uncertifiable'], 1)
            self.assertNotEqual(got['refused_refuted'], got['refused'])

        def test_entry_events_collapse_a_continuously_held_name_to_one_event(self):
            frames = {'2025-01-02': {'A': 1.0}}
            self.assertEqual(entry_events(frames, DAYS),
                             [{'code': 'A', 'session': '2025-01-03'}])

        def test_a_name_that_leaves_and_returns_enters_twice(self):
            long_days = ['2025-01-02', '2025-01-03', '2025-01-06', '2025-01-07',
                         '2025-01-08', '2025-01-09', '2025-01-10']
            frames = {'2025-01-02': {'A': 1.0},
                      '2025-01-03': {'B': 1.0},
                      '2025-01-07': {'A': 1.0}}
            events = entry_events(frames, long_days)
            self.assertIn({'code': 'A', 'session': '2025-01-03'}, events)
            self.assertIn({'code': 'A', 'session': '2025-01-08'}, events)
            self.assertEqual(len([e for e in events if e['code'] == 'A']), 2)

        def test_span_endpoints_use_the_next_targets_effective_session(self):
            # the first target stays in force through the session on which the second
            # one takes effect; ending it one session early would silently split every
            # continuously held name into an extra entry event
            frames = {'2025-01-02': {'A': 1.0}, '2025-01-06': {'B': 1.0}}
            events = entry_events(frames, DAYS)
            self.assertEqual(events, [{'code': 'A', 'session': '2025-01-03'},
                                      {'code': 'B', 'session': '2025-01-07'}])

        def test_spans_outside_the_calendar_are_reported_not_silently_skipped(self):
            frames = {'2025-01-08': {'A': 1.0}}
            self.assertEqual(entry_events(frames, DAYS), [])
            skipped = skipped_stamps(frames, DAYS)
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]['stamp'], '2025-01-08')
            self.assertIn('calendar_ends', skipped[0]['why'])

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GateTests)
    return suite


if __name__ == '__main__':
    import unittest
    import sys

    result = unittest.TextTestRunner(verbosity=2).run(tests())
    sys.exit(0 if result.wasSuccessful() else 1)
