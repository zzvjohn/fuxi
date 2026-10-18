# -*- coding: utf-8 -*-
"""P4 批次：结算两项工程欠账。

欠账一 —— 抬高 `FUXI_COLLECT_CAL_END` 默认值（采集范围变更, 需独立批次 + 对照）
欠账二 —— 补 R0 锁与 manifest 覆盖（data/code 指纹集不含判据输入与本批脚本）

本脚本只读封存/本地文件, 只写 reports/p4_debts_20260924/。
不调用任何厂商 API, 不改任何数据文件, 不翻开任何默认值以外的开关。

用法:
    python scripts/fuxi_p4_debts_20260924.py --tests
    python scripts/fuxi_p4_debts_20260924.py --run
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/p4_debts_20260924'
TMP = ROOT / 'reports/remediation_20260922/_test_tmp/p4'

sys.path.insert(0, str(ROOT / 'strategies/final'))
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))

import collect_daily_full as C  # noqa: E402

ORIGINAL_DEFAULT = '20260430'
NEW_DEFAULT = '20261231'


# ------------------------------------------------------------------ 工具

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + '\n'
    data = raw.encode('utf-8')
    assert data[:3] != b'\xef\xbb\xbf'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(data)
    return len(data)


def write_csv(path, header, rows):
    import csv
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(header)
    for row in rows:
        w.writerow(row)
    data = buf.getvalue().encode('utf-8')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(data)
    return len(data)


def source_of(path):
    return Path(path).read_text(encoding='utf-8')


# ------------------------------------------------------------------ 欠账一

def packaged_open_days():
    """随包交易日历里 is_open=1 的日期集合 (紧凑形式)。

    🔴 这是**旁路诊断用**的替代物, 不是厂商日历: 本批不调用 API, 故凡是用到 all_days
    的读数都只能算「以随包日历为代理」的诊断, 不得与运行读数混排。
    """
    import csv
    p = ROOT / 'data/raw/trade_calendar.csv'
    out = set()
    with open(p, 'r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        di, oi = header.index('cal_date'), header.index('is_open')
        for row in reader:
            if len(row) <= max(di, oi):
                continue
            day = C.normalize_day(row[di])
            if day and str(row[oi]).strip() == '1':
                out.add(day)
    return out


def raw_existing_dates(fp):
    """**旧实现**的读数: 不规范化、不指定 dtype —— 只用于对照, 不用于生产。"""
    import pandas as pd
    col = pd.read_csv(fp, usecols=['trade_date'], low_memory=False)
    return set(col['trade_date'].astype(str).unique())


def debt1_readings():
    files = {k: C.FILES[k][0] for k in C.FILES}
    normalized = {k: C.get_existing_dates(fp) for k, fp in files.items()}
    audits = {k: C.date_format_audit(fp) for k, fp in files.items()}
    packaged_end = C.packaged_calendar_end()

    # ---- 对照: 旧默认 vs 新默认, 谁是「落后」的 ----
    control = {}
    for label, cal_end in (('old_default_%s' % ORIGINAL_DEFAULT, ORIGINAL_DEFAULT),
                           ('new_default_%s' % NEW_DEFAULT, NEW_DEFAULT)):
        stale = C.stale_horizons(cal_end, normalized, packaged_end)
        control[label] = {
            'cal_end': cal_end,
            'stale_reference_count': len(stale),
            'stale_references': stale,
            'verdict': 'refused' if stale else 'runs',
        }

    # ---- 缺陷 A: 日期格式混用会让「抬高默认值」直接变成重复追加 ----
    proxy_all = packaged_open_days()
    proxy_all_sorted = sorted(proxy_all)
    format_defect = {}
    for key, fp in files.items():
        audit = audits[key]
        if not audit.get('canonical_ok'):
            raw_days = raw_existing_dates(fp)
            old_missing = sorted(proxy_all - raw_days)
            new_missing = sorted(proxy_all - normalized[key])
            redo = sorted(set(old_missing) - set(new_missing))
            dup_rows = 0
            if redo:
                import csv
                basket = set(redo)
                with open(fp, 'r', encoding='utf-8-sig', newline='') as fh:
                    reader = csv.reader(fh)
                    header = next(reader)
                    di = header.index('trade_date')
                    for row in reader:
                        if len(row) > di and C.normalize_day(row[di]) in basket:
                            dup_rows += 1
            format_defect[key] = {
                'audit': audit,
                'days_that_would_be_re_downloaded_without_normalisation': len(redo),
                'first_such_day': redo[0] if redo else None,
                'last_such_day': redo[-1] if redo else None,
                'rows_that_would_be_duplicated': dup_rows,
                'normalised_day_count': len(normalized[key]),
                'raw_day_count': len(raw_days),
                'same_cardinality': len(normalized[key]) == len(raw_days),
                'why_not_visible_before': (
                    '在旧默认 cal_end=%s 下这些日期都在采集窗口之外, 所以「判成缺失」'
                    '没有任何后果; 一旦抬到 %s, 它们就落进窗口' % (ORIGINAL_DEFAULT, NEW_DEFAULT)),
            }

    # ---- 缺陷 B: 读文件本身在混合类型列上会崩 ----
    reader_defect = {'what': 'pd.read_csv(usecols=[...]) 在混合类型列上的行为',
                     'observed_in_this_interpreter': {
                         'pandas_version': None, 'raw_call_raises': None,
                         'raw_exception': None, 'fixed_call_raises': None}}
    import pandas as pd
    reader_defect['observed_in_this_interpreter']['pandas_version'] = pd.__version__
    limit_fp = files['limit']
    try:
        pd.read_csv(limit_fp, usecols=['trade_date'])
        reader_defect['observed_in_this_interpreter']['raw_call_raises'] = False
    except Exception as exc:                       # noqa: BLE001 — 就是要抓这个
        reader_defect['observed_in_this_interpreter']['raw_call_raises'] = True
        reader_defect['observed_in_this_interpreter']['raw_exception'] = \
            '%s: %s' % (type(exc).__name__, exc)
    try:
        pd.read_csv(limit_fp, usecols=['trade_date'], dtype=str, low_memory=False)
        reader_defect['observed_in_this_interpreter']['fixed_call_raises'] = False
    except Exception as exc:                       # noqa: BLE001
        reader_defect['observed_in_this_interpreter']['fixed_call_raises'] = True
        reader_defect['observed_in_this_interpreter']['fixed_exception'] = str(exc)
    reader_defect['consequence_before_the_fix'] = (
        '采集脚本第 [2] 步「检查已有数据」就要读这四个文件 ⇒ 未修之前脚本会在采集之前'
        '就抛 IndexError, 连一次 API 调用都发不出去。')

    # 生产解释器 (系统 python, tushare 所在) 上的实测: 不崩, 但会静默误判
    probe = (
        'import pandas as pd, warnings, sys\n'
        'warnings.simplefilter("error", pd.errors.DtypeWarning)\n'
        'try:\n'
        '    pd.read_csv(%r, usecols=["trade_date"])\n'
        '    print("strict:no_warning")\n'
        'except pd.errors.DtypeWarning:\n'
        '    print("strict:DtypeWarning")\n'
        'except Exception as exc:\n'
        '    print("strict:%%s" %% type(exc).__name__)\n'
        % str(limit_fp)
    )
    prod = {'interpreter': 'C:/Python314/python.exe'}
    try:
        got = subprocess.run([prod['interpreter'], '-X', 'utf8', '-c', probe],
                             capture_output=True, text=True, timeout=180)
        prod['pandas_version'] = subprocess.run(
            [prod['interpreter'], '-X', 'utf8', '-c', 'import pandas;print(pandas.__version__)'],
            capture_output=True, text=True, timeout=180).stdout.strip()
        prod['strict_probe'] = (got.stdout or '').strip().splitlines()[-1] if got.stdout else None
    except Exception as exc:                       # noqa: BLE001
        prod['error'] = str(exc)
    reader_defect['observed_in_the_production_interpreter'] = prod
    reader_defect['why_it_still_matters'] = (
        '生产解释器是 pandas 2.x: 不崩, 只发一条 DtypeWarning —— 也就是说它**不会拦住你**, '
        '而是把混合类型静默读进来, 再由旧 get_existing_dates 把短横线日期原样存进集合, '
        '于是「判成缺失」这件事在两种解释器上都成立, 只是一个崩、一个不吭声。')

    # ---- 只增不改: 源码级证据 ----
    src = source_of(ROOT / 'strategies/final/collect_daily_full.py')
    additive = {
        'append_uses_mode_a': "mode='a'" in src,
        'write_set_is_driven_by_missing_days': 'missing_map[key] = set(miss)' in src and
                                               'miss = [d for d in work_days if d not in existing]' in src,
        'future_sessions_are_split_out_before_the_write_set':
            'work_days, future_days = not_yet_available_sessions(all_days, today)' in src,
        'no_row_is_ever_rewritten': src.count("to_csv(fp, mode='a'") == 1,
        'normalisation_rejects_instead_of_filling': 'raise ValueError' in src and
                                                    '含无法规范化的值' in src,
        'note': ('本脚本只按「缺失日期」追加, 从不改写已有行; 规范化把两种日期写法并成一种, '
                 '遇到不可解析的值是**抛错**而不是填一个默认值。'
                 '「缺失日期」的定义已收缩到**已发生**的交易日 —— 尚未发生的日子走另一个口径。'),
    }

    # ---- 缺陷 C: 抬高默认值的**镜像** —— 把「尚未发生」当成「缺失」 ----
    today = datetime.now().strftime('%Y%m%d')
    env_today = os.environ.get('FUXI_COLLECT_TODAY')
    work_days, future_days = C.not_yet_available_sessions(proxy_all_sorted, today)
    in_sync, sync_note = C.default_is_in_sync_with_the_packaged_calendar(packaged_end)
    calendar_defect = {
        'what': '抬高 cal_end 后, 交易日历本身**延伸进未来**; 若不区分「尚未发生」, '
                '未来日会被当成「缺失」',
        'today': today,
        'env_override_for_today_respected': env_today,
        'packaged_open_days': len(proxy_all_sorted),
        'already_happened': len(work_days),
        'not_yet_available': len(future_days),
        'first_not_yet_available': future_days[0] if future_days else None,
        'last_not_yet_available': future_days[-1] if future_days else None,
        'api_calls_that_would_be_wasted_per_run': len(future_days) * len(files),
        'last_update_txt_would_have_claimed': future_days[-1] if future_days else None,
        'default_value_in_sync_with_packaged_calendar': in_sync,
        'sync_note': sync_note,
        'consequence_if_not_fixed': (
            '① 每一天都真去调 5 次厂商 API, 拿到空表后记成「成功」; '
            '② 完整性检查永久报「仍缺 N 天」, 健康的库看起来永远不完整; '
            '③ last_update.txt 的 last_trade_date 被写成**未来日期** —— 下游若读它, '
            '就会以为未来已有数据。三条都是静默的, 与老 bug 同一个失败模式, 只是方向相反。'),
        'how_it_is_fixed': (
            'not_yet_available_sessions(all_days, today) 把交易日历拆成 '
            '已发生(应采) / 尚未发生(不采、不计缺失), 且两个数都进读数; '
            'default_is_in_sync_with_the_packaged_calendar() 把「日历延伸了但默认值没跟上」'
            '变成一次拒绝运行 (exit 3)。'),
    }
    debt = {
        'default_value': {'before': ORIGINAL_DEFAULT, 'after': NEW_DEFAULT,
                          'packaged_calendar_end': packaged_end,
                          'why_this_value': '与随包交易日历末端对齐 (项目自己声明的日历视野)'},
        'control_old_vs_new': control,
        'defect_A_date_format_mixed': format_defect,
        'defect_B_reader_on_mixed_dtype_column': reader_defect,
        'defect_C_future_sessions_taken_for_missing': calendar_defect,
        'additive_only_evidence': additive,
        'files_in_scope': {k: {'path': str(Path(v).relative_to(ROOT)).replace('\\', '/'),
                               'existing_days_normalised': len(normalized[k]),
                               'canonical': audits[k].get('canonical_ok')}
                           for k, v in files.items()},
        'hashes': {str(Path(v).relative_to(ROOT)).replace('\\', '/'): sha256_file(v)
                   for v in files.values()},
    }
    sets = {
        'proxy_calendar_note': ('以下 miss 数用**随包交易日历**的代开市日集合代替厂商日历 —— '
                                '属旁路诊断, 不是运行读数; 本批不调用 API。'),
        'packaged_open_days': len(proxy_all),
        'old_default_missing_days_per_file': {k: len(proxy_all - raw_existing_dates(v))
                                              for k, v in files.items()},
        'new_default_missing_days_per_file': {k: len(proxy_all - normalized[k])
                                              for k, v in files.items()},
    }
    return debt, sets, files


# ------------------------------------------------------------------ 欠账二

def debt2_readings():
    from run_contract import RunContract, _CORE_CODE, _DATA_SENTINELS, _manifest_key

    TMP.mkdir(parents=True, exist_ok=True)
    code = TMP / 'p4_batch_script.py'
    code.write_text('# batch\n', encoding='utf-8')
    data = TMP / 'p4_batch_table.csv'
    data.write_text('trade_date\n20260104\n', encoding='utf-8')

    runs = TMP / 'runs'
    lock = TMP / 'runtime.lock'
    # 🔴 R0 契约: run_id 不复用。第一次 --run (断言失败那次) 已占用 'p4' 并把状态留在
    #    'completed', 第二次直接抛 RunReuseError —— 这正是契约该有的行为, 不是环境问题。
    #    为让这条读数**确定性可复现** (而非依赖上次残留), 这里就地造一个探针 run:
    #    先正常完成一次, 再用同一 id 进一次, 必须被拒。
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    probe_id = 'p4_reuse_probe_%s' % stamp
    reused_run_id_refused = None
    with RunContract(parameters={}, run_id=probe_id, runs_dir=runs, lock_path=lock):
        pass
    try:
        with RunContract(parameters={}, run_id=probe_id, runs_dir=runs, lock_path=lock):
            pass
    except Exception as exc:
        reused_run_id_refused = '%s: %s' % (type(exc).__name__, exc)

    with RunContract(parameters={'batch': 'p4_debts_20260924'}, run_id='p4_%s' % stamp,
                     runs_dir=runs, lock_path=lock,
                     extra_code=[ROOT / 'scripts/fuxi_p4_debts_20260924.py', code],
                     extra_data=[ROOT / 'data/raw/delist_dates.csv', data]) as contract:
        manifest = json.loads(Path(contract.manifest_path).read_text(encoding='utf-8'))

    old_data_keys = ['data/raw/daily_prices.csv', 'data/raw/moneyflow_daily.csv',
                     'data/stage1_factor_proposals.json', 'data/mab_scheduler_state.json',
                     'data/experience_memory.json']
    all_data_keys = sorted(manifest['data_fingerprints'])
    cover = manifest['fingerprint_coverage']
    # 🔴 manifest 里的 data 键 = 标准集 + 本批声明的额外数据。两个集合必须分开报:
    #    ① 把「重复声明」混进「标准集新增」会把「本批自己登记的两条」误读成「标准集扩了六条」;
    #    ② 反过来, 把「已经在标准集里的声明」当成额外键剔掉, 会把新增数从 4 数成 3
    #       —— 本批第一次 --run 正是这样断言失败在 keys_added_count。
    #       extra_*_already_standard 才是「重复声明」这个第三类, 它既不是新增也不是额外。
    extra_keys = list(cover['extra_data'])
    newly_added_by_declaration = list(cover['extra_data_newly_added'])
    redeclared = list(cover['extra_data_already_standard'])
    standard_keys = [k for k in all_data_keys if k not in newly_added_by_declaration]
    added = sorted(set(standard_keys) - set(old_data_keys))

    declared_missing = None
    try:
        RunContract(parameters={}, run_id='p4_missing_%s' % stamp, runs_dir=runs, lock_path=lock,
                    extra_code=[TMP / 'does_not_exist.py'])
    except ValueError as exc:
        declared_missing = str(exc)

    outside_dir = Path(tempfile.mkdtemp(prefix='fuxi_p4_outside_'))
    outside = outside_dir / 'outside_script.py'
    outside.write_text('# outside\n', encoding='utf-8')
    outside_key = _manifest_key(outside)

    return {
        'standard_code_set_size': len(_CORE_CODE),
        'standard_data_set_size': len(_DATA_SENTINELS),
        'manifest_data_keys': all_data_keys,
        'manifest_key_count': len(all_data_keys),
        'standard_keys_added': added,
        'keys_added_to_the_standard_set': added,
        'keys_added_count': len(added),
        'extra_declared_keys': extra_keys,
        'extra_declared_but_already_standard': redeclared,
        'extra_declared_newly_added': newly_added_by_declaration,
        'coverage_block': manifest['fingerprint_coverage'],
        'extra_code_registered': manifest['fingerprint_coverage']['extra_code'],
        'extra_data_registered': manifest['fingerprint_coverage']['extra_data'],
        'code_entry_carries_sha256': all(
            len(v.get('sha256') or '') == 64 for v in manifest['code_fingerprints'].values()),
        'data_entry_carries_no_sha256': all(
            'sha256' not in v for v in manifest['data_fingerprints'].values()),
        'declared_but_missing_is_refused': declared_missing is not None,
        'declared_but_missing_message': declared_missing,
        'reused_run_id_is_refused': reused_run_id_refused,
        'run_ids_are_never_reused': reused_run_id_refused is not None,
        'path_outside_project_keeps_an_absolute_key': outside_key,
        'keys_use_posix_separators': all('\\' not in k for k in all_data_keys),
        'what_was_wrong_before': (
            '① data 标准集只有 5 个键, 而判据输入 suspend_d.csv / delist_dates.csv / '
            'stock_list.csv / trade_calendar.csv 一个都不在 ⇒ manifest 声称覆盖数据, '
            '却对「整日停牌」与「退市日期」这两条判据的输入一无所知; '
            '② code 标准集是 9 个固定模块, 批次自己的脚本**结构上无法登记** —— '
            '「本批脚本不在指纹里」不是偶发遗漏; '
            '③ 键用平台分隔符 (反斜杠), 跨平台 manifest 无法逐字比较。'),
    }


# ------------------------------------------------------------------ 测试

class CollectHorizonTests(unittest.TestCase):
    def test_normalize_day_accepts_both_writings(self):
        for value in ('20260104', '2026-01-04', '2026/01/04'):
            self.assertEqual(C.normalize_day(value), '20260104')
        for bad in ('', 'nan', 'None', 'NaT', '2026-1-4', 'abcdefgh'):
            self.assertEqual(C.normalize_day(bad), '')

    def test_stale_horizons_flags_only_strictly_later_references(self):
        existing = {'a': {'20260104', '20260630'}}
        self.assertEqual(C.stale_horizons('20260630', existing), {})
        self.assertEqual(C.stale_horizons('20260104', existing), {'a': '20260630'})
        self.assertEqual(C.stale_horizons('20260104', existing, '20261231'),
                         {'packaged_trade_calendar': '20261231', 'a': '20260630'})

    def test_stale_horizons_ignores_unparsable_values(self):
        self.assertEqual(C.stale_horizons('20260104', {'a': {'', 'nan', 'x'}}), {})

    def test_append_df_refuses_a_value_it_cannot_canonicalise(self):
        import pandas as pd
        target = TMP / 'append_probe.csv'
        TMP.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        good = pd.DataFrame({'ts_code': ['000001.SZ'], 'trade_date': ['2026-01-04']})
        C.append_df(str(target), good)
        rows = target.read_text(encoding='utf-8-sig').strip().splitlines()
        self.assertEqual(rows[-1], '000001.SZ,20260104')
        bad = pd.DataFrame({'ts_code': ['000001.SZ'], 'trade_date': ['oops']})
        with self.assertRaises(ValueError):
            C.append_df(str(target), bad)
        rows_after = target.read_text(encoding='utf-8-sig').strip().splitlines()
        self.assertEqual(rows_after, rows)

    def test_packaged_calendar_end_skips_header_like_rows(self):
        got = C.packaged_calendar_end()
        self.assertIsNotNone(got)
        self.assertTrue(got.isdigit() and len(got) == 8)

    # ---- 缺陷 C: 「尚未发生」必须与「缺失」分开 ----

    def test_future_sessions_are_split_out_of_the_work_set(self):
        days = ['20260923', '20260924', '20260925', '20260928', '20261231']
        work, future = C.not_yet_available_sessions(days, '20260925')
        self.assertEqual(work, ['20260923', '20260924', '20260925'])
        self.assertEqual(future, ['20260928', '20261231'])

    def test_a_session_equal_to_today_is_still_worked_on(self):
        """边界取「严格晚于 today」: today 当天仍要采 (否则当天数据永远补不上)。"""
        work, future = C.not_yet_available_sessions(['20260924', '20260925'], '20260925')
        self.assertEqual(work, ['20260924', '20260925'])
        self.assertEqual(future, [])

    def test_calendar_extension_without_moving_the_default_is_refused(self):
        """🔴 日历延伸而默认值没跟上 ⇒ 必须判不同步 (否则补采重新被静默截断)。"""
        ok, why = C.default_is_in_sync_with_the_packaged_calendar('20271231', '20261231')
        self.assertFalse(ok)
        self.assertIn('20271231', why)
        ok, _ = C.default_is_in_sync_with_the_packaged_calendar('20261231', '20261231')
        self.assertTrue(ok)
        ok, why = C.default_is_in_sync_with_the_packaged_calendar(None, '20261231')
        self.assertTrue(ok, '日历缺失时不应阻止运行 (另有 packaged_calendar_end 返回 None 的口径)')

    def test_the_shipped_default_matches_the_shipped_calendar(self):
        """默认值即契约: 仓库里的 CAL_END_DEFAULT 必须等于随包日历末端。"""
        self.assertEqual(C.CAL_END_DEFAULT, C.packaged_calendar_end())


def tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(CollectHorizonTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tests', action='store_true')
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args(argv)

    if args.tests:
        return tests()
    if not args.run:
        ap.print_help()
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    debt1, sets, files = debt1_readings()
    debt2 = debt2_readings()
    # ---- 断言 (fail-closed): 读数不成立就不要落盘 ----
    fmt = debt1['defect_A_date_format_mixed']
    assert fmt, '缺陷 A 没有被复现 —— 读数不可信'
    assert debt1['control_old_vs_new']['old_default_%s' % ORIGINAL_DEFAULT][
        'verdict'] == 'refused', '旧默认必须被判为落后'
    assert debt1['control_old_vs_new']['new_default_%s' % NEW_DEFAULT][
        'verdict'] == 'runs', '新默认必须通过落后检查'
    assert debt1['additive_only_evidence']['no_row_is_ever_rewritten'], '只增不改证据不足'
    cal_def = debt1['defect_C_future_sessions_taken_for_missing']
    assert cal_def['not_yet_available'] > 0, \
        '交易日历里应有「尚未发生」的日子 —— 否则缺陷 C 的读数是空的 (形状不对)'
    assert cal_def['api_calls_that_would_be_wasted_per_run'] > 0, '缺陷 C 的后果必须可数'
    assert cal_def['default_value_in_sync_with_packaged_calendar'], \
        '默认采集末端必须与随包日历末端一致'
    assert debt2['keys_added_count'] == 4, debt2['keys_added_count']
    assert debt2['declared_but_missing_is_refused'], '声明缺失必须被拒'
    assert debt2['run_ids_are_never_reused'], '复用 run_id 必须被拒 (R0 契约)'
    assert debt2['keys_use_posix_separators'], '键必须用 posix 分隔符'

    payload = {
        'batch': 'p4_debts_20260924',
        'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'purpose': ('P4: 结算两项工程欠账 —— 抬高 FUXI_COLLECT_CAL_END 默认值 + 补 R0 锁与 '
                    'manifest 覆盖。本批不调用厂商 API、不改任何数据文件。'),
        'scope_note': ('只读本地文件; 不翻任何开关; 不改账本/S5/JQ 准入; '
                       '所有涉及 all_days 的读数都标注为「以随包日历为代理」的旁路诊断。'),
        'debt1_calendar_default': debt1,
        'debt1_bypass_diagnostics': sets,
        'debt2_manifest_coverage': debt2,
        'claims_not_made': [
            '没有调用任何厂商 API —— 「抬高默认值后能补到多少天」不是本批的读数',
            '没有改写任何数据文件: 日期格式混用被**报告**, 没有被就地修复',
            '没有把混合格式当成「可容忍」: append_df 在写入口只允许一种形式, 不可解析即抛错',
            '没有把 manifest 封面说成「完整覆盖」: extra_* 是声明而非自动发现, 未声明的输入仍不在集内',
            '没有翻开 R0 锁: 本批只是让「登记本批脚本/数据」在结构上成为可能',
            '没有把旁路诊断(随包日历代理)与运行读数混排',
            '没有把「run_id 被拒复用」当成新功能: 它本来就是契约行为, 本批只是把它读出来',
        ],
    }
    size = write_json(OUT / 'readings.json', payload)

    rows = []
    for key, block in fmt.items():
        rows.append([key, block['audit']['compact'], block['audit']['dashed'],
                     block['audit']['other'], block['audit']['first_dashed_line'],
                     block['days_that_would_be_re_downloaded_without_normalisation'],
                     block['rows_that_would_be_duplicated']])
    write_csv(OUT / 'date_format_mixed.csv',
              ['file', 'compact_rows', 'dashed_rows', 'other_rows', 'first_dashed_line',
               'days_re_downloaded_without_normalisation', 'rows_duplicated'], rows)

    write_csv(OUT / 'manifest_keys.csv', ['key', 'kind'],
              [['data/raw/daily_prices.csv', 'standard(existed)'],
               ['data/raw/moneyflow_daily.csv', 'standard(existed)'],
               ['data/stage1_factor_proposals.json', 'standard(existed)'],
               ['data/mab_scheduler_state.json', 'standard(existed)'],
               ['data/experience_memory.json', 'standard(existed)']] +
              [[k, 'standard(new)'] for k in debt2['keys_added_to_the_standard_set']] +
              [[k, 'extra_data(declared)'] for k in debt2['extra_data_registered']])

    print('== 欠账一 日历默认值 ==')
    print('  默认值 %s -> %s (随包日历末端 %s)'
          % (ORIGINAL_DEFAULT, NEW_DEFAULT, debt1['default_value']['packaged_calendar_end']))
    for label, block in debt1['control_old_vs_new'].items():
        print('  对照 %-22s stale=%-2d -> %s'
              % (label, block['stale_reference_count'], block['verdict']))
    for key, block in fmt.items():
        print('  缺陷A %-8s dashed=%-7d 首行 %-9s 不规范化会重复追加 %d 天 / %d 行'
              % (key, block['audit']['dashed'], block['audit']['first_dashed_line'],
                 block['days_that_would_be_re_downloaded_without_normalisation'],
                 block['rows_that_would_be_duplicated']))
    rd = debt1['defect_B_reader_on_mixed_dtype_column']
    print('  缺陷B %s 原始调用抛错=%s / 修正后抛错=%s / 生产解释器=%s'
          % (rd['observed_in_this_interpreter']['pandas_version'],
             rd['observed_in_this_interpreter']['raw_call_raises'],
             rd['observed_in_this_interpreter']['fixed_call_raises'],
             rd['observed_in_the_production_interpreter'].get('strict_probe')))
    print('  只增不改: %s' % debt1['additive_only_evidence']['note'])
    print('  缺陷C 交易日历 %d 天 = 已发生 %d + 尚未发生 %d (%s..%s)'
          % (sets['packaged_open_days'], cal_def['already_happened'],
             cal_def['not_yet_available'], cal_def['first_not_yet_available'],
             cal_def['last_not_yet_available']))
    print('        若不修: 空转 API %d 次/轮, 永久误报缺 %d 天, last_update.txt 会写 %s'
          % (cal_def['api_calls_that_would_be_wasted_per_run'],
             cal_def['not_yet_available'], cal_def['last_update_txt_would_have_claimed']))
    print('        默认值与随包日历同步: %s' % cal_def['sync_note'])
    print('== 欠账二 manifest 覆盖 ==')
    print('  标准集 data %d 键 (新增 %d) / code %d 模块; 声明即须存在=%s; posix 键=%s'
          % (debt2['manifest_key_count'], debt2['keys_added_count'],
             debt2['standard_code_set_size'], debt2['declared_but_missing_is_refused'],
             debt2['keys_use_posix_separators']))
    print('  新增: %s' % ', '.join(debt2['keys_added_to_the_standard_set']))
    print('  R0 复用守卫: %s' % debt2['reused_run_id_is_refused'])
    print('readings.json %d bytes' % size)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
