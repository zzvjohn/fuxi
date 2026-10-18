import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))

import pandas as pd

from fuxi_delist_channel_20260924 import (
    explicit_facts, load_delist_dates, merge_freeze_facts,
)

HEADER = 'ts_code,name,list_status,list_date,delist_date,market'


def write_table(rows, newline='\r\n', bom=True):
    text = newline.join([HEADER] + rows) + newline
    handle = tempfile.NamedTemporaryFile('wb', suffix='.csv', delete=False)
    with handle:
        handle.write(text.encode('utf-8-sig' if bom else 'utf-8'))
    return handle.name


def prices(pairs):
    return pd.DataFrame({'ts_code': [c for c, _ in pairs],
                         'trade_date': [d for _, d in pairs]})


class DelistTableTests(unittest.TestCase):
    def test_reads_a_bom_crlf_table_as_iso_dates(self):
        path = write_table(['600355.SH,\u7cbe\u4f26,D,20020613,20260427,\u4e3b\u677f'])
        try:
            self.assertEqual(load_delist_dates(path), {'600355.SH': '2026-04-27'})
        finally:
            os.unlink(path)

    def test_rejects_a_table_with_bare_lf(self):
        path = write_table(['600355.SH,x,D,20020613,20260427,zh'], newline='\n')
        try:
            with self.assertRaises(AssertionError):
                load_delist_dates(path)
        finally:
            os.unlink(path)

    def test_rejects_a_table_without_a_bom(self):
        path = write_table(['600355.SH,x,D,20020613,20260427,zh'], bom=False)
        try:
            with self.assertRaises(AssertionError):
                load_delist_dates(path)
        finally:
            os.unlink(path)

    def test_skips_listed_rows_and_rows_without_a_date(self):
        path = write_table(['000001.SZ,a,L,19910403,,zh',
                            '600355.SH,b,D,20020613,20260427,zh'])
        try:
            self.assertEqual(load_delist_dates(path), {'600355.SH': '2026-04-27'})
        finally:
            os.unlink(path)

    def test_rejects_a_malformed_date(self):
        path = write_table(['600355.SH,x,D,20020613,2026-04-27,zh'])
        try:
            with self.assertRaises(AssertionError):
                load_delist_dates(path)
        finally:
            os.unlink(path)


class ExplicitFactTests(unittest.TestCase):
    sessions = ['2026-04-24', '2026-04-27', '2026-04-28']

    def test_accepts_a_delist_date_after_the_last_quote(self):
        frame = prices([('A', '2026-04-03')])
        accepted, refused = explicit_facts(frame, self.sessions, {'A': '2026-04-27'})
        self.assertEqual(refused, [])
        self.assertEqual(accepted['A']['date'], '2026-04-27')
        self.assertEqual(accepted['A']['channel'], 'vendor_delist_date')
        self.assertIn('2026-04-03', accepted['A']['evidence'])

    def test_refuses_a_name_quoted_on_the_delist_date(self):
        frame = prices([('A', '2026-04-27')])
        accepted, refused = explicit_facts(frame, self.sessions, {'A': '2026-04-27'})
        self.assertEqual(accepted, {})
        self.assertEqual([r['reason'] for r in refused],
                         ['the window still quotes the name on or after the '
                          'vendor delist date'])

    def test_refuses_a_delist_date_outside_the_window_sessions(self):
        frame = prices([('A', '2026-04-03')])
        accepted, refused = explicit_facts(frame, self.sessions, {'A': '2026-05-06'})
        self.assertEqual(accepted, {})
        self.assertEqual([r['reason'] for r in refused],
                         ['the vendor delist date is not a session of the window'])

    def test_ignores_names_without_a_vendor_delist_date(self):
        frame = prices([('A', '2026-04-03'), ('B', '2026-04-03')])
        accepted, refused = explicit_facts(frame, self.sessions, {'A': '2026-04-27'})
        self.assertEqual(set(accepted), {'A'})
        self.assertEqual(refused, [])

    def test_uses_the_last_quote_of_the_window_not_of_the_table(self):
        frame = prices([('A', '2026-04-03'), ('A', '2026-04-28')])
        accepted, refused = explicit_facts(frame, self.sessions, {'A': '2026-04-27'})
        self.assertEqual(accepted, {})
        self.assertEqual(len(refused), 1)


class MergeTests(unittest.TestCase):
    def test_adds_a_name_the_legacy_channel_never_reached(self):
        explicit = {'A': {'date': '2026-04-27', 'last_quote': '2026-04-03',
                          'evidence': 'x', 'channel': 'vendor_delist_date'}}
        facts, report = merge_freeze_facts({}, explicit)
        self.assertEqual(set(facts), {'A'})
        self.assertEqual(len(report['additions']), 1)
        self.assertEqual(report['overrides'], [])
        self.assertEqual(report['agreement'], [])

    def test_records_agreement_when_both_channels_speak(self):
        legacy = {'A': {'date': '2026-04-27', 'evidence': 'derived'}}
        explicit = {'A': {'date': '2026-04-27', 'last_quote': '2026-04-03',
                          'evidence': 'x', 'channel': 'vendor_delist_date'}}
        facts, report = merge_freeze_facts(legacy, explicit)
        self.assertEqual(facts['A']['date'], '2026-04-27')
        self.assertIn('derived', facts['A']['evidence'])
        self.assertEqual(len(report['agreement']), 1)
        self.assertEqual(report['overrides'], [])

    def test_prefers_the_vendor_date_and_publishes_the_disagreement(self):
        legacy = {'A': {'date': '2026-05-06', 'evidence': 'derived from the suspend tail'}}
        explicit = {'A': {'date': '2026-04-27', 'last_quote': '2026-04-03',
                          'evidence': 'x', 'channel': 'vendor_delist_date'}}
        facts, report = merge_freeze_facts(legacy, explicit)
        self.assertEqual(facts['A']['date'], '2026-04-27')
        self.assertEqual(report['agreement'], [])
        self.assertEqual(len(report['overrides']), 1)
        row = report['overrides'][0]
        self.assertEqual(row['vendor_delist_date'], '2026-04-27')
        self.assertEqual(row['suspend_derived_freeze'], '2026-05-06')
        self.assertTrue(row['moved_earlier'])
        self.assertIn('suspend', row['suspend_derived_evidence'])

    def test_never_drops_a_legacy_fact(self):
        legacy = {'A': {'date': '2025-01-02', 'evidence': 'derived'}}
        explicit = {'B': {'date': '2026-04-27', 'last_quote': '2026-04-03',
                          'evidence': 'x', 'channel': 'vendor_delist_date'}}
        facts, _ = merge_freeze_facts(legacy, explicit)
        self.assertEqual(set(facts), {'A', 'B'})
        self.assertEqual(facts['A']['date'], '2025-01-02')

    def test_does_not_mutate_the_inputs(self):
        legacy = {'A': {'date': '2026-05-06', 'evidence': 'derived'}}
        explicit = {'A': {'date': '2026-04-27', 'last_quote': '2026-04-03',
                          'evidence': 'x', 'channel': 'vendor_delist_date'}}
        before = dict(legacy['A'])
        merge_freeze_facts(legacy, explicit)
        self.assertEqual(legacy['A'], before)


if __name__ == '__main__':
    unittest.main()
