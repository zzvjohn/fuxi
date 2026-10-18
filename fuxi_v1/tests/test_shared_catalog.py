"""Federation/equivalence contracts, including actual missingness and barriers.

Fixtures do not grant market data assurance, economic acceptance or JQ access.
"""
from copy import deepcopy
import csv
from datetime import date, timedelta
from pathlib import Path
import unittest

from fuxi_v1.contracts import ContractError, digest, load_json
from fuxi_v1.datasets import file_hash
from fuxi_v1.dsl import evaluate_ast
from fuxi_v1.historical_adapter import HistoricalSnapshot
from fuxi_v1.shared_catalog import (build_catalog, data_contracts, signal_contract,
    translate_expression, reverse_projection, prove_representatives, REPRESENTATIVES,
    NATIVE_FIELDS, _legacy_values, _write_new)
from fuxi_v1.tests.test_kernel import scratch_directory


def history(n=150, codes=64):
    days = [(date(2001, 1, 1) + timedelta(days=i)).isoformat() for i in range(n)]
    instruments = ['%06d.XSHG' % (600000+i) for i in range(codes)]
    columns = {f: [[float(10+i*.01+j*.001) if f == 'close' else float(10000+i*17+j*2)
                    for j in range(codes)] for i in range(n)] for f in ('close', 'volume')}
    return {'schema_version': 'fuxi.historical_snapshot.1', 'exposure': 'research_exposed',
        'sessions': days, 'evaluation_start': days[120], 'universe': {d: list(instruments) for d in days},
        'decision_times': {d: d+'T18:30:00+08:00' for d in days}, 'instruments': instruments,
        'columns': columns, 'barriers': [[-1]*codes for _ in days],
        'fields': {f: NATIVE_FIELDS[f] for f in columns}}


class SharedCatalogTests(unittest.TestCase):
    def test_exact_three_existing_expression_shapes_roundtrip(self):
        for name, expression in REPRESENTATIVES:
            mapped = translate_expression(expression)
            projected = reverse_projection(mapped['canonical_ast'])
            self.assertEqual(translate_expression(projected['formula'])['canonical_ast'], mapped['canonical_ast'])
            self.assertEqual(projected['semantic_factor_id'], mapped['semantic_factor_id'])
            self.assertFalse(projected['eligible_for_jq'])
            self.assertIsNone(projected['qualification_receipt_id'])

    def test_unsupported_fill_axis_constant_or_python_rejected(self):
        expressions = ('close_p.pct_change()', 'close_p.rank()', 'close_p.ffill()',
            'close_p + 0.001', '__import__("os").getcwd()',
            'close_p.rolling(5).mean(axis=1)', '(close_p+volume_p).rolling(5).mean()',
            'turnover', 'turnover_p')
        for expression in expressions:
            with self.subTest(expression=expression), self.assertRaises(ContractError):
                translate_expression(expression)

    def test_unit_mismatch_rejected(self):
        with self.assertRaises(ContractError): translate_expression('close_p + volume_p')

    def test_name_and_format_never_define_semantic_equivalence(self):
        first = translate_expression('close_p / close_p.rolling(20).max()')
        second = translate_expression('(close_p)/(close_p.rolling(20).max())')
        self.assertEqual(first['semantic_factor_id'], second['semantic_factor_id'])
        changed = deepcopy(first['semantic_contract']); changed['missing'] = 'zero'
        self.assertNotEqual(digest(changed), first['semantic_contract']['semantic_contract_hash'])

    def test_true_turnover_and_s5_proxy_stay_distinct(self):
        contracts = data_contracts(); rows = {r['field']: r for r in contracts['fields']}
        self.assertEqual(rows['turnover_rate']['native_unit'], 'ratio')
        self.assertEqual(rows['legacy_S5_turnover']['native_unit'], 'shares/price')
        self.assertTrue(contracts['turnover_is_not_turnover_rate'])
        self.assertFalse(contracts['qualification_granted'])

    def test_actual_field_metadata_binds_identity_and_adjusted_price_rejects(self):
        raw={'close':{'unit':'price','frequency':'daily','adjustment':'raw'}}
        other={'close':{'unit':'price','frequency':'daily','adjustment':'not_applicable'}}
        expression=REPRESENTATIVES[1][1]
        self.assertNotEqual(translate_expression(expression,raw)['semantic_factor_id'],
                            translate_expression(expression,other)['semantic_factor_id'])
        adjusted={'close':{'unit':'price','frequency':'daily','adjustment':'adjusted'}}
        with self.assertRaises(ContractError):translate_expression(expression,adjusted)

    def test_real_pandas_and_native_keep_nan_and_action_barriers(self):
        payload = history(codes=3)
        payload['columns']['close'][144][0] = None
        payload['barriers'][149][1] = 145
        snapshot = HistoricalSnapshot(payload)
        for _, expression in REPRESENTATIVES:
            native = evaluate_ast(translate_expression(expression, payload['fields'])['canonical_ast'], snapshot)
            legacy = _legacy_values(expression, snapshot, 149)
            for code in snapshot.universe[payload['sessions'][149]]:
                a, b = legacy[code], native[payload['sessions'][149]][code]
                self.assertEqual(a is None, b is None)
                if a is not None: self.assertAlmostEqual(a, b, places=12)
        last = evaluate_ast(translate_expression(REPRESENTATIVES[-1][1])['canonical_ast'], snapshot)[payload['sessions'][149]]
        self.assertIsNone(last[payload['instruments'][0]])
        self.assertIsNone(last[payload['instruments'][1]])
        self.assertIsNotNone(last[payload['instruments'][2]])

    def test_zero_divisor_nulls_match(self):
        payload = history(codes=2)
        for row in payload['columns']['volume']: row[0] = 0.
        snapshot = HistoricalSnapshot(payload); expression = REPRESENTATIVES[0][1]
        day = payload['sessions'][-1]; code = payload['instruments'][0]
        self.assertIsNone(_legacy_values(expression, snapshot, 149)[code])
        self.assertIsNone(evaluate_ast(translate_expression(expression)['canonical_ast'], snapshot)[day][code])

    def assets(self, root):
        (root/'data').mkdir()
        with (root/'data/passed_factor_pool.csv').open('w', encoding='utf-8', newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['name','formula','status']);writer.writeheader()
            writer.writerow({'name':'same','formula':REPRESENTATIVES[0][1],'status':'PASS'})
            writer.writerow({'name':'same','formula':REPRESENTATIVES[1][1],'status':'reserve'})
            writer.writerow({'name':'bad','formula':'close_p.fillna(0)','status':''})
            writer.writerow({'name':'missing','formula':'','status':'data_insufficient'})
        _write_new(root/'data/tested_factors.json', {'tested': {'same': {'ic': None}}, 'names':['same'], 'total_tested':1})
        native = {'schema_version':'fuxi.factor_card.1', 'candidate_id':'native',
            'canonical_ast':translate_expression(REPRESENTATIVES[0][1])['canonical_ast'],
            'qualification_receipt_id':'old_cached_receipt', 'jq_candidate_allowed':True,
            'stage_states':{'E3':{'eval_status':'data_insufficient'}}}
        native['card_hash']=digest(native)
        _write_new(root/'reports/original/cards/a.json', native)
        _write_new(root/'reports/saved_copy/cards/a.json', native)

    def test_every_status_row_and_duplicate_card_appearance_retained(self):
        with scratch_directory() as temp:
            root=Path(temp);self.assets(root)
            sources={str(p):file_hash(p) for p in root.rglob('*') if p.is_file()}
            result=build_catalog(root,root/'output')
            rows=load_json(root/'output/CATALOG.json')['rows']
            self.assertEqual(result['row_count'],9)
            self.assertEqual(result['native_card_rows'],2)
            self.assertEqual(result['source_rows_filtered_by_status'],0)
            self.assertEqual(len({r['entry_id'] for r in rows}),len(rows))
            self.assertTrue(all(not r['eligible_for_jq'] and r['qualification_receipt_id'] is None for r in rows))
            self.assertGreater(result['conflict_name_count'],0)
            self.assertEqual(result['status_counts']['data_insufficient'],1)
            self.assertEqual(result['status_counts']['PASS'],1)
            self.assertEqual(sources,{p:file_hash(p) for p in sources})
            self.assertEqual(build_catalog(root,root/'output'),result)

    def test_native_card_hash_tampering_refused(self):
        with scratch_directory() as temp:
            root=Path(temp);self.assets(root)
            path=root/'reports/original/cards/a.json';row=load_json(path);row['jq_candidate_allowed']=False
            path.write_text(__import__('json').dumps(row),encoding='utf-8')
            with self.assertRaises(ContractError):build_catalog(root,root/'output')

    def test_legacy_nan_is_explicit_not_zero_or_missing_status(self):
        with scratch_directory() as temp:
            root=Path(temp);(root/'data').mkdir()
            path=root/'data/dirty_factor_reeval.json'
            path.write_bytes(b'{"results":[{"name":"unknown","ic":NaN,"status":"data_insufficient"}]}')
            old=file_hash(path);summary=build_catalog(root,root/'output')
            rows=load_json(root/'output/CATALOG.json')['rows']
            self.assertEqual(summary['legacy_nonfinite_cells'],1)
            self.assertEqual(rows[0]['original_record']['ic'],{'__fuxi_legacy_nonfinite_number__':'NaN'})
            self.assertEqual(rows[0]['source_nonfinite_markers'],[{'pointer':'/ic','token':'NaN'}])
            self.assertEqual(rows[0]['historical_declared_status'],'data_insufficient')
            self.assertEqual(file_hash(path),old)

    def test_existing_output_never_overwritten_on_source_change(self):
        with scratch_directory() as temp:
            root=Path(temp);self.assets(root);build_catalog(root,root/'output')
            old=file_hash(root/'output/CATALOG.json')
            with (root/'data/passed_factor_pool.csv').open('a',encoding='utf-8') as stream:stream.write('extra,,reserve\n')
            with self.assertRaises(ContractError):build_catalog(root,root/'output')
            self.assertEqual(file_hash(root/'output/CATALOG.json'),old)

    def test_fixed_scope_proof_has_nonempty_all_three_and_no_qualification(self):
        with scratch_directory() as temp:
            root=Path(temp);(root/'data').mkdir()
            with (root/'data/unified_factor_pool.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['name','formula','status']);writer.writeheader()
                writer.writerows({'name':n,'formula':f,'status':'reserve'} for n,f in REPRESENTATIVES)
            source=root/'input.json';_write_new(source,{'snapshot':history(),'market':{'execution':{'fixture_only':True}}})
            before=file_hash(source);result=prove_representatives(root,root/'proof',source)
            self.assertEqual(result['representative_count'],3)
            self.assertEqual(result['total_pairs'],3*20*64)
            self.assertEqual(result['mismatches'],0)
            self.assertFalse(result['economic_evaluation_performed'])
            self.assertFalse(result['qualification_transferred'])
            self.assertEqual(file_hash(source),before)

    def test_empty_comparison_request_refused(self):
        with scratch_directory() as temp:
            with self.assertRaises(ContractError):prove_representatives(temp,Path(temp)/'proof',decision_count=0)


if __name__ == '__main__': unittest.main()
