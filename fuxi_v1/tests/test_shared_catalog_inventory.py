"""Inventory expansion contract: publication bindings and static FA source assets.

These fixtures define the additional read-only inventory; no module definition
is executed and no bundled scores/evaluation can transfer qualification.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fuxi_v1.artifacts import publish
from fuxi_v1.contracts import canonical, ContractError, digest, load_json
from fuxi_v1.datasets import file_hash
from fuxi_v1.search import factor_spec
from fuxi_v1.shared_catalog import build_catalog, NATIVE_FIELDS


TEMP=Path('C:/Users/Zhong/.codex/visualizations/2026/10/09/01a12085-dfb6-76e3-bda3-164d4e9a37b9/catalog_inventory')


@contextmanager
def workspace():
    TEMP.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='inventory_',dir=TEMP) as tmp:yield Path(tmp)


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(value if isinstance(value,bytes) else canonical(value)+b'\n')


def bundle(root,base='reports/campaign',identifier='run_inventory1'):
    candidate=factor_spec({'op':'rank','args':[{'field':'close'}]},NATIVE_FIELDS,'inventory_fixture')
    directory=publish(root/base/'bundles',{
        'candidate.json':candidate,'scores.json':b'never load bundled score values',
        'evaluation.json':{'jq_candidate_allowed':True,'qualification_receipt_id':'stale_reference'},
        'scores.npy':b'not an array; inventory must not parse it'},
        {'fixture_only':True,'qualification_receipt_id':'cached_reference'},run_id=identifier)
    return directory,candidate


def catalog(root):
    summary=build_catalog(root,root/'catalog')
    return summary,load_json(root/'catalog/CATALOG.json')['rows']


class SharedCatalogInventoryTests(unittest.TestCase):
    def test_published_candidates_in_all_three_roots_are_one_bound_entity_each(self):
        with workspace() as root:
            originals=[]
            for i,base in enumerate(('reports/campaign','data/fuxi_v1/campaign','data/fuxi_unified/runs/unified_fixture/research')):
                originals.append(bundle(root,base,'run_inventory'+str(i)))
            summary,rows=catalog(root)
            self.assertEqual(summary['source_count'],6)
            self.assertEqual(summary['published_native_candidate_rows'],3)
            self.assertEqual(summary['publication_manifest_metadata_rows'],3)
            self.assertEqual(summary['row_count'],6)
            for directory,candidate in originals:
                matches=[r for r in rows if r['source_path']==str(directory/'candidate.json')]
                self.assertEqual(len(matches),1);row=matches[0]
                self.assertEqual(row['section'],'native_candidate');self.assertEqual(row['original_record'],candidate)
                manifest=load_json(directory/'manifest.json');binding=row['publication_binding']
                self.assertEqual(binding['candidate_sha256'],file_hash(directory/'candidate.json'))
                self.assertEqual(binding['manifest_sha256'],file_hash(directory/'manifest.json'))
                self.assertEqual(binding['manifest_hash'],manifest['manifest_hash'])
                self.assertEqual(binding['run_id'],directory.name);self.assertTrue(binding['complete'])
                self.assertFalse(binding['other_declared_files_verified'])
            self.assertTrue(all(not r['eligible_for_jq'] and r['qualification_receipt_id'] is None for r in rows))

    def test_manifest_hash_candidate_sha_incomplete_and_run_identity_tampering_refuse(self):
        for failure in ('manifest_hash','candidate_sha','incomplete','run_identity','unsafe_declared_path'):
            with self.subTest(failure=failure),workspace() as root:
                directory,_=bundle(root);path=directory/'manifest.json';manifest=load_json(path)
                put(root/'data/factor_pool.csv',b'name,formula,status\nfixture,,reserve\n')
                if failure=='manifest_hash':manifest['provenance']['tampered']=True
                else:
                    if failure=='candidate_sha':manifest['files']['candidate.json']='0'*64
                    if failure=='incomplete':manifest['complete']=False
                    if failure=='run_identity':manifest['run_id']='run_different'
                    if failure=='unsafe_declared_path':manifest['files']['../unsafe.npy']='0'*64
                    manifest['manifest_hash']=digest({k:v for k,v in manifest.items() if k!='manifest_hash'})
                put(path,manifest)
                with self.assertRaises(ContractError):catalog(root)
                self.assertFalse((root/'catalog/CATALOG.json').exists())

    def test_bundle_inventory_never_loads_scores_arrays_or_evaluation(self):
        with workspace() as root:
            directory,_=bundle(root);real_open=Path.open
            def forbid_large_lookup(path,*args,**kwargs):
                if path.name in ('scores.json','scores.npy','evaluation.json'):
                    raise AssertionError('inventory loaded values or evaluation: '+str(path))
                return real_open(path,*args,**kwargs)
            with patch.object(Path,'open',forbid_large_lookup):summary,rows=catalog(root)
            self.assertEqual(summary['published_native_candidate_rows'],1)
            source=next(s for s in summary['input_sources'] if s['path']==str(directory/'candidate.json'))
            self.assertFalse(source['publication_binding']['other_declared_files_verified'])
            self.assertEqual(summary['qualification_granted_count'],0)

    def test_fa_source_symbols_and_config_init_metadata_archive_without_execution(self):
        with workspace() as root:
            marker=root/'must_not_execute.txt'
            body=(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
                  "class PriceFactor:\n    def compute(self): return 'close_p'\n"
                  "def helper():\n    return 'close_p'\n"
                  "async def async_helper():\n    return None\n").encode()
            paths=[root/'research/factor_alchemy/factors/custom.py',root/'research/factor_alchemy/factors/__init__.py',
                   root/'research/factor_alchemy/config.py',root/'research/factor_alchemy/__init__.py']
            for path in paths:put(path,body if path.name=='custom.py' else f"raise RuntimeError({path.name!r})\n".encode())
            summary,rows=catalog(root)
            self.assertFalse(marker.exists());self.assertEqual(summary['source_module_assets'],4)
            self.assertEqual(summary['source_code_definition_rows'],3)
            symbols=[r for r in rows if r['section']=='source_code_definition']
            self.assertEqual([(r['original_record']['name'],r['original_record']['lineno']) for r in symbols],
                             [('PriceFactor',3),('helper',5),('async_helper',7)])
            self.assertEqual([r['original_record']['symbol_kind'] for r in symbols],['class','function','async_function'])
            for path in paths:
                self.assertEqual((root/'catalog/original_sources'/(file_hash(path)+'.py')).read_bytes(),path.read_bytes())
                module=next(r for r in rows if r['source_path']==str(path) and r['section']=='source_module_asset')
                self.assertFalse(module['original_record']['source_executed'])
            self.assertTrue(all(r['mapping'] is None and not r['evaluation_ready'] and not r['eligible_for_jq'] for r in rows))

    def test_repeated_module_and_native_metadata_appearances_keep_distinct_source_identity(self):
        with workspace() as root:
            first,candidate=bundle(root,'reports/original','run_original')
            second,_=bundle(root,'reports/copy','run_copy')
            body=b'class SameName:\n    pass\n'
            for name in ('one.py','two.py'):put(root/'research/factor_alchemy/factors'/name,body)
            summary,rows=catalog(root)
            self.assertEqual(summary['published_native_candidate_rows'],2)
            self.assertEqual(summary['publication_manifest_metadata_rows'],2)
            self.assertEqual(summary['source_module_assets'],2)
            definitions=[r for r in rows if r['section']=='source_code_definition']
            self.assertEqual(len(definitions),2);self.assertEqual(definitions[0]['name'],definitions[1]['name'])
            self.assertEqual(definitions[0]['source_sha256'],definitions[1]['source_sha256'])
            self.assertNotEqual(definitions[0]['entry_id'],definitions[1]['entry_id'])
            native=[r for r in rows if r['section']=='native_candidate']
            self.assertEqual(native[0]['original_record'],native[1]['original_record'])
            self.assertNotEqual(native[0]['entry_id'],native[1]['entry_id'])
            self.assertEqual(summary['source_rows_filtered_by_status'],0)

    def test_native_ast_outside_legacy_projection_stays_whole_unsupported_and_unqualified(self):
        with workspace() as root:
            directory,candidate=bundle(root);summary,rows=catalog(root)
            row=next(r for r in rows if r['section']=='native_candidate')
            self.assertEqual(row['original_record'],candidate)
            self.assertEqual(row['mapping_status'],'unsupported_or_missing')
            self.assertIsNone(row['semantic_factor_id']);self.assertFalse(row['qualification_transferred'])
            self.assertFalse(row['evaluation_ready']);self.assertIsNone(row['qualification_receipt_id'])
            self.assertEqual(summary['qualification_granted_count'],0)

    def test_unpublished_pending_candidate_and_unrelated_python_are_not_declared_assets(self):
        with workspace() as root:
            put(root/'data/factor_pool.csv',b'name,formula,status\nfixture,,reserve\n')
            put(root/'reports/campaign/bundles/.pending_run_fixture/candidate.json',{'schema_version':'fuxi.factor.2'})
            put(root/'reports/campaign/bundles/run_unpublished/candidate.json',{'schema_version':'fuxi.factor.2'})
            put(root/'research/factor_alchemy/unrelated.py',b"raise RuntimeError('do not execute')\n")
            summary,rows=catalog(root)
            self.assertEqual(summary['source_count'],1);self.assertEqual(summary['published_native_candidate_rows'],0)
            self.assertEqual(summary['source_module_assets'],0);self.assertEqual(len(rows),1)

    def test_python_syntax_unsupported_is_preserved_as_non_evaluable_module_metadata(self):
        with workspace() as root:
            path=root/'research/factor_alchemy/factors/incomplete.py';put(path,b'def incomplete(:\n')
            summary,rows=catalog(root)
            self.assertEqual(summary['source_module_assets'],1);self.assertEqual(summary['source_code_definition_rows'],0)
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['section'],'source_module_asset')
            self.assertEqual(rows[0]['original_record']['static_parse_status'],'unsupported_syntax')
            self.assertFalse(rows[0]['evaluation_ready']);self.assertIsNone(rows[0]['mapping'])
            self.assertEqual((root/'catalog/original_sources'/(file_hash(path)+'.py')).read_bytes(),path.read_bytes())


if __name__=='__main__':unittest.main()
