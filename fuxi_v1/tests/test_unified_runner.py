import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fuxi_v1.contracts import ContractError
from fuxi_v1.unified_runner import _limits, status, atomic, _verify_result, DEFAULT_LIMITS, file_sha
from fuxi_v1.legacy_writers import inventory

class UnifiedControllerContract(unittest.TestCase):
    def test_reservation_is_bounded_and_zero_external(self):
        owner={'trigger':'automatic','budgets':DEFAULT_LIMITS}
        request={'rounds':3,'candidates_per_round':2}
        self.assertEqual(_limits(owner,request)['max_candidates'],6)
        for altered in ({'rounds':4,'candidates_per_round':2},
                        {**request,'limits':{**DEFAULT_LIMITS,'max_external_requests':1}},
                        {**request,'limits':{**DEFAULT_LIMITS,'max_candidates':8}}):
            with self.assertRaises(ContractError):_limits(owner,altered)

    def test_readonly_status_does_not_create_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(status(tmp)['state'],'not_started')
            self.assertFalse((Path(tmp)/'data').exists())

    def test_completion_verifies_all_content_not_title(self):
        from fuxi_v1.contracts import digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic(root/'result.json',{'value':1})
            receipt={'state':'completed','artifacts':{'result.json':file_sha(root/'result.json')}}
            receipt['result_hash']=digest(receipt);atomic(root/'COMPLETE.json',receipt)
            self.assertEqual(_verify_result(root)['state'],'completed')
            atomic(root/'result.json',{'value':2})
            with self.assertRaises(ContractError):_verify_result(root)

    def test_declared_real_production_guards_cover_boundaries(self):
        root=Path(__file__).resolve().parents[2]
        entries=inventory(root)['entries']
        self.assertGreaterEqual(len(entries),20)
        self.assertTrue(all(e['guard_present'] for e in entries))
        for entry in entries:
            ast.parse((root/entry['path']).read_bytes(),filename=entry['path'])

    def test_output_escape_in_completion_is_rejected(self):
        from fuxi_v1.contracts import digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);receipt={'state':'completed','artifacts':{'../escape':'0'*64}}
            receipt['result_hash']=digest(receipt);atomic(root/'COMPLETE.json',receipt)
            with self.assertRaises(ContractError):_verify_result(root)

if __name__=='__main__':unittest.main()
