"""A bounded generated proposal rejection must not crash the entire search."""
import unittest
from unittest.mock import patch
from fuxi_v1.search import generate,factor_spec,node
from fuxi_v1.dsl import CATALOG,compile_ast
from fuxi_v1.contracts import ContractError
FIELDS={'close':{'unit':'price'}}

class GeneratorBoundsTests(unittest.TestCase):
    def test_valid_deep_parents_reject_oversized_child_and_fall_back(self):
        ast={'field':'close'}
        for _ in range(CATALOG['max_depth']-1):ast=node('rank',ast)
        parent=factor_spec(ast,FIELDS,'deep_parent')
        diagnostics=[]
        found=generate(FIELDS,arm='combine',count=1,seed=17,parents=[parent,parent],diagnostics=diagnostics)
        self.assertEqual(len(found),1);compile_ast(found[0]['canonical_ast'],FIELDS)
        self.assertEqual(found[0]['action'],'typed_random_template')
        self.assertEqual(len(diagnostics),1);self.assertEqual(diagnostics[0]['reason'],'AST budget exceeded')
        with self.assertRaisesRegex(ContractError,'AST budget exceeded'):
            compile_ast(node('rank',ast),FIELDS)

    def test_unexpected_program_errors_are_not_rejected_as_formulas(self):
        with patch('fuxi_v1.search.factor_spec',side_effect=RuntimeError('unexpected')):
            with self.assertRaisesRegex(RuntimeError,'unexpected'):
                generate(FIELDS,arm='random',count=1,seed=1,diagnostics=[])

if __name__=='__main__':unittest.main()
