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
        # GA 路径（combine_ga）把 oversized 子代的 reject 计入 counters.compiler_rejected
        # （生产链真实消费的记账），而非 diagnostics（legacy combine 的死代码明细）。
        counters={}
        found=generate(FIELDS,arm='combine',count=1,seed=17,parents=[parent,parent],counters=counters)
        self.assertEqual(len(found),1);compile_ast(found[0]['canonical_ast'],FIELDS)
        self.assertEqual(found[0]['action'],'typed_random_template')
        self.assertGreaterEqual(counters.get('compiler_rejected', 0), 1)
        with self.assertRaisesRegex(ContractError,'AST budget exceeded'):
            compile_ast(node('rank',ast),FIELDS)

    def test_unexpected_program_errors_are_not_rejected_as_formulas(self):
        with patch('fuxi_v1.search.factor_spec',side_effect=RuntimeError('unexpected')):
            with self.assertRaisesRegex(RuntimeError,'unexpected'):
                generate(FIELDS,arm='random',count=1,seed=1,diagnostics=[])

if __name__=='__main__':unittest.main()
