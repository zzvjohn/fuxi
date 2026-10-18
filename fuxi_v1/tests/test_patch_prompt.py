"""Regress the real provider's leaf/window confusion; no network calls."""
import unittest
from copy import deepcopy
from unittest.mock import patch
from fuxi_v1.edits import patch_prompt, llm_patches, parse_llm_patches, apply_patch
from fuxi_v1.search import factor_spec, node
from fuxi_v1.contracts import ContractError


class PatchPromptTests(unittest.TestCase):
    def setUp(self):
        self.fields = {'close':{'unit':'price'}, 'open':{'unit':'price'}}
        self.parent = factor_spec(node('rank', node('return', {'field':'close'}, window=20)),
                                  self.fields, 'frozen_parent')

    def test_window_targets_address_operator_and_example_compiles(self):
        prompt = patch_prompt(self.fields, [self.parent], [], 1)
        parent = prompt['parents'][0]
        self.assertEqual([r['path'] for r in parent['nodes']], [[], [0], [0, 0]])
        self.assertEqual([r['path'] for r in parent['window_edit_targets']], [[0]])
        self.assertNotIn(20, parent['window_edit_targets'][0]['allowed_values'])
        result, rejected, count = parse_llm_patches(prompt['schema'], [self.parent], self.fields, [], 1)
        self.assertEqual((len(result), rejected, count), (1, [], 1))
        self.assertEqual(result[0]['canonical_ast']['args'][0]['window'], 15)

    def test_provider_leaf_window_patch_stays_rejected_without_repair(self):
        response = {'patches':[{'parent_id':self.parent['candidate_id'], 'kind':'window',
                                'path':[0, 0], 'value':15}]}
        before = deepcopy(response)
        result, rejected, count = parse_llm_patches(response, [self.parent], self.fields, [], 1)
        self.assertEqual((result, rejected, count), ([], ['window edit invalid'], 1))
        self.assertEqual(response, before)

    def test_paths_are_derived_for_root_nested_and_no_window_parents(self):
        asts = [node('return', {'field':'close'}, window=20),
                node('neg', self.parent['canonical_ast']), node('rank', {'field':'close'})]
        for ast, expected in zip(asts, [[[]], [[0, 0]], []]):
            parent = factor_spec(ast, self.fields, 'frozen_parent')
            prompt = patch_prompt(self.fields, [parent], [], 1)
            self.assertEqual([x['path'] for x in prompt['parents'][0]['window_edit_targets']], expected)
            apply_patch(parent, prompt['schema']['patches'][0], self.fields)
        for count in (0, 11, True):
            with self.assertRaises(ContractError):patch_prompt(self.fields, [self.parent], [], count)

    def test_consumer_keeps_one_request_budget_and_only_evidence_differs(self):
        card = {'evidence_id':'frozen_training_only', 'memory_window':['2024-06-05','2025-07-30']}
        ordinary = patch_prompt(self.fields, [self.parent], [], 1)
        memory = patch_prompt(self.fields, [self.parent], [card], 1)
        self.assertEqual({k:v for k,v in ordinary.items() if k != 'evidence'},
                         {k:v for k,v in memory.items() if k != 'evidence'})
        with patch('fuxi_v1.search.llm_candidates', return_value=([], {'status':'offline'})) as provider:
            llm_patches(self.fields, [self.parent], [card], count=1, max_tokens=1600, timeout=45)
            provider.assert_called_once()
            self.assertEqual(provider.call_args.kwargs['_prompt'], memory)
            self.assertEqual(provider.call_args.kwargs['max_tokens'], 1600)
            self.assertEqual(provider.call_args.kwargs['timeout'], 45)
