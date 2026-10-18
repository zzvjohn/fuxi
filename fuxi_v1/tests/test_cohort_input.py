"""A sampled source-bound input must keep the original identifier-only draw."""
from copy import deepcopy
import unittest
from pathlib import Path
from fuxi_v1.contracts import digest, ContractError
from fuxi_v1.datasets import file_hash
from fuxi_v1.campaign import write_new
from fuxi_v1.pit_universe import read_lifecycle_file, reproduce_cohort
from fuxi_v1.cohort_input import verify_declared_cohort, channel_audit, halted_distribution_reference, resumed_distribution_reference
from fuxi_v1.tests.test_kernel import scratch_directory


SESSIONS = ['2024-01-02', '2024-01-03', '2024-01-04']


def sampled_fixture(root):
    root = Path(root)
    lifecycle = root / 'delist_dates.csv'
    rows = [{'ts_code': '%06d.SZ' % i, 'list_date': '20000101',
             'delist_date': '20240103' if i == 2 else None, 'list_status': 'D' if i == 2 else 'L'}
            for i in range(1, 17)]
    lifecycle.write_bytes(('ts_code,list_date,delist_date,list_status\n' + ''.join(
        '{ts_code},{list_date},{delist_date},{list_status}\n'.format(**dict(row, delist_date=row['delist_date'] or ''))
        for row in rows)).encode())
    declaration = {'rule_kind': 'full_population_hash_sample', 'salt': 'original-fixed-salt',
                   'target_size': 12, 'sampling_keys': ['ts_code'], 'sampling_frame': 'full mainboard lifecycle',
                   'frame_predicate': 'mainboard and listed<=window_end and not delisted<=window_start'}
    life = read_lifecycle_file(lifecycle)
    codes, _ = reproduce_cohort(life, declaration, SESSIONS)
    protocol = {'declaration': declaration, 'window': [SESSIONS[0], SESSIONS[-1]], 'draw_attempt_budget': 1,
                'lifecycle_path': str(lifecycle.resolve()), 'lifecycle_sha256': file_hash(lifecycle)}
    draw = {'codes': codes, 'declaration_hash': digest(declaration)}
    protocol_path, draw_path = root / 'PROTOCOL.json', root / 'DRAW.json'
    write_new(protocol_path, protocol)
    write_new(draw_path, draw)
    reference = lambda p: {'path': str(p.resolve()), 'sha256': file_hash(p)}
    plan = {'account_evidence_mode': 'source_bound_research', 'window': {'start': SESSIONS[0], 'end': SESSIONS[-1]},
            'cohort': dict(declaration, members=[dict(ts_code=c, **life[c]) for c in codes], posthoc_exclusion=False),
            'cohort_input_binding': {'protocol': reference(protocol_path), 'draw': reference(draw_path),
                                     'lifecycle': reference(lifecycle)}}
    return plan


class SampledInputTests(unittest.TestCase):
    def halted_term(self):
        term = {'status': 'ordinary_distribution_supported', 'instrument': '001331.XSHE',
                'record_date': '2026-06-25', 'ex_date': '2026-06-26', 'ann_date': '2026-06-22',
                'pay_date': '2026-06-26', 'share_credit_date': '2026-06-26', 'gross_cash_per_share': .06,
                'share_distribution': .45, 'evidence_id': 'unit_issuer_term', 'tax_fraction': '.20'}
        term['issuer_evidence'] = {k: term[k] for k in
                                  ('instrument', 'record_date', 'ex_date', 'ann_date', 'pay_date', 'share_credit_date')}
        term['issuer_evidence'].update(holder_cash=.06, holder_shares=.45, sources=[{'sha256': 'unit_only'}],
                                      independent_assurance=False)
        return term

    def test_halted_distribution_keeps_actual_2700_to_3915_rights_and_cash(self):
        from decimal import Decimal
        from fuxi_v1.account import Account
        from fuxi_v1.action_accounting import post_entitlements
        term = self.halted_term()
        policy = {'mode': 'declared_supported_distribution_reference', 'rounding': 'half_up'}
        reference = halted_distribution_reference(term, 69.78, policy,
                                                   {'session': '2026-05-27', 'close': 69.78, 'evidence_id': 'unit_quote'})
        self.assertEqual(reference['reference_mark'], 48.08)
        self.assertEqual(reference['rounding_sensitivity_prices'], ['48.08', '48.09'])
        account = Account('200000')
        account.apply({'event_id': 'unit_buy', 'kind': 'fill', 'timestamp': '2026-06-25T09:30:00+08:00',
                       'evidence_id': 'unit_fill', 'payload': {'instrument': term['instrument'], 'side': 'buy',
                           'quantity': 2700, 'price': '69.78', 'fee': '0', 'sellable_at': '2026-06-26T09:00:00+08:00'}})
        entitlements, gaps = {}, []
        post_entitlements(account, [term], term['ex_date'], {term['evidence_id']: 2700}, entitlements, 'unit_ir', {}, gaps)
        account.apply({'event_id': 'unit_cash', 'kind': 'payment', 'timestamp': '2026-06-26T09:00:00+08:00',
                       'evidence_id': term['evidence_id'], 'payload': {'entitlement_id': entitlements[term['evidence_id']]}})
        quantity = sum(lot['quantity'] for lot in account.positions[term['instrument']])
        self.assertEqual(quantity, 3915)
        self.assertEqual(account.cash, Decimal('11723.6'))
        self.assertFalse(gaps)
        self.assertFalse(account.unresolved)
        nav = account.cash + Decimal(str(reference['reference_mark'])) * quantity
        self.assertEqual(nav, Decimal('199956.8'))
        self.assertFalse(reference['observed_quote'])
        self.assertFalse(reference['trade_permission_granted'])

    def test_halted_model_does_not_fill_unknown_rights_or_future_price(self):
        term = self.halted_term()
        policy = {'mode': 'declared_supported_distribution_reference', 'rounding': 'half_up'}
        quote = {'session': '2026-05-27', 'close': 69.78, 'evidence_id': 'unit_quote'}
        unknown = deepcopy(term)
        unknown.pop('issuer_evidence')
        self.assertIsNone(halted_distribution_reference(unknown, 69.78, policy, quote))
        with self.assertRaisesRegex(ContractError, 'prior observed quote absent'):
            halted_distribution_reference(term, 69.78, policy, dict(quote, session='2026-06-29'))
        reference = halted_distribution_reference(term, 69.78, policy, quote)
        self.assertTrue(resumed_distribution_reference(reference, 2., 2., 48.08))
        self.assertTrue(resumed_distribution_reference(reference, 2., 2., 48.09))
        self.assertFalse(resumed_distribution_reference(reference, 2., 2., 48.10))
        self.assertFalse(resumed_distribution_reference(reference, 2., 3., 48.08))

    def test_fractional_holder_rights_remain_unknown_after_reference(self):
        from fuxi_v1.account import Account
        from fuxi_v1.action_accounting import post_entitlements
        term, account = self.halted_term(), Account('1000')
        gaps = []
        post_entitlements(account, [term], term['ex_date'], {term['evidence_id']: 2701}, {}, 'unit_ir', {}, gaps)
        self.assertEqual(gaps[0]['reason'], 'fractional_allotment_not_observed')
        self.assertTrue(account.unresolved)
        self.assertFalse(account.share_receivables)

    def test_supported_halted_cash_tick_and_deferred_factor_are_conditional(self):
        from decimal import Decimal
        from fuxi_v1.account import Account
        from fuxi_v1.action_accounting import post_entitlements
        term = {'status': 'cash_terms_supported', 'instrument': '600717.XSHG',
                'ann_date': '2026-06-12', 'record_date': '2026-06-17', 'ex_date': '2026-06-18',
                'pay_date': '2026-06-18', 'gross_cash_per_share': .102, 'share_distribution': 0.,
                'evidence_id': 'unit_frozen_cash_terms', 'tax_fraction': '.20'}
        quote = {'session': '2026-06-08', 'close': 4.31, 'evidence_id': 'unit_observed_quote'}
        old = {'mode': 'declared_supported_distribution_reference', 'rounding': 'half_up'}
        self.assertIsNone(halted_distribution_reference(term, 4.31, old, quote))
        policy = dict(old, cash_tick_reference='source_bound_cash_terms_tick_reference',
                      deferred_factor_policy={'mode': 'resume_factor_from_declared_reference',
                                              'quantum': '0.0001', 'rounding': 'half_up'})
        reference = halted_distribution_reference(term, 4.31, policy, quote,
                                                    prior_factor=34.8547, current_factor=34.8547)
        self.assertEqual(reference['theoretical_price'], '4.208')
        self.assertEqual(reference['reference_mark'], 4.21)
        self.assertEqual(reference['rounding_sensitivity_prices'], ['4.20', '4.21'])
        self.assertTrue(resumed_distribution_reference(reference, 34.8547, 35.6826, 4.21))
        self.assertFalse(resumed_distribution_reference(reference, 34.8547, 35.6825, 4.21))
        self.assertFalse(resumed_distribution_reference(reference, 34.8547, 35.6826, 4.22))
        already_applied = halted_distribution_reference(term, 4.31, policy, quote,
                                                         prior_factor=34.8547, current_factor=35.6826)
        self.assertNotIn('deferred_factor_anchor', already_applied)
        self.assertFalse(resumed_distribution_reference(already_applied, 34.8547, 35.6826, 4.21))
        account = Account('10000')
        account.apply({'event_id': 'cash_unit_buy', 'kind': 'fill', 'timestamp': '2026-06-17T09:30:00+08:00',
            'evidence_id': 'unit_fill', 'payload': {'instrument': term['instrument'], 'side': 'buy',
                'quantity': 1000, 'price': '4.31', 'fee': '0', 'sellable_at': '2026-06-18T09:00:00+08:00'}})
        entitlements, gaps = {}, []
        post_entitlements(account, [term], term['ex_date'], {term['evidence_id']: 1000}, entitlements, 'unit_ir', {}, gaps)
        account.apply({'event_id': 'cash_unit_pay', 'kind': 'payment', 'timestamp': '2026-06-18T18:00:00+08:00',
            'evidence_id': term['evidence_id'], 'payload': {'entitlement_id': entitlements[term['evidence_id']]}})
        self.assertEqual(account.cash, Decimal('5771.600'))
        self.assertEqual(sum(lot['quantity'] for lot in account.positions[term['instrument']]), 1000)
        self.assertEqual(account.cash + Decimal(str(reference['reference_mark'])) * 1000, Decimal('9981.600'))
        self.assertFalse(gaps or account.unresolved)
        self.assertFalse(reference['observed_quote'] or reference['trade_permission_granted'])

    def test_actual_original_rule_is_rebuilt_and_delistings_retained(self):
        with scratch_directory() as directory:
            plan = sampled_fixture(directory)
            proof = verify_declared_cohort(plan, directory, SESSIONS)
            self.assertEqual(len(proof['codes']), 12)
            self.assertTrue(proof['basis']['permutation_invariant'])
            self.assertFalse(proof['formal_data_assured'])

    def test_member_deletion_replacement_and_reorder_are_rejected(self):
        with scratch_directory() as directory:
            plan = sampled_fixture(directory)
            variants = [plan['cohort']['members'][:-1], list(reversed(plan['cohort']['members'])),
                        plan['cohort']['members'][:-1] + [plan['cohort']['members'][0]]]
            for members in variants:
                changed = deepcopy(plan)
                changed['cohort']['members'] = members
                with self.assertRaisesRegex(ContractError, 'members removed'):
                    verify_declared_cohort(changed, directory, SESSIONS)

    def test_salt_and_window_cannot_be_reselected(self):
        with scratch_directory() as directory:
            plan = sampled_fixture(directory)
            changed = deepcopy(plan)
            changed['cohort']['salt'] = 'try-a-better-sample'
            with self.assertRaisesRegex(ContractError, 'declaration changed'):
                verify_declared_cohort(changed, directory, SESSIONS)
            with self.assertRaisesRegex(ContractError, 'window changed'):
                verify_declared_cohort(plan, directory, SESSIONS[:-1])

    def test_changed_original_source_fails(self):
        with scratch_directory() as directory:
            plan = sampled_fixture(directory)
            path = Path(directory) / 'DRAW.json'
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaisesRegex(ContractError, 'source changed: draw'):
                verify_declared_cohort(plan, directory, SESSIONS)

    def test_member_lifecycle_cannot_erase_delisting(self):
        with scratch_directory() as directory:
            plan = sampled_fixture(directory)
            plan['cohort']['members'][0]['list_date'] = '20100101'
            with self.assertRaisesRegex(ContractError, 'member lifecycle changed'):
                verify_declared_cohort(plan, directory, SESSIONS)

    def test_posthoc_exclusion_and_missing_binding_fail_closed(self):
        with scratch_directory() as directory:
            plan = sampled_fixture(directory)
            changed = deepcopy(plan)
            changed['cohort']['posthoc_exclusion'] = True
            with self.assertRaisesRegex(ContractError, 'posthoc exclusion'):
                verify_declared_cohort(changed, directory, SESSIONS)
            plan.pop('cohort_input_binding')
            with self.assertRaisesRegex(ContractError, 'original binding missing'):
                verify_declared_cohort(plan, directory, SESSIONS)

    def test_missing_data_and_no_events_remain_in_denominator(self):
        payload = {'snapshot': {'sessions': SESSIONS[:2], 'instruments': ['A', 'B'],
                               'columns': {'close': [[10., None], [11., None]]}},
                   'labels': {'rows': [{'status': 'valid'}, {'status': 'data_insufficient'}]},
                   'market': {'action_terms': [{'instrument': 'A', 'status': 'cash_terms_supported'}],
                              'bars': [{'event_coverage': 'known'}, {'event_coverage': 'missing_evidence'}]}}
        audit = channel_audit(payload)
        self.assertEqual(audit['field_coverage']['close'], {'expected_cells': 4, 'observed_cells': 2, 'missing_cells': 2})
        self.assertEqual(audit['members_without_implemented_terms_in_window'], ['B'])
        self.assertEqual(audit['label_status']['data_insufficient'], 1)
        self.assertFalse(audit['formal_data_assured'])


if __name__ == '__main__':
    unittest.main()
