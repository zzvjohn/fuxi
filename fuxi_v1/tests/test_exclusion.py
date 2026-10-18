"""A survivor-selected diagnostic cannot become a formal admission receipt."""
from pathlib import Path
import unittest
from fuxi_v1.contracts import evaluation
from fuxi_v1.evidence import EvidenceStore
from fuxi_v1.acceptance import AcceptanceService
from fuxi_v1.tests.test_kernel import scratch_directory

class ExclusionTests(unittest.TestCase):
    def test_exclusion_blocks_even_when_stages_and_data_receipt_pass(self):
        with scratch_directory() as root:
            store=EvidenceStore(Path(root)/'evidence.sqlite')
            try:
                receipt=store.receipt('data_contract',{'snapshot_hash':'fixture_snapshot','pit_certified':True,
                    'universe_certified':True,'account_events_certified':True,'execution_rules_certified':True,
                    'schema_version':'fuxi.independent_data_assurance.1','investment_eligibility_certified':True})
                result={'candidate_id':'fixture','policy_hash':'policy','snapshot_hash':'fixture_snapshot',
                    'strategy_ir':{'ir_hash':'ir'},'source_scope':'research_exposed',
                    'evaluations':[evaluation('E'+str(i),'valid','fixture_pass',decision='pass',n_pairs=20) for i in range(5)]}
                normal=AcceptanceService(store).research(store.append('evaluation',result),receipt)
                self.assertTrue(normal['jq_candidate_allowed'])
                result['selection_audit']={'posthoc_exclusion':True,'diagnostic_only':True}
                excluded=AcceptanceService(store).research(store.append('evaluation',result),receipt)
                self.assertFalse(excluded['jq_candidate_allowed']);self.assertIsNone(excluded['receipt_id'])
                self.assertEqual(excluded['blockers'],['posthoc_universe_exclusion_requires_unbiased_revalidation'])
            finally:store.close()

if __name__=='__main__':unittest.main()
