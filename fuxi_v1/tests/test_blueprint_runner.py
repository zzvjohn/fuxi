"""Operator recovery, cohort identity and immutable output boundaries."""
import unittest
from pathlib import Path
from unittest.mock import patch
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.tests.test_remaining_study import compact
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.campaign import write_new
from fuxi_v1.contracts import load_json,ContractError
from fuxi_v1.blueprint_runner import freeze,_checked,_stage,status


class BlueprintRunnerTests(unittest.TestCase):
    def assets(self,root):
        payload=compact(synthetic_campaign(140,8));days=payload['snapshot']['sessions']
        values={'input_path':payload,'draw_path':{'codes':[s[:6]+'.SH' for s in payload['snapshot']['instruments']]},
          'industry_path':{'captured_at':'2001-01-01T00:00:00+08:00','queries':[{'code':'000001.SH','rows':[]}]},
          'benchmark_path':{'rows':[]},'calendar_path':{'sessions':[]},'lifecycle_path':{'test_only':True}}
        spec={'schema_version':'fuxi.blueprint_run_request.1','research_start':days[85],'research_end':days[-1]}
        for key,value in values.items():
            p=root/(key+'.json');write_new(p,value);spec[key]=str(p)
        return spec

    def test_exact_cohort_and_native_pool_size(self):
        with scratch_directory() as temp:
            root=Path(temp);spec=self.assets(root);out=root/'run';result=freeze(spec,out)
            self.assertEqual(result['cohort_size'],8)
            native=load_json(out/'research/REMEDIATION_PROTOCOL.json')
            self.assertEqual(native['request']['pool_top_n'],8)
            self.assertFalse(native['formal_qualification_allowed'])
            self.assertTrue(freeze(spec,out)['idempotent'])
            changed=dict(spec,research_start=spec['research_end'])
            with self.assertRaises(ContractError):freeze(changed,out)

    def test_freeze_interruption_recovers_original_scope(self):
        with scratch_directory() as temp:
            root=Path(temp);spec=self.assets(root);out=root/'run'
            with patch('fuxi_v1.remaining_study.freeze',side_effect=RuntimeError('interruption')):
                with self.assertRaises(RuntimeError):freeze(spec,out)
            old=load_json(out/'RUN_PROTOCOL.json')
            self.assertTrue(freeze(spec,out)['idempotent'])
            self.assertEqual(load_json(out/'RUN_PROTOCOL.json'),old)
            self.assertTrue((out/'research/REMEDIATION_PROTOCOL.json').is_file())

    def test_cohort_deletion_and_input_drift_refused(self):
        with scratch_directory() as temp:
            root=Path(temp);spec=self.assets(root);out=root/'run'
            p=Path(spec['draw_path']);original=p.read_bytes();draw=load_json(p);draw['codes'].pop()
            p.write_bytes(__import__('json').dumps(draw).encode())
            with self.assertRaises(ContractError):freeze(spec,out)
            p.write_bytes(original);freeze(spec,out)
            source=Path(spec['input_path']);source.write_bytes(source.read_bytes()+b' ')
            with self.assertRaises(ContractError):status(out)

    def test_stage_resume_preserves_failed_partial_and_checks_real_output(self):
        with scratch_directory() as temp:
            root=Path(temp);spec=self.assets(root);out=root/'run';freeze(spec,out);protocol=_checked(out)
            partial=out/'partial.json';write_new(partial,{'observed':1})
            def finish():
                write_new(out/'result.json',{'n_pairs':3});return {'n_pairs':3}
            first=_stage(out,protocol,'TEST',finish)
            self.assertEqual(_stage(out,protocol,'TEST',lambda:self.fail('repeated consumer')),first)
            self.assertEqual(status(out)['completed_stages'],['TEST'])
            partial.write_bytes(b'changed')
            with self.assertRaises(ContractError):status(out)

    def test_changed_code_cannot_resume(self):
        with scratch_directory() as temp:
            root=Path(temp);spec=self.assets(root);out=root/'run';freeze(spec,out)
            with patch('fuxi_v1.blueprint_runner.code_manifest',return_value={}):
                with self.assertRaises(ContractError):status(out)

    def test_status_rejects_tampered_completion(self):
        with scratch_directory() as temp:
            root=Path(temp);spec=self.assets(root);out=root/'run';freeze(spec,out)
            write_new(out/'COMPLETE.json',{'state':'completed','protocol_hash':_checked(out)['protocol_hash'],
                                        'result_hash':'forged'})
            with self.assertRaises(ContractError):status(out)


if __name__=='__main__':unittest.main()
