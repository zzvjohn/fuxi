from pathlib import Path
import tempfile
import unittest
from fuxi_v1.contracts import ContractError,load_json
from fuxi_v1.fusion_study import freeze
from fuxi_v1.frozen_study_runtime import prepare
from fuxi_v1.unified_runner import atomic,file_sha
from fuxi_v1.tests.test_fusion_study import _fixture

class FrozenStudyRuntimeTests(unittest.TestCase):
    def build(self,root):
        source=root/'input.json';atomic(source,_fixture())
        output=root/'study'
        protocol=freeze(source,output,{'session_count':120,'seed':17,'rounds':5,'candidates_per_round':2,
                                      'source_sha256':file_sha(source)})
        atomic(output/'CONTROLLER_REQUEST.json',{'code':{'native':protocol['code_hashes']}})
        return output,protocol

    def test_exact_original_copy_idempotent_and_extra_module_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output,protocol=self.build(Path(tmp))
            runtime=prepare(output)
            self.assertEqual({p.name:file_sha(p) for p in (runtime/'fuxi_v1').glob('*.py')},protocol['code_hashes'])
            self.assertEqual(prepare(output),runtime)
            (runtime/'fuxi_v1/injected.py').write_bytes(b'raise RuntimeError("must not execute")\n')
            with self.assertRaisesRegex(ContractError,'extra/changed'):prepare(output)

    def test_original_module_or_isolated_copy_change_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            output,_=self.build(Path(tmp));runtime=prepare(output)
            (runtime/'fuxi_v1/fusion_study.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ContractError,'isolated native'):prepare(output)
            (output/'code/fusion_study.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ContractError,'frozen study code'):prepare(output)

    def test_controller_native_identity_must_match_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            output,_=self.build(Path(tmp))
            atomic(output/'CONTROLLER_REQUEST.json',{'code':{'native':{'fake.py':'0'*64}}})
            with self.assertRaisesRegex(ContractError,'controller and original'):prepare(output)

if __name__=='__main__':unittest.main()
