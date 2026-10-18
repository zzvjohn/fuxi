"""Seal the actual complete kernel test run and all tested source hashes."""
import contextlib
import io
import json
import sys
import time
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.artifacts import code_manifest
from fuxi_v1.datasets import file_hash
from fuxi_v1.contracts import require
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_announcements_20261003/tests_v2'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'TEST_RESULT.json';require(not target.exists(),'test result already sealed')
    before=code_manifest();tests={p.name:file_hash(p) for p in (ROOT/'fuxi_v1/tests').glob('*.py')}
    output=io.StringIO();started=time.monotonic()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'fuxi_v1/tests'),top_level_dir=str(ROOT))
    with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
        result=unittest.TextTestRunner(stream=output,verbosity=1).run(suite)
    require(before==code_manifest(),'code changed while testing')
    require(tests=={p.name:file_hash(p) for p in (ROOT/'fuxi_v1/tests').glob('*.py')},'test source changed')
    log=OUT/'TEST_OUTPUT.txt'
    with log.open('x',encoding='utf-8',newline='\n') as handle:handle.write(output.getvalue())
    record={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'skipped':len(result.skipped),'successful':result.wasSuccessful(),'elapsed_seconds':time.monotonic()-started,
        'code_hashes':before,'test_hashes':tests,'test_output_sha256':file_hash(log),'runner_sha256':file_hash(Path(__file__))}
    with target.open('x',encoding='utf-8') as handle:json.dump(record,handle,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in record.items() if k not in ('code_hashes','test_hashes')},ensure_ascii=False),flush=True)
    require(result.wasSuccessful(),'kernel tests failed')


if __name__=='__main__':
    with controlled_writer(OUT):main()
