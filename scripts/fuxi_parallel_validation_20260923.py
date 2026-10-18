"""Persist regression and protected-input checks for the parallel remediation."""
import io
import sys
import unittest
from pathlib import Path
from fuxi_execution_lab_20260923 import read, write, digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/window_evidence_20260923'


def main():
    assert not (OUT/'regression.json').exists()
    protocol=read(OUT/'protocol.json')
    checked={}
    for name, expected in protocol['protected'].items():
        actual=digest(ROOT/name)
        assert actual==expected,name
        checked[name]=actual
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'research/factor_alchemy/tests'))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    paths=[ROOT/'scripts'/n for n in ['fuxi_distribution_evidence_20260923.py',
             'fuxi_window_reference_20260923.py','fuxi_window_replay_20260923.py',
             'fuxi_window_request_audit_20260923.py','fuxi_parallel_validation_20260923.py']]
    paths+=list((ROOT/'research/factor_alchemy/tests').glob('test_*.py'))
    write(OUT/'regression.json',{'tests_run':result.testsRun,'successful':result.wasSuccessful(),
          'output':stream.getvalue(),'protected_sha256':checked,
          'code_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in paths}})
    print('Tests:',result.testsRun,'success:',result.wasSuccessful(),'protected:',len(checked))
    assert result.testsRun>0 and result.wasSuccessful()


if __name__=='__main__':main()
