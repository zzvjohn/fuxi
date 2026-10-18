"""This explicitly requested manual campaign; no scheduler, account or broker access."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/fuxi_v1_announcements_20261003'
TARGET=OUT/'collection_v6/COLLECTION_RESULT.json'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--start-phase',type=int,choices=range(1,7),default=1)
    args=parser.parse_args()
    started=time.monotonic()
    while not TARGET.exists() or (ROOT/'data/fuxi_runtime.lock').exists():
        if time.monotonic()-started>14400:raise RuntimeError('collection wait exceeded bounded four-hour campaign')
        time.sleep(10)
    commands=[
        ['scripts/fuxi_notice_tests_20261003.py'],
        ['scripts/fuxi_notice_adjudicate_20261003.py'],
        ['scripts/fuxi_notice_ten_rounds_20261003.py','prepare'],
        ['scripts/fuxi_notice_ten_rounds_20261003.py','snapshot'],
        ['scripts/fuxi_notice_ten_rounds_20261003.py','preflight'],
        ['scripts/fuxi_notice_ten_rounds_20261003.py','rounds'],
    ]
    for phase,cmd in enumerate(commands,1):
        if phase<args.start_phase:continue
        print(json.dumps({'phase':phase,'command':cmd,'state':'starting'},ensure_ascii=False),flush=True)
        result=subprocess.run([sys.executable,'-X','utf8',*cmd],cwd=ROOT)
        if result.returncode:
            print(json.dumps({'phase':phase,'state':'stopped_for_repair','exit_code':result.returncode},ensure_ascii=False),flush=True)
            raise SystemExit(result.returncode)
        print(json.dumps({'phase':phase,'state':'completed'},ensure_ascii=False),flush=True)
    print(json.dumps({'state':'ten_round_campaign_completed','elapsed_seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':main()
