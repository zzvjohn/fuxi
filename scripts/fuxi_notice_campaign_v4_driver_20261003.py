"""Serial rc11 source-bound recovery; rc9 ten receipts remain immutable."""
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
commands=[['scripts/fuxi_notice_ten_rounds_v4_20261003.py',x] for x in ('prepare','snapshot','preflight','rounds')]
started=time.monotonic()
for number,command in enumerate(commands,1):
    print({'phase':number,'command':command,'state':'starting'},flush=True)
    p=subprocess.run([sys.executable,'-X','utf8',*command],cwd=ROOT)
    if p.returncode:raise SystemExit(p.returncode)
    print({'phase':number,'state':'completed'},flush=True)
print({'rc11_ten_rounds_completed':True,'elapsed_seconds':time.monotonic()-started},flush=True)
