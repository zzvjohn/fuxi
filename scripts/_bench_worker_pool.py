"""Benchmark: cold vs warm worker evaluation (M2 snapshot cache effect).

Feeds the SAME real request twice into one persistent loop worker. Request 1 is
cold (pays load_json + Snapshot construction of the ~249 MB input); request 2 is
warm (hits the (input_path, input_hash) cache). The delta is the M2 speedup.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "data/fuxi_unified/runs/unified_30590485f3f81771d1b85e5b/research" / \
    "campaign_30590485f3f81771d1b85e5b"
ATTEMPT = RUN / "attempts/attempt_04d1a221db254bc4b51feee21c6bf21d"
OUT = ROOT / "data/_bench_pool"


def main():
    req = json.loads((ATTEMPT / "request.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for index in (1, 2):
        payload = dict(req)
        payload["output"] = str(OUT / f"result_{index}.json")
        payload["request_index"] = index
        lines.append(json.dumps(payload, ensure_ascii=False))

    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", "-m", "fuxi_v1.workers",
         "--loop", "--max-requests", "2"],
        input="\n".join(lines) + "\n", capture_output=True, text=True, cwd=str(ROOT),
    )
    total = time.monotonic() - started
    print(f"total wall-clock : {total:.1f}s")
    print(f"stdout           : {(proc.stdout or '').strip()[-400:]}")
    if proc.returncode != 0:
        print(f"stderr tail      : {(proc.stderr or '')[-1200:]}")
    times = []
    for index in (1, 2):
        path = OUT / f"result_{index}.json"
        if not path.exists():
            print(f"request {index}    : MISSING")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        value = data.get("worker_elapsed_seconds")
        times.append(value)
        print(f"request {index}    : worker_elapsed_seconds = {value}")
    if len(times) == 2 and times[0] and times[1]:
        print(f"\ncold={times[0]:.1f}s  warm={times[1]:.1f}s  "
              f"speedup={times[0]/times[1]:.2f}x  saved={times[0]-times[1]:.1f}s/candidate")


if __name__ == "__main__":
    main()
