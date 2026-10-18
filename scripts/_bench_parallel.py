"""Benchmark: 4 parallel cold workers — wall-clock speedup and peak RSS.

Each worker is a fresh single-shot process (no pool cache), so this isolates the
M1 parallelism gain. Peak aggregate RSS tells us whether 4x concurrency fits in
the available memory (worker snapshot is a ~249 MB JSON -> larger in RAM).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "data/fuxi_unified/runs/unified_30590485f3f81771d1b85e5b/research" / \
    "campaign_30590485f3f81771d1b85e5b"
ATTEMPT = RUN / "attempts/attempt_04d1a221db254bc4b51feee21c6bf21d"
OUT = ROOT / "data/_bench_parallel"
N = 4


def main():
    base = json.loads((ATTEMPT / "request.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    procs = []
    started = time.monotonic()
    for index in range(N):
        request_path = OUT / f"request_{index}.json"
        request_path.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
        procs.append(subprocess.Popen(
            [sys.executable, "-B", "-X", "utf8", "-m", "fuxi_v1.workers",
             "--request", str(request_path), "--output", str(OUT / f"result_{index}.json")],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    peak = 0
    while any(p.poll() is None for p in procs):
        total = 0
        for p in procs:
            try:
                proc = psutil.Process(p.pid)
                total += proc.memory_info().rss + sum(
                    c.memory_info().rss for c in proc.children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        peak = max(peak, total)
        time.sleep(0.5)
    elapsed = time.monotonic() - started
    for p in procs:
        p.wait()
    print(f"parallel workers      : {N}")
    print(f"wall-clock (all done) : {elapsed:.1f}s")
    print(f"peak aggregate RSS    : {peak/1e9:.2f} GB")
    times = []
    for index in range(N):
        path = OUT / f"result_{index}.json"
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8")).get("worker_elapsed_seconds")
            times.append(value)
    print(f"worker_elapsed_seconds: {times}")
    if times:
        print(f"mean per-worker       : {sum(times)/len(times):.1f}s")
    print(f"\nvs serial sum         : {sum(times):.1f}s  -> speedup {sum(times)/elapsed:.2f}x")


if __name__ == "__main__":
    main()
