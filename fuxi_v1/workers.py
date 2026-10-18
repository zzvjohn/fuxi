"""Isolated trusted computation worker. Process isolation is not a hostile-code sandbox.

M2 (P-20261015-007): a ``--loop`` mode keeps the worker alive across requests and
reuses the read-only, input-hash-keyed snapshot cache. State discipline (SPEEDUP
4b): every per-request mutable semantic (score-ref stats, AST memo) is reset
before each job; the worker exits after ``--max-requests`` jobs so the launcher
can recycle the process against memory creep.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from .contracts import load_json, canonical, canonical_chunks, digest, require


def _load_payload(request):
    payload = load_json(request["input_path"])
    require(digest(payload) == request["input_hash"], "worker input changed")
    require(payload["snapshot"]["exposure"] != "forward_sealed", "research worker cannot read protected data")
    from .full_market_support import hydrate_payload
    return hydrate_payload(payload)


def _reset_per_request_state(snapshot):
    """SPEEDUP 4b: no cross-request mutable state may survive into a new job.

    ``_score_ref_stats`` carries per-request evidence semantics (it travels with
    the worker result) and ``_score_cache`` is the read-through AST memo. Both
    must start from the same empty state as the single-shot path, otherwise the
    pooled worker's score_ref_stats would diverge bit-wise from serial runs.
    """
    if hasattr(snapshot, "_score_ref_stats"):
        snapshot._score_ref_stats = None
    cache = getattr(snapshot, "_score_cache", None)
    if cache is not None:
        cache.clear()


def _evaluate_with(request, payload, snapshot, labels):
    """Evaluate one unpacked request against prebuilt read-only state."""
    from .dsl import evaluate_ast
    from .research_evaluation import evaluate_candidate
    if request.get("job_kind") == "portfolio":
        from .portfolio import portfolio_experiment
        return {"portfolio_experiment": portfolio_experiment(payload, request["policy"],
                 request["baseline_scores"], request["top_n"])}
    if request.get("job_kind") == "model":
        from .models import model_experiment
        return model_experiment(payload, request["features"], request["policy"], request["baseline_scores"],
                                request["top_n"], request["algorithm"], request["seed"])
    scores = evaluate_ast(request["candidate"]["canonical_ast"], snapshot)
    baseline = evaluate_ast(request["baseline_ast"], snapshot)
    pool_scores = [{"candidate_id": p["candidate_id"], "scores": evaluate_ast(p["canonical_ast"], snapshot)}
                   for p in request.get("pool_candidates", [])]
    result = evaluate_candidate(request["candidate"], scores, baseline, snapshot, labels, payload["market"],
                                request["policy"], payload["research_clock"], top_n=request["top_n"],
                                pool_scores=pool_scores)
    from .learning import paired_parent
    from .contracts import timestamp
    window = request['policy']['request']
    result['learning_parent_comparisons'] = []
    for parent in request.get('parent_candidates', []):
        parent_scores = evaluate_ast(parent['canonical_ast'], snapshot)
        comparison = paired_parent(scores, parent_scores, snapshot, labels, window['research_start'],
                                   window['research_end'], timestamp(payload['research_clock']))
        from .portfolio import compile_strategy, mix_scores
        from .simulation import run_account
        from .research_evaluation import paired_accounts
        spec = dict(result['strategy_ir']['strategy'], candidate_id=parent['candidate_id'])
        parent_ir = compile_strategy(spec, mix_scores(baseline, parent_scores, .5), snapshot, digest(payload['market']['execution']))
        parent_account = run_account(parent_ir, payload['market'], start=window['research_start'], end=window['research_end'])
        net = paired_accounts(result['candidate_account'], parent_account, window)
        comparison['economic_status'] = net['eval_status']
        comparison['economic_reason'] = net['reason_code']
        comparison['paired_net_increment'] = net.get('metrics', {}).get('paired_mean_increment') if net['economic_reward_eligible'] else None
        result['learning_parent_comparisons'].append({'parent_id': parent['candidate_id'], **comparison})
    # P5.1: score cache evidence (mode + compute/hit counters) travels with the worker result.
    stats = getattr(snapshot, "_score_ref_stats", None)
    if stats is not None:
        result["score_ref_stats"] = dict(stats)
    return {"result": result, "scores": scores}


def evaluate_job(request):
    from .data import Snapshot, Labels
    payload = _load_payload(request)
    from .full_market_support import unpack_grids
    request = unpack_grids(request)
    snapshot = Snapshot(payload["snapshot"])
    labels = Labels(payload["labels"], snapshot)
    return _evaluate_with(request, payload, snapshot, labels)


def _write_output(output_path, result):
    output = Path(output_path)
    require(not output.exists(), "worker output already exists")
    temp = output.with_suffix(".pending")
    from .full_market_support import pack_grids
    packed = pack_grids(result, output.parent / "result_panels")
    if output.suffix == ".gz":
        # M3 (P-20261015-008): gzip transport is byte-exact over the canonical
        # JSON payload; the launcher picks the filename, the worker honours it.
        import gzip
        with gzip.open(temp, "wb", compresslevel=6) as stream:
            for chunk in canonical_chunks(packed):
                stream.write(chunk)
            stream.write(b"\n")
    else:
        with temp.open("xb") as stream:
            for chunk in canonical_chunks(packed):
                stream.write(chunk)
            stream.write(b"\n")
    os.replace(temp, output)


def run_loop(max_requests=None):
    """M2 loop protocol: NDJSON requests on stdin, NDJSON status lines on stdout.

    Each request line carries the single-shot request fields plus ``output``
    (absolute result path) and ``request_index`` (echoed in the status line so
    the launcher can match completions deterministically). The worker exits
    after ``max_requests`` jobs or on stdin EOF; it never exits mid-job.
    """
    from .data import Snapshot, Labels
    from .full_market_support import unpack_grids
    cache = {}
    processed = 0
    while True:
        # Check the budget BEFORE reading: a request line we are not going to
        # process must never be consumed from the pipe (the launcher writes
        # the next job to the replacement process, so consuming it here would
        # silently drop that job and surface as worker_unexpected_exit).
        if max_requests is not None and processed >= max_requests:
            break
        raw = sys.stdin.readline()
        if not raw:
            break
        line = raw.strip()
        if not line:
            continue
        request = json.loads(line)
        index = request.get("request_index")
        output_path = request.get("output")
        started = time.monotonic()
        try:
            key = (request.get("input_path"), request.get("input_hash"))
            entry = cache.get(key)
            if entry is None:
                payload = _load_payload(request)
                snapshot = Snapshot(payload["snapshot"])
                labels = Labels(payload["labels"], snapshot)
                entry = cache[key] = (payload, snapshot, labels)
            payload, snapshot, labels = entry
            _reset_per_request_state(snapshot)
            result = _evaluate_with(unpack_grids(request), payload, snapshot, labels)
            result["worker_elapsed_seconds"] = time.monotonic() - started
            _write_output(output_path, result)
            print(json.dumps({"ok": True, "request_index": index},
                             ensure_ascii=False, separators=(",", ":")), flush=True)
        except Exception as exc:
            cache.pop(key, None)
            print(json.dumps({"ok": False, "request_index": index, "error_type": type(exc).__name__,
                              "reason": str(exc)}, ensure_ascii=False, separators=(",", ":")), flush=True)
        processed += 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request")
    parser.add_argument("--output")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max-requests", type=int, default=None)
    args = parser.parse_args()
    if args.loop:
        return run_loop(args.max_requests)
    require(args.request and args.output, "single-shot mode requires --request and --output")
    started = time.monotonic()
    try:
        result = evaluate_job(load_json(args.request))
        result["worker_elapsed_seconds"] = time.monotonic()-started
        _write_output(args.output, result)
        return 0
    except Exception as exc:
        print(json.dumps({"eval_status": "error", "error_type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
