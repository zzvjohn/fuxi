"""One-pair, frozen-prompt LLM generator comparison. No factor-pool writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from llm_generator import LLMGenerator, _schema_system_prompt, validate_llm_schema  # noqa: E402
from semantic_plan import load_plan  # noqa: E402

OUT = ROOT / "reports/fuxi_generator_rebuild_20260925/equal_budget"
PLAN_FILE = ROOT / "research/factor_alchemy/semantic_plans_20260925.json"
PLAN_ID = "pv_gap_rejection_v1"
N = 4
MAX_TOKENS = 3072
TEMPERATURE = 0.8
PARADIGM = "隔夜跳空回吐"


def sha(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def prepare() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    plan = load_plan(PLAN_FILE, PLAN_ID, "daily")
    generator = LLMGenerator(output_dir=OUT)
    system = _schema_system_prompt(weekly_mode=False)
    baseline = generator.build_prompt(PARADIGM, N, weekly_mode=False)
    generator.set_semantic_plan(plan)
    planned = generator.build_prompt(PARADIGM, N, weekly_mode=False)
    assert baseline != planned
    prompts = {"baseline": baseline, "semantic_plan": planned}
    contract = {
        "schema": "fuxi_equal_budget_generation/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "historical research-exposed, isolated prompt A/B; no S5 or JQ grant",
        "arms": list(prompts),
        "model": "deepseek-chat",
        "calls_per_arm": 1,
        "candidates_requested_per_arm": N,
        "max_tokens_per_call": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system_prompt_sha256": sha(system),
        "prompt_sha256": {arm: sha(prompt) for arm, prompt in prompts.items()},
        "plan_id": PLAN_ID,
        "plan_file_sha256": hashlib.sha256(PLAN_FILE.read_bytes()).hexdigest(),
        "frequency": "daily",
        "paradigm": PARADIGM,
        "reviewer": "off",
        "generator_feedback": "identical prompt context aside from semantic plan",
        "historical_data": "none read by this script; generator may read existing feedback in build_prompt",
        "promotion_eligible": False,
    }
    existing = OUT / "protocol.json"
    if existing.exists():
        prior = json.loads(existing.read_text(encoding="utf-8"))
        for key in ("prompt_sha256", "system_prompt_sha256", "plan_file_sha256"):
            if prior[key] != contract[key]:
                raise RuntimeError(f"frozen {key} changed")
        return prior
    for arm, prompt in prompts.items():
        (OUT / f"{arm}_prompt.txt").write_text(prompt, encoding="utf-8", newline="\n")
    (OUT / "system_prompt.txt").write_text(system, encoding="utf-8", newline="\n")
    existing.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    return contract


def execute(allow_transport_retry: bool = False) -> dict:
    contract = prepare()
    from llm_client import get_llm_client, resolve_api_key_with_source
    key, source = resolve_api_key_with_source()
    if not key:
        raise RuntimeError("DeepSeek credentials unavailable")
    client = get_llm_client()
    system = (OUT / "system_prompt.txt").read_text(encoding="utf-8")
    assert sha(system) == contract["system_prompt_sha256"]
    rows = {}
    for arm in contract["arms"]:
        path = OUT / f"{arm}_prompt.txt"
        prompt = path.read_text(encoding="utf-8")
        assert sha(prompt) == contract["prompt_sha256"][arm]
        response_file = OUT / f"{arm}_response.txt"
        result_file = OUT / f"{arm}_result.json"
        prior = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else None
        if response_file.exists():
            raise RuntimeError(f"{arm} already called; refusing another draw")
        if prior and not (allow_transport_retry and prior.get("status") == "error"):
            raise RuntimeError(f"{arm} already attempted; explicit transport retry required")
        try:
            response = client.chat_with_system(
                system_prompt=system, user_prompt=prompt,
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            response_file.write_text(response, encoding="utf-8", newline="\n")
            generator = LLMGenerator(output_dir=OUT)
            if arm == "semantic_plan":
                generator.set_semantic_plan(load_plan(PLAN_FILE, PLAN_ID, "daily"))
            raw = generator.parse_response(response)
            tagged = generator.tag_semantic_candidates(raw)
            unique = generator._batch_dedup_ast(tagged)
            rows[arm] = {
                "status": "completed", "response_sha256": sha(response),
                "parsed_n": len(raw), "unique_n": len(unique),
                "schema_valid_n": sum(not validate_llm_schema(
                    c["formula"], weekly_mode=False) for c in unique),
                "plan_valid_n": sum(not c.get("semantic_plan_violations") for c in unique)
                    if arm == "semantic_plan" else None,
                "candidates": unique,
            }
        except Exception as exc:
            rows[arm] = {"status": "error", "error_type": type(exc).__name__,
                         "error": str(exc)[:300]}
        if prior:
            rows[arm]["prior_transport_error"] = {
                key: prior.get(key) for key in ("status", "error_type", "error")}
        result_file.write_text(
            json.dumps(rows[arm], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n")
    report = {"schema": "fuxi_equal_budget_generation_result/v1",
              "protocol_sha256": hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest(),
              "credential_source": source, "arms": rows,
              "economic_verdict": "not_evaluated",
              "formal_drought_resolved": False}
    (OUT / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8", newline="\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--retry-transport", action="store_true",
                        help="Only after explicit approval; keep prior transport error")
    args = parser.parse_args()
    value = prepare() if args.action == "prepare" else execute(args.retry_transport)
    print(json.dumps({k: v for k, v in value.items() if k != "arms"},
                     ensure_ascii=False, indent=2))
    if args.action == "execute":
        print({arm: {"status": row["status"], "parsed_n": row.get("parsed_n"),
                     "schema_valid_n": row.get("schema_valid_n")}
               for arm, row in value["arms"].items()})
