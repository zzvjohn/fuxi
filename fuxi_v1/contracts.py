"""Strict, content-addressed contracts for the first independent kernel."""

from datetime import datetime
import base64
import gzip
import hashlib
import json
import math
from collections.abc import Mapping


class ContractError(ValueError):
    pass


def _mapping_json(value):
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("unsupported canonical value: " + type(value).__name__)


def canonical_chunks(value):
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False, default=_mapping_json)
    for chunk in encoder.iterencode(value):
        yield chunk.encode("utf-8")


def canonical(value):
    return b"".join(canonical_chunks(value))


def digest(value):
    content = hashlib.sha256()
    for chunk in canonical_chunks(value):
        content.update(chunk)
    return content.hexdigest()


# P-20261016-014: lossless at-rest compression for large fact payloads.  Only the
# serialised JSON text is compressed; the content digest (hash column) is always
# taken over the UNCOMPRESSED canonical form, so the content-addressed integrity
# chain is unchanged.  Small facts stay plain text for debuggability.
_FACT_GZIP_MARKER = "GZIP:"
_FACT_COMPRESS_THRESHOLD = 100 * 1024  # compress facts whose canonical text >= 100 KB


def encode_fact_payload(text):
    """Compress a canonical JSON string for storage; small strings pass through."""
    if len(text) < _FACT_COMPRESS_THRESHOLD:
        return text
    return _FACT_GZIP_MARKER + base64.b64encode(gzip.compress(text.encode("utf-8"))).decode("ascii")


def decode_fact_payload(text):
    """Reverse :func:`encode_fact_payload`; plain text is returned unchanged."""
    if text.startswith(_FACT_GZIP_MARKER):
        return gzip.decompress(base64.b64decode(text[len(_FACT_GZIP_MARKER):])).decode("utf-8")
    return text


def load_fact(text):
    """Parse a stored fact payload (transparently decompressing large facts)."""
    return json.loads(decode_fact_payload(text))


# P-20261015-012: fields that annotate a market manifest rather than define its
# research content. Every market-content digest (certificate market_digest,
# assurance market_digest, bind verification, run_account source_digest) uses
# this same exclusion set so a certification write-back or an assurance binding
# never breaks a content binding.
MARKET_CONTENT_EXCLUSIONS = ('market_hash', 'certification_hash', 'certification_snapshot_hash',
                             'event_coverage_certified', 'tradability_certified',
                             'investment_assurance')


def require(condition, message):
    if not condition:
        raise ContractError(message)


def keys(obj, required, optional=()):
    require(isinstance(obj, dict), "object required")
    require(set(required) <= obj.keys(), "missing keys: " + str(set(required) - obj.keys()))
    require(obj.keys() <= set(required) | set(optional),
            "unknown keys: " + str(obj.keys() - set(required) - set(optional)))


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def timestamp(value):
    require(isinstance(value, str), "ISO timestamp required")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ContractError("invalid ISO timestamp") from exc
    require(result.tzinfo is not None, "timestamp must include UTC offset")
    return result


def evaluation(stage, status, reason, *, decision="not_applicable", reject_class=None,
               metrics=None, diagnostics=None, scope="diagnostic", n_pairs=0):
    result = dict(schema_version="fuxi.evaluation.1", stage=stage,
                  eval_status=status, decision=decision, reject_class=reject_class,
                  reason_code=reason, metrics=metrics, diagnostics=diagnostics or {},
                  evidence_scope=scope, n_pairs=n_pairs,
                  economic_reward_eligible=False, promotable=False)
    validate_evaluation(result)
    return result


def validate_evaluation(result):
    keys(result, ("schema_version", "stage", "eval_status", "decision", "reject_class",
                  "reason_code", "metrics", "diagnostics", "evidence_scope", "n_pairs",
                  "economic_reward_eligible", "promotable"))
    require(result["schema_version"] == "fuxi.evaluation.1", "unsupported evaluation schema")
    require(result["stage"] in {"E0", "E1", "E2", "E3", "E4", "E5"}, "invalid stage")
    status, decision = result["eval_status"], result["decision"]
    require(status in {"not_evaluated", "data_insufficient", "error", "precheck_rejected", "valid"},
            "invalid evaluation status")
    require(decision in {"pass", "reject", "not_applicable"}, "invalid decision")
    require(status == "valid" or decision == "not_applicable", "nonvalid decision")
    require(result["reject_class"] in {None, "statistical", "redundancy", "economic",
                                      "execution", "protocol"}, "invalid reject class")
    require((decision == "reject") == (result["reject_class"] is not None), "reject class mismatch")
    require(status == "valid" or result["metrics"] is None, "nonvalid formal metrics")
    require(type(result["n_pairs"]) is int and result["n_pairs"] >= 0, "invalid pair count")
    require(result["evidence_scope"] in {"diagnostic", "research", "account", "forward_acceptance"},
            "invalid scope")
    require(type(result["economic_reward_eligible"]) is bool and result["promotable"] is False,
            "evaluation cannot grant promotion directly")
    if result["economic_reward_eligible"]:
        require(status == "valid" and result["evidence_scope"] == "account" and result["n_pairs"] > 0
                and result["metrics"] is not None, "economic feedback requires identified paired accounts")
    canonical(result)
    return result


def load_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    def reject_constant(value):
        raise ContractError("nonfinite JSON constant: " + value)
    # Transparent gzip support (M3, P-20261015-008): a *.json.gz payload is the
    # same contract, just compressed. Detect by magic bytes so every existing
    # caller keeps working with either encoding.
    with open(path, "rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    with open(path, encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
