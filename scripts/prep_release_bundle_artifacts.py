# -*- coding: utf-8 -*-
"""B12.4 (#14): compute the six release-bundle artifact hashes for a candidate.

Usage:
    python -X utf8 scripts/prep_release_bundle_artifacts.py <candidate_bundle_dir> [--data-receipt <id>]

Produces <bundle_dir>/release_artifacts.json with the six 64-hex hashes
required by ReleaseService.bundle (acceptance.py):
    code           -> digest of the fuxi_v1 code manifest (execution code identity)
    model          -> sha256 of candidate.json (the admitted candidate/model definition)
    strategy       -> sha256 of strategy_ir.json (frozen research IR, adapter-declared)
    data_contract  -> the bound data_contract receipt id (64-hex)
    policy         -> sha256 of policy.json (portfolio/execution policy)
    state_schema   -> sha256 of compiler.json (serving/state contract incl. universe adapter)

The JSON additionally records HOW each hash was derived so a reviewer can
re-verify. When steps jq_parity (3) and E5 forward_acceptance (6) actually
land, ReleaseService.bundle(forward_receipt, artifacts) consumes exactly
these six values — this script never fabricates the forward receipt itself.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('E:/quant')


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    data_receipt = None
    argv = sys.argv[1:]
    if '--data-receipt' in argv:
        data_receipt = argv[argv.index('--data-receipt') + 1]
    if not args:
        print('usage: prep_release_bundle_artifacts.py <candidate_bundle_dir> [--data-receipt <64hex>]')
        return 1
    bundle = Path(args[0])
    if not bundle.is_dir():
        print('bundle dir not found:', bundle)
        return 1

    if data_receipt is None:
        data_receipt = 'data_contract_0eefe834d1d248a496a8859157c7caac414009245a139dace1d3c827a29ee9ea'
    receipt_hex = data_receipt.replace('data_contract_', '')
    if len(receipt_hex) != 64 or any(c not in '0123456789abcdef' for c in receipt_hex):
        print('data receipt id is not 64-hex:', data_receipt)
        return 1

    needed = ['candidate.json', 'strategy_ir.json', 'policy.json', 'compiler.json']
    missing = [n for n in needed if not (bundle / n).is_file()]
    if missing:
        print('bundle missing artifacts:', missing)
        return 1

    sys.path.insert(0, str(ROOT))
    from fuxi_v1.contracts import digest
    from fuxi_v1.artifacts import code_manifest
    code_hash = digest(code_manifest())

    artifacts = {
        'code': code_hash,
        'model': sha256_file(bundle / 'candidate.json'),
        'strategy': sha256_file(bundle / 'strategy_ir.json'),
        'data_contract': receipt_hex,
        'policy': sha256_file(bundle / 'policy.json'),
        'state_schema': sha256_file(bundle / 'compiler.json'),
    }
    out = {
        'schema_version': 'fuxi.release_artifact_hashes.1',
        'bundle_dir': str(bundle),
        'artifacts': artifacts,
        'derivation': {
            'code': 'digest(code_manifest()) — fuxi_v1 execution code identity',
            'model': 'sha256(candidate.json) — admitted candidate/model definition',
            'strategy': 'sha256(strategy_ir.json) — frozen research IR',
            'data_contract': 'bound data_contract receipt id (%s)' % data_receipt,
            'policy': 'sha256(policy.json) — portfolio/execution policy',
            'state_schema': 'sha256(compiler.json) — serving/state contract (universe adapter included)',
        },
        'usage_note': ('Consumed by ReleaseService.bundle(forward_receipt, artifacts) ONLY after '
                       'jq_parity (real JQ, n_pairs>0) and E5 forward_acceptance actually occur; '
                       'this file does not fabricate the forward receipt.'),
        'produced_at': '2026-10-16',
    }
    path = bundle / 'release_artifacts.json'
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('release_artifacts.json written:', path)
    print(json.dumps(artifacts, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
