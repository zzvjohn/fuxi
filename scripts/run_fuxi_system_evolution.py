"""Shared manual system-improvement gate for legacy and native Fuxi engines."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.system_evolution import assess, freeze, history, inspect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("history")
    p = sub.add_parser("freeze")
    p.add_argument("--plan", required=True)
    p = sub.add_parser("assess")
    p.add_argument("--change-id", required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--incumbent", required=True)
    p.add_argument("--challenger", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("--change-id", required=True)
    args = parser.parse_args()
    if args.mode == "freeze":
        output = freeze(args.workspace, args.plan)
    elif args.mode == "assess":
        output = assess(args.workspace, args.change_id, args.anchor, args.incumbent, args.challenger)
    elif args.mode == "inspect":
        output = inspect(args.workspace, args.change_id)
    else:
        output = history(args.workspace)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
