"""伏羲系统慢循环（慢进化）自动化入口。

把已建成但零调用的 `research/system_evolution/gate.py`（v1.1 系统改进影子验收层）
接入闭环，补齐 BLUEPRINT 第 8 节的"系统慢循环"。全流程自动化，唯一人工闸是转正：

    observe  —— 从快循环观测原料聚合系统级观测快照（阶段 A，零副作用）
    draft    —— 规则提炼候选假设（非 LLM，dry-run，不落盘）（阶段 B）
    freeze   —— 冻结单机制假设为不可变 FROZEN.json（透传 gate.freeze）
    assess   —— 三版本同口径影子比对（透传 gate.assess）
    history  —— 列出已核对的系统学习记录（透传 gate.history）
    promote  —— 生成转正评分卡（建议 + 理由 + 标准逐条满足情况）（阶段 D）

核心引擎分模块（可独立 import）：
    research/system_evolution/gate.py       freeze/assess/inspect/history
    research/system_evolution/observe.py    observe/draft（观测沉淀 + 规则提炼）
    research/system_evolution/scorecard.py  build_scorecard（转正评分卡）

边界：本入口只出影子审理建议 + 评分卡，不授 S5/JQ/生产资格，不自动翻默认。
转正永远人工；`production_activation_allowed` 恒 false 不破。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.system_evolution.gate import assess, freeze, history, inspect  # noqa: E402
from research.system_evolution.observe import draft, observe  # noqa: E402
from research.system_evolution.scorecard import build_scorecard  # noqa: E402


def promote(workspace, change_id):
    """生成转正评分卡（建议 + 理由 + 标准逐条满足情况）。"""
    return build_scorecard(workspace, change_id)


def main():
    parser = argparse.ArgumentParser(description="伏羲系统慢循环（慢进化）自动化入口")
    parser.add_argument("--workspace", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("observe").add_argument("--limit", type=int, default=None)
    p = sub.add_parser("draft")
    p.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("freeze")
    p.add_argument("--plan", required=True)
    p = sub.add_parser("assess")
    p.add_argument("--change-id", required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--incumbent", required=True)
    p.add_argument("--challenger", required=True)
    p = sub.add_parser("history")
    p = sub.add_parser("promote")
    p.add_argument("--change-id", required=True)

    args = parser.parse_args()
    ws = args.workspace
    if args.mode == "observe":
        output = observe(ws, limit=args.limit)
    elif args.mode == "draft":
        output = draft(ws, limit=args.limit)
    elif args.mode == "freeze":
        output = freeze(ws, args.plan)
    elif args.mode == "assess":
        output = assess(ws, args.change_id, args.anchor, args.incumbent, args.challenger)
    elif args.mode == "history":
        output = history(ws)
    else:
        output = promote(ws, args.change_id)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
