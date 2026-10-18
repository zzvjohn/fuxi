# -*- coding: utf-8 -*-
"""补写 forge_gen4 轨迹条目的 JQ 富字段。

🔴 缺陷发现 (2026-09-18):
    ralph_loop.jq_feedback() 写回轨迹时, 只写 jq_return / jq_sharpe,
    **会清空** jq_maxdd / jq_ic / jq_icir / jq_rating。
    证据: 本因子在 jq_feedback 前已写入这 4 个字段 (步骤 1/5),
          jq_feedback 执行后回读全部变为 None。
    ⇒ 正确顺序必须是: jq_feedback **之后** 再补写富字段。
    ⇒ 影响面: trigger_d_plus.py 现有流程同样是「先写→后 feedback」,
      故其 JQ 条目的 maxdd/ic/icir 也会被清空 (P-001 的 jq_ic/jq_icir
      透传只在当次 construct 时有效, 落盘后被抹掉)。
    本脚本按正确顺序补齐; 缺陷另立提案修 trigger_d_plus 顺序。
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAJ = os.path.join(REPO, "data", "trajectory_log.json")
TARGET = "traj_2561b7d3c712"

FIELDS = {
    "jq_maxdd": 29.49,
    "jq_ic": 0.0280,
    "jq_icir": 0.3695,
    "jq_rating": "MARGINAL",
    "jq_plus_ic_pct": 62.7,
    "jq_backtest_period": "2021-01-01~2026-07-30",
    "jq_notes": (
        "JQ 单因子回测 2021-01-01~2026-07-30 (日志至 2026-07-27 week=285, 5.6年): "
        "累计 64.57% / Sharpe 0.266 / maxDD 29.49%; 滞后 Rank IC 0.0280 / ICIR 0.3695 "
        "/ IC>0 62.7% (255期); local ICIR 0.4649 -> JQ ICIR 0.3695 有衰减; "
        "基准 000905 / topn80 / run_weekly / 周频语义; "
        "暴露波动率 2021-02~06 曾报 28.37x 拥挤预警后恒为 0.00x (读数存疑)"
    ),
}

with open(TRAJ, "r", encoding="utf-8") as f:
    d = json.load(f)

n = 0
for t in d.get("trajectories", []):
    if t.get("trajectory_id") == TARGET:
        for k, v in FIELDS.items():
            t[k] = v
        n += 1

txt = json.dumps(d, ensure_ascii=False, indent=2)
with open(TRAJ, "w", encoding="utf-8", newline="") as f:
    f.write(txt.replace("\r\n", "\n").replace("\n", "\r\n"))

# 复核
with open(TRAJ, "r", encoding="utf-8") as f:
    d2 = json.load(f)
t2 = next(x for x in d2["trajectories"] if x["trajectory_id"] == TARGET)
print("patched entries:", n)
for k in ["jq_validated", "jq_return", "jq_sharpe", "jq_maxdd", "jq_ic", "jq_icir",
          "jq_rating", "jq_plus_ic_pct", "final_outcome", "natural_freq",
          "d_plus_applied", "d_plus_at"]:
    print(f"  {k}: {t2.get(k)}")
raw = open(TRAJ, "rb").read()
print("bare LF:", len(re.findall(rb"(?<!\r)\n", raw)), "| CRLF:", raw.count(b"\r\n"))
