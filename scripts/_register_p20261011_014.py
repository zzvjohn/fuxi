# -*- coding: utf-8 -*-
"""Register proposal P-20261011-014 (enforce 单写锁容差修复) into the proposal library.

Surgical append: preserve all existing bytes/CRLF, only insert the new object
before the final list-closing bracket. No whole-file line-ending rewrite.
"""
import json

PATH = "data/stage1_optimization_proposals.json"

new = {
    "proposal_id": "P-20261011-014",
    "date": "2026-10-11",
    "status": "landed",
    "title": "enforce 单写锁容差修复：create_time 1ms→1s 误判活跃 owner 为死",
    "category": "稳定性/单写锁治理",
    "domains": ["T3模块/通道"],
    "proposal_change_kind": "behavior",
    "generator_hint": "manual",
    "source": "FUXI-LOCK-TOLERANCE-20261011",
    "trigger": "全量 enforce 37/54 处锁丢失 FATAL（Errno2），根因定位到 _verified_owner_is_live 用 1ms 容差在 Windows 高负载下误判活跃 owner 为死进程",
    "landed_detail": "run_contract.py 三处 create_time 容差 .001(1ms)→1.0(1s)：_verified_owner_is_live L170、validate_writer_lease L194(>=1.0) 与 L200(<1.0)，与死代码 _lock_owner_is_live 对齐。create_time 跨调用恒等，1s 足以区分「真活」与「PID 复用」。物证 data/fuxi_runtime.lock.stale.20261011_013754_473148（argv 为 stage2_s5_chain enforce，started 00:41:25，stale rename 01:37:54，进程仍活）。",
    "default_changes": [],
    "new_JQ_qualification_count": 0,
    "forward_confirmed": False,
    "rollback_plan": "恢复三处 .001(1ms) 容差即可回退；不影响裁决口径，仅放宽锁活性判定的时间窗",
    "references": [
        "research/factor_alchemy/run_contract.py",
        "data/fuxi_runtime.lock.stale.20261011_013754_473148",
    ],
}

# Serialize the single object with indent=2 (LF), then to CRLF + 2-space indent.
raw = json.dumps(new, ensure_ascii=False, indent=2)          # uses \n
raw = raw.replace("\n", "\r\n")
indented = "\r\n".join(("  " + ln if ln else ln) for ln in raw.split("\r\n"))

with open(PATH, "rb") as fh:
    b = fh.read()

# File is a top-level JSON list; must end with '\r\n]'.
assert b.endswith(b"\r\n]"), "unexpected tail: %r" % b[-20:]
body = b[:-3]  # strip final '\r\n]' -> body ends with the last object's closing '}'

new_bytes = body + b",\r\n" + indented.encode("utf-8") + b"\r\n]"

with open(PATH, "wb") as fh:
    fh.write(new_bytes)

# Verify round-trip.
parsed = json.loads(new_bytes.decode("utf-8"))
assert isinstance(parsed, list), type(parsed)
assert parsed[-1]["proposal_id"] == "P-20261011-014", parsed[-1]["proposal_id"]
print("[register] appended P-20261011-014; total proposals now", len(parsed))
