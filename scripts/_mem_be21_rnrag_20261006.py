# -*- coding: utf-8 -*-
"""BE.21 RAG 收口巡检 —— 追加到 daily06（二进制追加，铁律 #15；写前重取双边 max ⇒ BE.21）"""
import hashlib
import sys
from pathlib import Path

P = Path("E:/quant/.workbuddy/memory/2026-10-06.md")
raw = P.read_bytes()

assert raw.count(b"## BE.21") == 0, "BE.21 already exists"
assert raw.count(b"## BE.19") == 1, "BE.19 missing (prereq)"
assert raw.endswith(b"\r\n"), "unexpected tail"

BLOCK = (
    "## BE.21 RAG 收口：BE.19/BE.20 增量入库与直取复核（2026-10-06）\r\n"
    "\r\n"
    "- ingest --only details,daily06,changelog：+14 chunk（去重跳 869）；源侧单元 322/15/135，chunk 352/21/510。\r\n"
    "- --verify：引用 38 / 目录 721，可达 38、缺失 0。\r\n"
    "- selfcheck：entries=721 ok=720 empty=0（唯一 title_mismatch=_intro 自定义标题例外，已知允许）。\r\n"
    "- 直取复核（铁律 #16）：--anchor BE.19 → daily06 M4 翻默认落地；--anchor BE.20 → details 验证链 F1-F5 + 探针自纠；无同位遮蔽。\r\n"
    "- 收尾链序（铁律 #14）：ingest → --verify → selfcheck 全绿。\r\n"
    "\r\n"
)

b = BLOCK.encode("utf-8")
assert b"\n" not in b.replace(b"\r\n", b""), "bare LF in block"

pre_bytes = len(raw)
pre_crlf = raw.count(b"\r\n")
pre_sha = hashlib.sha256(raw).hexdigest()[:16]

with open(P, "ab") as f:
    f.write(b)

new = P.read_bytes()
d_bytes = len(new) - pre_bytes
d_crlf = new.count(b"\r\n") - pre_crlf
new_sha = hashlib.sha256(new).hexdigest()[:16]

exp_crlf = BLOCK.count("\r\n")
new_crlf = new.count(b"\r\n")
print(f"[before] bytes={pre_bytes} crlf={pre_crlf} sha={pre_sha}")
print(f"[block ] bytes=+{len(b)} crlf=+{exp_crlf} (expected)")
print(f"[after ] bytes={len(new)} crlf={new_crlf} sha={new_sha}")
print(f"[check ] d_bytes={d_bytes} d_crlf={d_crlf}")

assert d_bytes == len(b), "byte delta mismatch"
assert d_crlf == BLOCK.count("\r\n"), "crlf delta mismatch"
print("[ok] BE.21 appended")
