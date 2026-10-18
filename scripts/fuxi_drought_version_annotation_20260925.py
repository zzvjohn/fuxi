"""Correct the generated v0.7.75 annotation to the actual partial change."""

import json
import os
import tempfile
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "data/version_registry.json"
VERSION = b'"version": "v0.7.75"'


def replace_once(blob, old, new):
    if blob.count(old) != 1:
        raise ValueError("expected unique version field not found")
    return blob.replace(old, new, 1)


if __name__ == "__main__":
    original = PATH.read_bytes()
    head, marker, tail = original.partition(VERSION)
    if not marker:
        raise ValueError("v0.7.75 missing")
    tail = replace_once(tail,
                        '"title": "S5 holdings-mark gate 近乎零容差（0.05% 缺失即判 data_"'.encode("utf-8"),
                        '"title": "S5 缺价诊断及低置信经济读数披露（P013部分）"'.encode("utf-8"))
    tail = replace_once(tail, b'"zero_adjudication": false',
                        b'"zero_adjudication": true')
    tail = replace_once(tail, b'"behavior": "\xe6\x9c\x89\xe6\x95\x88\xe5\x8f\x98\xe6\x9b\xb4"',
                        '"behavior": "只加诊断披露；S5裁决与默认零变化"'.encode("utf-8"))
    tail = replace_once(tail, '"rollback": "开关/环境变量回退"'.encode("utf-8"),
                        '"rollback": "从 reports/fuxi_drought_execution_20260925/baseline/ 恢复四份原始代码字节（无开关）"'.encode("utf-8"))
    changed = head + marker + tail
    reg = json.loads(changed)
    assert reg["versions"][-1]["version"] == "v0.7.75"
    assert reg["versions"][-1]["proposals"][0]["zero_adjudication"]
    fd, tmp = tempfile.mkstemp(prefix=".registry_v75_", dir=PATH.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(changed)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, PATH)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print("v0.7.75 corrected; historical registry prefix byte-identical")
