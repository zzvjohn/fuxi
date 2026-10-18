# -*- coding: utf-8 -*-
"""data/stage1_optimization_proposals.json（提案账本）安全读写模块 — 2026-09-25

存在理由（P-20260925-008 的落地实现）
--------------------------------------------------------------------
因子池侧早在 2026-08-21 覆盖事故后就建立了 scripts/proposals_io.py
（O_EXCL 锁 + 原子写 + 写后读回校验 + 锁内闭环），**账本侧却一直没有对等设施**：
历史上账本的生产写入方全部是 ad-hoc 脚本，各自裸「读 → 改 → 写」，
无锁、无原子写、无统一回读校验。后果有两类，2026-09-25 都已实测：

  1. 并发互覆：本次登记期间观测到账本被另一写入方连续改写（01:04 / 01:06 / 01:12），
     裸读写会静默丢掉一方的登记。
  2. **行尾翻转**：ad-hoc 脚本用 open(tmp,'w') 写盘，Windows 文本模式把 '\n' 换成
     '\r\n'。账本自 2026-09-24 起一贯为 LF（连续 ~10 次写入 bare_LF>0 / CRLF=0），
     被一次登记脚本静默翻成 CRLF（P-20260925-011 记录并已复原）。

本模块把这两件事一次解决：进程锁 + 原子写 + **行尾保真** + 写后字节回读校验。
实现委托 scripts/json_store.py，与 proposals_io.py 共用同一套底层。

用法
--------------------------------------------------------------------
    from ledger_io import load, load_and_modify, append_proposals
    append_proposals([{ "proposal_id": "P-20260926-001", ... }])

纪律
--------------------------------------------------------------------
  - 账本现存形态是**裸 list**（历史上有过 dict 形态的讨论）；本模块两种形态都能写回，
    且 **不改变** 调用时的形态 ⇒ 不做隐式格式迁移。
  - 追加时按 proposal_id 去重；已存在的 ID 一律跳过并回报，不覆盖既有条目。
  - 写入后必须回读断言「既有 ID 零丢失 + ID 无重复」。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json_store  # noqa: E402

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data",
                                    "stage1_optimization_proposals.json"))


def _records(data):
    """返回 (形态标记, 记录列表引用)。形态标记 True 表示 dict 包裹形态。"""
    if isinstance(data, dict):
        for key in ("proposals", "items"):
            if isinstance(data.get(key), list):
                return True, key, data[key]
        raise ValueError("账本 dict 形态但找不到 list 字段（proposals/items）")
    if isinstance(data, list):
        return False, None, data
    raise ValueError(f"账本结构不可识别: {type(data).__name__}")


def load(path: str | None = None):
    return json_store.load(path or PATH)


def eol_name(path: str | None = None) -> str:
    return "crlf" if json_store.detect_eol(path or PATH) == json_store.CRLF else "lf"


def save(data, path: str | None = None) -> dict:
    """原子写（行尾保真 + 写后字节回读），保留原 dict/list 形态。"""
    return json_store.atomic_write(path or PATH, data)


def load_and_modify(modifier, path: str | None = None):
    """锁内闭环：读最新 → modifier 修改 → 原子写 → 回读校验。"""
    return json_store.load_and_modify(path or PATH, modifier)


def append_proposals(new_items, path: str | None = None) -> dict:
    """按 proposal_id 去重追加。返回统计（含既有 ID 零丢失 / 无重复 断言结果）。"""
    p = path or PATH
    stat: dict = {}

    def _mod(data):
        is_dict, key, records = _records(data)
        ids_before = [x.get("proposal_id") for x in records]
        stat["n_before"] = len(records)
        existing = {i for i in ids_before if i}
        added = [x for x in new_items if x.get("proposal_id") not in existing]
        stat["skipped_existing"] = [x.get("proposal_id") for x in new_items
                                    if x.get("proposal_id") in existing]
        stat["added"] = [x.get("proposal_id") for x in added]
        records.extend(added)
        stat["n_after"] = len(records)
        stat["ids_before"] = ids_before
        return data

    load_and_modify(_mod, p)

    back = load(p)
    _, _, records_after = _records(back)
    ids_after = [x.get("proposal_id") for x in records_after]
    stat["lost_ids"] = sorted(set(stat.get("ids_before") or []) - set(ids_after))
    stat["dup_ids"] = sorted({i for i in ids_after if i and ids_after.count(i) > 1})
    if stat["lost_ids"] or stat["dup_ids"]:
        raise RuntimeError(f"账本写入断言失败: lost={stat['lost_ids'][:5]} dup={stat['dup_ids'][:5]}")
    stat["eol"] = eol_name(p)
    return stat


if __name__ == "__main__":
    d = load()
    is_dict, _, recs = _records(d)
    print(f"OK, 形态={'dict' if is_dict else 'list'} 条目数={len(recs)} 行尾={eol_name()}")
