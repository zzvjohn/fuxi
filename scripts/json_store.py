# -*- coding: utf-8 -*-
"""行尾保真的 JSON 存储层（2026-09-25 立项，P-20260925-011 / -012 的落地实现）。

存在理由
--------------------------------------------------------------------
本项目 2026-09-25 一天内连续暴露**三个同源缺陷**，全部由「文本模式写盘会静默转换行尾」
与「写入方路径 ≠ 校验方路径」引起：

  ① scripts/stage2_s5_chain.py::persist_to_pool 用 proposals_io.PATH 写、用模块常量
     FACTOR_POOL 回读 ⇒ 路径重定向场景下 verified=True 是**假阳性**，真池被测试污染。
  ② 同一函数的锁内备份用 write_text(..., newline='') 写成 LF，而因子池实为 CRLF
     ⇒ 备份不是字节可信备份，不能用于字节级回滚。
  ③ 账本 data/stage1_optimization_proposals.json 自 2026-09-24 起一贯为 LF
     （连续 ~10 次写入实测 bare_LF>0 / CRLF=0），而登记脚本用 open(tmp,'w')
     在 Windows 文本模式下把 '\n' 静默改写成 '\r\n' ⇒ 行尾被整体翻转，
     违反 AGENTS.md「行尾按文件实测保留；不要全局转换」。

本模块把这三件事收敛到一个地方：
  - `detect_eol()`     实测目标文件既有行尾，绝不假定
  - `dump_bytes()`     统一序列化（ensure_ascii=False, indent=2），行尾由调用方/实测决定
  - `atomic_write()`   tmp + os.replace 原子写 + **写后回读字节校验**；行尾保真
  - `file_lock()`      O_EXCL 跨进程锁 + 过期清理（沿用项目 2026-08-21 模式）

纪律
--------------------------------------------------------------------
  - 写入**必须**保真既有行尾；不存在时默认 LF（不引入 Windows 特有的 CRLF）。
  - 写后读回**必须**校验字节，而非仅 json.load 通过。
  - 备份**必须**是字节快照（write_bytes(read_bytes())），不得经过文本模式。
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, os.PathLike]

CRLF = b"\r\n"
LF = b"\n"
LOCK_STALE_SEC = 600
ACQUIRE_TIMEOUT = 60


# ---------------------------------------------------------------- 行尾
def detect_eol(path: PathLike, default: bytes = LF) -> bytes:
    """实测目标文件既有行尾。CRLF 占多数则判 CRLF，否则 LF。

    文件不存在 / 无换行 ⇒ default（默认 LF，避免在非 Windows 环境留下 CRLF）。
    """
    p = Path(path)
    if not p.exists():
        return default
    try:
        raw = p.read_bytes()
    except OSError:
        return default
    n_crlf = raw.count(CRLF)
    n_lf_total = raw.count(LF)
    n_bare_lf = n_lf_total - n_crlf
    if n_crlf == 0 and n_bare_lf == 0:
        return default
    return CRLF if n_crlf > n_bare_lf else LF


# ---------------------------------------------------------------- 序列化
def _json_safe(o):
    """numpy 标量 → 原生 Python（float32 JSON 铁律，与 s5_fullpool_rerun 同口径）。

    2026-09-28: stage2_s5_chain first-enforce 直写真实 S5 读数时遇 np.float32
    整链崩；本模块作为共享序列化通道，统一补 default 防护（纯增益：
    default 仅在原本会抛 TypeError 的输入上生效，正常路径字节不变）。
    """
    try:
        import numpy as _np
    except Exception:
        return str(o)
    if isinstance(o, (_np.floating, _np.integer)):
        return float(o)
    if isinstance(o, _np.bool_):
        return bool(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return str(o)


def dump_bytes(data: Any, eol: bytes = LF) -> bytes:
    """统一序列化：ensure_ascii=False + indent=2 + 指定行尾。

    与项目既有落盘风格一致（json.dumps(..., ensure_ascii=False, indent=2) 的字节）。
    """
    s = json.dumps(data, ensure_ascii=False, indent=2, default=_json_safe)
    b = s.encode("utf-8")
    if eol == CRLF:
        # '\n' 只可能作为行尾出现：UTF-8 续字节均 >= 0x80；字符串内的 CR/LF 已被 json 转义
        b = b.replace(LF, CRLF)
    return b


def load(path: PathLike) -> Any:
    return json.loads(Path(path).read_bytes().decode("utf-8"))


# ---------------------------------------------------------------- 锁
@contextmanager
def file_lock(path: PathLike, stale_sec: int = LOCK_STALE_SEC,
              timeout: int = ACQUIRE_TIMEOUT):
    """O_EXCL 锁文件 + 崩溃残留清理。与 scripts/proposals_io.py 行为一致。"""
    lock = Path(str(path) + ".lock")
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > stale_sec:
                    os.remove(lock)
                    continue
            except OSError:
                continue
            if time.time() - start > timeout:
                raise TimeoutError(f"{path} 被其他进程锁定超过 {timeout}s (锁: {lock})")
            time.sleep(1)
    try:
        yield lock
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


# ---------------------------------------------------------------- 原子写
def atomic_write(path: PathLike, data: Any, eol: bytes | None = None) -> dict:
    """原子写 JSON，**保真既有行尾**；写后回读字节校验。

    eol=None ⇒ 由 detect_eol(path) 实测决定（推荐）。
    返回 {eol, bytes, sha256, eol_preserved}，其中 eol_preserved 说明落盘行尾
    是否与写入前实测一致（同一次调用内必然一致；此字段供调用方留痕）。
    """
    p = Path(path)
    if eol is None:
        eol = detect_eol(p)
    before_eol = detect_eol(p) if p.exists() else None
    b = dump_bytes(data, eol)

    tmp = Path(str(p) + ".tmp")
    with open(tmp, "wb") as f:
        f.write(b)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)

    back = p.read_bytes()
    if back != b:
        raise RuntimeError(
            f"写后回读字节不一致: {p} 期望 {len(b)}B 实得 {len(back)}B")
    json.loads(back.decode("utf-8"))   # 解析可用性校验
    import hashlib
    return {
        "eol": "crlf" if eol == CRLF else "lf",
        "bytes": len(b),
        "sha256": hashlib.sha256(b).hexdigest(),
        "eol_preserved": (before_eol is None) or (before_eol == eol),
    }


def load_and_modify(path: PathLike, modifier, eol: bytes | None = None) -> Any:
    """锁内闭环：读最新 → modifier 修改 → 原子写（行尾保真）→ 读回校验。"""
    with file_lock(path):
        data = load(path)
        data = modifier(data)
        atomic_write(path, data, eol)
        return data


if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__)
    print(f"{p}: exists={p.exists()} eol={'crlf' if detect_eol(p) == CRLF else 'lf'}")
