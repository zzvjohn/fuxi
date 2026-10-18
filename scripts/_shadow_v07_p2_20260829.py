# -*- coding: utf-8 -*-
"""[S-5 归档转发] 实体已移至 scripts/once/_shadow_v07_p2_20260829.py (2026-09-18)。

本文件仅为**保留旧路径可执行性**的转发桩, 不是脚本本体。
以 __file__=<旧路径> 执行归档副本, 使脚本内 dirname(abspath(__file__))
仍解析到 scripts/ (语义与搬移前一致)。改脚本请改 scripts/once/_shadow_v07_p2_20260829.py,
或先 `python scripts/once/_restore.py --name _shadow_v07_p2_20260829.py` 还原。
"""
import os as _os

_S5_OLD = _os.path.abspath(__file__)
_S5_NEW = _os.path.join(_os.path.dirname(_S5_OLD), "once", "_shadow_v07_p2_20260829.py")
with open(_S5_NEW, "rb") as _f:
    _code = compile(_f.read(), _S5_OLD, "exec")
exec(_code, {
    "__name__": "__main__", "__file__": _S5_OLD,
    "__builtins__": __builtins__, "__doc__": None,
    "__package__": None, "__spec__": None,
})
