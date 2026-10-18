"""P-20260912-002: 幸存者偏差守卫 — 独立运行入口 + 核心 API 转发层。

核心实现位于 research/factor_alchemy/survivorship_guard.py
(与 exec_trap_guard / causal_closure_check 同处, 便于 ralph_loop 直接 import)。

⚠️ 2026-09-12 集成测试暴露的坑 (必读):
   本文件与核心模块**同名** `survivorship_guard`。run_v4_pipeline 会把
   e:/quant/scripts 插入 sys.path[0], 于是 ralph_loop 里的
   `from survivorship_guard import attach_computer, evaluate` 会解析到**本文件**;
   而本文件原先又 `from survivorship_guard import _cli` 自引用 → 循环导入:
     cannot import name '_cli' from partially initialized module 'survivorship_guard'
   修复方式: importlib 按**绝对路径**加载核心模块 (独立模块名注册), 再把公开 API
   绑定到本模块命名空间。这样无论 sys.path 顺序如何、无论解析到哪个文件,
   `attach_computer / evaluate / ...` 都可用, 且 `_CTX` 缓存只在核心模块一处。

用法:
  python scripts/survivorship_guard.py --replay           # 历史回放验证 (非平凡读数自检)
  python scripts/survivorship_guard.py --formulas-file X  # 指定因子清单诊断
开关: env FUXI_SURVIVORSHIP_GUARD = shadow(默认) | off
"""
import importlib.util
import sys
from pathlib import Path

_CORE = (Path(__file__).resolve().parent.parent
         / "research" / "factor_alchemy" / "survivorship_guard.py")
_ALCHEMY_DIR = _CORE.parent

# 核心模块内部会 import 同目录兄弟模块 (factor_ic_computer 等), 需其目录在 path 上
if str(_ALCHEMY_DIR) not in sys.path:
    sys.path.insert(0, str(_ALCHEMY_DIR))
if str(_ALCHEMY_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(_ALCHEMY_DIR.parent.parent))

_NAME = "_survivorship_guard_core"
if _NAME in sys.modules:
    _mod = sys.modules[_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_NAME, _CORE)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _mod
    _spec.loader.exec_module(_mod)

# ── 公开 API 转发 (保证同名解析歧义下 ralph_loop 的 import 仍可用) ──
get_mode = _mod.get_mode
panel_audit = _mod.panel_audit
factor_check = _mod.factor_check
attach_computer = _mod.attach_computer
evaluate = _mod.evaluate
_cli = _mod._cli

if __name__ == "__main__":
    sys.exit(_cli())
