"""P-20260915-001: 风格捷径阴影诊断 — 独立运行入口 + 核心 API 转发层。

核心实现位于 research/factor_alchemy/style_exposure_guard.py

🔴 同名模块坑 (2026-09-12 集成测试暴露, 必读):
   本文件与核心模块**同名** `style_exposure_guard`。run_v4_pipeline 会把
   e:/quant/scripts 插入 sys.path[0], 于是 ralph_loop 里的
   `from style_exposure_guard import attach_computer, evaluate` 可能解析到**本文件**;
   若本文件又自引用同名模块 → 循环导入
   (cannot import name 'X' from partially initialized module)。
   修复: importlib 按**绝对路径**加载核心模块 (注册独立模块名), 再把公开 API 绑定到
   本模块命名空间。这样无论 sys.path 顺序如何, 公开 API 都可用, 且 _CTX 缓存只在
   核心模块一处。

🔴 用途边界: **仅归因诊断**, 禁止把输出/残差用于因子中性化处理 (V5 路线已证伪)。

用法:
  python scripts/style_exposure_guard.py --replay            # 历史回放验证
  python scripts/style_exposure_guard.py --formulas-file X   # 指定因子清单
开关: env FUXI_STYLE_GUARD = on|true|1|shadow(默认, 生效) | off|0|false
"""
import importlib.util
import sys
from pathlib import Path

_CORE = (Path(__file__).resolve().parent.parent
         / "research" / "factor_alchemy" / "style_exposure_guard.py")
_ALCHEMY_DIR = _CORE.parent

# 核心模块内部会 import 同目录兄弟模块 (factor_ic_computer 等), 需其目录在 path 上
if str(_ALCHEMY_DIR) not in sys.path:
    sys.path.insert(0, str(_ALCHEMY_DIR))
if str(_ALCHEMY_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(_ALCHEMY_DIR.parent.parent))

_NAME = "_style_exposure_guard_core"
if _NAME in sys.modules:
    _mod = sys.modules[_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_NAME, _CORE)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _mod
    _spec.loader.exec_module(_mod)

# ── 公开 API 转发 (保证同名解析歧义下 ralph_loop 的 import 仍可用) ──
get_mode = _mod.get_mode
attach_computer = _mod.attach_computer
factor_check = _mod.factor_check
evaluate = _mod.evaluate
FORBIDDEN_USE = _mod.FORBIDDEN_USE
_cli = _mod._cli

if __name__ == "__main__":
    sys.exit(_cli())
