# -*- coding: utf-8 -*-
"""DeepSeek 凭据自检与落盘工具 (伏羲 LLM 调用前置检查)

用法:
    python scripts/check_llm_credentials.py            # 诊断: 来源 / 长度 / 认证探针
    python scripts/check_llm_credentials.py --probe    # 只做认证探针 (不打印值)
    python scripts/check_llm_credentials.py --set      # 交互式写入 .env 并立即验证
    python scripts/check_llm_credentials.py --set --from-env   # 从当前进程环境变量落盘

设计原则:
  * 任何路径都**不打印完整 Key** (只报来源 / 长度 / 前 6 位)
  * 认证探针只访问 /models 列表端点, 不生成内容、不消耗推理额度
  * 落盘写入项目 .env (本地私密文件), 不写入任何版本库文件
"""
import argparse
import getpass
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "factor_alchemy"))

try:
    from llm_client import (  # noqa: E402
        describe_api_key,
        probe_auth,
        resolve_api_key_with_source,
        _CRED_FILES,
        _parse_env_file,
    )
except Exception as exc:  # noqa: BLE001
    print("[FATAL] 无法导入 llm_client: %s" % exc)
    sys.exit(2)

ENV_PATH = PROJECT_ROOT / ".env"
KEY_NAME = "DEEPSEEK_API_KEY"


def _banner(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def cmd_diagnose(do_probe=True):
    _banner("1. 解析链状态 (进程环境 → 本地凭据文件 → Windows 用户注册表)")
    for p in _CRED_FILES:
        n = len(_parse_env_file(p)) if p.suffix != ".json" else -1
        mark = "EXISTS" if p.is_file() else "absent"
        extra = ("keys=%d" % n) if n >= 0 else "json"
        try:
            shown = p.relative_to(PROJECT_ROOT)
        except ValueError:
            shown = p
        print("   %-38s %-7s %s" % (shown, mark, extra))

    proc = os.environ.get(KEY_NAME)
    print("   进程环境变量 %-26s %s" % (KEY_NAME, "PRESENT len=%d" % len(proc) if proc else "MISSING"))

    _banner("2. 实际解析结果 (脱敏)")
    d = describe_api_key()
    if d["present"]:
        print("   ✅ 已找到凭据")
        print("      来源   : %s" % d["source"])
        print("      长度   : %s" % d["length"])
        print("      前缀   : %s" % d["prefix"])
    else:
        print("   ❌ 未找到 DEEPSEEK_API_KEY")
        print("      已尝试: 进程环境变量 → .env / .env.local / config/secrets.local.json")
        print("              → Windows 用户环境变量 (HKEY_CURRENT_USER\\Environment)")
        print("")
        print("   ⚠️  常见坑: 在「编辑账户的环境变量」里新增变量后,")
        print("       已启动的进程不会继承 —— 必须重启该应用;")
        print("       且该对话框若经 UAC 提升打开, 写入的是**管理员账户**,")
        print("       当前普通用户账户读不到。")
        print("")
        print("   👉 最稳做法 (脱离进程继承): 把下面一行加进 %s" % ENV_PATH)
        print("        %s=sk-你的key" % KEY_NAME)
        print("      或运行: python scripts/check_llm_credentials.py --set")

    if do_probe and d["present"]:
        _banner("3. 认证探针 (GET /models, 不生成内容 / 不计费推理)")
        r = probe_auth()
        if r["ok"]:
            print("   ✅ 认证通过, 可用模型数 = %s" % r["models"])
        else:
            print("   ❌ 认证失败: %s" % r["error"])
            print("      (若为 HTTP 401/403 ⇒ key 无效或已轮换)")
        return 0 if r["ok"] else 4

    if do_probe and not d["present"]:
        _banner("3. 认证探针")
        print("   跳过 (无凭据)")
        return 3
    return 0


def cmd_set(from_env=False):
    _banner("写入 DeepSeek Key 到 %s" % ENV_PATH)
    if from_env:
        val = os.environ.get(KEY_NAME)
        if not val:
            print("   ❌ 当前进程环境变量里没有 %s" % KEY_NAME)
            return 3
    else:
        print("   提示: 输入不回显。直接回车取消。")
        val = getpass.getpass("   粘贴 DEEPSEEK_API_KEY: ").strip()
        if not val:
            print("   已取消。")
            return 0

    if not val.startswith("sk-") or len(val) < 20:
        print("   ⚠️  形态可疑 (期望 sk- 开头且长度>=20), 长度=%d。仍将写入。" % len(val))

    try:
        existing = ENV_PATH.read_text(encoding="utf-8", errors="replace") if ENV_PATH.is_file() else ""
    except OSError:
        existing = ""

    lines = [l for l in existing.splitlines() if not l.strip().startswith(KEY_NAME + "=")]
    lines.append("%s=%s" % (KEY_NAME, val))
    # 保留原文件行尾风格: 若原本用 CRLF 则继续用 CRLF
    nl = "\r\n" if "\r\n" in existing else "\n"
    ENV_PATH.write_text(nl.join(lines) + nl, encoding="utf-8", newline="")
    print("   ✅ 已写入 %s (长度 %d)" % (ENV_PATH, len(val)))

    # 立即重载验证 (当前进程环境若已有旧值会优先, 故同时直接验证文件值)
    _banner("立即验证")
    import importlib
    import llm_client as lc
    importlib.reload(lc)
    d = lc.describe_api_key()
    print("   来源: %s | 长度: %s" % (d["source"], d["length"]))
    r = lc.probe_auth()
    if r["ok"]:
        print("   ✅ 认证通过, 可用模型数 = %s" % r["models"])
        return 0
    print("   ❌ 认证失败: %s" % r["error"])
    return 4


def main():
    ap = argparse.ArgumentParser(description="DeepSeek 凭据自检 / 落盘")
    ap.add_argument("--probe", action="store_true", help="只做认证探针")
    ap.add_argument("--set", action="store_true", help="交互式写入 .env")
    ap.add_argument("--from-env", action="store_true", help="配合 --set: 从进程环境变量落盘")
    args = ap.parse_args()

    if args.set:
        return cmd_set(from_env=args.from_env)
    return cmd_diagnose(do_probe=True)


if __name__ == "__main__":
    sys.exit(main())
