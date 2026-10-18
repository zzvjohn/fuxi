# -*- coding: utf-8 -*-
"""
DeepSeek LLM API 客户端 — 伏羲系统统一 LLM 调用接口
====================================================
同时服务于 AlphaAgent 和 llm_generator，提供:
  - 同步单次 chat 调用
  - 流式输出 (streaming)
  - 自动重试 + 指数退避
  - 超时保护
"""

import os
import re
import time
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Dict


# ─── 默认配置 ───────────────────────────────────────────

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 120  # 秒
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # 初始退避秒数

# 项目根目录 (research/factor_alchemy/llm_client.py -> 上溯两级)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 本地凭据候选文件 (按优先级)。均为私密本地文件, 不写入版本库。
_CRED_FILES = (
    _PROJECT_ROOT / ".env",
    _PROJECT_ROOT / ".env.local",
    _PROJECT_ROOT / "config" / "secrets.local.json",
)

# 凭据文件里可接受的键名 (大小写与命名习惯不一, 逐个尝试)
_CRED_KEY_NAMES = ("DEEPSEEK_API_KEY", "deepseek_api_key", "deepseek")

# 兼容旧引用: 这是 **import 时** 的环境快照, 可能为 None。
# 真正取值请用 _resolve_api_key() —— 它支持延迟解析与多来源回退。
DEFAULT_API_KEY = os.environ.get("DEEPSEEK_API_KEY")


def _parse_env_file(path: Path) -> Dict[str, str]:
    """极简 .env 解析 (KEY=VALUE, 支持 # 注释 / export 前缀 / 引号), 不引入外部依赖。"""
    out: Dict[str, str] = {}
    try:
        if not path.is_file():
            return out
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v:
            out[k] = v
    return out


def _keys_from_secrets_json(path: Path) -> Dict[str, str]:
    """从 config/secrets.local.json 取字符串型键值。"""
    out: Dict[str, str] = {}
    try:
        if not path.is_file():
            return out
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return out
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str) and v.strip():
                out[k] = v.strip()
    return out


# ── P-20260917-027: 凭据名宽容解析 (归一化匹配) ──────────
# 用户环境里实测写的是 `DeepSeek APIKEY` (含空格, 大小写混排); 凭据文件里
# 习惯写 `deepseek_api_key`; 代码约定是 `DEEPSEEK_API_KEY`。三者归一化后
# 同为 `deepseekapikey` ⇒ 用归一化名匹配可一次覆盖全部写法, 且不会误伤
# `DEEPSEEK_MODEL` / `DEEPSEEK_BASE_URL` 这类变量 (归一化后不同名)。

def _norm_cred_name(name) -> str:
    """去掉非字母数字并小写, 用于凭据变量名的宽容匹配。"""
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


# 归一化后可接受的凭据名 (顺序 = 优先级)
_CRED_NORM_NAMES = (
    "deepseekapikey",
    "deepseekapikeys",
    "deepseekkey",
    "deepseekapikeysecret",
    "deepseek",
)


def _is_cred_name(name) -> bool:
    return _norm_cred_name(name) in _CRED_NORM_NAMES


def _pick_cred_from_keys(names) -> tuple:
    """从一组候选名里挑出凭据, 返回 (值, 原名); 无则 (None, "")。"""
    for _n in names:
        if _is_cred_name(_n):
            _v = str(os.environ.get(_n, "") or "").strip()
            if _v:
                return _v, str(_n)
    return None, ""


def _scan_credfile_tolerant(kv) -> tuple:
    """从凭据文件键值里宽容匹配, 返回 (值, 键名)。"""
    for _n in sorted(kv.keys()):
        if _is_cred_name(_n):
            _v = str(kv.get(_n) or "").strip()
            if _v:
                return _v, str(_n)
    return None, ""


def _scan_registry_tolerant() -> tuple:
    """枚举 HKCU\\Environment 全部值做归一化匹配 (绕开进程环境快照)。"""
    if os.name != "nt":
        return None, ""
    try:
        import winreg  # type: ignore
    except ImportError:
        return None, ""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            _, n_val, _ = winreg.QueryInfoKey(key)
            for i in range(n_val):
                name, val, _ = winreg.EnumValue(key, i)
                if _is_cred_name(name):
                    v = str(val or "").strip() if isinstance(val, str) else ""
                    if v:
                        return v, str(name)
    except OSError:
        return None, ""
    return None, ""


def _from_windows_user_env(name: str) -> Optional[str]:
    """直接读 Windows 用户环境变量注册表 (HKCU\\Environment)。

    进程环境是**启动时的快照**: 用户后来在「编辑账户的环境变量」里新增的变量,
    不重启当前应用就读不到 —— 这正是伏羲 LLM 调用在 Codex 可用、
    在其它宿主进程却报「缺少 DEEPSEEK_API_KEY」的原因。此处绕过快照直接读注册表。
    """
    if os.name != "nt":
        return None
    try:
        import winreg  # type: ignore
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return (value or "").strip() or None
    except OSError:
        return None


def resolve_api_key_with_source(explicit: Optional[str] = None):
    """按优先级解析 DeepSeek API Key, 返回 (key, source); 未配置时 (None, "missing")。

    优先级:
      1) 显式传入
      2) 进程环境变量 DEEPSEEK_API_KEY
      3) 项目本地凭据文件 .env / .env.local / config/secrets.local.json
      4) Windows 用户环境变量 (读注册表, 绕过进程快照)

    P-20260917-027: 2)~4) 每一级均在精确名之后追加**归一化名兜底**
    (去非字母数字 + 小写), 覆盖 `DeepSeek APIKEY` / `deepseek_api_key` /
    `DEEPSEEK_API_KEY` 等全部写法。
    """
    if explicit and explicit.strip():
        return explicit.strip(), "explicit"

    # (2) 进程环境变量 —— 精确名优先, 归一化名兜底 (P-20260917-027)
    env = os.environ.get("DEEPSEEK_API_KEY")
    if env and env.strip():
        return env.strip(), "process-env"
    _v, _n = _pick_cred_from_keys(sorted(os.environ.keys()))
    if _v:
        return _v, "process-env:%s" % _n

    # (3) 项目本地凭据文件 —— 精确名优先, 归一化名兜底
    for path in _CRED_FILES:
        kv = _keys_from_secrets_json(path) if path.suffix == ".json" else _parse_env_file(path)
        try:
            rel = path.relative_to(_PROJECT_ROOT)
        except ValueError:
            rel = path
        for name in _CRED_KEY_NAMES:
            val = kv.get(name)
            if val and val.strip():
                return val.strip(), "file:%s" % rel
        _v, _n = _scan_credfile_tolerant(kv)
        if _v:
            return _v, "file:%s:%s" % (rel, _n)

    # (4) Windows 用户环境变量 (读注册表, 绕过进程快照) —— 精确名优先, 归一化名兜底
    reg = _from_windows_user_env("DEEPSEEK_API_KEY")
    if reg:
        return reg, "windows-user-env(registry)"
    _v, _n = _scan_registry_tolerant()
    if _v:
        return _v, "windows-user-env(registry):%s" % _n

    return None, "missing"


def _resolve_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """仅取值的便捷包装。"""
    return resolve_api_key_with_source(explicit)[0]


class DeepSeekClient:
    """DeepSeek API 客户端 — 单例模式"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        api_key, source = resolve_api_key_with_source(api_key)
        if not api_key:
            raise RuntimeError(
                "缺少 DEEPSEEK_API_KEY。已尝试: 进程环境变量 → 本地凭据文件 "
                "(.env / .env.local / config/secrets.local.json) → Windows 用户环境变量(注册表)。"
                "请任选其一配置后重试。"
            )
        self.key_source = source
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._call_count = 0

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 4096,
        stream: bool = False,
        response_format: Optional[Dict] = None,
    ) -> str:
        """
        同步 chat completion 调用，带自动重试。

        Args:
            messages: OpenAI 格式的消息列表
            temperature: 采样温度 (0-2)
            max_tokens: 最大输出 token 数
            stream: 是否启用流式输出 (暂不支持)
            response_format: {"type": "json_object"} 启用 JSON 模式

        Returns:
            LLM 响应文本

        Raises:
            RuntimeError: 所有重试耗尽后仍然失败
        """
        url = f"{self.base_url}/chat/completions"

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format:
            body["response_format"] = response_format

        data = json.dumps(body).encode("utf-8")

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                req = urllib.request.Request(url, data=data, method="POST")
                req.add_header("Content-Type", "application/json")
                req.add_header("Authorization", f"Bearer {self.api_key}")
                req.add_header("Accept", "application/json")

                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    self._call_count += 1

                    # 提取 content
                    choices = result.get("choices", [])
                    if not choices:
                        raise RuntimeError(f"LLM 返回空 choices: {json.dumps(result, ensure_ascii=False)[:500]}")

                    content = choices[0].get("message", {}).get("content", "")
                    if not content:
                        raise RuntimeError(f"LLM 返回空 content: {json.dumps(choices[0], ensure_ascii=False)[:500]}")

                    return content

            except urllib.error.HTTPError as e:
                error_body = ""
                try:
                    error_body = e.read().decode("utf-8")[:500]
                except Exception:
                    pass
                last_error = f"HTTP {e.code}: {error_body}"

                # 4xx 错误不重试 (除了 429)
                if e.code == 429:
                    delay = RETRY_DELAY * (2 ** attempt)
                    print(f"  [LLM] 429 Rate Limit, 重试 {attempt+1}/{MAX_RETRIES} (等 {delay:.0f}s)...")
                    time.sleep(delay)
                elif 400 <= e.code < 500 and e.code != 429:
                    raise RuntimeError(f"LLM 请求错误 (不可重试): {last_error}")
                else:
                    delay = RETRY_DELAY * (2 ** attempt)
                    print(f"  [LLM] {last_error}, 重试 {attempt+1}/{MAX_RETRIES} (等 {delay:.0f}s)...")
                    time.sleep(delay)

            except urllib.error.URLError as e:
                last_error = str(e)
                delay = RETRY_DELAY * (2 ** attempt)
                print(f"  [LLM] 连接错误: {e}, 重试 {attempt+1}/{MAX_RETRIES} (等 {delay:.0f}s)...")
                time.sleep(delay)

            except Exception as e:
                last_error = str(e)
                delay = RETRY_DELAY * (2 ** attempt)
                print(f"  [LLM] 未知错误: {e}, 重试 {attempt+1}/{MAX_RETRIES} (等 {delay:.0f}s)...")
                time.sleep(delay)

        raise RuntimeError(f"LLM 调用失败 ({MAX_RETRIES} 次重试后): {last_error}")

    def chat_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> str:
        """便捷方法: 系统提示 + 用户提示"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, response_format=response_format)

    @property
    def call_count(self) -> int:
        return self._call_count


# ─── 全局单例 ──────────────────────────────────────────

_global_client: Optional[DeepSeekClient] = None


def get_llm_client(
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> DeepSeekClient:
    """获取全局 LLM 客户端单例"""
    global _global_client
    if _global_client is None:
        _global_client = DeepSeekClient(api_key=api_key, base_url=base_url, model=model)
    return _global_client


def reset_llm_client():
    """重置全局客户端 (用于测试/配置变更)"""
    global _global_client
    _global_client = None


# ─── 凭据诊断 (自检用, 均不泄露 Key 内容) ─────────────────

def describe_api_key(api_key: Optional[str] = None) -> Dict[str, object]:
    """脱敏诊断: 只回报 来源 / 长度 / 前 6 位, 绝不回显完整 Key。"""
    key, source = resolve_api_key_with_source(api_key)
    if not key:
        return {"present": False, "source": "missing", "length": 0, "prefix": ""}
    return {
        "present": True,
        "source": source,
        "length": len(key),
        "prefix": key[:6] + "..." if len(key) > 6 else "(short)",
    }


def probe_auth(api_key: Optional[str] = None, timeout: int = 20) -> Dict[str, object]:
    """仅认证探针: 调用 /models 端点验证凭据, **不生成内容、不消耗推理额度**。

    返回: {"ok": bool, "models": int, "source": str, "error": str}
    """
    key, source = resolve_api_key_with_source(api_key)
    if not key:
        return {"ok": False, "models": 0, "source": "missing",
                "error": "未找到 DEEPSEEK_API_KEY"}
    url = "%s/models" % DEFAULT_BASE_URL.rstrip("/")
    req = urllib.request.Request(url, headers={
        "Authorization": "Bearer %s" % key,
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        models = payload.get("data") if isinstance(payload, dict) else None
        return {
            "ok": True,
            "models": len(models) if isinstance(models, list) else 0,
            "source": source,
            "error": "",
        }
    except urllib.error.HTTPError as e:
        return {"ok": False, "models": 0, "source": source, "error": "HTTP %s" % e.code}
    except Exception as e:  # noqa: BLE001 - 探针不抛异常, 只回报
        return {"ok": False, "models": 0, "source": source, "error": type(e).__name__}
