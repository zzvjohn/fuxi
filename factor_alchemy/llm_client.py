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
import time
import json
import urllib.request
import urllib.error
import os
from typing import Optional, List, Dict


# ─── 默认配置 ───────────────────────────────────────────

# 统一 DeepSeek API Key (2026-08-31 用户指定):
# 仅从环境变量 DEEPSEEK_API_KEY 读取; 未设置时为空串, 须配置后方可调用
DEFAULT_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")  # 通过环境变量注入, 禁止硬编码兜底
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 120  # 秒
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # 初始退避秒数


class DeepSeekClient:
    """DeepSeek API 客户端 — 单例模式"""

    def __init__(
        self,
        api_key: str = DEFAULT_API_KEY,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
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
    api_key: str = DEFAULT_API_KEY,
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
