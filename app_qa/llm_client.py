"""LLM 客户端：OpenAI 兼容协议，未配置时静默降级。

环境变量：
    OPENAI_API_KEY   (必需)
    OPENAI_BASE_URL  (可选，默认 OpenAI 官方；可填 DeepSeek/Kimi/智谱等)
    OPENAI_MODEL     (可选，默认 gpt-3.5-turbo)

接口：
    client = LLMClient()         # 自动读 env
    client.available             # True 表示可调用
    client.complete(prompt) -> str
"""

from __future__ import annotations

import os


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or ""
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        self.timeout = timeout
        self._client = None
        self._last_error: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _ensure(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI

            kwargs: dict = {"timeout": self.timeout}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
            return self._client
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"OpenAI 客户端初始化失败：{exc}"
            return None

    def complete(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 800) -> str:
        if not self.available:
            return ""
        client = self._ensure()
        if client is None:
            return ""
        messages = [{"role": "user", "content": prompt}]
        # 新版接口 (o1/o3 等) 用 max_completion_tokens 替代 max_tokens；
        # 旧接口 (gpt-3.5/4 / DeepSeek / Kimi) 仍用 max_tokens。
        # 先按旧参数调用，遇到参数错误再回退到新参数。
        kwargs_old: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = client.chat.completions.create(**kwargs_old)
            return resp.choices[0].message.content or ""
        except TypeError as exc:
            self._last_error = f"LLM 参数不兼容(旧版)：{exc}"
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            self._last_error = f"LLM 调用失败：{exc}"
            # 仅当报错明确指向 max_tokens 不支持时才尝试新参数
            if "max_tokens" not in msg and "max_completion_tokens" not in msg:
                return ""

        try:
            kwargs_new: dict = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            }
            resp = client.chat.completions.create(**kwargs_new)
            self._last_error = None
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"LLM 调用失败(回退后)：{exc}"
            return ""

    @property
    def last_error(self) -> str | None:
        return self._last_error
