"""端点抽象：统一 HTTP（OpenAI 兼容）与进程内仿真两种后端。

核心库只依赖抽象基类 Endpoint，因此单元测试与红队评估可以完全离线、确定性地跑，
也能一键切换到真实中转站。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Answer:
    text: str
    model_field: str = ""     # 端点自报的 model（可伪造，仅参考）
    latency: float = 0.0
    cached: bool = False       # 若元数据显示 prompt-cache 命中
    error: str | None = None


class Endpoint(Protocol):
    def ask(self, system: str, user: str) -> Answer: ...


class HTTPEndpoint:
    """OpenAI 兼容的 /chat/completions 端点。"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float = 1.0, max_tokens: int = 16, timeout: float = 60.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            import httpx
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def ask(self, system: str, user: str) -> Answer:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        client = self._client_lazy()
        last_err = "unknown"
        for attempt in range(3):
            t0 = time.time()
            try:
                r = client.post(self.url, json=payload, headers=headers)
                dt = time.time() - t0
                if r.status_code == 429 or r.status_code >= 500:
                    last_err = f"HTTP {r.status_code}"
                    time.sleep(1.0 * (attempt + 1))
                    continue
                if r.status_code >= 400:
                    return Answer(text="", error=f"HTTP {r.status_code}: {r.text[:160]}", latency=dt)
                j = r.json()
                content = (j.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                usage = j.get("usage", {}) or {}
                cached = bool(usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)) \
                    if isinstance(usage.get("prompt_tokens_details"), dict) else False
                return Answer(text=content, model_field=j.get("model", ""), latency=dt, cached=cached)
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(1.0 * (attempt + 1))
        return Answer(text="", error=last_err)

    def close(self):
        if self._client is not None:
            self._client.close()


class CallableEndpoint:
    """把任意 callable(system, user) -> Answer 包装成 Endpoint（用于仿真/测试）。"""

    def __init__(self, fn):
        self._fn = fn

    def ask(self, system: str, user: str) -> Answer:
        return self._fn(system, user)
