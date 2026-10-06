"""Optional LLM provider hook - DISABLED BY DEFAULT.

The assistant works fully offline with extractive answers. A provider can be
switched on purely through environment variables (never code or files):

    KB_ASSIST_LLM_ENABLED=true
    KB_ASSIST_LLM_ENDPOINT=https://<your-gateway>/v1/chat/completions
    KB_ASSIST_LLM_API_KEY=<from your secret store>
    KB_ASSIST_LLM_MODEL=<model name>

The bundled HTTP provider speaks the widely used "chat completions" JSON shape
so it can point at an approved internal AI gateway. Tests never enable it.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Mapping
from typing import Protocol


class LLMConfigError(ValueError):
    """Provider enabled but misconfigured."""


class LLMProvider(Protocol):
    name: str

    def generate(self, system: str, prompt: str) -> str:
        ...


class HttpChatProvider:
    name = "http-chat"

    def __init__(self, endpoint: str, api_key: str, model: str, timeout_s: float = 20.0) -> None:
        if not endpoint.startswith("https://"):
            raise LLMConfigError("KB_ASSIST_LLM_ENDPOINT must use https://")
        if not api_key:
            raise LLMConfigError("KB_ASSIST_LLM_API_KEY is required when the provider is enabled")
        self.endpoint, self._api_key, self.model, self.timeout_s = endpoint, api_key, model, timeout_s

    def __repr__(self) -> str:  # never expose the key in logs or tracebacks
        return f"HttpChatProvider(endpoint={self.endpoint!r}, model={self.model!r})"

    def generate(self, system: str, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=payload, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 (https enforced)
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


def provider_from_env(env: Mapping[str, str] | None = None) -> LLMProvider | None:
    """Return a provider only when explicitly enabled; otherwise None."""
    env = os.environ if env is None else env
    if env.get("KB_ASSIST_LLM_ENABLED", "false").strip().lower() != "true":
        return None
    return HttpChatProvider(
        endpoint=env.get("KB_ASSIST_LLM_ENDPOINT", ""),
        api_key=env.get("KB_ASSIST_LLM_API_KEY", ""),
        model=env.get("KB_ASSIST_LLM_MODEL", "default"),
    )
