from __future__ import annotations

import json
from typing import Dict, List
from urllib import error, request


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.1,
    ):
        api_key = (api_key or "").strip()
        if not api_key:
            raise ValueError("Missing DeepSeek API key.")

        self.api_key = api_key
        self.api_base = api_base or "https://api.deepseek.com"
        self.model = model
        try:
            self.temperature = float(temperature)
        except (TypeError, ValueError):
            self.temperature = 0.1

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"DeepSeek API error {exc.code}: {err_body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"DeepSeek API request failed: {exc.reason}") from exc

        payload = json.loads(body)
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek API response missing choices.") from exc
