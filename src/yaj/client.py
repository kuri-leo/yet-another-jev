from __future__ import annotations

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.model = model
        normalized_url = (
            base_url
            if base_url.rstrip("/").endswith("/v1")
            else f"{base_url.rstrip('/')}/v1"
        )
        self._client = AsyncOpenAI(base_url=normalized_url, api_key=api_key)

    async def complete(
        self, messages: list[dict], **kwargs
    ) -> ChatCompletion:
        return await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
