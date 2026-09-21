from __future__ import annotations

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.model = model
        self._client = AsyncOpenAI(base_url=base_url.rstrip("/"), api_key=api_key)

    async def complete(
        self, messages: list[dict], **kwargs
    ) -> ChatCompletion:
        return await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
