from __future__ import annotations

import abc

from yaj.client import LLMClient
from yaj.models import Answer, Question


class Strategy(abc.ABC):
    @abc.abstractmethod
    async def decide(
        self, client: LLMClient | None, messages: list[dict], question: Question
    ) -> Answer: ...
