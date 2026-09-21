from __future__ import annotations

import json

from yaj.client import LLMClient
from yaj.models import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    QuestionType,
    ScoreAnswer,
)
from yaj.strategy.base import Strategy


class StructuredOutputError(ValueError):
    """Raised when structured output parsing fails."""


class StructuredStrategy(Strategy):
    async def decide(
        self, client: LLMClient | None, messages: list[dict], question: Question
    ) -> Answer:
        assert client is not None
        json_hint = self._json_hint(question)
        augmented = list(messages)
        if augmented:
            last = {
                **augmented[-1],
                "content": augmented[-1]["content"] + "\n\n" + json_hint,
            }
            augmented[-1] = last
        else:
            augmented = [{"role": "user", "content": json_hint}]

        response = await client.complete(
            augmented,
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise StructuredOutputError("LLM response content is empty")

        try:
            data = json.loads(content)
        except json.JSONDecodeError as err:
            raise StructuredOutputError(
                f"Failed to parse JSON response: {content!r}"
            ) from err

        if not isinstance(data, dict) or "answer" not in data:
            raise StructuredOutputError(
                f"Expected JSON object with missing 'answer' key: {content!r}"
            )

        value = data["answer"]

        if question.type == QuestionType.NOUL:
            if isinstance(value, str):
                noul_val = 1.0 if value.strip().lower() in ("true", "1", "yes") else 0.0
            else:
                noul_val = 1.0 if value else 0.0
            return NoulAnswer(noul=noul_val)
        elif question.type == QuestionType.CHOICE:
            return ChoiceAnswer(choice=str(value))
        else:
            try:
                score_val = float(value)
            except (ValueError, TypeError) as err:
                raise StructuredOutputError(
                    f"Cannot parse score as float: {value!r}"
                ) from err
            return ScoreAnswer(score=score_val)

    def _json_hint(self, question: Question) -> str:
        if question.type == QuestionType.NOUL:
            return 'Respond with JSON: {"answer": true} or {"answer": false}'
        elif question.type == QuestionType.CHOICE:
            assert isinstance(question.criteria, dict)
            keys = list(question.criteria.keys())
            opts = ", ".join(f'"{k}"' for k in keys)
            return f'Respond with JSON: {{"answer": <one of {opts}>}}'
        else:
            assert isinstance(question.criteria, list)
            n = len(question.criteria)
            return f'Respond with JSON: {{"answer": <integer 0 to {n-1}>}}'
