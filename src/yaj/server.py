from __future__ import annotations

import asyncio
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from yaj.client import LLMClient
from yaj.models import DecisionRequest, DecisionResponse, QuestionType, Usage
from yaj.prompt import build_messages
from yaj.strategy.base import Strategy
from yaj.strategy.logprob import UnsupportedError


class _TrackingClient(LLMClient):
    """Wraps an LLMClient to accumulate usage across calls."""

    def __init__(self, inner: LLMClient) -> None:
        # Skip LLMClient.__init__; reuse the inner client's internals
        self.model = inner.model
        self._client = inner._client
        self._input_tokens = 0
        self._output_tokens = 0
        self._lock = asyncio.Lock()

    async def complete(self, messages: list[dict], **kwargs):
        response = await super().complete(messages, **kwargs)
        if response.usage:
            async with self._lock:
                self._input_tokens += response.usage.prompt_tokens or 0
                self._output_tokens += response.usage.completion_tokens or 0
        return response

    @property
    def usage(self) -> Usage:
        return Usage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )


def _validate_questions(request: DecisionRequest) -> str | None:
    """Return an error detail string if any question fails validation, else None."""
    for q in request.questions.values():
        if q.type == QuestionType.SCORE:
            assert isinstance(q.criteria, list)
            if not (2 <= len(q.criteria) <= 10):
                return "score criteria must have 2-10 levels"
    return None


def create_app(strategy: Strategy, client: LLMClient) -> FastAPI:
    app = FastAPI(title="yet-another-jev")

    @app.post("/api/alpha/decisions")
    async def decisions(request: DecisionRequest) -> DecisionResponse:
        # Validate questions before dispatching
        validation_error = _validate_questions(request)
        if validation_error:
            return JSONResponse(
                status_code=422,
                content={"detail": validation_error},
            )

        tracker = _TrackingClient(client)

        async def _solve(key: str):
            q = request.questions[key]
            msgs = build_messages(request.state, q)
            try:
                answer = await strategy.decide(tracker, msgs, q)
            except Exception as exc:
                raise RuntimeError(f"question '{key}': {exc}") from exc
            return key, answer

        t0 = time.monotonic()
        try:
            tasks = [_solve(k) for k in request.questions]
            results = await asyncio.gather(*tasks)
        except RuntimeError as exc:
            cause = exc.__cause__
            if isinstance(cause, UnsupportedError):
                msg = str(cause)
                if "choice" in msg.lower():
                    detail = "logprob strategy supports at most 20 choice options"
                elif "score" in msg.lower():
                    detail = "score criteria must have 2-10 levels"
                else:
                    detail = msg
                return JSONResponse(
                    status_code=422,
                    content={"detail": detail},
                )
            # All other causes → 502 with question key context
            return JSONResponse(
                status_code=502,
                content={"detail": f"upstream LLM error: {exc}"},
            )
        duration_ms = round((time.monotonic() - t0) * 1000, 1)

        answers = dict(results)
        usage = tracker.usage
        usage.duration_ms = duration_ms
        return DecisionResponse(
            model=request.model,
            answers=answers,
            usage=usage,
        )

    return app
