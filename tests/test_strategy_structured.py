import json
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from yaj.client import LLMClient
from yaj.models import ChoiceAnswer, NoulAnswer, Question, QuestionType, ScoreAnswer
from yaj.prompt import build_messages
from yaj.strategy.structured import StructuredOutputError, StructuredStrategy


def _make_structured_app(answer_value, raw_content=None):
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        content = raw_content if raw_content is not None else json.dumps({"answer": answer_value})
        return JSONResponse({
            "id": "test",
            "object": "chat.completion",
            "model": body.get("model", "test"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    return app


@pytest.mark.asyncio
async def test_noul_returns_binary():
    app = _make_structured_app(True)
    strategy = StructuredStrategy()
    q = Question(
        type=QuestionType.NOUL,
        instructions="Is urgent?",
        criteria={"true": "yes", "false": "no"},
    )
    msgs = build_messages("test", q)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        client = LLMClient(base_url="http://test", api_key="fake", model="test")
        client._client._client = http
        answer = await strategy.decide(client, msgs, q)
    assert isinstance(answer, NoulAnswer)
    assert answer.noul == 1.0


@pytest.mark.asyncio
async def test_noul_returns_zero_on_false():
    app = _make_structured_app(False)
    strategy = StructuredStrategy()
    q = Question(
        type=QuestionType.NOUL,
        instructions="Is urgent?",
        criteria={"true": "yes", "false": "no"},
    )
    msgs = build_messages("test", q)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        client = LLMClient(base_url="http://test", api_key="fake", model="test")
        client._client._client = http
        answer = await strategy.decide(client, msgs, q)
    assert isinstance(answer, NoulAnswer)
    assert answer.noul == 0.0


@pytest.mark.asyncio
async def test_choice_returns_key():
    app = _make_structured_app("billing")
    strategy = StructuredStrategy()
    q = Question(
        type=QuestionType.CHOICE,
        instructions="Which?",
        criteria={"billing": "pay", "technical": "bug"},
    )
    msgs = build_messages("test", q)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        client = LLMClient(base_url="http://test", api_key="fake", model="test")
        client._client._client = http
        answer = await strategy.decide(client, msgs, q)
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "billing"
    assert answer.probabilities is None
    assert answer.confidence is None


@pytest.mark.asyncio
async def test_score_returns_integer():
    app = _make_structured_app(2)
    strategy = StructuredStrategy()
    q = Question(
        type=QuestionType.SCORE,
        instructions="Rate",
        criteria=["Low", "Normal", "High"],
    )
    msgs = build_messages("test", q)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        client = LLMClient(base_url="http://test", api_key="fake", model="test")
        client._client._client = http
        answer = await strategy.decide(client, msgs, q)
    assert isinstance(answer, ScoreAnswer)
    assert answer.score == 2.0
    assert answer.probabilities is None
    assert answer.confidence is None


@pytest.mark.asyncio
async def test_calls_llm_with_json_object_and_temp_zero():
    captured_requests = []
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        captured_requests.append(body)
        return JSONResponse({
            "id": "test",
            "object": "chat.completion",
            "model": body.get("model", "test"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps({"answer": "billing"})},
                "finish_reason": "stop",
                "logprobs": None,
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    strategy = StructuredStrategy()
    q = Question(
        type=QuestionType.CHOICE,
        instructions="Which?",
        criteria={"billing": "pay", "technical": "bug"},
    )
    msgs = [{"role": "user", "content": "Original message"}]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        client = LLMClient(base_url="http://test", api_key="fake", model="test")
        client._client._client = http
        await strategy.decide(client, msgs, q)

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req["response_format"] == {"type": "json_object"}
    assert req["temperature"] == 0
    # Original messages must not be mutated
    assert msgs[0]["content"] == "Original message"
    # Sent messages has hint appended
    assert "Original message\n\nRespond with JSON:" in req["messages"][-1]["content"]


def test_json_hints():
    strategy = StructuredStrategy()
    noul_q = Question(
        type=QuestionType.NOUL,
        instructions="Urgent?",
        criteria={"true": "yes", "false": "no"},
    )
    assert strategy._json_hint(noul_q) == 'Respond with JSON: {"answer": true} or {"answer": false}'

    choice_q = Question(
        type=QuestionType.CHOICE,
        instructions="Which?",
        criteria={"alpha": "A", "beta": "B"},
    )
    assert strategy._json_hint(choice_q) == 'Respond with JSON: {"answer": <one of "alpha", "beta">}'

    score_q = Question(
        type=QuestionType.SCORE,
        instructions="Rate",
        criteria=["Low", "Med", "High"],
    )
    assert strategy._json_hint(score_q) == 'Respond with JSON: {"answer": <integer 0 to 2>}'


@pytest.mark.asyncio
async def test_invalid_json_raises_structured_error():
    app = _make_structured_app(None, raw_content="not valid json")
    strategy = StructuredStrategy()
    q = Question(
        type=QuestionType.NOUL,
        instructions="Is urgent?",
        criteria={"true": "yes", "false": "no"},
    )
    msgs = build_messages("test", q)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        client = LLMClient(base_url="http://test", api_key="fake", model="test")
        client._client._client = http
        with pytest.raises(StructuredOutputError, match="Failed to parse JSON"):
            await strategy.decide(client, msgs, q)


@pytest.mark.asyncio
async def test_missing_answer_field_raises_error():
    app = _make_structured_app(None, raw_content=json.dumps({"wrong": 123}))
    strategy = StructuredStrategy()
    q = Question(
        type=QuestionType.NOUL,
        instructions="Is urgent?",
        criteria={"true": "yes", "false": "no"},
    )
    msgs = build_messages("test", q)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        client = LLMClient(base_url="http://test", api_key="fake", model="test")
        client._client._client = http
        with pytest.raises(StructuredOutputError, match="missing 'answer' key"):
            await strategy.decide(client, msgs, q)


@pytest.mark.asyncio
async def test_invalid_score_format_raises_error():
    app = _make_structured_app("not-a-number")
    strategy = StructuredStrategy()
    q = Question(
        type=QuestionType.SCORE,
        instructions="Rate",
        criteria=["Low", "Med", "High"],
    )
    msgs = build_messages("test", q)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        client = LLMClient(base_url="http://test", api_key="fake", model="test")
        client._client._client = http
        with pytest.raises(StructuredOutputError, match="Cannot parse score"):
            await strategy.decide(client, msgs, q)
