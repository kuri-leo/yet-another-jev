import math
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from yaj.client import LLMClient
from yaj.models import ChoiceAnswer, NoulAnswer, Question, QuestionType, ScoreAnswer
from yaj.prompt import build_messages
from yaj.strategy import LogprobExtractionError, LogprobStrategy, Strategy, UnsupportedError


def _make_logprob_app(token: str, prob: float, alternatives: dict[str, float]):
    """Create a dummy app returning specific logprobs."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        top = [{"token": t, "logprob": math.log(p)} for t, p in alternatives.items()]
        return JSONResponse({
            "id": "test",
            "object": "chat.completion",
            "model": body.get("model", "test"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": token},
                "finish_reason": "stop",
                "logprobs": {
                    "content": [{
                        "token": token,
                        "logprob": math.log(prob),
                        "top_logprobs": top,
                    }]
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        })

    return app


@pytest.mark.asyncio
async def test_noul_extracts_probability():
    app = _make_logprob_app("Yes", 0.9, {"Yes": 0.9, "No": 0.1})
    strategy = LogprobStrategy()
    q = Question(
        type=QuestionType.NOUL,
        instructions="Is urgent?",
        criteria={"true": "yes", "false": "no"},
    )
    msgs = build_messages("test state", q)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test/v1") as http:
        client = LLMClient(base_url="http://test/v1", api_key="fake", model="test")
        client._client._client = http
        answer = await strategy.decide(client, msgs, q)
    assert isinstance(answer, NoulAnswer)
    assert 0.85 < answer.noul < 0.95


@pytest.mark.asyncio
async def test_choice_returns_argmax():
    app = _make_logprob_app("A", 0.7, {"A": 0.7, "B": 0.2, "C": 0.1})
    strategy = LogprobStrategy()
    q = Question(
        type=QuestionType.CHOICE,
        instructions="Which?",
        criteria={"billing": "pay", "technical": "bug", "sales": "price"},
    )
    msgs = build_messages("test", q)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test/v1") as http:
        client = LLMClient(base_url="http://test/v1", api_key="fake", model="test")
        client._client._client = http
        answer = await strategy.decide(client, msgs, q)
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "billing"
    assert answer.probabilities is not None
    assert answer.probabilities["billing"] > 0.6
    assert answer.confidence is not None
    assert 0.0 <= answer.confidence <= 1.0


@pytest.mark.asyncio
async def test_choice_over_20_raises():
    strategy = LogprobStrategy()
    criteria = {f"opt{i}": f"desc{i}" for i in range(21)}
    q = Question(type=QuestionType.CHOICE, instructions="Pick", criteria=criteria)
    msgs = build_messages("test", q)
    with pytest.raises(UnsupportedError, match="≤20"):
        await strategy.decide(None, msgs, q)


@pytest.mark.asyncio
async def test_score_returns_expected_value():
    app = _make_logprob_app("2", 0.78, {"0": 0.02, "1": 0.20, "2": 0.78})
    strategy = LogprobStrategy()
    q = Question(
        type=QuestionType.SCORE,
        instructions="Rate",
        criteria=["Calm", "Frustrated", "Angry"],
    )
    msgs = build_messages("test", q)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test/v1") as http:
        client = LLMClient(base_url="http://test/v1", api_key="fake", model="test")
        client._client._client = http
        answer = await strategy.decide(client, msgs, q)
    assert isinstance(answer, ScoreAnswer)
    assert 1.5 < answer.score < 2.0  # E = 0*0.02 + 1*0.20 + 2*0.78 = 1.76
    assert answer.probabilities is not None
    assert answer.confidence is not None
    assert 0.0 <= answer.confidence <= 1.0


@pytest.mark.asyncio
async def test_score_over_10_raises():
    strategy = LogprobStrategy()
    criteria = [f"level{i}" for i in range(11)]
    q = Question(type=QuestionType.SCORE, instructions="Rate", criteria=criteria)
    msgs = build_messages("test", q)
    with pytest.raises(UnsupportedError, match="≤10"):
        await strategy.decide(None, msgs, q)


@pytest.mark.asyncio
async def test_noul_fallback_when_both_missing():
    # LLM returns an unrelated token; neither Yes nor No is in top_logprobs
    app = _make_logprob_app("Maybe", 0.99, {"Maybe": 0.99, "Unknown": 0.01})
    strategy = LogprobStrategy()
    q = Question(
        type=QuestionType.NOUL,
        instructions="Is urgent?",
        criteria={"true": "yes", "false": "no"},
    )
    msgs = build_messages("test state", q)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test/v1") as http:
        client = LLMClient(base_url="http://test/v1", api_key="fake", model="test")
        client._client._client = http
        with pytest.raises(LogprobExtractionError, match="No Yes/No tokens found"):
            await strategy.decide(client, msgs, q)


@pytest.mark.asyncio
async def test_noul_case_insensitive_matching():
    app = _make_logprob_app("yes", 0.85, {"yes": 0.85, "no": 0.15})
    strategy = LogprobStrategy()
    q = Question(
        type=QuestionType.NOUL,
        instructions="Is urgent?",
        criteria={"true": "yes", "false": "no"},
    )
    msgs = build_messages("test state", q)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test/v1") as http:
        client = LLMClient(base_url="http://test/v1", api_key="fake", model="test")
        client._client._client = http
        answer = await strategy.decide(client, msgs, q)
    assert isinstance(answer, NoulAnswer)
    assert 0.80 < answer.noul < 0.90


@pytest.mark.asyncio
async def test_choice_missing_tokens_get_default_logprob():
    # Only "A" is in top_logprobs, "B" and "C" are missing
    app = _make_logprob_app("A", 0.99, {"A": 0.99})
    strategy = LogprobStrategy()
    q = Question(
        type=QuestionType.CHOICE,
        instructions="Which?",
        criteria={"opt1": "first", "opt2": "second", "opt3": "third"},
    )
    msgs = build_messages("test", q)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test/v1") as http:
        client = LLMClient(base_url="http://test/v1", api_key="fake", model="test")
        client._client._client = http
        answer = await strategy.decide(client, msgs, q)
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "opt1"
    assert answer.probabilities is not None
    assert answer.probabilities["opt1"] == 1.0
    assert "opt2" not in answer.probabilities
    assert "opt3" not in answer.probabilities


@pytest.mark.asyncio
async def test_calls_llm_with_max_tokens_1_and_logprob_params():
    """LogprobStrategy must send max_tokens=1, logprobs=True, top_logprobs=20, temperature=0."""
    captured_requests: list[dict] = []
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
                "message": {"role": "assistant", "content": "Yes"},
                "finish_reason": "stop",
                "logprobs": {
                    "content": [{
                        "token": "Yes",
                        "logprob": math.log(0.9),
                        "top_logprobs": [
                            {"token": "Yes", "logprob": math.log(0.9)},
                            {"token": "No", "logprob": math.log(0.1)},
                        ],
                    }]
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        })

    strategy = LogprobStrategy()
    q = Question(
        type=QuestionType.NOUL,
        instructions="Is urgent?",
        criteria={"true": "yes", "false": "no"},
    )
    msgs = build_messages("test state", q)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test/v1") as http:
        client = LLMClient(base_url="http://test/v1", api_key="fake", model="test")
        client._client._client = http
        await strategy.decide(client, msgs, q)

    assert len(captured_requests) == 1
    body = captured_requests[0]
    assert body["max_tokens"] == 1
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 20
    assert body["temperature"] == 0


def test_strategy_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        Strategy()  # type: ignore
