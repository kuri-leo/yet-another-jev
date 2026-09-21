import json
import math

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from yaj.client import LLMClient
from yaj.server import create_app
from yaj.strategy.logprob import LogprobStrategy
from yaj.strategy.structured import StructuredStrategy


def _make_dummy_llm():
    """Dummy LLM that handles both logprob and structured requests."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        logprobs_requested = body.get("logprobs", False)
        rf = body.get("response_format")

        if logprobs_requested:
            content = "Yes"
            lp_data = {
                "content": [
                    {
                        "token": "Yes",
                        "logprob": math.log(0.9),
                        "top_logprobs": [
                            {"token": "Yes", "logprob": math.log(0.9)},
                            {"token": "No", "logprob": math.log(0.1)},
                        ],
                    }
                ]
            }
        elif rf and rf.get("type") == "json_object":
            content = json.dumps({"answer": True})
            lp_data = None
        else:
            content = "Yes"
            lp_data = None

        return JSONResponse(
            {
                "id": "test",
                "object": "chat.completion",
                "model": body.get("model", "test"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                        "logprobs": lp_data,
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            }
        )

    return app


async def _make_yaj_client(strategy, dummy_llm):
    """Helper: wire up dummy LLM → LLMClient → yaj app → AsyncClient."""
    transport = ASGITransport(app=dummy_llm)
    http = AsyncClient(transport=transport, base_url="http://llm/v1")
    client = LLMClient(base_url="http://llm/v1", api_key="fake", model="test")
    client._client._client = http
    app = create_app(strategy, client)
    app_transport = ASGITransport(app=app)
    yaj = AsyncClient(transport=app_transport, base_url="http://yaj")
    return yaj, http


# ── happy-path tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_logprob_strategy_noul_e2e():
    dummy_llm = _make_dummy_llm()
    yaj, http = await _make_yaj_client(LogprobStrategy(), dummy_llm)
    async with http, yaj:
        resp = await yaj.post(
            "/api/alpha/decisions",
            json={
                "model": "test",
                "state": "Customer: I'm leaving!",
                "questions": {
                    "churn": {
                        "type": "noul",
                        "instructions": "Is this customer at risk?",
                        "criteria": {"true": "Threatening", "false": "Routine"},
                    }
                },
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "churn" in data["answers"]
    assert data["answers"]["churn"]["type"] == "noul"
    assert 0.0 <= data["answers"]["churn"]["noul"] <= 1.0


@pytest.mark.asyncio
async def test_structured_strategy_noul_e2e():
    dummy_llm = _make_dummy_llm()
    yaj, http = await _make_yaj_client(StructuredStrategy(), dummy_llm)
    async with http, yaj:
        resp = await yaj.post(
            "/api/alpha/decisions",
            json={
                "model": "test",
                "state": "Customer: I'm leaving!",
                "questions": {
                    "churn": {
                        "type": "noul",
                        "instructions": "Is this customer at risk?",
                        "criteria": {"true": "Threatening", "false": "Routine"},
                    }
                },
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answers"]["churn"]["noul"] in (0.0, 1.0)


@pytest.mark.asyncio
async def test_multiple_questions_concurrent():
    dummy_llm = _make_dummy_llm()
    yaj, http = await _make_yaj_client(LogprobStrategy(), dummy_llm)
    async with http, yaj:
        resp = await yaj.post(
            "/api/alpha/decisions",
            json={
                "model": "test",
                "state": "Test",
                "questions": {
                    "q1": {
                        "type": "noul",
                        "instructions": "Q1?",
                        "criteria": {"true": "y", "false": "n"},
                    },
                    "q2": {
                        "type": "noul",
                        "instructions": "Q2?",
                        "criteria": {"true": "y", "false": "n"},
                    },
                },
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "q1" in data["answers"] and "q2" in data["answers"]


@pytest.mark.asyncio
async def test_usage_aggregated():
    """Usage tokens should be summed across all sub-calls."""
    dummy_llm = _make_dummy_llm()
    yaj, http = await _make_yaj_client(LogprobStrategy(), dummy_llm)
    async with http, yaj:
        resp = await yaj.post(
            "/api/alpha/decisions",
            json={
                "model": "test",
                "state": "Test",
                "questions": {
                    "q1": {
                        "type": "noul",
                        "instructions": "Q1?",
                        "criteria": {"true": "y", "false": "n"},
                    },
                    "q2": {
                        "type": "noul",
                        "instructions": "Q2?",
                        "criteria": {"true": "y", "false": "n"},
                    },
                },
            },
        )
    data = resp.json()
    # Each call: 10 prompt + 1 completion. Two questions → 20 + 2.
    assert data["usage"]["input_tokens"] == 20
    assert data["usage"]["output_tokens"] == 2


# ── error handling tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_logprob_choice_over_20_returns_422():
    """UnsupportedError for >20 choice options → 422."""
    dummy_llm = _make_dummy_llm()
    yaj, http = await _make_yaj_client(LogprobStrategy(), dummy_llm)
    criteria = {f"opt{i}": f"desc{i}" for i in range(21)}
    async with http, yaj:
        resp = await yaj.post(
            "/api/alpha/decisions",
            json={
                "model": "test",
                "state": "test",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Pick",
                        "criteria": criteria,
                    }
                },
            },
        )
    assert resp.status_code == 422
    assert "logprob strategy supports at most 20 choice options" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_logprob_score_over_10_returns_422():
    """UnsupportedError for >10 score levels → 422."""
    dummy_llm = _make_dummy_llm()
    yaj, http = await _make_yaj_client(LogprobStrategy(), dummy_llm)
    criteria = [f"level{i}" for i in range(11)]
    async with http, yaj:
        resp = await yaj.post(
            "/api/alpha/decisions",
            json={
                "model": "test",
                "state": "test",
                "questions": {
                    "q": {
                        "type": "score",
                        "instructions": "Rate",
                        "criteria": criteria,
                    }
                },
            },
        )
    assert resp.status_code == 422
    assert "score criteria must have 2-10 levels" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_llm_error_returns_502():
    """LLM call failure → 502 with upstream LLM error."""
    # Build a dummy LLM that always returns 500
    error_app = FastAPI()

    @error_app.post("/v1/chat/completions")
    async def fail(_request: Request):
        return JSONResponse(
            {"error": {"message": "internal server error"}}, status_code=500
        )

    yaj, http = await _make_yaj_client(LogprobStrategy(), error_app)
    async with http, yaj:
        resp = await yaj.post(
            "/api/alpha/decisions",
            json={
                "model": "test",
                "state": "test",
                "questions": {
                    "q": {
                        "type": "noul",
                        "instructions": "X?",
                        "criteria": {"true": "y", "false": "n"},
                    }
                },
            },
        )
    assert resp.status_code == 502
    assert "upstream LLM error" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_llm_error_502_identifies_failing_question():
    """When one of multiple concurrent questions fails, 502 detail names the question."""
    error_app = FastAPI()

    @error_app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        # Inspect the messages to decide which question this is.
        # The failing question "bad_q" will get a 500; the other succeeds.
        msgs_text = str(body.get("messages", []))
        if "bad_q_instructions" in msgs_text:
            return JSONResponse(
                {"error": {"message": "internal server error"}}, status_code=500
            )
        # Normal success for other questions
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
                        "logprob": -0.105,
                        "top_logprobs": [
                            {"token": "Yes", "logprob": -0.105},
                            {"token": "No", "logprob": -2.302},
                        ],
                    }]
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        })

    yaj, http = await _make_yaj_client(LogprobStrategy(), error_app)
    async with http, yaj:
        resp = await yaj.post(
            "/api/alpha/decisions",
            json={
                "model": "test",
                "state": "test",
                "questions": {
                    "good_q": {
                        "type": "noul",
                        "instructions": "good_q_instructions",
                        "criteria": {"true": "y", "false": "n"},
                    },
                    "bad_q": {
                        "type": "noul",
                        "instructions": "bad_q_instructions",
                        "criteria": {"true": "y", "false": "n"},
                    },
                },
            },
        )
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "upstream LLM error" in detail
    assert "bad_q" in detail


@pytest.mark.asyncio
async def test_structured_output_error_returns_502():
    """StructuredOutputError → 502 with upstream LLM error."""
    # Build a dummy LLM that returns malformed JSON
    bad_app = FastAPI()

    @bad_app.post("/v1/chat/completions")
    async def bad_json(_request: Request):
        return JSONResponse(
            {
                "id": "test",
                "object": "chat.completion",
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "not valid json {{{",
                        },
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            }
        )

    yaj, http = await _make_yaj_client(StructuredStrategy(), bad_app)
    async with http, yaj:
        resp = await yaj.post(
            "/api/alpha/decisions",
            json={
                "model": "test",
                "state": "test",
                "questions": {
                    "q": {
                        "type": "noul",
                        "instructions": "X?",
                        "criteria": {"true": "y", "false": "n"},
                    }
                },
            },
        )
    assert resp.status_code == 502
    assert "upstream LLM error" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_invalid_request_body_returns_422():
    """Missing required fields → 422 from Pydantic validation."""
    dummy_llm = _make_dummy_llm()
    yaj, http = await _make_yaj_client(LogprobStrategy(), dummy_llm)
    async with http, yaj:
        resp = await yaj.post("/api/alpha/decisions", json={"model": "test"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_score_0_levels_returns_422():
    """Score criteria with 0 levels → 422."""
    dummy_llm = _make_dummy_llm()
    yaj, http = await _make_yaj_client(LogprobStrategy(), dummy_llm)
    async with http, yaj:
        resp = await yaj.post(
            "/api/alpha/decisions",
            json={
                "model": "test",
                "state": "test",
                "questions": {
                    "q": {
                        "type": "score",
                        "instructions": "Rate",
                        "criteria": [],
                    }
                },
            },
        )
    assert resp.status_code == 422
    assert "score criteria must have 2-10 levels" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_score_1_level_returns_422():
    """Score criteria with 1 level → 422."""
    dummy_llm = _make_dummy_llm()
    yaj, http = await _make_yaj_client(StructuredStrategy(), dummy_llm)
    async with http, yaj:
        resp = await yaj.post(
            "/api/alpha/decisions",
            json={
                "model": "test",
                "state": "test",
                "questions": {
                    "q": {
                        "type": "score",
                        "instructions": "Rate",
                        "criteria": ["only one"],
                    }
                },
            },
        )
    assert resp.status_code == 422
    assert "score criteria must have 2-10 levels" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_score_11_levels_structured_returns_422():
    """Score criteria with 11 levels → 422 even with structured strategy."""
    dummy_llm = _make_dummy_llm()
    yaj, http = await _make_yaj_client(StructuredStrategy(), dummy_llm)
    criteria = [f"level{i}" for i in range(11)]
    async with http, yaj:
        resp = await yaj.post(
            "/api/alpha/decisions",
            json={
                "model": "test",
                "state": "test",
                "questions": {
                    "q": {
                        "type": "score",
                        "instructions": "Rate",
                        "criteria": criteria,
                    }
                },
            },
        )
    assert resp.status_code == 422
    assert "score criteria must have 2-10 levels" in resp.json()["detail"]


# ── CLI tests ─────────────────────────────────────────────────────


def test_cli_strategy_accepts_only_valid_choices():
    """--strategy only accepts 'logprob' and 'structured'."""
    from click.testing import CliRunner
    from yaj.cli import main

    runner = CliRunner()
    result = runner.invoke(main, [
        "--strategy", "invalid",
        "--base-url", "http://x",
        "--api-key", "k",
        "--model", "m",
    ])
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "invalid" in result.output.lower()


def test_cli_requires_base_url():
    """--base-url is required."""
    from click.testing import CliRunner
    from yaj.cli import main

    runner = CliRunner()
    result = runner.invoke(main, [
        "--strategy", "logprob",
        "--api-key", "k",
        "--model", "m",
    ])
    assert result.exit_code != 0
    assert "base-url" in result.output.lower() or "missing" in result.output.lower()


def test_cli_requires_api_key():
    """--api-key is required."""
    from click.testing import CliRunner
    from yaj.cli import main

    runner = CliRunner()
    result = runner.invoke(main, [
        "--strategy", "logprob",
        "--base-url", "http://x",
        "--model", "m",
    ])
    assert result.exit_code != 0
    assert "api-key" in result.output.lower() or "missing" in result.output.lower()


def test_cli_requires_model():
    """--model is required."""
    from click.testing import CliRunner
    from yaj.cli import main

    runner = CliRunner()
    result = runner.invoke(main, [
        "--strategy", "logprob",
        "--base-url", "http://x",
        "--api-key", "k",
    ])
    assert result.exit_code != 0
    assert "model" in result.output.lower() or "missing" in result.output.lower()


def test_cli_defaults():
    """--host defaults to 0.0.0.0, --port defaults to 8000."""
    from unittest.mock import patch
    from click.testing import CliRunner
    from yaj.cli import main

    with patch("yaj.cli.uvicorn.run") as mock_run:
        runner = CliRunner()
        result = runner.invoke(main, [
            "--strategy", "logprob",
            "--base-url", "http://test",
            "--api-key", "fake",
            "--model", "gpt-test",
        ])
    assert result.exit_code == 0
    mock_run.assert_called_once()
    _args, kwargs = mock_run.call_args
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8000

