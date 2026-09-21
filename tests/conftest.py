import json
import math
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

dummy_app = FastAPI()


@dummy_app.post("/v1/chat/completions")
@dummy_app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    logprobs_requested = body.get("logprobs", False)
    response_format = body.get("response_format")

    # Default: echo back a simple response
    content = "Yes"
    logprobs_data = None

    if logprobs_requested:
        logprobs_data = {
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

    if response_format and response_format.get("type") == "json_object":
        content = json.dumps({"answer": True})

    return JSONResponse({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": body.get("model", "test-model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": logprobs_data,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    })


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def dummy_openai_url() -> str:
    return "http://test"


@pytest.fixture
def make_llm_client():
    from yaj.client import LLMClient

    def _factory(
        base_url: str = "http://test",
        api_key: str = "fake",
        model: str = "test-model",
    ) -> LLMClient:
        return LLMClient(base_url=base_url, api_key=api_key, model=model)

    return _factory
