import pytest
from httpx import ASGITransport, AsyncClient

try:
    from tests.conftest import dummy_app
except ModuleNotFoundError:
    from conftest import dummy_app

from yaj.client import LLMClient


@pytest.mark.asyncio
async def test_client_complete_returns_response():
    transport = ASGITransport(app=dummy_app)
    async with AsyncClient(transport=transport, base_url="http://test/v1") as http:
        client = LLMClient(base_url="http://test/v1", api_key="fake", model="test")
        client._client._client = http  # inject test transport
        messages = [{"role": "user", "content": "hello"}]
        response = await client.complete(messages)
        assert response.choices[0].message.content is not None
        assert response.choices[0].message.content == "Yes"


@pytest.mark.asyncio
async def test_client_complete_with_logprobs():
    transport = ASGITransport(app=dummy_app)
    async with AsyncClient(transport=transport, base_url="http://test/v1") as http:
        client = LLMClient(base_url="http://test/v1", api_key="fake", model="test")
        client._client._client = http
        messages = [{"role": "user", "content": "hello"}]
        response = await client.complete(messages, logprobs=True, top_logprobs=2)
        choice = response.choices[0]
        assert choice.logprobs is not None
        top_tokens = [entry.token for entry in choice.logprobs.content[0].top_logprobs]
        assert "Yes" in top_tokens
        assert "No" in top_tokens


@pytest.mark.asyncio
async def test_client_complete_with_json_object():
    transport = ASGITransport(app=dummy_app)
    async with AsyncClient(transport=transport, base_url="http://test/v1") as http:
        client = LLMClient(base_url="http://test/v1", api_key="fake", model="test")
        client._client._client = http
        messages = [{"role": "user", "content": "hello"}]
        response = await client.complete(
            messages, response_format={"type": "json_object"}
        )
        assert response.choices[0].message.content == '{"answer": true}'


def test_make_llm_client_fixture(make_llm_client, dummy_openai_url):
    assert dummy_openai_url == "http://test/v1"
    client = make_llm_client(base_url=dummy_openai_url, api_key="test-key", model="m")
    assert isinstance(client, LLMClient)
    assert client.model == "m"


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["http://test/v1", "http://test/v1/"])
async def test_client_base_url_variations(url):
    transport = ASGITransport(app=dummy_app)
    async with AsyncClient(transport=transport, base_url="http://test/v1") as http:
        client = LLMClient(base_url=url, api_key="fake", model="test")
        client._client._client = http
        messages = [{"role": "user", "content": "hello"}]
        response = await client.complete(messages)
        assert response.choices[0].message.content == "Yes"
