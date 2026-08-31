"""The native client's HTTP behavior, exercised through a mock transport."""

from __future__ import annotations

import json

import httpx
import pytest

from canit.client import (
    ClientError,
    ClientTimeout,
    NativeToolClient,
    assistant_payload,
)
from canit.config import RunConfig


def client_for(handler, **overrides) -> NativeToolClient:
    config = RunConfig(
        base_url="http://localhost:8080/v1",
        model="qwen",
        max_retries=overrides.pop("max_retries", 0),
        **overrides,
    )
    return NativeToolClient(config, transport=httpx.MockTransport(handler))


def ok_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=assistant_payload(content="hello"))


def test_request_targets_the_chat_completions_endpoint():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return ok_response(request)

    client_for(handler).complete([{"role": "user", "content": "hi"}], [])
    assert seen["url"] == "http://localhost:8080/v1/chat/completions"


def test_body_carries_model_tools_and_temperature():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return ok_response(request)

    tools = [{"type": "function", "function": {"name": "get_student"}}]
    client_for(handler).complete([{"role": "user", "content": "hi"}], tools)

    body = seen["body"]
    assert body["model"] == "qwen"
    assert body["temperature"] == 0.0
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"
    assert body["stream"] is False


def test_api_key_becomes_a_bearer_header():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return ok_response(request)

    client_for(handler, api_key="sk-test").complete([], [])
    assert seen["auth"] == "Bearer sk-test"


def test_no_authorization_header_without_an_api_key():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return ok_response(request)

    client_for(handler).complete([], [])
    assert seen["auth"] is None


def test_num_ctx_and_max_tokens_are_sent_only_when_configured():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return ok_response(request)

    client_for(handler).complete([], [])
    client_for(handler, num_ctx=8192, max_tokens=512).complete([], [])

    assert "options" not in bodies[0]
    assert "max_tokens" not in bodies[0]
    assert bodies[1]["options"] == {"num_ctx": 8192}
    assert bodies[1]["max_tokens"] == 512


def test_extra_body_overrides_defaults():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return ok_response(request)

    client_for(handler, extra_body={"temperature": 0.7, "top_p": 0.9}).complete([], [])
    assert seen["body"]["temperature"] == 0.7
    assert seen["body"]["top_p"] == 0.9


def test_timeout_is_raised_as_client_timeout():
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(ClientTimeout) as excinfo:
        client_for(handler).complete([], [])
    assert excinfo.value.kind == "timeout"


def test_transport_error_is_raised_as_client_error():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ClientError) as excinfo:
        client_for(handler).complete([], [])
    assert excinfo.value.kind == "transport_error"


def test_http_error_carries_the_status_and_body():
    def handler(request):
        return httpx.Response(400, text="bad model name")

    with pytest.raises(ClientError) as excinfo:
        client_for(handler).complete([], [])
    assert excinfo.value.kind == "http_error"
    assert excinfo.value.detail["status_code"] == 400
    assert "bad model name" in excinfo.value.detail["body"]


def test_non_json_body_is_a_protocol_error():
    def handler(request):
        return httpx.Response(200, text="<html>nope</html>")

    with pytest.raises(ClientError) as excinfo:
        client_for(handler).complete([], [])
    assert excinfo.value.kind == "protocol_error"


def test_retryable_status_is_retried_up_to_the_limit(monkeypatch):
    monkeypatch.setattr("canit.client._RETRY_BACKOFF_SECONDS", 0)
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429, text="slow down")
        return ok_response(request)

    response = client_for(handler, max_retries=2).complete([], [])
    assert len(attempts) == 3
    assert response.content == "hello"


def test_retries_are_exhausted_then_the_error_surfaces(monkeypatch):
    monkeypatch.setattr("canit.client._RETRY_BACKOFF_SECONDS", 0)
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(503, text="unavailable")

    with pytest.raises(ClientError):
        client_for(handler, max_retries=1).complete([], [])
    assert len(attempts) == 2


def test_a_400_is_not_retried(monkeypatch):
    monkeypatch.setattr("canit.client._RETRY_BACKOFF_SECONDS", 0)
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(400, text="bad request")

    with pytest.raises(ClientError):
        client_for(handler, max_retries=3).complete([], [])
    assert len(attempts) == 1


def test_usage_and_timings_pass_through():
    def handler(request):
        payload = assistant_payload(
            content="hi",
            usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            timings={"predicted_per_second": 33.5},
        )
        return httpx.Response(200, json=payload)

    response = client_for(handler).complete([], [])
    assert response.usage["total_tokens"] == 12
    assert response.timings["predicted_per_second"] == 33.5
    assert response.latency_ms >= 0
