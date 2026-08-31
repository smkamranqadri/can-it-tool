"""Chat-completions clients behind a protocol seam.

v1 implements only `native`: send the OpenAI `tools` array, read `message.tool_calls`.
A prompted-JSON fallback can be registered in `PROTOCOLS` later without the runner
changing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Protocol

from .config import RunConfig

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_RETRY_BACKOFF_SECONDS = 0.5


class ClientError(Exception):
    """Unrecoverable failure talking to the endpoint. Terminates the run."""

    def __init__(self, message: str, *, kind: str = "client_error", detail: dict | None = None):
        super().__init__(message)
        self.kind = kind
        self.detail = detail or {}


class ClientTimeout(ClientError):
    def __init__(self, message: str, detail: dict | None = None):
        super().__init__(message, kind="timeout", detail=detail)


@dataclass
class RawToolCall:
    call_id: str | None
    name: str
    raw_arguments: str


@dataclass
class ChatResponse:
    message: dict
    content: str | None
    tool_calls: list[RawToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict | None = None
    timings: dict | None = None
    latency_ms: float = 0.0
    raw: dict | None = None


class ChatClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> ChatResponse: ...

    def close(self) -> None: ...


def _coerce_raw_arguments(value: object) -> str:
    """Servers return arguments as a JSON string; some return an object. Keep both."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def parse_chat_response(payload: dict, latency_ms: float) -> ChatResponse:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ClientError(
            "response contained no choices",
            kind="protocol_error",
            detail={"payload": payload},
        )

    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ClientError(
            "response choice contained no message",
            kind="protocol_error",
            detail={"choice": choice},
        )

    tool_calls = []
    for entry in message.get("tool_calls") or []:
        if not isinstance(entry, dict):
            raise ClientError(
                "malformed tool_calls entry",
                kind="protocol_error",
                detail={"entry": entry},
            )
        function = entry.get("function") or {}
        tool_calls.append(
            RawToolCall(
                call_id=entry.get("id"),
                name=function.get("name") or "",
                raw_arguments=_coerce_raw_arguments(function.get("arguments")),
            )
        )

    return ChatResponse(
        message=message,
        content=message.get("content"),
        tool_calls=tool_calls,
        finish_reason=choice.get("finish_reason"),
        usage=payload.get("usage"),
        timings=payload.get("timings"),
        latency_ms=latency_ms,
        raw=payload,
    )


class NativeToolClient:
    """OpenAI-compatible `/chat/completions` with native tool calling."""

    def __init__(self, config: RunConfig, transport=None):
        import httpx

        self.config = config
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        self._httpx = httpx
        self._client = httpx.Client(
            timeout=config.timeout, headers=headers, transport=transport
        )

    def build_body(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.config.temperature,
            "stream": False,
        }
        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens
        if self.config.num_ctx is not None:
            body["options"] = {"num_ctx": self.config.num_ctx}
        body.update(self.config.extra_body)
        return body

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        body = self.build_body(messages, tools)
        url = self.config.endpoint()
        attempt = 0

        while True:
            started = time.perf_counter()
            try:
                response = self._client.post(url, json=body)
            except self._httpx.TimeoutException as exc:
                raise ClientTimeout(
                    f"request timed out after {self.config.timeout}s",
                    detail={"url": url},
                ) from exc
            except self._httpx.HTTPError as exc:
                raise ClientError(
                    f"transport error: {exc}",
                    kind="transport_error",
                    detail={"url": url},
                ) from exc
            latency_ms = (time.perf_counter() - started) * 1000.0

            if response.status_code in _RETRYABLE_STATUS and attempt < self.config.max_retries:
                attempt += 1
                time.sleep(_RETRY_BACKOFF_SECONDS)
                continue

            if response.status_code >= 400:
                raise ClientError(
                    f"endpoint returned HTTP {response.status_code}",
                    kind="http_error",
                    detail={
                        "status_code": response.status_code,
                        "body": response.text[:2000],
                    },
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise ClientError(
                    "endpoint returned a non-JSON body",
                    kind="protocol_error",
                    detail={"body": response.text[:2000]},
                ) from exc

            return parse_chat_response(payload, latency_ms)

    def close(self) -> None:
        self._client.close()


class ScriptedClient:
    """A client driven by a fixed script, for tests and for mock-model baselines.

    Each entry is either a payload dict shaped like a chat-completions response, a
    `ChatResponse`, or an exception instance to raise.
    """

    def __init__(self, script: list, latency_ms: float = 1.0):
        self.script = list(script)
        self.latency_ms = latency_ms
        self.requests: list[dict] = []
        self.closed = False

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self.script:
            raise ClientError("scripted client ran out of responses", kind="protocol_error")

        entry = self.script.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        if isinstance(entry, ChatResponse):
            return entry
        return parse_chat_response(entry, self.latency_ms)

    def close(self) -> None:
        self.closed = True


def assistant_payload(
    content: str | None = None,
    tool_calls: list[tuple[str, str, str]] | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
    timings: dict | None = None,
) -> dict:
    """Build a chat-completions payload. `tool_calls` are (id, name, raw_arguments)."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": raw_arguments},
            }
            for call_id, name, raw_arguments in tool_calls
        ]
    resolved_finish = finish_reason
    if resolved_finish is None:
        resolved_finish = "tool_calls" if tool_calls else "stop"
    payload = {
        "choices": [{"index": 0, "message": message, "finish_reason": resolved_finish}]
    }
    if usage:
        payload["usage"] = usage
    if timings:
        payload["timings"] = timings
    return payload


PROTOCOLS = {"native": NativeToolClient}


def build_client(config: RunConfig, **kwargs) -> ChatClient:
    if config.protocol not in PROTOCOLS:
        raise ValueError(
            f"unknown protocol {config.protocol!r}. "
            f"Available: {', '.join(sorted(PROTOCOLS))}"
        )
    return PROTOCOLS[config.protocol](config, **kwargs)
