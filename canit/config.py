"""Run configuration for one model under test."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

DEFAULT_MAX_STEPS = 8
DEFAULT_TIMEOUT = 120.0
DEFAULT_RUNS = 3


@dataclass(frozen=True)
class RunConfig:
    base_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    num_ctx: int | None = None
    max_tokens: int | None = None
    timeout: float = DEFAULT_TIMEOUT
    total_timeout: float | None = None
    runs: int = DEFAULT_RUNS
    max_steps: int = DEFAULT_MAX_STEPS
    protocol: str = "native"
    max_retries: int = 2
    extra_body: dict = field(default_factory=dict)
    label: str | None = None

    def __post_init__(self) -> None:
        if self.runs < 1:
            raise ValueError("runs must be at least 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    @property
    def display_name(self) -> str:
        return self.label or self.model

    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def with_overrides(self, **kwargs) -> RunConfig:
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        """Config as recorded in results JSON. The API key is never included."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "label": self.label,
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "total_timeout": self.total_timeout,
            "runs": self.runs,
            "max_steps": self.max_steps,
            "protocol": self.protocol,
            "max_retries": self.max_retries,
            "extra_body": self.extra_body,
            "api_key_provided": self.api_key is not None,
        }
