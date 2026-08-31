"""Chronological record of one scenario run, exactly as observed."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

TERMINATION_FINAL_ANSWER = "final_answer"
TERMINATION_MAX_STEPS = "max_steps"
TERMINATION_TIMEOUT = "timeout"
TERMINATION_CLIENT_ERROR = "client_error"

TERMINATION_REASONS = frozenset(
    {
        TERMINATION_FINAL_ANSWER,
        TERMINATION_MAX_STEPS,
        TERMINATION_TIMEOUT,
        TERMINATION_CLIENT_ERROR,
    }
)


@dataclass
class ToolCallRecord:
    step: int
    index: int
    call_id: str | None
    tool: str
    raw_arguments: str
    arguments: dict | None
    parse_error: str | None
    known_tool: bool
    is_write: bool
    result: dict
    latency_ms: float
    injected_fault: str | None = None
    mutated: bool = False


@dataclass
class StepRecord:
    index: int
    assistant_message: dict | None
    content: str | None
    finish_reason: str | None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    latency_ms: float = 0.0
    usage: dict | None = None
    timings: dict | None = None
    error: dict | None = None


@dataclass
class Trace:
    scenario_id: str
    category: str
    run_index: int
    model: str
    protocol: str
    started_at: str
    finished_at: str | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    messages: list[dict] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    final_answer: str | None = None
    termination: str = TERMINATION_CLIENT_ERROR
    termination_detail: str | None = None
    total_latency_ms: float = 0.0
    mutations: list[dict] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    @property
    def tool_calls(self) -> list[ToolCallRecord]:
        return [call for step in self.steps for call in step.tool_calls]

    @property
    def tool_sequence(self) -> list[str]:
        return [call.tool for call in self.tool_calls]

    @property
    def completed(self) -> bool:
        return self.termination == TERMINATION_FINAL_ANSWER

    def usage_totals(self) -> dict:
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        seen = False
        for step in self.steps:
            if not step.usage:
                continue
            seen = True
            for key in totals:
                value = step.usage.get(key)
                if isinstance(value, int):
                    totals[key] += value
        if not seen:
            return {}
        return totals

    def tokens_per_second(self) -> float | None:
        """Prefer server-reported generation timings; fall back to wall clock."""
        reported = [
            s.timings.get("predicted_per_second")
            for s in self.steps
            if s.timings and isinstance(s.timings.get("predicted_per_second"), (int, float))
        ]
        if reported:
            return sum(reported) / len(reported)

        usage = self.usage_totals()
        completion = usage.get("completion_tokens", 0)
        if not completion or self.total_latency_ms <= 0:
            return None
        return completion / (self.total_latency_ms / 1000.0)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["usage_totals"] = self.usage_totals()
        payload["tokens_per_second"] = self.tokens_per_second()
        payload["tool_sequence"] = self.tool_sequence
        return payload
