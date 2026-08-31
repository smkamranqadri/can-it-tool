"""The bounded agent loop.

Drives one scenario against one client and returns the complete observed trace. The
loop terminates for exactly four reasons: the model gave a final answer, the tool-call
step budget ran out, a request timed out, or the client hit an unrecoverable error.
Nothing the model can emit - malformed arguments, invented tool names, wrong types -
terminates the run; those are benchmark behavior and are recorded as such.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone

from .client import ChatClient, ClientError, ClientTimeout
from .config import RunConfig
from .prompts import SYSTEM_PROMPT
from .scenarios.base import Scenario
from .sim.db import Store
from .sim.errors import build_fault_plan
from .sim.schemas import openai_tool_specs
from .sim.tools import execute, record_malformed_call
from .trace import (
    TERMINATION_CLIENT_ERROR,
    TERMINATION_FINAL_ANSWER,
    TERMINATION_MAX_STEPS,
    TERMINATION_TIMEOUT,
    StepRecord,
    ToolCallRecord,
    Trace,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_tool_arguments(raw: str) -> tuple[dict | None, str | None]:
    """Parse a tool call's arguments. Returns (arguments, parse_error)."""
    text = (raw or "").strip()
    if text == "":
        return {}, None
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, f"expected a JSON object, got {type(parsed).__name__}"
    if any(not isinstance(key, str) for key in parsed):
        return None, "argument names must be strings"
    return parsed, None


def _tool_message(call_id: str | None, tool: str, result: dict) -> dict:
    message = {
        "role": "tool",
        "name": tool,
        "content": json.dumps(result, sort_keys=True),
    }
    if call_id is not None:
        message["tool_call_id"] = call_id
    return message


def run_scenario(
    config: RunConfig,
    client: ChatClient,
    scenario: Scenario,
    run_index: int = 0,
    store: Store | None = None,
) -> tuple[Trace, Store]:
    store = store or Store()
    store.faults = build_fault_plan(scenario.faults)

    tools = openai_tool_specs()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": scenario.prompt},
    ]

    trace = Trace(
        scenario_id=scenario.id,
        category=scenario.category,
        run_index=run_index,
        model=config.display_name,
        protocol=config.protocol,
        started_at=_now(),
        system_prompt=SYSTEM_PROMPT,
        user_prompt=scenario.prompt,
    )

    max_steps = scenario.max_steps or config.max_steps
    deadline = None
    if config.total_timeout is not None:
        deadline = time.perf_counter() + config.total_timeout

    for step_index in range(max_steps):
        if deadline is not None and time.perf_counter() >= deadline:
            _terminate(
                trace,
                TERMINATION_TIMEOUT,
                f"total time budget of {config.total_timeout}s exhausted",
            )
            break

        step = StepRecord(
            index=step_index, assistant_message=None, content=None, finish_reason=None
        )

        try:
            response = client.complete(copy.deepcopy(messages), tools)
        except ClientTimeout as exc:
            step.error = {"kind": exc.kind, "message": str(exc), "detail": exc.detail}
            trace.steps.append(step)
            trace.errors.append(step.error)
            _terminate(trace, TERMINATION_TIMEOUT, str(exc))
            break
        except ClientError as exc:
            step.error = {"kind": exc.kind, "message": str(exc), "detail": exc.detail}
            trace.steps.append(step)
            trace.errors.append(step.error)
            _terminate(trace, TERMINATION_CLIENT_ERROR, str(exc))
            break

        step.assistant_message = copy.deepcopy(response.message)
        step.content = response.content
        step.finish_reason = response.finish_reason
        step.latency_ms = response.latency_ms
        step.usage = response.usage
        step.timings = response.timings
        trace.total_latency_ms += response.latency_ms
        trace.steps.append(step)
        messages.append(copy.deepcopy(response.message))

        if not response.tool_calls:
            trace.final_answer = response.content
            _terminate(trace, TERMINATION_FINAL_ANSWER, response.finish_reason)
            break

        for call_index, raw_call in enumerate(response.tool_calls):
            record = _execute_call(store, step_index, call_index, raw_call)
            step.tool_calls.append(record)
            messages.append(
                _tool_message(raw_call.call_id, raw_call.name, record.result)
            )
    else:
        _terminate(
            trace,
            TERMINATION_MAX_STEPS,
            f"reached the {max_steps}-step tool-call budget without a final answer",
        )

    trace.messages = messages
    trace.mutations = [asdict(m) for m in store.mutations]
    trace.violations = [asdict(v) for v in store.violations]
    trace.finished_at = _now()
    return trace, store


def _execute_call(store: Store, step_index, call_index, raw_call) -> ToolCallRecord:
    started = time.perf_counter()
    arguments, parse_error = parse_tool_arguments(raw_call.raw_arguments)

    if parse_error is not None:
        result = record_malformed_call(
            store, raw_call.name, raw_call.raw_arguments, parse_error
        )
    else:
        result = execute(store, raw_call.name, arguments)

    latency_ms = (time.perf_counter() - started) * 1000.0
    logged = store.calls[-1]
    return ToolCallRecord(
        step=step_index,
        index=call_index,
        call_id=raw_call.call_id,
        tool=raw_call.name,
        raw_arguments=raw_call.raw_arguments,
        arguments=arguments,
        parse_error=parse_error,
        known_tool=logged["known_tool"],
        is_write=logged["is_write"],
        result=result,
        latency_ms=latency_ms,
        injected_fault=logged.get("injected_fault"),
        mutated=bool(logged.get("mutated")),
    )


def _terminate(trace: Trace, reason: str, detail: str | None) -> None:
    trace.termination = reason
    trace.termination_detail = detail
