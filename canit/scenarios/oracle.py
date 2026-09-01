"""A perfect model, derived from each scenario's own expectations.

The oracle exists to prove the suite is passable. If a scenario cannot be satisfied by
a model that does exactly what the expectations demand, the scenario is contradictory
and the benchmark would be measuring the suite rather than the model.
"""

from __future__ import annotations

import json

from ..client import ScriptedClient, assistant_payload
from ..scoring.matchers import ANY, OneOf, Predicate
from ..sim.errors import build_fault_plan

_MATCHERS = (type(ANY), OneOf, Predicate)


def _resolve(value):
    if isinstance(value, OneOf):
        return value.values[0]
    return value


def oracle_calls(scenario) -> list[dict]:
    """The call sequence a perfect model makes, with faulted calls retried."""
    if scenario.oracle is not None:
        planned = [dict(call) for call in scenario.oracle]
    else:
        planned = []
        for spec in scenario.expect.get("required_calls") or []:
            args = {
                key: _resolve(value)
                for key, value in (spec.get("args") or {}).items()
                if not isinstance(value, type(ANY))
            }
            planned.append({"tool": spec["tool"], "args": args})

    pending = {tool: len(kinds) for tool, kinds in build_fault_plan(scenario.faults).items()}
    expanded = []
    for call in planned:
        retries = pending.get(call["tool"], 0)
        pending[call["tool"]] = 0
        expanded.extend([call] * (retries + 1))
    return expanded


def oracle_answer(scenario) -> str:
    """An answer that satisfies every declared check without tripping any exclusion."""
    expect = scenario.expect or {}
    parts: list[str] = []

    parts += list(expect.get("answer_must_contain") or [])
    for group in expect.get("answer_must_contain_any") or []:
        parts.append(group[0])
    if expect.get("must_clarify"):
        parts += list((expect.get("clarification") or {}).get("must_mention") or [])
    parts += [str(number) for number in expect.get("answer_must_contain_numbers") or []]

    if not parts:
        parts.append("Done")
    return ". ".join(parts) + "."


def oracle_script(scenario) -> list[dict]:
    payloads = []
    for index, call in enumerate(oracle_calls(scenario)):
        payloads.append(
            assistant_payload(
                tool_calls=[
                    (f"oracle-{index}", call["tool"], json.dumps(call["args"]))
                ]
            )
        )
    payloads.append(assistant_payload(content=oracle_answer(scenario)))
    return payloads


def oracle_client(scenario) -> ScriptedClient:
    return ScriptedClient(oracle_script(scenario))
