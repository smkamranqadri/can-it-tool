"""Suite validation.

Catches expectations that are contradictory, impossible, or that reference entities
the seed data does not contain. A scenario that cannot be satisfied is a suite bug,
not a model failure, so this runs as a test rather than at benchmark time.
"""

from __future__ import annotations

from ..scoring.matchers import ABSENT, ANY, OneOf, Predicate
from ..sim.db import Store
from ..sim.errors import FAULT_RESPONSES
from ..sim.schemas import ALL_TOOLS, READ_TOOL_NAMES, TOOL_NAMES, WRITE_TOOL_NAMES
from ..sim.tools import execute, validate_arguments

_MATCHERS = (type(ANY), OneOf, Predicate, type(ABSENT))

_TOOL_KEYS = ("forbidden_tools", "optional_tools")
_ANSWER_TEXT_KEYS = ("answer_must_contain", "answer_must_not_contain")


def _concrete(args: dict) -> dict:
    return {k: v for k, v in (args or {}).items() if not isinstance(v, _MATCHERS)}


def _dig(payload, path: str):
    cursor = payload
    for part in path.split("."):
        if isinstance(cursor, list):
            cursor = cursor[int(part)]
        else:
            cursor = cursor[part]
    return cursor


def check_structure(scenario) -> list[str]:
    problems = []
    expect = scenario.expect or {}

    if not scenario.id or not scenario.prompt.strip():
        problems.append("scenario needs an id and a prompt")

    referenced = [spec["tool"] for spec in expect.get("required_calls") or []]
    referenced += [spec["tool"] for spec in expect.get("expected_writes") or []]
    for key in _TOOL_KEYS:
        referenced += list(expect.get(key) or [])
    referenced += list(scenario.faults or {})
    for call in scenario.oracle or []:
        referenced.append(call["tool"])

    for tool in referenced:
        if tool not in TOOL_NAMES:
            problems.append(f"references a tool that does not exist: {tool}")

    for tool, kinds in (scenario.faults or {}).items():
        for kind in kinds:
            if kind not in FAULT_RESPONSES:
                problems.append(f"unknown fault kind for {tool}: {kind}")

    for spec in expect.get("expected_writes") or []:
        if spec["tool"] in READ_TOOL_NAMES:
            problems.append(f"expected_writes names a read tool: {spec['tool']}")

    for spec in (expect.get("required_calls") or []) + (expect.get("expected_writes") or []):
        if spec["tool"] not in ALL_TOOLS:
            continue
        reason = validate_arguments(spec["tool"], _concrete(spec.get("args") or {}))
        if reason is not None and "missing required argument" not in reason:
            problems.append(f"{spec['tool']} expectation has invalid arguments: {reason}")

    return problems


def check_contradictions(scenario) -> list[str]:
    problems = []
    expect = scenario.expect or {}
    required = expect.get("required_calls") or []
    required_tools = {spec["tool"] for spec in required}
    forbidden = set(expect.get("forbidden_tools") or [])

    if expect.get("expect_no_tools") and required:
        problems.append("expect_no_tools is set alongside required_calls")

    if expect.get("expect_no_tools") and scenario.faults:
        problems.append("expect_no_tools is set but faults are injected on tools")

    if expect.get("must_clarify") and expect.get("must_not_clarify"):
        problems.append("must_clarify and must_not_clarify are both set")

    if expect.get("must_refuse") and expect.get("expected_writes"):
        problems.append("must_refuse is set alongside expected_writes")

    overlap = required_tools & forbidden
    if overlap:
        problems.append(f"tools are both required and forbidden: {sorted(overlap)}")

    write_overlap = {s["tool"] for s in expect.get("expected_writes") or []} & forbidden
    if write_overlap:
        problems.append(f"writes are both expected and forbidden: {sorted(write_overlap)}")

    budget = expect.get("max_calls")
    minimum = len(required) + sum(len(v) for v in (scenario.faults or {}).values())
    if budget is not None and budget < len(required):
        problems.append(f"max_calls {budget} is below the {len(required)} required calls")

    if expect.get("must_clarify") and not (expect.get("clarification") or {}).get("must_mention"):
        problems.append("must_clarify is set without clarification.must_mention tokens")

    if not expect.get("must_clarify") and expect.get("clarification"):
        problems.append("clarification tokens declared without must_clarify")

    contains = {t.casefold() for t in expect.get("answer_must_contain") or []}
    excludes = {t.casefold() for t in expect.get("answer_must_not_contain") or []}
    if contains & excludes:
        problems.append(f"answer must both contain and exclude: {sorted(contains & excludes)}")

    numbers_in = set(expect.get("answer_must_contain_numbers") or [])
    numbers_out = set(expect.get("answer_must_not_contain_numbers") or [])
    if numbers_in & numbers_out:
        problems.append(f"answer must both contain and exclude: {sorted(numbers_in & numbers_out)}")

    clarification = expect.get("clarification") or {}
    mention = {t.casefold() for t in clarification.get("must_mention") or []}
    commit = {t.casefold() for t in clarification.get("must_not_commit") or []}
    if mention & commit:
        problems.append(f"clarification must both mention and avoid: {sorted(mention & commit)}")

    for spec in expect.get("expected_writes") or []:
        if spec.get("applied"):
            problems.append(
                f"{spec['tool']} expects applied=True, unreachable in a single-turn run"
            )

    if expect.get("must_refuse"):
        for tool in required_tools:
            if tool in WRITE_TOOL_NAMES:
                problems.append(f"must_refuse requires the write tool {tool}")

    if minimum and budget is not None and minimum > budget + sum(
        len(v) for v in (scenario.faults or {}).values()
    ):
        problems.append("required calls plus injected faults exceed max_calls")

    return problems


def check_ground_truth(scenario) -> list[str]:
    """Replay each recorded fact against a fresh store."""
    problems = []
    for check in scenario.ground_truth or []:
        store = Store()
        result = execute(store, check["tool"], check["args"])
        for path, expected in (check.get("assert") or {}).items():
            try:
                actual = _dig(result, path)
            except (KeyError, IndexError, ValueError, TypeError):
                problems.append(
                    f"ground truth path {path!r} is missing from {check['tool']} result"
                )
                continue
            if actual != expected:
                problems.append(
                    f"ground truth drift on {check['tool']} {path}: "
                    f"expected {expected!r}, simulator says {actual!r}"
                )
        if store.mutations:
            problems.append(f"ground truth check on {check['tool']} mutated the store")
    return problems


def validate(scenario) -> list[str]:
    return (
        check_structure(scenario)
        + check_contradictions(scenario)
        + check_ground_truth(scenario)
    )


def validate_suite(scenarios) -> dict[str, list[str]]:
    seen: dict[str, int] = {}
    report: dict[str, list[str]] = {}
    for scenario in scenarios:
        problems = validate(scenario)
        seen[scenario.id] = seen.get(scenario.id, 0) + 1
        if seen[scenario.id] > 1:
            problems.append("duplicate scenario id")
        if problems:
            report[scenario.id] = problems
    return report
