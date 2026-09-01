"""Coverage analysis of the scenario suite itself."""

from __future__ import annotations

import itertools
import re

from ..sim.schemas import READ_TOOL_NAMES, TOOL_NAMES, WRITE_TOOL_NAMES
from .oracle import oracle_calls

_ENTITY = re.compile(r"\b(?:STU|ASG|FEE|TCH|CLS)-\d+\b")


def exercised_tools(scenario) -> list[str]:
    return [call["tool"] for call in oracle_calls(scenario)]


def distractor_tools(scenario) -> list[str]:
    return list((scenario.expect or {}).get("forbidden_tools") or [])


def entities(scenario) -> set[str]:
    text = repr(scenario.expect) + repr(scenario.ground_truth) + scenario.prompt
    found = set(_ENTITY.findall(text))
    for spec in (scenario.expect or {}).get("required_calls") or []:
        value = (spec.get("args") or {}).get("class_name")
        if isinstance(value, str):
            found.add(f"CLASS-{value.upper()}")
    return found


def is_write_scenario(scenario) -> bool:
    expect = scenario.expect or {}
    if expect.get("expected_writes"):
        return True
    return any(tool in WRITE_TOOL_NAMES for tool in exercised_tools(scenario))


def touches_write_surface(scenario) -> bool:
    """Write tools are either exercised, expected, or held out as forbidden."""
    if is_write_scenario(scenario):
        return True
    return any(tool in WRITE_TOOL_NAMES for tool in distractor_tools(scenario))


def signature(scenario) -> tuple:
    return (
        scenario.category,
        tuple(exercised_tools(scenario)),
        frozenset(entities(scenario)),
    )


def _jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def answer_tokens(scenario) -> set[str]:
    """What the scenario asserts about the answer - the differentiator when no tool runs."""
    expect = scenario.expect or {}
    tokens = {str(t).casefold() for t in expect.get("answer_must_contain") or []}
    for group in expect.get("answer_must_contain_any") or []:
        tokens |= {str(t).casefold() for t in group}
    tokens |= {str(n) for n in expect.get("answer_must_contain_numbers") or []}
    tokens |= {
        str(t).casefold()
        for t in (expect.get("clarification") or {}).get("must_mention") or []
    }
    return tokens


def redundancy_pairs(scenarios, threshold: float = 0.75) -> list[dict]:
    """Pairs testing substantially the same thing: same tools, entities, and assertions.

    Answer tokens matter because two scenarios that call no tool at all have identical
    tool and entity sets by construction, and are told apart only by what they assert.
    """
    pairs = []
    for first, second in itertools.combinations(scenarios, 2):
        if first.category != second.category:
            continue
        tools_a, tools_b = set(exercised_tools(first)), set(exercised_tools(second))
        score = (
            0.45 * _jaccard(tools_a, tools_b)
            + 0.30 * _jaccard(entities(first), entities(second))
            + 0.25 * _jaccard(answer_tokens(first), answer_tokens(second))
        )
        if score >= threshold:
            pairs.append(
                {
                    "a": first.id,
                    "b": second.id,
                    "category": first.category,
                    "similarity": round(score, 3),
                    "shared_tools": sorted(tools_a & tools_b),
                    "shared_entities": sorted(entities(first) & entities(second)),
                    "shared_assertions": sorted(
                        answer_tokens(first) & answer_tokens(second)
                    ),
                }
            )
    return sorted(pairs, key=lambda p: -p["similarity"])


def analyze(scenarios) -> dict:
    by_category: dict[str, int] = {}
    per_tool: dict[str, list[str]] = {tool: [] for tool in sorted(TOOL_NAMES)}
    distractor_per_tool: dict[str, list[str]] = {tool: [] for tool in sorted(TOOL_NAMES)}
    steps: dict[str, list[str]] = {"zero": [], "single": [], "multi": []}

    write_scenarios, clarify, refuse, adversarial, faulted, no_tool = [], [], [], [], [], []

    for scenario in scenarios:
        expect = scenario.expect or {}
        by_category[scenario.category] = by_category.get(scenario.category, 0) + 1

        calls = exercised_tools(scenario)
        for tool in set(calls):
            per_tool[tool].append(scenario.id)
        for tool in set(distractor_tools(scenario)):
            distractor_per_tool[tool].append(scenario.id)

        if not calls:
            steps["zero"].append(scenario.id)
        elif len(calls) == 1:
            steps["single"].append(scenario.id)
        else:
            steps["multi"].append(scenario.id)

        if is_write_scenario(scenario):
            write_scenarios.append(scenario.id)
        if expect.get("must_clarify"):
            clarify.append(scenario.id)
        if expect.get("must_refuse"):
            refuse.append(scenario.id)
        if scenario.category == "adversarial_safety":
            adversarial.append(scenario.id)
        if scenario.faults:
            faulted.append(scenario.id)
        if expect.get("expect_no_tools"):
            no_tool.append(scenario.id)

    read_only = [
        s.id
        for s in scenarios
        if not touches_write_surface(s) and exercised_tools(s)
    ]

    return {
        "total": len(scenarios),
        "by_category": by_category,
        "read_only_scenarios": read_only,
        "write_surface_scenarios": [s.id for s in scenarios if touches_write_surface(s)],
        "write_executing_scenarios": write_scenarios,
        "per_tool": per_tool,
        "distractor_per_tool": distractor_per_tool,
        "never_exercised": [t for t, ids in per_tool.items() if not ids],
        "never_referenced": [
            t
            for t in sorted(TOOL_NAMES)
            if not per_tool[t] and not distractor_per_tool[t]
        ],
        "steps": steps,
        "clarification_required": clarify,
        "must_refuse": refuse,
        "adversarial": adversarial,
        "fault_injected": faulted,
        "no_tool_expected": no_tool,
        "read_tools": sorted(READ_TOOL_NAMES),
        "write_tools": sorted(WRITE_TOOL_NAMES),
        "redundancy": redundancy_pairs(scenarios),
    }
