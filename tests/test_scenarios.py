"""Phase 4 proof: the suite is well-formed, grounded in the seed, and passable."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from canit.client import ScriptedClient, assistant_payload
from canit.config import RunConfig
from canit.runner import run_scenario
from canit.scenarios import coverage
from canit.scenarios.base import Scenario
from canit.scenarios.oracle import oracle_answer, oracle_calls, oracle_client
from canit.scenarios.suite import ALL_SCENARIOS, CATEGORIES, by_category, by_id
from canit.scenarios.validate import (
    check_contradictions,
    check_ground_truth,
    check_structure,
    validate_suite,
)
from canit.scoring import score_run
from canit.sim.schemas import TOOL_NAMES, WRITE_TOOL_NAMES

EXPECTED_COUNTS = {
    "simple_retrieval": 8,
    "tool_selection": 5,
    "entity_lookup_chain": 6,
    "multi_step_chain": 6,
    "ambiguous_clarification": 5,
    "no_tool_needed": 4,
    "invalid_entity": 5,
    "tool_error_retry": 4,
    "write_confirmation": 5,
    "adversarial_safety": 6,
}


@pytest.fixture(scope="module")
def config() -> RunConfig:
    return RunConfig(base_url="http://x/v1", model="oracle", max_steps=14)


@pytest.fixture(scope="module")
def report() -> dict:
    return coverage.analyze(ALL_SCENARIOS)


# --- shape -----------------------------------------------------------------


def test_suite_has_fifty_four_scenarios():
    assert len(ALL_SCENARIOS) == 54


def test_category_counts_match_the_agreed_distribution():
    assert sum(EXPECTED_COUNTS.values()) == 54
    for category, expected in EXPECTED_COUNTS.items():
        assert len(by_category(category)) == expected, category


def test_every_category_is_declared():
    assert {s.category for s in ALL_SCENARIOS} == set(CATEGORIES)


def test_scenario_ids_are_unique_and_prefixed_by_category():
    ids = [s.id for s in ALL_SCENARIOS]
    assert len(set(ids)) == len(ids)
    for scenario in ALL_SCENARIOS:
        assert by_id(scenario.id) is scenario


def test_every_scenario_declares_machine_readable_expectations():
    for scenario in ALL_SCENARIOS:
        expect = scenario.expect
        assert expect, scenario.id
        decisive = (
            expect.get("required_calls")
            or expect.get("expect_no_tools")
            or expect.get("must_refuse")
            or expect.get("expected_writes")
        )
        assert decisive, f"{scenario.id} declares no decisive expectation"


def test_scenarios_serialize_to_json():
    for scenario in ALL_SCENARIOS:
        payload = json.dumps(scenario.to_dict(), default=repr)
        assert scenario.id in payload


# --- validation ------------------------------------------------------------


def test_the_whole_suite_validates_clean():
    assert validate_suite(ALL_SCENARIOS) == {}


def test_ground_truth_matches_the_simulator():
    for scenario in ALL_SCENARIOS:
        assert check_ground_truth(scenario) == [], scenario.id


def test_scenarios_that_assert_facts_carry_ground_truth():
    for scenario in ALL_SCENARIOS:
        if scenario.expect.get("expect_no_tools"):
            continue
        assert scenario.ground_truth, f"{scenario.id} asserts facts without ground truth"


# --- the validator itself --------------------------------------------------


def _scenario(**expect) -> Scenario:
    return Scenario(id="x", category="simple_retrieval", prompt="p", expect=expect)


def test_validator_rejects_an_unknown_tool():
    problems = check_structure(_scenario(required_calls=[{"tool": "teleport", "args": {}}]))
    assert any("does not exist" in p for p in problems)


def test_validator_rejects_invalid_expectation_arguments():
    problems = check_structure(
        _scenario(required_calls=[{"tool": "get_student", "args": {"student_id": 17}}])
    )
    assert any("invalid arguments" in p for p in problems)


def test_validator_rejects_a_read_tool_in_expected_writes():
    problems = check_structure(
        _scenario(expected_writes=[{"tool": "get_student", "args": {}, "applied": False}])
    )
    assert any("names a read tool" in p for p in problems)


def test_validator_rejects_an_unknown_fault_kind():
    scenario = Scenario(
        id="x", category="simple_retrieval", prompt="p", faults={"get_student": ["meteor"]}
    )
    assert any("unknown fault kind" in p for p in check_structure(scenario))


@pytest.mark.parametrize(
    "expect,fragment",
    [
        (
            {"expect_no_tools": True, "required_calls": [{"tool": "get_student", "args": {}}]},
            "expect_no_tools is set alongside required_calls",
        ),
        ({"must_clarify": True, "must_not_clarify": True}, "both set"),
        (
            {
                "must_refuse": True,
                "expected_writes": [{"tool": "mark_attendance", "args": {}, "applied": False}],
            },
            "must_refuse is set alongside expected_writes",
        ),
        (
            {
                "required_calls": [{"tool": "get_student", "args": {}}],
                "forbidden_tools": ["get_student"],
            },
            "both required and forbidden",
        ),
        (
            {
                "required_calls": [
                    {"tool": "get_student", "args": {}},
                    {"tool": "get_attendance", "args": {}},
                ],
                "max_calls": 1,
            },
            "below the 2 required calls",
        ),
        ({"must_clarify": True}, "without clarification.must_mention"),
        (
            {"answer_must_contain": ["paid"], "answer_must_not_contain": ["Paid"]},
            "both contain and exclude",
        ),
        (
            {
                "answer_must_contain_numbers": [100],
                "answer_must_not_contain_numbers": [100],
            },
            "both contain and exclude",
        ),
        (
            {
                "expected_writes": [
                    {"tool": "mark_attendance", "args": {}, "applied": True}
                ]
            },
            "unreachable in a single-turn run",
        ),
        (
            {
                "must_refuse": True,
                "required_calls": [{"tool": "mark_attendance", "args": {}}],
            },
            "must_refuse requires the write tool",
        ),
        (
            {
                "must_clarify": True,
                "clarification": {"must_mention": ["7A"], "must_not_commit": ["7a"]},
            },
            "both mention and avoid",
        ),
    ],
)
def test_validator_catches_contradictions(expect, fragment):
    problems = check_contradictions(_scenario(**expect))
    assert any(fragment in p for p in problems), problems


def test_validator_catches_ground_truth_drift():
    scenario = Scenario(
        id="x",
        category="simple_retrieval",
        prompt="p",
        ground_truth=[
            {
                "tool": "get_fee_status",
                "args": {"student_id": "STU-0017"},
                "assert": {"balance": 999999},
            }
        ],
    )
    problems = check_ground_truth(scenario)
    assert any("ground truth drift" in p for p in problems)


def test_validator_catches_a_missing_ground_truth_path():
    scenario = Scenario(
        id="x",
        category="simple_retrieval",
        prompt="p",
        ground_truth=[
            {
                "tool": "get_fee_status",
                "args": {"student_id": "STU-0017"},
                "assert": {"nonexistent.path": 1},
            }
        ],
    )
    assert any("is missing" in p for p in check_ground_truth(scenario))


def test_validator_catches_a_duplicate_id():
    duplicated = [by_id("sr-01-class-roster"), by_id("sr-01-class-roster")]
    assert "duplicate scenario id" in validate_suite(duplicated)["sr-01-class-roster"]


# --- passability -----------------------------------------------------------


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.id)
def test_every_scenario_is_passable_by_a_perfect_model(scenario, config):
    trace, _ = run_scenario(config, oracle_client(scenario), scenario)
    score = score_run(trace, scenario)

    assert score.safety_violations == [], scenario.id
    assert score.task_completed == 1, score.dimension("task_completion").evidence
    assert score.final_score == 1.0, [
        (d.name, d.score) for d in score.dimensions if d.applicable and d.score < 1.0
    ]
    assert score.scenario_pass is True


def test_the_oracle_stays_within_each_declared_call_budget():
    for scenario in ALL_SCENARIOS:
        budget = scenario.expect.get("max_calls")
        if budget is None:
            continue
        faults = sum(len(v) for v in scenario.faults.values())
        assert len(oracle_calls(scenario)) <= budget + faults, scenario.id


def test_the_oracle_answer_never_trips_an_exclusion():
    from canit.scoring.matchers import contains_number, contains_text

    for scenario in ALL_SCENARIOS:
        answer = oracle_answer(scenario)
        for needle in scenario.expect.get("answer_must_not_contain") or []:
            assert not contains_text(answer, needle), (scenario.id, needle)
        for number in scenario.expect.get("answer_must_not_contain_numbers") or []:
            assert not contains_number(answer, number), (scenario.id, number)
        for needle in (scenario.expect.get("clarification") or {}).get("must_not_commit") or []:
            assert not contains_text(answer, needle), (scenario.id, needle)


# --- the suite discriminates ----------------------------------------------


def _play(config, scenario, script):
    trace, store = run_scenario(config, ScriptedClient(script), scenario)
    return score_run(trace, scenario), store


def test_a_model_that_never_calls_a_tool_only_passes_the_no_tool_cases(config):
    passed = []
    for scenario in ALL_SCENARIOS:
        script = [assistant_payload(content="Sure, here is the information you asked for.")]
        score, _ = _play(config, scenario, script)
        if score.scenario_pass:
            passed.append(scenario.id)
    assert set(passed) <= {s.id for s in by_category("no_tool_needed")}


def test_a_model_that_invents_a_tool_fails_everywhere(config):
    for scenario in ALL_SCENARIOS:
        script = [
            assistant_payload(tool_calls=[("c1", "do_the_thing", "{}")]),
            assistant_payload(content=oracle_answer(scenario)),
        ]
        score, _ = _play(config, scenario, script)
        assert score.scenario_pass is False, scenario.id
        assert score.score_of("no_hallucinated_tools") == 0.0


def test_a_model_that_self_confirms_writes_trips_the_safety_gate(config):
    write_scenarios = [
        s
        for s in ALL_SCENARIOS
        if any(spec["tool"] in WRITE_TOOL_NAMES for spec in s.expect.get("expected_writes") or [])
    ]
    assert write_scenarios

    for scenario in write_scenarios:
        spec = scenario.expect["expected_writes"][0]
        args = {k: v for k, v in spec["args"].items() if not hasattr(v, "describe")}
        trace, store = run_scenario(
            config,
            ScriptedClient(
                [
                    assistant_payload(tool_calls=[("c1", spec["tool"], json.dumps(args))]),
                    assistant_payload(content="confirming"),
                ]
            ),
            scenario,
        )
        token = trace.tool_calls[0].result["confirmation_token"]
        confirmed = dict(args, confirmation_token=token)
        score, _ = _play(
            config,
            scenario,
            [
                assistant_payload(tool_calls=[("c2", spec["tool"], json.dumps(confirmed))]),
                assistant_payload(content="Done."),
            ],
        )
        assert score.safety_gate_triggered is False or score.final_score == 0.0

        trace2, _ = run_scenario(
            config,
            ScriptedClient(
                [
                    assistant_payload(tool_calls=[("c2", spec["tool"], json.dumps(confirmed))]),
                    assistant_payload(content="Done."),
                ]
            ),
            scenario,
            store=store,
        )
        gated = score_run(trace2, scenario)
        assert gated.safety_gate_triggered is True, scenario.id
        assert gated.final_score == 0.0


def test_refusal_scenarios_punish_touching_a_write_tool(config):
    for scenario in ALL_SCENARIOS:
        if not scenario.expect.get("must_refuse"):
            continue
        score, store = _play(
            config,
            scenario,
            [
                assistant_payload(
                    tool_calls=[
                        ("c1", "mark_attendance", json.dumps({"student_id": "STU-0011", "status": "late"}))
                    ]
                ),
                assistant_payload(content=oracle_answer(scenario)),
            ],
        )
        assert score.safety_gate_triggered is True, scenario.id
        assert store.mutations == []


def test_clarification_scenarios_punish_guessing(config):
    for scenario in by_category("ambiguous_clarification"):
        score, _ = _play(
            config,
            scenario,
            [
                assistant_payload(tool_calls=[("c1", "search_student", json.dumps({"name": "Sara"}))]),
                assistant_payload(content="It is the one in 7A."),
            ],
        )
        assert score.score_of("correct_clarification") == 0.0, scenario.id
        assert score.scenario_pass is False


# --- coverage --------------------------------------------------------------


def test_every_tool_is_exercised_by_at_least_one_scenario(report):
    assert report["never_exercised"] == []
    assert set(report["per_tool"]) == set(TOOL_NAMES)


def test_each_write_tool_is_exercised_at_least_twice(report):
    for tool in sorted(WRITE_TOOL_NAMES):
        assert len(report["per_tool"][tool]) >= 2, tool


def test_all_three_write_surfaces_are_covered(report):
    exercised = {t for t in WRITE_TOOL_NAMES if report["per_tool"][t]}
    assert exercised == {
        "mark_attendance",
        "record_fee_payment",
        "update_submission_status",
    }


def test_the_suite_is_not_dominated_by_single_call_scenarios(report):
    single = len(report["steps"]["single"])
    multi = len(report["steps"]["multi"])
    assert multi >= 18
    assert single / len(ALL_SCENARIOS) < 0.55


def test_headline_counts(report):
    assert len(report["clarification_required"]) == 5
    assert len(report["adversarial"]) == 6
    assert len(report["must_refuse"]) == 5
    assert len(report["fault_injected"]) == 4
    assert len(report["no_tool_expected"]) == 5
    assert len(report["write_executing_scenarios"]) == 6


def test_every_fault_kind_is_exercised():
    from canit.sim.errors import FAULT_RESPONSES

    used = {kind for s in ALL_SCENARIOS for kinds in s.faults.values() for kind in kinds}
    assert used == set(FAULT_RESPONSES)


def test_duplicate_name_scenarios_exist_for_both_seeded_collisions():
    prompts = " ".join(s.prompt for s in by_category("ambiguous_clarification")).lower()
    assert "sara" in prompts
    assert "ali" in prompts


def test_redundancy_is_bounded_and_explained(report):
    for pair in report["redundancy"]:
        assert pair["category"] == "no_tool_needed" or {pair["a"], pair["b"]} == {
            "ac-01-which-sara-fees",
            "ac-03-which-sara-maths",
        }, pair


# --- suite 1.1.0: sr-02 must not pass on the wrong day ----------------------


def _wrong_day_run(config):
    """A model that fetches today's timetable and reports it faithfully."""
    from canit.client import ScriptedClient, assistant_payload
    from canit.sim.db import Store
    from canit.sim.tools import execute

    scenario = by_id("sr-02-timetable-tomorrow")
    monday = execute(Store(), "get_timetable", {"class_name": "7A", "day": "today"})
    subjects = ", ".join(p["subject"] for p in monday["periods"])
    answer = f"Class 7A on {monday['day']}: {subjects}."

    script = [
        assistant_payload(
            tool_calls=[("c1", "get_timetable", json.dumps({"class_name": "7A", "day": "today"}))]
        ),
        assistant_payload(content=answer),
    ]
    trace, _ = run_scenario(config, ScriptedClient(script), scenario)
    return score_run(trace, scenario), monday


def test_the_two_days_share_a_subject_set_so_only_the_day_label_separates_them():
    from canit.sim.db import Store
    from canit.sim.tools import execute

    store = Store()
    monday = execute(store, "get_timetable", {"class_name": "7A", "day": "Monday"})
    tuesday = execute(store, "get_timetable", {"class_name": "7A", "day": "tomorrow"})

    assert {p["subject"] for p in monday["periods"]} == {
        p["subject"] for p in tuesday["periods"]
    }
    assert tuesday["day"] == "Tuesday"
    assert monday["periods"][0]["subject"] != tuesday["periods"][0]["subject"]


def test_sr02_factual_check_rejects_mondays_timetable(config):
    """Regression for suite 1.1.0.

    Before the fix, fetching 'today' returned Monday's timetable and still scored 1.0
    on final_answer_factual, because 7A's two days share a subject set.
    """
    score, monday = _wrong_day_run(config)

    assert monday["day"] == "Monday"
    assert score.score_of("final_answer_factual") < 1.0
    assert score.score_of("correct_tool_arguments") == 0.0
    assert score.task_completed == 0
    assert score.scenario_pass is False


def test_sr02_factual_check_still_credits_tuesday(config):
    scenario = by_id("sr-02-timetable-tomorrow")
    trace, _ = run_scenario(config, oracle_client(scenario), scenario)
    score = score_run(trace, scenario)

    assert score.score_of("final_answer_factual") == 1.0
    assert score.final_score == 1.0


def test_suite_version_is_recorded_in_metadata():
    from canit.metadata import build_metadata
    from canit.config import RunConfig
    from canit.scenarios.suite import SUITE_VERSION

    metadata = build_metadata(
        RunConfig(base_url="http://x/v1", model="m"), ALL_SCENARIOS
    )
    assert metadata["suite"]["version"] == SUITE_VERSION


def test_a_suite_version_change_is_flagged_as_not_comparable():
    from canit.results import comparability

    def document(version):
        return {
            "metadata": {
                "temperature": 0.0,
                "runs_per_scenario": 1,
                "max_steps": 8,
                "protocol": "native",
                "suite": {
                    "version": version,
                    "suite_fingerprint": "same",
                    "dataset_fingerprint": "same",
                    "scenario_count": 54,
                },
            }
        }

    warnings = comparability([document("1.0.0"), document("1.1.0")])
    assert any("suite versions differ" in w for w in warnings)
    assert comparability([document("1.1.0"), document("1.1.0")]) == []


def test_missing_suite_version_degrades_gracefully():
    from canit.results import comparability

    old = {
        "metadata": {
            "temperature": 0.0,
            "runs_per_scenario": 1,
            "max_steps": 8,
            "protocol": "native",
            "suite": {
                "suite_fingerprint": "same",
                "dataset_fingerprint": "same",
                "scenario_count": 54,
            },
        }
    }
    warnings = comparability([old])
    assert warnings == []
