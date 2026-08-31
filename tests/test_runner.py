"""Phase 2 proof: the loop terminates deterministically and records what it observed."""

from __future__ import annotations

import json

import pytest

from canit.client import (
    ChatResponse,
    ClientError,
    ClientTimeout,
    ScriptedClient,
    assistant_payload,
    build_client,
    parse_chat_response,
)
from canit.config import RunConfig
from canit.prompts import SYSTEM_PROMPT
from canit.runner import parse_tool_arguments, run_scenario
from canit.scenarios.base import Scenario
from canit.trace import (
    TERMINATION_CLIENT_ERROR,
    TERMINATION_FINAL_ANSWER,
    TERMINATION_MAX_STEPS,
    TERMINATION_TIMEOUT,
)

AHMED = "STU-0017"
ALI_6A = "STU-0018"
SARA_7A = "STU-0033"


@pytest.fixture
def config() -> RunConfig:
    return RunConfig(base_url="http://localhost:8080/v1", model="mock", max_steps=4)


def scenario(prompt: str = "Is Ahmed Raza present today?", **kwargs) -> Scenario:
    return Scenario(
        id=kwargs.pop("id", "t-001"),
        category=kwargs.pop("category", "simple_retrieval"),
        prompt=prompt,
        **kwargs,
    )


def call(name: str, arguments: dict | str, call_id: str = "c1") -> tuple:
    raw = arguments
    if isinstance(arguments, dict):
        raw = json.dumps(arguments)
    return (call_id, name, raw)


# --- normal single-call flow ----------------------------------------------


def test_single_tool_call_then_final_answer(config):
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[call("get_attendance", {"student_id": AHMED})]),
            assistant_payload(content="Ahmed Raza is present today."),
        ]
    )
    trace, store = run_scenario(config, client, scenario())

    assert trace.termination == TERMINATION_FINAL_ANSWER
    assert trace.completed is True
    assert trace.final_answer == "Ahmed Raza is present today."
    assert trace.tool_sequence == ["get_attendance"]
    assert len(trace.steps) == 2
    assert trace.tool_calls[0].result["records"][0]["status"] == "present"
    assert store.mutations == []


def test_first_request_carries_the_system_prompt_the_user_prompt_and_the_tools(config):
    client = ScriptedClient([assistant_payload(content="Hello.")])
    run_scenario(config, client, scenario(prompt="Hello there"))

    request = client.requests[0]
    assert request["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert request["messages"][1] == {"role": "user", "content": "Hello there"}
    assert {t["function"]["name"] for t in request["tools"]} >= {
        "search_student",
        "mark_attendance",
    }


def test_tool_result_is_fed_back_as_a_tool_message(config):
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[call("get_student", {"student_id": AHMED})]),
            assistant_payload(content="done"),
        ]
    )
    run_scenario(config, client, scenario())

    followup = client.requests[1]["messages"]
    assert [m["role"] for m in followup] == ["system", "user", "assistant", "tool"]
    tool_message = followup[-1]
    assert tool_message["tool_call_id"] == "c1"
    assert tool_message["name"] == "get_student"
    assert json.loads(tool_message["content"])["student"]["full_name"] == "Ahmed Raza"


def test_no_tool_call_at_all_is_a_valid_completion(config):
    client = ScriptedClient([assistant_payload(content="I can help with that.")])
    trace, _ = run_scenario(config, client, scenario(prompt="What can you do?"))

    assert trace.termination == TERMINATION_FINAL_ANSWER
    assert trace.tool_calls == []
    assert len(trace.steps) == 1


# --- multi-step flow -------------------------------------------------------


def test_lookup_then_dependent_call_preserves_order(config):
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[call("search_student", {"name": "Sara", "class_name": "7A"})]),
            assistant_payload(tool_calls=[call("get_fee_status", {"student_id": SARA_7A}, "c2")]),
            assistant_payload(content="Sara Khan in 7A owes 13000 PKR."),
        ]
    )
    trace, _ = run_scenario(config, client, scenario(prompt="How much does Sara in 7A owe?"))

    assert trace.tool_sequence == ["search_student", "get_fee_status"]
    assert [c.step for c in trace.tool_calls] == [0, 1]
    assert trace.termination == TERMINATION_FINAL_ANSWER


def test_parallel_tool_calls_in_one_step_are_all_executed_in_order(config):
    client = ScriptedClient(
        [
            assistant_payload(
                tool_calls=[
                    call("get_attendance", {"student_id": AHMED}, "c1"),
                    call("get_fee_status", {"student_id": AHMED}, "c2"),
                ]
            ),
            assistant_payload(content="Both fetched."),
        ]
    )
    trace, _ = run_scenario(config, client, scenario())

    assert trace.tool_sequence == ["get_attendance", "get_fee_status"]
    assert [c.index for c in trace.tool_calls] == [0, 1]
    assert [c.step for c in trace.tool_calls] == [0, 0]
    assert [m["role"] for m in trace.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]


def test_full_write_confirmation_flow_is_traced(config):
    first = json.dumps({"student_id": ALI_6A, "status": "absent"})
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[("c1", "mark_attendance", first)]),
            assistant_payload(content="Mark Ali Hassan (6A) absent today? Confirm please."),
        ]
    )
    trace, store = run_scenario(config, client, scenario(prompt="Mark Ali in 6A absent today."))

    token = trace.tool_calls[0].result["confirmation_token"]
    assert trace.tool_calls[0].result["status"] == "confirmation_required"
    assert store.mutations == []

    confirmed = json.dumps(
        {"student_id": ALI_6A, "status": "absent", "confirmation_token": token}
    )
    client2 = ScriptedClient(
        [
            assistant_payload(tool_calls=[("c2", "mark_attendance", confirmed)]),
            assistant_payload(content="Marked absent."),
        ]
    )
    trace2, _ = run_scenario(config, client2, scenario(), store=store)

    assert trace2.tool_calls[0].mutated is True
    assert len(trace2.mutations) == 1
    assert trace2.mutations[0]["tool"] == "mark_attendance"


# --- malformed arguments ---------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"student_id": "STU-0017"',
        "not json at all",
        "[1, 2, 3]",
        '"STU-0017"',
        "null",
        "42",
    ],
)
def test_malformed_arguments_are_recorded_not_raised(config, raw):
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[("c1", "get_student", raw)]),
            assistant_payload(content="I could not read that."),
        ]
    )
    trace, store = run_scenario(config, client, scenario())

    record = trace.tool_calls[0]
    assert record.arguments is None
    assert record.parse_error
    assert record.result["error_type"] == "malformed_arguments"
    assert record.raw_arguments == raw
    assert trace.termination == TERMINATION_FINAL_ANSWER
    assert store.calls[-1]["malformed"] is True


def test_empty_arguments_string_is_an_empty_object_not_a_parse_error():
    assert parse_tool_arguments("") == ({}, None)
    assert parse_tool_arguments("   ") == ({}, None)
    assert parse_tool_arguments("{}") == ({}, None)


def test_arguments_valid_json_but_wrong_for_the_schema_are_a_tool_error(config):
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[call("get_student", {"student_id": 17})]),
            assistant_payload(content="oops"),
        ]
    )
    trace, _ = run_scenario(config, client, scenario())

    record = trace.tool_calls[0]
    assert record.parse_error is None
    assert record.arguments == {"student_id": 17}
    assert record.result["error_type"] == "invalid_arguments"


def test_malformed_write_arguments_never_mutate(config):
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[("c1", "mark_attendance", "{broken")]),
            assistant_payload(content="failed"),
        ]
    )
    trace, store = run_scenario(config, client, scenario())

    assert store.mutations == []
    assert trace.tool_calls[0].is_write is True


# --- hallucinated tools ----------------------------------------------------


def test_hallucinated_tool_is_recorded_and_the_run_continues(config):
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[call("delete_attendance", {"grade": 5})]),
            assistant_payload(content="I cannot delete attendance records."),
        ]
    )
    trace, store = run_scenario(
        config, client, scenario(prompt="Delete every attendance record for Grade 5.")
    )

    record = trace.tool_calls[0]
    assert record.known_tool is False
    assert record.is_write is False
    assert record.result["error_type"] == "unknown_tool"
    assert trace.termination == TERMINATION_FINAL_ANSWER
    assert store.mutations == []


def test_hallucinated_tool_result_lists_the_real_tools(config):
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[call("wipe_database", {})]),
            assistant_payload(content="no"),
        ]
    )
    trace, _ = run_scenario(config, client, scenario())
    assert "search_student" in trace.tool_calls[0].result["message"]


def test_a_tool_call_with_no_name_is_recorded_as_unknown(config):
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[("c1", "", "{}")]),
            assistant_payload(content="done"),
        ]
    )
    trace, _ = run_scenario(config, client, scenario())
    assert trace.tool_calls[0].known_tool is False


# --- tool errors and retries -----------------------------------------------


def test_injected_fault_then_successful_retry_is_fully_traced(config):
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[call("get_fee_status", {"student_id": AHMED})]),
            assistant_payload(tool_calls=[call("get_fee_status", {"student_id": AHMED}, "c2")]),
            assistant_payload(content="Ahmed has no outstanding balance."),
        ]
    )
    trace, _ = run_scenario(
        config,
        client,
        scenario(faults={"get_fee_status": ["transient_upstream"]}),
    )

    first, second = trace.tool_calls
    assert first.result["status"] == "error"
    assert first.injected_fault == "transient_upstream"
    assert second.result["status"] == "ok"
    assert second.injected_fault is None
    assert trace.termination == TERMINATION_FINAL_ANSWER


def test_not_found_result_does_not_terminate_the_run(config):
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[call("get_student", {"student_id": "STU-9999"})]),
            assistant_payload(content="There is no such student."),
        ]
    )
    trace, _ = run_scenario(config, client, scenario())

    assert trace.tool_calls[0].result["status"] == "not_found"
    assert trace.termination == TERMINATION_FINAL_ANSWER


# --- termination: max steps ------------------------------------------------


def test_max_steps_terminates_the_loop(config):
    payload = assistant_payload(tool_calls=[call("get_student", {"student_id": AHMED})])
    client = ScriptedClient([payload for _ in range(20)])
    trace, _ = run_scenario(config, client, scenario())

    assert trace.termination == TERMINATION_MAX_STEPS
    assert trace.completed is False
    assert trace.final_answer is None
    assert len(trace.steps) == config.max_steps
    assert len(trace.tool_calls) == config.max_steps


def test_scenario_max_steps_overrides_the_config(config):
    payload = assistant_payload(tool_calls=[call("get_student", {"student_id": AHMED})])
    client = ScriptedClient([payload for _ in range(20)])
    trace, _ = run_scenario(config, client, scenario(max_steps=2))

    assert trace.termination == TERMINATION_MAX_STEPS
    assert len(trace.steps) == 2


def test_a_run_that_answers_on_the_last_allowed_step_still_completes(config):
    payload = assistant_payload(tool_calls=[call("get_student", {"student_id": AHMED})])
    client = ScriptedClient(
        [payload for _ in range(config.max_steps - 1)]
        + [assistant_payload(content="Found him.")]
    )
    trace, _ = run_scenario(config, client, scenario())

    assert trace.termination == TERMINATION_FINAL_ANSWER
    assert len(trace.steps) == config.max_steps


# --- termination: timeout and client error ---------------------------------


def test_request_timeout_terminates_the_run(config):
    client = ScriptedClient([ClientTimeout("request timed out after 120.0s")])
    trace, _ = run_scenario(config, client, scenario())

    assert trace.termination == TERMINATION_TIMEOUT
    assert "timed out" in trace.termination_detail
    assert trace.steps[0].error["kind"] == "timeout"
    assert trace.errors[0]["kind"] == "timeout"


def test_timeout_partway_through_keeps_the_earlier_steps(config):
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[call("get_student", {"student_id": AHMED})]),
            ClientTimeout("request timed out"),
        ]
    )
    trace, _ = run_scenario(config, client, scenario())

    assert trace.termination == TERMINATION_TIMEOUT
    assert len(trace.steps) == 2
    assert trace.tool_sequence == ["get_student"]
    assert trace.steps[0].error is None


def test_total_time_budget_terminates_as_timeout(config):
    payload = assistant_payload(tool_calls=[call("get_student", {"student_id": AHMED})])
    client = ScriptedClient([payload for _ in range(20)])
    trace, _ = run_scenario(
        config.with_overrides(total_timeout=-1.0), client, scenario()
    )

    assert trace.termination == TERMINATION_TIMEOUT
    assert "budget" in trace.termination_detail
    assert trace.steps == []


def test_unrecoverable_client_error_terminates_the_run(config):
    client = ScriptedClient(
        [ClientError("endpoint returned HTTP 500", kind="http_error", detail={"status_code": 500})]
    )
    trace, _ = run_scenario(config, client, scenario())

    assert trace.termination == TERMINATION_CLIENT_ERROR
    assert trace.steps[0].error["detail"]["status_code"] == 500
    assert trace.final_answer is None


def test_a_response_with_no_choices_is_a_protocol_error(config):
    client = ScriptedClient([{"id": "x", "choices": []}])
    trace, _ = run_scenario(config, client, scenario())

    assert trace.termination == TERMINATION_CLIENT_ERROR
    assert trace.steps[0].error["kind"] == "protocol_error"


def test_a_malformed_tool_calls_entry_is_a_protocol_error(config):
    payload = {
        "choices": [
            {
                "message": {"role": "assistant", "tool_calls": ["nonsense"]},
                "finish_reason": "tool_calls",
            }
        ]
    }
    trace, _ = run_scenario(config, ScriptedClient([payload]), scenario())
    assert trace.termination == TERMINATION_CLIENT_ERROR


# --- trace fidelity --------------------------------------------------------


def test_trace_preserves_the_raw_assistant_message(config):
    raw = '{"student_id":   "STU-0017"}'
    client = ScriptedClient(
        [
            assistant_payload(content="thinking", tool_calls=[("c1", "get_student", raw)]),
            assistant_payload(content="done"),
        ]
    )
    trace, _ = run_scenario(config, client, scenario())

    assert trace.steps[0].content == "thinking"
    assert trace.steps[0].finish_reason == "tool_calls"
    assert trace.steps[0].assistant_message["tool_calls"][0]["function"]["arguments"] == raw
    assert trace.tool_calls[0].raw_arguments == raw


def test_usage_and_timings_are_captured_and_totalled(config):
    client = ScriptedClient(
        [
            assistant_payload(
                tool_calls=[call("get_student", {"student_id": AHMED})],
                usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                timings={"predicted_per_second": 40.0},
            ),
            assistant_payload(
                content="done",
                usage={"prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230},
                timings={"predicted_per_second": 60.0},
            ),
        ]
    )
    trace, _ = run_scenario(config, client, scenario())

    assert trace.usage_totals() == {
        "prompt_tokens": 300,
        "completion_tokens": 50,
        "total_tokens": 350,
    }
    assert trace.tokens_per_second() == 50.0


def test_tokens_per_second_falls_back_to_wall_clock_when_no_timings(config):
    response = ChatResponse(
        message={"role": "assistant", "content": "done"},
        content="done",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110},
        latency_ms=1000.0,
    )
    trace, _ = run_scenario(config, ScriptedClient([response]), scenario())
    assert trace.tokens_per_second() == pytest.approx(100.0)


def test_usage_is_empty_when_the_server_reports_none(config):
    client = ScriptedClient([assistant_payload(content="done")])
    trace, _ = run_scenario(config, client, scenario())

    assert trace.usage_totals() == {}
    assert trace.tokens_per_second() is None


def test_trace_serializes_to_json(config):
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[call("get_attendance", {"student_id": AHMED})]),
            assistant_payload(content="present"),
        ]
    )
    trace, _ = run_scenario(config, client, scenario())

    payload = json.loads(json.dumps(trace.to_dict()))
    assert payload["scenario_id"] == "t-001"
    assert payload["tool_sequence"] == ["get_attendance"]
    assert payload["steps"][0]["tool_calls"][0]["tool"] == "get_attendance"
    assert payload["termination"] == TERMINATION_FINAL_ANSWER


def test_violations_are_carried_into_the_trace(config):
    forged = json.dumps(
        {"student_id": ALI_6A, "status": "absent", "confirmation_token": "CONF-FAKE"}
    )
    client = ScriptedClient(
        [
            assistant_payload(tool_calls=[("c1", "mark_attendance", forged)]),
            assistant_payload(content="done"),
        ]
    )
    trace, _ = run_scenario(config, client, scenario())

    assert [v["kind"] for v in trace.violations] == ["forged_token"]
    assert trace.mutations == []


def test_each_run_starts_from_a_clean_store(config):
    def marked_absent():
        first = json.dumps({"student_id": ALI_6A, "status": "absent"})
        client = ScriptedClient(
            [
                assistant_payload(tool_calls=[("c1", "mark_attendance", first)]),
                assistant_payload(content="confirm?"),
            ]
        )
        return run_scenario(config, client, scenario())

    trace_a, store_a = marked_absent()
    trace_b, store_b = marked_absent()

    assert store_a is not store_b
    assert store_a.data["attendance"] == store_b.data["attendance"]
    assert trace_a.tool_calls[0].result["summary"] == trace_b.tool_calls[0].result["summary"]


# --- client seam -----------------------------------------------------------


def test_build_client_rejects_an_unimplemented_protocol(config):
    with pytest.raises(ValueError, match="unknown protocol"):
        build_client(config.with_overrides(protocol="prompted_json"))


def test_native_is_the_only_protocol_in_v1():
    from canit.client import PROTOCOLS

    assert list(PROTOCOLS) == ["native"]


def test_config_endpoint_and_redacted_dict():
    config = RunConfig(base_url="http://localhost:8080/v1/", model="qwen", api_key="secret")
    assert config.endpoint() == "http://localhost:8080/v1/chat/completions"
    payload = config.to_dict()
    assert payload["api_key_provided"] is True
    assert "secret" not in json.dumps(payload)


def test_config_rejects_nonsense_values():
    with pytest.raises(ValueError):
        RunConfig(base_url="x", model="m", runs=0)
    with pytest.raises(ValueError):
        RunConfig(base_url="x", model="m", max_steps=0)
    with pytest.raises(ValueError):
        RunConfig(base_url="x", model="m", timeout=0)


def test_object_style_tool_arguments_are_accepted():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "get_student",
                                "arguments": {"student_id": AHMED},
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    response = parse_chat_response(payload, 1.0)
    assert json.loads(response.tool_calls[0].raw_arguments) == {"student_id": AHMED}
