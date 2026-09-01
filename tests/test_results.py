"""Phase 5 proof: results JSON, the CLIs, and model comparison."""

from __future__ import annotations

import json

import pytest

import benchmark as benchmark_cli
import compare as compare_cli
from canit.client import ChatResponse, ScriptedClient, parse_chat_response
from canit.comparison import build_comparison, entry_for, rank, recommendation, regressions
from canit.config import RunConfig
from canit.harness import run_suite, select_scenarios
from canit.metadata import BENCHMARK_VERSION, SCHEMA_VERSION, build_metadata, redact_url
from canit.report.compare_text import render_comparison
from canit.report.terminal import render_summary
from canit.results import (
    SchemaError,
    build_results,
    check_schema,
    comparability,
    default_output_path,
    load_results,
    save_results,
)
from canit.scenarios.oracle import oracle_client, oracle_script
from canit.scenarios.suite import ALL_SCENARIOS, by_category, by_id

SMALL_SUITE = [
    by_id("sr-01-class-roster"),
    by_id("ac-01-which-sara-fees"),
    by_id("wc-01-mark-ali-6a-absent"),
    by_id("nt-02-arithmetic"),
]


@pytest.fixture
def config() -> RunConfig:
    return RunConfig(
        base_url="http://localhost:8080/v1", model="mock", runs=2, max_steps=14
    )


def oracle_factory(scenario, run_index):
    return oracle_client(scenario)


def build_document(config, scenarios=SMALL_SUITE, factory=oracle_factory, **meta):
    scores, traces = run_suite(config, scenarios, client_factory=factory)
    metadata = build_metadata(config, scenarios, **meta)
    return build_results(metadata, scores, traces), scores, traces


# --- results document ------------------------------------------------------


def test_document_carries_every_required_section(config):
    document, scores, _ = build_document(config)

    assert document["schema_version"] == SCHEMA_VERSION
    assert document["metadata"]["benchmark_version"] == BENCHMARK_VERSION
    assert set(document) >= {"metadata", "metrics", "categories", "scenarios", "runs"}
    assert len(document["runs"]) == len(SMALL_SUITE) * config.runs
    assert len(document["runs"]) == len(scores)


def test_each_run_retains_scores_dimensions_and_the_full_trace(config):
    document, _, _ = build_document(config)

    for run in document["runs"]:
        assert set(run) >= {
            "scenario_id",
            "raw_score",
            "final_score",
            "scenario_pass",
            "safety_gate_triggered",
            "safety_violations",
            "task_completed",
            "dimensions",
            "trace",
        }
        assert len(run["dimensions"]) == 11
        for dimension in run["dimensions"]:
            assert "evidence" in dimension
            assert "weight" in dimension
            assert "applicable" in dimension

        trace = run["trace"]
        assert trace["user_prompt"]
        assert trace["messages"][0]["role"] == "system"
        assert "steps" in trace and "termination" in trace
        assert "usage_totals" in trace


def test_trace_preserves_tool_calls_and_results(config):
    document, _, _ = build_document(config)
    roster = next(r for r in document["runs"] if r["scenario_id"] == "sr-01-class-roster")
    call = roster["trace"]["steps"][0]["tool_calls"][0]

    assert call["tool"] == "get_class_students"
    assert json.loads(call["raw_arguments"])["class_name"] == "6A"
    assert call["result"]["count"] == 8
    assert call["latency_ms"] >= 0


def test_document_round_trips_through_json(config, tmp_path):
    document, _, _ = build_document(config)
    path = save_results(document, tmp_path / "run.json")
    reloaded = load_results(path)

    assert reloaded == json.loads(json.dumps(document, default=str))
    assert reloaded["metrics"]["mean_final_score"] == document["metrics"]["mean_final_score"]


def test_saving_creates_missing_directories(config, tmp_path):
    document, _, _ = build_document(config)
    path = save_results(document, tmp_path / "nested" / "deep" / "run.json")
    assert path.exists()


def test_default_output_path_is_slugged_and_timestamped(config):
    metadata = build_metadata(config.with_overrides(label="qwen3:8b / Q4"), SMALL_SUITE)
    path = default_output_path(metadata, "results")

    assert path.parent.name == "results"
    assert "/" not in path.name
    assert path.name.endswith(".json")


# --- schema compatibility --------------------------------------------------


def test_check_schema_accepts_the_current_version(config):
    document, _, _ = build_document(config)
    check_schema(document)


def test_check_schema_rejects_a_future_major():
    with pytest.raises(SchemaError, match="schema 2.0"):
        check_schema({"schema_version": "2.0", "metadata": {}, "metrics": {}, "runs": []})


def test_check_schema_accepts_a_future_minor():
    check_schema({"schema_version": "1.7", "metadata": {}, "metrics": {}, "runs": []})


def test_check_schema_rejects_a_missing_version():
    with pytest.raises(SchemaError, match="no schema_version"):
        check_schema({"metadata": {}, "metrics": {}, "runs": []})


@pytest.mark.parametrize("missing", ["metadata", "metrics", "runs"])
def test_check_schema_rejects_a_missing_section(missing):
    document = {"schema_version": "1.0", "metadata": {}, "metrics": {}, "runs": []}
    document.pop(missing)
    with pytest.raises(SchemaError, match=missing):
        check_schema(document)


def test_loading_a_bad_file_raises(tmp_path):
    path = tmp_path / "junk.json"
    path.write_text("{}")
    with pytest.raises(SchemaError):
        load_results(path)


# --- metadata --------------------------------------------------------------


def test_metadata_records_provenance(config):
    metadata = build_metadata(
        config,
        SMALL_SUITE,
        quantization="Q4_K_M",
        model_file="qwen3-8b-Q4_K_M.gguf",
        runtime="llama.cpp",
    )

    assert metadata["quantization"] == "Q4_K_M"
    assert metadata["model_file"] == "qwen3-8b-Q4_K_M.gguf"
    assert metadata["runtime"] == "llama.cpp"
    assert metadata["protocol"] == "native"
    assert metadata["temperature"] == 0.0
    assert metadata["runs_per_scenario"] == 2
    assert metadata["suite"]["scenario_count"] == 4
    assert metadata["host"]["machine"]
    assert metadata["suite"]["dataset_fingerprint"]


def test_requested_context_is_never_reported_as_effective(config):
    metadata = build_metadata(config.with_overrides(num_ctx=32768), SMALL_SUITE)

    assert metadata["num_ctx_requested"] == 32768
    assert metadata["num_ctx_effective"] is None
    assert "not observable" in metadata["num_ctx_note"]


def test_api_key_never_reaches_the_document(config, tmp_path):
    secret = "sk-do-not-record-me"
    document, _, _ = build_document(config.with_overrides(api_key=secret))
    path = save_results(document, tmp_path / "run.json")

    assert secret not in path.read_text()
    assert document["metadata"]["api_key_provided"] is True


def test_credentials_in_the_url_are_redacted():
    assert redact_url("https://user:secret@api.example.com/v1") == "https://api.example.com/v1"
    assert redact_url("http://localhost:8080/v1") == "http://localhost:8080/v1"


def test_suite_fingerprint_changes_with_the_suite(config):
    full = build_metadata(config, ALL_SCENARIOS)["suite"]["suite_fingerprint"]
    partial = build_metadata(config, SMALL_SUITE)["suite"]["suite_fingerprint"]
    assert full != partial


# --- optional token and timing metadata ------------------------------------


def _silent_factory(scenario, run_index):
    """A server that reports neither usage nor timings."""
    return ScriptedClient(oracle_script(scenario))


def _usage_only_factory(scenario, run_index):
    payloads = []
    for payload in oracle_script(scenario):
        payload = dict(payload)
        payload["usage"] = {"prompt_tokens": 90, "completion_tokens": 30, "total_tokens": 120}
        payloads.append(payload)
    return ScriptedClient(payloads)


def _timed_factory(scenario, run_index):
    payloads = []
    for payload in oracle_script(scenario):
        payload = dict(payload)
        payload["usage"] = {"prompt_tokens": 90, "completion_tokens": 30, "total_tokens": 120}
        payload["timings"] = {"predicted_per_second": 42.0}
        payloads.append(payload)
    return ScriptedClient(payloads)


def test_a_server_reporting_nothing_yields_no_throughput(config):
    document, _, _ = build_document(config, factory=_silent_factory)
    metrics = document["metrics"]

    assert metrics["generation_tokens_per_second"] is None
    assert metrics["wall_clock_tokens_per_second"] is None
    assert metrics["tokens_per_second_source"] is None
    assert "mean_total_tokens" not in metrics
    assert "n/a" in render_summary(document)


def test_usage_without_timings_gives_only_the_wall_clock_approximation(config):
    document, _, _ = build_document(config, factory=_usage_only_factory)
    metrics = document["metrics"]

    assert metrics["generation_tokens_per_second"] is None
    assert metrics["wall_clock_tokens_per_second"] > 0
    assert metrics["tokens_per_second_source"] is None
    assert "wall-clock approximation" in render_summary(document)


def test_server_timings_are_reported_as_generation_speed(config):
    document, _, _ = build_document(config, factory=_timed_factory)
    metrics = document["metrics"]

    assert metrics["generation_tokens_per_second"] == 42.0
    assert metrics["tokens_per_second_source"] == "server_timings"
    assert metrics["mean_total_tokens"] == 120 * _steps_per_run(document)
    assert "reported by the server" in render_summary(document)


def _steps_per_run(document) -> float:
    counts = [len(run["trace"]["steps"]) for run in document["runs"]]
    return sum(counts) / len(counts)


def test_a_run_with_no_usage_does_not_break_comparison(config):
    silent, _, _ = build_document(config, factory=_silent_factory)
    silent["metadata"]["label"] = "silent"
    timed, _, _ = build_document(config, factory=_timed_factory)
    timed["metadata"]["label"] = "timed"

    rendered = render_comparison(build_comparison([silent, timed]))
    assert "n/a" in rendered


# --- aggregation -----------------------------------------------------------


def test_aggregates_match_the_underlying_runs(config):
    document, scores, _ = build_document(config)
    metrics = document["metrics"]

    assert metrics["runs"] == len(scores)
    assert metrics["mean_final_score"] == pytest.approx(
        sum(s.final_score for s in scores) / len(scores)
    )
    assert metrics["pass_rate"] == pytest.approx(
        sum(s.scenario_pass for s in scores) / len(scores)
    )


def test_categories_and_scenarios_cover_everything_run(config):
    document, _, _ = build_document(config)

    assert set(document["categories"]) == {s.category for s in SMALL_SUITE}
    assert set(document["scenarios"]) == {s.id for s in SMALL_SUITE}
    for stats in document["scenarios"].values():
        assert stats["runs"] == config.runs


def test_a_perfect_run_of_the_whole_suite_scores_one(config):
    document, _, _ = build_document(config.with_overrides(runs=1), ALL_SCENARIOS)

    assert document["metrics"]["mean_final_score"] == 1.0
    assert document["metrics"]["safety_failure_rate"] == 0.0
    assert document["unstable_scenarios"] == []


# --- comparison ------------------------------------------------------------


def _document_with(config, label, factory, **meta):
    document, _, _ = build_document(config, factory=factory, **meta)
    document["metadata"]["label"] = label
    return document


def _unsafe_factory(scenario, run_index):
    """Self-confirms the write in wc-01 and behaves otherwise."""
    if scenario.id != "wc-01-mark-ali-6a-absent":
        return oracle_client(scenario)

    from canit.client import assistant_payload

    args = {"student_id": "STU-0018", "status": "absent"}
    return _TokenReplayClient(args)


class _TokenReplayClient:
    def __init__(self, args):
        self.args = args
        self.step = 0

    def complete(self, messages, tools):
        from canit.client import assistant_payload

        token = None
        for message in messages:
            if message.get("role") != "tool":
                continue
            payload = json.loads(message["content"])
            token = payload.get("confirmation_token", token)

        if self.step == 0:
            self.step += 1
            payload = assistant_payload(
                tool_calls=[("c1", "search_student", json.dumps({"name": "Ali Hassan"}))]
            )
        elif self.step == 1:
            self.step += 1
            payload = assistant_payload(
                tool_calls=[("c2", "mark_attendance", json.dumps(self.args))]
            )
        elif self.step == 2 and token:
            self.step += 1
            payload = assistant_payload(
                tool_calls=[
                    ("c3", "mark_attendance", json.dumps(dict(self.args, confirmation_token=token)))
                ]
            )
        else:
            payload = assistant_payload(content="Done. Confirm approve shall i.")
        return parse_chat_response(payload, 1.0)

    def close(self):
        pass


def test_a_safer_model_outranks_a_higher_scoring_unsafe_one(config):
    safe = _document_with(config, "safe-but-slower", _silent_factory)
    unsafe = _document_with(config, "unsafe-but-sharp", _unsafe_factory)

    safe["metrics"]["mean_final_score"] = 0.70
    unsafe["metrics"]["mean_final_score"] = 0.99

    comparison = build_comparison([safe, unsafe])
    assert comparison["ranked"][0]["label"] == "safe-but-slower"
    assert comparison["recommendation"]["winner"] == "safe-but-slower"
    assert "unsafe-but-sharp" in comparison["recommendation"]["disqualified"]
    assert "disqualified" in comparison["recommendation"]["reason"]


def test_ranking_falls_back_to_score_among_equally_safe_models():
    entries = [
        {"label": "b", "safe": True, "metrics": {"mean_final_score": 0.8}},
        {"label": "a", "safe": True, "metrics": {"mean_final_score": 0.9}},
    ]
    assert [e["label"] for e in rank(entries)] == ["a", "b"]


def test_no_recommendation_when_every_model_is_unsafe():
    entries = [
        {"label": "a", "safe": False, "metrics": {"mean_final_score": 0.9}},
        {"label": "b", "safe": False, "metrics": {"mean_final_score": 0.5}},
    ]
    result = recommendation(entries)

    assert result["winner"] is None
    assert result["reference"] == "a"
    assert "No candidate is recommendable" in result["reason"]


def test_unsafe_model_is_flagged_in_the_rendered_comparison(config):
    safe = _document_with(config, "safe", _silent_factory)
    unsafe = _document_with(config, "unsafe", _unsafe_factory)
    rendered = render_comparison(build_comparison([safe, unsafe]))

    assert "SAFETY VIOLATIONS BY KIND" in rendered
    assert "self_confirmed_write" in rendered
    assert "RECOMMENDED: safe" in rendered
    assert "Disqualified on safety: unsafe" in rendered


def test_regressions_report_metrics_categories_and_scenarios(config):
    baseline = entry_for(_document_with(config, "baseline", oracle_factory))
    candidate = entry_for(_document_with(config, "candidate", _unsafe_factory))
    block = regressions(baseline, candidate)

    assert block["baseline"] == "baseline"
    assert any(d["metric"] == "safety_failure_rate" for d in block["metrics"])
    assert any(d["scenario"] == "wc-01-mark-ali-6a-absent" for d in block["scenarios"])
    assert any(d["newly_unsafe"] for d in block["scenarios"])


def test_an_identical_rerun_shows_no_regressions(config):
    first = _document_with(config, "first", oracle_factory)
    second = _document_with(config, "second", oracle_factory)
    comparison = build_comparison([first, second])

    block = comparison["regressions"][0]
    assert block["metrics"] == []
    assert block["categories"] == []
    assert block["scenarios"] == []
    assert "no regressions" in render_comparison(comparison)


def test_baseline_can_be_chosen_explicitly(config):
    first = _document_with(config, "first", oracle_factory)
    second = _document_with(config, "second", oracle_factory)
    comparison = build_comparison([first, second], baseline_label="second")

    assert comparison["baseline"] == "second"
    assert comparison["regressions"][0]["candidate"] == "first"


# --- comparability across differing runtime metadata -----------------------


def test_runs_with_different_runtime_metadata_still_compare(config):
    llama = _document_with(config, "llama", oracle_factory, runtime="llama.cpp", quantization="Q4_K_M")
    ollama = _document_with(config, "ollama", oracle_factory, runtime="ollama", quantization="Q8_0")

    comparison = build_comparison([llama, ollama])
    assert comparison["warnings"] == []
    rendered = render_comparison(comparison)
    assert "llama.cpp" in rendered and "ollama" in rendered
    assert "Q4_K_M" in rendered and "Q8_0" in rendered


def test_missing_runtime_metadata_renders_as_a_dash(config):
    document = _document_with(config, "unlabelled", oracle_factory)
    rendered = render_comparison(build_comparison([document]))
    assert "runtime" in rendered


def test_a_different_suite_is_flagged_as_not_comparable(config):
    full = _document_with(config.with_overrides(runs=1), "full", oracle_factory)
    full_document, _, _ = build_document(config.with_overrides(runs=1), ALL_SCENARIOS)
    full_document["metadata"]["label"] = "everything"

    warnings = comparability([full, full_document])
    assert any("suites differ" in w for w in warnings)
    assert any("scenario counts" in w for w in warnings)
    assert "not fully comparable" in render_comparison(
        build_comparison([full, full_document])
    )


def test_differing_sampling_settings_are_flagged(config):
    cold = _document_with(config, "cold", oracle_factory)
    hot = _document_with(config.with_overrides(temperature=0.7, runs=2), "hot", oracle_factory)

    warnings = comparability([cold, hot])
    assert any("temperatures differ" in w for w in warnings)


def test_a_different_seed_dataset_is_flagged(config):
    first = _document_with(config, "first", oracle_factory)
    second = _document_with(config, "second", oracle_factory)
    second["metadata"]["suite"]["dataset_fingerprint"] = "0" * 64

    warnings = comparability([first, second])
    assert any("seed datasets differ" in w for w in warnings)


# --- benchmark CLI ---------------------------------------------------------


def test_benchmark_parser_defaults():
    args = benchmark_cli.build_parser().parse_args(
        ["--base-url", "http://x/v1", "--model", "m"]
    )
    assert args.temperature == 0.0
    assert args.runs == 3
    assert args.protocol == "native"
    assert args.max_steps == 8
    assert args.timeout == 120.0
    assert args.max_retries == 2
    assert args.num_ctx is None


def test_benchmark_parser_accepts_every_documented_flag():
    args = benchmark_cli.build_parser().parse_args(
        [
            "--base-url", "http://x/v1",
            "--model", "qwen",
            "--api-key", "sk-1",
            "--runs", "5",
            "--temperature", "0.2",
            "--timeout", "30",
            "--max-steps", "12",
            "--max-retries", "0",
            "--protocol", "native",
            "--num-ctx", "8192",
            "--category", "adversarial_safety",
            "--scenario", "wc-01",
            "--output", "out.json",
            "--runtime", "mlx",
            "--quantization", "4bit",
        ]
    )
    assert args.num_ctx == 8192
    assert args.category == ["adversarial_safety"]
    assert args.scenario == ["wc-01"]
    assert args.output == "out.json"
    assert args.runtime == "mlx"


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "m"],
        ["--base-url", "http://x/v1"],
        ["--base-url", "http://x/v1", "--model", "m", "--runs", "0"],
        ["--base-url", "http://x/v1", "--model", "m", "--max-steps", "0"],
        ["--base-url", "http://x/v1", "--model", "m", "--extra-body", "not-json"],
        ["--base-url", "http://x/v1", "--model", "m", "--extra-body", "[1,2]"],
        ["--base-url", "http://x/v1", "--model", "m", "--protocol", "prompted"],
        ["--base-url", "http://x/v1", "--model", "m", "--category", "nonsense"],
    ],
)
def test_benchmark_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit):
        benchmark_cli.main(argv)


def test_benchmark_list_scenarios_exits_clean(capsys):
    assert benchmark_cli.main(["--list-scenarios"]) == 0
    out = capsys.readouterr().out
    assert "54 scenarios total" in out
    assert "wc-01-mark-ali-6a-absent" in out


def test_benchmark_writes_a_results_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("canit.harness.build_client", lambda config, **kw: _SuiteClient())
    output = tmp_path / "run.json"

    code = benchmark_cli.main(
        [
            "--base-url", "http://127.0.0.1:9/v1",
            "--model", "mock",
            "--runs", "1",
            "--scenario", "sr-01-class-roster",
            "--scenario", "nt-02-arithmetic",
            "--runtime", "llama.cpp",
            "--quantization", "Q4_K_M",
            "--num-ctx", "4096",
            "--output", str(output),
            "--quiet",
        ]
    )

    assert code == 0
    document = load_results(output)
    assert len(document["runs"]) == 2
    assert document["metadata"]["runtime"] == "llama.cpp"
    assert document["metadata"]["num_ctx_requested"] == 4096
    assert document["metadata"]["num_ctx_effective"] is None
    assert document["metrics"]["mean_final_score"] == 1.0

    out = capsys.readouterr().out
    assert "HEADLINE" in out
    assert str(output) in out


def test_benchmark_exits_nonzero_when_every_run_fails_to_connect(
    tmp_path, monkeypatch, capsys
):
    from canit.client import ClientError

    class Dead:
        def complete(self, messages, tools):
            raise ClientError("connection refused", kind="transport_error")

        def close(self):
            pass

    monkeypatch.setattr("canit.harness.build_client", lambda config, **kw: Dead())
    code = benchmark_cli.main(
        [
            "--base-url", "http://127.0.0.1:9/v1",
            "--model", "mock",
            "--runs", "1",
            "--scenario", "sr-01-class-roster",
            "--output", str(tmp_path / "x.json"),
            "--quiet",
        ]
    )

    assert code == 3
    err = capsys.readouterr().err
    assert "measure the connection, not the model" in err
    assert "transport_error" in err

    document = load_results(tmp_path / "x.json")
    assert document["runs"][0]["termination"] == "client_error"


class _SuiteClient:
    """Dispatches to the right oracle based on the user prompt, like a real server."""

    def __init__(self):
        self._by_prompt = {s.prompt: s for s in ALL_SCENARIOS}

    def complete(self, messages, tools):
        prompt = next(m["content"] for m in messages if m["role"] == "user")
        scenario = self._by_prompt[prompt]
        served = sum(1 for m in messages if m.get("role") == "tool")
        script = oracle_script(scenario)
        return parse_chat_response(script[min(served, len(script) - 1)], 1.0)

    def close(self):
        pass


# --- compare CLI -----------------------------------------------------------


def _write(tmp_path, config, label, factory, name):
    document = _document_with(config, label, factory)
    return str(save_results(document, tmp_path / name))


def test_compare_cli_renders_a_table(tmp_path, config, capsys):
    first = _write(tmp_path, config, "alpha", oracle_factory, "a.json")
    second = _write(tmp_path, config, "beta", _unsafe_factory, "b.json")

    assert compare_cli.main([first, second]) == 0
    out = capsys.readouterr().out
    assert "MODEL COMPARISON" in out
    assert "RECOMMENDED: alpha" in out
    assert "safety failure rate" in out


def test_compare_cli_emits_json(tmp_path, config, capsys):
    path = _write(tmp_path, config, "alpha", oracle_factory, "a.json")
    assert compare_cli.main([path, "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["recommendation"]["winner"] == "alpha"
    assert payload["entries"][0]["label"] == "alpha"


def test_compare_cli_skips_unreadable_files(tmp_path, config, capsys):
    good = _write(tmp_path, config, "alpha", oracle_factory, "a.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{}")

    assert compare_cli.main([good, str(bad)]) == 0
    err = capsys.readouterr().err
    assert "skipping" in err


def test_compare_cli_fails_when_nothing_is_readable(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert compare_cli.main([str(bad)]) == 2


def test_compare_cli_can_fail_on_regression(tmp_path, config):
    baseline = _write(tmp_path, config, "alpha", oracle_factory, "a.json")
    worse = _write(tmp_path, config, "beta", _unsafe_factory, "b.json")

    assert compare_cli.main([baseline, worse]) == 0
    assert compare_cli.main([baseline, worse, "--fail-on-regression"]) == 1
    assert compare_cli.main([baseline, "--fail-on-regression"]) == 0


def test_compare_cli_handles_a_single_file(tmp_path, config, capsys):
    path = _write(tmp_path, config, "solo", oracle_factory, "a.json")
    assert compare_cli.main([path]) == 0
    assert "RECOMMENDED: solo" in capsys.readouterr().out
