#!/usr/bin/env python3
"""Run the tool-calling benchmark against one OpenAI-compatible endpoint.

    python benchmark.py --base-url http://localhost:8080/v1 --model qwen --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from canit.client import ClientError, build_client
from canit.config import RunConfig
from canit.harness import run_suite, select_scenarios
from canit.metadata import build_metadata, utc_now
from canit.report.terminal import render_progress, render_summary
from canit.results import build_results, default_output_path, save_results
from canit.scenarios.suite import ALL_SCENARIOS, CATEGORIES
from canit.trace import TERMINATION_CLIENT_ERROR, TERMINATION_TIMEOUT

ENDPOINT_FAILURES = frozenset({TERMINATION_CLIENT_ERROR, TERMINATION_TIMEOUT})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.py",
        description="Benchmark an LLM's tool-calling reliability on a simulated school system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    endpoint = parser.add_argument_group("endpoint")
    endpoint.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. http://localhost:8080/v1")
    endpoint.add_argument("--model", help="Model name to send in the request body")
    endpoint.add_argument(
        "--api-key",
        default=os.environ.get("CANIT_API_KEY"),
        help="Bearer token if the endpoint needs one. Defaults to $CANIT_API_KEY. Never recorded.",
    )
    endpoint.add_argument("--protocol", default="native", choices=["native"], help="Tool-calling protocol")

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument("--temperature", type=float, default=0.0)
    sampling.add_argument("--num-ctx", type=int, default=None, help="Requested context size, if the runtime honours it")
    sampling.add_argument("--max-tokens", type=int, default=None)
    sampling.add_argument("--extra-body", default=None, help="Extra JSON merged into the request body")

    execution = parser.add_argument_group("execution")
    execution.add_argument("--runs", type=int, default=3, help="Runs per scenario")
    execution.add_argument("--max-steps", type=int, default=8, help="Tool-call steps before a run is abandoned")
    execution.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    execution.add_argument("--total-timeout", type=float, default=None, help="Wall-clock budget per scenario run")
    execution.add_argument("--max-retries", type=int, default=2, help="HTTP retries on 429/5xx")

    selection = parser.add_argument_group("scenario selection")
    selection.add_argument("--category", action="append", choices=list(CATEGORIES), help="Repeatable")
    selection.add_argument("--exclude-category", action="append", choices=list(CATEGORIES), help="Repeatable")
    selection.add_argument("--scenario", action="append", help="Scenario id or substring. Repeatable")
    selection.add_argument("--list-scenarios", action="store_true", help="Print the suite and exit")

    output = parser.add_argument_group("output and provenance")
    output.add_argument("--output", "-o", default=None, help="Results JSON path")
    output.add_argument("--results-dir", default="results")
    output.add_argument("--label", default=None, help="Display name for this run. Defaults to the model name")
    output.add_argument("--quantization", default=None, help="e.g. Q4_K_M. Recorded, never inferred")
    output.add_argument("--model-file", default=None, help="GGUF or weights filename, for provenance")
    output.add_argument("--runtime", default=None, help="e.g. llama.cpp, ollama, mlx, vllm")
    output.add_argument("--notes", default=None)
    output.add_argument("--quiet", action="store_true", help="Suppress per-run progress")
    output.add_argument("--no-summary", action="store_true", help="Skip the terminal summary")
    return parser


def _list_scenarios() -> int:
    for category in CATEGORIES:
        members = [s for s in ALL_SCENARIOS if s.category == category]
        print(f"\n{category}  ({len(members)})")
        for scenario in members:
            print(f"  {scenario.id:<36} {scenario.prompt[:70]}")
    print(f"\n{len(ALL_SCENARIOS)} scenarios total")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_scenarios:
        return _list_scenarios()

    missing = [name for name in ("base_url", "model") if not getattr(args, name)]
    if missing:
        parser.error(f"--{missing[0].replace('_', '-')} is required")

    extra_body = {}
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except ValueError as exc:
            parser.error(f"--extra-body is not valid JSON: {exc}")
        if not isinstance(extra_body, dict):
            parser.error("--extra-body must be a JSON object")

    try:
        config = RunConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            total_timeout=args.total_timeout,
            runs=args.runs,
            max_steps=args.max_steps,
            protocol=args.protocol,
            max_retries=args.max_retries,
            extra_body=extra_body,
            label=args.label,
        )
    except ValueError as exc:
        parser.error(str(exc))

    scenarios = select_scenarios(
        ALL_SCENARIOS,
        categories=args.category,
        scenario_ids=args.scenario,
        exclude_categories=args.exclude_category,
    )
    if not scenarios:
        parser.error("scenario selection matched nothing")

    started_at = utc_now()
    started = time.perf_counter()

    progress = None
    if not args.quiet:
        def progress(event):
            print(render_progress(event), flush=True)

        print(
            f"Running {len(scenarios)} scenarios x {config.runs} runs "
            f"against {config.model} at {config.base_url}\n",
            flush=True,
        )

    try:
        scores, traces = run_suite(config, scenarios, progress=progress)
    except ClientError as exc:
        print(f"\nendpoint error: {exc}", file=sys.stderr)
        print(f"detail: {exc.detail}", file=sys.stderr)
        return 2

    duration = time.perf_counter() - started
    metadata = build_metadata(
        config,
        scenarios,
        quantization=args.quantization,
        model_file=args.model_file,
        runtime=args.runtime,
        notes=args.notes,
        scenario_filter={
            "categories": args.category,
            "exclude_categories": args.exclude_category,
            "scenarios": args.scenario,
        },
        started_at=started_at,
        finished_at=utc_now(),
        duration_seconds=round(duration, 3),
    )

    document = build_results(metadata, scores, traces)
    destination = args.output or default_output_path(metadata, args.results_dir)
    path = save_results(document, destination)

    failed = [t for t in traces if t.termination in ENDPOINT_FAILURES]
    if len(failed) == len(traces):
        timed_out = sum(1 for t in failed if t.termination == "timeout")
        print(f"\nresults written to {path}", file=sys.stderr)
        print(
            f"every one of the {len(failed)} runs failed to reach the endpoint. "
            "The scores in this file measure the connection, not the model.",
            file=sys.stderr,
        )
        if timed_out:
            print(
                f"{timed_out} of them timed out after {config.timeout}s. "
                "If the endpoint is simply slow, raise --timeout.",
                file=sys.stderr,
            )
        first = next((t.errors[0] for t in traces if t.errors), None)
        if first:
            print(f"first error: {first['kind']}: {first['message']}", file=sys.stderr)
        return 3

    if not args.no_summary:
        print()
        print(render_summary(document))
    if failed:
        print(
            f"\nwarning: {len(failed)}/{len(traces)} runs failed to reach the "
            "endpoint and scored zero for reasons unrelated to the model",
            file=sys.stderr,
        )
    print(f"\nresults written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
