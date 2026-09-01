"""Running the suite against one endpoint."""

from __future__ import annotations

import time

from .client import build_client
from .runner import run_scenario
from .scoring import score_run


def run_suite(config, scenarios, client_factory=None, progress=None):
    """Run every scenario `config.runs` times. Returns (scores, traces)."""
    shared = None
    if client_factory is None:
        shared = build_client(config)

        def client_factory(scenario, run_index):
            return shared

    scores, traces = [], []
    total = len(scenarios) * config.runs
    completed = 0

    try:
        for scenario in scenarios:
            for run_index in range(config.runs):
                started = time.perf_counter()
                client = client_factory(scenario, run_index)
                trace, _ = run_scenario(config, client, scenario, run_index=run_index)
                score = score_run(trace, scenario)
                scores.append(score)
                traces.append(trace)
                completed += 1
                if progress is not None:
                    progress(
                        {
                            "completed": completed,
                            "total": total,
                            "scenario": scenario,
                            "run_index": run_index,
                            "score": score,
                            "trace": trace,
                            "elapsed": time.perf_counter() - started,
                        }
                    )
    finally:
        if shared is not None:
            shared.close()

    return scores, traces


def select_scenarios(
    scenarios, categories=None, scenario_ids=None, exclude_categories=None
):
    selected = list(scenarios)
    if categories:
        wanted = set(categories)
        selected = [s for s in selected if s.category in wanted]
    if scenario_ids:
        wanted = set(scenario_ids)
        selected = [s for s in selected if s.id in wanted or any(w in s.id for w in wanted)]
    if exclude_categories:
        unwanted = set(exclude_categories)
        selected = [s for s in selected if s.category not in unwanted]
    return selected
