"""Aggregation from per-run scores into scenario, category, and overall metrics."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .dimensions import DIMENSION_ORDER


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return round(statistics.fmean(values), 6)


def _stdev(values) -> float:
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return 0.0
    return round(statistics.pstdev(values), 6)


def _percentile(values, fraction: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    index = min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))
    return round(values[index], 4)


def _dimension_means(scores) -> dict:
    means = {}
    for name in DIMENSION_ORDER:
        applicable = [
            s.dimension(name).score for s in scores if s.dimension(name).applicable
        ]
        means[name] = {
            "mean": _mean(applicable),
            "applicable_runs": len(applicable),
        }
    return means


def _required_call_accuracy(scores) -> float | None:
    total = 0
    satisfied = 0
    for score in scores:
        for match in score.required_call_matches:
            total += 1
            satisfied += int(match["matched"])
    if not total:
        return None
    return round(satisfied / total, 6)


def summarize(scores, traces=None) -> dict:
    """Metrics over an arbitrary group of run scores."""
    if not scores:
        return {"runs": 0}

    traces = traces or []
    latencies = [t.total_latency_ms for t in traces]
    throughput = [t.tokens_per_second() for t in traces]

    summary = {
        "runs": len(scores),
        "mean_final_score": _mean(s.final_score for s in scores),
        "mean_raw_score": _mean(s.raw_score for s in scores),
        "pass_rate": _mean(float(s.scenario_pass) for s in scores),
        "task_completion_rate": _mean(float(s.task_completed) for s in scores),
        "safety_failure_rate": _mean(
            float(s.safety_gate_triggered) for s in scores
        ),
        "hallucinated_tool_rate": _mean(
            float(s.hallucinated_call_count > 0) for s in scores
        ),
        "tool_call_accuracy": _required_call_accuracy(scores),
        "mean_tool_calls": _mean(float(s.tool_call_count) for s in scores),
        "dimensions": _dimension_means(scores),
        "termination_counts": _counts(s.termination for s in scores),
        "safety_violation_counts": _counts(
            v["kind"] for s in scores for v in s.safety_violations
        ),
    }

    if traces:
        summary["latency_ms"] = {
            "mean": _mean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        }
        summary["tokens_per_second"] = _mean(throughput)
        usage = [t.usage_totals() for t in traces]
        reported = [u for u in usage if u]
        if reported:
            summary["mean_total_tokens"] = _mean(
                u.get("total_tokens") for u in reported
            )
    return summary


def _counts(values) -> dict:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


@dataclass
class Report:
    model: str
    overall: dict = field(default_factory=dict)
    categories: dict = field(default_factory=dict)
    scenarios: dict = field(default_factory=dict)
    unstable_scenarios: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "overall": self.overall,
            "categories": self.categories,
            "scenarios": self.scenarios,
            "unstable_scenarios": self.unstable_scenarios,
        }


def build_report(model: str, scores, traces=None) -> Report:
    traces = traces or []
    by_trace = {(t.scenario_id, t.run_index): t for t in traces}

    def traces_for(group):
        found = [by_trace.get((s.scenario_id, s.run_index)) for s in group]
        return [t for t in found if t is not None]

    report = Report(model=model)
    report.overall = summarize(scores, traces_for(scores))

    by_category: dict[str, list] = {}
    by_scenario: dict[str, list] = {}
    for score in scores:
        by_category.setdefault(score.category, []).append(score)
        by_scenario.setdefault(score.scenario_id, []).append(score)

    for category, group in sorted(by_category.items()):
        report.categories[category] = summarize(group, traces_for(group))

    for scenario_id, group in sorted(by_scenario.items()):
        finals = [s.final_score for s in group]
        passes = [s.scenario_pass for s in group]
        entry = {
            "category": group[0].category,
            "runs": len(group),
            "mean_final_score": _mean(finals),
            "mean_raw_score": _mean(s.raw_score for s in group),
            "score_stdev": _stdev(finals),
            "pass_count": sum(passes),
            "consistent": len(set(passes)) == 1,
            "safety_failures": sum(1 for s in group if s.safety_gate_triggered),
            "terminations": _counts(s.termination for s in group),
        }
        report.scenarios[scenario_id] = entry
        if not entry["consistent"]:
            report.unstable_scenarios.append(scenario_id)

    return report
