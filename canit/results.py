"""The versioned results document: build, save, load, and check compatibility."""

from __future__ import annotations

import json
from pathlib import Path

from .metadata import SCHEMA_VERSION
from .scoring.aggregate import build_report

COMPATIBLE_SCHEMA_MAJORS = {"1"}


class SchemaError(ValueError):
    pass


def build_results(metadata: dict, scores, traces) -> dict:
    by_key = {(t.scenario_id, t.run_index): t for t in traces}
    report = build_report(metadata["label"], scores, traces)

    runs = []
    for score in scores:
        payload = score.to_dict()
        trace = by_key.get((score.scenario_id, score.run_index))
        payload["trace"] = trace.to_dict() if trace else None
        runs.append(payload)

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "metrics": report.overall,
        "categories": report.categories,
        "scenarios": report.scenarios,
        "unstable_scenarios": report.unstable_scenarios,
        "runs": runs,
    }


def check_schema(document: dict, source: str = "document") -> None:
    version = document.get("schema_version")
    if not version:
        raise SchemaError(f"{source} has no schema_version")
    major = str(version).split(".")[0]
    if major not in COMPATIBLE_SCHEMA_MAJORS:
        raise SchemaError(
            f"{source} uses results schema {version}; this build reads "
            f"{'/'.join(sorted(COMPATIBLE_SCHEMA_MAJORS))}.x"
        )
    for key in ("metadata", "metrics", "runs"):
        if key not in document:
            raise SchemaError(f"{source} is missing the {key!r} section")


def save_results(document: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, default=str) + "\n")
    return target


def load_results(path: str | Path) -> dict:
    source = Path(path)
    document = json.loads(source.read_text())
    check_schema(document, source.name)
    return document


def default_output_path(metadata: dict, directory: str | Path = "results") -> Path:
    slug = "".join(
        char if char.isalnum() or char in "-_." else "-"
        for char in str(metadata["label"])
    ).strip("-")
    stamp = metadata["started_at"].replace(":", "").replace("-", "")[:15]
    return Path(directory) / f"{slug}-{stamp}.json"


def comparability(documents: list[dict]) -> list[str]:
    """Warnings that make two result files not directly comparable."""
    warnings = []
    versions = {d["metadata"]["suite"].get("version", "unknown") for d in documents}
    suites = {d["metadata"]["suite"]["suite_fingerprint"] for d in documents}
    datasets = {d["metadata"]["suite"]["dataset_fingerprint"] for d in documents}
    temperatures = {d["metadata"]["temperature"] for d in documents}
    runs = {d["metadata"]["runs_per_scenario"] for d in documents}
    steps = {d["metadata"]["max_steps"] for d in documents}
    counts = {d["metadata"]["suite"]["scenario_count"] for d in documents}
    protocols = {d["metadata"]["protocol"] for d in documents}

    if len(versions) > 1:
        warnings.append(
            f"scenario suite versions differ: {sorted(versions)}; a scenario change "
            "alters what a score means"
        )
    if len(suites) > 1:
        warnings.append(
            "scenario suites differ between files; scores are not directly comparable"
        )
    if len(datasets) > 1:
        warnings.append(
            "seed datasets differ between files; ground truth differs, so scores are "
            "not comparable"
        )
    if len(counts) > 1:
        warnings.append(f"different scenario counts were run: {sorted(counts)}")
    if len(temperatures) > 1:
        warnings.append(f"temperatures differ: {sorted(temperatures)}")
    if len(runs) > 1:
        warnings.append(f"runs per scenario differ: {sorted(runs)}")
    if len(steps) > 1:
        warnings.append(f"max_steps differ: {sorted(steps)}")
    if len(protocols) > 1:
        warnings.append(f"protocols differ: {sorted(protocols)}")
    return warnings
