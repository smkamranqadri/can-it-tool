"""Run metadata: enough to make two result files genuinely comparable.

Everything here is either supplied by the operator or safely derivable from the local
machine. No credentials, no hostname, no username.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

BENCHMARK_VERSION = "0.1.0"
SCHEMA_VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_url(url: str) -> str:
    """Strip any userinfo from the base URL so a key in the URL is never recorded."""
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def git_commit() -> dict | None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "short": commit[:12], "dirty": bool(status)}


def host_info() -> dict:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
    }


def suite_fingerprint(scenarios) -> str:
    payload = json.dumps(
        [
            {"id": s.id, "category": s.category, "prompt": s.prompt, "expect": repr(s.expect)}
            for s in scenarios
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_metadata(
    config,
    scenarios,
    *,
    quantization: str | None = None,
    model_file: str | None = None,
    runtime: str | None = None,
    notes: str | None = None,
    scenario_filter: dict | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    duration_seconds: float | None = None,
) -> dict:
    from .sim.db import dataset_fingerprint

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "schema_version": SCHEMA_VERSION,
        "git": git_commit(),
        "started_at": started_at or utc_now(),
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "model": config.model,
        "label": config.display_name,
        "quantization": quantization,
        "model_file": model_file,
        "runtime": runtime,
        "notes": notes,
        "protocol": config.protocol,
        "base_url": redact_url(config.base_url),
        "api_key_provided": config.api_key is not None,
        "temperature": config.temperature,
        "num_ctx_requested": config.num_ctx,
        "num_ctx_effective": None,
        "num_ctx_note": (
            "num_ctx_requested is what this harness asked for. Whether the serving "
            "runtime honoured it is not observable from an OpenAI-compatible response, "
            "so num_ctx_effective is left null unless recorded by hand."
        ),
        "max_tokens": config.max_tokens,
        "runs_per_scenario": config.runs,
        "max_steps": config.max_steps,
        "timeout_seconds": config.timeout,
        "total_timeout_seconds": config.total_timeout,
        "max_retries": config.max_retries,
        "extra_body": config.extra_body,
        "host": host_info(),
        "suite": {
            "scenario_count": len(scenarios),
            "suite_fingerprint": suite_fingerprint(scenarios),
            "dataset_fingerprint": dataset_fingerprint(),
        },
        "scenario_filter": scenario_filter or {},
    }
