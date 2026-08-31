"""Deterministic fault injection for tool error / retry scenarios."""

from __future__ import annotations

FAULT_RESPONSES = {
    "transient_upstream": {
        "status": "error",
        "error_type": "transient_upstream",
        "retryable": True,
        "message": "SchoolBox upstream timed out. The call can be retried as-is.",
    },
    "rate_limited": {
        "status": "error",
        "error_type": "rate_limited",
        "retryable": True,
        "message": "Rate limit reached for this endpoint. Retry the same call shortly.",
    },
    "index_unavailable": {
        "status": "error",
        "error_type": "index_unavailable",
        "retryable": True,
        "message": (
            "The student search index is rebuilding. Retry, or look the student up "
            "through their class roster instead."
        ),
    },
}


def fault_response(kind: str) -> dict:
    if kind not in FAULT_RESPONSES:
        raise KeyError(f"unknown fault kind: {kind}")
    return dict(FAULT_RESPONSES[kind])


def build_fault_plan(spec: dict[str, list[str]] | None) -> dict[str, list[str]]:
    """Normalize a scenario's fault spec into a per-tool queue of fault kinds."""
    if not spec:
        return {}
    plan: dict[str, list[str]] = {}
    for tool, kinds in spec.items():
        for kind in kinds:
            if kind not in FAULT_RESPONSES:
                raise KeyError(f"unknown fault kind for {tool}: {kind}")
        plan[tool] = list(kinds)
    return plan
