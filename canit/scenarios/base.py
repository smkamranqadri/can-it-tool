"""Scenario shape shared by the runner, the scorer, and the suite validator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    prompt: str
    expect: dict = field(default_factory=dict)
    faults: dict[str, list[str]] = field(default_factory=dict)
    max_steps: int | None = None
    notes: str = ""
    ground_truth: list[dict] = field(default_factory=list)
    oracle: list[dict] | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "expect": self.expect,
            "faults": self.faults,
            "max_steps": self.max_steps,
            "notes": self.notes,
            "ground_truth": self.ground_truth,
        }
