"""Scenario shape shared by the runner and the scorer."""

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

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "expect": self.expect,
            "faults": self.faults,
            "max_steps": self.max_steps,
            "notes": self.notes,
        }
