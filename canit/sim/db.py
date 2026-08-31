"""In-memory store for one scenario run.

Each run gets its own `Store`, built from a cached pristine dataset, so writes made
by one model on one scenario can never leak into another run.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from .seed import TODAY, WEEKDAYS, build_dataset

_PRISTINE: dict | None = None


def pristine_dataset() -> dict:
    global _PRISTINE
    if _PRISTINE is None:
        _PRISTINE = build_dataset()
    return _PRISTINE


def dataset_fingerprint() -> str:
    payload = json.dumps(pristine_dataset(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class Mutation:
    tool: str
    arguments: dict
    entity: str
    before: dict | None
    after: dict | None
    sequence: int


@dataclass
class Violation:
    kind: str
    tool: str
    arguments: dict
    detail: str
    sequence: int


@dataclass
class ConfirmationToken:
    token: str
    tool: str
    fingerprint: str
    arguments: dict
    summary: str
    spent: bool = False


@dataclass
class Store:
    data: dict = field(default_factory=lambda: copy.deepcopy(pristine_dataset()))
    tokens: dict[str, ConfirmationToken] = field(default_factory=dict)
    mutations: list[Mutation] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    faults: dict[str, list[str]] = field(default_factory=dict)
    _sequence: int = 0

    @property
    def today(self) -> date:
        return TODAY

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def resolve_date(self, value: str | None) -> date:
        """Accept an ISO date, or the relative words the prompts actually use."""
        if value is None:
            return TODAY
        text = value.strip().lower()
        if text in {"today", ""}:
            return TODAY
        if text == "tomorrow":
            return TODAY + timedelta(days=1)
        if text == "yesterday":
            return TODAY - timedelta(days=1)
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()

    def resolve_day_name(self, value: str | None) -> str:
        if value is None:
            return WEEKDAYS[TODAY.weekday()]
        text = value.strip().lower()
        if text == "today":
            return WEEKDAYS[TODAY.weekday()]
        if text == "tomorrow":
            return WEEKDAYS[(TODAY + timedelta(days=1)).weekday()]
        if text == "yesterday":
            return WEEKDAYS[(TODAY - timedelta(days=1)).weekday()]
        for day in WEEKDAYS:
            if day.lower() == text:
                return day
        raise ValueError(f"not a school day: {value}")

    def students(self) -> list[dict]:
        return self.data["students"]

    def student(self, student_id: str) -> dict | None:
        return next(
            (s for s in self.students() if s["student_id"] == student_id), None
        )

    def klass(self, class_name: str) -> dict | None:
        target = class_name.strip().lower()
        for entry in self.data["classes"]:
            if entry["class_name"].lower() == target:
                return entry
            if f"class {entry['class_name']}".lower() == target:
                return entry
            if f"grade {entry['class_name']}".lower() == target:
                return entry
        return None

    def assignment(self, assignment_id: str) -> dict | None:
        return next(
            (
                a
                for a in self.data["assignments"]
                if a["assignment_id"] == assignment_id
            ),
            None,
        )

    def fee_invoice(self, student_id: str) -> dict | None:
        return next(
            (f for f in self.data["fees"] if f["student_id"] == student_id), None
        )

    def record_violation(
        self, kind: str, tool: str, arguments: dict, detail: str
    ) -> None:
        self.violations.append(
            Violation(
                kind=kind,
                tool=tool,
                arguments=copy.deepcopy(arguments),
                detail=detail,
                sequence=self.next_sequence(),
            )
        )

    def record_mutation(
        self,
        tool: str,
        arguments: dict,
        entity: str,
        before: dict | None,
        after: dict | None,
    ) -> None:
        self.mutations.append(
            Mutation(
                tool=tool,
                arguments=copy.deepcopy(arguments),
                entity=entity,
                before=copy.deepcopy(before),
                after=copy.deepcopy(after),
                sequence=self.next_sequence(),
            )
        )

    def pop_fault(self, tool: str) -> str | None:
        queue = self.faults.get(tool)
        if not queue:
            return None
        return queue.pop(0)
