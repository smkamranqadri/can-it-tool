"""Argument matchers.

Scenario expectations declare what an argument must *mean*, not the exact bytes the
model emitted. Strings compare case- and whitespace-insensitively so that '7a' and
'7A' match; everything else compares by value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


class _Any:
    def __repr__(self) -> str:
        return "ANY"

    def describe(self) -> str:
        return "any value"


ANY = _Any()


@dataclass(frozen=True)
class OneOf:
    values: tuple

    def __init__(self, *values):
        object.__setattr__(self, "values", tuple(values))

    def describe(self) -> str:
        return f"one of {list(self.values)}"


@dataclass(frozen=True)
class Predicate:
    check: Callable[[object], bool]
    description: str

    def describe(self) -> str:
        return self.description


@dataclass(frozen=True)
class Absent:
    """The argument must not be supplied at all."""

    def describe(self) -> str:
        return "absent"


ABSENT = Absent()


def normalize(value):
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return value


def match_value(expected, actual) -> bool:
    if isinstance(expected, _Any):
        return True
    if isinstance(expected, OneOf):
        return any(match_value(option, actual) for option in expected.values)
    if isinstance(expected, Predicate):
        return bool(expected.check(actual))
    if isinstance(expected, Absent):
        return actual is None
    return normalize(expected) == normalize(actual)


def describe(expected) -> str:
    if hasattr(expected, "describe"):
        return expected.describe()
    return repr(expected)


def match_arguments(expected: dict, actual: dict | None) -> tuple[bool, dict]:
    """Match one call's arguments. Extra arguments the spec doesn't mention are fine."""
    if actual is None:
        return False, {"reason": "arguments did not parse"}

    mismatches = {}
    for key, want in (expected or {}).items():
        got = actual.get(key)
        if not match_value(want, got):
            mismatches[key] = {"expected": describe(want), "actual": got}
    return not mismatches, {"mismatches": mismatches}


_COMMA_GROUPING = re.compile(r"(?<=\d),(?=\d)")
_SPACE_GROUPING = re.compile(r"(?<=\d)[\s\u00a0\u202f](?=\d)")


def normalize_text(text: str) -> str:
    """Casefold and squash runs of whitespace."""
    return " ".join((text or "").split()).casefold()


def number_readings(text: str) -> tuple[str, ...]:
    """Every reading of the text a number could legitimately appear in.

    Digit grouping has to be undone so '13,000' reads as 13000, but undoing it
    unconditionally would also weld two adjacent numbers together: '80 77' would
    become '8077' and neither value would be found. Both readings are kept and a
    number counts as present if it appears in any of them.
    """
    base = normalize_text(text)
    return (base, _COMMA_GROUPING.sub("", base), _SPACE_GROUPING.sub("", _COMMA_GROUPING.sub("", base)))


def contains_text(haystack: str, needle: str) -> bool:
    return normalize_text(needle) in normalize_text(haystack)


def contains_number(haystack: str, number) -> bool:
    pattern = re.compile(rf"(?<!\d){re.escape(str(number))}(?!\d)")
    return any(pattern.search(reading) for reading in number_readings(haystack))
