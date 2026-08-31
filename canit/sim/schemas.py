"""OpenAI-compatible tool definitions and the READ / WRITE split."""

from __future__ import annotations

from .seed import ATTENDANCE_STATUSES, SUBJECTS, SUBMISSION_STATUSES, WEEKDAYS

CONFIRMATION_ARG = "confirmation_token"

_CONFIRMATION_PROPERTY = {
    CONFIRMATION_ARG: {
        "type": "string",
        "description": (
            "Token issued by a previous call to this same tool with these same "
            "arguments. Omit it on the first call: the tool will not mutate anything "
            "and will return a token plus a summary to confirm with the user first."
        ),
    }
}

READ_TOOLS = {
    "search_student": {
        "description": (
            "Search students by full or partial name. May return several matches; "
            "narrow with class_name or grade when the name is ambiguous."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full or partial name."},
                "class_name": {
                    "type": "string",
                    "description": "Optional class filter, e.g. '7A'.",
                },
                "grade": {"type": "integer", "description": "Optional grade filter."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    "get_student": {
        "description": "Fetch one student's profile and guardians by student_id.",
        "parameters": {
            "type": "object",
            "properties": {"student_id": {"type": "string"}},
            "required": ["student_id"],
            "additionalProperties": False,
        },
    },
    "get_attendance": {
        "description": (
            "Attendance records for one student. Defaults to today when no date is "
            "given; pass start_date and end_date for a range."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"},
                "date": {
                    "type": "string",
                    "description": "YYYY-MM-DD, or 'today' / 'yesterday'.",
                },
                "start_date": {"type": "string", "description": "YYYY-MM-DD."},
                "end_date": {"type": "string", "description": "YYYY-MM-DD."},
            },
            "required": ["student_id"],
            "additionalProperties": False,
        },
    },
    "list_assignments": {
        "description": (
            "List assignments, optionally filtered by class, grade, subject or status. "
            "Returns assignment metadata only, never submissions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string"},
                "grade": {"type": "integer"},
                "subject": {"type": "string", "enum": SUBJECTS},
                "status": {"type": "string", "enum": ["open", "closed"]},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    "get_assignment": {
        "description": "Fetch one assignment by assignment_id.",
        "parameters": {
            "type": "object",
            "properties": {"assignment_id": {"type": "string"}},
            "required": ["assignment_id"],
            "additionalProperties": False,
        },
    },
    "get_submissions": {
        "description": (
            "Submissions for one assignment, with student names. Filter by status to "
            "find who has not submitted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "assignment_id": {"type": "string"},
                "status": {"type": "string", "enum": SUBMISSION_STATUSES},
                "student_id": {"type": "string"},
            },
            "required": ["assignment_id"],
            "additionalProperties": False,
        },
    },
    "get_fee_status": {
        "description": (
            "Fee invoice for one student: total, amount paid, and outstanding balance."
        ),
        "parameters": {
            "type": "object",
            "properties": {"student_id": {"type": "string"}},
            "required": ["student_id"],
            "additionalProperties": False,
        },
    },
    "get_class_students": {
        "description": "Roster for one class, e.g. '6A'.",
        "parameters": {
            "type": "object",
            "properties": {"class_name": {"type": "string"}},
            "required": ["class_name"],
            "additionalProperties": False,
        },
    },
    "get_timetable": {
        "description": (
            "Timetable for one class on one weekday. Accepts a weekday name or "
            "'today' / 'tomorrow'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string"},
                "day": {
                    "type": "string",
                    "description": (
                        "Weekday name, or 'today' / 'tomorrow'. "
                        f"School days: {', '.join(WEEKDAYS)}."
                    ),
                },
            },
            "required": ["class_name", "day"],
            "additionalProperties": False,
        },
    },
    "get_academic_record": {
        "description": (
            "Term grades for one student across subjects. This is reported academic "
            "performance, not homework submission state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"},
                "subject": {"type": "string", "enum": SUBJECTS},
            },
            "required": ["student_id"],
            "additionalProperties": False,
        },
    },
}

WRITE_TOOLS = {
    "mark_attendance": {
        "description": (
            "Set one student's attendance status for one day. Requires confirmation: "
            "call once without a confirmation_token to get a summary and a token, "
            "confirm with the user, then call again with that token."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"},
                "status": {"type": "string", "enum": ATTENDANCE_STATUSES},
                "date": {
                    "type": "string",
                    "description": "YYYY-MM-DD, or 'today'. Defaults to today.",
                },
                **_CONFIRMATION_PROPERTY,
            },
            "required": ["student_id", "status"],
            "additionalProperties": False,
        },
    },
    "record_fee_payment": {
        "description": (
            "Record a fee payment against a student's invoice, reducing the balance. "
            "Requires confirmation, as above. Only records money actually received; "
            "it is not a way to adjust or write off a balance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"},
                "amount": {
                    "type": "integer",
                    "description": "Amount received, in PKR. Must be positive.",
                },
                "method": {
                    "type": "string",
                    "enum": ["cash", "bank_transfer", "card", "cheque"],
                },
                "reference": {"type": "string"},
                **_CONFIRMATION_PROPERTY,
            },
            "required": ["student_id", "amount"],
            "additionalProperties": False,
        },
    },
    "update_submission_status": {
        "description": (
            "Update one student's submission status, and optionally its score, for "
            "one assignment. Requires confirmation, as above."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "assignment_id": {"type": "string"},
                "student_id": {"type": "string"},
                "status": {"type": "string", "enum": SUBMISSION_STATUSES},
                "score": {"type": "integer"},
                **_CONFIRMATION_PROPERTY,
            },
            "required": ["assignment_id", "student_id", "status"],
            "additionalProperties": False,
        },
    },
}

ALL_TOOLS = {**READ_TOOLS, **WRITE_TOOLS}
READ_TOOL_NAMES = frozenset(READ_TOOLS)
WRITE_TOOL_NAMES = frozenset(WRITE_TOOLS)
TOOL_NAMES = frozenset(ALL_TOOLS)


def openai_tool_specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for name, spec in ALL_TOOLS.items()
    ]
