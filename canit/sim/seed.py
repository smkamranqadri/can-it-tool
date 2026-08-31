"""Deterministic seed data for the simulated SchoolBox environment.

Every value here is derived from fixed tables plus per-entity RNG streams keyed by
stable strings, so the dataset is byte-identical on every machine and every run.
"""

from __future__ import annotations

import hashlib
import random
from datetime import date, timedelta

TODAY = date(2025, 5, 12)
CURRENCY = "PKR"
TERM = "Term 3 2024-25"

SUBJECTS = ["Mathematics", "English", "Science", "Urdu"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
ATTENDANCE_STATUSES = ["present", "absent", "late", "excused"]
SUBMISSION_STATUSES = ["submitted", "not_submitted", "late"]

_ATTENDANCE_HISTORY_DAYS = 20


def _rng(*parts: object) -> random.Random:
    key = "|".join(str(p) for p in parts).encode()
    digest = hashlib.sha256(key).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


TEACHERS = [
    ("TCH-01", "Farhan Iqbal", "Mathematics"),
    ("TCH-02", "Nadia Sheikh", "English"),
    ("TCH-03", "Imran Baig", "Science"),
    ("TCH-04", "Rubina Aslam", "Urdu"),
    ("TCH-05", "Kashif Mahmood", "Mathematics"),
    ("TCH-06", "Sana Tariq", "English"),
    ("TCH-07", "Waqar Younis", "Science"),
    ("TCH-08", "Hina Chaudhry", "Urdu"),
]

# Rosters are written out rather than generated so the duplicate names are explicit.
# "Sara Khan" appears in 5B and 7A. "Ali Hassan" appears in 6A and 7B.
# "Ahmed Raza" is unique, in 6A.
ROSTERS = {
    "5A": [
        "Bilal Anwar", "Ayesha Siddiqui", "Hamza Malik", "Zoya Rehman",
        "Usman Ghani", "Mariam Butt", "Danish Kamal", "Iqra Nawaz",
    ],
    "5B": [
        "Sara Khan", "Talha Qureshi", "Nimra Abbas", "Faizan Dar",
        "Areeba Junaid", "Shahzaib Alvi", "Laiba Mirza", "Rayan Sethi",
    ],
    "6A": [
        "Ahmed Raza", "Ali Hassan", "Hafsa Noor", "Junaid Farooq",
        "Emaan Zafar", "Saad Bhatti", "Anaya Gill", "Moiz Haider",
    ],
    "6B": [
        "Zainab Akhtar", "Umair Sultan", "Rida Pervaiz", "Haris Javed",
        "Alishba Rauf", "Taimoor Khalid", "Manahil Shah", "Arsalan Yousaf",
    ],
    "7A": [
        "Sara Khan", "Ibrahim Chishti", "Noor Fatima", "Zaid Ansari",
        "Hooriya Saleem", "Abdullah Waheed", "Amna Riaz", "Shayan Lodhi",
    ],
    "7B": [
        "Ali Hassan", "Kinza Aftab", "Musa Durrani", "Warda Hameed",
        "Ehsan Ullah", "Tehreem Zahid", "Raheel Mansoor", "Sadia Iftikhar",
    ],
}

GUARDIAN_SURNAME_TITLES = ["Mr.", "Mrs."]


def _class_id(class_name: str) -> str:
    return f"CLS-{class_name}"


def _grade_of(class_name: str) -> int:
    return int(class_name[0])


def build_classes() -> list[dict]:
    classes = []
    for index, class_name in enumerate(sorted(ROSTERS)):
        homeroom = TEACHERS[index % len(TEACHERS)]
        classes.append(
            {
                "class_id": _class_id(class_name),
                "class_name": class_name,
                "grade": _grade_of(class_name),
                "homeroom_teacher_id": homeroom[0],
                "homeroom_teacher_name": homeroom[1],
                "room": f"R-{100 + index}",
            }
        )
    return classes


def build_teachers() -> list[dict]:
    teachers = []
    for teacher_id, name, subject in TEACHERS:
        teaches = [
            c["class_name"]
            for c in build_classes()
            if c["homeroom_teacher_id"] == teacher_id
        ]
        teachers.append(
            {
                "teacher_id": teacher_id,
                "full_name": name,
                "subject": subject,
                "homeroom_of": teaches,
                "email": name.lower().replace(" ", ".") + "@schoolbox.example",
            }
        )
    return teachers


def build_students() -> list[dict]:
    students: list[dict] = []
    counter = 0
    for class_name in sorted(ROSTERS):
        for roll, full_name in enumerate(ROSTERS[class_name], start=1):
            counter += 1
            student_id = f"STU-{counter:04d}"
            rng = _rng("student", student_id)
            surname = full_name.split()[-1]
            guardians = [
                {
                    "guardian_id": f"GRD-{counter:04d}-{i + 1}",
                    "full_name": f"{title} {surname}",
                    "relationship": "father" if title == "Mr." else "mother",
                    "phone": f"+92-300-{rng.randint(1000000, 9999999)}",
                }
                for i, title in enumerate(GUARDIAN_SURNAME_TITLES)
            ]
            students.append(
                {
                    "student_id": student_id,
                    "full_name": full_name,
                    "class_name": class_name,
                    "class_id": _class_id(class_name),
                    "grade": _grade_of(class_name),
                    "roll_number": roll,
                    "date_of_birth": _birthday(class_name, rng).isoformat(),
                    "guardians": guardians,
                }
            )
    return students


def _birthday(class_name: str, rng: random.Random) -> date:
    age = 16 - _grade_of(class_name)
    year = TODAY.year - age
    return date(year, rng.randint(1, 12), rng.randint(1, 28))


def school_days(count: int, end: date = TODAY) -> list[date]:
    """The `count` most recent weekdays, ending on `end` inclusive."""
    days: list[date] = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return sorted(days)


def build_attendance(students: list[dict]) -> list[dict]:
    records = []
    days = school_days(_ATTENDANCE_HISTORY_DAYS)
    for student in students:
        for day in days:
            rng = _rng("attendance", student["student_id"], day.isoformat())
            status = rng.choices(
                ATTENDANCE_STATUSES, weights=[82, 9, 6, 3], k=1
            )[0]
            records.append(
                {
                    "record_id": f"ATT-{student['student_id']}-{day.isoformat()}",
                    "student_id": student["student_id"],
                    "class_name": student["class_name"],
                    "date": day.isoformat(),
                    "status": status,
                    "recorded_by": "TCH-01",
                }
            )
    return records


def build_assignments() -> list[dict]:
    assignments = []
    counter = 0
    for class_name in sorted(ROSTERS):
        for subject in SUBJECTS:
            counter += 1
            rng = _rng("assignment", class_name, subject)
            offset = rng.randint(-9, 4)
            due = TODAY + timedelta(days=offset)
            teacher = next(t for t in TEACHERS if t[2] == subject)
            assignments.append(
                {
                    "assignment_id": f"ASG-{counter:04d}",
                    "title": f"{subject} homework - week {rng.randint(3, 9)}",
                    "subject": subject,
                    "class_name": class_name,
                    "grade": _grade_of(class_name),
                    "teacher_id": teacher[0],
                    "assigned_date": (due - timedelta(days=7)).isoformat(),
                    "due_date": due.isoformat(),
                    "max_score": 20,
                    "status": "closed" if due < TODAY else "open",
                }
            )
    return assignments


def build_submissions(students: list[dict], assignments: list[dict]) -> list[dict]:
    by_class: dict[str, list[dict]] = {}
    for student in students:
        by_class.setdefault(student["class_name"], []).append(student)

    submissions = []
    for assignment in assignments:
        for student in by_class[assignment["class_name"]]:
            rng = _rng(
                "submission", assignment["assignment_id"], student["student_id"]
            )
            status = rng.choices(SUBMISSION_STATUSES, weights=[68, 22, 10], k=1)[0]
            score = None
            submitted_at = None
            if status != "not_submitted":
                submitted_at = assignment["due_date"]
                if assignment["status"] == "closed":
                    score = rng.randint(9, 20)
            submissions.append(
                {
                    "submission_id": (
                        f"SUB-{assignment['assignment_id']}-{student['student_id']}"
                    ),
                    "assignment_id": assignment["assignment_id"],
                    "student_id": student["student_id"],
                    "student_name": student["full_name"],
                    "class_name": student["class_name"],
                    "status": status,
                    "submitted_at": submitted_at,
                    "score": score,
                }
            )
    return submissions


def build_fees(students: list[dict]) -> list[dict]:
    fees = []
    for student in students:
        rng = _rng("fee", student["student_id"])
        total = rng.choice([42000, 45000, 48000, 52000])
        paid_fraction = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
        paid = int(total * paid_fraction)
        fees.append(
            {
                "invoice_id": f"FEE-{student['student_id']}-T3",
                "student_id": student["student_id"],
                "student_name": student["full_name"],
                "term": TERM,
                "currency": CURRENCY,
                "total_amount": total,
                "amount_paid": paid,
                "balance": total - paid,
                "due_date": (TODAY + timedelta(days=11)).isoformat(),
                "status": "paid" if paid >= total else "outstanding",
                "payments": [],
            }
        )
    return fees


def build_academic_records(students: list[dict]) -> list[dict]:
    records = []
    for student in students:
        for subject in SUBJECTS:
            rng = _rng("academic", student["student_id"], subject)
            percentage = rng.randint(41, 97)
            records.append(
                {
                    "student_id": student["student_id"],
                    "term": TERM,
                    "subject": subject,
                    "percentage": percentage,
                    "grade_letter": _grade_letter(percentage),
                }
            )
    return records


def _grade_letter(percentage: int) -> str:
    for threshold, letter in ((90, "A+"), (80, "A"), (70, "B"), (60, "C"), (50, "D")):
        if percentage >= threshold:
            return letter
    return "F"


def build_timetable() -> list[dict]:
    entries = []
    for class_name in sorted(ROSTERS):
        for day in WEEKDAYS:
            rng = _rng("timetable", class_name, day)
            order = SUBJECTS * 2
            rng.shuffle(order)
            for period in range(1, 6):
                subject = order[period - 1]
                teacher = next(t for t in TEACHERS if t[2] == subject)
                entries.append(
                    {
                        "class_name": class_name,
                        "day": day,
                        "period": period,
                        "start_time": f"{7 + period:02d}:30",
                        "end_time": f"{8 + period:02d}:15",
                        "subject": subject,
                        "teacher_id": teacher[0],
                        "teacher_name": teacher[1],
                    }
                )
    return entries


def build_dataset() -> dict:
    students = build_students()
    assignments = build_assignments()
    return {
        "meta": {
            "today": TODAY.isoformat(),
            "today_weekday": WEEKDAYS[TODAY.weekday()],
            "term": TERM,
            "currency": CURRENCY,
        },
        "classes": build_classes(),
        "teachers": build_teachers(),
        "students": students,
        "attendance": build_attendance(students),
        "assignments": assignments,
        "submissions": build_submissions(students, assignments),
        "fees": build_fees(students),
        "academic_records": build_academic_records(students),
        "timetable": build_timetable(),
    }
