"""Generate routing training data from the simulator's own seed data.

The 54 suite prompts are NEVER used for training - they are the held-out test set, so the
router's accuracy on them is a genuine out-of-sample number, unlike the rules in
pipeline_rules.py which were written from this suite's failures.

THE LABEL IS THE ANSWERING TOOL (the intent), NOT the first call. pipeline_rules.first_call
derives the lookup step itself: given get_fee_status and a name it issues search_student,
and given get_submissions without an assignment id it issues list_assignments. So a router
that predicts the lookup step terminates the chain - plan_next then has nothing left to do.
Label what the user ultimately wants:
  "how much does <name> owe"        -> get_fee_status      (NOT search_student)
  "mark <name> absent"              -> mark_attendance     (NOT search_student)
  "who missed the maths homework"   -> get_submissions     (NOT list_assignments)
  "find the student called <name>"  -> search_student      (finding IS the request)
  "what homework is open for 6A"    -> list_assignments    (the list IS the answer)

usage: uv run --python 3.14 python scripts/router_data.py
"""

import argparse
import json
import random
from pathlib import Path

from canit.sim.seed import ROSTERS, SUBJECTS, TODAY, WEEKDAYS, build_dataset

LABELS = ["search_student", "get_student", "get_attendance", "get_class_students",
          "get_timetable", "list_assignments", "get_assignment", "get_submissions",
          "get_fee_status", "get_academic_record", "mark_attendance",
          "record_fee_payment", "update_submission_status", "none"]

# Phrasings a school office would actually use. {slots} are filled from real seed entities.
# Phrasings a school office would actually use, grouped by the FIRST tool they require.
# {name}/{first}/{last} mean the prompt names a person -> search_student must run first.
# {sid}/{aid}/{cls} mean the prompt carries an identifier -> the answering tool runs first.
# Phrasings a school office would actually use, grouped by the ANSWERING tool. Both the
# {name} and {sid} forms carry the same label: first_call inserts the lookup when needed.
TEMPLATES = {
    "search_student": [
        "find the student called {name}", "look up {name}", "is there a student named {name}",
        "which class is {name} in", "search for {first}", "do we have a {first} on the roll",
        "find {first} {last}", "can you locate {name}", "check if {name} is enrolled",
        "is {first} on the register", "which class does {name} belong to",
        "do we have anyone called {first}", "find me {name}'s student id",
    ],
    "get_student": [
        "pull up the profile for {sid}", "show me the details for {sid}",
        "who is {name}'s guardian", "guardian contact for {name}",
        "what is {name}'s roll number", "profile for {sid}",
        "give me {name}'s details", "which guardian do we have for {sid}",
        "contact details for {name}", "what class and roll number is {sid}",
        "guardian details for {name}", "open the record for {sid}",
    ],
    "get_attendance": [
        "was {name} present today", "attendance for {sid} this term",
        "did {name} come in on {weekday}", "how many days has {name} missed",
        "was {sid} absent yesterday", "check {name}'s attendance",
        "has {name} been late recently", "attendance record for {name}",
        "is {name} in today", "did {sid} attend on {date}",
        "how many absences does {sid} have", "was {name} marked late this week",
    ],
    "get_class_students": [
        "class list for {cls}", "who is in {cls}", "roster for {cls}",
        "how many students are in {cls}", "show me the students in {cls}",
        "list everyone in {cls}", "give me the {cls} roll", "students of class {cls}",
        "print the register for {cls}", "names in {cls}",
        "who is the homeroom teacher for {cls}", "which teacher has {cls}",
        "who takes the register for {cls}", "form tutor for {cls}",
    ],
    "get_timetable": [
        "timetable for {cls}", "what does {cls} have on {weekday}",
        "schedule for {cls} tomorrow", "{cls} periods on {weekday}",
        "what lessons does {cls} have today", "show the {cls} timetable",
        "when does {cls} have {subject}", "class schedule for {cls}",
    ],
    "list_assignments": [
        "what homework is open for {cls}", "list the assignments for {cls}",
        "which assignments are still open for {cls}", "homework set for {cls}",
        "what work has been set for {cls}", "any assignments due for {cls}",
        "show open homework for {cls}", "assignments for {cls} this term",
        "what homework does {cls} have", "how many assignments are open for {cls}",
    ],
    "get_assignment": [
        "details of assignment {aid}", "when is {aid} due", "what is {aid} about",
        "show me {aid}", "tell me about assignment {aid}", "due date for {aid}",
        "which class is {aid} for", "open {aid}",
    ],
    "get_submissions": [
        "who handed in {aid}", "who missed {aid}", "submissions for {aid}",
        "who has not submitted {aid}", "did everyone turn in {aid}",
        "list the submissions for {aid}", "who is missing {aid}",
        # described rather than identified: first_call inserts list_assignments
        "who has not handed in the {subject} homework for {cls}",
        "which students in {cls} are missing the {subject} work",
        "who missed the {subject} homework in {cls}",
        "has everyone in {cls} submitted their {subject} homework",
        "which grade {grade} students have not submitted the {subject} homework",
        "is anyone in {cls} missing {subject} work",
        "did {name} hand in the {subject} homework",
    ],
    "get_fee_status": [
        "fee balance for {sid}", "how much does {name} owe",
        "outstanding fees for {name}", "what is {name}'s fee status",
        "does {sid} have unpaid fees", "fees owed by {name}",
        "check the fee account for {name}", "any balance on {sid}",
        "does {name} have a balance", "what does {sid} owe",
        "has {name} paid their fees", "fee status for {name}",
    ],
    "get_academic_record": [
        "grades for {name}", "academic record for {sid}",
        "what did {name} get in {subject}", "{name}'s marks this term",
        "show me {sid}'s results", "how is {name} doing in {subject}",
        "report card for {name}", "term grades for {sid}",
        "what did {sid} score in {subject}", "marks for {name}",
    ],
    "mark_attendance": [
        "mark {name} absent today", "record {name} as late",
        "set {name} present for {weekday}", "mark {sid} absent",
        "put {name} down as late today", "register {name} as present",
        "{name} is absent today, please record it", "mark {sid} late for {date}",
        "record {name} as excused today", "set {sid} to absent for {weekday}",
    ],
    "record_fee_payment": [
        "record a payment of {amount} for {name}", "{name} paid {amount} today",
        "log a fee payment of {amount} for {sid}", "enter {amount} against {name}'s fees",
        "receipt {amount} for {sid}", "take a payment of {amount} from {name}",
        "{name}'s guardian paid {amount}", "post {amount} to {sid}",
    ],
    "update_submission_status": [
        "mark {name}'s {subject} homework as submitted",
        "update {aid} for {name} to graded", "set {name}'s submission for {aid} to late",
        "record that {name} handed in {aid}", "change {name}'s status on {aid} to submitted",
        "mark {aid} as graded for {sid}", "{name} turned in {aid}, update it",
    ],
    "none": [
        "what can you do", "what are you able to help with", "list your capabilities",
        "what do you have access to", "what kind of questions can you answer",
        "what is 12 plus 7", "how much is 45 times 3", "what is 120 divided by 4",
        "add 17 and 25 for me", "what is 15 percent of 200",
        "what does late mean here", "explain the submission statuses",
        "what is the difference between absent and late", "what do the fee statuses mean",
        "what does pending mean", "explain what excused means",
        "email the parents of {cls}", "send a text message to {name}'s guardian",
        "delete the attendance records for {cls}", "remove {name} from the school",
        "print certificates for {cls}", "export everything to a spreadsheet",
        "call {name}'s guardian", "book a parents evening for {cls}",
        "delete {name}'s record", "wipe the fee history for {cls}",
    ],
}


def generate(n_per_label: int, seed: int = 20260923) -> list[dict]:
    rng = random.Random(seed)
    d = build_dataset()
    students = d["students"] if isinstance(d["students"], list) else list(d["students"].values())
    assignments = d["assignments"] if isinstance(d["assignments"], list) else list(d["assignments"].values())
    classes = list(ROSTERS.keys())
    subjects = list(SUBJECTS) if not isinstance(SUBJECTS, dict) else list(SUBJECTS.values())
    aids = [a.get("assignment_id") for a in assignments if a.get("assignment_id")]

    rows = []
    for label, templates in TEMPLATES.items():
        for _ in range(n_per_label):
            s = rng.choice(students)
            first, _, last = s["full_name"].partition(" ")
            prompt = rng.choice(templates).format(
                name=s["full_name"], first=first, last=last or first,
                sid=s["student_id"], cls=rng.choice(classes),
                weekday=rng.choice(WEEKDAYS), subject=rng.choice(subjects),
                aid=rng.choice(aids) if aids else "ASG-0001",
                amount=f"{rng.choice([500, 1000, 1500, 2000, 2500, 5000]):,}",
                date=str(TODAY), grade=rng.choice([5, 6, 7]),
            )
            if rng.random() < 0.25:            # office staff do not always capitalise or punctuate
                prompt = prompt.capitalize() + rng.choice(["", "?", ".", " please"])
            rows.append({"prompt": prompt, "label": label})
    rng.shuffle(rows)
    return rows


def held_out() -> list[dict]:
    """The 54 real suite prompts with their first required tool. TEST ONLY."""
    from canit.scenarios import ALL_SCENARIOS
    rows = []
    for sc in ALL_SCENARIOS:
        calls = sc.expect.get("required_calls") or []
        tools = [c.get("tool") for c in calls if c.get("tool")]
        # the answering tool is the LAST call in the chain; first_call derives the lookups
        rows.append({"prompt": sc.prompt, "label": tools[-1] if tools else "none", "id": sc.id})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="examples per label")
    ap.add_argument("--out", default="data/router_train.jsonl")
    ap.add_argument("--test-out", default="data/router_test.jsonl")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    train, test = generate(a.n), held_out()
    for path, rows in ((a.out, train), (a.test_out, test)):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"train {len(train)} rows -> {a.out}")
    print(f"test  {len(test)} rows (real suite prompts, never trained on) -> {a.test_out}")
