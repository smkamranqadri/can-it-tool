"""Generate a HELD-OUT router test set with an LLM, independently of our own templates.

Why not write these by hand: the same author wrote pipeline_rules.py and the training
templates in router_data.py, so hand-written test prompts would inherit that phrasing. Here
an LLM is given only a plain-English description of ONE capability plus some real entity
names - never the tool names, the scenarios, the rules or the templates - and asked for
realistic school-office requests. The label is the capability that was requested, so it
comes from the generation condition rather than from a post-hoc judgement.

Independence is partial, not perfect: the capability descriptions are still ours, and a
model asked for "requests needing X" will phrase them helpfully. Treat this as a harder
test than the training distribution, not as production traffic.

usage: uv run --python 3.14 python scripts/heldout_prompts.py --base-url http://127.0.0.1:8081/v1
"""

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

from canit.sim.seed import ROSTERS, SUBJECTS, build_dataset

# Plain English only. No tool names, no argument names, nothing from the rules.
CAPABILITIES = {
    "search_student": "finding out which student is meant, when only a name is given",
    "get_student": "looking up a student's own details, like their guardian or roll number",
    "get_attendance": "checking whether a student was in school, or how often they were away",
    "get_class_students": "getting the list of children in a class, or who teaches that class",
    "get_timetable": "finding out what lessons a class has, and when",
    "list_assignments": "finding out what homework has been set for a class",
    "get_assignment": "looking up the details of one specific piece of homework",
    "get_submissions": "finding out who handed a piece of homework in and who did not",
    "get_fee_status": "checking what school fees a family still owes",
    "get_academic_record": "looking up a student's marks or results",
    "mark_attendance": "recording that a student was present, late or away today",
    "record_fee_payment": "recording that a family has paid some fees",
    "update_submission_status": "recording that a student handed a piece of homework in",
    "none": "things a school records system cannot do at all, like sending emails, deleting "
            "records, or general questions and simple sums that need no records",
}

SYSTEM = ("You write realistic one-line requests that a school office administrator would type "
          "to an assistant. Write them the way a busy person really types: sometimes terse, "
          "sometimes a full sentence, sometimes with context before the ask. Vary the wording a "
          "lot. Output ONLY the requests, one per line, no numbering, no quotes, no commentary.")


def chat(base, body):
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--out", default="data/heldout_router.jsonl")
    a = ap.parse_args()

    rng = random.Random(7)
    d = build_dataset()
    students = d["students"] if isinstance(d["students"], list) else list(d["students"].values())
    classes = list(ROSTERS.keys())
    subjects = list(SUBJECTS) if not isinstance(SUBJECTS, dict) else list(SUBJECTS.values())

    rows, seen = [], set()
    for label, description in CAPABILITIES.items():
        names = [s["full_name"] for s in rng.sample(students, 6)]
        ctx = (f"Real student names you may use: {', '.join(names)}. "
               f"Real class names: {', '.join(rng.sample(classes, 4))}. "
               f"Subjects taught: {', '.join(subjects)}.")
        user = (f"{ctx}\n\nWrite {a.per} different requests where the administrator needs help with: "
                f"{description}.\nOne per line.")
        text = chat(a.base_url, {"messages": [{"role": "system", "content": SYSTEM},
                                              {"role": "user", "content": user}],
                                 "temperature": 0.9, "max_tokens": 700})
        n = 0
        for line in text.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
            if len(line) < 8 or len(line) > 200 or line.lower() in seen:
                continue
            seen.add(line.lower())
            rows.append({"prompt": line, "label": label})
            n += 1
        print(f"{label:26s} {n:3d} prompts", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n{len(rows)} held-out prompts -> {a.out}")


if __name__ == "__main__":
    main()
