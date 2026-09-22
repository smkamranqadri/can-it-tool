"""Deterministic rules for the Laya pipeline: argument extraction, call chaining, safety
policy and fixed replies.

No model and no torch here, so the rules can be tested against the simulator on their own
(scripts/pipeline_dryrun.py). Laya only ranks the tools; these functions decide what to
call, what to refuse, and which replies need no model at all.

  extract(prompt)                -> dict of argument candidates found in the prompt
  choose_tool(probs, prompt)     -> Laya's pick, with a write tool demoted for a read question
  policy_refusal(prompt, tool, a) -> refusal text for requests that must not reach a write tool
  first_call(tool, args)         -> (tool, arguments) to issue first, or None if unresolvable
  plan_next(plan, messages)      -> next calls after the latest tool results, [] when done
  fixed_reply(plan, messages)    -> a reply built from tool results (not found, several
                                    matches, pending confirmation), or None to let the LLM answer

Safety is structural: a write tool is only ever called by first_call/plan_next for one
resolved student, never with a confirmation token, so every write stops at a confirmation
request. Anything else that asks for a change is refused by policy_refusal.
"""

import json
import re

WRITE_TOOLS = {"mark_attendance", "record_fee_payment", "update_submission_status"}
STUDENT_TOOLS = {"get_student", "get_attendance", "get_fee_status", "get_academic_record",
                 "mark_attendance", "record_fee_payment"}
MAX_FANOUT = 12

WEEKDAY = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow)"
SUBJECTS = {"math": "Mathematics", "maths": "Mathematics", "mathematics": "Mathematics", "english": "English",
            "science": "Science", "urdu": "Urdu"}
LEAD = {"is", "was", "how", "what", "who", "which", "when", "did", "does", "has", "have", "mark", "show", "give",
        "tell", "pull", "get", "list", "find", "record", "update", "just", "set", "delete", "please", "can",
        "could", "the", "for", "compare", "there", "i", "before", "quick", "email", "in", "on", "grade", "class",
        "student", "students", "assignment", "fee", "fees", "today", "tomorrow", "yes", "no", "you", "we",
        "my", "our", "a", "an"}
WRITE_VERBS = r"\b(mark|record|update|set|change|fix|correct|enter|log|clear|settle[ds]?|delete|remove|erase|wipe)\b"


def write_intent(prompt):
    return re.search(WRITE_VERBS, prompt, re.I) is not None


def extract(prompt):
    p = prompt
    a = {}
    if m := re.search(r"\bSTU-\d{4}\b", p):
        a["student_id"] = m.group(0)
    if m := re.search(r"\bASG-\d{4}\b", p):
        a["assignment_id"] = m.group(0)
    if m := re.search(r"\b(\d{1,2}[A-Z])\b", p):
        a["class_name"] = m.group(1)
    if m := re.search(r"\bgrade\s+(\d{1,2})\b", p, re.I):
        a["grade"] = int(m.group(1))
    if m := re.search(rf"\b{WEEKDAY}\b", p, re.I):
        a["day"] = m.group(1).lower()
    for k, v in SUBJECTS.items():
        if re.search(rf"\b{k}\b", p, re.I):
            a["subject"] = v
            break
    if m := re.search(r"\b(present|absent|late|excused)\b", p, re.I):
        a["attendance_status"] = m.group(1).lower()
    if re.search(r"\bafter the deadline|\blate\b", p, re.I):
        a["submission_status"] = "late"
    elif re.search(r"\b(handed in|submitted)\b", p, re.I):
        a["submission_status"] = "submitted"
    if re.search(r"\b(not submitted|has not|did not|missed|not handed)\b", p, re.I):
        a["filter_status"] = "not_submitted"
    if re.search(r"\b(in full|settled|paid off|full balance|whole balance|clear(ed)? (the|his|her) (balance|fees))\b",
                 p, re.I):
        a["in_full"] = True
    if re.search(r"\b(both|each|all of them|compare)\b", p, re.I):
        a["every_match"] = True
    p = re.sub(r"\b(STU|ASG)-\d{4}\b|\b\d{1,2}[A-Z]\b", " ", p)  # ids and class codes are not names or amounts
    if m := re.search(r"\b(\d[\d,]{2,})\s*(rupees|rs|pkr)?", p, re.I):
        a["amount"] = int(m.group(1).replace(",", ""))
    for meth, key in (("cash", "cash"), ("cheque", "cheque"), ("card", "card"), ("bank transfer", "bank_transfer")):
        if meth in p.lower():
            a["method"] = key
    # a name: the longest run of capitalized words that is not a leading verb or known token
    words = re.findall(r"[A-Za-z][A-Za-z']*", p)
    best, cur = [], []
    for w in words:
        base = re.sub(r"'s$", "", w)
        if base[:1].isupper() and base.lower() not in LEAD and base.lower() not in SUBJECTS \
                and not re.fullmatch(WEEKDAY, base, re.I):
            cur.append(base)
        else:
            best, cur = (cur if len(cur) > len(best) else best), []
    best = cur if len(cur) > len(best) else best
    if best:
        a["name"] = " ".join(best)
    return a


def choose_tool(probs, prompt):
    """Laya's top tool, except that a write tool is never chosen for a question that asks for
    no change: the most probable read tool is used instead."""
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    top = ranked[0][0]
    if top in WRITE_TOOLS and not write_intent(prompt):
        return next(n for n, _ in ranked if n not in WRITE_TOOLS and n != "none")
    # a homework status change needs homework context; without it, "late"/"absent" is attendance
    homework = re.search(r"\b(homework|assignment|submission|handed in|submit\w*|ASG-\d{4})\b", prompt, re.I) or \
        any(re.search(rf"\b{k}\b", prompt, re.I) for k in SUBJECTS)
    if top == "update_submission_status" and not homework and \
            re.search(r"\b(present|absent|late|excused)\b", prompt, re.I):
        return "mark_attendance"
    return top


def policy_refusal(prompt, tool, a):
    """Refusal text for change requests that must not reach a write tool, else None."""
    if re.search(r"\b(delete|remove|erase|wipe|purge)\b", prompt, re.I):
        return ("I can't delete records. The system keeps attendance, fee and grade history and has no way to "
                "remove it. I can look records up, or prepare a correction for one student for your approval.")
    if re.search(r"\b(email|e-mail|text|sms|whatsapp|message|call|phone|notify|print|post)\b", prompt, re.I):
        return ("I can't send messages or emails, and there is no tool for it. I can look the information up "
                "here so you can pass it on.")
    if re.search(r"\btoken\b", prompt, re.I) and re.search(r"\b(reuse|same|again|another|other)\b", prompt, re.I):
        return ("I can't reuse a confirmation token. Each token authorizes exactly the one change it was issued "
                "for, so every student and every change needs its own confirmation.")
    if not write_intent(prompt):
        return None
    if tool == "none":
        return ("I can't make that change. There is no tool for editing that record; I can only prepare "
                "attendance, fee payment and homework submission changes for your approval.")
    if tool in WRITE_TOOLS:
        if re.search(r"\b(says|said|claims?|claimed|told (me|us)|according to)\b", prompt, re.I):
            return ("I can't change records based on an unverified claim. I can look up the current record, and "
                    "prepare the change once there is a receipt or other verification.")
        if re.search(r"\b(whole|every|everyone|entire|all)\b", prompt, re.I) or \
                ("student_id" not in a and "name" not in a):
            return ("I can't change records in bulk. Changes are made one student at a time, each with your "
                    "confirmation. Tell me which student and I will prepare it.")
    return None


def lookup_args(a):
    return {"name": a["name"], **{k: a[k] for k in ("class_name", "grade") if k in a}}


def first_call(tool, a):
    """The first call to issue for `tool`, or None if the arguments cannot be resolved."""
    if tool == "get_class_students":
        return ("get_class_students", {"class_name": a["class_name"]}) if "class_name" in a else None
    if tool == "get_timetable":
        return ("get_timetable", {"class_name": a["class_name"], "day": a.get("day", "today")}) \
            if "class_name" in a else None
    if tool == "list_assignments":
        return ("list_assignments", {k: a[k] for k in ("class_name", "grade", "subject") if k in a})
    if tool == "search_student":
        return ("search_student", lookup_args(a)) if "name" in a else None
    if tool in ("get_assignment", "get_submissions", "update_submission_status") and "assignment_id" not in a:
        if tool == "update_submission_status" and "student_id" not in a and "name" in a:
            return ("search_student", lookup_args(a))
        return ("list_assignments", {k: a[k] for k in ("class_name", "grade", "subject") if k in a})
    if tool in STUDENT_TOOLS and "student_id" not in a:
        if "name" in a:
            return ("search_student", lookup_args(a))
        if "class_name" in a and tool not in WRITE_TOOLS and not ({"subject", "filter_status"} & a.keys()):
            # "everyone in 5B absent today": read each student on the roster (not for homework questions)
            return ("get_class_students", {"class_name": a["class_name"]})
        return None
    return target_call(tool, a)


def target_call(tool, a):
    sid, aid = a.get("student_id"), a.get("assignment_id")
    if tool in ("get_student", "get_attendance", "get_fee_status"):
        return (tool, {"student_id": sid})
    if tool == "get_academic_record":
        return (tool, {"student_id": sid, **({"subject": a["subject"]} if "subject" in a else {})})
    if tool == "get_assignment":
        return (tool, {"assignment_id": aid})
    if tool == "get_submissions":
        return (tool, {"assignment_id": aid, **({"status": a["filter_status"]} if "filter_status" in a else {}),
                       **({"student_id": sid} if sid else {})})
    if tool == "mark_attendance":
        return (tool, {"student_id": sid, "status": a["attendance_status"]}) if "attendance_status" in a else None
    if tool == "record_fee_payment":
        if "amount" not in a:
            return ("get_fee_status", {"student_id": sid}) if a.get("in_full") else None
        return (tool, {"student_id": sid, "amount": a["amount"], **({"method": a["method"]} if "method" in a else {})})
    if tool == "update_submission_status":
        return (tool, {"assignment_id": aid, "student_id": sid, "status": a.get("submission_status", "submitted")}) \
            if aid and sid else None
    return None


def _tail(messages):
    """The latest assistant tool calls and the tool results that answered them."""
    tail = []
    for m in reversed(messages):
        if m["role"] != "tool":
            break
        tail.insert(0, m)
    last = next((m for m in reversed(messages) if m["role"] == "assistant"), {})
    calls = last.get("tool_calls") or []
    results = [json.loads(m["content"]) if m["content"].strip().startswith("{") else {} for m in tail]
    return calls, results


def plan_next(plan, messages):
    """Decide the next tool calls from the latest tool results, or [] when done."""
    calls, results = _tail(messages)
    if not calls:
        return []
    retry = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
             for c, r in zip(calls, results) if r.get("retryable")]
    last = calls[0]["function"]["name"]
    tool, a = plan["tool"], plan["args"]
    if retry and last == "search_student" and "class_name" in a:
        # search is down and the class is known: the roster is one call, a retry may fail again
        return [("get_class_students", {"class_name": a["class_name"]})]
    if retry and plan["retries"] < 2:
        plan["retries"] += 1
        return retry
    if len(results) != 1 or tool == last:
        return []
    r = results[0]

    if last == "search_student":
        matches = r.get("matches") or []
        if len(matches) == 1:
            a["student_id"] = matches[0]["student_id"]
            nxt = first_call(tool, a) if tool == "update_submission_status" and "assignment_id" not in a \
                else target_call(tool, a)
            return [nxt] if nxt else []
        if 1 < len(matches) <= MAX_FANOUT and a.get("every_match") and tool not in WRITE_TOOLS:
            return [c for c in (target_call(tool, {**a, "student_id": x["student_id"]}) for x in matches) if c]
        return []  # zero or several matches: fixed_reply answers

    if last == "get_class_students":
        students = r.get("students") or []
        if "name" in a:
            hit = [s for s in students if a["name"].lower() in s.get("full_name", "").lower()]
            if len(hit) == 1:
                a["student_id"] = hit[0]["student_id"]
                nxt = target_call(tool, a)
                return [nxt] if nxt else []
            return []
        if tool not in WRITE_TOOLS and len(students) <= MAX_FANOUT:
            return [c for c in (target_call(tool, {**a, "student_id": s["student_id"]}) for s in students) if c]
        return []

    if last == "list_assignments":
        found = r.get("assignments") or []
        if tool == "get_submissions" and 1 <= len(found) <= 3:
            return [target_call(tool, {**a, "assignment_id": x["assignment_id"]}) for x in found]
        if len(found) == 1:
            a["assignment_id"] = found[0]["assignment_id"]
            nxt = target_call(tool, a)
            return [nxt] if nxt else []
        return []

    if last == "get_fee_status" and tool == "record_fee_payment" and a.get("in_full"):
        if isinstance(r.get("balance"), int) and r["balance"] > 0:
            a["amount"] = r["balance"]
            nxt = target_call(tool, a)
            return [nxt] if nxt else []
    return []


def fixed_reply(plan, messages):
    """A reply that needs no model: built only from the latest tool results."""
    _, results = _tail(messages)
    if not results:
        return None
    pending = [r["summary"] for r in results if r.get("status") == "confirmation_required" and r.get("summary")]
    if pending:
        return " ".join(pending) + " No change has been made yet. Please confirm and I will go ahead."
    for r in results:
        if r.get("status") == "not_found":
            msg = r.get("message", "").strip()
            return f"I couldn't find that record. {msg}" if msg.lower().startswith("no ") else msg
    if len(results) == 1 and "matches" in results[0]:
        r, needs_one = results[0], plan["tool"] != "search_student"
        matches = r["matches"]
        if not matches:
            return (f"I couldn't find a student named {r.get('query', 'that')}. "
                    "Please check the spelling, or tell me their class.")
        if len(matches) > 1 and needs_one and not (plan["args"].get("every_match") and plan["tool"] not in WRITE_TOOLS):
            who = ", ".join(f"{m['full_name']} in {m['class_name']} ({m['student_id']})" for m in matches)
            return f"I found {len(matches)} students matching {r.get('query')}: {who}. Which one do you mean?"
    return None
