"""Laya tool-routing proxy between the can-it-tool harness and llama-server.

First turn of a conversation: Laya answers one `choice` question - which tool does this
request need - over short generic labels for the 13 tools plus "none". The proxy keeps
the most probable tools until their probability mass reaches COVER (at most MAX_PICK),
then adds lookup dependencies read from the schemas: a tool taking `student_id` brings
`search_student`, one taking `assignment_id` brings `list_assignments`. If "none" wins,
the LLM gets no tools. The chosen set is fixed for the rest of that conversation.

The LLM still fills arguments, chains calls, and writes the reply. The harness's
wall-clock wait includes Laya's time. Decisions are logged as JSON lines on stderr.

usage: laya_proxy.py [upstream] [port]
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from laya import Router

UPSTREAM = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8090
COVER = float(os.environ.get("LAYA_COVER", "0.8"))
MAX_PICK = int(os.environ.get("LAYA_MAX_PICK", "3"))
torch.set_num_threads(int(os.environ.get("LAYA_THREADS", "8")))

# Short, generic labels: Laya's option budget is ~192 tokens across all options.
LABELS = {
    "search_student": "find a student by name",
    "get_student": "a student's profile or guardians",
    "get_attendance": "whether a student was present, absent or late",
    "list_assignments": "list homework or assignments",
    "get_assignment": "details of one assignment",
    "get_submissions": "who handed in or missed an assignment",
    "get_fee_status": "fees a student owes or has paid",
    "get_class_students": "class list or roster",
    "get_timetable": "class timetable or schedule",
    "get_academic_record": "a student's grades or marks",
    "mark_attendance": "mark or change attendance",
    "record_fee_payment": "record a fee payment",
    "update_submission_status": "mark homework as submitted or late",
    "none": "a general question or something none of these can do",
}
QUESTION = {"tool": {"type": "choice", "instructions": "Which school records action does this request need?",
                     "criteria": LABELS}}

router = Router()  # English prompts: only the English checkpoint loads, on the warm-up call
router.predict({"body": "warm up"}, QUESTION)
cache = {}


def route(prompt, tools):
    if prompt in cache:
        return cache[prompt]
    t0 = time.perf_counter()
    probs = router.predict({"body": prompt}, QUESTION)["answers"]["tool"]["probabilities"]
    ms = (time.perf_counter() - t0) * 1000
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    picked, mass = [], 0.0
    if ranked[0][0] != "none":
        for name, p in ranked:
            if name == "none":
                continue
            picked.append(name)
            mass += p
            if mass >= COVER or len(picked) >= MAX_PICK:
                break
    by_name = {t["function"]["name"]: t for t in tools}
    params = lambda n: by_name[n]["function"]["parameters"].get("properties", {})  # noqa: E731
    for n in list(picked):
        if "student_id" in params(n) and "search_student" not in picked:
            picked.append("search_student")
        if "assignment_id" in params(n) and "list_assignments" not in picked:
            picked.append("list_assignments")
    cache[prompt] = picked
    sys.stderr.write(json.dumps({"prompt": prompt[:80], "laya_ms": round(ms, 1), "picked": picked,
                                 "top": ranked[:4]}) + "\n")
    sys.stderr.flush()
    return picked


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        with urllib.request.urlopen(UPSTREAM + self.path) as r:
            self._send(r.status, r.read())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tools = body.get("tools")
        if tools:
            prompt = next(m["content"] for m in body["messages"] if m["role"] == "user")
            picked = route(prompt, tools)
            body["tools"] = [t for t in tools if t["function"]["name"] in picked]
            if not body["tools"]:
                body.pop("tools")
                body.pop("tool_choice", None)
        req = urllib.request.Request(UPSTREAM + self.path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                self._send(r.status, r.read())
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read())

    def _send(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
