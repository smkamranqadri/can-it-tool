"""Laya-routed pipeline: Laya picks the tool, rules fill the arguments and make the calls,
and the LLM only writes the answer from the results - with no tool schemas in its prompt.

It sits where the model sits, as an OpenAI endpoint in front of llama-server, so the
unchanged can-it-tool harness executes every tool call against its simulator and scores
the run exactly as it scores a model:

  turn 1   Laya ranks the 13 tools + none in one `choice` forward pass. choose_tool never
           picks a write tool for a question that asks for no change. policy_refusal
           refuses deletes, token reuse, bulk changes, changes on an unverified claim and
           changes no tool supports. Otherwise the rules build the first call. When they
           cannot, or Laya says "none" for a prompt that names a student, class or
           assignment, the LLM takes over with read-only tools (the fallback).
  turn n   plan_next chains lookups (name -> id, roster -> each student, assignment list ->
           submissions, balance -> full payment), retries retryable errors, and falls back
           to the roster when search is down.
  answer   fixed_reply answers without a model when the result is a pending confirmation,
           a not-found, or several matching students. Everything else is one llama-server
           call with a short answer-only system prompt, the question and the results.

Writes are only ever issued for one resolved student and never with a confirmation token,
so every write ends at a confirmation request; the fallback never sees a write tool.

pipeline version 2.1: results/17-19. Version 2.2 (results/20-22) dropped the "include the
figures" sentence from ANSWER_PROMPT; that cost Qwen3.5-0.8B 9 points (83% -> 74%) and left
the tiny answer models unchanged within noise, so 2.1's wording is kept. results/14-16 are
version 2; results/11 and results/latency/L19, L20, L30-L37 are version 1 (Laya refusal
question, full system prompt, no fixed replies).

usage: laya_pipeline.py [upstream] [port]
"""

import itertools
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from laya import Router

from pipeline_rules import WRITE_TOOLS, choose_tool, extract, first_call, fixed_reply, plan_next, policy_refusal

UPSTREAM = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8090
torch.set_num_threads(int(os.environ.get("LAYA_THREADS", "8")))
ANSWER_MAX_TOKENS = int(os.environ.get("ANSWER_MAX_TOKENS", "400"))

LABELS = {
    "search_student": "find a student by name",
    "get_student": "a student's profile or guardians",
    "get_attendance": "whether a student was present, absent or late",
    "list_assignments": "list homework or assignments",
    "get_assignment": "details of one assignment",
    "get_submissions": "who handed in or missed an assignment",
    "get_fee_status": "fees a student owes or has paid",
    "get_class_students": "class list, roster or homeroom teacher",
    "get_timetable": "class timetable or schedule",
    "get_academic_record": "a student's grades or marks",
    "mark_attendance": "mark or change attendance",
    "record_fee_payment": "record a fee payment",
    "update_submission_status": "mark homework as submitted or late",
    "none": "a general question or something none of these can do",
}
QUESTIONS = {"tool": {"type": "choice", "instructions": "Which school records action does this request need?",
                      "criteria": LABELS}}

ANSWER_PROMPT = """You are the SchoolBox assistant for school office staff. {today}
Answer the staff member's question using only the records provided. Be brief and direct.
Copy names, numbers, grades, dates and amounts exactly as they appear in the records, and
include the figures that answer the question: percentages, balances, amounts and dates.
If the records do not answer the question, say so plainly. Never invent information.
You cannot change any record yourself: never say that something was marked, recorded,
updated or changed.
You can look up students, attendance, fees, grades, homework, submissions, class rosters and
timetables, and prepare attendance, fee payment and homework changes for staff approval."""

router = Router()
router.predict({"body": "warm up"}, QUESTIONS)
ids = itertools.count(1)
plans = {}  # first tool call id of a conversation -> {"tool", "args", "retries", "mode"}
laya_lock = threading.Lock()


def llama(body):
    req = urllib.request.Request(UPSTREAM + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def fallback(body):
    """The LLM with its tools, minus every write tool."""
    req = dict(body)
    req["tools"] = [t for t in body.get("tools") or [] if t["function"]["name"] not in WRITE_TOOLS]
    # the same cap as the answer step: a looping small model here costs the user minutes
    req["max_tokens"] = min(body.get("max_tokens") or ANSWER_MAX_TOKENS, ANSWER_MAX_TOKENS)
    return llama(req)


def answer(body):
    msgs = body["messages"]
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    today = m.group(0) if (m := re.search(r"Today is [^\n]*", system)) else ""
    question = next(m["content"] for m in msgs if m["role"] == "user")
    results = [f"{m.get('name', 'tool')}: {m['content']}" for m in msgs if m["role"] == "tool"]
    user = question if not results else f"{question}\n\nRecords:\n" + "\n".join(results)
    req = {k: v for k, v in body.items() if k not in ("tools", "tool_choice", "messages")}
    req["messages"] = [{"role": "system", "content": ANSWER_PROMPT.format(today=today)},
                       {"role": "user", "content": user}]
    # an answer is a few sentences; without a cap a small model can generate until the
    # request times out, which costs the user minutes
    req["max_tokens"] = min(body.get("max_tokens") or ANSWER_MAX_TOKENS, ANSWER_MAX_TOKENS)
    return llama(req)


def text_response(text):
    return {"id": f"chatcmpl-{next(ids)}", "object": "chat.completion", "model": "laya-pipeline",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}


def tool_calls_response(calls):
    return {
        "id": f"chatcmpl-{next(ids)}", "object": "chat.completion", "model": "laya-pipeline",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"call_{next(ids)}", "type": "function",
                            "function": {"name": n, "arguments": json.dumps(args)}} for n, args in calls]}}],
    }


def first_call_id(response):
    calls = (response.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
    return calls[0]["id"] if calls else None


def handle(body):
    msgs = body["messages"]
    prompt = next(m["content"] for m in msgs if m["role"] == "user")
    if not any(m["role"] == "assistant" for m in msgs):
        t0 = time.perf_counter()
        with laya_lock:  # one Laya forward pass at a time, as a single server would run it
            probs = router.predict({"body": prompt}, QUESTIONS)["answers"]["tool"]["probabilities"]
        ms = (time.perf_counter() - t0) * 1000
        tool = choose_tool(probs, prompt)
        a = extract(prompt)
        refusal = policy_refusal(prompt, tool, a)
        call = None if refusal or tool == "none" else first_call(tool, a)
        names_something = any(k in a for k in ("student_id", "assignment_id", "class_name", "name"))
        if refusal:
            mode = "refuse"
        elif call:
            mode = "rules"
        elif tool == "none" and not names_something:
            mode = "answer"
        else:
            mode = "llm"  # rules cannot resolve it: the LLM takes over with read-only tools
        plan = {"tool": tool, "args": a, "retries": 0, "mode": mode}
        sys.stderr.write(json.dumps({"prompt": prompt[:70], "laya_ms": round(ms, 1), "tool": tool,
                                     "top": max(probs, key=probs.get), "mode": mode, "args": a,
                                     "call": call}) + "\n")
        sys.stderr.flush()
        if mode == "refuse":
            return text_response(refusal)
        response = fallback(body) if mode == "llm" else \
            tool_calls_response([call]) if call else answer(body)
        # a conversation is identified by its first tool call id, so concurrent users asking
        # the same question never share a plan
        if cid := first_call_id(response):
            plans[cid] = plan
        return response
    first = next(m for m in msgs if m["role"] == "assistant" and m.get("tool_calls"))
    plan = plans[first["tool_calls"][0]["id"]]
    if plan["mode"] == "llm":
        return fallback(body)
    nxt = [c for c in plan_next(plan, msgs) if c]
    if nxt:
        return tool_calls_response(nxt)
    fixed = fixed_reply(plan, msgs)
    return text_response(fixed) if fixed else answer(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        with urllib.request.urlopen(UPSTREAM + self.path) as r:
            self._send(r.status, r.read())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            self._send(200, json.dumps(handle(body)).encode())
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read())

    def _send(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
