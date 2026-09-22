"""The Laya pipeline's rules with no model involved, against the simulator.

Laya's pick is replaced by each scenario's own target tool (a perfect router). Then, per
scenario, the rules refuse or build calls, chain them against a fresh simulator store, and
fixed_reply answers where it can. Reports:

  reach    tool scenarios where the rules make every required call
  safety   must-refuse scenarios refused without touching a write tool, and whether any
           write call ever carried a confirmation token (it must not)
  fixed    scenarios answered by fixed_reply, i.e. with no LLM call at all

usage: uv run python scripts/pipeline_dryrun.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "adapters")]

from pipeline_rules import WRITE_TOOLS, extract, first_call, fixed_reply, plan_next, policy_refusal  # noqa: E402

from canit.scenarios.suite import ALL_SCENARIOS  # noqa: E402
from canit.sim.db import Store, build_dataset  # noqa: E402
from canit.runner import build_fault_plan  # noqa: E402
from canit.sim.tools import execute  # noqa: E402


def target_tool(expect):
    """The tool a perfect router would name."""
    required = [c["tool"] for c in expect.get("required_calls", [])]
    writes = [w["tool"] for w in expect.get("expected_writes", [])]
    forbidden = [t for t in expect.get("forbidden_tools", []) if t in WRITE_TOOLS]
    if writes:
        return writes[0]
    if expect.get("must_refuse") and forbidden:
        return forbidden[0]
    work = [t for t in required if t not in ("search_student", "list_assignments")]
    return (work or required or ["none"])[-1]


reach = total = refused_ok = must_refuse = fixed = 0
token_leaks = []
for s in ALL_SCENARIOS:
    e = s.expect
    required = [c["tool"] for c in e.get("required_calls", [])]
    tool = target_tool(e)
    a = extract(s.prompt)
    refusal = policy_refusal(s.prompt, tool, a)
    plan = {"tool": tool, "args": a, "retries": 0, "mode": "rules"}
    store, messages, sequence = Store(build_dataset()), [{"role": "user", "content": s.prompt}], []
    store.faults = build_fault_plan(s.faults)  # the same injected faults the harness uses
    call = None if refusal or tool == "none" else first_call(tool, a)
    calls = [call] if call else []
    while calls and len(sequence) < 16:
        tool_calls, results = [], []
        for name, args in calls:
            sequence.append(name)
            if name in WRITE_TOOLS and "confirmation_token" in args:
                token_leaks.append(s.id)
            tool_calls.append({"function": {"name": name, "arguments": json.dumps(args)}})
            results.append({"role": "tool", "name": name, "content": json.dumps(execute(store, name, args))})
        messages += [{"role": "assistant", "tool_calls": tool_calls}, *results]
        calls = [c for c in plan_next(plan, messages) if c]
    reply = refusal or (fixed_reply(plan, messages) if sequence else None)
    fixed += reply is not None

    if e.get("must_refuse"):
        must_refuse += 1
        ok = refusal is not None and not (set(sequence) & WRITE_TOOLS)
        refused_ok += ok
        if not ok:
            print(f"NOT REFUSED  {s.id:36} target={tool:24} calls={sequence}")
        continue
    if required:
        total += 1
        ok = all(required.count(t) <= sequence.count(t) for t in set(required))
        reach += ok
        if not ok:
            print(f"fallback     {s.id:36} target={tool:24} rules={sequence} need={required}"
                  + (f" refused: {refusal[:40]}" if refusal else ""))

print(f"\nreach   rules make every required call in {reach}/{total} tool scenarios (perfect router)")
print(f"safety  {refused_ok}/{must_refuse} must-refuse scenarios refused with no write tool touched; "
      f"write calls carrying a confirmation token: {len(token_leaks)}")
print(f"fixed   {fixed}/{len(ALL_SCENARIOS)} scenarios answered without an LLM call")
