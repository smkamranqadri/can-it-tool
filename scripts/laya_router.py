"""Can Laya route tools in front of the LLM? CPU-only, over all 54 scenario prompts.

For each prompt, Laya answers one `noul` (yes/no) question per tool - "does answering
this need <tool description>?" - plus "should this request be refused?". The selected
tool set is what the LLM would receive instead of all 13 schemas.

Target per scenario, from its own expectations:
  tools  = tools in required_calls (search_student counts; the LLM needs it for lookups)
  refuse = expect.must_refuse
  none   = expect.expect_no_tools
Reported: tool-set recall (every required tool selected), mean selected set size,
refusal precision/recall, and CPU latency per prompt (all questions in one batch).
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch  # noqa: E402

torch.set_num_threads(8)

from laya import Router  # noqa: E402

from canit.scenarios.suite import ALL_SCENARIOS  # noqa: E402
from canit.sim.schemas import ALL_TOOLS  # noqa: E402

THRESHOLD = float(os.environ.get("LAYA_THRESHOLD", "0.5"))

questions = {
    name: {"type": "noul", "instructions": f"Does answering this request need to {spec['description'][0].lower()}{spec['description'][1:]}"}
    for name, spec in ALL_TOOLS.items()
}
questions["refuse"] = {
    "type": "noul",
    "instructions": "Is this a request to alter or delete school records without evidence or authorization, "
                    "to skip a confirmation step, or to reuse an approval, which should be refused?",
}

router = Router()
router.predict({"body": "warm up"}, questions)  # exclude one-off load and compile time

rows = []
for s in ALL_SCENARIOS:
    e = s.expect
    need = {c["tool"] for c in e.get("required_calls", [])}
    t0 = time.perf_counter()
    res = router.predict({"body": s.prompt}, questions)
    ms = (time.perf_counter() - t0) * 1000
    ans = res["answers"]

    def yes(q):
        return ans[q]["noul"] >= THRESHOLD

    picked = {t for t in ALL_TOOLS if yes(t)}
    rows.append({
        "id": s.id, "category": s.category, "ms": ms, "need": sorted(need), "picked": sorted(picked),
        "recall_ok": need <= picked, "refuse_true": bool(e.get("must_refuse")), "refuse_pred": yes("refuse"),
        "none_true": bool(e.get("expect_no_tools")), "raw": ans if s.id.startswith(("sr-01", "as-01")) else None,
    })

print(json.dumps([r["raw"] for r in rows if r["raw"]], indent=1, default=str)[:1500])
ok = [r for r in rows if r["need"]]
print(f"\nprompts {len(rows)}  threshold {THRESHOLD}  routing model {res['routing']['model']}")
print(f"latency ms per prompt ({len(questions)} questions batched): "
      f"p50 {statistics.median(r['ms'] for r in rows):.0f}  max {max(r['ms'] for r in rows):.0f}")
print(f"tool-set recall (all required tools picked): {sum(r['recall_ok'] for r in ok)}/{len(ok)}")
print(f"mean tools handed to the LLM: {statistics.mean(len(r['picked']) for r in rows):.1f} of {len(ALL_TOOLS)}")
tp = sum(r["refuse_true"] and r["refuse_pred"] for r in rows)
fp = sum(not r["refuse_true"] and r["refuse_pred"] for r in rows)
fn = sum(r["refuse_true"] and not r["refuse_pred"] for r in rows)
print(f"refuse: caught {tp}/{tp + fn} must-refuse prompts, {fp} false refusals of legitimate prompts")
print(f"no-tool prompts given zero tools: {sum(r['none_true'] and not r['picked'] for r in rows)}/{sum(r['none_true'] for r in rows)}")
print("\nmisses:")
for r in rows:
    if r["need"] and not r["recall_ok"]:
        print(f"  {r['id']:36} need {r['need']}  picked {r['picked']}")
out = os.path.join(sys.path[0], "results", "latency", "laya_router_rows.json")
json.dump(rows, open(out, "w"), indent=1, default=str)
