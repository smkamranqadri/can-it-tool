"""Latency-first table over the 12-scenario subset.

User wait = trace finished_at - started_at, so timed-out runs count their full wait.
Full-suite result files are sliced down to the same 12 scenarios so every row is comparable.
"""

import glob
import os
import json
import statistics
import sys
from datetime import datetime


def wait_s(r):
    t = r["trace"]
    return (datetime.fromisoformat(t["finished_at"]) - datetime.fromisoformat(t["started_at"])).total_seconds()

SUBSET = "sr-01 sr-07 ts-02 el-01 el-02 ms-01 ac-01 nt-02 ie-01 tr-01 wc-01 as-05".split()
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "results") + "/"
# latency-subset runs, plus any full-suite files named on the command line (sliced to the subset)
files = sorted(glob.glob(RESULTS + "latency/*.json")) + [a for a in sys.argv[1:] if a.endswith(".json")]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


rows = []
for f in files:
    d = json.load(open(f))
    if not isinstance(d, dict) or "runs" not in d:  # e.g. laya_router_rows.json
        continue
    runs = [r for r in d["runs"] if any(r["scenario_id"].startswith(s) for s in SUBSET)]
    if not runs:
        continue
    lat = [wait_s(r) for r in runs]
    tps = [r["trace"]["server_tokens_per_second"] for r in runs if r["trace"].get("server_tokens_per_second")]
    label = d["metadata"].get("label") or d["metadata"]["model"]
    if "latency/" not in f:
        label += " (full-suite file)"
    rows.append({
        "model": label, "n": len(runs),
        "p50": statistics.median(lat), "p95": pct(lat, 95), "max": max(lat),
        "pass": sum(r["scenario_pass"] for r in runs) / len(runs),
        "score": statistics.mean(r["final_score"] for r in runs),
        "unsafe": sum(r["safety_gate_triggered"] for r in runs),
        "timeouts": sum(r["termination"] != "final_answer" for r in runs),
        "tps": statistics.mean(tps) if tps else 0,
        "slow": max(runs, key=wait_s)["scenario_id"][:5],
    })

rows.sort(key=lambda r: r["p50"])
print(f"{'model':32} {'n':>3} {'p50 s':>6} {'p95 s':>6} {'max s':>6} {'pass':>5} {'score':>6} {'unsafe':>6} {'tok/s':>6} {'stuck':>5}  slowest")
for r in rows:
    print(f"{r['model']:32} {r['n']:>3} {r['p50']:6.1f} {r['p95']:6.1f} {r['max']:6.1f} {r['pass']:5.0%} {r['score']:6.3f} {r['unsafe']:>6} {r['tps']:6.0f} {r['timeouts']:>5}  {r['slow']}")
if "--per-scenario" in sys.argv:
    for f in files:
        d = json.load(open(f))
        if not isinstance(d, dict) or "runs" not in d:
            continue
        print("\n" + (d["metadata"].get("label") or ""))
        for r in d["runs"]:
            if any(r["scenario_id"].startswith(s) for s in SUBSET):
                print(f"  {r['scenario_id']:36} {wait_s(r):6.1f}s {r['termination']:12}  {r['final_score']:.2f} {'pass' if r['scenario_pass'] else 'FAIL'}")
