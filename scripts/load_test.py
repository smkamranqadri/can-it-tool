"""How many conversations per minute an endpoint serves under concurrent users.

Each simulated user runs real, scored scenario conversations back to back against the
endpoint (the same run_scenario/score_run the benchmark uses, each with a fresh
simulator store), so the load is the product's real request mix: tool calls, tool
results, final answers. For each concurrency level it reports throughput, the user
wait distribution, and whether answers stay correct under load.

usage: uv run python scripts/load_test.py --base-url http://127.0.0.1:8080/v1 \
           --model qwen --levels 1 2 4 8 --seconds 90 [--scenario sr-01 ...] [--out file.json]

The server must have at least max(levels) parallel slots (llama-server -np N).
"""

import argparse
import itertools
import json
import os
import random
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from canit.client import build_client  # noqa: E402
from canit.config import RunConfig  # noqa: E402
from canit.runner import run_scenario  # noqa: E402
from canit.scenarios.suite import ALL_SCENARIOS  # noqa: E402
from canit.scoring import score_run  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))] if xs else float("nan")


def run_level(config, scenarios, users, seconds, seed):
    """`users` threads, each running conversations until `seconds` have passed."""
    deadline = time.perf_counter() + seconds
    records, lock = [], threading.Lock()

    def user(uid):
        client = build_client(config)
        order = scenarios[:]
        random.Random(seed + uid).shuffle(order)
        try:
            for scenario in itertools.cycle(order):
                if time.perf_counter() >= deadline:
                    break
                t0 = time.perf_counter()
                trace, _ = run_scenario(config, client, scenario)
                score = score_run(trace, scenario)
                with lock:
                    records.append({
                        "user": uid, "scenario": scenario.id, "wait_s": time.perf_counter() - t0,
                        "finished_at": time.perf_counter(), "pass": score.scenario_pass,
                        "unsafe": score.safety_gate_triggered, "termination": trace.termination,
                        "requests": len(trace.steps),
                    })
        finally:
            client.close()

    started = time.perf_counter()
    threads = [threading.Thread(target=user, args=(u,)) for u in range(users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - started  # includes finishing conversations begun before the deadline
    waits = [r["wait_s"] for r in records]
    return {
        "users": users, "elapsed_s": round(elapsed, 1), "conversations": len(records),
        "conversations_per_min": round(len(records) / elapsed * 60, 1),
        "llm_requests_per_min": round(sum(r["requests"] for r in records) / elapsed * 60, 1),
        "wait_p50_s": round(statistics.median(waits), 2) if waits else None,
        "wait_p95_s": round(pct(waits, 95), 2), "wait_max_s": round(max(waits), 2) if waits else None,
        "pass_rate": round(sum(r["pass"] for r in records) / len(records), 3) if records else None,
        "unsafe": sum(r["unsafe"] for r in records),
        "not_final_answer": sum(r["termination"] != "final_answer" for r in records),
        "records": records,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seconds", type=float, default=90, help="load duration per level")
    ap.add_argument("--scenario", action="append", help="scenario id prefix; default: whole suite")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out")
    args = ap.parse_args()

    scenarios = [s for s in ALL_SCENARIOS
                 if not args.scenario or any(s.id.startswith(p) for p in args.scenario)]
    config = RunConfig(base_url=args.base_url, model=args.model, timeout=args.timeout, runs=1, max_steps=8)

    levels = []
    print(f"{len(scenarios)} scenarios, {args.seconds:.0f} s per level")
    print(f"{'users':>5} {'conv':>5} {'conv/min':>9} {'req/min':>8} {'p50 s':>6} {'p95 s':>6} {'max s':>6} "
          f"{'pass':>5} {'unsafe':>6} {'stuck':>5}")
    for users in args.levels:
        r = run_level(config, scenarios, users, args.seconds, args.seed)
        levels.append(r)
        print(f"{r['users']:>5} {r['conversations']:>5} {r['conversations_per_min']:>9} {r['llm_requests_per_min']:>8} "
              f"{r['wait_p50_s']:>6} {r['wait_p95_s']:>6} {r['wait_max_s']:>6} {r['pass_rate']:>5.0%} "
              f"{r['unsafe']:>6} {r['not_final_answer']:>5}", flush=True)
    if args.out:
        json.dump({"base_url": args.base_url, "model": args.model, "seconds_per_level": args.seconds,
                   "scenarios": [s.id for s in scenarios], "levels": levels}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
