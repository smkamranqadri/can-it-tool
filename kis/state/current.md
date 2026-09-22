# Current State

Branch: main
Task: Pick the CPU setup for the user's project (latency, then accuracy, then throughput)
Mode: Research
Status: candidate settled - Laya pipeline v2.1; answer model is a two-way choice, see Next
Command: `uv run --with pytest --with httpx --python 3.14 pytest -q`

## Proof

- `pytest -q` - 329 passed (61 sim, 42 runner, 14 client, 50 scoring, 102 scenarios, 60 results)
- Harness phases 1-5.6 are done and were proved against real models; the measurements that
  matter now live in `kis/knowledge/benchmarks.md`, the design lessons in
  `kis/knowledge/technical.md`.
- `uv run python scripts/pipeline_dryrun.py` - rules alone, perfect router, against the
  simulator: every required call in 42/45 tool scenarios, 4/5 must-refuse scenarios refused
  without touching a write tool, 0 write calls carrying a confirmation token, 15/54
  scenarios answered with no LLM call.
- Full suite (54 x 3, llama.cpp b11100, CPU only, reasoning off): pipeline v2.1 scores 83%
  pass / 0.940 with Qwen3.5-0.8B at 0.4 s median, and 89% / 0.954 with Qwen3.5-2B at 0.7 s.
  Zero safety violations in every pipeline run; every plain model violated safety
  (qwen3.5-0.8b 3.7%, qwen3.5-2b 9.3%).
- CAVEAT: the v2/v2.1 rules were written from this suite's failures, so those numbers are
  partly in-sample. A held-out prompt set is the first item under Next.

## Blocker

None. Phase 6 (static results page) is parked while the deployment setup is settled.

## Next

1. Decide the answer model with the user: Qwen3.5-0.8B (0.4 s, 83%, ~2 GB) or Qwen3.5-2B
   (0.7 s, 89%, ~3.5 GB). On the target expect roughly 3x those waits.
2. Held-out prompts the rules were not designed against; score v2.1 on them.
3. Re-run the load test on v2.1 (fixed replies skip the LLM, so throughput should rise).
4. Port `scripts/run_cpu.sh` and `scripts/load_cpu.sh` to bash, then repeat the shortlist
   and load test on the target (Omarchy, Ryzen 5 PRO 3400G, 16 GB); try the Vulkan build on
   the Vega 11 iGPU and Q4_0 there.
