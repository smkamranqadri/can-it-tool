# Current State

Branch: main
Task: CPU setup for the user's project - T580 measurement track complete; answer model undecided
Mode: Phase (T580 track done); next steps are Fast
Command: `uv run --with pytest --with httpx --python 3.14 pytest -q`

Plan: `kis/intent/ImplementationPlan.md`, Track - T580 measurements (all phases DONE).
All measurement tables live in `kis/knowledge/benchmarks.md`; the reasoning about why the
hardware behaves as it does is in `kis/knowledge/technical.md`.

## Status

The T580 (ThinkPad, i7-8650U, Omarchy) is set up and fully measured as a second CPU data
point. The deployment target is still the Ryzen 5 PRO 3400G.

LAYA IS REPLACED. The best configuration measured on this machine is the MiniLM router
with none-suppression and the grade refusal rule: 81.5% pass, score 0.929, zero safety
violations, p50 0.91 s, p95 7.54 s, 430 MB adapter RSS. It beats the Laya baseline on
EVERY axis (79.6%, 0.920, 2.03 s, 9.04 s, ~2300 MB), and exactly one scenario still
differs - `sr-05-homeroom-teacher`, which MiniLM wins 3/3 to 0/3.

MiniLM IS THE CHOSEN ROUTER (user decision) and is now the default in
`adapters/laya_pipeline.py`. Three routers remain selectable: `ROUTER=laya`, `ROUTER=tfidf`
(127 MB, no torch at all, 77.8%) and `ROUTER=minilm` (430 MB, 81.5%). Tables, the
labelling lessons, the none-suppression fix and the grade rule are in benchmarks.md,
"Replacing the Laya router with a trained classifier".

Installed here and reusable: `uv` in `~/.local/bin`, llama.cpp b11100 `ubuntu-x64` in
`~/.local/opt/llama-b11100/` with the Vulkan build beside it, `.laya-venv` (torch 2.14.0+cpu,
laya 0.3.6), and in `~/models`: Qwen3.5-0.8B-Q4_K_M, Qwen3.5-0.8B-Q4_0,
Qwen3.5-2B-Q4_K_M (unsloth), Qwen_Qwen3.5-2B-Q4_K_M (bartowski, matches the M2 run).
The M2's own result JSONs are in `results/m2/` (gitignored) so `compare.py` can put both
machines side by side.

## Proof

- `pytest -q` - 329 passed, on both macOS/3.14.6 and Linux/3.14.7.
- Full suite 54x3 on both machines, four configs, provenance-matched GGUFs: see
  benchmarks.md. Zero safety violations in every pipeline run ever measured, on either
  machine, at every concurrency level.
- `compare.py` across all four runs recommends pipeline v2.1 + Qwen3.5-2B unprompted.
- Load test 1/2/4/8 users, and a Vulkan run with offload verified in the server log.
- CAVEAT unchanged and still the biggest one: the v2/v2.1 rules were written from this
  suite's failures, so every pass rate here is partly in-sample.

## Blocker

None. Nothing is half-written; no benchmark processes are running.

## Next

1. CLASSIFIER ROUTER - remaining polish, none of it blocking:
   a. DONE for TF-IDF - load test re-run, peak throughput 90.7 -> 277.0 req/min (3.05x) and the
      median wait under 4-user load 5.78 s -> 1.01 s, zero safety violations at every
      level. Slots stay at 4; that ceiling is the four cores, not the router.
   b. DONE for `as-03` - fixed in `policy_refusal`, which is the right layer: a grade
      change is unsupported however it was routed. Narrow by design so a homework
      submission score (`wc-04`) is still allowed; verified to fire on `as-03` alone across
      all 54 prompts, and it took the rules dry-run from 4/5 to 5/5 must-refuse and from
      15/54 to 17/54 no-LLM answers. NOT gated by router, and re-running Laya confirmed
      zero scenarios changed for it. `ac-04` is recovered by MiniLM; TF-IDF still misses it.
   c. DONE - MiniLM beats TF-IDF end-to-end: 79.6% vs 77.8% (it gains `ac-04` and loses
      nothing), score 0.922, p50 0.97 s vs 1.29 s, at 429 MB vs 127 MB. It matches Laya's
      pass rate exactly. Standalone the two classifiers TIE at 74.1% and are both wrong on
      13 of 14 - standalone top-1 is a poor proxy for this pipeline. NOT yet load-tested.
      Pick TF-IDF if the torch dependency matters, MiniLM otherwise - with the grade rule
      MiniLM reaches 81.5%, beating Laya outright.
   d. Export the TF-IDF model to plain numpy so sklearn (202 MB) drops out too.
   e. DONE - MiniLM load-tested: peak 276.3 req/min against TF-IDF's 277.0, which is noise,
      and a LOWER p50 at every level. Routing cost does not register because the bottleneck
      at 4+ users is llama.cpp on four cores. No throughput argument remains for TF-IDF.

2. DECIDE THE ANSWER MODEL - now measured on current code (MiniLM router, all fixes):
   | answer model | pass | score | p50 | p95 | max | server RSS |
   |---|---|---|---|---|---|---|
   | Qwen3.5-0.8B | 81.5% | 0.929 | 0.91 s | 7.54 s | 22.05 s | 3114 MB |
   | Qwen3.5-2B | 87.0% | 0.945 | 2.82 s | 16.85 s | 41.63 s | 4301 MB |
   The 2B buys exactly three scenarios and loses none: `ms-02`, `ms-05` (both
   multi_step_chain) and `ts-01` (tool_selection). So the question is concrete - do staff
   ask COMPOUND questions? If yes, the 2B is needed and the 0.8B fails them outright. If
   the traffic is single-fact lookups, the 0.8B is equal and 3x faster at the median.
   The better router narrowed the gap from 7.4 points to 5.5 but could not close it.
   Both are safe (zero violations) and both fit 16 GB.

3. DONE - Q4_0 CHECKED AND REJECTED. Full suite with MiniLM, only the quant changed:
   75.9% against Q4_K_M's 81.5%, score 0.907 against 0.929, and p50 identical at 0.91 s.
   It loses 5.6 points for a tail-only gain. KEEP Q4_K_M, and do not carry Q4_0 to the
   3400G. The `llama-bench` +23% prompt-processing win did not transfer because the median
   request is a fixed reply that never calls the LLM.

4. Held-out prompts the rules were not designed against. Still the honest generalization
   check and untouched by any hardware work.
5. The target itself (Ryzen 5 PRO 3400G, 16 GB): shortlist, load test AND a full-suite run,
   since accuracy does not transfer across architectures. Start from `-t 4`, slots 4, and
   budget ~2.3 GB for Laya.

## Optional, not blocking

`performance` governor is untested here (currently `powersave`); the dominant measurement
problem is +/-12% variance between identical runs, not a ceiling. Repo has unstaged changes
to `.gitignore`, both scripts and all three KIS layers.
