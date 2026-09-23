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

NEW: Laya has a working replacement, and it is done. A TF-IDF classifier trained on
synthetic prompts now matches Laya's score exactly (0.920) with zero safety violations,
and beats it on every latency measure - p50 1.29 s vs 2.03, p95 7.53 vs 9.04, max 21.0 vs
23.6 - using 127 MB instead of ~2300 MB and no torch at all. Pass rate is 77.8% vs 79.6%,
a one-scenario difference. Tables, the labelling lessons and the none-suppression fix are
in benchmarks.md, "Replacing the Laya router with a trained classifier". `ROUTER=laya`
remains the default; `ROUTER=tfidf` selects the classifier.

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
   a. DONE - load test re-run. Peak throughput 90.7 -> 277.0 req/min (3.05x) and the
      median wait under 4-user load 5.78 s -> 1.01 s, zero safety violations at every
      level. Slots stay at 4; that ceiling is the four cores, not the router.
   b. Recover `ac-04` (oblique phrasing, "none" 0.45 beat search_student 0.19) and `as-03`
      (weak rather than clean refusal). Both fail safe; neither is urgent.
   c. Try the MiniLM encoder as originally planned, to see if it beats TF-IDF's 77.8%.
   d. Export the model to plain numpy so sklearn (202 MB) drops out too.

2. DECIDE THE ANSWER MODEL. Recommendation: Qwen3.5-2B. It is not just more accurate but
   more portable - across the two architectures it lost 1.9 points where the 0.8B lost 3.7,
   so its margin widened from 5.6 to 7.4 points. Cost is the tail, not the median.
   | model | T580 pass | T580 p50 | T580 p95 | M2 pass | M2 p50 |
   |---|---|---|---|---|---|
   | qwen3.5-0.8b | 79.6% | 2.03 s | 9.04 s | 83.3% | 0.36 s |
   | qwen3.5-2b | 87.0% | 3.32 s | 17.24 s | 88.9% | 0.68 s |
2. DONE - LFM2.5-350M-Q4_0 tested here, full suite: 61.1% pass, p50 1.24 s, p95 3.61 s,
   0 unsafe. Fast but 26 points behind the 2B, and its failures are factual (drops list
   entries; inverted a negation into "Junaid Farooq did not submit" when he did). Not
   usable for this application. The valuable byproduct is in benchmarks.md: on slow CPUs
   the ROUTER is the bottleneck, not the answer model - Laya is ~77% of p50 with the 350M,
   so answer-model shrinking has no headroom left. Making the router cheaper is now plan
   item 4 and the highest-value remaining speed lever for the target.
3. VALIDATE Q4_0 FOR ACCURACY. It is 23%+ faster at prompt processing on x86 AVX2 and
   should carry to the 3400G, but it is a coarser quant and no suite has been run with it.
   One `RUNS=3` full-suite run against `Qwen3.5-0.8B-Q4_0.gguf` settles it.
4. Held-out prompts the rules were not designed against. Still the honest generalization
   check and untouched by any hardware work.
5. The target itself (Ryzen 5 PRO 3400G, 16 GB): shortlist, load test AND a full-suite run,
   since accuracy does not transfer across architectures. Start from `-t 4`, slots 4, and
   budget ~2.3 GB for Laya.

## Optional, not blocking

`performance` governor is untested here (currently `powersave`); the dominant measurement
problem is +/-12% variance between identical runs, not a ceiling. Repo has unstaged changes
to `.gitignore`, both scripts and all three KIS layers.
