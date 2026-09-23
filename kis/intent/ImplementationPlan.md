# Implementation Plan

Phase Mode. Each phase ends with real test output as proof.

## Phase 1 - Simulator

Seed data, in-memory store with per-run snapshot/restore, the ten+ tools, their JSON
schemas, the READ/WRITE split, and the two-phase confirmation contract with simulator-
issued tokens. Deliberate duplicate names in the seed.

Proof: `pytest tests/test_sim.py` - determinism, tool outputs, confirmation flow,
token forgery rejection.

## Phase 2 - Client and runner

`RunConfig`, the OpenAI-compatible client behind a protocol seam, and the bounded
agent loop that records a complete trace.

Proof: runner drives a scripted mock client end to end and produces a full trace.

## Phase 3 - Scoring engine

The eleven dimensions, the weighted rubric, the safety gate, and aggregation into
category and overall metrics.

Proof: `pytest tests/test_scoring.py` - each dimension in isolation, plus fixture
traces for a perfect model, a sloppy model, and an unsafe model.

## Phase 4 - Scenario suite

~52 cases across the ten categories, each with machine-readable expectations, plus a
schema test that every scenario is well-formed and references only real entities.

Proof: `pytest tests/test_scenarios.py`.

## Phase 5 - CLIs and artifacts

`benchmark.py`, `compare.py`, versioned results JSON, console summary.

Proof: a real run against a local llama-server or ollama endpoint, and a comparison
across two result files.

## Phase 6 - Results page (secondary)

SUPERSEDED for now - see the CPU setup track below; Phase 6 resumes once the deployment
setup is chosen. Static HTML generated from results JSON.

Proof: generated page opens and matches the JSON numbers.

## Track - CPU setup for deployment (user direction 2026-09-22)

Pick the setup for the user's own project, which runs CPU-only. Goals in order:
1. reply latency: median and p95 user wait (question in, final reply out)
2. accuracy: 12-scenario subset to shortlist, then the full suite, runs=3
3. throughput: conversations served under concurrent users, and the wait at each level
Safety violations must stay at zero throughout.

Settled: the architecture is the Laya pipeline (`adapters/laya_pipeline.py` +
`pipeline_rules.py`) on llama.cpp b11100 - Laya `choice` picks the tool, rules fill the
arguments and issue the calls, fixed replies answer the not-found / several-matches /
pending-confirmation cases with no model at all, and the LLM only phrases the rest with no
tool schemas in its prompt. It beat every single model measured on latency, accuracy and
safety at once. The model search is closed (see knowledge/benchmarks.md).

Open decision - the answer model:
- Qwen3.5-0.8B Q4_K_M: 0.4 s median, 83% pass, ~2 GB
- Qwen3.5-2B Q4_K_M: 0.7 s median, 89% pass, ~3.5 GB, and the only option that lifts
  multi-step requests (33% -> 50%)
Both are safe (0 violations) and both fit 16 GB. Expect roughly 3x the wait on the target.

Remaining work, in order:
1. DONE 2026-09-23. Measured on the T580; results in `kis/knowledge/benchmarks.md`.
   It turned out to answer more than the hardware-dependent half: accuracy moved too,
   which is why item 3 now needs its own full-suite run.
2. Held-out prompts written without reference to the rules, scored on v2.1: the honest
   generalization check, because the rules were tuned on this suite's failures. No
   hardware measurement substitutes for it; it stays open regardless of the T580 results.
3. On the target itself (Omarchy, Ryzen 5 PRO 3400G, 16 GB): the shortlist, the load
   test, AND A FULL-SUITE RUN, because the T580 proved accuracy does not transfer across
   architectures. Use `-t 4` and slots 4, both measured on a 4c/8t chip. Also try the
   Vulkan build on Vega 11 - worth testing for PROMPT PROCESSING specifically, since Vega
   11 has integer dot products and RADV where the UHD 620 has neither, but not for
   generation, which stays bandwidth-bound on shared memory. See technical.md, "Why an
   iGPU does not accelerate this workload". Also Q4_0 and `-c 4096` to cut resident
   memory. Budget ~2.3 GB for Laya.
4. PROMOTED from speculative: make the ROUTER cheaper. Measured on the T580, Laya's
   forward pass is ~0.95 s and is 77% of the p50 wait with a small answer model, and it is
   the largest memory consumer at ~2.2-2.4 GB. On slow CPUs this is now the highest-value
   remaining speed lever, ahead of any further answer-model shrinking, which has almost no
   headroom left. Options: swap Laya for Needle 3 as the router, or fine-tune a 270M model
   (FunctionGemma or Needle 3) on these 13 tools for accuracy.

Proof: the chosen setup holds its wait and pass rate on the held-out prompts and on the
target, with zero safety violations, and the load test shows the concurrency it sustains.

## Track - T580 measurements, gated (user direction 2026-09-23)

A second data point, not a new target. The bench machine is an M2 Pro at ~200 GB/s; the
target is a 3400G at ~47 GB/s. This ThinkPad T580 (i7-8650U, 4c/8t, 32 GB DDR4 ~38 GB/s,
UHD 620, Omarchy) sits much nearer the target on memory bandwidth, which is what CPU
generation is bound by.

Two things must be recorded with every number from it:
- It cannot hold 4.2 GHz across four cores (15 W nominal TDP, RAPL PL1 set to 25 W); the
  3400G is 65 W and can. T580 waits are a PESSIMISTIC BOUND on the target, not an estimate
  of it, so the sustained clock goes in the table beside the latency. Raising PL1 to 35 W
  was tried and changed nothing - PL1 averages over 64 s and the workload is bursty.
- It has 32 GB against the target's 16 GB, so memory pressure cannot show up here. Peak RSS
  is recorded so the target's ceiling can be checked against it.

Accuracy is not the point. At temperature 0 with the same build, quant and model the tokens
do not change, so the full suite here buys the latency DISTRIBUTION over 54 scenarios
(p95, max), not a second accuracy measurement.

### Phase 0 - make the machine able to run the harness  [DONE 2026-09-23]

`uv` (extra/uv 0.12.10), the b11100 ubuntu-x64 tarball into `~/.local/opt/llama-b11100/`,
`.laya-venv` on Python 3.12, Qwen3.5-0.8B Q4_K_M into `~/models` (the 2B waits for the
gate), and the bash port of `run_cpu.sh`: shebang, `${0:A:h:h}`, `${MON:h}`, `${=SFLAGS}`,
and the macOS-only `sysctl` MACHINE line. Keep it bash-3.2-safe so the M2 can still run it.

Proof: `pytest -q` 329 passed on Linux / Python 3.14.7; `llama-server --version` prints
b11100; `.laya-venv/bin/python -c "import laya, torch"`; `bash -n` clean; the script prints
a Linux MACHINE string.

### Phase 1 - the gate  [DONE 2026-09-23 - PASSED]

The 12-scenario shortlist through pipeline v2.1 + Qwen3.5-0.8B, runs=1, run twice:
`-t 4` with `LAYA_THREADS=4` against `-t 8` with `LAYA_THREADS=8`. Both defaults are
M2-shaped (eight performance cores); this machine and the target have four physical cores,
and llama.cpp's threads already compete with Laya's torch threads. Sample CPU frequency and
throttling beside the existing CPU%/RSS log.

Gate - Phase 2 happens only if:
- the run completes, writes a result JSON, and has no stuck or timed-out runs
- subset pass rate >= 75% (pipeline v1 scored 83% on this subset on the M2). A large drop
  means the port or the GGUF is wrong, not the machine.
- zero safety violations. Structural in this pipeline, so a violation is a defect.
A disappointing p50 is NOT a gate failure. It is the finding, and Phase 2 continues with a
changed conclusion.

### Phase 2 - only after the gate, in order  [DONE 2026-09-23 - all four parts]

- 2a: shortlist with Qwen3.5-2B Q4_K_M, at the better thread setting (-t 4, measured).
- 2b: full suite 54 x 3, both answer models. Roughly 30-60 min per config, not hours.
- 2c: load test at 1/2/4/8 users on the chosen model; needs `load_cpu.sh` ported too. This
  gives the concurrency peak on four cores, which is what sets llama-server slots on the
  3400G and is the one number no current measurement provides.
- 2d: Vulkan - the `ubuntu-vulkan-x64` tarball, shortlist on the UHD 620. A rehearsal of
  the mechanics and a bound, NOT evidence about Vega 11: different driver (ANV vs RADV),
  weaker GPU, shared RAM. A clean allocation failure is an acceptable recorded outcome.

Out of scope here: the held-out prompts, Phase 6, any change to `canit/` or the scenarios,
re-running plain models, and any change of deployment target. A needed edit to `adapters/`
or `canit/` is a finding, not routine work.

Assumption on record: the GGUF provenance is unmatched (the M2 result JSON is gitignored
and lives on that machine). Quant is Q4_K_M either way and provenance does not move latency.
It only matters if accuracy diverges from the benchmarks.md rows by more than a couple of
points; the M2 metadata settles it then.
