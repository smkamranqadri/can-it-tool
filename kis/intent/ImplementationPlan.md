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
1. Held-out prompts written without reference to the rules, scored on v2.1: the honest
   generalization check, because the rules were tuned on this suite's failures.
2. Load test re-run on v2.1 and on the chosen answer model; set llama-server slots to the
   measured peak (4 on the M2 Pro; expect ~2 on the 4-core 3400G).
3. Port the runner scripts to bash, then repeat the shortlist and load test on the target
   (Omarchy, Ryzen 5 PRO 3400G, 16 GB); there, also try the Vulkan build on the Vega 11
   iGPU and Q4_0 (AVX2 repacked kernels), and `-c 4096` to cut resident memory.
4. Only if more accuracy is needed: fine-tune a 270M model (FunctionGemma or Needle 3) on
   these 13 tools, or swap Laya for Needle 3 as the router on memory-constrained devices.

Proof: the chosen setup holds its wait and pass rate on the held-out prompts and on the
target, with zero safety violations, and the load test shows the concurrency it sustains.
