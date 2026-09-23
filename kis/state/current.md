# Current State

Branch: main
Task: PINNED 2026-09-23. CPU setup for the user's project. The T580 measurement track and
      the Laya replacement are complete; the answer model is the one decision still open.
Mode: Phase (all phases done); the remaining items are Fast and independent of each other
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
machines side by side. Also in `~/models`: LFM2.5-350M-Q4_0.gguf.

TO RESUME, REGENERATE THE ROUTER MODELS - they are NOT in git (88 MB of weights, and the
training data is regenerable). Both are seeded, so a retrain reproduces what was measured:

    uv run --python 3.14 python scripts/router_data.py            # 5600 rows + the 54-prompt test set
    uv run --with scikit-learn --python 3.14 python scripts/router_train.py    # TF-IDF -> data/router_tfidf.pkl
    THREADS=4 .laya-venv/bin/python scripts/router_train_minilm.py            # MiniLM -> data/router_minilm/

`.laya-venv` (torch, for MiniLM and Laya) and `.router-venv` (sklearn only, for TF-IDF) are
both gitignored too; `adapters/README.md` and technical.md record how they were built.

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

1. DECIDE THE ANSWER MODEL. The only open decision, measured on current code:
   | answer model | pass | score | p50 | p95 | server RSS |
   |---|---|---|---|---|---|
   | Qwen3.5-0.8B | 81.5% | 0.929 | 0.91 s | 7.54 s | 3114 MB |
   | Qwen3.5-2B | 87.0% | 0.945 | 2.82 s | 16.85 s | 4301 MB |
   The 2B buys exactly three scenarios and loses none: `ms-02`, `ms-05` (multi_step_chain)
   and `ts-01` (tool_selection). So it is a concrete question, not a statistical one: if
   staff ask COMPOUND questions the 0.8B fails them outright; if traffic is single-fact
   lookups the two are equal and the 0.8B is 3x faster at the median. Both safe, both fit
   16 GB. The 2B also held its accuracy better across architectures (-1.9 points against
   -3.7), so it should transfer to the 3400G more predictably.

2. REAL PROMPTS from the repo owner or school staff, 20-30. Worth more than hundreds of
   generated ones, because phrasing distribution is what is under test. These sharpen
   items 1 and 3 and are the standing ask whenever this is picked up.

3. WRITE/READ BIAS in the routers - see Intent, "Track - replacing the Laya router".
   Found by the held-out prompts, invisible to the 54-scenario suite, and it reverses the
   router ranking.

4. THE TARGET ITSELF (Ryzen 5 PRO 3400G, 16 GB): shortlist, load test AND a full-suite run,
   because accuracy does not transfer across architectures. Start from `-t 4` and slots 4,
   both measured on a 4c/8t chip. Budget ~430 MB for the MiniLM router, not Laya's 2.3 GB.

## Optional, not blocking

- `performance` governor is untested here (currently `powersave`). The dominant measurement
  problem on this machine is +/-12% variance between identical runs, not a ceiling.
- `adapters/laya_pipeline.py`'s docstring and `adapters/README.md` still describe the
  pipeline as Laya-routed. Accurate when written, stale now that `minilm` is the default.
