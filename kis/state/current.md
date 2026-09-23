# Current State

Branch: main
Task: PINNED 2026-09-23. CPU setup for the user's project. The T580 measurement track and
      the Laya replacement are complete; the answer model is the one decision still open.
Mode: Phase (all phases done); the remaining items are Fast and independent of each other
Command: `uv run --with pytest --with httpx --python 3.14 pytest -q`
Benchmark: `LLAMA=~/.local/opt/llama-b11100/llama-server THREADS=4 SFLAGS="--reasoning off"
  RUNS=3 ROUTER=minilm SHIM=adapters/laya_pipeline.py SHIM_PY=.laya-venv/bin/python
  OUT=results scripts/run_cpu.sh <label> <gguf> <quant> "<notes>"` - add `SCEN=...` for the
  12-scenario subset (listed in benchmarks.md), `scripts/load_cpu.sh <label> <gguf> 1 2 4 8`
  for the load test, `SHIM_PY=.router-venv/bin/python` with `ROUTER=tfidf`.

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

Installed here and reusable: `uv` in `~/.local/bin`; llama.cpp b11100 `ubuntu-x64` in
`~/.local/opt/llama-b11100/` with the Vulkan build beside it; `.laya-venv` (torch+laya) and
`.router-venv` (sklearn only); in `~/models` the four GGUFs measured (Qwen3.5-0.8B Q4_K_M
and Q4_0, Qwen_Qwen3.5-2B-Q4_K_M matching the M2 run, LFM2.5-350M-Q4_0); and the M2's own
result JSONs in `results/m2/` (gitignored) so `compare.py` can compare both machines.

TO RESUME, REGENERATE THE ROUTER MODELS - they are NOT in git (88 MB of weights, and the
training data is regenerable). Both are seeded, so a retrain reproduces what was measured:

    uv run --python 3.14 python scripts/router_data.py            # 5600 rows + the 54-prompt test set
    uv run --with scikit-learn --python 3.14 python scripts/router_train.py    # TF-IDF -> data/router_tfidf.pkl
    THREADS=4 .laya-venv/bin/python scripts/router_train_minilm.py            # MiniLM -> data/router_minilm/

Both venvs are gitignored; `adapters/README.md` and technical.md record how they were built.

## Proof

- `pytest -q` - 329 passed, on both macOS/3.14.6 and Linux/3.14.7.
- Full suite 54x3 on both machines, four configs, provenance-matched GGUFs: see
  benchmarks.md. Zero safety violations in every pipeline run ever measured, on either
  machine, at every concurrency level.
- `compare.py` recommended pipeline v2.1 + Qwen3.5-2B unprompted across the four
  LAYA-ERA runs (M2 and T580, both answer models). It has NOT been re-run on the current
  MiniLM configurations.
- Load test 1/2/4/8 users, and a Vulkan run with offload verified in the server log.
- CAVEAT, still the biggest one and now broader than it was: the v2/v2.1 rules were
  written from this suite's failures, AND the router's labelling scheme was corrected
  twice by diagnosing against the same 54 prompts. Both the rules and the router have
  therefore seen the test set, so every pass rate here is partly in-sample. This is why
  real prompts are Next item 2.

## Blocker

None. Nothing is half-written; no benchmark processes are running.

## Next

1. DECIDE THE ANSWER MODEL. The only open decision. 0.8B is 81.5% at p50 0.91 s; 2B is
   87.0% at p50 2.82 s (full table in benchmarks.md, "The answer-model choice").
   It is a concrete question, not a statistical one: the 2B buys exactly three scenarios
   and loses none - `ms-02`, `ms-05` (multi_step_chain) and `ts-01` (tool_selection). If
   staff ask COMPOUND questions the 0.8B fails them outright; if traffic is single-fact
   lookups the two are equal and the 0.8B is 3x faster. Both safe, both fit 16 GB. The 2B
   also held its accuracy better across architectures, so it should transfer to the 3400G
   more predictably.

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
