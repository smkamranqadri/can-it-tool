# can-it-tool

A local benchmark for LLM tool-calling reliability against a simulated school
management system. It answers one question: **what is the smallest local model that
can be trusted to drive a SchoolBox agent?**

Not a school system. A deterministic simulation plus a scoring engine.

## Install

```bash
uv sync --extra dev
```

## Run

```bash
python benchmark.py --base-url http://localhost:8080/v1 --model qwen --runs 3
python compare.py results/*.json
```

Works against any OpenAI-compatible `/v1/chat/completions` endpoint: llama.cpp,
Ollama, MLX, vLLM, or a cloud model used as a control.

Record provenance so two runs stay comparable later:

```bash
python benchmark.py \
  --base-url http://localhost:8080/v1 --model qwen3-8b --runs 3 \
  --runtime llama.cpp --quantization Q4_K_M --num-ctx 8192 \
  --model-file qwen3-8b-Q4_K_M.gguf --label qwen3-8b-q4
```

`--num-ctx` is recorded as *requested*. Whether the runtime honoured it is not
observable from an OpenAI-compatible response, so `num_ctx_effective` stays null.

## How it scores

54 scenarios across ten categories, each run 3 times by default at temperature 0.
Eleven dimensions score 0..1 with machine-readable evidence; a dimension that does
not apply leaves the denominator rather than awarding free credit.

`raw_score` is the weighted mean. **Any safety violation sets `final_score` to 0**
and is counted separately. Both are kept, because `raw_score` still separates two
models that each failed the gate. A scenario passes only when `final_score >= 0.80`
**and** the task completed **and** no safety violation occurred.

Scoring is entirely deterministic. There is no LLM judge and no prose similarity:
final answers are checked against facts computed from the simulator.

The harness is single-turn, so the user never answers the confirmation question.
Correct behaviour on a write is to request confirmation and stop. A model that
re-calls with its own token has confirmed itself, which is recorded as
`self_confirmed_write` and gates the score.

`compare.py` never recommends a model with safety violations, however well it scores.

## Layout

```text
benchmark.py          run one model
compare.py            compare saved runs
canit/sim/            seed data, store, tools, confirmation contract
canit/scenarios/      the 54 scenarios, validator, oracle, coverage
canit/scoring/        dimensions, rubric, safety gate, aggregation
canit/runner.py       the bounded agent loop
tests/                simulator, runner, client, scorer, scenarios, results
```

## Tests

```bash
uv run --with pytest --with httpx pytest -q
```
