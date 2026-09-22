# adapters/

OpenAI-compatible stand-ins that sit between `benchmark.py` and a runtime, for models or
pipelines the harness cannot talk to directly. The harness is unchanged: it still sends
`/v1/chat/completions` with `tools`, executes every returned tool call against the
simulator, and scores the run. Each adapter takes `[upstream] [port]` (default
`http://127.0.0.1:8080`, `8090`); point `--base-url` at the port.

| adapter | what it does |
|---|---|
| `needle_shim.py` | Needle 3's `needle --serve` (`POST /complete`) behind chat completions. Needle writes no prose, so the final content is its reasoning plus the tool results it was handed. Sends compact JSON: Needle's parser rejects `": "`. |
| `xlam_shim.py` | xLAM-2 on llama-server. llama.cpp cannot parse xLAM's bare JSON-array tool calls, and xLAM's template drops its format instruction when a system prompt is present, so the adapter renders the tools into the system message and parses the array back into `tool_calls`. |
| `laya_proxy.py` | Laya narrows the tools: a `choice` over the 13 tools picks the likely ones and the LLM gets only those schemas. The LLM still makes every call and writes the answer. |
| `laya_pipeline.py` | Laya picks the tool, `pipeline_rules.py` fills the arguments and chains lookups, the pipeline emits the calls itself, and the LLM makes one call to answer from the results with no tool schemas in its prompt. Falls back to the LLM with tools when the rules cannot resolve a request. |

The Laya adapters need `laya` (torch, transformers) in their own environment, not the
project's:

```bash
uv venv --python 3.12 .laya-venv && VIRTUAL_ENV=.laya-venv uv pip install laya
```

## scripts/

- `run_cpu.sh <label> <gguf> <quant> [notes]` - start a CPU-only llama-server, optionally
  put an adapter in front (`SHIM=adapters/laya_pipeline.py SHIM_PY=.laya-venv/bin/python`),
  run the suite or a subset (`SCEN=...`), and sample the server's CPU and RSS. See the
  header for every variable.
- `latency_report.py [full-suite.json ...]` - latency-first table over the 12-scenario
  subset: median and p95 user wait, pass rate, safety, stuck runs. User wait is the trace's
  `finished_at - started_at`, so a timed-out request counts its full wait.
- `laya_router.py` - Laya alone: tool-set recall, refusal accuracy and CPU latency over all
  54 prompts.
- `pipeline_dryrun.py` - the rules alone with a perfect router, against the simulator: an
  upper bound for `laya_pipeline.py` without the fallback.
- `load_cpu.sh <label> <gguf> [levels...]` + `load_test.py` - concurrent-user load test:
  llama-server with one slot per user, each simulated user running real scored
  conversations back to back. Reports conversations/min, LLM requests/min, wait
  p50/p95/max, pass rate and safety per concurrency level into `results/load/`.
