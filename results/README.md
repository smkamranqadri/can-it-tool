# results/

Benchmark output lands here as `<label>-<timestamp>.json`.

Result files are **not committed**. At `runs=1` each is roughly 1 MB, because every
run retains its complete chronological trace: every request, tool call, raw argument
string, tool result, latency and token count. They are reproducible from the CLI, and
published runs belong on a GitHub Release rather than in git history.

Regenerate with:

```bash
uv run python benchmark.py --base-url http://127.0.0.1:8080/v1 --model <name> --runs 3
```
