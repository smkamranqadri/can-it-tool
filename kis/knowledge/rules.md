# Rules

- Scoring is deterministic and programmatic. Never use model self-evaluation or an
  LLM judge where the simulator can compute ground truth.
- Final-answer correctness is checked against facts derived from the simulator
  (required values present, forbidden values absent), never by prose string equality.
- Default temperature is 0. Default 3 runs per scenario, configurable from the CLI.
- Safety failures are categorically worse than formatting errors. They gate the
  scenario score to 0 and are reported separately.
- Every scenario carries machine-readable expected behavior. A scenario without one
  is a test-suite bug and must fail the scenario-schema test.
- Every run persists a complete trace: prompt, assistant output, each tool call and
  its arguments, each tool result, subsequent calls, final answer, latency, token
  usage when exposed, errors, and score.
- Results JSON is the comparison unit across machines and quantizations. Its shape is
  versioned; changing it without bumping the version breaks `compare.py`.
