# Current State

Branch: main
Task: Phase 2 - client and runner. Complete, awaiting review before Phase 3.
Mode: Phase
Status: done, blocked on user review
Command: `uv run --with pytest --with httpx --python 3.14 pytest -q`

## Proof

- `pytest -q` - 117 passed (61 sim, 42 runner, 14 client)
- Phase 2: the loop terminates for exactly four reasons - final_answer, max_steps,
  timeout, client_error - and nothing the model emits can terminate it. Malformed
  arguments, invented tool names, and tool errors are recorded as benchmark
  behavior and the run continues.
- Simulator is deterministic: `build_dataset()` is equal across builds, simulated
  today is Monday 2025-05-12, 48 students / 24 assignments / 192 submissions.
- Seeded scenario facts verified: Ahmed Raza (STU-0017) is present today; two
  Sara Khans (STU-0009 in 5B, STU-0033 in 7A) with different balances (12000 vs
  13000); two Ali Hassans (STU-0018 in 6A, STU-0041 in 7B); grade 6 Mathematics
  has non-submitters in both 6A and 6B; 7A "tomorrow" resolves to Tuesday.

## Blocker

None. Waiting for the user to review Phase 2 before Phase 3 (scoring engine).

## Next

Phase 3: the eleven scoring dimensions, the weighted rubric, the safety gate, and
aggregation into category and overall metrics.

## Decisions taken

- Design approved by the user with three amendments: keep all 54 scenarios, add
  `record_fee_payment` and `update_submission_status` as WRITE tools, and bind
  confirmation tokens to the exact tool and arguments.
- Confirmation tokens are argument-bound, single-use, and issued only by the
  simulator. Token verification happens in `execute` before fault injection and
  before any business-rule validation, so a replay can never be masked by an
  unrelated error such as a zero balance or a missing entity. Token issuance still
  happens after entity validation, so no token is issued for a nonexistent student.
- The simulator records only mechanical violations (forged, mismatched, reused
  tokens). Scenario policy violations (a mutation in a `must_refuse` case) are the
  scorer's job, read from `store.mutations`.
- Native `tool_calls` only in v1; `PROTOCOLS` in `canit/client.py` is the seam and
  an unimplemented protocol name is rejected rather than silently falling back.
- The runner owns termination; the client raises `ClientError` / `ClientTimeout`
  and never retries a model mistake. HTTP-level retries (429/5xx) are the client's
  and are capped by `max_retries`.
- Tool results are fed back as `role: tool` messages with `tool_call_id`, and the
  raw assistant message is stored verbatim alongside the parsed view.
- Scoring is a weighted mean of eleven dimensions with a hard safety gate. No LLM
  judge. Fixed simulated date and RNG seed; fresh store per scenario run.
