# Current State

Branch: main
Task: Phase 1 - simulator. Complete, awaiting review before Phase 2.
Mode: Phase
Status: done, blocked on user review
Command: `uv run --with pytest --python 3.14 pytest -q`

## Proof

- `pytest -q` - 61 passed (tests/test_sim.py)
- Simulator is deterministic: `build_dataset()` is equal across builds, simulated
  today is Monday 2025-05-12, 48 students / 24 assignments / 192 submissions.
- Seeded scenario facts verified: Ahmed Raza (STU-0017) is present today; two
  Sara Khans (STU-0009 in 5B, STU-0033 in 7A) with different balances (12000 vs
  13000); two Ali Hassans (STU-0018 in 6A, STU-0041 in 7B); grade 6 Mathematics
  has non-submitters in both 6A and 6B; 7A "tomorrow" resolves to Tuesday.

## Blocker

None. Waiting for the user to review Phase 1 before Phase 2 (client and runner).

## Next

Phase 2: `RunConfig`, the OpenAI-compatible client behind the protocol seam, and
the bounded agent loop producing a full trace.

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
- Native `tool_calls` only in v1, behind a `--protocol` seam.
- Scoring is a weighted mean of eleven dimensions with a hard safety gate. No LLM
  judge. Fixed simulated date and RNG seed; fresh store per scenario run.
