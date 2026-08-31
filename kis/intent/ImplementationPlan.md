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

Static HTML generated from results JSON.

Proof: generated page opens and matches the JSON numbers.
