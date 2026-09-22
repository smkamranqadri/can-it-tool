# Project

## What

`can-it-tool` is a local benchmark harness that measures how reliably an LLM performs
tool calling against a simulated school-management system (SchoolBox / Hermes shaped).

It is not a school system. It is a deterministic simulation plus a scoring engine.

## Who

Single operator (the repo owner), running local models on their own machine.

## Problem

Before buying hardware, we need to know the smallest local model that is reliable
enough to drive a SchoolBox agent. "Reliable" here means: picks the right tool, passes
correct arguments, chains lookups, asks for clarification instead of guessing, and
never performs an unauthorized or unconfirmed mutation.

## Success

The harness answers, from JSON artifacts alone: for a given model + quantization +
runtime, what is its tool-call accuracy, task success rate, and safety failure rate,
and how does that compare against other models on other machines.

## Constraints

- No auth, no production database, no Docker, no vector DB, no RAG, no real SchoolBox
  integration, no elaborate frontend.
- CLI and scoring engine are primary. A static results page is secondary.
- Must work against any OpenAI-compatible `/v1/chat/completions` endpoint.
- Deployment target for the user's project: HP EliteDesk 705 G5, AMD Ryzen 5 PRO 3400G
  (Zen+, 4 cores / 8 threads, up to 4.2 GHz, AVX2, no AVX-512), dual-channel DDR4-2933
  (~47 GB/s), Radeon Vega 11 iGPU sharing system RAM, 16 GB RAM (expandable to 32), OS Omarchy
  (Arch Linux, bash by default). CPU-first; the iGPU is usable via
  llama.cpp's Vulkan build. All measurements so far are on an M2 Pro (~200 GB/s), so
  absolute latency and throughput must be re-measured on the target.

## Stage

v1 prototype, deliberately small.
