# Changelog

Strata Void's own releases. Upstream Strata's changes are in its history and
[release notes](https://github.com/Niko1221/Strata/releases); this fork branched from upstream `a5682fe`.

Strata Void's version numbers are its own, independent of upstream's engine versions (0.1.x). Its tags are named
`void-vX.Y.Z` so they never collide with upstream's `vX.Y.Z` tags.

## void-v0.1.0 (2026-10-07)

First public release. Measured on three Intel Arc Pro GPUs (B70 + B65 + B60, 88 GB) with Swift 1.5 Qwen3.8
Flash-Next IQ4_XS; see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**Task-Parallel Requests** (`serve/task_parallel.py`, `serve/context_pool.py`)

- `"task_parallel": "auto" | 2..8` per request (or a server default): a planner splits one request into subtasks
  that run concurrently in the engine's batch slots; one answer is streamed. Off unless asked for.
- AUTO: a model-free gate (12/12 trivial requests skipped the planner), free-slot awareness, the planner's effort
  rating, one retry of an unusable plan.
- Answer forms (`task_parallel_compose`): notes + one synthesis (AUTO's choice without a long document), sections
  written in parallel (`direct`, `sections`), and for a long shared document one stream that restores the planner's
  read.
- Large context: a document that fits a slot is read once and restored by every step (`ckpt=`); a larger one is
  cut at its own structure and assigned to subtasks (BM25, shards), with one bounded `NEED:` follow-up round.
- `ContextProvider` / `HttpContextProvider`: one retrieval per request, provenance and authority kept, live state
  marked with its time, fails open.
- Response metadata (`task_parallel`): decision, timings, tokens of every step, coverage.
- Measured: complex requests median 181 s → 68 s (notes + synthesis), AUTO geometric mean 0.50× wall-clock (0.29×
  on the requests it split), blind quality 8.0-8.1 vs 6.4-7.1 for the ordinary request. Visible decode unchanged.

**SYCL engine**

- Fast IQ4_XS / IQ4_NL → FP16 dequant on the prompt path, bit-identical: 334 → 659 tok/s at 16K (`--prefill 2048`),
  600-630 tok/s out to 122K tokens.
- `--slot-context N`: batch slots smaller than the main context; 32K / 64K / 128K contexts with six 16K slots
  (six full-size slots did not start at 32K+ on 88 GB).
- `ckpt=N` shared-prefix checkpoints (`INFO prefix_ckpt=1`).
- Ported: batch slots in pipelined groups across a layer split (upstream #465), per-stage expert residency, the
  plan-error flag (upstream #871), opt-in asynchronous stage commits, Arc Pro B65 / B60 PCI ids in setup.

**Tools and docs**

- `tools/task_parallel_bench.py` (cold runs by default), `task_parallel_score.py` (checks, blind sheets),
  `task_parallel_report.py`, `longctx_tasks.py` (seeded long documents with known answers), `moe_prefill_bench`.
- docs: TASK_PARALLEL, BENCHMARKS, ARCHITECTURE, BATCHING (`--slot-context`); example run configs.

Known limitations: [README](README.md#limitations).
