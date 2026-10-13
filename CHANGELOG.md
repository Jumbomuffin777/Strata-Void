# Changelog

Strata Void's own releases. Upstream Strata's changes are in its history and
[release notes](https://github.com/Niko1221/Strata/releases); this fork branched from upstream `a5682fe`.

Strata Void's version numbers are its own, independent of upstream's engine versions (0.1.x). Its tags are named
`void-vX.Y.Z` so they never collide with upstream's `vX.Y.Z` tags.

## Unreleased

**MoE expert placement maps (tooling; no engine change)**

- `tools/moe_placement/` + `tools/moe_place.py`: header-only adapters for `qwen4exp`, `glm5next` (GLM-5.3-Flash) and `deepseek4` (DeepSeek V4 Flash); a versioned, fingerprinted map format (`strata.moe-map` v1) with import validation; a planner that ranks experts by measured routing frequency under explicit VRAM, KV, state and bounded-RAM budgets; a canonical-id lookup whose reordering is tested bit-exact; STRP export for `--expert-profile` (`qwen4exp` only).
- GLM and DeepSeek are planned, not run: the engine has no layers for them and the exporter refuses. No routing statistics exist for them yet. See [MOE_PLACEMENT](docs/MOE_PLACEMENT.md).

## void-v0.1.1 (2026-10-11)

**Server protocol fixes, not a new SYCL kernel optimization**

- Port selected upstream fixes with original authors and source-commit attribution: honor `tool_choice`; tolerate malformed historical arguments; map flat namespace aliases; preserve images in Anthropic tool results; report invalid tool schemas as HTTP 400.
- Distinguish executable declared calls from quoted tool-call markup in reasoning and visible code examples. A scoped publication review corrected long-fence and multi-backtick tracking across reasoning boundaries while preserving server-written forced calls. Reasoning calls cut by length/cancel/error remain text; incomplete wrappers have a bounded buffer. Streamed and whole parsing are regression-tested.
- Keep agent conversations on the ordinary path so Task-Parallel decomposition does not lose tool calls/results, including when current tools are suppressed with `tool_choice: none`.
- Poll quiet-engine cancellation every 0.5 seconds without increasing HTTP heartbeat frequency, and explicitly close HTTP/1.0 connections. Native prompt-chunk cancellation latency is not newly qualified.
- Add parser specimens and HTTP/cancellation regression tests; make the optional `jsonschema` test dependency explicit. Scoped result: 315 tests passed, 1 optional test skipped, and 11 SYCL setup checks passed. Inert image-selector comparisons establish unchanged API-caller authority, without fetching URLs or reading image targets.

**Newly documented external integrations**

- IQ3_S inference and separate 32K/fp16 and 128K/int8 profile loading, alongside retained Swift IQ4_XS inference, on Arc Pro B70 32 GB + B65 32 GB + B60 24 GB. IQ3_S filled-128K reliability remains untested.
- OMP 18.3.4 coding edit plus approved real test execution, and REA 6.4.0 MCP static analysis with exact Evidence/result verification through OMP. These tools, approvals and OS containment belong to external harnesses, not the native engine.
- 50.6 tok/s live IQ3_S generation sample, its fixed workload and server-observed 2-second-window measurement method. No benchmark speedup is claimed.
- Generic IQ3_S profiles, OMP model/approval examples and a disabled-by-default upstream REA stdio registration. No private controller, configuration, credentials, workspace contents, binary or weights are included.

See [AGENT_INTEGRATION](docs/AGENT_INTEGRATION.md) and the [aggregate summary](bench/results/2026-10-11-agent-compatibility/summary.json). Existing upstream MIT attribution, earlier benchmarks and functionality are preserved.

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
