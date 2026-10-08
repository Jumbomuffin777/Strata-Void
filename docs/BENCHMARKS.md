# Benchmarks

Everything Strata Void claims, where it was measured, how, and how to run it again. The numbers describe **one
machine, one model and one build**; they are not a general property of Task-Parallel Requests. The per-task tables
are in [TASK_PARALLEL.md](TASK_PARALLEL.md#benchmarks-v2-enough-context-long-documents) (v2) and
[BATCHING.md](BATCHING.md#what-a-slot-costs-and-what-setup-recommends) (`--slot-context`); this page collects the
setup, the method and the summaries.

## The system

| | |
|---|---|
| CPU | AMD Ryzen 9 9950X, 16 cores / 32 threads |
| Board | ASUS ProArt B850-CREATOR WIFI NEO |
| RAM | 64 GB DDR5, 4 × 16 GB at 4800 MT/s |
| OS | Ubuntu Server 26.04.1 LTS, kernel 7.0.0 |
| GPU 0 | Intel Arc Pro **B70**, 32 GB (Battlemage G31, PCI id 8086:e223, AOT target `bmg-g31`) |
| GPU 1 | Intel Arc Pro **B65**, 32 GB (Battlemage G31, 8086:e222, `bmg-g31`) |
| GPU 2 | Intel Arc Pro **B60**, 24 GB (Battlemage G21, 8086:e211, `bmg-g21`) |
| Total VRAM | 88 GB |
| Software | Intel oneAPI DPC++ 2026.1, Intel compute runtime (`intel-opencl-icd`, `libze-intel-gpu1`) 26.31.39395.14 |
| Engine | this repository's SYCL engine, one AOT binary for `bmg-g21,bmg-g31` |

No other GPU took part in any result. Each card's idle VRAM was recorded before every run and checked after it
(below).

## The model

- [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF),
  folder `IQ4_XS/` (three shards, `Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-0000{1,2,3}-of-00003.gguf`): UkisAI's Swift 1.5
  fine-tune of Qwen3.8-Flash-Next, an MoE, quantized to IQ4_XS (~4-bit; a few tensors IQ4_NL).
- A Strata pack made with `tools/iq_pack.py --compat-bf16` (dense tensors in the engine's form; the experts are
  read from the GGUF), the MTP draft head from the base Qwen3.8-Flash-Next checkpoint (`tools/mtp_fetch.py`,
  `mtp_pack.py --experts q2_0`, `mtp_rt.py`) with the English draft vocabulary (`data/draft_vocab_en.bin` as
  the draft head's `draft_vocab.bin`), the repository's `data/expert-profile.bin`.
- **Every expert resident in VRAM**: 24,576 experts, 60.93 GiB (22.85 + 21.58 + 16.50 GiB on the B70 / B65 / B60;
  engine log: `100% of the experts resident`). With `--stream-experts` there is no host copy of the experts and the
  engine refuses to start unless every expert fits, so no configuration below spilled to RAM.
- Speculative decoding: MTP drafts, `--spec 4 --spec-min-p 0.5`, in every run.

## Configurations

Every configuration used `--prefill 2048 --no-prefill-borrow --vram-reserve-mib 512 --conversation-cache-mib 0`
and the environment in [examples/](../examples/). The layer split differs per configuration: it is whatever let
every expert fit beside that configuration's KV caches, so it is **not** one setting for all.

| used for | context | KV | `--batch` / `--batch-groups` | slot size | `--layer-split` | config |
|---|---|---|---|---|---|---|
| short complex requests; 8K / 16K documents | 32,768 | fp16 | 4 / 2 | full | 18,35 | `examples/strata-void-32k.json` |
| 32K documents | 65,536 | fp16 | 3 / 3 | full | 18,35 | `examples/strata-void-64k.json` |
| 64K and 120K documents | 131,072 | 8-bit | 4 / 2 | full | 18,36 | `examples/strata-void-128k.json` |
| context ceiling, partitioning test | 32K-128K | fp16 / 8-bit | 6 / 3 | `--slot-context 16384` | 18,34 / 18,35 / 18,36 | `examples/strata-void-128k-slots16k.json` (the 128K row) |
| v1 (4K, before this release's engine work) | 4,096 | fp16 | 6 / 3 | full | 18,34 | (`--prefill 512`) |

Full-size slots are what Task-Parallel Requests need to share a long document between steps. Six full-size slots
did not start at 32K or more on 88 GB; that is why `--slot-context` exists.

## Method

- **Wall-clock** is from sending the request to the last byte of the answer, through the HTTP server, streamed
  (`tools/task_parallel_bench.py`). Prompt token counts are the server's.
- **Every run is cold.** The runner puts a unique first line (`Run <id>.`) into every request, so no run reuses what an
  earlier one read. (An earlier draft without it let a later mode skip reading the same document; those numbers were
  discarded. `--warm` turns the line off.)
- **Greedy sampling**, the server's default reasoning effort.
- **Repeats:** the long-document headline cells (contracts-120k: ordinary, ordinary with a 4,096-token thinking
  budget, notes + synthesis; contracts-8k, config-8k, config-16k: ordinary and sections) ran **3 times** each. Every
  other cell, including the 10-task short-request table, ran **once** per task and mode. Medians and geometric means
  are over tasks, not over repeats.
- **Automatic checks** where a task has known answers (unit tests for code, expected figures for finance, expected
  names and figures for the seeded long documents: `tools/task_parallel_score.py`).
- **Blind grading** for the 7 open-ended short tasks: every answer shuffled under a random id, one sheet per task,
  graded by an LLM grader that did not know the modes (rubric items 0 / 0.5 / 1; correctness, completeness, overall
  1-10; truncated yes/no). Two separate grading passes: the first over the five modes of the main table, the second
  over ordinary / notes + synthesis / final AUTO. One grader per pass; the keys and grades are in the release assets.
- **Clean runs.** Each run was wrapped by a harness that recorded per-card VRAM before/after, the server process tree's
  swap, system memory pressure, and left-over processes, and gave a PASS / HOLD verdict. Every run released its VRAM
  and left no process behind. The HOLD verdicts came from system swap growth caused by unrelated services on the
  machine; the Strata process tree itself never swapped (`VmSwap` 0 for every run).
- One machine, one model, one build; LLM grading is noisy. Read differences of a few tenths in quality as ties.

## Results

### 1. Prompt reading: fast IQ4 dequant (SYCL)

New IQ4_XS / IQ4_NL → FP16 dequant kernels on the prompt path (`sycl/`), bit-identical output
(`STRATA_DQ_FAST=0` restores the old kernels):

| | old kernels | fast dequant |
|---|---:|---:|
| 16K context, `--prefill 2048` | 334 tok/s | **659 tok/s** |
| 16K context, `--prefill 512` | 195 tok/s | 417 tok/s |

With `--slot-context 16384` and the fast kernels, one request's prompt reading stayed at **~650-675 tok/s up to 60K
tokens and 626-631 tok/s at 121,698 tokens** (193-197 s), at 32K, 64K and 128K contexts. `--prefill 4096` does not
fit beside every expert. This is a prompt-reading result: generation after a long prompt is **32-38 tok/s**.

### 2. Context: `--slot-context`

| context | KV | layer split | starts (6 slots of 16K) | prompt read | decode |
|---|---|---|---|---|---|
| 32K | fp16 | 18,34 | no (17 experts short on the B60) | | |
| 32K | fp16 | 18,35 | yes | 675 tok/s at 31K tokens | 38 tok/s |
| 64K | fp16 | 18,35 | yes | 665 tok/s at 60K | 34 tok/s |
| 64K | 8-bit | 18,34 | yes | 652 tok/s at 60K | 32 tok/s |
| 128K | 8-bit | 18,34 | no (38 experts short) | | |
| 128K | 8-bit | 18,36 | yes | 626 tok/s at 122K (197 s) | 34 tok/s |
| 128K | fp16 | 18,36 | yes | 631 tok/s at 122K (193 s) | 34 tok/s |

Without `--slot-context` (six full-size slots) 32K and larger did not start. Full-size slots fit with fewer of them:
4 × 32K fp16, 3 × 64K fp16, 4 × 128K 8-bit (3 × 128K fp16 do not fit). The B65 and B60 give bit-identical results,
so moving their boundary changes no output. At 16K slots under a 32K and a 64K (8-bit) context, six requests run
concurrently gave the same text as each run alone (6 of 6, greedy), a cancelled stream left the others identical, and
a follow-up turn answered normally.

### 3. Short complex requests (32K context)

10 tasks (`bench/task_parallel_tasks.json`: code review, three functions, system design, database comparison,
NPV/IRR, four finance questions, research synthesis, contract review, latency hypotheses, migration plan), one run
each, cold:

| | ordinary | ordinary, 4,096 thinking budget | notes + synthesis (4) | sections only (4/direct) | AUTO (final) |
|---|---:|---:|---:|---:|---:|
| median wall-clock | 181 s | 184 s | **68 s** | 41 s | 78 s |
| geometric mean vs ordinary | 1.00 | 1.06 | **0.37** | 0.23 | **0.50** (0.29 on the 7 it split) |
| blind overall /10, pass 1 | 7.1 | 6.7 | **8.0** | 6.7 | |
| blind overall /10, pass 2 | 6.4 | | **8.0** | | **8.1** |
| cut off / empty / looping (of 7) | 2 (pass 2: 3) | 2 | 0 | 1 | 0 |

- The answer's own decode rate is the same in every mode (~40-45 tok/s). What changes is how long one request
  thinks before and while answering: the ordinary request thought for 8-16K tokens on four of these prompts.
- Sections only (`4/direct`) is the fastest form but its independently written sections contradicted each other and
  made arithmetic slips; it is **not** AUTO's choice for requests without a long document.
- Final AUTO split 7 of 10 (3-4 subtasks, notes + synthesis) and answered 3 normally (planner: "low" effort or no
  independent parts). AUTO is not always faster: on one task it answered normally and took 148 s where the ordinary
  run of the main table took 108 s (different cold runs think for different lengths).
- On the NPV/IRR task the ordinary request got every figure (9/9, after 5.8 minutes); every parallel form got the
  decision and NPVs right but some IRRs wrong (6/9).
- AUTO's gate sent **12 of 12 trivial requests** straight to the ordinary path (no planner call; 1.5-19 s).

### 4. Long documents (8K-120K tokens)

Seeded documents with known answers (`tools/longctx_tasks.py`, defaults: seed 11, contracts / reports / config at
8K, 16K, 32K, 64K, 120K). Full tables: [TASK_PARALLEL.md](TASK_PARALLEL.md#long-documents-the-scaling-curve).

- **Reading dominates**: ~16 s at 8K, ~53 s at 32K, ~104 s at 64K, ~202 s at 120K, the same in every mode.
- **Restoring instead of re-reading**: each later step is admitted in ~1 s (8-16K) to ~1.5-2 s (120K).
- Where the ordinary request thinks at length, task-parallel pays: contracts-120k, notes + synthesis
  **244 / 247 / 262 s** (6, 5, 4 of 6 checks) vs ordinary **439 s** (6/6) and two runs with **no answer** (thinking
  filled the context); a 4,096-token thinking budget: 302 / 294 / 306 s, 6/6 each.
- Where it does not (configuration lookups), it does not: 64K, 125 s ordinary vs 123-146 s.
- Sections written without thinking miss figures on long aggregations at 64K+ (1-3 of 6). Giving the section
  writers a thinking budget recovered some accuracy but lost the speed: batch slots decode without MTP drafts
  (~20-30 tok/s per slot vs ~45 alone).
- Partitioning a document into slices (6 slots of 16K over the 32K contracts): 147-201 s for 2-6 subtasks vs 161 s
  ordinary. Slices are read one after another, so it does not pay below the slot size.
- Final AUTO on long documents answers medium/high-effort requests in one stream that restores the planner's read:
  no faster than the ordinary request there (contracts-120k 427 vs 439 s; reports-120k: no answer either way), faster
  on lookups (contracts-8k 41 vs 45-51 s, config-16k 48 vs 63-126 s).

### 5. Retrieval hook

A stand-in JSON endpoint (five items: a pinned policy, a supplier list, a live budget, an irrelevant note, a
glossary) behind `HttpContextProvider`, with the 32K contracts plus three questions only the memory answers:

- one retrieval per parent request, **5 items, 204 tokens, 0-4 ms**;
- notes + synthesis (89.6 s) and AUTO (132 s) used the memory correctly (7/7); the ordinary request without memory
  (172 s) missed the budget question (6/7);
- with the endpoint failing (HTTP 404) every request still answered and reported `context.provider.error`.

### 6. Speculative decoding settings (single stream, 512 tokens, 4 prompts × 2)

| setting | decode (median) | draft acceptance | output |
|---|---:|---:|---|
| `--spec 4 --spec-min-p 0.5` (used everywhere) | 42.1 tok/s | 0.74 | reference |
| `--spec 4 --spec-min-p 0.3` | 41.3 tok/s | 0.69 | identical |
| `--spec 4 --spec-min-p 0.7` | 42.8 tok/s | 0.83 | identical |
| `--spec 5` | slower (20-26 s vs 11-14 s per answer) | 0.63 | **different**; on some prompts the model answered as if the prompt were garbled |
| `--spec 6` | slower (20-27 s per answer) | 0.62 | **different**, same symptom |
| `--spec 3` | | | did not start in this configuration |

Nothing above the qualified setting helped; values above 4 are unsupported on this build.

### 7. Feature off

- Server: with `task_parallel` absent, the outputs were **byte-identical** to the previous build's on the regression
  set (6 solo requests, 6 concurrent, a cancelled stream, a follow-up turn).
- Engine: the fast dequant kernels are bit-identical to the old ones; the final build's tokens matched the previous
  build's on five greedy base requests.

### 8. Release check (void-v0.1.0, this repository's tree)

A fresh checkout built with the README's commands (`build-sycl/strata`); the README's model steps rebuilt the pack
and the draft head byte-identical to the ones measured above (except a provenance path in `compat-bf16.json`); the
server started from `examples/strata-void-32k.json` in 95 s. Three ordinary greedy requests gave byte-identical
texts on this build and on the build the benchmarks used; a trivial `"auto"` request skipped the planner (1.5 s);
`"task_parallel": 2` on a two-database comparison planned, ran 2 subtasks and synthesized in 60 s. A one-line design
prompt (122 characters) sent with `"auto"` was answered as an ordinary request, as the gate intends: it thought for
5.6 minutes, which is the case the full-length prompt in the README is split for.

## Reproduce

```sh
# 1. the server (one of the example configs)
source /opt/intel/oneapi/setvars.sh
.venv/bin/python -m serve.server --engine strata --config examples/strata-void-32k.json --port 8095

# 2. short complex requests: the ordinary request, notes + synthesis, sections only, AUTO
.venv/bin/python tools/task_parallel_bench.py --url http://127.0.0.1:8095 --tasks bench/task_parallel_tasks.json \
    --modes off,4,4/direct,auto --out results/short.jsonl

# 3. long documents (generated deterministically; then use the 64K / 128K configs for the longer ones)
.venv/bin/python tools/longctx_tasks.py --out results/longctx_tasks.json
.venv/bin/python tools/task_parallel_bench.py --url http://127.0.0.1:8095 --tasks results/longctx_tasks.json \
    --ids contracts-8k,contracts-16k,reports-8k,reports-16k,config-8k,config-16k \
    --modes off,3,3/direct,auto --out results/long-16k.jsonl

# 4. automatic checks, and blind sheets (answers shuffled; the key is written separately)
.venv/bin/python tools/task_parallel_score.py --tasks bench/task_parallel_tasks.json --results results/short.jsonl \
    --out results/short-score.json --blind results/blind-sheet.md --key results/blind-key.json
```

Modes: `off`; `N` = N subtasks; `N/<compose>` sets `task_parallel_compose` (`synthesis`, `sections`, `direct`,
`auto`); `auto` = AUTO. `--repeat N` repeats every cell. Expect different absolute numbers on other hardware;
compare modes on one machine.

The raw evidence is attached to the [v0.1.0 release](https://github.com/Jumbomuffin777/Strata-Void/releases/tag/void-v0.1.0):
every answer with its `task_parallel` metadata and timings, the blind sheets, keys and grades, the speculative and
context probes, and the per-run cleanup verdicts. Machine-specific paths are removed from it.
