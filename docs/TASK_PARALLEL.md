# Task-parallel requests

One chat request, split into subtasks that run **at the same time** on the engine's concurrent request capacity, then
merged into one answer.

```
request ──► plan ──► subtask 1 ─┐
                     subtask 2 ─┤   internal requests, running concurrently in the engine's batch slots
                     ...        ─┤   (the same loaded model; no extra model instances)
                     subtask N ─┘──► synthesis ──► one answer (streamed)
```

Opt-in, per request (`"task_parallel"`) or as a server default. Off, the server behaves exactly as before.

## What it is - and what it is not

- It is **task parallelism**. A planner call decides whether a request has independent parts (or would gain from
  independent attempts at it); the parts run as ordinary internal requests side by side; a synthesis call writes
  the final answer from their work products. What can shrink is the **wall-clock** of a request whose work can be
  done side by side.
- It is **not** speculative decoding and it does **not** make any token stream faster. Each internal request decodes
  at the engine's normal per-request rate (in a batch slot that is *slower* than a request alone). The final answer
  is written by one request at the normal single-stream rate. A response's `worker_aggregate_tok_s` is the
  subtasks' combined rate, not the answer's.
- Workers hand over **work product** (findings, calculations, code, a proposed solution). An internal request's
  reasoning is never passed on, returned or stored: it is counted and discarded.
- It always costs **extra compute**: plan + subtasks + synthesis read several prompts and generate tokens that one
  request would not. Every response reports them (`total_generated_tokens`, per-subtask prompt and output counts).

## Modes

| `task_parallel` | behavior |
|---|---|
| absent, `null`, `false`, `"off"`, `0`, `1` | the ordinary request, unchanged |
| `"auto"` (or `true`) | a model-free gate first; if it passes, the planner chooses 1 (answer normally) or 2..`max_workers` subtasks |
| `2` .. `8` | the planner is asked for exactly that many subtasks (the ordinary request if it cannot give a valid plan) |

The server default comes from the run config: `"task_parallel": "auto"`, or an object
`{"default": "auto", "max_workers": 6, ...}` (every key below is optional). A request's own value always wins.

Not used (the request runs as usual) with `tools`, a structured `response_format`, or images; `/v1/chat/completions`
only (v1).

## When AUTO splits work

AUTO is conservative and transparent; every decision is reported (`decision`):

1. **Gate (no model call).** A score from the request text: short (< 160 characters) −3; ≥ 600 / ≥ 1500 characters
   +1 / +2; three or more listed items (lines, or inline `(1) (2) (3)`), or a sentence that enumerates four or more
   parts (parenthesized asides not counted) +2; three or more questions +1; analysis wording (compare, analyze,
   design, review, debug, evaluate, risk, plan, recommend, for each, ...) up to +3; code +1; a simple edit or chat
   opener (rewrite, translate, tell me a joke, hi, ...) −2. Below `gate_threshold` (2): the ordinary request, no
   planner call (cost: nothing).
2. **Planner** (no thinking, greedy, ≤ `planner_tokens`) returns JSON with an `effort` estimate and a
   `parallelism`. It is told to choose 1 for a short, simple, conversational or creative request, a quick fact, or
   one chain of dependent steps.
3. **Effort rule.** If the planner rates the request at an effort in `auto_skip_effort` (default `"low"`), AUTO
   answers normally: subtasks + synthesis only pay off when one request would otherwise think for a long time (see
   the break-even below).
4. **Load (v2).** AUTO counts only the batch slots other requests are not holding; with fewer than two free it
   answers normally (no planner call).
5. **Large documents (v2).** A document kept whole in the shared prefix: the planner reads it (the one read); a
   "low"-effort request gets its sections written in parallel, a "medium"/"high" one is answered by one stream that
   restores the planner's read. A document larger than a slot: sliced, see below.

## How a request flows

1. **Shared context.** One system message holds a short step guide, the application's instructions, the earlier
   conversation and the user's request. Every internal request starts with exactly this message, rendered with the
   same template settings (`internal_effort`), so the engine can read it **once**: with an engine that reports
   `INFO prefix_ckpt=1`, each internal request carries `ckpt=<length of that prefix>`; the engine checkpoints there
   and later internal requests restore the checkpoint and read only their own few tokens (copy, not aliasing: each
   slot gets its own KV state).
2. **Plan**: `{"effort": "low"|"medium"|"high", "parallelism": n, "strategy": "partition"|"independent",
   "subtasks": ["...", ...]}` (a subtask may also be an object with `objective`, `expected_output`, ...). Validated
   strictly: count within limits (exactly `n` when a count was asked), non-empty, distinct, bounded length; anything
   malformed → the ordinary request. `partition`: distinct parts, one per subtask. `independent`: each subtask
   attacks the whole problem from another angle (solution, alternative, review, edge cases).
3. **Subtasks**, all at once, each `STEP subtask i/N: <objective> Limit: about W words.` Output budgets: `total_tokens`
   shared out, each within `[min_tokens, max_tokens]`, scaled down so the synthesis still fits the context. Workers
   answer without thinking (`worker_reasoning` 0: measured faster at equal quality). They are plain internal requests:
   they cannot plan, spawn or recurse.
4. **Synthesis**: the original request plus the subtasks' notes; told the notes may be incomplete, overlapping or
   wrong, to check calculations and claims, resolve contradictions by the stronger reasoning, remove duplication,
   keep caveats, cover what is missing, and answer directly without mentioning the process. Its output is the
   response, streamed like any answer (thinking capped at `synthesis_reasoning` unless the request sets
   `reasoning_budget_tokens`; none if the request disabled thinking).

## Large context (v2)

A request can bring much more material than a short prompt: a long document pasted into the message (or the system
prompt), documents sent with it (`"context_items": [{"text", "title", "source", "kind", "authority", "as_of",
"pinned"}, ...]`) and what a retrieval provider returns. There are two regimes, chosen per request:

**1. Shared (the material fits one batch slot).** The material stays whole in the shared prefix. The planner's
prompt read is the one read of it - the same read an ordinary request would do - and the engine checkpoints there
(`ckpt=`); every later step (subtasks, section writers, the synthesis) *restores* that state instead of reading the
material again (a device-side copy, ~1 s per admission at 16K, measured `reused_tokens` = the whole prefix). Every
step sees all of the material, so facts that span documents are never split apart. This needs slots that hold the
whole material: on the three-card Arc deployment (B70 + B65 + B60) 4 slots at 32K (fp16), 3 at 64K (fp16) or 4 at 128K (8-bit KV) fit beside
every expert (see BATCHING.md).

```
request + material ──► planner (reads it all once; checkpoint at the end of the shared prefix)
                          │
                          ├─► step 1: restore (copy) + its own ~100 tokens ─┐
                          ├─► step 2: restore (copy) + its own ~100 tokens ─┼─► answer
                          └─► step N: restore (copy) + its own ~100 tokens ─┘
```

**2. Partitioned (larger than a slot holds, `share_context`).** The material is cut into chunks at its own
structure - whole top-level documents (a contract, a report, a file) where they fit a chunk - and the planner reads
an index of them, never the material:

```
material ──► chunks (whole documents where they fit; ~chunk_tokens each)
             ├─► INDEX: one line per chunk (id, title, first words)  ──► the planner reads only this + the request
             ├─► pinned items (pinned, live state, a few durable memories) ──► every step
             └─► the chunks, assigned: named "sources" of a subtask first, then every unnamed chunk to the subtask
                 it fits best (BM25), or an even split in reading order ("shard": the same question over every part)
```

- Strategies: `partition` (distinct parts), `independent` (angles on the whole), `shard` (one objective, the
  material split evenly, every part read once). When the planner sees no independent parts (or its plan is
  unusable) and the material is partitioned, the request is sharded anyway: one ordinary request would have to read
  all of it and think over all of it.
- A subtask reads only its own chunks, at most `worker_source_tokens` and never more than one batch slot holds
  (`--slot-context`), so it runs beside the others; a shard reports every relevant fact of its part (with chunk
  ids) and gets an output budget of about a fifth of what it reads. The response reports coverage (`sources`).
- **Facts that span partitions** are combined by the synthesis from the notes - a total over all contracts, a
  maximum over all units, a value one component defines in terms of another's parameter.
- **One bounded follow-up round**: a subtask that needs something outside its chunks writes `NEED: <what>` (at most
  `need_per_worker`, `need_total` in all); each is answered once from the best-matching chunks (BM25, at most
  `need_tokens`) by one more internal request. Never recursive: no follow-up of a follow-up, no second plan.
- Partitioning does **not** make the material faster to read: the engine reads one prompt at a time, so N slices
  cost what the whole costs (a little more), and a slot that is decoding pauses while another subtask's prompt is
  read. Measured, it pays only when one request would think at length over the whole material.

AUTO counts large material as a strong signal (+3 at the gate), uses only the slots other requests are not holding,
caps subtasks so no slice is thinner than `min_shard_tokens`, and for partitioned material compares its cost model's
estimate for one request against the parallel plan (`estimate` in the response).

## Memory and retrieval (v2)

`serve/context_pool.py` defines a small provider interface; nothing in it is specific to one memory system:

```python
class ContextProvider(Protocol):
    def retrieve(self, query: str, k: int, budget_tokens: int) -> list: ...   # items, never instructions
```

and `HttpContextProvider` adapts any JSON endpoint (`POST {"query", "k", "budget_tokens"}` -> `{"items": [...]}`),
configured as `"task_parallel": {"context_provider": {"url": "...", "timeout_s": 5}}`.

- **Retrieve once.** The provider is called once per parent request (with the request, or its head and tail when
  long), never by subtasks: no recursion, no per-worker queries. Items are deduplicated (by content) and packed
  into `provider_tokens`.
- **Kinds and authority.** Every item keeps its `source` (provenance) and `authority`, shown in its header.
  `memory` items are durable knowledge; `live` items are mutable state, always shown with their `as_of` time and
  never treated as durable facts; `document` items are material. Live state and pinned items are one shared copy
  every step sees (within `pinned_tokens`); the rest are assigned to subtasks like the material.
- **Fail open.** A provider error or timeout is reported (`context.provider.error`) and the request goes on
  without memory; memory augments a request, it never blocks one.

## Writing the answer in parallel (v2)

v1 wrote the answer with one synthesis call, which dominated long answers (45-70 % of the wall-clock). v2 has
three ways to write it, and `auto` chooses between them (`compose`, server config or `"task_parallel_compose"` per request):

| compose | stages after the plan | the answer |
|---|---|---|
| `synthesis` (v1) | subtasks write notes | one call writes it from the notes |
| `sections` | subtasks write notes; an outline call designs the sections and a **fact ledger** (the figures every section must use) | the sections written at the same time, from the notes and the ledger (implemented and unit-tested; not benchmarked in this pass, so `auto` does not use it) |
| `direct` | the planner designs the answer's sections; each subtask **writes its section** (no notes, no outline) | one parallel stage; at most two closing sections (an overall summary, the recommendation) are written after the others, from their text |
| `auto` (default) | a large document kept whole in the shared prefix: `direct` (AUTO answers a medium/high-effort request there in one stream that restores the planner's read); otherwise notes + one synthesis | measured: see Benchmarks (v2) |

Assembly is deterministic, not a rewrite: sections are streamed **in order as they are written** (section 1 as it
is generated, each later one once the ones before it are out), every section starts with its heading, a paragraph
that repeats an earlier one (3-word shingles, Jaccard >= 0.8) is dropped. The "editor" costs no model call. A
section that fails is retried once, then left out (`sections_missing`). The outline (or, for `direct`, the plan's
outline) is a second shared prefix: the engine reads it once and the writers restore it.

Section writers decode side by side, so the answer's aggregate rate is the slots' rate (e.g. 4 writers x ~23 tok/s);
each section's own stream is still one request's rate, and the client sees one ordered stream.

## Streaming

Workers run internally; the client receives one answer: the synthesis stream. Progress is sent as SSE comments
(`: task_parallel planning`, `: task_parallel running N subtasks`, `: task_parallel synthesizing`), which standard
clients ignore, plus keep-alives every 5 s. Worker answers are never streamed into the response.

## Failures and cancellation

- Planner error, timeout or unusable plan → the ordinary request (the response says why).
- A subtask that errors or returns nothing is retried once (`retries`); one that exceeds `worker_timeout_s` is
  stopped; the synthesis is told which parts did not complete and to cover them. No subtask output at all → the
  ordinary request.
- The whole orchestration is bounded (`total_timeout_s`); nothing waits forever.
- A client that disconnects (stream or not) stops the planner, every running subtask (`BSTOP` of its slot) and the
  synthesis; the engine is left clean for the next request.

## Response metadata

Every task-parallel response carries `task_parallel` (non-streamed: in the body; streamed: one last chunk with
`"choices": []` before `[DONE]`):

```json
"task_parallel": {"mode": "6", "enabled": true, "workers": 6, "strategy": "partition", "decision": "6: partition",
  "planning_ms": 5508, "planner_tokens": 161, "planner_prompt_tokens": 688, "workers_ms": 16129,
  "worker_tokens": 1156, "worker_aggregate_tok_s": 71.7, "synthesis_ms": 44018, "synthesis_tokens": 1606,
  "synthesis_reasoning_tokens": 132, "total_ms": 65655, "total_generated_tokens": 2923,
  "worker_results": [{"tokens": 184, "reasoning_tokens": 0, "prompt_tokens": 609, "reused_tokens": 561,
                      "finish": "stop", "start_ms": 2, "first_token_ms": 5640, "end_ms": 15354, "attempts": 1}, ...]}
```

(a real response: the contract-review task below, 6 workers; `reused_tokens` 561 of 609 is the shared prefix the
engine restored from its checkpoint; `first_token_ms` 5640 shows this was the last of six subtasks admitted.)

`"task_parallel_diagnostics": true` adds the subtasks' objectives. Worker text and any reasoning are never
returned. `usage` is the synthesis' own (the answer the client received).

## Configuration (run config `"task_parallel"` object)

| key | default | meaning |
|---|---:|---|
| `default` | off | the server-wide mode |
| `max_workers` | 6 | most subtasks (also capped by the engine's concurrent slots) |
| `min_tokens` / `max_tokens` | 96 / 320 | one subtask's output budget |
| `total_tokens` | 1200 | all subtasks' output budgets together |
| `worker_reasoning` | 0 | a subtask's thinking budget (0: none) |
| `planner_tokens` | 300 | the plan's budget |
| `synthesis_reasoning` | 192 | the synthesis' thinking budget (0: none set) |
| `synthesis_min_tokens` | 1024 | room kept in the context for the answer |
| `worker_timeout_s` / `total_timeout_s` | 240 / 600 | bounds |
| `retries` | 1 | a failed subtask is tried again once |
| `gate_threshold` | 2 | AUTO's heuristic score needed before the planner is asked |
| `auto_skip_effort` | "low" | planner effort ratings at which AUTO answers normally |
| `internal_effort` | "low" | the template reasoning effort all internal requests are rendered with |
| `partition_min_tokens` / `chunk_tokens` | 6000 / 1500 | material above this is partitioned (unless shared, below); target chunk size |
| `share_context` | -1 | material up to this many tokens stays whole in the shared prefix (read once, restored by every step); -1: what one batch slot holds less 4,096; 0: off |
| `worker_source_tokens` | 24000 | the most material one subtask reads (also bounded by a slot's context) |
| `min_shard_tokens` | 2000 | AUTO: fewer subtasks rather than slices thinner than this |
| `pinned_tokens` | 1500 | pinned items, live state and the top durable memories every step sees |
| `need_per_worker` / `need_total` / `need_tokens` | 2 / 6 / 3000 | the one follow-up round |
| `provider_k` / `provider_tokens` | 8 / 3000 | one retrieval per request |
| `context_provider` | none | `{"url", "timeout_s", "headers"}`: a JSON retrieval endpoint |
| `compose` | "auto" | how the answer is written: synthesis, sections, direct, auto |
| `auto_shared_one_effort` | "medium,high" | AUTO with a large shared document: these planner ratings are answered in one stream |
| `synthesis_max_tokens` | 12288 | the most the notes' synthesis writes when the request sets no `max_tokens` (0: no cap) |
| `section_tokens_per_word` | 2.2 | a section's output budget per planned word |
| `writer_reasoning` / `outline_tokens` / `outline_reasoning` | 0 / 500 / 192 | section writers' thinking; the outline's budgets |
| `max_sections` / `section_min_words` / `compose_min_words` | 6 / 120 / 500 | section limits; "auto" writes sections only for an answer at least this long |
| `prefill_tok_s` / `decode_tok_s` / `slot_tok_s` / `admission_s` | 600 / 38 / "2:30,3:26,4:23,6:19" / 0.9 | AUTO's cost model (this deployment's measured rates) |

## Engine requirements

Any engine behind `serve/server.py` works: the orchestration (`serve/task_parallel.py`) only issues ordinary requests
through a small `Backend` interface (run one request; how many can run at once) and knows nothing about GPUs or
engine builds. To gain wall-clock the engine must run requests concurrently (Strata: `--batch N`, with a layer split
also `--batch-groups G`); with one request at a time AUTO always answers normally. The shared-prefix checkpoint
(`ckpt=`, `INFO prefix_ckpt=1`) is an optimization; without it every internal request reads the shared context again.

## Latency vs compute (v1 measurements, 4K context)

Where the time goes (one request; medians over 10 decomposable tasks, the three-GPU deployment below):

| phase | 2 workers | 4 | 6 |
|---|---:|---:|---:|
| plan (read the shared context, write ~70-160 JSON tokens) | 6.6 s | 4.7 s | 5.1 s |
| subtasks (all admitted ~0.9 s apart, then decoded side by side) | 18.1 s | 19.4 s | 15.8 s |
| synthesis (read the notes, ≤ 192 thinking tokens, write the answer) | 44.1 s | 55.6 s | 47.0 s |
| subtasks' aggregate decode rate | 34.9 tok/s | 59.5 tok/s | 71.5 tok/s |
| final answer's own decode rate | 39.6 tok/s | 40.5 tok/s | 42.0 tok/s |

The final answer is still written by one stream at the normal rate; the gain comes from replacing a long
single-request reasoning phase with a short plan, a parallel subtask phase and a short synthesis thinking budget.
The price is a fixed delay before the answer starts: the first answer token arrives after 31-43 s (plan + subtasks
+ reading the notes + ≤ 192 thinking tokens), where an ordinary request with a 192-token thinking budget starts
answering after 7-10 s. So task-parallel pays when one ordinary request would spend longer than that before or
while answering - long thinking, or an answer that runs into the context limit - and loses when the request can be
answered well with little thinking (measured: three short-answer tasks, 20-29 s with a 192-token budget, 47-60 s
task-parallel).

The extra compute is real and reported: task-parallel reads 1,700-2,400 prompt tokens per request (planner, each
subtask's own tokens past the shared prefix, the synthesis' notes) where one request reads ~220, keeps up to N batch
slots busy for 15-20 s, and generates 2,300-3,100 tokens in all (plan + subtasks + answer) - about as many as one
ordinary request with a short thinking budget (2,900) and fewer than one thinking at length (3,300-3,600).

## Benchmarks (v2: enough context, long documents)

Measured on one deployment (three Intel GPUs: Arc Pro B70 + B65 + B60, Swift 1.5 Qwen3.8 Flash-Next IQ4_XS, MTP drafts,
every expert in VRAM, `--prefill 2048`, the fast prefill dequant). Unlike v1 (4K context, where the ordinary
request ran out of room), every configuration here gives the ordinary request **enough context**: 32K (fp16 KV,
4 full-size slots, split 18,35) for the short tasks and 8K/16K documents, 64K (fp16, 3 slots) for 32K documents,
128K (8-bit KV, 4 slots, split 18,36) for 64K and 120K documents. Within a row every mode ran on the same server.
**Every run is cold**: the benchmark puts a unique first line in each request (`Run <id>.`), so no run reuses what an
earlier one read (an earlier draft of this benchmark let the second mode over the same document skip reading it:
those numbers are discarded). Greedy sampling, one run per cell unless marked ×n, wall-clock seconds from sending
the request to the end of the answer.

Modes: `off` = the ordinary request (default effort, thinks as long as it wants); `off+b4096` = the same with a
4,096-token thinking budget; `N` = task-parallel with notes + one synthesis (`compose` "synthesis"); `N/direct` = the
subtasks write the answer's sections (`compose` "direct"); `auto/auto` = AUTO with `compose` "auto" (AUTO's rules
changed during this pass - see the end of this section).

### Short complex requests (the v1 task set, now with a 32K context)

Wall-clock s (automatic checks where the task has them; ¹ = AUTO answered with one request):

| task | prompt tokens | off | off+b4096 | 4 | 4/direct | auto/auto |
|---|---:|---:|---:|---:|---:|---:|
| code-bugs | 387 | 108 (3/3) | 116 (3/3) | 46 (3/3) | 31 (3/3) | 31 (3/3) |
| code-funcs | 204 | 39 (15/15) | 61 (15/15) | 52 (15/15) | 34 (15/15) | 49 (15/15)¹ |
| arch-dispatch | 174 | 320 | 232 | 89 | 53 | 52 |
| db-compare | 155 | 123 | 231 | 72 | 39 | 43 |
| fin-npv | 282 | 348 (9/9) | 152 (8/9) | 66 (6/9) | 46 (6/9) | 44 (6/9) |
| fin-quick | 210 | 70 (3/3) | 97 (3/3) | 48 (3/3) | 30 (2/3) | 71 (3/3)¹ |
| research-2008 | 146 | 155 | 150 | 75 | 38 | 609¹ |
| contract-review | 387 | 652 | 623 | 71 | 43 | 40 |
| latency-hypotheses | 231 | 207 | 216 | 66 | 56 | 56 |
| migration-plan | 149 | 319 | 414 | 83 | 45 | 44 |

| | off | off+b4096 | 4 | 4/direct | auto/auto (as run) |
|---|---:|---:|---:|---:|---:|
| median wall-clock | 181 s | 184 s | 68 s | 41 s | 46 s |
| geometric mean vs `off` | 1.00 | 1.06 | **0.37** | 0.23 | 0.34 |

Blind grading (7 rubric / figure-checked tasks, answers shuffled under random ids, graded by an LLM grader that did
not know the modes; scores 1-10):

| | off | off+b4096 | 4 | 4/direct | auto/auto (as run) |
|---|---:|---:|---:|---:|---:|
| rubric coverage | 0.77 | 0.85 | **0.94** | 0.84 | 0.73 |
| correctness | 7.4 | 7.0 | **7.6** | 6.7 | 5.7 |
| completeness | 7.7 | 8.0 | **9.0** | 7.3 | 6.1 |
| overall | 7.1 | 6.7 | **8.0** | 6.7 | 5.6 |
| cut off / empty / looping | 2 of 7 | 2 of 7 | 0 of 7 | 1 of 7 | 2 of 7 |

- **Notes + one synthesis (`4`) answered 2.7× sooner than the ordinary request (geometric mean) and was graded
  best.** Even with a 32K context the ordinary request thinks for minutes on these prompts (8-16K tokens on four of
  them; two ran into the context and gave no answer or looped).
- Sections written directly (`4/direct`) were fastest (4.4×) but contradicted each other (two sections, two different
  matching windows) and made arithmetic slips: graded below the ordinary request. Not used by AUTO without a large
  document.
- The figure-heavy NPV/IRR task was the exception: the ordinary request computed every figure right (9/9, after
  5.8 minutes); every parallel form got the decision and NPVs right but some IRRs wrong (6/9).
- `auto/auto` as run had two failures since fixed: a plan with 5 subtasks where 4 were allowed was rejected and the
  request fell back to an ordinary request that thought until the context was full (now: one retry, told what was
  wrong); and it then wrote sections directly (now: notes + synthesis).

### Long documents: the scaling curve

Seeded documents with known answers (`tools/longctx_tasks.py`): vendor contracts (which may be terminated on short
notice, their total fees, which caps are below fees), business-unit reports (the total over every unit, the best
margin, the units that fell), a configuration reference (a value one component defines from another's parameter).
Score = expected names and figures found. Prompt tokens measured by the server.

| task | prompt tokens | off | off+b4096 | 3 | 3/direct | auto/auto |
|---|---:|---:|---:|---:|---:|---:|
| contracts-8k | 7913 | 51 (6/6) | 39 (6/6) | 54 (6/6) | 37 (6/6) | 56 (6/6)¹ |
| contracts-16k | 15733 | 60 (6/6) | 80 (6/6) | 68 (6/6) | 53 (4/6) | 52 (5/6) |
| reports-8k | 8109 | 168 (5/5) | 106 (5/5) | 92 (4/5) | 43 (4/5) | 38 (4/5) |
| reports-16k | 16079 | 296 (5/5) | 121 (3/5) | 155 (4/5) | 51 (4/5) | 48 (4/5) |
| config-8k | 8136 | 121 (3/3) | 36 (3/3) | 54 (3/3) | 33 (3/3) | 60 (3/3)¹ |
| config-16k | 16044 | 63 (3/3) | 59 (3/3) | 65 (3/3) | 48 (3/3) | 80 (3/3)¹ |
| contracts-32k | 31951 | 128 (6/6) | 111 (6/6) | 92 (6/6) | 76 (6/6) | 181 (6/6)¹ |
| contracts-64k | 63736 | 242 (6/6) | 194 (6/6) | 1154 (1/6) | 139 (1/6) | 149 (2/6) |
| contracts-120k | 119639 | 439 (6/6) | 302 (6/6) | 244 (6/6) | 236 (2/6) | 253 (3/6) |
| reports-64k | 63851 | 1467 (0/5) | 200 (3/5) | 283 (4/5) | 134 (1/5) | 134 (1/5) |
| reports-120k | 119609 | 440 (0/5) | 302 (3/5) | 246 (3/5) | 224 (2/5) | 234 (1/5) |
| config-64k | 63974 | 125 (3/3) | 143 (3/3) | 146 (3/3) | 126 (3/3) | 123 (3/3) |
| config-120k | 120313 | 241 (3/3) | 251 (3/3) | 247 (3/3) | 232 (3/3) | 228 (3/3) |

32K documents on the 64K server (later runs; "3" there already ran under `compose` "auto", i.e. as sections):
reports-32k off 616 s (5/5, 26,885 tokens of thinking), off+b4096 139 s (3/5), 3 → sections 76 s (4/5), 3/direct
76 s (3/5), final AUTO 604 s (5/5, one stream); config-32k 78 / 98 / 70 / 78 / 69 s, all 3/3.

Repeats (3 cold runs each; the random first line changes how long the ordinary request thinks):

| task | off | off+b4096 | notes + synthesis | sections (direct) |
|---|---|---|---|---|
| contracts-120k | 439 s 6/6; 441 s and 447 s **no answer** (thinking filled the context) | 302 / 294 / 306 s, 6/6 each | 244 / 247 / 262 s; 6, 5, 4 of 6 | |
| contracts-8k | 51 / 50 / 41 s, 6/6 | | | 37 / 48 / 40 s; 6, 4, 4 of 6 |
| config-8k | 121 / 53 / 32 s, 3/3 | | | 33 / 34 / 34 s, 3/3 |
| config-16k | 63 / 75 / 126 s, 3/3 | | | 48 / 52 / 48 s, 3/3 |

How the time is spent (the shared regime, every step restoring the planner's read):

| | 8K | 16K | 32K | 64K | 120K |
|---|---:|---:|---:|---:|---:|
| planning: the one read of the document + the plan (~200 tokens) | 16 s | 28 s | 53 s | 104 s | 202 s |
| prompt reading rate (ordinary request, to its first token) | 600-630 tok/s at every length | | | | |
| each subtask's admission (restore, not re-read) | ~1 s | ~1 s | ~1.3 s | ~1.4 s | ~1.5-2 s |
| subtasks (3, side by side) | ~20 s | ~20 s | ~17 s | ~21 s | ~22 s |
| synthesis (notes; 192 thinking tokens) | ~18 s | ~18 s | ~22 s | ~18 s* | ~20 s |

(* one 64K synthesis looped for 51,180 tokens: the synthesis is now capped at `synthesis_max_tokens`, 12,288.)

What the curve shows:

- **Reading dominates at length** and nothing parallel makes it faster: the engine reads one prompt at a time, so
  every mode pays the same ~200 s at 120K. Task-parallel saves what comes after the read.
- **Where the ordinary request thinks at length, parallel steps pay**: contracts-120k 439 s → 244 s at the same
  score (6/6); the business-unit reports at 64K made the ordinary request think until the context was full (24
  minutes, no answer), task-parallel answered in 134-283 s.
- **Where it does not, they do not**: the configuration lookups are answered in one short pass by the ordinary
  request (125 s at 64K); parallel steps add only their own overhead.
- **Accuracy drops with length for parallel writers without thinking.** Totals over 38-570 units and "every
  contract with a short notice" over 170 contracts need one careful pass: at 64K and 120K the directly written
  sections missed figures (1-3 of 6) that the ordinary request found. Section writers with a thinking budget
  (1,024 / 3,072 tokens; measured at 8-16K) recovered some accuracy but lost the speed: a batch slot decodes without
  MTP drafts (~20-30 tok/s per slot against ~45 tok/s alone), so thinking in parallel slots is not faster than one
  request thinking alone.
- **Partitioning (slices of the document per subtask) did not pay** even before accuracy: N slices read one after
  another cost what the whole costs, a slot that is decoding pauses while another slice is read, and the notes then
  need a synthesis (measured with 6 slots of 16K over the 32K contracts: 147-201 s for 2-6 subtasks against 161 s for the ordinary
  request on the same server; those runs predate the cold-run fix, so later modes may have reused a little). It remains for
  material larger than a slot can hold.

AUTO after this pass (`compose` "auto", the default):

- without a large document: notes + one synthesis (the blind-graded best form above);
- with a large document that fits a slot: the planner reads it once; a "low"-effort request (a lookup) gets its
  sections written in parallel; a "medium"/"high" request is answered by **one stream that restores what the
  planner read** (quality of the ordinary request, no second read);
- trivial requests never reach the planner (12/12 here, 1.5-19 s).

Measured again on the same tasks:

| | short complex requests (10) | long documents |
|---|---|---|
| decision | notes + synthesis on the 7 it split; 3 answered as usual (planner: "low" effort or no independent parts) | lookups ("low"): sections; the rest: one stream over the planner's read |
| wall-clock vs ordinary request | geometric mean **0.50** (0.29 on the 7 it split); median 78 s vs 181 s | contracts-8k 41 vs 45-51 s, config-16k 48 vs 63-126 s; contracts-16k 70 vs 60 s, reports-16k 300 vs 296 s, contracts-120k 427 vs 439 s, reports-120k 437 vs 440 s (both: no answer, the context filled) |
| blind quality (7 rubric tasks; second grader, answers shuffled) | overall **8.1** (ordinary 6.4, notes + synthesis with 4: 8.0); 0 of 7 cut (ordinary: 3) | same scores as the ordinary request (one stream) or equal scores (lookups) |

So AUTO is **faster on complex short requests at better graded quality**, faster on long-document lookups at equal
scores, and **no faster on hard long-document requests**: there the remaining choice is the caller's - `3` (notes
+ one synthesis) answered contracts-120k in 244-262 s (6/6, 5/6, 4/6) where the ordinary request needed 439 s or
gave no answer, and `off` with a thinking budget gave 6/6 in ~300 s.

Memory (`context_provider` pointed at a stand-in JSON endpoint kept with the evidence, five items: a
pinned policy, a supplier list, a live budget, an irrelevant note, a glossary; the request: the 32K contracts plus
three questions only memory answers): one retrieval per request (0-4 ms, 5 items, 204 tokens); `3` (here: sections
over the shared document, 89.6 s) and
AUTO (132 s) used the policy, the list and the live budget correctly (7/7); the ordinary request without memory
(172 s) missed the budget question (6/7). With the endpoint unreachable (a port in use: HTTP 404) every request still answered,
reporting `provider.error` - memory augments, it never blocks.


## Benchmarks (v1: 4K context)

Measured on one deployment; the numbers describe that deployment, not the feature in general.

**Setup.** Swift 1.5 Qwen3.8 Flash-Next IQ4_XS with MTP speculative decoding (`--spec 4`), split by layers over
the three Arc Pro GPUs (B70 + B65 + B60, `--layer-split 18,34`, every expert in VRAM), served by
`serve/server.py` with the engine's batch slots (`--batch 6`, 3 pipelined groups), context 4096 tokens, greedy
sampling, the server's default reasoning effort. One request alone decodes at ~38-48 tok/s; six concurrent requests
reach ~113 tok/s aggregate on the engine (~72 end-to-end through the server). Every request was streamed; wall-clock
is from sending the request to the end of the answer. One run per cell (greedy).

**Modes compared** (the same prompts):

- `off`: the ordinary request, the server's defaults (it thinks as long as it needs).
- `off+b1536` / `off+b192`: the ordinary request with `reasoning_budget_tokens` 1536 / 192 (192 is what the synthesis
  step gets - the compute-matched baseline for the final writer).
- `2`, `4`, `6`: task-parallel with that many subtasks. `auto`: AUTO as first implemented; `auto-final`: AUTO after
  calibration (inline enumerations pass the gate; only "low" effort stays at 1).

### Wall-clock, 10 decomposable tasks (seconds; ᵗ = the answer ran into the 4096-token context and was cut)

| task | off | off+b1536 | off+b192 | 2 | 4 | 6 | auto | auto-final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| code review: find + fix bugs | 95.7ᵗ no answer | 54.2 | 28.8 | 48.5 | 52.4 | 46.9 | 54.1 | 104.0ᵗ no answer (1) |
| code: three functions | 33.8 | 31.0 | 25.1 | 57.1 | 59.2 | 51.3 | 59.6 | 41.3 (1) |
| architecture design | 110.1ᵗ no answer | 99.5ᵗ | 100.4ᵗ | 86.8 | 97.3ᵗ | 90.7ᵗ | 91.8ᵗ | 99.0ᵗ |
| database comparison | 106.8ᵗ | 103.4ᵗ | 102.6ᵗ | 88.2 | 83.1 | 80.8 | 86.7 | 83.7 |
| NPV / IRR / capital budget | 84.4ᵗ no answer | 80.6ᵗ | 70.1 | 66.2 | 80.1ᵗ | 76.1 | 70.4 | 80.3 |
| four finance questions | 84.0ᵗ no answer | 42.9 | 20.5 | 51.7 | 54.4 | 50.2 | 54.8 | 92.3ᵗ no answer (1) |
| research synthesis (2008 crisis) | 111.6ᵗ | 108.2ᵗ | 109.4ᵗ | 78.5 | 84.1 | 70.3 | 110.2ᵗ (1) | 66.1 |
| contract review | 96.0ᵗ | 99.3ᵗ | 91.0ᵗ | 72.9 | 78.1 | 65.7 | 62.4 | 69.1 |
| latency hypotheses | 106.1ᵗ no answer | 97.1ᵗ | 100.8ᵗ | 64.4 | 80.4 | 66.7 | 70.1 | 77.2 |
| migration plan | 106.3ᵗ | 104.8ᵗ | 100.6ᵗ | 80.6 | 93.1 | 94.3 | 94.6 | 84.4 |
| **median** | 101.1 | 98.2 | 95.7 | 69.6 | 80.3 | 68.5 | 70.3 | 82.0 |
| **geometric mean vs `off`** | 1.00 | 0.85 | 0.71 | 0.76 | 0.84 | 0.75 | 0.82 | 0.87 |
| **geometric mean vs `off+b192`** | 1.41 | 1.20 | 1.00 | 1.07 | 1.17 | 1.06 | 1.15 | 1.22 |
| answers cut by the context | 9 | 7 | 6 | 0 | 2 | 1 | 2 | 3 |

(1) = AUTO answered with the ordinary request.

- Against the ordinary request with default settings, task-parallel finished **25 % sooner** (6 subtasks, geometric
  mean; median 101 → 69 s), 37-39 % on the best tasks, and gave an answer where the ordinary request gave none
  (5 of 10 tasks: its thinking filled the 4096-token context).
- Against the compute-matched ordinary request (`off+b192`) it was **6 % slower overall**: about 2× slower on the
  three short-answer tasks (code, finance questions: 20-29 s vs 47-60 s), and faster on the seven long-answer tasks,
  where `off+b192` wrote until the context was full (5 of 7 cut) while the synthesis wrote a shorter, complete answer.
- 4 subtasks were slower than 2 or 6 here: their notes led to longer final answers (the synthesis dominates).

### Final-answer quality (blind)

The answers of the seven rubric tasks were shuffled under random ids and graded blind (by an LLM grader, one sheet
per task group; each rubric item 0 / 0.5 / 1, scores 1-10); the other three tasks were checked automatically.

| | off | off+b1536 | off+b192 | 2 | 4 | 6 | auto | auto-final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rubric coverage | 0.35 | 0.60 | 0.78 | 0.92 | 0.93 | 0.97 | 0.85 | 0.91 |
| overall / 10 | 2.7 | 4.1 | 5.3 | 7.7 | 7.9 | 8.1 | 7.3 | 7.9 |
| correctness / 10 | 5.0 | 7.4 | 7.0 | 7.1 | 7.7 | 7.6 | 7.7 | 7.6 |
| completeness / 10 | 2.7 | 4.3 | 6.0 | 9.0 | 8.6 | 9.1 | 8.1 | 9.0 |
| truncated | 4 | 7 | 5 | 0 | 1 | 1 | 2 | 1 |

Automatic checks: code bug-fix tests 3/3 in every mode that produced code (`off` and `auto-final`: no answer); code
functions 15/15 in every mode; NPV/IRR/payback numbers 7/9 (`off+b1536`, `off+b192`, `2`), 8/9 (`4`, `6`,
`auto-final`); four finance questions 3/3 in every mode that answered.

**Correctness was comparable** (7.0-7.7 in every answering mode); task-parallel answers scored higher overall
because they were **complete**: in this 4096-token context the ordinary request's answers were cut or missing. With
a larger context the ordinary request would finish its answers, and this quality gap would narrow; the wall-clock
of `off` would then grow, not shrink.

### Compute

| mean per request | off | off+b1536 | off+b192 | 2 | 4 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| tokens generated (all steps) | 3,613 | 3,285 | 2,934 | 2,283 | 3,063 | 2,906 |
| prompt tokens read | 218 | 218 | 218 | 1,700 | 2,399 | 2,437 |
| final-answer tokens | 3,613 | 3,285 | 2,934 | 1,589 | 1,779 | 1,660 |
| final answer's decode rate (median) | 39.0 | 39.7 | 40.5 | 39.6 | 40.5 | 42.0 tok/s |
| subtasks' aggregate decode rate (median) | - | - | - | 34.9 | 59.5 | 71.5 tok/s |

Phases (median, 10 tasks): plan 4.7-6.6 s; subtask phase 15.8-19.4 s (each later subtask admitted ~0.9-1.1 s after
the previous one: the sixth starts ~4.5 s after the first; 86 of 160 subtasks used their whole output budget);
synthesis 44-56 s, most of it writing the answer.

### AUTO

30 requests: the 10 decomposable tasks above, 6 trivial ones, 8 held out and 6 written after calibration (17
decomposable, 13 not):

- **13 of 13 non-decomposable requests stayed at 1**, decided by the gate alone (no planner call, no added latency):
  arithmetic, a rewrite, a joke, a capital, a translation, short explanations, TCP vs UDP, a palindrome function, a 3-sentence
  summary, a two-sentence email, a primality check, a sequential tank-volume calculation (one chain of dependent
  steps), a definition.
- 14 of 17 decomposable requests were split (3-6 subtasks); on the 7 split requests outside the calibration set it
  finished 12-39 % sooner than the ordinary request (95.6 vs 109.2, 68.7 vs 97.2, 71.0 vs 102.4, 79.5 vs 103.8,
  74.9 vs 105.1, 78.9 vs 113.9, 58.2 vs 94.8 s; the ordinary request gave no answer in 5 of these 7), graded 7-9 /10
  against 1-8 /10.
- 3 stayed at 1: the planner rated two "low" effort (the three code functions - correctly, task-parallel was slower
  there; the four finance questions) and found "no independent parts" in the bug-fix review. On the last two the
  ordinary request then gave no answer within the context, where task-parallel would have answered in ~50 s:
  AUTO's choice is right against a request with a short thinking budget and wrong against this deployment's
  defaults.

### Feature off

With `task_parallel` absent the server's outputs were byte-identical to the previous server's on the regression
set (six solo and six concurrent requests, a follow-up turn, a cancelled stream), and the engine's tokens were
identical to the previous engine build's (five requests).

## Privacy

Subtask outputs stay inside the request: they are passed to the synthesis and then dropped; they are not returned,
logged as text or stored. Any internal request's reasoning is discarded unread. The response carries only timings,
counts and the decision; the subtasks' one-line objectives only with `"task_parallel_diagnostics": true`. The
internal requests are ordinary requests to the same server process and model: nothing leaves it.

## Limitations

- **It does not make reading faster.** The engine reads one prompt at a time; with a long document every mode
  pays the same read (~200 s at 120K here). Task-parallel shortens what comes after it.
- **Parallel steps decode without MTP drafts** (a batch slot emits one token per window: ~20-30 tok/s per slot
  against ~45 tok/s for a request alone), so work that needs long careful reasoning is not faster split across
  slots; and sections written without thinking lose accuracy on long aggregations (totals over dozens of items).
  AUTO therefore answers medium/high-effort requests over a large document in one stream.
- The answer starts later than an ordinary request's first token when that request would think briefly
  (plan + subtasks + synthesis ≈ 40-70 s here); AUTO's gate and effort rule avoid most such requests, not all.
- The planner is the same model: its split can be uneven or miss parts, and its effort rating decides AUTO's form.
  An unusable plan is retried once with the error, then the request is answered as usual.
- A subtask's prompt read pauses the slots that are decoding (this engine reads one prompt at a time and does not
  interleave reads with decoding). With a shared document the admissions are restores (~1-2 s each), so this costs
  little; with partitioned material it is part of why partitioning did not pay.
- Full-size slots cost VRAM: on the three-card Arc deployment (B70 + B65 + B60) 4 at 32K, 3 at 64K, 4 at 128K (8-bit KV) fit beside every
  expert; `--slot-context` trades slot size for more slots.
- More compute per request (reported in every response); on a busy server AUTO uses only the free slots and
  answers normally when fewer than two are free.
- Chat completions without tools, structured output or images.
