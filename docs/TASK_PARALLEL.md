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

## Engine requirements

Any engine behind `serve/server.py` works: the orchestration (`serve/task_parallel.py`) only issues ordinary requests
through a small `Backend` interface (run one request; how many can run at once) and knows nothing about GPUs or
engine builds. To gain wall-clock the engine must run requests concurrently (Strata: `--batch N`, with a layer split
also `--batch-groups G`); with one request at a time AUTO always answers normally. The shared-prefix checkpoint
(`ckpt=`, `INFO prefix_ckpt=1`) is an optimization; without it every internal request reads the shared context again.

## Latency vs compute

Where the time goes (one request; medians over 10 decomposable tasks, tri-GPU deployment below):

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

## Benchmarks

Measured on one deployment; the numbers describe that deployment, not the feature in general.

**Setup.** One model (a 35B-class MoE, IQ4_XS, MTP speculative decoding) split by layers over three GPUs, served by
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

- The answer starts later (31-43 s here): a request that one ordinary request answers well with little thinking is
  answered sooner without task-parallel. AUTO's gate and effort rule avoid most such requests, not all.
- The final answer is written by one stream; on long-answer tasks the synthesis dominates (45-70 % of the
  wall-clock). Splitting the writing itself (sections written in parallel after an outline) is not implemented.
- Subtasks are admitted one after another (~0.9-1.1 s each here, while the running ones pause): the last of six
  starts ~4.5 s after the first. Reading new prompts beside decoding slots would remove most of that (upstream
  Strata has work in this direction); it is not part of this version.
- The planner is the same model: its split can be uneven or miss parts; the synthesis is told to cover what is
  missing, but cannot recover work no subtask did. Subtask budgets are fixed shares; about half the subtasks used
  their whole budget.
- More prompt processing and slot occupancy per request: on a busy server task-parallel requests compete with other
  users' requests for the same slots (subtasks are capped at the engine's slot count; AUTO does not yet look at
  the current load).
- v1: chat completions without tools, structured output or images.
