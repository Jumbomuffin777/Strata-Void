"""Task-parallel requests: one request decomposed into subtasks that run at the same time, then merged.

    request -> [plan] -> subtask 1 .. N (concurrent internal requests) -> [synthesis] -> one answer

What it is: TASK parallelism.  A planner call decides whether the request has independent parts (or would gain from
independent attempts at it), the parts run as ordinary internal requests on the engine's concurrent request
capacity (Strata's batch slots), and a synthesis call writes the one final answer from their work products.  Each
internal request decodes at the normal per-slot rate; what shortens is the wall-clock of a request whose work can be
done side by side.  It is not speculative decoding and it does not make any single token stream faster.

What crosses between the steps is WORK PRODUCT only - findings, calculations, code, a proposed solution - never a
model's reasoning text: an internal request's reasoning is counted and discarded.

The module is engine-agnostic.  It talks to a `Backend` (generate one internal request; how many can run at once),
so the server provides the Strata binding and the tests a fake one.  Nothing here knows about GPUs, cards or a
particular engine build.

Modes (request field `task_parallel`, or the server default): "off" (or absent / 0 / 1 / false: the request runs
exactly as before), "auto" (a cheap heuristic gate, then the planner decides 1 or 2..max), or a number of workers
(2..max: the planner is asked for exactly that many subtasks).
"""
from __future__ import annotations

import json
import re
import threading
import time
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Protocol

from serve import context_pool as cp

ALLOWED_WORKERS = (2, 3, 4, 5, 6, 7, 8)
STRATEGIES = ("partition", "independent", "shard")


# ------------------------------------------------------------------------------------------------ configuration
@dataclass
class Config:
    max_workers: int = 6                 # never more internal requests at once than this (and the engine's slots)
    min_tokens: int = 96                 # a subtask's output budget, clamped to [min_tokens, max_tokens]
    max_tokens: int = 320
    total_tokens: int = 1200             # all subtasks' output budgets together
    worker_reasoning: int = 0            # a worker's thinking budget; 0: workers write their work product without
    #                                      thinking (measured faster at equal quality: a budget's wrap-up re-reads the
    #                                      worker's prompt through the prompt path while the other slots wait)
    planner_tokens: int = 300            # the plan's own budget (no thinking)
    synthesis_reasoning: int = 192       # the synthesis' thinking budget when the request thinks (0: unlimited)
    synthesis_min_tokens: int = 1024     # room kept in the context for the final answer
    synthesis_max_tokens: int = 12288    # the most the synthesis writes (thinking included) when the request sets no
    #                                      max_tokens (0: what the context allows); measured: one looped for 51K tokens
    worker_timeout_s: float = 240.0      # one worker attempt; a timed-out worker is stopped
    total_timeout_s: float = 600.0       # planner + every worker attempt
    retries: int = 1                     # a failed worker (error / empty output) is tried again once
    gate_threshold: int = 2              # AUTO: the heuristic score a request needs before the planner is asked
    auto_skip_effort: str = "low"        # AUTO answers normally when the planner rates the request at an effort listed
    #                                      here (comma-separated; measured: "medium" requests still gained)
    auto_shared_one_effort: str = "medium,high"   # AUTO, large material in the shared prefix: these planner
    #                                      ratings are answered by one stream over what the planner read (see run())
    internal_effort: str = "low"         # the reasoning effort every internal request is rendered with ("": the
    #                                      template's default) - one setting for all, so their shared prefix matches
    # ---- large context (v2): material above partition_min_tokens is cut into chunks; the planner reads an index of
    # them, each subtask reads only the chunks it is given (plus pinned items), never the whole material
    partition_min_tokens: int = 6000     # a request's material (long messages, context items) above this is partitioned
    share_context: int = -1              # ... unless it fits here: material up to this many tokens stays whole in the
    #                                      shared prefix - read ONCE (by the planner) and restored, not re-read, by
    #                                      every later step, so each step sees all of it.  -1: what one batch slot
    #                                      holds beside a step's own text (--slot-context); 0: off
    chunk_tokens: int = 1500             # target chunk size (cut at the material's own structure)
    worker_source_tokens: int = 24000    # the most material one subtask reads (also bounded by the engine's context)
    min_shard_tokens: int = 2000         # AUTO: fewer subtasks rather than slices thinner than this
    pinned_tokens: int = 1500            # items every step sees (pinned, durable memory) - the rest are assigned
    need_per_worker: int = 2             # a subtask may ask for material it was not given: NEED lines, at most this many
    need_total: int = 6                  # ... and this many in all; one follow-up round, never recursive
    need_tokens: int = 3000              # the material one follow-up reads
    provider_k: int = 8                  # a ContextProvider's items for one request (one call per parent request)
    provider_tokens: int = 3000
    # ---- the answer (v2): "synthesis" = one call writes it from the notes (v1); "sections" = an outline call, then
    # the sections written at the same time and assembled in order; "direct" = the planner designs the answer's
    # sections and the subtasks write them (no notes, no outline: one parallel stage, plus closing sections that
    # read the others); "auto" = "direct" over a large material kept whole in the shared prefix (each writer sees all
    # of it; AUTO answers a medium/high-effort request there in one pass instead), else notes + one synthesis (the
    # blind-graded best form without large material: sections written without notes contradicted each other)
    compose: str = "auto"
    writer_reasoning: int = 0            # a section writer's thinking budget (0: none)
    section_tokens_per_word: float = 2.2  # a section's output budget per planned word (tables and figures run long:
    #                                       1.6 cut list-heavy sections short)
    outline_tokens: int = 800            # the outline's own budget (JSON: sections + the facts they must agree on)
    outline_reasoning: int = 192         # the outline's thinking budget when the request thinks
    max_sections: int = 6
    section_min_words: int = 120
    compose_min_words: int = 500         # (v2 drafts: "auto" wrote sections above this; "auto" now never writes sections)
    # ---- AUTO's cost model (seconds / tokens per second of this deployment; see docs/TASK_PARALLEL.md)
    prefill_tok_s: float = 600.0         # prompt reading (fast IQ4 dequant; 300 before)
    decode_tok_s: float = 38.0           # one request alone
    slot_tok_s: str = "2:30,3:26,4:23,6:19"   # one request among N in the batch slots
    admission_s: float = 0.9             # a subtask's admission (shared prefix restored, its own tokens read)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Config":
        """The server config's "task_parallel" object (every key optional; "default" is the server-wide mode and is
        read by the caller, not here)."""
        c = cls()
        for k, v in (d or {}).items():
            if k == "default":
                continue
            if not hasattr(c, k):
                raise ValueError(f"task_parallel config: unknown key {k!r}")
            cur = getattr(c, k)
            if isinstance(cur, str):
                if not isinstance(v, str):
                    raise ValueError(f"task_parallel config: {k} must be a string")
                setattr(c, k, v)
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(cur, int) and not isinstance(v, int)):
                raise ValueError(f"task_parallel config: {k} must be a {type(cur).__name__}")
            setattr(c, k, type(cur)(v))
        if not 1 <= c.max_workers <= 8 or c.min_tokens < 1 or c.max_tokens < c.min_tokens or c.retries < 0:
            raise ValueError("task_parallel config: bad worker limits")
        if c.internal_effort not in ("", "low", "medium", "high", "xhigh"):
            raise ValueError("task_parallel config: internal_effort must be \"\", low, medium, high or xhigh")
        if c.compose not in ("synthesis", "sections", "direct", "auto"):
            raise ValueError("task_parallel config: compose must be synthesis, sections, direct or auto")
        if not 2 <= c.max_sections <= 8 or c.chunk_tokens < 64 or c.partition_min_tokens < c.chunk_tokens:
            raise ValueError("task_parallel config: bad section or chunk limits")
        c.slot_rates()                   # validates slot_tok_s
        return c

    def slot_rates(self) -> dict:
        """slot_tok_s -> {active requests: tokens/s each}."""
        out = {}
        try:
            for part in self.slot_tok_s.split(","):
                k, v = part.split(":")
                out[int(k)] = float(v)
        except ValueError:
            raise ValueError("task_parallel config: slot_tok_s is \"N:rate,...\"") from None
        return out

    def slot_rate(self, n: int) -> float:
        r = self.slot_rates()
        if not r:
            return self.decode_tok_s
        keys = sorted(r)
        below = [k for k in keys if k <= n]
        return r[below[-1]] if below else r[keys[0]]


@dataclass(frozen=True)
class Mode:
    kind: str                            # "off" | "auto" | "fixed"
    workers: int = 1

    @property
    def off(self) -> bool:
        return self.kind == "off"


def parse_mode(value) -> Mode:
    """The request's `task_parallel` -> Mode.  Absent, None, False, "off", 0 and 1 are OFF (the request runs exactly
    as without the feature).  A bad value is a ValueError (a 400)."""
    if value is None or value is False:
        return Mode("off")
    if value is True:
        return Mode("auto")
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("", "off", "none", "false", "0", "1"):
            return Mode("off")
        if v == "auto":
            return Mode("auto")
        if v.isdigit():
            value = int(v)
        else:
            raise ValueError(f'task_parallel={value!r}: use "off", "auto" or a number of workers (2..8)')
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return Mode("off")
        if value in ALLOWED_WORKERS:
            return Mode("fixed", value)
    raise ValueError(f'task_parallel={value!r}: use "off", "auto" or a number of workers (2..8)')


# ------------------------------------------------------------------------------------------------ the backend
@dataclass
class Call:
    role: str                            # "planner" | "worker" | "synthesis"
    messages: list
    max_tokens: int
    thinking: str = "off"                # "off" | "on"
    reasoning_budget: int = 0            # 0: none
    shared_prefix: list | None = None    # the messages every internal request starts with (a cache hint)
    common_with: list | None = None      # a sibling call's messages: the prompt both share is a cache hint too
    on_text: Callable[[str], None] | None = None   # streamed content (the answer's sections)


@dataclass
class Result:
    text: str = ""                       # the content (the work product); reasoning text is never kept
    finish: str = ""
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    prompt_tokens: int = 0
    reused_tokens: int = 0
    t_start: float = 0.0
    t_first: float | None = None
    t_end: float = 0.0
    error: str | None = None
    attempts: int = 1


class Backend(Protocol):
    def concurrency(self) -> int: ...                       # internal requests the engine runs at once (1: serial)
    def context(self) -> int: ...                           # the engine's context (tokens)
    def count_tokens(self, call: Call) -> int: ...          # the prompt length of a call
    def generate(self, call: Call, cancel: threading.Event) -> Result: ...
    # optional: count_text(text) -> tokens (else an estimate); busy() -> slots in use by other requests;
    # slot_context() -> the context of one batch slot when smaller than context() (an internal request longer than
    # that runs alone, on the solo path, so the subtasks' sources are kept within it)


# ------------------------------------------------------------------------------------------------ plans
@dataclass
class Subtask:
    id: int
    objective: str
    expected_output: str
    max_tokens: int
    sources: list = field(default_factory=list)   # item ids the planner named ("all": every item)
    heading: str = ""                    # compose "direct": the answer section this subtask writes
    words: int = 0
    after: bool = False                  # ... written once the other sections are (a summary, a recommendation)


@dataclass
class Plan:
    parallelism: int
    strategy: str = "partition"
    reason: str = ""
    subtasks: list = field(default_factory=list)
    effort: str = ""                     # the planner's estimate of the work a careful single answer needs


class PlanError(ValueError):
    pass


def _json_object(text: str) -> dict:
    """The one JSON object in a model's answer (code fences and text around it tolerated)."""
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if m:
        t = m.group(1)
    a, b = t.find("{"), t.rfind("}")
    if a < 0:
        raise PlanError("the plan is not a JSON object")
    if b <= a:
        obj = _truncated_plan(t[a:])
        if obj is None:
            raise PlanError("the plan is not a JSON object")
        return obj
    try:
        obj = json.loads(t[a:b + 1])
    except json.JSONDecodeError as e:
        obj = _truncated_plan(t[a:])
        if obj is None:
            raise PlanError(f"the plan is not valid JSON: {e.msg}") from e
    if not isinstance(obj, dict):
        raise PlanError("the plan is not a JSON object")
    return obj


def _truncated_plan(t: str) -> dict | None:
    """A plan (or an outline) cut off by its token budget inside its list: the entries complete so far, the list
    closed (an outline cut inside its "facts" keeps its sections and the facts complete so far).  None when nothing
    usable is left (the caller reports the original error)."""
    key = next((x for x in ("subtasks", "sections") if f'"{x}"' in t), None)
    if key is None:
        return None
    k = t.find(f'"{key}"')
    for end in range(len(t) - 1, k, -1):
        if t[end] not in "}\"":                 # the end of a whole subtask (an object or a string)
            continue
        for tail in ("]}",):
            try:
                obj = json.loads(t[:end + 1] + tail)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get(key), list) and obj[key]:
                obj["_truncated"] = True
                if isinstance(obj.get("parallelism"), int):
                    obj["parallelism"] = min(obj["parallelism"], len(obj[key]))
                return obj
    return None


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def parse_plan(text: str, cfg: Config, max_workers: int, exact: int | None = None,
               token_room: int | None = None, known_ids: set | None = None) -> Plan:
    """Validate the planner's answer.  `exact`: the number of subtasks the request asked for.  `token_room`: the
    most tokens all subtask outputs may take together (the synthesis must still fit the context).  Raises PlanError
    for anything malformed; never invents subtasks."""
    obj = _json_object(text)
    p = obj.get("parallelism")
    if isinstance(p, float) and p.is_integer():
        p = int(p)
    if not isinstance(p, int) or isinstance(p, bool) or p < 1:
        raise PlanError("parallelism must be a positive whole number")
    effort = str(obj.get("effort") or "").strip().lower()
    effort = effort if effort in ("low", "medium", "high") else ""
    if p == 1:
        if exact:
            raise PlanError(f"the planner chose 1 subtask where {exact} were asked for")
        return Plan(1, reason=str(obj.get("reason") or "")[:300], effort=effort)
    if p > max_workers:
        raise PlanError(f"parallelism {p} exceeds the limit {max_workers}")
    if exact and p != exact:
        raise PlanError(f"parallelism {p} where exactly {exact} were asked for")
    strategy = str(obj.get("strategy") or "partition").strip().lower()
    if strategy not in STRATEGIES:
        raise PlanError(f"unknown strategy {strategy!r}")
    raw = obj.get("subtasks")
    if strategy == "shard" and isinstance(raw, list) and len(raw) == 1 and p > 1:
        raw = [raw[0]] * p               # one objective over p slices of the material
    if isinstance(raw, list) and len(raw) != p and not exact and 2 <= len(raw) <= max_workers:
        p = len(raw)                     # the list is the plan; its count field was off (measured: 3 said, 4 listed)
    elif isinstance(raw, list) and exact and len(raw) > exact and p == exact:
        raw = raw[:exact]
    if not isinstance(raw, list) or len(raw) != p:
        raise PlanError(f"{p} subtasks expected, got {len(raw) if isinstance(raw, list) else 'none'}")
    subtasks, seen = [], set()
    share = max(cfg.min_tokens, min(cfg.max_tokens, cfg.total_tokens // p))   # a subtask without its own budget
    for i, s in enumerate(raw, 1):
        if isinstance(s, str):
            s = {"objective": s}
        if not isinstance(s, dict):
            raise PlanError("a subtask is not an object")
        obj_text = str(s.get("objective") or "").strip()
        if len(obj_text) < 8:
            raise PlanError(f"subtask {i} has no objective")
        if len(obj_text) > 800:
            raise PlanError(f"subtask {i}'s objective is too long")
        key = _norm(obj_text)
        if key in seen and strategy != "shard":
            raise PlanError(f"subtask {i} duplicates another")
        seen.add(key)
        mt = s.get("max_tokens", share)
        if isinstance(mt, float) and mt.is_integer():
            mt = int(mt)
        if not isinstance(mt, int) or isinstance(mt, bool):
            raise PlanError(f"subtask {i}'s max_tokens is not a number")
        mt = max(cfg.min_tokens, min(cfg.max_tokens, mt))
        expected = str(s.get("expected_output") or "").strip()[:400] or "Concise findings for this subtask."
        src = s.get("sources") or []
        if isinstance(src, str):
            src = [src]
        if not isinstance(src, list):
            raise PlanError(f"subtask {i}'s sources are not a list")
        src = [str(x).strip().strip("[]") for x in src][:200]
        if known_ids is not None:
            src = ["all"] if "all" in src else [x for x in src if x in known_ids]   # unknown ids are dropped
        head = re.sub(r"\s+", " ", str(s.get("heading") or "")).strip().strip("#").strip()[:120]
        w = s.get("words", 0)
        w = int(w) if isinstance(w, (int, float)) and not isinstance(w, bool) else 0
        subtasks.append(Subtask(i, obj_text, expected, mt, src, head, max(0, min(1200, w)), s.get("after") is True))
    room = min(cfg.total_tokens, token_room) if token_room is not None else cfg.total_tokens
    total = sum(s.max_tokens for s in subtasks)
    if total > room:                     # scale the budgets down to the room, keeping their proportions
        if room < cfg.min_tokens * p:
            raise PlanError("not enough context left for the subtasks and the synthesis")
        for s in subtasks:
            s.max_tokens = max(cfg.min_tokens, int(s.max_tokens * room / total))
    return Plan(p, strategy, str(obj.get("reason") or "")[:300], subtasks, effort)


# ------------------------------------------------------------------------------------------------ AUTO's gate
_SPLIT_WORDS = ("compare", "comparison", "analy", "design", "architect", "review", "debug", "evaluate", "assess",
                "trade-off", "tradeoff", "pros and cons", "risk", "strateg", "plan ", "options", "alternatives",
                "research", "investigate", "audit", "each of", "for each", "dimension", "in detail", "breakdown",
                "break down", "step by step", "bugs", "issues", "recommend")
_SIMPLE_START = ("rewrite", "rephrase", "translate", "tell me a joke", "write a joke", "hi", "hello", "thanks",
                 "thank you", "fix the typo", "summarize this sentence")


def gate_score(text: str) -> tuple[int, list[str]]:
    """AUTO's first, model-free decision: is the request worth asking the planner about?  Transparent signals,
    each one listed in the reasons."""
    t = (text or "").strip()
    low = t.lower()
    score, why = 0, []
    n = len(t)
    if n < 160:
        score -= 3; why.append("short")
    elif n >= 1500:
        score += 2; why.append("long")
    elif n >= 600:
        score += 1; why.append("medium length")
    items = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)]|[a-z][.)])\s+\S", t))
    items = max(items, len(re.findall(r"(?:^|\s)\(\d+\)\s", t)))             # inline (1) ... (2) ... (3)
    if items >= 3:
        score += 2; why.append(f"{items} listed items")
    else:
        # an enumeration inside one sentence ("cover A, B, C, and D"): the most comma/semicolon-separated parts
        # in a sentence, parenthesized asides not counted
        flat = re.sub(r"\([^()]*\)", "", t)
        parts = max((len(re.findall(r"[,;]", x)) + 1 for x in re.split(r"[.!?](?:\s|$)|\n", flat)), default=0)
        if parts >= 4:
            score += 2; why.append(f"{parts} enumerated parts")
    q = t.count("?")
    if q >= 3:
        score += 1; why.append(f"{q} questions")
    hits = [w.strip() for w in _SPLIT_WORDS if w in low]
    if hits:
        score += min(3, len(hits)); why.append("asks to " + "/".join(hits[:4]))
    if "```" in t or re.search(r"(?m)^\s*(def|class|function|public|#include|import)\b", t):
        score += 1; why.append("code")
    if low.startswith(_SIMPLE_START):
        score -= 2; why.append("simple edit/chat")
    return score, why


# ------------------------------------------------------------------------------------------------ material (v2)
@dataclass
class Material:
    """The request's context as the steps see it.  Not partitioned: every step reads all of it (v1).  Partitioned:
    every step reads `request_view` (the request with the material cut out), the index and the pinned items; each
    subtask reads only the items it is assigned."""
    request_view: str = ""               # the last user message (material replaced by a pointer when partitioned)
    system_view: str = ""                # the application instructions (likewise)
    pinned: list = field(default_factory=list)
    assignable: list = field(default_factory=list)
    partitioned: bool = False
    total_tokens: int = 0                # the material (chunks + items), in tokens
    provider: dict = field(default_factory=dict)

    @property
    def items(self) -> list:
        return self.pinned + self.assignable

    def meta(self) -> dict:
        return {"partitioned": self.partitioned, "material_tokens": self.total_tokens,
                "items": len(self.items), "pinned_items": len(self.pinned),
                "kinds": {k: sum(1 for i in self.items if i.kind == k) for k in cp.KINDS
                          if any(i.kind == k for i in self.items)},
                **({"provider": self.provider} if self.provider else {})}


def _cut(text: str, ids: list, count) -> str:
    head, tail = cp.head_tail(text, 300, count)
    span = f"{ids[0]}..{ids[-1]}" if len(ids) > 1 else ids[0]
    return (head.rstrip() + f"\n\n[... the material ({count(text):,} tokens) is in the items {span} of the INDEX; "
            "each step is given the items it needs ...]\n\n" + tail.lstrip())


def build_material(messages: list, cfg: Config, count=None, context_items=None, provider=None) -> Material:
    """Collect the request's material: a long last user message or system prompt (cut into chunks), the request's
    `context_items`, and one retrieval from `provider` (if any).  Small material stays where it is (every step sees
    it, as in v1); material above `partition_min_tokens` is partitioned."""
    count = count or cp.approx_tokens
    m = Material()
    last_user = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), None)
    utext = _text(messages[last_user].get("content")) if last_user is not None else ""
    stext = "\n".join(_text(x.get("content")) for x in messages if x.get("role") == "system")
    m.request_view, m.system_view = utext, stext
    sent = cp.items_from_json(context_items, "document", "d", count) if context_items else []
    got = []
    if provider is not None:
        head, tail = cp.head_tail(utext, 200, count)
        query = utext if count(utext) <= 400 else head + "\n" + tail
        t = time.time()
        try:
            got = list(provider.retrieve(query, cfg.provider_k, cfg.provider_tokens) or [])
        except Exception as e:  # noqa: BLE001 - memory augments a request, it never blocks one
            m.provider = {"error": f"{type(e).__name__}: {e}"}
            got = []
        got = cp.pack(got, cfg.provider_tokens)
        m.provider = {**m.provider, "calls": 1, "items": len(got), "tokens": sum(i.tokens for i in got),
                      "ms": int((time.time() - t) * 1000)}
        if getattr(provider, "last_error", ""):
            m.provider["error"] = provider.last_error
    extra = cp.dedupe(sent + got)
    big_user = count(utext) > cfg.partition_min_tokens
    big_sys = count(stext) > cfg.partition_min_tokens
    chunks = []
    if big_sys:
        cs = cp.segment(stext, cfg.chunk_tokens, count, prefix="s")
        chunks += cs
        m.system_view = _cut(stext, [c.id for c in cs], count)
    if big_user:
        cs = cp.segment(utext, cfg.chunk_tokens, count, prefix="c")
        chunks += cs
        m.request_view = _cut(utext, [c.id for c in cs], count)
    m.total_tokens = sum(i.tokens for i in chunks + extra)
    m.partitioned = bool(chunks) or sum(i.tokens for i in extra) > cfg.partition_min_tokens
    if not m.partitioned:
        m.pinned = extra                 # small: every step reads them, as it reads the request
        return m
    # pinned: what the caller pinned, live state (it changes: one copy for all), then the most relevant durable
    # memories within pinned_tokens; everything else is assigned to subtasks
    pin, rest, used = [], [], 0
    for it in sorted(extra, key=lambda i: (not i.pinned, i.kind != "live", i.kind != "memory", -i.score)):
        if (it.pinned or it.kind in ("live", "memory")) and used + it.tokens <= cfg.pinned_tokens:
            pin.append(it)
            used += it.tokens
        else:
            rest.append(it)
    m.pinned = pin
    m.assignable = chunks + sorted(rest, key=lambda i: i.order)
    return m


# ------------------------------------------------------------------------------------------------ prompts
def shared_context(messages: list, material: Material | None = None) -> str:
    """The context every internal request starts with: the application's instructions, the earlier conversation
    and the user's request, as one text (the one system message all internal requests share, so the engine can
    read it once).  With partitioned material: the request with the material cut out, the INDEX of the material and
    the pinned items - each subtask is given its own items after this."""
    sys_parts, convo = [], []
    last_user = None
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user = i
    for i, m in enumerate(messages):
        role, text = m.get("role"), _text(m.get("content"))
        if role == "system":
            sys_parts.append(text)
        elif i == last_user:
            continue
        elif role in ("user", "assistant") and text:
            convo.append(f"{role.upper()}: {text}")
    if material is not None and material.partitioned:
        sys_parts = [material.system_view] if material.system_view else []
    out = [STEP_GUIDE]
    if any(sys_parts):
        out.append("APPLICATION INSTRUCTIONS (they apply to the final answer):\n" + "\n".join(p for p in sys_parts if p))
    if convo:
        out.append("EARLIER CONVERSATION:\n" + "\n\n".join(convo))
    req = _text(messages[last_user].get("content")) if last_user is not None else ""
    if material is not None and material.partitioned:
        req = material.request_view
    out.append("USER REQUEST:\n" + req)
    if material is not None and material.pinned:
        out.append("CONTEXT (every step has these; durable memory and live state are labeled - live state may have "
                   "changed since it was read):\n" + "\n\n".join(i.render() for i in material.pinned))
    if material is not None and material.partitioned and material.assignable:
        out.append("INDEX of the material (" + f"{sum(i.tokens for i in material.assignable):,}"
                   + " tokens; a step reads only the items it is given):\n" + cp.index_text(material.assignable))
    return "\n\n".join(out)


STEP_GUIDE = (
    "You are one step of a system that answers a user request in parts worked on at the same time. The request is "
    "below; the next message names your step.\n"
    "- STEP plan: decide how the request splits into subtasks (the step gives the rules).\n"
    "- STEP subtask: do only the subtask named. Return its work product - findings, calculations, code, evidence "
    "or a solution - specific and concise, within the word limit. No introduction, no summary of the whole "
    "request, do not address the user, do not split the work further.\n"
    "- STEP final answer: the subtasks' notes follow; they can be incomplete, overlapping or wrong. Write the final "
    "answer to the USER REQUEST: keep what is correct, check calculations and claims, resolve contradictions by the "
    "stronger reasoning, remove duplication, keep important caveats, cover what they missed, and answer directly "
    "and completely in one coherent response that follows the application instructions. Never mention notes, "
    "subtasks, steps or this process.")


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", None))


def planner_prompt(cfg: Config, max_workers: int, exact: int | None, partitioned: bool = False) -> str:
    if exact:
        n_rule = f"parallelism is exactly {exact}."
    else:
        n_rule = (f"parallelism is 1 to {max_workers}: 1 for a short, simple, conversational or creative request, a "
                  "quick fact, or one chain of dependent steps; else the fewest subtasks covering its independent "
                  "parts.")
    shard = ""
    src = ""
    if partitioned:
        shard = (' "shard": the SAME objective over different parts of the material (review, summarize, extract or '
                 'check all of it) - the material is split evenly, every part read once;')
        src = (' A subtask may list "sources": the INDEX ids it must read (it gets only those, plus related ones; '
               'items no subtask lists are given to the subtask they fit best, so every item is read). The material '
               'itself is not shown to you: plan from the request and the INDEX.')
    return (
        f"STEP plan. Rules: {n_rule} strategy \"partition\": distinct parts (items, components, dimensions, "
        "questions), one per subtask, together covering everything without overlap; \"independent\": one hard "
        "problem, each subtask solves or checks the WHOLE of it from a different angle (solution, alternative, "
        f"review, edge cases);{shard} Each subtask is one short sentence saying exactly what to produce.{src} effort: "
        "how much work one careful answer to the whole request needs - \"low\" (routine, a few minutes for an "
        "expert), \"medium\", \"high\" (extended analysis, design, debugging or calculation across several parts). "
        'Reply with only this JSON: {"effort": "low"|"medium"|"high", "parallelism": <n>, "strategy": '
        + ('"partition"|"independent"|"shard", "subtasks": [{"objective": "...", "sources": ["c1"]}, ...]}'
           if partitioned else '"partition"|"independent", "subtasks": ["...", "..."]}'))


def planner_prompt_direct(cfg: Config, max_workers: int, exact: int | None, partitioned: bool = False) -> str:
    """The plan when the subtasks write the answer itself (compose "direct"): the answer's sections."""
    if exact:
        n_rule = f"exactly {exact} sections."
    else:
        n_rule = (f"1 to {max_workers} sections: 1 for a short, simple, conversational or creative request, a quick "
                  "fact, or one chain of dependent steps; else the fewest sections the answer needs.")
    src = (' "sources": the INDEX ids a section must read (it reads only those, plus related ones; the material itself '
           'is not shown to you: plan from the request and the INDEX).') if partitioned else ""
    return (
        f"STEP plan. Design the final answer to the USER REQUEST as sections that different writers write at the same "
        f"time. Rules: {n_rule} In reading order; each a distinct part of the answer, no overlap, that can be written "
        "without seeing the other sections; a section that must build on the others (an overall summary or "
        "comparison across them, the final recommendation) gets \"after\": true and is written once they are done "
        "- at most two, placed last. \"objective\": exactly what the section says; \"words\": the length it needs."
        f"{src} effort: how much work one careful answer to the whole request needs - \"low\" (routine, a few "
        "minutes for an expert), \"medium\", \"high\" (extended analysis, design, debugging or calculation). Reply "
        'with only this JSON: {"effort": "low"|"medium"|"high", "parallelism": <number of sections>, "strategy": '
        '"partition", "subtasks": [{"heading": "...", "objective": "...", "words": <n>, "after": false'
        + (', "sources": ["c1"]' if partitioned else "") + "}, ...]}")


def _outline_lines(sections: list) -> str:
    return "\n".join(f"{i}. {s['heading']} - {s['covers']} (about {s['words']} words)" for i, s in enumerate(sections, 1))


def direct_writer_prompt(sections: list, k: int, sources: list | None) -> str:
    s = sections[k - 1]
    head = "ANSWER OUTLINE (the final answer is these sections, in this order):\n" + _outline_lines(sections) + "\n\n"
    src = ("SOURCES (the material for your section):\n\n" + "\n\n".join(i.render() for i in sources) + "\n\n"
           ) if sources else ""
    return (head + src + f"STEP write section {k}/{len(sections)}: \"{s['heading']}\". Write only this section of the "
            f"final answer: start with the line '## {s['heading']}', then about {s['words']} words covering: "
            f"{s['covers']}. Work it out carefully (check calculations), state facts precisely"
            + (" and cite the sources you use as [id]" if sources else "")
            + ". The other sections cover the rest - do not repeat them, no introduction to the whole answer. Never "
              "mention sections, writers, steps or this process.")


def closing_prompt(sections: list, k: int, texts: list) -> str:
    s = sections[k - 1]
    body = "\n\n".join(f"## {h}\n{t}" for h, t in texts if t)
    return ("ANSWER OUTLINE (the final answer is these sections, in this order):\n" + _outline_lines(sections)
            + "\n\nTHE SECTIONS ALREADY WRITTEN:\n\n" + body + "\n\n"
            + f"STEP write section {k}/{len(sections)}: \"{s['heading']}\". Write only this closing section: start "
              f"with the line '## {s['heading']}', then about {s['words']} words covering: {s['covers']}. Build on the "
              "sections above - do not repeat them; where they disagree or a figure is wrong, say what is right and "
              "why. Never mention sections, writers, steps or this process.")


def _short(text: str, words: int = 12) -> str:
    w = text.split()
    return " ".join(w[:words]) + ("..." if len(w) > words else "")


def worker_prompt(plan: Plan, st: Subtask, sources: list | None = None, need: bool = False) -> str:
    words = max(50, int(st.max_tokens * 0.7))
    angle = " One angle on the whole problem." if plan.strategy == "independent" else ""
    if plan.strategy == "shard":
        angle = (" (Your part of the material is below; other parts are done by other subtasks and everything is "
                 "combined after: report every relevant fact of YOUR part exactly - names, figures, dates, with their "
                 "[id] - as a compact list; no totals or conclusions about parts you have not seen.)")
    head = ""
    tail = ""
    if sources:
        head = "SOURCES (your part of the material):\n\n" + "\n\n".join(i.render() for i in sources) + "\n\n"
        tail = " Cite the sources you use as [id]."
        if need:
            tail += (" If your subtask needs information that is in another INDEX item, add at the end a line "
                     "'NEED: <what you need>' (at most two) - never guess it.")
    return f"{head}STEP subtask {st.id}/{len(plan.subtasks)}: {st.objective}{angle} Limit: about {words} words.{tail}"


def notes_block(plan: Plan, results: list) -> str:
    parts = []
    for st, r in zip(plan.subtasks, results):
        if r is not None and r.text.strip():
            cut = " (cut short)" if r.finish == "length" else ""
            parts.append(f"=== {st.id}. {st.objective}{cut}\n{r.text.strip()}")
        else:
            parts.append(f"=== {st.id}. {st.objective}\n(did not complete: cover it yourself if the request needs it)")
    kind = "Each note attacks the whole problem from one angle." if plan.strategy == "independent" else \
        "Each note covers one part."
    return kind + " Notes:\n\n" + "\n\n".join(parts)


def synthesis_prompt(plan: Plan, results: list) -> str:
    return "STEP final answer. " + notes_block(plan, results)


def outline_prompt(max_sections: int) -> str:
    return (
        "STEP outline. Plan the final answer to the USER REQUEST from the NOTES above (they can be incomplete, "
        "overlapping or wrong): check them against each other and check their calculations, resolve contradictions "
        "by the stronger evidence, then reply with only this JSON: {\"sections\": [{\"heading\": \"...\", \"covers\": "
        "\"...\", \"words\": <n>}, ...], \"facts\": [\"...\", ...]}. "
        f"2 to {max_sections} sections in reading order, each a distinct part of the answer with no overlap; the "
        "first one answers the request directly when it asks a question or for a recommendation; \"covers\": what "
        "exactly that section says; \"words\": the length it needs (all together about what the request needs). "
        "\"facts\": up to 12 short statements - the key facts, figures, decisions and caveats every section must use "
        "the same way.")


def writer_prompt(outline: dict, k: int) -> tuple[str, str]:
    """(the part every writer shares, this writer's own instruction)."""
    secs = outline["sections"]
    lines = [f"{i}. {s['heading']} - {s['covers']} (about {s['words']} words)" for i, s in enumerate(secs, 1)]
    facts = "\n".join(f"- {f}" for f in outline["facts"]) or "- (none)"
    shared = ("ANSWER OUTLINE (the final answer is these sections, in this order):\n" + "\n".join(lines)
              + "\n\nFACTS (use them as stated; never contradict them):\n" + facts + "\n\n")
    s = secs[k - 1]
    own = (f"STEP write section {k}/{len(secs)}: \"{s['heading']}\". Write only this section of the final answer: "
           f"start with the line '## {s['heading']}', then about {s['words']} words covering: {s['covers']}. The other "
           "sections cover the rest - do not repeat them, no introduction to the whole answer, no closing summary "
           "unless this section is one. Use the notes and the facts; never mention notes, sections, steps or this "
           "process.")
    return shared, own


def parse_outline(text: str, cfg: Config, max_sections: int) -> dict:
    obj = _json_object(text)
    secs = obj.get("sections")
    if not isinstance(secs, list) or not 2 <= len(secs) <= max_sections:
        raise PlanError(f"the outline needs 2 to {max_sections} sections")
    out, seen = [], set()
    for i, s in enumerate(secs, 1):
        if not isinstance(s, dict):
            raise PlanError("a section is not an object")
        h = re.sub(r"\s+", " ", str(s.get("heading") or "")).strip().strip("#").strip()[:120]
        c = re.sub(r"\s+", " ", str(s.get("covers") or "")).strip()[:500]
        if len(h) < 2 or _norm(h) in seen:
            raise PlanError(f"section {i} has no heading or repeats one")
        seen.add(_norm(h))
        w = s.get("words", 250)
        w = int(w) if isinstance(w, (int, float)) and not isinstance(w, bool) else 250
        out.append({"heading": h, "covers": c or h, "words": max(cfg.section_min_words, min(900, w))})
    facts = [re.sub(r"\s+", " ", str(f)).strip()[:300] for f in (obj.get("facts") or []) if str(f).strip()][:12]
    return {"sections": out, "facts": facts}


# ------------------------------------------------------------------------------------------------ orchestration
@dataclass
class Outcome:
    """What the caller does next.  kind "direct": run the original request as usual (meta says why).  kind
    "synthesize": run `synthesis` (a Call) as the request's answer.  kind "composed": the answer is being written
    in sections at the same time; stream `composer.stream()` (text, in order) as the request's answer."""
    kind: str
    synthesis: Call | None = None
    meta: dict = field(default_factory=dict)
    composer: "Composer | None" = None


def _ms(a: float, b: float) -> int:
    return int(round((b - a) * 1000))


def assign_sources(plan: Plan, material: Material, budget: int) -> tuple[list, dict]:
    """Each subtask's items.  "shard": the material split evenly in reading order.  Otherwise the items a subtask
    named (in order, within its budget), then every item no subtask named goes to the subtask whose objective it
    matches best (BM25) that still has room, so the material is read once; "independent" subtasks get the items
    most relevant to their angle instead.  Returns (items per subtask, coverage meta)."""
    items = material.assignable
    n = len(plan.subtasks)
    per: list = [[] for _ in range(n)]
    used = [0] * n
    by_id = {it.id: it for it in items}
    total = sum(it.tokens for it in items)
    if plan.strategy == "shard":
        target = max(1, total // n + 1)
        k = 0
        for it in items:
            if k < n - 1 and used[k] >= target:
                k += 1
            while k < n and used[k] + it.tokens > budget and used[k] > 0:
                k += 1
            if k >= n:
                break
            per[k].append(it)
            used[k] += it.tokens
    else:
        bm = cp.Bm25(items)
        obj_scores = [bm.scores(st.objective + " " + st.expected_output) for st in plan.subtasks]
        for k, st in enumerate(plan.subtasks):
            named = items if "all" in st.sources else [by_id[x] for x in st.sources if x in by_id]
            for it in named:
                if used[k] + it.tokens <= budget:
                    per[k].append(it)
                    used[k] += it.tokens
        if plan.strategy == "independent":
            for k in range(n):
                for i in sorted(range(len(items)), key=lambda i: -obj_scores[k][i]):
                    it = items[i]
                    if it not in per[k] and used[k] + it.tokens <= budget:
                        per[k].append(it)
                        used[k] += it.tokens
        else:
            taken = {it.id for p in per for it in p}
            for i, it in enumerate(items):
                if it.id in taken:
                    continue
                order = sorted(range(n), key=lambda k: (-obj_scores[k][i], used[k]))
                for k in order:
                    if used[k] + it.tokens <= budget:
                        per[k].append(it)
                        used[k] += it.tokens
                        break
    for k in range(n):
        per[k].sort(key=lambda it: it.order)
    read = {it.id for p in per for it in p}
    cov = {"items": len(items), "items_read": len(read),
           "coverage": round(sum(by_id[x].tokens for x in read) / max(1, total), 3),
           "source_tokens": [u for u in used],
           "duplicated_tokens": sum(used) - sum(by_id[x].tokens for x in read)}
    return per, cov


NEED_LINE = re.compile(r"(?im)^\s*NEED:\s*(.+?)\s*$")


def split_needs(text: str, limit: int) -> tuple[str, list]:
    needs = [m.group(1)[:300] for m in NEED_LINE.finditer(text or "")][:limit]
    return NEED_LINE.sub("", text or "").strip(), needs


class Orchestrator:
    def __init__(self, backend: Backend, cfg: Config | None = None,
                 progress: Callable[[str], None] | None = None):
        self.b, self.cfg = backend, cfg or Config()
        self.progress = progress or (lambda s: None)

    def _slot_context(self) -> int:
        """The context an internal request has when it runs beside others (a batch slot)."""
        ctx = int(self.b.context() or 0)
        f = getattr(self.b, "slot_context", None)
        try:
            sc = int(f() or 0) if f else 0
        except Exception:  # noqa: BLE001
            sc = 0
        return min(ctx, sc) if sc > 0 else ctx

    def _busy(self) -> int:
        f = getattr(self.b, "busy", None)
        try:
            return max(0, int(f() or 0)) if f else 0
        except Exception:  # noqa: BLE001
            return 0

    def _count(self, text: str) -> int:
        f = getattr(self.b, "count_text", None)
        try:
            return int(f(text)) if f else cp.approx_tokens(text)
        except Exception:  # noqa: BLE001
            return cp.approx_tokens(text)

    def estimate(self, material_tokens: int, request_tokens: int, n: int, effort: str, compose: str = "synthesis") -> dict:
        """AUTO's transparent cost model (seconds) from this deployment's measured rates (Config): one ordinary
        request (read everything, think, answer) against plan + N subtasks (the material read once in all, admissions
        one after another, notes decoded side by side) + the answer written from the notes ("synthesis"), or the
        answer's sections written side by side ("direct": no notes, no synthesis)."""
        cfg = self.cfg
        # measured on this deployment (default effort): ordinary requests think ~300 / ~2,500 / 6,000+ tokens; at
        # large context a "high" request often thinks until the context is full
        think = {"low": 300, "medium": 2500, "high": 6000}.get(effort, 2500)
        answer = {"low": 300, "medium": 800, "high": 1500}.get(effort, 800)
        pf = lambda t: 1.2 + t / cfg.prefill_tok_s
        single = pf(material_tokens + request_tokens) + (think + answer) / cfg.decode_tok_s
        out = {"single_s": round(single, 1)}
        if n >= 2:
            notes = min(cfg.max_tokens, cfg.total_tokens // n)
            index = 40 * (material_tokens // max(1, cfg.chunk_tokens)) if material_tokens else 0
            par = (pf(request_tokens + index) + 120 / cfg.decode_tok_s
                   + material_tokens / cfg.prefill_tok_s + n * cfg.admission_s + notes / cfg.slot_rate(n)
                   + pf(notes * n) + (cfg.synthesis_reasoning + answer) / cfg.decode_tok_s)
            if compose == "direct":
                sec = answer * 1.3 / n           # each writer's share of the (somewhat longer) answer
                par = (pf(request_tokens + index) + 160 / cfg.decode_tok_s + material_tokens / cfg.prefill_tok_s
                       + n * cfg.admission_s + sec / cfg.slot_rate(n))
            out["parallel_s"] = round(par, 1)
        return out

    def run(self, messages: list, mode: Mode, cancel: threading.Event, synthesis_thinking: str = "on",
            context_items=None, provider=None) -> Outcome:
        """Plan and run the subtasks, then return how to write the answer.  Never raises for a model's mistake:
        any failure falls back to "direct"."""
        cfg = self.cfg
        t0 = time.time()
        meta = {"mode": "auto" if mode.kind == "auto" else str(mode.workers), "enabled": False, "workers": 1}
        if mode.off:
            return Outcome("direct", meta=meta)
        slots = max(1, int(self.b.concurrency() or 1))
        max_workers = min(cfg.max_workers, 8, mode.workers if mode.kind == "fixed" else cfg.max_workers)
        if mode.kind == "auto" and slots < 2:
            meta.update(decision="1: the engine runs one request at a time")
            return Outcome("direct", meta=meta)
        thr = cfg.partition_min_tokens
        if cfg.share_context != 0:
            thr = max(thr, cfg.share_context if cfg.share_context > 0 else self._slot_context() - 4096)
        material = build_material(messages, dataclasses.replace(cfg, partition_min_tokens=thr), self._count,
                                  context_items, provider)
        if material.items or material.provider:
            meta["context"] = material.meta()
        if mode.kind == "auto":
            busy = self._busy()
            max_workers = min(max_workers, slots - busy)
            if busy:
                meta["slots_busy"] = busy
            if max_workers < 2:
                meta.update(decision=f"1: {busy} of {slots} slots are busy with other requests")
                return Outcome("direct", meta=meta)
            last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            score, why = gate_score(material.request_view if material.partitioned
                                    else _text((last or {}).get("content")))
            if material.partitioned:
                score += 3
                why.append(f"large context ({material.total_tokens:,} tokens)")
            meta["gate"] = {"score": score, "signals": why}
            if score < cfg.gate_threshold:
                meta.update(decision="1: not worth planning (" + (", ".join(why) or "no signal") + ")")
                return Outcome("direct", meta=meta)
        shared = [{"role": "system", "content": shared_context(messages, material)}]
        exact = mode.workers if mode.kind == "fixed" else None
        # ---- plan
        self.progress("planning")
        prompt_len = self.b.count_tokens(Call("synthesis", shared + [{"role": "user", "content": "x" * 64}], 1))
        # large material kept whole in the shared prefix (the planner reads it; every later step restores it)
        big_shared = not material.partitioned and prompt_len >= cfg.partition_min_tokens
        # compose "auto" (measured, see docs/TASK_PARALLEL.md): over a large shared material the answer's sections
        # are written directly from it (no notes: each writer sees all of it); a request without large material gets
        # notes + one synthesis (blind-graded best of the forms: sections written without notes contradicted each other)
        direct = slots >= 2 and (cfg.compose == "direct" or (cfg.compose == "auto" and big_shared))
        pprompt = (planner_prompt_direct if direct else planner_prompt)(cfg, max_workers, exact, material.partitioned)
        # sources lists (partitioned material) and section designs (direct) make longer plans than v1's
        ptoks = cfg.planner_tokens + (40 * max_workers if direct else 0) + \
            (min(600, 6 * len(material.assignable)) if material.partitioned else 0)
        pcall = Call("planner", shared + [{"role": "user", "content": pprompt}],
                     ptoks, thinking="off", shared_prefix=shared)
        pres = self._guarded(pcall, cancel, cfg.worker_timeout_s)
        t_plan = time.time()
        meta["planning_ms"] = _ms(t0, t_plan)
        meta["planner_tokens"] = pres.completion_tokens
        meta["planner_prompt_tokens"] = pres.prompt_tokens
        if cancel.is_set():
            meta.update(decision="cancelled while planning", cancelled=True)
            return Outcome("direct", meta=meta)
        if pres.error:
            meta.update(decision="1: the planner failed", fallback=f"planner: {pres.error}")
            return Outcome("direct", meta=meta)
        room = self.b.context() - prompt_len - cfg.synthesis_min_tokens - 600   # the synthesis' own text and slack
        # large material kept whole in the shared prefix: the planner has just read it.  If the request is answered
        # by one stream after all, that stream restores what the planner read instead of the ordinary request reading
        # it all again (measured: a 16K re-read costs ~25 s, 120K ~190 s)

        def one(why: str) -> Outcome:
            if not big_shared:
                meta.update(decision=why)
                return Outcome("direct", meta=meta)
            meta.update(decision=why + " - answered from the material already read", compose="one answer, shared prefix")
            meta["_t0"], meta["_t_work"] = t0, time.time()
            return Outcome("synthesize", meta=meta, synthesis=Call(
                "synthesis", shared + [{"role": "user", "content": "STEP final answer. Answer the USER REQUEST "
                                        "directly and completely; work it out carefully."}],
                0, thinking=synthesis_thinking, reasoning_budget=0, shared_prefix=shared))
        known = {it.id for it in material.assignable} if material.partitioned else None
        try:
            try:
                plan = parse_plan(pres.text, cfg, max_workers, exact, token_room=room, known_ids=known)
            except PlanError as e0:
                # one more try, told what was wrong (the planner is greedy: the same prompt would give the same
                # reply); it restores the shared prefix, so it costs only its own short read and the JSON
                meta["plan_retry"] = str(e0)
                rcall = Call("planner", shared + [{"role": "user", "content": pprompt + f" (Your last reply was not "
                             f"usable: {e0}. Reply with only the JSON object, nothing else.)"}],
                             ptoks, thinking="off", shared_prefix=shared)
                rres = self._guarded(rcall, cancel, cfg.worker_timeout_s)
                meta["planning_ms"] = _ms(t0, time.time())
                if rres.error or cancel.is_set():
                    raise
                plan = parse_plan(rres.text, cfg, max_workers, exact, token_room=room, known_ids=known)
        except PlanError as e:
            if not (material.partitioned and not direct):
                meta.update(fallback=f"plan: {e}")
                return one("1: the plan was not usable")
            plan = Plan(1, "partition", "", [], "high")
            meta["plan_error"] = str(e)
        if plan.parallelism == 1 and material.partitioned and not direct:
            # the material is too large for one careful read: whatever the planner saw, the same question over
            # slices of it, combined after, reads it once and answers sooner (one request would read all of it and
            # then think over all of it)
            n = exact or min(max_workers, max(2, material.total_tokens // max(1, cfg.min_shard_tokens)))
            obj = ("From your part of the material, extract everything the USER REQUEST needs (every relevant item "
                   "with its exact figures)")
            plan = Plan(n, "shard", "large material", [Subtask(i, obj, "", cfg.max_tokens) for i in range(1, n + 1)],
                        plan.effort or "high")
            meta["shard_fallback"] = True
        meta["strategy"] = plan.strategy if plan.parallelism > 1 else None
        meta["reason"] = plan.reason
        meta["effort"] = plan.effort or None
        if plan.parallelism == 1:
            return one("1: the planner found no independent parts")
        if mode.kind == "auto" and big_shared and plan.effort in cfg.auto_shared_one_effort.split(","):
            # measured (tri-Arc, 8K-120K documents): sections written in parallel without thinking missed figures that
            # one careful pass over the whole material found (totals over dozens of units, every contract with a
            # short notice); a lookup ("low") gained 1.3-2x at equal scores.  One stream that restores what the
            # planner read keeps the quality and skips the second read
            return one(f"1: {plan.effort}-effort request over a large material (one careful answer)")
        if (mode.kind == "auto" and plan.effort in cfg.auto_skip_effort.split(",") and not material.partitioned
                and not big_shared):
            # measured break-even: the plan, the admissions and the synthesis cost more than a request whose single
            # answer needs little work saves by running its parts side by side (with large material the reading
            # dominates, and it is already done: the parts then pay even for a "low" effort request)
            meta.update(decision=f"1: {plan.effort}-effort request (parallel work would not pay)")
            return Outcome("direct", meta=meta)
        if mode.kind == "auto" and material.partitioned:
            # no slice thinner than min_shard_tokens: fewer subtasks read more each
            cap = max(2, material.total_tokens // max(1, cfg.min_shard_tokens))
            if plan.parallelism > cap:
                plan.subtasks = plan.subtasks[:cap]
                plan.parallelism = cap
                meta["capped"] = f"{cap} subtasks: slices of at least {cfg.min_shard_tokens} tokens"
        if material.partitioned:
            est = self.estimate(material.total_tokens, self._count(material.request_view), plan.parallelism,
                                plan.effort or "medium", compose="direct" if direct else cfg.compose)
            meta["estimate"] = est
            if mode.kind == "auto" and est.get("parallel_s", 0) >= est["single_s"]:
                meta.update(decision=f"1: estimated {est['single_s']} s alone vs {est['parallel_s']} s in parallel")
                return Outcome("direct", meta=meta)
        if direct:
            return self._direct(plan, material, shared, cancel, slots, meta, t0, prompt_len)
        # ---- sources: each subtask reads its own part of the material
        sources = [[] for _ in plan.subtasks]
        if material.partitioned and material.assignable:
            wctx = self._slot_context() - prompt_len - max(cfg.max_tokens * 4, max(s.max_tokens for s in plan.subtasks)) - 1200
            budget = max(cfg.chunk_tokens, min(cfg.worker_source_tokens, wctx))
            sources, cov = assign_sources(plan, material, budget)
            meta["sources"] = cov
            # a subtask over more material reports more facts: room for them (about a fifth of what it reads,
            # measured: a shard's list of every unit's figures; still bounded)
            for st, src in zip(plan.subtasks, sources):
                st.max_tokens = max(st.max_tokens, min(cfg.max_tokens * 4, sum(i.tokens for i in src) // 5))
        # ---- the subtasks, at the same time
        self.progress(f"running {plan.parallelism} subtasks")
        deadline = t0 + cfg.total_timeout_s
        results = self._run_workers(plan, shared, cancel, deadline, slots, sources, material.partitioned)
        t_work = time.time()
        meta.update(enabled=True, workers=plan.parallelism, workers_ms=_ms(t_plan, t_work))
        meta["worker_results"] = [{
            "tokens": r.completion_tokens if r else 0, "reasoning_tokens": r.reasoning_tokens if r else 0,
            "prompt_tokens": r.prompt_tokens if r else 0, "reused_tokens": r.reused_tokens if r else 0,
            "finish": (r.finish or ("error" if r.error else "")) if r else "missing",
            "start_ms": _ms(t_plan, r.t_start) if r else None,
            "first_token_ms": _ms(t_plan, r.t_first) if r and r.t_first else None,
            "end_ms": _ms(t_plan, r.t_end) if r else None,
            "attempts": r.attempts if r else 0} for r in results]
        gen = sum(r.completion_tokens for r in results if r)
        meta["worker_tokens"] = gen
        meta["worker_aggregate_tok_s"] = round(gen / max(1e-6, t_work - t_plan), 1)
        if cancel.is_set():
            meta.update(cancelled=True, decision="cancelled while the subtasks ran")
            return Outcome("direct", meta=meta)
        ok = [r for r in results if r is not None and r.text.strip() and not r.error]
        if not ok:
            meta.update(enabled=False, decision="1: no subtask completed", fallback="no worker output")
            return Outcome("direct", meta=meta)
        if len(ok) < len(results):
            meta["partial"] = f"{len(ok)} of {len(results)} subtasks completed"
        # ---- one bounded follow-up round: material a subtask asked for (NEED lines)
        if material.partitioned and cfg.need_total > 0:
            self._needs(plan, results, sources, material, shared, cancel, deadline, slots, meta)
            if cancel.is_set():
                meta.update(cancelled=True, decision="cancelled during follow-ups")
                return Outcome("direct", meta=meta)
        meta["decision"] = f"{plan.parallelism}: {plan.strategy}"
        meta["_t0"], meta["_t_work"] = t0, time.time()
        meta["_subtasks"] = [{"objective": s.objective, "expected_output": s.expected_output,
                              "max_tokens": s.max_tokens,
                              **({"sources": [it.id for it in src]} if src else {})}
                             for s, src in zip(plan.subtasks, sources)]
        # ---- the answer
        if cfg.compose == "sections" and slots >= 2:   # ("auto" writes the notes' answer in one synthesis: measured best)
            out = self._compose(plan, results, shared, cancel, slots, synthesis_thinking, meta)
            if out is not None:
                return out
        self.progress("synthesizing")
        synth = Call("synthesis", shared + [{"role": "user", "content": synthesis_prompt(plan, results)}],
                     0, thinking=synthesis_thinking,
                     reasoning_budget=cfg.synthesis_reasoning if synthesis_thinking != "off" else 0,
                     shared_prefix=shared)
        meta["compose"] = meta.get("compose") or "synthesis"
        return Outcome("synthesize", synthesis=synth, meta=meta)

    # --------------------------------------------------------------------------------------------- direct
    def _direct(self, plan, material, shared, cancel, slots, meta, t0, prompt_len) -> Outcome:
        """compose "direct": the plan is the answer's outline; every section is written at once from the request (and
        its own sources), the closing ones ("after") once the others are done, from their text."""
        cfg = self.cfg
        secs = [{"heading": st.heading or _short(st.objective, 8).rstrip("."), "covers": st.objective,
                 "words": max(cfg.section_min_words, min(900, st.words or int(st.max_tokens * 0.7)))}
                for st in plan.subtasks]
        seen = set()
        for i, sc in enumerate(secs, 1):                       # headings must differ (they head the answer)
            if _norm(sc["heading"]) in seen:
                sc["heading"] = f"{sc['heading']} ({i})"
            seen.add(_norm(sc["heading"]))
        first = [k for k, st in enumerate(plan.subtasks) if not st.after]
        if not first:                                          # all "after": nothing to build on - write them at once
            first = list(range(len(plan.subtasks)))
        sources = [[] for _ in plan.subtasks]
        if material.partitioned and material.assignable:
            wctx = self._slot_context() - prompt_len - max(sc["words"] for sc in secs) * 2 - 1500
            budget = max(cfg.chunk_tokens, min(cfg.worker_source_tokens, wctx))
            sub = Plan(len(first), "partition", plan.reason, [plan.subtasks[k] for k in first], plan.effort)
            got, cov = assign_sources(sub, material, budget)
            for k, g in zip(first, got):
                sources[k] = g
            meta["sources"] = cov
        specs = []
        for k, (st, sec) in enumerate(zip(plan.subtasks, secs)):
            mt = int(sec["words"] * cfg.section_tokens_per_word) + 60 + cfg.writer_reasoning
            think = "on" if cfg.writer_reasoning > 0 else "off"
            if k in first:
                c = Call("writer", shared + [{"role": "user", "content": direct_writer_prompt(secs, k + 1, sources[k])}],
                         mt, thinking=think, reasoning_budget=cfg.writer_reasoning, shared_prefix=shared)
                specs.append({"section": sec, "call": c})
            else:
                def build(texts, k=k, mt=mt, think=think):
                    return Call("writer", shared + [{"role": "user", "content": closing_prompt(secs, k + 1, texts)}],
                                mt, thinking=think, reasoning_budget=cfg.writer_reasoning, shared_prefix=shared)
                specs.append({"section": sec, "build": build})
        if len(first) > 1:                                     # the outline is a second shared prefix of the writers
            calls = [sp["call"] for sp in specs if "call" in sp]
            for i, c in enumerate(calls):
                c.common_with = calls[(i + 1) % len(calls)].messages
        meta.update(enabled=True, workers=len(first), decision=f"{len(secs)}: answer sections", compose="direct",
                    sections=len(secs), closing_sections=len(secs) - len(first))
        meta["_t0"], meta["_t_work"] = t0, time.time()
        meta["_subtasks"] = [{"objective": st.objective, "heading": sc["heading"], "words": sc["words"],
                              **({"sources": [it.id for it in src]} if src else {}), **({"after": True} if st.after else {})}
                             for st, sc, src in zip(plan.subtasks, secs, sources)]
        meta["_outline"] = {"sections": secs, "facts": []}
        comp = Composer(self, specs, cancel, slots, meta)
        comp.start()
        return Outcome("composed", meta=meta, composer=comp)

    # --------------------------------------------------------------------------------------------- follow-ups
    def _needs(self, plan, results, sources, material, shared, cancel, deadline, slots, meta):
        cfg = self.cfg
        bm = cp.Bm25(material.assignable)
        asks, seen = [], set()
        for k, r in enumerate(results):
            if r is None or not r.text:
                continue
            r.text, needs = split_needs(r.text, cfg.need_per_worker)
            for q in needs:
                key = (k, _norm(q))
                if key in seen or len(asks) >= cfg.need_total:
                    continue
                seen.add(key)
                have = {it.id for it in sources[k]}
                got = cp.pack(bm.search(q, 8, exclude=have), cfg.need_tokens)
                asks.append((k, q, got))
        if not asks:
            return
        t = time.time()
        self.progress(f"following up {len(asks)} requests for more material")
        gate = threading.Semaphore(max(1, slots))
        answers: list = [None] * len(asks)

        def one(j):
            k, q, got = asks[j]
            st = plan.subtasks[k]
            if not got:
                answers[j] = Result(text="(not in the material)", finish="stop")
                return
            body = ("SOURCES:\n\n" + "\n\n".join(i.render() for i in got)
                    + f"\n\nSTEP subtask {st.id}/{len(plan.subtasks)} follow-up: for the subtask \"{st.objective}\" "
                      f"this was asked: {q} Answer only that, from the SOURCES, in at most 120 words, citing [id]; if "
                      "they do not contain it, say so.")
            call = Call("followup", shared + [{"role": "user", "content": body}], 220, thinking="off",
                        shared_prefix=shared)
            with gate:
                answers[j] = self._guarded(call, cancel, max(1.0, min(cfg.worker_timeout_s, deadline - time.time())))
        th = [threading.Thread(target=one, args=(j,), daemon=True) for j in range(len(asks))]
        for x in th:
            x.start()
        for x in th:
            x.join(max(0.0, deadline - time.time()) + 30)
        n_tok = 0
        for (k, q, got), a in zip(asks, answers):
            if a is not None and a.text.strip() and not a.error and results[k] is not None:
                results[k].text += f"\n\nFollow-up - {q}\n{a.text.strip()}"
                n_tok += a.completion_tokens
        meta["followups"] = {"asked": len(asks), "answered": sum(1 for a in answers if a and a.text.strip()
                                                                  and not a.error),
                             "tokens": n_tok, "ms": _ms(t, time.time())}

    # --------------------------------------------------------------------------------------------- the answer
    def _compose(self, plan, results, shared, cancel, slots, thinking, meta) -> "Outcome | None":
        """The answer in sections: an outline (sections + the facts they must agree on) from the notes, then every
        section written at the same time and streamed in order.  None: write it as one synthesis instead."""
        cfg = self.cfg
        t = time.time()
        self.progress("outlining")
        notes = notes_block(plan, results)
        base = [{"role": "system", "content": shared[0]["content"] + "\n\nNOTES (from the subtasks):\n" + notes}]
        ocall = Call("outline", base + [{"role": "user", "content": outline_prompt(min(cfg.max_sections, slots))}],
                     cfg.outline_tokens + (cfg.outline_reasoning if thinking != "off" else 0), thinking=thinking,
                     reasoning_budget=cfg.outline_reasoning if thinking != "off" else 0, shared_prefix=base)
        ores = self._guarded(ocall, cancel, cfg.worker_timeout_s)
        meta["outline_ms"] = _ms(t, time.time())
        meta["outline_tokens"] = ores.completion_tokens
        if cancel.is_set():
            return None
        try:
            if ores.error:
                raise PlanError(ores.error)
            outline = parse_outline(ores.text, cfg, min(cfg.max_sections, slots))
        except PlanError as e:
            meta["compose"] = f"synthesis (outline: {e})"
            return None
        words = sum(s["words"] for s in outline["sections"])
        if cfg.compose == "auto" and words < cfg.compose_min_words:
            meta["compose"] = f"synthesis ({words} words: too short for sections)"
            return None
        meta["compose"] = "sections"
        meta["sections"] = len(outline["sections"])
        meta["_outline"] = outline
        calls = []
        for k in range(1, len(outline["sections"]) + 1):
            common, own = writer_prompt(outline, k)
            calls.append(Call("writer", base + [{"role": "user", "content": common + own}],
                              int(outline["sections"][k - 1]["words"] * cfg.section_tokens_per_word) + 60 + cfg.writer_reasoning,
                              thinking="on" if cfg.writer_reasoning > 0 else "off",
                              reasoning_budget=cfg.writer_reasoning, shared_prefix=base))
        for k, c in enumerate(calls):    # the outline + facts are a second shared prefix: a sibling marks it
            c.common_with = calls[(k + 1) % len(calls)].messages if len(calls) > 1 else None
        comp = Composer(self, [{"section": sec, "call": c} for sec, c in zip(outline["sections"], calls)], cancel,
                        slots, meta)
        comp.start()
        return Outcome("composed", meta=meta, composer=comp)

    # --------------------------------------------------------------------------------------------- workers
    def _guarded(self, call: Call, parent: threading.Event, timeout: float) -> Result:
        """One internal request with its own stop (set on the parent's cancel or the timeout)."""
        mine = threading.Event()
        done = threading.Event()

        def watch():
            end = time.time() + timeout
            while not done.wait(0.05):
                if parent.is_set() or time.time() > end:
                    mine.set()
                    return
        w = threading.Thread(target=watch, daemon=True)
        w.start()
        t = time.time()
        try:
            r = self.b.generate(call, mine)
        except Exception as e:           # an internal request's failure is the orchestrator's to handle
            r = Result(error=f"{type(e).__name__}: {e}", t_start=t, t_end=time.time())
        finally:
            done.set()
        if mine.is_set() and not parent.is_set() and not r.error:
            r.error = "timed out"
            r.finish = r.finish or "cancel"
        return r

    def _run_workers(self, plan: Plan, shared: list, cancel: threading.Event, deadline: float,
                     slots: int | None = None, sources: list | None = None, need: bool = False) -> list:
        cfg = self.cfg
        results: list = [None] * len(plan.subtasks)
        # never more subtasks at once than the engine runs at once: a fixed count above it queues here, so a worker's
        # timeout counts its own run, not its wait for a slot
        gate = threading.Semaphore(max(1, slots or len(plan.subtasks)))
        sources = sources or [[] for _ in plan.subtasks]

        def one(i: int, st: Subtask):
            call = Call("worker", shared + [{"role": "user", "content": worker_prompt(plan, st, sources[i], need)}],
                        st.max_tokens + cfg.worker_reasoning,
                        thinking="on" if cfg.worker_reasoning > 0 else "off",
                        reasoning_budget=cfg.worker_reasoning, shared_prefix=shared)
            attempts = 0
            while True:
                attempts += 1
                while not gate.acquire(timeout=0.5):
                    if cancel.is_set() or time.time() > deadline:
                        results[i] = Result(error="cancelled" if cancel.is_set() else "timed out", attempts=attempts)
                        return
                try:
                    left = deadline - time.time()
                    r = self._guarded(call, cancel, max(1.0, min(cfg.worker_timeout_s, left)))
                finally:
                    gate.release()
                r.attempts = attempts
                bad = bool(r.error) or not r.text.strip()
                if not bad or cancel.is_set() or attempts > cfg.retries or deadline - time.time() < 5.0 \
                        or r.error == "timed out":
                    results[i] = r
                    return

        threads = [threading.Thread(target=one, args=(i, st), daemon=True) for i, st in enumerate(plan.subtasks)]
        for t in threads:
            t.start()
        for t in threads:                # every worker ends by its own timeout or the parent's cancel
            t.join(max(0.0, deadline - time.time()) + cfg.worker_timeout_s + 30.0)
        return results


# ------------------------------------------------------------------------------------------------ parallel writing
def _shingles(text: str) -> set:
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + 3]) for i in range(max(0, len(w) - 2))}


class Composer:
    """The answer's sections written at the same time (internal requests in the batch slots) and streamed in order:
    the first section as it is written, each later one once the ones before it are out.  `specs`: one per section,
    {"section": {"heading", ...}, "call": Call} for the sections written at once, or {"section", "build": f(texts) ->
    Call} for the closing ones (a summary, a recommendation) that are written after the others, from their text.
    Assembly is deterministic: a paragraph that repeats an earlier one (3-word shingles, Jaccard >= 0.8) is dropped,
    every section starts with its heading.  Nothing is rewritten."""

    def __init__(self, orch: "Orchestrator", specs: list, cancel: threading.Event, slots: int, meta: dict):
        self.o, self.specs, self.cancel, self.slots, self.meta = orch, specs, cancel, slots, meta
        self.sections = [sp["section"] for sp in specs]
        n = len(specs)
        self.buf = [[] for _ in range(n)]
        self.done = [False] * n
        self.res: list = [None] * n
        self.cv = threading.Condition()
        self.t0 = time.time()
        self.t_first_out = None
        self.t_wave2 = None
        self.dropped = 0
        self.threads: list = []

    def _write(self, k: int, call: Call, gate: threading.Semaphore, deadline: float) -> None:
        cfg = self.o.cfg

        def on_text(s: str):
            with self.cv:
                self.buf[k].append(s)
                self.cv.notify_all()
        call.on_text = on_text
        r = None
        for attempt in range(1, cfg.retries + 2):
            with self.cv:
                self.buf[k] = []
            with gate:
                r = self.o._guarded(call, self.cancel, max(1.0, min(cfg.worker_timeout_s, deadline - time.time())))
            r.attempts = attempt
            if (r.text.strip() and not r.error) or self.cancel.is_set():
                break
        with self.cv:
            self.res[k] = r
            self.done[k] = True
            if not r.text.strip() or r.error:
                self.buf[k] = []
            self.cv.notify_all()

    def start(self) -> None:
        cfg = self.o.cfg
        gate = threading.Semaphore(max(1, self.slots))
        first = [k for k, sp in enumerate(self.specs) if "call" in sp]
        later = [k for k, sp in enumerate(self.specs) if "call" not in sp]
        self.o.progress(f"writing {len(first)} sections" + (f" (then {len(later)})" if later else ""))
        deadline = time.time() + cfg.total_timeout_s
        threads = [threading.Thread(target=self._write, args=(k, self.specs[k]["call"], gate, deadline), daemon=True)
                   for k in first]

        def wave2():
            for t in threads:
                t.join()
            if self.cancel.is_set():
                for k in later:
                    with self.cv:
                        self.res[k] = Result(error="cancelled")
                        self.done[k] = True
                        self.cv.notify_all()
                return
            self.t_wave2 = time.time()
            texts = [(self.sections[k]["heading"], (self.res[k].text if self.res[k] else "").strip()) for k in first]
            ws = [threading.Thread(target=self._write, args=(k, self.specs[k]["build"](texts), gate, deadline),
                                   daemon=True) for k in later]
            for t in ws:
                t.start()
            for t in ws:
                t.join()
        self.threads = list(threads)
        for t in threads:
            t.start()
        if later:
            w = threading.Thread(target=wave2, daemon=True)
            w.start()
            self.threads.append(w)

    def stream(self):
        """The answer's text in order (paragraph by paragraph).  Blocks while sections are being written."""
        seen: list = []
        for k, sec in enumerate(self.sections):
            pos, head_done, pending = 0, False, ""
            while True:
                with self.cv:
                    while not self.done[k] and sum(len(x) for x in self.buf[k]) == pos and not self.cancel.is_set():
                        self.cv.wait(0.5)
                    text = "".join(self.buf[k])
                    fin = self.done[k] or self.cancel.is_set()
                if self.cancel.is_set():
                    return
                if len(text) < pos:      # a retry started the section again: nothing of it was sent yet
                    pos, pending = 0, ""
                new = text[pos:]
                pos = len(text)
                pending += new
                parts = pending.split("\n\n")
                pending = parts.pop() if not fin else ""
                if fin and parts and not parts[-1].strip():
                    parts.pop()
                for para in parts:
                    p = para.strip("\n")
                    if not p.strip():
                        continue
                    if not head_done:
                        head_done = True
                        if not p.lstrip().startswith("#"):
                            yield f"## {sec['heading']}\n\n"
                        else:
                            p = f"## {sec['heading']}" + ("\n" + p.split("\n", 1)[1] if "\n" in p else "")
                            if not p.strip():
                                continue
                    sh = _shingles(p)
                    if len(sh) >= 8 and any(len(sh & s2) / max(1, len(sh | s2)) >= 0.8 for s2 in seen):
                        self.dropped += 1
                        continue
                    if len(sh) >= 8:
                        seen.append(sh)
                    if self.t_first_out is None:
                        self.t_first_out = time.time()
                    yield p + "\n\n"
                if fin:
                    r = self.res[k]
                    if not head_done and r is not None and (r.error or not r.text.strip()):
                        self.meta.setdefault("sections_missing", []).append(sec["heading"])
                    break

    def finish(self) -> dict:
        """Timings and counts once the stream has ended."""
        for t in self.threads:
            t.join(1.0)
        end = time.time()
        rs = self.res
        tok = sum(r.completion_tokens for r in rs if r)
        out = {"writers_ms": _ms(self.t0, end), "writer_tokens": tok,
               "writer_aggregate_tok_s": round(tok / max(1e-6, end - self.t0), 1),
               "first_text_ms": _ms(self.t0, self.t_first_out) if self.t_first_out else None,
               "paragraphs_dropped": self.dropped,
               "writer_results": [{"tokens": r.completion_tokens if r else 0,
                                   "finish": (r.finish or ("error" if r.error else "")) if r else "missing",
                                   "prompt_tokens": r.prompt_tokens if r else 0,
                                   "reused_tokens": r.reused_tokens if r else 0,
                                   "start_ms": _ms(self.t0, r.t_start) if r and r.t_start else None,
                                   "first_token_ms": _ms(self.t0, r.t_first) if r and r.t_first else None,
                                   "end_ms": _ms(self.t0, r.t_end) if r and r.t_end else None,
                                   "closing": "call" not in sp,
                                   "attempts": r.attempts if r else 0} for r, sp in zip(rs, self.specs)]}
        if self.t_wave2:
            out["closing_start_ms"] = _ms(self.t0, self.t_wave2)
        return out


def finish_meta(meta: dict, t_synth_start: float, synthesis_tokens: int, synthesis_reasoning_tokens: int,
                t_end: float | None = None, diagnostics: bool = False, composed: dict | None = None) -> dict:
    """The response's `task_parallel` object: timings and token counts; the subtasks' objectives (and the answer's
    outline) only on request (`task_parallel_diagnostics`), and never any internal reasoning or worker text."""
    t_end = t_end or time.time()
    out = {k: v for k, v in meta.items() if not k.startswith("_")}
    if "_t0" in meta:
        if composed is not None:
            out.update(composed)
            synthesis_tokens = composed.get("writer_tokens", 0)
            out["compose_ms"] = _ms(t_synth_start, t_end) + meta.get("outline_ms", 0)
        else:
            out["synthesis_ms"] = _ms(t_synth_start, t_end)
            out["synthesis_tokens"] = synthesis_tokens
            out["synthesis_reasoning_tokens"] = synthesis_reasoning_tokens
        out["total_ms"] = _ms(meta["_t0"], t_end)
        out["total_generated_tokens"] = (meta.get("planner_tokens", 0) + meta.get("worker_tokens", 0) +
                                         (meta.get("followups") or {}).get("tokens", 0) +
                                         meta.get("outline_tokens", 0) + synthesis_tokens)
    if diagnostics and "_subtasks" in meta:
        out["subtasks"] = meta["_subtasks"]
        if "_outline" in meta:
            out["outline"] = {"sections": [s["heading"] for s in meta["_outline"]["sections"]],
                              "facts": len(meta["_outline"]["facts"])}
    return out
