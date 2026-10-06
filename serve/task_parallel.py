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
from dataclasses import dataclass, field
from typing import Callable, Protocol

ALLOWED_WORKERS = (2, 3, 4, 5, 6, 7, 8)
STRATEGIES = ("partition", "independent")


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
    worker_timeout_s: float = 240.0      # one worker attempt; a timed-out worker is stopped
    total_timeout_s: float = 600.0       # planner + every worker attempt
    retries: int = 1                     # a failed worker (error / empty output) is tried again once
    gate_threshold: int = 2              # AUTO: the heuristic score a request needs before the planner is asked
    auto_skip_effort: str = "low"  # AUTO answers normally when the planner rates the request this effort
    internal_effort: str = "low"         # the reasoning effort every internal request is rendered with ("": the
    #                                      template's default) - one setting for all, so their shared prefix matches

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
        return c


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


# ------------------------------------------------------------------------------------------------ plans
@dataclass
class Subtask:
    id: int
    objective: str
    expected_output: str
    max_tokens: int


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
    if a < 0 or b <= a:
        raise PlanError("the plan is not a JSON object")
    try:
        obj = json.loads(t[a:b + 1])
    except json.JSONDecodeError as e:
        raise PlanError(f"the plan is not valid JSON: {e.msg}") from e
    if not isinstance(obj, dict):
        raise PlanError("the plan is not a JSON object")
    return obj


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def parse_plan(text: str, cfg: Config, max_workers: int, exact: int | None = None,
               token_room: int | None = None) -> Plan:
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
        if key in seen:
            raise PlanError(f"subtask {i} duplicates another")
        seen.add(key)
        mt = s.get("max_tokens", share)
        if isinstance(mt, float) and mt.is_integer():
            mt = int(mt)
        if not isinstance(mt, int) or isinstance(mt, bool):
            raise PlanError(f"subtask {i}'s max_tokens is not a number")
        mt = max(cfg.min_tokens, min(cfg.max_tokens, mt))
        expected = str(s.get("expected_output") or "").strip()[:400] or "Concise findings for this subtask."
        subtasks.append(Subtask(i, obj_text, expected, mt))
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


# ------------------------------------------------------------------------------------------------ prompts
def shared_context(messages: list) -> str:
    """The context every internal request starts with: the application's instructions, the earlier conversation
    and the user's request, as one text (the one system message all internal requests share, so the engine can
    read it once)."""
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
    out = [STEP_GUIDE]
    if any(sys_parts):
        out.append("APPLICATION INSTRUCTIONS (they apply to the final answer):\n" + "\n".join(p for p in sys_parts if p))
    if convo:
        out.append("EARLIER CONVERSATION:\n" + "\n\n".join(convo))
    out.append("USER REQUEST:\n" + (_text(messages[last_user].get("content")) if last_user is not None else ""))
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


def planner_prompt(cfg: Config, max_workers: int, exact: int | None) -> str:
    if exact:
        n_rule = f"parallelism is exactly {exact}."
    else:
        n_rule = (f"parallelism is 1 to {max_workers}: 1 for a short, simple, conversational or creative request, a "
                  "quick fact, or one chain of dependent steps; else the fewest subtasks covering its independent "
                  "parts.")
    return (
        f"STEP plan. Rules: {n_rule} strategy \"partition\": distinct parts (items, components, dimensions, "
        "questions), one per subtask, together covering everything without overlap; \"independent\": one hard "
        "problem, each subtask solves or checks the WHOLE of it from a different angle (solution, alternative, "
        "review, edge cases). Each subtask is one short sentence saying exactly what to produce. effort: how much "
        "work one careful answer to the whole request needs - \"low\" (routine, a few minutes for an expert), "
        "\"medium\", \"high\" (extended analysis, design, debugging or calculation across several parts). Reply "
        'with only this JSON: {"effort": "low"|"medium"|"high", "parallelism": <n>, "strategy": '
        '"partition"|"independent", "subtasks": ["...", "..."]}')


def _short(text: str, words: int = 12) -> str:
    w = text.split()
    return " ".join(w[:words]) + ("..." if len(w) > words else "")


def worker_prompt(plan: Plan, st: Subtask) -> str:
    words = max(50, int(st.max_tokens * 0.7))
    angle = " One angle on the whole problem." if plan.strategy == "independent" else ""
    return f"STEP subtask {st.id}/{len(plan.subtasks)}: {st.objective}{angle} Limit: about {words} words."


def synthesis_prompt(plan: Plan, results: list) -> str:
    parts = []
    for st, r in zip(plan.subtasks, results):
        if r is not None and r.text.strip():
            cut = " (cut short)" if r.finish == "length" else ""
            parts.append(f"=== {st.id}. {st.objective}{cut}\n{r.text.strip()}")
        else:
            parts.append(f"=== {st.id}. {st.objective}\n(did not complete: cover it yourself if the request needs it)")
    kind = "Each note attacks the whole problem from one angle." if plan.strategy == "independent" else \
        "Each note covers one part."
    return "STEP final answer. " + kind + " Notes:\n\n" + "\n\n".join(parts)


# ------------------------------------------------------------------------------------------------ orchestration
@dataclass
class Outcome:
    """What the caller does next.  kind "direct": run the original request as usual (meta says why).  kind
    "synthesize": run `synthesis` (a Call) as the request's answer."""
    kind: str
    synthesis: Call | None = None
    meta: dict = field(default_factory=dict)


def _ms(a: float, b: float) -> int:
    return int(round((b - a) * 1000))


class Orchestrator:
    def __init__(self, backend: Backend, cfg: Config | None = None,
                 progress: Callable[[str], None] | None = None):
        self.b, self.cfg = backend, cfg or Config()
        self.progress = progress or (lambda s: None)

    def run(self, messages: list, mode: Mode, cancel: threading.Event, synthesis_thinking: str = "on") -> Outcome:
        """Plan and run the subtasks.  Never raises for a model's mistake: any failure falls back to "direct"."""
        cfg = self.cfg
        t0 = time.time()
        meta = {"mode": "auto" if mode.kind == "auto" else str(mode.workers), "enabled": False, "workers": 1}
        if mode.off:
            return Outcome("direct", meta=meta)
        slots = max(1, int(self.b.concurrency() or 1))
        max_workers = min(cfg.max_workers, 8, mode.workers if mode.kind == "fixed" else cfg.max_workers)
        if mode.kind == "auto":
            if slots < 2:
                meta.update(decision="1: the engine runs one request at a time")
                return Outcome("direct", meta=meta)
            max_workers = min(max_workers, slots)
            last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            score, why = gate_score(_text((last or {}).get("content")))
            meta["gate"] = {"score": score, "signals": why}
            if score < cfg.gate_threshold:
                meta.update(decision="1: not worth planning (" + (", ".join(why) or "no signal") + ")")
                return Outcome("direct", meta=meta)
        shared = [{"role": "system", "content": shared_context(messages)}]
        exact = mode.workers if mode.kind == "fixed" else None
        # ---- plan
        self.progress("planning")
        pcall = Call("planner", shared + [{"role": "user", "content": planner_prompt(cfg, max_workers, exact)}],
                     cfg.planner_tokens, thinking="off", shared_prefix=shared)
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
        prompt_len = self.b.count_tokens(Call("synthesis", shared + [{"role": "user", "content": "x" * 64}], 1))
        room = self.b.context() - prompt_len - cfg.synthesis_min_tokens - 600   # the synthesis' own text and slack
        try:
            plan = parse_plan(pres.text, cfg, max_workers, exact, token_room=room)
        except PlanError as e:
            meta.update(decision="1: the plan was not usable", fallback=f"plan: {e}")
            return Outcome("direct", meta=meta)
        meta["strategy"] = plan.strategy if plan.parallelism > 1 else None
        meta["reason"] = plan.reason
        meta["effort"] = plan.effort or None
        if plan.parallelism == 1:
            meta.update(decision="1: the planner found no independent parts")
            return Outcome("direct", meta=meta)
        if mode.kind == "auto" and plan.effort in cfg.auto_skip_effort.split(","):
            # measured break-even: the plan, the admissions and the synthesis cost more than a request whose single
            # answer needs little work saves by running its parts side by side
            meta.update(decision=f"1: {plan.effort}-effort request (parallel work would not pay)")
            return Outcome("direct", meta=meta)
        # ---- the subtasks, at the same time
        self.progress(f"running {plan.parallelism} subtasks")
        deadline = t0 + cfg.total_timeout_s
        results = self._run_workers(plan, shared, cancel, deadline)
        t_work = time.time()
        meta.update(enabled=True, workers=plan.parallelism, workers_ms=_ms(t_plan, t_work))
        ok = [r for r in results if r is not None and r.text.strip() and not r.error]
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
        if not ok:
            meta.update(enabled=False, decision="1: no subtask completed", fallback="no worker output")
            return Outcome("direct", meta=meta)
        if len(ok) < len(results):
            meta["partial"] = f"{len(ok)} of {len(results)} subtasks completed"
        self.progress("synthesizing")
        synth = Call("synthesis", shared + [{"role": "user", "content": synthesis_prompt(plan, results)}],
                     0, thinking=synthesis_thinking,
                     reasoning_budget=cfg.synthesis_reasoning if synthesis_thinking != "off" else 0,
                     shared_prefix=shared)
        meta["decision"] = f"{plan.parallelism}: {plan.strategy}"
        meta["_t0"], meta["_t_work"] = t0, t_work
        meta["_subtasks"] = [{"objective": s.objective, "expected_output": s.expected_output,
                              "max_tokens": s.max_tokens} for s in plan.subtasks]
        return Outcome("synthesize", synthesis=synth, meta=meta)

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

    def _run_workers(self, plan: Plan, shared: list, cancel: threading.Event, deadline: float) -> list:
        cfg = self.cfg
        results: list = [None] * len(plan.subtasks)

        def one(i: int, st: Subtask):
            call = Call("worker", shared + [{"role": "user", "content": worker_prompt(plan, st)}],
                        st.max_tokens + cfg.worker_reasoning,
                        thinking="on" if cfg.worker_reasoning > 0 else "off",
                        reasoning_budget=cfg.worker_reasoning, shared_prefix=shared)
            attempts = 0
            while True:
                attempts += 1
                left = deadline - time.time()
                r = self._guarded(call, cancel, max(1.0, min(cfg.worker_timeout_s, left)))
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


def finish_meta(meta: dict, t_synth_start: float, synthesis_tokens: int, synthesis_reasoning_tokens: int,
                t_end: float | None = None, diagnostics: bool = False) -> dict:
    """The response's `task_parallel` object: timings and token counts; the subtasks' objectives only on request
    (`task_parallel_diagnostics`), and never any internal reasoning or worker text."""
    t_end = t_end or time.time()
    out = {k: v for k, v in meta.items() if not k.startswith("_")}
    if "_t0" in meta:
        out["synthesis_ms"] = _ms(t_synth_start, t_end)
        out["total_ms"] = _ms(meta["_t0"], t_end)
        out["synthesis_tokens"] = synthesis_tokens
        out["synthesis_reasoning_tokens"] = synthesis_reasoning_tokens
        out["total_generated_tokens"] = (meta.get("planner_tokens", 0) + meta.get("worker_tokens", 0) +
                                         synthesis_tokens)
    if diagnostics and "_subtasks" in meta:
        out["subtasks"] = meta["_subtasks"]
    return out
