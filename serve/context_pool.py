"""Context for task-parallel requests: the material a request brings, cut into addressable pieces.

A request can carry far more material than one step should read: a long document pasted into a message, documents
sent with it (`context_items`), or durable memories / live state that a retrieval provider returns.  This module
turns all of it into `Item`s with stable ids, builds a short index of them (what a planner reads instead of the
material itself), finds the items relevant to a question (BM25, no model, no network) and packs items into token
budgets.  It knows nothing about models, engines or any particular memory system: a memory/RAG backend plugs in
through `ContextProvider` (one `retrieve` call per request; `HttpContextProvider` is a generic JSON adapter).

Item kinds: "document" (material the request brought), "memory" (durable knowledge from a provider: provenance and
authority are kept and shown), "live" (mutable system state: always shown with its as-of time, never treated as a
durable fact).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

KINDS = ("document", "memory", "live")


@dataclass
class Item:
    id: str                              # "c12" (a chunk of the request's material), "d3" (a sent document), "m5"
    text: str
    title: str = ""
    source: str = ""                     # provenance (a file, a URL, a memory store's key)
    kind: str = "document"               # "document" | "memory" | "live"
    authority: str = ""                  # the caller's label for how far to trust it ("verified", "user", ...)
    as_of: str = ""                      # live state: when it was read
    tokens: int = 0
    order: int = 0                       # position in the original material (assembly keeps it)
    pinned: bool = False                 # every step sees it (global instructions, definitions, key facts)
    score: float = 0.0                   # the provider's relevance, when it gave one

    def header(self) -> str:
        bits = [self.id]
        if self.title:
            bits.append(self.title)
        if self.kind != "document":
            bits.append(self.kind)
        if self.source:
            bits.append(f"source: {self.source}")
        if self.authority:
            bits.append(f"authority: {self.authority}")
        if self.as_of:
            bits.append(f"as of {self.as_of}")
        return "[" + " | ".join(bits) + "]"

    def render(self) -> str:
        return f"{self.header()}\n{self.text.strip()}"


def approx_tokens(text: str) -> int:
    """A tokenizer-free estimate (about 3.5 characters a token on English prose and code)."""
    return max(1, int(len(text) / 3.5) + 1)


# ------------------------------------------------------------------------------------------------ segmentation
_BOUNDARY = re.compile(r"(?m)^(?:#{1,6}\s+\S.*|={3,}.*|-{3,}\s*|_{3,}\s*|\*{3,}\s*|(?:Document|File|Section|Part|"
                       r"Chapter|Article|Appendix|Exhibit|Schedule)\b[^\n]{0,120}:?\s*)$")


def _blocks(text: str) -> list[str]:
    """The text cut at its structure: headings, separator lines, document/file/section titles.  Never loses a
    character: the blocks joined are the text."""
    cuts = sorted({0, len(text)} | {m.start() for m in _BOUNDARY.finditer(text)})
    return [text[a:b] for a, b in zip(cuts, cuts[1:]) if text[a:b]]


def _split_big(block: str, max_tokens: int, count: Callable[[str], int]) -> list[str]:
    """A block above `max_tokens`: cut at blank lines, then lines, then sentences - whatever keeps pieces whole."""
    if count(block) <= max_tokens:
        return [block]
    for pat in (r"\n\s*\n", r"\n", r"(?<=[.!?])\s+"):
        parts = re.split(f"({pat})", block)
        # re-attach the separators to the piece before them, so nothing is lost
        pieces = [x for x in ("".join(parts[i:i + 2]) for i in range(0, len(parts), 2)) if x]
        if len(pieces) > 1:
            out, cur = [], ""
            for p in pieces:
                if cur and count(cur + p) > max_tokens:
                    out.append(cur)
                    cur = p
                else:
                    cur += p
            if cur:
                out.append(cur)
            if len(out) < 2:              # nothing to cut at with this separator after all: the next one
                continue
            res = []
            for o in out:
                res.extend(_split_big(o, max_tokens, count) if count(o) > max_tokens else [o])
            return res
    # one unbroken run of text: cut by characters
    step = max(1, int(len(block) * max_tokens / max(1, count(block))))
    return [block[i:i + step] for i in range(0, len(block), step)]


def _title(text: str) -> str:
    for line in text.splitlines():
        s = line.strip().strip("#=-_* ").strip()
        if s:
            return s[:90]
    return ""


def _level(piece: str) -> int:
    """How high a section boundary is: 1-6 for a markdown heading, 1 for a separator line or a document / file /
    section title, 99 for no boundary."""
    m = re.match(r"(#{1,6})\s", piece)
    if m:
        return len(m.group(1))
    return 1 if _BOUNDARY.match(piece) else 99


def _pack(pieces: list, target_tokens: int, count: Callable[[str], int]) -> list[str]:
    """Pieces (text, starts a section) packed into chunks of about target_tokens, cut at section starts once a
    chunk is a third full."""
    chunks: list[str] = []
    cur = ""
    for p, starts in pieces:
        if cur and (count(cur + p) > target_tokens or (starts and count(cur) >= target_tokens // 3)):
            chunks.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        chunks.append(cur)
    return chunks


def segment(text: str, target_tokens: int = 1500, count: Callable[[str], int] = approx_tokens,
            prefix: str = "c", start: int = 1) -> list[Item]:
    """Cut a long text into chunks of about `target_tokens` at its own structure: whole top-level sections (a
    contract, a report, a file) where they fit, several small ones sharing a chunk; a section larger than a chunk is
    cut at its own sub-sections.  A top-level section is never split between chunks when it fits in one, so the
    facts of one document stay together.  Ids are `<prefix><n>` in reading order."""
    pieces: list[tuple[str, bool, int]] = []     # (text, starts one of the text's own sections, its level)
    for b in _blocks(text):
        for i, p in enumerate(_split_big(b, target_tokens, count)):
            st = i == 0 and bool(_BOUNDARY.match(p))
            pieces.append((p, st, _level(p) if st else 99))
    levels = [lv for _, st, lv in pieces if st]
    top = min((lv for lv in set(levels) if levels.count(lv) >= 2), default=None)
    chunks: list[str] = []
    if top is None:
        chunks = _pack([(p, st) for p, st, _ in pieces], target_tokens, count)
    else:
        sections: list[list] = []
        for p, st, lv in pieces:
            if not sections or (st and lv <= top):
                sections.append([])
            sections[-1].append((p, st))
        cur = ""
        for sec in sections:
            stext = "".join(p for p, _ in sec)
            if count(stext) > target_tokens:     # too big for one chunk: its own sub-sections
                if cur:
                    chunks.append(cur)
                    cur = ""
                chunks += _pack(sec, target_tokens, count)
            elif cur and count(cur + stext) > target_tokens:
                chunks.append(cur)
                cur = stext
            else:
                cur += stext
        if cur:
            chunks.append(cur)
    items = []
    for i, c in enumerate(chunks):
        if not c.strip():
            continue
        items.append(Item(id=f"{prefix}{start + len(items)}", text=c, title=_title(c), tokens=count(c),
                          order=start + i))
    return items


def head_tail(text: str, tokens: int = 300, count: Callable[[str], int] = approx_tokens) -> tuple[str, str]:
    """The start and the end of a long message (where a request usually says what it wants), about `tokens` each,
    cut at line boundaries."""
    lines = text.splitlines(keepends=True)
    head, n = "", 0
    for ln in lines:
        if n and n + count(ln) > tokens:
            break
        head += ln
        n += count(ln)
    tail, n = "", 0
    for ln in reversed(lines):
        if n and n + count(ln) > tokens:
            break
        tail = ln + tail
        n += count(ln)
    return head, tail


# ------------------------------------------------------------------------------------------------ the index
def index_text(items: list, words: int = 14) -> str:
    """One line per item: id, title, size, and its first words - what a planner reads instead of the material."""
    out = []
    for it in items:
        body = re.sub(r"\s+", " ", it.text).strip()
        first = " ".join(body.split()[:words])
        meta = [f"{it.tokens:,} tokens"]
        if it.kind != "document":
            meta.append(it.kind)
        if it.source:
            meta.append(it.source[:60])
        out.append(f"[{it.id}] {it.title[:80]} ({', '.join(meta)}): {first}...")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ retrieval
_WORD = re.compile(r"[A-Za-z0-9_]+(?:[.'-][A-Za-z0-9_]+)*")
_STOP = frozenset("""a an and are as at be but by for from has have how i if in into is it its of on or that the
their them then there these they this to was were what when where which who why will with you your do does did can
could should would may might must not no yes than so such also any all each every about above below over under
between more most other some only own same few both very just""".split())


def terms(text: str) -> list[str]:
    return [w for w in (t.lower() for t in _WORD.findall(text or "")) if w not in _STOP and len(w) > 1]


class Bm25:
    """BM25 over items (k1 1.2, b 0.75): lexical relevance, deterministic, no model."""

    def __init__(self, items: list):
        self.items = list(items)
        self.tf = []
        self.df: dict[str, int] = {}
        for it in self.items:
            c: dict[str, int] = {}
            for w in terms(it.title + " " + it.text):
                c[w] = c.get(w, 0) + 1
            self.tf.append(c)
            for w in c:
                self.df[w] = self.df.get(w, 0) + 1
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0

    def scores(self, query: str) -> list[float]:
        q = set(terms(query))
        n = len(self.items)
        out = []
        for c, ln in zip(self.tf, self.len):
            s = 0.0
            for w in q:
                f = c.get(w)
                if not f:
                    continue
                idf = math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * ln / max(1e-9, self.avg)))
            out.append(s)
        return out

    def search(self, query: str, k: int = 5, exclude: set | None = None) -> list:
        sc = self.scores(query)
        ranked = sorted(range(len(self.items)), key=lambda i: -sc[i])
        out = []
        for i in ranked:
            if sc[i] <= 0 or len(out) >= k:
                break
            if exclude and self.items[i].id in exclude:
                continue
            out.append(self.items[i])
        return out


def pack(items: list, budget: int) -> list:
    """The items in order until the budget is spent (an item that does not fit is skipped, a smaller later one may)."""
    out, used = [], 0
    for it in items:
        if used + it.tokens <= budget:
            out.append(it)
            used += it.tokens
    return out


def dedupe(items: list) -> list:
    """Items with the same id or the same normalized text, kept once (the first)."""
    seen_id, seen_txt, out = set(), set(), []
    for it in items:
        h = hashlib.sha1(re.sub(r"\s+", " ", it.text.strip().lower()).encode()).hexdigest()
        if it.id in seen_id or h in seen_txt:
            continue
        seen_id.add(it.id)
        seen_txt.add(h)
        out.append(it)
    return out


# ------------------------------------------------------------------------------------------------ providers
class ContextProvider(Protocol):
    """A memory / retrieval backend.  Called ONCE per parent request (and, if the server enables it, for at most a
    few deduplicated follow-up questions); it returns items, never instructions."""

    def retrieve(self, query: str, k: int, budget_tokens: int) -> list: ...


def items_from_json(raw, default_kind: str = "document", prefix: str = "d",
                    count: Callable[[str], int] = approx_tokens) -> list:
    """`context_items` of a request or a provider's answer -> Items.  Accepts strings or objects with text/content
    and the optional id, title, source, kind, authority, as_of, pinned, score.  Bad entries are skipped."""
    out = []
    for i, r in enumerate(raw or [], 1):
        if isinstance(r, str):
            r = {"text": r}
        if not isinstance(r, dict):
            continue
        text = r.get("text", r.get("content"))
        if not isinstance(text, str) or not text.strip():
            continue
        kind = str(r.get("kind") or default_kind).lower()
        kind = kind if kind in KINDS else default_kind
        rid = re.sub(r"[^A-Za-z0-9_.:-]", "", str(r.get("id") or ""))[:40]
        out.append(Item(id=f"{prefix}{i}" if not rid else f"{prefix}{i}:{rid}"[:48], text=text,
                        title=str(r.get("title") or "")[:120] or _title(text), source=str(r.get("source") or "")[:200],
                        kind=kind, authority=str(r.get("authority") or "")[:40], as_of=str(r.get("as_of") or "")[:40],
                        tokens=count(text), order=10_000 + i, pinned=r.get("pinned") is True,
                        score=float(r["score"]) if isinstance(r.get("score"), (int, float)) else 0.0))
    return out


@dataclass
class HttpContextProvider:
    """A generic JSON retrieval endpoint: POST {"query", "k", "budget_tokens"} -> {"items": [...]} (the item format of
    `items_from_json`; "kind" defaults to "memory").  Any error is no items: memory augments a request, it never
    blocks one."""
    url: str
    timeout_s: float = 5.0
    headers: dict = field(default_factory=dict)
    last_error: str = ""

    def retrieve(self, query: str, k: int, budget_tokens: int) -> list:
        self.last_error = ""
        body = json.dumps({"query": query, "k": k, "budget_tokens": budget_tokens}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json", **self.headers})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                d = json.loads(r.read() or b"{}")
        except Exception as e:  # noqa: BLE001 - a provider's failure is reported, not raised
            self.last_error = f"{type(e).__name__}: {e}"
            return []
        return items_from_json(d.get("items") if isinstance(d, dict) else None, "memory", "m")
