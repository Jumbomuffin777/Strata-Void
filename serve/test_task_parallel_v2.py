"""serve/test_task_parallel_v2.py - task-parallel v2 without a GPU: large-context partitioning, the bounded follow-up
round, context items / a retrieval provider, and the answer written in sections at the same time.

    python -m unittest serve.test_task_parallel_v2 -v
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import context_pool as cp  # noqa: E402
from serve import task_parallel as tp  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REQ = "Review this contract. For each section, list the risks for the buyer and recommend changes."


def contract(n_sections=12, words=400):
    """A long document with headed sections; section k mentions a unique fact 'clause-k-fact'."""
    out = [REQ, ""]
    for k in range(1, n_sections + 1):
        body = " ".join(f"term{k}x{i}" for i in range(words))
        out.append(f"## Section {k}: Topic {k}\n\nThe clause-{k}-fact is {k * 111}. {body}\n")
    out.append("\nAnswer with a recommendation at the end.")
    return "\n".join(out)


def plan(n, strategy="partition", sources=None, **over):
    st = [{"objective": f"Assess part {i} of the request thoroughly", "max_tokens": 200} for i in range(1, n + 1)]
    if sources:
        for s, src in zip(st, sources):
            s["sources"] = src
    d = {"effort": "high", "parallelism": n, "strategy": strategy, "subtasks": st}
    d.update(over)
    return json.dumps(d)


OUTLINE = json.dumps({"sections": [{"heading": "Summary", "covers": "the answer in short", "words": 150},
                                   {"heading": "Risks", "covers": "the risks per section", "words": 300},
                                   {"heading": "Recommendation", "covers": "what to change", "words": 200}],
                      "facts": ["clause-3-fact is 333"]})


class RoleBackend:
    """Answers by role.  Writers stream their section through call.on_text in small pieces."""

    def __init__(self, plan_text, outline=OUTLINE, slots=6, ctx=65536, worker=None, writer=None, delay=0.02):
        self.plan_text, self.outline, self.slots, self.ctx, self.delay = plan_text, outline, slots, ctx, delay
        self.worker, self.writer = worker, writer
        self.calls, self.lock, self.running, self.peak = [], threading.Lock(), 0, 0

    def concurrency(self):
        return self.slots

    def context(self):
        return self.ctx

    def count_tokens(self, call):
        return sum(len(m["content"]) for m in call.messages) // 4

    def generate(self, call, cancel):
        with self.lock:
            self.calls.append(call)
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            t0 = time.time()
            last = call.messages[-1]["content"]
            if call.role == "planner":
                return tp.Result(text=self.plan_text, finish="stop", completion_tokens=80, t_start=t0, t_end=time.time())
            time.sleep(self.delay)
            if call.role == "worker":
                r = self.worker(call, cancel) if self.worker else tp.Result(
                    text="finding " + last.split("STEP subtask ")[1][:3], finish="stop", completion_tokens=50)
            elif call.role == "followup":
                r = tp.Result(text="the follow-up answer [c2]", finish="stop", completion_tokens=20)
            elif call.role == "outline":
                r = tp.Result(text=self.outline, finish="stop", completion_tokens=90)
            elif call.role == "writer":
                k = int(last.split("STEP write section ")[1].split("/")[0])
                head = call.messages[-1]["content"].split('start with the line \'')[1].split("'")[0]
                text = self.writer(call, k) if self.writer else f"{head}\n\nBody of section {k}, first part.\n\nBody of section {k}, second part.\n"
                if isinstance(text, Exception):
                    raise text
                for i in range(0, len(text), 7):
                    if cancel.is_set():
                        return tp.Result(text=text[:i], finish="cancel", completion_tokens=i // 4)
                    if call.on_text:
                        call.on_text(text[i:i + 7])
                    time.sleep(0.001 * (4 - k) if k < 4 else 0)
                r = tp.Result(text=text, finish="stop", completion_tokens=len(text) // 4)
            else:
                r = tp.Result(text="final", finish="stop", completion_tokens=10)
            if isinstance(r, Exception):
                raise r
            r.t_start, r.t_end = t0, time.time()
            return r
        finally:
            with self.lock:
                self.running -= 1


def run(backend, text, mode=4, cfg=None, **kw):
    return tp.Orchestrator(backend, cfg or tp.Config()).run(
        [{"role": "system", "content": "Be precise."}, {"role": "user", "content": text}], tp.parse_mode(mode),
        threading.Event(), **kw)


class ContextPool(unittest.TestCase):
    def test_segment_keeps_every_character_and_structure(self):
        text = contract(6, 300)
        items = cp.segment(text, 500)
        self.assertEqual("".join(i.text for i in items), text)
        self.assertTrue(all(i.tokens <= 500 for i in items))
        self.assertEqual([i.id for i in items], [f"c{k}" for k in range(1, len(items) + 1)])
        self.assertTrue(any(i.title.startswith("Section 3") for i in items))

    def test_a_document_that_fits_is_never_split(self):
        docs = "".join(f"# Contract {k}\n\n## Parties\nVendor-{k} and the Customer.\n\n## Fees\nfee-{k} " +
                       "words " * 150 + f"\n\n## Liability\ncap-{k} " + "terms " * 150 + "\n\n" for k in range(1, 9))
        items = cp.segment(docs, 800)                            # one contract: ~550 tokens
        self.assertEqual("".join(i.text for i in items), docs)
        self.assertGreater(len(items), 4)
        for k in range(1, 9):
            holding = [i for i in items if f"Vendor-{k} " in i.text or f"fee-{k} " in i.text or f"cap-{k} " in i.text]
            self.assertEqual(len(holding), 1, k)                 # name, fees and cap of one contract in one chunk

    def test_unbroken_text_is_cut(self):
        items = cp.segment("x" * 20000, 400)
        self.assertEqual("".join(i.text for i in items), "x" * 20000)
        self.assertTrue(all(i.tokens <= 400 for i in items))

    def test_bm25_finds_the_section(self):
        items = cp.segment(contract(8, 200), 300)
        top = cp.Bm25(items).search("clause-5-fact", 1)
        self.assertIn("clause-5-fact", top[0].text)

    def test_items_from_json_and_dedupe(self):
        raw = ["plain text", {"text": "a memory", "kind": "memory", "source": "store/7", "authority": "verified"},
               {"content": "live load 3", "kind": "live", "as_of": "12:00"}, {"text": ""}, 7, {"text": "plain text"}]
        items = cp.dedupe(cp.items_from_json(raw, prefix="m"))
        self.assertEqual([i.kind for i in items], ["document", "memory", "live"])
        self.assertIn("authority: verified", items[1].header())
        self.assertIn("as of 12:00", items[2].header())

    def test_pack(self):
        items = [cp.Item(id=f"i{k}", text="x", tokens=t) for k, t in enumerate((50, 80, 30, 40))]
        self.assertEqual([i.id for i in cp.pack(items, 120)], ["i0", "i2", "i3"])


class Partitioning(unittest.TestCase):
    cfg = tp.Config(share_context=0, partition_min_tokens=2000, chunk_tokens=600, worker_source_tokens=6000, compose="synthesis")

    def test_small_material_is_not_partitioned(self):
        m = tp.build_material([{"role": "user", "content": "short request"}], self.cfg)
        self.assertFalse(m.partitioned)

    def test_large_material_is_partitioned_and_cut_out_of_the_request(self):
        text = contract(12, 400)
        m = tp.build_material([{"role": "user", "content": text}], self.cfg)
        self.assertTrue(m.partitioned)
        self.assertGreater(len(m.assignable), 4)
        self.assertLess(len(m.request_view), len(text) / 5)
        self.assertIn(REQ, m.request_view)                      # the instructions stay
        shared = tp.shared_context([{"role": "user", "content": text}], m)
        self.assertIn("INDEX of the material", shared)
        self.assertNotIn("term7x100 ", shared)                  # the material itself is not in the shared prefix

    def test_shard_reads_everything_once(self):
        b = RoleBackend(plan(4, "shard"))
        out = run(b, contract(12, 400), 4, self.cfg)
        self.assertEqual(out.kind, "synthesize")
        cov = out.meta["sources"]
        self.assertEqual(cov["coverage"], 1.0)
        self.assertEqual(cov["duplicated_tokens"], 0)
        workers = [c for c in b.calls if c.role == "worker"]
        self.assertEqual(len(workers), 4)
        seen = [set(c.messages[-1]["content"].split("clause-")[1:]) for c in workers]
        self.assertTrue(all(seen))                               # every worker got material
        joined = "".join(c.messages[-1]["content"] for c in workers)
        for k in range(1, 13):
            self.assertEqual(joined.count(f"clause-{k}-fact"), 1, k)   # each section read exactly once

    def test_named_sources_then_coverage(self):
        m = tp.build_material([{"role": "user", "content": contract(12, 400)}], self.cfg)
        ids = [i.id for i in m.assignable]
        b = RoleBackend(plan(3, sources=[[ids[0]], [ids[1], "bogus"], []]))
        out = run(b, contract(12, 400), 3, self.cfg)
        workers = {int(re.search(r"STEP subtask (\d+)/", c.messages[-1]["content"]).group(1)): c.messages[-1]["content"]
                   for c in b.calls if c.role == "worker"}         # threads: any order
        self.assertIn(f"[{ids[0]}", workers[1])
        self.assertIn(f"[{ids[1]}", workers[2])
        self.assertEqual(out.meta["sources"]["coverage"], 1.0)

    def test_budget_bounds_a_worker(self):
        cfg = tp.Config(share_context=0, partition_min_tokens=2000, chunk_tokens=600, worker_source_tokens=1500, compose="synthesis")
        b = RoleBackend(plan(2, "shard"))
        out = run(b, contract(12, 400), 2, cfg)
        self.assertTrue(all(t <= 1500 for t in out.meta["sources"]["source_tokens"]))
        self.assertLess(out.meta["sources"]["coverage"], 1.0)

    def test_needs_get_one_bounded_follow_up_round(self):
        def worker(call, cancel):
            return tp.Result(text="partial finding\nNEED: clause-11-fact value\nNEED: x\nNEED: y", finish="stop",
                             completion_tokens=30)
        b = RoleBackend(plan(2, "shard"), worker=worker)
        cfg = tp.Config(share_context=0, partition_min_tokens=2000, chunk_tokens=600, worker_source_tokens=3000, compose="synthesis",
                        need_per_worker=2, need_total=3)
        out = run(b, contract(12, 400), 2, cfg)
        f = out.meta["followups"]
        self.assertEqual(f["asked"], 3)                          # 2 + 1: per worker and in all, bounded
        notes = out.synthesis.messages[-1]["content"]
        self.assertNotIn("NEED:", notes)                         # the requests are not passed on as notes
        self.assertIn("the follow-up answer", notes)
        ups = [c for c in b.calls if c.role == "followup"]
        self.assertTrue(all("STEP subtask" in c.messages[-1]["content"] for c in ups))
        self.assertFalse(any(c.role == "planner" for c in b.calls[1:]))   # no recursion: one plan

    def test_auto_large_context_passes_the_gate_and_caps_workers(self):
        cfg = tp.Config(share_context=0, partition_min_tokens=2000, chunk_tokens=600, worker_source_tokens=3000, min_shard_tokens=2000,
                        compose="synthesis")
        b = RoleBackend(plan(6, "shard"))
        out = run(b, contract(12, 400), "auto", cfg)
        self.assertTrue(out.meta["gate"]["score"] >= cfg.gate_threshold)
        self.assertLessEqual(out.meta["workers"], max(2, out.meta["context"]["material_tokens"] // 2000))

    def test_slot_context_bounds_the_sources(self):
        """--slot-context: a subtask longer than a slot would run alone; its sources stay within a slot."""
        b = RoleBackend(plan(2, "shard"))
        b.slot_context = lambda: 4000
        out = run(b, contract(12, 400), 2, self.cfg)
        workers = [c for c in b.calls if c.role == "worker"]
        self.assertTrue(workers)
        self.assertTrue(all(b.count_tokens(c) + c.max_tokens <= 4000 for c in workers),
                        [b.count_tokens(c) for c in workers])
        self.assertLess(out.meta["sources"]["coverage"], 1.0)

    def test_auto_backs_off_when_the_slots_are_busy(self):
        b = RoleBackend(plan(4, "shard"))
        b.busy = lambda: 5
        out = run(b, contract(12, 400), "auto", self.cfg)
        self.assertEqual(out.kind, "direct")
        self.assertIn("busy", out.meta["decision"])
        self.assertFalse(b.calls)                                 # not even a plan

    def test_auto_uses_fewer_free_slots(self):
        b = RoleBackend(plan(6, "shard"))
        b.busy = lambda: 3
        cfg = tp.Config(share_context=0, partition_min_tokens=2000, chunk_tokens=600, worker_source_tokens=6000, min_shard_tokens=500,
                        compose="synthesis")
        out = run(b, contract(12, 400), "auto", cfg)
        self.assertLessEqual(out.meta["workers"], 3)

    def test_auto_estimate_says_alone_is_faster(self):
        cfg = tp.Config(share_context=0, partition_min_tokens=2000, chunk_tokens=600, worker_source_tokens=6000, compose="synthesis",
                        admission_s=60.0)
        b = RoleBackend(plan(4, "shard"))
        out = run(b, contract(12, 400), "auto", cfg)
        self.assertEqual(out.kind, "direct")
        self.assertIn("estimated", out.meta["decision"])
        self.assertIn("single_s", out.meta["estimate"])
        # a fixed number of workers is the caller's choice: the estimate is reported, not applied
        out = run(RoleBackend(plan(4, "shard")), contract(12, 400), 4, cfg)
        self.assertEqual(out.kind, "synthesize")
        self.assertIn("parallel_s", out.meta["estimate"])


    def test_material_that_fits_a_slot_is_shared_whole(self):
        """share_context: material a slot can hold stays in the shared prefix: the planner reads it once, every step
        restores it (one prefix for all) and sees all of it - no partition, no index."""
        b = RoleBackend(plan(3))
        b.slot_context = lambda: 32768
        text = contract(12, 400)
        out = run(b, text, 3, tp.Config(partition_min_tokens=2000, chunk_tokens=600, compose="synthesis"))
        self.assertFalse(out.meta.get("context", {}).get("partitioned"))
        workers = [c for c in b.calls if c.role == "worker"]
        self.assertEqual(len(workers), 3)
        for c in workers:
            self.assertIn("clause-11-fact", c.shared_prefix[0]["content"])   # the whole material, in the prefix
            self.assertEqual(c.shared_prefix, workers[0].shared_prefix)
        b2 = RoleBackend(plan(3, "shard"))
        b2.slot_context = lambda: 4096                           # does not fit a slot: partitioned
        out = run(b2, text, 3, tp.Config(partition_min_tokens=2000, chunk_tokens=600, worker_source_tokens=6000))
        self.assertTrue(out.meta["context"]["partitioned"])

    def test_auto_answers_hard_requests_over_shared_material_in_one_pass(self):
        """AUTO + large material kept whole: a "high"/"medium" request is answered by one stream that restores what
        the planner read (no second read, no parallel sections); a "low" one is written in parallel sections."""
        text = contract(12, 400)
        b = RoleBackend(plan(3))                                 # effort "high"
        b.slot_context = lambda: 32768
        out = run(b, text, "auto", tp.Config(compose="auto"))
        self.assertEqual(out.kind, "synthesize")
        self.assertEqual(out.meta["compose"], "one answer, shared prefix")
        self.assertIn("clause-11-fact", out.synthesis.shared_prefix[0]["content"])
        self.assertEqual(out.synthesis.messages[:1], out.synthesis.shared_prefix)
        low = json.dumps({"effort": "low", "parallelism": 3, "strategy": "partition", "subtasks": [
            {"heading": f"Part {i}", "objective": f"The findings for part {i} of the request", "words": 150}
            for i in (1, 2, 3)]})
        b = RoleBackend(low)
        b.slot_context = lambda: 32768
        out = run(b, text, "auto", tp.Config(compose="auto"))
        self.assertEqual(out.kind, "composed")
        "".join(out.composer.stream())
        out.composer.finish()

    def test_an_unusable_plan_is_asked_for_once_more(self):
        replies = ["Sure! Here is my plan: parallelism three", plan(3, "shard")]

        class Twice(RoleBackend):
            def generate(self, call, cancel):
                if call.role == "planner":
                    with self.lock:
                        self.calls.append(call)
                    return tp.Result(text=replies.pop(0), finish="stop", completion_tokens=40)
                return super().generate(call, cancel)
        b = Twice("")
        out = run(b, contract(12, 400), 3, self.cfg)
        self.assertEqual(out.kind, "synthesize")
        self.assertIn("plan_retry", out.meta)
        planners = [c for c in b.calls if c.role == "planner"]
        self.assertEqual(len(planners), 2)
        self.assertIn("was not usable", planners[1].messages[-1]["content"])
        self.assertFalse(out.meta.get("shard_fallback"))

    def test_large_material_is_sharded_when_the_planner_says_one(self):
        for text in (json.dumps({"effort": "high", "parallelism": 1, "strategy": "partition", "subtasks": ["all"]}),
                     "not a plan"):
            b = RoleBackend(text)
            out = run(b, contract(12, 400), "auto", self.cfg)
            self.assertEqual(out.kind, "synthesize")
            self.assertTrue(out.meta["shard_fallback"])
            self.assertEqual(out.meta["strategy"], "shard")
            self.assertEqual(out.meta["sources"]["coverage"], 1.0)

    def test_a_plan_cut_by_its_budget_keeps_its_whole_subtasks(self):
        cut = plan(4, sources=[["c1"], ["c2"], ["c3"], ["c4"]])
        cut = cut[:cut.index('{"objective": "Assess part 4')] + '{"objective": "Assess pa'
        b = RoleBackend(cut)
        out = run(b, contract(12, 400), "auto", self.cfg)
        self.assertEqual(out.meta["workers"], 3)
        planner = [c for c in b.calls if c.role == "planner"][0]
        self.assertGreater(planner.max_tokens, tp.Config().planner_tokens)   # sources lists get a larger budget


class Providers(unittest.TestCase):
    def test_one_retrieval_shared_by_every_step(self):
        class Prov:
            calls = 0

            def retrieve(self, query, k, budget):
                Prov.calls += 1
                return [cp.Item(id="m1", text="The user prefers metric units.", kind="memory", tokens=8,
                                source="mem/1", authority="user"),
                        cp.Item(id="m2", text="GPU load 40%", kind="live", as_of="10:00", tokens=5)]
        b = RoleBackend(plan(2))
        out = run(b, "Compare three databases for analytics, list the risks of each and recommend one. " * 3, 2,
                  tp.Config(compose="synthesis"), provider=Prov())
        self.assertEqual(Prov.calls, 1)
        self.assertEqual(out.meta["context"]["provider"]["items"], 2)
        for c in b.calls:
            self.assertIn("metric units", c.messages[0]["content"])   # in the shared prefix, once for all
            self.assertIn("live", c.messages[0]["content"])

    def test_a_failing_provider_never_blocks(self):
        class Bad:
            def retrieve(self, query, k, budget):
                raise RuntimeError("down")
        b = RoleBackend(plan(2))
        out = run(b, "Compare three databases for analytics, list the risks of each and recommend one. " * 3, 2,
                  tp.Config(compose="synthesis"), provider=Bad())
        self.assertEqual(out.kind, "synthesize")
        self.assertIn("error", out.meta["context"]["provider"])

    def test_http_provider_error_is_no_items(self):
        p = cp.HttpContextProvider("http://127.0.0.1:9/none", timeout_s=0.5)
        self.assertEqual(p.retrieve("q", 3, 100), [])
        self.assertTrue(p.last_error)


class Compose(unittest.TestCase):
    cfg = tp.Config(compose="sections")

    def test_sections_written_at_once_and_streamed_in_order(self):
        b = RoleBackend(plan(3))
        out = run(b, "Compare three databases for analytics, list the risks of each and recommend one. " * 3, 3, self.cfg)
        self.assertEqual(out.kind, "composed")
        text = "".join(out.composer.stream())
        m = tp.finish_meta(out.meta, time.time(), 0, 0, composed=out.composer.finish())
        heads = [ln for ln in text.splitlines() if ln.startswith("## ")]
        self.assertEqual(heads, ["## Summary", "## Risks", "## Recommendation"])
        self.assertIn("Body of section 2, second part.", text)
        self.assertEqual(m["compose"], "sections")
        self.assertEqual(len(m["writer_results"]), 3)
        writers = [c for c in b.calls if c.role == "writer"]
        self.assertEqual(len(writers), 3)
        self.assertGreaterEqual(b.peak, 3)                       # the sections ran at the same time
        common = writers[0].messages[-1]["content"].split("STEP write section")[0]
        self.assertTrue(all(c.messages[-1]["content"].startswith(common) for c in writers))   # a shared prefix
        self.assertIn("clause-3-fact is 333", common)            # the facts every section uses

    def test_a_heading_is_added_and_duplicates_are_dropped(self):
        dup = "The same long paragraph about the risks of this design repeated word for word across two sections."

        def writer(call, k):
            if k == 1:
                return f"No heading here.\n\n{dup}\n"
            return f"## Whatever\n\n{dup}\n\nOwn text {k}.\n"
        b = RoleBackend(plan(3), writer=writer)
        out = run(b, "Compare three databases for analytics, list the risks of each and recommend one. " * 3, 3, self.cfg)
        text = "".join(out.composer.stream())
        self.assertTrue(text.startswith("## Summary\n\nNo heading here."))
        self.assertEqual(text.count(dup), 1)
        self.assertIn("## Risks", text)
        self.assertEqual(out.composer.finish()["paragraphs_dropped"], 2)

    def test_bad_outline_falls_back_to_synthesis(self):
        for bad in ("not json", json.dumps({"sections": [{"heading": "Only one"}]}),
                    json.dumps({"sections": [{"heading": "A"}, {"heading": "A"}]})):
            b = RoleBackend(plan(3), outline=bad)
            out = run(b, "Compare three databases for analytics, list the risks of each and recommend one. " * 3, 3,
                      self.cfg)
            self.assertEqual(out.kind, "synthesize", bad)
            self.assertIn("outline", out.meta["compose"])

    def test_auto_short_answer_uses_one_synthesis(self):
        short = json.dumps({"sections": [{"heading": "A", "covers": "a", "words": 120},
                                         {"heading": "B", "covers": "b", "words": 120}], "facts": []})
        b = RoleBackend(plan(3, "shard"), outline=short)
        out = run(b, contract(12, 400), 3, tp.Config(compose="auto", compose_min_words=500, share_context=0, partition_min_tokens=2000,
                                                     chunk_tokens=600, worker_source_tokens=6000))
        self.assertEqual(out.kind, "synthesize")                 # large material: notes over slices, one synthesis
        self.assertTrue(out.meta["context"]["partitioned"])

    def test_auto_without_material_writes_notes_then_the_answer(self):
        """compose "auto" without large material: notes, then the answer from them (sections only for a long one) -
        never sections written blind of each other (measured: they contradicted each other)."""
        short = json.dumps({"sections": [{"heading": "A", "covers": "a", "words": 120},
                                         {"heading": "B", "covers": "b", "words": 120}], "facts": []})
        b = RoleBackend(plan(3), outline=short)
        out = run(b, "Compare three databases for analytics, list the risks of each and recommend one. " * 3, 3,
                  tp.Config(compose="auto"))
        self.assertEqual(out.kind, "synthesize")
        self.assertTrue(any(c.role == "worker" for c in b.calls))
        self.assertFalse(any(c.role == "writer" for c in b.calls))

    def test_explicit_direct_without_material(self):
        b = RoleBackend(json.dumps({"effort": "high", "parallelism": 3, "strategy": "partition", "subtasks": [
            {"heading": f"Part {i}", "objective": f"Part {i} of the answer", "words": 200} for i in (1, 2, 3)]}))
        out = run(b, "Compare three databases for analytics, list the risks of each and recommend one. " * 3, 3,
                  tp.Config(compose="direct"))
        self.assertEqual(out.kind, "composed")
        self.assertEqual(out.meta["compose"], "direct")
        "".join(out.composer.stream())
        out.composer.finish()

    def test_a_failed_section_is_retried_then_left_out(self):
        tries = {}

        def writer(call, k):
            tries[k] = tries.get(k, 0) + 1
            return RuntimeError("boom") if k == 2 else f"## X\n\nSection {k} text.\n"
        b = RoleBackend(plan(3), writer=writer)
        out = run(b, "Compare three databases for analytics, list the risks of each and recommend one. " * 3, 3, self.cfg)
        text = "".join(out.composer.stream())
        self.assertEqual(tries[2], 2)
        self.assertNotIn("## Risks", text)
        self.assertIn("Section 3 text.", text)
        self.assertEqual(out.meta["sections_missing"], ["Risks"])

    def test_cancel_stops_the_writers(self):
        def writer(call, k):
            return f"## H\n\n" + ("word " * 4000)
        b = RoleBackend(plan(3), writer=writer)
        cancel = threading.Event()
        out = tp.Orchestrator(b, self.cfg).run(
            [{"role": "user", "content": "Compare three databases for analytics, list the risks of each and recommend "
                                         "one. " * 3}], tp.parse_mode(3), cancel)
        it = out.composer.stream()
        next(it)
        cancel.set()
        rest = list(it)
        for t in out.composer.threads:
            t.join(5)
        self.assertFalse(any(t.is_alive() for t in out.composer.threads))
        self.assertLess(len("".join(rest)), 20000)


class Direct(unittest.TestCase):
    """compose "direct": the plan is the answer's outline; sections are written at once, closing ones after."""
    cfg = tp.Config(compose="direct")

    def plan(self, **over):
        st = [{"heading": "Consistency", "objective": "Compare the consistency models", "words": 200},
              {"heading": "Scaling", "objective": "Compare how each scales", "words": 200},
              {"heading": "Costs", "objective": "Compare operating costs", "words": 200},
              {"heading": "Recommendation", "objective": "Recommend one, from the comparison", "words": 150,
               "after": True}]
        d = {"effort": "high", "parallelism": 4, "strategy": "partition", "subtasks": st}
        d.update(over)
        return json.dumps(d)

    def test_sections_then_closing(self):
        b = RoleBackend(self.plan())
        out = run(b, "Compare three databases for analytics, list the risks of each and recommend one. " * 3, "auto",
                  self.cfg)
        self.assertEqual(out.kind, "composed")
        text = "".join(out.composer.stream())
        heads = [ln for ln in text.splitlines() if ln.startswith("## ")]
        self.assertEqual(heads, ["## Consistency", "## Scaling", "## Costs", "## Recommendation"])
        m = tp.finish_meta(out.meta, time.time(), 0, 0, composed=out.composer.finish())
        self.assertEqual((m["compose"], m["sections"], m["closing_sections"], m["workers"]), ("direct", 4, 1, 3))
        writers = [c for c in b.calls if c.role == "writer"]
        self.assertEqual(len(writers), 4)
        self.assertFalse(any(c.role in ("worker", "outline", "synthesis") for c in b.calls))   # one parallel stage
        closing = writers[-1].messages[-1]["content"]
        self.assertIn("THE SECTIONS ALREADY WRITTEN", closing)
        self.assertIn("Body of section 2, second part.", closing)
        self.assertTrue(m["writer_results"][-1]["closing"])
        self.assertGreaterEqual(m["closing_start_ms"], 0)

    def test_one_section_is_the_ordinary_request(self):
        b = RoleBackend(json.dumps({"effort": "low", "parallelism": 1}))
        out = run(b, "Compare three databases for analytics, list the risks of each and recommend one. " * 3, "auto",
                  self.cfg)
        self.assertEqual(out.kind, "direct")

    def test_partitioned_sections_read_their_sources(self):
        cfg = tp.Config(compose="direct", share_context=0, partition_min_tokens=2000, chunk_tokens=600,
                        worker_source_tokens=6000)
        m = tp.build_material([{"role": "user", "content": contract(12, 400)}], cfg)
        ids = [i.id for i in m.assignable]
        st = [{"heading": f"Part {k}", "objective": f"Risks in part {k}", "words": 200,
               "sources": ids[(k - 1) * len(ids) // 3: k * len(ids) // 3]} for k in (1, 2, 3)]
        st.append({"heading": "Overall", "objective": "Overall recommendation", "words": 150, "after": True})
        b = RoleBackend(json.dumps({"effort": "high", "parallelism": 4, "strategy": "partition", "subtasks": st}))
        out = run(b, contract(12, 400), 4, cfg)
        "".join(out.composer.stream())
        writers = {int(re.search(r"write section (\d+)/", c.messages[-1]["content"]).group(1)): c.messages[-1]["content"]
                   for c in b.calls if c.role == "writer"}         # the writers run in threads: any order
        self.assertIn("SOURCES", writers[1])
        self.assertIn(f"[{ids[0]}", writers[1])
        self.assertNotIn("SOURCES", writers[4])               # the closing section reads the sections, not the material
        self.assertEqual(out.meta["sources"]["coverage"], 1.0)


# ------------------------------------------------------------------------------------------------ over HTTP
class RoleEngine(MockEngine):
    def __init__(self, tok, plan_text, **kw):
        super().__init__(tok, "x", **kw)
        self.plan, self.prompts = plan_text, []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        text = self.tok.decode(ids)
        self.prompts.append(text)
        think = text.endswith("<think>\n")
        last = text.rsplit("<|im_start|>user\n", 1)[-1]
        pre = "</think>\n\n" if think else ""
        if last.startswith("STEP plan"):
            reply = self.plan
        elif last.startswith("STEP subtask"):
            reply = pre + "WORK" + last.split("STEP subtask ")[1].split("/")[0]
        elif "STEP outline" in last:
            reply = pre + OUTLINE
        elif "STEP write section" in last:
            k = last.split("STEP write section ")[1].split("/")[0]
            head = last.split("start with the line '")[1].split("'")[0]
            reply = pre + f"{head}\n\nSECTION{k} text."
        elif last.startswith("STEP final answer"):
            reply = pre + "FINAL ANSWER"
        else:
            reply = pre + "PLAIN ANSWER"
        script = self.tok.encode(reply, parse_special=True) + self.tok.encode("<|im_end|>", parse_special=True)
        for t in script[:max_new]:          # thread-safe: the internal requests run at the same time
            if cancel.is_set():
                return
            yield t


class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = RoleEngine(cls.tok, plan(2), max_context=1 << 16)
        cls.engine.batch = 3                    # the slots: requests at the same time (the sections need >= 2)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.task_parallel_cfg = tp.Config(compose="sections")
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body, raw=False):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
                return r.status, (data.decode() if raw else json.loads(data))
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def body(self, **kw):
        return {"model": "m", "max_tokens": 64, "messages": [{"role": "user", "content": "Compare three databases for "
                "analytics, list the risks of each and recommend one. " * 3}], **kw}

    def test_composed_answer(self):
        s, b = self.post(self.body(task_parallel=2))
        self.assertEqual(s, 200, b)
        content = b["choices"][0]["message"]["content"]
        self.assertEqual([ln for ln in content.splitlines() if ln.startswith("## ")],
                         ["## Summary", "## Risks", "## Recommendation"])
        self.assertIn("SECTION2 text.", content)
        self.assertNotIn("WORK", content)
        m = b["task_parallel"]
        self.assertEqual((m["compose"], m["sections"]), ("sections", 3))
        for k in ("planning_ms", "workers_ms", "outline_ms", "writers_ms", "compose_ms", "total_ms"):
            self.assertIsInstance(m[k], int, k)

    def test_composed_stream(self):
        s, text = self.post(self.body(task_parallel=2, stream=True), raw=True)
        self.assertEqual(s, 200)
        self.assertIn(": task_parallel outlining", text)
        events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
        content = "".join((e["choices"][0]["delta"].get("content") or "") for e in events if e.get("choices"))
        self.assertTrue(content.startswith("## Summary"))
        self.assertTrue(events[-1]["task_parallel"]["enabled"])
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))

    def test_context_items_must_be_a_list(self):
        s, b = self.post(self.body(task_parallel=2, context_items="nope"))
        self.assertEqual(s, 400)

    def test_off_ignores_the_new_fields(self):
        s, b = self.post(self.body())
        self.assertEqual(b["choices"][0]["message"]["content"], "PLAIN ANSWER")
        self.assertNotIn("task_parallel", b)


if __name__ == "__main__":
    unittest.main()
