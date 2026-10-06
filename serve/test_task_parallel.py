"""serve/test_task_parallel.py - task-parallel requests (serve/task_parallel.py) without a GPU.

    python -m unittest serve.test_task_parallel -v

The orchestration is tested against a fake Backend (concurrency, failures, timeouts, cancellation, budgets), and the
HTTP path against a mock engine that answers by role (planner JSON, subtask work, the final answer)."""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import task_parallel as tp  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BIG = ("Compare PostgreSQL, MongoDB and Cassandra for an order-tracking service. Analyze consistency, scaling, "
       "query flexibility and operational cost, list the risks of each, and recommend one with trade-offs.\n"
       "- consistency\n- scaling\n- operations\n") * 3


def plan_json(n, strategy="partition", **over):
    d = {"parallelism": n, "strategy": strategy, "reason": "independent dimensions",
         "subtasks": [{"objective": f"Analyze dimension number {i} in depth", "expected_output": f"notes {i}",
                       "max_tokens": 200} for i in range(1, n + 1)]}
    d.update(over)
    return json.dumps(d)


class FakeBackend:
    """Answers by role; records every call, the most calls running at once, and each call's stop event."""

    def __init__(self, plan_text, slots=6, worker=None, delay=0.05, ctx=4096):
        self.plan_text, self.slots, self.delay, self.ctx = plan_text, slots, delay, ctx
        self.worker = worker or (lambda call, n, stop: tp.Result(text=f"work for {call.messages[-1]['content'][:30]}",
                                                           finish="stop", completion_tokens=50))
        self.calls, self.lock, self.running, self.peak = [], threading.Lock(), 0, 0
        self.cancels = []
        self.n_worker = 0

    def concurrency(self):
        return self.slots

    def context(self):
        return self.ctx

    def count_tokens(self, call):
        return sum(len(m["content"]) for m in call.messages) // 4

    def generate(self, call, cancel):
        with self.lock:
            self.calls.append(call)
            self.cancels.append(cancel)
            self.running += 1
            self.peak = max(self.peak, self.running)
            n = self.n_worker = self.n_worker + (call.role == "worker")
        try:
            t0 = time.time()
            if call.role == "planner":
                return tp.Result(text=self.plan_text, finish="stop", completion_tokens=80, t_start=t0, t_end=time.time())
            time.sleep(self.delay)
            r = self.worker(call, n, cancel) if call.role == "worker" else tp.Result(text="final", finish="stop")
            if isinstance(r, Exception):
                raise r
            r.t_start, r.t_end = t0, time.time()
            return r
        finally:
            with self.lock:
                self.running -= 1


def msgs(text=BIG):
    return [{"role": "system", "content": "Answer in English."}, {"role": "user", "content": text}]


class Modes(unittest.TestCase):
    def test_parse(self):
        for v in (None, False, "off", "OFF", 0, 1, "1", "", "none"):
            self.assertTrue(tp.parse_mode(v).off, v)
        self.assertEqual(tp.parse_mode("auto").kind, "auto")
        self.assertEqual(tp.parse_mode(True).kind, "auto")
        for n in (2, 4, 6, "4", 6.0):
            m = tp.parse_mode(n)
            self.assertEqual((m.kind, m.workers), ("fixed", int(n)))
        for bad in ("fast", -1, 9, 100, 2.5, [], {}):
            with self.assertRaises(ValueError):
                tp.parse_mode(bad)

    def test_config(self):
        c = tp.Config.from_dict({"default": "auto", "max_workers": 4, "worker_timeout_s": 30})
        self.assertEqual((c.max_workers, c.worker_timeout_s), (4, 30.0))
        for bad in ({"nope": 1}, {"max_workers": 0}, {"max_workers": True}, {"min_tokens": 1.5}):
            with self.assertRaises(ValueError):
                tp.Config.from_dict(bad)


class Gate(unittest.TestCase):
    def test_trivial_requests_stay_at_one(self):
        for t in ("What's 2+2?", "Rewrite this sentence: the cat sat on the mat.", "Tell me a joke.",
                  "What is the capital of Australia?", "Translate 'good morning' into Spanish.", "hi"):
            self.assertLess(tp.gate_score(t)[0], tp.Config().gate_threshold, t)

    def test_decomposable_requests_pass(self):
        self.assertGreaterEqual(tp.gate_score(BIG)[0], tp.Config().gate_threshold)
        code = "Find and fix all the bugs in this function, explain each one:\n```python\n" + "x = 1\n" * 40 + "```"
        self.assertGreaterEqual(tp.gate_score(code)[0], tp.Config().gate_threshold)

    def test_enumerations_count(self):
        # parts named inside one sentence, or numbered inline, are independent dimensions; asides in parentheses
        # do not count as parts
        inline = ("Explain the causes of the crisis, organized around: housing and lending, financial innovation "
                  "(securitization, CDOs, swaps), leverage and regulation, and global imbalances.")
        self.assertTrue(any("enumerated parts" in w for w in tp.gate_score(inline)[1]))
        self.assertGreaterEqual(tp.gate_score(inline)[0], tp.Config().gate_threshold)
        numbered = "Work out (1) the runway, (2) the revenue in a year and (3) the risks, for our startup's numbers."
        self.assertIn("3 listed items", tp.gate_score(numbered)[1])
        aside = "Summarize this paragraph about apples (red, green, yellow, and pink ones) in one line please, ok."
        self.assertFalse(any("enumerated" in w for w in tp.gate_score(aside)[1]))


class Plans(unittest.TestCase):
    cfg = tp.Config()

    def test_valid_partition(self):
        p = tp.parse_plan(plan_json(4), self.cfg, 6)
        self.assertEqual((p.parallelism, p.strategy, len(p.subtasks)), (4, "partition", 4))
        self.assertEqual([s.id for s in p.subtasks], [1, 2, 3, 4])

    def test_fenced_and_surrounded_json(self):
        p = tp.parse_plan("Here is the plan:\n```json\n" + plan_json(2, "independent") + "\n```", self.cfg, 6)
        self.assertEqual((p.parallelism, p.strategy), (2, "independent"))

    def test_one(self):
        self.assertEqual(tp.parse_plan('{"parallelism": 1, "reason": "simple"}', self.cfg, 6).parallelism, 1)

    def test_malformed(self):
        for bad in ("no json here", "{not json}", "[1, 2]", '{"parallelism": "two"}', '{"parallelism": 0}',
                    plan_json(3, subtasks=[]), plan_json(2, strategy="magic"),
                    json.dumps({"parallelism": 2, "subtasks": [{"objective": "same objective here"},
                                                               {"objective": "Same objective, here!"}]}),
                    json.dumps({"parallelism": 2, "subtasks": [{"objective": ""}, {"objective": "a real one too"}]}),
                    json.dumps({"parallelism": 2, "subtasks": ["a", "b"]}),
                    json.dumps({"parallelism": 2, "subtasks": [{"objective": "x" * 900}, {"objective": "another one"}]})):
            with self.assertRaises(tp.PlanError, msg=bad):
                tp.parse_plan(bad, self.cfg, 6)

    def test_limits(self):
        with self.assertRaises(tp.PlanError):
            tp.parse_plan(plan_json(7), self.cfg, 6)                 # more than the limit: never created
        with self.assertRaises(tp.PlanError):
            tp.parse_plan(plan_json(3), self.cfg, 6, exact=4)        # the request asked for exactly 4
        with self.assertRaises(tp.PlanError):
            tp.parse_plan('{"parallelism": 1}', self.cfg, 6, exact=2)

    def test_subtasks_as_strings_get_a_share_of_the_budget(self):
        p = tp.parse_plan(json.dumps({"parallelism": 3, "strategy": "partition",
                                      "subtasks": ["first part of it", "second part of it", "third part of it"]}),
                          self.cfg, 6)
        self.assertEqual([s.objective for s in p.subtasks], ["first part of it", "second part of it", "third part of it"])
        self.assertTrue(all(s.max_tokens == min(self.cfg.max_tokens, self.cfg.total_tokens // 3) for s in p.subtasks))

    def test_budgets_clamped_and_scaled(self):
        d = json.loads(plan_json(3))
        d["subtasks"][0]["max_tokens"] = 5           # below the floor
        d["subtasks"][1]["max_tokens"] = 100000      # above the ceiling
        p = tp.parse_plan(json.dumps(d), self.cfg, 6)
        self.assertEqual(p.subtasks[0].max_tokens, self.cfg.min_tokens)
        self.assertEqual(p.subtasks[1].max_tokens, self.cfg.max_tokens)
        p = tp.parse_plan(plan_json(6, subtasks=[{"objective": f"part {i} of the work", "max_tokens": 600}
                                                 for i in range(6)]), self.cfg, 6, token_room=700)
        self.assertLessEqual(sum(s.max_tokens for s in p.subtasks), 700)
        with self.assertRaises(tp.PlanError):                       # no room for even the floor of each part
            tp.parse_plan(plan_json(6), self.cfg, 6, token_room=100)


class Orchestration(unittest.TestCase):
    def run_orch(self, backend, mode, cancel=None, cfg=None):
        return tp.Orchestrator(backend, cfg or tp.Config()).run(msgs(), tp.parse_mode(mode), cancel or threading.Event())

    def test_off_makes_no_call(self):
        b = FakeBackend(plan_json(2))
        out = self.run_orch(b, "off")
        self.assertEqual((out.kind, b.calls), ("direct", []))

    def test_auto_trivial_makes_no_call(self):
        b = FakeBackend(plan_json(2))
        out = tp.Orchestrator(b).run([{"role": "user", "content": "What's 2+2?"}], tp.parse_mode("auto"),
                                     threading.Event())
        self.assertEqual(out.kind, "direct")
        self.assertEqual(b.calls, [])
        self.assertIn("not worth planning", out.meta["decision"])

    def test_auto_serial_engine_stays_at_one(self):
        b = FakeBackend(plan_json(2), slots=1)
        self.assertEqual(self.run_orch(b, "auto").kind, "direct")
        self.assertEqual(b.calls, [])

    def test_planner_says_one(self):
        b = FakeBackend('{"parallelism": 1, "reason": "one chain of steps"}')
        out = self.run_orch(b, "auto")
        self.assertEqual(out.kind, "direct")
        self.assertEqual([c.role for c in b.calls], ["planner"])

    def test_auto_low_effort_stays_at_one(self):
        for effort, kind in (("low", "direct"), ("medium", "synthesize"), ("high", "synthesize"), ("", "synthesize")):
            b = FakeBackend(plan_json(2, effort=effort) if effort else plan_json(2))
            out = self.run_orch(b, "auto")
            self.assertEqual(out.kind, kind, effort)
            b = FakeBackend(plan_json(2, effort=effort) if effort else plan_json(2))
            self.assertEqual(self.run_orch(b, 2).kind, "synthesize")      # a fixed count ignores the estimate

    def test_workers_run_concurrently_then_synthesis(self):
        b = FakeBackend(plan_json(4), delay=0.3)
        t = time.time()
        out = self.run_orch(b, "auto")
        self.assertEqual(out.kind, "synthesize")
        self.assertLess(time.time() - t, 4 * 0.3)              # not one after another
        self.assertEqual(b.peak, 4)
        roles = [c.role for c in b.calls]
        self.assertEqual(roles.count("worker"), 4)
        self.assertEqual(roles[0], "planner")
        m = out.meta
        self.assertTrue(m["enabled"])
        self.assertEqual((m["workers"], m["strategy"], m["worker_tokens"]), (4, "partition", 200))
        syn = out.synthesis.messages[-1]["content"]
        for i in range(1, 5):
            self.assertIn(f"Analyze dimension number {i}", syn)
        self.assertNotIn("work for", out.synthesis.messages[0]["content"])   # the shared context has no worker text
        self.assertIn("work for", syn)

    def test_shared_prefix_and_budgets(self):
        cfg = tp.Config(worker_reasoning=100)
        b = FakeBackend(plan_json(2))
        out = self.run_orch(b, 2, cfg=cfg)
        shared = b.calls[0].messages[0]
        for c in b.calls + [out.synthesis]:
            self.assertEqual(c.messages[0], shared)           # every internal request starts the same
            self.assertEqual(c.shared_prefix, [shared])
        for c in b.calls[1:]:
            self.assertEqual((c.max_tokens, c.reasoning_budget, c.thinking), (200 + 100, 100, "on"))
            self.assertTrue(c.messages[-1]["content"].startswith("STEP subtask"))
            self.assertNotIn("STEP plan", c.messages[-1]["content"])
        self.assertIn("do not split the work further", shared["content"])          # no recursion
        self.assertEqual(b.calls[0].thinking, "off")
        self.assertIn("USER REQUEST", shared["content"])
        self.assertIn("Answer in English.", shared["content"])

    def test_fixed_count_must_match(self):
        b = FakeBackend(plan_json(3))
        out = self.run_orch(b, 4)
        self.assertEqual(out.kind, "direct")
        self.assertIn("exactly 4", out.meta["fallback"])

    def test_malformed_plan_falls_back(self):
        b = FakeBackend("I would split this into a few parts.")
        out = self.run_orch(b, "auto")
        self.assertEqual(out.kind, "direct")
        self.assertIn("plan:", out.meta["fallback"])

    def test_planner_error_falls_back(self):
        class Boom(FakeBackend):
            def generate(self, call, cancel):
                raise RuntimeError("engine died")
        out = self.run_orch(Boom(plan_json(2)), "auto")
        self.assertEqual(out.kind, "direct")
        self.assertIn("engine died", out.meta["fallback"])

    def test_worker_error_is_retried_once(self):
        tries = {}

        def worker(call, n, stop):
            k = call.messages[-1]["content"][:40]
            tries[k] = tries.get(k, 0) + 1
            if "subtask 2/" in k and tries[k] == 1:
                return RuntimeError("slot error")
            return tp.Result(text="ok", finish="stop", completion_tokens=10)
        b = FakeBackend(plan_json(3), worker=worker)
        out = self.run_orch(b, "auto")
        self.assertEqual(out.kind, "synthesize")
        self.assertNotIn("partial", out.meta)
        self.assertEqual([w["attempts"] for w in out.meta["worker_results"]], [1, 2, 1])

    def test_one_worker_fails_twice_partial_synthesis(self):
        def worker(call, n, stop):
            if "subtask 3/" in call.messages[-1]["content"]:
                return tp.Result(text="", finish="stop")          # malformed: no output
            return tp.Result(text="ok", finish="stop", completion_tokens=10)
        b = FakeBackend(plan_json(3), worker=worker)
        out = self.run_orch(b, "auto")
        self.assertEqual(out.kind, "synthesize")
        self.assertEqual(out.meta["partial"], "2 of 3 subtasks completed")
        self.assertIn("did not complete", out.synthesis.messages[-1]["content"])

    def test_every_worker_fails_falls_back(self):
        b = FakeBackend(plan_json(2), worker=lambda c, n, stop: RuntimeError("bad"))
        out = self.run_orch(b, "auto")
        self.assertEqual(out.kind, "direct")
        self.assertEqual(out.meta["fallback"], "no worker output")

    def test_timeout_stops_the_worker(self):
        cfg = tp.Config(worker_timeout_s=0.3, retries=0)

        def worker(call, n, stop):
            if "subtask 1/" in call.messages[-1]["content"]:
                stop.wait(5)                                    # a stuck worker: only its stop ends it
                return tp.Result(text="late", finish="cancel")
            return tp.Result(text="ok", finish="stop", completion_tokens=10)
        b = FakeBackend(plan_json(2), worker=worker, delay=0)
        t = time.time()
        out = tp.Orchestrator(b, cfg).run(msgs(), tp.parse_mode(2), threading.Event())
        self.assertLess(time.time() - t, 3.0)
        self.assertEqual(out.kind, "synthesize")
        self.assertEqual(out.meta["partial"], "1 of 2 subtasks completed")

    def test_parent_cancel_stops_everything(self):
        cancel = threading.Event()

        def worker(call, n, stop):
            if n == 2:
                cancel.set()                                     # the client goes away while the subtasks run
            stop.wait(5)
            return tp.Result(text="", finish="cancel")
        b = FakeBackend(plan_json(4), worker=worker, delay=0)
        t = time.time()
        out = self.run_orch(b, 4, cancel=cancel)
        self.assertLess(time.time() - t, 3.0)
        self.assertEqual(out.kind, "direct")
        self.assertTrue(out.meta.get("cancelled"))
        self.assertTrue(all(c.is_set() for c in b.cancels[1:]))   # every worker's stop was raised

    def test_meta_has_no_text(self):
        b = FakeBackend(plan_json(2))
        out = self.run_orch(b, 2)
        m = tp.finish_meta(out.meta, time.time(), 10, 5)
        blob = json.dumps(m)
        self.assertNotIn("work for", blob)
        self.assertNotIn("subtasks", m)
        self.assertIn("subtasks", tp.finish_meta(out.meta, time.time(), 10, 5, diagnostics=True))
        self.assertEqual(m["total_generated_tokens"], 80 + 100 + 10)


# ------------------------------------------------------------------------------------------------ over HTTP
class RoleEngine(MockEngine):
    """Answers by the step its prompt asks for."""

    def __init__(self, tok, plan, **kw):
        super().__init__(tok, "x", **kw)
        self.plan, self.prompts = plan, []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        text = self.tok.decode(ids)
        self.prompts.append(text)
        think = text.endswith("<think>\n")
        last = text.rsplit("<|im_start|>user\n", 1)[-1]
        if last.startswith("STEP plan"):
            reply = self.plan
        elif last.startswith("STEP subtask"):
            k = last.split("STEP subtask ")[1].split("/")[0]
            reply = ("</think>\n\n" if think else "") + f"WORK{k}"
        elif last.startswith("STEP final answer"):
            reply = ("</think>\n\n" if think else "") + "FINAL ANSWER"
        else:
            reply = ("</think>\n\n" if think else "") + "PLAIN ANSWER"
        self.script = self.tok.encode(reply, parse_special=True) + self.tok.encode("<|im_end|>", parse_special=True)
        self.scripts = [self.script]
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = RoleEngine(cls.tok, plan_json(2), max_context=1 << 16)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
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

    def body(self, text=BIG, **kw):
        return {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": 64, **kw}

    def test_off_is_the_ordinary_request(self):
        s0, b0 = self.post(self.body())
        p0 = self.engine.prompts[-1]
        for off in ("off", 1, 0, None):
            s1, b1 = self.post(self.body(task_parallel=off))
            self.assertEqual((s1, b1["choices"], self.engine.prompts[-1]), (s0, b0["choices"], p0))
            self.assertNotIn("task_parallel", b1)
        self.assertEqual(b0["choices"][0]["message"]["content"], "PLAIN ANSWER")

    def test_bad_value_is_a_400(self):
        s, b = self.post(self.body(task_parallel="fast"))
        self.assertEqual(s, 400)

    def test_auto_trivial_answers_directly(self):
        n = len(self.engine.prompts)
        s, b = self.post(self.body("What's 2+2?", task_parallel="auto"))
        self.assertEqual(s, 200)
        self.assertEqual(b["choices"][0]["message"]["content"], "PLAIN ANSWER")
        self.assertFalse(b["task_parallel"]["enabled"])
        self.assertEqual(len(self.engine.prompts), n + 1)       # no planner call

    def test_two_workers_and_synthesis(self):
        n = len(self.engine.prompts)
        s, b = self.post(self.body(task_parallel=2, task_parallel_diagnostics=True))
        self.assertEqual(s, 200, b)
        self.assertEqual(b["choices"][0]["message"]["content"], "FINAL ANSWER")
        m = b["task_parallel"]
        self.assertTrue(m["enabled"])
        self.assertEqual((m["workers"], len(m["subtasks"])), (2, 2))
        for k in ("planning_ms", "workers_ms", "synthesis_ms", "total_ms"):
            self.assertIsInstance(m[k], int)
        prompts = self.engine.prompts[n:]
        self.assertEqual(len(prompts), 4)                       # plan, 2 subtasks, synthesis
        self.assertIn("WORK1", prompts[-1])
        self.assertIn("WORK2", prompts[-1])
        self.assertNotIn("WORK", json.dumps(b["choices"]))      # the work products stay inside
        shared = prompts[0].split("<|im_end|>")[0]
        self.assertTrue(all(p.startswith(shared) for p in prompts))

    def test_stream_progress_and_meta(self):
        s, text = self.post(self.body(task_parallel=2, stream=True), raw=True)
        self.assertEqual(s, 200)
        self.assertIn(": task_parallel planning", text)
        self.assertIn(": task_parallel synthesizing", text)
        events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
        content = "".join((e["choices"][0]["delta"].get("content") or "") for e in events if e.get("choices"))
        self.assertEqual(content, "FINAL ANSWER")
        self.assertTrue(events[-1]["task_parallel"]["enabled"])
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))

    def test_internal_prompts_share_their_prefix(self):
        shared = [{"role": "system", "content": "the shared context"}]
        lens = set()
        for thinking, effort in (("off", "low"), ("on", "low"), ("off", ""), ("on", "")):
            ids = self.svc.render_internal(shared + [{"role": "user", "content": f"step {thinking}"}], thinking, effort)
            n = self.svc.internal_prefix_len(shared, ids, effort)
            self.assertGreater(n, 0)
            self.assertTrue(self.tok.decode(ids[:n]).endswith("the shared context<|im_end|>\n"))
            lens.add((effort, n))
        self.assertEqual(len(lens), 2)                   # one prefix per effort setting, whatever the thinking
        self.assertEqual(self.svc.internal_prefix_len(shared, [1, 2, 3]), 0)    # not a prefix: no hint

    def test_tools_run_as_usual(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        s, b = self.post(self.body(task_parallel=2, tools=tools))
        self.assertEqual(s, 200)
        self.assertNotIn("task_parallel", b)


if __name__ == "__main__":
    unittest.main()
