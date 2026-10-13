"""serve/test_llamacpp.py - the llama.cpp engine path (serve/llamacpp.py) without a GPU, a model or a llama.cpp build.

    python -m unittest serve.test_llamacpp -v

llama-server is replaced by serve/llamacpp_fake_server.py, a real child process with llama-server's command line and
HTTP routes; what it saw is read back from its log.  Covered: the process lifecycle (loading, failure to start,
death and restart, idle unload, load gates), OpenAI and Anthropic passthrough (whole and streamed, cancellation by
a client that goes away), the API key and routes, Task-Parallel Requests through LlamaCppBackend, the example
config, and serve/server.py's --engine llamacpp end to end."""
from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve import llamacpp as L  # noqa: E402
from serve import server as S  # noqa: E402
from serve import task_parallel as tp  # noqa: E402

FAKE = str(ROOT / "serve" / "llamacpp_fake_server.py")
KEY = "test-key"
BIG = ("Compare PostgreSQL, MongoDB and Cassandra for an order-tracking service. Analyze consistency, scaling, "
       "query flexibility and operational cost, list the risks of each, and recommend one with trade-offs.\n"
       "- consistency\n- scaling\n- operations\n") * 3


def quiet(*a, **k):
    pass


class Base(unittest.TestCase):
    parallel = 4
    env: dict = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.log = self.dir / "fake.jsonl"
        self.server_log = self.dir / "llama-server.log"
        self.server = self.make_server()
        self.svc = L.LlamaCppService(self.server, "fake-hotcold", aliases=["fake-alias"], api_key=KEY, say=quiet)
        self.httpd = L.serve(self.svc, "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]

    def make_server(self, **kw):
        env = {"FAKE_LLAMA_LOG": str(self.log), "FAKE_LLAMA_TOKEN_S": "0.05", **self.env, **kw.pop("env", {})}
        args = kw.pop("args", [FAKE, "--parallel", str(self.parallel), "--ctx-size", "65536"])
        return L.LlamaServer(sys.executable, args, model=str(self.dir / "m.gguf"), env=env, log=str(self.server_log),
                             start_timeout_s=kw.pop("start_timeout_s", 20), alias="fake-hotcold", **kw)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.server.close()
        self.tmp.cleanup()

    # -- helpers
    def seen(self) -> list:
        try:
            return [json.loads(x) for x in self.log.read_text().splitlines() if x.strip()]
        except FileNotFoundError:
            return []

    def bodies(self, path="/v1/chat/completions") -> list:
        return [x["body"] for x in self.seen() if x.get("path") == path and "body" in x]

    def req(self, method, path, body=None, key=KEY, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        h = {"Content-Type": "application/json", **(headers or {})}
        if key:
            h["Authorization"] = f"Bearer {key}"
        c.request(method, path, None if body is None else json.dumps(body).encode(), h)
        r = c.getresponse()
        data = r.read()
        c.close()
        ctype = r.getheader("Content-Type", "")
        return r.status, (json.loads(data) if "json" in ctype and data else data.decode())

    def chat(self, **body):
        body.setdefault("messages", [{"role": "user", "content": "hi"}])
        return self.req("POST", "/v1/chat/completions", body)

    @staticmethod
    def events(text: str) -> list:
        out = []
        for line in text.splitlines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                out.append(json.loads(line[6:]))
        return out

    @staticmethod
    def content(events) -> str:
        return "".join((c.get("delta") or {}).get("content") or "" for e in events for c in e.get("choices") or [])


class TestProcess(Base):
    def test_loads_then_answers_with_its_own_key(self):
        self.server.close()
        self.server = self.svc.server = self.make_server(env={"FAKE_LLAMA_LOAD_S": "0.8"}, hotcold=True)
        t0 = time.time()
        self.server.ensure(quiet)
        self.assertGreater(time.time() - t0, 0.7)           # waited through "Loading model"
        self.assertTrue(self.server.loaded())
        self.assertEqual(self.server.slots(), 4)
        self.assertEqual(self.server.slot_ctx(), 16384)
        argv = self.seen()[0]
        self.assertTrue(argv["has_key"])
        joined = " ".join(argv["argv"])
        self.assertIn("--no-op-offload", joined)            # a hot/cold model
        self.assertIn("--alias fake-hotcold", joined)
        self.assertIn("--host 127.0.0.1", joined)
        self.assertNotIn(self.server.key, joined)           # the key is never on the command line
        status, out = self.server.call("GET", "/props")
        self.assertEqual(status, 200)
        c = http.client.HTTPConnection(self.server.host, self.server.port, timeout=5)
        c.request("GET", "/props")
        self.assertEqual(c.getresponse().status, 401)       # without the key, llama-server refuses
        c.close()

    def test_no_op_offload_only_for_hotcold(self):
        self.server.ensure(quiet)
        self.assertNotIn("--no-op-offload", self.seen()[0]["argv"])

    def test_refuses_options_it_sets_itself(self):
        for bad in (["--port", "1"], ["--host=0.0.0.0"], ["--api-key", "x"], ["--api-key-file", "f"]):
            with self.assertRaises(ValueError):
                L.LlamaServer("llama-server", bad)

    def test_exit_at_start_reports_the_log(self):
        self.server.close()
        self.server = self.svc.server = self.make_server(env={"FAKE_LLAMA_EXIT_AT_START": "7"})
        with self.assertRaises(L.LoadError) as e:
            self.server.ensure(quiet)
        self.assertIn("code 7", str(e.exception))
        self.assertIn("failing at start as asked", str(e.exception))
        self.assertFalse(self.server.alive())
        status, out = self.chat()
        self.assertEqual(status, 503)
        self.assertIn("code 7", out["error"]["message"])

    def test_start_timeout(self):
        self.server.close()
        self.server = self.svc.server = self.make_server(env={"FAKE_LLAMA_LOAD_S": "30"}, start_timeout_s=0.6)
        with self.assertRaises(L.LoadError) as e:
            self.server.ensure(quiet)
        self.assertIn("did not load", str(e.exception))
        self.assertFalse(self.server.alive())

    def test_restarts_after_it_died(self):
        self.assertEqual(self.chat()[0], 200)
        self.server.proc.kill()
        self.server.proc.wait()
        status, out = self.chat()
        self.assertEqual(status, 200)
        self.assertEqual(self.server.starts, 2)

    def test_death_mid_stream_ends_the_stream_then_recovers(self):
        status, text = self.chat(stream=True, messages=[{"role": "user", "content": "die now please"}])
        self.assertEqual(status, 200)
        self.assertIn("llama-server stopped", text)
        status, out = self.chat()
        self.assertEqual(status, 200)
        self.assertEqual(out["choices"][0]["message"]["content"], "Hello from the fake llama-server.")
        self.assertEqual(self.server.starts, 2)

    def test_close_stops_the_process(self):
        self.server.ensure(quiet)
        proc = self.server.proc
        self.server.close()
        self.assertIsNotNone(proc.poll())

    def test_idle_unload_and_reload(self):
        self.svc.idle_unload_s = 0.2
        self.assertEqual(self.chat()[0], 200)
        self.assertFalse(self.svc.unload_if_idle(now=time.time()))   # not idle long enough yet
        time.sleep(0.3)
        self.assertTrue(self.svc.unload_if_idle())
        self.assertFalse(self.server.alive())
        self.assertEqual(self.chat()[0], 200)
        self.assertEqual(self.server.starts, 2)

    def test_before_load_can_refuse(self):
        self.server.before_load = [sys.executable, "-c", "raise SystemExit(1)"]
        status, out = self.chat()
        self.assertEqual(status, 503)
        self.assertIn("before_load", out["error"]["message"])
        self.assertIsNone(self.server.proc)
        self.server.before_load = [sys.executable, "-c", "pass"]
        self.assertEqual(self.chat()[0], 200)

    def test_min_free_ram(self):
        if L.mem_available_mib() is None:
            self.skipTest("no /proc/meminfo")
        self.server.min_free_ram_mib = 1 << 40
        status, out = self.chat()
        self.assertEqual(status, 503)
        self.assertIn("min_free_ram_mib", out["error"]["message"])
        self.assertIsNone(self.server.proc)


class TestPassthrough(Base):
    def test_health_and_models_need_no_load(self):
        status, h = self.req("GET", "/health", key=None)
        self.assertEqual((status, h["status"], h["engine"]), (200, "unloaded", "llamacpp"))
        status, m = self.req("GET", "/v1/models")
        self.assertEqual([x["id"] for x in m["data"]], ["fake-hotcold", "fake-alias"])
        self.assertIsNone(self.server.proc)                 # nothing was loaded for that

    def test_api_key(self):
        status, out = self.req("POST", "/v1/chat/completions", {"messages": []}, key=None)
        self.assertEqual(status, 401)
        self.assertEqual(out["error"]["type"], "authentication_error")
        status, out = self.req("POST", "/v1/messages", {"messages": []}, key="wrong")
        self.assertEqual(status, 401)
        self.assertEqual(out["type"], "error")              # Anthropic's error shape on Anthropic's route
        status, out = self.req("POST", "/v1/messages", {"max_tokens": 5, "messages": [{"role": "user", "content": "x"}]},
                               key=None, headers={"x-api-key": KEY})
        self.assertEqual(status, 200)

    def test_host_check_without_a_key(self):
        """run() wires serve/server.py's DNS-rebinding check: without an API key only this server's names pass."""
        names = S.host_names_for("127.0.0.1")
        self.svc.api_key = ""
        self.svc.host_allowed = lambda host: S.host_allowed(host, names, False)
        self.assertEqual(self.req("GET", "/v1/models", key=None, headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.req("GET", "/v1/models", key=None, headers={"Host": "localhost:1234"})[0], 200)

    def test_only_listed_routes(self):
        self.assertEqual(self.req("POST", "/slots/0?action=save", {"filename": "x"})[0], 404)
        self.assertEqual(self.req("POST", "/lora-adapters", [])[0], 404)
        self.assertEqual(self.req("GET", "/props")[0], 200)

    def test_openai_whole(self):
        status, out = self.chat(model="whatever", temperature=0.3)
        self.assertEqual(status, 200)
        self.assertEqual(out["choices"][0]["message"]["content"], "Hello from the fake llama-server.")
        self.assertEqual(self.bodies()[-1]["temperature"], 0.3)

    def test_openai_stream(self):
        status, text = self.chat(stream=True)
        self.assertEqual(status, 200)
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))
        self.assertEqual(self.content(self.events(text)), "Hello from the fake llama-server.")

    def test_strata_fields_are_not_forwarded(self):
        self.chat(task_parallel="off", task_parallel_diagnostics=True, context_items=[])
        body = self.bodies()[-1]
        for k in L.STRATA_FIELDS:
            self.assertNotIn(k, body)

    def test_anthropic_whole_and_stream(self):
        body = {"model": "x", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]}
        status, out = self.req("POST", "/v1/messages", body)
        self.assertEqual((status, out["content"][0]["text"]), (200, "Hello from the fake llama-server."))
        status, text = self.req("POST", "/v1/messages", {**body, "stream": True})
        self.assertEqual(status, 200)
        self.assertIn("event: message_stop", text)
        deltas = [e["delta"]["text"] for e in self.events(text) if e.get("type") == "content_block_delta"]
        self.assertEqual("".join(deltas), "Hello from the fake llama-server.")

    def test_bad_json(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", "/v1/chat/completions", b"{nope", {"Authorization": f"Bearer {KEY}"})
        r = c.getresponse()
        self.assertEqual(r.status, 400)
        c.close()

    def test_client_going_away_cancels(self):
        self.leave_early(stream=True)

    def test_client_going_away_cancels_a_whole_answer(self):
        self.leave_early(stream=False)

    def leave_early(self, stream: bool):
        self.server.ensure(quiet)
        body = json.dumps({"stream": stream, "messages": [{"role": "user", "content": "slow please"}]}).encode()
        s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                  + f"Authorization: Bearer {KEY}\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
        if stream:
            got = b""
            while b"tick" not in got:
                got += s.recv(4096)
        else:
            time.sleep(0.4)                                  # 50 tokens x 0.05 s: still generating
        s.close()
        deadline = time.time() + 5
        while time.time() < deadline and not any("cancelled" in x for x in self.seen()):
            time.sleep(0.1)
        self.assertTrue(any("cancelled" in x for x in self.seen()), "llama-server kept generating")


class TestTaskParallel(Base):
    def tp(self, **body):
        body.setdefault("messages", [{"role": "user", "content": BIG}])
        return self.chat(**body)

    def test_fixed_two_synthesis_whole(self):
        status, out = self.tp(task_parallel=2, task_parallel_compose="synthesis", task_parallel_diagnostics=True)
        self.assertEqual(status, 200, out)
        self.assertEqual(out["choices"][0]["message"]["content"], "FINAL ANSWER written from the notes.")
        meta = out["task_parallel"]
        self.assertTrue(meta["enabled"], meta)
        self.assertEqual(meta["workers"], 2)
        self.assertEqual(len(meta["subtasks"]), 2)
        roles = [b["messages"][-1]["content"].split(".")[0] for b in self.bodies()]
        planner = [b for b in self.bodies() if "STEP plan" in b["messages"][-1]["content"]]
        workers = [b for b in self.bodies() if "STEP subtask" in b["messages"][-1]["content"]]
        self.assertEqual((len(planner), len(workers)), (1, 2), roles)
        self.assertEqual(planner[0]["temperature"], 0.0)    # the plan is greedy
        self.assertFalse(planner[0]["chat_template_kwargs"]["enable_thinking"])
        for b in self.bodies():
            self.assertNotIn("task_parallel", b)

    def test_fixed_two_synthesis_stream(self):
        status, text = self.tp(task_parallel=2, task_parallel_compose="synthesis", stream=True)
        self.assertEqual(status, 200)
        self.assertIn(": task_parallel", text)              # progress as SSE comments
        ev = self.events(text)
        self.assertEqual(self.content(ev), "FINAL ANSWER written from the notes.")
        meta = [e["task_parallel"] for e in ev if "task_parallel" in e]
        self.assertEqual(len(meta), 1)
        self.assertTrue(meta[0]["enabled"])
        self.assertGreater(meta[0]["synthesis_tokens"], 0)  # from the stream's usage
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))

    def test_direct_sections(self):
        status, out = self.tp(task_parallel=2, task_parallel_compose="direct")
        self.assertEqual(status, 200, out)
        text = out["choices"][0]["message"]["content"]
        self.assertIn("## Consistency", text)
        self.assertIn("## Scaling", text)
        self.assertLess(text.index("## Consistency"), text.index("## Scaling"))

    def test_tools_take_the_ordinary_path(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        status, out = self.tp(task_parallel=2, tools=tools)
        self.assertEqual(status, 200)
        self.assertEqual(len(self.bodies()), 1)             # no planner, no subtasks
        self.assertNotIn("task_parallel", out)

    def test_bad_mode_is_400(self):
        self.assertEqual(self.tp(task_parallel="many")[0], 400)
        self.assertEqual(self.tp(task_parallel=2, context_items="x")[0], 400)

    def test_backend_protocol(self):
        self.server.ensure(quiet)
        b = L.LlamaCppBackend(self.server, {"temperature": 0.5, "stream": True})
        self.assertEqual((b.concurrency(), b.context()), (4, 16384))
        self.assertEqual(b.count_text("one two three"), 3)
        call = tp.Call("worker", [{"role": "user", "content": "a b c"}], 64)
        self.assertEqual(b.count_tokens(call), 4)            # "user: a b c"
        self.assertEqual(b.busy(), 0)
        seen = []
        call.on_text = seen.append
        r = b.generate(call, threading.Event())
        self.assertEqual(r.text, "Hello from the fake llama-server.")
        self.assertEqual("".join(seen), r.text)
        self.assertEqual((r.finish, r.prompt_tokens, r.reused_tokens), ("stop", 7, 3))
        self.assertEqual(self.bodies()[-1]["temperature"], 0.5)
        self.assertEqual(self.bodies()[-1]["max_tokens"], 64)

    def test_backend_cancel(self):
        self.server.ensure(quiet)
        b = L.LlamaCppBackend(self.server)
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        t0 = time.time()
        r = b.generate(tp.Call("worker", [{"role": "user", "content": "slow"}], 0), cancel)
        self.assertLess(time.time() - t0, 2.0)               # 50 tokens x 0.05 s would be 2.5 s
        self.assertEqual(r.finish, "cancelled")


class TestOneSlot(Base):
    parallel = 1

    def test_auto_needs_two_slots(self):
        status, out = self.chat(task_parallel="auto", messages=[{"role": "user", "content": BIG}])
        self.assertEqual(status, 200)
        self.assertFalse(out["task_parallel"]["enabled"])
        self.assertIn("one request at a time", out["task_parallel"]["decision"])
        self.assertEqual(len(self.bodies()), 1)


class TestConfig(unittest.TestCase):
    def test_example_configs(self):
        files = sorted((ROOT / "examples/llamacpp-hotcold").glob("*.json"))
        self.assertGreaterEqual(len(files), 2)
        for f in files:
            cfg = json.loads(f.read_text())
            s = L.server_from_config(cfg)                    # the model does not exist: hotcold from the -ot rule
            self.assertTrue(s.hotcold, f)
            self.assertTrue(os.path.isabs(s.exe) and os.path.isabs(s.model), f)
            self.assertIn("--no-op-offload", s.argv())
            self.assertEqual(s.alias, cfg["model_name"])
            self.assertIn("--parallel", s.args)
            if cfg.get("task_parallel"):
                svc = L.LlamaCppService(s, cfg["model_name"], say=quiet)
                L.task_parallel_from_config(svc, cfg["task_parallel"])

    def test_hotcold_detected_from_the_gguf(self):
        sys.path.insert(0, str(ROOT / "tools"))
        from test_moe_placement import glm_like, write_gguf  # noqa: PLC0415
        from moe_placement import gguf_hotcold  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "src.gguf", Path(d) / "hc.gguf"
            meta, t = glm_like()
            write_gguf(src, meta, t)
            gguf_hotcold.write(src, {"1": [0, 3]}, dst, progress=None)
            self.assertFalse(L.is_hotcold_gguf(str(src)))
            self.assertTrue(L.is_hotcold_gguf(str(dst)))
            cfg = {"exe": "llama-server", "model": str(dst), "args": ["--ctx-size", "4096"]}
            self.assertIn("--no-op-offload", L.server_from_config(cfg).argv())
            self.assertNotIn("--no-op-offload", L.server_from_config({**cfg, "model": str(src)}).argv())
            self.assertNotIn("--no-op-offload", L.server_from_config({**cfg, "hotcold": False}).argv())

    def test_bad_configs(self):
        for cfg, msg in [({}, "exe"), ({"exe": "x"}, "model"), ({"exe": "x", "model": "m", "args": "-c 4"}, "list"),
                         ({"exe": "x", "model": "m", "hotcold": "yes"}, "hotcold"),
                         ({"exe": "x", "model": "m", "args": ["--port", "9"]}, "set by Strata Void")]:
            with self.assertRaises(ValueError) as e:
                L.server_from_config(cfg)
            self.assertIn(msg, str(e.exception))

    def test_task_parallel_default(self):
        svc = L.LlamaCppService(L.LlamaServer("x", []), "m", say=quiet)
        L.task_parallel_from_config(svc, {"default": "auto", "max_workers": 3})
        self.assertEqual((svc.task_parallel_default, svc.task_parallel_cfg.max_workers), ("auto", 3))
        with self.assertRaises(ValueError):
            L.task_parallel_from_config(svc, {"max_workers": 99})


class TestFlow(unittest.TestCase):
    """docs/HOTCOLD_LLAMACPP.md's flow on a tiny synthetic GLM-like checkpoint: a routing trace (routing-dump's
    format) -> llamacpp_hotcold.py -> gguf_hotcold.py -> a server config with the printed placement -> served."""

    def test_trace_plan_rewrite_serve(self):
        import random  # noqa: PLC0415
        from test_moe_placement import glm_like, write_gguf  # noqa: PLC0415
        from moe_placement import stats  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            meta, t = glm_like()
            write_gguf(d / "m.gguf", meta, t, fill=True)
            rng, recs = random.Random(1), []
            for _ in range(300):                              # experts 0-3 carry ~85% of the routing
                for layer in (1, 2, 3, 4):
                    recs.append((layer, rng.sample(range(16) if rng.random() < 0.15 else range(4), 2), [0.5, 0.5]))
            stats.write_strata_trace(d / "trace.bin", recs)
            run = lambda *a: subprocess.run([sys.executable, *map(str, a)], cwd=ROOT, check=True,  # noqa: E731
                                            capture_output=True, text=True).stdout
            out = run("tools/moe_placement/llamacpp_hotcold.py", d / "m.gguf", "--trace", d / "trace.bin",
                      "--device", "GPU0=300KiB", "--device", "GPU1=300KiB", "--ram", "4MiB", "--ctx", "64",
                      "--slot-zeros", "GPU1", "--out-cold", d / "cold.json")
            lines = out.splitlines()
            self.assertTrue(lines[0].startswith("cold: "), out)
            placement = next(x for x in lines if x.startswith("--device")).split()
            zeros = next(x for x in lines if x.startswith("rewrite with: --slot-zeros")).split()[-1]
            cold = json.loads((d / "cold.json").read_text())
            self.assertTrue(cold and all(set(v) <= set(range(16)) for v in cold.values()))
            self.assertFalse(any(e in v for v in cold.values() for e in range(4)))   # the hot experts stay hot
            run("tools/moe_placement/gguf_hotcold.py", d / "m.gguf", "--cold", d / "cold.json",
                "--out", d / "m-hotcold.gguf", "--slot-zeros", zeros)
            self.assertTrue(L.is_hotcold_gguf(str(d / "m-hotcold.gguf")))
            cfg = {"exe": sys.executable, "model": str(d / "m-hotcold.gguf"), "model_name": "tiny-hotcold",
                   "args": [FAKE, "--ctx-size", "4096", "--parallel", "2", *placement],
                   "env": {"FAKE_LLAMA_LOG": str(d / "fake.jsonl")}}
            server = L.server_from_config(cfg)
            self.assertTrue(server.hotcold)
            try:
                server.ensure(quiet)
                argv = json.loads((d / "fake.jsonl").read_text().splitlines()[0])["argv"]
                self.assertIn("--no-op-offload", argv)
                self.assertEqual(argv[argv.index("-ot") + 1], "ffn_.*_exps_cold\\.=CPU")
                self.assertEqual(argv[argv.index("--model") + 1], str(d / "m-hotcold.gguf"))
                b = L.LlamaCppBackend(server)
                r = b.generate(tp.Call("worker", [{"role": "user", "content": "hi"}], 32), threading.Event())
                self.assertEqual(r.text, "Hello from the fake llama-server.")
            finally:
                server.close()


class TestServerMain(unittest.TestCase):
    """python -m serve.server --engine llamacpp: start, answer, stop (and its llama-server with it)."""

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "fake.jsonl"
            cfg = {"exe": sys.executable, "args": [FAKE, "--parallel", "2", "--ctx-size", "8192"],
                   "model": str(Path(d) / "m.gguf"), "hotcold": False, "model_name": "e2e", "api_key": KEY,
                   "env": {"FAKE_LLAMA_LOG": str(log)}, "log": str(Path(d) / "server.log")}
            (Path(d) / "m.gguf").write_bytes(b"")
            (Path(d) / "cfg.json").write_text(json.dumps(cfg))
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            p = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "llamacpp", "--config",
                                  str(Path(d) / "cfg.json"), "--port", str(port)], cwd=ROOT,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                lines = []
                deadline = time.time() + 30
                while time.time() < deadline:
                    line = p.stdout.readline()
                    lines.append(line)
                    if line.startswith("ready:") or not line:
                        break
                self.assertTrue(lines[-1].startswith("ready:"), "".join(lines))
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                c.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
                          {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
                r = c.getresponse()
                out = json.loads(r.read())
                c.close()
                self.assertEqual(out["choices"][0]["message"]["content"], "Hello from the fake llama-server.")
                child = [json.loads(x) for x in log.read_text().splitlines()][0]
                self.assertIn("e2e", child["argv"])
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)   # llama-server dies: the next
                c.request("GET", "/health")                                   # request starts it again
                self.assertEqual(json.loads(c.getresponse().read())["starts"], 1)
                os.kill(child["pid"], signal.SIGKILL)
                time.sleep(0.3)
                c.request("POST", "/v1/chat/completions", json.dumps({"messages": []}),
                          {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
                self.assertEqual(c.getresponse().status, 200)
                c.close()
            finally:
                p.send_signal(signal.SIGTERM)
                rest = p.communicate(timeout=40)[0]
            self.assertIn("stopped", rest)
            self.assertIn("starting it again", rest)
            self.assertEqual(p.returncode, 0)
            argv = [json.loads(x) for x in log.read_text().splitlines() if "argv" in x][-1]["argv"]
            child_port = int(argv[argv.index("--port") + 1])
            with self.assertRaises(OSError):                 # llama-server stopped with it
                socket.create_connection(("127.0.0.1", child_port), timeout=2).close()


if __name__ == "__main__":
    unittest.main()
