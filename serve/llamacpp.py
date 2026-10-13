"""serve/llamacpp.py - serve a GGUF through llama.cpp's llama-server behind Strata Void's API (--engine llamacpp).

    python -m serve.server --engine llamacpp --config examples/llamacpp-hotcold/server.json

Made for the hot/cold expert GGUFs of docs/MOE_PLACEMENT.md (GLM-5.3-Flash, DeepSeek V4 Flash: architectures the native
Strata engine does not run), and works with any GGUF the configured llama-server runs.  The native engine is not
involved: this is a separate engine path that serve/server.py selects with --engine llamacpp.

  * Lifecycle.  llama-server runs as a child on a private loopback port with a random key (clients never reach it
    directly), is waited for until /health says the model is loaded, is started again by the next request when it
    died, and can be unloaded when idle (idle_unload_s).  A load can be gated on free host RAM (min_free_ram_mib) and
    preceded by a command (before_load), so a large model is not started beside another one.
  * Passthrough.  OpenAI (/v1/chat/completions, /v1/completions, /v1/responses, /v1/embeddings) and Anthropic
    (/v1/messages, /v1/messages/count_tokens) requests go to llama-server unchanged, streaming included: its chat
    template, tool-call parsing and reasoning handling apply.  A client that goes away cancels its request.
  * Task-Parallel Requests.  A chat completion asking for task_parallel (or the config's default) runs
    serve/task_parallel.py's Orchestrator; its internal requests are ordinary llama-server requests in its parallel
    slots (--parallel N), through LlamaCppBackend (the orchestrator's Backend protocol).

Config (JSON; relative paths are the config's cwd's): exe, model, args (llama-server's own options, without
--host/--port/--api-key, which Strata Void sets), cwd, env, log, model_name, aliases, hotcold (true / false / "auto":
read the GGUF header; a hot/cold GGUF gets --no-op-offload), start_timeout_s, lazy_load, idle_unload_s, before_load,
min_free_ram_mib, thinking_kwarg, api_key, host, allowed_hosts, task_parallel.  See examples/llamacpp-hotcold/.
"""
from __future__ import annotations

import dataclasses
import hmac
import http.client
import json
import os
import queue
import secrets
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import weakref
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from serve import context_pool, task_parallel

ROOT = Path(__file__).resolve().parents[1]

# request fields only Strata Void reads: never sent to llama-server
STRATA_FIELDS = ("task_parallel", "task_parallel_compose", "task_parallel_diagnostics", "context_items")
# what is passed through (llama-server's other routes - slot save/restore to disk, LoRA, the router - are not)
POST_ROUTES = {"/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/embeddings", "/v1/messages",
               "/v1/messages/count_tokens", "/tokenize", "/detokenize", "/apply-template"}
GET_ROUTES = {"/props", "/slots", "/metrics"}
ANTHROPIC_ROUTES = {"/v1/messages", "/v1/messages/count_tokens"}
OWN_OPTIONS = ("--host", "--port", "--api-key", "--api-key-file", "--path")   # Strata Void sets / forbids these
SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed", "repeat_penalty", "repetition_penalty",
            "frequency_penalty", "presence_penalty", "repeat_last_n")


class LoadError(RuntimeError):
    """llama-server could not be started (or was not allowed to start): a 503."""


class UpstreamError(RuntimeError):
    """llama-server answered an internal request with an error, or went away during it."""


def say(msg: str) -> None:
    print(msg, flush=True)                            # the console (or a service's log) shows it at once


# ------------------------------------------------------------------------------------------------ the process
def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def is_hotcold_gguf(path: str) -> bool:
    """Whether a GGUF holds hot/cold expert tensors (tools/moe_placement/gguf_hotcold.py): header only."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile  # noqa: PLC0415
    return any("_exps_hot." in t.name for t in GGUFFile(Path(path)).tensors)


def abort(conn: http.client.HTTPConnection) -> None:
    """Close a connection to llama-server now, also under a thread blocked reading it (close() alone does not wake
    that read).  llama-server stops a request whose connection closes."""
    try:
        if conn.sock is not None:
            conn.sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    conn.close()


def mem_available_mib() -> int | None:
    """Linux' MemAvailable, else None (the check is skipped where it cannot be read)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


class LlamaServer:
    """One llama-server child process: started on demand, health-checked, restarted after it died, stopped."""

    def __init__(self, exe: str, args: list, model: str | None = None, cwd: str | None = None,
                 env: dict | None = None, log: str | None = None, start_timeout_s: float = 1800.0,
                 hotcold: bool = False, alias: str | None = None, before_load=None, min_free_ram_mib: int = 0):
        for x in args:
            if str(x).split("=", 1)[0] in OWN_OPTIONS:
                raise ValueError(f"llama-server option {x} is set by Strata Void (it listens on a private port with "
                                 "its own key): leave it out of \"args\"; clients use Strata Void's host, port and "
                                 "api_key")
        self.exe, self.args, self.model, self.cwd = exe, [str(x) for x in args], model, cwd
        self.env, self.log_path, self.start_timeout_s = dict(env or {}), log, float(start_timeout_s)
        self.hotcold, self.alias = hotcold, alias
        self.before_load, self.min_free_ram_mib = before_load, int(min_free_ram_mib or 0)
        self.host, self.port = "127.0.0.1", 0
        self.key = secrets.token_urlsafe(24)          # the child's own key: only Strata Void talks to it
        self.proc: subprocess.Popen | None = None
        self.props: dict = {}
        self.starts = 0
        self.lock = threading.Lock()
        self._log = None
        self._conns: "weakref.WeakSet[http.client.HTTPConnection]" = weakref.WeakSet()

    # -- the command line
    def argv(self) -> list:
        a = [self.exe] + self.args
        if self.model:
            a += ["--model", self.model]
        if self.alias and "--alias" not in self.args and "-a" not in self.args:
            a += ["--alias", self.alias]
        if self.hotcold and "--no-op-offload" not in self.args:
            a.append("--no-op-offload")               # op offload reads expert ids before the hot/cold split runs
        return a + ["--host", self.host, "--port", str(self.port)]

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def loaded(self) -> bool:
        return self.alive() and bool(self.props)

    def log_tail(self, n: int = 15) -> str:
        if not self.log_path:
            return ""
        try:
            lines = Path(self.log_path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])

    # -- start / stop
    def ensure(self, say=say) -> None:
        """Start llama-server if it is not running; returns once the model is loaded.  Raises LoadError."""
        with self.lock:
            if self.loaded():
                return
            if self.proc is not None:                 # it died: say how, then start it again
                code = self.proc.poll()
                say(f"[llamacpp] llama-server exited (code {code}); starting it again")
                self._stop()
            self._before_load(say)
            self._start(say)

    def _before_load(self, say) -> None:
        if self.before_load:
            cmd = self.before_load
            say(f"[llamacpp] before loading: {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}")
            try:
                r = subprocess.run(cmd, shell=isinstance(cmd, str), timeout=120, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.SubprocessError) as e:
                raise LoadError(f"the before_load command failed ({e}); the model stays unloaded") from e
            if r.returncode != 0:
                raise LoadError(f"the before_load command refused the load (exit code {r.returncode}); the model "
                                "stays unloaded")
        if self.min_free_ram_mib:
            free = mem_available_mib()
            if free is not None and free < self.min_free_ram_mib:
                raise LoadError(f"only {free} MiB of host RAM available, the model needs {self.min_free_ram_mib} "
                                "(min_free_ram_mib): another model or program is using it - the model stays unloaded")

    def _start(self, say) -> None:
        self.port = free_port(self.host)
        env = dict(os.environ)
        env.update({k: str(v) for k, v in self.env.items()})
        env["LLAMA_API_KEY"] = self.key               # by environment: not visible in the process list
        env.pop("LLAMA_ARG_HOST", None)
        env.pop("LLAMA_ARG_PORT", None)
        if self.log_path:
            self._log = open(self.log_path, "a", encoding="utf-8")
            self._log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(self.argv())}\n")
            self._log.flush()
        say(f"[llamacpp] starting llama-server ({Path(self.exe).name}"
            + (f", {Path(self.model).name}" if self.model else "") + (", hot/cold" if self.hotcold else "") + ")")
        try:
            self.proc = subprocess.Popen(self.argv(), cwd=self.cwd, env=env, stdin=subprocess.DEVNULL,
                                         stdout=self._log or subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                         start_new_session=(os.name != "nt"))
        except OSError as e:
            self.proc = None
            raise LoadError(f"cannot start llama-server ({self.exe}): {e}") from e
        self.starts += 1
        t0 = time.time()
        while True:
            code = self.proc.poll()
            if code is not None:
                tail = self.log_tail()
                self._stop()
                raise LoadError(f"llama-server exited with code {code} while loading"
                                + (f"; the end of its log:\n{tail}" if tail else " (set \"log\" in the config to see why)"))
            try:
                status, _ = self.call("GET", "/health", timeout=5)
            except (OSError, http.client.HTTPException):
                status = 0
            if status == 200:
                break
            if time.time() - t0 > self.start_timeout_s:
                self._stop()
                raise LoadError(f"llama-server did not load the model in {self.start_timeout_s:.0f} s "
                                "(start_timeout_s)")
            time.sleep(0.25)
        try:
            status, props = self.call("GET", "/props", timeout=30)
        except (OSError, http.client.HTTPException) as e:
            self._stop()
            raise LoadError(f"llama-server loaded but /props failed: {e}") from e
        self.props = props if status == 200 and isinstance(props, dict) else {"total_slots": 1}
        say(f"[llamacpp] loaded in {time.time() - t0:.0f} s: {self.slots()} slot(s) of {self.slot_ctx()} tokens")

    def _stop(self) -> None:
        for c in list(self._conns):                   # running requests end first: llama-server then exits at once
            abort(c)
        p, self.proc, self.props = self.proc, None, {}
        if p is not None and p.poll() is None:
            try:
                p.terminate()                         # llama-server frees its memory and exits on SIGTERM
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(10)
            except OSError:
                pass
        if self._log is not None:
            self._log.close()
            self._log = None

    def close(self) -> None:
        with self.lock:
            self._stop()

    # -- what it reports
    def slots(self) -> int:
        return max(1, int(self.props.get("total_slots") or 1))

    def slot_ctx(self) -> int:
        return int((self.props.get("default_generation_settings") or {}).get("n_ctx") or 0)

    # -- HTTP to the child
    def connection(self, timeout: float | None = None) -> http.client.HTTPConnection:
        c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        self._conns.add(c)
        return c

    def headers(self, content_type: str = "application/json") -> dict:
        return {"Authorization": f"Bearer {self.key}", "Content-Type": content_type}

    def call(self, method: str, path: str, body=None, timeout: float = 60.0):
        """-> (status, parsed JSON or None).  Only for small, quick requests (health, props, tokenize)."""
        c = self.connection(timeout)
        try:
            c.request(method, path, None if body is None else json.dumps(body).encode(), self.headers())
            r = c.getresponse()
            data = r.read()
            try:
                return r.status, json.loads(data) if data else None
            except ValueError:
                return r.status, None
        finally:
            c.close()


# ------------------------------------------------------------------------------------------------ SSE helpers
def sse_events(resp):
    """The `data:` payloads of a server-sent-events response, as parsed JSON ("[DONE]" ends it)."""
    while True:
        line = resp.readline()
        if not line:
            return
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            return
        try:
            yield json.loads(data)
        except ValueError:
            continue


def close_on(cancel: threading.Event, conn: http.client.HTTPConnection, done: threading.Event) -> None:
    """Close `conn` when `cancel` is set (llama-server stops a request whose connection closes)."""
    def watch():
        while not done.wait(0.2):
            if cancel.is_set():
                abort(conn)
                return
    threading.Thread(target=watch, daemon=True, name="llamacpp-cancel").start()


def thinking_kwargs(body: dict, kwarg: str | None, on: bool) -> None:
    if kwarg:
        kw = dict(body.get("chat_template_kwargs") or {})
        kw[kwarg] = on
        body["chat_template_kwargs"] = kw


# ------------------------------------------------------------------------------------------------ the Backend
class LlamaCppBackend:
    """serve/task_parallel.py's Backend on llama-server: an internal request is an ordinary streamed chat completion
    in one of its parallel slots (--parallel N).  llama-server reuses a slot's cached prompt prefix by itself
    (cache_prompt), so the shared prefix needs no checkpoint hint here."""

    def __init__(self, server: LlamaServer, req: dict | None = None, thinking_kwarg: str | None = "enable_thinking",
                 timeout_s: float = 3600.0):
        self.server, self.thinking_kwarg, self.timeout_s = server, thinking_kwarg, timeout_s
        req = req or {}
        self.sampling = {k: req[k] for k in SAMPLING if req.get(k) is not None}

    def concurrency(self) -> int:
        return self.server.slots()

    def context(self) -> int:
        return self.server.slot_ctx()

    def count_text(self, text: str) -> int:
        status, out = self.server.call("POST", "/tokenize", {"content": text})
        if status != 200 or not isinstance(out, dict):
            raise UpstreamError(f"/tokenize answered {status}")
        return len(out.get("tokens") or [])

    def count_tokens(self, call) -> int:
        body = {"messages": call.messages}
        thinking_kwargs(body, self.thinking_kwarg, call.thinking != "off")
        status, out = self.server.call("POST", "/apply-template", body)
        if status != 200 or not isinstance(out, dict) or not isinstance(out.get("prompt"), str):
            raise UpstreamError(f"/apply-template answered {status}")
        return self.count_text(out["prompt"])

    def busy(self) -> int:
        """Slots processing a request right now (AUTO's load signal); 0 when /slots is off (--no-slots)."""
        status, out = self.server.call("GET", "/slots", timeout=10)
        if status != 200 or not isinstance(out, list):
            return 0
        return sum(1 for s in out if isinstance(s, dict) and s.get("is_processing"))

    def request_body(self, call) -> dict:
        body = {"messages": call.messages, "stream": True, "stream_options": {"include_usage": True},
                "cache_prompt": True}
        if call.role == "planner":
            body["temperature"] = 0.0                 # the plan is greedy, as on the native path
        else:
            body.update(self.sampling)
        if call.max_tokens and call.max_tokens > 0:
            body["max_tokens"] = int(call.max_tokens)
        thinking_kwargs(body, self.thinking_kwarg, call.thinking != "off")
        if call.reasoning_budget:
            body["reasoning_budget_tokens"] = int(call.reasoning_budget)
        return body

    def generate(self, call, cancel: threading.Event):
        r = task_parallel.Result(t_start=time.time())
        conn = self.server.connection(self.timeout_s)
        done = threading.Event()
        close_on(cancel, conn, done)
        text, reasoning = [], 0
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps(self.request_body(call)).encode(),
                         self.server.headers())
            resp = conn.getresponse()
            if resp.status != 200:
                raise UpstreamError(f"llama-server answered {resp.status}: {resp.read()[:300]!r}")
            for ev in sse_events(resp):
                if "error" in ev:
                    raise UpstreamError(f"llama-server: {ev['error']}")
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("reasoning_content"):
                        reasoning += 1                # one delta per token
                        if r.t_first is None:
                            r.t_first = time.time()
                    if d.get("content"):
                        if r.t_first is None:
                            r.t_first = time.time()
                        text.append(d["content"])
                        if call.on_text is not None:
                            call.on_text(d["content"])
                    if ch.get("finish_reason"):
                        r.finish = ch["finish_reason"]
                u = ev.get("usage")
                if u:
                    r.completion_tokens = int(u.get("completion_tokens") or 0)
                    r.prompt_tokens = int(u.get("prompt_tokens") or 0)
                t = ev.get("timings")
                if t:
                    r.reused_tokens = int(t.get("cache_n") or 0)
        except (OSError, http.client.HTTPException, ValueError, AttributeError) as e:
            # (a connection closed under http.client's chunked reader can surface as any of these)
            if cancel.is_set():
                r.finish = "cancelled"
            else:
                raise UpstreamError(f"llama-server went away during the {call.role}'s request: {e}") from e
        finally:
            done.set()
            conn.close()
        r.text = "".join(text)
        r.reasoning_tokens = reasoning
        r.t_end = time.time()
        return r


# ------------------------------------------------------------------------------------------------ the service
def has_tool_turns(messages) -> bool:
    return any(isinstance(m, dict) and (m.get("role") == "tool" or m.get("tool_calls")) for m in messages or [])


def has_images(messages) -> bool:
    return any(isinstance(m, dict) and isinstance(m.get("content"), list) and
               any(isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image", "input_audio")
                   for p in m["content"]) for m in messages or [])


class LlamaCppService:
    def __init__(self, server: LlamaServer, model_name: str, aliases=(), api_key: str = "", host_names=(),
                 any_host: bool = False, idle_unload_s: float = 0.0, thinking_kwarg: str | None = "enable_thinking",
                 request_timeout_s: float = 3600.0, say=say):
        self.server, self.model_name, self.aliases = server, model_name, list(aliases)
        self.api_key, self.host_names, self.any_host = api_key, set(host_names), any_host
        self.idle_unload_s, self.thinking_kwarg = float(idle_unload_s or 0), thinking_kwarg
        self.request_timeout_s = request_timeout_s
        self.say = say
        self.task_parallel_default = None
        self.task_parallel_cfg = task_parallel.Config()
        self.task_parallel_provider = None
        self.host_allowed = lambda host: True       # serve/server.py's DNS-rebinding check, set by run()
        self._active, self._last = 0, time.time()
        self._mu = threading.Lock()

    # -- load / idle unload
    def ensure_loaded(self) -> None:
        self.server.ensure(self.say)

    def begin(self) -> None:
        with self._mu:
            self._active += 1
            self._last = time.time()

    def end(self) -> None:
        with self._mu:
            self._active -= 1
            self._last = time.time()

    def unload_if_idle(self, now: float | None = None) -> bool:
        if not self.idle_unload_s:
            return False
        with self._mu:
            idle = self._active == 0 and (now or time.time()) - self._last >= self.idle_unload_s
        if idle and self.server.alive():
            self.say(f"[llamacpp] idle for {self.idle_unload_s:.0f} s: unloading the model")
            self.server.close()
            return True
        return False

    def start_idle_unload(self) -> None:
        if self.idle_unload_s:
            def loop():
                while True:
                    time.sleep(min(30.0, max(1.0, self.idle_unload_s / 4)))
                    self.unload_if_idle()
            threading.Thread(target=loop, daemon=True, name="llamacpp-idle").start()

    def models(self) -> dict:
        now = int(time.time())
        return {"object": "list", "data": [{"id": n, "object": "model", "created": now, "owned_by": "strata-void"}
                                           for n in [self.model_name, *self.aliases]]}

    def health(self) -> dict:
        state = "ok" if self.server.loaded() else "unloaded"
        return {"status": state, "engine": "llamacpp", "model": self.model_name, "loaded": self.server.loaded(),
                "slots": self.server.slots() if self.server.loaded() else None,
                "slot_context": self.server.slot_ctx() if self.server.loaded() else None,
                "starts": self.server.starts}


def error_body(api: str, status: int, message: str) -> dict:
    if api == "anthropic":
        kind = {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error",
                503: "overloaded_error"}.get(status, "api_error")
        return {"type": "error", "error": {"type": kind, "message": message}}
    kind = {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error"}.get(status, "server_error")
    return {"error": {"type": kind, "message": message, "code": status}}


def make_handler(svc: LlamaCppService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "StrataVoid-llamacpp"

        def log_message(self, fmt, *args):            # quiet, like serve/server.py
            pass

        # -- replies
        def _json(self, status: int, obj) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str, api: str = "openai") -> None:
            self._json(status, error_body(api, status, message))

        def _sse_headers(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def _sse(self, obj) -> None:
            self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

        # -- checks
        def _allowed(self, api: str) -> bool:
            if not svc.host_allowed(self.headers.get("Host")):
                self._error(403, "this server does not answer to that host name (allowed_hosts)", api)
                return False
            if self.path.split("?", 1)[0] in ("/", "/health"):
                return True
            if svc.api_key:
                auth = self.headers.get("Authorization", "")
                given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
                if not hmac.compare_digest(given.encode(), svc.api_key.encode()):
                    self._error(401, "a valid API key is required (Authorization: Bearer ... or x-api-key)", api)
                    return False
            return True

        def _watch_client(self, gone) -> threading.Event:
            """Calls gone() when the client closes its connection (a request in its prompt sends nothing to notice
            a broken pipe with).  Set the returned event to stop watching."""
            stop = threading.Event()
            sock = self.connection

            def watch():
                while not stop.is_set():
                    try:
                        r, _, _ = select.select([sock], [], [], 0.5)
                        if r and not sock.recv(1, socket.MSG_PEEK):
                            gone()
                            return
                    except (OSError, ValueError):
                        gone()
                        return
            threading.Thread(target=watch, daemon=True, name="llamacpp-client").start()
            return stop

        # -- routes
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if not self._allowed("openai"):
                return
            if path == "/health":
                return self._json(200, svc.health())
            if path == "/":
                return self._json(200, {"engine": "llamacpp", "model": svc.model_name,
                                        "api": ["/v1/chat/completions", "/v1/messages", "/v1/models"]})
            if path in ("/v1/models", "/models"):
                return self._json(200, svc.models())
            if path in GET_ROUTES:
                return self._forward("GET", self.path, None)
            self._error(404, f"no route GET {path}")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            api = "anthropic" if path in ANTHROPIC_ROUTES else "openai"
            if not self._allowed(api):
                return
            if path not in POST_ROUTES:
                return self._error(404, f"no route POST {path}", api)
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0:
                return self._error(411, "a Content-Length is required", api)
            body = self.rfile.read(n)
            if path != "/v1/chat/completions":
                return self._forward("POST", self.path, body, api)
            try:
                req = json.loads(body or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("the body must be a JSON object")
            except ValueError as e:
                return self._error(400, f"invalid JSON: {e}")
            tp_value = req.get("task_parallel", svc.task_parallel_default)
            if tp_value is not None:
                try:
                    mode = task_parallel.parse_mode(tp_value)
                except ValueError as e:
                    return self._error(400, str(e))
                items = req.get("context_items")
                if items is not None and not isinstance(items, list):
                    return self._error(400, "context_items must be a list")
                comp = req.get("task_parallel_compose")
                if comp is not None and comp not in ("synthesis", "sections", "direct", "auto"):
                    return self._error(400, "task_parallel_compose: synthesis, sections, direct or auto")
                messages = req.get("messages") or []
                fmt = (req.get("response_format") or {}).get("type") if isinstance(req.get("response_format"), dict) \
                    else None
                if not mode.off and not req.get("tools") and not has_tool_turns(messages) and not has_images(messages) \
                        and fmt in (None, "text"):
                    return self._task_parallel(req, mode)
            for k in STRATA_FIELDS:
                req.pop(k, None)
            return self._forward("POST", self.path, json.dumps(req).encode())

        # -- passthrough
        def _forward(self, method: str, path: str, body: bytes | None, api: str = "openai"):
            svc.begin()                               # active before loading: an idle unload must not interleave
            try:
                try:
                    svc.ensure_loaded()
                except LoadError as e:
                    return self._error(503, str(e), api)
                self._upstream(method, path, body, api,
                               self.headers.get("Content-Type") or "application/json")
            finally:
                svc.end()

        def _upstream(self, method: str, path: str, body: bytes | None, api: str = "openai",
                      content_type: str = "application/json", on_done=None, headers_sent: bool = False) -> None:
            """Sends one request to llama-server and copies its answer to the client (_relay_body).  The client is
            watched from the start: one that leaves - also while a whole answer is still being generated, before
            llama-server has sent anything - closes the upstream connection, which stops the request there."""
            conn = svc.server.connection(svc.request_timeout_s)
            gone = threading.Event()
            stop = self._watch_client(lambda: (gone.set(), abort(conn)))
            try:
                try:
                    conn.request(method, path, body, svc.server.headers(content_type))
                    resp = conn.getresponse()
                except (OSError, http.client.HTTPException, ValueError, AttributeError) as e:
                    if gone.is_set():
                        return
                    why = ("llama-server stopped; the next request starts it again" if not svc.server.alive()
                           else f"llama-server: {e}")
                    if headers_sent:
                        self.wfile.write(b"data: " + json.dumps(error_body(api, 502, why)).encode()
                                         + b"\n\ndata: [DONE]\n\n")
                    else:
                        self._error(502, why, api)
                    return
                ctype = resp.getheader("Content-Type", "application/json")
                self._relay_body(conn, resp, ctype, {"usage": {}, "reasoning": 0}, gone, on_done, headers_sent, api)
            except OSError:                           # the client went away while we wrote
                abort(conn)
            finally:
                stop.set()
                conn.close()

        def _relay_body(self, conn, resp, ctype, seen, gone, on_done, headers_sent, api) -> dict:
            """Copies llama-server's answer to the client: a stream line by line as it comes, else whole.  With
            on_done(usage, reasoning_deltas) -> extra: that chunk is added to a stream before its [DONE] (or a whole
            answer's JSON gets its keys).  A stream cut off before its end gets an error event.  Returns `seen`."""
            if "text/event-stream" not in ctype:
                status = resp.status
                try:
                    data = resp.read()
                except (OSError, http.client.HTTPException, ValueError, AttributeError):
                    if gone.is_set():                 # the client left: nothing to answer
                        return seen
                    status, ctype = 502, "application/json"
                    data = json.dumps(error_body(api, 502, "llama-server stopped during the answer; the next "
                                                           "request starts it again")).encode()
                if on_done is not None and status == 200:
                    try:
                        obj = json.loads(data)
                        obj.update(on_done(obj.get("usage") or {}, 0) or {})
                        data = json.dumps(obj, ensure_ascii=False).encode()
                    except (ValueError, AttributeError):
                        pass
                if headers_sent:                      # the stream is already open: an error before any token
                    if status != 200:
                        try:
                            err = json.loads(data)
                            err = err if isinstance(err, dict) and "error" in err else None
                        except ValueError:
                            err = None
                        data = json.dumps(err or error_body(api, status, data.decode("utf-8", "replace")[:500])).encode()
                    self.wfile.write(b"data: " + data + b"\n\ndata: [DONE]\n\n")
                    return seen
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return seen
            if not headers_sent:
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
            ended = False                             # the stream's own end was seen ([DONE] / message_stop)
            try:
                while True:
                    try:
                        line = resp.readline()
                    except (OSError, http.client.HTTPException, ValueError, AttributeError):
                        line = b""
                    if not line:
                        break
                    s = line.strip()
                    if s in (b"data: [DONE]", b"event: message_stop", b"event: error"):
                        ended = True
                    if on_done is not None and s.startswith(b"data:"):
                        payload = s[5:].strip()
                        if payload == b"[DONE]":
                            extra = on_done(seen["usage"], seen["reasoning"])
                            if extra:
                                self.wfile.write(b"data: " + json.dumps(extra, ensure_ascii=False).encode() + b"\n\n")
                            on_done = None
                        else:
                            try:
                                ev = json.loads(payload)
                                if ev.get("usage"):
                                    seen["usage"] = ev["usage"]
                                seen["reasoning"] += sum(1 for c in ev.get("choices") or []
                                                         if (c.get("delta") or {}).get("reasoning_content"))
                            except (ValueError, AttributeError):
                                pass
                    self.wfile.write(line)
                    if not s:
                        self.wfile.flush()
                if not ended and not gone.is_set():   # cut off: say why, then end the stream properly
                    self._stream_error(api, "the answer ended early")
                self.wfile.flush()
            except OSError:                           # the client went away: closing upstream stops the request
                abort(conn)
            return seen

        def _stream_error(self, api: str, what: str) -> None:
            proc = svc.server.proc
            try:
                dead = proc is None or proc.wait(2) is not None
            except subprocess.TimeoutExpired:
                dead = False
            msg = ("llama-server stopped during the answer; the next request starts it again" if dead else
                   f"{what}: llama-server closed the stream")
            if api == "anthropic":
                self.wfile.write(b"event: error\ndata: " + json.dumps(error_body(api, 502, msg)).encode() + b"\n\n")
            else:
                self.wfile.write(b"data: " + json.dumps(error_body(api, 502, msg)).encode() + b"\n\ndata: [DONE]\n\n")

        # -- task-parallel requests
        def _task_parallel(self, req: dict, mode):
            """serve/task_parallel.py on llama-server: plan, subtasks at once in its slots, then the answer, streamed
            like any answer.  Only the answer reaches the client (with a final `task_parallel` object); a failure of
            the orchestration falls back to answering the request as usual."""
            svc.begin()                               # active before loading: an idle unload must not interleave
            try:
                try:
                    svc.ensure_loaded()
                except LoadError as e:
                    return self._error(503, str(e))
                self._task_parallel_run(req, mode)
            finally:
                svc.end()

        def _task_parallel_run(self, req: dict, mode):
            cancel = threading.Event()
            stream = bool(req.get("stream"))
            diagnostics = req.get("task_parallel_diagnostics") is True
            kw = req.get("chat_template_kwargs") or {}
            client_thinks = not (svc.thinking_kwarg and kw.get(svc.thinking_kwarg) is False)
            backend = LlamaCppBackend(svc.server, req, svc.thinking_kwarg, svc.request_timeout_s)
            cfg = svc.task_parallel_cfg
            if req.get("task_parallel_compose") and req["task_parallel_compose"] != cfg.compose:
                cfg = dataclasses.replace(cfg, compose=req["task_parallel_compose"])
            orch = task_parallel.Orchestrator(backend, cfg)
            events: queue.Queue = queue.Queue()
            orch.progress = lambda stage: events.put(("progress", stage))
            messages = req.get("messages") or []
            watch = self._watch_client(cancel.set)
            if stream:
                self._sse_headers()

            def work():
                try:
                    events.put(("outcome", orch.run(messages, mode, cancel, "on" if client_thinks else "off",
                                                    context_items=req.get("context_items"),
                                                    provider=svc.task_parallel_provider)))
                except Exception as e:  # noqa: BLE001 - the orchestration's own failure: answer as usual
                    events.put(("outcome", task_parallel.Outcome("direct", meta={
                        "mode": "auto" if mode.kind == "auto" else str(mode.workers), "enabled": False,
                        "workers": 1, "fallback": f"orchestration error: {type(e).__name__}: {e}"})))
            threading.Thread(target=work, daemon=True, name="llamacpp-task-parallel").start()
            outcome = None
            try:
                while outcome is None:
                    try:
                        kind, x = events.get(timeout=5.0)
                    except queue.Empty:
                        kind, x = "ping", None
                    if kind == "outcome":
                        outcome = x
                    elif stream:
                        self.wfile.write((f": task_parallel {x}" if kind == "progress" else ": keep-alive").encode()
                                         + b"\n\n")
                        self.wfile.flush()
            except OSError:
                cancel.set()
                return
            finally:
                watch.set()
            if cancel.is_set():
                return
            meta, t_answer = outcome.meta, time.time()
            if outcome.kind == "composed":
                return self._composed(req, outcome, t_answer, stream, diagnostics, cancel)
            if outcome.kind == "synthesize":
                synth = outcome.synthesis
                body = {"messages": synth.messages, "stream": stream, "cache_prompt": True}
                body.update({k: req[k] for k in SAMPLING if req.get(k) is not None})
                max_req = int(req.get("max_completion_tokens") or req.get("max_tokens") or 0)
                cap = svc.task_parallel_cfg.synthesis_max_tokens
                one = (meta or {}).get("compose") == "one answer, shared prefix"
                if max_req > 0:
                    body["max_tokens"] = max_req
                elif cap > 0 and not one:
                    body["max_tokens"] = cap          # a synthesis over long notes can loop for very long
                thinking_kwargs(body, svc.thinking_kwarg, synth.thinking != "off")
                if req.get("reasoning_budget_tokens") is not None:
                    body["reasoning_budget_tokens"] = req["reasoning_budget_tokens"]
                elif synth.reasoning_budget:
                    body["reasoning_budget_tokens"] = synth.reasoning_budget
            else:                                     # "direct": the request as usual
                body = {k: v for k, v in req.items() if k not in STRATA_FIELDS}
            if stream:
                body["stream_options"] = {"include_usage": True}

            def on_done(usage: dict, reasoning: int):
                m = task_parallel.finish_meta(meta, t_answer, int(usage.get("completion_tokens") or 0), reasoning,
                                              diagnostics=diagnostics)
                if not stream:
                    return {"task_parallel": m}
                return {"id": "chatcmpl-tp", "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": svc.model_name, "choices": [], "task_parallel": m}
            self._upstream("POST", "/v1/chat/completions", json.dumps(body).encode(), on_done=on_done,
                           headers_sent=stream)

        def _composed(self, req, outcome, t_start, stream, diagnostics, cancel):
            """The answer written in sections at the same time (task_parallel.Composer), streamed in order."""
            comp, meta = outcome.composer, outcome.meta
            cid, created, model = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time()), svc.model_name

            def chunk(delta, finish=None):
                return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

            def final():
                m = task_parallel.finish_meta(meta, t_start, 0, 0, diagnostics=diagnostics, composed=comp.finish())
                ct = m.get("writer_tokens", 0)
                return m, {"prompt_tokens": 0, "completion_tokens": ct, "total_tokens": ct}
            if not stream:
                text = "".join(comp.stream())
                if cancel.is_set():
                    return
                m, usage = final()
                return self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": model,
                                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                                     "finish_reason": "stop"}],
                                        "usage": usage, "task_parallel": m})
            try:
                self._sse(chunk({"role": "assistant", "content": ""}))
                for piece in comp.stream():
                    self._sse(chunk({"content": piece}))
                m, usage = final()
                last = chunk({}, "stop")
                last["usage"] = usage
                self._sse(last)
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                           "choices": [], "task_parallel": m})
                self.wfile.write(b"data: [DONE]\n\n")
            except OSError:
                cancel.set()

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):  # a client that went away is not an error worth a traceback
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            super().handle_error(request, client_address)


def serve(svc: LlamaCppService, host: str = "127.0.0.1", port: int = 8095) -> Server:
    httpd = Server((host, port), make_handler(svc))
    threading.Thread(target=httpd.serve_forever, daemon=True, name="llamacpp-http").start()
    return httpd


# ------------------------------------------------------------------------------------------------ from a config
def server_from_config(cfg: dict) -> LlamaServer:
    """The LlamaServer a config describes (relative exe / model / log paths are the config's cwd's).  ValueError."""
    if not cfg.get("exe"):
        raise ValueError("the config needs \"exe\": the llama-server binary (tools/llamacpp/build_hotcold.sh builds it)")
    cwd = cfg.get("cwd") or "."
    rel = lambda p: p if os.path.isabs(p) else os.path.abspath(os.path.join(cwd, p))  # noqa: E731
    exe, model = rel(cfg["exe"]), rel(cfg["model"]) if cfg.get("model") else None
    args = cfg.get("args") or []
    if not isinstance(args, list):
        raise ValueError("\"args\" must be a list of llama-server options")
    if not model and "--model" not in args and "-m" not in args:
        raise ValueError("the config needs \"model\" (a GGUF) or --model in \"args\"")
    hc = cfg.get("hotcold", "auto")
    if hc not in (True, False, "auto"):
        raise ValueError("\"hotcold\" must be true, false or \"auto\"")
    if hc == "auto":
        try:
            hc = bool(model) and os.path.exists(model) and is_hotcold_gguf(model)
        except (OSError, ValueError, KeyError) as e:
            raise ValueError(f"cannot read the model's GGUF header ({model}): {e}") from e
    if not hc and any("_exps_cold" in str(a) for a in args):
        hc = True                                     # an -ot rule for the cold parts: a hot/cold model
    return LlamaServer(exe, args, model=model, cwd=cfg.get("cwd"), env=cfg.get("env"),
                       log=rel(cfg["log"]) if cfg.get("log") else None,
                       start_timeout_s=float(cfg.get("start_timeout_s") or 1800), hotcold=hc,
                       alias=cfg.get("model_name"), before_load=cfg.get("before_load"),
                       min_free_ram_mib=int(cfg.get("min_free_ram_mib") or 0))


def task_parallel_from_config(svc: LlamaCppService, tp_cfg) -> None:
    if tp_cfg is None:
        return
    if isinstance(tp_cfg, dict):
        svc.task_parallel_cfg = task_parallel.Config.from_dict(
            {k: v for k, v in tp_cfg.items() if k != "context_provider"})
        default = tp_cfg.get("default")
        prov = tp_cfg.get("context_provider")
        if prov is not None:
            if not isinstance(prov, dict) or not isinstance(prov.get("url"), str):
                raise ValueError("task_parallel.context_provider needs a \"url\"")
            svc.task_parallel_provider = context_pool.HttpContextProvider(
                prov["url"], float(prov.get("timeout_s", 5.0)), dict(prov.get("headers") or {}))
    else:
        default = tp_cfg
    if default is not None and not task_parallel.parse_mode(default).off:
        svc.task_parallel_default = default


def run(a, cfg: dict, srv) -> int:
    """serve/server.py's main() for --engine llamacpp: `a` its parsed arguments, `srv` the serve.server module (its
    host-name checks are reused)."""
    try:
        server = server_from_config(cfg)
        if a.before_load:
            server.before_load = a.before_load
        api_key = a.api_key or cfg.get("api_key", "")
        allowed = srv.allowed_hosts_of(cfg.get("allowed_hosts"), os.environ.get("STRATA_ALLOWED_HOSTS", ""))
        svc = LlamaCppService(server, cfg.get("model_name") or Path(server.model or "model").stem,
                              aliases=cfg.get("aliases") or [], api_key=api_key,
                              idle_unload_s=a.idle_unload if a.idle_unload is not None else
                              float(cfg.get("idle_unload_s") or 0),
                              thinking_kwarg=cfg.get("thinking_kwarg", "enable_thinking"),
                              request_timeout_s=float(cfg.get("request_timeout_s") or 3600))
        task_parallel_from_config(svc, cfg.get("task_parallel"))
    except ValueError as e:
        raise SystemExit(f"[llamacpp] config {e}") from e
    if not os.path.exists(server.exe):
        raise SystemExit(f"[llamacpp] llama-server not found: {server.exe} (build it with "
                         "tools/llamacpp/build_hotcold.sh, or set \"exe\")")
    if server.model and not os.path.exists(server.model):
        raise SystemExit(f"[llamacpp] model not found: {server.model}")
    names = srv.host_names_for(a.host, allowed)
    svc.host_allowed = lambda host: srv.host_allowed(host, names, "*" in allowed or bool(svc.api_key))
    if svc.task_parallel_default is not None:
        print(f"[llamacpp] task-parallel requests: {svc.task_parallel_default} by default", flush=True)
    if not (a.lazy or cfg.get("lazy_load") is True):
        try:
            svc.ensure_loaded()
        except LoadError as e:
            print(f"[llamacpp] {e}", file=sys.stderr, flush=True)
            return 1
    else:
        print("[llamacpp] model unloaded; the first request loads it", flush=True)
    httpd = serve(svc, a.host, a.port)
    svc.start_idle_unload()
    here = "127.0.0.1" if a.host in ("0.0.0.0", "", "::") else a.host
    print(f"ready: http://{here}:{a.port}/v1  (llama.cpp engine; OpenAI: /v1/chat/completions, Anthropic: "
          f"/v1/messages{', API key required' if svc.api_key else ''})", flush=True)
    if a.host not in ("127.0.0.1", "localhost", "::1") and not svc.api_key:
        print("       WARNING: no API key - anyone on your network can use this model", flush=True)

    def on_sigterm(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, on_sigterm)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[llamacpp] stopping ...", flush=True)
        httpd.shutdown()
        server.close()
        print("[llamacpp] stopped", flush=True)
    return 0
