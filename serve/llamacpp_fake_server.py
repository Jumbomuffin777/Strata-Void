"""serve/llamacpp_fake_server.py - a stand-in for llama.cpp's llama-server, for serve/test_llamacpp.py (no GPU, no
model, no llama.cpp build).

    python serve/llamacpp_fake_server.py --model x.gguf --port 18080 --parallel 4 --ctx-size 16384

Takes llama-server's command line (unknown options are ignored) and LLAMA_API_KEY.  Routes: GET /health (503
"Loading model" for FAKE_LLAMA_LOAD_S seconds first), /props, /slots, /v1/models, /metrics; POST /tokenize (one token
per word), /apply-template, /v1/chat/completions and /v1/messages (whole or streamed).  A chat answer depends on the
last user message, like the native tests' mock engine answers by role: "STEP plan" -> a plan's JSON, "STEP subtask"
-> a note, "STEP outline" -> an outline's JSON, "STEP write section" -> that section, "STEP final answer" -> the
synthesis; "slow" answers one token per FAKE_LLAMA_TOKEN_S (default 0.2 s) and stops when its client leaves; "die" exits mid-answer; anything else
-> a greeting.  With chat_template_kwargs.enable_thinking true, a streamed answer starts with reasoning_content.

FAKE_LLAMA_LOG: append what the server saw (argv once, then every request: path, body, and cancellations) to that
file, one JSON line each.  FAKE_LLAMA_EXIT_AT_START: exit with that code before listening.
"""
from __future__ import annotations

import json
import os
import re
import select
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
STATE = {"busy": 0, "t0": time.time()}


def log(obj) -> None:
    path = os.environ.get("FAKE_LLAMA_LOG")
    if path:
        with LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")


def opt(argv, names, default=None):
    for i, x in enumerate(argv):
        for n in names:
            if x == n and i + 1 < len(argv):
                return argv[i + 1]
            if x.startswith(n + "="):
                return x.split("=", 1)[1]
    return default


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def answer(messages) -> str:
    last = next((text_of(m.get("content")) for m in reversed(messages or []) if m.get("role") == "user"), "")
    if "STEP plan" in last:
        if "Design the final answer" in last:
            return json.dumps({"effort": "medium", "parallelism": 2, "strategy": "partition", "subtasks": [
                {"heading": "Consistency", "objective": "Explain consistency", "words": 150, "after": False},
                {"heading": "Scaling", "objective": "Explain scaling", "words": 150, "after": False}]})
        return json.dumps({"effort": "medium", "parallelism": 2, "strategy": "partition",
                           "subtasks": ["Analyze consistency of each database", "Analyze scaling of each database"]})
    if "STEP subtask" in last:
        m = re.search(r"STEP subtask (\d+)", last)
        return f"Note {m.group(1) if m else '?'}: the facts this subtask found."
    if "STEP outline" in last:
        return json.dumps({"sections": [{"heading": "Answer", "covers": "everything", "words": 200}], "facts": []})
    if "STEP write section" in last:
        m = re.search(r'STEP write section (\d+)/\d+: "([^"]+)"', last)
        return f"## {m.group(2) if m else 'Section'}\nSection {m.group(1) if m else '?'} text."
    if "STEP final answer" in last:
        return "FINAL ANSWER written from the notes."
    if "slow" in last:
        return " ".join(["tick"] * 50)
    return "Hello from the fake llama-server."


def make_handler(argv):
    key = os.environ.get("LLAMA_API_KEY", "")
    n_par = int(opt(argv, ["--parallel", "-np"], "1"))
    n_ctx = int(opt(argv, ["--ctx-size", "-c"], "4096"))
    alias = opt(argv, ["--alias", "-a"], "fake-model")
    load_s = float(os.environ.get("FAKE_LLAMA_LOAD_S", "0"))
    tok_s = float(os.environ.get("FAKE_LLAMA_TOKEN_S", "0.2"))

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, status, obj):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _ok(self) -> bool:
            if time.time() - STATE["t0"] < load_s:
                self._json(503, {"error": {"message": "Loading model", "type": "unavailable_error", "code": 503}})
                return False
            if key and self.headers.get("Authorization") != f"Bearer {key}":
                self._json(401, {"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}})
                return False
            return True

        def do_GET(self):
            if self.path == "/health":
                if self._ok():
                    self._json(200, {"status": "ok"})
                return
            if not self._ok():
                return
            log({"path": self.path, "method": "GET"})
            if self.path == "/props":
                return self._json(200, {"default_generation_settings": {"n_ctx": n_ctx // max(1, n_par)},
                                        "total_slots": n_par, "model_alias": alias})
            if self.path == "/slots":
                with LOCK:
                    busy = STATE["busy"]
                return self._json(200, [{"id": i, "n_ctx": n_ctx // n_par, "is_processing": i < busy}
                                        for i in range(n_par)])
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": alias, "object": "model"}]})
            if self.path == "/metrics":
                data = b"llamacpp:requests_processing 0\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            self._json(404, {"error": {"message": "File Not Found", "code": 404}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if not self._ok():
                return
            log({"path": self.path, "body": body})
            if self.path == "/tokenize":
                return self._json(200, {"tokens": list(range(len(str(body.get("content", "")).split())))})
            if self.path == "/apply-template":
                return self._json(200, {"prompt": "\n".join(f"{m.get('role')}: {text_of(m.get('content'))}"
                                                            for m in body.get("messages") or [])})
            if self.path == "/v1/chat/completions":
                return self._chat(body)
            if self.path == "/v1/messages":
                return self._anthropic(body)
            self._json(404, {"error": {"message": "File Not Found", "code": 404}})

        def _words(self, body):
            text = answer(body.get("messages"))
            words = [w + " " for w in text.split(" ")]
            words[-1] = words[-1].rstrip()
            limit = int(body.get("max_tokens") or 0)
            if limit > 0 and len(words) > limit:
                return words[:limit], "length"
            return words, "stop"

        def _chat(self, body):
            words, finish = self._words(body)
            thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking") is True
            last = text_of(((body.get("messages") or [{}])[-1]).get("content"))
            usage = {"prompt_tokens": 7, "completion_tokens": len(words) + (2 if thinking else 0),
                     "total_tokens": 7 + len(words)}
            if not body.get("stream"):
                if "slow" in last:                    # like llama-server: notice a client that left while generating
                    for _ in words:
                        time.sleep(tok_s)
                        r, _, _ = select.select([self.connection], [], [], 0)
                        if r and not self.connection.recv(1, socket.MSG_PEEK):
                            return log({"cancelled": self.path, "last": last[:40]})
                return self._json(200, {"id": "chatcmpl-fake", "object": "chat.completion", "model": alias,
                                        "choices": [{"index": 0, "finish_reason": finish, "message": {
                                            "role": "assistant", "content": "".join(words)}}], "usage": usage})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def send(obj):
                data = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            def chunk(delta, fin=None):
                return {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": alias,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
            with LOCK:
                STATE["busy"] += 1
            try:
                send(chunk({"role": "assistant", "content": None}))
                if thinking:
                    send(chunk({"reasoning_content": "Let me "}))
                    send(chunk({"reasoning_content": "think."}))
                for i, w in enumerate(words):
                    if "die" in last and i == 2:
                        os._exit(3)
                    if "slow" in last:
                        time.sleep(tok_s)
                    send(chunk({"content": w}))
                send(chunk({}, finish))
                if (body.get("stream_options") or {}).get("include_usage"):
                    send({"id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": alias, "choices": [],
                          "usage": usage, "timings": {"cache_n": 3}})
                send(b"[DONE]")
                self.wfile.write(b"0\r\n\r\n")
            except OSError:
                log({"cancelled": self.path, "last": last[:40]})
            finally:
                with LOCK:
                    STATE["busy"] -= 1

        def _anthropic(self, body):
            words, _ = self._words(body)
            text = "".join(words)
            if not body.get("stream"):
                return self._json(200, {"id": "msg_fake", "type": "message", "role": "assistant", "model": alias,
                                        "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                                        "usage": {"input_tokens": 7, "output_tokens": len(words)}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for name, data in [("message_start", {"type": "message_start", "message": {"id": "msg_fake",
                                                                                      "role": "assistant"}}),
                               ("content_block_start", {"type": "content_block_start", "index": 0,
                                                        "content_block": {"type": "text", "text": ""}}),
                               *[("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                          "delta": {"type": "text_delta", "text": w}}) for w in words],
                               ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                               ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
                               ("message_stop", {"type": "message_stop"})]:
                self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()
            self.close_connection = True

    return H


def main(argv) -> int:
    code = os.environ.get("FAKE_LLAMA_EXIT_AT_START")
    if code:
        print("fake llama-server: failing at start as asked", flush=True)
        return int(code)
    log({"argv": argv, "pid": os.getpid(), "has_key": bool(os.environ.get("LLAMA_API_KEY"))})
    host, port = opt(argv, ["--host"], "127.0.0.1"), int(opt(argv, ["--port"], "8080"))
    srv = ThreadingHTTPServer((host, port), make_handler(argv))
    srv.daemon_threads = True
    print(f"fake llama-server listening on {host}:{port}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
