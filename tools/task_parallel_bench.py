#!/usr/bin/env python3
"""Task-parallel benchmark: the same requests answered normally and with task_parallel = 2 / 4 / 6 / auto, against a
running Strata server (serve/server.py).  Streams every request and records, per run, the wall-clock to the end of
the answer, the time to the first visible answer text, the answer, the usage and the response's task_parallel object.

    python tools/task_parallel_bench.py --url http://127.0.0.1:8095 --tasks bench/tasks.json \
        --modes off,2,4,6,auto --out results.jsonl

Greedy (temperature 0) by default so the modes are compared on the same model behavior.  The answers are kept for
grading; the worker texts never leave the server."""
from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
from urllib.parse import urlparse


def run_one(url: str, model: str, prompt: str, mode, max_tokens: int | None, temperature: float, timeout: float) -> dict:
    u = urlparse(url)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True,
            "temperature": temperature, "stream_options": {"include_usage": True}}
    # a mode is "off", "auto" or a worker count, optionally with "+b<N>": the request's reasoning_budget_tokens
    # (e.g. "off+b1536": an ordinary request whose thinking is capped like the synthesis' own)
    base, _, budget = str(mode).partition("+b")
    if budget:
        body["reasoning_budget_tokens"] = int(budget)
    if base not in ("None", "off"):
        body["task_parallel"] = int(base) if base.isdigit() else base
        body["task_parallel_diagnostics"] = True
    if max_tokens:
        body["max_tokens"] = max_tokens
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    rec = {"mode": str(mode), "status": r.status, "t_first_byte": None, "t_first_reasoning": None,
           "t_first_content": None, "progress": [], "content": "", "reasoning_chars": 0, "usage": None,
           "task_parallel": None, "finish": None, "error": None}
    content, reasoning = [], 0
    if r.status != 200:
        rec["error"] = r.read().decode()[:500]
        c.close()
        rec["wall_s"] = round(time.time() - t0, 3)
        return rec
    while True:
        line = r.fp.readline()
        if not line:
            break
        now = time.time() - t0
        if rec["t_first_byte"] is None:
            rec["t_first_byte"] = round(now, 3)
        line = line.decode().rstrip("\n")
        if line.startswith(": task_parallel"):
            rec["progress"].append([round(now, 3), line[len(": task_parallel "):]])
            continue
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if "error" in d:
            rec["error"] = d["error"]
            continue
        if d.get("task_parallel") is not None:
            rec["task_parallel"] = d["task_parallel"]
        if d.get("usage"):
            rec["usage"] = d["usage"]
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("reasoning_content"):
                reasoning += len(delta["reasoning_content"])
                if rec["t_first_reasoning"] is None:
                    rec["t_first_reasoning"] = round(now, 3)
            if delta.get("content"):
                content.append(delta["content"])
                if rec["t_first_content"] is None:
                    rec["t_first_content"] = round(now, 3)
            if ch.get("finish_reason"):
                rec["finish"] = ch["finish_reason"]
    c.close()
    rec["wall_s"] = round(time.time() - t0, 3)
    rec["content"] = "".join(content)
    rec["reasoning_chars"] = reasoning
    u_ = rec["usage"] or {}
    if rec["t_first_reasoning"] or rec["t_first_content"]:
        start = min(x for x in (rec["t_first_reasoning"], rec["t_first_content"]) if x)
        rec["visible_decode_tok_s"] = round((u_.get("completion_tokens") or 0) / max(1e-6, rec["wall_s"] - start), 2)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8095")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--ids", default="", help="comma-separated task ids (default: all)")
    ap.add_argument("--modes", default="off,2,4,6,auto")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = json.load(open(a.tasks))
    if a.ids:
        want = a.ids.split(",")
        tasks = [t for t in tasks if t["id"] in want]
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    with open(a.out, "a") as f:
        for rep in range(a.repeat):
            for t in tasks:
                for m in modes:
                    rec = run_one(a.url, a.model, t["prompt"], m, a.max_tokens or None, a.temperature, a.timeout)
                    rec.update(task=t["id"], category=t["category"], decomposable=t["decomposable"], rep=rep,
                               at=time.time())
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    tpm = rec.get("task_parallel") or {}
                    print(f"{t['id']:20s} {m:5s} wall {rec['wall_s']:7.1f}s  first answer "
                          f"{rec['t_first_content'] or 0:6.1f}s  tokens {(rec['usage'] or {}).get('completion_tokens')}"
                          f"  workers {tpm.get('workers', '-')}  {tpm.get('decision', '')[:60]}"
                          f"{'  ERROR ' + str(rec['error'])[:80] if rec['error'] else ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
