"""bench_hotcold.py - a repeatable single-stream benchmark for an OpenAI-compatible llama-server endpoint (directly, or
through Strata Void's --engine llamacpp): time to first token, prefill and decode speed at fixed context sizes, plus a
correctness check at every size.

    python tools/llamacpp/bench_hotcold.py --url http://127.0.0.1:8095 --contexts 8192,32768,65536 --out run.json

Every prompt is the same on every run: filler text (this repository's docs, cut at line boundaries) with a fact placed
near its start and asked for at its end ("needle"), sized with the server's own /tokenize. Generation is greedy
(temperature 0), thinking off or short (--template-kwargs), so two runs of the same model and settings must print the same text; the JSON
records it for that comparison. Prefill/decode speeds are llama-server's own `timings` (the stream's last chunk: decode = generated tokens / time
spent generating them, prompt excluded); with --decode N exactly N tokens are generated (ignore_eos);
TTFT is measured here. Run one request at a time: this measures single-stream speed, not throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEEDLE = "The maintenance code for the north gate is ORCHID-7431."
QUESTION = ("Answer in one short sentence, without explanation: what is the maintenance code for the north gate, "
            "according to the text above?")
EXPECT = "ORCHID-7431"
SHORT = ("Write a Python function that returns the n-th Fibonacci number iteratively, then explain in two sentences "
         "why it runs in linear time.")


def post(url: str, path: str, body: dict, key: str = "", timeout: float = 3600):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return urllib.request.urlopen(urllib.request.Request(url + path, json.dumps(body).encode(), h), timeout=timeout)


def n_tokens(url: str, key: str, text: str) -> int:
    with post(url, "/tokenize", {"content": text}, key) as r:
        return len(json.loads(r.read())["tokens"])


def filler_lines() -> list[str]:
    lines = []
    for p in sorted((ROOT / "docs").glob("*.md")):
        lines += [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return lines


def make_prompt(url: str, key: str, target: int) -> tuple[str, int]:
    """The user message of about `target` tokens (the chat template adds a little)."""
    lines = filler_lines()
    text_of = lambda k: "\n".join([lines[0], NEEDLE] + [lines[(i % (len(lines) - 1)) + 1] for i in range(k)])  # noqa: E731
    lo, hi = 1, 8
    while n_tokens(url, key, text_of(hi)) < target:
        lo, hi = hi, hi * 2
    while lo < hi - 1:                                 # the most lines that stay within the target
        mid = (lo + hi) // 2
        if n_tokens(url, key, text_of(mid)) <= target:
            lo = mid
        else:
            hi = mid
    body = text_of(lo) + "\n\n" + QUESTION
    return body, n_tokens(url, key, body)


def run_one(url: str, key: str, content: str, max_tokens: int, template_kwargs: dict, ignore_eos: bool = False,
            cache_prompt: bool = False) -> dict:
    body = {"messages": [{"role": "user", "content": content}], "stream": True, "temperature": 0, "top_k": 1,
            "max_tokens": max_tokens, "stream_options": {"include_usage": True}, "cache_prompt": cache_prompt,
            "ignore_eos": ignore_eos}
    if template_kwargs:
        body["chat_template_kwargs"] = template_kwargs
    t0 = time.time()
    first, text, reasoning, timings, usage, finish = None, [], [], {}, {}, None
    with post(url, "/v1/chat/completions", body, key) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            ev = json.loads(line[5:])
            if "error" in ev:
                raise RuntimeError(ev["error"])
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if (d.get("content") or d.get("reasoning_content")) and first is None:
                    first = time.time()
                text.append(d.get("content") or "")
                reasoning.append(d.get("reasoning_content") or "")
                finish = ch.get("finish_reason") or finish
            timings = ev.get("timings") or timings
            usage = ev.get("usage") or usage
    out = "".join(text)
    return {"ttft_s": round((first or time.time()) - t0, 3), "total_s": round(time.time() - t0, 3),
            "prompt_tokens": timings.get("prompt_n"), "cached_tokens": timings.get("cache_n"), "prefill_tok_s": round(timings.get("prompt_per_second", 0), 2),
            "generated_tokens": timings.get("predicted_n"), "decode_tok_s": round(timings.get("predicted_per_second", 0), 2),
            "finish": finish, "text": out, "reasoning": "".join(reasoning), "reasoning_chars": len("".join(reasoning)),
            "text_sha256": hashlib.sha256(out.encode()).hexdigest()[:16]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True, help="the server's base URL (no /v1)")
    ap.add_argument("--key", default="")
    ap.add_argument("--contexts", default="8192,32768,65536", help="prompt sizes in tokens; 0 = the short prompt only")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=1, help="runs per size (the first warms the caches)")
    ap.add_argument("--decode", type=int, default=0, metavar="N",
                    help="decode benchmark: generate exactly N tokens (ignore_eos) and keep the prompt cached, so a "
                         "repeat measures decode at the same depth without reading the prompt again")
    ap.add_argument("--template-kwargs", default='{"enable_thinking": false}',
                    help='chat_template_kwargs as JSON, e.g. \'{"reasoning_effort": "low"}\' for a template that always thinks')
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    a = ap.parse_args()
    results = {"label": a.label, "url": a.url, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": []}
    jobs = [("short", SHORT, None)]
    for c in [int(x) for x in a.contexts.split(",") if x.strip() and int(x) > 0]:
        prompt, n = make_prompt(a.url, a.key, c - a.max_tokens - 64)
        jobs.append((f"{c // 1024}K", prompt, n))
    for name, prompt, n in jobs:
        for i in range(a.repeat):
            r = run_one(a.url, a.key, prompt, a.decode or a.max_tokens, json.loads(a.template_kwargs),
                        ignore_eos=a.decode > 0, cache_prompt=a.decode > 0)
            r.update(name=name, attempt=i + 1,
                     needle_ok=None if name == "short" else EXPECT in r["text"] + r["reasoning"])
            results["runs"].append(r)
            print(f"{name:>6} #{i + 1}: prompt {r['prompt_tokens']} tok ({r['cached_tokens']} cached), TTFT {r['ttft_s']} s, prefill "
                  f"{r['prefill_tok_s']} tok/s, decode {r['decode_tok_s']} tok/s ({r['generated_tokens']} tok, "
                  f"{r['finish']}), needle {r['needle_ok']}: {r['text'][:90]!r}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))
    return 0 if all(r["needle_ok"] in (None, True) for r in results["runs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
