"""compare_logprobs.py - does a runtime change keep the model's outputs? Records greedy tokens with their top-k
log-probabilities from an OpenAI-compatible llama-server at fixed prompt sizes, then compares two recordings.

    python tools/llamacpp/compare_logprobs.py record --url http://127.0.0.1:8095 --contexts 8192,32768 --out a.json
    (restart the server with the change)
    python tools/llamacpp/compare_logprobs.py record --url http://127.0.0.1:8095 --contexts 8192,32768 --out b.json
    python tools/llamacpp/compare_logprobs.py compare a.json b.json

The prompts are bench_hotcold.py's (same text, same needle). `compare` reports, per prompt: whether the greedy token
sequences are identical, the first position where they differ, and the largest difference of the chosen token's
log-probability and of any top-k entry both runs list, over the common prefix. A change that only reorders a
floating-point sum shows identical tokens and differences of a few 1e-3; a wrong attention set shows divergence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_hotcold import SHORT, make_prompt, post  # noqa: E402


def record(a) -> int:
    out = {"url": a.url, "runs": []}
    jobs = [("short", SHORT)]
    for c in [int(x) for x in a.contexts.split(",") if x.strip() and int(x) > 0]:
        jobs.append((f"{c // 1024}K", make_prompt(a.url, a.key, c - a.max_tokens - 64)[0]))
    for name, prompt in jobs:
        body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_k": 1,
                "max_tokens": a.max_tokens, "logprobs": True, "top_logprobs": a.top, "cache_prompt": False,
                "ignore_eos": True,                     # always max_tokens tokens: a long comparison at every size
                "chat_template_kwargs": json.loads(a.template_kwargs)}
        with post(a.url, "/v1/chat/completions", body, a.key) as r:
            d = json.loads(r.read())
        ch = d["choices"][0]
        steps = []
        for e in (ch.get("logprobs") or {}).get("content") or []:
            steps.append({"t": e["token"], "lp": e["logprob"],
                          "top": {x["token"]: x["logprob"] for x in e.get("top_logprobs") or []}})
        out["runs"].append({"name": name, "steps": steps, "timings": d.get("timings")})
        print(f"{name}: {len(steps)} tokens recorded", flush=True)
    Path(a.out).write_text(json.dumps(out))
    return 0


def compare(a) -> int:
    x, y = json.loads(Path(a.a).read_text()), json.loads(Path(a.b).read_text())
    bad = 0
    for rx, ry in zip(x["runs"], y["runs"]):
        sx, sy = rx["steps"], ry["steps"]
        n = min(len(sx), len(sy))
        first = next((i for i in range(n) if sx[i]["t"] != sy[i]["t"]), None)
        upto = first if first is not None else n
        d_lp = max((abs(sx[i]["lp"] - sy[i]["lp"]) for i in range(upto)), default=0.0)
        d_top = 0.0
        for i in range(upto):
            for t, v in sx[i]["top"].items():
                if t in sy[i]["top"]:
                    d_top = max(d_top, abs(v - sy[i]["top"][t]))
        same = first is None and len(sx) == len(sy)
        bad += 0 if same else 1
        print(f"{rx['name']:>6}: {n} tokens, {'identical' if same else f'first difference at token {first}'}; "
              f"max |d logprob| chosen {d_lp:.4f}, top-k {d_top:.4f}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--url", required=True)
    r.add_argument("--key", default="")
    r.add_argument("--contexts", default="8192,32768")
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--top", type=int, default=5)
    r.add_argument("--template-kwargs", default='{"enable_thinking": false}')
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    return record(a) if a.cmd == "record" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
