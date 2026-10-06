#!/usr/bin/env python3
"""Summarizes task_parallel_bench.py results: one row per (task, mode) with the wall-clock, the time to the first
answer text, the final answer's own decode rate, the phases of a task-parallel run, the subtasks' aggregate rate and
the tokens generated / prompt tokens read in total; then the per-task speedup of each mode against a baseline mode.

    python tools/task_parallel_report.py r1.jsonl [r2.jsonl ...] [--baseline off] [--markdown]"""
from __future__ import annotations

import argparse
import json
import statistics
import sys


def row(r: dict) -> dict:
    tp = r.get("task_parallel") or {}
    u = r.get("usage") or {}
    comp = u.get("completion_tokens") or 0
    first = r.get("t_first_content") or r.get("t_first_reasoning")
    out = {"task": r["task"], "mode": r["mode"], "wall_s": r["wall_s"], "first_answer_s": r.get("t_first_content"),
           "finish": r.get("finish"), "answer_chars": len(r.get("content") or ""),
           "answer_tokens": comp, "workers": tp.get("workers") if tp.get("enabled") else 1,
           "decision": tp.get("decision", "") if tp else "",
           "effort": tp.get("effort")}
    # the final answer's own decode rate: its tokens over the time it was being written (reasoning included)
    if first and r["wall_s"] > first:
        out["answer_tok_s"] = round(comp / (r["wall_s"] - first), 1)
    if tp.get("enabled"):
        wr = tp.get("worker_results") or []
        out.update(plan_s=(tp.get("planning_ms") or 0) / 1e3, workers_s=(tp.get("workers_ms") or 0) / 1e3,
                   synthesis_s=(tp.get("synthesis_ms") or 0) / 1e3, worker_tokens=tp.get("worker_tokens"),
                   worker_aggregate_tok_s=tp.get("worker_aggregate_tok_s"),
                   longest_worker_s=round(max((w.get("end_ms") or 0) for w in wr) / 1e3, 1) if wr else None,
                   last_first_token_s=round(max((w.get("first_token_ms") or 0) for w in wr) / 1e3, 1) if wr else None,
                   generated_total=tp.get("total_generated_tokens") or comp,
                   prompt_read=(tp.get("planner_prompt_tokens") or 0)
                   + sum((w.get("prompt_tokens") or 0) - (w.get("reused_tokens") or 0) for w in wr)
                   + (u.get("prompt_tokens") or 0)
                   - ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0))
    else:
        out.update(generated_total=(tp.get("total_generated_tokens") or 0) + comp if tp else comp,
                   prompt_read=(u.get("prompt_tokens") or 0)
                   - ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--baseline", default="off")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    rows = [row(json.loads(l)) for f in a.results for l in open(f) if l.strip()]
    cols = ["task", "mode", "workers", "wall_s", "first_answer_s", "answer_tokens", "answer_tok_s", "finish",
            "plan_s", "workers_s", "synthesis_s", "worker_aggregate_tok_s", "generated_total", "prompt_read",
            "decision"]
    if a.markdown:
        print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        vals = [r.get(c) for c in cols]
        vals = [(f"{v:.1f}" if isinstance(v, float) else ("" if v is None else str(v))) for v in vals]
        print(("| " + " | ".join(vals) + " |") if a.markdown else "  ".join(vals))
    # speedup per task against the baseline mode (median over repeats)
    by = {}
    for r in rows:
        by.setdefault(r["task"], {}).setdefault(r["mode"], []).append(r["wall_s"])
    print("\nwall-clock relative to", a.baseline, "(< 1 = faster)")
    for t, ms in by.items():
        if a.baseline not in ms:
            continue
        b = statistics.median(ms[a.baseline])
        print(f"  {t:20s} " + "  ".join(f"{m}={statistics.median(v) / b:.2f}" for m, v in ms.items() if m != a.baseline))
    return 0


if __name__ == "__main__":
    sys.exit(main())
