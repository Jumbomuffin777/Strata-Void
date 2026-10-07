#!/usr/bin/env python3
"""Scores task_parallel_bench.py results: automatic checks where a task has them (Python tests run on the answer's
code in a subprocess with a timeout; "contains"), and a blind grading sheet for rubric tasks (answers shuffled under
random ids; the key is written separately so grading happens before unblinding).

    python tools/task_parallel_score.py --tasks bench/task_parallel_tasks.json --results r.jsonl --out score.json \
        [--blind sheet.md --key key.json]

Running a model's code: only do this in a sandbox you are willing to have that code run in."""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile


def code_blocks(text: str) -> list[str]:
    return re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)


def run_tests(code: str, tests: list, func: str | None) -> tuple[int, int, str]:
    prog = code + "\n\nimport json as _j\n_res = []\n"
    for call, expected in tests:
        if func and not call.startswith(func):
            call = f"{func}({call})"
        prog += (f"try:\n    _ok = ({call}) == ({expected})\nexcept Exception as _e:\n    _ok = False\n"
                 f"_res.append(bool(_ok))\n")
    prog += "print('RESULT', _j.dumps(_res))\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
    try:
        p = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return 0, len(tests), "timeout"
    m = re.search(r"RESULT (\[.*\])", p.stdout)
    if not m:
        return 0, len(tests), (p.stderr or p.stdout)[-300:]
    res = json.loads(m.group(1))
    return sum(res), len(tests), ""


def auto_score(task: dict, content: str) -> dict | None:
    chk = task.get("check") or {}
    kind = chk.get("type")
    if kind in ("python", "python_multi"):
        best = (0, len(chk["tests"]), "no code block")
        blocks = code_blocks(content)
        # the corrected function may be in any block; try each and the concatenation of all
        for code in blocks + (["\n\n".join(blocks)] if len(blocks) > 1 else []):
            r = run_tests(code, chk["tests"], chk.get("func"))
            if r[0] > best[0] or best[2] == "no code block":
                best = r
        return {"passed": best[0], "total": best[1], "note": best[2]}
    if kind == "numbers":
        # every numeric expectation must appear in the answer: some number within 0.5 % (at least 0.06 absolute,
        # for rounded percentages and years) of it; text expectations are left to the rubric
        text = re.sub(r"\{,\}|\\,|(?<=\d) (?=\d{3}\b)", ",", content or "")      # LaTeX / spaced thousands
        found = [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text)
                 if x.replace(",", "").replace(".", "").lstrip("-").isdigit()]
        nums = {k: v for k, v in chk["expected"].items() if isinstance(v, (int, float))}
        ok = [k for k, v in nums.items() if any(abs(f - v) <= max(0.005 * abs(v), 0.06) for f in found)]
        return {"passed": len(ok), "total": len(nums), "note": "missing " + ", ".join(k for k in nums if k not in ok)
                if len(ok) < len(nums) else ""}
    if kind == "contains":
        return {"passed": int(any(s.lower() in (content or "").lower() for s in chk["any"])), "total": 1, "note": ""}
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--results", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blind")
    ap.add_argument("--key")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    tasks = {t["id"]: t for t in json.load(open(a.tasks))}
    recs = [json.loads(l) for f in a.results for l in open(f) if l.strip()]
    out = []
    for r in recs:
        s = auto_score(tasks[r["task"]], r.get("content", ""))
        out.append({"task": r["task"], "mode": r["mode"], "rep": r.get("rep", 0), "auto": s,
                    "wall_s": r.get("wall_s"), "chars": len(r.get("content") or "")})
    json.dump(out, open(a.out, "w"), indent=1)
    if a.blind:
        rng = random.Random(a.seed)
        items = [r for r in recs if (tasks[r["task"]].get("check") or {}).get("rubric")]
        rng.shuffle(items)
        key, lines = {}, ["# Blind grading sheet\n"]
        by_task = {}
        for r in items:
            by_task.setdefault(r["task"], []).append(r)
        for tid, rs in by_task.items():
            lines.append(f"\n## Task {tid}\n\n**Prompt:** {tasks[tid]['prompt'][:3000]}\n\n**Rubric:**\n")
            lines += [f"- {x}" for x in tasks[tid]["check"]["rubric"]]
            for r in rs:
                aid = f"{tid}-{rng.randrange(16**6):06x}"
                key[aid] = {"mode": r["mode"], "rep": r.get("rep", 0)}
                lines.append(f"\n### Answer {aid}\n\n{r.get('content') or '(empty answer)'}\n")
        open(a.blind, "w").write("\n".join(lines))
        json.dump(key, open(a.key, "w"), indent=1)
    for o in out:
        print(o["task"], o["mode"], o["auto"], o["wall_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
