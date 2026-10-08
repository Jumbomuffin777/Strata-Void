# Contributing

Thanks for looking. Strata Void is a small experimental fork; the most useful contributions are:

1. **Results from other hardware.** CUDA or ROCm GPUs with upstream Strata's batch slots, other Arc cards, other
   models and quants. Use the "Benchmark results" issue template and the commands in
   [docs/BENCHMARKS.md](docs/BENCHMARKS.md#reproduce). Compare modes on one machine; say how many runs per cell.
2. **Bug reports** with the request, the run config's engine args and the response's `task_parallel` object.
3. **Ideas and pull requests**, for example: speculative (MTP) drafts inside batch slots (parallel steps currently
   decode without drafts, ~20-30 tok/s per slot vs ~45 alone), interleaving a prompt read with decoding slots on the
   SYCL engine, better planner prompts, a faster long-document path.

## Where a change belongs

- Engine, setup, server or model support that is not specific to this fork: **upstream**
  ([Niko1221/Strata](https://github.com/Niko1221/Strata)), so everyone gets it. This fork merges upstream from time to time.
- Task-Parallel Requests, the context provider, `--slot-context` and the Arc prompt-path work: here.

## Working on it

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest serve.test_task_parallel serve.test_task_parallel_v2 serve.test_parallel -v
.venv/bin/python tools/test_setup_sycl.py
```

These run without a GPU (fake backends and a fake engine). The SYCL engine builds as in the
[README](README.md#build-linux-intel-arc).

- Keep the feature **off by default**: with `task_parallel` absent the server must behave exactly as upstream.
- New behavior gets a test against the fake backend; numbers in docs say what they were measured on and how often.
- Plain words in the docs, no claim without a measurement (the same rule as upstream's [AGENTS.md](AGENTS.md)).
- Never commit model weights, private paths, keys or benchmark outputs with personal data.
- By contributing you agree that your contribution is licensed under the MIT License of this repository.
