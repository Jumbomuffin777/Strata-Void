# Example run configs

The run configs behind the benchmarks in [docs/BENCHMARKS.md](../docs/BENCHMARKS.md), with the machine-specific
paths replaced by the relative paths the README's [model steps](../README.md#model-files) create. Start the server
from the repository root (relative paths are resolved against the config's `cwd`, which defaults to the directory
you start from):

```sh
source /opt/intel/oneapi/setvars.sh
.venv/bin/python -m serve.server --engine strata --config examples/strata-void-32k.json --port 8095
```

| config | context | KV | slots (`--batch` / groups) | slot size | layer split | measured with |
|---|---|---|---|---|---|---|
| `strata-void-32k.json` | 32K | fp16 | 4 / 2 | full (32K) | `18,35` | short complex requests, 8K/16K documents |
| `strata-void-64k.json` | 64K | fp16 | 3 / 3 | full (64K) | `18,35` | 32K documents |
| `strata-void-128k.json` | 128K | 8-bit | 4 / 2 | full (128K) | `18,36` | 64K and 120K documents |
| `strata-void-128k-slots16k.json` | 128K | 8-bit | 6 / 3 | `--slot-context 16384` | `18,36` | the context-ceiling probe: one long request plus six 16K slots |

Task-Parallel Requests need **full-size slots** to share a long document between their steps (every step restores
the planner's read of it). With `--slot-context` the slots are smaller: shorter requests still use them, and a
document larger than a slot is cut into slices instead (slower; see [docs/TASK_PARALLEL.md](../docs/TASK_PARALLEL.md)).

## Before you use them

- **The layer split is specific to this machine**: three cards in the order `level_zero:0,1,2` = Arc Pro B70 (32 GB),
  B65 (32 GB), B60 (24 GB), with every expert in VRAM. `18,35` means layers 0-17 on the first card, 18-34 on the
  second, the rest on the third. A different set of cards, card order, model or KV format needs its own split: start
  with upstream's [docs/MULTI_GPU.md](../docs/MULTI_GPU.md) and watch the engine log for experts that do not fit.
- The same split did not fit at every context: `18,34` refused 32K fp16 (17 experts short on the B60) and 128K 8-bit
  (38 short); moving layers from the B60 to the B65 (`18,35`, `18,36`) fixed it. The B65 and B60 give bit-identical
  results, so this changes no output.
- `STRATA_VERIFY_NO_HOST=1` is only valid when **every** expert is in VRAM (upstream's
  [docs/INTEL.md](../docs/INTEL.md) explains this and the other `STRATA_*` variables).
- Task-parallel is **off unless a request asks for it**. To make AUTO the server default, set
  `"task_parallel": {"default": "auto", "max_workers": 4}`.
- `max_workers` is capped by `--batch` (the engine's slots).
