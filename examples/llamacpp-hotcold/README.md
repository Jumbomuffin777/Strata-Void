# llama.cpp hot/cold configs

Server configs for `python -m serve.server --engine llamacpp --config ...`: a hot/cold MoE GGUF (GLM-5.3-Flash,
DeepSeek V4 Flash) served through the patched `llama-server` that `tools/llamacpp/build_hotcold.sh` builds. The
flow that makes the GGUF and the placement arguments is in [docs/HOTCOLD_LLAMACPP.md](../../docs/HOTCOLD_LLAMACPP.md).

| config | layout | notes |
|---|---|---|
| `server.json` | three local SYCL GPUs, cold experts in host RAM | 64K context in 2 slots of 32K (Task-Parallel Requests use them), loaded at start |
| `server-intel-only.json` | three local SYCL GPUs, cold experts in host RAM, the ORIGINAL GGUF split at load time (`LLAMA_MOE_HOTCOLD`, no rewritten copy) | Maya-S at 64K; sparse-attention decode on (`LLAMA_DSA_GATHER`); measured in [docs/INTEL_ONLY_MOE.md](../../docs/INTEL_ONLY_MOE.md) |
| `server-rpc.json` | three local SYCL GPUs + one llama.cpp RPC device, cold experts in host RAM | 1 slot, loaded on the first request, unloaded after 30 idle minutes |

Before you use them:

- Replace `models/MODEL-hotcold.gguf` with your rewritten GGUF, and `--device` / `--tensor-split` (and `--rpc`) with
  what `llamacpp_hotcold.py` printed for **your** devices and budgets. The values here are placeholders of the right
  form, not a placement for any particular machine.
- `RPC_HOST:50052` is a `ggml-rpc-server` on another host (built by the same script). A CUDA RPC device whose layers
  are split needs a GGUF rewritten with `--slot-zeros`; one planned with `--no-split` holds whole layers.
- `--no-op-offload` is required for hot/cold GGUFs; Strata Void adds it if missing (`"hotcold": "auto"`).
- `min_free_ram_mib` refuses a load while host RAM is short (the cold experts live there); set it to about the cold
  size the planner printed plus the KV and compute buffers that stay on the host. `before_load` can run a check of
  your own (exit code non-zero: no load), e.g. that no other large model is resident.
- `--host`, `--port` and `--api-key` belong to Strata Void (`host`, `--port`, `api_key`), not to `args`.
