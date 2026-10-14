# Intel-only MoE work: where it stands and what is next

A handover for whoever continues the Intel-only GLM-5.3-Flash / DeepSeek V4 Flash work. Measurements:
[INTEL_ONLY_MOE.md](INTEL_ONLY_MOE.md). Flow and runtime switches: [HOTCOLD_LLAMACPP.md](HOTCOLD_LLAMACPP.md).

## Done and measured

- Both models run on three Intel Arc Pro GPUs + host RAM only, from their original GGUFs split at load time.
- Maya-S decode, gathered sparse attention on: 12.5 (short) / ~11.9 (8K) / 11.3 (32K) / 11.1 (64K) tok/s.
- DeepSeek V4 Flash decode (static, no gather): 10.3 / 9.3 / 7.6 / 6.4 tok/s.
- Kept: load-time split, CPU skip of absent experts, ubatch 1024, gathered DSA decode (opt-in), grouped MoE mat-vec
  for SYCL prefill (default on; `patches/llama.cpp-sycl-grouped-moe.patch`). Rejected after measurement: grouped
  oneMKL expert GEMM (prefill slower), f16 KV (no decode gain), fewer CPU threads (no gain).
- Maya-S prefill with the grouped MoE kernel: 8K 60.6 -> 86.8 tok/s, 32K 56.2 -> 75.8 tok/s; perplexity unchanged
  (1.8938 -> 1.8927); decode unchanged.

## Next, in order of expected value per risk

1. **DeepSeek decode at depth.** Finding (from the graph, 2026-10-14): its CSA layers (compression ratio 4) attend over
   the whole compressed cache plus the raw window under a mask that keeps the indexer's top-k, and pass `n_kv_max`
   (the bound on finite mask entries) to flash attention. CUDA, Vulkan and Metal kernels use that bound; ggml-sycl
   ignores it. `LLAMA_DSV4_GATHER=1` (opt-in, `patches/llama.cpp-dsv4-gather.patch`) gathers the top `n_kv_max`
   entries for single-token decode, which is exact by that contract. Measured: decode 7.48 -> 8.83 tok/s at 32K and
   6.32 -> 7.71 at 64K, greedy output identical to the dense path over 128 tokens at every size (INTEL_ONLY_MOE.md).
   Next: more runs before making it the default; the same idea for ggml-sycl's flash attention itself (honour
   `n_kv_max` inside the kernel) would help every model that sets it. HCA layers (ratio 128) and the window-only layers are already small. Earlier plan:
   profile it like GLM (`op-profile --decode 32 --sync ...` at 512 and 32K tokens) to
   find its depth term; if it is again attention over every cell under a mask, the same gather applies to its graph
   (`build_attn_sparse` / the DSA input of deepseek4), validated the same way (`test_dsa_gather.cpp`,
   `compare_logprobs.py` against a dense-vs-dense and q8-vs-f16 baseline).
2. **Prefill, next.** Done: the per-expert MUL_MAT_ID (dequantize to f16 + GEMM per expert) is replaced by one grouped
   launch with a decode-once IQ kernel (INTEL_ONLY_MOE.md, "Prefill: grouped MUL_MAT_ID"); the op is 1.6-2x faster at
   ubatch 1024 and 3.7-5.5x at 256, end to end +35-43 %. Next: profile a prefill ubatch again (`op-profile` with a
   1024-token batch) to rank what is left - expected: the DSA indexer and attention, the CPU cold part, and the hot
   experts with > 48 rows that still take the GEMM path (an XMX `joint_matrix` kernel over the decoded int8 tiles is
   the lever there). The decode-once idea also applies to the remaining codebook types (IQ2_XS, IQ3_S, IQ1_*).
3. **Hot-expert decode overhead.** 0.5-0.8 ms per layer per token where the weight reads take ~0.03 ms: per-kernel
   overhead of ~7 small kernels. SYCL graphs are off for multi-GPU and MUL_MAT_ID; a fused SYCL kernel for
   "ids -> gate/up -> swiglu -> down" of one token would remove most launches.
4. **GPU power profile.** All three GPUs run the driver's `power_saving` profile (sysfs `power_profile`), and in a
   layer split each GPU idles two thirds of every token. Switching to `base` needs root; measure decode before and
   after (it is a runtime setting, not firmware).
5. **Adaptive residency.** Implemented, unit-tested and measured: it raises the hot hit rate from 89-90 % to ~95.5 %
   on real prompts but leaves decode unchanged, because a miss is cheap here (INTEL_ONLY_MOE.md). It becomes worth
   it together with a runtime skip of the cold part for tokens that select no cold expert (then every extra hit saves
   a whole CPU round trip per layer), or with an NVMe tier.
6. **Native SYCL engine (Phase 3).** Project Maya (MIT, same Strata base) has a CUDA implementation of GLM-5.3-Flash
   (`src/core/glm_*.cu`: KDA, MLA/DSA, mHC, MTP) on the same expert-cache skeleton Strata Void's SYCL engine already
   ports. The first native step is a parity harness: one GLM layer's KDA and MLA/DSA kernels in SYCL against llama.cpp's
   outputs on the same inputs, before any end-to-end path. The llama.cpp backend stays the reference and fallback.

## Open issues

- Intermittent `GGML_ASSERT` "expert id out of range" in ggml-sycl's mul_mat_id (an invalid expert id reaching a GPU):
  once in a Maya-S warmup (f16 KV + gathered attention), and once in a DeepSeek 64K prefill with the grouped MoE path
  (2026-10-14). The second run of the same prompt with the same prefill path did not fail, so it is not deterministic
  for that input. The assertion now prints the ids tensor, token, slot and row; capture that the next time it happens.
  Stress-test loads of the recommended configuration before relying on it unattended.
- Prompt reading is slow (~50-60 tok/s): a 64K prompt takes 12-20 minutes before the first token.
