# Intel-only MoE work: where it stands and what is next

A handover for whoever continues the Intel-only GLM-5.3-Flash / DeepSeek V4 Flash work. Measurements:
[INTEL_ONLY_MOE.md](INTEL_ONLY_MOE.md). Flow and runtime switches: [HOTCOLD_LLAMACPP.md](HOTCOLD_LLAMACPP.md).

## Done and measured

- Both models run on three Intel Arc Pro GPUs + host RAM only, from their original GGUFs split at load time.
- Maya-S decode, gathered sparse attention on: 12.5 (short) / ~11.9 (8K) / 11.3 (32K) / 11.1 (64K) tok/s.
- DeepSeek V4 Flash decode (static, no gather): 10.3 / 9.3 / 7.6 / 6.4 tok/s.
- Kept: load-time split, CPU skip of absent experts, ubatch 1024, gathered DSA decode (opt-in). Rejected after
  measurement: grouped oneMKL expert GEMM (prefill slower), f16 KV (no decode gain), fewer CPU threads (no gain).

## Next, in order of expected value per risk

1. **DeepSeek decode at depth.** Profile it like GLM (`op-profile --decode 32 --sync ...` at 512 and 32K tokens) to
   find its depth term; if it is again attention over every cell under a mask, the same gather applies to its graph
   (`build_attn_sparse` / the DSA input of deepseek4), validated the same way (`test_dsa_gather.cpp`,
   `compare_logprobs.py` against a dense-vs-dense and q8-vs-f16 baseline).
2. **Prefill.** 72 % of a prefill batch is the GPUs' per-expert MUL_MAT_ID (ggml-sycl launches one dequant + GEMM per
   expert). A grouped oneMKL batch was slower; the next candidates are an MMQ-style kernel that takes the routed rows
   of many experts in one launch (ggml-sycl has MMQ disabled for accuracy reasons - check why), or dequantizing an
   expert once per batch into a reusable f16 tile and multiplying all its rows (fewer, larger GEMMs).
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

- One warmup crash in ~15 loads (GGML_ASSERT row id out of range in ggml-sycl's mul_mat_id, a -1 expert id reaching
  a GPU), seen once with f16 KV + gathered attention; not reproduced on retry. Stress-test loads of the recommended
  configuration before relying on it unattended.
- Prompt reading is slow (~50-60 tok/s): a 64K prompt takes 12-20 minutes before the first token.
