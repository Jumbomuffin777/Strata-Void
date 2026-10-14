# GLM-5.3-Flash and DeepSeek V4 Flash on Intel Arc only

Measurements and changes for running the two large MoE models on one host with three Intel Arc Pro GPUs and host RAM,
no remote GPUs: the hot/cold llama.cpp backend ([HOTCOLD_LLAMACPP.md](HOTCOLD_LLAMACPP.md)) with the additions of
`patches/llama.cpp-moe-dynamic-residency.patch`. Every number here was measured on one machine; the method is given so
it can be repeated on another.

## The machine, as measured

| | |
|---|---|
| GPUs | Arc Pro B70 32 GB, B65 32 GB (PCIe 5.0 x8 each, CPU lanes), B60 24 GB (PCIe 4.0 x4 behind the chipset) |
| host -> GPU (pinned) | B70 / B65 28.7 GB/s, B60 7.0 GB/s; one 9 MiB expert: 0.35 ms (B65), 1.4 ms (B60), 2.3 ms (B70: small-copy latency) |
| GPU clocks | B70 up to 2800 MHz, B65 / B60 2400 MHz; all three in the driver's `power_saving` profile |
| CPU / RAM | 16 cores, 64 GB DDR5 (60 GB usable, ~33 GB free beside the other services) |

For one decoded token the CPU reads a cold expert from DDR5 faster than it could be sent to a GPU, so cold experts are
computed where they are (the CPU); uploading pays only for prefill batches.

## Models and placement

| | Maya-S (Project Maya's GLM-5.3-Flash, `Maya-S-v2-IQ2_XXS`) | DeepSeek V4 Flash `UD-IQ3_XXS` |
|---|---|---|
| file | 96.5 GB, 3 shards | 103 GB, 4 shards |
| experts | IQ2_XXS gate/up, IQ2_S / IQ3_XXS down, 288 x top-8, 42 MoE layers | IQ2_S / IQ3_XXS, 256 x top-6, 43 layers |
| plan (64K, q8 KV, budgets 28.5/28.5/20 GiB) | 13.0 GiB cold, 2.3 % of measured routing | 16.4 GiB cold, 2.6 % (budgets 29.3/29.3/21, ubatch 1024) |
| layers per GPU | 17 / 16 / 13 (`--tensor-split 17,16,14`) | 16 / 16 / 11 |

Routing statistics are the earlier measurements of the same weights at other quantizations (routing transfers to
first order). Maya's files name their architecture `glm5-next`; the patch reads that spelling and its keys.

## Method

* **Speed**: `tools/llamacpp/bench_hotcold.py --decode 256 --repeat 2`: prompts of 8K / 32K / 64K tokens (repository
  docs with a fact placed at the start and asked about at the end), greedy, exactly 256 generated tokens
  (`ignore_eos`), the second run reusing the cached prompt so decode is measured twice at the same depth. Prefill and
  decode are llama-server's own timings (decode = generated tokens / generation time, prompt excluded). Context 64K,
  q8_0 KV, one slot, `--ubatch-size 1024`.
* **Where time goes**: `tools/llamacpp/op-profile --decode N --sync REGEX`: decodes N single tokens twice, without any
  callback (the real speed) and synchronizing only at a few named nodes per layer, each segment's time attributed to the
  node that ends it. The profiled run is within 3 % of the real one.
* **Correctness**: `tools/llamacpp/compare_logprobs.py`: 128 greedy tokens with top-5 log-probabilities, two runs
  compared. The dense path is not bit-reproducible between runs (8K: chosen-token log-probabilities within ~0.005,
  first different token at 83); a change must stay inside that.

## Maya-S, Intel only, static placement

| prompt | prefill | TTFT | decode (run 1 / run 2) | needle |
|---|---|---|---|---|
| short | - | 2.5 s | 12.51 / 12.44 tok/s | - |
| 8K | 54-60 tok/s | 128-145 s | 11.27 / 11.27 tok/s | found |
| 32K | 50-58 tok/s | 511-560 s | 9.06 / 9.10 tok/s | found |
| 64K | 43-53 tok/s | 780-1234 s | 6.52 / 6.51 tok/s | found |

Memory: GPU weights 28.3 / 28.3 / 20.1 GiB, KV 0.33 / 0.33 / 0.25 GiB (64K, q8), compute buffers ~1.2 GiB each
(ubatch 1024); host model buffer 11.0 GiB, peak of the whole test process 28.9 GB.

The earlier ~9.7 tok/s was a different file (unsloth UD-IQ3_XXS, 3-bit experts) on Arc + CUDA RPC GPUs, measured with
short prompts at a 64K/128K context; it is not a like-for-like baseline. With static placement Maya-S on Arc alone is
faster than that at short and 8K prompts and slower at 32K and 64K; with gathered sparse attention (below) it decodes
above 11 tok/s at every size measured.

### Static versus adaptive expert residency (Maya-S, gathered attention on, same workload)

`LLAMA_MOE_DYNAMIC=observe` (static placement, hit rate logged) against `=1` (adaptive: up to 24 swaps every 32
tokens), same prompts, same settings:

| | static (observe) | adaptive |
|---|---|---|
| hot hit rate, decoding after the 8K / 32K prompt | 89-90 % | 92.6 -> 95.9 % (8K), 89.6 -> 95.5 % (32K) |
| swaps | 0 | 1,920 over the run, 3.6 ms each (6.9 s in all) |
| decode short / 8K / 32K (run 1 / run 2), tok/s | 12.14 / 12.14, 10.08 / 11.43, 11.20 / 11.18 | 11.98 / 12.18, 10.01 / 11.53, 10.94 / 11.21 |

* The offline plan predicted 97.7 % hot from its calibration text; on these prompts static placement hits 89-90 %.
  Adaptive residency recovers most of the difference within a few hundred tokens of a new prompt.
* It does not make decode faster here: a miss costs little (the expert is computed by the CPU, which runs the cold part
  of every split layer anyway; that segment is mostly fixed round-trip cost), while observing the routing costs ~3 %
  (12.54 -> 12.14 tok/s short) and swaps cost while they happen. It stays off by default; it would matter where a
  miss is expensive (a slower CPU path, an NVMe tier, or a runtime that skips the cold part when no cold expert is
  selected).

## DeepSeek V4 Flash UD-IQ3_XXS, Intel only, static placement

Same method; load-time split of the original file (16.4 GiB cold), layers 16 / 16 / 11, ubatch 1024, 64K q8 KV; the
test process held 17.4 GB of host memory.

| prompt | prefill | TTFT | decode (run 1 / run 2) | needle |
|---|---|---|---|---|
| short | - | 1.3 s | 10.28 / 10.25 tok/s | - |
| 8K | 62.4 tok/s | 127 s | 9.10 / 9.46 tok/s | found |
| 32K | 46.9 tok/s | 546 s | 7.52 / 7.76 tok/s | found |
| 64K | 32.1 tok/s | 1053 s | 6.34 / 6.41 tok/s | found |

The earlier ~7.0 tok/s was UD-IQ3_S on Arc + CUDA RPC GPUs. DeepSeek's sparse attention (compressed cache plus a
window) is a different graph from GLM's DSA; the gathered-attention change does not apply to it yet, and its decode
also slows with depth.

### Where a decoded token goes (profile, ms per token)

| segment | depth 512 | depth 32K |
|---|---|---|
| hot experts on the GPUs (router ids -> gate/up/down) | 28.4 | 27.8 |
| KDA linear-attention layers (34), with norms and projections | 25.7 | 26.6 |
| MLA/DSA attention (11 layers): flash attention itself | ~2 | **31.5** |
| DSA indexer (scores, top-k) and q/kv projections | ~10 | ~12 |
| cold experts on the CPU, with the round trip (39 layers) | 9.8 | 12.8 |
| router / gating | 5.3 | 5.4 |
| total | 80 | 111 |

* The **depth cost is the sparse attention**: llama.cpp implements DSA as full attention over every cached cell with a
  mask that keeps the 2,048 selected ones, so it reads and multiplies the whole cache (2.9 ms per layer at 32K).
* Hot experts cost 0.5-0.8 ms per layer per token where reading the eight experts' weights takes ~0.03 ms: per-kernel
  overhead, not bandwidth. SYCL command graphs, which would cut it, are disabled by ggml-sycl for multi-GPU runs and for
  graphs with MUL_MAT_ID.
* CPU threads do not matter for decode (16 / 8 / 4 threads: 79.5 / 79.3 / 81.2 ms).

### Prefill

72.6 % of a prefill batch is the GPUs' MUL_MAT_ID: ggml-sycl multiplies each expert separately (one dequant + GEMM
launch per expert, a few rows each). A larger ubatch amortizes it: 256 -> 1024 raised prefill from 33.5 to 59.7 tok/s
(8K); 2048 runs out of GPU memory in flash attention's scratch at long prompts.

### Prefill: grouped MUL_MAT_ID (`patches/llama.cpp-sycl-grouped-moe.patch`)

Measured in isolation first: `test-backend-ops perf -o MUL_MAT_ID` on the B70 at the models' expert shapes (4096 x
2048 experts, uniform random routing; GLM 288 x top-8, DeepSeek 256 x top-6). The per-expert path costs ~190 us per
expert plus ~2 us per routed row however few rows the expert gets: it dequantizes the whole expert to f16 (16 MiB)
before its GEMM. The grouped path quantizes the sorted rows to Q8_1 once and runs one launch over tiles of <= 8 rows of
one expert; for IQ2_XXS / IQ2_S / IQ3_XXS a lane decodes its 32 codebook weights once and reuses them for every row of
the tile with dp4a (~5.4 us per routed row). Experts with more than 48 rows keep the GEMM path.

| op, tokens | per-expert (before) | generic grouped | decode-once grouped (default) |
|---|---|---|---|
| GLM gate/up IQ2_XXS, 256 | 57.3 ms | 29.0 ms | 10.5 ms |
| GLM gate/up IQ2_XXS, 1024 | 71.1 ms | 111.0 ms | 44.0 ms |
| GLM down IQ3_XXS, 256 | 55.4 ms | 45.6 ms | 15.1 ms |
| GLM down IQ3_XXS, 1024 | 74.1 ms | 171.7 ms | 43.2 ms |
| DeepSeek gate/up IQ2_S, 1024 | 55.9 ms | 84.7 ms | 35.4 ms |
| DeepSeek down IQ3_XXS, 1024 | 66.5 ms | 129.2 ms | 33.2 ms |
| decode, 1 token (unchanged path) | 130-133 us | - | 130 us |

The generic grouped kernel (the existing vec_dot per row) loses at 1024 tokens because it decodes the codebook again
for every row; IQ2_XXS's vec_dot is scalar. `test-backend-ops test -o MUL_MAT_ID`: 913/913 against the CPU on the B70
and the B60, at the default and with the threshold forced to 1, 4 and 1000 rows (every expert grouped / mixed / none).
Switch: `GGML_SYCL_MMID_GROUPED=0` (before), `1` (generic), `2` (default); `GGML_SYCL_MMID_GROUPED_MAX_ROWS=N`.

On the real model (Maya-S, Intel only, gathered attention on, ubatch 1024, 64K q8 KV; same server settings, one
load each; `prefill-ab` = bench_hotcold.py `--decode 64` then compare_logprobs.py on the cached prompts):

| | `GGML_SYCL_MMID_GROUPED=0` | `=2` (default) |
|---|---|---|
| 8K prompt (7,885 new tokens) | 60.6 tok/s, TTFT 133 s | **86.8 tok/s, TTFT 94 s** (+43 %) |
| 32K prompt (25,602 new tokens, 8K prefix cached) | 56.2 tok/s, TTFT 456 s | **75.8 tok/s, TTFT 338 s** (+35 %) |
| short prompt TTFT | 2.46 s | 1.73 s |
| decode after 8K / 32K (64 tokens) | 7.56 / 11.64 tok/s | 7.62 / 11.45 tok/s |
| GPU memory free after load | 2.4 / 2.0 / 2.0 GiB | same |
| needle | found / found | found / found |

Greedy tokens first differ at token 59 / 43 / 23 (short / 8K / 32K), chosen-token log-probability within 0.19 / 0.47 /
0.62. That is the size of an ordinary precision change (q8 vs f16 KV: token 10, 0.77) and larger than run-to-run noise
(token 83, 0.05): prefill activations are now quantized to Q8_1 for the expert products, as the decode path (and
llama.cpp's CUDA MMQ prefill) already does.
Perplexity (llama-perplexity, same split and settings, ctx 2048, ubatch 1024, 6 chunks of a fixed 60 KB text: two
licence texts and two of this repository's docs): `=0` **1.8938 +/- 0.042**, `=2` **1.8927 +/- 0.042**, every chunk
within 0.012; 29.9 -> 21.9 s per 2048-token pass. The greedy divergence above is precision noise, not a quality loss.

### DeepSeek V4 Flash: grouped prefill and gathered CSA decode (same session, same method, 128 decoded tokens)

`GROUPED:GATHER` = `GGML_SYCL_MMID_GROUPED` : `LLAMA_DSV4_GATHER` (`patches/llama.cpp-dsv4-gather.patch`).

| prompt | 0:0 prefill / decode | 2:0 prefill / decode | 2:1 prefill / decode |
|---|---|---|---|
| 8K | 62.8 / 9.04 tok/s | **79.9** / 9.07 | 80.2 / 7.03 (gather below its threshold now) |
| 32K | 47.0 / 7.48 | **55.5** / 7.42 | 55.6 / **8.83** |
| 64K | 32.1 / 6.32 | (run aborted, see open issues) | **36.0** / **7.71** |

Needle found in every completed run; host RSS 17.7 GiB. **Correctness:** 0:0 against 2:1 (both changes), 128 greedy
tokens: identical at short, 8K, 32K and 64K; chosen-token log-probability within 0.08 / 0.16 / 0.47 / 0.22.
`test_dsa_gather 640 768 2176 8320 16512` (DeepSeek's window 128 + top-512 over the 4x cache): kept set exact, error
against an f64 reference 1.3-1.5e-3 (dense path 2.3-2.4e-3). At 8K the gather was slower (9.04 -> 7.03 tok/s: the
top-k pass over ~2,200 cells costs more than it saves); it now applies only from 4,096 cells (about 16K of context).

## Changes and their measured effect

| change | effect | kept |
|---|---|---|
| load-time hot/cold split (`LLAMA_MOE_HOTCOLD`) | no rewritten copy of the model; tensors identical to a rewrite | yes |
| CPU skips experts its cold part does not hold (id -1) | decode 10.1 -> 12.6 tok/s (short), prefill 30 -> 34 tok/s (ubatch 256); greedy text identical | yes |
| ubatch 1024 | prefill 34 -> 60 tok/s (8K) | yes (config) |
| adaptive residency (`LLAMA_MOE_DYNAMIC=1`) | hit rate 89-90 % -> ~95.5 %; decode unchanged within noise | off by default |
| grouped expert GEMM for SYCL prefill (oneMKL group batch) | prefill **slower** on the real model (58.6 vs 77.3 tok/s profiled) | no, removed |
| f16 instead of q8_0 KV | decode at 32K 111.1 -> 109.2 ms | no |
| gathered sparse attention (`LLAMA_DSA_GATHER=1`) | decode at 32K 111.3 -> 87.1 ms/token (profile); 64K 6.5 -> 11.1 tok/s (benchmark) | yes, opt-in |

### Maya-S with gathered sparse attention (`LLAMA_DSA_GATHER=1`), same method and settings

| prompt | decode (run 1 / run 2) | static | needle |
|---|---|---|---|
| short | 12.54 / 12.54 tok/s | 12.51 / 12.44 | - |
| 8K | 10.37 / 11.88 tok/s | 11.27 / 11.27 | found |
| 32K | 11.25 / 11.27 tok/s | 9.06 / 9.10 | found |
| 64K | 11.23 / 11.05 tok/s | 6.52 / 6.51 | found |

(The first 8K run is the one outlier of the series; its repeat matches the profile.) Prefill is unchanged: the
gathered path applies to single-token decode only.

**Why it is correct.** DSA attends to the cells its indexer selected; llama.cpp computes that as attention over every
cell under a mask that is exactly 0 for the selected ones and -inf for the rest. The gathered path takes the top
`n_keep` entries of that same mask (`n_keep` >= the selected cells plus the open pool's tail, rounded to 256), gathers
those cells' keys and mask values, and runs flash attention over them: the same cells and the same softmax; the extra
entries keep their -inf. `tools/llamacpp/test_dsa_gather.cpp` checks it on the GPU at the model's shapes (64 heads,
512-wide latent, q8_0 cache, 8K / 32K / 64K cells, 2,051 attended): the kept set is exact (every attended cell once,
nothing else), and the error against an f64 reference equals the dense path's (1.4-1.7e-3 vs 1.4e-3, max relative).
End to end, greedy decoding is sensitive to ANY change of attention kernel at some tokens: over 128 tokens at 8K,
dense q8 vs dense f16 KV first differ at token 10 (max chosen log-probability difference 0.77 before that), dense vs
gathered at token 10 (0.64) with q8 and at token 69 with f16; two dense runs on one server first differ at token 83
(0.05). The gathered path stays inside the variation of an ordinary KV-type change and answers the needle the same.
