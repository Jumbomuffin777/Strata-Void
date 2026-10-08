# Acknowledgements

## Upstream

Strata Void is a fork of **[Strata](https://github.com/Niko1221/Strata)** by **Niko1221 and the Strata contributors**
(MIT). The engine, the expert cache and MTP drafting, the OpenAI/Anthropic-compatible server, conversation parking,
setup, the model tools and most of the documentation are theirs; this repository keeps upstream's full Git history
up to the commit it branched from (`a5682fe`) and upstream's license notice. Upstream's README is kept as
[README.strata.md](README.strata.md).

Upstream work this fork builds on directly:

- **The SYCL / Intel Arc port** (`sycl/`), started by **maxfridbe** (#423) and continued upstream.
- **Batch slots and the pipelined layer split** (#465) by **benoit lange** and **Niko1221**, which Task-Parallel
  Requests run on; ported to the SYCL engine here.
- **#870** (Arc Pro B60 PCI id; Niko1221 with Magh97) and **#871** (the all-resident graph's plan-error flag;
  Niko1221): the first is included as upstream's own commit, the second ported to the SYCL engine.

Every upstream contributor is in the Git history (`git shortlog -sn a5682fe`).

## Models

The benchmarks use **Swift 1.5** by **UkisAI**
([ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF),
Swift Open License 1.0), a fine-tune of **Qwen3.8-Flash-Next** by the **Qwen team**, whose base checkpoint supplies
the MTP draft head. No weights are redistributed here.

## Tools

Intel oneAPI DPC++ / oneMKL and the Level Zero runtime; llama.cpp's `gguf-py` (used by upstream's model tools);
Mermaid for the diagrams.

## Strata Void

- **Jumbomuffin777**: the Strata Void direction, the Task-Parallel Requests concept (use the engine's otherwise idle
  batch slots for different parts of one user's request), the test system (three Arc Pro GPUs) and the evaluation.
- Implementation, debugging and benchmarking were done with AI assistance (Anthropic's Claude) under his direction;
  commits made that way carry a `Co-Authored-By` trailer.
- Splitting one task into concurrent sub-requests is a known idea (parallel decoding, map-reduce over documents,
  multi-agent orchestration). Task-Parallel Requests are an independently designed serving extension for Strata;
  no claim is made to have invented parallel LLM reasoning.
