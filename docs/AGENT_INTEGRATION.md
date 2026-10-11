# Intel Arc inference with external coding agents

## Scope of the validation

On 2026-10-11, Strata Void was tested and functionally validated on **Intel Arc Pro B70 32 GB + B65 32 GB + B60 24 GB**, using the SYCL engine. Swift 1.5 Qwen3.8 Flash-Next **IQ4_XS** completed inference; **Qwen3.8-Flash-Next-GSQ-RCO IQ3_S** completed inference and the OMP coding/tool workflows below. This is one reference system, not official support from Intel, OMP, REA or upstream Strata, and not a universal production-stability claim.

| Capability | What provides it | What was checked |
|---|---|---|
| Model inference and KV/context capacity | Strata's native engine | IQ3_S 32K/fp16 and 128K/int8 profiles loaded and answered short requests; Swift IQ4_XS 32K inference passed. |
| Function-call transport | Strata's Python OpenAI-compatible API | Declared tools, returned `tool_calls` and streamed argument deltas; native HTTP protocol tests and real OMP requests. |
| Read/edit/test coding loop | OMP 18.3.4 | The local model read a small code/test fixture, corrected an arithmetic operation, and the actual test ran after one-call approval: 1/1 passed. |
| MCP discovery, invocation and continuation | OMP plus REA 6.4.0 | The model discovered a real REA tool, requested an approved static analysis, received its result and accurately reported its Evidence ID and counts. |
| On-demand loading/offloading between profiles | An external lifecycle controller in the tested deployment | Both IQ3_S profile loads and release of the prior engine were verified. No such controller or private wrapper is bundled here. |

Strata transports function calls; an external agent executes its tools and enforces approvals. OMP's shell commands and REA's analysis tools are **not native SYCL/C++ engine features**. These results do not establish every agent/provider/API combination. Responses namespace aliases and Anthropic image/tool-result handling have targeted protocol tests; the real OMP workflow used Chat Completions.

The latest source update contains selected, attributed upstream server fixes: tool choice, robust history parsing, quoted-call handling, reasoning-call recovery, namespace aliases, tool-result preservation and quiet-client cancellation. A scoped publication review additionally corrected long-fence and multi-backtick tracking across reasoning boundaries, with genuine forced calls preserved. There is no new SYCL kernel optimization in this update. The tool-calling API does not implement grammar-constrained arguments: an unknown or impossible `tool_choice` falls back to automatic selection. Client-side execution policy remains necessary.

## 32K and 128K profiles

The new generic IQ3_S examples are [32K](../examples/strata-void-iq3s-32k.json) and [128K](../examples/strata-void-iq3s-128k.json). They use the existing engine flags and the tested Arc stage order **B70, B65, B60**:

| Context | KV | Layer split | Batch / groups | Prompt chunk | Speculation / minimum p |
|---:|---|---|---|---:|---|
| 32768 | fp16 | 18,35 | 4 / 2 | 2048 | 4 / 0.5 |
| 131072 | int8 | 18,36 | 4 / 2 | 2048 | 4 / 0.5 |

Both use all-VRAM expert residency, a 512 MiB reserve, and no conversation-cache allocation. They reference relative model/pack paths; provide your own licensed GGUF shards, compatible Strata pack, tokenizer and MTP files. The IQ3_S variant's PLE tensors are in its second shard; `--native` names shard one while `--ple-gguf` names shard two. Do not substitute a different model's pack/tokenizer or assume file names guarantee compatibility. Model acquisition and packing are described in [README](../README.md#model-files); the same tools accept this IQ3_S input. No weights are included.

`ONEAPI_DEVICE_SELECTOR` is an example for a system whose Level Zero ordinals already follow the tested card order. Check `sycl-ls` and adjust it to your own machine. Do not bypass the native device or memory validation. The test loaded 128K and ran short inference; **reliability with a filled 131072-token prompt, sustained concurrency and long agent sessions remains untested for this IQ3_S profile**. Earlier Swift long-document experiments in [BENCHMARKS](BENCHMARKS.md) are a separate workload/model and do not qualify this one.

From the repository root, with the existing SYCL build and prepared model files:

```sh
source /opt/intel/oneapi/setvars.sh
.venv/bin/python -m serve.server --engine strata --config examples/strata-void-iq3s-32k.json --port 8095
```

The listener stays loopback by default. `/health` returns `loaded: true` when ready. Native `/load` and `/unload` endpoints exist, but switching a whole run profile and coordinating other GPU workloads require your own controller or a deliberate server restart. Neither the example model ID nor OMP's context metadata selects a larger native context automatically.

## Connect OMP 18.3.4

Install OMP separately using its [official project instructions](https://github.com/can1357/oh-my-pi). This repository does not install it. Use a new isolated agent directory so existing provider and tool settings stay intact:

```sh
export PI_CODING_AGENT_DIR="$PWD/.omp-strata-example"
mkdir -p "$PI_CODING_AGENT_DIR"
cp examples/agents/omp-models.yml "$PI_CODING_AGENT_DIR/models.yml"
cp examples/agents/omp-config.yml "$PI_CODING_AGENT_DIR/config.yml"
omp --cwd /path/to/your/authorized-project
```

The generic provider uses `http://localhost:8095/v1`, the native model name `flash-next-iq3s`, `api: openai-completions`, tool support, Qwen3 tokenization and a 32768-token budget. All six coding roles select that same local model. No cloud key or fallback is configured in these examples. An isolated agent directory does not by itself disable OMP's discovery of other clients or project configuration; review the effective model/MCP list before use.

For a keyed endpoint, replace `auth: none` with `auth: apiKey`, add `apiKey: STRATA_API_KEY`, and supply that variable securely to both client and server. Do not put a real key into a committed YAML file. Keep the listener local, or follow the repository's [security guidance](../SECURITY.md) before network exposure.

For 128K, first start Strata with `examples/strata-void-iq3s-128k.json`. Then change the example model's `contextWindow` to `131072` and its display name to identify 128K. Changing YAML alone cannot enlarge the loaded engine. Keep the 32K configuration for ordinary coding.

The example retains `tools.approvalMode: write` and explicitly prompts for bash, eval and task execution. Workspace writes can run in this mode; shell/test execution remains interactive. These approval settings are not OS filesystem/process isolation. Start in an authorized project and approve only the actual command you intend to execute.

To reproduce the coding workflow, use a disposable toy project with a deliberately incorrect arithmetic function and one unit test. Ask the model to inspect it, make the minimal correction and request permission to run the test. Verify the tool event, actual test exit code and resulting file, rather than trusting the model's assertion of success.

## Optional REA through MCP

REA is a separate project, not an engine dependency. Version **6.4.0** was tested with existing Node 24.21.0. Its supported Node range is 22.19+ within 22.x, 24.11+ within 24.x, or 26+. Check your installed Node/npm first; see [REA's release/setup boundary](https://github.com/morluto/rea/blob/main/docs/installation.md).

Generic local installation, with no system-wide Node change:

```sh
npm install --prefix /path/to/rea-runtime --save-exact --ignore-scripts --no-audit --no-fund rea-agents@6.4.0
node /path/to/rea-runtime/node_modules/rea-agents/scripts/rea.mjs --version
```

Merge the optional [REA MCP example](../examples/agents/rea-mcp.json) into your own OMP `mcp.json`; replace the generic path. It is **disabled by default**. It demonstrates upstream stdio registration, not the restricted adapter used for the recorded test. The direct upstream server exposes broader capabilities: configure your own target allowlist, OS containment and tool subset before enabling it. Neither this JSON nor OMP's approval mode creates a sandbox. Do not enable public HTTP exposure, arbitrary host/process access or target execution merely to repeat the static test.

Install the matching package skill and its supporting references through REA's reviewed `setup --client omp` plan, or copy the package's `skills/reverse-engineer-anything` directory into the isolated agent home's supported skills location. Review the plan first; do not approve optional native-toolchain installation unless needed. Follow the actual version's advertised schemas.

OMP exposes MCP tools as `xd://` devices. Read `xd://mcp__rea_analyze_javascript_application` to discover its schema, then invoke it by writing JSON arguments to that device. Select only an authorized harmless artifact and use **`detail: summary`**, then small retained Evidence views. A complete graph exceeded OMP's 20 kB tool-text budget in a trial and the model invented a summary; that trial was rejected. The accepted summary-mode run verified the exact returned Evidence ID and counts: **3 relevant files, 2 parsed JavaScript files, 0 parse failures**. The outer device-write and inner MCP approval gates were both preserved for that one call.

The recorded REA process loaded on demand, had no network, read one staged artifact through a read-only mount, and stopped after the OMP session. That containment belongs to the external deployment and is not provided by this repository's generic disabled registration. JavaScript static analysis passed without Ghidra. Managed .NET, PE-resource and EVM tools were registered but not separately exercised; native decompilation was not tested because Ghidra was absent. No universal REA safety or capability claim follows from this small smoke test.

## Decode observation: 50.6 tok/s

The measured **50.6 tok/s** was a live sample from **IQ3_S, 32K, SYCL, one active stream**, using the exact request in [iq3s-decode-probe.json](../examples/agents/iq3s-decode-probe.json): a 40-item code-review checklist, temperature 0, thinking disabled, no tools, and `max_tokens: 768`. It came from **`GET /metrics` → `live.tok_s`**, Strata's Python-server observation of accepted output-token deltas over its **2-second sliding window**. It is not a C++ GPU kernel clock, a whole-response client rate, an agent workflow rate or a new optimization speedup.

A repeated request returned 750 completion tokens from 41 prompt tokens (34 cached); client request-to-finish time was 14.85 seconds, including delivery and request overhead. A separate previous completed decode segment of the same workload reported **51.1 tok/s** through native engine `decode_ms`/generated-token telemetry. It was not the native clock paired with the captured live window. These observations supplement, rather than replace, the earlier IQ4_XS benchmark tables.

Reproduce the workload from the repository root on an already ready model:

```sh
curl -N http://localhost:8095/v1/chat/completions -H 'Content-Type: application/json'   --data-binary @examples/agents/iq3s-decode-probe.json
# In another terminal while the response is generating:
curl -s http://localhost:8095/metrics
```

For an authenticated server add its authorization header securely. Record the model/configuration, warm/cold and prefix-cache state, active request count and the rate's time/window. `live.tok_s` aggregates concurrent requests: do not label it single-stream throughput when several requests are running. Draft or rejected speculative tokens are not output tokens. Keep prompt throughput and engine decode time separate, and do not interpret an idle/missing live value as an active measurement.

The [public summary](../bench/results/2026-10-11-agent-compatibility/summary.json) contains only aggregate technical facts. It is not a raw trace, private deployment receipt or statistical benchmark dataset. No private workspace, hostname, credential, deployment script or model weight is included.

## Targeted checks

The changed Python server and parser were tested without GPUs against fake engines:

```sh
.venv/bin/python -m unittest serve.test_server serve.test_reasoning_tools serve.test_reasoning_rescue   serve.test_quiet_cancel serve.test_responses serve.test_task_parallel serve.test_task_parallel_v2 serve.test_parallel
.venv/bin/python tools/test_setup_sycl.py
```

The final scoped run completed **316 tests: 315 passed and 1 was explicitly skipped** because optional `jsonschema` was absent. All **11 SYCL setup checks** passed. Added inert regressions cover long Markdown fences, multi-backtick spans, chunked quoted-call suppression, legitimate forced calls and image-source equivalence. Anthropic tool-result images use the same URL/local-image authority already available to admitted vision API clients; this update adds no target allowlist or vision sandbox. The comparisons do not fetch URLs or read image targets. The skip concerns detailed structured-output validation; ordinary JSON and protocol checks still ran. No dependency was installed solely to remove it. C++/SYCL source was unchanged, so no new native binary build or GPU benchmark was performed. Existing live inference/OMP evidence is reused rather than loading models again for publication.
