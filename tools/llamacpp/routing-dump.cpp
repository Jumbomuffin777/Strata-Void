// routing-dump: records which routed experts every MoE layer selects while the model reads a text file.
// Output: Strata `--dump-routing` trace format (int32 layer, int32 k, k int32 expert ids, k float32 weights) per token,
// read by tools/moe_placement/stats.py. Only expert ids are recorded, never text. Same CLI as llama.cpp tools
// (model, --rpc, --device, --tensor-split, -ot, ...), plus: -f TEXT  --chunk N  --out TRACE  --max-tokens N.
//
// Build against a llama.cpp build tree (shared libs):
//   c++ -O2 -std=c++17 -I$LLAMA/include -I$LLAMA/common -I$LLAMA/ggml/include -I$LLAMA/vendor routing-dump.cpp \
//       -o routing-dump -L$LLAMA/build/bin -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin
// (the compiler and runtime environment of that build, e.g. icpx + oneAPI setvars for a SYCL build)
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct dump_state { FILE * out = nullptr; long long records = 0; std::vector<int32_t> buf; };

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    const char * pfx = "ffn_moe_topk-";
    if (strncmp(t->name, pfx, strlen(pfx)) != 0) return ask ? false : true;
    if (ask) return true;
    auto * st = (dump_state *) ud;
    const int layer = atoi(t->name + strlen(pfx));
    const int64_t k = t->ne[0], n = t->ne[1];
    if (t->type != GGML_TYPE_I32 || t->nb[0] != sizeof(int32_t) || ggml_nelements(t) != k * n) return true;
    // ffn_moe_topk is usually a VIEW of the per-token argsort ([n_expert, n_tokens]): row j starts at j * nb[1], not
    // j * k. Read each token's k ids at the tensor's own stride (ggml_backend_tensor_get handles device buffers).
    st->buf.resize(k * n);
    if (t->nb[1] == (size_t) k * sizeof(int32_t)) {
        ggml_backend_tensor_get(t, st->buf.data(), 0, k * n * sizeof(int32_t));
    } else {
        for (int64_t j = 0; j < n; ++j) ggml_backend_tensor_get(t, st->buf.data() + j * k, j * t->nb[1], k * sizeof(int32_t));
    }
    std::vector<float> w(k, 0.0f);
    for (int64_t j = 0; j < n; ++j) {
        const int32_t hdr[2] = {layer, (int32_t) k};
        fwrite(hdr, 4, 2, st->out);
        fwrite(st->buf.data() + j * k, 4, k, st->out);
        fwrite(w.data(), 4, k, st->out);
        st->records++;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::string out_path = "routing.bin"; int chunk = 1024; long long max_tokens = 1LL << 40;
    std::vector<char *> rest = {argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) max_tokens = atoll(argv[++i]);
        else rest.push_back(argv[i]);
    }
    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    dump_state st; st.out = fopen(out_path.c_str(), "wb");
    if (!st.out) { LOG_ERR("cannot open %s\n", out_path.c_str()); return 1; }
    params.cb_eval = cb; params.cb_eval_user_data = &st; params.warmup = false;
    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    if (!ctx) { LOG_ERR("failed to init\n"); return 1; }
    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, true, false);
    if ((long long) toks.size() > max_tokens) toks.resize(max_tokens);
    LOG_INF("routing-dump: %zu tokens in chunks of %d\n", toks.size(), chunk);
    const int64_t t0 = ggml_time_us();
    for (size_t i = 0; i < toks.size(); i += chunk) {
        const int n = (int) std::min<size_t>(chunk, toks.size() - i);
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, n))) { LOG_ERR("decode failed at %zu\n", i); break; }
        fflush(st.out);
        LOG_INF("routing-dump: %zu/%zu tokens, %lld records, %.1f tok/s\n", i + n, toks.size(), st.records,
                (i + n) / ((ggml_time_us() - t0) / 1e6));
    }
    fclose(st.out);
    LOG_INF("routing-dump: wrote %lld records to %s\n", st.records, out_path.c_str());
    llama_backend_free();
    return 0;
}
