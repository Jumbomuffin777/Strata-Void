// test_mmid_skip: ggml's CPU mul_mat_id with expert id -1 (Strata Void's hot/cold split: "no expert in this part")
// must give exactly 0 for that row and leave every other row exactly as before. Compares, on the CPU backend:
//   (a) ids with some entries -1                 against
//   (b) the same ids with -1 replaced by an all-zero expert (the layout before the change)
// for F32, F16 and quantized experts, single token and batch: every value equal, skipped rows exactly 0. Build against a patched llama.cpp tree:
//   c++ -std=c++17 -O2 -I$L/ggml/include tools/llamacpp/test_mmid_skip.cpp -L$L/build/bin -lggml -lggml-base -lggml-cpu
//       -Wl,-rpath,$L/build/bin -o test_mmid_skip && ./test_mmid_skip
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int failures = 0;

static bool run_case(ggml_type wtype, int n_embd, int n_ff, int n_expert, int n_used, int n_tokens, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    ggml_init_params ip = { (size_t) 256 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    // experts 0..n_expert-1, and expert n_expert = all zero (for layout (b))
    ggml_tensor * w = ggml_new_tensor_3d(ctx, wtype, n_embd, n_ff, n_expert + 1);
    std::vector<float> wf((size_t) n_embd * n_ff * (n_expert + 1));
    for (size_t i = 0; i < wf.size(); ++i) wf[i] = i < (size_t) n_embd * n_ff * n_expert ? u(g) : 0.0f;
    if (wtype == GGML_TYPE_F32) {
        memcpy(w->data, wf.data(), wf.size() * sizeof(float));
    } else if (wtype == GGML_TYPE_F16) {
        for (size_t i = 0; i < wf.size(); ++i) ((ggml_fp16_t *) w->data)[i] = ggml_fp32_to_fp16(wf[i]);
    } else {
        std::vector<float> imatrix(n_embd, 1.0f);                // i-quants need one: uniform importance
        ggml_quantize_chunk(wtype, wf.data(), w->data, 0, (int64_t) n_ff * (n_expert + 1), n_embd,
                            ggml_quantize_requires_imatrix(wtype) ? imatrix.data() : nullptr);
    }
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_used, n_tokens);
    for (int64_t i = 0; i < ggml_nelements(x); ++i) ((float *) x->data)[i] = u(g);
    ggml_tensor * ids_a = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * ids_b = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    std::vector<int> skipped;
    for (int t = 0; t < n_tokens; ++t) {
        for (int j = 0; j < n_used; ++j) {
            const bool skip = (g() % 3) == 0;                    // about a third of the slots: no expert here
            const int32_t e = (int32_t) (g() % n_expert);
            ((int32_t *) ids_a->data)[t * n_used + j] = skip ? -1 : e;
            ((int32_t *) ids_b->data)[t * n_used + j] = skip ? n_expert : e;
            if (skip) skipped.push_back(t * n_used + j);
        }
    }
    ggml_tensor * ya = ggml_mul_mat_id(ctx, w, x, ids_a);
    ggml_tensor * yb = ggml_mul_mat_id(ctx, w, x, ids_b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ya);
    ggml_build_forward_expand(gf, yb);
    memset(ya->data, 0x7f, ggml_nbytes(ya));                     // garbage: a skipped row must be written
    ggml_graph_compute_with_ctx(ctx, gf, 8);
    // equal as values (the zero expert can give -0.0 where the skipped row gives +0.0), every row
    bool ok = ggml_nbytes(ya) == ggml_nbytes(yb);
    for (int64_t i = 0; ok && i < ggml_nelements(ya); ++i) ok = ((const float *) ya->data)[i] == ((const float *) yb->data)[i];
    for (int r : skipped) {
        const float * row = (const float *) ya->data + (size_t) r * n_ff;
        for (int i = 0; i < n_ff; ++i) ok = ok && row[i] == 0.0f;
    }
    printf("%-8s embd %4d ff %4d experts %3d top-%d tokens %3d: %s (%zu rows skipped)\n", ggml_type_name(wtype),
           n_embd, n_ff, n_expert, n_used, n_tokens, ok ? "ok" : "MISMATCH", skipped.size());
    ggml_free(ctx);
    if (!ok) ++failures;
    return ok;
}

int main() {
    const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_IQ2_XXS,
                               GGML_TYPE_IQ3_XXS};
    unsigned seed = 1;
    for (ggml_type t : types) {
        if (ggml_get_type_traits(t)->blck_size > 1) ggml_quantize_init(t);
        run_case(t, 256, 64, 16, 4, 1, seed++);                 // decode: one token
        run_case(t, 256, 64, 16, 4, 37, seed++);                // prefill-like batch
        run_case(t, 512, 96, 32, 8, 64, seed++);
    }
    printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
