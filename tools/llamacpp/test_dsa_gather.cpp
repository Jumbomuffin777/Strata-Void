// test_dsa_gather: is decode attention over the gathered selected cells (LLAMA_DSA_GATHER) as accurate as the dense
// masked flash attention it replaces? Real GLM-5.3 DSA shapes: one query, 64 heads, one 512-wide latent KV head (K = V),
// q8_0 cache, n_kv cells of which ~2051 are attended (2048 selected + a tail), the rest -inf.
//
// Three results, compared per output value against (1):
//   (1) reference: CPU, f32 K/V (the q8_0 cache dequantized), plain softmax(QK^T * scale + mask) V
//   (2) dense: GPU flash_attn_ext over all n_kv cells, q8_0 K/V, f16 mask        (today's path)
//   (3) gather: GPU top_k over the mask -> get_rows of K and the mask -> f16 -> flash_attn_ext over n_keep cells
// Pass: (3)'s error is no larger than (2)'s (plus a small slack), at every size.
//
// Build against a SYCL llama.cpp tree (oneAPI environment loaded):
//   icpx -std=c++17 -O2 -I$L/ggml/include tools/llamacpp/test_dsa_gather.cpp -L$L/build/bin -lggml -lggml-base
//        -lggml-cpu -lggml-sycl -Wl,-rpath,$L/build/bin -o test_dsa_gather && ./test_dsa_gather
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static const int D = 512, H = 64;

struct result { std::vector<float> out; };

// runs one graph on `backend`: inputs q [D,1,H], k (q8_0) [D,n_kv], mask f16 [n_kv,1]; mode 0 dense, 1 gather
static std::vector<int32_t> g_keep;                     // the cells the gather path kept (mode 1)

static std::vector<float> run(ggml_backend_t backend, int mode, int n_kv, int n_keep, const std::vector<float> & qv,
                              const std::vector<uint8_t> & kq8, const std::vector<ggml_fp16_t> & mask, float scale) {
    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, 1, H, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, D, n_kv, 1, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, 1, 1, 1);
    ggml_tensor * kk = k, * mm = m, * keep_out = nullptr;
    if (mode == 1) {
        ggml_tensor * m32  = ggml_cast(ctx, ggml_reshape_1d(ctx, m, n_kv), GGML_TYPE_F32);
        ggml_tensor * keep = ggml_top_k(ctx, ggml_reshape_2d(ctx, m32, n_kv, 1), n_keep);
        keep = ggml_reshape_1d(ctx, ggml_cont(ctx, keep), n_keep);
        keep_out = keep;
        ggml_set_output(keep_out);
        ggml_tensor * kg = ggml_cast(ctx, ggml_get_rows(ctx, ggml_reshape_2d(ctx, k, D, n_kv), keep), GGML_TYPE_F16);
        kk = ggml_reshape_4d(ctx, kg, D, n_keep, 1, 1);
        ggml_tensor * mg = ggml_get_rows(ctx, ggml_reshape_2d(ctx, m32, 1, n_kv), keep);
        mm = ggml_cast(ctx, ggml_reshape_4d(ctx, mg, n_keep, 1, 1, 1), GGML_TYPE_F16);
    }
    ggml_tensor * vv = ggml_view_4d(ctx, kk, D, kk->ne[1], 1, 1, kk->nb[1], kk->nb[2], kk->nb[3], 0);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, kk, vv, mm, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (keep_out) ggml_build_forward_expand(gf, keep_out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, kq8.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(m, mask.data(), 0, ggml_nbytes(m));
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> r(ggml_nelements(out));
    ggml_backend_tensor_get(out, r.data(), 0, ggml_nbytes(out));
    if (keep_out) { g_keep.resize(n_keep); ggml_backend_tensor_get(keep_out, g_keep.data(), 0, ggml_nbytes(keep_out)); }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

int main() {
    ggml_backend_t gpu = ggml_backend_sycl_init(0);
    if (!gpu) { printf("no SYCL device\n"); return 1; }
    int failures = 0;
    for (int n_kv : {8192, 32768, 65536}) {
        std::mt19937 g(n_kv);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        const int n_sel = 2051, n_keep = ((2048 + 64 + 255) / 256) * 256;
        const float scale = 1.0f / std::sqrt(256.0f);
        std::vector<float> qv((size_t) D * H), kf((size_t) D * n_kv);
        for (auto & x : qv) x = nd(g);
        for (auto & x : kf) x = nd(g) * 0.5f;
        std::vector<uint8_t> kq8(ggml_row_size(GGML_TYPE_Q8_0, D) * n_kv);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, kf.data(), kq8.data(), 0, n_kv, D, nullptr);
        std::vector<float> kd((size_t) D * n_kv);                       // what both GPU paths read: dequantized q8_0
        ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(kq8.data(), kd.data(), (int64_t) D * n_kv);
        std::vector<int> cells(n_kv);
        for (int i = 0; i < n_kv; ++i) cells[i] = i;
        std::shuffle(cells.begin(), cells.end(), g);
        std::vector<ggml_fp16_t> mask(n_kv, ggml_fp32_to_fp16(-INFINITY));
        for (int i = 0; i < n_sel; ++i) mask[cells[i]] = ggml_fp32_to_fp16(0.0f);
        // (1) reference, double precision over the attended cells
        std::vector<float> ref((size_t) D * H);
        for (int h = 0; h < H; ++h) {
            std::vector<double> s(n_sel);
            double mx = -1e300;
            for (int i = 0; i < n_sel; ++i) {
                double d = 0;
                for (int c = 0; c < D; ++c) d += (double) qv[(size_t) h * D + c] * kd[(size_t) cells[i] * D + c];
                s[i] = d * scale; mx = std::max(mx, s[i]);
            }
            double z = 0;
            for (auto & x : s) { x = std::exp(x - mx); z += x; }
            for (int c = 0; c < D; ++c) {
                double acc = 0;
                for (int i = 0; i < n_sel; ++i) acc += s[i] * kd[(size_t) cells[i] * D + c];
                ref[(size_t) h * D + c] = (float) (acc / z);
            }
        }
        auto err = [&](const std::vector<float> & r) {
            double e = 0, n = 0;
            for (size_t i = 0; i < ref.size(); ++i) { e = std::max(e, (double) std::fabs(r[i] - ref[i])); n = std::max(n, (double) std::fabs(ref[i])); }
            return e / n;
        };
        const auto dense  = run(gpu, 0, n_kv, n_keep, qv, kq8, mask, scale);
        const auto gather = run(gpu, 1, n_kv, n_keep, qv, kq8, mask, scale);
        const double ed = err(dense), eg = err(gather);
        // exact set check: every attended cell kept exactly once, nothing else attended
        std::vector<int> seen(n_kv, 0);
        bool set_ok = (int) g_keep.size() == n_keep;
        for (int32_t c : g_keep) set_ok = set_ok && c >= 0 && c < n_kv && seen[c]++ == 0;
        for (int i = 0; i < n_sel; ++i) set_ok = set_ok && seen[cells[i]] == 1;
        const bool ok = set_ok && eg <= ed * 1.5 + 1e-3;
        printf("n_kv %6d: max relative error vs f64 reference: dense %.2e, gather %.2e; kept set %s  %s\n", n_kv, ed, eg,
               set_ok ? "exact" : "WRONG", ok ? "ok" : "FAIL");
        failures += ok ? 0 : 1;
    }
    ggml_backend_free(gpu);
    printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
