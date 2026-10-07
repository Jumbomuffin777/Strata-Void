// moe_prefill_bench: where the prompt path's expert time goes on one GPU, with random IQ blocks.
//   moe_prefill_bench [ty=23] [experts=128] [rows_per_expert=40] [reps=3] [down_ty=20]
// (the down projection's rows are 640 values: an IQ4_XS model stores them in a 32-value format, IQ4_NL or Q8_0)
// The prompt path runs, per routed expert: dequant gate/up, dequant down (FP16), a oneMKL GEMM, a SwiGLU, a oneMKL
// GEMM.  This times (a) that sequence expert after expert as the engine submits it (wall, and the host's submission
// time alone), (b) the dequant kernels by themselves launched once per expert, (c) ONE dequant launch over all the
// experts' blocks (the kernel's own speed, no launch overhead), (d) the GEMMs per expert vs one strided batch.
#include "strata/kernels/iq_kernels.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* only = std::getenv("MPB_ONLY");   // run one part: seq | dq | gemm | swiglu | one | batch
    auto want = [&](const char* k) { return only == nullptr || std::string(only) == k; };
    const int ty = argc > 1 ? std::atoi(argv[1]) : 23;
    const int E = argc > 2 ? std::atoi(argv[2]) : 128;
    const int M = argc > 3 ? std::atoi(argv[3]) : 40;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 3;
    const int dty = argc > 5 ? std::atoi(argv[5]) : 20;
    const int64_t NE = 2560, FF = 640;
    const size_t rb_gu = strata::kernels::iq_row_bytes(ty, NE), rb_d = strata::kernels::iq_row_bytes(dty, FF);
    if (!rb_gu || !rb_d) { std::printf("type %d not supported\n", ty); return 2; }
    const size_t g_bytes = FF * rb_gu, d_bytes = NE * rb_d;
    sycl::queue* q = &dpct::get_in_order_queue();
    std::printf("device: %s; type %d, %d experts x %d rows; gate %zu B, down %zu B per expert\n",
                q->get_device().get_info<sycl::info::device::name>().c_str(), ty, E, M, g_bytes, d_bytes);
    std::mt19937 rng(7);
    std::vector<uint8_t> hb((g_bytes * 2 + d_bytes) * E);
    for (auto& x : hb) x = (uint8_t) rng();
    // a small finite scale in every block (the first two bytes are the fp16 d of the IQ/K superblocks)
    const size_t blk = strata::kernels::iq_row_bytes(ty, 256);
    for (size_t o = 0; o + 1 < hb.size(); o += blk) { hb[o] = 0x00; hb[o + 1] = 0x1C; }
    uint8_t* W = sycl::malloc_device<uint8_t>(hb.size(), *q);
    q->memcpy(W, hb.data(), hb.size()).wait();
    auto gate = [&](int e) { return W + (size_t) e * (g_bytes * 2 + d_bytes); };
    auto up = [&](int e) { return gate(e) + g_bytes; };
    auto down = [&](int e) { return gate(e) + 2 * g_bytes; };
    const int DQ = 2;   // the engine's ring of dequantized experts
    uint16_t* dgu = sycl::malloc_device<uint16_t>((size_t) DQ * 2 * FF * NE, *q);
    uint16_t* dd = sycl::malloc_device<uint16_t>((size_t) DQ * NE * FF, *q);
    uint16_t* X = sycl::malloc_device<uint16_t>((size_t) E * M * NE, *q);
    float* GU = sycl::malloc_device<float>((size_t) E * M * 2 * FF, *q);
    uint16_t* H = sycl::malloc_device<uint16_t>((size_t) E * M * FF, *q);
    float* D = sycl::malloc_device<float>((size_t) E * M * NE, *q);
    q->memset(X, 0, (size_t) E * M * NE * 2).wait();
    q->memset(H, 0, (size_t) E * M * FF * 2).wait();
    dpct::blas::descriptor_ptr h = new dpct::blas::descriptor();
    h->set_queue(q);
    const float alpha = 1.f, beta = 0.f;
    auto gemm = [&](const uint16_t* x, const uint16_t* w, float* y, int T, int N, int K) {
        dpct::blas::gemm(h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, N, T, K, &alpha, w,
                         dpct::library_data_t::real_half, K, x, dpct::library_data_t::real_half, K, &beta, y,
                         dpct::library_data_t::real_float, N, dpct::compute_type::f32);
    };
    auto swiglu = [&](const float* gu, uint16_t* hh, int rows) {
        q->parallel_for(sycl::range<1>((size_t) rows * FF), [=](sycl::id<1> i) {
            const size_t r = i[0] / FF, c = i[0] % FF;
            const float g = gu[r * 2 * FF + 2 * c], u = gu[r * 2 * FF + 2 * c + 1];
            hh[i[0]] = sycl::bit_cast<uint16_t>(sycl::half(g / (1.f + sycl::exp(-g)) * u));
        });
    };
    auto seq = [&]() {   // what prefill.cpp submits per routed expert
        for (int e = 0; e < E; ++e) {
            const int k = e % DQ;
            strata::kernels::iq_dequant_gu_f16(ty, gate(e), up(e), FF, NE, dgu + (size_t) k * 2 * FF * NE, q);
            strata::kernels::iq_dequant_f16(dty, down(e), NE * FF, dd + (size_t) k * NE * FF, q);
            gemm(X + (size_t) e * M * NE, dgu + (size_t) k * 2 * FF * NE, GU + (size_t) e * M * 2 * FF, M, 2 * FF, NE);
            swiglu(GU + (size_t) e * M * 2 * FF, H + (size_t) e * M * FF, M);
            gemm(H + (size_t) e * M * FF, dd + (size_t) k * NE * FF, D + (size_t) e * M * NE, M, NE, FF);
        }
    };
    auto dq_only = [&]() {
        for (int e = 0; e < E; ++e) {
            const int k = e % DQ;
            strata::kernels::iq_dequant_gu_f16(ty, gate(e), up(e), FF, NE, dgu + (size_t) k * 2 * FF * NE, q);
            strata::kernels::iq_dequant_f16(dty, down(e), NE * FF, dd + (size_t) k * NE * FF, q);
        }
    };
    auto gemm_only = [&]() {
        for (int e = 0; e < E; ++e) {
            const int k = e % DQ;
            gemm(X + (size_t) e * M * NE, dgu + (size_t) k * 2 * FF * NE, GU + (size_t) e * M * 2 * FF, M, 2 * FF, NE);
            gemm(H + (size_t) e * M * FF, dd + (size_t) k * NE * FF, D + (size_t) e * M * NE, M, NE, FF);
        }
    };
    auto swiglu_only = [&]() { for (int e = 0; e < E; ++e) swiglu(GU + (size_t) e * M * 2 * FF, H + (size_t) e * M * FF, M); };
    auto timed = [&](const char* name, auto&& fn) {
        fn(); q->wait();
        double best_wall = 1e30, best_sub = 0;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = Clock::now();
            fn();
            const auto t1 = Clock::now();
            q->wait();
            const auto t2 = Clock::now();
            if (ms(t0, t2) < best_wall) { best_wall = ms(t0, t2); best_sub = ms(t0, t1); }
        }
        std::printf("%-34s wall %8.2f ms (%7.1f us/expert), host submission %8.2f ms (%6.1f us/expert)\n", name,
                    best_wall, 1000 * best_wall / E, best_sub, 1000 * best_sub / E);
    };
    // the fast IQ4 dequant against the generic kernels: the same bits?
    {
        const size_t ngu = 2 * FF * NE, nd = NE * FF;
        std::vector<uint16_t> a(ngu + nd), b2(ngu + nd);
        for (int fast = 0; fast < 2; ++fast) {
            strata::kernels::iq_set_dq_fast(fast);
            strata::kernels::iq_dequant_gu_f16(ty, gate(1), up(1), FF, NE, dgu, q);
            strata::kernels::iq_dequant_f16(dty, down(1), NE * FF, dd, q);
            q->wait();
            auto& o = fast ? b2 : a;
            q->memcpy(o.data(), dgu, ngu * 2).wait();
            q->memcpy(o.data() + ngu, dd, nd * 2).wait();
        }
        size_t diff = 0;
        for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b2[i];
        std::printf("check: fast vs generic dequant, %zu of %zu FP16 values differ\n", diff, a.size());
    }
    strata::kernels::iq_set_dq_fast(0);
    if (want("dq")) timed("GENERIC: the two dequants", dq_only);
    if (want("seq")) timed("GENERIC: dequant+gemm+swiglu", seq);
    strata::kernels::iq_set_dq_fast(1);
    if (want("seq")) timed("per expert: dequant+gemm+swiglu", seq);
    if (want("dq")) timed("per expert: the two dequants", dq_only);
    if (want("gemm")) timed("per expert: the two GEMMs", gemm_only);
    if (want("swiglu")) timed("per expert: swiglu", swiglu_only);
    // the dequant kernel's own speed: ONE launch over 16 experts' down blocks (they are not contiguous: dequant a
    // contiguous span of the weight buffer of the same size instead - random blocks, the same work)
    const int span = std::min(E, 16);
    uint16_t* big = sycl::malloc_device<uint16_t>((size_t) span * 3 * FF * NE, *q);
    // gate+up blocks of `span` experts made contiguous (the same bytes the per-expert launches read)
    uint8_t* GUc = sycl::malloc_device<uint8_t>((size_t) span * 2 * g_bytes, *q);
    for (int e = 0; e < span; ++e) q->memcpy(GUc + (size_t) e * 2 * g_bytes, gate(e), 2 * g_bytes);
    q->wait();
    const int64_t n_all = (int64_t) span * 2 * FF * NE;
    if (want("one")) timed("one gate/up dequant launch, 16 exp.", [&] { strata::kernels::iq_dequant_f16(ty, GUc, n_all, big, q); });
    // strided batched GEMM over all experts (same M each): one call
    if (want("batch")) timed("one batched gate/up GEMM (all experts)", [&] {
        const int64_t sa = 0, sb = (int64_t) M * NE, sc = (int64_t) M * 2 * FF;
        oneapi::mkl::blas::column_major::gemm_batch(*q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                                                    2 * FF, M, NE, alpha, (const sycl::half*) dgu, NE, sa,
                                                    (const sycl::half*) X, NE, sb, beta, GU, 2 * FF, sc, E);
    });
    return 0;
}
