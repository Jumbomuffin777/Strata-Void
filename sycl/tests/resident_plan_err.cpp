// #871 regression for the SYCL resident_plan: the device-built verify plan with and without its doorbell.
//   with `skip`: an expert not in VRAM hands the group to the host plan (*skip = 0), plan and plan_err untouched;
//                all resident: the plan is built in routing order and *skip = ring.
//   without `skip` (the all-resident graph): an expert not in VRAM leaves an empty plan and raises *plan_err;
//                all resident: the same plan, nothing stored through the null doorbell, plan_err untouched.
// Run with STRATA_PLAN_PARALLEL=1 (default) and =0: the parallel and the one-thread grouping are both checked.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/verify_kernels.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace sk = strata::kernels;
static int g_fail = 0;
#define EXPECT(c, ...) do { if (!(c)) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

int main() {
    auto& q = dpct::get_in_order_queue();
    if (!q.get_device().is_gpu()) { std::printf("GPU required\n"); return 2; }
    constexpr int kN = 6, kK = 2, kExperts = 16, kCapx = 60, kPlanInts = 1024;
    constexpr long long kBlob = 256;
    constexpr uint32_t kRing = 0x5a17, kSentinel = 0xdeadbeef;
    const int32_t ids_h[kN] = {5, 7, 5, 9, 7, 5};
    int32_t* ids = sycl::malloc_device<int32_t>(kN, q);
    int32_t* res = sycl::malloc_device<int32_t>(kExperts, q);
    int32_t* plan = sycl::malloc_shared<int32_t>(kPlanInts, q);
    uint32_t* skip = sycl::malloc_shared<uint32_t>(1, q);
    uint32_t* err = sycl::malloc_host<uint32_t>(1, q);
    const uint8_t* base = reinterpret_cast<const uint8_t*>(uintptr_t(1) << 32);   // never dereferenced
    q.memcpy(ids, ids_h, sizeof ids_h).wait();
    auto set_res = [&](bool expert9_out) {
        std::vector<int32_t> r(kExperts, -1);
        r[5] = 3; r[7] = 0; r[9] = expert9_out ? -1 : 6;
        q.memcpy(res, r.data(), r.size() * sizeof(int32_t)).wait();
    };
    auto reset = [&] { for (int i = 0; i < kPlanInts; ++i) plan[i] = int32_t(kSentinel); *skip = kSentinel; *err = 0; };
    auto run = [&](uint32_t* s, uint32_t* e) {
        sk::resident_plan(ids, kN, kK, res, kExperts, base, nullptr, kBlob, plan, kCapx, s, kRing, &q, e);
        q.wait_and_throw();
    };
    // the expected plan (host loop order): groups 5 {0,2,5}, 7 {1,4}, 9 {3}
    auto check_plan = [&](const char* what) {
        const long long ptr_off = ((4 + (kCapx + 1) + 2 * kCapx) + 1) & ~1ll;
        const int32_t* start = plan + 4; const int32_t* dst = start + kCapx + 1; const int32_t* tok = dst + kCapx;
        const unsigned long long* ptr = reinterpret_cast<const unsigned long long*>(plan + ptr_off);
        const int32_t* start2 = plan + ptr_off + 4 * kCapx;
        EXPECT(plan[0] == 3 && plan[1] == kN && plan[2] == 0, "%s: counts %d %d %d", what, plan[0], plan[1], plan[2]);
        const int32_t es[4] = {0, 3, 5, 6}, ed[kN] = {0, 2, 5, 1, 4, 3};
        for (int g = 0; g < 4; ++g) EXPECT(start[g] == es[g], "%s: start[%d]=%d", what, g, start[g]);
        for (int i = 0; i < kN; ++i) {
            EXPECT(dst[i] == ed[i], "%s: dst[%d]=%d", what, i, dst[i]);
            EXPECT(tok[i] == ed[i] / kK, "%s: tok[%d]=%d", what, i, tok[i]);
        }
        const int slot[3] = {3, 0, 6};
        for (int g = 0; g < 3; ++g)
            EXPECT(ptr[g] == (unsigned long long) (uintptr_t) (base + slot[g] * kBlob), "%s: ptr[%d]", what, g);
        EXPECT(start2[0] == kN, "%s: start2", what);
    };

    set_res(false); reset(); run(skip, err);
    check_plan("doorbell, resident"); EXPECT(*skip == kRing, "doorbell, resident: skip %#x", *skip); EXPECT(*err == 0, "doorbell, resident: err");

    set_res(true); reset(); run(skip, err);
    EXPECT(*skip == 0, "doorbell, miss: skip %#x", *skip); EXPECT(*err == 0, "doorbell, miss: err %u", *err);
    EXPECT(plan[0] == int32_t(kSentinel), "doorbell, miss: the device plan must be left to the host");

    set_res(true); reset(); run(nullptr, err);
    EXPECT(plan[0] == 0 && plan[1] == 0 && plan[2] == 0, "no doorbell, miss: plan not emptied (%d %d %d)", plan[0], plan[1], plan[2]);
    EXPECT(*err == 1, "no doorbell, miss: plan_err %u", *err);

    set_res(true); reset(); run(nullptr, nullptr);
    EXPECT(plan[0] == 0 && plan[1] == 0 && plan[2] == 0, "no doorbell, no flag, miss: plan not emptied");

    set_res(false); reset(); run(nullptr, err);
    check_plan("no doorbell, resident"); EXPECT(*err == 0, "no doorbell, resident: err %u", *err);

    sycl::free(ids, q); sycl::free(res, q); sycl::free(plan, q); sycl::free(skip, q); sycl::free(err, q);
    const char* par = std::getenv("STRATA_PLAN_PARALLEL");
    std::printf("%s resident_plan #871 (STRATA_PLAN_PARALLEL=%s): doorbell/no-doorbell x resident/miss\n",
                g_fail ? "FAILED" : "PASS", par ? par : "default");
    return g_fail ? 1 : 0;
}
