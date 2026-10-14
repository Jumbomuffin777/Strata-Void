// Unit tests for patches/llama-moe-residency.h (the adaptive hot/cold expert policy). No GPU, no llama.cpp:
//   g++ -std=c++17 -O2 -I patches tools/llamacpp/test_moe_residency.cpp -o /tmp/test_moe_residency && /tmp/test_moe_residency
#include "llama-moe-residency.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

// layers x n_expert, the first n_hot experts hot
static llama_moe_residency_policy make(int layers, int n_expert, int n_hot, llama_moe_residency_params p = {}) {
    std::vector<std::vector<int32_t>> hot(layers), cold(layers);
    for (int l = 0; l < layers; ++l)
        for (int e = 0; e < n_expert; ++e) (e < n_hot ? hot : cold)[l].push_back(e);
    return llama_moe_residency_policy(n_expert, hot, cold, p);
}

// every expert exactly once over hot + cold, sizes unchanged
static void check_partition(const llama_moe_residency_policy & r, int n_expert, size_t n_hot) {
    for (size_t l = 0; l < r.n_layers(); ++l) {
        std::set<int32_t> seen;
        for (int32_t e : r.hot((int) l)) seen.insert(e);
        for (int32_t e : r.cold((int) l)) seen.insert(e);
        CHECK((int) seen.size() == n_expert);
        CHECK(r.hot((int) l).size() == n_hot);
        CHECK(r.hot((int) l).size() + r.cold((int) l).size() == (size_t) n_expert);
    }
}

// tokens choosing k distinct experts from `pool` (uniform)
static std::vector<int32_t> route(std::mt19937 & g, const std::vector<int32_t> & pool, int n_tokens, int k) {
    std::vector<int32_t> ids;
    for (int t = 0; t < n_tokens; ++t) {
        std::vector<int32_t> p = pool;
        std::shuffle(p.begin(), p.end(), g);
        ids.insert(ids.end(), p.begin(), p.begin() + k);
    }
    return ids;
}

static void test_adapts_to_a_new_working_set() {
    const int E = 64, H = 16, K = 4;
    auto r = make(2, E, H);
    std::mt19937 g(1);
    std::vector<int32_t> pool;                         // the conversation now uses experts 40..55, all cold
    for (int e = 40; e < 56; ++e) pool.push_back(e);
    double first = -1;
    for (int round = 0; round < 40; ++round) {
        r.reset_counters();
        for (int l = 0; l < 2; ++l) { auto ids = route(g, pool, 16, K); r.observe(l, ids.data(), 16, K); }
        if (first < 0) first = r.hit_rate();
        r.apply(r.plan());
        check_partition(r, E, H);
    }
    r.reset_counters();
    for (int l = 0; l < 2; ++l) { auto ids = route(g, pool, 64, K); r.observe(l, ids.data(), 64, K); }
    std::printf("adapt: hit rate %.2f -> %.2f\n", first, r.hit_rate());
    CHECK(first < 0.05);
    CHECK(r.hit_rate() > 0.99);
}

static void test_no_swaps_without_a_clear_gain() {
    const int E = 32, H = 8, K = 2;
    auto r = make(1, E, H);
    std::vector<int32_t> all(E);
    for (int e = 0; e < E; ++e) all[e] = e;
    std::vector<int32_t> ids;                          // every expert exactly as often: nothing to gain
    for (int t = 0; t < 64; ++t) { ids.push_back(all[(2 * t) % E]); ids.push_back(all[(2 * t + 1) % E]); }
    r.observe(0, ids.data(), 64, K);
    CHECK(r.plan().empty());
}

static void test_budgets_and_determinism() {
    const int E = 128, H = 32, K = 8;
    llama_moe_residency_params p; p.max_swaps_layer = 3; p.max_swaps_total = 5;
    auto a = make(4, E, H, p), b = make(4, E, H, p);
    std::mt19937 g(7);
    std::vector<int32_t> pool;
    for (int e = 64; e < 128; ++e) pool.push_back(e);
    for (int l = 0; l < 4; ++l) {
        auto ids = route(g, pool, 32, K);
        a.observe(l, ids.data(), 32, K);
        b.observe(l, ids.data(), 32, K);
    }
    auto pa = a.plan(), pb = b.plan();
    CHECK((int) pa.size() == 5);
    CHECK(pa.size() == pb.size());
    std::vector<int> per(4, 0);
    for (size_t i = 0; i < pa.size(); ++i) {
        CHECK(pa[i].layer == pb[i].layer && pa[i].hot_slot == pb[i].hot_slot && pa[i].cold_slot == pb[i].cold_slot);
        per[pa[i].layer]++;
    }
    for (int n : per) CHECK(n <= 3);
    std::set<std::pair<int, int>> hs, cs;              // a slot is used once per round
    for (auto & s : pa) { CHECK(hs.insert({s.layer, s.hot_slot}).second); CHECK(cs.insert({s.layer, s.cold_slot}).second); }
}

static void test_cooldown_prevents_ping_pong() {
    const int E = 16, H = 4, K = 1;
    llama_moe_residency_params p; p.cooldown = 3; p.margin = 0.5f; p.ratio = 1.0f;
    auto r = make(1, E, H, p);
    int32_t e10 = 10;
    for (int i = 0; i < 8; ++i) r.observe(0, &e10, 1, K);
    auto s = r.plan();
    CHECK(s.size() == 1);
    r.apply(s);
    CHECK(r.where(0, 10) >= 0);                         // 10 is hot now
    const int32_t evicted = r.cold(0)[s[0].cold_slot];
    for (int i = 0; i < 30; ++i) r.observe(0, &evicted, 1, K);   // the evicted one is suddenly hot again
    CHECK(r.plan().empty() || r.plan()[0].hot_slot != s[0].hot_slot);   // 10 is not moved back at once
    r.apply({}); r.apply({}); r.apply({});
    bool back = false;
    for (auto & w : r.plan()) back |= r.cold(0)[w.cold_slot] == evicted;
    CHECK(back);                                         // after the cooldown it may return
}

static void test_ignores_bad_ids() {
    auto r = make(1, 8, 2);
    int32_t ids[4] = {-1, 8, 99, 3};
    r.observe(0, ids, 1, 4);
    CHECK(r.selections() == 4);
    check_partition(r, 8, 2);
}

int main() {
    test_adapts_to_a_new_working_set();
    test_no_swaps_without_a_clear_gain();
    test_budgets_and_determinism();
    test_cooldown_prevents_ping_pong();
    test_ignores_bad_ids();
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
