// Strata Void: a cheap host-side event trace of a decode round across layer-split stages (STRATA_SPLIT_TRACE=<file>).
// One line per event: "<us since start> <event> <stage first layer> <a> <b>"; off (one branch) when unset.
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace strata {
inline std::FILE* split_trace_file() {
    static std::FILE* f = [] {
        const char* p = std::getenv("STRATA_SPLIT_TRACE");
        return (p != nullptr && *p != 0) ? std::fopen(p, "w") : nullptr;
    }();
    return f;
}
inline void split_trace(const char* ev, long long stage, long long a = 0, long long b = 0) {
    std::FILE* f = split_trace_file();
    if (f == nullptr) return;
    static const auto t0 = std::chrono::steady_clock::now();
    static std::mutex m;
    const long long us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> g(m);
    std::fprintf(f, "%lld %s %lld %lld %lld\n", us, ev, stage, a, b);
}
inline void split_trace_flush() { if (std::FILE* f = split_trace_file()) std::fflush(f); }
}  // namespace strata
