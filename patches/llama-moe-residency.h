// llama-moe-residency.h - the adaptive part of Strata Void's hot/cold MoE runtime for llama.cpp (no ggml dependency,
// unit-tested by tools/llamacpp/test_moe_residency.cpp). Copied into llama.cpp's src/ by tools/llamacpp/build_hotcold.sh
// and used by patches/llama.cpp-moe-dynamic-residency.patch.
//
// Per hot/cold-split MoE layer the runtime keeps a fixed number of hot slots (expert weights in the layer's GPU) and
// cold slots (host RAM, computed by the CPU); hot + cold always hold every expert exactly once. This class decides,
// from the routing it observes, which cold expert should take which hot slot:
//
//   * score(e) = exponentially decayed count of tokens that selected e (half-life `half_life` tokens of the layer);
//   * a swap (hot slot of h <-> cold slot of x) is proposed when score(x) > score(h) * ratio + margin, best first,
//     at most `max_swaps_layer` per layer and `max_swaps_total` per round (bounded transfer work per rebalance);
//   * an expert that just moved is not moved back for `cooldown` rounds (no thrashing on a tie).
//
// The class only plans: the caller moves the weights and then calls apply() with the swaps it made.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

struct llama_moe_residency_params {
    float    half_life       = 256.0f; // tokens after which a count weighs half
    float    ratio           = 1.25f;  // a cold expert must be this much hotter than the hot one it replaces ...
    float    margin          = 2.0f;   // ... plus this many (decayed) tokens
    int      max_swaps_layer = 4;
    int      max_swaps_total = 24;
    int      cooldown        = 4;      // rounds an expert that moved stays where it is
};

struct llama_moe_swap {
    int layer;     // the caller's layer index
    int hot_slot;  // slot in the hot part: its expert moves to the cold slot
    int cold_slot; // slot in the cold part: its expert moves to the hot slot
};

class llama_moe_residency_policy {
public:
    // hot[l] / cold[l]: the expert held by each slot of layer l (canonical ids; every expert exactly once over both)
    llama_moe_residency_policy(int n_expert, std::vector<std::vector<int32_t>> hot, std::vector<std::vector<int32_t>> cold,
                               llama_moe_residency_params p = {})
        : n_expert_(n_expert), p_(p), hot_(std::move(hot)), cold_(std::move(cold)) {
        const size_t n = hot_.size();
        score_.assign(n, std::vector<double>(n_expert_, 0.0));
        moved_.assign(n, std::vector<int>(n_expert_, -1000000));
        tokens_.assign(n, 0);
        hits_.assign(n, 0);
        sel_.assign(n, 0);
        decay_ = std::pow(0.5, 1.0 / std::max(1.0f, p_.half_life));
    }

    size_t n_layers() const { return hot_.size(); }
    const std::vector<int32_t> & hot(int l)  const { return hot_[l]; }
    const std::vector<int32_t> & cold(int l) const { return cold_[l]; }

    // routing of n_tokens tokens of layer l: ids[t * k + j], each in [0, n_expert)
    void observe(int l, const int32_t * ids, int n_tokens, int k) {
        auto & s = score_[l];
        const double d = std::pow(decay_, (double) n_tokens);
        for (double & v : s) {
            v *= d;
        }
        if (in_hot_.size() != hot_.size()) {
            rebuild_index();
        }
        for (int i = 0; i < n_tokens * k; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= n_expert_) {
                continue;
            }
            s[e] += 1.0;
            hits_[l] += in_hot_[l][e] ? 1 : 0;
        }
        sel_[l]    += (uint64_t) n_tokens * k;
        tokens_[l] += (uint64_t) n_tokens;
    }

    // the swaps worth making now (not yet applied)
    std::vector<llama_moe_swap> plan() const {
        struct cand { double gain; llama_moe_swap s; };
        std::vector<cand> all;
        for (size_t l = 0; l < hot_.size(); ++l) {
            const auto & s = score_[l];
            std::vector<int> hs(hot_[l].size()), cs(cold_[l].size());
            for (size_t i = 0; i < hs.size(); ++i) hs[i] = (int) i;
            for (size_t i = 0; i < cs.size(); ++i) cs[i] = (int) i;
            // coldest hot slots first, hottest cold slots first; ties by expert id (deterministic)
            std::sort(hs.begin(), hs.end(), [&](int a, int b) {
                const int32_t ea = hot_[l][a], eb = hot_[l][b];
                return s[ea] != s[eb] ? s[ea] < s[eb] : ea < eb; });
            std::sort(cs.begin(), cs.end(), [&](int a, int b) {
                const int32_t ea = cold_[l][a], eb = cold_[l][b];
                return s[ea] != s[eb] ? s[ea] > s[eb] : ea < eb; });
            int n = 0;
            size_t ih = 0;
            for (size_t ic = 0; ic < cs.size() && ih < hs.size() && n < p_.max_swaps_layer; ++ic) {
                const int32_t x = cold_[l][cs[ic]];
                if (round_ - moved_[l][x] < p_.cooldown) {
                    continue;
                }
                while (ih < hs.size() && round_ - moved_[l][hot_[l][hs[ih]]] < p_.cooldown) {
                    ++ih;
                }
                if (ih == hs.size()) {
                    break;
                }
                const int32_t h = hot_[l][hs[ih]];
                if (!(s[x] > s[h] * p_.ratio + p_.margin)) {
                    break;                          // the best remaining cold expert is not worth the coldest hot slot
                }
                all.push_back({s[x] - s[h], {(int) l, hs[ih], cs[ic]}});
                ++ih;
                ++n;
            }
        }
        std::stable_sort(all.begin(), all.end(), [](const cand & a, const cand & b) { return a.gain > b.gain; });
        std::vector<llama_moe_swap> out;
        for (const auto & c : all) {
            if ((int) out.size() >= p_.max_swaps_total) break;
            out.push_back(c.s);
        }
        return out;
    }

    // record swaps the caller made (the weights of hot slot and cold slot exchanged); ends a round
    void apply(const std::vector<llama_moe_swap> & swaps) {
        for (const auto & w : swaps) {
            int32_t & h = hot_[w.layer][w.hot_slot];
            int32_t & c = cold_[w.layer][w.cold_slot];
            std::swap(h, c);
            moved_[w.layer][h] = round_;
            moved_[w.layer][c] = round_;
            if (!in_hot_.empty()) {
                in_hot_[w.layer][h] = 1;
                in_hot_[w.layer][c] = 0;
            }
        }
        ++round_;
    }

    // share of observed selections that found their expert hot, since construction or reset_counters()
    double hit_rate() const {
        uint64_t h = 0, s = 0;
        for (size_t l = 0; l < hits_.size(); ++l) { h += hits_[l]; s += sel_[l]; }
        return s ? (double) h / (double) s : 1.0;
    }
    uint64_t selections() const { uint64_t s = 0; for (auto v : sel_) s += v; return s; }
    void reset_counters() { std::fill(hits_.begin(), hits_.end(), 0); std::fill(sel_.begin(), sel_.end(), 0); }

    // where expert e of layer l is: hot slot (>= 0) or -(cold slot) - 1
    int where(int l, int32_t e) const {
        for (size_t i = 0; i < hot_[l].size(); ++i) if (hot_[l][i] == e) return (int) i;
        for (size_t i = 0; i < cold_[l].size(); ++i) if (cold_[l][i] == e) return -(int) i - 1;
        return INT32_MIN;
    }

private:
    void rebuild_index() {
        in_hot_.assign(hot_.size(), std::vector<uint8_t>(n_expert_, 0));
        for (size_t l = 0; l < hot_.size(); ++l) for (int32_t e : hot_[l]) in_hot_[l][e] = 1;
    }

    int n_expert_;
    llama_moe_residency_params p_;
    std::vector<std::vector<int32_t>> hot_, cold_;
    std::vector<std::vector<double>> score_;
    std::vector<std::vector<int>> moved_;
    std::vector<std::vector<uint8_t>> in_hot_;
    std::vector<uint64_t> tokens_, hits_, sel_;
    double decay_ = 1.0;
    int round_ = 0;
};
