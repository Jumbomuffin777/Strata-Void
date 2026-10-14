// op-profile: where one prompt-sized batch spends its time, per ggml op and per buffer (device), in llama.cpp.
// Same CLI as llama.cpp tools (model, --device, --tensor-split, -ot, -ub, ...), plus --tokens N (default 512) and
// --runs N (default 2; the first warms up and is not counted). Prints a table sorted by time.
//
// Decode: --decode N reads --tokens N0 of prompt first, then times N single-token decodes twice: once without any
// callback (the real speed at that depth) and once synchronizing only at nodes whose name matches --sync REGEX
// (e.g. "^(kda_out|dsa_out|ffn_moe_ids_hot|ffn_moe_ids_cold|ffn_moe_down|l_out)-"): each segment's time is then
// attributed to the node that ends it ("until <name> @ <buffer>"), with a few syncs per layer instead of per node.
//
// It sets an eval callback that asks for every node, so the scheduler computes node by node and synchronizes after
// each: the time between two callbacks is that node's time on its device (plus a launch + sync, ~tens of us). Totals
// are therefore slower than a normal run; the shares are what to read. Text is never printed.
//
// Build against a llama.cpp build tree like routing-dump.cpp (tools/llamacpp/build_hotcold.sh builds both).
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <vector>

struct prof_state {
    bool on = false;
    bool ask_none = false;          // the unprofiled reference run: no node is asked for
    bool use_re = false;
    std::regex re;
    int64_t last = 0;
    std::map<std::string, std::pair<double, long long>> by_op;    // "op @ buffer" -> (us, nodes)
    std::map<std::string, std::pair<double, long long>> by_name;  // node name without the layer -> (us, nodes)
};

static std::string base_name(const char * n) {
    std::string s(n);
    const auto d = s.find_last_of('-');
    if (d != std::string::npos && d + 1 < s.size() && isdigit((unsigned char) s[d + 1])) s.resize(d);
    return s;
}

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (prof_state *) ud;
    if (ask) return !st->ask_none && (!st->use_re || std::regex_search(t->name, st->re));
    const int64_t now = ggml_time_us();
    if (st->on && st->last) {
        const double us = (double) (now - st->last);
        const char * buf = t->buffer ? ggml_backend_buffer_name(t->buffer) : (t->view_src && t->view_src->buffer ?
                           ggml_backend_buffer_name(t->view_src->buffer) : "?");
        auto & a = st->by_op[std::string(ggml_op_desc(t)) + " @ " + buf];
        a.first += us; a.second++;
        auto & b = st->by_name[(st->use_re ? "until " : "") + base_name(t->name) + (st->use_re ? std::string(" @ ") + buf : "")];
        b.first += us; b.second++;
    }
    st->last = now;
    return true;
}

static void print(const char * title, const std::map<std::string, std::pair<double, long long>> & m, double total, int top) {
    std::vector<std::pair<std::string, std::pair<double, long long>>> v(m.begin(), m.end());
    std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.second.first > b.second.first; });
    printf("\n%-52s %10s %7s %8s\n", title, "ms", "share", "nodes");
    for (int i = 0; i < (int) v.size() && i < top; ++i)
        printf("%-52s %10.1f %6.1f%% %8lld\n", v[i].first.c_str(), v[i].second.first / 1000.0,
               100.0 * v[i].second.first / total, v[i].second.second);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    int n_tok = 512, runs = 2, n_decode = 0;
    std::string sync_re;
    std::vector<char *> rest = {argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--tokens") && i + 1 < argc) n_tok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--runs") && i + 1 < argc) runs = std::max(2, atoi(argv[++i]));
        else if (!strcmp(argv[i], "--decode") && i + 1 < argc) n_decode = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sync") && i + 1 < argc) sync_re = argv[++i];
        else rest.push_back(argv[i]);
    }
    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    prof_state st;
    if (!sync_re.empty()) { st.use_re = true; st.re = std::regex(sync_re); }
    params.cb_eval = cb; params.cb_eval_user_data = &st; params.warmup = false;
    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    if (!ctx) { LOG_ERR("failed to init\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::vector<llama_token> toks(n_tok);
    for (int i = 0; i < n_tok; ++i) toks[i] = (llama_token) (1000 + (i * 7919) % (llama_vocab_n_tokens(vocab) - 2000));
    if (n_decode > 0) {
        // the prompt (not profiled), then the reference decodes, then the profiled ones, all at about the same depth
        st.on = false; st.ask_none = true;
        for (int i = 0; i < n_tok; i += 1024) {
            const int n = std::min(1024, n_tok - i);
            if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, n))) { LOG_ERR("prompt decode failed\n"); return 1; }
        }
        llama_token tok = 1234;
        auto step = [&](int n) {
            const int64_t t0 = ggml_time_us();
            for (int i = 0; i < n; ++i) {
                if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) { LOG_ERR("decode failed\n"); exit(1); }
                llama_synchronize(ctx);
                tok = 1000 + (tok * 31 + 7) % 100000;
            }
            return (ggml_time_us() - t0) / 1e3 / n;
        };
        step(4);                                              // warm-up
        const double ref_ms = step(n_decode);
        st.ask_none = false; st.on = true; st.last = 0;
        const double prof_ms = step(n_decode);
        double total = 0;
        for (auto & kv : st.by_name) total += kv.second.first;
        printf("op-profile decode at depth %d: %.2f ms/token (%.2f tok/s) unprofiled; %.2f ms/token profiled, %.2f ms/token "
               "attributed\n", n_tok, ref_ms, 1000.0 / ref_ms, prof_ms, total / 1000.0 / n_decode);
        print(st.use_re ? "segment (ends at node @ buffer)" : "node (layer number removed)", st.by_name, total, 40);
        if (!st.use_re) print("op @ buffer", st.by_op, total, 30);
        llama_backend_free();
        return 0;
    }
    double wall = 0;
    for (int r = 0; r < runs; ++r) {
        llama_memory_clear(llama_get_memory(ctx), true);
        st.on = r > 0; st.last = 0;
        const int64_t t0 = ggml_time_us();
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), n_tok))) { LOG_ERR("decode failed\n"); return 1; }
        llama_synchronize(ctx);
        const double s = (ggml_time_us() - t0) / 1e6;
        if (r > 0) wall += s;
        LOG_INF("op-profile: run %d: %d tokens in %.2f s (%.1f tok/s, profiled)\n", r, n_tok, s, n_tok / s);
    }
    double total = 0;
    for (auto & kv : st.by_op) total += kv.second.first;
    printf("op-profile: %d runs of %d tokens, %.2f s wall, %.2f s attributed to nodes\n", runs - 1, n_tok, wall, total / 1e6);
    print("op @ buffer", st.by_op, total, 30);
    print("node (layer number removed)", st.by_name, total, 30);
    llama_backend_free();
    return 0;
}
