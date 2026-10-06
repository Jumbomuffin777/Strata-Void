// strata::env_flag: STRATA_VERIFY_NO_HOST=0 must mean off (it was tested by presence).
#include "strata/env_flag.hpp"
#include <cstdio>
#include <cstdlib>

int main() {
    struct { const char* v; bool on; } cases[] = {
        {nullptr, false}, {"", false}, {"0", false}, {" 0 ", false}, {"false", false}, {"FALSE", false},
        {"no", false}, {"Off", false}, {"1", true}, {"true", true}, {"yes", true}, {"on", true}, {"2", true},
        {"00", true}, {"falsey", true}};
    int bad = 0;
    for (const auto& c : cases) {
        if (c.v) setenv("STRATA_ENV_FLAG_TEST", c.v, 1); else unsetenv("STRATA_ENV_FLAG_TEST");
        if (strata::env_flag("STRATA_ENV_FLAG_TEST") != c.on) {
            std::printf("FAIL value '%s': expected %s\n", c.v ? c.v : "(unset)", c.on ? "on" : "off");
            ++bad;
        }
    }
    std::printf("%s env_flag (%zu cases)\n", bad ? "FAILED" : "PASS", sizeof cases / sizeof cases[0]);
    return bad ? 1 : 0;
}
