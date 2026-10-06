// Strata Void: boolean environment switches that mean what they say.
#pragma once
#include <cctype>
#include <cstdlib>
#include <cstring>

namespace strata {
/// On when `name` is set to anything except "", "0", "false", "no" or "off" (any case, surrounding blanks ignored).
/// The port tested some switches by presence, so `STRATA_VERIFY_NO_HOST=0` turned the switch on.
inline bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    while (*v == ' ' || *v == '\t') ++v;
    size_t n = std::strlen(v);
    while (n > 0 && (v[n - 1] == ' ' || v[n - 1] == '\t')) --n;
    if (n == 0) return false;
    if (n > 5) return true;
    char b[6] = {0};
    for (size_t i = 0; i < n; ++i) b[i] = (char) std::tolower((unsigned char) v[i]);
    return std::strcmp(b, "0") != 0 && std::strcmp(b, "false") != 0 && std::strcmp(b, "no") != 0 &&
           std::strcmp(b, "off") != 0;
}
}  // namespace strata
