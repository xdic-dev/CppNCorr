#ifndef NCORR_INTERNAL_DIAGNOSTICS_HPP
#define NCORR_INTERNAL_DIAGNOSTICS_HPP

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>

namespace ncorr {
namespace details {

inline bool diagnostics_enabled() {
    static const bool enabled = []() {
        const char *env = std::getenv("NCORR_ENABLE_DIAGNOSTICS");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();

    return enabled;
}

template <class... Args>
inline void diagnostic_log(std::ostream &os, Args&&... args) {
    if (!diagnostics_enabled()) {
        return;
    }

    (os << ... << std::forward<Args>(args)) << std::endl;
}

} // namespace details
} // namespace ncorr

#endif
