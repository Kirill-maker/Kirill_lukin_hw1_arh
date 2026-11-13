#include <cstdio>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "check.hpp"
#include "hw/hw.hpp"

int main(int argc, char** argv) {
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        std::printf("This CPU does not support AVX2 and FMA\n");
        return 2;
    }
#ifdef HW_AVX512
    if (!__builtin_cpu_supports("avx512f")) {
        std::printf("Built with HW_AVX512=ON, but this CPU does not support AVX-512F\n");
        return 2;
    }
#endif
#ifdef _OPENMP
    if (omp_get_max_threads() < 4) omp_set_num_threads(4);
    std::printf("OpenMP: %d threads\n", omp_get_max_threads());
#else
    std::printf("OpenMP is off: hw::par runs sequentially\n");
#endif

    const char* filter = argc > 1 ? argv[1] : "";
    int ran = 0, failed = 0;
    for (const t::Case& test : t::cases()) {
        if (!std::strstr(test.name, filter)) continue;
        t::failures() = 0;
        t::context().clear();
        test.fn();
        ++ran;
        if (t::failures() > 0) {
            ++failed;
            std::printf("[FAIL] %s (%d checks failed)\n", test.name, t::failures());
        } else {
            std::printf("[ ok ] %s\n", test.name);
        }
    }
    std::printf("\n%d tests, %d failed\n", ran, failed);
    return failed == 0 ? 0 : 1;
}
