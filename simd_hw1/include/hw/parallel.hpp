#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hw {

struct Exec {
    bool parallel = false;
    std::size_t grain = std::size_t{1} << 15;
};

inline constexpr Exec seq{false};
inline constexpr Exec par{true};

inline int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

namespace detail {

inline std::size_t chunk_count(const Exec& ex, std::size_t n) {
#ifdef _OPENMP
    if (ex.parallel && !omp_in_parallel()) {
        const std::size_t by_size = n / std::max<std::size_t>(ex.grain, 1);
        return std::clamp<std::size_t>(by_size, 1, static_cast<std::size_t>(max_threads()));
    }
#else
    (void)ex;
    (void)n;
#endif
    return 1;
}

inline std::size_t chunk_begin(std::size_t n, std::size_t chunks, std::size_t c,
                               std::size_t align) {
    if (c >= chunks) return n;
    return n * c / chunks / align * align;
}

}

template <class Body>
void parallel_for(const Exec& ex, std::size_t n, Body&& body, std::size_t align = 1) {
    const std::size_t chunks = detail::chunk_count(ex, n);
    if (chunks == 1) {
        body(std::size_t{0}, n);
        return;
    }
    (void)align;
#ifdef _OPENMP
    const int count = static_cast<int>(chunks);
#pragma omp parallel for schedule(static) num_threads(count)
    for (int c = 0; c < count; ++c) {
        body(detail::chunk_begin(n, chunks, c, align), detail::chunk_begin(n, chunks, c + 1, align));
    }
#endif
}

template <class Body, class Join>
auto parallel_reduce(const Exec& ex, std::size_t n, Body&& body, Join&& join,
                     std::size_t align = 1) {
    const std::size_t chunks = detail::chunk_count(ex, n);
    if (chunks == 1) return body(std::size_t{0}, n);

    using R = decltype(body(std::size_t{0}, n));
    std::vector<R> part(chunks);
    (void)align;
#ifdef _OPENMP
    const int count = static_cast<int>(chunks);
#pragma omp parallel for schedule(static) num_threads(count)
    for (int c = 0; c < count; ++c) {
        part[c] =
            body(detail::chunk_begin(n, chunks, c, align), detail::chunk_begin(n, chunks, c + 1, align));
    }
#endif
    R result = part[0];
    for (std::size_t c = 1; c < chunks; ++c) result = join(result, part[c]);
    return result;
}

}
