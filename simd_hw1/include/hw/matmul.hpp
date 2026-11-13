#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include "backend_scalar.hpp"
#include "common.hpp"
#include "parallel.hpp"

namespace hw {

namespace detail {

inline constexpr std::size_t matmul_block_k = 64;
inline constexpr std::size_t matmul_block_j = 256;

template <class X, class T, std::size_t U>
void axpy_step(T* c, const std::array<const T*, U>& rows, const std::array<T, U>& coef,
               std::size_t j) {
    X acc = X::load(c + j);
    for (std::size_t u = 0; u < U; ++u) acc = fmadd(X(coef[u]), X::load(rows[u] + j), acc);
    acc.store(c + j);
}

template <class B, Element T, std::size_t U>
void axpy(T* c, const std::array<const T*, U>& rows, const std::array<T, U>& coef,
          std::size_t j0, std::size_t j1) {
    using V = typename B::template Vec<T>;
    using S = Scalar::Vec<T>;

    std::size_t j = j0;
    for (; j + V::width <= j1; j += V::width) axpy_step<V>(c, rows, coef, j);
    for (; j < j1; ++j) axpy_step<S>(c, rows, coef, j);
}

template <class B, Element T>
void matmul_rows(const T* a, const T* b, T* c, std::size_t r0, std::size_t r1, std::size_t k,
                 std::size_t m) {
    std::fill(c + r0 * m, c + r1 * m, T{});

    for (std::size_t j0 = 0; j0 < m; j0 += matmul_block_j) {
        const std::size_t j1 = std::min(m, j0 + matmul_block_j);
        for (std::size_t k0 = 0; k0 < k; k0 += matmul_block_k) {
            const std::size_t k1 = std::min(k, k0 + matmul_block_k);
            for (std::size_t i = r0; i < r1; ++i) {
                const T* ai = a + i * k;
                T* ci = c + i * m;
                std::size_t p = k0;
                for (; p + 4 <= k1; p += 4) {
                    const T* bp = b + p * m;
                    axpy<B, T, 4>(ci, {bp, bp + m, bp + 2 * m, bp + 3 * m},
                                  {ai[p], ai[p + 1], ai[p + 2], ai[p + 3]}, j0, j1);
                }
                for (; p < k1; ++p) axpy<B, T, 1>(ci, {b + p * m}, {ai[p]}, j0, j1);
            }
        }
    }
}

}

template <class B, Element T>
void matmul(const T* a, const T* b, T* c, std::size_t n, std::size_t k, std::size_t m,
            const Exec& ex = seq) {
    const std::size_t row_cost = std::max<std::size_t>(k * m, 1);
    const Exec rows_ex{ex.parallel, std::max<std::size_t>(ex.grain / row_cost, 1)};
    parallel_for(rows_ex, n, [=](std::size_t r0, std::size_t r1) {
        detail::matmul_rows<B>(a, b, c, r0, r1, k, m);
    });
}

}
