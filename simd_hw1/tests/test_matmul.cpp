#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "check.hpp"
#include "util.hpp"

namespace {

struct Shape {
    std::size_t n, k, m;
};

const Shape shapes[] = {{1, 1, 1},   {1, 5, 1},    {5, 1, 7},     {3, 4, 5},    {8, 8, 8},
                        {16, 16, 16}, {17, 33, 9},  {2, 200, 3},   {9, 7, 100},  {33, 65, 31},
                        {31, 67, 300}, {70, 130, 257}, {5, 64, 256}, {4, 63, 255}, {6, 129, 513}};

template <class T>
std::string describe(const char* backend, const Shape& s, const hw::Exec& ex) {
    return std::string("matmul ") + backend + " " + t::type_name<T>() + " " +
           std::to_string(s.n) + "x" + std::to_string(s.k) + "x" + std::to_string(s.m) +
           (ex.parallel ? " par" : " seq");
}

const hw::Exec matmul_modes[] = {hw::seq, hw::Exec{true, 1}};

}

TEST(matmul_exact_on_integer_valued_data) {
    t::for_each_backend_and_type([]<class B, class T>() {
        for (const Shape& s : shapes) {
            const bool is_int = std::is_integral_v<T>;
            const std::int32_t range = is_int ? std::numeric_limits<std::int32_t>::max() : 8;
            const auto ia = t::random_values<std::int32_t>(s.n * s.k, 21, -range, range);
            const auto ib = t::random_values<std::int32_t>(s.k * s.m, 22, -range, range);
            const std::vector<T> a(ia.begin(), ia.end()), b(ib.begin(), ib.end());

            std::vector<T> expected(s.n * s.m);
            for (std::size_t i = 0; i < s.n; ++i) {
                for (std::size_t j = 0; j < s.m; ++j) {
                    std::uint64_t acc = 0;
                    for (std::size_t p = 0; p < s.k; ++p) {
                        acc += static_cast<std::uint64_t>(static_cast<std::int64_t>(ia[i * s.k + p]) *
                                                          ib[p * s.m + j]);
                    }
                    expected[i * s.m + j] = is_int ? static_cast<T>(static_cast<std::int32_t>(acc))
                                                   : static_cast<T>(static_cast<std::int64_t>(acc));
                }
            }

            for (const hw::Exec& ex : matmul_modes) {
                t::context() = describe<T>(B::name, s, ex);
                std::vector<T> c(s.n * s.m, T(-1));
                hw::matmul<B>(a.data(), b.data(), c.data(), s.n, s.k, s.m, ex);
                CHECK(t::same_values(c, expected));
            }
        }
    });
}

TEST(matmul_close_on_real_data) {
    t::for_each_backend_and_float([]<class B, class T>() {
        for (const Shape& s : shapes) {
            const std::vector<T> a = t::random_values<T>(s.n * s.k, 23, T(-2), T(2));
            const std::vector<T> b = t::random_values<T>(s.k * s.m, 24, T(-2), T(2));
            for (const hw::Exec& ex : matmul_modes) {
                t::context() = describe<T>(B::name, s, ex);
                std::vector<T> c(s.n * s.m);
                hw::matmul<B>(a.data(), b.data(), c.data(), s.n, s.k, s.m, ex);

                for (std::size_t i = 0; i < s.n; ++i) {
                    for (std::size_t j = 0; j < s.m; ++j) {
                        long double expected = 0, magnitude = 0;
                        for (std::size_t p = 0; p < s.k; ++p) {
                            const long double term =
                                static_cast<long double>(a[i * s.k + p]) * b[p * s.m + j];
                            expected += term;
                            magnitude += std::fabs(term);
                        }
                        const long double tolerance =
                            s.k * std::numeric_limits<T>::epsilon() * magnitude;
                        CHECK(std::fabs(c[i * s.m + j] - expected) <= tolerance);
                    }
                }
            }
        }
    });
}

TEST(matmul_simd_and_parallel_equal_scalar_bitwise) {
    t::for_each_backend_and_float([]<class B, class T>() {
        for (const Shape& s : shapes) {
            const std::vector<T> a = t::random_values<T>(s.n * s.k, 25, T(-2), T(2));
            const std::vector<T> b = t::random_values<T>(s.k * s.m, 26, T(-2), T(2));
            std::vector<T> expected(s.n * s.m);
            hw::matmul<hw::Scalar>(a.data(), b.data(), expected.data(), s.n, s.k, s.m);
            for (const hw::Exec& ex : matmul_modes) {
                t::context() = describe<T>(B::name, s, ex);
                std::vector<T> c(s.n * s.m);
                hw::matmul<B>(a.data(), b.data(), c.data(), s.n, s.k, s.m, ex);
                CHECK(t::same_values(c, expected));
            }
        }
    });
}

TEST(matmul_by_identity_returns_same_matrix) {
    t::for_each_backend_and_type([]<class B, class T>() {
        const std::size_t n = 37, m = 53;
        t::context() = describe<T>(B::name, {n, m, m}, hw::seq);
        const std::vector<T> a = t::random_values<T>(n * m, 27, T(-1000), T(1000));
        std::vector<T> identity(m * m, T(0)), c(n * m);
        for (std::size_t i = 0; i < m; ++i) identity[i * m + i] = T(1);

        hw::matmul<B>(a.data(), identity.data(), c.data(), n, m, m);
        CHECK(t::same_values(c, a));
    });
}

TEST(matmul_with_empty_dimensions) {
    t::for_each_backend_and_type([]<class B, class T>() {
        t::context() = describe<T>(B::name, {3, 0, 4}, hw::seq);
        std::vector<T> c(12, T(5));
        const T* nothing = nullptr;
        hw::matmul<B>(nothing, nothing, c.data(), 3, 0, 4);
        for (T x : c) CHECK(x == T(0));

        hw::matmul<B>(nothing, nothing, static_cast<T*>(nullptr), 0, 5, 5);
        hw::matmul<B>(nothing, nothing, static_cast<T*>(nullptr), 5, 5, 0, hw::par);
    });
}
