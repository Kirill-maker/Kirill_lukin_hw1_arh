#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "check.hpp"
#include "util.hpp"

namespace {

template <class T>
long double ref_sum(const std::vector<T>& v) {
    long double s = 0;
    for (T x : v) s += x;
    return s;
}

}

TEST(sum_and_mean_exact_on_integer_valued_data) {
    t::for_each_backend_and_type([]<class B, class T>() {
        for (std::size_t n : t::sizes) {
            std::vector<T> v(n);
            const std::vector<std::int32_t> ints = t::random_values<std::int32_t>(n, 5, -100, 100);
            std::int64_t expected = 0;
            for (std::size_t i = 0; i < n; ++i) {
                v[i] = static_cast<T>(ints[i]);
                expected += ints[i];
            }
            for (const hw::Exec& ex : t::exec_modes) {
                t::context() = t::describe<B, T>("sum", n, ex);
                CHECK(hw::sum<B>(v.data(), n, ex) == static_cast<hw::sum_t<T>>(expected));

                const auto expected_mean =
                    n == 0 ? hw::mean_t<T>(0)
                           : static_cast<hw::mean_t<T>>(expected) / static_cast<hw::mean_t<T>>(n);
                CHECK(hw::mean<B>(v.data(), n, ex) == expected_mean);
            }
        }
    });
}

TEST(sum_int_does_not_overflow) {
    t::for_each_backend([]<class B>() {
        const std::int32_t lo = std::numeric_limits<std::int32_t>::min();
        const std::int32_t hi = std::numeric_limits<std::int32_t>::max();
        for (std::size_t n : {std::size_t{5}, std::size_t{1000}, std::size_t{70001}}) {
            for (const hw::Exec& ex : t::exec_modes) {
                t::context() = t::describe<B, std::int32_t>("sum of extremes", n, ex);
                const std::vector<std::int32_t> highs(n, hi), lows(n, lo);
                CHECK(hw::sum<B>(highs.data(), n, ex) == static_cast<std::int64_t>(n) * hi);
                CHECK(hw::sum<B>(lows.data(), n, ex) == static_cast<std::int64_t>(n) * lo);
                CHECK(hw::mean<B>(highs.data(), n, ex) == static_cast<double>(hi));
            }
        }

        const std::vector<std::int32_t> v =
            t::random_values<std::int32_t>(100003, 6, lo, hi);
        std::int64_t expected = 0;
        for (auto x : v) expected += x;
        for (const hw::Exec& ex : t::exec_modes) {
            t::context() = t::describe<B, std::int32_t>("sum of random", v.size(), ex);
            CHECK(hw::sum<B>(v.data(), v.size(), ex) == expected);
        }
    });
}

TEST(sum_and_mean_close_on_real_data) {
    t::for_each_backend_and_float([]<class B, class T>() {
        for (std::size_t n : t::sizes) {
            const std::vector<T> v = t::random_values<T>(n, 7, T(-10), T(10));
            const long double expected = ref_sum(v);
            long double magnitude = 0;
            for (T x : v) magnitude += std::fabs(x);
            const long double tolerance = n * std::numeric_limits<T>::epsilon() * magnitude;

            for (const hw::Exec& ex : t::exec_modes) {
                t::context() = t::describe<B, T>("sum", n, ex);
                CHECK(std::fabs(hw::sum<B>(v.data(), n, ex) - expected) <= tolerance);
                if (n > 0) {
                    CHECK(std::fabs(hw::mean<B>(v.data(), n, ex) - expected / n) <= tolerance / n);
                }
            }
        }
    });
}

TEST(mean_of_empty_array_is_zero) {
    t::for_each_backend_and_type([]<class B, class T>() {
        t::context() = t::describe<B, T>("mean", 0, hw::seq);
        const T* nothing = nullptr;
        CHECK(hw::mean<B>(nothing, 0) == hw::mean_t<T>(0));
        CHECK(hw::sum<B>(nothing, 0) == hw::sum_t<T>(0));
    });
}

TEST(sum_int_long_arrays_cross_flush_boundary) {
    const std::int32_t lo = std::numeric_limits<std::int32_t>::min();
    const std::int32_t hi = std::numeric_limits<std::int32_t>::max();
    const std::size_t n = 5'000'003;

    std::vector<std::vector<std::int32_t>> inputs;
    inputs.push_back(std::vector<std::int32_t>(n, hi));
    inputs.push_back(std::vector<std::int32_t>(n, lo));
    inputs.push_back(t::random_values<std::int32_t>(n, 8, lo, hi));
    std::vector<std::int32_t> alternating(n);
    for (std::size_t i = 0; i < n; ++i) alternating[i] = (i % 3 == 0) ? lo : hi;
    inputs.push_back(std::move(alternating));
    std::vector<std::int32_t> low_bits(n);
    for (std::size_t i = 0; i < n; ++i) low_bits[i] = (i % 2) ? 65535 : -65536;
    inputs.push_back(std::move(low_bits));

    t::for_each_backend([&]<class B>() {
        for (std::size_t which = 0; which < inputs.size(); ++which) {
            const std::vector<std::int32_t>& v = inputs[which];
            std::int64_t expected = 0;
            for (auto x : v) expected += x;
            for (const hw::Exec& ex : t::exec_modes) {
                t::context() = t::describe<B, std::int32_t>("long sum", n, ex) + " input " +
                               std::to_string(which);
                CHECK(hw::sum<B>(v.data(), n, ex) == expected);
                CHECK(hw::mean<B>(v.data(), n, ex) ==
                      static_cast<double>(expected) / static_cast<double>(n));
            }
        }
    });
}
