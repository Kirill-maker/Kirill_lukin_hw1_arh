#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "hw/hw.hpp"

namespace t {

template <class T>
const char* type_name() {
    if constexpr (std::is_same_v<T, std::int32_t>) return "int32";
    if constexpr (std::is_same_v<T, float>) return "float32";
    return "float64";
}

template <class F>
void for_each_backend(F&& f) {
    f.template operator()<hw::Scalar>();
    f.template operator()<hw::Avx2>();
#ifdef HW_AVX512
    f.template operator()<hw::Avx512>();
#endif
}

template <class F>
void for_each_backend_and_type(F&& f) {
    for_each_backend([&]<class B>() {
        f.template operator()<B, std::int32_t>();
        f.template operator()<B, float>();
        f.template operator()<B, double>();
    });
}

template <class F>
void for_each_backend_and_float(F&& f) {
    for_each_backend([&]<class B>() {
        f.template operator()<B, float>();
        f.template operator()<B, double>();
    });
}

inline const hw::Exec exec_modes[] = {hw::seq, hw::Exec{true, 64}};

template <class B, class T>
std::string describe(const char* what, std::size_t n, const hw::Exec& ex) {
    return std::string(what) + " " + B::name + " " + type_name<T>() + " n=" + std::to_string(n) +
           (ex.parallel ? " par" : " seq");
}

inline const std::size_t sizes[] = {0,  1,  2,  3,  4,   5,   7,   8,    9,    15,   16,
                                    17, 31, 32, 33, 63,  64,  65,  100,  127,  128,  129,
                                    255, 256, 257, 1000, 1023, 1024, 1025, 4099, 20001};

template <class T>
std::vector<T> random_values(std::size_t n, std::uint32_t seed, T lo, T hi) {
    std::mt19937 gen(seed);
    std::vector<T> v(n);
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(lo, hi);
        for (auto& x : v) x = dist(gen);
    } else {
        std::uniform_real_distribution<T> dist(lo, hi);
        for (auto& x : v) x = dist(gen);
    }
    return v;
}

template <class T>
std::vector<T> random_divisors(std::size_t n, std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::vector<T> v(n);
    for (auto& x : v) {
        if constexpr (std::is_integral_v<T>) {
            const int bits = static_cast<int>(gen() % 30) + 1;
            const T magnitude = static_cast<T>(gen() % (std::uint32_t{1} << bits)) + 1;
            x = (gen() & 1) ? magnitude : static_cast<T>(-magnitude);
        } else {
            const T magnitude = std::uniform_real_distribution<T>(T(0.01), T(100))(gen);
            x = (gen() & 1) ? magnitude : -magnitude;
        }
    }
    return v;
}

template <class T>
std::vector<T> random_operands(std::size_t n, std::uint32_t seed) {
    if constexpr (std::is_integral_v<T>) {
        return random_values<T>(n, seed, std::numeric_limits<T>::min(),
                                std::numeric_limits<T>::max());
    } else {
        return random_values<T>(n, seed, T(-1000), T(1000));
    }
}

template <class T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <class T>
bool same_value(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(a) && std::isnan(b)) return true;
    }
    return same_bits(a, b);
}

template <class T>
bool same_values(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!same_value(a[i], b[i])) return false;
    }
    return true;
}

}
