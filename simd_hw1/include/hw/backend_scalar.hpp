#pragma once

#include <bit>
#include <cmath>

#include "common.hpp"

namespace hw {

namespace scalar {

template <class T>
using wrap_t = typename std::conditional_t<std::is_integral_v<T>, std::make_unsigned<T>,
                                           std::type_identity<T>>::type;

template <Element T>
struct Vec {
    using value_type = T;
    static constexpr std::size_t width = 1;

    T v;

    Vec() = default;
    explicit Vec(T x) : v(x) {}

    static Vec load(const T* p) { return Vec(*p); }
    void store(T* p) const { *p = v; }

    friend Vec operator+(Vec a, Vec b) { return Vec(T(wrap_t<T>(a.v) + wrap_t<T>(b.v))); }
    friend Vec operator-(Vec a, Vec b) { return Vec(T(wrap_t<T>(a.v) - wrap_t<T>(b.v))); }
    friend Vec operator*(Vec a, Vec b) { return Vec(T(wrap_t<T>(a.v) * wrap_t<T>(b.v))); }
    friend Vec operator/(Vec a, Vec b) { return Vec(T(a.v / b.v)); }

    friend Vec fmadd(Vec a, Vec b, Vec c) {
        if constexpr (std::is_floating_point_v<T>) {
            return Vec(std::fma(a.v, b.v, c.v));
        } else {
            return a * b + c;
        }
    }

    friend Vec min(Vec a, Vec b) { return a.v < b.v ? a : b; }
    friend Vec max(Vec a, Vec b) { return a.v > b.v ? a : b; }

    friend Vec pow2(Vec k)
        requires std::floating_point<T>
    {
        if constexpr (std::same_as<T, float>) {
            return Vec(std::bit_cast<float>((static_cast<std::int32_t>(k.v) + 127) << 23));
        } else {
            return Vec(std::bit_cast<double>((static_cast<std::int64_t>(k.v) + 1023) << 52));
        }
    }

    struct Acc {
        sum_t<T> s{};
        void add(Vec x) { s += x.v; }
        void merge(const Acc& o) { s += o.s; }
        sum_t<T> get() const { return s; }
    };
};

}

struct Scalar {
    static constexpr const char* name = "scalar";
    template <Element T>
    using Vec = scalar::Vec<T>;
};

}
