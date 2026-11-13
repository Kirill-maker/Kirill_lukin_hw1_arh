#pragma once

#include <array>
#include <concepts>

#include "common.hpp"

namespace hw::kernel {

template <class Op, class T>
struct BindScalar {
    Op op;
    T s;
    template <class V>
    V operator()(V x) const {
        return op(x, V(s));
    }
};

template <std::floating_point T, int N>
constexpr std::array<T, N + 1> exp_taylor() {
    std::array<T, N + 1> c{};
    long double factorial = 1;
    for (int i = 0; i <= N; ++i) {
        if (i > 0) factorial *= i;
        c[i] = static_cast<T>(1.0L / factorial);
    }
    return c;
}

template <std::floating_point T>
struct ExpConst;

template <>
struct ExpConst<float> {
    static constexpr float lo = -86.9f;
    static constexpr float hi = 89.0f;
    static constexpr float log2e = 1.44269504088896341f;
    static constexpr float ln2_hi = 0.693359375f;
    static constexpr float ln2_lo = -2.12194440e-4f;
    static constexpr float magic = 12582912.0f;
    static constexpr auto poly = exp_taylor<float, 7>();
};

template <>
struct ExpConst<double> {
    static constexpr double lo = -708.0;
    static constexpr double hi = 710.0;
    static constexpr double log2e = 1.44269504088896341;
    static constexpr double ln2_hi = 6.93145751953125e-1;
    static constexpr double ln2_lo = 1.42860682030941723212e-6;
    static constexpr double magic = 6755399441055744.0;
    static constexpr auto poly = exp_taylor<double, 13>();
};

template <class V>
V exp(V x) {
    using T = typename V::value_type;
    using C = ExpConst<T>;

    x = min(max(x, V(C::lo)), V(C::hi));
    const V k = (x * V(C::log2e) + V(C::magic)) - V(C::magic);
    const V r = (x - k * V(C::ln2_hi)) - k * V(C::ln2_lo);

    V p(C::poly.back());
    for (std::size_t i = C::poly.size() - 1; i-- > 0;) p = fmadd(p, r, V(C::poly[i]));
    return (p * pow2(k - V(T(1)))) * V(T(2));
}

template <class V>
V times_sigmoid(V x, V z) {
    using T = typename V::value_type;
    return x / (V(T(1)) + exp(V(T(0)) - z));
}

struct Silu {
    template <class V>
    V operator()(V x) const {
        return times_sigmoid(x, x);
    }
};

struct Gelu {
    static constexpr double c1 = 1.5957691216057308;
    static constexpr double c3 = 0.07135481627260025;

    template <class V>
    V operator()(V x) const {
        using T = typename V::value_type;
        const V z = x * fmadd(x * x, V(T(c3)), V(T(c1)));
        return times_sigmoid(x, z);
    }
};

}
