#pragma once

#include <concepts>
#include <cstddef>

#include "backend_scalar.hpp"
#include "common.hpp"
#include "parallel.hpp"

namespace hw {

template <class B, Element T, class Op, std::same_as<T>... Ts>
void map_range(Op op, T* out, std::size_t begin, std::size_t end, const Ts*... in) {
    using V = typename B::template Vec<T>;
    using S = Scalar::Vec<T>;

    std::size_t i = begin;
    for (; i + V::width <= end; i += V::width) op(V::load(in + i)...).store(out + i);
    for (; i < end; ++i) op(S::load(in + i)...).store(out + i);
}

template <class B, Element T, class Op, std::same_as<T>... Ts>
void map(const Exec& ex, Op op, T* out, std::size_t n, const Ts*... in) {
    parallel_for(
        ex, n,
        [=](std::size_t begin, std::size_t end) { map_range<B>(op, out, begin, end, in...); },
        16);
}

template <class B, Element T>
sum_t<T> sum_range(const T* p, std::size_t begin, std::size_t end) {
    using V = typename B::template Vec<T>;
    constexpr std::size_t W = V::width;

    typename V::Acc a0, a1, a2, a3;
    std::size_t i = begin;
    for (; i + 4 * W <= end; i += 4 * W) {
        a0.add(V::load(p + i));
        a1.add(V::load(p + i + W));
        a2.add(V::load(p + i + 2 * W));
        a3.add(V::load(p + i + 3 * W));
    }
    for (; i + W <= end; i += W) a0.add(V::load(p + i));
    a0.merge(a1);
    a2.merge(a3);
    a0.merge(a2);

    sum_t<T> s = a0.get();
    for (; i < end; ++i) s += p[i];
    return s;
}

template <class B, Element T>
sum_t<T> sum(const T* p, std::size_t n, const Exec& ex = seq) {
    return parallel_reduce(
        ex, n, [=](std::size_t begin, std::size_t end) { return sum_range<B>(p, begin, end); },
        [](sum_t<T> a, sum_t<T> b) { return a + b; }, 16);
}

}
