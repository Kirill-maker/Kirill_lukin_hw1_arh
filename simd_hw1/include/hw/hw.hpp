#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>

#include "algorithms.hpp"
#include "backend_avx2.hpp"
#include "backend_scalar.hpp"
#include "common.hpp"
#include "kernels.hpp"
#include "matmul.hpp"
#include "parallel.hpp"

#ifdef HW_AVX512
#include "backend_avx512.hpp"
#endif

namespace hw {

template <class B, Element T>
void add(const T* a, const T* b, T* out, std::size_t n, const Exec& ex = seq) {
    map<B>(ex, std::plus<>{}, out, n, a, b);
}
template <class B, Element T>
void sub(const T* a, const T* b, T* out, std::size_t n, const Exec& ex = seq) {
    map<B>(ex, std::minus<>{}, out, n, a, b);
}
template <class B, Element T>
void mul(const T* a, const T* b, T* out, std::size_t n, const Exec& ex = seq) {
    map<B>(ex, std::multiplies<>{}, out, n, a, b);
}
template <class B, Element T>
void div(const T* a, const T* b, T* out, std::size_t n, const Exec& ex = seq) {
    map<B>(ex, std::divides<>{}, out, n, a, b);
}

template <class B, Element T>
void add(const T* a, std::type_identity_t<T> s, T* out, std::size_t n, const Exec& ex = seq) {
    map<B>(ex, kernel::BindScalar{std::plus<>{}, s}, out, n, a);
}
template <class B, Element T>
void sub(const T* a, std::type_identity_t<T> s, T* out, std::size_t n, const Exec& ex = seq) {
    map<B>(ex, kernel::BindScalar{std::minus<>{}, s}, out, n, a);
}
template <class B, Element T>
void mul(const T* a, std::type_identity_t<T> s, T* out, std::size_t n, const Exec& ex = seq) {
    map<B>(ex, kernel::BindScalar{std::multiplies<>{}, s}, out, n, a);
}
template <class B, Element T>
void div(const T* a, std::type_identity_t<T> s, T* out, std::size_t n, const Exec& ex = seq) {
    map<B>(ex, kernel::BindScalar{std::divides<>{}, s}, out, n, a);
}

template <class B, Element T>
mean_t<T> mean(const T* p, std::size_t n, const Exec& ex = seq) {
    if (n == 0) return mean_t<T>{};
    return static_cast<mean_t<T>>(sum<B>(p, n, ex)) / static_cast<mean_t<T>>(n);
}

template <class B, std::floating_point T>
void silu(const T* x, T* out, std::size_t n, const Exec& ex = seq) {
    map<B>(ex, kernel::Silu{}, out, n, x);
}
template <class B, std::floating_point T>
void gelu(const T* x, T* out, std::size_t n, const Exec& ex = seq) {
    map<B>(ex, kernel::Gelu{}, out, n, x);
}

}
