#pragma once

#include <immintrin.h>

#include "common.hpp"

namespace hw {

namespace avx512 {

template <Element T>
struct Vec;

template <>
struct Vec<float> {
    using value_type = float;
    static constexpr std::size_t width = 16;

    __m512 v;

    Vec() = default;
    Vec(__m512 x) : v(x) {}
    explicit Vec(float x) : v(_mm512_set1_ps(x)) {}

    static Vec load(const float* p) { return _mm512_loadu_ps(p); }
    void store(float* p) const { _mm512_storeu_ps(p, v); }

    friend Vec operator+(Vec a, Vec b) { return _mm512_add_ps(a.v, b.v); }
    friend Vec operator-(Vec a, Vec b) { return _mm512_sub_ps(a.v, b.v); }
    friend Vec operator*(Vec a, Vec b) { return _mm512_mul_ps(a.v, b.v); }
    friend Vec operator/(Vec a, Vec b) { return _mm512_div_ps(a.v, b.v); }
    friend Vec fmadd(Vec a, Vec b, Vec c) { return _mm512_fmadd_ps(a.v, b.v, c.v); }
    friend Vec min(Vec a, Vec b) { return _mm512_min_ps(a.v, b.v); }
    friend Vec max(Vec a, Vec b) { return _mm512_max_ps(a.v, b.v); }

    friend Vec pow2(Vec k) {
        const __m512i e = _mm512_add_epi32(_mm512_cvtps_epi32(k.v), _mm512_set1_epi32(127));
        return _mm512_castsi512_ps(_mm512_slli_epi32(e, 23));
    }

    struct Acc {
        __m512 s = _mm512_setzero_ps();
        void add(Vec x) { s = _mm512_add_ps(s, x.v); }
        void merge(const Acc& o) { s = _mm512_add_ps(s, o.s); }
        float get() const { return _mm512_reduce_add_ps(s); }
    };
};

template <>
struct Vec<double> {
    using value_type = double;
    static constexpr std::size_t width = 8;

    __m512d v;

    Vec() = default;
    Vec(__m512d x) : v(x) {}
    explicit Vec(double x) : v(_mm512_set1_pd(x)) {}

    static Vec load(const double* p) { return _mm512_loadu_pd(p); }
    void store(double* p) const { _mm512_storeu_pd(p, v); }

    friend Vec operator+(Vec a, Vec b) { return _mm512_add_pd(a.v, b.v); }
    friend Vec operator-(Vec a, Vec b) { return _mm512_sub_pd(a.v, b.v); }
    friend Vec operator*(Vec a, Vec b) { return _mm512_mul_pd(a.v, b.v); }
    friend Vec operator/(Vec a, Vec b) { return _mm512_div_pd(a.v, b.v); }
    friend Vec fmadd(Vec a, Vec b, Vec c) { return _mm512_fmadd_pd(a.v, b.v, c.v); }
    friend Vec min(Vec a, Vec b) { return _mm512_min_pd(a.v, b.v); }
    friend Vec max(Vec a, Vec b) { return _mm512_max_pd(a.v, b.v); }

    friend Vec pow2(Vec k) {
        const __m512i e = _mm512_add_epi64(_mm512_cvtepi32_epi64(_mm512_cvtpd_epi32(k.v)),
                                           _mm512_set1_epi64(1023));
        return _mm512_castsi512_pd(_mm512_slli_epi64(e, 52));
    }

    struct Acc {
        __m512d s = _mm512_setzero_pd();
        void add(Vec x) { s = _mm512_add_pd(s, x.v); }
        void merge(const Acc& o) { s = _mm512_add_pd(s, o.s); }
        double get() const { return _mm512_reduce_add_pd(s); }
    };
};

template <>
struct Vec<std::int32_t> {
    using value_type = std::int32_t;
    static constexpr std::size_t width = 16;

    __m512i v;

    Vec() = default;
    Vec(__m512i x) : v(x) {}
    explicit Vec(std::int32_t x) : v(_mm512_set1_epi32(x)) {}

    static Vec load(const std::int32_t* p) { return _mm512_loadu_si512(p); }
    void store(std::int32_t* p) const { _mm512_storeu_si512(p, v); }

    friend Vec operator+(Vec a, Vec b) { return _mm512_add_epi32(a.v, b.v); }
    friend Vec operator-(Vec a, Vec b) { return _mm512_sub_epi32(a.v, b.v); }
    friend Vec operator*(Vec a, Vec b) { return _mm512_mullo_epi32(a.v, b.v); }
    friend Vec fmadd(Vec a, Vec b, Vec c) { return a * b + c; }

    friend Vec operator/(Vec a, Vec b) {
        const __m256i q_lo = div8(_mm512_castsi512_si256(a.v), _mm512_castsi512_si256(b.v));
        const __m256i q_hi =
            div8(_mm512_extracti64x4_epi64(a.v, 1), _mm512_extracti64x4_epi64(b.v, 1));
        return _mm512_inserti64x4(_mm512_castsi256_si512(q_lo), q_hi, 1);
    }

    struct Acc {
        static constexpr int limit = 1 << 15;

        __m512i wrap = _mm512_setzero_si512();
        __m512i hi = _mm512_setzero_si512();
        __m512i total = _mm512_setzero_si512();
        int count = 0;

        void add(Vec x) {
            wrap = _mm512_add_epi32(wrap, x.v);
            hi = _mm512_add_epi32(hi, _mm512_srai_epi32(x.v, 16));
            if (++count == limit) flush();
        }
        void merge(const Acc& o) {
            Acc other = o;
            other.flush();
            flush();
            total = _mm512_add_epi64(total, other.total);
        }
        std::int64_t get() const {
            Acc a = *this;
            a.flush();
            return _mm512_reduce_add_epi64(a.total);
        }

    private:
        void flush() {
            const __m512i lo = _mm512_sub_epi32(wrap, _mm512_slli_epi32(hi, 16));
            const __m512i lo64 = _mm512_add_epi64(
                _mm512_cvtepu32_epi64(_mm512_castsi512_si256(lo)),
                _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(lo, 1)));
            const __m512i hi64 = _mm512_add_epi64(
                _mm512_cvtepi32_epi64(_mm512_castsi512_si256(hi)),
                _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(hi, 1)));
            total = _mm512_add_epi64(total, _mm512_add_epi64(lo64, _mm512_slli_epi64(hi64, 16)));
            wrap = _mm512_setzero_si512();
            hi = _mm512_setzero_si512();
            count = 0;
        }
    };

private:
    static __m256i div8(__m256i a, __m256i b) {
        return _mm512_cvttpd_epi32(_mm512_div_pd(_mm512_cvtepi32_pd(a), _mm512_cvtepi32_pd(b)));
    }
};

}

struct Avx512 {
    static constexpr const char* name = "avx512";
    template <Element T>
    using Vec = avx512::Vec<T>;
};

}
