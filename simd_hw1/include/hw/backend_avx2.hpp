#pragma once

#include <immintrin.h>

#include "common.hpp"

namespace hw {

namespace avx2 {

template <Element T>
struct Vec;

template <>
struct Vec<float> {
    using value_type = float;
    static constexpr std::size_t width = 8;

    __m256 v;

    Vec() = default;
    Vec(__m256 x) : v(x) {}
    explicit Vec(float x) : v(_mm256_set1_ps(x)) {}

    static Vec load(const float* p) { return _mm256_loadu_ps(p); }
    void store(float* p) const { _mm256_storeu_ps(p, v); }

    friend Vec operator+(Vec a, Vec b) { return _mm256_add_ps(a.v, b.v); }
    friend Vec operator-(Vec a, Vec b) { return _mm256_sub_ps(a.v, b.v); }
    friend Vec operator*(Vec a, Vec b) { return _mm256_mul_ps(a.v, b.v); }
    friend Vec operator/(Vec a, Vec b) { return _mm256_div_ps(a.v, b.v); }
    friend Vec fmadd(Vec a, Vec b, Vec c) { return _mm256_fmadd_ps(a.v, b.v, c.v); }
    friend Vec min(Vec a, Vec b) { return _mm256_min_ps(a.v, b.v); }
    friend Vec max(Vec a, Vec b) { return _mm256_max_ps(a.v, b.v); }

    friend Vec pow2(Vec k) {
        const __m256i e = _mm256_add_epi32(_mm256_cvtps_epi32(k.v), _mm256_set1_epi32(127));
        return _mm256_castsi256_ps(_mm256_slli_epi32(e, 23));
    }

    struct Acc {
        __m256 s = _mm256_setzero_ps();
        void add(Vec x) { s = _mm256_add_ps(s, x.v); }
        void merge(const Acc& o) { s = _mm256_add_ps(s, o.s); }
        float get() const {
            __m128 r = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
            r = _mm_add_ps(r, _mm_movehl_ps(r, r));
            r = _mm_add_ss(r, _mm_movehdup_ps(r));
            return _mm_cvtss_f32(r);
        }
    };
};

template <>
struct Vec<double> {
    using value_type = double;
    static constexpr std::size_t width = 4;

    __m256d v;

    Vec() = default;
    Vec(__m256d x) : v(x) {}
    explicit Vec(double x) : v(_mm256_set1_pd(x)) {}

    static Vec load(const double* p) { return _mm256_loadu_pd(p); }
    void store(double* p) const { _mm256_storeu_pd(p, v); }

    friend Vec operator+(Vec a, Vec b) { return _mm256_add_pd(a.v, b.v); }
    friend Vec operator-(Vec a, Vec b) { return _mm256_sub_pd(a.v, b.v); }
    friend Vec operator*(Vec a, Vec b) { return _mm256_mul_pd(a.v, b.v); }
    friend Vec operator/(Vec a, Vec b) { return _mm256_div_pd(a.v, b.v); }
    friend Vec fmadd(Vec a, Vec b, Vec c) { return _mm256_fmadd_pd(a.v, b.v, c.v); }
    friend Vec min(Vec a, Vec b) { return _mm256_min_pd(a.v, b.v); }
    friend Vec max(Vec a, Vec b) { return _mm256_max_pd(a.v, b.v); }

    friend Vec pow2(Vec k) {
        const __m256i e = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(k.v)),
                                           _mm256_set1_epi64x(1023));
        return _mm256_castsi256_pd(_mm256_slli_epi64(e, 52));
    }

    struct Acc {
        __m256d s = _mm256_setzero_pd();
        void add(Vec x) { s = _mm256_add_pd(s, x.v); }
        void merge(const Acc& o) { s = _mm256_add_pd(s, o.s); }
        double get() const {
            __m128d r = _mm_add_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd(s, 1));
            r = _mm_add_sd(r, _mm_unpackhi_pd(r, r));
            return _mm_cvtsd_f64(r);
        }
    };
};

template <>
struct Vec<std::int32_t> {
    using value_type = std::int32_t;
    static constexpr std::size_t width = 8;

    __m256i v;

    Vec() = default;
    Vec(__m256i x) : v(x) {}
    explicit Vec(std::int32_t x) : v(_mm256_set1_epi32(x)) {}

    static Vec load(const std::int32_t* p) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    void store(std::int32_t* p) const {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }

    friend Vec operator+(Vec a, Vec b) { return _mm256_add_epi32(a.v, b.v); }
    friend Vec operator-(Vec a, Vec b) { return _mm256_sub_epi32(a.v, b.v); }
    friend Vec operator*(Vec a, Vec b) { return _mm256_mullo_epi32(a.v, b.v); }
    friend Vec fmadd(Vec a, Vec b, Vec c) { return a * b + c; }

    friend Vec operator/(Vec a, Vec b) {
        const __m128i q_lo = div4(_mm256_castsi256_si128(a.v), _mm256_castsi256_si128(b.v));
        const __m128i q_hi =
            div4(_mm256_extracti128_si256(a.v, 1), _mm256_extracti128_si256(b.v, 1));
        return _mm256_inserti128_si256(_mm256_castsi128_si256(q_lo), q_hi, 1);
    }

    struct Acc {
        static constexpr int limit = 1 << 15;

        __m256i wrap = _mm256_setzero_si256();
        __m256i hi = _mm256_setzero_si256();
        __m256i total = _mm256_setzero_si256();
        int count = 0;

        void add(Vec x) {
            wrap = _mm256_add_epi32(wrap, x.v);
            hi = _mm256_add_epi32(hi, _mm256_srai_epi32(x.v, 16));
            if (++count == limit) flush();
        }
        void merge(const Acc& o) {
            Acc other = o;
            other.flush();
            flush();
            total = _mm256_add_epi64(total, other.total);
        }
        std::int64_t get() const {
            Acc a = *this;
            a.flush();
            __m128i r =
                _mm_add_epi64(_mm256_castsi256_si128(a.total), _mm256_extracti128_si256(a.total, 1));
            r = _mm_add_epi64(r, _mm_unpackhi_epi64(r, r));
            return _mm_cvtsi128_si64(r);
        }

    private:
        void flush() {
            const __m256i lo = _mm256_sub_epi32(wrap, _mm256_slli_epi32(hi, 16));
            const __m256i lo64 = _mm256_add_epi64(
                _mm256_cvtepu32_epi64(_mm256_castsi256_si128(lo)),
                _mm256_cvtepu32_epi64(_mm256_extracti128_si256(lo, 1)));
            const __m256i hi64 = _mm256_add_epi64(
                _mm256_cvtepi32_epi64(_mm256_castsi256_si128(hi)),
                _mm256_cvtepi32_epi64(_mm256_extracti128_si256(hi, 1)));
            total = _mm256_add_epi64(total, _mm256_add_epi64(lo64, _mm256_slli_epi64(hi64, 16)));
            wrap = _mm256_setzero_si256();
            hi = _mm256_setzero_si256();
            count = 0;
        }
    };

private:
    static __m128i div4(__m128i a, __m128i b) {
        return _mm256_cvttpd_epi32(_mm256_div_pd(_mm256_cvtepi32_pd(a), _mm256_cvtepi32_pd(b)));
    }
};

}

struct Avx2 {
    static constexpr const char* name = "avx2";
    template <Element T>
    using Vec = avx2::Vec<T>;
};

}
