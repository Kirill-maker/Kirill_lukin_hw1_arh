#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "check.hpp"
#include "util.hpp"

namespace {

long double ref_silu(long double x) { return x / (1 + std::exp(-x)); }

long double ref_gelu_tanh(long double x) {
    const long double u = std::sqrt(2 / std::acos(-1.0L)) * (x + 0.044715L * x * x * x);
    if (x >= 0) return 0.5L * x * (1 + std::tanh(u));
    return x / (1 + std::exp(-2 * u));
}

long double gelu_z(long double x) {
    return 2 * std::sqrt(2 / std::acos(-1.0L)) * (x + 0.044715L * x * x * x);
}

long double ref_gelu_exact(long double x) { return 0.5L * x * (1 + std::erf(x / std::sqrt(2.0L))); }

template <class T>
std::vector<T> grid(T lo, T hi, std::size_t points) {
    std::vector<T> x(points);
    for (std::size_t i = 0; i < points; ++i) x[i] = lo + (hi - lo) * T(i) / T(points - 1);
    return x;
}

template <class T, class Ref, class Cond>
double max_error_in_eps(const std::vector<T>& x, const std::vector<T>& y, Ref ref, Cond cond) {
    const long double tiny = 4 * std::numeric_limits<T>::min();
    long double worst = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const long double expected = ref(x[i]);
        if (std::fabs(expected) < tiny) {
            CHECK(std::fabs(y[i]) < 64 * tiny);
            continue;
        }
        worst = std::max(worst, std::fabs((y[i] - expected) / expected) / cond(x[i]));
    }
    return static_cast<double>(worst / std::numeric_limits<T>::epsilon());
}

}

TEST(exp_kernel_accuracy_and_limits) {
    t::for_each_backend_and_float([]<class B, class T>() {
        t::context() = t::describe<B, T>("exp", 0, hw::seq);
        const auto exp_op = [](auto v) { return hw::kernel::exp(v); };
        const T lo = hw::kernel::ExpConst<T>::lo;
        const T top = std::log(std::numeric_limits<T>::max());

        const std::vector<T> x = grid<T>(lo, top, 400001);
        std::vector<T> y(x.size());
        hw::map<B>(hw::seq, exp_op, y.data(), x.size(), x.data());
        long double worst = 0;
        for (std::size_t i = 0; i + 1 < x.size(); ++i) {
            const long double expected = std::exp(static_cast<long double>(x[i]));
            worst = std::max(worst, std::fabs((y[i] - expected) / expected));
        }
        CHECK(worst < 2 * std::numeric_limits<T>::epsilon());

        const T inf = std::numeric_limits<T>::infinity();
        const std::vector<T> far = {top + 1, T(1e5), inf, T(-1e5), -inf, T(0)};
        std::vector<T> out(far.size());
        hw::map<B>(hw::seq, exp_op, out.data(), far.size(), far.data());
        CHECK(out[0] == inf && out[1] == inf && out[2] == inf);
        CHECK(out[3] > 0 && out[3] < 4 * std::numeric_limits<T>::min());
        CHECK(out[4] == out[3]);
        CHECK(out[5] == 1);
    });
}

TEST(activation_simd_equals_scalar_bitwise) {
    t::for_each_backend_and_float([]<class B, class T>() {
        for (std::size_t n : t::sizes) {
            const std::vector<T> x = t::random_values<T>(n, 11, T(-40), T(40));
            std::vector<T> expected(n), out(n);
            for (const hw::Exec& ex : t::exec_modes) {
                t::context() = t::describe<B, T>("silu", n, ex);
                hw::silu<hw::Scalar>(x.data(), expected.data(), n);
                hw::silu<B>(x.data(), out.data(), n, ex);
                CHECK(t::same_values(out, expected));

                t::context() = t::describe<B, T>("gelu", n, ex);
                hw::gelu<hw::Scalar>(x.data(), expected.data(), n);
                hw::gelu<B>(x.data(), out.data(), n, ex);
                CHECK(t::same_values(out, expected));
            }
        }
    });
}

TEST(activation_close_to_libm_formula) {
    t::for_each_backend_and_float([]<class B, class T>() {
        const std::vector<T> x = grid<T>(T(-20), T(20), 400001);
        std::vector<T> y(x.size());

        t::context() = t::describe<B, T>("silu", x.size(), hw::seq);
        hw::silu<B>(x.data(), y.data(), x.size());
        const double silu_err =
            max_error_in_eps(x, y, ref_silu, [](long double) { return 1.0L; });
        CHECK(silu_err < 4.0);

        t::context() = t::describe<B, T>("gelu", x.size(), hw::seq);
        hw::gelu<B>(x.data(), y.data(), x.size());
        const double gelu_err = max_error_in_eps(x, y, ref_gelu_tanh, [](long double v) {
            return v < 0 ? 1 + std::fabs(gelu_z(v)) : 1.0L;
        });
        CHECK(gelu_err < 4.0);

        std::printf("    %-7s %-8s max error: silu %.2f eps, gelu %.2f cond*eps\n", B::name,
                    t::type_name<T>(), silu_err, gelu_err);
    });
}

TEST(gelu_tanh_approximation_close_to_exact_gelu) {
    t::for_each_backend_and_float([]<class B, class T>() {
        t::context() = t::describe<B, T>("gelu vs erf", 0, hw::seq);
        const std::vector<T> x = grid<T>(T(-10), T(10), 200001);
        std::vector<T> y(x.size());
        hw::gelu<B>(x.data(), y.data(), x.size());

        long double worst = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            worst = std::max(worst, std::fabs(y[i] - ref_gelu_exact(x[i])));
        }
        CHECK(worst < 1e-3L);
        if (std::string(B::name) == "scalar") {
            std::printf("    %-8s max |gelu_tanh - gelu_erf| = %.2e\n", t::type_name<T>(),
                        static_cast<double>(worst));
        }
    });
}

TEST(activation_special_values) {
    t::for_each_backend_and_float([]<class B, class T>() {
        const T inf = std::numeric_limits<T>::infinity();
        const T nan = std::numeric_limits<T>::quiet_NaN();
        std::vector<T> x = {T(0),   T(-0.0), T(1e3),  T(-1e3), T(1e30), T(-1e30), inf, nan,
                            T(100), T(-100), T(1e-30), T(-1e-30), T(50), T(-50),  T(1), T(-1),
                            T(0),   T(1e3),  T(-1e3), nan,     inf};
        const std::size_t n = x.size();
        std::vector<T> silu(n), gelu(n);
        hw::silu<B>(x.data(), silu.data(), n);
        hw::gelu<B>(x.data(), gelu.data(), n);

        for (const std::vector<T>* out : {&silu, &gelu}) {
            const std::vector<T>& y = *out;
            t::context() = t::describe<B, T>(out == &silu ? "silu" : "gelu", n, hw::seq);
            for (std::size_t i = 0; i < n; ++i) {
                if (std::isnan(x[i])) {
                    CHECK(std::isnan(y[i]));
                } else if (x[i] == 0) {
                    CHECK(y[i] == 0);
                } else if (x[i] >= 50) {
                    CHECK(y[i] == x[i]);
                } else if (x[i] <= -50) {
                    CHECK(std::fabs(y[i]) < T(1e-15));
                } else if (std::fabs(x[i]) < T(1e-20)) {
                    CHECK(y[i] == x[i] / 2);
                }
            }
        }
    });
}
