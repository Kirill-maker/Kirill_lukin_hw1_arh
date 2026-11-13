#include <cstdint>
#include <limits>
#include <vector>

#include "check.hpp"
#include "util.hpp"

namespace {

constexpr std::int32_t int_min = std::numeric_limits<std::int32_t>::min();
constexpr std::int32_t int_max = std::numeric_limits<std::int32_t>::max();

template <class T>
T ref_op(char op, T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        const std::int64_t x = a, y = b;
        switch (op) {
            case '+': return static_cast<T>(x + y);
            case '-': return static_cast<T>(x - y);
            case '*': return static_cast<T>(x * y);
            default: return static_cast<T>(x / y);
        }
    } else {
        switch (op) {
            case '+': return a + b;
            case '-': return a - b;
            case '*': return a * b;
            default: return a / b;
        }
    }
}

template <class B, class T>
void run_op(char op, const T* a, const T* b, T* out, std::size_t n, const hw::Exec& ex) {
    switch (op) {
        case '+': hw::add<B>(a, b, out, n, ex); break;
        case '-': hw::sub<B>(a, b, out, n, ex); break;
        case '*': hw::mul<B>(a, b, out, n, ex); break;
        default: hw::div<B>(a, b, out, n, ex); break;
    }
}

template <class B, class T>
void run_op(char op, const T* a, T s, T* out, std::size_t n, const hw::Exec& ex) {
    switch (op) {
        case '+': hw::add<B>(a, s, out, n, ex); break;
        case '-': hw::sub<B>(a, s, out, n, ex); break;
        case '*': hw::mul<B>(a, s, out, n, ex); break;
        default: hw::div<B>(a, s, out, n, ex); break;
    }
}

template <class T>
void make_operands(std::size_t n, std::vector<T>& a, std::vector<T>& b) {
    a = t::random_operands<T>(n, 1);
    b = t::random_divisors<T>(n, 2);
    if constexpr (std::is_integral_v<T>) {
        for (std::size_t i = 0; i < n; ++i) {
            if (a[i] == int_min && b[i] == -1) b[i] = 1;
        }
    }
}

const char ops[] = {'+', '-', '*', '/'};

}

TEST(arith_array_array_matches_reference) {
    t::for_each_backend_and_type([]<class B, class T>() {
        for (std::size_t n : t::sizes) {
            std::vector<T> a, b;
            make_operands(n, a, b);
            for (char op : ops) {
                std::vector<T> expected(n);
                for (std::size_t i = 0; i < n; ++i) expected[i] = ref_op(op, a[i], b[i]);
                for (const hw::Exec& ex : t::exec_modes) {
                    t::context() = t::describe<B, T>("array op array", n, ex) + " op=" + op;
                    std::vector<T> out(n);
                    run_op<B>(op, a.data(), b.data(), out.data(), n, ex);
                    CHECK(t::same_values(out, expected));
                }
            }
        }
    });
}

TEST(arith_array_scalar_matches_reference) {
    t::for_each_backend_and_type([]<class B, class T>() {
        const T s = T(7);
        for (std::size_t n : t::sizes) {
            const std::vector<T> a = t::random_operands<T>(n, 3);
            for (char op : ops) {
                std::vector<T> expected(n);
                for (std::size_t i = 0; i < n; ++i) expected[i] = ref_op(op, a[i], s);
                for (const hw::Exec& ex : t::exec_modes) {
                    t::context() = t::describe<B, T>("array op scalar", n, ex) + " op=" + op;
                    std::vector<T> out(n);
                    run_op<B>(op, a.data(), s, out.data(), n, ex);
                    CHECK(t::same_values(out, expected));
                }
            }
        }
    });
}

TEST(arith_in_place_and_unaligned) {
    t::for_each_backend_and_type([]<class B, class T>() {
        for (std::size_t n : {std::size_t{1}, std::size_t{9}, std::size_t{67}, std::size_t{1000}}) {
            std::vector<T> a, b;
            make_operands(n + 1, a, b);
            for (char op : ops) {
                t::context() = t::describe<B, T>("in place", n, hw::seq) + " op=" + op;
                std::vector<T> expected(n);
                for (std::size_t i = 0; i < n; ++i) expected[i] = ref_op(op, a[i + 1], b[i + 1]);

                std::vector<T> work = a;
                run_op<B>(op, work.data() + 1, b.data() + 1, work.data() + 1, n, hw::seq);
                CHECK(t::same_values(std::vector<T>(work.begin() + 1, work.end()), expected));
                CHECK(t::same_value(work[0], a[0]));
            }
        }
    });
}

TEST(arith_int_overflow_wraps) {
    t::for_each_backend([]<class B>() {
        t::context() = B::name;
        const std::vector<std::int32_t> a(19, int_max), one(19, 1), big(19, 65536);
        std::vector<std::int32_t> out(19);

        hw::add<B>(a.data(), one.data(), out.data(), 19);
        for (auto x : out) CHECK(x == int_min);

        hw::sub<B>(out.data(), one.data(), out.data(), 19);
        for (auto x : out) CHECK(x == int_max);

        hw::mul<B>(big.data(), big.data(), out.data(), 19);
        for (auto x : out) CHECK(x == 0);

        hw::mul<B>(a.data(), a.data(), out.data(), 19);
        for (auto x : out) CHECK(x == 1);
    });
}

TEST(arith_int_division_is_exact) {
    t::for_each_backend([]<class B>() {
        t::context() = B::name;
        const std::int32_t edges[] = {int_min,    int_min + 1, -1073741824, -65537, -65536, -3,
                                      -2,         -1,          0,           1,      2,      3,
                                      7,          10,          65535,       65536,  65537,  46341,
                                      1073741823, 1073741824,  int_max - 1, int_max};
        std::vector<std::int32_t> a, b;
        for (auto x : edges) {
            for (auto y : edges) {
                if (y == 0 || (x == int_min && y == -1)) continue;
                a.push_back(x);
                b.push_back(y);
            }
        }
        std::vector<std::int32_t> ra, rb;
        make_operands<std::int32_t>(300000, ra, rb);
        a.insert(a.end(), ra.begin(), ra.end());
        b.insert(b.end(), rb.begin(), rb.end());

        std::vector<std::int32_t> out(a.size());
        hw::div<B>(a.data(), b.data(), out.data(), a.size());
        for (std::size_t i = 0; i < a.size(); ++i) CHECK(out[i] == a[i] / b[i]);
    });
}

TEST(arith_float_special_values) {
    t::for_each_backend_and_float([]<class B, class T>() {
        const T inf = std::numeric_limits<T>::infinity();
        const T nan = std::numeric_limits<T>::quiet_NaN();
        const T tiny = std::numeric_limits<T>::denorm_min();
        const T specials[] = {T(0), T(-0.0), T(1), T(-1), inf, -inf, nan, tiny, -tiny,
                              std::numeric_limits<T>::max(), std::numeric_limits<T>::min(),
                              T(1.5), T(-2.25), T(1e-3), T(3e10), T(0.1), T(-7), T(123456.789)};
        std::vector<T> a, b;
        for (T x : specials) {
            for (T y : specials) {
                a.push_back(x);
                b.push_back(y);
            }
        }
        const std::size_t n = a.size();
        for (char op : ops) {
            t::context() = t::describe<B, T>("special values", n, hw::seq) + " op=" + op;
            std::vector<T> out(n), expected(n);
            for (std::size_t i = 0; i < n; ++i) expected[i] = ref_op(op, a[i], b[i]);
            run_op<B>(op, a.data(), b.data(), out.data(), n, hw::seq);
            CHECK(t::same_values(out, expected));
        }
    });
}
