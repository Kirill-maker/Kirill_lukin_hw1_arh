#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <memory>
#include <new>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "hw/hw.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

template <class T>
void keep(const T& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template <class F>
double time_per_call(F&& run) {
    constexpr double sample_seconds = 0.05;
    constexpr int samples = 7;
    constexpr double budget_seconds = 3.0;

    auto start = Clock::now();
    run();
    const double first = seconds_since(start);
    double best = first;

    start = Clock::now();
    run();
    const double once = std::max(seconds_since(start), 1e-9);
    best = std::min(best, once);

    const auto repeats = static_cast<std::size_t>(std::max(1.0, sample_seconds / once));
    const auto total_start = Clock::now();
    for (int s = 0; s < samples; ++s) {
        start = Clock::now();
        for (std::size_t r = 0; r < repeats; ++r) run();
        best = std::min(best, seconds_since(start) / static_cast<double>(repeats));
        if (s >= 1 && seconds_since(total_start) > budget_seconds) break;
    }
    return best;
}

template <class T>
class Buffer {
public:
    explicit Buffer(std::size_t n)
        : size_(n),
          data_(static_cast<T*>(::operator new(std::max<std::size_t>(n, 1) * sizeof(T),
                                               std::align_val_t{64}))) {}
    T* data() { return data_.get(); }
    std::size_t size() const { return size_; }
    T* begin() { return data(); }
    T* end() { return data() + size_; }

private:
    struct Free {
        void operator()(T* p) const { ::operator delete(p, std::align_val_t{64}); }
    };
    std::size_t size_;
    std::unique_ptr<T, Free> data_;
};

template <class T>
void fill_random(Buffer<T>& buf, std::uint32_t seed, double lo, double hi, bool nonzero = false) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(lo, hi);
    for (T& x : buf) {
        double v = dist(gen);
        if constexpr (std::is_integral_v<T>) v = std::round(v);
        if (nonzero && (gen() & 1)) v = -v;
        x = static_cast<T>(v);
    }
}

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

struct Options {
    bool quick = false;
    std::string csv;
    std::string filter;
    std::string backend;
    int threads = 0;
};

struct Row {
    std::string op, type, backend, exec;
    std::size_t n;
    double elements;
    int threads;
    double seconds;
};

class Report {
public:
    explicit Report(const Options& options) : options_(options) {
        std::printf("%-7s %-8s %10s  %-7s %-4s %12s %10s\n", "op", "type", "n", "backend", "exec",
                    "ns/elem", "vs scalar");
    }

    bool wants(const char* op) const {
        return options_.filter.empty() || std::strstr(op, options_.filter.c_str());
    }

    template <class B, class T, class F>
    void measure(const char* op, std::size_t n, double elements, F&& run) {
        if (!options_.backend.empty() && options_.backend != B::name) return;
        const int threads = hw::max_threads();
        for (const hw::Exec& ex : {hw::seq, hw::par}) {
            if (ex.parallel && threads < 2) continue;
            const double seconds = time_per_call([&] { run(ex); });
            add({op, type_name<T>(), B::name, ex.parallel ? "par" : "seq", n, elements,
                 ex.parallel ? threads : 1, seconds});
        }
    }

    void write_csv() const {
        if (options_.csv.empty()) return;
        std::FILE* f = std::fopen(options_.csv.c_str(), "w");
        if (!f) {
            std::printf("cannot open %s\n", options_.csv.c_str());
            return;
        }
        std::fprintf(f, "op,type,n,backend,exec,threads,seconds,ns_per_element\n");
        for (const Row& r : rows_) {
            std::fprintf(f, "%s,%s,%zu,%s,%s,%d,%.9g,%.6g\n", r.op.c_str(), r.type.c_str(), r.n,
                         r.backend.c_str(), r.exec.c_str(), r.threads, r.seconds,
                         r.seconds / r.elements * 1e9);
        }
        std::fclose(f);
        std::printf("\nresults saved to %s\n", options_.csv.c_str());
    }

private:
    void add(Row row) {
        const double ns = row.seconds / row.elements * 1e9;
        const auto key = std::make_tuple(row.op, row.type, row.n, row.exec);
        if (row.backend == "scalar") scalar_ns_[key] = ns;

        std::printf("%-7s %-8s %10zu  %-7s %-4s %12.4f", row.op.c_str(), row.type.c_str(), row.n,
                    row.backend.c_str(), row.exec.c_str(), ns);
        const auto base = scalar_ns_.find(key);
        if (row.backend != "scalar" && base != scalar_ns_.end()) {
            std::printf(" %9.2fx", base->second / ns);
        }
        std::printf("\n");
        std::fflush(stdout);
        rows_.push_back(std::move(row));
    }

    const Options& options_;
    std::vector<Row> rows_;
    std::map<std::tuple<std::string, std::string, std::size_t, std::string>, double> scalar_ns_;
};

template <class T>
void bench_arrays(Report& report, std::size_t n) {
    Buffer<T> a(n), b(n), x(n), out(n);
    const bool is_int = std::is_integral_v<T>;
    fill_random(a, 1, is_int ? -1e9 : -100.0, is_int ? 1e9 : 100.0);
    fill_random(b, 2, 1.0, is_int ? 1000.0 : 100.0, true);
    fill_random(x, 3, -8.0, 8.0);
    std::fill(out.begin(), out.end(), T{});
    const auto count = static_cast<double>(n);

    const T* pa = a.data();
    const T* pb = b.data();
    const T* px = x.data();
    T* po = out.data();

    for_each_backend([&]<class B>() {
        const auto run = [&](const char* op, auto&& body) {
            if (report.wants(op)) report.measure<B, T>(op, n, count, body);
        };
        run("add", [&](const hw::Exec& ex) { hw::add<B>(pa, pb, po, n, ex), keep(po); });
        run("sub", [&](const hw::Exec& ex) { hw::sub<B>(pa, pb, po, n, ex), keep(po); });
        run("mul", [&](const hw::Exec& ex) { hw::mul<B>(pa, pb, po, n, ex), keep(po); });
        run("div", [&](const hw::Exec& ex) { hw::div<B>(pa, pb, po, n, ex), keep(po); });
        run("sum", [&](const hw::Exec& ex) { keep(hw::sum<B>(pa, n, ex)); });
        run("mean", [&](const hw::Exec& ex) { keep(hw::mean<B>(pa, n, ex)); });
        if constexpr (std::is_floating_point_v<T>) {
            run("silu", [&](const hw::Exec& ex) { hw::silu<B>(px, po, n, ex), keep(po); });
            run("gelu", [&](const hw::Exec& ex) { hw::gelu<B>(px, po, n, ex), keep(po); });
        }
    });
}

template <class T>
void bench_matmul(Report& report, std::size_t n) {
    if (!report.wants("matmul")) return;
    Buffer<T> a(n * n), b(n * n), c(n * n);
    const double range = std::is_integral_v<T> ? 100.0 : 1.0;
    fill_random(a, 4, -range, range);
    fill_random(b, 5, -range, range);
    std::fill(c.begin(), c.end(), T{});
    const double count = static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);

    for_each_backend([&]<class B>() {
        report.measure<B, T>("matmul", n, count, [&](const hw::Exec& ex) {
            hw::matmul<B>(a.data(), b.data(), c.data(), n, n, n, ex);
            keep(c.data());
        });
    });
}

template <class T>
void bench_type(Report& report, const Options& options) {
    const std::vector<std::size_t> array_sizes =
        options.quick ? std::vector<std::size_t>{1u << 10, 1u << 16, 1u << 20}
                      : std::vector<std::size_t>{1u << 10, 1u << 14, 1u << 18, 1u << 22, 1u << 24};
    const std::vector<std::size_t> matrix_sizes =
        options.quick ? std::vector<std::size_t>{32, 128, 256}
                      : std::vector<std::size_t>{32, 64, 128, 256, 512, 1024};

    for (std::size_t n : array_sizes) bench_arrays<T>(report, n);
    for (std::size_t n : matrix_sizes) bench_matmul<T>(report, n);
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--quick") {
            options.quick = true;
        } else if (arg == "--csv" && has_value) {
            options.csv = argv[++i];
        } else if (arg == "--filter" && has_value) {
            options.filter = argv[++i];
        } else if (arg == "--backend" && has_value) {
            options.backend = argv[++i];
        } else if (arg == "--threads" && has_value) {
            options.threads = std::atoi(argv[++i]);
        } else {
            std::printf("usage: %s [--quick] [--csv FILE] [--filter TEXT] [--backend NAME] "
                        "[--threads N]\n",
                        argv[0]);
            std::exit(arg == "--help" ? 0 : 2);
        }
    }
    return options;
}

}

int main(int argc, char** argv) {
    const Options options = parse(argc, argv);

    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        std::printf("This CPU does not support AVX2 and FMA\n");
        return 2;
    }
#ifdef HW_AVX512
    if (!__builtin_cpu_supports("avx512f")) {
        std::printf("Built with HW_AVX512=ON, but this CPU does not support AVX-512F\n");
        return 2;
    }
#endif
#ifdef _OPENMP
    if (options.threads > 0) omp_set_num_threads(options.threads);
#endif
    std::printf("threads: %d%s\n\n", hw::max_threads(),
                hw::max_threads() < 2 ? " (parallel runs are skipped)" : "");

    Report report(options);
    bench_type<std::int32_t>(report, options);
    bench_type<float>(report, options);
    bench_type<double>(report, options);
    report.write_csv();
    return 0;
}
