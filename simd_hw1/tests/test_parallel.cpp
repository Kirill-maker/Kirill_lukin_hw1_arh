#include <atomic>
#include <cstdint>
#include <numeric>
#include <vector>

#include "check.hpp"
#include "util.hpp"

namespace {

const std::size_t lengths[] = {0, 1, 5, 63, 64, 65, 100, 1000, 4097, 100000};
const std::size_t grains[] = {1, 7, 64, 1000};
const std::size_t aligns[] = {1, 16};

}

TEST(parallel_for_visits_every_index_once) {
    for (std::size_t n : lengths) {
        for (std::size_t grain : grains) {
            for (std::size_t align : aligns) {
                t::context() = "n=" + std::to_string(n) + " grain=" + std::to_string(grain) +
                               " align=" + std::to_string(align);
                std::vector<std::atomic<int>> hits(n);
                hw::parallel_for(
                    hw::Exec{true, grain}, n,
                    [&](std::size_t begin, std::size_t end) {
                        CHECK(begin <= end && end <= n);
                        for (std::size_t i = begin; i < end; ++i) ++hits[i];
                    },
                    align);
                bool once = true;
                for (auto& h : hits) once = once && h == 1;
                CHECK(once);
            }
        }
    }
}

TEST(parallel_reduce_matches_sequential) {
    for (std::size_t n : lengths) {
        std::vector<std::int64_t> v(n);
        std::iota(v.begin(), v.end(), std::int64_t{1});
        const std::int64_t expected = static_cast<std::int64_t>(n) * (n + 1) / 2;

        for (std::size_t grain : grains) {
            for (std::size_t align : aligns) {
                t::context() = "n=" + std::to_string(n) + " grain=" + std::to_string(grain) +
                               " align=" + std::to_string(align);
                const std::int64_t got = hw::parallel_reduce(
                    hw::Exec{true, grain}, n,
                    [&](std::size_t begin, std::size_t end) {
                        return std::accumulate(v.begin() + begin, v.begin() + end, std::int64_t{0});
                    },
                    [](std::int64_t a, std::int64_t b) { return a + b; }, align);
                CHECK(got == expected);
            }
        }
    }
}

TEST(parallel_for_respects_grain) {
    auto count_chunks = [](std::size_t n, std::size_t grain) {
        std::atomic<int> chunks{0};
        hw::parallel_for(hw::Exec{true, grain}, n, [&](std::size_t, std::size_t) { ++chunks; });
        return chunks.load();
    };

    CHECK(count_chunks(100, 1000) == 1);
    CHECK(count_chunks(1999, 1000) == 1);
#ifdef _OPENMP
    const int threads = hw::max_threads();
    t::context() = "threads=" + std::to_string(threads);
    CHECK(count_chunks(2000, 1000) == std::min(threads, 2));
    CHECK(count_chunks(1000000, 1000) == threads);
#endif

    std::atomic<int> chunks{0};
    hw::parallel_for(hw::seq, 1000000, [&](std::size_t, std::size_t) { ++chunks; });
    CHECK(chunks == 1);
}

TEST(parallel_for_nested_call_does_not_split) {
    std::atomic<int> inner_chunks{0};
    std::atomic<int> outer_chunks{0};
    hw::parallel_for(hw::Exec{true, 1}, 4, [&](std::size_t, std::size_t) {
        ++outer_chunks;
        hw::parallel_for(hw::Exec{true, 1}, 1000, [&](std::size_t, std::size_t) { ++inner_chunks; });
    });
    CHECK(inner_chunks == outer_chunks);
}
