#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hw {

template <class T>
concept Element =
    std::same_as<T, std::int32_t> || std::same_as<T, float> || std::same_as<T, double>;

template <class T>
using sum_t = std::conditional_t<std::is_integral_v<T>, std::int64_t, T>;

template <class T>
using mean_t = std::conditional_t<std::is_integral_v<T>, double, T>;

}
